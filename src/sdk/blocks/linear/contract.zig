//! The linear area's kernel contract: the affine-quantized GEMVs of the verify and MTP paths. Three replace MLX bit for
//! bit (the row-axis qmv per row, the paired 3-bit coarse head against the row kernel, the wide dense verify tile against
//! `quantized_matmul`); the verify lanes (split-K, the wide multi-simdgroup tile, the NAX m16 tile) are held
//! to float64. Listed in `kernel_contract/registry.zig`.
//! Shapes: the row kernels' rows are input rows; the lanes that compile M into their source count rows in output
//! column tiles (N = rows x the lane's column granule), M fixed per registration. N is a runtime argument except for
//! the wide tile, so huge-N heads (the 248320-row lm head) run at N 4096 here: the host-built weights fit a Debug test.
//! The MLX qmv family (the row kernels) factors each group's scale and bias out of the dot
//! (s * sum(x q) + b * sum(x)), so their float64 reference does too; it differs from q * s + b per term only in NaN
//! classes (0 * Inf). Known issues: none.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const qmv_rows = @import("qmv_rows.zig");
const vp = @import("verify_projection.zig");
const vq = @import("verify_qmm.zig");
const vqn = @import("verify_qmm_nax.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const Lane = enum { rows, coarse, wide, splitk, msg, nax };

const qmv_inputs: []const [*:0]const u8 = &.{ "x", "w_q", "scales", "biases", "K_size", "N_size" };

const rows_launch = metal.Launch{ .name = "msv_qmv_rows", .inputs = qmv_inputs, .outputs = &.{"y"}, .source = qmv_rows.MSV_QMV_ROWS_SOURCE, .tg_bytes = 0, .m1 = 576, .m2 = 576 };
const coarse_launch = metal.Launch{ .name = "mtp_coarse_pairs", .inputs = &.{ "x", "w", "sc", "bi", "K_size", "N_size" }, .outputs = &.{"y"}, .source = vp.MTP_COARSE_PAIRS_SOURCE, .tg_bytes = 0, .m1 = 576, .m2 = 576 };
const wide_launch = metal.Launch{ .name = "verify_dense_wide", .inputs = &.{ "x", "w", "sc", "bi" }, .outputs = &.{"y"}, .source = vp.VERIFY_WIDE_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// The split-K tile at M 2..7 with each M's column tile (4 through M 6, 2 at 7): `partials[K_PARTS * bn * m]` floats.
/// M1/M2 compiled limits by row count 2..7 (`offline.zig`; M 2 and 7 are INCONCLUSIVE, so left at 1024).
const splitk_m1m2 = [6]u32{ 1024, 640, 512, 512, 512, 1024 };
const msg_m1m2 = [6]u32{ 1024, 640, 576, 512, 512, 1024 };
const splitk_launches = blk: {
    var out: [6]metal.Launch = undefined;
    for (2..8) |m| {
        const bn: c_int = if (m <= 6) 4 else 2;
        out[m - 2] = .{
            .name = std.fmt.comptimePrint("mlxserve_vqmm_ks_m{d}_n{d}", .{ m, bn }),
            .inputs = qmv_inputs,
            .outputs = &.{"y"},
            .source = vq.VQMM_SOURCES[vq.vqmmSourceIdx(m, bn)],
            .tg_bytes = 4 * 4 * @as(u32, @intCast(bn)) * m,
            .m1 = splitk_m1m2[m - 2],
            .m2 = splitk_m1m2[m - 2],
        };
    }
    break :blk out;
};
const msg_launches = blk: {
    var out: [6]metal.Launch = undefined;
    for (2..8) |m| out[m - 2] = .{ .name = std.fmt.comptimePrint("mlxserve_vqmm_msg_m{d}", .{m}), .inputs = qmv_inputs, .outputs = &.{"y"}, .source = vq.VQMM_MSG_SOURCES[m - 2], .tg_bytes = 0, .m1 = msg_m1m2[m - 2], .m2 = msg_m1m2[m - 2] };
    break :blk out;
};
/// `B_tile[8][16 * 32]` T and `partial[8][BMT * 32]` floats, at the largest tile (24 rows: exactly 32 KiB).
const nax_launch = metal.Launch{ .name = "mlxserve_vqmm_nax_m16", .inputs = &.{ "x", "w_q", "scales", "biases", "N_size" }, .outputs = &.{"y"}, .source = vqn.VQMM_NAX_SOURCE, .header = vqn.VQMM_NAX_HEADER, .tg_bytes = 8 * 16 * 32 * 2 + 8 * 24 * 32 * 4, .m1 = 1024, .m2 = 1024, .offline = false, .min_gen = 16 };

/// One registration: a lane at (bits, group, dtype), M baked where the lane bakes it.
const Qmm = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    lane: Lane,
    bits: u32,
    gs: usize,
    dt: mlx.mlx_dtype = .bfloat16,
    /// The baked input rows (split-K, msg, wide: 12) or the NAX tile's rows.
    m: usize = 0,
    /// The wide tile's rows per threadgroup column.
    nv: usize = 6,
    shape: Shape = undefined,
    x: []f64 = &.{},
    /// |x|, and the float64 outputs and magnitudes of the prepared call (filled on first use).
    ax: []f64 = &.{},
    want: []f64 = &.{},
    w: host.Affine,
    args: [6]mlx.mlx_array = undefined,
    n_args: usize = 0,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, lane: Lane, bits: u32, gs: usize, m: usize) Qmm {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .lane = lane, .bits = bits, .gs = gs, .m = m, .w = .{ .bits = bits, .gs = gs, .n = 0, .k = 0 } };
    }

    fn withDtype(q: Qmm, dt: mlx.mlx_dtype) Qmm {
        var r = q;
        r.dt = dt;
        return r;
    }

    pub fn deinit(self: *Qmm) void {
        const a = self.rig.a;
        self.rig.deinit();
        a.free(self.x);
        a.free(self.ax);
        a.free(self.want);
        self.w.deinit(a);
    }

    /// Output columns a row of the shape stands for (1: rows are input rows).
    fn granule(self: *const Qmm) usize {
        return switch (self.lane) {
            .rows, .coarse => 1,
            .splitk, .msg => 4,
            .wide => 8,
            .nax => 32,
        };
    }

    const Dims = struct { m: usize, n: usize, k: usize };

    fn dims(self: *const Qmm, shape: Shape) Dims {
        return if (self.granule() == 1) .{ .m = shape.rows, .n = shape.n, .k = shape.k } else .{ .m = self.m, .n = shape.rows * self.granule(), .k = shape.k };
    }

    fn colTile(self: *const Qmm) usize {
        return if (self.m <= 6) 4 else 2;
    }

    pub fn geometry(self: *const Qmm, shape: Shape) ?metal.Geometry {
        const d = self.dims(shape);
        if (d.m == 0 or d.n == 0 or d.k == 0 or d.k % self.gs != 0) return null;
        const m: c_int = @intCast(d.m);
        const n: c_int = @intCast(d.n);
        var g: metal.Geometry = undefined;
        switch (self.lane) {
            .rows => {
                // MLX qmv_fast's packs: a thread's values sit inside one group, a block is 32 threads of them.
                const vpt: usize = switch (self.bits) {
                    3, 5 => 16,
                    6 => 8,
                    else => (32 / self.bits) * 2,
                };
                if (d.m < 2 or d.m > 16 or d.n % 8 != 0 or d.k % (vpt * 32) != 0) return null;
                g = .{ .grid = .{ 32, @divExact(n, 4), m }, .tg = .{ 32, 2, 1 } };
                g.out(&.{ m, n }, self.dt);
                g.dtype("T", self.dt);
                g.int("BITS", self.bits);
                g.int("GS", self.gs);
            },
            .coarse => {
                if (d.m < 2 or d.m > 8 or d.n % 8 != 0 or d.k % 512 != 0) return null;
                g = .{ .grid = .{ 32, @divExact(n, 4), @divTrunc(m + 1, 2) }, .tg = .{ 32, 2, 1 } };
                g.out(&.{ m, 1, n }, self.dt);
                g.dtype("T", self.dt);
            },
            .wide => {
                // The joined verify's widths: the replica of qmv_wide's reduction is claimed at these K only.
                if (d.n % 8 != 0 or (d.k != 640 and d.k != 2560 and d.k != 6144)) return null;
                const nv: c_int = @intCast(self.nv);
                g = .{ .grid = .{ @divTrunc(m + nv - 1, nv) * 32, @divExact(n, 8) * 2, 1 }, .tg = .{ 32, 2, 1 } };
                g.out(&.{ 1, m, n }, self.dt);
                g.dtype("T", self.dt);
                g.int("M", d.m);
                g.int("N", d.n);
                g.int("K", d.k);
                g.int("NV", self.nv);
            },
            .splitk => {
                const bn: c_int = @intCast(self.colTile());
                if (d.n % 4 != 0 or d.k % 64 != 0) return null;
                const kp: c_int = if (n >= 4096) 2 else 4;
                g = .{ .grid = .{ 32 * kp, @divExact(n, bn), 1 }, .tg = .{ 32 * kp, 1, 1 } };
                g.out(&.{ m, n }, self.dt);
                g.dtype("T", self.dt);
                g.int("GS", self.gs);
                g.int("BITS", self.bits);
                g.int("K_PARTS", kp);
            },
            .msg => {
                if (d.n % 4 != 0 or d.k % 64 != 0) return null;
                const cols: c_int = @intCast(self.colTile() * 8);
                g = .{ .grid = .{ 256, @divTrunc(n + cols - 1, cols), 1 }, .tg = .{ 256, 1, 1 } };
                g.out(&.{ m, n }, self.dt);
                g.dtype("T", self.dt);
                g.int("GS", self.gs);
                g.int("BITS", self.bits);
                g.int("NSG", 8);
            },
            .nax => {
                if (d.n % 32 != 0 or d.k % 256 != 0) return null;
                g = .{ .grid = .{ 256, @divExact(n, 32), 1 }, .tg = .{ 256, 1, 1 } };
                g.out(&.{ m, n }, self.dt);
                g.dtype("T", self.dt);
                g.int("BITS", self.bits);
                g.int("GS", self.gs);
                g.int("KCONST", d.k);
                g.int("BMT", d.m);
            },
        }
        return g;
    }

    pub fn prepare(self: *Qmm, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        const d = self.dims(shape);
        a.free(self.x);
        a.free(self.ax);
        a.free(self.want);
        self.x = &.{};
        self.ax = &.{};
        self.want = &.{};
        self.x = try a.alloc(f64, d.m * d.k);
        self.ax = try a.alloc(f64, d.m * d.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        try self.w.fill(a, rng, d.n, d.k, self.dt, fill == .stress);
        if (fill == .stress) host.mirrored(rng, self.x, d.k) else host.uniformInto(rng, self.x, -2, 2, self.dt);
        if (fill == .nonfinite) {
            // Input rows: NaN, +Inf, -Inf, both; past four rows the rest stay finite. With fewer rows (M baked at
            // 2..4) a NaN row and NaN scales on a few columns leave the others finite.
            const nan = std.math.nan(f64);
            const inf = std.math.inf(f64);
            if (d.m >= 5) {
                const at = [_]struct { usize, usize, f64 }{ .{ 0, d.k / 3, nan }, .{ 1, d.k / 2, inf }, .{ 2, d.k - 1, -inf }, .{ 3, 0, inf }, .{ 3, 1, -inf } };
                for (at) |p| self.x[p[0] * d.k + p[1]] = p[2];
            } else {
                self.x[d.k / 3] = nan;
                self.w.scales[1 * d.k / self.gs] = nan;
                self.w.scales[2 * d.k / self.gs + 1] = nan;
            }
        }
        for (self.ax, self.x) |*o, v| o.* = @abs(v);
        try self.bind(d);
        return true;
    }

    fn bind(self: *Qmm, d: Dims) !void {
        const rig = &self.rig;
        rig.reset();
        const m: c_int = @intCast(d.m);
        const n: c_int = @intCast(d.n);
        const k: c_int = @intCast(d.k);
        const xs: []const c_int = switch (self.lane) {
            .coarse => &.{ m, 1, k },
            .wide => &.{ 1, m, k },
            else => &.{ m, k },
        };
        self.args[0] = try rig.input(self.dt, xs, self.x);
        self.args[1] = try rig.input(.uint32, &.{ n, @intCast(d.k * self.bits / 32) }, self.w.words);
        const g: c_int = @intCast(d.k / self.gs);
        self.args[2] = try rig.input(self.dt, &.{ n, g }, self.w.scales);
        self.args[3] = try rig.input(self.dt, &.{ n, g }, self.w.biases);
        self.n_args = 4;
        switch (self.lane) {
            .wide => {},
            .nax => {
                self.args[4] = try rig.input(.int32, &.{}, &.{@floatFromInt(d.n)});
                self.n_args = 5;
            },
            else => {
                self.args[4] = try rig.input(.int32, &.{}, &.{@floatFromInt(d.k)});
                self.args[5] = try rig.input(.int32, &.{}, &.{@floatFromInt(d.n)});
                self.n_args = 6;
            },
        }
    }

    pub fn run(self: *Qmm) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Qmm) usize {
        const d = self.dims(self.shape);
        return d.m * d.n;
    }

    fn factored(self: *const Qmm) bool {
        return self.lane == .rows or self.lane == .coarse;
    }

    /// Output (m, n) in float64 and its terms' magnitude: per term (q * s + b, rounded to T where the NAX tile
    /// stages it), or factored per group (MLX qmv).
    fn output(self: *const Qmm, mi: usize, w: []const f64, wm: []const f64, codes: []const u8, n: usize) [2]f64 {
        const k = self.shape.k;
        const xr = self.x[mi * k ..][0..k];
        if (self.factored()) {
            var y: f64 = 0;
            var mag: f64 = 0;
            var g: usize = 0;
            while (g < k) : (g += self.gs) {
                var xq: f64 = 0;
                var xsum: f64 = 0;
                var aq: f64 = 0;
                var ax: f64 = 0;
                for (g..g + self.gs) |i| {
                    const q: f64 = @floatFromInt(codes[i]);
                    xq += xr[i] * q;
                    xsum += xr[i];
                    aq += @abs(xr[i] * q);
                    ax += @abs(xr[i]);
                }
                const s = self.w.scales[(n * k + g) / self.gs];
                const b = self.w.biases[(n * k + g) / self.gs];
                y += s * xq + b * xsum;
                mag += @abs(s) * aq + @abs(b) * ax;
            }
            return .{ y, mag };
        }
        return .{ host.dot(xr, w)[0], host.dot(self.ax[mi * k ..][0..k], wm)[0] };
    }

    /// Every output and its magnitude, computed once a prepare.
    fn outputs(self: *Qmm, want: []f64, mag: []f64) !void {
        const n = want.len;
        if (self.want.len != 2 * n) {
            self.rig.a.free(self.want);
            self.want = &.{};
            self.want = try self.rig.a.alloc(f64, 2 * n);
            try self.compute(self.want[0..n], self.want[n..]);
        }
        @memcpy(want, self.want[0..n]);
        @memcpy(mag, self.want[n..]);
    }

    fn compute(self: *const Qmm, want: []f64, mag: []f64) !void {
        const a = self.rig.a;
        const d = self.dims(self.shape);
        const w = try a.alloc(f64, 2 * d.k);
        defer a.free(w);
        for (0..d.n) |n| {
            self.w.row(n, w[0..d.k], w[d.k..]);
            if (self.lane == .nax) for (w[0..d.k]) |*v| {
                v.* = host.rt(self.dt, v.*);
            };
            for (0..d.m) |mi| {
                const o = self.output(mi, w[0..d.k], w[d.k..], self.w.codes[n * d.k ..][0..d.k], n);
                want[mi * d.n + n] = o[0];
                mag[mi * d.n + n] = o[1];
            }
        }
    }

    pub fn reference(self: *Qmm, want: []f64, mag: []f64) !void {
        try self.outputs(want, mag);
    }

    /// fp32 sums of the integer stress terms (under 2^24) are exact in any order; fp16 and bf16 are run for the first
    /// eight columns of every row: a lower bound on the miss.
    pub fn simulate(self: *Qmm, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const a = self.rig.a;
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        try self.outputs(out, mag);
        if (acc == .f32) {
            for (mag) |m| if (m >= 0x1p24) return error.StressPastExact;
            return;
        }
        const d = self.dims(self.shape);
        const t = try a.alloc(f64, 3 * d.k);
        defer a.free(t);
        for (0..@min(8, d.n)) |n| {
            self.w.row(n, t[0..d.k], t[d.k .. 2 * d.k]);
            for (0..d.m) |mi| {
                for (t[2 * d.k ..], t[0..d.k], 0..) |*o, w, i| o.* = self.x[mi * d.k + i] * w;
                out[mi * d.n + n] = spec.sumIn(acc, order, t[2 * d.k ..]);
            }
        }
    }

    /// The MLX kernel a replacement must equal.
    pub fn replaced(self: *Qmm) ![]const u8 {
        const s = self.rig.stream();
        const x = self.args;
        const d = self.dims(self.shape);
        const bits: c_int = @intCast(self.bits);
        const gs: c_int = @intCast(self.gs);
        switch (self.lane) {
            .rows => {
                var g = ops.Graph.init(self.rig.a, s);
                defer g.deinit();
                var rows: [16]mlx.mlx_array = undefined;
                for (0..d.m) |r| rows[r] = try g.qmm(try g.sliceAxis(x[0], 0, @intCast(r), @intCast(r + 1)), x[1], x[2], x[3], gs, bits);
                return self.rig.bytesOf(rows[0..d.m]);
            },
            .coarse => {
                const y = (try qmv_rows.msvQmvRows(s, x[0], x[1], x[2], x[3], 3, 64)) orelse return error.Declined;
                defer _ = mlx.mlx_array_free(y);
                return self.rig.bytesOf(&.{y});
            },
            .wide => {
                // Each stream's rows alone (M 6: MLX's qmv_wide on every M5), as the joined verify must match them.
                var g = ops.Graph.init(self.rig.a, s);
                defer g.deinit();
                var parts: [4]mlx.mlx_array = undefined;
                var r: usize = 0;
                while (r < d.m) : (r += self.nv) parts[r / self.nv] = try g.qmm(try g.sliceAxis(x[0], 1, @intCast(r), @intCast(@min(d.m, r + self.nv))), x[1], x[2], x[3], gs, bits);
                return self.rig.bytesOf(&.{try g.concat(parts[0 .. (d.m + self.nv - 1) / self.nv], 1)});
            },
            else => unreachable,
        }
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *Qmm) ![]const u8 {
        const s = self.rig.stream();
        const x = self.args;
        const d = self.dims(self.shape);
        const m: c_int = @intCast(d.m);
        const n: c_int = @intCast(d.n);
        const k: c_int = @intCast(d.k);
        const xsh: []const c_int = &.{ m, k };
        const y = switch (self.lane) {
            .rows => try qmv_rows.msvQmvRows(s, x[0], x[1], x[2], x[3], self.bits, @intCast(self.gs)),
            .coarse => unreachable, // the wrapper takes the lm head's shape only: its own test below
            .wide => blk: {
                const outer = vp.verify_group_rows;
                vp.verify_group_rows = 2;
                defer vp.verify_group_rows = outer;
                break :blk try vp.verifyWideProjection(s, x[0], x[1], x[2], x[3], @intCast(self.nv));
            },
            .splitk => blk: {
                const outer = vq.vqmm_mixed_plain_test_force;
                vq.vqmm_mixed_plain_test_force = true;
                defer vq.vqmm_mixed_plain_test_force = outer;
                break :blk try vq.verifyQmm(s, x[0], x[1], x[2], x[3], self.bits, @intCast(self.gs));
            },
            .msg => try vq.runVerifyQmmMsg(s, x[0], x[1], x[2], x[3], self.bits, @intCast(self.gs), m, k, n, self.dt, xsh),
            .nax => try vqn.runVerifyQmmNax(s, x[0], x[1], x[2], x[3], self.bits, @intCast(self.gs), m, m, k, n, self.dt, xsh),
        } orelse return error.Declined;
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

fn depthK(shape: Shape) usize {
    return shape.k + 4; // a row's sum, behind the dequant (q * s + b) and the product
}

/// A registration's spec: `tiles` and `served` in the lane's own row units.
fn qmmSpec(comptime name: []const u8, q: Qmm, served: []const Shape, replaces: ?[]const u8) spec.Spec {
    const vpt: usize = switch (q.bits) {
        3, 5 => 16,
        6 => 8,
        else => (32 / q.bits) * 2,
    };
    const tiles: [3]spec.Tile = switch (q.lane) {
        // n: two simdgroups of four output rows; k: one 32-thread block of packs.
        .rows => .{ .{}, .{ .size = 8, .step = 8, .min = 8 }, .{ .size = vpt * 32, .step = vpt * 32, .min = vpt * 32 } },
        .coarse => .{ .{ .size = 2 }, .{ .size = 8, .step = 8, .min = 8 }, .{ .size = 512, .step = 512, .min = 512 } },
        // Column tiles in rows; k: 32 lanes of 8 values.
        .splitk, .msg => .{ .{ .size = 8 }, .{}, .{ .size = 256, .step = 64, .min = 64 } },
        .wide => .{ .{ .size = 2 }, .{}, .{ .size = 512, .step = 64, .min = 64 } },
        .nax => .{ .{ .size = 2 }, .{}, .{ .size = 256, .step = 256, .min = 256 } },
    };
    return .{
        .name = name,
        .accum = .f32,
        .depth = depthK,
        .out = host.elemOf(q.dt),
        .replaces = replaces,
        .tiles = tiles,
        .served = served,
    };
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    // The row-axis qmv (MTP head rows, lm head rows): every packed width at g64, g32 and g128 at 4-bit, f16.
    const rows_served: []const Shape = &.{ .{ .rows = 2, .n = 512, .k = 2560 }, .{ .rows = 16, .n = 2560, .k = 2560 }, .{ .rows = 4, .n = 640, .k = 2560 } };
    const per_row = "quantized_matmul (MLX qmv) per row";
    inline for (.{ 3, 4, 5, 6, 8 }) |bits| {
        const q = Qmm.init(a, &rows_launch, .rows, bits, 64, 0);
        try set.add(Qmm, q, qmmSpec(std.fmt.comptimePrint("qmv_rows bf16 ({d}-bit g64)", .{bits}), q, rows_served, per_row), null);
    }
    inline for (.{ 32, 128 }) |gs| {
        const q = Qmm.init(a, &rows_launch, .rows, 4, gs, 0);
        try set.add(Qmm, q, qmmSpec(std.fmt.comptimePrint("qmv_rows bf16 (4-bit g{d})", .{gs}), q, rows_served, per_row), null);
    }
    {
        const q = Qmm.init(a, &rows_launch, .rows, 4, 64, 0).withDtype(.float16);
        try set.add(Qmm, q, qmmSpec("qmv_rows f16 (4-bit g64)", q, rows_served, per_row), null);
    }
    {
        const q = Qmm.init(a, &coarse_launch, .coarse, 3, 64, 0);
        try set.add(Qmm, q, qmmSpec("mtp_coarse_pairs bf16 (3-bit g64)", q, &.{ .{ .rows = 8, .n = 1024, .k = 2560 }, .{ .rows = 3, .n = 1024, .k = 2560 } }, "msv_qmv_rows"), null);
    }
    {
        // The joined verify group of 12 rows (two six-row streams): 8-bit g64 at the gdn, lm head and attention widths.
        const q = Qmm.init(a, &wide_launch, .wide, 8, 64, 12);
        var sp = qmmSpec("verify_wide bf16 (8-bit g64, M 12, NV 6)", q, &.{ .{ .rows = 768, .n = 12, .k = 2560 }, .{ .rows = 320, .n = 12, .k = 6144 }, .{ .rows = 1280, .n = 12, .k = 2560 } }, "quantized_matmul per six-row stream (MLX qmv_wide)");
        sp.nax = true; // the engine runs it only where verifySharedHardware(), which needs NAX
        try set.add(Qmm, q, sp, null);
    }
    // Split-K at M 2..7, 4-bit g64 (N 640 and 2560 on the 4B trunk, 12288 past the K_PARTS switch).
    const ks_served: []const Shape = &.{ .{ .rows = 160, .n = 0, .k = 2560 }, .{ .rows = 128, .n = 0, .k = 640 }, .{ .rows = 1024, .n = 0, .k = 640 } };
    inline for (2..8) |m| {
        const q = Qmm.init(a, &splitk_launches[m - 2], .splitk, 4, 64, m);
        try set.add(Qmm, q, qmmSpec(std.fmt.comptimePrint("vqmm split-K bf16 (M {d}, 4-bit g64)", .{m}), q, ks_served, null), null);
    }
    inline for (.{ .{ 4, 32 }, .{ 4, 128 }, .{ 5, 64 }, .{ 6, 128 }, .{ 8, 64 } }) |bg| {
        const q = Qmm.init(a, &splitk_launches[4 - 2], .splitk, bg[0], bg[1], 4);
        try set.add(Qmm, q, qmmSpec(std.fmt.comptimePrint("vqmm split-K bf16 (M 4, {d}-bit g{d})", .{ bg[0], bg[1] }), q, &.{ .{ .rows = 160, .n = 0, .k = 2560 }, .{ .rows = 640, .n = 0, .k = 640 } }, null), null);
    }
    {
        const q = Qmm.init(a, &splitk_launches[5 - 2], .splitk, 4, 64, 5).withDtype(.float16);
        try set.add(Qmm, q, qmmSpec("vqmm split-K f16 (M 5, 4-bit g64)", q, ks_served, null), null);
    }
    // The multi-simdgroup tile of the lm head (served at N >= 100000; N 2048 here).
    inline for (2..8) |m| {
        const q = Qmm.init(a, &msg_launches[m - 2], .msg, 4, 64, m);
        try set.add(Qmm, q, qmmSpec(std.fmt.comptimePrint("vqmm msg bf16 (M {d}, 4-bit g64)", .{m}), q, &.{.{ .rows = 512, .n = 0, .k = 2560 }}, null), null);
    }
    {
        const q = Qmm.init(a, &msg_launches[6 - 2], .msg, 6, 64, 6);
        try set.add(Qmm, q, qmmSpec("vqmm msg bf16 (M 6, 6-bit g64)", q, &.{.{ .rows = 512, .n = 0, .k = 2560 }}, null), null);
    }
    // The NAX m16 tile (and the M4 shader tile's 8 and 24 rows): 4-bit at every group, the mixed widths at g64.
    const nax_served: []const Shape = &.{ .{ .rows = 16, .n = 0, .k = 2560 }, .{ .rows = 8, .n = 0, .k = 5120 } };
    inline for (.{ .{ 4, 64, 16 }, .{ 4, 32, 16 }, .{ 4, 128, 16 }, .{ 5, 64, 16 }, .{ 6, 64, 16 }, .{ 8, 64, 16 }, .{ 4, 64, 8 }, .{ 4, 64, 24 } }) |c| {
        const q = Qmm.init(a, &nax_launch, .nax, c[0], c[1], c[2]);
        try set.add(Qmm, q, qmmSpec(std.fmt.comptimePrint("vqmm nax bf16 ({d} rows, {d}-bit g{d})", .{ c[2], c[0], c[1] }), q, nax_served, null), null);
    }
}

// ── tests ──

test "kernel contract linear: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const t: *Qmm = @ptrCast(@alignCast(k.ctx));
        if (t.lane == .coarse) continue;
        if ((t.lane == .nax or t.lane == .wide) and !vp.verifySharedCapable()) continue;
        try std.testing.expect(try t.prepare(k.spec.served[0], .random, 7));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        const theirs = try t.wrapper();
        if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{k.spec.name});
        try std.testing.expectEqualSlices(u8, theirs, ours);
    }
}

test "kernel contract linear: the coarse pairs launch is the wrapper's at the lm head's shape" {
    if (mlx.noGpuBackend() or !vp.verifySharedCapable()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const outer = vp.verify_group_rows;
    vp.verify_group_rows = 2; // a two-stream group: the dual-die M5 takes the joined kernels only then
    defer vp.verify_group_rows = outer;
    var q = Qmm.init(a, &coarse_launch, .coarse, 3, 64, 0);
    defer q.deinit();
    const s = q.rig.stream();
    // The wrapper's only shape (K 2560, N 248320), its operands drawn on the device: too large to build on the host.
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    var key = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_key(&key, 11));
    _ = try g.adopt(key);
    var w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_bits(&w, &[_]c_int{ 248320, 240 }, 2, 4, key, s));
    const wq = try g.adopt(w);
    const lo = try g.scalar(0.001, .float32);
    const hi = try g.scalar(0.02, .float32);
    var su = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_uniform(&su, lo, hi, &[_]c_int{ 248320, 40 }, 2, .float32, key, s));
    const sc = try g.astype(try g.adopt(su), .bfloat16);
    const bi = try g.astype(try g.mulScalar(sc, -4), .bfloat16);
    var xu = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_uniform(&xu, try g.scalar(-2, .float32), try g.scalar(2, .float32), &[_]c_int{ 4, 1, 2560 }, 3, .float32, key, s));
    const x = try g.astype(try g.adopt(xu), .bfloat16);
    try ops.evalMany(&.{ x, wq, sc, bi });
    const shape = Shape{ .rows = 4, .n = 248320, .k = 2560 };
    q.shape = shape;
    const kn = [2]mlx.mlx_array{ mlx.mlx_array_new_int(2560), mlx.mlx_array_new_int(248320) };
    defer for (kn) |v| {
        _ = mlx.mlx_array_free(v);
    };
    try q.rig.launch(q.launch, q.geometry(shape).?, &.{ x, wq, sc, bi, kn[0], kn[1] });
    const ours = try a.dupe(u8, q.rig.outputBytes());
    defer a.free(ours);
    const y = (try vp.mtpCoarsePairs(s, x, wq, sc, bi, 3, 64)) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(y);
    try std.testing.expectEqualSlices(u8, try q.rig.bytesOf(&.{y}), ours);
}
