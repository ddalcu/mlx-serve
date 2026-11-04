//! The quant area's kernel contract: Prism's block Hadamard rotation (both sides, every shipped block), its fused
//! add + RMS norm + rotation, and the row-scaled e4m3 GEMV of Sushi / ModelOpt packs. Listed in
//! `kernel_contract/registry.zig`. e4m3's NaN codes (0x7f, 0xff) decode as +-480 in the GEMV; packs never store them,
//! so they are not drawn. Served widths are the engine's (rows, K); the GEMV's N (MiMo's QKV rows, a runtime argument)
//! runs at 512, its tail covered by the N edges, so the float64 reference fits a Debug build. Known issues: rot_bf16_m1 (gate 5: the bf16 rotation-only norm
//! rotation at hidden 4096 compiles for 96 threads on M1/M2, launched at 128).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../testing/kernel_contract/root.zig");
const rht = @import("hadamard/rht.zig");
const sushi = @import("exl3/sushi_pack.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const rotate_launch = metal.Launch{ .name = "msv_hadamard_block", .inputs = &.{ "x", "signs" }, .outputs = &.{"y"}, .source = rht.KERNEL_SOURCE, .tg_bytes = 4096 * 4, .m1 = 512, .m2 = 512 };
const norm_rotate_launch = metal.Launch{ .name = "msv_norm_rotate", .inputs = &.{ "a", "b", "w", "eps", "signs" }, .outputs = &.{ "sum", "normed", "rot" }, .source = rht.NORM_ROTATE_SOURCE, .tg_bytes = 1024 * 4 + 4 * 4, .m1 = 1024, .m2 = 1024 };
/// The bf16 rotation-only instantiations: M1/M2 compile hidden 4096 (NB 4) for 96 threads, 5120 for 768
/// (`offline.zig`); the launch is 128.
const norm_rotate_rot_bf16_launch = blk: {
    var l = norm_rotate_launch;
    l.m1 = 96;
    l.m2 = 96;
    l.caps = &.{ .{ .k = 1024, .m1 = 1024, .m2 = 1024 }, .{ .k = 2048, .m1 = 1024, .m2 = 1024 }, .{ .k = 5120, .m1 = 768, .m2 = 768 } };
    break :blk l;
};
const rot_bf16_m1: spec.Known = .{ .gate = .limits, .finding = "M1/M2 compile the bf16 rotation-only instantiation at hidden 4096 (NB 4) for 96 threads a group and normRotate launches 128 without probing: on M1/M2 that dispatch exceeds the pipeline's limit" };
const fp8_launch = metal.Launch{ .name = "msv_fp8_gemv", .inputs = &.{ "x", "w", "scales" }, .outputs = &.{"y"}, .source = sushi.GEMV_SOURCE, .header = sushi.GEMV_HEADER, .tg_bytes = 0, .m1 = 512, .m2 = 512 };

fn gamma(d: f64) f64 {
    return spec.gamma(d, 0x1p-24);
}

/// Non-finite plants at known rows: NaN, +Inf, -Inf, and one row holding both infinities (rows past 4 stay finite).
fn plant(v: []f64, rows: usize, n: usize) void {
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    const at = [_]struct { usize, usize, f64 }{ .{ 0, n / 3, nan }, .{ 1, n / 2, inf }, .{ 2, n - 1, -inf }, .{ 3, 0, inf }, .{ 3, 1, -inf } };
    for (at) |p| if (p[0] < rows) {
        v[p[0] * n + p[1]] = p[2];
    };
}

// ── the block rotation ──

/// `H(signs * x)` (or `signs * H(x)`) per `block`-wide stripe of x [rows, k], scaled 1/sqrt(block).
const Rotate = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    block: usize,
    inv: bool,
    shape: Shape = undefined,
    x: []f64 = &.{},
    signs: []f64 = &.{},
    args: [2]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, dt: mlx.mlx_dtype, block: usize, inv: bool) Rotate {
        return .{ .rig = metal.Rig.init(a), .launch = &rotate_launch, .dt = dt, .block = block, .inv = inv };
    }

    pub fn deinit(self: *Rotate) void {
        const a = self.rig.a;
        self.rig.deinit();
        a.free(self.x);
        a.free(self.signs);
    }

    pub fn geometry(self: *const Rotate, shape: Shape) ?metal.Geometry {
        if (shape.k == 0 or shape.k % self.block != 0) return null;
        const groups: c_int = @intCast(shape.rows * shape.k / self.block);
        var g = metal.Geometry{ .grid = .{ groups * 128, 1, 1 }, .tg = .{ 128, 1, 1 } };
        g.out(&.{ @intCast(shape.rows), @intCast(shape.k) }, self.dt);
        g.dtype("T", self.dt);
        g.int("J", self.block / 32);
        g.flag("INV", self.inv);
        return g;
    }

    pub fn prepare(self: *Rotate, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        a.free(self.x);
        a.free(self.signs);
        self.x = &.{};
        self.signs = &.{};
        self.x = try a.alloc(f64, shape.rows * shape.k);
        self.signs = try a.alloc(f64, shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.signs) |*s| s.* = if (rng.boolean()) 1 else -1;
        switch (fill) {
            .random => host.uniformInto(rng, self.x, -3, 3, self.dt),
            .nonfinite => {
                host.uniformInto(rng, self.x, -3, 3, self.dt);
                plant(self.x, shape.rows, shape.k);
            },
            .stress => {
                // The rotated values v mirror within each block, so each block's first output (all + signs) sums
                // halves past fp16's range to a small integer; x carries the signs on the input side.
                host.mirroredWide(rng, self.x, self.block);
                if (!self.inv) for (self.x, 0..) |*v, i| {
                    v.* *= self.signs[i % shape.k];
                };
            },
        }
        self.rig.reset();
        self.args[0] = try self.rig.input(self.dt, &.{ @intCast(shape.rows), @intCast(shape.k) }, self.x);
        self.args[1] = try self.rig.input(.float32, &.{@intCast(shape.k)}, self.signs);
        return true;
    }

    pub fn run(self: *Rotate) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Rotate) usize {
        return self.shape.rows * self.shape.k;
    }

    /// Block `b`'s rotated inputs (the summed terms of its first output) into `v`.
    fn terms(self: *const Rotate, b: usize, v: []f64) void {
        const at = b * self.block;
        for (v, 0..) |*o, j| {
            const x = self.x[at + j];
            o.* = if (self.inv) x else x * self.signs[(at + j) % self.shape.k];
        }
    }

    fn scale(self: *const Rotate) f64 {
        return 1 / @sqrt(@as(f64, @floatFromInt(self.block)));
    }

    pub fn reference(self: *const Rotate, want: []f64, mag: []f64) !void {
        const k = self.shape.k;
        for (0..self.count() / self.block) |b| {
            const v = want[b * self.block ..][0..self.block];
            self.terms(b, v);
            const m = host.sumAbs(v) * self.scale();
            host.fwht(v);
            for (v, 0..) |*o, i| {
                o.* *= self.scale();
                if (self.inv) o.* *= self.signs[(b * self.block + i) % k];
            }
            @memset(mag[b * self.block ..][0..self.block], m);
        }
    }

    /// fp32 butterflies of integer terms under 2^24 are exact in any order (the float64 values); fp16 and bf16 are
    /// run for each block's first output, whose terms are the rotated inputs themselves: a lower bound on the miss.
    pub fn simulate(self: *const Rotate, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const a = self.rig.a;
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        try self.reference(out, mag);
        if (acc == .f32) {
            for (mag) |m| if (m / self.scale() >= 0x1p24) return error.StressPastExact;
            return;
        }
        const v = try a.alloc(f64, self.block);
        defer a.free(v);
        for (0..self.count() / self.block) |b| {
            self.terms(b, v);
            const first = spec.sumIn(acc, order, v) * self.scale();
            out[b * self.block] = if (self.inv) first * self.signs[(b * self.block) % self.shape.k] else first;
        }
    }

    fn wrapper(self: *Rotate) ![]const u8 {
        const y = try rht.transform(self.args[0], self.args[1], @intCast(self.block), self.inv, self.rig.stream());
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

fn rotateDepth(_: Shape) usize {
    return 13; // log2(4096) butterfly levels and the scale
}

fn rotateSpec(comptime name: []const u8, dt: mlx.mlx_dtype, block: usize) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = rotateDepth,
        // rsqrt(block) (fast rsqrt at 512 and 2048, exact at 1024 and 4096) and the product with it.
        .ops = 3,
        .out = host.elemOf(dt),
        .tiles = .{ .{}, .{}, .{ .size = block, .step = block, .min = block } },
        .served = switch (block) {
            512, 1024 => &.{ .{ .rows = 1, .n = 1, .k = 4096 }, .{ .rows = 8, .n = 1, .k = 4096 }, .{ .rows = 64, .n = 1, .k = 5120 } },
            else => &.{ .{ .rows = 1, .n = 1, .k = 4096 }, .{ .rows = 8, .n = 1, .k = 4096 }, .{ .rows = 64, .n = 1, .k = 12288 } },
        },
    };
}

// ── add + RMS norm + rotation ──

const NormRotate = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    add: bool,
    write_normed: bool,
    shape: Shape = undefined,
    eps: f64 = 1e-6,
    a_v: []f64 = &.{},
    b_v: []f64 = &.{},
    w: []f64 = &.{},
    signs: []f64 = &.{},
    args: [5]mlx.mlx_array = undefined,
    msg: [256]u8 = undefined,

    const BLOCK = 1024;

    fn init(a: std.mem.Allocator, dt: mlx.mlx_dtype, add: bool, write_normed: bool) NormRotate {
        return .{ .rig = metal.Rig.init(a), .launch = &norm_rotate_launch, .dt = dt, .add = add, .write_normed = write_normed };
    }

    pub fn deinit(self: *NormRotate) void {
        const a = self.rig.a;
        self.rig.deinit();
        for ([_][]f64{ self.a_v, self.b_v, self.w, self.signs }) |v| a.free(v);
    }

    pub fn geometry(self: *const NormRotate, shape: Shape) ?metal.Geometry {
        if (shape.k == 0 or shape.k % BLOCK != 0 or shape.rows == 0 or shape.rows > rht.NORM_ROTATE_MAX_ROWS) return null;
        const nb: c_int = @intCast(shape.k / BLOCK);
        var g = metal.Geometry{ .grid = .{ nb * 128, @intCast(shape.rows), 1 }, .tg = .{ 128, 1, 1 } };
        for (0..3) |_| g.out(&.{ @intCast(shape.rows), @intCast(shape.k) }, self.dt);
        g.dtype("T", self.dt);
        g.int("NB", nb);
        g.flag("HAS_ADD", self.add);
        g.flag("WRITE_NORMED", self.write_normed);
        g.unread = @as(u8, @intFromBool(!self.add)) | @as(u8, @intFromBool(!self.write_normed)) << 1;
        return g;
    }

    pub fn prepare(self: *NormRotate, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        const n = shape.rows * shape.k;
        for ([_]*[]f64{ &self.a_v, &self.b_v, &self.w, &self.signs }, [_]usize{ n, n, shape.k, shape.k }) |p, len| {
            a.free(p.*);
            p.* = &.{};
            p.* = try a.alloc(f64, len);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.signs) |*s| s.* = if (rng.boolean()) 1 else -1;
        host.uniformInto(rng, self.a_v, -3, 3, self.dt);
        host.uniformInto(rng, self.b_v, -3, 3, self.dt);
        host.uniformInto(rng, self.w, 0.5, 1.5, self.dt);
        self.eps = 1e-6;
        switch (fill) {
            .random => {},
            .nonfinite => plant(self.a_v, shape.rows, shape.k),
            .stress => {
                // Integers from {1, 5, 7} whose squares sum to exactly 16 a row: the sum of squares passes fp16's range
                // and lands off bf16's grid, while 1 / rms is 0.25 exactly (eps 0), so every later stage is exact.
                self.eps = 0;
                @memset(self.b_v, 0);
                @memset(self.w, 1);
                for (0..shape.rows) |r| stressRow(rng, self.a_v[r * shape.k ..][0..shape.k]);
            },
        }
        try self.bind();
        return true;
    }

    fn stressRow(rng: std.Random, row: []f64) void {
        const pick = [3]f64{ 1, 5, 7 };
        var total: f64 = 0;
        for (row) |*v| {
            v.* = pick[rng.uintLessThan(usize, 3)];
            total += v.* * v.*;
        }
        // Every square is 1 mod 24, so the gap to 16 per element is a multiple of 24: step entries up or down by one
        // rung (1 <-> 5 <-> 7 moves a square by 24).
        var gap = 16 * @as(f64, @floatFromInt(row.len)) - total;
        var i: usize = 0;
        while (gap != 0) : (i = (i + 1) % row.len) {
            const v = row[i];
            if (gap > 0 and v < 7) {
                row[i] = if (v == 1) 5 else 7;
                gap -= 24;
            } else if (gap < 0 and v > 1) {
                row[i] = if (v == 7) 5 else 1;
                gap += 24;
            }
        }
        for (row) |*v| if (rng.boolean()) {
            v.* = -v.*;
        };
    }

    fn bind(self: *NormRotate) !void {
        const rig = &self.rig;
        rig.reset();
        const rows: c_int = @intCast(self.shape.rows);
        const k: c_int = @intCast(self.shape.k);
        self.args[0] = try rig.input(self.dt, &.{ rows, k }, self.a_v);
        self.args[1] = if (self.add) try rig.input(self.dt, &.{ rows, k }, self.b_v) else self.args[0];
        self.args[2] = try rig.input(self.dt, &.{k}, self.w);
        self.args[3] = try rig.scalar(@floatCast(self.eps));
        self.args[4] = try rig.input(.float32, &.{k}, self.signs);
    }

    pub fn run(self: *NormRotate) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    fn outs(self: *const NormRotate) usize {
        return @as(usize, @intFromBool(self.add)) + @intFromBool(self.write_normed) + 1;
    }

    pub fn count(self: *const NormRotate) usize {
        return self.outs() * self.shape.rows * self.shape.k;
    }

    fn rt(self: *const NormRotate, v: f64) f64 {
        return host.rt(self.dt, v);
    }

    /// Every output with the sum of squares in `acc` (null: float64); `flip` (when given, two rows x k long) each
    /// normed value's spread over 1 / rms within its fp32 error, then what that spread moves in each rotation output.
    fn outputsIn(self: *const NormRotate, want: []f64, mag: []f64, flip: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const a = self.rig.a;
        const k = self.shape.k;
        const n = self.shape.rows * k;
        const e = try a.alloc(f64, k);
        defer a.free(e);
        const sq = try a.alloc(f64, k);
        defer a.free(sq);
        const hk: f64 = @floatFromInt(k);
        const delta = gamma(hk + 4);
        var slot: usize = 0;
        const sum_at = slot;
        if (self.add) slot += 1;
        const normed_at = slot;
        if (self.write_normed) slot += 1;
        const rot_at = slot;
        for (0..self.shape.rows) |r| {
            for (e, sq, 0..) |*o, *q, i| {
                const at = r * k + i;
                o.* = if (self.add) self.rt(self.a_v[at] + self.b_v[at]) else self.a_v[at];
                q.* = o.* * o.*;
            }
            const ss = if (acc) |ac| spec.sumIn(ac, order, sq) else host.sum(sq);
            const inv = 1 / @sqrt(ss / hk + self.eps);
            const rot = want[rot_at * n + r * k ..][0..k];
            const spread = if (flip) |f| f[r * k ..][0..k] else rot; // unused without `flip`
            const moved_row = if (flip) |f| f[n + r * k ..][0..k] else rot;
            for (e, 0..) |x, i| {
                const at = r * k + i;
                const v = x * inv;
                const nv = self.rt(self.w[i] * self.rt(v));
                if (self.add) {
                    want[sum_at * n + at] = x;
                    mag[sum_at * n + at] = @abs(self.a_v[at]) + @abs(self.b_v[at]);
                }
                if (self.write_normed) {
                    want[normed_at * n + at] = nv;
                    mag[normed_at * n + at] = @abs(nv);
                }
                if (flip != null) {
                    const lo = self.rt(self.w[i] * self.rt(v * (1 - delta)));
                    const hi = self.rt(self.w[i] * self.rt(v * (1 + delta)));
                    spread[i] = @max(@abs(lo - nv), @abs(hi - nv));
                }
                rot[i] = nv * self.signs[i];
            }
            var b: usize = 0;
            while (b < k) : (b += BLOCK) {
                const blk = rot[b..][0..BLOCK];
                const m = host.sumAbs(blk) / 32;
                host.fwht(blk);
                for (blk) |*o| o.* /= 32;
                @memset(mag[rot_at * n + r * k + b ..][0..BLOCK], m);
                if (flip != null) {
                    // A normed value one rounding off moves every output of its block by it / 32.
                    @memset(moved_row[b..][0..BLOCK], host.sum(spread[b..][0..BLOCK]) / 32);
                }
            }
        }
    }

    pub fn reference(self: *const NormRotate, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, null, .sequential);
    }

    pub fn simulate(self: *const NormRotate, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const mag = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(mag);
        try self.outputsIn(out, mag, null, acc, order);
    }

    /// Gates 1 and 3: the bound per output, the rotation's widened by the normed values' possible roundings.
    pub fn check(self: *NormRotate, at: spec.Gate) !?[]const u8 {
        const a = self.rig.a;
        const n = self.count();
        const per = self.shape.rows * self.shape.k;
        const buf = try a.alloc(f64, 3 * n + 2 * per);
        defer a.free(buf);
        const want = buf[0..n];
        const mag = buf[n .. 2 * n];
        const got = buf[2 * n .. 3 * n];
        const flip = buf[3 * n ..];
        try self.outputsIn(want, mag, flip, null, .sequential);
        const s = registered(self.dt);
        const out = self.rig.outputBytes();
        for (got, 0..) |*g, i| g.* = spec.read(s.out, out, i);
        const allow = mag; // reused: the allowance per output
        // The rotation carries what the spread moves; the normed values (the output before it) their own spread.
        const normed_at: usize = @intFromBool(self.add);
        for (allow, want, 0..) |*l, w, i| {
            const extra = if (i >= n - per) flip[per + i - (n - per)] else if (self.write_normed and i / per == normed_at) flip[i % per] else 0;
            l.* = spec.bound(s, s.out, self.shape, w, l.*) + extra;
        }
        return host.within(&self.msg, at, got, want, allow);
    }

    fn wrapper(self: *NormRotate) ![]const u8 {
        const got = (try rht.normRotate(self.args[0], if (self.add) self.args[1] else null, self.args[2], self.args[3], self.args[4], BLOCK, self.write_normed, self.rig.stream())) orelse return error.Declined;
        var list: [3]mlx.mlx_array = undefined;
        var len: usize = 0;
        if (got.sum) |x| {
            list[len] = x;
            len += 1;
        }
        if (got.normed) |x| {
            list[len] = x;
            len += 1;
        }
        list[len] = got.rot;
        len += 1;
        defer for (list[0..len]) |x| {
            _ = mlx.mlx_array_free(x);
        };
        return self.rig.bytesOf(list[0..len]);
    }

    fn registered(dt: mlx.mlx_dtype) spec.Spec {
        return normRotateSpec("", dt);
    }
};

fn normRotateDepth(shape: Shape) usize {
    return shape.k + 12; // the row's sum of squares, then the butterfly and its scale
}

fn normRotateSpec(comptime name: []const u8, dt: mlx.mlx_dtype) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = normRotateDepth,
        // The mean, + eps, precise rsqrt, the product with the row, the scale.
        .ops = 6,
        .mids = 1,
        .out = host.elemOf(dt),
        .tiles = .{ .{}, .{}, .{ .size = 1024, .step = 1024, .min = 1024 } },
        .served = &.{ .{ .rows = 1, .n = 1, .k = 4096 }, .{ .rows = 8, .n = 1, .k = 4096 }, .{ .rows = 16, .n = 1, .k = 5120 } },
        // As for the norms: a sum of positive squares rounded in bf16 lands within the normed outputs' own rounding.
        .sums = .{ .bf16 = 0, .why = "positive squares: a pairwise bf16 sum stays within the normed values' rounding" },
    };
}

// ── e4m3 GEMV ──

const Fp8 = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    shape: Shape = undefined,
    x: []f64 = &.{},
    codes: []f64 = &.{},
    scales: []f64 = &.{},
    /// The dequantized weight [n, k] (code value times its row's block scale).
    w: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, dt: mlx.mlx_dtype) Fp8 {
        return .{ .rig = metal.Rig.init(a), .launch = &fp8_launch, .dt = dt };
    }

    pub fn deinit(self: *Fp8) void {
        const a = self.rig.a;
        self.rig.deinit();
        for ([_][]f64{ self.x, self.codes, self.scales, self.w }) |v| a.free(v);
    }

    pub fn geometry(self: *const Fp8, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.rows > 8 or shape.k == 0 or shape.k % 128 != 0 or shape.n == 0) return null;
        const nr: c_int = if (shape.rows <= 4) 1 else 2;
        const sgs: c_int = if (shape.rows <= 4) 2 else 8;
        const n: c_int = @intCast(shape.n);
        var g = metal.Geometry{ .grid = .{ 32 * sgs * @divTrunc(n + nr * sgs - 1, nr * sgs), 1, 1 }, .tg = .{ 32 * sgs, 1, 1 } };
        g.out(&.{ @intCast(shape.rows), n }, self.dt);
        g.dtype("T", self.dt);
        g.int("M", shape.rows);
        g.int("N", shape.n);
        g.int("K", shape.k);
        g.int("NR", nr);
        g.int("SGS", sgs);
        return g;
    }

    pub fn prepare(self: *Fp8, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        const nk = shape.n * shape.k;
        for ([_]*[]f64{ &self.x, &self.codes, &self.scales, &self.w }, [_]usize{ shape.rows * shape.k, nk, shape.n * shape.k / 128, nk }) |p, len| {
            a.free(p.*);
            p.* = &.{};
            p.* = try a.alloc(f64, len);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // Integer codes (1..15, either sign) mirrored across K with scale 1, against mirrored integer x. Only x's
            // +1 offsets (32..39) survive the mirror, and the codes there are 1, a, -a, b, -b, c, -c, 0: every output is
            // exactly 1, so a lower-precision sum's miss is measured against the smallest output, never a lucky large one.
            host.mirrored(rng, self.x, shape.k);
            const h = shape.k / 2;
            for (0..shape.n) |o| {
                var prev: f64 = 0;
                for (0..h) |j| {
                    const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 15));
                    const signed: f64 = switch (j) {
                        32 => 1,
                        33, 35, 37 => v,
                        34, 36, 38 => -prev,
                        39 => 0,
                        else => if (rng.boolean()) v else -v,
                    };
                    prev = signed;
                    const c: f64 = @floatFromInt(intCode(signed));
                    self.codes[o * shape.k + j] = c;
                    self.codes[o * shape.k + h + j] = c;
                }
            }
            @memset(self.scales, 1);
        } else {
            host.uniformInto(rng, self.x, -2, 2, self.dt);
            for (self.codes) |*c| c.* = @floatFromInt(host.e4m3Code(rng, -4, 2, true));
            host.uniformInto(rng, self.scales, 0.01, 0.05, .float32);
            if (fill == .nonfinite) plant(self.x, shape.rows, shape.k);
        }
        for (self.w, self.codes, 0..) |*w, c, i| w.* = host.e4m3(@intFromFloat(c)) * self.scales[(i / shape.k) * (shape.k / 128) + (i % shape.k) / 128];
        self.rig.reset();
        const rows: c_int = @intCast(shape.rows);
        const n: c_int = @intCast(shape.n);
        const k: c_int = @intCast(shape.k);
        self.args[0] = try self.rig.input(self.dt, &.{ rows, k }, self.x);
        self.args[1] = try self.rig.input(.uint8, &.{ n, k }, self.codes);
        self.args[2] = try self.rig.input(.float32, &.{ n, @divExact(k, 128) }, self.scales);
        return true;
    }

    /// The e4m3 code of a small integer.
    fn intCode(v: f64) u8 {
        const mag: usize = @intFromFloat(@abs(v));
        return int_codes[mag] | @as(u8, if (v < 0) 0x80 else 0);
    }

    const int_codes = blk: {
        @setEvalBranchQuota(100_000);
        var t: [16]u8 = @splat(0);
        for (1..16) |v| {
            var c: u8 = 0;
            while (host.e4m3(c) != @as(f64, @floatFromInt(v))) c += 1;
            t[v] = c;
        }
        break :blk t;
    };

    pub fn run(self: *Fp8) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Fp8) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *const Fp8, want: []f64, mag: []f64) !void {
        const k = self.shape.k;
        for (0..self.shape.rows) |m| for (0..self.shape.n) |o| {
            const d = host.dot(self.x[m * k ..][0..k], self.w[o * k ..][0..k]);
            want[m * self.shape.n + o] = d[0];
            mag[m * self.shape.n + o] = d[1];
        };
    }

    /// fp32 sums of the integer stress terms (under 2^24) are exact in any order; fp16 and bf16 are run for the first
    /// eight outputs of each row: a lower bound on the miss.
    pub fn simulate(self: *const Fp8, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const a = self.rig.a;
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        try self.reference(out, mag);
        if (acc == .f32) {
            for (mag) |m| if (m >= 0x1p24) return error.StressPastExact;
            return;
        }
        const k = self.shape.k;
        const t = try a.alloc(f64, k);
        defer a.free(t);
        for (0..self.shape.rows) |m| for (0..@min(8, self.shape.n)) |o| {
            for (t, 0..) |*v, i| v.* = self.x[m * k + i] * self.w[o * k + i];
            out[m * self.shape.n + o] = spec.sumIn(acc, order, t);
        };
    }

    fn wrapper(self: *Fp8) ![]const u8 {
        const y = try sushi.fp8Linear(self.args[0], self.args[1], self.args[2], self.rig.stream());
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

fn fp8Depth(shape: Shape) usize {
    return shape.k + 3; // the decode times its scale, the product, the row's sum
}

fn fp8Spec(comptime name: []const u8, dt: mlx.mlx_dtype) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = fp8Depth,
        .out = host.elemOf(dt),
        // Rows 1..8 (two weight rows a simdgroup past 4), N against 16 rows a group, K against 32 lanes of 16 codes.
        .tiles = .{ .{ .size = 4 }, .{ .size = 16 }, .{ .size = 512, .step = 128, .min = 128 } },
        .served = &.{ .{ .rows = 1, .n = 512, .k = 4096 }, .{ .rows = 4, .n = 512, .k = 4096 }, .{ .rows = 8, .n = 512, .k = 4096 } },
    };
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    inline for (.{ .float16, .bfloat16 }) |dt| inline for (.{ 512, 1024, 2048, 4096 }) |block| inline for (.{ false, true }) |inv| {
        const name = std.fmt.comptimePrint("hadamard_block {s} (block {d}, {s})", .{ if (dt == .float16) "f16" else "bf16", block, if (inv) "inverse" else "input side" });
        try set.add(Rotate, .init(a, dt, block, inv), rotateSpec(name, dt, block), null);
    };
    inline for (.{ .float16, .bfloat16 }) |dt| inline for (.{ .{ true, true }, .{ true, false }, .{ false, false } }) |f| {
        const name = std.fmt.comptimePrint("norm_rotate {s} ({s}, {s})", .{ if (dt == .float16) "f16" else "bf16", if (f[0]) "add" else "no add", if (f[1]) "normed out" else "rotation only" });
        var t = NormRotate.init(a, dt, f[0], f[1]);
        var sp = normRotateSpec(name, dt);
        if (dt == .bfloat16 and !f[0] and !f[1]) {
            t.launch = &norm_rotate_rot_bf16_launch;
            sp.known = rot_bf16_m1;
        }
        try set.add(NormRotate, t, sp, null);
    };
    inline for (.{ .bfloat16, .float16, .float32 }) |dt| {
        const name = std.fmt.comptimePrint("fp8_gemv {s}", .{switch (dt) {
            .bfloat16 => "bf16",
            .float16 => "f16",
            else => "f32",
        }});
        try set.add(Fp8, .init(a, dt), fp8Spec(name, dt), null);
    }
}

// ── tests ──

test "kernel contract quant: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const shape = k.spec.served[0];
        const ours, const theirs = if (std.mem.startsWith(u8, k.spec.name, "hadamard")) try drift(Rotate, a, k.ctx, shape) else if (std.mem.startsWith(u8, k.spec.name, "norm")) try drift(NormRotate, a, k.ctx, shape) else try drift(Fp8, a, k.ctx, shape);
        defer a.free(ours);
        if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{k.spec.name});
        try std.testing.expectEqualSlices(u8, theirs, ours);
    }
}

fn drift(comptime T: type, a: std.mem.Allocator, ctx: *anyopaque, shape: Shape) !struct { []u8, []const u8 } {
    const t: *T = @ptrCast(@alignCast(ctx));
    try std.testing.expect(try t.prepare(shape, .random, 7));
    try t.run();
    const ours = try a.dupe(u8, t.rig.outputBytes());
    errdefer a.free(ours);
    return .{ ours, try t.wrapper() };
}
