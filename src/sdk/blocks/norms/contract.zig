//! The norms area's kernel contract: add_norm's three arms (plain, MoE combine, adaLN gate + modulation) and the
//! residual-norm seam, each bit-equal to the MLX op chain it replaces. Listed in `kernel_contract/registry.zig`.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const add_norm = @import("add_norm.zig");
const residual_norm = @import("residual_norm.zig");

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

const Cap = metal.Cap;

/// One launch per registration: each instantiates its own pipelines, whose M1/M2 limits `offline.zig` recorded.
/// Threadgroup bytes: `local_sums[32]` (add_norm), plus `local_inv[1]` (residual norm).
fn addNormLaunch(comptime name: [:0]const u8, comptime inputs: []const [*:0]const u8, comptime src: [:0]const u8, comptime caps: []const Cap) metal.Launch {
    return .{ .name = name, .inputs = inputs, .outputs = &.{ "H_NEW", "HN" }, .source = src, .tg_bytes = 128, .m1 = 1024, .m2 = 1024, .caps = caps };
}

fn residualLaunch(comptime caps: []const Cap) metal.Launch {
    return .{
        .name = "mlxserve_residual_norm",
        .inputs = &.{ "a", "b1", "b2", "w1", "eps1", "w2", "eps2", "wc", "epsc", "scale", "wp", "wp2", "eps" },
        .outputs = &.{ "res", "out", "out2" },
        .source = residual_norm.RESIDUAL_NORM_SOURCE,
        .header = residual_norm.RESIDUAL_NORM_HEADER,
        .tg_bytes = 132,
        .m1 = 1024,
        .m2 = 1024,
        .caps = caps,
    };
}

const plain_in: []const [*:0]const u8 = &.{ "H", "X", "W", "eps" };
const plain_bf16 = addNormLaunch("msv_add_norm", plain_in, add_norm.PLAIN_SOURCE, &.{ .{ .k = 4097, .m1 = 928, .m2 = 928 }, .{ .k = 8192, .m1 = 928, .m2 = 928 } });
const plain_f16 = addNormLaunch("msv_add_norm", plain_in, add_norm.PLAIN_SOURCE, &.{.{ .k = 5120, .m1 = 928, .m2 = 928 }});
const moe_bf16 = addNormLaunch("msv_moe_combine_add_norm", &.{ "H", "Y", "WT", "XS", "W", "eps" }, add_norm.MOE_SOURCE, &.{.{ .k = 2688, .m1 = 928, .m2 = 928 }});
const mod_in: []const [*:0]const u8 = &.{ "H", "X", "G", "W", "SC", "SH", "ROWMOD", "eps" };
const mod_bf16 = addNormLaunch("msv_gate_norm_mod", mod_in, add_norm.MOD_SOURCE, &.{ .{ .k = 4097, .m1 = 928, .m2 = 928 }, .{ .k = 5376, .m1 = 928, .m2 = 928 } });
const mod_no_add_bf16 = addNormLaunch("msv_gate_norm_mod", mod_in, add_norm.MOD_SOURCE, &.{.{ .k = 5376, .m1 = 928, .m2 = 928 }});
const seam_post_bf16 = residualLaunch(&.{});
const seam_post_f32 = residualLaunch(&.{});
const seam_gemma_bf16 = residualLaunch(&.{});
const seam_dense_bf16 = residualLaunch(&.{});
const seam_moe_bf16 = residualLaunch(&.{.{ .m1 = 928, .m2 = 928 }});

const Arm = enum { plain, moe, mod, mod_no_add, residual };

/// The residual seam's stages (its kernel's template flags; the post norm is always on).
const Flags = struct { pre1: bool = false, b2: bool = false, pre2: bool = false, prec: bool = false, scale: bool = false, post2: bool = false };
/// Gemma 4's MoE seam: both expert outputs normed, summed, normed, added, scaled, then the next norm.
const moe_seam = Flags{ .pre1 = true, .b2 = true, .pre2 = true, .prec = true, .scale = true };

const EPS: f32 = 1e-5;
/// MoE slots, and the adaLN modulation table's rows.
const K = 8;
const R = 9;

/// One registration: an arm at a dtype. Host values live in `arena`, one prepare at a time.
const Norm = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    arm: Arm,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    flags: Flags = .{},
    shape: Shape = undefined,
    v: Values = .{},
    args: [13]mlx.mlx_array = undefined,
    n_args: usize = 0,

    /// Every arm's inputs by role; an arm reads the ones it binds.
    const Values = struct {
        h: []f64 = &.{}, // the residual (`a` on the residual arms)
        x: []f64 = &.{}, // the added branch (`b1`)
        w: []f64 = &.{}, // the norm weight (`wp`)
        y: []f64 = &.{},
        wt: []f64 = &.{},
        xs: []f64 = &.{},
        g: []f64 = &.{},
        sc: []f64 = &.{},
        sh: []f64 = &.{},
        rm: []f64 = &.{},
        b2: []f64 = &.{},
        w1: []f64 = &.{},
        w2: []f64 = &.{},
        wc: []f64 = &.{},
        wp2: []f64 = &.{},
        scale: []f64 = &.{},
    };

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, arm: Arm, dt: mlx.mlx_dtype) Norm {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .arm = arm, .dt = dt, .arena = .init(a) };
    }

    fn seam(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype, flags: Flags) Norm {
        var n = init(a, launch, .residual, dt);
        n.flags = flags;
        return n;
    }

    pub fn deinit(self: *Norm) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn elem(self: *const Norm) spec.Elem {
        return switch (self.dt) {
            .float32 => .f32,
            .float16 => .f16,
            else => .bf16,
        };
    }

    fn rt(self: *const Norm, v: f64) f64 {
        return spec.roundTo(self.elem(), v);
    }

    fn residual(self: *const Norm) bool {
        return self.arm == .residual;
    }

    pub fn geometry(self: *const Norm, shape: Shape) ?metal.Geometry {
        const d: c_int = @intCast(shape.k);
        const rows: c_int = @intCast(shape.rows);
        if (self.residual()) {
            // MLX's rms_single_row launch: ceil(D / 4) threads rounded up to simdgroups, one group a row.
            const tg = @divTrunc(@divTrunc(d + 3, 4) + 31, 32) * 32;
            if (d <= 0 or tg > 1024) return null;
            var g = metal.Geometry{ .grid = .{ tg, rows, 1 }, .tg = .{ tg, 1, 1 } };
            for (0..3) |_| g.out(&.{ rows, d }, self.dt);
            const f = self.flags;
            g.dtype("T", self.dt);
            g.int("AXIS", d);
            g.flag("PRE1", f.pre1);
            g.flag("B2", f.b2);
            g.flag("PRE2", f.b2 and f.pre2);
            g.flag("PREC", f.prec);
            g.flag("SCALE", f.scale);
            g.flag("POST", true);
            g.flag("POST2", f.post2);
            if (!f.post2) g.unread = 0b100;
            return g;
        }
        // MLX's single-row rms_norm up to 4096 (ceil(D / 4) threads), its looped one (1024) to 8192.
        if (d <= 0 or d > 8192) return null;
        const tn: c_int = if (d <= 4096) @divTrunc(d + 3, 4) else 1024;
        var g = metal.Geometry{ .grid = .{ tn * rows, 1, 1 }, .tg = .{ tn, 1, 1 } };
        const two_d = self.arm == .mod or self.arm == .mod_no_add;
        for (0..2) |_| if (two_d) g.out(&.{ rows, d }, self.dt) else g.out(&.{ 1, rows, d }, self.dt);
        g.dtype("T", self.dt);
        g.int("D", d);
        g.int("TN", tn);
        if (two_d) g.int("HAS_ADD", @intFromBool(self.arm == .mod));
        if (self.arm == .mod_no_add) g.unread = 0b1;
        if (self.arm == .moe) {
            g.int("K", K);
            g.int("SHARED", 1);
        }
        return g;
    }

    pub fn prepare(self: *Norm, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        try self.fillValues(fill, seed);
        try self.bind();
        return true;
    }

    fn fillValues(self: *Norm, mode: spec.Fill, seed: u64) !void {
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = self.shape.rows;
        const d = self.shape.k;
        const n = rows * d;
        const Gen = struct {
            fn vals(al: std.mem.Allocator, r: std.Random, len: usize, lo: f64, hi: f64, nrm: *const Norm) ![]f64 {
                const out = try al.alloc(f64, len);
                for (out) |*v| v.* = nrm.rt(lo + (hi - lo) * r.float(f64));
                return out;
            }
        };
        var v: Values = .{};
        v.h = try Gen.vals(a, rng, n, -3, 3, self);
        v.x = try Gen.vals(a, rng, n, -3, 3, self);
        v.w = try Gen.vals(a, rng, d, 0.5, 1.5, self);
        switch (self.arm) {
            .plain => {},
            .moe => {
                v.y = try Gen.vals(a, rng, rows * K * d, -2, 2, self);
                v.wt = try a.alloc(f64, rows * K);
                for (v.wt) |*t| t.* = spec.roundTo(.f32, rng.float(f64) / K);
                v.xs = try Gen.vals(a, rng, n, -1, 1, self);
            },
            .mod, .mod_no_add => {
                v.g = try Gen.vals(a, rng, R * d, -0.5, 0.5, self);
                v.sc = try Gen.vals(a, rng, R * d, -0.5, 0.5, self);
                v.sh = try Gen.vals(a, rng, R * d, -0.5, 0.5, self);
                v.rm = try a.alloc(f64, rows);
                for (v.rm) |*t| t.* = @floatFromInt(rng.uintLessThan(u32, R));
            },
            .residual => {
                v.b2 = try Gen.vals(a, rng, n, -3, 3, self);
                v.w1 = try Gen.vals(a, rng, d, 0.5, 1.5, self);
                v.w2 = try Gen.vals(a, rng, d, 0.5, 1.5, self);
                v.wc = try Gen.vals(a, rng, d, 0.5, 1.5, self);
                v.wp2 = try Gen.vals(a, rng, d, 0.5, 1.5, self);
                v.scale = try Gen.vals(a, rng, 1, 0.8, 1.2, self);
            },
        }
        switch (mode) {
            .random => {},
            .nonfinite => {
                const at = [_][2]usize{ .{ 0, d / 3 }, .{ 1, d / 2 }, .{ 2, d - 1 }, .{ 3, 0 }, .{ 3, d - 1 } };
                const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64) };
                for (at, put) |p, val| if (p[0] < rows) {
                    v.h[p[0] * d + p[1]] = val;
                };
                // A NaN through the added branch too.
                if (rows > 4) (if (self.arm == .moe) v.xs else v.x)[4 * d + d / 4] = std.math.nan(f64);
            },
            .stress => {
                // Gate 4: the norm's sum of squares over integers (exact in fp32, past fp16's range); every other
                // stage exact, so only that sum can miss.
                for (0..rows) |r| for (0..d) |i| {
                    v.h[r * d + i] = @floatFromInt(if (i < d / 2) @as(i32, @intCast(1 + rng.uintLessThan(u32, 64))) - 32 else @as(i32, @intCast(rng.uintLessThan(u32, 65))) - 32);
                };
                @memset(v.w, 1);
                switch (self.arm) {
                    .plain, .mod, .mod_no_add => @memset(v.x, 0),
                    .moe => {
                        @memset(v.y, 0);
                        @memset(v.xs, 0);
                        @memset(v.x, 0);
                    },
                    .residual => {
                        // Pre-normed rows of +-4 with the same signs: each normed row is +-1 exactly, their sum +-2,
                        // its norm +-1 again, so the residual stays an integer. A branch added raw is zero.
                        const f = self.flags;
                        for (v.x, v.b2) |*b1, *b2| {
                            const sign: f64 = if (rng.boolean()) 1 else -1;
                            b1.* = if (f.pre1) 4 * sign else 0;
                            b2.* = if (f.pre2) 4 * sign else 0;
                        }
                        for ([_][]f64{ v.w1, v.w2, v.wc, v.wp2, v.scale }) |t| @memset(t, 1);
                    },
                }
                if (self.arm == .mod or self.arm == .mod_no_add) {
                    @memset(v.sc, 0);
                    @memset(v.sh, 0);
                }
            },
        }
        self.v = v;
    }

    fn bind(self: *Norm) !void {
        const rig = &self.rig;
        rig.reset();
        const rows: c_int = @intCast(self.shape.rows);
        const d: c_int = @intCast(self.shape.k);
        const dt = self.dt;
        const v = self.v;
        const eps = try rig.scalar(EPS);
        const row3: []const c_int = &.{ 1, rows, d };
        const row2: []const c_int = &.{ rows, d };
        const vec: []const c_int = &.{d};
        const table: []const c_int = &.{ R, d };
        self.n_args = switch (self.arm) {
            .plain => self.setArgs(.{ try rig.input(dt, row3, v.h), try rig.input(dt, row3, v.x), try rig.input(dt, vec, v.w), eps }),
            .moe => self.setArgs(.{
                try rig.input(dt, row3, v.h),  try rig.input(dt, &.{ rows, K, d }, v.y), try rig.input(.float32, &.{ rows, K }, v.wt),
                try rig.input(dt, row2, v.xs), try rig.input(dt, vec, v.w),              eps,
            }),
            .mod, .mod_no_add => blk: {
                const h = try rig.input(dt, row2, v.h);
                const sc = try rig.input(dt, table, v.sc);
                const add = self.arm == .mod;
                const x = if (add) try rig.input(dt, row2, v.x) else h;
                const g = if (add) try rig.input(dt, table, v.g) else sc;
                break :blk self.setArgs(.{ h, x, g, try rig.input(dt, vec, v.w), sc, try rig.input(dt, table, v.sh), try rig.input(.int32, &.{rows}, v.rm), eps });
            },
            .residual => blk: {
                // Stages that are off bind the wrapper's stand-ins (b1, one eps); the kernel never reads them.
                const f = self.flags;
                const b1 = try rig.input(dt, row2, v.x);
                const b2 = if (f.b2) try rig.input(dt, row2, v.b2) else b1;
                const w1 = if (f.pre1) try rig.input(dt, vec, v.w1) else b1;
                const w2 = if (f.b2 and f.pre2) try rig.input(dt, vec, v.w2) else b1;
                const wc = if (f.prec) try rig.input(dt, vec, v.wc) else b1;
                const scale = if (f.scale) try rig.input(dt, &.{1}, v.scale) else b1;
                const wp2 = if (f.post2) try rig.input(dt, vec, v.wp2) else b1;
                break :blk self.setArgs(.{ try rig.input(dt, row2, v.h), b1, b2, w1, eps, w2, eps, wc, eps, scale, try rig.input(dt, vec, v.w), wp2, eps });
            },
        };
    }

    /// The kernel's inputs, in its order; returns how many.
    fn setArgs(self: *Norm, list: anytype) usize {
        inline for (list, 0..) |x, i| self.args[i] = x;
        return list.len;
    }

    pub fn run(self: *Norm) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    fn outputs(self: *const Norm) usize {
        return switch (self.arm) {
            .mod_no_add => 1,
            .residual => if (self.flags.post2) 3 else 2,
            else => 2,
        };
    }

    pub fn count(self: *const Norm) usize {
        return self.outputs() * self.shape.rows * self.shape.k;
    }

    /// Row `r`'s residual after the add, rounded at the op chain's points (an output, and the norm's input).
    fn residualRow(self: *const Norm, r: usize, out: []f64) !void {
        const d = self.shape.k;
        const v = self.v;
        const at = r * d;
        switch (self.arm) {
            .plain => for (out, 0..) |*o, i| {
                o.* = self.rt(v.h[at + i] + v.x[at + i]);
            },
            .moe => for (out, 0..) |*o, i| {
                var routed: f64 = 0;
                for (0..K) |e| routed += v.y[(r * K + e) * d + i] * v.wt[r * K + e];
                o.* = self.rt(v.h[at + i] + self.rt(self.rt(routed) + v.xs[at + i]));
            },
            .mod, .mod_no_add => for (out, 0..) |*o, i| {
                const m: usize = @intFromFloat(v.rm[r]);
                const delta = if (self.arm == .mod) self.rt(v.x[at + i] * v.g[m * d + i]) else 0;
                o.* = self.rt(v.h[at + i] + delta);
            },
            .residual => {
                const f = self.flags;
                const a = self.arena.child_allocator;
                const c = try a.alloc(f64, d);
                defer a.free(c);
                const b1 = v.x[at..][0..d];
                const b2 = v.b2[at..][0..d];
                const inv1 = invRms(b1);
                const inv2 = invRms(b2);
                for (c, 0..) |*o, i| {
                    const n1 = if (f.pre1) self.rt(v.w1[i] * self.rt(b1[i] * inv1)) else b1[i];
                    const n2 = if (f.pre2) self.rt(v.w2[i] * self.rt(b2[i] * inv2)) else b2[i];
                    o.* = if (f.b2) self.rt(n1 + n2) else n1;
                }
                const invc = invRms(c);
                for (out, 0..) |*o, i| {
                    const nc = if (f.prec) self.rt(v.wc[i] * self.rt(c[i] * invc)) else c[i];
                    const sum = self.rt(v.h[at + i] + nc);
                    o.* = if (f.scale) self.rt(sum * v.scale[0]) else sum;
                }
            },
        }
    }

    /// The norm stage, unrounded in its middle (`mids` covers those roundings): output 1 (and 2) of row `r`.
    fn normRow(self: *const Norm, r: usize, hn: []const f64, inv: f64, out1: []f64, out2: []f64) void {
        const d = self.shape.k;
        const v = self.v;
        for (hn, 0..) |x, i| {
            const n = v.w[i] * x * inv;
            out1[i] = switch (self.arm) {
                .mod, .mod_no_add => blk: {
                    const m: usize = @intFromFloat(v.rm[r]);
                    break :blk n * self.rt(v.sc[m * d + i] + 1) + v.sh[m * d + i];
                },
                else => n,
            };
            if (self.flags.post2) out2[i] = v.wp2[i] * x * inv;
        }
    }

    /// Every output: float64 (`acc` null) or with the sum of squares run in `acc` in `order` (gate 4).
    fn outputsIn(self: *const Norm, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const rows = self.shape.rows;
        const d = self.shape.k;
        const n = rows * d;
        const a = self.arena.child_allocator;
        const hn = try a.alloc(f64, d);
        defer a.free(hn);
        const sq = try a.alloc(f64, d);
        defer a.free(sq);
        const first: usize = if (self.arm == .mod_no_add) 0 else 1;
        for (0..rows) |r| {
            try self.residualRow(r, hn);
            for (sq, hn) |*q, x| q.* = x * x;
            const ss = if (acc) |ac| spec.sumIn(ac, order, sq) else sumF64(sq);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(d)) + EPS);
            if (first == 1) @memcpy(want[r * d ..][0..d], hn);
            const out1 = want[first * n + r * d ..][0..d];
            const out2 = if (self.flags.post2) want[2 * n + r * d ..][0..d] else out1;
            self.normRow(r, hn, inv, out1, out2);
        }
        if (mag) |m| for (m, want, 0..) |*o, w, i| {
            o.* = if (i < first * n) 0 else @abs(w) / 2;
        };
    }

    pub fn reference(self: *const Norm, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Norm, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }

    /// The op chain this arm replaces, on the same inputs: its outputs' bytes.
    pub fn replaced(self: *Norm) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        return switch (self.arm) {
            .plain => blk: {
                const sum = try g.add(x[0], x[1]);
                break :blk self.rig.bytesOf(&.{ sum, try g.rmsNorm(sum, x[2], EPS) });
            },
            .moe => blk: {
                const prod = try g.mul(x[1], try g.expandDims(x[2], -1));
                const routed = try g.astype(try g.sum(prod, 1, false), self.dt);
                const delta = try g.reshape(try g.add(routed, x[3]), &.{ 1, @intCast(self.shape.rows), @intCast(self.shape.k) });
                const sum = try g.add(x[0], delta);
                break :blk self.rig.bytesOf(&.{ sum, try g.rmsNorm(sum, x[4], EPS) });
            },
            .mod, .mod_no_add => blk: {
                const h = if (self.arm == .mod) try g.add(x[0], try g.mul(x[1], try g.take(x[2], x[6], 0))) else x[0];
                const n = try g.rmsNorm(h, x[3], EPS);
                const sc1 = try g.add(try g.take(x[4], x[6], 0), try g.scalar(1, self.dt));
                const m = try g.add(try g.mul(n, sc1), try g.take(x[5], x[6], 0));
                break :blk if (self.arm == .mod) self.rig.bytesOf(&.{ h, m }) else self.rig.bytesOf(&.{m});
            },
            .residual => blk: {
                const f = self.flags;
                const n1 = if (f.pre1) try g.rmsNorm(x[1], x[3], EPS) else x[1];
                const c = if (f.b2) try g.add(n1, if (f.pre2) try g.rmsNorm(x[2], x[5], EPS) else x[2]) else n1;
                const sum = try g.add(x[0], if (f.prec) try g.rmsNorm(c, x[7], EPS) else c);
                const res = if (f.scale) try g.mul(sum, x[9]) else sum;
                const out = try g.rmsNorm(res, x[10], EPS);
                break :blk if (f.post2) self.rig.bytesOf(&.{ res, out, try g.rmsNorm(res, x[11], EPS) }) else self.rig.bytesOf(&.{ res, out });
            },
        };
    }

    /// The shipped wrapper on the same inputs: its outputs' bytes (the declared launch must be its launch).
    fn wrapper(self: *Norm) ![]const u8 {
        const s = self.rig.stream();
        const x = self.args;
        var outs: [3]mlx.mlx_array = @splat(.{ .ctx = null });
        defer for (outs) |o| if (o.ctx != null) {
            _ = mlx.mlx_array_free(o);
        };
        switch (self.arm) {
            .plain, .moe, .mod, .mod_no_add => {
                const got = switch (self.arm) {
                    .plain => try add_norm.addNorm(x[0], x[1], x[2], x[3], s),
                    .moe => try add_norm.moeCombineAddNorm(x[0], x[1], x[2], x[3], x[4], x[5], s),
                    .mod => try add_norm.gateNormMod(x[0], .{ .x = x[1], .gate = x[2] }, x[3], x[4], x[5], x[6], x[7], s),
                    else => try add_norm.gateNormMod(x[0], null, x[3], x[4], x[5], x[6], x[7], s),
                } orelse return error.Declined;
                outs[0] = got.h;
                outs[1] = got.normed;
            },
            .residual => {
                const f = self.flags;
                const none = mlx.mlx_array{ .ctx = null };
                const got = (try residual_norm.fusedResidualNorm(s, x[0], x[1], .{
                    .b2 = if (f.b2) x[2] else none,
                    .w1 = if (f.pre1) x[3] else none,
                    .eps1 = if (f.pre1) x[4] else none,
                    .w2 = if (f.b2 and f.pre2) x[5] else none,
                    .eps2 = if (f.b2 and f.pre2) x[6] else none,
                    .wc = if (f.prec) x[7] else none,
                    .epsc = if (f.prec) x[8] else none,
                    .scale = if (f.scale) x[9] else none,
                    .w_post = x[10],
                    .w_post2 = if (f.post2) x[11] else none,
                    .eps = x[12],
                })) orelse return error.Declined;
                outs = .{ got.sum, got.normed, got.normed2 };
            },
        }
        return switch (self.arm) {
            .mod_no_add => self.rig.bytesOf(outs[1..2]),
            .residual => if (self.flags.post2) self.rig.bytesOf(&outs) else self.rig.bytesOf(outs[0..2]),
            else => self.rig.bytesOf(outs[0..2]),
        };
    }
};

fn sumF64(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |v| s += v;
    return s;
}

/// 1 / rms(x) in float64, eps added to the mean.
fn invRms(x: []const f64) f64 {
    var s: f64 = 0;
    for (x) |v| s += v * v;
    return 1 / @sqrt(s / @as(f64, @floatFromInt(x.len)) + EPS);
}

fn rowLength(shape: Shape) usize {
    return shape.k;
}

fn normSpec(comptime name: []const u8, out: spec.Elem, replaces: []const u8, mids: f64, tile: usize, served: []const Shape, known: ?spec.Known) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = rowLength,
        // f32 roundings after the sum: the mean, + eps, rsqrt (2), the product with the row.
        .ops = 5,
        .mids = mids,
        .out = out,
        .replaces = replaces,
        .tiles = .{ .{}, .{}, .{ .size = tile } },
        .served = served,
        // A sum of squares cannot cancel: summed pairwise in bf16 it stays within the output's own rounding, and
        // overflowing fp16 zeroes a row by only (1 + mids) output ulps more than the bound.
        .sums = .{ .f16 = 32, .bf16 = 0, .why = "positive squares: a pairwise bf16 sum stays near the output's rounding" },
        .known = known,
    };
}

const add_shapes: []const Shape = &.{ .{ .rows = 1, .n = 1, .k = 2048 }, .{ .rows = 16, .n = 1, .k = 4096 }, .{ .rows = 4, .n = 1, .k = 5120 }, .{ .rows = 16, .n = 1, .k = 8192 } };
const moe_shapes: []const Shape = &.{ .{ .rows = 1, .n = 1, .k = 2688 }, .{ .rows = 4, .n = 1, .k = 4096 } };
const mod_shapes: []const Shape = &.{.{ .rows = 64, .n = 1, .k = 5376 }};
const residual_shapes: []const Shape = &.{ .{ .rows = 1, .n = 1, .k = 2816 }, .{ .rows = 3, .n = 1, .k = 4096 }, .{ .rows = 1, .n = 1, .k = 512 } };

const chain_add = "add -> fast::rms_norm";

/// Gate 5's findings on shipped launches (see the report): each must keep failing until its fix lands.
const looped_probe: spec.Known = .{ .gate = .limits, .finding = "M1/M2 build these looped widths (D > 4096) for 928 threads and the launch is 1024; the wrapper probes only the first dispatch past 256 threads in a process (one `looped_ok` for every arm and width), so after another width's probe passes, a looped dispatch goes unprobed and fails on M1/M2" };
const seam_unprobed: spec.Known = .{ .gate = .limits, .finding = "M1/M2 build this flag set for 928 threads; fusedResidualNorm launches ceil(D / 4) rounded to 32 (1024 at D 3969..4096) with no probe (no served Gemma 4 width reaches it: 2816 launches 704)" };

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(Norm, .init(a, &plain_bf16, .plain, .bfloat16), normSpec("add_norm bf16", .bf16, chain_add, 1, 4096, add_shapes, looped_probe), null);
    try set.add(Norm, .init(a, &plain_f16, .plain, .float16), normSpec("add_norm f16", .f16, chain_add, 1, 4096, add_shapes, looped_probe), null);
    try set.add(Norm, .init(a, &moe_bf16, .moe, .bfloat16), normSpec("moe_combine_add_norm bf16 (K 8, shared)", .bf16, "multiply -> sum f32 -> astype -> add shared -> " ++ chain_add, 1, 4096, moe_shapes, null), null);
    try set.add(Norm, .init(a, &mod_bf16, .mod, .bfloat16), normSpec("gate_norm_mod bf16", .bf16, "gate multiply -> add -> rms_norm -> (1 + scale) multiply -> add shift", 3, 4096, mod_shapes, looped_probe), null);
    try set.add(Norm, .init(a, &mod_no_add_bf16, .mod_no_add, .bfloat16), normSpec("gate_norm_mod bf16 (no add)", .bf16, "rms_norm -> (1 + scale) multiply -> add shift", 3, 4096, mod_shapes, looped_probe), null);
    try set.add(Norm, .seam(a, &seam_post_bf16, .bfloat16, .{}), normSpec("residual_norm bf16 (add + post norm)", .bf16, chain_add, 1, 128, residual_shapes, null), null);
    try set.add(Norm, .seam(a, &seam_post_f32, .float32, .{}), normSpec("residual_norm f32 (add + post norm)", .f32, chain_add, 1, 128, residual_shapes, null), null);
    try set.add(Norm, .seam(a, &seam_gemma_bf16, .bfloat16, .{ .pre1 = true, .post2 = true }), normSpec("residual_norm bf16 (pre norm, two post norms)", .bf16, "rms_norm -> add -> rms_norm x2", 1, 128, residual_shapes, null), null);
    try set.add(Norm, .seam(a, &seam_dense_bf16, .bfloat16, .{ .pre1 = true, .scale = true }), normSpec("residual_norm bf16 (pre norm, scale, post norm)", .bf16, "rms_norm -> add -> multiply -> rms_norm", 1, 128, residual_shapes, null), null);
    try set.add(Norm, .seam(a, &seam_moe_bf16, .bfloat16, moe_seam), normSpec("residual_norm bf16 (MoE seam)", .bf16, "rms_norm x2 -> add -> rms_norm -> add -> multiply -> rms_norm", 1, 128, residual_shapes, seam_unprobed), null);
}

// ── tests ──

test "kernel contract norms: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const n: *Norm = @ptrCast(@alignCast(k.ctx));
        try std.testing.expect(try n.prepare(k.spec.served[0], .random, 7));
        try n.run();
        const ours = try a.dupe(u8, n.rig.outputBytes());
        defer a.free(ours);
        const theirs = try n.wrapper();
        if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{k.spec.name});
        try std.testing.expectEqualSlices(u8, theirs, ours);
    }
}
