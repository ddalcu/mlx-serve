//! Inkling 2's kernel contract (`kernel_contract/registry.zig` lists it): the kernels with no sum of their own.
//! `gated` is the attention read times its output gate, bit-equal to the cast, multiply, cast it replaces; `state` a
//! visit's conv state after its rows; `relmask` a few rows' additive attention mask; `act` the experts' soft-capped
//! SwiGLU over a prompt's gathered gate and up rows; `relsplit` and `relcomb` the split read of a few rows behind a
//! cache (each split's online softmax, then the join under the output gate); `oproj` one row's output projection
//! under its gate (affine at every width, or dense).
//! Shapes: `gated` rows x 1 x (heads * head_dim); `state` rows x (kv width) x (hidden); `relmask` rows x keys x
//! window (0 on a global layer); `act` picks x (1, or 2 with per-expert scales) x (expert width).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const source = @import("../../mlx/kernels.zig").source;
const kernels = @import("kernels.zig");
const vision_contract = @import("vision_contract.zig");
const layers = @import("layers.zig");
const config = @import("config.zig");
const attn = @import("../../sdk/blocks/attention/contract.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

fn elemOf(dt: mlx.mlx_dtype) spec.Elem {
    return if (dt == .bfloat16) .bf16 else .f32;
}

/// `n` values of `e`, magnitudes spread over 2^-8 .. 2^8.
fn spread(a: std.mem.Allocator, rng: std.Random, n: usize, e: spec.Elem) ![]f64 {
    const out = try a.alloc(f64, n);
    for (out) |*v| v.* = spec.roundTo(e, (rng.float(f64) * 2 - 1) * std.math.pow(f64, 2, @floatFromInt(rng.intRangeAtMost(i32, -8, 8))));
    return out;
}

/// NaN in row 0, +Inf in row 1, -Inf in row 2, both infinities in row 3, for the rows `x` (`width` a row) holds.
fn plant(x: []f64, width: usize) void {
    const at = [_][2]usize{ .{ 0, width / 3 }, .{ 1, width / 2 }, .{ 2, width - 1 }, .{ 3, 0 }, .{ 3, width - 1 } };
    const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64) };
    for (at, put) |p, v| if (p[0] * width + p[1] < x.len) {
        x[p[0] * width + p[1]] = v;
    };
}

const gated_launch = metal.Launch{ .name = "msv_inkling2_gated", .inputs = &.{ "ATT", "GATE" }, .outputs = &.{"OUT"}, .source = source("plugins/inkling2/kernels/gated.metal"), .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

fn Gated(comptime dt: mlx.mlx_dtype) type {
    return struct {
        rig: metal.Rig,
        launch: *const metal.Launch = &gated_launch,
        arena: std.heap.ArenaAllocator,
        shape: Shape = undefined,
        att: []f64 = &.{},
        gate: []f64 = &.{},
        args: [2]mlx.mlx_array = undefined,

        const Self = @This();

        fn init(a: std.mem.Allocator) Self {
            return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
            if (shape.rows == 0 or shape.k == 0 or shape.rows * shape.k % 64 != 0) return null;
            var g = metal.Geometry{ .grid = .{ @intCast(shape.rows * shape.k), 1, 1 }, .tg = .{ 64, 1, 1 } };
            g.out(&.{ 1, @intCast(shape.rows), @intCast(shape.k) }, dt);
            g.dtype("T", dt);
            return g;
        }

        pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
            if (self.geometry(shape) == null) return false;
            self.shape = shape;
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const n = shape.rows * shape.k;
            self.att = try spread(a, rng, n, elemOf(dt));
            // A sigmoid's range, away from zero so an infinity keeps its sign.
            self.gate = try a.alloc(f64, n);
            for (self.gate) |*v| v.* = spec.roundTo(.f32, 0.02 + 0.97 * rng.float(f64));
            if (fill == .nonfinite) plant(self.att, shape.k);
            self.rig.reset();
            // The attention read arrives as heads, the gate flat.
            self.args = .{
                try self.rig.input(dt, &.{ 1, @intCast(shape.rows), 1, @intCast(shape.k) }, self.att),
                try self.rig.input(.float32, &.{@intCast(n)}, self.gate),
            };
            return true;
        }

        pub fn run(self: *Self) !void {
            try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
        }

        pub fn count(self: *const Self) usize {
            return self.shape.rows * self.shape.k;
        }

        pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
            for (want, mag, self.att, self.gate) |*w, *m, x, gt| {
                w.* = x * gt;
                m.* = @abs(w.*);
            }
        }

        /// The chain it replaces on the same inputs: the read in f32 times the gate, rounded once.
        pub fn replaced(self: *Self) ![]const u8 {
            var g = ops.Graph.init(self.rig.a, self.rig.stream());
            defer g.deinit();
            const flat = try g.reshape(self.args[0], &.{ 1, @intCast(self.shape.rows), @intCast(self.shape.k) });
            const gate = try g.reshape(self.args[1], &.{ 1, @intCast(self.shape.rows), @intCast(self.shape.k) });
            return self.rig.bytesOf(&.{try g.astype(try g.mul(try g.astype(flat, .float32), gate), dt)});
        }

        fn wrapper(self: *Self) ![]const u8 {
            var g = ops.Graph.init(self.rig.a, self.rig.stream());
            defer g.deinit();
            return self.rig.bytesOf(&.{try kernels.gated(&g, self.args[0], self.args[1], @intCast(self.shape.rows))});
        }
    };
}

const oproj_launch = metal.Launch{ .name = "msv_inkling2_oproj", .inputs = &.{ "ATT", "GATE", "WQ", "WS", "WB" }, .outputs = &.{"Y"}, .source = source("plugins/inkling2/kernels/oproj.metal"), .header = kernels.bank_header, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// One row's output projection under its gate: an affine weight of `bits` at group `gs`, or dense (`bits` 0). Shape:
/// (output rows) x 1 x (heads * head_dim): the rows the gates vary are the weight's.
const OProj = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &oproj_launch,
    bits: u32,
    gs: usize,
    shape: Shape = undefined,
    /// The gated read as the kernel rounds it, its magnitudes, and a dense weight's values.
    x: []f64 = &.{},
    ax: []f64 = &.{},
    dense: []f64 = &.{},
    w: host.Affine,
    args: [5]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, bits: u32, gs: usize) OProj {
        return .{ .rig = metal.Rig.init(a), .bits = bits, .gs = gs, .w = .{ .bits = @max(bits, 2), .gs = gs, .n = 0, .k = 0 } };
    }

    pub fn deinit(self: *OProj) void {
        const a = self.rig.a;
        self.rig.deinit();
        for ([_][]f64{ self.x, self.ax, self.dense }) |v| a.free(v);
        self.w.deinit(a);
    }

    pub fn geometry(self: *const OProj, shape: Shape) ?metal.Geometry {
        if (shape.n != 1 or shape.rows == 0 or shape.k == 0 or shape.k % 16 != 0 or shape.k % self.gs != 0) return null;
        var g = metal.Geometry{ .grid = .{ 32, @intCast(shape.rows), 1 }, .tg = .{ 32, 4, 1 } };
        g.out(&.{ 1, 1, @intCast(shape.rows) }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.dtype("S", .bfloat16);
        g.int("K", shape.k);
        g.int("GS", if (self.bits == 0) @as(usize, 16) else self.gs);
        g.int("BITS", if (self.bits == 0) @as(u32, 16) else self.bits);
        g.int("FMT", @as(u32, if (self.bits == 0) 4 else 3));
        return g;
    }

    pub fn prepare(self: *OProj, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        const n = shape.rows;
        const k = shape.k;
        for ([_]*[]f64{ &self.x, &self.ax, &self.dense }) |v| {
            a.free(v.*);
            v.* = &.{};
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const stress = fill == .stress;
        const att = try a.alloc(f64, k);
        defer a.free(att);
        const gate = try a.alloc(f64, k);
        defer a.free(gate);
        // Stress: integer reads under a gate of one, so the gated read is the read and every sum is exact in fp32.
        if (stress) host.mirrored(rng, att, k) else host.uniformInto(rng, att, -2, 2, .bfloat16);
        for (gate) |*v| v.* = if (stress) 1 else spec.roundTo(.f32, 0.02 + 0.97 * rng.float(f64));
        self.x = try a.alloc(f64, k);
        self.ax = try a.alloc(f64, k);
        for (self.x, self.ax, att, gate) |*x, *m, v, gt| {
            x.* = spec.roundTo(.bf16, v * gt);
            m.* = @abs(x.*);
        }
        self.rig.reset();
        self.args[0] = try self.rig.input(.bfloat16, &.{ 1, @intCast(k) }, att);
        self.args[1] = try self.rig.input(.float32, &.{@intCast(k)}, gate);
        if (self.bits == 0) {
            self.dense = try a.alloc(f64, n * k);
            if (stress) {
                // As the affine stress: integers under 16, each row's halves equal, against the mirrored inputs.
                const h = k / 2;
                for (0..n) |r| for (0..h) |j| {
                    const v: f64 = @floatFromInt(rng.intRangeAtMost(i32, 0, 15));
                    self.dense[r * k + j] = v;
                    self.dense[r * k + h + j] = v;
                };
            } else host.uniformInto(rng, self.dense, -0.05, 0.05, .bfloat16);
            // Every output reads every input, so the planted values go in the weight's rows: NaN in row 0, +Inf in 1,
            // -Inf in 2, both in 3.
            if (fill == .nonfinite) plant(self.dense, k);
            self.args[2] = try self.rig.input(.bfloat16, &.{ @intCast(n), @intCast(k) }, self.dense);
            self.args[3] = self.args[2];
            self.args[4] = self.args[2];
        } else {
            try self.w.fill(a, rng, n, k, .bfloat16, stress);
            // A NaN scale in rows 0 and 1 (an infinite scale's sign would follow each chunk's partial sum, not the row's).
            if (fill == .nonfinite) for (0..@min(2, n)) |r| {
                self.w.scales[r * k / self.gs + r] = std.math.nan(f64);
            };
            self.args[2] = try self.rig.input(.uint32, &.{ @intCast(n), @intCast(k * self.bits / 32) }, self.w.words);
            self.args[3] = try self.rig.input(.bfloat16, &.{ @intCast(n), @intCast(k / self.gs) }, self.w.scales);
            self.args[4] = try self.rig.input(.bfloat16, &.{ @intCast(n), @intCast(k / self.gs) }, self.w.biases);
        }
        return true;
    }

    pub fn run(self: *OProj) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const OProj) usize {
        return self.shape.rows;
    }

    /// Row `r` in float64 and its terms' magnitude, factored per group for an affine weight as the kernel sums it.
    fn output(self: *const OProj, r: usize) [2]f64 {
        const k = self.shape.k;
        if (self.bits == 0) return host.dot(self.x, self.dense[r * k ..][0..k]);
        var y: f64 = 0;
        var mag: f64 = 0;
        var g: usize = 0;
        while (g < k) : (g += self.gs) {
            var xq: f64 = 0;
            var xsum: f64 = 0;
            var aq: f64 = 0;
            var ax: f64 = 0;
            for (g..g + self.gs) |i| {
                const q: f64 = @floatFromInt(self.w.codes[r * k + i]);
                xq += self.x[i] * q;
                xsum += self.x[i];
                aq += @abs(self.x[i] * q);
                ax += self.ax[i];
            }
            const s = self.w.scales[(r * k + g) / self.gs];
            const b = self.w.biases[(r * k + g) / self.gs];
            y += s * xq + b * xsum;
            mag += @abs(s) * aq + @abs(b) * ax;
        }
        return .{ y, mag };
    }

    pub fn reference(self: *OProj, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, r| {
            const o = self.output(r);
            w.* = o[0];
            m.* = o[1];
        }
    }

    /// fp32 sums of the integer stress terms are exact in any order; fp16 and bf16 run for the first eight rows.
    pub fn simulate(self: *OProj, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const a = self.rig.a;
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        try self.reference(out, mag);
        if (acc == .f32) {
            for (mag) |m| if (m >= 0x1p24) return error.StressPastExact;
            return;
        }
        const k = self.shape.k;
        const t = try a.alloc(f64, 3 * k);
        defer a.free(t);
        for (0..@min(8, self.shape.rows)) |r| {
            if (self.bits == 0) @memcpy(t[0..k], self.dense[r * k ..][0..k]) else self.w.row(r, t[0..k], t[k .. 2 * k]);
            for (t[2 * k ..], t[0..k], self.x) |*o, w, x| o.* = x * w;
            out[r] = spec.sumIn(acc, order, t[2 * k ..]);
        }
    }

    fn linear(self: *const OProj) layers.Linear {
        if (self.bits == 0) return .{ .w = self.args[2] };
        return .{ .w = self.args[2], .s = self.args[3], .b = self.args[4], .bits = self.bits, .group_size = @intCast(self.gs) };
    }

    fn wrapper(self: *OProj) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const o = self.linear();
        return self.rig.bytesOf(&.{try kernels.oGated(&g, &o, self.args[0], self.args[1])});
    }
};

fn oprojDepth(shape: Shape) usize {
    return shape.k + 4; // a row's sum, behind the gate's product and the dequant
}

fn oprojSpec(comptime what: []const u8, step: usize, sums: spec.Sums) spec.Spec {
    return .{
        .name = what,
        .accum = .f32,
        .depth = oprojDepth,
        .ops = 2,
        .mids = 1,
        .out = .bf16,
        // Rows: four SIMD groups of one row a threadgroup; k: a sweep of 32 lanes x 16 inputs.
        .tiles = .{ .{ .size = 4, .step = 4, .min = 1 }, .{}, .{ .size = 512, .step = step, .min = step } },
        // The release's geometry (24 x 128 into 2304) and a narrow row count.
        .served = &.{ .{ .rows = 2304, .n = 1, .k = 3072 }, .{ .rows = 64, .n = 1, .k = 3072 } },
        .sums = sums,
    };
}

const state_launch = metal.Launch{ .name = "msv_inkling2_state", .inputs = &.{ "ST", "KP", "VP", "OP", "XM" }, .outputs = &.{"NST"}, .source = source("plugins/inkling2/kernels/state.metal"), .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// The conv kernel's taps: the state holds one row fewer.
const KT = 4;

fn State(comptime dt: mlx.mlx_dtype) type {
    return struct {
        rig: metal.Rig,
        launch: *const metal.Launch = &state_launch,
        arena: std.heap.ArenaAllocator,
        shape: Shape = undefined,
        /// `st` `[KT - 1, 2 C + 2 D]`, then the rows' `kp`, `vp` `[S, C]` and `op`, `xm` `[S, D]`.
        st: []f64 = &.{},
        new: [4][]f64 = @splat(&.{}),
        args: [5]mlx.mlx_array = undefined,

        const Self = @This();

        fn init(a: std.mem.Allocator) Self {
            return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        fn width(shape: Shape) usize {
            return 2 * shape.n + 2 * shape.k;
        }

        pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
            if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or (KT - 1) * width(shape) % 64 != 0) return null;
            var g = metal.Geometry{ .grid = .{ @intCast((KT - 1) * width(shape)), 1, 1 }, .tg = .{ 64, 1, 1 } };
            g.out(&.{ 1, KT - 1, @intCast(width(shape)) }, .float32);
            g.int("KT", @as(c_int, KT));
            return g;
        }

        pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
            if (self.geometry(shape) == null) return false;
            self.shape = shape;
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const s: c_int = @intCast(shape.rows);
            self.st = try spread(a, rng, (KT - 1) * width(shape), .f32);
            const widths = [4]usize{ shape.n, shape.n, shape.k, shape.k };
            for (&self.new, widths) |*x, w| x.* = try spread(a, rng, shape.rows * w, elemOf(dt));
            if (fill == .nonfinite) {
                plant(self.st, width(shape));
                // The newest row is always kept.
                for (self.new, widths, [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64) }) |x, w, v| x[(shape.rows - 1) * w + w / 2] = v;
            }
            self.rig.reset();
            self.args[0] = try self.rig.input(.float32, &.{ 1, KT - 1, @intCast(width(shape)) }, self.st);
            for (self.args[1..], self.new, widths) |*arg, x, w| arg.* = try self.rig.input(dt, &.{ 1, s, @intCast(w) }, x);
            return true;
        }

        pub fn run(self: *Self) !void {
            try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
        }

        pub fn count(self: *const Self) usize {
            return (KT - 1) * width(self.shape);
        }

        /// Row `at` of [the old state | the new rows], column `col`.
        fn value(self: *const Self, at: usize, col: usize) f64 {
            const sw = width(self.shape);
            if (at < KT - 1) return self.st[at * sw + col];
            const r = at - (KT - 1);
            const c = self.shape.n;
            const d = self.shape.k;
            if (col < c) return self.new[0][r * c + col];
            if (col < 2 * c) return self.new[1][r * c + col - c];
            if (col < 2 * c + d) return self.new[2][r * d + col - 2 * c];
            return self.new[3][r * d + col - 2 * c - d];
        }

        pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
            const sw = width(self.shape);
            for (want, mag, 0..) |*w, *m, e| {
                w.* = self.value(self.shape.rows + e / sw, e % sw);
                m.* = @abs(w.*);
            }
        }

        fn wrapper(self: *Self) ![]const u8 {
            var g = ops.Graph.init(self.rig.a, self.rig.stream());
            defer g.deinit();
            const x = self.args;
            return self.rig.bytesOf(&.{try kernels.convState(&g, x[0], x[1], x[2], x[3], x[4])});
        }
    };
}

const mask_launch = metal.Launch{ .name = "msv_inkling2_relmask", .inputs = &.{ "rel", "sinks", "sizes" }, .outputs = &.{"M"}, .source = source("plugins/inkling2/kernels/relmask.metal"), .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// The model's query heads and relative extent.
const HQ = 24;
const R = 1024;

fn Mask(comptime dt: mlx.mlx_dtype) type {
    return struct {
        rig: metal.Rig,
        launch: *const metal.Launch = &mask_launch,
        arena: std.heap.ArenaAllocator,
        shape: Shape = undefined,
        /// `[L, HQ, R]` and `[L, HQ]`.
        rel: []f64 = &.{},
        sinks: []f64 = &.{},
        args: [3]mlx.mlx_array = undefined,

        const Self = @This();

        fn init(a: std.mem.Allocator) Self {
            return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
            if (shape.rows == 0 or shape.rows > kernels.small_rows or shape.n < shape.rows) return null;
            const keys: c_int = @intCast(shape.n);
            var g = metal.Geometry{ .grid = .{ @divTrunc(keys + 32, 32) * 32, @intCast(shape.rows), HQ }, .tg = .{ 32, 1, 1 } };
            g.out(&.{ 1, HQ, @intCast(shape.rows), keys + 1 }, dt);
            g.dtype("T", dt);
            g.int("HQ", @as(c_int, HQ));
            g.int("R", @as(c_int, R));
            g.int("WINDOW", shape.k);
            g.int("SINK", @as(c_int, 1));
            return g;
        }

        pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
            if (self.geometry(shape) == null) return false;
            self.shape = shape;
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            self.rel = try spread(a, rng, shape.rows * HQ * R, elemOf(dt));
            self.sinks = try spread(a, rng, shape.rows * HQ, .f32);
            if (fill == .nonfinite) {
                // Distance 0 is visible to every row: head 0's rows carry the classes.
                for ([_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, 0..) |v, row| if (row < shape.rows) {
                    self.rel[row * HQ * R] = v;
                };
                self.sinks[0] = std.math.inf(f64);
                self.sinks[HQ - 1] = std.math.nan(f64);
            }
            self.rig.reset();
            const l: c_int = @intCast(shape.rows);
            self.args = .{
                try self.rig.input(dt, &.{ 1, l, HQ, R }, self.rel),
                try self.rig.input(.float32, &.{ 1, l, HQ }, self.sinks),
                try self.rig.input(.int32, &.{2}, &.{ @floatFromInt(shape.rows), @floatFromInt(shape.n) }),
            };
            return true;
        }

        pub fn run(self: *Self) !void {
            try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
        }

        pub fn count(self: *const Self) usize {
            return HQ * self.shape.rows * (self.shape.n + 1);
        }

        pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
            const l = self.shape.rows;
            const keys = self.shape.n;
            const window = self.shape.k;
            for (want, mag, 0..) |*w, *m, e| {
                const key = e % (keys + 1);
                const row = e / (keys + 1) % l;
                const head = e / (keys + 1) / l;
                w.* = if (key == keys) self.sinks[row * HQ + head] else blk: {
                    // The key's distance behind the row; a key ahead of it, or past the window, is masked.
                    const at = keys - l + row;
                    if (key > at or (window > 0 and at - key >= window)) break :blk -std.math.inf(f64);
                    break :blk if (at - key < R) self.rel[(row * HQ + head) * R + at - key] else 0;
                };
                m.* = if (std.math.isFinite(w.*)) @abs(w.*) else 0;
            }
        }

        fn wrapper(self: *Self) ![]const u8 {
            var g = ops.Graph.init(self.rig.a, self.rig.stream());
            defer g.deinit();
            return self.rig.bytesOf(&.{try kernels.relMask(&g, self.args[0], self.args[1], @intCast(self.shape.n), @intCast(self.shape.k), true)});
        }
    };
}

const act_launch = metal.Launch{ .name = "msv_inkling2_act", .inputs = &.{ "GU", "IDX", "SC", "PR" }, .outputs = &.{"ACT"}, .source = source("plugins/inkling2/kernels/act.metal"), .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// The model's experts and its soft caps on the gate and up arms.
const EXPERTS = 128;
const caps = [4]f64{ 1.0 / 4.0, 4.0, 1.0 / 25.0, 25.0 };

/// The stream's dtype in every pack: bf16.
const Act = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &act_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    /// `[picks, 2 I]`: each pick's gate (even) and up (odd) rows.
    gu: []f64 = &.{},
    idx: []f64 = &.{},
    sc: []f64 = &.{},
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) Act {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    pub fn deinit(self: *Act) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn scaled(shape: Shape) bool {
        return shape.n == 2;
    }

    pub fn geometry(_: *const Act, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.k == 0 or shape.n == 0 or shape.n > 2 or shape.rows * shape.k % 64 != 0) return null;
        var g = metal.Geometry{ .grid = .{ @intCast(shape.rows * shape.k), 1, 1 }, .tg = .{ 64, 1, 1 } };
        g.out(&.{ @intCast(shape.rows), 1, @intCast(shape.k) }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("SCALED", @as(c_int, @intFromBool(scaled(shape))));
        return g;
    }

    pub fn prepare(self: *Act, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // Rows of a normed stream through an expert: tens at most (past 88 the sigmoid's exp leaves fp32).
        self.gu = try a.alloc(f64, shape.rows * 2 * shape.k);
        for (self.gu) |*v| v.* = spec.roundTo(.bf16, (rng.float(f64) * 2 - 1) * std.math.pow(f64, 2, @floatFromInt(rng.intRangeAtMost(i32, -8, 5))));
        self.idx = try a.alloc(f64, shape.rows);
        for (self.idx) |*v| v.* = @floatFromInt(rng.uintLessThan(u32, EXPERTS));
        self.sc = try a.alloc(f64, EXPERTS);
        for (self.sc) |*v| v.* = spec.roundTo(.f32, 0.25 + rng.float(f64));
        if (fill == .nonfinite) plant(self.gu, 2 * shape.k);
        self.rig.reset();
        self.args = .{
            try self.rig.input(.bfloat16, &.{ @intCast(shape.rows), 1, @intCast(2 * shape.k) }, self.gu),
            try self.rig.input(.uint32, &.{@intCast(shape.rows)}, self.idx),
            try self.rig.input(.float32, &.{EXPERTS}, self.sc),
            try self.rig.input(.float32, &.{4}, &caps),
        };
        return true;
    }

    pub fn run(self: *Act) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Act) usize {
        return self.shape.rows * self.shape.k;
    }

    pub fn reference(self: *const Act, want: []f64, mag: []f64) !void {
        const i_n = self.shape.k;
        for (want, mag, 0..) |*w, *m, e| {
            const k = e / i_n;
            const sc: f64 = if (scaled(self.shape)) self.sc[@intFromFloat(self.idx[k])] else 1;
            const gate = self.gu[k * 2 * i_n + 2 * (e % i_n)] * sc;
            const up = self.gu[k * 2 * i_n + 2 * (e % i_n) + 1] * sc;
            w.* = (std.math.tanh(gate * caps[0]) * caps[1]) * (std.math.tanh(up * caps[2]) * caps[3]) / (1 + @exp(-gate));
            m.* = @abs(w.*);
        }
    }

    fn wrapper(self: *Act) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        return self.rig.bytesOf(&.{try kernels.activation(&g, x[0], x[1], x[2], x[3], scaled(self.shape))});
    }
};

const sum_launch = metal.Launch{ .name = "msv_inkling2_sum", .inputs = &.{ "DN", "IDX", "INV", "SC", "WTS", "RES", "ST", "MW", "PR", "PARENT" }, .outputs = &.{ "HOUT", "XM" }, .source = source("plugins/inkling2/kernels/sum.metal"), .tg_bytes = 0, .m1 = 928, .m2 = 928 };

/// The model's picks a row, and the kv width a conv state carries before its two stream-wide halves.
const PICKS = 8;
const KV = 1024;

/// A prompt's rows: `rows` x (1, or 2 with per-expert scales) x hidden, bf16 with bf16 conv taps.
const Sum = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &sum_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    /// `[rows * PICKS, D]` in expert order, `inv[row * PICKS + k]` the place of pair (row, pick) in it.
    dn: []f64 = &.{},
    idx: []f64 = &.{},
    inv: []f64 = &.{},
    sc: []f64 = &.{},
    wts: []f64 = &.{},
    res: []f64 = &.{},
    /// `[KT - 1, 2 KV + 2 D]`: the conv reads its last D columns.
    st: []f64 = &.{},
    mw: []f64 = &.{},
    pr: f64 = 0,
    args: [10]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) Sum {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    pub fn deinit(self: *Sum) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn scaled(shape: Shape) bool {
        return shape.n == 2;
    }

    fn width(shape: Shape) usize {
        return 2 * KV + 2 * shape.k;
    }

    pub fn geometry(_: *const Sum, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.k == 0 or shape.n == 0 or shape.n > 2 or shape.rows * shape.k % 64 != 0) return null;
        const out: []const c_int = &.{ 1, @intCast(shape.rows), @intCast(shape.k) };
        var g = metal.Geometry{ .grid = .{ @intCast(shape.rows * shape.k), 1, 1 }, .tg = .{ 64, 1, 1 } };
        g.out(out, .bfloat16);
        g.out(out, .bfloat16);
        g.dtype("T", .bfloat16);
        g.dtype("WT", .bfloat16);
        g.int("K", @as(c_int, PICKS));
        g.int("KT", @as(c_int, KT));
        g.int("SCALED", @as(c_int, @intFromBool(scaled(shape))));
        g.int("TREE", @as(c_int, 0));
        return g;
    }

    pub fn prepare(self: *Sum, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const d = shape.k;
        const n = shape.rows * PICKS;
        const stress = fill == .stress;
        const ints = struct {
            fn of(al: std.mem.Allocator, r: std.Random, len: usize, most: i32) ![]f64 {
                const out = try al.alloc(f64, len);
                for (out) |*v| v.* = @floatFromInt(r.intRangeAtMost(i32, -most, most));
                return out;
            }
        };
        const unit = struct {
            fn of(al: std.mem.Allocator, r: std.Random, len: usize, e: spec.Elem, lo: f64, hi: f64) ![]f64 {
                const out = try al.alloc(f64, len);
                for (out) |*v| v.* = spec.roundTo(e, (lo + (hi - lo) * r.float(f64)) * (if (r.boolean()) @as(f64, 1) else -1));
                return out;
            }
        };
        // Stress: integers every sum and product of holds exactly in fp32 and in the bf16 it is rounded to.
        self.dn = if (stress) try ints.of(a, rng, n * d, 1) else try unit.of(a, rng, n * d, .bf16, 0.01, 1);
        self.idx = try a.alloc(f64, n);
        for (self.idx) |*v| v.* = @floatFromInt(rng.uintLessThan(u32, EXPERTS));
        // A shuffle: where each (row, pick) pair sits in the expert order.
        self.inv = try a.alloc(f64, n);
        for (self.inv, 0..) |*v, i| v.* = @floatFromInt(i);
        rng.shuffle(f64, self.inv);
        self.sc = try a.alloc(f64, EXPERTS);
        for (self.sc) |*v| v.* = if (stress) 1 else spec.roundTo(.f32, 0.25 + rng.float(f64));
        self.wts = try a.alloc(f64, n);
        for (self.wts) |*v| v.* = if (stress) 1 else spec.roundTo(.f32, 0.05 + 0.3 * rng.float(f64));
        // The stream beside the MLP's output: larger, so a rounding of the output inside it stays under the bound.
        self.res = if (stress) try ints.of(a, rng, shape.rows * d, 64) else try unit.of(a, rng, shape.rows * d, .f32, 16, 64);
        self.st = if (stress) try ints.of(a, rng, (KT - 1) * width(shape), 8) else try unit.of(a, rng, (KT - 1) * width(shape), .f32, 0.01, 1);
        self.mw = if (stress) try ints.of(a, rng, d * KT, 2) else try unit.of(a, rng, d * KT, .bf16, 0.01, 0.5);
        self.pr = if (stress) 0.5 else spec.roundTo(.f32, 1.0 / @sqrt(@as(f64, KT + 1)));
        if (fill == .nonfinite) {
            // A NaN, a +Inf and a -Inf in three rows' picks, both infinities in a fourth's residual and state.
            for ([_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, 0..) |v, row| if (row < shape.rows) {
                self.dn[@as(usize, @intFromFloat(self.inv[row * PICKS])) * d + row] = v;
            };
            self.res[(shape.rows - 1) * d + d - 1] = std.math.inf(f64);
            self.st[2 * KV + d + d / 2] = -std.math.inf(f64);
        }
        self.rig.reset();
        const rows: c_int = @intCast(shape.rows);
        const dd: c_int = @intCast(d);
        self.args = .{
            try self.rig.input(.bfloat16, &.{ @intCast(n), 1, dd }, self.dn),
            try self.rig.input(.uint32, &.{@intCast(n)}, self.idx),
            try self.rig.input(.uint32, &.{@intCast(n)}, self.inv),
            try self.rig.input(.float32, &.{EXPERTS}, self.sc),
            try self.rig.input(.float32, &.{ 1, rows, PICKS }, self.wts),
            try self.rig.input(.float32, &.{ 1, rows, dd }, self.res),
            try self.rig.input(.float32, &.{ 1, KT - 1, @intCast(width(shape)) }, self.st),
            try self.rig.input(.bfloat16, &.{ dd, KT, 1 }, self.mw),
            try self.rig.input(.float32, &.{1}, &.{self.pr}),
            try self.rig.input(.int32, &.{1}, &.{0}),
        };
        return true;
    }

    pub fn run(self: *Sum) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    /// HOUT's elements, then XM's.
    pub fn count(self: *const Sum) usize {
        return 2 * self.shape.rows * self.shape.k;
    }

    /// Row `r`'s pick terms at column `c`, in pick order.
    fn picks(self: *const Sum, r: usize, c: usize, t: *[PICKS]f64) void {
        for (t, 0..) |*v, k| {
            const at: usize = @intFromFloat(self.inv[r * PICKS + k]);
            const sc: f64 = if (scaled(self.shape)) self.sc[@intFromFloat(self.idx[at])] else 1;
            v.* = self.dn[at * self.shape.k + c] * sc * self.wts[r * PICKS + k];
        }
    }

    /// The outputs with the pick sum and the conv sum run by `add` (float64: the reference).
    fn outputs(self: *const Sum, ctx: anytype, out: []f64, mag: ?[]f64) void {
        const d = self.shape.k;
        const rows = self.shape.rows;
        const off = width(self.shape) - d;
        var t: [PICKS]f64 = undefined;
        for (0..rows * d) |e| {
            const s = e / d;
            const c = e % d;
            // Tap j is row `s + j` of [the state | the rows' MLP outputs, rounded as the stream holds them].
            var win: [KT]f64 = undefined;
            var mine: f64 = 0;
            var mine_mag: f64 = 0;
            for (&win, 0..) |*w, j| {
                if (s + j < KT - 1) {
                    w.* = self.st[(s + j) * width(self.shape) + off + c];
                    continue;
                }
                self.picks(s + j - (KT - 1), c, &t);
                const x = ctx.add(&t);
                w.* = spec.roundTo(.bf16, x);
                if (j == KT - 1) {
                    mine = x;
                    for (t) |v| mine_mag += @abs(v);
                }
            }
            var taps: [KT]f64 = undefined;
            for (&taps, win, 0..) |*v, w, j| v.* = w * self.mw[c * KT + j];
            const y = spec.roundTo(.bf16, ctx.add(&taps));
            const inner = spec.roundTo(.bf16, (y + win[KT - 1]) * self.pr);
            out[e] = self.res[e] + inner;
            out[rows * d + e] = mine;
            if (mag) |m| {
                m[e] = @abs(self.res[e]) + @abs(inner);
                m[rows * d + e] = mine_mag;
            }
        }
    }

    pub fn reference(self: *const Sum, want: []f64, mag: []f64) !void {
        const exact = struct {
            fn add(_: @This(), t: []const f64) f64 {
                var x: f64 = 0;
                for (t) |v| x += v;
                return x;
            }
        };
        self.outputs(exact{}, want, mag);
    }

    pub fn simulate(self: *const Sum, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const In = struct {
            acc: spec.Accum,
            order: spec.Order,
            fn add(x: @This(), t: []const f64) f64 {
                return spec.sumIn(x.acc, x.order, t);
            }
        };
        self.outputs(In{ .acc = acc, .order = order }, out, null);
    }

    fn wrapper(self: *Sum) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        const o = try kernels.pickSum(&g, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], PICKS, scaled(self.shape));
        return self.rig.bytesOf(&o);
    }
};

const gate_launch = metal.Launch{ .name = "msv_inkling2_gate", .inputs = &.{ "A", "MS", "ST", "F", "G" }, .outputs = &.{"Y"}, .source = source("plugins/inkling2/kernels/gate.metal"), .tg_bytes = 0, .m1 = 928, .m2 = 928 };

/// The model's gate rank.
const RANK = 64;

/// One row's loop message gates, the threads taking the forget and message projections themselves: gates x 1 x
/// hidden, the gates' states bf16 (the stream's dtype).
const Gate = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &gate_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    /// `[J, D]` each; `st` `[J, RANK]`; `f`, `g` `[J, D, RANK]`.
    a: []f64 = &.{},
    ms: []f64 = &.{},
    st: []f64 = &.{},
    f: []f64 = &.{},
    g: []f64 = &.{},
    args: [5]mlx.mlx_array = undefined,
    /// The forget and message projections' stored dtype: the release's bf16, a converted pack's f32.
    wt: mlx.mlx_dtype,

    fn init(al: std.mem.Allocator, wt: mlx.mlx_dtype) Gate {
        return .{ .rig = metal.Rig.init(al), .arena = .init(al), .wt = wt };
    }

    pub fn deinit(self: *Gate) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const Gate, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n != 1 or shape.k == 0 or shape.rows * shape.k % 64 != 0) return null;
        var geo = metal.Geometry{ .grid = .{ @intCast(shape.rows * shape.k), 1, 1 }, .tg = .{ 64, 1, 1 } };
        geo.out(&.{ @intCast(shape.rows), 1, @intCast(shape.k) }, .float32);
        geo.int("D", shape.k);
        geo.int("RANK", @as(c_int, RANK));
        geo.int("PROJ", @as(c_int, 0));
        return geo;
    }

    pub fn prepare(self: *Gate, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const al = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const j = shape.rows;
        const d = shape.k;
        self.a = try spread(al, rng, j * d, .f32);
        self.ms = try spread(al, rng, j * d, .f32);
        self.st = try al.alloc(f64, j * RANK);
        for (self.st) |*v| v.* = spec.roundTo(.bf16, rng.float(f64) * 2 - 1);
        self.f = try al.alloc(f64, j * d * RANK);
        self.g = try al.alloc(f64, j * d * RANK);
        for ([_][]f64{ self.f, self.g }) |w| for (w) |*v| {
            v.* = spec.roundTo(elemOf(self.wt), 0.2 * (rng.float(f64) * 2 - 1));
        };
        switch (fill) {
            .random => {},
            .nonfinite => {
                plant(self.a, d);
                // An infinite state saturates its gate: the stream leaves, the message stays whole.
                if (j > 1) self.st[RANK + 3] = std.math.inf(f64);
            },
            // Gate 4: states of one, projections 1, then 62 of 2^-12, then -1: fp32 adds every small one, a half
            // or bf16 sum drops those it meets beside the 1. The stream and its message differ, so a sum shows.
            .stress => {
                @memset(self.a, 2);
                @memset(self.ms, 1);
                @memset(self.st, 1);
                for ([_][]f64{ self.f, self.g }) |w| for (w, 0..) |*v, i| {
                    v.* = switch (i % RANK) {
                        0 => 1,
                        RANK - 1 => -1,
                        else => 0x1p-12,
                    };
                };
            },
        }
        self.rig.reset();
        const jj: c_int = @intCast(j);
        const dd: c_int = @intCast(d);
        self.args = .{
            try self.rig.input(.float32, &.{ jj, 1, dd }, self.a),
            try self.rig.input(.float32, &.{ jj, 1, dd }, self.ms),
            try self.rig.input(.bfloat16, &.{ jj, 1, RANK }, self.st),
            try self.rig.input(self.wt, &.{ jj, dd, RANK }, self.f),
            try self.rig.input(self.wt, &.{ jj, dd, RANK }, self.g),
        };
        return true;
    }

    pub fn run(self: *Gate) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Gate) usize {
        return self.shape.rows * self.shape.k;
    }

    /// Element `e`'s forget and message terms.
    fn terms(self: *const Gate, e: usize, tf: *[RANK]f64, tg: *[RANK]f64) void {
        const j = e / self.shape.k;
        for (tf, tg, 0..) |*x, *y, r| {
            x.* = self.f[e * RANK + r] * self.st[j * RANK + r];
            y.* = self.g[e * RANK + r] * self.st[j * RANK + r];
        }
    }

    fn mix(self: *const Gate, e: usize, f: f64, g: f64) f64 {
        return (1 - std.math.tanh(f)) * self.a[e] + std.math.tanh(g) * self.ms[e];
    }

    pub fn reference(self: *const Gate, want: []f64, mag: []f64) !void {
        var tf: [RANK]f64 = undefined;
        var tg: [RANK]f64 = undefined;
        for (want, mag, 0..) |*w, *m, e| {
            self.terms(e, &tf, &tg);
            var f: f64 = 0;
            var g: f64 = 0;
            var af: f64 = 0;
            var ag: f64 = 0;
            for (tf, tg) |x, y| {
                f += x;
                g += y;
                af += @abs(x);
                ag += @abs(y);
            }
            w.* = self.mix(e, f, g);
            // A sum's error reaches the output through a tanh (slope at most one) times what it gates.
            m.* = af * @abs(self.a[e]) + ag * @abs(self.ms[e]) + @abs((1 - std.math.tanh(f)) * self.a[e]) + @abs(std.math.tanh(g) * self.ms[e]);
            if (!std.math.isFinite(m.*)) m.* = 0;
        }
    }

    pub fn simulate(self: *const Gate, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        var tf: [RANK]f64 = undefined;
        var tg: [RANK]f64 = undefined;
        for (out, 0..) |*o, e| {
            self.terms(e, &tf, &tg);
            o.* = self.mix(e, spec.sumIn(acc, order, &tf), spec.sumIn(acc, order, &tg));
        }
    }

    fn wrapper(self: *Gate) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        const gates: layers.Gates = .{ .down = x[2], .forget = x[3], .message = x[4] };
        return self.rig.bytesOf(&.{try kernels.gate(&g, &gates, x[0], x[1], x[2], 0, @intCast(self.shape.rows))});
    }
};

fn rankDepth(_: Shape) usize {
    return RANK;
}

const topk_launch = metal.Launch{ .name = "msv_inkling2_topk", .inputs = &.{ "SC", "RB", "PR" }, .outputs = &.{ "IDX", "WTS" }, .source = source("plugins/inkling2/kernels/topk.metal"), .tg_bytes = 4 * EXPERTS + 8 * PICKS, .m1 = 928, .m2 = 928 };

/// The router's picks: `rows` x PICKS x EXPERTS. A row's indices, then its weights.
const TopK = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &topk_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    sc: []f64 = &.{},
    rb: []f64 = &.{},
    pr: f64 = 0,
    note: [200]u8 = undefined,
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) TopK {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    pub fn deinit(self: *TopK) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const TopK, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n != PICKS or shape.k != EXPERTS) return null;
        const out: []const c_int = &.{ 1, @intCast(shape.rows), PICKS };
        var g = metal.Geometry{ .grid = .{ EXPERTS, @intCast(shape.rows), 1 }, .tg = .{ EXPERTS, 1, 1 } };
        g.out(out, .uint32);
        g.out(out, .float32);
        g.int("E", @as(c_int, EXPERTS));
        g.int("K", @as(c_int, PICKS));
        return g;
    }

    pub fn prepare(self: *TopK, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.sc = try a.alloc(f64, shape.rows * EXPERTS);
        for (self.sc) |*v| v.* = spec.roundTo(.f32, 8 * (rng.float(f64) * 2 - 1));
        self.rb = try a.alloc(f64, EXPERTS);
        for (self.rb) |*v| v.* = spec.roundTo(.f32, 0.2 * (rng.float(f64) * 2 - 1));
        self.pr = spec.roundTo(.f32, 2.5);
        switch (fill) {
            .random => {},
            // A NaN, a +Inf and a -Inf logit in three rows, both infinities in a fourth, and a row with three logits
            // that are numbers at all: its other five slots keep experts 3 to 7 at weight zero.
            .nonfinite => {
                const at = [_][2]usize{ .{ 0, 5 }, .{ 1, 64 }, .{ 2, EXPERTS - 1 }, .{ 3, 0 }, .{ 3, 9 } };
                const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64) };
                for (at, put) |p, v| if (p[0] < shape.rows) {
                    self.sc[p[0] * EXPERTS + p[1]] = v;
                };
                if (shape.rows > 4) for (self.sc[4 * EXPERTS ..][0..EXPERTS], 0..) |*v, e| {
                    if (e % 50 != 7) v.* = std.math.nan(f64);
                };
            },
            // Gate 4: one pick whose sigmoid is 1 and seven near 2^-12, which fp32 adds to it and a half or bf16
            // sum drops; the rest far below. No bias, a scale that is a power of two.
            .stress => {
                @memset(self.rb, 0);
                self.pr = 0.5;
                for (self.sc, 0..) |*v, i| {
                    const e = i % EXPERTS;
                    v.* = if (e == 11) 40 else if (e % 16 == 3 and e < 16 * 7) spec.roundTo(.f32, -@log(4095.0)) else -40;
                }
            },
        }
        self.rig.reset();
        self.args = .{
            try self.rig.input(.float32, &.{ @intCast(shape.rows), EXPERTS }, self.sc),
            try self.rig.input(.float32, &.{EXPERTS}, self.rb),
            try self.rig.input(.float32, &.{1}, &.{self.pr}),
        };
        return true;
    }

    pub fn run(self: *TopK) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const TopK) usize {
        return 2 * self.shape.rows * PICKS;
    }

    fn half(shape: Shape) usize {
        return shape.rows * PICKS;
    }

    fn sigmoid(x: f64) f64 {
        return 1 / (1 + @exp(-x));
    }

    /// Row `r`'s picks by rank in float64 (ties to the lower index; a NaN ranks nowhere), and how many ranked.
    fn rank(self: *const TopK, r: usize, sel: *[PICKS]usize) usize {
        var keys: [EXPERTS]f64 = undefined;
        for (&keys, 0..) |*k, e| k.* = sigmoid(self.sc[r * EXPERTS + e]) + self.rb[e];
        var used: [EXPERTS]bool = @splat(false);
        var got: usize = 0;
        for (sel, 0..) |*s, slot| {
            s.* = slot;
            if (got < slot) continue;
            var best: ?usize = null;
            for (keys, 0..) |k, e| {
                if (std.math.isNan(k) or used[e]) continue;
                if (best == null or k > keys[best.?]) best = e;
            }
            if (best) |e| {
                s.* = e;
                used[e] = true;
                got += 1;
            }
        }
        return got;
    }

    /// The weights of row `r`'s picks, their sum run by `ctx`.
    fn weights(self: *const TopK, r: usize, ctx: anytype, idx: []f64, wts: []f64) void {
        var sel: [PICKS]usize = undefined;
        const ranked = self.rank(r, &sel);
        var raw: [PICKS]f64 = @splat(0);
        for (sel[0..ranked], raw[0..ranked]) |e, *v| v.* = sigmoid(self.sc[r * EXPERTS + e]);
        const total = ctx.add(&raw);
        for (idx, wts, sel, raw) |*i, *w, e, v| {
            i.* = @floatFromInt(e);
            w.* = v / (total + 1e-20) * self.pr;
        }
    }

    pub fn reference(self: *const TopK, want: []f64, mag: []f64) !void {
        const exact = struct {
            fn add(_: @This(), t: []const f64) f64 {
                var x: f64 = 0;
                for (t) |v| x += v;
                return x;
            }
        };
        const n = half(self.shape);
        for (0..self.shape.rows) |r| self.weights(r, exact{}, want[r * PICKS ..][0..PICKS], want[n + r * PICKS ..][0..PICKS]);
        @memcpy(mag, want);
    }

    pub fn simulate(self: *const TopK, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const In = struct {
            acc: spec.Accum,
            order: spec.Order,
            fn add(x: @This(), t: []const f64) f64 {
                return spec.sumIn(x.acc, x.order, t);
            }
        };
        const n = half(self.shape);
        for (0..self.shape.rows) |r| self.weights(r, In{ .acc = acc, .order = order }, out[r * PICKS ..][0..PICKS], out[n + r * PICKS ..][0..PICKS]);
    }

    fn fail(self: *TopK, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&self.note, fmt, args) catch "a pick breaks the rule";
    }

    /// The stated rule on the device's picks: each ranked slot holds an expert whose key is the float64 key of that
    /// rank (two keys fp32 cannot tell apart may swap), never a NaN one; a slot past the ranked keeps its own index
    /// at weight zero; the weights are the picks' sigmoids over their sum.
    pub fn check(self: *TopK, _: spec.Gate) !?[]const u8 {
        const bytes = self.rig.outputBytes();
        const n = half(self.shape);
        for (0..self.shape.rows) |r| {
            var want: [PICKS]usize = undefined;
            const ranked = self.rank(r, &want);
            var total: f64 = 0;
            var sig: [PICKS]f64 = @splat(0);
            for (0..PICKS) |k| {
                const e: usize = @intFromFloat(spec.read(.u32, bytes, r * PICKS + k));
                if (e >= EXPERTS) return self.fail("row {d} slot {d} holds expert {d} of {d}", .{ r, k, e, EXPERTS });
                if (k >= ranked) {
                    if (e != k) return self.fail("row {d} slot {d} past its {d} ranked keys holds {d}", .{ r, k, ranked, e });
                    continue;
                }
                const key = sigmoid(self.sc[r * EXPERTS + e]) + self.rb[e];
                const best = sigmoid(self.sc[r * EXPERTS + want[k]]) + self.rb[want[k]];
                if (!(@abs(key - best) <= 0x1p-21 * @max(1, @abs(best)))) return self.fail("row {d} rank {d}: expert {d} (key {d}) where {d} ranks ({d})", .{ r, k, e, key, want[k], best });
                sig[k] = sigmoid(self.sc[r * EXPERTS + e]);
                total += sig[k];
            }
            for (0..PICKS) |k| {
                const got = spec.read(.f32, bytes[4 * n ..], r * PICKS + k);
                const w = sig[k] / (total + 1e-20) * self.pr;
                // The sum's eight adds, the sigmoids' exp, add and divide, the divide and the scale.
                if (!(@abs(got - w) <= (PICKS + 8) * 0x1p-24 * @abs(w) + 0x1p-149)) return self.fail("row {d} slot {d}: weight {e} where float64 gives {e}", .{ r, k, got, w });
            }
        }
        return null;
    }

    fn wrapper(self: *TopK) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const o = try kernels.topK(&g, self.args[0], self.args[1], self.args[2], PICKS);
        return self.rig.bytesOf(&o);
    }
};

/// The model's head width and query heads a KV head; the split read's keys a threadgroup and a tile.
const SD = 128;
const SG = 3;
const SKPS = 128;
const SBK = 16;
/// The kernels' f32 log2(e).
const LOG2E: f64 = @as(f32, 1.4426950408889634);
/// P rounds to half before P·V (2^-11 of it, 2^-25 at least); q, k and v enter as halves, exact for bf16 values
/// but under half's normal range, where they move by at most 2^-25.
const split_soft = attn.Soft{ .base2 = true, .p_unit = 0x1p-11, .p_abs = 0x1p-25 };
const HALF_SUB = 0x1p-25;
const half_sub_row: [SD]f64 = @splat(HALF_SUB);

const split_launch = metal.Launch{ .name = "msv_inkling2_relsplit", .inputs = &.{ "q", "k", "v", "kb", "vb", "rel", "sinks", "sizes", "scale" }, .outputs = &.{ "PART", "ML" }, .source = source("plugins/inkling2/kernels/relsplit.metal"), .contiguous = false, .tg_bytes = 2 * (SD * (SBK + 8) + SBK * (SD + 8)), .m1 = 1024, .m2 = 1024 };

/// The split read's splits of a few rows behind a cache: rows = KV heads, n = rows read, k = keys; `window` 0 on a
/// global layer. `own`: the rows' keys and values are a second source behind the cache's (a verify's open read).
const RelSplit = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &split_launch,
    arena: std.heap.ArenaAllocator,
    window: usize,
    own: bool,
    shape: Shape = undefined,
    /// `[HQ, L, SD]`, `[HKV, S, SD]` twice, `[L, HQ, R]`, `[L, HQ]`.
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    rel: []f64 = &.{},
    sinks: []f64 = &.{},
    scale: f64 = @as(f32, 1.0 / @sqrt(@as(f32, SD))),
    args: [9]mlx.mlx_array = undefined,

    const Self = @This();

    fn init(a: std.mem.Allocator, window: usize, own: bool) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a), .window = window, .own = own };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn first(self: *const Self, shape: Shape) usize {
        return if (self.window > 0 and shape.k + 1 > shape.n + self.window) shape.k + 1 - shape.n - self.window else 0;
    }

    fn splits(self: *const Self, shape: Shape) usize {
        return (shape.k - self.first(shape) + SKPS - 1) / SKPS;
    }

    pub fn geometry(self: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n < 1 or shape.n > kernels.split_rows or shape.k < shape.n) return null;
        // A second source holds the rows' keys behind at least one of the cache's.
        if (self.own and shape.k == shape.n) return null;
        const ns: c_int = @intCast(self.splits(shape));
        const lh: c_int = @intCast(shape.n * shape.rows * SG);
        var g = metal.Geometry{ .grid = .{ ns * 32 * SG, @intCast(shape.rows), 1 }, .tg = .{ 32 * SG, 1, 1 } };
        g.out(&.{ ns, lh, SD }, .float32);
        g.out(&.{ ns, lh, 2 }, .float32);
        g.dtype("T", .bfloat16);
        g.int("D", @as(c_int, SD));
        g.int("HQ", @as(c_int, @intCast(shape.rows * SG)));
        g.int("HKV", @as(c_int, @intCast(shape.rows)));
        g.int("R", @as(c_int, R));
        g.int("WINDOW", @as(c_int, @intCast(self.window)));
        g.int("KPS", @as(c_int, SKPS));
        return g;
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const hq = shape.rows * SG;
        self.q = try host.uniform(a, rng, hq * shape.n * SD, -2, 2, .bfloat16);
        self.k = try host.uniform(a, rng, shape.rows * shape.k * SD, -2, 2, .bfloat16);
        self.v = try host.uniform(a, rng, shape.rows * shape.k * SD, -2, 2, .bfloat16);
        self.rel = try host.uniform(a, rng, shape.n * hq * R, -2, 2, .bfloat16);
        self.sinks = try host.uniform(a, rng, shape.n * hq, -3, 3, .float32);
        if (fill == .nonfinite) {
            host.plant(self.q, hq * shape.n, SD);
            host.plant(self.v, shape.rows * shape.k, SD);
        }
        self.rig.reset();
        const ci = struct {
            fn f(x: usize) c_int {
                return @intCast(x);
            }
        }.f;
        // The keys before the rows', and the rows' own: one source holds them all and reads the second never.
        const cached = if (self.own) shape.k - shape.n else shape.k;
        var kv: [4]mlx.mlx_array = undefined;
        for ([_][]const f64{ self.k, self.v }, 0..) |all, i| {
            const head = try a.alloc(f64, shape.rows * cached * SD);
            const tail = try a.alloc(f64, shape.rows * (shape.k - cached) * SD);
            for (0..shape.rows) |h| {
                const row = all[h * shape.k * SD ..][0 .. shape.k * SD];
                @memcpy(head[h * cached * SD ..][0 .. cached * SD], row[0 .. cached * SD]);
                @memcpy(tail[h * (shape.k - cached) * SD ..][0 .. (shape.k - cached) * SD], row[cached * SD ..]);
            }
            kv[i] = try self.rig.input(.bfloat16, &.{ 1, ci(shape.rows), ci(cached), SD }, head);
            kv[2 + i] = if (self.own) try self.rig.input(.bfloat16, &.{ 1, ci(shape.rows), ci(shape.k - cached), SD }, tail) else kv[i];
        }
        self.args = .{
            try self.rig.input(.bfloat16, &.{ 1, ci(hq), ci(shape.n), SD }, self.q),
            kv[0],
            kv[1],
            kv[2],
            kv[3],
            try self.rig.input(.bfloat16, &.{ 1, ci(shape.n), ci(hq), R }, self.rel),
            try self.rig.input(.float32, &.{ 1, ci(shape.n), ci(hq) }, self.sinks),
            try self.rig.input(.int32, &.{4}, &.{ @floatFromInt(shape.n), @floatFromInt(shape.k), @floatFromInt(self.first(shape)), @floatFromInt(cached) }),
            try self.rig.scalar(@floatCast(self.scale)),
        };
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return self.splits(self.shape) * self.shape.n * self.shape.rows * SG * (SD + 2);
    }

    fn emit(self: *const Self, want: []f64, mag: ?[]f64, sim: attn.Sim) !void {
        const a = self.rig.a;
        const l = self.shape.n;
        const keys_n = self.shape.k;
        const hq = self.shape.rows * SG;
        const ns = self.splits(self.shape);
        const keys = try a.alloc(attn.Key, SKPS);
        defer a.free(keys);
        const ends = try a.alloc(usize, SKPS);
        defer a.free(ends);
        const terms = try a.alloc(f64, SD);
        defer a.free(terms);
        const eo = try a.alloc(f64, SD);
        defer a.free(eo);
        const qs: f64 = @as(f32, @floatCast(self.scale * LOG2E));
        const parts = ns * l * hq;
        for (0..ns) |s| for (0..l) |r| for (0..hq) |h| {
            const kv = h / SG;
            const kbeg = self.first(self.shape) + s * SKPS;
            const kend = @min(keys_n, kbeg + SKPS);
            const q = self.q[(h * l + r) * SD ..][0..SD];
            var n: usize = 0;
            var nb: usize = 0;
            var t0 = kbeg;
            while (t0 < kend) : (t0 += SBK) {
                for (t0..@min(t0 + SBK, kend)) |j| {
                    // The key's distance behind the row; a key ahead of it, or past the window, is masked, its value
                    // still in the tile's P·V at weight 0.
                    const at = keys_n - l + r;
                    const vrow = self.v[(kv * keys_n + j) * SD ..][0..SD];
                    if (j > at or (self.window > 0 and at - j >= self.window)) {
                        keys[n] = .{ .s = -std.math.inf(f64), .on = false, .v = vrow, .ve = &half_sub_row };
                        n += 1;
                        continue;
                    }
                    const k = self.k[(kv * keys_n + j) * SD ..][0..SD];
                    const d = attn.dot(q, k, qs, SD, 1, sim, terms);
                    var under: f64 = 0;
                    for (q, k) |x, y| under += @abs(x) + @abs(y);
                    const b: f64 = if (at - j < R) @as(f64, @as(f32, @floatCast(self.rel[(r * hq + h) * R + at - j] * LOG2E))) else 0;
                    const sv = d.v + b;
                    keys[n] = .{ .s = sv, .e = d.e + qs * HALF_SUB * under + host.U * (@abs(b) + @abs(sv)), .v = vrow, .ve = &half_sub_row };
                    n += 1;
                }
                ends[nb] = n;
                nb += 1;
            }
            const start: attn.Init = if (s == 0)
                .{ .m = @as(f32, @floatCast(self.sinks[r * hq + h] * LOG2E)), .l = 1 }
            else
                .{ .m = @as(f32, -1e30) };
            const at = (s * l + r) * hq + h;
            const o = want[at * SD ..][0..SD];
            const st = attn.online(split_soft, start, keys[0..n], ends[0..nb], o, eo);
            want[parts * SD + 2 * at] = st.m;
            want[parts * SD + 2 * at + 1] = st.l;
            if (mag) |m| {
                for (m[at * SD ..][0..SD], eo) |*x, e| x.* = host.allowance(e);
                m[parts * SD + 2 * at] = host.allowance(st.em);
                m[parts * SD + 2 * at + 1] = host.allowance(st.el);
            }
        };
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    fn wrapper(self: *Self) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        const parts = try kernels.splitParts(&g, x[0], x[1], x[2], if (self.own) .{ x[3], x[4] } else null, x[5], x[6], @intCast(self.window));
        return self.rig.bytesOf(&parts);
    }
};

const NAX_DREL = 16;
const nax_launch = metal.Launch{ .name = "msv_inkling2_relattn_nax", .inputs = &.{ "q", "k", "v", "rel", "rproj", "sinks", "sizes", "scale" }, .outputs = &.{"out"}, .source = source("plugins/inkling2/kernels/relattn_nax.metal"), .header = source("kernels/common/mpp_tensor_ops.metal"), .contiguous = false, .tg_bytes = 0, .m1 = 1024, .m2 = 1024, .offline = false, .min_gen = 17 };
const NAX_ROWS = 16;
const NAX_BK = 32;

/// A prompt's rows on the tensor unit: rows = query rows, n = KV heads (3 query heads each), k = keys, the rows at the last
/// `rows` of them; `window` 0 on a global layer. Each SIMD group's 16 rows walk 32-key tiles from their first row's window
/// to their last row, the keys a row may not see entering P V at weight 0.
const RelNax = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &nax_launch,
    arena: std.heap.ArenaAllocator,
    dt: mlx.mlx_dtype,
    window: usize,
    shape: Shape = undefined,
    /// `[HQ, L, SD]`, `[HKV, S, SD]` twice, `[L, HQ, NAX_DREL]`, `[R, NAX_DREL]`, `[L, HQ]`.
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    rel: []f64 = &.{},
    rproj: []f64 = &.{},
    sinks: []f64 = &.{},
    scale: f64 = @as(f32, 1.0 / @sqrt(@as(f32, SD))),
    args: [8]mlx.mlx_array = undefined,
    msg: [256]u8 = undefined,

    const Self = @This();

    fn init(a: std.mem.Allocator, dt: mlx.mlx_dtype, window: usize) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a), .dt = dt, .window = window };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n == 0 or shape.k < shape.rows) return null;
        const hq: c_int = @intCast(shape.n * SG);
        var g = metal.Geometry{ .grid = .{ @intCast((shape.rows + 63) / 64 * 128), hq, 1 }, .tg = .{ 128, 1, 1 } };
        g.out(&.{ 1, @intCast(shape.rows), hq, @as(c_int, SD) }, self.dt);
        g.dtype("T", self.dt);
        g.int("D", @as(c_int, SD));
        g.int("HQ", hq);
        g.int("HKV", @as(c_int, @intCast(shape.n)));
        g.int("R", @as(c_int, R));
        g.int("DREL", @as(c_int, NAX_DREL));
        g.int("WINDOW", @as(c_int, @intCast(self.window)));
        return g;
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        if (!@import("../../mlx/device.zig").naxAvailable()) return error.NaxOnly;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const hq = shape.n * SG;
        self.q = try host.uniform(a, rng, hq * shape.rows * SD, -2, 2, self.dt);
        self.k = try host.uniform(a, rng, shape.n * shape.k * SD, -2, 2, self.dt);
        self.v = try host.uniform(a, rng, shape.n * shape.k * SD, -2, 2, self.dt);
        self.rel = try host.uniform(a, rng, shape.rows * hq * NAX_DREL, -1, 1, self.dt);
        self.rproj = try host.uniform(a, rng, R * NAX_DREL, -0.5, 0.5, self.dt);
        self.sinks = try host.uniform(a, rng, shape.rows * hq, -3, 3, .float32);
        if (fill == .nonfinite) {
            host.plant(self.q, hq * shape.rows, SD);
            host.plant(self.v, shape.n * shape.k, SD);
        }
        self.rig.reset();
        const ci = struct {
            fn f(x: usize) c_int {
                return @intCast(x);
            }
        }.f;
        self.args = .{
            try self.rig.input(self.dt, &.{ 1, ci(hq), ci(shape.rows), SD }, self.q),
            try self.rig.input(self.dt, &.{ 1, ci(shape.n), ci(shape.k), SD }, self.k),
            try self.rig.input(self.dt, &.{ 1, ci(shape.n), ci(shape.k), SD }, self.v),
            try self.rig.input(self.dt, &.{ 1, ci(shape.rows), ci(hq), NAX_DREL }, self.rel),
            try self.rig.input(self.dt, &.{ R, NAX_DREL }, self.rproj),
            try self.rig.input(.float32, &.{ 1, ci(shape.rows), ci(hq) }, self.sinks),
            try self.rig.input(.int32, &.{2}, &.{ @floatFromInt(shape.rows), @floatFromInt(shape.k) }),
            try self.rig.scalar(@floatCast(self.scale)),
        };
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return self.shape.rows * self.shape.n * SG * SD;
    }

    fn elem(self: *const Self) spec.Elem {
        return if (self.dt == .bfloat16) .bf16 else .f16;
    }

    /// Float64's outputs and what the device may differ by: the shared online softmax over each row's tiles (P rounded to
    /// the input type before P V, the 128-term f32 score sums, the fast exp), then the output's own rounding.
    fn outputs(self: *const Self, want: []f64, allow: []f64) !void {
        const a = self.rig.a;
        const l = self.shape.rows;
        const keys_n = self.shape.k;
        const hq = self.shape.n * SG;
        const keys = try a.alloc(attn.Key, keys_n + NAX_BK);
        defer a.free(keys);
        const ends = try a.alloc(usize, keys_n / NAX_BK + 2);
        defer a.free(ends);
        const terms = try a.alloc(f64, SD);
        defer a.free(terms);
        const o = try a.alloc(f64, SD);
        defer a.free(o);
        const eo = try a.alloc(f64, SD);
        defer a.free(eo);
        const qs: f64 = @as(f32, @floatCast(self.scale * LOG2E));
        const wide = self.dt == .bfloat16;
        const soft = attn.Soft{ .base2 = true, .p_unit = if (wide) 0x1p-8 else 0x1p-11, .p_abs = if (wide) 0 else 0x1p-25 };
        const out_unit: f64 = if (wide) 0x1p-8 else 0x1p-11;
        for (0..l) |r| for (0..hq) |h| {
            const kv = h / SG;
            const at = keys_n - l + r;
            // The tile this row's SIMD group walks: its first row's window start to its last row's position.
            const g0 = r / NAX_ROWS * NAX_ROWS;
            const d0 = keys_n - l + g0;
            const d_last = keys_n - l + @min(g0 + NAX_ROWS, l) - 1;
            const k_begin: usize = if (self.window > 0 and d0 + 1 > self.window) (d0 + 1 - self.window) / NAX_BK * NAX_BK else 0;
            const q = self.q[(h * l + r) * SD ..][0..SD];
            var n: usize = 0;
            var nb: usize = 0;
            var t0 = k_begin;
            while (t0 <= d_last) : (t0 += NAX_BK) {
                for (t0..@min(t0 + NAX_BK, keys_n)) |j| {
                    const vrow = self.v[(kv * keys_n + j) * SD ..][0..SD];
                    if (j > at or (self.window > 0 and at - j >= self.window)) {
                        keys[n] = .{ .s = -std.math.inf(f64), .on = false, .v = vrow };
                        n += 1;
                        continue;
                    }
                    const k = self.k[(kv * keys_n + j) * SD ..][0..SD];
                    const d = attn.dot(q, k, qs, SD, 1, null, terms);
                    var b: f64 = 0;
                    var bmag: f64 = 0;
                    if (at - j < R) {
                        const rs = self.rel[(r * hq + h) * NAX_DREL ..][0..NAX_DREL];
                        const rp = self.rproj[(at - j) * NAX_DREL ..][0..NAX_DREL];
                        for (rs, rp) |rx, ry| {
                            const t = rx * ry;
                            b += t;
                            bmag += @abs(t);
                        }
                        b *= LOG2E;
                        bmag *= LOG2E;
                    }
                    const sv = d.v + b;
                    keys[n] = .{ .s = sv, .e = d.e + host.U * (@as(f64, NAX_DREL + 2) * bmag + @abs(sv)), .v = vrow };
                    n += 1;
                }
                ends[nb] = n;
                nb += 1;
            }
            const row = attn.online(soft, .{ .m = @as(f32, @floatCast(self.sinks[r * hq + h] * LOG2E)), .l = 1 }, keys[0..n], ends[0..nb], o, eo);
            attn.normalize(row, o, eo);
            for (o, eo, 0..) |x, e, c| {
                const i = (r * hq + h) * SD + c;
                want[i] = x;
                allow[i] = e + out_unit * @abs(x);
            }
        };
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        try self.outputs(want, mag);
    }

    /// The rule: within the allowance of `outputs`, in the float64 output's class (a planted NaN or infinity reaches every
    /// row whose tiles hold it, a masked key's value included).
    pub fn check(self: *Self, at: spec.Gate) !?[]const u8 {
        const a = self.rig.a;
        const n = self.count();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        try self.outputs(buf[0..n], buf[n .. 2 * n]);
        for (buf[2 * n ..], 0..) |*g, i| g.* = spec.read(self.elem(), self.rig.outputBytes(), i);
        return host.within(&self.msg, at, buf[2 * n ..], buf[0..n], buf[n .. 2 * n]);
    }

    fn wrapper(self: *Self) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        const out = [_]mlx.mlx_array{try kernels.naxRead(&g, x[0], x[1], x[2], x[3], x[4], x[5], @intCast(self.window))};
        return self.rig.bytesOf(&out);
    }
};

const slots_launch = metal.Launch{
    .name = "msv_inkling2_relslots",
    .inputs = &.{ "q", "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "rel", "sinks", "KEYS", "scale" },
    .outputs = &.{ "PART", "ML" },
    .source = source("plugins/inkling2/kernels/relslots.metal"),
    .contiguous = false,
    // sm, sl [NSG][G] and so [NSG][G][D] f32 at 8 groups of 3 heads.
    .tg_bytes = 4 * (2 * 8 * SG + 8 * SG * SD),
    .m1 = 1024,
    .m2 = 1024,
};
/// The KV heads, the kernel's keys a threadgroup and SIMD groups.
const SHKV = 8;
const SLOT_KB = 256;
const SLOT_NSG = 8;

/// Batched slots' one-row reads, each behind its own cache: rows = slots (up to 8), n = slot 0's keys (slot i holds
/// 37 i more), k = the window (0 on a global layer). PART [splits, slots * HQ, SD], then ML.
const RelSlots = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &slots_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    /// `[NS, HQ, SD]`, `[NS, HQ, R]`, `[NS, HQ]`; slot i's K and V `[HKV, keys_i, SD]`.
    q: []f64 = &.{},
    rel: []f64 = &.{},
    sinks: []f64 = &.{},
    k: [8][]f64 = undefined,
    v: [8][]f64 = undefined,
    scale: f64 = @as(f32, 1.0 / @sqrt(@as(f32, SD))),
    owned: [16]mlx.mlx_array = @splat(.{ .ctx = null }),
    args: [21]mlx.mlx_array = undefined,

    const Self = @This();
    const HQS = SHKV * SG;

    fn init(a: std.mem.Allocator) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    fn freeOwned(self: *Self) void {
        for (&self.owned) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
    }

    pub fn deinit(self: *Self) void {
        self.freeOwned();
        self.rig.deinit();
        self.arena.deinit();
    }

    fn keys(shape: Shape, slot: usize) usize {
        return shape.n + 37 * slot;
    }

    fn first(shape: Shape, slot: usize) usize {
        const n = keys(shape, slot);
        return if (shape.k > 0 and n > shape.k) n - shape.k else 0;
    }

    fn splits(shape: Shape) usize {
        var most: usize = 1;
        for (0..shape.rows) |i| most = @max(most, (keys(shape, i) - first(shape, i) + SLOT_KB - 1) / SLOT_KB);
        return most;
    }

    pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.rows > 8 or shape.n == 0) return null;
        const ns = cint(splits(shape));
        const lh = cint(shape.rows * HQS);
        var g = metal.Geometry{ .grid = .{ ns * 32 * SLOT_NSG, SHKV, cint(shape.rows) }, .tg = .{ 32 * SLOT_NSG, 1, 1 } };
        g.out(&.{ ns, lh, SD }, .float32);
        g.out(&.{ ns, lh, 2 }, .float32);
        g.dtype("T", .bfloat16);
        g.int("D", @as(c_int, SD));
        g.int("HQ", @as(c_int, HQS));
        g.int("HKV", @as(c_int, SHKV));
        g.int("R", @as(c_int, R));
        g.int("NS", cint(shape.rows));
        g.int("NSG", @as(c_int, SLOT_NSG));
        g.int("KB", @as(c_int, SLOT_KB));
        g.int("WINDOW", cint(shape.k));
        return g;
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        self.freeOwned();
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const ns = shape.rows;
        self.q = try host.uniform(a, rng, ns * HQS * SD, -0.5, 0.5, .bfloat16);
        self.rel = try host.uniform(a, rng, ns * HQS * R, -2, 2, .bfloat16);
        self.sinks = try host.uniform(a, rng, ns * HQS, -3, 3, .float32);
        for (0..ns) |i| {
            self.k[i] = try host.uniform(a, rng, SHKV * keys(shape, i) * SD, -2, 2, .bfloat16);
            self.v[i] = try host.uniform(a, rng, SHKV * keys(shape, i) * SD, -2, 2, .bfloat16);
        }
        if (fill == .nonfinite) {
            host.plant(self.q, ns * HQS, SD);
            host.plant(self.v[0], SHKV * keys(shape, 0), SD);
        }
        self.rig.reset();
        // Each slot's K and V a view of the first keys of a longer buffer, as a cache holds them.
        var pad = std.Random.DefaultPrng.init(91);
        var views: [16]mlx.mlx_array = undefined;
        for (0..ns) |i| for ([_][]const f64{ self.k[i], self.v[i] }, 0..) |src, kvi| {
            const n = keys(shape, i);
            const cap = n + 24;
            const buf = try host.uniform(a, pad.random(), SHKV * cap * SD, -1, 1, .bfloat16);
            for (0..SHKV) |h| @memcpy(buf[h * cap * SD ..][0 .. n * SD], src[h * n * SD ..][0 .. n * SD]);
            const full = try self.rig.input(.bfloat16, &.{ 1, SHKV, cint(cap), SD }, buf);
            const at = 2 * i + kvi;
            self.owned[at] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&self.owned[at], full, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, SHKV, cint(n), SD }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, self.rig.stream()));
            views[at] = self.owned[at];
        };
        var lens: [8]f64 = undefined;
        for (0..ns) |i| lens[i] = @floatFromInt(keys(shape, i));
        self.args[0] = try self.rig.input(.bfloat16, &.{ 1, cint(ns), HQS, SD }, self.q);
        for (0..8) |j| {
            const i = if (j < ns) j else 0;
            self.args[1 + j] = views[2 * i];
            self.args[9 + j] = views[2 * i + 1];
        }
        self.args[17] = try self.rig.input(.bfloat16, &.{ 1, cint(ns), HQS, R }, self.rel);
        self.args[18] = try self.rig.input(.float32, &.{ 1, cint(ns), HQS }, self.sinks);
        self.args[19] = try self.rig.input(.int32, &.{cint(ns)}, lens[0..ns]);
        self.args[20] = try self.rig.scalar(@floatCast(self.scale));
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return splits(self.shape) * self.shape.rows * HQS * (SD + 2);
    }

    fn emit(self: *const Self, want: []f64, mag: ?[]f64, sim: attn.Sim) !void {
        const a = self.rig.a;
        const ns = self.shape.rows;
        const nsp = splits(self.shape);
        const parts = nsp * ns * HQS;
        const qs: f64 = @as(f32, @floatCast(self.scale * LOG2E));
        const keys_buf = try a.alloc(attn.Key, SLOT_KB);
        defer a.free(keys_buf);
        const ends = try a.alloc(usize, SLOT_KB);
        defer a.free(ends);
        const terms = try a.alloc(f64, SD);
        defer a.free(terms);
        var qv: [SD]f64 = undefined;
        var so: [SLOT_NSG][SD]f64 = undefined;
        var se: [SLOT_NSG][SD]f64 = undefined;
        var rows: [SLOT_NSG]attn.Row = undefined;
        for (0..nsp) |sp| for (0..ns) |slot| for (0..HQS) |h| {
            const kv = h / SG;
            const n = keys(self.shape, slot);
            const kbeg = first(self.shape, slot) + sp * SLOT_KB;
            const kend = @min(n, kbeg + SLOT_KB);
            // As the kernel: q times the f32 scale, rounded to f32, a lane's share.
            for (&qv, self.q[(slot * HQS + h) * SD ..][0..SD]) |*x, y| x.* = @as(f32, @floatCast(y * qs));
            for (0..SLOT_NSG) |t| {
                var m: usize = 0;
                var j = kbeg + t;
                while (j < kend) : (j += SLOT_NSG) {
                    const krow = self.k[slot][(kv * n + j) * SD ..][0..SD];
                    const d = attn.dot(&qv, krow, 1, SD, 0, sim, terms);
                    const dist = n - 1 - j;
                    const b: f64 = if (dist < R) @as(f32, @floatCast(self.rel[(slot * HQS + h) * R + dist] * LOG2E)) else 0;
                    const sv = d.v + b;
                    keys_buf[m] = .{ .s = sv, .e = d.e + host.U * (@abs(b) + @abs(sv)), .v = self.v[slot][(kv * n + j) * SD ..][0..SD] };
                    ends[m] = m + 1;
                    m += 1;
                }
                const start: attn.Init = if (sp == 0 and t == 0) .{ .m = @as(f32, @floatCast(self.sinks[slot * HQS + h] * LOG2E)), .l = 1 } else .{ .m = @as(f32, -1e30) };
                rows[t] = attn.online(.{ .base2 = true }, start, keys_buf[0..m], ends[0..m], &so[t], &se[t]);
            }
            // The groups joined as the kernel's last group does: weights exp2(m_t - max), a fast exp each.
            var mx = -std.math.inf(f64);
            for (rows) |r| mx = @max(mx, r.m);
            var ls: f64 = 0;
            var el: f64 = 0;
            const at = (sp * ns + slot) * HQS + h;
            const o = want[at * SD ..][0..SD];
            @memset(o, 0);
            var eo: [SD]f64 = @splat(0);
            const join = spec.gamma(SLOT_NSG + 1, host.U);
            for (rows, so, se) |r, ot, et| {
                const w = @exp2(r.m - mx);
                const ew = host.fastExp((mx - r.m) * std.math.ln2) + join;
                ls += w * r.l;
                el += w * (r.el + r.l * ew);
                for (o, &eo, ot, et) |*x, *e, v, ev| {
                    x.* += w * v;
                    e.* += w * (ev + @abs(v) * ew);
                }
            }
            want[parts * SD + 2 * at] = mx;
            want[parts * SD + 2 * at + 1] = ls;
            if (mag) |mg| {
                var em: f64 = 0;
                for (rows) |r| em = @max(em, r.em);
                for (mg[at * SD ..][0..SD], eo) |*x, e| x.* = if (std.math.isFinite(e)) host.allowance(e) else 0;
                mg[parts * SD + 2 * at] = host.allowance(em);
                mg[parts * SD + 2 * at + 1] = if (std.math.isFinite(el)) host.allowance(el) else 0;
            }
        };
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    fn wrapper(self: *Self) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        const ns = self.shape.rows;
        const parts = try kernels.slotParts(&g, x[0], x[1 .. 1 + ns], x[9 .. 9 + ns], x[17], x[18], @intCast(self.shape.k));
        return self.rig.bytesOf(&parts);
    }
};

const comb_launch = metal.Launch{ .name = "msv_inkling2_relcomb", .inputs = &.{ "PART", "ML", "GATE" }, .outputs = &.{"OUT"}, .source = source("plugins/inkling2/kernels/relcomb.metal"), .tg_bytes = 4 * 128, .m1 = 1024, .m2 = 1024 };

/// The split read's join: rows = rows read (24 query heads each), n = splits, k = head width.
const RelComb = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &comb_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    /// `[NS, LH, D]`, `[NS, LH]` twice, `[LH, D]`.
    o: []f64 = &.{},
    m: []f64 = &.{},
    l: []f64 = &.{},
    gate: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    const Self = @This();

    fn init(a: std.mem.Allocator) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn lh(shape: Shape) usize {
        return shape.rows * HQ;
    }

    pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 4 != 0) return null;
        var g = metal.Geometry{ .grid = .{ 32, @intCast(lh(shape)), 1 }, .tg = .{ 32, 1, 1 } };
        g.out(&.{ 1, @intCast(shape.rows), @intCast(HQ * shape.k) }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("D", @as(c_int, @intCast(shape.k)));
        return g;
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = lh(shape);
        const ns = shape.n;
        const d = shape.k;
        self.m = try host.uniform(a, rng, ns * rows, -3, 3, .float32);
        self.l = try host.uniform(a, rng, ns * rows, 1, 20, .float32);
        self.o = try host.uniform(a, rng, ns * rows * d, -20, 20, .float32);
        self.gate = try host.uniform(a, rng, rows * d, 0, 1, .float32);
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.o, ns * rows, d),
            // Gate 4: maxes 0, sums 1, gate 1; the outputs' first half cancels the second but for small integers, so
            // the true sum is small while its partial sums leave fp16's range.
            .stress => {
                @memset(self.m, 0);
                @memset(self.l, 1);
                @memset(self.gate, 1);
                for (0..ns) |s| for (0..rows * d) |i| {
                    const at = s * rows * d + i;
                    self.o[at] = if (s < ns / 2) @floatFromInt(rng.intRangeAtMost(i32, 20000, 40000)) else if (s < 2 * (ns / 2)) -self.o[at - (ns / 2) * rows * d] + @as(f64, @floatFromInt(rng.intRangeAtMost(i32, 0, 3))) else 1;
                };
            },
        }
        if (fill != .stress) for (0..rows) |r| if (r % 3 == 1 and ns > 1) {
            // A split past a row's visible keys: max at the kernel's floor, sum and output zero.
            self.m[(ns - 1) * rows + r] = @as(f32, -1e30);
            self.l[(ns - 1) * rows + r] = 0;
            @memset(self.o[((ns - 1) * rows + r) * d ..][0..d], 0);
        };
        const ml = try a.alloc(f64, 2 * ns * rows);
        for (0..ns * rows) |i| {
            ml[2 * i] = self.m[i];
            ml[2 * i + 1] = self.l[i];
        }
        self.rig.reset();
        const n: c_int = @intCast(ns);
        const lr: c_int = @intCast(rows);
        self.args = .{
            try self.rig.input(.float32, &.{ n, lr, @intCast(d) }, self.o),
            try self.rig.input(.float32, &.{ n, lr, 2 }, ml),
            try self.rig.input(.float32, &.{@intCast(rows * d)}, self.gate),
        };
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return lh(self.shape) * self.shape.k;
    }

    fn emit(self: *const Self, want: []f64, mag: ?[]f64, sim: attn.Sim) !void {
        const rows = lh(self.shape);
        const ns = self.shape.n;
        const d = self.shape.k;
        const t = try self.rig.a.alloc(f64, 2 * ns);
        defer self.rig.a.free(t);
        // A precise exp2 (one f32 ulp) a weight, the f32 sums, the divide.
        const sums = spec.gamma(@floatFromInt(ns + 1), host.U);
        for (0..rows) |r| {
            var mx = -std.math.inf(f64);
            for (0..ns) |s| mx = @max(mx, self.m[s * rows + r]);
            for (0..d) |i| {
                var num: f64 = 0;
                var den: f64 = 0;
                var en: f64 = 0;
                var ed: f64 = 0;
                for (0..ns) |s| {
                    const w = @exp2(self.m[s * rows + r] - mx);
                    const x = self.o[(s * rows + r) * d + i];
                    t[s] = w * x;
                    t[ns + s] = w * self.l[s * rows + r];
                    num += t[s];
                    den += t[ns + s];
                    en += @abs(t[s]) * (0x1p-23 + sums);
                    ed += @abs(t[ns + s]) * (0x1p-23 + sums);
                }
                if (sim) |x| {
                    num = spec.sumIn(x.acc, x.order, t[0..ns]);
                    den = spec.sumIn(x.acc, x.order, t[ns..]);
                }
                const read = num / den;
                const g = self.gate[r * d + i];
                want[r * d + i] = read * g;
                if (mag) |m| {
                    const err = (en + @abs(read) * ed) / den + host.U * @abs(read);
                    m[r * d + i] = if (std.math.isFinite(err)) host.allowance(err * g) else 0;
                }
            }
        }
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    fn wrapper(self: *Self) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        return self.rig.bytesOf(&.{try kernels.splitJoin(&g, .{ x[0], x[1] }, x[2], @intCast(self.shape.rows), .bfloat16)});
    }
};

// ── the drafter's Markov step ──

/// The block's steps, the previous-token table's rows a test binds (one is read), f32 unit roundoff of a precise
/// tanh / exp (a few ulps).
const MSTEPS = 7;
const MPREV = 64;
const PRECISE = 4 * 0x1p-24;
const NONE_IDX: f64 = 0xffffffff;
/// The release's logit soft cap.
const MCAP = 30.0;

/// `x` under the soft cap, as the kernels' precise tanh gives it.
fn capped(x: f64) f64 {
    return MCAP * std.math.tanh(x / MCAP);
}

fn markovLaunch(comptime file: [:0]const u8, comptime ins: []const [*:0]const u8, comptime outs: []const [*:0]const u8, comptime header: bool, comptime tg_bytes: u32) metal.Launch {
    return .{ .name = "msv_inkling2_" ++ file, .inputs = ins, .outputs = outs, .source = source("plugins/inkling2/kernels/" ++ file ++ ".metal"), .header = if (header) source("plugins/inkling2/kernels/markov_header.metal") else "", .tg_bytes = tg_bytes, .m1 = 1024, .m2 = 1024 };
}

const mk_rnn_launch = markovLaunch("markov_rnn", &.{ "W", "B", "ST", "PREV", "ROWS", "NEW", "TOK", "AT" }, &.{"P"}, false, 0);
const mk_fold_launch = markovLaunch("markov_fold", &.{ "P", "FOLD", "FB", "BASE", "AT", "CAP" }, &.{ "PM", "PI", "PS" }, true, 4 * (256 + 3 * 8));
const mk_pick_launch = markovLaunch("markov_pick", &.{ "PM", "PI", "PS", "P", "ST", "CAP" }, &.{ "TOKN", "CONF", "NST" }, true, 4 * 3 * 8);

fn cint(x: usize) c_int {
    return @intCast(x);
}

/// `x` rounded to bf16 where the device holds it within `e` of it.
fn bf(x: host.Val) host.Val {
    return host.round(.bfloat16, x);
}

/// MLX's argmax order (markov_header.metal): a NaN first, the lower index among NaNs and equal values.
fn better(va: f64, ia: usize, vb: f64, ib: usize) bool {
    if (std.math.isNan(va)) return !std.math.isNan(vb) or ia < ib;
    return !std.math.isNan(vb) and (va > vb or (va == vb and ia < ib));
}

/// A step's RNN projection: n = rank, k = hidden (rows unused: one step). `P [3 R]` f32 holding T(T(W . fed) + B).
const MarkovRnn = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &mk_rnn_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    w: []f64 = &.{},
    b: []f64 = &.{},
    st: []f64 = &.{},
    prev: []f64 = &.{},
    rows: []f64 = &.{},
    new: []f64 = &.{},
    tok: usize = 0,
    at: usize = 0,
    args: [8]mlx.mlx_array = undefined,

    const Self = @This();

    fn init(a: std.mem.Allocator) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn in(shape: Shape) usize {
        return 2 * shape.n + 2 * shape.k;
    }

    pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.n == 0 or shape.k == 0 or 3 * shape.n * 32 % 256 != 0) return null;
        var g = metal.Geometry{ .grid = .{ cint(32 * 3 * shape.n), 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{cint(3 * shape.n)}, .float32);
        g.dtype("T", .bfloat16);
        g.int("R", cint(shape.n));
        g.int("H", cint(shape.k));
        return g;
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.n;
        const h = shape.k;
        self.w = try host.uniform(a, rng, 3 * r * in(shape), -0.05, 0.05, .bfloat16);
        self.b = try host.uniform(a, rng, 3 * r, -0.2, 0.2, .bfloat16);
        self.st = try host.uniform(a, rng, r, -1, 1, .bfloat16);
        self.prev = try host.uniform(a, rng, MPREV * r, -0.5, 0.5, .bfloat16);
        self.rows = try host.uniform(a, rng, MSTEPS * h, -1, 1, .bfloat16);
        self.new = try host.uniform(a, rng, h, -1, 1, .bfloat16);
        self.tok = rng.uintLessThan(usize, MPREV);
        self.at = rng.uintLessThan(usize, MSTEPS);
        if (fill == .nonfinite) host.plant(self.w, 3 * r, in(shape));
        self.rig.reset();
        self.args = .{
            try self.rig.input(.bfloat16, &.{ cint(3 * r), cint(in(shape)) }, self.w),
            try self.rig.input(.bfloat16, &.{cint(3 * r)}, self.b),
            try self.rig.input(.bfloat16, &.{ 1, 1, cint(r) }, self.st),
            try self.rig.input(.bfloat16, &.{ MPREV, cint(r) }, self.prev),
            try self.rig.input(.bfloat16, &.{ 1, MSTEPS, cint(h) }, self.rows),
            try self.rig.input(.bfloat16, &.{ 1, 1, cint(h) }, self.new),
            try self.rig.input(.int32, &.{1}, &.{@floatFromInt(self.tok)}),
            try self.rig.input(.int32, &.{1}, &.{@floatFromInt(self.at)}),
        };
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return 3 * self.shape.n;
    }

    fn fed(self: *const Self, c: usize) f64 {
        const r = self.shape.n;
        const h = self.shape.k;
        if (c < r) return self.st[c];
        if (c < 2 * r) return self.prev[self.tok * r + c - r];
        if (c < 2 * r + h) return self.rows[self.at * h + c - 2 * r];
        return self.new[c - 2 * r - h];
    }

    fn emit(self: *const Self, want: []f64, mag: ?[]f64, sim: attn.Sim) !void {
        const n = in(self.shape);
        const t = try self.rig.a.alloc(f64, n);
        defer self.rig.a.free(t);
        for (want, 0..) |*w, row| {
            var abs: f64 = 0;
            for (t, 0..) |*x, c| {
                x.* = self.w[row * n + c] * self.fed(c);
                abs += @abs(x.*);
            }
            const s = if (sim) |x| spec.sumIn(x.acc, x.order, t) else host.sum(t);
            const r1 = bf(.{ .v = s, .e = spec.gamma(@floatFromInt(n), host.U) * abs });
            const r2 = bf(.{ .v = r1.v + self.b[row], .e = r1.e });
            w.* = r2.v;
            if (mag) |m| m[row] = if (std.math.isFinite(r2.e)) host.allowance(r2.e) else 0;
        }
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    fn wrapper(self: *Self) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        return self.rig.bytesOf(&.{try kernels.markovRnn(&g, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])});
    }
};

/// The fold's columns a threadgroup (markov_fold's TG x VPT).
const MCOLS = 256 * 8;

fn foldGroups(shape: Shape) usize {
    return (shape.k + MCOLS - 1) / MCOLS;
}

/// A step's latent bias and its threadgroups' share of the pick: rows = rank, n = the fold's columns, k = the
/// vocab's real ones. PM, PI, PS a threadgroup.
const MarkovFold = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &mk_fold_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    p: []f64 = &.{},
    fold: []f64 = &.{},
    fb: []f64 = &.{},
    base: []f64 = &.{},
    at: usize = 0,
    /// Each real column's logit and its error.
    logit: []host.Val = &.{},
    note: [200]u8 = undefined,
    args: [6]mlx.mlx_array = undefined,

    const Self = @This();

    fn init(a: std.mem.Allocator) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.k == 0 or shape.k > shape.n or shape.n % 8 != 0) return null;
        const ng = cint(foldGroups(shape));
        var g = metal.Geometry{ .grid = .{ ng * 256, 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ng}, .float32);
        g.out(&.{ng}, .uint32);
        g.out(&.{ng}, .float32);
        g.dtype("T", .bfloat16);
        g.int("R", cint(shape.rows));
        g.int("TG", @as(c_int, 256));
        g.int("VPT", @as(c_int, 8));
        g.int("VOCAB", cint(shape.k));
        return g;
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.rows;
        const v = shape.n;
        self.p = try host.uniform(a, rng, 3 * r, -2, 2, .float32);
        self.fold = try host.uniform(a, rng, r * v, -0.3, 0.3, .bfloat16);
        self.fb = try host.uniform(a, rng, v, -0.5, 0.5, .bfloat16);
        self.base = try host.uniform(a, rng, MSTEPS * v, -8, 8, .float32);
        self.at = rng.uintLessThan(usize, MSTEPS);
        const row = self.base[self.at * v ..][0..v];
        switch (fill) {
            .random => {},
            // A NaN in the first group, +Inf in the second, -Inf in the third.
            .nonfinite => for ([_]usize{ 5, MCOLS + 9, 2 * MCOLS + 1 }, [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }) |c, x| {
                if (c < shape.k) row[c] = x;
            },
            // Gate 4: one column a group stands 40 above the rest, so the pick has no near tie.
            .stress => for (0..foldGroups(shape)) |g| {
                row[@min(shape.k - 1, g * MCOLS + rng.uintLessThan(usize, MCOLS))] = 48;
            },
        }
        self.logit = try a.alloc(host.Val, shape.k);
        self.rig.reset();
        self.args = .{
            try self.rig.input(.float32, &.{cint(3 * r)}, self.p),
            try self.rig.input(.bfloat16, &.{ cint(r), cint(v) }, self.fold),
            try self.rig.input(.bfloat16, &.{ 1, 1, cint(v) }, self.fb),
            try self.rig.input(.float32, &.{ 1, MSTEPS, cint(v) }, self.base),
            try self.rig.input(.int32, &.{1}, &.{@floatFromInt(self.at)}),
            try self.rig.input(.float32, &.{1}, &.{MCAP}),
        };
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return 3 * foldGroups(self.shape);
    }

    /// Each real column's logit as the kernel rounds it, the latent's sums in `sim`'s accumulator.
    fn logits(self: *Self, sim: attn.Sim) !void {
        const r = self.shape.rows;
        const v = self.shape.n;
        const a = self.rig.a;
        const lat = try a.alloc(host.Val, r);
        defer a.free(lat);
        for (lat, 0..) |*l, i| {
            const t = std.math.tanh(self.p[2 * r + i]);
            l.* = bf(.{ .v = t, .e = PRECISE * @abs(t) });
        }
        const t = try a.alloc(f64, r);
        defer a.free(t);
        for (self.logit, 0..) |*o, c| {
            var abs: f64 = 0;
            var flips: f64 = 0;
            for (t, lat, 0..) |*x, l, i| {
                x.* = l.v * self.fold[i * v + c];
                abs += @abs(x.*);
                flips += l.e * @abs(self.fold[i * v + c]);
            }
            const s = if (sim) |x| spec.sumIn(x.acc, x.order, t) else host.sum(t);
            const r1 = bf(.{ .v = s, .e = spec.gamma(@floatFromInt(r), host.U) * abs + flips });
            const r2 = bf(.{ .v = r1.v + self.fb[c], .e = r1.e });
            const x = self.base[self.at * v + c] + r2.v;
            o.* = .{ .v = x, .e = r2.e + host.U * @abs(x) };
        }
    }

    /// Group `g`'s argmax column in MLX's order, its largest error, and its sum of exp(logit - the max).
    fn group(self: *const Self, g: usize) struct { at: usize, e: f64, sum: f64 } {
        const lo = g * MCOLS;
        const hi = @min(self.shape.k, lo + MCOLS);
        var best = lo;
        var e: f64 = 0;
        for (lo..hi) |c| {
            if (better(self.logit[c].v, c, self.logit[best].v, best)) best = c;
            if (std.math.isFinite(self.logit[c].e)) e = @max(e, self.logit[c].e);
        }
        var sum: f64 = 0;
        for (lo..hi) |c| sum += @exp(capped(self.logit[c].v) - capped(self.logit[best].v));
        return .{ .at = best, .e = e, .sum = sum };
    }

    fn emit(self: *Self, want: []f64, mag: ?[]f64, sim: attn.Sim) !void {
        try self.logits(sim);
        const ng = foldGroups(self.shape);
        for (0..ng) |g| {
            const x = self.group(g);
            want[g] = self.logit[x.at].v;
            want[ng + g] = @floatFromInt(x.at);
            want[2 * ng + g] = x.sum;
            if (mag) |m| {
                m[g] = host.allowance(x.e);
                m[ng + g] = 0;
                // Every term moves by its own and the max's error, and a precise exp's.
                m[2 * ng + g] = host.allowance(x.sum * (2 * x.e + PRECISE + spec.gamma(@floatFromInt(MCOLS), host.U)));
            }
        }
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        try @constCast(self).emit(want, mag, null);
    }

    pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try @constCast(self).emit(out, null, .{ .acc = acc, .order = order });
    }

    fn fail(self: *Self, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&self.note, fmt, args) catch "a group breaks the rule";
    }

    /// The stated rule: each group's pick is a column whose logit can be the group's largest within the columns'
    /// rounding errors (MLX's order: a NaN first), PM is that column's logit, PS its sum of exp(c(logit) - c(PM)) under
    /// the soft cap c.
    pub fn check(self: *Self, _: spec.Gate) !?[]const u8 {
        try self.logits(null);
        const bytes = self.rig.outputBytes();
        const ng = foldGroups(self.shape);
        for (0..ng) |g| {
            const x = self.group(g);
            const pm = spec.read(.f32, bytes, g);
            const pi: usize = @intFromFloat(spec.read(.u32, bytes[4 * ng ..], g));
            const ps = spec.read(.f32, bytes[8 * ng ..], g);
            const top = self.logit[x.at];
            if (!std.math.isFinite(top.v)) {
                if (pi != x.at) return self.fail("group {d}: column {d} where {d} leads", .{ g, pi, x.at });
                if (std.math.isNan(top.v) != std.math.isNan(pm) or (!std.math.isNan(top.v) and pm != top.v)) return self.fail("group {d}: max {d} where float64 gives {d}", .{ g, pm, top.v });
                if (std.math.isNan(x.sum) != std.math.isNan(ps)) return self.fail("group {d}: sum {d} where float64 gives {d}", .{ g, ps, x.sum });
                continue;
            }
            if (pi < g * MCOLS or pi >= @min(self.shape.k, (g + 1) * MCOLS)) return self.fail("group {d}: column {d} outside it", .{ g, pi });
            const mine = self.logit[pi];
            if (!(mine.v + mine.e + x.e >= top.v)) return self.fail("group {d}: column {d} ({d}) where {d} leads ({d})", .{ g, pi, mine.v, x.at, top.v });
            if (!(@abs(pm - mine.v) <= mine.e + 0x1p-24 * @abs(mine.v))) return self.fail("group {d}: max {d} where its column's logit is {d}", .{ g, pm, mine.v });
            var sum: f64 = 0;
            for (g * MCOLS..@min(self.shape.k, (g + 1) * MCOLS)) |c| sum += @exp(capped(self.logit[c].v) - capped(mine.v));
            if (!(@abs(ps - sum) <= sum * (2 * x.e + PRECISE + spec.gamma(@floatFromInt(MCOLS), host.U)))) return self.fail("group {d}: sum {d} where float64 gives {d}", .{ g, ps, sum });
        }
        return null;
    }

    fn wrapper(self: *Self) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        return self.rig.bytesOf(&(try kernels.markovFold(&g, x[0], x[1], x[2], x[3], x[4], @intCast(self.shape.k), x[5])));
    }
};

/// A step's pick and next state: rows = rank, n = the fold's threadgroups. TOKN, CONF, then NST [R].
const MarkovPick = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &mk_pick_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    pm: []f64 = &.{},
    pi: []f64 = &.{},
    ps: []f64 = &.{},
    p: []f64 = &.{},
    st: []f64 = &.{},
    args: [6]mlx.mlx_array = undefined,

    const Self = @This();

    fn init(a: std.mem.Allocator) Self {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    pub fn deinit(self: *Self) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n == 0) return null;
        var g = metal.Geometry{ .grid = .{ 256, 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{1}, .int32);
        g.out(&.{1}, .float32);
        g.out(&.{ 1, 1, cint(shape.rows) }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("R", cint(shape.rows));
        return g;
    }

    pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.n;
        self.pm = try host.uniform(a, rng, n, -8, 8, .float32);
        self.ps = try host.uniform(a, rng, n, 1, 300, .float32);
        self.pi = try a.alloc(f64, n);
        for (self.pi, 0..) |*x, g| x.* = @floatFromInt(g * MCOLS + rng.uintLessThan(usize, MCOLS));
        // The vocab's last group may hold no column.
        if (n > 2) self.pi[n - 1] = NONE_IDX;
        self.p = try host.uniform(a, rng, 3 * shape.rows, -3, 3, .float32);
        self.st = try host.uniform(a, rng, shape.rows, -1, 1, .bfloat16);
        switch (fill) {
            .random => {},
            .nonfinite => {
                if (n > 3) self.pm[3] = std.math.nan(f64);
                host.plant(self.p, 3, shape.rows);
            },
            // Gate 4: maxes equal and integer sums: exact in fp32 in any order, past fp16's range in a few.
            .stress => {
                @memset(self.pm, 0);
                for (self.ps) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, 20000, 40000));
            },
        }
        self.rig.reset();
        self.args = .{
            try self.rig.input(.float32, &.{cint(n)}, self.pm),
            try self.rig.input(.uint32, &.{cint(n)}, self.pi),
            try self.rig.input(.float32, &.{cint(n)}, self.ps),
            try self.rig.input(.float32, &.{cint(3 * shape.rows)}, self.p),
            try self.rig.input(.bfloat16, &.{ 1, 1, cint(shape.rows) }, self.st),
            try self.rig.input(.float32, &.{1}, &.{MCAP}),
        };
        return true;
    }

    pub fn run(self: *Self) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Self) usize {
        return 2 + self.shape.rows;
    }

    fn emit(self: *const Self, want: []f64, mag: ?[]f64, sim: attn.Sim) !void {
        const n = self.shape.n;
        var best: ?usize = null;
        for (0..n) |g| if (self.pi[g] != NONE_IDX) {
            if (best == null or better(self.pm[g], @intFromFloat(self.pi[g]), self.pm[best.?], @intFromFloat(self.pi[best.?]))) best = g;
        };
        const top = self.pm[best.?];
        const t = try self.rig.a.alloc(f64, n);
        defer self.rig.a.free(t);
        var k: usize = 0;
        for (0..n) |g| if (self.pi[g] != NONE_IDX) {
            t[k] = self.ps[g] * @exp(capped(self.pm[g]) - capped(top));
            k += 1;
        };
        const tot = if (sim) |x| spec.sumIn(x.acc, x.order, t[0..k]) else host.sum(t[0..k]);
        want[0] = self.pi[best.?];
        want[1] = 1 / tot;
        if (mag) |m| {
            m[0] = 0;
            m[1] = host.allowance(@abs(want[1]) * (PRECISE + spec.gamma(@floatFromInt(k + 8), host.U)));
        }
        // The state as the chain rounds it: MLX's sigmoid in bf16, then four bf16 roundings.
        const r = self.shape.rows;
        for (0..r) |i| {
            const keep = host.mlxSig(.bfloat16, .{ .v = self.p[i] });
            const th = std.math.tanh(self.p[r + i]);
            const cand = bf(.{ .v = th, .e = PRECISE * @abs(th) });
            const kept = bf(.{ .v = keep.v * self.st[i], .e = keep.e * @abs(self.st[i]) });
            const left = bf(.{ .v = 1 - keep.v, .e = keep.e });
            const blend = bf(.{ .v = left.v * cand.v, .e = left.e * @abs(cand.v) + cand.e * @abs(left.v) });
            const out = bf(.{ .v = kept.v + blend.v, .e = kept.e + blend.e });
            want[2 + i] = out.v;
            if (mag) |m| m[2 + i] = if (std.math.isFinite(out.e)) host.allowance(out.e) else 0;
        }
    }

    pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    fn wrapper(self: *Self) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        return self.rig.bytesOf(&(try kernels.markovPick(&g, .{ x[0], x[1], x[2] }, x[3], x[4], x[5])));
    }
};

fn one(_: Shape) usize {
    return 1;
}

fn picksOf(shape: Shape) usize {
    return shape.rows * PICKS;
}

fn picksDepth(_: Shape) usize {
    return PICKS;
}

/// The picks' chain of adds, then the taps'.
fn sumDepth(_: Shape) usize {
    return PICKS + KT;
}

fn none(_: Shape) usize {
    return 0;
}

const router_launch = metal.Launch{ .name = "msv_inkling2_router", .inputs = &.{ "X", "RT" }, .outputs = &.{"SC"}, .source = source("plugins/inkling2/kernels/router.metal"), .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// The widest stream a router row reads here.
const ROUTER_MAX_D = 4096;

/// A few rows' router logits: rows x experts x hidden, f32 throughout (`kernels.routerRows`).
const Router = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &router_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    /// `x` `[S, D]`, `rt` `[E, D]`.
    x: []f64 = &.{},
    rt: []f64 = &.{},
    args: [2]mlx.mlx_array = undefined,

    fn init(al: std.mem.Allocator) Router {
        return .{ .rig = metal.Rig.init(al), .arena = .init(al) };
    }

    pub fn deinit(self: *Router) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const Router, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k > ROUTER_MAX_D) return null;
        var geo = metal.Geometry{ .grid = .{ @intCast(32 * shape.n), @intCast(shape.rows), 1 }, .tg = .{ 32, 1, 1 } };
        geo.out(&.{ @intCast(shape.rows), @intCast(shape.n) }, .float32);
        geo.int("D", shape.k);
        geo.int("E", shape.n);
        return geo;
    }

    pub fn prepare(self: *Router, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const al = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const d = shape.k;
        self.x = try spread(al, rng, shape.rows * d, .f32);
        self.rt = try spread(al, rng, shape.n * d, .f32);
        switch (fill) {
            .random => {},
            // In the router's rows, which every row of the stream reads.
            .nonfinite => plant(self.rt, d),
            // Gate 4: a stream of ones over rows of 1, then 2^-12s, then -1: fp32 adds every small one, a half or
            // bf16 sum drops those it meets beside the 1.
            .stress => {
                @memset(self.x, 1);
                for (self.rt, 0..) |*v, i| v.* = if (i % d == 0) 1 else if (i % d == d - 1) -1 else 0x1p-12;
            },
        }
        self.rig.reset();
        self.args = .{
            try self.rig.input(.float32, &.{ 1, @intCast(shape.rows), @intCast(d) }, self.x),
            try self.rig.input(.float32, &.{ @intCast(shape.n), @intCast(d) }, self.rt),
        };
        return true;
    }

    pub fn run(self: *Router) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Router) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Output `o`'s products, row by expert.
    fn products(self: *const Router, o: usize, out: []f64) void {
        const d = self.shape.k;
        const xr = self.x[(o / self.shape.n) * d ..][0..d];
        const wr = self.rt[(o % self.shape.n) * d ..][0..d];
        for (out, xr, wr) |*t, p, q| t.* = p * q;
    }

    pub fn reference(self: *const Router, want: []f64, mag: []f64) !void {
        var t: [ROUTER_MAX_D]f64 = undefined;
        const terms = t[0..self.shape.k];
        for (want, mag, 0..) |*w, *m, o| {
            self.products(o, terms);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (terms) |v| {
                sum += v;
                abs += @abs(v);
            }
            w.* = sum;
            m.* = if (std.math.isFinite(abs)) abs else 0;
        }
    }

    pub fn simulate(self: *const Router, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        var t: [ROUTER_MAX_D]f64 = undefined;
        const terms = t[0..self.shape.k];
        for (out, 0..) |*o, i| {
            self.products(i, terms);
            o.* = spec.sumIn(acc, order, terms);
        }
    }

    fn wrapper(self: *Router) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        return self.rig.bytesOf(&.{try kernels.routerRows(&g, self.args[0], self.args[1])});
    }
};

fn routerDepth(shape: Shape) usize {
    return shape.k;
}

const norm_launch = metal.Launch{ .name = "msv_inkling2_norm", .inputs = &.{ "Y", "NW0", "NW1", "NW2", "RT", "PR" }, .outputs = &.{ "X", "SCORES" }, .source = source("plugins/inkling2/kernels/norm.metal"), .tg_bytes = 4 * (768 / 32) * 3, .m1 = 1024, .m2 = 1024, .adapts = true };

/// The model's norm epsilon, as the kernel reads it.
const NORM_EPS: f32 = 1e-6;

fn sum64(x: []const f64) f64 {
    var t: f64 = 0;
    for (x) |v| t += v;
    return t;
}

/// The visit's weighted RMS norms (`kernels.norm`): a later pass's three attention inputs, or the MLP input with the
/// router's scores over it; rows x streams x hidden.
fn Norm(comptime dt: mlx.mlx_dtype, comptime routed: bool) type {
    return struct {
        const Self = @This();
        const J: usize = if (routed) 1 else 3;
        const E: usize = if (routed) EXPERTS else 0;
        rig: metal.Rig,
        launch: *const metal.Launch = &norm_launch,
        arena: std.heap.ArenaAllocator,
        shape: Shape = undefined,
        /// `y` `[J, S, D]`, `nw` `[D]` each, `rt` `[E, D]`.
        y: []f64 = &.{},
        nw: [3][]f64 = undefined,
        rt: []f64 = &.{},
        args: [6]mlx.mlx_array = undefined,

        fn init(al: std.mem.Allocator) Self {
            return .{ .rig = metal.Rig.init(al), .arena = .init(al) };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
            if (shape.rows == 0 or shape.n != J or shape.k == 0) return null;
            const tg = kernels.streamGroup(@intCast(shape.k)) orelse return null;
            const d: c_int = @intCast(shape.k);
            const extra: c_int = if (routed) @divTrunc(@as(c_int, E) * 32 + tg - 1, tg) * tg else 0;
            var geo = metal.Geometry{ .grid = .{ d + extra, @intCast(shape.rows), 1 }, .tg = .{ tg, 1, 1 } };
            geo.out(&.{ @intCast(J), @intCast(shape.rows), d }, dt);
            if (routed) geo.out(&.{ @intCast(shape.rows), @intCast(E) }, .float32) else {
                geo.out(&.{1}, .float32);
                geo.unread = 0b10;
            }
            geo.dtype("T", dt);
            geo.int("TG", tg);
            geo.int("D", d);
            geo.int("J", @as(c_int, J));
            geo.int("E", @as(c_int, E));
            return geo;
        }

        pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
            if (self.geometry(shape) == null) return false;
            self.shape = shape;
            _ = self.arena.reset(.retain_capacity);
            const al = self.arena.allocator();
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const d = shape.k;
            self.y = try spread(al, rng, J * shape.rows * d, .f32);
            for (&self.nw) |*w| w.* = try spread(al, rng, d, elemOf(dt));
            self.rt = try spread(al, rng, E * d, .f32);
            switch (fill) {
                .random => {},
                .nonfinite => plant(self.y, d),
                // Gate 4: the sums of squares over integers (exact in fp32, past fp16's range); weights of one and a
                // router of zeros, so only those sums can miss.
                .stress => {
                    for (self.y) |*v| v.* = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 65))) - 32);
                    for (self.nw) |w| @memset(w, 1);
                    @memset(self.rt, 0);
                },
            }
            self.rig.reset();
            const dd: c_int = @intCast(d);
            const y = try self.rig.input(.float32, &.{ @intCast(J), @intCast(shape.rows), dd }, self.y);
            self.args = .{
                y,
                try self.rig.input(dt, &.{dd}, self.nw[0]),
                try self.rig.input(dt, &.{dd}, self.nw[1]),
                try self.rig.input(dt, &.{dd}, self.nw[2]),
                if (routed) try self.rig.input(.float32, &.{ @intCast(E), dd }, self.rt) else y,
                try self.rig.input(.float32, &.{1}, &.{@as(f64, NORM_EPS)}),
            };
            return true;
        }

        pub fn run(self: *Self) !void {
            try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
        }

        pub fn count(self: *const Self) usize {
            return (J * self.shape.k + E) * self.shape.rows;
        }

        /// Every output: float64 (`acc` null) or with the sums run in `acc` in `order` (gate 4).
        fn outputsIn(self: *const Self, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
            const s = self.shape.rows;
            const d = self.shape.k;
            const nx = J * s * d;
            const a = self.arena.child_allocator;
            const sq = try a.alloc(f64, d);
            defer a.free(sq);
            const normed = try a.alloc(f64, d);
            defer a.free(normed);
            const terms = try a.alloc(f64, d);
            defer a.free(terms);
            for (0..J) |j| for (0..s) |r| {
                const row = self.y[(j * s + r) * d ..][0..d];
                for (sq, row) |*q, v| q.* = v * v;
                const ss = if (acc) |ac| spec.sumIn(ac, order, sq) else sum64(sq);
                const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(d)) + @as(f64, NORM_EPS));
                for (want[(j * s + r) * d ..][0..d], row, self.nw[j]) |*o, v, w| o.* = v * inv * w;
                if (!routed or j != 0) continue;
                // The router reads the normed row as the stream holds it.
                for (normed, row, self.nw[0]) |*n, v, w| n.* = spec.roundTo(elemOf(dt), v * inv * w);
                for (0..E) |ex| {
                    for (terms, normed, self.rt[ex * d ..][0..d]) |*t, n, q| t.* = n * q;
                    want[nx + r * E + ex] = if (acc) |ac| spec.sumIn(ac, order, terms) else sum64(terms);
                    if (mag) |m| {
                        var abs: f64 = 0;
                        for (terms) |t| abs += @abs(t);
                        m[nx + r * E + ex] = if (std.math.isFinite(abs)) abs else 0;
                    }
                }
            };
            if (mag) |m| for (m[0..nx], want[0..nx]) |*o, w| {
                o.* = @abs(w) / 2;
            };
        }

        pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
            try self.outputsIn(want, mag, null, .sequential);
        }

        pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
            try self.outputsIn(out, null, acc, order);
        }

        fn wrapper(self: *Self) ![]const u8 {
            var g = ops.Graph.init(self.rig.a, self.rig.stream());
            defer g.deinit();
            var cfg: config.Config = undefined;
            cfg.hidden_size = @intCast(self.shape.k);
            cfg.num_experts = @intCast(EXPERTS);
            cfg.rms_norm_eps = NORM_EPS;
            const x = self.args;
            const o = try kernels.norm(&g, &cfg, x[0], .{ x[1], x[2], x[3] }, @intCast(J), dt, if (routed) x[4] else null);
            return self.rig.bytesOf(if (routed) &.{ o[0], o[1] } else &.{o[0]});
        }
    };
}

fn normsOf(shape: Shape) usize {
    return shape.n * shape.rows * shape.k;
}

fn scoresOf(shape: Shape) usize {
    return shape.rows * EXPERTS;
}

fn rowLength(shape: Shape) usize {
    return shape.k;
}

const dtypes = [_]mlx.mlx_dtype{ .bfloat16, .float32 };

fn name(comptime what: []const u8, comptime dt: mlx.mlx_dtype) []const u8 {
    return what ++ (if (dt == .bfloat16) " bf16" else " f32");
}

pub fn register(set: *kc.Set) !void {
    try set.add(Act, .init(set.a), .{
        .name = "inkling2 act bf16",
        .accum = .none,
        .depth = none,
        // The scale, each arm's two products and tanh, their product, the sigmoid's exp and add, the divide.
        .ops = 11,
        .out = .bf16,
        // A 32-row and a 2048-row prompt chunk's picks (8 a row), without and with the release's per-expert scales.
        .served = &.{ .{ .rows = 256, .n = 1, .k = 1152 }, .{ .rows = 256, .n = 2, .k = 1152 }, .{ .rows = 16384, .n = 2, .k = 1152 } },
    }, null);
    try set.add(Sum, .init(set.a), .{
        .name = "inkling2 sum bf16",
        .accum = .f32,
        .depth = sumDepth,
        // A pick's two products; the tap products, the scale and the residual's add.
        .ops = 5,
        // The MLP output as the stream holds it, the conv's sum, and the scaled output, each rounded before the next.
        .mids = 3,
        .out = .bf16,
        // The conv's window: rows under it read the state.
        .tiles = .{ .{ .size = KT - 1 }, .{}, .{} },
        // A 32-row chunk without and with the release's per-expert scales, the narrowed prompt chunk.
        .served = &.{ .{ .rows = 32, .n = 1, .k = 2304 }, .{ .rows = 32, .n = 2, .k = 2304 }, .{ .rows = 2048, .n = 2, .k = 2304 } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "eight picks and four taps, each sum then rounded to bf16: a half or bf16 sum's error is inside that rounding" },
    }, null);
    try set.add(TopK, .init(set.a), .{
        .name = "inkling2 topk",
        .accum = .f32,
        .depth = picksDepth,
        // The sigmoids' exp, add and divide; the weight's add, divide and scale.
        .ops = 6,
        .out = .f32,
        .outs = &.{ .{ .elem = .u32, .count = picksOf }, .{ .elem = .f32, .count = picksOf } },
        .nan = .{ .stated = "a NaN logit ranks nowhere; a slot past the ranked keys keeps its own index at weight zero" },
        // One row, a draft verify, a prompt chunk.
        .served = &.{ .{ .rows = 1, .n = PICKS, .k = EXPERTS }, .{ .rows = 8, .n = PICKS, .k = EXPERTS }, .{ .rows = 2048, .n = PICKS, .k = EXPERTS } },
        .sums = .{ .f16 = 100, .bf16 = 100, .exact = false },
    }, null);
    try set.add(Gate, .init(set.a, .float32), .{
        .name = "inkling2 gate (one row, f32 projections)",
        .accum = .f32,
        .depth = rankDepth,
        // Each arm's tanh and product, the one's complement, their add.
        .ops = 6,
        .out = .f32,
        // The attention inputs' three gates and the MLP's one, of a looped visit.
        .served = &.{ .{ .rows = 3, .n = 1, .k = 2304 }, .{ .rows = 1, .n = 1, .k = 2304 } },
        .sums = .{ .f16 = 16, .exact = false, .why = "a half sum drops only the few small terms it meets beside the one" },
    }, null);
    inline for (dtypes) |dt| {
        try set.add(Gated(dt), .init(set.a), .{
            .name = name("inkling2 gated", dt),
            .accum = .none,
            .depth = none,
            // The f32 product.
            .ops = 1,
            .out = elemOf(dt),
            .replaces = "astype f32 -> multiply -> astype",
            // One row, a draft verify, a prompt chunk: rows of 24 heads x 128.
            .served = &.{ .{ .rows = 1, .n = 1, .k = 3072 }, .{ .rows = 8, .n = 1, .k = 3072 }, .{ .rows = 2048, .n = 1, .k = 3072 } },
        }, null);
        try set.add(State(dt), .init(set.a), .{
            .name = name("inkling2 state", dt),
            .accum = .none,
            .depth = none,
            .out = .f32,
            // Fewer rows than the state holds keep some of the old one.
            .tiles = .{ .{ .size = KT - 1 }, .{}, .{} },
            // One row, and the narrowed prompt chunk: kv width 1024, hidden 2304.
            .served = &.{ .{ .rows = 1, .n = 1024, .k = 2304 }, .{ .rows = 2048, .n = 1024, .k = 2304 } },
        }, null);
        try set.add(Mask(dt), .init(set.a), .{
            .name = name("inkling2 relmask", dt),
            .accum = .none,
            .depth = none,
            .out = elemOf(dt),
            // A draft verify on a sliding and on a global layer, the most rows it takes.
            .served = &.{ .{ .rows = 8, .n = 1031, .k = 1024 }, .{ .rows = 8, .n = 4104, .k = 0 }, .{ .rows = kernels.small_rows, .n = 1055, .k = 1024 }, .{ .rows = 2, .n = 2, .k = 1024 } },
        }, null);
    }
    const split_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "the softmax's exp leaves no sum exact: the stress inputs check the fp32 sums within the float64 bound" };
    // 4 and 8 rows behind a sliding layer's window and a global layer's keys; 8 KV heads of 3 query heads.
    try set.add(RelSplit, .init(set.a, 1024, false), .{
        .name = "inkling2 relsplit bf16 (sliding 1024)",
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f32,
        .tiles = .{ .{}, .{}, .{ .size = SKPS, .min = kernels.split_rows } },
        .served = &.{ .{ .rows = 8, .n = kernels.split_rows, .k = 1031 }, .{ .rows = 8, .n = kernels.split_from, .k = 1027 } },
        .sums = split_sums,
    }, null);
    try set.add(RelSplit, .init(set.a, 0, false), .{
        .name = "inkling2 relsplit bf16 (global)",
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f32,
        .tiles = .{ .{}, .{}, .{ .size = SKPS, .min = kernels.split_rows } },
        .served = &.{ .{ .rows = 8, .n = kernels.split_rows, .k = 4104 }, .{ .rows = 8, .n = kernels.split_from, .k = 300 } },
        .sums = split_sums,
    }, null);
    // A verify's read: 2 and 8 rows of their own behind a sliding window's keys and a global layer's.
    try set.add(RelSplit, .init(set.a, 1024, true), .{
        .name = "inkling2 relsplit bf16 (sliding 1024, own rows)",
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f32,
        .tiles = .{ .{}, .{}, .{ .size = SKPS, .min = kernels.split_rows + 1 } },
        .served = &.{ .{ .rows = 8, .n = kernels.split_rows, .k = 1031 }, .{ .rows = 8, .n = 2, .k = 1025 } },
        .sums = split_sums,
    }, null);
    try set.add(RelSplit, .init(set.a, 0, true), .{
        .name = "inkling2 relsplit bf16 (global, own rows)",
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f32,
        .tiles = .{ .{}, .{}, .{ .size = SKPS, .min = kernels.split_rows + 1 } },
        .served = &.{ .{ .rows = 8, .n = kernels.split_rows, .k = 4104 }, .{ .rows = 8, .n = 2, .k = 300 } },
        .sums = split_sums,
    }, null);
    // A prompt's rows on the tensor unit: a sliding window's tiles and a global layer's, both dtypes the kernel serves.
    inline for (.{ .{ mlx.mlx_dtype.bfloat16, "bf16" }, .{ mlx.mlx_dtype.float16, "f16" } }) |d| inline for (.{ .{ 1024, "sliding 1024" }, .{ 0, "global" } }) |w| {
        try set.add(RelNax, .init(set.a, d[0], w[0]), .{
            .name = "inkling2 relattn_nax " ++ d[1] ++ " (" ++ w[1] ++ ")",
            .accum = .none,
            .depth = host.depthOne,
            .out = if (d[0] == .bfloat16) .bf16 else .f16,
            .nan = .{ .stated = "a planted NaN or infinity reaches every row whose key tiles hold it, a masked key's value at weight 0 included" },
            .nax = true,
            // 64 query rows a threadgroup in tiles of 16, keys in tiles of 32.
            .tiles = .{ .{ .size = 64, .min = 1 }, .{}, .{ .size = NAX_BK, .min = 1 } },
            .served = &.{ .{ .rows = 70, .n = 1, .k = 1200 }, .{ .rows = 40, .n = 2, .k = 64 } },
        }, null);
    };
    try set.add(RelComb, .init(set.a), .{
        .name = "inkling2 relcomb bf16",
        .accum = .f32,
        .depth = host.depthOne,
        // The gate's product, after the read's rounding to bf16.
        .ops = 1,
        .mids = 1,
        .out = .bf16,
        // 8 and 4 rows over a sliding window's 9 splits, a 4k and a 16k global layer's.
        .served = &.{ .{ .rows = 8, .n = 9, .k = SD }, .{ .rows = 4, .n = 33, .k = SD }, .{ .rows = 8, .n = 129, .k = SD } },
    }, null);
    // 8 slots behind a sliding window and behind a global layer's keys, and a two-slot launch.
    try set.add(RelSlots, .init(set.a), .{
        .name = "inkling2 relslots bf16",
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f32,
        .served = &.{ .{ .rows = 8, .n = 1031, .k = 1024 }, .{ .rows = 8, .n = 4104, .k = 0 }, .{ .rows = 2, .n = 300, .k = 0 } },
        .sums = split_sums,
    }, null);
    const markov_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "the outputs round to bf16 or pass an exp: the stress inputs check the fp32 sums within the float64 bound" };
    // The release's drafter: rank 256, hidden 2304, 201024 head rows (the fold's real vocab 200058, 98 groups).
    try set.add(MarkovRnn, .init(set.a), .{
        .name = "inkling2 markov_rnn bf16",
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f32,
        .served = &.{.{ .rows = 1, .n = 256, .k = 2304 }},
        .sums = markov_sums,
    }, null);
    try set.add(MarkovFold, .init(set.a), .{
        .name = "inkling2 markov_fold bf16",
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f32,
        .outs = &.{ .{ .elem = .f32, .count = foldGroups }, .{ .elem = .u32, .count = foldGroups }, .{ .elem = .f32, .count = foldGroups } },
        .nan = .{ .stated = "MLX's argmax order: a NaN logit is a group's pick, its max and sum NaN" },
        .served = &.{ .{ .rows = 256, .n = 16384, .k = 16000 }, .{ .rows = 256, .n = 4096, .k = 4096 } },
        .sums = markov_sums,
    }, null);
    try set.add(MarkovPick, .init(set.a), .{
        .name = "inkling2 markov_pick bf16",
        .accum = .f32,
        .depth = host.depthOne,
        // The probability's exp and divide; the state's four bf16 roundings before its last.
        .ops = 2,
        .mids = 4,
        .out = .bf16,
        .outs = &.{ .{ .elem = .u32, .count = one }, .{ .elem = .f32, .count = one }, .{ .elem = .bf16, .count = one } },
        .served = &.{ .{ .rows = 256, .n = 98, .k = 1 }, .{ .rows = 256, .n = 8, .k = 1 } },
        .sums = markov_sums,
    }, null);
    try set.add(Gate, .init(set.a, .bfloat16), .{
        .name = "inkling2 gate (one row, bf16 projections)",
        .accum = .f32,
        .depth = rankDepth,
        // Each arm's tanh and product, the one's complement, their add.
        .ops = 6,
        .out = .f32,
        // The attention inputs' three gates and the MLP's one, of a looped visit.
        .served = &.{ .{ .rows = 3, .n = 1, .k = 2304 }, .{ .rows = 1, .n = 1, .k = 2304 } },
        .sums = .{ .f16 = 16, .exact = false, .why = "a half sum drops only the few small terms it meets beside the one" },
    }, null);
    try set.add(Router, .init(set.a), .{
        .name = "inkling2 router (few rows, f32)",
        .accum = .f32,
        .depth = routerDepth,
        // Each product.
        .ops = 1,
        .out = .f32,
        // A draft verify's rows, two rows and `small_rows`, over the model's experts.
        .served = &.{ .{ .rows = 8, .n = EXPERTS, .k = 2304 }, .{ .rows = 2, .n = EXPERTS, .k = 2304 }, .{ .rows = kernels.small_rows, .n = EXPERTS, .k = 2304 } },
    }, null);
    inline for (dtypes) |dt| {
        try set.add(Norm(dt, false), .init(set.a), .{
            .name = name("inkling2 norm (three streams)", dt),
            .accum = .f32,
            .depth = rowLength,
            // f32 roundings after the sum: the mean, + eps, rsqrt (2), the products with the row and the weight.
            .ops = 6,
            .out = elemOf(dt),
            // A later pass's attention inputs: one row, a draft verify, a prompt chunk.
            .served = &.{ .{ .rows = 1, .n = 3, .k = 2304 }, .{ .rows = 8, .n = 3, .k = 2304 }, .{ .rows = 512, .n = 3, .k = 2304 } },
            .sums = .{ .f16 = 32, .bf16 = 0, .why = "positive squares: a pairwise bf16 sum stays near the output's rounding" },
        }, null);
        try set.add(Norm(dt, true), .init(set.a), .{
            .name = name("inkling2 norm (MLP input + router)", dt),
            .accum = .f32,
            .depth = rowLength,
            .ops = 6,
            .out = elemOf(dt),
            .outs = &.{ .{ .elem = comptime elemOf(dt), .count = normsOf }, .{ .elem = .f32, .count = scoresOf } },
            // A later pass's MLP input with its router: one row, a draft verify, a prompt chunk.
            .served = &.{ .{ .rows = 1, .n = 1, .k = 2304 }, .{ .rows = 8, .n = 1, .k = 2304 }, .{ .rows = 64, .n = 1, .k = 2304 } },
            .sums = .{ .f16 = 32, .bf16 = 0, .why = "positive squares: a pairwise bf16 sum stays near the output's rounding" },
        }, null);
    }
    // Every affine width a pack's attention may carry at g64, 4-bit at g32 and g128, and the release's dense bf16.
    inline for (.{ 2, 3, 4, 5, 6, 8 }) |bits| try set.add(OProj, .init(set.a, bits, 64), oprojSpec(std.fmt.comptimePrint("inkling2 oproj bf16 ({d}-bit g64)", .{bits}), 64, .{}), null);
    try set.add(OProj, .init(set.a, 4, 32), oprojSpec("inkling2 oproj bf16 (4-bit g32)", 32, .{ .f16 = 40, .why = "stress codes under 16 in groups of 32: an fp16 sum misses by ~45x, short of 100x but far past fp32's exact sums" }), null);
    try set.add(OProj, .init(set.a, 4, 128), oprojSpec("inkling2 oproj bf16 (4-bit g128)", 128, .{}), null);
    try set.add(OProj, .init(set.a, 0, 16), oprojSpec("inkling2 oproj bf16 (dense)", 16, .{}), null);
    try vision_contract.register(set);
}

// ── tests ──

test "kernel contract inkling2: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    inline for (.{ Act, Sum, TopK, Gate }, 0..) |T, i| {
        const k = set.kernels.items[i];
        const p: *T = @ptrCast(@alignCast(k.ctx));
        for (k.spec.served) |shape| {
            try std.testing.expect(try p.prepare(shape, .random, 7));
            try p.run();
            const ours = try a.dupe(u8, p.rig.outputBytes());
            defer a.free(ours);
            try std.testing.expectEqualSlices(u8, try p.wrapper(), ours);
        }
    }
    var at: usize = 4;
    inline for (dtypes) |dt| inline for (.{ Gated(dt), State(dt), Mask(dt) }) |T| {
        const k = set.kernels.items[at];
        at += 1;
        const p: *T = @ptrCast(@alignCast(k.ctx));
        for (k.spec.served) |shape| {
            try std.testing.expect(try p.prepare(shape, .random, 7));
            try p.run();
            const ours = try a.dupe(u8, p.rig.outputBytes());
            defer a.free(ours);
            try std.testing.expectEqualSlices(u8, try p.wrapper(), ours);
        }
    };
    inline for (.{ RelSplit, RelSplit, RelSplit, RelSplit, RelNax, RelNax, RelNax, RelNax, RelComb, RelSlots, MarkovRnn, MarkovFold, MarkovPick, Gate, Router, Norm(.bfloat16, false), Norm(.bfloat16, true), Norm(.float32, false), Norm(.float32, true) }) |T| {
        const k = set.kernels.items[at];
        at += 1;
        // Off NAX the tensor-unit read is not dispatched.
        const served = T != RelNax or @import("../../mlx/device.zig").naxAvailable();
        const p: *T = @ptrCast(@alignCast(k.ctx));
        if (served) for (k.spec.served) |shape| {
            try std.testing.expect(try p.prepare(shape, .random, 7));
            try p.run();
            const ours = try a.dupe(u8, p.rig.outputBytes());
            defer a.free(ours);
            try std.testing.expectEqualSlices(u8, try p.wrapper(), ours);
        };
    }
    var end = at;
    while (end < set.kernels.items.len and std.mem.startsWith(u8, set.kernels.items[end].spec.name, "inkling2 oproj")) end += 1;
    for (set.kernels.items[at..end]) |k| {
        const p: *OProj = @ptrCast(@alignCast(k.ctx));
        for (k.spec.served) |shape| {
            try std.testing.expect(try p.prepare(shape, .random, 7));
            try p.run();
            const ours = try a.dupe(u8, p.rig.outputBytes());
            defer a.free(ours);
            try std.testing.expectEqualSlices(u8, try p.wrapper(), ours);
        }
    }
    try vision_contract.checkWrappers(a, set.kernels.items[end..]);
}

// Bar: gates 1-4 on this area alone, from the plugin's own file (the registry's inkling2 test runs the same gates).
test "kernel contract inkling2: gates 1-4 pass" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    var t: kc.gates.Tally = .{};
    try kc.gates.runAll(a, set.kernels.items, 0xc0de, &t);
    try std.testing.expectEqual(@as(usize, 0), t.problems());
    // Off NAX the tensor-unit reads skip.
    try std.testing.expectEqual(set.kernels.items.len, t.passed + t.skipped);
}
