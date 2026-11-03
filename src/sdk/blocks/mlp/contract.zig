//! The mlp area's kernel contract: the table-lookup gated activations (SwiGLU, its clamped and joined-row forms, and
//! the GELU-erf table times up), each bit-equal to the MLX op chain it replaces. Listed in `kernel_contract/registry.zig`.
//! Known issue: swiglu_clamp turns a NaN gate or up into a finite output (gate 1; `clamp_drops_nan`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const fa = @import("fused_activations.zig");
const activations = @import("activations.zig");
const scalarOf = @import("../../../mlx/array.zig").scalarOf;

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

/// The clamp a `swiglu_limit` model (GLM-5-Next) serves, as the wrapper takes it.
const LIMIT: f32 = 10;

const Arm = enum { silu, clamp, gelu_erf, joined };

fn gateLaunch(comptime name: [:0]const u8, comptime src: [:0]const u8, comptime clamp: bool) metal.Launch {
    const ins: []const [*:0]const u8 = if (clamp) &.{ "gate", "up", "sigtab", "N_size", "lim" } else &.{ "gate", "up", "sigtab", "N_size" };
    return .{ .name = name, .inputs = ins, .outputs = &.{"y"}, .source = src, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };
}

const silu_launch = gateLaunch("mlxserve_swiglu", fa.SWIGLU_SOURCE, false);
const clamp_launch = gateLaunch("mlxserve_swiglu_clamp", fa.SWIGLU_CLAMP_SOURCE, true);
const gelu_launch = gateLaunch("mlxserve_gelu_erf_mul", fa.TABLE_MUL_SOURCE, false);
const joined_launch = metal.Launch{ .name = "mlxserve_swiglu_joined", .inputs = &.{ "both", "sigtab", "N_size" }, .outputs = &.{"y"}, .source = fa.SWIGLU_JOINED_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// One registration: an arm at a dtype over gate/up [rows, k] (joined: one [rows, 2k] row of gate | up).
const Gated = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    arm: Arm,
    dt: mlx.mlx_dtype,
    /// The activation over every 16-bit pattern, from MLX's own op(s): what the kernel indexes.
    table: []f64 = &.{},
    shape: Shape = undefined,
    gate: []f64 = &.{},
    up: []f64 = &.{},
    args: [5]mlx.mlx_array = undefined,
    n_args: usize = 0,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, arm: Arm, dt: mlx.mlx_dtype) Gated {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .arm = arm, .dt = dt };
    }

    pub fn deinit(self: *Gated) void {
        const a = self.rig.a;
        self.rig.deinit();
        a.free(self.table);
        a.free(self.gate);
        a.free(self.up);
    }

    pub fn geometry(self: *const Gated, shape: Shape) ?metal.Geometry {
        const n: c_int = @intCast(shape.rows * shape.k);
        if (n <= 0) return null;
        const tg: c_int = if (self.arm == .joined) 256 else @min(256, @max(32, @divTrunc(n + 31, 32) * 32));
        var g = metal.Geometry{ .grid = .{ @divTrunc(n + tg - 1, tg) * tg, 1, 1 }, .tg = .{ tg, 1, 1 } };
        g.out(&.{ @intCast(shape.rows), @intCast(shape.k) }, self.dt);
        g.dtype("T", self.dt);
        if (self.arm == .joined) g.int("F", shape.k);
        return g;
    }

    fn buildTable(self: *Gated) !void {
        if (self.table.len > 0) return;
        const s = self.rig.stream();
        const t = if (self.arm == .gelu_erf) blk: {
            const vals = try fa.all16BitPatterns(s, self.dt, std.heap.c_allocator);
            defer _ = mlx.mlx_array_free(vals);
            const one = try scalarOf(1.0, self.dt, s);
            defer _ = mlx.mlx_array_free(one);
            const half = try scalarOf(0.5, self.dt, s);
            defer _ = mlx.mlx_array_free(half);
            const chain = try fa.geluErfChain(vals, one, half, s);
            defer _ = mlx.mlx_array_free(chain);
            break :blk try host.read(self.rig.a, chain, s);
        } else try host.read(self.rig.a, try fa.swigluSigTable(s, self.dt, std.heap.c_allocator), s);
        self.table = t;
    }

    pub fn prepare(self: *Gated, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        try self.buildTable();
        const a = self.rig.a;
        self.shape = shape;
        const n = shape.rows * shape.k;
        a.free(self.gate);
        a.free(self.up);
        self.gate = &.{};
        self.up = &.{};
        self.gate = try a.alloc(f64, n);
        self.up = try a.alloc(f64, n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const reach: f64 = if (self.arm == .clamp) 16 else 8;
        host.uniformInto(rng, self.gate, -reach, reach, self.dt);
        host.uniformInto(rng, self.up, -reach, reach, self.dt);
        if (fill == .nonfinite) {
            const k = shape.k;
            const nan = std.math.nan(f64);
            const inf = std.math.inf(f64);
            const at = [_]struct { usize, usize, f64, bool }{ .{ 0, k / 3, nan, true }, .{ 1, k / 2, inf, true }, .{ 2, k - 1, -inf, true }, .{ 3, 0, nan, false }, .{ 4, k / 4, inf, true }, .{ 4, k / 4 + 1, -inf, false } };
            for (at) |p| if (p[0] < shape.rows) {
                (if (p[3]) self.gate else self.up)[p[0] * k + p[1]] = p[2];
            };
            if (shape.rows > 4) self.up[4 * k + k / 4] = 0; // +inf times a zero up: NaN
        }
        try self.bind();
        return true;
    }

    fn bind(self: *Gated) !void {
        const rig = &self.rig;
        rig.reset();
        const rows: c_int = @intCast(self.shape.rows);
        const k: c_int = @intCast(self.shape.k);
        const n: f64 = @floatFromInt(self.shape.rows * self.shape.k);
        if (self.arm == .joined) {
            const both = try rig.a.alloc(f64, 2 * self.gate.len);
            defer rig.a.free(both);
            for (0..self.shape.rows) |r| {
                @memcpy(both[2 * r * self.shape.k ..][0..self.shape.k], self.gate[r * self.shape.k ..][0..self.shape.k]);
                @memcpy(both[(2 * r + 1) * self.shape.k ..][0..self.shape.k], self.up[r * self.shape.k ..][0..self.shape.k]);
            }
            self.args[0] = try rig.input(self.dt, &.{ rows, 2 * k }, both);
            self.args[1] = try rig.input(self.dt, &.{65536}, self.table);
            self.args[2] = try rig.input(.int32, &.{}, &.{n});
            self.n_args = 3;
            return;
        }
        self.args[0] = try rig.input(self.dt, &.{ rows, k }, self.gate);
        self.args[1] = try rig.input(self.dt, &.{ rows, k }, self.up);
        self.args[2] = try rig.input(self.dt, &.{65536}, self.table);
        self.args[3] = try rig.input(.int32, &.{}, &.{n});
        self.n_args = 4;
        if (self.arm == .clamp) {
            self.args[4] = try rig.input(self.dt, &.{}, &.{LIMIT});
            self.n_args = 5;
        }
    }

    pub fn run(self: *Gated) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Gated) usize {
        return self.shape.rows * self.shape.k;
    }

    fn pattern(self: *const Gated, v: f64) usize {
        return if (self.dt == .bfloat16) spec.toBf16(@floatCast(v)) else @as(u16, @bitCast(@as(f16, @floatCast(v))));
    }

    /// MLX's Minimum / Clip: a NaN operand stays NaN.
    fn clampNaN(v: f64, lo: f64, hi: f64) f64 {
        if (std.math.isNan(v)) return v;
        return @max(@min(v, hi), lo);
    }

    /// `act(g) * u` with the activation read from the table (MLX's own value), rounded at each op of the chain.
    pub fn reference(self: *const Gated, want: []f64, mag: []f64) !void {
        const elem = host.elemOf(self.dt);
        for (want, mag, self.gate, self.up) |*y, *m, g0, up0| {
            const g = if (self.arm == .clamp) clampNaN(g0, -std.math.inf(f64), LIMIT) else g0;
            const u = if (self.arm == .clamp) clampNaN(up0, -LIMIT, LIMIT) else up0;
            const t = self.table[self.pattern(g)];
            const act = if (self.arm == .gelu_erf) t else spec.roundTo(elem, g * t);
            y.* = act * u;
            m.* = 0;
        }
    }

    /// The chain the kernel replaces, on the same inputs.
    pub fn replaced(self: *Gated) ![]const u8 {
        const s = self.rig.stream();
        var g = ops.Graph.init(self.rig.a, s);
        defer g.deinit();
        const x = self.args;
        const pair: [2]mlx.mlx_array = if (self.arm == .joined) .{
            try g.sliceAxis(x[0], 1, 0, @intCast(self.shape.k)),
            try g.sliceAxis(x[0], 1, @intCast(self.shape.k), @intCast(2 * self.shape.k)),
        } else .{ x[0], x[1] };
        const y = switch (self.arm) {
            .silu, .joined => try g.mul(try g.mul(pair[0], try g.sigmoid(pair[0])), pair[1]),
            .clamp => try g.adopt(try activations.clampedSwiGLU(s, pair[0], pair[1], LIMIT)),
            .gelu_erf => blk: {
                const one = try g.adopt(try scalarOf(1.0, self.dt, s));
                const half = try g.adopt(try scalarOf(0.5, self.dt, s));
                break :blk try g.mul(try g.adopt(try fa.geluErfChain(pair[0], one, half, s)), pair[1]);
            },
        };
        return self.rig.bytesOf(&.{y});
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *Gated) ![]const u8 {
        const s = self.rig.stream();
        const x = self.args;
        const y = switch (self.arm) {
            .silu => try fa.fusedSwiGLU(s, x[0], x[1]),
            .clamp => try fa.fusedClampedSwiGLU(s, x[0], x[1], LIMIT),
            .joined => try fa.swigluJoined(s, x[0]),
            .gelu_erf => blk: {
                const one = try scalarOf(1.0, self.dt, s);
                defer _ = mlx.mlx_array_free(one);
                const half = try scalarOf(0.5, self.dt, s);
                defer _ = mlx.mlx_array_free(half);
                break :blk try fa.fusedGeluErfMul(s, x[0], x[1], one, half);
            },
        } orelse return error.Declined;
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

fn none(_: Shape) usize {
    return 0;
}

fn gatedSpec(comptime name: []const u8, dt: mlx.mlx_dtype, replaces: []const u8, known: ?spec.Known) spec.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = none,
        .out = host.elemOf(dt),
        .replaces = replaces,
        // The flat element count against the 256-thread group; rows 1 at the edges.
        .tiles = .{ .{}, .{}, .{ .size = 256 } },
        .served = served,
        .known = known,
    };
}

/// Decode at a dense MLP width, a MoE layer's 16 tokens, a prefill chunk.
const served: []const Shape = &.{ .{ .rows = 1, .n = 1, .k = 12288 }, .{ .rows = 16, .n = 1, .k = 512 }, .{ .rows = 64, .n = 1, .k = 2048 } };

const chain_silu = "sigmoid -> multiply -> multiply";

/// Gate 1's finding on the shipped clamp (see the report): it must keep failing until its fix lands.
const clamp_drops_nan: spec.Known = .{ .gate = .nonfinite, .finding = "metal::min / max return the other operand for a NaN, where MLX's Minimum and Clip (the clampedSwiGLU chain it replaces) keep the NaN: a NaN gate or up gives a finite output" };

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(Gated, .init(a, &silu_launch, .silu, .bfloat16), gatedSpec("swiglu bf16", .bfloat16, chain_silu, null), null);
    try set.add(Gated, .init(a, &silu_launch, .silu, .float16), gatedSpec("swiglu f16", .float16, chain_silu, null), null);
    try set.add(Gated, .init(a, &clamp_launch, .clamp, .bfloat16), gatedSpec("swiglu_clamp bf16 (limit 10)", .bfloat16, "minimum, clip -> " ++ chain_silu, clamp_drops_nan), null);
    try set.add(Gated, .init(a, &clamp_launch, .clamp, .float16), gatedSpec("swiglu_clamp f16 (limit 10)", .float16, "minimum, clip -> " ++ chain_silu, clamp_drops_nan), null);
    try set.add(Gated, .init(a, &gelu_launch, .gelu_erf, .bfloat16), gatedSpec("table_mul bf16 (gelu_erf)", .bfloat16, "gelu_erf chain -> multiply", null), null);
    try set.add(Gated, .init(a, &gelu_launch, .gelu_erf, .float16), gatedSpec("table_mul f16 (gelu_erf)", .float16, "gelu_erf chain -> multiply", null), null);
    try set.add(Gated, .init(a, &joined_launch, .joined, .bfloat16), gatedSpec("swiglu_joined bf16", .bfloat16, "slice halves -> " ++ chain_silu, null), null);
    try set.add(Gated, .init(a, &joined_launch, .joined, .float16), gatedSpec("swiglu_joined f16", .float16, "slice halves -> " ++ chain_silu, null), null);
}

// ── tests ──

test "kernel contract mlp: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    fa.swiglu_fused_override = true;
    defer fa.swiglu_fused_override = null;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const t: *Gated = @ptrCast(@alignCast(k.ctx));
        try std.testing.expect(try t.prepare(k.spec.served[0], .random, 7));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        const theirs = try t.wrapper();
        if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{k.spec.name});
        try std.testing.expectEqualSlices(u8, theirs, ours);
    }
}
