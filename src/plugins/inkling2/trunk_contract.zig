//! Inkling 2's trunk kernels in the kernel contract (`contract.zig` registers them): `gate`'s prompt arm (a prompt's
//! message gates from the projections' products past `packed_rows`, the GATE_MM=0 arm). Shape: gates x rows x
//! hidden. The kernel has one planted control (an element short) that gate 3 must catch.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const source = @import("../../mlx/kernels.zig").source;
const env = @import("../../env.zig");
const kernels = @import("kernels.zig");
const layers = @import("layers.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;
const U = host.U;

// ── Inkling 2 Mini's sizes ──

/// The stream width.
const D = 2304;
const RANK = 64;
/// A precise tanh's error relative to its value (4 ulps).
const TANH = 8 * U;
// ── shared ──

fn ci(x: usize) c_int {
    return @intCast(x);
}

fn none(_: Shape) usize {
    return 0;
}

/// `n` values of `e` with magnitudes in [lo, hi), either sign.
fn spreadIn(a: std.mem.Allocator, rng: std.Random, n: usize, e: spec.Elem, lo: f64, hi: f64) ![]f64 {
    const out = try a.alloc(f64, n);
    for (out) |*v| {
        const sign: f64 = if (rng.boolean()) 1 else -1;
        v.* = spec.roundTo(e, (lo + (hi - lo) * rng.float(f64)) * sign);
    }
    return out;
}

// ── gate, a prompt's rows ──

const gate_launch = metal.Launch{ .name = "msv_inkling2_gate", .inputs = &.{ "A", "MS", "ST", "F", "G" }, .outputs = &.{"Y"}, .source = source("plugins/inkling2/kernels/gate.metal"), .tg_bytes = 0, .m1 = 928, .m2 = 928 };

const gate_spec: spec.Spec = .{
    .name = "inkling2 gate (prompt rows, from the projections' products)",
    .accum = .none,
    .depth = none,
    .out = .f32,
    // The attention inputs' three gates and the MLP's one over a prompt chunk's rows, past `packed_rows` (the arm
    // GATE_MM=0 serves; with it on, `gate_mm` takes these rows).
    .served = &.{ .{ .rows = 3, .n = 65, .k = D }, .{ .rows = 1, .n = 100, .k = D }, .{ .rows = 3, .n = 128, .k = D }, .{ .rows = 1, .n = 512, .k = D } },
};

/// `gate.metal`'s PROJ arm: a prompt's loop message gates over the products F st and G st a matmul made (`kernels.gate`
/// past `packed_rows` with GATE_MM off): gates x rows x hidden, f32 throughout; the states' only use is their shape.
/// Y = (1 - tanh F) A + tanh(G) MS, no sums: checked within its own allowance (the precise tanh, the products, the add).
const GatePrompt = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &gate_launch,
    arena: std.heap.ArenaAllocator,
    /// The planted control: a grid one thread short, its last element never written.
    short: bool = false,
    shape: Shape = undefined,
    /// `[J, S, D]` each.
    a: []f64 = &.{},
    ms: []f64 = &.{},
    f: []f64 = &.{},
    g: []f64 = &.{},
    args: [5]mlx.mlx_array = undefined,
    msg: [256]u8 = undefined,

    fn init(al: std.mem.Allocator) GatePrompt {
        return .{ .rig = metal.Rig.init(al), .arena = .init(al) };
    }

    pub fn deinit(self: *GatePrompt) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const GatePrompt, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n <= kernels.packed_rows or shape.k == 0) return null;
        const n: c_int = ci(shape.rows * shape.n * shape.k);
        var geo = metal.Geometry{ .grid = .{ n - @intFromBool(self.short), 1, 1 }, .tg = .{ 64, 1, 1 } };
        geo.out(&.{ ci(shape.rows), ci(shape.n), ci(shape.k) }, .float32);
        geo.int("D", ci(shape.k));
        geo.int("RANK", @as(c_int, RANK));
        geo.int("PROJ", @as(c_int, 1));
        return geo;
    }

    pub fn prepare(self: *GatePrompt, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const al = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const j = shape.rows;
        const s = shape.n;
        const d = shape.k;
        const n = j * s * d;
        self.a = try spreadIn(al, rng, n, .f32, 0.01, 4);
        self.ms = try spreadIn(al, rng, n, .f32, 0.01, 4);
        // A gate's down projection over its two norms, times a forget or message row: within a few of zero.
        self.f = try spreadIn(al, rng, n, .f32, 0.01, 3);
        self.g = try spreadIn(al, rng, n, .f32, 0.01, 3);
        const st = try spreadIn(al, rng, j * s * RANK, .bf16, 0.01, 1);
        if (fill == .nonfinite) {
            host.plant(self.a, j * s, d);
            // An infinite forget product shuts its gate (the stream leaves); a NaN message product reaches its element.
            if (n > 5 * d) {
                self.f[4 * d + 1] = std.math.inf(f64);
                self.g[4 * d + 2] = std.math.nan(f64);
            }
        }
        self.rig.reset();
        const sh = [_]c_int{ ci(j), ci(s), ci(d) };
        self.args = .{
            try self.rig.input(.float32, &sh, self.a),
            try self.rig.input(.float32, &sh, self.ms),
            try self.rig.input(.bfloat16, &.{ ci(j), ci(s), RANK }, st),
            try self.rig.input(.float32, &sh, self.f),
            try self.rig.input(.float32, &sh, self.g),
        };
        return true;
    }

    pub fn run(self: *GatePrompt) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const GatePrompt) usize {
        return self.shape.rows * self.shape.n * self.shape.k;
    }

    /// Float64's outputs and what the device may differ by: each tanh's error, the one's complement, the two products
    /// and the add, and the floor a flushed subnormal product takes.
    fn emit(self: *const GatePrompt, want: []f64, allow: []f64) void {
        for (want, allow, 0..) |*w, *l, e| {
            const tf = std.math.tanh(self.f[e]);
            const tg = std.math.tanh(self.g[e]);
            const one = 1 - tf;
            const p1 = one * self.a[e];
            const p2 = tg * self.ms[e];
            const y = p1 + p2;
            w.* = y;
            l.* = @abs(self.a[e]) * (TANH * @abs(tf) + U * @abs(one)) + U * @abs(p1) + @abs(self.ms[e]) * TANH * @abs(tg) + U * @abs(p2) + U * @abs(y) + 3 * 0x1p-126;
        }
    }

    pub fn reference(self: *const GatePrompt, want: []f64, mag: []f64) !void {
        self.emit(want, mag);
    }

    /// The rule: within `emit`'s allowance, in the float64 output's class.
    pub fn check(self: *GatePrompt, at: spec.Gate) !?[]const u8 {
        const al = self.rig.a;
        const n = self.count();
        const buf = try al.alloc(f64, 3 * n);
        defer al.free(buf);
        self.emit(buf[0..n], buf[n .. 2 * n]);
        for (buf[2 * n ..], 0..) |*o, i| o.* = spec.read(.f32, self.rig.outputBytes(), i);
        return host.within(&self.msg, at, buf[2 * n ..], buf[0..n], buf[n .. 2 * n]);
    }

    /// The shipped wrapper (`kernels.gate`, GATE_MM off) over states and projections; the declared launch over the
    /// products made as the wrapper makes them: the same bytes.
    fn parity(self: *GatePrompt) !void {
        const ov = env.override(.MLX_SERVE_INKLING2_GATE_MM, "0");
        defer ov.restore();
        const j = self.shape.rows;
        const s = self.shape.n;
        const d = self.shape.k;
        const al = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(0x9a7e);
        const rng = prng.random();
        const st = try self.rig.input(.bfloat16, &.{ ci(j), ci(s), RANK }, try spreadIn(al, rng, j * s * RANK, .bf16, 0.01, 1));
        const fw = try self.rig.input(.bfloat16, &.{ ci(j), ci(d), RANK }, try spreadIn(al, rng, j * d * RANK, .bf16, 0.005, 0.2));
        const gw = try self.rig.input(.bfloat16, &.{ ci(j), ci(d), RANK }, try spreadIn(al, rng, j * d * RANK, .bf16, 0.005, 0.2));
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const gates: layers.Gates = .{ .down = st, .forget = fw, .message = gw };
        const theirs = try self.rig.a.dupe(u8, try self.rig.bytesOf(&.{try kernels.gate(&g, &gates, self.args[0], self.args[1], st, 0, ci(j))}));
        defer self.rig.a.free(theirs);
        const st32 = try g.astype(st, .float32);
        var prods: [2]mlx.mlx_array = undefined;
        for (&prods, [_]mlx.mlx_array{ fw, gw }) |*p, wt| p.* = try g.matmul(st32, try g.transpose(try g.astype(try g.sliceAxis(wt, 0, 0, ci(j)), .float32), &.{ 0, 2, 1 }));
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &.{ self.args[0], self.args[1], st, prods[0], prods[1] });
        try std.testing.expectEqualSlices(u8, theirs, self.rig.outputBytes());
    }
};

// ── registration ──

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(GatePrompt, .init(a), gate_spec, null);

    // The planted control, last (`checkWrappers` walks the kernels before it): the launch with one fault gate 3 must
    // catch.
    var gate = GatePrompt.init(a);
    gate.short = true;
    var gate_c = gate_spec;
    gate_c.name = "control inkling2 gate (prompt rows) a thread short (its last element never written)";
    gate_c.served = &.{.{ .rows = 3, .n = 100, .k = D }};
    try set.add(GatePrompt, gate, gate_c, .shapes);
}

/// `T`'s declared launch against its shipped wrapper at every served shape.
fn drift(comptime T: type, k: kc.gates.Kernel) !void {
    const p: *T = @ptrCast(@alignCast(k.ctx));
    for (k.spec.served) |shape| {
        try std.testing.expect(try p.prepare(shape, .random, 7));
        try p.run();
        try p.parity();
    }
}

/// The drift guard of `register`'s kernels, from the first (`items` starts there): each declared launch's bytes equal
/// the shipped wrapper's (`kernels.gate`).
pub fn checkWrappers(items: []const kc.gates.Kernel) !void {
    try drift(GatePrompt, items[0]);
}
