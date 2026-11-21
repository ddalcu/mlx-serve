//! Inkling 2's kernel contract: `relative_attention` (prefill attention with a relative position bias and a sink
//! logit per query) and `depth_attention` (learned queries over the block states), each against the float64 formula
//! of the composed path it stands in for (`attention.referenceAttention`, `depth.attendComposed`), with the error an
//! online softmax and the kernels' half tiles allow. Listed in `kernel_contract/registry.zig`.
//! Shapes: relative_attention `rows` = L (queries), `n` = S (keys in the view), `k` = head dim, with 24 q heads,
//! 8 kv heads, R 1024, bf16 and the window fixed per registration. depth_attention `rows` = the partial state's rows
//! (L), `n` = K (block states), `k` = D (2304), with Q (1, 3, 4) fixed per registration.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const attn = @import("../../sdk/blocks/attention/contract.zig");
const kernels = @import("kernels.zig");
const env = @import("../../env.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const inf = std.math.inf(f64);
const LOG2E = 1.4426950408889634;
const Q23 = 0x1p-23;

fn ci(x: usize) c_int {
    return @intCast(x);
}

/// Gate 4 rows: dots of 0, -mag or -2 mag, exact in f32 in any order, whose partial sums leave fp16's range and bf16's
/// precision; neighbouring rows' dots differ, so a stalled bf16 sum cannot pass.
fn stressRows(q: []f64, x: []f64, d: usize, mag: f64, first: usize) void {
    const half = d / 2;
    for (q, 0..) |*v, i| v.* = if (i % d < half) mag else -mag;
    var r: usize = 0;
    while (r < x.len) : (r += d) {
        const ones = (first + r / d) % 3;
        for (0..half) |i| x[r + half + i] = x[r + (i + 1) % half] + @as(f64, if (i < ones) 1 else 0);
    }
}

// ── relative_attention ──

const HQ = 24;
const HKV = 8;
const HD = 128;
const R = 1024;
const BQ = kernels.BQ;
const BK = kernels.BK;
const TG = kernels.TG;
/// Gate 4's q: the largest 8-bit integer, exact in bf16 and half.
const STRESS_Q = 251;

const rel_launch = metal.Launch{
    .name = "inkling2_relative_attention",
    .inputs = &.{ "q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale" },
    .outputs = &.{"out"},
    .source = kernels.RELATIVE_ATTENTION_SOURCE,
    .contiguous = false,
    .tg_bytes = kernels.relTgBytes(HD),
    // `offline.zig`: every instantiation builds for 576 threads on M1 and M2; the launch is 128.
    .m1 = 576,
    .m2 = 576,
};

/// `n` bf16 values in [-2, 2) with those under half's normal range flushed to 0: the kernel holds q, k and v in half
/// tiles, exact for these (served q and k are clipped, v has unit RMS).
fn halfExact(a: std.mem.Allocator, rng: std.Random, n: usize) ![]f64 {
    const out = try host.uniform(a, rng, n, -2, 2, .bfloat16);
    for (out) |*v| if (@abs(v.*) < 0x1p-14) {
        v.* = 0;
    };
    return out;
}

const Rel = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    window: u32,
    /// The WINDOW the launch is built with: `window`, but for the planted control.
    launch_window: u32,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    pb: []f64 = &.{},
    sinks: []f64 = &.{},
    key_start: i32 = 0,
    scale: f64 = 0,
    args: [8]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, window: u32, launch_window: u32) Rel {
        return .{ .rig = metal.Rig.init(a), .launch = &rel_launch, .window = window, .launch_window = launch_window, .arena = .init(a) };
    }

    pub fn deinit(self: *Rel) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Rel, shape: Shape) ?metal.Geometry {
        if (shape.k != HD or shape.rows < 2 or shape.n < shape.rows) return null;
        const l = ci(shape.rows);
        var g = metal.Geometry{ .grid = .{ @divTrunc(l + BQ - 1, BQ) * TG, HQ, 1 }, .tg = .{ TG, 1, 1 } };
        g.out(&.{ 1, l, HQ, HD }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("D", HD);
        g.int("HQ", HQ);
        g.int("HKV", HKV);
        g.int("R", R);
        g.int("WINDOW", self.launch_window);
        g.int("BQ", BQ);
        g.int("TG", TG);
        return g;
    }

    pub fn prepare(self: *Rel, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const l = shape.rows;
        const s = shape.n;
        if (fill == .stress) {
            self.q = try a.alloc(f64, HQ * l * HD);
            self.k = try host.ints(a, rng, HKV * s * HD, 0, 127);
            stressRows(self.q, self.k, HD, STRESS_Q, 0);
        } else {
            self.q = try halfExact(a, rng, HQ * l * HD);
            self.k = try halfExact(a, rng, HKV * s * HD);
        }
        self.v = try halfExact(a, rng, HKV * s * HD);
        self.pb = try host.uniform(a, rng, l * HQ * R, -2, 2, .bfloat16);
        if (fill == .stress) @memset(self.pb, 0);
        self.sinks = try host.uniform(a, rng, l * HQ, -1, 3, .float32);
        if (fill == .nonfinite) {
            // q: head 0's first four queries. k: kv head 0's last four keys (scores the last queries see); v: kv head
            // 1's last four keys (read by every row of the last block, a hidden key through a zero probability).
            // The bias: query 2's heads 6..9. The sinks: queries 4.. .
            host.plant(self.q, HQ * l, HD);
            host.plant(self.k[(s - 4) * HD ..], 4, HD);
            host.plant(self.v[(s + s - 4) * HD ..], 4, HD);
            host.plant(self.pb[(2 * HQ + 6) * R ..], 4, R);
            host.plant(self.sinks[4 * HQ ..], l - 4, HQ);
        }
        // A sliding view starts past the history it dropped (key_start = S - total < 0); a global one at 0.
        self.key_start = if (self.window > 0) -7 else 0;
        self.scale = host.rt(.float32, 1 / @sqrt(@as(f64, HD)));
        const rig = &self.rig;
        rig.reset();
        self.args = .{
            try rig.input(.bfloat16, &.{ 1, HQ, ci(l), HD }, self.q),
            try rig.input(.bfloat16, &.{ 1, HKV, ci(s), HD }, self.k),
            try rig.input(.bfloat16, &.{ 1, HKV, ci(s), HD }, self.v),
            try rig.input(.bfloat16, &.{ 1, ci(l), HQ, R }, self.pb),
            try rig.input(.float32, &.{ 1, ci(l), HQ }, self.sinks),
            try rig.input(.int32, &.{1}, &.{@floatFromInt(self.key_start)}),
            try rig.input(.int32, &.{2}, &.{ @floatFromInt(l), @floatFromInt(s) }),
            try rig.scalar(@floatCast(self.scale)),
        };
        return true;
    }

    pub fn run(self: *Rel) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Rel) usize {
        return self.shape.rows * HQ * HD;
    }

    /// Each row's online softmax over the key blocks its query tile walks (every key of a block, a hidden one with a
    /// zero probability), from its sink logit; scores in log2 units, P rounded to half.
    fn emit(self: *const Rel, want: []f64, mag: ?[]f64, sim: attn.Sim) !void {
        const a = self.rig.a;
        const l = self.shape.rows;
        const s = self.shape.n;
        const keys = try a.alloc(attn.Key, (s / BK + 2) * BK);
        defer a.free(keys);
        const ends = try a.alloc(usize, s / BK + 2);
        defer a.free(ends);
        const zeros = try a.alloc(f64, HD);
        defer a.free(zeros);
        @memset(zeros, 0);
        var terms: [HD]f64 = undefined;
        var o: [HD]f64 = undefined;
        var eo: [HD]f64 = undefined;
        const l2e = host.rt(.float32, LOG2E);
        const qscale = host.rt(.float32, self.scale * l2e);
        const window: i64 = self.window;
        const start: i64 = self.key_start;
        const off: i64 = @intCast(s - l);
        for (0..HQ) |h| for (0..l) |row| {
            const hk = h / (HQ / HKV);
            const q0 = row / BQ * BQ;
            const first_diag = off + @as(i64, @intCast(q0));
            const last_diag = off + @as(i64, @intCast(@min(q0 + BQ, l))) - 1;
            const first_key = @max(if (window > 0) @max(0, first_diag - window + 1) else 0, start);
            const diag = off + @as(i64, @intCast(row));
            const qrow = self.q[(h * l + row) * HD ..][0..HD];
            var n: usize = 0;
            var nb: usize = 0;
            var k0 = @divTrunc(first_key, BK) * BK;
            while (k0 <= last_diag) : (k0 += BK) {
                for (0..BK) |j| {
                    const key = k0 + @as(i64, @intCast(j));
                    const d = diag - key;
                    const ku: usize = @intCast(key);
                    const vrow = if (ku < s) self.v[(hk * s + ku) * HD ..][0..HD] else zeros;
                    const visible = d >= 0 and key >= start and (window == 0 or d < window);
                    keys[n] = if (!visible) .{ .s = -inf, .on = false, .v = vrow } else blk: {
                        const dt = attn.dot(qrow, self.k[(hk * s + ku) * HD ..][0..HD], qscale, HD, 2, sim, &terms);
                        const b = if (d < R) self.pb[(row * HQ + h) * R + @as(usize, @intCast(d))] * l2e else 0;
                        const sc = dt.v + b;
                        break :blk .{ .s = sc, .e = dt.e + host.U * (@abs(b) + @abs(sc)), .v = vrow };
                    };
                    n += 1;
                }
                ends[nb] = n;
                nb += 1;
            }
            const m0 = host.rt(.float32, self.sinks[row * HQ + h] * l2e);
            const st = attn.online(.{ .base2 = true, .p_unit = 0x1p-11, .p_abs = 0x1p-25 }, .{ .m = m0, .e = host.U * @abs(m0), .l = 1 }, keys[0..n], ends[0..nb], &o, &eo);
            attn.normalize(st, &o, &eo);
            const at = (row * HQ + h) * HD;
            @memcpy(want[at..][0..HD], &o);
            if (mag) |m| for (m[at..][0..HD], eo) |*x, e| {
                x.* = host.allowance(e);
            };
        };
    }

    pub fn reference(self: *const Rel, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Rel, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *Rel) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        const y = (try kernels.relativeAttention(&g, x[0], x[1], x[2], x[3], x[4], @floatCast(self.scale), self.window, self.key_start)) orelse return error.Declined;
        return self.rig.bytesOf(&.{y});
    }
};

fn relSpec(comptime name: []const u8, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = host.depthOne,
        .out = .bf16,
        // L in query tiles (L = 1 is the decode path, which the wrapper declines), S in key blocks.
        .tiles = .{ .{ .size = BQ, .min = 2 }, .{ .size = BK }, .{} },
        .served = served,
        .sums = .{ .exact = false },
    };
}

// ── depth_attention ──

const DIM = 2304;
/// A dot's f32 chain: a thread's D / TG strided terms, the simdgroup's tree of 32, then the simdgroups in turn.
const DEPTH_DOT = DIM / kernels.rowThreads(DIM) + 5 + kernels.rowThreads(DIM) / 32;
/// Gate 4's query: q x stays under 2^24 summed over D (64 x 127 x 2304 / 2).
const STRESS_DEPTH_Q = 64;
/// Most block states a registration serves (the coda's final read: one block per boundary layer).
const MAX_K = 4;

const depth_launch = metal.Launch{
    .name = "inkling2_depth_attention",
    .inputs = &.{ "blocks", "partial", "queries", "scale" },
    .outputs = &.{"out"},
    .source = kernels.DEPTH_ATTENTION_SOURCE,
    .tg_bytes = kernels.depthTgBytes(MAX_K, 4, kernels.rowThreads(DIM)),
    // `offline.zig`: the least over the instantiations (Q 4, K 4) is 576 threads on M1 and M2; the launch is 288.
    .m1 = 576,
    .m2 = 576,
};

fn fmax(a: f64, b: f64) f64 {
    if (std.math.isNan(a)) return b;
    if (std.math.isNan(b)) return a;
    return @max(a, b);
}

const Depth = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    nq: usize,
    /// The planted control: the launch's K one less than the blocks bound.
    short: bool,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    blocks: []f64 = &.{},
    partial: []f64 = &.{},
    queries: []f64 = &.{},
    scale: f64 = 0,
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, nq: usize, short: bool) Depth {
        return .{ .rig = metal.Rig.init(a), .launch = &depth_launch, .nq = nq, .short = short, .arena = .init(a) };
    }

    pub fn deinit(self: *Depth) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const Depth, shape: Shape) ?metal.Geometry {
        if (shape.k != DIM or shape.n < 1 or shape.rows < 1) return null;
        const tg = ci(kernels.rowThreads(DIM));
        var g = metal.Geometry{ .grid = .{ ci(shape.rows) * tg, 1, 1 }, .tg = .{ tg, 1, 1 } };
        g.out(&.{ 1, ci(shape.rows), ci(self.nq), DIM }, .float32);
        g.int("K", shape.n - @intFromBool(self.short));
        g.int("D", DIM);
        g.int("Q", self.nq);
        g.int("TG", tg);
        return g;
    }

    pub fn prepare(self: *Depth, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const l = shape.rows;
        const nk = shape.n;
        if (fill == .stress) {
            self.queries = try a.alloc(f64, self.nq * DIM);
            self.blocks = try host.ints(a, rng, nk * l * DIM, 0, 127);
            self.partial = try host.ints(a, rng, l * DIM, 0, 127);
            // Source i of a row starts its count at i: a row's sources never share a dot with their neighbour.
            for (0..nk) |i| stressRows(self.queries, self.blocks[i * l * DIM ..][0 .. l * DIM], DIM, STRESS_DEPTH_Q, i);
            stressRows(self.queries, self.partial, DIM, STRESS_DEPTH_Q, nk);
        } else {
            self.queries = try host.uniform(a, rng, self.nq * DIM, -2, 2, .float32);
            self.blocks = try host.uniform(a, rng, nk * l * DIM, -2, 2, .float32);
            self.partial = try host.uniform(a, rng, l * DIM, -2, 2, .float32);
        }
        if (fill == .nonfinite) host.plant(self.partial, l, DIM);
        self.scale = host.rt(.float32, 1 / @sqrt(@as(f64, DIM)));
        const rig = &self.rig;
        rig.reset();
        self.args = .{
            try rig.input(.float32, &.{ ci(nk), 1, ci(l), DIM }, self.blocks),
            try rig.input(.float32, &.{ 1, ci(l), DIM }, self.partial),
            try rig.input(.float32, &.{ ci(self.nq), DIM }, self.queries),
            try rig.scalar(@floatCast(self.scale)),
        };
        return true;
    }

    pub fn run(self: *Depth) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Depth) usize {
        return self.shape.rows * self.nq * DIM;
    }

    fn source(self: *const Depth, i: usize, row: usize) []const f64 {
        const l = self.shape.rows;
        return if (i < self.shape.n) self.blocks[(i * l + row) * DIM ..][0..DIM] else self.partial[row * DIM ..][0..DIM];
    }

    /// Per row and query: f32 logits (the dot, then the scale), a softmax over the K + 1 sources (max ignoring NaN as
    /// Metal's does, a fast exp, a divide), and the weighted sum of the sources, each in the kernel's order.
    fn emit(self: *const Depth, want: []f64, mag: ?[]f64, sim: attn.Sim) !void {
        const n = self.shape.n + 1;
        var terms: [DIM]f64 = undefined;
        var lg: [MAX_K + 1]host.Val = undefined;
        var p: [MAX_K + 1]f64 = undefined;
        var rp: [MAX_K + 1]f64 = undefined;
        const sums = spec.gamma(@floatFromInt(n), host.U);
        for (0..self.shape.rows) |row| for (0..self.nq) |j| {
            const qj = self.queries[j * DIM ..][0..DIM];
            var m = -inf;
            var em: f64 = 0;
            for (lg[0..n], 0..) |*x, i| {
                x.* = attn.dot(self.source(i, row), qj, self.scale, DEPTH_DOT, 1, sim, &terms);
                m = fmax(m, x.v);
                em = @max(em, x.e);
            }
            var z: f64 = 0;
            var rz: f64 = 0;
            for (lg[0..n], p[0..n], rp[0..n]) |x, *pi, *ri| {
                const span = @abs(x.v - m);
                pi.* = @exp(x.v - m);
                ri.* = std.math.expm1(x.e + em + host.U * span) + host.fastExp(span);
                z += pi.*;
                rz = @max(rz, ri.*);
            }
            rz += sums;
            for (p[0..n], rp[0..n]) |*pi, *ri| {
                pi.* /= z;
                ri.* += rz + 2.5 * Q23;
            }
            const at = (row * self.nq + j) * DIM;
            for (0..DIM) |d| {
                var y: f64 = 0;
                var e: f64 = 0;
                for (p[0..n], rp[0..n], 0..) |w, rw, i| {
                    const t = w * self.source(i, row)[d];
                    y += t;
                    e += @abs(t) * (rw + sums);
                }
                want[at + d] = y;
                if (mag) |mg| mg[at + d] = host.allowance(e);
            }
        };
    }

    pub fn reference(self: *const Depth, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Depth, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *Depth) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        const y = (try kernels.depthAttention(&g, x[0], x[1], x[2], @floatCast(self.scale))) orelse return error.Declined;
        return self.rig.bytesOf(&.{y});
    }
};

fn depthSpec(comptime name: []const u8, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f32,
        // One threadgroup a row.
        .tiles = .{ .{ .size = 1 }, .{}, .{} },
        .served = served,
        .sums = .{ .exact = false },
    };
}

/// Prefill chunks of one tile and of many, each block count the trunk reads (1 to 4).
const depth_served: []const Shape = &.{
    .{ .rows = 1, .n = 1, .k = DIM },
    .{ .rows = 32, .n = 2, .k = DIM },
    .{ .rows = 512, .n = 3, .k = DIM },
    .{ .rows = 32, .n = 4, .k = DIM },
};

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    // The band view a sliding visit hands over (window + L - 1 keys, and a longer one), and the global visits' views.
    try set.add(Rel, .init(a, 1024, 1024), relSpec("inkling2 relative_attention bf16 (24/8 heads, hd 128, R 1024, window 1024)", &.{
        .{ .rows = 32, .n = 1055, .k = HD },
        .{ .rows = 512, .n = 1535, .k = HD },
    }), null);
    try set.add(Rel, .init(a, 0, 0), relSpec("inkling2 relative_attention bf16 (24/8 heads, hd 128, R 1024, global)", &.{
        .{ .rows = 32, .n = 32, .k = HD },
        .{ .rows = 512, .n = 512, .k = HD },
        .{ .rows = 512, .n = 1024, .k = HD },
    }), null);
    inline for (.{ 1, 3, 4 }) |nq| {
        try set.add(Depth, .init(a, nq, false), depthSpec(std.fmt.comptimePrint("inkling2 depth_attention f32 (D 2304, Q {d})", .{nq}), depth_served), null);
    }
    try set.add(Rel, .init(a, 1024, 0), relSpec("control inkling2 relative_attention launched without its band", &.{.{ .rows = 32, .n = 1535, .k = HD }}), .shapes);
    try set.add(Depth, .init(a, 1, true), depthSpec("control inkling2 depth_attention reading one block short", &.{.{ .rows = 32, .n = 2, .k = DIM }}), .shapes);
}

// ── tests ──

test "kernel contract inkling2: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const on = env.override(.MLX_SERVE_INKLING2_KERNELS, null);
    defer on.restore();
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    var checked: usize = 0;
    for (set.kernels.items) |k| {
        if (k.control != null) continue;
        const ours, const theirs = if (k.vt.launch.?(k.ctx) == &rel_launch) blk: {
            const t: *Rel = @ptrCast(@alignCast(k.ctx));
            try std.testing.expect(try t.prepare(k.spec.served[0], .random, 7));
            try t.run();
            const ours = try a.dupe(u8, t.rig.outputBytes());
            errdefer a.free(ours);
            break :blk .{ ours, try t.wrapper() };
        } else blk: {
            const t: *Depth = @ptrCast(@alignCast(k.ctx));
            try std.testing.expect(try t.prepare(k.spec.served[0], .random, 7));
            try t.run();
            const ours = try a.dupe(u8, t.rig.outputBytes());
            errdefer a.free(ours);
            break :blk .{ ours, try t.wrapper() };
        };
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, theirs, ours);
        checked += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), checked);
}
