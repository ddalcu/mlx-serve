//! The rope area's kernel contract: the fused per-head QK-norm + RoPE kernels (hd 128 and hd 256), each bit-equal to
//! the composed chain it replaces (fast::rms_norm -> transpose -> fast::rope, optional mscale multiply). Listed in
//! `kernel_contract/registry.zig`. Served on every Apple GPU (M1 and up).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const qk = @import("qk_norm_rope.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

/// One launch per instantiation: M1/M2 compile each for its own thread limit (`offline.zig`).
fn launch128(comptime m: u32) metal.Launch {
    return .{
        .name = "mlxserve_qk_norm_rope",
        .inputs = &.{ "q", "k", "qw", "kw", "angles", "ms", "eps" },
        .outputs = &.{ "oq", "ok" },
        .source = qk.QK_NORM_ROPE_SOURCE,
        .tg_bytes = 0,
        .m1 = m,
        .m2 = m,
    };
}

/// Threadgroup bytes: `local_sums[32]` + `local_inv[1]` (f32) and the staged row `nrm[256]` (2-byte T).
fn launch256(comptime m: u32) metal.Launch {
    return .{
        .name = "mlxserve_qk_norm_rope_256",
        .inputs = &.{ "q", "k", "v", "qw", "kw", "vw", "angles", "eps" },
        .outputs = &.{ "oq", "ok", "ov" },
        .source = qk.QK_NORM_ROPE_256_SOURCE,
        .tg_bytes = 132 + 512,
        .m1 = m,
        .m2 = m,
    };
}

const EPS: f32 = 1e-6;
const BASE: f32 = 10000;
/// The decode position the random fills rotate at (gate 4's fill sits at 0, where the rotation is the identity).
const OFFSET: c_int = 4093;

/// One instantiation. hd 128: a shape's rows are the q heads (k heads fixed), one position. hd 256: a shape's rows
/// are the positions S; the head counts are fixed.
const Config = struct {
    wide: bool,
    dt: mlx.mlx_dtype = .bfloat16,
    hq: usize = 0,
    hk: usize,
    hv: usize = 0,
    rd: usize,
    scaled: bool = false,
    norm: bool = true,
};

const QkRope = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    cfg: Config,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    offset: c_int = 0,
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    qw: []f64 = &.{},
    kw: []f64 = &.{},
    vw: []f64 = &.{},
    angles: []f64 = &.{},
    ms: []f64 = &.{},
    args: [8]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, cfg: Config) QkRope {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .cfg = cfg, .arena = .init(a) };
    }

    pub fn deinit(self: *QkRope) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn hd(self: *const QkRope) usize {
        return if (self.cfg.wide) 256 else 128;
    }

    /// (q heads, k heads, v heads, positions) at `shape`.
    fn dims(self: *const QkRope, shape: Shape) [4]usize {
        const c = self.cfg;
        if (c.wide) return .{ c.hq, c.hk, c.hv, shape.rows };
        return .{ shape.rows, c.hk, 0, 1 };
    }

    pub fn geometry(self: *const QkRope, shape: Shape) ?metal.Geometry {
        const c = self.cfg;
        const d = self.dims(shape);
        const hq: c_int = @intCast(d[0]);
        const hk: c_int = @intCast(d[1]);
        const hv: c_int = @intCast(d[2]);
        const s: c_int = @intCast(d[3]);
        if (d[0] == 0) return null;
        if (!c.wide) {
            var g = metal.Geometry{ .grid = .{ (hq + hk) * 32, 1, 1 }, .tg = .{ 32, 1, 1 } };
            g.out(&.{ 1, hq, 1, 128 }, c.dt);
            g.out(&.{ 1, hk, 1, 128 }, c.dt);
            g.dtype("T", c.dt);
            g.int("HQ", hq);
            g.int("RD", c.rd);
            g.int("SCALED", @intFromBool(c.scaled));
            return g;
        }
        if (s < 1 or s > 32) return null;
        var g = metal.Geometry{ .grid = .{ (hq + hk + hv) * s * 64, 1, 1 }, .tg = .{ 64, 1, 1 } };
        g.out(&.{ 1, hq, s, 256 }, c.dt);
        g.out(&.{ 1, hk, s, 256 }, c.dt);
        g.out(&.{ 1, @max(hv, 1), if (hv > 0) s else 1, 256 }, c.dt);
        if (hv == 0) g.unread = 0b100;
        g.dtype("T", c.dt);
        g.int("HQ", hq);
        g.int("HK", hk);
        g.int("HV", hv);
        g.int("S", s);
        g.int("RD", c.rd);
        g.flag("NORM", c.norm);
        return g;
    }

    pub fn prepare(self: *QkRope, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const c = self.cfg;
        const d = self.dims(shape);
        const n = self.hd();
        const dt = c.dt;
        self.offset = if (fill == .stress) 0 else OFFSET;
        if (fill == .stress) {
            // Gate 4: integer rows whose squares sum past fp16's range, unit weights, the identity rotation.
            self.q = try host.ints(a, rng, d[0] * d[3] * n, -64, 64);
            self.k = try host.ints(a, rng, d[1] * d[3] * n, -64, 64);
            self.v = try host.ints(a, rng, @max(d[2], 1) * d[3] * n, -64, 64);
        } else {
            self.q = try host.uniform(a, rng, d[0] * d[3] * n, -3, 3, dt);
            self.k = try host.uniform(a, rng, d[1] * d[3] * n, -3, 3, dt);
            self.v = try host.uniform(a, rng, @max(d[2], 1) * d[3] * n, -3, 3, dt);
        }
        self.qw = try host.uniform(a, rng, n, 0.5, 1.5, dt);
        self.kw = try host.uniform(a, rng, n, 0.5, 1.5, dt);
        self.vw = try host.uniform(a, rng, n, 1, 1, dt);
        self.ms = try host.uniform(a, rng, n, 0.9, 1.2, dt);
        if (fill == .stress) for ([_][]f64{ self.qw, self.kw, self.ms }) |t| @memset(t, 1);
        if (fill == .nonfinite) {
            host.plant(self.q, d[0] * d[3], n);
            self.k[self.k.len / 2] = std.math.nan(f64);
        }
        self.rig.reset();
        // The angle table comes from the stock rope kernel, as the wrapper's callers build it.
        if (c.rd > 0) {
            const rows = try qk.ropeAngleRows(self.rig.stream(), @intCast(c.rd), .{ .value = BASE, .has_value = true }, .{ .ctx = null }, self.offset, @intCast(d[3]), self.rig.a, 1.0, 1.0);
            defer _ = mlx.mlx_array_free(rows);
            self.angles = try host.read(a, rows, self.rig.stream());
        } else self.angles = try host.uniform(a, rng, 8, 0, 0, .float32);
        try self.bind();
        return true;
    }

    fn bind(self: *QkRope) !void {
        const rig = &self.rig;
        const c = self.cfg;
        const d = self.dims(self.shape);
        const dt = c.dt;
        const n: c_int = @intCast(self.hd());
        const ang = try rig.input(.float32, &.{@intCast(self.angles.len)}, self.angles);
        const eps = try rig.scalar(EPS);
        if (!c.wide) {
            const q = try rig.input(dt, &.{ 1, 1, @intCast(d[0] * 128) }, self.q);
            const k = try rig.input(dt, &.{ 1, 1, @intCast(d[1] * 128) }, self.k);
            const qw = try rig.input(dt, &.{n}, self.qw);
            const kw = try rig.input(dt, &.{n}, self.kw);
            // The unscaled arm binds the q norm weight in the mscale slot, as the wrapper does.
            const ms = if (c.scaled) try rig.input(dt, &.{n}, self.ms) else qw;
            self.args[0..7].* = .{ q, k, qw, kw, ang, ms, eps };
            return;
        }
        const s: c_int = @intCast(d[3]);
        const q = try rig.input(dt, &.{ 1, s, @intCast(d[0]), n }, self.q);
        const k = try rig.input(dt, &.{ 1, s, @intCast(d[1]), n }, self.k);
        const v = if (d[2] > 0) try rig.input(dt, &.{ 1, s, @intCast(d[2]), n }, self.v) else k;
        const qw = if (c.norm) try rig.input(dt, &.{n}, self.qw) else q;
        const kw = if (c.norm) try rig.input(dt, &.{n}, self.kw) else k;
        const vw = if (d[2] > 0) try rig.input(dt, &.{n}, self.vw) else k;
        self.args = .{ q, k, v, qw, kw, vw, ang, eps };
    }

    fn nargs(self: *const QkRope) usize {
        return if (self.cfg.wide) 8 else 7;
    }

    pub fn run(self: *QkRope) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.nargs()]);
    }

    pub fn count(self: *const QkRope) usize {
        const d = self.dims(self.shape);
        return (d[0] + d[1] + d[2]) * d[3] * self.hd();
    }

    /// Every output row (q heads, then k, then v; head-major, positions inside), float64 with the sum of squares
    /// run in `acc` (null: float64) in `order`.
    fn outputsIn(self: *const QkRope, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const c = self.cfg;
        const d = self.dims(self.shape);
        const n = self.hd();
        const half = c.rd / 2;
        const sq = try self.rig.a.alloc(f64, n);
        defer self.rig.a.free(sq);
        const nrm = try self.rig.a.alloc(f64, n);
        defer self.rig.a.free(nrm);
        var at: usize = 0;
        for (0..3) |which| for (0..d[which]) |h| for (0..d[3]) |p| {
            const src = switch (which) {
                0 => self.q[(p * d[0] + h) * n ..][0..n],
                1 => self.k[(p * d[1] + h) * n ..][0..n],
                else => self.v[(p * d[2] + h) * n ..][0..n],
            };
            const w = switch (which) {
                0 => self.qw,
                1 => self.kw,
                else => self.vw,
            };
            for (sq, src) |*o, x| o.* = x * x;
            const ss = if (acc) |ac| spec.sumIn(ac, order, sq) else host.sum(sq);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(n)) + EPS);
            for (nrm, src, w) |*o, x, wi| o.* = if (c.norm or which == 2) wi * x * inv else x;
            const out = want[at..][0..n];
            const row = self.angles[p * c.rd ..];
            for (out, 0..) |*o, j| {
                o.* = if (which == 2 or j >= c.rd) nrm[j] else if (j < half)
                    nrm[j] * row[j] - nrm[j + half] * row[half + j]
                else
                    nrm[j - half] * row[j] + nrm[j] * row[j - half];
                if (c.scaled) o.* *= self.ms[j];
            }
            if (mag) |m| for (m[at..][0..n], out) |*o, y| {
                o.* = @abs(y);
            };
            at += n;
        };
    }

    pub fn reference(self: *const QkRope, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const QkRope, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }

    /// The composed chain on the same inputs: per head rms_norm, transpose to head-major, rope, mscale.
    pub fn replaced(self: *QkRope) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const c = self.cfg;
        const d = self.dims(self.shape);
        const x = self.args;
        const n: c_int = @intCast(self.hd());
        var outs: [3]mlx.mlx_array = undefined;
        const parts: usize = if (d[2] > 0) 3 else 2;
        for (0..parts) |which| {
            const nh: c_int = @intCast(d[which]);
            const src = if (c.wide) x[which] else try g.reshape(x[which], &.{ 1, 1, nh, n });
            const w = if (c.wide) x[3 + which] else x[2 + which];
            const normed = if (c.norm or which == 2) try g.rmsNorm(src, w, EPS) else src;
            const t = try g.transpose(normed, &.{ 0, 2, 1, 3 });
            const roped = if (which == 2 or c.rd == 0) t else try g.rope(t, @intCast(c.rd), BASE, self.offset);
            outs[which] = if (c.scaled) try g.mul(roped, x[5]) else roped;
        }
        return self.rig.bytesOf(outs[0..parts]);
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *QkRope) ![]const u8 {
        const c = self.cfg;
        const d = self.dims(self.shape);
        const x = self.args;
        const s = self.rig.stream();
        const none = mlx.mlx_array{ .ctx = null };
        if (!c.wide) {
            const got = (try qk.fusedQkNormRope(s, x[0], x[1], x[2], x[3], x[4], if (c.scaled) x[5] else null, x[6], @intCast(d[0]), @intCast(d[1]), @intCast(c.rd))) orelse return error.Declined;
            defer for (got) |o| {
                _ = mlx.mlx_array_free(o);
            };
            return self.rig.bytesOf(&got);
        }
        const with_v = d[2] > 0;
        const got = (try qk.fusedQkNormRope256(s, x[0], x[1], if (with_v) x[2] else none, if (c.norm) x[3] else none, if (c.norm) x[4] else none, if (with_v) x[5] else none, x[6], x[7], @intCast(d[0]), @intCast(d[1]), @intCast(d[3]), @intCast(c.rd))) orelse return error.Declined;
        defer for (got) |o| if (o.ctx != null) {
            _ = mlx.mlx_array_free(o);
        };
        return self.rig.bytesOf(if (with_v) got[0..3] else got[0..2]);
    }
};

fn headDim(shape: Shape) usize {
    return shape.k;
}

fn qkSpec(comptime name: []const u8, cfg: Config, served: []const Shape) spec.Spec {
    const norm_mids: f64 = if (cfg.norm) 2 else 0;
    return .{
        .name = name,
        .accum = if (cfg.norm) .f32 else .none,
        .depth = headDim,
        // The mean, + eps, rsqrt (2), the product with the row; the rotation's two products and its add.
        .ops = 8,
        // T(x * inv) and the weight product; the rotation's own rounding before an mscale multiply.
        .mids = norm_mids + @as(f64, if (cfg.scaled) 1 else 0),
        .out = host.elemOf(cfg.dt),
        .replaces = if (cfg.scaled) "rms_norm -> transpose -> rope -> mscale multiply" else if (cfg.rd == 0) "rms_norm -> transpose" else if (cfg.norm) "rms_norm -> transpose -> rope" else "transpose -> rope",
        .served = served,
        // A 128- or 256-term sum of squares: summed in bf16 it stays near the bf16 output's own rounding.
        // fp16 sums overflow (the output reads 0); the bound's mid roundings cap a bf16 output's margin at 85x.
        .sums = .{ .f16 = if (cfg.dt == .bfloat16) 50 else 100, .bf16 = 0, .why = "positive squares: a bf16 sum stays near the output's rounding" },
    };
}

/// hd 128: rows = q heads.
fn heads(comptime list: []const usize) []const Shape {
    comptime var out: [list.len]Shape = undefined;
    inline for (list, 0..) |r, i| out[i] = .{ .rows = r, .n = 1, .k = 128 };
    const final = out;
    return &final;
}

/// hd 256: rows = positions.
fn positions(comptime list: []const usize) []const Shape {
    comptime var out: [list.len]Shape = undefined;
    inline for (list, 0..) |r, i| out[i] = .{ .rows = r, .n = 1, .k = 256 };
    const final = out;
    return &final;
}

const qwen_rd32 = Config{ .wide = false, .hk = 2, .rd = 32 };
const laguna_yarn = Config{ .wide = false, .hk = 4, .rd = 64, .scaled = true };
const sliding_rd128 = Config{ .wide = false, .hk = 2, .rd = 128 };
const nope_rd0 = Config{ .wide = false, .hk = 4, .rd = 0 };
const qwen35_bf16 = Config{ .wide = true, .hq = 24, .hk = 4, .rd = 64 };
const qwen35_f16 = Config{ .wide = true, .dt = .float16, .hq = 24, .hk = 4, .rd = 64 };
const gemma_v_rd256 = Config{ .wide = true, .hq = 8, .hk = 2, .hv = 2, .rd = 256 };
const gemma_v_rd64 = Config{ .wide = true, .hq = 8, .hk = 2, .hv = 2, .rd = 64 };
const spark_rope_only = Config{ .wide = true, .hq = 16, .hk = 4, .rd = 256, .norm = false };

const registrations = .{
    .{ "qk_norm_rope bf16 (rd 32)", qwen_rd32, heads(&.{ 8, 16 }), launch128(832) },
    .{ "qk_norm_rope bf16 (rd 64, YaRN mscale)", laguna_yarn, heads(&.{ 32, 8 }), launch128(928) },
    .{ "qk_norm_rope bf16 (rd 128)", sliding_rd128, heads(&.{ 8, 32 }), launch128(832) },
    .{ "qk_norm_rope bf16 (rd 0, NoPE)", nope_rd0, heads(&.{ 32, 4 }), launch128(1024) },
    .{ "qk_norm_rope_256 bf16 (24q/4kv, rd 64)", qwen35_bf16, positions(&.{ 1, 3, 8, 16, 32 }), launch256(1024) },
    .{ "qk_norm_rope_256 f16 (24q/4kv, rd 64)", qwen35_f16, positions(&.{ 1, 9, 32 }), launch256(1024) },
    .{ "qk_norm_rope_256 bf16 (8q/2kv + v, rd 256)", gemma_v_rd256, positions(&.{1}), launch256(1024) },
    .{ "qk_norm_rope_256 bf16 (8q/2kv + v, rd 64)", gemma_v_rd64, positions(&.{1}), launch256(1024) },
    .{ "qk_norm_rope_256 bf16 (rope only, rd 256)", spark_rope_only, positions(&.{1}), launch256(1024) },
};

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    inline for (registrations) |r| {
        const l = &struct {
            const launch = r[3];
        }.launch;
        try set.add(QkRope, .init(a, l, r[1]), qkSpec(r[0], r[1], r[2]), null);
    }
}

// ── tests ──

test "kernel contract rope: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    qk.qk_fused_override = true;
    defer qk.qk_fused_override = null;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const t: *QkRope = @ptrCast(@alignCast(k.ctx));
        try std.testing.expect(try t.prepare(k.spec.served[0], .random, 7));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        const theirs = try t.wrapper();
        if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{k.spec.name});
        try std.testing.expectEqualSlices(u8, theirs, ours);
    }
}
