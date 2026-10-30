//! The attention area's kernel contract. The output gates (the sigmoid gate by channel or by head, the strided sigmoid gate over a transposed output and a split projection) are each bit-equal
//! to the MLX op chain they replace. The attention kernels meet the float64 bound of an online softmax simulated
//! block by block as each kernel walks its keys (`online`: fast-math exp, the rescales, P rounded where the kernel
//! rounds it): msv_attn_p256 (causal, band, QSA mask, MiMo's 192/128 with sinks, and the carried-state chunk
//! passes), the decode windows (row_attn and lane_attn: partial, tail, merge), the packed-KV kernels (dec, verify,
//! the matmul2d page kernel) and the NAX prefill kernel (one pass, key-range passes, the int8 Q/K entry). Rows are
//! KV heads. M1 and up but the tensor-unit kernels (M4 / M5, declared per launch). Known issue:
//! msv_nax_attn_256_q8's scale inputs (`q8_scales_known`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const gates = @import("gates.zig");
const attn256 = @import("attn256.zig");
const row_attn = @import("row_attn.zig");
const lane_attn = @import("lane_attn.zig");
const kvq = @import("kv_quant_attn.zig");
const nax = @import("nax_attention.zig");
const int8_attn = @import("../dit/int8_attn.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const kv_view = @import("kv_view.zig");
const qsa_score = @import("kv_quant_attn.zig");
const fused_activations = @import("../mlp/fused_activations.zig");
const device = @import("../../../mlx/device.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

// ── output gates ──

const GateKind = enum { table, strided };

/// One gate instantiation: `heads` x `hd` channels a token; `headwise` gates a head by one value.
const GateCfg = struct { kind: GateKind, dt: mlx.mlx_dtype = .bfloat16, heads: usize, hd: usize, headwise: bool = false };

const table_launch = metal.Launch{
    .name = "mlxserve_sigmoid_gate_mul",
    .inputs = &.{ "x", "gate", "sigtab", "N_size" },
    .outputs = &.{"y"},
    .source = gates.SIGMOID_GATE_MUL_SOURCE,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

const strided_launch = metal.Launch{
    .name = "mlxserve_strided_sigmoid_gate_mul",
    .inputs = &.{ "x", "gate", "sigtab", "N_size" },
    .outputs = &.{"y"},
    .source = gates.STRIDED_SIGMOID_GATE_MUL_SOURCE,
    .contiguous = false,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

/// The f16 instantiation compiles for fewer threads on M1/M2 than the bf16 one.
const strided_f16_launch = blk: {
    var l = strided_launch;
    l.m1 = 928;
    l.m2 = 928;
    break :blk l;
};

/// A shape's rows are the tokens of one sequence; n = heads x hd.
const Gate = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    cfg: GateCfg,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    g: []f64 = &.{},
    args: [4]mlx.mlx_array = undefined,
    /// The strided arm's views (a transposed output, the gate half of a split projection), freed at the next prepare.
    views: [2]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null } },

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, cfg: GateCfg) Gate {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .cfg = cfg, .arena = .init(a) };
    }

    fn freeViews(self: *Gate) void {
        for (&self.views) |*v| if (v.ctx != null) {
            _ = mlx.mlx_array_free(v.*);
            v.* = .{ .ctx = null };
        };
    }

    pub fn deinit(self: *Gate) void {
        self.freeViews();
        self.rig.deinit();
        self.arena.deinit();
    }

    fn width(self: *const Gate) usize {
        return self.cfg.heads * self.cfg.hd;
    }

    /// Gate values a token: one per head (headwise) or one per channel.
    fn gateWidth(self: *const Gate) usize {
        return if (self.cfg.headwise) self.cfg.heads else self.width();
    }

    pub fn geometry(self: *const Gate, shape: Shape) ?metal.Geometry {
        const c = self.cfg;
        const s: c_int = @intCast(shape.rows);
        const w: c_int = @intCast(self.width());
        switch (c.kind) {
            .table, .strided => {
                const n = s * w;
                const tg: c_int = if (c.kind == .table) @min(256, @max(32, @divTrunc(n + 31, 32) * 32)) else @min(256, @divTrunc(n + 31, 32) * 32);
                var g = metal.Geometry{ .grid = .{ @divTrunc(n + tg - 1, tg) * tg, 1, 1 }, .tg = .{ tg, 1, 1 } };
                if (c.kind == .table) {
                    g.out(&.{ 1, s, w }, c.dt);
                    g.dtype("T", c.dt);
                    g.int("GROUP", if (c.headwise) c.hd else 1);
                } else {
                    g.out(&.{ 1, s, @intCast(c.heads), @intCast(c.hd) }, c.dt);
                    g.dtype("T", c.dt);
                }
                return g;
            },
        }
    }

    pub fn prepare(self: *Gate, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        self.freeViews();
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const c = self.cfg;
        const rows = shape.rows;
        // Gate logits past sigmoid's saturation.
        self.x = try host.uniform(a, rng, rows * self.width(), -3, 3, c.dt);
        self.g = try host.uniform(a, rng, rows * self.gateWidth(), -40, 40, c.dt);
        if (fill == .nonfinite) {
            host.plant(self.x, rows, self.width());
            host.plant(self.g, rows, self.gateWidth());
        }
        self.rig.reset();
        const rig = &self.rig;
        const s: c_int = @intCast(rows);
        const h: c_int = @intCast(c.heads);
        const d: c_int = @intCast(c.hd);
        const w: c_int = @intCast(self.width());
        switch (c.kind) {
            .table => {
                self.args[0] = try rig.input(c.dt, &.{ 1, s, w }, self.x);
                self.args[1] = try rig.input(c.dt, &.{ 1, s, @intCast(self.gateWidth()) }, self.g);
            },
            .strided => {
                // x: the attention output [1, H, S, D] seen as [1, S, H, D]; the gate: the second half of the
                // [1, S, H, 2D] q|gate projection (its first half random).
                const xs = try a.alloc(f64, self.x.len);
                for (0..rows) |t| for (0..c.heads) |hh| for (0..c.hd) |j| {
                    xs[(hh * rows + t) * c.hd + j] = self.x[(t * c.heads + hh) * c.hd + j];
                };
                const qg = try host.uniform(a, rng, 2 * self.g.len, -40, 40, c.dt);
                for (0..rows * c.heads) |r| @memcpy(qg[(2 * r + 1) * c.hd ..][0..c.hd], self.g[r * c.hd ..][0..c.hd]);
                const xb = try rig.input(c.dt, &.{ 1, h, s, d }, xs);
                const qb = try rig.input(c.dt, &.{ 1, s, h, 2 * d }, qg);
                self.views[0] = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_transpose_axes(&self.views[0], xb, &[_]c_int{ 0, 2, 1, 3 }, 4, rig.stream()));
                self.views[1] = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&self.views[1], qb, &[_]c_int{ 0, 0, 0, d }, 4, &[_]c_int{ 1, s, h, 2 * d }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, rig.stream()));
                self.args[0] = self.views[0];
                self.args[1] = self.views[1];
            },
        }
        // MLX's own sigmoid over every 16-bit pattern, as the wrapper binds it.
        self.args[2] = try fused_activations.swigluSigTable(rig.stream(), c.dt, std.heap.c_allocator);
        self.args[3] = try host.intScalar(rig, s * w);
        return true;
    }

    pub fn run(self: *Gate) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..4]);
    }

    pub fn count(self: *const Gate) usize {
        return self.shape.rows * self.width();
    }

    pub fn reference(self: *const Gate, want: []f64, mag: []f64) !void {
        const c = self.cfg;
        const gw = self.gateWidth();
        const per: usize = if (gw == self.width()) 1 else c.hd;
        for (want, mag, 0..) |*o, *m, i| {
            const g = self.g[(i / self.width()) * gw + (i % self.width()) / per];
            const f = 1 / (1 + @exp(-g));
            o.* = host.rt(c.dt, self.x[i] * host.rt(c.dt, f));
            m.* = @abs(o.*);
        }
    }

    /// The op chain each arm replaces, on the same inputs.
    pub fn replaced(self: *Gate) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const c = self.cfg;
        const x = self.args;
        const s: c_int = @intCast(self.shape.rows);
        const h: c_int = @intCast(c.heads);
        const d: c_int = @intCast(c.hd);
        const out = switch (c.kind) {
            .table => if (c.headwise)
                try g.mul(try g.reshape(x[0], &.{ 1, s, h, d }), try g.reshape(try g.sigmoid(x[1]), &.{ 1, s, h, 1 }))
            else
                try g.mul(x[0], try g.sigmoid(x[1])),
            .strided => try g.mul(x[0], try g.sigmoid(x[1])),
        };
        return self.rig.bytesOf(&.{out});
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *Gate) ![]const u8 {
        const c = self.cfg;
        const x = self.args;
        const st = self.rig.stream();
        const got = switch (c.kind) {
            .table => try gates.fusedSigmoidGateMul(st, x[0], x[1], if (c.headwise) @intCast(c.heads) else 0),
            .strided => try gates.stridedSigmoidGateMul(st, x[0], x[1]),
        } orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        return self.rig.bytesOf(&.{got});
    }
};


fn tokens(comptime list: []const usize, comptime n: usize) []const Shape {
    comptime var out: [list.len]Shape = undefined;
    inline for (list, 0..) |r, i| out[i] = .{ .rows = r, .n = n, .k = 1 };
    const final = out;
    return &final;
}

fn gateSpec(comptime name: []const u8, cfg: GateCfg, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = host.depthOne,
        .ops = 4,
        .mids = 1,
        .out = host.elemOf(cfg.dt),
        .replaces = switch (cfg.kind) {
            .table => if (cfg.headwise) "sigmoid -> broadcast multiply" else "sigmoid -> multiply",
            .strided => "sigmoid -> multiply (strided views)",
        },
        .served = served,
    };
}

/// Muse-Glimmer (32 x 128 by channel), Spark-X2.5 (16 x 256 by head), the qwen3_5 full
/// layers' gate (24 x 256; f16 on Bonsai's packs) up to the add-norm row cap.
const gate_registrations = .{
    .{ "sigmoid_gate_mul bf16 (by channel, 32 x 128)", GateCfg{ .kind = .table, .heads = 32, .hd = 128 }, tokens(&.{ 1, 5, 64 }, 4096), &table_launch },
    .{ "sigmoid_gate_mul bf16 (by head, 16 x 256)", GateCfg{ .kind = .table, .heads = 16, .hd = 256, .headwise = true }, tokens(&.{ 1, 5, 64 }, 4096), &table_launch },
    .{ "strided_sigmoid_gate_mul bf16 (24 x 256)", GateCfg{ .kind = .strided, .heads = 24, .hd = 256 }, tokens(&.{ 1, 3, 16 }, 6144), &strided_launch },
    .{ "strided_sigmoid_gate_mul f16 (24 x 256)", GateCfg{ .kind = .strided, .dt = .float16, .heads = 24, .hd = 256 }, tokens(&.{ 1, 3, 16 }, 6144), &strided_f16_launch },
};

// ── online softmax in float64 ──

const Q23 = 0x1p-23;

/// A key a kernel block takes for one query row: its score in the kernel's units (log2 units under exp2), the
/// score's error, whether a mask hides it (the score is then the kernel's mask value), the weight its exp carries
/// into the sum (a joined chunk's sum; 1 for a key), its value row and that row's errors (a dequantized row).
const Key = struct { s: f64, e: f64 = 0, on: bool = true, w: f64 = 1, v: []const f64, ve: ?[]const f64 = null };

/// exp2 on log2-unit scores or exp; P's rounding before P·V (half: 2^-11 of it, 2^-25 at least); a block whose
/// keys are all masked to -Inf leaves the max and sum as they were (lane_attn's rule).
const Soft = struct { base2: bool = false, p_unit: f64 = 0, p_abs: f64 = 0, keep_empty: bool = false };

/// The state a row starts from: the kernel's starting max, or a carried / sink state (`l` > 0) whose max is `e` off.
const Init = struct { m: f64, e: f64 = 0, l: f64 = 0, o: ?[]const f64 = null };

/// A row's max and sum after its blocks, with their errors (the unnormalized output and its errors land in o, eo).
const Row = struct { m: f64, em: f64, l: f64, el: f64 };

const inf = std.math.inf(f64);

fn expIn(base2: bool, x: f64) f64 {
    return if (base2) @exp2(x) else @exp(x);
}

/// One query row's online softmax over `keys` in the kernel's blocks (`ends`), in float64 block by block as the
/// kernel runs it, so a non-finite value meets the same max, rescale and 0 x Inf as on the device (max ignores NaN,
/// as Metal's does; a -Inf score weighs 0; no rescale while the max is still -Inf). Each output's error is bounded against the final max: per visible term
/// its score's and the max's error, its exp, the rescales it passed (a fast exp: 3 ulps each, plus 2 per unit the
/// max moved), P's rounding and the f32 sums; the carried term the same without a score.
fn online(soft: Soft, init: Init, keys: []const Key, ends: []const usize, o: []f64, eo: []f64) Row {
    var m = init.m;
    var l = init.l;
    if (init.o) |c| @memcpy(o, c) else @memset(o, 0);
    var at: usize = 0;
    for (ends) |end| {
        const blk = keys[at..end];
        at = end;
        var bm = -inf;
        for (blk) |k| bm = @max(bm, k.s);
        const nm = @max(m, bm);
        const empty = soft.keep_empty and bm == -inf;
        const f = if (empty or nm == -inf) 1 else expIn(soft.base2, m - nm);
        l *= f;
        for (o) |*x| x.* *= f;
        for (blk) |k| {
            const p = if (k.s == -inf) 0 else expIn(soft.base2, k.s - nm);
            if (!empty) l += p * k.w;
            for (o, k.v) |*x, v| x.* += p * v;
        }
        if (!empty) m = nm;
    }
    const lnb: f64 = if (soft.base2) std.math.ln2 else 1;
    var em: f64 = if (init.l > 0) init.e else 0;
    for (keys) |k| if (k.on) {
        em = @max(em, k.e);
    };
    const sums = spec.gamma(@floatFromInt(keys.len + 8), host.U);
    const rescales = @as(f64, @floatFromInt(ends.len)) * (3 * Q23 + 2 * host.U);
    var el: f64 = 0;
    @memset(eo, 0);
    if (init.l > 0) {
        const w = expIn(soft.base2, init.m - m);
        const eps = std.math.expm1(lnb * (init.e + em)) + rescales + 2 * Q23 * @abs(m - init.m) + sums;
        el += init.l * w * eps;
        if (init.o) |c| for (eo, c) |*x, v| {
            x.* += @abs(v) * w * eps;
        };
    }
    for (keys) |k| {
        if (!k.on) continue;
        const span = @abs(m - k.s);
        const p = expIn(soft.base2, k.s - m);
        const eps = std.math.expm1(lnb * (k.e + em)) + host.fastExp(span) + rescales + 2 * Q23 * span + sums;
        el += p * k.w * eps;
        for (eo, k.v) |*x, v| x.* += (p * (eps + soft.p_unit) + soft.p_abs) * @abs(v);
        if (k.ve) |ve| for (eo, ve) |*x, e| {
            x.* += p * e;
        };
    }
    return .{ .m = m, .em = em, .l = l, .el = el };
}

/// Chunk states joined max-first as row_attn's merge does: weights exp(m_c - max), zero where the sum is not
/// positive, one fused multiply-add a chunk; the inputs exact, the weights' fast exp and the f32 sums their error.
fn joinMax(m: []const f64, l: []const f64, o: []const f64, d: usize, out: []f64, eo: []f64) Row {
    var mx = -inf;
    for (m) |x| mx = @max(mx, x);
    var ls: f64 = 0;
    var el: f64 = 0;
    @memset(out, 0);
    @memset(eo, 0);
    const sums = spec.gamma(@floatFromInt(m.len + 1), host.U);
    for (m, l, 0..) |mc, lc, c| {
        const w = if (lc > 0) @exp(mc - mx) else 0;
        const eps = if (w > 0) host.fastExp(mx - mc) + sums else 0;
        ls += lc * w;
        el += @abs(lc) * w * eps;
        for (out, eo, o[c * d ..][0..d]) |*x, *e, v| {
            x.* += v * w;
            e.* += @abs(v) * w * eps;
        }
    }
    return .{ .m = mx, .em = 0, .l = ls, .el = el };
}

/// o / l as a kernel stores it (a reciprocal and a product, or a divide): values in o, errors in eo.
fn normalize(r: Row, o: []f64, eo: []f64) void {
    for (o, eo) |*y, *e| {
        y.* /= r.l;
        e.* = (e.* + @abs(y.*) * r.el) / r.l + 2 * host.U * @abs(y.*);
    }
}

/// The largest power of two `t` is a multiple of (Inf for 0, 0 for a non-finite).
fn granule(t: f64) f64 {
    if (t == 0) return inf;
    if (!std.math.isFinite(t)) return 0;
    const fr = std.math.frexp(t);
    const m: u64 = @intFromFloat(@abs(fr.significand) * 0x1p53);
    return std.math.ldexp(@as(f64, 1), fr.exponent - 53 + @as(i32, @ctz(m)));
}

/// How gate 4 runs a kernel's sums (null: exact).
const Sim = ?struct { acc: spec.Accum, order: spec.Order };

/// q·k times `scale` in float64, with its error: k's own errors (`kerr`, a dequantized row), f32 sums `depth` deep
/// over the products' magnitude (none when every product is an integer and the magnitude stays under 2^24, where any
/// f32 order is exact), then `roundings` of the scaled value. Gate 4 runs the sum in its accumulator.
fn dot(q: []const f64, k: []const f64, scale: f64, depth: usize, roundings: f64, sim: Sim, terms: []f64) host.Val {
    return dotErr(q, k, null, scale, depth, roundings, sim, terms);
}

fn dotErr(q: []const f64, k: []const f64, kerr: ?[]const f64, scale: f64, depth: usize, roundings: f64, sim: Sim, terms: []f64) host.Val {
    var mag: f64 = 0;
    var ek: f64 = 0;
    var grain: f64 = inf;
    for (terms[0..q.len], q, k, 0..) |*t, a, b, i| {
        t.* = a * b;
        mag += @abs(t.*);
        if (kerr) |e| ek += @abs(a) * e[i];
        grain = @min(grain, granule(t.*));
    }
    const s = if (sim) |x| spec.sumIn(x.acc, x.order, terms[0..q.len]) else host.sum(terms[0..q.len]);
    // Every partial sum is a multiple of the terms' finest power of two: under 2^24 of them, f32 holds it exactly.
    const sums = if (mag <= 0x1p24 * grain) 0 else spec.gamma(@floatFromInt(depth), host.U) * mag;
    const v = s * scale;
    return .{ .v = v, .e = (sums + ek) * @abs(scale) + roundings * host.U * @abs(v) };
}

/// Gate 4's q and k rows: integers exact in f32 in any order whose partial sums leave fp16's range and bf16's
/// precision. q is +`mag` on the first half and -`mag` on the second; each k row's second half is its first rotated
/// by one (so no summation tree pairs the halves alike) but for +1 at two places, so a dot is 0, -mag or -2 mag (a softmax
/// over a few levels) while its partial sums reach mag x 127 x D/2. `k` arrives drawn (codes 0..127).
/// Gate 4's float q: the largest 8-bit integer, exact in bf16 and f16.
const STRESS_Q = 251;

fn stressRows(rng: std.Random, q: []f64, k: []f64, d: usize, mag: f64) void {
    const half = d / 2;
    for (q, 0..) |*x, i| x.* = if (i % d < half) mag else -mag;
    var r: usize = 0;
    while (r < k.len) : (r += d) for (0..half) |i| {
        k[r + half + i] = k[r + (i + 1) % half] + @as(f64, if (i < 2) @floatFromInt(rng.uintLessThan(u32, 2)) else 0);
    };
}

// ── msv_attn_p256: flash prefill attention (hd 256, or MiMo's q/k 192 with v 128) ──

const BQ = 64;
const BK = 32;
const LOG2E = 1.44269504088896340736;

/// Threadgroup bytes: the shared K^T / V tile, max(LDK * BDK, BK * LDV) 2-byte elements.
fn a256Launch(comptime bdk: usize, comptime bdv: usize) metal.Launch {
    return .{
        .name = "msv_attn_p256",
        .inputs = &.{ "q", "k", "v", "scl", "win", "kr", "phase", "m_in", "l_in", "o_in", "mask", "skip", "sinks" },
        .outputs = &.{ "out", "m_out", "l_out", "o_out" },
        .source = attn256.ATTN256_KERNEL_SOURCE,
        .header = attn256.ATTN256_KERNEL_HEADER,
        .contiguous = false,
        .tg_bytes = 2 * @as(u32, @max((BK + 8) * bdk, BK * (bdv + 8))),
        .m1 = 1024,
        .m2 = 1024,
    };
}

/// One dispatch of the key range: the whole of it, or a budgeted chunk that hands its state on (out), takes and
/// hands it (through), or takes it and normalizes (in).
const Phase = enum { single, out, through, in };

const A256Cfg = struct {
    dt: mlx.mlx_dtype = .bfloat16,
    bdk: usize = 256,
    bdv: usize = 256,
    gqa: usize = 2,
    window: usize = 0,
    qsa: bool = false,
    sink: bool = false,
    phase: Phase = .single,
};

/// A shape's rows are the kv heads (`gqa` query heads each), n the query length, k the key length. K and V are the
/// cache's views: the first kL rows of a longer buffer.
const A256 = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    cfg: A256Cfg,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    sinks: []f64 = &.{},
    mask: []bool = &.{},
    skip: []bool = &.{},
    m_in: []f64 = &.{},
    l_in: []f64 = &.{},
    o_in: []f64 = &.{},
    scale: f64 = 0,
    kr: [2]usize = .{ 0, 0 },
    args: [13]mlx.mlx_array = undefined,
    /// K and V views, the bool mask and skip tables: freed at the next prepare.
    owned: [4]mlx.mlx_array = @splat(.{ .ctx = null }),

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, cfg: A256Cfg) A256 {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .cfg = cfg, .arena = .init(a) };
    }

    fn freeOwned(self: *A256) void {
        for (&self.owned) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
    }

    pub fn deinit(self: *A256) void {
        self.freeOwned();
        self.rig.deinit();
        self.arena.deinit();
    }

    fn final(phase: Phase) bool {
        return phase == .single or phase == .in;
    }

    pub fn geometry(self: *const A256, shape: Shape) ?metal.Geometry {
        return self.geometryAt(shape, self.cfg.phase);
    }

    fn geometryAt(self: *const A256, shape: Shape, phase: Phase) ?metal.Geometry {
        const c = self.cfg;
        if (shape.n < 16 or shape.k < shape.n) return null;
        const hq: c_int = @intCast(shape.rows * c.gqa);
        const ql: c_int = @intCast(shape.n);
        const bdv: c_int = @intCast(c.bdv);
        var g = metal.Geometry{ .grid = .{ @divTrunc(ql + BQ - 1, BQ) * 32, hq * 8, 1 }, .tg = .{ 32, 8, 1 } };
        if (final(phase)) {
            g.out(&.{ 1, hq, ql, bdv }, c.dt);
            for (0..3) |_| g.out(&.{1}, .float32);
            g.unread = 0b1110;
        } else {
            g.out(&.{1}, c.dt);
            g.out(&.{ 1, hq, ql }, .float32);
            g.out(&.{ 1, hq, ql }, .float32);
            g.out(&.{ 1, hq, ql, bdv }, .float32);
            g.unread = 0b0001;
        }
        g.dtype("T", c.dt);
        g.int("QSA", @intFromBool(c.qsa));
        g.int("BDK", c.bdk);
        g.int("BDV", c.bdv);
        g.int("SINK", @intFromBool(c.sink));
        return g;
    }

    /// The budgeted chunk length the reference splits a causal range at (BK-aligned, a third of the keys).
    fn chunk(kl: usize) usize {
        return @max(BK, kl / 3 / BK * BK);
    }

    fn rangeOf(phase: Phase, kl: usize) [2]usize {
        const c = chunk(kl);
        return switch (phase) {
            .single => .{ 0, kl },
            .out => .{ 0, c },
            .through => .{ c, 2 * c },
            .in => .{ 2 * c, kl },
        };
    }

    pub fn prepare(self: *A256, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        self.freeOwned();
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const c = self.cfg;
        const hk = shape.rows;
        const hq = hk * c.gqa;
        const ql = shape.n;
        const kl = shape.k;
        if (fill == .stress) {
            self.q = try a.alloc(f64, hq * ql * c.bdk);
            self.k = try host.ints(a, rng, hk * kl * c.bdk, 0, 127);
            stressRows(rng, self.q, self.k, c.bdk, STRESS_Q);
        } else {
            self.q = try host.uniform(a, rng, hq * ql * c.bdk, -2, 2, c.dt);
            self.k = try host.uniform(a, rng, hk * kl * c.bdk, -2, 2, c.dt);
        }
        self.v = try host.uniform(a, rng, hk * kl * c.bdv, -2, 2, c.dt);
        if (fill == .nonfinite) {
            // A query row's own NaN / Inf (rows 0..3 of head 0), and value columns every row reads (keys 0..3).
            host.plant(self.q, hq * ql, c.bdk);
            host.plant(self.v, hk * kl, c.bdv);
        }
        self.sinks = try host.uniform(a, rng, hq, -1, 3, c.dt);
        self.scale = host.rt(.float32, 1 / @sqrt(@as(f64, @floatFromInt(c.bdk))));
        self.kr = rangeOf(c.phase, kl);
        self.m_in = try host.uniform(a, rng, hq * ql, -2, 3, .float32);
        self.l_in = try host.uniform(a, rng, hq * ql, 1, 8, .float32);
        self.o_in = try host.uniform(a, rng, hq * ql * c.bdv, -8, 8, .float32);
        if (c.qsa) try self.drawMask(a, rng);
        try self.bind();
        return true;
    }

    /// A QSA visibility: causal, each key kept at 3/5, the diagonal always; per 64-row tile some 32-key blocks
    /// dropped whole, so the kernel's skip table skips them.
    fn drawMask(self: *A256, a: std.mem.Allocator, rng: std.Random) !void {
        const ql = self.shape.n;
        const kl = self.shape.k;
        const nq = (ql + BQ - 1) / BQ;
        const nk = (kl + BK - 1) / BK;
        self.mask = try a.alloc(bool, ql * kl);
        const drop = try a.alloc(bool, nq * nk);
        for (drop) |*d| d.* = rng.uintLessThan(u32, 4) == 0;
        for (0..ql) |r| for (0..kl) |col| {
            const pos = kl - ql + r;
            self.mask[r * kl + col] = col == pos or (col < pos and !drop[(r / BQ) * nk + col / BK] and rng.uintLessThan(u32, 5) < 3);
        };
        self.skip = try a.alloc(bool, nq * nk);
        @memset(self.skip, false);
        for (0..ql) |r| for (0..kl) |col| {
            if (self.mask[r * kl + col]) self.skip[(r / BQ) * nk + col / BK] = true;
        };
    }

    fn bind(self: *A256) !void {
        const rig = &self.rig;
        rig.reset();
        const c = self.cfg;
        const hk = self.shape.rows;
        const hq = hk * c.gqa;
        const ql = self.shape.n;
        const kl = self.shape.k;
        const st = rig.stream();
        const a = self.arena.allocator();
        const ci = struct {
            fn f(x: usize) c_int {
                return @intCast(x);
            }
        }.f;
        const q = try rig.input(c.dt, &.{ 1, ci(hq), ci(ql), ci(c.bdk) }, self.q);
        // The cache buffers hold `cap` rows; the kernel reads their first kL through the strides.
        const cap = kl + 24;
        var kv: [2]mlx.mlx_array = undefined;
        var pad = std.Random.DefaultPrng.init(91);
        for ([_][]const f64{ self.k, self.v }, [_]usize{ c.bdk, c.bdv }, 0..) |src, d, i| {
            const buf = try host.uniform(a, pad.random(), hk * cap * d, -1, 1, c.dt);
            for (0..hk) |h| @memcpy(buf[h * cap * d ..][0 .. kl * d], src[h * kl * d ..][0 .. kl * d]);
            const full = try rig.input(c.dt, &.{ 1, ci(hk), ci(cap), ci(d) }, buf);
            self.owned[i] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&self.owned[i], full, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, ci(hk), ci(kl), ci(d) }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, st));
            kv[i] = self.owned[i];
        }
        const dummy = try rig.input(.float32, &.{1}, &.{0});
        const dummy_b = [_]bool{false};
        var mask = mlx.mlx_array{ .ctx = null };
        var skip = mlx.mlx_array{ .ctx = null };
        if (c.qsa) {
            self.owned[2] = mlx.mlx_array_new_data(self.mask.ptr, &[_]c_int{ 1, 1, ci(ql), ci(kl) }, 4, .bool_);
            self.owned[3] = mlx.mlx_array_new_data(self.skip.ptr, &[_]c_int{ 1, ci((ql + BQ - 1) / BQ), ci((kl + BK - 1) / BK) }, 3, .bool_);
        } else {
            self.owned[2] = mlx.mlx_array_new_data(&dummy_b, &[_]c_int{ 1, 1, 1, 1 }, 4, .bool_);
            self.owned[3] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&self.owned[3], self.owned[2]));
        }
        mask = self.owned[2];
        skip = self.owned[3];
        const phase = self.cfg.phase;
        const carry = phase == .through or phase == .in;
        self.args = .{
            q,
            kv[0],
            kv[1],
            try rig.input(.float32, &.{1}, &.{self.scale}),
            try rig.input(.int32, &.{1}, &.{@floatFromInt(c.window)}),
            try rig.input(.int32, &.{2}, &.{ @floatFromInt(self.kr[0]), @floatFromInt(self.kr[1]) }),
            try rig.input(.int32, &.{1}, &.{@floatFromInt(@as(u32, @intFromBool(carry)) | (@as(u32, @intFromBool(final(phase))) << 1))}),
            if (carry) try rig.input(.float32, &.{ 1, ci(hq), ci(ql) }, self.m_in) else dummy,
            if (carry) try rig.input(.float32, &.{ 1, ci(hq), ci(ql) }, self.l_in) else dummy,
            if (carry) try rig.input(.float32, &.{ 1, ci(hq), ci(ql), ci(c.bdv) }, self.o_in) else dummy,
            mask,
            skip,
            if (c.sink) try rig.input(c.dt, &.{ci(hq)}, self.sinks) else dummy,
        };
    }

    pub fn run(self: *A256) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const A256) usize {
        const rows = self.shape.rows * self.cfg.gqa * self.shape.n;
        return if (final(self.cfg.phase)) rows * self.cfg.bdv else rows * (2 + self.cfg.bdv);
    }

    fn emit(self: *const A256, want: []f64, mag: ?[]f64, sim: Sim) !void {
        const a = self.rig.a;
        const c = self.cfg;
        const hq = self.shape.rows * c.gqa;
        const ql = self.shape.n;
        const kl = self.shape.k;
        const nkt = (kl + BK - 1) / BK;
        const keys = try a.alloc(Key, (nkt + 1) * BK);
        defer a.free(keys);
        const ends = try a.alloc(usize, nkt + 1);
        defer a.free(ends);
        const zeros = try a.alloc(f64, c.bdv);
        defer a.free(zeros);
        @memset(zeros, 0);
        const terms = try a.alloc(f64, c.bdk);
        defer a.free(terms);
        const o = try a.alloc(f64, c.bdv);
        defer a.free(o);
        const eo = try a.alloc(f64, c.bdv);
        defer a.free(eo);
        const scale2 = self.scale * LOG2E;
        const q_off = kl - ql;
        const carry = c.phase == .through or c.phase == .in;
        for (0..hq) |h| for (0..ql) |r| {
            const hk = h / c.gqa;
            const tile = r / BQ;
            const q_lo = tile * BQ + q_off;
            const k_end = @min(self.kr[1], kl);
            const kb_lim = @min((k_end + BK - 1) / BK, (q_lo + BQ - 1 + BK) / BK);
            var kb = self.kr[0] / BK;
            if (c.window > 0) kb = @max(kb, (q_lo + 1 -| c.window) / BK);
            const pos = q_lo + r % BQ;
            const qrow = self.q[(h * ql + r) * c.bdk ..][0..c.bdk];
            var n: usize = 0;
            var nb: usize = 0;
            while (kb < kb_lim) : (kb += 1) {
                if (c.qsa and !self.skip[tile * nkt + kb]) continue;
                for (0..BK) |j| {
                    const col = kb * BK + j;
                    const v = if (col < kl) self.v[(hk * kl + col) * c.bdv ..][0..c.bdv] else zeros;
                    const hide = col >= kl or pos < col or (c.window > 0 and pos - col >= c.window) or (c.qsa and !self.mask[r * kl + col]);
                    keys[n] = if (hide) .{ .s = -std.math.inf(f64), .on = false, .v = v } else blk: {
                        const d = dot(qrow, self.k[(hk * kl + col) * c.bdk ..][0..c.bdk], scale2, c.bdk, 3, sim, terms);
                        break :blk .{ .s = d.v, .e = d.e, .v = v };
                    };
                    n += 1;
                }
                ends[nb] = n;
                nb += 1;
            }
            const row = h * ql + r;
            const start: Init = if (carry)
                .{ .m = self.m_in[row], .l = self.l_in[row], .o = self.o_in[row * c.bdv ..][0..c.bdv] }
            else if (c.sink)
                .{ .m = self.sinks[h] * LOG2E, .e = 2 * host.U * @abs(self.sinks[h] * LOG2E), .l = 1 }
            else
                .{ .m = @as(f32, -3.0e38) };
            const st = online(.{ .base2 = true }, start, keys[0..n], ends[0..nb], o, eo);
            if (final(c.phase)) {
                normalize(st, o, eo);
                @memcpy(want[row * c.bdv ..][0..c.bdv], o);
                if (mag) |m| for (m[row * c.bdv ..][0..c.bdv], eo) |*x, e| {
                    x.* = host.allowance(e);
                };
                continue;
            }
            const rows = hq * ql;
            want[row] = st.m;
            want[rows + row] = st.l;
            @memcpy(want[2 * rows + row * c.bdv ..][0..c.bdv], o);
            if (mag) |m| {
                m[row] = host.allowance(st.em);
                m[rows + row] = host.allowance(st.el);
                for (m[2 * rows + row * c.bdv ..][0..c.bdv], eo) |*x, e| x.* = host.allowance(e);
            }
        };
    }

    pub fn reference(self: *const A256, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const A256, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    /// The shipped wrapper on the same inputs (single dispatch: budget 0, or `budget`).
    fn wrapper(self: *A256, budget: i64) ![]const u8 {
        const c = self.cfg;
        attn256.fused256_budget_override = budget;
        defer attn256.fused256_budget_override = null;
        const got = (try attn256.fusedSdpa256Impl(self.rig.stream(), self.args[0], self.args[1], self.args[2], @floatCast(self.scale), @intCast(c.window), if (c.qsa) self.args[10] else null, if (c.sink) self.args[12] else null)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        return self.rig.bytesOf(&.{got});
    }

    /// The declared launch run as the budgeted chain at `chunk`: out, through, in, each chunk's state read back and
    /// bound as the next one's carry. The final output's bytes.
    fn chained(self: *A256, a: std.mem.Allocator) ![]u8 {
        const rig = &self.rig;
        const kl = self.shape.k;
        const hq: c_int = @intCast(self.shape.rows * self.cfg.gqa);
        const ql: c_int = @intCast(self.shape.n);
        const bdv: c_int = @intCast(self.cfg.bdv);
        var args = self.args;
        for ([_]Phase{ .out, .through, .in }) |phase| {
            const r = rangeOf(phase, kl);
            args[5] = try rig.input(.int32, &.{2}, &.{ @floatFromInt(r[0]), @floatFromInt(r[1]) });
            const carry = phase != .out;
            args[6] = try rig.input(.int32, &.{1}, &.{@floatFromInt(@as(u32, @intFromBool(carry)) | (@as(u32, @intFromBool(phase == .in)) << 1))});
            try rig.launch(self.launch, self.geometryAt(self.shape, phase).?, &args);
            if (phase == .in) break;
            const m = try host.read(a, rig.output(1), rig.stream());
            defer a.free(m);
            const l = try host.read(a, rig.output(2), rig.stream());
            defer a.free(l);
            const o = try host.read(a, rig.output(3), rig.stream());
            defer a.free(o);
            args[7] = try rig.input(.float32, &.{ 1, hq, ql }, m);
            args[8] = try rig.input(.float32, &.{ 1, hq, ql }, l);
            args[9] = try rig.input(.float32, &.{ 1, hq, ql, bdv }, o);
        }
        return a.dupe(u8, rig.outputBytes());
    }
};

fn a256Spec(comptime name: []const u8, cfg: A256Cfg, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = host.depthOne,
        .out = if (A256.final(cfg.phase)) host.elemOf(cfg.dt) else .f32,
        .tiles = .{ .{}, .{ .size = BQ, .min = 16 }, .{ .size = BK, .min = 16 } },
        .served = served,
        .sums = .{ .exact = false },
    };
}

/// rows = kv heads, n = query length, k = key length.
fn qk(comptime list: []const [3]usize) []const Shape {
    comptime var out: [list.len]Shape = undefined;
    inline for (list, 0..) |x, i| out[i] = .{ .rows = x[0], .n = x[1], .k = x[2] };
    const final = out;
    return &final;
}

const a256_256 = a256Launch(256, 256);
const a256_192 = a256Launch(192, 128);

/// The causal arm (qwen3_5 / Gemma full layers off NAX, every chip for f16 Bonsai packs), Gemma's sliding band
/// (512), the budgeted chain's three dispatch kinds, qwen4_exp's QSA mask, MiMo's sink layers (q/k 192, v 128, the
/// sliding ones banded to 128) off NAX.
const a256_registrations = .{
    .{ "msv_attn_p256 bf16 (causal)", A256Cfg{}, qk(&.{ .{ 1, 16, 40 }, .{ 1, 64, 300 }, .{ 2, 130, 200 } }), &a256_256 },
    .{ "msv_attn_p256 f16 (causal)", A256Cfg{ .dt = .float16 }, qk(&.{ .{ 1, 16, 40 }, .{ 1, 100, 260 } }), &a256_256 },
    .{ "msv_attn_p256 bf16 (band 512)", A256Cfg{ .window = 512 }, qk(&.{ .{ 1, 16, 560 }, .{ 1, 80, 640 } }), &a256_256 },
    .{ "msv_attn_p256 bf16 (causal, chunk hands its state on)", A256Cfg{ .phase = .out }, qk(&.{ .{ 1, 16, 96 }, .{ 1, 64, 288 } }), &a256_256 },
    .{ "msv_attn_p256 bf16 (causal, chunk carries the state through)", A256Cfg{ .phase = .through }, qk(&.{ .{ 1, 16, 96 }, .{ 1, 64, 288 } }), &a256_256 },
    .{ "msv_attn_p256 bf16 (causal, last chunk takes the state)", A256Cfg{ .phase = .in }, qk(&.{ .{ 1, 16, 96 }, .{ 1, 64, 288 } }), &a256_256 },
    .{ "msv_attn_p256 bf16 (QSA mask)", A256Cfg{ .qsa = true }, qk(&.{ .{ 1, 16, 64 }, .{ 1, 100, 300 } }), &a256_256 },
    .{ "msv_attn_p256 bf16 (q/k 192, v 128, sinks, causal)", A256Cfg{ .bdk = 192, .bdv = 128, .sink = true }, qk(&.{ .{ 1, 16, 40 }, .{ 1, 90, 250 } }), &a256_192 },
    .{ "msv_attn_p256 bf16 (q/k 192, v 128, sinks, band 128)", A256Cfg{ .bdk = 192, .bdv = 128, .sink = true, .window = 128 }, qk(&.{ .{ 1, 16, 200 }, .{ 1, 100, 300 } }), &a256_192 },
};

// ── decode windows: row_attn and lane_attn (qwen3_5 full layers, each row the serial step's bits) ──

const WD = 256;

/// A window of W query rows at the last W of L key positions, as a chain or a draft tree: q [1, H, W, D] (H = HKV x
/// g), the cache views k, v [1, HKV, L, D] (the first L rows of longer buffers), each row's depth and path.
const Win = struct {
    g: usize,
    tree: bool,
    hkv: usize = 0,
    w: usize = 0,
    l: usize = 0,
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    depth: []usize = &.{},
    path: []usize = &.{},
    maxd: usize = 0,

    fn p(self: *const Win) usize {
        return self.l - self.w;
    }

    fn draw(self: *Win, a: std.mem.Allocator, rng: std.Random, shape: Shape, fill: spec.Fill) !void {
        self.hkv = shape.rows;
        self.w = shape.n;
        self.l = shape.k;
        const h = self.hkv * self.g;
        if (fill == .stress) {
            self.q = try a.alloc(f64, h * self.w * WD);
            self.k = try host.ints(a, rng, self.hkv * self.l * WD, 0, 127);
            stressRows(rng, self.q, self.k, WD, STRESS_Q);
        } else {
            self.q = try host.uniform(a, rng, h * self.w * WD, -2, 2, .bfloat16);
            self.k = try host.uniform(a, rng, self.hkv * self.l * WD, -2, 2, .bfloat16);
        }
        self.v = try host.uniform(a, rng, self.hkv * self.l * WD, -2, 2, .bfloat16);
        if (fill == .nonfinite) {
            host.plant(self.q, h * self.w, WD);
            host.plant(self.v, self.hkv * self.l, WD);
        }
        self.depth = try a.alloc(usize, self.w);
        if (!self.tree) {
            self.maxd = self.w;
            self.path = try a.alloc(usize, self.w * self.w);
            for (0..self.w) |r| {
                self.depth[r] = r;
                for (0..self.w) |i| self.path[r * self.w + i] = i;
            }
            return;
        }
        // Each node under an earlier one: its path is its parent's, then itself.
        const par = try a.alloc(usize, self.w);
        self.depth[0] = 0;
        par[0] = 0;
        self.maxd = 1;
        for (1..self.w) |n| {
            par[n] = rng.uintLessThan(usize, n);
            self.depth[n] = self.depth[par[n]] + 1;
            self.maxd = @max(self.maxd, self.depth[n] + 1);
        }
        self.path = try a.alloc(usize, self.w * self.maxd);
        @memset(self.path, 0);
        for (0..self.w) |n| {
            var x = n;
            var d = self.depth[n] + 1;
            while (d > 0) : (x = par[x]) {
                d -= 1;
                self.path[n * self.maxd + d] = x;
            }
        }
    }

    /// The cache row holding key `pos` for window row `n`: the committed prefix, then the row's path.
    fn phys(self: *const Win, n: usize, pos: usize) usize {
        const pp = self.p();
        return if (pos < pp) pos else pp + self.path[n * self.maxd + pos - pp];
    }

    fn qRow(self: *const Win, h: usize, n: usize) []const f64 {
        return self.q[(h * self.w + n) * WD ..][0..WD];
    }

    fn kRow(self: *const Win, hk: usize, row: usize) []const f64 {
        return self.k[(hk * self.l + row) * WD ..][0..WD];
    }

    fn vRow(self: *const Win, hk: usize, row: usize) []const f64 {
        return self.v[(hk * self.l + row) * WD ..][0..WD];
    }

    /// q, the k and v views (kept in `owned`), depth [W] and path [W, MAXD] as int32.
    fn bind(self: *const Win, a: std.mem.Allocator, rig: *metal.Rig, owned: *[2]mlx.mlx_array) ![5]mlx.mlx_array {
        const ci = cInt;
        const q = try rig.input(.bfloat16, &.{ 1, ci(self.hkv * self.g), ci(self.w), WD }, self.q);
        const cap = self.l + 24;
        var pad = std.Random.DefaultPrng.init(91);
        var kv: [2]mlx.mlx_array = undefined;
        for ([_][]const f64{ self.k, self.v }, 0..) |src, i| {
            const buf = try host.uniform(a, pad.random(), self.hkv * cap * WD, -1, 1, .bfloat16);
            for (0..self.hkv) |h| @memcpy(buf[h * cap * WD ..][0 .. self.l * WD], src[h * self.l * WD ..][0 .. self.l * WD]);
            const full = try rig.input(.bfloat16, &.{ 1, ci(self.hkv), ci(cap), WD }, buf);
            owned[i] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&owned[i], full, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, ci(self.hkv), ci(self.l), WD }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, rig.stream()));
            kv[i] = owned[i];
        }
        const depth = try a.alloc(f64, self.w);
        for (depth, self.depth) |*o, d| o.* = @floatFromInt(d);
        const path = try a.alloc(f64, self.path.len);
        for (path, self.path) |*o, d| o.* = @floatFromInt(d);
        return .{ q, kv[0], kv[1], try rig.input(.int32, &.{ci(self.w)}, depth), try rig.input(.int32, &.{ ci(self.w), ci(self.maxd) }, path) };
    }

    fn tree_(self: *const Win, x: [5]mlx.mlx_array) ?row_attn.Tree {
        return if (self.tree) .{ .depth = x[3], .path = x[4], .max_depth = @intCast(self.maxd - 1) } else null;
    }
};

fn cInt(x: usize) c_int {
    return @intCast(x);
}

/// The (m, l, o) rows of a partial kernel's three f32 outputs, read back and bound as the next kernel's inputs.
fn rebind(a: std.mem.Allocator, rig: *metal.Rig, which: [3]usize) ![3]mlx.mlx_array {
    var out: [3]mlx.mlx_array = undefined;
    var vals: [3][]f64 = undefined;
    for (which, &vals) |i, *v| v.* = try host.read(a, rig.output(i), rig.stream());
    defer for (vals) |v| a.free(v);
    for (which, vals, &out) |i, v, *o| o.* = try rig.input(.float32, &.{@intCast(mlx.mlx_array_size(rig.output(i)))}, v);
    return out;
}

// row_attn: keys in CK-position chunks, each chunk's keys over SPLIT simdgroups in blocks of BLK.
const ROW_CK = 128;
const ROW_SPLIT = 4;
const ROW_BLK = 4;

fn rowPartLaunch(comptime g: usize) metal.Launch {
    return .{
        .name = "msv_row_attn_partial",
        .inputs = &.{ "Q", "K", "V", "depth", "path", "scale", "dims" },
        .outputs = &.{ "PM", "PL", "PO" },
        .source = row_attn.PARTIAL,
        .contiguous = false,
        // sm, sl [G * SPLIT] and so [G * SPLIT][D], f32.
        .tg_bytes = g * ROW_SPLIT * (2 + WD) * 4,
        .m1 = 1024,
        .m2 = 1024,
    };
}

const row_merge_launch = metal.Launch{
    .name = "msv_row_attn_merge",
    .inputs = &.{ "PM", "PL", "PO", "dims" },
    .outputs = &.{"OUT"},
    .source = row_attn.MERGE,
    .tg_bytes = 0,
    .m1 = 832,
    .m2 = 832,
};

/// row_attn's partial pass. rows = kv heads, n = window rows, k = keys.
const RowPart = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    win: Win,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    scale: f64 = 1.0 / 16.0,
    args: [7]mlx.mlx_array = undefined,
    owned: [2]mlx.mlx_array = @splat(.{ .ctx = null }),

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, g: usize, tree: bool) RowPart {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .win = .{ .g = g, .tree = tree }, .arena = .init(a) };
    }

    fn freeOwned(self: *RowPart) void {
        for (&self.owned) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
    }

    pub fn deinit(self: *RowPart) void {
        self.freeOwned();
        self.rig.deinit();
        self.arena.deinit();
    }

    fn nch(shape: Shape) usize {
        return (shape.k + ROW_CK - 1) / ROW_CK;
    }

    pub fn geometry(self: *const RowPart, shape: Shape) ?metal.Geometry {
        const g = self.win.g;
        if (shape.n < 1 or shape.n > 16 or shape.k < shape.n or g * ROW_SPLIT * 32 > 1024) return null;
        const slots = cInt(shape.rows * g * shape.n * nch(shape));
        var geo = metal.Geometry{ .grid = .{ cInt(32 * g * ROW_SPLIT), cInt(nch(shape)), cInt(shape.rows * shape.n) }, .tg = .{ cInt(32 * g * ROW_SPLIT), 1, 1 } };
        geo.out(&.{slots}, .float32);
        geo.out(&.{slots}, .float32);
        geo.out(&.{ slots, WD }, .float32);
        geo.int("D", WD);
        geo.int("G", g);
        geo.int("CK", ROW_CK);
        geo.int("SPLIT", ROW_SPLIT);
        geo.int("BLK", ROW_BLK);
        return geo;
    }

    pub fn prepare(self: *RowPart, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        self.freeOwned();
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        try self.win.draw(a, prng.random(), shape, fill);
        self.rig.reset();
        const x = try self.win.bind(a, &self.rig, &self.owned);
        const dims = [_]f64{ @floatFromInt(self.win.p()), @floatFromInt(shape.n), @floatFromInt(nch(shape)), @floatFromInt(self.win.maxd) };
        self.args = .{ x[0], x[1], x[2], x[3], x[4], try self.rig.input(.float32, &.{1}, &.{self.scale}), try self.rig.input(.int32, &.{4}, &dims) };
        return true;
    }

    pub fn run(self: *RowPart) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const RowPart) usize {
        return self.shape.rows * self.win.g * self.shape.n * nch(self.shape) * (2 + WD);
    }

    fn emit(self: *const RowPart, want: []f64, mag: ?[]f64, sim: Sim) !void {
        const a = self.rig.a;
        const w = &self.win;
        const keys = try a.alloc(Key, ROW_CK);
        defer a.free(keys);
        const ends = try a.alloc(usize, ROW_CK);
        defer a.free(ends);
        const terms = try a.alloc(f64, WD);
        defer a.free(terms);
        const o = try a.alloc(f64, WD);
        defer a.free(o);
        const eo = try a.alloc(f64, WD);
        defer a.free(eo);
        const n_ch = nch(self.shape);
        const slots = self.shape.rows * w.g * w.w * n_ch;
        for (0..self.shape.rows * w.g) |qh| for (0..w.w) |node| for (0..n_ch) |c| {
            const hk = qh / w.g;
            const k0 = c * ROW_CK;
            const k1 = @min(k0 + ROW_CK, w.p() + w.depth[node] + 1);
            var n: usize = 0;
            var nb: usize = 0;
            for (0..ROW_SPLIT) |sg| {
                var base = k0 + sg;
                while (base < k1) : (base += ROW_SPLIT * ROW_BLK) {
                    for (0..ROW_BLK) |j| {
                        const pos = base + j * ROW_SPLIT;
                        if (pos >= k1) continue;
                        const row = w.phys(node, pos);
                        const d = dot(w.qRow(qh, node), w.kRow(hk, row), self.scale, 13, 1, sim, terms);
                        keys[n] = .{ .s = d.v, .e = d.e, .v = w.vRow(hk, row) };
                        n += 1;
                    }
                    ends[nb] = n;
                    nb += 1;
                }
            }
            const st = online(.{}, .{ .m = -inf }, keys[0..n], ends[0..nb], o, eo);
            const slot = (qh * w.w + node) * n_ch + c;
            want[slot] = st.m;
            want[slots + slot] = st.l;
            @memcpy(want[2 * slots + slot * WD ..][0..WD], o);
            if (mag) |m| {
                m[slot] = host.allowance(st.em);
                m[slots + slot] = host.allowance(st.el);
                for (m[2 * slots + slot * WD ..][0..WD], eo) |*x, e| x.* = host.allowance(e);
            }
        };
    }

    pub fn reference(self: *const RowPart, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const RowPart, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    /// The declared partial then merge on the same inputs, the partials read back between them.
    fn chained(self: *RowPart, a: std.mem.Allocator) ![]u8 {
        try self.run();
        const parts = try rebind(a, &self.rig, .{ 0, 1, 2 });
        const h = cInt(self.shape.rows * self.win.g);
        const w = cInt(self.shape.n);
        var g = metal.Geometry{ .grid = .{ 32, h, w }, .tg = .{ 32, 1, 1 } };
        g.out(&.{ 1, h, w, WD }, .bfloat16);
        g.int("D", WD);
        try self.rig.launch(&row_merge_launch, g, &.{ parts[0], parts[1], parts[2], self.args[6] });
        return a.dupe(u8, self.rig.outputBytes());
    }

    fn wrapper(self: *RowPart) ![]const u8 {
        const x = self.args;
        var out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out);
        if (!try row_attn.sdpa(&out, x[0], x[1], x[2], @floatCast(self.scale), self.win.tree_(x[0..5].*), self.rig.stream())) return error.Declined;
        return self.rig.bytesOf(&.{out});
    }
};

/// Chunk states for a merge kernel's random fill: a max, a positive sum, an output row; some chunks empty (-Inf, 0,
/// 0) as a row's chunks past its last key are. Gate 4's: max 0, sum 1, outputs whose halves cancel but for small
/// integers, so the true sum is small while its partial sums leave fp16's range.
fn drawStates(a: std.mem.Allocator, rng: std.Random, rows: usize, nch: usize, d: usize, fill: spec.Fill, m: *[]f64, l: *[]f64, o: *[]f64) !void {
    m.* = try host.uniform(a, rng, rows * nch, -3, 3, .float32);
    l.* = try host.uniform(a, rng, rows * nch, 1, 20, .float32);
    o.* = try host.uniform(a, rng, rows * nch * d, -20, 20, .float32);
    if (fill == .stress) {
        @memset(m.*, 0);
        @memset(l.*, 1);
        for (0..rows) |r| for (0..nch) |c| for (0..d) |i| {
            const at = (r * nch + c) * d + i;
            o.*[at] = if (c < nch / 2) @floatFromInt(rng.intRangeAtMost(i32, 20000, 40000)) else if (c < 2 * (nch / 2)) -o.*[at - (nch / 2) * d] + @as(f64, @floatFromInt(rng.intRangeAtMost(i32, 0, 3))) else 1;
        };
        return;
    }
    for (0..rows) |r| if (r % 3 == 1 and nch > 1) {
        m.*[r * nch + nch - 1] = -inf;
        l.*[r * nch + nch - 1] = 0;
        @memset(o.*[(r * nch + nch - 1) * d ..][0..d], 0);
    };
    if (fill == .nonfinite) host.plant(o.*, rows * nch, d);
}

/// row_attn's merge. rows = q heads, n = window rows, k = chunks.
const RowMerge = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    m: []f64 = &.{},
    l: []f64 = &.{},
    o: []f64 = &.{},
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch) RowMerge {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .arena = .init(a) };
    }

    pub fn deinit(self: *RowMerge) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const RowMerge, shape: Shape) ?metal.Geometry {
        if (shape.n < 1 or shape.n > 16 or shape.k < 1) return null;
        var g = metal.Geometry{ .grid = .{ 32, cInt(shape.rows), cInt(shape.n) }, .tg = .{ 32, 1, 1 } };
        g.out(&.{ 1, cInt(shape.rows), cInt(shape.n), WD }, .bfloat16);
        g.int("D", WD);
        return g;
    }

    pub fn prepare(self: *RowMerge, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rows = shape.rows * shape.n;
        try drawStates(a, prng.random(), rows, shape.k, WD, fill, &self.m, &self.l, &self.o);
        self.rig.reset();
        const rig = &self.rig;
        const slots = cInt(rows * shape.k);
        self.args = .{
            try rig.input(.float32, &.{slots}, self.m),
            try rig.input(.float32, &.{slots}, self.l),
            try rig.input(.float32, &.{ slots, WD }, self.o),
            try rig.input(.int32, &.{4}, &.{ 0, @floatFromInt(shape.n), @floatFromInt(shape.k), 1 }),
        };
        return true;
    }

    pub fn run(self: *RowMerge) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const RowMerge) usize {
        return self.shape.rows * self.shape.n * WD;
    }

    fn emit(self: *const RowMerge, want: []f64, mag: ?[]f64, sim: Sim) !void {
        const nch = self.shape.k;
        const a = self.rig.a;
        const eo = try a.alloc(f64, WD);
        defer a.free(eo);
        const t = try a.alloc(f64, nch);
        defer a.free(t);
        for (0..self.shape.rows * self.shape.n) |r| {
            const out = want[r * WD ..][0..WD];
            var st = joinMax(self.m[r * nch ..][0..nch], self.l[r * nch ..][0..nch], self.o[r * nch * WD ..][0 .. nch * WD], WD, out, eo);
            if (sim) |x| {
                // Gate 4: the sums in its accumulator (the stress fill's weights are all 1).
                for (t, 0..) |*v, c| v.* = self.l[r * nch + c];
                st.l = spec.sumIn(x.acc, x.order, t);
                for (out, 0..) |*y, i| {
                    for (t, 0..) |*v, c| v.* = self.o[(r * nch + c) * WD + i];
                    y.* = spec.sumIn(x.acc, x.order, t);
                }
            }
            normalize(st, out, eo);
            if (mag) |m| for (m[r * WD ..][0..WD], eo) |*x, e| {
                x.* = host.allowance(e);
            };
        }
    }

    pub fn reference(self: *const RowMerge, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const RowMerge, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }
};

// lane_attn: 512-key chunks of 64-key tiles; the window's rows in 16-row tiles.
const LANE_CK = 512;
const LANE_TK = 64;
const LANE_TAIL_SG = 4;
/// P's rounding to half before P·V: 2^-11 relative, 2^-25 at least (half's subnormal step).
const lane_soft = Soft{ .p_unit = 0x1p-11, .p_abs = 0x1p-25, .keep_empty = true };

fn lanePartLaunch(comptime sg: usize) metal.Launch {
    return .{
        .name = "msv_lane_attn_partial",
        .inputs = &.{ "Qp", "K", "V", "scale", "dims" },
        .outputs = &.{ "PO", "PM", "PL" },
        .source = lane_attn.PARTIAL,
        .header = lane_attn.HEADER,
        .contiguous = false,
        // Ps [SG * 16 * TK] half.
        .tg_bytes = sg * 16 * LANE_TK * 2,
        .offline = false,
        .min_gen = 17,
    };
}

const lane_tail_launch = metal.Launch{
    .name = "msv_lane_attn_tail",
    .inputs = &.{ "QB", "K", "V", "scale", "dims", "paths", "depths", "POA", "PMA", "PLA" },
    .outputs = &.{ "PO", "PM", "PL" },
    .source = lane_attn.TAIL,
    .header = lane_attn.HEADER,
    .contiguous = false,
    // myP [16 * TK] half, KV [32 * D] bf16.
    .tg_bytes = 16 * LANE_TK * 2 + 32 * WD * 2,
    .offline = false,
    .min_gen = 17,
};

const lane_merge_launch = metal.Launch{
    .name = "msv_lane_attn_merge",
    .inputs = &.{ "POA", "PMA", "PLA", "POB", "PMB", "PLB", "dims" },
    .outputs = &.{"OUT"},
    .source = lane_attn.MERGE,
    .header = lane_attn.HEADER,
    .tg_bytes = 0,
    .offline = false,
    .min_gen = 17,
};

const LaneKind = enum { partial, tail, merge };

/// The window's lane geometry at a shape: the committed keys' whole tiles, their chunks, the tail's chunks, the
/// partial's 16-row tiles.
const LaneDims = struct {
    p: usize,
    pt: usize,
    ca: usize,
    ncb: usize,
    sga: usize,
    rp: usize,
    sg: usize,

    fn of(w: *const Win) LaneDims {
        const p = w.p();
        const pt = p / LANE_TK * LANE_TK;
        const sga = (w.g * w.w + 15) / 16;
        return .{
            .p = p,
            .pt = pt,
            .ca = (pt + LANE_CK - 1) / LANE_CK,
            .ncb = (p + w.maxd - 1) / LANE_CK - pt / LANE_CK + 1,
            .sga = sga,
            .rp = 16 * sga,
            .sg = @min(sga, 16),
        };
    }
};

/// lane_attn's three passes. rows = kv heads, n = window rows, k = keys. The tail and merge take the passes before
/// them as host-drawn states (the partial's for the tail's first chunk; both for the merge).
const Lane = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kind: LaneKind,
    win: Win,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    ld: LaneDims = undefined,
    scale: f64 = 1.0 / 16.0,
    // The states a later pass starts from: the partial's (A) and the tail's (B).
    am: []f64 = &.{},
    al: []f64 = &.{},
    ao: []f64 = &.{},
    bm: []f64 = &.{},
    bl: []f64 = &.{},
    bo: []f64 = &.{},
    x: [5]mlx.mlx_array = undefined,
    args: [10]mlx.mlx_array = undefined,
    nargs: usize = 0,
    owned: [2]mlx.mlx_array = @splat(.{ .ctx = null }),

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, kind: LaneKind, g: usize, tree: bool) Lane {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .kind = kind, .win = .{ .g = g, .tree = tree }, .arena = .init(a) };
    }

    fn freeOwned(self: *Lane) void {
        for (&self.owned) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
    }

    pub fn deinit(self: *Lane) void {
        self.freeOwned();
        self.rig.deinit();
        self.arena.deinit();
    }

    fn dimsAt(self: *const Lane, shape: Shape) ?LaneDims {
        if (shape.n < 1 or shape.n > 16 or shape.k < shape.n or self.win.g > 16) return null;
        var w = self.win;
        w.hkv = shape.rows;
        w.w = shape.n;
        w.l = shape.k;
        if (!w.tree) w.maxd = shape.n;
        // A tree's depth is drawn: bound the tail's chunks by the deepest it can be (the reference uses the drawn).
        if (w.tree and w.maxd == 0) w.maxd = shape.n;
        const ld = LaneDims.of(&w);
        if (self.kind == .partial and ld.ca == 0) return null;
        return ld;
    }

    pub fn geometry(self: *const Lane, shape: Shape) ?metal.Geometry {
        const at = self.dimsAt(shape) orelse return null;
        const ld = if (self.win.w == shape.n and self.win.l == shape.k and self.win.hkv == shape.rows) self.ld else at;
        const hkv = shape.rows;
        const g = self.win.g;
        const d = WD;
        switch (self.kind) {
            .partial => {
                var geo = metal.Geometry{ .grid = .{ cInt(hkv * 32 * ld.sg), cInt(ld.ca), cInt((ld.sga + ld.sg - 1) / ld.sg) }, .tg = .{ cInt(32 * ld.sg), 1, 1 } };
                geo.out(&.{cInt(hkv * ld.ca * ld.rp * d)}, .float32);
                geo.out(&.{cInt(hkv * ld.ca * ld.rp)}, .float32);
                geo.out(&.{cInt(hkv * ld.ca * ld.rp)}, .float32);
                laneDims(&geo, g, null, 0);
                geo.int("SG", ld.sg);
                return geo;
            },
            .tail => {
                const nb = hkv * ld.ncb * shape.n * 16;
                var geo = metal.Geometry{ .grid = .{ cInt(hkv * 32 * LANE_TAIL_SG), cInt(ld.ncb), cInt(shape.n) }, .tg = .{ 32 * LANE_TAIL_SG, 1, 1 } };
                geo.out(&.{cInt(nb * d)}, .float32);
                geo.out(&.{cInt(nb)}, .float32);
                geo.out(&.{cInt(nb)}, .float32);
                const maxd = if (self.win.w == shape.n and self.win.maxd > 0) self.win.maxd else shape.n;
                laneDims(&geo, g, "MAXD", maxd);
                geo.int("TSG", LANE_TAIL_SG);
                return geo;
            },
            .merge => {
                var geo = metal.Geometry{ .grid = .{ cInt(hkv * 32), cInt(shape.n * g), 1 }, .tg = .{ 32, 1, 1 } };
                geo.out(&.{ 1, cInt(hkv * g), cInt(shape.n), d }, .bfloat16);
                laneDims(&geo, g, null, 0);
                return geo;
            },
        }
    }

    pub fn prepare(self: *Lane, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.dimsAt(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        self.freeOwned();
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        try self.win.draw(a, rng, shape, fill);
        self.ld = LaneDims.of(&self.win);
        const ld = self.ld;
        const hkv = shape.rows;
        // Partial states over the A rows and tail states over the B rows (the merge reads both; the tail only A).
        try drawStates(a, rng, hkv * @max(ld.ca, 1) * ld.rp, 1, WD, fill, &self.am, &self.al, &self.ao);
        try drawStates(a, rng, hkv * ld.ncb * shape.n * 16, 1, WD, fill, &self.bm, &self.bl, &self.bo);
        if (fill == .stress) self.stressJoin(rng) else for (self.bm, self.bl, 0..) |*m, *l, i| if (i % 5 == 4) {
            // Chunks past a row's keys hand on (-Inf, 0, 0).
            m.* = -inf;
            l.* = 0;
            @memset(self.bo[i * WD ..][0..WD], 0);
        };
        self.rig.reset();
        const rig = &self.rig;
        self.x = try self.win.bind(a, rig, &self.owned);
        const scale = try rig.input(.float32, &.{1}, &.{self.scale});
        const dims_b = [_]f64{ @floatFromInt(ld.p), @floatFromInt(ld.pt), @floatFromInt(shape.n), @floatFromInt(ld.ncb), @floatFromInt(@max(ld.ca, 1)), @floatFromInt(ld.rp) };
        switch (self.kind) {
            .partial => {
                const dims_a = [_]f64{ @floatFromInt(ld.pt), @floatFromInt(ld.ca), @floatFromInt(shape.n), @floatFromInt(ld.sga) };
                self.args[0..5].* = .{ self.x[0], self.x[1], self.x[2], scale, try rig.input(.int32, &.{4}, &dims_a) };
                self.nargs = 5;
            },
            .tail => {
                const a_in = try self.stateInputs(.a);
                self.args = .{ self.x[0], self.x[1], self.x[2], scale, try rig.input(.int32, &.{6}, &dims_b), try reshapePath(rig, self.x[4]), self.x[3], a_in[0], a_in[1], a_in[2] };
                self.nargs = 10;
            },
            .merge => {
                const a_in = try self.stateInputs(.a);
                const b_in = try self.stateInputs(.b);
                self.args[0..7].* = .{ a_in[0], a_in[1], a_in[2], b_in[0], b_in[1], b_in[2], try rig.input(.int32, &.{6}, &dims_b) };
                self.nargs = 7;
            },
        }
        return true;
    }

    /// Gate 4's merge inputs: per output row, its chunks' outputs cancel in halves but for small integers (the
    /// maxes 0, the sums 1), so the true sum is small while the partial sums leave fp16's range.
    fn stressJoin(self: *Lane, rng: std.Random) void {
        const w = &self.win;
        const ld = self.ld;
        const ct = ld.pt / LANE_CK;
        var at: [64]usize = undefined;
        var side: [64]bool = undefined;
        for (0..w.hkv) |hk| for (0..w.w) |node| for (0..w.g) |g| {
            const n = ct + ld.ncb;
            for (0..n) |c| {
                side[c] = c < ct;
                at[c] = if (c < ct) (hk * ld.ca + c) * ld.rp + g * w.w + node else ((hk * ld.ncb + c - ct) * w.w + node) * 16 + g;
            }
            for (0..WD) |d| for (0..n) |c| {
                const o = if (side[c]) self.ao else self.bo;
                o[at[c] * WD + d] = if (c < n / 2) @floatFromInt(rng.intRangeAtMost(i32, 20000, 40000)) else if (c < 2 * (n / 2)) blk: {
                    const o0 = if (side[c - n / 2]) self.ao else self.bo;
                    break :blk -o0[at[c - n / 2] * WD + d] + @as(f64, @floatFromInt(rng.intRangeAtMost(i32, 0, 3)));
                } else 1;
            };
        };
    }

    /// (PO, PM, PL) of the partial (A; the wrapper's 16-float placeholder without one) or the tail (B).
    fn stateInputs(self: *Lane, which: enum { a, b }) ![3]mlx.mlx_array {
        const rig = &self.rig;
        if (which == .a and self.ld.ca == 0) {
            const z: [16]f64 = @splat(0);
            const ph = try rig.input(.float32, &.{16}, &z);
            return .{ ph, ph, ph };
        }
        const m = if (which == .a) self.am else self.bm;
        const l = if (which == .a) self.al else self.bl;
        const o = if (which == .a) self.ao else self.bo;
        return .{ try rig.input(.float32, &.{cInt(o.len)}, o), try rig.input(.float32, &.{cInt(m.len)}, m), try rig.input(.float32, &.{cInt(l.len)}, l) };
    }

    pub fn run(self: *Lane) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.nargs]);
    }

    pub fn count(self: *const Lane) usize {
        const hkv = self.shape.rows;
        return switch (self.kind) {
            .partial => hkv * self.ld.ca * self.ld.rp * (WD + 2),
            .tail => hkv * self.ld.ncb * self.shape.n * 16 * (WD + 2),
            .merge => hkv * self.win.g * self.shape.n * WD,
        };
    }

    fn emit(self: *const Lane, want: []f64, mag: ?[]f64, sim: Sim) !void {
        const a = self.rig.a;
        const w = &self.win;
        const ld = self.ld;
        const keys = try a.alloc(Key, LANE_CK + 2 * ld.ncb + ld.ca);
        defer a.free(keys);
        const ends = try a.alloc(usize, LANE_CK + 2 * ld.ncb + ld.ca);
        defer a.free(ends);
        const zeros = try a.alloc(f64, WD);
        defer a.free(zeros);
        @memset(zeros, 0);
        const terms = try a.alloc(f64, WD);
        defer a.free(terms);
        const o = try a.alloc(f64, WD);
        defer a.free(o);
        const eo = try a.alloc(f64, WD);
        defer a.free(eo);
        const hkv = self.shape.rows;
        switch (self.kind) {
            .partial => {
                const rows = hkv * ld.ca * ld.rp;
                for (0..hkv) |hk| for (0..ld.ca) |c| for (0..ld.rp) |r| {
                    const live = r < w.g * w.w;
                    var n: usize = 0;
                    var nb: usize = 0;
                    var kt = c * LANE_CK;
                    while (kt < @min(c * LANE_CK + LANE_CK, ld.pt)) : (kt += LANE_TK) {
                        for (kt..kt + LANE_TK) |key| {
                            keys[n] = if (live) blk: {
                                const d = dot(w.qRow(hk * w.g + r / w.w, r % w.w), w.kRow(hk, key), self.scale, WD, 1, sim, terms);
                                break :blk .{ .s = d.v, .e = d.e, .v = w.vRow(hk, key) };
                            } else .{ .s = -inf, .on = false, .v = w.vRow(hk, key) };
                            n += 1;
                        }
                        ends[nb] = n;
                        nb += 1;
                    }
                    const st = online(lane_soft, .{ .m = -inf }, keys[0..n], ends[0..nb], o, eo);
                    const at = (hk * ld.ca + c) * ld.rp + r;
                    putState(want, mag, rows, at, st, o, eo);
                };
            },
            .tail => {
                const rows = hkv * ld.ncb * w.w * 16;
                const c0 = ld.pt / LANE_CK;
                for (0..hkv) |hk| for (0..ld.ncb) |cb| for (0..w.w) |node| for (0..16) |row| {
                    const live = row < w.g;
                    const c = c0 + cb;
                    const nmax = ld.p + w.depth[node] + 1;
                    const kbeg = @max(c * LANE_CK, ld.pt);
                    const kend = @min((c + 1) * LANE_CK, nmax);
                    var start: Init = .{ .m = -inf };
                    if (cb == 0 and ld.pt > c0 * LANE_CK and live) {
                        const at = (hk * ld.ca + c0) * ld.rp + row * w.w + node;
                        start = .{ .m = self.am[at], .l = self.al[at], .o = self.ao[at * WD ..][0..WD] };
                    }
                    var n: usize = 0;
                    var nb: usize = 0;
                    var kt = kbeg;
                    while (kt < kend) : (kt += LANE_TK) {
                        for (kt..kt + LANE_TK) |key| {
                            const vis = live and key < nmax;
                            const v = if (key < nmax) w.vRow(hk, w.phys(node, key)) else zeros;
                            keys[n] = if (vis) blk: {
                                const d = dot(w.qRow(hk * w.g + row, node), w.kRow(hk, w.phys(node, key)), self.scale, WD, 1, sim, terms);
                                break :blk .{ .s = d.v, .e = d.e, .v = v };
                            } else .{ .s = -inf, .on = false, .v = v };
                            n += 1;
                        }
                        ends[nb] = n;
                        nb += 1;
                    }
                    const st = online(lane_soft, start, keys[0..n], ends[0..nb], o, eo);
                    const at = ((hk * ld.ncb + cb) * w.w + node) * 16 + row;
                    putState(want, mag, rows, at, st, o, eo);
                };
            },
            .merge => {
                const ct = ld.pt / LANE_CK;
                for (0..hkv) |hk| for (0..w.w) |node| for (0..w.g) |g| {
                    var n: usize = 0;
                    for (0..ct + ld.ncb) |c| {
                        const a_side = c < ct;
                        const at = if (a_side) (hk * ld.ca + c) * ld.rp + g * w.w + node else ((hk * ld.ncb + c - ct) * w.w + node) * 16 + g;
                        const m = if (a_side) self.am[at] else self.bm[at];
                        if (m == -inf) continue;
                        keys[n] = .{ .s = m, .w = if (a_side) self.al[at] else self.bl[at], .v = (if (a_side) self.ao else self.bo)[at * WD ..][0..WD] };
                        ends[n] = n + 1;
                        n += 1;
                    }
                    var st = online(.{}, .{ .m = -inf }, keys[0..n], ends[0..n], o, eo);
                    if (sim) |x| simJoin(x, keys[0..n], &st, o);
                    normalize(st, o, eo);
                    const at = ((hk * w.g + g) * w.w + node) * WD;
                    @memcpy(want[at..][0..WD], o);
                    if (mag) |mg| for (mg[at..][0..WD], eo) |*x, e| {
                        x.* = host.allowance(e);
                    };
                };
            },
        }
    }

    pub fn reference(self: *const Lane, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Lane, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    /// The declared partial (when the prefix fills a tile), tail and merge on the same window, each pass's states
    /// read back and bound as the next one's.
    fn chained(self: *Lane, a: std.mem.Allocator, part: *const metal.Launch, merge: *const metal.Launch) ![]u8 {
        const rig = &self.rig;
        const ld = self.ld;
        const saved = self.kind;
        defer self.kind = saved;
        var pa: [3]mlx.mlx_array = .{ self.args[7], self.args[8], self.args[9] };
        if (ld.ca > 0) {
            self.kind = .partial;
            const dims_a = [_]f64{ @floatFromInt(ld.pt), @floatFromInt(ld.ca), @floatFromInt(self.shape.n), @floatFromInt(ld.sga) };
            try rig.launch(part, self.geometry(self.shape).?, &.{ self.args[0], self.args[1], self.args[2], self.args[3], try rig.input(.int32, &.{4}, &dims_a) });
            pa = try rebind(a, rig, .{ 0, 1, 2 });
        }
        self.kind = .tail;
        var targs = self.args;
        targs[7..10].* = pa;
        try rig.launch(&lane_tail_launch, self.geometry(self.shape).?, &targs);
        const pb = try rebind(a, rig, .{ 0, 1, 2 });
        self.kind = .merge;
        try rig.launch(merge, self.geometry(self.shape).?, &.{ pa[0], pa[1], pa[2], pb[0], pb[1], pb[2], self.args[4] });
        return a.dupe(u8, rig.outputBytes());
    }

    fn wrapper(self: *Lane) ![]const u8 {
        var out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out);
        if (!try lane_attn.sdpa(&out, self.x[0], self.x[1], self.x[2], @floatCast(self.scale), self.win.tree_(self.x), self.rig.stream())) return error.Declined;
        return self.rig.bytesOf(&.{out});
    }
};

fn laneDims(geo: *metal.Geometry, g: usize, extra: ?[*:0]const u8, v: usize) void {
    geo.int("G", g);
    geo.int("D", WD);
    geo.int("CK", LANE_CK);
    geo.int("TK", LANE_TK);
    if (extra) |n| geo.int(n, v);
}

/// The tree's path [W, MAXD] flattened, as the wrapper passes it.
fn reshapePath(rig: *metal.Rig, path: mlx.mlx_array) !mlx.mlx_array {
    const vals = try host.read(rig.a, path, rig.stream());
    defer rig.a.free(vals);
    return rig.input(.int32, &.{@intCast(vals.len)}, vals);
}

/// A state's (m, l, o) into a partial kernel's outputs (o rows first, then m, then l), with their allowances.
fn putState(want: []f64, mag: ?[]f64, rows: usize, at: usize, st: Row, o: []const f64, eo: []const f64) void {
    @memcpy(want[at * WD ..][0..WD], o);
    want[rows * WD + at] = st.m;
    want[rows * WD + rows + at] = st.l;
    if (mag) |m| {
        for (m[at * WD ..][0..WD], eo) |*x, e| x.* = host.allowance(e);
        m[rows * WD + at] = host.allowance(st.em);
        m[rows * WD + rows + at] = host.allowance(st.el);
    }
}

/// Gate 4 on a merge: the joined sums in its accumulator (the stress fill's maxes are equal, so every weight is 1).
fn simJoin(x: anytype, keys: []const Key, st: *Row, o: []f64) void {
    var t: [64]f64 = undefined;
    for (keys, 0..) |k, i| t[i] = k.w;
    st.l = spec.sumIn(x.acc, x.order, t[0..keys.len]);
    for (o, 0..) |*y, d| {
        for (keys, 0..) |k, i| t[i] = k.v[d];
        y.* = spec.sumIn(x.acc, x.order, t[0..keys.len]);
    }
}

fn winSpec(comptime name: []const u8, out: spec.Elem, served: []const Shape, ck: usize) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = host.depthOne,
        .out = out,
        .tiles = .{ .{}, .{}, .{ .size = ck, .min = 16 } },
        .served = served,
        .sums = .{ .exact = false },
    };
}

const row_part6 = rowPartLaunch(6);
const row_part4 = rowPartLaunch(4);
const lane_part6 = lanePartLaunch(6);
const lane_part4 = lanePartLaunch(4);

/// qwen3_5's full layers (27B: 24 q / 4 kv heads, G 6; 9B and down: G 4) under a bound DFlash drafter: windows of 1
/// to 16 rows, chains or draft trees; lane_attn on the M5's tensor units, row_attn elsewhere.
const row_registrations = .{
    .{ "msv_row_attn_partial (G 6, chain)", &row_part6, 6, false, qk(&.{ .{ 1, 1, 300 }, .{ 1, 5, 700 }, .{ 1, 16, 520 } }) },
    .{ "msv_row_attn_partial (G 6, tree)", &row_part6, 6, true, qk(&.{ .{ 1, 8, 400 }, .{ 1, 16, 300 } }) },
    .{ "msv_row_attn_partial (G 4, chain)", &row_part4, 4, false, qk(&.{.{ 1, 8, 600 }}) },
};

const lane_registrations = .{
    .{ "msv_lane_attn_partial (G 6)", &lane_part6, LaneKind.partial, 6, false, qk(&.{ .{ 1, 1, 600 }, .{ 1, 8, 1100 }, .{ 1, 16, 700 } }) },
    .{ "msv_lane_attn_partial (G 4)", &lane_part4, LaneKind.partial, 4, false, qk(&.{.{ 1, 8, 600 }}) },
    .{ "msv_lane_attn_tail (G 6, chain)", &lane_tail_launch, LaneKind.tail, 6, false, qk(&.{ .{ 1, 1, 300 }, .{ 1, 8, 600 }, .{ 1, 16, 1100 } }) },
    .{ "msv_lane_attn_tail (G 4, chain)", &lane_tail_launch, LaneKind.tail, 4, false, qk(&.{.{ 1, 8, 700 }}) },
    .{ "msv_lane_attn_tail (G 6, tree)", &lane_tail_launch, LaneKind.tail, 6, true, qk(&.{ .{ 1, 8, 700 }, .{ 1, 16, 400 } }) },
    .{ "msv_lane_attn_merge (G 6)", &lane_merge_launch, LaneKind.merge, 6, false, qk(&.{ .{ 1, 8, 1100 }, .{ 1, 1, 600 }, .{ 1, 16, 700 } }) },
    .{ "msv_lane_attn_merge (G 4)", &lane_merge_launch, LaneKind.merge, 4, false, qk(&.{.{ 1, 8, 600 }}) },
};

// ── packed-KV attention (kv-quant 4 / 8): decode, verify widths, the matmul2d page kernel ──

const QkvKind = enum { dec, verify, mpp };

/// One instantiation: the cache's bits and group, q/k head width, query heads per kv head; `mask` binds the
/// additive mask row (decode only).
const QkvCfg = struct { kind: QkvKind, dt: mlx.mlx_dtype = .bfloat16, bits: u32, gs: usize = 64, dk: usize, gqa: usize, mask: bool = false };

const QKV_NT = 128;
const MPP_PAGE = 32;

/// kv_quant_attn's block choices, as it makes them (the drift guard holds them to it).
fn qkvDecBlock(gqa: usize, dk: usize) ?usize {
    for ([_]usize{ 512, 256, 128 }) |b| if (gqa * b * 4 + gqa * dk * 4 + QKV_NT * 4 + 256 <= 10_000) return b;
    return null;
}

fn qkvVerifyBlock(lq: usize) ?usize {
    for ([_]usize{ 256, 128, 64 }) |b| if (lq * b * 4 + QKV_NT * 4 + 8 * lq + 256 <= 16_000) return b;
    return null;
}

fn mppSplits(tk: usize) usize {
    return std.math.clamp((tk + MPP_PAGE - 1) / MPP_PAGE / 8, 1, 128);
}

/// `m`: the M1/M2 limit the offline build gives dec and verify (`caps` where it moves with the shape).
fn qkvLaunch(comptime kind: QkvKind, comptime tg: u32, comptime m: u32, comptime caps: []const metal.Cap) metal.Launch {
    return switch (kind) {
        .dec => .{
            .name = "qkv_attn_dec",
            .inputs = &.{ "q", "kq", "ksc", "kbi", "vq", "vsc", "vbi", "scl", "tk", "msk" },
            .outputs = &.{ "m_out", "l_out", "o_out" },
            .source = kvq.QKV_DEC_KERNEL_SOURCE,
            .contiguous = false,
            .tg_bytes = tg,
            .m1 = m,
            .m2 = m,
            .caps = caps,
        },
        .verify => .{
            .name = "qkv_attn_verify",
            .inputs = &.{ "q", "kq", "ksc", "kbi", "vq", "vsc", "vbi", "scl", "tk" },
            .outputs = &.{ "m_out", "l_out", "o_out" },
            .source = kvq.QKV_VERIFY_KERNEL_SOURCE,
            .contiguous = false,
            .tg_bytes = tg,
            .m1 = m,
            .m2 = m,
            .caps = caps,
        },
        .mpp => .{
            .name = "msv_qkv_mpp",
            .inputs = &.{ "q", "kq", "ksc", "kbi", "vq", "vsc", "vbi", "scl", "tk" },
            .outputs = &.{ "m_out", "l_out", "o_out" },
            .source = kvq.QKV_MPP_KERNEL_SOURCE,
            .header = qsa_score.QSA_SCORE_KERNEL_HEADER,
            .contiguous = false,
            .tg_bytes = tg,
            .offline = false,
            .min_gen = 16,
        },
    };
}

/// rows = kv heads, n = query rows a head (1 at decode), k = keys. The packed K/V are the cache's views (the first
/// Tk rows of longer buffers): codes, scales and biases.
const Qkv = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    cfg: QkvCfg,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    cap: usize = 0,
    scale: f64 = 0,
    q: []f64 = &.{},
    kq: host.Quant = undefined,
    vq: host.Quant = undefined,
    /// K and V as the kernel reads them (f32 dequant, or rounded to T in the page tile) and their errors.
    kv: [2][]f64 = undefined,
    ke: [2][]f64 = undefined,
    mask: []f64 = &.{},
    args: [10]mlx.mlx_array = undefined,
    owned: [6]mlx.mlx_array = @splat(.{ .ctx = null }),

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, cfg: QkvCfg) Qkv {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .cfg = cfg, .arena = .init(a) };
    }

    fn freeOwned(self: *Qkv) void {
        for (&self.owned) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
    }

    pub fn deinit(self: *Qkv) void {
        self.freeOwned();
        self.rig.deinit();
        self.arena.deinit();
    }

    fn mpad(self: *const Qkv, tq: usize) usize {
        return (self.cfg.gqa * tq + 7) / 8 * 8;
    }

    /// Output rows (dec / verify: q heads x rows; mpp: kv heads x padded rows) and blocks (or splits) at a shape.
    fn grid(self: *const Qkv, shape: Shape) ?[2]usize {
        const c = self.cfg;
        const tq = shape.n;
        const tk = shape.k;
        if (tk < tq or tq < 1 or tq > 8) return null;
        switch (c.kind) {
            .dec => {
                if (tq != 1) return null;
                const b = qkvDecBlock(c.gqa, c.dk) orelse return null;
                return .{ shape.rows * c.gqa, (tk + b - 1) / b };
            },
            .verify => {
                if (tq < 2 or c.gqa * tq > QKV_NT) return null;
                const b = qkvVerifyBlock(c.gqa * tq) orelse return null;
                return .{ shape.rows * c.gqa * tq, (tk + b - 1) / b };
            },
            .mpp => {
                if (self.mpad(tq) > 56) return null;
                return .{ shape.rows * self.mpad(tq), mppSplits(tk) };
            },
        }
    }

    fn block(self: *const Qkv, tq: usize) usize {
        return switch (self.cfg.kind) {
            .dec => qkvDecBlock(self.cfg.gqa, self.cfg.dk).?,
            .verify => qkvVerifyBlock(self.cfg.gqa * tq).?,
            .mpp => MPP_PAGE,
        };
    }

    pub fn geometry(self: *const Qkv, shape: Shape) ?metal.Geometry {
        const c = self.cfg;
        const rb = self.grid(shape) orelse return null;
        const hkv = cInt(shape.rows);
        const nb = cInt(rb[1]);
        const dk = cInt(c.dk);
        var g = metal.Geometry{ .grid = .{ if (c.kind == .mpp) 256 else QKV_NT, hkv, nb }, .tg = .{ if (c.kind == .mpp) 256 else QKV_NT, 1, 1 } };
        if (c.kind == .mpp) {
            const m = cInt(self.mpad(shape.n));
            g.out(&.{ hkv, nb, m }, .float32);
            g.out(&.{ hkv, nb, m }, .float32);
            g.out(&.{ hkv, nb, m, dk }, .float32);
        } else {
            const rows = cInt(rb[0]);
            g.out(&.{ rows, nb }, .float32);
            g.out(&.{ rows, nb }, .float32);
            g.out(&.{ rows, nb, dk }, .float32);
        }
        g.dtype("T", c.dt);
        g.int("BITS", c.bits);
        g.int("GS", c.gs);
        g.int("DK", c.dk);
        if (c.kind != .mpp) g.int("DV", c.dk);
        g.int("GQA", c.gqa);
        switch (c.kind) {
            .dec => {
                g.int("BLOCK", self.block(1));
                g.int("HAS_MASK", @intFromBool(c.mask));
            },
            .verify => {
                g.int("TQ", shape.n);
                g.int("BLOCK", self.block(shape.n));
            },
            .mpp => {
                g.int("TQ", shape.n);
                g.int("MPAD", self.mpad(shape.n));
            },
        }
        return g;
    }

    /// Codes for `rows` cache rows: random, or gate 4's (each row's halves equal codes 0..14 but for +1 at two places,
    /// so with q's +37 / -37 halves a dot is small while its partial sums leave fp16's range).
    fn codes(a: std.mem.Allocator, rng: std.Random, rows: usize, d: usize, bits: u32, stress: bool) ![]u32 {
        const out = try a.alloc(u32, rows * d);
        const top: u32 = (@as(u32, 1) << @intCast(bits)) - 1;
        for (out) |*x| x.* = if (stress) rng.uintAtMost(u32, 14) else rng.uintAtMost(u32, top);
        if (stress) for (0..rows) |r| for (0..d / 2) |i| {
            out[r * d + d / 2 + i] = out[r * d + i] + @as(u32, if (i < 2) rng.uintLessThan(u32, 2) else 0);
        };
        return out;
    }

    pub fn prepare(self: *Qkv, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        self.freeOwned();
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const c = self.cfg;
        const hkv = shape.rows;
        const tq = shape.n;
        const tk = shape.k;
        self.cap = tk + 40;
        self.scale = host.rt(.float32, 1 / @sqrt(@as(f64, @floatFromInt(c.dk))));
        const rows = hkv * self.cap;
        if (fill == .stress) {
            // 151 (exact in bf16): products past f16's 2048 integer range at 4-bit codes, sums past its 65504.
            self.q = try a.alloc(f64, hkv * c.gqa * tq * c.dk);
            for (self.q, 0..) |*x, i| x.* = if (i % c.dk < c.dk / 2) 151 else -151;
        } else self.q = try host.uniform(a, rng, hkv * c.gqa * tq * c.dk, -2, 2, c.dt);
        if (fill == .nonfinite) host.plant(self.q, hkv * c.gqa * tq, c.dk);
        for ([_]*host.Quant{ &self.kq, &self.vq }, 0..) |qt, i| {
            qt.* = try host.quantize(a, rng, rows, c.dk, c.bits, c.gs, c.dt);
            if (fill == .stress) {
                @memset(qt.scales, 1);
                @memset(qt.biases, 0);
            }
            // V's NaN / Inf ride its scales: every element of a planted group.
            if (fill == .nonfinite and i == 1) host.plant(qt.scales, rows, c.dk / c.gs);
            host.dequant(qt, rows, c.dk, c.bits, c.gs, try codes(a, rng, rows, c.dk, c.bits, fill == .stress and i == 0));
            // f32 dequant (one rounding), or that rounded to T in the page tile.
            self.kv[i] = try a.alloc(f64, qt.w.len);
            self.ke[i] = try a.alloc(f64, qt.w.len);
            for (self.kv[i], self.ke[i], qt.w) |*v, *e, w| {
                const x = host.Val{ .v = w, .e = host.U * @abs(w) };
                const r = if (c.kind == .mpp) host.round(c.dt, x) else x;
                v.* = r.v;
                e.* = r.e;
            }
        }
        self.mask = try a.alloc(f64, tk);
        for (self.mask, 0..) |*m, t| m.* = if (t < 3) -inf else if (t % 7 == 0) -2.5 else 0;
        try self.bind();
        return true;
    }

    fn bind(self: *Qkv) !void {
        const rig = &self.rig;
        rig.reset();
        const c = self.cfg;
        const hkv = cInt(self.shape.rows);
        const tq = self.shape.n;
        const st = rig.stream();
        const a = self.arena.allocator();
        if (c.kind == .mpp) {
            // [1, Hq, TQ, D] is [Hkv, GQA x TQ, D]: zero-padded to MPAD rows, as the wrapper pads it.
            const mp = self.mpad(tq);
            const mr = c.gqa * tq;
            const qp = try a.alloc(f64, self.shape.rows * mp * c.dk);
            @memset(qp, 0);
            for (0..self.shape.rows) |hk| for (0..mr) |m| @memcpy(qp[(hk * mp + m) * c.dk ..][0..c.dk], self.q[(hk * mr + m) * c.dk ..][0..c.dk]);
            self.args[0] = try rig.input(c.dt, &.{ hkv, cInt(mp), cInt(c.dk) }, qp);
        } else self.args[0] = try rig.input(c.dt, &.{ 1, hkv * cInt(c.gqa), cInt(tq), cInt(c.dk) }, self.q);
        const cap = cInt(self.cap);
        const tk = cInt(self.shape.k);
        const wpr = cInt(c.dk * c.bits / 32);
        const gpr = cInt(c.dk / c.gs);
        for ([_]*const host.Quant{ &self.kq, &self.vq }, 0..) |qt, i| {
            const parts = [_]struct { mlx.mlx_dtype, c_int, []const f64 }{ .{ .uint32, wpr, qt.words }, .{ c.dt, gpr, qt.scales }, .{ c.dt, gpr, qt.biases } };
            for (parts, 0..) |p, j| {
                const full = try rig.input(p[0], &.{ 1, hkv, cap, p[1] }, p[2]);
                const o = &self.owned[i * 3 + j];
                o.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(o, full, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, hkv, tk, p[1] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, st));
                self.args[1 + i * 3 + j] = o.*;
            }
        }
        self.args[7] = try rig.input(.float32, &.{1}, &.{self.scale});
        self.args[8] = try rig.input(.int32, &.{1}, &.{@floatFromInt(self.shape.k)});
        if (c.kind == .dec) self.args[9] = if (c.mask) try rig.input(c.dt, &.{ 1, 1, 1, tk }, self.mask) else try rig.input(.float32, &.{1}, &.{0});
    }

    pub fn run(self: *Qkv) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..if (self.cfg.kind == .dec) 10 else 9]);
    }

    pub fn count(self: *const Qkv) usize {
        const rb = self.grid(self.shape).?;
        return rb[0] * rb[1] * (2 + self.cfg.dk);
    }

    fn emit(self: *const Qkv, want: []f64, mag: ?[]f64, sim: Sim) !void {
        const a = self.rig.a;
        const c = self.cfg;
        const d = c.dk;
        const tq = self.shape.n;
        const tk = self.shape.k;
        const rb = self.grid(self.shape).?;
        const nrows = rb[0] * rb[1];
        const blk = self.block(tq);
        const keys = try a.alloc(Key, blk + tk + 2 * MPP_PAGE);
        defer a.free(keys);
        const ends = try a.alloc(usize, tk / MPP_PAGE + 2);
        defer a.free(ends);
        const terms = try a.alloc(f64, d);
        defer a.free(terms);
        const zeros = try a.alloc(f64, d);
        defer a.free(zeros);
        @memset(zeros, 0);
        const o = try a.alloc(f64, d);
        defer a.free(o);
        const eo = try a.alloc(f64, d);
        defer a.free(eo);
        const qs = try a.alloc(f64, d);
        defer a.free(qs);
        const kr = struct {
            fn f(x: []const f64, cap: usize, dd: usize, hk: usize, t: usize) []const f64 {
                return x[(hk * cap + t) * dd ..][0..dd];
            }
        }.f;
        if (c.kind == .mpp) {
            const mp = self.mpad(tq);
            const mr = c.gqa * tq;
            const ns = rb[1];
            const pages = (tk + MPP_PAGE - 1) / MPP_PAGE;
            const per = (pages + ns - 1) / ns;
            for (0..self.shape.rows) |hk| for (0..ns) |split| for (0..mp) |m| {
                const qrow = if (m < mr) self.q[(hk * mr + m) * d ..][0..d] else zeros;
                const lim = if (m < mr) tk - tq + 1 + m % tq else tk;
                var n: usize = 0;
                var nb: usize = 0;
                var page = split * per;
                while (page < @min(pages, split * per + per)) : (page += 1) {
                    for (page * MPP_PAGE..page * MPP_PAGE + MPP_PAGE) |tok| {
                        const r = @min(tok, tk - 1);
                        const v = kr(self.kv[1], self.cap, d, hk, r);
                        const ve = kr(self.ke[1], self.cap, d, hk, r);
                        keys[n] = if (tok < lim) blk2: {
                            const x = dotErr(qrow, kr(self.kv[0], self.cap, d, hk, r), kr(self.ke[0], self.cap, d, hk, r), self.scale, d, 1, sim, terms);
                            break :blk2 .{ .s = x.v, .e = x.e, .v = v, .ve = ve };
                        } else .{ .s = -inf, .on = false, .v = v };
                        n += 1;
                    }
                    ends[nb] = n;
                    nb += 1;
                }
                var st = online(.{}, .{ .m = -inf }, keys[0..n], ends[0..nb], o, eo);
                st.m = @max(st.m, -1e30);
                putQkv(want, mag, nrows, d, (hk * ns + split) * mp + m, st, o, eo);
            };
            return;
        }
        const nblk = rb[1];
        for (0..self.shape.rows) |hk| for (0..c.gqa) |gi| for (0..tq) |t| for (0..nblk) |b| {
            const qrow = self.q[(((hk * c.gqa + gi) * tq) + t) * d ..][0..d];
            for (qs, qrow) |*x, y| x.* = host.rt(.float32, y * self.scale);
            const t0 = b * blk;
            var n: usize = 0;
            for (t0..@min(t0 + blk, tk)) |pos| {
                const v = kr(self.kv[1], self.cap, d, hk, pos);
                const ve = kr(self.ke[1], self.cap, d, hk, pos);
                // Decode stages q * scale in f32 and sums against it; verify sums q, then scales.
                const x = if (c.kind == .dec)
                    dotErr(qs, kr(self.kv[0], self.cap, d, hk, pos), kr(self.ke[0], self.cap, d, hk, pos), 1, d, 0, sim, terms)
                else
                    dotErr(qrow, kr(self.kv[0], self.cap, d, hk, pos), kr(self.ke[0], self.cap, d, hk, pos), self.scale, d, 1, sim, terms);
                const madd = if (c.kind == .dec and c.mask) self.mask[pos] else 0;
                const vis = if (c.kind == .verify) pos < tk - tq + 1 + t else madd != -inf;
                keys[n] = if (vis) .{ .s = x.v + madd, .e = x.e + host.U * @abs(x.v + madd), .v = v, .ve = ve } else .{ .s = -inf, .on = false, .v = v };
                n += 1;
            }
            const st = online(.{}, .{ .m = -1e30 }, keys[0..n], &.{n}, o, eo);
            const row = (hk * c.gqa + gi) * tq + t;
            putQkv(want, mag, nrows, d, row * nblk + b, st, o, eo);
        };
    }

    pub fn reference(self: *const Qkv, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const Qkv, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    fn view(self: *const Qkv) kv_view.DenseKVView {
        const x = self.args;
        return .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .k_triple_q = x[1], .k_triple_scales = x[2], .k_triple_biases = x[3], .v_triple_q = x[4], .v_triple_scales = x[5], .v_triple_biases = x[6], .has_quant_triple = true, .bits = @intCast(self.cfg.bits), .group_size = @intCast(self.cfg.gs) };
    }

    /// The declared launch, then the wrapper's own merge of the partials (the mpp's after its transpose and pad-row
    /// cut): [1, Hq, TQ, D] bytes.
    fn merged(self: *Qkv) ![]const u8 {
        try self.run();
        const c = self.cfg;
        const st = self.rig.stream();
        var g = ops.Graph.init(self.rig.a, st);
        defer g.deinit();
        const hkv = cInt(self.shape.rows);
        const tq = cInt(self.shape.n);
        const hq = hkv * cInt(c.gqa);
        const d = cInt(c.dk);
        var m = self.rig.output(0);
        var l = self.rig.output(1);
        var o = self.rig.output(2);
        if (c.kind == .mpp) {
            const ns = cInt(self.grid(self.shape).?[1]);
            const mr = cInt(c.gqa) * tq;
            m = try g.reshape(try g.slice(try g.transpose(m, &.{ 0, 2, 1 }), &.{ 0, 0, 0 }, &.{ hkv, mr, ns }), &.{ hkv * mr, ns });
            l = try g.reshape(try g.slice(try g.transpose(l, &.{ 0, 2, 1 }), &.{ 0, 0, 0 }, &.{ hkv, mr, ns }), &.{ hkv * mr, ns });
            o = try g.reshape(try g.slice(try g.transpose(o, &.{ 0, 2, 1, 3 }), &.{ 0, 0, 0, 0 }, &.{ hkv, mr, ns, d }), &.{ hkv * mr, ns, d });
        }
        const out = try g.adopt(try kvq.qkvMergeArrays(st, m, l, o, c.dt));
        return self.rig.bytesOf(&.{try g.reshape(out, &.{ 1, hq, tq, d })});
    }

    fn wrapper(self: *Qkv) ![]const u8 {
        const c = self.cfg;
        const vw = self.view();
        kvq.qkv_ver_override = true;
        defer kvq.qkv_ver_override = null;
        const st = self.rig.stream();
        const got = switch (c.kind) {
            .dec => try kvq.qkvAttnDecodeKernel(st, self.args[0], &vw, @floatCast(self.scale), if (c.mask) "array" else "", if (c.mask) self.args[9] else .{ .ctx = null }),
            .verify => try kvq.qkvAttnVerifyKernel(st, self.args[0], &vw, @floatCast(self.scale), "causal"),
            .mpp => blk: {
                // The wrapper pads q itself: hand it the [1, Hq, TQ, D] rows.
                const q = try self.rig.input(c.dt, &.{ 1, cInt(self.shape.rows * c.gqa), cInt(self.shape.n), cInt(c.dk) }, self.q);
                break :blk try kvq.qkvAttnMppKernel(st, q, &vw, @floatCast(self.scale), "causal");
            },
        } orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        return self.rig.bytesOf(&.{got});
    }
};

/// A packed kernel's (m, l, o) partial at row `at` of `rows` (m's, then l's, then o's), with their allowances.
fn putQkv(want: []f64, mag: ?[]f64, rows: usize, d: usize, at: usize, st: Row, o: []const f64, eo: []const f64) void {
    want[at] = st.m;
    want[rows + at] = st.l;
    @memcpy(want[2 * rows + at * d ..][0..d], o);
    if (mag) |m| {
        m[at] = host.allowance(st.em);
        m[rows + at] = host.allowance(st.el);
        for (m[2 * rows + at * d ..][0..d], eo) |*x, e| x.* = host.allowance(e);
    }
}

fn qkvTg(comptime c: QkvCfg, comptime tq: usize) u32 {
    return switch (c.kind) {
        .dec => c.gqa * c.dk * 4 + c.gqa * qkvDecBlock(c.gqa, c.dk).? * 4 + QKV_NT * 4 + 8 * c.gqa,
        .verify => c.gqa * tq * qkvVerifyBlock(c.gqa * tq).? * 4 + QKV_NT * 4 + 8 * c.gqa * tq,
        // tile [32 * D] T, scores and probs [MPAD * 32] f32, rmax / rsum / pscale [MPAD] f32.
        .mpp => MPP_PAGE * c.dk * 2 + 2 * ((c.gqa * tq + 7) / 8 * 8) * MPP_PAGE * 4 + 3 * ((c.gqa * tq + 7) / 8 * 8) * 4,
    };
}

fn qkvSpec(comptime name: []const u8, cfg: QkvCfg, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = host.depthOne,
        .out = .f32,
        .tiles = .{ .{}, .{}, .{ .size = if (cfg.kind == .mpp) MPP_PAGE else 128, .min = 16 } },
        .served = served,
        // Decode stages q x 1/sqrt(DK) in f32 before its sums: off a power of two (DK 128) no input makes them exact.
        .sums = if (cfg.kind == .dec and cfg.dk == 128) .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "q is staged as q / sqrt(128) in f32, so no input makes its sums exact in f32" } else .{ .exact = false },
    };
}

const qkv_dec_4_256 = QkvCfg{ .kind = .dec, .bits = 4, .dk = 256, .gqa = 6 };
const qkv_dec_8_256 = QkvCfg{ .kind = .dec, .bits = 8, .dk = 256, .gqa = 6 };
const qkv_dec_8_128m = QkvCfg{ .kind = .dec, .bits = 8, .dk = 128, .gqa = 4, .mask = true };
const qkv_ver_4_256 = QkvCfg{ .kind = .verify, .bits = 4, .dk = 256, .gqa = 6 };
const qkv_ver_8_256 = QkvCfg{ .kind = .verify, .bits = 8, .dk = 256, .gqa = 6 };
const qkv_ver_8_128 = QkvCfg{ .kind = .verify, .bits = 8, .dk = 128, .gqa = 4 };
const qkv_mpp_8 = QkvCfg{ .kind = .mpp, .bits = 8, .dk = 256, .gqa = 6 };
const qkv_mpp_4 = QkvCfg{ .kind = .mpp, .bits = 4, .dk = 256, .gqa = 6 };

const qkv_dec_4_256_l = qkvLaunch(.dec, qkvTg(qkv_dec_4_256, 1), 1024, &.{});
const qkv_dec_8_256_l = qkvLaunch(.dec, qkvTg(qkv_dec_8_256, 1), 1024, &.{});
const qkv_dec_8_128m_l = qkvLaunch(.dec, qkvTg(qkv_dec_8_128m, 1), 1024, &.{});
const qkv_ver_4_256_l = qkvLaunch(.verify, qkvTg(qkv_ver_4_256, 2), 576, &.{});
const qkv_ver_8_256_l = qkvLaunch(.verify, qkvTg(qkv_ver_8_256, 2), 640, &.{});
const qkv_ver_128_l = qkvLaunch(.verify, qkvTg(qkv_ver_8_128, 3), 640, &.{.{ .n = 2, .m1 = 832, .m2 = 832 }});
const qkv_mpp_l = qkvLaunch(.mpp, qkvTg(qkv_mpp_8, 8), 0, &.{});

/// --kv-quant 4 / 8 on qwen3_5's full layers (24 q / 4 kv, hd 256) and a Llama-class hd 128 (32 / 8; masked: a
/// sliding layer's decode row): decode on every chip; the verify widths off the M5 (where dense wins, so it is
/// opt-in there), GQA x TQ up to 12; the matmul2d page kernel on M4 and M5 past its measured window (t_q 4+ from 2K
/// keys, 2-3 from 8K, 1 from 16K).
const qkv_registrations = .{
    .{ "qkv_attn_dec bf16 (4-bit, DK 256, GQA 6)", qkv_dec_4_256, qk(&.{ .{ 1, 1, 300 }, .{ 2, 1, 1000 } }), &qkv_dec_4_256_l },
    .{ "qkv_attn_dec bf16 (8-bit, DK 256, GQA 6)", qkv_dec_8_256, qk(&.{ .{ 1, 1, 300 }, .{ 1, 1, 700 } }), &qkv_dec_8_256_l },
    .{ "qkv_attn_dec bf16 (8-bit, DK 128, GQA 4, mask row)", qkv_dec_8_128m, qk(&.{ .{ 1, 1, 300 }, .{ 2, 1, 600 } }), &qkv_dec_8_128m_l },
    .{ "qkv_attn_verify bf16 (4-bit, DK 256, GQA 6)", qkv_ver_4_256, qk(&.{ .{ 1, 2, 300 }, .{ 1, 2, 900 } }), &qkv_ver_4_256_l },
    .{ "qkv_attn_verify bf16 (8-bit, DK 256, GQA 6)", qkv_ver_8_256, qk(&.{ .{ 1, 2, 300 }, .{ 1, 2, 700 } }), &qkv_ver_8_256_l },
    .{ "qkv_attn_verify bf16 (8-bit, DK 128, GQA 4)", qkv_ver_8_128, qk(&.{ .{ 1, 3, 300 }, .{ 1, 2, 500 } }), &qkv_ver_128_l },
    .{ "msv_qkv_mpp bf16 (8-bit, GQA 6)", qkv_mpp_8, qk(&.{ .{ 1, 4, 2100 }, .{ 1, 8, 2048 }, .{ 1, 2, 8200 }, .{ 1, 1, 16400 } }), &qkv_mpp_l },
    .{ "msv_qkv_mpp bf16 (4-bit, GQA 6)", qkv_mpp_4, qk(&.{ .{ 1, 4, 2100 }, .{ 1, 8, 2300 } }), &qkv_mpp_l },
};

// ── NAX prefill attention: MLX's tensor-unit flash kernel with a value head dim and key-range passes (oMLX) ──

const NAX_BK = 32;
const NAX_THREADS = 128;
const NAX_HEADER = kernelSource("kernels/attention/nax_tiles.metal") ++ kernelSource("kernels/attention/nax_attention.metal");
/// The masked score: f32's lowest finite value, not -Inf (a row's first fully masked blocks weigh 1 until a key
/// shows, whose rescale then wipes them).
const FINITE_MIN: f64 = -std.math.floatMax(f32);

/// One instantiation: q/k width (v is 128), the sliding window, sinks; `phase` single is one dispatch (WN 1),
/// out / through / in the key-range passes (WN 2). `q8`: the int8 Q/K hd-256 entry (WN 2, causal).
const NaxCfg = struct {
    dt: mlx.mlx_dtype = .bfloat16,
    dk: usize = 192,
    gqa: usize = 2,
    window: usize = 0,
    sinks: bool = false,
    phase: Phase = .single,
    q8: bool = false,
};

fn naxLaunch(comptime q8: bool, comptime wn: u32) metal.Launch {
    return .{
        .name = if (q8) "msv_nax_attn_256_q8" else "msv_nax_attn_dk_128",
        .inputs = if (q8) &.{ "q", "k", "v", "sinks", "state", "params", "qi", "ki", "qs", "ks" } else &.{ "q", "k", "v", "sinks", "state", "params" },
        .outputs = &.{ "out", "state_out" },
        .source = if (q8) nax.SOURCE_256Q8 else nax.SOURCE,
        .header = NAX_HEADER,
        .contiguous = false,
        // xchg [WM * 2 * 16 * 32] f32 when the head dims split over simdgroup pairs.
        .tg_bytes = if (q8) 4 * 2 * 16 * 32 * 4 else if (wn == 2) 2 * 2 * 16 * 32 * 4 else 4,
        .offline = false,
        .min_gen = 17,
    };
}

/// rows = kv heads, n = query length, k = key length.
const NaxAttn = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    cfg: NaxCfg,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    sinks: []f64 = &.{},
    state: []f64 = &.{},
    qi: []f64 = &.{},
    ki: []f64 = &.{},
    qs: []f64 = &.{},
    ks: []f64 = &.{},
    scale: f64 = 0,
    args: [10]mlx.mlx_array = undefined,
    owned: [2]mlx.mlx_array = @splat(.{ .ctx = null }),

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, cfg: NaxCfg) NaxAttn {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .cfg = cfg, .arena = .init(a) };
    }

    fn freeOwned(self: *NaxAttn) void {
        for (&self.owned) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
    }

    pub fn deinit(self: *NaxAttn) void {
        self.freeOwned();
        self.rig.deinit();
        self.arena.deinit();
    }

    fn wn(self: *const NaxAttn) usize {
        return if (self.cfg.q8 or self.cfg.phase != .single) 2 else 1;
    }

    fn rowsPerTile(self: *const NaxAttn) usize {
        return if (self.cfg.q8) 64 else 16 * (4 / self.wn());
    }

    fn vWidth(self: *const NaxAttn) usize {
        return if (self.cfg.q8) 256 else 128;
    }

    fn last(phase: Phase) bool {
        return phase == .single or phase == .in;
    }

    pub fn geometry(self: *const NaxAttn, shape: Shape) ?metal.Geometry {
        return self.geometryAt(shape, self.cfg.phase);
    }

    fn geometryAt(self: *const NaxAttn, shape: Shape, phase: Phase) ?metal.Geometry {
        const c = self.cfg;
        const ql = shape.n;
        const kl = shape.k;
        if (kl < ql or (c.q8 and ql < 64) or (!c.q8 and ql <= 8)) return null;
        const h = cInt(shape.rows * c.gqa);
        const bq = self.rowsPerTile();
        const nq = (ql + bq - 1) / bq;
        const threads = cInt(if (c.q8) 256 else NAX_THREADS);
        var g = metal.Geometry{ .grid = .{ cInt(nq) * threads, h, 1 }, .tg = .{ threads, 1, 1 } };
        if (last(phase)) {
            g.out(&.{ 1, cInt(ql), h, cInt(self.vWidth()) }, c.dt);
            g.out(&.{1}, .float32);
            g.unread = 0b10;
        } else {
            g.out(&.{1}, c.dt);
            g.out(&.{cInt(shape.rows * c.gqa * nq * bq * 130)}, .float32);
            g.unread = 0b01;
        }
        g.dtype("T", c.dt);
        g.int("WM", if (c.q8) 4 else 4 / self.wn());
        if (!c.q8) {
            g.int("WN", self.wn());
            g.int("WINDOW", c.window);
            g.int("DK", c.dk);
        }
        g.flag("ALIGN_Q", ql % bq == 0);
        g.flag("ALIGN_K", kl % NAX_BK == 0);
        if (!c.q8) {
            g.flag("HAS_SINKS", c.sinks);
            g.flag("FIRST", phase == .single or phase == .out);
            g.flag("LAST", last(phase));
        }
        return g;
    }

    /// The pass's key blocks: all of them, or a third (the wrapper's edges at three passes).
    fn range(phase: Phase, nk: usize) [2]usize {
        return switch (phase) {
            .single => .{ 0, nk },
            .out => .{ 0, nk / 3 },
            .through => .{ nk / 3, 2 * nk / 3 },
            .in => .{ 2 * nk / 3, nk },
        };
    }

    pub fn prepare(self: *NaxAttn, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        self.freeOwned();
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const c = self.cfg;
        const hk = shape.rows;
        const h = hk * c.gqa;
        const ql = shape.n;
        const kl = shape.k;
        const nq = (ql + self.rowsPerTile() - 1) / self.rowsPerTile();
        const nk = (kl + NAX_BK - 1) / NAX_BK;
        if (fill == .stress) {
            self.q = try a.alloc(f64, h * ql * c.dk);
            self.k = try host.ints(a, rng, hk * kl * c.dk, 0, 127);
            stressRows(rng, self.q, self.k, c.dk, STRESS_Q);
        } else {
            self.q = try host.uniform(a, rng, h * ql * c.dk, -2, 2, c.dt);
            self.k = try host.uniform(a, rng, hk * kl * c.dk, -2, 2, c.dt);
        }
        self.v = try host.uniform(a, rng, hk * kl * self.vWidth(), -2, 2, c.dt);
        if (fill == .nonfinite) {
            host.plant(self.q, h * ql, c.dk);
            host.plant(self.v, hk * kl, self.vWidth());
        }
        self.sinks = try host.uniform(a, rng, h, -1, 3, c.dt);
        self.scale = host.rt(.float32, 1 / @sqrt(@as(f64, @floatFromInt(c.dk))));
        // A carried state per row (the padded tail rows too): O, max (log2 units), sum.
        self.state = try a.alloc(f64, h * nq * self.rowsPerTile() * 130);
        for (0..self.state.len / 130) |r| {
            const row = self.state[r * 130 ..][0..130];
            for (row[0..128]) |*x| x.* = host.rt(.float32, -8 + 16 * rng.float(f64));
            row[128] = host.rt(.float32, -2 + 5 * rng.float(f64));
            row[129] = host.rt(.float32, 1 + 7 * rng.float(f64));
        }
        if (c.q8) {
            // The int8 rows and scales the wrapper quantizes Q (per 64-row tile) and K (per 32-key block) to.
            self.qi = try host.ints(a, rng, h * ql * 256, -127, 127);
            self.ki = try host.ints(a, rng, hk * kl * 256, if (fill == .stress) 0 else -127, if (fill == .stress) 126 else 127);
            self.qs = try host.uniform(a, rng, h * nq, 0.005, 0.02, .float32);
            self.ks = try host.uniform(a, rng, hk * nk, 0.005, 0.02, .float32);
            if (fill == .stress) {
                stressRows(rng, self.qi, self.ki, 256, 113);
                @memset(self.qs, 2);
                @memset(self.ks, 2);
            }
            if (fill == .nonfinite) self.qs[0] = std.math.nan(f64);
        }
        try self.bind();
        return true;
    }

    fn params(self: *const NaxAttn, phase: Phase) [16]f64 {
        const c = self.cfg;
        const ql = self.shape.n;
        const kl = self.shape.k;
        const bq = self.rowsPerTile();
        const nq = (ql + bq - 1) / bq;
        const nk = (kl + NAX_BK - 1) / NAX_BK;
        const r = range(phase, nk);
        const sc: u32 = @bitCast(@as(f32, @floatCast(self.scale)));
        const vals = [16]i64{ 1, @intCast(self.shape.rows * c.gqa), @intCast(c.dk), @intCast(ql), @intCast(kl), @intCast(c.gqa), @as(i32, @bitCast(sc)), @intCast(nq), @intCast(nk), @intCast(ql / bq), @intCast(kl / NAX_BK), @intCast(ql - ql / bq * bq), @intCast(kl - kl / NAX_BK * NAX_BK), @intCast(kl - ql), @intCast(r[0]), @intCast(r[1]) };
        var out: [16]f64 = undefined;
        for (&out, vals) |*o, v| o.* = @floatFromInt(v);
        return out;
    }

    fn bind(self: *NaxAttn) !void {
        const rig = &self.rig;
        rig.reset();
        const c = self.cfg;
        const hk = cInt(self.shape.rows);
        const h = hk * cInt(c.gqa);
        const ql = cInt(self.shape.n);
        const kl = cInt(self.shape.k);
        const a = self.arena.allocator();
        const cap = kl + 24;
        var pad = std.Random.DefaultPrng.init(91);
        var kv: [2]mlx.mlx_array = undefined;
        for ([_][]const f64{ self.k, self.v }, [_]usize{ c.dk, self.vWidth() }, 0..) |src, d, i| {
            const buf = try host.uniform(a, pad.random(), self.shape.rows * @as(usize, @intCast(cap)) * d, -1, 1, c.dt);
            for (0..self.shape.rows) |x| @memcpy(buf[x * @as(usize, @intCast(cap)) * d ..][0 .. self.shape.k * d], src[x * self.shape.k * d ..][0 .. self.shape.k * d]);
            const full = try rig.input(c.dt, &.{ 1, hk, cap, cInt(d) }, buf);
            self.owned[i] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&self.owned[i], full, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, hk, kl, cInt(d) }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, rig.stream()));
            kv[i] = self.owned[i];
        }
        const phase = c.phase;
        const pv = self.params(phase);
        self.args[0..6].* = .{
            try rig.input(c.dt, &.{ 1, h, ql, cInt(c.dk) }, self.q),
            kv[0],
            kv[1],
            if (c.sinks) try rig.input(c.dt, &.{h}, self.sinks) else try rig.input(c.dt, &.{1}, &.{0}),
            if (phase == .single or phase == .out) try rig.input(.float32, &.{1}, &.{0}) else try rig.input(.float32, &.{cInt(self.state.len)}, self.state),
            try rig.input(.int32, &.{16}, &pv),
        };
        if (c.q8) self.args[6..10].* = .{
            try rig.input(.int8, &.{ 1, h, ql, 256 }, self.qi),
            try rig.input(.int8, &.{ 1, hk, kl, 256 }, self.ki),
            try rig.input(.float32, &.{cInt(self.qs.len)}, self.qs),
            try rig.input(.float32, &.{cInt(self.ks.len)}, self.ks),
        };
    }

    fn nargs(self: *const NaxAttn) usize {
        return if (self.cfg.q8) 10 else 6;
    }

    pub fn run(self: *NaxAttn) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.nargs()]);
    }

    pub fn count(self: *const NaxAttn) usize {
        const h = self.shape.rows * self.cfg.gqa;
        if (last(self.cfg.phase)) return h * self.shape.n * self.vWidth();
        return h * ((self.shape.n + self.rowsPerTile() - 1) / self.rowsPerTile()) * self.rowsPerTile() * 130;
    }

    fn emit(self: *const NaxAttn, want: []f64, mag: ?[]f64, sim: Sim) !void {
        const a = self.rig.a;
        const c = self.cfg;
        const h = self.shape.rows * c.gqa;
        const ql = self.shape.n;
        const kl = self.shape.k;
        const bq = self.rowsPerTile();
        const nq = (ql + bq - 1) / bq;
        const nk = (kl + NAX_BK - 1) / NAX_BK;
        const dv = self.vWidth();
        const r = range(c.phase, nk);
        const keys = try a.alloc(Key, (nk + 1) * NAX_BK);
        defer a.free(keys);
        const ends = try a.alloc(usize, nk + 1);
        defer a.free(ends);
        const zq = try a.alloc(f64, c.dk);
        defer a.free(zq);
        @memset(zq, 0);
        const zv = try a.alloc(f64, dv);
        defer a.free(zv);
        @memset(zv, 0);
        const terms = try a.alloc(f64, c.dk);
        defer a.free(terms);
        const o = try a.alloc(f64, dv);
        defer a.free(o);
        const eo = try a.alloc(f64, dv);
        defer a.free(eo);
        const scale2 = self.scale * LOG2E;
        const q_off = kl - ql;
        const rows = if (last(c.phase)) ql else nq * bq;
        for (0..h) |hh| for (0..rows) |row| {
            const hk = hh / c.gqa;
            const tile = row / bq;
            const pos = row + q_off;
            const kb_lim = @min(nk, ((tile + 1) * bq + q_off + NAX_BK - 1) / NAX_BK);
            const kb_win = if (c.window > 0) (tile * bq + q_off + 1 -| c.window) / NAX_BK else 0;
            var n: usize = 0;
            var nb: usize = 0;
            var kb = @max(r[0], kb_win);
            while (kb < @min(kb_lim, r[1])) : (kb += 1) {
                for (0..NAX_BK) |j| {
                    const col = kb * NAX_BK + j;
                    const v = if (col < kl) self.v[(hk * kl + col) * dv ..][0..dv] else zv;
                    const hide = col >= kl or pos < col or (c.window > 0 and col + c.window <= pos);
                    keys[n] = if (hide) .{ .s = FINITE_MIN, .on = false, .v = v } else blk: {
                        if (c.q8) {
                            // int32 products of the int8 rows (exact), times the tile's and block's scales and scale x log2(e).
                            const qrow = if (row < ql) self.qi[(hh * ql + row) * 256 ..][0..256] else zq;
                            const d = dot(qrow, self.ki[(hk * kl + col) * 256 ..][0..256], self.qs[hh * nq + tile] * self.ks[hk * nk + kb] * scale2, 256, 6, sim, terms);
                            break :blk .{ .s = d.v, .e = d.e, .v = v };
                        }
                        const qrow = if (row < ql) self.q[(hh * ql + row) * c.dk ..][0..c.dk] else zq;
                        const d = dot(qrow, self.k[(hk * kl + col) * c.dk ..][0..c.dk], scale2, c.dk + 1, 3, sim, terms);
                        break :blk .{ .s = d.v, .e = d.e, .v = v };
                    };
                    n += 1;
                }
                ends[nb] = n;
                nb += 1;
            }
            const srow = hh * nq * bq + row;
            const start: Init = if (c.phase == .through or c.phase == .in)
                .{ .m = self.state[srow * 130 + 128], .l = self.state[srow * 130 + 129], .o = self.state[srow * 130 ..][0..128] }
            else if (c.sinks)
                .{ .m = self.sinks[hh] * LOG2E, .e = 2 * host.U * @abs(self.sinks[hh] * LOG2E), .l = 1 }
            else
                .{ .m = FINITE_MIN };
            // P is rounded to T for the tensor op's P x V.
            const st = online(.{ .base2 = true, .p_unit = 0.5 * host.ulpOf(c.dt), .p_abs = if (c.dt == .float16) 0x1p-25 else 0 }, start, keys[0..n], ends[0..nb], o, eo);
            if (last(c.phase)) {
                normalize(st, o, eo);
                const at = (row * h + hh) * dv;
                @memcpy(want[at..][0..dv], o);
                if (mag) |m| for (m[at..][0..dv], eo) |*x, e| {
                    x.* = host.allowance(e);
                };
                continue;
            }
            @memcpy(want[srow * 130 ..][0..128], o);
            want[srow * 130 + 128] = st.m;
            want[srow * 130 + 129] = st.l;
            if (mag) |m| {
                for (m[srow * 130 ..][0..128], eo) |*x, e| x.* = host.allowance(e);
                m[srow * 130 + 128] = host.allowance(st.em);
                m[srow * 130 + 129] = host.allowance(st.el);
            }
        };
    }

    pub fn reference(self: *const NaxAttn, want: []f64, mag: []f64) !void {
        try self.emit(want, mag, null);
    }

    pub fn simulate(self: *const NaxAttn, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.emit(out, null, .{ .acc = acc, .order = order });
    }

    /// The declared launch as the wrapper runs it: one dispatch, or the three passes with the state read back and
    /// bound between them; the output viewed [B, H, qL, 128] as the wrapper returns it.
    fn declared(self: *NaxAttn, a: std.mem.Allocator) ![]u8 {
        const rig = &self.rig;
        var args = self.args;
        if (self.cfg.phase == .single or self.cfg.q8) {
            try rig.launch(self.launch, self.geometryAt(self.shape, .single).?, args[0..self.nargs()]);
        } else for ([_]Phase{ .out, .through, .in }) |phase| {
            const pv = self.params(phase);
            args[5] = try rig.input(.int32, &.{16}, &pv);
            try rig.launch(self.launch, self.geometryAt(self.shape, phase).?, args[0..6]);
            if (phase == .in) break;
            const st = try host.read(a, rig.output(1), rig.stream());
            defer a.free(st);
            args[4] = try rig.input(.float32, &.{cInt(st.len)}, st);
        }
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, rig.output(0), &[_]c_int{ 0, 2, 1, 3 }, 4, rig.stream()));
        return a.dupe(u8, try rig.bytesOf(&.{t}));
    }

    fn wrapper(self: *NaxAttn) ![]const u8 {
        const c = self.cfg;
        const s = self.rig.stream();
        const x = self.args;
        const nk: c_int = cInt((self.shape.k + NAX_BK - 1) / NAX_BK);
        const got = if (c.q8)
            (try nax.attention256Int8Impl(s, x[0], x[1], x[2], @floatCast(self.scale))) orelse return error.Declined
        else if (c.phase == .single)
            try nax.dispatch(s, x[0], x[1], x[2], @floatCast(self.scale), cInt(c.window), if (c.sinks) x[3] else null, &.{ 0, nk }, 1)
        else
            try nax.dispatch(s, x[0], x[1], x[2], @floatCast(self.scale), cInt(c.window), if (c.sinks) x[3] else null, &.{ 0, @divTrunc(nk, 3), @divTrunc(2 * nk, 3), nk }, 2);
        defer _ = mlx.mlx_array_free(got);
        return self.rig.bytesOf(&.{got});
    }

    /// The int8 rows and scales the wrapper derives from q and k (its own quantizer), bound in place of the drawn ones.
    fn quantizeLikeWrapper(self: *NaxAttn) !void {
        const s = self.rig.stream();
        var mean = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mean);
        try mlx.check(mlx.mlx_mean_axes(&mean, self.args[1], &[_]c_int{2}, 1, true, s));
        const qq = try int8_attn.quantRows(self.args[0], 64, null, 1e-6, false, false, s);
        const kq = try int8_attn.quantRows(self.args[1], NAX_BK, mean, 1e-6, false, false, s);
        for ([_]mlx.mlx_array{ qq[0], kq[0], qq[1], kq[1] }, 6..) |arr, i| {
            const vals = try host.read(self.rig.a, arr, s);
            defer self.rig.a.free(vals);
            _ = mlx.mlx_array_free(arr);
            self.args[i] = try self.rig.input(if (i < 8) .int8 else .float32, if (i < 8) &.{@intCast(vals.len)} else &.{@intCast(vals.len)}, vals);
        }
    }
};

/// The int8 entry's two scale inputs share one template parameter, but MLX binds an input under 8 elements in the
/// constant address space and a larger one in device: when one of the Q tile scales (B x H x tiles) and the K block
/// scales (B x Hkv x blocks) is under 8 and the other is not, the instantiation does not compile.
const q8_scales_known = spec.Known{ .gate = .shapes, .finding = "msv_nax_attn_256_q8 does not compile when exactly one of its scale inputs has under 8 elements (qs = H x q tiles, ks = Hkv x key blocks): MLX binds that one `constant`, the other `device`, and `attention_nax_bdv_qt` takes both as one ScalePtr. 8 q / 2 kv heads (Qwen3.5-2B) at a 64-token first prefill: qs 8, ks 4, MLX error; also 4 q / 1 kv heads (Gemma 3 1B) at 64..127 rows past 224 cached keys. Opt-in (--int8-qk), M5 only" };

fn naxSpec(comptime name: []const u8, cfg: NaxCfg, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .nax = true,
        .accum = .f32,
        .depth = host.depthOne,
        .out = if (NaxAttn.last(cfg.phase)) host.elemOf(cfg.dt) else .f32,
        .tiles = .{ .{}, .{ .size = if (cfg.q8 or cfg.phase == .single) 64 else 32, .min = 9 }, .{ .size = NAX_BK, .min = 9 } },
        .served = served,
        .sums = .{ .exact = false },
        .known = if (cfg.q8) q8_scales_known else null,
    };
}

const nax_one = naxLaunch(false, 1);
const nax_two = naxLaunch(false, 2);
const nax_q8 = naxLaunch(true, 2);

/// MiMo-V2 (q/k 192, v 128, sinks; sliding layers banded to 128) and Flint (128 / 128, sinks; sliding 513) on the
/// M5: one dispatch, or key-range passes once the grid is wide enough (the three pass kinds); the int8 Q/K hd-256
/// causal entry (opt-in, `--int8-qk`).
const nax_registrations = .{
    .{ "msv_nax_attn_dk_128 bf16 (q/k 192, v 128, sinks, causal)", NaxCfg{ .sinks = true }, qk(&.{ .{ 1, 16, 40 }, .{ 1, 100, 300 } }), &nax_one },
    .{ "msv_nax_attn_dk_128 bf16 (q/k 192, v 128, sinks, band 128)", NaxCfg{ .sinks = true, .window = 128 }, qk(&.{ .{ 1, 16, 200 }, .{ 1, 90, 300 } }), &nax_one },
    .{ "msv_nax_attn_dk_128 bf16 (q/k 128, v 128, sinks, causal)", NaxCfg{ .dk = 128, .sinks = true }, qk(&.{ .{ 1, 16, 40 }, .{ 1, 100, 300 } }), &nax_one },
    .{ "msv_nax_attn_dk_128 bf16 (q/k 128, v 128, sinks, band 513)", NaxCfg{ .dk = 128, .sinks = true, .window = 513 }, qk(&.{ .{ 1, 16, 560 }, .{ 1, 70, 600 } }), &nax_one },
    .{ "msv_nax_attn_dk_128 bf16 (q/k 192, v 128, sinks, first key-range pass)", NaxCfg{ .sinks = true, .phase = .out }, qk(&.{ .{ 1, 16, 96 }, .{ 1, 64, 300 } }), &nax_two },
    .{ "msv_nax_attn_dk_128 bf16 (q/k 192, v 128, sinks, middle key-range pass)", NaxCfg{ .sinks = true, .phase = .through }, qk(&.{ .{ 1, 16, 96 }, .{ 1, 64, 300 } }), &nax_two },
    .{ "msv_nax_attn_dk_128 bf16 (q/k 192, v 128, sinks, last key-range pass)", NaxCfg{ .sinks = true, .phase = .in }, qk(&.{ .{ 1, 16, 96 }, .{ 1, 64, 300 } }), &nax_two },
    .{ "msv_nax_attn_256_q8 bf16 (causal, int8 Q/K)", NaxCfg{ .dk = 256, .q8 = true, .gqa = 4 }, qk(&.{ .{ 2, 128, 300 }, .{ 2, 64, 64 } }), &nax_q8 },
};

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    inline for (gate_registrations) |r| try set.add(Gate, .init(a, r[3], r[1]), gateSpec(r[0], r[1], r[2]), null);
    inline for (a256_registrations) |r| try set.add(A256, .init(a, r[3], r[1]), a256Spec(r[0], r[1], r[2]), null);
    inline for (row_registrations) |r| try set.add(RowPart, .init(a, r[1], r[2], r[3]), winSpec(r[0], .f32, r[4], ROW_CK), null);
    try set.add(RowMerge, .init(a, &row_merge_launch), winSpec("msv_row_attn_merge", .bf16, qk(&.{ .{ 6, 1, 3 }, .{ 24, 8, 6 }, .{ 6, 16, 5 } }), 0), null);
    inline for (qkv_registrations) |r| try set.add(Qkv, .init(a, r[3], r[1]), qkvSpec(r[0], r[1], r[2]), null);
    inline for (nax_registrations) |r| try set.add(NaxAttn, .init(a, r[3], r[1]), naxSpec(r[0], r[1], r[2]), null);
    inline for (lane_registrations) |r| {
        var sp = winSpec(r[0], if (r[2] == .merge) .bf16 else .f32, r[5], if (r[2] == .merge) 0 else LANE_CK);
        sp.nax = r[2] != .merge;
        try set.add(Lane, .init(a, r[1], r[2], r[3], r[4]), sp, null);
    }
}

// ── tests ──

test "kernel contract attention: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    fused_activations.swiglu_fused_override = true;
    defer fused_activations.swiglu_fused_override = null;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    var at: usize = 0;
    inline for (.{ .{ Gate, gate_registrations.len }, .{ A256, a256_registrations.len }, .{ RowPart, row_registrations.len }, .{ RowMerge, 1 }, .{ Qkv, qkv_registrations.len }, .{ NaxAttn, nax_registrations.len }, .{ Lane, lane_registrations.len } }) |fam| {
        for (set.kernels.items[at..][0..fam[1]]) |k| {
            const t: *fam[0] = @ptrCast(@alignCast(k.ctx));
            // The NAX wrappers decline on a GPU without NAX.
            if (fam[0] == NaxAttn and !device.naxAvailable()) continue;
            for (k.spec.served, 0..) |shape, si| {
                // The int8 entry's second served shape is its known compile failure (both launches fail there).
                if (fam[0] == NaxAttn and t.cfg.q8 and si > 0) continue;
                try std.testing.expect(try t.prepare(shape, .random, 7));
                if (fam[0] == NaxAttn and t.cfg.q8) try t.quantizeLikeWrapper();
                // A wrapper that declines here (a lane this GPU lacks) is never launched by the engine on it.
                if (fam[0] == Qkv) _ = t.wrapper() catch |e| if (e == error.Declined) continue else return e;
                const ours, const theirs = if (fam[0] == RowMerge) continue else if (fam[0] == NaxAttn) switch (t.cfg.phase) {
                    // The passes ride the last pass's registration.
                    .out, .through => continue,
                    .single, .in => .{ try t.declared(a), try t.wrapper() },
                } else if (fam[0] == Qkv)
                    .{ try a.dupe(u8, try t.merged()), try t.wrapper() }
                else if (fam[0] == RowPart)
                    .{ try t.chained(a), try t.wrapper() }
                else if (fam[0] == Lane) switch (t.kind) {
                    // The three passes against the wrapper; the partial and merge registrations ride the tails'.
                    .partial, .merge => continue,
                    .tail => .{ try t.chained(a, if (t.win.g == 6) &lane_part6 else &lane_part4, &lane_merge_launch), try t.wrapper() },
                } else if (fam[0] == A256) switch (t.cfg.phase) {
                    // The chain's three dispatch kinds against the wrapper at the chunk the reference splits at.
                    .out, .through => continue,
                    .in => .{ try t.chained(a), try t.wrapper(@intCast(shape.rows * t.cfg.gqa * shape.n * A256.chunk(shape.k))) },
                    .single => blk: {
                        try t.run();
                        break :blk .{ try a.dupe(u8, t.rig.outputBytes()), try t.wrapper(0) };
                    },
                } else blk: {
                    try t.run();
                    break :blk .{ try a.dupe(u8, t.rig.outputBytes()), try t.wrapper() };
                };
                defer a.free(ours);
                if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{k.spec.name});
                try std.testing.expectEqualSlices(u8, theirs, ours);
            }
        }
        at += fam[1];
    }
}
