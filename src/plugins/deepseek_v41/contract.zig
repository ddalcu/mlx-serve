//! DeepSeek-V4.1's kernel contract: the router (gate GEMV, sqrt-softplus top-k), the indexer (scores, tiled scores,
//! radix top-k), the hyper-connections (mix, split, norm, stage), the decode attention (one threadgroup, split and
//! merge, the wide softmax), the rope / QAT tails, the dense f32 GEMV, Engram's gate, and the MoE kernels (decode
//! gate+up and down, prefill act and combine). Listed in `kernel_contract/registry.zig`; findings in AGENTS.md.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kit = @import("../deepseek_v4/contract_kit.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const v41 = @import("deepseek_v41.zig");
const qtail = @import("prompt/q_tail.zig");
const hcf = @import("prompt/hc_fuse.zig");
const sx1 = @import("prompt/scratch_x1.zig");
const gtwin = @import("decode/gemv_twin.zig");
const ggemm = @import("prompt/gate_gemm.zig");
const markov = @import("spec/markov.zig");
const mplan = @import("spec/markov_plan.zig");
const fa = @import("prompt/flash_attn.zig");
const satt = @import("../../sdk/blocks/attention/contract.zig");
const gguf = @import("../../sdk/quant/gguf/kernels.zig");
const gguf_contract = @import("../../sdk/quant/gguf/contract.zig");
const gguf_quants = @import("../../sdk/quant/gguf/quants.zig");

const spec = kit.spec;
const metal = kit.metal;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;
const nan = kit.nan;
const inf = kit.inf;
const rt = kc.host.rt;

/// fp32's unit roundoff.
pub const u = 0x1p-24;

fn ci(v: usize) c_int {
    return @intCast(v);
}

/// The release: 5120 hidden, 4 hc streams, 384 experts (6 a token, 2304 wide), 64 heads of 512, the indexer's 32
/// heads of 128 picking 512, a 128-row window, a 64-wide rope.
const DIM = 5120;
pub const HC = 4;
pub const MIX = (2 + HC) * HC;
pub const ITERS = 20;
const EXPERTS = 384;
const TOPK = 6;
const INTER = 2304;
const HEADS = 64;
const HD = 512;
const IH = 32;
pub const IHD = 128;
const WINDOW = 128;
pub const RD = 64;
pub const EPS_RMS = 1e-20;
pub const EPS_HC = 1e-6;
/// The decode widths: one token, a DSpark block, the widest region (`region_rows`).
const REGION = 16;

pub const act: mlx.mlx_dtype = .bfloat16;

pub fn bf(v: f64) f64 {
    return spec.roundTo(.bf16, v);
}

/// MLX-style `maximum`: NaN propagates (Metal's `max` drops it).
const maxN = kit.maxN;

// ── interval arithmetic for outputs that pass through intermediate roundings ──

/// A value's float64 range: every f32 path through the kernel's operations lands in it.
pub const Iv = struct {
    lo: f64,
    hi: f64,

    pub fn of(v: f64, e: f64) Iv {
        return .{ .lo = v - e, .hi = v + e };
    }
    pub fn pt(v: f64) Iv {
        return .{ .lo = v, .hi = v };
    }
    /// Rounded to `dt` at both ends (round-to-nearest is monotone).
    pub fn round(a: Iv, dt: mlx.mlx_dtype) Iv {
        return .{ .lo = rt(dt, a.lo), .hi = rt(dt, a.hi) };
    }
    pub fn mul(a: Iv, b: Iv) Iv {
        const c = [_]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(c[0], c[1]), @min(c[2], c[3])), .hi = @max(@max(c[0], c[1]), @max(c[2], c[3])) };
    }
    pub fn add(a: Iv, b: Iv) Iv {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    fn neg(a: Iv) Iv {
        return .{ .lo = -a.hi, .hi = -a.lo };
    }
    /// Widened by `rel` of its magnitude plus `abs` (an f32 operation's own rounding).
    pub fn widen(a: Iv, rel: f64, abs: f64) Iv {
        const m = @max(@abs(a.lo), @abs(a.hi));
        return .{ .lo = a.lo - rel * m - abs, .hi = a.hi + rel * m + abs };
    }
    pub fn has(a: Iv, x: f64) bool {
        return x >= a.lo and x <= a.hi;
    }
};

/// Compares classes of `got` against float64's `want` (gate 1 under a stated rule).
pub fn classes(buf: []u8, got: []const f64, want: []const f64) ?[]const u8 {
    for (got, want, 0..) |g, w, i| if (spec.classOf(g) != spec.classOf(w))
        return std.fmt.bufPrint(buf, "output {d} is {e} where float64 gives {e}", .{ i, g, w }) catch "class";
    return null;
}

/// Output `i` of the last run, read as `e` from `off` bytes into the output bytes.
fn outAt(k: anytype, e: spec.Elem, off: usize, i: usize) f64 {
    return spec.read(e, k.rig.outputBytes()[off..], i);
}

/// Mirrored-integer stress rows (gate 4): the first half's terms positive, the second half the same terms negated on a
/// shuffle of its positions, x's halves one apart on 8 entries; entry 8 is the pair (xmax, wmax). Partial sums grow
/// past fp16's range and bf16's resolution; the total is a small integer; fp32 holds every partial sum exactly in
/// any order while k/2 * xmax * wmax stays below 2^24.
pub fn stressDot(rng: std.Random, x: []f64, w: []f64, rows: usize, k: usize, xmax: i32, wmax: i32, scratch: std.mem.Allocator) !void {
    const h = k / 2;
    const perm = try scratch.alloc(usize, h);
    defer scratch.free(perm);
    for (perm, 0..) |*q, i| q.* = i;
    rng.shuffle(usize, perm);
    @memset(x, 0);
    for (0..h) |i| {
        x[i] = if (i == 8) @floatFromInt(xmax) else @floatFromInt(rng.intRangeAtMost(i32, 1, xmax));
        x[h + perm[i]] = x[i] + @as(f64, if (i < 8) 1 else 0);
    }
    @memset(w, 0);
    for (0..rows) |r| for (0..h) |i| {
        const c: f64 = if (i == 8) @floatFromInt(wmax) else @floatFromInt(rng.intRangeAtMost(i32, 1, if (i < 8) @min(wmax, 3) else wmax));
        w[r * k + i] = c;
        w[r * k + h + perm[i]] = -c;
    };
}

fn depthK(s: Shape) usize {
    return s.k;
}
fn depthN(s: Shape) usize {
    return s.n;
}
fn depthK1(s: Shape) usize {
    return s.k + 1;
}
pub fn depthOne(_: Shape) usize {
    return 1;
}
fn rowsN(s: Shape) usize {
    return s.rows * s.n;
}
fn rowsMix(s: Shape) usize {
    return s.rows * MIX;
}

// ── router logits: one simdgroup an expert, 8 inputs a lane per step ──

const Gate = struct {
    pub const Params = struct {};
    const SGS = 8;

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.rows > REGION or s.n == 0 or s.k == 0 or s.k % 8 != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci((s.n + SGS - 1) / SGS * 32), ci(SGS * s.rows), 1 }, .tg = .{ 32, SGS, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n) }, .float32);
        g.int("SGS", SGS);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const x = try k.add(act, &.{ L, k.shape.k });
        const w = try k.add(act, &.{ k.shape.n, k.shape.k });
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, w, -0.05, 0.05);
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(x, L, k.shape.k),
            .stress => {
                const D = k.shape.k;
                try stressDot(rng, x[0..D], w, k.shape.n, D, 64, 7, k.scratch());
                for (1..L) |l| @memcpy(x[l * D ..][0..D], x[0..D]);
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const D = k.shape.k;
        const E = k.shape.n;
        const t = try k.scratch().alloc(f64, D);
        defer k.scratch().free(t);
        for (0..k.shape.rows) |l| for (0..E) |e| {
            for (t, 0..) |*v, i| v.* = k.in(0)[l * D + i] * k.in(1)[e * D + i];
            out[l * E + e] = sum.of(t);
            for (t) |v| mag[l * E + e] += @abs(v);
        };
    }
};

// ── dense f32 GEMV: x [L, K] f32 against rows w [N, K] read as stored, RPS rows a simdgroup ──

const DenseF32x = struct {
    pub const Params = struct {};
    const RPS = 4;

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.rows > REGION or s.k == 0 or s.k % 128 != 0 or s.n == 0 or s.n % RPS != 0) return null;
        var g = metal.Geometry{ .grid = .{ 32, ci(s.n / RPS), ci(s.rows) }, .tg = .{ 32, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n) }, .float32);
        g.int("RPS", RPS);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const K = k.shape.k;
        const x = try k.add(.float32, &.{ L, K });
        const w = try k.add(act, &.{ k.shape.n, K });
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, w, -0.05, 0.05);
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(x, L, K),
            .stress => {
                try stressDot(rng, x[0..K], w, k.shape.n, K, 64, 7, k.scratch());
                for (1..L) |l| @memcpy(x[l * K ..][0..K], x[0..K]);
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const K = k.shape.k;
        const N = k.shape.n;
        const t = try k.scratch().alloc(f64, K);
        defer k.scratch().free(t);
        for (0..k.shape.rows) |l| for (0..N) |n| {
            for (t, 0..) |*v, i| v.* = k.in(0)[l * K + i] * k.in(1)[n * K + i];
            out[l * N + n] = sum.of(t);
            for (t) |v| mag[l * N + n] += @abs(v);
        };
    }
};

// ── the router's pick: sqrt-softplus scores, the biased top K (lowest index on ties), weights normalized and scaled ──

fn softplusSqrt(z: f64) f64 {
    if (std.math.isNan(z)) return nan;
    const y = @exp(-@abs(z));
    return @sqrt(@max(z, 0) + std.math.log1p(y));
}

const Route = struct {
    pub const Params = struct { k: usize = TOPK, norm: bool = true, scal: f64 = 1.5 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        const E = s.k;
        if (s.rows == 0 or E == 0 or E > 1024 or p.k > E) return null;
        const tg = (E + 31) / 32 * 32;
        var g = metal.Geometry{ .grid = .{ ci(tg), ci(s.rows), 1 }, .tg = .{ ci(tg), 1, 1 } };
        g.out(&.{ ci(s.rows), ci(p.k) }, .uint32);
        g.out(&.{ ci(s.rows), ci(p.k) }, .float32);
        g.int("K", p.k);
        g.flag("NORM", p.norm);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const E = k.shape.k;
        const z = try k.add(.float32, &.{ L, E });
        const bias = try k.add(.float32, &.{ 1, E });
        const scal = try k.add(.float32, &.{1});
        kit.uniform(rng, z, -6, 6);
        kit.uniform(rng, bias, -0.5, 0.5);
        scal[0] = k.p.scal;
        if (f == .nonfinite) kit.plantRows(z, L, E);
    }

    pub fn count(p: Params, s: Shape) usize {
        return 2 * s.rows * p.k;
    }

    /// Row `l`'s picks by the stated rule (a NaN score is never picked), and the float64 weights.
    fn picks(k: anytype, l: usize, idx: []usize, wts: []f64) void {
        const E = k.shape.k;
        const z = k.in(0)[l * E ..][0..E];
        const bias = k.in(1);
        var taken: [1024]bool = @splat(false);
        for (idx) |*pk| {
            var best: ?usize = null;
            for (0..E) |e| {
                const v = softplusSqrt(z[e]) + bias[e];
                if (taken[e] or std.math.isNan(v)) continue;
                if (best == null or v > softplusSqrt(z[best.?]) + bias[best.?]) best = e;
            }
            pk.* = best orelse 0;
            taken[pk.*] = true;
        }
        var total: f64 = 0;
        for (idx) |e| total += softplusSqrt(z[e]);
        const div = if (k.p.norm) total + 1e-20 else 1;
        for (idx, wts) |e, *w| w.* = softplusSqrt(z[e]) / div * spec.roundTo(.f32, k.p.scal);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, _: Sum) !void {
        const K = k.p.k;
        const L = k.shape.rows;
        var idx: [64]usize = undefined;
        for (0..L) |l| {
            picks(k, l, idx[0..K], out[L * K + l * K ..][0..K]);
            for (idx[0..K], 0..) |e, r| out[l * K + r] = @floatFromInt(e);
        }
        @memset(mag, 0);
    }

    /// The stated rule: the picks are the biased scores in descending order (a near-tie within the scores' own error
    /// either way, a NaN never), each weight the picked score over the picks' sum.
    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const K = k.p.k;
        const L = k.shape.rows;
        const E = k.shape.k;
        const z = k.in(0);
        const bias = k.in(1);
        const wo = 4 * L * K;
        var idx: [64]usize = undefined;
        var want: [64]f64 = undefined;
        for (0..L) |l| {
            picks(k, l, idx[0..K], want[0..K]);
            var taken: [1024]bool = @splat(false);
            var sumv: f64 = 0;
            for (0..K) |r| {
                const got: usize = @intCast(std.mem.readInt(u32, k.rig.outputBytes()[4 * (l * K + r) ..][0..4], .little));
                if (got >= E or taken[got]) return std.fmt.bufPrint(&k.msg, "row {d} pick {d} is {d}: out of range or repeated", .{ l, r, got }) catch "pick";
                const vg = softplusSqrt(z[l * E + got]) + bias[got];
                var best = -inf;
                for (0..E) |e| if (!taken[e]) {
                    const v = softplusSqrt(z[l * E + e]) + bias[e];
                    if (!std.math.isNan(v)) best = @max(best, v);
                };
                const tol = if (at == .nonfinite) 0 else 16 * u * (@abs(best) + 1);
                if (!(vg >= best - tol) and !(std.math.isInf(best) and vg == best))
                    return std.fmt.bufPrint(&k.msg, "row {d} pick {d} is expert {d} at {e}; {e} was left", .{ l, r, got, vg, best }) catch "pick";
                taken[got] = true;
                sumv += softplusSqrt(z[l * E + got]);
            }
            const div = if (k.p.norm) sumv + 1e-20 else 1;
            for (0..K) |r| {
                const e: usize = @intCast(std.mem.readInt(u32, k.rig.outputBytes()[4 * (l * K + r) ..][0..4], .little));
                const w = softplusSqrt(z[l * E + e]) / div * spec.roundTo(.f32, k.p.scal);
                const g = outAt(k, .f32, wo, l * K + r);
                if (spec.classOf(g) != spec.classOf(w)) return std.fmt.bufPrint(&k.msg, "row {d} weight {d} is {e} where float64 gives {e}", .{ l, r, g, w }) catch "weight";
                if (std.math.isFinite(w) and !(@abs(g - w) <= 32 * u * @abs(w) + 0x1p-140))
                    return std.fmt.bufPrint(&k.msg, "row {d} weight {d} is {e} where float64 gives {e}", .{ l, r, g, w }) catch "weight";
            }
        }
        return null;
    }
};

// ── radix top-k: the K largest of each row by the float's order key, as ascending indices, ties to the lower index ──

fn orderKey(v: f64) u32 {
    const b: u32 = @bitCast(@as(f32, @floatCast(v)));
    return if (b & 0x80000000 != 0) ~b else b | 0x80000000;
}

const TopK = struct {
    pub const Params = struct {};
    const TG = 1024;

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0 or s.n > s.k) return null;
        var g = metal.Geometry{ .grid = .{ TG, ci(s.rows), 1 }, .tg = .{ TG, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n) }, .int32);
        g.int("TG", TG);
        g.int("K", s.n);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const n = k.shape.k;
        const x = try k.add(.float32, &.{ L, n });
        // Few distinct values (ties at every rank) and masked entries, as the indexer's scores carry.
        for (x) |*v| v.* = if (rng.uintLessThan(u32, 8) == 0) -inf else @as(f64, @floatFromInt(rng.intRangeAtMost(i32, -40, 40))) * 0.25;
        if (f == .nonfinite) kit.plantRows(x, L, n);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    fn want(k: anytype, l: usize, out: []u32) !void {
        const n = k.shape.k;
        const K = k.shape.n;
        const x = k.in(0)[l * n ..][0..n];
        const keys = try k.scratch().alloc(u32, n);
        defer k.scratch().free(keys);
        for (keys, x) |*q, v| q.* = orderKey(v);
        // The K-th largest key, then every larger key and the first ties up to K, in index order.
        const sorted = try k.scratch().dupe(u32, keys);
        defer k.scratch().free(sorted);
        std.mem.sort(u32, sorted, {}, std.sort.desc(u32));
        const kth = sorted[K - 1];
        var gt: usize = 0;
        for (keys) |q| gt += @intFromBool(q > kth);
        var ties = K - gt;
        var o: usize = 0;
        for (keys, 0..) |q, i| {
            if (q > kth or (q == kth and ties > 0)) {
                if (q == kth) ties -= 1;
                out[o] = @intCast(i);
                o += 1;
            }
        }
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, _: Sum) !void {
        const K = k.shape.n;
        const w = try k.scratch().alloc(u32, K);
        defer k.scratch().free(w);
        for (0..k.shape.rows) |l| {
            try want(k, l, w);
            for (w, 0..) |v, r| out[l * K + r] = @floatFromInt(v);
        }
        @memset(mag, 0);
    }

    /// Exact: the same indices as the order-key selection, at every gate.
    pub fn check(k: anytype, _: spec.Gate) !?[]const u8 {
        const K = k.shape.n;
        const w = try k.scratch().alloc(u32, K);
        defer k.scratch().free(w);
        for (0..k.shape.rows) |l| {
            try want(k, l, w);
            for (w, 0..) |v, r| {
                const g = std.mem.readInt(u32, k.rig.outputBytes()[4 * (l * K + r) ..][0..4], .little);
                if (g != v) return std.fmt.bufPrint(&k.msg, "row {d} pick {d} is index {d} where the order keys give {d}", .{ l, r, g, v }) catch "pick";
            }
        }
        return null;
    }
};

// ── MoE prefill act and combine (one element a thread) ──

/// MLX's clipped SwiGLU in f32 as the kernel orders it, times the routing weight (float64; NaN propagates).
fn clippedSwiglu(g0: f64, up0: f64, lim: f64) [2]f64 {
    var g = g0;
    var up = up0;
    if (lim > 0) {
        g = kit.minN(g, lim);
        up = if (std.math.isNan(up)) nan else std.math.clamp(up, -lim, lim);
    }
    const y = 1 / (1 + @exp(@abs(g)));
    return .{ g * (if (g < 0) y else 1 - y), up };
}

const MoeAct = struct {
    pub const Params = struct { lim: f64 = 10 };

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        const total = s.rows * s.n;
        if (total == 0) return null;
        var g = metal.Geometry{ .grid = .{ ci((total + 255) / 256 * 256), 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ ci(s.rows), 1, ci(s.n) }, act);
        g.int("N", s.n);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const n = k.shape.rows;
        const N = k.shape.n;
        const y = try k.add(act, &.{ n, 2 * N });
        const wts = try k.add(.float32, &.{n});
        const lim = try k.add(.float32, &.{1});
        kit.uniform(rng, y, -14, 14);
        kit.uniform(rng, wts, 0, 0.5);
        lim[0] = k.p.lim;
        if (f == .nonfinite) kit.plantRows(y, n, 2 * N);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, _: Sum) !void {
        const N = k.shape.n;
        for (0..k.shape.rows) |r| for (0..N) |c| {
            const y = k.in(0)[r * 2 * N ..];
            const a = clippedSwiglu(y[c], y[N + c], k.in(2)[0]);
            out[r * N + c] = a[0] * a[1] * k.in(1)[r];
        };
        @memset(mag, 0);
    }
};

/// A fused gate/up's activation `a` [rows, 1, n] times its routing weight, one rounding.
const MoeWt = struct {
    pub const Params = struct {};

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        const total = s.rows * s.n;
        if (total == 0) return null;
        var g = metal.Geometry{ .grid = .{ ci((total + 255) / 256 * 256), 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ ci(s.rows), 1, ci(s.n) }, act);
        g.int("N", s.n);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const a = try k.add(act, &.{ k.shape.rows, 1, k.shape.n });
        const wts = try k.add(.float32, &.{k.shape.rows});
        kit.uniform(rng, a, -8, 8);
        kit.uniform(rng, wts, 0, 2.5);
        if (f == .nonfinite) kit.plantRows(a, k.shape.rows, k.shape.n);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, _: Sum) !void {
        for (out, 0..) |*o, i| o.* = k.in(0)[i] * k.in(1)[i / k.shape.n];
        @memset(mag, 0);
    }
};

const MoeCombine = struct {
    pub const Params = struct { k: usize = TOPK };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        const total = s.rows * s.n;
        if (total == 0 or s.k != p.k) return null;
        var g = metal.Geometry{ .grid = .{ ci((total + 255) / 256 * 256), 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n) }, act);
        g.int("D", s.n);
        g.int("K", p.k);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const D = k.shape.n;
        const K = k.p.k;
        const dn = try k.add(act, &.{ L * K, D });
        const inv = try k.add(.uint32, &.{L * K});
        const sh = try k.add(act, &.{ L, D });
        kit.uniform(rng, dn, -2, 2);
        kit.uniform(rng, sh, -2, 2);
        // A permutation of the sorted rows, as argsort of the expert order gives.
        for (inv, 0..) |*v, i| v.* = @floatFromInt(i);
        const perm = try k.scratch().alloc(f64, inv.len);
        defer k.scratch().free(perm);
        @memcpy(perm, inv);
        rng.shuffle(f64, perm);
        @memcpy(inv, perm);
        switch (f) {
            .random => {},
            .nonfinite => {
                // Row l's slot terms: NaN, +Inf, -Inf, both infinities in rows 0-3.
                const put = [_][3]f64{ .{ 0, 0, nan }, .{ 1, 1, inf }, .{ 2, 0, -inf }, .{ 3, 0, inf } };
                for (put) |q| {
                    const l: usize = @intFromFloat(q[0]);
                    if (l >= L) continue;
                    const row: usize = @intFromFloat(inv[l * K + @as(usize, @intFromFloat(q[1]))]);
                    dn[row * D + l % D] = q[2];
                }
                if (L > 3) {
                    const row: usize = @intFromFloat(inv[3 * K + 1]);
                    dn[row * D + 3 % D] = -inf;
                }
            },
            .stress => {
                // Slot terms B, r1, -B, r2, B, -B (B = 2^16) and a small shared term: exact in fp32 in any order;
                // fp16 overflows and bf16 drops the small terms riding on B.
                const B = 0x1p16;
                for (0..L) |l| for (0..D) |c| {
                    const seq = [_]f64{ B, @floatFromInt(rng.intRangeAtMost(i32, 1, 8)), -B, @floatFromInt(rng.intRangeAtMost(i32, 1, 8)), B, -B };
                    for (0..K) |j| {
                        const row: usize = @intFromFloat(inv[l * K + j]);
                        dn[row * D + c] = if (j < seq.len) seq[j] else 0;
                    }
                    sh[l * D + c] = @floatFromInt(rng.intRangeAtMost(i32, -8, 8));
                };
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const D = k.shape.n;
        const K = k.p.k;
        var t: [17]f64 = undefined;
        for (0..k.shape.rows) |l| for (0..D) |c| {
            for (0..K) |j| {
                const row: usize = @intFromFloat(k.in(1)[l * K + j]);
                t[j] = k.in(0)[row * D + c];
            }
            t[K] = k.in(2)[l * D + c];
            out[l * D + c] = sum.of(t[0 .. K + 1]);
            for (t[0 .. K + 1]) |v| mag[l * D + c] += @abs(v);
        };
    }
};

// ── hyper-connections: mix (dots and rms of the flattened stream), split (Sinkhorn), norm (split + collapse + rms) ──

/// One row's split of the mixes `z`: pre = sigmoid + eps, post = 2 sigmoid, comb = softmax rows + eps, then
/// 2 * ITERS - 1 normalizations, columns first (float64; `out` = pre | post | comb in row-major [a][b]).
pub fn splitRow(z: []const f64, e: f64, sum: Sum, out: []f64) void {
    for (0..HC) |j| {
        out[j] = 1 / (1 + @exp(-z[j])) + e;
        out[HC + j] = 2 / (1 + @exp(-z[HC + j]));
    }
    const v = out[2 * HC ..][0 .. HC * HC];
    var t4: [HC]f64 = undefined;
    for (0..HC) |a| {
        var mx = z[2 * HC + a * HC];
        for (1..HC) |b| mx = maxN(mx, z[2 * HC + a * HC + b]);
        for (0..HC) |b| {
            v[a * HC + b] = @exp(z[2 * HC + a * HC + b] - mx);
            t4[b] = v[a * HC + b];
        }
        const s = sum.of(&t4);
        for (0..HC) |b| v[a * HC + b] = v[a * HC + b] / s + e;
    }
    for (0..2 * ITERS - 1) |t| {
        const rows = t % 2 == 1;
        var sums: [HC]f64 = undefined;
        for (0..HC) |i| {
            for (0..HC) |kk| t4[kk] = if (rows) v[i * HC + kk] else v[kk * HC + i];
            sums[i] = sum.of(&t4);
        }
        for (0..HC) |a| for (0..HC) |b| {
            v[a * HC + b] /= (sums[if (rows) a else b] + e);
        };
    }
}

pub const sinkhorn_ops = 5 * (2 * ITERS - 1) + 25;

const HcSplit = struct {
    pub const Params = struct {};

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0) return null;
        var g = metal.Geometry{ .grid = .{ 32, ci(s.rows), 1 }, .tg = .{ 32, 1, 1 } };
        g.out(&.{ ci(s.rows), HC }, .float32);
        g.out(&.{ ci(s.rows), HC }, .float32);
        g.out(&.{ ci(s.rows), HC, HC }, .float32);
        g.int("HC", HC);
        g.int("ITERS", ITERS);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const z = try k.add(.float32, &.{ k.shape.rows, MIX });
        const eps = try k.add(.float32, &.{2});
        kit.uniform(rng, z, -3, 3);
        eps[0] = EPS_RMS;
        eps[1] = EPS_HC;
        if (f == .nonfinite) kit.plantRows(z, k.shape.rows, MIX);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * (2 * HC + HC * HC);
    }

    /// The kernel writes pre | post | comb as three outputs; `splitRow` gives one row's pack.
    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const L = k.shape.rows;
        var pk: [2 * HC + HC * HC]f64 = undefined;
        for (0..L) |l| {
            splitRow(k.in(0)[l * MIX ..][0..MIX], spec.roundTo(.f32, EPS_HC), sum, &pk);
            @memcpy(out[l * HC ..][0..HC], pk[0..HC]);
            @memcpy(out[L * HC + l * HC ..][0..HC], pk[HC .. 2 * HC]);
            @memcpy(out[2 * L * HC + l * HC * HC ..][0 .. HC * HC], pk[2 * HC ..]);
        }
        for (mag, out) |*m, w| m.* = @abs(w);
    }
};

const HcMix = struct {
    pub const Params = struct { tg: usize = 1024, cols: usize = 6 };
    const R = 4;

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or MIX % p.cols != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(MIX / p.cols * p.tg), ci((s.rows + R - 1) / R), 1 }, .tg = .{ ci(p.tg), 1, 1 } };
        g.out(&.{ ci(s.rows), MIX }, .float32);
        g.int("TG", p.tg);
        g.int("R", R);
        g.int("M", p.cols);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const N = HC * k.shape.n;
        const x = try k.add(act, &.{ L, HC, k.shape.n });
        const fnw = try k.add(.float32, &.{ MIX, N });
        const scale = try k.add(.float32, &.{MIX});
        const base = try k.add(.float32, &.{MIX});
        const eps = try k.add(.float32, &.{2});
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, fnw, -0.02, 0.02);
        kit.uniform(rng, scale, 0.5, 1.5);
        kit.uniform(rng, base, -0.5, 0.5);
        eps[0] = EPS_RMS;
        eps[1] = EPS_HC;
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(x, L, N),
            .stress => {
                // Every row the same +-32 (an exact rsqrt, eps 0), unit scale, no base; each mix's weights +-(2^10 + a
                // small offset) on the first half against -+2^10 on the second, so a term is +32 (2^10 + o) or
                // -32 2^10: partial sums stay below 2^29 in steps of 32 (exact in fp32 in any order), past fp16's
                // range and below bf16's resolution, the total a small multiple of 32.
                eps[0] = 0;
                @memset(scale, 1);
                @memset(base, 0);
                for (x[0..N]) |*v| v.* = if (rng.boolean()) 32 else -32;
                for (1..L) |l| @memcpy(x[l * N ..][0..N], x[0..N]);
                const h = N / 2;
                for (0..MIX) |m| for (0..h) |i| {
                    const b: f64 = 0x1p10 + @as(f64, @floatFromInt(rng.intRangeAtMost(i32, 0, 8)));
                    fnw[m * N + i] = b * x[i] / 32;
                    fnw[m * N + h + i] = -0x1p10 * x[h + i] / 32;
                };
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * MIX;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const N = HC * k.shape.n;
        const t = try k.scratch().alloc(f64, N);
        defer k.scratch().free(t);
        const sq = try k.scratch().alloc(f64, N);
        defer k.scratch().free(sq);
        const eps = spec.roundTo(.f32, k.in(4)[0]);
        for (0..k.shape.rows) |l| {
            const x = k.in(0)[l * N ..][0..N];
            for (sq, x) |*q, v| q.* = v * v;
            const ss = sum.of(sq);
            const r = 1 / @sqrt(ss / @as(f64, @floatFromInt(N)) + eps);
            for (0..MIX) |m| {
                for (t, 0..) |*v, i| v.* = x[i] * k.in(1)[m * N + i];
                const d = sum.of(t);
                out[l * MIX + m] = d * r * k.in(2)[m] + k.in(3)[m];
                var mg: f64 = 0;
                for (t) |v| mg += @abs(v);
                // The dot's error, then the rms's (a positive sum: half its relative error) on the product.
                mag[l * MIX + m] = mg * @abs(r * k.in(2)[m]) + @abs(d * r * k.in(2)[m]) / 2;
            }
        }
    }
};

// ── the indexer's scores: sum over heads of weight * relu(q . k), -inf past the groups a query sees ──

pub fn idxCompute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
    const L = k.shape.rows;
    const nb = k.shape.n;
    const ih = k.p.ih;
    const ihd = k.shape.k;
    const p0: usize = @intFromFloat(k.in(3)[0]);
    const ratio: usize = @intFromFloat(k.in(3)[1]);
    const wsc = k.in(4)[0];
    const t = try k.scratch().alloc(f64, @max(ihd, ih));
    defer k.scratch().free(t);
    var dots: [128]f64 = undefined;
    var dmag: [128]f64 = undefined;
    for (0..L) |l| for (0..nb) |n| {
        const i = l * nb + n;
        if (n >= (p0 + l + 1) / ratio) {
            out[i] = -inf;
            continue;
        }
        for (0..ih) |h| {
            for (0..ihd) |j| t[j] = k.in(0)[(l * ih + h) * ihd + j] * k.in(1)[n * ihd + j];
            dots[h] = sum.of(t[0..ihd]);
            dmag[h] = 0;
            for (t[0..ihd]) |v| dmag[h] += @abs(v);
        }
        var m: f64 = 0;
        for (0..ih) |h| {
            const w = spec.roundTo(.f32, k.in(2)[l * ih + h] * wsc);
            t[h] = w * (if (std.math.isNan(dots[h])) nan else @max(dots[h], 0));
            m += @abs(w) * (dmag[h] + @abs(dots[h]));
        }
        out[i] = sum.of(t[0..ih]);
        mag[i] = m;
    };
}

/// q [L, IH, IHD], keys [nb, IHD], head weights [L, IH], meta {p0, ratio}, the weights' scale.
pub fn idxFill(k: anytype, f: Fill, rng: std.Random, qdt: mlx.mlx_dtype, kdt: mlx.mlx_dtype) !void {
    const L = k.shape.rows;
    const nb = k.shape.n;
    const ih = k.p.ih;
    const ihd = k.shape.k;
    const q = try k.add(qdt, &.{ L, ih, ihd });
    const keys = try k.add(kdt, &.{ nb, ihd });
    const w = try k.add(act, &.{ L, ih });
    const meta = try k.add(.int32, &.{2});
    const wsc = try k.add(.float32, &.{1});
    kit.uniform(rng, q, -2, 2);
    kit.uniform(rng, keys, -2, 2);
    kit.uniform(rng, w, -1, 1);
    // Every query sees most groups; the last few are past the latest ones' reach.
    meta[1] = 1;
    meta[0] = @floatFromInt(nb -| L -| 3);
    wsc[0] = 1 / @sqrt(@as(f64, @floatFromInt(ihd * ih)));
    switch (f) {
        .random => {},
        .nonfinite => kit.plantRows(keys, nb, ihd),
        .stress => {
            // One head, integer keys and queries up to 256 (exact in bf16): each dot mirrored as in `stressDot`, keys
            // negated so relu passes the small positive total; the pair 256 x 256 overflows fp16 in any order.
            @memset(w, 1);
            wsc[0] = 1;
            @memset(q, 0);
            @memset(keys, 0);
            const xs = try k.scratch().alloc(f64, ihd);
            defer k.scratch().free(xs);
            try stressDot(rng, xs, keys, nb, ihd, 256, 256, k.scratch());
            for (keys) |*v| v.* = -v.*;
            for (0..L) |l| @memcpy(q[l * ih * ihd ..][0..ihd], xs);
        },
    }
}

const IdxScore = struct {
    pub const Params = struct { ih: usize = IH };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0 or p.ih * s.k * 4 > 32768) return null;
        var g = metal.Geometry{ .grid = .{ ci((s.n + 255) / 256 * 256), ci(s.rows), 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n) }, .float32);
        g.int("IH", p.ih);
        g.int("IHD", s.k);
        g.dtype("KT", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        try idxFill(k, f, rng, .float32, act);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        try idxCompute(k, out, mag, sum);
    }
};

const IdxTile = struct {
    pub const Params = struct { ih: usize = IH };
    const KT = 64;

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0 or p.ih % 8 != 0 or p.ih > 128 or s.k % 8 != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci((s.n + KT - 1) / KT * 128), ci(s.rows), 1 }, .tg = .{ 128, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n) }, .float32);
        g.int("IH", p.ih);
        g.int("IHD", s.k);
        g.int("KT", KT);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        // The wrapper casts q to the keys' dtype (exact: the rope kernel rounded it there).
        try idxFill(k, f, rng, act, act);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        try idxCompute(k, out, mag, sum);
    }
};

// ── the V4.1 quantizers (`FP8_HEADER`): e4m3 round trips, ue8m0 / e4m3 scales, e2m1 codes ──

/// e4m3fn round-to-nearest-even, saturating at 448 (the kernel sends 464 and above, NaN included, to 448).
fn fp8(v: f64) f64 {
    if (std.math.isNan(v)) return nan;
    const a = @abs(v);
    if (a >= 464) return std.math.copysign(@as(f64, 448), v);
    const e = if (a < 0x1p-6) -9 else std.math.frexp(a).exponent - 1 - 3;
    const step = std.math.ldexp(@as(f64, 1), e);
    return std.math.copysign(rintEven(a / step) * step, v);
}

fn rintEven(x: f64) f64 {
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return if (@mod(f, 2) == 0) f else f + 1;
}

/// The smallest power of two at or above `a` (an f32; subnormals give 2^-126).
fn pow2ceil(a: f32) f64 {
    const b: u32 = @bitCast(a);
    const e = ((b >> 23) & 0xff) + @intFromBool(b & 0x7fffff != 0);
    return @as(f32, @bitCast(e << 23));
}

/// e2m1 by the kernel's thresholds (a NaN propagates in float64).
fn e2m1(v: f64) f64 {
    if (std.math.isNan(v)) return nan;
    const m = @abs(v);
    const lut = [_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const k = @as(usize, @intFromBool(m > 0.25)) + @intFromBool(m > 1.25) + @intFromBool(m > 2.5) + @intFromBool(m > 5) + @intFromBool(m >= 0.75) + @intFromBool(m >= 1.75) + @intFromBool(m >= 3.5);
    return if (v > 0) lut[k] else if (v < 0) -lut[k] else 0;
}

/// A QAT round trip: fp8 / 32 / ue8m0 (1), fp4 / 32 / ue8m0 (2), fp4 / 16 / e4m3 scale (3).
pub const Qk = enum(u8) {
    none = 0,
    fp8 = 1,
    fp4 = 2,
    fp4_e4m3 = 3,

    fn block(q: Qk) usize {
        return if (q == .fp4_e4m3) 16 else 32;
    }

    /// The block scale for a block max `amax` (exact in f32, as the kernel computes it).
    fn scale(q: Qk, amax: f64) f64 {
        const a: f32 = @floatCast(amax);
        return switch (q) {
            .fp8 => pow2ceil(@max(a, @as(f32, 1e-4)) * (1.0 / 448.0)),
            .fp4 => pow2ceil(@max(a, @as(f32, 6.0 * 1.17549435e-38)) * (1.0 / 6.0)),
            .fp4_e4m3 => fp8(@as(f32, @max(a, @as(f32, 6.0 * 0.001953125)) / 6.0)),
            .none => 1,
        };
    }

    /// The scales between the block maxima `lo` and `hi`, in order (`out`; returns how many).
    fn scales(q: Qk, lo: f64, hi: f64, out: []f64) usize {
        var s = q.scale(lo);
        const top = q.scale(hi);
        var n: usize = 0;
        while (n < out.len) {
            out[n] = s;
            n += 1;
            if (s >= top) break;
            // The next scale up: a power of two, or the next e4m3 value.
            s = if (q == .fp4_e4m3) s + std.math.ldexp(@as(f64, 1), if (s < 0x1p-6) -9 else std.math.frexp(s).exponent - 4) else s * 2;
        }
        return n;
    }

    /// `r` through the round trip at `scale`, the quotient's rounding (`slack` f32 ulps) widened toward `dir`.
    fn apply(q: Qk, r: f64, sc: f64, dir: f64) f64 {
        if (std.math.isNan(r)) return nan;
        var y = r / sc;
        if (q == .fp4_e4m3) y += dir * 2 * u * @abs(y);
        return switch (q) {
            .fp8 => fp8(std.math.clamp(y, -448, 448)) * sc,
            .fp4, .fp4_e4m3 => e2m1(std.math.clamp(y, -6, 6)) * sc,
            .none => r,
        };
    }
};

/// Whether `got` (a block) can be the round trip of values in their intervals `iv` (T-grid values, rounded at both
/// ends): every block scale the maxima allow, each value's quantization monotone between its ends; `out_dt` rounds
/// the result. Null when it can, else the finding.
pub fn qatOk(q: Qk, iv: []const Iv, got: []const f64, out_dt: mlx.mlx_dtype, buf: []u8, base: usize) ?[]const u8 {
    if (q == .none) {
        for (iv, got, 0..) |r, g, j| if (!r.round(out_dt).has(g))
            return std.fmt.bufPrint(buf, "output {d} is {e} outside [{e}, {e}]", .{ base + j, g, r.lo, r.hi }) catch "outside";
        return null;
    }
    var lo: f64 = 0;
    var hi: f64 = 0;
    for (iv) |r| {
        const m_lo = if (r.lo <= 0 and r.hi >= 0) 0 else @min(@abs(r.lo), @abs(r.hi));
        lo = @max(lo, m_lo);
        hi = @max(hi, @max(@abs(r.lo), @abs(r.hi)));
    }
    var sc: [8]f64 = undefined;
    const ns = q.scales(lo, hi, &sc);
    for (iv, got, 0..) |r, g, j| {
        const ok = for (sc[0..ns]) |s| {
            const a = rt(out_dt, q.apply(r.lo, s, -1));
            const b = rt(out_dt, q.apply(r.hi, s, 1));
            if (g >= @min(a, b) and g <= @max(a, b)) break true;
        } else false;
        if (!ok) return std.fmt.bufPrint(buf, "output {d} is {e}: [{e}, {e}] round-trips to {e} at scale {e}", .{ base + j, g, r.lo, r.hi, q.apply((r.lo + r.hi) / 2, sc[0], 0), sc[0] }) catch "qat";
    }
    return null;
}

/// The point value of a block's round trip (float64 maxima; gate 1's classes).
pub fn qatPoint(q: Qk, v: []f64) void {
    if (q == .none) return;
    var amax: f64 = 0;
    for (v) |x| amax = maxN(amax, @abs(x));
    const s = q.scale(amax);
    for (v) |*x| x.* = if (std.math.isNan(amax)) nan else q.apply(x.*, s, 0);
}

/// The fast trig's error on an angle, beside the angle's own f32 roundings (3 ulps of it).
const TRIG = 0x1p-16;

/// One adjacent rope pair (traditional) at angle `theta`: the even or odd output's interval from the pair's.
pub fn ropeIv(x1: Iv, x2: Iv, theta: f64, even: bool) Iv {
    const d = @abs(theta) * 3 * u + TRIG;
    const cs = Iv.of(@cos(theta), d);
    const sn = Iv.of(@sin(theta), d);
    const r = if (even) x1.mul(cs).add(x2.mul(sn).neg()) else x1.mul(sn).add(x2.mul(cs));
    return r.widen(2 * u, 0);
}

// ── rope tail (+ QAT): the last RD channels of each D-wide vector roped at position scal * (row + off) ──

/// The release's rope periods (`10000^(k / 32)`), negated for the inverse rope.
pub fn periodsFill(p: []f64, sign: f64) void {
    for (p, 0..) |*v, k| v.* = sign * std.math.pow(f64, 10000, @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(p.len)));
}

pub const RopeTail = struct {
    pub const Params = struct { qk: Qk, ot: mlx.mlx_dtype = act, scal: f64 = 1, off: i32, sign: f64 = 1 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        const n = s.rows * s.n * s.k;
        if (n == 0 or s.k < RD or s.k % p.qk.block() != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(n), 1, 1 }, .tg = .{ ci(@min(n, 256)), 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n), ci(s.k) }, p.ot);
        g.int("RD", RD);
        g.int("ROW", s.n * s.k);
        g.int("QK", @intFromEnum(p.qk));
        g.dtype("T", act);
        g.dtype("OT", p.ot);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const x = try k.add(act, &.{ k.shape.rows, k.shape.n, k.shape.k });
        const periods = try k.add(.float32, &.{RD / 2});
        const off = try k.add(.int32, &.{});
        const scal = try k.add(.float32, &.{1});
        kit.uniform(rng, x, -3, 3);
        periodsFill(periods, k.p.sign);
        off[0] = @floatFromInt(k.p.off);
        scal[0] = k.p.scal;
        if (f == .nonfinite) kit.plantRows(x, k.shape.rows, k.shape.n * k.shape.k);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n * s.k;
    }

    fn theta(k: anytype, row: usize, pair: usize) f64 {
        return k.p.scal * (@as(f64, @floatFromInt(row)) + @as(f64, @floatFromInt(k.p.off))) / spec.roundTo(.f32, k.in(1)[pair]);
    }

    /// Element `i`'s value before the round trip: as an interval, or (point) in float64.
    fn pre(k: anytype, i: usize, point: bool) Iv {
        const D = k.shape.k;
        const x = k.in(0);
        const c = i % D;
        const head = D - RD;
        if (c < head) return Iv.pt(x[i]);
        const b = i - (c - head) % 2;
        const th = theta(k, i / (k.shape.n * D), (c - head) / 2);
        const even = (c - head) % 2 == 0;
        if (point) {
            const v = if (even) x[b] * @cos(th) - x[b + 1] * @sin(th) else x[b] * @sin(th) + x[b + 1] * @cos(th);
            return Iv.pt(bf(v));
        }
        return ropeIv(Iv.pt(x[b]), Iv.pt(x[b + 1]), th, even).round(act);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, _: Sum) !void {
        const blk = k.p.qk.block();
        var i: usize = 0;
        while (i < out.len) : (i += blk) {
            for (0..blk) |j| out[i + j] = pre(k, i + j, true).lo;
            qatPoint(k.p.qk, out[i..][0..blk]);
            for (out[i..][0..blk]) |*v| v.* = rt(k.p.ot, bf(v.*));
        }
        @memset(mag, 0);
    }

    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const n = k.shape.rows * k.shape.n * k.shape.k;
        const a = k.scratch();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        const got = buf[0..n];
        k.outputs(kc.host.elemOf(k.p.ot), got);
        if (at == .nonfinite) {
            try compute(k, buf[n .. 2 * n], buf[2 * n ..], .{});
            return classes(&k.msg, got, buf[n .. 2 * n]);
        }
        const blk = k.p.qk.block();
        var iv: [32]Iv = undefined;
        var i: usize = 0;
        while (i < n) : (i += blk) {
            for (0..blk) |j| iv[j] = pre(k, i + j, false);
            if (qatOk(k.p.qk, iv[0..blk], got[i..][0..blk], act, &k.msg, i)) |why| return why;
        }
        return null;
    }
};

// ── hc_mix_norm: hc_mix's mixes and hc_norm's normed collapse of the same rows in one launch ──

/// hc_fuse.zig's claims are bit-equality with the stock launches: gate 3 proves them against those (`replaced`), the
/// float64 bound the rest.
const fused_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "bit-equal to the stock hc launches (gate 3); random rows, not a stress" };

/// The normed collapse of row `l`'s copies (`x` [L, HC, D], `cpre` [L, HC]) under each weight in `ws`, into
/// `out[c][l * D ..]` (hc_norm's statements in float64); `mag` the error's scale.
fn normedCollapse(x: []const f64, cpre: []const f64, ws: []const []const f64, eps: f64, l: usize, D: usize, sum: Sum, hb: []f64, sq: []f64, out: []const []f64, mag: []const []f64) void {
    for (0..D) |d| {
        var h: f64 = 0;
        for (0..HC) |j| h += cpre[l * HC + j] * x[(l * HC + j) * D + d];
        hb[d] = bf(h);
        sq[d] = hb[d] * hb[d];
    }
    const inv = 1 / @sqrt(sum.of(sq[0..D]) / @as(f64, @floatFromInt(D)) + eps);
    for (ws, out, mag) |w, o, m| for (0..D) |d| {
        o[l * D + d] = w[d] * bf(hb[d] * inv);
        m[l * D + d] = @abs(o[l * D + d]);
    };
}

const HcMixNorm = struct {
    pub const Params = struct { tg: usize = 1024, cols: usize = 6, h2: bool = true };
    const R = 4;

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0 or MIX % p.cols != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci((MIX / p.cols + 1) * p.tg), ci((s.rows + R - 1) / R), 1 }, .tg = .{ ci(p.tg), 1, 1 } };
        g.out(&.{ ci(s.rows), MIX }, .float32);
        g.out(&.{ ci(s.rows), ci(s.n) }, act);
        if (p.h2) g.out(&.{ ci(s.rows), ci(s.n) }, act) else {
            g.out(&.{1}, act);
            g.unread = 1 << 2;
        }
        g.int("TG", p.tg);
        g.int("R", R);
        g.int("M", p.cols);
        g.int("HC", HC);
        g.int("H2", @intFromBool(p.h2));
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        try HcMix.fill(k, f, rng);
        const cpre = try k.add(.float32, &.{ k.shape.rows, HC });
        const w = try k.add(act, &.{k.shape.n});
        const w2 = try k.add(act, &.{k.shape.n});
        kit.uniform(rng, cpre, 0, 1);
        kit.uniform(rng, w, 0.5, 1.5);
        kit.uniform(rng, w2, 0.5, 1.5);
    }

    pub fn count(p: Params, s: Shape) usize {
        return s.rows * MIX + s.rows * s.n * @as(usize, if (p.h2) 2 else 1);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const L = k.shape.rows;
        const D = k.shape.n;
        const zn = L * MIX;
        try HcMix.compute(k, out[0..zn], mag[0..zn], sum);
        const a = k.scratch();
        const buf = try a.alloc(f64, 2 * D);
        defer a.free(buf);
        const ws = [_][]const f64{ k.in(6), k.in(7) };
        const outs = [_][]f64{ out[zn..][0 .. L * D], if (k.p.h2) out[zn + L * D ..][0 .. L * D] else &.{} };
        const mags = [_][]f64{ mag[zn..][0 .. L * D], if (k.p.h2) mag[zn + L * D ..][0 .. L * D] else &.{} };
        const nh: usize = if (k.p.h2) 2 else 1;
        for (0..L) |l| normedCollapse(k.in(0), k.in(5), ws[0..nh], spec.roundTo(.f32, k.in(4)[0]), l, D, sum, buf[0..D], buf[D..], outs[0..nh], mags[0..nh]);
    }

    /// The stock launches: hc_mix's mixes, then hc_norm's normed collapse (twice with `w2`).
    pub fn replaced(k: anytype) ![]const u8 {
        var sc = v41.Scope.init(k.rig.a, k.rig.stream());
        defer sc.deinit();
        var m: v41.Dsv41Model = undefined;
        m.s = k.rig.stream();
        m.cfg = .{};
        m.cfg.dsv4_hc_mult = HC;
        m.cfg.dsv4_hc_sinkhorn_iters = ITERS;
        m.hc_tg = k.p.tg;
        m.hc_mix_kernel = try v41.buildKernel("dsv41_hc_mix", &.{ "x", "fn", "scale", "base", "eps" }, &.{"z"}, v41.HC_MIX_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.hc_mix_kernel);
        m.hc_norm_kernel = try v41.buildKernel("dsv41_hc_norm", &.{ "z", "x", "cpre", "w", "eps" }, &.{ "pre", "post", "comb", "h_out" }, v41.HC_NORM_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.hc_norm_kernel);
        const hc: v41.Hc = .{ .fn_w = k.args[1], .scale = k.args[2], .base = k.args[3] };
        const z = try v41.hcMixWide(&sc, &m, k.args[0], &hc, k.args[4]);
        const n = try v41.hcNorm(&sc, &m, z, k.args[0], k.args[5], k.args[6], k.args[4]);
        if (!k.p.h2) return k.rig.bytesOf(&.{ z, n.h });
        const n2 = try v41.hcNorm(&sc, &m, z, k.args[0], k.args[5], k.args[7], k.args[4]);
        return k.rig.bytesOf(&.{ z, n.h, n2.h });
    }
};

// ── the hc expand: the new stream `post * y + comb^T res`, into a bound region or with its normed collapse ──

/// Row `l`'s new stream (`y` [L, D], `res` [L, HC, D], `post` [L, HC], `comb` [L, HC, HC]) into `out[l * HC * D ..]`,
/// rounded to `act` when a later stage reads it (`round`); `mag` the sum of its terms' magnitudes.
fn expandRow(y: []const f64, res: []const f64, post: []const f64, comb: []const f64, l: usize, D: usize, round: bool, out: []f64, mag: []f64) void {
    for (0..HC) |i| for (0..D) |d| {
        const o = (l * HC + i) * D + d;
        var v = post[l * HC + i] * y[l * D + d];
        var m = @abs(v);
        for (0..HC) |j| {
            const t = comb[(l * HC + j) * HC + i] * res[(l * HC + j) * D + d];
            v += t;
            m += @abs(t);
        }
        out[o] = if (round) bf(v) else v;
        mag[o] = m;
    };
}

/// The expand's inputs in the wrapper's order (`y`, `res`, `post`, `comb`), shaped as `expandInto` or the fused launch
/// binds them (`lead`: `y` [1, L, D] and `res` [1, L, HC, D]).
fn expandFill(k: anytype, f: Fill, rng: std.Random, lead: bool) !void {
    const L = k.shape.rows;
    const D = k.shape.n;
    const y = try k.add(act, if (lead) &.{ 1, L, D } else &.{ L, D });
    const res = try k.add(act, if (lead) &.{ 1, L, HC, D } else &.{ L, HC, D });
    const post = try k.add(.float32, &.{ L, HC });
    const comb = try k.add(.float32, &.{ L, HC, HC });
    kit.uniform(rng, y, -3, 3);
    kit.uniform(rng, res, -3, 3);
    kit.uniform(rng, post, 0, 2);
    kit.uniform(rng, comb, 0, 0.5);
    if (f == .nonfinite) {
        kit.plantRows(y, L, D);
        kit.plantRows(res, L, HC * D);
    }
}

/// The stock expand (glm5's) over the case's inputs, [L, HC, D].
fn stockExpand(sc: *v41.Scope, k: anytype) !mlx.mlx_array {
    const L = ci(k.shape.rows);
    const D = ci(k.shape.n);
    const y = try sc.reshape(k.args[0], &.{ L, D });
    const res = try sc.reshape(k.args[1], &.{ L, HC, D });
    return v41.hcExpand(sc, res, y, &.{ .pre = none, .post = k.args[2], .comb = k.args[3] });
}

/// kernels/hc_expand_into.metal: the new stream into `dst`, a bound region.
const ExpandInto = struct {
    pub const Params = struct {};

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0 or s.n % 4 != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(s.n / 4), ci(s.rows), 1 }, .tg = .{ ci(@min(256, s.n / 4)), 1, 1 } };
        g.out(&.{1}, .uint32);
        g.unread = 1;
        g.dtype("T", act);
        g.int("HC", HC);
        g.int("D", s.n);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        try expandFill(k, f, rng, true);
        const dst = try k.add(act, &.{ k.shape.rows, HC, k.shape.n });
        @memset(dst, nan);
        _ = try k.add(.uint32, &.{1});
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * HC * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, _: Sum) !void {
        for (0..k.shape.rows) |l| expandRow(k.in(0), k.in(1), k.in(2), k.in(3), l, k.shape.n, false, out, mag);
    }

    /// The region the kernel wrote.
    pub fn output(k: anytype, own: *std.ArrayList(u8)) !void {
        try own.appendSlice(k.scratch(), try k.rig.bytesOf(&.{k.args[4]}));
    }

    pub fn replaced(k: anytype) ![]const u8 {
        var sc = v41.Scope.init(k.rig.a, k.rig.stream());
        defer sc.deinit();
        return k.rig.bytesOf(&.{try stockExpand(&sc, k)});
    }
};

/// kernels/hc_expand_norm.metal: the expand and hc_norm's normed collapse of the new stream in one launch, the new
/// stream a fresh `x1` or (`into`) written into `dst`.
const ExpandNorm = struct {
    pub const Params = struct { into: bool, tg: usize = 1024 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(p.tg), ci(s.rows), 1 }, .tg = .{ ci(p.tg), 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n) }, act);
        if (p.into) {
            g.out(&.{1}, .uint32);
            g.unread = 1 << 1;
        } else g.out(&.{ ci(s.rows), HC, ci(s.n) }, act);
        g.int("TG", p.tg);
        g.int("HC", HC);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        try expandFill(k, f, rng, false);
        const cpre = try k.add(.float32, &.{ k.shape.rows, HC });
        const w = try k.add(act, &.{k.shape.n});
        const eps = try k.add(.float32, &.{2});
        kit.uniform(rng, cpre, 0, 1);
        kit.uniform(rng, w, 0.5, 1.5);
        eps[0] = EPS_RMS;
        eps[1] = EPS_HC;
        if (k.p.into) {
            const dst = try k.add(act, &.{ k.shape.rows, HC, k.shape.n });
            @memset(dst, nan);
            _ = try k.add(.uint32, &.{1});
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n + s.rows * HC * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const L = k.shape.rows;
        const D = k.shape.n;
        const x1 = out[L * D ..];
        for (0..L) |l| expandRow(k.in(0), k.in(1), k.in(2), k.in(3), l, D, true, x1, mag[L * D ..]);
        const a = k.scratch();
        const buf = try a.alloc(f64, 2 * D);
        defer a.free(buf);
        for (0..L) |l| normedCollapse(x1, k.in(4), &.{k.in(5)}, spec.roundTo(.f32, k.in(6)[0]), l, D, sum, buf[0..D], buf[D..], &.{out[0 .. L * D]}, &.{mag[0 .. L * D]});
    }

    /// `h_out`, then the new stream (`dst` when written into it).
    pub fn output(k: anytype, own: *std.ArrayList(u8)) !void {
        const a = k.scratch();
        try own.appendSlice(a, k.rig.outputBytes()[0 .. k.shape.rows * k.shape.n * 2]);
        if (k.p.into) try own.appendSlice(a, try k.rig.bytesOf(&.{k.args[7]})) else try own.appendSlice(a, k.rig.outputBytes()[k.shape.rows * k.shape.n * 2 ..]);
    }

    /// The stock launches: glm5's expand, then hc_norm's normed collapse of the new stream.
    pub fn replaced(k: anytype) ![]const u8 {
        var sc = v41.Scope.init(k.rig.a, k.rig.stream());
        defer sc.deinit();
        var m: v41.Dsv41Model = undefined;
        m.s = k.rig.stream();
        m.cfg = .{};
        m.cfg.dsv4_hc_mult = HC;
        m.cfg.dsv4_hc_sinkhorn_iters = ITERS;
        m.hc_tg = k.p.tg;
        m.hc_norm_kernel = try v41.buildKernel("dsv41_hc_norm", &.{ "z", "x", "cpre", "w", "eps" }, &.{ "pre", "post", "comb", "h_out" }, v41.HC_NORM_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.hc_norm_kernel);
        const x1 = try stockExpand(&sc, k);
        const z = try sc.zeros(&.{ ci(k.shape.rows), MIX }, .float32);
        const n = try v41.hcNorm(&sc, &m, z, x1, k.args[4], k.args[5], k.args[6]);
        return k.rig.bytesOf(&.{ n.h, x1 });
    }
};

// ── decode/gemv_twin.zig: MLX's one-row gemv over a verify's rows, each row its one-row words ──

/// A launch shape of the twin: the header `gemv_twin.header` writes for the plan of an [n, k] weight tiled by `v`;
/// `m12` its M1/M2 compiled thread limit.
fn twinLaunch(comptime n: usize, comptime k: usize, comptime v: gtwin.Variant, comptime m12: u32) metal.Launch {
    const p = comptime gtwin.tiled(gtwin.plan(n, k), n, v).?;
    const h: [:0]const u8 = comptime blk: {
        var buf: [512]u8 = undefined;
        const s = gtwin.header(&buf, "bfloat16_t", p, v.nr, v.ilv) catch unreachable;
        var out: [s.len:0]u8 = undefined;
        @memcpy(out[0..s.len], s);
        const final = out;
        break :blk &final;
    };
    const block_m = p.bm * p.sm * p.tm;
    return .{
        .name = std.fmt.comptimePrint("dsv41_gemv_twin_bf16_bm{d}_bn{d}_sm{d}_sn{d}_tm{d}_tn{d}_r{d}{s}", .{ p.bm, p.bn, p.sm, p.sn, p.tm, p.tn, v.nr, if (v.ilv) "_i" else "" }),
        .inputs = &.{ "x", "w" },
        .outputs = &.{"y"},
        .source = gtwin.source,
        .header = h,
        .tg_bytes = 4 * v.nr * (if (p.bn > 1) p.bn * (block_m + p.tm) else 1),
        .m1 = m12,
        .m2 = m12,
    };
}

const GemvTwin = struct {
    pub const Params = struct { v: gtwin.Variant };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows < 2 or s.rows > gtwin.max_rows or s.n == 0 or s.k == 0) return null;
        const t = gtwin.tiled(gtwin.plan(s.n, s.k), s.n, p.v) orelse return null;
        const chunks = (s.rows + p.v.nr - 1) / p.v.nr;
        var g = if (p.v.ilv)
            metal.Geometry{ .grid = .{ ci(t.tgs * chunks * 32), ci(t.bn), ci(t.bm) }, .tg = .{ 32, ci(t.bn), ci(t.bm) } }
        else
            metal.Geometry{ .grid = .{ ci(t.tgs * 32), ci(t.bn), ci(chunks * t.bm) }, .tg = .{ 32, ci(t.bn), ci(t.bm) } };
        g.out(&.{ ci(s.rows), ci(s.n) }, act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const K = k.shape.k;
        const x = try k.add(act, &.{ L, K });
        const w = try k.add(act, &.{ k.shape.n, K });
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, w, -2, 2);
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(x, L, K),
            // A short row takes larger terms (k/2 x 64 x 64 stays below 2^24) to push fp16's partial sums out of range.
            .stress => {
                try stressDot(rng, x[0..K], w, k.shape.n, K, 64, if (K <= 1024) 64 else 7, k.scratch());
                for (1..L) |r| @memcpy(x[r * K ..][0..K], x[0..K]);
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const K = k.shape.k;
        const N = k.shape.n;
        const t = try k.scratch().alloc(f64, K);
        defer k.scratch().free(t);
        for (0..k.shape.rows) |r| for (0..N) |j| {
            var m: f64 = 0;
            for (t, k.in(0)[r * K ..][0..K], k.in(1)[j * K ..][0..K]) |*v, a, b| {
                v.* = a * b;
                m += @abs(v.*);
            }
            out[r * N + j] = sum.of(t);
            mag[r * N + j] = m;
        };
    }

    /// MLX's matmul of each row alone (its M = 1 gemv), the rows stacked.
    pub fn replaced(k: anytype) ![]const u8 {
        var sc = v41.Scope.init(k.rig.a, k.rig.stream());
        defer sc.deinit();
        const wt = try sc.transpose(k.args[1], &.{ 1, 0 });
        var rows: [gtwin.max_rows]A = undefined;
        for (rows[0..k.shape.rows], 0..) |*y, r| y.* = try sc.matmul(try sc.slice(k.args[0], 0, r, r + 1), wt);
        return k.rig.bytesOf(&.{try sc.concat(rows[0..k.shape.rows], 0)});
    }
};

// ── prompt/gate_gemm.zig: the wide router's f32 logits as MLX's fused NAX GEMM over the stored types ──

const GateGemm = struct {
    pub const Params = struct {};
    const t = ggemm.tile;

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0 or s.k == 0 or !ggemm.mlxRegular(@intCast(s.rows), @intCast(s.n), @intCast(s.k))) return null;
        const swz: usize = @as(usize, 1) << @intCast(t.swz);
        const tiles_m = (s.rows + t.bm - 1) / t.bm;
        const tiles_n = (s.n + t.bn - 1) / t.bn;
        var g = metal.Geometry{ .grid = .{ ci(tiles_n * swz * 32), ci((tiles_m + swz - 1) / swz * t.wn), ci(t.wm) }, .tg = .{ 32, ci(t.wn), ci(t.wm) } };
        g.out(&.{ ci(s.rows), ci(s.n) }, .float32);
        g.dtype("T", .float32);
        g.dtype("TA", act);
        g.dtype("TB", .float16);
        g.dtype("OUT_R", .float32);
        g.int("BM", t.bm);
        g.int("BN", t.bn);
        g.int("BK", t.bk);
        g.int("WM", t.wm);
        g.int("WN", t.wn);
        g.int("SWZ", t.swz);
        g.int("AM", @intFromBool(s.rows % t.bm == 0));
        g.int("AN", @intFromBool(s.n % t.bn == 0));
        g.int("AK", @intFromBool(s.k % t.bk == 0));
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const M = k.shape.rows;
        const K = k.shape.k;
        const x = try k.add(act, &.{ M, K });
        const w = try k.add(.float16, &.{ k.shape.n, K });
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, w, -2, 2);
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(x, M, K),
            .stress => {
                try stressDot(rng, x[0..K], w, k.shape.n, K, 64, 7, k.scratch());
                for (1..M) |r| @memcpy(x[r * K ..][0..K], x[0..K]);
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        return GemvTwin.compute(k, out, mag, sum);
    }

    /// The chain the router ran: both operands cast to f32, MLX's matmul.
    pub fn replaced(k: anytype) ![]const u8 {
        var sc = v41.Scope.init(k.rig.a, k.rig.stream());
        defer sc.deinit();
        const w = try sc.astype(k.args[1], .float32);
        return k.rig.bytesOf(&.{try sc.matmul(try sc.astype(k.args[0], .float32), try sc.transpose(w, &.{ 1, 0 }))});
    }
};

// ── prompt/flash_attn.zig: a wide span's attention as one kernel, an online softmax over blocks of 128 keys ──

/// Lossy (P rounds to f16 against the running max), so within the float64 online softmax's bound (the SDK attention
/// contract's `online`: the sink's start, each block's rescale, the fast exp, P's rounding), not the gather's words.
const AttnFlash = struct {
    /// `Gathered`'s geometry (its keys), grouped as attnOut reads the heads; `qt`: the roped tail read from `qt`.
    pub const Params = struct { wp: usize = 100, nc: usize = 700, p0: usize = 40, grouped: bool = true, og: usize = 8, qt: bool };
    const HGN: usize = @intCast(fa.hgn);
    const BK = 128;

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (!Gathered.ok(s) or s.n % (16 * HGN) != 0 or s.n % p.og != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(32 * s.n / (16 * HGN)), ci(s.rows), ci(4 * HGN) }, .tg = .{ 32, 1, ci(4 * HGN) } };
        g.out(&.{ ci(p.og), ci(s.rows), ci(s.n / p.og * HD) }, act);
        g.int("HGN", HGN);
        g.int("QT", @intFromBool(p.qt));
        g.int("RD", RD);
        g.dtype("T", act);
        g.dtype("OT", act);
        return g;
    }

    /// q, the keys (`Gathered.addKeys`: window rows, compressed rows, picks, meta), qt, the sink, the scale, the
    /// inverse rope's periods.
    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const q = try k.add(act, &.{ L, H, HD });
        kit.uniform(rng, q, -2, 2);
        try Gathered.addKeys(k, rng);
        const qt = try k.add(act, &.{ L, H, 64 });
        kit.uniform(rng, qt, -2, 2);
        kit.uniform(rng, try k.add(.float32, &.{ 1, H, 1 }), -3, 3);
        const scal = try k.add(.float32, &.{1});
        scal[0] = 0.05;
        periodsFill(try k.add(.float32, &.{RD / 2}), -1);
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(q, L * H, HD),
            // Every query one stress vector (its tail in qt too), every key row a stress row against it; scale 1, so a
            // score's sum error moves the weights.
            .stress => {
                scal[0] = 1;
                const kw = k.in(1);
                const comp = k.in(2);
                const rows = (kw.len + comp.len) / HD;
                const w = try k.scratch().alloc(f64, rows * HD);
                defer k.scratch().free(w);
                try stressDot(rng, q[0..HD], w, rows, HD, 64, 64, k.scratch());
                @memcpy(kw, w[0..kw.len]);
                @memcpy(comp, w[kw.len..]);
                for (1..L * H) |r| @memcpy(q[r * HD ..][0..HD], q[0..HD]);
                for (0..L * H) |r| @memcpy(qt[r * 64 ..][0..64], q[HD - 64 .. HD]);
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n * HD;
    }

    /// Each head's online softmax from its sink over its keys (absent ones weigh 0), O / l, then the epilogue's
    /// inverse rope on the f32 values (fast trig); `mag` carries the whole allowance (depth one).
    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const NK = WINDOW + k.shape.k;
        const hg = H / k.p.og;
        const a = k.scratch();
        const keys = try a.alloc(satt.Key, NK);
        defer a.free(keys);
        const ends = try a.alloc(usize, (NK + BK - 1) / BK);
        defer a.free(ends);
        for (ends, 0..) |*e, b| e.* = @min(NK, (b + 1) * BK);
        const buf = try a.alloc(f64, 5 * HD);
        defer a.free(buf);
        const qv = buf[0..HD];
        const terms = buf[HD .. 2 * HD];
        const o = buf[2 * HD .. 3 * HD];
        const eo = buf[3 * HD .. 4 * HD];
        const zero = buf[4 * HD ..];
        @memset(zero, 0);
        const sim: satt.Sim = if (sum.acc) |x| .{ .acc = x, .order = sum.order } else null;
        const scale = k.in(7)[0];
        for (0..L) |l| for (0..H) |h| {
            @memcpy(qv, k.in(0)[(l * H + h) * HD ..][0..HD]);
            if (k.p.qt) @memcpy(qv[HD - 64 ..], k.in(5)[(l * H + h) * 64 ..][0..64]);
            for (keys, 0..) |*key, c| key.* = if (Gathered.key(k, 1, l, c)) |row| blk: {
                const d = satt.dot(qv, row, scale, HD, 1, sim, terms);
                break :blk .{ .s = d.v, .e = d.e, .v = row };
            } else .{ .s = -inf, .on = false, .v = zero };
            const st = satt.online(.{ .p_unit = 0x1p-11, .p_abs = 0x1p-25 }, .{ .m = k.in(6)[h], .l = 1 }, keys, ends, o, eo);
            satt.normalize(st, o, eo);
            const base = ((h / hg) * L + l) * hg * HD + (h % hg) * HD;
            for (0..HD - RD) |d| {
                out[base + d] = o[d];
                mag[base + d] = kc.host.allowance(eo[d]);
            }
            const pos: f64 = @floatFromInt(k.p.p0 + l);
            var d: usize = HD - RD;
            while (d < HD) : (d += 2) {
                const th = pos * spec.roundTo(.f32, 1 / spec.roundTo(.f32, k.in(8)[(d - (HD - RD)) / 2]));
                const x0 = o[d];
                const x1 = o[d + 1];
                out[base + d] = x0 * @cos(th) - x1 * @sin(th);
                out[base + d + 1] = x0 * @sin(th) + x1 * @cos(th);
                const e = eo[d] + eo[d + 1] + (@abs(x0) + @abs(x1)) * (@abs(th) * 3 * u + TRIG + 2 * u);
                mag[base + d] = kc.host.allowance(e);
                mag[base + d + 1] = kc.host.allowance(e);
            }
        };
    }
};

// ── spec/markov.zig: the DSpark draft's Markov chain, a step a launch and the last one ──

const MK_NT: usize = mplan.step_groups;
const MK_NONE: u32 = 0xFFFFFFFF;

/// markov_header.metal's order on (value, index) pairs: NaN first (the lower index among NaNs), then the larger value,
/// then the lower index.
fn mkBetter(va: f32, ia: u32, vb: f32, ib: u32) bool {
    const na = std.math.isNan(va);
    const nb = std.math.isNan(vb);
    if (na != nb) return na;
    if (na) return ia < ib;
    if (va != vb) return va > vb;
    return ia < ib;
}

fn bitsOf(v: f64) f32 {
    return @bitCast(@as(u32, @intFromFloat(v)));
}

/// The index of the best of `n` (value bits, index) pairs at `p`.
fn mkBest(p: []const f64, n: usize) u32 {
    var bv = -std.math.inf(f32);
    var bi = MK_NONE;
    for (0..n) |j| {
        const v = bitsOf(p[2 * j]);
        const i: u32 = @intFromFloat(p[2 * j + 1]);
        if (mkBetter(v, i, bv, bi)) {
            bv = v;
            bi = i;
        }
    }
    return bi;
}

/// Step `K`'s pairs from its row: threadgroup g's best over the quads its simdgroups read (with zig-zag), into `out`
/// as (value bits, index) per threadgroup.
fn mkPairs(row: []const f32, K: usize, out: []f64) void {
    const Q: u32 = @intCast(row.len / 4);
    const ns: u32 = @intCast(MK_NT * mplan.step_simdgroups);
    for (0..MK_NT) |g| {
        var bv = -std.math.inf(f32);
        var bi = MK_NONE;
        for (0..mplan.step_simdgroups) |sg| {
            var q: u32 = @intCast(g * mplan.step_simdgroups + sg);
            while (q < Q) : (q += ns) {
                const r0 = 4 * mplan.quadAt(q, Q, @intCast(K), true);
                for (r0..r0 + 4) |i| if (mkBetter(row[i], @intCast(i), bv, bi)) {
                    bv = row[i];
                    bi = @intCast(i);
                };
            }
        }
        out[2 * g] = @floatFromInt(@as(u32, @bitCast(bv)));
        out[2 * g + 1] = @floatFromInt(bi);
    }
}

fn mkPairsBytes(a: std.mem.Allocator, pairs: []const f64) ![]u8 {
    const b = try a.alloc(u8, 4 * pairs.len);
    for (pairs, 0..) |v, i| std.mem.writeInt(u32, b[4 * i ..][0..4], @intFromFloat(v), .little);
    return b;
}

/// kernels/markov_step.metal at step K = rows - 1 (`shape.n` the vocabulary, `shape.k` the rank), KEEP: the earlier
/// steps' pairs passed on, this step's pairs a threadgroup, then the biased row.
const MarkovStep = struct {
    pub const Params = struct {};

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.rows > mplan.max_block) return null;
        if (mplan.refusal(.{ .act = .bf16, .embed = .bf16, .hout = .bf16, .vocab = s.n, .rank = s.k, .dim = 1, .block = s.rows }) != null) return null;
        var g = metal.Geometry{ .grid = .{ ci(MK_NT * mplan.step_threads), 1, 1 }, .tg = .{ ci(mplan.step_threads), 1, 1 } };
        g.out(&.{ci(s.rows * MK_NT * 2)}, .uint32);
        g.out(&.{ci(s.n)}, .float32);
        g.dtype("T", act);
        g.dtype("E", act);
        g.int("R", s.k);
        g.int("V", s.n);
        g.int("K", s.rows - 1);
        g.int("NT", MK_NT);
        g.int("KEEP", 1);
        g.int("ZZ", 1);
        return g;
    }

    /// The token before step K: t1, or the best of step K - 1's pairs.
    fn prev(k: anytype) usize {
        const K = k.shape.rows - 1;
        if (K == 0) return @intFromFloat(k.in(3)[0]);
        return mkBest(k.in(4)[2 * (K - 1) * MK_NT ..], MK_NT);
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const K = k.shape.rows - 1;
        const V = k.shape.n;
        const R = k.shape.k;
        const lg = try k.add(.float32, &.{ k.shape.rows, V });
        const emb = try k.add(act, &.{ V, R });
        const mh = try k.add(act, &.{ V, R });
        const t1 = try k.add(.int32, &.{1});
        const pin = try k.add(.uint32, &.{@max(K * MK_NT * 2, 8)});
        kit.uniform(rng, lg, -4, 4);
        kit.uniform(rng, emb, -1, 1);
        kit.uniform(rng, mh, -1, 1);
        t1[0] = @floatFromInt(rng.uintLessThan(usize, V));
        for (0..K * MK_NT) |j| {
            pin[2 * j] = @floatFromInt(@as(u32, @bitCast(@as(f32, @floatCast(rng.float(f64) * 20 - 10)))));
            pin[2 * j + 1] = @floatFromInt(rng.uintLessThan(usize, V));
        }
        switch (f) {
            .random => {},
            .nonfinite => {
                kit.plantRows(lg[K * V ..][0..V], 4, V / 4);
                kit.plantRows(mh, V, R);
                if (K > 0) pin[2 * ((K - 1) * MK_NT + 7)] = @floatFromInt(@as(u32, 0x7FC00000));
            },
            // The token's row and every head row a stress pair: each bias a small integer, fp32-exact in any order.
            .stress => {
                for (lg) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -3, 3));
                try stressDot(rng, emb[prev(k) * R ..][0..R], mh, V, R, 64, 64, k.scratch());
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * MK_NT * 2 + s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const K = k.shape.rows - 1;
        const V = k.shape.n;
        const R = k.shape.k;
        const np = k.shape.rows * MK_NT * 2;
        @memset(mag, 0);
        @memcpy(out[0 .. K * MK_NT * 2], k.in(4)[0 .. K * MK_NT * 2]);
        const a = k.scratch();
        const terms = try a.alloc(f64, R);
        defer a.free(terms);
        const row = try a.alloc(f32, V);
        defer a.free(row);
        const e = k.in(1)[prev(k) * R ..][0..R];
        for (0..V) |v| {
            var m: f64 = 0;
            for (terms, k.in(2)[v * R ..][0..R], e) |*x, w, ev| {
                x.* = w * ev;
                m += @abs(x.*);
            }
            out[np + v] = k.in(0)[K * V + v] + bf(sum.of(terms));
            mag[np + v] = m;
            row[v] = @floatCast(out[np + v]);
        }
        mkPairs(row, K, out[K * MK_NT * 2 .. np]);
    }

    /// The op chain (markov_embed's row in the activation dtype by markov_head^T, cast to f32, plus the logits row), its
    /// pairs a threadgroup from that row, the earlier pairs as given.
    pub fn replaced(k: anytype) ![]const u8 {
        const K = k.shape.rows - 1;
        const V = k.shape.n;
        var sc = v41.Scope.init(k.rig.a, k.rig.stream());
        defer sc.deinit();
        const p = prev(k);
        const e = try sc.astype(try sc.slice(k.args[1], 0, p, p + 1), act);
        const bias = try sc.astype(try sc.matmul(e, try sc.t2(k.args[2])), .float32);
        const rowa = try sc.add(try sc.slice(k.args[0], 0, K, K + 1), bias);
        const ar = k.arena.allocator();
        const rb = try ar.dupe(u8, try k.rig.bytesOf(&.{rowa}));
        const row = try ar.alloc(f32, V);
        for (row, 0..) |*x, i| x.* = @bitCast(std.mem.readInt(u32, rb[4 * i ..][0..4], .little));
        const pairs = try ar.alloc(f64, k.shape.rows * MK_NT * 2);
        @memcpy(pairs[0 .. K * MK_NT * 2], k.in(4)[0 .. K * MK_NT * 2]);
        mkPairs(row, K, pairs[K * MK_NT * 2 ..]);
        return std.mem.concat(ar, u8, &.{ try mkPairsBytes(ar, pairs), rb });
    }
};

/// kernels/markov_final.metal, CONF: each position's token (the best of its step's pairs) and the confidence head's
/// logit over [hout, markov_embed of the token before] (`shape.n` the hidden width, `shape.k` the rank).
const MarkovFinal = struct {
    pub const Params = struct {};
    /// markov_embed's rows (the tokens' range).
    const V = 2048;

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.rows > mplan.max_block or s.n == 0 or s.k == 0 or s.n + s.k > mplan.max_conf_len) return null;
        var g = metal.Geometry{ .grid = .{ ci(mplan.final_threads), ci(s.rows), 1 }, .tg = .{ ci(mplan.final_threads), 1, 1 } };
        g.out(&.{ci(s.rows)}, .int32);
        g.out(&.{ci(s.rows)}, .float32);
        g.int("B", s.rows);
        g.int("NT", MK_NT);
        g.int("D", s.n);
        g.int("R", s.k);
        g.dtype("H", act);
        g.dtype("E", act);
        g.int("CONF", 1);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const B = k.shape.rows;
        const D = k.shape.n;
        const R = k.shape.k;
        const parts = try k.add(.uint32, &.{B * MK_NT * 2});
        const t1 = try k.add(.int32, &.{1});
        const hout = try k.add(act, &.{ B, D });
        const emb = try k.add(act, &.{ V, R });
        const cw = try k.add(.float32, &.{ 1, D + R });
        for (0..B * MK_NT) |j| {
            parts[2 * j] = @floatFromInt(@as(u32, @bitCast(@as(f32, @floatCast(rng.float(f64) * 20 - 10)))));
            parts[2 * j + 1] = @floatFromInt(rng.uintLessThan(usize, V));
        }
        t1[0] = @floatFromInt(rng.uintLessThan(usize, V));
        kit.uniform(rng, hout, -2, 2);
        kit.uniform(rng, emb, -2, 2);
        // The fused logit's claim holds for 16-bit floats' weights (`plan.exact16`).
        kit.uniform(rng, cw, -0.5, 0.5);
        for (cw) |*v| v.* = bf(v.*);
        switch (f) {
            .random => {},
            .nonfinite => {
                kit.plantRows(hout, B, D);
                parts[2 * (2 * MK_NT + 5)] = @floatFromInt(@as(u32, 0x7FC00000));
            },
            // Every step's best the same token p (also t1): every logit one stress dot over [hout row, emb row p].
            .stress => {
                const p = rng.uintLessThan(usize, V);
                t1[0] = @floatFromInt(p);
                for (0..B) |s| {
                    parts[2 * s * MK_NT] = @floatFromInt(@as(u32, @bitCast(@as(f32, 1e9))));
                    parts[2 * s * MK_NT + 1] = @floatFromInt(p);
                }
                const xv = try k.scratch().alloc(f64, D + R);
                defer k.scratch().free(xv);
                try stressDot(rng, xv, cw, 1, D + R, 64, 7, k.scratch());
                for (0..B) |s| @memcpy(hout[s * D ..][0..D], xv[0..D]);
                @memcpy(emb[p * R ..][0..R], xv[D..]);
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return 2 * s.rows;
    }

    fn toks(k: anytype, out: []u32) void {
        for (out, 0..) |*t, s| t.* = mkBest(k.in(0)[2 * s * MK_NT ..], MK_NT);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const B = k.shape.rows;
        const D = k.shape.n;
        const R = k.shape.k;
        var tk: [mplan.max_block]u32 = undefined;
        toks(k, tk[0..B]);
        const terms = try k.scratch().alloc(f64, D + R);
        defer k.scratch().free(terms);
        @memset(mag, 0);
        for (0..B) |s| {
            out[s] = @floatFromInt(tk[s]);
            const p: usize = if (s == 0) @intFromFloat(k.in(1)[0]) else tk[s - 1];
            var m: f64 = 0;
            for (terms, 0..) |*x, i| {
                x.* = (if (i < D) k.in(2)[s * D + i] else k.in(3)[p * R + i - D]) * k.in(4)[i];
                m += @abs(x.*);
            }
            out[B + s] = sum.of(terms);
            mag[B + s] = m;
        }
    }

    /// The op chain: each token MLX's argmax of its row (the best of its pairs), each logit MLX's matmul of
    /// [f32 hout row, f32 markov_embed row of the token before] by the head.
    pub fn replaced(k: anytype) ![]const u8 {
        const B = k.shape.rows;
        var sc = v41.Scope.init(k.rig.a, k.rig.stream());
        defer sc.deinit();
        var tk: [mplan.max_block]u32 = undefined;
        toks(k, tk[0..B]);
        var ti: [mplan.max_block]i32 = undefined;
        for (ti[0..B], tk[0..B]) |*x, v| x.* = @intCast(v);
        const ta = try sc.keep(mlx.mlx_array_new_data(&ti, &[_]c_int{ci(B)}, 1, .int32));
        const cwt = try sc.t2(k.args[4]);
        var confs: [mplan.max_block]A = undefined;
        for (confs[0..B], 0..) |*c, s| {
            const p: usize = if (s == 0) @intFromFloat(k.in(1)[0]) else tk[s - 1];
            const h = try sc.astype(try sc.slice(k.args[2], 0, s, s + 1), .float32);
            const e = try sc.astype(try sc.slice(k.args[3], 0, p, p + 1), .float32);
            c.* = try sc.matmul(try sc.concat(&.{ h, e }, -1), cwt);
        }
        return k.rig.bytesOf(&.{ ta, try sc.concat(confs[0..B], 0) });
    }
};

// ── the query tail: rope_tail's roped channels alone, [L, H, RD] ──

const RopeTailOnly = struct {
    /// `RopeTail`'s (its `fill` and `pre` read them).
    pub const Params = struct { scal: f64 = 1, off: i32 = 3001, sign: f64 = 1 };

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        const n = s.rows * s.n * RD;
        if (n == 0 or s.k < RD or s.k % 2 != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(n), 1, 1 }, .tg = .{ ci(@min(n, 256)), 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n), RD }, act);
        g.int("RD", RD);
        g.int("HH", s.n);
        g.dtype("T", act);
        g.dtype("OT", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        return RopeTail.fill(k, f, rng);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n * RD;
    }

    /// Output `i`'s element of the whole heads `x`.
    fn src(k: anytype, i: usize) usize {
        const D = k.shape.k;
        return i / RD * D + (D - RD) + i % RD;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, _: Sum) !void {
        for (out, 0..) |*o, i| o.* = rt(act, RopeTail.pre(k, src(k, i), true).lo);
        @memset(mag, 0);
    }

    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const n = count(k.p, k.shape);
        const a = k.scratch();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        const got = buf[0..n];
        k.outputs(kc.host.elemOf(act), got);
        if (at == .nonfinite) {
            try compute(k, buf[n .. 2 * n], buf[2 * n ..], .{});
            return classes(&k.msg, got, buf[n .. 2 * n]);
        }
        for (got, 0..) |g, i| if (qatOk(.none, &.{RopeTail.pre(k, src(k, i), false)}, &.{g}, act, &k.msg, i)) |why| return why;
        return null;
    }

    /// rope_tail over the whole heads, its last `RD` channels: the words the tail must equal.
    pub fn replaced(k: anytype) ![]const u8 {
        var sc = v41.Scope.init(k.rig.a, k.rig.stream());
        defer sc.deinit();
        var m: v41.Dsv41Model = undefined;
        m.cfg = .{};
        m.cfg.dsv4_rope_head_dim = RD;
        m.qat_off = false;
        m.rope_kernel = try v41.buildKernel("dsv41_rope_tail", &.{ "x", "periods", "off", "scal" }, &.{"out"}, v41.ROPE_TAIL_SOURCE, v41.FP8_HEADER);
        defer _ = mlx.mlx_fast_metal_kernel_free(m.rope_kernel);
        const whole = try v41.ropeQat(&sc, &m, k.args[0], k.args[1], k.args[2], @floatCast(k.p.scal), null, null);
        const D: c_int = ci(k.shape.k);
        var tail = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tail);
        try mlx.check(mlx.mlx_slice(&tail, whole, &[_]c_int{ 0, 0, D - RD }, 3, &[_]c_int{ ci(k.shape.rows), ci(k.shape.n), D }, 3, &[_]c_int{ 1, 1, 1 }, 3, k.rig.stream()));
        return k.rig.bytesOf(&.{tail});
    }
};

// ── the window row: rms norm, the tail rope at its position, the fp8 round trip ──

const KvRow = struct {
    pub const Params = struct { off: i32 = 3001 };

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.k == 0 or s.k > 1024 or s.k % 32 != 0 or s.k < RD) return null;
        var g = metal.Geometry{ .grid = .{ ci(s.k), ci(s.rows), 1 }, .tg = .{ ci(s.k), 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.k) }, act);
        g.int("RD", RD);
        g.flag("QAT", true);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const D = k.shape.k;
        const x = try k.add(act, &.{ L, D });
        const w = try k.add(act, &.{D});
        const periods = try k.add(.float32, &.{RD / 2});
        const off = try k.add(.int32, &.{});
        const eps = try k.add(.float32, &.{1});
        kit.uniform(rng, x, -3, 3);
        kit.uniform(rng, w, 0.5, 1.5);
        periodsFill(periods, 1);
        off[0] = @floatFromInt(k.p.off);
        eps[0] = EPS_RMS;
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(x, L, D),
            .stress => {
                // Rows of +-32 (an exact rsqrt, eps 0), unit weights, angle 0 (infinite periods): every stage exact.
                for (x) |*v| v.* = if (rng.boolean()) 32 else -32;
                @memset(w, 1);
                @memset(periods, inf);
                eps[0] = 0;
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.k;
    }

    /// Row `l`'s values before the round trip: intervals (`point` false) or float64 points (lo == hi).
    fn row(k: anytype, l: usize, sum: Sum, point: bool, out: []Iv) !void {
        const D = k.shape.k;
        const x = k.in(0)[l * D ..][0..D];
        const w = k.in(1);
        const sq = try k.scratch().alloc(f64, D);
        defer k.scratch().free(sq);
        for (sq, x) |*q, v| q.* = v * v;
        const tot = sum.of(sq);
        const eps = spec.roundTo(.f32, k.in(4)[0]);
        const g = if (point) 0 else spec.gamma(@floatFromInt(D), u);
        const inv = Iv{ .lo = 1 / @sqrt(tot * (1 + g) / @as(f64, @floatFromInt(D)) + eps), .hi = 1 / @sqrt(tot * (1 - g) / @as(f64, @floatFromInt(D)) + eps) };
        const iw = if (point) Iv.pt((inv.lo + inv.hi) / 2) else inv.widen(2 * u, 0);
        const r_u: f64 = if (point) 0 else u;
        var y: [1024]Iv = undefined;
        for (0..D) |d| y[d] = Iv.pt(x[d]).mul(iw).widen(r_u, 0).round(act).mul(Iv.pt(w[d])).widen(r_u, 0).round(act);
        const head = D - RD;
        for (0..D) |d| {
            if (d < head) {
                out[d] = y[d];
                continue;
            }
            const b = d - (d - head) % 2;
            const th = (@as(f64, @floatFromInt(l)) + @as(f64, @floatFromInt(k.p.off))) / spec.roundTo(.f32, k.in(2)[(d - head) / 2]);
            const even = (d - head) % 2 == 0;
            if (point) {
                const x1 = y[b].lo;
                const x2 = y[b + 1].lo;
                out[d] = Iv.pt(bf(if (even) x1 * @cos(th) - x2 * @sin(th) else x1 * @sin(th) + x2 * @cos(th)));
            } else out[d] = ropeIv(y[b], y[b + 1], th, even).round(act);
        }
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const D = k.shape.k;
        var iv: [1024]Iv = undefined;
        for (0..k.shape.rows) |l| {
            try row(k, l, sum, true, iv[0..D]);
            const o = out[l * D ..][0..D];
            for (o, iv[0..D]) |*v, r| v.* = r.lo;
            var b: usize = 0;
            while (b < D) : (b += 32) qatPoint(.fp8, o[b..][0..32]);
        }
        @memset(mag, 0);
    }

    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const D = k.shape.k;
        const n = k.shape.rows * D;
        const a = k.scratch();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        const got = buf[0..n];
        k.outputs(.bf16, got);
        if (at == .nonfinite) {
            try compute(k, buf[n .. 2 * n], buf[2 * n ..], .{});
            return classes(&k.msg, got, buf[n .. 2 * n]);
        }
        var iv: [1024]Iv = undefined;
        for (0..k.shape.rows) |l| {
            try row(k, l, .{}, false, iv[0..D]);
            var b: usize = 0;
            while (b < D) : (b += 32) if (qatOk(.fp8, iv[b..][0..32], got[l * D + b ..][0..32], act, &k.msg, l * D + b)) |why| return why;
        }
        return null;
    }
};

// ── hc_norm: the split of z, and the stream collapsed with cpre, rounded to T, rms-normed ──

/// A collapsed row `h` [D] (f32 sums of HC products, rounded to T) through MLX's rms norm with `w`: intervals of the
/// kernel's `T(w * T(T(h) * inv))`. `point`: float64 points.
pub fn collapseNorm(h: []const f64, hmag: []const f64, w: []const f64, eps: f64, point: bool, out: []Iv) void {
    const D = h.len;
    var lo: f64 = 0;
    var hi: f64 = 0;
    for (h, hmag, out) |v, m, *o| {
        o.* = if (point) Iv.pt(bf(v)) else Iv.of(v, spec.gamma(2 * HC, u) * m).round(act);
        const a = if (o.lo <= 0 and o.hi >= 0) 0 else @min(@abs(o.lo), @abs(o.hi));
        const b = @max(@abs(o.lo), @abs(o.hi));
        lo += a * a;
        hi += b * b;
    }
    const g = if (point) 0 else spec.gamma(@floatFromInt(D), u);
    const df: f64 = @floatFromInt(D);
    var inv = Iv{ .lo = 1 / @sqrt(hi * (1 + g) / df + eps), .hi = 1 / @sqrt(lo * (1 - g) / df + eps) };
    if (!point) inv = inv.widen(2 * u, 0);
    const r: f64 = if (point) 0 else u;
    for (out, w) |*o, wd| o.* = o.mul(inv).widen(r, 0).round(act).mul(Iv.pt(wd)).widen(r, 0).round(act);
}

/// The f32 split outputs' allowance: the Sinkhorn's declared roundings and the output's own.
pub fn splitOk(buf: []u8, got: []const f64, want: []const f64) ?[]const u8 {
    for (got, want, 0..) |g, w, i| {
        if (spec.classOf(g) != spec.classOf(w)) return std.fmt.bufPrint(buf, "split output {d} is {e} where float64 gives {e}", .{ i, g, w }) catch "split";
        if (spec.classOf(w) == .finite and !(@abs(g - w) <= (sinkhorn_ops + 8) * u * @abs(w) + 0x1p-140))
            return std.fmt.bufPrint(buf, "split output {d} is {e} where float64 gives {e}", .{ i, g, w }) catch "split";
    }
    return null;
}

const HcNorm = struct {
    pub const Params = struct { tg: usize = 1024 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(p.tg), ci(s.rows), 1 }, .tg = .{ ci(p.tg), 1, 1 } };
        g.out(&.{ ci(s.rows), HC }, .float32);
        g.out(&.{ ci(s.rows), HC }, .float32);
        g.out(&.{ ci(s.rows), HC, HC }, .float32);
        g.out(&.{ ci(s.rows), ci(s.n) }, act);
        g.int("TG", p.tg);
        g.int("HC", HC);
        g.int("ITERS", ITERS);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const D = k.shape.n;
        const z = try k.add(.float32, &.{ L, MIX });
        const x = try k.add(act, &.{ L, HC, D });
        const cpre = try k.add(.float32, &.{ L, HC });
        const w = try k.add(act, &.{D});
        const eps = try k.add(.float32, &.{2});
        kit.uniform(rng, z, -3, 3);
        kit.uniform(rng, x, -3, 3);
        kit.uniform(rng, cpre, 0, 1);
        kit.uniform(rng, w, 0.5, 1.5);
        eps[0] = EPS_RMS;
        eps[1] = EPS_HC;
        if (f == .nonfinite) {
            kit.plantRows(z, L, MIX);
            kit.plantRows(x, L, HC * D);
        }
    }

    /// Gate 4 reads the f32 split outputs (the first three); the T rows follow them.
    pub fn count(_: Params, s: Shape) usize {
        return s.rows * (2 * HC + HC * HC);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const L = k.shape.rows;
        var pk: [2 * HC + HC * HC]f64 = undefined;
        for (0..L) |l| {
            splitRow(k.in(0)[l * MIX ..][0..MIX], spec.roundTo(.f32, EPS_HC), sum, &pk);
            @memcpy(out[l * HC ..][0..HC], pk[0..HC]);
            @memcpy(out[L * HC + l * HC ..][0..HC], pk[HC .. 2 * HC]);
            @memcpy(out[2 * L * HC + l * HC * HC ..][0 .. HC * HC], pk[2 * HC ..]);
        }
        for (mag, out) |*m, w| m.* = @abs(w);
    }

    /// Row `l`'s collapse `h` and its terms' magnitude.
    fn collapse(k: anytype, l: usize, h: []f64, hm: []f64) void {
        const D = k.shape.n;
        for (0..D) |d| {
            h[d] = 0;
            hm[d] = 0;
            for (0..HC) |j| {
                const t = k.in(2)[l * HC + j] * k.in(1)[(l * HC + j) * D + d];
                h[d] += t;
                hm[d] += @abs(t);
            }
        }
    }

    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const L = k.shape.rows;
        const D = k.shape.n;
        const ns = count(k.p, k.shape);
        const a = k.scratch();
        const buf = try a.alloc(f64, 3 * ns + 2 * D);
        defer a.free(buf);
        for (buf[0..ns], 0..) |*g, i| g.* = outAt(k, .f32, 0, i);
        try compute(k, buf[ns .. 2 * ns], buf[2 * ns + 2 * D ..][0..ns], .{});
        if (if (at == .nonfinite) classes(&k.msg, buf[0..ns], buf[ns .. 2 * ns]) else splitOk(&k.msg, buf[0..ns], buf[ns .. 2 * ns])) |why| return why;
        const h = buf[2 * ns ..][0..D];
        const hm = buf[2 * ns + D ..][0..D];
        var iv: [8192]Iv = undefined;
        for (0..L) |l| {
            collapse(k, l, h, hm);
            collapseNorm(h, hm, k.in(3), spec.roundTo(.f32, EPS_RMS), at == .nonfinite, iv[0..D]);
            for (iv[0..D], 0..) |r, d| {
                const g = outAt(k, .bf16, 4 * ns, l * D + d);
                const ok = if (at == .nonfinite) spec.classOf(g) == spec.classOf(r.lo) else r.has(g);
                if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} h {d} is {e} outside [{e}, {e}]", .{ l, d, g, r.lo, r.hi }) catch "h";
            }
        }
        return null;
    }
};

// ── hc_mix's and hc_stage's dot: z[m] = (stream . fn[m]) * rsqrt(mean(stream^2) + eps) * scale[m] + base[m] ──

/// One row's mixes from the flattened stream `v` [N]; `allow` the bound each carries (the dot's gamma on its terms,
/// the rms's half on the product, the f32 tail).
pub fn mixRow(v: []const f64, fnw: []const f64, scale: []const f64, base: []const f64, eps: f64, z: []f64, allow: []f64) void {
    const N = v.len;
    var ss: f64 = 0;
    for (v) |x| ss += x * x;
    const r = 1 / @sqrt(ss / @as(f64, @floatFromInt(N)) + eps);
    const g = spec.gamma(@floatFromInt(N), u);
    for (z, allow, 0..) |*o, *al, m| {
        var d: f64 = 0;
        var mg: f64 = 0;
        for (v, 0..) |x, i| {
            d += x * fnw[m * N + i];
            mg += @abs(x * fnw[m * N + i]);
        }
        const y = d * r * scale[m];
        o.* = y + base[m];
        al.* = g * mg * @abs(r * scale[m]) + (g / 2 + 6 * u) * @abs(y) + 2 * u * @abs(o.*) + 0x1p-140;
    }
}

// ── hc_stage: role 0 the collapse + norm of this stage's stream, roles 1..MIX one mix each; with expand, the stream
// rebuilt from the previous block's output ──

const HcStage = struct {
    pub const Params = struct { expand: bool, tg: usize = 1024 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0) return null;
        var g = metal.Geometry{ .grid = .{ ci((MIX + 1) * p.tg), ci(s.rows), 1 }, .tg = .{ ci(p.tg), 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n) }, act);
        g.out(&.{ ci(s.rows), MIX }, .float32);
        if (p.expand) g.out(&.{ ci(s.rows), HC, ci(s.n) }, act);
        g.int("TG", p.tg);
        g.int("HC", HC);
        g.int("ND", (s.n + p.tg - 1) / p.tg);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const D = k.shape.n;
        const x = try k.add(act, &.{ L, HC, D });
        kit.uniform(rng, x, -3, 3);
        if (k.p.expand) {
            const y = try k.add(act, &.{ L, D });
            const post = try k.add(.float32, &.{ L, HC });
            const comb = try k.add(.float32, &.{ L, HC, HC });
            kit.uniform(rng, y, -3, 3);
            kit.uniform(rng, post, 0, 2);
            kit.uniform(rng, comb, 0, 0.5);
            if (f == .nonfinite) kit.plantRows(y, L, D);
        }
        const cpre = try k.add(.float32, &.{ L, HC });
        const w = try k.add(act, &.{D});
        const fnw = try k.add(.float32, &.{ MIX, HC * D });
        const scale = try k.add(.float32, &.{MIX});
        const base = try k.add(.float32, &.{MIX});
        const eps = try k.add(.float32, &.{2});
        kit.uniform(rng, cpre, 0, 1);
        kit.uniform(rng, w, 0.5, 1.5);
        kit.uniform(rng, fnw, -0.02, 0.02);
        kit.uniform(rng, scale, 0.5, 1.5);
        kit.uniform(rng, base, -0.5, 0.5);
        eps[0] = EPS_RMS;
        eps[1] = EPS_HC;
        if (f == .nonfinite and !k.p.expand) kit.plantRows(x, L, HC * D);
        if (f == .stress) {
            // The stream +-32 on every row (with expand: the identity comb over x, no post), unit norm weights, eps
            // 0, a quarter of each stream collapsed: the collapse is a multiple of 8 up to 32 and its squares sum
            // below 2^22, exact in fp32, past fp16's range. The mixes are hc_mix's stress (their own comment).
            const N = HC * D;
            eps[0] = 0;
            @memset(w, 1);
            @memset(cpre, 0.25);
            @memset(scale, 1);
            @memset(base, 0);
            for (x[0..N]) |*v| v.* = if (rng.boolean()) 32 else -32;
            for (1..L) |l| @memcpy(x[l * N ..][0..N], x[0..N]);
            if (k.p.expand) {
                @memset(at_(k, .y), 0);
                @memset(at_(k, .post), 0);
                const comb = at_(k, .comb);
                for (comb, 0..) |*v, i| v.* = if (i / HC % HC == i % HC) 1 else 0;
            }
            const h = N / 2;
            for (0..MIX) |m| for (0..h) |i| {
                const b: f64 = 0x1p10 + @as(f64, @floatFromInt(rng.intRangeAtMost(i32, 0, 8)));
                fnw[m * N + i] = b * x[i] / 32;
                fnw[m * N + h + i] = -0x1p10 * x[h + i] / 32;
            };
        }
    }

    /// Gate 4 reads the normed collapse (T) and the mixes (f32); an expanded stream follows them.
    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n + s.rows * MIX;
    }

    fn at_(k: anytype, name: enum { x, y, post, comb, cpre, w, fnw, scale, base }) []f64 {
        const e = k.p.expand;
        return k.in(switch (name) {
            .x => 0,
            .y => 1,
            .post => 2,
            .comb => 3,
            .cpre => if (e) 4 else 1,
            .w => if (e) 5 else 2,
            .fnw => if (e) 6 else 3,
            .scale => if (e) 7 else 4,
            .base => if (e) 8 else 5,
        });
    }

    /// Row `l`'s expanded stream `T(post * y + sum_j comb[j][i] x_j)` as intervals (`point`: float64 points).
    fn expandIv(k: anytype, l: usize, siv: []Iv, point: bool) void {
        const D = k.shape.n;
        const x = at_(k, .x)[l * HC * D ..][0 .. HC * D];
        const y = at_(k, .y)[l * D ..][0..D];
        const post = at_(k, .post)[l * HC ..][0..HC];
        const comb = at_(k, .comb)[l * HC * HC ..][0 .. HC * HC];
        for (0..HC) |i| for (0..D) |d| {
            var acc: f64 = post[i] * y[d];
            var m: f64 = @abs(acc);
            for (0..HC) |j| {
                acc += comb[j * HC + i] * x[j * D + d];
                m += @abs(comb[j * HC + i] * x[j * D + d]);
            }
            siv[i * D + d] = if (point) Iv.pt(bf(acc)) else Iv.of(acc, spec.gamma(HC + 1, u) * m).round(act);
        };
    }

    /// The collapse `T(w * T(T(h) * rsqrt(mean(T(h)^2) + eps)))` and the mixes, every sum as `sum` asks (gate 4).
    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const L = k.shape.rows;
        const D = k.shape.n;
        const N = HC * D;
        const a = k.scratch();
        const v = try a.alloc(f64, 2 * N + D);
        defer a.free(v);
        const t = v[N..][0..N];
        const hb = v[2 * N ..][0..D];
        const eps = spec.roundTo(.f32, k.in(if (k.p.expand) 9 else 6)[0]);
        for (0..L) |l| {
            const x = at_(k, .x)[l * N ..][0..N];
            if (k.p.expand) {
                const y = at_(k, .y)[l * D ..][0..D];
                const post = at_(k, .post)[l * HC ..][0..HC];
                const comb = at_(k, .comb)[l * HC * HC ..][0 .. HC * HC];
                for (0..HC) |i| for (0..D) |d| {
                    var terms: [HC + 1]f64 = undefined;
                    terms[0] = post[i] * y[d];
                    for (0..HC) |j| terms[j + 1] = comb[j * HC + i] * x[j * D + d];
                    v[i * D + d] = bf(sum.of(&terms));
                };
            } else @memcpy(v[0..N], x);
            const cp = at_(k, .cpre)[l * HC ..][0..HC];
            for (hb, 0..) |*o, d| {
                var terms: [HC]f64 = undefined;
                for (&terms, 0..) |*tt, j| tt.* = cp[j] * v[j * D + d];
                o.* = bf(sum.of(&terms));
            }
            for (t[0..D], hb) |*q, hv| q.* = hv * hv;
            const inv = 1 / @sqrt(sum.of(t[0..D]) / @as(f64, @floatFromInt(D)) + eps);
            for (hb, at_(k, .w), 0..) |hv, wd, d| out[l * D + d] = bf(wd * bf(hv * inv));
            for (t, v[0..N]) |*q, sv| q.* = sv * sv;
            const r = 1 / @sqrt(sum.of(t) / @as(f64, @floatFromInt(N)) + eps);
            for (0..MIX) |m| {
                for (t, v[0..N], 0..) |*q, sv, i| q.* = sv * at_(k, .fnw)[m * N + i];
                out[L * D + l * MIX + m] = sum.of(t) * r * at_(k, .scale)[m] + at_(k, .base)[m];
            }
        }
        for (mag, out) |*m, w| m.* = @abs(w);
    }

    /// The stream is checked against its intervals (expand), then the mixes and the normed collapse are judged on
    /// the stream as the kernel wrote it.
    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const L = k.shape.rows;
        const D = k.shape.n;
        const N = HC * D;
        const point = at == .nonfinite;
        const a = k.scratch();
        const v = try a.alloc(f64, N + 2 * D);
        defer a.free(v);
        const siv = try a.alloc(Iv, N + D);
        defer a.free(siv);
        var z: [MIX]f64 = undefined;
        var al: [MIX]f64 = undefined;
        const z_off = 2 * L * D;
        const x1_off = z_off + 4 * L * MIX;
        for (0..L) |l| {
            if (k.p.expand) {
                expandIv(k, l, siv[0..N], point);
                for (siv[0..N], v[0..N], 0..) |r, *sv, j| {
                    const g = outAt(k, .bf16, x1_off, l * N + j);
                    const ok = if (point) spec.classOf(g) == spec.classOf(r.lo) else r.has(g);
                    if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} stream {d} is {e} outside [{e}, {e}]", .{ l, j, g, r.lo, r.hi }) catch "x1";
                    sv.* = if (point) r.lo else g;
                }
            } else @memcpy(v[0..N], at_(k, .x)[l * N ..][0..N]);
            mixRow(v[0..N], at_(k, .fnw), at_(k, .scale), at_(k, .base), spec.roundTo(.f32, EPS_RMS), &z, &al);
            for (z, al, 0..) |w, allow, m| {
                const g = outAt(k, .f32, z_off, l * MIX + m);
                const ok = if (point) spec.classOf(g) == spec.classOf(w) else @abs(g - w) <= allow;
                if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} mix {d} is {e} where float64 gives {e}", .{ l, m, g, w }) catch "z";
            }
            const h = v[N..][0..D];
            const hm = v[N + D ..][0..D];
            const cp = at_(k, .cpre)[l * HC ..][0..HC];
            for (0..D) |d| {
                h[d] = 0;
                hm[d] = 0;
                for (0..HC) |j| {
                    h[d] += cp[j] * v[j * D + d];
                    hm[d] += @abs(cp[j] * v[j * D + d]);
                }
            }
            collapseNorm(h, hm, at_(k, .w), spec.roundTo(.f32, EPS_RMS), point, siv[N..][0..D]);
            for (siv[N..][0..D], 0..) |r, d| {
                const g = outAt(k, .bf16, 0, l * D + d);
                const ok = if (point) spec.classOf(g) == spec.classOf(r.lo) else r.has(g);
                if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} h {d} is {e} outside [{e}, {e}]", .{ l, d, g, r.lo, r.hi }) catch "h";
            }
        }
        return null;
    }
};

// ── Engram's gate: sigmoid of the signed sqrt of a normalized stream-key dot, times the value, added ──

/// MLX's row-reduce threads for a row of `n` (`mlxRowReduceThreads`).
fn rowThreads(n: usize) usize {
    if (n <= 64) return 1;
    if (n <= 512) return 32;
    if (n <= 1024) return 128;
    return @min(1024, ((n + 3) / 4 + 31) / 32 * 32);
}

const Engram = struct {
    pub const Params = struct { xdt: mlx.mlx_dtype = act };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.k == 0) return null;
        const tg = rowThreads(s.k);
        var g = metal.Geometry{ .grid = .{ ci(tg), HC, ci(s.rows) }, .tg = .{ ci(tg), 1, 1 } };
        g.out(&.{ ci(s.rows), HC, ci(s.k) }, p.xdt);
        g.dtype("OT", p.xdt);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const D = k.shape.k;
        const x = try k.add(k.p.xdt, &.{ L, HC, D });
        const kv = try k.add(act, &.{ L, (HC + 1) * D });
        const qk = try k.add(.float32, &.{ HC, D });
        const cst = try k.add(.float32, &.{3});
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, kv, -2, 2);
        kit.uniform(rng, qk, 0.5, 1.5);
        const df: f64 = @floatFromInt(D);
        cst[0] = spec.roundTo(.f32, 1 / df);
        cst[1] = EPS_RMS;
        cst[2] = spec.roundTo(.f32, 1 / @as(f64, @sqrt(@as(f32, @floatCast(df)))));
        if (f == .nonfinite) kit.plantRows(x, L * HC, D);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * HC * s.k;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const D = k.shape.k;
        const a = k.scratch();
        const t = try a.alloc(f64, 3 * D);
        defer a.free(t);
        const cst = k.in(3);
        for (0..k.shape.rows) |l| for (0..HC) |j| {
            const h = k.in(0)[(l * HC + j) * D ..][0..D];
            const key = k.in(1)[l * (HC + 1) * D + j * D ..][0..D];
            const v = k.in(1)[l * (HC + 1) * D + HC * D ..][0..D];
            var dm: f64 = 0;
            for (0..D) |c| {
                t[c] = h[c] * h[c];
                t[D + c] = key[c] * key[c];
                t[2 * D + c] = spec.roundTo(.f32, h[c] * k.in(2)[j * D + c]) * key[c];
                dm += @abs(t[2 * D + c]);
            }
            const sh = sum.of(t[0..D]);
            const sk = sum.of(t[D .. 2 * D]);
            const sd = sum.of(t[2 * D ..]);
            const dot = sd * (1 / @sqrt(sh * cst[0] + cst[1])) * (1 / @sqrt(sk * cst[0] + cst[1])) * cst[2];
            const ad = @abs(dot);
            const mg = @sqrt(if (std.math.isNan(ad) or ad > 1e-6) ad else 1e-6);
            const s = if (dot < 0) -mg else mg;
            const y = 1 / (1 + @exp(@abs(s)));
            const g = if (s < 0) y else 1 - y;
            // The gate's slope times the dot's relative error (its terms over its value, the norms' halves).
            const rel = if (@abs(sd) > 0) @min(dm / @abs(sd) + 2, 1e6) else 1e6;
            for (0..D) |c| {
                const i = (l * HC + j) * D + c;
                out[i] = h[c] + g * v[c];
                mag[i] = @abs(v[c]) * g * (1 - g) * @abs(s) * rel + @abs(g * v[c]) * 16 * u / spec.gamma(@floatFromInt(D), u);
            }
        };
    }

    /// The op chain the kernel replaces (`engramGateOps`), on the same inputs.
    pub fn replaced(k: anytype) ![]const u8 {
        var sc = v41.Scope.init(k.rig.a, k.rig.stream());
        defer sc.deinit();
        var m: v41.Dsv41Model = undefined;
        m.cfg = .{};
        m.cfg.hidden_size = @intCast(k.shape.k);
        m.cfg.dsv4_hc_mult = HC;
        m.cfg.rms_norm_eps = EPS_RMS;
        const o = try v41.engramGateOps(&sc, &m, k.args[2], k.args[0], k.args[1]);
        return k.rig.bytesOf(&.{o});
    }
};

// ── decode attention: the window rows and the selected compressed rows, online softmax with the sink ──

/// The attention case's geometry beyond the shape (rows L, heads H, selected K): the position, the cache.
const AttnGeo = struct { p0: usize = 3001, nc: usize = 700, cap: usize = WINDOW + 16, scale: f64 = 0.05 };

/// Row `t`'s key in the window + selected list of (query i), or null (a skipped pick).
fn attnKey(k: anytype, g: AttnGeo, i: usize, t: usize) ?[]const f64 {
    const pos = g.p0 + i;
    const lo = (pos + 1) -| WINDOW;
    const nw = pos - lo + 1;
    if (t < nw) return k.in(1)[((lo + t) % g.cap) * HD ..][0..HD];
    const K = k.shape.k;
    const c = k.in(3)[i * K + t - nw];
    if (c < 0 or c >= @as(f64, @floatFromInt(g.nc))) return null;
    return k.in(2)[@as(usize, @intFromFloat(c)) * HD ..][0..HD];
}

fn attnRows(g: AttnGeo, i: usize, K: usize) usize {
    const pos = g.p0 + i;
    return pos - ((pos + 1) -| WINDOW) + 1 + K;
}

/// One (query, head)'s rows t in [t0, t1): scores, the max, the weights' sum and the weighted keys (`acc` [HD]),
/// in float64 with `sum` for the sums; `err` [HD] the allowance on `acc`, `lerr` on the sum, `serr` on the max.
const Part = struct { m: f64, l: f64, lerr: f64, serr: f64 };

fn attnPart(k: anytype, g: AttnGeo, i: usize, h: usize, t0: usize, t1: usize, sum: Sum, acc: []f64, err: []f64) !Part {
    const H = k.shape.n;
    const q = k.in(0)[(i * H + h) * HD ..][0..HD];
    const a = k.scratch();
    const n = t1 -| t0;
    const buf = try a.alloc(f64, 4 * n + HD);
    defer a.free(buf);
    const s = buf[0..n];
    const ds = buf[n .. 2 * n];
    const p = buf[2 * n .. 3 * n];
    const terms = buf[3 * n .. 4 * n];
    const t = buf[4 * n ..][0..HD];
    const keys = try a.alloc(?[]const f64, n);
    defer a.free(keys);
    var m = -inf;
    var serr: f64 = 0;
    var any = false;
    for (keys, s, ds, 0..) |*key_slot, *sv, *dv, r| {
        key_slot.* = attnKey(k, g, i, t0 + r);
        const key = key_slot.* orelse {
            sv.* = -inf;
            dv.* = 0;
            continue;
        };
        var mg: f64 = 0;
        for (t, q, key) |*x, qv, kv| {
            x.* = qv * kv;
            mg += @abs(x.*);
        }
        sv.* = sum.of(t) * g.scale;
        dv.* = spec.gamma(HD, u) * mg * g.scale + 2 * u * @abs(sv.*);
        m = maxN(m, sv.*);
        serr = @max(serr, dv.*);
        any = true;
    }
    @memset(acc, 0);
    @memset(err, 0);
    if (!any) return .{ .m = -inf, .l = 0, .lerr = 0, .serr = 0 };
    // Each weight's relative error: its score's and the max's, the exp's own, and the online rescalings.
    const nf: f64 = @floatFromInt(n);
    for (p, s) |*w, sv| w.* = @exp(sv - m);
    const l = sum.of(p);
    var lerr: f64 = 0;
    const rel = try a.alloc(f64, n);
    defer a.free(rel);
    for (rel, p, s, ds, keys) |*r, w, sv, d, key| {
        r.* = if (key == null) 0 else 2 * d + 2 * serr + (@abs(sv - m) + nf + 16) * u;
        lerr += w * r.*;
    }
    for (0..HD) |d| {
        var e: f64 = 0;
        for (terms, p, keys, rel) |*x, w, key, r| {
            x.* = if (key) |kv| w * kv[d] else 0;
            e += @abs(x.*) * r;
        }
        acc[d] = sum.of(terms);
        err[d] = e;
    }
    return .{ .m = m, .l = l, .lerr = lerr, .serr = serr };
}

/// q [L, H, HD], the ring [cap, HD], the compressed rows [nc, HD], picks [L, K] (some unset, some past the cache),
/// the sink [H], meta, the scale, (periods).
fn attnFill(k: anytype, f: Fill, rng: std.Random, g: AttnGeo, with_periods: bool) !void {
    const L = k.shape.rows;
    const H = k.shape.n;
    const K = k.shape.k;
    const q = try k.add(act, &.{ L, H, HD });
    const ring = try k.add(act, &.{ g.cap, HD });
    const comp = try k.add(act, if (K > 0) &.{ g.nc, HD } else &.{ g.cap, HD });
    const topk = try k.add(.int32, if (K > 0) &.{ L, K } else &.{6});
    const sink = try k.add(.float32, &.{H});
    const meta = try k.add(.int32, &.{6});
    const scal = try k.add(.float32, &.{1});
    kit.uniform(rng, q, -2, 2);
    kit.uniform(rng, ring, -2, 2);
    kit.uniform(rng, comp, -2, 2);
    kit.uniform(rng, sink, -3, 3);
    @memcpy(meta, &[_]f64{ @floatFromInt(g.p0), @floatFromInt(L), @floatFromInt(K), WINDOW, @floatFromInt(g.cap), if (K > 0) @floatFromInt(g.nc) else 0 });
    if (K > 0) {
        for (topk, 0..) |*v, i| v.* = if (i % 37 == 5) -1 else if (i % 41 == 7) @floatFromInt(g.nc + 200) else @floatFromInt((i * 13) % g.nc);
    } else @memcpy(topk, meta);
    scal[0] = g.scale;
    if (with_periods) periodsFill(try k.add(.float32, &.{RD / 2}), -1);
    if (f == .nonfinite) kit.plantRows(q, L * H, HD);
}

/// The output rounding and (rd) the folded inverse rope: intervals of `OT(unrope(OT(o)))` for one head.
fn unropeIv(o: []const Iv, rd: usize, pos: usize, periods: []const f64, out: []Iv, point: bool) void {
    for (o, out) |r, *x| x.* = if (point) Iv.pt(bf(r.lo)) else r.round(act);
    if (rd == 0) return;
    const head = HD - rd;
    var tmp: [HD]Iv = undefined;
    @memcpy(&tmp, out[0..HD]);
    for (head..HD) |d| {
        const even = (d - head) % 2 == 0;
        const b = d - (d - head) % 2;
        const th = @as(f64, @floatFromInt(pos)) * spec.roundTo(.f32, 1 / spec.roundTo(.f32, periods[(d - head) / 2]));
        if (point) {
            const x1 = tmp[b].lo;
            const x2 = tmp[b + 1].lo;
            out[d] = Iv.pt(bf(if (even) x1 * @cos(th) - x2 * @sin(th) else x1 * @sin(th) + x2 * @cos(th)));
        } else out[d] = ropeIv(tmp[b], tmp[b + 1], th, even).round(act);
    }
}

const AttnDec = struct {
    pub const Params = struct { rd: usize, g: AttnGeo = .{} };
    const NSG = 8;

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(s.n * NSG * 32), ci(s.rows), 1 }, .tg = .{ NSG * 32, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n), HD }, act);
        g.int("D", HD);
        g.int("NSG", NSG);
        g.int("S", 1);
        g.dtype("KT", act);
        g.dtype("OT", act);
        g.int("RD", p.rd);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        try attnFill(k, f, rng, k.p.g, true);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n * HD;
    }

    /// Head (i, h)'s output intervals (`point`: float64 points).
    fn head(k: anytype, i: usize, h: usize, sum: Sum, point: bool, out: []Iv) !void {
        const g = k.p.g;
        var acc: [HD]f64 = undefined;
        var err: [HD]f64 = undefined;
        const pr = try attnPart(k, g, i, h, 0, attnRows(g, i, k.shape.k), sum, &acc, &err);
        const sk = k.in(4)[h];
        const M = maxN(pr.m, sk);
        const w = @exp(pr.m - M);
        const total = @exp(sk - M) + pr.l * w;
        var o: [HD]Iv = undefined;
        for (&o, acc, err) |*x, av, e| {
            const v = av * w / total;
            x.* = if (point) Iv.pt(v) else Iv.of(v, (e * w + @abs(v) * pr.lerr * w) / total + 8 * u * @abs(v));
        }
        unropeIv(&o, k.p.rd, g.p0 + i, k.in(7), out, point);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const H = k.shape.n;
        var iv: [HD]Iv = undefined;
        for (0..k.shape.rows) |i| for (0..H) |h| {
            try head(k, i, h, sum, true, &iv);
            for (iv, 0..) |r, d| out[(i * H + h) * HD + d] = r.lo;
        };
        @memset(mag, 0);
    }

    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const H = k.shape.n;
        var iv: [HD]Iv = undefined;
        for (0..k.shape.rows) |i| for (0..H) |h| {
            try head(k, i, h, .{}, at == .nonfinite, &iv);
            for (iv, 0..) |r, d| {
                const idx = (i * H + h) * HD + d;
                const got = outAt(k, .bf16, 0, idx);
                const ok = if (at == .nonfinite) spec.classOf(got) == spec.classOf(r.lo) else r.has(got);
                if (!ok) return std.fmt.bufPrint(&k.msg, "query {d} head {d} channel {d} is {e} outside [{e}, {e}]", .{ i, h, d, got, r.lo, r.hi }) catch "attn";
            }
        };
        return null;
    }
};

pub const AttnSplit = struct {
    pub const Params = struct { s: usize, g: AttnGeo = .{} };
    const NSG = 8;

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0 or p.s < 2) return null;
        var g = metal.Geometry{ .grid = .{ ci(s.n * NSG * 32), ci(s.rows), ci(p.s) }, .tg = .{ NSG * 32, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n), ci(p.s), HD }, .float32);
        g.out(&.{ ci(s.rows), ci(s.n), ci(p.s) }, .float32);
        g.out(&.{ ci(s.rows), ci(s.n), ci(p.s) }, .float32);
        g.int("D", HD);
        g.int("NSG", NSG);
        g.int("S", p.s);
        g.dtype("KT", act);
        g.dtype("OT", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        try attnFill(k, f, rng, k.p.g, false);
    }

    pub fn count(p: Params, s: Shape) usize {
        return s.rows * s.n * p.s * (HD + 2);
    }

    fn bounds(g: AttnGeo, i: usize, K: usize, S: usize, sp: usize) [2]usize {
        const n = attnRows(g, i, K);
        const chunk = (n + S - 1) / S;
        return .{ sp * chunk, @min(n, sp * chunk + chunk) };
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const S = k.p.s;
        var acc: [HD]f64 = undefined;
        var err: [HD]f64 = undefined;
        const nm = L * H * S * HD;
        for (0..L) |i| for (0..H) |h| for (0..S) |sp| {
            const b = bounds(k.p.g, i, k.shape.k, S, sp);
            const pr = try attnPart(k, k.p.g, i, h, b[0], @max(b[0], b[1]), sum, &acc, &err);
            const base = (i * H + h) * S + sp;
            @memcpy(out[base * HD ..][0..HD], &acc);
            out[nm + base] = pr.m;
            out[nm + L * H * S + base] = pr.l;
        };
        @memset(mag, 0);
    }

    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const L = k.shape.rows;
        const H = k.shape.n;
        const S = k.p.s;
        const nm = L * H * S * HD;
        var acc: [HD]f64 = undefined;
        var err: [HD]f64 = undefined;
        for (0..L) |i| for (0..H) |h| for (0..S) |sp| {
            const b = bounds(k.p.g, i, k.shape.k, S, sp);
            const pr = try attnPart(k, k.p.g, i, h, b[0], @max(b[0], b[1]), .{}, &acc, &err);
            const base = (i * H + h) * S + sp;
            const gm = outAt(k, .f32, 0, nm + base);
            const gl = outAt(k, .f32, 0, nm + L * H * S + base);
            const nf = at == .nonfinite;
            // The stated rule: a split whose weights went NaN reports max -inf (or its +inf) and sum 0, Metal's max
            // and `den > 0` dropping the NaN, and carries it in its partial output, which the merge propagates.
            if (nf and std.math.isNan(pr.l) and (gl == 0 or std.math.isNan(gl)) and (std.math.isNan(pr.m) or spec.classOf(gm) == spec.classOf(pr.m) or gm == -inf)) {
                for (0..HD) |d| if (!std.math.isNan(outAt(k, .f32, 0, base * HD + d)))
                    return std.fmt.bufPrint(&k.msg, "split ({d}, {d}, {d}) dropped its NaN weights without a NaN partial at {d}", .{ i, h, sp, d }) catch "nan";
                continue;
            }
            // The split's own max stands in its partials: its error moves them together, which merge cancels.
            if (if (nf) spec.classOf(gm) != spec.classOf(pr.m) else !(gm == pr.m or @abs(gm - pr.m) <= pr.serr))
                return std.fmt.bufPrint(&k.msg, "split ({d}, {d}, {d}) max is {e} where float64 gives {e}", .{ i, h, sp, gm, pr.m }) catch "max";
            const shift = if (std.math.isFinite(gm) and std.math.isFinite(pr.m)) @exp(pr.m - gm) else 1;
            if (if (nf) spec.classOf(gl) != spec.classOf(pr.l) else !(@abs(gl - pr.l * shift) <= pr.lerr * shift + 0x1p-140))
                return std.fmt.bufPrint(&k.msg, "split ({d}, {d}, {d}) sum is {e} where float64 gives {e}", .{ i, h, sp, gl, pr.l * shift }) catch "sum";
            for (acc, err, 0..) |av, e, d| {
                const g = outAt(k, .f32, 0, base * HD + d);
                if (if (nf) spec.classOf(g) != spec.classOf(av) else !(@abs(g - av * shift) <= e * shift + 0x1p-140))
                    return std.fmt.bufPrint(&k.msg, "split ({d}, {d}, {d}) channel {d} is {e} where float64 gives {e}", .{ i, h, sp, d, g, av * shift }) catch "acc";
            }
        };
        return null;
    }
};

const AttnMerge = struct {
    pub const Params = struct { rd: usize, s: usize, p0: usize = 3001 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0) return null;
        var g = metal.Geometry{ .grid = .{ HD, ci(s.n), ci(s.rows) }, .tg = .{ HD, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n), HD }, act);
        g.int("D", HD);
        g.int("S", p.s);
        g.dtype("OT", act);
        g.int("RD", p.rd);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const S = k.p.s;
        const pacc = try k.add(.float32, &.{ L, H, S, HD });
        const pm = try k.add(.float32, &.{ L, H, S });
        const pl = try k.add(.float32, &.{ L, H, S });
        const sink = try k.add(.float32, &.{H});
        const meta = try k.add(.int32, &.{6});
        const periods = try k.add(.float32, &.{RD / 2});
        kit.uniform(rng, pm, -2, 4);
        kit.uniform(rng, pl, 0.5, 20);
        kit.uniform(rng, pacc, -5, 5);
        kit.uniform(rng, sink, -3, 3);
        for (pm, pl, 0..) |*m, *l, j| {
            // An empty split (no rows of its own) every so often.
            if (j % 11 == 3) {
                m.* = -inf;
                l.* = 0;
                @memset(pacc[j * HD ..][0..HD], 0);
            } else for (pacc[j * HD ..][0..HD]) |*v| {
                v.* *= l.*;
            }
        }
        meta[0] = @floatFromInt(k.p.p0);
        periodsFill(periods, -1);
        if (f == .nonfinite) kit.plantRows(pacc, L * H * S, HD);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n * HD;
    }

    fn head(k: anytype, i: usize, h: usize, point: bool, out: []Iv) void {
        const H = k.shape.n;
        const S = k.p.s;
        const base = (i * H + h) * S;
        const sk = k.in(3)[h];
        var M = sk;
        for (0..S) |j| M = maxN(M, k.in(1)[base + j]);
        var total = @exp(sk - M);
        var o: [HD]f64 = @splat(0);
        var om: [HD]f64 = @splat(0);
        var spread: f64 = @abs(sk - M);
        for (0..S) |j| {
            const l = k.in(2)[base + j];
            const w = if (l > 0) @exp(k.in(1)[base + j] - M) else 0;
            if (l > 0) spread = @max(spread, @abs(k.in(1)[base + j] - M));
            total += l * w;
            for (0..HD) |d| {
                o[d] += k.in(0)[(base + j) * HD + d] * w;
                om[d] += @abs(k.in(0)[(base + j) * HD + d] * w);
            }
        }
        const rel = (@as(f64, @floatFromInt(S)) + 8 + spread) * u;
        var iv: [HD]Iv = undefined;
        for (&iv, o, om) |*x, v, m| {
            const y = v / total;
            x.* = if (point) Iv.pt(y) else Iv.of(y, m / total * rel + @abs(y) * rel);
        }
        unropeIv(&iv, k.p.rd, k.p.p0 + i, k.in(5), out, point);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, _: Sum) !void {
        const H = k.shape.n;
        var iv: [HD]Iv = undefined;
        for (0..k.shape.rows) |i| for (0..H) |h| {
            head(k, i, h, true, &iv);
            for (iv, 0..) |r, d| out[(i * H + h) * HD + d] = r.lo;
        };
        @memset(mag, 0);
    }

    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const H = k.shape.n;
        var iv: [HD]Iv = undefined;
        for (0..k.shape.rows) |i| for (0..H) |h| {
            head(k, i, h, at == .nonfinite, &iv);
            for (iv, 0..) |r, d| {
                const got = outAt(k, .bf16, 0, (i * H + h) * HD + d);
                const ok = if (at == .nonfinite) spec.classOf(got) == spec.classOf(r.lo) else r.has(got);
                if (!ok) return std.fmt.bufPrint(&k.msg, "query {d} head {d} channel {d} is {e} outside [{e}, {e}]", .{ i, h, d, got, r.lo, r.hi }) catch "merge";
            }
        };
        return null;
    }
};

// ── the wide span's softmax: window scores under the band, selected scores where picked, the sink ──

const AttnSoftmax = struct {
    pub const Params = struct { wp: usize = 127, scal: f64 = 1.4 };

    fn nj(wk: usize, K: usize) usize {
        return (wk + K + 127) / 128 * 4;
    }

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        const L = s.rows;
        const H = s.n;
        const K = s.k;
        const wk = p.wp + L;
        if (L == 0 or H == 0 or wk + K > 1024) return null;
        var g = metal.Geometry{ .grid = .{ ci(H * 32), ci(L), 1 }, .tg = .{ ci(@min(H * 32, 128)), 1, 1 } };
        g.out(&.{ ci(L * H), ci(wk) }, .float16);
        g.out(&.{ ci(if (K > 0) L else 1), ci(if (K > 0) H else 1), ci(@max(K, 1)) }, .float16);
        if (K == 0) g.unread = 0b10;
        g.int("NJ", nj(wk, K));
        g.dtype("OT", .float16);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const K = k.shape.k;
        const wk = k.p.wp + L;
        const lw = try k.add(.float16, &.{ L * H, wk });
        const lc = try k.add(.float16, if (K > 0) &.{ L, H, K } else &.{ L * H, wk });
        const topk = try k.add(.int32, if (K > 0) &.{ L, K } else &.{4});
        const sink = try k.add(.float32, &.{ 1, H, 1 });
        const meta = try k.add(.int32, &.{4});
        const scal = try k.add(.float32, &.{1});
        kit.uniform(rng, lw, -4, 4);
        kit.uniform(rng, lc, -4, 4);
        kit.uniform(rng, sink, -3, 3);
        @memcpy(meta, &[_]f64{ @floatFromInt(k.p.wp), WINDOW, @floatFromInt(wk), @floatFromInt(K) });
        if (K > 0) {
            for (topk, 0..) |*v, i| v.* = if (i % 37 == 5) -1 else @floatFromInt(i % 400);
        } else @memcpy(topk, meta);
        scal[0] = k.p.scal;
        if (f == .nonfinite) kit.plantRows(lw, L * H, wk);
    }

    pub fn count(p: Params, s: Shape) usize {
        return s.rows * s.n * (p.wp + s.rows + s.k);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const K = k.shape.k;
        const wk = k.p.wp + L;
        const N = wk + K;
        const a = k.scratch();
        const sv = try a.alloc(f64, N);
        defer a.free(sv);
        const scale = spec.roundTo(.f32, k.p.scal);
        const pc0 = L * H * wk;
        for (0..L) |l| for (0..H) |h| {
            const rw = l * H + h;
            var mx = -inf;
            for (sv, 0..) |*v, c| {
                v.* = -inf;
                if (c < wk) {
                    if (c <= k.p.wp + l and c + WINDOW > k.p.wp + l) v.* = scale * k.in(0)[rw * wk + c];
                } else if (k.in(2)[l * K + c - wk] >= 0) v.* = scale * k.in(1)[rw * K + c - wk];
                mx = maxN(mx, v.*);
            }
            const M = maxN(mx, k.in(3)[h]);
            for (sv) |*v| v.* = @exp(v.* - M);
            const den = sum.of(sv) + @exp(k.in(3)[h] - M);
            for (sv, 0..) |v, c| {
                const i = if (c < wk) rw * wk + c else pc0 + rw * K + c - wk;
                out[i] = v / den;
                mag[i] = @abs(v / den);
            }
        };
    }
};

// ── the prompt-width core on the matrix units (`attnGather`): QK, the masked softmax, PV ──

/// The keys every gather kernel reads: row l's window (key c < W is window row wp + l - W + 1 + c, absent outside
/// [0, wp + L)) then its picks (absent below 0); window [wp + L, HD], compressed rows [nc, HD], picks [L, K]
/// (a 9-int `meta` stands in at K = 0), meta (wp, W, wp + L, K, H, HD, p0, L, o_groups).
const Gathered = struct {
    pub const Params = struct { wp: usize = 100, nc: usize = 700, p0: usize = 40, grouped: bool = false, og: usize = 8 };

    fn ok(s: Shape) bool {
        return s.rows > 0 and s.n == HEADS and s.k % 64 == 0 and WINDOW + s.k <= 1024;
    }

    fn addKeys(k: anytype, rng: std.Random) !void {
        const L = k.shape.rows;
        const K = k.shape.k;
        const kw = try k.add(act, &.{ k.p.wp + L, HD });
        const comp = try k.add(act, &.{ k.p.nc, HD });
        const picks = try k.add(.int32, if (K > 0) &.{ L, K } else &.{9});
        const meta = try k.add(.int32, &.{9});
        kit.uniform(rng, kw, -2, 2);
        kit.uniform(rng, comp, -2, 2);
        @memcpy(meta, &[_]f64{ @floatFromInt(k.p.wp), WINDOW, @floatFromInt(k.p.wp + L), @floatFromInt(K), HEADS, HD, @floatFromInt(k.p.p0), @floatFromInt(L), @floatFromInt(k.p.og) });
        if (K > 0) {
            for (picks, 0..) |*v, i| v.* = if (i % 37 == 5) -1 else @floatFromInt((i * 13) % k.p.nc);
        } else @memcpy(picks, meta);
    }

    /// Key `c` of row `l` (inputs from `first`: window, compressed rows, picks), or null where it is absent.
    fn key(k: anytype, first: usize, l: usize, c: usize) ?[]const f64 {
        const K = k.shape.k;
        if (c < WINDOW) {
            const r = @as(i64, @intCast(k.p.wp + l + c)) - (WINDOW - 1);
            if (r < 0 or r >= k.p.wp + k.shape.rows) return null;
            return k.in(first)[@as(usize, @intCast(r)) * HD ..][0..HD];
        }
        const r = k.in(first + 2)[l * K + c - WINDOW];
        if (r < 0) return null;
        return k.in(first + 1)[@as(usize, @intFromFloat(r)) * HD ..][0..HD];
    }
};

const AttnQk = struct {
    pub const Params = Gathered.Params;

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (!Gathered.ok(s)) return null;
        const NK = WINDOW + s.k;
        var g = metal.Geometry{ .grid = .{ ci(NK / 64 * 128), ci(s.rows), 1 }, .tg = .{ 128, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n), ci(NK) }, .float32);
        g.int("HS", s.n / 4);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const q = try k.add(act, &.{ k.shape.rows, k.shape.n, HD });
        kit.uniform(rng, q, -2, 2);
        try Gathered.addKeys(k, rng);
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(q, k.shape.rows * k.shape.n, HD),
            // Every query row one stress vector, every key row a stress row against it.
            .stress => {
                const kw = k.in(1);
                const comp = k.in(2);
                try stressDot(rng, q[0..HD], kw, kw.len / HD, HD, 64, 7, k.scratch());
                for (0..comp.len / HD) |r| @memcpy(comp[r * HD ..][0..HD], kw[(r % (kw.len / HD)) * HD ..][0..HD]);
                for (1..q.len / HD) |r| @memcpy(q[r * HD ..][0..HD], q[0..HD]);
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n * (WINDOW + s.k);
    }

    /// An absent key stages as zeros: its score is q . 0 (0, or NaN where q is not finite).
    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const NK = WINDOW + k.shape.k;
        const t = try k.scratch().alloc(f64, HD);
        defer k.scratch().free(t);
        for (0..L) |l| for (0..H) |h| for (0..NK) |c| {
            const q = k.in(0)[(l * H + h) * HD ..][0..HD];
            const row = Gathered.key(k, 1, l, c);
            var m: f64 = 0;
            for (t, q, 0..) |*v, x, d| {
                v.* = x * (if (row) |r| r[d] else 0);
                m += @abs(v.*);
            }
            const i = (l * H + h) * NK + c;
            out[i] = sum.of(t);
            mag[i] = m;
        };
    }
};

/// kernels/attn_qk_t.metal: attn_qk with the heads in two parts, `q` [L, H, D] for every 64-deep step but the last
/// and `qt` [L, H, 64] (the roped tail, rope_tail_only) for it.
const AttnQkT = struct {
    pub const Params = Gathered.Params;
    const TAIL = 64;

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        return AttnQk.geometry(p, s);
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        try AttnQk.fill(k, f, rng);
        const qt = try k.add(act, &.{ k.shape.rows, k.shape.n, TAIL });
        // The tail is the heads' own last channels (as rope_tail_only writes the roped ones), so the stress rows stay
        // stress rows.
        const q = k.in(0);
        for (0..k.shape.rows * k.shape.n) |r| @memcpy(qt[r * TAIL ..][0..TAIL], q[r * HD + HD - TAIL ..][0..TAIL]);
        if (f == .random) kit.uniform(rng, qt, -2, 2);
    }

    pub fn count(p: Params, s: Shape) usize {
        return AttnQk.count(p, s);
    }

    /// The heads the scores read: `q`'s channels with `qt` written over the last 64.
    fn tailInto(k: anytype, q: []f64) void {
        for (0..k.shape.rows * k.shape.n) |r| @memcpy(q[r * HD + HD - TAIL ..][0..TAIL], k.in(5)[r * TAIL ..][0..TAIL]);
    }

    /// attn_qk's formula over those heads (`q` restored after).
    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const q = k.in(0);
        const saved = try k.scratch().dupe(f64, q);
        defer k.scratch().free(saved);
        defer @memcpy(q, saved);
        tailInto(k, q);
        try AttnQk.compute(k, out, mag, sum);
    }

    /// attn_qk over the whole roped heads on its own rig: the scores the split read must equal.
    pub fn replaced(k: anytype) ![]const u8 {
        const a = k.scratch();
        const qr = try a.dupe(f64, k.in(0));
        defer a.free(qr);
        tailInto(k, qr);
        var r2 = metal.Rig.init(a);
        defer r2.deinit();
        var args: [5]A = undefined;
        args[0] = try r2.input(act, k.ins[0].dims[0..3], qr);
        for (args[1..], k.ins[1..5]) |*x, in| x.* = try r2.input(in.dt, in.dims[0..in.nd], in.v);
        try r2.launch(&qk_l, AttnQk.geometry(.{}, k.shape).?, &args);
        return k.arena.allocator().dupe(u8, r2.outputBytes());
    }
};

const AttnSm = struct {
    pub const Params = struct { wp: usize = 100, scal: f64 = 1.4 };

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (!Gathered.ok(s)) return null;
        const NK = WINDOW + s.k;
        var g = metal.Geometry{ .grid = .{ ci(s.n * 32), ci(s.rows), 1 }, .tg = .{ 128, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n), ci(NK) }, .float16);
        g.int("NJ", (NK + 31) / 32);
        g.dtype("PT", .float16);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const K = k.shape.k;
        const NK = WINDOW + K;
        const s = try k.add(.float32, &.{ L, H, NK });
        const picks = try k.add(.int32, if (K > 0) &.{ L, K } else &.{9});
        const sink = try k.add(.float32, &.{ 1, H, 1 });
        const meta = try k.add(.int32, &.{9});
        const scal = try k.add(.float32, &.{1});
        kit.uniform(rng, s, -4, 4);
        kit.uniform(rng, sink, -3, 3);
        @memcpy(meta, &[_]f64{ @floatFromInt(k.p.wp), WINDOW, @floatFromInt(k.p.wp + L), @floatFromInt(K), @floatFromInt(H), HD, 0, @floatFromInt(L), 8 });
        if (K > 0) {
            for (picks, 0..) |*v, i| v.* = if (i % 37 == 5) -1 else @floatFromInt(i % 400);
        } else @memcpy(picks, meta);
        scal[0] = k.p.scal;
        if (f == .nonfinite) kit.plantRows(s, L * H, NK);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n * (WINDOW + s.k);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const K = k.shape.k;
        const NK = WINDOW + K;
        const sv = try k.scratch().alloc(f64, NK);
        defer k.scratch().free(sv);
        const scale = spec.roundTo(.f32, k.p.scal);
        for (0..L) |l| for (0..H) |h| {
            const rw = (l * H + h) * NK;
            var mx = -inf;
            for (sv, 0..) |*v, c| {
                const live = if (c < WINDOW) k.p.wp + l + c >= WINDOW - 1 else k.in(1)[l * K + c - WINDOW] >= 0;
                v.* = if (live) scale * k.in(0)[rw + c] else -inf;
                mx = maxN(mx, v.*);
            }
            const M = maxN(mx, k.in(2)[h]);
            for (sv) |*v| v.* = @exp(v.* - M);
            const den = sum.of(sv) + @exp(k.in(2)[h] - M);
            for (sv, 0..) |v, c| {
                out[rw + c] = v / den;
                mag[rw + c] = @abs(v / den);
            }
        };
    }
};

const AttnPv = struct {
    pub const Params = Gathered.Params;

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (!Gathered.ok(s) or (p.grouped and s.n % p.og != 0)) return null;
        var g = metal.Geometry{ .grid = .{ HD / 128 * 128, ci(s.rows), 1 }, .tg = .{ 128, 1, 1 } };
        if (p.grouped) g.out(&.{ ci(p.og), ci(s.rows), ci(s.n / p.og * HD) }, act) else g.out(&.{ ci(s.rows), ci(s.n), HD }, act);
        g.int("HS", s.n / 4);
        g.int("GROUPED", @intFromBool(p.grouped));
        g.int("RD", RD);
        g.dtype("T", act);
        g.dtype("PT", .float16);
        g.dtype("OT", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const NK = WINDOW + k.shape.k;
        const p = try k.add(.float16, &.{ L, H, NK });
        kit.uniform(rng, p, 0, 2.0 / @as(f64, @floatFromInt(NK)));
        try Gathered.addKeys(k, rng);
        const periods = try k.add(.float32, &.{RD / 2});
        periodsFill(periods, -1);
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(p, L * H, NK),
            // Row 0's live keys (an even count of them) carry one stress vector, column d of their rows a stress row
            // against it; absent keys get probability 0. Every probability row copies row 0's.
            .stress => {
                const a = k.scratch();
                const rows = try a.alloc(?[]f64, NK);
                defer a.free(rows);
                const live = try a.alloc(usize, NK);
                defer a.free(live);
                var n: usize = 0;
                for (rows, 0..) |*row, c| {
                    row.* = if (c < WINDOW) blk: {
                        const r = @as(i64, @intCast(k.p.wp + c)) - (WINDOW - 1);
                        break :blk if (r < 0) null else k.in(1)[@as(usize, @intCast(r)) * HD ..][0..HD];
                    } else blk: {
                        const r = k.in(3)[c - WINDOW];
                        break :blk if (r < 0) null else k.in(2)[@as(usize, @intFromFloat(r)) * HD ..][0..HD];
                    };
                    if (row.* != null) {
                        live[n] = c;
                        n += 1;
                    }
                }
                n -= n % 2;
                const x = try a.alloc(f64, n);
                defer a.free(x);
                const w = try a.alloc(f64, HD * n);
                defer a.free(w);
                // f16 probabilities hold integers to 2048, bf16 values to 256: every sum stays exact in fp32.
                try stressDot(rng, x, w, HD, n, 256, 31, a);
                @memset(p[0..NK], 0);
                for (live[0..n], 0..) |c, j| {
                    p[c] = x[j];
                    for (rows[c].?, 0..) |*v, d| v.* = w[d * n + j];
                }
                for (1..L * H) |r| @memcpy(p[r * NK ..][0..NK], p[0..NK]);
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n * HD;
    }

    /// An absent key stages as zeros. Grouped: the last RD channels rotate at position p0 + l (adjacent pairs, the
    /// inverse periods) and head h lands at [h / hg, l, (h % hg) * HD].
    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const NK = WINDOW + k.shape.k;
        const a = k.scratch();
        const t = try a.alloc(f64, NK);
        defer a.free(t);
        const row = try a.alloc(f64, 2 * HD);
        defer a.free(row);
        const hg = H / k.p.og;
        for (0..L) |l| for (0..H) |h| {
            const pr = k.in(0)[(l * H + h) * NK ..][0..NK];
            for (0..HD) |d| {
                var m: f64 = 0;
                for (t, pr, 0..) |*v, x, c| {
                    v.* = x * (if (Gathered.key(k, 1, l, c)) |r| r[d] else 0);
                    m += @abs(v.*);
                }
                row[d] = sum.of(t);
                row[HD + d] = m;
            }
            if (k.p.grouped) {
                const pos: f64 = @floatFromInt(k.p.p0 + l);
                var d: usize = HD - RD;
                while (d < HD) : (d += 2) {
                    const th = pos / spec.roundTo(.f32, k.in(5)[(d - (HD - RD)) / 2]);
                    const x0 = row[d];
                    const x1 = row[d + 1];
                    row[d] = x0 * @cos(th) - x1 * @sin(th);
                    row[d + 1] = x0 * @sin(th) + x1 * @cos(th);
                    const mm = row[HD + d] + row[HD + d + 1];
                    row[HD + d] = mm;
                    row[HD + d + 1] = mm;
                }
            }
            const base = if (k.p.grouped) ((h / hg) * L + l) * hg * HD + (h % hg) * HD else (l * H + h) * HD;
            @memcpy(out[base..][0..HD], row[0..HD]);
            @memcpy(mag[base..][0..HD], row[HD..][0..HD]);
        };
    }
};

// ── the decode MoE: one token's routed + shared gate+up (`moe_gu`), its down + shared + hc expand (`moe_dn`) ──

/// `codes` packed densely (element i at bit i * bits of the little-endian stream, as `mx.quantize` packs), into
/// 32-bit words.
fn packDense(codes: []const u32, bits: u32, words: []f64) void {
    var acc: u64 = 0;
    var have: u6 = 0;
    var w: usize = 0;
    for (codes) |c| {
        acc |= @as(u64, c) << have;
        have += @intCast(bits);
        if (have >= 32) {
            words[w] = @floatFromInt(acc & 0xffffffff);
            w += 1;
            acc >>= 32;
            have -= 32;
        }
    }
}

fn codeAt(words: []const f64, i: usize, bits: u32) f64 {
    const bit = i * bits;
    const lo: u64 = @intFromFloat(words[bit / 32]);
    const hi: u64 = if (bit / 32 + 1 < words.len) @intFromFloat(words[bit / 32 + 1]) else 0;
    const both = lo | (hi << 32);
    return @floatFromInt((both >> @intCast(bit % 32)) & ((@as(u64, 1) << @intCast(bits)) - 1));
}

/// One affine bank's input slots: the packed codes, the scales, the biases ([rows, K] at `bits` / `gs`).
pub const Bank = struct {
    w: []f64,
    s: []f64,
    b: []f64,
    bits: u32,
    gs: usize,
    k: usize,

    pub fn add(c: anytype, rows: []const usize, bits: u32, gs: usize, K: usize) !Bank {
        var wd: [3]usize = undefined;
        @memcpy(wd[0..rows.len], rows);
        var sd = wd;
        wd[rows.len] = K * bits / 32;
        sd[rows.len] = K / gs;
        return .{
            .w = try c.add(.uint32, wd[0 .. rows.len + 1]),
            .s = try c.add(act, sd[0 .. rows.len + 1]),
            .b = try c.add(act, sd[0 .. rows.len + 1]),
            .bits = bits,
            .gs = gs,
            .k = K,
        };
    }

    fn rowCount(self: Bank) usize {
        return self.s.len / (self.k / self.gs);
    }

    /// Random codes, scales around 0.04 / levels, biases centring them.
    pub fn random(self: Bank, rng: std.Random, scratch: std.mem.Allocator) !void {
        const levels: u32 = @as(u32, 1) << @intCast(self.bits);
        const codes = try scratch.alloc(u32, self.k);
        defer scratch.free(codes);
        const kw = self.k * self.bits / 32;
        for (0..self.rowCount()) |r| {
            for (codes) |*q| q.* = rng.uintLessThan(u32, levels);
            packDense(codes, self.bits, self.w[r * kw ..][0..kw]);
        }
        for (self.s, self.b) |*sc, *bi| {
            sc.* = (0.5 + rng.float(f64)) * 0.04 / @as(f64, @floatFromInt(levels));
            bi.* = -sc.* * @as(f64, @floatFromInt(levels / 2));
        }
    }

    /// Integer weights c - levels/2 (scale 1, bias -levels/2), positive c on the first half and -c on the
    /// shuffled second (`perm`); c 1 on the first 8 entries, at most 31 elsewhere.
    pub fn mirrored(self: Bank, rng: std.Random, perm: []const usize, scratch: std.mem.Allocator) !void {
        const levels: u32 = @as(u32, 1) << @intCast(self.bits);
        const mid: i32 = @intCast(levels / 2);
        const top: i32 = @min(mid - 1, 31);
        const codes = try scratch.alloc(u32, self.k);
        defer scratch.free(codes);
        const h = self.k / 2;
        const kw = self.k * self.bits / 32;
        @memset(self.s, 1);
        @memset(self.b, -@as(f64, @floatFromInt(mid)));
        for (0..self.rowCount()) |r| {
            for (0..h) |i| {
                const c = if (i < 8) 1 else rng.intRangeAtMost(i32, 1, top);
                codes[i] = @intCast(mid + c);
                codes[h + perm[i]] = @intCast(mid - c);
            }
            packDense(codes, self.bits, self.w[r * kw ..][0..kw]);
        }
    }

    /// Row `r` against `x` [K]: the sum (`sum`'s accumulator), and its terms' magnitude (scale and bias apart).
    pub fn dot(self: Bank, r: usize, x: []const f64, sum: Sum, t: []f64) [2]f64 {
        const kw = self.k * self.bits / 32;
        const kg = self.k / self.gs;
        var mag: f64 = 0;
        for (0..self.k) |i| {
            const q = codeAt(self.w[r * kw ..][0..kw], i, self.bits);
            const s = self.s[r * kg + i / self.gs];
            const b = self.b[r * kg + i / self.gs];
            t[i] = x[i] * (q * s + b);
            mag += @abs(x[i]) * (@abs(q * s) + @abs(b));
        }
        return .{ sum.of(t[0..self.k]), mag };
    }

    /// Row `r`'s first group with every code at least 1 and scale `v`: an infinite scale then gives the same
    /// infinity whether a lane scales its pack's sum (the kernel, MLX's qmv) or each dequantized weight.
    pub fn plant(self: Bank, r: usize, v: f64, scratch: std.mem.Allocator) !void {
        const kw = self.k * self.bits / 32;
        const codes = try scratch.alloc(u32, self.k);
        defer scratch.free(codes);
        const row = self.w[r * kw ..][0..kw];
        for (codes, 0..) |*q, i| q.* = @intFromFloat(codeAt(row, i, self.bits));
        for (codes[0..self.gs]) |*q| q.* = @max(q.*, 1);
        packDense(codes, self.bits, row);
        self.s[r * (self.k / self.gs)] = v;
    }
};

/// A shuffle of the second half's positions and x's mirrored halves (8 entries one apart), as `stressDot`.
pub fn mirroredX(rng: std.Random, x: []f64, perm: []usize) void {
    const h = x.len / 2;
    for (perm, 0..) |*q, i| q.* = i;
    rng.shuffle(usize, perm);
    for (0..h) |i| {
        x[i] = @floatFromInt(rng.intRangeAtMost(i32, 1, 64));
        x[h + perm[i]] = x[i] + @as(f64, if (i < 8) 1 else 0);
    }
}

const MoeGu = struct {
    pub const Params = struct { bits: u32, gs: usize = 64, sbits: u32 = 8, sgs: usize = 64, experts: usize = 8, lim: f64 = 10 };
    const RPS = 2;

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        const vpw: usize = if (p.bits == 3) 8 else 32 / p.bits;
        if (s.rows == 0 or s.rows > p.experts or s.n == 0 or s.n % RPS != 0 or s.k % p.gs != 0 or s.k % p.sgs != 0 or p.gs % vpw != 0) return null;
        var g = metal.Geometry{ .grid = .{ 32, ci(s.n / RPS), ci(s.rows + 1) }, .tg = .{ 32, 1, 1 } };
        g.out(&.{ ci(s.rows + 1), ci(s.n) }, act);
        g.int("RPS", RPS);
        g.int("K", s.k);
        g.int("N", s.n);
        g.int("TOPK", s.rows);
        g.int("BITS", p.bits);
        g.int("GS", p.gs);
        g.int("SBITS", p.sbits);
        g.int("SGS", p.sgs);
        g.flag("SMX", false);
        g.dtype("T", act);
        return g;
    }

    fn banks(k: anytype) [2]Bank {
        const p = k.p;
        const K = k.shape.k;
        return .{
            .{ .w = k.in(1), .s = k.in(2), .b = k.in(3), .bits = p.bits, .gs = p.gs, .k = K },
            .{ .w = k.in(4), .s = k.in(5), .b = k.in(6), .bits = p.sbits, .gs = p.sgs, .k = K },
        };
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = k.p;
        const N = k.shape.n;
        const K = k.shape.k;
        const x = try k.add(act, &.{K});
        const routed = try Bank.add(k, &.{ p.experts, 2 * N }, p.bits, p.gs, K);
        const sh = try Bank.add(k, &.{2 * N}, p.sbits, p.sgs, K);
        const inds = try k.add(.uint32, &.{k.shape.rows});
        const wts = try k.add(.float32, &.{k.shape.rows});
        const lim = try k.add(.float32, &.{1});
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, wts, 0.05, 0.5);
        lim[0] = p.lim;
        var used: [64]bool = @splat(false);
        for (inds) |*v| {
            var e = rng.uintLessThan(usize, p.experts);
            while (used[e]) e = (e + 1) % p.experts;
            used[e] = true;
            v.* = @floatFromInt(e);
        }
        switch (f) {
            .random, .nonfinite => {
                try routed.random(rng, k.scratch());
                try sh.random(rng, k.scratch());
            },
            .stress => {
                const perm = try k.scratch().alloc(usize, K / 2);
                defer k.scratch().free(perm);
                mirroredX(rng, x, perm);
                try routed.mirrored(rng, perm, k.scratch());
                try sh.mirrored(rng, perm, k.scratch());
            },
        }
        if (f == .nonfinite) {
            // A NaN or infinite first-group scale in one gate or up row of slots 0-3 (gate, gate, up, both), that
            // group's x positive.
            kit.uniform(rng, x[0..p.gs], 0.1, 2);
            const put = [_][3]f64{ .{ 0, nan, 0 }, .{ 1, inf, 0 }, .{ 2, 0, -inf }, .{ 3, inf, -inf } };
            for (put) |q| {
                const e: usize = @intFromFloat(q[0]);
                if (e >= inds.len) continue;
                const base = @as(usize, @intFromFloat(inds[e])) * 2 * N + e % N;
                if (q[1] != 0) try routed.plant(base, q[1], k.scratch());
                if (q[2] != 0) try routed.plant(base + N, q[2], k.scratch());
            }
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return (s.rows + 1) * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const N = k.shape.n;
        const T = k.shape.rows;
        const bk = banks(k);
        const t = try k.scratch().alloc(f64, k.shape.k);
        defer k.scratch().free(t);
        const lim = k.in(9)[0];
        for (0..T + 1) |e| for (0..N) |n| {
            const b = if (e < T) bk[0] else bk[1];
            const base = if (e < T) @as(usize, @intFromFloat(k.in(7)[e])) * 2 * N else 0;
            const g = b.dot(base + n, k.in(0), sum, t);
            const up = b.dot(base + N + n, k.in(0), sum, t);
            const wt = if (e < T) k.in(8)[e] else 1;
            const gt = bf(g[0]);
            const ut = bf(up[0]);
            const gc = kit.minN(gt, lim);
            const uc = if (std.math.isNan(ut)) nan else std.math.clamp(ut, -lim, lim);
            const sig = 1 / (1 + @exp(-gc));
            out[e * N + n] = gc * sig * uc * wt;
            // Each sum's error (and its bf16 rounding) reaches y scaled by y's slope in it.
            const slope_g = @abs(sig * (1 + gc * (1 - sig)) * uc * wt);
            mag[e * N + n] = (g[1] + @abs(g[0]) * 0x1p-8 / spec.gamma(@floatFromInt(k.shape.k), u)) * slope_g * 2 + (up[1] + @abs(up[0]) * 0x1p-8 / spec.gamma(@floatFromInt(k.shape.k), u)) * @abs(gc * sig * wt) * 2;
        };
    }
};

const MoeDn = struct {
    pub const Params = struct { bits: u32, gs: usize = 64, sbits: u32 = 8, sgs: usize = 64, experts: usize = 8 };
    const RPS = 2;

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        const vpw: usize = if (p.bits == 3) 8 else 32 / p.bits;
        if (s.rows == 0 or s.rows > p.experts or s.n == 0 or s.n % RPS != 0 or s.k % p.gs != 0 or s.k % p.sgs != 0 or p.gs % vpw != 0) return null;
        var g = metal.Geometry{ .grid = .{ 32, ci(s.n / RPS), 1 }, .tg = .{ 32, 1, 1 } };
        g.out(&.{ 1, HC, ci(s.n) }, act);
        g.int("RPS", RPS);
        g.int("K", s.k);
        g.int("N", s.n);
        g.int("TOPK", s.rows);
        g.int("BITS", p.bits);
        g.int("GS", p.gs);
        g.int("SBITS", p.sbits);
        g.int("SGS", p.sgs);
        g.flag("SMX", false);
        g.int("HC", HC);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = k.p;
        const T = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const x = try k.add(act, &.{ T + 1, K });
        const routed = try Bank.add(k, &.{ p.experts, N }, p.bits, p.gs, K);
        const sh = try Bank.add(k, &.{N}, p.sbits, p.sgs, K);
        const inds = try k.add(.uint32, &.{T});
        const res = try k.add(act, &.{ 1, HC, N });
        const post = try k.add(.float32, &.{ 1, HC });
        const comb = try k.add(.float32, &.{ 1, HC, HC });
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, res, -3, 3);
        kit.uniform(rng, post, 0, 2);
        kit.uniform(rng, comb, 0, 0.5);
        var used: [64]bool = @splat(false);
        for (inds) |*v| {
            var e = rng.uintLessThan(usize, p.experts);
            while (used[e]) e = (e + 1) % p.experts;
            used[e] = true;
            v.* = @floatFromInt(e);
        }
        switch (f) {
            .random => {
                try routed.random(rng, k.scratch());
                try sh.random(rng, k.scratch());
            },
            .nonfinite => {
                // A NaN or infinite group scale in output d's row of slots 0-3 (d = slot): NaN, +inf, -inf, then
                // +inf in slot 3 and -inf in the shared row of the same output.
                try routed.random(rng, k.scratch());
                try sh.random(rng, k.scratch());
                const put = [_]f64{ nan, inf, -inf, inf };
                for (put, 0..) |v, e| if (e < T) {
                    kit.uniform(rng, x[e * K ..][0..p.gs], 0.1, 2);
                    try routed.plant(@as(usize, @intFromFloat(inds[e])) * N + e, v, k.scratch());
                };
                if (T > 3) {
                    kit.uniform(rng, x[T * K ..][0..p.sgs], 0.1, 2);
                    try sh.plant(3, -inf, k.scratch());
                }
            },
            .stress => {
                // Every slot's x the same mirrored halves; post 1, comb 0: the output is the summed dots, exact.
                const perm = try k.scratch().alloc(usize, K / 2);
                defer k.scratch().free(perm);
                mirroredX(rng, x[0..K], perm);
                for (1..T + 1) |e| @memcpy(x[e * K ..][0..K], x[0..K]);
                try routed.mirrored(rng, perm, k.scratch());
                try sh.mirrored(rng, perm, k.scratch());
                @memset(post, 1);
                @memset(comb, 0);
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return HC * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const p = k.p;
        const N = k.shape.n;
        const K = k.shape.k;
        const T = k.shape.rows;
        const routed = Bank{ .w = k.in(1), .s = k.in(2), .b = k.in(3), .bits = p.bits, .gs = p.gs, .k = K };
        const sh = Bank{ .w = k.in(4), .s = k.in(5), .b = k.in(6), .bits = p.sbits, .gs = p.sgs, .k = K };
        const t = try k.scratch().alloc(f64, K);
        defer k.scratch().free(t);
        const flip = 0x1p-8 / spec.gamma(@floatFromInt(K), u);
        var parts: [17]f64 = undefined;
        for (0..N) |d| {
            var m: f64 = 0;
            for (0..T) |e| {
                const r = routed.dot(@as(usize, @intFromFloat(k.in(7)[e])) * N + d, k.in(0)[e * K ..][0..K], sum, t);
                parts[e] = bf(r[0]);
                m += r[1] + @abs(r[0]) * flip;
            }
            const s = sh.dot(d, k.in(0)[T * K ..][0..K], sum, t);
            const tot = sum.of(parts[0..T]);
            const y = bf(tot + bf(s[0]));
            m += s[1] + @abs(s[0]) * flip + (@abs(tot) + @abs(y)) * flip;
            for (0..HC) |i| {
                var acc = k.in(9)[i] * y;
                var cm: f64 = 0;
                for (0..HC) |j| {
                    acc += k.in(10)[j * HC + i] * k.in(8)[j * N + d];
                    cm += @abs(k.in(10)[j * HC + i] * k.in(8)[j * N + d]);
                }
                out[i * N + d] = acc;
                mag[i * N + d] = @abs(k.in(9)[i]) * m + cm;
            }
        }
    }
};

// ── the GGUF decode MoE (`ffnDecodeGguf`): shapes (decode rows, outputs, K), TOPK slots over 8 experts ──

/// A GGUF expert bank's stored bytes (one value a byte, as the rig binds uint8; split types packed as the loader packs
/// them), `rows` rows of `k` weights.
const GgufBank = struct {
    bytes: []f64,
    ty: gguf_quants.GgmlType,
    k: usize,

    fn add(c: anytype, ty: gguf_quants.GgmlType, rows: []const usize, k: usize) !GgufBank {
        var d: [3]usize = undefined;
        @memcpy(d[0..rows.len], rows);
        d[rows.len] = gguf.rowBytes(ty, k);
        return .{ .bytes = try c.add(.uint8, d[0 .. rows.len + 1]), .ty = ty, .k = k };
    }

    /// Random finite blocks, or (`stress`) integer blocks whose second half repeats the first negated (the scale's
    /// sign flipped), against x rows from `mirroredRows`: weight 1 on a row's first 8 entries, so the 8 terms the
    /// mirror leaves sum exactly in bf16 too (Q8_0 and Q4_0, the integer types).
    fn fill(self: GgufBank, rng: std.Random, stress: bool, scratch: std.mem.Allocator) !void {
        const bb = self.ty.blockBytes();
        const nb = self.k / self.ty.blockElems();
        const rb = gguf.rowBytes(self.ty, self.k);
        const rows = self.bytes.len / rb;
        const raw = try scratch.alloc(u8, rows * nb * bb);
        defer scratch.free(raw);
        var dec: [256]f32 = undefined;
        for (0..rows * nb) |b| {
            const blk = raw[b * bb ..][0..bb];
            if (stress and b % nb >= nb / 2) {
                @memcpy(blk, raw[(b - nb / 2) * bb ..][0..bb]);
                gguf_contract.negate(self.ty, blk);
            } else if (stress) {
                gguf_contract.integerBlock(self.ty, blk, rng);
                if (b % nb == 0) gguf_contract.onesFirst8(self.ty, blk);
            } else try gguf_contract.randomBlock(self.ty, blk, dec[0..self.ty.blockElems()], rng, false);
        }
        const stored = try scratch.alloc(u8, rows * rb);
        defer scratch.free(stored);
        if (gguf.isSplit(self.ty, self.k)) gguf.packRows(self.ty, self.k, stored, raw) else @memcpy(stored, raw);
        for (self.bytes, stored) |*o, v| o.* = @floatFromInt(v);
    }

    /// Block `j` of row `r` in ggml's own layout (a split row's payload and head put back together).
    fn block(self: GgufBank, r: usize, j: usize, out: []u8) void {
        const bb = self.ty.blockBytes();
        const row = self.bytes[r * gguf.rowBytes(self.ty, self.k) ..];
        const l = gguf.splitLayout(self.ty) orelse {
            for (out[0..bb], row[j * bb ..][0..bb]) |*o, v| o.* = @intFromFloat(v);
            return;
        };
        const nb = self.k / self.ty.blockElems();
        const head_at: usize = if (l.payload_at == 0) l.payload else 0;
        for (out[l.payload_at..][0..l.payload], row[j * l.payload ..][0..l.payload]) |*o, v| o.* = @intFromFloat(v);
        for (out[head_at..][0..l.head], row[nb * l.payload + j * l.head ..][0..l.head]) |*o, v| o.* = @intFromFloat(v);
    }

    /// Row `r`'s first block scale (its fp16 `d`) set to `v`.
    fn plant(self: GgufBank, r: usize, v: f64) void {
        const nb = self.k / self.ty.blockElems();
        const d = gguf_contract.scaleAt(self.ty);
        const at = r * gguf.rowBytes(self.ty, self.k) + if (gguf.isSplit(self.ty, self.k)) blk: {
            const l = gguf.splitLayout(self.ty).?;
            break :blk nb * l.payload + d - (if (l.payload_at == 0) l.payload else 0);
        } else d;
        const bits: u16 = @bitCast(@as(f16, @floatCast(v)));
        self.bytes[at] = @floatFromInt(bits & 0xFF);
        self.bytes[at + 1] = @floatFromInt(bits >> 8);
    }

    /// Row `r` against `x` [K]: the sum and its terms' magnitude.
    fn dot(self: GgufBank, r: usize, x: []const f64, sum: Sum, t: []f64) ![2]f64 {
        const be = self.ty.blockElems();
        var blk: [256]u8 = undefined;
        var dec: [256]f32 = undefined;
        var mag: f64 = 0;
        for (0..self.k / be) |j| {
            self.block(r, j, &blk);
            try gguf_quants.dequantize(self.ty, blk[0..self.ty.blockBytes()], dec[0..be]);
            for (0..be) |i| {
                t[j * be + i] = x[j * be + i] * dec[i];
                mag += @abs(x[j * be + i] * dec[i]);
            }
        }
        return .{ sum.of(t[0..self.k]), mag };
    }
};

/// Integer x rows 1..16 whose second half is the first plus 1 on its first 8 entries: against a bank whose second
/// half negates its first (`GgufBank`, `Bank.mirrored` over the identity), eight terms of each sum remain.
fn mirroredRows(rng: std.Random, x: []f64, k: usize, perm: []usize) void {
    for (perm, 0..) |*q, i| q.* = i;
    const h = k / 2;
    for (0..x.len / k) |r| for (0..h) |i| {
        x[r * k + i] = @floatFromInt(rng.intRangeAtMost(i32, 1, 16));
        x[r * k + h + i] = x[r * k + i] + @as(f64, if (i < 8) 1 else 0);
    };
}

/// `TOPK` distinct experts of 8 a row.
fn routeRows(rng: std.Random, inds: []f64) void {
    for (0..inds.len / TOPK) |row| {
        var pool: [8]usize = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
        rng.shuffle(usize, &pool);
        for (0..TOPK) |j| inds[row * TOPK + j] = @floatFromInt(pool[j]);
    }
}

const MoeGuGguf = struct {
    pub const Params = struct { ty: gguf_quants.GgmlType, sbits: u32, sgs: usize, experts: usize = 8, lim: f64 = 10 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.rows > v41.region_rows or s.n == 0 or s.k % 32 != 0 or s.k % p.sgs != 0) return null;
        const sg = v41.gguf_gu_sg;
        var g = metal.Geometry{ .grid = .{ 32, ci((s.n + sg - 1) / sg * sg), ci(s.rows * (TOPK + 1)) }, .tg = .{ 32, ci(sg), 1 } };
        g.out(&.{ ci(s.rows * (TOPK + 1)), ci(s.n) }, act);
        for (gguf.layoutConsts(p.ty, ci(s.k), ci(gguf.rowBytes(p.ty, s.k)))) |c| g.int(std.mem.span(c[0]), c[1]);
        g.int("RB", v41.gguf_dn_rb);
        g.int("N", s.n);
        g.int("TOPK", TOPK);
        g.int("SBITS", p.sbits);
        g.int("SGS", p.sgs);
        g.flag("SMX", false);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = k.p;
        const L = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const x = try k.add(act, &.{ L, K });
        const wg = try GgufBank.add(k, p.ty, &.{ p.experts, N }, K);
        const wu = try GgufBank.add(k, p.ty, &.{ p.experts, N }, K);
        const sh = try Bank.add(k, &.{2 * N}, p.sbits, p.sgs, K);
        const inds = try k.add(.uint32, &.{L * TOPK});
        const wts = try k.add(.float32, &.{L * TOPK});
        const lim = try k.add(.float32, &.{1});
        // Gate sums of an RMS-normed input's size against GGUF weights up to 64: f32's exp overflows past |g| 88.
        kit.uniform(rng, x, -1.0 / 32.0, 1.0 / 32.0);
        kit.uniform(rng, wts, 0.05, 0.5);
        lim[0] = p.lim;
        routeRows(rng, inds);
        try wg.fill(rng, f == .stress, k.scratch());
        try wu.fill(rng, f == .stress, k.scratch());
        if (f == .stress) {
            const perm = try k.scratch().alloc(usize, K / 2);
            defer k.scratch().free(perm);
            mirroredRows(rng, x, K, perm);
            try sh.mirrored(rng, perm, k.scratch());
        } else try sh.random(rng, k.scratch());
        if (f == .nonfinite) {
            // A NaN or infinite first-block scale in one gate or up row of row 0's slots 0-3 (gate, gate, up, both),
            // that block's x positive.
            kit.uniform(rng, x[0..32], 0.1, 2);
            const put = [_][3]f64{ .{ 0, nan, 0 }, .{ 1, inf, 0 }, .{ 2, 0, -inf }, .{ 3, inf, -inf } };
            for (put) |q| {
                const e: usize = @intFromFloat(q[0]);
                const row = @as(usize, @intFromFloat(inds[e])) * N + e % N;
                if (q[1] != 0) wg.plant(row, q[1]);
                if (q[2] != 0) wu.plant(row, q[2]);
            }
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * (TOPK + 1) * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const p = k.p;
        const L = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const wg = GgufBank{ .bytes = k.in(1), .ty = p.ty, .k = K };
        const wu = GgufBank{ .bytes = k.in(2), .ty = p.ty, .k = K };
        const sh = Bank{ .w = k.in(3), .s = k.in(4), .b = k.in(5), .bits = p.sbits, .gs = p.sgs, .k = K };
        const t = try k.scratch().alloc(f64, K);
        defer k.scratch().free(t);
        const lim = k.in(8)[0];
        const flip = 0x1p-8 / spec.gamma(@floatFromInt(K), u);
        for (0..L) |l| for (0..TOPK + 1) |j| for (0..N) |c| {
            const x = k.in(0)[l * K ..][0..K];
            var g: [2]f64 = undefined;
            var up: [2]f64 = undefined;
            var wt: f64 = 1;
            var z = L * TOPK + l;
            if (j < TOPK) {
                z = l * TOPK + j;
                const e: usize = @intFromFloat(k.in(6)[z]);
                g = try wg.dot(e * N + c, x, sum, t);
                up = try wu.dot(e * N + c, x, sum, t);
                wt = k.in(7)[z];
            } else {
                g = sh.dot(c, x, sum, t);
                up = sh.dot(N + c, x, sum, t);
            }
            const gc = kit.minN(bf(g[0]), lim);
            const ut = bf(up[0]);
            const uc = if (std.math.isNan(ut)) nan else std.math.clamp(ut, -lim, lim);
            const sig = 1 / (1 + @exp(-gc));
            out[z * N + c] = gc * sig * uc * wt;
            // Each sum's error (and its bf16 rounding) reaches y scaled by y's slope in it.
            const slope_g = @abs(sig * (1 + gc * (1 - sig)) * uc * wt);
            mag[z * N + c] = (g[1] + @abs(g[0]) * flip) * slope_g * 2 + (up[1] + @abs(up[0]) * flip) * @abs(gc * sig * wt) * 2;
        };
    }
};

const MoeDnGguf = struct {
    pub const Params = struct { ty: gguf_quants.GgmlType, sbits: u32, sgs: usize, experts: usize = 8 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        const rb = v41.gguf_dn_rb;
        if (s.rows == 0 or s.rows > v41.region_rows or s.n == 0 or s.n % rb != 0 or s.k % 32 != 0 or s.k % p.sgs != 0) return null;
        var g = metal.Geometry{ .grid = .{ 32, ci(s.n / rb * (TOPK + 1)), ci(s.rows) }, .tg = .{ 32, TOPK + 1, 1 } };
        g.out(&.{ ci(s.rows), HC, ci(s.n) }, act);
        for (gguf.layoutConsts(p.ty, ci(s.k), ci(gguf.rowBytes(p.ty, s.k)))) |c| g.int(std.mem.span(c[0]), c[1]);
        g.int("RB", rb);
        g.int("N", s.n);
        g.int("TOPK", TOPK);
        g.int("SBITS", p.sbits);
        g.int("SGS", p.sgs);
        g.flag("SMX", false);
        g.int("HC", HC);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = k.p;
        const L = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const x = try k.add(act, &.{ L * (TOPK + 1), K });
        const w = try GgufBank.add(k, p.ty, &.{ p.experts, N }, K);
        const sh = try Bank.add(k, &.{N}, p.sbits, p.sgs, K);
        const inds = try k.add(.uint32, &.{L * TOPK});
        const res = try k.add(act, &.{ L, HC, N });
        const post = try k.add(.float32, &.{ L, HC });
        const comb = try k.add(.float32, &.{ L, HC, HC });
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, res, -3, 3);
        kit.uniform(rng, post, 0.1, 2);
        kit.uniform(rng, comb, 0, 0.5);
        routeRows(rng, inds);
        try w.fill(rng, f == .stress, k.scratch());
        if (f == .stress) {
            // Post 1, comb 0: the output is the summed dots, exact.
            const perm = try k.scratch().alloc(usize, K / 2);
            defer k.scratch().free(perm);
            mirroredRows(rng, x, K, perm);
            try sh.mirrored(rng, perm, k.scratch());
            @memset(post, 1);
            @memset(comb, 0);
        } else try sh.random(rng, k.scratch());
        // A NaN or infinite first-block scale in output d's row of row 0's slots 0-3 (d = slot): NaN, +inf, -inf,
        // then +inf in slot 3 and -inf in the shared row of the same output; each block's x positive.
        if (f == .nonfinite) {
            for ([_]f64{ nan, inf, -inf, inf }, 0..) |v, e| {
                kit.uniform(rng, x[e * K ..][0..32], 0.1, 2);
                w.plant(@as(usize, @intFromFloat(inds[e])) * N + e, v);
            }
            kit.uniform(rng, x[L * TOPK * K ..][0..p.sgs], 0.1, 2);
            try sh.plant(3, -inf, k.scratch());
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * HC * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const p = k.p;
        const L = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const w = GgufBank{ .bytes = k.in(1), .ty = p.ty, .k = K };
        const sh = Bank{ .w = k.in(2), .s = k.in(3), .b = k.in(4), .bits = p.sbits, .gs = p.sgs, .k = K };
        const t = try k.scratch().alloc(f64, K);
        defer k.scratch().free(t);
        const flip = 0x1p-8 / spec.gamma(@floatFromInt(K), u);
        var parts: [TOPK]f64 = undefined;
        for (0..L) |l| for (0..N) |d| {
            var m: f64 = 0;
            for (0..TOPK) |e| {
                const z = l * TOPK + e;
                const r = try w.dot(@as(usize, @intFromFloat(k.in(5)[z])) * N + d, k.in(0)[z * K ..][0..K], sum, t);
                parts[e] = bf(r[0]);
                m += r[1] + @abs(r[0]) * flip;
            }
            const s = sh.dot(d, k.in(0)[(L * TOPK + l) * K ..][0..K], sum, t);
            var tot: f64 = 0;
            for (parts) |v| tot += v;
            const y = bf(tot + bf(s[0]));
            m += s[1] + @abs(s[0]) * flip + (@abs(tot) + @abs(y)) * flip;
            for (0..HC) |i| {
                var acc = k.in(7)[l * HC + i] * y;
                var cm: f64 = 0;
                for (0..HC) |j| {
                    acc += k.in(8)[(l * HC + j) * HC + i] * k.in(6)[(l * HC + j) * N + d];
                    cm += @abs(k.in(8)[(l * HC + j) * HC + i] * k.in(6)[(l * HC + j) * N + d]);
                }
                out[(l * HC + i) * N + d] = acc;
                mag[(l * HC + i) * N + d] = @abs(k.in(7)[l * HC + i]) * m + cm;
            }
        };
    }
};

// ── registrations ──

fn depthHc(_: Shape) usize {
    return HC;
}
fn depthIdx(s: Shape) usize {
    return s.k + IH;
}
fn depthSoftmax(s: Shape) usize {
    return WINDOW - 1 + s.rows + s.k;
}
pub fn depthAttn(s: Shape) usize {
    return HD + WINDOW + s.k;
}
fn depthHd(_: Shape) usize {
    return HD;
}
fn depthGather(s: Shape) usize {
    return WINDOW + s.k + 1;
}

/// A launch of one of the module's kernels: `metal_kernel` input / output names as `buildKernel` gives them; `m12`
/// the M1 and M2 limit Xcode's `metal-tt` compiles it for (`kernel contract offline`).
pub fn launch(comptime name: [:0]const u8, comptime ins: []const [*:0]const u8, comptime outs: []const [*:0]const u8, comptime src: [:0]const u8, comptime header: [:0]const u8, comptime tg_bytes: u32, comptime m12: u32, comptime adapts: bool) metal.Launch {
    return .{ .name = name, .inputs = ins, .outputs = outs, .source = src, .header = header, .tg_bytes = tg_bytes, .m1 = m12, .m2 = m12, .adapts = adapts };
}

const attn_ins: []const [*:0]const u8 = &.{ "q", "ring", "comp", "topk", "sink", "meta", "scal" };
const moe_gu_ins: []const [*:0]const u8 = &.{ "x", "w", "scales", "biases", "sw", "sscales", "sbiases", "inds", "wts", "lim" };
const moe_dn_ins: []const [*:0]const u8 = &.{ "x", "w", "scales", "biases", "sw", "sscales", "sbiases", "inds", "res", "post", "comb" };
const hc_stage_ins: []const [*:0]const u8 = &.{ "x", "cpre", "w", "fn", "scale", "base", "eps" };
const hc_stage_x_ins: []const [*:0]const u8 = &.{ "x", "y", "spost", "scomb", "cpre", "w", "fn", "scale", "base", "eps" };

const gate_l = launch("dsv41_gate", &.{ "x", "w" }, &.{"logits"}, v41.GATE_SOURCE, "", 0, 1024, false);
const dense_l = launch("dsv41_dense_f32x", &.{ "x", "w" }, &.{"out"}, v41.DENSE_F32X_SOURCE, "", 0, 928, false);
const route_l = launch("dsv41_route", &.{ "logits", "bias", "scal" }, &.{ "idx", "wts" }, v41.ROUTE_SOURCE, "", 4 * (1024 + 32 + 32 + TOPK), 1024, false);
const topk_l = launch("dsv41_topk", &.{"x"}, &.{"out"}, v41.TOPK_SOURCE, v41.TOPK_HEADER, 4 * (256 + 2 + TopK.TG / 32), 1024, false);
const split_l = launch("dsv41_hc_split", &.{ "z", "eps" }, &.{ "pre", "post", "comb" }, v41.HC_SPLIT_SOURCE, "", 0, 1024, false);
const mix_l = launch("dsv41_hc_mix", &.{ "x", "fn", "scale", "base", "eps" }, &.{"z"}, v41.HC_MIX_SOURCE, "", 4 * (4 * 6 + 4) * 32, 1024, true);
const norm_l = launch("dsv41_hc_norm", &.{ "z", "x", "cpre", "w", "eps" }, &.{ "pre", "post", "comb", "h_out" }, v41.HC_NORM_SOURCE, "", 4 * 32, 1024, true);
const stage_l = launch("dsv41_hc_stage", hc_stage_ins, &.{ "h_out", "z" }, v41.HC_STAGE_SOURCE, "", 2 * 4 * 32, 1024, true);
const stage_x_l = launch("dsv41_hc_stage_x", hc_stage_x_ins, &.{ "h_out", "z", "x1" }, v41.HC_STAGE_X_SOURCE, "", 2 * 4 * 32, 1024, true);
const idx_l = launch("dsv41_index_scores", &.{ "q", "keys", "w", "meta", "wsc" }, &.{"scores"}, v41.IDX_SCORE_SOURCE, "", 4 * IH * IHD, 416, false);
const tile_l = launch("dsv41_index_tile", &.{ "q", "keys", "w", "meta", "wsc" }, &.{"scores"}, v41.IDX_TILE_SOURCE, "", 2 * IdxTile.KT * IHD + 4 * IH + 4 * 4 * 64, 1024, false);
pub const rope_l = launch("dsv41_rope_tail", &.{ "x", "periods", "off", "scal" }, &.{"out"}, v41.ROPE_TAIL_SOURCE, v41.FP8_HEADER, 0, 1024, false);
const expand_into_l = launch("dsv41_hc_expand_into", &.{ "x", "res", "post", "comb", "dst", "after" }, &.{"carrier"}, sx1.expand_into_source, "", 0, 704, false);
const expand_norm_l = launch("dsv41_hc_expand_norm", &.{ "y", "res", "post", "comb", "cpre", "w", "eps" }, &.{ "h_out", "x1" }, hcf.expand_norm_source, hcf.expand_norm_fresh_header, 4 * 32, 1024, true);
const expand_norm_into_l = launch("dsv41_hc_expand_norm_into", &.{ "y", "res", "post", "comb", "cpre", "w", "eps", "dst", "after" }, &.{ "h_out", "carrier" }, hcf.expand_norm_source, hcf.expand_norm_into_header, 4 * 32, 1024, true);
const gate_gemm_l = metal.Launch{ .name = "dsv41_gate_gemm", .inputs = &.{ "a", "b" }, .outputs = &.{"y"}, .source = ggemm.body, .header = ggemm.header, .tg_bytes = 0, .offline = false, .min_gen = 17 };
const markov_step_l = launch("dsv41_markov_step", &.{ "lg", "emb", "mh", "t1", "pin" }, &.{ "pout", "row" }, markov.step_source, markov.header, 4 * (2 * 8 + 1), 1024, false);
const markov_final_l = launch("dsv41_markov_final", &.{ "parts", "t1", "hout", "emb", "cw" }, &.{ "toks", "conf" }, markov.final_source, markov.header, 4 * (3 * 16 + 1), 1024, false);
const flash_l = metal.Launch{ .name = "dsv41_attn_flash", .inputs = &v41.GatherAttn.FLASH_INS, .outputs = &.{"o"}, .source = fa.source, .header = @import("decode/attn.zig").HEADER, .tg_bytes = 4 * 2 * 128 + 4 * 2 * 4 * 4 * 16 + 2 * 4 * 8 * 256, .offline = false, .min_gen = 17 };
const gather_served: []const Shape = &.{ .{ .rows = 3, .n = HEADS, .k = 512 }, .{ .rows = 2, .n = HEADS, .k = 0 }, .{ .rows = 5, .n = HEADS, .k = 128 } };
const qk_t_l = metal.Launch{ .name = "dsv41_attn_qk_t", .inputs = &.{ "q", "kw", "comp", "picks", "meta", "qt" }, .outputs = &.{"s"}, .source = qtail.attn_qk_t_source, .header = v41.MPP_HEADER, .tg_bytes = 2 * 64 * 64 * 2, .offline = false, .min_gen = 17 };
const mix_norm_l = launch("dsv41_hc_mix_norm", &.{ "x", "fn", "scale", "base", "eps", "cpre", "w", "w2" }, &.{ "z", "h_out", "h2" }, hcf.mix_norm_source, "", 4 * (4 * 6 + 4) * 32, 1024, true);
const rope_only_l = launch("dsv41_rope_tail_only", &.{ "x", "periods", "off", "scal" }, &.{"out"}, qtail.rope_tail_only_source, "", 0, 1024, false);
const kv_l = launch("dsv41_kv_row", &.{ "x", "w", "periods", "off", "eps" }, &.{"out"}, v41.KV_ROW_SOURCE, v41.FP8_HEADER, 4 * (32 + 1024), 1024, false);
const engram_l = launch("dsv41_engram", &.{ "x", "kv", "qk", "cst" }, &.{"out"}, v41.ENGRAM_SOURCE, "", 4 * (3 * 32 + 1), 928, false);
const dec_l = launch("dsv41_attn_decode", attn_ins ++ &[_][*:0]const u8{"periods"}, &.{"out"}, v41.ATTN_DEC_SOURCE, v41.UNROPE_HEADER, 4 * (AttnDec.NSG * HD + 2 * AttnDec.NSG), 768, false);
const split_attn_l = launch("dsv41_attn_split", attn_ins, &.{ "pacc", "pm", "pl" }, v41.ATTN_SPLIT_SOURCE, "", 4 * (AttnSplit.NSG * HD + 2 * AttnSplit.NSG), 768, false);
const merge_l = launch("dsv41_attn_merge", &.{ "pacc", "pm", "pl", "sink", "meta", "periods" }, &.{"out"}, v41.ATTN_MERGE_SOURCE, v41.UNROPE_HEADER, 0, 928, false);
const softmax_l = launch("dsv41_attn_softmax", &.{ "lw", "lc", "topk", "sink", "meta", "scal" }, &.{ "pw", "pc" }, v41.ATTN_SOFTMAX_SOURCE, "", 0, 928, false);
const qk_l = metal.Launch{ .name = "dsv41_attn_qk", .inputs = &.{ "q", "kw", "comp", "picks", "meta" }, .outputs = &.{"s"}, .source = v41.ATTN_QK_SOURCE, .header = v41.MPP_HEADER, .tg_bytes = 2 * 64 * 64 * 2, .offline = false, .min_gen = 17 };
const sm_l = metal.Launch{ .name = "dsv41_attn_sm", .inputs = &.{ "s", "picks", "sink", "meta", "scal" }, .outputs = &.{"p"}, .source = v41.ATTN_SM_SOURCE, .tg_bytes = 0, .offline = false, .min_gen = 17 };
const pv_l = metal.Launch{ .name = "dsv41_attn_pv", .inputs = &.{ "p", "kw", "comp", "picks", "meta", "periods" }, .outputs = &.{"o"}, .source = v41.ATTN_PV_SOURCE, .header = v41.MPP_HEADER, .tg_bytes = 2 * 32 * 128 * 2, .offline = false, .min_gen = 17 };
const wt_l = launch("dsv41_moe_wt", &.{ "a", "wts" }, &.{"out"}, v41.MOE_WT_SOURCE, "", 0, 1024, false);
const gu_l = launch("dsv41_moe_gu", moe_gu_ins, &.{"out"}, v41.MOE_GU_SOURCE, v41.MOE_HEADER, 0, 1024, false);

/// The GGUF decode MoE kernels at a routed type, as `gguf.pluginKernel` builds them.
fn GgufMoe(comptime ty: gguf_quants.GgmlType) type {
    return struct {
        const gu = launch("dsv41_moe_gu_gguf_" ++ @tagName(ty), &v41.MOE_GU_GGUF_INS, &.{"out"}, gguf.pluginSource(ty, v41.MOE_GU_GGUF_SOURCE), gguf.pluginHeader(ty, v41.MOE_HEADER), 0, 1024, false);
        const dn = launch("dsv41_moe_dn_gguf_" ++ @tagName(ty), &v41.MOE_DN_GGUF_INS, &.{"out"}, gguf.pluginSource(ty, v41.MOE_DN_GGUF_SOURCE), gguf.pluginHeader(ty, v41.MOE_HEADER), 4 * (TOPK + 1) * v41.gguf_dn_rb, 1024, false);
    };
}
const dn_l = launch("dsv41_moe_dn", moe_dn_ins, &.{"out"}, v41.MOE_DN_SOURCE, v41.MOE_HEADER, 0, 928, false);
const act_l = launch("dsv41_moe_act", &.{ "y", "wts", "lim" }, &.{"out"}, v41.MOE_ACT_SOURCE, "", 0, 1024, false);
const combine_l = launch("dsv41_moe_combine", &.{ "dn", "inv", "sh" }, &.{"out"}, v41.MOE_COMBINE_SOURCE, "", 0, 1024, false);

pub const relu_nan: spec.Known = .{ .gate = .nonfinite, .finding = "a NaN q . k dot scores as 0: metal::max(dot, 0) drops NaN where relu (the reference, torch) propagates it" };
pub const quant_nan: spec.Nan = .{ .stated = "classes as float64 gives them; quantized entries within the quantizer's envelope of their bounds" };
pub const nan_code: spec.Known = .{ .gate = .nonfinite, .finding = "a NaN in a quantized block is written as a finite code: metal::max drops it from the block max and the fp8 encoder saturates it to 448 (float64 and the reference propagate it)" };
const squares: spec.Sums = .{ .f16 = 0, .bf16 = 0, .why = "a sum of squares cannot cancel" };
const short_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "four positive terms a sum, inside 39 normalizations" };
const softmax_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "positive exp terms: the sum cannot cancel" };
const swiglu_nan: spec.Known = .{ .gate = .nonfinite, .finding = "a NaN gate sum becomes the clip limit: metal::min drops NaN where MLX's minimum (the composed chain) propagates it" };

/// Served shapes cut the independent axes (heads, bank rows) to a few where the float64 reference would run for
/// seconds: each runs the same code at its own index; the reduction axis and the row widths stay the release's.
pub fn register(set: *kc.Set) !void {
    try @import("decode/decode.zig").register(set);
    try registerStreamed(set);
    try kit.add(set, Gate, &gate_l, .{}, .{
        .name = "dsv41 gate (router logits)",
        .accum = .f32,
        .depth = depthK,
        .out = .f32,
        .tiles = .{ .{}, .{ .size = 8 }, .{ .size = 256, .step = 8, .min = 8 } },
        .served = &.{ .{ .rows = 1, .n = EXPERTS, .k = DIM }, .{ .rows = REGION, .n = EXPERTS, .k = DIM } },
    });
    try kit.add(set, DenseF32x, &dense_l, .{}, .{
        .name = "dsv41 dense_f32x (compressor wkv)",
        .accum = .f32,
        .depth = depthK,
        .out = .f32,
        .tiles = .{ .{}, .{ .size = 4, .step = 4, .min = 4 }, .{ .size = 128, .step = 128, .min = 128 } },
        .served = &.{ .{ .rows = 1, .n = 2 * IHD, .k = DIM }, .{ .rows = REGION, .n = 2 * IHD, .k = DIM } },
    });
    try kit.add(set, Route, &route_l, .{}, .{
        .name = "dsv41 route (sqrt-softplus top-6)",
        .accum = .none,
        .depth = depthOne,
        .out = .f32,
        .outs = &.{ .{ .elem = .u32, .count = rowsN }, .{ .elem = .f32, .count = rowsN } },
        .nan = .{ .stated = "a NaN score is never picked; weights classed as float64 gives them" },
        .served = &.{ .{ .rows = 1, .n = TOPK, .k = EXPERTS }, .{ .rows = REGION, .n = TOPK, .k = EXPERTS } },
    });
    try kit.add(set, TopK, &topk_l, .{}, .{
        .name = "dsv41 topk (radix select)",
        .accum = .none,
        .depth = depthOne,
        .out = .f32,
        .nan = .{ .stated = "a NaN takes its order key's place (positive NaN above +inf, negative below -inf), as the reference's selection does on the keys" },
        .served = &.{ .{ .rows = 1, .n = 512, .k = 2048 }, .{ .rows = 4, .n = 512, .k = 3000 }, .{ .rows = REGION, .n = 64, .k = 600 } },
    });
    try kit.add(set, HcSplit, &split_l, .{}, .{
        .name = "dsv41 hc_split (Sinkhorn)",
        .accum = .f32,
        .depth = depthHc,
        .ops = sinkhorn_ops,
        .out = .f32,
        .served = &.{ .{ .rows = 1, .n = DIM, .k = HC }, .{ .rows = REGION, .n = DIM, .k = HC } },
        .sums = short_sums,
    });
    try kit.add(set, HcMix, &mix_l, .{}, .{
        .name = "dsv41 hc_mix (wide mixes)",
        .accum = .f32,
        .depth = depthK,
        .ops = 4,
        .out = .f32,
        .tiles = .{ .{ .size = 4 }, .{}, .{} },
        .served = &.{ .{ .rows = 17, .n = DIM, .k = HC * DIM }, .{ .rows = 64, .n = DIM, .k = HC * DIM } },
    });
    try kit.add(set, HcNorm, &norm_l, .{}, .{
        .name = "dsv41 hc_norm (split + normed collapse)",
        .accum = .f32,
        .depth = depthHc,
        .ops = sinkhorn_ops,
        .out = .f32,
        .nan = .{ .stated = "classes as float64 gives them; the normed rows within the intervals of their roundings" },
        .served = &.{ .{ .rows = 17, .n = DIM, .k = HC }, .{ .rows = 64, .n = DIM, .k = HC } },
        .sums = short_sums,
    });
    const stage_nan: spec.Nan = .{ .stated = "classes as float64 gives them; the normed rows and the stream within the intervals of their roundings" };
    // The normed collapse (T) then the mixes (f32); the two roundings to T inside the collapse are its mids.
    const stage_outs: []const spec.Part = &.{ .{ .elem = .bf16, .count = rowsN }, .{ .elem = .f32, .count = rowsMix } };
    try kit.add(set, HcStage, &stage_l, .{ .expand = false }, .{
        .name = "dsv41 hc_stage (decode mixes + normed collapse)",
        .accum = .f32,
        .depth = depthK,
        .ops = 4,
        .mids = 2,
        .out = .bf16,
        .outs = stage_outs,
        .nan = stage_nan,
        .served = &.{ .{ .rows = 1, .n = DIM, .k = HC * DIM }, .{ .rows = REGION, .n = DIM, .k = HC * DIM } },
    });
    try kit.add(set, HcStage, &stage_x_l, .{ .expand = true }, .{
        .name = "dsv41 hc_stage_x (expand, mixes, normed collapse)",
        .accum = .f32,
        .depth = depthK,
        .ops = 4,
        .mids = 2,
        .out = .bf16,
        .outs = stage_outs,
        .nan = stage_nan,
        .served = &.{ .{ .rows = 1, .n = DIM, .k = HC * DIM }, .{ .rows = REGION, .n = DIM, .k = HC * DIM } },
    });
    try kit.add(set, IdxScore, &idx_l, .{}, .{
        .name = "dsv41 idx_score (decode indexer)",
        .accum = .f32,
        .depth = depthIdx,
        .out = .f32,
        .tiles = .{ .{}, .{ .size = 256 }, .{} },
        .served = &.{ .{ .rows = 1, .n = 512, .k = IHD }, .{ .rows = REGION, .n = 300, .k = IHD } },
        .known = relu_nan,
    });
    try kit.add(set, IdxTile, &tile_l, .{}, .{
        .name = "dsv41 idx_tile (wide indexer, 8x8 matrices)",
        .accum = .f32,
        .depth = depthIdx,
        .out = .f32,
        .tiles = .{ .{}, .{ .size = IdxTile.KT }, .{} },
        .served = &.{ .{ .rows = 17, .n = 300, .k = IHD }, .{ .rows = 24, .n = 200, .k = IHD } },
        .known = relu_nan,
    });
    const rope_served: []const Shape = &.{ .{ .rows = 1, .n = 8, .k = HD }, .{ .rows = REGION, .n = 8, .k = HD } };
    try kit.add(set, RopeTail, &rope_l, .{ .qk = .none, .off = 3001 }, .{
        .name = "dsv41 rope_tail q",
        .accum = .none,
        .depth = depthOne,
        .out = .bf16,
        .nan = quant_nan,
        .served = rope_served,
    });
    try kit.add(set, RopeTail, &rope_l, .{ .qk = .none, .off = 3001, .sign = -1 }, .{
        .name = "dsv41 rope_tail o (inverse)",
        .accum = .none,
        .depth = depthOne,
        .out = .bf16,
        .nan = quant_nan,
        .served = rope_served,
    });
    try kit.add(set, RopeTail, &rope_l, .{ .qk = .fp4, .ot = .float32, .off = 3001 }, .{
        .name = "dsv41 rope_tail indexer q (fp4 / 32 / ue8m0, f32 out)",
        .accum = .none,
        .depth = depthOne,
        .out = .f32,
        .nan = quant_nan,
        .known = nan_code,
        .served = &.{ .{ .rows = 1, .n = IH, .k = IHD }, .{ .rows = REGION, .n = IH, .k = IHD } },
    });
    try kit.add(set, RopeTail, &rope_l, .{ .qk = .fp4, .scal = 4, .off = 750 }, .{
        .name = "dsv41 rope_tail indexer keys (fp4 / 32 / ue8m0, ratio 4)",
        .accum = .none,
        .depth = depthOne,
        .out = .bf16,
        .nan = quant_nan,
        .known = nan_code,
        .served = &.{ .{ .rows = 1, .n = 1, .k = IHD }, .{ .rows = 64, .n = 1, .k = IHD } },
    });
    try kit.add(set, RopeTail, &rope_l, .{ .qk = .fp4_e4m3, .scal = 4, .off = 750 }, .{
        .name = "dsv41 rope_tail latent (fp4 / 16 / e4m3)",
        .accum = .none,
        .depth = depthOne,
        .out = .bf16,
        .nan = quant_nan,
        .known = nan_code,
        .served = &.{ .{ .rows = 1, .n = 1, .k = HD }, .{ .rows = 64, .n = 1, .k = HD } },
    });
    try kit.add(set, KvRow, &kv_l, .{}, .{
        .name = "dsv41 kv_row (rms, rope, fp8 / 32 / ue8m0)",
        .accum = .f32,
        .depth = depthK,
        .out = .bf16,
        .nan = quant_nan,
        .known = nan_code,
        .served = &.{ .{ .rows = 1, .n = 1, .k = HD }, .{ .rows = 64, .n = 1, .k = HD } },
        .sums = squares,
    });
    try kit.add(set, Engram, &engram_l, .{}, .{
        .name = "dsv41 engram (gated write)",
        .accum = .f32,
        .depth = depthK,
        .ops = 16,
        .out = .bf16,
        .replaces = "engramGateOps (the op chain)",
        .known = .{ .gate = .limits, .finding = "at the release's 5120 wide rows it launches MLX's row-sum thread count, 1024, and M1/M2 compile it for 928 with no probe: a decode-width Engram write fails on M1/M2 (the op chain past region_rows does not)" },
        .served = &.{ .{ .rows = 1, .n = HC, .k = DIM }, .{ .rows = REGION, .n = HC, .k = DIM } },
        .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "bit-equal to the op chain's MLX row sums (gate 3); random rows, not a stress" },
    });
    const attn_nan: spec.Nan = .{ .stated = "classes as float64 gives them; outputs within the intervals of the online softmax's roundings" };
    const split_nan: spec.Nan = .{ .stated = "a split whose weights go NaN reports max -inf and sum 0 (Metal's max and den > 0 drop the NaN) and a NaN partial output, which the merge propagates; else classes as float64 gives them" };
    try kit.add(set, AttnDec, &dec_l, .{ .rd = RD }, .{
        .name = "dsv41 attn_dec (one threadgroup, inverse rope)",
        .accum = .none,
        .depth = depthAttn,
        .out = .bf16,
        .nan = attn_nan,
        .served = &.{ .{ .rows = 1, .n = 8, .k = 0 }, .{ .rows = 4, .n = 8, .k = 150 } },
    });
    try kit.add(set, AttnSplit, &split_attn_l, .{ .s = 4 }, .{
        .name = "dsv41 attn_split S=4",
        .accum = .none,
        .depth = depthAttn,
        .out = .f32,
        .nan = split_nan,
        .served = &.{ .{ .rows = 1, .n = 8, .k = 512 }, .{ .rows = 4, .n = 8, .k = 512 } },
    });
    try kit.add(set, AttnSplit, &split_attn_l, .{ .s = 8, .g = .{ .nc = 1500 } }, .{
        .name = "dsv41 attn_split S=8",
        .accum = .none,
        .depth = depthAttn,
        .out = .f32,
        .nan = split_nan,
        .served = &.{.{ .rows = 1, .n = 8, .k = 1200 }},
    });
    try kit.add(set, AttnMerge, &merge_l, .{ .rd = RD, .s = 4 }, .{
        .name = "dsv41 attn_merge S=4 (inverse rope)",
        .accum = .none,
        .depth = depthOne,
        .out = .bf16,
        .nan = attn_nan,
        .served = &.{ .{ .rows = 1, .n = HEADS, .k = 512 }, .{ .rows = REGION, .n = HEADS, .k = 512 } },
    });
    try kit.add(set, AttnSoftmax, &softmax_l, .{}, .{
        .name = "dsv41 attn_softmax (wide span, sink)",
        .accum = .f32,
        .depth = depthSoftmax,
        .ops = 8,
        .out = .f16,
        .served = &.{ .{ .rows = 17, .n = 8, .k = 512 }, .{ .rows = 64, .n = 8, .k = 0 } },
        .sums = softmax_sums,
    });
    try kit.add(set, AttnQk, &qk_l, .{}, .{
        .name = "dsv41 attn_qk (prompt width, matrix units)",
        .accum = .f32,
        .depth = depthHd,
        .out = .f32,
        .nax = true,
        .served = gather_served,
    });
    try kit.add(set, AttnSm, &sm_l, .{}, .{
        .name = "dsv41 attn_sm (masked softmax, sink)",
        .accum = .f32,
        .depth = depthGather,
        .ops = 8,
        .out = .f16,
        .served = gather_served,
        .sums = softmax_sums,
    });
    try kit.add(set, AttnPv, &pv_l, .{}, .{
        .name = "dsv41 attn_pv (prompt width, matrix units)",
        .accum = .f32,
        .depth = depthGather,
        .out = .bf16,
        .nax = true,
        .served = gather_served,
    });
    try kit.add(set, AttnPv, &pv_l, .{ .grouped = true }, .{
        .name = "dsv41 attn_pv grouped (inverse rope, wo_a layout)",
        .accum = .f32,
        .depth = depthGather,
        .ops = 4,
        .out = .bf16,
        .nax = true,
        .served = gather_served,
        .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "the rotation after the sums rounds in f32; the plain arm proves the sums" },
    });
    try kit.add(set, MoeWt, &wt_l, .{}, .{
        .name = "dsv41 moe_wt (routing weight on a fused gate/up activation)",
        .accum = .none,
        .depth = depthOne,
        .out = .bf16,
        .served = &.{ .{ .rows = 102, .n = INTER, .k = 1 }, .{ .rows = 7, .n = 64, .k = 1 } },
    });
    inline for (.{ 2, 3, 4 }) |bits| {
        try kit.add(set, MoeGu, &gu_l, .{ .bits = bits }, .{
            .name = std.fmt.comptimePrint("dsv41 moe_gu {d}-bit g64 + 8-bit shared", .{bits}),
            .accum = .f32,
            .depth = depthK,
            .ops = 8,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 2, .step = 2, .min = 2 }, .{} },
            .served = &.{ .{ .rows = TOPK, .n = 64, .k = DIM }, .{ .rows = 2, .n = 64, .k = 128 } },
            .sums = .{ .f16 = 30, .bf16 = 30, .why = "a wrong sum's activation tends to 0, so a miss tops out near 1 / the bound" },
            .known = swiglu_nan,
        });
        try kit.add(set, MoeDn, &dn_l, .{ .bits = bits }, .{
            .name = std.fmt.comptimePrint("dsv41 moe_dn {d}-bit g64 + 8-bit shared, hc expand", .{bits}),
            .accum = .f32,
            .depth = depthK,
            .ops = 4,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 2, .step = 2, .min = 2 }, .{} },
            .served = &.{ .{ .rows = TOPK, .n = 64, .k = INTER }, .{ .rows = 2, .n = 64, .k = 128 } },
        });
    }
    // The GGUF decode MoE at the routed types whose blocks carry integer weights (the K-quants a 32-weight slice at a
    // time), beside 5- and 8-bit shared experts.
    inline for (.{ .{ gguf_quants.GgmlType.q8_0, 5, 32 }, .{ gguf_quants.GgmlType.q4_0, 8, 64 }, .{ gguf_quants.GgmlType.q2_k, 8, 64 }, .{ gguf_quants.GgmlType.q4_k, 8, 64 }, .{ gguf_quants.GgmlType.q5_k, 5, 32 }, .{ gguf_quants.GgmlType.q6_k, 8, 64 } }) |r| {
        const L = GgufMoe(r[0]);
        try kit.add(set, MoeGuGguf, &L.gu, .{ .ty = r[0], .sbits = r[1], .sgs = r[2] }, .{
            .name = std.fmt.comptimePrint("dsv41 moe_gu_gguf {s} + {d}-bit g{d} shared", .{ @tagName(r[0]), r[1], r[2] }),
            .accum = .f32,
            .depth = depthK,
            .ops = 8,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = v41.gguf_gu_sg }, .{ .size = 1024, .step = @max(64, r[0].blockElems()), .min = @max(64, r[0].blockElems()) } },
            .served = &.{ .{ .rows = 1, .n = 64, .k = 2048 }, .{ .rows = 3, .n = 37, .k = 1024 }, .{ .rows = v41.region_rows, .n = 16, .k = 512 } },
            .sums = .{ .f16 = 30, .bf16 = 30, .why = "a wrong sum's activation tends to 0, so a miss tops out near 1 / the bound" },
            .known = swiglu_nan,
        });
        try kit.add(set, MoeDnGguf, &L.dn, .{ .ty = r[0], .sbits = r[1], .sgs = r[2] }, .{
            .name = std.fmt.comptimePrint("dsv41 moe_dn_gguf {s} + {d}-bit g{d} shared, hc expand", .{ @tagName(r[0]), r[1], r[2] }),
            .accum = .f32,
            .depth = depthK,
            .ops = 4,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = v41.gguf_dn_rb, .step = v41.gguf_dn_rb, .min = v41.gguf_dn_rb }, .{ .size = 1024, .step = @max(64, r[0].blockElems()), .min = @max(64, r[0].blockElems()) } },
            .served = &.{ .{ .rows = 1, .n = 64, .k = 2304 }, .{ .rows = 3, .n = 36, .k = 1024 }, .{ .rows = v41.region_rows, .n = 16, .k = 512 } },
            .sums = if (r[0].blockElems() == 256) .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "K 2304 is an odd count of 256-weight blocks, which the mirrored stress rows cannot cancel; the Q8_0 and Q4_0 arms prove these sums, the gathers the K-quant slices" } else .{},
        });
    }
    try kit.add(set, MoeAct, &act_l, .{}, .{
        .name = "dsv41 moe_act (prefill clipped SwiGLU x weight)",
        .accum = .none,
        .depth = depthOne,
        .ops = 12,
        .out = .bf16,
        .served = &.{ .{ .rows = 102, .n = INTER, .k = 1 }, .{ .rows = 7, .n = 64, .k = 1 } },
        .known = swiglu_nan,
    });
    try kit.add(set, MoeCombine, &combine_l, .{}, .{
        .name = "dsv41 moe_combine (prefill unsort + K-sum + shared)",
        .accum = .f32,
        .depth = depthK1,
        .out = .bf16,
        .served = &.{ .{ .rows = 17, .n = DIM, .k = TOPK }, .{ .rows = 64, .n = 256, .k = TOPK } },
    });
}

// ── tests ──

test "kernel contract dsv41: dense packing puts element i at bit i * bits" {
    var words: [3]f64 = undefined;
    const codes = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 7, 7, 7, 7, 7, 7, 7, 7 };
    packDense(&codes, 3, &words);
    for (codes, 0..) |c, i| try std.testing.expectEqual(@as(f64, @floatFromInt(c)), codeAt(&words, i, 3));
}

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

/// A model shell carrying the kernels and config fields the wrappers read, built as `deepseek_v41.init` builds them.
const Shell = struct {
    m: v41.Dsv41Model,

    fn init(s: mlx.mlx_stream) !Shell {
        var m: v41.Dsv41Model = undefined;
        m.s = s;
        m.cfg = .{};
        m.cfg.hidden_size = DIM;
        m.cfg.rms_norm_eps = EPS_RMS;
        m.cfg.dsv4_hc_eps = EPS_HC;
        m.cfg.dsv4_hc_mult = HC;
        m.cfg.dsv4_hc_sinkhorn_iters = ITERS;
        m.cfg.dsv4_rope_head_dim = RD;
        m.cfg.sliding_window = WINDOW;
        m.cfg.router_scaling_factor = 1.5;
        m.cfg.moe_route_norm = true;
        m.act = act;
        m.qat_off = false;
        m.hc_tg = 1024;
        m.pair = null;
        const b = v41.buildKernel;
        m.attn_dec_kernel = try b("dsv41_attn_decode", &v41.ATTN_DEC_INS, &.{"out"}, v41.ATTN_DEC_SOURCE, v41.UNROPE_HEADER);
        m.attn_split_kernel = try b("dsv41_attn_split", &v41.ATTN_INS, &.{ "pacc", "pm", "pl" }, v41.ATTN_SPLIT_SOURCE, "");
        m.attn_merge_kernel = try b("dsv41_attn_merge", &v41.ATTN_MERGE_INS, &.{"out"}, v41.ATTN_MERGE_SOURCE, v41.UNROPE_HEADER);
        m.idx_score_kernel = try v41.buildIdxScoreKernel();
        m.idx_tile_kernel = try b("dsv41_index_tile", &v41.IDX_INS, &.{"scores"}, v41.IDX_TILE_SOURCE, "");
        m.gate_kernel = try b("dsv41_gate", &.{ "x", "w" }, &.{"logits"}, v41.GATE_SOURCE, "");
        m.topk_kernel = try b("dsv41_topk", &.{"x"}, &.{"out"}, v41.TOPK_SOURCE, v41.TOPK_HEADER);
        m.topk_tg = 1024;
        m.dense_f32x_kernel = try b("dsv41_dense_f32x", &.{ "x", "w" }, &.{"out"}, v41.DENSE_F32X_SOURCE, "");
        m.route_kernel = try b("dsv41_route", &.{ "logits", "bias", "scal" }, &.{ "idx", "wts" }, v41.ROUTE_SOURCE, "");
        m.hc_mix_kernel = try b("dsv41_hc_mix", &.{ "x", "fn", "scale", "base", "eps" }, &.{"z"}, v41.HC_MIX_SOURCE, "");
        m.hc_norm_kernel = try b("dsv41_hc_norm", &.{ "z", "x", "cpre", "w", "eps" }, &.{ "pre", "post", "comb", "h_out" }, v41.HC_NORM_SOURCE, "");
        m.hc_stage_kernel = try b("dsv41_hc_stage", &v41.HC_STAGE_INS, &v41.HC_STAGE_OUTS, v41.HC_STAGE_SOURCE, "");
        m.hc_stage_x_kernel = try b("dsv41_hc_stage_x", &v41.HC_STAGE_X_INS, &v41.HC_STAGE_X_OUTS, v41.HC_STAGE_X_SOURCE, "");
        m.hc_split_kernel = try b("dsv41_hc_split", &.{ "z", "eps" }, &.{ "pre", "post", "comb" }, v41.HC_SPLIT_SOURCE, "");
        m.rope_kernel = try b("dsv41_rope_tail", &.{ "x", "periods", "off", "scal" }, &.{"out"}, v41.ROPE_TAIL_SOURCE, v41.FP8_HEADER);
        m.rope_tail_only_kernel = try b("dsv41_rope_tail_only", &.{ "x", "periods", "off", "scal" }, &.{"out"}, qtail.rope_tail_only_source, "");
        m.kv_row_kernel = try b("dsv41_kv_row", &.{ "x", "w", "periods", "off", "eps" }, &.{"out"}, v41.KV_ROW_SOURCE, v41.FP8_HEADER);
        m.engram_kernel = try b("dsv41_engram", &v41.ENGRAM_INS, &.{"out"}, v41.ENGRAM_SOURCE, "");
        m.moe_gu_kernel = try b("dsv41_moe_gu", &v41.MOE_GU_INS, &.{"out"}, v41.MOE_GU_SOURCE, v41.MOE_HEADER);
        m.moe_dn_kernel = try b("dsv41_moe_dn", &v41.MOE_DN_INS, &.{"out"}, v41.MOE_DN_SOURCE, v41.MOE_HEADER);
        m.moe_act_kernel = try b("dsv41_moe_act", &v41.MOE_ACT_INS, &.{"out"}, v41.MOE_ACT_SOURCE, "");
        m.moe_combine_kernel = try b("dsv41_moe_combine", &v41.MOE_COMBINE_INS, &.{"out"}, v41.MOE_COMBINE_SOURCE, "");
        return .{ .m = m };
    }

    fn deinit(self: *Shell) void {
        const m = &self.m;
        for ([_]mlx.mlx_fast_metal_kernel{ m.attn_dec_kernel, m.attn_split_kernel, m.attn_merge_kernel, m.idx_score_kernel, m.idx_tile_kernel, m.gate_kernel, m.topk_kernel, m.dense_f32x_kernel, m.route_kernel, m.hc_mix_kernel, m.hc_norm_kernel, m.hc_stage_kernel, m.hc_stage_x_kernel, m.hc_split_kernel, m.rope_kernel, m.rope_tail_only_kernel, m.kv_row_kernel, m.engram_kernel, m.moe_gu_kernel, m.moe_dn_kernel, m.moe_act_kernel, m.moe_combine_kernel }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
    }
};

/// The kernels the streamed forward brought (`prompt/`, `spec/markov.zig`, `decode/gemv_twin.zig`).
fn mkPairCount(s: Shape) usize {
    return s.rows * MK_NT * 2;
}
fn mkRow(s: Shape) usize {
    return s.n;
}
fn rowsOnly(s: Shape) usize {
    return s.rows;
}
fn depthNK(s: Shape) usize {
    return s.n + s.k;
}

fn registerStreamed(set: *kc.Set) !void {
    inline for (.{ true, false }) |qt| try kit.add(set, AttnFlash, &flash_l, .{ .qt = qt }, .{
        .name = if (qt) "dsv41 attn_flash (one launch, f16 weights, the roped tail from qt)" else "dsv41 attn_flash (one launch, f16 weights)",
        .accum = .f32,
        .depth = kc.host.depthOne,
        .out = .bf16,
        .nax = true,
        .served = gather_served,
        .sums = .{ .exact = false },
    });
    // The vocabulary cut to 40,000 (129,280 served): the step is vocabulary-generic, and this still gives a simdgroup
    // more quads than one turn.
    try kit.add(set, MarkovStep, &markov_step_l, .{}, .{
        .name = "dsv41 markov_step (biased row, its pairs a threadgroup)",
        .accum = .f32,
        .depth = depthK,
        .ops = 1,
        .mids = 1,
        .out = .f32,
        .outs = &.{ .{ .elem = .u32, .count = mkPairCount }, .{ .elem = .f32, .count = mkRow } },
        .replaces = "the draft's op chain (take, matmul, cast, add) and its row's argmax",
        .served = &.{ .{ .rows = 1, .n = 40000, .k = 256 }, .{ .rows = 2, .n = 40000, .k = 256 }, .{ .rows = 6, .n = 40000, .k = 256 } },
    });
    try kit.add(set, MarkovFinal, &markov_final_l, .{}, .{
        .name = "dsv41 markov_final (tokens, confidence logits)",
        .accum = .f32,
        .depth = depthNK,
        .out = .f32,
        .outs = &.{ .{ .elem = .u32, .count = rowsOnly }, .{ .elem = .f32, .count = rowsOnly } },
        .replaces = "the draft's argmax and confidence matmul",
        .served = &.{ .{ .rows = 6, .n = DIM, .k = 256 }, .{ .rows = 16, .n = DIM, .k = 256 } },
    });
    // The twin's served launches (`gemv_twin.choices`) on the indexer's weights_proj (32 x 5120) and wk (128 x 512);
    // Engram's wkv (25600 x 6144) by its plan's class at 4096 x 1024 (the same MLX tiling: bm 8, bn 1).
    const twins = .{
        .{ 32, 5120, gtwin.Variant{ .nr = 1, .tm = 4, .bm = 1 }, &[_]Shape{ .{ .rows = 2, .n = 32, .k = 5120 }, .{ .rows = 5, .n = 32, .k = 5120 } } , 1024 },
        .{ 32, 5120, gtwin.Variant{ .nr = 1, .tm = 1, .bm = 1 }, &[_]Shape{.{ .rows = 6, .n = 32, .k = 5120 }} , 1024 },
        .{ 128, 512, gtwin.Variant{ .nr = 1, .tm = 4, .bm = 4 }, &[_]Shape{ .{ .rows = 2, .n = 128, .k = 512 }, .{ .rows = 5, .n = 128, .k = 512 } }, 928 },
        .{ 128, 512, gtwin.Variant{ .nr = 1, .tm = 1, .bm = 4 }, &[_]Shape{.{ .rows = 6, .n = 128, .k = 512 }} , 1024 },
        .{ 4096, 1024, gtwin.Variant{ .nr = 2, .tm = 1, .bm = 8 }, &[_]Shape{.{ .rows = 2, .n = 4096, .k = 1024 }} , 1024 },
        .{ 4096, 1024, gtwin.Variant{ .nr = 3, .tm = 1, .bm = 4 }, &[_]Shape{.{ .rows = 3, .n = 4096, .k = 1024 }} , 1024 },
        .{ 4096, 1024, gtwin.Variant{ .nr = 4, .tm = 1, .bm = 8 }, &[_]Shape{.{ .rows = 4, .n = 4096, .k = 1024 }} , 1024 },
        .{ 4096, 1024, gtwin.Variant{ .nr = 3, .tm = 2, .bm = 8, .ilv = true }, &[_]Shape{ .{ .rows = 5, .n = 4096, .k = 1024 }, .{ .rows = 6, .n = 4096, .k = 1024 } } , 1024 },
    };
    inline for (twins) |tw| {
        const l = comptime twinLaunch(tw[0], tw[1], tw[2], tw[4]);
        try kit.add(set, GemvTwin, &l, .{ .v = tw[2] }, .{
            .name = "dsv41 gemv_twin (" ++ l.name["dsv41_gemv_twin_".len..] ++ std.fmt.comptimePrint(", {d} x {d})", .{ tw[0], tw[1] }),
            .accum = .f32,
            .depth = depthK,
            .out = .bf16,
            .replaces = "MLX's matmul of each row alone",
            .served = tw[3],
        });
    }
    try kit.add(set, GateGemm, &gate_gemm_l, .{}, .{
        .name = "dsv41 gate_gemm (MLX's NAX GEMM over bf16 x and f16 gate_w)",
        .accum = .f32,
        .depth = depthK,
        .out = .f32,
        .nax = true,
        .replaces = "astype f32, MLX's matmul (its regular NAX kernel)",
        .served = &.{ .{ .rows = 256, .n = 384, .k = 512 }, .{ .rows = 100, .n = 384, .k = 768 }, .{ .rows = 77, .n = 200, .k = 300 } },
    });
    try kit.add(set, AttnQkT, &qk_t_l, .{}, .{
        .name = "dsv41 attn_qk_t (prompt width, the heads and their roped tail)",
        .accum = .f32,
        .depth = depthHd,
        .out = .f32,
        .nax = true,
        .replaces = "attn_qk over the whole roped heads",
        .served = gather_served,
    });
    const expand_served: []const Shape = &.{ .{ .rows = 17, .n = DIM, .k = HC }, .{ .rows = 64, .n = DIM, .k = HC } };
    try kit.add(set, ExpandInto, &expand_into_l, .{}, .{
        .name = "dsv41 hc_expand_into (the new stream into a region)",
        .accum = .f32,
        .depth = depthHc,
        .ops = 2,
        .out = .bf16,
        .in_place = true,
        .replaces = "glm5's expand (the stock launch)",
        .served = expand_served,
        .sums = fused_sums,
    });
    inline for (.{ false, true }) |into| try kit.add(set, ExpandNorm, if (into) &expand_norm_into_l else &expand_norm_l, .{ .into = into }, .{
        .name = if (into) "dsv41 hc_expand_norm (expand into a region + normed collapse)" else "dsv41 hc_expand_norm (expand + normed collapse)",
        .accum = .f32,
        .depth = depthN,
        .ops = 4,
        .mids = 3,
        .out = .bf16,
        .in_place = into,
        .replaces = "glm5's expand then hc_norm (the stock launches)",
        .served = expand_served,
        .sums = fused_sums,
    });
    const mix_norm_served: []const Shape = &.{ .{ .rows = 17, .n = DIM, .k = HC * DIM }, .{ .rows = 64, .n = DIM, .k = HC * DIM } };
    inline for (.{ true, false }) |h2| try kit.add(set, HcMixNorm, &mix_norm_l, .{ .h2 = h2 }, .{
        .name = if (h2) "dsv41 hc_mix_norm (mixes + normed collapse, twice)" else "dsv41 hc_mix_norm (mixes + normed collapse)",
        .accum = .f32,
        .depth = depthK,
        .ops = 4,
        .mids = 2,
        .out = .f32,
        .outs = &.{ .{ .elem = .f32, .count = rowsMix }, .{ .elem = .bf16, .count = rowsN } },
        .replaces = "hc_mix then hc_norm (the stock launches)",
        .tiles = .{ .{ .size = 4 }, .{}, .{} },
        .served = mix_norm_served,
        .sums = .{ .bf16 = 0, .exact = false, .why = "on hc_mix's stress rows the collapse's sums are not exact, so the float64 bound holds them, and a bf16 miss hides under it" },
    });
    try kit.add(set, RopeTailOnly, &rope_only_l, .{}, .{
        .name = "dsv41 rope_tail_only (the query tail)",
        .accum = .none,
        .depth = depthOne,
        .out = .bf16,
        .replaces = "rope_tail's roped channels",
        .nan = quant_nan,
        .served = &.{ .{ .rows = 1, .n = HEADS, .k = HD }, .{ .rows = REGION, .n = HEADS, .k = HD } },
    });
}

/// A prepared case run on our launch: its output bytes (owned) and the case for the wrapper's call.
fn ours(comptime D: type, a: std.mem.Allocator, l: *const metal.Launch, p: D.Params, shape: Shape) !Ran(D) {
    return oursAt(D, a, l, p, shape, .random);
}

fn Ran(comptime D: type) type {
    return struct { c: *kit.Case(D), bytes: []u8 };
}

fn oursAt(comptime D: type, a: std.mem.Allocator, l: *const metal.Launch, p: D.Params, shape: Shape, f: Fill) !Ran(D) {
    const c = try a.create(kit.Case(D));
    errdefer a.destroy(c);
    c.* = .init(a, l, p);
    try std.testing.expect(try c.prepare(shape, f, 7));
    try c.run();
    return .{ .c = c, .bytes = try a.dupe(u8, c.outBytes()) };
}

/// A NaN-filled [L, HC, D] region on `c`'s rig, for a wrapper to write.
fn freshRegion(c: anytype, L: usize, D: usize) !A {
    const v = try c.scratch().alloc(f64, L * HC * D);
    defer c.scratch().free(v);
    @memset(v, nan);
    return c.rig.input(act, &.{ ci(L), HC, ci(D) }, v);
}

fn held(x: A) A {
    var o = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&o, x);
    return o;
}

test "kernel contract dsv41: each declared launch is the shipped wrapper's, byte for byte" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var sh = try Shell.init(mlx.gpuStream());
    defer sh.deinit();
    const m = &sh.m;
    var sc = v41.Scope.init(a, m.s);
    defer sc.deinit();
    const Expect = struct {
        fn same(c: anytype, mine: []u8, theirs: []const A) !void {
            defer std.testing.allocator.free(mine);
            defer {
                c.deinit();
                std.testing.allocator.destroy(c);
            }
            try std.testing.expectEqualSlices(u8, try c.rig.bytesOf(theirs), mine);
        }
    };
    var ly: v41.Layer = undefined;
    {
        const r = try ours(Gate, a, &gate_l, .{}, .{ .rows = 3, .n = 40, .k = 256 });
        ly.gate_w = r.c.args[1];
        try Expect.same(r.c, r.bytes, &.{try v41.gateLogits(&sc, m, &ly, r.c.args[0])});
    }
    {
        // The router: our gate launch, then our route launch on its logits.
        const r = try ours(Gate, a, &gate_l, .{}, .{ .rows = 3, .n = EXPERTS, .k = 256 });
        const logits = held(r.c.rig.output(0));
        defer _ = mlx.mlx_array_free(logits);
        var bias_v: [EXPERTS]f64 = undefined;
        var prng = std.Random.DefaultPrng.init(3);
        kit.uniform(prng.random(), &bias_v, -0.5, 0.5);
        for (&bias_v) |*v| v.* = spec.roundTo(.f32, v.*);
        const bias = try r.c.rig.input(.float32, &.{ 1, EXPERTS }, &bias_v);
        const scal = try r.c.rig.input(.float32, &.{1}, &.{1.5});
        try r.c.rig.launch(&route_l, Route.geometry(.{}, .{ .rows = 3, .n = TOPK, .k = EXPERTS }).?, &.{ logits, bias, scal });
        a.free(r.bytes);
        ly.gate_w = r.c.args[1];
        ly.gate_bias = bias;
        ly.gate_bias_vl = null;
        ly.top_k = TOPK;
        const rt_ = try v41.route(&sc, m, &ly, r.c.args[0], none);
        try Expect.same(r.c, try a.dupe(u8, r.c.rig.outputBytes()), &.{ rt_.idx, rt_.wts });
    }
    {
        const r = try ours(DenseF32x, a, &dense_l, .{}, .{ .rows = 2, .n = 64, .k = 256 });
        const lin: v41.Lin = .{ .w = r.c.args[1] };
        try Expect.same(r.c, r.bytes, &.{try v41.denseF32x(&sc, m, &lin, r.c.args[0])});
    }
    {
        const r = try ours(TopK, a, &topk_l, .{}, .{ .rows = 3, .n = 64, .k = 600 });
        try Expect.same(r.c, r.bytes, &.{try v41.topkRows(&sc, m, r.c.args[0], 64)});
    }
    inline for (.{
        RopeTail.Params{ .qk = .none, .off = 3001 },
        RopeTail.Params{ .qk = .none, .off = 3001, .sign = -1 },
        RopeTail.Params{ .qk = .fp4, .ot = .float32, .off = 3001 },
        RopeTail.Params{ .qk = .fp4, .scal = 4, .off = 750 },
        RopeTail.Params{ .qk = .fp4_e4m3, .scal = 4, .off = 750 },
    }) |p| {
        const r = try ours(RopeTail, a, &rope_l, p, .{ .rows = 3, .n = if (p.qk == .none) 4 else 1, .k = if (p.qk == .fp4 and p.scal == 1) IHD else HD });
        const qat: ?v41.Qat = switch (p.qk) {
            .none => null,
            .fp8 => .fp8_ue8m0,
            .fp4 => .fp4_ue8m0,
            .fp4_e4m3 => .fp4_e4m3,
        };
        try Expect.same(r.c, r.bytes, &.{try v41.ropeQat(&sc, m, r.c.args[0], r.c.args[1], r.c.args[2], @floatCast(p.scal), qat, p.ot)});
    }
    inline for (.{ gtwin.Variant{ .nr = 1, .tm = 4, .bm = 4 }, gtwin.Variant{ .nr = 3, .tm = 2, .bm = 8, .ilv = true } }, .{ 128, 4096 }, .{ 512, 1024 }, .{ 928, 1024 }) |v, n, kk, m12| {
        const l = comptime twinLaunch(n, kk, v, m12);
        const r = try ours(GemvTwin, a, &l, .{ .v = v }, .{ .rows = 5, .n = n, .k = kk });
        const y = (try gtwin.linearWith(m.s, r.c.args[0], r.c.args[1], v)).?;
        defer _ = mlx.mlx_array_free(y);
        try Expect.same(r.c, r.bytes, &.{y});
    }
    if (@import("../../mlx/device.zig").naxAvailable()) {
        const r = try ours(GateGemm, a, &gate_gemm_l, .{}, .{ .rows = 100, .n = 384, .k = 768 });
        const y = (try ggemm.router(m.s, r.c.args[0], r.c.args[1], true)).?;
        defer _ = mlx.mlx_array_free(y);
        try Expect.same(r.c, r.bytes, &.{y});
    }
    // The flash kernel as the model launches it (`flashLaunch`, grouped).
    if (@import("../../mlx/device.zig").naxAvailable()) inline for (.{ true, false }) |qt| {
        const r = try ours(AttnFlash, a, &flash_l, .{ .qt = qt }, .{ .rows = 3, .n = HEADS, .k = 128 });
        const fk = try v41.buildKernel("dsv41_attn_flash", &v41.GatherAttn.FLASH_INS, &.{"o"}, fa.source, @import("decode/attn.zig").HEADER);
        defer _ = mlx.mlx_fast_metal_kernel_free(fk);
        var got: [1]A = .{none};
        try v41.flashLaunch(&sc, fk, r.c.args[0..9], &.{.{ .shape = &.{ 8, 3, HEADS / 8 * HD }, .dt = act }}, 3, HEADS, qt, act, &got);
        try Expect.same(r.c, r.bytes, &got);
    };
    // The Markov chain through our two launches, step by step, against markov.chain's.
    {
        var rig = metal.Rig.init(a);
        defer rig.deinit();
        var prng = std.Random.DefaultPrng.init(7);
        const rng = prng.random();
        const B = 3;
        const V = 8192;
        const R = 256;
        const D = 512;
        const vals = try a.alloc(f64, V * R);
        defer a.free(vals);
        kit.uniform(rng, vals[0 .. B * V], -4, 4);
        const lg = try rig.input(.float32, &.{ B, V }, vals[0 .. B * V]);
        kit.uniform(rng, vals, -1, 1);
        const emb = try rig.input(act, &.{ V, R }, vals);
        kit.uniform(rng, vals, -1, 1);
        const mh = try rig.input(act, &.{ V, R }, vals);
        kit.uniform(rng, vals[0 .. B * D], -2, 2);
        const hout = try rig.input(act, &.{ B, D }, vals[0 .. B * D]);
        kit.uniform(rng, vals[0 .. D + R], -0.5, 0.5);
        for (vals[0 .. D + R]) |*v| v.* = bf(v.*);
        const cw = try rig.input(.float32, &.{ 1, D + R }, vals[0 .. D + R]);
        const t1 = try rig.input(.int32, &.{1}, &.{5});
        var pin = try rig.input(.uint32, &.{8}, &@as([8]f64, @splat(0)));
        var held_pin: ?A = null;
        defer if (held_pin) |x| {
            _ = mlx.mlx_array_free(x);
        };
        var rows: std.ArrayList(u8) = .empty;
        defer rows.deinit(a);
        for (0..B) |K| {
            try rig.launch(&markov_step_l, MarkovStep.geometry(.{}, .{ .rows = K + 1, .n = V, .k = R }).?, &.{ lg, emb, mh, t1, pin });
            var next = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&next, rig.output(0)));
            if (held_pin) |x| _ = mlx.mlx_array_free(x);
            held_pin = next;
            pin = next;
            try rows.appendSlice(a, try rig.bytesOf(&.{rig.output(1)}));
        }
        try rig.launch(&markov_final_l, MarkovFinal.geometry(.{}, .{ .rows = B, .n = D, .k = R }).?, &.{ pin, t1, hout, emb, cw });
        const fin = try a.dupe(u8, rig.outputBytes());
        defer a.free(fin);
        const o = (try markov.chain(rig.stream(), .{ .logits = lg, .hout = hout, .t1 = t1, .emb = emb, .mh = mh, .cw = cw, .want_conf = true, .keep = true })).?;
        defer for ([_]A{ o.toks, o.conf.?, o.rows.? }) |x| {
            _ = mlx.mlx_array_free(x);
        };
        try std.testing.expectEqualSlices(u8, fin, try rig.bytesOf(&.{ o.toks, o.conf.? }));
        try std.testing.expectEqualSlices(u8, rows.items, try rig.bytesOf(&.{o.rows.?}));
    }
    // The fused hc launches (hc_fuse.zig), built and probed as a load does.
    try v41.hcFuseInit(a, m);
    inline for (.{ false, true }) |into| {
        const r = try ours(ExpandNorm, a, if (into) &expand_norm_into_l else &expand_norm_l, .{ .into = into }, .{ .rows = 6, .n = 300, .k = HC });
        const sp: v41.Split = .{ .pre = none, .post = r.c.args[2], .comb = r.c.args[3] };
        // The wrapper writes a fresh region: one that wrote nothing could not pass on ours.
        const dst = if (into) try freshRegion(r.c, 6, 300) else none;
        const f = try v41.hcExpandNormLaunch(&sc, m, r.c.args[1], r.c.args[0], &sp, r.c.args[4], r.c.args[5], r.c.args[6], if (into) dst else null, if (into) r.c.args[8] else none);
        if (into) {
            try mlx.check(mlx.mlx_array_eval(f.carrier));
            try Expect.same(r.c, r.bytes, &.{ f.h, dst });
        } else try Expect.same(r.c, r.bytes, &.{ f.h, f.x1 });
    }
    {
        const r = try ours(ExpandInto, a, &expand_into_l, .{}, .{ .rows = 6, .n = 300, .k = HC });
        const dst = try freshRegion(r.c, 6, 300);
        const carrier = try sx1.expandInto(m.s, r.c.args[0], r.c.args[1], r.c.args[2], r.c.args[3], dst, r.c.args[5]);
        defer _ = mlx.mlx_array_free(carrier);
        try mlx.check(mlx.mlx_array_eval(carrier));
        try Expect.same(r.c, r.bytes, &.{dst});
    }
    // Gate 3 runs random rows: hc_mix's stress rows (partial sums past fp16's range) give the stock launches' words too.
    inline for (.{ true, false }) |h2| {
        const r = try oursAt(HcMixNorm, a, &mix_norm_l, .{ .h2 = h2 }, .{ .rows = 17, .n = DIM, .k = HC * DIM }, .stress);
        defer {
            a.free(r.bytes);
            r.c.deinit();
            a.destroy(r.c);
        }
        try std.testing.expectEqualSlices(u8, try r.c.replaced(), r.bytes);
    }
    inline for (.{ true, false }) |h2| {
        const r = try ours(HcMixNorm, a, &mix_norm_l, .{ .h2 = h2 }, .{ .rows = 6, .n = 300, .k = HC * 300 });
        const hc: v41.Hc = .{ .fn_w = r.c.args[1], .scale = r.c.args[2], .base = r.c.args[3] };
        const f = try v41.hcMixNormFused(&sc, m, r.c.args[0], r.c.args[5], &hc, r.c.args[6], if (h2) r.c.args[7] else null, r.c.args[4]);
        try Expect.same(r.c, r.bytes, if (h2) &.{ f.z, f.n.h, f.h2 } else &.{ f.z, f.n.h });
    }
    {
        const r = try ours(RopeTailOnly, a, &rope_only_l, .{}, .{ .rows = 3, .n = 4, .k = HD });
        try Expect.same(r.c, r.bytes, &.{try v41.ropeTailOnly(&sc, m, r.c.args[0], r.c.args[1], r.c.args[2])});
    }
    {
        const r = try ours(KvRow, a, &kv_l, .{}, .{ .rows = 3, .n = 1, .k = HD });
        ly.kv_norm = r.c.args[1];
        try Expect.same(r.c, r.bytes, &.{try v41.kvRow(&sc, m, &ly, r.c.args[2], r.c.args[0], r.c.args[3])});
    }
    {
        const r = try ours(HcNorm, a, &norm_l, .{}, .{ .rows = 2, .n = 300, .k = HC });
        const n = try v41.hcNorm(&sc, m, r.c.args[0], r.c.args[1], r.c.args[2], r.c.args[3], r.c.args[4]);
        try Expect.same(r.c, r.bytes, &.{ n.sp.pre, n.sp.post, n.sp.comb, n.h });
    }
    {
        // The wide mixes: our mix launch, then our norm launch on its z, against `hcPreNormWide`.
        const D = 300;
        const r = try ours(HcMix, a, &mix_l, .{}, .{ .rows = 18, .n = D, .k = HC * D });
        const z = held(r.c.rig.output(0));
        defer _ = mlx.mlx_array_free(z);
        var prng = std.Random.DefaultPrng.init(5);
        var cp: [18 * HC]f64 = undefined;
        var w: [D]f64 = undefined;
        kit.uniform(prng.random(), &cp, 0, 1);
        kit.uniform(prng.random(), &w, 0.5, 1.5);
        for (&cp) |*v| v.* = spec.roundTo(.f32, v.*);
        for (&w) |*v| v.* = bf(v.*);
        const cpre = try r.c.rig.input(.float32, &.{ 18, HC }, &cp);
        const nw = try r.c.rig.input(act, &.{D}, &w);
        try r.c.rig.launch(&norm_l, HcNorm.geometry(.{}, .{ .rows = 18, .n = D, .k = HC }).?, &.{ z, r.c.args[0], cpre, nw, r.c.args[4] });
        a.free(r.bytes);
        const hc: v41.Hc = .{ .fn_w = r.c.args[1], .scale = r.c.args[2], .base = r.c.args[3] };
        const n = try v41.hcPreNormWide(&sc, m, r.c.args[0], cpre, &hc, nw);
        try Expect.same(r.c, try a.dupe(u8, r.c.rig.outputBytes()), &.{ n.sp.pre, n.sp.post, n.sp.comb, n.h });
    }
    inline for (.{ false, true }) |expand| {
        // The stage, then our split launch on its z, against `hcStage`.
        const D = 300;
        const L = 3;
        const r = try ours(HcStage, a, if (expand) &stage_x_l else &stage_l, .{ .expand = expand }, .{ .rows = L, .n = D, .k = HC * D });
        const outs = .{ held(r.c.rig.output(0)), held(r.c.rig.output(1)), if (expand) held(r.c.rig.output(2)) else none };
        defer inline for (outs) |o| if (o.ctx != null) {
            _ = mlx.mlx_array_free(o);
        };
        const e = if (expand) 3 else 0;
        try r.c.rig.launch(&split_l, HcSplit.geometry(.{}, .{ .rows = L, .n = D, .k = HC }).?, &.{ outs[1], r.c.args[6 + e] });
        const split_bytes = try a.dupe(u8, r.c.rig.outputBytes());
        defer a.free(split_bytes);
        const hc: v41.Hc = .{ .fn_w = r.c.args[3 + e], .scale = r.c.args[4 + e], .base = r.c.args[5 + e] };
        const prev: v41.Split = .{ .pre = none, .post = if (expand) r.c.args[2] else none, .comb = if (expand) r.c.args[3] else none };
        const ex: ?v41.Expand = if (expand) .{ .y = r.c.args[1], .sp = &prev } else null;
        const st = try v41.hcStage(&sc, m, r.c.args[0], ex, r.c.args[1 + e], &hc, r.c.args[2 + e]);
        try std.testing.expectEqualSlices(u8, try r.c.rig.bytesOf(&.{ st.sp.pre, st.sp.post, st.sp.comb }), split_bytes);
        // Our stage's h (and stream), its z judged through the split above.
        const h_bytes = r.bytes[0 .. 2 * L * D];
        try std.testing.expectEqualSlices(u8, try r.c.rig.bytesOf(&.{st.h}), h_bytes);
        if (expand) try std.testing.expectEqualSlices(u8, try r.c.rig.bytesOf(&.{st.x1}), r.bytes[2 * L * D + 4 * L * MIX ..]);
        a.free(r.bytes);
        r.c.deinit();
        a.destroy(r.c);
    }
    {
        const r = try ours(IdxScore, a, &idx_l, .{}, .{ .rows = 3, .n = 300, .k = IHD });
        try Expect.same(r.c, r.bytes, &.{try v41.indexScoresScalar(&sc, m, r.c.args[0], r.c.args[1], r.c.args[2], @floatCast(r.c.in(4)[0]), @intFromFloat(r.c.in(3)[0]), @intFromFloat(r.c.in(3)[1]))});
    }
    {
        const r = try ours(IdxTile, a, &tile_l, .{}, .{ .rows = 18, .n = 200, .k = IHD });
        try Expect.same(r.c, r.bytes, &.{try v41.indexScoresTiled(&sc, m, r.c.args[0], r.c.args[1], r.c.args[2], @floatCast(r.c.in(4)[0]), @intFromFloat(r.c.in(3)[0]), @intFromFloat(r.c.in(3)[1]))});
    }
    {
        const r = try ours(Engram, a, &engram_l, .{}, .{ .rows = 2, .n = HC, .k = DIM });
        try Expect.same(r.c, r.bytes, &.{try v41.engramGate(&sc, m, r.c.args[2], r.c.args[0], r.c.args[1])});
    }
    inline for (.{ 0, 150 }) |K| {
        const g: AttnGeo = .{};
        const r = try ours(AttnDec, a, &dec_l, .{ .rd = RD }, .{ .rows = 2, .n = 4, .k = K });
        ly.sink = r.c.args[4];
        const win: v41.Ring = .{ .buf = r.c.args[1], .cap = g.cap, .width = HD };
        try Expect.same(r.c, r.bytes, &.{try v41.attnDecodeSplit(&sc, m, &ly, r.c.args[0], win, if (K > 0) r.c.args[2] else none, r.c.args[3], g.p0, @floatCast(g.scale), 1, r.c.args[7])});
    }
    {
        // Split S=4 then our merge launch, against the wrapper's split + merge.
        const g: AttnGeo = .{};
        const L = 2;
        const H = 4;
        const r = try ours(AttnSplit, a, &split_attn_l, .{ .s = 4 }, .{ .rows = L, .n = H, .k = 512 });
        const parts = [3]A{ held(r.c.rig.output(0)), held(r.c.rig.output(1)), held(r.c.rig.output(2)) };
        defer for (parts) |x| {
            _ = mlx.mlx_array_free(x);
        };
        var pv: [RD / 2]f64 = undefined;
        periodsFill(&pv, -1);
        for (&pv) |*v| v.* = spec.roundTo(.f32, v.*);
        const periods = try r.c.rig.input(.float32, &.{RD / 2}, &pv);
        try r.c.rig.launch(&merge_l, AttnMerge.geometry(.{ .rd = RD, .s = 4 }, .{ .rows = L, .n = H, .k = 512 }).?, &.{ parts[0], parts[1], parts[2], r.c.args[4], r.c.args[5], periods });
        a.free(r.bytes);
        ly.sink = r.c.args[4];
        const win: v41.Ring = .{ .buf = r.c.args[1], .cap = g.cap, .width = HD };
        const o = try v41.attnDecodeSplit(&sc, m, &ly, r.c.args[0], win, r.c.args[2], r.c.args[3], g.p0, @floatCast(g.scale), 4, periods);
        try Expect.same(r.c, try a.dupe(u8, r.c.rig.outputBytes()), &.{o});
    }
    inline for (.{ 2, 3, 4 }) |bits| {
        const qp: v41.QuantParams = .{ .bits = bits, .group_size = 64 };
        const sqp: v41.QuantParams = .{ .bits = 8, .group_size = 64 };
        {
            const r = try ours(MoeGu, a, &gu_l, .{ .bits = bits }, .{ .rows = TOPK, .n = 64, .k = 256 });
            try Expect.same(r.c, r.bytes, &.{try v41.moeKernel(&sc, m.moe_gu_kernel, r.c.args[0..10], &.{ TOPK + 1, 64 }, act, qp, sqp, 256, 64, TOPK)});
        }
        {
            const r = try ours(MoeDn, a, &dn_l, .{ .bits = bits }, .{ .rows = TOPK, .n = 64, .k = 256 });
            try Expect.same(r.c, r.bytes, &.{try v41.moeKernel(&sc, m.moe_dn_kernel, r.c.args[0..11], &.{ 1, HC, 64 }, act, qp, sqp, 256, 64, TOPK)});
        }
    }
    inline for (.{ .{ gguf_quants.GgmlType.q8_0, 5, 32 }, .{ gguf_quants.GgmlType.q4_0, 8, 64 } }) |q| {
        const L = GgufMoe(q[0]);
        const sqp: v41.QuantParams = .{ .bits = q[1], .group_size = q[2] };
        const R = 3;
        {
            const r = try ours(MoeGuGguf, a, &L.gu, .{ .ty = q[0], .sbits = q[1], .sgs = q[2] }, .{ .rows = R, .n = 40, .k = 1024 });
            const kern = try gguf.pluginKernel("dsv41_moe_gu_gguf", &v41.MOE_GU_GGUF_INS, v41.MOE_GU_GGUF_SOURCE, v41.MOE_HEADER, q[0]);
            try Expect.same(r.c, r.bytes, &.{try v41.ggufMoeKernel(&sc, kern, r.c.args[0..9], &.{ R * (TOPK + 1), 40 }, act, q[0], sqp, 1024, 40, TOPK, R, gguf.rowBytes(q[0], 1024), null)});
        }
        {
            const r = try ours(MoeDnGguf, a, &L.dn, .{ .ty = q[0], .sbits = q[1], .sgs = q[2] }, .{ .rows = R, .n = 40, .k = 1024 });
            const kern = try gguf.pluginKernel("dsv41_moe_dn_gguf", &v41.MOE_DN_GGUF_INS, v41.MOE_DN_GGUF_SOURCE, v41.MOE_HEADER, q[0]);
            try Expect.same(r.c, r.bytes, &.{try v41.ggufMoeKernel(&sc, kern, r.c.args[0..9], &.{ R, HC, 40 }, act, q[0], sqp, 1024, 40, TOPK, R, gguf.rowBytes(q[0], 1024), HC)});
        }
    }
    {
        const r = try ours(MoeAct, a, &act_l, .{}, .{ .rows = 7, .n = 64, .k = 1 });
        try Expect.same(r.c, r.bytes, &.{try v41.flatKernel(&sc, m.moe_act_kernel, r.c.args[0..3], &.{ 7, 1, 64 }, act, 7 * 64, &.{.{ .name = "N", .v = 64 }})});
    }
    {
        const r = try ours(MoeCombine, a, &combine_l, .{}, .{ .rows = 3, .n = 256, .k = TOPK });
        try Expect.same(r.c, r.bytes, &.{try v41.flatKernel(&sc, m.moe_combine_kernel, r.c.args[0..3], &.{ 3, 256 }, act, 3 * 256, &.{ .{ .name = "D", .v = 256 }, .{ .name = "K", .v = TOPK } })});
    }
}
