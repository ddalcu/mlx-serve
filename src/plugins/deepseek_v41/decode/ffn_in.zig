//! DeepSeek-V4.1 decode, the FFN's entry in two launches. Launch 1: wo_b over the attention output, then each output
//! column's hc expand and collapse. Launch 2: the FFN's 24 hc mixes with their Sinkhorn split, the rms-normed collapse
//! and the router with its top K. Rows 1..6, every weight read once whatever the rows.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mxfp8_qmv = @import("../../../sdk/blocks/qmm/mxfp8_qmv.zig");
const kit = @import("../../deepseek_v4/contract_kit.zig");
const dsv41 = @import("../deepseek_v41.zig");
const contract = @import("../contract.zig");
const decode = @import("decode.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };
const act: mlx.mlx_dtype = .bfloat16;
const hc: usize = 4;
const mixes = (2 + hc) * hc;

const A_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_ffn_in.metal");
const B_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_ffn_in_post.metal");
const HEADER = mxfp8_qmv.HEADER;

const A_INS = [_][*:0]const u8{ "oa", "w_q", "scales", "x", "spre", "spost", "scomb" };
/// `ctr`, `parts` and `scores` are launch 2's scratch: launch 1 zeroes the counters and writes neither of the others.
const A_OUTS = [_][*:0]const u8{ "x1", "hv", "ctr", "parts", "logits" };
const B_INS = [_][*:0]const u8{ "x1", "hv", "fn", "scale", "base", "w", "gate_w", "bias", "scal", "eps", "ctr", "parts", "logits" };
const B_OUTS = [_][*:0]const u8{ "h2", "pre", "post", "comb", "idx", "wts" };

pub const Kernels = struct {
    a: mlx.mlx_fast_metal_kernel,
    b: mlx.mlx_fast_metal_kernel,
};

/// Both launches' kernels, built once.
pub fn build() !Kernels {
    const a = try dsv41.buildKernel("dsv41_dec_ffn_in", &A_INS, &A_OUTS, A_SOURCE, HEADER);
    errdefer _ = mlx.mlx_fast_metal_kernel_free(a);
    return .{ .a = a, .b = try dsv41.buildKernel("dsv41_dec_ffn_in_post", &B_INS, &B_OUTS, B_SOURCE, "") };
}

pub fn free(k: Kernels) void {
    _ = mlx.mlx_fast_metal_kernel_free(k.a);
    _ = mlx.mlx_fast_metal_kernel_free(k.b);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

// ── launch geometry ──

/// Launch 1: simdgroups a threadgroup, K steps loaded ahead, and the output columns a simdgroup: two once the rows' x
/// reads outweigh the weights.
const a_nsg: usize = 8;
const a_unroll: usize = 2;
fn aCols(rows: usize) usize {
    return if (rows > 4) 2 else 1;
}
/// Launch 2: columns a mix slice, experts a router threadgroup (two simdgroups an expert), threads a threadgroup.
const slice_cols: usize = 128;
const experts_tg: usize = 4;
const b_threads: usize = 256;

/// What a call is for: `rows` stream rows through a `d`-wide layer with wo_b `[d, k]` and `e` routed experts.
pub const Dims = struct {
    rows: usize,
    d: usize = 5120,
    k: usize = 8192,
    e: usize = 384,
    top_k: usize = 6,
    iters: usize = 20,
    norm: bool = true,
};

pub const TArg = struct { name: [*:0]const u8, v: usize };

fn servesA(p: Dims) bool {
    return p.rows >= 1 and p.rows <= decode.max_rows and p.d % (a_nsg * aCols(p.rows)) == 0 and p.d >= slice_cols and p.k % (512 * a_unroll) == 0;
}

/// Floats of threadgroup memory launch 2 uses: a router threadgroup's h2 rows, the slices' partials with the reduced
/// mixes, or the scores with their biased values.
fn bWords(p: Dims) usize {
    return @max(@max(rowsTg(p) * (p.d / 2 + 16), p.rows * (slices(p) + 1) * (mixes + 1)), 2 * p.rows * p.e);
}

fn servesB(p: Dims) bool {
    if (p.rows < 1 or p.rows > decode.max_rows or p.d % 512 != 0 or p.d > 6144) return false;
    if (p.top_k < 1 or p.top_k > 32 or p.e < p.top_k) return false;
    return 4 * bWords(p) + 16 <= 32 * 1024;
}

/// Whether the launches take `p` (rows 1..6, D a multiple of 512 up to 6144, K a multiple of 1024); otherwise the caller
/// keeps today's chain.
pub fn serves(p: Dims) bool {
    return servesA(p) and servesB(p);
}

fn slices(p: Dims) usize {
    return p.d / slice_cols;
}

fn routerGroups(p: Dims) usize {
    return (p.e + experts_tg - 1) / experts_tg;
}

/// Rows a router threadgroup takes: it reads its experts' gate rows once for all of them.
fn rowsTg(p: Dims) usize {
    return @min(p.rows, 3);
}

fn routerTgs(p: Dims) usize {
    return routerGroups(p) * ((p.rows + rowsTg(p) - 1) / rowsTg(p));
}

/// Floats of mix partials launch 2's threadgroups hand to its last one: a slice's 24 dots and its sum of squares.
fn partsLen(p: Dims) usize {
    return p.rows * slices(p) * (mixes + 1);
}

fn gridA(p: Dims) usize {
    return p.d / aCols(p.rows) * 32;
}

fn gridB(p: Dims) usize {
    return (slices(p) + routerTgs(p)) * b_threads;
}

fn aTargs(p: Dims) [7]TArg {
    return .{
        .{ .name = "R", .v = p.rows }, .{ .name = "NSG", .v = a_nsg }, .{ .name = "NR", .v = aCols(p.rows) }, .{ .name = "U", .v = a_unroll },
        .{ .name = "K", .v = p.k },    .{ .name = "D", .v = p.d },     .{ .name = "HC", .v = hc },
    };
}

fn bTargs(p: Dims) [10]TArg {
    return .{
        .{ .name = "R", .v = p.rows },                  .{ .name = "D", .v = p.d },         .{ .name = "E", .v = p.e },       .{ .name = "TOPK", .v = p.top_k },
        .{ .name = "HC", .v = hc },                     .{ .name = "ITERS", .v = p.iters }, .{ .name = "P", .v = slices(p) }, .{ .name = "G", .v = routerGroups(p) },
        .{ .name = "NORM", .v = @intFromBool(p.norm) }, .{ .name = "RG", .v = rowsTg(p) },
    };
}

pub const Weights = struct {
    /// wo_b: e4m3 words `[D, K/4]` u32 and e8m0 scales `[D, K/32]` u8.
    wo_w: A,
    wo_s: A,
    /// The FFN's hc mixes: f32 `[24, 4 D]`, `[24]`, `[24]`.
    fn_w: A,
    fn_scale: A,
    fn_base: A,
    norm_w: A,
    /// bf16 `[E, D]` and f32 `[1, E]`.
    gate_w: A,
    gate_bias: A,
};

pub const Out = struct { x1: A, h2: A, sf: dsv41.Split, idx: A, wts: A };
pub const AOut = struct { x1: A, hv: A, ctr: A, parts: A, logits: A };
pub const BOut = struct { h2: A, sf: dsv41.Split, idx: A, wts: A };

const OutSpec = struct { shape: []const c_int, dt: mlx.mlx_dtype };

fn run(comptime n: usize, sc: *dsv41.Scope, kern: mlx.mlx_fast_metal_kernel, ins: []const A, outs: [n]OutSpec, grid: usize, tg: usize, targs: []const TArg) ![n]A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(grid), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", act));
    for (targs) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, ci(t.v)));
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var res: [n]A = undefined;
    for (&res, 0..) |*r, i| {
        r.* = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(r, ov, i)) catch |e| {
            _ = mlx.mlx_array_free(r.*);
            return e;
        };
        sc.items.append(sc.a, r.*) catch |e| {
            _ = mlx.mlx_array_free(r.*);
            return e;
        };
    }
    return res;
}

/// Launch 1 for `p.rows` rows: wo_b over `oa` `[R, K]`, then each output column's hc expand with `x` `[R, 4, D]` and `sa`
/// (the attention's split), and the collapse; it also zeroes launch 2's counters and sizes its scratch.
pub fn launchA(sc: *dsv41.Scope, ks: Kernels, p: Dims, w: Weights, oa: A, x: A, sa: *const dsv41.Split) !AOut {
    const a = try run(5, sc, ks.a, &.{ oa, w.wo_w, w.wo_s, x, sa.pre, sa.post, sa.comb }, .{
        .{ .shape = &.{ ci(p.rows), hc, ci(p.d) }, .dt = act },
        .{ .shape = &.{ ci(p.rows), ci(p.d) }, .dt = act },
        .{ .shape = &.{16}, .dt = .uint32 },
        .{ .shape = &.{ci(partsLen(p))}, .dt = .float32 },
        .{ .shape = &.{ ci(p.rows), ci(p.e) }, .dt = .float32 },
    }, gridA(p), a_nsg * 32, &aTargs(p));
    return .{ .x1 = a[0], .hv = a[1], .ctr = a[2], .parts = a[3], .logits = a[4] };
}

/// Launch 2: the FFN's 24 hc mixes and their split, the rms-normed collapse `h2`, the router's scores and each row's top K.
/// `eps` is `[rms eps, hc eps]`, `scal` the routed scale.
pub fn launchB(sc: *dsv41.Scope, ks: Kernels, p: Dims, w: Weights, a: AOut, eps: A, scal: A) !BOut {
    const b = try run(6, sc, ks.b, &.{ a.x1, a.hv, w.fn_w, w.fn_scale, w.fn_base, w.norm_w, w.gate_w, w.gate_bias, scal, eps, a.ctr, a.parts, a.logits }, .{
        .{ .shape = &.{ ci(p.rows), ci(p.d) }, .dt = act },
        .{ .shape = &.{ ci(p.rows), hc }, .dt = .float32 },
        .{ .shape = &.{ ci(p.rows), hc }, .dt = .float32 },
        .{ .shape = &.{ ci(p.rows), hc, hc }, .dt = .float32 },
        .{ .shape = &.{ ci(p.rows), ci(p.top_k) }, .dt = .uint32 },
        .{ .shape = &.{ ci(p.rows), ci(p.top_k) }, .dt = .float32 },
    }, gridB(p), b_threads, &bTargs(p));
    return .{ .h2 = b[0], .sf = .{ .pre = b[1], .post = b[2], .comb = b[3] }, .idx = b[4], .wts = b[5] };
}

/// wo_b, the hc expand, the FFN norm and mixes, the router and its top K; null where `serves` says no.
pub fn launch(sc: *dsv41.Scope, ks: Kernels, p: Dims, w: Weights, oa: A, x: A, sa: *const dsv41.Split, eps: A, scal: A) !?Out {
    if (!serves(p)) return null;
    const a = try launchA(sc, ks, p, w, oa, x, sa);
    const b = try launchB(sc, ks, p, w, a, eps, scal);
    return .{ .x1 = a.x1, .h2 = b.h2, .sf = b.sf, .idx = b.idx, .wts = b.wts };
}

// ── the kernel contract ──

const spec = kit.spec;
const metal = kit.metal;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;
const Iv = contract.Iv;
const u = 0x1p-24;
const eps_rms = 1e-20;
const eps_hc = 1e-6;
const router_scale = 1.5;

fn bf(v: f64) f64 {
    return spec.roundTo(.bf16, v);
}

/// An e4m3 byte's value (the OCP fn variant); the kernel reads S.1111.111, MLX's NaN, as 480, so fills never draw it.
fn e4m3(b: u8) f64 {
    const sign: f64 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 15;
    const m: f64 = @floatFromInt(b & 7);
    return sign * if (e == 0) m / 8 * 0x1p-6 else (1 + m / 8) * std.math.pow(f64, 2, @floatFromInt(e - 7));
}

/// The e4m3 byte of the integer `c` in 0..16 (exact there).
fn intByte(c: u8) u8 {
    var b: u8 = 0;
    while (e4m3(b) != @as(f64, @floatFromInt(c))) b += 1;
    return b;
}

fn outAt(k: anytype, e: spec.Elem, off: usize, i: usize) f64 {
    return spec.read(e, k.rig.outputBytes()[off..], i);
}

fn perD(s: Shape) usize {
    return s.rows * s.n;
}
fn perHc(s: Shape) usize {
    return s.rows * hc;
}
fn perComb(s: Shape) usize {
    return s.rows * hc * hc;
}
fn perPick(s: Shape) usize {
    return s.rows * 6;
}
/// Rounding depth of wo_b's sums: K steps of 512 a lane and 16 products a step, then the simdgroup's tree.
fn depthA(s: Shape) usize {
    return s.k / 512 + 32;
}
/// The mixes' dot over the stream (4 D terms).
fn depthB(s: Shape) usize {
    return hc * s.n;
}

/// wo_b over the attention output, then each column's expand and collapse. Shape: rows, D columns, K.
const FfnA = struct {
    pub const Params = struct {};

    fn dims(s: Shape) Dims {
        return .{ .rows = s.rows, .d = s.n, .k = s.k };
    }

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        const p = dims(s);
        if (!servesA(p)) return null;
        // The counters, partials and scores are launch 2's scratch (launch 1 writes only the counters).
        var g = metal.Geometry{ .grid = .{ ci(gridA(p)), 1, 1 }, .tg = .{ ci(a_nsg * 32), 1, 1 }, .unread = 0b11100 };
        g.out(&.{ ci(p.rows), hc, ci(p.d) }, act);
        g.out(&.{ ci(p.rows), ci(p.d) }, act);
        g.out(&.{16}, .uint32);
        g.out(&.{ci(partsLen(p))}, .float32);
        g.out(&.{ ci(p.rows), ci(p.e) }, .float32);
        for (aTargs(p)) |t| g.int(t.name, t.v);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const R = k.shape.rows;
        const D = k.shape.n;
        const K = k.shape.k;
        const oa = try k.add(act, &.{ R, K });
        const words = try k.add(.uint32, &.{ D, K / 4 });
        const exps = try k.add(.uint8, &.{ D, K / 32 });
        const x = try k.add(act, &.{ R, hc, D });
        const pre = try k.add(.float32, &.{ R, hc });
        const post = try k.add(.float32, &.{ R, hc });
        const comb = try k.add(.float32, &.{ R, hc, hc });
        kit.uniform(rng, oa, -1, 1);
        kit.uniform(rng, x, -3, 3);
        kit.uniform(rng, pre, 0, 1);
        kit.uniform(rng, post, 0, 2);
        kit.uniform(rng, comb, 0, 0.5);
        for (exps) |*e| e.* = @floatFromInt(rng.intRangeAtMost(u8, 117, 121));
        for (words) |*wd| {
            var b: [4]u8 = undefined;
            rng.bytes(&b);
            for (&b) |*q| if (q.* & 0x7f == 0x7f) {
                q.* -= 1;
            };
            wd.* = @floatFromInt(std.mem.readInt(u32, &b, .little));
        }
        switch (f) {
            .random => {},
            .nonfinite => {
                kit.plantRows(oa, R, K);
                kit.plantRows(x, R, hc * D);
            },
            .stress => {
                // Mirrored integer rows (`stressDot`) against integer weights under unit scales: wo_b's sums are exact in
                // fp32 in any order and its total a small integer. The stream is 0 and post 1, comb 0, pre 1/4, so
                // every x1 stream and hv is that total.
                const wf = try k.scratch().alloc(f64, D * K);
                defer k.scratch().free(wf);
                try contract.stressDot(rng, oa[0..K], wf, D, K, 64, 7, k.scratch());
                for (1..R) |r| @memcpy(oa[r * K ..][0..K], oa[0..K]);
                var table: [8]u8 = undefined;
                for (&table, 0..) |*t, c| t.* = intByte(@intCast(c));
                for (words, 0..) |*wd, i| {
                    var b: [4]u8 = undefined;
                    for (&b, 0..) |*q, j| {
                        const c = wf[4 * i + j];
                        q.* = table[@intFromFloat(@abs(c))] | @as(u8, if (c < 0) 0x80 else 0);
                    }
                    wd.* = @floatFromInt(std.mem.readInt(u32, &b, .little));
                }
                @memset(exps, 127);
                @memset(x, 0);
                @memset(post, 1);
                @memset(comb, 0);
                @memset(pre, 0.25);
            },
        }
    }

    /// x1 then hv.
    pub fn count(_: Params, s: Shape) usize {
        return (hc + 1) * s.rows * s.n;
    }

    /// Row `r`'s wo_b column `n` as f64: the products `oa . w` into `t` (K of them), their magnitude returned.
    fn dot(k: anytype, r: usize, n: usize, t: []f64) f64 {
        const K = k.shape.k;
        var mg: f64 = 0;
        for (0..K / 32) |g| {
            const sc = std.math.pow(f64, 2, k.in(2)[n * (K / 32) + g] - 127);
            for (0..32) |q| {
                const i = g * 32 + q;
                const wd: u32 = @intFromFloat(k.in(1)[n * (K / 4) + i / 4]);
                const b: u8 = @truncate(wd >> @intCast(8 * (i % 4)));
                t[i] = k.in(0)[r * K + i] * e4m3(b) * sc;
                mg += @abs(t[i]);
            }
        }
        return mg;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const R = k.shape.rows;
        const D = k.shape.n;
        const t = try k.scratch().alloc(f64, k.shape.k);
        defer k.scratch().free(t);
        for (0..R) |r| for (0..D) |n| {
            _ = dot(k, r, n, t);
            const yb = bf(sum.of(t));
            var th: [hc]f64 = undefined;
            for (0..hc) |i| {
                var terms: [hc + 1]f64 = undefined;
                terms[0] = k.in(5)[r * hc + i] * yb;
                for (0..hc) |j| terms[j + 1] = k.in(6)[r * hc * hc + j * hc + i] * k.in(3)[(r * hc + j) * D + n];
                const x1 = bf(sum.of(&terms));
                out[(r * hc + i) * D + n] = x1;
                th[i] = k.in(4)[r * hc + i] * x1;
            }
            out[R * hc * D + r * D + n] = bf(sum.of(&th));
        };
        for (mag, out) |*m, w| m.* = @abs(w);
    }

    /// x1 within the intervals of y's f32 sums and the roundings after them; hv judged on the x1 the kernel wrote.
    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const R = k.shape.rows;
        const D = k.shape.n;
        const point = at == .nonfinite;
        const t = try k.scratch().alloc(f64, k.shape.k);
        defer k.scratch().free(t);
        for (0..R) |r| for (0..D) |n| {
            const mg = dot(k, r, n, t);
            var y: f64 = 0;
            for (t) |v| y += v;
            const yiv = if (point) Iv.pt(bf(y)) else Iv.of(y, spec.gamma(@floatFromInt(depthA(k.shape)), u) * mg + 0x1p-140).round(act);
            var h: f64 = 0;
            var hm: f64 = 0;
            for (0..hc) |i| {
                var a: f64 = 0;
                var am: f64 = 0;
                for (0..hc) |j| {
                    const term = k.in(6)[r * hc * hc + j * hc + i] * k.in(3)[(r * hc + j) * D + n];
                    a += term;
                    am += @abs(term);
                }
                const post = k.in(5)[r * hc + i];
                const iv = if (point)
                    Iv.pt(bf(post * yiv.lo + a))
                else
                    Iv.pt(post).mul(yiv).add(Iv.of(a, spec.gamma(2 * hc, u) * am)).widen(u, 0x1p-140).round(act);
                const g = outAt(k, .bf16, 0, (r * hc + i) * D + n);
                const ok = if (point) spec.classOf(g) == spec.classOf(iv.lo) else iv.has(g);
                if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} stream {d} column {d}: x1 is {e} outside [{e}, {e}]", .{ r, i, n, g, iv.lo, iv.hi }) catch "x1";
                const pre = k.in(4)[r * hc + i];
                h += pre * g;
                hm += @abs(pre * g);
            }
            const hiv = if (point) Iv.pt(bf(h)) else Iv.of(h, spec.gamma(2 * hc, u) * hm + 0x1p-140).round(act);
            const g = outAt(k, .bf16, 2 * hc * R * D, r * D + n);
            const ok = if (point) spec.classOf(g) == spec.classOf(hiv.lo) else hiv.has(g);
            if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} column {d}: hv is {e} outside [{e}, {e}]", .{ r, n, g, hiv.lo, hiv.hi }) catch "hv";
        };
        return null;
    }
};

/// The hc split of 24 mixes `z`, in float64: sigmoid pre and post, a Sinkhorn-normalized comb (`out` = pre | post | comb).
fn refSplit(z: []const f64, eps: f64, iters: usize, out: []f64) void {
    for (0..hc) |i| {
        out[i] = 1 / (1 + @exp(-z[i])) + eps;
        out[hc + i] = 2 / (1 + @exp(-z[hc + i]));
    }
    const m = out[2 * hc ..][0 .. hc * hc];
    for (0..hc) |a| {
        var mx = z[2 * hc + a * hc];
        for (1..hc) |b| mx = kit.maxN(mx, z[2 * hc + a * hc + b]);
        var sum: f64 = 0;
        for (0..hc) |b| {
            m[a * hc + b] = @exp(z[2 * hc + a * hc + b] - mx);
            sum += m[a * hc + b];
        }
        for (0..hc) |b| m[a * hc + b] = m[a * hc + b] / sum + eps;
    }
    for (0..2 * iters - 1) |t| {
        var sums: [hc]f64 = undefined;
        for (0..hc) |q| {
            var sum: f64 = 0;
            for (0..hc) |kk| sum += if (t % 2 == 1) m[q * hc + kk] else m[kk * hc + q];
            sums[q] = sum;
        }
        for (0..hc) |a| for (0..hc) |b| {
            m[a * hc + b] /= (sums[if (t % 2 == 1) a else b] + eps);
        };
    }
}

/// fp32 roundings after the mixes' sums: five operations a normalization step, then the exps and divisions around them.
const split_ops: f64 = 5 * (2 * 20 - 1) + 25;

fn softplusSqrt(z: f64) f64 {
    if (std.math.isNan(z)) return kit.nan;
    return @sqrt(@max(z, 0) + std.math.log1p(@exp(-@abs(z))));
}

/// Rounding depth along a mix's dot: a lane's 16 products a slice, the simdgroup's tree, then the slices'.
const mix_depth: f64 = 64;

/// One row's 24 mixes of the flattened stream `v` in float64, and each one's allowance over the kernel's depth: the dot's
/// gamma on its terms, the rms's half on the product, the f32 tail.
fn mixZ(v: []const f64, fnw: []const f64, scale: []const f64, base: []const f64, eps: f64, z: *[mixes]f64, allow: *[mixes]f64) void {
    const N = v.len;
    var ss: f64 = 0;
    for (v) |x| ss += x * x;
    const r = 1 / @sqrt(ss / @as(f64, @floatFromInt(N)) + eps);
    const g = spec.gamma(mix_depth, u);
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

/// The key the router picks by: a NaN score is the lowest, so it is picked last, lowest index first.
fn pickKey(v: f64) f64 {
    return if (std.math.isNan(v)) -3.4e38 else v;
}

/// The FFN's mixes and split, the rms-normed collapse and the router. Shape: rows, D, E.
const FfnB = struct {
    pub const Params = struct {};

    fn dims(s: Shape) Dims {
        return .{ .rows = s.rows, .d = s.n, .e = s.k };
    }

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        const p = dims(s);
        if (!servesB(p)) return null;
        var g = metal.Geometry{ .grid = .{ ci(gridB(p)), 1, 1 }, .tg = .{ ci(b_threads), 1, 1 } };
        g.out(&.{ ci(p.rows), ci(p.d) }, act);
        g.out(&.{ ci(p.rows), hc }, .float32);
        g.out(&.{ ci(p.rows), hc }, .float32);
        g.out(&.{ ci(p.rows), hc, hc }, .float32);
        g.out(&.{ ci(p.rows), ci(p.top_k) }, .uint32);
        g.out(&.{ ci(p.rows), ci(p.top_k) }, .float32);
        for (bTargs(p)) |t| g.int(t.name, t.v);
        g.dtype("T", act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = dims(k.shape);
        const R = p.rows;
        const D = p.d;
        const E = p.e;
        const N = hc * D;
        const x1 = try k.add(act, &.{ R, hc, D });
        const hv = try k.add(act, &.{ R, D });
        const fnw = try k.add(.float32, &.{ mixes, N });
        const scale = try k.add(.float32, &.{mixes});
        const base = try k.add(.float32, &.{mixes});
        const w = try k.add(act, &.{D});
        const gw = try k.add(act, &.{ E, D });
        const bias = try k.add(.float32, &.{ 1, E });
        const scal = try k.add(.float32, &.{1});
        const eps = try k.add(.float32, &.{2});
        _ = try k.add(.uint32, &.{16});
        _ = try k.add(.float32, &.{partsLen(p)});
        _ = try k.add(.float32, &.{ R, E });
        kit.uniform(rng, x1, -3, 3);
        kit.uniform(rng, hv, -3, 3);
        kit.uniform(rng, fnw, -0.02, 0.02);
        kit.uniform(rng, scale, 0.5, 1.5);
        kit.uniform(rng, base, -0.5, 0.5);
        kit.uniform(rng, w, 0.5, 1.5);
        kit.uniform(rng, gw, -0.05, 0.05);
        kit.uniform(rng, bias, -0.5, 0.5);
        scal[0] = router_scale;
        eps[0] = eps_rms;
        eps[1] = eps_hc;
        switch (f) {
            .random => {},
            .nonfinite => {
                kit.plantRows(x1, R, N);
                kit.plantRows(hv, R, D);
            },
            .stress => {
                // The stream +-32 on every row, hc_mix's mirrored weights (a mix's terms are +32 (2^10 + a) on the first
                // half against -32 (2^10 + b) on the second, so its total is 32 (a - b) and z = a - b with the exact
                // rsqrt of eps 0); hv 32 with unit rms so h2 is the norm weight, which carries gate-dot's mirrored
                // integers against the experts' mirrored rows. Partial sums are multiples of 32 below 2^29: exact in fp32.
                eps[0] = 0;
                @memset(scale, 1);
                @memset(base, 0);
                @memset(bias, 0);
                for (x1[0..N]) |*v| v.* = if (rng.boolean()) 32 else -32;
                for (1..R) |r| @memcpy(x1[r * N ..][0..N], x1[0..N]);
                const h = N / 2;
                for (0..mixes) |m| {
                    const a = rng.uintLessThan(usize, 4);
                    const b = rng.uintLessThan(usize, 4);
                    for (0..h) |i| {
                        fnw[m * N + i] = (0x1p10 + @as(f64, if (i < a) 1 else 0)) * x1[i] / 32;
                        fnw[m * N + h + i] = -(0x1p10 + @as(f64, if (i < b) 1 else 0)) * x1[h + i] / 32;
                    }
                }
                @memset(hv, 32);
                try contract.stressDot(rng, w, gw, E, D, 64, 7, k.scratch());
            },
        }
    }

    /// h2, pre, post, comb, then the picks and their weights.
    pub fn count(_: Params, s: Shape) usize {
        return s.rows * (s.n + 2 * hc + hc * hc + 2 * 6);
    }

    /// The router's picks by the stated rule (the biased scores, a NaN lowest, ties to the lower index) and its weights.
    fn pick(key: []const f64, score: []const f64, idx: []usize, wts: []f64) void {
        var taken: [1024]bool = @splat(false);
        for (idx) |*pk| {
            var best: usize = 0;
            var bv = -std.math.inf(f64);
            for (key, 0..) |v, e| if (!taken[e] and v > bv) {
                bv = v;
                best = e;
            };
            pk.* = best;
            taken[best] = true;
        }
        var total: f64 = 0;
        for (idx) |e| total += score[e];
        for (idx, wts) |e, *w| w.* = score[e] / (total + 1e-20) * spec.roundTo(.f32, router_scale);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const p = dims(k.shape);
        const R = p.rows;
        const D = p.d;
        const E = p.e;
        const N = hc * D;
        const a = k.scratch();
        const t = try a.alloc(f64, N);
        defer a.free(t);
        const h2 = try a.alloc(f64, D);
        defer a.free(h2);
        const key = try a.alloc(f64, E);
        defer a.free(key);
        const score = try a.alloc(f64, E);
        defer a.free(score);
        const eps0 = spec.roundTo(.f32, k.in(9)[0]);
        const eps1 = spec.roundTo(.f32, k.in(9)[1]);
        const o_pre = R * D;
        const o_post = o_pre + R * hc;
        const o_comb = o_post + R * hc;
        const o_idx = o_comb + R * hc * hc;
        const o_wts = o_idx + R * p.top_k;
        for (0..R) |r| {
            const hv = k.in(1)[r * D ..][0..D];
            for (t[0..D], hv) |*q, v| q.* = v * v;
            const inv = 1 / @sqrt(sum.of(t[0..D]) / @as(f64, @floatFromInt(D)) + eps0);
            for (h2, hv, k.in(5), 0..) |*o, v, wd, d| {
                o.* = bf(wd * bf(v * inv));
                out[r * D + d] = o.*;
            }
            const x = k.in(0)[r * N ..][0..N];
            for (t, x) |*q, v| q.* = v * v;
            const rs = 1 / @sqrt(sum.of(t) / @as(f64, @floatFromInt(N)) + eps0);
            var z: [mixes]f64 = undefined;
            for (0..mixes) |m| {
                for (t, x, 0..) |*q, v, i| q.* = v * k.in(2)[m * N + i];
                z[m] = sum.of(t) * rs * k.in(3)[m] + k.in(4)[m];
            }
            var sp: [mixes]f64 = undefined;
            refSplit(&z, eps1, p.iters, &sp);
            @memcpy(out[o_pre + r * hc ..][0..hc], sp[0..hc]);
            @memcpy(out[o_post + r * hc ..][0..hc], sp[hc .. 2 * hc]);
            @memcpy(out[o_comb + r * hc * hc ..][0 .. hc * hc], sp[2 * hc ..]);
            for (0..E) |e| {
                for (t[0..D], h2, 0..) |*q, v, d| q.* = v * k.in(6)[e * D + d];
                score[e] = softplusSqrt(sum.of(t[0..D]));
                key[e] = pickKey(score[e] + k.in(7)[e]);
            }
            var idx: [32]usize = undefined;
            var wts: [32]f64 = undefined;
            pick(key, score, idx[0..p.top_k], wts[0..p.top_k]);
            for (idx[0..p.top_k], wts[0..p.top_k], 0..) |e, w, q| {
                out[o_idx + r * p.top_k + q] = @floatFromInt(e);
                out[o_wts + r * p.top_k + q] = w;
            }
        }
        for (mag, out) |*m, w| m.* = @abs(w);
    }

    /// h2 within its roundings' intervals; the split within the mixes' propagated allowance; the picks the best of the
    /// scores of the h2 the kernel wrote, up to those scores' own error, the weights by their sums.
    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const p = dims(k.shape);
        const R = p.rows;
        const D = p.d;
        const E = p.e;
        const N = hc * D;
        const point = at == .nonfinite;
        const a = k.scratch();
        const iv = try a.alloc(Iv, D);
        defer a.free(iv);
        const zeros = try a.alloc(f64, D);
        defer a.free(zeros);
        @memset(zeros, 0);
        const h2 = try a.alloc(f64, D);
        defer a.free(h2);
        const key = try a.alloc(f64, E);
        defer a.free(key);
        const score = try a.alloc(f64, E);
        defer a.free(score);
        const tol = try a.alloc(f64, E);
        defer a.free(tol);
        const eps0 = spec.roundTo(.f32, k.in(9)[0]);
        const eps1 = spec.roundTo(.f32, k.in(9)[1]);
        const o_pre = 2 * R * D;
        const o_post = o_pre + 4 * R * hc;
        const o_comb = o_post + 4 * R * hc;
        const o_idx = o_comb + 4 * R * hc * hc;
        const o_wts = o_idx + 4 * R * p.top_k;
        for (0..R) |r| {
            contract.collapseNorm(k.in(1)[r * D ..][0..D], zeros, k.in(5), eps0, point, iv);
            for (iv, h2, 0..) |q, *o, d| {
                o.* = outAt(k, .bf16, 0, r * D + d);
                const ok = if (point) spec.classOf(o.*) == spec.classOf(q.lo) else q.has(o.*);
                if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} h2 {d} is {e} outside [{e}, {e}]", .{ r, d, o.*, q.lo, q.hi }) catch "h2";
            }
            var z: [mixes]f64 = undefined;
            var allow: [mixes]f64 = undefined;
            mixZ(k.in(0)[r * N ..][0..N], k.in(2), k.in(3), k.in(4), eps0, &z, &allow);
            var want: [mixes]f64 = undefined;
            refSplit(&z, eps1, p.iters, &want);
            var slack: [mixes]f64 = undefined;
            for (&slack, want) |*s, v| s.* = split_ops * u * @abs(v) + 0x1p-140;
            if (!point) for (0..mixes) |m| {
                // First order: each mix's own allowance moved through the split.
                const step = @max(allow[m], 1e-9);
                var zz = z;
                zz[m] += step;
                var alt: [mixes]f64 = undefined;
                refSplit(&zz, eps1, p.iters, &alt);
                for (&slack, alt, want) |*s, v, w| s.* += 2 * @abs(v - w) * allow[m] / step;
            };
            for (want, slack, 0..) |w, s, o| {
                const g = if (o < hc) outAt(k, .f32, o_pre, r * hc + o) else if (o < 2 * hc) outAt(k, .f32, o_post, r * hc + o - hc) else outAt(k, .f32, o_comb, r * hc * hc + o - 2 * hc);
                const ok = if (point) spec.classOf(g) == spec.classOf(w) else @abs(g - w) <= s;
                if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} split output {d} is {e} where float64 gives {e}", .{ r, o, g, w }) catch "split";
            }
            // The router: logits from the h2 the kernel wrote; a score's error is the dot's, carried by the score's slope (<= 0.6).
            for (0..E) |e| {
                var acc: f64 = 0;
                var mg: f64 = 0;
                for (h2, 0..) |v, d| {
                    const term = v * k.in(6)[e * D + d];
                    acc += term;
                    mg += @abs(term);
                }
                score[e] = softplusSqrt(acc);
                key[e] = pickKey(score[e] + k.in(7)[e]);
                tol[e] = 0.6 * spec.gamma(@floatFromInt(D / 32 + 8), u) * mg + 16 * u * (@abs(score[e]) + 1);
            }
            var taken: [1024]bool = @splat(false);
            var picked: [32]usize = undefined;
            for (0..p.top_k) |q| {
                const g: usize = @intCast(std.mem.readInt(u32, k.rig.outputBytes()[o_idx + 4 * (r * p.top_k + q) ..][0..4], .little));
                if (g >= E or taken[g]) return std.fmt.bufPrint(&k.msg, "row {d} pick {d} is {d}: out of range or repeated", .{ r, q, g }) catch "pick";
                var best: usize = E;
                for (0..E) |e| if (!taken[e] and (best == E or key[e] > key[best])) {
                    best = e;
                };
                if (key[best] <= -3.3e38) {
                    // Only NaN scores are left: the lowest index first.
                    var lowest: usize = 0;
                    while (taken[lowest]) lowest += 1;
                    if (g != lowest) return std.fmt.bufPrint(&k.msg, "row {d} pick {d} is {d}: NaN scores go by index, {d} was next", .{ r, q, g, lowest }) catch "pick";
                } else if (!(key[g] >= key[best] - tol[g] - tol[best]))
                    return std.fmt.bufPrint(&k.msg, "row {d} pick {d} is expert {d} at {e}; {d} at {e} was left", .{ r, q, g, key[g], best, key[best] }) catch "pick";
                taken[g] = true;
                picked[q] = g;
            }
            var total: f64 = 0;
            for (picked[0..p.top_k]) |e| total += score[e];
            const div = if (p.norm) total + 1e-20 else 1;
            for (picked[0..p.top_k], 0..) |e, q| {
                const w = score[e] / div * spec.roundTo(.f32, router_scale);
                const g = outAt(k, .f32, o_wts, r * p.top_k + q);
                const ok = if (std.math.isFinite(w)) @abs(g - w) <= 4 * (tol[e] / @max(score[e], 0.5) + 32 * u) * @abs(w) + 0x1p-140 else spec.classOf(g) == spec.classOf(w);
                if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} weight {d} is {e} where float64 gives {e}", .{ r, q, g, w }) catch "weight";
            }
        }
        return null;
    }
};

// A launch of one of the two kernels as the contract builds it: threadgroup bytes at the widest instantiation, M1/M2
// thread limits from `kernel contract offline`.
const launch_a = metal.Launch{
    .name = "dsv41_dec_ffn_in",
    .inputs = &A_INS,
    .outputs = &A_OUTS,
    .source = A_SOURCE,
    .header = HEADER,
    .tg_bytes = 4 * decode.max_rows * a_nsg * 2,
    .m1 = 512,
    .m2 = 512,
};
const launch_b = metal.Launch{
    .name = "dsv41_dec_ffn_in_post",
    .inputs = &B_INS,
    .outputs = &B_OUTS,
    .source = B_SOURCE,
    .tg_bytes = 4 * 7728 + 4,
    .m1 = 704,
    .m2 = 704,
};

const nan_rule: spec.Nan = .{ .stated = "classes as float64 gives them; every output within the intervals of its roundings, the picks by the stated rule (a NaN score is picked last, lowest index first)" };

pub fn register(set: *kc.Set) !void {
    try kit.add(set, FfnA, &launch_a, .{}, .{
        .name = "dsv41 dec ffn_in A (wo_b, expand, collapse)",
        .accum = .f32,
        .depth = depthA,
        .mids = 1,
        .out = .bf16,
        .nan = nan_rule,
        .tiles = .{ .{ .size = 3 }, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 1024, .step = 1024, .min = 1024 } },
        .served = &.{ .{ .rows = 1, .n = 256, .k = 8192 }, .{ .rows = 6, .n = 256, .k = 8192 }, .{ .rows = 1, .n = 5120, .k = 8192 } },
    });
    try kit.add(set, FfnB, &launch_b, .{}, .{
        .name = "dsv41 dec ffn_in B (mixes, split, norm, router)",
        .accum = .f32,
        .depth = depthB,
        .ops = split_ops,
        .mids = 1,
        .out = .bf16,
        .outs = &.{
            .{ .elem = .bf16, .count = perD },
            .{ .elem = .f32, .count = perHc },
            .{ .elem = .f32, .count = perHc },
            .{ .elem = .f32, .count = perComb },
            .{ .elem = .u32, .count = perPick },
            .{ .elem = .f32, .count = perPick },
        },
        .in_place = true,
        .nan = nan_rule,
        .tiles = .{ .{ .size = 3 }, .{}, .{ .size = 16, .step = 8, .min = 8 } },
        .served = &.{ .{ .rows = 1, .n = 5120, .k = 384 }, .{ .rows = 6, .n = 5120, .k = 384 } },
    });
}

// ── tests ──

const Fixture = @import("fixture.zig").Fixture;

pub fn weightsOf(ly: *const dsv41.Layer) Weights {
    return .{ .wo_w = ly.wo_b.w, .wo_s = ly.wo_b.s, .fn_w = ly.hc_ffn.fn_w, .fn_scale = ly.hc_ffn.scale, .fn_base = ly.hc_ffn.base, .norm_w = ly.ffn_norm, .gate_w = ly.gate_w, .gate_bias = ly.gate_bias };
}

/// `ly` with `w` in the places today's chain reads them.
fn layerWith(ly: *const dsv41.Layer, w: Weights) dsv41.Layer {
    var l = ly.*;
    l.wo_b.w = w.wo_w;
    l.wo_b.s = w.wo_s;
    l.hc_ffn = .{ .fn_w = w.fn_w, .scale = w.fn_scale, .base = w.fn_base };
    l.ffn_norm = w.norm_w;
    l.gate_w = w.gate_w;
    l.gate_bias = w.gate_bias;
    return l;
}

/// Float64 truth of the launch pair from the f32 views of its inputs: no intermediate rounding anywhere.
const Truth = struct {
    gpa: std.mem.Allocator,
    e: usize,
    x1: []f32,
    h2: []f32,
    sf_pre: []f32,
    sf_post: []f32,
    sf_comb: []f32,
    /// Per row: the router's scores (sqrt-softplus of the logit) and their biased values.
    score: []f64,
    biased: []f64,

    fn deinit(t: *Truth) void {
        inline for (.{ t.x1, t.h2, t.sf_pre, t.sf_post, t.sf_comb }) |a| t.gpa.free(a);
        t.gpa.free(t.score);
        t.gpa.free(t.biased);
    }

    /// Row `r`'s top-k expert ids by biased score (ties to the lower id).
    fn picks(t: *const Truth, r: usize, out: []usize) void {
        var taken: [1024]bool = @splat(false);
        for (out) |*o| {
            var best: usize = 0;
            var bv = -std.math.inf(f64);
            for (0..t.e) |ex| if (!taken[ex] and t.biased[r * t.e + ex] > bv) {
                bv = t.biased[r * t.e + ex];
                best = ex;
            };
            o.* = best;
            taken[best] = true;
        }
    }

    /// The routed weights of row `r` for the experts `ids`, as the router normalizes and scales them.
    fn weights(t: *const Truth, r: usize, ids: []const usize, scal: f64, norm: bool, out: []f32) void {
        var sum: f64 = 0;
        for (ids) |ex| sum += t.score[r * t.e + ex];
        const div = if (norm) sum + 1e-20 else 1.0;
        for (ids, out) |ex, *o| o.* = @floatCast(t.score[r * t.e + ex] / div * scal);
    }
};

const HostIn = struct {
    p: Dims,
    /// wo_b dequantized `[D, K]`, the activations, the stream, the attention's split.
    wo: []const f32,
    oa: []const f32,
    x: []const f32,
    pre: []const f32,
    post: []const f32,
    comb: []const f32,
    fnw: []const f32,
    scale: []const f32,
    base: []const f32,
    norm_w: []const f32,
    gate_w: []const f32,
    bias: []const f32,
    eps0: f64,
    eps1: f64,
};

fn truthOf(gpa: std.mem.Allocator, h: HostIn) !Truth {
    const R = h.p.rows;
    const D = h.p.d;
    const N = hc * D;
    const t = Truth{
        .gpa = gpa,
        .e = h.p.e,
        .x1 = try gpa.alloc(f32, R * N),
        .h2 = try gpa.alloc(f32, R * D),
        .sf_pre = try gpa.alloc(f32, R * hc),
        .sf_post = try gpa.alloc(f32, R * hc),
        .sf_comb = try gpa.alloc(f32, R * hc * hc),
        .score = try gpa.alloc(f64, R * h.p.e),
        .biased = try gpa.alloc(f64, R * h.p.e),
    };
    const y = try gpa.alloc(f64, D);
    defer gpa.free(y);
    const x1 = try gpa.alloc(f64, N);
    defer gpa.free(x1);
    const hv = try gpa.alloc(f64, D);
    defer gpa.free(hv);
    const h2 = try gpa.alloc(f64, D);
    defer gpa.free(h2);
    for (0..R) |r| {
        for (0..D) |n| {
            var acc: f64 = 0;
            for (0..h.p.k) |k| acc += @as(f64, h.wo[n * h.p.k + k]) * h.oa[r * h.p.k + k];
            y[n] = acc;
        }
        for (0..hc) |i| for (0..D) |d| {
            var acc: f64 = @as(f64, h.post[r * hc + i]) * y[d];
            for (0..hc) |j| acc += @as(f64, h.comb[r * hc * hc + j * hc + i]) * h.x[(r * hc + j) * D + d];
            x1[i * D + d] = acc;
            t.x1[r * N + i * D + d] = @floatCast(acc);
        };
        var ss: f64 = 0;
        for (0..D) |d| {
            var acc: f64 = 0;
            for (0..hc) |j| acc += @as(f64, h.pre[r * hc + j]) * x1[j * D + d];
            hv[d] = acc;
            ss += acc * acc;
        }
        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(D)) + h.eps0);
        for (0..D) |d| {
            h2[d] = @as(f64, h.norm_w[d]) * hv[d] * inv;
            t.h2[r * D + d] = @floatCast(h2[d]);
        }
        var sq: f64 = 0;
        for (x1) |v| sq += v * v;
        const rs = 1 / @sqrt(sq / @as(f64, @floatFromInt(N)) + h.eps0);
        var z: [mixes]f64 = undefined;
        for (0..mixes) |m| {
            var acc: f64 = 0;
            for (x1, 0..) |v, i| acc += v * h.fnw[m * N + i];
            z[m] = acc * rs * h.scale[m] + h.base[m];
        }
        var sp: [mixes]f64 = undefined;
        refSplit(&z, h.eps1, h.p.iters, &sp);
        for (sp[0..hc], t.sf_pre[r * hc ..][0..hc]) |v, *o| o.* = @floatCast(v);
        for (sp[hc .. 2 * hc], t.sf_post[r * hc ..][0..hc]) |v, *o| o.* = @floatCast(v);
        for (sp[2 * hc ..], t.sf_comb[r * hc * hc ..][0 .. hc * hc]) |v, *o| o.* = @floatCast(v);
        for (0..h.p.e) |ex| {
            var acc: f64 = 0;
            for (0..D) |d| acc += h2[d] * h.gate_w[ex * D + d];
            const s = softplusSqrt(acc);
            t.score[r * h.p.e + ex] = s;
            t.biased[r * h.p.e + ex] = s + h.bias[ex];
        }
    }
    return t;
}

/// One layer's launch pair against today's `wo_b`, `hcExpandPreNorm` and `route` on the same inputs, `ly` holding `w`.
fn equivalence(gpa: std.mem.Allocator, fx: *Fixture, ks: Kernels, ly: *const dsv41.Layer, w: Weights, rows: usize) !void {
    const m = fx.m;
    const s = mlx.gpuStream();
    var sc = dsv41.Scope.init(gpa, s);
    defer sc.deinit();
    const D = m.cfg.hidden_size;
    const oa = try fx.normal(&.{ ci(rows), 8192 }, act, 0, 1);
    defer _ = mlx.mlx_array_free(oa);
    const x = try fx.stream(rows);
    defer _ = mlx.mlx_array_free(x);
    const cpre = try fx.cpre(rows);
    defer _ = mlx.mlx_array_free(cpre);
    const sa = (try dsv41.hcStage(&sc, m, x, null, cpre, &ly.hc_attn, ly.attn_norm)).sp;
    // Today's chain.
    const y = try ly.wo_b.apply(&sc, oa);
    const pf = try dsv41.hcExpandPreNorm(&sc, m, x, y, &sa, sa.pre, &ly.hc_ffn, ly.ffn_norm);
    const rt = try dsv41.route(&sc, m, ly, pf.h, none);
    // The launch pair.
    const eps = try dsv41.hcEps(&sc, m);
    const scal = mlx.mlx_array_new_data(&[_]f32{m.cfg.router_scaling_factor}, &[_]c_int{1}, 1, .float32);
    try sc.items.append(sc.a, scal);
    const p: Dims = .{ .rows = rows, .d = D, .e = @intCast(mlx.getShape(ly.gate_w)[0]), .top_k = ly.top_k, .iters = m.cfg.dsv4_hc_sinkhorn_iters, .norm = m.cfg.moe_route_norm and ly.top_k > 1 };
    const out = (try launch(&sc, ks, p, w, oa, x, &sa, eps, scal)) orelse return error.Declined;
    try dsv41.evalAll(&.{ pf.x1, pf.h, pf.sp.pre, pf.sp.post, pf.sp.comb, rt.idx, rt.wts, out.x1, out.h2, out.sf.pre, out.sf.post, out.sf.comb, out.idx, out.wts });
    // Float64 truth over the dequantized wo_b.
    var wd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wd);
    try mlx.check(mlx.mlx_dequantize(&wd, w.wo_w, w.wo_s, none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, .{ .value = .float32, .has_value = true }, s));
    var hosts: [12][]f32 = undefined;
    const srcs = [_]A{ wd, oa, x, sa.pre, sa.post, sa.comb, w.fn_w, w.fn_scale, w.fn_base, w.norm_w, w.gate_w, w.gate_bias };
    for (srcs, &hosts) |a, *hs| hs.* = try dsv41.hostF32(a);
    defer for (hosts) |hs| std.testing.allocator.free(hs);
    var truth = try truthOf(gpa, .{
        .p = p,
        .wo = hosts[0],
        .oa = hosts[1],
        .x = hosts[2],
        .pre = hosts[3],
        .post = hosts[4],
        .comb = hosts[5],
        .fnw = hosts[6],
        .scale = hosts[7],
        .base = hosts[8],
        .norm_w = hosts[9],
        .gate_w = hosts[10],
        .bias = hosts[11],
        .eps0 = m.cfg.rms_norm_eps,
        .eps1 = m.cfg.dsv4_hc_eps,
    });
    defer truth.deinit();
    const names = [_][]const u8{ "x1", "h2", "sf.pre", "sf.post", "sf.comb" };
    const chain_a = [_]A{ pf.x1, pf.h, pf.sp.pre, pf.sp.post, pf.sp.comb };
    const new_a = [_]A{ out.x1, out.h2, out.sf.pre, out.sf.post, out.sf.comb };
    const truth_a = [_][]const f32{ truth.x1, truth.h2, truth.sf_pre, truth.sf_post, truth.sf_comb };
    for (names, chain_a, new_a, truth_a) |name, ca, na, tv| {
        const ch = try dsv41.hostF32(ca);
        defer std.testing.allocator.free(ch);
        const nw = try dsv41.hostF32(na);
        defer std.testing.allocator.free(nw);
        var label: [64]u8 = undefined;
        try decode.expectNoWorse(try std.fmt.bufPrint(&label, "ffn_in R={d} E={d} {s}", .{ rows, p.e, name }), nw, ch, tv);
    }
    // Routes: equal to the chain's except at true near-ties of the biased scores.
    const chain_idx = try dsv41.hostF32(rt.idx);
    defer std.testing.allocator.free(chain_idx);
    const new_idx = try dsv41.hostF32(out.idx);
    defer std.testing.allocator.free(new_idx);
    const chain_wts = try dsv41.hostF32(rt.wts);
    defer std.testing.allocator.free(chain_wts);
    const new_wts = try dsv41.hostF32(out.wts);
    defer std.testing.allocator.free(new_wts);
    const K = p.top_k;
    var differ: usize = 0;
    const truth_wts = try gpa.alloc(f32, rows * K);
    defer gpa.free(truth_wts);
    var ids: [16]usize = undefined;
    for (0..rows) |r| {
        if (!std.mem.eql(f32, chain_idx[r * K ..][0..K], new_idx[r * K ..][0..K])) differ += 1;
        for (new_idx[r * K ..][0..K], ids[0..K]) |v, *o| o.* = @intFromFloat(v);
        truth.weights(r, ids[0..K], m.cfg.router_scaling_factor, p.norm, truth_wts[r * K ..][0..K]);
    }
    std.debug.print("[dec] ffn_in R={d} E={d}: {d} of {d} rows pick differently from the chain\n", .{ rows, p.e, differ, rows });
    if (differ == 0) try decode.expectNoWorse("ffn_in wts", new_wts, chain_wts, truth_wts);
    try std.testing.expect(differ <= 1);
}

test "dsv41 dec ffn_in: R = 1, 2, 4, 6 on layers 20..25 match today's wo_b, expand, norm, mixes and router, no further from float64" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .ctx = 8 });
    defer fx.deinit();
    const ks = try build();
    defer free(ks);
    for ([_]usize{ 21, 23 }) |li| for ([_]usize{ 1, 2, 4, 6 }) |rows| {
        const ly = fx.layer(li);
        try equivalence(gpa, fx, ks, ly, weightsOf(ly), rows);
    };
}

fn quant(w: A) ![2]A {
    defer _ = mlx.mlx_array_free(w);
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, w, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, mlx.gpuStream()));
    var out: [2]A = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    for (&out, 0..) |*a, i| try mlx.check(mlx.mlx_vector_array_get(a, qv, i));
    try dsv41.evalAll(&out);
    return out;
}

fn normal(seed: u64, shape: []const c_int, dt: mlx.mlx_dtype, mean: f32, std_: f32) !A {
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    try mlx.check(mlx.mlx_random_key(&k, seed));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&o, shape.ptr, shape.len, dt, mean, std_, k, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

/// The release's shapes over random weights: wo_b `[5120, 8192]` as mxfp8, E = 384 experts.
fn randomWeights(seed: u64) !Weights {
    const q = try quant(try normal(seed, &.{ 5120, 8192 }, act, 0, 0.008));
    return .{
        .wo_w = q[0],
        .wo_s = q[1],
        .fn_w = try normal(seed + 1, &.{ mixes, hc * 5120 }, .float32, 0, 0.01),
        .fn_scale = try normal(seed + 2, &.{mixes}, .float32, 0.5, 0.1),
        .fn_base = try normal(seed + 3, &.{mixes}, .float32, 0, 0.5),
        .norm_w = try normal(seed + 4, &.{5120}, act, 1, 0.1),
        .gate_w = try normal(seed + 5, &.{ 384, 5120 }, act, 0, 0.014),
        .gate_bias = try normal(seed + 6, &.{ 1, 384 }, .float32, 0, 0.05),
    };
}

fn freeWeights(w: Weights) void {
    for ([_]A{ w.wo_w, w.wo_s, w.fn_w, w.fn_scale, w.fn_base, w.norm_w, w.gate_w, w.gate_bias }) |a| _ = mlx.mlx_array_free(a);
}

test "dsv41 dec ffn_in: R = 1 and 6 at the release's 384 experts match today's chain, no further from float64" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .ctx = 8 });
    defer fx.deinit();
    const ks = try build();
    defer free(ks);
    const w = try randomWeights(77);
    defer freeWeights(w);
    const ly = layerWith(fx.layer(21), w);
    for ([_]usize{ 1, 6 }) |rows| try equivalence(gpa, fx, ks, &ly, w, rows);
}

/// `a` and `b` hold the same bytes.
fn sameBytes(rig: *metal.Rig, a: A, b: A) !bool {
    const first = try rig.a.dupe(u8, try rig.bytesOf(&.{a}));
    defer rig.a.free(first);
    return std.mem.eql(u8, first, try rig.bytesOf(&.{b}));
}

test "dsv41 dec ffn_in: the contract's launches write the shipped wrapper's bytes" {
    if (mlx.virtualGpu()) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const ks = try build();
    defer free(ks);
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var ca = kit.Case(FfnA).init(gpa, &launch_a, .{});
    defer ca.deinit();
    try std.testing.expect(try ca.prepare(.{ .rows = 3, .n = 256, .k = 8192 }, .random, 7));
    try ca.run();
    const pa: Dims = .{ .rows = 3, .d = 256 };
    const sa: dsv41.Split = .{ .pre = ca.args[4], .post = ca.args[5], .comb = ca.args[6] };
    const wa: Weights = .{ .wo_w = ca.args[1], .wo_s = ca.args[2], .fn_w = none, .fn_scale = none, .fn_base = none, .norm_w = none, .gate_w = none, .gate_bias = none };
    const a = try launchA(&sc, ks, pa, wa, ca.args[0], ca.args[3], &sa);
    try std.testing.expect(try sameBytes(&ca.rig, a.x1, ca.rig.output(0)));
    try std.testing.expect(try sameBytes(&ca.rig, a.hv, ca.rig.output(1)));
    var cb = kit.Case(FfnB).init(gpa, &launch_b, .{});
    defer cb.deinit();
    try std.testing.expect(try cb.prepare(.{ .rows = 3, .n = 5120, .k = 384 }, .random, 9));
    try cb.run();
    const pb: Dims = .{ .rows = 3 };
    const wb: Weights = .{ .wo_w = none, .wo_s = none, .fn_w = cb.args[2], .fn_scale = cb.args[3], .fn_base = cb.args[4], .norm_w = cb.args[5], .gate_w = cb.args[6], .gate_bias = cb.args[7] };
    const ab: AOut = .{ .x1 = cb.args[0], .hv = cb.args[1], .ctr = cb.args[10], .parts = cb.args[11], .logits = cb.args[12] };
    const b = try launchB(&sc, ks, pb, wb, ab, cb.args[9], cb.args[8]);
    const got = [_]A{ b.h2, b.sf.pre, b.sf.post, b.sf.comb, b.idx, b.wts };
    for (got, 0..) |g, i| try std.testing.expect(try sameBytes(&cb.rig, g, cb.rig.output(i)));
}

// ── ubench ──

const Timing = struct { total: f64, build: f64 };

/// Today's chain or the launch pair, one step a layer's worth of weights from `sets`.
const Bench = struct {
    gpa: std.mem.Allocator,
    rows: usize,
    m: *dsv41.Dsv41Model,
    ks: Kernels,
    sets: []const Weights,
    layers: []const dsv41.Layer,
    x0: A,
    sa0: dsv41.Split,
    eps: A,
    scal: A,
    mode: enum { new, chain, launch_1 },
    prev: A = undefined,
    sa: dsv41.Split = undefined,
    sc: ?dsv41.Scope = null,
    /// Outputs nothing downstream reads, evaluated with the chain's last.
    extra: std.ArrayList(A) = .empty,

    fn deinit(b: *Bench) void {
        if (b.sc) |*sc| sc.deinit();
        b.extra.deinit(b.gpa);
    }

    /// The previous stream, read as the next step's attention output `[R, 8192]`.
    fn attnOut(b: *Bench, sc: *dsv41.Scope) !A {
        const s = sc.s;
        var flat = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&flat, b.prev, &[_]c_int{-1}, 1, s));
        try sc.items.append(sc.a, flat);
        var part = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&part, flat, &[_]c_int{0}, 1, &[_]c_int{ci(b.rows * 8192)}, 1, &[_]c_int{1}, 1, s));
        try sc.items.append(sc.a, part);
        var oa = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&oa, part, &[_]c_int{ ci(b.rows), 8192 }, 2, s));
        try sc.items.append(sc.a, oa);
        return oa;
    }

    fn step(b: *Bench, i: usize) !A {
        if (i == 0) {
            if (b.sc) |*sc| sc.deinit();
            b.sc = dsv41.Scope.init(b.gpa, mlx.gpuStream());
            b.extra.clearRetainingCapacity();
            b.prev = b.x0;
            b.sa = b.sa0;
        }
        const sc = &b.sc.?;
        const oa = try b.attnOut(sc);
        const set = b.sets[i % b.sets.len];
        const p: Dims = .{ .rows = b.rows };
        switch (b.mode) {
            .launch_1 => {
                const a = try launchA(sc, b.ks, p, set, oa, b.prev, &b.sa0);
                b.prev = a.x1;
                return a.x1;
            },
            .new => {
                const out = (try launch(sc, b.ks, p, set, oa, b.prev, &b.sa, b.eps, b.scal)).?;
                try b.extra.appendSlice(b.gpa, &.{ out.idx, out.wts });
                b.prev = out.x1;
                b.sa = out.sf;
                return out.sf.comb;
            },
            .chain => {
                const ly = &b.layers[i % b.layers.len];
                const y = try ly.wo_b.apply(sc, oa);
                const pf = try dsv41.hcExpandPreNorm(sc, b.m, b.prev, y, &b.sa, b.sa.pre, &ly.hc_ffn, ly.ffn_norm);
                const rt = try dsv41.route(sc, b.m, ly, pf.h, none);
                try b.extra.appendSlice(b.gpa, &.{ rt.idx, rt.wts });
                b.prev = pf.x1;
                b.sa = pf.sp;
                return pf.sp.comb;
            },
        }
    }

    /// Microseconds a step of a dependent chain of `n` takes, the graph build apart (medians of 7 after 2 warm-ups, as `decode.chainUs`).
    fn us(b: *Bench, n: usize) !Timing {
        const io = std.Io.Threaded.global_single_threaded.io();
        const iou = @import("../../../sdk/util/io_util.zig");
        var total: [7]f64 = undefined;
        var built: [7]f64 = undefined;
        for (0..2 + total.len) |rep| {
            const t0 = iou.nowUsMonotonic(io);
            var last: A = .{ .ctx = null };
            for (0..n) |i| last = try b.step(i);
            const t1 = iou.nowUsMonotonic(io);
            try b.extra.append(b.gpa, last);
            const vec = mlx.mlx_vector_array_new_data(b.extra.items.ptr, b.extra.items.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_eval(vec));
            const t2 = iou.nowUsMonotonic(io);
            if (rep >= 2) {
                total[rep - 2] = @as(f64, @floatFromInt(t2 - t0)) / @as(f64, @floatFromInt(n));
                built[rep - 2] = @as(f64, @floatFromInt(t1 - t0)) / @as(f64, @floatFromInt(n));
            }
        }
        std.mem.sort(f64, &total, {}, std.sort.asc(f64));
        std.mem.sort(f64, &built, {}, std.sort.asc(f64));
        return .{ .total = total[total.len / 2], .build = built[built.len / 2] };
    }
};

test "dsv41 dec ffn_in: ubench, R = 1 and 6 against today's chain over 2.3 GB of distinct weights" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (!decode.ubenchOn()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .ctx = 8 });
    defer fx.deinit();
    const ks = try build();
    defer free(ks);
    const nsets = 48;
    var sets: [nsets]Weights = undefined;
    var layers: [nsets]dsv41.Layer = undefined;
    for (&sets, &layers, 0..) |*w, *l, i| {
        w.* = try randomWeights(1000 + 10 * i);
        l.* = layerWith(fx.layer(21), w.*);
    }
    defer for (sets) |w| freeWeights(w);
    const s = mlx.gpuStream();
    var sc = dsv41.Scope.init(gpa, s);
    defer sc.deinit();
    const eps = try dsv41.hcEps(&sc, fx.m);
    const scal = mlx.mlx_array_new_data(&[_]f32{fx.m.cfg.router_scaling_factor}, &[_]c_int{1}, 1, .float32);
    try sc.items.append(sc.a, scal);
    for ([_]usize{ 1, 6 }) |rows| {
        const x = try fx.stream(rows);
        defer _ = mlx.mlx_array_free(x);
        const cpre = try fx.cpre(rows);
        defer _ = mlx.mlx_array_free(cpre);
        const sa0 = (try dsv41.hcStage(&sc, fx.m, x, null, cpre, &fx.layer(21).hc_attn, fx.layer(21).attn_norm)).sp;
        var us: [3]Timing = undefined;
        for (&us, [_]@TypeOf(@as(Bench, undefined).mode){ .new, .chain, .launch_1 }) |*t, mode| {
            var b = Bench{ .gpa = gpa, .rows = rows, .m = fx.m, .ks = ks, .sets = &sets, .layers = &layers, .x0 = x, .sa0 = sa0, .eps = eps, .scal = scal, .mode = mode };
            defer b.deinit();
            t.* = try b.us(nsets);
        }
        decode.report("ffn_in", rows, us[0].total, 49.2e6, us[1].total);
        std.debug.print("[dec-ubench] ffn_in R={d}: graph build {d:.1} us (chain {d:.1}); launch 1 alone {d:.1} us\n", .{ rows, us[0].build, us[1].build, us[2].total });
    }
}
