//! DeepSeek-V4.1 decode, the FFN's entry in two launches. Launch 1: wo_b over the attention output, then each output
//! column's hc expand and collapse. Launch 2: the FFN's 24 hc mixes with their Sinkhorn split, the rms-normed collapse
//! and the router with its top K. Rows 1..6, every weight read once whatever the rows.

const std = @import("std");
const device = @import("../../../mlx/device.zig");
const env = @import("../../../env.zig");
const lev = @import("../levers.zig");
const mlx = @import("../../../mlx/ffi.zig");
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
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
const HEADER = decode.FP8_READ_HEADER;
/// `MLX_SERVE_DSV41_FFN_IN_SINK`: launch 2 without its mix tail, and that tail as its own launch.
const SINK_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_ffn_in_sink.metal");
const B_APART_HEADER = "#define DSV41_FFN_SINK_APART 1\n";

const A_INS = [_][*:0]const u8{ "oa", "w_q", "scales", "x", "spre", "spost", "scomb" };
/// `ctr`, `parts` and `scores` are launch 2's scratch: launch 1 zeroes the counters and writes neither of the others.
const A_OUTS = [_][*:0]const u8{ "x1", "hv", "ctr", "parts", "logits" };
const B_INS = [_][*:0]const u8{ "x1", "hv", "fn", "scale", "base", "w", "gate_w", "bias", "scal", "eps", "ctr", "parts", "logits" };
const B_OUTS = [_][*:0]const u8{ "h2", "pre", "post", "comb", "idx", "wts" };
const B_APART_OUTS = [_][*:0]const u8{ "h2", "idx", "wts" };
const SINK_INS = [_][*:0]const u8{ "parts", "scale", "base", "eps", "order" };
const SINK_OUTS = [_][*:0]const u8{ "pre", "post", "comb" };

pub const Kernels = struct {
    a: mlx.mlx_fast_metal_kernel,
    b: mlx.mlx_fast_metal_kernel,
    /// `MLX_SERVE_DSV41_FFN_IN_SINK`: launch 2 without its mix tail, and the tail's own launch.
    b_apart: mlx.mlx_fast_metal_kernel,
    sink: mlx.mlx_fast_metal_kernel,
};

/// The launches' kernels, built once.
pub fn build() !Kernels {
    const a = try dsv41.buildKernel("dsv41_dec_ffn_in", &A_INS, &A_OUTS, A_SOURCE, HEADER);
    errdefer _ = mlx.mlx_fast_metal_kernel_free(a);
    const b = try dsv41.buildKernel("dsv41_dec_ffn_in_post", &B_INS, &B_OUTS, B_SOURCE, "");
    errdefer _ = mlx.mlx_fast_metal_kernel_free(b);
    const ba = try dsv41.buildKernel("dsv41_dec_ffn_in_post_apart", &B_INS, &B_APART_OUTS, B_SOURCE, B_APART_HEADER);
    errdefer _ = mlx.mlx_fast_metal_kernel_free(ba);
    const sk = try dsv41.buildKernel("dsv41_dec_ffn_in_sink", &SINK_INS, &SINK_OUTS, SINK_SOURCE, "");
    errdefer _ = mlx.mlx_fast_metal_kernel_free(sk);
    return .{ .a = a, .b = b, .b_apart = ba, .sink = sk };
}

pub fn free(k: Kernels) void {
    _ = mlx.mlx_fast_metal_kernel_free(k.a);
    _ = mlx.mlx_fast_metal_kernel_free(k.b);
    _ = mlx.mlx_fast_metal_kernel_free(k.b_apart);
    _ = mlx.mlx_fast_metal_kernel_free(k.sink);
}

/// `MLX_SERVE_DSV41_FFN_IN_SINK` (read each call): =1 launch 2's mix tail (the slices' reduce and the Sinkhorn split)
/// runs as its own one-threadgroup launch, read only by the MoE down, so the MoE up does not wait on it.
pub fn sinkOn() bool {
    return lev.onUnlessZero(.FFN_IN_SINK);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

// ── launch geometry ──

/// Launch 1: simdgroups a threadgroup, K steps loaded ahead, and the output columns a simdgroup: two once the rows' x
/// reads outweigh the weights.
const a_nsg: usize = 8;
const a_unroll: usize = 2;
/// `MLX_SERVE_DSV41_FFN_IN_R5_COLS1=1` (read each call) keeps 5 rows at one column: a column's sums are the same
/// either way (each column's lanes, steps and simd sum do not depend on the plan), so the bytes are too.
fn aCols(rows: usize) usize {
    if (cols_test) |t| return t[rows];
    if (rows == 5 and lev.onUnlessZero(.FFN_IN_R5_COLS1)) {
        if (!r5_logged.swap(true, .monotonic)) @import("../../../sdk/util/log.zig").info(lev.line(.FFN_IN_R5_COLS1) ++ ": ffn_in launch 1 at 5 rows on one column a simdgroup\n", .{});
        return 1;
    }
    return if (rows > 4) 2 else 1;
}
var r5_logged = std.atomic.Value(bool).init(false);
/// The ubench's and tests' plan per row count (null: `aCols`'s).
var cols_test: ?[decode.max_rows + 1]usize = null;
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

pub const Out = struct { x1: A, h2: A, sf: dsv41.Split, idx: A, wts: A, sink: bool = false };
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
    }, gridA(p), a_nsg * 32, &(aTargs(p) ++ [_]TArg{.{ .name = "FOLD", .v = @intCast(decode.fp8Fold()) }}));
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

/// `launchB` as two launches (`MLX_SERVE_DSV41_FFN_IN_SINK`): launch 2 without its mix tail (`DSV41_FFN_SINK_APART`:
/// h2, the picks and weights), then the mix tail's own launch over the partials launch 2 left in `a.parts` (the split).
/// The same words as `launchB`'s.
pub fn launchBApart(sc: *dsv41.Scope, ks: Kernels, p: Dims, w: Weights, a: AOut, eps: A, scal: A) !BOut {
    const b = try run(3, sc, ks.b_apart, &.{ a.x1, a.hv, w.fn_w, w.fn_scale, w.fn_base, w.norm_w, w.gate_w, w.gate_bias, scal, eps, a.ctr, a.parts, a.logits }, .{
        .{ .shape = &.{ ci(p.rows), ci(p.d) }, .dt = act },
        .{ .shape = &.{ ci(p.rows), ci(p.top_k) }, .dt = .uint32 },
        .{ .shape = &.{ ci(p.rows), ci(p.top_k) }, .dt = .float32 },
    }, gridB(p), b_threads, &bTargs(p));
    return .{ .h2 = b[0], .sf = try launchSink(sc, ks, p, a.parts, w.fn_scale, w.fn_base, eps, b[0]), .idx = b[1], .wts = b[2] };
}

/// The sink launch: the partials `parts` reduced over the slices, then the split (`dec_ffn_in_sink.metal`); `order` is
/// launch 2's h2, read only so the launch follows launch 2.
pub fn launchSink(sc: *dsv41.Scope, ks: Kernels, p: Dims, parts: A, scale: A, base: A, eps: A, order: A) !dsv41.Split {
    const s = try run(3, sc, ks.sink, &.{ parts, scale, base, eps, order }, .{
        .{ .shape = &.{ ci(p.rows), hc }, .dt = .float32 },
        .{ .shape = &.{ ci(p.rows), hc }, .dt = .float32 },
        .{ .shape = &.{ ci(p.rows), hc, hc }, .dt = .float32 },
    }, b_threads, b_threads, &sinkTargs(p));
    return .{ .pre = s[0], .post = s[1], .comb = s[2] };
}

/// Whether the sink launch takes `p`: whole 128-column slices, decode rows, its threadgroup memory.
fn servesSink(p: Dims) bool {
    return p.rows >= 1 and p.rows <= decode.max_rows and p.d % slice_cols == 0 and p.d >= slice_cols and 4 * p.rows * (slices(p) + 1) * (mixes + 1) <= 32768;
}

fn sinkTargs(p: Dims) [5]TArg {
    return .{ .{ .name = "R", .v = p.rows }, .{ .name = "D", .v = p.d }, .{ .name = "P", .v = slices(p) }, .{ .name = "HC", .v = hc }, .{ .name = "ITERS", .v = p.iters } };
}

/// wo_b, the hc expand, the FFN norm and mixes, the router and its top K; null where `serves` says no.
pub fn launch(sc: *dsv41.Scope, ks: Kernels, p: Dims, w: Weights, oa: A, x: A, sa: *const dsv41.Split, eps: A, scal: A) !?Out {
    if (!serves(p)) return null;
    const a = try launchA(sc, ks, p, w, oa, x, sa);
    const apart = sinkOn();
    const b = if (apart) try launchBApart(sc, ks, p, w, a, eps, scal) else try launchB(sc, ks, p, w, a, eps, scal);
    return .{ .x1 = a.x1, .h2 = b.h2, .sf = b.sf, .idx = b.idx, .wts = b.wts, .sink = apart };
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
fn perX1(s: Shape) usize {
    return s.rows * hc * s.n;
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
/// The matrix units' depth: a simdgroup's K / (16 NSG) tile steps of 16 products, then the simdgroups' sum.
/// The mixes' dot over the stream (4 D terms).
fn depthB(s: Shape) usize {
    return hc * s.n;
}
/// The sink's sums: a lane's slices, then the simdgroup's tree.
fn depthSink(s: Shape) usize {
    return s.n / slice_cols / 32 + 1 + 5;
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
        g.int("FOLD", 0);
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
    /// `apart`: launch 2 without its mix tail (`MLX_SERVE_DSV41_FFN_IN_SINK`): h2 and the picks, no split.
    pub const Params = struct { apart: bool = false };

    fn dims(s: Shape) Dims {
        return .{ .rows = s.rows, .d = s.n, .e = s.k };
    }

    pub fn geometry(prm: Params, s: Shape) ?metal.Geometry {
        const p = dims(s);
        if (!servesB(p)) return null;
        var g = metal.Geometry{ .grid = .{ ci(gridB(p)), 1, 1 }, .tg = .{ ci(b_threads), 1, 1 } };
        g.out(&.{ ci(p.rows), ci(p.d) }, act);
        if (!prm.apart) {
            g.out(&.{ ci(p.rows), hc }, .float32);
            g.out(&.{ ci(p.rows), hc }, .float32);
            g.out(&.{ ci(p.rows), hc, hc }, .float32);
        }
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

    /// h2, pre, post, comb (not `apart`), then the picks and their weights.
    pub fn count(prm: Params, s: Shape) usize {
        return s.rows * (s.n + (if (prm.apart) 0 else 2 * hc + hc * hc) + 2 * 6);
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
        if (!k.p.apart) return computeFull(k, out, mag, sum);
        // `apart`: the full outputs, then h2 and the picks moved up over the split's.
        const R = k.shape.rows;
        const D = k.shape.n;
        const a = k.scratch();
        const full = try a.alloc(f64, count(.{}, k.shape));
        defer a.free(full);
        const fmag = try a.alloc(f64, full.len);
        defer a.free(fmag);
        try computeFull(k, full, fmag, sum);
        const o_idx = R * D + R * (2 * hc + hc * hc);
        @memcpy(out[0 .. R * D], full[0 .. R * D]);
        @memcpy(out[R * D ..], full[o_idx..]);
        for (mag, out) |*m, w| m.* = @abs(w);
    }

    fn computeFull(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
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
        const apart = k.p.apart;
        const o_pre = 2 * R * D;
        const o_post = o_pre + 4 * R * hc;
        const o_comb = o_post + 4 * R * hc;
        const o_idx = if (apart) o_pre else o_comb + 4 * R * hc * hc;
        const o_wts = o_idx + 4 * R * p.top_k;
        for (0..R) |r| {
            contract.collapseNorm(k.in(1)[r * D ..][0..D], zeros, k.in(5), eps0, point, iv);
            for (iv, h2, 0..) |q, *o, d| {
                o.* = outAt(k, .bf16, 0, r * D + d);
                const ok = if (point) spec.classOf(o.*) == spec.classOf(q.lo) else q.has(o.*);
                if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} h2 {d} is {e} outside [{e}, {e}]", .{ r, d, o.*, q.lo, q.hi }) catch "h2";
            }
            // `apart`: the split is the sink launch's (`FfnSink`).
            if (!apart) {
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

/// `MLX_SERVE_DSV41_FFN_IN_SINK`'s second launch: launch 2's mix partials (`D / 128` slices of each row's 24 dots and
/// its sum of squares) reduced over the slices, then the split. Shape: rows, D; k unused.
const FfnSink = struct {
    pub const Params = struct {};

    fn dims(s: Shape) Dims {
        return .{ .rows = s.rows, .d = s.n };
    }

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        const p = dims(s);
        if (!servesSink(p)) return null;
        var g = metal.Geometry{ .grid = .{ ci(b_threads), 1, 1 }, .tg = .{ ci(b_threads), 1, 1 } };
        g.out(&.{ ci(p.rows), hc }, .float32);
        g.out(&.{ ci(p.rows), hc }, .float32);
        g.out(&.{ ci(p.rows), hc, hc }, .float32);
        g.dtype("T", act);
        for (sinkTargs(p)) |t| g.int(t.name, t.v);
        return g;
    }

    /// The partials as launch 2's slices leave them (a slice's 4 x 128 terms of each dot, its 512 squares), the mixes'
    /// scales and bases, the eps pair, and launch 2's h2 (read for order only).
    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = dims(k.shape);
        const R = p.rows;
        const P = slices(p);
        const PL = mixes + 1;
        const parts = try k.add(.float32, &.{R * P * PL});
        const scale = try k.add(.float32, &.{mixes});
        const base = try k.add(.float32, &.{mixes});
        const eps = try k.add(.float32, &.{2});
        _ = try k.add(act, &.{ R, p.d });
        kit.uniform(rng, scale, 0.5, 1.5);
        kit.uniform(rng, base, -0.5, 0.5);
        eps[0] = eps_rms;
        eps[1] = eps_hc;
        for (0..R * P) |q| {
            const row = parts[q * PL ..][0..PL];
            kit.uniform(rng, row[0..mixes], -0.4, 0.4);
            kit.uniform(rng, row[mixes..], 400, 600);
        }
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(parts, R, P * PL),
            .stress => {
                // Mirrored integers: each dot's even slices +(2^12 + a), its odd ones -(2^12 + b) (a, b below 2^10),
                // the squares integers, eps 0. Every partial sum is an integer below 2^18, exact in fp32 in any order;
                // fp16 and bf16 round the terms themselves (2^12 + a needs 13 bits) and miss the small total.
                eps[0] = 0;
                @memset(scale, 1);
                @memset(base, 0);
                for (0..R) |r| for (0..PL) |m| for (0..P) |sl| {
                    const c: f64 = @floatFromInt(rng.uintLessThan(u32, 1024));
                    const v: *f64 = &parts[(r * P + sl) * PL + m];
                    v.* = if (m == mixes) 512 + @as(f64, @floatFromInt(rng.uintLessThan(u32, 64))) else if (sl % 2 == 0) 0x1p12 + c else -(0x1p12 + c);
                };
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * (2 * hc + hc * hc);
    }

    /// One row's 24 mixes from its slices in float64, and each one's allowance: the slices' sum (lanes, then the tree)
    /// on its terms, the rms's half on the product, the f32 tail.
    /// `sum` (the gates' summation, the reference's when null) adds each sum's slices.
    fn rowZ(k: anytype, r: usize, z: *[mixes]f64, allow: *[mixes]f64, sum: ?Sum) void {
        const p = dims(k.shape);
        const P = slices(p);
        const PL = mixes + 1;
        const parts = k.in(0);
        const eps0 = spec.roundTo(.f32, k.in(3)[0]);
        var col: [256]f64 = undefined;
        for (0..P) |sl| col[sl] = parts[(r * P + sl) * PL + mixes];
        var ss: f64 = 0;
        var ssm: f64 = 0;
        for (col[0..P]) |v| {
            ss += v;
            ssm += @abs(v);
        }
        if (sum) |sm| ss = sm.of(col[0..P]);
        const g = spec.gamma(@as(f64, @floatFromInt(P / 32 + 1)) + 5, u);
        const rs = 1 / @sqrt(ss / @as(f64, @floatFromInt(hc * p.d)) + eps0);
        const rs_rel = g * ssm / @max(@abs(ss), 0x1p-140) / 2 + 3 * u;
        for (z, allow, 0..) |*o, *al, m| {
            var d: f64 = 0;
            var mg: f64 = 0;
            for (0..P) |sl| {
                const v = parts[(r * P + sl) * PL + m];
                col[sl] = v;
                d += v;
                mg += @abs(v);
            }
            if (sum) |sm| d = sm.of(col[0..P]);
            const y = d * rs * k.in(1)[m];
            o.* = y + k.in(2)[m];
            al.* = g * mg * @abs(rs * k.in(1)[m]) + (rs_rel + 3 * u) * @abs(y) + 2 * u * @abs(o.*) + 0x1p-140;
        }
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const R = k.shape.rows;
        const eps1 = spec.roundTo(.f32, k.in(3)[1]);
        for (0..R) |r| {
            var z: [mixes]f64 = undefined;
            var allow: [mixes]f64 = undefined;
            rowZ(k, r, &z, &allow, sum);
            var sp: [mixes]f64 = undefined;
            refSplit(&z, eps1, sinkIters(), &sp);
            @memcpy(out[r * hc ..][0..hc], sp[0..hc]);
            @memcpy(out[R * hc + r * hc ..][0..hc], sp[hc .. 2 * hc]);
            @memcpy(out[2 * R * hc + r * hc * hc ..][0 .. hc * hc], sp[2 * hc ..]);
        }
        for (mag, out) |*m, w| m.* = @abs(w);
    }

    /// The split within the mixes' propagated allowance, as `FfnB` checks it.
    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const R = k.shape.rows;
        const point = at == .nonfinite;
        const eps1 = spec.roundTo(.f32, k.in(3)[1]);
        const o_post = 4 * R * hc;
        const o_comb = o_post + 4 * R * hc;
        for (0..R) |r| {
            var z: [mixes]f64 = undefined;
            var allow: [mixes]f64 = undefined;
            rowZ(k, r, &z, &allow, null);
            var want: [mixes]f64 = undefined;
            refSplit(&z, eps1, sinkIters(), &want);
            var slack: [mixes]f64 = undefined;
            for (&slack, want) |*sl, v| sl.* = split_ops * u * @abs(v) + 0x1p-140;
            if (!point) for (0..mixes) |m| {
                const step = @max(allow[m], 1e-9);
                var zz = z;
                zz[m] += step;
                var alt: [mixes]f64 = undefined;
                refSplit(&zz, eps1, sinkIters(), &alt);
                for (&slack, alt, want) |*sl, v, w| sl.* += 2 * @abs(v - w) * allow[m] / step;
            };
            for (want, slack, 0..) |w, sl, o| {
                const g = if (o < hc) outAt(k, .f32, 0, r * hc + o) else if (o < 2 * hc) outAt(k, .f32, o_post, r * hc + o - hc) else outAt(k, .f32, o_comb, r * hc * hc + o - 2 * hc);
                const ok = if (point) spec.classOf(g) == spec.classOf(w) else @abs(g - w) <= sl;
                if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} split output {d} is {e} where float64 gives {e}", .{ r, o, g, w }) catch "split";
            }
        }
        return null;
    }
};

/// The Sinkhorn iterations the contract's sink launches run (the release's).
fn sinkIters() usize {
    return (Dims{ .rows = 1 }).iters;
}

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

const launch_b_apart = metal.Launch{
    .name = "dsv41_dec_ffn_in_post_apart",
    .inputs = &B_INS,
    .outputs = &B_APART_OUTS,
    .source = B_SOURCE,
    .header = B_APART_HEADER,
    .tg_bytes = 4 * 7728 + 4,
    // `launch_b`'s records: the same source less the mix tail (a lower bound on its own limits).
    .m1 = 704,
    .m2 = 704,
};

const launch_sink = metal.Launch{
    .name = "dsv41_dec_ffn_in_sink",
    .inputs = &SINK_INS,
    .outputs = &SINK_OUTS,
    .source = SINK_SOURCE,
    .tg_bytes = 4 * decode.max_rows * (40 + 1) * (mixes + 1),
    // its own 256-thread launch (a lower bound; no `kernel contract offline` record yet).
    .m1 = 256,
    .m2 = 256,
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
    try kit.add(set, FfnB, &launch_b_apart, .{ .apart = true }, .{
        .name = "dsv41 dec ffn_in B apart (the mixes' partials, norm, router; the split in the sink launch)",
        .accum = .f32,
        .depth = depthB,
        .mids = 1,
        .out = .bf16,
        .outs = &.{
            .{ .elem = .bf16, .count = perD },
            .{ .elem = .u32, .count = perPick },
            .{ .elem = .f32, .count = perPick },
        },
        .in_place = true,
        .nan = nan_rule,
        .tiles = .{ .{ .size = 3 }, .{}, .{ .size = 16, .step = 8, .min = 8 } },
        .served = &.{ .{ .rows = 1, .n = 5120, .k = 384 }, .{ .rows = 6, .n = 5120, .k = 384 } },
    });
    try kit.add(set, FfnSink, &launch_sink, .{}, .{
        .name = "dsv41 dec ffn_in sink (the mixes' slices reduced, the split)",
        .accum = .f32,
        .depth = depthSink,
        .ops = split_ops,
        .mids = 1,
        .out = .f32,
        .outs = &.{
            .{ .elem = .f32, .count = perHc },
            .{ .elem = .f32, .count = perHc },
            .{ .elem = .f32, .count = perComb },
        },
        .nan = nan_rule,
        .tiles = .{ .{ .size = 3 }, .{ .size = 128, .step = 128, .min = 128 }, .{} },
        .served = &.{ .{ .rows = 1, .n = 5120, .k = 1 }, .{ .rows = 6, .n = 5120, .k = 1 } },
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
    if (!mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
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
    if (!mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
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

/// `launch` on one case with `MLX_SERVE_DSV41_FFN_IN_SINK` off, then on (or the planted sink at `plant_iters` Sinkhorn
/// iterations): the number of output words that differ (x1, h2, pre, post, comb, the picks, their weights).
fn sinkDiff(gpa: std.mem.Allocator, fx: *Fixture, ks: Kernels, ly: *const dsv41.Layer, w: Weights, rows: usize, plant_iters: ?usize) !usize {
    const m = fx.m;
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const oa = try fx.normal(&.{ ci(rows), 8192 }, act, 0, 1);
    defer _ = mlx.mlx_array_free(oa);
    const x = try fx.stream(rows);
    defer _ = mlx.mlx_array_free(x);
    const cpre = try fx.cpre(rows);
    defer _ = mlx.mlx_array_free(cpre);
    const sa = (try dsv41.hcStage(&sc, m, x, null, cpre, &ly.hc_attn, ly.attn_norm)).sp;
    const eps = try dsv41.hcEps(&sc, m);
    const scal = mlx.mlx_array_new_data(&[_]f32{m.cfg.router_scaling_factor}, &[_]c_int{1}, 1, .float32);
    try sc.items.append(sc.a, scal);
    const p: Dims = .{ .rows = rows, .d = m.cfg.hidden_size, .e = @intCast(mlx.getShape(ly.gate_w)[0]), .top_k = ly.top_k, .iters = m.cfg.dsv4_hc_sinkhorn_iters, .norm = m.cfg.moe_route_norm and ly.top_k > 1 };
    var outs: [2][7]A = undefined;
    for (0..2) |k| {
        var ov = env.override(.MLX_SERVE_DSV41_FFN_IN_SINK, if (k == 0) "0" else "1");
        defer ov.restore();
        var o = (try launch(&sc, ks, p, w, oa, x, &sa, eps, scal)) orelse return error.Declined;
        if (k == 1) try std.testing.expect(o.sink);
        if (k == 1) if (plant_iters) |it| {
            var q = p;
            q.iters = it;
            const a = try launchA(&sc, ks, q, w, oa, x, &sa);
            const b = try launchBApart(&sc, ks, q, w, a, eps, scal);
            o.sf = b.sf;
        };
        outs[k] = .{ o.x1, o.h2, o.sf.pre, o.sf.post, o.sf.comb, o.idx, o.wts };
        try dsv41.evalAll(&outs[k]);
    }
    var differ: usize = 0;
    for (outs[0], outs[1]) |a, b| {
        const n = mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a);
        try std.testing.expectEqual(n, mlx.mlx_array_size(b) * mlx.mlx_array_itemsize(b));
        const xa = mlx.mlx_array_data_uint8(a) orelse return error.NoData;
        const xb = mlx.mlx_array_data_uint8(b) orelse return error.NoData;
        var i: usize = 0;
        while (i < n) : (i += 4) differ += @intFromBool(!std.mem.eql(u8, xa[i..][0..4], xb[i..][0..4]));
    }
    return differ;
}

test "dsv41 dec ffn_in sink: MLX_SERVE_DSV41_FFN_IN_SINK=1 gives launch 2's words, rows 1..6 on layers 21 and 23 and at 384 experts; a planted 1-iteration Sinkhorn fails" {
    if (!mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .ctx = 8 });
    defer fx.deinit();
    const ks = try build();
    defer free(ks);
    const w = try randomWeights(86);
    defer freeWeights(w);
    const wide = layerWith(fx.layer(21), w);
    var cases: usize = 0;
    for ([_]usize{ 21, 23, 0 }) |li| for (1..decode.max_rows + 1) |rows| {
        const ly = if (li == 0) &wide else fx.layer(li);
        const wl = if (li == 0) w else weightsOf(ly);
        try std.testing.expectEqual(@as(usize, 0), try sinkDiff(gpa, fx, ks, ly, wl, rows, null));
        cases += 1;
    };
    // (the split converges within the served 20 iterations' f32 words well before the last: a short split is 1 iteration)
    const planted = try sinkDiff(gpa, fx, ks, &wide, w, 3, 1);
    std.debug.print("[dec] ffn_in sink: {d} cases word for word; the planted 1-iteration Sinkhorn moves {d} words\n", .{ cases, planted });
    try std.testing.expect(planted > 0);
}

test "dsv41 dec ffn_in: the same inputs give the same bytes, launch after launch, at the release's shapes" {
    if (!mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .ctx = 8 });
    defer fx.deinit();
    const ks = try build();
    defer free(ks);
    const w = try randomWeights(84);
    defer freeWeights(w);
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const eps = try dsv41.hcEps(&sc, fx.m);
    const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{fx.m.cfg.router_scaling_factor}, &[_]c_int{1}, 1, .float32));
    for ([_]usize{ 1, 6 }) |rows| {
        const oa = try fx.normal(&.{ ci(rows), 8192 }, act, 0, 1);
        defer _ = mlx.mlx_array_free(oa);
        const x = try fx.stream(rows);
        defer _ = mlx.mlx_array_free(x);
        const cpre = try fx.cpre(rows);
        defer _ = mlx.mlx_array_free(cpre);
        const sa = (try dsv41.hcStage(&sc, fx.m, x, null, cpre, &fx.layer(21).hc_attn, fx.layer(21).attn_norm)).sp;
        const p: Dims = .{ .rows = rows };
        var first: ?[]u8 = null;
        defer if (first) |f| gpa.free(f);
        // The router's logits and the hv sums gather from 40 column slices: their sum must not follow arrival order.
        for (0..24) |_| {
            const o = (try launch(&sc, ks, p, w, oa, x, &sa, eps, scal)) orelse return error.Declined;
            const outs = [_]A{ o.h2, o.sf.pre, o.sf.post, o.sf.comb, o.idx, o.wts };
            try dsv41.evalAll(&outs);
            var bytes: std.ArrayList(u8) = .empty;
            defer bytes.deinit(gpa);
            for (outs) |a| {
                const n = mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a);
                const d = mlx.mlx_array_data_uint8(a) orelse return error.NoData;
                try bytes.appendSlice(gpa, d[0..n]);
            }
            if (first) |f| try std.testing.expectEqualSlices(u8, f, bytes.items) else first = try gpa.dupe(u8, bytes.items);
        }
    }
}

test "dsv41 dec ffn_in: a planted dropped router bias fails the equivalence" {
    if (!mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .ctx = 8 });
    defer fx.deinit();
    const ks = try build();
    defer free(ks);
    const w = try randomWeights(79);
    defer freeWeights(w);
    const ly = layerWith(fx.layer(21), w);
    var bad = w;
    var z = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ 1, 384 }, 2, .float32, mlx.gpuStream()));
    defer _ = mlx.mlx_array_free(z);
    bad.gate_bias = z;
    try std.testing.expectError(error.TestUnexpectedResult, equivalence(gpa, fx, ks, &ly, bad, 6));
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
    // `MLX_SERVE_DSV41_FFN_IN_SINK`'s two launches.
    var cp = kit.Case(FfnB).init(gpa, &launch_b_apart, .{ .apart = true });
    defer cp.deinit();
    try std.testing.expect(try cp.prepare(.{ .rows = 3, .n = 5120, .k = 384 }, .random, 9));
    try cp.run();
    const wp: Weights = .{ .wo_w = none, .wo_s = none, .fn_w = cp.args[2], .fn_scale = cp.args[3], .fn_base = cp.args[4], .norm_w = cp.args[5], .gate_w = cp.args[6], .gate_bias = cp.args[7] };
    const ap: AOut = .{ .x1 = cp.args[0], .hv = cp.args[1], .ctr = cp.args[10], .parts = cp.args[11], .logits = cp.args[12] };
    const bp = try launchBApart(&sc, ks, pb, wp, ap, cp.args[9], cp.args[8]);
    for ([_]A{ bp.h2, bp.idx, bp.wts }, 0..) |g, i| try std.testing.expect(try sameBytes(&cp.rig, g, cp.rig.output(i)));
    var cs = kit.Case(FfnSink).init(gpa, &launch_sink, .{});
    defer cs.deinit();
    try std.testing.expect(try cs.prepare(.{ .rows = 3, .n = 5120, .k = 1 }, .random, 11));
    try cs.run();
    const sp = try launchSink(&sc, ks, pb, cs.args[0], cs.args[1], cs.args[2], cs.args[3], cs.args[4]);
    for ([_]A{ sp.pre, sp.post, sp.comb }, 0..) |g, i| try std.testing.expect(try sameBytes(&cs.rig, g, cs.rig.output(i)));
}

/// A `[n]` f32 array of `v`, as `dt`.
fn hostArray(v: []const f32, shape: []const c_int, dt: mlx.mlx_dtype) !A {
    const a = mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32);
    if (dt == .float32) return a;
    defer _ = mlx.mlx_array_free(a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, a, dt, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

test "dsv41 dec ffn_in: launch 2's router scores at logits near -9 (e^z about 1e-4) are within 1e-5 of float64" {
    if (!mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const ks = try build();
    defer free(ks);
    const D = 5120;
    const E = 384;
    // hv = 1 on every column (oa 0, a unit stream, comb the identity, pre 1/4) and a unit norm weight: expert e's logit
    // is its one gate weight z_e = -(8 + (e % 24) / 16), and the bias makes experts 16..21 (z -9 .. -9.3125) the picks.
    const zs = try gpa.alloc(f32, E * D);
    defer gpa.free(zs);
    @memset(zs, 0);
    var bias: [E]f32 = undefined;
    for (0..E) |e| {
        zs[e * D + e] = -(8.0 + @as(f32, @floatFromInt(e % 24)) / 16.0);
        bias[e] = if (e >= 16 and e < 22) 10 else 0;
    }
    const ones = try gpa.alloc(f32, 4 * D);
    defer gpa.free(ones);
    @memset(ones, 1);
    const zeros = try gpa.alloc(f32, 8192);
    defer gpa.free(zeros);
    @memset(zeros, 0);
    var eye: [16]f32 = @splat(0);
    for (0..4) |i| eye[i * 4 + i] = 1;
    const w0 = try randomWeights(85);
    defer freeWeights(w0);
    var w = w0;
    w.norm_w = try hostArray(ones[0..D], &.{D}, act);
    defer _ = mlx.mlx_array_free(w.norm_w);
    w.gate_w = try hostArray(zs, &.{ E, D }, act);
    defer _ = mlx.mlx_array_free(w.gate_w);
    w.gate_bias = try hostArray(&bias, &.{ 1, E }, .float32);
    defer _ = mlx.mlx_array_free(w.gate_bias);
    const arrs = [_]A{
        try hostArray(zeros, &.{ 1, 8192 }, act),         try hostArray(ones, &.{ 1, 4, D }, act),
        try hostArray(&.{ 0.25, 0.25, 0.25, 0.25 }, &.{ 1, 4 }, .float32), try hostArray(&.{ 1, 1, 1, 1 }, &.{ 1, 4 }, .float32),
        try hostArray(&eye, &.{ 1, 4, 4 }, .float32),     try hostArray(&.{ 1e-6, 1e-6 }, &.{2}, .float32),
        try hostArray(&.{1.5}, &.{1}, .float32),
    };
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const sa: dsv41.Split = .{ .pre = arrs[2], .post = arrs[3], .comb = arrs[4] };
    const p: Dims = .{ .rows = 1, .e = E, .top_k = 6, .norm = false };
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const a = try launchA(&sc, ks, p, w, arrs[0], arrs[1], &sa);
    const b = try launchB(&sc, ks, p, w, a, arrs[5], arrs[6]);
    try dsv41.evalAll(&.{ b.idx, b.wts });
    const ih = try dsv41.hostF32(b.idx);
    defer gpa.free(ih);
    const wh = try dsv41.hostF32(b.wts);
    defer gpa.free(wh);
    // Launch 2 routes off h2 = bf16(1 * bf16(inv)) = 1.
    for (ih, wh, 0..) |iv, wv, q| {
        const e: usize = @intFromFloat(iv);
        try std.testing.expectEqual(16 + q, e);
        const z: f64 = zs[e * D + e];
        const want = @sqrt(std.math.log1p(@exp(-@abs(z)))) * 1.5;
        const rel = @abs(@as(f64, wv) - want) / want;
        std.debug.print("[dec] ffn_in launch 2 pick {d}: z {d:.4}, weight {e} where float64 gives {e} ({e} rel)\n", .{ e, z, wv, want, rel });
        try std.testing.expect(rel <= 1e-5);
    }
}

test "dsv41 dec ffn_in: launch 2 passes the contract's gates 1-4 and its sums gate at stress seed 0xc0de" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var set = kc.Set.init(gpa);
    defer set.deinit();
    try register(&set);
    var pick: std.ArrayList(kit.gates.Kernel) = .empty;
    defer pick.deinit(gpa);
    for (set.kernels.items) |k| if (std.mem.indexOf(u8, k.spec.name, "ffn_in B (") != null) try pick.append(gpa, k);
    try std.testing.expectEqual(@as(usize, 1), pick.items.len);
    var t: kit.gates.Tally = .{};
    try kit.gates.runAll(gpa, pick.items, 0xc0de, &t);
    try std.testing.expectEqual(@as(usize, 0), t.problems());
    // The stress inputs where f32 log(1 + y) put the routed weights 6.5x past the exact-sum bound.
    try std.testing.expectEqual(kit.gates.Verdict.pass, try kit.gates.sums(gpa, pick.items[0], 0xc0de));
}

test "dsv41 dec ffn_in sink: launch 2 apart and the sink launch pass the contract's gates 1-4" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var set = kc.Set.init(gpa);
    defer set.deinit();
    try register(&set);
    var pick: std.ArrayList(kit.gates.Kernel) = .empty;
    defer pick.deinit(gpa);
    for (set.kernels.items) |k| if (std.mem.indexOf(u8, k.spec.name, "ffn_in B apart") != null or std.mem.indexOf(u8, k.spec.name, "ffn_in sink") != null) try pick.append(gpa, k);
    try std.testing.expectEqual(@as(usize, 2), pick.items.len);
    var t: kit.gates.Tally = .{};
    try kit.gates.runAll(gpa, pick.items, 0xc0de, &t);
    try std.testing.expectEqual(@as(usize, 0), t.problems());
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
    /// The ubench's bound: launch 2 without its counters and tails (`DSV41_FFN_NOTAIL`), in place of the served one.
    notail: ?mlx.mlx_fast_metal_kernel = null,
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
                var ks = b.ks;
                if (b.notail) |k| ks.b = k;
                const out = (try launch(sc, ks, p, set, oa, b.prev, &b.sa, b.eps, b.scal)).?;
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
    if (!mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
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

test "dsv41 dec ffn_in: ubench, R = 1..6: launch 1, and the pair with and without launch 2's tails" {
    if (!mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    if (!decode.ubenchOn()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .ctx = 8 });
    defer fx.deinit();
    const ks = try build();
    defer free(ks);
    const notail = try dsv41.buildKernel("dsv41_dec_ffn_in_post_notail", &B_INS, &B_OUTS, B_SOURCE, "#define DSV41_FFN_NOTAIL 1\n");
    defer _ = mlx.mlx_fast_metal_kernel_free(notail);
    const nsets = 48;
    var sets: [nsets]Weights = undefined;
    var layers: [nsets]dsv41.Layer = undefined;
    for (&sets, &layers, 0..) |*w, *l, i| {
        w.* = try randomWeights(1000 + 10 * i);
        l.* = layerWith(fx.layer(21), w.*);
    }
    defer for (sets) |w| freeWeights(w);
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const eps = try dsv41.hcEps(&sc, fx.m);
    const scal = mlx.mlx_array_new_data(&[_]f32{fx.m.cfg.router_scaling_factor}, &[_]c_int{1}, 1, .float32);
    try sc.items.append(sc.a, scal);
    for (1..decode.max_rows + 1) |rows| {
        const x = try fx.stream(rows);
        defer _ = mlx.mlx_array_free(x);
        const cpre = try fx.cpre(rows);
        defer _ = mlx.mlx_array_free(cpre);
        const sa0 = (try dsv41.hcStage(&sc, fx.m, x, null, cpre, &fx.layer(21).hc_attn, fx.layer(21).attn_norm)).sp;
        var us: [3]f64 = undefined;
        // Launch 1, the pair, the pair without launch 2's tails.
        const arms = [_]struct { mode: @TypeOf(@as(Bench, undefined).mode), notail: bool = false }{
            .{ .mode = .launch_1 },
            .{ .mode = .new },
            .{ .mode = .new, .notail = true },
        };
        for (&us, arms) |*t, arm| {
            var b = Bench{ .gpa = gpa, .rows = rows, .m = fx.m, .ks = ks, .sets = &sets, .layers = &layers, .x0 = x, .sa0 = sa0, .eps = eps, .scal = scal, .mode = arm.mode, .notail = if (arm.notail) notail else null };
            defer b.deinit();
            t.* = (try b.us(nsets)).total;
        }
        std.debug.print("[dec-ubench] ffn_in R={d}: launch 1 {d:.1} us; pair {d:.1} us; pair without launch 2's tails {d:.1} us (launch 2 {d:.1} us, its tails {d:.1} us)\n", .{ rows, us[0], us[1], us[2], us[1] - us[0], us[1] - us[2] });
    }
}

test "dsv41 dec ffn_in: MLX_SERVE_DSV41_FFN_IN_R5_COLS1=1, launch 1 at 5 rows on one column a simdgroup writes the served two-column plan's bytes on layers 21 and 23 and random weights; another wo_b differs" {
    if (!mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .ctx = 8 });
    defer fx.deinit();
    const ks = try build();
    defer free(ks);
    const rw = try randomWeights(80);
    defer freeWeights(rw);
    const other = try randomWeights(81);
    defer freeWeights(other);
    const rl = layerWith(fx.layer(21), rw);
    const cases = [_]struct { ly: *const dsv41.Layer, w: Weights }{ .{ .ly = fx.layer(21), .w = weightsOf(fx.layer(21)) }, .{ .ly = fx.layer(23), .w = weightsOf(fx.layer(23)) }, .{ .ly = &rl, .w = rw } };
    const rows = 5;
    for (cases, 0..) |cs, ci_| {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const oa = try fx.normal(&.{ ci(rows), 8192 }, act, 0, 1);
        defer _ = mlx.mlx_array_free(oa);
        const x = try fx.stream(rows);
        defer _ = mlx.mlx_array_free(x);
        const cpre = try fx.cpre(rows);
        defer _ = mlx.mlx_array_free(cpre);
        const sa = (try dsv41.hcStage(&sc, fx.m, x, null, cpre, &cs.ly.hc_attn, cs.ly.attn_norm)).sp;
        const p: Dims = .{ .rows = rows, .d = fx.m.cfg.hidden_size, .e = @intCast(mlx.getShape(cs.ly.gate_w)[0]), .top_k = cs.ly.top_k, .iters = fx.m.cfg.dsv4_hc_sinkhorn_iters, .norm = fx.m.cfg.moe_route_norm and cs.ly.top_k > 1 };
        // The reference is the two-column plan (=0); the switch is on by default.
        var o0 = env.override(.MLX_SERVE_DSV41_FFN_IN_R5_COLS1, "0");
        defer o0.restore();
        try std.testing.expectEqual(@as(usize, 2), aCols(5));
        const served = try launchA(&sc, ks, p, cs.w, oa, x, &sa);
        const one = blk: {
            var o1 = env.override(.MLX_SERVE_DSV41_FFN_IN_R5_COLS1, "1");
            defer o1.restore();
            try std.testing.expectEqual(@as(usize, 1), aCols(5));
            break :blk try launchA(&sc, ks, p, cs.w, oa, x, &sa);
        };
        try std.testing.expectEqual(@as(usize, 2), aCols(5));
        const dx = try wordsDiffer(served.x1, one.x1);
        const dh = try wordsDiffer(served.hv, one.hv);
        std.debug.print("[ffn_in r5 cols1] case {d}: x1 differs in {d} words, hv in {d}\n", .{ ci_, dx, dh });
        try std.testing.expectEqual(@as(usize, 0), dx + dh);
        if (ci_ == 2) {
            // The planted bug: another wo_b under the one-column plan; the comparison sees it.
            var bad = cs.w;
            bad.wo_w = other.wo_w;
            var o1 = env.override(.MLX_SERVE_DSV41_FFN_IN_R5_COLS1, "1");
            defer o1.restore();
            const pb = try launchA(&sc, ks, p, bad, oa, x, &sa);
            try std.testing.expect(try wordsDiffer(served.x1, pb.x1) > 0);
        }
    }
}

/// The words of `a` and `b` (same shape and dtype) that differ, as bits.
fn wordsDiffer(a: A, b: A) !usize {
    const x = try dsv41.hostF32(a);
    defer std.testing.allocator.free(x);
    const y = try dsv41.hostF32(b);
    defer std.testing.allocator.free(y);
    var n: usize = 0;
    for (x, y) |xa, ya| n += @intFromBool(@as(u32, @bitCast(xa)) != @as(u32, @bitCast(ya)));
    return n;
}

test "dsv41 dec ffn_in: ubench, launch 1 on the vector kernel at R = 3..6 on one and two columns a simdgroup" {
    if (!mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
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
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const eps = try dsv41.hcEps(&sc, fx.m);
    const scal = mlx.mlx_array_new_data(&[_]f32{fx.m.cfg.router_scaling_factor}, &[_]c_int{1}, 1, .float32);
    try sc.items.append(sc.a, scal);
    defer cols_test = null;
    for (3..decode.max_rows + 1) |rows| {
        const x = try fx.stream(rows);
        defer _ = mlx.mlx_array_free(x);
        const cpre = try fx.cpre(rows);
        defer _ = mlx.mlx_array_free(cpre);
        const sa0 = (try dsv41.hcStage(&sc, fx.m, x, null, cpre, &fx.layer(21).hc_attn, fx.layer(21).attn_norm)).sp;
        var us: [2]f64 = undefined;
        for (&us, [_]usize{ 1, 2 }) |*t, nr| {
            var plan: [decode.max_rows + 1]usize = @splat(1);
            plan[rows] = nr;
            cols_test = plan;
            var b = Bench{ .gpa = gpa, .rows = rows, .m = fx.m, .ks = ks, .sets = &sets, .layers = &layers, .x0 = x, .sa0 = sa0, .eps = eps, .scal = scal, .mode = .launch_1 };
            defer b.deinit();
            t.* = (try b.us(nsets)).total;
        }
        std.debug.print("[dec-ubench] ffn_in launch 1 R={d}: one column {d:.1} us ({d:.0}% of the wo_b floor), two columns {d:.1} us ({d:.0}%)\n", .{ rows, us[0], 100 * 43.2e6 / decode.stream_bytes_per_us / us[0], us[1], 100 * 43.2e6 / decode.stream_bytes_per_us / us[1] });
    }
}

test "dsv41 dec ffn_in: wo_b's MLX mxfp8 reads give the FOLD form's bytes on finite codes" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try decode.expectFoldSame(FfnA, &launch_a, .{}, &.{ .{ .rows = 1, .n = 256, .k = 8192 }, .{ .rows = 6, .n = 256, .k = 8192 }, .{ .rows = 1, .n = 5120, .k = 8192 } });
}

/// One layer's launch pair over rows of fixed inputs (`decode.rowsInvariant`): `sa` the attention split of all of them.
const RowsCase = struct {
    sc: *dsv41.Scope,
    ks: Kernels,
    m: *const dsv41.Dsv41Model,
    ly: *const dsv41.Layer,
    oa: A,
    x: A,
    sa: dsv41.Split,
    eps: A,
    scal: A,

    pub fn runRows(c: *RowsCase, lo: usize, hi: usize) !decode.RowOuts {
        const sc = c.sc;
        const sa: dsv41.Split = .{ .pre = try sc.slice(c.sa.pre, 0, lo, hi), .post = try sc.slice(c.sa.post, 0, lo, hi), .comb = try sc.slice(c.sa.comb, 0, lo, hi) };
        const p: Dims = .{ .rows = hi - lo, .d = c.m.cfg.hidden_size, .k = @intCast(mlx.getShape(c.oa)[1]), .e = @intCast(mlx.getShape(c.ly.gate_w)[0]), .top_k = c.ly.top_k, .iters = c.m.cfg.dsv4_hc_sinkhorn_iters, .norm = c.m.cfg.moe_route_norm and c.ly.top_k > 1 };
        const o = (try launch(sc, c.ks, p, weightsOf(c.ly), try sc.slice(c.oa, 0, lo, hi), try sc.slice(c.x, 0, lo, hi), &sa, c.eps, c.scal)) orelse return error.Declined;
        return .{ o.x1, o.h2, o.sf.pre, o.sf.post, o.sf.comb, o.idx, o.wts, none };
    }
};

test "dsv41 exact rows: ffn_in: each row of an R-row launch pair is its one-row launch's bytes (wo_b and the expand, the FFN norm, mixes and router), R = 1..6, FFN_IN_SINK on and off, layers 21 and 23" {
    if (!mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .ctx = 8 });
    defer fx.deinit();
    const ks = try build();
    defer free(ks);
    const R = decode.max_rows;
    var bad: usize = 0;
    for ([_]?[*:0]const u8{ null, "0" }) |sink| for ([_]usize{ 21, 23 }) |li| {
        const sw = env.override(.MLX_SERVE_DSV41_FFN_IN_SINK, sink);
        defer sw.restore();
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const ly = fx.layer(li);
        const x = try sc.keep(try fx.stream(R));
        const cpre = try sc.keep(try fx.cpre(R));
        const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{fx.m.cfg.router_scaling_factor}, &[_]c_int{1}, 1, .float32));
        var c: RowsCase = .{
            .sc = &sc,
            .ks = ks,
            .m = fx.m,
            .ly = ly,
            .oa = try sc.keep(try fx.normal(&.{ ci(R), 8192 }, act, 0, 1)),
            .x = x,
            .sa = (try dsv41.hcStage(&sc, fx.m, x, null, cpre, &ly.hc_attn, ly.attn_norm)).sp,
            .eps = try dsv41.hcEps(&sc, fx.m),
            .scal = scal,
        };
        var buf: [48]u8 = undefined;
        bad += try decode.rowsInvariant(try std.fmt.bufPrint(&buf, "ffn_in layer {d} SINK {s}", .{ li, if (sink == null) "on" else "off" }), &c);
    };
    try std.testing.expectEqual(@as(usize, 0), bad);
}
