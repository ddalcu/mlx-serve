//! Tensor-unit (NAX) prefill attention at q/k 192, v 128 (MiMo-V2): MLX's own
//! NAX flash-attention kernel with a separate value head dim, ported from oMLX
//! (`omlx/utils/nax_attention.py`, kernels in `kernels/nax_attention.metal`).
//! Causal, optionally banded to a sliding window, with learned sinks. Long key
//! ranges run as consecutive key-range passes that hand the fp32 row state on
//! (bit-identical to one dispatch), so a pass's K/V stays in the on-chip caches;
//! those passes also split the head dims over simdgroup pairs.
const std = @import("std");
const mlx = @import("mlx.zig");

const BK: c_int = 32;
const THREADS: c_int = 128;
/// Keys per pass, and the grid size from which passes pay (oMLX's measured defaults).
const PASS_KEYS: c_int = 8192;
const PASS_MIN_GROUPS: c_int = 512;
const MAX_PASSES = 64;

const SOURCE =
    \\  threadgroup float xchg[WN == 2 ? WM * 2 * 16 * 32 : 1];
    \\  omlx_nax::attention_nax_bdv<
    \\      T, 16 * WM, 32, 192, 128, WM, WN,
    \\      ALIGN_Q, ALIGN_K, true, WINDOW, HAS_SINKS,
    \\      FIRST, LAST, float>(
    \\      q, k, v, out,
    \\      reinterpret_cast<const device omlx_nax::AttnParams*>(params),
    \\      q_strides, k_strides, v_strides,
    \\      sinks, state, state_out, xchg,
    \\      simdgroup_index_in_threadgroup,
    \\      thread_index_in_simdgroup,
    \\      threadgroup_position_in_grid);
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const ins = [_][*:0]const u8{ "q", "k", "v", "sinks", "state", "params" };
    const outs = [_][*:0]const u8{ "out", "state_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const header = @embedFile("kernels/nax_tiles.metal") ++ @embedFile("kernels/nax_attention.metal");
    // Q/K/V are read through their strides (cache views, transposed projections).
    const k = mlx.mlx_fast_metal_kernel_new("msv_nax_attn_192_128", in_vec, out_vec, SOURCE, header, false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// Key-block boundaries of the passes over `nk` key blocks: one dispatch for a
/// sliding window (a tile reads only its band) or a grid too small to drift apart.
fn passEdges(out: *[MAX_PASSES + 1]c_int, groups: c_int, nk: c_int, window: c_int) []const c_int {
    const n_pass: c_int = if (window > 0 or groups < PASS_MIN_GROUPS)
        1
    else
        @max(1, @min(@min(nk, MAX_PASSES), @divTrunc(nk * BK + PASS_KEYS / 2, PASS_KEYS)));
    for (0..@intCast(n_pass + 1)) |i| out[i] = @divTrunc(@as(c_int, @intCast(i)) * nk, n_pass);
    return out[0..@intCast(n_pass + 1)];
}

/// Causal (bottom-right) attention, banded to the last `window` keys when > 0,
/// for q [B, H, L, 192] over k [B, Hk, S, 192], v [B, Hk, S, 128] (any strides
/// with a contiguous head dim), `sinks` [H] or null: [B, H, L, 128] in q's dtype.
/// Null off NAX and outside this shape.
pub fn attention(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, window: c_int, sinks: ?mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !@import("transformer.zig").verifyQmmNaxAvailable()) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (qs[3] != 192 or ks[3] != 192 or vs[3] != 128) return null;
    if (ks[0] != qs[0] or vs[0] != qs[0] or vs[1] != ks[1] or vs[2] != ks[2]) return null;
    if (ks[1] == 0 or @rem(qs[1], ks[1]) != 0 or qs[2] <= 8 or ks[2] < qs[2]) return null;
    const dt = mlx.mlx_array_dtype(q);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(k) != dt or mlx.mlx_array_dtype(v) != dt) return null;
    if (sinks) |sk| if (mlx.mlx_array_ndim(sk) != 1 or mlx.getShape(sk)[0] != qs[1]) return null;

    var edges_buf: [MAX_PASSES + 1]c_int = undefined;
    const nk = @divTrunc(ks[2] + BK - 1, BK);
    const edges = passEdges(&edges_buf, qs[0] * qs[1] * @divTrunc(qs[2] + 63, 64), nk, window);
    const out = try dispatch(s, q, k, v, scale, window, sinks, edges, if (edges.len > 2) 2 else 1);
    if (!engaged) {
        engaged = true;
        @import("log.zig").info("[attn] NAX 192/128 prefill attention engaged (window={d}, passes={d})\n", .{ window, edges.len - 1 });
    }
    return out;
}

/// The passes between consecutive `edges` (key blocks) at head-dim split `wn`.
fn dispatch(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, window: c_int, sinks: ?mlx.mlx_array, edges: []const c_int, wn: c_int) !mlx.mlx_array {
    const kern = try getKernel();
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const b = qs[0];
    const h = qs[1];
    const ql = qs[2];
    const kl = ks[2];
    const wm = @divExact(4, wn);
    const bq = 16 * wm;
    const nq = @divTrunc(ql + bq - 1, bq);
    const nq_aligned = @divTrunc(ql, bq);
    const nk = @divTrunc(kl + BK - 1, BK);
    const nk_aligned = @divTrunc(kl, BK);
    const dt = mlx.mlx_array_dtype(q);

    // Absent sinks and the first pass's state read a one-element placeholder.
    const zero = [_]f32{0};
    const one = [_]c_int{1};
    const dummy = mlx.mlx_array_new_data(&zero, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(dummy);
    var sinks_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinks_t);
    try mlx.check(mlx.mlx_astype(&sinks_t, sinks orelse dummy, dt, s));

    var state = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(state);
    try mlx.check(mlx.mlx_array_set(&state, dummy));
    const state_size = b * h * nq * bq * (128 + 2);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    for (edges[0 .. edges.len - 1], edges[1..], 0..) |kb_begin, kb_end, i| {
        const first = i == 0;
        const last = i == edges.len - 2;
        // omlx_nax::AttnParams: B H D qL kL gqa scale NQ NK NQ_aligned NK_aligned qL_rem kL_rem qL_off kb_begin kb_end.
        const pdata = [16]i32{ b, h, 192, ql, kl, @divExact(h, ks[1]), @bitCast(scale), nq, nk, nq_aligned, nk_aligned, ql - nq_aligned * bq, kl - nk_aligned * BK, kl - ql, kb_begin, kb_end };
        const params = mlx.mlx_array_new_data(&pdata, &[_]c_int{16}, 1, .int32);
        defer _ = mlx.mlx_array_free(params);

        const c = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        if (last) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ b, ql, h, 128 }, 4, dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &one, 1, .float32));
        } else {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &one, 1, dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{state_size}, 1, .float32));
        }
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, nq * THREADS, h, b));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, THREADS, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WM", wm));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WN", wn));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WINDOW", window));
        for ([_]struct { [*:0]const u8, bool }{
            .{ "ALIGN_Q", @rem(ql, bq) == 0 }, .{ "ALIGN_K", @rem(kl, BK) == 0 }, .{ "HAS_SINKS", sinks != null },
            .{ "FIRST", first },               .{ "LAST", last },
        }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, t[0], t[1]));

        const ins = [_]mlx.mlx_array{ q, k, v, sinks_t, state, params };
        const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var outs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kern, in_vec, c, s));
        if (last) {
            try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
        } else {
            try mlx.check(mlx.mlx_vector_array_get(&state, outs, 1));
        }
    }
    // [B, qL, H, 128] rows viewed as [B, H, qL, 128], like MLX's SDPA output.
    defer _ = mlx.mlx_array_free(out);
    var view = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_transpose_axes(&view, out, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    return view;
}

const testing = std.testing;

fn randBf16(rnd: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = rnd.floatNorm(f32) * 0.5;
    const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn maxAbsDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(af);
    try mlx.check(mlx.mlx_astype(&af, a, .float32, s));
    var bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf);
    try mlx.check(mlx.mlx_astype(&bf, b, .float32, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, af, bf, s));
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d, s));
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_max(&m, ad, false, s));
    try mlx.check(mlx.mlx_array_eval(m));
    var r: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&r, m));
    return if (std.math.isFinite(r)) r else std.math.inf(f32);
}

/// fp32 truth: causal (bottom-right), banded to `window` when > 0, sinks as one extra softmax column.
fn truth(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, window: c_int, sinks: ?mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var f: [4]mlx.mlx_array = undefined;
    for (&f, [_]mlx.mlx_array{ q, k, v, sinks orelse q }) |*dst, src| {
        dst.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(dst, src, .float32, s));
    }
    defer for (f) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const ql = mlx.getShape(q)[2];
    const kl = mlx.getShape(k)[2];
    const vis = try testing.allocator.alloc(bool, @intCast(ql * kl));
    defer testing.allocator.free(vis);
    for (0..@intCast(ql)) |i| for (0..@intCast(kl)) |j| {
        const p: c_int = @as(c_int, @intCast(i)) + kl - ql;
        const c: c_int = @intCast(j);
        vis[i * @as(usize, @intCast(kl)) + j] = c <= p and (window == 0 or c > p - window);
    };
    const mask = mlx.mlx_array_new_data(vis.ptr, &[_]c_int{ 1, 1, ql, kl }, 4, .bool_);
    defer _ = mlx.mlx_array_free(mask);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, f[0], f[1], f[2], scale, "array", mask, if (sinks != null) f[3] else .{ .ctx = null }, false, s));
    return out;
}

test "nax attention 192/128: causal, windowed and sunk prefill track fp32 attention" {
    if (mlx.noGpuBackend() or !@import("transformer.zig").verifyQmmNaxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(192128);
    const rnd = prng.random();
    // (heads, kv heads, queries, keys, window, sinks): ragged tails, a cached prefix, a sliding band.
    const cases = [_]struct { h: c_int, hk: c_int, ql: c_int, kl: c_int, window: c_int, sinks: bool }{
        .{ .h = 4, .hk = 2, .ql = 100, .kl = 300, .window = 0, .sinks = false },
        .{ .h = 8, .hk = 1, .ql = 128, .kl = 128, .window = 0, .sinks = false },
        .{ .h = 8, .hk = 2, .ql = 200, .kl = 333, .window = 128, .sinks = true },
        .{ .h = 4, .hk = 4, .ql = 64, .kl = 64, .window = 16, .sinks = false },
        .{ .h = 4, .hk = 4, .ql = 40, .kl = 40, .window = 8, .sinks = true },
    };
    for (cases) |cs| {
        const q = try randBf16(rnd, &.{ 1, cs.h, cs.ql, 192 }, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try randBf16(rnd, &.{ 1, cs.hk, cs.kl, 192 }, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try randBf16(rnd, &.{ 1, cs.hk, cs.kl, 128 }, s);
        defer _ = mlx.mlx_array_free(v);
        const sk: ?mlx.mlx_array = if (cs.sinks) try randBf16(rnd, &.{cs.h}, s) else null;
        defer if (sk) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const scale: f32 = 1.0 / @sqrt(192.0);
        const ours = (try attention(s, q, k, v, scale, cs.window, sk)) orelse return error.TestUnexpectedDecline;
        defer _ = mlx.mlx_array_free(ours);
        const ref = try truth(q, k, v, scale, cs.window, sk, s);
        defer _ = mlx.mlx_array_free(ref);
        // bf16 output of an fp32-accumulated kernel: a few bf16 ulps at |O| < 2.
        const err = try maxAbsDiff(ours, ref, s);
        if (err > 2e-2) {
            std.debug.print("nax attention {any}: max |d| {d} vs fp32\n", .{ cs, err });
            return error.TestExpectedEqual;
        }
    }
}

test "nax attention 192/128: key-range passes reproduce one dispatch bit for bit" {
    if (mlx.noGpuBackend() or !@import("transformer.zig").verifyQmmNaxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(192129);
    const rnd = prng.random();
    const q = try randBf16(rnd, &.{ 1, 4, 100, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try randBf16(rnd, &.{ 1, 2, 300, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try randBf16(rnd, &.{ 1, 2, 300, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);
    for ([_]c_int{ 1, 2 }) |wn| {
        const single = try dispatch(s, q, k, v, scale, 0, null, &.{ 0, 10 }, wn);
        defer _ = mlx.mlx_array_free(single);
        const passes = try dispatch(s, q, k, v, scale, 0, null, &.{ 0, 3, 6, 10 }, wn);
        defer _ = mlx.mlx_array_free(passes);
        try testing.expectEqual(@as(f32, 0), try maxAbsDiff(single, passes, s));
        const ref = try truth(q, k, v, scale, 0, null, s);
        defer _ = mlx.mlx_array_free(ref);
        try testing.expect(try maxAbsDiff(passes, ref, s) < 2e-2);
    }
}

test "nax attention: a long causal prefill splits into ~8k-key passes, a window never does" {
    var buf: [MAX_PASSES + 1]c_int = undefined;
    try testing.expectEqualSlices(c_int, &.{ 0, 1024 }, passEdges(&buf, 8192, 1024, 128));
    try testing.expectEqualSlices(c_int, &.{ 0, 1024 }, passEdges(&buf, 100, 1024, 0));
    try testing.expectEqualSlices(c_int, &.{ 0, 256, 512, 768, 1024 }, passEdges(&buf, 8192, 1024, 0));
}
