//! Split-K decode attention for MiMo's global layers (q/k 192, v 128). The query
//! heads of one KV group, times the query positions (one at decode, up to 8 in an
//! MTP verify), are the rows of one MMA tile, so a split reads its keys once for
//! every head that shares them. Threadgroup = (rows / 8) row groups x KS key
//! slices over a staged 32-key tile; the slices merge in threadgroup memory and a
//! second dispatch merges the splits. MLX's vector kernel re-reads the keys per
//! query head and declines verify widths (rows x GQA > 32) to its composed path.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const ATTN256_KERNEL_HEADER = @import("attn256.zig").ATTN256_KERNEL_HEADER;

/// Query positions a dispatch takes (an MTP verify window).
pub const MAX_ROWS: c_int = 8;
/// Below this many keys one query position keeps MLX's vector kernel.
const MIN_KEYS_ONE_ROW: c_int = 8192;
/// Threadgroups a dispatch aims for across all KV heads.
const TARGET_TGS: c_int = 128;
const BK: c_int = 32;

const SPLIT_SOURCE = kernelSource("kernels/attention/dec_attn_split.metal");

/// One threadgroup per (head, position) row; a split no key of the row reached has l = 0.
const MERGE_SOURCE = kernelSource("kernels/attention/dec_attn_merge.metal");

var split_kernel: ?mlx.mlx_fast_metal_kernel = null;
var merge_kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // K/V are cache views: read through their strides, never copied.
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// Attention for q [1, Hq, S, 192] over the cache views k [1, Hk, N, 192] and
/// v [1, Hk, N, 128], causal at the bottom right: [1, Hq, S, 128] in q's dtype.
/// Null outside the measured shape and where MLX's own kernel is as fast.
pub fn attention(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (qs[0] != 1 or ks[0] != 1 or vs[0] != 1) return null;
    if (qs[3] != 192 or ks[3] != 192 or vs[3] != 128) return null;
    const hq = qs[1];
    const rows = qs[2];
    const hk = ks[1];
    const n = ks[2];
    if (vs[1] != hk or vs[2] != n or hk <= 0 or @rem(hq, hk) != 0) return null;
    const g = @divExact(hq, hk);
    if (rows < 1 or rows > MAX_ROWS or @rem(g * rows, 8) != 0 or n < rows) return null;
    if (rows == 1 and n < MIN_KEYS_ONE_ROW) return null;
    const dt = mlx.mlx_array_dtype(q);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(k) != dt or mlx.mlx_array_dtype(v) != dt) return null;

    if (split_kernel == null) split_kernel = try makeKernel("msv_dec_attn_split", &.{ "q", "k", "v", "scl" }, &.{ "pacc", "pm", "pl" }, SPLIT_SOURCE, ATTN256_KERNEL_HEADER);
    if (merge_kernel == null) merge_kernel = try makeKernel("msv_dec_attn_merge", &.{ "pacc", "pm", "pl" }, &.{"out"}, MERGE_SOURCE, "");

    const rg = @divExact(g * rows, 8);
    // Eight simdgroups per threadgroup while the rows leave room for key slices.
    const ksl: c_int = if (rg < 8) @divTrunc(8, rg) else 1;
    const splits: c_int = @max(1, @min(@divTrunc(TARGET_TGS, hk), @divTrunc(n + BK - 1, BK)));
    const r_all = hq * rows;

    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);

    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ r_all, splits, 128 }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ r_all, splits }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ r_all, splits }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, splits * 32, hk * rg * ksl, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, rg * ksl, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
        for ([_]struct { [*:0]const u8, c_int }{ .{ "G", g }, .{ "S", rows }, .{ "BDK", 192 }, .{ "BDV", 128 }, .{ "KS", ksl }, .{ "BK", BK } }) |t|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
        const ins = [_]mlx.mlx_array{ q, k, v, scl };
        const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&parts, split_kernel.?, in_vec, c, s));
    }
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ 1, hq, rows, 128 }, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 128, r_all, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "BDV", 128));
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, merge_kernel.?, parts, c, s));
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    if (!engaged) {
        engaged = true;
        @import("../../util/log.zig").info("[attn] split-K decode attention engaged: rows={d} heads={d}/{d} keys={d} splits={d}\n", .{ rows, hq, hk, n, splits });
    }
    return out;
}

const testing = std.testing;

fn randArr(rnd: std.Random, shape: []const c_int, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = rnd.floatNorm(f32);
    const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, dt, s));
    return out;
}

fn maxAbsDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(af);
    try mlx.check(mlx.mlx_astype(&af, a, .float32, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, af, b, s));
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d, s));
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_max(&m, ad, false, s));
    try mlx.check(mlx.mlx_array_eval(m));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, m));
    return v;
}

test "dec_attn: decode and verify rows match fp32 attention no worse than MLX's bf16 kernel" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    // (kv heads, gqa, keys, query positions): a long decode, an MTP verify, a ragged short verify.
    const cases = [_][4]c_int{ .{ 4, 16, 9000, 1 }, .{ 4, 16, 3000, 4 }, .{ 4, 16, 40, 7 }, .{ 8, 8, 8200, 1 } };
    const none = mlx.mlx_array{ .ctx = null };
    for (cases) |cs| {
        const hk = cs[0];
        const hq = cs[0] * cs[1];
        const q = try randArr(rnd, &.{ 1, hq, cs[3], 192 }, .bfloat16, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try randArr(rnd, &.{ 1, hk, cs[2], 192 }, .bfloat16, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try randArr(rnd, &.{ 1, hk, cs[2], 128 }, .bfloat16, s);
        defer _ = mlx.mlx_array_free(v);
        const scale: f32 = 1.0 / @sqrt(192.0);
        const mode: [*:0]const u8 = if (cs[3] > 1) "causal" else "";

        const ours = (try attention(s, q, k, v, scale)) orelse return error.TestUnexpectedDecline;
        defer _ = mlx.mlx_array_free(ours);
        var stock = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stock);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&stock, q, k, v, scale, mode, none, none, false, s));

        var f32s: [3]mlx.mlx_array = undefined;
        for (&f32s, [_]mlx.mlx_array{ q, k, v }) |*dst, src| {
            dst.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(dst, src, .float32, s));
        }
        defer for (f32s) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var truth = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&truth, f32s[0], f32s[1], f32s[2], scale, mode, none, none, false, s));

        const e_ours = try maxAbsDiff(ours, truth, s);
        const e_stock = try maxAbsDiff(stock, truth, s);
        try testing.expect(std.math.isFinite(e_ours));
        if (e_ours > 2.0 * e_stock + 1e-3) {
            std.debug.print("dec_attn {any}: ours {d} vs MLX {d} from fp32\n", .{ cs, e_ours, e_stock });
            return error.TestExpectedEqual;
        }
    }
}

test "dec_attn: one query row below the crossover stays on MLX" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(3);
    const q = try randArr(prng.random(), &.{ 1, 64, 1, 192 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try randArr(prng.random(), &.{ 1, 4, 4096, 192 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try randArr(prng.random(), &.{ 1, 4, 4096, 128 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(v);
    try testing.expect((try attention(s, q, k, v, 0.07)) == null);
}
