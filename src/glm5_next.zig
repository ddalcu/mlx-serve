//! GLM-5-Next (glm5_next) pieces outside the trunk's shared layers: DeepSeek-V4's
//! 4-stream Sinkhorn hyper-connection (mHC) around every sublayer, and the pooled
//! indexer + absorbed latent attention of its DeepSeek sparse attention layers.
//! Numerics follow mlx-vlm's `glm5_next` (the reference this port is held to); the
//! fused Sinkhorn-collapse kernel is mlx-vlm's `hc_sinkhorn_collapse`.
const std = @import("std");
const mlx = @import("mlx.zig");

pub const HC: c_int = 4;
const MIX: c_int = (2 + HC) * HC;

// ── mHC ────────────────────────────────────────────────────────────────

/// mlx-vlm `deepseek_v4/hyper_connection.py` `_make_hc_sinkhorn_collapse_kernel`
/// (Copyright (c) 2026 Apple Inc., MIT): sigmoid pre/post gates and the Sinkhorn-normalized
/// comb from one row's mixes, on simdgroup 0.
const GATES_HEADER =
    \\template <int HC, int ITERS, typename M, typename B, typename PO, typename CO>
    \\inline void msv_hc_gates(M mix, B base, float pre_scale, float post_scale, float comb_scale,
    \\                         float EPS, uint lane, threadgroup float* pre_shared,
    \\                         PO post_out, CO comb_out) {
    \\    constexpr int BASE_OFF = 2 * HC;
    \\    const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
    \\    const uint  llane  = metal::min(lane, (uint)(HC - 1));
    \\    float pre_z  = mix[llane]      * pre_scale  + base[llane];
    \\    float post_z = mix[HC + llane] * post_scale + base[HC + llane];
    \\    float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
    \\    float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    \\    if (lane < (uint)HC) {
    \\        pre_shared[lane] = pre_v;
    \\        post_out[lane]   = post_v;
    \\    }
    \\    const int o = BASE_OFF + int(llane) * HC;
    \\    float4 v = (float4(mix[o], mix[o + 1], mix[o + 2], mix[o + 3]) * comb_scale
    \\              + float4(base[o], base[o + 1], base[o + 2], base[o + 3])) * active;
    \\    float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    \\    float4 e = metal::fast::exp(v - row_max) * active;
    \\    float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;
    \\    float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
    \\    r *= col_inv;
    \\    for (int iter = 1; iter < ITERS; ++iter) {
    \\        r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
    \\        col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
    \\        r *= col_inv;
    \\    }
    \\    if (lane < (uint)HC) {
    \\        comb_out[lane * HC] = r.x;
    \\        comb_out[lane * HC + 1] = r.y;
    \\        comb_out[lane * HC + 2] = r.z;
    \\        comb_out[lane * HC + 3] = r.w;
    \\    }
    \\}
    \\
    \\// msv_hc_gates' post and comb on one thread (HC == 4): no cross-lane sums per iteration.
    \\template <int ITERS, typename M, typename B, typename PO, typename CO>
    \\inline void msv_hc_gates1(M mix, B base, float post_scale, float comb_scale, float EPS,
    \\                          PO post_out, CO comb_out) {
    \\    for (int i = 0; i < 4; ++i)
    \\        post_out[i] = 2.0f / (1.0f + metal::fast::exp(-(mix[4 + i] * post_scale + base[4 + i])));
    \\    float4 r[4];
    \\    for (int i = 0; i < 4; ++i) {
    \\        const int o = 8 + 4 * i;
    \\        const float4 v = float4(mix[o], mix[o + 1], mix[o + 2], mix[o + 3]) * comb_scale
    \\                       + float4(base[o], base[o + 1], base[o + 2], base[o + 3]);
    \\        const float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    \\        const float4 e = metal::fast::exp(v - row_max);
    \\        r[i] = e * (metal::fast::divide(1.0f, e.x + e.y + e.z + e.w + EPS)) + EPS;
    \\    }
    \\    float4 ci = metal::fast::divide(1.0f, (r[0] + r[1]) + (r[2] + r[3]) + EPS);
    \\    for (int i = 0; i < 4; ++i) r[i] *= ci;
    \\    for (int iter = 1; iter < ITERS; ++iter) {
    \\        for (int i = 0; i < 4; ++i) r[i] *= metal::fast::divide(1.0f, r[i].x + r[i].y + r[i].z + r[i].w + EPS);
    \\        ci = metal::fast::divide(1.0f, (r[0] + r[1]) + (r[2] + r[3]) + EPS);
    \\        for (int i = 0; i < 4; ++i) r[i] *= ci;
    \\    }
    \\    for (int i = 0; i < 4; ++i) {
    \\        comb_out[4 * i] = r[i].x;
    \\        comb_out[4 * i + 1] = r[i].y;
    \\        comb_out[4 * i + 2] = r[i].z;
    \\        comb_out[4 * i + 3] = r[i].w;
    \\    }
    \\}
    \\
    \\template <typename T>
    \\inline float4 msv_hc_load4(device T* h, int k) {
    \\    device atomic_uint* w = (device atomic_uint*)(h + k);
    \\    const uint a = atomic_load_explicit(w, memory_order_relaxed);
    \\    const uint b = atomic_load_explicit(w + 1, memory_order_relaxed);
    \\    return float4(float2(as_type<vec<T, 2>>(a)), float2(as_type<vec<T, 2>>(b)));
    \\}
    \\
    \\// Stream element k of `expand(y, h)`, rounded where `expand` rounds it.
    \\template <typename T, int HC, int D, typename Y, typename P, typename C, typename H>
    \\inline float msv_hc_hval(int k, Y y, P post, C comb, H h) {
    \\    const int i = k / D;
    \\    const int d = k - i * D;
    \\    float acc = 0.0f;
    \\    for (int j = 0; j < HC; ++j) acc = fma(comb[j * HC + i], float(h[j * D + d]), acc);
    \\    return float(T(fma(post[i], float(y[d]), acc)));
    \\}
;

/// The gates, then the pre-weighted collapse of the 4 streams, one row per threadgroup.
const COLLAPSE_SOURCE =
    \\uint tid  = thread_position_in_threadgroup.x;
    \\uint row  = threadgroup_position_in_grid.x;
    \\uint lane = tid % 32;
    \\uint sg   = tid / 32;
    \\constexpr int MIX = (2 + HC) * HC;
    \\constexpr float EPS = EPS_INT * 1e-9;
    \\threadgroup float pre_shared[HC];
    \\if (sg == 0) {
    \\    msv_hc_gates<HC, ITERS>(mixes + row * MIX, base, scale[0], scale[1], scale[2], EPS, lane,
    \\                           pre_shared, post + row * HC, comb + row * HC * HC);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const float p0 = pre_shared[0];
    \\const float p1 = pre_shared[1];
    \\const float p2 = pre_shared[2];
    \\const float p3 = pre_shared[3];
    \\const device T* x_row  = (const device T*)x_in + row * (HC * D);
    \\device U*       out_row = (device U*)collapsed + row * D;
    \\using T4 = vec<T, 4>;
    \\using U4 = vec<U, 4>;
    \\const device T4* x_row0 = (const device T4*)(x_row + 0*D);
    \\const device T4* x_row1 = (const device T4*)(x_row + 1*D);
    \\const device T4* x_row2 = (const device T4*)(x_row + 2*D);
    \\const device T4* x_row3 = (const device T4*)(x_row + 3*D);
    \\device U4*       out4   = (device U4*)out_row;
    \\constexpr uint D4 = (uint)D / 4;
    \\for (uint d4 = tid; d4 < D4; d4 += 256) {
    \\    float4 x0 = float4(x_row0[d4]);
    \\    float4 x1 = float4(x_row1[d4]);
    \\    float4 x2 = float4(x_row2[d4]);
    \\    float4 x3 = float4(x_row3[d4]);
    \\    float4 result = fma(float4(p0), x0, fma(float4(p1), x1, fma(float4(p2), x2, float4(p3) * x3)));
    \\    out4[d4] = U4(result);
    \\}
;

var collapse_kernel: ?mlx.mlx_fast_metal_kernel = null;
pub const Collapsed = struct {
    /// [B, S, D] in the stream's dtype: the sublayer input before its norm.
    x: mlx.mlx_array,
    /// f32 [B, S, HC]
    post: mlx.mlx_array,
    /// f32 [B, S, HC, HC]
    comb: mlx.mlx_array,

    pub fn deinit(self: *Collapsed) void {
        for ([_]mlx.mlx_array{ self.x, self.post, self.comb }) |a| _ = mlx.mlx_array_free(a);
    }
};

/// `rms_norm(flatten(stream)) @ fn^T` per row, straight from the bf16 stream: 8 rows per
/// threadgroup on simdgroup matrices, SG simdgroups splitting K = HC*D, the RMS scale applied
/// after the dot products.
const MIXES_SOURCE =
    \\const uint tg = threadgroup_position_in_grid.x;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint tid = sg * 32 + lane;
    \\constexpr int K = HC * D;
    \\constexpr int MIX = (2 + HC) * HC;
    \\constexpr int KS = K / SG;
    \\const int rows = params;
    \\const short qid = short(lane / 4);
    \\const short mr = (qid & 4) + short((lane / 2) % 4);
    \\const short mc = (qid & 2) * 2 + short(lane % 2) * 2;
    \\const int r = int(tg) * 8 + mr;
    \\const bool rv = r < rows;
    \\const device T* xrow = stream + (size_t)(rv ? r : 0) * K;
    \\simdgroup_matrix<float, 8, 8> a, b;
    \\simdgroup_matrix<float, 8, 8> c[MIX / 8];
    \\for (int t = 0; t < MIX / 8; ++t) c[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\float ss = 0.0f;
    \\for (int k0 = int(sg) * KS; k0 < int(sg + 1) * KS; k0 += 8) {
    \\    const float2 av = rv ? float2(float(xrow[k0 + mc]), float(xrow[k0 + mc + 1])) : float2(0.0f);
    \\    ss = fma(av.x, av.x, fma(av.y, av.y, ss));
    \\    reinterpret_cast<thread float2&>(a.thread_elements()) = av;
    \\    const device float* wcol = fn + (size_t)mc * K + k0 + mr;
    \\    for (int t = 0; t < MIX / 8; ++t) {
    \\        reinterpret_cast<thread float2&>(b.thread_elements()) = float2(wcol[t * 8 * K], wcol[(t * 8 + 1) * K]);
    \\        simdgroup_multiply_accumulate(c[t], a, b, c[t]);
    \\    }
    \\}
    \\threadgroup float part[SG][8][MIX];
    \\threadgroup float ssp[SG][32];
    \\threadgroup float rowss[8];
    \\for (int t = 0; t < MIX / 8; ++t) {
    \\    const float2 cv = reinterpret_cast<thread float2&>(c[t].thread_elements());
    \\    part[sg][mr][t * 8 + mc] = cv.x;
    \\    part[sg][mr][t * 8 + mc + 1] = cv.y;
    \\}
    \\ssp[sg][lane] = ss;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (tid < 8) {
    \\    float acc = 0.0f;
    \\    for (uint g = 0; g < SG; ++g)
    \\        for (uint l = 0; l < 32; ++l) {
    \\            const short q = short(l / 4);
    \\            if ((q & 4) + short((l / 2) % 4) == short(tid)) acc += ssp[g][l];
    \\        }
    \\    rowss[tid] = acc;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (tid < 8 * MIX) {
    \\    const int row = int(tid) / MIX;
    \\    const int col = int(tid) % MIX;
    \\    const int gr = int(tg) * 8 + row;
    \\    if (gr < rows) {
    \\        float acc = 0.0f;
    \\        for (uint g = 0; g < SG; ++g) acc += part[g][row][col];
    \\        mixes[(size_t)gr * MIX + col] = acc * metal::precise::rsqrt(rowss[row] / float(K) + eps[0]);
    \\    }
    \\}
;
const MIXES_SG: c_int = 8;
const MIXES_KERNEL_MIN_ROWS: c_int = 512;
var mixes_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The per-token mixes `rms_norm(flatten(stream)) @ fn^T`, f32 [B, S, MIX]: the op chain below
/// `MIXES_KERNEL_MIN_ROWS`, the kernel above.
fn mixes(s: mlx.mlx_stream, stream: mlx.mlx_array, fn_rows: mlx.mlx_array, eps: f32) !mlx.mlx_array {
    const sh = mlx.getShape(stream);
    // A threadgroup per 8 rows streams the whole `fn`: at decode widths that is one core
    // reading 1.5 MB per sublayer, and MLX's GEMV spreads it instead.
    if (sh[0] * sh[1] < MIXES_KERNEL_MIN_ROWS) {
        var y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_astype(&y, stream, .float32, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, y, &[_]c_int{ sh[0], sh[1], sh[2] * sh[3] }, 3, s));
        var ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones);
        try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{sh[2] * sh[3]}, 1, .float32, s));
        var z = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(z);
        try mlx.check(mlx.mlx_fast_rms_norm(&z, flat, ones, eps, s));
        var fn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(fn_t);
        try mlx.check(mlx.mlx_transpose(&fn_t, fn_rows, s));
        var m = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&m, z, fn_t, s));
        return m;
    }
    const rows = sh[0] * sh[1];
    const kern = mixes_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "stream", "fn", "params", "eps" };
        const outs = [_][*:0]const u8{"mixes"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("glm5_hc_mixes", iv, ov, MIXES_SOURCE, "#include <metal_simdgroup_matrix>\n", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        mixes_kernel = k;
        break :blk k;
    };
    const params = mlx.mlx_array_new_int(rows);
    defer _ = mlx.mlx_array_free(params);
    const ev = [1]f32{eps};
    const eps_a = mlx.mlx_array_new_data(&ev, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(eps_a);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], MIX }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(rows + 7, 8) * 32 * MIXES_SG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * MIXES_SG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx.mlx_array_dtype(stream)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", sh[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SG", MIXES_SG));
    const inputs = [_]mlx.mlx_array{ stream, fn_rows, params, eps_a };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var m = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_vector_array_get(&m, ov, 0));
    return m;
}

/// Collapse the [B, S, HC, D] stream to one sublayer input, with the gates the
/// matching `expand` needs.
pub fn collapse(s: mlx.mlx_stream, stream: mlx.mlx_array, fn_rows: mlx.mlx_array, scale: mlx.mlx_array, base: mlx.mlx_array, iters: u32, hc_eps: f32, norm_eps: f32) !Collapsed {
    const sh = mlx.getShape(stream);
    if (sh.len != 4 or sh[2] != HC or @rem(sh[3], 4) != 0) return error.Glm5HcShape;
    const m = try mixes(s, stream, fn_rows, norm_eps);
    defer _ = mlx.mlx_array_free(m);
    const kernel = collapse_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "x_in", "mixes", "scale", "base" };
        const outs = [_][*:0]const u8{ "collapsed", "post", "comb" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("glm5_hc_sinkhorn_collapse", iv, ov, COLLAPSE_SOURCE, GATES_HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        collapse_kernel = k;
        break :blk k;
    };
    const rows = sh[0] * sh[1];
    const dt = mlx.mlx_array_dtype(stream);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], sh[3] }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], HC }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], HC, HC }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "U", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", @intCast(iters)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", sh[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EPS_INT", @intFromFloat(@round(hc_eps / 1e-9))));
    const inputs = [_]mlx.mlx_array{ stream, m, scale, base };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var out: Collapsed = .{ .x = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new() };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.x, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.post, ov, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.comb, ov, 2));
    return out;
}

// One token, after oMLX `decode_kernels.py` `hc_pre_fused` (Apache-2.0, see NOTICE): the
// only dispatch between two sublayers. HC_PRE_NT threadgroups each take a K-slice of all the
// mixes; the last to finish (a device counter nothing resets: NT divides 2^32) reduces them,
// collapses and norms, then writes the post/comb gates. The sublayer's expand is left to the
// next `hcPre`, which rebuilds each stream element from (y, post, comb, h) as `expand` does.
const HC_PRE_TPG: c_int = 1024;
const HC_PRE_NT: c_int = 16;
/// Rows one `hcPre` dispatch takes (a verify window): one counter word each.
pub const HC_PRE_MAX_ROWS: c_int = 16;
/// Threads per threadgroup the GPU grants this kernel: Metal caps it per compiled
/// kernel by register use (M1/M2 below 1024), so the first eval probes and halves.
var hc_pre_tpg: c_int = HC_PRE_TPG;
var hc_pre_probed = [2]bool{ false, false };
const HC_PRE_BODY =
    \\constexpr int K = HC * D;
    \\constexpr int MIX = (2 + HC) * HC;
    \\constexpr int SL = K / NT;
    \\constexpr float EPS = EPS_INT * 1e-9;
    \\const uint g = threadgroup_position_in_grid.x;
    \\const uint row = threadgroup_position_in_grid.y;
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint lane = tid % 32;
    \\const uint sg = tid / 32;
    \\float acc[MIX + 1];
    \\for (int r = 0; r <= MIX; ++r) acc[r] = 0.0f;
    \\for (int k = int(g) * SL + int(tid); k < int(g + 1) * SL; k += TPG) {
    \\    const float hv = HVAL(k);
    \\    STORE_H(k, hv);
    \\    acc[MIX] = fma(hv, hv, acc[MIX]);
    \\    for (int r = 0; r < MIX; ++r) acc[r] = fma(hv, fn[r * K + k], acc[r]);
    \\}
    \\constexpr int NSG = TPG / 32;
    \\threadgroup float red[NSG][MIX + 1];
    \\threadgroup float tot[MIX + 1];
    \\threadgroup bool last;
    \\for (int r = 0; r <= MIX; ++r) {
    \\    const float v = simd_sum(acc[r]);
    \\    if (lane == 0) red[sg][r] = v;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\// Partials cross threadgroups through L2: plain loads can hit another core's stale L1.
    \\if (tid <= uint(MIX)) {
    \\    float t = 0.0f;
    \\    for (int q = 0; q < NSG; ++q) t += red[q][tid];
    \\    atomic_store_explicit((device atomic_float*)parts + (row * NT + g) * (MIX + 1) + tid, t, memory_order_relaxed);
    \\}
    \\atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\threadgroup_barrier(mem_flags::mem_device);
    \\if (tid == 0) {
    \\    const uint old = atomic_fetch_add_explicit((device atomic_uint*)counter + row, 1u, memory_order_relaxed);
    \\    last = ((old + 1u) % uint(NT)) == 0u;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (!last) return;
    \\atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\threadgroup float all[NT * (MIX + 1)];
    \\for (uint q = tid; q < uint(NT * (MIX + 1)); q += TPG) all[q] = atomic_load_explicit((device atomic_float*)parts + row * NT * (MIX + 1) + q, memory_order_relaxed);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (tid <= uint(MIX)) {
    \\    float t = 0.0f;
    \\    for (int q = 0; q < NT; ++q) t += all[q * (MIX + 1) + tid];
    \\    tot[tid] = t;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const float inv = metal::precise::rsqrt(tot[MIX] / float(K) + eps[0]);
    \\threadgroup float mix_n[MIX];
    \\if (tid < uint(MIX)) mix_n[tid] = tot[tid] * inv;
    \\float p[HC];
    \\for (int j = 0; j < HC; ++j) {
    \\    const float z = tot[j] * inv * scale[0] + base[j];
    \\    p[j] = 1.0f / (1.0f + metal::fast::exp(-z)) + EPS;
    \\}
    \\constexpr int D4 = D / 4;
    \\constexpr int CPT = (D4 + TPG - 1) / TPG;
    \\float4 cf[CPT];
    \\float s2 = 0.0f;
    \\for (int u = 0; u < CPT; ++u) {
    \\    cf[u] = 0.0f;
    \\    const int d4 = u * TPG + int(tid);
    \\    if (d4 < D4) {
    \\        const int d = 4 * d4;
    \\        float4 xs[HC];
    \\        for (int j = 0; j < HC; ++j) xs[j] = TAIL4(j * D + d);
    \\        const float4 c = fma(float4(p[0]), xs[0], fma(float4(p[1]), xs[1], fma(float4(p[2]), xs[2], float4(p[3]) * xs[3])));
    \\        cf[u] = float4(vec<T, 4>(c));
    \\        s2 += dot(cf[u], cf[u]);
    \\    }
    \\}
    \\s2 = simd_sum(s2);
    \\threadgroup float red2[NSG];
    \\if (lane == 0) red2[sg] = s2;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg == 0) {
    \\    const float t = simd_sum(lane < uint(NSG) ? red2[lane] : 0.0f);
    \\    if (lane == 0) red2[0] = t;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const float inv2 = metal::precise::rsqrt(red2[0] / float(D) + eps[1]);
    \\for (int u = 0; u < CPT; ++u) {
    \\    const int d4 = u * TPG + int(tid);
    \\    if (d4 < D4) ((device vec<T, 4>*)(normed + row * D))[d4] = ((const device vec<T, 4>*)nw)[d4] * vec<T, 4>(cf[u] * inv2);
    \\}
    \\if (tid == 0) msv_hc_gates1<ITERS>((threadgroup const float*)mix_n, base, scale[1], scale[2], EPS, post + row * HC, comb + row * HC * HC);
;
const HC_DEFER_HVAL = "#define HVAL(k) msv_hc_hval<T, HC, D>((k), y + row * D, post_in + row * HC, comb_in + row * HC * HC, h_in + row * HC * D)\n";
const HC_PRE_SOURCE = "#define HVAL(k) float(x[row * K + (k)])\n#define STORE_H(k, v)\n#define TAIL4(k) float4(*(const device vec<T, 4>*)(x + row * K + (k)))\n" ++ HC_PRE_BODY;
// The tail reads the stream the other threadgroups stored, two elements per 32-bit atomic.
const HC_PRE_DEFER_SOURCE = HC_DEFER_HVAL ++ "#define STORE_H(k, v) h[row * K + (k)] = T(v)\n#define TAIL4(k) msv_hc_load4<T>(h + row * K, (k))\n" ++ HC_PRE_BODY;
const HC_MATERIALIZE_SOURCE = HC_DEFER_HVAL ++ "const int k = int(thread_position_in_grid.x);\nconst uint row = thread_position_in_grid.y;\nh[row * HC * D + k] = T(HVAL(k));\n";
var hc_pre_kernel: ?mlx.mlx_fast_metal_kernel = null;
var hc_pre_defer_kernel: ?mlx.mlx_fast_metal_kernel = null;
var hc_materialize_kernel: ?mlx.mlx_fast_metal_kernel = null;
/// `hcPre`'s arrival counter (word 0): every dispatch increments it, nothing resets it.
var hc_pre_counter: ?mlx.mlx_array = null;

/// A sublayer's expand not applied yet: the stream is `expand(y, h)` under post/comb.
pub const Deferred = struct {
    /// [1, R, D] the sublayer's output.
    y: mlx.mlx_array,
    /// f32 [1, R, HC] and [1, R, HC, HC]: the sublayer's gates.
    post: mlx.mlx_array,
    comb: mlx.mlx_array,
    /// [1, R, HC, D] the sublayer's input stream.
    h: mlx.mlx_array,

    pub fn deinit(self: *Deferred) void {
        inline for (.{ self.y, self.post, self.comb, self.h }) |a| _ = mlx.mlx_array_free(a);
    }

    /// The [1, R, HC, D] stream this stands for, as `expand` writes it.
    pub fn materialize(self: *const Deferred, s: mlx.mlx_stream) !mlx.mlx_array {
        const d = mlx.getShape(self.y)[2];
        const rows = mlx.getShape(self.y)[1];
        const dt = mlx.mlx_array_dtype(self.y);
        const kern = try hcKernel(&hc_materialize_kernel, "glm5_hc_materialize", &.{ "y", "post_in", "comb_in", "h_in" }, &.{"h"}, HC_MATERIALIZE_SOURCE);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, HC, d }, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, HC * d, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
        const ins = [_]mlx.mlx_array{ self.y, self.post, self.comb, self.h };
        const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
        return out;
    }
};

pub const Pre = struct {
    /// [1, R, D] the sublayer's normed input.
    normed: mlx.mlx_array,
    /// f32 [1, R, HC] and [1, R, HC, HC]
    post: mlx.mlx_array,
    comb: mlx.mlx_array,
    /// [1, R, HC, D] the stream rebuilt from a `Deferred`, else empty.
    h: mlx.mlx_array,

    pub fn deinit(self: *Pre) void {
        inline for (.{ self.normed, self.post, self.comb, self.h }) |a| _ = mlx.mlx_array_free(a);
    }

    /// The `Deferred` this sublayer leaves: its output `y` over its input stream `h`.
    pub fn defer_(self: *const Pre, y: mlx.mlx_array, h: mlx.mlx_array) Deferred {
        var out: Deferred = .{ .y = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new(), .h = mlx.mlx_array_new() };
        _ = mlx.mlx_array_set(&out.y, y);
        _ = mlx.mlx_array_set(&out.post, self.post);
        _ = mlx.mlx_array_set(&out.comb, self.comb);
        _ = mlx.mlx_array_set(&out.h, h);
        return out;
    }
};

/// Whether `hcPre` serves a [1, R, HC, d] stream, R up to `HC_PRE_MAX_ROWS`.
pub fn hcRowsServes(s: mlx.mlx_stream, sh: []const c_int, dt: mlx.mlx_dtype, fn_rows: mlx.mlx_array, norm_w: mlx.mlx_array) bool {
    return mlx.streamIsGpu(s) and sh.len == 4 and sh[0] == 1 and sh[1] >= 1 and sh[1] <= HC_PRE_MAX_ROWS and sh[2] == HC and @rem(sh[3], 4) == 0 and
        @divTrunc(sh[3], 4) <= HC_PRE_TPG and @rem(HC * sh[3], HC_PRE_NT) == 0 and (dt == .bfloat16 or dt == .float16) and
        mlx.mlx_array_dtype(norm_w) == dt and mlx.mlx_array_dtype(fn_rows) == .float32;
}

fn hcKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, GATES_HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// The normed sublayer input and gates of R rows, from their stream `x` [1, R, HC, D] or
/// from the previous sublayer's `Deferred` (then `Pre.h` is the stream). Callers check
/// `hcRowsServes` first.
pub fn hcPre(s: mlx.mlx_stream, x: ?mlx.mlx_array, deferred: ?*const Deferred, fn_rows: mlx.mlx_array, scale: mlx.mlx_array, base: mlx.mlx_array, norm_w: mlx.mlx_array, iters: u32, hc_eps: f32, norm_eps: f32, rms_eps: f32) !Pre {
    const d = mlx.getShape(norm_w)[0];
    const dt = mlx.mlx_array_dtype(norm_w);
    // Rows ride the grid's y: each has its own K-slices, counter word and gates.
    const rows: c_int = if (x) |xa| mlx.getShape(xa)[1] else mlx.getShape(deferred.?.y)[1];
    if (rows < 1 or rows > HC_PRE_MAX_ROWS) return error.Glm5HcShape;
    const kern = if (deferred != null)
        try hcKernel(&hc_pre_defer_kernel, "glm5_hc_pre_defer", &.{ "y", "post_in", "comb_in", "h_in", "fn", "scale", "base", "nw", "eps", "counter" }, &.{ "normed", "post", "comb", "parts", "h" }, HC_PRE_DEFER_SOURCE)
    else
        try hcKernel(&hc_pre_kernel, "glm5_hc_pre", &.{ "x", "fn", "scale", "base", "nw", "eps", "counter" }, &.{ "normed", "post", "comb", "parts" }, HC_PRE_SOURCE);
    const counter = hc_pre_counter orelse blk: {
        var z = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(z);
        // One word per row; a counter of 8 or more words also keeps the input out of the
        // read-only constant address space.
        try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{HC_PRE_MAX_ROWS}, 1, .uint32, s));
        try mlx.check(mlx.mlx_array_eval(z));
        hc_pre_counter = z;
        break :blk z;
    };
    const ev = [2]f32{ norm_eps, rms_eps };
    const eps = mlx.mlx_array_new_data(&ev, &[_]c_int{2}, 1, .float32);
    defer _ = mlx.mlx_array_free(eps);
    var ins_buf: [10]mlx.mlx_array = undefined;
    const ins: []const mlx.mlx_array = if (deferred) |df| blk: {
        ins_buf[0..10].* = .{ df.y, df.post, df.comb, df.h, fn_rows, scale, base, norm_w, eps, counter };
        break :blk ins_buf[0..10];
    } else blk: {
        ins_buf[0..7].* = .{ x.?, fn_rows, scale, base, norm_w, eps, counter };
        break :blk ins_buf[0..7];
    };
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    while (true) {
        const tpg = hc_pre_tpg;
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, d }, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, HC }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, HC, HC }, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows * HC_PRE_NT, MIX + 1 }, 2, .float32));
        if (deferred != null) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, HC, d }, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tpg * HC_PRE_NT, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tpg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
        inline for (.{ .{ "HC", HC }, .{ "D", d }, .{ "NT", HC_PRE_NT }, .{ "TPG", tpg }, .{ "ITERS", @as(c_int, @intCast(iters)) }, .{ "EPS_INT", @as(c_int, @intFromFloat(@round(hc_eps / 1e-9))) } }) |kv| {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
        }
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
        var out: Pre = .{ .normed = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new(), .h = mlx.mlx_array_new() };
        errdefer out.deinit();
        try mlx.check(mlx.mlx_vector_array_get(&out.normed, ov, 0));
        try mlx.check(mlx.mlx_vector_array_get(&out.post, ov, 1));
        try mlx.check(mlx.mlx_vector_array_get(&out.comb, ov, 2));
        if (deferred != null) try mlx.check(mlx.mlx_vector_array_get(&out.h, ov, 4));
        const probed = &hc_pre_probed[@intFromBool(deferred != null)];
        if (probed.*) return out;
        // First launch: where Metal caps this kernel's threadgroup below `tpg`, halve and relaunch.
        mlx.check(mlx.mlx_array_eval(out.normed)) catch |e| {
            if (e != error.MlxError or tpg <= 256 or !mlx.takeErrorIf("Thread group size")) return e;
            out.deinit();
            hc_pre_tpg = @divTrunc(tpg, 2);
            @import("log.zig").info("[glm5] hcPre: {d} threads per threadgroup on this GPU\n", .{hc_pre_tpg});
            continue;
        };
        probed.* = true;
        return out;
    }
}

/// `post * x + comb^T @ stream` in f32, one thread per 4 columns of a row (all HC streams).
const EXPAND_SOURCE =
    \\const uint d4 = thread_position_in_grid.x;
    \\const uint row = thread_position_in_grid.y;
    \\if (d4 >= uint(D / 4)) return;
    \\const auto pr = post + row * HC;
    \\const auto cb = comb + row * HC * HC;
    \\const float4 xv = float4(*(const device vec<T, 4>*)(x + (size_t)row * D + 4 * d4));
    \\float4 sv[HC];
    \\for (int j = 0; j < HC; ++j)
    \\    sv[j] = float4(*(const device vec<T, 4>*)(res + ((size_t)row * HC + j) * D + 4 * d4));
    \\for (int i = 0; i < HC; ++i) {
    \\    float4 acc = 0.0f;
    \\    for (int j = 0; j < HC; ++j) acc = fma(float4(cb[j * HC + i]), sv[j], acc);
    \\    *(device vec<T, 4>*)(out + ((size_t)row * HC + i) * D + 4 * d4) = vec<T, 4>(fma(float4(pr[i]), xv, acc));
    \\}
;
var expand_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// `post * x + comb^T @ stream`, in f32, back to the stream's dtype: the new stream.
pub fn expand(s: mlx.mlx_stream, x: mlx.mlx_array, stream: mlx.mlx_array, c: *const Collapsed) !mlx.mlx_array {
    const sh = mlx.getShape(stream);
    const dt = mlx.mlx_array_dtype(stream);
    const kern = expand_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "x", "res", "post", "comb" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("glm5_hc_expand", iv, ov, EXPAND_SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        expand_kernel = k;
        break :blk k;
    };
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_astype(&xc, x, dt, s));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(sh[3], 4), sh[0] * sh[1], 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(256, @divTrunc(sh[3], 4)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", sh[3]));
    const inputs = [_]mlx.mlx_array{ xc, stream, c.post, c.comb };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

// ── DeepSeek sparse attention: token selection ─────────────────────────

pub const Indexer = struct {
    topk: c_int,
    kpool: c_int,
    heads: c_int,
    head_dim: c_int,
};

fn iota(s: mlx.mlx_stream, n: c_int, start: c_int) !mlx.mlx_array {
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_arange(&a, @floatFromInt(start), @floatFromInt(start + n), 1, .int32, s));
    return a;
}

fn scalarI(v: i32) mlx.mlx_array {
    return mlx.mlx_array_new_int(v);
}

/// oMLX `patches/glm_moe_dsa/indexer_nax.py` (Apache-2.0, see NOTICE): the indexer's
/// head-summed `relu(q . k) * w` per (query, pool) on the tensor units, the causal pool mask
/// folded into the epilogue. Here the scores leave in f32 (`O`), as the reference ranks them.
const INDEXER_NAX_SOURCE =
    \\    constexpr int D = 128;
    \\    const int S = params[0];
    \\    const int P = params[1];
    \\    const int before = params[2];
    \\    const int pool_len = params[3];
    \\    const int ratio = params[4];
    \\    const int H = params[5];
    \\    const int q_row_stride = H * D;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint lane = thread_index_in_simdgroup;
    \\    const int sm = int(threadgroup_position_in_grid.y) * (32 * WM) + int(sg / WN) * 32;
    \\    const int sn = int(threadgroup_position_in_grid.x) * (32 * WN) + int(sg % WN) * 32;
    \\
    \\    const short qid = short(lane >> 2);
    \\    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    \\    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    \\
    \\    // Every score of the block is masked when its first pooled key is not
    \\    // yet complete for the block's last query row.
    \\    const int s_last = min(sm + 31, S - 1);
    \\    const bool live = sm < S && sn < P && sn < pool_len &&
    \\        (sn + 1) * ratio - 1 <= before + s_last;
    \\
    \\    float acc[2][16];
    \\    for (short a = 0; a < 2; ++a) {
    \\        for (short e = 0; e < 16; ++e) {
    \\            acc[a][e] = 0.0f;
    \\        }
    \\    }
    \\
    \\    int qrow[2][2];
    \\    for (short tm = 0; tm < 2; ++tm) {
    \\        for (short i = 0; i < 2; ++i) {
    \\            qrow[tm][i] = min(sm + tm * 16 + fm + i * 8, S - 1);
    \\        }
    \\    }
    \\
    \\    if (live) {
    \\        constexpr auto desc = matmul2d_descriptor(
    \\            16, 32, 16, false, true, true,
    \\            matmul2d_descriptor::mode::multiply_accumulate);
    \\        matmul2d<desc, execution_simdgroup> op;
    \\
    \\        int krow[2][2];
    \\        for (short tn = 0; tn < 2; ++tn) {
    \\            for (short i = 0; i < 2; ++i) {
    \\                krow[tn][i] = min(sn + tn * 16 + fm + i * 8, P - 1);
    \\            }
    \\        }
    \\        const device T* kbase[2][2];
    \\        for (short tn = 0; tn < 2; ++tn) {
    \\            for (short i = 0; i < 2; ++i) {
    \\                kbase[tn][i] = k + ulong(krow[tn][i]) * D + fn;
    \\            }
    \\        }
    \\
    \\        for (int h = 0; h < H; ++h) {
    \\            auto ct_a0 = op.template get_left_input_cooperative_tensor<T, T, float>();
    \\            auto ct_a1 = op.template get_left_input_cooperative_tensor<T, T, float>();
    \\            auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    \\            auto c0 = op.template get_destination_cooperative_tensor<
    \\                metal::remove_addrspace_t<decltype(ct_a0)>,
    \\                metal::remove_addrspace_t<decltype(ct_b)>, float>();
    \\            auto c1 = op.template get_destination_cooperative_tensor<
    \\                metal::remove_addrspace_t<decltype(ct_a0)>,
    \\                metal::remove_addrspace_t<decltype(ct_b)>, float>();
    \\            for (short e = 0; e < 16; ++e) {
    \\                c0[e] = 0.0f;
    \\                c1[e] = 0.0f;
    \\            }
    \\            const device T* a00 = q + ulong(qrow[0][0]) * q_row_stride + h * D + fn;
    \\            const device T* a01 = q + ulong(qrow[0][1]) * q_row_stride + h * D + fn;
    \\            const device T* a10 = q + ulong(qrow[1][0]) * q_row_stride + h * D + fn;
    \\            const device T* a11 = q + ulong(qrow[1][1]) * q_row_stride + h * D + fn;
    \\            for (short kk = 0; kk < D; kk += 16) {
    \\                for (short j = 0; j < 4; ++j) {
    \\                    ct_a0[j] = a00[kk + j];
    \\                    ct_a0[4 + j] = a01[kk + j];
    \\                    ct_a1[j] = a10[kk + j];
    \\                    ct_a1[4 + j] = a11[kk + j];
    \\                }
    \\                for (short tn = 0; tn < 2; ++tn) {
    \\                    for (short i = 0; i < 2; ++i) {
    \\                        for (short j = 0; j < 4; ++j) {
    \\                            ct_b[tn * 8 + i * 4 + j] = kbase[tn][i][kk + j];
    \\                        }
    \\                    }
    \\                }
    \\                op.run(ct_a0, ct_b, c0);
    \\                op.run(ct_a1, ct_b, c1);
    \\            }
    \\            // relu * head weight in fp32, heads accumulated in order.
    \\            for (short i = 0; i < 2; ++i) {
    \\                const float w0 = float(w[ulong(qrow[0][i]) * H + h]);
    \\                const float w1 = float(w[ulong(qrow[1][i]) * H + h]);
    \\                for (short tn = 0; tn < 2; ++tn) {
    \\                    for (short j = 0; j < 4; ++j) {
    \\                        const short e = tn * 8 + i * 4 + j;
    \\                        acc[0][e] += max(float(c0[e]), 0.0f) * w0;
    \\                        acc[1][e] += max(float(c1[e]), 0.0f) * w1;
    \\                    }
    \\                }
    \\            }
    \\        }
    \\    }
    \\
    \\    const O masked = O(-1e30f);
    \\    for (short tm = 0; tm < 2; ++tm) {
    \\        for (short i = 0; i < 2; ++i) {
    \\            const int s = sm + tm * 16 + fm + i * 8;
    \\            if (s >= S) {
    \\                continue;
    \\            }
    \\            for (short tn = 0; tn < 2; ++tn) {
    \\                for (short j = 0; j < 4; ++j) {
    \\                    const int p = sn + tn * 16 + fn + j;
    \\                    if (p >= P) {
    \\                        continue;
    \\                    }
    \\                    const bool valid = p < pool_len && (p + 1) * ratio - 1 <= before + s;
    \\                    out[ulong(s) * P + p] = valid ? O(acc[tm][tn * 8 + i * 4 + j]) : masked;
    \\                }
    \\            }
    \\        }
    \\    }
;
const INDEXER_WM: c_int = 2;
const INDEXER_WN: c_int = 2;
var indexer_nax_kernel: ?mlx.mlx_fast_metal_kernel = null;
var indexer_nax_engaged = false;

/// Masked scores f32 [1, S, P] of queries at positions `q_start..` (`q_idx` [1, S, H, 128],
/// `w` [1, S, H] scaled) against pooled keys `pk` [1, P, 128] (`pool_len` of them complete,
/// `kpool` tokens each); a pool a query cannot see scores -1e30. Null off NAX or outside the shape.
fn indexerScoresNax(s: mlx.mlx_stream, q_idx: mlx.mlx_array, w: mlx.mlx_array, pk: mlx.mlx_array, q_start: c_int, kpool: c_int) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !@import("transformer.zig").verifyQmmNaxAvailable()) return null;
    const qs = mlx.getShape(q_idx);
    const ps = mlx.getShape(pk);
    if (qs.len != 4 or qs[0] != 1 or qs[3] != 128 or ps.len != 3 or ps[0] != 1 or ps[2] != 128) return null;
    if (mlx.mlx_array_dtype(q_idx) != .bfloat16 or mlx.mlx_array_dtype(pk) != .bfloat16) return null;
    const sq = qs[1];
    const h = qs[2];
    const p = ps[1];
    if (sq < 1 or p < 1) return null;
    const kern = indexer_nax_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "q", "k", "w", "params" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("msv_glm5_indexer_nax", iv, ov, INDEXER_NAX_SOURCE, SPARSE_NAX_HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        indexer_nax_kernel = k;
        break :blk k;
    };
    const pdata = [6]i32{ sq, p, q_start, p, kpool, h };
    const params = mlx.mlx_array_new_data(&pdata, &[_]c_int{6}, 1, .int32);
    defer _ = mlx.mlx_array_free(params);
    const tg_x = @divTrunc(p + 32 * INDEXER_WN - 1, 32 * INDEXER_WN);
    const tg_y = @divTrunc(sq + 32 * INDEXER_WM - 1, 32 * INDEXER_WM);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, sq, p }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tg_x * INDEXER_WM * INDEXER_WN * 32, tg_y, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, INDEXER_WM * INDEXER_WN * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "O", .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WM", INDEXER_WM));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WN", INDEXER_WN));
    const inputs = [_]mlx.mlx_array{ q_idx, pk, w, params };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    if (!indexer_nax_engaged) {
        indexer_nax_engaged = true;
        @import("log.zig").info("[dsa] NAX indexer scores engaged\n", .{});
    }
    return out;
}

/// A forward's indexer rows (key | gate) as the cache stores them: the row that completes a
/// pool (position % kpool == kpool - 1) carries the pool's key in its gate half, the gates'
/// softmax-weighted keys (only pooled keys are read once a pool is whole). One thread per
/// (row, channel); `prev` holds the up-to kpool - 1 cached rows before the forward.
const POOLED_ROWS_SOURCE =
    \\const uint d = thread_position_in_grid.x;
    \\const int r = int(thread_position_in_grid.y);
    \\const int b = int(thread_position_in_grid.z);
    \\const int S = int(threads_per_grid.y);
    \\const int pos0 = dims[0];
    \\const int prev_len = dims[1];
    \\const device T* crow = chunk + (size_t(b) * S + r) * (2 * DH);
    \\device T* orow = out + (size_t(b) * S + r) * (2 * DH);
    \\orow[d] = crow[d];
    \\const int pos = pos0 + r;
    \\if ((pos + 1) % KPOOL != 0) {
    \\    orow[DH + d] = crow[DH + d];
    \\    return;
    \\}
    \\float lg[KPOOL];
    \\T kv[KPOOL];
    \\float mx = -INFINITY;
    \\for (int j = 0; j < KPOOL; ++j) {
    \\    const int q = pos - KPOOL + 1 + j;
    \\    const device T* row = q >= pos0 ? chunk + (size_t(b) * S + (q - pos0)) * (2 * DH)
    \\                                    : prev + (size_t(b) * prev_len + (q - pos0 + prev_len)) * (2 * DH);
    \\    kv[j] = row[d];
    \\    lg[j] = float(row[DH + d]) + ape[j * DH + d];
    \\    mx = metal::max(mx, lg[j]);
    \\}
    \\float sum = 0.0f;
    \\for (int j = 0; j < KPOOL; ++j) { lg[j] = metal::precise::exp(lg[j] - mx); sum += lg[j]; }
    \\float acc = 0.0f;
    \\for (int j = 0; j < KPOOL; ++j) acc += float(T(T(lg[j] / sum) * kv[j]));
    \\orow[DH + d] = T(acc);
;
var pooled_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// `chunk` [B, S, 2*Dh] (rows at positions `pos0..`) with each completed pool's key in its
/// completing row's gate half; `prev` [B, n, 2*Dh] are the n <= kpool - 1 cached rows
/// before `pos0` (any one row when n is 0).
pub fn withPooledKeys(s: mlx.mlx_stream, prev: mlx.mlx_array, chunk: mlx.mlx_array, ape: mlx.mlx_array, pos0: c_int, ix: Indexer) !mlx.mlx_array {
    const csh = mlx.getShape(chunk);
    const dt = mlx.mlx_array_dtype(chunk);
    const prev_len: c_int = @min(pos0, mlx.getShape(prev)[1]);
    const kern = pooled_rows_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "prev", "chunk", "ape", "dims" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("glm5_dsa_pooled_rows", iv, ov, POOLED_ROWS_SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        pooled_rows_kernel = k;
        break :blk k;
    };
    const dims = mlx.mlx_array_new_data(&[_]i32{ pos0, prev_len }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, csh.ptr, csh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ix.head_dim, csh[1], csh[0]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ix.head_dim, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    inline for (.{ .{ "KPOOL", ix.kpool }, .{ "DH", ix.head_dim } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    return applyOne(s, kern, cfg, &.{ prev, chunk, ape, dims });
}

/// [B, P, Dh] keys of the first `p` pools: the completing rows' gate halves.
fn storedPoolKeys(s: mlx.mlx_stream, idx_cache: mlx.mlx_array, p: c_int, ix: Indexer) !mlx.mlx_array {
    const b = mlx.getShape(idx_cache)[0];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, idx_cache, &[_]c_int{ 0, ix.kpool - 1, ix.head_dim }, 3, &[_]c_int{ b, p * ix.kpool, 2 * ix.head_dim }, 3, &[_]c_int{ 1, ix.kpool, 1 }, 3, s));
    return out;
}

/// oMLX `decode_kernels.py` `_DSA_EXPAND_SOURCE` (Apache-2.0, see NOTICE): the selected pools
/// [S, SEL_K] of queries at `qpos..` to their token indices, the query's partial tail pool
/// after them, -1 for an unusable slot or the padding up to the output width.
const EXPAND_SEL_SOURCE =
    \\const int col = int(thread_position_in_grid.x);
    \\const int row = int(thread_position_in_grid.y);
    \\if (col >= OUT_W) {
    \\  return;
    \\}
    \\const int qp = int(qpos[0]) + row;
    \\const int pool_len = int(qpos[1]);
    \\int v = -1;
    \\if (col < SEL_K * KPOOL) {
    \\  const int s = int(selected[row * SEL_K + col / KPOOL]);
    \\  const bool valid = s < pool_len && (s + 1) * KPOOL - 1 <= qp;
    \\  if (valid) {
    \\    v = s * KPOOL + (col % KPOOL);
    \\  }
    \\} else if (col >= TOPK_W && col < TOPK_W + TAIL_W) {
    \\  const int t = col - TOPK_W;
    \\  const int tail_count = (qp + 1) % KPOOL;
    \\  if (t < tail_count) {
    \\    v = qp + 1 - tail_count + t;
    \\  }
    \\}
    \\out[row * OUT_W + col] = v;
;
var expand_sel_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// int32 [B, S, topk + kpool - 1] token indices from the selected pools `sel` [B, S, select_k].
fn expandSelection(s: mlx.mlx_stream, sel: mlx.mlx_array, q_start: c_int, p: c_int, ix: Indexer) !mlx.mlx_array {
    const ssh = mlx.getShape(sel);
    const kern = expand_sel_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "selected", "qpos" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("glm5_dsa_expand_sel", iv, ov, EXPAND_SEL_SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        expand_sel_kernel = k;
        break :blk k;
    };
    const out_w = ix.topk + ix.kpool - 1;
    const qv = [2]i32{ q_start, p };
    const qpos = mlx.mlx_array_new_data(&qv, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(qpos);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ssh[0], ssh[1], out_w }, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_w, ssh[0] * ssh[1], 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(256, out_w), 1, 1));
    inline for (.{ .{ "SEL_K", ssh[2] }, .{ "KPOOL", ix.kpool }, .{ "TOPK_W", ix.topk }, .{ "TAIL_W", ix.kpool - 1 }, .{ "OUT_W", out_w } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    const inputs = [_]mlx.mlx_array{ sel, qpos };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// `x` that also evaluates `dep`: a cache row nothing in this forward reads (the indexer below
/// its budget) would otherwise stay a lazy chain across tokens until the first read.
pub fn withDependency(x: mlx.mlx_array, dep: mlx.mlx_array) !mlx.mlx_array {
    const iv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{x}, 1);
    defer _ = mlx.mlx_vector_array_free(iv);
    const dv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{dep}, 1);
    defer _ = mlx.mlx_vector_array_free(dv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_depends(&ov, iv, dv));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Whether every query of a forward reaching `kv_len` keys reads every earlier key: the
/// pools all fit the budget, so the selection is plain causal attention.
pub fn selectsAll(kv_len: c_int, ix: Indexer) bool {
    return @divTrunc(kv_len, ix.kpool) <= @divTrunc(ix.topk, ix.kpool);
}

/// The token indices each query reads, int32 [B, S, topk + kpool - 1] with -1 for an
/// unused slot: the `topk/kpool` best complete pools ending at or before the query, in
/// pool order, padded to `topk`, then the query's partial tail pool.
/// `q_idx` [B, S, H, Dh] and `w` f32 [B, S, H] (already scaled by H^-0.5) score pools;
/// `idx_cache` [B, T, 2*Dh] holds each token's indexer key and pool gate. The queries are
/// the last S tokens; rows are scored in chunks of at most 2^27 (query, pool) scores.
pub fn selectTokens(s: mlx.mlx_stream, q_idx: mlx.mlx_array, w: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer) !mlx.mlx_array {
    const t = mlx.getShape(idx_cache)[1];
    const sq = mlx.getShape(q_idx)[1];
    const rows = @max(64, @divTrunc(@divTrunc(@as(c_int, 1 << 27), @max(1, @divTrunc(t, ix.kpool))), 64) * 64);
    if (sq <= rows) return selectRows(s, q_idx, w, idx_cache, ix, t - sq);
    var parts: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(std.heap.c_allocator);
    }
    var r0: c_int = 0;
    while (r0 < sq) : (r0 += rows) {
        const r1 = @min(sq, r0 + rows);
        const qsh = mlx.getShape(q_idx);
        var qc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qc);
        try mlx.check(mlx.mlx_slice(&qc, q_idx, &[_]c_int{ 0, r0, 0, 0 }, 4, &[_]c_int{ qsh[0], r1, qsh[2], qsh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        var wc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wc);
        try mlx.check(mlx.mlx_slice(&wc, w, &[_]c_int{ 0, r0, 0 }, 3, &[_]c_int{ qsh[0], r1, qsh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try parts.append(std.heap.c_allocator, try selectRows(s, qc, wc, idx_cache, ix, t - sq + r0));
    }
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
    return out;
}

/// One query's indexer scores in one dispatch: each simdgroup loads PPS pool keys in turn
/// (`withPooledKeys`' layout) and lane h scores head h against each; the head sum is the
/// pool's score, -FLT_MAX for a pool the query cannot see.
const DEC_SCORE_SGS: c_int = 8;
const DEC_SCORE_PPS: c_int = 4;
const DEC_SCORE_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint tid = sg * 32 + lane;
    \\const int q_pos = dims[0];
    \\const int P = dims[1];
    \\threadgroup float qT[DH * 32];
    \\threadgroup float kp[SGS][DH];
    \\for (uint i = tid; i < uint(32 * DH); i += SGS * 32) qT[(i % DH) * 32 + i / DH] = float(q[i]);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (int pp = 0; pp < PPS; ++pp) {
    \\const int p = (int(threadgroup_position_in_grid.y) * SGS + int(sg)) * PPS + pp;
    \\if (p >= P) return;
    \\// The pool's key: its completing row's gate half; lane l loads channels 4l..4l+3 (DH == 128).
    \\const device vec<T, 4>* pooled = (const device vec<T, 4>*)(cache + (size_t(p) * KPOOL + KPOOL - 1) * (2 * DH) + DH);
    \\((threadgroup float4*)kp[sg])[lane] = float4(pooled[lane]);
    \\simdgroup_barrier(mem_flags::mem_threadgroup);
    \\float dot = 0.0f;
    \\for (int d = 0; d < DH; ++d) dot = fma(qT[d * 32 + lane], kp[sg][d], dot);
    \\const float sc = simd_sum(w[lane] * metal::max(dot, 0.0f));
    \\if (lane == 0) out[p] = (p * KPOOL + KPOOL - 1 <= q_pos) ? sc : -FLT_MAX;
    \\simdgroup_barrier(mem_flags::mem_threadgroup);
    \\}
;
var dec_score_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The top SEL_K of one query's pool scores as token indices in `expandSelection`'s layout,
/// in one threadgroup: a 4-pass 8-bit radix select finds the SEL_K-th largest score, then an
/// ordered compaction (ties to the lower index) writes the pools chronologically.
const TOPK_EXPAND_SOURCE =
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint lane = tid % 32;
    \\const uint sg = tid / 32;
    \\const int q_pos = dims[0];
    \\const int P = dims[1];
    \\threadgroup atomic_uint hist[256];
    \\threadgroup uint st[3];
    \\threadgroup uint part_g[32];
    \\threadgroup uint part_e[32];
    \\if (tid == 0) { st[0] = 0u; st[1] = 0u; st[2] = uint(SEL_K); }
    \\for (int pass = 0; pass < 4; ++pass) {
    \\    const uint shift = uint(24 - 8 * pass);
    \\    if (tid < 256u) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const uint prefix = st[0];
    \\    const uint pmask = st[1];
    \\    for (int i = int(tid); i < P; i += 1024) {
    \\        const uint k = msv_score_key(scores[i]);
    \\        if ((k & pmask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (tid == 0) {
    \\        const uint need = st[2];
    \\        uint cum = 0u;
    \\        int bin = 255;
    \\        for (; bin > 0; --bin) {
    \\            const uint c = atomic_load_explicit(&hist[bin], memory_order_relaxed);
    \\            if (cum + c >= need) break;
    \\            cum += c;
    \\        }
    \\        st[2] = need - cum;
    \\        st[0] = prefix | (uint(bin) << shift);
    \\        st[1] = pmask | (255u << shift);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\const uint thr = st[0];
    \\const uint need = st[2];
    \\const int seg = (P + 1023) / 1024;
    \\const int s0 = int(tid) * seg;
    \\const int s1 = metal::min(s0 + seg, P);
    \\uint g = 0u, e = 0u;
    \\for (int i = s0; i < s1; ++i) {
    \\    const uint k = msv_score_key(scores[i]);
    \\    g += k > thr ? 1u : 0u;
    \\    e += k == thr ? 1u : 0u;
    \\}
    \\const uint pg = simd_prefix_exclusive_sum(g);
    \\const uint pe = simd_prefix_exclusive_sum(e);
    \\if (lane == 31) { part_g[sg] = pg + g; part_e[sg] = pe + e; }
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg == 0) {
    \\    const uint a = part_g[lane];
    \\    const uint c = part_e[lane];
    \\    part_g[lane] = simd_prefix_exclusive_sum(a);
    \\    part_e[lane] = simd_prefix_exclusive_sum(c);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\uint er = part_e[sg] + pe;
    \\uint pos = part_g[sg] + pg + metal::min(er, need);
    \\for (int i = s0; i < s1; ++i) {
    \\    const uint k = msv_score_key(scores[i]);
    \\    bool take = k > thr;
    \\    if (k == thr) { take = er < need; ++er; }
    \\    if (take) {
    \\        const bool valid = (i + 1) * KPOOL - 1 <= q_pos;
    \\        for (int c = 0; c < KPOOL; ++c) out[int(pos) * KPOOL + c] = valid ? i * KPOOL + c : -1;
    \\        ++pos;
    \\    }
    \\}
    \\for (int col = SEL_K * KPOOL + int(tid); col < OUT_W; col += 1024) {
    \\    int v = -1;
    \\    if (col >= TOPK_W && col < TOPK_W + TAIL_W) {
    \\        const int t = col - TOPK_W;
    \\        const int tail_count = (q_pos + 1) % KPOOL;
    \\        if (t < tail_count) v = q_pos + 1 - tail_count + t;
    \\    }
    \\    out[col] = v;
    \\}
;
const TOPK_EXPAND_HEADER =
    \\inline uint msv_score_key(float x) {
    \\    const uint u = as_type<uint>(x);
    \\    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    \\}
;
var topk_expand_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn decodeKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, src: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

fn applyOne(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array) !mlx.mlx_array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Whether one query's selection runs on `decodeSelect`.
fn decodeSelectServes(s: mlx.mlx_stream, q_idx: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer) bool {
    return mlx.getShape(q_idx)[1] == 1 and decodeRowsServe(s, q_idx, idx_cache, ix);
}

/// `decodeSelect`'s shape conditions short of the row count.
fn decodeRowsServe(s: mlx.mlx_stream, q_idx: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer) bool {
    const qs = mlx.getShape(q_idx);
    const dt = mlx.mlx_array_dtype(idx_cache);
    return mlx.streamIsGpu(s) and qs[0] == 1 and ix.heads == 32 and ix.head_dim == 128 and
        (dt == .bfloat16 or dt == .float16) and mlx.mlx_array_dtype(q_idx) == dt;
}

/// f32 [p] scores of one query at `q_pos` (`ws` carries the Dh^-0.5).
fn decodeScores(s: mlx.mlx_stream, q_idx: mlx.mlx_array, ws: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer, q_pos: c_int, p: c_int) !mlx.mlx_array {
    const dims = mlx.mlx_array_new_data(&[_]i32{ q_pos, p }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    const sk = try decodeKernel(&dec_score_kernel, "glm5_dsa_decode_scores", &.{ "cache", "q", "w", "dims" }, DEC_SCORE_SOURCE, "");
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{p}, 1, .float32));
    const per_tg = DEC_SCORE_SGS * DEC_SCORE_PPS;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, DEC_SCORE_SGS * 32, @divTrunc(p + per_tg - 1, per_tg), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, DEC_SCORE_SGS * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx.mlx_array_dtype(idx_cache)));
    inline for (.{ .{ "DH", ix.head_dim }, .{ "KPOOL", ix.kpool }, .{ "SGS", DEC_SCORE_SGS }, .{ "PPS", DEC_SCORE_PPS } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    return applyOne(s, sk, cfg, &.{ idx_cache, q_idx, ws, dims });
}

/// `selectRows` for one query at `q_pos` with `select_k` < `p` pools: int32 [1, 1, topk + kpool - 1].
fn decodeSelect(s: mlx.mlx_stream, q_idx: mlx.mlx_array, ws: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer, q_pos: c_int, p: c_int, select_k: c_int) !mlx.mlx_array {
    const scores = try decodeScores(s, q_idx, ws, idx_cache, ix, q_pos, p);
    defer _ = mlx.mlx_array_free(scores);
    return topkExpand(s, scores, ix, q_pos, p, select_k);
}

/// The chronological top `select_k` of `scores` [p] expanded to token indices.
fn topkExpand(s: mlx.mlx_stream, scores: mlx.mlx_array, ix: Indexer, q_pos: c_int, p: c_int, select_k: c_int) !mlx.mlx_array {
    const dims = mlx.mlx_array_new_data(&[_]i32{ q_pos, p }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    const tk = try decodeKernel(&topk_expand_kernel, "glm5_dsa_topk_expand", &.{ "scores", "dims" }, TOPK_EXPAND_SOURCE, TOPK_EXPAND_HEADER);
    const out_w = ix.topk + ix.kpool - 1;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1, out_w }, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    inline for (.{ .{ "SEL_K", select_k }, .{ "KPOOL", ix.kpool }, .{ "TOPK_W", ix.topk }, .{ "TAIL_W", ix.kpool - 1 }, .{ "OUT_W", out_w } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    return applyOne(s, tk, cfg, &.{ scores, dims });
}

/// `selectTokens` for queries at positions `q_start .. q_start + S`.
fn selectRows(s: mlx.mlx_stream, q_idx: mlx.mlx_array, w: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer, q_start: c_int) !mlx.mlx_array {
    const csh = mlx.getShape(idx_cache);
    const b = csh[0];
    const t = csh[1];
    const sq = mlx.getShape(q_idx)[1];
    const p = @divTrunc(t, ix.kpool);
    const select_k = @min(@divTrunc(ix.topk, ix.kpool), p);

    var sel = mlx.mlx_array_new(); // int32 [B, S, select_k] pool ids, chronological
    defer _ = mlx.mlx_array_free(sel);
    if (select_k == p) {
        // Every complete pool (none yet: pool 0, which the expansion finds unusable).
        const pi = try iota(s, @max(p, 1), 0);
        defer _ = mlx.mlx_array_free(pi);
        try mlx.check(mlx.mlx_broadcast_to(&sel, pi, &[_]c_int{ b, sq, @max(p, 1) }, 3, s));
    } else if (decodeSelectServes(s, q_idx, idx_cache, ix)) {
        const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
        defer _ = mlx.mlx_array_free(scale);
        var ws = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ws);
        try mlx.check(mlx.mlx_multiply(&ws, w, scale, s));
        return decodeSelect(s, q_idx, ws, idx_cache, ix, q_start, p, select_k);
    } else if (sq <= HC_PRE_MAX_ROWS and decodeRowsServe(s, q_idx, idx_cache, ix)) {
        // A verify window: the one-query select per row (its own position), joined.
        const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
        defer _ = mlx.mlx_array_free(scale);
        var ws = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ws);
        try mlx.check(mlx.mlx_multiply(&ws, w, scale, s));
        const qsh = mlx.getShape(q_idx);
        var rows_out: [HC_PRE_MAX_ROWS]mlx.mlx_array = undefined;
        var n: usize = 0;
        defer for (rows_out[0..n]) |a| {
            _ = mlx.mlx_array_free(a);
        };
        while (n < @as(usize, @intCast(sq))) : (n += 1) {
            const i: c_int = @intCast(n);
            var qi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(qi);
            try mlx.check(mlx.mlx_slice(&qi, q_idx, &[_]c_int{ 0, i, 0, 0 }, 4, &[_]c_int{ 1, i + 1, qsh[2], qsh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
            var wi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wi);
            try mlx.check(mlx.mlx_slice(&wi, ws, &[_]c_int{ 0, i, 0 }, 3, &[_]c_int{ 1, i + 1, qsh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            rows_out[n] = try decodeSelect(s, qi, wi, idx_cache, ix, q_start + i, p, select_k);
        }
        const vec = mlx.mlx_vector_array_new_data(&rows_out, n);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
        return out;
    } else {
        // Scores: sum_h w_h * relu(q_h . pool_key) * Dh^-0.5, f32; pools a query cannot see lose.
        const pk = try storedPoolKeys(s, idx_cache, p, ix);
        defer _ = mlx.mlx_array_free(pk);
        const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
        defer _ = mlx.mlx_array_free(scale);
        var ws = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ws);
        try mlx.check(mlx.mlx_multiply(&ws, w, scale, s));
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        // One row pads a whole NAX tile: the composed scores are cheaper there.
        if (if (sq > 1) try indexerScoresNax(s, q_idx, ws, pk, q_start, ix.kpool) else null) |nax| {
            _ = mlx.mlx_array_free(masked);
            masked = nax;
        } else try maskedScores(s, &masked, q_idx, ws, pk, q_start, p, ix);
        var neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg);
        try mlx.check(mlx.mlx_negative(&neg, masked, s));
        var part = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(part);
        try mlx.check(mlx.mlx_argpartition_axis(&part, neg, select_k - 1, -1, s));
        var top = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(top);
        try mlx.check(mlx.mlx_slice(&top, part, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ b, sq, select_k }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var top32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(top32);
        try mlx.check(mlx.mlx_astype(&top32, top, .int32, s));
        try mlx.check(mlx.mlx_sort_axis(&sel, top32, -1, s));
    }
    return expandSelection(s, sel, q_start, p, ix);
}

/// The composed scores off NAX: [B, S, P] f32, -max for a pool past the query.
fn maskedScores(s: mlx.mlx_stream, out: *mlx.mlx_array, q_idx: mlx.mlx_array, ws: mlx.mlx_array, pk: mlx.mlx_array, q_start: c_int, p: c_int, ix: Indexer) !void {
    const sq = mlx.getShape(q_idx)[1];
    var pkf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pkf);
    try mlx.check(mlx.mlx_astype(&pkf, pk, .float32, s));
    var pk_t = mlx.mlx_array_new(); // [B, 1, Dh, P]
    defer _ = mlx.mlx_array_free(pk_t);
    {
        var e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e);
        try mlx.check(mlx.mlx_expand_dims(&e, pkf, 1, s));
        try mlx.check(mlx.mlx_swapaxes(&pk_t, e, -1, -2, s));
    }
    var qf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qf);
    try mlx.check(mlx.mlx_astype(&qf, q_idx, .float32, s));
    var sc = mlx.mlx_array_new(); // [B, S, H, P]
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_matmul(&sc, qf, pk_t, s));
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var rl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rl);
    try mlx.check(mlx.mlx_maximum(&rl, sc, zero, s));
    var ws4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ws4);
    try mlx.check(mlx.mlx_expand_dims(&ws4, ws, 3, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, rl, ws4, s));
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_sum_axis(&scores, weighted, 2, false, s));
    // Pool p is visible once its last token (p*kpool + kpool - 1) is at or before the query.
    const pi = try iota(s, p, 0);
    defer _ = mlx.mlx_array_free(pi);
    const kp = scalarI(ix.kpool);
    defer _ = mlx.mlx_array_free(kp);
    var pend = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pend);
    try mlx.check(mlx.mlx_multiply(&pend, pi, kp, s));
    const off = scalarI(ix.kpool - 1 - q_start);
    defer _ = mlx.mlx_array_free(off);
    var pe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pe);
    try mlx.check(mlx.mlx_add(&pe, pend, off, s)); // pool end relative to the first query
    var pe2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pe2);
    try mlx.check(mlx.mlx_reshape(&pe2, pe, &[_]c_int{ 1, 1, p }, 3, s));
    const qi = try iota(s, sq, 0);
    defer _ = mlx.mlx_array_free(qi);
    var qi2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qi2);
    try mlx.check(mlx.mlx_reshape(&qi2, qi, &[_]c_int{ 1, sq, 1 }, 3, s));
    var cand = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand);
    try mlx.check(mlx.mlx_less_equal(&cand, pe2, qi2, s));
    const lowest = mlx.mlx_array_new_float(-std.math.floatMax(f32));
    defer _ = mlx.mlx_array_free(lowest);
    try mlx.check(mlx.mlx_where(out, cand, scores, lowest, s));
}

// ── absorbed latent attention ──────────────────────────────────────────

/// Single-query latent attention whose heads share one key set: q [N, H, 1, R] against keys
/// [N, W, R] (values = keys), `valid` [N, 1, W] bool or null; a null `scale` means q arrives
/// scaled. The heads fold into the GEMM's M: MLX's sdpa declines a GQA factor past 32 and
/// its fallback reads the keys once per head.
/// bool [1, heads * rows, t]: query row i (the last `rows` positions of `t`) sees keys up to its
/// own position, for `latentAttendFolded` over a verify window folded into the head axis.
pub fn latentCausalMask(s: mlx.mlx_stream, heads: c_int, rows: c_int, t: c_int) !mlx.mlx_array {
    const ks = try iota(s, t, 0);
    defer _ = mlx.mlx_array_free(ks);
    const qs = try iota(s, rows, t - rows);
    defer _ = mlx.mlx_array_free(qs);
    var k2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k2);
    try mlx.check(mlx.mlx_reshape(&k2, ks, &[_]c_int{ 1, 1, t }, 3, s));
    var q2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q2);
    try mlx.check(mlx.mlx_reshape(&q2, qs, &[_]c_int{ 1, rows, 1 }, 3, s));
    var valid = mlx.mlx_array_new(); // [1, rows, t]
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_less_equal(&valid, k2, q2, s));
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_broadcast_to(&wide, valid, &[_]c_int{ heads, rows, t }, 3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, wide, &[_]c_int{ 1, heads * rows, t }, 3, s));
    return out;
}

pub fn latentAttendFolded(s: mlx.mlx_stream, q: mlx.mlx_array, keys: mlx.mlx_array, valid: ?mlx.mlx_array, scale: ?f32) !mlx.mlx_array {
    const qsh = mlx.getShape(q);
    const n = qsh[0];
    const h = qsh[1];
    const r = qsh[3];
    const dt = mlx.mlx_array_dtype(q);
    var q3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q3);
    try mlx.check(mlx.mlx_reshape(&q3, q, &[_]c_int{ n, h, r }, 3, s));
    var qs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qs);
    if (scale) |sv| {
        const sc_f = mlx.mlx_array_new_float(sv);
        defer _ = mlx.mlx_array_free(sc_f);
        var sc_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc_t);
        try mlx.check(mlx.mlx_astype(&sc_t, sc_f, dt, s));
        try mlx.check(mlx.mlx_multiply(&qs, q3, sc_t, s));
    } else _ = mlx.mlx_array_set(&qs, q3);
    var kt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kt);
    try mlx.check(mlx.mlx_swapaxes(&kt, keys, 1, 2, s));
    var scores = mlx.mlx_array_new(); // [N, H, W]
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_matmul(&scores, qs, kt, s));
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    if (valid) |v| {
        const ninf_f = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(ninf_f);
        var ninf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ninf);
        try mlx.check(mlx.mlx_astype(&ninf, ninf_f, dt, s));
        try mlx.check(mlx.mlx_where(&masked, v, scores, ninf, s));
    } else _ = mlx.mlx_array_set(&masked, scores);
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_softmax_axis(&p, masked, -1, true, s));
    var o = mlx.mlx_array_new(); // [N, H, R]
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_matmul(&o, p, keys, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, o, &[_]c_int{ n, h, 1, r }, 4, s));
    return out;
}

/// Attention of absorbed queries over the latent rows each query selects.
/// `q_abs` [B, H, S, R], `latent` [B, 1, T, R], `idx` int32 [B, S, W] (-1 = unused).
/// Returns [B, H, S, R]. Rows are processed `chunk` queries at a time so the
/// gathered keys stay bounded.
pub fn sparseLatentAttention(s: mlx.mlx_stream, q_abs: mlx.mlx_array, latent: mlx.mlx_array, idx: mlx.mlx_array, scale: ?f32, chunk: c_int) !mlx.mlx_array {
    const qsh = mlx.getShape(q_abs);
    const b = qsh[0];
    const h = qsh[1];
    const sq = qsh[2];
    const r = qsh[3];
    const t = mlx.getShape(latent)[2];
    const wdt = mlx.getShape(idx)[2];
    var lat2 = mlx.mlx_array_new(); // [B, T, R]
    defer _ = mlx.mlx_array_free(lat2);
    try mlx.check(mlx.mlx_reshape(&lat2, latent, &[_]c_int{ b, t, r }, 3, s));

    var outs: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (outs.items) |a| _ = mlx.mlx_array_free(a);
        outs.deinit(std.heap.c_allocator);
    }
    var at: c_int = 0;
    while (at < sq) : (at += chunk) {
        const n = @min(chunk, sq - at);
        var ic = mlx.mlx_array_new(); // [B, n, W]
        defer _ = mlx.mlx_array_free(ic);
        try mlx.check(mlx.mlx_slice(&ic, idx, &[_]c_int{ 0, at, 0 }, 3, &[_]c_int{ b, at + n, wdt }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        const zero = mlx.mlx_array_new_int(0);
        defer _ = mlx.mlx_array_free(zero);
        var valid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(valid);
        try mlx.check(mlx.mlx_greater_equal(&valid, ic, zero, s));
        // Unused slots are -1, every other index is a cached row.
        var safe = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(safe);
        try mlx.check(mlx.mlx_maximum(&safe, ic, zero, s));
        // Gather [B, n*W, R] rows, then fold each query into the batch.
        var rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows);
        if (b == 1) {
            var flat_i = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat_i);
            try mlx.check(mlx.mlx_reshape(&flat_i, safe, &[_]c_int{n * wdt}, 1, s));
            try mlx.check(mlx.mlx_take_axis(&rows, lat2, flat_i, 1, s));
        } else {
            var flat_i = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat_i);
            try mlx.check(mlx.mlx_reshape(&flat_i, safe, &[_]c_int{ b, n * wdt, 1 }, 3, s));
            try mlx.check(mlx.mlx_take_along_axis(&rows, lat2, flat_i, 1, s));
        }
        var keys = mlx.mlx_array_new(); // [B*n, W, R]
        defer _ = mlx.mlx_array_free(keys);
        try mlx.check(mlx.mlx_reshape(&keys, rows, &[_]c_int{ b * n, wdt, r }, 3, s));
        var qc = mlx.mlx_array_new(); // [B, H, n, R]
        defer _ = mlx.mlx_array_free(qc);
        try mlx.check(mlx.mlx_slice(&qc, q_abs, &[_]c_int{ 0, 0, at, 0 }, 4, &[_]c_int{ b, h, at + n, r }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        var qt = mlx.mlx_array_new(); // [B, n, H, R]
        defer _ = mlx.mlx_array_free(qt);
        try mlx.check(mlx.mlx_transpose_axes(&qt, qc, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        var q4 = mlx.mlx_array_new(); // [B*n, H, 1, R]
        defer _ = mlx.mlx_array_free(q4);
        try mlx.check(mlx.mlx_reshape(&q4, qt, &[_]c_int{ b * n, h, 1, r }, 4, s));
        var mask = mlx.mlx_array_new(); // [B*n, 1, W]
        defer _ = mlx.mlx_array_free(mask);
        try mlx.check(mlx.mlx_reshape(&mask, valid, &[_]c_int{ b * n, 1, wdt }, 3, s));
        const o = try latentAttendFolded(s, q4, keys, mask, scale); // [B*n, H, 1, R]
        defer _ = mlx.mlx_array_free(o);
        var o4 = mlx.mlx_array_new(); // [B, n, H, R]
        defer _ = mlx.mlx_array_free(o4);
        try mlx.check(mlx.mlx_reshape(&o4, o, &[_]c_int{ b, n, h, r }, 4, s));
        var ot = mlx.mlx_array_new(); // [B, H, n, R]
        try mlx.check(mlx.mlx_transpose_axes(&ot, o4, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        try outs.append(std.heap.c_allocator, ot);
    }
    if (outs.items.len == 1) {
        const only = outs.items[0];
        outs.items.len = 0;
        return only;
    }
    const vec = mlx.mlx_vector_array_new_data(outs.items.ptr, outs.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    return out;
}

/// oMLX `patches/glm_moe_dsa/sparse_mla_nax.py` (Apache-2.0, see NOTICE): the sparse
/// latent attention on the tensor units. One threadgroup per (query, 32-head half), 8
/// simdgroups; per 128-slot tile, scores read key rows through the indices (never gathered),
/// online exp2 softmax in f32, P as fp16 hi + lo pieces against bf16 values, f32 accumulate.
const SPARSE_NAX_HEADER =
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace mpp::tensor_ops;
;
const SPARSE_NAX_SOURCE =
    \\    constexpr int D = 512;
    \\    constexpr int BK = 128;
    \\    const int L = params[0];
    \\    const int Kn = params[1];
    \\    const int TOPK = params[2];
    \\    const int q_off = params[3];
    \\    const float scale_log2 = scale[0] * 1.44269504088896341f;
    \\    const int qi = int(threadgroup_position_in_grid.x);
    \\    const int hh = int(threadgroup_position_in_grid.y);
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint lane = thread_index_in_simdgroup;
    \\    const uint tid = sg * 32 + lane;
    \\    const int hg = int(sg) / 4;
    \\    const int j4 = int(sg) % 4;   // QK + softmax: key quarter; PV: dim quarter
    \\    const int q_abs = q_off + qi;
    \\    const int head0 = hh * 32 + hg * 16;
    \\
    \\    const short qid = short(lane >> 2);
    \\    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    \\    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    \\    // Lanes with fn == 0: one per row (fm) of a fragment after the
    \\    // xor-1 / xor-8 row reductions.
    \\    const bool row_writer = (lane & 9u) == 0u;
    \\
    \\    threadgroup int sel[2][BK];
    \\    threadgroup int live[2][BK / 32];
    \\    threadgroup float red_max[2][4][16];
    \\    threadgroup float red_sum[2][4][16];
    \\    // P pieces per (head group, 16-key step, lane): the 8 values of the
    \\    // lane's fragment slots (rows fm, fm + 8 x keys fn .. fn + 3).
    \\    threadgroup half p_hi[2][BK / 16][32 * 8];
    \\    threadgroup half p_lo[2][BK / 16][32 * 8];
    \\
    \\    constexpr auto qk_desc = matmul2d_descriptor(
    \\        16, 32, 16, false, true, true,
    \\        matmul2d_descriptor::mode::multiply_accumulate);
    \\    matmul2d<qk_desc, execution_simdgroup> qk_op;
    \\    constexpr auto pv_desc = matmul2d_descriptor(
    \\        16, 32, 16, false, false, true,
    \\        matmul2d_descriptor::mode::multiply_accumulate);
    \\    matmul2d<pv_desc, execution_simdgroup> pv_op;
    \\
    \\    auto pa = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
    \\    auto pm = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
    \\    auto pb = pv_op.template get_right_input_cooperative_tensor<half, T, float>();
    \\    auto o0 = pv_op.template get_destination_cooperative_tensor<
    \\        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    \\    auto o1 = pv_op.template get_destination_cooperative_tensor<
    \\        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    \\    auto o2 = pv_op.template get_destination_cooperative_tensor<
    \\        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    \\    auto o3 = pv_op.template get_destination_cooperative_tensor<
    \\        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    \\    for (short e = 0; e < 16; ++e) {
    \\        o0[e] = 0.0f;
    \\        o1[e] = 0.0f;
    \\        o2[e] = 0.0f;
    \\        o3[e] = 0.0f;
    \\    }
    \\    float m_run[2] = {-FLT_MAX, -FLT_MAX};
    \\    float l_run[2] = {0.0f, 0.0f};
    \\
    \\    const device T* qr0 = q + (ulong(head0 + fm) * L + qi) * D + fn;
    \\    const device T* qr1 = q + (ulong(head0 + fm + 8) * L + qi) * D + fn;
    \\    const device int32_t* idx_row = idx + ulong(qi) * TOPK;
    \\
    \\    auto stage = [&](int t, int b) {
    \\        if (tid < uint(BK)) {
    \\            const int slot = t * BK + int(tid);
    \\            int kp = slot < TOPK ? int(idx_row[slot]) : -1;
    \\            if (kp < 0 || kp >= Kn || kp > q_abs) {
    \\                kp = -1;
    \\            }
    \\            sel[b][tid] = kp;
    \\            const bool any_live = simd_any(kp >= 0);
    \\            if (lane == 0) {
    \\                live[b][sg] = any_live ? 1 : 0;
    \\            }
    \\        }
    \\    };
    \\
    \\    const int n_tiles = (TOPK + BK - 1) / BK;
    \\    stage(0, 0);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (int t = 0; t < n_tiles; ++t) {
    \\        const int buf = t & 1;
    \\        const int tile_keys = min(BK, TOPK - t * BK);
    \\        // Unused slots sort last in the indexer's top-k rows: skip tiles
    \\        // with no live key (uniform across the threadgroup).
    \\        if ((live[buf][0] | live[buf][1] | live[buf][2] | live[buf][3]) == 0) {
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            if (t + 1 < n_tiles) {
    \\                stage(t + 1, buf ^ 1);
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            continue;
    \\        }
    \\
    \\        // ---- S = Q K^T for key quarter j4 (16 heads x 32 keys).
    \\        auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
    \\        auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
    \\        auto sc = qk_op.template get_destination_cooperative_tensor<
    \\            metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
    \\        for (short e = 0; e < 16; ++e) {
    \\            sc[e] = 0.0f;
    \\        }
    \\        if (j4 * 32 < tile_keys) {
    \\            const device T* kr[2][2];
    \\            for (short tn = 0; tn < 2; ++tn) {
    \\                for (short i = 0; i < 2; ++i) {
    \\                    const int kp = sel[buf][j4 * 32 + tn * 16 + fm + i * 8];
    \\                    kr[tn][i] = kv + ulong(max(kp, 0)) * D + fn;
    \\                }
    \\            }
    \\            _Pragma("clang loop unroll_count(4)")
    \\            for (short kk = 0; kk < D; kk += 16) {
    \\                for (short j = 0; j < 4; ++j) {
    \\                    qa[j] = qr0[kk + j];
    \\                    qa[4 + j] = qr1[kk + j];
    \\                }
    \\                for (short tn = 0; tn < 2; ++tn) {
    \\                    for (short i = 0; i < 2; ++i) {
    \\                        for (short j = 0; j < 4; ++j) {
    \\                            kb[tn * 8 + i * 4 + j] = kr[tn][i][kk + j];
    \\                        }
    \\                    }
    \\                }
    \\                qk_op.run(qa, kb, sc);
    \\            }
    \\        }
    \\        bool ok[2][4];
    \\        for (short tn = 0; tn < 2; ++tn) {
    \\            for (short j = 0; j < 4; ++j) {
    \\                ok[tn][j] = sel[buf][j4 * 32 + tn * 16 + fn + j] >= 0;
    \\            }
    \\        }
    \\        // Partial row max of the raw scores over the quarter (scale > 0 and
    \\        // rounding is monotonic: max(s) * c == max(s * c) bitwise).
    \\        float pmx[2];
    \\        for (short i = 0; i < 2; ++i) {
    \\            float m = -FLT_MAX;
    \\            for (short tn = 0; tn < 2; ++tn) {
    \\                for (short j = 0; j < 4; ++j) {
    \\                    m = ok[tn][j] ? max(m, float(sc[tn * 8 + i * 4 + j])) : m;
    \\                }
    \\            }
    \\            m = max(m, simd_shuffle_xor(m, ushort(1)));
    \\            m = max(m, simd_shuffle_xor(m, ushort(8)));
    \\            pmx[i] = m;
    \\        }
    \\        if (row_writer) {
    \\            red_max[hg][j4][fm] = pmx[0];
    \\            red_max[hg][j4][fm + 8] = pmx[1];
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        if (t + 1 < n_tiles) {
    \\            stage(t + 1, buf ^ 1);
    \\        }
    \\
    \\        // ---- Tile row max (the same value in the 4 simdgroups of the group).
    \\        float factor[2];
    \\        float mnew[2];
    \\        for (short i = 0; i < 2; ++i) {
    \\            const int r = fm + i * 8;
    \\            const float m = max(max(red_max[hg][0][r], red_max[hg][1][r]),
    \\                                max(red_max[hg][2][r], red_max[hg][3][r]));
    \\            // no usable key in the tile: keep the running max
    \\            const float cand = m == -FLT_MAX ? -FLT_MAX : m * scale_log2;
    \\            mnew[i] = max(m_run[i], cand);
    \\            factor[i] = fast::exp2(m_run[i] - mnew[i]);
    \\            m_run[i] = mnew[i];
    \\        }
    \\        // ---- P of this quarter, once: exp2, fp16 hi + lo pieces, row sums.
    \\        float rs[2] = {0.0f, 0.0f};
    \\        for (short tn = 0; tn < 2; ++tn) {
    \\            vec<half, 8> h, lo;
    \\            for (short i = 0; i < 2; ++i) {
    \\                for (short j = 0; j < 4; ++j) {
    \\                    const float e = ok[tn][j]
    \\                        ? fast::exp2(float(sc[tn * 8 + i * 4 + j]) * scale_log2 - mnew[i])
    \\                        : 0.0f;
    \\                    const half hi = half(e);
    \\                    h[i * 4 + j] = hi;
    \\                    lo[i * 4 + j] = half(e - float(hi));
    \\                    rs[i] += e;
    \\                }
    \\            }
    \\            const int ks = j4 * 2 + tn;
    \\            *(threadgroup vec<half, 8>*)(&p_hi[hg][ks][lane * 8]) = h;
    \\            *(threadgroup vec<half, 8>*)(&p_lo[hg][ks][lane * 8]) = lo;
    \\        }
    \\        for (short i = 0; i < 2; ++i) {
    \\            rs[i] += simd_shuffle_xor(rs[i], ushort(1));
    \\            rs[i] += simd_shuffle_xor(rs[i], ushort(8));
    \\        }
    \\        if (row_writer) {
    \\            red_sum[hg][j4][fm] = rs[0];
    \\            red_sum[hg][j4][fm + 8] = rs[1];
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        // ---- O = O * factor + P V for dim quarter j4 over the whole tile.
    \\        // (factor == 1 exactly for every row of the simdgroup: O * 1 == O.)
    \\        if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) {
    \\            for (short e = 0; e < 16; ++e) {
    \\                const float f = factor[(e >> 2) & 1];
    \\                o0[e] *= f;
    \\                o1[e] *= f;
    \\                o2[e] *= f;
    \\                o3[e] *= f;
    \\            }
    \\        }
    \\        for (short i = 0; i < 2; ++i) {
    \\            const int r = fm + i * 8;
    \\            const float tsum = ((red_sum[hg][0][r] + red_sum[hg][1][r]) + red_sum[hg][2][r]) + red_sum[hg][3][r];
    \\            l_run[i] = l_run[i] * factor[i] + tsum;
    \\        }
    \\        const int n_ks = (tile_keys + 15) / 16;
    \\        for (short ks = 0; ks < n_ks; ++ks) {
    \\            const vec<half, 8> h = *(const threadgroup vec<half, 8>*)(&p_hi[hg][ks][lane * 8]);
    \\            const vec<half, 8> lo = *(const threadgroup vec<half, 8>*)(&p_lo[hg][ks][lane * 8]);
    \\            for (short e = 0; e < 8; ++e) {
    \\                pa[e] = h[e];
    \\                pm[e] = lo[e];
    \\            }
    \\            const int kp0 = sel[buf][ks * 16 + fm];
    \\            const int kp1 = sel[buf][ks * 16 + fm + 8];
    \\            const device T* v0 = kv + ulong(max(kp0, 0)) * D + j4 * 128 + fn;
    \\            const device T* v1 = kv + ulong(max(kp1, 0)) * D + j4 * 128 + fn;
    \\            for (short np = 0; np < 4; ++np) {
    \\                for (short tn = 0; tn < 2; ++tn) {
    \\                    for (short j = 0; j < 4; ++j) {
    \\                        pb[tn * 8 + j] = v0[np * 32 + tn * 16 + j];
    \\                        pb[tn * 8 + 4 + j] = v1[np * 32 + tn * 16 + j];
    \\                    }
    \\                }
    \\                if (np == 0) {
    \\                    pv_op.run(pa, pb, o0);
    \\                    pv_op.run(pm, pb, o0);
    \\                } else if (np == 1) {
    \\                    pv_op.run(pa, pb, o1);
    \\                    pv_op.run(pm, pb, o1);
    \\                } else if (np == 2) {
    \\                    pv_op.run(pa, pb, o2);
    \\                    pv_op.run(pm, pb, o2);
    \\                } else {
    \\                    pv_op.run(pa, pb, o3);
    \\                    pv_op.run(pm, pb, o3);
    \\                }
    \\            }
    \\        }
    \\    }
    \\
    \\    for (short i = 0; i < 2; ++i) {
    \\        device T* orow = out + (ulong(head0 + fm + i * 8) * L + qi) * D + j4 * 128 + fn;
    \\        const float denom = l_run[i] > 0.0f ? l_run[i] : 1.0f;
    \\        for (short tn = 0; tn < 2; ++tn) {
    \\            for (short j = 0; j < 4; ++j) {
    \\                const short e = tn * 8 + i * 4 + j;
    \\                orow[tn * 16 + j] = T(o0[e] / denom);
    \\                orow[32 + tn * 16 + j] = T(o1[e] / denom);
    \\                orow[64 + tn * 16 + j] = T(o2[e] / denom);
    \\                orow[96 + tn * 16 + j] = T(o3[e] / denom);
    \\            }
    \\        }
    \\    }
;

var sparse_nax_kernel: ?mlx.mlx_fast_metal_kernel = null;
var sparse_nax_engaged = false;

/// `sparseLatentAttention` on the tensor units, for queries at the LAST rows of `latent`:
/// `q_abs` [1, H, L, 512], `latent` [1, 1, K, 512], `idx` int32 [1, L, W]. An unused slot
/// is -1 and unused pools sort last, so whole tiles skip. Null off NAX and outside the shape.
pub fn sparseLatentNax(s: mlx.mlx_stream, q_abs: mlx.mlx_array, latent: mlx.mlx_array, idx: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !@import("transformer.zig").verifyQmmNaxAvailable()) return null;
    const qs = mlx.getShape(q_abs);
    const ls = mlx.getShape(latent);
    const is = mlx.getShape(idx);
    if (qs.len != 4 or ls.len != 4 or is.len != 3) return null;
    if (qs[0] != 1 or ls[0] != 1 or ls[1] != 1 or is[0] != 1) return null;
    if (qs[3] != 512 or ls[3] != 512 or @rem(qs[1], 32) != 0) return null;
    const dt = mlx.mlx_array_dtype(q_abs);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(latent) != dt or mlx.mlx_array_dtype(idx) != .int32) return null;
    const h = qs[1];
    const l = qs[2];
    const k = ls[2];
    const w = is[2];
    if (l < 1 or k < l or is[1] != l or w < 1) return null;
    const kern = sparse_nax_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "q", "kv", "idx", "params", "scale" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kn = mlx.mlx_fast_metal_kernel_new("msv_glm5_sparse_mla_nax", iv, ov, SPARSE_NAX_SOURCE, SPARSE_NAX_HEADER, true, false);
        if (kn.ctx == null) return error.MetalKernelCompileFailed;
        sparse_nax_kernel = kn;
        break :blk kn;
    };
    var q3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q3);
    try mlx.check(mlx.mlx_reshape(&q3, q_abs, &[_]c_int{ h, l, 512 }, 3, s));
    var kv2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv2);
    try mlx.check(mlx.mlx_reshape(&kv2, latent, &[_]c_int{ k, 512 }, 2, s));
    var idx2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx2);
    try mlx.check(mlx.mlx_reshape(&idx2, idx, &[_]c_int{ l, w }, 2, s));
    const pdata = [4]i32{ l, k, w, k - l };
    const params = mlx.mlx_array_new_data(&pdata, &[_]c_int{4}, 1, .int32);
    defer _ = mlx.mlx_array_free(params);
    const sv = [1]f32{scale};
    const scale_a = mlx.mlx_array_new_data(&sv, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scale_a);

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ h, l, 512 }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, l * 256, @divExact(h, 32), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    const inputs = [_]mlx.mlx_array{ q3, kv2, idx2, params, scale_a };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var o3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o3);
    try mlx.check(mlx.mlx_vector_array_get(&o3, ov, 0));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, o3, &[_]c_int{ 1, h, l, 512 }, 4, s));
    if (!sparse_nax_engaged) {
        sparse_nax_engaged = true;
        @import("log.zig").info("[dsa] NAX sparse latent attention engaged\n", .{});
    }
    return out;
}

// ── MoE decode ─────────────────────────────────────────────────────────

/// An affine-quantized weight and its geometry.
pub const QBank = struct { w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array, bits: u32, gs: u32 };

const MOE_RPS: c_int = 4;
const MOE_NSG: c_int = 2;
var moe_gate_up_kernel: ?mlx.mlx_fast_metal_kernel = null;
var moe_down_kernel: ?mlx.mlx_fast_metal_kernel = null;
var moe_decode_engaged = false;

/// Shapes MLX's qmv_fast serves (the kernels replay its lane mapping).
fn qmvFastOk(bank: QBank, n: c_int, k: c_int) bool {
    if (bank.s.ctx == null or bank.b.ctx == null) return false;
    if (bank.bits != 4 and bank.bits != 5 and bank.bits != 6 and bank.bits != 8) return false;
    if (bank.gs != 32 and bank.gs != 64 and bank.gs != 128) return false;
    const pack: u32 = if (bank.bits == 5) 8 else if (bank.bits == 6) 4 else 32 / bank.bits;
    const vpt: c_int = @intCast(pack * 2);
    return @rem(@as(c_int, @intCast(bank.gs)), vpt) == 0 and @rem(n, 8) == 0 and @rem(k, vpt * 32) == 0 and @rem(n, MOE_RPS * MOE_NSG) == 0;
}

/// oMLX `decode_kernels.py` `_MH_QMV_SOURCE` (Apache-2.0, see NOTICE): the MLA per-head
/// projections for one token, `x[h] @ w[h]^T`, as qmv_fast rows (4 per simdgroup, NSG
/// simdgroups per threadgroup); MLX's batched qmv runs one 64-thread threadgroup per 8 rows.
const HEAD_QMV_SOURCE =
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const int sg = int(simdgroup_index_in_threadgroup);
    \\  const int h = int(threadgroup_position_in_grid.z);
    \\  const int r0 = (int(threadgroup_position_in_grid.y) * NSG + sg) * 4;
    \\  constexpr int WB = K * glm_bytes_per_pack<BITS>() / glm_pack_factor<BITS>();
    \\  constexpr int G = K / GS;
    \\  const device uint8_t* wh = (const device uint8_t*)w + (size_t(h) * N + r0) * WB;
    \\  const device T* sh = scales + (size_t(h) * N + r0) * G;
    \\  const device T* bh = biases + (size_t(h) * N + r0) * G;
    \\  const device T* xh = x + size_t(h) * K;
    \\  float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    \\  glm_qmv_rows<T, K, GS, BITS, 4>(wh, sh, bh, xh, lane, result);
    \\  for (int r = 0; r < 4; r++) {
    \\    const float v = simd_sum(result[r]);
    \\    if (lane == 0) out[size_t(h) * N + r0 + r] = static_cast<T>(v * scale[0]);
    \\  }
;
var head_qmv_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// One token's per-head projection: x [1, H, 1, K] by w [H, N, K] (affine, transposed) ->
/// [1, H, 1, N], times `out_scale` before the rounding. Null outside qmv_fast's shapes.
pub fn headQmv(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, scales: mlx.mlx_array, biases: mlx.mlx_array, bits: u32, gs: u32, out_scale: f32) !?mlx.mlx_array {
    const xs = mlx.getShape(x);
    const ss = mlx.getShape(scales);
    if (!mlx.streamIsGpu(s) or xs.len != 4 or xs[0] != 1 or xs[2] != 1 or ss.len != 3 or ss[0] != xs[1] or biases.ctx == null) return null;
    const h = xs[1];
    const k = xs[3];
    const n = ss[1];
    const g: c_int = @intCast(gs);
    const pack: c_int = switch (bits) {
        4, 8 => @divExact(32, @as(c_int, @intCast(bits))),
        5 => 8,
        6 => 4,
        else => return null,
    };
    const dt = mlx.mlx_array_dtype(x);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(scales) != dt or mlx.mlx_array_dtype(biases) != dt) return null;
    if (g != 32 and g != 64 and g != 128) return null;
    if (@rem(k, g) != 0 or ss[2] != @divTrunc(k, g) or @rem(n, 8) != 0 or @rem(k, pack * 2 * 32) != 0) return null;
    const nsg: c_int = if (@rem(n, 32) == 0) 8 else 2;
    const kern = head_qmv_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "x", "w", "scales", "biases", "scale" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kn = mlx.mlx_fast_metal_kernel_new("msv_glm5_head_qmv", iv, ov, HEAD_QMV_SOURCE, @embedFile("kernels/glm5_qmv_header.metal"), true, false);
        if (kn.ctx == null) return error.MetalKernelCompileFailed;
        head_qmv_kernel = kn;
        break :blk kn;
    };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, h, 1, n }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, @divExact(n, 4 * nsg) * nsg, h));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, nsg, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    inline for (.{ .{ "K", k }, .{ "N", n }, .{ "BITS", @as(c_int, @intCast(bits)) }, .{ "GS", g }, .{ "NSG", nsg } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    const sc = mlx.mlx_array_new_data(&[_]f32{out_scale}, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(sc);
    return try applyOne(s, kern, cfg, &.{ x, w, scales, biases, sc });
}

fn moeKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, @embedFile("kernels/glm5_qmv_header.metal"), true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

const MoeOut = struct { shape: []const c_int, dt: mlx.mlx_dtype };

fn moeLaunch(s: mlx.mlx_stream, k: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, outs: []const MoeOut, results: []mlx.mlx_array, tiles: c_int, slots: c_int, slot_major: bool, ints: []const struct { [*:0]const u8, c_int }) !void {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dt));
    if (slot_major)
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, slots * MOE_NSG, tiles))
    else
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, tiles * MOE_NSG, slots));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SM", @intFromBool(slot_major)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, MOE_NSG, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", outs[0].dt));
    for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    for (results, 0..) |*r, idx| {
        r.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(r, ov, idx));
    }
}

/// One token through the routed experts and the ungated shared expert in two dispatches
/// (oMLX's decode kernels): `silu(min(g, limit)) * clip(u, ±limit)` for every route and the
/// shared slot, then each output row's routing-weighted sum of the down projections plus the
/// shared one. `x` is one [.., hidden] row, `inds` uint32 [topk], `scores` [topk]; routed banks
/// are [E, out, in]. Returns [hidden] in x's dtype, or null outside the shapes served.
/// Rows one decode launch takes: a token or a verify window (the kernels index by token).
pub const MOE_DECODE_MAX_ROWS: c_int = 16;

pub fn moeDecode(s: mlx.mlx_stream, x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, gate: QBank, up: QBank, down: QBank, sh_gate: QBank, sh_up: QBank, sh_down: QBank, limit: f32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32 or mlx.mlx_array_ndim(inds) != 1) return null;
    const gsh = mlx.getShape(gate.w);
    const dsh = mlx.getShape(down.w);
    if (gsh.len != 3 or dsh.len != 3) return null;
    const inter = gsh[1];
    const hidden: c_int = dsh[1];
    if (@rem(@as(c_int, @intCast(mlx.mlx_array_size(x))), hidden) != 0) return null;
    const rows: c_int = @intCast(@divExact(@as(c_int, @intCast(mlx.mlx_array_size(x))), hidden));
    if (rows < 1 or rows > MOE_DECODE_MAX_ROWS or @rem(mlx.getShape(inds)[0], rows) != 0) return null;
    const topk = @divExact(mlx.getShape(inds)[0], rows);
    if (mlx.mlx_array_size(scores) != @as(usize, @intCast(rows * topk))) return null;
    if (gate.bits != up.bits or gate.gs != up.gs or !std.mem.eql(c_int, gsh, mlx.getShape(up.w))) return null;
    if (sh_gate.bits != sh_up.bits or sh_gate.gs != sh_up.gs or sh_gate.bits != sh_down.bits or sh_gate.gs != sh_down.gs) return null;
    if (mlx.getShape(sh_gate.w)[0] != inter or dsh[1] != hidden or mlx.getShape(sh_down.w)[0] != hidden) return null;
    if (!qmvFastOk(gate, inter, hidden) or !qmvFastOk(sh_gate, inter, hidden) or !qmvFastOk(down, hidden, inter) or !qmvFastOk(sh_down, hidden, inter)) return null;
    inline for (.{ gate, up, down, sh_gate, sh_up, sh_down }) |q| if (mlx.mlx_array_dtype(q.s) != dt) return null;

    const lv = [1]f32{limit};
    const lim = mlx.mlx_array_new_data(&lv, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(lim);
    // Several rows: run the routed slots expert-sorted and slot-major, so a shared expert's
    // rows are read from cache; one row keeps the tile-major order (one slot per expert).
    const many = rows > 1;
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    if (many) {
        var sorted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted);
        try mlx.check(mlx.mlx_argsort_axis(&sorted, inds, 0, s));
        try mlx.check(mlx.mlx_astype(&order, sorted, .uint32, s));
    } else _ = mlx.mlx_array_set(&order, inds);
    const gu = try moeKernel(&moe_gate_up_kernel, "msv_glm5_moe_gate_up", &.{ "x", "indices", "order", "limit", "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b", "sh_gate_w", "sh_gate_s", "sh_gate_b", "sh_up_w", "sh_up_s", "sh_up_b" }, &.{"out"}, @embedFile("kernels/glm5_moe_gate_up.metal"));
    var act = [1]mlx.mlx_array{undefined};
    try moeLaunch(s, gu, &.{ x, inds, order, lim, gate.w, gate.s, gate.b, up.w, up.s, up.b, sh_gate.w, sh_gate.s, sh_gate.b, sh_up.w, sh_up.s, sh_up.b }, &.{.{ .shape = &.{ rows, topk + 1, inter }, .dt = dt }}, &act, @divExact(inter, MOE_RPS * MOE_NSG), rows * (topk + 1), many, &.{
        .{ "K", hidden },                 .{ "N", inter },     .{ "TOPK", topk },     .{ "NTOK", rows },                 .{ "RBITS", @intCast(gate.bits) }, .{ "RGS", @intCast(gate.gs) },
        .{ "RPS", MOE_RPS },              .{ "NSG", MOE_NSG }, .{ "ESTRIDE", inter }, .{ "UP_OFF", 0 },                  .{ "SBITS", @intCast(sh_gate.bits) },
        .{ "SGS", @intCast(sh_gate.gs) },
    });
    defer _ = mlx.mlx_array_free(act[0]);
    const dn = try moeKernel(&moe_down_kernel, "msv_glm5_moe_down", &.{ "act", "indices", "scores", "down_w", "down_s", "down_b", "sh_down_w", "sh_down_s", "sh_down_b" }, &.{"out"}, @embedFile("kernels/glm5_moe_down.metal"));
    var out = [1]mlx.mlx_array{undefined};
    try moeLaunch(s, dn, &.{ act[0], inds, scores, down.w, down.s, down.b, sh_down.w, sh_down.s, sh_down.b }, &.{.{ .shape = &.{ rows, hidden }, .dt = dt }}, &out, @divExact(hidden, MOE_RPS * MOE_NSG), rows, many, &.{
        .{ "K", inter },     .{ "N", hidden },    .{ "TOPK", topk },                    .{ "RBITS", @intCast(down.bits) }, .{ "RGS", @intCast(down.gs) },
        .{ "RPS", MOE_RPS }, .{ "NSG", MOE_NSG }, .{ "SBITS", @intCast(sh_down.bits) }, .{ "SGS", @intCast(sh_down.gs) },
    });
    if (!moe_decode_engaged) {
        moe_decode_engaged = true;
        @import("log.zig").info("[moe] glm5 decode kernels engaged: topk={d} inter={d} bits={d}/{d}\n", .{ topk, inter, gate.bits, down.bits });
    }
    return out[0];
}

// ── tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

/// Reference pooling for the kernels' tests: softmax over each pool's slots of (gate + ape),
/// weighting its keys. `keys`/`gates` [B, P*kpool, Dh] in the cache dtype, `ape` f32
/// [kpool, Dh]. Returns [B, P, Dh] in the keys' dtype.
fn poolKeys(s: mlx.mlx_stream, keys: mlx.mlx_array, gates: mlx.mlx_array, ape: mlx.mlx_array, p: c_int, ix: Indexer) !mlx.mlx_array {
    const sh = mlx.getShape(keys);
    const shape4 = [_]c_int{ sh[0], p, ix.kpool, ix.head_dim };
    var k4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k4);
    try mlx.check(mlx.mlx_reshape(&k4, keys, &shape4, 4, s));
    var g4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g4);
    try mlx.check(mlx.mlx_reshape(&g4, gates, &shape4, 4, s));
    var gf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gf);
    try mlx.check(mlx.mlx_astype(&gf, g4, .float32, s));
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_add(&logits, gf, ape, s));
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, logits, 2, true, s));
    var pd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pd);
    try mlx.check(mlx.mlx_astype(&pd, probs, mlx.mlx_array_dtype(keys), s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_multiply(&w, pd, k4, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, w, 2, false, s));
    return out;
}

fn readF32(a: mlx.mlx_array, s: mlx.mlx_stream, alloc: std.mem.Allocator) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a, .float32, s));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    const out = try alloc.alloc(f32, n);
    @memcpy(out, (mlx.mlx_array_data_float32(c) orelse return error.NoData)[0..n]);
    return out;
}

test "glm5 sparse latent attention equals masked softmax over the selected rows" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    const b: c_int = 1;
    const h: c_int = 4;
    const sq: c_int = 5;
    const r: c_int = 16;
    const t: c_int = 12;
    const w: c_int = 4;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    var qv: [1 * 4 * 5 * 16]f32 = undefined;
    for (&qv) |*x| x.* = rand.floatNorm(f32);
    var lv: [1 * 12 * 16]f32 = undefined;
    for (&lv) |*x| x.* = rand.floatNorm(f32);
    // Each query picks w rows (some unused); duplicates are not produced by the indexer.
    const idx_v = [_]i32{ 0, 3, -1, 7, 2, 5, 11, -1, -1, -1, 1, 4, 6, 8, 9, 10, 0, 1, 2, 3 };
    const q = mlx.mlx_array_new_data(&qv, &[_]c_int{ b, h, sq, r }, 4, .float32);
    defer _ = mlx.mlx_array_free(q);
    const lat = mlx.mlx_array_new_data(&lv, &[_]c_int{ b, 1, t, r }, 4, .float32);
    defer _ = mlx.mlx_array_free(lat);
    const idx = mlx.mlx_array_new_data(&idx_v, &[_]c_int{ b, sq, w }, 3, .int32);
    defer _ = mlx.mlx_array_free(idx);
    const scale: f32 = 0.25;
    for ([_]c_int{ 64, 2 }) |chunk| {
        const out = try sparseLatentAttention(s, q, lat, idx, scale, chunk);
        defer _ = mlx.mlx_array_free(out);
        const got = try readF32(out, s, alloc);
        defer alloc.free(got);
        for (0..@intCast(h)) |hh| for (0..@intCast(sq)) |qq| {
            var scores: [4]f32 = undefined;
            var mx_s: f32 = -std.math.inf(f32);
            for (0..@intCast(w)) |j| {
                const k = idx_v[qq * 4 + j];
                if (k < 0) {
                    scores[j] = -std.math.inf(f32);
                    continue;
                }
                var dot: f32 = 0;
                for (0..@intCast(r)) |d| dot += qv[((hh * 5) + qq) * 16 + d] * lv[@as(usize, @intCast(k)) * 16 + d];
                scores[j] = dot * scale;
                mx_s = @max(mx_s, scores[j]);
            }
            var sum: f32 = 0;
            for (&scores) |*x| {
                x.* = @exp(x.* - mx_s);
                sum += x.*;
            }
            for (0..@intCast(r)) |d| {
                var acc: f32 = 0;
                for (0..@intCast(w)) |j| {
                    const k = idx_v[qq * 4 + j];
                    if (k >= 0) acc += scores[j] / sum * lv[@as(usize, @intCast(k)) * 16 + d];
                }
                // MLX's f32 matmuls round on the tensor units.
                try testing.expectApproxEqAbs(acc, got[((hh * 5) + qq) * 16 + d], 1e-2);
            }
        };
    }
}

test "glm5 NAX sparse latent attention matches the gathered sdpa" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    const h: c_int = 64;
    const k: c_int = 300;
    const l: c_int = 40;
    const w: c_int = 260;
    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();
    const qv = try alloc.alloc(f32, @intCast(h * l * 512));
    defer alloc.free(qv);
    for (qv) |*x| x.* = rand.floatNorm(f32);
    const lv = try alloc.alloc(f32, @intCast(k * 512));
    defer alloc.free(lv);
    for (lv) |*x| x.* = rand.floatNorm(f32);
    // Each query reads a random subset of the keys at or before it, holes as -1 (some past it too).
    const iv = try alloc.alloc(i32, @intCast(l * w));
    defer alloc.free(iv);
    for (0..@intCast(l)) |qi| {
        const pos: i32 = k - l + @as(i32, @intCast(qi));
        for (0..@intCast(w)) |j| {
            const r = rand.intRangeAtMost(i32, -40, pos + 3);
            iv[qi * @as(usize, @intCast(w)) + j] = if (r < 0 or r > pos) -1 else r;
        }
    }
    const qf = mlx.mlx_array_new_data(qv.ptr, &[_]c_int{ 1, h, l, 512 }, 4, .float32);
    defer _ = mlx.mlx_array_free(qf);
    const lf = mlx.mlx_array_new_data(lv.ptr, &[_]c_int{ 1, 1, k, 512 }, 4, .float32);
    defer _ = mlx.mlx_array_free(lf);
    const idx = mlx.mlx_array_new_data(iv.ptr, &[_]c_int{ 1, l, w }, 3, .int32);
    defer _ = mlx.mlx_array_free(idx);
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_astype(&q, qf, .bfloat16, s));
    var lat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lat);
    try mlx.check(mlx.mlx_astype(&lat, lf, .bfloat16, s));
    const nax = (try sparseLatentNax(s, q, lat, idx, 0.0625)) orelse return error.SkipZigTest;
    defer _ = mlx.mlx_array_free(nax);
    const ref = try sparseLatentAttention(s, q, lat, idx, 0.0625, 64);
    defer _ = mlx.mlx_array_free(ref);
    const a = try readF32(nax, s, alloc);
    defer alloc.free(a);
    const b = try readF32(ref, s, alloc);
    defer alloc.free(b);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var maxd: f32 = 0;
    for (a, b) |x, y| {
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxd = @max(maxd, @abs(x - y));
    }
    const cos = dot / @sqrt(na * nb);
    std.debug.print("[glm5 nax] cos {d:.6} max|d| {d:.4}\n", .{ cos, maxd });
    try testing.expect(cos > 0.9999 and maxd < 0.05);
}

test "glm5 mHC mixes and expand kernels match the op chain" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const rand = prng.random();
    for ([_]c_int{ 256, 4096 }) |d| {
        const rows: c_int = MIXES_KERNEL_MIN_ROWS + 9;
        const n: usize = @intCast(rows * HC * d);
        const sv = try alloc.alloc(f32, n);
        defer alloc.free(sv);
        for (sv) |*x| x.* = rand.floatNorm(f32);
        const wv = try alloc.alloc(f32, @intCast(HC * d * MIX));
        defer alloc.free(wv);
        for (wv) |*x| x.* = 0.02 * rand.floatNorm(f32);
        const xv = try alloc.alloc(f32, @intCast(rows * d));
        defer alloc.free(xv);
        for (xv) |*x| x.* = rand.floatNorm(f32);
        const pv = try alloc.alloc(f32, @intCast(rows * 4));
        defer alloc.free(pv);
        for (pv) |*x| x.* = 2.0 * rand.float(f32);
        const cv = try alloc.alloc(f32, @intCast(rows * 16));
        defer alloc.free(cv);
        for (cv) |*x| x.* = rand.float(f32);

        const sf = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ 1, rows, HC, d }, 4, .float32);
        defer _ = mlx.mlx_array_free(sf);
        var stream = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stream);
        try mlx.check(mlx.mlx_astype(&stream, sf, .bfloat16, s));
        const fn_rows = mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ MIX, HC * d }, 2, .float32);
        defer _ = mlx.mlx_array_free(fn_rows);

        // mixes vs rms_norm(stream) @ fn^T in f64 (MLX's f32 matmul rounds on the tensor units).
        const got = try mixes(s, stream, fn_rows, 1e-5);
        defer _ = mlx.mlx_array_free(got);
        {
            const a = try readF32(got, s, alloc);
            defer alloc.free(a);
            const ssv = try readF32(stream, s, alloc);
            defer alloc.free(ssv);
            const K: usize = @intCast(HC * d);
            for (0..@intCast(rows)) |r| {
                var sq: f64 = 0;
                for (0..K) |k| sq += @as(f64, ssv[r * K + k]) * ssv[r * K + k];
                const inv = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(K)) + 1e-5);
                for (0..MIX) |col| {
                    var dot: f64 = 0;
                    for (0..K) |k| dot += @as(f64, ssv[r * K + k]) * wv[col * K + k];
                    try testing.expectApproxEqAbs(dot * inv, a[r * MIX + col], 1e-4);
                }
            }
        }

        // expand vs post * x + comb^T @ stream in f32
        const xf = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ 1, rows, d }, 3, .float32);
        defer _ = mlx.mlx_array_free(xf);
        var c: Collapsed = .{ .x = mlx.mlx_array_new(), .post = mlx.mlx_array_new_data(pv.ptr, &[_]c_int{ 1, rows, HC }, 3, .float32), .comb = mlx.mlx_array_new_data(cv.ptr, &[_]c_int{ 1, rows, HC, HC }, 4, .float32) };
        defer c.deinit();
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));
        const out = try expand(s, x, stream, &c);
        defer _ = mlx.mlx_array_free(out);
        const of = try readF32(out, s, alloc);
        defer alloc.free(of);
        const xs = try readF32(x, s, alloc);
        defer alloc.free(xs);
        const ss = try readF32(stream, s, alloc);
        defer alloc.free(ss);
        const D: usize = @intCast(d);
        for (0..@intCast(rows)) |r| for (0..4) |i| for (0..D) |k| {
            var acc: f32 = pv[r * 4 + i] * xs[r * D + k];
            for (0..4) |j| acc += cv[r * 16 + j * 4 + i] * ss[(r * 4 + j) * D + k];
            try testing.expectApproxEqAbs(acc, of[(r * 4 + i) * D + k], 0.02 + 0.004 * @abs(acc));
        };
    }
}

test "glm5 NAX indexer scores equal relu(q . k) * w summed over heads, causally masked" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    const h: c_int = 32;
    const sq: c_int = 70;
    const p: c_int = 45;
    const q_start: c_int = 120; // pools end at 4p+3: rows see 30..47 of them
    var prng = std.Random.DefaultPrng.init(9);
    const rand = prng.random();
    const qv = try alloc.alloc(f32, @intCast(sq * h * 128));
    defer alloc.free(qv);
    for (qv) |*x| x.* = rand.floatNorm(f32);
    const kv = try alloc.alloc(f32, @intCast(p * 128));
    defer alloc.free(kv);
    for (kv) |*x| x.* = rand.floatNorm(f32);
    const wv = try alloc.alloc(f32, @intCast(sq * h));
    defer alloc.free(wv);
    for (wv) |*x| x.* = rand.floatNorm(f32);
    const qf = mlx.mlx_array_new_data(qv.ptr, &[_]c_int{ 1, sq, h, 128 }, 4, .float32);
    defer _ = mlx.mlx_array_free(qf);
    const kf = mlx.mlx_array_new_data(kv.ptr, &[_]c_int{ 1, p, 128 }, 3, .float32);
    defer _ = mlx.mlx_array_free(kf);
    const w = mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ 1, sq, h }, 3, .float32);
    defer _ = mlx.mlx_array_free(w);
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_astype(&q, qf, .bfloat16, s));
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    try mlx.check(mlx.mlx_astype(&k, kf, .bfloat16, s));
    const got = (try indexerScoresNax(s, q, w, k, q_start, 4)) orelse return error.SkipZigTest;
    defer _ = mlx.mlx_array_free(got);
    const g = try readF32(got, s, alloc);
    defer alloc.free(g);
    const qb = try readF32(q, s, alloc);
    defer alloc.free(qb);
    const kb = try readF32(k, s, alloc);
    defer alloc.free(kb);
    for (0..@intCast(sq)) |r| for (0..@intCast(p)) |pi| {
        const visible = @as(c_int, @intCast(pi)) * 4 + 3 <= q_start + @as(c_int, @intCast(r));
        const v = g[r * @as(usize, @intCast(p)) + pi];
        if (!visible) {
            try testing.expect(v < -1e29);
            continue;
        }
        var want: f64 = 0;
        for (0..@intCast(h)) |hh| {
            var dot: f64 = 0;
            for (0..128) |d| dot += @as(f64, qb[(r * @as(usize, @intCast(h)) + hh) * 128 + d]) * kb[pi * 128 + d];
            want += @max(dot, 0) * wv[r * @as(usize, @intCast(h)) + hh];
        }
        try testing.expectApproxEqAbs(want, v, 1e-3 * @max(1.0, @abs(want)));
    };
}

test "glm5 one-token hcPre chain matches collapse, rms_norm and expand" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(21);
    const rand = prng.random();
    const d: c_int = 4096;
    const sv = try alloc.alloc(f32, @intCast(HC * d));
    defer alloc.free(sv);
    for (sv) |*x| x.* = rand.floatNorm(f32);
    const wv = try alloc.alloc(f32, @intCast(HC * d * MIX));
    defer alloc.free(wv);
    for (wv) |*x| x.* = 0.02 * rand.floatNorm(f32);
    var nv: [4096]f32 = undefined;
    for (&nv) |*x| x.* = 1.0 + 0.1 * rand.floatNorm(f32);
    var yv: [4096]f32 = undefined;
    for (&yv) |*x| x.* = rand.floatNorm(f32);
    var bv: [MIX]f32 = undefined;
    for (&bv) |*x| x.* = 0.5 * rand.floatNorm(f32);
    const scv = [3]f32{ 0.8, 1.1, 0.9 };
    const bf = struct {
        fn of(st: mlx.mlx_stream, v: []const f32, shape: []const c_int) !mlx.mlx_array {
            const f = mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f);
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&o, f, .bfloat16, st));
            return o;
        }
    };
    const stream = try bf.of(s, sv, &.{ 1, 1, HC, d });
    defer _ = mlx.mlx_array_free(stream);
    const nw = try bf.of(s, &nv, &.{d});
    defer _ = mlx.mlx_array_free(nw);
    const y = try bf.of(s, &yv, &.{ 1, 1, d });
    defer _ = mlx.mlx_array_free(y);
    const fn_rows = mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ MIX, HC * d }, 2, .float32);
    defer _ = mlx.mlx_array_free(fn_rows);
    const base = mlx.mlx_array_new_data(&bv, &[_]c_int{MIX}, 1, .float32);
    defer _ = mlx.mlx_array_free(base);
    const scale = mlx.mlx_array_new_data(&scv, &[_]c_int{3}, 1, .float32);
    defer _ = mlx.mlx_array_free(scale);
    try testing.expect(hcRowsServes(s, mlx.getShape(stream), .bfloat16, fn_rows, nw));

    const near = struct {
        fn check(st: mlx.mlx_stream, a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, al: std.mem.Allocator) !void {
            const a = try readF32(a_arr, st, al);
            defer al.free(a);
            const b = try readF32(b_arr, st, al);
            defer al.free(b);
            try testing.expectEqual(b.len, a.len);
            for (a, b) |x, w| try testing.expectApproxEqAbs(w, x, 0.02 + 0.01 * @abs(w));
        }
    };
    // Sublayer 1 from a materialized stream.
    var pre0 = try hcPre(s, stream, null, fn_rows, scale, base, nw, 20, 1e-6, 1e-5, 1e-5);
    defer pre0.deinit();
    var c0 = try collapse(s, stream, fn_rows, scale, base, 20, 1e-6, 1e-5);
    defer c0.deinit();
    var n0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n0);
    try mlx.check(mlx.mlx_fast_rms_norm(&n0, c0.x, nw, 1e-5, s));
    try near.check(s, pre0.normed, n0, alloc);
    // The last threadgroup reduces the others' partials: every dispatch must see all of them.
    {
        const first = try readF32(pre0.normed, s, alloc);
        defer alloc.free(first);
        for (0..32) |_| {
            var again = try hcPre(s, stream, null, fn_rows, scale, base, nw, 20, 1e-6, 1e-5, 1e-5);
            defer again.deinit();
            const v = try readF32(again.normed, s, alloc);
            defer alloc.free(v);
            try testing.expectEqualSlices(f32, first, v);
        }
    }
    try near.check(s, pre0.post, c0.post, alloc);
    try near.check(s, pre0.comb, c0.comb, alloc);
    var df = pre0.defer_(y, stream);
    defer df.deinit();
    const want_h = try expand(s, y, stream, &c0);
    defer _ = mlx.mlx_array_free(want_h);
    const got_h = try df.materialize(s);
    defer _ = mlx.mlx_array_free(got_h);
    try near.check(s, got_h, want_h, alloc);

    // Sublayer 2 from the deferred stream: it rebuilds the materialized bits exactly.
    var pre1 = try hcPre(s, null, &df, fn_rows, scale, base, nw, 20, 1e-6, 1e-5, 1e-5);
    defer pre1.deinit();
    {
        const a = try readF32(pre1.h, s, alloc);
        defer alloc.free(a);
        const b = try readF32(got_h, s, alloc);
        defer alloc.free(b);
        try testing.expectEqualSlices(f32, b, a);
    }
    var c1 = try collapse(s, got_h, fn_rows, scale, base, 20, 1e-6, 1e-5);
    defer c1.deinit();
    var n1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n1);
    try mlx.check(mlx.mlx_fast_rms_norm(&n1, c1.x, nw, 1e-5, s));
    try near.check(s, pre1.normed, n1, alloc);
}

test "glm5 hcPre over R rows equals R one-row dispatches, up to the widest verify window" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(33);
    const rand = prng.random();
    const d: c_int = 1024;
    const wv = try alloc.alloc(f32, @intCast(HC * d * MIX));
    defer alloc.free(wv);
    for (wv) |*x| x.* = 0.02 * rand.floatNorm(f32);
    var nv: [1024]f32 = undefined;
    for (&nv) |*x| x.* = 1.0 + 0.1 * rand.floatNorm(f32);
    var bv: [MIX]f32 = undefined;
    for (&bv) |*x| x.* = 0.5 * rand.floatNorm(f32);
    const scv = [3]f32{ 0.8, 1.1, 0.9 };
    const bf = struct {
        fn of(st: mlx.mlx_stream, v: []const f32, shape: []const c_int) !mlx.mlx_array {
            const f = mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f);
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&o, f, .bfloat16, st));
            return o;
        }
    };
    const nw = try bf.of(s, &nv, &.{d});
    defer _ = mlx.mlx_array_free(nw);
    const fn_rows = mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ MIX, HC * d }, 2, .float32);
    defer _ = mlx.mlx_array_free(fn_rows);
    const base = mlx.mlx_array_new_data(&bv, &[_]c_int{MIX}, 1, .float32);
    defer _ = mlx.mlx_array_free(base);
    const scale = mlx.mlx_array_new_data(&scv, &[_]c_int{3}, 1, .float32);
    defer _ = mlx.mlx_array_free(scale);

    var rows: c_int = 2;
    while (rows <= 16) : (rows += 1) {
        const sv = try alloc.alloc(f32, @intCast(rows * HC * d));
        defer alloc.free(sv);
        for (sv) |*x| x.* = rand.floatNorm(f32);
        const stream = try bf.of(s, sv, &.{ 1, rows, HC, d });
        defer _ = mlx.mlx_array_free(stream);
        try testing.expect(hcRowsServes(s, mlx.getShape(stream), .bfloat16, fn_rows, nw));
        var all = try hcPre(s, stream, null, fn_rows, scale, base, nw, 20, 1e-6, 1e-5, 1e-5);
        defer all.deinit();
        const got = try readF32(all.normed, s, alloc);
        defer alloc.free(got);
        const got_post = try readF32(all.post, s, alloc);
        defer alloc.free(got_post);
        var r: c_int = 0;
        while (r < rows) : (r += 1) {
            var one = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(one);
            try mlx.check(mlx.mlx_slice(&one, stream, &[_]c_int{ 0, r, 0, 0 }, 4, &[_]c_int{ 1, r + 1, HC, d }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
            var pre = try hcPre(s, one, null, fn_rows, scale, base, nw, 20, 1e-6, 1e-5, 1e-5);
            defer pre.deinit();
            const want = try readF32(pre.normed, s, alloc);
            defer alloc.free(want);
            const want_post = try readF32(pre.post, s, alloc);
            defer alloc.free(want_post);
            const off: usize = @intCast(r * d);
            try testing.expectEqualSlices(f32, want, got[off .. off + @as(usize, @intCast(d))]);
            const poff: usize = @intCast(r * HC);
            try testing.expectEqualSlices(f32, want_post, got_post[poff .. poff + HC]);
        }
    }
}

test "glm5 head-folded latent attention matches sdpa with the keys broadcast to every head" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(9);
    const rand = prng.random();
    const h: c_int = 64;
    const r: c_int = 512;
    const w: c_int = 300;
    const qv = try alloc.alloc(f32, @intCast(h * r));
    defer alloc.free(qv);
    for (qv) |*x| x.* = rand.floatNorm(f32);
    const kv = try alloc.alloc(f32, @intCast(w * r));
    defer alloc.free(kv);
    for (kv) |*x| x.* = rand.floatNorm(f32);
    var mv: [300]bool = undefined;
    for (&mv, 0..) |*m, i| m.* = i % 7 != 3;
    const qf = mlx.mlx_array_new_data(qv.ptr, &[_]c_int{ 1, h, 1, r }, 4, .float32);
    defer _ = mlx.mlx_array_free(qf);
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_astype(&q, qf, .bfloat16, s));
    const kf = mlx.mlx_array_new_data(kv.ptr, &[_]c_int{ 1, w, r }, 3, .float32);
    defer _ = mlx.mlx_array_free(kf);
    var keys = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keys);
    try mlx.check(mlx.mlx_astype(&keys, kf, .bfloat16, s));
    const valid = mlx.mlx_array_new_data(&mv, &[_]c_int{ 1, 1, w }, 3, .bool_);
    defer _ = mlx.mlx_array_free(valid);
    var k4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k4);
    try mlx.check(mlx.mlx_reshape(&k4, keys, &[_]c_int{ 1, 1, w, r }, 4, s));
    var m4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m4);
    try mlx.check(mlx.mlx_reshape(&m4, valid, &[_]c_int{ 1, 1, 1, w }, 4, s));
    const none = mlx.mlx_array{ .ctx = null };
    for ([_]bool{ false, true }) |masked| {
        const got = try latentAttendFolded(s, q, keys, if (masked) valid else null, 0.05);
        defer _ = mlx.mlx_array_free(got);
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&want, q, k4, k4, 0.05, if (masked) "array" else "", if (masked) m4 else none, none, false, s));
        const a = try readF32(got, s, alloc);
        defer alloc.free(a);
        const b = try readF32(want, s, alloc);
        defer alloc.free(b);
        try testing.expectEqual(b.len, a.len);
        for (a, b) |x, y| try testing.expectApproxEqAbs(y, x, 0.02 + 0.02 * @abs(y));
    }
}

test "glm5 one-query indexer scores match the NAX scorer over the pooled keys" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(17);
    const rand = prng.random();
    const ix: Indexer = .{ .topk = 512, .kpool = 4, .heads = 32, .head_dim = 128 };
    const p: c_int = 700;
    const t: c_int = p * ix.kpool + 3;
    const q_pos: c_int = t - 9; // the last pools are not visible yet
    const bf = struct {
        fn of(st: mlx.mlx_stream, al: std.mem.Allocator, r: std.Random, shape: []const c_int, sd: f32) !mlx.mlx_array {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const v = try al.alloc(f32, n);
            defer al.free(v);
            for (v) |*x| x.* = sd * r.floatNorm(f32);
            const f = mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f);
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&o, f, .bfloat16, st));
            return o;
        }
    };
    const raw = try bf.of(s, alloc, rand, &.{ 1, t, 2 * ix.head_dim }, 1.0);
    defer _ = mlx.mlx_array_free(raw);
    const q = try bf.of(s, alloc, rand, &.{ 1, 1, 32, ix.head_dim }, 1.0);
    defer _ = mlx.mlx_array_free(q);
    var av: [4 * 128]f32 = undefined;
    for (&av) |*x| x.* = rand.floatNorm(f32);
    const ape = mlx.mlx_array_new_data(&av, &[_]c_int{ 4, 128 }, 2, .float32);
    defer _ = mlx.mlx_array_free(ape);
    var wv: [32]f32 = undefined;
    for (&wv) |*x| x.* = 0.05 * rand.floatNorm(f32);
    const ws = mlx.mlx_array_new_data(&wv, &[_]c_int{ 1, 1, 32 }, 3, .float32);
    defer _ = mlx.mlx_array_free(ws);
    try testing.expect(decodeSelectServes(s, q, raw, ix));
    const cache = try withPooledKeys(s, raw, raw, ape, 0, ix);
    defer _ = mlx.mlx_array_free(cache);

    const got = try decodeScores(s, q, ws, cache, ix, q_pos, p);
    defer _ = mlx.mlx_array_free(got);
    const pk = try storedPoolKeys(s, cache, p, ix);
    defer _ = mlx.mlx_array_free(pk);
    const want = (try indexerScoresNax(s, q, ws, pk, q_pos, ix.kpool)) orelse return error.SkipZigTest;
    defer _ = mlx.mlx_array_free(want);
    const a = try readF32(got, s, alloc);
    defer alloc.free(a);
    const b = try readF32(want, s, alloc);
    defer alloc.free(b);
    try testing.expectEqual(b.len, a.len);
    for (a, b) |x, y| {
        if (y <= -1e30) try testing.expect(x <= -1e30) else try testing.expectApproxEqAbs(y, x, 1e-3 + 1e-3 * @abs(y));
    }
}

test "glm5 one-query top-k expands the chronological top pools, ties to the lower index" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(23);
    const rand = prng.random();
    const ix: Indexer = .{ .topk = 512, .kpool = 4, .heads = 32, .head_dim = 128 };
    const k: c_int = 128;
    for ([_]c_int{ 700, 5000 }) |p| {
        const q_pos: c_int = p * ix.kpool - 6; // the last two pools are not visible
        const sv = try alloc.alloc(f32, @intCast(p));
        defer alloc.free(sv);
        for (sv) |*x| x.* = rand.floatNorm(f32);
        // A tie straddling the threshold, and the invisible pools' scores.
        const sorted = try alloc.dupe(f32, sv);
        defer alloc.free(sorted);
        std.mem.sort(f32, sorted, {}, std.sort.desc(f32));
        const thr = sorted[@intCast(k - 2)];
        for (sv, 0..) |*x, i| if (i % 97 == 5) {
            x.* = thr;
        };
        sv[@intCast(p - 1)] = -std.math.floatMax(f32);
        sv[@intCast(p - 2)] = -std.math.floatMax(f32);
        const scores = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{p}, 1, .float32);
        defer _ = mlx.mlx_array_free(scores);
        const got = try topkExpand(s, scores, ix, q_pos, p, k);
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_array_eval(got));
        const gv = mlx.mlx_array_data_int32(got).?[0..@intCast(ix.topk + ix.kpool - 1)];

        // Host: rank by (score desc, index asc), keep k, chronological, expanded.
        const order = try alloc.alloc(u32, @intCast(p));
        defer alloc.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        const Ctx = struct {
            v: []const f32,
            fn lt(c: @This(), x: u32, y: u32) bool {
                return if (c.v[x] != c.v[y]) c.v[x] > c.v[y] else x < y;
            }
        };
        std.mem.sort(u32, order, Ctx{ .v = sv }, Ctx.lt);
        const top = order[0..@intCast(k)];
        std.mem.sort(u32, top, {}, std.sort.asc(u32));
        for (top, 0..) |pool, slot| for (0..4) |c| {
            const pi: c_int = @intCast(pool);
            const want: i32 = if ((pi + 1) * 4 - 1 <= q_pos) pi * 4 + @as(c_int, @intCast(c)) else -1;
            try testing.expectEqual(want, gv[slot * 4 + c]);
        };
        for (@intCast(k * 4)..@intCast(ix.topk)) |col| try testing.expectEqual(@as(i32, -1), gv[col]);
        const tail_count = @rem(q_pos + 1, 4);
        for (0..3) |tcol| {
            const want: i32 = if (@as(c_int, @intCast(tcol)) < tail_count) q_pos + 1 - tail_count + @as(c_int, @intCast(tcol)) else -1;
            try testing.expectEqual(want, gv[@as(usize, @intCast(ix.topk)) + tcol]);
        }
    }
}

test "glm5 one-token per-head qmv matches MLX's batched quantized matmul" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(31);
    const rand = prng.random();
    for ([_][3]c_int{ .{ 64, 512, 256 }, .{ 64, 256, 512 } }) |dims| {
        const h = dims[0];
        const n = dims[1];
        const k = dims[2];
        for ([_]u32{ 4, 5, 8 }) |bits| {
            const wv = try alloc.alloc(f32, @intCast(h * n * k));
            defer alloc.free(wv);
            for (wv) |*v| v.* = 0.05 * rand.floatNorm(f32);
            const xv = try alloc.alloc(f32, @intCast(h * k));
            defer alloc.free(xv);
            for (xv) |*v| v.* = rand.floatNorm(f32);
            const wf = mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ h, n, k }, 3, .float32);
            defer _ = mlx.mlx_array_free(wf);
            var wb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wb);
            try mlx.check(mlx.mlx_astype(&wb, wf, .bfloat16, s));
            var qv = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(qv);
            try mlx.check(mlx.mlx_quantize(&qv, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, s));
            var wq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            var sc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sc);
            var bi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bi);
            try mlx.check(mlx.mlx_vector_array_get(&wq, qv, 0));
            try mlx.check(mlx.mlx_vector_array_get(&sc, qv, 1));
            try mlx.check(mlx.mlx_vector_array_get(&bi, qv, 2));
            const xf = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ 1, h, 1, k }, 4, .float32);
            defer _ = mlx.mlx_array_free(xf);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));
            // qmv_fast needs K in whole 2-pack blocks (512 at 4 and 5 bits): K 256 is declined there.
            const got = (try headQmv(s, x, wq, sc, bi, bits, 64, 1.0)) orelse {
                try testing.expect(bits != 8 and k == 256);
                continue;
            };
            defer _ = mlx.mlx_array_free(got);
            var want = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(want);
            try mlx.check(mlx.mlx_quantized_matmul(&want, x, wq, sc, bi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));
            const a = try readF32(got, s, alloc);
            defer alloc.free(a);
            const b = try readF32(want, s, alloc);
            defer alloc.free(b);
            try testing.expectEqual(b.len, a.len);
            for (a, b) |p, q| try testing.expectApproxEqAbs(q, p, 0.01 + 0.01 * @abs(q));
            // The output scale lands before the one rounding.
            const scaled = (try headQmv(s, x, wq, sc, bi, bits, 64, 0.25)).?;
            defer _ = mlx.mlx_array_free(scaled);
            const c = try readF32(scaled, s, alloc);
            defer alloc.free(c);
            for (c, b) |p, q| try testing.expectApproxEqAbs(0.25 * q, p, 0.01 + 0.01 * @abs(q));
        }
    }
}

test "glm5 an output tied to a cache update evaluates the update with it" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const a = mlx.mlx_array_new_float(2.0);
    defer _ = mlx.mlx_array_free(a);
    var side = mlx.mlx_array_new(); // stands in for the indexer row nothing else reads
    defer _ = mlx.mlx_array_free(side);
    try mlx.check(mlx.mlx_multiply(&side, a, a, s));
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, a, a, s));
    const tied = try withDependency(out, side);
    defer _ = mlx.mlx_array_free(tied);
    try mlx.check(mlx.mlx_array_eval(tied));
    var done = false;
    try mlx.check(mlx._mlx_array_is_available(&done, side));
    try testing.expect(done);
}

test "glm5 a pool's completing row carries its pooled key, across a chunk boundary" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(47);
    const rand = prng.random();
    const ix: Indexer = .{ .topk = 8, .kpool = 4, .heads = 2, .head_dim = 128 };
    const t: c_int = 30;
    const cv = try alloc.alloc(f32, @intCast(t * 2 * ix.head_dim));
    defer alloc.free(cv);
    for (cv) |*x| x.* = rand.floatNorm(f32);
    const av = try alloc.alloc(f32, @intCast(ix.kpool * ix.head_dim));
    defer alloc.free(av);
    for (av) |*x| x.* = rand.floatNorm(f32);
    const cf = mlx.mlx_array_new_data(cv.ptr, &[_]c_int{ 1, t, 2 * ix.head_dim }, 3, .float32);
    defer _ = mlx.mlx_array_free(cf);
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_astype(&rows, cf, .bfloat16, s));
    const ape = mlx.mlx_array_new_data(av.ptr, &[_]c_int{ ix.kpool, ix.head_dim }, 2, .float32);
    defer _ = mlx.mlx_array_free(ape);
    const w = 2 * ix.head_dim;
    // Rows [0, 13) are cached, [13, 30) arrive in this forward: pool 3 straddles the boundary.
    const split: c_int = 13;
    var prev = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prev);
    try mlx.check(mlx.mlx_slice(&prev, rows, &[_]c_int{ 0, split - 3, 0 }, 3, &[_]c_int{ 1, split, w }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var chunk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chunk);
    try mlx.check(mlx.mlx_slice(&chunk, rows, &[_]c_int{ 0, split, 0 }, 3, &[_]c_int{ 1, t, w }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    const got = try withPooledKeys(s, prev, chunk, ape, split, ix);
    defer _ = mlx.mlx_array_free(got);

    var keys = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keys);
    var gates = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gates);
    const p: c_int = 7;
    try mlx.check(mlx.mlx_slice(&keys, rows, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, p * 4, ix.head_dim }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try mlx.check(mlx.mlx_slice(&gates, rows, &[_]c_int{ 0, 0, ix.head_dim }, 3, &[_]c_int{ 1, p * 4, w }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    const want_pk = try poolKeys(s, keys, gates, ape, p, ix);
    defer _ = mlx.mlx_array_free(want_pk);
    const g = try readF32(got, s, alloc);
    defer alloc.free(g);
    const r = try readF32(rows, s, alloc);
    defer alloc.free(r);
    const pk = try readF32(want_pk, s, alloc);
    defer alloc.free(pk);
    const W: usize = @intCast(w);
    const D: usize = @intCast(ix.head_dim);
    for (0..@intCast(t - split)) |i| {
        const pos = i + @as(usize, @intCast(split));
        for (0..W) |c| {
            const want = if (pos % 4 == 3 and c >= D) pk[(pos / 4) * D + (c - D)] else r[pos * W + c];
            // The chain sums 4 bf16 products in bf16; the kernel in f32.
            try testing.expectApproxEqAbs(want, g[i * W + c], 0.03 + 0.02 * @abs(want));
        }
    }
}

test "glm5 a deferred expand over R rows rebuilds the materialized stream and feeds the next hcPre, bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(77);
    const rand = prng.random();
    const d: c_int = 1024;
    const wv = try alloc.alloc(f32, @intCast(HC * d * MIX));
    defer alloc.free(wv);
    for (wv) |*x| x.* = 0.02 * rand.floatNorm(f32);
    var nv: [1024]f32 = undefined;
    for (&nv) |*x| x.* = 1.0 + 0.1 * rand.floatNorm(f32);
    var bv: [MIX]f32 = undefined;
    for (&bv) |*x| x.* = 0.5 * rand.floatNorm(f32);
    const scv = [3]f32{ 0.8, 1.1, 0.9 };
    const fn_rows = mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ MIX, HC * d }, 2, .float32);
    defer _ = mlx.mlx_array_free(fn_rows);
    const base = mlx.mlx_array_new_data(&bv, &[_]c_int{MIX}, 1, .float32);
    defer _ = mlx.mlx_array_free(base);
    const scale = mlx.mlx_array_new_data(&scv, &[_]c_int{3}, 1, .float32);
    defer _ = mlx.mlx_array_free(scale);
    const bf = struct {
        fn of(st: mlx.mlx_stream, r: std.Random, a: std.mem.Allocator, shape: []const c_int) !mlx.mlx_array {
            var n: usize = 1;
            for (shape) |x| n *= @intCast(x);
            const buf = try a.alloc(f32, n);
            defer a.free(buf);
            for (buf) |*x| x.* = r.floatNorm(f32);
            const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f);
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&o, f, .bfloat16, st));
            return o;
        }
    };
    const nw = try bf.of(s, rand, alloc, &.{d});
    defer _ = mlx.mlx_array_free(nw);
    for ([_]c_int{ 2, 5, HC_PRE_MAX_ROWS }) |rows| {
        const stream = try bf.of(s, rand, alloc, &.{ 1, rows, HC, d });
        defer _ = mlx.mlx_array_free(stream);
        const y = try bf.of(s, rand, alloc, &.{ 1, rows, d });
        defer _ = mlx.mlx_array_free(y);
        try testing.expect(hcRowsServes(s, mlx.getShape(stream), .bfloat16, fn_rows, nw));
        var pre0 = try hcPre(s, stream, null, fn_rows, scale, base, nw, 20, 1e-6, 1e-5, 1e-5);
        defer pre0.deinit();
        const c0: Collapsed = .{ .x = pre0.normed, .post = pre0.post, .comb = pre0.comb };
        const want_h = try expand(s, y, stream, &c0);
        defer _ = mlx.mlx_array_free(want_h);
        var df = pre0.defer_(y, stream);
        defer df.deinit();
        const got_h = try df.materialize(s);
        defer _ = mlx.mlx_array_free(got_h);
        var want_pre = try hcPre(s, want_h, null, fn_rows, scale, base, nw, 20, 1e-6, 1e-5, 1e-5);
        defer want_pre.deinit();
        var got_pre = try hcPre(s, null, &df, fn_rows, scale, base, nw, 20, 1e-6, 1e-5, 1e-5);
        defer got_pre.deinit();
        const pairs = [_][2]mlx.mlx_array{ .{ got_h, want_h }, .{ got_pre.h, want_h }, .{ got_pre.normed, want_pre.normed }, .{ got_pre.post, want_pre.post }, .{ got_pre.comb, want_pre.comb } };
        for (pairs, 0..) |pr, i| {
            const a = try readF32(pr[0], s, alloc);
            defer alloc.free(a);
            const b = try readF32(pr[1], s, alloc);
            defer alloc.free(b);
            if (!std.mem.eql(f32, a, b)) std.debug.print("{d} rows: output {d} differs\n", .{ rows, i });
            try testing.expectEqualSlices(f32, b, a);
        }
    }
}
