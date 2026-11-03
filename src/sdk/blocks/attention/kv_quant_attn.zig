//! sdk/blocks/attention/kv_quant_attn.
//! Owns QKV_DEC_NT, qkvDecBlockFor, QKV_DEC_KERNEL_SOURCE, qkv_dec_kernel_cached, getQkvDecKernel, qkv_dec_override, and 52 more.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const verifyQmmNaxAvailable = @import("../linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const verifyQmmTile = @import("../linear/verify_qmm_nax.zig").verifyQmmTile;
const DenseKVView = @import("kv_view.zig").DenseKVView;

// ── Decode quantized-KV attention kernel (docs/kv-quant-perf.md) ──
//
// The composed fused path (quantAttention: qmm → softmax → qmm) measured
// SLOWER than dense-mode dequant+SDPA at real GQA ratios in the Phase 0
// µbench (48/8 @ 32K: composed 1.47 ms vs dense 1.11 vs kv-off 0.51) —
// mlx_quantized_matmul at M=g rows over batched 4-D banks runs ~6x below
// bandwidth. This kernel reads the packed cache views IN PLACE at decode
// width: one threadgroup per q head, NSG simdgroups striding the kv rows
// with per-simdgroup online-softmax carry (m/l/acc), merged once through
// threadgroup memory. The dense [B,H,T,D] K/V never exist, so per layer per
// token the traffic is exactly the packed bytes — BELOW kv-quant-off's dense
// read, which is the whole point of quantized KV at long context.
//
// GQA is handled in-kernel (head h_q reads bank h_q / GQA) — no repeat, no
// grouping reshape. K/V arrive as cache VIEWS (non-contiguous along the head
// axis whenever buffer capacity > logical length), hence
// ensure_row_contiguous=false + explicit stride indexing, same contract as
// msv_attn_p256. Dequant arithmetic mirrors gatherQmv's affine form
// (q*scale+bias per GS-group, f32 accumulate). NOT bit-identical to dense
// SDPA (different reduction order + f32 dequant vs bf16-rounded dense
// views) — sanctioned qmv-vs-qmm class, pinned no-worse-than-reference.
const QKV_DEC_NT = 128; // threads per threadgroup

/// Threadgroup block length along the kv axis, chosen so the score buffer
/// (GQA·BLOCK f32) + the staged q rows (GQA·DK f32) fit Metal's 32 KiB
/// threadgroup budget (a kernel's block size and its memory footprint are
/// ONE decision — the gdnBlockTFor class). Returns null when no block fits.
fn qkvDecBlockFor(gqa: c_int, dk: c_int) ?c_int {
    // ≤ ~10 KiB per threadgroup, NOT the full 32 KiB budget: threadgroup
    // memory is the residency limiter — a 28 KiB footprint leaves ONE
    // threadgroup per core and the latency-hiding collapses (measured
    // 1.9 ms vs dense 1.1 at 48/8 32K with 24 KiB score buffers).
    const cands = [_]c_int{ 512, 256, 128 };
    for (cands) |b| {
        const bytes: i64 = @as(i64, gqa) * @as(i64, b) * 4 + @as(i64, gqa) * @as(i64, dk) * 4 + QKV_DEC_NT * 4 + 256;
        if (bytes <= 10_000) return b;
    }
    return null;
}

// Two-phase block-parallel layout (v1 — one threadgroup per Q HEAD with a
// per-row simd_sum online-softmax chain — measured 5x SLOWER than dense
// mode: 48 threadgroups starve the GPU and the per-row reduction+exp chain
// serializes each simdgroup; see docs/kv-quant-perf.md Phase 0 table):
//
//   grid = (NT, H_kv, ceil(Tk / BLOCK))
//   Phase A — scores: thread t computes rows t, t+NT, … of the block: one
//     full-D dequant dot per (row, q-sub-head) against the g staged q rows.
//     No cross-thread reductions on the critical path.
//   Phase M — per sub-head block max + exp + row-weight sum via a shared
//     tree reduction in threadgroup memory.
//   Phase B — output: thread e walks ALL block rows for element e, dequants
//     V[t][e] ONCE and feeds all g sub-head accumulators (the packed bank is
//     read exactly once per block regardless of GQA ratio).
//
// The kernel emits per-block PARTIALS (m, l, unnormalized O) — the standard
// flash-attention split-KV carry — merged by a handful of KB-scale mlx ops
// in the dispatch wrapper. A fully-masked block clamps its max to -1e30 so
// exp never sees (-inf) - (-inf); its l = 0 partial merges to zero weight.
pub const QKV_DEC_KERNEL_SOURCE = kernelSource("kernels/attention/qkv_dec.metal");

var qkv_dec_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQkvDecKernel() !mlx.mlx_fast_metal_kernel {
    if (qkv_dec_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "kq", "ksc", "kbi", "vq", "vsc", "vbi", "scl", "tk", "msk" };
    const output_names = [_][*:0]const u8{ "m_out", "l_out", "o_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "qkv_attn_dec",
        in_vec,
        out_vec,
        QKV_DEC_KERNEL_SOURCE,
        "",
        false, // K/V are cache VIEWS (non-contiguous along the head axis) —
        // a forced contiguous copy of the packed cache would erase the win.
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qkv_dec_kernel_cached = kernel;
    return kernel;
}

pub var qkv_dec_override: ?bool = null; // test seam
fn qkvDecKernelEnabled() bool {
    return qkv_dec_override orelse true;
}

const QkvDecCfgKey = struct {
    hq: c_int,
    dk: c_int,
    dv: c_int,
    bits: u8,
    gs: u32,
    gqa: c_int,
    nblk: c_int,
    has_mask: bool,
    dtype: mlx.mlx_dtype,
};
var qkv_dec_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var qkv_dec_cfg_key: QkvDecCfgKey = std.mem.zeroes(QkvDecCfgKey);
var qkv_dec_engaged: bool = false;

/// Try the decode-width quantized-KV attention kernel. Returns null when a
/// precondition doesn't hold — the caller falls back to the composed
/// quantAttention. Preconditions are the kernel's OWN requirements, never a
/// model_type list.
pub fn qkvAttnDecodeKernel(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    view: *const DenseKVView,
    scale: f32,
    mask_mode: []const u8,
    mask_arr: mlx.mlx_array,
) !?mlx.mlx_array {
    if (!qkvDecKernelEnabled()) return null;
    if (!view.has_quant_triple) return null;
    if (!qkvVerifyForced() and mlx.mlx_array_ndim(q) == 4 and mlx.mlx_array_ndim(view.k_triple_q) == 4 and
        qkvMppWins(mlx.getShape(q)[2], mlx.getShape(view.k_triple_q)[2]))
    {
        if (try qkvAttnMppKernel(s, q, view, scale, mask_mode)) |o| return o;
    }
    if (view.bits != 4 and view.bits != 8) return null;
    if (view.group_size == 0) return null;
    if (mlx.mlx_array_ndim(q) != 4) return null;
    const qs = mlx.getShape(q);
    if (qs[0] != 1 or qs[2] != 1) return null; // decode width only; batched decode keeps dense reads
    const qdt = mlx.mlx_array_dtype(q);
    if (qdt != .bfloat16 and qdt != .float16 and qdt != .float32) return null;
    if (mlx.mlx_array_ndim(view.k_triple_q) != 4 or mlx.mlx_array_ndim(view.v_triple_q) != 4) return null;
    const ks = mlx.getShape(view.k_triple_q);
    const vs = mlx.getShape(view.v_triple_q);
    const vpw: c_int = @divExact(@as(c_int, 32), @as(c_int, view.bits));
    const dk: c_int = ks[3] * vpw;
    const dv: c_int = vs[3] * vpw;
    if (qs[3] != dk) return null;
    if (@rem(dk, 32) != 0 or @rem(dv, 32) != 0) return null;
    if (@rem(dk, @as(c_int, @intCast(view.group_size))) != 0 or @rem(dv, @as(c_int, @intCast(view.group_size))) != 0) return null;
    const h_q: c_int = qs[1];
    const h_kv: c_int = ks[1];
    if (h_kv <= 0 or @rem(h_q, h_kv) != 0) return null;
    if (vs[1] != h_kv or vs[2] != ks[2] or ks[0] != 1 or vs[0] != 1) return null;
    const gqa: c_int = @divExact(h_q, h_kv);
    const t_k: c_int = ks[2];
    if (t_k <= 0) return null;
    // Scales/biases layout must match the (dk/gs) geometry.
    const ksc_sh = mlx.getShape(view.k_triple_scales);
    const vsc_sh = mlx.getShape(view.v_triple_scales);
    if (ksc_sh.len != 4 or vsc_sh.len != 4) return null;
    if (ksc_sh[3] != @divExact(dk, @as(c_int, @intCast(view.group_size)))) return null;
    if (vsc_sh[3] != @divExact(dv, @as(c_int, @intCast(view.group_size)))) return null;

    // Mask: none, trivially-causal (T_q == 1), or a [.., 1, T_k] additive row.
    var has_mask = false;
    if (std.mem.eql(u8, mask_mode, "array")) {
        if (mask_arr.ctx == null) return null;
        const msize = mlx.mlx_array_size(mask_arr);
        if (msize != @as(usize, @intCast(t_k))) return null;
        has_mask = true;
    } else if (!std.mem.eql(u8, mask_mode, "") and !std.mem.eql(u8, mask_mode, "causal")) {
        return null;
    }

    const kernel = getQkvDecKernel() catch return null;

    const block = qkvDecBlockFor(gqa, dk) orelse return null;
    const nblk: c_int = @divTrunc(t_k + block - 1, block);

    const key = QkvDecCfgKey{
        .hq = h_q,
        .dk = dk,
        .dv = dv,
        .bits = view.bits,
        .gs = view.group_size,
        .gqa = gqa,
        .nblk = nblk,
        .has_mask = has_mask,
        .dtype = qdt,
    };
    if (qkv_dec_cfg == null or !std.meta.eql(qkv_dec_cfg_key, key)) {
        if (qkv_dec_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const ml_shape = [_]c_int{ h_q, nblk };
        const o_shape = [_]c_int{ h_q, nblk, dv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, QKV_DEC_NT, h_kv, nblk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, QKV_DEC_NT, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", qdt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(view.bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(view.group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DK", dk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DV", dv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GQA", gqa));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BLOCK", block));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HAS_MASK", @intFromBool(has_mask)));
        qkv_dec_cfg = config;
        qkv_dec_cfg_key = key;
    }

    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const tk_data = [_]i32{t_k};
    const tk = mlx.mlx_array_new_data(&tk_data, &one, 1, .int32);
    defer _ = mlx.mlx_array_free(tk);
    // Absent mask: a 1-elem f32 stand-in (HAS_MASK=0 never reads it).
    const dummy_data = [_]f32{0};
    const dummy = mlx.mlx_array_new_data(&dummy_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(dummy);

    const inputs_arr = [_]mlx.mlx_array{
        q,
        view.k_triple_q,
        view.k_triple_scales,
        view.k_triple_biases,
        view.v_triple_q,
        view.v_triple_scales,
        view.v_triple_biases,
        scl,
        tk,
        if (has_mask) mask_arr else dummy,
    };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, qkv_dec_cfg.?, s));
    const o_t = try qkvMergePartials(s, outputs_vec, qdt);
    defer _ = mlx.mlx_array_free(o_t);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const out_shape = [_]c_int{ 1, h_q, 1, dv };
    try mlx.check(mlx.mlx_reshape(&out, o_t, &out_shape, 4, s));
    if (!qkv_dec_engaged) {
        qkv_dec_engaged = true;
        log.info("[kv-attn] decode kernel engaged: bits={d} gs={d} DK={d} DV={d} Hq={d} Hkv={d} block={d} mask={}\n", .{ view.bits, view.group_size, dk, dv, h_q, h_kv, block, has_mask });
    }
    return out;
}

/// Merge per-block split-KV partials (m, l, unnormalized O) into a typed
/// [ROWS, DV] tensor — the standard flash-attention carry:
///   M  = max_b m_b            [ROWS, 1]
///   w  = exp(m_b - M)         [ROWS, NBLK]
///   L  = Σ_b l_b · w          [ROWS, 1]
///   O  = Σ_b o_b · w          [ROWS, DV]
///   out = (O / L) as T.
/// KB-scale tensors — a handful of elementwise/reduce ops per layer. Shared
/// by the decode and verify packed-attention kernels; the op sequence is
/// load-bearing (decode bit-identity across refactors is a bar).
fn qkvMergePartials(s: mlx.mlx_stream, outputs_vec: mlx.mlx_vector_array, qdt: mlx.mlx_dtype) !mlx.mlx_array {
    if (mlx.mlx_vector_array_size(outputs_vec) != 3) return error.MetalKernelBadOutputCount;
    var m_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m_b);
    var l_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l_b);
    var o_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_b);
    try mlx.check(mlx.mlx_vector_array_get(&m_b, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&l_b, outputs_vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&o_b, outputs_vec, 2));
    return qkvMergeArrays(s, m_b, l_b, o_b, qdt);
}

pub fn qkvMergeArrays(s: mlx.mlx_stream, m_b: mlx.mlx_array, l_b: mlx.mlx_array, o_b: mlx.mlx_array, qdt: mlx.mlx_dtype) !mlx.mlx_array {
    var m_max = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m_max);
    try mlx.check(mlx.mlx_max_axis(&m_max, m_b, 1, true, s));
    var m_shift = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m_shift);
    try mlx.check(mlx.mlx_subtract(&m_shift, m_b, m_max, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_exp(&w, m_shift, s));
    var lw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lw);
    try mlx.check(mlx.mlx_multiply(&lw, l_b, w, s));
    var l_tot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l_tot);
    try mlx.check(mlx.mlx_sum_axis(&l_tot, lw, 1, true, s)); // [ROWS, 1]
    var w3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w3);
    try mlx.check(mlx.mlx_expand_dims(&w3, w, 2, s)); // [ROWS, NBLK, 1]
    var ow = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ow);
    try mlx.check(mlx.mlx_multiply(&ow, o_b, w3, s));
    var o_tot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_tot);
    try mlx.check(mlx.mlx_sum_axis(&o_tot, ow, 1, false, s)); // [ROWS, DV]
    var o_norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_norm);
    try mlx.check(mlx.mlx_divide(&o_norm, o_tot, l_tot, s)); // [ROWS,DV] / [ROWS,1]
    var o_t = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o_t);
    try mlx.check(mlx.mlx_astype(&o_t, o_norm, qdt, s));
    return o_t;
}

// ── Verify-width packed attention (kv8-fused-spec-plan.md Phase 2) ──
//
// Same two-phase block-parallel layout as QKV_DEC_KERNEL_SOURCE, generalized
// to TQ (2..8) query rows per head — the spec-verify widths that Phase 1
// routed to dense dequant + SDPA. Differences from the decode kernel:
//   - q is read from DEVICE memory, not staged: LQ = GQA·TQ rows of DK f32
//     is up to 49 KB of threadgroup memory at TQ 8. The reads are lockstep
//     across threads (same element for every row of the block) and the whole
//     q tensor is L1-scale.
//   - the causal TAIL is applied from geometry: q row tq sees keys with
//     global index < Tk - TQ + 1 + tq — the same end-aligned contract as
//     SDPA "causal", which is what the dense path uses at these widths.
//     "array" masks (sliding layers) decline to dense.
//   - scale is folded at score write, not at q read.
// Partials merge through the SAME qkvMergePartials carry; row order is
// (head-major, tq inner) so the merged [Hq·TQ, DV] reshapes straight to
// [1, Hq, TQ, DV].
pub const QKV_VERIFY_MAX_TQ: c_int = 8;

pub const QKV_VERIFY_KERNEL_SOURCE = kernelSource("kernels/attention/qkv_verify.metal");

var qkv_ver_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQkvVerifyKernel() !mlx.mlx_fast_metal_kernel {
    if (qkv_ver_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "kq", "ksc", "kbi", "vq", "vsc", "vbi", "scl", "tk" };
    const output_names = [_][*:0]const u8{ "m_out", "l_out", "o_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "qkv_attn_verify",
        in_vec,
        out_vec,
        QKV_VERIFY_KERNEL_SOURCE,
        "",
        false, // K/V are cache VIEWS — see getQkvDecKernel.
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qkv_ver_kernel_cached = kernel;
    return kernel;
}

pub var qkv_ver_override: ?bool = null; // test seam
pub fn qkvVerifyKernelEnabledFrom(nax_available: bool) bool {
    // The dk<=256 adoption set was calibrated on M4. The M5/G17 round found
    // it slower at 9K KV for q4/q6/q8 targets, so G17 keeps dense verify
    // attention.
    return !nax_available;
}

/// The kernel keeps one accumulator per q row (gqa x t_q); past 12 they spill
/// out of registers and dequant + dense sdpa wins at every kv length.
const QKV_VERIFY_MAX_ROWS: c_int = 12;
pub fn qkvVerifyRowsPay(gqa: c_int, t_q: c_int) bool {
    return gqa * t_q <= QKV_VERIFY_MAX_ROWS;
}

/// The test seam false kills the whole verify width class.
fn qkvVerifyEnvAllows() bool {
    return qkv_ver_override orelse true;
}

/// The test seam keeps every width on the kernel.
fn qkvVerifyForced() bool {
    return qkv_ver_override orelse false;
}

fn qkvVerifyKernelEnabled() bool {
    if (qkv_ver_override) |v| return v;
    return qkvVerifyKernelEnabledFrom(verifyQmmNaxAvailable());
}

/// Threadgroup budget for the verify kernel: scores only (q is device-read).
/// 16 KB rather than decode's 10 KB — at TQ x the arithmetic intensity the
/// occupancy loss buys more reuse per byte; the live A/B is the judge.
fn qkvVerifyBlockFor(gqa: c_int, tq: c_int) ?c_int {
    const lq: i64 = @as(i64, gqa) * @as(i64, tq);
    const cands = [_]c_int{ 256, 128, 64 };
    for (cands) |b| {
        const bytes: i64 = lq * @as(i64, b) * 4 + QKV_DEC_NT * 4 + 8 * lq + 256;
        if (bytes <= 16_000) return b;
    }
    return null;
}

const QkvVerCfgKey = struct {
    hq: c_int,
    dk: c_int,
    dv: c_int,
    bits: u8,
    gs: u32,
    gqa: c_int,
    tq: c_int,
    nblk: c_int,
    dtype: mlx.mlx_dtype,
};
// Per-TQ config slots: verify widths alternate round to round (partial MTP
// rounds, variable PLD match lengths), so a single-entry cache would rebuild
// per call — the ShapeKey lesson, one slot per width instead.
var qkv_ver_cfgs: [@intCast(QKV_VERIFY_MAX_TQ + 1)]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var qkv_ver_cfg_keys: [@intCast(QKV_VERIFY_MAX_TQ + 1)]QkvVerCfgKey = @splat(std.mem.zeroes(QkvVerCfgKey));
var qkv_ver_engaged_mask: u16 = 0; // one-shot log per TQ

/// Try the verify-width packed-KV attention kernel (T_q 2..8, causal only).
/// Returns null when a precondition or machine adoption gate doesn't hold —
/// the caller falls back to dense SDPA. G17 keeps dense after its q4/q6/q8 A/B.
pub fn qkvAttnVerifyKernel(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    view: *const DenseKVView,
    scale: f32,
    mask_mode: []const u8,
) !?mlx.mlx_array {
    if (!qkvVerifyForced() and view.has_quant_triple and mlx.mlx_array_ndim(q) == 4 and mlx.mlx_array_ndim(view.k_triple_q) == 4) {
        if (qkvMppWins(mlx.getShape(q)[2], mlx.getShape(view.k_triple_q)[2])) {
            if (try qkvAttnMppKernel(s, q, view, scale, mask_mode)) |o| return o;
        }
    }
    if (!qkvVerifyKernelEnabled()) return null;
    if (!view.has_quant_triple) return null;
    if (view.bits != 4 and view.bits != 8) return null;
    if (view.group_size == 0) return null;
    if (mlx.mlx_array_ndim(q) != 4) return null;
    const qs = mlx.getShape(q);
    const t_q: c_int = qs[2];
    if (qs[0] != 1 or t_q < 2 or t_q > QKV_VERIFY_MAX_TQ) return null;
    // The dense path masks these widths "causal" (end-aligned); the kernel
    // applies exactly that tail from geometry. Anything else declines.
    if (!std.mem.eql(u8, mask_mode, "causal")) return null;
    const qdt = mlx.mlx_array_dtype(q);
    if (qdt != .bfloat16 and qdt != .float16 and qdt != .float32) return null;
    if (mlx.mlx_array_ndim(view.k_triple_q) != 4 or mlx.mlx_array_ndim(view.v_triple_q) != 4) return null;
    const ks = mlx.getShape(view.k_triple_q);
    const vs = mlx.getShape(view.v_triple_q);
    const vpw: c_int = @divExact(@as(c_int, 32), @as(c_int, view.bits));
    const dk: c_int = ks[3] * vpw;
    const dv: c_int = vs[3] * vpw;
    if (qs[3] != dk) return null;
    if (@rem(dk, 32) != 0 or @rem(dv, 32) != 0) return null;
    // Shape gate is MEASURED adoption (the mixedNaxShapeEnabled rule): every
    // win shape is dk <= 256 (qwen3_5 24/4 hd256: +4..+7% over dense at
    // 8k/16k/32k spec-on); gemma4's full-attn gqa16 x dk512 measured a
    // consistent −1% at Tq 2 and reads dense instead.
    if (dk > 256) return null;
    if (@rem(dk, @as(c_int, @intCast(view.group_size))) != 0 or @rem(dv, @as(c_int, @intCast(view.group_size))) != 0) return null;
    const h_q: c_int = qs[1];
    const h_kv: c_int = ks[1];
    if (h_kv <= 0 or @rem(h_q, h_kv) != 0) return null;
    if (vs[1] != h_kv or vs[2] != ks[2] or ks[0] != 1 or vs[0] != 1) return null;
    const gqa: c_int = @divExact(h_q, h_kv);
    const lq: c_int = gqa * t_q;
    if (lq > QKV_DEC_NT) return null; // the m/l writer covers one row per thread
    if (!qkvVerifyRowsPay(gqa, t_q) and !qkvVerifyForced()) return null;
    const t_k: c_int = ks[2];
    if (t_k < t_q) return null;
    const ksc_sh = mlx.getShape(view.k_triple_scales);
    const vsc_sh = mlx.getShape(view.v_triple_scales);
    if (ksc_sh.len != 4 or vsc_sh.len != 4) return null;
    if (ksc_sh[3] != @divExact(dk, @as(c_int, @intCast(view.group_size)))) return null;
    if (vsc_sh[3] != @divExact(dv, @as(c_int, @intCast(view.group_size)))) return null;

    const kernel = getQkvVerifyKernel() catch return null;
    const block = qkvVerifyBlockFor(gqa, t_q) orelse return null;
    const nblk: c_int = @divTrunc(t_k + block - 1, block);

    const slot: usize = @intCast(t_q);
    const key = QkvVerCfgKey{
        .hq = h_q,
        .dk = dk,
        .dv = dv,
        .bits = view.bits,
        .gs = view.group_size,
        .gqa = gqa,
        .tq = t_q,
        .nblk = nblk,
        .dtype = qdt,
    };
    if (qkv_ver_cfgs[slot] == null or !std.meta.eql(qkv_ver_cfg_keys[slot], key)) {
        if (qkv_ver_cfgs[slot]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const rows_total: c_int = h_q * t_q;
        const ml_shape = [_]c_int{ rows_total, nblk };
        const o_shape = [_]c_int{ rows_total, nblk, dv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, QKV_DEC_NT, h_kv, nblk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, QKV_DEC_NT, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", qdt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(view.bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(view.group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DK", dk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DV", dv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GQA", gqa));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TQ", t_q));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BLOCK", block));
        qkv_ver_cfgs[slot] = config;
        qkv_ver_cfg_keys[slot] = key;
    }

    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const tk_data = [_]i32{t_k};
    const tk = mlx.mlx_array_new_data(&tk_data, &one, 1, .int32);
    defer _ = mlx.mlx_array_free(tk);

    const inputs_arr = [_]mlx.mlx_array{
        q,
        view.k_triple_q,
        view.k_triple_scales,
        view.k_triple_biases,
        view.v_triple_q,
        view.v_triple_scales,
        view.v_triple_biases,
        scl,
        tk,
    };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, qkv_ver_cfgs[slot].?, s));
    const o_t = try qkvMergePartials(s, outputs_vec, qdt);
    defer _ = mlx.mlx_array_free(o_t);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const out_shape = [_]c_int{ 1, h_q, t_q, dv };
    try mlx.check(mlx.mlx_reshape(&out, o_t, &out_shape, 4, s));
    const bit: u16 = @as(u16, 1) << @intCast(t_q);
    if (qkv_ver_engaged_mask & bit == 0) {
        qkv_ver_engaged_mask |= bit;
        log.info("[kv-attn] verify kernel engaged: Tq={d} bits={d} gs={d} DK={d} DV={d} Hq={d} Hkv={d} block={d} (G17 reads dense)\n", .{ t_q, view.bits, view.group_size, dk, dv, h_q, h_kv, block });
    }
    return out;
}

// ── matmul2d packed attention (t_q 1..8) ──
//
// One threadgroup (256 threads, 8 simdgroups) per (kv head, kv split). Per 32-token page:
// dequantize K into a threadgroup tile, QK through the matmul2d tensor op, an online
// softmax per q row, dequantize V into the same tile, PV accumulated into a cooperative
// running output that is rescaled when a row's max grows. Each split emits the same
// (m, l, unnormalized O) partials as the other packed kernels. Structure after Inco
// Splash's q8 attention tile (Apache-2.0, see NOTICE).
const QKV_MPP_PAGE: c_int = 32;
const QKV_MPP_MAX_SPLITS: c_int = 128;
const QKV_MPP_PAGES_PER_SPLIT: c_int = 8;

/// The QSA score sheet's matmul2d header, shared by the quantized-KV MPP kernel.
pub const QSA_SCORE_KERNEL_HEADER = kernelSource("kernels/attention/qsa_score_header.metal");
pub const QKV_MPP_KERNEL_SOURCE = kernelSource("kernels/attention/qkv_mpp.metal");

var qkv_mpp_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
const QkvMppCfgKey = struct { hq: c_int, dk: c_int, bits: u8, gs: u32, gqa: c_int, tq: c_int, nsplit: c_int, dtype: mlx.mlx_dtype };
var qkv_mpp_cfgs: [@intCast(QKV_VERIFY_MAX_TQ + 1)]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var qkv_mpp_cfg_keys: [@intCast(QKV_VERIFY_MAX_TQ + 1)]QkvMppCfgKey = @splat(std.mem.zeroes(QkvMppCfgKey));
var qkv_mpp_engaged: bool = false;

/// Measured window (M4 Max, 24/4 hd 256 kv8): the matmul2d kernel's per-page fixed cost
/// needs rows or length to amortize.
pub fn qkvMppWins(t_q: c_int, t_k: c_int) bool {
    if (t_q >= 4) return t_k >= 2048;
    if (t_q >= 2) return t_k >= 8192;
    return t_k >= 16384;
}

pub fn qkvMppSplits(t_k: c_int) c_int {
    const pages = @divTrunc(t_k + QKV_MPP_PAGE - 1, QKV_MPP_PAGE);
    return std.math.clamp(@divTrunc(pages, QKV_MPP_PAGES_PER_SPLIT), 1, QKV_MPP_MAX_SPLITS);
}

/// Packed-KV attention through matmul2d, t_q 1..8, no mask or end-aligned causal. Null = declined.
pub fn qkvAttnMppKernel(s: mlx.mlx_stream, q_in: mlx.mlx_array, view: *const DenseKVView, scale: f32, mask_mode: []const u8) !?mlx.mlx_array {
    // Same machines as the verify-qmm matmul2d tile: M5 matrix units or M4-family shader code.
    if (verifyQmmTile() == .off) return null;
    if (!view.has_quant_triple or view.group_size == 0) return null;
    if (view.bits != 4 and view.bits != 8) return null;
    if (mlx.mlx_array_ndim(q_in) != 4) return null;
    const qs = mlx.getShape(q_in);
    const t_q: c_int = qs[2];
    if (qs[0] != 1 or t_q < 1 or t_q > QKV_VERIFY_MAX_TQ) return null;
    const causal = std.mem.eql(u8, mask_mode, "causal");
    if (!causal and !(t_q == 1 and mask_mode.len == 0)) return null;
    const qdt = mlx.mlx_array_dtype(q_in);
    if (qdt != .bfloat16 and qdt != .float16) return null;
    if (mlx.mlx_array_ndim(view.k_triple_q) != 4 or mlx.mlx_array_ndim(view.v_triple_q) != 4) return null;
    const ks = mlx.getShape(view.k_triple_q);
    const vs = mlx.getShape(view.v_triple_q);
    const vpw: c_int = @divExact(@as(c_int, 32), @as(c_int, view.bits));
    const dk: c_int = ks[3] * vpw;
    if (qs[3] != dk or vs[3] != ks[3] or dk != 256) return null;
    const gs: c_int = @intCast(view.group_size);
    if (@rem(dk, gs) != 0 or @divExact(dk, gs) > 8) return null;
    const h_q: c_int = qs[1];
    const h_kv: c_int = ks[1];
    if (h_kv <= 0 or @rem(h_q, h_kv) != 0 or vs[1] != h_kv or vs[2] != ks[2] or ks[0] != 1) return null;
    const gqa: c_int = @divExact(h_q, h_kv);
    if (@divTrunc(gqa * t_q + 7, 8) * 8 > 56) return null; // past 56 padded rows the threadgroup arrays exceed Metal's 32 KiB
    const t_k: c_int = ks[2];
    if (t_k < t_q) return null;

    if (qkv_mpp_kernel_cached == null) {
        const input_names = [_][*:0]const u8{ "q", "kq", "ksc", "kbi", "vq", "vsc", "vbi", "scl", "tk" };
        const output_names = [_][*:0]const u8{ "m_out", "l_out", "o_out" };
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kk = mlx.mlx_fast_metal_kernel_new("msv_qkv_mpp", in_vec, out_vec, QKV_MPP_KERNEL_SOURCE, QSA_SCORE_KERNEL_HEADER, false, false);
        if (kk.ctx == null) return null;
        qkv_mpp_kernel_cached = kk;
    }
    const nsplit = qkvMppSplits(t_k);
    const real_rows: c_int = gqa * t_q;
    const m_rows: c_int = @divTrunc(real_rows + 7, 8) * 8;
    const slot: usize = @intCast(t_q);
    const key = QkvMppCfgKey{ .hq = h_q, .dk = dk, .bits = view.bits, .gs = view.group_size, .gqa = gqa, .tq = t_q, .nsplit = nsplit, .dtype = qdt };
    if (qkv_mpp_cfgs[slot] == null or !std.meta.eql(qkv_mpp_cfg_keys[slot], key)) {
        if (qkv_mpp_cfgs[slot]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const ml_shape = [_]c_int{ h_kv, nsplit, m_rows };
        const o_shape = [_]c_int{ h_kv, nsplit, m_rows, dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256, h_kv, nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", qdt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(view.bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", gs));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DK", dk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GQA", gqa));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TQ", t_q));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "MPAD", m_rows));
        qkv_mpp_cfgs[slot] = config;
        qkv_mpp_cfg_keys[slot] = key;
    }

    // [1, Hq, TQ, D] is already (kv head, group head, row)-major: [Hkv, real_rows, D], zero-padded.
    var q3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q3);
    const q3_shape = [_]c_int{ h_kv, real_rows, dk };
    try mlx.check(mlx.mlx_reshape(&q3, q_in, &q3_shape, 3, s));
    var qp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qp);
    {
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        const axes = [_]c_int{1};
        const lo = [_]c_int{0};
        const hi = [_]c_int{m_rows - real_rows};
        try mlx.check(mlx.mlx_pad(&qp, q3, &axes, 1, &lo, 1, &hi, 1, zero, "constant", s));
    }
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_contiguous(&q, qp, false, s));
    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const tk_data = [_]i32{t_k};
    const tk = mlx.mlx_array_new_data(&tk_data, &one, 1, .int32);
    defer _ = mlx.mlx_array_free(tk);
    const inputs_arr = [_]mlx.mlx_array{ q, view.k_triple_q, view.k_triple_scales, view.k_triple_biases, view.v_triple_q, view.v_triple_scales, view.v_triple_biases, scl, tk };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, qkv_mpp_kernel_cached.?, inputs_vec, qkv_mpp_cfgs[slot].?, s));

    // [Hkv, NS, M(, D)] -> [Hkv*M, NS(, D)] for the shared merge.
    var parts: [3]mlx.mlx_array = undefined;
    var n_parts: usize = 0;
    defer for (parts[0..n_parts]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (0..3) |i| {
        var raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(raw);
        try mlx.check(mlx.mlx_vector_array_get(&raw, outputs_vec, i));
        var sw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sw);
        const ax4 = [_]c_int{ 0, 2, 1, 3 };
        try mlx.check(mlx.mlx_transpose_axes(&sw, raw, &ax4, if (i == 2) 4 else 3, s));
        // Drop the pad rows: [Hkv, M, NS(, D)] -> [Hkv, real_rows, ...].
        var cut = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cut);
        const start = [_]c_int{ 0, 0, 0, 0 };
        const stop = [_]c_int{ h_kv, real_rows, nsplit, dk };
        const strd = [_]c_int{ 1, 1, 1, 1 };
        const nd: usize = if (i == 2) 4 else 3;
        try mlx.check(mlx.mlx_slice(&cut, sw, &start, nd, &stop, nd, &strd, nd, s));
        var flat = mlx.mlx_array_new();
        const sh3 = [_]c_int{ h_kv * real_rows, nsplit, dk };
        try mlx.check(mlx.mlx_reshape(&flat, cut, &sh3, if (i == 2) 3 else 2, s));
        parts[n_parts] = flat;
        n_parts += 1;
    }
    const o_t = try qkvMergeArrays(s, parts[0], parts[1], parts[2], qdt);
    defer _ = mlx.mlx_array_free(o_t);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const out_shape = [_]c_int{ 1, h_q, t_q, dk };
    try mlx.check(mlx.mlx_reshape(&out, o_t, &out_shape, 4, s));
    if (!qkv_mpp_engaged) {
        qkv_mpp_engaged = true;
        log.info("[kv-attn] matmul2d packed attention engaged: bits={d} gs={d} Hq={d} Hkv={d}\n", .{ view.bits, view.group_size, h_q, h_kv });
    }
    return out;
}

// ── Fused quant-attention gate (docs/kv-quant-perf.md Phase 1) ──


/// Per-LAYER kv-length floor: below this the packed read cannot pay for the
/// kernel's split-KV merge ops (~11 dispatches/layer) — Laguna's 512-window
/// sliding layers measured a net −2% when they fused, and gemma4-12B's
/// 1024-window sliding layers sat exactly AT the old 1024 floor and measured
/// a ~4 tok/s decode loss fused (masked kernel at ~1k KV vs near-free dense
/// SDPA). Short-KV layers keep dense reads even inside a fused
/// request (a PERF gate, not a correctness gate).
pub const KV_ATTN_FUSED_MIN_TK: c_int = 2048;

/// The ONE eligibility gate every fused quant-attention call site reads
/// (`forwardStandardWith`, `gatedFullAttnWith`, `lagunaAttnWith`).
/// `forwardBatchedDecode` deliberately keeps dense-mode reads: batched decode
/// is the packed-words-to-SDPA server-kill class (guarded by
/// tests/test_batched_equivalence.sh).
///
/// Decode width (t_q == 1) ONLY: that is the width `qkvAttnDecodeKernel`
/// serves (the measured +10..+56% serial win). Verify widths (T_q 2..32
/// under MTP/spec) used to be admitted and fell to the composed qmm chain —
/// 2 packed matmuls x every attention layer per spec round at qmm's M 2..7
/// dead zone, measured a 2.1x decode LOSS at 8k+. They read
/// the dense view now, like prefill always has. A forced per-request
/// `kv_attn_mode:"fused"` gets the dense read at those widths too.
pub fn kvAttnFusedEligible(view: *const DenseKVView, t_q: c_int) bool {
    if (!(view.has_quant_triple and t_q == 1)) return false;
    const ks = mlx.getShape(view.k_triple_q);
    if (ks.len < 4) return false;
    return ks[2] >= KV_ATTN_FUSED_MIN_TK;
}

/// Verify-width packed-attention gate (Phase 2): T_q 2..8 served by
/// `qkvAttnVerifyKernel`, same triple + per-layer kv floor as decode.
/// Wider multi-token forwards (real prefill chunks) stay on dense SDPA;
/// G17 keeps dense; the test seam forces the kernel or kills the width class.
pub fn kvAttnVerifyEligible(view: *const DenseKVView, t_q: c_int) bool {
    if (!(view.has_quant_triple and t_q >= 2 and t_q <= QKV_VERIFY_MAX_TQ and qkvVerifyEnvAllows())) return false;
    const ks = mlx.getShape(view.k_triple_q);
    if (ks.len < 4) return false;
    return ks[2] >= KV_ATTN_FUSED_MIN_TK;
}

var kv_attn_fused_engaged: bool = false; // one-shot log guard
pub fn logKvAttnFusedEngaged(view: *const DenseKVView, q: mlx.mlx_array, t_q: c_int) void {
    if (kv_attn_fused_engaged) return;
    kv_attn_fused_engaged = true;
    const q_shape = mlx.getShape(q);
    const h_q: c_int = if (q_shape.len >= 4) q_shape[1] else 0;
    const k_shape = mlx.getShape(view.k_triple_q);
    const h_kv: c_int = if (k_shape.len >= 4) k_shape[1] else 0;
    log.info("[kv-attn] fused engaged: bits={d} gs={d} Hq={d} Hkv={d} Tq={d} (--kv-attn-mode dense restores dense reads)\n", .{ view.bits, view.group_size, h_q, h_kv, t_q });
}
