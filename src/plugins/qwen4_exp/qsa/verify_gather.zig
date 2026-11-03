//! QSA attention at verify widths (S 2..15) over gathered rows (`qsaVerifyGatherAttn`) and its floors per KV scheme.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kv_quant = @import("../../../engines/mlx/cache/kv_quant.zig");
const log = @import("../../../sdk/util/log.zig");
const arms = @import("arms.zig");
const sparse_attn = @import("sparse_attn.zig");
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const bf16Scalar = @import("../../../engines/mlx/load/weights.zig").bf16Scalar;
const takeContig = @import("../../../mlx/array.zig").takeContig;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const attn256Cosine = @import("../../../sdk/blocks/attention/attn256.zig").attn256Cosine;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const attn256Reference = @import("../../../sdk/blocks/attention/attn256.zig").attn256Reference;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const splitMaskedSdpa256 = @import("../../../sdk/blocks/attention/sdpa.zig").splitMaskedSdpa256;
const testRandWeightBf16 = @import("../../../sdk/testing/parity.zig").testRandWeightBf16;
const QsaArmCounts = @import("arms.zig").QsaArmCounts;
const QsaDecline = @import("arms.zig").QsaDecline;
const qsaEngagedBit = @import("arms.zig").qsaEngagedBit;
const qsaBatchedGatherFloor = @import("gather.zig").qsaBatchedGatherFloor;
const qsaGatherMinKv = @import("gather.zig").qsaGatherMinKv;
const qsaPrefillGatherMinKv = @import("gather.zig").qsaPrefillGatherMinKv;
const qsaMaskFromBlocks = @import("../../../sdk/blocks/attention/sparse_block/mask.zig").qsaMaskFromBlocks;
const qsaAttnCacheFixture = @import("sparse_attn.zig").qsaAttnCacheFixture;
const qsaSlotAttn = @import("sparse_attn.zig").qsaSlotAttn;
const qsaWantsBlocks = @import("sparse_attn.zig").qsaWantsBlocks;


/// Verify-width QSA gather (2 <= S < FUSED256_MIN_Q_LEN — an MTP/DFlash
/// verify block): read the UNION of the S rows' selections + the union tail
/// instead of the whole cache under a dense [S, kv] mask. With `--kv-quant 8`
/// the dense mask arm also dequantizes the whole stored range per layer per
/// forward; this arm dequantizes only the gathered rows.
/// Test seams: the verify gather on or off, and its KV floor.
pub var qsa_verify_gather_override: ?bool = null;
pub var qsa_verify_gather_min_kv_override: ?c_int = null;

pub fn qsaVerifyGatherEnabled() bool {
    return qsa_verify_gather_override orelse true;
}

/// The union is FIXED-size (S * budget blocks): at the production budget
/// (512 blocks of 4) S=7 is 14336 rows. On dense KV the gather COPIES those
/// rows while the mask arm reads the cache in place, so the copy only pays
/// once the union is well under half the cache (M4 Max, S=7: gather loses 7%
/// at 17k keys, breaks even at 34k). Quantized KV dequantizes only the
/// gathered rows, so its floor stays lower.
pub const QSA_VERIFY_GATHER_MIN_KV_DENSE: c_int = 32768;
pub const QSA_VERIFY_GATHER_MIN_KV_QUANT: c_int = 16384;

/// `quantized` = the arm will dequantize only the gathered rows, i.e. the
/// view carries affine triples (`DenseKVView.has_quant_triple`).
pub fn qsaVerifyGatherMinKvFor(quantized: bool) c_int {
    if (qsa_verify_gather_min_kv_override) |v| return v;
    return if (quantized) QSA_VERIFY_GATHER_MIN_KV_QUANT else QSA_VERIFY_GATHER_MIN_KV_DENSE;
}

/// Geometry of the verify-width QSA gather — pure arithmetic, the ONE owner
/// of R (the gate, the MLX gatherer and the host reference plan all read it).
///
/// The gathered set is two DISJOINT parts:
///   A. the union tail `[tail_start, kv)`, where `tail_start` is row 0's own
///      tail start `ratio * floor((offset+1)/ratio)`. Tail starts are
///      non-decreasing in position and tails are SUFFIXES, so row 0's (the
///      SMALLEST start) is the largest and covers every other row's.
///   B. the union of the rows' selected blocks that lie ENTIRELY below
///      `tail_start` (`b < nb_lo`), deduplicated and padded to a fixed
///      `k_blocks` slots with blocks no row selected. A padded block's tokens
///      are below every row's tail start and in no row's selection, so the
///      per-row mask hides them everywhere — no host sync is needed to learn
///      the real union size.
/// Selected blocks at/above `tail_start` need no slot: their tokens are in A.
pub const QsaVerifyGeom = struct {
    /// Complete blocks in the cache.
    nb: c_int,
    /// Blocks lying entirely below the union tail (`tail_start / ratio`).
    nb_lo: c_int,
    /// Row 0's tail start = the union tail's first token.
    tail_start: c_int,
    tail_len: c_int,
    /// Fixed union slot count (part B).
    k_blocks: c_int,
    /// R — gathered rows total.
    rows: c_int,

    pub fn compute(seq_len: c_int, kv: c_int, ratio: c_int, kb: c_int) ?QsaVerifyGeom {
        if (ratio <= 0 or kb <= 0) return null;
        if (seq_len < 2 or seq_len >= FUSED256_MIN_Q_LEN) return null;
        if (kv <= seq_len) return null;
        const offset = kv - seq_len;
        const nb = @divTrunc(kv, ratio);
        const tail_start = @divTrunc(offset + 1, ratio) * ratio;
        if (tail_start > kv) return null;
        const nb_lo = @divTrunc(tail_start, ratio);
        const k_blocks = @min(nb_lo, seq_len * kb);
        return .{
            .nb = nb,
            .nb_lo = nb_lo,
            .tail_start = tail_start,
            .tail_len = kv - tail_start,
            .k_blocks = k_blocks,
            .rows = k_blocks * ratio + (kv - tail_start),
        };
    }
};

/// Verify-width QSA gather (2 <= S < FUSED256_MIN_Q_LEN, sorted `blocks`):
/// gather the fixed-size union of the S rows' selections + the union tail
/// (see `QsaVerifyGeom`), dequantizing ONLY those rows on a quantized cache,
/// then dense SDPA under the per-row `[S, R]` additive mask. Null = declined
/// (kill switch, kv floor, no sparsity win, dtype, geometry) — the caller
/// falls back to the dense `[S, kv]` mask arm.
///
/// Parity argument: the gathered rows are STRICTLY ASCENDING (part B blocks
/// are all below the tail's first token and the block ids are sorted), so no
/// key can be counted twice; the mask is the dense formula evaluated at the
/// gathered tokens; and every token the dense mask shows a row is present
/// (a token below the union tail that a row sees is in that row's selection,
/// hence in the union, and `k_blocks >= |union|` by construction). Remaining
/// deltas are kernel tiling order.
pub fn qsaVerifyGatherAttn(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array, // [1,Hq,S,D] bf16
    kv_view: *const DenseKVView, // full cache AFTER this block's append
    blocks: mlx.mlx_array, // [1,S,kb] int32 sorted asc, INT_MAX sentinel
    ratio: c_int,
    attn_scale: f32,
) !?mlx.mlx_array {
    var dq = QsaDecline{ .bits = &arms.qsa_verify_decline_bits, .arm = "qsa-verify-gather" };
    if (!qsaVerifyGatherEnabled()) return dq.no(.disabled);
    if (mlx.mlx_array_ndim(q_rope) != 4 or mlx.mlx_array_ndim(blocks) != 3) return dq.no(.geometry);
    if (mlx.mlx_array_dtype(q_rope) != .bfloat16 or mlx.mlx_array_dtype(blocks) != .int32) return dq.no(.dtype);
    const qs = mlx.getShape(q_rope);
    if (qs[0] != 1) return dq.no(.geometry);
    const seq_len: c_int = qs[2];
    dq.s = seq_len;
    const head_dim: c_int = qs[3];
    const h_q: c_int = qs[1];
    if (h_q <= 0 or head_dim <= 0) return dq.no(.geometry);
    const bs = mlx.getShape(blocks);
    if (bs[0] != 1 or bs[1] != seq_len or bs[2] <= 0) return dq.no(.geometry);
    const kb: c_int = bs[2];

    if (kv_view.k.ctx == null or kv_view.v.ctx == null) return dq.no(.geometry);
    const ks = mlx.getShape(kv_view.k);
    if (ks.len != 4 or ks[0] != 1) return dq.no(.geometry);
    const h_kv: c_int = ks[1];
    const kv: c_int = ks[2];
    dq.kv = kv;
    if (h_kv <= 0 or kv <= 0 or @rem(h_q, h_kv) != 0) return dq.no(.geometry);
    if (ks[3] != head_dim) return dq.no(.geometry);
    const vs = mlx.getShape(kv_view.v);
    if (vs.len != 4 or vs[0] != 1 or vs[1] != h_kv or vs[2] != kv or vs[3] != head_dim) return dq.no(.geometry);
    if (kv <= qsaVerifyGatherMinKvFor(kv_view.has_quant_triple)) return dq.no(.kv_floor);

    const geom = QsaVerifyGeom.compute(seq_len, kv, ratio, kb) orelse return dq.no(.width);
    if (geom.k_blocks <= 0 or geom.rows <= 0 or geom.rows >= kv) return dq.no(.no_win);
    const offset: c_int = kv - seq_len;
    const nb = geom.nb;
    const sel_w: c_int = nb + 2; // [.. nb-1] real, `nb` = the always-false
    // membership sink for tokens past the last complete block, `nb+1` the
    // sentinel sink.

    // Per-row block selection [S, nb+2] bool.
    const nb_arr = mlx.mlx_array_new_int(nb);
    defer _ = mlx.mlx_array_free(nb_arr);
    const sink_arr = mlx.mlx_array_new_int(nb + 1);
    defer _ = mlx.mlx_array_free(sink_arr);
    var valid_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid_b);
    try mlx.check(mlx.mlx_less(&valid_b, blocks, nb_arr, s));
    var safe_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(safe_b);
    try mlx.check(mlx.mlx_where(&safe_b, valid_b, blocks, sink_arr, s));
    var falses = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(falses);
    try mlx.check(mlx.mlx_zeros(&falses, &[_]c_int{ 1, seq_len, sel_w }, 3, .bool_, s));
    const true_v = mlx.mlx_array_new_bool(true);
    defer _ = mlx.mlx_array_free(true_v);
    var sel3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel3);
    try mlx.check(mlx.mlx_put_along_axis(&sel3, falses, safe_b, true_v, -1, s));
    var sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel);
    try mlx.check(mlx.mlx_reshape(&sel, sel3, &[_]c_int{ seq_len, sel_w }, 2, s)); // [S, nb+2]

    // Union over rows of the blocks BELOW the union tail, then a fixed-size
    // ascending id list: argsort a key that ranks union members first
    // (`b` vs `nb_lo + b`), keep k_blocks, sort so the rows come out
    // ascending. No host sync — the padding blocks mask out everywhere.
    const strides2 = [_]c_int{ 1, 1 };
    var sel_lo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_lo);
    try mlx.check(mlx.mlx_slice(&sel_lo, sel, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ seq_len, geom.nb_lo }, 2, &strides2, 2, s));
    var union_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(union_b);
    try mlx.check(mlx.mlx_any_axes(&union_b, sel_lo, &[_]c_int{0}, 1, false, s)); // [nb_lo]
    var lo_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lo_idx);
    try mlx.check(mlx.mlx_arange(&lo_idx, 0, @floatFromInt(geom.nb_lo), 1.0, .int32, s));
    const nb_lo_arr = mlx.mlx_array_new_int(geom.nb_lo);
    defer _ = mlx.mlx_array_free(nb_lo_arr);
    var lo_shift = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lo_shift);
    try mlx.check(mlx.mlx_add(&lo_shift, lo_idx, nb_lo_arr, s));
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_where(&key, union_b, lo_idx, lo_shift, s));
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, key, -1, s));
    var order_i32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_i32);
    try mlx.check(mlx.mlx_astype(&order_i32, order, .int32, s));
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try mlx.check(mlx.mlx_slice(&picked, order_i32, &[_]c_int{0}, 1, &[_]c_int{geom.k_blocks}, 1, &[_]c_int{1}, 1, s));
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_sort_axis(&ids, picked, -1, s)); // [K] ascending

    // Token ids: each picked block's `ratio` tokens, then the union tail.
    const ratio_arr = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_arr);
    var ids2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids2);
    try mlx.check(mlx.mlx_reshape(&ids2, ids, &[_]c_int{ geom.k_blocks, 1 }, 2, s));
    var ids_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_scaled);
    try mlx.check(mlx.mlx_multiply(&ids_scaled, ids2, ratio_arr, s));
    var rvec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rvec);
    try mlx.check(mlx.mlx_arange(&rvec, 0, @floatFromInt(ratio), 1.0, .int32, s));
    var r2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r2);
    try mlx.check(mlx.mlx_reshape(&r2, rvec, &[_]c_int{ 1, ratio }, 2, s));
    var tok2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok2d);
    try mlx.check(mlx.mlx_add(&tok2d, ids_scaled, r2, s)); // [K, ratio]
    var sel_tok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_tok);
    try mlx.check(mlx.mlx_reshape(&sel_tok, tok2d, &[_]c_int{geom.k_blocks * ratio}, 1, s));
    var tok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok);
    if (geom.tail_len > 0) {
        var tail_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tail_ids);
        try mlx.check(mlx.mlx_arange(&tail_ids, @floatFromInt(geom.tail_start), @floatFromInt(kv), 1.0, .int32, s));
        const parts = [_]mlx.mlx_array{ sel_tok, tail_ids };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&tok, vec, 0, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&tok, sel_tok));
    }
    var tok_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_u32);
    try mlx.check(mlx.mlx_astype(&tok_u32, tok, .uint32, s));

    // [S, R] visibility: the dense formula evaluated at the gathered tokens.
    var blk_of = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_of);
    try mlx.check(mlx.mlx_floor_divide(&blk_of, tok, ratio_arr, s)); // [R], <= nb
    var blk_of_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_of_u32);
    try mlx.check(mlx.mlx_astype(&blk_of_u32, blk_of, .uint32, s));
    var member = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(member);
    try mlx.check(mlx.mlx_take_axis(&member, sel, blk_of_u32, 1, s)); // [S, R]
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    try mlx.check(mlx.mlx_arange(&pos, @floatFromInt(offset), @floatFromInt(kv), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos, &[_]c_int{ seq_len, 1 }, 2, s));
    const one_i = mlx.mlx_array_new_int(1);
    defer _ = mlx.mlx_array_free(one_i);
    var p1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p1);
    try mlx.check(mlx.mlx_add(&p1, pos2, one_i, s));
    var nb_p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nb_p);
    try mlx.check(mlx.mlx_floor_divide(&nb_p, p1, ratio_arr, s));
    var ts = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ts);
    try mlx.check(mlx.mlx_multiply(&ts, nb_p, ratio_arr, s)); // [S,1]
    var tok_row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_row);
    try mlx.check(mlx.mlx_reshape(&tok_row, tok, &[_]c_int{ 1, geom.rows }, 2, s));
    var tailvis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tailvis);
    try mlx.check(mlx.mlx_greater_equal(&tailvis, tok_row, ts, s));
    var causal = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(causal);
    try mlx.check(mlx.mlx_less_equal(&causal, tok_row, pos2, s));
    var any_vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(any_vis);
    try mlx.check(mlx.mlx_logical_or(&any_vis, member, tailvis, s));
    var vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vis);
    try mlx.check(mlx.mlx_logical_and(&vis, any_vis, causal, s));
    const zero_f = bf16Scalar(0.0, s);
    defer _ = mlx.mlx_array_free(zero_f);
    const neg_inf = bf16Scalar(-std.math.inf(f32), s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var addv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(addv);
    try mlx.check(mlx.mlx_where(&addv, vis, zero_f, neg_inf, s));
    var add4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(add4);
    try mlx.check(mlx.mlx_reshape(&add4, addv, &[_]c_int{ 1, 1, seq_len, geom.rows }, 4, s));

    // Gather rows: quant triples when the view carries affine 4/8-bit (then
    // dequantize ONLY the subset — the whole point at `--kv-quant 8`), else
    // dense rows. Dequantize groups along D, so subset-first is exactly
    // full-first-restricted (pinned by the take-then-dequantize test).
    var gk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gk);
    var gv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gv);
    if (kv_view.has_quant_triple and (kv_view.bits == 4 or kv_view.bits == 8)) {
        const kq = kv_view.k_triple_q;
        const vq = kv_view.v_triple_q;
        if (kq.ctx == null or vq.ctx == null) return dq.no(.quant_layout);
        const kqs = mlx.getShape(kq);
        const vqs = mlx.getShape(vq);
        if (kqs.len != 4 or vqs.len != 4 or kqs[0] != 1 or vqs[0] != 1 or
            kqs[1] != h_kv or vqs[1] != h_kv or kqs[2] != kv or vqs[2] != kv) return dq.no(.quant_layout);
        const gs: u32 = kv_view.group_size;
        if (gs == 0) return dq.no(.quant_layout);
        // Contiguous copies: dequantize misreads the strided take views.
        const gkq = try takeContig(s, kq, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gkq);
        const gks = try takeContig(s, kv_view.k_triple_scales, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gks);
        const gkb = try takeContig(s, kv_view.k_triple_biases, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gkb);
        const gvq = try takeContig(s, vq, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gvq);
        const gvs = try takeContig(s, kv_view.v_triple_scales, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gvs);
        const gvb = try takeContig(s, kv_view.v_triple_biases, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gvb);
        const gkd = try kv_quant.dequantizeAffine(s, gkq, gks, gkb, gs, kv_view.bits);
        defer _ = mlx.mlx_array_free(gkd);
        const gvd = try kv_quant.dequantizeAffine(s, gvq, gvs, gvb, gs, kv_view.bits);
        defer _ = mlx.mlx_array_free(gvd);
        try mlx.check(mlx.mlx_array_set(&gk, gkd));
        try mlx.check(mlx.mlx_array_set(&gv, gvd));
    } else {
        try mlx.check(mlx.mlx_take_axis(&gk, kv_view.k, tok_u32, 2, s));
        try mlx.check(mlx.mlx_take_axis(&gv, kv_view.v, tok_u32, 2, s));
    }
    if (mlx.mlx_array_dtype(gk) != .bfloat16 or mlx.mlx_array_dtype(gv) != .bfloat16) return dq.no(.dtype);
    const gks2 = mlx.getShape(gk);
    const gvs2 = mlx.getShape(gv);
    if (gks2.len != 4 or gvs2.len != 4 or gks2[0] != 1 or gvs2[0] != 1 or
        gks2[1] != h_kv or gvs2[1] != h_kv or gks2[2] != geom.rows or gvs2[2] != geom.rows or
        gks2[3] != head_dim or gvs2[3] != head_dim) return dq.no(.gather_shape);

    // hd 256 + an ARRAY mask has no fused arm in MLX: the same row-group
    // split the dense mask arm uses, now over [R] instead of [kv].
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (try splitMaskedSdpa256(s, q_rope, gk, gv, attn_scale, add4)) |split_out| {
        _ = mlx.mlx_array_free(out);
        out = split_out;
    } else {
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q_rope, gk, gv, attn_scale, "array", add4, .{ .ctx = null }, false, s));
    }
    if (arms.qsa_engaged_bits.take(qsaEngagedBit(.verify_gather, seq_len))) {
        log.info("[qsa-verify-gather] engaged (S={d} kv={d} rows={d})\n", .{ seq_len, kv, geom.rows });
    }
    return out;
}

// ── tests ──

test "qsa verify gather floor: dense KV 32768, quantized 16384, override wins" {
    const prev_o = qsa_verify_gather_min_kv_override;
    defer qsa_verify_gather_min_kv_override = prev_o;
    qsa_verify_gather_min_kv_override = null;
    try std.testing.expectEqual(@as(c_int, 32768), qsaVerifyGatherMinKvFor(false));
    try std.testing.expectEqual(@as(c_int, 16384), qsaVerifyGatherMinKvFor(true));
    qsa_verify_gather_min_kv_override = 4096;
    try std.testing.expectEqual(@as(c_int, 4096), qsaVerifyGatherMinKvFor(false));
    try std.testing.expectEqual(@as(c_int, 4096), qsaVerifyGatherMinKvFor(true));
}

/// Host reference plan for the verify gather: the ascending gathered token
/// ids and the `[S, R]` visibility mask. Executable spec of the MLX arm —
/// `qsaVerifyGatherAttn` builds the same two arrays in ops, and the parity
/// test pins its row ids against this.
pub const QsaVerifyPlan = struct {
    rows: []i32,
    mask: []bool, // [seq_len, rows.len], row-major

    pub fn deinit(self: *QsaVerifyPlan, alloc: std.mem.Allocator) void {
        alloc.free(self.rows);
        alloc.free(self.mask);
        self.rows = &.{};
        self.mask = &.{};
    }
};

pub fn qsaVerifyPlanHost(
    alloc: std.mem.Allocator,
    blocks: []const i32, // [seq_len, kb] ascending, INT_MAX past each row's count
    seq_len: c_int,
    kb: c_int,
    kv: c_int,
    ratio: c_int,
) !QsaVerifyPlan {
    const geom = QsaVerifyGeom.compute(seq_len, kv, ratio, kb) orelse return error.QsaVerifyGeometry;
    const su: usize = @intCast(seq_len);
    const kbu: usize = @intCast(kb);
    if (blocks.len != su * kbu) return error.QsaVerifyBadSelection;
    const ru: usize = @intCast(geom.rows);
    const ku: usize = @intCast(geom.k_blocks);

    const in_union = try alloc.alloc(bool, @intCast(geom.nb_lo));
    defer alloc.free(in_union);
    @memset(in_union, false);
    for (0..su) |i| {
        for (0..kbu) |c| {
            const b = blocks[i * kbu + c];
            if (b >= 0 and b < geom.nb_lo) in_union[@intCast(b)] = true;
        }
    }
    // Union members ascending, then non-union fillers ascending, truncated to
    // the fixed slot count — the `where(u, b, nb_lo + b)` argsort key the MLX
    // arm uses — then sorted so the gathered rows come out ascending.
    const ids = try alloc.alloc(i32, ku);
    defer alloc.free(ids);
    var w: usize = 0;
    for (in_union, 0..) |u, b| {
        if (u and w < ku) {
            ids[w] = @intCast(b);
            w += 1;
        }
    }
    for (in_union, 0..) |u, b| {
        if (!u and w < ku) {
            ids[w] = @intCast(b);
            w += 1;
        }
    }
    std.mem.sort(i32, ids, {}, std.sort.asc(i32));

    const rows = try alloc.alloc(i32, ru);
    errdefer alloc.free(rows);
    var r: usize = 0;
    for (ids) |b| {
        var j: c_int = 0;
        while (j < ratio) : (j += 1) {
            rows[r] = b * ratio + j;
            r += 1;
        }
    }
    var t: c_int = geom.tail_start;
    while (t < kv) : (t += 1) {
        rows[r] = t;
        r += 1;
    }

    const mask = try alloc.alloc(bool, su * ru);
    errdefer alloc.free(mask);
    @memset(mask, false);
    const offset = kv - seq_len;
    for (0..su) |i| {
        const p: c_int = offset + @as(c_int, @intCast(i));
        const ts: c_int = @divTrunc(p + 1, ratio) * ratio;
        for (rows, 0..) |tok, slot| {
            if (tok > p) continue;
            var vis = tok >= ts;
            if (!vis) {
                const b: i32 = @divTrunc(tok, ratio);
                if (b < geom.nb) {
                    for (0..kbu) |c| {
                        if (blocks[i * kbu + c] == b) {
                            vis = true;
                            break;
                        }
                    }
                }
            }
            mask[i * ru + slot] = vis;
        }
    }
    return .{ .rows = rows, .mask = mask };
}

pub fn qsaDecodeGatherIdsHost(
    alloc: std.mem.Allocator,
    blocks: []const i32,
    kb: c_int,
    kv: c_int,
    ratio: c_int,
) ![]i32 {
    const kbu: usize = @intCast(kb);
    if (blocks.len != kbu) return error.QsaDecodeBadSelection;
    if (ratio <= 0 or kb <= 0 or kv <= 0) return error.QsaDecodeGeometry;
    const nb: c_int = @divTrunc(kv, ratio);
    const tail_start: c_int = nb * ratio;
    const tail_len: c_int = kv - tail_start;
    const sel_len: c_int = kb * ratio;
    const gathered_len: c_int = sel_len + tail_len;
    const rows = try alloc.alloc(i32, @intCast(gathered_len));
    var r: usize = 0;
    for (0..kbu) |c| {
        const b = blocks[c];
        const use = b >= 0 and b < nb;
        const base: i32 = if (use) b else 0;
        var j: c_int = 0;
        while (j < ratio) : (j += 1) {
            rows[r] = base * ratio + j;
            r += 1;
        }
    }
    var t: c_int = tail_start;
    while (t < kv) : (t += 1) {
        rows[r] = t;
        r += 1;
    }
    return rows;
}

pub const QsaSlotPlan = struct {
    rows: []i32,
    mask: []bool,

    pub fn deinit(self: *QsaSlotPlan, alloc: std.mem.Allocator) void {
        alloc.free(self.rows);
        if (self.mask.len != 0) alloc.free(self.mask);
        self.rows = &.{};
        self.mask = &.{};
    }
};

pub fn qsaBatchedPlanHost(
    alloc: std.mem.Allocator,
    slot_blocks: []const []const i32,
    seq_len: c_int,
    kb: c_int,
    kv_lens: []const c_int,
    ratio: c_int,
) ![]QsaSlotPlan {
    if (slot_blocks.len != kv_lens.len) return error.QsaBatchedPlanLen;
    const out = try alloc.alloc(QsaSlotPlan, slot_blocks.len);
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |*p| p.deinit(alloc);
        alloc.free(out);
    }
    for (slot_blocks, kv_lens, 0..) |blocks, kv, i| {
        if (seq_len == 1) {
            const rows = try qsaDecodeGatherIdsHost(alloc, blocks, kb, kv, ratio);
            out[i] = .{ .rows = rows, .mask = &.{} };
        } else {
            var plan = try qsaVerifyPlanHost(alloc, blocks, seq_len, kb, kv, ratio);
            out[i] = .{ .rows = plan.rows, .mask = plan.mask };
            plan.rows = &.{};
            plan.mask = &.{};
        }
        built = i + 1;
    }
    return out;
}

// ── Verify-width QSA gather (S = 2 .. FUSED256_MIN_Q_LEN-1) ──

/// Host `[S, kb]` block selection in the shape `qsaSelectBlocks` hands the
/// gatherer: ascending VISIBLE block ids per row, `INT_MAX` past the count.
pub fn qsaVerifyBlocksHost(alloc: std.mem.Allocator, rnd: std.Random, seq_len: c_int, kv: c_int, kb: c_int, ratio: c_int) ![]i32 {
    const su: usize = @intCast(seq_len);
    const kbu: usize = @intCast(kb);
    const nb: c_int = @divTrunc(kv, ratio);
    const out = try alloc.alloc(i32, su * kbu);
    errdefer alloc.free(out);
    const picked = try alloc.alloc(bool, @intCast(nb + 1));
    defer alloc.free(picked);
    for (0..su) |i| {
        const p: c_int = kv - seq_len + @as(c_int, @intCast(i));
        const complete: c_int = @min(@divTrunc(p + 1, ratio), nb);
        const count: c_int = @min(complete, kb);
        @memset(picked, false);
        var n: c_int = 0;
        while (n < count) {
            const b = rnd.uintLessThan(usize, @intCast(complete));
            if (picked[b]) continue;
            picked[b] = true;
            n += 1;
        }
        var w: usize = 0;
        for (0..@intCast(complete)) |b| if (picked[b]) {
            out[i * kbu + w] = @intCast(b);
            w += 1;
        };
        while (w < kbu) : (w += 1) out[i * kbu + w] = std.math.maxInt(i32);
    }
    return out;
}

/// Brute-force `[S, kv]` visibility — the contract `qsaMaskFromBlockSel`
/// implements: the row's selected blocks' tokens ∪ its own incomplete tail
/// (tokens at/after `ratio * floor((p+1)/ratio)`), ∧ causal.
fn qsaVerifyDenseHost(alloc: std.mem.Allocator, blocks: []const i32, seq_len: c_int, kb: c_int, kv: c_int, ratio: c_int) ![]bool {
    const su: usize = @intCast(seq_len);
    const kvu: usize = @intCast(kv);
    const kbu: usize = @intCast(kb);
    const nb: c_int = @divTrunc(kv, ratio);
    const out = try alloc.alloc(bool, su * kvu);
    errdefer alloc.free(out);
    @memset(out, false);
    for (0..su) |i| {
        const p: c_int = kv - seq_len + @as(c_int, @intCast(i));
        const ts: c_int = @divTrunc(p + 1, ratio) * ratio;
        for (0..kvu) |t| {
            const ti: c_int = @intCast(t);
            if (ti > p) continue;
            if (ti >= ts) {
                out[i * kvu + t] = true;
                continue;
            }
            if (ti >= nb * ratio) continue;
            const b: i32 = @intCast(@divTrunc(ti, ratio));
            for (0..kbu) |c| {
                if (blocks[i * kbu + c] == b) {
                    out[i * kvu + t] = true;
                    break;
                }
            }
        }
    }
    return out;
}

test "qsa verify gather plan: gathered rows carry exactly the dense [S, kv] mask, no duplicates" {
    // (a) The pure planner is the executable spec of the MLX gatherer: the
    // fixed-size union of the S rows' selections (dedup'd; padded with
    // non-union blocks that every row masks out) plus the union tail. Every
    // (row, token) the dense mask shows must be reachable through EXACTLY
    // one gathered slot, and nothing else may be — a token gathered twice and
    // visible twice would be double-counted under the softmax.
    const ta = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xbe1f7);
    const rnd = prng.random();
    const cases = [_]struct { s: c_int, kv: c_int, kb: c_int, ratio: c_int }{
        .{ .s = 2, .kv = 4096, .kb = 32, .ratio = 4 },
        .{ .s = 5, .kv = 4096, .kb = 64, .ratio = 4 },
        .{ .s = 7, .kv = 20000, .kb = 128, .ratio = 4 },
        .{ .s = 15, .kv = 20000, .kb = 128, .ratio = 4 },
        .{ .s = 3, .kv = 61, .kb = 3, .ratio = 4 }, // ragged tail (kv % ratio != 0)
        .{ .s = 4, .kv = 64, .kb = 2, .ratio = 4 }, // kv divisible by ratio
        .{ .s = 8, .kv = 257, .kb = 5, .ratio = 8 }, // ratio != 4
    };
    for (cases) |c| {
        const blocks = try qsaVerifyBlocksHost(ta, rnd, c.s, c.kv, c.kb, c.ratio);
        defer ta.free(blocks);
        const dense = try qsaVerifyDenseHost(ta, blocks, c.s, c.kb, c.kv, c.ratio);
        defer ta.free(dense);
        var plan = try qsaVerifyPlanHost(ta, blocks, c.s, c.kb, c.kv, c.ratio);
        defer plan.deinit(ta);

        for (1..plan.rows.len) |r| try std.testing.expect(plan.rows[r] > plan.rows[r - 1]);
        for (plan.rows) |t| try std.testing.expect(t >= 0 and t < c.kv);

        const su: usize = @intCast(c.s);
        const kvu: usize = @intCast(c.kv);
        const slot_of = try ta.alloc(isize, kvu);
        defer ta.free(slot_of);
        @memset(slot_of, -1);
        for (plan.rows, 0..) |t, r| slot_of[@intCast(t)] = @intCast(r);
        for (0..su) |i| {
            for (0..kvu) |t| {
                const want = dense[i * kvu + t];
                const sl = slot_of[t];
                const got = sl >= 0 and plan.mask[i * plan.rows.len + @as(usize, @intCast(sl))];
                if (want != got) {
                    std.debug.print("[qsa-verify-plan] S={d} kv={d} row {d} token {d}: dense={} gathered={}\n", .{ c.s, c.kv, i, t, want, got });
                    return error.VerifyPlanMismatch;
                }
            }
        }
    }
}

test "qsa verify gather geometry: rows, the union tail, and the no-win decline" {
    // The ONE owner of R. Production budget is 512 blocks of 4: at S=7 the
    // union is 14336 rows + tail, so the arm needs kv well past that; at
    // S=15 it is 30720 and only pays past ~31k.
    const g7 = QsaVerifyGeom.compute(7, 65536, 4, 512).?;
    try std.testing.expectEqual(@as(c_int, 3584), g7.k_blocks);
    try std.testing.expectEqual(@as(c_int, 65528), g7.tail_start); // ratio*floor((65529+1)/4)
    try std.testing.expectEqual(@as(c_int, 8), g7.tail_len);
    try std.testing.expectEqual(@as(c_int, 14344), g7.rows);
    try std.testing.expectEqual(@as(c_int, 30736), QsaVerifyGeom.compute(15, 65536, 4, 512).?.rows);
    // 20k at the production budget: S=7 wins, S=15 gathers the whole cache.
    try std.testing.expect(QsaVerifyGeom.compute(7, 20000, 4, 512).?.rows < 20000);
    try std.testing.expectEqual(@as(c_int, 20000), QsaVerifyGeom.compute(15, 20000, 4, 512).?.rows);
    // Decode and prefill widths are not this arm's business.
    try std.testing.expect(QsaVerifyGeom.compute(1, 65536, 4, 512) == null);
    try std.testing.expect(QsaVerifyGeom.compute(FUSED256_MIN_Q_LEN, 65536, 4, 512) == null);
}

test "qsa verify gather: subset SDPA matches masked full SDPA at verify widths (dense + affine-8 cache)" {
    // (b) End-to-end parity of `qsaVerifyGatherAttn` against the arm it
    // replaces: the production dense [S, kv] expansion (`qsaMaskFromBlocks`)
    // over the FULL cache. The quant arm reads a real affine-8 `KVCache`
    // (strided views, capacity > logical) so the subset dequant sees the live
    // layout, and its reference is the same cache's FULL dense view.
    const s = mlx.gpuStream();
    qsa_verify_gather_override = true;
    qsa_verify_gather_min_kv_override = 1024;
    defer qsa_verify_gather_override = null;
    defer qsa_verify_gather_min_kv_override = null;
    var prng = std.Random.DefaultPrng.init(0x7e21f);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const cases = [_]struct { s: c_int, kv: c_int, kb: c_int }{
        .{ .s = 2, .kv = 4096, .kb = 32 },
        .{ .s = 5, .kv = 4096, .kb = 32 },
        .{ .s = 7, .kv = 20000, .kb = 128 },
        .{ .s = 15, .kv = 20000, .kb = 128 },
    };
    for (cases) |c| {
        const q_shape = [_]c_int{ 1, 24, c.s, 256 }; // gqa 12, hd 256
        const kv_shape = [_]c_int{ 1, 2, c.kv, 256 };
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k_dense = try testRandWeightBf16(rnd, &kv_shape, s);
        defer _ = mlx.mlx_array_free(k_dense);
        const v_dense = try testRandWeightBf16(rnd, &kv_shape, s);
        defer _ = mlx.mlx_array_free(v_dense);

        const blocks_host = try qsaVerifyBlocksHost(std.testing.allocator, rnd, c.s, c.kv, c.kb, ratio);
        defer std.testing.allocator.free(blocks_host);
        const bshape = [_]c_int{ 1, c.s, c.kb };
        const blocks = mlx.mlx_array_new_data(blocks_host.ptr, &bshape, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);
        const mask = try qsaMaskFromBlocks(s, blocks, c.kv, ratio);
        defer _ = mlx.mlx_array_free(mask);
        const scale: f32 = 1.0 / 16.0;

        var dview = DenseKVView{ .k = k_dense, .v = v_dense, .owned = false };
        const got = (try qsaVerifyGatherAttn(s, q, &dview, blocks, ratio, scale)) orelse return error.VerifyGatherDeclined;
        defer _ = mlx.mlx_array_free(got);
        const ref = try attn256Reference(q, k_dense, v_dense, scale, "array", mask, s);
        defer _ = mlx.mlx_array_free(ref);
        const d_diff = try attn256MaxDiff(got, ref, s);
        const d_cos = try attn256Cosine(got, ref, s);
        std.debug.print("[qsa-verify-gather] S={d} kv={d} dense: max diff {d:.6} cos {d:.7}\n", .{ c.s, c.kv, d_diff, d_cos });
        try std.testing.expect(d_diff < 0.005);
        try std.testing.expect(d_cos > 0.9999);

        var cache = try KVCache.initWithConfig(std.testing.allocator, 1, kv_quant.KVQuantConfig.affine(8));
        defer cache.deinit();
        var view = try cache.update(0, k_dense, v_dense, s, 0);
        defer view.deinit();
        try std.testing.expect(view.has_quant_triple);
        const gotq = (try qsaVerifyGatherAttn(s, q, &view, blocks, ratio, scale)) orelse return error.VerifyGatherDeclined;
        defer _ = mlx.mlx_array_free(gotq);
        const refq = try attn256Reference(q, view.k, view.v, scale, "array", mask, s);
        defer _ = mlx.mlx_array_free(refq);
        const q_diff = try attn256MaxDiff(gotq, refq, s);
        const q_cos = try attn256Cosine(gotq, refq, s);
        std.debug.print("[qsa-verify-gather] S={d} kv={d} affine-8: max diff {d:.6} cos {d:.7}\n", .{ c.s, c.kv, q_diff, q_cos });
        try std.testing.expect(q_diff < 0.01);
        try std.testing.expect(q_cos > 0.9999);
    }

    // Kill switch and the kv floor both decline back to the dense mask arm.
    const q_shape = [_]c_int{ 1, 24, 5, 256 };
    const kv_shape = [_]c_int{ 1, 2, 4096, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const kd = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(kd);
    const vd = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(vd);
    const bh = try qsaVerifyBlocksHost(std.testing.allocator, rnd, 5, 4096, 32, ratio);
    defer std.testing.allocator.free(bh);
    const bsh = [_]c_int{ 1, 5, 32 };
    const bl = mlx.mlx_array_new_data(bh.ptr, &bsh, 3, .int32);
    defer _ = mlx.mlx_array_free(bl);
    var dv2 = DenseKVView{ .k = kd, .v = vd, .owned = false };
    qsa_verify_gather_override = false;
    try std.testing.expect((try qsaVerifyGatherAttn(s, q, &dv2, bl, ratio, 1.0)) == null);
    qsa_verify_gather_override = true;
    qsa_verify_gather_min_kv_override = 16384;
    try std.testing.expect((try qsaVerifyGatherAttn(s, q, &dv2, bl, ratio, 1.0)) == null);
}

test "qsa route: dense bf16 verify at kv 20000 takes the fused kernel through the forward's own selector and arm chain (S=4, S=6)" {
    // The production route is two calls: the indexer's `qsaWantsBlocks` (blocks or the dense
    // mask) and `qsaSlotAttn` (the arm chain). Below the union gather's 32768 floor a dense
    // cache used to get the mask; the fused kernel must now serve, and
    // the dense-variant seam's false must restore the mask route.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const kv: c_int = 20000;
    const kb: c_int = 512;
    const scale: f32 = 1.0 / 16.0;
    var prng = std.Random.DefaultPrng.init(0x20_7E);
    const rnd = prng.random();
    defer sparse_attn.qsa_attn_dense_override = null;

    const kd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(kd);
    const vd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(vd);
    var cache = try KVCache.init(ta, 1);
    defer cache.deinit();
    var view = try qsaAttnCacheFixture(ta, s, kd, vd, kv, @divTrunc(kv, 2), kv_quant.KVQuantConfig.dense, &cache);
    defer view.deinit();

    for ([_]c_int{ 4, 6 }) |seq| {
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, seq, 256 }, s);
        defer _ = mlx.mlx_array_free(q);
        const bh = try qsaVerifyBlocksHost(ta, rnd, seq, kv, kb, ratio);
        defer ta.free(bh);
        const blocks = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ 1, seq, kb }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);
        const none = mlx.mlx_array{ .ctx = null };

        // New route: blocks, one fused dispatch per layer, the mask never built.
        sparse_attn.qsa_attn_dense_override = null;
        try std.testing.expect(qsaWantsBlocks(1, kv, seq, false));
        var mask_new = mlx.mlx_array{ .ctx = null };
        var arms_new = QsaArmCounts{};
        {
            const w = try qsaSlotAttn(s, q, &view, blocks, &mask_new, &arms_new, seq, ratio, scale);
            _ = mlx.mlx_array_free(w.out);
        }
        const before_new = mlx.op_count.load(.monotonic);
        var fused_out: mlx.mlx_array = .{ .ctx = null };
        defer if (fused_out.ctx != null) {
            _ = mlx.mlx_array_free(fused_out);
        };
        for (0..12) |li| {
            const r = try qsaSlotAttn(s, q, &view, blocks, &mask_new, &arms_new, seq, ratio, scale);
            try std.testing.expectEqual(@as(u8, 1), r.cost_bit);
            if (li == 0) fused_out = r.out else _ = mlx.mlx_array_free(r.out);
        }
        const new_ops = mlx.op_count.load(.monotonic) - before_new;
        try std.testing.expect(mask_new.ctx == null);
        try std.testing.expectEqual(@as(u32, 13), arms_new.verify);
        try std.testing.expectEqual(@as(u32, 0), arms_new.mask);

        // Old route (kill switch): the selector declines blocks, the indexer builds the
        // [S, kv] mask once per forward, every layer runs the mask arm.
        sparse_attn.qsa_attn_dense_override = false;
        try std.testing.expect(!qsaWantsBlocks(1, kv, seq, false));
        var arms_old = QsaArmCounts{};
        const before_old = mlx.op_count.load(.monotonic);
        var mask_old = try qsaMaskFromBlocks(s, blocks, kv, ratio);
        defer _ = mlx.mlx_array_free(mask_old);
        var mask_out: mlx.mlx_array = .{ .ctx = null };
        defer if (mask_out.ctx != null) {
            _ = mlx.mlx_array_free(mask_out);
        };
        for (0..12) |li| {
            const r = try qsaSlotAttn(s, q, &view, none, &mask_old, &arms_old, seq, ratio, scale);
            try std.testing.expectEqual(@as(u8, 16), r.cost_bit);
            if (li == 0) mask_out = r.out else _ = mlx.mlx_array_free(r.out);
        }
        const old_ops = mlx.op_count.load(.monotonic) - before_old;
        try std.testing.expectEqual(@as(u32, 12), arms_old.mask);

        std.debug.print("[qsa-route] dense S={d} kv={d}, 12 layers: fused route {d} ops, mask route {d} ops\n", .{ seq, kv, new_ops, old_ops });
        // Measured on the M5 Ultra: S=4 60 vs 105, S=6 60 vs 141. The mask route's cost grows
        // with S (one split-sdpa group per two rows); the bar is three ops saved per layer.
        try std.testing.expect(old_ops > new_ops);
        try std.testing.expect(old_ops - new_ops >= 3 * 12);
        try std.testing.expect((try attn256Cosine(fused_out, mask_out, s)) > 0.99999);
    }
}

test "qsa route: qsaWantsBlocks admits verify widths wherever the fused kernel serves" {
    const transformer = @import("../../../engines/mlx/transformer.zig");
    defer {
        sparse_attn.qsa_attn_dense_override = null;
        transformer.qsa_attn_kernel_override = null;
        sparse_attn.qsa_attn_min_s_override = null;
        transformer.qsa_gather_override = null;
    }
    sparse_attn.qsa_attn_dense_override = null;
    transformer.qsa_attn_kernel_override = null;
    sparse_attn.qsa_attn_min_s_override = null;
    transformer.qsa_gather_override = null;
    const dense_floor = @max(qsaGatherMinKv(), qsaVerifyGatherMinKvFor(false));
    const quant_floor = @max(qsaGatherMinKv(), qsaVerifyGatherMinKvFor(true));
    // Just past the indexer budget, both KV types, every verify width.
    for ([_]c_int{ 2, 4, 6, 15 }) |sq| {
        for ([_]c_int{ 2052, 9000, 20000, 32768, 40000 }) |kv| {
            try std.testing.expect(qsaWantsBlocks(1, kv, sq, false));
            try std.testing.expect(qsaWantsBlocks(1, kv, sq, true));
        }
        try std.testing.expect(!qsaWantsBlocks(2, 20000, sq, false));
    }
    // Decode and prefill keep their own floor.
    try std.testing.expect(!qsaWantsBlocks(1, qsaGatherMinKv(), 1, false));
    try std.testing.expect(!qsaWantsBlocks(1, qsaPrefillGatherMinKv(), FUSED256_MIN_Q_LEN, false));
    try std.testing.expect(qsaWantsBlocks(1, qsaPrefillGatherMinKv() + 1, FUSED256_MIN_Q_LEN, false));
    // The scheduler's pad estimate follows the same predicate.
    try std.testing.expectEqual(@as(c_int, 0), qsaBatchedGatherFloor(4, false));

    // The dense switch restores the union gather's floor on dense KV only.
    sparse_attn.qsa_attn_dense_override = false;
    try std.testing.expect(!qsaWantsBlocks(1, dense_floor, 4, false));
    try std.testing.expect(qsaWantsBlocks(1, dense_floor + 1, 4, false));
    try std.testing.expect(qsaWantsBlocks(1, 9000, 4, true));
    try std.testing.expectEqual(dense_floor, qsaBatchedGatherFloor(4, false));
    // The kernel switch restores it on both.
    sparse_attn.qsa_attn_dense_override = null;
    transformer.qsa_attn_kernel_override = false;
    try std.testing.expect(!qsaWantsBlocks(1, dense_floor, 4, false));
    try std.testing.expect(!qsaWantsBlocks(1, quant_floor, 4, true));
    try std.testing.expect(qsaWantsBlocks(1, quant_floor + 1, 4, true));
    // A width the fused kernel gives up (MIN_S) falls back to the union-gather floor.
    transformer.qsa_attn_kernel_override = null;
    sparse_attn.qsa_attn_min_s_override = 6;
    try std.testing.expect(!qsaWantsBlocks(1, 9000, 4, false));
    try std.testing.expect(qsaWantsBlocks(1, 9000, 6, false));
    // The gather seam's false still turns every block route off.
    sparse_attn.qsa_attn_min_s_override = null;
    transformer.qsa_gather_override = false;
    try std.testing.expect(!qsaWantsBlocks(1, 40000, 4, false));
}
