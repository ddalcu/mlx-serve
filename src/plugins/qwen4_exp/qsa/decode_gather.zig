//! QSA attention at decode and in a batched tick over the indexer's gathered blocks (`qsaDecodeGatherAttn`,
//! `qsaBatchedGatherAttn`).

const std = @import("std");
const verify_gather = @import("verify_gather.zig");
const qwen4_mod = @import("../qwen4_exp.zig");
const mlx = @import("../../../mlx/ffi.zig");
const kv_quant = @import("../../../engines/mlx/cache/kv_quant.zig");
const log = @import("../../../sdk/util/log.zig");
const arms_mod = @import("arms.zig");
const gather = @import("gather.zig");
const sparse_attn = @import("sparse_attn.zig");
const standin = @import("../../../sdk/diag/standin.zig");
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const bf16Scalar = @import("../../../engines/mlx/load/weights.zig").bf16Scalar;
const Trunk = @import("../trunk.zig").Trunk;
const takeContig = @import("../../../mlx/array.zig").takeContig;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const attn256Reference = @import("../../../sdk/blocks/attention/attn256.zig").attn256Reference;
const fusedSdpa256Masked = @import("../../../sdk/blocks/attention/attn256.zig").fusedSdpa256Masked;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const splitMaskedSdpa256 = @import("../../../sdk/blocks/attention/sdpa.zig").splitMaskedSdpa256;
const qwen4Standin = @import("../../../sdk/diag/standin.zig").qwen4Standin;
const maxAbsDiffF32 = @import("../../../sdk/testing/parity.zig").maxAbsDiffF32;
const testRandWeightBf16 = @import("../../../sdk/testing/parity.zig").testRandWeightBf16;
const QsaArm = @import("arms.zig").QsaArm;
const QsaDecline = @import("arms.zig").QsaDecline;
const qsaEngagedBit = @import("arms.zig").qsaEngagedBit;
const qsaWidthBucket = @import("../../../sdk/blocks/attention/sparse_block/one_shot.zig").widthBucket;
const QsaBlockFixture = @import("gather.zig").QsaBlockFixture;
const qsaBatchedGatherEnabled = @import("gather.zig").qsaBatchedGatherEnabled;
const qsaBatchedGatherOn = @import("gather.zig").qsaBatchedGatherOn;
const qsaDecodeGatherEnabled = @import("gather.zig").qsaDecodeGatherEnabled;
const qsaMaskFromBlocks = @import("../../../sdk/blocks/attention/sparse_block/mask.zig").qsaMaskFromBlocks;
const qsaAttnBatchedMinS = @import("sparse_attn.zig").qsaAttnBatchedMinS;
const qsaAttnCacheFixture = @import("sparse_attn.zig").qsaAttnCacheFixture;
const qsaAttnCfgBuilds = @import("sparse_attn.zig").qsaAttnCfgBuilds;
const qsaAttnMinS = @import("sparse_attn.zig").qsaAttnMinS;
const qsaSparseAttnFloor = @import("sparse_attn.zig").qsaSparseAttnFloor;
const qsaBatchedPlanHost = @import("verify_gather.zig").qsaBatchedPlanHost;
const qsaDecodeGatherIdsHost = @import("verify_gather.zig").qsaDecodeGatherIdsHost;
const qsaVerifyBlocksHost = @import("verify_gather.zig").qsaVerifyBlocksHost;
const qsaVerifyGatherAttn = @import("verify_gather.zig").qsaVerifyGatherAttn;
const qsaVerifyPlanHost = @import("verify_gather.zig").qsaVerifyPlanHost;

/// Decode-width QSA gather (S==1, sorted `blocks`, no mask): expand the
/// selection + incomplete tail into ascending token indices, gather those rows
/// (quant triples when the view carries them — dequantizing ONLY the subset —
/// else dense rows), then dense SDPA under an additive validity mask. Null =
/// declined (kill switch, dtype, geometry) — the caller keeps the dense mask.
///
/// Parity argument: the mask arm attends exactly the selected complete blocks'
/// tokens ∪ the incomplete tail ∧ causal. At decode the query is the last
/// cache row, so every gathered token is causal-visible and the ascending
/// gather order matches the masked row's summation order; the masked-out slots
/// contributed exact zeros there. Remaining deltas are kernel tiling order.
pub fn qsaDecodeGatherAttn(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array, // [1,Hq,1,D] bf16
    kv_view: *const DenseKVView, // full cache AFTER this token's append
    blocks: mlx.mlx_array, // [1,1,kb] int32 sorted asc, INT_MAX sentinel
    ratio: c_int,
    attn_scale: f32,
) !?mlx.mlx_array {
    var dq = QsaDecline{ .bits = &arms_mod.qsa_decode_decline_bits, .arm = "qsa-decode-gather" };
    if (!qsaDecodeGatherEnabled()) return dq.no(.disabled);
    if (mlx.mlx_array_ndim(q_rope) != 4 or mlx.mlx_array_ndim(blocks) != 3) return dq.no(.geometry);
    const qs = mlx.getShape(q_rope);
    dq.s = qs[2];
    if (qs[0] != 1) return dq.no(.geometry);
    if (qs[2] != 1) return dq.no(.width);
    if (mlx.mlx_array_dtype(q_rope) != .bfloat16) return dq.no(.dtype);
    if (mlx.mlx_array_dtype(blocks) != .int32) return dq.no(.dtype);
    const bs = mlx.getShape(blocks);
    if (bs[0] != 1 or bs[1] != 1) return dq.no(.geometry);
    const kb: c_int = bs[2];
    if (kb <= 0 or ratio <= 0) return dq.no(.geometry);
    const h_q: c_int = qs[1];
    const head_dim: c_int = qs[3];
    if (h_q <= 0 or head_dim <= 0) return dq.no(.geometry);

    // Cache geometry off the dense K (present in every scheme; shapes are
    // known without executing the lazy dequant graph).
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

    const nb: c_int = @divTrunc(kv, ratio);
    const tail_start: c_int = nb * ratio;
    const tail_len: c_int = kv - tail_start;
    if (tail_len < 0) return dq.no(.geometry);
    const sel_len: c_int = kb * ratio;
    const gathered_len: c_int = sel_len + tail_len;
    // No sparsity win (or degenerate): keep the mask arm.
    if (gathered_len <= 0 or gathered_len >= kv) return dq.no(.no_win);

    // Token ids: selected blocks' ratio tokens each (sentinel → 0, masked
    // below) then the incomplete tail. Reachable decode rows have no
    // sentinels — every complete block is visible from the last position, so
    // visible == nb >= kb — and this stays ascending; the mask is the
    // backstop, not the plan.
    const nb_arr = mlx.mlx_array_new_int(nb);
    defer _ = mlx.mlx_array_free(nb_arr);
    var valid_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid_b);
    try mlx.check(mlx.mlx_less(&valid_b, blocks, nb_arr, s)); // [1,1,kb] bool
    const zero_i = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero_i);
    var safe_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(safe_b);
    try mlx.check(mlx.mlx_where(&safe_b, valid_b, blocks, zero_i, s));
    var blk4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk4);
    try mlx.check(mlx.mlx_expand_dims(&blk4, safe_b, -1, s)); // [1,1,kb,1]
    var rvec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rvec);
    try mlx.check(mlx.mlx_arange(&rvec, 0, @floatFromInt(ratio), 1.0, .int32, s)); // [ratio]
    var r4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r4);
    try mlx.check(mlx.mlx_reshape(&r4, rvec, &[_]c_int{ 1, 1, 1, ratio }, 4, s));
    const ratio_arr = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_arr);
    var blk_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_scaled);
    try mlx.check(mlx.mlx_multiply(&blk_scaled, blk4, ratio_arr, s));
    var tok4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok4);
    try mlx.check(mlx.mlx_add(&tok4, blk_scaled, r4, s)); // [1,1,kb,ratio]
    var tok_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_flat);
    try mlx.check(mlx.mlx_reshape(&tok_flat, tok4, &[_]c_int{sel_len}, 1, s));
    var valid4e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid4e);
    try mlx.check(mlx.mlx_expand_dims(&valid4e, valid_b, -1, s)); // [1,1,kb,1]
    var valid4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid4);
    try mlx.check(mlx.mlx_broadcast_to(&valid4, valid4e, &[_]c_int{ 1, 1, kb, ratio }, 4, s));
    var valid_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid_flat);
    try mlx.check(mlx.mlx_reshape(&valid_flat, valid4, &[_]c_int{sel_len}, 1, s));
    var tail_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_ids);
    var tail_ok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_ok);
    if (tail_len > 0) {
        try mlx.check(mlx.mlx_arange(&tail_ids, @floatFromInt(tail_start), @floatFromInt(kv), 1.0, .int32, s));
        try mlx.check(mlx.mlx_ones(&tail_ok, &[_]c_int{tail_len}, 1, .bool_, s));
    } else {
        const eshp = [_]c_int{0};
        try mlx.check(mlx.mlx_zeros(&tail_ids, &eshp, 1, .int32, s));
        try mlx.check(mlx.mlx_zeros(&tail_ok, &eshp, 1, .bool_, s));
    }
    const idx_parts = [_]mlx.mlx_array{ tok_flat, tail_ids };
    const idx_vec = mlx.mlx_vector_array_new_data(&idx_parts, 2);
    defer _ = mlx.mlx_vector_array_free(idx_vec);
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_concatenate_axis(&idx, idx_vec, 0, s)); // [K'] int32
    const vok_parts = [_]mlx.mlx_array{ valid_flat, tail_ok };
    const vok_vec = mlx.mlx_vector_array_new_data(&vok_parts, 2);
    defer _ = mlx.mlx_vector_array_free(vok_vec);
    var vok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vok);
    try mlx.check(mlx.mlx_concatenate_axis(&vok, vok_vec, 0, s)); // [K'] bool
    // take_axis wants unsigned indices; every id is >= 0 by construction.
    var idx_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx_u32);
    try mlx.check(mlx.mlx_astype(&idx_u32, idx, .uint32, s));

    // Additive validity mask, same bf16 convention as the sliding masks.
    const zero_f = bf16Scalar(0.0, s);
    defer _ = mlx.mlx_array_free(zero_f);
    const neg_inf = bf16Scalar(-std.math.inf(f32), s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var addv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(addv);
    try mlx.check(mlx.mlx_where(&addv, vok, zero_f, neg_inf, s));
    var add4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(add4);
    try mlx.check(mlx.mlx_reshape(&add4, addv, &[_]c_int{ 1, 1, 1, gathered_len }, 4, s));

    // Gather rows: quant triples when the view carries affine 4/8-bit (then
    // dequantize ONLY the subset), else dense rows. Dequantize is per-token
    // along D, so subset-first is exactly full-first-restricted.
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
        // Contiguous copies: dequantize misreads the strided take views.
        const gkq = try takeContig(s, kq, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gkq);
        const gks = try takeContig(s, kv_view.k_triple_scales, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gks);
        const gkb = try takeContig(s, kv_view.k_triple_biases, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gkb);
        const gvq = try takeContig(s, vq, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gvq);
        const gvs = try takeContig(s, kv_view.v_triple_scales, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gvs);
        const gvb = try takeContig(s, kv_view.v_triple_biases, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gvb);
        const gs: u32 = kv_view.group_size;
        if (gs == 0) return dq.no(.quant_layout);
        const gkd = try kv_quant.dequantizeAffine(s, gkq, gks, gkb, gs, kv_view.bits);
        defer _ = mlx.mlx_array_free(gkd);
        const gvd = try kv_quant.dequantizeAffine(s, gvq, gvs, gvb, gs, kv_view.bits);
        defer _ = mlx.mlx_array_free(gvd);
        try mlx.check(mlx.mlx_array_set(&gk, gkd));
        try mlx.check(mlx.mlx_array_set(&gv, gvd));
    } else {
        if (kv_view.k.ctx == null or kv_view.v.ctx == null) return dq.no(.geometry);
        try mlx.check(mlx.mlx_take_axis(&gk, kv_view.k, idx_u32, 2, s));
        try mlx.check(mlx.mlx_take_axis(&gv, kv_view.v, idx_u32, 2, s));
    }
    if (mlx.mlx_array_dtype(gk) != mlx.mlx_array_dtype(q_rope)) return dq.no(.dtype);
    const gks2 = mlx.getShape(gk);
    const gvs2 = mlx.getShape(gv);
    if (gks2.len != 4 or gvs2.len != 4 or gks2[0] != 1 or gvs2[0] != 1 or
        gks2[1] != h_kv or gvs2[1] != h_kv or gks2[2] != gathered_len or gvs2[2] != gathered_len or
        gks2[3] != head_dim or gvs2[3] != head_dim) return dq.no(.gather_shape);

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q_rope, gk, gv, attn_scale, "array", add4, .{ .ctx = null }, false, s));
    if (arms_mod.qsa_engaged_bits.take(qsaEngagedBit(.decode_gather, qs[2]))) {
        log.info("[qsa-decode-gather] engaged (S={d} kv={d} rows={d})\n", .{ qs[2], kv, gathered_len });
    }
    return out;
}

pub fn qsaBatchedGatherAttn(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array,
    views: []const DenseKVView,
    blocks: []const mlx.mlx_array,
    masks: []const mlx.mlx_array,
    ratio: c_int,
    attn_scale: f32,
    out_arms: []QsaArm,
) !mlx.mlx_array {
    const qs = mlx.getShape(q_rope);
    const n: usize = @intCast(qs[0]);
    const seq_len: c_int = qs[2];
    const h_count: c_int = qs[1];
    const hd: c_int = qs[3];
    if (views.len != n or blocks.len != n) return error.QsaBatchedGatherLen;
    if (masks.len != 0 and masks.len != n) return error.QsaBatchedGatherLen;
    if (out_arms.len != 0 and out_arms.len != n) return error.QsaBatchedGatherLen;
    const fused_min_s = if (n >= 2 and seq_len == 1) qsaAttnBatchedMinS() else qsaAttnMinS();
    const standin_sdpa = qwen4Standin().attn_sdpa;
    // S=1 slots on block lists share one fused launch; a decline falls through to the per-slot arms.
    const all_blocks = for (blocks, 0..) |b, i| {
        if (b.ctx == null or (i < masks.len and masks[i].ctx != null)) break false;
    } else true;
    if (n >= 2 and seq_len == 1 and all_blocks and !standin_sdpa) {
        if (try qsaSparseAttnFloor(s, q_rope, views, blocks, ratio, attn_scale, fused_min_s)) |out| {
            for (out_arms) |*a| a.* = .decode_gather;
            return out;
        }
    }
    var outs = try allocator.alloc(mlx.mlx_array, n);
    defer allocator.free(outs);
    var built: usize = 0;
    errdefer {
        var fi: usize = 0;
        while (fi < built) : (fi += 1) _ = mlx.mlx_array_free(outs[fi]);
    }
    for (views, blocks, 0..) |*kv_view, blk, i| {
        const slot_mask: mlx.mlx_array = if (i < masks.len) masks[i] else .{ .ctx = null };
        const i_c: c_int = @intCast(i);
        var q_slot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_slot);
        try mlx.check(mlx.mlx_slice(
            &q_slot,
            q_rope,
            &[_]c_int{ i_c, 0, 0, 0 },
            4,
            &[_]c_int{ i_c + 1, h_count, seq_len, hd },
            4,
            &[_]c_int{ 1, 1, 1, 1 },
            4,
            s,
        ));
        var gathered: ?mlx.mlx_array = null;
        const qsa_ok = blk.ctx != null and slot_mask.ctx == null and !standin_sdpa;
        if (qsa_ok and seq_len >= fused_min_s and seq_len < FUSED256_MIN_Q_LEN) {
            gathered = try qsaSparseAttnFloor(s, q_slot, kv_view[0..1], &.{blk}, ratio, attn_scale, fused_min_s);
        }
        if (gathered == null and qsa_ok and seq_len == 1) {
            gathered = try qsaDecodeGatherAttn(s, q_slot, kv_view, blk, ratio, attn_scale);
        }
        if (gathered == null and qsa_ok and seq_len >= 2 and seq_len < FUSED256_MIN_Q_LEN) {
            gathered = try qsaVerifyGatherAttn(s, q_slot, kv_view, blk, ratio, attn_scale);
        }
        if (gathered) |g| {
            outs[i] = g;
            built = i + 1;
            if (out_arms.len == n) out_arms[i] = switch (qsaWidthBucket(seq_len)) {
                0 => .decode_gather,
                1 => .verify_gather,
                else => .prefill_gather,
            };
            continue;
        }
        var mask_tmp: mlx.mlx_array = .{ .ctx = null };
        defer if (mask_tmp.ctx != null) {
            _ = mlx.mlx_array_free(mask_tmp);
        };
        const mask: mlx.mlx_array = if (slot_mask.ctx != null) slot_mask else if (blk.ctx != null) blk: {
            mask_tmp = try qsaMaskFromBlocks(s, blk, mlx.getShape(kv_view.k)[2], ratio);
            break :blk mask_tmp;
        } else .{ .ctx = null };
        outs[i] = mlx.mlx_array_new();
        built = i + 1;
        if (out_arms.len == n) out_arms[i] = .mask;
        if (mask.ctx != null) {
            if (try fusedSdpa256Masked(s, q_slot, kv_view.k, kv_view.v, attn_scale, mask)) |fused| {
                _ = mlx.mlx_array_free(outs[i]);
                outs[i] = fused;
            } else if (try splitMaskedSdpa256(s, q_slot, kv_view.k, kv_view.v, attn_scale, mask)) |split_out| {
                _ = mlx.mlx_array_free(outs[i]);
                outs[i] = split_out;
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&outs[i], q_slot, kv_view.k, kv_view.v, attn_scale, "array", mask, .{ .ctx = null }, false, s));
            }
        } else {
            const none = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(none);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&outs[i], q_slot, kv_view.k, kv_view.v, attn_scale, "causal", none, .{ .ctx = null }, false, s));
        }
    }
    const vec = mlx.mlx_vector_array_new_data(outs.ptr, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var stacked = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, vec, 0, s));
    var fi: usize = 0;
    while (fi < built) : (fi += 1) _ = mlx.mlx_array_free(outs[fi]);
    return stacked;
}

pub fn qsaBatchedAttn(
    self: *Trunk,
    ctx: *ForwardCtx,
    slots: []const *ForwardCtx,
    q_rope: mlx.mlx_array,
    layer: u32,
    seq_len: c_int,
    attn_scale: f32,
) !?mlx.mlx_array {
    if (!qsaBatchedGatherEnabled()) return null;
    if (self.qwen4 == null) return null;
    if (ctx.qsa_mask.ctx != null) return null;
    var any_blocks = false;
    for (slots) |sc| if (sc.qsa_blocks.ctx != null) {
        any_blocks = true;
    };
    if (!any_blocks) return null;
    const ratio: c_int = @intCast(self.config.indexer_compress_ratio);
    var views = try self.allocator.alloc(DenseKVView, slots.len);
    defer {
        for (views) |*dv| dv.deinit();
        self.allocator.free(views);
    }
    var blks = try self.allocator.alloc(mlx.mlx_array, slots.len);
    defer self.allocator.free(blks);
    var slot_masks = try self.allocator.alloc(mlx.mlx_array, slots.len);
    defer self.allocator.free(slot_masks);
    const arms = try self.allocator.alloc(QsaArm, slots.len);
    defer self.allocator.free(arms);
    for (slots, 0..) |slot_ctx, i| {
        views[i] = try slot_ctx.cache.denseView(layer, self.s);
        blks[i] = slot_ctx.qsa_blocks;
        slot_masks[i] = slot_ctx.qsa_mask;
    }
    if (!gather.qsa_batched_gather_logged) {
        gather.qsa_batched_gather_logged = true;
        log.info("[qsa-batched-gather] engaged (slots={d} S={d})\n", .{ slots.len, seq_len });
    }
    const stacked = try qsaBatchedGatherAttn(self.allocator, self.s, q_rope, views, blks, slot_masks, ratio, attn_scale, arms);
    for (slots, arms) |slot_ctx, arm| {
        slot_ctx.qsa_arms.note(arm);
        if (self.cost_trace_active) self.cost_attention |= switch (arm) {
            .mask => @as(u8, 16),
            .decode_gather => 2,
            .verify_gather => 4,
            .prefill_gather => 8,
        };
    }
    return stacked;
}

// ── tests ──

test "qsa decode gather: take-then-dequantize equals dequantize-then-take" {
    // The subset-dequant primitive the decode gatherer is built on: per-token
    // groups along D make the T axis independent, so gathering rows before or
    // after dequantization must agree bitwise. Production-like sizes (T=256,
    // take 64 scattered rows).
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xdec0de);
    const rnd = prng.random();
    const dense = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 256, 64 }, s);
    defer _ = mlx.mlx_array_free(dense);
    var qq = try kv_quant.quantizeAffine(s, dense, 32, 4);
    defer qq.deinit();
    const n_take = 64;
    const idx_host = blk: {
        var tmp: [n_take]i32 = undefined;
        var i: usize = 0;
        while (i < n_take) : (i += 1) tmp[i] = @intCast((i * 37 + 11) % 256);
        break :blk tmp;
    };
    const idx_sh = [_]c_int{n_take};
    const idx_i32 = mlx.mlx_array_new_data(&idx_host, &idx_sh, 1, .int32);
    defer _ = mlx.mlx_array_free(idx_i32);
    // take_axis wants unsigned indices; every id is >= 0 by construction.
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_astype(&idx, idx_i32, .uint32, s));
    // take-then-dequantize, through the same contiguous helper the decode
    // gatherer uses (dequantize misreads strided take views).
    const tq = try takeContig(s, qq.q, idx, 2);
    defer _ = mlx.mlx_array_free(tq);
    const ts = try takeContig(s, qq.scales, idx, 2);
    defer _ = mlx.mlx_array_free(ts);
    const tb = try takeContig(s, qq.biases, idx, 2);
    defer _ = mlx.mlx_array_free(tb);
    // Exact check of the taken packed words against the source rows they
    // name (u32 — float32 cannot distinguish large packed words; raw pointer
    // walks go through mlx_contiguous first since a take result need not lay
    // out densely). This pins the +0-materializer footgun below: an add of a
    // strided take view reads back wrong, so takeContig uses mlx_contiguous.
    {
        var takenc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(takenc);
        try mlx.check(mlx.mlx_contiguous(&takenc, tq, false, s));
        const evx = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(evx);
        _ = mlx.mlx_vector_array_append_value(evx, takenc);
        _ = mlx.mlx_vector_array_append_value(evx, qq.q);
        try mlx.check(mlx.mlx_eval(evx));
        const td = mlx.mlx_array_data_uint32(takenc) orelse return error.TestUnexpectedNullData;
        const qd = mlx.mlx_array_data_uint32(qq.q) orelse return error.TestUnexpectedNullData;
        var bad: usize = 0;
        for (0..2) |h| {
            for (0..n_take) |i| {
                const src_row: usize = @intCast(idx_host[i]);
                for (0..8) |w| {
                    if (td[(h * n_take + i) * 8 + w] != qd[(h * 256 + src_row) * 8 + w]) bad += 1;
                }
            }
        }
        try std.testing.expectEqual(@as(usize, 0), bad);
    }
    const sub = try kv_quant.dequantizeAffine(s, tq, ts, tb, 32, 4);
    defer _ = mlx.mlx_array_free(sub);
    const full = try kv_quant.dequantizeAffine(s, qq.q, qq.scales, qq.biases, 32, 4);
    defer _ = mlx.mlx_array_free(full);
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_take_axis(&want, full, idx, 2, s));
    // Compare through explicit contiguous copies on BOTH sides: the shared
    // comparator's +0 materializer may itself misread the strided take view.
    var subc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(subc);
    try mlx.check(mlx.mlx_contiguous(&subc, sub, false, s));
    var wantc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wantc);
    try mlx.check(mlx.mlx_contiguous(&wantc, want, false, s));
    const diff = try maxAbsDiffF32(subc, wantc, s);
    std.debug.print("[qsa-take-dequant] max abs diff {d}\n", .{diff});
    try std.testing.expectEqual(@as(f32, 0), diff);
}

test "qsa decode gather: subset SDPA matches masked full SDPA at decode width" {
    const transformer = @import("../../../engines/mlx/transformer.zig");
    // End-to-end parity for `qsaDecodeGatherAttn` on synthetic geometry:
    // same selection + tail, ascending order, causal-visible by construction.
    // Remaining deltas are kernel tiling order only.
    const s = mlx.gpuStream();
    transformer.qsa_decode_gather_override = true;
    defer transformer.qsa_decode_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0x60d31);
    const rnd = prng.random();
    const kL: c_int = 50;
    const kb: c_int = 3;
    const ratio: c_int = 4;
    const q_shape = [_]c_int{ 1, 4, 1, 64 };
    const kv_shape = [_]c_int{ 1, 2, kL, 64 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try testRandWeightBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try testRandWeightBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v_dense);
    var fx = try QsaBlockFixture.build(rnd, 1, kL, kb, ratio);
    defer fx.deinit();
    const scale: f32 = 1.0 / 8.0;

    // Quant path: the view carries affine 4-bit triples AND dense arrays.
    var kq = try kv_quant.quantizeAffine(s, k_dense, 32, 4);
    defer kq.deinit();
    var vq = try kv_quant.quantizeAffine(s, v_dense, 32, 4);
    defer vq.deinit();
    var view = DenseKVView{
        .k = k_dense,
        .v = v_dense,
        .owned = false,
        .k_triple_q = kq.q,
        .k_triple_scales = kq.scales,
        .k_triple_biases = kq.biases,
        .v_triple_q = vq.q,
        .v_triple_scales = vq.scales,
        .v_triple_biases = vq.biases,
        .has_quant_triple = true,
        .bits = 4,
        .group_size = 32,
    };
    const got = (try qsaDecodeGatherAttn(s, q, &view, fx.blocks, ratio, scale)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(got);
    // Reference: masked SDPA over the dense dequantized full KV (what the
    // mask arm computes, up to its own kernel tiling).
    const dk_full = try kv_quant.dequantizeAffine(s, kq.q, kq.scales, kq.biases, 32, 4);
    defer _ = mlx.mlx_array_free(dk_full);
    const dv_full = try kv_quant.dequantizeAffine(s, vq.q, vq.scales, vq.biases, 32, 4);
    defer _ = mlx.mlx_array_free(dv_full);
    const ref = try attn256Reference(q, dk_full, dv_full, scale, "array", fx.mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const diff = try attn256MaxDiff(got, ref, s);
    std.debug.print("[qsa-decode-gather] quant path max diff {d:.6}\n", .{diff});
    try std.testing.expect(diff < 0.01);

    // Dense path (no triple): same answer off the dense rows.
    var dview = DenseKVView{ .k = k_dense, .v = v_dense, .owned = false };
    const got_dense = (try qsaDecodeGatherAttn(s, q, &dview, fx.blocks, ratio, scale)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(got_dense);
    const ref_dense = try attn256Reference(q, k_dense, v_dense, scale, "array", fx.mask, s);
    defer _ = mlx.mlx_array_free(ref_dense);
    const diff_dense = try attn256MaxDiff(got_dense, ref_dense, s);
    std.debug.print("[qsa-decode-gather] dense path max diff {d:.6}\n", .{diff_dense});
    try std.testing.expect(diff_dense < 0.01);

    // Kill switch + wrong-dtype blocks decline to the mask arm.
    transformer.qsa_decode_gather_override = false;
    try std.testing.expect((try qsaDecodeGatherAttn(s, q, &view, fx.blocks, ratio, scale)) == null);
}

test "qsa batched gather S=1: N=1 is byte-identical to solo, N=2 matches per-slot solos" {
    const transformer = @import("../../../engines/mlx/transformer.zig");
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    transformer.qsa_decode_gather_override = true;
    defer transformer.qsa_decode_gather_override = null;
    transformer.qsa_batched_gather_override = true;
    defer transformer.qsa_batched_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0xb07a);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const kb: c_int = 3;
    const hq: c_int = 4;
    const hkv: c_int = 2;
    const hd: c_int = 64;
    const scale: f32 = 1.0 / 8.0;

    const kL0: c_int = 50;
    const kL1: c_int = 36;
    const q0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, 1, hd }, s);
    defer _ = mlx.mlx_array_free(q0);
    const q1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, 1, hd }, s);
    defer _ = mlx.mlx_array_free(q1);
    const k0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL0, hd }, s);
    defer _ = mlx.mlx_array_free(k0);
    const v0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL0, hd }, s);
    defer _ = mlx.mlx_array_free(v0);
    const k1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL1, hd }, s);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL1, hd }, s);
    defer _ = mlx.mlx_array_free(v1);
    var fx0 = try QsaBlockFixture.build(rnd, 1, kL0, kb, ratio);
    defer fx0.deinit();
    var fx1 = try QsaBlockFixture.build(rnd, 1, kL1, kb, ratio);
    defer fx1.deinit();
    const view0 = DenseKVView{ .k = k0, .v = v0, .owned = false };
    const view1 = DenseKVView{ .k = k1, .v = v1, .owned = false };

    const solo0 = (try qsaDecodeGatherAttn(s, q0, &view0, fx0.blocks, ratio, scale)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(solo0);
    const bat1 = try qsaBatchedGatherAttn(ta, s, q0, &.{view0}, &.{fx0.blocks}, &.{}, ratio, scale, &.{});
    defer _ = mlx.mlx_array_free(bat1);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(bat1, solo0, s));

    var q_n2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_n2);
    const qparts = [_]mlx.mlx_array{ q0, q1 };
    const qvec = mlx.mlx_vector_array_new_data(&qparts, 2);
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_concatenate_axis(&q_n2, qvec, 0, s));
    const views = [_]DenseKVView{ view0, view1 };
    const blks = [_]mlx.mlx_array{ fx0.blocks, fx1.blocks };
    const bat2 = try qsaBatchedGatherAttn(ta, s, q_n2, &views, &blks, &.{}, ratio, scale, &.{});
    defer _ = mlx.mlx_array_free(bat2);
    const solo1 = (try qsaDecodeGatherAttn(s, q1, &view1, fx1.blocks, ratio, scale)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(solo1);
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    const wparts = [_]mlx.mlx_array{ solo0, solo1 };
    const wvec = mlx.mlx_vector_array_new_data(&wparts, 2);
    defer _ = mlx.mlx_vector_array_free(wvec);
    try mlx.check(mlx.mlx_concatenate_axis(&want, wvec, 0, s));
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(bat2, want, s));
}

test "qsa batched gather S=1: two or more slots take the fused kernel exactly as solo fused launches; one slot keeps the decode gather" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    try std.testing.expectEqual(@as(c_int, 2), qsaAttnMinS());
    try std.testing.expectEqual(@as(c_int, 1), qsaAttnBatchedMinS());
    var prng = std.Random.DefaultPrng.init(0x51_0B47);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const kb: c_int = 64;
    const kvs = [_]c_int{ 3004, 3002, 3777, 3333 };
    const scale: f32 = 1.0 / 16.0;
    const qcfg = kv_quant.KVQuantConfig.affine(8);

    var ks: [4]mlx.mlx_array = undefined;
    var vs: [4]mlx.mlx_array = undefined;
    var dviews: [4]DenseKVView = undefined;
    var qviews: [4]DenseKVView = undefined;
    var caches: [4]KVCache = undefined;
    var qviews4: [4]DenseKVView = undefined;
    var caches4: [4]KVCache = undefined;
    var blks: [4]mlx.mlx_array = undefined;
    var hosts: [4][]i32 = undefined;
    var qs: [4]mlx.mlx_array = undefined;
    for (0..4) |i| {
        ks[i] = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kvs[i], 256 }, s);
        vs[i] = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kvs[i], 256 }, s);
        dviews[i] = DenseKVView{ .k = ks[i], .v = vs[i], .owned = false };
        caches[i] = try KVCache.initWithConfig(ta, 1, qcfg);
        qviews[i] = try qsaAttnCacheFixture(ta, s, ks[i], vs[i], kvs[i], @divTrunc(kvs[i], 2), qcfg, &caches[i]);
        caches4[i] = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(4));
        qviews4[i] = try qsaAttnCacheFixture(ta, s, ks[i], vs[i], kvs[i], @divTrunc(kvs[i], 2), kv_quant.KVQuantConfig.affine(4), &caches4[i]);
        hosts[i] = try qsaVerifyBlocksHost(ta, rnd, 1, kvs[i], kb, ratio);
        blks[i] = mlx.mlx_array_new_data(hosts[i].ptr, &[_]c_int{ 1, 1, kb }, 3, .int32);
        qs[i] = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, 1, 256 }, s);
    }
    defer for (0..4) |i| {
        _ = mlx.mlx_array_free(ks[i]);
        _ = mlx.mlx_array_free(vs[i]);
        qviews[i].deinit();
        caches[i].deinit();
        qviews4[i].deinit();
        caches4[i].deinit();
        _ = mlx.mlx_array_free(blks[i]);
        ta.free(hosts[i]);
        _ = mlx.mlx_array_free(qs[i]);
    };
    // Parent capacity past kv: a slot strided by its view's own shape would read the wrong rows.
    for (qviews, qviews4, kvs) |v, v4, kv| try std.testing.expect(v.kv_cap > kv and v4.kv_cap > kv);

    for ([_]*const [4]DenseKVView{ &dviews, &qviews, &qviews4 }) |views| {
        for ([_]c_int{ 16, 8, 1 }) |nsp| {
            sparse_attn.qsa_attn_nsplit_override = nsp;
            defer sparse_attn.qsa_attn_nsplit_override = null;
            var solos: [4]mlx.mlx_array = undefined;
            for (0..4) |i| solos[i] = (try qsaSparseAttnFloor(s, qs[i], views[i..][0..1], blks[i..][0..1], ratio, scale, 1)) orelse return error.SparseAttnDeclined;
            defer for (solos) |o| {
                _ = mlx.mlx_array_free(o);
            };
            for (2..5) |n| {
                var qn = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(qn);
                const qvec = mlx.mlx_vector_array_new_data(&qs, n);
                defer _ = mlx.mlx_vector_array_free(qvec);
                try mlx.check(mlx.mlx_concatenate_axis(&qn, qvec, 0, s));
                var want = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(want);
                const wvec = mlx.mlx_vector_array_new_data(&solos, n);
                defer _ = mlx.mlx_vector_array_free(wvec);
                try mlx.check(mlx.mlx_concatenate_axis(&want, wvec, 0, s));
                // The one N-slot launch builds its own config; a per-slot fallback reuses the solos' (dense always falls back).
                const builds = qsaAttnCfgBuilds();
                const got = try qsaBatchedGatherAttn(ta, s, qn, views[0..n], blks[0..n], &.{}, ratio, scale, &.{});
                defer _ = mlx.mlx_array_free(got);
                try std.testing.expectEqual(builds + @intFromBool(views != &dviews), qsaAttnCfgBuilds());
                try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(got, want, s));
                const again = try qsaBatchedGatherAttn(ta, s, qn, views[0..n], blks[0..n], &.{}, ratio, scale, &.{});
                _ = mlx.mlx_array_free(again);
                try std.testing.expectEqual(builds + @intFromBool(views != &dviews), qsaAttnCfgBuilds());
            }
        }

        const lone_ref = (try qsaDecodeGatherAttn(s, qs[0], &views[0], blks[0], ratio, scale)) orelse return error.GatherDeclined;
        defer _ = mlx.mlx_array_free(lone_ref);
        const lone = try qsaBatchedGatherAttn(ta, s, qs[0], views[0..1], blks[0..1], &.{}, ratio, scale, &.{});
        defer _ = mlx.mlx_array_free(lone);
        try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(lone, lone_ref, s));
    }
}

test "qsa batched gather mixed group: a mask-only slot matches its solo masked SDPA" {
    const transformer = @import("../../../engines/mlx/transformer.zig");
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    transformer.qsa_decode_gather_override = true;
    defer transformer.qsa_decode_gather_override = null;
    transformer.qsa_batched_gather_override = true;
    defer transformer.qsa_batched_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0x51a0);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const kb: c_int = 3;
    const hq: c_int = 4;
    const hkv: c_int = 2;
    const hd: c_int = 64;
    const scale: f32 = 1.0 / 8.0;
    const kL0: c_int = 50;
    const kL1: c_int = 36;
    const q0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, 1, hd }, s);
    defer _ = mlx.mlx_array_free(q0);
    const q1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, 1, hd }, s);
    defer _ = mlx.mlx_array_free(q1);
    const k0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL0, hd }, s);
    defer _ = mlx.mlx_array_free(k0);
    const v0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL0, hd }, s);
    defer _ = mlx.mlx_array_free(v0);
    const k1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL1, hd }, s);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL1, hd }, s);
    defer _ = mlx.mlx_array_free(v1);
    var fx0 = try QsaBlockFixture.build(rnd, 1, kL0, kb, ratio);
    defer fx0.deinit();
    var fx1 = try QsaBlockFixture.build(rnd, 1, kL1, kb, ratio);
    defer fx1.deinit();
    const view0 = DenseKVView{ .k = k0, .v = v0, .owned = false };
    const view1 = DenseKVView{ .k = k1, .v = v1, .owned = false };
    const solo0 = (try qsaDecodeGatherAttn(s, q0, &view0, fx0.blocks, ratio, scale)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(solo0);
    const solo1 = try attn256Reference(q1, k1, v1, scale, "array", fx1.mask, s);
    defer _ = mlx.mlx_array_free(solo1);
    var q_n2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_n2);
    const qparts = [_]mlx.mlx_array{ q0, q1 };
    const qvec = mlx.mlx_vector_array_new_data(&qparts, 2);
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_concatenate_axis(&q_n2, qvec, 0, s));
    const views = [_]DenseKVView{ view0, view1 };
    const blks = [_]mlx.mlx_array{ fx0.blocks, .{ .ctx = null } };
    const masks = [_]mlx.mlx_array{ .{ .ctx = null }, fx1.mask };
    const bat = try qsaBatchedGatherAttn(ta, s, q_n2, &views, &blks, &masks, ratio, scale, &.{});
    defer _ = mlx.mlx_array_free(bat);
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    const wparts = [_]mlx.mlx_array{ solo0, solo1 };
    const wvec = mlx.mlx_vector_array_new_data(&wparts, 2);
    defer _ = mlx.mlx_vector_array_free(wvec);
    try mlx.check(mlx.mlx_concatenate_axis(&want, wvec, 0, s));
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(bat, want, s));
}

test "qsa batched gather S=2 and S=4: N=1 is byte-identical to solo verify, N=2 matches per-slot solos" {
    const transformer = @import("../../../engines/mlx/transformer.zig");
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    verify_gather.qsa_verify_gather_override = true;
    defer verify_gather.qsa_verify_gather_override = null;
    verify_gather.qsa_verify_gather_min_kv_override = 16;
    defer verify_gather.qsa_verify_gather_min_kv_override = null;
    transformer.qsa_batched_gather_override = true;
    defer transformer.qsa_batched_gather_override = null;
    // Dense views: keep the fused kernel's bf16 variant out so the union gather is the arm under test.
    sparse_attn.qsa_attn_dense_override = false;
    defer sparse_attn.qsa_attn_dense_override = null;
    var prng = std.Random.DefaultPrng.init(0xc0de);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const kb: c_int = 3;
    const hq: c_int = 4;
    const hkv: c_int = 2;
    const hd: c_int = 256;
    const scale: f32 = 1.0 / 16.0;
    const kL0: c_int = 96;
    const kL1: c_int = 80;
    for ([_]c_int{ 2, 4 }) |seq| {
        const q0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, seq, hd }, s);
        defer _ = mlx.mlx_array_free(q0);
        const q1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, seq, hd }, s);
        defer _ = mlx.mlx_array_free(q1);
        const k0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL0, hd }, s);
        defer _ = mlx.mlx_array_free(k0);
        const v0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL0, hd }, s);
        defer _ = mlx.mlx_array_free(v0);
        const k1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL1, hd }, s);
        defer _ = mlx.mlx_array_free(k1);
        const v1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL1, hd }, s);
        defer _ = mlx.mlx_array_free(v1);
        var fx0 = try QsaBlockFixture.build(rnd, seq, kL0, kb, ratio);
        defer fx0.deinit();
        var fx1 = try QsaBlockFixture.build(rnd, seq, kL1, kb, ratio);
        defer fx1.deinit();
        const view0 = DenseKVView{ .k = k0, .v = v0, .owned = false };
        const view1 = DenseKVView{ .k = k1, .v = v1, .owned = false };
        const solo0 = (try qsaVerifyGatherAttn(s, q0, &view0, fx0.blocks, ratio, scale)) orelse return error.GatherDeclined;
        defer _ = mlx.mlx_array_free(solo0);
        const bat1 = try qsaBatchedGatherAttn(ta, s, q0, &.{view0}, &.{fx0.blocks}, &.{}, ratio, scale, &.{});
        defer _ = mlx.mlx_array_free(bat1);
        try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(bat1, solo0, s));

        var q_n2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_n2);
        const qparts = [_]mlx.mlx_array{ q0, q1 };
        const qvec = mlx.mlx_vector_array_new_data(&qparts, 2);
        defer _ = mlx.mlx_vector_array_free(qvec);
        try mlx.check(mlx.mlx_concatenate_axis(&q_n2, qvec, 0, s));
        const views = [_]DenseKVView{ view0, view1 };
        const blks = [_]mlx.mlx_array{ fx0.blocks, fx1.blocks };
        const bat2 = try qsaBatchedGatherAttn(ta, s, q_n2, &views, &blks, &.{}, ratio, scale, &.{});
        defer _ = mlx.mlx_array_free(bat2);
        const solo1 = (try qsaVerifyGatherAttn(s, q1, &view1, fx1.blocks, ratio, scale)) orelse return error.GatherDeclined;
        defer _ = mlx.mlx_array_free(solo1);
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        const wparts = [_]mlx.mlx_array{ solo0, solo1 };
        const wvec = mlx.mlx_vector_array_new_data(&wparts, 2);
        defer _ = mlx.mlx_vector_array_free(wvec);
        try mlx.check(mlx.mlx_concatenate_axis(&want, wvec, 0, s));
        try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(bat2, want, s));
    }
}

test "qsaBatchedAttn: the kill switch refuses the gather arm" {
    const transformer = @import("../../../engines/mlx/transformer.zig");
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var xfm_bytes: [@sizeOf(Trunk)]u8 align(@alignOf(Trunk)) = @splat(0);
    const xfm: *Trunk = @ptrCast(&xfm_bytes);
    xfm.allocator = std.testing.allocator;
    var q4_bytes: [@sizeOf(qwen4_mod.Qwen4State)]u8 align(@alignOf(qwen4_mod.Qwen4State)) = @splat(0);
    xfm.qwen4 = @ptrCast(&q4_bytes);
    var cache_bytes: [@sizeOf(KVCache)]u8 align(@alignOf(KVCache)) = @splat(0);
    const cache: *KVCache = @ptrCast(&cache_bytes);
    var off: usize = 0;
    var ctx: ForwardCtx = .{ .cache = cache, .moe_seq_offset = &off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    var slot: ForwardCtx = .{ .cache = cache, .moe_seq_offset = &off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    const slots = [_]*ForwardCtx{&slot};
    const dummy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dummy);

    const prev_g = transformer.qsa_batched_gather_override;
    defer transformer.qsa_batched_gather_override = prev_g;
    transformer.qsa_batched_gather_override = false;
    try std.testing.expect((try xfm.qsaBatchedAttn(&ctx, &slots, dummy, 0, 4, 1.0)) == null);
}

test "qsaBatchedAttn: an M-RoPE slot takes the gather arm, byte-identical to the same slot without positions" {
    const transformer = @import("../../../engines/mlx/transformer.zig");
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    transformer.qsa_decode_gather_override = true;
    defer transformer.qsa_decode_gather_override = null;
    transformer.qsa_batched_gather_override = true;
    defer transformer.qsa_batched_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0x3b0e);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const kb: c_int = 3;
    const hq: c_int = 4;
    const hkv: c_int = 2;
    const hd: c_int = 64;
    const kls = [_]c_int{ 50, 36 };

    var xfm_bytes: [@sizeOf(Trunk)]u8 align(@alignOf(Trunk)) = @splat(0);
    const xfm: *Trunk = @ptrCast(&xfm_bytes);
    xfm.allocator = ta;
    xfm.s = s;
    xfm.config.indexer_compress_ratio = @intCast(ratio);
    var q4_bytes: [@sizeOf(qwen4_mod.Qwen4State)]u8 align(@alignOf(qwen4_mod.Qwen4State)) = @splat(0);
    xfm.qwen4 = @ptrCast(&q4_bytes);

    var caches: [2]KVCache = undefined;
    var fxs: [2]QsaBlockFixture = undefined;
    var offs: [2]usize = undefined;
    var ctxs: [2]ForwardCtx = undefined;
    var qs: [2]mlx.mlx_array = undefined;
    for (0..2) |i| {
        caches[i] = try KVCache.init(ta, 1);
        const k = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kls[i], hd }, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kls[i], hd }, s);
        defer _ = mlx.mlx_array_free(v);
        var view = try caches[i].update(0, k, v, s, 128);
        view.deinit();
        fxs[i] = try QsaBlockFixture.build(rnd, 1, kls[i], kb, ratio);
        offs[i] = @intCast(kls[i]);
        ctxs[i] = .{ .cache = &caches[i], .moe_seq_offset = &offs[i], .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
        ctxs[i].qsa_blocks = fxs[i].blocks;
        qs[i] = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, 1, hd }, s);
    }
    defer for (0..2) |i| {
        caches[i].deinit();
        fxs[i].deinit();
        _ = mlx.mlx_array_free(qs[i]);
    };
    var q_n2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_n2);
    const qvec = mlx.mlx_vector_array_new_data(&qs, 2);
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_concatenate_axis(&q_n2, qvec, 0, s));

    const slots = [_]*ForwardCtx{ &ctxs[0], &ctxs[1] };
    const plain = (try xfm.qsaBatchedAttn(&ctxs[0], &slots, q_n2, 0, 1, 0.125)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(plain);
    const pos = [_]i32{ 0, 1, 2 };
    ctxs[1].mrope_pos = &pos;
    const with_pos = (try xfm.qsaBatchedAttn(&ctxs[0], &slots, q_n2, 0, 1, 0.125)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(with_pos);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(with_pos, plain, s));
}

test "qsaBatchedGatherOn matches the arm's switches" {
    const transformer = @import("../../../engines/mlx/transformer.zig");
    const prev_b = transformer.qsa_batched_gather_override;
    const prev_g = transformer.qsa_gather_override;
    const prev_d = transformer.qsa_decode_gather_override;
    const prev_v = verify_gather.qsa_verify_gather_override;
    defer {
        transformer.qsa_batched_gather_override = prev_b;
        transformer.qsa_gather_override = prev_g;
        transformer.qsa_decode_gather_override = prev_d;
        verify_gather.qsa_verify_gather_override = prev_v;
    }
    transformer.qsa_batched_gather_override = true;
    transformer.qsa_gather_override = true;
    transformer.qsa_decode_gather_override = true;
    verify_gather.qsa_verify_gather_override = true;
    try std.testing.expect(qsaBatchedGatherOn(1));
    try std.testing.expect(qsaBatchedGatherOn(4));
    transformer.qsa_batched_gather_override = false;
    try std.testing.expect(!qsaBatchedGatherOn(1));
    transformer.qsa_batched_gather_override = true;
    transformer.qsa_gather_override = false;
    try std.testing.expect(!qsaBatchedGatherOn(1));
    transformer.qsa_gather_override = true;
    transformer.qsa_decode_gather_override = false;
    try std.testing.expect(!qsaBatchedGatherOn(1));
    try std.testing.expect(qsaBatchedGatherOn(4));
    transformer.qsa_decode_gather_override = true;
    verify_gather.qsa_verify_gather_override = false;
    try std.testing.expect(qsaBatchedGatherOn(1));
    // The fused verify kernel still takes blocks at verify widths; its switch too turns them off.
    try std.testing.expect(qsaBatchedGatherOn(4));
    {
        const prev_k = transformer.qsa_attn_kernel_override;
        defer transformer.qsa_attn_kernel_override = prev_k;
        transformer.qsa_attn_kernel_override = false;
        try std.testing.expect(!qsaBatchedGatherOn(4));
    }
    verify_gather.qsa_verify_gather_override = true;
    const prev_st = standin.qwen4_standin_override;
    defer standin.qwen4_standin_override = prev_st;
    standin.qwen4_standin_override = .{ .attn_qsa = true };
    try std.testing.expect(!qsaBatchedGatherOn(1));
    standin.qwen4_standin_override = .{ .attn_sdpa = true };
    try std.testing.expect(!qsaBatchedGatherOn(1));
    standin.qwen4_standin_override = .{};
    try std.testing.expect(qsaBatchedGatherOn(1));
}

test "qsaBatchedGatherEnabled: default on, override restores the dense padded mask" {
    const transformer = @import("../../../engines/mlx/transformer.zig");
    const prev = transformer.qsa_batched_gather_override;
    defer transformer.qsa_batched_gather_override = prev;
    transformer.qsa_batched_gather_override = true;
    try std.testing.expect(qsaBatchedGatherEnabled());
    transformer.qsa_batched_gather_override = false;
    try std.testing.expect(!qsaBatchedGatherEnabled());
}

test "qsa batched gather plan: per-slot rows match the solo planner at S=1 and S=2..8" {
    const ta = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xa11c3);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const kb: c_int = 32;
    const kvs = [_]c_int{ 4096, 20000 };
    for (kvs) |kv_a| {
        const kv_b: c_int = @divTrunc(kv_a, 2) + 17;
        {
            const s: c_int = 1;
            const ba = try qsaVerifyBlocksHost(ta, rnd, s, kv_a, kb, ratio);
            defer ta.free(ba);
            const bb = try qsaVerifyBlocksHost(ta, rnd, s, kv_b, kb, ratio);
            defer ta.free(bb);
            const slot_blocks = [_][]const i32{ ba, bb };
            const kv_lens = [_]c_int{ kv_a, kv_b };
            var plans = try qsaBatchedPlanHost(ta, &slot_blocks, s, kb, &kv_lens, ratio);
            defer {
                var pi: usize = 0;
                while (pi < plans.len) : (pi += 1) plans[pi].deinit(ta);
                ta.free(plans);
            }
            try std.testing.expectEqual(@as(usize, 2), plans.len);
            const ea = try qsaDecodeGatherIdsHost(ta, ba, kb, kv_a, ratio);
            defer ta.free(ea);
            const eb = try qsaDecodeGatherIdsHost(ta, bb, kb, kv_b, ratio);
            defer ta.free(eb);
            try std.testing.expectEqualSlices(i32, ea, plans[0].rows);
            try std.testing.expectEqualSlices(i32, eb, plans[1].rows);
        }
        for ([_]c_int{ 2, 4, 5, 8 }) |s| {
            const ba = try qsaVerifyBlocksHost(ta, rnd, s, kv_a, kb, ratio);
            defer ta.free(ba);
            const bb = try qsaVerifyBlocksHost(ta, rnd, s, kv_b, kb, ratio);
            defer ta.free(bb);
            const slot_blocks = [_][]const i32{ ba, bb };
            const kv_lens = [_]c_int{ kv_a, kv_b };
            var plans = try qsaBatchedPlanHost(ta, &slot_blocks, s, kb, &kv_lens, ratio);
            defer {
                var pi: usize = 0;
                while (pi < plans.len) : (pi += 1) plans[pi].deinit(ta);
                ta.free(plans);
            }
            var pa = try qsaVerifyPlanHost(ta, ba, s, kb, kv_a, ratio);
            defer pa.deinit(ta);
            var pb = try qsaVerifyPlanHost(ta, bb, s, kb, kv_b, ratio);
            defer pb.deinit(ta);
            try std.testing.expectEqualSlices(i32, pa.rows, plans[0].rows);
            try std.testing.expectEqualSlices(bool, pa.mask, plans[0].mask);
            try std.testing.expectEqualSlices(i32, pb.rows, plans[1].rows);
            try std.testing.expectEqualSlices(bool, pb.mask, plans[1].mask);
        }
    }
}

test "qsa decode cuda: the one-token kernel is no worse than the gathered SDPA against the f64 truth" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const cuda = @import("../../../sdk/blocks/attention/sparse_block/decode_cuda.zig");
    const testing = std.testing;
    const BD: c_int = 256;
    const s = mlx.gpuStream();
    const hq: usize = 24;
    const hk: usize = 2;
    const gqa = hq / hk;
    const kb: usize = 512;
    const ratio: usize = 4;
    const cap: usize = 12288;
    const scale: f32 = 1.0 / 16.0;
    var prng = std.Random.DefaultPrng.init(0xDEC0DE);
    const rnd = prng.random();
    for ([_]usize{ 2052, 2055, 4099, 8076, 11000 }, 0..) |kv, case| {
        const q = try cuda.randomBf16(s, &[_]c_int{ 1, @intCast(hq), 1, BD }, 100 + case);
        defer _ = mlx.mlx_array_free(q);
        const kfull = try cuda.randomBf16(s, &[_]c_int{ 1, @intCast(hk), @intCast(cap), BD }, 200 + case);
        defer _ = mlx.mlx_array_free(kfull);
        const vfull = try cuda.randomBf16(s, &[_]c_int{ 1, @intCast(hk), @intCast(cap), BD }, 300 + case);
        defer _ = mlx.mlx_array_free(vfull);
        // The cache views a decode step sees: a leading slice of a larger buffer.
        var kview = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kview);
        var vview = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vview);
        const lo = [_]c_int{ 0, 0, 0, 0 };
        const hi = [_]c_int{ 1, @intCast(hk), @intCast(kv), BD };
        const st = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&kview, kfull, &lo, 4, &hi, 4, &st, 4, s));
        try mlx.check(mlx.mlx_slice(&vview, vfull, &lo, 4, &hi, 4, &st, 4, s));

        const nb = kv / ratio;
        const pool = try testing.allocator.alloc(i32, nb);
        defer testing.allocator.free(pool);
        for (pool, 0..) |*p, i| p.* = @intCast(i);
        rnd.shuffle(i32, pool);
        const sel = try testing.allocator.dupe(i32, pool[0..kb]);
        defer testing.allocator.free(sel);
        std.mem.sort(i32, sel, {}, std.sort.asc(i32));
        const blocks = mlx.mlx_array_new_data(sel.ptr, &[_]c_int{ 1, 1, @intCast(kb) }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);

        // f64 truth over the same bf16 values.
        const qh = try cuda.hostF32(s, q);
        defer _ = mlx.mlx_array_free(qh.arr);
        const kh = try cuda.hostF32(s, kview);
        defer _ = mlx.mlx_array_free(kh.arr);
        const vh = try cuda.hostF32(s, vview);
        defer _ = mlx.mlx_array_free(vh.arr);
        var pos_list: std.ArrayList(usize) = .empty;
        defer pos_list.deinit(testing.allocator);
        for (sel) |b| for (0..ratio) |r| try pos_list.append(testing.allocator, @as(usize, @intCast(b)) * ratio + r);
        var t = nb * ratio;
        while (t < kv) : (t += 1) try pos_list.append(testing.allocator, t);
        const truth = try testing.allocator.alloc(f64, hq * BD);
        defer testing.allocator.free(truth);
        const sc = try testing.allocator.alloc(f64, pos_list.items.len);
        defer testing.allocator.free(sc);
        for (0..hq) |h| {
            const g = h / gqa;
            var mx: f64 = -std.math.inf(f64);
            for (pos_list.items, 0..) |pos, i| {
                var dot: f64 = 0;
                for (0..BD) |d| dot += @as(f64, qh.data[h * BD + d]) * @as(f64, kh.data[(g * kv + pos) * BD + d]);
                sc[i] = dot * scale;
                mx = @max(mx, sc[i]);
            }
            var z: f64 = 0;
            for (sc) |*x| {
                x.* = @exp(x.* - mx);
                z += x.*;
            }
            for (0..BD) |d| {
                var o: f64 = 0;
                for (pos_list.items, 0..) |pos, i| o += sc[i] * @as(f64, vh.data[(g * kv + pos) * BD + d]);
                truth[h * BD + d] = o / z;
            }
        }

        const got = (try cuda.attend(s, q, kview, vview, blocks, @intCast(ratio), scale)) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(got);
        // The arm it replaces on CUDA: gather the picks and the tail, then the masked SDPA.
        var view: DenseKVView = .{ .k = kview, .v = vview, .owned = false };
        const stock = (try qsaDecodeGatherAttn(s, q, &view, blocks, @intCast(ratio), scale)) orelse return error.StockDeclined;
        defer _ = mlx.mlx_array_free(stock);
        const gh = try cuda.hostF32(s, got);
        defer _ = mlx.mlx_array_free(gh.arr);
        const sh = try cuda.hostF32(s, stock);
        defer _ = mlx.mlx_array_free(sh.arr);
        try testing.expectEqual(hq * BD, gh.data.len);
        var e_got: f64 = 0;
        var e_stock: f64 = 0;
        for (truth, 0..) |tv, i| {
            try testing.expect(std.math.isFinite(gh.data[i]));
            e_got += (@as(f64, gh.data[i]) - tv) * (@as(f64, gh.data[i]) - tv);
            e_stock += (@as(f64, sh.data[i]) - tv) * (@as(f64, sh.data[i]) - tv);
        }
        const rms_got = @sqrt(e_got / @as(f64, @floatFromInt(truth.len)));
        const rms_stock = @sqrt(e_stock / @as(f64, @floatFromInt(truth.len)));
        std.debug.print("[qsa-dec cuda parity] kv={d} rms_err kernel={e:.3} stock={e:.3}\n", .{ kv, rms_got, rms_stock });
        try testing.expect(rms_got <= rms_stock);
    }
}
