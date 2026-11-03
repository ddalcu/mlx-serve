//! engines/mlx/methods/gated_full_attn: `Transformer` methods (aliased from engines/mlx/transformer.zig).
//! Owns gatedFullAttnWith, gatedFullAttnProjected, gatedAttnTail.

const mlx = @import("../../../mlx/ffi.zig");
const rowqmv = @import("../../../sdk/blocks/qmm/rowqmv.zig");
const lane_attn = @import("../../../sdk/blocks/attention/lane_attn.zig");
const row_attn = @import("../../../sdk/blocks/attention/row_attn.zig");
const log = @import("../../../sdk/util/log.zig");
const decode_fused = @import("../../../sdk/blocks/gdn/decode_fused.zig");
const residual_norm = @import("../../../sdk/blocks/norms/residual_norm.zig");
const MAX_BATCH_ROWS = @import("../../../sdk/bases/run.zig").MAX_BATCH_ROWS;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const KVCache = @import("../kv/cache.zig").KVCache;
const FullAttnWeights = @import("../layer_weights.zig").FullAttnWeights;
const Transformer = @import("../transformer.zig").Transformer;
const AttnVerifyInputs = @import("../verify/joined_rows.zig").AttnVerifyInputs;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;
const constTableAs = @import("../../../mlx/dtype.zig").constTableAs;
const Qwen4AttnProf = @import("../../../plugins/qwen4_exp/prof.zig").Qwen4AttnProf;
const noteQsaArmForSlots = @import("../../../plugins/qwen4_exp/qsa/arms.zig").noteQsaArmForSlots;
const qsaSlotAttn = @import("../../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaSlotAttn;
const standinRef = @import("../../../sdk/diag/standin.zig").standinRef;
const fusedQkNormRope = @import("../../../sdk/blocks/rope/qk_norm_rope.zig").fusedQkNormRope;
const fusedQkNormRope256 = @import("../../../sdk/blocks/rope/qk_norm_rope.zig").fusedQkNormRope256;
const qkNormRopeFusedEnabled = @import("../../../sdk/blocks/rope/qk_norm_rope.zig").qkNormRopeFusedEnabled;
const fusedSdpa256Prefill = @import("../../../sdk/blocks/attention/attn256.zig").fusedSdpa256Prefill;
const batchedAttnPerSlot = @import("../../../sdk/blocks/attention/batched.zig").batchedAttnPerSlot;
const stridedSigmoidGateMul = @import("../../../sdk/blocks/attention/gates.zig").stridedSigmoidGateMul;
const kvAttnFusedEligible = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").kvAttnFusedEligible;
const kvAttnVerifyEligible = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").kvAttnVerifyEligible;
const logKvAttnFusedEngaged = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").logKvAttnFusedEngaged;
const qkvAttnDecodeKernel = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").qkvAttnDecodeKernel;
const qkvAttnVerifyKernel = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").qkvAttnVerifyKernel;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const rowByRowSdpa = @import("../../../sdk/blocks/attention/sdpa.zig").rowByRowSdpa;
const sdpaForceFused = @import("../../../sdk/blocks/attention/sdpa.zig").sdpaForceFused;
const splitCausalSdpa = @import("../../../sdk/blocks/attention/sdpa.zig").splitCausalSdpa;

pub fn gatedFullAttnWith(
    self: *Transformer,
    ctx: *ForwardCtx,
    x: mlx.mlx_array,
    fa: *const FullAttnWeights,
    layer: u32,
    offset: c_int,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
) !mlx.mlx_array {
    return self.gatedFullAttnProjected(ctx, x, fa, layer, offset, batch, seq_len, is_prefill, null, false);
}

pub fn gatedFullAttnProjected(
    self: *Transformer,
    ctx: *ForwardCtx,
    x: mlx.mlx_array,
    fa: *const FullAttnWeights,
    layer: u32,
    offset: c_int,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
    projected: ?*const AttnVerifyInputs,
    skip_output: bool,
) !mlx.mlx_array {
    const cache = ctx.cache;
    const cfg = &self.config;
    const h_count: c_int = @intCast(cfg.num_attention_heads);
    const kv_h: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
    const rope_dims: c_int = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
    const flat_shape = [_]c_int{ batch, seq_len, h_count * hd };

    // Q, K, V projections: one joined matmul where the loader built it.
    var joined: [3]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (joined) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    if (projected == null and fa.qkv.w.ctx != null and self.rowGroupServes(batch * seq_len)) {
        const all = try self.qmatmul(x, fa.qkv.w, fa.qkv.s, fa.qkv.b);
        defer _ = mlx.mlx_array_free(all);
        for (&joined, 0..) |*p, i| {
            p.* = mlx.mlx_array_new();
            try fa.qkv.part(p, all, i, self.s);
        }
        if (!decode_fused.attn_qkv_joined_engaged) {
            decode_fused.attn_qkv_joined_engaged = true;
            log.info("[attn] joined q|k|v engaged: rows={d}\n", .{batch * seq_len});
        }
    }
    const own = projected == null and joined[0].ctx == null;
    const q_proj = if (projected) |values| values.q else if (!own) joined[0] else try self.attnProj(x, fa.q_w, fa.q_s, fa.q_b, batch == 1 and !is_prefill, layer);
    defer if (own) {
        _ = mlx.mlx_array_free(q_proj);
    };

    // With output gate: q_proj outputs [B, S, 2*H*D], split into queries + gate
    // Without: q_proj outputs [B, S, H*D], used directly as queries
    var queries: mlx.mlx_array = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(queries);
    var gate: mlx.mlx_array = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);

    if (cfg.attn_output_gate) {
        // Mirror mlx-lm qwen3_next.py:130-134: reshape Q-proj output to [B, S, H, D*2]
        // then `mx.split(_, 2, axis=-1)` into (queries, gate). The single split op
        // replaces our prior two-slice pattern (2 dispatches → 1 dispatch). Adds up
        // across all `full_attention_interval` layers — was the dominant Qwen 3.5/3.6
        // hybrid decode gap vs mlx-lm (5.7% → ~tied).
        const q_gate_shape = [_]c_int{ batch, seq_len, h_count, hd * 2 };
        var q_gate_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_gate_r);
        try mlx.check(mlx.mlx_reshape(&q_gate_r, q_proj, &q_gate_shape, 4, self.s));

        var split_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(split_vec);
        try mlx.check(mlx.mlx_split(&split_vec, q_gate_r, 2, -1, self.s));
        if (mlx.mlx_vector_array_size(split_vec) != 2) return error.UnexpectedSplitCount;

        try mlx.check(mlx.mlx_vector_array_get(&queries, split_vec, 0));

        // The gate STAYS 4-D: flattening the split view here merges the
        // head axis across the packed q/gate interleave — a REAL Copy
        // kernel per call. Elementwise sigmoid/multiply below take the
        // strided view copy-free; the element pairing (h, d) <-> flat
        // h*D+d is identical either way.
        try mlx.check(mlx.mlx_vector_array_get(&gate, split_vec, 1));
    } else {
        const q_shape = [_]c_int{ batch, seq_len, h_count, hd };
        try mlx.check(mlx.mlx_reshape(&queries, q_proj, &q_shape, 4, self.s));
    }

    const k_proj = if (projected) |values| values.k else if (!own) joined[1] else try self.attnProj(x, fa.k_w, fa.k_s, fa.k_b, batch == 1 and !is_prefill, layer);
    defer if (own) {
        _ = mlx.mlx_array_free(k_proj);
    };
    const v_proj = if (projected) |values| values.v else if (!own) joined[2] else try self.attnProj(x, fa.v_w, fa.v_s, fa.v_b, batch == 1 and !is_prefill, layer);
    defer if (own) {
        _ = mlx.mlx_array_free(v_proj);
    };

    const kv_shape = [_]c_int{ batch, seq_len, kv_h, hd };
    var k_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_r);
    var v_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_r);
    try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, self.s));
    try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, self.s));

    // V transpose (independent of the q/k chain).
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var v_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_t);
    try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));

    // Fused decode QK-norm+RoPE (the laguna kernel, hd-128 partial-rotary):
    // one dispatch replaces per-head RMSNorm → transpose → partial RoPE for
    // q AND k. Bit-identical to the composed chain below (parity-pinned
    // incl. the rd=32 qwen geometry). With `attn_output_gate` the q input
    // is the post-split strided view — ensure_row_contiguous materializes
    // it, which is the one extra (tiny) copy this path pays. M-RoPE prefill
    // chunks keep the composed path; decode is scalar rope at
    // offset+delta, which the probe row reproduces exactly.
    //
    // YaRN (a `rope_type: "yarn"` rope_parameters): the scaled denominator
    // array goes to every rope here (mlx_fast_rope takes base OR freqs, so
    // `has_value` flips) and the mscale rides along — post-rotation on q/k
    // for the composed arms, folded into the probe's cos|sin rows for the
    // fused ones (scaling both halves of a rotary pair is the same
    // multiply). `family` 1 keeps the angle cache from serving an unscaled
    // row to a scaled layer, exactly as laguna's yarn split does.
    const use_yarn = self.yarnActive();
    const rope_family: usize = if (use_yarn) 1 else 0;
    const rope_base = mlx.mlx_optional_float{
        .value = self.config.rope_theta,
        .has_value = !use_yarn,
    };
    const rope_freqs: mlx.mlx_array = if (use_yarn) self.rope_freqs_yarn.? else .{ .ctx = null };
    const rope_mscale: f32 = if (use_yarn) self.config.yarn_attention_factor else 1.0;
    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    var fused_qk = false;
    if (batch == 1 and seq_len == 1 and hd == 128 and ctx.mrope_cos_cur == null and ctx.head_rope_pos.ctx == null and
        fa.q_norm.ctx != null and fa.k_norm.ctx != null and
        self.rms_eps_arr.ctx != null and qkNormRopeFusedEnabled())
    blk: {
        const eff_off: c_int = offset + (if (ctx.mrope_pos != null) ctx.mrope_delta else 0);
        const angles = self.qkAngleFor(rope_family, rope_dims, rope_base, rope_freqs, eff_off, 1.0) catch break :blk;
        const ms: ?mlx.mlx_array = if (use_yarn)
            try constTableAs(self.yarn_mscale.?, mlx.mlx_array_dtype(queries), &self.yarn_mscale_cast, self.s)
        else
            null;
        const pair = (fusedQkNormRope(self.s, queries, k_r, fa.q_norm, fa.k_norm, angles, ms, self.rms_eps_arr, h_count, kv_h, rope_dims) catch null) orelse break :blk;
        _ = mlx.mlx_array_free(q_rope);
        _ = mlx.mlx_array_free(k_rope);
        q_rope = pair[0];
        k_rope = pair[1];
        fused_qk = true;
    }
    // hd-256 sibling (qwen3.5/3.6 full-attention layers), decode AND
    // spec-verify widths (S 1..16) — the hd-128 gate above never fires
    // there, so those layers paid the composed chain on every step.
    if (!fused_qk and batch == 1 and hd == 256 and seq_len <= 32 and
        ctx.mrope_cos_cur == null and ctx.head_rope_pos.ctx == null and
        fa.q_norm.ctx != null and fa.k_norm.ctx != null and
        self.rms_eps_arr.ctx != null and qkNormRopeFusedEnabled())
    blk: {
        const eff_off: c_int = offset + (if (ctx.mrope_pos != null) ctx.mrope_delta else 0);
        // The kernel has no mscale slot: YaRN's factor is uniform across the
        // rotated slice, so folding it into the angle rows scales exactly the
        // rotation and nothing else (the pass-through dims never appear).
        const angle_rows = self.qkAngleRowsFor(rope_family, rope_dims, rope_base, rope_freqs, eff_off, if (ctx.tree) |t| t.attn.max_depth + 1 else seq_len, rope_mscale, 1.0) catch break :blk;
        // A tree row sits at offset + its depth.
        var by_depth: mlx.mlx_array = .{ .ctx = null };
        defer if (by_depth.ctx != null) {
            _ = mlx.mlx_array_free(by_depth);
        };
        if (ctx.tree) |t| {
            var table = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(table);
            try mlx.check(mlx.mlx_reshape(&table, angle_rows, &[_]c_int{ t.attn.max_depth + 1, rope_dims }, 2, self.s));
            var rows = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(rows);
            try mlx.check(mlx.mlx_take_axis(&rows, table, t.attn.depth, 0, self.s));
            by_depth = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&by_depth, rows, &[_]c_int{seq_len * rope_dims}, 1, self.s));
        }
        const angles = if (ctx.tree != null) by_depth else angle_rows;
        const none_v = mlx.mlx_array{ .ctx = null };
        const pair = (fusedQkNormRope256(self.s, queries, k_r, none_v, fa.q_norm, fa.k_norm, none_v, angles, self.rms_eps_arr, h_count, kv_h, seq_len, rope_dims) catch null) orelse break :blk;
        _ = mlx.mlx_array_free(q_rope);
        _ = mlx.mlx_array_free(k_rope);
        q_rope = pair[0];
        k_rope = pair[1];
        fused_qk = true;
    }
    if (!fused_qk and ctx.tree != null) return error.SpecTreeUnsupported;
    if (!fused_qk) {
        // Q/K norms
        const q_normed = try self.rmsNorm(queries, fa.q_norm);
        defer _ = mlx.mlx_array_free(q_normed);
        const k_normed = try self.rmsNorm(k_r, fa.k_norm);
        defer _ = mlx.mlx_array_free(k_normed);

        // Transpose to [B, H, S, D]
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));

        // Partial RoPE. Qwen3-VL image requests use interleaved M-RoPE: manual
        // per-token cos/sin on the prefill chunk (3D t/h/w angles at image
        // tokens), and scalar RoPE at offset+delta on decode (decode tokens are
        // text → t=h=w). Plain scalar partial RoPE otherwise (zero-cost for
        // text-only qwen3_5). YaRN needs nothing here: the M-RoPE tables are
        // filled from the scaled spectrum + mscale at the one choke point
        // (`mropeCosSinAt`), and the scalar arms take the freqs array below.
        if (ctx.mrope_cos_cur) |cos| {
            const sin = ctx.mrope_sin_cur.?;
            _ = mlx.mlx_array_free(q_rope);
            _ = mlx.mlx_array_free(k_rope);
            q_rope = try self.applyMrope(q_t, cos, sin, rope_dims);
            k_rope = try self.applyMrope(k_t, cos, sin, rope_dims);
        } else if (ctx.head_rope_pos.ctx != null) {
            try mlx.check(mlx.mlx_fast_rope_dynamic(&q_rope, q_t, rope_dims, false, rope_base, 1.0, ctx.head_rope_pos, rope_freqs, self.s));
            try mlx.check(mlx.mlx_fast_rope_dynamic(&k_rope, k_t, rope_dims, false, rope_base, 1.0, ctx.head_rope_pos, rope_freqs, self.s));
            if (use_yarn) try self.yarnScaleQK(&q_rope, &k_rope);
        } else if (ctx.batch_rope_offsets) |off_arr| {
            // Batched decode: every slot sits at its own position, so the
            // offset is an [N] array, not a scalar. The driver already
            // folded each slot's M-RoPE delta into it.
            try mlx.check(mlx.mlx_fast_rope_dynamic(&q_rope, q_t, rope_dims, false, rope_base, 1.0, off_arr, rope_freqs, self.s));
            try mlx.check(mlx.mlx_fast_rope_dynamic(&k_rope, k_t, rope_dims, false, rope_base, 1.0, off_arr, rope_freqs, self.s));
            if (use_yarn) try self.yarnScaleQK(&q_rope, &k_rope);
        } else {
            const eff_offset: c_int = offset + (if (ctx.mrope_pos != null) ctx.mrope_delta else 0);
            try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, rope_base, 1.0, eff_offset, rope_freqs, self.s));
            try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, rope_base, 1.0, eff_offset, rope_freqs, self.s));
            if (use_yarn) try self.yarnScaleQK(&q_rope, &k_rope);
        }
    }

    // Batched decode: each slot owns its own KV cache and its own kv_len,
    // so the update happens per slot at B=1 (a slice of the stacked k/v)
    // and the reads are padded to a common width and stacked — the same
    // shape the standard batched path builds. Everything downstream (gate,
    // o_proj, MLP) is batch-invariant and stays shared.
    if (ctx.batch_slots) |slots| {
        var attn_out_b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out_b);
        {
            var cache_buf: [MAX_BATCH_ROWS]*KVCache = undefined;
            for (slots, 0..) |slot_ctx, i| cache_buf[i] = slot_ctx.cache;
            try KVCache.appendFromStacked(
                k_rope,
                v_t,
                cache_buf[0..slots.len],
                layer,
                self.s,
                0,
                self.batched_kv_append_override orelse true,
            );

            {
                const dense_views = try self.allocator.alloc(DenseKVView, slots.len);
                defer {
                    for (dense_views) |*dv| dv.deinit();
                    self.allocator.free(dense_views);
                }
                for (dense_views) |*dv| dv.* = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false };
                var fused_buf: [MAX_BATCH_ROWS]bool = undefined;
                for (slots, 0..) |slot_ctx, i| {
                    dense_views[i] = try slot_ctx.cache.denseView(layer, self.s);
                    fused_buf[i] = slot_ctx.kv_attn_fused;
                }
                _ = mlx.mlx_array_free(attn_out_b);
                attn_out_b = mlx.mlx_array_new();
                if (batchedAttnPerSlot(dense_views)) {
                    attn_out_b = try self.perSlotBatchedAttn(q_rope, dense_views, fused_buf[0..slots.len], seq_len, attn_scale);
                } else {
                    attn_out_b = try self.stackedBatchedAttn(q_rope, dense_views, seq_len, attn_scale, .{ .ctx = null });
                }
            }
        }
        return self.gatedAttnTail(attn_out_b, gate, fa, flat_shape, batch, is_prefill, layer, skip_output);
    }

    var kv_view = try cache.update(layer, k_rope, v_t, self.s, 0);
    defer kv_view.deinit();
    const full_k = kv_view.k;
    const full_v = kv_view.v;

    // Attention
    var attn_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_out);
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    // Fused-attn opt-in: see standard attention site for design notes
    // (kernel-or-DENSE — a declined kernel falls to the arms below).
    const sel_mode_moe: []const u8 = if (is_prefill) "causal" else "";
    var kv_fused_done = false;
    if (ctx.kv_attn_fused and kvAttnFusedEligible(&kv_view, seq_len)) {
        if (try qkvAttnDecodeKernel(self.s, q_rope, &kv_view, attn_scale, sel_mode_moe, none_mask)) |fused| {
            logKvAttnFusedEngaged(&kv_view, q_rope, seq_len);
            _ = mlx.mlx_array_free(attn_out);
            attn_out = fused;
            kv_fused_done = true;
        }
    } else if (ctx.kv_attn_fused and kvAttnVerifyEligible(&kv_view, seq_len)) {
        // Spec-verify widths (Phase 2): is_prefill is seq_len > 1, so
        // sel_mode_moe is "causal" here — the mode the kernel serves.
        if (try qkvAttnVerifyKernel(self.s, q_rope, &kv_view, attn_scale, sel_mode_moe)) |fused| {
            _ = mlx.mlx_array_free(attn_out);
            attn_out = fused;
            kv_fused_done = true;
        }
    }
    if (kv_fused_done) {
        // packed kernel handled this layer
    } else if (self.config.rowExactDecode() and seq_len <= row_attn.MAX_ROWS and batch == 1 and
        ((naxAvailable() and try lane_attn.sdpa(&attn_out, q_rope, full_k, full_v, attn_scale, if (ctx.tree) |t| t.attn else null, self.s)) or
            try row_attn.sdpa(&attn_out, q_rope, full_k, full_v, attn_scale, if (ctx.tree) |t| t.attn else null, self.s)))
    {
        // every row, serial steps included, through the one fixed arithmetic
    } else if (ctx.tree != null) {
        return error.SpecTreeUnsupported;
    } else if (self.config.rowExactDecode() and seq_len > 1 and seq_len <= rowqmv.MAX_ROWS and batch == 1) {
        try rowByRowSdpa(&attn_out, q_rope, full_k, full_v, attn_scale, seq_len, self.s);
    } else if (is_prefill) {
        if (try fusedSdpa256Prefill(self.s, q_rope, full_k, full_v, attn_scale, 0)) |fused| {
            _ = mlx.mlx_array_free(attn_out);
            attn_out = fused;
        } else if (try splitCausalSdpa(self.s, q_rope, full_k, full_v, attn_scale)) |split_out| {
            // Verify widths past the vector wall (qL*gqa > 32): vector-path
            // row groups beat MLX's unfused fallback.
            _ = mlx.mlx_array_free(attn_out);
            attn_out = split_out;
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, sdpaForceFused(q_rope, full_k), self.s));
        }
    } else {
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, false, self.s));
    }

    return self.gatedAttnTail(attn_out, gate, fa, flat_shape, batch, is_prefill, layer, skip_output);
}

/// Shared tail of `gatedFullAttnWith`: transpose the attention output back
/// to [B,S,H,D], apply the optional sigmoid output gate, flatten, project.
/// One copy so the batched-decode branch cannot drift from the serial one.
pub fn gatedAttnTail(
    self: *Transformer,
    attn_out: mlx.mlx_array,
    gate: mlx.mlx_array,
    fa: *const FullAttnWeights,
    flat_shape: [3]c_int,
    batch: c_int,
    is_prefill: bool,
    layer: u32,
    skip_output: bool,
) !mlx.mlx_array {
    // Transpose back [B,H,S,D] -> [B,S,H,D] (view)
    const perm_back = [_]c_int{ 0, 2, 1, 3 };
    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));

    // Optional output gating: multiply 4-D (strided inputs are copy-free
    // in the elementwise kernel), then flatten the CONTIGUOUS product —
    // a free view. Flattening attn_t/gate first paid two REAL Copy
    // kernels per call (reshape of a transpose/split view).
    if (self.config.attn_output_gate and flat_shape[0] * flat_shape[1] <= residual_norm.add_norm_max_rows) {
        if (try stridedSigmoidGateMul(self.s, attn_t, gate)) |gated_4d| {
            defer _ = mlx.mlx_array_free(gated_4d);
            var gated = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gated);
            try mlx.check(mlx.mlx_reshape(&gated, gated_4d, &flat_shape, 3, self.s));
            if (skip_output) return standinRef(gated);
            return self.attnProj(gated, fa.o_w, fa.o_s, fa.o_b, batch == 1 and !is_prefill, layer);
        }
    }
    if (self.config.attn_output_gate) {
        var gate_sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_sig);
        try mlx.check(mlx.mlx_sigmoid(&gate_sig, gate, self.s));
        var gated_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated_4d);
        try mlx.check(mlx.mlx_multiply(&gated_4d, attn_t, gate_sig, self.s));
        var gated = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated);
        try mlx.check(mlx.mlx_reshape(&gated, gated_4d, &flat_shape, 3, self.s));
        if (skip_output) return standinRef(gated);
        return self.attnProj(gated, fa.o_w, fa.o_s, fa.o_b, batch == 1 and !is_prefill, layer);
    }

    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));
    if (skip_output) return standinRef(attn_flat);
    return self.attnProj(attn_flat, fa.o_w, fa.o_s, fa.o_b, batch == 1 and !is_prefill, layer);
}
