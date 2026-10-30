//! The trunk's full-attention layer: gated q/k/v projections, QK norm and partial (M-)RoPE (YaRN), the KV append,
//! the QSA-served or dense attention arm, the sigmoid output gate; the batched-decode arms over each slot's cache.
//! copied_from: sdk/blocks/attention/{gated_full,proj,batched}.zig, sdk/blocks/rope/{rope,mrope,yarn}.zig (the
//! qwen4_exp arms: no spec tree, no row-exact lanes).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const mrope = @import("../../../sdk/vision/mrope.zig");
const log = @import("../../../sdk/util/log.zig");
const decode_fused = @import("../../../sdk/blocks/gdn/decode_fused.zig");
const residual_norm = @import("../../../sdk/blocks/norms/residual_norm.zig");
const layered = @import("../../../engines/mlx/batched/layered.zig");
const Trunk = @import("../trunk.zig").Trunk;
const MAX_BATCH_ROWS = layered.MAX_BATCH_ROWS;
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const FullAttnWeights = @import("../../../engines/mlx/layer_weights.zig").FullAttnWeights;
const AttnVerifyInputs = @import("verify.zig").AttnVerifyInputs;
const scalarOf = @import("../../../engines/mlx/load/weights.zig").scalarOf;
const bf16Scalar = @import("../../../engines/mlx/load/weights.zig").bf16Scalar;
const axisView = @import("../../../mlx/array.zig").axisView;
const constTableAs = @import("../../../mlx/dtype.zig").constTableAs;
const Qwen4AttnProf = @import("../prof.zig").Qwen4AttnProf;
const noteQsaArmForSlots = @import("../qsa/arms.zig").noteQsaArmForSlots;
const qsaSlotAttn = @import("../qsa/sparse_attn.zig").qsaSlotAttn;
const standinRef = @import("../../../sdk/diag/standin.zig").standinRef;
const fusedQkNormRope = @import("../../../sdk/blocks/rope/qk_norm_rope.zig").fusedQkNormRope;
const fusedQkNormRope256 = @import("../../../sdk/blocks/rope/qk_norm_rope.zig").fusedQkNormRope256;
const qkNormRopeFusedEnabled = @import("../../../sdk/blocks/rope/qk_norm_rope.zig").qkNormRopeFusedEnabled;
const ropeAngleRows = @import("../../../sdk/blocks/rope/qk_norm_rope.zig").ropeAngleRows;
const MropeCosSin = @import("../../../sdk/blocks/rope/mrope.zig").MropeCosSin;
const mropeContext = @import("../../../engines/mlx/methods/mrope.zig").mropeContext;
const yarnMscale = @import("../../../engines/mlx/methods/yarn.zig").yarnMscale;
const fusedSdpa256Prefill = @import("../../../sdk/blocks/attention/attn256.zig").fusedSdpa256Prefill;
const batchedAttnPerSlot = @import("../../../sdk/blocks/attention/batched.zig").batchedAttnPerSlot;
const stridedSigmoidGateMul = @import("../../../sdk/blocks/attention/gates.zig").stridedSigmoidGateMul;
const kvAttnFusedEligible = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").kvAttnFusedEligible;
const kvAttnVerifyEligible = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").kvAttnVerifyEligible;
const logKvAttnFusedEngaged = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").logKvAttnFusedEngaged;
const qkvAttnDecodeKernel = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").qkvAttnDecodeKernel;
const qkvAttnVerifyKernel = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").qkvAttnVerifyKernel;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const BATCHED_PER_SLOT_ATTN_MIN_KV = @import("../../../sdk/blocks/attention/batched.zig").BATCHED_PER_SLOT_ATTN_MIN_KV;
const sdpaForceFused = @import("../../../sdk/blocks/attention/sdpa.zig").sdpaForceFused;
const splitCausalSdpa = @import("../../../sdk/blocks/attention/sdpa.zig").splitCausalSdpa;

pub fn gatedFullAttnWith(
    self: *Trunk,
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
    self: *Trunk,
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
        const angles = self.qkAngleRowsFor(rope_family, rope_dims, rope_base, rope_freqs, eff_off, seq_len, rope_mscale, 1.0) catch break :blk;
        const none_v = mlx.mlx_array{ .ctx = null };
        const pair = (fusedQkNormRope256(self.s, queries, k_r, none_v, fa.q_norm, fa.k_norm, none_v, angles, self.rms_eps_arr, h_count, kv_h, seq_len, rope_dims) catch null) orelse break :blk;
        _ = mlx.mlx_array_free(q_rope);
        _ = mlx.mlx_array_free(k_rope);
        q_rope = pair[0];
        k_rope = pair[1];
        fused_qk = true;
    }
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

            if (try self.qsaBatchedAttn(ctx, slots, q_rope, layer, seq_len, attn_scale)) |gathered| {
                _ = mlx.mlx_array_free(attn_out_b);
                attn_out_b = gathered;
            } else {
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
                if (ctx.qsa_mask.ctx != null) {
                    // qwen4 QSA: the per-slot block selection (bool, false-padded to
                    // kv_max) narrows the pad mask. One dense-mask call serves the
                    // whole group, so it is billed to every slot: the tally is per REQUEST.
                    noteQsaArmForSlots(slots, .mask);
                    if (self.cost_trace_active) self.cost_attention |= 16;
                    attn_out_b = try self.stackedBatchedAttn(q_rope, dense_views, seq_len, attn_scale, ctx.qsa_mask);
                } else if (batchedAttnPerSlot(dense_views)) {
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
    if (ctx.qsa_blocks.ctx != null or ctx.qsa_mask.ctx != null) {
        // qwen4_exp QSA: the indexer already chose the visible blocks
        // (causal folded in). Prefill gathers them by index; decode/
        // verify widths (and a declined gather) run under the bool mask.
        const ratio: c_int = @intCast(self.config.indexer_compress_ratio);
        if (Qwen4AttnProf.active) {
            Qwen4AttnProf.sync(full_k);
            Qwen4AttnProf.sync(full_v);
            Qwen4AttnProf.lap(q_rope, .proj);
        }
        const served = try qsaSlotAttn(self.s, q_rope, &kv_view, ctx.qsa_blocks, &ctx.qsa_mask, &ctx.qsa_arms, seq_len, ratio, attn_scale);
        if (Qwen4AttnProf.active) Qwen4AttnProf.lap(served.out, .qsa);
        if (self.cost_trace_active) self.cost_attention |= served.cost_bit;
        _ = mlx.mlx_array_free(attn_out);
        attn_out = served.out;
        kv_fused_done = true;
    } else if (ctx.kv_attn_fused and kvAttnFusedEligible(&kv_view, seq_len)) {
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
    self: *Trunk,
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

/// Multi-row variant for the hd-256 fused path at verify widths: one
/// [rows, rd] table for positions offset..offset+rows-1, same per-forward
/// cache (every layer of one verify round shares the table).
///
/// `mscale` is the YaRN attention factor folded into the rows (1.0 = no
/// scaling). The kernels take cos|sin rows and nothing else, and YaRN's
/// mscale is one uniform scalar over the rotated slice, so folding it here
/// reaches both halves of every pair — the fused path needs no mscale slot
/// of its own. It is baked into the cached rows, which stay keyed by
/// family: a scaled model ropes in family 1, so a scaled row and an
/// unscaled one can never be served to each other.
pub fn qkAngleRowsFor(self: *Trunk, family: usize, rd: c_int, base: mlx.mlx_optional_float, freqs: mlx.mlx_array, offset: c_int, rows: c_int, mscale: f32, scale: f32) !mlx.mlx_array {
    const slot = &self.qk_angle_cache[family];
    if (slot.arr.ctx != null and slot.offset == offset and slot.rd == rd and slot.rows == rows and slot.scale == scale) return slot.arr;
    if (slot.arr.ctx != null) _ = mlx.mlx_array_free(slot.arr);
    slot.arr = .{ .ctx = null };
    slot.arr = try ropeAngleRows(self.s, rd, base, freqs, offset, rows, self.allocator, mscale, scale);
    slot.offset = offset;
    slot.rd = rd;
    slot.rows = rows;
    slot.scale = scale;
    return slot.arr;
}

/// The angle row for `offset` in the given rope family, rebuilt when the
/// offset moves (per decode token: 38-of-40 layers hit the cache).
pub fn qkAngleFor(self: *Trunk, family: usize, rd: c_int, base: mlx.mlx_optional_float, freqs: mlx.mlx_array, offset: c_int, mscale: f32) !mlx.mlx_array {
    return self.qkAngleRowsFor(family, rd, base, freqs, offset, 1, mscale, 1.0);
}

/// The build half of `ropeAtFreqs`: cos/sin `[n, rope_dims]` for positions `base + step*i`. `inv_freq` is taken.
pub fn ropeCosSinFromFreqs(self: *Trunk, rope_dims: c_int, inv_freq: mlx.mlx_array, base: f32, step: f32, n: c_int, dt: mlx.mlx_dtype) !MropeCosSin {
    defer _ = mlx.mlx_array_free(inv_freq);
    std.debug.assert(mlx.getShape(inv_freq)[0] == @divExact(rope_dims, 2));
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    try mlx.check(mlx.mlx_arange(&pos, base, base + step * @as(f32, @floatFromInt(n)) - step * 0.5, step, .float32, self.s)); // [n]
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos, &[_]c_int{ n, 1 }, 2, self.s));
    var ang = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ang);
    try mlx.check(mlx.mlx_multiply(&ang, pos2, inv_freq, self.s)); // [n, half]
    const parts = [_]mlx.mlx_array{ ang, ang };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var emb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(emb);
    try mlx.check(mlx.mlx_concatenate_axis(&emb, vec, 1, self.s)); // [n, rd]
    var cos_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cos_f);
    try mlx.check(mlx.mlx_cos(&cos_f, emb, self.s));
    var sin_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sin_f);
    try mlx.check(mlx.mlx_sin(&sin_f, emb, self.s));
    var cosv = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cosv);
    try mlx.check(mlx.mlx_astype(&cosv, cos_f, dt, self.s));
    var sinv = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sinv);
    try mlx.check(mlx.mlx_astype(&sinv, sin_f, dt, self.s));
    if (self.yarnActive()) {
        const ms = try scalarOf(self.config.yarn_attention_factor, dt, self.s);
        defer _ = mlx.mlx_array_free(ms);
        var cos_s = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&cos_s, cosv, ms, self.s));
        _ = mlx.mlx_array_free(cosv);
        cosv = cos_s;
        var sin_s = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&sin_s, sinv, ms, self.s));
        _ = mlx.mlx_array_free(sinv);
        sinv = sin_s;
    }
    return .{ .cos = cosv, .sin = sinv };
}

/// The `[half]` f32 inverse-frequency spectrum this trunk ropes with. Caller owns the result.
pub fn ropeInvFreq(self: *Trunk, rope_dims: c_int, theta: f32) !mlx.mlx_array {
    const half = @divExact(rope_dims, 2);
    // YaRN: the load-time host spectrum (f64 → the tensor dtype), the SAME
    // table attention and the M-RoPE tables rotate with. Deriving inv_freq
    // from `theta` in-graph below is the unscaled rope only.
    if (self.yarnActive()) {
        const yf = self.yarn_inv_freq.?;
        std.debug.assert(half == @as(c_int, @intCast(yf.len)));
        const f32s = try self.allocator.alloc(f32, yf.len);
        defer self.allocator.free(f32s);
        for (yf, 0..) |v, i| f32s[i] = @floatCast(v);
        const shape = [_]c_int{@intCast(yf.len)};
        return mlx.mlx_array_new_data(f32s.ptr, &shape, 1, .float32);
    }
    var j = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(j);
    try mlx.check(mlx.mlx_arange(&j, 0, @floatFromInt(half), 1.0, .float32, self.s));
    // inv_freq_j = theta^(-2j/rope_dims) = exp(-j * 2 ln(theta) / rope_dims)
    const neg = mlx.mlx_array_new_float(-2.0 * @log(theta) / @as(f32, @floatFromInt(rope_dims)));
    defer _ = mlx.mlx_array_free(neg);
    var jl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(jl);
    try mlx.check(mlx.mlx_multiply(&jl, j, neg, self.s));
    var inv_freq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_exp(&inv_freq, jl, self.s)); // [half]
    return inv_freq;
}

/// The apply half of `ropeAtFreqs`: rotate the first `rope_dims` dims with borrowed cos/sin.
pub fn ropeApplyCosSin(self: *Trunk, x: mlx.mlx_array, rope_dims: c_int, cosv: mlx.mlx_array, sinv: mlx.mlx_array) !mlx.mlx_array {
    const half = @divExact(rope_dims, 2);
    const sh = mlx.getShape(x);
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var xr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xr);
    try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], rope_dims }, 4, &strides, 4, self.s));
    var xp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xp);
    try mlx.check(mlx.mlx_slice(&xp, x, &[_]c_int{ 0, 0, 0, rope_dims }, 4, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, 4, &strides, 4, self.s));
    var x1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x1);
    try mlx.check(mlx.mlx_slice(&x1, xr, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], half }, 4, &strides, 4, self.s));
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_slice(&x2, xr, &[_]c_int{ 0, 0, 0, half }, 4, &[_]c_int{ sh[0], sh[1], sh[2], rope_dims }, 4, &strides, 4, self.s));
    var nx2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nx2);
    try mlx.check(mlx.mlx_negative(&nx2, x2, self.s));
    const rparts = [_]mlx.mlx_array{ nx2, x1 };
    const rvec = mlx.mlx_vector_array_new_data(&rparts, 2);
    defer _ = mlx.mlx_vector_array_free(rvec);
    var rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot);
    try mlx.check(mlx.mlx_concatenate_axis(&rot, rvec, -1, self.s));
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_multiply(&a, xr, cosv, self.s));
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_multiply(&b, rot, sinv, self.s));
    var roped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(roped);
    try mlx.check(mlx.mlx_add(&roped, a, b, self.s));
    const oparts = [_]mlx.mlx_array{ roped, xp };
    const ovec = mlx.mlx_vector_array_new_data(&oparts, 2);
    defer _ = mlx.mlx_vector_array_free(ovec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, ovec, -1, self.s));
    return out;
}

/// Apply NeoX RoPE to the first `rope_dims` dims of `arr` [B,H,S,hd] using
/// precomputed cos/sin [1,1,S,rope_dims] (broadcast over B,H); pass remaining
/// dims through. Equivalent to `mlx_fast_rope(traditional=false)` but with
/// per-token (M-RoPE) angles instead of a scalar offset.
pub fn applyMrope(self: *Trunk, arr: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, rope_dims: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(arr);
    const b = sh[0];
    const h = sh[1];
    const s = sh[2];
    const hd = sh[3];
    const half = @divExact(rope_dims, 2);
    const st = [_]c_int{ 1, 1, 1, 1 };

    var rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot);
    try mlx.check(mlx.mlx_slice(&rot, arr, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ b, h, s, rope_dims }, 4, &st, 4, self.s));

    // rotate_half(rot) = concat(-rot[..,half:], rot[..,:half])
    var r2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r2);
    try mlx.check(mlx.mlx_slice(&r2, rot, &[_]c_int{ 0, 0, 0, half }, 4, &[_]c_int{ b, h, s, rope_dims }, 4, &st, 4, self.s));
    var r1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r1);
    try mlx.check(mlx.mlx_slice(&r1, rot, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ b, h, s, half }, 4, &st, 4, self.s));
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, r2, self.s));
    const rh_arrs = [_]mlx.mlx_array{ neg, r1 };
    const rh_vec = mlx.mlx_vector_array_new_data(&rh_arrs, 2);
    defer _ = mlx.mlx_vector_array_free(rh_vec);
    var rh = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rh);
    try mlx.check(mlx.mlx_concatenate_axis(&rh, rh_vec, -1, self.s));

    var xcos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xcos);
    try mlx.check(mlx.mlx_multiply(&xcos, rot, cos, self.s));
    var rsin = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsin);
    try mlx.check(mlx.mlx_multiply(&rsin, rh, sin, self.s));
    var out_rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_rot);
    try mlx.check(mlx.mlx_add(&out_rot, xcos, rsin, self.s));

    if (hd == rope_dims) {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, out_rot, mlx.mlx_array_dtype(arr), self.s));
        return out;
    }
    var pass = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pass);
    try mlx.check(mlx.mlx_slice(&pass, arr, &[_]c_int{ 0, 0, 0, rope_dims }, 4, &[_]c_int{ b, h, s, hd }, 4, &st, 4, self.s));
    const cat_arrs = [_]mlx.mlx_array{ out_rot, pass };
    const cat_vec = mlx.mlx_vector_array_new_data(&cat_arrs, 2);
    defer _ = mlx.mlx_vector_array_free(cat_vec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, cat_vec, -1, self.s));
    return out;
}

/// Per-prefill-chunk M-RoPE cos/sin on `ctx.mrope_cos/sin_cur`, built once
/// per forward (freed by `endMropeChunk`). Forwards past the explicit
/// position table (decode, multi-token spec verify) leave them null and
/// take the scalar `offset + delta` path.
pub fn beginMropeChunk(self: *Trunk, ctx: *ForwardCtx, offset: usize, seq_len: usize, dtype: mlx.mlx_dtype) !void {
    if (ctx.mrope_pos == null or seq_len <= 1 or offset + seq_len > ctx.mrope_total) return;
    const cs = try self.mropeCosSinAt(mropeContext(ctx), offset, 1, seq_len, dtype);
    ctx.mrope_cos_cur = cs.cos;
    ctx.mrope_sin_cur = cs.sin;
}

/// cos/sin [1,1,n,rope_dims] at the `n` absolute positions `start +
/// stride*i`, in `dtype`. `stride 1` = a prefill chunk; the qwen4 QSA
/// indexer ropes pooled block keys at block-START positions (`stride =
/// ratio`) — the reference reads `full_cos[group_starts]`, which is 3-D
/// inside an image and `abs + delta` past the prompt, exactly like the
/// per-token table.
pub fn mropeCosSinAt(self: *Trunk, positions: mrope.PositionContext, start: usize, stride: usize, n: usize, dtype: mlx.mlx_dtype) !MropeCosSin {
    const cfg = &self.config;
    const rope_dims: usize = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
    const half = rope_dims / 2;

    const inv_freq = try self.allocator.alloc(f64, half);
    defer self.allocator.free(inv_freq);
    // One spectrum for every rope consumer: the YaRN table built at load
    // (when the config scales) or the plain theta^(-2j/d) one.
    if (self.yarnActive()) {
        const yf = self.yarn_inv_freq.?;
        std.debug.assert(yf.len == half);
        @memcpy(inv_freq, yf);
    } else mrope.computeInvFreq(inv_freq, rope_dims, cfg.rope_theta);
    const sel = try self.allocator.alloc(u8, half);
    defer self.allocator.free(sel);
    mrope.interleavedSelector(sel, cfg.mrope_section);

    const cos_buf = try self.allocator.alloc(f32, n * rope_dims);
    defer self.allocator.free(cos_buf);
    const sin_buf = try self.allocator.alloc(f32, n * rope_dims);
    defer self.allocator.free(sin_buf);
    mrope.fillCosSin(cos_buf, sin_buf, positions, start, stride, n, inv_freq, sel, rope_dims, yarnMscale(cfg));
    const shape = [_]c_int{ 1, 1, @intCast(n), @intCast(rope_dims) };
    const cf = mlx.mlx_array_new_data(cos_buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(cf);
    const sf = mlx.mlx_array_new_data(sin_buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(sf);
    var cos = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&cos, cf, dtype, self.s));
    errdefer _ = mlx.mlx_array_free(cos);
    var sin = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sin);
    try mlx.check(mlx.mlx_astype(&sin, sf, dtype, self.s));
    return .{ .cos = cos, .sin = sin };
}

/// Does this trunk rotate with a YaRN-scaled rope? The DECISION comes from
/// the config — every Trunk has one, including the hand-built fixtures
/// in the tests, which leave the optionals below undefined. The short-circuit
/// means an unscaled model never reads them.
pub fn yarnActive(self: *const Trunk) bool {
    return self.config.rope_yarn and self.rope_freqs_yarn != null;
}

/// YaRN mscale on a rope'd array whose last dim may not be `head_dim`
/// (the QSA indexer is 128-wide / 64 rotated). Slice the first `rope_dims`,
/// multiply by a same-dtype scalar, concat the pass-through tail. NEVER
/// `yarnScaleQK` here: that table is `[head_dim]` (256 on qwen4).
pub fn yarnScaleRotated(self: *Trunk, q: *mlx.mlx_array, rope_dims: c_int) !void {
    const sh = mlx.getShape(q.*);
    const dt = mlx.mlx_array_dtype(q.*);
    const st = [_]c_int{ 1, 1, 1, 1 };
    var rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot);
    try mlx.check(mlx.mlx_slice(&rot, q.*, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], rope_dims }, 4, &st, 4, self.s));
    var pass = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pass);
    try mlx.check(mlx.mlx_slice(&pass, q.*, &[_]c_int{ 0, 0, 0, rope_dims }, 4, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, 4, &st, 4, self.s));
    const ms = try scalarOf(self.config.yarn_attention_factor, dt, self.s);
    defer _ = mlx.mlx_array_free(ms);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, rot, ms, self.s));
    const parts = [_]mlx.mlx_array{ scaled, pass };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, -1, self.s));
    _ = mlx.mlx_array_free(q.*);
    q.* = out;
}

/// YaRN mscale on rope'd q and k: multiply by the [head_dim] table
/// (attention_factor on the rotated slice, 1.0 on the pass-through tail).
/// That is the reference's `cos, sin = cos*af, sin*af` — every rotary pair
/// shares one frequency and both of its dims carry it, so scaling the
/// rotation and scaling q/k are the same product. The table MUST be read in
/// the tensors' dtype: an f32 operand promotes the result, and the f32 then
/// rides SDPA → o_proj → residual, upcasting every weight read downstream.
pub fn yarnScaleQK(self: *Trunk, q: *mlx.mlx_array, k: *mlx.mlx_array) !void {
    const ms = try constTableAs(self.yarn_mscale.?, mlx.mlx_array_dtype(q.*), &self.yarn_mscale_cast, self.s);
    var qs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&qs, q.*, ms, self.s));
    _ = mlx.mlx_array_free(q.*);
    q.* = qs;
    var ks = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&ks, k.*, ms, self.s));
    _ = mlx.mlx_array_free(k.*);
    k.* = ks;
}

/// The stacked batched read: every slot's view padded to the longest and concatenated,
/// one array-mask SDPA for the group. `narrow` (bool, null-ctx for none) removes more
/// positions from the pad mask.
pub fn stackedBatchedAttn(
    self: *const Trunk,
    q: mlx.mlx_array,
    views: []const DenseKVView,
    seq_len: c_int,
    attn_scale: f32,
    narrow: mlx.mlx_array,
) !mlx.mlx_array {
    var kv_len_buf: [MAX_BATCH_ROWS]i32 = undefined;
    var kv_max: c_int = 0;
    for (views, 0..) |dv, i| {
        kv_len_buf[i] = mlx.getShape(dv.k)[2];
        kv_max = @max(kv_max, kv_len_buf[i]);
    }
    const stacked_k = try self.padAndStackBatchedKV(views, true, kv_max);
    defer _ = mlx.mlx_array_free(stacked_k);
    const stacked_v = try self.padAndStackBatchedKV(views, false, kv_max);
    defer _ = mlx.mlx_array_free(stacked_v);
    var mask = try self.buildBatchedDecodeMask(kv_len_buf[0..views.len], kv_max, seq_len, mlx.mlx_array_dtype(q));
    defer _ = mlx.mlx_array_free(mask);
    if (narrow.ctx != null) {
        const neg_inf = try scalarOf(-std.math.inf(f32), mlx.mlx_array_dtype(mask), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var narrowed = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&narrowed, narrow, mask, neg_inf, self.s));
        _ = mlx.mlx_array_free(mask);
        mask = narrowed;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, stacked_k, stacked_v, attn_scale, "array", mask, .{ .ctx = null }, false, self.s));
    return out;
}

/// `[N, 1, S, kv_max]` additive mask: row j of slot n sees positions below
/// `kv_len_n - S + 1 + j` (its own causal prefix; the pad past `kv_len_n` never).
/// Additive [N, 1, S, kv_max] mask in `dt`, the queries' dtype: sdpa
/// rejects a mask that does not promote to its output type.
pub fn buildBatchedDecodeMask(
    self: *const Trunk,
    kv_lens: []const i32,
    kv_max: c_int,
    seq_len: c_int,
    dt: mlx.mlx_dtype,
) !mlx.mlx_array {
    const N: c_int = @intCast(kv_lens.len);
    var positions = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(positions);
    try mlx.check(mlx.mlx_arange(&positions, 0, @floatFromInt(kv_max), 1, .int32, self.s));
    const pos_shape = [_]c_int{ 1, 1, 1, kv_max };
    var pos_4d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_4d);
    try mlx.check(mlx.mlx_reshape(&pos_4d, positions, &pos_shape, 4, self.s));

    // limit[n, 0, j, 0] = kv_len_n - S + 1 + j
    var lim_buf: [MAX_BATCH_ROWS * 16]i32 = undefined;
    if (kv_lens.len * @as(usize, @intCast(seq_len)) > lim_buf.len) return error.BatchedMaskTooWide;
    for (kv_lens, 0..) |kl, n| {
        for (0..@intCast(seq_len)) |j| lim_buf[n * @as(usize, @intCast(seq_len)) + j] = kl - seq_len + 1 + @as(i32, @intCast(j));
    }
    const lim_shape = [_]c_int{ N, 1, seq_len, 1 };
    const lens_4d = mlx.mlx_array_new_data(lim_buf[0 .. kv_lens.len * @as(usize, @intCast(seq_len))].ptr, &lim_shape, 4, .int32);
    defer _ = mlx.mlx_array_free(lens_4d);

    var valid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_less(&valid, pos_4d, lens_4d, self.s));

    const zero = try scalarOf(0.0, dt, self.s);
    defer _ = mlx.mlx_array_free(zero);
    const neg_inf = try scalarOf(-std.math.inf(f32), dt, self.s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&mask, valid, zero, neg_inf, self.s));
    return mask;
}

pub fn padAndStackBatchedKV(
    self: *const Trunk,
    views: []const DenseKVView,
    key_not_value: bool,
    kv_max: c_int,
) !mlx.mlx_array {
    const pad_axes = [_]c_int{2};
    const pad_value = bf16Scalar(0.0, self.s);
    defer _ = mlx.mlx_array_free(pad_value);

    const padded_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(padded_vec);
    // Track padded arrays so we can free them after the concat.
    var padded_arrs = try self.allocator.alloc(mlx.mlx_array, views.len);
    defer {
        for (padded_arrs) |a| _ = mlx.mlx_array_free(a);
        self.allocator.free(padded_arrs);
    }

    for (views, 0..) |dv, i| {
        const view = if (key_not_value) dv.k else dv.v;
        const vshape = mlx.getShape(view);
        const klen: c_int = vshape[2];
        const high_pad: c_int = kv_max - klen;
        const low_pad_arr = [_]c_int{0};
        const high_pad_arr = [_]c_int{high_pad};
        padded_arrs[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_pad(
            &padded_arrs[i],
            view,
            &pad_axes,
            1,
            &low_pad_arr,
            1,
            &high_pad_arr,
            1,
            pad_value,
            "constant",
            self.s,
        ));
        _ = mlx.mlx_vector_array_append_value(padded_vec, padded_arrs[i]);
    }

    var stacked = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, padded_vec, 0, self.s));
    return stacked;
}

/// The per-slot batched read: each slot's query rows attend its OWN view, so nothing is
/// padded, copied or masked, and the causal and quantized-KV kernels serve it as they
/// serve a solo slot. `fused[i]` is slot i's `ForwardCtx.kv_attn_fused`.
pub fn perSlotBatchedAttn(
    self: *const Trunk,
    q: mlx.mlx_array,
    views: []const DenseKVView,
    fused: []const bool,
    seq_len: c_int,
    attn_scale: f32,
) !mlx.mlx_array {
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    var outs: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (outs[0..n]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    if (!layered.per_slot_attn_logged) {
        layered.per_slot_attn_logged = true;
        log.info("[batched] per-slot attention engaged (slots={d}, rows={d}): no KV pad or stack past {d} tokens\n", .{ views.len, seq_len, BATCHED_PER_SLOT_ATTN_MIN_KV });
    }
    for (views, 0..) |*dv, i| {
        const q_i = try axisView(self.s, q, 0, i);
        defer _ = mlx.mlx_array_free(q_i);
        const mode: [:0]const u8 = if (seq_len > 1) "causal" else "";
        var out: ?mlx.mlx_array = null;
        if (fused[i] and kvAttnFusedEligible(dv, seq_len)) {
            out = try qkvAttnDecodeKernel(self.s, q_i, dv, attn_scale, mode, none_mask);
            if (out != null) logKvAttnFusedEngaged(dv, q_i, seq_len);
        } else if (fused[i] and kvAttnVerifyEligible(dv, seq_len)) {
            out = try qkvAttnVerifyKernel(self.s, q_i, dv, attn_scale, mode);
        }
        if (out == null and seq_len > 1) {
            out = try fusedSdpa256Prefill(self.s, q_i, dv.k, dv.v, attn_scale, 0);
            if (out == null) out = try splitCausalSdpa(self.s, q_i, dv.k, dv.v, attn_scale);
        }
        if (out == null) {
            var o = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(o);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q_i, dv.k, dv.v, attn_scale, mode.ptr, none_mask, .{ .ctx = null }, seq_len > 1 and sdpaForceFused(q_i, dv.k), self.s));
            out = o;
        }
        outs[n] = out.?;
        n += 1;
    }
    const vec = mlx.mlx_vector_array_new_data(&outs, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var joined = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(joined);
    try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 0, self.s));
    return joined;
}

/// Partial neox RoPE of `x [B,1,N,hd]` at positions `base + step*i` (the QSA rope cache's oracle).
/// copied_from: engines/mlx/methods/rope.zig (`ropeAtPositions`).
pub fn ropeAtPositions(self: *Trunk, x: mlx.mlx_array, rope_dims: c_int, theta: f32, base: f32, step: f32, n: c_int) !mlx.mlx_array {
    const inv_freq = try self.ropeInvFreq(rope_dims, theta);
    const cs = try self.ropeCosSinFromFreqs(rope_dims, inv_freq, base, step, n, mlx.mlx_array_dtype(x));
    defer _ = mlx.mlx_array_free(cs.cos);
    defer _ = mlx.mlx_array_free(cs.sin);
    return self.ropeApplyCosSin(x, rope_dims, cs.cos, cs.sin);
}
