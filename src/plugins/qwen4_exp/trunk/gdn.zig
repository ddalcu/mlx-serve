//! The trunk's GatedDeltaNet layer over a slot's recurrent entry: the joined or split projections, the fused decode
//! and verify recurrences (with the per-row state capture a partial accept rolls back to), the fused or composed
//! prefill prework, the recurrence kernels, and the sigmoid-gated norm (qwen4_exp's `output_gate_type`).
//! copied_from: sdk/blocks/gdn/{gated_delta_net,gate}.zig, sdk/blocks/conv/conv1d_cache.zig (the qwen4_exp arms: no
//! combined projection, KDA, low-rank gates, Hadamard, ANE or spec-tree paths).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const gdn_decode = @import("../../../sdk/blocks/gdn/decode.zig");
const decode_fused = @import("../../../sdk/blocks/gdn/decode_fused.zig");
const checkpoint = @import("../../../engines/mlx/ssm/checkpoint.zig");
const Trunk = @import("../trunk.zig").Trunk;
const LinearAttnWeights = @import("../../../sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const GdnProj = @import("../../../sdk/blocks/gdn/gated_delta_net.zig").GdnProj;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const GdnVerifyInputs = @import("verify.zig").GdnVerifyInputs;
const PREFILL_EVAL_MIN_SEQ = @import("../../../engines/mlx/eval_cadence.zig").PREFILL_EVAL_MIN_SEQ;
const compactConvStateEnabled = checkpoint.compactConvStateEnabled;
const scalarOf = @import("../../../engines/mlx/load/weights.zig").scalarOf;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const qwen4Standin = @import("../../../sdk/diag/standin.zig").qwen4Standin;
const standinOnes = @import("../../../sdk/diag/standin.zig").standinOnes;
const standinRef = @import("../../../sdk/diag/standin.zig").standinRef;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const depthwiseConvCuda = @import("../../../sdk/blocks/conv/depthwise.zig").depthwiseConvCuda;
const depthwiseConvStep = @import("../../../sdk/blocks/conv/depthwise.zig").depthwiseConvStep;
const gdnGateChain = @import("../../../sdk/blocks/gdn/gate.zig").gdnGateChain;
const gdnChainGateDtype = @import("../../../sdk/blocks/gdn/gate.zig").gdnChainGateDtype;
const GDN_FUSED_MAX_ROWS = decode_fused.GDN_FUSED_MAX_ROWS;
const GdnPrework = decode_fused.GdnPrework;
const gdnDecodeFusedEnabled = decode_fused.gdnDecodeFusedEnabled;
const gdnDecodeRecurEnabled = decode_fused.gdnDecodeRecurEnabled;
const gdnNormGateFused = decode_fused.gdnNormGateFused;
const gdnPrefillFusedFor = decode_fused.gdnPrefillFusedFor;
const gdnPreworkFused = decode_fused.gdnPreworkFused;
const gdnVerifyFoldEnabled = decode_fused.gdnVerifyFoldEnabled;
const kernels = @import("../../../sdk/blocks/gdn/kernels.zig");
const GdnRoute = kernels.GdnRoute;
const GDN_BLOCK_T = kernels.GDN_BLOCK_T;
const gdnBlockTFor = kernels.gdnBlockTFor;
const gdnBlockedEligible = kernels.gdnBlockedEligible;
const gdnBlockedEnabled = kernels.gdnBlockedEnabled;
const gdnPipelinedEnabled = kernels.gdnPipelinedEnabled;
const gdnRecurCuda = kernels.gdnRecurCuda;
const gdnRecurOps = kernels.gdnRecurOps;
const gdnRouteKernel = kernels.gdnRouteKernel;
const gdnSetGrid = kernels.gdnSetGrid;
const getGdnKernelSeq = kernels.getGdnKernelSeq;

pub fn gatedDeltaNet(
    self: *Trunk,
    x: mlx.mlx_array,
    la: *const LinearAttnWeights,
    ssm: *SSMCacheEntry,
    layer_idx: usize,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
) !mlx.mlx_array {
    return self.gatedDeltaNetProjected(x, la, ssm, layer_idx, batch, seq_len, is_prefill, null, false);
}

/// Takes ownership of supplied projection slices; recurrent state remains per request.
pub fn gatedDeltaNetProjected(
    self: *Trunk,
    x: mlx.mlx_array,
    la: *const LinearAttnWeights,
    ssm: *SSMCacheEntry,
    layer_idx: usize,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
    projected: ?GdnVerifyInputs,
    skip_output: bool,
) !mlx.mlx_array {
    const cfg = &self.config;
    const num_k_heads: c_int = @intCast(cfg.linear_num_key_heads);
    const num_v_heads: c_int = @intCast(cfg.linear_num_value_heads);
    const dk: c_int = @intCast(cfg.linear_key_head_dim);
    const dv: c_int = @intCast(cfg.linear_value_head_dim);
    const key_dim: c_int = dk * num_k_heads;
    const value_dim: c_int = dv * num_v_heads;
    const conv_dim: c_int = key_dim * 2 + value_dim;
    const kernel: c_int = @intCast(cfg.linear_conv_kernel_dim);

    // Projections: combined (qkvz+ba) or separate (qkv+z+a+b)
    var qkv: mlx.mlx_array = undefined;
    var z_proj: mlx.mlx_array = undefined;
    // [qkv | z | a | b] from one joined matmul: the verify kernels read its columns in place.
    var in_all: mlx.mlx_array = .{ .ctx = null };
    defer if (in_all.ctx != null) {
        _ = mlx.mlx_array_free(in_all);
    };
    var a_proj: mlx.mlx_array = undefined;
    var b_proj: mlx.mlx_array = undefined;
    defer _ = mlx.mlx_array_free(qkv);
    defer _ = mlx.mlx_array_free(z_proj);
    defer _ = mlx.mlx_array_free(a_proj);
    defer _ = mlx.mlx_array_free(b_proj);
    if (projected) |values| {
        qkv = values.qkv;
        z_proj = values.z;
        a_proj = values.a;
        b_proj = values.b;
    } else if (la.combined_proj) {
        // Combined QKVZ: output is interleaved by key-head groups.
        // Reshape to [B, S, nk, per_head], split per-head into q/k/v/z, then flatten back.
        const vph = @divExact(num_v_heads, num_k_heads); // value heads per key head group
        const qkvz_raw = try self.qmatmul(x, la.qkv_w, la.qkv_s, la.qkv_b);
        defer _ = mlx.mlx_array_free(qkvz_raw);
        const per_head = dk + dk + vph * dv + vph * dv;
        const gh_shape = [_]c_int{ batch, seq_len, num_k_heads, per_head };
        var qkvz_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qkvz_g);
        try mlx.check(mlx.mlx_reshape(&qkvz_g, qkvz_raw, &gh_shape, 4, self.s));

        const strides4 = [_]c_int{ 1, 1, 1, 1 };
        // q: [B,S,nk,dk]
        var q_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_g);
        try mlx.check(mlx.mlx_slice(&q_g, qkvz_g, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ batch, seq_len, num_k_heads, dk }, 4, &strides4, 4, self.s));
        // k: [B,S,nk,dk]
        var k_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_g);
        try mlx.check(mlx.mlx_slice(&k_g, qkvz_g, &[_]c_int{ 0, 0, 0, dk }, 4, &[_]c_int{ batch, seq_len, num_k_heads, dk * 2 }, 4, &strides4, 4, self.s));
        // v: [B,S,nk,vph*dv]
        const v_off = dk * 2;
        const v_end = v_off + vph * dv;
        var v_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_g);
        try mlx.check(mlx.mlx_slice(&v_g, qkvz_g, &[_]c_int{ 0, 0, 0, v_off }, 4, &[_]c_int{ batch, seq_len, num_k_heads, v_end }, 4, &strides4, 4, self.s));
        // z: [B,S,nk,vph*dv]
        var z_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(z_g);
        try mlx.check(mlx.mlx_slice(&z_g, qkvz_g, &[_]c_int{ 0, 0, 0, v_end }, 4, &[_]c_int{ batch, seq_len, num_k_heads, per_head }, 4, &strides4, 4, self.s));

        // Flatten: q/k -> [B,S,key_dim], v/z -> [B,S,value_dim]
        const flat3_qk = [_]c_int{ batch, seq_len, key_dim };
        const flat3_vz = [_]c_int{ batch, seq_len, value_dim };
        var q_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_flat);
        var k_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_flat);
        var v_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_flat);
        try mlx.check(mlx.mlx_reshape(&q_flat, q_g, &flat3_qk, 3, self.s));
        try mlx.check(mlx.mlx_reshape(&k_flat, k_g, &flat3_qk, 3, self.s));
        try mlx.check(mlx.mlx_reshape(&v_flat, v_g, &flat3_vz, 3, self.s));
        z_proj = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&z_proj, z_g, &flat3_vz, 3, self.s));

        // Concatenate [q, k, v] -> qkv [B,S,conv_dim]
        const qkv_arr = [_]mlx.mlx_array{ q_flat, k_flat, v_flat };
        const qkv_vec = mlx.mlx_vector_array_new_data(&qkv_arr, 3);
        defer _ = mlx.mlx_vector_array_free(qkv_vec);
        qkv = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&qkv, qkv_vec, 2, self.s));

        // Combined BA: interleaved by key-head groups
        const ba_raw = try self.qmatmul(x, la.b_w, la.b_s, la.b_b);
        defer _ = mlx.mlx_array_free(ba_raw);
        const ba_per_head = vph * 2;
        const ba_shape = [_]c_int{ batch, seq_len, num_k_heads, ba_per_head };
        var ba_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ba_g);
        try mlx.check(mlx.mlx_reshape(&ba_g, ba_raw, &ba_shape, 4, self.s));
        var b_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b_g);
        var a_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a_g);
        try mlx.check(mlx.mlx_slice(&b_g, ba_g, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ batch, seq_len, num_k_heads, vph }, 4, &strides4, 4, self.s));
        try mlx.check(mlx.mlx_slice(&a_g, ba_g, &[_]c_int{ 0, 0, 0, vph }, 4, &[_]c_int{ batch, seq_len, num_k_heads, ba_per_head }, 4, &strides4, 4, self.s));
        const flat3_ba = [_]c_int{ batch, seq_len, num_v_heads };
        b_proj = mlx.mlx_array_new();
        a_proj = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&b_proj, b_g, &flat3_ba, 3, self.s));
        try mlx.check(mlx.mlx_reshape(&a_proj, a_g, &flat3_ba, 3, self.s));
    } else if (qwen4Standin().gdn_proj) {
        qkv = try standinOnes(&[_]c_int{ batch, seq_len, conv_dim }, self.s);
        z_proj = try standinOnes(&[_]c_int{ batch, seq_len, value_dim }, self.s);
        a_proj = try standinOnes(&[_]c_int{ batch, seq_len, num_v_heads }, self.s);
        b_proj = try standinOnes(&[_]c_int{ batch, seq_len, num_v_heads }, self.s);
    } else if (la.in.w.ctx != null and self.rowGroupServes(batch * seq_len)) {
        const all = try self.qmatmul(x, la.in.w, la.in.s, la.in.b);
        in_all = all;
        qkv = mlx.mlx_array_new();
        z_proj = mlx.mlx_array_new();
        a_proj = mlx.mlx_array_new();
        b_proj = mlx.mlx_array_new();
        try la.in.part(&qkv, all, 0, self.s);
        try la.in.part(&z_proj, all, 1, self.s);
        try la.in.part(&a_proj, all, 2, self.s);
        try la.in.part(&b_proj, all, 3, self.s);
    } else {
        const proj = try self.gdnProjGpu(x, la);
        qkv = proj.qkv;
        z_proj = proj.z;
        a_proj = try self.qmatmul(x, la.a_w, la.a_s, la.a_b);
        b_proj = try self.qmatmul(x, la.b_w, la.b_s, la.b_b);
    }
    // Scale scalars + the parameter-free rms_norm ones-weight (mlx-c
    // requires a non-empty weight) are cached on the Trunk — they
    // used to be rebuilt every layer every decode step.
    if (self.gdn_q_scale == null) {
        const inv_scale = 1.0 / @as(f32, @floatFromInt(cfg.linear_key_head_dim));
        self.gdn_q_scale = try scalarOf(inv_scale, mlx.mlx_array_dtype(qkv), self.s);
        self.gdn_k_scale = try scalarOf(@sqrt(inv_scale), mlx.mlx_array_dtype(qkv), self.s);
    }
    const inv_scale_sq = self.gdn_q_scale.?;
    const inv_sqrt_sc = self.gdn_k_scale.?;

    // Other packs, plain decode: prework + recurrence in one dispatch, then
    // the norm-gate epilogue.
    if (batch == 1 and seq_len == 1 and projected == null and
        !self.spec_capture_ssm and ssm.initialized and ssm.ssm_state.ctx != null and kernel == 4 and gdnDecodeRecurEnabled())
    fast: {
        if (self.gdn_eps == null) self.gdn_eps = mlx.mlx_array_new_float(cfg.rms_norm_eps);
        const r = (try gdn_decode.recur(.{ .hk = num_k_heads, .hv = num_v_heads, .dk = dk, .dv = dv }, .{
            .qkv = qkv,
            .z = z_proj,
            .a = a_proj,
            .b = b_proj,
            .conv_state = ssm.conv_state,
            .ssm_state = ssm.ssm_state,
            .conv_w = la.conv1d_w,
            .A_log = la.A_log,
            .dt_bias = la.dt_bias,
            .q_scale = inv_scale_sq,
            .k_scale = inv_sqrt_sc,
            .norm_w = la.norm_w,
            .eps = self.gdn_eps.?,
            .signs = .{ .ctx = null },
        }, self.s)) orelse break :fast;
        defer _ = mlx.mlx_array_free(r.y);
        const flat = (try gdnNormGateFused(self.s, r.y, z_proj, 0, value_dim, la.norm_w, self.gdn_eps.?, false, num_v_heads, dv, 1, 1)) orelse {
            _ = mlx.mlx_array_free(r.conv_state);
            _ = mlx.mlx_array_free(r.ssm_state);
            break :fast;
        };
        defer _ = mlx.mlx_array_free(flat);
        _ = mlx.mlx_array_free(ssm.conv_state);
        ssm.conv_state = r.conv_state;
        _ = mlx.mlx_array_free(ssm.ssm_state);
        ssm.ssm_state = r.ssm_state;
        if (!decode_fused.gdn_decode_recur_engaged) {
            decode_fused.gdn_decode_recur_engaged = true;
            log.info("[gdn] decode recur engaged: Hk={d} Hv={d} gate={s}\n", .{ num_k_heads, num_v_heads, "sigmoid" });
        }
        return if (skip_output) standinRef(flat) else self.qmatmul(flat, la.out_w, la.out_s, la.out_b);
    }

    // MTP verify rows (spec capture, 2 <= S <= 8): the same kernel over S
    // tokens with the per-step state capture rollback reads, then the norm-gate.
    // A grouped verify's `projected` slices ride it too, or its rows round unlike a solo verify's.
    if (batch == 1 and seq_len >= 1 and seq_len <= gdn_decode.MAX_SEQ and
        self.spec_capture_ssm and ssm.initialized and ssm.ssm_state.ctx != null and kernel == 4 and gdnDecodeRecurEnabled())
    fast: {
        if (self.gdn_eps == null) self.gdn_eps = mlx.mlx_array_new_float(cfg.rms_norm_eps);
        const geo = gdn_decode.Geometry{ .hk = num_k_heads, .hv = num_v_heads, .dk = dk, .dv = dv };
        const ins = gdn_decode.Inputs{
            .qkv = qkv,
            .z = z_proj,
            .a = a_proj,
            .b = b_proj,
            .conv_state = ssm.conv_state,
            .ssm_state = ssm.ssm_state,
            .conv_w = la.conv1d_w,
            .A_log = la.A_log,
            .dt_bias = la.dt_bias,
            .q_scale = inv_scale_sq,
            .k_scale = inv_sqrt_sc,
            .norm_w = la.norm_w,
            .eps = self.gdn_eps.?,
            .signs = .{ .ctx = null },
        };
        // A chain in one dispatch: recurrence, norm-gate and the rollback conv input.
        if (gdnVerifyFoldEnabled()) fold: {
            const f = (try gdn_decode.recurSeqFold(geo, seq_len, ins, false, self.s)) orelse break :fold;
            defer _ = mlx.mlx_array_free(f.gated);
            if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
            ssm.spec_conv_input = f.conv_input;
            if (ssm.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(ssm.spec_state_seq);
            ssm.spec_state_seq = f.state_seq;
            _ = mlx.mlx_array_free(ssm.conv_state);
            ssm.conv_state = f.conv_state;
            _ = mlx.mlx_array_free(ssm.ssm_state);
            ssm.ssm_state = f.ssm_state;
            if (!decode_fused.gdn_verify_fold_engaged) {
                decode_fused.gdn_verify_fold_engaged = true;
                log.info("[gdn] verify fold engaged: S={d} Hk={d} Hv={d}\n", .{ seq_len, num_k_heads, num_v_heads });
            }
            return if (skip_output) standinRef(f.gated) else self.qmatmul(f.gated, la.out_w, la.out_s, la.out_b);
        }
        const r = (try gdn_decode.recurSeq(geo, seq_len, ins, self.s)) orelse break :fast;
        defer _ = mlx.mlx_array_free(r.y);
        const flat = (try gdnNormGateFused(self.s, r.y, z_proj, 0, value_dim, la.norm_w, self.gdn_eps.?, false, num_v_heads, dv, 1, seq_len)) orelse {
            inline for (.{ r.conv_state, r.ssm_state, r.state_seq }) |a| _ = mlx.mlx_array_free(a);
            break :fast;
        };
        defer _ = mlx.mlx_array_free(flat);
        // Rollback slices the conv INPUT, as the composed path stashes it.
        var conv_input = mlx.mlx_array_new();
        {
            const arr = [_]mlx.mlx_array{ ssm.conv_state, qkv };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s)) catch |e| {
                inline for (.{ conv_input, r.conv_state, r.ssm_state, r.state_seq }) |a| _ = mlx.mlx_array_free(a);
                return e;
            };
        }
        if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
        ssm.spec_conv_input = conv_input;
        if (ssm.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(ssm.spec_state_seq);
        ssm.spec_state_seq = r.state_seq;
        _ = mlx.mlx_array_free(ssm.conv_state);
        ssm.conv_state = r.conv_state;
        _ = mlx.mlx_array_free(ssm.ssm_state);
        ssm.ssm_state = r.ssm_state;
        if (!decode_fused.gdn_verify_recur_engaged) {
            decode_fused.gdn_verify_recur_engaged = true;
            log.info("[gdn] verify recur engaged: S={d} Hk={d} Hv={d}\n", .{ seq_len, num_k_heads, num_v_heads });
        }
        return if (skip_output) standinRef(flat) else self.qmatmul(flat, la.out_w, la.out_s, la.out_b);
    }

    var q_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_scaled);
    var k_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_scaled);
    var v_heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_heads);
    var fused_prework = false;
    var g_fused: ?mlx.mlx_array = null;
    var beta_fused: ?mlx.mlx_array = null;

    // Wide prework is token-local; a cold chunk supplies the same zero history as conv1dWithCache.
    const prefill_fused = is_prefill and gdnPrefillFusedFor(seq_len, batch);
    if ((batch * seq_len <= GDN_FUSED_MAX_ROWS or prefill_fused) and
        kernel == 4 and (ssm.initialized or prefill_fused))
    blk: {
        var zero_conv = mlx.mlx_array{ .ctx = null };
        defer if (zero_conv.ctx != null) {
            _ = mlx.mlx_array_free(zero_conv);
        };
        const cold = !ssm.initialized;
        if (cold) {
            zero_conv = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&zero_conv, &[_]c_int{ batch, 3, conv_dim }, 3, mlx.mlx_array_dtype(qkv), self.s));
        }
        const incoming_conv = if (cold) zero_conv else ssm.conv_state;
        const pre = (gdnPreworkFused(self.s, .{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = conv_dim,
            .b = b_proj,
            .b_off = 0,
            .b_stride = num_v_heads,
            .a = a_proj,
            .a_off = 0,
            .a_stride = num_v_heads,
            .A_log = la.A_log,
            .dt_bias = la.dt_bias,
            .conv_state = incoming_conv,
            .conv_w = la.conv1d_w,
            .q_scale = inv_scale_sq,
            .k_scale = inv_sqrt_sc,
            .qk_eps = cfg.gdnQkNormEps(),
            .hk = num_k_heads,
            .hv = num_v_heads,
            .dk = dk,
            .dv = dv,
            .seq = seq_len,
            .batch = batch,
            .gate = true,
            .g_dtype = gdn_decode.gateDtype(mlx.mlx_array_dtype(qkv), self.ssmStateDtype()),
        }) catch null) orelse break :blk;
        // Spec-capture rollback slices the conv INPUT; build the concat
        // the composed path would have stashed (lazy, one launch).
        if (self.spec_capture_ssm) {
            const arr = [_]mlx.mlx_array{ incoming_conv, qkv };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
            ssm.spec_conv_input = mlx.mlx_array_new();
            mlx.check(mlx.mlx_concatenate_axis(&ssm.spec_conv_input, vec, 1, self.s)) catch {
                // Free the fused outputs and fall back to the composed
                // chain rather than adopt state with no rollback capture.
                _ = mlx.mlx_array_free(pre.q);
                _ = mlx.mlx_array_free(pre.k);
                _ = mlx.mlx_array_free(pre.v);
                _ = mlx.mlx_array_free(pre.conv_state);
                _ = mlx.mlx_array_free(pre.g);
                _ = mlx.mlx_array_free(pre.beta);
                break :blk;
            };
        }
        if (env.on(.QWEN4_GDN_CHECK)) self.gdnCheckFused(la, qkv, incoming_conv, b_proj, a_proj, &pre, layer_idx, num_k_heads, num_v_heads, dk, dv, batch, seq_len) catch |e| log.warn("[gdn-check] failed: {s}\n", .{@errorName(e)});
        _ = mlx.mlx_array_free(ssm.conv_state);
        ssm.conv_state = pre.conv_state;
        ssm.initialized = true;
        if (prefill_fused and seq_len >= 512 and !decode_fused.gdn_prefill_engaged) {
            decode_fused.gdn_prefill_engaged = true;
            log.info("[gdn-prefill] engaged: S={d} B={d} cold={}\n", .{ seq_len, batch, cold });
        }
        _ = mlx.mlx_array_free(q_scaled);
        _ = mlx.mlx_array_free(k_scaled);
        _ = mlx.mlx_array_free(v_heads);
        q_scaled = pre.q;
        k_scaled = pre.k;
        v_heads = pre.v;
        fused_prework = true;
        if (gdnDecodeFusedEnabled()) {
            g_fused = pre.g;
            beta_fused = pre.beta;
        } else {
            _ = mlx.mlx_array_free(pre.g);
            _ = mlx.mlx_array_free(pre.beta);
        }
    }

    if (!fused_prework) {
        // Conv1d with cache: prepend conv_state, apply depthwise conv + silu
        const conv_out = try self.conv1dWithCache(qkv, la.conv1d_w, null, ssm, batch, conv_dim, kernel, true);
        defer _ = mlx.mlx_array_free(conv_out);

        // Split conv output into Q, K, V
        // Q: [B, S, key_dim] → [B, S, num_k_heads, dk]
        // K: [B, S, key_dim] → [B, S, num_k_heads, dk]
        // V: [B, S, value_dim] → [B, S, num_v_heads, dv]
        const strides3 = [_]c_int{ 1, 1, 1 };
        var q_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_flat);
        {
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ batch, seq_len, key_dim };
            try mlx.check(mlx.mlx_slice(&q_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, self.s));
        }
        var k_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_flat);
        {
            const start = [_]c_int{ 0, 0, key_dim };
            const stop = [_]c_int{ batch, seq_len, key_dim * 2 };
            try mlx.check(mlx.mlx_slice(&k_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, self.s));
        }
        var v_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_flat);
        {
            const start = [_]c_int{ 0, 0, key_dim * 2 };
            const stop = [_]c_int{ batch, seq_len, key_dim * 2 + value_dim };
            try mlx.check(mlx.mlx_slice(&v_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, self.s));
        }

        // Reshape to head dims
        const q_shape = [_]c_int{ batch, seq_len, num_k_heads, dk };
        const k_shape = [_]c_int{ batch, seq_len, num_k_heads, dk };
        const v_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };
        var q_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_heads);
        var k_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_heads);
        try mlx.check(mlx.mlx_reshape(&q_heads, q_flat, &q_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&k_heads, k_flat, &k_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&v_heads, v_flat, &v_shape, 4, self.s));

        // Q/K normalization: q = (1/dk) * rms_norm(q, null), k = (1/sqrt(dk)) * rms_norm(k, null)
        if (self.gdn_ones_w == null) {
            const ones_shape = [_]c_int{dk};
            var w = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&w, &ones_shape, 1, mlx.mlx_array_dtype(qkv), self.s));
            self.gdn_ones_w = w;
        }
        const ones_w = self.gdn_ones_w.?;

        var q_norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_norm);
        try mlx.check(mlx.mlx_fast_rms_norm(&q_norm, q_heads, ones_w, cfg.gdnQkNormEps(), self.s));
        try mlx.check(mlx.mlx_multiply(&q_scaled, q_norm, inv_scale_sq, self.s));

        var k_norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_norm);
        try mlx.check(mlx.mlx_fast_rms_norm(&k_norm, k_heads, ones_w, cfg.gdnQkNormEps(), self.s));
        try mlx.check(mlx.mlx_multiply(&k_scaled, k_norm, inv_sqrt_sc, self.s));
    }

    // g = exp(-exp(A_log) * softplus(a + dt_bias)), one scalar per (token, value head) [B, S, Hv]: the compiled
    // closure (mlx-lm's compute_g), the chain as fallback.
    const vector_gate = false;
    const g = g_fused orelse try self.computeGdnGate(la.A_log, a_proj, la.dt_bias);
    defer if (g.ctx != null) {
        _ = mlx.mlx_array_free(g);
    };

    // beta = sigmoid(b)
    const beta = beta_fused orelse blk: {
        var bt = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sigmoid(&bt, b_proj, self.s)); // [B, S, Hv]
        break :blk bt;
    };
    defer _ = mlx.mlx_array_free(beta);

    // Initialize SSM state if needed.
    // Can't use ssm.initialized — conv1dWithCache already set it to true.
    // Check if ssm_state is empty (ctx == null) as the actual init indicator.
    if (ssm.ssm_state.ctx == null) {
        const state_shape = [_]c_int{ batch, num_v_heads, dv, dk };
        ssm.ssm_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&ssm.ssm_state, &state_shape, 4, self.ssmStateDtype(), self.s));
    }

    // Fused Metal kernel: runs the full T-step delta recurrence in one dispatch.
    // Inputs (shapes): q,k [B,T,Hk,Dk] (GQA handled in kernel), v [B,T,Hv,Dv],
    //                  g,beta [B,T,Hv], state_in [B,Hv,Dv,Dk], T scalar.
    // Outputs: y [B,T,Hv,Dv], state_out [B,Hv,Dv,Dk].
    const T_scalar = mlx.mlx_array_new_int(seq_len);
    defer _ = mlx.mlx_array_free(T_scalar);

    const y_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };

    // The kernels specialize on the ACTUAL widths, never an assumed one.
    // bf16 checkpoints land here as bf16, but an f16 checkpoint promotes
    // its activations to f32 (f16 ⊕ f32-scalar → f32) and a hardcoded
    // bf16 pointer type then fails to COMPILE, which is an uncatchable
    // MLX error that kills the server mid-request. `g`/`beta` are read
    // through their raw buffer names with an explicit float cast, so they
    // are width-agnostic and need no template arg. Outputs are ours: y in
    // the activation dtype below, the state in the incoming state's
    // (`ssmStateDtype` when it was allocated).
    const gdn_in_dtype = mlx.mlx_array_dtype(q_scaled);
    const gdn_in_itemsize = mlx.mlx_array_itemsize(q_scaled);
    const gdn_state_dtype = mlx.mlx_array_dtype(ssm.ssm_state);
    // y leaves in the activation dtype (bf16 for every checkpoint but an
    // f16 Hadamard pack); an f32 activation keeps the historical bf16 y.
    const gdn_out_dtype: mlx.mlx_dtype = if (gdn_in_dtype == .float16) .float16 else .bfloat16;

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, gdn_out_dtype));

    var y_bthd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_bthd);

    if (!mlx.streamIsGpu(self.s)) {
        const r = (try gdnRecurCuda(self.s, q_scaled, k_scaled, v_heads, g, beta, ssm.ssm_state, vector_gate, self.spec_capture_ssm, gdn_out_dtype)) orelse
            try gdnRecurOps(self.s, q_scaled, k_scaled, v_heads, g, beta, ssm.ssm_state, vector_gate, self.spec_capture_ssm, gdn_out_dtype);
        _ = mlx.mlx_array_free(y_bthd);
        y_bthd = r.y;
        _ = mlx.mlx_array_free(ssm.ssm_state);
        ssm.ssm_state = r.state;
        if (self.spec_capture_ssm) {
            if (ssm.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(ssm.spec_state_seq);
            ssm.spec_state_seq = r.state_seq;
        }
    } else if (self.spec_capture_ssm) {
        // Spec verify pass: emit per-position states so partial-accept
        // rollback needs no re-forward. state_seq: [T, B, Hv, Dv, Dk]
        // (last row unwritten — capture-tail trim), final state as its
        // own [B, Hv, Dv, Dk] output.
        const state_seq_shape = [_]c_int{ seq_len, batch, num_v_heads, dv, dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_seq_shape, 5, gdn_state_dtype));
        const state_out_shape = [_]c_int{ batch, num_v_heads, dv, dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_out_shape, 4, gdn_state_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, batch * num_v_heads));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", gdn_in_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", gdn_state_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", gdn_out_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", num_k_heads));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", num_v_heads));

        // seq_stride = per-timestep element stride into state_seq.
        const seq_stride = mlx.mlx_array_new_int(batch * num_v_heads * dv * dk);
        defer _ = mlx.mlx_array_free(seq_stride);
        const inputs_arr = [_]mlx.mlx_array{ q_scaled, k_scaled, v_heads, g, beta, ssm.ssm_state, T_scalar, seq_stride };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);

        const gdn_kernel = try getGdnKernelSeq(vector_gate);
        var outputs_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, gdn_kernel, inputs_vec, config, self.s));
        if (mlx.mlx_vector_array_size(outputs_vec) != 3) return error.MetalKernelBadOutputCount;
        try mlx.check(mlx.mlx_vector_array_get(&y_bthd, outputs_vec, 0));

        // Stash the per-position sequence for rollback (takes ownership).
        var state_seq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&state_seq, outputs_vec, 1));
        if (ssm.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(ssm.spec_state_seq);
        ssm.spec_state_seq = state_seq;

        // Continue normal flow with the kernel's own final state — no
        // slice view into the capture buffer (a view would pin the whole
        // [T,...] buffer across rounds and add 2 graph nodes per layer).
        var final_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&final_state, outputs_vec, 2));
        _ = mlx.mlx_array_free(ssm.ssm_state);
        ssm.ssm_state = final_state;
    } else {
        const state_shape_out = [_]c_int{ batch, num_v_heads, dv, dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_shape_out, 4, gdn_state_dtype));
        // Blocked-seq kernel at prefill widths on the 128-Dk/32-aligned-Dv
        // geometry (oMLX port, ~2x per GDN layer at 16K). Decode (T==1),
        // spec verify widths, and off-geometry shapes keep the stock
        // per-token kernel; the PLD capture path above is untouched.
        // Block size and input width are ONE decision: the staging arrays
        // are declared in the input dtype, so fp32 activations need a
        // smaller TB to fit the threadgroup budget. A null means even the
        // smallest block overflows — decline and stay on the stock kernel
        // rather than emit a kernel that cannot compile.
        // The blocked kernel reads a PER-HEAD gate; a per-channel (KDA)
        // gate is a different indexing contract, so it declines outright
        // rather than silently reading the wrong element.
        const blocked_tb: ?u32 = if (!vector_gate and gdnBlockedEnabled() and gdnBlockedEligible(seq_len, dk, dv, num_k_heads, num_v_heads))
            gdnBlockTFor(GDN_BLOCK_T, dk, gdn_in_itemsize)
        else
            null;
        const route: GdnRoute = if (blocked_tb == null) .stock else if (gdnPipelinedEnabled(cfg.model_type)) .pipelined else .blocked;
        try gdnSetGrid(config, route, dv, num_v_heads, batch);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", gdn_in_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", gdn_state_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", gdn_out_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", num_k_heads));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", num_v_heads));

        const inputs_arr = [_]mlx.mlx_array{ q_scaled, k_scaled, v_heads, g, beta, ssm.ssm_state, T_scalar };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);

        const gdn_kernel = try gdnRouteKernel(route, blocked_tb orelse 0, vector_gate);
        var outputs_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs_vec);
        if (qwen4Standin().gdn_recur) {
            _ = mlx.mlx_array_free(y_bthd);
            y_bthd = try standinRef(v_heads);
        } else {
            try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, gdn_kernel, inputs_vec, config, self.s));
            if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
            try mlx.check(mlx.mlx_vector_array_get(&y_bthd, outputs_vec, 0));

            var new_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&new_state, outputs_vec, 1));
            _ = mlx.mlx_array_free(ssm.ssm_state);
            ssm.ssm_state = new_state;
        }
    }

    // Fused epilogue: rms_norm(y) * silu(z) straight to the flat
    // out_proj input (decode/verify widths, per-head swish gate archs).
    if (fused_prework) {
        if (self.gdn_eps == null) self.gdn_eps = mlx.mlx_array_new_float(cfg.rms_norm_eps);
        if (try gdnNormGateFused(self.s, y_bthd, z_proj, 0, value_dim, la.norm_w, self.gdn_eps.?, false, num_v_heads, dv, batch, seq_len)) |flat| {
            defer _ = mlx.mlx_array_free(flat);
            return if (skip_output) standinRef(flat) else self.qmatmul(flat, la.out_w, la.out_s, la.out_b);
        }
    }

    // Reshape z to [B, S, Hv, Dv]
    const z_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };
    var z_heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(z_heads);
    try mlx.check(mlx.mlx_reshape(&z_heads, z_proj, &z_shape, 4, self.s));

    // RMSNormGated. FLA's FusedRMSNormGated applies the gate's ACTIVATION,
    // which the checkpoint chooses: qwen3.5 uses swish (silu(z) * y), KDA
    // uses a plain sigmoid (y * sigmoid(z)).
    const y_normed = try self.rmsNorm(y_bthd, la.norm_w);
    defer _ = mlx.mlx_array_free(y_normed);
    const out_gated = blk: {
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_sigmoid(&gate, z_heads, self.s));
        var gated = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&gated, y_normed, gate, self.s));
        break :blk gated;
    };
    defer _ = mlx.mlx_array_free(out_gated);

    // Flatten [B, S, Hv, Dv] → [B, S, value_dim]
    const out_flat_shape = [_]c_int{ batch, seq_len, value_dim };
    var out_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_flat);
    try mlx.check(mlx.mlx_reshape(&out_flat, out_gated, &out_flat_shape, 3, self.s));

    return if (skip_output) standinRef(out_flat) else self.qmatmul(out_flat, la.out_w, la.out_s, la.out_b);
}

/// QWEN4_GDN_CHECK=1: run the composed prework beside the fused kernel
/// and log per-component max |diff| (diagnostic).
pub fn gdnCheckFused(self: *Trunk, la: *const LinearAttnWeights, qkv: mlx.mlx_array, old_state: mlx.mlx_array, b_proj: mlx.mlx_array, a_proj: mlx.mlx_array, pre: *const GdnPrework, layer_idx: usize, hk: c_int, hv: c_int, dk: c_int, dv: c_int, batch: c_int, seq: c_int) !void {
    const s = self.s;
    const key_dim = hk * dk;
    const value_dim = hv * dv;
    const c_dim = key_dim * 2 + value_dim;
    if (self.gdn_ones_w == null) {
        var w = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&w, &[_]c_int{dk}, 1, .bfloat16, s));
        self.gdn_ones_w = w;
    }
    var conv_input = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_input);
    {
        const arr = [_]mlx.mlx_array{ old_state, qkv };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, s));
    }
    var ref_state = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_state);
    try mlx.check(mlx.mlx_slice(&ref_state, conv_input, &[_]c_int{ 0, seq, 0 }, 3, &[_]c_int{ batch, seq + 3, c_dim }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var conv_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_raw);
    try mlx.check(mlx.mlx_conv1d(&conv_raw, conv_input, la.conv1d_w, 1, 0, 1, c_dim, s));
    const conv_out = try self.silu(conv_raw);
    defer _ = mlx.mlx_array_free(conv_out);
    const strides3 = [_]c_int{ 1, 1, 1 };
    var v_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_flat);
    try mlx.check(mlx.mlx_slice(&v_flat, conv_out, &[_]c_int{ 0, 0, key_dim * 2 }, 3, &[_]c_int{ batch, seq, c_dim }, 3, &strides3, 3, s));
    var v_heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_heads);
    try mlx.check(mlx.mlx_reshape(&v_heads, v_flat, &[_]c_int{ batch, seq, hv, dv }, 4, s));
    const dv_ = try attn256MaxDiff(pre.v, v_heads, s);
    var dq: f32 = 0;
    var dk_: f32 = 0;
    inline for (.{ .{ @as(c_int, 0), key_dim, self.gdn_q_scale.?, pre.q, &dq }, .{ key_dim, key_dim * 2, self.gdn_k_scale.?, pre.k, &dk_ } }) |case| {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_slice(&flat, conv_out, &[_]c_int{ 0, 0, case[0] }, 3, &[_]c_int{ batch, seq, case[1] }, 3, &strides3, 3, s));
        var heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(heads);
        try mlx.check(mlx.mlx_reshape(&heads, flat, &[_]c_int{ batch, seq, hk, dk }, 4, s));
        var nrm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(nrm);
        try mlx.check(mlx.mlx_fast_rms_norm(&nrm, heads, self.gdn_ones_w.?, self.config.gdnQkNormEps(), s));
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, nrm, case[2], s));
        case[4].* = try attn256MaxDiff(case[3], scaled, s);
    }
    const dst = try attn256MaxDiff(pre.conv_state, ref_state, s);
    const g_ref = try self.computeGdnGate(la.A_log, a_proj, la.dt_bias);
    defer _ = mlx.mlx_array_free(g_ref);
    const dg = try attn256MaxDiff(pre.g, g_ref, s);
    var beta_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(beta_ref);
    try mlx.check(mlx.mlx_sigmoid(&beta_ref, b_proj, s));
    const db = try attn256MaxDiff(pre.beta, beta_ref, s);
    log.info("[gdn-check] layer {d} S={d}: q {e} k {e} v {e} state {e} g {e} beta {e}\n", .{ layer_idx, seq, dq, dk_, dv_, dst, dg, db });
}

pub fn gdnProjGpu(self: *Trunk, x: mlx.mlx_array, la: *const LinearAttnWeights) !GdnProj {
    if (la.qkv_w.ctx != null) return self.gdnProjArrays(x, la.qkv_w, la.qkv_s, la.qkv_b, la.z_w, la.z_s, la.z_b);
    var parts: [3]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (parts) |p| if (p.ctx != null) {
        _ = mlx.mlx_array_free(p);
    };
    for (&parts, la.qkv_parts) |*out, q| out.* = try self.qmatmul(x, q.w, q.s, q.b);
    const vec = mlx.mlx_vector_array_new_data(&parts, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var qkv = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(qkv);
    try mlx.check(mlx.mlx_concatenate_axis(&qkv, vec, -1, self.s));
    const z = try self.qmatmul(x, la.z_w, la.z_s, la.z_b);
    return .{ .qkv = qkv, .z = z };
}

/// GDN projections through explicit weight arrays (the channel rest
/// slices ride this; gdnProjGpu delegates for the full weights).
pub fn gdnProjArrays(self: *Trunk, x: mlx.mlx_array, qkv_w: mlx.mlx_array, qkv_s: mlx.mlx_array, qkv_b: mlx.mlx_array, z_w: mlx.mlx_array, z_s: mlx.mlx_array, z_b: mlx.mlx_array) !GdnProj {
    const qkv = try self.qmatmul(x, qkv_w, qkv_s, qkv_b);
    errdefer _ = mlx.mlx_array_free(qkv);
    const z = try self.qmatmul(x, z_w, z_s, z_b);
    return .{ .qkv = qkv, .z = z };
}

/// Prepends cached conv state, applies depthwise conv1d, updates cache.
/// If apply_silu is true, applies SiLU activation after conv.
/// conv_b is an optional bias added after conv1d.
pub fn conv1dWithCache(
    self: *Trunk,
    x: mlx.mlx_array,
    conv_w: mlx.mlx_array,
    conv_b: ?mlx.mlx_array,
    ssm: *SSMCacheEntry,
    batch: c_int,
    cdim: c_int,
    kernel: c_int,
    apply_silu: bool,
) !mlx.mlx_array {
    // Prepend conv_state or zeros
    var conv_input: mlx.mlx_array = undefined;
    defer _ = mlx.mlx_array_free(conv_input);
    if (ssm.initialized) {
        const arr = [_]mlx.mlx_array{ ssm.conv_state, x };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        conv_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s));
    } else {
        const zero_shape = [_]c_int{ batch, kernel - 1, cdim };
        var zero_state = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zero_state);
        try mlx.check(mlx.mlx_zeros(&zero_state, &zero_shape, 3, mlx.mlx_array_dtype(x), self.s));
        const arr = [_]mlx.mlx_array{ zero_state, x };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        conv_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s));
    }

    // Update conv_state: keep last (kernel-1) positions
    {
        const ci_shape = mlx.getShape(conv_input);
        const ci_len = ci_shape[1];
        const keep_start = ci_len - (kernel - 1);
        const start = [_]c_int{ 0, keep_start, 0 };
        const stop = [_]c_int{ batch, ci_len, cdim };
        const strides = [_]c_int{ 1, 1, 1 };
        var new_conv_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&new_conv_state, conv_input, &start, 3, &stop, 3, &strides, 3, self.s));
        errdefer _ = mlx.mlx_array_free(new_conv_state);
        // The tail must be an OWNED copy: a slice of the chunk-wide `conv_input` keeps
        // the whole chunk alive in this layer until its next forward. Only a chunk-wide
        // parent is worth the copy; at decode width the parent is already tiny.
        const t_rows = ci_len - (kernel - 1);
        if (compactConvStateEnabled() and t_rows >= PREFILL_EVAL_MIN_SEQ) {
            const owned = try materializedOwnedCopy(self.s, new_conv_state);
            _ = mlx.mlx_array_free(new_conv_state);
            new_conv_state = owned;
            @import("../../../sdk/blocks/conv/depthwise.zig").compact_conv_state_copy_count += 1;
        }
        _ = mlx.mlx_array_free(ssm.conv_state);
        ssm.conv_state = new_conv_state;
        ssm.initialized = true;
    }

    // PLD capture: stash the full conv input ([B, (kernel-1)+T, conv_dim])
    // so partial-accept rollback can slice the accepted-position conv_state
    // without a re-forward. Refcount-shared; freed at the end of the round.
    if (self.spec_capture_ssm) {
        if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
        ssm.spec_conv_input = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&ssm.spec_conv_input, conv_input);
    }

    // Depthwise conv1d (groups = cdim)
    var conv_out = if (mlx.cudaAvailable() and mlx.getShape(conv_input)[1] == kernel)
        try depthwiseConvStep(conv_input, conv_w, cdim, kernel, self.s)
    else if (mlx.cudaAvailable())
        try depthwiseConvCuda(conv_input, conv_w, cdim, kernel, self.s)
    else blk: {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_conv1d(&o, conv_input, conv_w, 1, 0, 1, cdim, self.s));
        break :blk o;
    };

    // Optional bias
    if (conv_b) |cb| {
        var biased = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&biased, conv_out, cb, self.s));
        _ = mlx.mlx_array_free(conv_out);
        conv_out = biased;
    }

    // Optional SiLU activation
    if (apply_silu) {
        const activated = try self.silu(conv_out);
        _ = mlx.mlx_array_free(conv_out);
        return activated;
    }
    return conv_out;
}

/// Compile the GatedDeltaNet gating chain (astype→exp→add→astype→exp→log1p→
/// multiply→negative→exp→astype, ~10 dispatches) into one fused kernel.
/// shapeless=true: pure elementwise chain, same trace for any [B,S,Hv].
pub fn compileGdnGate(self: *Trunk) void {
    const raw_closure = mlx.mlx_closure_new_func_payload(
        &gdnGateClosureCallback,
        @ptrCast(self),
        null,
    );
    var compiled = mlx.mlx_closure{ .ctx = null };
    const rc = mlx.mlx_compile(&compiled, raw_closure, true);
    _ = mlx.mlx_closure_free(raw_closure);
    if (rc == 0 and compiled.ctx != null) {
        self.compiled_gdn_gate = compiled;
        log.info("GDN gate compiled (kernel fusion enabled); recurrent state {s}, gate {s}\n", .{ @tagName(self.ssmStateDtype()), @tagName(gdn_decode.gateDtype(self.config.actDtype(), self.ssmStateDtype())) });
    }
}

pub fn gdnGateClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Trunk = @ptrCast(@alignCast(payload.?));
    var A_log = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&A_log, input, 0) != 0) return -1;
    defer _ = mlx.mlx_array_free(A_log);
    var a = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&a, input, 1) != 0) return -1;
    defer _ = mlx.mlx_array_free(a);
    var dt_bias = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&dt_bias, input, 2) != 0) return -1;
    defer _ = mlx.mlx_array_free(dt_bias);

    const g = gdnGateChain(A_log, a, dt_bias, gdnChainGateDtype(mlx.mlx_array_dtype(a), self.ssmStateDtype()), self.s) catch return -1;
    const out_arr = [_]mlx.mlx_array{g};
    res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
    _ = mlx.mlx_array_free(g);
    return 0;
}

/// Apply the compiled GDN gate closure if available, else the raw chain.
/// Returns owned g (the state's dtype, shape of `a`).
pub fn computeGdnGate(self: *const Trunk, A_log: mlx.mlx_array, a: mlx.mlx_array, dt_bias: mlx.mlx_array) !mlx.mlx_array {
    if (self.compiled_gdn_gate) |compiled| {
        const in_arr = [_]mlx.mlx_array{ A_log, a, dt_bias };
        const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 3);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array{ .ctx = null };
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
        if (mlx.mlx_vector_array_size(out_vec) == 1) {
            var g = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&g, out_vec, 0));
            return g;
        }
    }
    return gdnGateChain(A_log, a, dt_bias, gdnChainGateDtype(mlx.mlx_array_dtype(a), self.ssmStateDtype()), self.s);
}
