//! engines/mlx/methods/moe: `Transformer` methods (aliased from engines/mlx/transformer.zig).
//! Owns moeRouterLogits, moeSharedDown, moeSharedProject, moeSharedGateTail, ungatedSharedA4, moeAddUngatedShared, and 10 more.

const distributed_mod = @import("../cluster/distributed.zig");
const mlx = @import("../../../mlx/ffi.zig");
const exl3 = @import("../../../sdk/quant/exl3/format.zig");
const gather_qmm_nax = @import("../../../sdk/blocks/moe/gather_qmm_nax.zig");
const log = @import("../../../sdk/util/log.zig");
const moe_affine4 = @import("../../../sdk/blocks/moe/moe_affine4.zig");
const gather_qmv_rows = @import("../../../sdk/blocks/moe/gather_qmv_rows.zig");
const profile = @import("../diag/profile.zig");
const verify_projection = @import("../../../sdk/blocks/linear/verify_projection.zig");
const MAX_BATCH_ROWS = @import("../../../sdk/bases/run.zig").MAX_BATCH_ROWS;
const ProfClock = @import("../diag/profile.zig").ProfClock;
const decodeProfileEnabled = @import("../diag/profile.zig").decodeProfileEnabled;
const dispatchProbeCount = @import("../diag/profile.zig").dispatchProbeCount;
const lastDim = @import("../../../mlx/array.zig").lastDim;
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const Transformer = @import("../transformer.zig").Transformer;
const qwen4Standin = @import("../../../sdk/diag/standin.zig").qwen4Standin;
const qmatmulBits = @import("../../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
const SharedProjKind = @import("../../../sdk/blocks/linear/verify_projection.zig").SharedProjKind;
const sharedGateRowsEligible = @import("../../../sdk/blocks/linear/verify_projection.zig").sharedGateRowsEligible;
const verifySharedProjection = @import("../../../sdk/blocks/linear/verify_projection.zig").verifySharedProjection;
const verifyQmmNaxAvailable = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const swigluClampLimit = @import("../../../sdk/blocks/mlp/activations.zig").swigluClampLimit;
const swigluFusedEnabled = @import("../../../sdk/blocks/mlp/fused_activations.zig").swigluFusedEnabled;
const moeFp4DecodeEligible = @import("../../../sdk/blocks/moe/decode.zig").moeFp4DecodeEligible;
const useGatherQmvDecode = @import("moe_decode.zig").useGatherQmvDecode;
const gatherExpertMm = @import("../../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const moeCudaFormat = @import("../../../sdk/blocks/moe/gather_qmv_rows.zig").moeCudaFormat;
const moeDecodeDispatchArm = @import("../../../sdk/blocks/moe/gather_qmv_rows.zig").moeDecodeDispatchArm;
const RouterGate = @import("../../../sdk/blocks/moe/router_kernel.zig").RouterGate;
const moeRouterFusedEnabled = @import("../../../sdk/blocks/moe/router_kernel.zig").moeRouterFusedEnabled;
const moePrefillExpertReduce = @import("../../../sdk/blocks/moe/verify_kernels.zig").moePrefillExpertReduce;
const moePrefillRoutePack = @import("../../../sdk/blocks/moe/verify_kernels.zig").moePrefillRoutePack;
const verifyExpertReduce = @import("../../../sdk/blocks/moe/verify_kernels.zig").verifyExpertReduce;
const verifyExpertReuse = @import("../../../sdk/blocks/moe/verify_kernels.zig").verifyExpertReuse;
const verifyIndexedExpertInput = @import("../../../sdk/blocks/moe/verify_kernels.zig").verifyIndexedExpertInput;
const verifyRoutePack = @import("../../../sdk/blocks/moe/verify_kernels.zig").verifyRoutePack;
const MoeMlpWeights = @import("../../../sdk/blocks/moe/weights.zig").MoeMlpWeights;
const transformer = @import("../transformer.zig");

pub fn moeRouterLogits(self: *Transformer, router_x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    const cfg = &self.config;
    if (cfg.moe_logit_select) {
        // kolibri1 routes on fp32 logits (router_w is bound as f32, [in, out]).
        var x32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x32);
        try mlx.check(mlx.mlx_astype(&x32, router_x, .float32, self.s));
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_matmul(&out, x32, mw.router_w, self.s));
        return out;
    }
    const logits = if (mw.router_scale) |rs| blk: {
        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        try mlx.check(mlx.mlx_fast_rms_norm(&normed, router_x, rs, cfg.rms_norm_eps, self.s));
        const qp = self.quantParamsHinted(mw.router_w, mw.router_s, lastDim(normed));
        break :blk try qmatmulBits(normed, mw.router_w, mw.router_s, mw.router_b, qp.bits, qp.group_size, qp.mode, self.s);
    } else blk: {
        const qp = self.quantParamsHinted(mw.router_w, mw.router_s, lastDim(router_x));
        break :blk try qmatmulBits(router_x, mw.router_w, mw.router_s, mw.router_b, qp.bits, qp.group_size, qp.mode, self.s);
    };
    return logits;
}

/// Gate/up/geglu/down of the shared expert. Only `.verify_rows` declines (null);
/// the other primitives always project.
pub fn moeSharedDown(self: *Transformer, comptime proj: SharedProjKind, x: mlx.mlx_array, mw: *const MoeMlpWeights, widths: []const c_int) !?mlx.mlx_array {
    const gate = (try self.moeSharedProject(proj, x, mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b, widths)) orelse return null;
    defer _ = mlx.mlx_array_free(gate);
    const up = (try self.moeSharedProject(proj, x, mw.shared_up_w, mw.shared_up_s, mw.shared_up_b, widths)) orelse return null;
    defer _ = mlx.mlx_array_free(up);
    const act = try self.computeGeglu(gate, up);
    defer _ = mlx.mlx_array_free(act);
    return self.moeSharedProject(proj, act, mw.shared_down_w, mw.shared_down_s, mw.shared_down_b, widths);
}

pub fn moeSharedProject(self: *Transformer, comptime proj: SharedProjKind, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return switch (proj) {
        .qmatmul => try self.qmatmul(x, w, sc, bi),
        .mtp_rows => try self.mtpProjectionRows(x, w, sc, bi, false),
        .verify_rows => try verifySharedProjection(self.s, x, w, sc, bi, widths),
    };
}

pub fn moeSharedGateTail(self: *Transformer, expert_sum: mlx.mlx_array, gate_logit: mlx.mlx_array, down: mlx.mlx_array) !mlx.mlx_array {
    var gate_sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_sig);
    try mlx.check(mlx.mlx_sigmoid(&gate_sig, gate_logit, self.s));
    var shared_gated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shared_gated);
    try mlx.check(mlx.mlx_multiply(&shared_gated, gate_sig, down, self.s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_add(&result, expert_sum, shared_gated, self.s));
    return result;
}

/// An ungated shared expert the affine-4 decode kernels can carry: 4-bit affine at the routed group size.
pub fn ungatedSharedA4(self: *Transformer, mw: *const MoeMlpWeights, group_size: u32) ?moe_affine4.Ungated {
    if (!mw.shared_ungated or mw.shared_gate_w.ctx == null or mw.shared_gate_s.ctx == null) return null;
    const hidden: u32 = self.config.hidden_size;
    const inter: ?u32 = if (self.config.shared_expert_intermediate_size > 0) self.config.shared_expert_intermediate_size else null;
    inline for (.{ .{ mw.shared_gate_w, mw.shared_gate_s, hidden }, .{ mw.shared_up_w, mw.shared_up_s, hidden }, .{ mw.shared_down_w, mw.shared_down_s, inter } }) |t| {
        const qp = self.quantParamsHinted(t[0], t[1], t[2]);
        if (qp.mode != .affine or qp.bits != 4 or qp.group_size != group_size) return null;
    }
    return .{
        .gate = .{ .w = mw.shared_gate_w, .s = mw.shared_gate_s, .b = mw.shared_gate_b },
        .up = .{ .w = mw.shared_up_w, .s = mw.shared_up_s, .b = mw.shared_up_b },
        .down = .{ .w = mw.shared_down_w, .s = mw.shared_down_s, .b = mw.shared_down_b },
    };
}

/// `gate_logit`: the shared-expert gate's logit when the router kernel already computed it.
/// `expert_sum` plus the always-on shared expert; consumes `expert_sum`.
pub fn moeAddUngatedShared(self: *Transformer, expert_sum: mlx.mlx_array, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    if (mw.shared_gate_w.ctx == null) return expert_sum;
    defer _ = mlx.mlx_array_free(expert_sum);
    const sh_gate = try self.qmatmul(expert_x, mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b);
    defer _ = mlx.mlx_array_free(sh_gate);
    const sh_up = try self.qmatmul(expert_x, mw.shared_up_w, mw.shared_up_s, mw.shared_up_b);
    defer _ = mlx.mlx_array_free(sh_up);
    const sh_act = try self.computeGeglu(sh_gate, sh_up);
    defer _ = mlx.mlx_array_free(sh_act);
    const sh_down = try self.qmatmul(sh_act, mw.shared_down_w, mw.shared_down_s, mw.shared_down_b);
    defer _ = mlx.mlx_array_free(sh_down);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&result, expert_sum, sh_down, self.s));
    return result;
}

pub fn moeAddGatedShared(self: *Transformer, expert_sum: mlx.mlx_array, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights, gate_logit: ?mlx.mlx_array) !mlx.mlx_array {
    const down = (try self.moeSharedDown(.qmatmul, expert_x, mw, &.{})).?;
    defer _ = mlx.mlx_array_free(down);
    return self.moeAddSharedDown(expert_sum, expert_x, down, mw, gate_logit);
}

pub fn moeAddSharedDown(self: *Transformer, expert_sum: mlx.mlx_array, expert_x: mlx.mlx_array, down: mlx.mlx_array, mw: *const MoeMlpWeights, gate_logit_in: ?mlx.mlx_array) !mlx.mlx_array {
    if (gate_logit_in) |logit| return self.moeSharedGateTail(expert_sum, logit, down);
    const seg_w = mw.shared_expert_gate_w.?;
    const seg_qp = self.quantParamsHinted(seg_w, mw.shared_expert_gate_s.?, lastDim(expert_x));
    const gate_logit = try qmatmulBits(expert_x, seg_w, mw.shared_expert_gate_s.?, mw.shared_expert_gate_b.?, seg_qp.bits, seg_qp.group_size, seg_qp.mode, self.s);
    defer _ = mlx.mlx_array_free(gate_logit);
    return self.moeSharedGateTail(expert_sum, gate_logit, down);
}

pub fn moeMLP(self: *Transformer, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    return self.moeMLP2(x, x, mw);
}

/// Tensor-parallel MoE: this rank's slice of every expert plus the worker's, both
/// routed by the logits computed here so the two halves pick the same experts.
pub fn moeTensorParallel(self: *Transformer, tp: *distributed_mod.Peer, x: mlx.mlx_array, mw: *const MoeMlpWeights, layer: u32) !mlx.mlx_array {
    const logits = try self.moeRouterLogits(x, mw);
    defer _ = mlx.mlx_array_free(logits);
    try tp.sendMoe(x, logits, layer);
    const mine = try self.moeMLP2WithRouter(x, x, mw, logits, false);
    defer _ = mlx.mlx_array_free(mine);
    const theirs = try tp.recvLike(x);
    defer _ = mlx.mlx_array_free(theirs);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, mine, theirs, self.s));
    return out;
}

/// The worker's half of `moeTensorParallel`: its expert slices under rank 0's routing.
pub fn moeExpertsRouted(self: *Transformer, x: mlx.mlx_array, layer: u32, logits: mlx.mlx_array) !mlx.mlx_array {
    const mw = &self.moe_layers.?[layer].mlp.moe;
    return self.moeMLP2WithRouter(x, x, mw, logits, false);
}

pub fn moeMLP2WithRouter(self: *Transformer, router_x: mlx.mlx_array, expert_x_in: mlx.mlx_array, mw: *const MoeMlpWeights, router_override: ?mlx.mlx_array, skip_shared: bool) !mlx.mlx_array {
    const cfg = &self.config;
    // DIAGNOSTIC (diag dispatch-probe=N): inject N extra small
    // elementwise kernels per MoE layer and read the slope. Multiplying by
    // an exact 1.0 is output-identical for every finite value, and the
    // result FEEDS the expert path so MLX cannot elide it — the two
    // properties the "duplicate a component" anti-pattern lacks. The slope
    // is what says whether a fusion that removes k dispatches per layer is
    // worth writing: it prices one dispatch in the LIVE graph, which no
    // micro-benchmark of an isolated kernel can.
    var probe_chain = mlx.mlx_array{ .ctx = null };
    defer if (probe_chain.ctx != null) {
        _ = mlx.mlx_array_free(probe_chain);
    };
    const expert_x = blk: {
        const n = dispatchProbeCount();
        if (n == 0) break :blk expert_x_in;
        var cur = expert_x_in;
        for (0..n) |_| {
            var next = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&next, cur, self.one, self.s));
            if (probe_chain.ctx != null) _ = mlx.mlx_array_free(probe_chain);
            probe_chain = next;
            cur = next;
        }
        break :blk cur;
    };
    // Decode MoE-internals profiler (diag decode-profile, S==1 only).
    const moe_prof = decodeProfileEnabled() and mlx.getShape(expert_x)[1] == 1;
    var mclk: ProfClock = if (moe_prof) ProfClock.init() else undefined;
    // Per-expert-weight params: mixed-precision MoE checkpoints vary bits
    // (and, with non-affine modes, group size + mode) per weight — resolve
    // each individually. gate/up consume the hidden dim; down consumes the
    // expert intermediate dim.
    const gate_qp = self.quantParamsHinted(mw.switch_gate_w, mw.switch_gate_s, lastDim(expert_x));
    const up_qp = self.quantParamsHinted(mw.switch_up_w, mw.switch_up_s, lastDim(expert_x));
    const down_qp = self.quantParamsHinted(mw.switch_down_w, mw.switch_down_s, if (cfg.moe_intermediate_size > 0) cfg.moe_intermediate_size else null);

    const router_logits = if (router_override) |given| blk: {
        var owned = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&owned, given));
        break :blk owned;
    } else try self.moeRouterLogits(router_x, mw);
    defer _ = mlx.mlx_array_free(router_logits);

    // Top-K + softmax/renormalize as a single fused kernel (when compiled).
    // Hy3 (expert_bias bound): sigmoid+bias selection instead of softmax.
    // At one decode row the shared-expert gate's logit rides the softmax router kernel (RouterMode.softmax_gate).
    const gate_fold: ?RouterGate = blk: {
        if (mw.expert_bias != null or router_override != null or skip_shared or mw.shared_ungated or mw.shared_gate_vec.ctx == null) break :blk null;
        if (qwen4Standin().moe_shared or qwen4Standin().moe_router) break :blk null;
        const lsh = mlx.getShape(router_logits);
        var rows: usize = 1;
        for (lsh[0 .. lsh.len - 1]) |d| rows *= @intCast(d);
        if (rows != 1 or mlx.mlx_array_size(expert_x) != mlx.mlx_array_size(mw.shared_gate_vec)) break :blk null;
        break :blk .{ .x = expert_x, .w = mw.shared_gate_vec };
    };
    const routed = if (mw.expert_bias) |bias|
        try self.computeHy3Routing(router_logits, bias)
    else
        try self.computeMoeRouting(router_logits, gate_fold);
    defer if (routed.gate_logit.ctx != null) {
        _ = mlx.mlx_array_free(routed.gate_logit);
    };
    // With the gate's logit in hand the shared expert's down output is built up front: the affine-4
    // down kernel adds it in its epilogue, and the tail below uses it when another arm served the routed part.
    const shared_down: ?mlx.mlx_array = if (routed.gate_logit.ctx != null) (try self.moeSharedDown(.qmatmul, expert_x, mw, &.{})) else null;
    defer if (shared_down) |sd| {
        _ = mlx.mlx_array_free(sd);
    };
    var inds = routed.inds;
    defer _ = mlx.mlx_array_free(inds);
    var norm_scores = routed.norm_scores;
    defer _ = mlx.mlx_array_free(norm_scores);
    if (qwen4Standin().moe_router) {
        // Diagnostic: fixed experts 0..K-1, uniform scores (routing off the critical path).
        const ish = mlx.getShape(inds);
        const k: c_int = ish[ish.len - 1];
        var ar = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ar);
        try mlx.check(mlx.mlx_arange(&ar, 0, @floatFromInt(k), 1, mlx.mlx_array_dtype(inds), self.s));
        var fixed = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_broadcast_to(&fixed, ar, ish.ptr, @intCast(ish.len), self.s));
        var fixed_c = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(&fixed_c, fixed, false, self.s));
        _ = mlx.mlx_array_free(fixed);
        try mlx.check(mlx.mlx_array_eval(fixed_c));
        _ = mlx.mlx_array_free(inds);
        inds = fixed_c;
        const ssh = mlx.getShape(norm_scores);
        var uni = mlx.mlx_array_new();
        const one_k = try scalarOf(1.0 / @as(f32, @floatFromInt(k)), mlx.mlx_array_dtype(norm_scores), self.s);
        defer _ = mlx.mlx_array_free(one_k);
        try mlx.check(mlx.mlx_full(&uni, ssh.ptr, @intCast(ssh.len), one_k, mlx.mlx_array_dtype(norm_scores), self.s));
        try mlx.check(mlx.mlx_array_eval(uni));
        _ = mlx.mlx_array_free(norm_scores);
        norm_scores = uni;
    }

    // The group-limited chain deliberately hands its weights back in f32
    // (a bf16 round trip after routed_scaling_factor costs real accuracy),
    // and it is the CALLER's job to land them in the activation dtype.
    // Skipping that multiplies the residual stream up to f32 at the first
    // MoE layer, which then promotes every weight on read for the rest of
    // the forward — a whole-model slowdown that shows up only as a
    // `[dtype-trace] residual widened` line.
    if (mlx.mlx_array_dtype(norm_scores) != mlx.mlx_array_dtype(expert_x)) {
        var cast_scores = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cast_scores, norm_scores, mlx.mlx_array_dtype(expert_x), self.s));
        _ = mlx.mlx_array_free(norm_scores);
        norm_scores = cast_scores;
    }

    // Sigma-MoE: per-expert scale on selected indices (pes[inds]).
    // Stays outside the closure: depends on per-layer weights.
    if (mw.per_expert_scale) |pes| {
        var selected_scales = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(selected_scales);
        try mlx.check(mlx.mlx_take(
            &selected_scales,
            pes,
            inds,
            self.s,
        ));
        var scaled_scores = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled_scores, norm_scores, selected_scales, self.s));
        _ = mlx.mlx_array_free(norm_scores);
        norm_scores = scaled_scores;
    }

    if (cfg.exl3) |spec| {
        const bank: exl3.Bank = .{
            .gate = .{ .trellis = mw.switch_gate_w, .suh = mw.switch_gate_s, .svh = mw.switch_gate_b },
            .up = .{ .trellis = mw.switch_up_w, .suh = mw.switch_up_s, .svh = mw.switch_up_b },
            .down = .{ .trellis = mw.switch_down_w, .suh = mw.switch_down_s, .svh = mw.switch_down_b },
        };
        const clamp = swigluClampLimit(cfg.swiglu_limit, cfg.swiglu_clamp) orelse return error.Exl3SwigluLimitUnsupported;
        const opts: exl3.Routed = .{ .limit = @intCast(clamp), .verify_rows = skip_shared and router_override != null };
        const y = try @import("../modules/call.zig").call("exl3_routed", .{ self.s, expert_x, bank, inds, norm_scores, spec, opts });
        if (skip_shared) return y;
        if (mw.shared_ungated) return self.moeAddUngatedShared(y, expert_x, mw);
        if (mw.shared_expert_gate_w == null or qwen4Standin().moe_shared) return y;
        defer _ = mlx.mlx_array_free(y);
        return self.moeAddGatedShared(y, expert_x, mw, null);
    }

    // Expert computation. Two paths:
    //
    //   Decode (S=1): per-expert gather_qmm with `rhs_indices=inds` shape
    //   [B,S,K]. Output is [B,S,K,1,inter]; each token reads K expert
    //   blocks from random offsets, but at S=1 there are at most K unique
    //   experts so HBM scatter is bounded.
    //
    //   Multi-position (S>1): mlx-lm's `_gather_sort` flow. Flatten inds →
    //   argsort globally → `lhs_indices = order // K` selects which token
    //   row to feed each sorted slot; `rhs_indices = inds[order]` selects
    //   the expert (now sorted, so consecutive slots hit the same expert
    //   block → one HBM stream). After down_proj, an inverse permutation
    //   restores the original [B,S,K] layout. Critical for drafter verify
    //   on MoE: at block_size=4 + top_k=8 the old `total_inds >= 64`
    //   threshold left verify (32 inds) on the slow scatter path while the
    //   sorted path's argsort overhead is negligible at that size.
    if (moe_prof) {
        try mlx.check(mlx.mlx_array_eval(inds));
        try mlx.check(mlx.mlx_array_eval(norm_scores));
        profile.profileState().decode_prof.moe_router_ns += mclk.lap();
    }

    const x_shape = mlx.getShape(expert_x);
    const B = x_shape[0];
    const S = x_shape[1];
    const D = x_shape[x_shape.len - 1];
    const inds_shape = mlx.getShape(inds);
    const K = inds_shape[inds_shape.len - 1];
    const total_inds: c_int = B * S * K;
    // A per-expert gate scale forces the sorted path (the fused decode kernels have nowhere to apply it).
    const has_gate_scale = mw.gate_expert_scale.ctx != null;
    const do_sort = B * S > 1 or total_inds >= 64 or has_gate_scale;
    const no_idx = mlx.mlx_array{ .ctx = null };

    var down_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(down_out);
    // True when the decode fast path already produced the WEIGHTED expert
    // sum [B,S,hidden] (fused down+reduce kernel) — the multiply+sum
    // below must then be skipped, not repeated.
    var moe_reduced = false;
    // True when the affine-4 down kernel already added the gated shared expert.
    var shared_fused = false;
    var cost_arm: u8 = 4;

    const cuda_fmt = if (mlx.cudaAvailable()) moeCudaFormat(B * S, gate_qp.mode, up_qp.mode, down_qp.mode, cfg.hidden_act == .silu, false, mw.up_expert_scale.ctx != null, cfg.swiglu_clamp or cfg.swiglu_limit > 0.0) else null;
    const ungated_a4: ?moe_affine4.Ungated = if (skip_shared or B * S != 1) null else self.ungatedSharedA4(mw, gate_qp.group_size);
    const fp4_shared: ?moe_affine4.Shared = if (B * S == 1) if (shared_down) |sd| .{ .down = sd, .logit = routed.gate_logit } else null else null;
    if (moeFp4DecodeEligible(B * S, false, swigluClampLimit(self.config.swiglu_limit, self.config.swiglu_clamp), mlx.mlx_array_dtype(expert_x), gate_qp.mode, up_qp.mode, down_qp.mode) and
        try self.moeFp4Decode(&down_out, expert_x, inds, norm_scores, mw, gate_qp.mode, gate_qp.group_size, fp4_shared))
    {
        moe_reduced = true;
        shared_fused = fp4_shared != null;
        cost_arm = 1;
    } else if (B * S == 1 and (swigluClampLimit(self.config.swiglu_limit, self.config.swiglu_clamp) orelse 1) == 0 and cfg.hidden_act == .silu and
        gate_qp.mode == .affine and up_qp.mode == .affine and down_qp.mode == .affine and
        gate_qp.bits == 4 and up_qp.bits == 4 and down_qp.bits == 4 and gate_qp.group_size == up_qp.group_size and gate_qp.group_size == down_qp.group_size and
        try self.moeAffine4Decode(&down_out, expert_x, inds, norm_scores, mw, gate_qp.group_size, if (shared_down) |sd| .{ .down = sd, .logit = routed.gate_logit } else null, ungated_a4))
    {
        moe_reduced = true;
        shared_fused = shared_down != null or ungated_a4 != null;
        cost_arm = 1;
    } else if (moeDecodeDispatchArm(B, S, K, false, (swigluClampLimit(self.config.swiglu_limit, self.config.swiglu_clamp) orelse 1) != 0) == .rows and
        useGatherQmvDecode(self, gate_qp, up_qp) and mw.switch_gate_s.ctx != null and mw.switch_up_s.ctx != null and mw.switch_down_s.ctx != null and
        try self.moeDecodeGatherQmvRows(&down_out, expert_x, inds, norm_scores, &moe_reduced, mw, gate_qp, up_qp, down_qp, D, K, B, S))
    {
        gather_qmv_rows.moeStats().moe_rows_fused_layers +%= 1;
        cost_arm = 1;
    } else if (cuda_fmt != null and try self.moeCudaDecode(&down_out, expert_x, inds, norm_scores, mw, cuda_fmt.?)) {
        moe_reduced = true;
        cost_arm = 1;
    } else if (do_sort) {
        gather_qmv_rows.moeStats().moe_sorted_layers +%= 1;
        cost_arm = 2;
        // ── Global-sort prefill path ──

        // Flatten inds → [N] where N = B*S*K
        const flat_shape = [_]c_int{total_inds};
        var flat_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat_inds);
        try mlx.check(mlx.mlx_reshape(&flat_inds, inds, &flat_shape, 1, self.s));

        const verify_route = self.verifyFeatureEnabled(.routing, skip_shared, B, S);
        const route_pack = if (verify_route) try verifyRoutePack(self.s, flat_inds, K) else try moePrefillRoutePack(self.s, flat_inds, K, @intCast(cfg.num_experts));
        var inv_order = if (route_pack) |value| value.inverse else mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inv_order);
        var sorted_inds = if (route_pack) |value| value.sorted else mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted_inds);
        var lhs_idx = if (route_pack) |value| value.lhs else mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lhs_idx);
        if (route_pack == null) {
            var order = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(order);
            try mlx.check(mlx.mlx_argsort_axis(&order, flat_inds, 0, self.s));
            try mlx.check(mlx.mlx_argsort_axis(&inv_order, order, 0, self.s));
            try mlx.check(mlx.mlx_take_axis(&sorted_inds, flat_inds, order, 0, self.s));
            const k_arr = mlx.mlx_array_new_int(K);
            defer _ = mlx.mlx_array_free(k_arr);
            try mlx.check(mlx.mlx_floor_divide(&lhs_idx, order, k_arr, self.s));
        } else if (verify_route) {
            verify_projection.mtp_verify_kernel_calls[2] +%= 1;
            const Once = struct {
                var logged = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[mtp-verify] fused verifier route packing engaged\n", .{});
            }
        } else {
            const Once = struct {
                var logged = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[moe] prefill counting-sort route packing engaged (assignments={d})\n", .{total_inds});
            }
        }

        // x_flat: [B,S,D] → [B*S, D]
        const bs_d_shape = [_]c_int{ B * S, D };
        var x_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_flat);
        try mlx.check(mlx.mlx_reshape(&x_flat, expert_x, &bs_d_shape, 2, self.s));

        var indexed_lhs: mlx.mlx_array = .{};
        defer if (indexed_lhs.ctx != null) {
            _ = mlx.mlx_array_free(indexed_lhs);
        };
        const input_view = if (self.verifyFeatureEnabled(.routing, skip_shared, B, S) and
            gate_qp.bits == 4 and up_qp.bits == 4 and gate_qp.group_size == 64 and up_qp.group_size == 64 and
            gate_qp.mode == .affine and up_qp.mode == .affine and
            mlx.getShape(mw.switch_gate_w)[0] == mlx.getShape(mw.switch_up_w)[0])
        blk: {
            try mlx.check(mlx.mlx_astype(&indexed_lhs, lhs_idx, .uint32, self.s));
            break :blk try verifyIndexedExpertInput(self.s, x_flat, indexed_lhs, mlx.getShape(mw.switch_gate_w)[0]);
        } else null;
        var fused_act: ?mlx.mlx_array = null;
        var mapped_gate: ?mlx.mlx_array = null;
        var mapped_up: ?mlx.mlx_array = null;
        errdefer {
            if (mapped_gate) |value| _ = mlx.mlx_array_free(value);
            if (mapped_up) |value| _ = mlx.mlx_array_free(value);
        }
        const nax_available = verifyQmmNaxAvailable();
        const gate_experts = mlx.getShape(mw.switch_gate_w)[0];
        const row_block = total_inds >= 16 and gate_experts > 0 and @divTrunc(total_inds, gate_experts) >= 4;
        if (input_view == null and !self.verifyFeatureEnabled(.routing, skip_shared, B, S) and
            row_block and nax_available and gate_qp.mode == up_qp.mode and (gate_qp.mode == .affine or gate_qp.mode.isFp4()))
        {
            var x_tok = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_tok);
            try mlx.check(mlx.mlx_reshape(&x_tok, x_flat, &.{ B * S, 1, D }, 3, self.s));
            var row_map = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row_map);
            try mlx.check(mlx.mlx_astype(&row_map, lhs_idx, .uint32, self.s));
            const clamp = swigluClampLimit(cfg.swiglu_limit, cfg.swiglu_clamp);
            if (self.compiled_geglu == null and swigluFusedEnabled() and cfg.hidden_act == .silu and clamp != null and
                mw.gate_expert_scale.ctx == null and mw.up_expert_scale.ctx == null and
                gate_qp.bits == up_qp.bits and gate_qp.group_size == up_qp.group_size)
            {
                const sigtab = try @import("../../../sdk/blocks/hc/hc_prefill.zig").sigmoidTable(self.s);
                if (try gather_qmm_nax.sortedGateUp(x_tok, row_map, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, sorted_inds, sigtab, clamp.?, gate_qp.bits, gate_qp.group_size, gate_qp.mode.isFp4(), nax_available, self.s)) |act3| {
                    // The kernel writes [M, 1, n]; the down gather below takes [M, n] like the squeezed stock path.
                    defer _ = mlx.mlx_array_free(act3);
                    var act = mlx.mlx_array_new();
                    errdefer _ = mlx.mlx_array_free(act);
                    const a3 = mlx.getShape(act3);
                    try mlx.check(mlx.mlx_reshape(&act, act3, &.{ a3[0], a3[2] }, 2, self.s));
                    fused_act = act;
                }
            }
            if (fused_act == null) {
                mapped_gate = try gather_qmm_nax.sortedGatherMapped(x_tok, row_map, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, sorted_inds, gate_qp.bits, gate_qp.group_size, gate_qp.mode.isFp4(), nax_available, self.s);
                mapped_up = try gather_qmm_nax.sortedGatherMapped(x_tok, row_map, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, sorted_inds, up_qp.bits, up_qp.group_size, up_qp.mode.isFp4(), nax_available, self.s);
                if (mapped_gate == null or mapped_up == null) {
                    if (mapped_gate) |value| _ = mlx.mlx_array_free(value);
                    if (mapped_up) |value| _ = mlx.mlx_array_free(value);
                    mapped_gate = null;
                    mapped_up = null;
                }
            }
        }
        if (fused_act == null and mapped_gate == null and input_view == null and mlx.cudaAvailable() and
            cfg.hidden_act == .silu and !cfg.swiglu_clamp and cfg.swiglu_limit == 0.0 and self.compiled_geglu == null and
            mw.up_expert_scale.ctx == null and
            gate_qp.mode == up_qp.mode and (gate_qp.mode == .nvfp4 or gate_qp.mode == .mxfp8))
        {
            var row_map = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row_map);
            try mlx.check(mlx.mlx_astype(&row_map, lhs_idx, .uint32, self.s));
            const fmt: @import("../../../sdk/blocks/moe/moe_prefill_cuda.zig").Format = if (gate_qp.mode == .nvfp4) .nvfp4 else .mxfp8;
            const gscale: ?mlx.mlx_array = if (mw.gate_expert_scale.ctx != null) mw.gate_expert_scale else null;
            fused_act = try @import("../../../sdk/blocks/moe/moe_prefill_cuda.zig").gateUp(self.s, x_flat, row_map, sorted_inds, mw.switch_gate_w, mw.switch_gate_s, mw.switch_up_w, mw.switch_up_s, gscale, fmt);
        }
        var x_gathered = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_gathered);
        if (fused_act == null and mapped_gate == null and input_view == null) try mlx.check(mlx.mlx_take_axis(&x_gathered, x_flat, lhs_idx, 0, self.s));
        var x_rep = input_view orelse mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rep);
        if (fused_act == null and mapped_gate == null and input_view == null) try mlx.check(mlx.mlx_reshape(&x_rep, x_gathered, &.{ total_inds, 1, D }, 3, self.s));
        const projection_lhs = if (input_view != null) indexed_lhs else no_idx;
        if (input_view != null) {
            verify_projection.mtp_verify_indexed_input_calls +%= 1;
            const Once = struct {
                var logged = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[mtp-verify] indexed expert verify inputs engaged (tokens={d})\n", .{B * S});
            }
        }

        const expert_act = if (fused_act) |act| act else blk: {
            const use_mapped_gate = mapped_gate != null;
            var gate_out_3d = mapped_gate orelse mlx.mlx_array_new();
            mapped_gate = null;
            defer _ = mlx.mlx_array_free(gate_out_3d);
            if (!use_mapped_gate) try gatherExpertMm(&gate_out_3d, x_rep, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, projection_lhs, sorted_inds, gate_qp.bits, gate_qp.group_size, gate_qp.mode, true, self.s);
            var gate_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_out);
            try mlx.check(mlx.mlx_squeeze(&gate_out, gate_out_3d, self.s));
            try self.scaleExperts(&gate_out, mw.gate_expert_scale, sorted_inds);

            const use_mapped_up = mapped_up != null;
            var up_out_3d = mapped_up orelse mlx.mlx_array_new();
            mapped_up = null;
            defer _ = mlx.mlx_array_free(up_out_3d);
            if (!use_mapped_up) try gatherExpertMm(&up_out_3d, x_rep, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, projection_lhs, sorted_inds, up_qp.bits, up_qp.group_size, up_qp.mode, true, self.s);
            var up_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_out);
            try mlx.check(mlx.mlx_squeeze(&up_out, up_out_3d, self.s));
            try self.scaleExperts(&up_out, mw.up_expert_scale, sorted_inds);

            break :blk if (cfg.swiglu_limit > 0.0)
                try self.computeClampedSwiGLU(gate_out, up_out)
            else
                try self.computeGeglu(gate_out, up_out);
        };
        defer _ = mlx.mlx_array_free(expert_act);

        // down: expand inner singleton → [N,1,intermediate] → gather_qmm → [N,1,hidden]
        var act_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act_exp);
        try mlx.check(mlx.mlx_expand_dims(&act_exp, expert_act, -2, self.s));
        var down_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_3d);
        const paired = if (self.verifyFeatureEnabled(.down_reduce, skip_shared, B, S) and down_qp.bits == 4 and down_qp.group_size == 64 and down_qp.mode == .affine)
            try verifyExpertReuse(self.s, act_exp, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, sorted_inds)
        else
            null;
        if (paired) |value| {
            _ = mlx.mlx_array_free(down_3d);
            down_3d = value;
            verify_projection.mtp_verify_expert_pairs_calls +%= 1;
            const Once = struct {
                var logged = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[mtp-verify] paired expert down verify engaged (assignments={d})\n", .{total_inds});
            }
        } else try gatherExpertMm(&down_3d, act_exp, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, no_idx, sorted_inds, down_qp.bits, down_qp.group_size, down_qp.mode, true, self.s);
        var down_squeezed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_squeezed);
        try mlx.check(mlx.mlx_squeeze(&down_squeezed, down_3d, self.s)); // [N, hidden]

        const verify_reduce = self.verifyFeatureEnabled(.down_reduce, skip_shared, B, S);
        const fused_sum = if (verify_reduce)
            try verifyExpertReduce(self.s, down_squeezed, inv_order, norm_scores)
        else
            try moePrefillExpertReduce(self.s, down_squeezed, inv_order, norm_scores);
        if (fused_sum) |value| {
            _ = mlx.mlx_array_free(down_out);
            down_out = value;
            moe_reduced = true;
            const Once = struct {
                var verify_logged = false;
                var prefill_logged = false;
            };
            if (verify_reduce) {
                verify_projection.mtp_verify_expert_reduce_calls +%= 1;
                if (!Once.verify_logged) {
                    Once.verify_logged = true;
                    log.info("[mtp-verify] fused expert reduction verify engaged (tokens={d})\n", .{B * S});
                }
            } else if (!Once.prefill_logged) {
                Once.prefill_logged = true;
                log.info("[moe] fused prefill weighted sum engaged (tokens={d})\n", .{B * S});
            }
        } else {
            // Inverse permute → original order, then reshape back to [B,S,K,hidden].
            var down_unsorted = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(down_unsorted);
            try mlx.check(mlx.mlx_take_axis(&down_unsorted, down_squeezed, inv_order, 0, self.s));
            const hidden = mlx.getShape(down_unsorted)[1];
            const bskh_shape = [_]c_int{ B, S, K, hidden };
            try mlx.check(mlx.mlx_reshape(&down_out, down_unsorted, &bskh_shape, 4, self.s));
        }
    } else if (B * S == 1 and useGatherQmvDecode(self, gate_qp, up_qp) and mw.switch_gate_s.ctx != null and mw.switch_up_s.ctx != null and mw.switch_down_s.ctx != null and
        try self.moeDecodeGatherQmv(&down_out, expert_x, inds, norm_scores, &moe_reduced, mw, gate_qp, up_qp, down_qp, D, K, B, S))
    {
        // ── Decode fast path: in-place gather-qmv kernel ──
        // Reads the expert bank directly with GPU-resident indices, so it
        // moves the ideal 9.8 MB/projection instead of the take path's 3x
        // (µbench at Laguna's 201 MB bank: 37 us vs batched 72 vs stock
        // gather_qmm 349). Everything is done inside the predicate; a null
        // return (unsupported quant mode or width) falls through to the
        // stock gather below with no behaviour change.
    } else {
        // ── Decode / small-prefill path (gather_qmm) ──
        const exp_shape = [_]c_int{ B, S, 1, 1, D };
        var x_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_exp);
        try mlx.check(mlx.mlx_reshape(&x_exp, expert_x, &exp_shape, 5, self.s));

        var gate_out_5d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_out_5d);
        try gatherExpertMm(&gate_out_5d, x_exp, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, no_idx, inds, gate_qp.bits, gate_qp.group_size, gate_qp.mode, false, self.s);
        var gate_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_out);
        try mlx.check(mlx.mlx_squeeze(&gate_out, gate_out_5d, self.s));

        var up_out_5d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_out_5d);
        try gatherExpertMm(&up_out_5d, x_exp, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, no_idx, inds, up_qp.bits, up_qp.group_size, up_qp.mode, false, self.s);
        var up_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_out);
        try mlx.check(mlx.mlx_squeeze(&up_out, up_out_5d, self.s));

        const expert_act = try self.computeGeglu(gate_out, up_out);
        defer _ = mlx.mlx_array_free(expert_act);

        var act_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act_exp);
        try mlx.check(mlx.mlx_expand_dims(&act_exp, expert_act, -2, self.s));
        var down_5d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_5d);
        try gatherExpertMm(&down_5d, act_exp, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, no_idx, inds, down_qp.bits, down_qp.group_size, down_qp.mode, false, self.s);
        try mlx.check(mlx.mlx_squeeze(&down_out, down_5d, self.s)); // [B, S, K, hidden]
    }

    if (self.cost_trace_active) self.cost_moe |= cost_arm;

    if (moe_prof) {
        try mlx.check(mlx.mlx_array_eval(down_out));
        profile.profileState().decode_prof.moe_experts_ns += mclk.lap();
    }

    // Weight by scores: down_out * norm_scores[..., None] → sum over K.
    // The fused down+reduce path already did both inside its one dispatch.
    var expert_sum = mlx.mlx_array_new();
    if (moe_reduced) {
        _ = mlx.mlx_array_set(&expert_sum, down_out); // [B, S, hidden]
    } else {
        var scores_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores_exp);
        try mlx.check(mlx.mlx_expand_dims(&scores_exp, norm_scores, -1, self.s)); // [B, S, K, 1]
        var weighted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(weighted);
        try mlx.check(mlx.mlx_multiply(&weighted, down_out, scores_exp, self.s));
        try mlx.check(mlx.mlx_sum_axis(&expert_sum, weighted, -2, false, self.s)); // [B, S, hidden]
    }

    if (skip_shared or shared_fused) return expert_sum;

    // Hy3: shared expert ALWAYS added, no gate (reference MoE.__call__:
    // `y = y + self.shared_mlp(x)`). shared_gate_w carries a real handle
    // only when the checkpoint shipped mlp.shared_mlp.* weights.
    if (mw.shared_ungated) {
        const result = try self.moeAddUngatedShared(expert_sum, expert_x, mw);
        if (moe_prof) {
            try mlx.check(mlx.mlx_array_eval(result));
            profile.profileState().decode_prof.moe_shared_ns += mclk.lap();
        }
        return result;
    }

    // Gemma 4: shared expert is handled separately in forwardMoe, just return expert_sum
    if (mw.shared_expert_gate_w == null or qwen4Standin().moe_shared) return expert_sum;
    defer _ = mlx.mlx_array_free(expert_sum);
    if (shared_down) |sd| return self.moeSharedGateTail(expert_sum, routed.gate_logit, sd);
    return self.moeAddGatedShared(expert_sum, expert_x, mw, null);
}
