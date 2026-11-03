//! The legacy `Transformer`'s methods over `sdk/blocks/moe/decode.zig` (what reads the engine's model state).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const QuantMode = @import("../../../sdk/quant/params.zig").QuantMode;
const moe_fp4 = @import("../../../sdk/blocks/moe/moe_fp4.zig");
const moe_affine4 = @import("../../../sdk/blocks/moe/moe_affine4.zig");
const glm5 = @import("../../../plugins/glm5_next/glm5_next.zig");
const down_reduce = @import("../../../sdk/blocks/moe/down_reduce.zig");
const gather_qmv = @import("../../../sdk/blocks/moe/gather_qmv.zig");
const gather_qmv_rows = @import("../../../sdk/blocks/moe/gather_qmv_rows.zig");
const lastDim = @import("../../../mlx/array.zig").lastDim;
const Transformer = @import("../transformer.zig").Transformer;
const qwen4Standin = @import("../../../sdk/diag/standin.zig").qwen4Standin;
const standinOnes = @import("../../../sdk/diag/standin.zig").standinOnes;
const QuantParams = @import("../../../sdk/quant/params.zig").QuantParams;
const swigluClampLimit = @import("../../../sdk/blocks/mlp/activations.zig").swigluClampLimit;
const swigluSigTable = @import("../../../sdk/blocks/mlp/fused_activations.zig").swigluSigTable;
const gatherQmvDownReduce = @import("../../../sdk/blocks/moe/down_reduce.zig").gatherQmvDownReduce;
const gatherQmvDownReduceRows = @import("../../../sdk/blocks/moe/down_reduce.zig").gatherQmvDownReduceRows;
const gatherQmvDownReduceRowsEligible = @import("../../../sdk/blocks/moe/down_reduce.zig").gatherQmvDownReduceRowsEligible;
const gatherQmv = @import("../../../sdk/blocks/moe/gather_qmv.zig").gatherQmv;
const gatherQmvGateUp = @import("../../../sdk/blocks/moe/gather_qmv.zig").gatherQmvGateUp;
const gatherQmvGateUpEnabled = @import("../../../sdk/blocks/moe/gather_qmv.zig").gatherQmvGateUpEnabled;
const gatherQmvGateUpRows = @import("../../../sdk/blocks/moe/gather_qmv_rows.zig").gatherQmvGateUpRows;
const MoeMlpWeights = @import("../../../sdk/blocks/moe/weights.zig").MoeMlpWeights;

/// Decode-width MoE expert compute through the in-place gather-qmv kernel
/// (`gatherQmv`), which indexes the expert bank directly instead of
/// materializing the top-K experts. On success writes [B,S,K,hidden] into
/// `out` and returns true; returns false when any projection is outside the
/// kernel's supported set (mxfp4/mxfp8 bank, 3/5/6-bit affine width, odd
/// geometry), leaving `out` untouched so the caller runs the batched take path.
pub fn moeDecodeGatherQmv(
    self: *Transformer,
    out: *mlx.mlx_array,
    expert_x: mlx.mlx_array,
    inds: mlx.mlx_array,
    norm_scores: mlx.mlx_array,
    reduced: *bool,
    mw: *const MoeMlpWeights,
    gate_qp: anytype,
    up_qp: anytype,
    down_qp: anytype,
    D: c_int,
    K: c_int,
    B: c_int,
    S: c_int,
) !bool {
    // Routing indices arrive as int32/uint32 [.., K]; the kernel reads a
    // flat uint32 [K].
    var inds_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inds_u32);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const kshape = [_]c_int{K};
        try mlx.check(mlx.mlx_reshape(&flat, inds, &kshape, 1, self.s));
        try mlx.check(mlx.mlx_astype(&inds_u32, flat, .uint32, self.s));
    }
    var x_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_flat);
    const dshape = [_]c_int{D};
    try mlx.check(mlx.mlx_reshape(&x_flat, expert_x, &dshape, 1, self.s));

    // gate / up share the single token's hidden state, so one kernel can
    // do both dot products and the activation — three dispatches off the
    // chain into down_proj. Falls back to the split form on any geometry,
    // quant mode or dtype the fused kernel declines.
    var act_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act_2d);
    var inter: c_int = 0;
    if (qwen4Standin().moe_gateup and self.config.moe_intermediate_size > 0) {
        // Diagnostic: unit activations, the gate+up kernel drops out.
        const sh = [_]c_int{ K, @intCast(self.config.moe_intermediate_size) };
        _ = mlx.mlx_array_free(act_2d);
        act_2d = try standinOnes(&sh, self.s);
        inter = sh[1];
    } else if (self.config.hidden_act == .silu and
        gate_qp.bits == up_qp.bits and gate_qp.group_size == up_qp.group_size and gate_qp.mode == up_qp.mode)
    {
        if (try gatherQmvGateUp(
            self.s,
            x_flat,
            mw.switch_gate_w,
            mw.switch_gate_s,
            mw.switch_gate_b,
            mw.switch_up_w,
            mw.switch_up_s,
            mw.switch_up_b,
            inds_u32,
            gate_qp.bits,
            gate_qp.group_size,
            gate_qp.mode,
            swigluClampLimit(self.config.swiglu_limit, self.config.swiglu_clamp).?,
        )) |fused| {
            _ = mlx.mlx_array_free(act_2d);
            act_2d = fused; // [K, inter], activation already applied
            inter = mlx.getShape(act_2d)[1];
        }
    }
    if (inter == 0) {
        const gate_2d = try gatherQmv(self.s, x_flat, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, inds_u32, gate_qp.bits, gate_qp.group_size, gate_qp.mode, false) orelse return false;
        defer _ = mlx.mlx_array_free(gate_2d);
        const up_2d = try gatherQmv(self.s, x_flat, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, inds_u32, up_qp.bits, up_qp.group_size, up_qp.mode, false) orelse return false;
        defer _ = mlx.mlx_array_free(up_2d);

        inter = mlx.getShape(gate_2d)[1];
        const k1i = [_]c_int{ K, 1, inter };
        var gate_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_out);
        try mlx.check(mlx.mlx_reshape(&gate_out, gate_2d, &k1i, 3, self.s));
        var up_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_out);
        try mlx.check(mlx.mlx_reshape(&up_out, up_2d, &k1i, 3, self.s));

        const expert_act = try self.computeGeglu(gate_out, up_out); // [K,1,inter]
        defer _ = mlx.mlx_array_free(expert_act);

        // down: each expert consumes ITS OWN activation → per-expert x layout.
        const ki = [_]c_int{ K, inter };
        _ = mlx.mlx_array_free(act_2d);
        act_2d = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&act_2d, expert_act, &ki, 2, self.s));
    }
    // Fused down + router weighting + K-reduction: one dispatch replaces
    // the down gather, the score multiply and the sum over K — and the
    // [K, hidden] intermediate never materializes. Declines (dtype
    // mismatch, geometry, kill switch) fall through to the split form
    // with `reduced` untouched.
    if (qwen4Standin().moe_down and @rem(D, inter) == 0) {
        // Diagnostic: the down kernel drops out; the activations are still
        // consumed (sum over K, tiled to hidden) so gate+up stays live.
        var summed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(summed);
        try mlx.check(mlx.mlx_sum_axis(&summed, act_2d, 0, false, self.s));
        const reps = [_]c_int{@divExact(D, inter)};
        var tiled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tiled);
        try mlx.check(mlx.mlx_tile(&tiled, summed, &reps, 1, self.s));
        const bsh_shape = [_]c_int{ B, S, D };
        try mlx.check(mlx.mlx_reshape(out, tiled, &bsh_shape, 3, self.s));
        reduced.* = true;
        return true;
    }
    {
        var scores_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores_flat);
        const kshape2 = [_]c_int{K};
        try mlx.check(mlx.mlx_reshape(&scores_flat, norm_scores, &kshape2, 1, self.s));
        if (try gatherQmvDownReduce(self.s, act_2d, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, inds_u32, scores_flat, down_qp.bits, down_qp.group_size, down_qp.mode)) |sum_flat| {
            defer _ = mlx.mlx_array_free(sum_flat);
            const hidden = mlx.getShape(sum_flat)[0];
            const bsh_shape = [_]c_int{ B, S, hidden };
            try mlx.check(mlx.mlx_reshape(out, sum_flat, &bsh_shape, 3, self.s));
            reduced.* = true;
            if (!down_reduce.downred_engaged) {
                down_reduce.downred_engaged = true;
                log.info("[moe] fused down+reduce kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, down_qp.bits, down_qp.group_size });
            }
            if (!gather_qmv.gqmv_engaged) {
                gather_qmv.gqmv_engaged = true;
                log.info("[moe] gather-qmv kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, gate_qp.bits, gate_qp.group_size });
            }
            return true;
        }
    }
    const down_2d = try gatherQmv(self.s, act_2d, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, inds_u32, down_qp.bits, down_qp.group_size, down_qp.mode, true) orelse return false;
    defer _ = mlx.mlx_array_free(down_2d);

    const hidden = mlx.getShape(down_2d)[1];
    const bskh_shape = [_]c_int{ B, S, K, hidden };
    try mlx.check(mlx.mlx_reshape(out, down_2d, &bskh_shape, 4, self.s));
    // Engagement is COUNTED, never inferred: a silent fallback to the
    // batched take path is output-identical and would look like a null win.
    if (!gather_qmv.gqmv_engaged) {
        gather_qmv.gqmv_engaged = true;
        log.info("[moe] gather-qmv kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, gate_qp.bits, gate_qp.group_size });
    }
    return true;
}

/// fp4 banks at decode and verify widths: `moe_fp4`'s two dispatches over
/// every token's own experts. Writes the weighted sum [B,S,D] into `out`;
/// false when the kernels decline.
pub fn moeFp4Decode(self: *Transformer, out: *mlx.mlx_array, expert_x: mlx.mlx_array, inds: mlx.mlx_array, norm_scores: mlx.mlx_array, mw: *const MoeMlpWeights, mode: QuantMode, group_size: u32, shared: ?moe_affine4.Shared) !bool {
    const xs = mlx.getShape(expert_x);
    const rows = xs[0] * xs[1];
    const ks = mlx.getShape(inds);
    const k = ks[ks.len - 1];
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, expert_x, &.{ rows, xs[2] }, 2, self.s));
    var ind2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ind2);
    try mlx.check(mlx.mlx_reshape(&ind2, inds, &.{ rows, k }, 2, self.s));
    var ind_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ind_u);
    try mlx.check(mlx.mlx_astype(&ind_u, ind2, .uint32, self.s));
    var sc2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc2);
    try mlx.check(mlx.mlx_reshape(&sc2, norm_scores, &.{ rows, k }, 2, self.s));
    const sigtab = try swigluSigTable(self.s, mlx.mlx_array_dtype(expert_x), std.heap.c_allocator);
    const Bank = moe_fp4.Bank;
    const scales: moe_fp4.Scales = .{
        .gate = if (mw.gate_expert_scale.ctx != null) mw.gate_expert_scale else null,
        .up = if (mw.up_expert_scale.ctx != null) mw.up_expert_scale else null,
        .clamp = swigluClampLimit(self.config.swiglu_limit, self.config.swiglu_clamp) orelse return false,
    };
    const y = (try moe_fp4.decode(self.s, x2, Bank{ .w = mw.switch_gate_w, .s = mw.switch_gate_s }, Bank{ .w = mw.switch_up_w, .s = mw.switch_up_s }, Bank{ .w = mw.switch_down_w, .s = mw.switch_down_s }, mw.bank_views, scales, shared, ind_u, sc2, sigtab, mode, group_size)) orelse return false;
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(out, y, &.{ xs[0], xs[1], xs[2] }, 3, self.s));
    return true;
}

/// Affine-4 banks at ONE decode token: `moe_affine4`'s two dispatches over the token's
/// experts. Writes the weighted sum [1,1,D] into `out` (with the gated shared expert added when
/// `shared` is given); false when the kernels decline.
pub fn moeAffine4Decode(self: *Transformer, out: *mlx.mlx_array, expert_x: mlx.mlx_array, inds: mlx.mlx_array, norm_scores: mlx.mlx_array, mw: *const MoeMlpWeights, group_size: u32, shared: ?moe_affine4.Shared, ungated: ?moe_affine4.Ungated) !bool {
    // The sigmoid table covers 16-bit patterns only; an f32 stream (the fixture oracle) declines.
    const xd = mlx.mlx_array_dtype(expert_x);
    if (xd != .bfloat16 and xd != .float16) return false;
    const xs = mlx.getShape(expert_x);
    const ks = mlx.getShape(inds);
    const k = ks[ks.len - 1];
    var x1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x1);
    try mlx.check(mlx.mlx_reshape(&x1, expert_x, &.{xs[2]}, 1, self.s));
    var ind1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ind1);
    try mlx.check(mlx.mlx_reshape(&ind1, inds, &.{k}, 1, self.s));
    var ind_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ind_u);
    try mlx.check(mlx.mlx_astype(&ind_u, ind1, .uint32, self.s));
    var sc1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc1);
    try mlx.check(mlx.mlx_reshape(&sc1, norm_scores, &.{k}, 1, self.s));
    const sigtab = try swigluSigTable(self.s, mlx.mlx_array_dtype(expert_x), std.heap.c_allocator);
    const Bank = moe_affine4.Bank;
    const y = (try moe_affine4.decode(self.s, x1, Bank{ .w = mw.switch_gate_w, .s = mw.switch_gate_s, .b = mw.switch_gate_b }, Bank{ .w = mw.switch_up_w, .s = mw.switch_up_s, .b = mw.switch_up_b }, Bank{ .w = mw.switch_down_w, .s = mw.switch_down_s, .b = mw.switch_down_b }, mw.bank_views, shared, ungated, ind_u, sc1, sigtab, group_size)) orelse return false;
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(out, y, &.{ xs[0], xs[1], xs[2] }, 3, self.s));
    return true;
}

/// fp banks at decode on CUDA: `moe_cuda`'s two launches, the weighted sum [B,S,D] into `out`;
/// false when they decline (the composed chain serves the call).
pub fn moeCudaDecode(self: *Transformer, out: *mlx.mlx_array, expert_x: mlx.mlx_array, inds: mlx.mlx_array, norm_scores: mlx.mlx_array, mw: *const MoeMlpWeights, fmt: @import("../../../sdk/blocks/moe/moe_cuda.zig").Format) !bool {
    const moe_cuda = @import("../../../sdk/blocks/moe/moe_cuda.zig");
    const xs = mlx.getShape(expert_x);
    const ks = mlx.getShape(inds);
    const rows = xs[0] * xs[1];
    const k = ks[ks.len - 1];
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, expert_x, &.{ rows, xs[2] }, 2, self.s));
    var ind2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ind2);
    try mlx.check(mlx.mlx_reshape(&ind2, inds, &.{ rows, k }, 2, self.s));
    var ind_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ind_u);
    try mlx.check(mlx.mlx_astype(&ind_u, ind2, .uint32, self.s));
    var sc2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc2);
    try mlx.check(mlx.mlx_reshape(&sc2, norm_scores, &.{ rows, k }, 2, self.s));
    const gate_scale: ?mlx.mlx_array = if (mw.gate_expert_scale.ctx != null) mw.gate_expert_scale else null;
    const y = (try moe_cuda.decode(self.s, x2, .{ .w = mw.switch_gate_w, .s = mw.switch_gate_s }, .{ .w = mw.switch_up_w, .s = mw.switch_up_s }, .{ .w = mw.switch_down_w, .s = mw.switch_down_s }, gate_scale, ind_u, sc2, fmt)) orelse return false;
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(out, y, &.{ xs[0], xs[1], xs[2] }, 3, self.s));
    return true;
}

/// The fused-rows MoE arm at decode (B >= 2). It runs ahead of the sorted
/// chain, which stays the fallback whenever this declines: a decline is never
/// a wrong answer, only the sorted chain's extra dispatches.
pub fn moeDecodeGatherQmvRows(
    self: *Transformer,
    out: *mlx.mlx_array,
    expert_x: mlx.mlx_array,
    inds: mlx.mlx_array,
    norm_scores: mlx.mlx_array,
    reduced: *bool,
    mw: *const MoeMlpWeights,
    gate_qp: anytype,
    up_qp: anytype,
    down_qp: anytype,
    D: c_int,
    K: c_int,
    batch: c_int,
    seq: c_int,
) !bool {
    const B = batch * seq; // one kernel row per token
    if (!gatherQmvDownReduceRowsEligible(self.config.hidden_act, gate_qp, up_qp, down_qp, D)) return false;
    var inds_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inds_u32);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const kshape = [_]c_int{ B, K };
        try mlx.check(mlx.mlx_reshape(&flat, inds, &kshape, 2, self.s));
        try mlx.check(mlx.mlx_astype(&inds_u32, flat, .uint32, self.s));
    }
    var x_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_2d);
    const dshape = [_]c_int{ B, D };
    try mlx.check(mlx.mlx_reshape(&x_2d, expert_x, &dshape, 2, self.s));

    const act_3d = (try gatherQmvGateUpRows(
        self.s,
        x_2d,
        mw.switch_gate_w,
        mw.switch_gate_s,
        mw.switch_gate_b,
        mw.switch_up_w,
        mw.switch_up_s,
        mw.switch_up_b,
        inds_u32,
        gate_qp.bits,
        gate_qp.group_size,
        gate_qp.mode,
        if (mw.gate_expert_scale.ctx != null) mw.gate_expert_scale else null,
    )) orelse return false;
    defer _ = mlx.mlx_array_free(act_3d);
    var scores_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_2d);
    const kshape2 = [_]c_int{ B, K };
    try mlx.check(mlx.mlx_reshape(&scores_2d, norm_scores, &kshape2, 2, self.s));
    const sum_2d = (try gatherQmvDownReduceRows(
        self.s,
        act_3d,
        mw.switch_down_w,
        mw.switch_down_s,
        mw.switch_down_b,
        inds_u32,
        scores_2d,
        down_qp.bits,
        down_qp.group_size,
        down_qp.mode,
    )) orelse return false;
    defer _ = mlx.mlx_array_free(sum_2d);
    const hidden = mlx.getShape(sum_2d)[1];
    const bsh_shape = [_]c_int{ batch, seq, hidden };
    try mlx.check(mlx.mlx_reshape(out, sum_2d, &bsh_shape, 3, self.s));
    if (seq > 1 and !gather_qmv_rows.moe_verify_rows_logged) {
        gather_qmv_rows.moe_verify_rows_logged = true;
        log.info("[moe-rows] verify rows engaged: S={d} K={d}\n", .{ seq, K });
    }
    reduced.* = true;
    return true;
}

/// MoE MLP with separate router and expert inputs.
/// router_x: input for routing (raw hidden states).
/// expert_x: input for expert computation (possibly normalized).
pub fn moeMLP2(self: *Transformer, router_x: mlx.mlx_array, expert_x_in: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    return self.moeMLP2WithRouter(router_x, expert_x_in, mw, null, false);
}

pub fn useGatherQmvDecode(self: *const Transformer, gate_qp: QuantParams, up_qp: QuantParams) bool {
    if (gate_qp.mode == .gguf) return false; // raw ggml blocks: gatherExpertMm has their kernel
    if (self.config.hidden_act != .silu) return false;
    // The fused kernel bakes in silu(gate)*up, clamped at a whole-number limit; others stay composed.
    if (swigluClampLimit(self.config.swiglu_limit, self.config.swiglu_clamp) == null) return false;
    if (!gatherQmvGateUpEnabled()) return false;
    return gate_qp.bits == up_qp.bits and
        gate_qp.group_size == up_qp.group_size and
        gate_qp.mode == up_qp.mode;
}
