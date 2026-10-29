//! The trunk's MoE: router logits, top-k routing (fused kernel, compiled closure or chain), the routed experts
//! (fp4 and affine-4 decode kernels, gather-qmv rows, CUDA banks, the sorted NAX prefill, EXL3 banks), the gated
//! shared expert, grouped verify rows and the tensor-parallel split.
//! copied_from: sdk/blocks/moe/{moe,decode,experts,routing}.zig (the qwen4_exp arms: no GLM decode, Hy3 sigmoid
//! routing, sigma-MoE scales, clamped SwiGLU, ungated shared expert or fp32 logit select).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const exl3 = @import("../../../sdk/quant/exl3/format.zig");
const distributed_mod = @import("../../../engines/mlx/cluster/distributed.zig");
const gather_qmm_nax = @import("../../../sdk/blocks/moe/gather_qmm_nax.zig");
const gather_qmv = @import("../../../sdk/blocks/moe/gather_qmv.zig");
const gather_qmv_rows = @import("../../../sdk/blocks/moe/gather_qmv_rows.zig");
const down_reduce = @import("../../../sdk/blocks/moe/down_reduce.zig");
const moe_fp4 = @import("../../../sdk/blocks/moe/moe_fp4.zig");
const moe_affine4 = @import("../../../sdk/blocks/moe/moe_affine4.zig");
const profile = @import("../../../engines/mlx/diag/profile.zig");
const verify_projection = @import("../../../sdk/blocks/linear/verify_projection.zig");
const Trunk = @import("../trunk.zig").Trunk;
const MAX_BATCH_ROWS = @import("../../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const ProfClock = profile.ProfClock;
const decodeProfileEnabled = profile.decodeProfileEnabled;
const dispatchProbeCount = profile.dispatchProbeCount;
const lastDim = @import("../../../engines/mlx/load/weights.zig").lastDim;
const scalarOf = @import("../../../engines/mlx/load/weights.zig").scalarOf;
const qwen4Standin = @import("../../../sdk/diag/standin.zig").qwen4Standin;
const standinOnes = @import("../../../sdk/diag/standin.zig").standinOnes;
const QuantParams = @import("../../../sdk/quant/params.zig").QuantParams;
const QuantMode = @FieldType(QuantParams, "mode");
const qmatmulBits = @import("../../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
const SharedProjKind = verify_projection.SharedProjKind;
const sharedGateRowsEligible = verify_projection.sharedGateRowsEligible;
const verifySharedProjection = verify_projection.verifySharedProjection;
const verifyQmmNaxAvailable = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const swigluClampLimit = @import("../../../sdk/blocks/mlp/activations.zig").swigluClampLimit;
const swigluFusedEnabled = @import("../../../sdk/blocks/mlp/fused_activations.zig").swigluFusedEnabled;
const swigluSigTable = @import("../../../sdk/blocks/mlp/fused_activations.zig").swigluSigTable;
const moeFp4DecodeEligible = @import("../../../sdk/blocks/moe/decode.zig").moeFp4DecodeEligible;
const gatherExpertMm = @import("../../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const moeCudaFormat = gather_qmv_rows.moeCudaFormat;
const moeDecodeDispatchArm = gather_qmv_rows.moeDecodeDispatchArm;
const gatherQmvGateUpRows = gather_qmv_rows.gatherQmvGateUpRows;
const gatherQmvDownReduce = down_reduce.gatherQmvDownReduce;
const gatherQmvDownReduceRows = down_reduce.gatherQmvDownReduceRows;
const gatherQmvDownReduceRowsEligible = down_reduce.gatherQmvDownReduceRowsEligible;
const gatherQmv = gather_qmv.gatherQmv;
const gatherQmvGateUp = gather_qmv.gatherQmvGateUp;
const gatherQmvGateUpEnabled = gather_qmv.gatherQmvGateUpEnabled;
const router_kernel = @import("../../../sdk/blocks/moe/router_kernel.zig");
const RouterGate = router_kernel.RouterGate;
const moeRouterFusedEnabled = router_kernel.moeRouterFusedEnabled;
const moeRouterTopK = router_kernel.moeRouterTopK;
const routing = @import("../../../sdk/blocks/moe/routing.zig");
const MoeRouting = routing.MoeRouting;
const moeRoutingChain = routing.moeRoutingChain;
const verify_kernels = @import("../../../sdk/blocks/moe/verify_kernels.zig");
const moePrefillExpertReduce = verify_kernels.moePrefillExpertReduce;
const moePrefillRoutePack = verify_kernels.moePrefillRoutePack;
const verifyExpertReduce = verify_kernels.verifyExpertReduce;
const verifyExpertReuse = verify_kernels.verifyExpertReuse;
const verifyIndexedExpertInput = verify_kernels.verifyIndexedExpertInput;
const verifyRoutePack = verify_kernels.verifyRoutePack;
const MoeMlpWeights = @import("../../../sdk/blocks/moe/weights.zig").MoeMlpWeights;

pub fn moeRouterLogits(self: *Trunk, router_x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    const qp = self.quantParamsHinted(mw.router_w, mw.router_s, lastDim(router_x));
    return qmatmulBits(router_x, mw.router_w, mw.router_s, mw.router_b, qp.bits, qp.group_size, qp.mode, self.s);
}

/// Gate/up/geglu/down of the shared expert. Only `.verify_rows` declines (null);
/// the other primitives always project.
pub fn moeSharedDown(self: *Trunk, comptime proj: SharedProjKind, x: mlx.mlx_array, mw: *const MoeMlpWeights, widths: []const c_int) !?mlx.mlx_array {
    const gate = (try self.moeSharedProject(proj, x, mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b, widths)) orelse return null;
    defer _ = mlx.mlx_array_free(gate);
    const up = (try self.moeSharedProject(proj, x, mw.shared_up_w, mw.shared_up_s, mw.shared_up_b, widths)) orelse return null;
    defer _ = mlx.mlx_array_free(up);
    const act = try self.computeGeglu(gate, up);
    defer _ = mlx.mlx_array_free(act);
    return self.moeSharedProject(proj, act, mw.shared_down_w, mw.shared_down_s, mw.shared_down_b, widths);
}

pub fn moeSharedProject(self: *Trunk, comptime proj: SharedProjKind, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return switch (proj) {
        .qmatmul => try self.qmatmul(x, w, sc, bi),
        .mtp_rows => try self.mtpProjectionRows(x, w, sc, bi, false),
        .verify_rows => try verifySharedProjection(self.s, x, w, sc, bi, widths),
    };
}

pub fn moeSharedGateTail(self: *Trunk, expert_sum: mlx.mlx_array, gate_logit: mlx.mlx_array, down: mlx.mlx_array) !mlx.mlx_array {
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

pub fn moeAddGatedShared(self: *Trunk, expert_sum: mlx.mlx_array, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights, gate_logit: ?mlx.mlx_array) !mlx.mlx_array {
    const down = (try self.moeSharedDown(.qmatmul, expert_x, mw, &.{})).?;
    defer _ = mlx.mlx_array_free(down);
    return self.moeAddSharedDown(expert_sum, expert_x, down, mw, gate_logit);
}

pub fn moeAddSharedDown(self: *Trunk, expert_sum: mlx.mlx_array, expert_x: mlx.mlx_array, down: mlx.mlx_array, mw: *const MoeMlpWeights, gate_logit_in: ?mlx.mlx_array) !mlx.mlx_array {
    if (gate_logit_in) |logit| return self.moeSharedGateTail(expert_sum, logit, down);
    const seg_w = mw.shared_expert_gate_w.?;
    const seg_qp = self.quantParamsHinted(seg_w, mw.shared_expert_gate_s.?, lastDim(expert_x));
    const gate_logit = try qmatmulBits(expert_x, seg_w, mw.shared_expert_gate_s.?, mw.shared_expert_gate_b.?, seg_qp.bits, seg_qp.group_size, seg_qp.mode, self.s);
    defer _ = mlx.mlx_array_free(gate_logit);
    return self.moeSharedGateTail(expert_sum, gate_logit, down);
}

pub fn moeAddSharedGateRows(self: *Trunk, expert_sum: mlx.mlx_array, joined_input: mlx.mlx_array, inputs: []const mlx.mlx_array, down: mlx.mlx_array, widths: []const c_int, mw: *const MoeMlpWeights) !?mlx.mlx_array {
    if (!sharedGateRowsEligible(widths) or inputs.len != widths.len) return null;
    const gate_logit = try self.qmatmul(joined_input, mw.shared_expert_gate_w.?, mw.shared_expert_gate_s.?, mw.shared_expert_gate_b.?);
    defer _ = mlx.mlx_array_free(gate_logit);
    const result = try self.moeSharedGateTail(expert_sum, gate_logit, down);
    verify_projection.mtp_verify_shared_gate_rows_calls +%= 1;
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[batched] shared-expert gate tail engaged (slots={d})\n", .{inputs.len});
    }
    return result;
}

pub fn moeSharedDownRows(self: *Trunk, joined_input: mlx.mlx_array, widths: []const c_int, mw: *const MoeMlpWeights) !?mlx.mlx_array {
    const down = (try self.moeSharedDown(.verify_rows, joined_input, mw, widths)) orelse return null;
    verify_projection.mtp_verify_shared_rows_calls +%= 1;
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[batched] shared-expert verify projections engaged (slots={d})\n", .{widths.len});
    }
    return down;
}

pub fn moeVerifyRowsSolo(self: *Trunk, inputs: []const mlx.mlx_array, mw: *const MoeMlpWeights) ![]mlx.mlx_array {
    const out = try self.allocator.alloc(mlx.mlx_array, inputs.len);
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |value| _ = mlx.mlx_array_free(value);
        self.allocator.free(out);
    }
    for (inputs, 0..) |input, row| {
        out[row] = try self.moeMLP(input, mw);
        built += 1;
    }
    return out;
}

pub fn moeVerifyRows(self: *Trunk, inputs: []const mlx.mlx_array, mw: *const MoeMlpWeights) ![]mlx.mlx_array {
    if (!mlx.streamIsGpu(self.s) or !moeRouterFusedEnabled() or
        inputs.len < 2 or inputs.len > 8 or self.config.hidden_size < 32 or self.config.num_experts_per_tok >= 32 or
        mw.shared_expert_gate_w == null or qwen4Standin().moe_shared or
        self.config.num_experts == 0 or self.config.num_experts_per_tok == 0 or
        mw.expert_bias != null or mw.router_scale != null or mw.per_expert_scale != null)
    {
        return self.moeVerifyRowsSolo(inputs, mw);
    }
    var total_tokens: u64 = 0;
    var hidden: c_int = 0;
    for (inputs) |input| {
        const shape = mlx.getShape(input);
        if (shape.len != 3 or shape[0] != 1 or shape[1] < 1) return self.moeVerifyRowsSolo(inputs, mw);
        if (hidden == 0) hidden = shape[2] else if (shape[2] != hidden) return self.moeVerifyRowsSolo(inputs, mw);
        total_tokens += @intCast(shape[1]);
    }
    const total_inds = total_tokens * self.config.num_experts_per_tok;
    if (total_inds / self.config.num_experts >= 4) return self.moeVerifyRowsSolo(inputs, mw);

    var router_parts: [MAX_BATCH_ROWS]mlx.mlx_array = @splat(.{ .ctx = null });
    defer {
        for (router_parts[0..inputs.len]) |part| {
            if (part.ctx != null) _ = mlx.mlx_array_free(part);
        }
    }
    const router_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(router_vec);
    const input_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(input_vec);
    for (inputs, 0..) |input, row| {
        router_parts[row] = try self.moeRouterLogits(input, mw);
        _ = mlx.mlx_vector_array_append_value(router_vec, router_parts[row]);
        _ = mlx.mlx_vector_array_append_value(input_vec, input);
    }
    var joined_router = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(joined_router);
    try mlx.check(mlx.mlx_concatenate_axis(&joined_router, router_vec, 1, self.s));
    var joined_input = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(joined_input);
    try mlx.check(mlx.mlx_concatenate_axis(&joined_input, input_vec, 1, self.s));
    var routed = mw.*;
    routed.shared_expert_gate_w = null;
    const joined_sum = try self.moeMLP2WithRouter(joined_input, joined_input, &routed, joined_router, true);
    defer _ = mlx.mlx_array_free(joined_sum);
    var widths: [MAX_BATCH_ROWS]c_int = undefined;
    for (inputs, 0..) |input, i| widths[i] = mlx.getShape(input)[1];
    const shared_down = try self.moeSharedDownRows(joined_input, widths[0..inputs.len], mw);
    defer if (shared_down) |value| {
        _ = mlx.mlx_array_free(value);
    };
    const shared_joined = if (shared_down) |down| try self.moeAddSharedGateRows(joined_sum, joined_input, inputs, down, widths[0..inputs.len], mw) else null;
    defer if (shared_joined) |value| {
        _ = mlx.mlx_array_free(value);
    };

    const out = try self.allocator.alloc(mlx.mlx_array, inputs.len);
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |value| _ = mlx.mlx_array_free(value);
        self.allocator.free(out);
    }
    var start: c_int = 0;
    for (inputs, 0..) |input, row| {
        const seq = mlx.getShape(input)[1];
        if (shared_joined) |joined| {
            out[row] = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(out[row]);
            try mlx.check(mlx.mlx_slice(&out[row], joined, &.{ 0, start, 0 }, 3, &.{ 1, start + seq, hidden }, 3, &.{ 1, 1, 1 }, 3, self.s));
            built += 1;
            start += seq;
            continue;
        }
        var expert_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(expert_row);
        try mlx.check(mlx.mlx_slice(&expert_row, joined_sum, &.{ 0, start, 0 }, 3, &.{ 1, start + seq, hidden }, 3, &.{ 1, 1, 1 }, 3, self.s));
        out[row] = if (shared_down) |down| blk: {
            var down_row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(down_row);
            try mlx.check(mlx.mlx_slice(&down_row, down, &.{ 0, start, 0 }, 3, &.{ 1, start + seq, hidden }, 3, &.{ 1, 1, 1 }, 3, self.s));
            break :blk try self.moeAddSharedDown(expert_row, input, down_row, mw, null);
        } else try self.moeAddGatedShared(expert_row, input, mw, null);
        built += 1;
        start += seq;
    }
    return out;
}

pub fn moeMLP(self: *Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    return self.moeMLP2(x, x, mw);
}

/// Tensor-parallel MoE: this rank's slice of every expert plus the worker's, both
/// routed by the logits computed here so the two halves pick the same experts.
pub fn moeTensorParallel(self: *Trunk, tp: *distributed_mod.Peer, x: mlx.mlx_array, mw: *const MoeMlpWeights, layer: u32) !mlx.mlx_array {
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
pub fn moeExpertsRouted(self: *Trunk, x: mlx.mlx_array, layer: u32, logits: mlx.mlx_array) !mlx.mlx_array {
    const mw = &self.moe_layers[layer].mlp.moe;
    return self.moeMLP2WithRouter(x, x, mw, logits, false);
}

pub fn moeMLP2WithRouter(self: *Trunk, router_x: mlx.mlx_array, expert_x_in: mlx.mlx_array, mw: *const MoeMlpWeights, router_override: ?mlx.mlx_array, skip_shared: bool) !mlx.mlx_array {
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
    // down consumes what gate emits: its rows, not the config's width (a tensor-parallel rank's trunk holds half,
    // the MTP head every row).
    const gsh = mlx.getShape(mw.switch_gate_w);
    const inter: ?u32 = if (gsh.len == 3 and gsh[1] > 0) @intCast(gsh[1]) else if (cfg.moe_intermediate_size > 0) cfg.moe_intermediate_size else null;
    const down_qp = self.quantParamsHinted(mw.switch_down_w, mw.switch_down_s, inter);

    const router_logits = if (router_override) |given| blk: {
        var owned = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&owned, given));
        break :blk owned;
    } else try self.moeRouterLogits(router_x, mw);
    defer _ = mlx.mlx_array_free(router_logits);

    // Top-K + softmax/renormalize as a single fused kernel (when compiled).
    // At one decode row the shared-expert gate's logit rides the softmax router kernel (RouterMode.softmax_gate).
    const gate_fold: ?RouterGate = blk: {
        if (router_override != null or skip_shared or mw.shared_gate_vec.ctx == null) break :blk null;
        if (qwen4Standin().moe_shared or qwen4Standin().moe_router) break :blk null;
        const lsh = mlx.getShape(router_logits);
        var rows: usize = 1;
        for (lsh[0 .. lsh.len - 1]) |d| rows *= @intCast(d);
        if (rows != 1 or mlx.mlx_array_size(expert_x) != mlx.mlx_array_size(mw.shared_gate_vec)) break :blk null;
        break :blk .{ .x = expert_x, .w = mw.shared_gate_vec };
    };
    const routed = try self.computeMoeRouting(router_logits, gate_fold);
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

    // A per-expert routing scale (ModelOpt NVFP4: each expert's up x down `weight_scale_2`) multiplies the
    // selected experts' weights; it stays outside the compiled router (per-layer weights).
    if (mw.per_expert_scale) |pes| {
        var selected_scales = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(selected_scales);
        try mlx.check(mlx.mlx_take(&selected_scales, pes, inds, self.s));
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
        const y = try exl3.routed(self.s, expert_x, bank, inds, norm_scores, spec, .{ .limit = @intCast(clamp), .verify_rows = skip_shared and router_override != null });
        if (skip_shared) return y;
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
        try self.moeAffine4Decode(&down_out, expert_x, inds, norm_scores, mw, gate_qp.group_size, if (shared_down) |sd| .{ .down = sd, .logit = routed.gate_logit } else null, null))
    {
        moe_reduced = true;
        shared_fused = shared_down != null;
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
            if (swigluFusedEnabled() and cfg.hidden_act == .silu and clamp != null and
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
            cfg.hidden_act == .silu and !cfg.swiglu_clamp and cfg.swiglu_limit == 0.0 and
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

            break :blk try self.computeGeglu(gate_out, up_out);
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

    if (mw.shared_expert_gate_w == null or qwen4Standin().moe_shared) return expert_sum;
    defer _ = mlx.mlx_array_free(expert_sum);
    if (shared_down) |sd| return self.moeSharedGateTail(expert_sum, routed.gate_logit, sd);
    return self.moeAddGatedShared(expert_sum, expert_x, mw, null);
}

/// Decode-width MoE expert compute through the in-place gather-qmv kernel
/// (`gatherQmv`), which indexes the expert bank directly instead of
/// materializing the top-K experts. On success writes [B,S,K,hidden] into
/// `out` and returns true; returns false when any projection is outside the
/// kernel's supported set (mxfp4/mxfp8 bank, 3/5/6-bit affine width, odd
/// geometry), leaving `out` untouched so the caller runs the batched take path.
pub fn moeDecodeGatherQmv(
    self: *Trunk,
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

/// fp banks at decode on CUDA: `moe_cuda`'s two launches, the weighted sum [B,S,D] into `out`;
/// false when they decline (the composed chain serves the call).
pub fn moeCudaDecode(self: *Trunk, out: *mlx.mlx_array, expert_x: mlx.mlx_array, inds: mlx.mlx_array, norm_scores: mlx.mlx_array, mw: *const MoeMlpWeights, fmt: @import("../../../sdk/blocks/moe/moe_cuda.zig").Format) !bool {
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
    self: *Trunk,
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

/// Affine-4 banks at ONE decode token: `moe_affine4`'s two dispatches over the token's
/// experts. Writes the weighted sum [1,1,D] into `out` (with the gated shared expert added when
/// `shared` is given); false when the kernels decline.
pub fn moeAffine4Decode(self: *Trunk, out: *mlx.mlx_array, expert_x: mlx.mlx_array, inds: mlx.mlx_array, norm_scores: mlx.mlx_array, mw: *const MoeMlpWeights, group_size: u32, shared: ?moe_affine4.Shared, ungated: ?moe_affine4.Ungated) !bool {
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

/// fp4 banks at decode and verify widths: `moe_fp4`'s two dispatches over
/// every token's own experts. Writes the weighted sum [B,S,D] into `out`;
/// false when the kernels decline.
pub fn moeFp4Decode(self: *Trunk, out: *mlx.mlx_array, expert_x: mlx.mlx_array, inds: mlx.mlx_array, norm_scores: mlx.mlx_array, mw: *const MoeMlpWeights, mode: QuantMode, group_size: u32, shared: ?moe_affine4.Shared) !bool {
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

/// MoE MLP with separate router and expert inputs.
/// router_x: input for routing (raw hidden states).
/// expert_x: input for expert computation (possibly normalized).
pub fn moeMLP2(self: *Trunk, router_x: mlx.mlx_array, expert_x_in: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    return self.moeMLP2WithRouter(router_x, expert_x_in, mw, null, false);
}

pub fn useGatherQmvDecode(self: *const Trunk, gate_qp: QuantParams, up_qp: QuantParams) bool {
    if (self.config.hidden_act != .silu) return false;
    // The fused kernel bakes in silu(gate)*up, clamped at a whole-number limit; others stay composed.
    if (swigluClampLimit(self.config.swiglu_limit, self.config.swiglu_clamp) == null) return false;
    if (!gatherQmvGateUpEnabled()) return false;
    return gate_qp.bits == up_qp.bits and
        gate_qp.group_size == up_qp.group_size and
        gate_qp.mode == up_qp.mode;
}

/// Scale a gather_qmm result by a per-expert scale, in place. `sel` MUST be the index vector the matmul used, in
/// its order (`sorted_inds` on the sorted path).
pub fn scaleExperts(self: *const Trunk, out: *mlx.mlx_array, scale: mlx.mlx_array, sel: mlx.mlx_array) !void {
    if (scale.ctx == null) return;
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try mlx.check(mlx.mlx_take_axis(&picked, scale, sel, 0, self.s));
    var col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_expand_dims(&col, picked, -1, self.s));
    var scaled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&scaled, out.*, col, self.s));
    _ = mlx.mlx_array_free(out.*);
    out.* = scaled;
}

/// Apply the compiled MoE routing closure if available, else fall back.
/// Returns owned `inds` + `norm_scores` — caller must free both.
pub fn computeMoeRouting(self: *const Trunk, router_logits: mlx.mlx_array, gate: ?RouterGate) !MoeRouting {
    const k: c_int = @intCast(self.config.num_experts_per_tok);
    if (try moeRouterTopK(
        self.s,
        router_logits,
        .{ .ctx = null },
        k,
        .softmax,
        true,
        1.0,
        mlx.mlx_array_dtype(router_logits),
        0,
        0,
        gate,
    )) |fused| return fused;
    if (self.compiled_moe_routing) |compiled| {
        const in_arr = [_]mlx.mlx_array{router_logits};
        const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 1);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array{ .ctx = null };
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));

        var inds = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(inds);
        try mlx.check(mlx.mlx_vector_array_get(&inds, out_vec, 0));
        var norm_scores = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(norm_scores);
        try mlx.check(mlx.mlx_vector_array_get(&norm_scores, out_vec, 1));
        return .{ .inds = inds, .norm_scores = norm_scores };
    }
    return self.moeRoutingUncompiled(router_logits, k);
}

/// Reference implementation of the MoE routing chain (used both as fallback
/// and as the body the compiled closure traces). Returns owned `inds` +
/// `norm_scores` arrays — caller must free both.
pub fn moeRoutingUncompiled(self: *const Trunk, router_logits: mlx.mlx_array, k: c_int) !MoeRouting {
    return moeRoutingChain(router_logits, k, self.s);
}

/// Compile MoE routing (negate→argpartition→slice→softmax→take_along_axis→sum→expand→divide)
/// into a single fused kernel. Input: router_logits. Outputs: inds, norm_scores.
/// shapeless=false: slice bounds derive from input ndim, so the closure must
/// re-trace per input shape. MoE inference only sees two shapes in practice
/// (decode seq_len=1, prefill seq_len=N), so the trace cost amortizes after
/// the first prefill + first decode.
pub fn compileMoeRouting(self: *Trunk) void {
    const raw_closure = mlx.mlx_closure_new_func_payload(
        &moeRoutingClosureCallback,
        @ptrCast(self),
        null,
    );
    var compiled = mlx.mlx_closure{ .ctx = null };
    const rc = mlx.mlx_compile(&compiled, raw_closure, false);
    _ = mlx.mlx_closure_free(raw_closure);
    if (rc == 0 and compiled.ctx != null) {
        self.compiled_moe_routing = compiled;
        log.info("MoE routing compiled (kernel fusion enabled)\n", .{});
    }
}

/// Pure subgraph for MoE routing. Inputs:
///   [0] router_logits — shape [..., num_experts]
/// Outputs:
///   [0] inds         — shape [..., K], int32 expert indices (top-K)
///   [1] norm_scores  — shape [..., K], renormalized top-K softmax weights
///
/// The sigma-MoE per-expert-scale path stays outside the closure because it
/// branches on per-layer weights at model-load time.
pub fn moeRoutingClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Trunk = @ptrCast(@alignCast(payload.?));
    const k: c_int = @intCast(self.config.num_experts_per_tok);

    var router_logits = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&router_logits, input, 0) != 0) return -1;
    defer _ = mlx.mlx_array_free(router_logits);

    const inds_norm = self.moeRoutingUncompiled(router_logits, k) catch return -1;
    defer _ = mlx.mlx_array_free(inds_norm.inds);
    defer _ = mlx.mlx_array_free(inds_norm.norm_scores);

    const out_arr = [_]mlx.mlx_array{ inds_norm.inds, inds_norm.norm_scores };
    res.* = mlx.mlx_vector_array_new_data(&out_arr, 2);
    return 0;
}
