//! The generic routed-expert MLP: a router (softmax top-k renormed, sigmoid + selection-only `expert_bias`,
//! group-limited, or top-k on biased fp32 logits), the experts (one-token fused kernels, the batched-rows kernels,
//! the sorted `gather_qmm` prefill, the stock gather) and an optional shared expert, gated or always added.
//! Serves qwen3_5_moe, laguna, flint, kolibri1, hy_v3 and lfm2_moe as the legacy MoE forward does.
//! copied_from: sdk/blocks/moe/moe.zig (`moeMLP2WithRouter`, `moeRouterLogits`, the shared-expert helpers, without
//! the exl3, GLM, tensor-parallel, qwen4 verify-feature, Gemma 4 sigma-MoE and gpt_oss bias arms),
//! sdk/blocks/moe/decode.zig (the decode arms), sdk/blocks/moe/routing.zig (`computeMoeRouting`,
//! `computeHy3Routing`), sdk/blocks/mlp/activations.zig (`computeGeglu`'s silu arms).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const log = @import("../../util/log.zig");
const block = @import("../block.zig");
const linear = @import("linear.zig");
const compiled = @import("compiled.zig");
const QuantParams = @import("../../quant/params.zig").QuantParams;
const quant_abi = @import("../../../abi/quant.zig");
const routing = @import("../../blocks/moe/routing.zig");
const router_kernel = @import("../../blocks/moe/router_kernel.zig");
const moe_decode = @import("../../blocks/moe/decode.zig");
const moe_fp4 = @import("../../blocks/moe/moe_fp4.zig");
const moe_cuda = @import("../../blocks/moe/moe_cuda.zig");
const moe_prefill_cuda = @import("../../blocks/moe/moe_prefill_cuda.zig");
const moe_affine4 = @import("../../blocks/moe/moe_affine4.zig");
const gather_qmv = @import("../../blocks/moe/gather_qmv.zig");
const gather_qmv_rows = @import("../../blocks/moe/gather_qmv_rows.zig");
const gather_qmm_nax = @import("../../blocks/moe/gather_qmm_nax.zig");
const down_reduce = @import("../../blocks/moe/down_reduce.zig");
const verify_kernels = @import("../../blocks/moe/verify_kernels.zig");
const gatherExpertMm = @import("../../blocks/moe/experts.zig").gatherExpertMm;
const verifyQmmNaxAvailable = @import("../../blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const fused_act = @import("../../blocks/mlp/fused_activations.zig");
const clampedSwiGLU = @import("../../blocks/mlp/activations.zig").clampedSwiGLU;
const sigmoidTable = @import("../../blocks/hc/hc_prefill.zig").sigmoidTable;
const routed = @import("../../weights/routed.zig");
const HiddenAct = @import("../../blocks/mlp/activations.zig").HiddenAct;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;
const Linear = linear.Linear;

/// An expert bank `[E, out, in]` and its quant geometry.
pub const Bank = struct {
    w: Array,
    s: Array = .{ .ctx = null },
    b: Array = .{ .ctx = null },
    qp: QuantParams,
    /// ModelOpt NVFP4: each expert's global scale `[E]` on this projection's output; null = none.
    expert_scale: Array = .{ .ctx = null },

    fn a4(bank: *const Bank) moe_affine4.Bank {
        return .{ .w = bank.w, .s = bank.s, .b = bank.b };
    }
};

/// How the router picks and weights the top-k.
pub const Router = union(enum) {
    /// Softmax over the experts, top-k renormalized (qwen3_5_moe).
    softmax,
    /// Sigmoid scores; top-k selected on scores + `expert_bias` `[E]` f32, weighted by the unbiased scores
    /// (laguna, flint, hy_v3, lfm2_moe), group-limited when `n_group > 1`.
    sigmoid_bias: struct { expert_bias: Array, norm: bool, scale: f32, n_group: c_int = 1, topk_group: c_int = 1 },
    /// Top-k on the f32 logits + `expert_bias`, weights sigmoid(logits) (kolibri1; the router weight is f32 `[in, E]`).
    logit_bias: Array,
};

/// Routed experts a quant format runs as one SwiGLU dispatch (`QuantTable.swiglu`, EXL3): its three banks as the
/// format loaded them; the `Bank`s stay unbuilt.
pub const FormatBanks = struct {
    table: *const quant_abi.QuantTable,
    gate: quant_abi.Weight,
    up: quant_abi.Weight,
    down: quant_abi.Weight,
};

/// The shared expert; `gate_logit` unset = always added.
pub const SharedExpert = struct {
    gate: Linear,
    up: Linear,
    down: Linear,
    /// One logit per token whose sigmoid scales the shared expert (`shared_expert_gate`).
    gate_logit: Linear = .{},
    /// `gate_logit`'s weight as one contiguous `[hidden]` vector: the softmax router kernel dots it in at one row.
    gate_vec: Array = .{ .ctx = null },

    fn gated(se: *const SharedExpert) bool {
        return se.gate_logit.w.ctx != null;
    }
};

pub const Moe = struct {
    pub const kind: block.Kind = .mlp;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        router: Linear,
        route: Router,
        gate: Bank,
        up: Bank,
        down: Bank,
        shared: ?SharedExpert = null,
        /// One-expert views of the affine-4 banks the one-token kernels bind.
        bank_views: ?moe_affine4.Views = null,
        top_k: c_int,
        experts: c_int,
        /// A whole-number SwiGLU clamp (0 = none); null = a clamp no kernel serves.
        clamp: ?u32 = 0,
        /// The clamp as the composed chain applies it (`swiglu_limit` with `swiglu_clamp`); 0 = none.
        clamp_limit: f32 = 0,
        /// The routing chain compiled (`compileRouting`); null = op by op.
        route_fn: compiled.Closure = .{ .ctx = null },
        /// Set: the experts are the format's (`gate`/`up`/`down` unread).
        format_banks: ?FormatBanks = null,
        /// Set: the experts are a streamed bank's (`gate`/`up`/`down` give its quant params only).
        streamed: ?Streamed = null,
        /// ModelOpt NVFP4: each expert's up x down global scale `[E]`, multiplied into its routing weight.
        route_scale: Array = .{ .ctx = null },
        /// A model's own kernel for the routed and shared experts together; tried before the block's arms.
        fused: ?Fused = null,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        return mlp(w, c.g, x);
    }
};

/// Routed experts streamed from an SSD bank (`sdk/weights/streamed.zig`): model layer `layer`'s through `routed`.
pub const Streamed = struct {
    routed: routed.Routed,
    layer: u32,
    /// The next routed layer's router (dense f32 `[hidden, experts]`) and selection bias: a decode call's
    /// `sigmoid(x @ router) + bias` starts that layer's likely reads (`Routed.trunk`'s scores); null = none.
    next: ?struct { router: Array, bias: Array } = null,
};

/// A model's one-kernel MoE at up to `rows` rows (GLM's `moeDecode`): the routed sum plus the shared expert in `x`'s
/// shape, or null to take the block's arms. `limit` is the model's SwiGLU limit as its kernel reads it.
pub const Fused = struct {
    rows: c_int,
    limit: f32 = 0,
    run: *const fn (f: *const Fused, w: *const Moe.Weights, s: ops.Stream, x: Array, inds: Array, scores: Array) anyerror!?Array,
};

// ── routing ──

const RoutePayload = struct { k: c_int, norm: bool, scale: f32, sigmoid: bool, s: ops.Stream };

fn routeBody(p: *const RoutePayload, in: []const Array, out: *[2]Array) anyerror!void {
    const r = if (p.sigmoid)
        try routing.hy3RoutingChain(in[0], in[1], p.k, p.norm, p.scale, p.s)
    else
        try routing.moeRoutingChain(in[0], p.k, p.s);
    out.* = .{ r.inds, r.norm_scores };
}

/// The routing chain compiled per input shape, as the legacy load compiles it (the sigmoid form for a
/// `sigmoid_bias` router).
pub fn compileRouting(w: *const Moe.Weights, s: ops.Stream) compiled.Closure {
    const p: RoutePayload = switch (w.route) {
        .softmax => .{ .k = w.top_k, .norm = true, .scale = 1.0, .sigmoid = false, .s = s },
        .sigmoid_bias => |r| .{ .k = w.top_k, .norm = r.norm, .scale = r.scale, .sigmoid = true, .s = s },
        .logit_bias => return .{ .ctx = null },
    };
    return compiled.compile(RoutePayload, p, 2, routeBody, false);
}

const Routed = struct { inds: Array, scores: Array, gate_logit: ?Array = null };

fn adoptRouting(g: *Graph, r: anytype) !Routed {
    const inds = g.adopt(r.inds) catch |e| {
        _ = mlx.mlx_array_free(r.norm_scores);
        if (r.gate_logit.ctx != null) _ = mlx.mlx_array_free(r.gate_logit);
        return e;
    };
    const scores = g.adopt(r.norm_scores) catch |e| {
        if (r.gate_logit.ctx != null) _ = mlx.mlx_array_free(r.gate_logit);
        return e;
    };
    const gl: ?Array = if (r.gate_logit.ctx != null) try g.adopt(r.gate_logit) else null;
    return .{ .inds = inds, .scores = scores, .gate_logit = gl };
}

fn routerLogits(w: *const Moe.Weights, parent: *Graph, x: Array) !Array {
    if (w.route != .logit_bias) return w.router.forward(parent, x);
    var t = parent.scope();
    defer t.deinit();
    return parent.keep(try t.matmul(try t.astype(x, .float32), w.router.w));
}

fn route(w: *const Moe.Weights, g: *Graph, logits: Array, fold: ?router_kernel.RouterGate) !Routed {
    switch (w.route) {
        .logit_bias => |bias| {
            if (try router_kernel.moeRouterTopK(g.s, logits, bias, w.top_k, .logit_bias, false, 1.0, .float32, 0, 0, null)) |r| return adoptRouting(g, r);
            return adoptRouting(g, try routing.kolibriRoutingChain(logits, bias, w.top_k, g.s));
        },
        .sigmoid_bias => |r| {
            if (r.n_group > 1) return adoptRouting(g, try routing.groupLimitedRouting(logits, r.expert_bias, w.top_k, r.n_group, r.topk_group, r.norm, r.scale, g.s));
            if (try router_kernel.moeRouterTopK(g.s, logits, r.expert_bias, w.top_k, .sigmoid_bias, r.norm, r.scale, .bfloat16, 0, 0, null)) |f| return adoptRouting(g, f);
            if (w.route_fn.ctx != null) if (try compiled.apply(g, w.route_fn, &.{ logits, r.expert_bias }, 2)) |o| return .{ .inds = o[0], .scores = o[1] };
            return adoptRouting(g, try routing.hy3RoutingChain(logits, r.expert_bias, w.top_k, r.norm, r.scale, g.s));
        },
        .softmax => {
            if (try router_kernel.moeRouterTopK(g.s, logits, .{ .ctx = null }, w.top_k, .softmax, true, 1.0, ops.dtype(logits), 0, 0, fold)) |r| return adoptRouting(g, r);
            if (w.route_fn.ctx != null) if (try compiled.apply(g, w.route_fn, &.{logits}, 2)) |o| return .{ .inds = o[0], .scores = o[1] };
            return adoptRouting(g, try routing.moeRoutingChain(logits, w.top_k, g.s));
        },
    }
}

// ── the MLP ──

/// `act(gate) * up`: SiLU, clamped where the pack clamps.
fn geglu(w: *const Moe.Weights, parent: *Graph, gate: Array, up: Array) !Array {
    const s = parent.s;
    if (w.clamp_limit > 0) {
        if (try fused_act.fusedClampedSwiGLU(s, gate, up, w.clamp_limit)) |y| return parent.adopt(y);
        return parent.adopt(try clampedSwiGLU(s, gate, up, w.clamp_limit));
    }
    if (try fused_act.fusedSwiGLU(s, gate, up)) |y| return parent.adopt(y);
    var t = parent.scope();
    defer t.deinit();
    return parent.keep(try t.mul(try t.mul(gate, try t.sigmoid(gate)), up));
}

pub fn mlp(w: *const Moe.Weights, parent: *Graph, x: Array) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const logits = try routerLogits(w, g, x);

    // At one row the shared expert's gate logit rides the softmax router kernel.
    const fold: ?router_kernel.RouterGate = blk: {
        const se = w.shared orelse break :blk null;
        if (w.route != .softmax or !se.gated() or se.gate_vec.ctx == null) break :blk null;
        const lsh = ops.shape(logits);
        var rows: usize = 1;
        for (lsh[0 .. lsh.len - 1]) |d| rows *= @intCast(d);
        if (rows != 1 or mlx.mlx_array_size(x) != mlx.mlx_array_size(se.gate_vec)) break :blk null;
        break :blk .{ .x = x, .w = se.gate_vec };
    };
    var r = try route(w, g, logits, fold);
    const shared_down: ?Array = if (r.gate_logit != null) try sharedDown(w, g, &w.shared.?, x) else null;
    // The group-limited chain hands its weights back in f32; they land in the activation dtype.
    if (ops.dtype(r.scores) != ops.dtype(x)) r.scores = try g.astype(r.scores, ops.dtype(x));
    if (w.route_scale.ctx != null) r.scores = try g.mul(r.scores, try g.call(mlx.mlx_take, .{ w.route_scale, r.inds }));
    if (w.fused) |*f| {
        const xs = ops.shape(x);
        if (xs[0] * xs[1] <= f.rows) if (try f.run(f, w, g.s, x, r.inds, r.scores)) |y| return parent.keep(try g.adopt(y));
    }
    const y = try routedExperts(w, g, x, r, shared_down, null);
    if (y.shared_fused) return parent.keep(y.sum);
    const se = w.shared orelse return parent.keep(y.sum);
    if (!se.gated()) return parent.keep(try g.add(y.sum, try sharedDown(w, g, &se, x)));
    const down = shared_down orelse try sharedDown(w, g, &se, x);
    const logit = r.gate_logit orelse try se.gate_logit.forward(g, x);
    return parent.keep(try g.add(y.sum, try g.mul(try g.sigmoid(logit), down)));
}

/// The routed experts' weighted sum `[B, S, D]` for routes `inds` and their weights `scores` (in `x`'s dtype), no
/// shared expert: `mlp`'s arms, as the Routed hook's contract reads them.
pub fn routedSum(w: *const Moe.Weights, parent: *Graph, x: Array, inds: Array, scores: Array) !Array {
    var t = parent.scope();
    defer t.deinit();
    return parent.keep((try routedExperts(w, &t, x, .{ .inds = inds, .scores = scores }, null, null)).sum);
}

const Summed = struct { sum: Array, shared_fused: bool };

/// The routed experts' sum; `shared_fused` when the kernel added the shared expert (`shared_down`) itself. `slots`: `w`'s
/// banks hold a streamed call's experts, whose sorted arm runs on the call's slot banks (`sortedSlots`).
fn routedExperts(w: *const Moe.Weights, g: *Graph, x: Array, r: Routed, shared_down: ?Array, slots: ?SlotCall) anyerror!Summed {
    const xs = ops.shape(x);
    const B = xs[0];
    const S = xs[1];
    const D = xs[xs.len - 1];
    const K = ops.shape(r.inds)[ops.shape(r.inds).len - 1];
    const total: c_int = B * S * K;
    const gq = w.gate.qp;
    const uq = w.up.qp;
    const dq = w.down.qp;
    const has_gate_scale = w.gate.expert_scale.ctx != null;
    const do_sort = B * S > 1 or total >= 64 or has_gate_scale;

    var out: Array = undefined;
    var reduced = false;
    var shared_fused = false;
    const ungated_a4: ?moe_affine4.Ungated = if (B * S != 1) null else ungatedA4(w, gq.group_size);
    const fp4_shared: ?moe_affine4.Shared = if (B * S == 1) if (shared_down) |sd| .{ .down = sd, .logit = r.gate_logit.? } else null else null;
    const cuda_fmt = if (mlx.cudaAvailable()) gather_qmv_rows.moeCudaFormat(B * S, gq.mode, uq.mode, dq.mode, true, false, w.up.expert_scale.ctx != null, w.clamp_limit > 0) else null;
    if (w.format_banks) |fb| {
        out = try formatSwiglu(w, &fb, g, x, r);
        reduced = true;
    } else if (w.streamed) |*st| {
        const y = try streamedRouted(w, st, g, x, r, shared_down, fp4_shared, cuda_fmt != null);
        out = y.out;
        reduced = y.reduced;
        shared_fused = y.shared_fused;
    } else if (moe_decode.moeFp4DecodeEligible(B * S, false, w.clamp, ops.dtype(x), gq.mode, uq.mode, dq.mode) and
        try fp4Decode(w, g, &out, x, r, fp4_shared))
    {
        reduced = true;
        shared_fused = fp4_shared != null;
    } else if (B * S == 1 and (w.clamp orelse 1) == 0 and gq.mode == .affine and uq.mode == .affine and dq.mode == .affine and
        gq.bits == 4 and uq.bits == 4 and dq.bits == 4 and gq.group_size == uq.group_size and gq.group_size == dq.group_size and
        try affine4Decode(w, g, &out, x, r, if (shared_down) |sd| .{ .down = sd, .logit = r.gate_logit.? } else null, ungated_a4))
    {
        reduced = true;
        shared_fused = shared_down != null or ungated_a4 != null;
    } else if (gather_qmv_rows.moeDecodeDispatchArm(B, S, K, false, (w.clamp orelse 1) != 0) == .rows and
        useGatherQmv(w) and w.gate.s.ctx != null and w.up.s.ctx != null and w.down.s.ctx != null and
        try gatherQmvRows(w, g, &out, x, r, D, K, B, S))
    {
        gather_qmv_rows.moeStats().moe_rows_fused_layers +%= 1;
        reduced = true;
    } else if (cuda_fmt != null and try cudaDecode(w, g, &out, x, r, cuda_fmt.?)) {
        reduced = true;
    } else if (do_sort) {
        gather_qmv_rows.moeStats().moe_sorted_layers +%= 1;
        const s = if (slots) |sc| try sortedSlots(w, g, x, r, sc.sl, sc.n, B, S, D, K) else try sorted(w, g, x, r, B, S, D, K);
        out = s.out;
        reduced = s.reduced;
    } else if (B * S == 1 and useGatherQmv(w) and w.gate.s.ctx != null and w.up.s.ctx != null and w.down.s.ctx != null and
        try gatherQmvDecode(w, g, &out, &reduced, x, r, D, K, B, S))
    {} else {
        const x5 = try g.reshape(x, &.{ B, S, 1, 1, D });
        const gate = try g.call(mlx.mlx_squeeze, .{try expertMm(g, x5, &w.gate, .{ .ctx = null }, r.inds, false)});
        const up = try g.call(mlx.mlx_squeeze, .{try expertMm(g, x5, &w.up, .{ .ctx = null }, r.inds, false)});
        const act = try g.expandDims(try geglu(w, g, gate, up), -2);
        out = try g.call(mlx.mlx_squeeze, .{try expertMm(g, act, &w.down, .{ .ctx = null }, r.inds, false)});
    }

    return .{ .sum = if (reduced) out else try g.sum(try g.mul(out, try g.expandDims(r.scores, -1)), -2, false), .shared_fused = shared_fused };
}

/// The format's routed SwiGLU over the top-k, weighted and summed (the engine's exl3 arm).
fn formatSwiglu(w: *const Moe.Weights, fb: *const FormatBanks, g: *Graph, x: Array, r: Routed) !Array {
    const swiglu = fb.table.swiglu orelse return error.QuantFormatHasNoSwiglu;
    const limit = w.clamp orelse return error.QuantFormatClampUnsupported;
    var y: quant_abi.Array = .{};
    const v = swiglu(&fb.gate, &fb.up, &fb.down, .{ .ctx = x.ctx }, .{ .ctx = r.inds.ctx }, .{ .ctx = r.scores.ctx }, &.{ .limit = @intCast(limit) }, .{ .ctx = g.s.ctx }, &y);
    if (!v.isOk()) return error.QuantFormatSwigluFailed;
    return g.adopt(.{ .ctx = y.ctx });
}

fn sharedDown(w: *const Moe.Weights, parent: *Graph, se: *const SharedExpert, x: Array) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    return parent.keep(try se.down.forward(g, try geglu(w, g, try se.gate.forward(g, x), try se.up.forward(g, x))));
}

/// An ungated shared expert the affine-4 kernels carry: 4-bit affine at the routed group size.
fn ungatedA4(w: *const Moe.Weights, group_size: u32) ?moe_affine4.Ungated {
    const se = w.shared orelse return null;
    if (se.gated() or se.gate.s.ctx == null) return null;
    for ([_]*const Linear{ &se.gate, &se.up, &se.down }) |l| {
        if (l.mode != .affine or l.bits != 4 or l.group_size != group_size) return null;
    }
    return .{
        .gate = .{ .w = se.gate.w, .s = se.gate.s, .b = se.gate.b },
        .up = .{ .w = se.up.w, .s = se.up.s, .b = se.up.b },
        .down = .{ .w = se.down.w, .s = se.down.s, .b = se.down.b },
    };
}

fn useGatherQmv(w: *const Moe.Weights) bool {
    if (w.gate.qp.mode == .gguf) return false;
    if (w.clamp == null) return false;
    if (!gather_qmv.gatherQmvGateUpEnabled()) return false;
    return w.gate.qp.bits == w.up.qp.bits and w.gate.qp.group_size == w.up.qp.group_size and w.gate.qp.mode == w.up.qp.mode;
}

fn expertMm(g: *Graph, x: Array, bank: *const Bank, lhs: Array, rhs: Array, is_sorted: bool) !Array {
    var y = mlx.mlx_array_new();
    gatherExpertMm(&y, x, bank.w, bank.s, bank.b, lhs, rhs, bank.qp.bits, bank.qp.group_size, bank.qp.mode, is_sorted, g.s) catch |e| {
        _ = mlx.mlx_array_free(y);
        return e;
    };
    return g.adopt(y);
}

/// Owned by `g`, or false (`out` untouched) when the kernels decline.
fn put(g: *Graph, out: *Array, y: ?Array, shape: []const c_int) !bool {
    const v = y orelse return false;
    out.* = try g.reshape(try g.adopt(v), shape);
    return true;
}

fn fp4Decode(w: *const Moe.Weights, g: *Graph, out: *Array, x: Array, r: Routed, shared: ?moe_affine4.Shared) !bool {
    const xs = ops.shape(x);
    const rows = xs[0] * xs[1];
    const k = ops.shape(r.inds)[ops.shape(r.inds).len - 1];
    const x2 = try g.reshape(x, &.{ rows, xs[2] });
    const ind = try g.astype(try g.reshape(r.inds, &.{ rows, k }), .uint32);
    const sc = try g.reshape(r.scores, &.{ rows, k });
    const sigtab = try fused_act.swigluSigTable(g.s, ops.dtype(x), std.heap.c_allocator);
    const scales: moe_fp4.Scales = .{
        .gate = if (w.gate.expert_scale.ctx != null) w.gate.expert_scale else null,
        .up = if (w.up.expert_scale.ctx != null) w.up.expert_scale else null,
        .clamp = w.clamp orelse return false,
    };
    const y = try moe_fp4.decode(g.s, x2, .{ .w = w.gate.w, .s = w.gate.s }, .{ .w = w.up.w, .s = w.up.s }, .{ .w = w.down.w, .s = w.down.s }, w.bank_views, scales, shared, ind, sc, sigtab, w.gate.qp.mode, w.gate.qp.group_size);
    return put(g, out, y, &.{ xs[0], xs[1], xs[2] });
}

fn affine4Decode(w: *const Moe.Weights, g: *Graph, out: *Array, x: Array, r: Routed, shared: ?moe_affine4.Shared, ungated: ?moe_affine4.Ungated) !bool {
    // The sigmoid table covers 16-bit patterns only.
    const xd = ops.dtype(x);
    if (xd != .bfloat16 and xd != .float16) return false;
    const xs = ops.shape(x);
    const k = ops.shape(r.inds)[ops.shape(r.inds).len - 1];
    const x1 = try g.reshape(x, &.{xs[2]});
    const ind = try g.astype(try g.reshape(r.inds, &.{k}), .uint32);
    const sc = try g.reshape(r.scores, &.{k});
    const sigtab = try fused_act.swigluSigTable(g.s, xd, std.heap.c_allocator);
    const y = try moe_affine4.decode(g.s, x1, w.gate.a4(), w.up.a4(), w.down.a4(), w.bank_views, shared, ungated, ind, sc, sigtab, w.gate.qp.group_size);
    return put(g, out, y, &.{ xs[0], xs[1], xs[2] });
}

fn cudaDecode(w: *const Moe.Weights, g: *Graph, out: *Array, x: Array, r: Routed, fmt: moe_cuda.Format) !bool {
    const xs = ops.shape(x);
    const rows = xs[0] * xs[1];
    const k = ops.shape(r.inds)[ops.shape(r.inds).len - 1];
    const x2 = try g.reshape(x, &.{ rows, xs[2] });
    const ind = try g.astype(try g.reshape(r.inds, &.{ rows, k }), .uint32);
    const sc = try g.reshape(r.scores, &.{ rows, k });
    const gscale: ?Array = if (w.gate.expert_scale.ctx != null) w.gate.expert_scale else null;
    const y = try moe_cuda.decode(g.s, x2, .{ .w = w.gate.w, .s = w.gate.s }, .{ .w = w.up.w, .s = w.up.s }, .{ .w = w.down.w, .s = w.down.s }, gscale, ind, sc, fmt);
    return put(g, out, y, &.{ xs[0], xs[1], xs[2] });
}

fn gatherQmvRows(w: *const Moe.Weights, g: *Graph, out: *Array, x: Array, r: Routed, D: c_int, K: c_int, batch: c_int, seq: c_int) !bool {
    const B = batch * seq;
    if (!down_reduce.gatherQmvDownReduceRowsEligible(HiddenAct.silu, w.gate.qp, w.up.qp, w.down.qp, D)) return false;
    const ind = try g.astype(try g.reshape(r.inds, &.{ B, K }), .uint32);
    const x2 = try g.reshape(x, &.{ B, D });
    const gscale: ?Array = if (w.gate.expert_scale.ctx != null) w.gate.expert_scale else null;
    const act = try g.adopt((try gather_qmv_rows.gatherQmvGateUpRows(g.s, x2, w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, ind, w.gate.qp.bits, w.gate.qp.group_size, w.gate.qp.mode, gscale)) orelse return false);
    const sc = try g.reshape(r.scores, &.{ B, K });
    const sum = (try down_reduce.gatherQmvDownReduceRows(g.s, act, w.down.w, w.down.s, w.down.b, ind, sc, w.down.qp.bits, w.down.qp.group_size, w.down.qp.mode)) orelse return false;
    const hidden = ops.shape(sum)[1];
    _ = try put(g, out, sum, &.{ batch, seq, hidden });
    if (seq > 1 and !gather_qmv_rows.moe_verify_rows_logged) {
        gather_qmv_rows.moe_verify_rows_logged = true;
        log.info("[moe-rows] verify rows engaged: S={d} K={d}\n", .{ seq, K });
    }
    return true;
}

fn gatherQmvDecode(w: *const Moe.Weights, g: *Graph, out: *Array, reduced: *bool, x: Array, r: Routed, D: c_int, K: c_int, B: c_int, S: c_int) !bool {
    const ind = try g.astype(try g.reshape(r.inds, &.{K}), .uint32);
    const x1 = try g.reshape(x, &.{D});
    const gq = w.gate.qp;
    const uq = w.up.qp;
    const dq = w.down.qp;
    var act: ?Array = null;
    if (gq.bits == uq.bits and gq.group_size == uq.group_size and gq.mode == uq.mode) {
        if (try gather_qmv.gatherQmvGateUp(g.s, x1, w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, ind, gq.bits, gq.group_size, gq.mode, w.clamp.?)) |y|
            act = try g.adopt(y);
    }
    if (act == null) {
        const gate2 = try g.adopt((try gather_qmv.gatherQmv(g.s, x1, w.gate.w, w.gate.s, w.gate.b, ind, gq.bits, gq.group_size, gq.mode, false)) orelse return false);
        const up2 = try g.adopt((try gather_qmv.gatherQmv(g.s, x1, w.up.w, w.up.s, w.up.b, ind, uq.bits, uq.group_size, uq.mode, false)) orelse return false);
        const inter = ops.shape(gate2)[1];
        const e = try geglu(w, g, try g.reshape(gate2, &.{ K, 1, inter }), try g.reshape(up2, &.{ K, 1, inter }));
        act = try g.reshape(e, &.{ K, inter });
    }
    const inter = ops.shape(act.?)[1];
    const sc = try g.reshape(r.scores, &.{K});
    if (try down_reduce.gatherQmvDownReduce(g.s, act.?, w.down.w, w.down.s, w.down.b, ind, sc, dq.bits, dq.group_size, dq.mode)) |sum| {
        const hidden = ops.shape(sum)[0];
        _ = try put(g, out, sum, &.{ B, S, hidden });
        reduced.* = true;
        if (!down_reduce.downred_engaged) {
            down_reduce.downred_engaged = true;
            log.info("[moe] fused down+reduce kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, dq.bits, dq.group_size });
        }
        if (!gather_qmv.gqmv_engaged) {
            gather_qmv.gqmv_engaged = true;
            log.info("[moe] gather-qmv kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, gq.bits, gq.group_size });
        }
        return true;
    }
    const down2 = try g.adopt((try gather_qmv.gatherQmv(g.s, act.?, w.down.w, w.down.s, w.down.b, ind, dq.bits, dq.group_size, dq.mode, true)) orelse return false);
    const hidden = ops.shape(down2)[1];
    out.* = try g.reshape(down2, &.{ B, S, K, hidden });
    if (!gather_qmv.gqmv_engaged) {
        gather_qmv.gqmv_engaged = true;
        log.info("[moe] gather-qmv kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, gq.bits, gq.group_size });
    }
    return true;
}

/// `w`'s per-expert scale on `y`'s rows (by `sel`), as `scaleExperts`.
fn scaleExperts(parent: *Graph, y: Array, scale: Array, sel: Array) !Array {
    if (scale.ctx == null) return y;
    var t = parent.scope();
    defer t.deinit();
    return parent.keep(try t.mul(y, try t.expandDims(try t.take(scale, sel, 0), -1)));
}

/// mlx-lm's `_gather_sort`: every (token, expert) assignment sorted by expert so consecutive rows read one bank,
/// the down output weighted and summed back in token order.
fn sorted(w: *const Moe.Weights, g: *Graph, x: Array, r: Routed, B: c_int, S: c_int, D: c_int, K: c_int) !Sorted {
    const total = B * S * K;
    const flat = try g.reshape(r.inds, &.{total});
    var inv: Array = undefined;
    var sorted_inds: Array = undefined;
    var lhs: Array = undefined;
    if (try verify_kernels.moePrefillRoutePack(g.s, flat, K, w.experts)) |p| {
        inv = try g.adopt(p.inverse);
        sorted_inds = try g.adopt(p.sorted);
        lhs = try g.adopt(p.lhs);
        if (!route_pack_logged) {
            route_pack_logged = true;
            log.info("[moe] prefill counting-sort route packing engaged (assignments={d})\n", .{total});
        }
    } else {
        const order = try g.argsort(flat, 0);
        inv = try g.argsort(order, 0);
        sorted_inds = try g.take(flat, order, 0);
        lhs = try g.call(mlx.mlx_floor_divide, .{ order, try g.adopt(mlx.mlx_array_new_int(K)) });
    }
    const experts = ops.shape(w.gate.w)[0];
    const row_block = total >= 16 and experts > 0 and @divTrunc(total, experts) >= 4;
    const down = try sortedDown(w, g, try g.reshape(x, &.{ B * S, D }), lhs, &w.gate, &w.up, &w.down, sorted_inds, row_block, true);
    return sortedSum(g, down, inv, r.scores, B, S, K);
}

/// The sorted rows `down` `[n, D]` back in the router's order (`inv`), weighted and summed where the fused kernel serves.
fn sortedSum(g: *Graph, down: Array, inv: Array, scores: Array, B: c_int, S: c_int, K: c_int) !Sorted {
    if (try verify_kernels.moePrefillExpertReduce(g.s, down, inv, scores)) |sum| {
        if (!prefill_reduce_logged) {
            prefill_reduce_logged = true;
            log.info("[moe] fused prefill weighted sum engaged (tokens={d})\n", .{B * S});
        }
        return .{ .out = try g.adopt(sum), .reduced = true };
    }
    const unsorted = try g.take(down, inv, 0);
    return .{ .out = try g.reshape(unsorted, &.{ B, S, K, ops.shape(unsorted)[1] }), .reduced = false };
}

const Sorted = struct { out: Array, reduced: bool, shared_fused: bool = false };

/// One sorted call's down rows `[n, D]` over `gate`/`up`/`down` at rows `idx` (sorted), token `lhs[i]` of `x_flat` for
/// row i: in `blocked` (MLX's row-block class) the NAX kernels where they serve the bank, else gathers of the gathered
/// rows (`is_sorted` on MLX's).
fn sortedDown(w: *const Moe.Weights, g: *Graph, x_flat: Array, lhs: Array, gate: *const Bank, up: *const Bank, down: *const Bank, idx: Array, blocked: bool, is_sorted: bool) !Array {
    const gq = gate.qp;
    const uq = up.qp;
    const n = ops.shape(idx)[0];
    const D = ops.shape(x_flat)[1];
    var act: ?Array = null;
    var mapped_gate: ?Array = null;
    var mapped_up: ?Array = null;
    const nax = verifyQmmNaxAvailable();
    if (blocked and nax and gq.mode == uq.mode and (gq.mode == .affine or gq.mode.isFp4())) {
        const x_tok = try g.reshape(x_flat, &.{ ops.shape(x_flat)[0], 1, D });
        const row_map = try g.astype(lhs, .uint32);
        if (fused_act.swigluFusedEnabled() and w.clamp != null and gate.expert_scale.ctx == null and up.expert_scale.ctx == null and
            gq.bits == uq.bits and gq.group_size == uq.group_size)
        {
            const sigtab = try sigmoidTable(g.s);
            if (try gather_qmm_nax.sortedGateUp(x_tok, row_map, gate.w, gate.s, gate.b, up.w, up.s, up.b, idx, sigtab, w.clamp.?, gq.bits, gq.group_size, gq.mode.isFp4(), nax, g.s)) |a3| {
                const sh = ops.shape(a3);
                act = try g.reshape(try g.adopt(a3), &.{ sh[0], sh[2] });
            }
        }
        if (act == null) {
            mapped_gate = if (try gather_qmm_nax.sortedGatherMapped(x_tok, row_map, gate.w, gate.s, gate.b, idx, gq.bits, gq.group_size, gq.mode.isFp4(), nax, g.s)) |y| try g.adopt(y) else null;
            mapped_up = if (try gather_qmm_nax.sortedGatherMapped(x_tok, row_map, up.w, up.s, up.b, idx, uq.bits, uq.group_size, uq.mode.isFp4(), nax, g.s)) |y| try g.adopt(y) else null;
            if (mapped_gate == null or mapped_up == null) {
                mapped_gate = null;
                mapped_up = null;
            }
        }
    }
    if (act == null and mapped_gate == null and mlx.cudaAvailable() and w.clamp_limit == 0 and
        up.expert_scale.ctx == null and gq.mode == uq.mode and (gq.mode == .nvfp4 or gq.mode == .mxfp8))
    {
        const row_map = try g.astype(lhs, .uint32);
        const fmt: moe_prefill_cuda.Format = if (gq.mode == .nvfp4) .nvfp4 else .mxfp8;
        const gscale: ?Array = if (gate.expert_scale.ctx != null) gate.expert_scale else null;
        if (try moe_prefill_cuda.gateUp(g.s, x_flat, row_map, idx, gate.w, gate.s, up.w, up.s, gscale, fmt)) |y| act = try g.adopt(y);
    }
    const expert_act = act orelse blk: {
        const x_rep: ?Array = if (mapped_gate == null) try g.reshape(try g.take(x_flat, lhs, 0), &.{ n, 1, D }) else null;
        const none = Array{ .ctx = null };
        const gate3 = mapped_gate orelse try expertMm(g, x_rep.?, gate, none, idx, is_sorted);
        const gate_y = try scaleExperts(g, try rows2(g, gate3, n), gate.expert_scale, idx);
        const up3 = mapped_up orelse try expertMm(g, x_rep.?, up, none, idx, is_sorted);
        const up_y = try scaleExperts(g, try rows2(g, up3, n), up.expert_scale, idx);
        break :blk try geglu(w, g, gate_y, up_y);
    };
    return rows2(g, try expertMm(g, try g.expandDims(expert_act, -2), down, .{ .ctx = null }, idx, is_sorted), n);
}

/// `y` as `[n, width]` (a gather's `[n, 1, width]`, or the NAX kernels' rows).
fn rows2(g: *Graph, y: Array, n: c_int) !Array {
    const sh = ops.shape(y);
    return g.reshape(y, &.{ n, sh[sh.len - 1] });
}

// ── streamed experts ──

/// A call's slot banks (`routed.Fetched`): per bank its gate, up and down at the pack's quant params, and per routed
/// pair (the fetched ids' order) its bank and row.
const Slots = struct { banks: [3]?[3]Bank = @splat(null), kind: []const u8, row: []const u32 };

/// A streamed call's slots and the call's routed pairs, whose count sets the sorted gathers' kernel class.
const SlotCall = struct { sl: *const Slots, n: usize };

/// The routes `inds` as host expert ids (owned by `g.a`).
fn hostIds(g: *Graph, inds: Array) ![]u16 {
    const host = try g.astype(inds, .uint32);
    try mlx.check(mlx.mlx_array_eval(host));
    const n = mlx.mlx_array_size(host);
    if (n > routed.max_call_ids) return error.StreamedCallTooWide;
    const src = (mlx.mlx_array_data_uint32(host) orelse return error.NoData)[0..n];
    const ids = try g.a.alloc(u16, n);
    for (ids, src) |*d, v| d.* = @intCast(v);
    return ids;
}

/// A decode call's prediction of the next routed layer (`Streamed.next`) as host f32, evaluated with the routes;
/// empty past the decode lane's rows or without a predictor.
fn nextScores(st: *const Streamed, g: *Graph, x: Array, inds: Array) ![]const f32 {
    const nx = st.next orelse return &.{};
    const xs = ops.shape(x);
    const rows = xs[0] * xs[1];
    if (rows > 8) return &.{};
    const sc = try g.add(try g.sigmoid(try g.matmul(try g.astype(try g.reshape(x, &.{ rows, xs[2] }), .float32), nx.router)), nx.bias);
    try ops.evalMany(&.{ sc, inds });
    return (mlx.mlx_array_data_float32(sc) orelse return error.NoData)[0..mlx.mlx_array_size(sc)];
}

fn fetchSlots(w: *const Moe.Weights, st: *const Streamed, ids: []const u16) !Slots {
    return fetchSlotsWith(w, st, ids, &.{});
}

fn fetchSlotsWith(w: *const Moe.Weights, st: *const Streamed, ids: []const u16, scores: []const f32) !Slots {
    const f = try st.routed.math.trunk(st.routed.ctx, st.layer, ids, scores).get();
    var out: Slots = .{ .kind = f.kind, .row = f.row };
    for (f.banks, &out.banks) |maybe, *o| if (maybe) |bk| {
        o.* = .{
            .{ .w = bk.gate.w, .s = bk.gate.scales, .b = bk.gate.biases, .qp = w.gate.qp },
            .{ .w = bk.up.w, .s = bk.up.scales, .b = bk.up.biases, .qp = w.up.qp },
            .{ .w = bk.down.w, .s = bk.down.scales, .b = bk.down.biases, .qp = w.down.qp },
        };
    };
    return out;
}

/// Distinct experts among `ids`.
fn distinctCount(ids: []const u16) usize {
    var seen = std.StaticBitSet(1 << 16).empty;
    for (ids) |e| seen.set(e);
    return seen.count();
}

/// The routed experts of a streamed bank: the source's sum (`source`), or (`trunk`) the in-memory arms run on the call's
/// slots, so each pair's rows and the k-sum's order are the arm's over the in-memory banks: one token and the rows arm
/// on one bank holding the call's experts (`slotArms`), the sorted gathers on the slot banks a group of experts at a
/// time (`sortedGroups`).
fn streamedRouted(w: *const Moe.Weights, st: *const Streamed, g: *Graph, x: Array, r: Routed, shared_down: ?Array, fp4_shared: ?moe_affine4.Shared, cuda: bool) anyerror!Sorted {
    const xs = ops.shape(x);
    const B = xs[0];
    const S = xs[1];
    const D = xs[xs.len - 1];
    const K = ops.shape(r.inds)[ops.shape(r.inds).len - 1];
    switch (st.routed.math) {
        .source => |f| {
            const y = try g.adopt(try f(st.routed.ctx, st.layer, try g.reshape(x, &.{ B * S, D }), try g.reshape(r.inds, &.{ B * S, K }), try g.reshape(r.scores, &.{ B * S, K })).get());
            return .{ .out = try g.reshape(try g.astype(y, ops.dtype(x)), &.{ B, S, D }), .reduced = true };
        },
        .trunk => {},
    }
    if (cuda) return error.StreamedArmUnsupported;
    if (w.gate.expert_scale.ctx != null or w.up.expert_scale.ctx != null) return error.StreamedExpertScales;
    if (@as(usize, @intCast(B * S * K)) > routed.max_call_ids) return wideCall(w, st, g, x, r, S, K);
    const scores = try nextScores(st, g, x, r.inds);
    const ids = try hostIds(g, r.inds);
    defer g.a.free(ids);
    const n: usize = ids.len;
    const gq = w.gate.qp;
    if (moe_decode.moeFp4DecodeEligible(B * S, false, w.clamp, ops.dtype(x), gq.mode, w.up.qp.mode, w.down.qp.mode)) {
        const sl = try fetchSlotsWith(w, st, ids, scores);
        if (try fp4DecodeSlots(w, g, x, r, &sl, fp4_shared)) |y| return .{ .out = y, .reduced = true, .shared_fused = fp4_shared != null };
        return slotArms(w, g, x, r, &sl, shared_down, n);
    }
    const one_row = B * S == 1;
    const rows_arm = gather_qmv_rows.moeDecodeDispatchArm(B, S, K, false, (w.clamp orelse 1) != 0) == .rows;
    if (!one_row and !rows_arm) return sortedGroups(w, st, g, x, r, ids, B, S, D, K);
    if (distinctCount(ids) <= routed.max_call_experts) return slotArms(w, g, x, r, &(try fetchSlotsWith(w, st, ids, scores)), shared_down, n);
    return rowChunks(w, st, g, x, r, ids, B, S, D, K);
}

/// A prompt chunk past the bank's pair cap (`routed.max_call_ids`): its rows in calls under the cap, each piece summed
/// over its top-k.
fn wideCall(w: *const Moe.Weights, st: *const Streamed, g: *Graph, x: Array, r: Routed, S: c_int, K: c_int) anyerror!Sorted {
    if (ops.shape(x)[0] != 1) return error.StreamedCallTooWide;
    // Even pieces: a short tail would take a narrower arm's numerics than the rest of the chunk.
    const per: usize = routed.max_call_ids / @as(usize, @intCast(K));
    const rows: usize = @intCast(S);
    const pieces = std.math.divCeil(usize, rows, per) catch unreachable;
    var outs: std.ArrayList(Array) = .empty;
    defer outs.deinit(g.a);
    for (0..pieces) |c| {
        const lo: c_int = @intCast(rows * c / pieces);
        const hi: c_int = @intCast(rows * (c + 1) / pieces);
        const rc: Routed = .{
            .inds = try g.sliceAxis(r.inds, 1, lo, hi),
            .scores = try g.sliceAxis(r.scores, 1, lo, hi),
            .gate_logit = if (r.gate_logit) |gl| try g.sliceAxis(gl, 1, lo, hi) else null,
        };
        const y = try streamedRouted(w, st, g, try g.sliceAxis(x, 1, lo, hi), rc, null, null, false);
        try outs.append(g.a, if (y.reduced) y.out else try g.sum(try g.mul(y.out, try g.expandDims(rc.scores, -1)), -2, false));
    }
    return .{ .out = try g.concat(outs.items, 1), .reduced = true };
}

/// The in-memory arms of `routedExperts` on a bank holding the call's experts (`callBank`), every pair's route its row
/// there; the sorted arm on the slot banks themselves, in the class of the call's `n` pairs.
fn slotArms(w: *const Moe.Weights, g: *Graph, x: Array, r: Routed, sl: *const Slots, shared_down: ?Array, n: usize) anyerror!Sorted {
    const k = ops.shape(r.inds).len;
    const cb = try callBank(g, sl, ops.shape(r.inds)[0..k]);
    var wv = w.*;
    wv.gate = cb.banks[0];
    wv.up = cb.banks[1];
    wv.down = cb.banks[2];
    wv.bank_views = null;
    wv.streamed = null;
    const y = try routedExperts(&wv, g, x, .{ .inds = cb.inds, .scores = r.scores, .gate_logit = r.gate_logit }, shared_down, .{ .sl = sl, .n = n });
    return .{ .out = y.sum, .reduced = true, .shared_fused = y.shared_fused };
}

/// One bank holding the call's experts and each pair's row in it (`shape`: the routes'): the slot bank itself when
/// every pair sits in one, else the call's slots gathered from their banks (a pair's rows are a copy of its slot's).
fn callBank(g: *Graph, sl: *const Slots, shape: []const c_int) !struct { banks: [3]Bank, inds: Array } {
    const n = sl.kind.len;
    const first = sl.kind[0];
    const one = for (sl.kind) |k| {
        if (k != first) break false;
    } else true;
    if (one) return .{ .banks = sl.banks[first].?, .inds = try g.fromSlice(u32, sl.row, shape) };
    const at = try g.a.alloc(u32, n);
    defer g.a.free(at);
    var taken: [3]std.ArrayList(u32) = @splat(.empty);
    defer for (&taken) |*t| t.deinit(g.a);
    for (sl.kind, sl.row, at) |b, row, *o| {
        const i = std.mem.indexOfScalar(u32, taken[b].items, row) orelse blk: {
            try taken[b].append(g.a, row);
            break :blk taken[b].items.len - 1;
        };
        o.* = @intCast(i);
    }
    var base: [3]u32 = undefined;
    var total: u32 = 0;
    for (taken, &base) |t, *bs| {
        bs.* = total;
        total += @intCast(t.items.len);
    }
    for (sl.kind, at) |b, *o| o.* += base[b];
    var banks: [3]Bank = undefined;
    for (&banks, 0..) |*out, p| {
        var parts: [3][3]Array = undefined;
        var np: usize = 0;
        for (sl.banks, taken) |maybe, t| if (t.items.len > 0) {
            const bk = &maybe.?[p];
            const rows = try g.fromSlice(u32, t.items, &.{@intCast(t.items.len)});
            parts[np] = .{ try g.take(bk.w, rows, 0), try g.take(bk.s, rows, 0), if (bk.b.ctx != null) try g.take(bk.b, rows, 0) else .{ .ctx = null } };
            np += 1;
        };
        var cat: [3]Array = undefined;
        for (0..3) |c| {
            if (parts[0][c].ctx == null) {
                cat[c] = .{ .ctx = null };
                continue;
            }
            var list: [3]Array = undefined;
            for (parts[0..np], list[0..np]) |pt, *l| l.* = pt[c];
            cat[c] = try g.concat(list[0..np], 0);
        }
        out.* = .{ .w = cat[0], .s = cat[1], .b = cat[2], .qp = sl.banks[first].?[p].qp };
    }
    return .{ .banks = banks, .inds = try g.fromSlice(u32, at, shape) };
}

/// A call of 2..8 rows whose experts outnumber one fetch: its rows in chunks of at least two (the arm the whole call
/// takes; each row's sum is its own), each fetched, run and evaluated before the next.
fn rowChunks(w: *const Moe.Weights, st: *const Streamed, g: *Graph, x: Array, r: Routed, ids: []const u16, B: c_int, S: c_int, D: c_int, K: c_int) anyerror!Sorted {
    const rows: usize = @intCast(B * S);
    const k: usize = @intCast(K);
    const per = routed.max_call_experts / k;
    if (per < 2) return error.StreamedCallTooWide;
    const chunks = std.math.divCeil(usize, rows, per) catch unreachable;
    const xf = try g.reshape(x, &.{ B * S, D });
    const inds = try g.reshape(r.inds, &.{ B * S, K });
    const scores = try g.reshape(r.scores, &.{ B * S, K });
    var outs: std.ArrayList(Array) = .empty;
    defer outs.deinit(g.a);
    var lo: usize = 0;
    for (0..chunks) |c| {
        const hi = rows * (c + 1) / chunks;
        const m: c_int = @intCast(hi - lo);
        const sh: [2]c_int = if (B == 1) .{ 1, m } else .{ m, 1 };
        const lo_i: c_int = @intCast(lo);
        const hi_i: c_int = @intCast(hi);
        const rc: Routed = .{
            .inds = try g.reshape(try g.sliceAxis(inds, 0, lo_i, hi_i), &.{ sh[0], sh[1], K }),
            .scores = try g.reshape(try g.sliceAxis(scores, 0, lo_i, hi_i), &.{ sh[0], sh[1], K }),
        };
        const sl = try fetchSlots(w, st, ids[lo * k .. hi * k]);
        const y = try slotArms(w, g, try g.reshape(try g.sliceAxis(xf, 0, lo_i, hi_i), &.{ sh[0], sh[1], D }), rc, &sl, null, ids.len);
        const yr = try g.reshape(y.out, &.{ m, D });
        try mlx.check(mlx.mlx_array_eval(yr));
        try outs.append(g.a, yr);
        lo = hi;
    }
    return .{ .out = try g.reshape(try g.concat(outs.items, 0), &.{ B, S, D }), .reduced = true };
}

/// `sorted` on a streamed bank: the call's distinct experts a fetch's worth at a time, each group's pairs gathered on its
/// slot banks in the class of the whole call (`slotDown`) and evaluated before the next group's fetch, the rows put back
/// in the router's order and weighted and summed as `sorted` does.
fn sortedGroups(w: *const Moe.Weights, st: *const Streamed, g: *Graph, x: Array, r: Routed, ids: []const u16, B: c_int, S: c_int, D: c_int, K: c_int) anyerror!Sorted {
    const a = g.a;
    const n = ids.len;
    var experts = std.StaticBitSet(1 << 16).empty;
    for (ids) |e| experts.set(e);
    const x_flat = try g.reshape(x, &.{ B * S, D });
    const order = try a.alloc(u32, n);
    defer a.free(order);
    const group_ids = try a.alloc(u16, n);
    defer a.free(group_ids);
    var parts: std.ArrayList(Array) = .empty;
    defer parts.deinit(a);
    var at: usize = 0;
    var it = experts.iterator(.{});
    var group = std.StaticBitSet(1 << 16).empty;
    var more = true;
    while (more) {
        group = std.StaticBitSet(1 << 16).empty;
        var count: usize = 0;
        while (count < routed.max_call_experts) : (count += 1) group.set(it.next() orelse {
            more = false;
            break;
        });
        if (count == 0) break;
        const first = at;
        for (ids, 0..) |e, i| if (group.isSet(e)) {
            order[at] = @intCast(i);
            group_ids[at - first] = e;
            at += 1;
        };
        const sl = try fetchSlots(w, st, group_ids[0 .. at - first]);
        const d = try slotDown(w, g, x_flat, &sl, order[first..at], n, K);
        try mlx.check(mlx.mlx_array_eval(d));
        try parts.append(a, d);
    }
    const inv = try a.alloc(u32, n);
    defer a.free(inv);
    for (order, 0..) |p, j| inv[p] = @intCast(j);
    const down = try g.take(try g.concat(parts.items, 0), try g.fromSlice(u32, inv, &.{@intCast(n)}), 0);
    return sortedSum(g, down, try g.arange(0, @floatFromInt(n), 1, .uint32), r.scores, B, S, K);
}

/// The slot banks present, in order, and each pair's index among them (`kinds`).
fn presentBanks(sl: *const Slots, list: *[3][3]Bank, kinds: []u32) usize {
    var at: [3]u32 = undefined;
    var n: usize = 0;
    for (sl.banks, 0..) |maybe, b| if (maybe) |bk| {
        list[n] = bk;
        at[b] = @intCast(n);
        n += 1;
    };
    for (kinds, sl.kind[0..kinds.len]) |*k, b| k.* = at[b];
    return n;
}

fn fp4DecodeSlots(w: *const Moe.Weights, g: *Graph, x: Array, r: Routed, sl: *const Slots, shared: ?moe_affine4.Shared) !?Array {
    const xs = ops.shape(x);
    const rows = xs[0] * xs[1];
    const k = ops.shape(r.inds)[ops.shape(r.inds).len - 1];
    const n: usize = @intCast(rows * k);
    var list: [3][3]Bank = undefined;
    const kinds = try g.a.alloc(u32, n);
    defer g.a.free(kinds);
    const nb = presentBanks(sl, &list, kinds);
    var fp: [3][3]moe_fp4.Bank = undefined;
    for (list[0..nb], fp[0..nb]) |bk, *o| o.* = .{ .{ .w = bk[0].w, .s = bk[0].s }, .{ .w = bk[1].w, .s = bk[1].s }, .{ .w = bk[2].w, .s = bk[2].s } };
    const x2 = try g.reshape(x, &.{ rows, xs[2] });
    const sc = try g.reshape(r.scores, &.{ rows, k });
    const sigtab = try fused_act.swigluSigTable(g.s, ops.dtype(x), std.heap.c_allocator);
    const y = try moe_fp4.decodeBanks(g.s, x2, fp[0..nb], try g.fromSlice(u32, kinds, &.{ rows, k }), .{ .clamp = w.clamp.? }, shared, try g.fromSlice(u32, sl.row[0..n], &.{ rows, k }), sc, sigtab, w.gate.qp.mode, w.gate.qp.group_size);
    const v = y orelse return null;
    return try g.reshape(try g.adopt(v), &.{ xs[0], xs[1], xs[2] });
}

/// `sorted` over the slot banks for a call of `n` pairs.
fn sortedSlots(w: *const Moe.Weights, g: *Graph, x: Array, r: Routed, sl: *const Slots, n: usize, B: c_int, S: c_int, D: c_int, K: c_int) !Sorted {
    const pairs = try g.a.alloc(u32, sl.kind.len);
    defer g.a.free(pairs);
    for (pairs, 0..) |*p, i| p.* = @intCast(i);
    const down = try slotDown(w, g, try g.reshape(x, &.{ B * S, D }), sl, pairs, n, K);
    return sortedSum(g, down, try g.arange(0, @floatFromInt(pairs.len), 1, .uint32), r.scores, B, S, K);
}

/// The down rows `[pairs.len, D]` of a call's pairs `pairs` (flat router indices, pair i at slot `sl.kind[i]` /
/// `sl.row[i]`), in `pairs`' order, each gather in the kernel class `sorted` takes for the whole call of `n` pairs. MLX
/// and the NAX gathers choose it by rows per expert, `n / E` over a bank of `E` rows, and within a class a row's sums are
/// its own. In the row-block class (`q = n / experts` >= 4) a bank's pairs run, sorted by row, in calls of `B` pairs
/// with `B / E` = q (padded with row 0's); below it, unsorted (the per-pair kernel, as MLX runs a sorted call there).
fn slotDown(w: *const Moe.Weights, g: *Graph, x_flat: Array, sl: *const Slots, pairs: []const u32, n: usize, K: c_int) !Array {
    const a = g.a;
    const m = pairs.len;
    const q: usize = n / @as(usize, @intCast(w.experts));
    const blocked = n >= 16 and q >= 4;
    const inv = try a.alloc(u32, m);
    defer a.free(inv);
    const own = try a.alloc(u32, m);
    defer a.free(own);
    var parts: std.ArrayList(Array) = .empty;
    defer parts.deinit(a);
    var at: u32 = 0;
    for (sl.banks, 0..) |maybe, b| if (maybe) |*bk| {
        var cnt: usize = 0;
        for (0..m) |i| if (sl.kind[i] == b) {
            own[cnt] = @intCast(i);
            cnt += 1;
        };
        if (cnt == 0) continue;
        std.sort.block(u32, own[0..cnt], sl.row, struct {
            fn lt(rows: []const u32, p: u32, o: u32) bool {
                return rows[p] < rows[o];
            }
        }.lt);
        const E: usize = @intCast(ops.shape(bk[0].w)[0]);
        const lo = @max(q * E, 16);
        const hi = if (blocked) (q + 1) * E - 1 else cnt;
        if (blocked and hi < lo) return error.StreamedBankTooFewRows;
        var off: usize = 0;
        while (off < cnt) {
            const take_n = @min(hi, cnt - off);
            const pad = if (blocked and take_n < lo) lo - take_n else 0;
            const calls = pad + take_n;
            const tok_rows = try a.alloc(u32, 2 * calls);
            defer a.free(tok_rows);
            @memset(tok_rows, 0);
            for (own[off..][0..take_n], pad..) |i, j| {
                tok_rows[j] = pairs[i] / @as(u32, @intCast(K));
                tok_rows[calls + j] = sl.row[i];
                inv[i] = at + @as(u32, @intCast(j));
            }
            const c: c_int = @intCast(calls);
            const lhs = try g.fromSlice(u32, tok_rows[0..calls], &.{c});
            const idx = try g.fromSlice(u32, tok_rows[calls..], &.{c});
            try parts.append(a, try sortedDown(w, g, x_flat, lhs, &bk[0], &bk[1], &bk[2], idx, blocked, blocked));
            at += @intCast(calls);
            off += take_n;
        }
    };
    return g.take(try g.concat(parts.items, 0), try g.fromSlice(u32, inv, &.{@intCast(m)}), 0);
}

var route_pack_logged = false;
var prefill_reduce_logged = false;
