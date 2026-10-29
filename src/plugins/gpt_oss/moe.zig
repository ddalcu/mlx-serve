//! gpt-oss routed experts: a biased router (softmax over the top-k), per-expert ADDITIVE biases on gate, up and down,
//! and the clamped SwiGLU `clip(gate, max=l) * sigmoid(alpha * gate) * (clip(up, ±l) + 1)`. The biases force the
//! sorted path at every width (no fused decode arm has a bias slot), as the legacy.
//! copied_from: sdk/bases/blocks/moe.zig (`route`'s softmax arm, `sorted`), sdk/blocks/moe/moe.zig
//! (`moeMLP2WithRouter`'s bias and gpt_oss activation arms), sdk/blocks/moe/experts.zig (`addExpertBias`).

const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const block = @import("../../sdk/bases/block.zig");
const linear = @import("../../sdk/bases/blocks/linear.zig");
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const Bank = @import("../../sdk/bases/blocks/moe.zig").Bank;
const routing = @import("../../sdk/blocks/moe/routing.zig");
const router_kernel = @import("../../sdk/blocks/moe/router_kernel.zig");
const verify_kernels = @import("../../sdk/blocks/moe/verify_kernels.zig");
const gather_qmm_nax = @import("../../sdk/blocks/moe/gather_qmm_nax.zig");
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const gptOssSwiGLU = @import("../../sdk/blocks/mlp/activations.zig").gptOssSwiGLU;
const verifyQmmNaxAvailable = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;

pub const Moe = struct {
    pub const kind: block.Kind = .mlp;
    pub const supports_batched_rows = false;
    pub const Weights = struct {
        /// The router with its additive bias (`mlp.router.bias`).
        router: linear.Linear,
        gate: Bank,
        up: Bank,
        down: Bank,
        /// `[E, out]` additive biases, read in sorted order.
        gate_bias: Array,
        up_bias: Array,
        down_bias: Array,
        top_k: c_int,
        experts: c_int,
        limit: f32,
        alpha: f32 = 1.702,
        /// The softmax routing chain compiled (`compileRouting`); null = op by op.
        route_fn: compiled.Closure = .{ .ctx = null },
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const parent = c.g;
        var t = parent.scope();
        defer t.deinit();
        const g = &t;
        const r = try route(w, g, try w.router.forward(g, x));
        // The routing weights land in the activation dtype.
        const scores = if (ops.dtype(r[1]) != ops.dtype(x)) try g.astype(r[1], ops.dtype(x)) else r[1];
        const xs = ops.shape(x);
        const B = xs[0];
        const S = xs[1];
        const D = xs[xs.len - 1];
        const K = ops.shape(r[0])[ops.shape(r[0]).len - 1];
        const s = try sorted(w, g, x, r[0], scores, B, S, D, K);
        if (s.reduced) return parent.keep(s.out);
        return parent.keep(try g.sum(try g.mul(s.out, try g.expandDims(scores, -1)), -2, false));
    }
};

const RoutePayload = struct { k: c_int, s: ops.Stream };

fn routeBody(p: *const RoutePayload, in: []const Array, out: *[2]Array) anyerror!void {
    const r = try routing.moeRoutingChain(in[0], p.k, p.s);
    out.* = .{ r.inds, r.norm_scores };
}

/// The softmax top-k chain compiled per input shape, as the legacy load compiles it for an MoE trunk.
pub fn compileRouting(w: *const Moe.Weights, s: ops.Stream) compiled.Closure {
    return compiled.compile(RoutePayload, .{ .k = w.top_k, .s = s }, 2, routeBody, false);
}

/// `[inds, scores]` owned by `g`: the fused router kernel, else the compiled chain, else the ops.
fn route(w: *const Moe.Weights, g: *Graph, logits: Array) ![2]Array {
    const r = (try router_kernel.moeRouterTopK(g.s, logits, .{ .ctx = null }, w.top_k, .softmax, true, 1.0, ops.dtype(logits), 0, 0, null)) orelse blk: {
        if (w.route_fn.ctx != null) if (try compiled.apply(g, w.route_fn, &.{logits}, 2)) |o| return o;
        break :blk try routing.moeRoutingChain(logits, w.top_k, g.s);
    };
    if (r.gate_logit.ctx != null) _ = mlx.mlx_array_free(r.gate_logit);
    const inds = g.adopt(r.inds) catch |e| {
        _ = mlx.mlx_array_free(r.norm_scores);
        return e;
    };
    return .{ inds, try g.adopt(r.norm_scores) };
}

fn expertMm(parent: *Graph, x: Array, bank: *const Bank, lhs: Array, rhs: Array) !Array {
    var t = parent.scope();
    defer t.deinit();
    var y = mlx.mlx_array_new();
    gatherExpertMm(&y, x, bank.w, bank.s, bank.b, lhs, rhs, bank.qp.bits, bank.qp.group_size, bank.qp.mode, true, t.s) catch |e| {
        _ = mlx.mlx_array_free(y);
        return e;
    };
    return parent.keep(try t.adopt(y));
}

/// `y + bias[sel]`: the per-expert bias of the bank each sorted row was gathered from.
fn addBias(parent: *Graph, y: Array, bias: Array, sel: Array) !Array {
    var t = parent.scope();
    defer t.deinit();
    return parent.keep(try t.add(y, try t.take(bias, sel, 0)));
}

/// mlx-lm's `_gather_sort` with the biases added in sorted order, before the inverse permutation.
fn sorted(w: *const Moe.Weights, g: *Graph, x: Array, inds: Array, scores: Array, B: c_int, S: c_int, D: c_int, K: c_int) !struct { out: Array, reduced: bool } {
    const total = B * S * K;
    const gq = w.gate.qp;
    const uq = w.up.qp;
    const flat = try g.reshape(inds, &.{total});
    var inv: Array = undefined;
    var sorted_inds: Array = undefined;
    var lhs: Array = undefined;
    if (try verify_kernels.moePrefillRoutePack(g.s, flat, K, w.experts)) |p| {
        inv = try g.adopt(p.inverse);
        sorted_inds = try g.adopt(p.sorted);
        lhs = try g.adopt(p.lhs);
    } else {
        const order = try g.argsort(flat, 0);
        inv = try g.argsort(order, 0);
        sorted_inds = try g.take(flat, order, 0);
        lhs = try g.call(mlx.mlx_floor_divide, .{ order, try g.adopt(mlx.mlx_array_new_int(K)) });
    }
    const x_flat = try g.reshape(x, &.{ B * S, D });

    var mapped_gate: ?Array = null;
    var mapped_up: ?Array = null;
    const nax = verifyQmmNaxAvailable();
    const experts = ops.shape(w.gate.w)[0];
    const row_block = total >= 16 and experts > 0 and @divTrunc(total, experts) >= 4;
    if (row_block and nax and gq.mode == uq.mode and (gq.mode == .affine or gq.mode.isFp4())) {
        const x_tok = try g.reshape(x_flat, &.{ B * S, 1, D });
        const row_map = try g.astype(lhs, .uint32);
        mapped_gate = if (try gather_qmm_nax.sortedGatherMapped(x_tok, row_map, w.gate.w, w.gate.s, w.gate.b, sorted_inds, gq.bits, gq.group_size, gq.mode.isFp4(), nax, g.s)) |y| try g.adopt(y) else null;
        mapped_up = if (try gather_qmm_nax.sortedGatherMapped(x_tok, row_map, w.up.w, w.up.s, w.up.b, sorted_inds, uq.bits, uq.group_size, uq.mode.isFp4(), nax, g.s)) |y| try g.adopt(y) else null;
        if (mapped_gate == null or mapped_up == null) {
            mapped_gate = null;
            mapped_up = null;
        }
    }
    const x_rep: ?Array = if (mapped_gate == null) try g.reshape(try g.take(x_flat, lhs, 0), &.{ total, 1, D }) else null;
    const none = Array{ .ctx = null };
    const gate3 = mapped_gate orelse try expertMm(g, x_rep.?, &w.gate, none, sorted_inds);
    const gate = try addBias(g, try g.call(mlx.mlx_squeeze, .{gate3}), w.gate_bias, sorted_inds);
    const up3 = mapped_up orelse try expertMm(g, x_rep.?, &w.up, none, sorted_inds);
    const up = try addBias(g, try g.call(mlx.mlx_squeeze, .{up3}), w.up_bias, sorted_inds);
    const act = try g.adopt(try gptOssSwiGLU(g.s, gate, up, w.limit, w.alpha));
    const down3 = try expertMm(g, try g.expandDims(act, -2), &w.down, none, sorted_inds);
    const down = try addBias(g, try g.call(mlx.mlx_squeeze, .{down3}), w.down_bias, sorted_inds);
    if (try verify_kernels.moePrefillExpertReduce(g.s, down, inv, scores)) |sum| return .{ .out = try g.adopt(sum), .reduced = true };
    const unsorted = try g.take(down, inv, 0);
    return .{ .out = try g.reshape(unsorted, &.{ B, S, K, ops.shape(unsorted)[1] }), .reduced = false };
}
