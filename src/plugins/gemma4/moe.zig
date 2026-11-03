//! Gemma 4's routed experts (26B-A4B): the sigma-MoE router (the raw residual rms-normed by the
//! folded `router.scale`, a softmax top-k renormed, times `router.per_expert_scale` of the picks) over GeGLU experts.
//! The shared dense MLP and the norms around it are the layer's (`decoder.zig`). Arms as the legacy takes them for a
//! tanh-GELU pack: one decode row through the stock gather, wider calls through mlx-lm's `_gather_sort` (the NAX
//! mapped gathers where they serve); the SwiGLU-only kernels never engage.
//! copied_from: sdk/bases/blocks/moe.zig (`route`, `sorted`, `expertMm`), sdk/blocks/moe/moe.zig
//! (`moeRouterLogits`' router-scale arm, `moeMLP2WithRouter`'s sigma-MoE scale and gather arms).

const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const QuantParams = @import("../../sdk/quant/params.zig").QuantParams;
const routing = @import("../../sdk/blocks/moe/routing.zig");
const router_kernel = @import("../../sdk/blocks/moe/router_kernel.zig");
const gather_qmm_nax = @import("../../sdk/blocks/moe/gather_qmm_nax.zig");
const verify_kernels = @import("../../sdk/blocks/moe/verify_kernels.zig");
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const verifyQmmNaxAvailable = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const fused_act = @import("../../sdk/blocks/mlp/fused_activations.zig");
const compiled_mod = @import("../gemma3/compiled.zig");
const Compiled = compiled_mod.Compiled;

const Array = ops.Array;
const Graph = ops.Graph;
const Linear = bases.blocks.linear.Linear;

/// An expert bank `[E, out, in]` and its quant geometry.
pub const Bank = struct { w: Array, s: Array = .{}, b: Array = .{}, qp: QuantParams };

pub const Weights = struct {
    router: Linear,
    /// `router.scale * hidden^-0.5` (folded at load, bf16).
    router_scale: Array,
    per_expert_scale: Array = .{},
    gate: Bank,
    up: Bank,
    down: Bank,
    top_k: c_int,
    experts: c_int,
    /// The softmax top-k chain compiled (`compileRouting`); null = op by op.
    route_fn: compiled.Closure = .{ .ctx = null },
    compiled: *const Compiled,
};

const RoutePayload = struct { k: c_int, s: ops.Stream };

fn routeBody(p: *const RoutePayload, in: []const Array, out: *[2]Array) anyerror!void {
    const r = try routing.moeRoutingChain(in[0], p.k, p.s);
    out.* = .{ r.inds, r.norm_scores };
}

/// The routing chain compiled, as the legacy load compiles it for every MoE model.
pub fn compileRouting(top_k: c_int, s: ops.Stream) compiled.Closure {
    return compiled.compile(RoutePayload, .{ .k = top_k, .s = s }, 2, routeBody, false);
}

const Routed = struct { inds: Array, scores: Array };

fn adoptRouting(g: *Graph, r: anytype) !Routed {
    if (r.gate_logit.ctx != null) _ = mlx.mlx_array_free(r.gate_logit);
    const inds = g.adopt(r.inds) catch |e| {
        _ = mlx.mlx_array_free(r.norm_scores);
        return e;
    };
    return .{ .inds = inds, .scores = try g.adopt(r.norm_scores) };
}

fn route(w: *const Weights, g: *Graph, logits: Array) !Routed {
    if (try router_kernel.moeRouterTopK(g.s, logits, .{ .ctx = null }, w.top_k, .softmax, true, 1.0, ops.dtype(logits), 0, 0, null)) |r| return adoptRouting(g, r);
    if (w.route_fn.ctx != null) if (try compiled.apply(g, w.route_fn, &.{logits}, 2)) |o| return .{ .inds = o[0], .scores = o[1] };
    return adoptRouting(g, try routing.moeRoutingChain(logits, w.top_k, g.s));
}

/// `gelu_tanh(gate) * up` as the engine's `computeGeglu` takes it: the table kernel, the compiled GeGLU, the ops.
fn product(w: *const Weights, parent: *Graph, gate: Array, up: Array) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const dt = ops.dtype(gate);
    if (dt == .bfloat16 or dt == .float16) {
        const table = try w.compiled.geluTable(dt);
        if (table.ctx != null) if (try fused_act.tableGateMul(g.s, .gelu_tanh, gate, up, table, null)) |y| return parent.keep(try g.adopt(y));
    }
    if (w.compiled.geglu.ctx != null) return parent.keep(try g.adopt(try compiled_mod.apply(w.compiled.geglu, &.{ gate, up })));
    return parent.keep(try g.mul(try g.adopt(try w.compiled.geluOf(gate)), up));
}

fn expertMm(g: *Graph, x: Array, bank: *const Bank, lhs: Array, rhs: Array, is_sorted: bool) !Array {
    var y = mlx.mlx_array_new();
    gatherExpertMm(&y, x, bank.w, bank.s, bank.b, lhs, rhs, bank.qp.bits, bank.qp.group_size, bank.qp.mode, is_sorted, g.s) catch |e| {
        _ = mlx.mlx_array_free(y);
        return e;
    };
    return g.adopt(y);
}

/// The weighted expert sum `[B, S, hidden]` for `x` (the experts' input), routed off `router_x` (the raw residual).
pub fn forward(w: *const Weights, parent: *Graph, eps: f32, router_x: Array, x: Array) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const normed = try g.rmsNorm(router_x, w.router_scale, eps);
    const logits = try w.router.forward(g, normed);
    var r = try route(w, g, logits);
    if (ops.dtype(r.scores) != ops.dtype(x)) r.scores = try g.astype(r.scores, ops.dtype(x));
    if (w.per_expert_scale.ctx != null) r.scores = try g.mul(r.scores, try g.call(mlx.mlx_take, .{ w.per_expert_scale, r.inds }));

    const xs = ops.shape(x);
    const B = xs[0];
    const S = xs[1];
    const D = xs[xs.len - 1];
    const K = ops.shape(r.inds)[ops.shape(r.inds).len - 1];
    if (B * S > 1 or B * S * K >= 64) {
        const s = try sorted(w, g, x, r, B, S, D, K);
        if (s.reduced) return parent.keep(s.out);
        return parent.keep(try g.sum(try g.mul(s.out, try g.expandDims(r.scores, -1)), -2, false));
    }
    const x5 = try g.reshape(x, &.{ B, S, 1, 1, D });
    const none = Array{ .ctx = null };
    const gate = try g.call(mlx.mlx_squeeze, .{try expertMm(g, x5, &w.gate, none, r.inds, false)});
    const up = try g.call(mlx.mlx_squeeze, .{try expertMm(g, x5, &w.up, none, r.inds, false)});
    const act = try g.expandDims(try product(w, g, gate, up), -2);
    const out = try g.call(mlx.mlx_squeeze, .{try expertMm(g, act, &w.down, none, r.inds, false)});
    return parent.keep(try g.sum(try g.mul(out, try g.expandDims(r.scores, -1)), -2, false));
}

/// mlx-lm's `_gather_sort`: every (token, expert) assignment sorted by expert so consecutive rows read one bank,
/// the down output weighted and summed back in token order.
fn sorted(w: *const Weights, g: *Graph, x: Array, r: Routed, B: c_int, S: c_int, D: c_int, K: c_int) !struct { out: Array, reduced: bool } {
    const total = B * S * K;
    const gq = w.gate.qp;
    const uq = w.up.qp;
    const flat = try g.reshape(r.inds, &.{total});
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

    // NAX: each projection gathers its rows through the row map (the SwiGLU gate+up kernel never serves GELU).
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
    const gate = try g.call(mlx.mlx_squeeze, .{mapped_gate orelse try expertMm(g, x_rep.?, &w.gate, none, sorted_inds, true)});
    const up = try g.call(mlx.mlx_squeeze, .{mapped_up orelse try expertMm(g, x_rep.?, &w.up, none, sorted_inds, true)});
    const act = try product(w, g, gate, up);
    const down = try g.call(mlx.mlx_squeeze, .{try expertMm(g, try g.expandDims(act, -2), &w.down, none, sorted_inds, true)});
    if (try verify_kernels.moePrefillExpertReduce(g.s, down, inv, r.scores)) |sum| return .{ .out = try g.adopt(sum), .reduced = true };
    const unsorted = try g.take(down, inv, 0);
    return .{ .out = try g.reshape(unsorted, &.{ B, S, K, ops.shape(unsorted)[1] }), .reduced = false };
}
