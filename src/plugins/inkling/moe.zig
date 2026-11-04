//! Inkling's routed + shared-expert MoE block and its router chain (`InklingTopkRouter`).

const mlx = @import("../../mlx/ffi.zig");
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;

const InkgroupLimitedRouting = struct { topk_weights: mlx.mlx_array, topk_idx: mlx.mlx_array, shared_gammas: mlx.mlx_array };

/// The Inkling router chain (`InklingTopkRouter`), all in f32: sigmoid scores
/// + selection bias pick the top-k routed experts, then ONE softmax over the
/// logsigmoid of [selected routed logits, shared logits] produces the routed
/// weights AND the shared-expert gammas (the "sink"), scaled by
/// route_scale × global_scale. `logits` = x_f32 @ gate_weight.T [T, E+ns].
pub fn inklingRouterChain(s: mlx.mlx_stream, logits: mlx.mlx_array, bias: mlx.mlx_array, top_k: c_int, n_shared: c_int, route_scale: f32, global_scale: f32) !InkgroupLimitedRouting {
    const l_shape = mlx.getShape(logits); // [T, E+ns]
    const t_count = l_shape[0];
    const n_routed = l_shape[1] - n_shared;
    const strides2 = [_]c_int{ 1, 1 };

    var routed_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(routed_logits);
    try mlx.check(mlx.mlx_slice(&routed_logits, logits, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ t_count, n_routed }, 2, &strides2, 2, s));
    var shared_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shared_logits);
    try mlx.check(mlx.mlx_slice(&shared_logits, logits, &[_]c_int{ 0, n_routed }, 2, &[_]c_int{ t_count, l_shape[1] }, 2, &strides2, 2, s));

    // top-k on sigmoid(routed) + bias (selection only; weights use the logits)
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_sigmoid(&scores, routed_logits, s));
    var choice = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(choice);
    try mlx.check(mlx.mlx_add(&choice, scores, bias, s));
    var neg_choice = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_choice);
    try mlx.check(mlx.mlx_negative(&neg_choice, choice, s));
    var part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_argpartition_axis(&part, neg_choice, top_k - 1, -1, s));
    var topk_idx = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(topk_idx);
    {
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, part, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ t_count, top_k }, 2, &strides2, 2, s));
        try mlx.check(mlx.mlx_astype(&topk_idx, sliced, .uint32, s));
    }

    var gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered);
    try mlx.check(mlx.mlx_take_along_axis(&gathered, routed_logits, topk_idx, 1, s));
    var topk_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_logits);
    {
        const arr = [_]mlx.mlx_array{ gathered, shared_logits };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&topk_logits, vec, 1, s));
    }

    // logsigmoid(x) = -logaddexp(0, -x), then softmax over [K + ns]
    var neg_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_logits);
    try mlx.check(mlx.mlx_negative(&neg_logits, topk_logits, s));
    const zero_f = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero_f);
    var lae = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lae);
    try mlx.check(mlx.mlx_logaddexp(&lae, zero_f, neg_logits, s));
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_negative(&log_probs, lae, s));
    var soft = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(soft);
    try mlx.check(mlx.mlx_softmax_axis(&soft, log_probs, -1, true, s));
    const scale_f = mlx.mlx_array_new_float(route_scale * global_scale);
    defer _ = mlx.mlx_array_free(scale_f);
    var weights = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weights);
    try mlx.check(mlx.mlx_multiply(&weights, soft, scale_f, s));

    var topk_weights = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(topk_weights);
    try mlx.check(mlx.mlx_slice(&topk_weights, weights, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ t_count, top_k }, 2, &strides2, 2, s));
    var shared_gammas = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(shared_gammas);
    try mlx.check(mlx.mlx_slice(&shared_gammas, weights, &[_]c_int{ 0, top_k }, 2, &[_]c_int{ t_count, top_k + n_shared }, 2, &strides2, 2, s));

    return .{ .topk_weights = topk_weights, .topk_idx = topk_idx, .shared_gammas = shared_gammas };
}

// ── the plugin's block ──

const ops = @import("../../mlx/ops.zig");
const block = @import("../../sdk/bases/block.zig");
const Bank = @import("../../sdk/bases/blocks/moe.zig").Bank;
const swiglu = @import("../../sdk/bases/blocks/mlp.zig").swiglu;

const Graph = ops.Graph;
const Array = ops.Array;

/// `InklingMoE`: the f32 router over the routed AND shared experts' rows, routed top-k plus the router-gated shared
/// experts (the sink) in one softmax (`inklingRouterChain`). Expert math in the stream's dtype; the shared sum in f32.
pub const Moe = struct {
    pub const kind: block.Kind = .mlp;
    pub const supports_batched_rows = false;
    pub const Weights = struct {
        /// `[hidden, experts + n_shared]` f32.
        router: Array,
        /// The selection bias `[experts]` f32 (zeros where the pack has none).
        bias: Array,
        /// gate, up, down.
        routed: [3]Bank,
        shared: [3]Bank,
        top_k: c_int,
        n_shared: c_int,
        /// `route_scale × gate.global_scale`.
        scale: f32,
    };

    pub fn forward(w: *const Weights, c: *block.Ctx, x: Array) !Array {
        const g = c.g;
        const sh = ops.shape(x);
        const t = sh[0] * sh[1];
        const x_flat = try g.reshape(x, &.{ t, sh[2] });
        const logits = try g.matmul(try g.astype(x_flat, .float32), w.router);
        const r = try inklingRouterChain(g.s, logits, w.bias, w.top_k, w.n_shared, w.scale, 1.0);
        const idx = g.adopt(r.topk_idx) catch |e| {
            _ = mlx.mlx_array_free(r.topk_weights);
            _ = mlx.mlx_array_free(r.shared_gammas);
            return e;
        };
        const tw = g.adopt(r.topk_weights) catch |e| {
            _ = mlx.mlx_array_free(r.shared_gammas);
            return e;
        };
        const gammas = try g.adopt(r.shared_gammas);

        // The reference casts the routing weights to x's dtype before the multiply; the sum accumulates there.
        const dt = ops.dtype(x_flat);
        const routed = try experts(g, x_flat, idx, &w.routed);
        const routed_sum = try g.call(mlx.mlx_sum_axis, .{ try g.mul(routed, try g.call(mlx.mlx_expand_dims, .{ try g.astype(tw, dt), @as(c_int, -1) })), @as(c_int, 1), false });

        // Gammas rounded to x's dtype, then the shared sum in f32 cast to the routed sum's dtype.
        const row = try g.reshape(try g.arange(0, @floatFromInt(w.n_shared), 1, .uint32), &.{ 1, w.n_shared });
        const sh_idx = try g.call(mlx.mlx_broadcast_to, .{ row, &[_]c_int{ t, w.n_shared }, @as(usize, 2) });
        const shared = try g.astype(try experts(g, x_flat, sh_idx, &w.shared), .float32);
        const sg = try g.call(mlx.mlx_expand_dims, .{ try g.astype(try g.astype(gammas, dt), .float32), @as(c_int, -1) });
        const shared_sum = try g.astype(try g.call(mlx.mlx_sum_axis, .{ try g.mul(shared, sg), @as(c_int, 1), false }), ops.dtype(routed_sum));
        return g.reshape(try g.add(routed_sum, shared_sum), sh);
    }

    /// `x_flat` `[T, D]` through the experts `inds` `[T, K]` names, sorted so each expert streams its rows once:
    /// `[T, K, hidden]`.
    fn experts(g: *Graph, x_flat: Array, inds: Array, banks: *const [3]Bank) !Array {
        const d = ops.shape(x_flat)[1];
        const t = ops.shape(inds)[0];
        const k = ops.shape(inds)[1];
        const flat = try g.reshape(inds, &.{t * k});
        const order = try g.call(mlx.mlx_argsort_axis, .{ flat, @as(c_int, 0) });
        const inv = try g.call(mlx.mlx_argsort_axis, .{ order, @as(c_int, 0) });
        const sorted = try g.call(mlx.mlx_take_axis, .{ flat, order, @as(c_int, 0) });
        const lhs = try g.call(mlx.mlx_floor_divide, .{ order, try g.adopt(mlx.mlx_array_new_int(k)) });
        const x_rep = try g.reshape(try g.call(mlx.mlx_take_axis, .{ x_flat, lhs, @as(c_int, 0) }), &.{ t * k, 1, d });
        const gate = try rows(g, try gather(g, x_rep, &banks[0], sorted));
        const up = try rows(g, try gather(g, x_rep, &banks[1], sorted));
        const act = try g.call(mlx.mlx_expand_dims, .{ try swiglu(g, gate, up), @as(c_int, -2) });
        const down = try rows(g, try gather(g, act, &banks[2], sorted));
        const back = try g.call(mlx.mlx_take_axis, .{ down, inv, @as(c_int, 0) });
        return g.reshape(back, &.{ t, k, ops.shape(back)[1] });
    }

    /// `[n, 1, w]` as `[n, w]` (the legacy's squeeze, which also dropped `n` at one row).
    fn rows(g: *Graph, y: Array) !Array {
        const sh = ops.shape(y);
        return g.reshape(y, &.{ sh[0], sh[2] });
    }

    fn gather(g: *Graph, x: Array, bank: *const Bank, rhs: Array) !Array {
        var y = mlx.mlx_array_new();
        gatherExpertMm(&y, x, bank.w, bank.s, bank.b, .{ .ctx = null }, rhs, bank.qp.bits, bank.qp.group_size, bank.qp.mode, true, g.s) catch |e| {
            _ = mlx.mlx_array_free(y);
            return e;
        };
        return g.adopt(y);
    }
};
