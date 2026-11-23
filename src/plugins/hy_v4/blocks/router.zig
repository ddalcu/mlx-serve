//! Hy4-flash's sigmoid router: f32 logits, sigmoid scores, the selection bias added ONLY to pick experts (the
//! returned weights are the unbiased scores), the group step (a no-op at `n_group == 1`), top-k sorted ascending
//! by expert id (the reference's `index_add_` accumulates in that order), normalized by `sum + 1e-20`, then
//! scaled by `routed_scaling_factor`.
//!
//! Reference (`HYV4FlashTopkRouter`): `scores = sigmoid(x @ gate^T)`, `scores_for_choice = scores + bias`,
//! `topk(scores_for_choice, k, sorted=False)` picks ids, `weights = scores.gather(ids)`.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");

const Array = ops.Array;
const Graph = ops.Graph;

pub const Params = struct {
    /// `[hidden, experts]` f32 (`gate.weight^T`, pre-transposed at load).
    gate_t: Array,
    /// `[experts]` f32 selection bias.
    bias: Array,
    experts: u32,
    hidden: u32,
    top_k: u32,
    /// `norm_topk_prob`: normalize the chosen weights by their sum.
    norm: bool,
    /// `routed_scaling_factor`.
    scale: f32,
    /// `n_group` / `topk_group`: 1 / 1 means every expert is eligible (the pack's shape).
    n_group: u32,
    topk_group: u32,
};

pub const Choice = struct {
    /// `[slots, rows, top_k]` i32, ascending expert id.
    ids: Array,
    /// `[slots, rows, top_k]` f32 routed weights (folded scale, normalized).
    weights: Array,
};

/// The router's choice for `x` `[slots, rows, hidden]`.
pub fn route(g: *Graph, x: Array, p: *const Params) !Choice {
    var t = g.scope();
    defer t.deinit();
    const sh = ops.shape(x);
    const n = sh[0];
    const s = sh[1];
    const k: c_int = @intCast(p.top_k);
    const experts: c_int = @intCast(p.experts);

    const logits = try t.matmul(try t.astype(x, .float32), p.gate_t);
    const scores = try t.sigmoid(logits);
    var choice = try t.add(scores, p.bias);
    if (p.n_group > 1 and p.topk_group < p.n_group) {
        // Best-two-of-group scores summed per group, the `topk_group` groups kept, everything else masked.
        const per: c_int = @intCast(p.experts / p.n_group);
        const grouped = try t.reshape(choice, &.{ n, s, @intCast(p.n_group), per });
        const sorted = try t.argsort(grouped, -1);
        const top2 = try t.slice(sorted, &.{ 0, 0, 0, per - 2 }, &.{ n, s, @intCast(p.n_group), per });
        const picked2 = try t.takeAlong(grouped, top2, -1);
        const gs = try t.sum(picked2, 3, false); // [n, s, n_group]
        const gs_sorted = try t.argsort(gs, -1);
        const ng: c_int = @intCast(p.n_group);
        const tg: c_int = @intCast(p.topk_group);
        const keep = try t.slice(gs_sorted, &.{ 0, 0, ng - tg }, &.{ n, s, ng });
        const mask = try scatterMask(&t, n, s, keep, experts, @intCast(p.n_group));
        choice = try t.where(mask, choice, try t.full(&.{ n, s, experts }, try t.scalar(-std.math.inf(f32), .float32), .float32));
    }

    // k largest by choice value, ties broken by ascending expert id (stable argsort of the negated value).
    const order = try t.argsort(try t.neg(choice), -1);
    const ids_desc = try t.slice(order, &.{ 0, 0, 0 }, &.{ n, s, k });
    // Ascending id: the reference's accumulation order.
    const ids = try t.argsort(ids_desc, -1);
    const ids_i32 = try t.astype(try t.takeAlong(ids_desc, ids, -1), .int32);
    const chosen = try t.takeAlong(scores, ids_i32, -1);
    var w = chosen;
    if (p.norm) {
        const denom = try t.addScalar(try t.sum(chosen, -1, true), 1e-20);
        w = try t.div(chosen, denom);
    }
    return .{
        .ids = try g.adopt(t.escape(ids_i32)),
        .weights = try g.adopt(t.escape(try t.mulScalar(w, p.scale))),
    };
}

/// A boolean `[n, s, experts]` mask naming the kept groups: `keep` is the `[n, s, topk_group]` group ids.
fn scatterMask(t: *Graph, n: c_int, s: c_int, keep: Array, experts: c_int, n_group: c_int) !Array {
    const tg: c_int = @intCast(ops.shape(keep)[2]);
    const per: c_int = @divExact(experts, n_group);
    const group_ids = try t.arange(0, @floatFromInt(n_group), 1, .int32); // [n_group]
    const cmp = try t.equal(
        try t.broadcast(try t.reshape(group_ids, &.{ 1, 1, n_group, 1 }), &.{ n, s, n_group, tg }),
        try t.broadcast(try t.expandDims(keep, -1), &.{ n, s, n_group, tg }),
    );
    const hits = try t.sum(try t.astype(cmp, .int32), -1, false); // [n, s, n_group]
    const group_keep = try t.greater(hits, try t.scalar(0, .int32)); // [n, s, n_group]
    const wide = try t.broadcast(try t.expandDims(group_keep, -1), &.{ n, s, n_group, per });
    return t.reshape(wide, &.{ n, s, experts });
}

const testing = std.testing;

fn readF32(a: Array) ![]const f32 {
    try ops.eval(a);
    const n = mlx.mlx_array_size(a);
    const data = mlx.mlx_array_data_float32(a) orelse return error.NoData;
    return data[0..n];
}

fn readI32(a: Array) ![]const i32 {
    try ops.eval(a);
    const n = mlx.mlx_array_size(a);
    const data = mlx.mlx_array_data_int32(a) orelse return error.NoData;
    return data[0..n];
}

test "hy_v4 router: sigmoid scores, unbiased weights, ascending ids, normalization and scale" {
    const experts: usize = 6;
    const hidden: usize = 3;
    const rows: usize = 2;
    const top_k: usize = 2;

    // gate rows: expert e's logit = dot(x, gate[e]).
    const gate: [experts * hidden]f32 = .{ 0.5, -0.2, 0.1, -0.3, 0.4, 0.2, 0.9, 0.1, -0.5, 0.2, 0.7, -0.4, -0.1, -0.6, 0.3, 0.3, -0.3, 0.8 };
    const bias: [experts]f32 = .{ 0, 0, 0.9, 0, 0, 0 }; // expert 2 gets picked despite a low score
    const x: [rows * hidden]f32 = .{ 1, 2, -1, 0.5, -1, 1 };

    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = Graph.init(testing.allocator, stream);
    defer g.deinit();

    // gate_t [hidden, experts]
    var gate_t: [hidden * experts]f32 = undefined;
    for (0..experts) |e| {
        for (0..hidden) |j| gate_t[j * experts + e] = gate[e * hidden + j];
    }
    const gate_arr = try g.fromSlice(f32, &gate_t, &.{ @intCast(hidden), @intCast(experts) });
    defer g.drop(gate_arr);
    const bias_arr = try g.fromSlice(f32, &bias, &.{@intCast(experts)});
    defer g.drop(bias_arr);
    const x_arr = try g.fromSlice(f32, &x, &.{ 1, @intCast(rows), @intCast(hidden) });
    defer g.drop(x_arr);

    const p: Params = .{ .gate_t = gate_arr, .bias = bias_arr, .experts = experts, .hidden = hidden, .top_k = top_k, .norm = true, .scale = 3.998, .n_group = 1, .topk_group = 1 };
    const r = try route(&g, x_arr, &p);
    defer g.drop(r.ids);
    defer g.drop(r.weights);
    const got_ids = try readI32(r.ids);
    const got_w = try readF32(r.weights);

    for (0..rows) |row| {
        var scores: [experts]f64 = undefined;
        for (0..experts) |e| {
            var logit: f64 = 0;
            for (0..hidden) |j| logit += @as(f64, x[row * hidden + j]) * gate[e * hidden + j];
            scores[e] = 1.0 / (1.0 + @exp(-logit));
        }
        var choice: [experts]f64 = undefined;
        for (0..experts) |e| choice[e] = scores[e] + bias[e];
        // Top-2 by choice, ties by ascending id.
        var best: [2]usize = .{ 0, 0 };
        for (0..experts) |e| {
            if (e == best[0] or e == best[1]) continue;
            if (choice[e] > choice[best[0]] or (choice[e] == choice[best[0]] and e < best[0])) {
                best[1] = best[0];
                best[0] = e;
            } else if (choice[e] > choice[best[1]] or (choice[e] == choice[best[1]] and e < best[1])) {
                best[1] = e;
            }
        }
        if (best[0] > best[1]) std.mem.swap(usize, &best[0], &best[1]);
        try testing.expectEqual(@as(i32, @intCast(best[0])), got_ids[row * top_k]);
        try testing.expectEqual(@as(i32, @intCast(best[1])), got_ids[row * top_k + 1]);

        const sum = scores[best[0]] + scores[best[1]] + 1e-20;
        const want0 = scores[best[0]] / sum * 3.998;
        const want1 = scores[best[1]] / sum * 3.998;
        try testing.expectApproxEqAbs(@as(f32, @floatCast(want0)), got_w[row * top_k], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, @floatCast(want1)), got_w[row * top_k + 1], 1e-5);
    }
}
