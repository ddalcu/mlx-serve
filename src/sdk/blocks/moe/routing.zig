//! sdk/blocks/moe/routing.
//! Owns MoeRouting, and 6 more.

const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const testing = @import("std").testing;
const RouterGate = @import("router_kernel.zig").RouterGate;
const moeRouterTopK = @import("router_kernel.zig").moeRouterTopK;

/// Result type for the MoE routing helpers. Both fields are owned arrays —
/// caller is responsible for freeing them.
/// `gate_logit` rides along when the router kernel also computed the shared-expert gate's logit.
pub const MoeRouting = struct { inds: mlx.mlx_array, norm_scores: mlx.mlx_array, gate_logit: mlx.mlx_array = .{ .ctx = null } };

pub fn moeRoutingChain(router_logits: mlx.mlx_array, k: c_int, s: mlx.mlx_stream) !MoeRouting {
    var neg_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_logits);
    try mlx.check(mlx.mlx_negative(&neg_logits, router_logits, s));

    var partitioned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(partitioned);
    try mlx.check(mlx.mlx_argpartition_axis(&partitioned, neg_logits, k - 1, -1, s));

    const p_shape = mlx.getShape(partitioned);
    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    {
        var start_arr: [4]c_int = undefined;
        var stop_arr: [4]c_int = undefined;
        var strides_arr: [4]c_int = undefined;
        for (0..p_shape.len) |d| {
            start_arr[d] = 0;
            stop_arr[d] = if (d == p_shape.len - 1) k else p_shape[d];
            strides_arr[d] = 1;
        }
        try mlx.check(mlx.mlx_slice(&inds, partitioned, &start_arr, p_shape.len, &stop_arr, p_shape.len, &strides_arr, p_shape.len, s));
    }

    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, router_logits, -1, true, s));

    var top_weights = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_weights);
    try mlx.check(mlx.mlx_take_along_axis(&top_weights, probs, inds, -1, s));

    var weight_sum_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weight_sum_raw);
    try mlx.check(mlx.mlx_sum_axis(&weight_sum_raw, top_weights, -1, false, s));
    var weight_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weight_sum);
    try mlx.check(mlx.mlx_expand_dims(&weight_sum, weight_sum_raw, -1, s));
    var norm_scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(norm_scores);
    try mlx.check(mlx.mlx_divide(&norm_scores, top_weights, weight_sum, s));

    return .{ .inds = inds, .norm_scores = norm_scores };
}

/// Hy3 (hy_v3 / DeepSeek-V3-style) sigmoid routing chain — mirrors the
/// reference `expert_select` (hy_v3.py):
///   scores = sigmoid(logits) in FLOAT32 (the fp32-router class: a bf16
///   sigmoid+bias flips near-tie expert picks on real checkpoints);
///   top-k SELECTED on scores + expert_bias, WEIGHTED by the unbiased scores;
///   if route_norm and k > 1: weights /= (sum + 1e-20);
///   weights *= route_scale; cast bf16 for the expert-combine multiply
///   (shipped checkpoints run enable_moe_fp32_combine = false).
/// Returns owned `inds` + `norm_scores` — caller must free both.
/// Group-limited routing mask (DeepSeek-style `group_limited_topk`): split the BIASED scores
/// into `n_group` equal groups, score each group by the SUM OF ITS TOP TWO
/// experts, keep the best `topk_group` groups, and drop every expert outside
/// them before the final top-k.
///
/// Returns the biased scores with the losing groups pushed far below any
/// reachable value. The reference writes -inf; we subtract a large FINITE
/// constant instead, which selects identically (biased scores are
/// sigmoid+bias, so bounded well inside +/-2) and keeps a non-finite value
/// out of a tensor that later feeds argpartition. Non-finite sentinels are
/// how the composed causal-mask arm shipped an all-NaN path that scored a
/// perfect parity max_err of 0.
fn groupLimitMask(
    biased: mlx.mlx_array,
    n_group: c_int,
    topk_group: c_int,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const bsh = mlx.getShape(biased);
    const num_experts = bsh[bsh.len - 1];
    if (n_group <= 0 or topk_group <= 0 or topk_group > n_group) return error.BadShape;
    if (@rem(num_experts, n_group) != 0) return error.BadShape;
    const group_size = @divExact(num_experts, n_group);
    if (group_size < 2) return error.BadShape; // the group score is a top-2 sum

    var rows: c_int = 1;
    for (bsh[0 .. bsh.len - 1]) |d| rows *= d;

    // [rows, n_group, group_size]
    const grouped_shape = [_]c_int{ rows, n_group, group_size };
    var grouped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grouped);
    try mlx.check(mlx.mlx_reshape(&grouped, biased, &grouped_shape, 3, s));

    // Group score = sum of the top 2 within the group. mlx_topk_axis, NOT
    // mlx_topk: the latter is the FLATTENED reduction and silently collapses
    // every group into one, which is an uncatchable shape error downstream.
    var top2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top2);
    try mlx.check(mlx.mlx_topk_axis(&top2, grouped, 2, -1, s));
    var gscore = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gscore);
    try mlx.check(mlx.mlx_sum_axis(&gscore, top2, -1, false, s)); // [rows, n_group]

    // Indices of the best `topk_group` groups.
    var gneg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gneg);
    try mlx.check(mlx.mlx_negative(&gneg, gscore, s));
    var gpart = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gpart);
    try mlx.check(mlx.mlx_argpartition_axis(&gpart, gneg, topk_group - 1, -1, s));
    var gidx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gidx);
    {
        const start = [_]c_int{ 0, 0 };
        const stop = [_]c_int{ rows, topk_group };
        const stride = [_]c_int{ 1, 1 };
        try mlx.check(mlx.mlx_slice(&gidx, gpart, &start, 2, &stop, 2, &stride, 2, s));
    }

    // keep[rows, n_group] = 1 for a surviving group.
    const keep_shape = [_]c_int{ rows, n_group };
    var keep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep);
    try mlx.check(mlx.mlx_zeros(&keep, &keep_shape, 2, .float32, s));
    const ones_shape = [_]c_int{ rows, topk_group };
    var ones = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones);
    try mlx.check(mlx.mlx_ones(&ones, &ones_shape, 2, .float32, s));
    var keep_set = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep_set);
    try mlx.check(mlx.mlx_put_along_axis(&keep_set, keep, gidx, ones, -1, s));

    // Broadcast a group's decision across its experts: [rows, G, 1] -> [rows, G, S] -> [rows, E].
    const keep3_shape = [_]c_int{ rows, n_group, 1 };
    var keep3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep3);
    try mlx.check(mlx.mlx_reshape(&keep3, keep_set, &keep3_shape, 3, s));
    var keep_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep_b);
    try mlx.check(mlx.mlx_broadcast_to(&keep_b, keep3, &grouped_shape, 3, s));
    const flat_shape = [_]c_int{ rows, num_experts };
    var keep_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep_flat);
    try mlx.check(mlx.mlx_reshape(&keep_flat, keep_b, &flat_shape, 2, s));

    var biased_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(biased_flat);
    try mlx.check(mlx.mlx_reshape(&biased_flat, biased, &flat_shape, 2, s));

    const penalty = mlx.mlx_array_new_float(-1.0e4);
    defer _ = mlx.mlx_array_free(penalty);
    var pen_full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pen_full);
    try mlx.check(mlx.mlx_full(&pen_full, &flat_shape, 2, penalty, .float32, s));

    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var cond = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cond);
    try mlx.check(mlx.mlx_greater(&cond, keep_flat, zero, s));

    var masked_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked_flat);
    try mlx.check(mlx.mlx_where(&masked_flat, cond, biased_flat, pen_full, s));

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, masked_flat, bsh.ptr, bsh.len, s));
    return out;
}

/// Group-limited routing = hy3's sigmoid+bias chain with the group limit inserted
/// between the bias and the top-k. Weights still come from the UNBIASED
/// sigmoid scores, sum-normalized with the reference's 1e-20 and scaled by
/// `routed_scaling_factor`.
pub fn groupLimitedRouting(
    router_logits: mlx.mlx_array,
    expert_bias: mlx.mlx_array,
    k: c_int,
    n_group: c_int,
    topk_group: c_int,
    route_norm: bool,
    route_scale: f32,
    s: mlx.mlx_stream,
) !MoeRouting {
    // One dispatch instead of ~28. Weights stay f32 for the reason the comment
    // at the tail of this function gives, so the fused kernel is asked for f32
    // too. A single group (Nemotron-H) is the ungrouped sigmoid+bias kernel.
    // A decline (geometry outside the kernel's set, kill switch) falls
    // through to the chain below, which stays the definition of correct.
    if (try moeRouterTopK(
        s,
        router_logits,
        expert_bias,
        k,
        if (n_group <= 1) .sigmoid_bias else .sigmoid_bias_grouped,
        route_norm,
        route_scale,
        .float32,
        n_group,
        topk_group,
        null,
    )) |fused| return fused;

    // Router runs in fp32 (`router_dtype: "fp32"`).
    var logits_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits_f32);
    try mlx.check(mlx.mlx_astype(&logits_f32, router_logits, .float32, s));
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_sigmoid(&scores, logits_f32, s));

    var biased = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(biased);
    try mlx.check(mlx.mlx_add(&biased, scores, expert_bias, s));

    const limited = try groupLimitMask(biased, n_group, topk_group, s);
    defer _ = mlx.mlx_array_free(limited);

    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, limited, s));
    var partitioned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(partitioned);
    try mlx.check(mlx.mlx_argpartition_axis(&partitioned, neg, k - 1, -1, s));

    const p_shape = mlx.getShape(partitioned);
    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    {
        var start_arr: [4]c_int = undefined;
        var stop_arr: [4]c_int = undefined;
        var strides_arr: [4]c_int = undefined;
        for (0..p_shape.len) |d| {
            start_arr[d] = 0;
            stop_arr[d] = if (d == p_shape.len - 1) k else p_shape[d];
            strides_arr[d] = 1;
        }
        try mlx.check(mlx.mlx_slice(&inds, partitioned, &start_arr, p_shape.len, &stop_arr, p_shape.len, &strides_arr, p_shape.len, s));
    }

    var top = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top);
    try mlx.check(mlx.mlx_take_along_axis(&top, scores, inds, -1, s));

    var weights_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weights_f32);
    if (route_norm and k > 1) {
        var sum_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum_raw);
        try mlx.check(mlx.mlx_sum_axis(&sum_raw, top, -1, true, s));
        const eps = mlx.mlx_array_new_float(1e-20);
        defer _ = mlx.mlx_array_free(eps);
        var sum_eps = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum_eps);
        try mlx.check(mlx.mlx_add(&sum_eps, sum_raw, eps, s));
        try mlx.check(mlx.mlx_divide(&weights_f32, top, sum_eps, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&weights_f32, top));
    }

    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    if (route_scale != 1.0) {
        const scale_arr = mlx.mlx_array_new_float(route_scale);
        defer _ = mlx.mlx_array_free(scale_arr);
        try mlx.check(mlx.mlx_multiply(&scaled, weights_f32, scale_arr, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&scaled, weights_f32));
    }

    // Weights stay f32. hy3's chain casts to bf16 here because that arch's
    // stream is bf16 anyway, but a group-limited arch multiplies these into the
    // routed expert sum AFTER `routed_scaling_factor` (2.5), so a bf16 round trip lands ~0.4%
    // of 2.5 on every token -- measured 2.6e-3 max error on the reference MoE
    // block, which is 500x the tolerance the rest of the port holds. The caller
    // casts to the activation dtype.
    var norm_scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(norm_scores);
    try mlx.check(mlx.mlx_array_set(&norm_scores, scaled));

    return .{ .inds = inds, .norm_scores = norm_scores };
}

pub fn hy3RoutingChain(router_logits: mlx.mlx_array, expert_bias: mlx.mlx_array, k: c_int, route_norm: bool, route_scale: f32, s: mlx.mlx_stream) !MoeRouting {
    var logits_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits_f32);
    try mlx.check(mlx.mlx_astype(&logits_f32, router_logits, .float32, s));
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_sigmoid(&scores, logits_f32, s));

    var biased = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(biased);
    try mlx.check(mlx.mlx_add(&biased, scores, expert_bias, s));

    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, biased, s));
    var partitioned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(partitioned);
    try mlx.check(mlx.mlx_argpartition_axis(&partitioned, neg, k - 1, -1, s));

    const p_shape = mlx.getShape(partitioned);
    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    {
        var start_arr: [4]c_int = undefined;
        var stop_arr: [4]c_int = undefined;
        var strides_arr: [4]c_int = undefined;
        for (0..p_shape.len) |d| {
            start_arr[d] = 0;
            stop_arr[d] = if (d == p_shape.len - 1) k else p_shape[d];
            strides_arr[d] = 1;
        }
        try mlx.check(mlx.mlx_slice(&inds, partitioned, &start_arr, p_shape.len, &stop_arr, p_shape.len, &strides_arr, p_shape.len, s));
    }

    // Weights = ORIGINAL (unbiased) sigmoid scores at the selected indices.
    var top = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top);
    try mlx.check(mlx.mlx_take_along_axis(&top, scores, inds, -1, s));

    var weights_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weights_f32);
    if (route_norm and k > 1) {
        var sum_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum_raw);
        try mlx.check(mlx.mlx_sum_axis(&sum_raw, top, -1, true, s));
        const eps = mlx.mlx_array_new_float(1e-20);
        defer _ = mlx.mlx_array_free(eps);
        var sum_eps = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum_eps);
        try mlx.check(mlx.mlx_add(&sum_eps, sum_raw, eps, s));
        try mlx.check(mlx.mlx_divide(&weights_f32, top, sum_eps, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&weights_f32, top));
    }

    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    if (route_scale != 1.0) {
        const scale_arr = mlx.mlx_array_new_float(route_scale);
        defer _ = mlx.mlx_array_free(scale_arr);
        try mlx.check(mlx.mlx_multiply(&scaled, weights_f32, scale_arr, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&scaled, weights_f32));
    }

    var norm_scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(norm_scores);
    try mlx.check(mlx.mlx_astype(&norm_scores, scaled, .bfloat16, s));

    return .{ .inds = inds, .norm_scores = norm_scores };
}

/// kolibri1 routing: the top-k is taken on `logits + expert_bias` (raw logits,
/// not sigmoid scores), the weights are `sigmoid(logits)` of the picks with no
/// renormalisation and no scale. Weights stay f32; the caller casts them.
pub fn kolibriRoutingChain(router_logits: mlx.mlx_array, expert_bias: mlx.mlx_array, k: c_int, s: mlx.mlx_stream) !MoeRouting {
    var logits_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits_f32);
    try mlx.check(mlx.mlx_astype(&logits_f32, router_logits, .float32, s));

    var biased = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(biased);
    try mlx.check(mlx.mlx_add(&biased, logits_f32, expert_bias, s));
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, biased, s));
    var partitioned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(partitioned);
    try mlx.check(mlx.mlx_argpartition_axis(&partitioned, neg, k - 1, -1, s));

    const p_shape = mlx.getShape(partitioned);
    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    {
        var start_arr: [4]c_int = undefined;
        var stop_arr: [4]c_int = undefined;
        var strides_arr: [4]c_int = undefined;
        for (0..p_shape.len) |d| {
            start_arr[d] = 0;
            stop_arr[d] = if (d == p_shape.len - 1) k else p_shape[d];
            strides_arr[d] = 1;
        }
        try mlx.check(mlx.mlx_slice(&inds, partitioned, &start_arr, p_shape.len, &stop_arr, p_shape.len, &strides_arr, p_shape.len, s));
    }

    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try mlx.check(mlx.mlx_take_along_axis(&picked, logits_f32, inds, -1, s));
    var norm_scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(norm_scores);
    try mlx.check(mlx.mlx_sigmoid(&norm_scores, picked, s));
    return .{ .inds = inds, .norm_scores = norm_scores };
}

// ── tests ──

test "moeRoutingChain produces top-K indices and renormalized softmax weights" {
    const s = mlx.gpuStream();

    // Two rows of router logits over 6 experts. Top-2 of each row is unambiguous:
    //   row 0: experts {0, 3} (logits 10 and 5)
    //   row 1: experts {1, 4} (logits 10 and 5)
    const n_rows: c_int = 2;
    const n_exp: c_int = 6;
    const k: c_int = 2;
    const data = [_]f32{
        10.0, 0.0,  0.0, 5.0, 0.0, 0.0,
        0.0,  10.0, 0.0, 0.0, 5.0, 0.0,
    };
    const shape = [_]c_int{ n_rows, n_exp };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const routed = try moeRoutingChain(logits, k, s);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);

    // Shape check (cheap, no data read needed).
    {
        const inds_shape = mlx.getShape(routed.inds);
        try testing.expectEqual(@as(usize, 2), inds_shape.len);
        try testing.expectEqual(n_rows, inds_shape[0]);
        try testing.expectEqual(k, inds_shape[1]);
        const sc_shape = mlx.getShape(routed.norm_scores);
        try testing.expectEqual(@as(usize, 2), sc_shape.len);
        try testing.expectEqual(n_rows, sc_shape[0]);
        try testing.expectEqual(k, sc_shape[1]);
    }

    // To verify top-K correctness without reading non-contiguous slice memory
    // directly, gather the original logits at the selected indices: gathered[i,j]
    // == logits[i, inds[i,j]]. Then sum across K — for our fixture, the top-2
    // logits in each row are {10, 5}, so the per-row sum must be 15.
    var gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered);
    try mlx.check(mlx.mlx_take_along_axis(&gathered, logits, routed.inds, -1, s));
    var gathered_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered_sum);
    try mlx.check(mlx.mlx_sum_axis(&gathered_sum, gathered, -1, false, s));

    // norm_scores must sum to 1 along K (verifies the renormalize step).
    var scores_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_sum);
    try mlx.check(mlx.mlx_sum_axis(&scores_sum, routed.norm_scores, -1, false, s));

    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, gathered_sum);
        _ = mlx.mlx_vector_array_append_value(ev, scores_sum);
        try mlx.check(mlx.mlx_eval(ev));
    }

    // gathered_sum and scores_sum are 1D outputs of sum_axis (contiguous).
    const gs = mlx.mlx_array_data_float32(gathered_sum) orelse return error.InvalidDtype;
    const ss = mlx.mlx_array_data_float32(scores_sum) orelse return error.InvalidDtype;
    const tol: f32 = 1e-3;
    try testing.expect(@abs(gs[0] - 15.0) < tol);
    try testing.expect(@abs(gs[1] - 15.0) < tol);
    try testing.expect(@abs(ss[0] - 1.0) < tol);
    try testing.expect(@abs(ss[1] - 1.0) < tol);
}

test "hy3RoutingChain selects on biased scores but weights by original sigmoid scores" {
    const s = mlx.gpuStream();

    // 1 row over 6 experts, k=2. sigmoid(logits):
    //   [0.8808, 0.5, 0.1192, 0.7311, 0.2689, 0.6225]
    // With NO bias the top-2 would be {0 (0.8808), 3 (0.7311)}. The expert bias
    // +0.5 on expert 1 lifts it to 1.0, so selection must pick {1, 0} — but the
    // WEIGHTS must come from the ORIGINAL sigmoid scores {0.5, 0.8808}
    // (sum 1.3808), renormalized to 1 then scaled by router_scaling_factor.
    const logits_data = [_]f32{ 2.0, 0.0, -2.0, 1.0, -1.0, 0.5 };
    const shape = [_]c_int{ 1, 6 };
    const logits = mlx.mlx_array_new_data(&logits_data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const bias_data = [_]f32{ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0 };
    const bias_shape = [_]c_int{6};
    const bias = mlx.mlx_array_new_data(&bias_data, &bias_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const scale: f32 = 2.0;
    const routed = try hy3RoutingChain(logits, bias, 2, true, scale, s);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);

    // Gather the ORIGINAL sigmoid scores at the selected indices. If selection
    // ignored the bias the gathered sum would be 0.8808+0.7311 = 1.6119; with
    // bias-driven selection it must be 0.5+0.8808 = 1.3808.
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, logits, s));
    var gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered);
    try mlx.check(mlx.mlx_take_along_axis(&gathered, sig, routed.inds, -1, s));
    var gathered_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered_sum);
    try mlx.check(mlx.mlx_sum_axis(&gathered_sum, gathered, -1, false, s));

    // Renorm + scale: scores sum to exactly router_scaling_factor.
    var scores_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_f32);
    try mlx.check(mlx.mlx_astype(&scores_f32, routed.norm_scores, .float32, s));
    var scores_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_sum);
    try mlx.check(mlx.mlx_sum_axis(&scores_sum, scores_f32, -1, false, s));

    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, gathered_sum);
        _ = mlx.mlx_vector_array_append_value(ev, scores_sum);
        try mlx.check(mlx.mlx_eval(ev));
    }

    const gs = mlx.mlx_array_data_float32(gathered_sum) orelse return error.InvalidDtype;
    const ss = mlx.mlx_array_data_float32(scores_sum) orelse return error.InvalidDtype;
    // norm_scores are bf16 (cast for the expert-combine multiply) → ~0.4% rel.
    try testing.expect(@abs(gs[0] - 1.3808) < 1e-3);
    try testing.expect(@abs(ss[0] - scale) < 0.02);

    // Per-element weights pin "weighted by ORIGINAL scores": {0.5, 0.8808}
    // renormed ×2 → {0.7242, 1.2758} (any order). Had the chain weighted by
    // the BIASED scores {1.0, 0.8808} the pair would be {1.0634, 0.9366} —
    // same sum, different elements, so the sum check alone can't see it.
    try mlx.check(mlx.mlx_array_eval(scores_f32));
    const sv = mlx.mlx_array_data_float32(scores_f32) orelse return error.InvalidDtype;
    const lo = @min(sv[0], sv[1]);
    const hi = @max(sv[0], sv[1]);
    try testing.expect(@abs(lo - 0.7242) < 0.02);
    try testing.expect(@abs(hi - 1.2758) < 0.02);
}

test "hy3RoutingChain route_norm=false keeps raw sigmoid weights (scaled only)" {
    const s = mlx.gpuStream();

    const logits_data = [_]f32{ 2.0, 0.0, -2.0, 1.0, -1.0, 0.5 };
    const shape = [_]c_int{ 1, 6 };
    const logits = mlx.mlx_array_new_data(&logits_data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Zero bias: selection = top-2 of raw sigmoid = {0.8808, 0.7311}.
    const bias_data: [6]f32 = @splat(0.0);
    const bias_shape = [_]c_int{6};
    const bias = mlx.mlx_array_new_data(&bias_data, &bias_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const routed = try hy3RoutingChain(logits, bias, 2, false, 2.0, s);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);

    var scores_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_f32);
    try mlx.check(mlx.mlx_astype(&scores_f32, routed.norm_scores, .float32, s));
    var scores_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_sum);
    try mlx.check(mlx.mlx_sum_axis(&scores_sum, scores_f32, -1, false, s));
    try mlx.check(mlx.mlx_array_eval(scores_sum));

    const ss = mlx.mlx_array_data_float32(scores_sum) orelse return error.InvalidDtype;
    // (0.880797 + 0.731059) × 2.0 = 3.2237 — un-normalized, scaled.
    try testing.expect(@abs(ss[0] - 3.2237) < 0.02);
}
