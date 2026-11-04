//! Batched MTP verify graphs: exact, accept, typical, TokenV3 and lossy.
const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const mtp_acceptance = @import("../../../../sdk/spec/mtp_acceptance.zig");
const testing = std.testing;
const Generator = @import("../../generate/generator.zig").Generator;
const SamplingParams = @import("../../sampling/params.zig").SamplingParams;
const deinit = @import("../../generate/generator.zig").Generator.deinit;
const seedKey = @import("../../sampling/lazy.zig").seedKey;
const verifyRows2d = @import("../../sampling/spec_sampling.zig").verifyRows2d;


/// Batched accept/correction graph for one stochastic round. Replaces
/// the per-position construction — m single-element gathers + concat
/// for accept_p (again for accept_q under sharp drafts) and 1+m
/// SEPARATE full-vocab log+categorical correction samplers — with
/// single batched kernels: one [m,1] take_along_axis per accept vector
/// and ONE [1+m, V] log+categorical covering every possible reject
/// position plus the full-accept bonus. Identical math and identical
/// output distributions; only the dispatch count (and the PRNG key
/// split pattern — one batched categorical draw instead of 1+m
/// sequential ones) changes.
pub const MtpBatchedGraph = struct {
    /// [1+m] int32 pre-sampled corrections (index a = residual sample
    /// for a reject at position a; index m = full-accept bonus).
    corr_samples: mlx.mlx_array,
    /// [m] f32 filtered target probability of each draft.
    accept_p: mlx.mlx_array,
    /// [m] f32 proposal density at each draft (sharp drafts only;
    /// null-ctx under greedy proposals). Typical mode stores its entropy
    /// floor here instead.
    accept_q: mlx.mlx_array,
    /// [m] bool TokenV3 deferral decisions; null for other modes.
    accept_defer: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *MtpBatchedGraph) void {
        _ = mlx.mlx_array_free(self.corr_samples);
        _ = mlx.mlx_array_free(self.accept_p);
        if (self.accept_q.ctx != null) _ = mlx.mlx_array_free(self.accept_q);
        if (self.accept_defer.ctx != null) _ = mlx.mlx_array_free(self.accept_defer);
    }
};

pub fn mtpBatchedExactGraph(
    probs_all: mlx.mlx_array,
    draft_arrs: []const mlx.mlx_array,
    q_probs: ?[]const mlx.mlx_array,
    m: u32,
    _: f32,
    sampling: SamplingParams,
    s: mlx.mlx_stream,
) !MtpBatchedGraph {
    return mtpBatchedAcceptGraph(probs_all, draft_arrs, q_probs, m, sampling, s);
}

pub fn mtpBatchedAcceptGraph(
    probs_all: mlx.mlx_array,
    draft_arrs: []const mlx.mlx_array,
    q_probs: ?[]const mlx.mlx_array,
    m: u32,
    sampling: SamplingParams,
    s: mlx.mlx_stream,
) !MtpBatchedGraph {
    const shape = mlx.getShape(probs_all); // [1, 1+m, V]
    const vocab = shape[2];
    const mi: c_int = @intCast(m);
    const strides2 = [_]c_int{ 1, 1 };

    const p2d = try verifyRows2d(probs_all, m, s);
    defer _ = mlx.mlx_array_free(p2d);

    var ids_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_2d);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (draft_arrs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_concatenate_axis(&flat, vec, 0, s));
        const id2_shape = [_]c_int{ mi, 1 };
        try mlx.check(mlx.mlx_reshape(&ids_2d, flat, &id2_shape, 2, s));
    }

    // Draft-position rows [m, V] and the bonus row [1, V] (slice views).
    var p_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p_rows);
    var bonus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bonus);
    {
        const r_start = [_]c_int{ 0, 0 };
        const r_stop = [_]c_int{ mi, vocab };
        try mlx.check(mlx.mlx_slice(&p_rows, p2d, &r_start, 2, &r_stop, 2, &strides2, 2, s));
        const b_start = [_]c_int{ mi, 0 };
        const b_stop = [_]c_int{ mi + 1, vocab };
        try mlx.check(mlx.mlx_slice(&bonus, p2d, &b_start, 2, &b_stop, 2, &strides2, 2, s));
    }

    // Proposal stack [m, V]: the full sharpened q rows under sharp
    // drafts (exact Leviathan residual), one-hot of the lazy draft ids
    // under greedy proposals.
    var proposal = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(proposal);
    if (q_probs) |qs| {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (qs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        try mlx.check(mlx.mlx_concatenate_axis(&proposal, vec, 0, s));
    } else {
        var indices = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(indices);
        try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));
        var onehot_b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(onehot_b);
        try mlx.check(mlx.mlx_equal(&onehot_b, indices, ids_2d, s));
        try mlx.check(mlx.mlx_astype(&proposal, onehot_b, mlx.mlx_array_dtype(p2d), s));
    }

    // residual = max(p − proposal, 0) rows, then [residual; bonus] and
    // ONE categorical over the whole stack.
    var corr_samples = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(corr_samples);
    {
        var diff = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(diff);
        try mlx.check(mlx.mlx_subtract(&diff, p_rows, proposal, s));
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        var residual = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(residual);
        try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));
        var stack = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stack);
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, residual);
            _ = mlx.mlx_vector_array_append_value(vec, bonus);
            try mlx.check(mlx.mlx_concatenate_axis(&stack, vec, 0, s));
        }
        var log_stack = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(log_stack);
        try mlx.check(mlx.mlx_log(&log_stack, stack, s));
        const key = seedKey(sampling);
        defer _ = mlx.mlx_array_free(key);
        var sampled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sampled);
        try mlx.check(mlx.mlx_random_categorical(&sampled, log_stack, -1, key, s));
        try mlx.check(mlx.mlx_astype(&corr_samples, sampled, .int32, s));
    }

    // accept_p[k] = p_rows[k][draft_k]; accept_q[k] = proposal density.
    var accept_p = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(accept_p);
    var accept_q: mlx.mlx_array = .{ .ctx = null };
    errdefer if (accept_q.ctx != null) {
        _ = mlx.mlx_array_free(accept_q);
    };
    {
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_along_axis(&taken, p_rows, ids_2d, -1, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const m_shape = [_]c_int{mi};
        try mlx.check(mlx.mlx_reshape(&flat, taken, &m_shape, 1, s));
        try mlx.check(mlx.mlx_astype(&accept_p, flat, .float32, s));
    }
    if (q_probs != null) {
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_along_axis(&taken, proposal, ids_2d, -1, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const m_shape = [_]c_int{mi};
        try mlx.check(mlx.mlx_reshape(&flat, taken, &m_shape, 1, s));
        accept_q = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&accept_q, flat, .float32, s));
    }

    return .{ .corr_samples = corr_samples, .accept_p = accept_p, .accept_q = accept_q };
}

pub fn mtpBatchedTypicalGraph(
    probs_all: mlx.mlx_array,
    draft_arrs: []const mlx.mlx_array,
    q_probs: ?[]const mlx.mlx_array,
    m: u32,
    delta: f32,
    sampling: SamplingParams,
    s: mlx.mlx_stream,
) !MtpBatchedGraph {
    return mtpBatchedLossyGraph(.typical, probs_all, draft_arrs, q_probs, m, delta, sampling, s);
}

pub fn mtpBatchedTokenV3Graph(
    probs_all: mlx.mlx_array,
    draft_arrs: []const mlx.mlx_array,
    q_probs: ?[]const mlx.mlx_array,
    m: u32,
    alpha: f32,
    sampling: SamplingParams,
    s: mlx.mlx_stream,
) !MtpBatchedGraph {
    return mtpBatchedLossyGraph(.tokenv3, probs_all, draft_arrs, q_probs, m, alpha, sampling, s);
}

/// Specialized at compile time; the installed route adds no mode switch
/// to a measured MTP round. Both lanes keep corrections in one lazy graph.
pub fn mtpBatchedLossyGraph(
    comptime mode: enum { typical, tokenv3 },
    probs_all: mlx.mlx_array,
    draft_arrs: []const mlx.mlx_array,
    q_probs: ?[]const mlx.mlx_array,
    m: u32,
    param: f32,
    sampling: SamplingParams,
    s: mlx.mlx_stream,
) !MtpBatchedGraph {
    const shape = mlx.getShape(probs_all); // [1, 1+m, V]
    const vocab = shape[2];
    const mi: c_int = @intCast(m);
    const strides2 = [_]c_int{ 1, 1 };

    const p2d = try verifyRows2d(probs_all, m, s);
    defer _ = mlx.mlx_array_free(p2d);

    var ids_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_2d);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (draft_arrs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_concatenate_axis(&flat, vec, 0, s));
        const id2_shape = [_]c_int{ mi, 1 };
        try mlx.check(mlx.mlx_reshape(&ids_2d, flat, &id2_shape, 2, s));
    }

    var p_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p_rows);
    var bonus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bonus);
    {
        const r_start = [_]c_int{ 0, 0 };
        const r_stop = [_]c_int{ mi, vocab };
        try mlx.check(mlx.mlx_slice(&p_rows, p2d, &r_start, 2, &r_stop, 2, &strides2, 2, s));
        const b_start = [_]c_int{ mi, 0 };
        const b_stop = [_]c_int{ mi + 1, vocab };
        try mlx.check(mlx.mlx_slice(&bonus, p2d, &b_start, 2, &b_stop, 2, &strides2, 2, s));
    }

    // TokenV3 needs the FULL q row. Typical acceptance uses only p and
    // does not add a proposal concat or one-hot operation to the graph.
    var proposal: mlx.mlx_array = .{ .ctx = null };
    defer if (proposal.ctx != null) {
        _ = mlx.mlx_array_free(proposal);
    };
    if (comptime mode == .tokenv3) {
        proposal = mlx.mlx_array_new();
        if (q_probs) |qs| {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (qs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
            try mlx.check(mlx.mlx_concatenate_axis(&proposal, vec, 0, s));
        } else {
            var indices = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(indices);
            try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));
            var onehot_b = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(onehot_b);
            try mlx.check(mlx.mlx_equal(&onehot_b, indices, ids_2d, s));
            try mlx.check(mlx.mlx_astype(&proposal, onehot_b, mlx.mlx_array_dtype(p2d), s));
        }
    }

    // TokenV3 Eq. 11: pi = q*1[Top] + p*sum(q outside Top).
    // The p*eta term applies to EVERY token, including those inside Top.
    var pi: mlx.mlx_array = .{ .ctx = null };
    defer if (pi.ctx != null) {
        _ = mlx.mlx_array_free(pi);
    };
    var defer_mask: mlx.mlx_array = .{ .ctx = null };
    defer if (defer_mask.ctx != null) {
        _ = mlx.mlx_array_free(defer_mask);
    };
    if (comptime mode == .tokenv3) {
        var peak = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(peak);
        try mlx.check(mlx.mlx_max_axis(&peak, p_rows, 1, true, s));
        const complement = mlx.mlx_array_new_float(1.0 - param);
        defer _ = mlx.mlx_array_free(complement);
        var cutoff = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cutoff);
        try mlx.check(mlx.mlx_multiply(&cutoff, peak, complement, s));
        defer_mask = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_less(&defer_mask, p_rows, cutoff, s));
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        var deferred_q = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(deferred_q);
        try mlx.check(mlx.mlx_where(&deferred_q, defer_mask, proposal, zero, s));
        var eta = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eta);
        try mlx.check(mlx.mlx_sum_axis(&eta, deferred_q, 1, true, s));
        var kept_q = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kept_q);
        try mlx.check(mlx.mlx_where(&kept_q, defer_mask, zero, proposal, s));
        var recycled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(recycled);
        try mlx.check(mlx.mlx_multiply(&recycled, p_rows, eta, s));
        pi = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&pi, kept_q, recycled, s));
    }
    const decision_rows = if (comptime mode == .tokenv3) pi else p_rows;

    var corr_samples = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(corr_samples);
    {
        var corr_rows: mlx.mlx_array = .{ .ctx = null };
        defer if (corr_rows.ctx != null) {
            _ = mlx.mlx_array_free(corr_rows);
        };
        if (comptime mode == .typical) {
            corr_rows = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&corr_rows, p_rows));
        } else {
            var diff = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(diff);
            try mlx.check(mlx.mlx_subtract(&diff, pi, proposal, s));
            const zero = mlx.mlx_array_new_float(0.0);
            defer _ = mlx.mlx_array_free(zero);
            corr_rows = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_maximum(&corr_rows, diff, zero, s));
        }
        var stack = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stack);
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, corr_rows);
        _ = mlx.mlx_vector_array_append_value(vec, bonus);
        try mlx.check(mlx.mlx_concatenate_axis(&stack, vec, 0, s));
        var log_stack = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(log_stack);
        try mlx.check(mlx.mlx_log(&log_stack, stack, s));
        const key = seedKey(sampling);
        defer _ = mlx.mlx_array_free(key);
        var sampled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sampled);
        try mlx.check(mlx.mlx_random_categorical(&sampled, log_stack, -1, key, s));
        try mlx.check(mlx.mlx_astype(&corr_samples, sampled, .int32, s));
    }

    var accept_p = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(accept_p);
    {
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_along_axis(&taken, decision_rows, ids_2d, -1, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const m_shape = [_]c_int{mi};
        try mlx.check(mlx.mlx_reshape(&flat, taken, &m_shape, 1, s));
        try mlx.check(mlx.mlx_astype(&accept_p, flat, .float32, s));
    }

    var accept_q: mlx.mlx_array = .{ .ctx = null };
    errdefer if (accept_q.ctx != null) {
        _ = mlx.mlx_array_free(accept_q);
    };
    if (comptime mode == .typical) {
        // H(p) = -sum p*log(p), with log(0) safely masked by clamping
        // only the log operand. The p factor remains exactly zero.
        const tiny = mlx.mlx_array_new_float(1e-30);
        defer _ = mlx.mlx_array_free(tiny);
        var safe_p = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(safe_p);
        try mlx.check(mlx.mlx_maximum(&safe_p, p_rows, tiny, s));
        var log_p = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(log_p);
        try mlx.check(mlx.mlx_log(&log_p, safe_p, s));
        var p_log_p = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(p_log_p);
        try mlx.check(mlx.mlx_multiply(&p_log_p, p_rows, log_p, s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_sum_axis(&sum, p_log_p, 1, false, s));
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_exp(&scaled, sum, s)); // exp(-H)
        const delta = mlx.mlx_array_new_float(param);
        defer _ = mlx.mlx_array_free(delta);
        var raw_floor = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(raw_floor);
        try mlx.check(mlx.mlx_multiply(&raw_floor, scaled, delta, s));
        const eps = mlx.mlx_array_new_float(1.0);
        defer _ = mlx.mlx_array_free(eps);
        accept_q = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_minimum(&accept_q, raw_floor, eps, s));
    } else if (q_probs != null) {
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_along_axis(&taken, proposal, ids_2d, -1, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const m_shape = [_]c_int{mi};
        try mlx.check(mlx.mlx_reshape(&flat, taken, &m_shape, 1, s));
        accept_q = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&accept_q, flat, .float32, s));
    }

    var accept_defer: mlx.mlx_array = .{ .ctx = null };
    errdefer if (accept_defer.ctx != null) {
        _ = mlx.mlx_array_free(accept_defer);
    };
    if (comptime mode == .tokenv3) {
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_along_axis(&taken, defer_mask, ids_2d, -1, s));
        const m_shape = [_]c_int{mi};
        accept_defer = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&accept_defer, taken, &m_shape, 1, s));
    }
    return .{ .corr_samples = corr_samples, .accept_p = accept_p, .accept_q = accept_q, .accept_defer = accept_defer };
}

// ── tests ──


test "batched corrections: point-mass residuals sample deterministically, accept vec gathers draft probs" {
    const s = mlx.gpuStream();
    // V=4, m=2. Row residuals are POINT MASSES so the batched categorical
    // is deterministic end-to-end (no seed dependence):
    //   row0: p=[.5,.5,0,0], draft=0 → residual [0,.5,0,0] → corr 1
    //   row1: p=[0,.25,.75,0], draft=2 → residual [0,.25,0,0] → corr 1
    //   bonus: p=[0,0,0,1] → corr 3
    const p_data = [_]f32{ 0.5, 0.5, 0, 0, 0, 0.25, 0.75, 0, 0, 0, 0, 1.0 };
    const p_shape = [_]c_int{ 1, 3, 4 };
    const probs_all = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs_all);
    const id_shape = [_]c_int{1};
    const d0_data: i32 = 0;
    const d1_data: i32 = 2;
    const d0 = mlx.mlx_array_new_data(&d0_data, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d0);
    const d1 = mlx.mlx_array_new_data(&d1_data, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d1);
    const drafts = [_]mlx.mlx_array{ d0, d1 };

    // Greedy proposals (q == null): one-hot residuals.
    var g = try Generator.mtpBatchedAcceptGraph(probs_all, &drafts, null, 2, .{}, s);
    defer g.deinit();
    try mlx.check(mlx.mlx_array_eval(g.corr_samples));
    const corr = mlx.mlx_array_data_int32(g.corr_samples) orelse return error.InvalidDtype;
    try testing.expectEqual(@as(i32, 1), corr[0]);
    try testing.expectEqual(@as(i32, 1), corr[1]);
    try testing.expectEqual(@as(i32, 3), corr[2]);
    try mlx.check(mlx.mlx_array_eval(g.accept_p));
    const ap = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 0.5), ap[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.75), ap[1], 1e-6);
    try testing.expect(g.accept_q.ctx == null);

    // Sharp proposals: q rows equal to the one-hots reproduce the same
    // residuals, and accept_q gathers the proposal's own density.
    const q_data0 = [_]f32{ 1, 0, 0, 0 };
    const q_data1 = [_]f32{ 0, 0, 1, 0 };
    const q_shape = [_]c_int{ 1, 4 };
    const q0 = mlx.mlx_array_new_data(&q_data0, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q0);
    const q1 = mlx.mlx_array_new_data(&q_data1, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q1);
    const qs = [_]mlx.mlx_array{ q0, q1 };
    var gs = try Generator.mtpBatchedAcceptGraph(probs_all, &drafts, &qs, 2, .{}, s);
    defer gs.deinit();
    try mlx.check(mlx.mlx_array_eval(gs.corr_samples));
    const corr2 = mlx.mlx_array_data_int32(gs.corr_samples) orelse return error.InvalidDtype;
    try testing.expectEqual(@as(i32, 1), corr2[0]);
    try testing.expectEqual(@as(i32, 1), corr2[1]);
    try testing.expectEqual(@as(i32, 3), corr2[2]);
    try mlx.check(mlx.mlx_array_eval(gs.accept_q));
    const aq = mlx.mlx_array_data_float32(gs.accept_q) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 1.0), aq[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), aq[1], 1e-6);
}

test "batched corrections read only 1+m rows of a group-padded verify block" {
    const s = mlx.gpuStream();
    // The block above plus one pad row, m=2.
    const p_data = [_]f32{ 0.5, 0.5, 0, 0, 0, 0.25, 0.75, 0, 0, 0, 0, 1.0, 1.0, 0, 0, 0 };
    const p_shape = [_]c_int{ 1, 4, 4 };
    const probs_all = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs_all);
    const id_shape = [_]c_int{1};
    const d0_data: i32 = 0;
    const d1_data: i32 = 2;
    const d0 = mlx.mlx_array_new_data(&d0_data, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d0);
    const d1 = mlx.mlx_array_new_data(&d1_data, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d1);
    const drafts = [_]mlx.mlx_array{ d0, d1 };

    var g = try Generator.mtpBatchedAcceptGraph(probs_all, &drafts, null, 2, .{}, s);
    defer g.deinit();
    try mlx.check(mlx.mlx_array_eval(g.corr_samples));
    const corr = mlx.mlx_array_data_int32(g.corr_samples) orelse return error.InvalidDtype;
    try testing.expectEqual(@as(i32, 1), corr[0]);
    try testing.expectEqual(@as(i32, 1), corr[1]);
    try testing.expectEqual(@as(i32, 3), corr[2]);
}

test "MTP typical batched graph uses entropy floor and target-row correction" {
    const s = mlx.gpuStream();
    const p_data = [_]f32{ 0.5, 0.5, 0, 0, 0, 0, 0, 1 };
    const p_shape = [_]c_int{ 1, 2, 4 };
    const probs = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const id: i32 = 0;
    const id_shape = [_]c_int{1};
    const draft = mlx.mlx_array_new_data(&id, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(draft);
    const drafts = [_]mlx.mlx_array{draft};
    var g = try Generator.mtpBatchedTypicalGraph(probs, &drafts, null, 1, 0.2, .{}, s);
    defer g.deinit();
    try mlx.check(mlx.mlx_array_eval(g.accept_p));
    try mlx.check(mlx.mlx_array_eval(g.accept_q));
    const p = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
    const floor = mlx.mlx_array_data_float32(g.accept_q) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 0.5), p[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), floor[0], 1e-6);
    try testing.expect(g.accept_defer.ctx == null);
    try mlx.check(mlx.mlx_array_eval(g.corr_samples));
    const corr = mlx.mlx_array_data_int32(g.corr_samples) orelse return error.InvalidDtype;
    try testing.expect(corr[0] == 0 or corr[0] == 1);
    try testing.expectEqual(@as(i32, 3), corr[1]);
}

test "a prompt-lookup draft is kept only as often as sampling would keep it under typical acceptance" {
    // A copied token at p = 0.05 in a high-entropy row clears typical's floor (0.2 * e^-H), so
    // the typical route keeps it every time; a lookup round must keep it about 5% of the time.
    const s = mlx.gpuStream();
    const V = 32;
    var p_data: [2 * V]f32 = undefined;
    p_data[0] = 0.05;
    for (p_data[1..V]) |*v| v.* = 0.95 / @as(f32, V - 1);
    for (p_data[V..]) |*v| v.* = 1.0 / @as(f32, V);
    const p_shape = [_]c_int{ 1, 2, V };
    const probs = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const id: i32 = 0;
    const id_shape = [_]c_int{1};
    const draft = mlx.mlx_array_new_data(&id, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(draft);
    const drafts = [_]mlx.mlx_array{draft};

    var g: Generator = undefined;
    g.mtp_accept_graph = Generator.mtpBatchedTypicalGraph;
    g.mtp_accept_prefix = mtp_acceptance.typicalPrefix;
    g.mtp_accept_param = 0.2;
    var prng = std.Random.DefaultPrng.init(0x100C);
    var kept: [2]u32 = .{ 0, 0 };
    for ([_]bool{ false, true }, 0..) |lookup, arm| {
        var bg = try g.acceptGraphFor(lookup)(probs, &drafts, null, 1, g.mtp_accept_param, .{}, s);
        defer bg.deinit();
        try mlx.check(mlx.mlx_array_eval(bg.accept_p));
        const p = mlx.mlx_array_data_float32(bg.accept_p) orelse return error.InvalidDtype;
        var q: ?[]const f32 = null;
        if (bg.accept_q.ctx != null) {
            try mlx.check(mlx.mlx_array_eval(bg.accept_q));
            q = (mlx.mlx_array_data_float32(bg.accept_q) orelse return error.InvalidDtype)[0..1];
        }
        for (0..400) |_| kept[arm] += g.acceptPrefixFor(lookup)(p[0..1], q, null, &prng);
    }
    try testing.expectEqual(@as(u32, 400), kept[0]);
    try testing.expect(kept[1] > 0 and kept[1] < 60);
}

test "MTP TokenV3 batched graph uses full pi for sampled and one-hot drafts" {
    const s = mlx.gpuStream();
    const p_data = [_]f32{ 0.8, 0.19, 0.01, 0, 0, 0, 0, 1 };
    const p_shape = [_]c_int{ 1, 2, 4 };
    const probs = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const id: i32 = 2;
    const id_shape = [_]c_int{1};
    const draft = mlx.mlx_array_new_data(&id, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(draft);
    const drafts = [_]mlx.mlx_array{draft};
    const q_data = [_]f32{ 0.1, 0.2, 0.7, 0 };
    const q_shape = [_]c_int{ 1, 4 };
    const q = mlx.mlx_array_new_data(&q_data, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q);
    const qs = [_]mlx.mlx_array{q};

    var sampled = try Generator.mtpBatchedTokenV3Graph(probs, &drafts, &qs, 1, 0.95, .{}, s);
    defer sampled.deinit();
    try mlx.check(mlx.mlx_array_eval(sampled.accept_p));
    try mlx.check(mlx.mlx_array_eval(sampled.accept_q));
    try mlx.check(mlx.mlx_array_eval(sampled.accept_defer));
    const pi_draft = mlx.mlx_array_data_float32(sampled.accept_p) orelse return error.InvalidDtype;
    const q_draft = mlx.mlx_array_data_float32(sampled.accept_q) orelse return error.InvalidDtype;
    const deferred = mlx.mlx_array_data_bool(sampled.accept_defer) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 0.007), pi_draft[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.7), q_draft[0], 1e-6);
    try testing.expect(deferred[0]);
    try mlx.check(mlx.mlx_array_eval(sampled.corr_samples));
    const corr = mlx.mlx_array_data_int32(sampled.corr_samples) orelse return error.InvalidDtype;
    try testing.expect(corr[0] == 0 or corr[0] == 1);
    try testing.expectEqual(@as(i32, 3), corr[1]);

    var onehot = try Generator.mtpBatchedTokenV3Graph(probs, &drafts, null, 1, 0.95, .{}, s);
    defer onehot.deinit();
    try mlx.check(mlx.mlx_array_eval(onehot.accept_p));
    const onehot_pi = mlx.mlx_array_data_float32(onehot.accept_p) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 0.01), onehot_pi[0], 1e-6);
    try testing.expect(onehot.accept_q.ctx == null);
}

test "MTP TokenV3 full Qwen3.8 vocabulary graph stays finite at depth three" {
    const s = mlx.gpuStream();
    const vocab: usize = 248_320;
    const p_data = try testing.allocator.alloc(f32, 4 * vocab);
    defer testing.allocator.free(p_data);
    @memset(p_data, 0);
    for (0..3) |row| {
        p_data[row * vocab] = 0.8;
        p_data[row * vocab + 1] = 0.19;
        p_data[row * vocab + 2] = 0.01;
    }
    p_data[3 * vocab + 3] = 1.0;
    const p_shape = [_]c_int{ 1, 4, @intCast(vocab) };
    const probs = mlx.mlx_array_new_data(p_data.ptr, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const q_data = try testing.allocator.alloc(f32, vocab);
    defer testing.allocator.free(q_data);
    @memset(q_data, 0);
    q_data[0] = 0.1;
    q_data[1] = 0.2;
    q_data[2] = 0.7;
    const q_shape = [_]c_int{ 1, @intCast(vocab) };
    const q = mlx.mlx_array_new_data(q_data.ptr, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q);
    const draft_id: i32 = 2;
    const id_shape = [_]c_int{1};
    const draft = mlx.mlx_array_new_data(&draft_id, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(draft);
    const drafts = [_]mlx.mlx_array{ draft, draft, draft };
    const qs = [_]mlx.mlx_array{ q, q, q };
    var g = try Generator.mtpBatchedTokenV3Graph(probs, &drafts, &qs, 3, 0.95, .{}, s);
    defer g.deinit();
    try mlx.check(mlx.mlx_array_eval(g.accept_p));
    try mlx.check(mlx.mlx_array_eval(g.accept_defer));
    try mlx.check(mlx.mlx_array_eval(g.corr_samples));
    const pi_draft = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
    const deferred = mlx.mlx_array_data_bool(g.accept_defer) orelse return error.InvalidDtype;
    const corr = mlx.mlx_array_data_int32(g.corr_samples) orelse return error.InvalidDtype;
    for (0..3) |k| {
        try testing.expectApproxEqAbs(@as(f32, 0.007), pi_draft[k], 1e-5);
        try testing.expect(deferred[k]);
        try testing.expect(corr[k] == 0 or corr[k] == 1);
    }
    try testing.expectEqual(@as(i32, 3), corr[3]);
}
