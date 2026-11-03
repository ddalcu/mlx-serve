//! Speculative sampling helpers: draft/verify probability rows, shortlist proposals, residual draws.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const dflash_mod = @import("../spec/dflash/dflash.zig");
const testing = std.testing;
const Generator = @import("../generate/generator.zig").Generator;
const SamplingParams = @import("params.zig").SamplingParams;
const applyMinP = @import("filters.zig").applyMinP;
const applySuppressMask = @import("params.zig").applySuppressMask;
const filterTopKTopP = @import("filters.zig").filterTopKTopP;
const sampleFromProbsLazy = @import("lazy.zig").sampleFromProbsLazy;
const sampleTokenLazy = @import("lazy.zig").sampleTokenLazy;
const seedKey = @import("lazy.zig").seedKey;


/// Sample a token lazily from logits — returns a lazy MLX array (no eval).
/// Handles temperature scaling, top-k, and top-p, but defers materialization.
/// The returned array has shape [1] with the sampled token ID.
/// Caller must free the returned array.
/// Compute the probability distribution over the vocabulary at the LAST
/// position of `logits_3d` (shape `[B, S, V]`), with the SAME temperature +
/// top-k + top-p masking the sampler would apply. Both `target_p` and `draft_q`
/// in the stochastic-verify accept test must be computed via this function so
/// the ratio `p[draft] / q[draft]` is well-defined over the kept support.
/// Caller owns the returned array; shape `[B, V]`.
/// Batched sibling of `probsAtLastPos`: temperature → top-k → top-p → min-p →
/// softmax over EVERY position of `[1, L, V]` logits in one set of
/// row-parallel kernels. A per-position loop pays L separate ~vocab-sized
/// sort/topk kernel launches per spec-decode round; batched it's one each.
/// All filter helpers operate on the last axis, so leading dims pass through.
pub fn probsAllPositions(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var current = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(current);
    try mlx.check(mlx.mlx_array_set(&current, logits_3d));

    // Reserved-token suppression (batched MTP stochastic verify) — same
    // rationale as `probsAtLastPos`.
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applySuppressMask(&masked, current, m, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        filterTopKTopP(&masked, current, sampling.top_p, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }
    if ((sampling.min_p orelse 0) > 0) {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applyMinP(&masked, current, sampling.min_p.?, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    var probs = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    current = .{ .ctx = null };
    return probs;
}

/// The target densities of a whole GROUP of verify rows in one filtered block:
/// the rows' `[1, Σ(1+m_i), V]` logits filtered as one. The filter is row-wise
/// on the last axis, so a row's slice of this block is byte-for-byte the
/// `probsAllPositions` of that row alone.
pub fn groupProbsBlock(block_logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    return probsAllPositions(block_logits, sampling, s);
}

/// The `[1+m, V]` rows a round reads from its verify block: a batched group
/// right-pads every row to its widest draft, so the block may be longer.
pub fn verifyRows2d(probs_all: mlx.mlx_array, m: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(probs_all);
    const rows: c_int = @intCast(1 + m);
    if (shape.len != 3 or shape[0] != 1 or shape[1] < rows) return error.MtpVerifyBlockShape;
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, probs_all, &.{ 0, 0, 0 }, 3, &.{ 1, rows, shape[2] }, 3, &.{ 1, 1, 1 }, 3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, head, &.{ rows, shape[2] }, 2, s));
    return out;
}

/// One row's `[1, 1+m, V]` densities as a VIEW into the group's block.
pub fn groupProbsRow(block_probs: mlx.mlx_array, offset: c_int, len: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(block_probs);
    const start = [_]c_int{ 0, offset, 0 };
    const stop = [_]c_int{ shape[0], offset + @as(c_int, @intCast(len)), shape[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, block_probs, &start, 3, &stop, 3, &strides, 3, s));
    return out;
}

/// Lazy log-confidence of one MTP draft: `logits[draft] − logsumexp(logits)`
/// = log p_head(draft). Two vocab reductions on the head's own (draft-head)
/// logits — must be built BEFORE the caller frees the logits handle (lazy
/// graphs hold their inputs internally). Returns a `[1]`-shaped lazy array.
pub fn draftConfidenceGraph(logits: mlx.mlx_array, draft_id: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var lse = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lse);
    try mlx.check(mlx.mlx_logsumexp_axis(&lse, logits, -1, false, s));
    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, logits, draft_id, -1, s));
    const flat = [_]c_int{1};
    var t_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t_flat);
    try mlx.check(mlx.mlx_reshape(&t_flat, taken, &flat, 1, s));
    var l_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l_flat);
    try mlx.check(mlx.mlx_reshape(&l_flat, lse, &flat, 1, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_subtract(&out, t_flat, l_flat, s));
    return out;
}

/// The chunk-A boundary sync: ONE bounded GPU round-trip that realizes the
/// chunk's draft ids (needed on the CPU later anyway) plus their
/// confidences, and returns the chain log-confidence
/// `Σ min(0, ln p_head(draft_i))` for the extension gate.
pub fn readChainConfidence(draft_arrs: []const mlx.mlx_array, conf_arrs: []const mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var conf_vec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conf_vec);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (conf_arrs) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
        try mlx.check(mlx.mlx_astype(&conf_vec, cat, .float32, s));
    }
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        for (draft_arrs) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
        _ = mlx.mlx_vector_array_append_value(eval_vec, conf_vec);
        try mlx.check(mlx.mlx_async_eval(eval_vec));
    }
    try mlx.check(mlx.mlx_array_eval(conf_vec));
    const data = mlx.mlx_array_data_float32(conf_vec) orelse return error.MlxArrayDataNull;
    return Generator.mtpChainLogConf(data[0..conf_arrs.len]);
}

pub fn probsAtLastPos(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(logits_3d);
    const seq_len = shape[1];
    var current = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(current);
    if (seq_len == 1) {
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, logits_3d, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ shape[0], seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits_3d, &start, 3, &stop, 3, &strides, 3, s));
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s));
    }

    // Reserved-token suppression: the filtered probs feed spec-verify
    // acceptance AND the residual corrections, so a suppressed draft's
    // acceptance probability is exactly 0 and the residual can't re-draw it.
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applySuppressMask(&masked, current, m, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    const owned = current;
    current = .{ .ctx = null };
    return filteredProbsRows(owned, sampling, s);
}

/// Temperature → top-k → top-p → min-p → softmax over the LAST axis of an owned
/// `[.., V]` array (consumed). The ONE place a proposal or target density is
/// built: a draft sampled from a distribution that is not byte-for-byte the
/// `q` handed to `specAcceptProb` breaks exactness silently, so both sides
/// come through here. Row-independent, so it serves a single `[1, V]` row and
/// a whole `[m, V]` block identically.
fn filteredProbsRows(owned_rows: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var current = owned_rows;
    errdefer _ = mlx.mlx_array_free(current);
    // Apply temperature → top-k → top-p → min-p (same order as `sampleTokenLazy`).
    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        filterTopKTopP(&masked, current, sampling.top_p, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }
    if ((sampling.min_p orelse 0) > 0) {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applyMinP(&masked, current, sampling.min_p.?, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    // Softmax: tokens at -inf become 0, kept tokens renormalize to sum=1.
    var probs = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    current = .{ .ctx = null };
    return probs;
}

/// Proposal densities for EVERY row of a `[1, m, V]` draft-logit block, as
/// `[m, V]`. One filter pass and (at the call site) one categorical over m
/// rows, rather than m of each — the draft block is on the critical path
/// ahead of the verify forward.
pub fn filteredProbsBlock(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(logits_3d);
    var rows = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(rows);
    const rows_shape = [_]c_int{ shape[0] * shape[1], shape[2] };
    try mlx.check(mlx.mlx_reshape(&rows, logits_3d, &rows_shape, 2, s));
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applySuppressMask(&masked, rows, m, s);
        _ = mlx.mlx_array_free(rows);
        rows = masked;
    }
    const owned = rows;
    rows = .{ .ctx = null };
    return filteredProbsRows(owned, sampling, s);
}

/// Read `probs[0, token_id]` as f32. Forces realization with a single eval.
pub fn probAt(probs: mlx.mlx_array, token_id: u32, s: mlx.mlx_stream) !f32 {
    const idx_val: i32 = @intCast(token_id);
    const idx_shape = [_]c_int{1};
    const idx_arr = mlx.mlx_array_new_data(&idx_val, &idx_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(idx_arr);

    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, probs, idx_arr, -1, s));

    // Cast to f32 so item_float32 is exact regardless of source dtype (bf16 etc.).
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, taken, .float32, s));
    try mlx.check(mlx.mlx_array_eval(as_f32));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, as_f32));
    return v;
}

/// Sample one token from probability distribution `probs` (shape `[B, V]`) with
/// the request's key (`seedKey`: a seeded request replays, `draw` = the position).
/// Returns a u32 token id (caller can append directly).
pub fn sampleFromProbs(probs: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !u32 {
    // mlx_random_categorical takes logits and applies softmax. Feed log(probs)
    // so the categorical's softmax recovers the original distribution.
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_log(&log_probs, probs, s));

    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, log_probs, -1, key, s));
    try mlx.check(mlx.mlx_array_eval(sampled));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, sampled));
    return @intCast(v);
}

/// Build a one-hot float32 row vector of shape `[1, vocab]` with 1.0 at
/// `index` and 0.0 elsewhere. Used by PLD's stochastic-verify reject path,
/// which models the draft (an n-gram lookup, not a probabilistic model) as a
/// degenerate one-hot distribution. Caller owns the returned array.
pub fn pldOneHotRow(index: u32, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var indices = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices);
    try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));

    const target_val: i32 = @intCast(index);
    const tgt_shape = [_]c_int{1};
    const target_idx = mlx.mlx_array_new_data(&target_val, &tgt_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(target_idx);

    var mask_bool = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_bool);
    try mlx.check(mlx.mlx_equal(&mask_bool, indices, target_idx, s));

    var mask_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_f32);
    try mlx.check(mlx.mlx_astype(&mask_f32, mask_bool, .float32, s));

    const out_shape = [_]c_int{ 1, vocab };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, mask_f32, &out_shape, 2, s));
    return out;
}

/// One row of a `[m, V]` density block as its own `[1, V]` array, so it can
/// feed `probAt` and `sampleResidual` (both row-shaped). Caller owns it.
pub fn sliceProbRow(rows_2d: mlx.mlx_array, row: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(rows_2d);
    const start = [_]c_int{ @intCast(row), 0 };
    const stop = [_]c_int{ @as(c_int, @intCast(row)) + 1, shape[1] };
    const strides = [_]c_int{ 1, 1 };
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, rows_2d, &start, 2, &stop, 2, &strides, 2, s));
    return out;
}

/// The DFlash2 selector's proposal density for one step as a `[1, V]` row:
/// its per-candidate softmax scattered into zeros — q is zero off the
/// candidate set by construction, so this is the exact density the draft was
/// drawn from (the residual-correction contract).
pub fn selectorQRow(sp: *const dflash_mod.SelectedPath, step: usize, m: u32, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const kk = sp.cand_ids.len / @as(usize, m);
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    const z_shape = [_]c_int{ 1, vocab };
    try mlx.check(mlx.mlx_zeros(&zeros, &z_shape, 2, .float32, s));
    const row_shape = [_]c_int{ 1, @intCast(kk) };
    const ids_arr = mlx.mlx_array_new_data(sp.cand_ids.ptr + step * kk, &row_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(ids_arr);
    const vals_arr = mlx.mlx_array_new_data(sp.q.?.ptr + step * kk, &row_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(vals_arr);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_put_along_axis(&out, zeros, ids_arr, vals_arr, 1, s));
    return out;
}

/// One draft depth's token and the density it was drawn from: the `[1, V]`
/// row the verify's accept ratio and residual read for that depth.
const StepProposal = struct { id: mlx.mlx_array, q: mlx.mlx_array };

/// Sampled draft from the exact re-scored top-32: q is the draft sampler's
/// filtered softmax over the shortlist, the draft is drawn from that q, and q
/// goes to the verify as a dense `[1, V]` row that is zero off the shortlist.
/// Leviathan is exact for any q, so the residual reaches the tokens the
/// shortlist missed; filtering inside 32 rows costs acceptance at most. The
/// contract: the row the verify sees is the array the draft was drawn from.
pub fn shortlistProposal(sl: mtp_mod.Shortlist, sampling: SamplingParams, s: mlx.mlx_stream) !StepProposal {
    const shape = mlx.getShape(sl.exact);
    const n = shape[shape.len - 1];

    // f32: the density `min(1, p/q)` divides by must be the one the
    // categorical drew from, and a bf16 softmax does not sum to 1.
    var rows = mlx.mlx_array_new();
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const row_shape = [_]c_int{ 1, n };
        try mlx.check(mlx.mlx_reshape(&flat, sl.exact, &row_shape, 2, s));
        errdefer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_astype(&rows, flat, .float32, s));
    }
    // The suppress mask already applied to the coarse readout; the shortlist
    // holds no reserved id.
    var draft = sampling;
    draft.suppress_mask = null;
    const q32 = try filteredProbsRows(rows, draft, s);
    errdefer _ = mlx.mlx_array_free(q32);

    var id = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(id);
    {
        const pick = try sampleFromProbsLazy(q32, sampling, s);
        defer _ = mlx.mlx_array_free(pick);
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_take_axis(&picked, sl.cands, pick, 0, s));
        try mlx.check(mlx.mlx_astype(&id, picked, .int32, s));
    }

    const q = try scatterShortlistQ(sl, q32, s);
    _ = mlx.mlx_array_free(q32);
    return .{ .id = id, .q = q };
}

/// An argmax depth of a chain that carries densities (consumes `amax`): the id
/// reshaped to the `[1]` a sampled depth drafts, since the accept graphs
/// concatenate a chain's ids, and its one-hot q.
pub fn argmaxStepProposal(amax: mlx.mlx_array, vocab: c_int, s: mlx.mlx_stream) !StepProposal {
    defer _ = mlx.mlx_array_free(amax);
    var id = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(id);
    try mlx.check(mlx.mlx_reshape(&id, amax, &[_]c_int{1}, 1, s));
    return .{ .id = id, .q = try lazyOneHotRow(id, vocab, s) };
}

/// A depth of a chain that carries densities, off the rerank shortlist: a
/// sampling depth draws from it (`shortlistProposal`), an argmax depth (the
/// greedy tail) takes the token a greedy chain drafts.
pub fn shortlistStepProposal(sl: mtp_mod.Shortlist, sampling: SamplingParams, s: mlx.mlx_stream) !StepProposal {
    if (sampling.temperature > 0.01) return shortlistProposal(sl, sampling, s);
    return argmaxStepProposal(try mtp_mod.shortlistArgmax(s, sl), sl.rows, s);
}

/// `shortlistStepProposal` off the draft head's full `[1, 1, V]` logits: q is
/// the draft sampler's filtered softmax, the density the draft is drawn from.
pub fn logitsStepProposal(logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !StepProposal {
    if (sampling.temperature <= 0.01) {
        const shape = mlx.getShape(logits);
        return argmaxStepProposal(sampleTokenLazy(logits, sampling, s), shape[shape.len - 1], s);
    }
    const q = try probsAtLastPos(logits, sampling, s);
    errdefer _ = mlx.mlx_array_free(q);
    return .{ .id = try sampleFromProbsLazy(q, sampling, s), .q = q };
}

/// `shortlistProposal` for K rows at once: ONE filtered block over the
/// stacked `[K, 32]` logits and ONE categorical (the kernels are launch-bound,
/// so rows are nearly free and a per-row loop is not). Requires one shared
/// sampler (`sampleRowsHomogeneous`) and no seeded row: one categorical
/// cannot carry a per-row key.
pub fn shortlistProposalRows(
    shortlists: []const mtp_mod.Shortlist,
    rows: []const usize,
    sampling: SamplingParams,
    s: mlx.mlx_stream,
) !struct {
    ids: [32]mlx.mlx_array,
    qs: [32]mlx.mlx_array,
} {
    const K = rows.len;
    std.debug.assert(K > 0 and K <= 32);
    const n = mlx.getShape(shortlists[rows[0]].exact)[mlx.getShape(shortlists[rows[0]].exact).len - 1];

    // [K, 32] f32, for the same reason as the single-row proposal.
    var block = mlx.mlx_array_new();
    {
        var parts: [32]mlx.mlx_array = @splat(.{ .ctx = null });
        defer for (parts[0..K]) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        };
        for (rows, 0..) |row, j| {
            parts[j] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&parts[j], shortlists[row].exact, &[_]c_int{ 1, n }, 2, s));
        }
        const vec = mlx.mlx_vector_array_new_data(parts[0..K].ptr, K);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
        errdefer _ = mlx.mlx_array_free(block);
        try mlx.check(mlx.mlx_astype(&block, cat, .float32, s));
    }
    // The suppress mask already applied to the coarse readout.
    var draft = sampling;
    draft.suppress_mask = null;
    const q_block = try filteredProbsRows(block, draft, s);
    defer _ = mlx.mlx_array_free(q_block);

    // ONE categorical over the block -> [K] shortlist indices.
    var picks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picks);
    {
        var logq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(logq);
        try mlx.check(mlx.mlx_log(&logq, q_block, s));
        const null_key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(null_key);
        try mlx.check(mlx.mlx_random_categorical(&picks, logq, -1, null_key, s));
    }

    var ids: [32]mlx.mlx_array = @splat(.{ .ctx = null });
    var qs: [32]mlx.mlx_array = @splat(.{ .ctx = null });
    var done: usize = 0;
    errdefer for (0..done) |j| {
        if (ids[j].ctx != null) _ = mlx.mlx_array_free(ids[j]);
        if (qs[j].ctx != null) _ = mlx.mlx_array_free(qs[j]);
    };
    for (rows, 0..) |row, j| {
        const j_c: c_int = @intCast(j);
        var pick = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pick);
        try mlx.check(mlx.mlx_slice(&pick, picks, &[_]c_int{j_c}, 1, &[_]c_int{j_c + 1}, 1, &[_]c_int{1}, 1, s));
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_take_axis(&picked, shortlists[row].cands, pick, 0, s));
        ids[j] = mlx.mlx_array_new();
        done = j + 1;
        try mlx.check(mlx.mlx_astype(&ids[j], picked, .int32, s));

        var q_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_row);
        try mlx.check(mlx.mlx_slice(&q_row, q_block, &[_]c_int{ j_c, 0 }, 2, &[_]c_int{ j_c + 1, n }, 2, &[_]c_int{ 1, 1 }, 2, s));
        qs[j] = try scatterShortlistQ(shortlists[row], q_row, s);
    }
    return .{ .ids = ids, .qs = qs };
}

/// One row's `[1, 32]` shortlist density as the dense `[1, V]` q the verify
/// takes, zero off the shortlist.
fn scatterShortlistQ(sl: mtp_mod.Shortlist, q32: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const n = mlx.getShape(q32)[mlx.getShape(q32).len - 1];
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, &[_]c_int{ 1, sl.rows }, 2, .float32, s));
    var cand_row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_row);
    try mlx.check(mlx.mlx_reshape(&cand_row, sl.cands, &[_]c_int{ 1, n }, 2, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_put_along_axis(&out, zeros, cand_row, q32, 1, s));
    return out;
}

/// A lazy one-hot `[1, V]` f32 proposal row: the q of an argmax proposal, so
/// `min(1, p/q) == p`. The sampled path's fallback when the coarse head
/// retires mid-chain.
fn lazyOneHotRow(id: mlx.mlx_array, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var indices = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices);
    try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));
    var flat_id = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_id);
    const one = [_]c_int{1};
    try mlx.check(mlx.mlx_reshape(&flat_id, id, &one, 1, s));
    var hit = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hit);
    try mlx.check(mlx.mlx_equal(&hit, indices, flat_id, s));
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, hit, .float32, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const row_shape = [_]c_int{ 1, vocab };
    try mlx.check(mlx.mlx_reshape(&out, as_f32, &row_shape, 2, s));
    return out;
}

/// Sample from the residual distribution `residual = max(target - draft, 0)`,
/// renormalized. Used on stochastic-verify reject so the corrected token
/// preserves the target distribution (per Leviathan et al. speculative
/// decoding paper).
pub fn sampleResidual(target_probs: mlx.mlx_array, draft_probs: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !u32 {
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, target_probs, draft_probs, s));

    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var residual = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(residual);
    try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));

    return sampleFromProbs(residual, sampling, s);
}

// ── tests ──


/// One flat f32 host copy of a lazy array, for the proposal-row assertions.
pub fn proposalTestRead(allocator: std.mem.Allocator, arr: mlx.mlx_array, n: usize, s: mlx.mlx_stream) ![]f32 {
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, arr, .float32, s));
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_contiguous(&dense, as_f32, false, s));
    const flat_shape = [_]c_int{@intCast(n)};
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, dense, &flat_shape, 1, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    const out = try allocator.alloc(f32, n);
    @memcpy(out, mlx.mlx_array_data_float32(flat).?[0..n]);
    return out;
}

test "shortlistProposal: q is zero off the shortlist, sums to 1, and peaks on the best candidate" {
    // Bar: q is a normalized density over the 32 candidates only, and the
    // draft is one of them.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 4096;
    const N: usize = 32;

    var ids: [N]u32 = undefined;
    var logits: [N]f32 = undefined;
    for (0..N) |i| {
        ids[i] = @intCast(37 + i * 101); // spread over the row, none at 0
        logits[i] = @as(f32, @floatFromInt(i)) * 0.25;
    }
    const best_id = ids[N - 1]; // highest logit
    const id_shape = [_]c_int{@intCast(N)};
    const cands = mlx.mlx_array_new_data(&ids, &id_shape, 1, .uint32);
    const exact_shape = [_]c_int{ 1, 1, @intCast(N) };
    const exact = mlx.mlx_array_new_data(&logits, &exact_shape, 3, .float32);
    var sl = mtp_mod.Shortlist{ .cands = cands, .exact = exact, .rows = V };
    defer sl.deinit();

    const sampling = Generator.mtpDraftSamplingFor(.{ .temperature = 1.0 }, false, Generator.MTP_DRAFT_TEMP);
    const prop = try shortlistProposal(sl, sampling, s);
    defer _ = mlx.mlx_array_free(prop.id);
    defer _ = mlx.mlx_array_free(prop.q);

    const q = try proposalTestRead(allocator, prop.q, @intCast(V), s);
    defer allocator.free(q);
    var sum: f64 = 0;
    var nonzero: usize = 0;
    var peak: usize = 0;
    for (q, 0..) |v, i| {
        sum += v;
        if (v != 0) {
            nonzero += 1;
            try testing.expect(std.mem.indexOfScalar(u32, &ids, @intCast(i)) != null);
        }
        if (v > q[peak]) peak = i;
    }
    try testing.expect(nonzero > 0 and nonzero <= N);
    try testing.expect(@abs(sum - 1.0) < 1e-4);
    try testing.expectEqual(best_id, @as(u32, @intCast(peak)));

    // Every draw lands on a candidate — never on a token q calls impossible.
    var draw: usize = 0;
    while (draw < 8) : (draw += 1) {
        const p2 = try shortlistProposal(sl, sampling, s);
        defer _ = mlx.mlx_array_free(p2.id);
        defer _ = mlx.mlx_array_free(p2.q);
        const drawn = try proposalTestRead(allocator, p2.id, 1, s);
        defer allocator.free(drawn);
        const tok: u32 = @intFromFloat(drawn[0]);
        try testing.expect(std.mem.indexOfScalar(u32, &ids, tok) != null);
    }
}

test "shortlistProposalRows: every row's q belongs to that row's own shortlist" {
    // The axis-bug class: one batched filter + one categorical over [K, 32] must
    // stay row-independent. Three rows with DISJOINT candidate ids and different
    // logits, so a cross-row leak or a flattened reduction is visible.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 4096;
    const N: usize = 3;
    const M: usize = 32;

    var ids: [N][M]u32 = undefined;
    var logits: [N][M]f32 = undefined;
    var sl: [N]mtp_mod.Shortlist = undefined;
    for (0..N) |r| {
        for (0..M) |i| {
            ids[r][i] = @intCast(11 + r * 41 + i * 127); // disjoint per row
            // Row r peaks at a different position, so an argmax read off the
            // wrong row lands on the wrong id.
            logits[r][i] = if (i == (r * 7 + 3) % M) 6.0 else @as(f32, @floatFromInt(i)) * 0.05;
        }
        const id_shape = [_]c_int{@intCast(M)};
        const ex_shape = [_]c_int{ 1, 1, @intCast(M) };
        sl[r] = .{
            .cands = mlx.mlx_array_new_data(&ids[r], &id_shape, 1, .uint32),
            .exact = mlx.mlx_array_new_data(&logits[r], &ex_shape, 3, .float32),
            .rows = V,
        };
    }
    defer for (&sl) |*x| x.deinit();

    const sampling = Generator.mtpDraftSamplingFor(.{ .temperature = 1.0 }, false, Generator.MTP_DRAFT_TEMP);
    const rows = [_]usize{ 0, 1, 2 };
    const out = try shortlistProposalRows(&sl, &rows, sampling, s);
    defer for (0..N) |j| {
        _ = mlx.mlx_array_free(out.ids[j]);
        _ = mlx.mlx_array_free(out.qs[j]);
    };

    for (0..N) |r| {
        const q = try proposalTestRead(allocator, out.qs[r], @intCast(V), s);
        defer allocator.free(q);
        var sum: f64 = 0;
        var peak: usize = 0;
        for (q, 0..) |v, i| {
            sum += v;
            if (v != 0) try testing.expect(std.mem.indexOfScalar(u32, &ids[r], @intCast(i)) != null);
            if (v > q[peak]) peak = i;
        }
        try testing.expect(@abs(sum - 1.0) < 1e-4);
        try testing.expectEqual(ids[r][(r * 7 + 3) % M], @as(u32, @intCast(peak)));
        const drawn = try proposalTestRead(allocator, out.ids[r], 1, s);
        defer allocator.free(drawn);
        try testing.expect(std.mem.indexOfScalar(u32, &ids[r], @intFromFloat(drawn[0])) != null);
    }
}

test "a sampled chain whose coarse head retires mid-chain verifies: the argmax fallback drafts a [1] id" {
    // Bar: `draftSelect`'s `[1,1]` id after a sampled `[1]` depth concatenates in both accept graphs.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const p_data = [_]f32{ 0.5, 0.3, 0.2, 0.0, 0.1, 0.6, 0.2, 0.1, 0.0, 0.0, 0.0, 1.0 };
    const probs = mlx.mlx_array_new_data(&p_data, &[_]c_int{ 1, 3, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const one: i32 = 1;
    const d0 = mlx.mlx_array_new_data(&one, &[_]c_int{1}, 1, .int32);
    defer _ = mlx.mlx_array_free(d0);
    const q0_data = [_]f32{ 0.3, 0.5, 0.2, 0.0 };
    const q0 = mlx.mlx_array_new_data(&q0_data, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(q0);
    const fallback = try argmaxStepProposal(mlx.mlx_array_new_data(&one, &[_]c_int{ 1, 1 }, 2, .int32), 4, s);
    defer {
        _ = mlx.mlx_array_free(fallback.id);
        _ = mlx.mlx_array_free(fallback.q);
    }
    for ([_]Generator.MtpGraphFn{ Generator.mtpBatchedExactGraph, Generator.mtpBatchedTypicalGraph }) |graph| {
        var g = try graph(probs, &.{ d0, fallback.id }, &.{ q0, fallback.q }, 2, 0.2, .{}, s);
        defer g.deinit();
        try mlx.check(mlx.mlx_array_eval(g.accept_p));
        const p = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
        try testing.expectEqual(@as(f32, 0.3), p[0]);
        try testing.expectEqual(@as(f32, 0.6), p[1]);
    }
}

test "filteredProbsBlock: every draft row is the SAME density the sample is drawn from" {
    // Exactness rests on q being the true proposal: the draft must be drawn
    // from byte-for-byte the row handed to `specAcceptProb`. The way that
    // breaks silently is filtering the two differently — so pin that a row
    // is a normalized distribution supported ONLY on the kept set, and that
    // `mlx_random_categorical` over log(q) can never land outside it.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 8;
    const M: c_int = 3;
    // Three rows with deliberately different argmaxes, so a row-blind
    // implementation (broadcasting one row's mask) fails.
    const raw = [_]f32{
        5.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0, 0.0,
        0.0, 9.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0,
    };
    const shape = [_]c_int{ 1, M, V };
    const logits = mlx.mlx_array_new_data(&raw, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    var sampling = SamplingParams{};
    sampling.temperature = 0.8;
    sampling.top_k = 2;
    const q = try filteredProbsBlock(logits, sampling, s);
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_array_eval(q));
    const qsh = mlx.getShape(q);
    try testing.expectEqual(M, qsh[0]);
    try testing.expectEqual(V, qsh[1]);

    const data = mlx.mlx_array_data_float32(q) orelse return error.MlxArrayDataNull;
    // Per row: sums to 1, and exactly top_k entries carry mass.
    const want_support = [_][2]usize{ .{ 0, 1 }, .{ 5, 6 }, .{ 1, 3 } };
    for (0..@intCast(M)) |r| {
        var sum: f32 = 0;
        var nonzero: usize = 0;
        for (0..@intCast(V)) |c| {
            const v = data[r * @as(usize, @intCast(V)) + c];
            sum += v;
            if (v > 0) nonzero += 1;
        }
        try testing.expect(@abs(sum - 1.0) < 1e-5);
        try testing.expectEqual(@as(usize, 2), nonzero);
        for (want_support[r]) |c| {
            try testing.expect(data[r * @as(usize, @intCast(V)) + c] > 0);
        }
    }

    // Draw the way nextDflash does and assert every sample is in q's support.
    var logq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logq);
    try mlx.check(mlx.mlx_log(&logq, q, s));
    var draws: usize = 0;
    while (draws < 24) : (draws += 1) {
        const null_key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(null_key);
        var sampled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sampled);
        try mlx.check(mlx.mlx_random_categorical(&sampled, logq, -1, null_key, s));
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_astype(&ids, sampled, .int32, s));
        try mlx.check(mlx.mlx_array_eval(ids));
        const idd = mlx.mlx_array_data_int32(ids) orelse return error.MlxArrayDataNull;
        for (0..@intCast(M)) |r| {
            const tok: usize = @intCast(idd[r]);
            // q(draft) > 0 is what keeps the accept ratio finite.
            try testing.expect(data[r * @as(usize, @intCast(V)) + tok] > 0);
            // And `sliceProbRow` must hand back THAT row, not another.
            const row = try sliceProbRow(q, @intCast(r), s);
            defer _ = mlx.mlx_array_free(row);
            const got = try probAt(row, @intCast(tok), s);
            try testing.expectApproxEqAbs(data[r * @as(usize, @intCast(V)) + tok], got, 1e-6);
        }
    }
    _ = allocator;
}

test "specAcceptProb: full Leviathan ratio, q-clamped" {
    // p <= q: accept with p/q.
    try testing.expect(@abs(Generator.specAcceptProb(0.2, 0.4) - 0.5) < 1e-6);
    // p > q: always accept.
    try testing.expectEqual(@as(f32, 1.0), Generator.specAcceptProb(0.4, 0.2));
    try testing.expectEqual(@as(f32, 1.0), Generator.specAcceptProb(0.4, 0.4));
    // Degenerate q underflow never divides by zero.
    try testing.expectEqual(@as(f32, 1.0), Generator.specAcceptProb(0.5, 0.0));
}

test "spec sampling exactness: draft-from-q + ratio-accept + residual reproduces target p (toy vocab)" {
    // Host-level simulation of the exact per-position algorithm the MTP
    // stochastic round runs: draft ~ q, accept with min(1, p/q), on reject
    // sample from normalize(max(p - q, 0)). The output distribution must
    // equal p (Leviathan/Chen) — this is the invariant the one-hot rule
    // broke for sampled proposals.
    const p = [_]f64{ 0.1, 0.2, 0.3, 0.4 };
    const q = [_]f64{ 0.4, 0.3, 0.2, 0.1 };
    var residual: [4]f64 = undefined;
    var res_sum: f64 = 0;
    for (0..4) |i| {
        residual[i] = @max(p[i] - q[i], 0);
        res_sum += residual[i];
    }
    for (&residual) |*r| r.* /= res_sum;

    var prng = std.Random.DefaultPrng.init(0x5A3E);
    const rnd = prng.random();
    var counts = [_]u64{ 0, 0, 0, 0 };
    const N: usize = 400_000;
    for (0..N) |_| {
        // draft ~ q
        var u = rnd.float(f64);
        var draft: usize = 0;
        var acc: f64 = 0;
        for (q, 0..) |qi, i| {
            acc += qi;
            if (u < acc) {
                draft = i;
                break;
            }
        }
        const a = Generator.specAcceptProb(@floatCast(p[draft]), @floatCast(q[draft]));
        if (rnd.float(f32) < a) {
            counts[draft] += 1;
        } else {
            u = rnd.float(f64);
            acc = 0;
            var res_tok: usize = 3;
            for (residual, 0..) |ri, i| {
                acc += ri;
                if (u < acc) {
                    res_tok = i;
                    break;
                }
            }
            counts[res_tok] += 1;
        }
    }
    for (0..4) |i| {
        const freq = @as(f64, @floatFromInt(counts[i])) / @as(f64, @floatFromInt(N));
        try testing.expect(@abs(freq - p[i]) < 0.01);
    }
}

test "spec sampling exactness: a SHORTLIST proposal (q zero off the top-k) still reproduces target p" {
    // Bar: with q supported on a subset, the marginal is still p, and the
    // off-shortlist tokens arrive only through the residual.
    const p = [_]f64{ 0.30, 0.25, 0.20, 0.15, 0.10 };
    const q = [_]f64{ 0.50, 0.30, 0.20, 0.00, 0.00 }; // shortlist = {0, 1, 2}
    const off_shortlist = [_]usize{ 3, 4 };

    var residual: [5]f64 = undefined;
    var res_sum: f64 = 0;
    for (0..5) |i| {
        residual[i] = @max(p[i] - q[i], 0);
        res_sum += residual[i];
    }
    for (&residual) |*r| r.* /= res_sum;

    var prng = std.Random.DefaultPrng.init(0x51157);
    const rnd = prng.random();
    var counts = [_]u64{ 0, 0, 0, 0, 0 };
    const N: usize = 400_000;
    for (0..N) |_| {
        var u = rnd.float(f64);
        var draft: usize = 0;
        var acc: f64 = 0;
        for (q, 0..) |qi, i| {
            acc += qi;
            if (u < acc) {
                draft = i;
                break;
            }
        }
        // A draft is never drawn off the shortlist, so q[draft] > 0 always.
        try testing.expect(q[draft] > 0);
        if (rnd.float(f32) < Generator.specAcceptProb(@floatCast(p[draft]), @floatCast(q[draft]))) {
            counts[draft] += 1;
        } else {
            u = rnd.float(f64);
            acc = 0;
            var res_tok: usize = 4;
            for (residual, 0..) |ri, i| {
                acc += ri;
                if (u < acc) {
                    res_tok = i;
                    break;
                }
            }
            counts[res_tok] += 1;
        }
    }
    for (0..5) |i| {
        const freq = @as(f64, @floatFromInt(counts[i])) / @as(f64, @floatFromInt(N));
        try testing.expect(@abs(freq - p[i]) < 0.01);
    }
    // The residual is the ONLY route to a token the shortlist missed.
    for (off_shortlist) |i| try testing.expect(counts[i] > 0);
}
