//! Sampling filters: ranks, top-k / top-p / min-p, penalties, argmax.
const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const tokenizer_mod = @import("../../../sdk/tokenizer/tokenizer.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const plugin_driver = @import("../plugin/driver.zig");
const token_mask = @import("../../../sdk/grammar/token_mask.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const testing = std.testing;
const Generator = @import("../generate/generator.zig").Generator;
const SamplingParams = @import("params.zig").SamplingParams;
const SchemaConstraint = @import("../grammar/constraint.zig").SchemaConstraint;
const evalLazyToken = @import("params.zig").evalLazyToken;
const filteredProbsBlock = @import("spec_sampling.zig").filteredProbsBlock;
const generate = @import("../generate/oneshot.zig").generate;
const groupProbsBlock = @import("spec_sampling.zig").groupProbsBlock;
const groupProbsRow = @import("spec_sampling.zig").groupProbsRow;
const maskForLogitVocab = @import("sampler.zig").maskForLogitVocab;
const probsAllPositions = @import("spec_sampling.zig").probsAllPositions;
const sampleRows = @import("lazy.zig").sampleRows;
const sampleRowsHomogeneous = @import("lazy.zig").sampleRowsHomogeneous;
const sampleToken = @import("sampler.zig").sampleToken;
const sampleTokenLazy = @import("lazy.zig").sampleTokenLazy;
const seedKey = @import("lazy.zig").seedKey;


/// Rank cap for the filter helpers. The widest shape a caller passes is
/// `[B, S, V]`; anything past this is a named error, never a wrong axis.
const SAMPLER_MAX_NDIM = 4;

/// `a[..., start..stop]`, every leading axis passed through whole.
fn sliceLastAxis(res: *mlx.mlx_array, a: mlx.mlx_array, start: usize, stop: usize, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(a);
    if (shape.len > SAMPLER_MAX_NDIM) return error.UnsupportedSamplerRank;
    var lo: [SAMPLER_MAX_NDIM]c_int = undefined;
    var hi: [SAMPLER_MAX_NDIM]c_int = undefined;
    var st: [SAMPLER_MAX_NDIM]c_int = undefined;
    for (shape, 0..) |dim, i| {
        const last = i == shape.len - 1;
        lo[i] = if (last) @intCast(start) else 0;
        hi[i] = if (last) @intCast(stop) else dim;
        st[i] = 1;
    }
    const n = shape.len;
    try mlx.check(mlx.mlx_slice(res, a, &lo, n, &hi, n, &st, n, s));
}

/// Column ids of `a` ranked by value DESCENDING, ties by LOWEST ORIGINAL INDEX.
///
/// mlx's merge sort is stable — `ThreadSort` swaps only on a strict less-than and
/// the merge takes from B only when `b < a` — so an ascending argsort of the
/// negated row is exactly that order, and it is the `msv_qsa_select` tie rule.
/// Both filters and both of their routes rank through here, which is what makes a
/// shortlist cutoff agree with the whole row's on a tie-heavy row. Pinned by
/// `sampler ranks break ties by the lowest original index`.
fn ranksDescending(res: *mlx.mlx_array, a: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, a, s));
    try mlx.check(mlx.mlx_argsort_axis(res, neg, -1, s));
}

/// Candidate-chunk width for `topRanksDescending`: cost is one sort of
/// `V / chunk` chunk maxima plus one of `m * chunk` candidates, balanced at
/// sqrt(V / m).
fn shortlistChunkWidth(v: usize, m: usize) usize {
    const bal = @sqrt(@as(f64, @floatFromInt(v)) / @as(f64, @floatFromInt(@max(m, 1))));
    const c: usize = @intFromFloat(@max(16.0, @min(2048.0, bal)));
    return @min(c, v);
}

/// The top `m` of every row — values and their column ids, both in
/// `ranksDescending` order — without ranking the whole row.
///
/// Every element of a chunk is at most that chunk's max, so a chunk whose max is
/// not among the `m` largest chunk maxima holds no top-`m` element: at most `m`
/// chunks can, and ranking their `m * chunk` candidates is exact. On Metal a
/// full-row `mlx_argsort_axis` is a multi-block merge sort over the whole
/// vocabulary — and so are `mlx_topk` and `mlx_argpartition`, which mlx routes to
/// that same sort, so neither of those buys a shortlist.
fn topRanksDescending(
    vals: *mlx.mlx_array,
    ids: *mlx.mlx_array,
    logits: mlx.mlx_array,
    m: u32,
    s: mlx.mlx_stream,
) !void {
    const shape = mlx.getShape(logits);
    if (shape.len == 0 or shape.len > SAMPLER_MAX_NDIM) return error.UnsupportedSamplerRank;
    const v: usize = @intCast(shape[shape.len - 1]);
    const want: usize = @min(@as(usize, m), v);
    if (want == 0) return error.EmptyShortlist;
    const chunk = shortlistChunkWidth(v, want);
    const groups = (v + chunk - 1) / chunk;

    // Nothing left to win once the candidate set is the row itself.
    if (want >= v or want * chunk >= v) {
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try ranksDescending(&order, logits, s);
        try sliceLastAxis(ids, order, 0, want, s);
        return mlx.check(mlx.mlx_take_along_axis(vals, logits, ids.*, -1, s));
    }

    // Pad the last chunk with -inf so the reshape is exact.
    var padded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(padded);
    if (groups * chunk != v) {
        const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg_inf);
        var pad_val = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pad_val);
        try mlx.check(mlx.mlx_astype(&pad_val, neg_inf, mlx.mlx_array_dtype(logits), s));
        const axes = [_]c_int{@intCast(shape.len - 1)};
        const low = [_]c_int{0};
        const high = [_]c_int{@intCast(groups * chunk - v)};
        try mlx.check(mlx.mlx_pad(&padded, logits, &axes, 1, &low, 1, &high, 1, pad_val, "constant", s));
    } else {
        try mlx.check(mlx.mlx_array_set(&padded, logits));
    }

    var dims: [SAMPLER_MAX_NDIM + 1]c_int = undefined;
    @memcpy(dims[0 .. shape.len - 1], shape[0 .. shape.len - 1]);
    const lead = shape.len - 1;
    dims[lead] = @intCast(groups);
    dims[lead + 1] = @intCast(chunk);
    var blk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk);
    try mlx.check(mlx.mlx_reshape(&blk, padded, &dims, lead + 2, s));

    var chunk_max = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chunk_max);
    try mlx.check(mlx.mlx_max_axis(&chunk_max, blk, -1, false, s));

    // Chunks ranked by max, ties by lowest chunk id. A tie group's contributing
    // chunks are a PREFIX of its ascending-id list — a lower id holds a lower
    // column id, which outranks — so taking the lowest ids is what keeps the
    // shortlist equal to the whole row's ranking.
    var chunk_ranks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chunk_ranks);
    try ranksDescending(&chunk_ranks, chunk_max, s);
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try sliceLastAxis(&picked, chunk_ranks, 0, want, s);
    // Ascending chunk ids lay the candidates out in ascending COLUMN id, so the
    // stable rank below breaks candidate ties the way the whole row would.
    var win = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(win);
    try mlx.check(mlx.mlx_sort_axis(&win, picked, -1, s));
    var win_i32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(win_i32);
    try mlx.check(mlx.mlx_astype(&win_i32, win, .int32, s));

    var win_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(win_col);
    try mlx.check(mlx.mlx_expand_dims(&win_col, win_i32, -1, s));
    dims[lead] = @intCast(want);
    var gather_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gather_idx);
    try mlx.check(mlx.mlx_broadcast_to(&gather_idx, win_col, &dims, lead + 2, s));
    var cand_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_2d);
    try mlx.check(mlx.mlx_take_along_axis(&cand_2d, blk, gather_idx, @intCast(lead), s));

    // Column id of every candidate: chunk id * chunk + offset within the chunk.
    const chunk_arr = mlx.mlx_array_new_int(@intCast(chunk));
    defer _ = mlx.mlx_array_free(chunk_arr);
    var base = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(base);
    try mlx.check(mlx.mlx_multiply(&base, win_col, chunk_arr, s));
    var offsets = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(offsets);
    try mlx.check(mlx.mlx_arange(&offsets, 0, @floatFromInt(chunk), 1, .int32, s));
    var cand_ids_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ids_2d);
    try mlx.check(mlx.mlx_add(&cand_ids_2d, base, offsets, s));

    dims[lead] = @intCast(want * chunk);
    var cand_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_vals);
    try mlx.check(mlx.mlx_reshape(&cand_vals, cand_2d, &dims, lead + 1, s));
    var cand_ids_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ids_flat);
    try mlx.check(mlx.mlx_reshape(&cand_ids_flat, cand_ids_2d, &dims, lead + 1, s));
    // A padded tail column is -inf, so it only ranks when the row holds fewer
    // than `want` finite values; clamp so the scatter stays in range regardless.
    const last_col = mlx.mlx_array_new_int(@intCast(v - 1));
    defer _ = mlx.mlx_array_free(last_col);
    var cand_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ids);
    try mlx.check(mlx.mlx_minimum(&cand_ids, cand_ids_flat, last_col, s));

    var cand_ranks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ranks);
    try ranksDescending(&cand_ranks, cand_vals, s);
    var sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel);
    try sliceLastAxis(&sel, cand_ranks, 0, want, s);
    try mlx.check(mlx.mlx_take_along_axis(vals, cand_vals, sel, -1, s));
    try mlx.check(mlx.mlx_take_along_axis(ids, cand_ids, sel, -1, s));
}

/// Apply top-k filtering: keep only the top k logits, set the rest to -inf.
fn applyTopK(res: *mlx.mlx_array, logits: mlx.mlx_array, k: u32, s: mlx.mlx_stream) !void {
    // Per-ROW top-k by RANK: the k highest-ranked column ids, scattered back as a
    // keep mask. A value cutoff (`logits >= kth largest`) keeps every token tied
    // with the k-th, and bf16 logits tie at the top constantly on a 250k vocab —
    // top_k 1 stopped being greedy.
    const shape = mlx.getShape(logits);
    const v: usize = @intCast(shape[shape.len - 1]);
    if (k == 0 or k >= v) return mlx.check(mlx.mlx_array_set(res, logits));
    var vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try topRanksDescending(&vals, &ids, logits, k, s);
    try keepByIndex(res, logits, ids, s);
}

/// Mask `logits` to -inf everywhere except the per-row `keep_idx` columns.
fn keepByIndex(res: *mlx.mlx_array, logits: mlx.mlx_array, keep_idx: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, shape.ptr, shape.len, .bool_, s));
    const t = mlx.mlx_array_new_bool(true);
    defer _ = mlx.mlx_array_free(t);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_put_along_axis(&mask, zeros, keep_idx, t, -1, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask, logits, neg_inf, s));
}

/// Top-k and top-p in ONE pass. A bounded `applyTopP` ranks the row's top
/// `top_k` columns and scatters its keep flags at exactly those column ids, so a
/// preceding `applyTopK` selects the same shortlist and masks a SUPERSET of what
/// the nucleus keeps — one full-vocab selection and one full-row `where` of pure
/// tax on every filtered row. Byte-identical to the pair either way round.
pub fn filterTopKTopP(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, top_k: u32, s: mlx.mlx_stream) !void {
    if (top_p >= 1.0) return applyTopK(res, logits, top_k, s);
    return applyTopP(res, logits, top_p, top_k, s);
}

pub fn applyMinP(res: *mlx.mlx_array, logits: mlx.mlx_array, min_p: f32, s: mlx.mlx_stream) !void {
    var max_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(max_logits);
    try mlx.check(mlx.mlx_max_axis(&max_logits, logits, -1, true, s));
    const log_min_p = mlx.mlx_array_new_float(@log(min_p));
    defer _ = mlx.mlx_array_free(log_min_p);
    var threshold = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(threshold);
    try mlx.check(mlx.mlx_add(&threshold, max_logits, log_min_p, s));
    var keep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep);
    try mlx.check(mlx.mlx_greater_equal(&keep, logits, threshold, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, keep, logits, neg_inf, s));
}

/// Apply top-p (nucleus) sampling: mask logits outside the top-p probability
/// mass. The nucleus is decided over ranked values and scattered back by column
/// id, so a token tied with the cutoff value but outside the mass is masked
/// (top_p -> 0 is greedy even on tied bf16 logits).
///
/// `nucleus_bound` is the caller's top-k. With top-k applied first the row holds
/// exactly `k` finite columns, so the nucleus cannot reach past the k-shortlist
/// and the cutoff is exact without ranking the row; 0 means the row is unfiltered
/// and the nucleus is unbounded.
fn applyTopP(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, nucleus_bound: u32, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const v: usize = @intCast(shape[shape.len - 1]);
    const bounded = nucleus_bound > 0 and nucleus_bound < v;

    var vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    if (bounded) {
        try topRanksDescending(&vals, &ids, logits, nucleus_bound, s);
    } else {
        try ranksDescending(&ids, logits, s);
        try mlx.check(mlx.mlx_take_along_axis(&vals, logits, ids, -1, s));
    }

    // f32 throughout: mlx instantiates cumsum at the INPUT dtype and a bf16
    // accumulator loses nucleus tail mass. The ranked values ARE the row's
    // finite entries (top-k left exactly k), so their softmax is the row's.
    var vals_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals_f32);
    try mlx.check(mlx.mlx_astype(&vals_f32, vals, .float32, s));
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, vals_f32, -1, true, s));

    // Mass STRICTLY above each rank — an exclusive scan down the ranking, so the
    // term is a function of the shortlist alone (everything outranking a top-k
    // column is itself top-k). Rank 0 sees 0, so the argmax is always kept —
    // the threshold floors above 0 so a literal top_p 0 is greedy, not empty.
    var above = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(above);
    try mlx.check(mlx.mlx_cumsum(&above, probs, -1, false, false, s));
    const threshold = mlx.mlx_array_new_float(@max(top_p, std.math.floatMin(f32)));
    defer _ = mlx.mlx_array_free(threshold);
    var in_nucleus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(in_nucleus);
    try mlx.check(mlx.mlx_less(&in_nucleus, above, threshold, s));

    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, shape.ptr, shape.len, .bool_, s));
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_put_along_axis(&mask, zeros, ids, in_nucleus, -1, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask, logits, neg_inf, s));
}

/// Penalties over the tokens generated so far: `repeat_penalty` divides a positive logit and
/// multiplies a negative one; `presence_penalty` is subtracted once per seen id and
/// `frequency_penalty` once per occurrence (OpenAI's definition).
pub fn applyPenalties(res: *mlx.mlx_array, logits: mlx.mlx_array, generated_ids: []const u32, sampling: SamplingParams, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);
    const arr_shape = [_]c_int{ 1, @intCast(vocab_size) };

    // Occurrences per id; reused below for what each id loses.
    const table = try std.heap.page_allocator.alloc(f32, vocab_size);
    defer std.heap.page_allocator.free(table);
    @memset(table, 0);
    for (generated_ids) |id| if (id < vocab_size) {
        table[id] += 1;
    };

    var current = logits;
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    if (sampling.repeat_penalty != 1.0) {
        const counts = mlx.mlx_array_new_data(table.ptr, &arr_shape, 2, .float32);
        defer _ = mlx.mlx_array_free(counts);
        const rp = mlx.mlx_array_new_float(sampling.repeat_penalty);
        defer _ = mlx.mlx_array_free(rp);
        const inv_rp = mlx.mlx_array_new_float(1.0 / sampling.repeat_penalty);
        defer _ = mlx.mlx_array_free(inv_rp);
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);

        var seen = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(seen);
        try mlx.check(mlx.mlx_greater(&seen, counts, zero, s));
        var positive = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(positive);
        try mlx.check(mlx.mlx_greater(&positive, logits, zero, s));
        var pen_pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_pos);
        try mlx.check(mlx.mlx_multiply(&pen_pos, logits, inv_rp, s));
        var pen_neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_neg);
        try mlx.check(mlx.mlx_multiply(&pen_neg, logits, rp, s));
        var sign_selected = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sign_selected);
        try mlx.check(mlx.mlx_where(&sign_selected, positive, pen_pos, pen_neg, s));
        try mlx.check(mlx.mlx_where(&scaled, seen, sign_selected, logits, s));
        current = scaled;
    }

    if (sampling.presence_penalty == 0.0 and sampling.frequency_penalty == 0.0) {
        try mlx.check(mlx.mlx_copy(res, current, s));
        return;
    }
    for (table) |*v| if (v.* > 0) {
        v.* = sampling.presence_penalty + sampling.frequency_penalty * v.*;
    };
    const subtract = mlx.mlx_array_new_data(table.ptr, &arr_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(subtract);
    try mlx.check(mlx.mlx_subtract(res, current, subtract, s));
}

/// Greedy argmax over the last axis.
pub fn argmax(last_logits: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var argmax_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(argmax_arr);
    try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, last_logits, -1, false, s));

    try mlx.check(mlx.mlx_array_eval(argmax_arr));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, argmax_arr));

    return @intCast(val);
}

// ── tests ──


test "frequency_penalty subtracts per occurrence, presence_penalty once, and neither scales a logit" {
    const s = mlx.gpuStream();
    var data = [_]f32{ 2, 2, 2, 2, 2, -2, 2, 2 };
    const logits = mlx.mlx_array_new_data(&data, &[_]c_int{ 1, 8 }, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try applyPenalties(&out, logits, &.{ 3, 3, 5 }, .{ .frequency_penalty = 0.5, .presence_penalty = 0.25 }, s);
    try mlx.check(mlx.mlx_array_eval(out));
    const got = mlx.mlx_array_data_float32(out).?[0..8];
    try testing.expectEqualSlices(f32, &.{ 2, 2, 2, 0.75, 2, -2.75, 2, 2 }, got);
    try testing.expect((SamplingParams{ .frequency_penalty = 0.5 }).penalized());
}

test "maskForLogitVocab pads and truncates to logits size" {
    const short_mask = [_]bool{ true, false, true };
    const padded = try maskForLogitVocab(testing.allocator, &short_mask, 5);
    defer padded.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 5), padded.slice.len);
    try testing.expect(padded.slice[0]);
    try testing.expect(!padded.slice[1]);
    try testing.expect(padded.slice[2]);
    try testing.expect(!padded.slice[3]);
    try testing.expect(!padded.slice[4]);

    const long_mask = [_]bool{ false, true, true, true };
    const truncated = try maskForLogitVocab(testing.allocator, &long_mask, 2);
    defer truncated.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), truncated.slice.len);
    try testing.expect(!truncated.slice[0]);
    try testing.expect(truncated.slice[1]);
}

test "argmax selects highest value" {
    // Create a simple logits array [1, 5] with values [0.1, 0.5, 0.9, 0.2, 0.3]
    const data = [_]f32{ 0.1, 0.5, 0.9, 0.2, 0.3 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 2), result); // index 2 has value 0.9
}

test "argmax with negative values" {
    const data = [_]f32{ -5.0, -1.0, -3.0, -0.5, -2.0 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 3), result); // index 3 has value -0.5 (highest)
}

test "applyPenalties reduces seen token logits" {
    const s = mlx.gpuStream();
    // logits: [1.0, 2.0, 3.0, 4.0, 5.0]
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const shape = [_]c_int{ 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize tokens at indices 1 and 3
    const generated = [_]u32{ 1, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyPenalties(&res, logits, &generated, .{ .repeat_penalty = 2.0 }, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: untouched → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[1], 0.01);
    // Index 2: untouched → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
    // Index 4: untouched → 5.0
    try testing.expectApproxEqAbs(@as(f32, 5.0), res_data[4], 0.01);
}

test "applyPenalties with negative logits" {
    const s = mlx.gpuStream();
    // Mix of positive and negative logits
    const data = [_]f32{ -2.0, 3.0, -1.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize all tokens
    const generated = [_]u32{ 0, 1, 2, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyPenalties(&res, logits, &generated, .{ .repeat_penalty = 2.0 }, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: negative, multiplied by 2.0 → -4.0
    try testing.expectApproxEqAbs(@as(f32, -4.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.5
    try testing.expectApproxEqAbs(@as(f32, 1.5), res_data[1], 0.01);
    // Index 2: negative, multiplied by 2.0 → -2.0
    try testing.expectApproxEqAbs(@as(f32, -2.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
}

test "applyPenalties presence penalty" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 2 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyPenalties(&res, logits, &generated, .{ .presence_penalty = 0.5 }, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: seen, presence penalty subtracted → 1.0 - 0.5 = 0.5
    try testing.expectApproxEqAbs(@as(f32, 0.5), res_data[0], 0.01);
    // Index 1: unseen → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[1], 0.01);
    // Index 2: seen → 3.0 - 0.5 = 2.5
    try testing.expectApproxEqAbs(@as(f32, 2.5), res_data[2], 0.01);
    // Index 3: unseen → 4.0
    try testing.expectApproxEqAbs(@as(f32, 4.0), res_data[3], 0.01);
}

test "applyPenalties combined penalties" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 2.0, -1.0, 3.0 };
    const shape = [_]c_int{ 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 1 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyPenalties(&res, logits, &generated, .{ .repeat_penalty = 2.0, .presence_penalty = 1.0 }, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: positive, divide by 2.0 = 1.0, then - 1.0 = 0.0
    try testing.expectApproxEqAbs(@as(f32, 0.0), res_data[0], 0.01);
    // Index 1: negative, multiply by 2.0 = -2.0, then - 1.0 = -3.0
    try testing.expectApproxEqAbs(@as(f32, -3.0), res_data[1], 0.01);
    // Index 2: unseen → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
}

test "min_p keeps only eligible tokens in scalar, lazy and batched sampling" {
    const s = mlx.gpuStream();
    const raw = [_]f32{ 3, 2, 0, -5, -5, 0, 2, 3 };
    const logits = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 2, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const strict = SamplingParams{ .temperature = 1, .top_k = 2, .min_p = 0.5 };
    const probs = try probsAllPositions(logits, strict, s);
    defer _ = mlx.mlx_array_free(probs);
    const values = try samplerTestReadFlat(testing.allocator, probs, raw.len, s);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(f32, &.{ 1, 0, 0, 0, 0, 0, 0, 1 }, values);

    const first = mlx.mlx_array_new_data(&raw[0], &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(first);
    const second = mlx.mlx_array_new_data(&raw[4], &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(second);
    try testing.expectEqual(@as(u32, 0), (try sampleToken(testing.allocator, first, strict, null, 0, s)).token_id);
    try testing.expectEqual(@as(u32, 0), try evalLazyToken(sampleTokenLazy(first, strict, s)));
    var params = [_]SamplingParams{ strict, strict };
    var ids: [2]i32 = undefined;
    try sampleRows(&ids, &.{ first, second }, &params, s);
    try testing.expectEqualSlices(i32, &.{ 0, 3 }, &ids);
    try testing.expect(!sampleRowsHomogeneous(&.{ strict, .{ .temperature = 1, .top_k = 2, .min_p = 0 } }));
    try testing.expect(sampleRowsHomogeneous(&.{ SamplingParams{ .temperature = 1, .top_k = 2 }, .{ .temperature = 1, .top_k = 2, .min_p = 0 } }));
}

test "min_p keeps tied maxima and zero matches the omitted filter" {
    const s = mlx.gpuStream();
    const tied = mlx.mlx_array_new_data(&[_]f32{ 0, 0, -1, -2 }, &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(tied);
    const strict = try probsAllPositions(tied, .{ .min_p = 1 }, s);
    defer _ = mlx.mlx_array_free(strict);
    const values = try samplerTestReadFlat(testing.allocator, strict, 4, s);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(f32, &.{ 0.5, 0.5, 0, 0 }, values);

    const omitted = try probsAllPositions(tied, .{}, s);
    defer _ = mlx.mlx_array_free(omitted);
    const zero = try probsAllPositions(tied, .{ .min_p = 0 }, s);
    defer _ = mlx.mlx_array_free(zero);
    try samplerTestExpectSameBits(omitted, zero, 4, s);
}

test "min_p uses temperature and agrees across speculative target and proposal" {
    const s = mlx.gpuStream();
    const raw = [_]f32{ 3, 2.8, 0, -5, -5, 0, 2.8, 3 };
    const logits = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 2, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const warm = try probsAllPositions(logits, .{ .temperature = 2, .min_p = 0.9 }, s);
    defer _ = mlx.mlx_array_free(warm);
    const warm_values = try samplerTestReadFlat(testing.allocator, warm, raw.len, s);
    defer testing.allocator.free(warm_values);
    try testing.expect(warm_values[0] > 0 and warm_values[1] > 0);
    try testing.expectEqual(@as(f32, 0), warm_values[2]);
    const cold = try probsAllPositions(logits, .{ .temperature = 1, .min_p = 0.9 }, s);
    defer _ = mlx.mlx_array_free(cold);
    const cold_values = try samplerTestReadFlat(testing.allocator, cold, raw.len, s);
    defer testing.allocator.free(cold_values);
    try testing.expectEqual(@as(f32, 0), cold_values[1]);

    const policy = SamplingParams{ .temperature = 2, .top_k = 3, .top_p = 0.9, .min_p = 0.5 };
    const target = try probsAllPositions(logits, policy, s);
    defer _ = mlx.mlx_array_free(target);
    const grouped = try groupProbsBlock(logits, policy, s);
    defer _ = mlx.mlx_array_free(grouped);
    const proposal = try filteredProbsBlock(logits, policy, s);
    defer _ = mlx.mlx_array_free(proposal);
    try samplerTestExpectSameBits(target, grouped, raw.len, s);
    try samplerTestExpectSameBits(target, proposal, raw.len, s);
}

test "top-k and top-p break exact ties by rank, not by value" {
    // bf16 logits tie at the top constantly on a 250k vocab; a value cutoff
    // keeps every tied token and top_k 1 / top_p -> 0 stop being greedy.
    const s = mlx.gpuStream();
    const data = [_]f32{
        2.0, 5.0, 5.0, 5.0, 1.0,
        0.1, 0.2, 9.0, 0.3, 0.4,
    };
    const shape = [_]c_int{ 2, 5 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    for ([_]bool{ true, false }) |use_k| {
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        if (use_k) try applyTopK(&masked, logits, 1, s) else try applyTopP(&masked, logits, 1e-6, 0, s);
        try mlx.check(mlx.mlx_array_eval(masked));
        const out = mlx.mlx_array_data_float32(masked).?[0..10];
        var kept0: usize = 0;
        for (out[0..5]) |v| kept0 += @intFromBool(v != -std.math.inf(f32));
        try testing.expectEqual(@as(usize, 1), kept0);
        for (out[5..10], 0..) |v, i| try testing.expect((v != -std.math.inf(f32)) == (i == 2));
    }
    // A 3-of-5 top-k on the tied row keeps exactly the three tied tokens.
    var masked3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked3);
    try applyTopK(&masked3, logits, 3, s);
    try mlx.check(mlx.mlx_array_eval(masked3));
    const out3 = mlx.mlx_array_data_float32(masked3).?[0..5];
    for (out3, 0..) |v, i| try testing.expect((v != -std.math.inf(f32)) == (i >= 1 and i <= 3));
}

/// Deterministic Gaussian logits (Box-Muller over an xorshift stream): diffuse
/// enough that the nucleus for top_p 0.95 spans thousands of tokens, which is
/// where a narrow cumulative accumulator loses the tail.
fn samplerTestRow(host: []f32, seed: u64, sigma: f32) void {
    var rng: u64 = seed | 1;
    var i: usize = 0;
    while (i < host.len) : (i += 2) {
        var u: [2]f64 = undefined;
        for (&u) |*d| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            d.* = (@as(f64, @floatFromInt(rng >> 11)) + 1.0) / 9007199254740994.0;
        }
        const r = @sqrt(-2.0 * @log(u[0]));
        const a = 6.283185307179586 * u[1];
        host[i] = @floatCast(r * @cos(a) * @as(f64, sigma));
        if (i + 1 < host.len) host[i + 1] = @floatCast(r * @sin(a) * @as(f64, sigma));
    }
}

/// Nucleus the sampler is meant to keep, in f64: ranked descending, a column is
/// in when less than top_p of the mass outranks it.
fn samplerTestExactKeep(allocator: std.mem.Allocator, row: []const f32, top_p: f32) !usize {
    const sorted = try allocator.alloc(f64, row.len);
    defer allocator.free(sorted);
    for (row, sorted) |x, *d| d.* = @floatCast(x);
    std.mem.sort(f64, sorted, {}, comptime std.sort.desc(f64));
    const mx = sorted[0];
    var z: f64 = 0;
    for (sorted) |d| z += @exp(d - mx);
    var above: f64 = 0;
    var keep: usize = 0;
    for (sorted) |d| {
        if (above >= @as(f64, @floatCast(top_p))) break;
        above += @exp(d - mx) / z;
        keep += 1;
    }
    return keep;
}

/// Read an mlx array back as f32 host values. A slice is a strided VIEW, so the
/// read goes through a contiguous copy and a flat reshape, never the base
/// pointer.
pub fn samplerTestReadFlat(allocator: std.mem.Allocator, arr: mlx.mlx_array, n: usize, s: mlx.mlx_stream) ![]f32 {
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

/// Bit equality of two mlx arrays read back as f32: both routes end in one
/// `where` over the same logits, so agreeing on the kept columns is agreeing on
/// every byte.
fn samplerTestExpectSameBits(a: mlx.mlx_array, b: mlx.mlx_array, n: usize, s: mlx.mlx_stream) !void {
    const ad = try samplerTestReadFlat(testing.allocator, a, n, s);
    defer testing.allocator.free(ad);
    const bd = try samplerTestReadFlat(testing.allocator, b, n, s);
    defer testing.allocator.free(bd);
    for (ad, bd, 0..) |x, y, i| {
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
            std.debug.print("mismatch at {d}: {d} vs {d}\n", .{ i, x, y });
            return error.FilteredLogitsDiffer;
        }
    }
}

fn samplerTestKeptCount(row: []const f32) usize {
    var n: usize = 0;
    for (row) |x| {
        if (x != -std.math.inf(f32)) n += 1;
    }
    return n;
}

test "sampler ranks break ties by the lowest original index" {
    // The contract both filters and both of their routes share. mlx's stable
    // merge sort is what delivers it; if a bump breaks stability this fails.
    const s = mlx.gpuStream();
    const data = [_]f32{ 2.0, 5.0, 5.0, 5.0, 1.0, 5.0 };
    const shape = [_]c_int{ 1, 6 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    var ranks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ranks);
    try ranksDescending(&ranks, logits, s);
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, ranks, .float32, s));
    const got = try samplerTestReadFlat(testing.allocator, as_f32, 6, s);
    defer testing.allocator.free(got);
    // The four tied 5.0s come first in ascending column order, then 2.0, then 1.0.
    try testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 5, 0, 4 }, got);
}

test "applyTopP keeps the exact nucleus on a bf16 vocab row" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 248320;
    const host = try allocator.alloc(f32, v);
    defer allocator.free(host);
    const rounded = try allocator.alloc(f32, v);
    defer allocator.free(rounded);

    for ([_]u64{ 1, 2, 3, 4, 5, 6 }) |seed| {
        samplerTestRow(host, seed *% 0x9E3779B97F4A7C15, 4.0);
        const shape = [_]c_int{ 1, @intCast(v) };
        const f32_row = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(f32_row);
        var bf16_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bf16_row);
        try mlx.check(mlx.mlx_astype(&bf16_row, f32_row, .bfloat16, s));

        // The reference reads the row the kernel sees: bf16-rounded values.
        var rt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rt);
        try mlx.check(mlx.mlx_astype(&rt, bf16_row, .float32, s));
        const rt_host = try samplerTestReadFlat(allocator, rt, v, s);
        defer allocator.free(rt_host);
        @memcpy(rounded, rt_host);
        const want = try samplerTestExactKeep(allocator, rounded, 0.95);

        var res = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res);
        try applyTopP(&res, bf16_row, 0.95, 0, s);
        const res_host = try samplerTestReadFlat(allocator, res, v, s);
        defer allocator.free(res_host);
        // The boundary column's own probability is ~1e-5 here while the f32 scan
        // carries ~1e-7 of error, so the exclusive mass can cross top_p one
        // column early or late against the f64 reference. The defect this pins
        // was 148 columns wide.
        const got = samplerTestKeptCount(res_host);
        const diff = @as(i64, @intCast(want)) - @as(i64, @intCast(got));
        try testing.expect(diff >= -1 and diff <= 1);
    }
}

test "applyTopP at top_p 0 keeps exactly the argmax" {
    // The nucleus is "mass strictly above the rank < top_p"; rank 0 sees 0, so a
    // literal 0 kept nothing and the sampler drew uniformly from the vocabulary.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 8192;
    const host = try allocator.alloc(f32, v);
    defer allocator.free(host);
    samplerTestRow(host, 0x9E3779B97F4A7C15, 4.0);
    const shape = [_]c_int{ 1, @intCast(v) };
    const row = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(row);
    for ([_]u32{ 0, 40 }) |bound| {
        var res = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res);
        try applyTopP(&res, row, 0.0, bound, s);
        const res_host = try samplerTestReadFlat(allocator, res, v, s);
        defer allocator.free(res_host);
        try testing.expectEqual(@as(usize, 1), samplerTestKeptCount(res_host));
        var best: usize = 0;
        for (host, 0..) |x, i| if (x > host[best]) {
            best = i;
        };
        try testing.expect(std.math.isFinite(res_host[best]));
    }
}

test "topRanksDescending returns the exact top m values and column ids" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    for ([_]usize{ 8192, 248320 }) |v| {
        const rows: usize = 3;
        const host = try allocator.alloc(f32, rows * v);
        defer allocator.free(host);
        for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 1) *% 0xD1B54A32D192ED03, 4.0);
        // Row 2 quantized to 0.25 steps: thousands of exact ties.
        for (host[2 * v ..][0..v]) |*x| x.* = @round(x.* * 4.0) / 4.0;
        const shape = [_]c_int{ @intCast(rows), @intCast(v) };
        const block = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(block);

        for ([_]u32{ 1, 20, 257 }) |m| {
            var vals = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vals);
            var ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ids);
            try topRanksDescending(&vals, &ids, block, m, s);
            // The whole row's ranking, restricted to its first m: the bar the
            // shortlist has to reproduce, ties included.
            var order = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(order);
            try ranksDescending(&order, block, s);
            var ref_ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref_ids);
            try sliceLastAxis(&ref_ids, order, 0, m, s);
            var ref_vals = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref_vals);
            try mlx.check(mlx.mlx_take_along_axis(&ref_vals, block, ref_ids, -1, s));
            var ids_f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ids_f);
            try mlx.check(mlx.mlx_astype(&ids_f, ids, .float32, s));
            var ref_ids_f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref_ids_f);
            try mlx.check(mlx.mlx_astype(&ref_ids_f, ref_ids, .float32, s));
            try samplerTestExpectSameBits(vals, ref_vals, rows * m, s);
            try samplerTestExpectSameBits(ids_f, ref_ids_f, rows * m, s);
        }
    }
}

test "applyTopK keeps exactly k columns, the highest ranked ones" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 248320;
    const rows: usize = 3;
    const host = try allocator.alloc(f32, rows * v);
    defer allocator.free(host);
    for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 7) *% 0xD1B54A32D192ED03, 4.0);
    // Row 1 quantized to 1.0 steps: a value cutoff would keep thousands here.
    for (host[v..][0..v]) |*x| x.* = @round(x.*);
    const shape = [_]c_int{ @intCast(rows), @intCast(v) };
    const block = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(block);

    for ([_]u32{ 1, 20, 64 }) |k| {
        var res = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res);
        try applyTopK(&res, block, k, s);
        const rd = try samplerTestReadFlat(allocator, res, rows * v, s);
        defer allocator.free(rd);
        for (0..rows) |r| {
            try testing.expectEqual(@as(usize, k), samplerTestKeptCount(rd[r * v ..][0..v]));
        }
        // Same columns as ranking the whole row.
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try ranksDescending(&order, block, s);
        var ref_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref_ids);
        try sliceLastAxis(&ref_ids, order, 0, k, s);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try keepByIndex(&ref, block, ref_ids, s);
        try samplerTestExpectSameBits(res, ref, rows * v, s);
    }
}

test "shortlist top-p equals the whole row's ranking, ties and 248320-wide rows" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    for ([_]usize{ 8192, 248320 }) |v| {
        const rows: usize = 4;
        const host = try allocator.alloc(f32, rows * v);
        defer allocator.free(host);
        for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 11) *% 0xD1B54A32D192ED03, 4.0);
        // Row 2: 0.25-step ties. Row 3: every column identical — the extreme the
        // rank contract exists for.
        for (host[2 * v ..][0..v]) |*x| x.* = @round(x.* * 4.0) / 4.0;
        for (host[3 * v ..][0..v]) |*x| x.* = 1.5;

        const shape = [_]c_int{ @intCast(rows), @intCast(v) };
        const block = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(block);

        for ([_]struct { p: f32, k: u32 }{
            .{ .p = 0.95, .k = 20 },
            .{ .p = 0.8, .k = 50 },
            .{ .p = 0.99, .k = 5 },
            .{ .p = 0.5, .k = 1 },
            .{ .p = 0.95, .k = 257 },
            .{ .p = 1e-6, .k = 20 },
        }) |cfg| {
            var masked = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(masked);
            try applyTopK(&masked, block, cfg.k, s);

            var short = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(short);
            try applyTopP(&short, masked, cfg.p, cfg.k, s);
            var full = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(full);
            try applyTopP(&full, masked, cfg.p, 0, s);
            try samplerTestExpectSameBits(short, full, rows * v, s);

            // Same filtered logits and the same key must draw the same token.
            for (0..8) |d| {
                const key = seedKey(.{ .seed = 12345, .draw = @intCast(d) });
                defer _ = mlx.mlx_array_free(key);
                var a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(a);
                var b = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(b);
                try mlx.check(mlx.mlx_random_categorical(&a, short, -1, key, s));
                try mlx.check(mlx.mlx_random_categorical(&b, full, -1, key, s));
                try samplerTestExpectSameBits(a, b, rows, s);
            }
        }
    }
}

test "shortlist top-p holds on rank-3 [B, L, V] blocks" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 8192;
    const rows: usize = 3;
    const host = try allocator.alloc(f32, rows * v);
    defer allocator.free(host);
    for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 23) *% 0xD1B54A32D192ED03, 4.0);
    const shape = [_]c_int{ 1, @intCast(rows), @intCast(v) };
    const block = mlx.mlx_array_new_data(host.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(block);

    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try applyTopK(&masked, block, 20, s);
    var short = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(short);
    try applyTopP(&short, masked, 0.95, 20, s);
    var full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(full);
    try applyTopP(&full, masked, 0.95, 0, s);
    try samplerTestExpectSameBits(short, full, rows * v, s);

    // Per-ROW cutoffs: a block-wide filter would mask whole rows to -inf.
    const sd = try samplerTestReadFlat(allocator, short, rows * v, s);
    defer allocator.free(sd);
    for (0..rows) |r| {
        const kept = samplerTestKeptCount(sd[r * v ..][0..v]);
        try testing.expect(kept > 0 and kept <= 20);
    }
}

test "a group's joined probability block slices row-for-row into the per-row filter" {
    // Ragged widths, three rows: the axis-bug class ("one row cannot reveal an
    // axis bug"), and the offsets the projection's own walk produces.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 8192;
    const widths = [_]u32{ 3, 2, 4 };
    const sampling = SamplingParams{ .temperature = 1.0, .top_p = 0.95, .top_k = 20 };

    var rows: [widths.len]mlx.mlx_array = undefined;
    var built: usize = 0;
    defer for (rows[0..built]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (widths, 0..) |w, r| {
        const host = try allocator.alloc(f32, @as(usize, w) * v);
        defer allocator.free(host);
        for (0..w) |p| samplerTestRow(host[p * v ..][0..v], (r * 8 + p + 1) *% 0xD1B54A32D192ED03, 4.0);
        const shape = [_]c_int{ 1, @intCast(w), @intCast(v) };
        const f32_rows = mlx.mlx_array_new_data(host.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(f32_rows);
        rows[r] = mlx.mlx_array_new();
        built = r + 1;
        // bf16 like the verify block the lm_head produces.
        try mlx.check(mlx.mlx_astype(&rows[r], f32_rows, .bfloat16, s));
    }

    var joined = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(joined);
    {
        const vec = mlx.mlx_vector_array_new_data(&rows, widths.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 1, s));
    }
    const block = try groupProbsBlock(joined, sampling, s);
    defer _ = mlx.mlx_array_free(block);

    var offset: c_int = 0;
    for (widths, 0..) |w, r| {
        const got = try groupProbsRow(block, offset, w, s);
        defer _ = mlx.mlx_array_free(got);
        const want = try probsAllPositions(rows[r], sampling, s);
        defer _ = mlx.mlx_array_free(want);
        const g_shape = mlx.getShape(got);
        try testing.expectEqual(@as(c_int, 1), g_shape[0]);
        try testing.expectEqual(@as(c_int, @intCast(w)), g_shape[1]);
        try samplerTestExpectSameBits(got, want, @as(usize, w) * v, s);
        offset += @intCast(w);
    }
}

test "the group accept block declines a mixed sampler and any seeded row" {
    const a = SamplingParams{ .temperature = 1.0, .top_p = 0.95, .top_k = 20 };
    var b = a;
    b.top_k = 40;
    var seeded = a;
    seeded.seed = 7;
    try testing.expect(Generator.mtpGroupAcceptBatchable(&.{ a, a }));
    try testing.expect(!Generator.mtpGroupAcceptBatchable(&.{ a, b }));
    try testing.expect(!Generator.mtpGroupAcceptBatchable(&.{ a, seeded }));
    try testing.expect(!Generator.mtpGroupAcceptBatchable(&.{}));
}

/// The composition every sampling site ran before `filterTopKTopP`: top-k, then
/// top-p bounded by that same k. The reference the one-pass filter is pinned to.
fn samplerTestComposed(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, top_k: u32, s: mlx.mlx_stream) !void {
    var after_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(after_k);
    if (top_k > 0) {
        try applyTopK(&after_k, logits, top_k, s);
    } else {
        try mlx.check(mlx.mlx_array_set(&after_k, logits));
    }
    if (top_p < 1.0) return applyTopP(res, after_k, top_p, top_k, s);
    return mlx.check(mlx.mlx_array_set(res, after_k));
}

test "filterTopKTopP is byte-identical to the two-pass composition" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const Setting = struct { p: f32, k: u32 };
    const settings = [_]Setting{
        .{ .p = 0.95, .k = 20 }, .{ .p = 0.9, .k = 1 },  .{ .p = 1.0, .k = 20 },
        .{ .p = 0.95, .k = 0 },  .{ .p = 0.5, .k = 50 }, .{ .p = 0.95, .k = 999999 },
    };
    for ([_]usize{ 8192, 248320 }) |v| {
        const rows: usize = 3;
        const host = try allocator.alloc(f32, rows * v);
        defer allocator.free(host);
        samplerTestRow(host[0..v], 0x9E3779B97F4A7C15, 4.0);
        // Row 1 quantized to 0.25 steps: thousands of exact ties at the cutoff.
        samplerTestRow(host[v..][0..v], 0xD1B54A32D192ED03, 4.0);
        for (host[v..][0..v]) |*x| x.* = @round(x.* * 4.0) / 4.0;
        // Row 2 all equal: every column ties, so the ranking alone decides.
        @memset(host[2 * v ..][0..v], 1.25);

        // `[m, V]` and `[1, L, V]`: the mask must stay per ROW on both ranks.
        for ([_]usize{ 2, 3 }) |rank| {
            const shape2 = [_]c_int{ @intCast(rows), @intCast(v) };
            const shape3 = [_]c_int{ 1, @intCast(rows), @intCast(v) };
            const shape: []const c_int = if (rank == 2) &shape2 else &shape3;
            const f32_block = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f32_block);
            var block = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(block);
            try mlx.check(mlx.mlx_astype(&block, f32_block, .bfloat16, s));
            for (settings) |cfg| {
                var want = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(want);
                try samplerTestComposed(&want, block, cfg.p, cfg.k, s);
                var got = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(got);
                try filterTopKTopP(&got, block, cfg.p, cfg.k, s);
                samplerTestExpectSameBits(got, want, rows * v, s) catch |e| {
                    std.debug.print("v={d} rank={d} top_p={d} top_k={d}\n", .{ v, rank, cfg.p, cfg.k });
                    return e;
                };
            }
        }
    }
}

test "repeat penalty shapes every sampling path, not only the logprobs one" {
    // Bar: a penalty changes greedy output with or without logprobs, the same way, and under a grammar.
    const path = env.get(.LOGPROBS_TEST_MODEL) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.mem.span(path);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    var xfm = try plugin_driver.load(io, a, &config, dir, mlx.gpuStream());
    defer xfm.deinit();
    var tok = try tokenizer_mod.loadTokenizer(io, a, dir);
    defer tok.deinit();
    config.applyTokenizer(&tok, "<|im_end|>");
    var token_bytes = try token_mask.build(a, &tok);
    defer token_bytes.deinit();

    const prompt = try tok.encode(a, "<|im_start|>user\nWrite the word apple 40 times separated by spaces. Answer in JSON with a words array.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    defer a.free(prompt);
    const schema_src =
        \\{"type":"object","properties":{"words":{"type":"array","items":{"type":"string"}}},"required":["words"],"additionalProperties":false}
    ;
    var schema = try std.json.parseFromSlice(std.json.Value, a, schema_src, .{});
    defer schema.deinit();

    const Run = struct {
        fn ids(al: std.mem.Allocator, x: *Transformer, t: *const Tokenizer, p: []const u32, eos: []const u32, rp: f32, lp: u32, sv: ?std.json.Value, tb: *token_mask.TokenBytes) ![]u32 {
            var sc: SchemaConstraint = undefined;
            if (sv) |v| try sc.initFromValue(al, v, tb);
            defer if (sv != null) sc.deinit();
            const sampling: SamplingParams = .{ .temperature = 0, .repeat_penalty = rp, .constraint = if (sv != null) &sc.constraint else null };
            const r = try generate(std.Io.Threaded.global_single_threaded.io(), al, x, t, p, 48, sampling, eos, 0, lp);
            al.free(r.text);
            if (r.logprobs) |lps| {
                for (lps) |e| al.free(e.top_logprobs);
                al.free(lps);
            }
            return r.token_ids;
        }
    };
    const eos = config.eosTokenSlice();
    const plain = try Run.ids(a, &xfm, &tok, prompt, eos, 1.0, 0, null, &token_bytes);
    defer a.free(plain);
    const pen = try Run.ids(a, &xfm, &tok, prompt, eos, 2.0, 0, null, &token_bytes);
    defer a.free(pen);
    const pen_lp = try Run.ids(a, &xfm, &tok, prompt, eos, 2.0, 1, null, &token_bytes);
    defer a.free(pen_lp);
    try testing.expect(!std.mem.eql(u32, plain, pen_lp));
    try testing.expectEqualSlices(u32, pen_lp, pen);

    const json_plain = try Run.ids(a, &xfm, &tok, prompt, eos, 1.0, 0, schema.value, &token_bytes);
    defer a.free(json_plain);
    const json_pen = try Run.ids(a, &xfm, &tok, prompt, eos, 4.0, 0, schema.value, &token_bytes);
    defer a.free(json_pen);
    try testing.expect(!std.mem.eql(u32, json_plain, json_pen));
}
