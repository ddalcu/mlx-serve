//! Embeddings: padded batches, masks, pooling, normalization and the batched forward.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const Transformer = transformer_mod.Transformer;
const plugin_driver = @import("../plugin/driver.zig");
const testing = std.testing;


/// Compute a pooled (per the model's `pooling_mode` — mean by default),
/// L2-normalized embedding from token IDs.
/// Returns a float32 array of shape [hidden_size]. Caller must free the returned slice.
pub fn computeEmbedding(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    token_ids: []const u32,
) ![]f32 {
    const seqs = [_][]const u32{token_ids};
    const rows = try computeEmbeddingsBatch(allocator, xfm, &seqs);
    defer allocator.free(rows);
    return rows[0];
}

/// GPU batch-size cap for encoder embedding forwards: bounds padded-batch
/// memory while keeping the GPU saturated.
pub const EMBED_MAX_BATCH: usize = 64;

/// Padded-token budget per embedding sub-batch (issue #117): the item cap
/// alone lets ONE long input inflate the whole padded allocation (64 rows
/// padded to a 32K outlier = 2M positions of hidden state). The budget bounds
/// rows × padded length at the historical worst case (64 rows × 512 — every
/// legacy BERT batch is unchanged); long-context inputs simply ride smaller
/// sub-batches. A single over-budget input still runs, alone.
pub const EMBED_TOKEN_BUDGET: usize = EMBED_MAX_BATCH * 512;

/// End index (exclusive) of the embedding sub-batch starting at `start`:
/// grows while under BOTH the item cap and the padded-footprint budget
/// (rows × running max length). Always takes at least one item, so an input
/// longer than the whole budget is processed rather than looping forever.
/// Input order is preserved — sub-batches are contiguous slices.
pub fn embedSubBatchEnd(seqs: []const []const u32, start: usize, max_items: usize, budget: usize) usize {
    var end = start;
    var max_len: usize = 0;
    while (end < seqs.len and end - start < max_items) {
        const grown_max = @max(max_len, seqs[end].len);
        if (end > start and (end - start + 1) * grown_max > budget) break;
        max_len = grown_max;
        end += 1;
    }
    return end;
}

/// One padded batch of token sequences ready for an encoder forward.
pub const PaddedBatch = struct {
    ids: []i32, // [B * max_len] row-major, pad id 0
    lengths: []usize, // [B]
    max_len: usize,

    pub fn deinit(self: *PaddedBatch, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.lengths);
    }
};

/// Pad `seqs` into one [B, max_len] i32 buffer (pad id 0). Padded positions
/// are excluded from attention (`buildKeyPadMask`) and pooling
/// (`maskedMeanPoolNormalize`), so the pad id value never leaks into results.
pub fn buildPaddedBatch(allocator: std.mem.Allocator, seqs: []const []const u32) !PaddedBatch {
    var max_len: usize = 0;
    for (seqs) |seq| max_len = @max(max_len, seq.len);
    if (max_len == 0) return error.EmptyInput;

    const ids = try allocator.alloc(i32, seqs.len * max_len);
    errdefer allocator.free(ids);
    const lengths = try allocator.alloc(usize, seqs.len);
    errdefer allocator.free(lengths);
    @memset(ids, 0);
    for (seqs, 0..) |seq, b| {
        lengths[b] = seq.len;
        for (seq, 0..) |id, t| ids[b * max_len + t] = @intCast(id);
    }
    return .{ .ids = ids, .lengths = lengths, .max_len = max_len };
}

/// Additive key-padding mask [B, 1, 1, max_len] (bf16): 0 over real keys,
/// -inf over padding. Broadcasts across heads and query positions; padded
/// QUERIES still produce garbage rows, but pooling drops them.
pub fn buildKeyPadMask(allocator: std.mem.Allocator, lengths: []const usize, max_len: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const buf = try allocator.alloc(f32, lengths.len * max_len);
    defer allocator.free(buf);
    for (lengths, 0..) |len, b| {
        for (0..max_len) |t| {
            buf[b * max_len + t] = if (t < len) 0 else -std.math.inf(f32);
        }
    }
    const shape = [_]c_int{ @intCast(lengths.len), 1, 1, @intCast(max_len) };
    const f32_mask = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_mask);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&mask, f32_mask, .bfloat16, s));
    return mask;
}

/// Mean-pool `hidden` [B, T, H] over each row's first `lengths[b]` positions.
/// Returns the pooled [B, H] mlx array (f32-promoted); caller frees.
pub fn maskedMeanPool(allocator: std.mem.Allocator, hidden: mlx.mlx_array, lengths: []const usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(hidden);
    const batch: usize = @intCast(shape[0]);
    const seq_len: usize = @intCast(shape[1]);

    // Pool weights [B, T, 1]: 1/len over real positions, 0 over padding — a
    // weighted sum along T is then exactly the masked mean. f32 weights also
    // promote a bf16 hidden so the final data extraction is float32-safe.
    const wbuf = try allocator.alloc(f32, batch * seq_len);
    defer allocator.free(wbuf);
    for (lengths, 0..) |len, b| {
        const denom: f32 = @floatFromInt(@max(len, 1));
        for (0..seq_len) |t| {
            wbuf[b * seq_len + t] = if (t < len) 1.0 / denom else 0.0;
        }
    }
    const wshape = [_]c_int{ shape[0], shape[1], 1 };
    const weights = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 3, .float32);
    defer _ = mlx.mlx_array_free(weights);

    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, hidden, weights, s));

    var pooled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_sum_axis(&pooled, weighted, 1, false, s)); // [B, H]
    return pooled;
}

/// Mean-pool `hidden` [B, T, H] over each row's first `lengths[b]` positions
/// and L2-normalize. Returns B owned rows of H f32 each (plus the outer
/// slice); caller frees all.
pub fn maskedMeanPoolNormalize(allocator: std.mem.Allocator, hidden: mlx.mlx_array, lengths: []const usize, s: mlx.mlx_stream) ![][]f32 {
    const pooled = try maskedMeanPool(allocator, hidden, lengths, s);
    defer _ = mlx.mlx_array_free(pooled);
    return l2NormalizeRows(allocator, pooled, s);
}

/// Single-position pooling (issue #116): select ONE hidden row per batch
/// element — the CLS token (position 0; bge/mxbai) or the last real,
/// non-padding token (`lengths[b]-1`; Qwen3-Embedding). Same one-hot
/// weighted-sum shape as `maskedMeanPool`, so padded garbage can never leak
/// and a bf16 hidden is promoted to f32 by the weights. Returns the pooled
/// [B, H] mlx array; caller frees.
pub fn gatherTokenPool(allocator: std.mem.Allocator, hidden: mlx.mlx_array, lengths: []const usize, mode: model_mod.PoolingMode, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(hidden);
    const batch: usize = @intCast(shape[0]);
    const seq_len: usize = @intCast(shape[1]);

    const wbuf = try allocator.alloc(f32, batch * seq_len);
    defer allocator.free(wbuf);
    @memset(wbuf, 0);
    for (lengths, 0..) |len, b| {
        const pos: usize = switch (mode) {
            .cls => 0,
            .last_token => @max(len, 1) - 1,
            .mean => return error.InvalidPoolingMode,
        };
        wbuf[b * seq_len + @min(pos, seq_len - 1)] = 1.0;
    }
    const wshape = [_]c_int{ shape[0], shape[1], 1 };
    const weights = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 3, .float32);
    defer _ = mlx.mlx_array_free(weights);

    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, hidden, weights, s));

    var pooled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_sum_axis(&pooled, weighted, 1, false, s)); // [B, H]
    return pooled;
}

/// L2-normalize each row of `pooled` [B, H] and read out as owned f32 rows.
pub fn l2NormalizeRows(allocator: std.mem.Allocator, pooled: mlx.mlx_array, s: mlx.mlx_stream) ![][]f32 {
    const pshape = mlx.getShape(pooled);
    const batch: usize = @intCast(pshape[0]);
    const dim: usize = @intCast(pshape[1]);

    // L2 normalize rows: pooled / max(sqrt(sum(pooled^2)), eps).
    var squared = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(squared);
    try mlx.check(mlx.mlx_multiply(&squared, pooled, pooled, s));

    var sum_sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum_sq);
    try mlx.check(mlx.mlx_sum_axis(&sum_sq, squared, -1, true, s));

    var norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm);
    try mlx.check(mlx.mlx_sqrt(&norm, sum_sq, s));

    const eps = mlx.mlx_array_new_float(1e-12);
    defer _ = mlx.mlx_array_free(eps);
    var norm_safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm_safe);
    try mlx.check(mlx.mlx_maximum(&norm_safe, norm, eps, s));

    var normalized = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normalized);
    try mlx.check(mlx.mlx_divide(&normalized, pooled, norm_safe, s));

    try mlx.check(mlx.mlx_array_eval(normalized));
    const data_ptr = mlx.mlx_array_data_float32(normalized) orelse return error.MlxError;

    const rows = try allocator.alloc([]f32, batch);
    var done: usize = 0;
    errdefer {
        for (rows[0..done]) |r| allocator.free(r);
        allocator.free(rows);
    }
    for (rows, 0..) |*row, b| {
        row.* = try allocator.alloc(f32, dim);
        @memcpy(row.*, data_ptr[b * dim .. (b + 1) * dim]);
        done += 1;
    }
    return rows;
}

/// Compute embeddings for many token sequences in GPU batches: each chunk of
/// up to `EMBED_MAX_BATCH` sequences is padded to its own max length,
/// forwarded ONCE through the encoder with a key-padding mask,
/// masked-mean-pooled, and L2-normalized. Input order preserved. Caller
/// frees every returned row and the outer slice.
pub fn computeEmbeddingsBatch(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    seqs: []const []const u32,
) ![][]f32 {
    return computeEmbeddingsBatchWith(allocator, xfm, seqs, null);
}

/// `computeEmbeddingsBatch` for ONE sequence carrying image/video placeholders: `vision` holds the soft-token
/// rows [1, n, hidden] of every item in prompt order, spliced in at the placeholder ids.
pub fn computeEmbeddingsBatchWith(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    seqs: []const []const u32,
    vision: ?mlx.mlx_array,
) ![][]f32 {
    // The rows are consumed in sequence order by one scatter: they belong to a single prompt.
    if (vision != null and seqs.len != 1) return error.MultimodalNeedsOneSequence;
    const results = try allocator.alloc([]f32, seqs.len);
    var filled: usize = 0;
    errdefer {
        for (results[0..filled]) |r| allocator.free(r);
        allocator.free(results);
    }
    const max_items: usize = if (plugin_driver.batchesEmbeddings(&xfm.config)) EMBED_MAX_BATCH else 1;
    var start: usize = 0;
    while (start < seqs.len) {
        // A decoder-arch embedder (Qwen3-Embedding) forwards through the KV
        // cache: without a reset, a sub-batch attends to earlier sub-batches' rows.
        try xfm.resetCache();
        const sub = seqs[start..embedSubBatchEnd(seqs, start, max_items, EMBED_TOKEN_BUDGET)];
        var pb = try buildPaddedBatch(allocator, sub);
        defer pb.deinit(allocator);

        const shape = [_]c_int{ @intCast(sub.len), @intCast(pb.max_len) };
        const input = mlx.mlx_array_new_data(pb.ids.ptr, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(input);

        // A single sequence has no padding, so it needs no mask.
        var mask: ?mlx.mlx_array = null;
        defer if (mask) |m| {
            _ = mlx.mlx_array_free(m);
        };
        if (sub.len > 1) mask = try buildKeyPadMask(allocator, pb.lengths, pb.max_len, xfm.s);

        const hidden = try xfm.forwardEmbeddingMasked(input, mask, vision);
        defer _ = mlx.mlx_array_free(hidden);

        // Sentence-transformers pipeline order: pool (per the checkpoint's
        // declared mode — mean by default, CLS for bge/mxbai, last-token for
        // Qwen3-Embedding) → dense head (when the checkpoint ships one —
        // EmbeddingGemma) → normalize.
        const pooled = switch (xfm.config.effectivePooling()) {
            .mean => try maskedMeanPool(allocator, hidden, pb.lengths, xfm.s),
            .cls, .last_token => |m| try gatherTokenPool(allocator, hidden, pb.lengths, m, xfm.s),
        };
        defer _ = mlx.mlx_array_free(pooled);
        const rows = if (xfm.hasEmbedProjection()) blk: {
            const projected = try xfm.embedProjection(pooled);
            defer _ = mlx.mlx_array_free(projected);
            break :blk try l2NormalizeRows(allocator, projected, xfm.s);
        } else try l2NormalizeRows(allocator, pooled, xfm.s);
        defer allocator.free(rows);
        for (rows, 0..) |r, i| {
            results[start + i] = r;
            filled += 1;
        }
        start += sub.len;
    }
    return results;
}

// ── tests ──


test "buildPaddedBatch pads to max length with zeros and records lengths" {
    const seqs = [_][]const u32{
        &[_]u32{ 101, 7592, 102 },
        &[_]u32{ 101, 102 },
    };
    var pb = try buildPaddedBatch(testing.allocator, &seqs);
    defer pb.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 3), pb.max_len);
    try testing.expectEqual(@as(usize, 2), pb.lengths.len);
    try testing.expectEqual(@as(usize, 3), pb.lengths[0]);
    try testing.expectEqual(@as(usize, 2), pb.lengths[1]);
    const expected = [_]i32{ 101, 7592, 102, 101, 102, 0 };
    try testing.expectEqualSlices(i32, &expected, pb.ids);
}

test "buildKeyPadMask is additive zero on real keys, -inf on padding" {
    const s = mlx.gpuStream();
    const lengths = [_]usize{ 3, 1 };
    const mask = try buildKeyPadMask(testing.allocator, &lengths, 3, s);
    defer _ = mlx.mlx_array_free(mask);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2, 1, 1, 3 }, mlx.getShape(mask));
    var f32_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_mask);
    try mlx.check(mlx.mlx_astype(&f32_mask, mask, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32_mask));
    const data = mlx.mlx_array_data_float32(f32_mask).?;
    // Batch row 0: all three keys real.
    try testing.expectEqual(@as(f32, 0), data[0]);
    try testing.expectEqual(@as(f32, 0), data[1]);
    try testing.expectEqual(@as(f32, 0), data[2]);
    // Batch row 1: one real key, two padded.
    try testing.expectEqual(@as(f32, 0), data[3]);
    try testing.expect(std.math.isInf(data[4]) and data[4] < 0);
    try testing.expect(std.math.isInf(data[5]) and data[5] < 0);
}

test "maskedMeanPoolNormalize excludes padded positions and unit-normalizes" {
    const s = mlx.gpuStream();
    // hidden [2, 3, 2]; row 0 has 2 real positions (pad slot holds garbage
    // that must not leak into the pool), row 1 has 3.
    const data = [_]f32{
        1, 0, 3, 4, 100, 100,
        0, 2, 0, 4, 0,   6,
    };
    const shape = [_]c_int{ 2, 3, 2 };
    const hidden = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hidden);
    const lengths = [_]usize{ 2, 3 };
    const rows = try maskedMeanPoolNormalize(testing.allocator, hidden, &lengths, s);
    defer {
        for (rows) |r| testing.allocator.free(r);
        testing.allocator.free(rows);
    }
    // Row 0: mean of (1,0),(3,4) = (2,2) → L2-normalized (1/√2, 1/√2).
    try testing.expectApproxEqAbs(@as(f32, 0.70710678), rows[0][0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.70710678), rows[0][1], 1e-4);
    // Row 1: mean of (0,2),(0,4),(0,6) = (0,4) → normalized (0, 1).
    try testing.expectApproxEqAbs(@as(f32, 0.0), rows[1][0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0), rows[1][1], 1e-4);
}

test "embedSubBatchEnd: item cap, padded-footprint budget, oversize singleton, order" {
    const short = [_]u32{ 1, 2 };
    const long: [600]u32 = @splat(0);
    // Item cap alone (all short): full batch.
    const all_short = [_][]const u32{ &short, &short, &short };
    try testing.expectEqual(@as(usize, 3), embedSubBatchEnd(&all_short, 0, 64, 64 * 512));
    try testing.expectEqual(@as(usize, 2), embedSubBatchEnd(&all_short, 0, 2, 64 * 512));

    // One long input caps how many rows pad to its length: 600-token rows fit
    // budget/600 = 54 per sub-batch at the default budget, not 64.
    const many_long: [60][]const u32 = @splat(&long);
    const end = embedSubBatchEnd(&many_long, 0, 64, 64 * 512);
    try testing.expectEqual(@as(usize, (64 * 512) / 600), end);

    // A mixed batch stops BEFORE a long input would inflate the padded
    // footprint of everything before it.
    const mixed = [_][]const u32{ &short, &short, &long };
    const mixed_end = embedSubBatchEnd(&mixed, 0, 64, 4);
    try testing.expectEqual(@as(usize, 2), mixed_end);

    // An input larger than the whole budget still runs — alone.
    const huge: [40000]u32 = @splat(0);
    const with_huge = [_][]const u32{ &huge, &short };
    try testing.expectEqual(@as(usize, 1), embedSubBatchEnd(&with_huge, 0, 64, 64 * 512));
    // ...and the next sub-batch starts right after it (order preserved).
    try testing.expectEqual(@as(usize, 2), embedSubBatchEnd(&with_huge, 1, 64, 64 * 512));
}

test "gatherTokenPool: cls takes position 0, last_token takes the last real position" {
    const s = mlx.gpuStream();
    // hidden [2, 3, 2]; row 0 has 2 real positions (position 2 is pad garbage
    // that must never be selected), row 1 has 3.
    const data = [_]f32{
        1, 0, 3, 4, 100, 100,
        0, 2, 0, 4, 0,   6,
    };
    const shape = [_]c_int{ 2, 3, 2 };
    const hidden = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hidden);
    const lengths = [_]usize{ 2, 3 };

    const cls = try gatherTokenPool(testing.allocator, hidden, &lengths, .cls, s);
    defer _ = mlx.mlx_array_free(cls);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2, 2 }, mlx.getShape(cls));
    try mlx.check(mlx.mlx_array_eval(cls));
    const cls_data = mlx.mlx_array_data_float32(cls).?;
    try testing.expectEqual(@as(f32, 1), cls_data[0]); // hidden[0,0,:]
    try testing.expectEqual(@as(f32, 0), cls_data[1]);
    try testing.expectEqual(@as(f32, 0), cls_data[2]); // hidden[1,0,:]
    try testing.expectEqual(@as(f32, 2), cls_data[3]);

    const last = try gatherTokenPool(testing.allocator, hidden, &lengths, .last_token, s);
    defer _ = mlx.mlx_array_free(last);
    try mlx.check(mlx.mlx_array_eval(last));
    const last_data = mlx.mlx_array_data_float32(last).?;
    try testing.expectEqual(@as(f32, 3), last_data[0]); // hidden[0,1,:] — NOT the pad slot
    try testing.expectEqual(@as(f32, 4), last_data[1]);
    try testing.expectEqual(@as(f32, 0), last_data[2]); // hidden[1,2,:]
    try testing.expectEqual(@as(f32, 6), last_data[3]);
}

test "gatherTokenPool: mean mode is not a gather — callers must dispatch it to maskedMeanPool" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1, 2 };
    const shape = [_]c_int{ 1, 1, 2 };
    const hidden = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hidden);
    const lengths = [_]usize{1};
    try testing.expectError(error.InvalidPoolingMode, gatherTokenPool(testing.allocator, hidden, &lengths, .mean, s));
}
