//! Lazy (graph-building) sampling, batched rows, and the per-draw seed key.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const keyed_sample = @import("keyed_sample.zig");
const log = @import("../../../sdk/util/log.zig");
const scheduler_mod = @import("../scheduler.zig");
const testing = std.testing;
const SamplingParams = @import("params.zig").SamplingParams;
const applyMinP = @import("filters.zig").applyMinP;
const applySuppressMask = @import("params.zig").applySuppressMask;
const buildSuppressMask = @import("params.zig").buildSuppressMask;
const evalLazyToken = @import("params.zig").evalLazyToken;
const filterTopKTopP = @import("filters.zig").filterTopKTopP;
const pldOneHotRow = @import("spec_sampling.zig").pldOneHotRow;
const sampleFromProbs = @import("spec_sampling.zig").sampleFromProbs;
const sampleResidual = @import("spec_sampling.zig").sampleResidual;


/// Lazy categorical sample from an already-filtered probability row
/// ([1, vocab]): log(probs) puts masked tokens at -inf, categorical draws
/// within the kept set — the same distribution as sampling the filtered
/// logits directly, but the caller keeps `probs` as the proposal density q.
pub fn sampleFromProbsLazy(probs: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var logp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logp);
    try mlx.check(mlx.mlx_log(&logp, probs, s));
    var sampled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sampled);
    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_categorical(&sampled, logp, -1, key, s));
    return sampled;
}

pub fn sampleTokenLazy(logits_in: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) mlx.mlx_array {
    // Reserved-token suppression first, so every path below (greedy fast
    // path included — the collapse drew `<|fim_hole|>` at temp 0) sees the
    // masked row. Lazy like everything else here. The handle starts null-ctx
    // and is only materialized when a mask exists — the fast path below
    // exists to save ONE FFI call per decode step, so the no-mask case (every
    // non-suppressing model) must not spend two on an empty array.
    var suppressed: mlx.mlx_array = .{ .ctx = null };
    defer if (suppressed.ctx != null) {
        _ = mlx.mlx_array_free(suppressed);
    };
    const logits = if (sampling.suppress_mask) |m| blk: {
        suppressed = mlx.mlx_array_new();
        applySuppressMask(&suppressed, logits_in, m, s) catch break :blk logits_in;
        break :blk suppressed;
    } else logits_in;

    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Greedy + seq_len==1 (the decode hot path): one mlx op total. argmax_axis
    // over the vocab dim of a `[1, 1, V]` tensor yields a `[1, 1]` int array,
    // which downstream (resolvePendingToken / lazyForward / async_eval vector)
    // treats identically to `[1]`. Skipping the otherwise-needed reshape +
    // argmax-on-2D combo cuts ~one FFI call per decode step.
    if (seq_len == 1 and sampling.temperature < 0.01) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, logits, -1, false, s);
        return result;
    }

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    // `current` is the single owned intermediate — freed before each reassignment.
    var current = mlx.mlx_array_new();

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, logits, &sq_shape, 2, s);
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        _ = mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s);

        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s);
    }

    // Greedy: argmax (no temperature)
    if (sampling.temperature < 0.01) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, current, -1, false, s);
        _ = mlx.mlx_array_free(current);
        return result;
    }

    if (sampling.keyed and sampling.seed != null) {
        defer _ = mlx.mlx_array_free(current);
        const p: keyed_sample.Params = .{ .seed = sampling.seed.?, .temperature = sampling.temperature, .top_p = sampling.top_p, .top_k = sampling.top_k, .min_p = sampling.min_p orelse 0 };
        const pos = [_]u32{@intCast(sampling.position_base + sampling.draw)};
        return keyed_sample.sample(current, p, &pos, s) catch mlx.mlx_array_new();
    }

    // Scale by 1/temperature
    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        var next = mlx.mlx_array_new();
        _ = mlx.mlx_divide(&next, current, temp_arr, s);
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Apply top-k and top-p in one pass (lazy)
    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        var next = mlx.mlx_array_new();
        filterTopKTopP(&next, current, sampling.top_p, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = next;
    }
    if ((sampling.min_p orelse 0) > 0) {
        var next = mlx.mlx_array_new();
        if (applyMinP(&next, current, sampling.min_p.?, s)) |_| {
            _ = mlx.mlx_array_free(current);
            current = next;
        } else |err| {
            _ = mlx.mlx_array_free(next);
            log.warn("[sampling] min_p failed ({s}); this draw skips min_p\n", .{@errorName(err)});
        }
    }

    // Sample from categorical distribution (lazy — no eval!)
    var sampled = mlx.mlx_array_new();
    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    _ = mlx.mlx_random_categorical(&sampled, current, -1, key, s);
    _ = mlx.mlx_array_free(current);

    return sampled; // Shape [1], lazy
}

pub var sample_rows_evals: u64 = 0;

/// One sampling eval for every row of a batched tick. Homogeneous unseeded rows
/// draw over one stacked `[N, V]` graph (one argmax or one categorical draw);
/// a seed, mixed filters or a lone row take the per-row arm, whose parts are
/// reshaped to `[1]` before the concat because a greedy draw is `[1, 1]` and a
/// categorical draw is `[1]`. Every row advances its `draw` before the eval, so
/// a failed eval never replays a key.
pub fn sampleRows(out: []i32, row_logits: []const mlx.mlx_array, params: []SamplingParams, s: mlx.mlx_stream) !void {
    std.debug.assert(out.len == row_logits.len and out.len == params.len);
    const n = row_logits.len;
    if (n == 0) return;
    const lazy = try sampleRowsLazy(row_logits, params, s);
    defer _ = mlx.mlx_array_free(lazy);
    for (params) |*p| p.draw +%= 1;
    sample_rows_evals += 1;
    try mlx.check(mlx.mlx_array_eval(lazy));
    if (n == 1) {
        try mlx.check(mlx.mlx_array_item_int32(&out[0], lazy));
        return;
    }
    const data = mlx.mlx_array_data_uint32(lazy) orelse return error.MlxError;
    for (out, data[0..n]) |*dst, v| dst.* = @intCast(v);
}

/// Rows may share one stacked graph only when every filter matches; a seed
/// always takes the per-row arm so its key stays per row.
pub fn sampleRowsHomogeneous(params: []const SamplingParams) bool {
    const p0 = params[0];
    for (params[1..]) |p| {
        if (p.temperature != p0.temperature or p.top_k != p0.top_k or p.top_p != p0.top_p or (p.min_p orelse 0) != (p0.min_p orelse 0)) return false;
        const m0 = p0.suppress_mask;
        const m1 = p.suppress_mask;
        if (m0 == null and m1 == null) continue;
        if (m0 == null or m1 == null) return false;
        if (m0.?.ctx != m1.?.ctx) return false;
    }
    return true;
}

fn sampleRowsAllGreedy(params: []const SamplingParams) bool {
    for (params) |p| {
        if (p.temperature >= 0.01) return false;
    }
    return true;
}

fn concatLogitRows(rows: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    if (rows.len == 1) {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&out, rows[0]));
        return out;
    }
    const vec = mlx.mlx_vector_array_new_data(rows.ptr, rows.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}

fn logitsAsNV(logits: mlx.mlx_array, n: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(logits);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (sh.len == 2) {
        try mlx.check(mlx.mlx_array_set(&out, logits));
        return out;
    }
    if (sh.len != 3 or sh[0] != n) return error.BadShape;
    if (sh[1] == 1) {
        const sq = [_]c_int{ n, sh[2] };
        try mlx.check(mlx.mlx_reshape(&out, logits, &sq, 2, s));
        return out;
    }
    const start = [_]c_int{ 0, sh[1] - 1, 0 };
    const stop = [_]c_int{ n, sh[1], sh[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));
    const sq = [_]c_int{ n, sh[2] };
    try mlx.check(mlx.mlx_reshape(&out, sliced, &sq, 2, s));
    return out;
}

fn sampleRowsLazyConcat(row_logits: []const mlx.mlx_array, params: []const SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var parts: [scheduler_mod.MAX_BATCH_GROUP]mlx.mlx_array = undefined;
    const n = row_logits.len;
    if (n > parts.len) return error.BatchTooWide;
    var n_parts: usize = 0;
    // One release on every exit: the concat output holds its own references.
    defer {
        for (parts[0..n_parts]) |a| _ = mlx.mlx_array_free(a);
    }
    const one = [_]c_int{1};
    for (row_logits, params) |logits, p| {
        const raw = sampleTokenLazy(logits, p, s);
        defer _ = mlx.mlx_array_free(raw);
        parts[n_parts] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&parts[n_parts], raw, &one, 1, s));
        n_parts += 1;
    }
    return concatLogitRows(parts[0..n_parts], s);
}

pub fn sampleRowsAnySeed(params: []const SamplingParams) bool {
    for (params) |p| {
        if (p.seed != null) return true;
    }
    return false;
}

fn sampleRowsLazy(row_logits: []const mlx.mlx_array, params: []const SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const n: c_int = @intCast(row_logits.len);
    if (row_logits.len == 1 or !sampleRowsHomogeneous(params) or sampleRowsAnySeed(params)) {
        return sampleRowsLazyConcat(row_logits, params, s);
    }
    const stacked = try concatLogitRows(row_logits, s);
    defer _ = mlx.mlx_array_free(stacked);
    var current = try logitsAsNV(stacked, n, s);
    var owned = true;
    errdefer if (owned) {
        _ = mlx.mlx_array_free(current);
    };
    const p0 = params[0];

    var suppressed: mlx.mlx_array = .{ .ctx = null };
    defer if (suppressed.ctx != null) {
        _ = mlx.mlx_array_free(suppressed);
    };
    if (p0.suppress_mask) |m| {
        suppressed = mlx.mlx_array_new();
        applySuppressMask(&suppressed, current, m, s) catch {
            _ = mlx.mlx_array_free(suppressed);
            suppressed = .{ .ctx = null };
        };
        if (suppressed.ctx != null) {
            _ = mlx.mlx_array_free(current);
            current = suppressed;
            suppressed = .{ .ctx = null };
        }
    }

    if (sampleRowsAllGreedy(params)) {
        var result = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_argmax_axis(&result, current, -1, false, s));
        _ = mlx.mlx_array_free(current);
        owned = false;
        return result;
    }

    if (p0.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(p0.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        var next = mlx.mlx_array_new();
        mlx.check(mlx.mlx_divide(&next, current, temp_arr, s)) catch |err| {
            _ = mlx.mlx_array_free(next);
            return err;
        };
        _ = mlx.mlx_array_free(current);
        current = next;
    }
    if (p0.top_k > 0 or p0.top_p < 1.0) {
        var next = mlx.mlx_array_new();
        filterTopKTopP(&next, current, p0.top_p, p0.top_k, s) catch {
            _ = mlx.mlx_array_free(next);
            next = current;
            current = mlx.mlx_array_new();
        };
        if (current.ctx != null) _ = mlx.mlx_array_free(current);
        current = next;
    }
    if ((p0.min_p orelse 0) > 0) {
        var next = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(next);
        try applyMinP(&next, current, p0.min_p.?, s);
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    const key = seedKey(p0);
    defer _ = mlx.mlx_array_free(key);
    var sampled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, key, s));
    _ = mlx.mlx_array_free(current);
    owned = false;
    return sampled;
}

/// PRNG key for one draw: `seed` mixed with the draw index, so a seeded
/// request replays byte-for-byte and consecutive draws never share a key.
/// Null-ctx (MLX global RNG) when the request set no seed.
/// `logits + Gumbel(key)` over the vocab row `[1, V]`, in f32.
pub fn seedKey(sampling: SamplingParams) mlx.mlx_array {
    const seed = sampling.seed orelse return mlx.mlx_array_new();
    var key = mlx.mlx_array_new();
    _ = mlx.mlx_random_key(&key, seed +% sampling.draw *% 0x9E3779B97F4A7C15);
    return key;
}

// ── tests ──


test "seeded lazy sampling replays the same draws and advances per draw" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 0.0, 0.0 };
    const shape = [_]c_int{ 1, 1, 2 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    var runs: [2][24]i32 = undefined;
    for (&runs) |*run| {
        var params = SamplingParams{ .temperature = 1.0, .seed = 42 };
        for (run) |*out| {
            const lazy = sampleTokenLazy(logits, params, s);
            defer _ = mlx.mlx_array_free(lazy);
            try mlx.check(mlx.mlx_array_eval(lazy));
            try mlx.check(mlx.mlx_array_item_int32(out, lazy));
            params.draw +%= 1;
        }
    }
    try testing.expectEqualSlices(i32, &runs[0], &runs[1]);
    // The same key every draw replays ONE coin flip 24 times (p = 2^-23).
    var all_same = true;
    for (runs[0][1..]) |v| all_same = all_same and v == runs[0][0];
    try testing.expect(!all_same);
}

test "batched greedy ids equal two solo ids and one sampling eval" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 0.5, 0.6, 9.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const greedy = SamplingParams{ .temperature = 0.0 };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, greedy, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, greedy, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ greedy, greedy };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
}

test "batched mixed greedy and sampled rows produce two ids" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 0.5, 0.6, 9.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const greedy = SamplingParams{ .temperature = 0.0 };
    const sampled = SamplingParams{ .temperature = 0.7 };
    const solo_g = try evalLazyToken(sampleTokenLazy(r0, greedy, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ greedy, sampled };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo_g)), ids[0]);
    try testing.expect(ids[1] >= 0 and ids[1] < 4);
}

test "batched stacked seedless top_k top_p rows equal solo draws" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 0.5, 0.6, 9.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const p = SamplingParams{ .temperature = 0.7, .top_k = 1, .top_p = 0.9 };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, p, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, p, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ p, p };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
}

test "batched stacked seedless rows honour a suppress mask" {
    const s = mlx.gpuStream();
    const mask = try buildSuppressMask(&[_]u32{1}, 0, 4, s);
    defer _ = mlx.mlx_array_free(mask);
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 9.0, 0.5, 0.6,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const p = SamplingParams{ .temperature = 0.7, .top_k = 1, .top_p = 0.9, .suppress_mask = mask };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, p, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, p, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ p, p };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
    try testing.expect(ids[0] != 1);
    try testing.expect(ids[1] != 1);
}

test "batched seeded draws keep per-row keys" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const p0 = SamplingParams{ .temperature = 1.0, .seed = 7, .draw = 2 };
    const p1 = SamplingParams{ .temperature = 1.0, .seed = 99, .draw = 5 };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, p0, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, p1, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ p0, p1 };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
    try testing.expectEqual(p0.draw +% 1, params[0].draw);
    try testing.expectEqual(p1.draw +% 1, params[1].draw);
}

test "a seeded correction draw replays: same seed and position, same token; positions differ" {
    // Bar: the bonus/residual draw of a verify round is keyed like every other draw.
    const s = mlx.gpuStream();
    const v: c_int = 64;
    var vals: [64]f32 = undefined;
    var sum: f32 = 0;
    for (&vals, 0..) |*x, i| {
        x.* = 1.0 / @as(f32, @floatFromInt(i + 1));
        sum += x.*;
    }
    for (&vals) |*x| x.* /= sum;
    const shape = [_]c_int{ 1, v };
    const probs = mlx.mlx_array_new_data(&vals, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const onehot = try pldOneHotRow(0, v, s);
    defer _ = mlx.mlx_array_free(onehot);

    var distinct = false;
    var prev: ?u32 = null;
    for (0..16) |d| {
        const sp: SamplingParams = .{ .seed = 11, .draw = d };
        const a = try sampleFromProbs(probs, sp, s);
        try testing.expectEqual(a, try sampleFromProbs(probs, sp, s));
        try testing.expectEqual(try sampleResidual(probs, onehot, sp, s), try sampleResidual(probs, onehot, sp, s));
        try testing.expect(try sampleResidual(probs, onehot, sp, s) != 0); // the rejected draft is never redrawn
        if (prev) |q| distinct = distinct or q != a;
        prev = a;
    }
    try testing.expect(distinct);
}

test "keyed sampling draws the softmax distribution" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const probs = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    var logv: [4]f32 = undefined;
    for (probs, &logv) |p, *l| l.* = @log(p);
    const logits = mlx.mlx_array_new_data(&logv, &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    var counts: [4]u32 = @splat(0);
    const n: u32 = 4000;
    for (0..n) |d| {
        const tok = sampleTokenLazy(logits, .{ .temperature = 1.0, .seed = 11, .draw = d, .keyed = true }, s);
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_array_eval(tok));
        var v: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&v, tok));
        counts[@intCast(v)] += 1;
    }
    for (probs, counts) |p, c| try testing.expect(@abs(@as(f32, @floatFromInt(c)) / @as(f32, @floatFromInt(n)) - p) < 0.03);
}
