//! sampleToken and the grammar logit mask.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const keyed_sample = @import("keyed_sample.zig");
const testing = std.testing;
const LogprobResult = @import("logprobs.zig").LogprobResult;
const SamplingParams = @import("params.zig").SamplingParams;
const applyMinP = @import("filters.zig").applyMinP;
const applyPenalties = @import("filters.zig").applyPenalties;
const applySuppressMask = @import("params.zig").applySuppressMask;
const argmax = @import("filters.zig").argmax;
const computeLogprobs = @import("logprobs.zig").computeLogprobs;
const filterTopKTopP = @import("filters.zig").filterTopKTopP;
const seedKey = @import("lazy.zig").seedKey;


const SampleResult = struct {
    token_id: u32,
    logprob_result: ?LogprobResult = null,
};

/// Sample a token from the last position's logits.
/// temperature <= 0.01: greedy argmax. Otherwise: scale logits, apply top_p, and sample.
/// If logprobs_n > 0, also computes logprobs for the sampled token and top N alternatives.
pub fn sampleToken(allocator: std.mem.Allocator, logits: mlx.mlx_array, sampling: SamplingParams, generated_ids: ?[]const u32, logprobs_n: u32, s: mlx.mlx_stream) !SampleResult {
    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    var last_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last_logits);

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, logits, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));

        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, sliced, &sq_shape, 2, s));
    }

    // Track current working logits (avoid copies when no transform needed)
    var current = last_logits;

    // Reserved-token suppression — before any other transform, so the greedy
    // arm below can't draw a suppressed id. `logprobs_logits` stays
    // `last_logits`: the reported distribution is the MODEL's, so under
    // suppression rank 1 may legitimately differ from the chosen token.
    var suppressed = mlx.mlx_array_new();
    var suppressed_owned = false;
    defer if (suppressed_owned) {
        _ = mlx.mlx_array_free(suppressed);
    };
    if (sampling.suppress_mask) |m| {
        try applySuppressMask(&suppressed, current, m, s);
        current = suppressed;
        suppressed_owned = true;
    }

    // Apply repeat penalty to already-generated tokens
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    if (sampling.penalized()) {
        if (generated_ids) |ids| {
            if (ids.len > 0) {
                try applyPenalties(&penalized, current, ids, sampling, s);
                current = penalized;
                penalized_owned = true;
            }
        }
    }

    // Greedy if temperature is ~0
    if (sampling.temperature < 0.01) {
        const token_id = try argmax(current, s);
        var logprob_result: ?LogprobResult = null;
        if (logprobs_n > 0) {
            logprob_result = try computeLogprobs(allocator, last_logits, token_id, logprobs_n, s);
        }
        return .{ .token_id = token_id, .logprob_result = logprob_result };
    }

    if (sampling.keyed and sampling.seed != null) {
        const pos = [_]u32{@intCast(sampling.position_base + sampling.draw)};
        const tok = try keyed_sample.sample(current, .{ .seed = sampling.seed.?, .temperature = sampling.temperature, .top_p = sampling.top_p, .top_k = sampling.top_k, .min_p = sampling.min_p orelse 0 }, &pos, s);
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_array_eval(tok));
        const token_id = (mlx.mlx_array_data_uint32(tok) orelse return error.MlxArrayDataNull)[0];
        const logprob_result: ?LogprobResult = if (logprobs_n > 0) try computeLogprobs(allocator, last_logits, token_id, logprobs_n, s) else null;
        return .{ .token_id = token_id, .logprob_result = logprob_result };
    }

    // Scale logits by 1/temperature
    var scaled = mlx.mlx_array_new();
    var scaled_owned = false;
    defer if (scaled_owned) {
        _ = mlx.mlx_array_free(scaled);
    };

    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        try mlx.check(mlx.mlx_divide(&scaled, current, temp_arr, s));
        current = scaled;
        scaled_owned = true;
    }

    // Logprobs report the MODEL's distribution, so they read the position's raw
    // logits — not `current`, which by here carries the client's temperature
    // (and any penalty) and would make the same token's logprob move with a
    // sampling knob.
    const logprobs_logits = last_logits;

    // Apply top-k and top-p in one pass
    var filtered = mlx.mlx_array_new();
    var filtered_owned = false;
    defer if (filtered_owned) {
        _ = mlx.mlx_array_free(filtered);
    };

    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        try filterTopKTopP(&filtered, current, sampling.top_p, sampling.top_k, s);
        current = filtered;
        filtered_owned = true;
    }
    var min_filtered: mlx.mlx_array = .{ .ctx = null };
    defer if (min_filtered.ctx != null) {
        _ = mlx.mlx_array_free(min_filtered);
    };
    if ((sampling.min_p orelse 0) > 0) {
        min_filtered = mlx.mlx_array_new();
        try applyMinP(&min_filtered, current, sampling.min_p.?, s);
        current = min_filtered;
    }

    // Sample from categorical distribution
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);

    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, key, s));

    // Eval and extract
    try mlx.check(mlx.mlx_array_eval(sampled));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, sampled));

    const token_id: u32 = @intCast(val);

    // Compute logprobs after sampling (we now know the token_id)
    var logprob_result: ?LogprobResult = null;
    if (logprobs_n > 0) {
        logprob_result = try computeLogprobs(allocator, logprobs_logits, token_id, logprobs_n, s);
    }

    return .{ .token_id = token_id, .logprob_result = logprob_result };
}

/// Apply a grammar token mask to logits. `mask[i]==true` keeps `logits[i]`,
/// `false` replaces it with `-inf`. The mask is broadcast over leading dims so
/// `logits` can be either `[1, vocab]` or `[1, 1, vocab]`.
pub fn applyGrammarMask(allocator: std.mem.Allocator, res: *mlx.mlx_array, logits: mlx.mlx_array, mask: []const bool, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);
    const logit_mask = try maskForLogitVocab(allocator, mask, vocab_size);
    defer logit_mask.deinit(allocator);

    // Zig's `bool` is one byte and matches MLX's `.bool_` storage exactly.
    const arr_shape = [_]c_int{@intCast(vocab_size)};
    const mask_arr = mlx.mlx_array_new_data(@ptrCast(logit_mask.slice.ptr), &arr_shape, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);

    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);

    try mlx.check(mlx.mlx_where(res, mask_arr, logits, neg_inf, s));
}

const LogitMaskView = struct {
    slice: []const bool,
    owned: ?[]bool = null,

    pub fn deinit(self: LogitMaskView, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
    }
};

pub fn maskForLogitVocab(allocator: std.mem.Allocator, mask: []const bool, vocab_size: usize) !LogitMaskView {
    if (mask.len == vocab_size) return .{ .slice = mask };

    var adjusted = try allocator.alloc(bool, vocab_size);
    @memset(adjusted, false);
    const copy_len = @min(mask.len, vocab_size);
    @memcpy(adjusted[0..copy_len], mask[0..copy_len]);
    return .{ .slice = adjusted, .owned = adjusted };
}

// ── tests ──


test "sampleToken greedy selects argmax" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // Create logits [1, 1, 5] — 5 vocab entries, token at index 3 has highest
    const data = [_]f32{ 1.0, 0.5, 0.1, 5.0, 0.2 };
    const logits_shape = [_]c_int{ 1, 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    try testing.expectEqual(@as(u32, 3), result.token_id);
}

test "sampleToken with temperature produces valid token" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    const data = [_]f32{ 1.0, 2.0, 3.0 };
    const logits_shape = [_]c_int{ 1, 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.5 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    // Token should be in valid range
    try testing.expect(result.token_id < 3);
}

test "sampleToken from prefill logits (seq_len > 1)" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // [1, 3, 4] — 3 positions, 4 vocab, should take last position
    const data = [_]f32{
        0.1, 0.2, 0.3, 0.4, // pos 0
        0.5, 0.6, 0.7, 0.8, // pos 1
        9.0, 0.1, 0.1, 0.1, // pos 2 — token 0 is clearly highest
    };
    const logits_shape = [_]c_int{ 1, 3, 4 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    try testing.expectEqual(@as(u32, 0), result.token_id); // pos 2, index 0 = 9.0
}
