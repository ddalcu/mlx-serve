//! Logprobs: the model's distribution per returned token and per prompt row.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const testing = std.testing;
const SamplingParams = @import("params.zig").SamplingParams;
const sampleToken = @import("sampler.zig").sampleToken;


pub const TokenLogprob = @import("../../../sdk/request/logprob.zig").TokenLogprob;
pub const LogprobResult = @import("../../../sdk/request/logprob.zig").LogprobResult;

/// Logprobs for the FIRST generated token, taken from the prefill's
/// final-position logits — the one distribution the decode loop never sees.
/// `chosen` is the id the lazy sampler actually drew, so the reported
/// `token_logprob` belongs to the token that was really emitted (re-sampling
/// here would disagree with it at any temperature above 0).
pub fn firstTokenLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen: u32, logprobs_n: u32, s: mlx.mlx_stream) !?LogprobResult {
    if (logprobs_n == 0) return null;
    const shape = mlx.getShape(logits);
    if (shape.len != 3) return null;
    var last = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last);
    const sq_shape = [_]c_int{ 1, shape[2] };
    if (shape[1] == 1) {
        try mlx.check(mlx.mlx_reshape(&last, logits, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, shape[1] - 1, 0 };
        const stop = [_]c_int{ 1, shape[1], shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));
        try mlx.check(mlx.mlx_reshape(&last, sliced, &sq_shape, 2, s));
    }
    return try computeLogprobs(allocator, last, chosen, logprobs_n, s);
}

/// Compute log-probabilities from logits. Returns the logprob of the chosen token
/// and the top N alternatives with their token IDs and logprobs.
///
/// `logits` are the MODEL's logits for the position — before temperature,
/// penalties and top-k/top-p — so the reported distribution is the model's, as
/// OpenAI's is. Reading them after temperature made every value move with a
/// knob the client set (and at temp 0 the distribution SATURATES, so most
/// entries report exactly 0.0).
///
/// Ids travel WITH their values through `mlx_argpartition_axis` + a gather.
/// Recovering them afterwards by scanning the vocab for float equality — what
/// this did — is ambiguous the moment two logits tie, which under the
/// saturation above is everywhere: rank 1 was measured to be the chosen token
/// in 0 of 5 positions on a trivial greedy prompt.
pub const MAX_TOP_LOGPROBS = @import("../../../sdk/request/logprob.zig").MAX_TOP_LOGPROBS;

pub fn computeLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen_token: u32, top_n: u32, s: mlx.mlx_stream) !LogprobResult {
    // Every caller passes one position, as [1, V] or [1, 1, V].
    const shape = mlx.getShape(logits);
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_reshape(&row, logits, &[_]c_int{ 1, shape[shape.len - 1] }, 2, s));
    var one: std.ArrayList(LogprobResult) = .empty;
    defer one.deinit(allocator);
    try rowLogprobs(allocator, row, &.{chosen_token}, top_n, s, &one);
    return one.items[0];
}

/// argpartition leaves the winners UNORDERED; ties break on the lower id so
/// the ranking is deterministic run to run.
fn rankedBefore(_: void, a: TokenLogprob, b: TokenLogprob) bool {
    if (a.logprob != b.logprob) return a.logprob > b.logprob;
    return a.token_id < b.token_id;
}

/// Row r of `logits` ([rows, V]) scores `chosen[r]` and lists its `top_n` best; an id past
/// the vocab scores -100. log_softmax is a per-row shift, so ranks and gathers read the f32
/// logits and only the gathered values subtract the row's logsumexp. Every intermediate
/// handle is gone before the one eval, so the peak is the top-k argpartition: its f32
/// input, its indices and the sort's scratch, five [rows, V] 4-byte arrays (none at top_n 0).
fn rowLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen: []const u32, top_n: u32, s: mlx.mlx_stream, out: *std.ArrayList(LogprobResult)) !void {
    const shape = mlx.getShape(logits);
    const rows: c_int = @intCast(chosen.len);
    const vocab: usize = @intCast(shape[shape.len - 1]);
    const k: usize = @min(@as(usize, @min(top_n, MAX_TOP_LOGPROBS)), vocab);
    const clamped = try allocator.alloc(u32, chosen.len);
    defer allocator.free(clamped);
    for (chosen, clamped) |c, *d| d.* = @min(c, @as(u32, @intCast(vocab - 1)));

    var chosen_lp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chosen_lp);
    var top_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_vals);
    var top_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_ids);
    {
        // f32 first: the logits dtype rounds bf16/f16 before the log, and f16 underflows to -inf.
        var logits32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(logits32);
        try mlx.check(mlx.mlx_astype(&logits32, logits, .float32, s));
        var lse = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lse);
        try mlx.check(mlx.mlx_logsumexp_axis(&lse, logits32, -1, true, s));
        const ids = mlx.mlx_array_new_data(@ptrCast(clamped.ptr), &[_]c_int{ rows, 1 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_take_along_axis(&picked, logits32, ids, -1, s));
        try mlx.check(mlx.mlx_subtract(&chosen_lp, picked, lse, s));
        if (k > 0) {
            // Ascending partition at V - k leaves the k largest in the tail.
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_argpartition_axis(&part, logits32, @intCast(vocab - k), -1, s));
            var idx = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(idx);
            try mlx.check(mlx.mlx_slice(&idx, part, &[_]c_int{ 0, @intCast(vocab - k) }, 2, &[_]c_int{ rows, @intCast(vocab) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            var vals = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vals);
            try mlx.check(mlx.mlx_take_along_axis(&vals, logits32, idx, -1, s));
            try mlx.check(mlx.mlx_subtract(&top_vals, vals, lse, s));
            try mlx.check(mlx.mlx_astype(&top_ids, idx, .int32, s));
        }
    }
    const outs = [_]mlx.mlx_array{ chosen_lp, top_vals, top_ids };
    const vec = mlx.mlx_vector_array_new_data(&outs, if (k > 0) 3 else 1);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));

    const chosen_ptr = mlx.mlx_array_data_float32(chosen_lp) orelse return error.LogprobsUnreadable;
    const val_ptr: [*]const f32 = if (k > 0) mlx.mlx_array_data_float32(top_vals) orelse return error.LogprobsUnreadable else undefined;
    const id_ptr: [*]const i32 = if (k > 0) mlx.mlx_array_data_int32(top_ids) orelse return error.LogprobsUnreadable else undefined;
    try out.ensureUnusedCapacity(allocator, chosen.len);
    for (chosen, 0..) |id, r| {
        const top = try allocator.alloc(TokenLogprob, k);
        for (top, 0..) |*t, j| t.* = .{ .token_id = @intCast(id_ptr[r * k + j]), .logprob = val_ptr[r * k + j] };
        std.mem.sort(TokenLogprob, top, {}, rankedBefore);
        out.appendAssumeCapacity(.{ .token_logprob = if (id < vocab) chosen_ptr[r] else -100.0, .top_logprobs = top });
    }
}

/// Logprobs of a prompt's own tokens (`/v1/completions` echo): row r of `logits`
/// ([1, S, V]) scores `next_ids[r]`, the token after it.
pub fn promptLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, next_ids: []const u32, top_n: u32, s: mlx.mlx_stream, out: *std.ArrayList(LogprobResult)) !void {
    if (next_ids.len == 0) return;
    const shape = mlx.getShape(logits);
    if (shape.len != 3 or shape[1] < next_ids.len) return error.PromptLogprobsUnavailable;
    const rows: c_int = @intCast(next_ids.len);
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, logits, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, rows, shape[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, head, &[_]c_int{ rows, shape[2] }, 2, s));
    try rowLogprobs(allocator, flat, next_ids, top_n, s, out);
}

// ── tests ──


test "computeLogprobs: rank 1 is the argmax and ranks descend, under a tie-saturated distribution" {
    // The producer used to recover token ids by scanning the whole vocab for
    // FLOAT EQUALITY with each `mlx_topk` value. Ties make that ambiguous, and
    // at temp 0 the post-temperature distribution saturates so ties are
    // everywhere — measured 0/5 positions where rank 1 was the chosen token on
    // a trivial prompt. The observable contract: with greedy sampling the
    // chosen token IS the argmax, so rank 1 must equal it.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    // Six exactly-tied entries, then a strict runner-up and a strict max.
    const data = [_]f32{ 1, 1, 1, 1, 1, 1, 2, 3 };
    const shape = [_]c_int{ 1, 8 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const chosen: u32 = 7; // argmax
    const r = try computeLogprobs(allocator, logits, chosen, 3, s);
    defer allocator.free(r.top_logprobs);

    try testing.expectEqual(@as(usize, 3), r.top_logprobs.len);
    try testing.expectEqual(chosen, r.top_logprobs[0].token_id);
    try testing.expectEqual(@as(u32, 6), r.top_logprobs[1].token_id);
    try testing.expect(r.top_logprobs[2].token_id < 6);
    // Strictly descending, and ids distinct.
    try testing.expect(r.top_logprobs[0].logprob > r.top_logprobs[1].logprob);
    try testing.expect(r.top_logprobs[1].logprob > r.top_logprobs[2].logprob);
    try testing.expect(r.top_logprobs[0].token_id != r.top_logprobs[1].token_id);
    try testing.expect(r.top_logprobs[1].token_id != r.top_logprobs[2].token_id);
    // The chosen token's own logprob must agree with its rank-1 entry.
    try testing.expectApproxEqAbs(r.top_logprobs[0].logprob, r.token_logprob, 1e-6);
}

test "computeLogprobs: f16 logits keep finite, exact log-probabilities" {
    // p(token 1) = e^-25 underflows f16, so log(softmax) in the logits dtype read -inf.
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const raw = [_]f32{ 0.0, -20.0, -30.0, 5.0 };
    const a32 = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(a32);
    var a16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a16);
    try mlx.check(mlx.mlx_astype(&a16, a32, .float16, s));
    const r = try computeLogprobs(testing.allocator, a16, 1, 4, s);
    defer testing.allocator.free(r.top_logprobs);
    const lse: f32 = 5.0 + @log(1.0 + @exp(@as(f32, -5.0)) + @exp(@as(f32, -25.0)) + @exp(@as(f32, -35.0)));
    try testing.expectApproxEqAbs(-20.0 - lse, r.token_logprob, 1e-4);
    for (r.top_logprobs) |t| try testing.expect(std.math.isFinite(t.logprob));
}

test "computeLogprobs: an id past the vocab reports -100, top_n 0 lists none, [1,1,V] reads as [1,V]" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const allocator = testing.allocator;
    const raw = [_]f32{ 0, 1, 2, 3 };
    const flat = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(flat);
    const deep = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(deep);
    const lse: f32 = 3.0 + @log(1.0 + @exp(@as(f32, -1)) + @exp(@as(f32, -2)) + @exp(@as(f32, -3)));

    const past = try computeLogprobs(allocator, flat, 9, 2, s);
    defer allocator.free(past.top_logprobs);
    try testing.expectEqual(@as(f32, -100.0), past.token_logprob);
    try testing.expectEqual(@as(usize, 2), past.top_logprobs.len);
    try testing.expectEqual(@as(u32, 3), past.top_logprobs[0].token_id);

    const none = try computeLogprobs(allocator, flat, 1, 0, s);
    defer allocator.free(none.top_logprobs);
    try testing.expectEqual(@as(usize, 0), none.top_logprobs.len);
    try testing.expectApproxEqAbs(1.0 - lse, none.token_logprob, 1e-5);

    const d = try computeLogprobs(allocator, deep, 2, 3, s);
    defer allocator.free(d.top_logprobs);
    try testing.expectApproxEqAbs(2.0 - lse, d.token_logprob, 1e-5);
    try testing.expectEqual(@as(usize, 3), d.top_logprobs.len);
    try testing.expectEqual(@as(u32, 1), d.top_logprobs[2].token_id);
}

test "promptLogprobs: no vocab-wide array outlives its use, so the top-k sort is the peak" {
    // Over the bar on GitHub's virtual GPU: https://github.com/ddalcu/mlx-serve/actions/runs/40896964158
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const rows = 64;
    const vocab = 65536;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 7));
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_random_normal(&logits, &[_]c_int{ 1, rows, vocab }, 3, .bfloat16, 0, 1, key, s));
    try mlx.check(mlx.mlx_array_eval(logits));
    var ids: [rows]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i * 997 % vocab);
    var out: std.ArrayList(LogprobResult) = .empty;
    defer {
        for (out.items) |r| allocator.free(r.top_logprobs);
        out.deinit(allocator);
    }

    var base: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&base));
    _ = mlx.mlx_reset_peak_memory();
    try promptLogprobs(allocator, logits, &ids, 5, s, &out);
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    // Bar: under six [rows, vocab] 4-byte arrays above what was live before (the sort alone holds five).
    try testing.expect(peak - base < 6 * rows * vocab * 4);
    try testing.expectEqual(@as(usize, rows), out.items.len);
}

test "promptLogprobs: every row scores its NEXT token, ranks tie on the lower id" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const allocator = testing.allocator;
    const rows = [3][8]f32{
        .{ 1, 1, 1, 1, 1, 1, 2, 3 },
        .{ 0, 0, 0, 0, 0, 3, 0, 3 },
        .{ -1.5, 0.25, 4, -3, 2, 0, 1, 0.5 },
    };
    const logits = mlx.mlx_array_new_data(&rows, &[_]c_int{ 1, 3, 8 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const next = [_]u32{ 7, 5, 3 };

    var out: std.ArrayList(LogprobResult) = .empty;
    defer {
        for (out.items) |r| allocator.free(r.top_logprobs);
        out.deinit(allocator);
    }
    try promptLogprobs(allocator, logits, &next, 2, s, &out);
    try testing.expectEqual(@as(usize, 3), out.items.len);
    for (rows, next, out.items) |row, id, r| {
        var lse: f64 = 0;
        for (row) |v| lse += @exp(@as(f64, v));
        lse = @log(lse);
        try testing.expectApproxEqAbs(@as(f64, row[id]) - lse, r.token_logprob, 1e-5);
        try testing.expectEqual(@as(usize, 2), r.top_logprobs.len);
        try testing.expectApproxEqAbs(@as(f64, row[r.top_logprobs[0].token_id]) - lse, r.top_logprobs[0].logprob, 1e-5);
    }
    try testing.expectEqual(@as(u32, 7), out.items[0].top_logprobs[0].token_id);
    try testing.expectEqual(@as(u32, 6), out.items[0].top_logprobs[1].token_id);
    try testing.expectEqual(@as(u32, 5), out.items[1].top_logprobs[0].token_id);
    try testing.expectEqual(@as(u32, 7), out.items[1].top_logprobs[1].token_id);
    try testing.expectEqual(@as(u32, 2), out.items[2].top_logprobs[0].token_id);

    const too_many = [_]u32{ 1, 2, 3, 4 };
    try testing.expectError(error.PromptLogprobsUnavailable, promptLogprobs(allocator, logits, &too_many, 2, s, &out));
}

test "sampleToken: reported logprobs are the model's, not the client's temperature" {
    // Same prompt, same chosen token, three temperatures used to report three
    // different logprobs (-0.2129 / -0.0607 / -2.1566 at 0 / 0.6 / 2.0). The
    // value belongs to the model, so it must not move with a sampling knob.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const data = [_]f32{ 0, 1, 2, 3 };
    const shape = [_]c_int{ 1, 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // log_softmax([0,1,2,3])[3], computed independently.
    var denom: f64 = 0;
    for (data) |v| denom += @exp(@as(f64, v));
    const expect_top: f32 = @floatCast(3.0 - @log(denom));

    const temps = [_]f32{ 0.0, 0.6, 2.0 };
    for (temps) |t| {
        const sp = SamplingParams{ .temperature = t, .seed = 7 };
        const r = try sampleToken(allocator, logits, sp, null, 2, s);
        const lp = r.logprob_result orelse return error.NoLogprobs;
        defer allocator.free(lp.top_logprobs);
        try testing.expectEqual(@as(usize, 2), lp.top_logprobs.len);
        try testing.expectEqual(@as(u32, 3), lp.top_logprobs[0].token_id);
        try testing.expectEqual(@as(u32, 2), lp.top_logprobs[1].token_id);
        try testing.expectApproxEqAbs(expect_top, lp.top_logprobs[0].logprob, 1e-5);
    }
}

test "logprobs publish through the one-token delay, never straight from sampleToken" {
    // The decode loop returns `next_token_id` and, in the SAME call, forwards
    // it to sample its successor — so a path that publishes `sampleToken`'s
    // result as `last_logprob` shifts the whole array by one. Live symptom: a
    // one-token "OK" reply came back with `<|role_end|>` at rank 1, which reads
    // as a broken ranking rather than a broken pairing, and sent a real
    // quantization hunt down the wrong path for a day.
    //
    // The needles are concatenated so this test's OWN source doesn't match.
    const needle = "last_logprob" ++ " = ";
    const allowed = "last_logprob" ++ " = self.pending_logprob";
    const src = @import("../generate.zig").all_source;
    var it = std.mem.splitScalar(u8, src, '\n');
    var lineno: usize = 0;
    var seen: usize = 0;
    while (it.next()) |raw| {
        lineno += 1;
        const line = if (std.mem.indexOf(u8, raw, "//")) |c| raw[0..c] else raw;
        if (std.mem.indexOf(u8, line, needle) == null) continue;
        // `= null` is the ownership hand-off to the scheduler, not a publish.
        if (std.mem.indexOf(u8, line, "= null") != null) continue;
        seen += 1;
        if (std.mem.indexOf(u8, line, allowed) == null) {
            std.debug.print("generate.zig:{d}: logprobs published without the one-token delay: {s}\n", .{
                lineno, std.mem.trim(u8, line, " "),
            });
            return error.LogprobPairingBypass;
        }
    }
    try testing.expect(seen >= 1);
}
