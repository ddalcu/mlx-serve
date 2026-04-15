const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;

/// Per-token logprob info (OpenAI format).
pub const TokenLogprob = struct {
    token_id: u32,
    logprob: f32,
};

/// Logprob result for a single generated token.
pub const LogprobResult = struct {
    token_logprob: f32, // logprob of the chosen token
    top_logprobs: []TokenLogprob, // top N alternatives (caller must free)
};

/// Sampling parameters for token generation.
pub const SamplingParams = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0, // 0 = disabled
    repeat_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0, // 0.0 = disabled
    seed: ?u64 = null,
};

/// Generation result (for non-streaming use).
pub const GenerationResult = struct {
    text: []u8,
    token_ids: []u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    prefill_tps: f64,
    decode_tps: f64,
    logprobs: ?[]LogprobResult = null, // per-token logprobs (caller must free)
};

/// Step-based generator. Call `init` to prefill, then `next` per token.
/// Uses a fully-lazy async pipeline matching mlx-lm: sample + next forward are
/// built as a single lazy computation graph, async_eval'd together. The GPU
/// never idles between token generation steps.
pub const Generator = struct {
    xfm: *Transformer,
    tok: *const Tokenizer,
    next_token_id: u32,
    step: u32,
    max_tokens: u32,
    sampling: SamplingParams,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    done: bool,
    eos_token_ids: []const u32,
    generated_ids: std.ArrayList(u32),
    consecutive_pad: u32 = 0, // count of consecutive token-0 (pad) generations
    timeout_ns: u64, // 0 = no timeout
    timer: std.time.Timer,
    logprobs_n: u32 = 0, // 0 = disabled, >0 = number of top_logprobs to return
    last_logprob: ?LogprobResult = null, // logprob result for the most recently returned token
    // Async pipeline state: pre-computed forward pass logits for next decode step
    pending_logits: mlx.mlx_array = .{},
    has_pending_logits: bool = false,
    // Deferred token: lazy array from async pipeline, eval'd at start of next iteration
    pending_token: mlx.mlx_array = .{},
    has_pending_token: bool = false,

    /// Prefill the prompt and prepare for token-by-token generation.
    pub fn init(
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling: SamplingParams,
        eos_token_ids: []const u32,
    ) !Generator {
        const s = xfm.s;

        const ids_i32 = try allocator.alloc(i32, prompt_ids.len);
        defer allocator.free(ids_i32);
        for (prompt_ids, 0..) |id, i| {
            ids_i32[i] = @intCast(id);
        }

        // Split prefill: process first N-1 tokens (cache-only, skip lm_head eval),
        // then the last token (produces logits for sampling). This mirrors mlx-lm's
        // generate_step which avoids the expensive lm_head projection over the full
        // sequence length. For vocab_size=262144, skipping lm_head on N-1 tokens
        // avoids a [N-1, hidden] @ [hidden, 262144] matmul.
        if (prompt_ids.len > 1) {
            const prefix_len: c_int = @intCast(prompt_ids.len - 1);
            const prefix_shape = [_]c_int{ 1, prefix_len };
            const prefix_input = mlx.mlx_array_new_data(ids_i32.ptr, &prefix_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(prefix_input);

            // Forward pass on prefix — returns logits lazily (lm_head is in the
            // graph but we never eval it, so MLX skips the computation).
            const prefix_logits = try xfm.forward(prefix_input);
            _ = mlx.mlx_array_free(prefix_logits); // drop logits ref

            // Eval only cache state — materializes attention K/V without lm_head
            {
                const eval_vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(eval_vec);
                for (xfm.cache.entries) |*entry| {
                    if (!entry.initialized) continue;
                    _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                    _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
                }
                _ = mlx.mlx_eval(eval_vec);
            }
            _ = mlx.mlx_clear_cache();
        }

        // Process last token (or single token for len=1) — this applies lm_head
        // on just 1 token, producing the logits we need for sampling.
        const last_shape = [_]c_int{ 1, 1 };
        const last_idx = prompt_ids.len - 1;
        const last_input = mlx.mlx_array_new_data(&ids_i32[last_idx], &last_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(last_input);

        const logits = try xfm.forward(last_input);

        // Sample first token lazily, then build the next forward pass
        const lazy_token = sampleTokenLazy(logits, sampling, s);
        _ = mlx.mlx_array_free(logits);

        const next_logits = try lazyForward(xfm, lazy_token);

        // Async-eval the decode pipeline (single-token graphs, much smaller)
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lazy_token);
            _ = mlx.mlx_vector_array_append_value(eval_vec, next_logits);
            _ = mlx.mlx_async_eval(eval_vec);
        }

        // Sync to get the first token value
        try mlx.check(mlx.mlx_array_eval(lazy_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
        _ = mlx.mlx_array_free(lazy_token);

        var gen = Generator{
            .xfm = xfm,
            .tok = tok,
            .next_token_id = @intCast(val),
            .step = 0,
            .max_tokens = max_tokens,
            .sampling = sampling,
            .prompt_tokens = @intCast(prompt_ids.len),
            .completion_tokens = 0,
            .finish_reason = "length",
            .done = false,
            .eos_token_ids = eos_token_ids,
            .generated_ids = std.ArrayList(u32).empty,
            .timeout_ns = 0,
            .timer = try std.time.Timer.start(),
        };

        gen.pending_logits = next_logits;
        gen.has_pending_logits = true;

        return gen;
    }

    pub fn deinit(self: *Generator, allocator: std.mem.Allocator) void {
        if (self.last_logprob) |*lp| {
            allocator.free(lp.top_logprobs);
        }
        if (self.has_pending_logits) {
            _ = mlx.mlx_array_free(self.pending_logits);
            self.has_pending_logits = false;
        }
        if (self.has_pending_token) {
            _ = mlx.mlx_array_free(self.pending_token);
            self.has_pending_token = false;
        }
        self.generated_ids.deinit(allocator);
    }

    /// Resolve the deferred pending token: eval the lazy array and extract the u32 value.
    /// This is called at the START of each iteration, giving the GPU maximum time
    /// to compute since the async_eval at the END of the previous iteration.
    fn resolvePendingToken(self: *Generator) !void {
        if (!self.has_pending_token) return;
        try mlx.check(mlx.mlx_array_eval(self.pending_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, self.pending_token));
        _ = mlx.mlx_array_free(self.pending_token);
        self.has_pending_token = false;
        self.next_token_id = @intCast(val);
    }

    /// Returns the next token ID, or null when generation is finished.
    ///
    /// Pipeline architecture (matches mlx-lm's generator pattern):
    ///
    ///   The KEY to effective pipelining is the ORDER of operations:
    ///   1. Build next step's lazy graph (depends on pending lazy token)
    ///   2. async_eval the next graph — GPU computes pending token as a DEPENDENCY,
    ///      then continues with the forward pass
    ///   3. eval(pending_token) — returns INSTANTLY since GPU already computed it
    ///   4. Return the token while GPU continues computing the next forward pass
    ///
    ///   This mirrors mlx-lm's: _step(y) → async_eval(next_y) → yield y.item()
    ///   where y.item() is instant because async_eval forced y's computation.
    pub fn next(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (self.done) return null;

        // ── Phase 1: Build and submit the NEXT step FIRST ──
        // This forces the GPU to compute the pending token as a dependency,
        // so when we eval it in Phase 2, it's already ready.
        if (self.has_pending_logits and self.logprobs_n == 0 and self.step + 1 < self.max_tokens) {
            const step_logits = self.pending_logits;
            self.has_pending_logits = false;

            const lazy_token = sampleTokenLazy(step_logits, self.sampling, self.xfm.s);
            _ = mlx.mlx_array_free(step_logits);

            if (lazyForward(self.xfm, lazy_token)) |next_logits| {
                const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
                const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                _ = mlx.mlx_async_eval(vec);
                _ = mlx.mlx_vector_array_free(vec);

                // NOW resolve the pending token — GPU already computed it as a
                // dependency of the graph we just submitted. Should be instant.
                try self.resolvePendingToken();

                // Check stop conditions on the resolved token
                if (try self.checkStop()) return null;

                const token = self.next_token_id;
                self.completion_tokens += 1;
                self.step += 1;
                try self.generated_ids.append(allocator, token);

                if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

                // Store new pending state
                self.pending_token = lazy_token;
                self.has_pending_token = true;
                self.pending_logits = next_logits;
                self.has_pending_logits = true;

                return token;
            } else |_| {
                // lazyForward failed — fall through to slow path
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                self.next_token_id = @intCast(val);
                self.has_pending_token = false;
            }
        }

        // ── Phase 2: Slow path (first token, last token, logprobs, or pipeline miss) ──
        try self.resolvePendingToken();

        if (try self.checkStop()) return null;

        const token = self.next_token_id;
        self.completion_tokens += 1;
        self.step += 1;
        try self.generated_ids.append(allocator, token);

        if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

        const step_logits = if (self.has_pending_logits) blk: {
            const logits = self.pending_logits;
            self.has_pending_logits = false;
            break :blk logits;
        } else blk: {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            break :blk try self.xfm.forward(tok_input);
        };

        // Logprobs: fully synchronous
        if (self.logprobs_n > 0) {
            defer _ = mlx.mlx_array_free(step_logits);
            const result = try sampleToken(allocator, step_logits, self.sampling, self.generated_ids.items, self.logprobs_n, self.xfm.s);
            self.next_token_id = result.token_id;
            if (self.last_logprob) |*lp| allocator.free(lp.top_logprobs);
            self.last_logprob = result.logprob_result;
            if (self.step < self.max_tokens) self.startAsyncForward(result.token_id);
            return token;
        }

        // Last token or pipeline bootstrap
        const lazy_token = sampleTokenLazy(step_logits, self.sampling, self.xfm.s);
        _ = mlx.mlx_array_free(step_logits);

        if (self.step < self.max_tokens) {
            const next_logits = lazyForward(self.xfm, lazy_token) catch {
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                self.next_token_id = @intCast(val);
                return token;
            };

            const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);

            self.pending_token = lazy_token;
            self.has_pending_token = true;
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            try mlx.check(mlx.mlx_array_eval(lazy_token));
            var val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
            _ = mlx.mlx_array_free(lazy_token);
            self.next_token_id = @intCast(val);
        }

        return token;
    }

    /// Check all stop conditions. Returns true if generation should stop.
    fn checkStop(self: *Generator) !bool {
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        if (self.timeout_ns > 0 and self.timer.read() >= self.timeout_ns) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        for (self.eos_token_ids) |eos_id| {
            if (self.next_token_id == eos_id) {
                self.done = true;
                self.finish_reason = "stop";
                return true;
            }
        }
        if (self.next_token_id == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                self.done = true;
                self.finish_reason = "stop";
                return true;
            }
        } else {
            self.consecutive_pad = 0;
        }
        return false;
    }

    /// Legacy sync forward for logprobs path.
    fn startAsyncForward(self: *Generator, token_id: u32) void {
        const tok_i32: i32 = @intCast(token_id);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);

        const logits = self.xfm.forward(tok_input) catch return;
        const arr = [_]mlx.mlx_array{logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);

        self.pending_logits = logits;
        self.has_pending_logits = true;
    }
};

/// Build forward pass from a lazy sampled token array.
/// Reshapes [1] -> [1, 1] and calls transformer forward. All lazy (no eval).
fn lazyForward(xfm: *Transformer, lazy_token: mlx.mlx_array) !mlx.mlx_array {
    const tok_shape = [_]c_int{ 1, 1 };
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, lazy_token, &tok_shape, 2, xfm.s));
    return try xfm.forward(reshaped);
}

/// Sample a token lazily from logits — returns a lazy MLX array (no eval).
/// Handles temperature scaling, top-k, and top-p, but defers materialization.
/// The returned array has shape [1] with the sampled token ID.
/// Caller must free the returned array.
fn sampleTokenLazy(logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) mlx.mlx_array {
    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

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

    // Scale by 1/temperature
    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        var next = mlx.mlx_array_new();
        _ = mlx.mlx_divide(&next, current, temp_arr, s);
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Apply top-k filtering (lazy)
    if (sampling.top_k > 0) {
        var next = mlx.mlx_array_new();
        applyTopK(&next, current, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Apply top-p filtering (lazy)
    if (sampling.top_p < 1.0) {
        var next = mlx.mlx_array_new();
        applyTopP(&next, current, sampling.top_p, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Sample from categorical distribution (lazy — no eval!)
    var sampled = mlx.mlx_array_new();
    const null_key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(null_key);
    _ = mlx.mlx_random_categorical(&sampled, current, -1, null_key, s);
    _ = mlx.mlx_array_free(current);

    return sampled; // Shape [1], lazy
}

/// Convenience: generate all tokens at once (non-streaming).
pub fn generate(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    logprobs_n: u32,
) !GenerationResult {
    var timer = try std.time.Timer.start();
    var gen = try Generator.init(allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids);
    gen.timeout_ns = timeout_ns;
    gen.logprobs_n = logprobs_n;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    var logprob_results = std.ArrayList(LogprobResult).empty;
    defer {
        if (logprobs_n == 0) {
            for (logprob_results.items) |*lp| allocator.free(lp.top_logprobs);
            logprob_results.deinit(allocator);
        }
    }

    timer.reset();
    while (try gen.next(allocator)) |token_id| {
        try output_ids.append(allocator, token_id);
        if (logprobs_n > 0) {
            if (gen.last_logprob) |lp| {
                try logprob_results.append(allocator, lp);
                gen.last_logprob = null; // Transfer ownership
            }
        }
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    log.debug("Decode: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        decode_ns / std.time.ns_per_ms,
        num_decoded,
        decode_tps,
    });

    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = if (logprobs_n > 0) try logprob_results.toOwnedSlice(allocator) else null,
    };
}

/// Compute mean-pooled, L2-normalized embedding from token IDs.
/// Returns a float32 array of shape [hidden_size]. Caller must free the returned slice.
pub fn computeEmbedding(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    token_ids: []const u32,
) ![]f32 {
    const s = xfm.s;
    const prompt_len: c_int = @intCast(token_ids.len);
    const shape = [_]c_int{ 1, prompt_len };

    const ids_i32 = try allocator.alloc(i32, token_ids.len);
    defer allocator.free(ids_i32);
    for (token_ids, 0..) |id, i| {
        ids_i32[i] = @intCast(id);
    }

    const input = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(input);

    // Get hidden states: [1, seq_len, hidden_size]
    const hidden = try xfm.forwardEmbedding(input);
    defer _ = mlx.mlx_array_free(hidden);

    // Mean pool over sequence dimension (axis=1): [1, hidden_size]
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_mean_axis(&pooled, hidden, 1, false, s));

    // L2 normalize: pooled / sqrt(sum(pooled^2))
    var squared = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(squared);
    try mlx.check(mlx.mlx_multiply(&squared, pooled, pooled, s));

    var sum_sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum_sq);
    try mlx.check(mlx.mlx_sum_axis(&sum_sq, squared, -1, true, s));

    var norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm);
    try mlx.check(mlx.mlx_sqrt(&norm, sum_sq, s));

    // Add epsilon to avoid division by zero
    const eps = mlx.mlx_array_new_float(1e-12);
    defer _ = mlx.mlx_array_free(eps);
    var norm_safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm_safe);
    try mlx.check(mlx.mlx_maximum(&norm_safe, norm, eps, s));

    var normalized = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normalized);
    try mlx.check(mlx.mlx_divide(&normalized, pooled, norm_safe, s));

    // Eval and extract float data
    try mlx.check(mlx.mlx_array_eval(normalized));
    const n_shape = mlx.getShape(normalized);
    const dim: usize = @intCast(n_shape[n_shape.len - 1]);
    const data_ptr = mlx.mlx_array_data_float32(normalized) orelse return error.MlxError;

    const result = try allocator.alloc(f32, dim);
    @memcpy(result, data_ptr[0..dim]);
    return result;
}

const SampleResult = struct {
    token_id: u32,
    logprob_result: ?LogprobResult = null,
};

/// Sample a token from the last position's logits.
/// temperature <= 0.01: greedy argmax. Otherwise: scale logits, apply top_p, and sample.
/// If logprobs_n > 0, also computes logprobs for the sampled token and top N alternatives.
fn sampleToken(allocator: std.mem.Allocator, logits: mlx.mlx_array, sampling: SamplingParams, generated_ids: ?[]const u32, logprobs_n: u32, s: mlx.mlx_stream) !SampleResult {
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

    // Apply repeat penalty to already-generated tokens
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    const needs_penalty = (sampling.repeat_penalty != 1.0 or sampling.presence_penalty != 0.0);
    if (needs_penalty) {
        if (generated_ids) |ids| {
            if (ids.len > 0) {
                try applyRepeatPenalty(&penalized, current, ids, sampling.repeat_penalty, sampling.presence_penalty, s);
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
            logprob_result = try computeLogprobs(allocator, current, token_id, logprobs_n, s);
        }
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

    // For logprobs, remember the logits after temp scaling but before filtering
    const logprobs_logits = current;

    // Apply top-k filtering
    var after_topk = mlx.mlx_array_new();
    var topk_owned = false;
    defer if (topk_owned) {
        _ = mlx.mlx_array_free(after_topk);
    };

    if (sampling.top_k > 0) {
        try applyTopK(&after_topk, current, sampling.top_k, s);
        current = after_topk;
        topk_owned = true;
    }

    // Apply top-p (nucleus) sampling
    var after_topp = mlx.mlx_array_new();
    var topp_owned = false;
    defer if (topp_owned) {
        _ = mlx.mlx_array_free(after_topp);
    };

    if (sampling.top_p < 1.0) {
        try applyTopP(&after_topp, current, sampling.top_p, s);
        current = after_topp;
        topp_owned = true;
    }

    // Sample from categorical distribution
    var sampled = mlx.mlx_array_new();

    if (sampling.seed) |seed| {
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, key, s));
    } else {
        const null_key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(null_key);
        try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, null_key, s));
    }

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

    _ = mlx.mlx_array_free(sampled);
    return .{ .token_id = token_id, .logprob_result = logprob_result };
}

/// Compute log-probabilities from logits. Returns the logprob of the chosen token
/// and the top N alternatives with their token IDs and logprobs.
fn computeLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen_token: u32, top_n: u32, s: mlx.mlx_stream) !LogprobResult {
    // Compute log_softmax = log(softmax(logits)) on GPU
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, logits, -1, true, s));

    var log_probs_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs_raw);
    try mlx.check(mlx.mlx_log(&log_probs_raw, probs, s));

    // Cast to float32 for CPU readback (model may produce float16 logits)
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_astype(&log_probs, log_probs_raw, .float32, s));

    // Get top-k logprobs using mlx_topk (returns top values in descending order)
    const k: c_int = @intCast(@min(top_n + 1, 20)); // +1 to ensure chosen token is included
    var topk_vals_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_vals_raw);
    try mlx.check(mlx.mlx_topk(&topk_vals_raw, log_probs, k, s));

    var topk_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_vals);
    try mlx.check(mlx.mlx_astype(&topk_vals, topk_vals_raw, .float32, s));

    // Eval both arrays to CPU
    try mlx.check(mlx.mlx_array_eval(log_probs));
    try mlx.check(mlx.mlx_array_eval(topk_vals));

    // Read the logprob of the chosen token from the full array
    const lp_shape = mlx.getShape(log_probs);
    const vocab_size: usize = @intCast(lp_shape[lp_shape.len - 1]);
    const lp_data = mlx.mlx_array_data_float32(log_probs);
    const chosen_logprob: f32 = if (lp_data) |ptr|
        (if (chosen_token < vocab_size) ptr[chosen_token] else -100.0)
    else
        -100.0;

    // Read top-k values and find their token IDs by scanning the full logprobs
    const topk_data = mlx.mlx_array_data_float32(topk_vals);
    const actual_k: usize = @intCast(k);

    var top_logprobs = try allocator.alloc(TokenLogprob, @min(top_n, @as(u32, @intCast(actual_k))));
    var filled: usize = 0;

    if (topk_data) |tk_ptr| {
        if (lp_data) |full_ptr| {
            // Track used token IDs to avoid duplicates
            var used = std.AutoHashMap(u32, void).init(std.heap.page_allocator);
            defer used.deinit();

            // For each top-k value, find the matching token ID in the full array
            for (0..actual_k) |i| {
                if (filled >= top_n) break;
                const target_val = tk_ptr[i];
                // Find token ID with this logprob value (skip already-used IDs)
                for (0..vocab_size) |tid| {
                    if (full_ptr[tid] == target_val and !used.contains(@intCast(tid))) {
                        const tid_u32: u32 = @intCast(tid);
                        top_logprobs[filled] = .{
                            .token_id = tid_u32,
                            .logprob = target_val,
                        };
                        used.put(tid_u32, {}) catch {};
                        filled += 1;
                        break;
                    }
                }
            }
        }
    }

    // Shrink if we didn't fill all slots
    if (filled < top_logprobs.len) {
        top_logprobs = allocator.realloc(top_logprobs, filled) catch top_logprobs;
    }

    return .{
        .token_logprob = chosen_logprob,
        .top_logprobs = top_logprobs,
    };
}

/// Apply top-k filtering: keep only the top k logits, set the rest to -inf.
fn applyTopK(res: *mlx.mlx_array, logits: mlx.mlx_array, k: u32, s: mlx.mlx_stream) !void {
    // Get the top-k values (returned in descending order)
    var topk_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_vals);
    try mlx.check(mlx.mlx_topk(&topk_vals, logits, @intCast(k), s));

    // Get the minimum of the top-k values (the k-th largest) as cutoff
    var cutoff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cutoff);
    try mlx.check(mlx.mlx_min_axis(&cutoff, topk_vals, -1, true, s));

    // Mask: logits >= cutoff
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_greater_equal(&mask, logits, cutoff, s));

    // Replace masked-out logits with -inf
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask, logits, neg_inf, s));
}

/// Apply top-p (nucleus) sampling: mask logits outside the top-p probability mass.
/// Works on the original (unsorted) logits by computing which tokens to keep.
fn applyTopP(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, s: mlx.mlx_stream) !void {
    // Sort logits ascending to get sorted probabilities
    var sorted_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_logits);
    try mlx.check(mlx.mlx_sort_axis(&sorted_logits, logits, -1, s));

    // Softmax of sorted logits (ascending order: smallest probs first)
    var sorted_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_probs);
    try mlx.check(mlx.mlx_softmax_axis(&sorted_probs, sorted_logits, -1, true, s));

    // Cumulative sum from smallest to largest
    var cumsum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cumsum);
    try mlx.check(mlx.mlx_cumsum(&cumsum, sorted_probs, -1, false, true, s));

    // Find the cutoff: tokens where cumsum <= (1 - top_p) are outside the nucleus
    const threshold = mlx.mlx_array_new_float(1.0 - top_p);
    defer _ = mlx.mlx_array_free(threshold);

    var outside_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(outside_mask);
    try mlx.check(mlx.mlx_less_equal(&outside_mask, cumsum, threshold, s));

    // Set outside-nucleus logits to -inf in sorted space
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);

    // where(outside_mask, -inf, sorted_logits) — mask out the low-prob tokens
    try mlx.check(mlx.mlx_where(res, outside_mask, neg_inf, sorted_logits, s));

    // Note: categorical sampling doesn't care about token ordering,
    // but the sampled index will be in sorted space. We need to unsort.
    // Since categorical returns an index into the logits array, and we want
    // the original vocab index, we need to work in original space instead.

    // Better approach: find the minimum logit value that's in the nucleus,
    // then mask original logits below that threshold.
    _ = mlx.mlx_array_free(res.*);
    res.* = mlx.mlx_array_new();

    // The cutoff logit is the smallest logit still in the nucleus.
    // In sorted (ascending) order, tokens with cumsum > (1-top_p) are in nucleus.
    // The first such token's logit value is our threshold.
    // We can achieve this by: where(cumsum > 1-top_p, sorted_logits, +inf) then take min
    var in_nucleus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(in_nucleus);
    try mlx.check(mlx.mlx_greater(&in_nucleus, cumsum, threshold, s));

    const pos_inf = mlx.mlx_array_new_float(std.math.inf(f32));
    defer _ = mlx.mlx_array_free(pos_inf);

    var nucleus_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nucleus_logits);
    try mlx.check(mlx.mlx_where(&nucleus_logits, in_nucleus, sorted_logits, pos_inf, s));

    // Min value = the cutoff
    var min_val = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(min_val);
    try mlx.check(mlx.mlx_min_axis(&min_val, nucleus_logits, -1, true, s));

    // Mask original logits: keep if >= cutoff, else -inf
    var keep_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep_mask);
    try mlx.check(mlx.mlx_greater_equal(&keep_mask, logits, min_val, s));

    try mlx.check(mlx.mlx_where(res, keep_mask, logits, neg_inf, s));
}

/// Apply repeat penalty to already-generated tokens.
/// Uses pure MLX GPU ops — no CPU readback, preserves lazy evaluation graph.
fn applyRepeatPenalty(res: *mlx.mlx_array, logits: mlx.mlx_array, generated_ids: []const u32, repeat_penalty: f32, presence_penalty: f32, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);

    // Collect unique token ids
    var seen_set = std.AutoHashMap(u32, void).init(std.heap.page_allocator);
    defer seen_set.deinit();
    for (generated_ids) |id| {
        if (id < vocab_size) {
            seen_set.put(id, {}) catch continue;
        }
    }

    if (seen_set.count() == 0) return;

    // Build boolean mask: true at positions of seen tokens
    const mask_data = try std.heap.page_allocator.alloc(u8, vocab_size);
    defer std.heap.page_allocator.free(mask_data);
    @memset(mask_data, 0);

    var it = seen_set.keyIterator();
    while (it.next()) |id_ptr| {
        mask_data[id_ptr.*] = 1;
    }

    const arr_shape = [_]c_int{ 1, @intCast(vocab_size) };
    const mask_arr = mlx.mlx_array_new_data(mask_data.ptr, &arr_shape, 2, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);

    var current = logits;

    // Repeat penalty: multiply seen tokens by 1/penalty (positive) or penalty (negative)
    // This is equivalent to: where(mask & logits > 0, logits / penalty, where(mask, logits * penalty, logits))
    // Simplified: where(mask, where(logits > 0, logits / penalty, logits * penalty), logits)
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    if (repeat_penalty != 1.0) {
        const rp = mlx.mlx_array_new_float(repeat_penalty);
        defer _ = mlx.mlx_array_free(rp);
        const inv_rp = mlx.mlx_array_new_float(1.0 / repeat_penalty);
        defer _ = mlx.mlx_array_free(inv_rp);
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);

        // positive_mask = logits > 0
        var positive_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(positive_mask);
        try mlx.check(mlx.mlx_greater(&positive_mask, current, zero, s));

        // penalized_positive = logits * (1/penalty)
        var pen_pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_pos);
        try mlx.check(mlx.mlx_multiply(&pen_pos, current, inv_rp, s));

        // penalized_negative = logits * penalty
        var pen_neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_neg);
        try mlx.check(mlx.mlx_multiply(&pen_neg, current, rp, s));

        // sign_selected = where(positive, logits/penalty, logits*penalty)
        var sign_selected = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sign_selected);
        try mlx.check(mlx.mlx_where(&sign_selected, positive_mask, pen_pos, pen_neg, s));

        // result = where(mask, sign_selected, logits)
        try mlx.check(mlx.mlx_where(&penalized, mask_arr, sign_selected, current, s));
        current = penalized;
        penalized_owned = true;
    }

    // Presence penalty: subtract from seen tokens
    if (presence_penalty != 0.0) {
        const pp = mlx.mlx_array_new_float(presence_penalty);
        defer _ = mlx.mlx_array_free(pp);

        // Cast mask to float for arithmetic
        var mask_float = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_float);
        try mlx.check(mlx.mlx_astype(&mask_float, mask_arr, .float16, s));

        // subtract = mask * presence_penalty
        var subtract = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(subtract);
        try mlx.check(mlx.mlx_multiply(&subtract, mask_float, pp, s));

        // result = current - subtract
        try mlx.check(mlx.mlx_subtract(res, current, subtract, s));
    } else {
        try mlx.check(mlx.mlx_copy(res, current, s));
    }
}

/// Greedy argmax over the last axis.
fn argmax(last_logits: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var argmax_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(argmax_arr);
    try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, last_logits, -1, false, s));

    try mlx.check(mlx.mlx_array_eval(argmax_arr));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, argmax_arr));

    return @intCast(val);
}

// ── Tests ──

const testing = std.testing;

test "SamplingParams defaults" {
    const params = SamplingParams{};
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 0), params.top_k);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.repeat_penalty, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), params.presence_penalty, 0.001);
    try testing.expect(params.seed == null);
}

test "SamplingParams custom values" {
    const params = SamplingParams{
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
        .repeat_penalty = 1.1,
        .presence_penalty = 0.5,
        .seed = 42,
    };
    try testing.expectApproxEqAbs(@as(f32, 0.7), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.9), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 40), params.top_k);
    try testing.expectEqual(@as(u64, 42), params.seed.?);
}

test "GenerationResult fields" {
    // Just verifying the struct shape compiles correctly with all fields
    const result = GenerationResult{
        .text = @constCast("hello"),
        .token_ids = @constCast(&[_]u32{ 1, 2, 3 }),
        .prompt_tokens = 10,
        .completion_tokens = 3,
        .finish_reason = "stop",
        .prefill_tps = 100.0,
        .decode_tps = 35.0,
        .logprobs = null,
    };
    try testing.expectEqual(@as(u32, 10), result.prompt_tokens);
    try testing.expectEqual(@as(u32, 3), result.completion_tokens);
    try testing.expectEqualStrings("stop", result.finish_reason);
    try testing.expect(result.logprobs == null);
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

test "applyRepeatPenalty reduces seen token logits" {
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

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 0.0, s);
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

test "applyRepeatPenalty with negative logits" {
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

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 0.0, s);
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

test "applyRepeatPenalty presence penalty" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 2 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 1.0, 0.5, s);
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

test "applyRepeatPenalty combined penalties" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 2.0, -1.0, 3.0 };
    const shape = [_]c_int{ 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 1 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 1.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: positive, divide by 2.0 = 1.0, then - 1.0 = 0.0
    try testing.expectApproxEqAbs(@as(f32, 0.0), res_data[0], 0.01);
    // Index 1: negative, multiply by 2.0 = -2.0, then - 1.0 = -3.0
    try testing.expectApproxEqAbs(@as(f32, -3.0), res_data[1], 0.01);
    // Index 2: unseen → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
}

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
