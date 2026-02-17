const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;

/// Generation result (for non-streaming use).
pub const GenerationResult = struct {
    text: []u8,
    token_ids: []u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    prefill_tps: f64,
    decode_tps: f64,
};

/// Step-based generator. Call `init` to prefill, then `next` per token.
pub const Generator = struct {
    xfm: *Transformer,
    tok: *const Tokenizer,
    next_token_id: u32,
    step: u32,
    max_tokens: u32,
    temperature: f32,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    done: bool,
    eos_token_ids: []const u32,

    /// Prefill the prompt and prepare for token-by-token generation.
    pub fn init(
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        temperature: f32,
        eos_token_ids: []const u32,
    ) !Generator {
        const s = xfm.s;

        // Create input tensor for prefill: [1, seq_len]
        const prompt_len: c_int = @intCast(prompt_ids.len);
        const shape = [_]c_int{ 1, prompt_len };

        const ids_i32 = try allocator.alloc(i32, prompt_ids.len);
        defer allocator.free(ids_i32);
        for (prompt_ids, 0..) |id, i| {
            ids_i32[i] = @intCast(id);
        }

        const input = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(input);

        const logits = try xfm.forward(input);
        const first_token = try sampleToken(logits, temperature, s);
        _ = mlx.mlx_array_free(logits);

        return .{
            .xfm = xfm,
            .tok = tok,
            .next_token_id = first_token,
            .step = 0,
            .max_tokens = max_tokens,
            .temperature = temperature,
            .prompt_tokens = @intCast(prompt_ids.len),
            .completion_tokens = 0,
            .finish_reason = "length",
            .done = false,
            .eos_token_ids = eos_token_ids,
        };
    }

    /// Returns the next token ID, or null when generation is finished.
    pub fn next(self: *Generator) !?u32 {
        if (self.done) return null;

        // Check stop conditions
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }

        // Check all EOS tokens
        for (self.eos_token_ids) |eos_id| {
            if (self.next_token_id == eos_id) {
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        }

        const token = self.next_token_id;
        self.completion_tokens += 1;
        self.step += 1;

        // Forward pass for next prediction
        const tok_i32: i32 = @intCast(token);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);

        const step_logits = try self.xfm.forward(tok_input);
        defer _ = mlx.mlx_array_free(step_logits);
        self.next_token_id = try sampleToken(step_logits, self.temperature, self.xfm.s);

        return token;
    }
};

/// Convenience: generate all tokens at once (non-streaming).
pub fn generate(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    temperature: f32,
    eos_token_ids: []const u32,
) !GenerationResult {
    var timer = try std.time.Timer.start();
    var gen = try Generator.init(allocator, xfm, tok, prompt_ids, max_tokens, temperature, eos_token_ids);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    std.debug.print("Prefill: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();
    while (try gen.next()) |token_id| {
        try output_ids.append(allocator, token_id);
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    std.debug.print("Decode: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
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
    };
}

/// Sample a token from the last position's logits.
/// temperature <= 0.01: greedy argmax. Otherwise: scale logits and sample.
fn sampleToken(logits: mlx.mlx_array, temperature: f32, s: mlx.mlx_stream) !u32 {
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

    // Greedy if temperature is ~0
    if (temperature < 0.01) {
        return argmax(last_logits, s);
    }

    // Scale logits by 1/temperature
    const temp_arr = mlx.mlx_array_new_float(temperature);
    defer _ = mlx.mlx_array_free(temp_arr);

    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_divide(&scaled, last_logits, temp_arr, s));

    // Sample from categorical distribution (logits are unnormalized log-probs)
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);
    const null_key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(null_key);
    try mlx.check(mlx.mlx_random_categorical(&sampled, scaled, -1, null_key, s));

    // Eval and extract
    try mlx.check(mlx.mlx_array_eval(sampled));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, sampled));

    return @intCast(val);
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
