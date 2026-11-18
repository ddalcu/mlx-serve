//! engines/mlx/warmup.
//! Owns warmup.

const mlx = @import("../../mlx/ffi.zig");
const Transformer = @import("transformer.zig").Transformer;

/// Pre-fault weight pages and trigger first-touch kernel compiles before
/// the first real request so cold prefill doesn't pay 800+ms of GPU page
/// faulting (measured on Gemma 4 E4B 4-bit). Runs three forward passes:
///   1. [1, 1] decode-shape: faults embed matrix + compiles decode kernel
///   2. [1, 8] prefill-shape: compiles short-prefill kernel
/// then resets the cache so the first real request starts from clean state.
/// Idempotent — calling twice is wasted work but not incorrect.
/// `verify_rows` is the widest block a speculative round verifies, 0 on a load that leaves the sweep to its idle
/// warm-up or owes none (`warm_requests.owed`).
pub fn warmup(self: *Transformer, verify_rows: u32) !void {
    const dummy_id: i32 = 0; // BOS-ish placeholder; the actual id doesn't matter for warmup
    const decode_shape = [_]c_int{ 1, 1 };
    const decode_input = mlx.mlx_array_new_data(&dummy_id, &decode_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(decode_input);
    const decode_logits = try self.forward(decode_input);
    _ = mlx.mlx_array_free(decode_logits);
    // Materialize the cache update so subsequent forwards see initialized entries.
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        for (self.cache.entries) |*entry| {
            if (!entry.initialized) continue;
            _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
            _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
        }
        _ = mlx.mlx_eval(eval_vec);
    }
    _ = mlx.mlx_clear_cache();

    // Reset before the prefill-shape pass so we exercise the cold-init path,
    // not the partial-cache path.
    try self.resetCache();

    // A verify-width pass, then a prompt-width one (past 16 rows the lane matmul runs 32-row blocks).
    const ids: [32]i32 = @splat(0);
    for ([_]c_int{ 8, 32 }) |width| {
        const prefill_shape = [_]c_int{ 1, width };
        const prefill_input = mlx.mlx_array_new_data(&ids, &prefill_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(prefill_input);
        const prefill_logits = try self.forward(prefill_input);
        _ = mlx.mlx_array_free(prefill_logits);
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            for (self.cache.entries) |*entry| {
                if (!entry.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
            }
            _ = mlx.mlx_eval(eval_vec);
        }
        _ = mlx.mlx_clear_cache();
        try self.resetCache();
    }

    for (0..verify_rows) |i| try sweepWidth(self, @intCast(i + 1));
}

/// One width of the verify sweep: a kernel sized by its row count compiles one variant a width, so `width` rows run
/// once on a cache that already holds a prompt, as a round's does.
pub fn sweepWidth(self: *Transformer, width: u32) !void {
    const ids: [32]i32 = @splat(0);
    for ([_]u32{ 8, @min(width, ids.len) }) |rows| {
        const shape = [_]c_int{ 1, @intCast(rows) };
        const input = mlx.mlx_array_new_data(&ids, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(input);
        const logits = try self.forward(input);
        defer _ = mlx.mlx_array_free(logits);
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        _ = mlx.mlx_vector_array_append_value(eval_vec, logits);
        for (self.cache.entries) |*entry| {
            if (!entry.initialized) continue;
            _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
            _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
        }
        _ = mlx.mlx_eval(eval_vec);
    }
    _ = mlx.mlx_clear_cache();
    try self.resetCache();
}
