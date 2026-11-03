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
pub fn warmup(self: *Transformer) !void {
    if (self.mlx_stream_ext != null) return; // mlx-stream warms its own kernels at construction
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
}
