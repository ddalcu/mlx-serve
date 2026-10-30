//! engines/mlx/modules/shell.
//! Owns moduleShell.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const KVCache = @import("../kv/cache.zig").KVCache;
const bf16Scalar = @import("../load/weights.zig").bf16Scalar;
const Transformer = @import("../transformer.zig").Transformer;

/// A Transformer whose every standard field is empty: the arch lives in its
/// own module, which the caller sets.
pub fn moduleShell(allocator: std.mem.Allocator, config: ModelConfig, s: mlx.mlx_stream) !Transformer {
    const cache = try KVCache.init(allocator, 0);
    return .{
        .config = config,
        .cache = cache,
        .s = s,
        .allocator = allocator,
        .emb_w = mlx.mlx_array_new(),
        .emb_s = mlx.mlx_array_new(),
        .emb_b = mlx.mlx_array_new(),
        .emb_scale = null,
        .lm_head_w = mlx.mlx_array_new(),
        .lm_head_s = mlx.mlx_array_new(),
        .lm_head_b = mlx.mlx_array_new(),
        .owns_lm_head = false,
        .gelu_coeff = bf16Scalar(0.7978845608028654, s),
        .gelu_inner = bf16Scalar(0.044715, s),
        .half = bf16Scalar(0.5, s),
        .one = bf16Scalar(1.0, s),
        .rms_eps_arr = mlx.mlx_array_new_float(config.rms_norm_eps),
        .post_eps_arr = mlx.mlx_array_new_float(config.postNormEps()),
        .three = bf16Scalar(3.0, s),
        .moe_layers = null,
        .ssm_entries = null,
        .moe_seq_offset = 0,
        .prompt_cache = null,
    };
}
