//! The hyper-connection block's and the PLE layer's loaded weights (`HcWeights`, `PleWeights`).

const mlx = @import("../../mlx/ffi.zig");

/// qwen4_exp gated residual ("hyper connection"). `norm_w` is the grouped
/// hc_norm weight viewed as [hc, hidden]; `inject_*` is absent on the final
/// mixer (`use_combine=False`).
pub const HcWeights = struct {
    norm_w: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    inject_w: mlx.mlx_array = .{ .ctx = null },
    inject_s: mlx.mlx_array = .{ .ctx = null },
    inject_b: mlx.mlx_array = .{ .ctx = null },
    /// Dense inject weight materialized row-major `[hc*hidden, hc]` for the
    /// fused decode kernel (the qmatmul path reads the lazy transpose view).
    inject_flat: mlx.mlx_array = .{ .ctx = null },
    /// Dense bf16 down/up as the fused read binds them: u32 words of the row-major `[out, in]`
    /// buffers the chain reads through transposed views.
    down_rows: mlx.mlx_array = .{ .ctx = null },
    up_rows: mlx.mlx_array = .{ .ctx = null },
};

/// qwen4_exp PLE: n-gram embedding → per-stream keys + one value, gated by
/// the normalized streams, plus a dilated depthwise short conv.
pub const PleWeights = struct {
    key_w: mlx.mlx_array,
    key_s: mlx.mlx_array,
    key_b: mlx.mlx_array,
    value_w: mlx.mlx_array,
    value_s: mlx.mlx_array,
    value_b: mlx.mlx_array,
    norm_key: mlx.mlx_array, // [hc, hidden]
    norm_query: mlx.mlx_array,
    norm_conv: mlx.mlx_array,
    conv_w: mlx.mlx_array, // [hc*hidden, K, 1]
};
