//! The GGUF format: GGUF files served as is on MLX, their raw ggml blocks read by our own Metal kernels.
const core = @import("core.zig");
pub const gguf = core.gguf;
pub const quants = core.quants;
pub const meta = core.meta;
pub const kernels = @import("kernels.zig");
pub const weights = @import("weights.zig");

test {
    _ = core;
    _ = kernels;
    _ = weights;
}
