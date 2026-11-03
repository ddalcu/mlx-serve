//! sdk/blocks/rope/rope.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const MropeCosSin = @import("mrope.zig").MropeCosSin;

/// Llama 3.x `rope_type: "llama3"` (HF `_compute_llama3_parameters`).
pub const Llama3Rope = struct {
    factor: f32,
    low_freq_factor: f32 = 1.0,
    high_freq_factor: f32 = 4.0,
    original_max_position_embeddings: f32 = 8192.0,
};
