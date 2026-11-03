//! sdk/blocks/rope/mrope.
//! Owns MropeCosSin, and 1 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const mrope = @import("../../vision/mrope.zig");

// ── Full Attention for MoE models (with optional output gate) ──

pub const MropeCosSin = struct { cos: mlx.mlx_array, sin: mlx.mlx_array };

