//! GLM-5.3-Flash's and Ling's MoE layers run on the shared `Moe` block (weights built by `trunk.zig`). A model's own
//! one-kernel decode is the block's `fused` arm, given at load (`trunk.Hooks.fused_decode`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const moe = @import("../blocks/moe.zig");

const Array = mlx.mlx_array;

/// The block over `x` (owned by the caller).
pub fn forward(w: *const moe.Moe.Weights, s: mlx.mlx_stream, x: Array) !Array {
    var g = ops.Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    return g.escape(try moe.mlp(w, &g, x));
}

/// `forward` with each row given its own one-row call's bytes (`moe.mlpRows`).
pub fn forwardRows(w: *const moe.Moe.Weights, s: mlx.mlx_stream, x: Array) !Array {
    var g = ops.Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    return g.escape(try moe.mlpRows(w, &g, x));
}
