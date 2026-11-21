//! Inkling 2's Metal kernels: relative-logit prefill attention and the depth (block) attention read. Every wrapper
//! returns null when it cannot serve the call; the caller then runs the composed path
//! (`attention.referenceAttention`, `depth.attendComposed`), which stays the reference.

const ops = @import("../../mlx/ops.zig");

const Array = ops.Array;
const Graph = ops.Graph;

/// q bf16 [1,H,L,hd]; k,v bf16 [1,Hk,S,hd]; pb bf16 [1,L,H,R]; sinks [1,L,H] (cast to f32 inside);
/// key_start = S - total (<= 0). Returns bf16 [1,L,H,hd] owned by g, or null = use referenceAttention.
pub fn relativeAttention(g: *Graph, q: Array, k: Array, v: Array, pb: Array, sinks: Array, scale: f32, window: u32, key_start: i32) !?Array {
    _ = .{ g, q, k, v, pb, sinks, scale, window, key_start };
    return null;
}

/// blocks f32 [N,...,D] or null; partial f32 [...,D]; queries f32 [Q,D]. Returns f32 [...,Q,D] or null = attendComposed.
pub fn depthAttention(g: *Graph, blocks: ?Array, partial: Array, queries: Array, scale: f32) !?Array {
    _ = .{ g, blocks, partial, queries, scale };
    return null;
}
