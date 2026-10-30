//! copied_from: sdk/blocks/linear/qmatmul.zig (`lmHeadProject`) and engines/mlx/forward/encoder_head.zig
//! (`denseHead`, `embedProjection`). The output projections: the LM head and the embedding projection head.

const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const Linear = @import("linear.zig").Linear;
const row_exact = @import("row_exact.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;

/// `x @ w^T` for a head kept in its checkpoint layout: a dense `[out, in]` weight (also the tied embedding table,
/// so never pre-transposed) projects through a transposed view, a quantized one as any projection.
pub fn project(g: *Graph, w: *const Linear, x: Array) !Array {
    if (w.s.ctx != null) return w.forward(g, x);
    var t = g.scope();
    defer t.deinit();
    const sh = ops.shape(w.w);
    var perm: [8]c_int = undefined;
    for (0..sh.len) |i| perm[i] = @intCast(i);
    perm[sh.len - 2] = @intCast(sh.len - 1);
    perm[sh.len - 1] = @intCast(sh.len - 2);
    return g.keep(try t.matmul(x, try t.transpose(w.w, perm[0..sh.len])));
}

/// The LM head: hidden rows to logits.
pub const LmHead = struct {
    pub const kind: block.Kind = .head;
    pub const supports_batched_rows = true;
    pub const Weights = Linear;

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        if (w.s.ctx != null) return row_exact.linear(c.row_exact, c.g, w, x);
        return project(c.g, w, x);
    }
};

/// The sentence-transformers projection over pooled rows `[B, H]`: bf16 in, one or two dense layers with identity
/// activations, f32 out (the normalization that follows squares and sums it).
pub const Projection = struct {
    first: Linear,
    second: ?Linear = null,

    pub fn forward(p: *const Projection, g: *Graph, pooled: Array) !Array {
        var t = g.scope();
        defer t.deinit();
        const mid = try project(&t, &p.first, try t.astype(pooled, .bfloat16));
        const out = if (p.second) |*s| try project(&t, s, mid) else mid;
        return g.keep(try t.astype(out, .float32));
    }
};
