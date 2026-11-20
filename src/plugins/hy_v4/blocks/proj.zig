//! Hy4-flash's projections over the pack's own quant geometry: one affine triple (U32 nibbles + bf16
//! scales/biases) or a dense pre-transposed `[in, out]` copy. The pack stores linears as `[out, packed]`, which
//! is exactly what MLX's transpose=true quantized matmul reads; a dense copy is transposed at load.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const qmatmul = @import("../../../sdk/blocks/linear/qmatmul.zig");

const Array = ops.Array;
const Graph = ops.Graph;

pub const Proj = struct {
    /// Packed `[out, in/8]` U32 nibbles, or a dense pre-transposed `[in, out]`.
    w: Array = .{},
    s: Array = .{},
    b: Array = .{},
    bits: u32 = 0,
    gs: u32 = 0,

    pub fn forward(p: *const Proj, g: *Graph, x: Array) !Array {
        const out = try qmatmul.qmatmulBits(x, p.w, p.s, p.b, p.bits, p.gs, .affine, g.s);
        return g.adopt(out);
    }
};

/// A fused expert bank: `[experts, out, in/8]` nibbles with `[experts, out, in/gs]` scales/biases.
pub const ExpertBank = struct {
    w: Array = .{},
    s: Array = .{},
    b: Array = .{},
    bits: u32 = 0,
    gs: u32 = 0,

    /// The bank's rows the `ids` name, for activations of any batch shape: `x` `[batch..., in]` with `ids`
    /// `[batch..., k]` comes back `[batch..., k, out]` (one token may hit several experts). The gather's own
    /// M axis goes in as `[batch..., 1, in]`; a dense bank reads through `gather_mm` (`[experts, in, out]`).
    pub fn gather(q: *const ExpertBank, g: *Graph, x: Array, ids: Array) !Array {
        // `gather_mm` broadcast the index axes over the activation's batch dims, whose shape must end in a
        // 1-wide axis aligned with the index axis: pad the activation with 1s until it has `ids.ndim + 2` axes
        // (mlx-lm's switch layers read `[B,T,1,1,in]` this way).
        const x_rank = ops.shape(x).len;
        const id_rank = ops.shape(ids).len;
        const want: usize = id_rank + 2 - x_rank;
        var xb = x;
        var inserted: usize = 0;
        while (inserted < want) : (inserted += 1) xb = try g.expandDims(xb, -2);
        var y = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(y);
        if (q.s.ctx == null) {
            const b = try g.transpose(q.w, &.{ 0, 2, 1 });
            try mlx.check(mlx.mlx_gather_mm(&y, xb, b, .{ .ctx = null }, ids, true, g.s));
        } else try mlx.check(mlx.mlx_gather_qmm(
            &y,
            xb,
            q.w,
            q.s,
            q.b,
            .{ .ctx = null }, // lhs indices: none
            ids,
            true, // transpose_w: the pack stores [out, packed]
            mlx.mlx_optional_int.some(@intCast(q.gs)),
            mlx.mlx_optional_int.some(@intCast(q.bits)),
            "affine",
            true, // sorted rhs indices
            g.s,
        ));
        const owned = try g.adopt(y);
        // Drop the gather's 1-wide M axis and its broadcast axis: `[batch..., k, 1, out]` -> `[batch..., k, out]`.
        const sh = ops.shape(owned);
        const out_dim = sh[sh.len - 1];
        var buf: [8]c_int = undefined;
        if (sh.len - 2 > buf.len) return error.TooManyDims;
        for (sh[0 .. sh.len - 2], 0..) |v, i| buf[i] = v;
        buf[sh.len - 2] = out_dim;
        return g.reshape(owned, buf[0 .. sh.len - 1]);
    }
};

/// SwiGLU with the MoE clamp: `silu(min(gate, limit)) * clamp(up, -limit, limit)`.
pub fn clampedSwiGLU(g: *Graph, gate: Array, up: Array, limit: f32) !Array {
    const hi = try g.scalar(limit, ops.dtype(gate));
    const lo = try g.scalar(-limit, ops.dtype(gate));
    const gated = try g.silu(try g.minimum(gate, hi));
    return g.mul(gated, try g.maximum(try g.minimum(up, hi), lo));
}

/// Plain SwiGLU (the dense MLP and the shared expert carry no clamp).
pub fn swiGLU(g: *Graph, gate: Array, up: Array) !Array {
    return g.mul(try g.silu(gate), up);
}
