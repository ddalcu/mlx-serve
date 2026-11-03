//! The legacy `Transformer`'s methods over `sdk/quant/hadamard.zig` (what reads the engine's model state).

const mlx = @import("../../../mlx/ffi.zig");
const rht = @import("../../../sdk/quant/hadamard/rht.zig");
const Transformer = @import("../transformer.zig").Transformer;
const MoeLayerWeights = @import("../layer_weights.zig").MoeLayerWeights;

/// Hadamard packs: gathered embedding rows back to the residual basis.
/// Takes ownership of `rows`.
pub fn unrotateEmbedding(self: *const Transformer, rows: mlx.mlx_array) !mlx.mlx_array {
    const reg = self.rht orelse return rows;
    const signs = reg.get(self.emb_w) orelse return rows;
    defer _ = mlx.mlx_array_free(rows);
    return rht.transform(rows, signs, reg.block, true, self.s);
}

/// Hadamard packs: a row-joined group reads ONE rotation of its input, so it
/// is served only when its parts share a sign vector; else the parts' views are.
pub fn aliasFusedRotations(reg: *rht.Registry, layers: []MoeLayerWeights) !void {
    for (layers) |*lw| {
        if (lw.attn == .linear) {
            const la = &lw.attn.linear;
            if (la.in.w.ctx != null and !try reg.alias(la.in.w, (&[_]mlx.mlx_array{ la.qkv_w, la.z_w, la.a_w, la.b_w })[0..la.in.count])) la.in.w = .{ .ctx = null };
        }
        if (lw.mlp == .dense) {
            const dw = &lw.mlp.dense;
            if (dw.gu.w.ctx != null and !try reg.alias(dw.gu.w, &.{ dw.gate_w, dw.up_w })) dw.gu.w = .{ .ctx = null };
        }
    }
}
