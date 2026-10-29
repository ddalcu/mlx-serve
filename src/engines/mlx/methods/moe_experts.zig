//! The legacy `Transformer`'s methods over `sdk/blocks/moe/experts.zig` (what reads the engine's model state).

const mlx = @import("../../../mlx/ffi.zig");
const Transformer = @import("../transformer.zig").Transformer;

/// Scale a gather_qmm result by a per-expert scale, in place. `sel` MUST be the index vector the matmul used, in
/// its order (`sorted_inds` on the sorted path).
pub fn scaleExperts(self: *const Transformer, out: *mlx.mlx_array, scale: mlx.mlx_array, sel: mlx.mlx_array) !void {
    if (scale.ctx == null) return;
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try mlx.check(mlx.mlx_take_axis(&picked, scale, sel, 0, self.s));
    var col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_expand_dims(&col, picked, -1, self.s));
    var scaled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&scaled, out.*, col, self.s));
    _ = mlx.mlx_array_free(out.*);
    out.* = scaled;
}
