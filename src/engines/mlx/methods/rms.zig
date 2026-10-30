//! engines/mlx/methods/rms: `Transformer` methods (aliased from engines/mlx/transformer.zig).
//! Owns rmsNormEps, rhtAddNormRotate, narrowAddNorm, rmsNorm, layerNorm.

const mlx = @import("../../../mlx/ffi.zig");
const rht = @import("../../../sdk/quant/hadamard/rht.zig");
const add_norm = @import("../../../sdk/blocks/norms/add_norm.zig");
const residual_norm = @import("../../../sdk/blocks/norms/residual_norm.zig");
const Transformer = @import("../transformer.zig").Transformer;
const groupedRmsNorm = @import("../../../sdk/blocks/norms/grouped_rms.zig").groupedRmsNorm;

/// rmsNorm with an explicit epsilon — muse_glimmer's post-attention /
/// post-feedforward norms run at post_norm_eps (1e-8) while every other
/// norm keeps rms_norm_eps.
pub inline fn rmsNormEps(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, eps: f32) !mlx.mlx_array {
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&result, x, w, eps, self.s));
    return result;
}

/// Hadamard packs: `sum = h + delta` and `rms_norm(sum, w)` with its input
/// rotation in one dispatch; the rotation is seeded so the projections that
/// read `normed` skip their own. Without `want_normed`, `normed` IS the
/// rotated array (only rotated projections read it). Null = not a Hadamard
/// pack or a shape the kernel declines.
pub fn rhtAddNormRotate(self: *Transformer, h: mlx.mlx_array, delta: mlx.mlx_array, w: mlx.mlx_array, want_normed: bool) !?struct { sum: mlx.mlx_array, normed: mlx.mlx_array } {
    const reg = self.rht orelse return null;
    const hidden: c_int = @intCast(self.config.hidden_size);
    const signs = reg.signsFor(hidden) orelse return null;
    const r = (try rht.normRotate(h, delta, w, self.rms_eps_arr, signs, reg.block, want_normed, self.s)) orelse return null;
    const key = r.normed orelse r.rot;
    try reg.seed(key, signs, r.rot);
    if (r.normed != null) _ = mlx.mlx_array_free(r.rot);
    return .{ .sum = r.sum.?, .normed = key };
}

/// `h + delta` and its RMSNorm in one dispatch at decode/verify widths,
/// bit-equal to the two ops (`add_norm.zig`). Null: the caller keeps them.
pub fn narrowAddNorm(self: *const Transformer, h: mlx.mlx_array, delta: mlx.mlx_array, w: mlx.mlx_array) !?add_norm.Out {
    if (self.config.norm_groups > 1 or w.ctx == null) return null;
    const sh = mlx.getShape(h);
    if (sh.len != 3 or sh[0] * sh[1] > residual_norm.add_norm_max_rows) return null;
    return add_norm.addNorm(h, delta, w, self.rms_eps_arr, self.s);
}

pub inline fn rmsNorm(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
    if (self.config.norm_groups > 1) {
        // Only the residual-width norms are grouped; a per-head q/k norm
        // is already one group per head.
        const shape = mlx.getShape(x);
        if (shape[shape.len - 1] == @as(c_int, @intCast(self.config.hidden_size))) {
            return groupedRmsNorm(x, w, self.config.norm_groups, self.config.rms_norm_eps, @constCast(&self.grouped_norm_ones), self.s);
        }
    }
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&result, x, w, self.config.rms_norm_eps, self.s));
    return result;
}

pub inline fn layerNorm(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&result, x, w, b, self.config.layer_norm_eps, self.s));
    return result;
}
