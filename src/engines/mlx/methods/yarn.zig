//! The legacy `Transformer`'s methods over `sdk/blocks/rope/yarn.zig` (what reads the engine's model state).

const mlx = @import("../../../mlx/ffi.zig");
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const Transformer = @import("../transformer.zig").Transformer;
const constTableAs = @import("../../../mlx/dtype.zig").constTableAs;
const mrope = @import("../../../sdk/vision/mrope.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;

/// Does this trunk rotate with a YaRN-scaled rope? The DECISION comes from
/// the config — every Transformer has one, including the hand-built fixtures
/// in the tests, which leave the optionals below undefined. The short-circuit
/// means an unscaled model never reads them.
pub fn yarnActive(self: *const Transformer) bool {
    return self.config.rope_yarn and self.rope_freqs_yarn != null;
}

/// YaRN mscale on rope'd q and k: multiply by the [head_dim] table
/// (attention_factor on the rotated slice, 1.0 on the pass-through tail).
/// That is the reference's `cos, sin = cos*af, sin*af` — every rotary pair
/// shares one frequency and both of its dims carry it, so scaling the
/// rotation and scaling q/k are the same product. The table MUST be read in
/// the tensors' dtype: an f32 operand promotes the result, and the f32 then
/// rides SDPA → o_proj → residual, upcasting every weight read downstream.
pub fn yarnScaleQK(self: *Transformer, q: *mlx.mlx_array, k: *mlx.mlx_array) !void {
    const ms = try constTableAs(self.yarn_mscale.?, mlx.mlx_array_dtype(q.*), &self.yarn_mscale_cast, self.s);
    var qs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&qs, q.*, ms, self.s));
    _ = mlx.mlx_array_free(q.*);
    q.* = qs;
    var ks = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&ks, k.*, ms, self.s));
    _ = mlx.mlx_array_free(k.*);
    k.* = ks;
}

/// YaRN mscale on a rope'd array whose last dim may not be `head_dim`
/// (the QSA indexer is 128-wide / 64 rotated). Slice the first `rope_dims`,
/// multiply by a same-dtype scalar, concat the pass-through tail. NEVER
/// `yarnScaleQK` here: that table is `[head_dim]` (256 on qwen4).
pub fn yarnScaleRotated(self: *Transformer, q: *mlx.mlx_array, rope_dims: c_int) !void {
    const sh = mlx.getShape(q.*);
    const dt = mlx.mlx_array_dtype(q.*);
    const st = [_]c_int{ 1, 1, 1, 1 };
    var rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot);
    try mlx.check(mlx.mlx_slice(&rot, q.*, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], rope_dims }, 4, &st, 4, self.s));
    var pass = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pass);
    try mlx.check(mlx.mlx_slice(&pass, q.*, &[_]c_int{ 0, 0, 0, rope_dims }, 4, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, 4, &st, 4, self.s));
    const ms = try scalarOf(self.config.yarn_attention_factor, dt, self.s);
    defer _ = mlx.mlx_array_free(ms);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, rot, ms, self.s));
    const parts = [_]mlx.mlx_array{ scaled, pass };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, -1, self.s));
    _ = mlx.mlx_array_free(q.*);
    q.* = out;
}

/// This checkpoint's YaRN spectrum: the config's rope geometry (theta, the
/// rotating slice, the pre-trained window) resolved into `mrope.Yarn`, the one
/// implementation of HF's `_compute_yarn_parameters` in the engine. Only
/// meaningful when `config.rope_yarn`.
pub fn yarnSpec(config: *const ModelConfig) mrope.Yarn {
    return .{
        .theta = config.rope_theta,
        .factor = config.yarn_factor,
        .orig_max = config.yarn_orig_max_pos,
        .rotary_dim = config.yarnRotaryDim(),
        .beta_fast = config.yarn_beta_fast,
        .beta_slow = config.yarn_beta_slow,
        .truncate = config.yarn_truncate,
    };
}

/// The mscale the host cos/sin tables are multiplied by: HF's `attention_factor`
/// when the checkpoint scales its rope, 1.0 (a no-op) when it does not.
pub fn yarnMscale(config: *const ModelConfig) f64 {
    return if (config.rope_yarn) config.yarn_attention_factor else 1.0;
}
