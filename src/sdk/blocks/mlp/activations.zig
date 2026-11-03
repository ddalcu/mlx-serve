//! sdk/blocks/mlp/activations.
//! Owns and 10 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const fused_activations = @import("fused_activations.zig");
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const all16BitPatterns = @import("fused_activations.zig").all16BitPatterns;
const fusedClampedSwiGLU = @import("fused_activations.zig").fusedClampedSwiGLU;
const fusedGeluErfMul = @import("fused_activations.zig").fusedGeluErfMul;
const fusedSwiGLU = @import("fused_activations.zig").fusedSwiGLU;
const geluErfChain = @import("fused_activations.zig").geluErfChain;
const swigluFusedEnabled = @import("fused_activations.zig").swigluFusedEnabled;
const tableGateMul = @import("fused_activations.zig").tableGateMul;

// ── Activation functions ──

pub fn reluSquaredOp(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const zero = bf16Scalar(0.0, s);
    defer _ = mlx.mlx_array_free(zero);
    var relu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(relu);
    try mlx.check(mlx.mlx_maximum(&relu, x, zero, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_square(&out, relu, s));
    return out;
}

/// gpt_oss clamped SwiGLU (mlx-lm `gpt_oss.swiglu`):
///
///     x_glu    = clip(gate, max=limit)        // asymmetric — no lower clip
///     x_linear = clip(up, -limit, +limit)
///     out      = x_glu * sigmoid(alpha * x_glu) * (x_linear + 1)
///
/// The `+ 1` on the linear branch is part of the trained function, not a
/// numerical guard — dropping it changes every expert's output. Scalars are
/// built in the activation's own dtype so a bf16 stream is not promoted to f32
/// for the rest of the layer.
/// GLM-5-Next's SwiGLU: silu(min(gate, limit)) * clip(up, ±limit), in the activation dtype.
pub fn clampedSwiGLU(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array, limit: f32) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(gate);
    const hi = try scalarOf(limit, dt, s);
    defer _ = mlx.mlx_array_free(hi);
    const lo = try scalarOf(-limit, dt, s);
    defer _ = mlx.mlx_array_free(lo);
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_minimum(&g, gate, hi, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_clip(&u, up, lo, hi, s));
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_multiply(&act, g, sig, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_multiply(&out, act, u, s));
    return out;
}

pub fn gptOssSwiGLU(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array, limit: f32, alpha: f32) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(gate);

    const lim_f = mlx.mlx_array_new_float(limit);
    defer _ = mlx.mlx_array_free(lim_f);
    var lim = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lim);
    try mlx.check(mlx.mlx_astype(&lim, lim_f, dt, s));

    const neg_lim_f = mlx.mlx_array_new_float(-limit);
    defer _ = mlx.mlx_array_free(neg_lim_f);
    var neg_lim = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_lim);
    try mlx.check(mlx.mlx_astype(&neg_lim, neg_lim_f, dt, s));

    const alpha_f = mlx.mlx_array_new_float(alpha);
    defer _ = mlx.mlx_array_free(alpha_f);
    var alpha_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(alpha_a);
    try mlx.check(mlx.mlx_astype(&alpha_a, alpha_f, dt, s));

    const one_f = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one_f);
    var one = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_astype(&one, one_f, dt, s));

    // x_glu = min(gate, limit) — upper clip only.
    var x_glu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_glu);
    try mlx.check(mlx.mlx_minimum(&x_glu, gate, lim, s));

    // x_linear = clip(up, -limit, +limit)
    var x_lin = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_lin);
    try mlx.check(mlx.mlx_clip(&x_lin, up, neg_lim, lim, s));

    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, x_glu, alpha_a, s));
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, scaled, s));

    var out_glu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_glu);
    try mlx.check(mlx.mlx_multiply(&out_glu, x_glu, sig, s));

    var lin_plus1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lin_plus1);
    try mlx.check(mlx.mlx_add(&lin_plus1, x_lin, one, s));

    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, out_glu, lin_plus1, s));
    return out;
}

/// Whether decode should try the in-place gather-qmv kernels at all.
///
/// The gate is "would the FUSED gate+up+SwiGLU kernel engage", because that is
/// what changed the answer. SPLIT gatherQmv loses to stock `gather_qmm` (Laguna
/// XS, GPU eval 12.73 vs 12.47 ms) and lost on gemma4-26B-A4B too, which is why
/// no arch opted in. Fusing the two projections and the activation into one
/// simdgroup flips it: 13.007/13.568 -> 12.768/13.114 ms/forward, both pairs.
///
/// So the eligibility predicate is deliberately NARROW — silu activation
/// (baked into the kernel), matching quant geometry on gate and up — rather
/// than a model_type list. An arch that does not meet it, gemma4's gelu MoE
/// included, keeps stock gather and cannot regress.
/// The `swiglu_limit` the fused SwiGLU epilogues clamp at: 0 for none, the limit when it is a
/// whole-number clamp (GLM-5-Next's 10), null for any other limited activation (gpt_oss).
pub fn swigluClampLimit(swiglu_limit: f32, swiglu_clamp: bool) ?u32 {
    if (swiglu_limit <= 0.0) return 0;
    if (!swiglu_clamp or @round(swiglu_limit) != swiglu_limit or swiglu_limit > 1 << 16) return null;
    return @intFromFloat(swiglu_limit);
}

pub const HiddenAct = enum { gelu_approx, gelu, silu, relu_sq };
