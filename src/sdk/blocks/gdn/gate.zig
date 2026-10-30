//! sdk/blocks/gdn/gate.
//! Owns gdnChainGateDtype, gdnGateChain.

const mlx = @import("../../../mlx/ffi.zig");
const gdn_decode = @import("decode.zig");
const log = @import("../../util/log.zig");

/// MoE routing chain (negate→argpartition→slice→softmax→take→sum→expand→divide).
/// Free-function variant of `Transformer.moeRoutingUncompiled` so unit tests can
/// exercise the pure subgraph without constructing a full Transformer. Returns
/// owned `inds` (int32, [..., k]) and `norm_scores` (bf16, [..., k]) — caller
/// must free both.
/// GatedDeltaNet gating chain: g = exp(-exp(A_log) * softplus(a + dt_bias)),
/// computed in float32 and returned at `g_dtype`, the recurrent state's. Mirrors
/// mlx-lm's `compute_g` (which is `@mx.compile`d). Pure — serves as both the
/// compiled-closure body and the uncompiled fallback. Returns owned array.
/// The composed chain's gate dtype: an f32 state under bf16 activations takes an f32 gate;
/// every other pack keeps the bf16 gate this chain has always returned.
pub fn gdnChainGateDtype(act: mlx.mlx_dtype, state: mlx.mlx_dtype) mlx.mlx_dtype {
    return if (act == .bfloat16) state else .bfloat16;
}

pub fn gdnGateChain(A_log: mlx.mlx_array, a: mlx.mlx_array, dt_bias: mlx.mlx_array, g_dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var A_log_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(A_log_f32);
    try mlx.check(mlx.mlx_astype(&A_log_f32, A_log, .float32, s));
    var exp_A = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(exp_A);
    try mlx.check(mlx.mlx_exp(&exp_A, A_log_f32, s));

    // softplus(a + dt_bias) = log1p(exp(a + dt_bias)); an f32 gate adds in f32 too,
    // as the reference's `a.float() + dt_bias` does.
    var a_plus_dt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_plus_dt);
    if (g_dtype == .float32) {
        var a32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a32);
        var dt32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt32);
        try mlx.check(mlx.mlx_astype(&a32, a, .float32, s));
        try mlx.check(mlx.mlx_astype(&dt32, dt_bias, .float32, s));
        try mlx.check(mlx.mlx_add(&a_plus_dt, a32, dt32, s));
    } else try mlx.check(mlx.mlx_add(&a_plus_dt, a, dt_bias, s));
    var a_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_f32);
    try mlx.check(mlx.mlx_astype(&a_f32, a_plus_dt, .float32, s));
    var exp_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(exp_a);
    try mlx.check(mlx.mlx_exp(&exp_a, a_f32, s));
    var sp_inner = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sp_inner);
    try mlx.check(mlx.mlx_log1p(&sp_inner, exp_a, s));

    var neg_decay = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_decay);
    try mlx.check(mlx.mlx_multiply(&neg_decay, exp_A, sp_inner, s));
    var neg_neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_neg);
    try mlx.check(mlx.mlx_negative(&neg_neg, neg_decay, s));
    var g_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_f32);
    try mlx.check(mlx.mlx_exp(&g_f32, neg_neg, s));
    var g = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_astype(&g, g_f32, g_dtype, s));
    return g;
}
