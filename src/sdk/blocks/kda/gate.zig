//! sdk/blocks/kda/gate.
//! Owns kdaGateChain.

const mlx = @import("../../../mlx/ffi.zig");

/// KDA forget gate (fla/ops/kda/fused_recurrent.py, the USE_LOWER_BOUND arm):
///
///   g = exp(bound * sigmoid(exp(A_log) * (a + dt_bias)))
///
/// The bound REPLACES the softplus form rather than clamping it — the decay
/// lands in (exp(bound), 1) by construction, where `-exp(A_log)·softplus(·)`
/// is unbounded below. Reading `kda_lower_bound` as a clamp on the softplus
/// gate would be a plausible-looking, silently wrong recurrence.
///
/// Everything here is elementwise over the FLAT [B,S,Hv*Dk] gate: `A_log`
/// arrives already expanded per key channel at load (see initMoeLayers), so
/// the hot path needs no reshape and no per-head broadcast.
pub fn kdaGateChain(A_log: mlx.mlx_array, a: mlx.mlx_array, dt_bias: mlx.mlx_array, bound: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var A_log_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(A_log_f32);
    try mlx.check(mlx.mlx_astype(&A_log_f32, A_log, .float32, s));
    var exp_A = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(exp_A);
    try mlx.check(mlx.mlx_exp(&exp_A, A_log_f32, s));

    var a_plus_dt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_plus_dt);
    try mlx.check(mlx.mlx_add(&a_plus_dt, a, dt_bias, s));
    var a_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_f32);
    try mlx.check(mlx.mlx_astype(&a_f32, a_plus_dt, .float32, s));

    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, exp_A, a_f32, s));
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, scaled, s));

    const bound_arr = mlx.mlx_array_new_float(bound);
    defer _ = mlx.mlx_array_free(bound_arr);
    var g_log = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_log);
    try mlx.check(mlx.mlx_multiply(&g_log, sig, bound_arr, s));

    // f32 like the reference's compute_g_safe: near one, bf16 spacing (2^-8) is a
    // decay error the recurrence compounds every token.
    var g = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_exp(&g, g_log, s));
    return g;
}
