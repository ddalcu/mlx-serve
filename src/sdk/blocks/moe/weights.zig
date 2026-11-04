//! sdk/blocks/moe/weights.
//! Owns MoeMlpWeights.

const mlx = @import("../../../mlx/ffi.zig");
const moe_affine4 = @import("moe_affine4.zig");

pub const MoeMlpWeights = struct {
    router_w: mlx.mlx_array,
    router_s: mlx.mlx_array,
    router_b: mlx.mlx_array,
    switch_gate_w: mlx.mlx_array,
    switch_gate_s: mlx.mlx_array,
    switch_gate_b: mlx.mlx_array,
    switch_up_w: mlx.mlx_array,
    switch_up_s: mlx.mlx_array,
    switch_up_b: mlx.mlx_array,
    switch_down_w: mlx.mlx_array,
    switch_down_s: mlx.mlx_array,
    switch_down_b: mlx.mlx_array,
    shared_gate_w: mlx.mlx_array,
    shared_gate_s: mlx.mlx_array,
    shared_gate_b: mlx.mlx_array,
    shared_up_w: mlx.mlx_array,
    shared_up_s: mlx.mlx_array,
    shared_up_b: mlx.mlx_array,
    shared_down_w: mlx.mlx_array,
    shared_down_s: mlx.mlx_array,
    shared_down_b: mlx.mlx_array,
    // The dense shared-expert gate as one contiguous [hidden] vector (null-ctx otherwise): the softmax
    // router kernel dots it with x.
    shared_gate_vec: mlx.mlx_array = .{ .ctx = null },
    // Shared expert gating (Qwen3.5; null for Gemma 4)
    shared_expert_gate_w: ?mlx.mlx_array = null,
    shared_expert_gate_s: ?mlx.mlx_array = null,
    shared_expert_gate_b: ?mlx.mlx_array = null,
    // Sigma-MoE routing (Gemma 4; null for Qwen3.5)
    router_scale: ?mlx.mlx_array = null,
    per_expert_scale: ?mlx.mlx_array = null,
    /// ModelOpt NVFP4 banks: each expert's gate global scale [E], applied before the activation.
    gate_expert_scale: mlx.mlx_array = .{ .ctx = null },
    /// The same for `up` when the activation clamps it (GLM): the scale must land before the clamp.
    up_expert_scale: mlx.mlx_array = .{ .ctx = null },
    // Hy3 (hy_v3) sigmoid routing: f32 [num_experts] bias added to the sigmoid
    // scores for top-k SELECTION only (weights come from the unbiased scores).
    // Non-null expert_bias routes moeMLP2 through hy3RoutingChain.
    expert_bias: ?mlx.mlx_array = null,
    route_norm: bool = true,
    route_scale: f32 = 1.0,
    // Hy3 shared expert is ALWAYS added, with no shared_expert_gate.
    shared_ungated: bool = false,
    // One-expert views of the affine-4 banks the decode kernels bind (`moe_affine4.bindViews`).
    bank_views: ?moe_affine4.Views = null,
};
