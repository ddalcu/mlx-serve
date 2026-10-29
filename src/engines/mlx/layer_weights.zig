//! engines/mlx/layer_weights.
//! Owns FullAttnWeights, QW, HybridMlpWeights, MoeLayerWeights.

const mlx = @import("../../mlx/ffi.zig");
const FusedRows = @import("../../sdk/blocks/linear/fused_rows.zig").FusedRows;
const HcWeights = @import("../../plugins/qwen4_exp/weights.zig").HcWeights;
const PleWeights = @import("../../plugins/qwen4_exp/weights.zig").PleWeights;
const LinearAttnWeights = @import("../../sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const DenseMlpWeights = @import("../../sdk/blocks/mlp/dense.zig").DenseMlpWeights;
const MoeMlpWeights = @import("../../sdk/blocks/moe/weights.zig").MoeMlpWeights;

// ── MoE model per-layer weights ──

pub const FullAttnWeights = struct {
    q_w: mlx.mlx_array,
    q_s: mlx.mlx_array,
    q_b: mlx.mlx_array,
    k_w: mlx.mlx_array,
    k_s: mlx.mlx_array,
    k_b: mlx.mlx_array,
    v_w: mlx.mlx_array,
    v_s: mlx.mlx_array,
    v_b: mlx.mlx_array,
    o_w: mlx.mlx_array,
    o_s: mlx.mlx_array,
    o_b: mlx.mlx_array,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    /// Row-joined [q | k | v] for decode width; q/k/v are views of it once
    /// built (fuseRowGroup). Only the hybrid loader builds it so far.
    qkv: FusedRows = .{},
    // Per-head attention OUTPUT gate (self_attn.g_proj / attention.g_proj →
    // [B,S,n_heads]). The ACTIVATION is the arch's, not the weight's: Laguna
    // applies softplus. Null-ctx on every
    // other arch → skipped by appendFullAttnWeights and by the gate branches.
    g_w: mlx.mlx_array = .{ .ctx = null },
    g_s: mlx.mlx_array = .{ .ctx = null },
    g_b: mlx.mlx_array = .{ .ctx = null },
    // qwen4_exp QSA indexer: one projection for the index queries + the
    // single shared key head, and their per-head RMS norms. Null-ctx on
    // every other arch.
    idx_qk_w: mlx.mlx_array = .{ .ctx = null },
    idx_qk_s: mlx.mlx_array = .{ .ctx = null },
    idx_qk_b: mlx.mlx_array = .{ .ctx = null },
    idx_q_norm: mlx.mlx_array = .{ .ctx = null },
    idx_k_norm: mlx.mlx_array = .{ .ctx = null },
    // gpt_oss: ADDITIVE projection biases (`self_attn.{q,k,v,o}_proj.bias`).
    // These are NOT the affine quantizer's `*_b` zero-points above — a gpt_oss
    // checkpoint carries BOTH spellings on the same projection
    // (`q_proj.bias` next to `q_proj.biases`), with different shapes and
    // different meanings. Null-ctx on every other arch.
    q_bias: mlx.mlx_array = .{ .ctx = null },
    k_bias: mlx.mlx_array = .{ .ctx = null },
    v_bias: mlx.mlx_array = .{ .ctx = null },
    o_bias: mlx.mlx_array = .{ .ctx = null },
    // gpt_oss: learned per-head attention sinks (`self_attn.sinks`, [n_heads]).
    // One extra column in the softmax DENOMINATOR, letting a head attend to
    // "nothing". Passed straight to mlx's fused SDPA, which takes them as a
    // function constant (`_has_sinks_`) — no unfused fallback.
    sinks: mlx.mlx_array = .{ .ctx = null },
};

pub const QW = @import("../../sdk/blocks/linear/fused_rows.zig").QW;

/// A layer's MLP: dense, or a routed MoE.
pub const HybridMlpWeights = union(enum) {
    dense: DenseMlpWeights,
    moe: MoeMlpWeights,
};

pub const MoeLayerWeights = struct {
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    is_linear: bool,
    attn: union(enum) { full: FullAttnWeights, linear: LinearAttnWeights },
    mlp: HybridMlpWeights,
    // Gemma 4 MoE: separate shared expert MLP (null for Qwen3.5)
    shared_mlp: ?DenseMlpWeights = null,
    // Gemma 4 feedforward norms (null for Qwen3.5)
    pre_ff_norm: ?mlx.mlx_array = null,
    post_ff_norm: ?mlx.mlx_array = null,
    pre_ff_norm_2: ?mlx.mlx_array = null,
    post_ff_norm_1: ?mlx.mlx_array = null,
    post_ff_norm_2: ?mlx.mlx_array = null,
    layer_scalar: ?mlx.mlx_array = null,
    // qwen4_exp: the two gated residual reads/writes around the block and
    // the n-gram PLE on its one layer. Null for every other arch.
    hc_attn: ?HcWeights = null,
    hc_mlp: ?HcWeights = null,
    ple: ?PleWeights = null,
};
