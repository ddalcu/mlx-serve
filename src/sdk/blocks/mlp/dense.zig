//! sdk/blocks/mlp/dense.
//! Owns DenseMlpWeights, and 1 more.

const mlx = @import("../../../mlx/ffi.zig");
const lane_qmm = @import("../qmm/lane_qmm.zig");
const simd_qmm = @import("../qmm/simd_qmm.zig");
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;
const FUSED_ROWS_MAX_M = @import("../linear/fused_rows.zig").FUSED_ROWS_MAX_M;
const FusedRows = @import("../linear/fused_rows.zig").FusedRows;
const fusedResidualNorm = @import("../norms/residual_norm.zig").fusedResidualNorm;
const swigluJoined = @import("fused_activations.zig").swigluJoined;

pub const DenseMlpWeights = struct {
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
    /// Row-joined [gate | up] for decode width; gate/up are views of it once
    /// built (fuseRowGroup).
    gu: FusedRows = .{},
};

// ── Dense MLP (SwiGLU: SiLU(gate(x)) * up(x) -> down) ──

