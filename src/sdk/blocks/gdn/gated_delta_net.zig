//! sdk/blocks/gdn/gated_delta_net.
//! Owns LinearAttnWeights, GdnProj, and 1 more.

const mlx = @import("../../../mlx/ffi.zig");
const gdn_decode = @import("decode.zig");
const log = @import("../../util/log.zig");
const decode_fused = @import("decode_fused.zig");
const env = @import("../../../env.zig");
const QW = @import("../linear/fused_rows.zig").QW;
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const qwen4Standin = @import("../../diag/standin.zig").qwen4Standin;
const standinOnes = @import("../../diag/standin.zig").standinOnes;
const standinRef = @import("../../diag/standin.zig").standinRef;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;
const kdaGateChain = @import("../kda/gate.zig").kdaGateChain;
const FusedRows = @import("../linear/fused_rows.zig").FusedRows;
const GDN_FUSED_MAX_ROWS = @import("decode_fused.zig").GDN_FUSED_MAX_ROWS;
const GdnPrework = @import("decode_fused.zig").GdnPrework;
const gdnDecodeFusedEnabled = @import("decode_fused.zig").gdnDecodeFusedEnabled;
const gdnDecodeRecurEnabled = @import("decode_fused.zig").gdnDecodeRecurEnabled;
const gdnNormGateFused = @import("decode_fused.zig").gdnNormGateFused;
const gdnPrefillFusedFor = @import("decode_fused.zig").gdnPrefillFusedFor;
const gdnPreworkFused = @import("decode_fused.zig").gdnPreworkFused;
const gdnVerifyFoldEnabled = @import("decode_fused.zig").gdnVerifyFoldEnabled;
const GdnRoute = @import("kernels.zig").GdnRoute;
const GDN_BLOCK_T = @import("kernels.zig").GDN_BLOCK_T;
const gdnBlockTFor = @import("kernels.zig").gdnBlockTFor;
const gdnBlockedEligible = @import("kernels.zig").gdnBlockedEligible;
const gdnBlockedEnabled = @import("kernels.zig").gdnBlockedEnabled;
const gdnPipelinedEnabled = @import("kernels.zig").gdnPipelinedEnabled;
const gdnRecurCuda = @import("kernels.zig").gdnRecurCuda;
const gdnRecurOps = @import("kernels.zig").gdnRecurOps;
const gdnRouteKernel = @import("kernels.zig").gdnRouteKernel;
const gdnSetGrid = @import("kernels.zig").gdnSetGrid;
const getGdnKernelSeq = @import("kernels.zig").getGdnKernelSeq;

pub const LinearAttnWeights = struct {
    // For separate projections (qwen3_5_moe): qkv=QKV, z=Z, a=A, b=B
    // For combined projections (qwen3_next): qkv=QKVZ, b=BA, z/a unused
    combined_proj: bool = false,
    qkv_w: mlx.mlx_array,
    qkv_s: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    z_w: mlx.mlx_array,
    z_s: mlx.mlx_array,
    z_b: mlx.mlx_array,
    a_w: mlx.mlx_array,
    a_s: mlx.mlx_array,
    a_b: mlx.mlx_array,
    b_w: mlx.mlx_array,
    b_s: mlx.mlx_array,
    b_b: mlx.mlx_array,
    conv1d_w: mlx.mlx_array,
    A_log: mlx.mlx_array,
    dt_bias: mlx.mlx_array,
    norm_w: mlx.mlx_array,
    out_w: mlx.mlx_array,
    out_s: mlx.mlx_array,
    out_b: mlx.mlx_array,
    /// Row-joined [qkv | z | a | b] (separate projections only); the fields
    /// above are views of it once built (fuseRowGroup).
    in: FusedRows = .{},
    /// q/k/v quantized at different widths cannot join: `qkv_w` is null and each projects alone.
    qkv_parts: [3]QW = .{ .{}, .{}, .{} },

    /// The ANE GDN program projects one joined qkv and a full-rank z.
    pub fn aneGdnServes(la: *const LinearAttnWeights) bool {
        return !la.combined_proj and la.qkv_w.ctx != null;
    }
};

// ── GatedDeltaNet (linear attention layers) ──

pub const GdnProj = struct { qkv: mlx.mlx_array, z: mlx.mlx_array };

