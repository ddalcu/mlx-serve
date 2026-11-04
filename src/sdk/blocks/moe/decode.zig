//! sdk/blocks/moe/decode.
//! Owns batchedExpertMm, and 9 more.

const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const moe_fp4 = @import("moe_fp4.zig");
const moe_affine4 = @import("moe_affine4.zig");
const down_reduce = @import("down_reduce.zig");
const gather_qmv = @import("gather_qmv.zig");
const gather_qmv_rows = @import("gather_qmv_rows.zig");
const lastDim = @import("../../../mlx/array.zig").lastDim;
const qwen4Standin = @import("../../diag/standin.zig").qwen4Standin;
const standinOnes = @import("../../diag/standin.zig").standinOnes;
const QuantParams = @import("../../quant/params.zig").QuantParams;
const swigluSigTable = @import("../mlp/fused_activations.zig").swigluSigTable;
const gatherQmvDownReduce = @import("down_reduce.zig").gatherQmvDownReduce;
const gatherQmvDownReduceRows = @import("down_reduce.zig").gatherQmvDownReduceRows;
const gatherQmvDownReduceRowsEligible = @import("down_reduce.zig").gatherQmvDownReduceRowsEligible;
const gatherQmv = @import("gather_qmv.zig").gatherQmv;
const gatherQmvGateUp = @import("gather_qmv.zig").gatherQmvGateUp;
const gatherQmvGateUpEnabled = @import("gather_qmv.zig").gatherQmvGateUpEnabled;
const gatherQmvGateUpRows = @import("gather_qmv_rows.zig").gatherQmvGateUpRows;
const MoeMlpWeights = @import("weights.zig").MoeMlpWeights;

/// The fp4 MoE decode kernels read no expert biases, and apply a SwiGLU clamp only as a whole-number
/// limit: a pack with a bias, or `clamp` null (`swigluClampLimit`: a clamp no kernel serves), keeps the
/// composed chain, as does an activation that is not 16-bit (the SwiGLU table covers 16-bit patterns).
pub fn moeFp4DecodeEligible(rows: c_int, has_expert_bias: bool, clamp: ?u32, act: mlx.mlx_dtype, gate: QuantMode, up: QuantMode, down: QuantMode) bool {
    return rows <= moe_fp4.MAX_ROWS and !has_expert_bias and clamp != null and (act == .bfloat16 or act == .float16) and (gate == .mxfp4 or gate == .nvfp4) and up == gate and down == gate;
}
