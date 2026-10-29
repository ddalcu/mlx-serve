//! sdk/blocks/attention/batched.
//! Owns BATCHED_PER_SLOT_ATTN_MIN_KV, batchedAttnPerSlot.

const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const MAX_BATCH_ROWS = @import("../../bases/run.zig").MAX_BATCH_ROWS;
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const axisView = @import("../../../mlx/array.zig").axisView;
const fusedSdpa256Prefill = @import("attn256.zig").fusedSdpa256Prefill;
const kvAttnFusedEligible = @import("kv_quant_attn.zig").kvAttnFusedEligible;
const kvAttnVerifyEligible = @import("kv_quant_attn.zig").kvAttnVerifyEligible;
const logKvAttnFusedEngaged = @import("kv_quant_attn.zig").logKvAttnFusedEngaged;
const qkvAttnDecodeKernel = @import("kv_quant_attn.zig").qkvAttnDecodeKernel;
const qkvAttnVerifyKernel = @import("kv_quant_attn.zig").qkvAttnVerifyKernel;
const DenseKVView = @import("kv_view.zig").DenseKVView;
const sdpaForceFused = @import("sdpa.zig").sdpaForceFused;
const splitCausalSdpa = @import("sdpa.zig").splitCausalSdpa;

/// Longest slot view at or past which a batched group attends per slot: below it one
/// stacked dispatch is cheaper than N, above it the pad + concat copy dominates.
pub const BATCHED_PER_SLOT_ATTN_MIN_KV: c_int = 1024;

pub fn batchedAttnPerSlot(views: []const DenseKVView) bool {
    for (views) |dv| if (mlx.getShape(dv.k)[2] >= BATCHED_PER_SLOT_ATTN_MIN_KV) return true;
    return false;
}

// Pads each slot's dense KV view (shape [1, kv_h, kv_len_i, head_dim]) to a
// common kv_max along axis 2 with bf16 zeros and concatenates axis 0 →
// [N, kv_h, kv_max, head_dim]. Views come from KVCache.denseView so quant
// schemes are already dequantized; `key_not_value` selects k or v.

// Builds the additive per-slot decode mask [N,1,1,kv_max] in bf16 where
// valid columns are 0 and out-of-range columns are -inf. Computed via
// broadcasting: positions[1,1,1,kv_max] < kv_lens[N,1,1,1].
