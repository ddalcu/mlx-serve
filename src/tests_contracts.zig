//! Split unit-test root: kernel contracts, sdk/blocks, sdk/quant and sdk/weights.

// pub: Sushi's EXL3 module (src/sdk/quant/exl3/sushi) reaches these through its host root.
pub const mlx = @import("mlx/ffi.zig");
pub const io_util = @import("sdk/util/io_util.zig");
pub const log = @import("sdk/util/log.zig");
pub const env = @import("env.zig");
comptime {
    _ = @import("plugins/registry_test_stubs.zig");
    @import("plugin_host_check.zig").reexports(@This());
}

/// The test process caps MLX's buffer pool as the server's `main` does; `build/test_runner.zig` calls it first.
export fn msv_test_process_init() void {
    @import("mlx/state.zig").applyMlxCacheLimit();
}

export fn mlx_serve_mlx_api() callconv(.c) *const anyopaque {
    @panic("mlx_serve_mlx_api is not linked in test-contracts");
}

export fn mlx_llama_log(_: c_int, _: [*:0]const u8) callconv(.c) void {}

test {
    _ = @import("sdk/blocks/hc/hc_cuda_test.zig");
    _ = @import("plugins/gemma4/gelu_ple_test.zig");
    _ = @import("sdk/quant/gguf/claim_test.zig");
    _ = @import("sdk/blocks/attention/batched_test.zig");
    _ = @import("sdk/blocks/attention/kv_quant_attn_test.zig");
    _ = @import("sdk/blocks/attention/sdpa_test.zig");
    _ = @import("sdk/blocks/gdn/decode_fused_test.zig");
    _ = @import("sdk/blocks/hc/hc_decode2_test.zig");
    _ = @import("sdk/blocks/linear/qmatmul_test.zig");
    _ = @import("sdk/blocks/linear/qmv_rows_test.zig");
    _ = @import("sdk/blocks/linear/verify_projection_test.zig");
    _ = @import("sdk/blocks/linear/verify_qmm_test.zig");
    _ = @import("sdk/blocks/mlp/fused_activations_test.zig");
    _ = @import("sdk/blocks/moe/decode_test.zig");
    _ = @import("sdk/blocks/moe/verify_kernels_test.zig");
    _ = @import("sdk/quant/block_test.zig");
    _ = @import("sdk/quant/hadamard/format_test.zig");
    _ = @import("sdk/blocks/kda/gate_test.zig");
    _ = @import("sdk/blocks/attention/attn256_test.zig");
    _ = @import("sdk/quant/gguf/claim.zig");
    _ = @import("sdk/weights/streamed/expert.zig");
    _ = @import("sdk/weights/streamed/expert/io.zig");
    _ = @import("sdk/weights/streamed/expert/policy.zig");
    _ = @import("sdk/weights/streamed/expert/routes.zig");
    _ = @import("sdk/weights/streamed/expert/usage.zig");
    _ = @import("sdk/weights/streamed/expert/tally.zig");
    _ = @import("sdk/weights/streamed/expert/stream.zig");
    _ = @import("sdk/weights/streamed/expert/bank_pool.zig");
    _ = @import("sdk/weights/streamed/expert/stage.zig");
    _ = @import("sdk/weights/streamed/expert/stage_votes.zig");
    _ = @import("sdk/weights/streamed/expert/stage_trim.zig");
    _ = @import("sdk/weights/streamed/expert/la_wide.zig");
    _ = @import("sdk/weights/streamed/expert/io_test.zig");
    _ = @import("sdk/weights/streamed/expert/event_test.zig");
    _ = @import("sdk/weights/streamed/expert/policy_test.zig");
    _ = @import("sdk/weights/streamed/expert/stream_of_test.zig");
    _ = @import("sdk/weights/streamed/expert/seed_free_test.zig");
    _ = @import("sdk/weights/streamed/expert/stage_test.zig");
    _ = @import("sdk/weights/streamed/expert/direct_parts_test.zig");
    _ = @import("sdk/weights/streamed/expert/demand_first_test.zig");
    _ = @import("sdk/weights/streamed/mlx_pack_test.zig");
    _ = @import("plugins/deepseek_v41/streamed_test.zig");
    _ = @import("sdk/testing/kernel_contract/registry.zig");
    _ = @import("plugins/qwen4_exp/trunk/warmup.zig");
    _ = @import("sdk/quant/hadamard/rht.zig");
    _ = @import("sdk/blocks/qmm/qmv2.zig");
    _ = @import("sdk/blocks/qmm/qmv_cuda.zig");
    _ = @import("sdk/blocks/moe/moe_gather_ubench.zig");
    _ = @import("sdk/blocks/moe/moe_affine4.zig");
    _ = @import("sdk/blocks/hc/hc_decode2.zig");
    _ = @import("sdk/blocks/hc/hc_cuda.zig");
    _ = @import("sdk/blocks/moe/moe_cuda.zig");
    _ = @import("sdk/blocks/moe/moe_prefill_cuda.zig");
    _ = @import("sdk/blocks/attention/sparse_block/block_test.zig");
    _ = @import("sdk/blocks/attention/sparse_block/contract.zig");
    _ = @import("sdk/blocks/attention/sparse_block/decode.zig");
    _ = @import("sdk/blocks/attention/sparse_block/decode_cuda.zig");
    _ = @import("sdk/blocks/attention/sparse_block/mask.zig");
    _ = @import("sdk/blocks/attention/sparse_block/params.zig");
    _ = @import("sdk/blocks/attention/sparse_block/reference.zig");
    _ = @import("sdk/blocks/attention/sparse_block/row_mask.zig");
    _ = @import("sdk/blocks/attention/sparse_block/score.zig");
    _ = @import("sdk/blocks/attention/sparse_block/select.zig");
    _ = @import("sdk/blocks/qmm/rowqmv.zig");
    _ = @import("sdk/blocks/qmm/simd_qmm.zig");
    _ = @import("sdk/blocks/qmm/lane_qmm.zig");
    _ = @import("sdk/blocks/attention/lane_attn.zig");
    _ = @import("sdk/blocks/norms/add_norm.zig");
    _ = @import("sdk/blocks/moe/moe_fp4.zig");
    _ = @import("sdk/quant/exl3/sushi_pack.zig");
    _ = @import("sdk/quant/outside_model.zig");
    _ = @import("sdk/quant/bank_fit_key.zig");
    _ = @import("sdk/quant/nvfp4/modelopt_pack.zig");
    _ = @import("sdk/quant/formats.zig");
    _ = @import("sdk/blocks/qmm/mxfp8_qmv.zig");
    _ = @import("sdk/blocks/attention/dec_attn.zig");
    _ = @import("sdk/blocks/conv/gated_conv_step.zig");
    _ = @import("sdk/blocks/attention/nax_attention.zig");
    _ = @import("sdk/blocks/kda/recurrence.zig");
    _ = @import("sdk/blocks/gdn/decode.zig");
    _ = @import("sdk/blocks/attention/row_attn.zig");
    _ = @import("sdk/blocks/qmm/qmv_nax2.zig");
    _ = @import("sdk/blocks/moe/gather_qmm_nax.zig");
    _ = @import("sdk/blocks/qmm/qmm_int8.zig");
    _ = @import("sdk/blocks/dit/w8a8.zig");
    _ = @import("sdk/blocks/dit/int8_attn.zig");
    _ = @import("sdk/blocks/attention/kv_quant.zig");
    _ = @import("sdk/weights/nocache_reader.zig");
    _ = @import("sdk/quant/gguf/engine_pick.zig");
    _ = @import("sdk/blocks/dit/w8a8_bill.zig");
    _ = @import("sdk/weights/lora.zig");
}
