//! mlx-stream's test root: every file's tests, built and run by `zig build mlx-stream-test`. A comptime block, not a
//! test block: the root adds no test of its own.

comptime {
    _ = @import("sdk/weights/streamed/mimo/bank.zig");
    _ = @import("sdk/weights/streamed/mimo/stream.zig");
    _ = @import("sdk/weights/streamed/mimo/stream_source.zig");
    _ = @import("sdk/weights/streamed/mimo/prefill_combine.zig");
    _ = @import("sdk/weights/streamed/mimo/gpu_warm.zig");
    _ = @import("sdk/weights/streamed/glm5n/experts.zig");
    _ = @import("sdk/weights/streamed/glm5n/exl3_bank.zig");
    _ = @import("sdk/weights/streamed/glm5n/exl3_math.zig");
    _ = @import("sdk/weights/streamed/glm5n/exl3_kernels.zig");
    _ = @import("sdk/weights/streamed/glm5n/exl3_forms.zig");
    _ = @import("sdk/weights/streamed/glm5n/settings.zig");
    _ = @import("sdk/weights/streamed/ops.zig");
    _ = @import("sdk/weights/streamed/exl3_source.zig");
    _ = @import("sdk/weights/streamed/experts.zig");
    _ = @import("sdk/weights/streamed/expert/bank.zig");
    _ = @import("sdk/weights/streamed/expert/io_test.zig");
    _ = @import("sdk/weights/streamed/mlx_pack.zig");
    _ = @import("sdk/weights/streamed/expert/lookahead.zig");
    _ = @import("sdk/weights/streamed/expert/event_test.zig");
    _ = @import("sdk/weights/streamed/expert/admission.zig");
    _ = @import("mlx_stream.zig");
    _ = @import("sdk/quant/exl3/stream/source.zig");
    _ = @import("sdk/weights/streamed/phase_memory.zig");
    _ = @import("sdk/weights/streamed/profile/decode_timers.zig");
    _ = @import("sdk/weights/streamed/profile/decode_first.zig");
    _ = @import("sdk/weights/streamed/profile/verify_timeline.zig");
    _ = @import("sdk/weights/streamed/expert/policy_test.zig");
    _ = @import("sdk/weights/streamed/expert/stream.zig");
    _ = @import("sdk/weights/streamed/expert/stream_of_test.zig");
    _ = @import("sdk/weights/streamed/expert/routes_test.zig");
    _ = @import("sdk/quant/exl3/stream/kernels.zig");
    _ = @import("sdk/quant/exl3/stream/selfcheck.zig");
    _ = @import("sdk/quant/exl3/stream/kernel_ops_gate.zig");
    _ = @import("sdk/quant/exl3/stream/quant.zig");
    _ = @import("sdk/quant/exl3/stream/kernel_routes.zig");
    _ = @import("sdk/weights/streamed/sdk_ext.zig");
    _ = @import("sdk/weights/streamed/ngram_table.zig");
    _ = @import("sdk/quant/exl3/stream/sushi_parity.zig");
    _ = @import("plugins/glm_moe_dsa/glm_moe_dsa.zig");
    _ = @import("plugins/glm_moe_dsa/bank.zig");
    _ = @import("plugins/glm_moe_dsa/exl3_bank.zig");
    _ = @import("plugins/glm_moe_dsa/exl3_quant.zig");
    _ = @import("plugins/glm_moe_dsa/settings.zig");
    _ = @import("plugins/glm_moe_dsa/bill.zig");
    _ = @import("plugins/glm_moe_dsa/cache.zig");
    _ = @import("plugins/glm_moe_dsa/graph.zig");
    _ = @import("plugins/glm_moe_dsa/experts.zig");
    _ = @import("plugins/glm_moe_dsa/topk.zig");
    _ = @import("plugins/glm_moe_dsa/trunk_kernels.zig");
    _ = @import("plugins/glm_moe_dsa/trunk_selfcheck.zig");
    _ = @import("plugins/glm_moe_dsa/split.zig");
    _ = @import("plugins/glm_moe_dsa/module.zig");
    _ = @import("plugins/glm_moe_dsa/arch.zig");
    _ = @import("plugins/glm_moe_dsa/mtp.zig");
    _ = @import("plugins/glm_moe_dsa/glm53_exl3/kernels.zig");
    _ = @import("plugins/glm_moe_dsa/glm53_exl3/selfcheck.zig");
    _ = @import("plugins/glm_moe_dsa/glm53_exl3/routes.zig");
    _ = @import("plugins/glm_moe_dsa/glm53_exl3/quant.zig");
    // GLM-5.3-Flash (glm5_next) needs the host's DSA kernels: its files build only on a host that exports them.
    // GLM-5.3's refusal on a host without its surface: its own file, compiled only on such a host (the GLM tests above
    // skip there).
}
