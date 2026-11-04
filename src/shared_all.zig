//! Every `Shared` the process holds, referenced so the unit that owns their storage under `-Dunits` (the host,
//! `src/host_unit.zig`) exports each one even when its own code never reads it; the engine and plugin units link to
//! those exports (`sdk/util/shared.zig`). A new `Shared` goes on this list.

comptime {
    _ = @import("sdk/util/err_name.zig").last_tls;
    if (@import("build_options").mlx_stream) _ = @import("mlx_stream").shared_state;
    _ = @import("host/plugins/loader.zig").process;
    _ = @import("host/plugin_host_impl.zig").process_shared;
    _ = @import("host/host_table.zig").published;
    _ = @import("host/http/server.zig").serverState_shared;
    _ = @import("host/http/server.zig").launch_shared;
    _ = @import("host/registry/memory_ceiling.zig").overrides_shared;
    _ = @import("host/registry/registry.zig").generations_shared;
    _ = @import("host/engines/refusal.zig").refused_tls;
    _ = @import("plugins/minimax_h3/cluster.zig").hooks_shared;
    _ = @import("plugins/minimax_h3/h3.zig").residents_shared;
    _ = @import("plugins/deepseek_v41/deepseek_v41.zig").clusterHooks_shared;
    _ = @import("plugins/qwen4_exp/ple_gpu.zig").pleGpu_shared;
    _ = @import("plugins/qwen4_exp/plugin.zig").EngineCtx;
    _ = @import("plugins/qwen4_exp/qwen4_exp.zig").warm_shared;
    _ = @import("sdk/blocks/attention/nax_attention.zig").naxCli_shared;
    _ = @import("sdk/blocks/attention/decode_attn_quant.zig").cli_shared;
    _ = @import("sdk/blocks/dit/w8a8.zig").w8a8State_shared;
    _ = @import("sdk/blocks/moe/gather_qmv_rows.zig").moeStats_shared;
    _ = @import("sdk/quant/gguf/claim.zig").claimState_shared;
    _ = @import("sdk/util/log.zig").sinks_shared;
    _ = @import("sdk/config/model.zig").modelCli_shared;
    _ = @import("sdk/ane/offload.zig").aneState_shared;
    _ = @import("sdk/media/progress.zig").GenYield;
    _ = @import("sdk/media/progress.zig").StreamKeepalive;
    _ = @import("engines/mlx/transformer.zig").cli_shared;
    _ = @import("engines/mlx/generate.zig").prefillCli_shared;
    _ = @import("engines/mlx/cluster/distributed.zig").cluster_shared;
    _ = @import("engines/mlx/diag/profile.zig").profileState_shared;
    _ = @import("engines/mlx/ssm/group.zig").ssmGroup_shared;
    _ = @import("engines/mlx/modules/stream_glue.zig").glueState_shared;
    _ = @import("mlx/ffi.zig").latch_shared;
    _ = @import("mlx/ffi.zig").fault_chunk;
    _ = @import("mlx/ffi.zig").fault_step;
}
