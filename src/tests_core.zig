//! Split unit-test root: host, engines, non-contract SDK and ABI.

// pub: Sushi's EXL3 module (src/sdk/quant/exl3/sushi) reaches these through its host root.
pub const mlx = @import("mlx/ffi.zig");
pub const io_util = @import("sdk/util/io_util.zig");
pub const log = @import("sdk/util/log.zig");
pub const env = @import("env.zig");
comptime {
    _ = @import("plugins/registry_test_core.zig");
    @import("plugin_host_check.zig").reexports(@This());
}

/// The test process caps MLX's buffer pool as the server's `main` does; `build/test_runner.zig` calls it first.
export fn msv_test_process_init() void {
    @import("mlx/state.zig").applyMlxCacheLimit();
}

test {
    _ = @import("host/media_jobs/plugin/served.zig");
    _ = @import("sdk/vision/qwen_vision_test.zig");
    _ = @import("sdk/vision/splice_test.zig");
    _ = @import("engines/mlx/kv/cache_ops.zig");
    _ = @import("engines/ds4/engine.zig");
    _ = @import("host/plugins/loader.zig");
    _ = @import("sdk/util/log.zig");
    _ = @import("host/plugin_host_impl.zig");
    _ = @import("host/plugin_ane.zig");
    _ = @import("sdk/util/os_mutex.zig");
    _ = @import("sdk/util/crt.zig");
    _ = @import("sdk/util/rtfmt.zig");
    _ = @import("env.zig");
    _ = @import("sdk/util/win_io.zig");
    _ = @import("version.zig");
    _ = @import("mlx/ops.zig");
    _ = @import("mlx/kernels.zig");
    _ = @import("sdk/bases/bases.zig");
    _ = @import("engines/mlx/plugin/audit.zig");
    _ = @import("host/chat/chat.zig");
    _ = @import("engines/llama/engine.zig");
    _ = @import("host/engines/served.zig");
    _ = @import("engines/mlx/runtime/offload.zig");
    _ = @import("host/engines/stream.zig");
    _ = @import("host/engines/refusal.zig");
    _ = @import("host/engines/mlx.zig");
    _ = @import("engines/mlx/api.zig");
    _ = @import("engines/mlx/configure.zig");
    _ = @import("engines/mlx/request_policy.zig");
    _ = @import("engines/mlx/runtime/model_state.zig");
    _ = @import("host/gpu_thread/scheduler.zig");
    _ = @import("host/gpu_thread/params.zig");
    _ = @import("host/gpu_thread/requests.zig");
    _ = @import("host/api/ablation.zig");
    _ = @import("host/api/request_options.zig");
    _ = @import("host/api/schema_check.zig");
    _ = @import("sdk/util/err_name.zig");
    _ = @import("sdk/request/request.zig");
    _ = @import("sdk/spec/dflash_contract.zig");
    _ = @import("host/engines/prompt.zig");
    _ = @import("engines/mlx/runtime/engine_load.zig");
    _ = @import("host/chat/format_corpus_test.zig");
    _ = @import("host/chat/tool_traffic_replay_test.zig");
    _ = @import("engines/mlx/spec/mtp_replay_test.zig");
    _ = @import("engines/mlx/spec/mtp_lookup.zig");
    _ = @import("host/http/server.zig");
    _ = @import("engines/mlx/admission/admit.zig");
    _ = @import("engines/mlx/admission/bills.zig");
    _ = @import("engines/mlx/admission/bills_test.zig");
    _ = @import("engines/mlx/admission/context.zig");
    _ = @import("engines/mlx/admission/glm5_bill.zig");
    _ = @import("engines/mlx/admission/hot_cache_plan.zig");
    _ = @import("engines/mlx/admission/prefill_chunk.zig");
    _ = @import("sdk/kv/media_span.zig");
    _ = @import("sdk/quant/mxfp8.zig");
    _ = @import("host/api/generation_settings.zig");
    _ = @import("sdk/config/model.zig");
    _ = @import("engines/mlx/generate.zig");
    _ = @import("engines/mlx/transformer.zig");
    _ = @import("engines/mlx/vision_encoder.zig");
    _ = @import("sdk/vision/qwen_vision.zig");
    _ = @import("host/api/audio/transcription.zig");
    _ = @import("host/api/speech.zig");
    _ = @import("sdk/vision/mrope.zig");
    _ = @import("engines/ane/engine.zig");
    _ = @import("engines/mlx/plugin/fit_key.zig");
    _ = @import("engines/mlx/load/sushi_adapt.zig");
    _ = @import("engines/mlx/sampling/keyed_sample.zig");
    _ = @import("sdk/tokenizer/regex.zig");
    _ = @import("sdk/grammar/json_schema.zig");
    _ = @import("sdk/grammar/json_grammar.zig");
    _ = @import("sdk/grammar/token_mask.zig");
    _ = @import("host/api/responses/model.zig");
    _ = @import("host/http/ws.zig");
    _ = @import("sdk/spec/pld_index.zig");
    _ = @import("sdk/spec/rounds.zig");
    _ = @import("host/registry/model_settings.zig");
    _ = @import("host/registry/load_progress.zig");
    _ = @import("engines/mlx/spec/dflash/dflash.zig");
    _ = @import("engines/mlx/spec/dflash/lilicorr.zig");
    _ = @import("engines/mlx/spec/dflash/policy.zig");
    _ = @import("sdk/spec/mtp/mtp.zig");
    _ = @import("engines/mlx/spec/mtp_test.zig");
    _ = @import("engines/mlx/spec/round_cost.zig");
    _ = @import("engines/mlx/spec/mtp_group_planner.zig");
    _ = @import("sdk/bases/kda_hybrid/cuts_test.zig");
    _ = @import("sdk/tokenizer/tokenizer.zig");
    _ = @import("engines/mlx/cache/prefix_cache.zig");
    _ = @import("engines/mlx/cache/restore_dump.zig");
    _ = @import("host/metrics/metrics.zig");
    _ = @import("sdk/util/system.zig");
    _ = @import("engines/mlx/runtime/sleep_inhibit.zig");
    _ = @import("engines/mlx/cache/ssd/disk_tier.zig");
    _ = @import("engines/mlx/cache/ssd/writer.zig");
    _ = @import("host/registry/discovery.zig");
    _ = @import("host/registry/stub_cache.zig");
    _ = @import("host/registry/pack_dir.zig");
    _ = @import("host/registry/registry.zig");
    _ = @import("engines/mlx/scheduler.zig");
    if (@import("build_options").ds4_engine) {
        _ = @import("engines/ds4/ffi.zig");
        _ = @import("engines/ds4/bridge.zig");
    }
    if (@import("build_options").llama_engine) {
        _ = @import("engines/llama/ffi.zig");
        _ = @import("engines/llama/bridge.zig");
    }
    _ = @import("sdk/ane/offload.zig");
    _ = @import("engines/ane/dflash.zig");
    _ = @import("engines/ane/lm.zig");
    _ = @import("sdk/ane/mil.zig");
    _ = @import("engines/ane/pack.zig");
    _ = @import("engines/ane/prefix.zig");
    _ = @import("engines/ane/qwen35.zig");
    _ = @import("engines/ane/rt.zig");
    _ = @import("engines/mlx/cluster/distributed.zig");
    _ = @import("host/media_jobs/plugin/http.zig");
    _ = @import("host/http/multipart.zig");
    _ = @import("host/media_jobs/gen.zig");
    _ = @import("host/media_jobs/sse.zig");
    _ = @import("host/api/ollama/translate.zig");
    _ = @import("host/cli/cli.zig");
    _ = @import("host/launch/launch.zig");
    _ = @import("host/lan/lan.zig");
    _ = @import("host/api/providers.zig");
    _ = @import("host/launch/budget.zig");
    _ = @import("host/registry/modality.zig");
    _ = @import("host/api/model_row_facts.zig");
    _ = @import("mlx/ffi.zig");
    _ = @import("testing/fake_engine.zig"); // the engine contract suite against the fake engines
    _ = @import("kit_header.zig"); // the C names a released header had
    _ = @import("abi/exactness_test.zig"); // the exactness contracts (run on the fakes from fake_engine.zig)
}
