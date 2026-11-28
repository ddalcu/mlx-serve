//! Split unit-test root: LLM model plugins, clef, proof/served tests and model conformance.

// pub: Sushi's EXL3 module (src/sdk/quant/exl3/sushi) reaches these through its host root.
pub const mlx = @import("mlx/ffi.zig");
pub const io_util = @import("sdk/util/io_util.zig");
pub const log = @import("sdk/util/log.zig");
pub const env = @import("env.zig");
comptime {
    _ = @import("plugins/registry_test_core.zig");
    _ = @import("engines/mlx/api.zig");
    @import("plugin_host_check.zig").reexports(@This());
}

/// The test process caps MLX's buffer pool as the server's `main` does; `build/test_runner.zig` calls it first.
export fn msv_test_process_init() void {
    @import("mlx/state.zig").applyMlxCacheLimit();
}

test {
    _ = @import("host/media_jobs/plugin/clef_flip_test.zig");
    _ = @import("plugins/spark2_5/proof_test.zig");
    _ = @import("plugins/spark2_5/bases_test.zig");
    _ = @import("plugins/laguna/proof_test.zig");
    _ = @import("plugins/embedding_gemma/proof_test.zig");
    _ = @import("plugins/kolibri1/proof_test.zig");
    _ = @import("plugins/gpt_oss/proof_test.zig");
    _ = @import("plugins/qwen3_next/proof_test.zig");
    _ = @import("plugins/hy_v3/proof_test.zig");
    _ = @import("plugins/hy_v3/mtp_head_test.zig");
    _ = @import("plugins/gemma3/proof_test.zig");
    _ = @import("plugins/inkling/proof_test.zig");
    _ = @import("plugins/inkling2/proof_test.zig");
    _ = @import("plugins/inkling2/served_test.zig");
    _ = @import("plugins/nemotron_h/proof_test.zig");
    _ = @import("plugins/nemotron_h/mtp_head_test.zig");
    _ = @import("plugins/gemma4/proof_test.zig");
    _ = @import("plugins/k2_horizon/proof_test.zig");
    _ = @import("plugins/embedding_gemma2/proof_test.zig");
    _ = @import("plugins/lfm2/proof_test.zig");
    _ = @import("plugins/lfm3/proof_test.zig");
    _ = @import("plugins/lfm3/served_test.zig");
    _ = @import("plugins/nemotron_transcribe/proof_test.zig");
    _ = @import("plugins/qwen3/proof_test.zig");
    _ = @import("plugins/prism_hadamard_qwen35/proof_test.zig");
    _ = @import("plugins/mimo_v2/proof_test.zig");
    _ = @import("plugins/mimo_v2/mtp_head_test.zig");
    _ = @import("plugins/mimo_v2/sushi_test.zig");
    _ = @import("plugins/bert/proof_test.zig");
    _ = @import("plugins/deepseek_v4/proof_test.zig");
    _ = @import("plugins/deepseek_v4/served_test.zig");
    _ = @import("plugins/deepseek_v41/proof_test.zig");
    _ = @import("plugins/deepseek_v41/draft_test.zig");
    _ = @import("plugins/muse_glimmer/proof_test.zig");
    _ = @import("plugins/qwen3_asr/proof_test.zig");
    _ = @import("plugins/qwen3_5/proof_test.zig");
    _ = @import("plugins/qwen3_5/served_test.zig");
    _ = @import("plugins/qwen3_5/mtp_test.zig");
    _ = @import("plugins/qwen3_5/ane_test.zig");
    _ = @import("plugins/qwen3_5/streamed_test.zig");
    _ = @import("plugins/flint/proof_test.zig");
    _ = @import("plugins/llama/proof_test.zig");
    _ = @import("plugins/registry.zig");
    _ = @import("engines/mlx/plugin/proof.zig");
    _ = @import("plugins/qwen3/plugin.zig");
    _ = @import("plugins/qwen3_5/plugin.zig");
    _ = @import("plugins/qwen3_next/plugin.zig");
    _ = @import("plugins/qwen3_asr/plugin.zig");
    _ = @import("plugins/nemotron_h/radio_vision.zig");
    _ = @import("plugins/muse_glimmer/vision.zig");
    _ = @import("plugins/lfm2/vision.zig");
    _ = @import("plugins/glm5_next/vision.zig");
    _ = @import("plugins/deepseek_v41/vision.zig");
    _ = @import("plugins/qwen3_5/minicpm_vision.zig");
    _ = @import("plugins/qwen3_5/minicpm_canvas.zig");
    _ = @import("plugins/qwen3_asr/audio_tower.zig");
    _ = @import("plugins/qwen3_asr/forced_align.zig");
    _ = @import("plugins/glm5_next/glm5_next.zig");
    _ = @import("plugins/glm5_next/dsa_nax.zig");
    _ = @import("plugins/prism_hadamard_qwen35/mtp_graft.zig");
    _ = @import("plugins/gemma4/assistant_drafter.zig");
    _ = @import("plugins/deepseek_v4/deepseek_v4.zig");
    _ = @import("plugins/deepseek_v4/model_config.zig");
    _ = @import("plugins/glm5_next/model_config.zig");
    _ = @import("plugins/glm5_next/proof_test.zig");
    _ = @import("plugins/glm5_next/served_test.zig");
    _ = @import("plugins/bailing_hybrid/bill_test.zig");
    _ = @import("engines/mlx/plugin/deepseek_test.zig");
    _ = @import("plugins/deepseek_v41/engram.zig");
    _ = @import("plugins/deepseek_v41/deepseek_v41.zig");
    _ = @import("plugins/qwen4_exp/qwen4_exp.zig");
    _ = @import("plugins/qwen4_exp/ple_gpu.zig");
    _ = @import("plugins/clef/clef.zig");
    _ = @import("plugins/clef/http_test.zig");
}
