//! The model types the MLX engine serves through their plugin's table. A served type's pack its plugin declines is
//! refused by name; an mlx-stream pack goes to its stream module.

const std = @import("std");

pub const types = [_][]const u8{
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "k2_horizon",
    "spark2_5",
    "laguna",
    "flint",
    "kolibri1",
    "muse_glimmer",
    "muse_glimmer_text",
    "gpt_oss",
    "mimo_v2",
    "mimo_v2_flash",
    "bert",
    "lfm2",
    "lfm2_moe",
    "lfm3",
    "lfm3_vl",
    "lfm3_audio",
    "lfm3_omni",
    "nemotron_h",
    "nemotron_h_puzzle",
    "hy_v3",
    "deepseek_v4",
    "deepseek_v41",
    "inkling_mm_model",
    "lfm2_vl",
    "lfm2-vl",
    "nemotron_h_omni",
    "qwen3_moe",
    "qwen3_moe_text",
    "qwen3_next",
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "minicpmv4_7",
    "qwen3_asr",
    "prism_hadamard_qwen35",
    "gemma3",
    "gemma4",
    "gemma4_text",
    "gemma3_text",
    "embedding_gemma2",
    "gemma4_unified",
    "gemma4_unified_text",
    "qwen4_exp",
    "qwen4_exp_text",
    "glm5_next",
    "glm5_next_text",
    "bailing_hybrid",
};

/// Decoder types whose plugin serves the embedding arm too (a padded batch, no cache): a pack whose folder name makes
/// it an embedding model (`model.poolingFromDirName`) is served only here.
pub const decoder_embedders = [_][]const u8{"qwen3"};

pub fn servesDecoderEmbedding(model_type: []const u8) bool {
    for (decoder_embedders) |t| if (std.mem.eql(u8, t, model_type)) return true;
    return false;
}

pub fn serves(model_type: []const u8) bool {
    for (types) |t| if (std.mem.eql(u8, t, model_type)) return true;
    return false;
}

test "served: the listed types only" {
    try std.testing.expect(serves("llama") and serves("qwen2") and serves("laguna") and serves("gemma4") and serves("gemma3_text") and !serves(""));
}
