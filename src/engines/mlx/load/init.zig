//! engines/mlx/load/init: a pack no plugin serves, refused by name.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const Weights = @import("../../../sdk/config/model.zig").Weights;
const weightsHaveDenseAttnProj = @import("../../../sdk/blocks/attention/decode_attn_quant.zig").weightsHaveDenseAttnProj;
const Transformer = @import("../transformer.zig").Transformer;

/// A pack no plugin serves, refused by name.
pub fn init(io: std.Io, allocator: std.mem.Allocator, config: ModelConfig) !Transformer {
    _ = io;
    _ = allocator;
    log.err("[load] {s}: no plugin serves this pack\n", .{config.model_type});
    return error.PluginDeclinedNoLegacy;
}

/// What `/props` `decode_attn_quant` reports: a MoE forward's dense attention projections (the plugin load sets it too).
pub fn denseAttnProj(config: *const ModelConfig, moe_forward: bool, weights: *const Weights) bool {
    return attnProjRouted(config, moe_forward) and weightsHaveDenseAttnProj(weights);
}

/// `attnProj` is reached only from the MoE/qwen4 forward, and not by its MLA or Gemma 4 arms.
fn attnProjRouted(config: *const ModelConfig, moe_forward: bool) bool {
    return moe_forward and !config.isMla() and !config.isGemma4Layers();
}

test "attnProj is routed on a MoE forward that is neither MLA nor Gemma 4" {
    try std.testing.expect(attnProjRouted(&.{ .model_type = "laguna" }, true));
    try std.testing.expect(!attnProjRouted(&.{ .model_type = "llama" }, false));
    try std.testing.expect(!attnProjRouted(&.{ .model_type = "gemma4" }, true));
    try std.testing.expect(!attnProjRouted(&.{ .model_type = "bailing_hybrid", .mla_kv_lora_rank = 512 }, true));
}
