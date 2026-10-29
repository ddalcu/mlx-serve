//! Qwen3-TTS's claims, caps and bills, MLX-free: a `qwen3_tts` config.json, complete once the speech tokenizer's
//! decoder lands; the speech route; the talker, predictor and codec decoder all resident.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const bill = @import("../../sdk/media/bill.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "qwen3_tts",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"},
};

pub const model_types = [_][]const u8{"qwen3_tts"};

pub const claims = sdk.mediaClaimsByType(&model_types);

/// No completeness marker: the pack counts as discovered (legacy had none).
pub const required_file = "";

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .routes = .{ .speech = true } };
}

pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    _ = m;
    return bill.safetensorsIn(dir, "");
}

pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m, body };
    return 0;
}
