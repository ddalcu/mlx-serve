//! Breeze-TTS-2's claims, caps and bills, MLX-free: a `breeze_tts` config.json, complete once the Mimi
//! audio_tokenizer lands; the speech route; backbone, depth decoder, text encoder and codec all resident.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const bill = @import("../../sdk/media/bill.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "breeze_tts",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/Breeze-TTS-2-mlx"},
};

pub const model_types = [_][]const u8{"breeze_tts"};

pub const claims = sdk.mediaClaimsByType(&model_types);

/// The Mimi codec lives in `audio_tokenizer/`; a partial pull can miss the subdir.
pub const required_file = "audio_tokenizer/model.safetensors";

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
