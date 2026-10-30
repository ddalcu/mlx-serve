//! MiniMax Music 3's claims, caps and bills, MLX-free: a `minimax_music3` config.json, complete once the vocoder
//! lands; the music route; every stage resident for the engine's lifetime plus the AR stage's working set.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const pack_bytes = @import("../../sdk/media/pack_bytes.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "music3",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"ddalcu/MiniMax-Music3-MLX-Serve-8bit"},
};

pub const model_types = [_][]const u8{"minimax_music3"};

pub const claims = sdk.mediaClaimsByType(&model_types);

pub const required_file = "vocoder.safetensors";

/// The non-weight working set at the request caps: the batch-2 KV cache for 36 layers at 9000 frames + 5000 prompt
/// tokens (~4.1 GB), the bf16 frame-hidden buffer (~0.6 GB), and the DiT/vocoder window transients.
pub const gen_buffer_bytes: u64 = 6 * 1024 * 1024 * 1024;

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .routes = .{ .music = true } };
}

pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    const weights = pack_bytes.residentWeights(dir, m, true);
    if (weights == 0) return 0; // unknown dir: never block
    return weights + gen_buffer_bytes;
}

pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m, body };
    return 0;
}
