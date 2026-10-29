//! ACE-Step's claims, caps and bills, MLX-free: an `acestep` config.json, complete once the text encoder lands; the
//! music route; DiT + condition encoder, VAE and text encoder resident for the engine's lifetime.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const pack_bytes = @import("../../sdk/media/pack_bytes.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "acestep",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit"},
};

pub const model_types = [_][]const u8{"acestep"};

pub const claims = sdk.mediaClaimsByType(&model_types);

/// The text encoder is a subdir a partial pull can miss.
pub const required_file = "text_encoder/model.safetensors";

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .routes = .{ .music = true } };
}

/// The weights resident: the DiT at its int8 copy under `--w8a8`, packed when `--ane-audio` reads it.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    return pack_bytes.residentWeights(dir, m, !m.ane_audio);
}

pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m, body };
    return 0;
}
