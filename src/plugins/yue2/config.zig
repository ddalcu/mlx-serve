//! YuE2's claims, caps and bills, MLX-free: a `yue2` config.json, complete once the folded-in VAE lands; the music
//! route; both weight files resident plus the working set the 9000-frame cap needs.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const bill = @import("../../sdk/media/bill.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "yue2",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"ddalcu/YuE2-3B-MLX-Serve-8bit"},
};

pub const model_types = [_][]const u8{"yue2"};

pub const claims = sdk.mediaClaimsByType(&model_types);

/// The upstream repo has no decoder: only a pack with the folded-in VAE can run.
pub const required_file = "vae.safetensors";

/// The non-weight working set at the 9000-frame cap: two AR K/V caches (~115 KB a token each, ~4 GB with a long
/// score), then the NAR prefix K/V (~1.5 GB) + canvas activations and one f32 VAE tile (~3 GB). The stages run one
/// after another; billed at the larger.
pub const gen_buffer_bytes: u64 = 6 * 1024 * 1024 * 1024;

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .routes = .{ .music = true } };
}

pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    _ = m;
    const weights = bill.safetensorsIn(dir, "");
    if (weights == 0) return 0; // unknown dir: never block
    return weights + gen_buffer_bytes;
}

pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m, body };
    return 0;
}
