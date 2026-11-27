//! HeartMuLa's claims, caps and bills, MLX-free: a `heartmula` config.json, complete once the codec lands; the
//! music route; both weight files resident plus the larger of the two stages' working sets.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const bill = @import("../../sdk/media/bill.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "heartmula",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"ddalcu/HeartMuLa-3B-MLX-Serve-8bit"},
};

pub const model_types = [_][]const u8{"heartmula"};

pub const claims = sdk.mediaClaimsByType(&model_types);

/// The converter writes the codec last: a pack without it is still converting.
pub const required_file = "codec.safetensors";

/// The non-weight working set, billed at the larger stage (the AR K/V is freed before the codec runs): the batch-2
/// backbone K/V at the 8192-position cap (~1.8 GB), or one decode window (the DiT's batch-2 activations, then the
/// f32 scalar decoder at 1.4 M samples a band, ~3 GB).
pub const gen_buffer_bytes: u64 = 4 * 1024 * 1024 * 1024;

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
