//! Kokoro's claims, caps and bills, MLX-free: a `kokoro` config.json, the speech route, every weight resident.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const bill = @import("../../sdk/media/bill.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "kokoro",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"ddalcu/Kokoro-82M-MLX-Serve"},
};

pub const model_types = [_][]const u8{"kokoro"};

pub const claims = sdk.mediaClaimsByType(&model_types);

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .routes = .{ .speech = true } };
}

/// The trunk and the voice table stay resident; a request adds nothing measured.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    _ = m;
    return bill.safetensorsIn(dir, "");
}

pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m, body };
    return 0;
}
