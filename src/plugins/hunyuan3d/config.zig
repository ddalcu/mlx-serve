//! Hunyuan3D-2.1's claims, caps and bills: MLX-free. The converted pack is `config.json` (`hunyuan3d_2_1`) beside
//! `conditioner`, `dit` and `vae.safetensors`; the paint stage lives in `paint/` or a sibling folder.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const pack_bytes = @import("../../sdk/media/pack_bytes.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "hunyuan3d",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"tencent/Hunyuan3D-2.1"},
};

pub const model_types = [_][]const u8{"hunyuan3d"};

/// No marker (as discovery has always treated these packs).
pub const required_file = "";

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    _ = dir;
    return if (std.mem.startsWith(u8, model_type.slice(), "hunyuan3d")) .by_type else .no;
}

/// One image to a mesh (textured on request); no edit, LoRA or views.
pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .guidance = true, .default_guidance = 5 };
}

/// The shape engine stays resident; its DiT bills its int8 copy under `--w8a8`. A paint stage inside the pack
/// (`paint/`) is billed with it (one folder deep).
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    return pack_bytes.residentWeights(dir, m, true);
}

/// A request holds nothing the load did not bill.
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m, body };
    return 0;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");

test "hunyuan3d config: any hunyuan3d model_type is claimed; the load bills the pack's shards" {
    const mem: fixture.MemDir = .{ .files = &.{ .{ .name = "vae.safetensors", .bytes = "12345678" }, .{ .name = "paint/unet.safetensors", .bytes = "1234" } } };
    const d = mem.packDir();
    try testing.expectEqual(abi.Claim.by_type, claims(.of("hunyuan3d_2_1"), &d));
    try testing.expectEqual(abi.Claim.no, claims(.of("trellis2"), &d));
    try testing.expectEqual(@as(u64, 12), billLoad(&d, &.{}));
}
