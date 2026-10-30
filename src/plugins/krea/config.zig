//! Krea-2-Turbo's claims, caps and bills: MLX-free (discovery and admission run them with no device). The pack is
//! assembled with a synthesized config.json (`krea2_turbo`).

const std = @import("std");
const abi = @import("../../abi/media.zig");
const pack_bytes = @import("../../sdk/media/pack_bytes.zig");
const pack_claims = @import("../../sdk/media/pack_claims.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "krea",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"krea/Krea-2-Turbo"},
};

pub const model_types = [_][]const u8{"krea2_turbo"};

/// No marker (as discovery has always treated these packs).
pub const required_file = "";

pub const max_dim = 2048;
/// The encoder's tapped layers: one `cond_weights` value each.
pub const cond_weight_count = 12;
/// The turbo checkpoint's step count when the request names none.
pub const default_steps = 4;

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    _ = dir;
    return if (std.mem.startsWith(u8, model_type.slice(), "krea")) .by_type else .no;
}

/// Variation rides the VAE encoder; no edit training, no negative-prompt path.
pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    out.* = .{
        .img2img = pack_claims.shardsName(dir, "vae", "encoder.conv_in.weight"),
        .cond_weights = true,
        .default_guidance = 1,
        .max_dim = max_dim,
        .lora_arch = .of("krea2"),
    };
}

/// Everything stays resident; the DiT at its int8 size under `--w8a8`, unless `--ane-image` keeps it packed for
/// the ANE seam.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    return pack_bytes.residentWeights(dir, m, !m.ane_image);
}

/// No per-request term: the load bill is the model's peak.
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = dir;
    _ = m;
    _ = body;
    return 0;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");

test "krea config: claimed by type, variation only with the VAE encoder" {
    const empty: fixture.MemDir = .{ .files = &.{} };
    const dir = empty.packDir();
    try testing.expectEqual(abi.Claim.by_type, claims(.of("krea2_turbo"), &dir));
    try testing.expectEqual(abi.Claim.no, claims(.of("flux2"), &dir));
    try testing.expectEqual(abi.Claim.no, claims(.{}, &dir));
    var c: abi.MediaCaps = .{};
    caps(&dir, &c);
    try testing.expect(!c.img2img and !c.edit and !c.guidance and c.cond_weights and c.max_dim == 2048);
}
