//! FLUX.2 klein's claims, caps and bills: MLX-free (discovery and admission run them with no device). The 4B mirror
//! ships a root config.json (`flux2-klein-4b`); the mlx-community 9B conversion ships none, so its DiT's
//! shared-modulation tensor, a name no other diffusers repo carries, identifies it.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const pack_bytes = @import("../../sdk/media/pack_bytes.zig");
const pack_claims = @import("../../sdk/media/pack_claims.zig");

/// [0] is the type discovery reports for a configless (mflux) pack.
pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "flux2",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/flux2-klein-4b-4bit"},
};

pub const model_types = [_][]const u8{ "flux2-klein", "flux2" };

/// No marker (as discovery has always treated these packs).
pub const required_file = "";

/// Sides clamp to the /32 grid up to this (the 1024 trained regime plus headroom).
pub const max_dim = 1536;
/// The encoder's tapped layers (9, 18, 27): one `cond_weights` value each.
pub const cond_weight_count = 3;
pub const edit_ref_cap = 4;
/// Distilled klein's step count when the request names none.
pub const default_steps = 4;

const dit_marker = "double_stream_modulation_img";

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    const t = model_type.slice();
    if (std.mem.startsWith(u8, t, "flux2")) return .by_type;
    if (t.len == 0 and pack_claims.shardsName(dir, "transformer", dit_marker)) return .by_marker;
    return .no;
}

/// Variation and edits ride the VAE encoder, which the release ships beside the decoder; distilled klein takes
/// `guidance_scale` 1.0 (a no-op) and the undistilled base runs real CFG with it.
pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    const enc = pack_claims.shardsName(dir, "vae", "encoder.conv_in.weight");
    out.* = .{
        .img2img = enc,
        .edit = enc,
        .guidance = true,
        .cond_weights = true,
        .default_guidance = 1,
        .max_dim = max_dim,
        .edit_ref_cap = if (enc) edit_ref_cap else 0,
        .lora_arch = .of("flux2"),
    };
}

/// Everything stays resident (the text encoder too, except in low-memory mode); the DiT at its int8 copy under
/// `--w8a8`.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    return pack_bytes.residentWeights(dir, m, true);
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

test "flux2 config: claimed by type, or configless by its DiT marker" {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "transformer/model.safetensors.index.json", .bytes = "{\"weight_map\":{\"double_stream_modulation_img.linear.weight\":\"0.safetensors\"}}" }} };
    const dir = mem.packDir();
    try testing.expectEqual(abi.Claim.by_marker, claims(.{}, &dir));
    try testing.expectEqual(abi.Claim.by_type, claims(.of("flux2-klein-9b"), &dir));
    try testing.expectEqual(abi.Claim.no, claims(.of("krea2"), &dir));
    const empty: fixture.MemDir = .{ .files = &.{} };
    const none = empty.packDir();
    try testing.expectEqual(abi.Claim.no, claims(.{}, &none));
    var c: abi.MediaCaps = .{};
    caps(&none, &c);
    try testing.expect(!c.img2img and !c.edit and c.guidance and c.max_dim == 1536);
}
