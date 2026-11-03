//! MOSS-Image's claims, caps and bills: MLX-free (discovery and admission run them with no device). The release
//! ships no root config.json; its diffusers `model_index.json` names the pipeline.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const pack_bytes = @import("../../sdk/media/pack_bytes.zig");
const pack_claims = @import("../../sdk/media/pack_claims.zig");
const image_request = @import("../../sdk/media/image_request.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "moss_image",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/MOSS-Image-SFT-bf16"},
};

pub const model_types = [_][]const u8{"moss_image"};

/// Every pack names its VAE weights so; `model_index.json` lands first in a pull.
pub const required_file = "vae/diffusion_pytorch_model.safetensors";

pub const DEFAULT_GUIDANCE: f32 = 3.0;
pub const VAE_DOWNSAMPLE: u32 = 16;
pub const max_dim = 2048;

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    if (std.mem.eql(u8, model_type.slice(), model_types[0])) return .by_type;
    if (model_type.len == 0 and pack_claims.modelIndexClassIs(dir, "MOSSImagePipeline")) return .by_marker;
    return .no;
}

/// Text to image only (the release has no edit or variation training); undistilled, so real CFG at 3 by default.
pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .guidance = true, .default_guidance = DEFAULT_GUIDANCE, .max_dim = max_dim, .lora_arch = .of("generic") };
}

/// The peak is the f32 full-resolution VAE stage, linear in the canvas; a full 4096-token prompt peaks earlier and
/// only wins on small canvases.
const GEN_TOKEN_BYTES: u64 = 1536 << 10;
const GEN_BASE_BYTES: u64 = 512 << 20;
const GEN_PROMPT_BYTES: u64 = 3 << 30;

pub fn genTransientBytes(width: u32, height: u32) u64 {
    const latents: u64 = @as(u64, width / VAE_DOWNSAMPLE) * (height / VAE_DOWNSAMPLE);
    return @max(GEN_PROMPT_BYTES, GEN_BASE_BYTES + latents * GEN_TOKEN_BYTES);
}

/// Everything stays resident (the Qwen3.5 encoder too) and nothing converts, plus a default-size request.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    const sum = pack_bytes.residentWeights(dir, m, false);
    if (sum == 0) return 0; // unknown pack: never block
    return sum + genTransientBytes(1024, 1024);
}

/// The request's canvas transient, against the GPU's live headroom (the load reserved a default-size one).
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m, body };
    return 0;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");

test "moss_image config: claimed by type or by its model_index; a canvas is the job's to refuse" {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "model_index.json", .bytes = "{\"_class_name\":\"MOSSImagePipeline\"}" }} };
    const dir = mem.packDir();
    try testing.expectEqual(abi.Claim.by_marker, claims(.{}, &dir));
    try testing.expectEqual(abi.Claim.by_type, claims(.of("moss_image"), &dir));
    try testing.expectEqual(abi.Claim.no, claims(.of("flux2"), &dir));
    const m: abi.Machine = .{};
    try testing.expectEqual(@as(u64, 0), billRequest(&dir, &m, .of("{\"size\":\"2048x2048\"}")));
    try testing.expectEqual(@as(u64, 3 << 30), genTransientBytes(256, 256));
}

test "moss_image config: the text-to-image transient covers the measured peak at every canvas" {
    const GiB: f64 = 1 << 30;
    // Peaks above the bf16 weights (M4 Max): the VAE sets them, except a full-length prompt at 512².
    const measured = [_]struct { w: u32, h: u32, peak: f64 }{
        .{ .w = 512, .h = 512, .peak = 2.88 },  .{ .w = 1024, .h = 1024, .peak = 6.15 },
        .{ .w = 1456, .h = 1456, .peak = 11.66 }, .{ .w = 1792, .h = 1792, .peak = 17.66 },
    };
    for (measured) |m| {
        const gib = @as(f64, @floatFromInt(genTransientBytes(m.w, m.h))) / GiB;
        try testing.expect(gib >= m.peak and gib < m.peak * 1.1);
    }
}

