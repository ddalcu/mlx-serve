//! Pixal3D's claims and caps: MLX-free. Its pack keeps TRELLIS.2's `pipeline.json` name and is told apart by its
//! flows' `image_attn_mode: proj` (`trellis2/config.kind`); the bills are TRELLIS.2's (the same cascade and decoders).

const std = @import("std");
const abi = @import("../../abi/media.zig");
const t2_config = @import("../trellis2/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "pixal3d",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"TencentARC/Pixal3D"},
};

pub const model_types = [_][]const u8{"pixal3d"};

/// Pixal3D ships its decoders in its own ckpts/; the texture decoder sorts last.
pub const required_file = "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors";

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    const mt = model_type.slice();
    if (std.mem.eql(u8, mt, "pixal3d")) return .by_type;
    if (mt.len == 0 and t2_config.kind(dir) == .pixal3d) return .by_marker;
    return .no;
}

/// One image, or several posed views of one object, to a textured GLB.
pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .multiview = true };
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");

test "pixal3d config: a pipeline.json whose SS flow says proj is Pixal3D's, any other TRELLIS.2's" {
    const pipe = "{\"name\":\"Trellis2ImageTo3DPipeline\",\"args\":{\"models\":{\"sparse_structure_flow_model\":\"ckpts/ss\"}}}";
    const px: fixture.MemDir = .{ .files = &.{ .{ .name = "pipeline.json", .bytes = pipe }, .{ .name = "ckpts/ss.json", .bytes = "{\"args\":{\"image_attn_mode\":\"proj\"}}" } } };
    try testing.expectEqual(abi.Claim.by_marker, claims(.{}, &px.packDir()));
    try testing.expectEqual(abi.Claim.by_type, claims(.of("pixal3d"), &px.packDir()));
    const t2: fixture.MemDir = .{ .files = &.{ .{ .name = "pipeline.json", .bytes = pipe }, .{ .name = "ckpts/ss.json", .bytes = "{\"args\":{}}" } } };
    try testing.expectEqual(abi.Claim.no, claims(.{}, &t2.packDir()));
}
