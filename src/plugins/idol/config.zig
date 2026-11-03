//! Idol's claims, caps and bills: MLX-free (discovery and admission run them with no device). A pack is the release's
//! own folder plus `dinov3/` and `vae/`; the release's `config.json` has no `model_type`, so it is told apart by its
//! denoiser's class name (a converted pack also says `model_type: idol`).

const std = @import("std");
const abi = @import("../../abi/media.zig");
const pack_claims = @import("../../sdk/media/pack_claims.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "idol",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"speridlabs/idol"},
};

pub const model_types = [_][]const u8{"idol"};

/// The FLUX.2 VAE sorts last of what a pull lands.
pub const required_file = "vae/model.safetensors";

pub const DEFAULT_STEPS: u32 = 50;
pub const MAX_STEPS: u32 = 200;
pub const MIN_GAUSSIANS: u32 = 32768;
pub const MAX_GAUSSIANS: u32 = 1048576;
pub const DEFAULT_GAUSSIANS: u32 = 262144;

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    const mt = model_type.slice();
    if (std.mem.eql(u8, mt, "idol")) return .by_type;
    if (mt.len == 0 and pack_claims.jsonStringIs(dir, "config.json", &.{ "denoiser", "name" }, "LatentSeqMMFlowModel")) return .by_marker;
    return .no;
}

/// One image to a 3D Gaussian splat; no edit, guidance image or LoRA.
pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{};
}

/// What the pack's files weigh: its weights stay resident (denoiser, both decoders, BiRefNet, DINOv3, the VAE).
fn packBytes(dir: *const abi.PackDir) u64 {
    var total: u64 = 0;
    for ([_][]const u8{ "denoiser.safetensors", "decoder.safetensors", "decoder_gs.safetensors", "birefnet.safetensors", "dinov3/model.safetensors", "vae/model.safetensors" }) |f| {
        total += dir.fileBytes(f);
    }
    return total;
}

/// Weights as stored, plus the transient of casting a stored f32 pack to f16 at load.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    _ = m;
    const w = packBytes(dir);
    return if (w == 0) 0 else w + w / 8 + (1 << 30);
}

/// One request's activations on top of the resident weights: DINOv3 and the VAE over a 1024 image, the denoiser at
/// ~12k tokens, the Gaussian decoder at one token per anchor.
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m };
    const n = gaussiansOf(body.slice());
    return (3 << 30) + @as(u64, n) * 4096;
}

/// The body's `num_gaussians` (the default when absent or not a whole number).
pub fn gaussiansOf(body: []const u8) u32 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), body, .{}) catch return DEFAULT_GAUSSIANS;
    const r = if (v == .object) v.object.get("num_gaussians") else null;
    if (r == null or r.? != .integer or r.?.integer < 0 or r.?.integer > MAX_GAUSSIANS) return DEFAULT_GAUSSIANS;
    return @intCast(r.?.integer);
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");

test "idol config: the release's config.json is claimed by its denoiser class, a converted pack by type" {
    const rel: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = "{\"denoiser\":{\"name\":\"LatentSeqMMFlowModel\",\"args\":{}}}" }} };
    try testing.expectEqual(abi.Claim.by_marker, claims(.{}, &rel.packDir()));
    try testing.expectEqual(abi.Claim.by_type, claims(.of("idol"), &rel.packDir()));
    const other: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = "{\"denoiser\":{\"name\":\"Other\"}}" }} };
    try testing.expectEqual(abi.Claim.no, claims(.{}, &other.packDir()));
    const none: fixture.MemDir = .{ .files = &.{} };
    try testing.expectEqual(abi.Claim.no, claims(.{}, &none.packDir()));
}

test "idol config: a request bills by its Gaussian count" {
    const dir: abi.PackDir = undefined;
    try testing.expectEqual(DEFAULT_GAUSSIANS, gaussiansOf("{}"));
    try testing.expectEqual(@as(u32, 65536), gaussiansOf("{\"num_gaussians\":65536}"));
    try testing.expectEqual(DEFAULT_GAUSSIANS, gaussiansOf("{\"num_gaussians\":-1}"));
    try testing.expect(billRequest(&dir, &.{}, .of("{\"num_gaussians\":1048576}")) > billRequest(&dir, &.{}, .of("{}")));
}
