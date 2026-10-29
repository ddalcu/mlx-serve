//! TRELLIS.2's claims, caps and bills: MLX-free (discovery and admission run them with no device). A pack has no
//! root config.json: `pipeline.json` names the pipeline, and Pixal3D keeps that name, told apart by its flows'
//! `image_attn_mode: proj` (`kind`).

const std = @import("std");
const abi = @import("../../abi/media.zig");
const pack_claims = @import("../../sdk/media/pack_claims.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "trellis2",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"microsoft/TRELLIS.2-4B"},
};

pub const model_types = [_][]const u8{"trellis2"};

/// The SS decoder sorts last of the weights a pull lands (`pipeline.json` comes first).
pub const required_file = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors";

pub const DEFAULT_RESOLUTION: u32 = 1024;

pub const Kind = enum { trellis2, pixal3d };

/// Which of the two pipelines `pipeline.json` describes, null for neither.
pub fn kind(dir: *const abi.PackDir) ?Kind {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pipe = pack_claims.readJson(a, dir, "pipeline.json") orelse return null;
    const name = pack_claims.path(pipe, &.{"name"}) orelse return null;
    if (name != .string or !std.mem.eql(u8, name.string, "Trellis2ImageTo3DPipeline")) return null;
    const ss = pack_claims.path(pipe, &.{ "args", "models", "sparse_structure_flow_model" }) orelse return .trellis2;
    if (ss != .string) return .trellis2;
    const cfg_path = std.fmt.allocPrint(a, "{s}.json", .{ss.string}) catch return .trellis2;
    const cfg = pack_claims.readJson(a, dir, cfg_path) orelse return .trellis2;
    const mode = pack_claims.path(cfg, &.{ "args", "image_attn_mode" }) orelse return .trellis2;
    return if (mode == .string and std.mem.eql(u8, mode.string, "proj")) .pixal3d else .trellis2;
}

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    const mt = model_type.slice();
    if (std.mem.eql(u8, mt, "trellis2")) return .by_type;
    if (mt.len == 0 and kind(dir) == .trellis2) return .by_marker;
    return .no;
}

/// One image to a textured GLB; no edit, guidance knob or LoRA.
pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{};
}

/// Whether the pack serves 512: its `pipeline.json` names a 512 texture flow that is on disk.
pub fn serves512(dir: *const abi.PackDir) bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pipe = pack_claims.readJson(a, dir, "pipeline.json") orelse return false;
    const rel = pack_claims.path(pipe, &.{ "args", "models", "tex_slat_flow_model_512" }) orelse return false;
    if (rel != .string or rel.string.len == 0) return false;
    const path = std.fmt.allocPrint(a, "{s}.safetensors", .{rel.string}) catch return false;
    return dir.has(path);
}

/// The whole-request MLX peak at each served resolution, for a car-sized subject (TRELLIS.2 and Pixal3D run the
/// same cascade and decoders): 512 skips the 1024 cascade, and above that the decoders' per-voxel features grow with
/// the surface.
pub fn peakBytes(resolution: u32) u64 {
    const mib: u64 = switch (resolution) {
        512 => 5800,
        1024 => 23000,
        else => 35800,
    };
    return mib << 20;
}

/// The load preflight's requirement for `weights` (`host/registry/preflight.loadRequirementBytes`).
fn loadRequirement(weights: u64) u64 {
    return weights + @min(weights / 8, 6 << 30) + (1 << 30);
}

/// Every stage loads its weights and frees them, so the load bill is the default request's peak (a load evicts
/// idle models for it); a GPU whose ceiling cannot pass that through the preflight bills 512 where the pack serves
/// it, and each request is still refused by name past its own peak.
pub fn loadBill(has512: bool, gpu_ceiling: u64) u64 {
    const def = peakBytes(DEFAULT_RESOLUTION);
    return if (gpu_ceiling != 0 and loadRequirement(def) > gpu_ceiling and has512) peakBytes(512) else def;
}

pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    if (!dir.has("pipeline.json")) return 0; // unknown pack: never block
    return loadBill(serves512(dir), m.gpu_working_set);
}

/// The request's own whole peak: its resolution moves it by 12 GB.
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m };
    return peakBytes(resolutionOf(body.slice()));
}

/// The body's `resolution` (the default when absent or not a whole number; an unservable one bills the largest
/// cascade, and the job refuses it).
pub fn resolutionOf(body: []const u8) u32 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), body, .{}) catch return DEFAULT_RESOLUTION;
    const r = if (v == .object) v.object.get("resolution") else null;
    if (r == null or r.? != .integer or r.?.integer < 0 or r.?.integer > std.math.maxInt(u32)) return DEFAULT_RESOLUTION;
    return @intCast(r.?.integer);
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");

const pipe_json = "{\"name\":\"Trellis2ImageTo3DPipeline\",\"args\":{\"models\":{\"sparse_structure_flow_model\":\"ckpts/ss\",\"tex_slat_flow_model_512\":\"ckpts/tex512\"}}}";

test "trellis2 config: pipeline.json claims the pack; a proj flow is Pixal3D's" {
    const t2: fixture.MemDir = .{ .files = &.{ .{ .name = "pipeline.json", .bytes = pipe_json }, .{ .name = "ckpts/ss.json", .bytes = "{\"args\":{}}" } } };
    const d = t2.packDir();
    try testing.expectEqual(abi.Claim.by_marker, claims(.{}, &d));
    try testing.expectEqual(abi.Claim.by_type, claims(.of("trellis2"), &d));
    const px: fixture.MemDir = .{ .files = &.{ .{ .name = "pipeline.json", .bytes = pipe_json }, .{ .name = "ckpts/ss.json", .bytes = "{\"args\":{\"image_attn_mode\":\"proj\"}}" } } };
    try testing.expectEqual(Kind.pixal3d, kind(&px.packDir()).?);
    try testing.expectEqual(abi.Claim.no, claims(.{}, &px.packDir()));
    const none: fixture.MemDir = .{ .files = &.{} };
    try testing.expect(kind(&none.packDir()) == null);
}

test "trellis2 config: the load bills the default peak, 512 where the GPU cannot pass it; a request its own" {
    const with512: fixture.MemDir = .{ .files = &.{ .{ .name = "pipeline.json", .bytes = pipe_json }, .{ .name = "ckpts/tex512.safetensors", .bytes = "x" } } };
    const d = with512.packDir();
    try testing.expect(serves512(&d));
    const roomy: abi.Machine = .{ .gpu_working_set = 64 << 30 };
    const small: abi.Machine = .{ .gpu_working_set = 16 << 30 };
    try testing.expectEqual(peakBytes(1024), billLoad(&d, &roomy));
    try testing.expectEqual(peakBytes(512), billLoad(&d, &small));
    try testing.expectEqual(peakBytes(1536), billRequest(&d, &roomy, .of("{\"resolution\":1536}")));
    try testing.expectEqual(peakBytes(512), billRequest(&d, &roomy, .of("{\"resolution\":512}")));
    try testing.expectEqual(peakBytes(1024), billRequest(&d, &roomy, .of("{}")));
    const without: fixture.MemDir = .{ .files = &.{.{ .name = "pipeline.json", .bytes = pipe_json }} };
    try testing.expect(!serves512(&without.packDir()));
    try testing.expectEqual(peakBytes(1024), billLoad(&without.packDir(), &small));
}
