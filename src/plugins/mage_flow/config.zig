//! MageFlow's claims, caps and bills: MLX-free (discovery and admission run them with no device). The official
//! diffusers layout ships no root config.json; its `model_index.json` names the pipeline.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const pack_bytes = @import("../../sdk/media/pack_bytes.zig");
const pack_claims = @import("../../sdk/media/pack_claims.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "mage_flow",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"microsoft/Mage-Flow-Turbo"},
};

pub const model_types = [_][]const u8{ "mage_flow", "mageflow" };

/// No marker (as discovery has always treated these packs).
pub const required_file = "";

pub const max_dim = 2048;
/// References one edit takes, the primary included.
pub const edit_ref_cap = 4;

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    const mt = model_type.slice();
    if (std.mem.startsWith(u8, mt, "mage_flow") or std.mem.eql(u8, mt, "mageflow")) return .by_type;
    if (mt.len == 0 and (pack_claims.modelIndexClassIs(dir, "MageFlowPipeline") or isVersioned(dir))) return .by_marker;
    return .no;
}

/// A `model_index.json` that carries `_mage_flow_version`, whatever its class name.
fn isVersioned(dir: *const abi.PackDir) bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = pack_claims.readJson(arena.allocator(), dir, "model_index.json") orelse return false;
    return v == .object and v.object.get("_mage_flow_version") != null;
}

/// Turbo text to image; the Edit checkpoint's multi-reference editor is decided at load by the pack's name (both
/// releases carry the same class and the tower weights), so `edit` here says only that the pack may edit.
pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .edit = true, .edit_raw_bytes = true, .max_dim = max_dim, .edit_ref_cap = edit_ref_cap };
}

/// The whole pipeline stays resident; under `--w8a8` the DiT is billed at its int8 size.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    return pack_bytes.residentWeights(dir, m, true);
}

pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = dir;
    _ = m;
    _ = body;
    return 0;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");

test "mage_flow config: claimed by either spelling or by its model_index" {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "model_index.json", .bytes = "{\"_class_name\":\"X\",\"_mage_flow_version\":\"0.1.0\"}" }} };
    const dir = mem.packDir();
    try testing.expectEqual(abi.Claim.by_marker, claims(.{}, &dir));
    try testing.expectEqual(abi.Claim.by_type, claims(.of("mageflow"), &dir));
    try testing.expectEqual(abi.Claim.by_type, claims(.of("mage_flow_edit"), &dir));
    try testing.expectEqual(abi.Claim.no, claims(.of("krea2_turbo"), &dir));
    const other: fixture.MemDir = .{ .files = &.{.{ .name = "model_index.json", .bytes = "{\"_class_name\":\"MageFlowPipeline\"}" }} };
    const od = other.packDir();
    try testing.expectEqual(abi.Claim.by_marker, claims(.{}, &od));
}
