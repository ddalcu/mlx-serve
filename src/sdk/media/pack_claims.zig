//! The marker reads a configless media pack is claimed by (MLX-free): a diffusers `model_index.json`'s
//! `_class_name`, a JSON file's value at a key path, or a tensor name in a folder's shards.

const std = @import("std");
const abi = @import("../../abi/media.zig");

const PackDir = abi.PackDir;

/// `rel` parsed, into `arena`; null when it is missing, past 1 MiB or not JSON.
pub fn readJson(arena: std.mem.Allocator, dir: *const PackDir, rel: []const u8) ?std.json.Value {
    const buf = arena.alloc(u8, 1 << 20) catch return null;
    var why: abi.Verdict = .{};
    const bytes = dir.readFile(rel, buf, &why) orelse return null;
    return std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{}) catch null;
}

/// The value at `keys` under `v`, or null.
pub fn path(v: std.json.Value, keys: []const []const u8) ?std.json.Value {
    var cur = v;
    for (keys) |k| {
        if (cur != .object) return null;
        cur = cur.object.get(k) orelse return null;
    }
    return cur;
}

/// Whether `rel`'s string at `keys` equals `want`.
pub fn jsonStringIs(dir: *const PackDir, rel: []const u8, keys: []const []const u8, want: []const u8) bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = readJson(arena.allocator(), dir, rel) orelse return false;
    const s = path(v, keys) orelse return false;
    return s == .string and std.mem.eql(u8, s.string, want);
}

/// Whether `model_index.json` names the diffusers pipeline `class_name`.
pub fn modelIndexClassIs(dir: *const PackDir, class_name: []const u8) bool {
    return jsonStringIs(dir, "model_index.json", &.{"_class_name"}, class_name);
}

/// Whether a tensor name in `folder` contains `needle`: its `model.safetensors.index.json` when it has one, else
/// every shard's header (never the weights).
pub fn shardsName(dir: *const PackDir, folder: []const u8, needle: []const u8) bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const buf = a.alloc(u8, 16 << 20) catch return false;
    var pbuf: [1024]u8 = undefined;
    var why: abi.Verdict = .{};
    const index = std.fmt.bufPrint(&pbuf, "{s}/model.safetensors.index.json", .{folder}) catch return false;
    if (dir.readFile(index, buf, &why)) |idx| return std.mem.indexOf(u8, idx, needle) != null;
    const names = a.alloc(u8, 64 << 10) catch return false;
    var it = dir.entries(folder, names) orelse return false;
    while (it.next()) |e| {
        if (!std.mem.endsWith(u8, e, ".safetensors")) continue;
        const rel = std.fmt.bufPrint(&pbuf, "{s}/{s}", .{ folder, e }) catch continue;
        var len: usize = 0;
        if (!dir.safetensors_header(dir.ctx, .of(rel), buf.ptr, buf.len, &len).isOk()) continue;
        if (std.mem.indexOf(u8, buf[0..len], needle) != null) return true;
    }
    return false;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../testing/fixture.zig");

test "pack claims: a model_index class and a nested key are read, anything else is no" {
    const mem: fixture.MemDir = .{ .files = &.{
        .{ .name = "model_index.json", .bytes = "{\"_class_name\":\"MOSSImagePipeline\"}" },
        .{ .name = "pipeline.json", .bytes = "{\"args\":{\"models\":{\"x\":\"ckpts/x\"}}}" },
    } };
    const dir = mem.packDir();
    try testing.expect(modelIndexClassIs(&dir, "MOSSImagePipeline") and !modelIndexClassIs(&dir, "Other"));
    try testing.expect(jsonStringIs(&dir, "pipeline.json", &.{ "args", "models", "x" }, "ckpts/x"));
    try testing.expect(!jsonStringIs(&dir, "missing.json", &.{"a"}, "b"));
}

test "pack claims: a tensor name is found through the index or a shard header" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t = @import("../testing/tensors.zig").Tensors.init(arena.allocator(), 1);
    try t.fill("double_stream_modulation_img.linear.weight", .{}, &.{1}, .F32, 0);
    const mem: fixture.MemDir = .{ .files = &.{
        .{ .name = "transformer/0.safetensors", .bytes = try t.bytes() },
        .{ .name = "indexed/model.safetensors.index.json", .bytes = "{\"weight_map\":{\"x.encoder.conv_in.weight\":\"0.safetensors\"}}" },
    } };
    const dir = mem.packDir();
    try testing.expect(shardsName(&dir, "transformer", "double_stream_modulation_img"));
    try testing.expect(!shardsName(&dir, "transformer", "encoder.conv_in"));
    try testing.expect(shardsName(&dir, "indexed", "encoder.conv_in"));
    try testing.expect(!shardsName(&dir, "missing", "x"));
}
