//! A config.json with overrides deep-merged in (`--config-overrides`, a plugin that re-binds a trunk).

const std = @import("std");
const log = @import("../util/log.zig");

/// Deep-merge `overrides` into a config.json document: objects merge key by key
/// — so `{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4}}}`
/// keeps every sibling it passes through, exactly like vLLM's
/// `_apply_dict_overrides` — and anything else replaces. The whole merge lives
/// in an arena that dies before this returns; only the re-serialized bytes (in
/// `allocator`) escape.
pub fn mergeConfigJson(allocator: std.mem.Allocator, base: []const u8, overrides: []const u8) ![]const u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dst = try std.json.parseFromSliceLeaky(std.json.Value, a, base, .{});
    const src = try std.json.parseFromSliceLeaky(std.json.Value, a, overrides, .{});
    if (dst != .object or src != .object) return error.ConfigOverridesMustBeObject;
    if (src.object.contains("qwen4_norm_convention")) {
        log.err("[qwen4] qwen4_norm_convention cannot be set through server-wide --config-overrides; remove that override and use a checkpoint-local config.json marker.\n", .{});
        return error.GlobalQwen4NormOverride;
    }
    try mergeObjects(a, &dst.object, src.object);
    var out: std.Io.Writer.Allocating = .init(a);
    var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try dst.jsonStringify(&jws);
    return allocator.dupe(u8, out.written());
}

fn mergeObjects(a: std.mem.Allocator, dst: *std.json.ObjectMap, src: std.json.ObjectMap) !void {
    var it = src.iterator();
    while (it.next()) |e| {
        if (dst.getPtr(e.key_ptr.*)) |p| {
            if (p.* == .object and e.value_ptr.* == .object) {
                // The handle is copied, so a rehash inside the recursion would
                // be lost — merge through the copy and store it back. `p` stays
                // valid: `dst` itself is not written during the recursion.
                var child = p.object;
                try mergeObjects(a, &child, e.value_ptr.object);
                p.* = .{ .object = child };
                continue;
            }
        }
        // Keys and values are arena-owned by the override document, which
        // outlives this merge.
        try dst.put(a, e.key_ptr.*, e.value_ptr.*);
    }
}
