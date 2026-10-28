//! Images for a served outside plugin: its `MediaHooks.prepare` decodes and plans the image, and the host carries the
//! result as the patch-grid `ImageData` the other towers use (float32 `[grid_h * grid_w, feat]` pixels).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const chat_mod = @import("../chat/chat.zig");
const Binding = @import("../../engines/mlx/plugin/config.zig").Binding;
const log = @import("../../sdk/util/log.zig");

/// The image as the plugin prepared it, copied off its buffer. The plugin hands host bytes: this runs on a connection
/// thread, where nothing calls MLX.
pub fn prepareImage(allocator: std.mem.Allocator, b: *const Binding, encoded: []const u8) ?chat_mod.ImageData {
    const hooks = b.table.media orelse return null;
    var item: abi.MediaItem = .{};
    const v = hooks.prepare(&b.parsed, .image, abi.Str.of(encoded), 0, &item);
    if (!v.isOk()) {
        log.warn("[plugin] {s}: {s}\n", .{ b.table.name.slice(), v.why.slice() });
        return null;
    }
    defer if (item.free) |release| release(&item);
    const rows = @as(usize, item.grid[1]) * item.grid[2];
    const bytes = item.bytes.slice();
    if (item.pixels.ctx != null or rows == 0 or bytes.len == 0 or bytes.len % (rows * 4) != 0) {
        log.warn("[plugin] {s}: prepare must return float32 pixels as host bytes, a whole row per grid cell\n", .{b.table.name.slice()});
        return null;
    }
    const owned = allocator.dupe(u8, bytes) catch return null;
    return .{ .pixels = owned, .width = 0, .height = 0, .grid_h = item.grid[1], .grid_w = item.grid[2] };
}

// ── tests ──

const px = [_]f32{ 1, 2, 3, 4, 5, 6 };

const Fake = struct {
    var freed: usize = 0;

    fn release(_: *abi.MediaItem) callconv(.c) void {
        freed += 1;
    }
    fn prepare(_: *const abi.Parsed, _: abi.MediaKind, bytes: abi.Str, _: u64, out: *abi.MediaItem) callconv(.c) abi.Verdict {
        const mode = if (bytes.len > 0) bytes.slice()[0] else 0;
        switch (mode) {
            'g' => out.* = .{ .grid = .{ 1, 2, 1 }, .bytes = .of(std.mem.sliceAsBytes(&px)), .free = &release },
            'a' => out.* = .{ .grid = .{ 1, 2, 1 }, .pixels = .{ .ctx = mlx.mlx_array_new_data(&px, &[_]c_int{ 2, 3 }, 2, .float32).ctx } },
            's' => out.* = .{ .grid = .{ 1, 2, 1 }, .bytes = .of(std.mem.sliceAsBytes(px[0..5])), .free = &release },
            else => return abi.Verdict.refuse(.failed, "no image"),
        }
        return .ok;
    }
    fn tower(_: *anyopaque, _: *const abi.MediaItem, _: *abi.Array) callconv(.c) abi.Verdict {
        return .ok;
    }
};

test "plugin images: host pixels and the grid ride an ImageData, the buffer is released; bad returns are no image" {
    const hooks = abi.MediaHooks{ .tower = &Fake.tower, .prepare = &Fake.prepare };
    const table = abi.ModelTable{ .name = .of("t"), .claims = undefined, .parse = undefined, .free_parsed = undefined, .bill = undefined, .load = undefined, .unload = undefined, .forward = undefined, .cache = undefined, .weights = undefined, .media = &hooks };
    const b: Binding = .{ .table = &table, .outside = true };
    const a = std.testing.allocator;
    const img = prepareImage(a, &b, "g") orelse return error.TestUnexpectedResult;
    defer a.free(img.pixels);
    try std.testing.expectEqual(@as(u32, 2), img.grid_h);
    try std.testing.expectEqual(@as(u32, 1), img.grid_w);
    try std.testing.expectEqualSlices(f32, &px, @as([*]const f32, @ptrCast(@alignCast(img.pixels.ptr)))[0..6]);
    try std.testing.expectEqual(@as(usize, 1), Fake.freed);
    // An array, a buffer that is not a whole row per cell, and a refusal.
    try std.testing.expect(prepareImage(a, &b, "s") == null);
    try std.testing.expectEqual(@as(usize, 2), Fake.freed);
    try std.testing.expect(prepareImage(a, &b, "x") == null);
    const arr = prepareImage(a, &b, "a");
    try std.testing.expect(arr == null);
}
