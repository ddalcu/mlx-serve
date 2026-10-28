//! Images for a served outside plugin: its `MediaHooks.prepare` decodes and plans the image, and the host carries the
//! result as the patch-grid `ImageData` the other towers use (float32 `[grid_h * grid_w, feat]` pixels).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const chat_mod = @import("../chat/chat.zig");
const Binding = @import("../../engines/mlx/plugin/config.zig").Binding;
const log = @import("../../sdk/util/log.zig");

pub fn prepareImage(allocator: std.mem.Allocator, b: *const Binding, encoded: []const u8) ?chat_mod.ImageData {
    const hooks = b.table.media orelse return null;
    var item: abi.MediaItem = .{};
    const v = hooks.prepare(&b.parsed, .image, abi.Str.of(encoded), 0, &item);
    if (!v.isOk()) {
        log.warn("[plugin] {s}: {s}\n", .{ b.table.name.slice(), v.why.slice() });
        return null;
    }
    const px: mlx.mlx_array = .{ .ctx = item.pixels.ctx };
    defer _ = mlx.mlx_array_free(px);
    if (item.grid[1] == 0 or item.grid[2] == 0) return null;
    mlx.check(mlx.mlx_array_eval(px)) catch return null;
    const data = mlx.mlx_array_data_float32(px) orelse return null;
    const bytes = std.mem.sliceAsBytes(data[0..mlx.mlx_array_size(px)]);
    const owned = allocator.dupe(u8, bytes) catch return null;
    return .{ .pixels = owned, .width = 0, .height = 0, .grid_h = item.grid[1], .grid_w = item.grid[2] };
}

// ── tests ──

test "plugin images: the hook's pixels and grid ride an ImageData; a refusal is no image" {
    const T = struct {
        fn prepare(_: *const abi.Parsed, kind: abi.MediaKind, bytes: abi.Str, _: u64, out: *abi.MediaItem) callconv(.c) abi.Verdict {
            if (kind != .image or bytes.len == 0) return abi.Verdict.refuse(.failed, "no image");
            const px = [_]f32{ 1, 2, 3, 4, 5, 6 };
            out.* = .{ .rows = 4, .grid = .{ 1, 2, 1 }, .pixels = .{ .ctx = mlx.mlx_array_new_data(&px, &[_]c_int{ 2, 3 }, 2, .float32).ctx } };
            return .ok;
        }
        fn tower(_: *anyopaque, _: *const abi.MediaItem, _: *abi.Array) callconv(.c) abi.Verdict {
            return .ok;
        }
    };
    const hooks = abi.MediaHooks{ .tower = &T.tower, .prepare = &T.prepare };
    const table = abi.ModelTable{ .name = .of("t"), .claims = undefined, .parse = undefined, .free_parsed = undefined, .bill = undefined, .load = undefined, .unload = undefined, .forward = undefined, .cache = undefined, .weights = undefined, .media = &hooks };
    const b: Binding = .{ .table = &table, .outside = true };
    const a = std.testing.allocator;
    const img = prepareImage(a, &b, "x") orelse return error.TestUnexpectedResult;
    defer a.free(img.pixels);
    try std.testing.expectEqual(@as(u32, 2), img.grid_h);
    try std.testing.expectEqual(@as(u32, 1), img.grid_w);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, @as([*]const f32, @ptrCast(@alignCast(img.pixels.ptr)))[0..6]);
    try std.testing.expect(prepareImage(a, &b, "") == null);
}
