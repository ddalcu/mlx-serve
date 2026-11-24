//! The interleaved M-RoPE table for the newline-row vision layout (`HYV4FlashModel.get_rope_index` +
//! `get_vision_position_ids`).
//!
//! An image run is `(h/merge) * (w/merge + 1)` rows: every merged row ends with a `image_newline` row
//! (the tower appends it). Each frame starts a new block of `max(h/merge, w/merge+1)` positions, all three
//! axes start at the same offset, and text continues at `max(ids)+1`. Pure arithmetic, no MLX.

const std = @import("std");
const model_mod = @import("../../sdk/config/model.zig");
const qvis = @import("../../sdk/vision/qwen_vision.zig");
const log = @import("../../sdk/util/log.zig");
const MropeData = @import("../../host/media/encode.zig").MropeData;
const MediaItem = @import("../../host/media/placeholders.zig").MediaItem;

pub fn computeNewlineMrope(
    allocator: std.mem.Allocator,
    prompt_ids: []const u32,
    items: []const MediaItem,
    config: *const model_mod.ModelConfig,
) !MropeData {
    const merge = config.qv_merge;
    if (merge == 0) return .{};
    const patch = config.qv_patch;
    // Grids in prompt order: `[t, h, w]` patch counts (Hy4 images are one piece each). The plugin's
    // `prepare` runs on the encode; a piece without a grid is sized here with the same processor bounds.
    var grids = std.ArrayList([3]u32).empty;
    defer grids.deinit(allocator);
    for (items) |item| switch (item) {
        .image => |pieces| for (pieces) |im| {
            if (im.grid_h > 0 and im.grid_w > 0) {
                try grids.append(allocator, .{ 1, im.grid_h, im.grid_w });
            } else if (im.width > 0 and im.height > 0) {
                const size = qvis.smartResizeImage(im.height, im.width, patch * merge, 4096, @min(@as(u32, 4194304), qvis.ENGINE_MAX_PIXELS));
                try grids.append(allocator, .{ 1, size.h / patch, size.w / patch });
            }
        },
        .video => return error.VideoUnsupported,
        .audio => {},
    };
    if (grids.items.len == 0) return .{};
    const n = prompt_ids.len;
    const pos = try allocator.alloc(i32, 3 * n);
    errdefer allocator.free(pos);
    @memset(pos, 0);

    var cur: i32 = 0;
    var gi: usize = 0;
    var i: usize = 0;
    while (i < n) {
        if (prompt_ids[i] == config.image_token_id) {
            const start = i;
            while (i < n and prompt_ids[i] == config.image_token_id) : (i += 1) {}
            const run = i - start;
            if (gi >= grids.items.len) return error.MediaPlaceholderMismatch;
            const g = grids.items[gi];
            gi += 1;
            const mh = g[1] / merge;
            const mw = g[2] / merge + 1; // the newline column
            const block: i32 = @intCast(@max(mh, mw));
            const want: usize = @as(usize, g[0]) * mh * mw;
            if (run != want) return error.MediaPlaceholderMismatch;
            var k: usize = 0;
            var t: u32 = 0;
            while (t < g[0]) : (t += 1) {
                const off = cur + @as(i32, @intCast(t)) * block;
                var h: u32 = 0;
                while (h < mh) : (h += 1) {
                    var w: u32 = 0;
                    while (w < mw) : (w += 1) {
                        pos[start + k] = off;
                        pos[n + start + k] = off + @as(i32, @intCast(h));
                        pos[2 * n + start + k] = off + @as(i32, @intCast(w));
                        k += 1;
                    }
                }
            }
            cur += @as(i32, @intCast(g[0])) * block;
        } else {
            pos[i] = cur;
            pos[n + i] = cur;
            pos[2 * n + i] = cur;
            cur += 1;
            i += 1;
        }
    }
    if (gi != grids.items.len) return error.MediaPlaceholderMismatch;

    var max: i32 = 0;
    for (pos) |p| max = @max(max, p);
    const delta: i32 = max + 1 - @as(i32, @intCast(n));
    log.info("  Hy4 M-RoPE: {d} image(s), position ids over {d} tokens, decode delta {d}\n", .{ grids.items.len, n, delta });
    return .{ .pos = pos, .total = n, .delta = delta };
}

// ── tests ──

const testing = std.testing;
const chat = @import("../../host/chat/types.zig");

test "hy4 mrope: one image row block carries a newline column and text resumes at max+1" {
    const cfg = model_mod.ModelConfig{ .image_token_id = 120120, .qv_merge = 2 };
    const pieces = [_]chat.ImageData{.{ .pixels = "", .width = 512, .height = 512, .grid_h = 32, .grid_w = 32 }};
    const items = [_]MediaItem{.{ .image = &pieces }};
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    try ids.appendSlice(testing.allocator, &.{ 1, 2, 3 });
    try ids.appendNTimes(testing.allocator, 120120, 272);
    try ids.appendSlice(testing.allocator, &.{ 4, 5 });
    const m = try computeNewlineMrope(testing.allocator, ids.items, &items, &cfg);
    defer testing.allocator.free(m.pos.?);
    const p = m.pos.?;
    try testing.expectEqual(@as(usize, ids.items.len), m.total);
    // Text before the image.
    for (0..3) |i| try testing.expectEqual(@as(i32, @intCast(i)), p[i]);
    // Image: 16 rows of 17 columns (the newline), every axis starting at 3, time fixed.
    for (0..272) |k| {
        const h = k / 17;
        const w = k % 17;
        try testing.expectEqual(@as(i32, 3), p[3 + k]);
        try testing.expectEqual(@as(i32, 3 + @as(i32, @intCast(h))), p[ids.items.len + 3 + k]);
        try testing.expectEqual(@as(i32, 3 + @as(i32, @intCast(w))), p[2 * ids.items.len + 3 + k]);
    }
    // Text resumes after max(ids)+1 = 20.
    try testing.expectEqual(@as(i32, 20), p[275]);
    try testing.expectEqual(@as(i32, 21), p[276]);
    try testing.expectEqual(@as(i32, 20), p[ids.items.len + 275]);
    try testing.expectEqual(@as(i32, 20), p[2 * ids.items.len + 275]);
    try testing.expectEqual(@as(i32, 22 - @as(i32, @intCast(ids.items.len))), m.delta);
}

test "hy4 mrope: a placeholder run of the wrong width is refused" {
    const cfg = model_mod.ModelConfig{ .image_token_id = 120120, .qv_merge = 2 };
    const pieces = [_]chat.ImageData{.{ .pixels = "", .width = 512, .height = 512, .grid_h = 32, .grid_w = 32 }};
    const items = [_]MediaItem{.{ .image = &pieces }};
    try testing.expectError(error.MediaPlaceholderMismatch, computeNewlineMrope(testing.allocator, &.{ 120120, 120120 }, &items, &cfg));
}
