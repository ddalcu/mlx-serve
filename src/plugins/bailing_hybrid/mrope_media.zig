//! computeBailingMrope: the M-RoPE table for a Ling VL request, dispatched by the host
//! (`host/media/encode.zig`) the way MiniCPM's canvas table is.

const std = @import("std");
const model_mod = @import("../../sdk/config/model.zig");
const mrope_mod = @import("../../sdk/vision/mrope.zig");
const bailing_mrope = @import("mrope.zig");
const log = @import("../../sdk/util/log.zig");

const MediaItem = @import("../../host/media/placeholders.zig").MediaItem;
const MropeData = @import("../../host/media/encode.zig").MropeData;

/// The body-diagonal 3-D table over the FINAL (expanded) prompt. Video is not served: a video
/// means no table at all, and the slot then decodes on plain scalar positions.
pub fn computeBailingMrope(allocator: std.mem.Allocator, prompt_ids: []const u32, items: []const MediaItem, config: *const model_mod.ModelConfig) !MropeData {
    var image_grids = std.ArrayList(mrope_mod.ImageGrid).empty;
    defer image_grids.deinit(allocator);
    for (items) |item| switch (item) {
        .image => |pieces| for (pieces) |im| {
            if (im.grid_h > 0) try image_grids.append(allocator, .{ .t = 1, .h = im.grid_h, .w = im.grid_w });
        },
        .video => return .{},
        .audio => {},
    };
    if (image_grids.items.len == 0) return .{};
    var ri = bailing_mrope.getRopeIndexBailing(allocator, prompt_ids, image_grids.items, config.image_token_id, config.qv_merge) catch |err| {
        log.warn("M-RoPE bailing get_rope_index failed ({s}); falling back to scalar RoPE\n", .{@errorName(err)});
        return .{};
    };
    defer ri.deinit();
    const total = prompt_ids.len;
    const flat = try allocator.alloc(i32, 3 * total);
    @memcpy(flat[0..total], ri.pos[0]);
    @memcpy(flat[total .. 2 * total], ri.pos[1]);
    @memcpy(flat[2 * total .. 3 * total], ri.pos[2]);
    log.info("  M-RoPE (bailing): {d} images, position ids over {d} tokens, decode delta {d}\n", .{ image_grids.items.len, total, ri.delta });
    return .{ .pos = flat, .total = total, .delta = ri.delta };
}
