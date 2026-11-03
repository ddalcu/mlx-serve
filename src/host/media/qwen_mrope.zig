//! computeQwenMrope (Qwen3-VL positions)

const std = @import("std");
const model_mod = @import("../../sdk/config/model.zig");
const mrope_mod = @import("../../sdk/vision/mrope.zig");
const log = @import("../../sdk/util/log.zig");

const MediaItem = @import("placeholders.zig").MediaItem;
const MropeData = @import("encode.zig").MropeData;

/// The interleaved-M-RoPE table over the FINAL (expanded) prompt; empty when
/// the model isn't Qwen-vision or carries no image/video. Caller owns `pos`.
pub fn computeQwenMrope(allocator: std.mem.Allocator, prompt_ids: []const u32, items: []const MediaItem, config: *const model_mod.ModelConfig) !MropeData {
    if (!config.qwen_vision) return .{};
    // Each modality's grids in prompt order; getRopeIndex interleaves the two
    // lists by whichever marker occurs first in the tokens.
    var image_grids = std.ArrayList(mrope_mod.ImageGrid).empty;
    defer image_grids.deinit(allocator);
    var video_grids = std.ArrayList(mrope_mod.ImageGrid).empty;
    defer video_grids.deinit(allocator);
    for (items) |item| switch (item) {
        .image => |pieces| for (pieces) |im| {
            if (im.grid_h > 0) try image_grids.append(allocator, .{ .t = 1, .h = im.grid_h, .w = im.grid_w });
        },
        .video => |vd| try video_grids.append(allocator, .{ .t = vd.grid_t, .h = vd.grid_h, .w = vd.grid_w }),
        .audio => {},
    };
    if (image_grids.items.len == 0 and video_grids.items.len == 0) return .{};

    var ri = mrope_mod.getRopeIndex(allocator, prompt_ids, image_grids.items, video_grids.items, config.image_token_id, config.video_token_id, config.vision_start_token_id, config.qv_merge) catch |err| {
        log.warn("M-RoPE get_rope_index failed ({s}); falling back to scalar RoPE\n", .{@errorName(err)});
        return .{};
    };
    defer ri.deinit();
    const total = prompt_ids.len;
    const flat = try allocator.alloc(i32, 3 * total);
    @memcpy(flat[0..total], ri.pos[0]);
    @memcpy(flat[total .. 2 * total], ri.pos[1]);
    @memcpy(flat[2 * total .. 3 * total], ri.pos[2]);
    log.info("  M-RoPE: {d} images, {d} videos, position ids over {d} tokens, decode delta {d}\n", .{ image_grids.items.len, video_grids.items.len, total, ri.delta });
    return .{ .pos = flat, .total = total, .delta = ri.delta };
}
