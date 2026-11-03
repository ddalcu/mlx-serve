//! DeepSeek-V4.1's image hooks: plan a decoded image as the release's processor does (`vision.preprocess`),
//! encode it with the tower (the whole span: start, rows with their newlines, end) and lay out its placeholder run,
//! every span position carrying `image_token_id`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const vision = @import("vision.zig");
const config = @import("config.zig");

const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

/// The span an image of ViT grid `gh x gw` takes in the prompt.
pub fn spanRows(c: *const config.Config, gh: u32, gw: u32) usize {
    const ds = c.dv_downsample;
    return vision.spanTokens((gh + ds - 1) / ds, (gw + ds - 1) / ds);
}

pub fn prepare(p: *const abi.Parsed, in: *const abi.MediaInput, budget: u64, out: *abi.MediaItem) callconv(.c) Verdict {
    _ = budget; // the release's processor bounds the image by its own token budget
    if (in.kind != .image or in.frames != 1) return Verdict.refuse(.unsupported, "deepseek_v41: images only");
    const c = config.of(p);
    const px = vision.preprocess(gpa, in.rgb.slice(), in.height, in.width, c.vision_patch_size, c.dv_downsample, c.dv_max_image_tokens, c.dv_min_pixels, c.dv_max_wh_ratio) catch
        return Verdict.refuse(.failed, "deepseek_v41: the image's dimensions have no grid");
    defer gpa.free(px.bytes);
    const feat: c_int = @intCast(3 * c.vision_patch_size * c.vision_patch_size);
    const n: c_int = @intCast(px.grid_h * px.grid_w);
    out.* = .{
        .kind = .image,
        .rows = @intCast(spanRows(c, px.grid_h, px.grid_w)),
        .grid = .{ 1, px.grid_h, px.grid_w },
        .pixels = .{ .ctx = mlx.mlx_array_new_data(px.bytes.ptr, &[_]c_int{ n, feat }, 2, .float32).ctx },
    };
    return .ok;
}

pub fn tower(mp: *anyopaque, item: *const abi.MediaItem, out: *abi.Array) callconv(.c) Verdict {
    const m: *@import("plugin.zig").Model = @ptrCast(@alignCast(mp));
    const t = &(m.tower orelse return Verdict.refuse(.unsupported, "deepseek_v41: this pack has no vision tower"));
    const rows = t.forward(.{ .ctx = item.pixels.ctx }, item.grid[1], item.grid[2]) catch return Verdict.refuse(.failed, "deepseek_v41: the vision tower failed");
    out.* = .{ .ctx = rows.ctx };
    return .ok;
}

pub fn placeholderRows(p: *const abi.Parsed, item: *const abi.MediaItem, _: *const abi.TextTokens, out: [*]u32, cap: u32) callconv(.c) u32 {
    const n = @min(item.rows, cap);
    @memset(out[0..n], config.of(p).image_token_id);
    return n;
}

// ── tests ──

test "deepseek_v41 media: an image's run is its span, every position the image token" {
    var c: config.Config = .{ .image_token_id = 7, .dv_downsample = 3 };
    var p: abi.Parsed = .{ .private = &c };
    // A 5 x 7 ViT grid merges to 2 x 3: two rows of three plus a newline each, a start and an end.
    var item: abi.MediaItem = .{ .rows = @intCast(spanRows(&c, 5, 7)), .grid = .{ 1, 5, 7 } };
    try std.testing.expectEqual(@as(u32, 10), item.rows);
    var buf: [16]u32 = undefined;
    const no_tok: abi.TextTokens = .{ .ctx = null, .encode = undefined };
    try std.testing.expectEqual(@as(u32, 10), placeholderRows(&p, &item, &no_tok, &buf, buf.len));
    for (buf[0..10]) |id| try std.testing.expectEqual(@as(u32, 7), id);
}
