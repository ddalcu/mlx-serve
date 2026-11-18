//! DeepSeek-V4.1's image hooks: plan a decoded image as the release's processor does (`vision.preprocess`),
//! encode it with the tower (the whole span: start, rows with their newlines, end) and lay out its placeholder run,
//! every span position carrying `image_token_id`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const vision = @import("vision.zig");
const config = @import("config.zig");

const weights_load = @import("../../sdk/weights/load.zig");
const log = @import("../../sdk/util/log.zig");
const dsv41 = @import("deepseek_v41.zig");
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
    // An outside plugin's prepare runs on a connection thread, which never calls MLX: the pixels go back as host bytes
    // (float32, a row per grid cell), which the host copies, then frees through `freeBytes`.
    out.* = .{
        .kind = .image,
        .rows = @intCast(spanRows(c, px.grid_h, px.grid_w)),
        .grid = .{ 1, px.grid_h, px.grid_w },
        .bytes = .of(px.bytes),
        .free = &freeBytes,
    };
    return .ok;
}

fn freeBytes(item: *abi.MediaItem) callconv(.c) void {
    gpa.free(@constCast(item.bytes.slice()));
    item.bytes = .{};
}
pub fn tower(mp: *anyopaque, item: *const abi.MediaItem, out: *abi.Array) callconv(.c) Verdict {
    const m: *@import("plugin.zig").Model = @ptrCast(@alignCast(mp));
    if (m.tower == null) if (m.vision_dir) |dir| return lazyTower(m, dir, item, out);
    const t = &(m.tower orelse return Verdict.refuse(.unsupported, "deepseek_v41: this pack has no vision tower"));
    const rows = t.forward(.{ .ctx = item.pixels.ctx }, item.grid[1], item.grid[2]) catch return Verdict.refuse(.failed, "deepseek_v41: the vision tower failed");
    out.* = .{ .ctx = rows.ctx };
    return .ok;
}

/// the tower read from the pack for this image and freed before the hook
/// returns. A decode-phase bank holds its prompt rows meanwhile (the prompt's working set, unused until a prompt's wide
/// calls, holds the tower) and grows its rows again after, so a decode call between two hooks finds the bank it left.
fn lazyTower(m: *@import("plugin.zig").Model, dir: [:0]const u8, item: *const abi.MediaItem, out: *abi.Array) Verdict {
    const decoding = @import("plugin.zig").bankDecoding(m);
    if (decoding) m.mdl.streamed.?.phase(true) catch |e| return lazyFailed(e);
    const rows = lazyRows(m, dir, item) catch |e| {
        if (decoding) m.mdl.streamed.?.phase(false) catch {};
        return lazyFailed(e);
    };
    // the tower's buffers back to the system before the bank grows again
    _ = mlx.mlx_clear_cache();
    if (decoding) m.mdl.streamed.?.phase(false) catch |e| {
        _ = mlx.mlx_array_free(rows);
        return lazyFailed(e);
    };
    out.* = .{ .ctx = rows.ctx };
    return .ok;
}
/// The image's span on a tower made from the pack's vision weights alone, computed before the tower goes.
fn lazyRows(m: *@import("plugin.zig").Model, dir: []const u8, item: *const abi.MediaItem) !mlx.mlx_array {
    const io = std.Io.Threaded.global_single_threaded.io();
    var w = try weights_load.loadWeightsOpt(io, gpa, dir, .{ .vision = true, .dsv41 = true, .keep_f16 = true, .skip = &notVision });
    defer w.deinit();
    var t = try vision.Dsv41Vision.init(gpa, m.cfg, &w);
    defer t.deinit();
    const rows = try t.forward(.{ .ctx = item.pixels.ctx }, item.grid[1], item.grid[2]);
    errdefer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_array_eval(rows));
    return rows;
}
fn notVision(key: []const u8) bool {
    const k = if (std.mem.startsWith(u8, key, weights_load.dsv41_text_prefix)) key[weights_load.dsv41_text_prefix.len..] else key;
    return !weights_load.dsv41VisionKey(k) or dsv41.visionUnread(key);
}
fn lazyFailed(e: anyerror) Verdict {
    log.err("deepseek_v41: the vision tower read for this image failed ({s})\n", .{@errorName(e)});
    return Verdict.refuse(.failed, "deepseek_v41: the vision tower failed");
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
