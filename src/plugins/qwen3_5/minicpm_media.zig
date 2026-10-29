//! MiniCPM image run; computeMiniCpmMrope; appendMiniCpmPieces

const std = @import("std");
const chat_mod = @import("../../host/chat/chat.zig");
const model_mod = @import("../../sdk/config/model.zig");
const minicpm_vision = @import("minicpm_vision.zig");
const minicpm_canvas = @import("minicpm_canvas.zig");
const log = @import("../../sdk/util/log.zig");
const testing = std.testing;

const MediaItem = @import("../../host/media/placeholders.zig").MediaItem;
const collectMediaItems = @import("../../host/media/placeholders.zig").collectMediaItems;
const expandMediaPlaceholders = @import("../../host/media/placeholders.zig").expandMediaPlaceholders;
const MropeData = @import("../../host/media/encode.zig").MropeData;
const decodeRgbOwned = @import("../../host/media/decode.zig").decodeRgbOwned;

/// Soft tokens one MiniCPM-V piece contributes: its patch grid after both 2x2 merges (`16x`) or the last (`4x`).
fn miniCpmPieceTokens(img: chat_mod.ImageData, config: *const model_mod.ModelConfig) usize {
    const merge: usize = if (config.mcv_downsample_4x) 2 else 4;
    return (img.grid_h / merge) * (img.grid_w / merge);
}

/// `MiniCPMV4_7Processor.replace_image_token`: `<image_id>N</image_id>` (N = the image's index in
/// the prompt), `<image>` + the thumbnail's pads + `</image>`, then the crops row by row, each
/// `<slice>` + pads + `</slice>`, rows separated by a newline token.
pub fn appendMiniCpmImageRun(allocator: std.mem.Allocator, ids: *std.ArrayList(u32), pieces: []const chat_mod.ImageData, ordinal: usize, config: *const model_mod.ModelConfig) !void {
    if (pieces.len == 0) return;
    if (config.mcv_image_id_start != 0 and config.mcv_image_id_end != 0) {
        var digits: [20]u8 = undefined;
        const text = std.fmt.bufPrint(&digits, "{d}", .{ordinal}) catch unreachable;
        try ids.append(allocator, config.mcv_image_id_start);
        for (text) |d| try ids.append(allocator, config.mcv_digit_ids[d - '0']);
        try ids.append(allocator, config.mcv_image_id_end);
    }
    try ids.append(allocator, config.mcv_image_start_id);
    try ids.appendNTimes(allocator, config.image_token_id, miniCpmPieceTokens(pieces[0], config));
    try ids.append(allocator, config.mcv_image_end_id);
    if (pieces.len == 1) return;
    const rows: usize = pieces[1].tile_rows;
    const cols: usize = pieces[1].tile_cols;
    if (rows * cols + 1 != pieces.len) return error.InvalidImagePieces;
    for (0..rows) |row| {
        if (row > 0) try ids.append(allocator, config.mcv_newline_id);
        for (0..cols) |col| {
            try ids.append(allocator, config.mcv_slice_start_id);
            try ids.appendNTimes(allocator, config.image_token_id, miniCpmPieceTokens(pieces[1 + row * cols + col], config));
            try ids.append(allocator, config.mcv_slice_end_id);
        }
    }
}

/// Canvas M-RoPE table (`mrope_mode: "canvas"`) over the expanded prompt; caller owns `pos`.
pub fn computeMiniCpmMrope(allocator: std.mem.Allocator, prompt_ids: []const u32, items: []const MediaItem, config: *const model_mod.ModelConfig) !MropeData {
    var grids = std.ArrayList(minicpm_canvas.Grid).empty;
    defer grids.deinit(allocator);
    for (items) |item| switch (item) {
        .image => |pieces| for (pieces) |img| try grids.append(allocator, .{ img.grid_h, img.grid_w }),
        else => return error.UnsupportedMedia,
    };
    if (grids.items.len == 0) return .{};
    const r = try minicpm_canvas.canvasPositions(allocator, prompt_ids, .{
        .markers = .{
            .slice_start = config.mcv_slice_start_id,
            .slice_end = config.mcv_slice_end_id,
            .image_start = config.mcv_image_start_id,
            .image_end = config.mcv_image_end_id,
            .newline = config.mcv_newline_id,
        },
        .image_token = config.image_token_id,
        .video_token = config.video_token_id,
        .merge = if (config.mcv_downsample_4x) 2 else 4,
    }, grids.items, &.{});
    log.info("  Canvas M-RoPE: {d} crops, position ids over {d} tokens, decode delta {d}\n", .{ grids.items.len, prompt_ids.len, r.delta });
    return .{ .pos = r.pos, .total = prompt_ids.len, .delta = r.delta };
}

/// MiniCPM-V's processor: a thumbnail plus, for a large source, a grid of detail crops, each its own entry.
pub fn appendMiniCpmPieces(allocator: std.mem.Allocator, list: *std.ArrayList(chat_mod.ImageData), url: []const u8, vp: chat_mod.VisionPreproc) void {
    const sep = std.mem.indexOf(u8, url, ";base64,") orelse return;
    const b64 = url[sep + 8 ..];
    const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(b64) catch return;
    const raw = allocator.alloc(u8, decoded_size) catch return;
    defer allocator.free(raw);
    std.base64.standard.Decoder.decode(raw, b64) catch return;
    const src = decodeRgbOwned(allocator, raw, true) orelse return;
    defer src.deinit(allocator);

    const sliced = minicpm_vision.preprocess(allocator, src.rgb, src.h, src.w, .{ .patch = vp.patch, .max_slice_nums = vp.max_slices }) catch return;
    defer allocator.free(sliced.pieces);
    minicpm_vision.appendImageData(allocator, list, sliced) catch {
        for (sliced.pieces) |piece| allocator.free(piece.bytes);
        return;
    };
    log.info("  Decoded {d}x{d} image → minicpm {d} piece(s), slice grid {d}x{d}\n", .{ src.w, src.h, sliced.pieces.len, sliced.grid.rows, sliced.grid.cols });
}

// ── tests ──

// `src/fixtures/minicpmv4_7_canvas_rope.json` holds MiniCPMV4_7Processor's prompts (placeholders
// before and after expansion, tokenized) and `get_rope_index` of the expanded ones.
test "MiniCPM-V 4.7: expanded image runs and canvas positions equal the processor and reference" {
    const t = testing;
    const a = t.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, @embedFile("../../fixtures/minicpmv4_7_canvas_rope.json"), .{});
    defer parsed.deinit();
    const doc = parsed.value.object;
    const id = struct {
        fn of(v: std.json.Value) u32 {
            return @intCast(v.integer);
        }
    }.of;
    // The marker ids come from the checkpoint's own config; only the tokenizer-only ids are set here.
    var config = try model_mod.parseConfigFromJson(a, @embedFile("../../fixtures/model-configs/minicpmv4_7.json"));
    defer config.deinit(a);
    try t.expectEqual(id(doc.get("image_token").?), config.image_token_id);
    try t.expectEqual(id(doc.get("newline_id").?), config.mcv_newline_id);
    config.mcv_image_id_start = id(doc.get("image_id_start").?);
    config.mcv_image_id_end = id(doc.get("image_id_end").?);
    for (doc.get("digits").?.array.items, &config.mcv_digit_ids) |v, *d| d.* = id(v);

    for (doc.get("cases").?.array.items) |case| {
        const c = case.object;
        config.mcv_downsample_4x = id(c.get("merge").?) == 2;
        var pieces: std.ArrayList(chat_mod.ImageData) = .empty;
        defer pieces.deinit(a);
        for (c.get("images").?.array.items) |image| {
            const im = image.object;
            const rows: u16 = @intCast(id(im.get("rows").?));
            const cols: u16 = @intCast(id(im.get("cols").?));
            for (im.get("pieces").?.array.items, 0..) |grid, k| try pieces.append(a, .{
                .pixels = "",
                .width = 0,
                .height = 0,
                .grid_h = id(grid.array.items[0]),
                .grid_w = id(grid.array.items[1]),
                .tile_rows = rows,
                .tile_cols = cols,
                .tile_index = @intCast(k),
            });
        }
        const msgs = [_]chat_mod.Message{.{ .role = "user", .content = "hi", .images = pieces.items }};
        const items = try collectMediaItems(a, &msgs);
        defer a.free(items);
        try t.expectEqual(c.get("images").?.array.items.len, items.len);

        const pre = try a.alloc(u32, c.get("pre_ids").?.array.items.len);
        defer a.free(pre);
        for (c.get("pre_ids").?.array.items, pre) |v, *p| p.* = id(v);
        const zeros = try a.alloc(usize, items.len);
        defer a.free(zeros);
        @memset(zeros, 0);
        const keys = try a.alloc(u64, items.len);
        defer a.free(keys);
        @memset(keys, 1);
        const out = try expandMediaPlaceholders(a, pre, items, zeros, keys, &config);
        defer a.free(out.ids);
        defer a.free(out.media);
        const want = try a.alloc(u32, c.get("ids").?.array.items.len);
        defer a.free(want);
        for (c.get("ids").?.array.items, want) |v, *w| w.* = id(v);
        t.expectEqualSlices(u32, want, out.ids) catch |e| {
            std.debug.print("{s}: expanded prompt differs\n", .{c.get("name").?.string});
            return e;
        };

        const mrope = try computeMiniCpmMrope(a, out.ids, items, &config);
        defer a.free(mrope.pos.?);
        try t.expectEqual(@as(i32, @intCast(c.get("delta").?.integer)), mrope.delta);
        const pos = c.get("pos").?.array.items;
        for (0..3) |row| for (pos[row].array.items, 0..) |w, k| {
            try t.expectEqual(@as(i32, @intCast(w.integer)), mrope.pos.?[row * out.ids.len + k]);
        };
    }
}
