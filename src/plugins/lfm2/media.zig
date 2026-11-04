//! appendLfm2ImageRun; LFM2 tiling, canvas, region

const std = @import("std");
const chat_mod = @import("../../host/chat/chat.zig");
const model_mod = @import("../../sdk/config/model.zig");
const qwen_vision = @import("../../sdk/vision/qwen_vision.zig");
const lfm2_vision = @import("vision.zig");
const log = @import("../../sdk/util/log.zig");

const resampleFilterFor = @import("../../host/media/decode.zig").resampleFilterFor;
const DecodedRgb = @import("../../host/media/decode.zig").DecodedRgb;
const decodeRgbOwned = @import("../../host/media/decode.zig").decodeRgbOwned;

/// LFM2-VL's image block is NOT a flat run of pads: it opens with
/// `<|image_start|>` and closes with `<|image_end|>`, and a TILED source labels
/// every tile with `<|img_row_R_col_C|>` and its thumbnail with
/// `<|img_thumbnail|>` before that piece's pads, in the order the encoder
/// concatenated the pieces.
pub fn appendLfm2ImageRun(allocator: std.mem.Allocator, seg: *std.ArrayList(u32), pieces: []const chat_mod.ImageData, config: *const model_mod.ModelConfig) !void {
    const merge = if (config.lv_downsample > 0) config.lv_downsample else 1;
    if (config.boi_token_id > 0) try seg.append(allocator, config.boi_token_id);
    for (pieces) |img| {
        if (img.tile_rows > 0) {
            const tiles: u16 = img.tile_rows * img.tile_cols;
            if (img.tile_index == tiles) {
                if (config.lv_thumbnail_token_id > 0) try seg.append(allocator, config.lv_thumbnail_token_id);
            } else if (config.lv_row_col_base_id > 0) {
                // The `<|img_row_R_col_C|>` block is contiguous and row-major
                // over the MAX tile grid, not this image's.
                const row = img.tile_index / img.tile_cols;
                const col = img.tile_index % img.tile_cols;
                const max_cols = if (config.lv_max_tiles > 0) config.lv_max_tiles else 10;
                try seg.append(allocator, config.lv_row_col_base_id + row * max_cols + col);
            }
        }
        try seg.appendNTimes(allocator, config.image_token_id, (img.grid_h / merge) * (img.grid_w / merge));
    }
    if (config.eoi_token_id > 0) try seg.append(allocator, config.eoi_token_id);
}

/// `Lfm2VlImageProcessor.resize_and_split`: a source inside the budget is one
/// resized image; past it, the WHOLE image is resized onto a `cols`x`rows`
/// canvas of `tile_size` tiles, cut into tiles, and a thumbnail of the whole
/// image is appended after them. The canvas is resampled ONCE and read tile by
/// tile — re-resizing per tile would resample each region on its own and is not
/// what `split_to_tiles` does.
pub fn appendLfm2Tiles(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(chat_mod.ImageData),
    url: []const u8,
    vp: chat_mod.VisionPreproc,
) void {
    const sep = std.mem.indexOf(u8, url, ";base64,") orelse return;
    const b64 = url[sep + 8 ..];
    const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(b64) catch return;
    const raw = allocator.alloc(u8, decoded_size) catch return;
    defer allocator.free(raw);
    std.base64.standard.Decoder.decode(raw, b64) catch return;

    const src = decodeRgbOwned(allocator, raw, true) orelse return;
    defer src.deinit(allocator);

    const split = vp.max_tiles > 1 and
        lfm2_vision.isImageTooLarge(src.h, src.w, vp.patch, vp.merge, vp.max_tokens, vp.pixels_tolerance);
    // Doubles as the whole-image geometry when the source stays one tile.
    const thumb = lfm2_vision.smartResize(src.h, src.w, vp.patch, vp.merge, vp.min_tokens, vp.max_tokens);
    const thumb_tokens = (thumb.h / vp.patch / vp.merge) * (thumb.w / vp.patch / vp.merge);

    if (!split) {
        const chw = lfm2Canvas(allocator, src, thumb.h, thumb.w, vp) orelse return;
        defer allocator.free(chw);
        const img = lfm2Region(allocator, chw, thumb.h, thumb.w, vp, 0, 0, thumb.h, thumb.w) orelse return;
        list.append(allocator, img) catch {
            allocator.free(img.pixels);
            return;
        };
        log.info("  Decoded {d}x{d} image → lfm2 grid {d}x{d} ({d} tokens, resized {d}x{d})\n", .{
            src.w, src.h, thumb.h / vp.patch, thumb.w / vp.patch, thumb_tokens, thumb.w, thumb.h,
        });
        return;
    }

    const grid = lfm2_vision.gridLayout(src.h, src.w, vp.min_tiles, vp.max_tiles, vp.tile_size);
    const canvas_h = vp.tile_size * grid.rows;
    const canvas_w = vp.tile_size * grid.cols;
    const per_tile = (vp.tile_size / vp.patch / vp.merge) * (vp.tile_size / vp.patch / vp.merge);
    const n_tiles: u16 = @intCast(grid.rows * grid.cols);
    const before = list.items.len;

    // A partial tile set would be spliced against a token layout that assumes
    // all of them, so any failure drops the whole image rather than some of it.
    var ok = true;
    {
        const chw = lfm2Canvas(allocator, src, canvas_h, canvas_w, vp) orelse return;
        defer allocator.free(chw);
        outer: for (0..grid.rows) |row| {
            for (0..grid.cols) |col| {
                var img = lfm2Region(
                    allocator,
                    chw,
                    canvas_h,
                    canvas_w,
                    vp,
                    @intCast(row * vp.tile_size),
                    @intCast(col * vp.tile_size),
                    vp.tile_size,
                    vp.tile_size,
                ) orelse {
                    ok = false;
                    break :outer;
                };
                img.tile_rows = @intCast(grid.rows);
                img.tile_cols = @intCast(grid.cols);
                img.tile_index = @intCast(row * grid.cols + col);
                list.append(allocator, img) catch {
                    allocator.free(img.pixels);
                    ok = false;
                    break :outer;
                };
            }
        }
    }
    if (!ok) {
        for (list.items[before..]) |prior| allocator.free(prior.pixels);
        list.shrinkRetainingCapacity(before);
        return;
    }

    var thumb_note: []const u8 = "";
    if (vp.use_thumbnail and n_tiles != 1) {
        if (lfm2Canvas(allocator, src, thumb.h, thumb.w, vp)) |tchw| {
            defer allocator.free(tchw);
            if (lfm2Region(allocator, tchw, thumb.h, thumb.w, vp, 0, 0, thumb.h, thumb.w)) |t| {
                var img = t;
                img.tile_rows = @intCast(grid.rows);
                img.tile_cols = @intCast(grid.cols);
                img.tile_index = n_tiles;
                list.append(allocator, img) catch allocator.free(img.pixels);
                thumb_note = " + thumbnail";
            }
        }
    }
    log.info("  Decoded {d}x{d} image → lfm2 {d}x{d} tiles{s} on a {d}x{d} canvas ({d} tokens)\n", .{
        src.w,      src.h,
        grid.rows,  grid.cols,
        thumb_note, canvas_w,
        canvas_h,   @as(usize, n_tiles) * per_tile + (if (thumb_note.len > 0) thumb_tokens else 0),
    });
}

fn lfm2Canvas(allocator: std.mem.Allocator, src: DecodedRgb, canvas_h: u32, canvas_w: u32, vp: chat_mod.VisionPreproc) ?[]f32 {
    const chw = allocator.alloc(f32, 3 * @as(usize, canvas_h) * canvas_w) catch return null;
    qwen_vision.resizeRgbNormalizedChw(allocator, chw, src.rgb, src.h, src.w, canvas_h, canvas_w, resampleFilterFor(vp)) catch {
        allocator.free(chw);
        return null;
    };
    return chw;
}

/// Emit the patch region at (`y0`, `x0`) of a prepared canvas as one `ImageData`.
fn lfm2Region(
    allocator: std.mem.Allocator,
    chw: []const f32,
    canvas_h: u32,
    canvas_w: u32,
    vp: chat_mod.VisionPreproc,
    y0: u32,
    x0: u32,
    region_h: u32,
    region_w: u32,
) ?chat_mod.ImageData {
    const C: u32 = 3;
    const gh = region_h / vp.patch;
    const gw = region_w / vp.patch;
    const n: usize = @as(usize, gh) * gw;
    const feat: usize = @as(usize, C) * vp.patch * vp.patch;
    const bytes = allocator.alloc(u8, n * feat * 4) catch return null;
    const out = @as([*]f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. n * feat];
    lfm2_vision.buildPixelValuesRegion(out, chw, C, canvas_h, canvas_w, vp.patch, y0, x0, region_h, region_w);
    return .{ .pixels = bytes, .width = region_w, .height = region_h, .grid_h = gh, .grid_w = gw };
}
