//! image decode, budgets; decodeVideoUrlContent; appendVideoUrlContent + tests

const std = @import("std");
const chat_mod = @import("../chat/chat.zig");
const model_mod = @import("../../sdk/config/model.zig");
const qwen_vision = @import("../../sdk/vision/qwen_vision.zig");
const muse_vision = @import("../../plugins/muse_glimmer/vision.zig");
const lfm2_vision = @import("../../plugins/lfm2/vision.zig");
const glm_vision = @import("../../plugins/glm5_next/vision.zig");
const dsv41_vision = @import("../../plugins/deepseek_v41/vision.zig");
const radio_vision = @import("../../plugins/nemotron_h/radio_vision.zig");
const log = @import("../../sdk/util/log.zig");
const plugin_media = @import("plugin_media.zig");
const stb = @import("stb");
const webp = @import("webp");
const testing = std.testing;

const visionPreprocFromConfig = @import("prepare.zig").visionPreprocFromConfig;
const videoPreprocFromConfig = @import("prepare.zig").videoPreprocFromConfig;
const logVisionPixelClamp = @import("prepare.zig").logVisionPixelClamp;
const RequestMedia = @import("request_media.zig").RequestMedia;
const decodeRadioVideo = @import("../../plugins/nemotron_h/media.zig").decodeRadioVideo;
const solidPngUrl = @import("request_media.zig").solidPngUrl;
const eg2ProcessorConfig = @import("prepare.zig").eg2ProcessorConfig;

/// Resize `src` onto a `canvas_h` x `canvas_w` normalized CHW buffer, owned by
/// the caller. One resample serves every tile cut from it.
/// The resample filter belongs to the ARCH's reference processor, not to the
/// resampler it shares: muse smart-resizes with Lanczos, LFM2-VL with PIL
/// BILINEAR (`Lfm2VlImageProcessor` uses it at all three sites — tile canvas,
/// thumbnail, single view), Qwen with bicubic. Reusing a neighbouring tower's
/// filter is silent: the geometry and token counts still match, the pixels do
/// not (bicubic loses 3 of 144 ScreenSpot-v2 items on
/// LFM2.5-VL and never wins one).
pub fn resampleFilterFor(vp: chat_mod.VisionPreproc) qwen_vision.Filter {
    return switch (vp.mode) {
        .muse => .lanczos,
        .lfm2 => .bilinear,
        else => .bicubic,
    };
}

/// A decoded source image as packed RGB8, owned by `allocator`. One owner and
/// one free path — stb and libwebp buffers are copied out and released here, so
/// callers that need the pixels for more than one pass (LFM2-VL's tiles) do not
/// juggle three different deallocators.
pub const DecodedRgb = struct {
    rgb: []u8,
    w: u32,
    h: u32,

    pub fn deinit(self: DecodedRgb, allocator: std.mem.Allocator) void {
        allocator.free(self.rgb);
    }
};

pub fn decodeRgbOwned(allocator: std.mem.Allocator, encoded: []const u8, composite_alpha: bool) ?DecodedRgb {
    var w: c_int = 0;
    var h: c_int = 0;
    var channels: c_int = 0;
    // stb first (JPEG/PNG/GIF first frame/BMP) — 4 channels so transparency composites onto white.
    if (stb.stbi_load_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &channels, 4)) |rgba| {
        defer stb.stbi_image_free(rgba);
        const total_px: usize = @intCast(w * h);
        const rgb = allocator.alloc(u8, total_px * 3) catch return null;
        for (0..total_px) |i| {
            const a: u16 = if (composite_alpha) rgba[i * 4 + 3] else 255;
            const inv_a = 255 - a;
            rgb[i * 3 + 0] = @intCast((a * @as(u16, rgba[i * 4 + 0]) + inv_a * 255) / 255);
            rgb[i * 3 + 1] = @intCast((a * @as(u16, rgba[i * 4 + 1]) + inv_a * 255) / 255);
            rgb[i * 3 + 2] = @intCast((a * @as(u16, rgba[i * 4 + 2]) + inv_a * 255) / 255);
        }
        return .{ .rgb = rgb, .w = @intCast(w), .h = @intCast(h) };
    }

    var webp_w: c_int = 0;
    var webp_h: c_int = 0;
    const decoded = webp.WebPDecodeRGB(encoded.ptr, encoded.len, &webp_w, &webp_h) orelse return null;
    defer webp.WebPFree(decoded);
    const total: usize = @as(usize, @intCast(webp_w)) * @as(usize, @intCast(webp_h)) * 3;
    const rgb = allocator.alloc(u8, total) catch return null;
    @memcpy(rgb, decoded[0..total]);
    return .{ .rgb = rgb, .w = @intCast(webp_w), .h = @intCast(webp_h) };
}

pub fn decodeImageToPixels(allocator: std.mem.Allocator, encoded: []const u8, vp: chat_mod.VisionPreproc) ?chat_mod.ImageData {
    const target: u32 = 768; // Gemma 4 default for square images
    if (vp.mode == .plugin) return plugin_media.prepareImage(allocator, @ptrCast(@alignCast(vp.plugin.?)), encoded);

    const src = decodeRgbOwned(allocator, encoded, vp.composite_alpha) orelse return null;
    if (vp.mode == .plugin_rgb) return .{ .pixels = src.rgb, .width = src.w, .height = src.h };
    defer src.deinit(allocator);
    const px = src.rgb.ptr;
    const src_w: u32 = src.w;
    const src_h: u32 = src.h;

    if (vp.mode == .radio) {
        const img = radio_vision.prepareImage(allocator, src.rgb, src_h, src_w) catch return null;
        log.info("  Decoded {d}x{d} image → RADIO {d}x{d} float32 CHW ({d} tokens, PIL bicubic)\n", .{ src_w, src_h, img.w, img.h, img.tokens });
        return .{ .pixels = img.pixels, .width = img.w, .height = img.h };
    }

    if (vp.mode == .gemma_budget) {
        const img = qwen_vision.budgetPixels(allocator, src.rgb, src_h, src_w, vp.patch, vp.merge, vp.max_tokens) catch return null;
        log.info("  Decoded {d}x{d} image → {d}x{d} float32 CHW ({d} soft tokens)\n", .{ src_w, src_h, img.w, img.h, (img.h / vp.patch / vp.merge) * (img.w / vp.patch / vp.merge) });
        return .{ .pixels = img.pixels, .width = img.w, .height = img.h };
    }

    // MiniCPM-V images are several pieces: `appendImageUrlContent` makes them.
    if (vp.mode == .minicpm) return null;

    if (vp.mode == .dsv41) {
        const pv = dsv41_vision.preprocess(allocator, src.rgb, src_h, src_w, vp.patch, vp.merge, vp.max_tokens, vp.min_pixels, vp.max_wh_ratio) catch return null;
        log.info("  Decoded {d}x{d} image → deepseek grid {d}x{d} ({d} tokens, canvas {d}x{d})\n", .{ src_w, src_h, pv.grid_h, pv.grid_w, dsv41_vision.spanTokens((pv.grid_h + vp.merge - 1) / vp.merge, (pv.grid_w + vp.merge - 1) / vp.merge), pv.w, pv.h });
        return .{ .pixels = pv.bytes, .width = pv.w, .height = pv.h, .grid_h = pv.grid_h, .grid_w = pv.grid_w };
    }

    if (vp.mode == .glm) {
        const pv = glm_vision.preprocess(allocator, src.rgb, src_h, src_w, vp.patch, vp.tps, vp.merge, vp.min_tokens, vp.max_tokens) catch return null;
        log.info("  Decoded {d}x{d} image → glm grid {d}x{d} ({d} tokens, canvas {d}x{d})\n", .{ src_w, src_h, pv.grid_h, pv.grid_w, (pv.grid_h / vp.merge) * (pv.grid_w / vp.merge), pv.w, pv.h });
        return .{ .pixels = pv.bytes, .width = pv.w, .height = pv.h, .grid_h = pv.grid_h, .grid_w = pv.grid_w };
    }

    // Patch-grid towers: smart-resize to a multiple of patch·merge, normalize
    // (x/255−0.5)/0.5 (both processors use mean/std 0.5), then emit that
    // processor's pixel_values. `grid_h/grid_w` carry the full patch grid;
    // token count = (gh/merge)·(gw/merge). Qwen bounds the PIXEL area, muse the
    // merged-token count, and their patch orders differ — see muse_vision.zig.
    if (vp.mode != .gemma) {
        const factor = std.math.mul(u32, vp.patch, vp.merge) catch return null;
        if (factor == 0 or vp.tps == 0) return null;
        const bounds = qwen_vision.effectivePixelBounds(vp.min_pixels, vp.max_pixels);
        const min_pixels = bounds.min;
        const max_pixels = bounds.max;
        if (bounds.clamped and vp.mode == .qwen) logVisionPixelClamp(vp.max_pixels);
        const rs = switch (vp.mode) {
            .muse => muse_vision.smartResize(src_h, src_w, factor, if (vp.max_tokens > 0) vp.max_tokens else model_mod.MUSE_MAX_IMAGE_TOKENS),
            .lfm2 => lfm2_vision.smartResize(src_h, src_w, vp.patch, vp.merge, vp.min_tokens, vp.max_tokens),
            else => qwen_vision.smartResizeImage(src_h, src_w, factor, min_pixels, max_pixels),
        };
        const rh = rs.h;
        const rw = rs.w;
        const C: u32 = 3;
        const gh = rh / vp.patch;
        const gw = rw / vp.patch;
        const n: usize = @as(usize, gh) * gw;
        const feat: usize = @as(usize, C) * vp.tps * vp.patch * vp.patch;
        const plane: usize = @as(usize, rh) * rw;

        const chw = allocator.alloc(f32, @as(usize, C) * plane) catch return null;
        defer allocator.free(chw);
        const source_len: usize = @as(usize, src_h) * src_w * C;
        qwen_vision.resizeRgbNormalizedChw(
            allocator,
            chw,
            px[0..source_len],
            src_h,
            src_w,
            rh,
            rw,
            resampleFilterFor(vp),
        ) catch return null;

        const pv_bytes = allocator.alloc(u8, n * feat * 4) catch return null;
        const pv_f32 = @as([*]f32, @ptrCast(@alignCast(pv_bytes.ptr)))[0 .. n * feat];
        switch (vp.mode) {
            .muse => muse_vision.buildPixelValues(pv_f32, chw, C, rh, rw, vp.patch, vp.tps),
            .lfm2 => lfm2_vision.buildPixelValues(pv_f32, chw, C, rh, rw, vp.patch),
            else => qwen_vision.buildPixelValues(pv_f32, chw, C, rh, rw, vp.patch, vp.tps, vp.merge),
        }
        log.info("  Decoded {d}x{d} image → {s} grid {d}x{d} ({d} tokens, resized {d}x{d})\n", .{ src_w, src_h, @tagName(vp.mode), gh, gw, n / (@as(usize, vp.merge) * vp.merge), rw, rh });
        return .{ .pixels = pv_bytes, .width = rw, .height = rh, .grid_h = gh, .grid_w = gw };
    }

    // Allocate float32 CHW output: [3, target, target]
    const out_size = 3 * target * target;
    const out_buf = allocator.alloc(u8, out_size * 4) catch return null;
    const float_buf: [*]f32 = @ptrCast(@alignCast(out_buf.ptr));

    // Bilinear resize + HWC→CHW + rescale to [0,1]
    for (0..target) |ty| {
        for (0..target) |tx| {
            // Map target pixel to source coordinates
            const sx_f: f32 = @as(f32, @floatFromInt(tx)) * @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(target));
            const sy_f: f32 = @as(f32, @floatFromInt(ty)) * @as(f32, @floatFromInt(src_h)) / @as(f32, @floatFromInt(target));

            // Nearest-neighbor for simplicity (bilinear adds complexity for marginal benefit here)
            const sx: u32 = @min(@as(u32, @intFromFloat(sx_f)), src_w - 1);
            const sy: u32 = @min(@as(u32, @intFromFloat(sy_f)), src_h - 1);

            const src_idx = (sy * src_w + sx) * 3;
            const dst_idx = ty * target + tx;

            // CHW: channel 0 (R), channel 1 (G), channel 2 (B)
            float_buf[0 * target * target + dst_idx] = @as(f32, @floatFromInt(px[src_idx + 0])) / 255.0;
            float_buf[1 * target * target + dst_idx] = @as(f32, @floatFromInt(px[src_idx + 1])) / 255.0;
            float_buf[2 * target * target + dst_idx] = @as(f32, @floatFromInt(px[src_idx + 2])) / 255.0;
        }
    }

    log.info("  Decoded {d}x{d} image → {d}x{d} float32 CHW\n", .{ src_w, src_h, target, target });
    return .{ .pixels = out_buf, .width = target, .height = target };
}

/// One frame of a client's `frames` array: a base64 image data URL.
pub fn decodeFrameUrl(allocator: std.mem.Allocator, url: []const u8, composite_alpha: bool) ?DecodedRgb {
    const sep = std.mem.indexOf(u8, url, ";base64,") orelse return null;
    const b64 = url[sep + 8 ..];
    const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(b64) catch return null;
    const raw = allocator.alloc(u8, decoded_size) catch return null;
    defer allocator.free(raw);
    std.base64.standard.Decoder.decode(raw, b64) catch return null;
    return decodeRgbOwned(allocator, raw, composite_alpha);
}

/// EmbeddingGemma 2's video processor on a client's frames: at most `vp.max_frames` of them, an even spread
/// that keeps the first and last, each fitted to the frame budget at the size the first kept frame decides
/// (a clip is one tensor in the reference). `grid_t` is the frame count and `pixels` the frames' float32 CHW
/// back to back. Null when any kept frame is not a decodable image.
fn decodeBudgetVideo(allocator: std.mem.Allocator, frame_urls: []const []const u8, vp: chat_mod.VisionPreproc) ?chat_mod.VideoData {
    if (frame_urls.len == 0 or vp.max_frames == 0) return null;
    const kept = @min(frame_urls.len, vp.max_frames);
    var bytes: []u8 = &.{};
    var size: qwen_vision.Resized = undefined;
    var frame_bytes: usize = 0;
    var done = false;
    defer if (!done and bytes.len > 0) allocator.free(bytes);
    for (0..kept) |k| {
        const rgb = decodeFrameUrl(allocator, frame_urls[qwen_vision.sampledFrame(k, frame_urls.len, vp.max_frames)], vp.composite_alpha) orelse return null;
        defer rgb.deinit(allocator);
        if (k == 0) {
            size = qwen_vision.budgetSize(rgb.h, rgb.w, vp.patch, vp.merge, vp.max_tokens) orelse return null;
            frame_bytes = 3 * @as(usize, size.h) * size.w * @sizeOf(f32);
            bytes = allocator.alloc(u8, kept * frame_bytes) catch return null;
        }
        const frame = qwen_vision.resizeToUnitChw(allocator, rgb.rgb, rgb.h, rgb.w, size.h, size.w) catch return null;
        defer allocator.free(frame.pixels);
        @memcpy(bytes[k * frame_bytes ..][0..frame_bytes], frame.pixels);
    }
    log.info("  Decoded {d} frames → {d} kept, {d}x{d} float32 CHW each ({d} soft tokens per frame)\n", .{
        frame_urls.len, kept, size.w, size.h, (size.h / vp.patch / vp.merge) * (size.w / vp.patch / vp.merge),
    });
    done = true;
    return .{ .pixels = bytes, .grid_t = @intCast(kept), .grid_h = size.h / vp.patch, .grid_w = size.w / vp.patch };
}

/// Decode a `video_url` block's `frames` array — already-decoded-by-the-client
/// JPEG/PNG data URLs, one per sampled frame; no video codec exists anywhere in
/// this codebase, so frame extraction is the client's job — into ONE
/// `chat_mod.VideoData`. All frames share ONE smart-resize target, computed
/// from the FIRST frame and applied to every frame (a video's whole patch grid
/// must be identical across frames), then grouped into `vp.tps`-sized temporal-
/// patch groups — the last group pads by repeating its final frame, matching
/// HF's video processor. Qwen's clips; EmbeddingGemma 2's (`.gemma_budget`) are `decodeBudgetVideo`'s.
fn decodeVideoUrlContent(allocator: std.mem.Allocator, frame_urls: []const []const u8, vp: chat_mod.VisionPreproc) ?chat_mod.VideoData {
    if (vp.mode == .gemma_budget) return decodeBudgetVideo(allocator, frame_urls, vp);
    if (vp.mode == .radio) return decodeRadioVideo(allocator, frame_urls);
    if (vp.mode == .plugin_rgb) return decodePluginVideo(allocator, frame_urls);
    if (vp.mode != .qwen or frame_urls.len == 0) return null;
    const factor = std.math.mul(u32, vp.patch, vp.merge) catch return null;
    if (factor == 0 or vp.tps == 0 or vp.tps > 8) return null;

    // Decode every frame to RGB8 first — frame 0's natural size decides the
    // resize target every later frame must also resize to.
    var decoded = std.ArrayList(DecodedRgb).empty;
    defer {
        for (decoded.items) |d| d.deinit(allocator);
        decoded.deinit(allocator);
    }
    for (frame_urls) |url| {
        const rgb = decodeFrameUrl(allocator, url, true) orelse return null;
        decoded.append(allocator, rgb) catch {
            rgb.deinit(allocator);
            return null;
        };
    }

    const bounds = qwen_vision.effectivePixelBounds(vp.min_pixels, vp.max_pixels);
    const min_pixels = bounds.min;
    const max_pixels = bounds.max;
    if (bounds.clamped) logVisionPixelClamp(vp.max_pixels);
    const first = decoded.items[0];
    const rs = qwen_vision.smartResizeImage(first.h, first.w, factor, min_pixels, max_pixels);
    const rh = rs.h;
    const rw = rs.w;
    const C: u32 = 3;
    const gh = rh / vp.patch;
    const gw = rw / vp.patch;
    const n_per_group: usize = @as(usize, gh) * gw;
    const feat: usize = @as(usize, C) * vp.tps * vp.patch * vp.patch;

    // Resize every frame to the shared (rh, rw) target.
    var frames_chw = std.ArrayList([]f32).empty;
    defer {
        for (frames_chw.items) |f| allocator.free(f);
        frames_chw.deinit(allocator);
    }
    for (decoded.items) |d| {
        const source_len: usize = @as(usize, d.h) * d.w * C;
        const chw = allocator.alloc(f32, @as(usize, C) * rh * rw) catch return null;
        qwen_vision.resizeRgbNormalizedChw(allocator, chw, d.rgb[0..source_len], d.h, d.w, rh, rw, resampleFilterFor(vp)) catch {
            allocator.free(chw);
            return null;
        };
        frames_chw.append(allocator, chw) catch {
            allocator.free(chw);
            return null;
        };
    }

    // Group into tps-sized temporal patches, padding the last group by
    // repeating its final frame.
    const grid_t: usize = (frames_chw.items.len + vp.tps - 1) / vp.tps;
    const pv_bytes = allocator.alloc(u8, grid_t * n_per_group * feat * 4) catch return null;
    const pv_f32 = @as([*]f32, @ptrCast(@alignCast(pv_bytes.ptr)))[0 .. grid_t * n_per_group * feat];

    var group_frames: [8][]const f32 = undefined;
    var g: usize = 0;
    while (g < grid_t) : (g += 1) {
        var k: usize = 0;
        while (k < vp.tps) : (k += 1) {
            const idx = @min(g * vp.tps + k, frames_chw.items.len - 1);
            group_frames[k] = frames_chw.items[idx];
        }
        const out_slice = pv_f32[g * n_per_group * feat ..][0 .. n_per_group * feat];
        qwen_vision.buildPixelValuesVideo(out_slice, group_frames[0..vp.tps], C, rh, rw, vp.patch, vp.merge);
    }

    log.info("  Decoded {d} frames → qwen video grid_t={d} grid {d}x{d} ({d} tokens, resized {d}x{d})\n", .{
        frame_urls.len, grid_t, gh, gw, grid_t * n_per_group / (@as(usize, vp.merge) * vp.merge), rw, rh,
    });
    return .{ .pixels = pv_bytes, .grid_t = @intCast(grid_t), .grid_h = gh, .grid_w = gw };
}

/// A plugin tower's clip: every frame decoded to RGB8 back to back, all at the first frame's size (the plugin
/// resizes; frames of a clip share one size). Null for an undecodable frame or one of another size.
fn decodePluginVideo(allocator: std.mem.Allocator, frame_urls: []const []const u8) ?chat_mod.VideoData {
    if (frame_urls.len == 0) return null;
    var bytes: []u8 = &.{};
    var w: u32 = 0;
    var h: u32 = 0;
    var done = false;
    defer if (!done and bytes.len > 0) allocator.free(bytes);
    for (frame_urls, 0..) |url, k| {
        const rgb = decodeFrameUrl(allocator, url, true) orelse return null;
        defer rgb.deinit(allocator);
        if (k == 0) {
            w = rgb.w;
            h = rgb.h;
            bytes = allocator.alloc(u8, frame_urls.len * rgb.rgb.len) catch return null;
        }
        if (rgb.w != w or rgb.h != h) return null;
        @memcpy(bytes[k * rgb.rgb.len ..][0..rgb.rgb.len], rgb.rgb);
    }
    done = true;
    return .{ .pixels = bytes, .grid_t = @intCast(frame_urls.len), .grid_h = h, .grid_w = w };
}

/// Decode a `video_url` block's `frames` array into `list`, mirroring
/// `appendImageUrlContent`'s append-or-drop shape.
pub fn appendVideoUrlContent(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(chat_mod.VideoData),
    frame_urls: []const []const u8,
    vp: chat_mod.VisionPreproc,
) void {
    if (decodeVideoUrlContent(allocator, frame_urls, vp)) |vid| {
        list.append(allocator, vid) catch allocator.free(vid.pixels);
    }
}

// ── Anthropic Messages API ──

// ── tests ──

test "decodeRgbOwned reads a GIF's first frame and a BMP, the formats the Anthropic and OpenAI image blocks allow beside JPEG/PNG/WebP" {
    // 2x2: red, green / blue, white (Pillow-written).
    const gif = "\x47\x49\x46\x38\x37\x61\x02\x00\x02\x00\x81\x00\x00\xff\xff\xff\x00\xff\x00\xff\x00\x00\x00\x00\xff\x2c\x00\x00\x00\x00\x02\x00\x02\x00\x00\x08\x07\x00\x05\x04\x18\x00\x20\x20\x00\x3b";
    const bmp = "\x42\x4d\x46\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x10\x00\x00\x00\xc4\x0e\x00\x00\xc4\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x00\x00\xff\xff\xff\x00\x00\x00\x00\xff\x00\xff\x00\x00\x00";
    for ([_][]const u8{ gif, bmp }) |bytes| {
        const d = decodeRgbOwned(std.testing.allocator, bytes, true) orelse return error.TestUnexpectedResult;
        defer std.testing.allocator.free(d.rgb);
        try std.testing.expectEqual(@as(u32, 2), d.w);
        try std.testing.expectEqual(@as(u32, 2), d.h);
        try std.testing.expectEqualSlices(u8, &.{ 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255 }, d.rgb);
    }
    try std.testing.expect(decodeRgbOwned(std.testing.allocator, "II*\x00not a tiff we read", true) == null);
}

test "clef: transparent image RGB follows the checkpoint processor" {
    const a = std.testing.allocator;
    const bytes = try @import("../../plugins/clef/http_test.zig").readFile(std.testing.io, "tests/fixtures/robot.png");
    defer a.free(bytes);
    const composited = decodeRgbOwned(a, bytes, true) orelse return error.BadImage;
    defer composited.deinit(a);
    const rgb = decodeRgbOwned(a, bytes, false) orelse return error.BadImage;
    defer rgb.deinit(a);
    try std.testing.expectEqualSlices(u8, &.{ 255, 255, 255 }, composited.rgb[0..3]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0 }, rgb.rgb[0..3]);
}

test "RequestMedia frees every decoded buffer it was handed" {
    // std.testing.allocator fails this test on a leak, which is the whole
    // point: the owner is the only thing that frees `pixels`/`samples`, so a
    // gutted `deinit` is caught here rather than in a week of RSS growth.
    const a = std.testing.allocator;
    var media = RequestMedia.init(a);

    const img_slot = try media.openImages();
    try media.images(img_slot).append(a, .{
        .pixels = try a.alloc(u8, 4096),
        .width = 32,
        .height = 32,
    });
    try media.images(img_slot).append(a, .{
        .pixels = try a.alloc(u8, 2048),
        .width = 16,
        .height = 32,
    });

    const vid_slot = try media.openVideos();
    try media.videos(vid_slot).append(a, .{
        .pixels = try a.alloc(u8, 1024),
        .grid_t = 1,
        .grid_h = 2,
        .grid_w = 2,
    });

    const aud_slot = try media.openAudio();
    try media.audio(aud_slot).append(a, .{ .samples = try a.alloc(u8, 512) });

    // A borrowed slice for `Message.images`; empty slots read back as null so a
    // message with no media keeps a null field rather than an empty slice.
    try std.testing.expectEqual(@as(usize, 2), (media.imagesSlice(img_slot) orelse return error.NoImages).len);
    try std.testing.expectEqual(@as(usize, 1), (media.videosSlice(vid_slot) orelse return error.NoVideos).len);
    try std.testing.expectEqual(@as(usize, 1), (media.audioSlice(aud_slot) orelse return error.NoAudio).len);
    const empty = try media.openImages();
    try std.testing.expect(media.imagesSlice(empty) == null);

    media.deinit();
}

test "EmbeddingGemma 2 video: client frames become one clip, cut to the cap with an even spread" {
    const a = testing.allocator;
    var config = eg2ProcessorConfig();
    config.gv_video_frames = 4;
    const vp = videoPreprocFromConfig(&config);

    // Ten frames whose red level is their index: the four kept are 0, 3, 6, 9 (np.linspace(0, 9, 4, dtype=int)).
    var urls: [10][]u8 = undefined;
    for (&urls, 0..) |*u, i| u.* = try solidPngUrl(a, 6, 4, .{ @intCast(i * 20), 0, 0, 255 });
    defer for (urls) |u| a.free(u);
    var frames: [10][]const u8 = undefined;
    for (urls, 0..) |u, i| frames[i] = u;

    var media = RequestMedia.init(a);
    defer media.deinit();
    const cut = try media.openVideos();
    appendVideoUrlContent(a, media.videos(cut), &frames, vp);
    const vid = (media.videosSlice(cut) orelse return error.Undecoded)[0];
    try testing.expectEqual(@as(u32, 4), vid.grid_t);
    try testing.expectEqual(@as(u32, 288 / 16), vid.grid_h);
    try testing.expectEqual(@as(u32, 480 / 16), vid.grid_w);
    const frame_floats: usize = 3 * 288 * 480;
    try testing.expectEqual(@as(usize, 4 * frame_floats * 4), vid.pixels.len);
    const px: []const f32 = @as([*]const f32, @ptrCast(@alignCast(vid.pixels.ptr)))[0 .. 4 * frame_floats];
    for ([_]usize{ 0, 3, 6, 9 }, 0..) |source, k| {
        const want = @as(f32, @floatFromInt(source * 20)) * (1.0 / 255.0);
        try testing.expectApproxEqAbs(want, px[k * frame_floats], 1e-6);
    }

    // Fewer frames than the cap are all kept; a frame that is not an image fails the clip, never silently thins it.
    const few = try media.openVideos();
    appendVideoUrlContent(a, media.videos(few), frames[0..3], vp);
    try testing.expectEqual(@as(u32, 3), (media.videosSlice(few) orelse return error.Undecoded)[0].grid_t);
    const bad = try media.openVideos();
    const broken = [_][]const u8{ frames[0], "data:image/png;base64,AAAA" };
    appendVideoUrlContent(a, media.videos(bad), &broken, vp);
    try testing.expect(media.videosSlice(bad) == null);
}

test "radio video frames share frame 0's patch grid" {
    const a = testing.allocator;
    var config = model_mod.ModelConfig{};
    config.radio_vision = true;
    const vp = visionPreprocFromConfig(&config);
    try testing.expectEqual(.radio, vp.mode);
    // 32×24 is the same 4:3 as a 640×480 frame, so the grid is the clip's grid.
    const url = try solidPngUrl(a, 32, 24, .{ 10, 20, 30, 255 });
    defer a.free(url);
    const frames = [_][]const u8{ url, url };
    const vid = decodeVideoUrlContent(a, &frames, vp) orelse return error.Undecoded;
    defer a.free(vid.pixels);
    try testing.expectEqual(@as(u32, 2), vid.grid_t);
    try testing.expectEqual(@as(u32, 576), vid.grid_w);
    try testing.expectEqual(@as(u32, 448), vid.grid_h);
    try testing.expectEqual(@as(usize, 2 * 3 * 576 * 448 * 4), vid.pixels.len);
    const broken = [_][]const u8{ url, "data:image/png;base64,AAAA" };
    try testing.expect(decodeVideoUrlContent(a, &broken, vp) == null);
}
