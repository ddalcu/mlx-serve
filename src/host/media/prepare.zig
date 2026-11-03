//! visionPreprocFromConfig, videoPreprocFromConfig

const std = @import("std");
const chat_mod = @import("../chat/chat.zig");
const model_mod = @import("../../sdk/config/model.zig");
const qwen_vision = @import("../../sdk/vision/qwen_vision.zig");
const radio_vision = @import("../../plugins/nemotron_h/radio_vision.zig");
const log = @import("../../sdk/util/log.zig");
const testing = std.testing;

const parseImageUrlContent = @import("request_media.zig").parseImageUrlContent;
const decodeImageToPixels = @import("decode.zig").decodeImageToPixels;

/// Decode a JPEG/PNG image buffer to float32 CHW pixels, resized to target_size.
/// Uses stb_image for decoding, then nearest-neighbor resize + CHW conversion.
/// Decode an image content URL into preprocessed float32 CHW pixels. Supports:
///   data:image/x-mlx-pixels;base64,... (already-preprocessed float32 CHW)
///   data:image/jpeg|png|webp|...;base64,... (decoded + resized via stb_image / libwebp)
/// Returns null on any decode failure (caller treats as missing image).
/// Derive per-request image preprocessing params from the loaded model config.
pub fn visionPreprocFromConfig(config: *const model_mod.ModelConfig) chat_mod.VisionPreproc {
    // The plugin's own `prepare` resizes: the host only decodes.
    if (config.plugin_media) return .{ .mode = .plugin_rgb };
    if (config.radio_vision) return .{ .mode = .radio, .patch = radio_vision.PATCH, .merge = radio_vision.DOWNSAMPLE };
    if (config.lfm2_vision) {
        // NaFlex: no temporal axis and no merge-block patch order — the
        // projector unshuffles AFTER the tower, so the grid stays plain
        // row-major and `merge` only sizes the token count.
        return .{
            .mode = .lfm2,
            .patch = config.vision_patch_size,
            .tps = 1,
            .merge = config.lv_downsample,
            .min_tokens = config.lv_min_image_tokens,
            .max_tokens = config.lv_max_image_tokens,
            .tile_size = if (config.lv_split_images) config.lv_tile_size else 0,
            .min_tiles = config.lv_min_tiles,
            .max_tiles = config.lv_max_tiles,
            .use_thumbnail = config.lv_use_thumbnail,
            .pixels_tolerance = config.lv_pixels_tolerance,
        };
    }
    if (config.minicpm_vision) return .{
        .mode = .minicpm,
        .patch = config.vision_patch_size,
        .max_slices = config.mcv_max_slice_nums,
    };
    if (config.plugin) |b| if (b.outside and b.table.media != null) return .{ .mode = .plugin, .plugin = b };
    if (config.dsv41_vision) return .{
        .mode = .dsv41,
        .patch = config.vision_patch_size,
        .merge = config.dv_downsample,
        .max_tokens = config.dv_max_image_tokens,
        .min_pixels = config.dv_min_pixels,
        .max_wh_ratio = config.dv_max_wh_ratio,
    };
    if (config.glm_vision) return .{
        .mode = .glm,
        .patch = config.qv_patch,
        .tps = config.qv_temporal_patch,
        .merge = config.qv_merge,
        .min_tokens = config.glmv_min_image_tokens,
        .max_tokens = config.glmv_max_image_tokens,
    };
    if (config.gv_budgeted) return .{
        .mode = .gemma_budget,
        .patch = config.vision_patch_size,
        .merge = config.vision_pooling_kernel,
        .max_tokens = config.gv_image_tokens,
        .composite_alpha = false,
    };
    if (!config.qwen_vision and !config.muse_vision) return .{};
    return .{
        .mode = if (config.muse_vision) .muse else .qwen,
        .patch = config.qv_patch,
        .tps = config.qv_temporal_patch,
        .merge = config.qv_merge,
        .min_pixels = config.qv_min_pixels,
        .max_pixels = config.qv_max_pixels,
        .max_tokens = config.mv_max_image_tokens,
    };
}

/// The video processor's view of `visionPreprocFromConfig`: frames get their own, smaller, token budget.
pub fn videoPreprocFromConfig(config: *const model_mod.ModelConfig) chat_mod.VisionPreproc {
    var vp = visionPreprocFromConfig(config);
    if (config.gv_budgeted) {
        vp.max_tokens = config.gv_video_tokens;
        vp.max_frames = config.gv_video_frames;
    }
    return vp;
}

var vision_pixel_clamp_logged: bool = false;

/// One-shot: the checkpoint declared more pixels than the ViT can attend.
pub fn logVisionPixelClamp(declared: u32) void {
    if (vision_pixel_clamp_logged) return;
    vision_pixel_clamp_logged = true;
    log.info("[vision] image area capped at {d} px (checkpoint declares {d}): the ViT materializes its full attention\n", .{ qwen_vision.ENGINE_MAX_PIXELS, declared });
}

// ── tests ──

test "visionPreprocFromConfig threads each tower's processor bounds" {
    const qwen = visionPreprocFromConfig(&.{
        .qwen_vision = true,
        .qv_patch = 14,
        .qv_temporal_patch = 2,
        .qv_merge = 2,
        .qv_min_pixels = 65536,
        .qv_max_pixels = 16777216,
    });
    try std.testing.expectEqual(.qwen, qwen.mode);
    try std.testing.expectEqual(@as(u32, 14), qwen.patch);
    try std.testing.expectEqual(@as(u32, 2), qwen.tps);
    try std.testing.expectEqual(@as(u32, 2), qwen.merge);
    try std.testing.expectEqual(@as(u32, 65536), qwen.min_pixels);
    try std.testing.expectEqual(@as(u32, 16777216), qwen.max_pixels);

    // muse caps on MERGED tokens instead of pixels, and must not be routed
    // through the Qwen resize (different algorithm, different patch order).
    const muse = visionPreprocFromConfig(&.{
        .muse_vision = true,
        .qv_patch = 14,
        .qv_temporal_patch = 2,
        .qv_merge = 2,
        .mv_max_image_tokens = 4096,
    });
    try std.testing.expectEqual(.muse, muse.mode);
    try std.testing.expectEqual(@as(u32, 4096), muse.max_tokens);
    try std.testing.expectEqual(.gemma, visionPreprocFromConfig(&.{}).mode);
}

test "GLM-5.3-Flash images decode onto the processor's padded canvas" {
    const glm = visionPreprocFromConfig(&.{ .glm_vision = true, .qv_patch = 14, .qv_temporal_patch = 2, .qv_merge = 2, .glmv_min_image_tokens = 16, .glmv_max_image_tokens = 8000 });
    try std.testing.expectEqual(.glm, glm.mode);
    try std.testing.expectEqual(@as(u32, 8000), glm.max_tokens);
    const a = std.testing.allocator;
    const jpeg = try @import("../../plugins/clef/http_test.zig").readFile(std.testing.io, "tests/fixtures/house.jpeg");
    defer a.free(jpeg);
    const img = decodeImageToPixels(a, jpeg, glm) orelse return error.BadImage;
    defer a.free(img.pixels);
    // 730x487 fits the budget: padded to 756x504, never resized (the reference's own numbers).
    try std.testing.expectEqual(@as(u32, 756), img.width);
    try std.testing.expectEqual(@as(u32, 504), img.height);
    try std.testing.expectEqual(@as(u32, 36), img.grid_h);
    try std.testing.expectEqual(@as(u32, 54), img.grid_w);
    try std.testing.expectEqual(@as(usize, 36 * 54 * 3 * 2 * 14 * 14 * 4), img.pixels.len);
}

test "an x-mlx-pixels payload is refused by a patch-grid tower (it is a Gemma format)" {
    // Live crash the app preprocesses images into Gemma's square
    // `x-mlx-pixels` buffer unless it recognises the arch, so Muse-Glimmer got
    // one. That buffer carries NO patch grid, so the encode routed to the SigLIP
    // `forward` — whose weights are null sentinels on a patch-grid encoder —
    // and killed the server. The original image is unrecoverable from it (already
    // resized and normalized), so the only honest answer is to refuse the block.
    const url = "data:image/x-mlx-pixels;base64,AAAAAM3MzD3NzEw+mpmZPs3MzD4AAAA/mpkZPzMzMz/NzEw/ZmZmPwAAgD/NzIw/";
    const gemma = parseImageUrlContent(std.testing.allocator, url, .{}) orelse return error.GemmaMustAccept;
    defer std.testing.allocator.free(gemma.pixels);
    try std.testing.expectEqual(@as(u32, 2), gemma.width);
    try std.testing.expectEqual(@as(u32, 0), gemma.grid_h);

    for ([_]chat_mod.VisionPreproc{ .{ .mode = .muse }, .{ .mode = .qwen }, .{ .mode = .glm }, .{ .mode = .minicpm }, .{ .mode = .gemma_budget }, .{ .mode = .dsv41 } }) |vp| {
        try std.testing.expect(parseImageUrlContent(std.testing.allocator, url, vp) == null);
    }
}

pub fn eg2ProcessorConfig() model_mod.ModelConfig {
    var config = model_mod.ModelConfig{ .model_type = "embedding_gemma2", .gv_budgeted = true };
    config.vision_patch_size = 16;
    config.vision_pooling_kernel = 3;
    config.gv_image_tokens = 70;
    config.gv_video_tokens = 70;
    config.gv_video_frames = 32;
    config.image_token_id = 9;
    config.video_token_id = 10;
    config.boi_token_id = 5;
    config.eoi_token_id = 6;
    return config;
}

test "EmbeddingGemma 2 preprocessing: images and video frames are fitted to their own soft-token budgets" {
    var config = eg2ProcessorConfig();
    config.gv_image_tokens = 280;
    config.gv_video_tokens = 140;
    const image = visionPreprocFromConfig(&config);
    try testing.expectEqual(.gemma_budget, image.mode);
    try testing.expectEqual(@as(u32, 16), image.patch);
    try testing.expectEqual(@as(u32, 3), image.merge);
    try testing.expectEqual(@as(u32, 280), image.max_tokens);
    // The processor's PIL conversion drops alpha instead of compositing it.
    try testing.expect(!image.composite_alpha);
    const video = videoPreprocFromConfig(&config);
    try testing.expectEqual(.gemma_budget, video.mode);
    try testing.expectEqual(@as(u32, 140), video.max_tokens);
    try testing.expectEqual(@as(u32, 32), video.max_frames);
}
