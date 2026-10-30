//! MediaItem, placeholder expansion, appendMediaRun

const std = @import("std");
const chat_mod = @import("../chat/chat.zig");
const model_mod = @import("../../sdk/config/model.zig");
const radio_vision = @import("../../plugins/nemotron_h/radio_vision.zig");
const MediaSpan = @import("../../sdk/kv/media_span.zig").MediaSpan;
const testing = std.testing;

const appendRadioVideoRun = @import("../../plugins/nemotron_h/media.zig").appendRadioVideoRun;
const appendMiniCpmImageRun = @import("../../plugins/qwen3_5/minicpm_media.zig").appendMiniCpmImageRun;
const appendLfm2ImageRun = @import("../../plugins/lfm2/media.zig").appendLfm2ImageRun;
const eg2ProcessorConfig = @import("prepare.zig").eg2ProcessorConfig;

/// One media item: what the chat template renders as ONE placeholder.
pub const MediaItem = union(enum) {
    /// One source image: a single entry, or an LFM2-VL tile grid + thumbnail.
    image: []const chat_mod.ImageData,
    video: *const chat_mod.VideoData,
    audio: *const chat_mod.AudioData,
};

/// Every media item in `messages`, in the order `chat.appendMediaContentParts`
/// renders them.
pub fn collectMediaItems(allocator: std.mem.Allocator, messages: []const chat_mod.Message) ![]MediaItem {
    var items = std.ArrayList(MediaItem).empty;
    errdefer items.deinit(allocator);
    for (messages) |*m| {
        const imgs = m.images orelse &.{};
        var i: usize = 0;
        while (i < imgs.len) {
            var j = i + 1;
            while (j < imgs.len and !chat_mod.isImageItemStart(imgs[j])) j += 1;
            try items.append(allocator, .{ .image = imgs[i..j] });
            i = j;
        }
        for (m.videos orelse &.{}) |*v| try items.append(allocator, .{ .video = v });
        for (m.audio orelse &.{}) |*a| try items.append(allocator, .{ .audio = a });
    }
    return items.toOwnedSlice(allocator);
}

pub const MediaTokens = struct {
    config: *const model_mod.ModelConfig,
    /// `<img>`, emitted for each RADIO video part. Distinct from the image token.
    video_sentinel: u32 = 0,
    /// `</img>`, closing each tubelet's kept `<image>` run.
    img_end: u32 = 0,
};

pub fn placeholderKindOf(token: u32, mt: MediaTokens) ?std.meta.Tag(MediaItem) {
    if (token == 0) return null;
    if (token == mt.config.image_token_id) return .image;
    if (mt.video_sentinel != 0 and token == mt.video_sentinel) return .video;
    if (token == mt.config.video_token_id) return .video;
    if (token == mt.config.audio_token_id) return .audio;
    return null;
}

pub const MEDIA_PLACEHOLDER_REJECT = "the rendered prompt does not carry one media placeholder per attached item: the chat template failed to render (see the server log), cannot place media in this message, or the text contains a literal media token";

/// The k-th placeholder the template rendered must be item k, by kind.
pub fn placeholdersMatchOf(prompt_ids: []const u32, items: []const MediaItem, mt: MediaTokens) bool {
    var k: usize = 0;
    for (prompt_ids) |t| {
        const kind = placeholderKindOf(t, mt) orelse continue;
        if (k == items.len or std.meta.activeTag(items[k]) != kind) return false;
        k += 1;
    }
    return k == items.len;
}

pub fn placeholdersMatch(prompt_ids: []const u32, items: []const MediaItem, config: *const model_mod.ModelConfig) bool {
    return placeholdersMatchOf(prompt_ids, items, .{ .config = config });
}

const ExpandedPrompt = struct { ids: []u32, media: []MediaSpan };

/// Replace the k-th rendered placeholder with item k's soft-token run;
/// `media[k].start` is the run's first soft-token row.
pub fn expandMediaPlaceholders(
    allocator: std.mem.Allocator,
    prompt_ids: []const u32,
    items: []const MediaItem,
    item_tokens: []const usize,
    item_keys: []const u64,
    config: *const model_mod.ModelConfig,
) !ExpandedPrompt {
    return expandMedia(allocator, prompt_ids, items, item_tokens, item_keys, .{ .config = config }, ByteTokenizer{}, &.{});
}

/// `item_keep[k]`, when set, is the pre-EVS mask for a RADIO video. Null keeps
/// the tower's own run (Qwen, Gemma, still images).
pub fn expandMedia(
    allocator: std.mem.Allocator,
    prompt_ids: []const u32,
    items: []const MediaItem,
    item_tokens: []const usize,
    item_keys: []const u64,
    mt: MediaTokens,
    tok: anytype,
    item_keep: []const ?[]const bool,
) !ExpandedPrompt {
    if (!placeholdersMatchOf(prompt_ids, items, mt)) return error.MediaPlaceholderMismatch;
    var ids = std.ArrayList(u32).empty;
    errdefer ids.deinit(allocator);
    const media = try allocator.alloc(MediaSpan, items.len);
    errdefer allocator.free(media);
    var k: usize = 0;
    var images_before: usize = 0;
    for (prompt_ids) |t| {
        if (placeholderKindOf(t, mt) == null) {
            try ids.append(allocator, t);
            continue;
        }
        const run_start = ids.items.len;
        if (items[k] == .video and mt.config.radio_vision) {
            const mask: ?[]const bool = if (k < item_keep.len) item_keep[k] else null;
            try appendRadioVideoRun(allocator, &ids, items[k].video, item_tokens[k], mask, mt, tok);
        } else {
            try appendMediaRun(allocator, &ids, items[k], item_tokens[k], images_before, mt.config);
        }
        images_before += @intFromBool(items[k] == .image);
        var first = run_start;
        while (first < ids.items.len and placeholderKindOf(ids.items[first], mt) == null) first += 1;
        media[k] = .{ .start = @intCast(first), .key = item_keys[k] };
        k += 1;
    }
    return .{ .ids = try ids.toOwnedSlice(allocator), .media = media };
}

/// Each rendered placeholder replaced by its item's run as the plugin laid it out (`runs[k]`, which holds the rows
/// the splice fills); the item's span starts at the run's first placeholder.
pub fn expandRuns(allocator: std.mem.Allocator, prompt_ids: []const u32, items: []const MediaItem, runs: []const []const u32, item_keys: []const u64, mt: MediaTokens) !ExpandedPrompt {
    if (runs.len != items.len or !placeholdersMatchOf(prompt_ids, items, mt)) return error.MediaPlaceholderMismatch;
    var ids = std.ArrayList(u32).empty;
    errdefer ids.deinit(allocator);
    const media = try allocator.alloc(MediaSpan, items.len);
    errdefer allocator.free(media);
    var k: usize = 0;
    for (prompt_ids) |t| {
        if (placeholderKindOf(t, mt) == null) {
            try ids.append(allocator, t);
            continue;
        }
        const run_start = ids.items.len;
        try ids.appendSlice(allocator, runs[k]);
        var first = run_start;
        while (first < ids.items.len and placeholderKindOf(ids.items[first], mt) == null) first += 1;
        media[k] = .{ .start = @intCast(first), .key = item_keys[k] };
        k += 1;
    }
    return .{ .ids = try ids.toOwnedSlice(allocator), .media = media };
}

/// What EmbeddingGemma 2's media runs add to a prompt once their placeholders are expanded, from the
/// preprocessed sizes alone: each image or video FRAME is its soft tokens between `<boi>` and `<eoi>`.
pub fn mediaRunTokens(items: []const MediaItem, config: *const model_mod.ModelConfig) usize {
    const side = @as(usize, config.vision_patch_size) * config.vision_pooling_kernel;
    var total: usize = 0;
    for (items) |item| switch (item) {
        .image => |pieces| for (pieces) |img| {
            total += 2 + (img.height / side) * (img.width / side);
        },
        .video => |clip| total += clip.grid_t * (2 + (clip.grid_h / config.vision_pooling_kernel) * (clip.grid_w / config.vision_pooling_kernel)),
        .audio => {},
    };
    return total;
}

/// Qwen's template renders the `<|vision_start|>`/`<|vision_end|>` wrap
/// itself; every other tower's run carries its own open/close tokens.
fn appendMediaRun(allocator: std.mem.Allocator, ids: *std.ArrayList(u32), item: MediaItem, n: usize, images_before: usize, config: *const model_mod.ModelConfig) !void {
    if (item == .image and config.minicpm_vision) return appendMiniCpmImageRun(allocator, ids, item.image, images_before, config);
    // EmbeddingGemma 2 wraps every video FRAME like an image of its own.
    if (item == .video and config.gv_budgeted) {
        const frames: usize = item.video.grid_t;
        if (frames == 0) return;
        for (0..frames) |_| {
            try ids.append(allocator, config.boi_token_id);
            try ids.appendNTimes(allocator, config.video_token_id, n / frames);
            try ids.append(allocator, config.eoi_token_id);
        }
        return;
    }
    const wrap = !config.qwen_vision and !config.radio_vision;
    const open: u32, const pad: u32, const close: u32 = switch (item) {
        .image => |pieces| blk: {
            if (config.lfm2_vision) return appendLfm2ImageRun(allocator, ids, pieces, config);
            break :blk .{ if (wrap) config.boi_token_id else 0, config.image_token_id, if (wrap) config.eoi_token_id else 0 };
        },
        .video => .{ if (wrap) config.boi_token_id else 0, config.video_token_id, if (wrap) config.eoi_token_id else 0 },
        .audio => .{ config.boa_token_id, config.audio_token_id, config.eoa_token_id },
    };
    if (open > 0) try ids.append(allocator, open);
    try ids.appendNTimes(allocator, pad, n);
    if (close > 0) try ids.append(allocator, close);
}

// ── tests ──

test "expandRuns: each placeholder becomes its item's plugin run, text around the rows kept, spans at the first row" {
    const t = testing;
    const img = [_]chat_mod.ImageData{.{ .pixels = "a", .width = 1, .height = 1 }};
    const clip = chat_mod.VideoData{ .pixels = "", .grid_t = 2, .grid_h = 1, .grid_w = 1 };
    var config = model_mod.ModelConfig{};
    config.image_token_id = 9;
    config.video_token_id = 10;
    const items = [_]MediaItem{ .{ .image = &img }, .{ .video = &clip } };
    // `<|vision_start|>` 7, `<|vision_end|>` 8; the video run opens with a timestamp's text (50, 51).
    const prompt = [_]u32{ 1, 7, 9, 8, 2, 7, 10, 8 };
    const runs = [_][]const u32{ &.{ 9, 9, 9 }, &.{ 50, 51, 7, 10, 10, 8 } };
    const out = try expandRuns(t.allocator, &prompt, &items, &runs, &.{ 11, 12 }, .{ .config = &config });
    defer t.allocator.free(out.ids);
    defer t.allocator.free(out.media);
    try t.expectEqualSlices(u32, &.{ 1, 7, 9, 9, 9, 8, 2, 7, 50, 51, 7, 10, 10, 8, 8 }, out.ids);
    try t.expectEqual(@as(u32, 2), out.media[0].start);
    try t.expectEqual(@as(u32, 11), out.media[1].start);
    try t.expectEqual(@as(u64, 12), out.media[1].key);
    try t.expectError(error.MediaPlaceholderMismatch, expandRuns(t.allocator, &prompt, &items, runs[0..1], &.{11}, .{ .config = &config }));
}

test "expandMediaPlaceholders: each rendered placeholder becomes its own item's run" {
    const t = testing;
    const img_a = [_]chat_mod.ImageData{.{ .pixels = "a", .width = 1, .height = 1 }};
    const img_b = [_]chat_mod.ImageData{.{ .pixels = "b", .width = 1, .height = 1 }};
    const aud = chat_mod.AudioData{ .samples = "pcm!" };
    // Qwen: the template rendered the vision_start/end wrap (7/8) itself.
    {
        var config = model_mod.ModelConfig{};
        config.qwen_vision = true;
        config.image_token_id = 9;
        config.boi_token_id = 5;
        config.eoi_token_id = 6;
        const items = [_]MediaItem{ .{ .image = &img_a }, .{ .image = &img_b } };
        const prompt = [_]u32{ 1, 7, 9, 8, 2, 7, 9, 8, 3 };
        const out = try expandMediaPlaceholders(t.allocator, &prompt, &items, &.{ 3, 2 }, &.{ 11, 12 }, &config);
        defer t.allocator.free(out.ids);
        defer t.allocator.free(out.media);
        try t.expectEqualSlices(u32, &.{ 1, 7, 9, 9, 9, 8, 2, 7, 9, 9, 8, 3 }, out.ids);
        try t.expectEqual(@as(u32, 2), out.media[0].start);
        try t.expectEqual(@as(u32, 8), out.media[1].start);
    }
    // Gemma / Muse: the run carries its own BOI/EOI, audio BOA/EOA.
    {
        var config = model_mod.ModelConfig{};
        config.image_token_id = 9;
        config.boi_token_id = 5;
        config.eoi_token_id = 6;
        config.audio_token_id = 4;
        config.boa_token_id = 2;
        config.eoa_token_id = 3;
        const items = [_]MediaItem{ .{ .image = &img_a }, .{ .audio = &aud } };
        const prompt = [_]u32{ 1, 9, 7, 4, 8 };
        const out = try expandMediaPlaceholders(t.allocator, &prompt, &items, &.{ 2, 3 }, &.{ 11, 12 }, &config);
        defer t.allocator.free(out.ids);
        defer t.allocator.free(out.media);
        try t.expectEqualSlices(u32, &.{ 1, 5, 9, 9, 6, 7, 2, 4, 4, 4, 3, 8 }, out.ids);
        try t.expectEqual(@as(u32, 2), out.media[0].start);
        try t.expectEqual(@as(u32, 7), out.media[1].start);
    }
    // A count or kind mismatch is refused, never shifted onto the next item.
    {
        var config = model_mod.ModelConfig{};
        config.qwen_vision = true;
        config.image_token_id = 9;
        config.video_token_id = 10;
        const one = [_]MediaItem{.{ .image = &img_a }};
        try t.expectError(error.MediaPlaceholderMismatch, expandMediaPlaceholders(t.allocator, &.{ 1, 9, 9, 2 }, &one, &.{1}, &.{11}, &config));
        try t.expectError(error.MediaPlaceholderMismatch, expandMediaPlaceholders(t.allocator, &.{ 1, 2 }, &one, &.{1}, &.{11}, &config));
        try t.expectError(error.MediaPlaceholderMismatch, expandMediaPlaceholders(t.allocator, &.{ 1, 10, 2 }, &one, &.{1}, &.{11}, &config));
    }
}

test "expandMediaPlaceholders: an LFM2-VL tiled source is one item, every tile labelled" {
    // `<|image_start|>`, per tile `<|img_row_R_col_C|>` + its pads, then
    // `<|img_thumbnail|>` + the thumbnail's pads, then `<|image_end|>`. A flat
    // pad run still splices (the counts match) but loses where each tile sits.
    var config = model_mod.ModelConfig{ .model_type = "lfm2" };
    config.lfm2_vision = true;
    config.image_token_id = 124907;
    config.boi_token_id = 125009;
    config.eoi_token_id = 125010;
    config.lv_thumbnail_token_id = 125008;
    config.lv_row_col_base_id = 124908;
    config.lv_max_tiles = 10;
    config.lv_downsample = 2;

    // A 2x2 tile grid (4 patches each → 1 pad each) plus a 1-pad thumbnail,
    // then an untiled image.
    const imgs = [_]chat_mod.ImageData{
        .{ .pixels = "", .width = 0, .height = 0, .grid_h = 2, .grid_w = 2, .tile_rows = 2, .tile_cols = 2, .tile_index = 0 },
        .{ .pixels = "", .width = 0, .height = 0, .grid_h = 2, .grid_w = 2, .tile_rows = 2, .tile_cols = 2, .tile_index = 1 },
        .{ .pixels = "", .width = 0, .height = 0, .grid_h = 2, .grid_w = 2, .tile_rows = 2, .tile_cols = 2, .tile_index = 2 },
        .{ .pixels = "", .width = 0, .height = 0, .grid_h = 2, .grid_w = 2, .tile_rows = 2, .tile_cols = 2, .tile_index = 3 },
        .{ .pixels = "", .width = 0, .height = 0, .grid_h = 2, .grid_w = 2, .tile_rows = 2, .tile_cols = 2, .tile_index = 4 },
        .{ .pixels = "", .width = 0, .height = 0, .grid_h = 4, .grid_w = 2 },
    };
    const msgs = [_]chat_mod.Message{.{ .role = "user", .content = "hi", .images = &imgs }};
    const items = try collectMediaItems(testing.allocator, &msgs);
    defer testing.allocator.free(items);
    try testing.expectEqual(@as(usize, 2), items.len);

    const prompt = [_]u32{ 1, 124907, 124907, 2 };
    const out = try expandMediaPlaceholders(testing.allocator, &prompt, items, &.{ 5, 2 }, &.{ 11, 12 }, &config);
    defer testing.allocator.free(out.ids);
    defer testing.allocator.free(out.media);
    // Row/col ids are laid out over the MAX grid (10 wide), not this image's:
    // (0,0)=124908, (0,1)=124909, (1,0)=124918, (1,1)=124919.
    const want = [_]u32{
        1,
        125009,
        124908,
        124907,
        124909,
        124907,
        124918,
        124907,
        124919,
        124907,
        125008,
        124907,
        125010,
        125009,
        124907,
        124907,
        125010,
        2,
    };
    try testing.expectEqualSlices(u32, &want, out.ids);
    try testing.expectEqual(@as(u32, 3), out.media[0].start);
    try testing.expectEqual(@as(u32, 14), out.media[1].start);
}

test "expandMediaPlaceholders: an EmbeddingGemma 2 video is one wrapped block per frame" {
    var config = eg2ProcessorConfig();
    // Three frames of 2x2 patches pool to one soft token each... here 4 per frame: 12 rows in all.
    const clip = chat_mod.VideoData{ .pixels = "", .grid_t = 3, .grid_h = 6, .grid_w = 6 };
    const img = [_]chat_mod.ImageData{.{ .pixels = "", .width = 0, .height = 0 }};
    const items = [_]MediaItem{ .{ .video = &clip }, .{ .image = &img } };
    const prompt = [_]u32{ 1, 10, 7, 9, 2 };
    const out = try expandMediaPlaceholders(testing.allocator, &prompt, &items, &.{ 12, 2 }, &.{ 11, 12 }, &config);
    defer testing.allocator.free(out.ids);
    defer testing.allocator.free(out.media);
    const frame = [_]u32{ 5, 10, 10, 10, 10, 6 };
    const want = [_]u32{1} ++ frame ++ frame ++ frame ++ [_]u32{7} ++ [_]u32{ 5, 9, 9, 6 } ++ [_]u32{2};
    try testing.expectEqualSlices(u32, &want, out.ids);
}

/// A tokenizer for the prompt tests: one id per byte, so the ids spell the text.
pub const ByteTokenizer = struct {
    pub fn encode(_: ByteTokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        const out = try allocator.alloc(u32, text.len);
        for (text, out) |c, *o| o.* = c;
        return out;
    }
};

test "radio prompt: a video sentinel expands into labeled tubelets and the keep mask" {
    const a = testing.allocator;
    var config = model_mod.ModelConfig{};
    config.radio_vision = true;
    config.image_token_id = 18;
    const mt = MediaTokens{ .config = &config, .video_sentinel = 19, .img_end = 20 };
    const img = [_]chat_mod.ImageData{.{ .pixels = "", .width = 64, .height = 64 }};

    // A still image stays a bare run of `<image>`. The tubelet wrap is video-only.
    const image_items = [_]MediaItem{.{ .image = &img }};
    const image_out = try expandMedia(a, &.{ 1, 18, 2 }, &image_items, &.{4}, &.{11}, mt, ByteTokenizer{}, &.{});
    defer a.free(image_out.ids);
    defer a.free(image_out.media);
    try testing.expectEqualSlices(u32, &.{ 1, 18, 18, 18, 18, 2 }, image_out.ids);
    try testing.expectEqual(@as(u32, 1), image_out.media[0].start);

    // 64×64 is a 4×4 patch grid and 2×2 tokens after the shuffle. Three frames
    // are two tubelets; the second keeps only its first and third token.
    const clip = chat_mod.VideoData{ .pixels = "", .grid_t = 3, .grid_h = 64, .grid_w = 64 };
    const items = [_]MediaItem{ .{ .image = &img }, .{ .video = &clip } };
    const mask = [_]bool{ true, true, true, true, true, false, true, false };
    const keeps = [_]?[]const bool{ null, &mask };
    const out = try expandMedia(a, &.{ 1, 18, 19, 2 }, &items, &.{ 2, 6 }, &.{ 11, 12 }, mt, ByteTokenizer{}, &keeps);
    defer a.free(out.ids);
    defer a.free(out.media);

    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(a);
    try want.append(a, 1);
    try want.appendNTimes(a, 18, 2);
    var buf: [96]u8 = undefined;
    const lab0 = radio_vision.videoGroupSegment(&buf, 0, 3);
    var lab0_owned: [96]u8 = undefined;
    @memcpy(lab0_owned[0..lab0.len], lab0);
    const ids0 = try ByteTokenizer.encode(.{}, a, lab0_owned[0..lab0.len]);
    defer a.free(ids0);
    try want.appendSlice(a, ids0);
    try want.append(a, 19);
    try want.appendNTimes(a, 18, 4);
    try want.append(a, 20);
    const lab1 = radio_vision.videoGroupSegment(&buf, 1, 3);
    const ids1 = try ByteTokenizer.encode(.{}, a, lab1);
    defer a.free(ids1);
    try want.appendSlice(a, ids1);
    try want.append(a, 19);
    try want.appendNTimes(a, 18, 2);
    try want.append(a, 20);
    try want.append(a, 2);
    try testing.expectEqualSlices(u32, want.items, out.ids);
    try testing.expectEqual(@as(u32, 1), out.media[0].start);
    try testing.expectEqual(@as(u32, @intCast(3 + lab0.len)), out.media[1].start);

    const bad = [_]bool{true};
    const bad_keeps = [_]?[]const bool{&bad};
    const only = [_]MediaItem{.{ .video = &clip }};
    try testing.expectError(error.InvalidVisionInput, expandMedia(a, &.{ 1, 19, 2 }, &only, &.{6}, &.{1}, mt, ByteTokenizer{}, &bad_keeps));
}

test "mediaRunTokens: the length check before the tower is the length the expansion produces" {
    // An image of 288x480 pools to 6x10 soft tokens; a 3-frame clip of the same size is three wrapped blocks.
    var config = eg2ProcessorConfig();
    const image = [_]chat_mod.ImageData{.{ .pixels = "", .width = 480, .height = 288 }};
    const clip = chat_mod.VideoData{ .pixels = "", .grid_t = 3, .grid_h = 18, .grid_w = 30 };
    const items = [_]MediaItem{ .{ .image = &image }, .{ .video = &clip } };
    const prompt = [_]u32{ 2, 7, 9, 7, 10, 1 };
    const out = try expandMediaPlaceholders(testing.allocator, &prompt, &items, &.{ 60, 180 }, &.{ 11, 12 }, &config);
    defer testing.allocator.free(out.ids);
    defer testing.allocator.free(out.media);
    try testing.expectEqual(out.ids.len, prompt.len - items.len + mediaRunTokens(&items, &config));
}
