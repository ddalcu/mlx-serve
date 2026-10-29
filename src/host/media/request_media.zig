//! image url parts, MediaBag, RequestMedia; appendAnthropicImageBlock

const std = @import("std");
const chat_mod = @import("../chat/chat.zig");
const log = @import("../../sdk/util/log.zig");
const testing = std.testing;

const placeholdersMatch = @import("placeholders.zig").placeholdersMatch;
const visionPreprocFromConfig = @import("prepare.zig").visionPreprocFromConfig;
const appendMiniCpmPieces = @import("../../plugins/qwen3_5/minicpm_media.zig").appendMiniCpmPieces;
const appendLfm2Tiles = @import("../../plugins/lfm2/media.zig").appendLfm2Tiles;
const decodeImageToPixels = @import("decode.zig").decodeImageToPixels;
const eg2ProcessorConfig = @import("prepare.zig").eg2ProcessorConfig;
const promptFor = @import("../api/embeddings.zig").promptFor;

pub fn parseImageUrlContent(allocator: std.mem.Allocator, url: []const u8, vp: chat_mod.VisionPreproc) ?chat_mod.ImageData {
    const sep = std.mem.indexOf(u8, url, ";base64,") orelse return null;
    const b64_data = url[sep + 8 ..];
    const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(b64_data) catch return null;
    const raw_buf = allocator.alloc(u8, decoded_size) catch return null;
    std.base64.standard.Decoder.decode(raw_buf, b64_data) catch {
        allocator.free(raw_buf);
        return null;
    };

    const mime = url[0..sep];
    if (std.mem.eql(u8, mime, "data:image/x-mlx-pixels")) {
        // Already preprocessed float32 CHW pixels — a GEMMA format. It carries
        // no patch grid and the source image cannot be recovered from it, so a
        // patch-grid tower has to refuse rather than hand a gridless buffer to
        // an encoder that has no SigLIP weights.
        if (vp.mode != .gemma) {
            log.warn("Dropping x-mlx-pixels image: {s} needs server-side preprocessing (send the encoded image instead)\n", .{@tagName(vp.mode)});
            allocator.free(raw_buf);
            return null;
        }
        const n_pixels = raw_buf.len / 4;
        const per_channel = n_pixels / 3;
        const side = std.math.sqrt(per_channel);
        if (side * side == per_channel and n_pixels == 3 * side * side) {
            return .{
                .pixels = raw_buf,
                .width = @intCast(side),
                .height = @intCast(side),
            };
        }
        allocator.free(raw_buf);
        return null;
    }

    if (std.mem.startsWith(u8, mime, "data:image/")) {
        // JPEG/PNG/WebP — decode + resize + convert to float32 CHW (Gemma) or
        // smart-resized merge-order pixel_values (Qwen3-VL).
        const img = decodeImageToPixels(allocator, raw_buf, vp);
        allocator.free(raw_buf);
        return img;
    }

    allocator.free(raw_buf);
    return null;
}

/// The one heap buffer every decoded media entry carries. `ImageData` and
/// `VideoData` spell it `pixels`, `AudioData` spells it `samples`; a fourth
/// modality that spells it something else fails to COMPILE here rather than
/// leaking quietly.
inline fn mediaBuffer(item: anytype) []const u8 {
    return if (@hasField(@TypeOf(item), "pixels")) item.pixels else item.samples;
}

/// One modality's decoded entries, grouped per message. Slots are INDICES, not
/// pointers: opening another slot may grow the backing list and move every
/// `ArrayList` header, while the buffers those headers point at stay put.
fn MediaBag(comptime T: type) type {
    return struct {
        const Self = @This();
        lists: std.ArrayList(std.ArrayList(T)) = .empty,

        fn open(self: *Self, allocator: std.mem.Allocator) !usize {
            try self.lists.append(allocator, .empty);
            return self.lists.items.len - 1;
        }

        fn at(self: *Self, slot: usize) *std.ArrayList(T) {
            return &self.lists.items[slot];
        }

        /// Borrowed view for a `chat_mod.Message` field. Null when nothing
        /// decoded, so a message with no media keeps a null field rather than
        /// an empty slice (every downstream reader tests the optional).
        fn slice(self: *Self, slot: usize) ?[]const T {
            const l = self.lists.items[slot];
            return if (l.items.len == 0) null else l.items;
        }

        fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            for (self.lists.items) |*l| {
                for (l.items) |item| allocator.free(mediaBuffer(item));
                l.deinit(allocator);
            }
            self.lists.deinit(allocator);
        }
    };
}

/// Every image / video / audio buffer decoded out of ONE request body, owned in
/// ONE place.
///
/// `chat_mod.Message.images/videos/audio` BORROW these slices — the handler
/// hands the message a view and keeps ownership here — so the `defer
/// messages.deinit(allocator)` every handler already has, which frees the
/// Message array and nothing it points at, can never be the whole story. Both
/// `handleChatCompletions` and the Anthropic `/v1/messages` handler used to
/// hand-roll a local list per message and `toOwnedSlice` it into the Message,
/// which leaked the full decoded CHW buffer of every image (megabytes each) on
/// the SUCCESS path and on every return after the parse loop.
///
/// Ownership is by PROVENANCE: a list only ever comes from `openImages` /
/// `openVideos` / `openAudio`, and is owned from its first append — so a `try`
/// between the decode and the message append cannot leak either, and a new
/// early return cannot leak by omission: a handler never declares a media list of its own.
pub const RequestMedia = struct {
    allocator: std.mem.Allocator,
    image_bag: MediaBag(chat_mod.ImageData) = .{},
    video_bag: MediaBag(chat_mod.VideoData) = .{},
    audio_bag: MediaBag(chat_mod.AudioData) = .{},

    pub fn init(allocator: std.mem.Allocator) RequestMedia {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *RequestMedia) void {
        self.image_bag.deinit(self.allocator);
        self.video_bag.deinit(self.allocator);
        self.audio_bag.deinit(self.allocator);
    }

    pub fn openImages(self: *RequestMedia) !usize {
        return self.image_bag.open(self.allocator);
    }
    pub fn images(self: *RequestMedia, slot: usize) *std.ArrayList(chat_mod.ImageData) {
        return self.image_bag.at(slot);
    }
    pub fn imagesSlice(self: *RequestMedia, slot: usize) ?[]const chat_mod.ImageData {
        return self.image_bag.slice(slot);
    }

    pub fn openVideos(self: *RequestMedia) !usize {
        return self.video_bag.open(self.allocator);
    }
    pub fn videos(self: *RequestMedia, slot: usize) *std.ArrayList(chat_mod.VideoData) {
        return self.video_bag.at(slot);
    }
    pub fn videosSlice(self: *RequestMedia, slot: usize) ?[]const chat_mod.VideoData {
        return self.video_bag.slice(slot);
    }

    pub fn openAudio(self: *RequestMedia) !usize {
        return self.audio_bag.open(self.allocator);
    }
    pub fn audio(self: *RequestMedia, slot: usize) *std.ArrayList(chat_mod.AudioData) {
        return self.audio_bag.at(slot);
    }
    pub fn audioSlice(self: *RequestMedia, slot: usize) ?[]const chat_mod.AudioData {
        return self.audio_bag.slice(slot);
    }
};

/// Decode one `image_url` into as many entries as the model's processor makes
/// of it — one per tower call. Only LFM2-VL ever yields more than one: past its
/// single-tile token budget it splits the source into a tile grid plus a
/// thumbnail, and each piece is encoded separately. Every other arch appends
/// exactly one entry. Returns false when nothing could be decoded (a remote
/// URL, bad base64, an unreadable payload) so the caller can refuse by name
/// instead of answering a prompt the image silently fell out of.
pub fn appendImageUrlContent(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(chat_mod.ImageData),
    url: []const u8,
    vp: chat_mod.VisionPreproc,
) bool {
    const before = list.items.len;
    if (vp.mode == .lfm2 and vp.tile_size > 0 and std.mem.startsWith(u8, url, "data:image/") and
        !std.mem.startsWith(u8, url, "data:image/x-mlx-pixels"))
    {
        appendLfm2Tiles(allocator, list, url, vp);
        return list.items.len > before;
    }
    if (vp.mode == .minicpm and std.mem.startsWith(u8, url, "data:image/") and
        !std.mem.startsWith(u8, url, "data:image/x-mlx-pixels"))
    {
        appendMiniCpmPieces(allocator, list, url, vp);
        return list.items.len > before;
    }
    if (parseImageUrlContent(allocator, url, vp)) |img| {
        list.append(allocator, img) catch allocator.free(img.pixels);
    }
    return list.items.len > before;
}

/// Decode an Anthropic `image` block (`source` base64 or url) into `list`.
/// True when `block` is not an image or decoded; false = undecodable.
pub fn appendAnthropicImageBlock(allocator: std.mem.Allocator, list: *std.ArrayList(chat_mod.ImageData), block: std.json.Value, vp: chat_mod.VisionPreproc) !bool {
    if (block != .object) return true;
    const btype = block.object.get("type") orelse return true;
    if (btype != .string or !std.mem.eql(u8, btype.string, "image")) return true;
    const src_val = block.object.get("source") orelse return false;
    if (src_val != .object) return false;
    const str = struct {
        fn get(o: std.json.ObjectMap, k: []const u8) []const u8 {
            const v = o.get(k) orelse return "";
            return if (v == .string) v.string else "";
        }
    }.get;
    const stype = str(src_val.object, "type");
    if (std.mem.eql(u8, stype, "url")) return appendImageUrlContent(allocator, list, str(src_val.object, "url"), vp);
    if (!std.mem.eql(u8, stype, "base64")) return false;
    const data = str(src_val.object, "data");
    if (data.len == 0) return false;
    const media_type = str(src_val.object, "media_type");
    const url = try std.fmt.allocPrint(allocator, "data:{s};base64,{s}", .{ if (media_type.len > 0) media_type else "image/png", data });
    defer allocator.free(url);
    return appendImageUrlContent(allocator, list, url, vp);
}

pub const AUDIO_DECODE_REJECT = "input_audio could not be decoded: send base64 `wav`, `pcm16` (24 kHz s16le mono) or `mlx_pcm_f32` (16 kHz f32le mono) in `data` with its `format`";
pub const IMAGE_DECODE_REJECT = "image could not be decoded: send a base64 data URL (data:image/jpeg|png|webp|gif|bmp;base64,...) with a readable payload; remote URLs are not fetched";

// ── tests ──

/// A solid-colour image as the data URL a client sends (`frames` entries are the same thing).
pub fn solidPngUrl(allocator: std.mem.Allocator, w: u32, h: u32, rgba: [4]u8) ![]u8 {
    const png = @import("../../sdk/media/png.zig");
    const px = try allocator.alloc(u8, @as(usize, w) * h * 4);
    defer allocator.free(px);
    for (0..@as(usize, w) * h) |i| @memcpy(px[i * 4 ..][0..4], &rgba);
    const bytes = try png.encodeRgba(allocator, px, w, h);
    defer allocator.free(bytes);
    const prefix = "data:image/png;base64,";
    const url = try allocator.alloc(u8, prefix.len + std.base64.standard.Encoder.calcSize(bytes.len));
    @memcpy(url[0..prefix.len], prefix);
    _ = std.base64.standard.Encoder.encode(url[prefix.len..], bytes);
    return url;
}

test "EmbeddingGemma 2 images keep their aspect ratio, fill the budget, and lose their alpha" {
    const a = testing.allocator;
    var config = eg2ProcessorConfig();
    const vp = visionPreprocFromConfig(&config);

    // 4x6 at 70 tokens is 288x480 (the processor's table), not a fixed square.
    const url = try solidPngUrl(a, 6, 4, .{ 255, 0, 0, 255 });
    defer a.free(url);
    const img = parseImageUrlContent(a, url, vp) orelse return error.Undecoded;
    defer a.free(img.pixels);
    try testing.expectEqual(@as(u32, 288), img.height);
    try testing.expectEqual(@as(u32, 480), img.width);
    try testing.expectEqual(@as(u32, 0), img.grid_h);
    try testing.expectEqual(@as(usize, 3 * 288 * 480 * 4), img.pixels.len);
    const px: []const f32 = @as([*]const f32, @ptrCast(@alignCast(img.pixels.ptr)))[0 .. 3 * 288 * 480];
    const plane = 288 * 480;
    try testing.expectEqual(@as(f32, 1.0), px[0]);
    try testing.expectEqual(@as(f32, 0.0), px[plane]);
    try testing.expectEqual(@as(f32, 0.0), px[2 * plane + plane - 1]);

    // A fully transparent red pixel is red: PIL's convert("RGB") keeps the colour channels.
    const clear = try solidPngUrl(a, 6, 4, .{ 255, 0, 0, 0 });
    defer a.free(clear);
    const seen = parseImageUrlContent(a, clear, vp) orelse return error.Undecoded;
    defer a.free(seen.pixels);
    const seen_px: []const f32 = @as([*]const f32, @ptrCast(@alignCast(seen.pixels.ptr)))[0 .. 3 * 288 * 480];
    try testing.expectEqual(@as(f32, 1.0), seen_px[0]);
    try testing.expectEqual(@as(f32, 0.0), seen_px[plane]);

    // Garbage and zero-size payloads are not images.
    try testing.expect(parseImageUrlContent(a, "data:image/png;base64,AAAA", vp) == null);
}

test "embeddingPrompt: a message's parts are one sequence, in order, system turns first" {
    const a = testing.allocator;
    var config = eg2ProcessorConfig();
    config.image_token_id = 1000;
    config.video_token_id = 1001;
    var media = RequestMedia.init(a);
    defer media.deinit();
    const img = try solidPngUrl(a, 6, 4, .{ 10, 20, 30, 255 });
    defer a.free(img);
    const body = try std.fmt.allocPrint(a,
        \\[{{"role":"user","content":[
        \\   {{"type":"text","text":"ab"}},
        \\   {{"type":"image_url","image_url":{{"url":"{s}"}}}},
        \\   {{"type":"video_url","video_url":{{"frames":["{s}","{s}"]}}}},
        \\   {{"type":"text","text":"cd"}}]}},
        \\ {{"role":"system","content":"S"}},
        \\ {{"role":"user","content":"e"}}]
    , .{ img, img, img });
    defer a.free(body);

    var result = try promptFor(a, &config, body, &media, true);
    defer result.deinit(a);
    const prompt = switch (result) {
        .ok => |p| p,
        .reject => |why| {
            std.debug.print("rejected: {s}\n", .{why});
            return error.Rejected;
        },
    };
    // The system turn leads (the model's own template renders it first); no separator is invented anywhere.
    try testing.expectEqualSlices(u32, &.{ 'S', 'a', 'b', 1000, 1001, 'c', 'd', 'e' }, prompt.ids);
    try testing.expectEqual(@as(usize, 2), prompt.items.len);
    try testing.expectEqual(.image, std.meta.activeTag(prompt.items[0]));
    try testing.expectEqual(@as(u32, 288), prompt.items[0].image[0].height);
    try testing.expectEqual(.video, std.meta.activeTag(prompt.items[1]));
    try testing.expectEqual(@as(u32, 2), prompt.items[1].video.grid_t);
    // …and every item has the one placeholder `expandMediaPlaceholders` will look for.
    try testing.expect(placeholdersMatch(prompt.ids, prompt.items, &config));
}

test "embeddingPrompt: a video before an image stays before it" {
    const a = testing.allocator;
    var config = eg2ProcessorConfig();
    config.image_token_id = 1000;
    config.video_token_id = 1001;
    var media = RequestMedia.init(a);
    defer media.deinit();
    const img = try solidPngUrl(a, 6, 4, .{ 10, 20, 30, 255 });
    defer a.free(img);
    const body = try std.fmt.allocPrint(a,
        \\[{{"role":"user","content":[
        \\   {{"type":"video_url","video_url":{{"frames":["{s}"]}}}},
        \\   {{"type":"image_url","image_url":{{"url":"{s}"}}}}]}}]
    , .{ img, img });
    defer a.free(body);
    var result = try promptFor(a, &config, body, &media, true);
    defer result.deinit(a);
    const prompt = result.ok;
    try testing.expectEqualSlices(u32, &.{ 1001, 1000 }, prompt.ids);
    try testing.expectEqual(.video, std.meta.activeTag(prompt.items[0]));
    try testing.expectEqual(.image, std.meta.activeTag(prompt.items[1]));
}

test "embeddingPrompt: what cannot be embedded is refused by name" {
    const a = testing.allocator;
    var config = eg2ProcessorConfig();
    config.image_token_id = 1000;
    config.video_token_id = 1001;
    const img = try solidPngUrl(a, 6, 4, .{ 10, 20, 30, 255 });
    defer a.free(img);
    const with_image = try std.fmt.allocPrint(a, "[{{\"role\":\"user\",\"content\":[{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"{s}\"}}}}]}}]", .{img});
    defer a.free(with_image);
    const cases = [_]struct { body: []const u8, can_see: bool, why: []const u8 }{
        .{ .body = with_image, .can_see = false, .why = "vision encoder" },
        .{ .body = "[{\"role\":\"user\",\"content\":[{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"AAAA\"}}]}]", .can_see = true, .why = "audio" },
        .{ .body = "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AAAA\"}}]}]", .can_see = true, .why = "could not be decoded" },
        .{ .body = "[{\"role\":\"user\",\"content\":[{\"type\":\"video_url\",\"video_url\":{\"frames\":[]}}]}]", .can_see = true, .why = "frames" },
        .{ .body = "[{\"role\":\"user\",\"content\":[{\"type\":\"video_url\",\"video_url\":{\"frames\":[\"data:image/png;base64,AAAA\"]}}]}]", .can_see = true, .why = "frames" },
        .{ .body = "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"\"}]}]", .can_see = true, .why = "nothing to embed" },
        .{ .body = "[{\"role\":\"user\",\"content\":[{\"type\":\"file\"}]}]", .can_see = true, .why = "text, image_url or video_url" },
        .{ .body = "[{\"role\":\"user\",\"content\":7}]", .can_see = true, .why = "content" },
        .{ .body = "[7]", .can_see = true, .why = "objects" },
    };
    for (cases) |c| {
        var media = RequestMedia.init(a);
        defer media.deinit();
        var result = try promptFor(a, &config, c.body, &media, c.can_see);
        defer result.deinit(a);
        switch (result) {
            .ok => {
                std.debug.print("accepted: {s}\n", .{c.body});
                return error.Accepted;
            },
            .reject => |why| testing.expect(std.mem.indexOf(u8, why, c.why) != null) catch |e| {
                std.debug.print("rejected with '{s}', wanted '{s}'\n", .{ why, c.why });
                return e;
            },
        }
    }
}
