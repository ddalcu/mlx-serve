//! MediaPieces, encodeMediaItems; MultimodalPrompt; prepareMultimodalItems, parseAudioInput

const std = @import("std");
const wav = @import("../../sdk/media/wav.zig");
const mlx = @import("../../mlx/ffi.zig");
const chat_mod = @import("../chat/chat.zig");
const model_mod = @import("../../sdk/config/model.zig");
const radio_vision = @import("../../plugins/nemotron_h/radio_vision.zig");
const vision_mod = @import("../../abi/mlx_handles.zig");
const log = @import("../../sdk/util/log.zig");
const MediaSpan = @import("../../sdk/kv/media_span.zig").MediaSpan;
const scheduler_mod = @import("../gpu_thread/requests.zig");
const model_registry_mod = @import("../registry/registry.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const testing = std.testing;
const http_server = @import("../http/server.zig");

const loadedSupportsVideo = @import("../api/models.zig").loadedSupportsVideo;
const MediaItem = @import("placeholders.zig").MediaItem;
const collectMediaItems = @import("placeholders.zig").collectMediaItems;
const placeholderKindOf = @import("placeholders.zig").placeholderKindOf;
const MEDIA_PLACEHOLDER_REJECT = @import("placeholders.zig").MEDIA_PLACEHOLDER_REJECT;
const placeholdersMatchOf = @import("placeholders.zig").placeholdersMatchOf;
const expandMedia = @import("placeholders.zig").expandMedia;
const computeQwenMrope = @import("qwen_mrope.zig").computeQwenMrope;
const computeMiniCpmMrope = @import("../../plugins/qwen3_5/minicpm_media.zig").computeMiniCpmMrope;
const radioMediaTokens = @import("../../plugins/nemotron_h/media.zig").radioMediaTokens;
const ByteTokenizer = @import("placeholders.zig").ByteTokenizer;
const expandRuns = @import("placeholders.zig").expandRuns;

/// The encoder inputs for a request's items: one piece per tower call (an
/// LFM2-VL tile, else the whole item), keyed by its pixels/PCM; an item's key
/// folds its pieces'.
const MediaPieces = struct {
    pieces: []scheduler_mod.VisionItem,
    piece_keys: []u64,
    item_keys: []u64,

    fn init(allocator: std.mem.Allocator, items: []const MediaItem) !MediaPieces {
        var pieces = std.ArrayList(scheduler_mod.VisionItem).empty;
        defer pieces.deinit(allocator);
        for (items) |item| switch (item) {
            .image => |imgs| for (imgs) |img| try pieces.append(allocator, .{ .image = .{
                .pixels = img.pixels,
                .width = img.width,
                .height = img.height,
                .grid_h = img.grid_h,
                .grid_w = img.grid_w,
            } }),
            .video => |vid| try pieces.append(allocator, .{ .video = .{
                .pixels = vid.pixels,
                .grid_t = vid.grid_t,
                .grid_h = vid.grid_h,
                .grid_w = vid.grid_w,
                .fps = vid.fps,
            } }),
            .audio => |au| try pieces.append(allocator, .{ .audio = au.samples }),
        };
        const piece_keys = try allocator.alloc(u64, pieces.items.len);
        errdefer allocator.free(piece_keys);
        for (pieces.items, piece_keys) |piece, *k| k.* = pieceKey(piece);
        const item_keys = try allocator.alloc(u64, items.len);
        errdefer allocator.free(item_keys);
        var p: usize = 0;
        for (items, item_keys) |item, *k| {
            const n = piecesOf(item);
            var h = std.hash.Wyhash.init(0x5ec0de);
            h.update(std.mem.sliceAsBytes(piece_keys[p .. p + n]));
            k.* = h.final() | 1;
            p += n;
        }
        return .{ .pieces = try pieces.toOwnedSlice(allocator), .piece_keys = piece_keys, .item_keys = item_keys };
    }

    fn deinit(self: *MediaPieces, allocator: std.mem.Allocator) void {
        allocator.free(self.pieces);
        allocator.free(self.piece_keys);
        allocator.free(self.item_keys);
    }
};

fn piecesOf(item: MediaItem) usize {
    return if (item == .image) item.image.len else 1;
}

fn pieceKey(piece: scheduler_mod.VisionItem) u64 {
    var h = std.hash.Wyhash.init(@backingInt(std.meta.activeTag(piece)));
    switch (piece) {
        .image => |im| {
            h.update(std.mem.asBytes(&[_]u32{ im.width, im.height, im.grid_h, im.grid_w }));
            h.update(im.pixels);
        },
        .video => |vd| {
            h.update(std.mem.asBytes(&[_]u32{ vd.grid_t, vd.grid_h, vd.grid_w, @bitCast(vd.fps) }));
            h.update(vd.pixels);
        },
        .audio => |pcm| h.update(pcm),
    }
    return h.final();
}

const EncodedMedia = struct {
    rows: mlx.mlx_array,
    /// Parallel to encoder pieces. Empty when nothing was pruned. A mask is the
    /// full pre-EVS length. Owned by the caller when `len != 0`.
    item_keep: []?[]bool = &.{},
    /// A plugin tower's placeholder run per piece (`freeRuns`); empty for the engine's own towers.
    runs: [][]u32 = &.{},
};

fn freeRuns(allocator: std.mem.Allocator, runs: [][]u32) void {
    for (runs) |r| allocator.free(r);
    if (runs.len != 0) allocator.free(runs);
}

fn freeKeepMasks(allocator: std.mem.Allocator, keeps: []?[]bool) void {
    for (keeps) |m| if (m) |s| allocator.free(s);
    if (keeps.len != 0) allocator.free(keeps);
}

/// Encode `mp` on the inference thread (the sole mlx caller) and return every
/// item's soft tokens concatenated in prompt order; `item_tokens[k]` receives
/// item k's row count. RADIO video rows are already the EVS keep.
fn encodeMediaItems(allocator: std.mem.Allocator, lm: *LoadedModel, items: []const MediaItem, mp: MediaPieces, item_tokens: []usize, plugin: bool) !EncodedMedia {
    const sch = http_server.launch().global_scheduler orelse return error.NoScheduler;
    const piece_tokens = try allocator.alloc(usize, mp.pieces.len);
    defer allocator.free(piece_tokens);
    const runs: [][]u32 = if (plugin) try allocator.alloc([]u32, mp.pieces.len) else &.{};
    for (runs) |*r| r.* = &.{};
    errdefer freeRuns(allocator, runs);
    var req = scheduler_mod.VisionEncodeRequest{
        .model = lm,
        .items = mp.pieces,
        .item_keys = mp.piece_keys,
        .item_tokens = piece_tokens,
        .item_runs = runs,
        .allocator = allocator,
    };
    const arr = sch.encodeVision(&req) catch |err| {
        if (req.item_keep) |keeps| freeKeepMasks(allocator, keeps);
        if (req.error_name) |e| {
            log.err("Vision encode (via scheduler) failed: {s}\n", .{e});
            allocator.free(e);
        }
        return err;
    };
    var p: usize = 0;
    for (items, item_tokens) |item, *n| {
        const count = piecesOf(item);
        n.* = 0;
        for (piece_tokens[p .. p + count]) |t| n.* += t;
        p += count;
    }
    return .{ .rows = arr, .item_keep = req.item_keep orelse &.{}, .runs = runs };
}

/// Qwen3-VL interleaved M-RoPE position-id table for a request. `pos` (owned) is
/// the flat [3 × total] i32 table threaded to the slot; null for non-Qwen / no
/// images. Pass-through bundle so sub-handlers thread one value, not three.
pub const MropeData = struct {
    pos: ?[]const i32 = null,
    total: usize = 0,
    delta: i32 = 0,
};

/// One request's media, ready for a slot: the soft tokens, the per-item
/// prefix-cache key and the M-RoPE table. Owns all three until moved out.
pub const MultimodalPrompt = struct {
    embeddings: ?mlx.mlx_array = null,
    media: []MediaSpan = &.{},
    mrope: MropeData = .{},
    /// Soft-token rows in `embeddings`.
    rows: usize = 0,

    /// Bytes of the concatenated soft tokens (bf16 at the text width).
    pub fn bytes(self: MultimodalPrompt, config: *const model_mod.ModelConfig) u64 {
        return @as(u64, self.rows) * config.hidden_size * 2;
    }

    pub fn deinit(self: *MultimodalPrompt, allocator: std.mem.Allocator) void {
        if (self.embeddings) |e| _ = mlx.mlx_array_free(e);
        allocator.free(self.media);
        if (self.mrope.pos) |p| allocator.free(p);
        self.* = .{};
    }
};

/// Encode every media item in `messages` and expand the placeholders the
/// template rendered for them; `prompt_ids` is replaced by the expanded prompt.
pub fn prepareMultimodalPrompt(allocator: std.mem.Allocator, lm: *LoadedModel, config: *const model_mod.ModelConfig, messages: []const chat_mod.Message, prompt_ids: *[]u32) !MultimodalPrompt {
    const items = try collectMediaItems(allocator, messages);
    defer allocator.free(items);
    return prepareMultimodalItems(allocator, lm, config, items, prompt_ids);
}

/// `prepareMultimodalPrompt` for items a caller already put in placeholder order.
pub fn prepareMultimodalItems(allocator: std.mem.Allocator, lm: *LoadedModel, config: *const model_mod.ModelConfig, items: []const MediaItem, prompt_ids: *[]u32) !MultimodalPrompt {
    if (items.len == 0) return .{};
    var videos: usize = 0;
    for (items) |it| videos += @intFromBool(it == .video);
    if (videos > 0 and !loadedSupportsVideo(lm)) return error.VideoUnsupported;
    const mt = radioMediaTokens(lm, config);
    if (config.radio_vision and videos > 0 and (mt.video_sentinel == 0 or mt.img_end == 0 or mt.video_sentinel == config.image_token_id)) {
        log.warn("  media: RADIO video needs distinct <img> and </img> special tokens\n", .{});
        return error.MediaPlaceholderMismatch;
    }
    // Before the encode: a mismatch is the client's, and cheap to refuse.
    if (!placeholdersMatchOf(prompt_ids.*, items, mt)) {
        var rendered: usize = 0;
        for (prompt_ids.*) |t| rendered += @intFromBool(placeholderKindOf(t, mt) != null);
        log.warn("  media: the prompt renders {d} placeholder(s) for {d} item(s); a `jinja render failed` line above means the generic fallback rendered none\n", .{ rendered, items.len });
        return error.MediaPlaceholderMismatch;
    }
    const item_tokens = try allocator.alloc(usize, items.len);
    defer allocator.free(item_tokens);

    var mp = try MediaPieces.init(allocator, items);
    defer mp.deinit(allocator);

    var out: MultimodalPrompt = .{};
    errdefer out.deinit(allocator);
    // A plugin's item is one encoder piece: its run is the piece's.
    if (config.plugin_media and mp.pieces.len != items.len) return error.MediaPlaceholderMismatch;
    const encoded = try encodeMediaItems(allocator, lm, items, mp, item_tokens, config.plugin_media);
    defer freeKeepMasks(allocator, encoded.item_keep);
    defer freeRuns(allocator, encoded.runs);
    out.embeddings = encoded.rows;
    const item_keep = try allocator.alloc(?[]const bool, items.len);
    defer allocator.free(item_keep);
    var piece_at: usize = 0;
    for (items, item_keep) |item, *slot| {
        const n = piecesOf(item);
        slot.* = if (item == .video and piece_at < encoded.item_keep.len) encoded.item_keep[piece_at] else null;
        piece_at += n;
    }
    const expanded = if (encoded.runs.len > 0)
        try expandRuns(allocator, prompt_ids.*, items, encoded.runs, mp.item_keys, mt)
    else if (lm.tokenizer) |tok|
        try expandMedia(allocator, prompt_ids.*, items, item_tokens, mp.item_keys, mt, tok, item_keep)
    else
        try expandMedia(allocator, prompt_ids.*, items, item_tokens, mp.item_keys, mt, ByteTokenizer{}, item_keep);
    out.media = expanded.media;
    // The rows the prefix-cache tiers key on this item's pixels; a restore past one needs the same key.
    for (out.media, 0..) |m, k| log.debug("  media: item {d} at row {d} (key {x:0>16})\n", .{ k, m.start, m.key });
    var soft = std.enums.EnumArray(std.meta.Tag(MediaItem), usize).initFill(0);
    for (items, item_tokens) |item, n| {
        soft.getPtr(std.meta.activeTag(item)).* += n;
        out.rows += n;
    }
    log.info("  Multimodal: {d} item(s), {d} encoder piece(s), {d} image + {d} video + {d} audio soft tokens; prompt {d} -> {d} tokens\n", .{
        items.len, mp.pieces.len, soft.get(.image), soft.get(.video), soft.get(.audio), prompt_ids.len, expanded.ids.len,
    });
    allocator.free(prompt_ids.*);
    prompt_ids.* = expanded.ids;
    out.mrope = (if (config.minicpm_vision) computeMiniCpmMrope(allocator, prompt_ids.*, items, config) else computeQwenMrope(allocator, prompt_ids.*, items, config)) catch |err| blk: {
        log.warn("M-RoPE table failed ({s}); falling back to scalar RoPE\n", .{@errorName(err)});
        break :blk .{};
    };
    return out;
}

/// The error response for a failed `prepareMultimodalPrompt`.
pub fn multimodalFailure(err: anyerror) struct { status: []const u8, code: u16, kind: []const u8, message: []const u8 } {
    if (err == error.VideoUnsupported) return .{ .status = "400 Bad Request", .code = 400, .kind = "invalid_request_error", .message = "video input is not supported by this model's loaded vision tower" };
    if (err == error.MediaPlaceholderMismatch) return .{ .status = "400 Bad Request", .code = 400, .kind = "invalid_request_error", .message = MEDIA_PLACEHOLDER_REJECT };
    return .{ .status = "500 Internal Server Error", .code = 500, .kind = "server_error", .message = "media encoding failed" };
}

/// An `input_audio` part as 16 kHz mono f32-LE: OpenAI's `wav` (any rate and channel count, mixed down and linearly
/// resampled) and `pcm16` (24 kHz s16le mono), or our raw `mlx_pcm_f32` (no `format` reads as it). Null for another
/// format or an undecodable payload (the caller's named 400).
pub fn parseAudioInput(allocator: std.mem.Allocator, data: []const u8, format: ?[]const u8) ?chat_mod.AudioData {
    const f = format orelse return parseAudioContent(allocator, data);
    if (std.mem.eql(u8, f, "mlx_pcm_f32")) return parseAudioContent(allocator, data);
    const wav_in = std.mem.eql(u8, f, "wav");
    if (!wav_in and !std.mem.eql(u8, f, "pcm16")) return null;
    const b64 = if (std.mem.indexOf(u8, data, ";base64,")) |sep| data[sep + 8 ..] else data;
    const size = std.base64.standard.Decoder.calcSizeForSlice(b64) catch return null;
    const bytes = allocator.alloc(u8, size) catch return null;
    defer allocator.free(bytes);
    std.base64.standard.Decoder.decode(bytes, b64) catch return null;
    var mono: []f32 = undefined;
    var rate: u32 = 24000;
    if (wav_in) {
        const d = wav.decode(allocator, bytes) catch return null;
        defer allocator.free(d.pcm);
        const ch: usize = d.channels;
        if (ch == 0 or d.pcm.len < ch) return null;
        mono = allocator.alloc(f32, d.pcm.len / ch) catch return null;
        for (mono, 0..) |*m, i| {
            var sum: f32 = 0;
            for (d.pcm[i * ch ..][0..ch]) |x| sum += x;
            m.* = sum / @as(f32, @floatFromInt(ch));
        }
        rate = d.sample_rate;
    } else {
        if (bytes.len < 2) return null;
        mono = allocator.alloc(f32, bytes.len / 2) catch return null;
        for (mono, 0..) |*m, i| m.* = @as(f32, @floatFromInt(std.mem.readInt(i16, bytes[i * 2 ..][0..2], .little))) / 32768.0;
    }
    defer allocator.free(mono);
    const at16 = wav.resampleLinear(allocator, mono, 1, rate, 16000) catch return null;
    defer allocator.free(at16);
    // `AudioData.samples` is freed as bytes.
    return .{ .samples = allocator.dupe(u8, std.mem.sliceAsBytes(at16)) catch return null };
}

/// Decode an `input_audio.data` payload into raw float32-LE PCM samples for the
/// Gemma 4 12B unified audio embedder. Accepts a bare base64 string or a
/// `data:audio/...;base64,...` URL. The decoded bytes are interpreted as
/// little-endian float32 mono samples at 16 kHz (the client resamples). Returns
/// null on decode failure or a non-multiple-of-4 byte length.
pub fn parseAudioContent(allocator: std.mem.Allocator, data: []const u8) ?chat_mod.AudioData {
    const b64 = if (std.mem.indexOf(u8, data, ";base64,")) |sep| data[sep + 8 ..] else data;
    const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(b64) catch return null;
    if (decoded_size == 0 or decoded_size % 4 != 0) return null;
    const raw_buf = allocator.alloc(u8, decoded_size) catch return null;
    std.base64.standard.Decoder.decode(raw_buf, b64) catch {
        allocator.free(raw_buf);
        return null;
    };
    return .{ .samples = raw_buf };
}

// ── tests ──

test "parseAudioInput: OpenAI's wav and pcm16 become 16 kHz mono f32, mlx_pcm_f32 passes through, others refuse" {
    const a = testing.allocator;
    // A 2-channel 8 kHz 16-bit WAV of 4 frames: L = 0.5, R = -0.25 (mono 0.125), resampled 2x to 16 kHz.
    var wav_buf: [44 + 16]u8 = undefined;
    @memcpy(wav_buf[0..4], "RIFF");
    std.mem.writeInt(u32, wav_buf[4..8], 36 + 16, .little);
    @memcpy(wav_buf[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, wav_buf[16..20], 16, .little);
    std.mem.writeInt(u16, wav_buf[20..22], 1, .little);
    std.mem.writeInt(u16, wav_buf[22..24], 2, .little);
    std.mem.writeInt(u32, wav_buf[24..28], 8000, .little);
    std.mem.writeInt(u32, wav_buf[28..32], 8000 * 4, .little);
    std.mem.writeInt(u16, wav_buf[32..34], 4, .little);
    std.mem.writeInt(u16, wav_buf[34..36], 16, .little);
    @memcpy(wav_buf[36..40], "data");
    std.mem.writeInt(u32, wav_buf[40..44], 16, .little);
    for (0..4) |i| {
        std.mem.writeInt(i16, wav_buf[44 + i * 4 ..][0..2], 16384, .little);
        std.mem.writeInt(i16, wav_buf[46 + i * 4 ..][0..2], -8192, .little);
    }
    var b64: [128]u8 = undefined;
    const wav_b64 = std.base64.standard.Encoder.encode(&b64, &wav_buf);
    const w = parseAudioInput(a, wav_b64, "wav") orelse return error.TestUnexpectedResult;
    defer a.free(w.samples);
    const ws = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(w.samples)));
    try testing.expectEqual(@as(usize, 8), ws.len);
    for (ws) |x| try testing.expectApproxEqAbs(@as(f32, 0.125), x, 1e-3);

    // pcm16 at 24 kHz: 3 samples -> 2 at 16 kHz.
    var pcm: [6]u8 = undefined;
    for (0..3) |i| std.mem.writeInt(i16, pcm[i * 2 ..][0..2], -16384, .little);
    const pcm_b64 = std.base64.standard.Encoder.encode(&b64, &pcm);
    const p16 = parseAudioInput(a, pcm_b64, "pcm16") orelse return error.TestUnexpectedResult;
    defer a.free(p16.samples);
    try testing.expectEqual(@as(usize, 2 * 4), p16.samples.len);

    const raw = parseAudioInput(a, "AAAAAAAAgD8=", "mlx_pcm_f32") orelse return error.TestUnexpectedResult;
    defer a.free(raw.samples);
    try testing.expectEqual(@as(usize, 8), raw.samples.len);
    try testing.expect(parseAudioInput(a, wav_b64, "mp3") == null);
    try testing.expect(parseAudioInput(a, "AAAA", "wav") == null);
}

test "RADIO video availability rejects a forced clip before encoding with a named 400" {
    const a = testing.allocator;
    var radio: radio_vision.RadioVision = undefined;
    radio.video_w = .{ .ctx = null };
    var encoder: vision_mod.VisionEncoder = undefined;
    encoder.radio = radio;
    encoder.qwen_audio = null;
    encoder.plugin_media = null;
    var entry: LoadedModel = undefined;
    entry.vision_encoder = &encoder;
    entry.tokenizer = null;
    const config = model_mod.ModelConfig{ .radio_vision = true, .image_token_id = 18 };
    const clip = chat_mod.VideoData{ .pixels = "", .grid_t = 3, .grid_h = 64, .grid_w = 64 };
    const items = [_]MediaItem{.{ .video = &clip }};
    var ids = try a.dupe(u32, &.{19});
    defer a.free(ids);
    try testing.expectError(error.VideoUnsupported, prepareMultimodalItems(a, &entry, &config, &items, &ids));
    const failure = multimodalFailure(error.VideoUnsupported);
    try testing.expectEqual(@as(u16, 400), failure.code);
    try testing.expect(std.mem.indexOf(u8, failure.message, "video") != null);
}

test "parseAudioContent decodes base64 float32 PCM and rejects bad lengths" {
    // 2 float32 samples = 8 bytes. base64 of 8 zero bytes = "AAAAAAAAAAA=".
    const eight_zeros = "AAAAAAAAAAA=";
    const aud = parseAudioContent(testing.allocator, eight_zeros) orelse return error.TestUnexpectedNull;
    defer testing.allocator.free(aud.samples);
    try testing.expectEqual(@as(usize, 8), aud.samples.len);

    // A data-URL prefix is tolerated.
    const with_prefix = "data:audio/x-mlx-pcm;base64," ++ eight_zeros;
    const aud2 = parseAudioContent(testing.allocator, with_prefix) orelse return error.TestUnexpectedNull;
    defer testing.allocator.free(aud2.samples);
    try testing.expectEqual(@as(usize, 8), aud2.samples.len);

    // 6 decoded bytes is not a whole number of float32s → rejected.
    try testing.expect(parseAudioContent(testing.allocator, "AAAAAAAA") == null);
}
