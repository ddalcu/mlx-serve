//! Chat request types: tool calls, image data, the vision preprocessing selector, messages,
//! the chat config, and the system-turn fold.

const std = @import("std");

pub const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    arguments: []const u8, // JSON string
};

/// Raw image pixel data for vision encoder (float32, CHW format).
pub const ImageData = struct {
    pixels: []const u8, // Gemma: CHW float32 [3*H*W*4]. Qwen: merge-order patches [N*1536*4].
    width: u32,
    height: u32,
    // Qwen3-VL only: full patch grid (h=resized_H/patch, w=resized_W/patch). 0 ⇒
    // Gemma CHW layout. When >0, `pixels` holds the processor's merge-order
    // pixel_values and the encoder is QwenVision (see src/sdk/vision/qwen_vision.zig).
    grid_h: u32 = 0,
    grid_w: u32 = 0,
    /// LFM2-VL tiling: a source image past the single-tile budget is split into
    /// a `tile_rows` x `tile_cols` grid plus a thumbnail, and EACH piece is its
    /// own entry here — separate patch grid, separate encoder call. `tile_index`
    /// runs row-major over the tiles, and `tile_index == rows*cols` is the
    /// thumbnail. All three stay 0 for an untiled image, which is every other
    /// tower we serve.
    tile_rows: u16 = 0,
    tile_cols: u16 = 0,
    tile_index: u16 = 0,
};

/// Per-request image preprocessing selector, derived from the loaded model's
/// config. Threaded into `parseImageUrlContent`/`decodeImageToPixels` so decode
/// stays race-safe (no global state) under `--max-concurrent ≥ 2`.
pub const VisionPreproc = struct {
    /// Clef's Pillow RGB conversion discards alpha instead of compositing it.
    composite_alpha: bool = true,
    /// Which processor produced `ImageData.pixels`: Gemma's fixed CHW square,
    /// `gemma_budget` (Gemma 4's own processor, as EmbeddingGemma 2 uses it: a CHW image of any
    /// aspect ratio sized to `max_tokens` soft tokens), or one of the patch-grid towers
    /// (each with its own resize + patch order).
    mode: enum { gemma, qwen, muse, lfm2, glm, minicpm, gemma_budget, radio, dsv41, plugin, plugin_rgb } = .gemma,
    patch: u32 = 16,
    tps: u32 = 2,
    merge: u32 = 2,
    min_pixels: u32 = 0,
    max_pixels: u32 = 0,
    /// muse/lfm2/glm: the resize cap is on MERGED tokens, not pixels.
    max_tokens: u32 = 0,
    /// lfm2/glm: the budget has a FLOOR too — a small image is upscaled to it.
    min_tokens: u32 = 0,
    /// minicpm: most detail crops a source is sliced into.
    max_slices: u32 = 0,
    /// lfm2 tiling. A source past `max_tokens * pixels_tolerance` is split into
    /// a grid of `tile_size` tiles (plus a thumbnail) and encoded piece by
    /// piece. `tile_size == 0` disables splitting, which is every other tower.
    tile_size: u32 = 0,
    min_tiles: u32 = 0,
    max_tiles: u32 = 0,
    use_thumbnail: bool = false,
    pixels_tolerance: f32 = 0,
    /// gemma_budget video: the most frames kept from a client's list.
    max_frames: u32 = 0,
    /// dsv41: an image wider than this many heights is stretched into the capped canvas (0 = no cap).
    max_wh_ratio: u32 = 0,
    /// plugin: the served outside plugin (`plugin_config.Binding`) whose media hooks prepare the image.
    plugin: ?*anyopaque = null,
};

/// Raw mono 16 kHz audio samples for the Gemma 4 12B unified audio embedder.
/// Bytes are little-endian float32 PCM; the encoder frames them into 640-sample
/// tokens (40 ms @ 16 kHz) and projects each straight into language-model space.
pub const AudioData = struct {
    samples: []const u8, // Raw float32-LE bytes (n_samples * 4)
};

/// Qwen3-VL stores pre-patchified pixel values for every temporal-patch group.
/// `grid_t` is that group count and `grid_h` / `grid_w` are the shared patch grid.
/// RADIO stores concatenated float32 CHW frames. `grid_t` is the raw frame count
/// (the encoder repeats a short last tubelet) and `grid_h` / `grid_w` are the
/// shared pixel size.
pub const VideoData = struct {
    pixels: []const u8,
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
    /// A plugin tower's clip (`.plugin_rgb`): `pixels` is `grid_t` RGB8 frames of `grid_w x grid_h`, sampled at `fps`
    /// (0 = the client did not say).
    fps: f32 = 0,
};

/// OpenAI spells the system turn `developer` for reasoning models (pi does
/// when `supportsDeveloperRole` is unset). No template we serve tells the two
/// apart and Qwen's raises on the unknown role, which is a SILENT fallback.
pub fn canonicalRole(role: []const u8) []const u8 {
    return if (std.mem.eql(u8, role, "developer")) "system" else role;
}

/// Join system messages into one leading turn. With `leading_only`, later
/// turns keep their position. Caller owns the joined buffer, if any.
pub fn foldSystemMessages(allocator: std.mem.Allocator, messages: *std.ArrayList(Message), leading_only: bool) !?[]const u8 {
    var end = messages.items.len;
    if (leading_only) {
        end = 0;
        while (end < messages.items.len and std.mem.eql(u8, messages.items[end].role, "system")) : (end += 1) {}
    }
    var extra: usize = 0;
    for (messages.items[@min(end, 1)..end]) |m| {
        if (std.mem.eql(u8, m.role, "system")) extra += 1;
    }
    if (extra == 0) return null;
    const lead_is_system = messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system");
    var joined = std.ArrayList(u8).empty;
    errdefer joined.deinit(allocator);
    if (lead_is_system) try joined.appendSlice(allocator, messages.items[0].content);
    var i: usize = if (lead_is_system) 1 else 0;
    while (i < end) {
        if (!std.mem.eql(u8, messages.items[i].role, "system")) {
            i += 1;
            continue;
        }
        if (joined.items.len > 0) try joined.appendSlice(allocator, "\n\n");
        try joined.appendSlice(allocator, messages.items[i].content);
        _ = messages.orderedRemove(i);
        end -= 1;
    }
    const text = try joined.toOwnedSlice(allocator);
    if (lead_is_system) {
        messages.items[0].content = text;
    } else {
        try messages.insert(allocator, 0, .{ .role = "system", .content = text });
    }
    return text;
}

pub const Message = struct {
    role: []const u8,
    content: []const u8,
    tool_calls: ?[]const ToolCall = null,
    tool_call_id: ?[]const u8 = null,
    images: ?[]const ImageData = null, // Preprocessed image data for vision
    videos: ?[]const VideoData = null, // Preprocessed video data for vision
    audio: ?[]const AudioData = null, // Raw PCM for the unified audio embedder
    /// Reasoning the client round-trips on assistant HISTORY messages
    /// (`reasoning_content`/`reasoning` on chat completions, `thinking`
    /// blocks on /v1/messages). Templates that persist reasoning across
    /// turns (laguna) read it; templates that strip history reasoning
    /// (Qwen, Gemma) never reference the field and render unchanged.
    reasoning_content: ?[]const u8 = null,
};

/// Chat template configuration loaded from tokenizer_config.json.
pub const ChatConfig = struct {
    chat_template: []const u8,
    bos_token: ?[]const u8,
    eos_token: ?[]const u8,
    add_bos_token: bool,
    allocator: std.mem.Allocator,
    /// Template variables as a JSON object: the model's `chat_template_kwargs`
    /// (`model-settings.json`), with a request's own merged over them per request.
    chat_template_kwargs: ?[]const u8 = null,
    /// `templateRendersToolTurn` for this template, probed ONCE at load (a probe render
    /// per request is a second parse of a multi-KB template); null = unknown, probe.
    renders_tool_turn: ?bool = null,
    /// A model-specific video atom emitted only for an attached media part.
    video_placeholder: ?[]const u8 = null,
    /// A template that reads `content` as a STRING (DeepSeek-V4.1): each attached image becomes this
    /// atom, joined to the text by blank lines, the way the release's encoder renders image blocks.
    image_placeholder: ?[]const u8 = null,

    pub fn deinit(self: *ChatConfig) void {
        self.allocator.free(self.chat_template);
        if (self.chat_template_kwargs) |k| self.allocator.free(k);
        if (self.bos_token) |t| self.allocator.free(t);
        if (self.eos_token) |t| self.allocator.free(t);
    }
};

// ── tests ──

test "foldSystemMessages: a system turn past index 0 joins the leading system message" {
    // Claude Code carries SessionStart hook output as a
    // `system`-role message INSIDE `messages`, after the top-level system
    // prompt. Qwen's template raises on a system turn that is not first, and
    // the raise is a silent generic fallback (the model loses its stop token).
    const al = std.testing.allocator;
    var msgs = std.ArrayList(Message).empty;
    defer msgs.deinit(al);
    try msgs.append(al, .{ .role = "system", .content = "You are S." });
    try msgs.append(al, .{ .role = "system", .content = "hook output" });
    try msgs.append(al, .{ .role = "user", .content = "hi" });
    try msgs.append(al, .{ .role = "system", .content = "late note" });
    const owned = try foldSystemMessages(al, &msgs, false);
    defer if (owned) |o| al.free(o);
    try std.testing.expectEqual(@as(usize, 2), msgs.items.len);
    try std.testing.expectEqualStrings("You are S.\n\nhook output\n\nlate note", msgs.items[0].content);
    try std.testing.expectEqualStrings("user", msgs.items[1].role);

    // No leading system: the fold creates one at index 0.
    var lone = std.ArrayList(Message).empty;
    defer lone.deinit(al);
    try lone.append(al, .{ .role = "user", .content = "hi" });
    try lone.append(al, .{ .role = "system", .content = "hook output" });
    const owned2 = try foldSystemMessages(al, &lone, false);
    defer if (owned2) |o| al.free(o);
    try std.testing.expectEqualStrings("system", lone.items[0].role);
    try std.testing.expectEqualStrings("hook output", lone.items[0].content);
    try std.testing.expectEqual(@as(usize, 2), lone.items.len);

    // Nothing to fold: untouched, nothing allocated.
    var plain = std.ArrayList(Message).empty;
    defer plain.deinit(al);
    try plain.append(al, .{ .role = "system", .content = "You are S." });
    try plain.append(al, .{ .role = "user", .content = "hi" });
    try std.testing.expect((try foldSystemMessages(al, &plain, false)) == null);
    try std.testing.expectEqual(@as(usize, 2), plain.items.len);
}
