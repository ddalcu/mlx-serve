//! TextGenTarget, media/text-gen reject reasons, dropMedia

const std = @import("std");
const chat_mod = @import("../chat/chat.zig");
const model_mod = @import("../../sdk/config/model.zig");
const model_registry_mod = @import("../registry/registry.zig");
const media_mod = @import("../media_jobs/gen.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const PluginEngine = @import("../media_jobs/plugin/engine.zig").Engine;

/// Facts about a request's target model that decide whether a TEXT-
/// GENERATION route (/v1/chat/completions, /v1/completions, /v1/messages,
/// /v1/responses + WS, /api/chat, /api/generate) may serve it. Extracted
/// from a LoadedModel by `textGenTargetOf`; kept as plain facts so the
/// decision is hermetically testable.
pub const TextGenTarget = struct {
    is_encoder_only: bool = false,
    arch_hint: []const u8 = "",
    has_image_engine: bool = false,
    has_audio_engine: bool = false,
    has_video_engine: bool = false,
    has_mesh_engine: bool = false,
    has_decision_engine: bool = false,
    /// A text-capable LM is resident (transformer / ds4 / llama engine) —
    /// or the entry isn't loaded yet, in which case stubs default to
    /// "assume text until the arch hint or a load says otherwise".
    has_text_lm: bool = true,
};

/// Reason a text-generation route must reject this model with a 400, or
/// null when it can serve text. Crash class (live SIGSEGV): a
/// chat request routed at a media model has only the gen stub CPU state —
/// the empty stub tokenizer produced 0 prompt tokens and prefill deref'd
/// `transformer == null`, killing the whole server from one request (any
/// remote client naming a media model could down it). Media entries are
/// detected BOTH pre-load (discovery arch_hint) and post-load (engine
/// slots) — either alone has gaps: `--model` primaries carry no hint,
/// engines exist only while resident.
/// Media the user attached in the latest turn, on a model serving WITHOUT its
/// tower (`--no-vision`, no vision weights), is refused by name: silently
/// dropped, the model answered the text alone (a hallucinated "Sky" for a
/// house). Older and tool-result media are dropped from the render instead
/// (`dropMedia`), so a session that switched models, or an agent's image read,
/// does not wedge every later turn.
pub fn mediaRejectReason(messages: []const chat_mod.Message, continue_final: bool) ?[]const u8 {
    var i = messages.len;
    // A continuation's trailing assistant prefix belongs to the latest turn.
    if (continue_final and i > 0) i -= 1;
    while (i > 0) {
        i -= 1;
        const m = messages[i];
        if (std.mem.eql(u8, m.role, "assistant")) break;
        if (!std.mem.eql(u8, m.role, "user")) continue;
        if (m.images != null or m.videos != null) return "This model is serving without its vision tower (--no-vision, no vision weights, or on the Neural Engine); image/video content is not supported";
        if (m.audio != null) return "This model is serving without its audio embedder; input_audio content is not supported";
    }
    return null;
}

const MEDIA_DROPPED_NOTE = "[attachment omitted: this model cannot see images, video or audio]";

/// Strip media the model cannot read, leaving a note where it was so an agent
/// whose read returned an image learns the model never saw it. The rewritten
/// content is appended to `owned`.
pub fn dropMedia(allocator: std.mem.Allocator, messages: []chat_mod.Message, owned: *std.ArrayList([]const u8)) !void {
    for (messages) |*m| {
        if (!chat_mod.messageHasMedia(m.*)) continue;
        const text = if (m.content.len > 0)
            try std.fmt.allocPrint(allocator, "{s}\n{s}", .{ m.content, MEDIA_DROPPED_NOTE })
        else
            try allocator.dupe(u8, MEDIA_DROPPED_NOTE);
        owned.append(allocator, text) catch |err| {
            allocator.free(text);
            return err;
        };
        m.content = text;
        m.images = null;
        m.videos = null;
        m.audio = null;
    }
}

/// The URL of an OpenAI `image_url` part: `{"url": ...}`, or the bare string
/// some clients send. Null for any other shape, which the caller refuses by
/// name: a dropped image leaves the model describing a picture it never saw.
pub fn imageUrlOfPart(part: std.json.ObjectMap) ?[]const u8 {
    return switch (part.get("image_url") orelse return null) {
        .string => |s| s,
        .object => |o| if (o.get("url")) |u| (if (u == .string) u.string else null) else null,
        else => null,
    };
}

pub fn textGenRejectReason(t: TextGenTarget) ?[]const u8 {
    if (t.is_encoder_only) return "Encoder-only models do not support text generation. Use /v1/embeddings instead.";
    if (std.mem.eql(u8, t.arch_hint, "qwen3_forced_aligner")) return "This is a forced aligner: it times the words of a transcript and serves no request itself. Target a Qwen3-ASR model on /v1/audio/transcriptions; verbose_json, srt and vtt use the aligner.";
    const modality: ?media_mod.Modality = blk: {
        if (t.has_image_engine) break :blk .image;
        if (t.has_video_engine) break :blk .video;
        if (t.has_mesh_engine) break :blk .mesh;
        if (t.has_audio_engine) break :blk .audio;
        if (t.has_decision_engine) break :blk .decision;
        break :blk media_mod.modalityFromType(t.arch_hint);
    };
    if (modality) |m| return switch (m) {
        .image => "This is an image generation model; it cannot serve chat/text requests. Use POST /v1/images/generations instead.",
        .audio => "This is an audio generation model; it cannot serve chat/text requests. Use POST /v1/audio/speech (TTS), /v1/audio/music-generations (music) or /v1/audio/sound-generations (text-to-audio) instead.",
        .video => "This is a video generation model; it cannot serve chat/text requests. Use POST /v1/video/generations instead.",
        .mesh => "This is a 3D generation model; it cannot serve chat/text requests. Use POST /v1/3d/generations instead.",
        .decision => "This is a typed-decision model; it cannot serve chat/text requests. Use POST /v1/decisions instead.",
        .diarization => "This is a speaker diarizer: it labels the speakers of a transcript and serves no request itself. Target a Qwen3-ASR model on /v1/audio/transcriptions with response_format=diarized_json.",
    };
    if (!t.has_text_lm) return "This model cannot serve text generation (no language model resident).";
    return null;
}

pub fn textGenTargetOf(lm: *LoadedModel) TextGenTarget {
    // `releaseRetainedCpuState`'s reader contract. Skipping the config costs
    // the pre-load 400 for the duration of one reload; the post-load
    // `text_gen_reject` is the authoritative barrier either way.
    const cpu_state_stable = lm.state != .loading;
    return .{
        .is_encoder_only = cpu_state_stable and lm.config != null and lm.config.?.is_encoder_only,
        .arch_hint = lm.arch_hint,
        .has_image_engine = PluginEngine.serves(lm.media_plugin, media_mod.Modality.image),
        .has_audio_engine = PluginEngine.serves(lm.media_plugin, media_mod.Modality.audio),
        .has_video_engine = PluginEngine.serves(lm.media_plugin, media_mod.Modality.video),
        .has_mesh_engine = PluginEngine.serves(lm.media_plugin, media_mod.Modality.mesh),
        .has_decision_engine = PluginEngine.serves(lm.media_plugin, media_mod.Modality.decision),
        .has_text_lm = lm.state != .ready or lm.transformer != null or lm.engineBacked(),
    };
}

// ── tests ──

test "textGenTargetOf: reads config unless the entry is mid-load" {
    // Bar: a `.loading` entry's retained CPU state can be freed off-mutex, so
    // the encoder-only flag must not come from `config` in that state.
    const io = std.Io.Threaded.global_single_threaded.io();
    var reg = try model_registry_mod.ModelRegistry.init(std.testing.allocator, io, null, 3, 0, null);
    defer reg.deinit();
    const e = try reg.registerStubWithArch("m", "/path/to/m", 1024, "");

    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    cfg.is_encoder_only = true;
    e.config = &cfg;
    defer {
        e.config = null;
        e.state = .unloaded;
    }

    // Resident, and retained across an unload: the pre-load 400 still fires.
    e.state = .ready;
    try std.testing.expect(textGenTargetOf(e).is_encoder_only);
    e.state = .unloaded;
    try std.testing.expect(textGenTargetOf(e).is_encoder_only);
    try std.testing.expect(textGenRejectReason(textGenTargetOf(e)) != null);

    // Mid-load: no config read at all, so the route falls through to
    // `ensureLoaded` and the authoritative post-load check.
    e.state = .loading;
    try std.testing.expect(!textGenTargetOf(e).is_encoder_only);
    try std.testing.expect(textGenRejectReason(textGenTargetOf(e)) == null);
}

test "mediaRejectReason: media on a tower-less model is refused by name, text passes" {
    const t = std.testing;
    const text = [_]chat_mod.Message{.{ .role = "user", .content = "hi" }};
    try t.expect(mediaRejectReason(&text, false) == null);
    const img = [_]chat_mod.Message{ .{ .role = "user", .content = "hi" }, .{ .role = "user", .content = "look", .images = &[_]chat_mod.ImageData{} } };
    try t.expect(std.mem.indexOf(u8, mediaRejectReason(&img, false).?, "vision tower") != null);
    const aud = [_]chat_mod.Message{.{ .role = "user", .content = "listen", .audio = &[_]chat_mod.AudioData{} }};
    try t.expect(std.mem.indexOf(u8, mediaRejectReason(&aud, false).?, "audio") != null);
    const history = [_]chat_mod.Message{ img[1], .{ .role = "assistant", .content = "a house" }, .{ .role = "user", .content = "thanks" } };
    try t.expect(mediaRejectReason(&history, false) == null);
    // A continuation's trailing assistant prefix is not the turn boundary.
    const prefill = [_]chat_mod.Message{ img[1], .{ .role = "assistant", .content = "It shows" } };
    try t.expect(mediaRejectReason(&prefill, true) != null);
}

test "imageUrlOfPart: the object and bare-string shapes name the URL, any other shape is refused" {
    const t = std.testing;
    const cases = [_]struct { json: []const u8, url: ?[]const u8 }{
        .{ .json = "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:a\"}}", .url = "data:a" },
        .{ .json = "{\"type\":\"image_url\",\"image_url\":\"data:b\"}", .url = "data:b" },
        .{ .json = "{\"type\":\"image_url\",\"image_url\":{\"href\":\"data:c\"}}", .url = null },
        .{ .json = "{\"type\":\"image_url\",\"image_url\":{\"url\":7}}", .url = null },
        .{ .json = "{\"type\":\"image_url\",\"image_url\":null}", .url = null },
        .{ .json = "{\"type\":\"image_url\"}", .url = null },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, c.json, .{});
        defer parsed.deinit();
        const got = imageUrlOfPart(parsed.value.object);
        if (c.url) |want| try t.expectEqualStrings(want, got.?) else try t.expect(got == null);
    }
}

test "dropMedia tells the model an attachment it cannot see was there" {
    var owned = std.ArrayList([]const u8).empty;
    defer {
        for (owned.items) |o| std.testing.allocator.free(o);
        owned.deinit(std.testing.allocator);
    }
    const imgs = [_]chat_mod.ImageData{.{ .pixels = "", .width = 1, .height = 1 }};
    var msgs = [_]chat_mod.Message{
        .{ .role = "tool", .content = "Read image file [image/png]", .images = &imgs },
        .{ .role = "user", .content = "next" },
    };
    try dropMedia(std.testing.allocator, &msgs, &owned);
    try std.testing.expect(msgs[0].images == null);
    try std.testing.expectEqualStrings("Read image file [image/png]\n" ++ MEDIA_DROPPED_NOTE, msgs[0].content);
    try std.testing.expectEqualStrings("next", msgs[1].content);
}

test "textGenRejectReason: media + encoder models rejected, chat models pass (SIGSEGV class 2026-07-06)" {
    // Chat-capable targets pass — resident MLX LM, embedded engines, and
    // unloaded stubs with a chat arch hint (or none: benefit of the doubt
    // until load).
    try std.testing.expect(textGenRejectReason(.{ .arch_hint = "gemma4" }) == null);
    try std.testing.expect(textGenRejectReason(.{ .arch_hint = "gguf" }) == null);
    try std.testing.expect(textGenRejectReason(.{}) == null);

    // The live crash shape: a resident image model — engine slot set, no
    // transformer. Must reject, never reach prefill.
    {
        const r = textGenRejectReason(.{ .has_image_engine = true, .has_text_lm = false });
        try std.testing.expect(r != null);
        try std.testing.expect(std.mem.indexOf(u8, r.?, "/v1/images/generations") != null);
    }
    // Pre-load detection via the discovery arch hint — no engine resident yet.
    {
        const r = textGenRejectReason(.{ .arch_hint = "flux2-klein-4b" });
        try std.testing.expect(std.mem.indexOf(u8, r.?, "/v1/images/generations") != null);
    }
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .arch_hint = "hunyuan3d_2_1" }).?, "/v1/3d/generations") != null);
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .arch_hint = "qwen3_tts" }).?, "/v1/audio/speech") != null);
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .arch_hint = "acestep" }).?, "music-generations") != null);
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .arch_hint = "AudioVideo" }).?, "/v1/video/generations") != null);
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .has_mesh_engine = true }).?, "/v1/3d/generations") != null);

    // Encoder-only keeps its dedicated message (pre-existing guard, now
    // routed through the same decision).
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .is_encoder_only = true }).?, "/v1/embeddings") != null);
    // The forced aligner serves nothing itself: it points at the ASR that uses it.
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .arch_hint = "qwen3_forced_aligner" }).?, "Qwen3-ASR") != null);

    // Catch-all: a READY entry with no LM and no engines (future modality)
    // still rejects instead of crashing.
    try std.testing.expect(textGenRejectReason(.{ .has_text_lm = false }) != null);
}
