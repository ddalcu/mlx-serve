//! OpenAI's audio output on `/v1/chat/completions` (`modalities: ["text", "audio"]`, `audio: {voice, format}`) for a
//! model that speaks (`Parsed.speech`): what the request asks, the card's voice-chat recipe for the system prompt, and
//! the speech as WAV or 16-bit PCM.

const std = @import("std");
const wav_mod = @import("../../sdk/media/wav.zig");

pub const Format = enum { wav, pcm16 };

pub const Ask = struct { format: Format };

pub const Parse = union(enum) {
    /// Text only.
    none,
    ask: Ask,
    /// A named 400.
    reject: []const u8,
};

pub const NOT_SPOKEN = "this model does not produce audio: 'modalities' may only ask for \"text\"";
pub const BAD_MODALITY = "'modalities' must be an array of \"text\" and \"audio\"";
pub const BAD_FORMAT = "'audio.format' must be \"wav\" or \"pcm16\"";
pub const STREAM_FORMAT = "streamed audio is \"pcm16\": set 'audio.format' to \"pcm16\"";
pub const BAD_VOICE = "this model answers a chat in its own one voice: 'audio.voice' must be \"default\" (its named voices are /v1/audio/speech's)";
pub const NO_STOP = "'stop' and 'logprobs' do not apply to a spoken reply: drop them or drop \"audio\" from 'modalities'";
pub const TEXT_TOOLS = "this model answers a request with 'tools' in text: drop \"audio\" from 'modalities'";

/// The request's audio ask. `speaks` = the model has `SpeechFacts`; a streamed reply is 16-bit PCM, as OpenAI's.
pub fn parse(root: std.json.ObjectMap, speaks: bool, stream: bool) Parse {
    var audio = false;
    if (root.get("modalities")) |m| {
        if (m != .array) return .{ .reject = BAD_MODALITY };
        for (m.array.items) |v| {
            if (v != .string) return .{ .reject = BAD_MODALITY };
            if (std.mem.eql(u8, v.string, "audio")) audio = true else if (!std.mem.eql(u8, v.string, "text")) return .{ .reject = BAD_MODALITY };
        }
    }
    if (!audio) return .none;
    if (!speaks) return .{ .reject = NOT_SPOKEN };
    if (root.get("stop")) |v| if (v != .null) return .{ .reject = NO_STOP };
    if (root.get("logprobs")) |v| if (v != .null and !(v == .bool and !v.bool)) return .{ .reject = NO_STOP };
    var format: Format = if (stream) .pcm16 else .wav;
    if (root.get("audio")) |a| if (a == .object) {
        if (a.object.get("format")) |f| {
            if (f != .string) return .{ .reject = BAD_FORMAT };
            format = std.meta.stringToEnum(Format, f.string) orelse return .{ .reject = BAD_FORMAT };
        }
        if (a.object.get("voice")) |v| if (v != .string or !std.mem.eql(u8, v.string, "default")) return .{ .reject = BAD_VOICE };
    };
    if (stream and format != .pcm16) return .{ .reject = STREAM_FORMAT };
    return .{ .ask = .{ .format = format } };
}

/// The languages the card's TTS recipe names (`Perform TTS in <language> using a <gender> voice.`).
pub const LANGUAGES = [_][]const u8{
    "American English", "British English", "Arabic",  "Chinese", "Danish",     "Dutch",   "Finnish",
    "French",           "German",          "Greek",   "Hebrew",  "Italian",    "Japanese", "Korean",
    "Polish",           "Portuguese",      "Spanish", "Swedish", "Turkish",
};
pub const DEFAULT_VOICE = "american-english-female";

pub const Voice = struct { language: []const u8, gender: []const u8 };

/// A voice id, `<language>-<female|male>` (lowercase, spaces as dashes: `british-english-male`).
pub fn voice(id: []const u8) ?Voice {
    const cut = std.mem.lastIndexOfScalar(u8, id, '-') orelse return null;
    const gender = id[cut + 1 ..];
    if (!std.mem.eql(u8, gender, "female") and !std.mem.eql(u8, gender, "male")) return null;
    for (LANGUAGES) |l| {
        if (l.len != cut) continue;
        for (l, id[0..cut]) |c, d| {
            if ((if (c == ' ') '-' else std.ascii.toLower(c)) != d) break;
        } else return .{ .language = l, .gender = gender };
    }
    return null;
}

/// Every voice id, as a JSON array.
pub fn voicesJson(a: std.mem.Allocator) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.append(a, '[');
    for (LANGUAGES, 0..) |l, i| for ([_][]const u8{ "female", "male" }, 0..) |g, j| {
        if (i + j > 0) try out.append(a, ',');
        try out.append(a, '"');
        for (l) |c| try out.append(a, if (c == ' ') '-' else std.ascii.toLower(c));
        try out.print(a, "-{s}\"", .{g});
    };
    try out.append(a, ']');
    return out.toOwnedSlice(a);
}

/// An audio id: `audio_` and 128 random bits as hex, so one client cannot name another's reply.
pub const Id = [6 + 32]u8;

/// Spoken replies' transcripts under their ids (`audio_<hex>`) until they expire, so a history turn that names one
/// (`{"role": "assistant", "audio": {"id": ...}}`) replays as that text. A bounded ring: the oldest goes first.
pub const Store = struct {
    mu: std.Io.Mutex = .init,
    entries: [CAP]Entry = @splat(.{}),
    next: usize = 0,

    pub const CAP = 256;
    pub const TTL_SECS = 3600;
    const Entry = struct { id: Id = @splat(0), expires: i64 = 0, text: []u8 = &.{} };
    const gpa = std.heap.c_allocator;

    pub fn newId(io: std.Io) Id {
        var rnd: [16]u8 = undefined;
        io.random(&rnd);
        return "audio_".* ++ std.fmt.bytesToHex(rnd, .lower);
    }

    pub fn put(self: *Store, io: std.Io, id: Id, now: i64, text: []const u8) !void {
        const copy = try gpa.dupe(u8, text);
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        const e = &self.entries[self.next];
        gpa.free(e.text);
        e.* = .{ .id = id, .expires = now + TTL_SECS, .text = copy };
        self.next = (self.next + 1) % CAP;
    }

    /// The transcript of the reply `name` (caller frees); null when unknown or expired.
    pub fn get(self: *Store, io: std.Io, a: std.mem.Allocator, name: []const u8, now: i64) !?[]u8 {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        for (self.entries) |e| if (e.expires > now and std.mem.eql(u8, &e.id, name)) return try a.dupe(u8, e.text);
        return null;
    }
};

pub var store: Store = .{};

pub const UNKNOWN_AUDIO_ID = "an assistant turn names an 'audio.id' this server no longer holds (expired or unknown): send its transcript as 'content'";

/// The card's voice-chat system prompt; with tools the model calls them or answers in speech, and the template's tool
/// list follows its last line.
pub fn recipe(has_tools: bool) []const u8 {
    return if (has_tools) "Respond with function calls, and interleaved text and audio.\n\nList of tools:" else "Respond with interleaved text and audio.";
}

/// Mono f32 samples as little-endian signed 16-bit, clipped: a WAV's body.
pub fn pcm16(a: std.mem.Allocator, samples: []const f32) ![]u8 {
    const w = try wav_mod.encodePcm16Mono(a, samples, 0);
    std.mem.copyForwards(u8, w[0 .. w.len - 44], w[44..]);
    return a.realloc(w, w.len - 44);
}

/// The speech in `format`, base64.
pub fn encode(a: std.mem.Allocator, samples: []const f32, rate: u32, format: Format) ![]u8 {
    const raw = switch (format) {
        .wav => try wav_mod.encodePcm16Mono(a, samples, rate),
        .pcm16 => try pcm16(a, samples),
    };
    defer a.free(raw);
    const enc = std.base64.standard.Encoder;
    const out = try a.alloc(u8, enc.calcSize(raw.len));
    _ = enc.encode(out, raw);
    return out;
}

// ── tests ──

const testing = std.testing;

fn parseJson(js: []const u8, speaks: bool, stream: bool) !Parse {
    const p = try std.json.parseFromSlice(std.json.Value, testing.allocator, js, .{});
    defer p.deinit();
    return parse(p.value.object, speaks, stream);
}

test "speech ask: audio in modalities asks for speech; the format defaults per transport; refusals are named" {
    try testing.expectEqual(Parse.none, try parseJson("{}", true, false));
    try testing.expectEqual(Parse.none, try parseJson("{\"modalities\":[\"text\"]}", false, false));
    try testing.expectEqual(Format.wav, (try parseJson("{\"modalities\":[\"text\",\"audio\"]}", true, false)).ask.format);
    try testing.expectEqual(Format.pcm16, (try parseJson("{\"modalities\":[\"audio\"]}", true, true)).ask.format);
    try testing.expectEqual(Format.pcm16, (try parseJson("{\"modalities\":[\"audio\"],\"audio\":{\"voice\":\"default\",\"format\":\"pcm16\"}}", true, false)).ask.format);
    // A chat reply speaks in the model's one voice (the card's recipe names none): another voice is refused, not ignored.
    try testing.expectEqualStrings(BAD_VOICE, (try parseJson("{\"modalities\":[\"audio\"],\"audio\":{\"voice\":\"alloy\"}}", true, false)).reject);
    try testing.expectEqualStrings(BAD_VOICE, (try parseJson("{\"modalities\":[\"audio\"],\"audio\":{\"voice\":\"british-english-male\"}}", true, false)).reject);
    try testing.expectEqualStrings(NO_STOP, (try parseJson("{\"modalities\":[\"audio\"],\"stop\":[\"x\"]}", true, false)).reject);
    try testing.expectEqualStrings(NO_STOP, (try parseJson("{\"modalities\":[\"audio\"],\"logprobs\":true}", true, true)).reject);
    try testing.expectEqualStrings(NOT_SPOKEN, (try parseJson("{\"modalities\":[\"audio\"]}", false, false)).reject);
    try testing.expectEqualStrings(BAD_MODALITY, (try parseJson("{\"modalities\":[\"video\"]}", true, false)).reject);
    try testing.expectEqualStrings(BAD_MODALITY, (try parseJson("{\"modalities\":\"audio\"}", true, false)).reject);
    try testing.expectEqualStrings(BAD_FORMAT, (try parseJson("{\"modalities\":[\"audio\"],\"audio\":{\"format\":\"mp3\"}}", true, false)).reject);
    try testing.expectEqualStrings(STREAM_FORMAT, (try parseJson("{\"modalities\":[\"audio\"],\"audio\":{\"format\":\"wav\"}}", true, true)).reject);
}

test "speech voices: <language>-<gender> ids name the card's TTS languages" {
    const v = voice("british-english-male").?;
    try testing.expectEqualStrings("British English", v.language);
    try testing.expectEqualStrings("male", v.gender);
    try testing.expectEqualStrings("Japanese", voice("japanese-female").?.language);
    try testing.expect(voice(DEFAULT_VOICE) != null);
    try testing.expect(voice("alloy") == null);
    try testing.expect(voice("english-female") == null);
    try testing.expect(voice("japanese-robot") == null);
    const all = try voicesJson(testing.allocator);
    defer testing.allocator.free(all);
    try testing.expect(std.mem.startsWith(u8, all, "[\"american-english-female\",\"american-english-male\",\"british-english-female\""));
    try testing.expectEqual(@as(usize, 2 * LANGUAGES.len), std.mem.count(u8, all, ","  ) + 1);
}

test "speech store: a reply's transcript is found by its id until it expires; the ring drops the oldest" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var st: Store = .{};
    const id = Store.newId(io);
    try st.put(io, id, 100, "Paris.");
    const name: []const u8 = &id;
    try testing.expect(!std.mem.eql(u8, name, &Store.newId(io)));
    const got = (try st.get(io, testing.allocator, name, 200)).?;
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("Paris.", got);
    try testing.expect(try st.get(io, testing.allocator, name, 100 + Store.TTL_SECS) == null);
    try testing.expect(try st.get(io, testing.allocator, "audio_1", 200) == null);
    for (0..Store.CAP) |_| try st.put(io, Store.newId(io), 100, "later");
    try testing.expect(try st.get(io, testing.allocator, name, 200) == null);
    for (&st.entries) |*e| Store.gpa.free(e.text);
}

test "speech pcm16: a WAV's body, clipped little-endian samples" {
    const p = try pcm16(testing.allocator, &.{ 0, 1, -2 });
    defer testing.allocator.free(p);
    try testing.expectEqual(@as(usize, 6), p.len);
    try testing.expectEqual(@as(i16, 32767), std.mem.readInt(i16, p[2..4], .little));
    try testing.expectEqual(@as(i16, -32767), std.mem.readInt(i16, p[4..6], .little));
}
