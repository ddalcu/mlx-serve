//! `POST /v1/audio/transcriptions` (OpenAI's form) over Qwen3-ASR: the request
//! form, the language names the model was trained on, the reference prompt
//! (mlx-audio `qwen3_asr._build_prompt`) and the model's `language X<asr_text>`
//! lead. No model state here; the handler lives in server.zig.

const std = @import("std");
const multipart = @import("multipart.zig");
const wav = @import("wav.zig");
const chat = @import("chat.zig");
const forced_align = @import("forced_align.zig");

pub const SAMPLE_RATE: u32 = 16000;

pub const Format = enum { json, text, verbose_json, srt, vtt, diarized_json };

/// Formats that carry times.
pub fn timed(f: Format) bool {
    return f == .verbose_json or f == .srt or f == .vtt or f == .diarized_json;
}

pub const Form = struct {
    file: []const u8,
    /// A Qwen3-ASR language name, resolved from an ISO-639-1 code or a name.
    language: ?[]const u8 = null,
    /// Context the transcript should follow (names, vocabulary): the system turn.
    prompt: ?[]const u8 = null,
    format: Format = .json,
    temperature: f32 = 0,
    /// `timestamp_granularities[]` asked for `word`.
    words: bool = false,
};

pub const FormError = error{ NotMultipart, MissingFile, UnsupportedResponseFormat, StreamUnsupported, BadTemperature, UnsupportedLanguage, KnownSpeakersUnsupported };

pub fn formErrorMessage(err: FormError) []const u8 {
    return switch (err) {
        error.NotMultipart => "send the audio as multipart/form-data with a `file` field",
        error.MissingFile => "`file` is required",
        error.UnsupportedResponseFormat => "response_format must be json, text, verbose_json, srt, vtt or diarized_json",
        error.StreamUnsupported => "stream is not supported on /v1/audio/transcriptions",
        error.BadTemperature => "temperature must be a number in [0, 1]",
        error.UnsupportedLanguage => "language is not one Qwen3-ASR transcribes; omit it to detect the language",
        error.KnownSpeakersUnsupported => "known speakers are not supported: diarized_json labels speakers A, B, ... in the order they first speak",
    };
}

/// Every field a client may send; unknown fields are ignored like OpenAI's.
pub fn parseForm(body: []const u8, content_type: []const u8) FormError!Form {
    const boundary = multipart.boundaryFromContentType(content_type) orelse return error.NotMultipart;
    var it = multipart.Iterator.init(body, boundary) catch return error.NotMultipart;
    var form = Form{ .file = "" };
    while (it.next()) |part| {
        const v = part.data;
        if (std.mem.eql(u8, part.name, "file")) {
            form.file = v;
        } else if (std.mem.eql(u8, part.name, "language")) {
            if (v.len != 0) form.language = languageName(v) orelse return error.UnsupportedLanguage;
        } else if (std.mem.eql(u8, part.name, "prompt")) {
            if (v.len != 0) form.prompt = v;
        } else if (std.mem.eql(u8, part.name, "response_format")) {
            if (v.len != 0) form.format = std.meta.stringToEnum(Format, v) orelse return error.UnsupportedResponseFormat;
        } else if (std.mem.eql(u8, part.name, "temperature")) {
            if (v.len != 0) {
                form.temperature = std.fmt.parseFloat(f32, v) catch return error.BadTemperature;
                if (!(form.temperature >= 0 and form.temperature <= 1)) return error.BadTemperature;
            }
        } else if (std.mem.eql(u8, part.name, "stream")) {
            if (std.mem.eql(u8, v, "true")) return error.StreamUnsupported;
        } else if (std.mem.eql(u8, part.name, "timestamp_granularities[]")) {
            if (std.mem.eql(u8, v, "word")) form.words = true;
        } else if (std.mem.eql(u8, part.name, "known_speaker_names[]") or std.mem.eql(u8, part.name, "known_speaker_references[]")) {
            return error.KnownSpeakersUnsupported;
        }
    }
    if (form.file.len == 0) return error.MissingFile;
    return form;
}

/// The model card's 30 languages, each with its ISO-639 code (Whisper's for
/// Cantonese and Filipino).
const LANGUAGES = [_][2][]const u8{
    .{ "zh", "Chinese" },    .{ "en", "English" },  .{ "yue", "Cantonese" }, .{ "ar", "Arabic" },    .{ "de", "German" },
    .{ "fr", "French" },     .{ "es", "Spanish" },  .{ "pt", "Portuguese" }, .{ "id", "Indonesian" }, .{ "it", "Italian" },
    .{ "ko", "Korean" },     .{ "ru", "Russian" },  .{ "th", "Thai" },       .{ "vi", "Vietnamese" }, .{ "ja", "Japanese" },
    .{ "tr", "Turkish" },    .{ "hi", "Hindi" },    .{ "ms", "Malay" },      .{ "nl", "Dutch" },      .{ "sv", "Swedish" },
    .{ "da", "Danish" },     .{ "fi", "Finnish" },  .{ "pl", "Polish" },     .{ "cs", "Czech" },      .{ "tl", "Filipino" },
    .{ "fa", "Persian" },    .{ "el", "Greek" },    .{ "ro", "Romanian" },   .{ "hu", "Hungarian" },  .{ "mk", "Macedonian" },
};

/// The model's name for an ISO code or a name in any case; null when unsupported.
pub fn languageName(code_or_name: []const u8) ?[]const u8 {
    for (LANGUAGES) |l| {
        if (std.ascii.eqlIgnoreCase(code_or_name, l[0]) or std.ascii.eqlIgnoreCase(code_or_name, l[1])) return l[1];
    }
    return null;
}

/// One bit per language of the table (a set of languages fits a u32); 0 for any other name.
pub fn languageBit(name: []const u8) u32 {
    for (LANGUAGES, 0..) |l, i| {
        if (std.ascii.eqlIgnoreCase(name, l[1])) return @as(u32, 1) << @intCast(i);
    }
    return 0;
}

/// Longest piece the aligner times: the reference's `MAX_FORCE_ALIGN_INPUT_SECONDS`.
pub const ALIGN_MAX_SECONDS = 180;

/// Ends of the pieces `pcm` is cut into, each about `max` samples: the cut
/// sits in the quietest 100 ms within 5 s either side of its limit (the
/// reference's `split_audio_into_chunks`). The last end is `pcm.len`.
pub fn splitPoints(a: std.mem.Allocator, pcm: []const f32, max: usize) ![]usize {
    const expand: usize = 5 * SAMPLE_RATE;
    const win: usize = SAMPLE_RATE / 10;
    var ends = std.ArrayList(usize).empty;
    errdefer ends.deinit(a);
    var start: usize = 0;
    while (pcm.len - start > max) {
        const cut = start + max;
        const left = @max(start, cut -| expand);
        const right = @min(pcm.len, cut + expand);
        var boundary = cut;
        if (right - left > win) {
            // The quietest window (its sum of magnitudes), then its quietest sample.
            var sum: f64 = 0;
            for (pcm[left .. left + win]) |x| sum += @abs(x);
            var best = sum;
            var best_at: usize = 0;
            for (1..right - left - win + 1) |p| {
                sum += @abs(pcm[left + p + win - 1]) - @abs(pcm[left + p - 1]);
                if (sum < best) {
                    best = sum;
                    best_at = p;
                }
            }
            var inner: usize = 0;
            for (pcm[left + best_at .. left + best_at + win], 0..) |x, k| {
                if (@abs(x) < @abs(pcm[left + best_at + inner])) inner = k;
            }
            boundary = left + best_at + inner;
        }
        boundary = @min(@max(boundary, start + 1), pcm.len);
        try ends.append(a, boundary);
        start = boundary;
    }
    try ends.append(a, pcm.len);
    return ends.toOwnedSlice(a);
}

/// A timed span of `Timeline.text`, in seconds, and its speaker.
pub const Span = struct { start: f64, end: f64, lo: usize, hi: usize, speaker: u8 = 0 };

/// Speaker activity at 10 ms over a whole clip: `speakers` probabilities a frame.
pub const Activity = struct {
    probs: []const f32,
    speakers: usize,

    /// The speaker most active over `[start_ms, end_ms)` of the clip (at least
    /// its first frame); `fallback` when nobody is.
    pub fn speakerOver(self: Activity, start_ms: u64, end_ms: u64, fallback: u8) u8 {
        const frames = self.probs.len / self.speakers;
        const lo = @min(start_ms / 10, frames);
        const hi = @min(@max(end_ms / 10, lo + 1), frames);
        var best = fallback;
        var best_sum: f32 = 0;
        for (0..self.speakers) |s| {
            var sum: f32 = 0;
            for (lo..hi) |t| sum += self.probs[t * self.speakers + s];
            if (sum > best_sum) {
                best_sum = sum;
                best = @intCast(s);
            }
        }
        return best;
    }

    /// Each word's speaker, its times the piece's from `offset_ms` into the
    /// clip; a word nobody is heard saying keeps the previous word's.
    pub fn wordSpeakers(self: Activity, a: std.mem.Allocator, words: []const forced_align.Timed, offset_ms: u64) ![]u8 {
        const out = try a.alloc(u8, words.len);
        var last: u8 = 0;
        for (words, out) |w, *o| {
            o.* = self.speakerOver(offset_ms + w.start_ms, offset_ms + w.end_ms, last);
            last = o.*;
        }
        return out;
    }
};

/// A transcript put together from pieces of audio, with its segments and words.
pub const Timeline = struct {
    text: std.ArrayList(u8) = .empty,
    segments: std.ArrayList(Span) = .empty,
    words: std.ArrayList(Span) = .empty,
    /// The first piece with text names it (the first piece when none has text).
    language: ?[]u8 = null,

    pub fn deinit(self: *Timeline, a: std.mem.Allocator) void {
        self.text.deinit(a);
        self.segments.deinit(a);
        self.words.deinit(a);
        if (self.language) |l| a.free(l);
    }

    /// Append the transcript of the piece `offset .. offset + duration` seconds:
    /// its aligned words and segments (spans of `piece`), or, without any, one
    /// segment over the whole piece said by `speaker`. Chinese, Cantonese and
    /// Japanese join without a space.
    pub fn add(self: *Timeline, a: std.mem.Allocator, piece: []const u8, language: ?[]const u8, offset: f64, duration: f64, words: ?[]const forced_align.Timed, segs: ?[]const forced_align.Segment, speaker: u8) !void {
        if (language) |l| if (self.language == null or (self.segments.items.len == 0 and piece.len > 0)) {
            const owned = try a.dupe(u8, l);
            if (self.language) |old| a.free(old);
            self.language = owned;
        };
        if (piece.len == 0) return;
        const lang = language orelse "";
        const unspaced = std.mem.eql(u8, lang, "Chinese") or std.mem.eql(u8, lang, "Cantonese") or std.mem.eql(u8, lang, "Japanese");
        if (self.text.items.len > 0 and !unspaced) try self.text.append(a, ' ');
        const base = self.text.items.len;
        try self.text.appendSlice(a, piece);
        const at = struct {
            fn span(t: anytype, b: usize, o: f64) Span {
                return .{ .start = o + @as(f64, @floatFromInt(t.start_ms)) / 1000, .end = o + @as(f64, @floatFromInt(t.end_ms)) / 1000, .lo = b + t.start, .hi = b + t.end, .speaker = if (@hasField(@TypeOf(t), "speaker")) t.speaker else 0 };
            }
        }.span;
        const ss = segs orelse &.{};
        if (ss.len == 0) return self.segments.append(a, .{ .start = offset, .end = offset + duration, .lo = base, .hi = base + piece.len, .speaker = speaker });
        for (ss) |s| try self.segments.append(a, at(s, base, offset));
        for (words orelse &.{}) |w| try self.words.append(a, at(w, base, offset));
    }

    /// OpenAI's `verbose_json`; `words` when the request asked for them.
    pub fn verboseJson(self: *const Timeline, a: std.mem.Allocator, duration: f64, with_words: bool) ![]u8 {
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(a);
        // OpenAI names the language in lower case ("english").
        var lang_buf: [32]u8 = undefined;
        const name = self.language orelse "";
        try out.appendSlice(a, "{\"task\":\"transcribe\",\"language\":");
        try chat.appendJsonString(a, &out, std.ascii.lowerString(&lang_buf, name[0..@min(name.len, lang_buf.len)]));
        try out.print(a, ",\"duration\":{d:.3},\"text\":", .{duration});
        try chat.appendJsonString(a, &out, self.text.items);
        try out.appendSlice(a, ",\"segments\":[");
        for (self.segments.items, 0..) |s, i| {
            if (i > 0) try out.append(a, ',');
            try out.print(a, "{{\"id\":{d},\"seek\":0,\"start\":{d:.3},\"end\":{d:.3},\"text\":", .{ i, s.start, s.end });
            try chat.appendJsonString(a, &out, self.text.items[s.lo..s.hi]);
            try out.append(a, '}');
        }
        try out.append(a, ']');
        if (with_words) {
            try out.appendSlice(a, ",\"words\":[");
            for (self.words.items, 0..) |w, i| {
                if (i > 0) try out.append(a, ',');
                try out.appendSlice(a, "{\"word\":");
                try chat.appendJsonString(a, &out, self.text.items[w.lo..w.hi]);
                try out.print(a, ",\"start\":{d:.3},\"end\":{d:.3}}}", .{ w.start, w.end });
            }
            try out.append(a, ']');
        }
        try out.append(a, '}');
        return out.toOwnedSlice(a);
    }

    /// OpenAI's `diarized_json`: the segments with their speakers, lettered in
    /// the order they first speak.
    pub fn diarizedJson(self: *const Timeline, a: std.mem.Allocator, duration: f64) ![]u8 {
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(a);
        try out.print(a, "{{\"task\":\"transcribe\",\"duration\":{d:.3},\"text\":", .{duration});
        try chat.appendJsonString(a, &out, self.text.items);
        try out.appendSlice(a, ",\"segments\":[");
        var letters: [256]u8 = @splat(0);
        var next: u8 = 'A';
        for (self.segments.items, 1..) |s, n| {
            if (letters[s.speaker] == 0) {
                letters[s.speaker] = next;
                next += 1;
            }
            if (n > 1) try out.append(a, ',');
            try out.print(a, "{{\"type\":\"transcript.text.segment\",\"id\":\"seg_{d:0>3}\",\"start\":{d:.3},\"end\":{d:.3},\"text\":", .{ n, s.start, s.end });
            try chat.appendJsonString(a, &out, self.text.items[s.lo..s.hi]);
            try out.print(a, ",\"speaker\":\"{c}\"}}", .{letters[s.speaker]});
        }
        try out.appendSlice(a, "]}");
        return out.toOwnedSlice(a);
    }

    /// The segments as SubRip or WebVTT cues.
    pub fn subtitles(self: *const Timeline, a: std.mem.Allocator, format: Format) ![]u8 {
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(a);
        const sep: u8 = if (format == .srt) ',' else '.';
        if (format == .vtt) try out.appendSlice(a, "WEBVTT\n\n");
        var b0: [16]u8 = undefined;
        var b1: [16]u8 = undefined;
        for (self.segments.items, 1..) |s, n| {
            if (format == .srt) try out.print(a, "{d}\n", .{n});
            try out.print(a, "{s} --> {s}\n{s}\n\n", .{ clock(&b0, s.start, sep), clock(&b1, s.end, sep), self.text.items[s.lo..s.hi] });
        }
        return out.toOwnedSlice(a);
    }
};

/// `HH:MM:SS` then `sep` and milliseconds.
fn clock(buf: *[16]u8, seconds: f64, sep: u8) []const u8 {
    const ms: u64 = @intFromFloat(@round(@max(seconds, 0) * 1000));
    return std.fmt.bufPrint(buf, "{d:0>2}:{d:0>2}:{d:0>2}{c}{d:0>3}", .{ ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, sep, ms % 1000 }) catch unreachable;
}

/// The reference prompt with ONE `<|audio_pad|>`, which the server expands to
/// the tower's rows. A known language is committed as the answer's lead, as
/// the reference does; otherwise the model names the language itself.
pub fn buildPrompt(a: std.mem.Allocator, context: ?[]const u8, language: ?[]const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "<|im_start|>system\n");
    if (context) |c| {
        try out.appendSlice(a, c);
        try out.append(a, '\n');
    }
    try out.appendSlice(a, "<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n");
    if (language) |l| try out.print(a, "language {s}<asr_text>", .{l});
    return out.toOwnedSlice(a);
}

/// A WAV file as 16 kHz mono f32: channels averaged, then resampled.
pub fn decodeWav(a: std.mem.Allocator, bytes: []const u8) ![]f32 {
    const d = try wav.decode(a, bytes);
    defer a.free(d.pcm);
    const ch: usize = d.channels;
    const mono = try a.alloc(f32, d.pcm.len / ch);
    defer a.free(mono);
    for (mono, 0..) |*m, i| {
        var sum: f32 = 0;
        for (d.pcm[i * ch ..][0..ch]) |v| sum += v;
        m.* = sum / @as(f32, @floatFromInt(ch));
    }
    return wav.resampleLinear(a, mono, 1, d.sample_rate, SAMPLE_RATE);
}

pub const Transcript = struct { language: ?[]const u8, text: []const u8 };

/// Split the model's `language English<asr_text>...` lead; a forced language
/// was in the prompt, so its output is the text alone.
pub fn splitOutput(out: []const u8, forced: ?[]const u8) Transcript {
    const marker = "<asr_text>";
    if (std.mem.startsWith(u8, out, "language ")) {
        if (std.mem.indexOf(u8, out, marker)) |i| return .{
            .language = std.mem.trim(u8, out["language ".len..i], " "),
            .text = std.mem.trim(u8, out[i + marker.len ..], " \n\r\t"),
        };
    }
    return .{ .language = forced, .text = std.mem.trim(u8, out, " \n\r\t") };
}

// ── Tests ──

const testing = std.testing;

fn formBody(comptime parts: []const [2][]const u8) []const u8 {
    comptime var b: []const u8 = "";
    inline for (parts) |p| b = b ++ "--X\r\nContent-Disposition: form-data; name=\"" ++ p[0] ++ "\"\r\n\r\n" ++ p[1] ++ "\r\n";
    return b ++ "--X--\r\n";
}

const CT = "multipart/form-data; boundary=X";

test "parseForm reads OpenAI's transcription form" {
    const f = try parseForm(formBody(&.{
        .{ "file", "RIFF" },           .{ "model", "whisper-1" },          .{ "language", "en" },
        .{ "prompt", "Zig, MLX" },     .{ "response_format", "verbose_json" }, .{ "temperature", "0.2" },
        .{ "timestamp_granularities[]", "word" }, .{ "chunking_strategy", "auto" },
    }), CT);
    try testing.expectEqualStrings("RIFF", f.file);
    try testing.expectEqualStrings("English", f.language.?);
    try testing.expectEqualStrings("Zig, MLX", f.prompt.?);
    try testing.expectEqual(Format.verbose_json, f.format);
    try testing.expectEqual(@as(f32, 0.2), f.temperature);
    const d = try parseForm(formBody(&.{.{ "file", "RIFF" }}), CT);
    try testing.expectEqual(Format.json, d.format);
    try testing.expect(d.language == null and d.prompt == null);
}

test "parseForm takes subtitles and word times" {
    const s = try parseForm(formBody(&.{ .{ "file", "x" }, .{ "response_format", "srt" } }), CT);
    try testing.expect(s.format == .srt and timed(s.format) and !s.words);
    const v = try parseForm(formBody(&.{ .{ "file", "x" }, .{ "response_format", "vtt" } }), CT);
    try testing.expect(v.format == .vtt and timed(v.format));
    const w = try parseForm(formBody(&.{ .{ "file", "x" }, .{ "response_format", "verbose_json" }, .{ "timestamp_granularities[]", "segment" }, .{ "timestamp_granularities[]", "word" } }), CT);
    try testing.expect(w.words and timed(w.format));
    const seg = try parseForm(formBody(&.{ .{ "file", "x" }, .{ "timestamp_granularities[]", "segment" } }), CT);
    try testing.expect(!seg.words and !timed(seg.format));
    const d = try parseForm(formBody(&.{ .{ "file", "x" }, .{ "response_format", "diarized_json" }, .{ "chunking_strategy", "auto" } }), CT);
    try testing.expect(d.format == .diarized_json and timed(d.format));
}

test "Activity names the speaker most active over a span" {
    // 10 ms frames of two speakers: 0 talks for 50 ms, then 1, then nobody.
    const probs = [_]f32{ 0.9, 0.1, 0.9, 0.1, 0.8, 0.3, 0.7, 0.4, 0.6, 0.2, 0.2, 0.9, 0.1, 0.9, 0.1, 0.8, 0, 0, 0, 0 };
    const act = Activity{ .probs = &probs, .speakers = 2 };
    try testing.expectEqual(@as(u8, 0), act.speakerOver(0, 50, 1));
    try testing.expectEqual(@as(u8, 1), act.speakerOver(50, 80, 0));
    // A span shorter than a frame reads its frame; a silent or past-the-end span keeps the fallback.
    try testing.expectEqual(@as(u8, 1), act.speakerOver(61, 62, 0));
    try testing.expectEqual(@as(u8, 1), act.speakerOver(80, 100, 1));
    try testing.expectEqual(@as(u8, 0), act.speakerOver(500, 600, 0));
    // A piece's words: their times are the piece's, the activity the clip's.
    const words = [_]forced_align.Timed{ .{ .start_ms = 0, .end_ms = 20, .start = 0, .end = 1 }, .{ .start_ms = 30, .end_ms = 40, .start = 2, .end = 3 }, .{ .start_ms = 50, .end_ms = 60, .start = 4, .end = 5 } };
    const who = try act.wordSpeakers(testing.allocator, &words, 20);
    defer testing.allocator.free(who);
    try testing.expectEqualSlices(u8, &.{ 0, 1, 1 }, who);
}

test "parseForm refuses by name what it cannot honour" {
    try testing.expectError(error.NotMultipart, parseForm("{}", "application/json"));
    try testing.expectError(error.MissingFile, parseForm(formBody(&.{.{ "model", "m" }}), CT));
    try testing.expectError(error.UnsupportedResponseFormat, parseForm(formBody(&.{ .{ "file", "x" }, .{ "response_format", "url" } }), CT));
    try testing.expectError(error.KnownSpeakersUnsupported, parseForm(formBody(&.{ .{ "file", "x" }, .{ "known_speaker_names[]", "agent" } }), CT));
    try testing.expectError(error.StreamUnsupported, parseForm(formBody(&.{ .{ "file", "x" }, .{ "stream", "true" } }), CT));
    try testing.expectError(error.BadTemperature, parseForm(formBody(&.{ .{ "file", "x" }, .{ "temperature", "2" } }), CT));
    try testing.expectError(error.UnsupportedLanguage, parseForm(formBody(&.{ .{ "file", "x" }, .{ "language", "xx" } }), CT));
}

test "languageName takes ISO codes and names in any case" {
    try testing.expectEqualStrings("English", languageName("en").?);
    try testing.expectEqualStrings("Cantonese", languageName("yue").?);
    try testing.expectEqualStrings("German", languageName("GERMAN").?);
    try testing.expect(languageName("klingon") == null);
}

test "buildPrompt is the reference prompt" {
    const a = testing.allocator;
    const plain = try buildPrompt(a, null, null);
    defer a.free(plain);
    try testing.expectEqualStrings("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n", plain);
    const full = try buildPrompt(a, "Names: Zig", "English");
    defer a.free(full);
    try testing.expect(std.mem.startsWith(u8, full, "<|im_start|>system\nNames: Zig\n<|im_end|>\n"));
    try testing.expect(std.mem.endsWith(u8, full, "<|im_start|>assistant\nlanguage English<asr_text>"));
}

test "decodeWav returns 16 kHz mono from any rate and channel count" {
    const a = testing.allocator;
    // 0.5 s of 48 kHz stereo, left 0.5 and right -0.25: the mix is 0.125.
    var pcm: [48000]f32 = undefined;
    for (0..24000) |i| {
        pcm[2 * i] = 0.5;
        pcm[2 * i + 1] = -0.25;
    }
    const bytes = try wav.encodePcm16(a, &pcm, 48000, 2);
    defer a.free(bytes);
    const out = try decodeWav(a, bytes);
    defer a.free(out);
    try testing.expectEqual(@as(usize, 8000), out.len);
    for (out) |v| try testing.expectApproxEqAbs(@as(f32, 0.125), v, 1e-3);
    try testing.expectError(error.BadWav, decodeWav(a, "ID3 not a wav file at all, mp3 bytes here....."));
}

test "languageBit names the aligner's languages within the transcription table" {
    try testing.expect(languageBit("English") != 0 and languageBit("Korean") != 0);
    try testing.expect(languageBit("English") != languageBit("Korean"));
    try testing.expectEqual(@as(u32, 0), languageBit("None"));
}

test "splitPoints cuts in the quietest stretch near each limit and tiles the clip" {
    const a = testing.allocator;
    // 25 s of tone with silences at 9.0-9.4 s and 18.5-18.9 s, cut at most every 10 s.
    const pcm = try a.alloc(f32, 25 * SAMPLE_RATE);
    defer a.free(pcm);
    for (pcm, 0..) |*x, i| {
        const t = @as(f32, @floatFromInt(i)) / SAMPLE_RATE;
        const quiet = (t >= 9.0 and t < 9.4) or (t >= 18.5 and t < 18.9);
        x.* = if (quiet) 0 else 0.5 * @sin(2 * std.math.pi * 220 * t);
    }
    const ends = try splitPoints(a, pcm, 10 * SAMPLE_RATE);
    defer a.free(ends);
    try testing.expectEqual(@as(usize, 3), ends.len);
    try testing.expect(ends[0] >= 9 * SAMPLE_RATE and ends[0] < 94 * SAMPLE_RATE / 10);
    try testing.expect(ends[1] >= 185 * SAMPLE_RATE / 10 and ends[1] < 189 * SAMPLE_RATE / 10);
    try testing.expectEqual(pcm.len, ends[2]);
    // A clip within the limit is one piece.
    const one = try splitPoints(a, pcm[0 .. 5 * SAMPLE_RATE], 10 * SAMPLE_RATE);
    defer a.free(one);
    try testing.expectEqualSlices(usize, &.{5 * SAMPLE_RATE}, one);
}

test "Timeline renders verbose_json, srt and vtt from timed pieces" {
    const a = testing.allocator;
    var tl = Timeline{};
    defer tl.deinit(a);
    // Two pieces of audio: one timed word by word, one in a language without times.
    const first = "Hello there. \"Bye\"";
    const words = [_]forced_align.Timed{
        .{ .start_ms = 0, .end_ms = 400, .start = 0, .end = 5 },
        .{ .start_ms = 400, .end_ms = 900, .start = 6, .end = 11 },
        .{ .start_ms = 2400, .end_ms = 2800, .start = 14, .end = 17 },
    };
    const segs = [_]forced_align.Segment{ .{ .start_ms = 0, .end_ms = 900, .start = 0, .end = 12 }, .{ .start_ms = 2400, .end_ms = 2800, .start = 13, .end = 18 } };
    try tl.add(a, first, "English", 0, 3, &words, &segs, 0);
    try tl.add(a, "Hallo", "German", 3.5, 2, null, null, 0);
    try testing.expectEqualStrings("Hello there. \"Bye\" Hallo", tl.text.items);

    const v = try tl.verboseJson(a, 5.5, true);
    defer a.free(v);
    try testing.expectEqualStrings(
        \\{"task":"transcribe","language":"english","duration":5.500,"text":"Hello there. \"Bye\" Hallo","segments":[{"id":0,"seek":0,"start":0.000,"end":0.900,"text":"Hello there."},{"id":1,"seek":0,"start":2.400,"end":2.800,"text":"\"Bye\""},{"id":2,"seek":0,"start":3.500,"end":5.500,"text":"Hallo"}],"words":[{"word":"Hello","start":0.000,"end":0.400},{"word":"there","start":0.400,"end":0.900},{"word":"Bye","start":2.400,"end":2.800}]}
    , v);
    const no_words = try tl.verboseJson(a, 5.5, false);
    defer a.free(no_words);
    try testing.expect(std.mem.indexOf(u8, no_words, "\"words\"") == null);

    const srt = try tl.subtitles(a, .srt);
    defer a.free(srt);
    try testing.expectEqualStrings("1\n00:00:00,000 --> 00:00:00,900\nHello there.\n\n2\n00:00:02,400 --> 00:00:02,800\n\"Bye\"\n\n3\n00:00:03,500 --> 00:00:05,500\nHallo\n\n", srt);
    const vtt = try tl.subtitles(a, .vtt);
    defer a.free(vtt);
    try testing.expectEqualStrings("WEBVTT\n\n00:00:00.000 --> 00:00:00.900\nHello there.\n\n00:00:02.400 --> 00:00:02.800\n\"Bye\"\n\n00:00:03.500 --> 00:00:05.500\nHallo\n\n", vtt);
}

test "Timeline renders diarized_json, its speakers lettered in the order they first speak" {
    const a = testing.allocator;
    var tl = Timeline{};
    defer tl.deinit(a);
    const text = "Hi. Hello. Bye.";
    const segs = [_]forced_align.Segment{
        .{ .start_ms = 0, .end_ms = 500, .start = 0, .end = 3, .speaker = 2 },
        .{ .start_ms = 600, .end_ms = 1200, .start = 4, .end = 10, .speaker = 0 },
        .{ .start_ms = 1300, .end_ms = 1700, .start = 11, .end = 15, .speaker = 2 },
    };
    try tl.add(a, text, "English", 0, 2, &.{}, &segs, 0);
    // A piece without word times is one segment, its main speaker given.
    try tl.add(a, "Ciao", "Italian", 2, 1, null, null, 0);
    const d = try tl.diarizedJson(a, 3);
    defer a.free(d);
    try testing.expectEqualStrings(
        \\{"task":"transcribe","duration":3.000,"text":"Hi. Hello. Bye. Ciao","segments":[{"type":"transcript.text.segment","id":"seg_001","start":0.000,"end":0.500,"text":"Hi.","speaker":"A"},{"type":"transcript.text.segment","id":"seg_002","start":0.600,"end":1.200,"text":"Hello.","speaker":"B"},{"type":"transcript.text.segment","id":"seg_003","start":1.300,"end":1.700,"text":"Bye.","speaker":"A"},{"type":"transcript.text.segment","id":"seg_004","start":2.000,"end":3.000,"text":"Ciao","speaker":"B"}]}
    , d);
}

test "Timeline: Chinese and Japanese pieces join without a space, silence adds no segment" {
    const a = testing.allocator;
    var tl = Timeline{};
    defer tl.deinit(a);
    try tl.add(a, "你好。", "Chinese", 0, 1, null, null, 0);
    try tl.add(a, "", "None", 1, 1, null, null, 0);
    try tl.add(a, "再见。", "Chinese", 2, 1, null, null, 0);
    try testing.expectEqualStrings("你好。再见。", tl.text.items);
    try testing.expectEqual(@as(usize, 2), tl.segments.items.len);
    // An hour-long timestamp keeps its hours.
    var buf: [16]u8 = undefined;
    try testing.expectEqualStrings("01:02:03,045", clock(&buf, 3723.045, ','));
}

test "splitOutput separates the detected language from the transcript" {
    const t = splitOutput("language English<asr_text>The quick brown fox.", null);
    try testing.expectEqualStrings("English", t.language.?);
    try testing.expectEqualStrings("The quick brown fox.", t.text);
    const forced = splitOutput(" Hallo Welt.", "German");
    try testing.expectEqualStrings("German", forced.language.?);
    try testing.expectEqualStrings("Hallo Welt.", forced.text);
    // No speech: the model names no language and transcribes nothing.
    const silent = splitOutput("language None<asr_text>", null);
    try testing.expectEqualStrings("None", silent.language.?);
    try testing.expectEqualStrings("", silent.text);
}
