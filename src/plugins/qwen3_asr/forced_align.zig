//! Qwen3-ForcedAligner (`qwen3_forced_aligner` thinker in a `qwen3_asr` pack,
//! mlx-audio layout): when each word of a transcript is spoken. The official
//! `Qwen3ForceAlignProcessor` cuts the text into units, a `<timestamp>` pair
//! follows each, ONE trunk pass with the audio rows spliced in classifies every
//! pair into 80 ms steps, and the reference's monotonic repair mends outliers.
//!
//! Japanese and Korean go one character per unit, Korean's joined back into
//! its spaced words: the reference splits them with Python segmenters
//! (nagisa, a dictionary), and per-character units time the same words.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const tokenizer = @import("../../sdk/tokenizer/tokenizer.zig");
const transformer_mod = @import("../../engines/mlx/transformer.zig");
const model_mod = @import("../../sdk/config/model.zig");
const qwen_audio = @import("audio_tower.zig");
const Scope = @import("../../mlx/scope.zig").Scope;

pub const AUDIO_PROMPT = "<|audio_start|><|audio_pad|><|audio_end|>";
const PAIR = "<timestamp><timestamp>";

pub const Script = enum { spaced, japanese, korean };

pub fn scriptFor(language: []const u8) Script {
    if (std.mem.eql(u8, language, "Japanese")) return .japanese;
    if (std.mem.eql(u8, language, "Korean")) return .korean;
    return .spaced;
}

/// One aligned piece: what the model reads (`text`, the kept characters) and
/// where it is written (`start..end`, first to last kept character, so inner
/// punctuation stays: "3.5"). `word` numbers the whitespace word it came from.
pub const Unit = struct {
    text: []u8,
    start: usize,
    end: usize,
    word: usize,
};

pub fn freeUnits(a: std.mem.Allocator, us: []Unit) void {
    for (us) |u| a.free(u.text);
    a.free(us);
}

/// The reference's `is_cjk_char`.
fn isCjk(cp: u21) bool {
    return switch (cp) {
        0x4E00...0x9FFF, 0x3400...0x4DBF, 0x20000...0x2A6DF, 0x2A700...0x2B73F, 0x2B740...0x2B81F, 0x2B820...0x2CEAF, 0xF900...0xFAFF => true,
        else => false,
    };
}

/// Letters, numbers and the apostrophe (`is_kept_char`); the rest is dropped.
fn isKept(cp: u21) bool {
    return cp == '\'' or tokenizer.isLetter(cp) or tokenizer.isDigit(cp) or isCjk(cp);
}

/// Characters that are units of their own.
fn standsAlone(cp: u21, script: Script) bool {
    return isCjk(cp) or switch (script) {
        .spaced => false,
        .japanese => switch (cp) {
            0x3040...0x30FF, 0x31F0...0x31FF, 0xFF66...0xFF9F => true,
            else => false,
        },
        .korean => cp >= 0xAC00 and cp <= 0xD7A3,
    };
}

/// The reference's units of `text`: each whitespace word with all but its
/// kept characters dropped, then cut so that every character standing alone
/// is a unit and every run between them another.
pub fn units(a: std.mem.Allocator, text: []const u8, script: Script) ![]Unit {
    var out = std.ArrayList(Unit).empty;
    errdefer {
        for (out.items) |u| a.free(u.text);
        out.deinit(a);
    }
    var run = std.ArrayList(u8).empty;
    defer run.deinit(a);
    var run_start: usize = 0;
    var run_end: usize = 0;
    var word: usize = 0;
    var in_word = false;
    var i: usize = 0;
    while (i <= text.len) {
        var cp: ?u21 = null;
        var next = i + 1;
        if (i < text.len) {
            if (std.unicode.utf8ByteSequenceLength(text[i])) |len| {
                if (i + len <= text.len) {
                    if (std.unicode.utf8Decode(text[i .. i + len])) |c| {
                        cp = c;
                        next = i + len;
                    } else |_| {}
                }
            } else |_| {}
        }
        const space = i == text.len or (cp != null and tokenizer.isWhitespaceCp(cp.?));
        const alone = cp != null and isKept(cp.?) and standsAlone(cp.?, script);
        if ((space or alone) and run.items.len > 0) {
            try out.append(a, .{ .text = try run.toOwnedSlice(a), .start = run_start, .end = run_end, .word = word });
        }
        if (space) {
            if (in_word) word += 1;
            in_word = false;
        } else if (cp != null and isKept(cp.?)) {
            in_word = true;
            if (alone) {
                try out.append(a, .{ .text = try a.dupe(u8, text[i..next]), .start = i, .end = next, .word = word });
            } else {
                if (run.items.len == 0) run_start = i;
                try run.appendSlice(a, text[i..next]);
                run_end = next;
            }
        }
        i = next;
    }
    return out.toOwnedSlice(a);
}

/// The aligner's input: the audio placeholder, then each unit and its pair.
pub fn promptText(a: std.mem.Allocator, us: []const Unit) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, AUDIO_PROMPT);
    for (us) |u| {
        try out.appendSlice(a, u.text);
        try out.appendSlice(a, PAIR);
    }
    return out.toOwnedSlice(a);
}

/// Each class times `step_ms`, then the reference's repair: values off the
/// longest non-decreasing run take the nearer kept neighbour (runs of one or
/// two) or a straight line between kept ones (longer runs). Milliseconds.
pub fn fixTimestamps(a: std.mem.Allocator, classes: []const u32, step_ms: f32) ![]u32 {
    const n = classes.len;
    const v = try a.alloc(f64, n);
    defer a.free(v);
    for (v, classes) |*x, c| x.* = @as(f32, @floatFromInt(c)) * step_ms;
    const out = try a.alloc(u32, n);
    errdefer a.free(out);
    if (n == 0) return out;

    // The longest non-decreasing subsequence, with the reference's ties: the
    // first predecessor that lengthens it, the first end that is longest.
    const dp = try a.alloc(usize, n);
    defer a.free(dp);
    const parent = try a.alloc(?usize, n);
    defer a.free(parent);
    for (0..n) |i| {
        dp[i] = 1;
        parent[i] = null;
        for (0..i) |j| {
            if (v[j] <= v[i] and dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }
    const kept = try a.alloc(bool, n);
    defer a.free(kept);
    @memset(kept, false);
    var at: ?usize = std.mem.indexOfMax(usize, dp);
    while (at) |k| : (at = parent[k]) kept[k] = true;

    var i: usize = 0;
    while (i < n) {
        if (kept[i]) {
            i += 1;
            continue;
        }
        var j = i;
        while (j < n and !kept[j]) j += 1;
        const left: ?f64 = if (i > 0) v[i - 1] else null;
        const right: ?f64 = if (j < n) v[j] else null;
        for (i..j) |k| {
            if (j - i <= 2) {
                v[k] = if (left == null) right.? else if (right == null) left.? else if (k + 1 - i <= j - k) left.? else right.?;
            } else if (left != null and right != null) {
                v[k] = left.? + (right.? - left.?) / @as(f64, @floatFromInt(j - i + 1)) * @as(f64, @floatFromInt(k - i + 1));
            } else {
                v[k] = left orelse right.?;
            }
        }
        i = j;
    }
    for (out, v) |*o, x| o.* = @intFromFloat(x);
    return out;
}

/// A timed span of the transcript.
pub const Timed = struct { start_ms: u32, end_ms: u32, start: usize, end: usize };

/// The words to report, `ms` holding a start and an end per unit: each unit,
/// or for Korean each spaced word its units came from.
pub fn timedWords(a: std.mem.Allocator, us: []const Unit, ms: []const u32, script: Script) ![]Timed {
    var out = std.ArrayList(Timed).empty;
    errdefer out.deinit(a);
    for (us, 0..) |u, k| {
        const t = Timed{ .start_ms = ms[2 * k], .end_ms = ms[2 * k + 1], .start = u.start, .end = u.end };
        if (script == .korean and k > 0 and us[k - 1].word == u.word) {
            const last = &out.items[out.items.len - 1];
            last.end_ms = t.end_ms;
            last.end = t.end;
        } else try out.append(a, t);
    }
    return out.toOwnedSlice(a);
}

/// Words shown together, `start..end` their text with its punctuation, and
/// who said them when the speakers are known.
pub const Segment = struct { start_ms: u32, end_ms: u32, start: usize, end: usize, speaker: u8 = 0 };

const PAUSE_MS = 1000;
const MAX_SEGMENT_MS = 10_000;

/// Words into segments: one closes after a word whose trailing punctuation ends
/// a sentence, before a pause of a second, before it would pass ten seconds, or
/// where the speaker changes (`speakers`, one per word). Each runs to the next,
/// so every character of `text` lands in one.
pub fn segments(a: std.mem.Allocator, text: []const u8, words: []const Timed, speakers: ?[]const u8) ![]Segment {
    var out = std.ArrayList(Segment).empty;
    errdefer out.deinit(a);
    var first: usize = 0;
    var lo: usize = 0;
    for (words, 0..) |w, k| {
        const turn = if (speakers) |who| who[k] != who[k - @intFromBool(k > 0)] else false;
        if (k > first and (turn or w.start_ms -| words[k - 1].end_ms >= PAUSE_MS or w.end_ms -| words[first].start_ms > MAX_SEGMENT_MS)) {
            lo = try close(a, &out, text, words[first..k], lo, punctuationEnd(text, words[k - 1].end), speakers, first);
            first = k;
        }
        const tail = punctuationEnd(text, w.end);
        if (std.mem.indexOfAny(u8, text[w.end..tail], ".!?") != null or containsAny(text[w.end..tail], &.{ "…", "。", "！", "？" })) {
            lo = try close(a, &out, text, words[first .. k + 1], lo, tail, speakers, first);
            first = k + 1;
        }
    }
    if (first < words.len) _ = try close(a, &out, text, words[first..], lo, text.len, speakers, first);
    return out.toOwnedSlice(a);
}

/// Append the segment of `ws` (from word `first`) spanning `text[lo..hi]`
/// (trimmed); returns `hi`.
fn close(a: std.mem.Allocator, out: *std.ArrayList(Segment), text: []const u8, ws: []const Timed, lo: usize, hi: usize, speakers: ?[]const u8, first: usize) !usize {
    var s = lo;
    var e = hi;
    while (s < e and std.ascii.isWhitespace(text[s])) s += 1;
    while (e > s and std.ascii.isWhitespace(text[e - 1])) e -= 1;
    try out.append(a, .{ .start_ms = ws[0].start_ms, .end_ms = ws[ws.len - 1].end_ms, .start = s, .end = e, .speaker = if (speakers) |who| who[first] else 0 });
    return hi;
}

/// Past the punctuation that follows a word: up to the next space or kept character.
fn punctuationEnd(text: []const u8, from: usize) usize {
    var i = from;
    while (i < text.len) {
        const len = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
        if (i + len > text.len) break;
        const cp = std.unicode.utf8Decode(text[i .. i + len]) catch {
            i += 1;
            continue;
        };
        if (tokenizer.isWhitespaceCp(cp) or isKept(cp)) break;
        i += len;
    }
    return i;
}

fn containsAny(s: []const u8, needles: []const []const u8) bool {
    for (needles) |n| if (std.mem.indexOf(u8, s, n) != null) return true;
    return false;
}

/// The class of every `timestamp_token` in `ids`: one pass of the aligner's
/// trunk with `audio_rows` spliced at its audio placeholders, the head over
/// those rows alone. Inference thread.
pub fn timestampClasses(a: std.mem.Allocator, xfm: *transformer_mod.Transformer, ids: []const u32, audio_rows: mlx.mlx_array, timestamp_token: u32) ![]u32 {
    var pos = std.ArrayList(i32).empty;
    defer pos.deinit(a);
    for (ids, 0..) |t, i| if (t == timestamp_token) try pos.append(a, @intCast(i));
    const out = try a.alloc(u32, pos.items.len);
    errdefer a.free(out);
    if (pos.items.len == 0) return out;

    const ids32 = try a.alloc(i32, ids.len);
    defer a.free(ids32);
    for (ids32, ids) |*d, t| d.* = @intCast(t);
    const ids_arr = mlx.mlx_array_new_data(ids32.ptr, &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32);
    defer _ = mlx.mlx_array_free(ids_arr);

    try xfm.resetCache();
    defer xfm.resetCache() catch {};
    var ctx = xfm.defaultCtx();
    ctx.vision_embeddings = audio_rows;
    ctx.skip_lm_head = true;
    var sc = Scope.init(a, xfm.s);
    defer sc.deinit();
    const hidden = try sc.keep(try xfm.forwardWith(&ctx, ids_arr));
    var rows = mlx.mlx_array_new();
    const picked = try sc.res(mlx.mlx_take_axis(&rows, hidden, try sc.fromI32(pos.items, &.{@intCast(pos.items.len)}), 1, xfm.s), &rows);
    const logits = try sc.keep(try xfm.lmHeadForDraft(picked));
    var am = mlx.mlx_array_new();
    const classes = try sc.astype(try sc.res(mlx.mlx_argmax_axis(&am, logits, -1, false, xfm.s), &am), .uint32);
    try mlx.check(mlx.mlx_array_eval(classes));
    @memcpy(out, (mlx.mlx_array_data_uint32(classes) orelse return error.MlxError)[0..out.len]);
    return out;
}

// ── Tests ──

const testing = std.testing;

fn expectUnits(text: []const u8, script: Script, want: []const []const u8, written: []const []const u8) !void {
    const us = try units(testing.allocator, text, script);
    defer freeUnits(testing.allocator, us);
    try testing.expectEqual(want.len, us.len);
    for (us, want, written) |u, w, s| {
        try testing.expectEqualStrings(w, u.text);
        try testing.expectEqualStrings(s, text[u.start..u.end]);
    }
}

test "units: spaced words keep letters, digits and apostrophes, written as spoken" {
    try expectUnits("The fox, 3.5% — don't (stop)!", .spaced, &.{ "The", "fox", "35", "don't", "stop" }, &.{ "The", "fox", "3.5", "don't", "stop" });
    // A comma inside a word joins it, as the reference cleans before it splits.
    try expectUnits("hello,world", .spaced, &.{"helloworld"}, &.{"hello,world"});
    try expectUnits(" \t…  ", .spaced, &.{}, &.{});
}

test "units: Chinese characters stand alone, Latin runs between them stay whole" {
    try expectUnits("我爱北京，天安门。MLX很快", .spaced, &.{ "我", "爱", "北", "京", "天", "安", "门", "MLX", "很", "快" }, &.{ "我", "爱", "北", "京", "天", "安", "门", "MLX", "很", "快" });
}

test "units: Japanese kana and kanji go one per unit" {
    try expectUnits("今日はコーヒー、iPhoneを", .japanese, &.{ "今", "日", "は", "コ", "ー", "ヒ", "ー", "iPhone", "を" }, &.{ "今", "日", "は", "コ", "ー", "ヒ", "ー", "iPhone", "を" });
}

test "units: Korean syllables go one per unit and remember their word" {
    const text = "오늘은 MLX는 좋네요.";
    const us = try units(testing.allocator, text, .korean);
    defer freeUnits(testing.allocator, us);
    const want = [_][]const u8{ "오", "늘", "은", "MLX", "는", "좋", "네", "요" };
    const words = [_]usize{ 0, 0, 0, 1, 1, 2, 2, 2 };
    try testing.expectEqual(want.len, us.len);
    for (us, want, words) |u, w, n| {
        try testing.expectEqualStrings(w, u.text);
        try testing.expectEqual(n, u.word);
    }
}

test "promptText: the audio placeholder, then a timestamp pair after every unit" {
    const us = try units(testing.allocator, "The quick", .spaced);
    defer freeUnits(testing.allocator, us);
    const p = try promptText(testing.allocator, us);
    defer testing.allocator.free(p);
    try testing.expectEqualStrings("<|audio_start|><|audio_pad|><|audio_end|>The<timestamp><timestamp>quick<timestamp><timestamp>", p);
}

test "fixTimestamps matches the official repair" {
    // Expected values: qwen-asr 0.0.6 `Qwen3ForceAlignProcessor.fix_timestamp` on classes x 80.
    const cases = [_]struct { in: []const u32, out: []const u32 }{
        .{ .in = &.{ 0, 1, 1, 3, 5, 8 }, .out = &.{ 0, 80, 80, 240, 400, 640 } },
        .{ .in = &.{ 0, 2, 40, 4, 6, 8 }, .out = &.{ 0, 160, 160, 320, 480, 640 } },
        .{ .in = &.{ 10, 12, 3, 4, 14, 16 }, .out = &.{ 800, 960, 960, 1120, 1120, 1280 } },
        .{ .in = &.{ 10, 12, 90, 91, 92, 20, 22 }, .out = &.{ 800, 960, 7200, 7280, 7360, 7360, 7360 } },
        .{ .in = &.{ 50, 2, 3, 4, 5 }, .out = &.{ 160, 160, 240, 320, 400 } },
        .{ .in = &.{ 1, 2, 3, 0, 0, 0 }, .out = &.{ 80, 160, 240, 240, 240, 240 } },
        .{ .in = &.{ 5, 3, 3, 3, 7 }, .out = &.{ 240, 240, 240, 240, 560 } },
        // A run past two between kept values is a straight line.
        .{ .in = &.{ 0, 50, 51, 52, 1, 2, 3, 4 }, .out = &.{ 0, 20, 40, 60, 80, 160, 240, 320 } },
        .{ .in = &.{ 0, 90, 91, 92, 1 }, .out = &.{ 0, 7200, 7280, 7360, 7360 } },
        // Truncated like Python's int().
        .{ .in = &.{ 0, 50, 51, 52, 53, 54, 2, 3, 4, 5, 6, 7 }, .out = &.{ 0, 26, 53, 80, 106, 133, 160, 240, 320, 400, 480, 560 } },
    };
    for (cases) |c| {
        const got = try fixTimestamps(testing.allocator, c.in, 80);
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(u32, c.out, got);
    }
}

test "segments: a sentence end, a pause or ten seconds closes one, punctuation kept" {
    const text = "The quick fox. It ran away, fast! ¿Qué?";
    const us = try units(testing.allocator, text, .spaced);
    defer freeUnits(testing.allocator, us);
    const ms = [_]u32{ 0, 100, 100, 200, 200, 400, 500, 600, 600, 700, 2000, 2200, 2200, 2400, 2500, 2700 };
    const words = try timedWords(testing.allocator, us, &ms, .spaced);
    defer testing.allocator.free(words);
    const segs = try segments(testing.allocator, text, words, null);
    defer testing.allocator.free(segs);
    const want = [_]struct { []const u8, u32, u32 }{ .{ "The quick fox.", 0, 400 }, .{ "It ran", 500, 700 }, .{ "away, fast!", 2000, 2400 }, .{ "¿Qué?", 2500, 2700 } };
    try testing.expectEqual(want.len, segs.len);
    for (segs, want) |s, w| {
        try testing.expectEqualStrings(w[0], text[s.start..s.end]);
        try testing.expectEqual(w[1], s.start_ms);
        try testing.expectEqual(w[2], s.end_ms);
    }

    // Twelve one-second words with no punctuation and no pause: ten, then two.
    const long = "a b c d e f g h i j k l";
    const lu = try units(testing.allocator, long, .spaced);
    defer freeUnits(testing.allocator, lu);
    var lms: [24]u32 = undefined;
    for (0..12) |k| {
        lms[2 * k] = @intCast(k * 1000);
        lms[2 * k + 1] = @intCast(k * 1000 + 1000);
    }
    const lw = try timedWords(testing.allocator, lu, &lms, .spaced);
    defer testing.allocator.free(lw);
    const ls = try segments(testing.allocator, long, lw, null);
    defer testing.allocator.free(ls);
    try testing.expectEqual(@as(usize, 2), ls.len);
    try testing.expectEqualStrings("a b c d e f g h i j", long[ls[0].start..ls[0].end]);
    try testing.expectEqual(@as(u32, 10_000), ls[0].end_ms);
    try testing.expectEqualStrings("k l", long[ls[1].start..ls[1].end]);
}

test "segments: a change of speaker closes one too, and names its speaker" {
    const text = "Okay, so we start now";
    const us = try units(testing.allocator, text, .spaced);
    defer freeUnits(testing.allocator, us);
    const ms = [_]u32{ 0, 300, 300, 500, 600, 800, 800, 1000, 1000, 1200 };
    const words = try timedWords(testing.allocator, us, &ms, .spaced);
    defer testing.allocator.free(words);
    const segs = try segments(testing.allocator, text, words, &.{ 1, 1, 0, 0, 0 });
    defer testing.allocator.free(segs);
    try testing.expectEqual(@as(usize, 2), segs.len);
    try testing.expectEqualStrings("Okay, so", text[segs[0].start..segs[0].end]);
    try testing.expectEqual(@as(u8, 1), segs[0].speaker);
    try testing.expectEqualStrings("we start now", text[segs[1].start..segs[1].end]);
    try testing.expectEqual(@as(u8, 0), segs[1].speaker);
    try testing.expectEqual(@as(u32, 600), segs[1].start_ms);
}

// Parity vs the EXECUTED reference: mlx-audio's aligner on the pack's own
// weights (tests/dump_qwen3_aligner_fixtures.py), so a diff is a layout or
// math bug, not a different recipe.
//
//   QWEN3_ALIGNER_MODEL=<pack> QWEN3_ALIGNER_FIXTURE=<file> \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="qwen3-aligner"
test "qwen3-aligner: the prompt, the classes and the times match the reference" {
    const fix_path = envPath("QWEN3_ALIGNER_FIXTURE") orelse return error.SkipZigTest;
    const model_dir = envPath("QWEN3_ALIGNER_MODEL") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = testing.io;
    var config = try model_mod.parseConfig(io, a, model_dir);
    defer config.deinit(a);
    const fa = config.forced_aligner orelse return error.NotAnAligner;
    // The tower's tensors; the trunk is the plugin's, as the engine serves the pack.
    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    var xfm = try @import("../../engines/mlx/plugin/driver.zig").loadWith(io, a, &config, model_dir, mlx.gpuStream(), &weights);
    defer xfm.deinit();
    var qa = try qwen_audio.QwenAudio.init(a, config.qwen_audio.?, &weights);
    defer qa.deinit();
    var tok = try tokenizer.loadTokenizerAny(io, a, model_dir);
    defer tok.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();

    // Our units, prompt and tokenizer, the one audio placeholder expanded.
    const text = try fixture(u8, &fx, "text");
    const us = try units(a, text, .spaced);
    defer freeUnits(a, us);
    const prompt = try promptText(a, us);
    defer a.free(prompt);
    const short = try tok.encode(a, prompt);
    defer a.free(short);
    const want_ids = try fixture(i32, &fx, "input_ids");
    var n_audio: usize = 0;
    for (want_ids) |t| n_audio += @intFromBool(t == @as(i32, @intCast(config.audio_token_id)));
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(a);
    for (short) |t| {
        if (t == config.audio_token_id) try ids.appendNTimes(a, t, n_audio) else try ids.append(a, t);
    }
    try testing.expectEqual(want_ids.len, ids.items.len);
    for (want_ids, ids.items) |w, g| try testing.expectEqual(@as(u32, @intCast(w)), g);

    // The tower from the clip, then the trunk and head.
    const pcm = try fixture(f32, &fx, "pcm");
    const rows = try qa.forward(std.mem.sliceAsBytes(pcm));
    defer _ = mlx.mlx_array_free(rows);
    const got = try timestampClasses(a, &xfm, ids.items, rows, fa.timestamp_token_id);
    defer a.free(got);
    const want = try fixture(i32, &fx, "classes");
    try testing.expectEqual(want.len, got.len);
    var same: usize = 0;
    var worst: u32 = 0;
    for (want, got) |w, g| {
        same += @intFromBool(@as(u32, @intCast(w)) == g);
        worst = @max(worst, @as(u32, @intCast(@abs(@as(i64, w) - g))));
    }
    const times = try fixTimestamps(a, got, fa.step_ms);
    defer a.free(times);
    const want_ms = try fixture(i32, &fx, "times_ms");
    var worst_ms: u32 = 0;
    for (want_ms, times) |w, g| worst_ms = @max(worst_ms, @as(u32, @intCast(@abs(@as(i64, w) - g))));
    std.debug.print("[qwen3-aligner] {d} units: classes {d}/{d} equal, worst {d} step(s); times worst {d} ms\n", .{ us.len, same, got.len, worst, worst_ms });
    // Same weights, same math, another op order: a near-tie may land one step over.
    try testing.expect(same * 100 >= got.len * 95 and worst <= 1 and worst_ms <= @as(u32, @intFromFloat(fa.step_ms)));
}

fn envPath(name: [*:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name) orelse return null;
    const p = std.mem.sliceTo(raw, 0);
    return if (p.len == 0) null else p;
}

fn fixture(comptime T: type, fx: *model_mod.Weights, name: []const u8) ![]const T {
    const arr = fx.get(name) orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(arr));
    const data = switch (T) {
        u8 => mlx.mlx_array_data_uint8(arr),
        i32 => mlx.mlx_array_data_int32(arr),
        f32 => mlx.mlx_array_data_float32(arr),
        else => @compileError("fixture type"),
    } orelse return error.MissingFixtureTensor;
    return data[0..mlx.mlx_array_size(arr)];
}

test "timedWords: each unit, or Korean units joined into their word" {
    const text = "오늘은 좋네요";
    const us = try units(testing.allocator, text, .korean);
    defer freeUnits(testing.allocator, us);
    const ms = [_]u32{ 0, 80, 80, 160, 160, 480, 560, 640, 640, 720, 720, 960 };
    const ko = try timedWords(testing.allocator, us, &ms, .korean);
    defer testing.allocator.free(ko);
    try testing.expectEqual(@as(usize, 2), ko.len);
    try testing.expectEqualStrings("오늘은", text[ko[0].start..ko[0].end]);
    try testing.expectEqual(Timed{ .start_ms = 0, .end_ms = 480, .start = ko[0].start, .end = ko[0].end }, ko[0]);
    try testing.expectEqual(@as(u32, 560), ko[1].start_ms);
    try testing.expectEqual(@as(u32, 960), ko[1].end_ms);
    const each = try timedWords(testing.allocator, us, &ms, .spaced);
    defer testing.allocator.free(each);
    try testing.expectEqual(us.len, each.len);
    try testing.expectEqualStrings("늘", text[each[1].start..each[1].end]);
}
