//! Ming-3.0-tts request layer, ported from the vendor's pure-Python modules:
//!   `tts_tasks.py`      task aliases, captions, controls, request validation
//!   `prompt_contract.py` dense (0.5b) token layout
//!   `frontend/`          text cleaning, English number expansion, fragmenting
//!
//! The regexes are hand-translated scanners with Python `re.sub` semantics
//! (leftmost, non-overlapping, Unicode `\s`/`\w`/`\b`).
//!
//! Ceilings (documented, not silent):
//! - Chinese text normalization (vendor TalkerTN, pynini FSTs) is not ported;
//!   the Python engine also runs without it (identity normalizer), so Chinese
//!   numbers and dates are read as written in both.
//! - Unicode classes (`isalpha`, `isalnum`, `\w`) cover Latin, Greek,
//!   Cyrillic, CJK, kana and hangul; other scripts count as non-letters.
//! - No NFC normalization of input text (inputs are expected in NFC).

const std = @import("std");

pub const Cp = u21;
pub const Cps = std.ArrayList(Cp);

// ─────────────────────────────────────────────────────────────────────────
// Errors with the the reference messages (HTTP 400 bodies)
// ─────────────────────────────────────────────────────────────────────────

pub const Diag = struct {
    buf: [512]u8 = undefined,
    len: usize = 0,

    pub fn set(d: *Diag, comptime fmt: []const u8, args: anytype) error{BadRequest} {
        const s = std.fmt.bufPrint(&d.buf, fmt, args) catch d.buf[0..];
        d.len = s.len;
        return error.BadRequest;
    }
    pub fn msg(d: *const Diag) []const u8 {
        return d.buf[0..d.len];
    }
};

// ─────────────────────────────────────────────────────────────────────────
// Vocabulary (tts_tasks.py, frontend/toolkit.py, cookbooks)
// ─────────────────────────────────────────────────────────────────────────

pub const DEFAULT_TTS_PROMPT = "Please generate speech based on the following description.\n";
pub const PARALINGUISTIC_TAGS = [_][]const u8{
    "[Breathing]", "[Cough]",           "[Sigh]",        "[Laughter]",    "[Surprise-ah]", "[Surprise-oh]", "[Dissatisfaction-hnn]",
    "[Uhm]",       "[Confirmation-en]", "[Question-ei]", "[Question-ah]", "[Question-en]", "[Question-oh]", "[Question-huh]",
};
pub const EMOTION_VALUES = [_][]const u8{ "高兴", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶", "中性" };
pub const DIALECT_VALUES = [_][]const u8{ "chuanyu dialect", "cantonese" };
pub const BASIC_SPEED = [_][]const u8{ "慢速", "中速", "快速" };
pub const BASIC_LEVEL = [_][]const u8{ "低", "中", "高" };
pub const MULTILINGUAL = [_][2][]const u8{ .{ "ja", "Japanese" }, .{ "ko", "Korean" }, .{ "de", "German" }, .{ "fr", "French" }, .{ "es", "Spanish" } };
pub const EN_VOICES = [_][]const u8{ "US_Emma", "UK_George", "UK_Charlotte" };
pub const ZH_VOICES = [_][]const u8{
    "灵小安",
    "灵语汐",
    "灵小萌",
    "灵小柔",
    "灵小雅",
    "花小呗",
    "灵韵",
    "妩媚妲己",
    "灵婉",
    "灵小月",
    "灵朵儿",
    "灵小锋",
    "灵小川",
    "灵小甄",
    "灵小知",
    "灵小惠",
    "灵小晴",
    "灵小琪",
    "灵梦",
    "灵恬",
    "灵小汐",
    "灵小棠",
    "灵小梨",
    "灵小芸",
    "灵空",
    "灵宇",
    "灵朗",
    "灵楷",
    "灵旭",
    "灵江",
    "悲伤女声",
    "皮六",
    "悟空",
};

pub const VOICE_HINT = "built-in voices: US_Emma, UK_George, UK_Charlotte, or a Chinese name such as 灵小柔";

pub fn isVoice(v: []const u8) bool {
    for (EN_VOICES) |x| if (std.mem.eql(u8, x, v)) return true;
    for (ZH_VOICES) |x| if (std.mem.eql(u8, x, v)) return true;
    return false;
}

const LANG_NAMES = [_][2][]const u8{
    .{ "zh", "Chinese" },      .{ "zh-cn", "Chinese" },     .{ "chinese", "Chinese" },   .{ "en", "English" },
    .{ "english", "English" }, .{ "ja", "Japanese" },       .{ "japanese", "Japanese" }, .{ "ko", "Korean" },
    .{ "korean", "Korean" },   .{ "de", "German" },         .{ "german", "German" },     .{ "fr", "French" },
    .{ "french", "French" },   .{ "es", "Spanish" },        .{ "spanish", "Spanish" },   .{ "yue", "Chinese" },
    .{ "zh-yue", "Chinese" },  .{ "cantonese", "Chinese" },
};

// ─────────────────────────────────────────────────────────────────────────
// Unicode helpers
// ─────────────────────────────────────────────────────────────────────────

pub fn decode(a: std.mem.Allocator, s: []const u8) !Cps {
    var out: Cps = .empty;
    errdefer out.deinit(a);
    var it = (try std.unicode.Utf8View.init(s)).iterator();
    while (it.nextCodepoint()) |c| try out.append(a, c);
    return out;
}

pub fn encode(a: std.mem.Allocator, cps: []const Cp) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    for (cps) |c| {
        var b: [4]u8 = undefined;
        const n = try std.unicode.utf8Encode(c, &b);
        try out.appendSlice(a, b[0..n]);
    }
    return out.toOwnedSlice(a);
}

/// Python `str.isspace` / `re` `\s` (Unicode).
pub fn isSpace(c: Cp) bool {
    return switch (c) {
        '\t', '\n', 0x0b, 0x0c, '\r', ' ', 0x1c...0x1f, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

fn isAsciiSpace(c: Cp) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 0x0b or c == 0x0c;
}

pub fn isCjk(c: Cp) bool {
    return c >= 0x4e00 and c <= 0x9fff;
}

/// Python `str.isalpha` for the scripts the model speaks (see ceilings).
pub fn isAlpha(c: Cp) bool {
    if (c < 0x80) return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z');
    return (c >= 0xc0 and c <= 0x24f and c != 0xd7 and c != 0xf7) or c == 0xaa or c == 0xb5 or c == 0xba or
        (c >= 0x370 and c <= 0x3ff and c != 0x37e and c != 0x387) or (c >= 0x400 and c <= 0x52f) or
        (c >= 0x1e00 and c <= 0x1fff) or
        (c >= 0x3041 and c <= 0x3096) or (c >= 0x309d and c <= 0x309f) or (c >= 0x30a1 and c <= 0x30fa) or (c >= 0x30fc and c <= 0x30ff) or
        (c >= 0x3400 and c <= 0x4dbf) or (c >= 0x4e00 and c <= 0x9fff) or (c >= 0xf900 and c <= 0xfaff) or
        (c >= 0x1100 and c <= 0x11ff) or (c >= 0x3131 and c <= 0x318e) or (c >= 0xac00 and c <= 0xd7a3) or
        (c >= 0xff21 and c <= 0xff3a) or (c >= 0xff41 and c <= 0xff5a) or (c >= 0xff66 and c <= 0xffdc) or
        (c >= 0x20000 and c <= 0x2fa1f);
}

pub fn isDigit(c: Cp) bool {
    return (c >= '0' and c <= '9') or (c >= 0xff10 and c <= 0xff19);
}

fn isAsciiDigit(c: Cp) bool {
    return c >= '0' and c <= '9';
}

pub fn isAlnum(c: Cp) bool {
    return isAlpha(c) or isDigit(c);
}

/// `re` `\w` (Unicode): alphanumerics and underscore.
fn isWord(c: Cp) bool {
    return isAlnum(c) or c == '_';
}

fn isAsciiLetter(c: Cp) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z');
}

/// `\b` at position i of s.
fn boundary(s: []const Cp, i: usize) bool {
    const before = i > 0 and isWord(s[i - 1]);
    const after = i < s.len and isWord(s[i]);
    return before != after;
}

fn stripSlice(s: []const Cp) []const Cp {
    var b: usize = 0;
    var e: usize = s.len;
    while (b < e and isSpace(s[b])) b += 1;
    while (e > b and isSpace(s[e - 1])) e -= 1;
    return s[b..e];
}

fn hasCjk(s: []const Cp) bool {
    for (s) |c| if (isCjk(c)) return true;
    return false;
}

fn appendStr(a: std.mem.Allocator, out: *Cps, s: []const u8) !void {
    var it = (std.unicode.Utf8View.init(s) catch unreachable).iterator();
    while (it.nextCodepoint()) |c| try out.append(a, c);
}

fn eqlStr(s: []const Cp, lit: []const u8) bool {
    var it = (std.unicode.Utf8View.init(lit) catch return false).iterator();
    var i: usize = 0;
    while (it.nextCodepoint()) |c| : (i += 1) {
        if (i >= s.len or s[i] != c) return false;
    }
    return i == s.len;
}

fn startsWithStr(s: []const Cp, lit: []const u8) bool {
    var it = (std.unicode.Utf8View.init(lit) catch return false).iterator();
    var i: usize = 0;
    while (it.nextCodepoint()) |c| : (i += 1) {
        if (i >= s.len or s[i] != c) return false;
    }
    return true;
}

fn litLen(lit: []const u8) usize {
    return std.unicode.utf8CountCodepoints(lit) catch lit.len;
}

fn inSet(c: Cp, comptime set: []const u8) bool {
    const view = comptime blk: {
        @setEvalBranchQuota(100000);
        var cps: [set.len]Cp = undefined;
        var n: usize = 0;
        var it = (std.unicode.Utf8View.init(set) catch unreachable).iterator();
        while (it.nextCodepoint()) |x| : (n += 1) cps[n] = x;
        const final = cps[0..n].*;
        break :blk final;
    };
    for (view) |x| if (x == c) return true;
    return false;
}

/// str.replace on codepoints (all non-overlapping occurrences, left to right).
fn replaceAll(a: std.mem.Allocator, s: *Cps, from: []const u8, to: []const u8) !void {
    const fl = litLen(from);
    if (fl == 0) return;
    var out: Cps = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < s.items.len) {
        if (startsWithStr(s.items[i..], from)) {
            try appendStr(a, &out, to);
            i += fl;
        } else {
            try out.append(a, s.items[i]);
            i += 1;
        }
    }
    s.deinit(a);
    s.* = out;
}

// ─────────────────────────────────────────────────────────────────────────
// English numbers (inflect.number_to_words / ordinal)
// ─────────────────────────────────────────────────────────────────────────

const ONES = [_][]const u8{ "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };
const TENS = [_][]const u8{ "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };
const SCALES = [_][]const u8{ "", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion", "sextillion", "septillion", "octillion", "nonillion", "decillion" };

fn under100(a: std.mem.Allocator, out: *std.ArrayList(u8), n: u32) !void {
    if (n < 20) return out.appendSlice(a, ONES[n]);
    try out.appendSlice(a, TENS[n / 10]);
    if (n % 10 != 0) {
        try out.append(a, '-');
        try out.appendSlice(a, ONES[n % 10]);
    }
}

fn under1000(a: std.mem.Allocator, out: *std.ArrayList(u8), n: u32, and_word: bool) !void {
    if (n >= 100) {
        try out.appendSlice(a, ONES[n / 100]);
        try out.appendSlice(a, " hundred");
        if (n % 100 != 0) {
            try out.appendSlice(a, if (and_word) " and " else " ");
            try under100(a, out, n % 100);
        }
    } else try under100(a, out, n);
}

/// inflect `number_to_words(n)` (and_word) or `(n, andword="")`.
pub fn numberToWords(a: std.mem.Allocator, digits: []const u8, and_word: bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var d = digits;
    while (d.len > 1 and d[0] == '0') d = d[1..];
    if (d.len > 3 * SCALES.len) {
        // Beyond the scale names: read digit by digit.
        for (d, 0..) |c, i| {
            if (i > 0) try out.append(a, ' ');
            try out.appendSlice(a, ONES[c - '0']);
        }
        return out.toOwnedSlice(a);
    }
    if (d.len == 1 and d[0] == '0') {
        try out.appendSlice(a, "zero");
        return out.toOwnedSlice(a);
    }
    var groups: [SCALES.len]u32 = @splat(0);
    var ng: usize = 0;
    var end = d.len;
    while (end > 0) : (ng += 1) {
        const start = if (end >= 3) end - 3 else 0;
        groups[ng] = std.fmt.parseInt(u32, d[start..end], 10) catch 0;
        end = start;
    }
    var first = true;
    var gi = ng;
    while (gi > 0) {
        gi -= 1;
        const v = groups[gi];
        if (v == 0) continue;
        if (!first) {
            // The last group under 100 joins with "and" (or a space), others with ", ".
            if (gi == 0 and v < 100) try out.appendSlice(a, if (and_word) " and " else " ") else try out.appendSlice(a, ", ");
        }
        try under1000(a, &out, v, and_word);
        if (gi > 0) {
            try out.append(a, ' ');
            try out.appendSlice(a, SCALES[gi]);
        }
        first = false;
    }
    return out.toOwnedSlice(a);
}

/// inflect `ordinal(words)`: inflect the last word.
fn ordinalWords(a: std.mem.Allocator, words: []const u8) ![]u8 {
    var cut: usize = 0;
    for (words, 0..) |c, i| if (c == ' ' or c == '-') {
        cut = i + 1;
    };
    const last = words[cut..];
    const irregular = [_][2][]const u8{ .{ "one", "first" }, .{ "two", "second" }, .{ "three", "third" }, .{ "five", "fifth" }, .{ "eight", "eighth" }, .{ "nine", "ninth" }, .{ "twelve", "twelfth" } };
    for (irregular) |p| if (std.mem.eql(u8, last, p[0])) return std.fmt.allocPrint(a, "{s}{s}", .{ words[0..cut], p[1] });
    if (last.len > 0 and last[last.len - 1] == 'y') return std.fmt.allocPrint(a, "{s}{s}ieth", .{ words[0..cut], last[0 .. last.len - 1] });
    return std.fmt.allocPrint(a, "{s}th", .{words});
}

fn asciiDigits(s: []const Cp, buf: []u8) ?[]const u8 {
    if (s.len == 0 or s.len > buf.len) return null;
    for (s, 0..) |c, i| {
        if (!isAsciiDigit(c)) return null;
        buf[i] = @intCast(c);
    }
    return buf[0..s.len];
}

/// The shared "<int> point <d> <d>" reading (andword="", 'zero' for 0).
/// Returns null when the parts are not plain digits (Python falls back).
fn decimalWords(a: std.mem.Allocator, num: []const Cp, and_word: bool, zero_word: bool) !?[]u8 {
    var buf: [256]u8 = undefined;
    const dot = std.mem.indexOfScalar(Cp, num, '.');
    if (dot) |p| {
        const ip = if (p == 0) &[_]Cp{'0'} else num[0..p];
        const ipd = asciiDigits(ip, buf[0..128]) orelse return null;
        const dp = num[p + 1 ..];
        var buf2: [128]u8 = undefined;
        const dpd = asciiDigits(dp, &buf2) orelse return null;
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        if (zero_word and std.mem.eql(u8, ipd, "0")) try out.appendSlice(a, "zero") else {
            const w = try numberToWords(a, ipd, and_word);
            defer a.free(w);
            try out.appendSlice(a, w);
        }
        try out.appendSlice(a, " point");
        for (dpd) |c| {
            try out.append(a, ' ');
            try out.appendSlice(a, ONES[c - '0']);
        }
        return try out.toOwnedSlice(a);
    }
    const d = asciiDigits(num, &buf) orelse return null;
    return try numberToWords(a, d, and_word);
}

fn parseFloatCps(s: []const Cp) ?f64 {
    var buf: [128]u8 = undefined;
    if (s.len > buf.len) return null;
    for (s, 0..) |c, i| {
        if (c > 0x7f) return null;
        buf[i] = @intCast(c);
    }
    return std.fmt.parseFloat(f64, buf[0..s.len]) catch null;
}

/// _expand_unit / _expand_percent / _expand_dollars / _expand_pounds / _expand_number body.
fn expandAmount(a: std.mem.Allocator, out: *Cps, num: []const Cp, comptime kind: enum { number, unit, percent, money }, unit_word: []const u8, raw_unit: []const Cp, money_one: []const u8, money_many: []const u8) !void {
    const neg = num.len > 0 and num[0] == '-';
    var clean = num;
    while (clean.len > 0 and clean[0] == '-') clean = clean[1..];
    const clean_or_zero: []const Cp = if (clean.len == 0) &[_]Cp{'0'} else clean;
    const words = try decimalWords(a, clean_or_zero, false, true);
    if (words == null) {
        // Python returns the raw text around the unit word.
        try out.append(a, ' ');
        try out.appendSlice(a, if (kind == .money) clean_or_zero else num);
        try out.append(a, ' ');
        switch (kind) {
            .number => {},
            .unit => try out.appendSlice(a, raw_unit),
            .percent => try appendStr(a, out, "percent"),
            .money => try appendStr(a, out, money_many),
        }
        if (kind != .number) try out.append(a, ' ');
        return;
    }
    defer a.free(words.?);
    try out.append(a, ' ');
    if (neg) try appendStr(a, out, "minus ");
    try appendStr(a, out, words.?);
    switch (kind) {
        .number => {},
        .unit => {
            try out.append(a, ' ');
            try appendStr(a, out, unit_word);
        },
        .percent => try appendStr(a, out, " percent"),
        .money => {
            const v = parseFloatCps(clean_or_zero) orelse 0;
            try out.append(a, ' ');
            try appendStr(a, out, if (@abs(v) == 1.0) money_one else money_many);
        },
    }
    try out.append(a, ' ');
}

const UNITS = [_][2][]const u8{
    .{ "ms", "milliseconds" },          .{ "s", "seconds" },                .{ "Hz", "hertz" },            .{ "kHz", "kilohertz" },
    .{ "MHz", "megahertz" },            .{ "GHz", "gigahertz" },            .{ "kb", "kilobytes" },        .{ "mb", "megabytes" },
    .{ "gb", "gigabytes" },             .{ "tb", "terabytes" },             .{ "KB", "kilobytes" },        .{ "MB", "megabytes" },
    .{ "GB", "gigabytes" },             .{ "TB", "terabytes" },             .{ "bps", "bits per second" }, .{ "kbps", "kilobits per second" },
    .{ "Mbps", "megabits per second" }, .{ "Gbps", "gigabits per second" }, .{ "cm", "centimeters" },      .{ "km", "kilometers" },
    .{ "kg", "kilograms" },             .{ "V", "volts" },                  .{ "A", "amperes" },           .{ "W", "watts" },
    .{ "°C", "degrees celsius" },
    .{ "°F", "degrees fahrenheit" },
};

fn lowerCp(c: Cp) Cp {
    return if (c >= 'A' and c <= 'Z') c + 32 else c;
}

fn matchCaseless(s: []const Cp, i: usize, lit: []const u8) ?usize {
    var it = (std.unicode.Utf8View.init(lit) catch return null).iterator();
    var j = i;
    while (it.nextCodepoint()) |c| : (j += 1) {
        if (j >= s.len or lowerCp(s[j]) != lowerCp(c)) return null;
    }
    return j;
}

/// Run one scanner rule over `s`: at each position `rule` returns the match
/// end and appends its replacement, or null to copy one codepoint.
fn subAll(a: std.mem.Allocator, s: *Cps, ctx: anytype, comptime rule: fn (@TypeOf(ctx), std.mem.Allocator, []const Cp, usize, *Cps) anyerror!?usize) !void {
    var out: Cps = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < s.items.len) {
        if (try rule(ctx, a, s.items, i, &out)) |e| {
            i = e;
        } else {
            try out.append(a, s.items[i]);
            i += 1;
        }
    }
    s.deinit(a);
    s.* = out;
}

fn digitRun(s: []const Cp, i: usize) usize {
    var j = i;
    while (j < s.len and isAsciiDigit(s[j])) j += 1;
    return j;
}

/// `-?\d+(?:\.\d+)?` from i; returns ends with and without the fraction.
fn numberAt(s: []const Cp, i: usize, allow_minus: bool) ?struct { start: usize, with_frac: ?usize, int_end: usize } {
    var j = i;
    if (allow_minus and j < s.len and s[j] == '-') j += 1;
    const de = digitRun(s, j);
    if (de == j) return null;
    var frac: ?usize = null;
    if (de + 1 < s.len and s[de] == '.' and isAsciiDigit(s[de + 1])) frac = digitRun(s, de + 1);
    return .{ .start = i, .with_frac = frac, .int_end = de };
}

const R = struct {
    fn commas(_: void, a: std.mem.Allocator, s: []const Cp, i: usize, out: *Cps) !?usize {
        // ([0-9][0-9,]+[0-9])
        if (!isAsciiDigit(s[i])) return null;
        var j = i + 1;
        while (j < s.len and (isAsciiDigit(s[j]) or s[j] == ',')) j += 1;
        var e = j;
        while (e > i + 2 and !isAsciiDigit(s[e - 1])) e -= 1;
        if (e < i + 3 or !isAsciiDigit(s[e - 1])) return null;
        for (s[i..e]) |c| if (c != ',') try out.append(a, c);
        return e;
    }

    fn unit(_: void, a: std.mem.Allocator, s: []const Cp, i: usize, out: *Cps) !?usize {
        // \b(-?\d+(?:\.\d+)?)\s*(units)\b, IGNORECASE
        if (!boundary(s, i)) return null;
        const n = numberAt(s, i, true) orelse return null;
        const nend = n.with_frac orelse n.int_end;
        var j = nend;
        while (j < s.len and isSpace(s[j])) j += 1;
        for (UNITS) |u| {
            if (matchCaseless(s, j, u[0])) |ue| {
                if (!boundary(s, ue)) continue;
                const raw = s[j..ue];
                var key_buf: [8]Cp = undefined;
                for (raw, 0..) |c, k| key_buf[k] = lowerCp(c);
                var word: []const u8 = u[1];
                for (UNITS) |uu| { // unit.lower() lookup
                    if (matchCaseless(key_buf[0..raw.len], 0, uu[0])) |e2| if (e2 == raw.len) {
                        word = uu[1];
                        break;
                    };
                }
                try expandAmount(a, out, s[i..nend], .unit, word, raw, "", "");
                return ue;
            }
        }
        return null;
    }

    fn money(sym: Cp, a: std.mem.Allocator, s: []const Cp, i: usize, out: *Cps) !?usize {
        // £(-?[0-9,]*[0-9]+(?:\.[0-9]+)?)   $(-?[0-9.,]*[0-9]+(?:\.[0-9]+)?)
        if (s[i] != sym) return null;
        var j = i + 1;
        if (j < s.len and s[j] == '-') j += 1;
        var e = j;
        while (e < s.len and (isAsciiDigit(s[e]) or s[e] == ',' or (sym == '$' and s[e] == '.'))) e += 1;
        while (e > j and !isAsciiDigit(s[e - 1])) e -= 1;
        if (e == j) return null;
        if (sym == 0xa3 and e + 1 < s.len and s[e] == '.' and isAsciiDigit(s[e + 1])) e = digitRun(s, e + 1);
        const one: []const u8 = if (sym == '$') "dollar" else "pound";
        const many: []const u8 = if (sym == '$') "dollars" else "pounds";
        try expandAmount(a, out, s[i + 1 .. e], .money, "", &.{}, one, many);
        return e;
    }

    fn fraction(_: void, a: std.mem.Allocator, s: []const Cp, i: usize, out: *Cps) !?usize {
        // ([0-9]+)\/([0-9]+)
        const ne = digitRun(s, i);
        if (ne == i or ne >= s.len or s[ne] != '/') return null;
        const de = digitRun(s, ne + 1);
        if (de == ne + 1) return null;
        var b1: [128]u8 = undefined;
        var b2: [128]u8 = undefined;
        const num = asciiDigits(s[i..ne], &b1) orelse return null;
        const den = asciiDigits(s[ne + 1 .. de], &b2) orelse return null;
        const nv = std.fmt.parseInt(u64, num, 10) catch std.math.maxInt(u64);
        const dv = std.fmt.parseInt(u64, den, 10) catch std.math.maxInt(u64);
        const nw = try numberToWords(a, num, true);
        defer a.free(nw);
        var txt: []u8 = undefined;
        if (nv == 1 and dv == 2) {
            txt = try a.dupe(u8, " one half ");
        } else if (nv == 1 and dv == 4) {
            txt = try a.dupe(u8, " one quarter ");
        } else if (dv == 2) {
            txt = try std.fmt.allocPrint(a, " {s}{s} ", .{ nw, if (nv == 1) " half" else " halves" });
        } else if (dv == 4) {
            txt = try std.fmt.allocPrint(a, " {s}{s} ", .{ nw, if (nv == 1) " quarter" else " quarters" });
        } else {
            const dw = try numberToWords(a, den, true);
            defer a.free(dw);
            const ord = try ordinalWords(a, dw);
            defer a.free(ord);
            txt = try std.fmt.allocPrint(a, " {s} {s} ", .{ nw, ord });
        }
        defer a.free(txt);
        try appendStr(a, out, txt);
        return de;
    }

    fn percent(_: void, a: std.mem.Allocator, s: []const Cp, i: usize, out: *Cps) !?usize {
        // (-?[0-9.,]*[0-9]+)%
        var j = i;
        if (s[j] == '-') j += 1;
        var e = j;
        while (e < s.len and (isAsciiDigit(s[e]) or s[e] == ',' or s[e] == '.')) e += 1;
        // Longest prefix ending in a digit and followed by '%'.
        while (e > j) : (e -= 1) {
            if (isAsciiDigit(s[e - 1]) and e < s.len and s[e] == '%') break;
        }
        if (e == j) return null;
        try expandAmount(a, out, s[i..e], .percent, "", &.{}, "", "");
        return e + 1;
    }

    fn ordinal(_: void, a: std.mem.Allocator, s: []const Cp, i: usize, out: *Cps) !?usize {
        // \b[0-9]+(st|nd|rd|th)\b
        if (!boundary(s, i)) return null;
        const de = digitRun(s, i);
        if (de == i or de + 2 > s.len) return null;
        const suf = s[de .. de + 2];
        const ok = eqlStr(suf, "st") or eqlStr(suf, "nd") or eqlStr(suf, "rd") or eqlStr(suf, "th");
        if (!ok or !boundary(s, de + 2)) return null;
        var b: [128]u8 = undefined;
        const d = asciiDigits(s[i..de], &b) orelse return null;
        const w = try numberToWords(a, d, true);
        defer a.free(w);
        try out.append(a, ' ');
        try appendStr(a, out, w);
        try out.append(a, ' ');
        return de + 2;
    }

    fn version(_: void, a: std.mem.Allocator, s: []const Cp, i: usize, out: *Cps) !?usize {
        // \b([a-zA-Z]+)([-]?)([0-9]+(?:\.[0-9]+)?)\b
        if (!boundary(s, i) or !isAsciiLetter(s[i])) return null;
        var le = i;
        while (le < s.len and isAsciiLetter(s[le])) le += 1;
        var j = le;
        if (j < s.len and s[j] == '-') j += 1;
        const n = numberAt(s, j, false) orelse return null;
        var end: usize = undefined;
        if (n.with_frac != null and boundary(s, n.with_frac.?)) end = n.with_frac.? else if (boundary(s, n.int_end)) end = n.int_end else return null;
        const words = (try decimalWords(a, s[j..end], true, false)) orelse return null;
        defer a.free(words);
        try out.appendSlice(a, s[i..le]);
        try out.append(a, ' ');
        try appendStr(a, out, words);
        return end;
    }

    fn number(_: void, a: std.mem.Allocator, s: []const Cp, i: usize, out: *Cps) !?usize {
        // \b-?[0-9]+(?:\.[0-9]+)?\b
        if (!boundary(s, i)) return null;
        const n = numberAt(s, i, true) orelse return null;
        var end: usize = undefined;
        if (n.with_frac != null and boundary(s, n.with_frac.?)) end = n.with_frac.? else if (boundary(s, n.int_end)) end = n.int_end else return null;
        try expandAmount(a, out, s[i..end], .number, "", &.{}, "", "");
        return end;
    }
};

/// frontend/number_en.normalize_numbers
pub fn normalizeNumbers(a: std.mem.Allocator, s: *Cps) !void {
    try subAll(a, s, {}, R.commas);
    try subAll(a, s, {}, R.unit);
    try subAll(a, s, @as(Cp, 0xa3), R.money);
    try subAll(a, s, @as(Cp, '$'), R.money);
    try subAll(a, s, {}, R.fraction);
    try subAll(a, s, {}, R.percent);
    try subAll(a, s, {}, R.ordinal);
    try subAll(a, s, {}, R.version);
    try subAll(a, s, {}, R.number);
    try collapseSpaces(a, s, isSpace);
    const t = stripSlice(s.items);
    try setSlice(a, s, t);
}

fn setSlice(a: std.mem.Allocator, s: *Cps, t: []const Cp) !void {
    var out: Cps = .empty;
    try out.appendSlice(a, t);
    s.deinit(a);
    s.* = out;
}

fn collapseSpaces(a: std.mem.Allocator, s: *Cps, comptime pred: fn (Cp) bool) !void {
    var out: Cps = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < s.items.len) {
        if (pred(s.items[i])) {
            while (i < s.items.len and pred(s.items[i])) i += 1;
            try out.append(a, ' ');
        } else {
            try out.append(a, s.items[i]);
            i += 1;
        }
    }
    s.deinit(a);
    s.* = out;
}

// ─────────────────────────────────────────────────────────────────────────
// frontend/toolkit.py
// ─────────────────────────────────────────────────────────────────────────

fn isCnSpace(c: Cp) bool {
    return c == ' ' or c == '\t' or c == 0xa0 or c == 0x3000;
}

const CH_PUNCS = "，。！？；：“”‘’（）【】《》——……、";

fn isChPunc(c: Cp) bool {
    return inSet(c, CH_PUNCS);
}

/// clean_text_formatting: leading [，。！？；：、…….!?-—]+ → ' ', then strip.
fn cleanTextFormatting(a: std.mem.Allocator, s: *Cps) !void {
    var i: usize = 0;
    while (i < s.items.len and inSet(s.items[i], "，。！？；：、…" ++ ".!?-—")) i += 1;
    var out: Cps = .empty;
    errdefer out.deinit(a);
    if (i > 0) try out.append(a, ' ');
    try out.appendSlice(a, s.items[i..]);
    s.deinit(a);
    s.* = out;
    try setSlice(a, s, stripSlice(s.items));
}

/// fallback_replace_non_alnum: off-whitelist chars → '，', runs merged, strip.
fn fallbackReplace(a: std.mem.Allocator, s: *Cps) !void {
    var out: Cps = .empty;
    errdefer out.deinit(a);
    for (s.items) |c| {
        const keep = isCjk(c) or isAsciiLetter(c) or isAsciiDigit(c) or
            inSet(c, "，。！？；：、,.!?;: '\"()（）") or c == 0x2019 or c == 0x2018 or c == 0x201c or c == 0x201d;
        const r: Cp = if (keep) c else '，';
        if (r == '，' and out.items.len > 0 and out.items[out.items.len - 1] == '，' and !keep) continue;
        if (r == '，' and out.items.len > 0 and out.items[out.items.len - 1] == '，') continue;
        try out.append(a, r);
    }
    s.deinit(a);
    s.* = out;
    try setSlice(a, s, stripSlice(s.items));
}

/// One `re.sub` of a "punct class followed by punct class" collapse rule.
fn collapseRule(a: std.mem.Allocator, s: *Cps, comptime rule: enum { same, weak_strong, strong_weak, comma_dun, strong_strong, weak_weak }) !void {
    var out: Cps = .empty;
    errdefer out.deinit(a);
    const x = s.items;
    var i: usize = 0;
    const weak = struct {
        fn f(c: Cp) bool {
            return c == '，' or c == '；' or c == '、';
        }
    }.f;
    const strong = struct {
        fn f(c: Cp) bool {
            return c == '。' or c == '！' or c == '？';
        }
    }.f;
    while (i < x.len) {
        const c = x[i];
        switch (rule) {
            .same => if (isChPunc(c) and i + 1 < x.len and x[i + 1] == c) {
                var j = i + 1;
                while (j < x.len and x[j] == c) j += 1;
                try out.append(a, c);
                i = j;
                continue;
            },
            .weak_strong => if (weak(c)) {
                var j = i;
                while (j < x.len and weak(x[j])) j += 1;
                if (j < x.len and strong(x[j])) {
                    try out.append(a, x[j]);
                    i = j + 1;
                    continue;
                }
            },
            .strong_weak => if (strong(c) and i + 1 < x.len and weak(x[i + 1])) {
                var j = i + 1;
                while (j < x.len and weak(x[j])) j += 1;
                try out.append(a, c);
                i = j;
                continue;
            },
            .comma_dun => {
                // [，；]+[、] | [、][，；]+
                if (c == '，' or c == '；') {
                    var j = i;
                    while (j < x.len and (x[j] == '，' or x[j] == '；')) j += 1;
                    if (j < x.len and x[j] == '、') {
                        try out.append(a, '、');
                        i = j + 1;
                        continue;
                    }
                } else if (c == '、' and i + 1 < x.len and (x[i + 1] == '，' or x[i + 1] == '；')) {
                    var j = i + 1;
                    while (j < x.len and (x[j] == '，' or x[j] == '；')) j += 1;
                    try out.append(a, '、');
                    i = j;
                    continue;
                }
            },
            .strong_strong => if (strong(c) and i + 1 < x.len and strong(x[i + 1])) {
                var j = i + 1;
                while (j < x.len and strong(x[j])) j += 1;
                try out.append(a, c);
                i = j;
                continue;
            },
            .weak_weak => if (weak(c) and i + 1 < x.len and weak(x[i + 1])) {
                var j = i + 1;
                while (j < x.len and weak(x[j])) j += 1;
                try out.append(a, c);
                i = j;
                continue;
            },
        }
        try out.append(a, c);
        i += 1;
    }
    s.deinit(a);
    s.* = out;
}

/// chinese_punc_en2cn
fn chinesePuncEn2Cn(a: std.mem.Allocator, s: *Cps) !void {
    var has = false;
    for (s.items) |c| if (c >= 0x4e00 and c <= 0x9fa5) {
        has = true;
    };
    if (!has) return cleanTextFormatting(a, s);

    // URL placeholders.
    var urls: std.ArrayList([]Cp) = .empty;
    defer {
        for (urls.items) |u| a.free(u);
        urls.deinit(a);
    }
    {
        var i: usize = 0;
        while (i < s.items.len) {
            const x = s.items;
            const p = if (startsWithStr(x[i..], "https://")) @as(usize, 8) else if (startsWithStr(x[i..], "http://")) @as(usize, 7) else 0;
            if (p > 0) {
                var j = i + p;
                while (j < x.len and !isSpace(x[j]) and !inSet(x[j], "，。！？")) j += 1;
                if (j > i + p) {
                    try urls.append(a, try a.dupe(Cp, x[i..j]));
                    i = j;
                    continue;
                }
            }
            i += 1;
        }
    }
    for (urls.items, 0..) |u, k| {
        const from = try encode(a, u);
        defer a.free(from);
        var to_buf: [32]u8 = undefined;
        try replaceAll(a, s, from, try std.fmt.bufPrint(&to_buf, "URLHOLDER{d}", .{k}));
    }
    // Decimal points between digits survive the punctuation map.
    {
        var out: Cps = .empty;
        errdefer out.deinit(a);
        const x = s.items;
        for (x, 0..) |c, i| {
            if (c == '.' and i > 0 and isDigit(x[i - 1]) and i + 1 < x.len and isDigit(x[i + 1])) {
                try appendStr(a, &out, "DOTPLACEHOLDER");
            } else try out.append(a, c);
        }
        s.deinit(a);
        s.* = out;
    }
    try replaceAll(a, s, "...", "……");
    {
        var out: Cps = .empty;
        errdefer out.deinit(a);
        for (s.items) |c| {
            const m: ?[]const u8 = switch (c) {
                ',' => "，",
                '.' => "。",
                ';' => "；",
                ':' => "：",
                '!' => "！",
                '?' => "？",
                '(' => "（",
                ')' => "）",
                '[' => "【",
                ']' => "】",
                '<' => "《",
                '>' => "》",
                '_' => "——",
                else => null,
            };
            if (m) |r| try appendStr(a, &out, r) else try out.append(a, c);
        }
        s.deinit(a);
        s.* = out;
    }
    try replaceAll(a, s, "DOTPLACEHOLDER", ".");
    {
        var dq: usize = 0;
        var sq: usize = 0;
        for (s.items) |*c| {
            if (c.* == '"') {
                dq += 1;
                c.* = if (dq % 2 != 0) 0x201c else 0x201d;
            } else if (c.* == '\'') {
                sq += 1;
                c.* = if (sq % 2 != 0) 0x2018 else 0x2019;
            }
        }
    }
    // Drop blanks around Chinese punctuation.
    {
        var out: Cps = .empty;
        errdefer out.deinit(a);
        const x = s.items;
        var i: usize = 0;
        while (i < x.len) {
            var j = i;
            while (j < x.len and isCnSpace(x[j])) j += 1;
            if (j < x.len and isChPunc(x[j])) {
                try out.append(a, x[j]);
                var k = j + 1;
                while (k < x.len and isCnSpace(x[k])) k += 1;
                i = k;
                continue;
            }
            try out.append(a, x[i]);
            i += 1;
        }
        s.deinit(a);
        s.* = out;
    }
    while (true) {
        const before = try a.dupe(Cp, s.items);
        defer a.free(before);
        try collapseRule(a, s, .same);
        try collapseRule(a, s, .weak_strong);
        try collapseRule(a, s, .strong_weak);
        try collapseRule(a, s, .comma_dun);
        try collapseRule(a, s, .strong_strong);
        try collapseRule(a, s, .weak_weak);
        if (std.mem.eql(Cp, before, s.items)) break;
    }
    try collapseSpaces(a, s, isCnSpace);
    for (urls.items, 0..) |u, k| {
        var from_buf: [32]u8 = undefined;
        const to = try encode(a, u);
        defer a.free(to);
        try replaceAll(a, s, try std.fmt.bufPrint(&from_buf, "URLHOLDER{d}", .{k}), to);
    }
    try cleanTextFormatting(a, s);
}

/// normalize_text_kit
fn normalizeTextKit(a: std.mem.Allocator, s: *Cps) !void {
    try fallbackReplace(a, s);
    try chinesePuncEn2Cn(a, s);
    try collapseSpaces(a, s, isSpace);
    try setSlice(a, s, stripSlice(s.items));
}

/// core.py `_normalize_fragment` with the identity Chinese normalizer.
fn normalizeFragment(a: std.mem.Allocator, s: *Cps) !void {
    if (!hasCjk(s.items)) try normalizeNumbers(a, s);
    try normalizeTextKit(a, s);
    if (s.items.len > 0 and s.items[0] == '，') try setSlice(a, s, s.items[1..]);
}

/// protect_paralinguistic_tags(text, normalizeFragment)
fn protectTags(a: std.mem.Allocator, s: *Cps) !void {
    var out: Cps = .empty;
    errdefer out.deinit(a);
    const x = s.items;
    var seg_start: usize = 0;
    var any = false;
    var i: usize = 0;
    var pieces_done: usize = 0;
    while (i <= x.len) {
        var tag_len: usize = 0;
        if (i < x.len) for (PARALINGUISTIC_TAGS) |t| {
            if (startsWithStr(x[i..], t)) {
                tag_len = litLen(t);
                break;
            }
        };
        if (tag_len == 0 and i < x.len) {
            i += 1;
            continue;
        }
        if (tag_len > 0) any = true;
        // Segment before this tag (or the tail).
        const seg = x[seg_start..i];
        if (i == x.len and !any) {
            // No tags: normalize the whole text.
            var whole: Cps = .empty;
            defer whole.deinit(a);
            try whole.appendSlice(a, seg);
            try normalizeFragment(a, &whole);
            try out.appendSlice(a, whole.items);
            break;
        }
        if (stripSlice(seg).len > 0) {
            var n: Cps = .empty;
            defer n.deinit(a);
            try n.appendSlice(a, seg);
            try normalizeFragment(a, &n);
            if (isSpace(seg[0])) try out.append(a, ' ');
            try out.appendSlice(a, stripSlice(n.items));
            if (isSpace(seg[seg.len - 1])) try out.append(a, ' ');
        } else try out.appendSlice(a, seg);
        pieces_done += 1;
        if (tag_len > 0) {
            try out.appendSlice(a, x[i .. i + tag_len]);
            i += tag_len;
            seg_start = i;
        } else break;
    }
    s.deinit(a);
    s.* = out;
}

/// preprosess_text: strip ``…`` and `…` markup, then stray backticks.
fn stripMarkdownCode(a: std.mem.Allocator, s: *Cps) !void {
    inline for (.{ 2, 1 }) |ticks| {
        var out: Cps = .empty;
        errdefer out.deinit(a);
        const x = s.items;
        var i: usize = 0;
        outer: while (i < x.len) {
            if (i + ticks <= x.len and std.mem.allEqual(Cp, x[i .. i + ticks], '`')) {
                // non-greedy (.+?) then the closing run; '.' excludes newline
                var j = i + ticks + 1;
                while (j + ticks <= x.len) : (j += 1) {
                    if (x[j - 1] == '\n') break;
                    if (std.mem.allEqual(Cp, x[j .. j + ticks], '`')) {
                        try out.appendSlice(a, x[i + ticks .. j]);
                        i = j + ticks;
                        continue :outer;
                    }
                }
            }
            try out.append(a, x[i]);
            i += 1;
        }
        s.deinit(a);
        s.* = out;
    }
    var out: Cps = .empty;
    for (s.items) |c| if (c != '`') try out.append(a, c);
    s.deinit(a);
    s.* = out;
}

/// add_period
fn addPeriod(a: std.mem.Allocator, s: *Cps) !void {
    var e = s.items.len;
    while (e > 0 and isSpace(s.items[e - 1])) e -= 1;
    if (e == 0) {
        s.clearRetainingCapacity();
        try s.append(a, '.');
        return;
    }
    s.shrinkRetainingCapacity(e);
    const last = s.items[e - 1];
    if (inSet(last, "，,、;；")) {
        s.items[e - 1] = if (hasCjk(s.items)) '。' else '.';
        return;
    }
    if (!isAlnum(last)) return;
    const zh = (last >= 0x4e00 and last <= 0x9fff) or (last >= 0x3400 and last <= 0x4dbf);
    try s.append(a, if (zh) '。' else '.');
}

// ── frontend/text_segment_cut.py ──

/// get_semantic_length: CJK/kana/hangul chars + Latin words.
pub fn semanticLength(s: []const Cp) usize {
    var n: usize = 0;
    var i: usize = 0;
    const latin = struct {
        fn f(c: Cp) bool {
            return isAsciiLetter(c) or (c >= 0xc0 and c <= 0xd6) or (c >= 0xd8 and c <= 0xf6) or (c >= 0xf8 and c <= 0x24f) or (c >= 0x1e00 and c <= 0x1eff);
        }
    }.f;
    while (i < s.len) {
        const c = s[i];
        if ((c >= 0x4e00 and c <= 0x9fff) or (c >= 0x3041 and c <= 0x3096) or (c >= 0x309d and c <= 0x309f) or
            (c >= 0x30a1 and c <= 0x30fa) or (c >= 0x30fc and c <= 0x30ff) or (c >= 0xac00 and c <= 0xd7a3))
        {
            n += 1;
            i += 1;
        } else if (latin(c)) {
            while (true) {
                while (i < s.len and latin(s[i])) i += 1;
                if (i + 1 < s.len and (s[i] == '\'' or s[i] == 0x2019) and latin(s[i + 1])) {
                    i += 1;
                    continue;
                }
                break;
            }
            n += 1;
        } else i += 1;
    }
    return n;
}

fn hasValidContent(s: []const Cp) bool {
    for (s) |c| {
        const ascii_punct = (c >= 0x21 and c <= 0x2f) or (c >= 0x3a and c <= 0x40) or (c >= 0x5b and c <= 0x60) or (c >= 0x7b and c <= 0x7e);
        if (!ascii_punct and !isAsciiSpace(c)) return true;
    }
    return false;
}

const Frags = std.ArrayList(Cps);

fn freeFrags(a: std.mem.Allocator, f: *Frags) void {
    for (f.items) |*x| x.deinit(a);
    f.deinit(a);
}

fn appendFragment(a: std.mem.Allocator, frags: *Frags, new_in: []const Cp, max_len: usize, min_tail: usize) !void {
    var b: usize = 0;
    while (b < new_in.len and (inSet(new_in[b], "，,:;") or isAsciiSpace(new_in[b]))) b += 1;
    const new = new_in[b..];
    if (!hasValidContent(new)) return;
    if (frags.items.len == 0) {
        var f: Cps = .empty;
        try f.appendSlice(a, new);
        return frags.append(a, f);
    }
    const last = &frags.items[frags.items.len - 1];
    const ll = semanticLength(last.items);
    const nl = semanticLength(new);
    if (ll + nl <= max_len) {
        const lc = if (last.items.len > 0) last.items[last.items.len - 1] else 0;
        if ((lc == '。' or lc == '！' or lc == '？') and nl < min_tail) {
            var f: Cps = .empty;
            try f.appendSlice(a, new);
            return frags.append(a, f);
        }
        if (lc != ' ' and new.len > 0 and (isAsciiLetter(new[0]) or isAsciiDigit(new[0]))) try last.append(a, ' ');
        try last.appendSlice(a, new);
    } else {
        var f: Cps = .empty;
        try f.appendSlice(a, new);
        try frags.append(a, f);
    }
}

fn isClausePunct(c: Cp) bool {
    return inSet(c, "，、；。！？,;!?");
}

/// split_long_fragment
fn splitLong(a: std.mem.Allocator, text: []const Cp, max_len: usize) !Frags {
    var res: Frags = .empty;
    errdefer freeFrags(a, &res);
    if (semanticLength(text) <= max_len) {
        var f: Cps = .empty;
        try f.appendSlice(a, text);
        try res.append(a, f);
        return res;
    }
    // re.split(r'([，、；。！？,;!?]|\s+[-•·])') with separators merged back.
    var clauses: Frags = .empty;
    defer freeFrags(a, &clauses);
    var i: usize = 0;
    var start: usize = 0;
    while (i < text.len) {
        var sep_end: ?usize = null;
        if (isClausePunct(text[i])) sep_end = i + 1 else if (isSpace(text[i])) {
            var j = i;
            while (j < text.len and isSpace(text[j])) j += 1;
            if (j < text.len and (text[j] == '-' or text[j] == 0x2022 or text[j] == 0xb7)) sep_end = j + 1;
        }
        if (sep_end) |e| {
            if (i > start) {
                var f: Cps = .empty;
                try f.appendSlice(a, text[start..i]);
                try clauses.append(a, f);
            }
            if (clauses.items.len > 0) try clauses.items[clauses.items.len - 1].appendSlice(a, text[i..e]) else {
                var f: Cps = .empty;
                try f.appendSlice(a, text[i..e]);
                try clauses.append(a, f);
            }
            i = e;
            start = e;
        } else i += 1;
    }
    if (start < text.len) {
        var f: Cps = .empty;
        try f.appendSlice(a, text[start..]);
        try clauses.append(a, f);
    }
    var cur: Cps = .empty;
    defer cur.deinit(a);
    for (clauses.items) |cl| {
        const cl_len = semanticLength(cl.items);
        if (cl_len > max_len) {
            if (cur.items.len > 0) {
                try res.append(a, cur);
                cur = .empty;
            }
            var f: Cps = .empty;
            try f.appendSlice(a, cl.items);
            try res.append(a, f);
            continue;
        }
        if (semanticLength(cur.items) + cl_len <= max_len) {
            try cur.appendSlice(a, cl.items);
        } else {
            if (cur.items.len > 0) try res.append(a, cur) else cur.deinit(a);
            cur = .empty;
            try cur.appendSlice(a, cl.items);
        }
    }
    if (cur.items.len > 0) {
        try res.append(a, cur);
        cur = .empty;
    }
    return res;
}

const DOT_PH = "##DOT##";

/// cut_text_by_semantic_length(...)["fragments"]
pub fn cutText(a: std.mem.Allocator, text_in: []const Cp, max_len: usize, min_tail: usize) !Frags {
    var res: Frags = .empty;
    errdefer freeFrags(a, &res);
    if (!hasValidContent(text_in)) return res;
    var p: Cps = .empty;
    defer p.deinit(a);
    {
        // (\d)\.(\d) → \1##DOT##\2, non-overlapping.
        var i: usize = 0;
        const x = text_in;
        while (i < x.len) {
            if (isDigit(x[i]) and i + 2 < x.len and x[i + 1] == '.' and isDigit(x[i + 2])) {
                try p.append(a, x[i]);
                try appendStr(a, &p, DOT_PH);
                try p.append(a, x[i + 2]);
                i += 3;
            } else {
                try p.append(a, x[i]);
                i += 1;
            }
        }
    }
    for (0..3) |_| {
        var out: Cps = .empty;
        errdefer out.deinit(a);
        var i: usize = 0;
        const x = p.items;
        while (i < x.len) {
            const up = struct {
                fn f(c: Cp) bool {
                    return c >= 'A' and c <= 'Z';
                }
            }.f;
            if (up(x[i]) and i + 2 < x.len and x[i + 1] == '.' and up(x[i + 2])) {
                try out.append(a, x[i]);
                try appendStr(a, &out, DOT_PH);
                try out.append(a, x[i + 2]);
                i += 3;
            } else {
                try out.append(a, x[i]);
                i += 1;
            }
        }
        p.deinit(a);
        p = out;
    }
    try replaceAll(a, &p, "\n", " ");
    try replaceAll(a, &p, "。，", "。");
    if (semanticLength(p.items) <= max_len) {
        var f: Cps = .empty;
        try f.appendSlice(a, p.items);
        try replaceAll(a, &f, DOT_PH, ".");
        try res.append(a, f);
        return res;
    }
    for (p.items) |*c| c.* = switch (c.*) {
        '.' => '。',
        '!' => '！',
        '?' => '？',
        ',' => '，',
        else => c.*,
    };
    // Sentences.
    var sentences: Frags = .empty;
    defer freeFrags(a, &sentences);
    {
        var cur: Cps = .empty;
        defer cur.deinit(a);
        for (p.items) |c| {
            try cur.append(a, c);
            if (c == '。' or c == '！' or c == '？') {
                const st = stripSlice(cur.items);
                if (st.len > 0) {
                    var f: Cps = .empty;
                    try f.appendSlice(a, st);
                    try sentences.append(a, f);
                }
                cur.clearRetainingCapacity();
            }
        }
        if (cur.items.len > 0) {
            var f: Cps = .empty;
            try f.appendSlice(a, stripSlice(cur.items));
            const lc = if (f.items.len > 0) f.items[f.items.len - 1] else 0;
            if (!(lc == '。' or lc == '！' or lc == '？')) try f.append(a, '。');
            try sentences.append(a, f);
        }
    }
    for (sentences.items) |sent| {
        var clauses: Frags = .empty;
        defer freeFrags(a, &clauses);
        var cur: Cps = .empty;
        defer cur.deinit(a);
        for (sent.items) |c| {
            try cur.append(a, c);
            if (c == '，' or c == ';' or c == '；') {
                const cl = stripSlice(cur.items);
                if (cl.len > 0 and hasValidContent(cl)) {
                    var f: Cps = .empty;
                    try f.appendSlice(a, cl);
                    try clauses.append(a, f);
                } else if (cl.len > 0 and clauses.items.len > 0) {
                    try clauses.items[clauses.items.len - 1].appendSlice(a, cl);
                }
                cur.clearRetainingCapacity();
            }
        }
        if (cur.items.len > 0) {
            const cl = stripSlice(cur.items);
            if (cl.len > 0 and hasValidContent(cl)) {
                var f: Cps = .empty;
                try f.appendSlice(a, cl);
                try clauses.append(a, f);
            } else if (cl.len > 0 and clauses.items.len > 0) {
                try clauses.items[clauses.items.len - 1].appendSlice(a, cl);
            }
        }
        var i: usize = 0;
        while (i < clauses.items.len) {
            const cl = clauses.items[i].items;
            const cl_len = semanticLength(cl);
            if (cl_len < min_tail and i + 1 < clauses.items.len) {
                var comb: Cps = .empty;
                defer comb.deinit(a);
                try comb.appendSlice(a, cl);
                try comb.appendSlice(a, clauses.items[i + 1].items);
                if (semanticLength(comb.items) <= max_len) {
                    try appendFragment(a, &res, comb.items, max_len, min_tail);
                    i += 2;
                    continue;
                }
            }
            if (cl_len > max_len) {
                var subs = try splitLong(a, cl, max_len);
                defer freeFrags(a, &subs);
                for (subs.items) |sf| try appendFragment(a, &res, sf.items, max_len, min_tail);
            } else try appendFragment(a, &res, cl, max_len, min_tail);
            i += 1;
        }
    }
    for (res.items) |*f| try replaceAll(a, f, DOT_PH, ".");
    return res;
}

/// Python `str.strip()` + `not s` emptiness on UTF-8.
pub fn isBlank(s: []const u8) bool {
    var it = (std.unicode.Utf8View.init(s) catch return s.len == 0).iterator();
    while (it.nextCodepoint()) |c| if (!isSpace(c)) return false;
    return true;
}

pub fn strip(s: []const u8) []const u8 {
    var b: usize = 0;
    var e: usize = s.len;
    // Strip whole codepoints from both ends.
    while (b < e) {
        const n = std.unicode.utf8ByteSequenceLength(s[b]) catch break;
        if (b + n > e) break;
        const c = std.unicode.utf8Decode(s[b .. b + n]) catch break;
        if (!isSpace(c)) break;
        b += n;
    }
    while (e > b) {
        var st = e - 1;
        while (st > b and (s[st] & 0xc0) == 0x80) st -= 1;
        const c = std.unicode.utf8Decode(s[st..e]) catch break;
        if (!isSpace(c)) break;
        e = st;
    }
    return s[b..e];
}

/// tts_tasks.preserve_multilingual_text
pub fn preserveMultilingual(language_id: ?[]const u8) bool {
    const l = language_id orelse return false;
    const t = strip(l);
    if (t.len == 0) return false;
    var buf: [32]u8 = undefined;
    if (t.len > buf.len) return true;
    for (t, 0..) |c, i| buf[i] = if (c == '_') '-' else std.ascii.toLower(c);
    const low = buf[0..t.len];
    for ([_][]const u8{ "zh", "zh-cn", "chinese", "en", "english" }) |x| if (std.mem.eql(u8, low, x)) return false;
    return true;
}

pub fn isFrench(language_id: ?[]const u8) bool {
    const l = strip(language_id orelse return false);
    return std.ascii.eqlIgnoreCase(l, "fr") or std.ascii.eqlIgnoreCase(l, "french") or std.ascii.startsWithIgnoreCase(l, "fr-") or std.ascii.startsWithIgnoreCase(l, "fr_");
}

/// MULTILINGUAL fr/de/es: the cutter rewrites the punctuation it splits on
/// to the Chinese full-width forms ("Bonjour à tous，voici …。"), as the reference
/// does. Put the ASCII forms back (Latin-script languages only; Japanese
/// keeps 、。).
fn isLatinMultilingual(language_id: ?[]const u8) bool {
    const l = strip(language_id orelse return false);
    for ([_][]const u8{ "fr", "french", "de", "german", "es", "spanish" }) |x| if (std.ascii.eqlIgnoreCase(l, x)) return true;
    return isFrench(language_id);
}

fn restoreAsciiPunct(a: std.mem.Allocator, f: *std.ArrayList(Cp)) !void {
    var out: std.ArrayList(Cp) = .empty;
    errdefer out.deinit(a);
    for (f.items, 0..) |c, i| {
        const ascii: ?Cp = switch (c) {
            0xFF0C => ',', // ，
            0x3002 => '.', // 。
            0xFF01 => '!',
            0xFF1F => '?',
            0xFF1B => ';',
            0xFF1A => ':',
            else => null,
        };
        try out.append(a, ascii orelse c);
        // "tous，voici" → "tous, voici"
        if (ascii != null and i + 1 < f.items.len and !isSpace(f.items[i + 1])) try out.append(a, ' ');
    }
    f.deinit(a);
    f.* = out;
}

/// MingMLX.prepare_text: text → synthesizable fragments (owned UTF-8).
pub fn prepareText(a: std.mem.Allocator, text_in: []const u8, language_id: ?[]const u8, max_len: usize) !std.ArrayList([]u8) {
    const fr_text: ?[]u8 = if (isFrench(language_id)) try @import("fr.zig").normalize(a, text_in) else null;
    defer if (fr_text) |t| a.free(t);
    const text_utf8 = fr_text orelse text_in;
    var s = try decode(a, text_utf8);
    defer s.deinit(a);
    try stripMarkdownCode(a, &s);
    const multilingual = preserveMultilingual(language_id);
    if (multilingual) try setSlice(a, &s, stripSlice(s.items)) else try addPeriod(a, &s);
    var frags = try cutText(a, s.items, max_len, 5);
    defer freeFrags(a, &frags);
    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |x| a.free(x);
        out.deinit(a);
    }
    const latin = isLatinMultilingual(language_id);
    for (frags.items) |*f| {
        if (stripSlice(f.items).len == 0) continue;
        if (latin) try restoreAsciiPunct(a, f);
        if (!multilingual) try protectTags(a, f);
        if (stripSlice(f.items).len == 0) continue;
        try out.append(a, try encode(a, f.items));
    }
    if (out.items.len == 0) return error.NoFragments;
    return out;
}

/// What separates a fragment from the next one; `Engine.synthesize` picks
/// the pause length from it.
pub const Boundary = enum { none, clause, sentence, paragraph };

pub const Fragment = struct {
    text: []u8,
    /// The boundary after this fragment (`.none` for the last one).
    after: Boundary,
};

pub fn freeFragments(a: std.mem.Allocator, frags: *std.ArrayList(Fragment)) void {
    for (frags.items) |f| a.free(f.text);
    frags.deinit(a);
}

/// `prepareText` for long text: split into paragraphs first, run
/// `prepareText` on each, and tag every fragment with the boundary that
/// follows it. A line break starts a new paragraph unless the line does not
/// end a sentence and the next line starts with a lowercase letter (a hard
/// wrap, joined with a space). Text without line breaks gives the same
/// fragments as `prepareText`. Departs from the reference, which turns line breaks
/// into spaces.
pub fn prepareLongText(a: std.mem.Allocator, text_in: []const u8, language_id: ?[]const u8, max_len: usize) !std.ArrayList(Fragment) {
    var out: std.ArrayList(Fragment) = .empty;
    errdefer freeFragments(a, &out);
    var paras = try splitParagraphs(a, text_in);
    defer {
        for (paras.items) |p| a.free(p);
        paras.deinit(a);
    }
    for (paras.items) |p| {
        var frags = prepareText(a, p, language_id, max_len) catch |e| switch (e) {
            error.NoFragments => continue, // e.g. a line of dashes
            else => return e,
        };
        defer frags.deinit(a);
        if (out.items.len > 0) out.items[out.items.len - 1].after = .paragraph;
        out.ensureUnusedCapacity(a, frags.items.len) catch |e| {
            for (frags.items) |f| a.free(f);
            return e;
        };
        for (frags.items) |f| out.appendAssumeCapacity(.{ .text = f, .after = if (endsSentence(f)) .sentence else .clause });
    }
    if (out.items.len == 0) return error.NoFragments;
    out.items[out.items.len - 1].after = .none;
    return out;
}

/// Paragraphs of `s` (owned UTF-8, non-blank), see `prepareLongText`.
fn splitParagraphs(a: std.mem.Allocator, s: []const u8) !std.ArrayList([]u8) {
    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |p| a.free(p);
        out.deinit(a);
    }
    var cur: std.ArrayList(u8) = .empty;
    defer cur.deinit(a);
    var lines = std.mem.splitAny(u8, s, "\n\r");
    while (lines.next()) |raw| {
        const line = strip(raw);
        if (line.len == 0) {
            if (cur.items.len > 0) try out.append(a, try cur.toOwnedSlice(a));
            continue;
        }
        if (cur.items.len > 0) {
            const hard_wrap = !endsSentence(cur.items) and std.ascii.isLower(line[0]);
            if (hard_wrap) try cur.append(a, ' ') else try out.append(a, try cur.toOwnedSlice(a));
        }
        try cur.appendSlice(a, line);
    }
    if (cur.items.len > 0) try out.append(a, try cur.toOwnedSlice(a));
    return out;
}

/// Ends with sentence punctuation, ignoring trailing spaces, closing quotes
/// and brackets.
fn endsSentence(s: []const u8) bool {
    var e = s.len;
    while (e > 0) {
        var st = e - 1;
        while (st > 0 and (s[st] & 0xc0) == 0x80) st -= 1;
        const c = std.unicode.utf8Decode(s[st..e]) catch return false;
        if (isSpace(c) or inSet(c, "\"')]}»”’」』）")) {
            e = st;
            continue;
        }
        return inSet(c, ".!?。！？…");
    }
    return false;
}

/// core.spoken_units: Latin words + every digit and non-ASCII letter.
pub fn spokenUnits(text_utf8: []const u8) usize {
    var it = (std.unicode.Utf8View.init(text_utf8) catch return 0).iterator();
    var n: usize = 0;
    var in_word = false;
    while (it.nextCodepoint()) |c| {
        const latin = isAsciiLetter(c);
        if (latin and !in_word) n += 1;
        in_word = latin;
        if (isDigit(c) or (c > 0x7f and isAlpha(c))) n += 1;
    }
    return n;
}

// ─────────────────────────────────────────────────────────────────────────
// tts_tasks.build_task_generation_kwargs
// ─────────────────────────────────────────────────────────────────────────

pub const Controls = struct {
    speed: ?[]const u8 = null,
    pitch: ?[]const u8 = null,
    volume: ?[]const u8 = null,
    emotion: ?[]const u8 = null,
    dialect: ?[]const u8 = null,
    style: ?[]const u8 = null,
    ip: ?[]const u8 = null,

    pub fn isEmpty(c: Controls) bool {
        inline for (@typeInfo(Controls).@"struct".field_names) |f| if (@field(c, f) != null) return false;
        return true;
    }
};

pub const TaskInput = struct {
    task: []const u8,
    text: []const u8,
    language_id: ?[]const u8 = null,
    instruction: ?[]const u8 = null,
    controls: Controls = .{},
    /// Control keys outside `Controls` (refused like the reference).
    unknown_controls: []const []const u8 = &.{},
    prompt_wav: ?[]const u8 = null,
    prompt_text: ?[]const u8 = null,
    use_spk_emb: bool = false,
    use_zero_spk_emb: bool = false,
    cfg: f32 = 2.0,
    sigma: f32 = 0.25,
    temperature: f32 = 0.0,
};

pub const Task = enum { TTS, BASIC, EMOTION, DIALECT, IP, STYLE };

/// Resolved request (owned strings).
pub const Request = struct {
    prompt: []u8,
    text: []u8,
    instruction: ?[]u8,
    use_spk_emb: bool,
    use_zero_spk_emb: bool,
    prompt_wav: ?[]const u8,
    prompt_text: ?[]const u8,
    cfg: f32,
    sigma: f32,
    temperature: f32,
    taskname: Task,
    text_language: ?[]const u8,

    pub fn deinit(r: *const Request, a: std.mem.Allocator) void {
        a.free(r.prompt);
        a.free(r.text);
        if (r.instruction) |x| a.free(x);
    }
};

const TASK_ALIASES = [_]struct { []const u8, Task }{
    .{ "TTS", .TTS },     .{ "SEED", .TTS },        .{ "PLAIN", .TTS },       .{ "MULTILINGUAL", .TTS }, .{ "MULTILINGUAL_TTS", .TTS },
    .{ "BASIC", .BASIC }, .{ "EMOTION", .EMOTION }, .{ "DIALECT", .DIALECT }, .{ "IP", .IP },            .{ "VOICE_DESIGN", .STYLE },
    .{ "STYLE", .STYLE }, .{ "FREEFORM", .STYLE },
};

/// build_tts_prompt
pub fn buildTtsPrompt(a: std.mem.Allocator, target_language: ?[]const u8) ![]u8 {
    const l = if (target_language) |x| strip(x) else "";
    if (l.len == 0) return a.dupe(u8, DEFAULT_TTS_PROMPT);
    var name: []const u8 = l;
    var low_buf: [64]u8 = undefined;
    if (l.len <= low_buf.len) {
        const low = std.ascii.lowerString(&low_buf, l);
        for (LANG_NAMES) |p| if (std.mem.eql(u8, p[0], low)) {
            name = p[1];
        };
    }
    return std.fmt.allocPrint(a, "{s}Target language: {s}.\n", .{ DEFAULT_TTS_PROMPT, name });
}

pub fn jsonStr(a: std.mem.Allocator, out: *std.ArrayList(u8), v: ?[]const u8) !void {
    const s = v orelse return out.appendSlice(a, "null");
    try out.append(a, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(a, "\\\""),
        '\\' => try out.appendSlice(a, "\\\\"),
        '\n' => try out.appendSlice(a, "\\n"),
        '\r' => try out.appendSlice(a, "\\r"),
        '\t' => try out.appendSlice(a, "\\t"),
        0x08 => try out.appendSlice(a, "\\b"),
        0x0c => try out.appendSlice(a, "\\f"),
        0...0x07, 0x0b, 0x0e...0x1f, 0x7f => {
            var b: [8]u8 = undefined;
            try out.appendSlice(a, try std.fmt.bufPrint(&b, "\\u{x:0>4}", .{c}));
        },
        else => try out.append(a, c),
    };
    try out.append(a, '"');
}

/// build_structured_instruction: the trained JSON caption, key order fixed.
pub fn buildStructuredInstruction(a: std.mem.Allocator, c: Controls, zh_schema: bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const fields: [8][]const u8 = if (zh_schema)
        .{ "序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感" }
    else
        .{ "id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion" };
    const vals = [_]?[]const u8{ c.dialect, c.style, c.speed, c.pitch, c.volume, c.emotion };
    try out.appendSlice(a, "{\"audio_sequence\": [{");
    try out.appendSlice(a, "\"");
    try out.appendSlice(a, fields[0]);
    try out.appendSlice(a, "\": 1, \"");
    try out.appendSlice(a, fields[1]);
    try out.appendSlice(a, "\": \"speaker_1\"");
    for (vals, 2..) |v, i| {
        try out.appendSlice(a, ", \"");
        try out.appendSlice(a, fields[i]);
        try out.appendSlice(a, "\": ");
        try jsonStr(a, &out, v);
    }
    try out.appendSlice(a, ", \"BGM\": {\"Genre\": null, \"Mood\": null, \"Instrument\": null, \"Theme\": null, \"ENV\": null, \"SNR\": null}, \"");
    try out.appendSlice(a, if (zh_schema) "IP" else "ip");
    try out.appendSlice(a, "\": ");
    try jsonStr(a, &out, c.ip);
    try out.appendSlice(a, "}]}");
    return out.toOwnedSlice(a);
}

fn normalizeInstruction(a: std.mem.Allocator, instr: ?[]const u8) !?[]u8 {
    const x = instr orelse return null;
    const t = strip(x);
    if (t.len == 0) return null;
    return try a.dupe(u8, t);
}

fn upperTask(buf: []u8, task: []const u8) []const u8 {
    const t = strip(task);
    const n = @min(t.len, buf.len);
    for (t[0..n], 0..) |c, i| buf[i] = if (c == '-') '_' else std.ascii.toUpper(c);
    return buf[0..n];
}

/// Python repr of a sorted list of strings, e.g. ['emotion', 'ip'].
fn reprList(buf: []u8, items: []const []const u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.writeAll("[") catch {};
    for (items, 0..) |it, i| {
        if (i > 0) w.writeAll(", ") catch {};
        w.print("'{s}'", .{it}) catch {};
    }
    w.writeAll("]") catch {};
    return w.buffered();
}

/// tts_tasks.build_task_generation_kwargs (raises → error.BadRequest + Diag).
pub fn buildRequest(a: std.mem.Allocator, in: TaskInput, d: *Diag) !Request {
    var tbuf: [32]u8 = undefined;
    const task = upperTask(&tbuf, in.task);
    var runtime: ?Task = null;
    for (TASK_ALIASES) |p| if (std.mem.eql(u8, p[0], task)) {
        runtime = p[1];
    };
    if (runtime == null) return d.set("Unsupported Ming TTS task '{s}'; expected one of ['BASIC', 'DIALECT', 'EMOTION', 'FREEFORM', 'IP', 'MULTILINGUAL', 'MULTILINGUAL_TTS', 'PLAIN', 'SEED', 'STYLE', 'TTS', 'VOICE_DESIGN']", .{task});
    var language_id = in.language_id;
    if (std.mem.eql(u8, task, "DIALECT")) language_id = null; // the reference warns and drops it
    if ((std.mem.eql(u8, task, "MULTILINGUAL") or std.mem.eql(u8, task, "MULTILINGUAL_TTS")) and (language_id == null or isBlank(language_id.?)))
        return d.set("MULTILINGUAL requires language_id", .{});
    const text_s = strip(in.text);
    if (text_s.len == 0) return d.set("text must not be empty", .{});
    if (in.use_spk_emb and in.use_zero_spk_emb) return d.set("use_spk_emb and use_zero_spk_emb cannot both be true", .{});
    if (in.use_spk_emb and in.prompt_wav == null) return d.set("use_spk_emb requires a prompt wav or a speaker embedding", .{});

    const has_controls = !in.controls.isEmpty() or in.unknown_controls.len > 0;
    var instruction: ?[]u8 = null;
    errdefer if (instruction) |x| a.free(x);
    const structured = runtime.? != .TTS and !std.mem.eql(u8, task, "FREEFORM");
    if (std.mem.eql(u8, task, "FREEFORM")) {
        if (has_controls) return d.set("FREEFORM uses instruction, not controls", .{});
        instruction = try normalizeInstruction(a, in.instruction);
        if (instruction == null) return d.set("FREEFORM requires a non-empty instruction", .{});
    } else if (structured) {
        if (in.instruction != null) return d.set("{s} builds its instruction from controls; do not pass instruction", .{task});
        if (!has_controls) return d.set("{s} requires task controls", .{task});
        const prompt_task: []const u8 = if (std.mem.eql(u8, task, "STYLE")) "VOICE_DESIGN" else task;
        // build_task_instruction: refuse controls outside the task's set.
        var unknown: [16][]const u8 = undefined;
        var nu: usize = 0;
        const c = in.controls;
        const allow = struct { sp: bool = false, pi: bool = false, vo: bool = false, em: bool = false, di: bool = false, st: bool = false, ip: bool = false }{
            .sp = runtime.? == .BASIC,
            .pi = runtime.? == .BASIC,
            .vo = runtime.? == .BASIC,
            .em = runtime.? == .EMOTION,
            .di = runtime.? == .DIALECT,
            .st = runtime.? == .STYLE,
            .ip = runtime.? == .IP,
        };
        if (c.dialect != null and !allow.di) {
            unknown[nu] = "dialect";
            nu += 1;
        }
        if (c.emotion != null and !allow.em) {
            unknown[nu] = "emotion";
            nu += 1;
        }
        if (c.ip != null and !allow.ip) {
            unknown[nu] = "ip";
            nu += 1;
        }
        if (c.pitch != null and !allow.pi) {
            unknown[nu] = "pitch";
            nu += 1;
        }
        if (c.speed != null and !allow.sp) {
            unknown[nu] = "speed";
            nu += 1;
        }
        if (c.style != null and !allow.st) {
            unknown[nu] = "style";
            nu += 1;
        }
        if (c.volume != null and !allow.vo) {
            unknown[nu] = "volume";
            nu += 1;
        }
        for (in.unknown_controls) |k| if (nu < unknown.len) {
            unknown[nu] = k;
            nu += 1;
        };
        if (nu > 0) {
            std.mem.sort([]const u8, unknown[0..nu], {}, struct {
                fn lt(_: void, x: []const u8, y: []const u8) bool {
                    return std.mem.order(u8, x, y) == .lt;
                }
            }.lt);
            var rb: [256]u8 = undefined;
            return d.set("{s} does not accept control fields: {s}", .{ prompt_task, reprList(&rb, unknown[0..nu]) });
        }
        instruction = try buildStructuredInstruction(a, c, runtime.? != .DIALECT);
    } else {
        if (has_controls) return d.set("{s} does not accept controls", .{task});
        instruction = try normalizeInstruction(a, in.instruction);
    }
    // Text without a language that reads as fr/de/es/ja/ko takes the MULTILINGUAL text path: no zh/en normalization
    // (which spells numbers in English and turns accented letters into "，"), and the French normalizer for fr. The
    // "Target language" line joins only where the voice comes from a clip or from none: on IP, FREEFORM and VOICE_DESIGN
    // it moves the captioned voice, and the reference keeps DIALECT languageless.
    const detected = if (language_id != null and !isBlank(language_id.?)) null else detectLanguage(text_s);
    const text_language = detected orelse language_id;
    const line = runtime.? == .TTS or runtime.? == .BASIC or runtime.? == .EMOTION;
    const prompt = try buildTtsPrompt(a, if (line and detected != null) detected else language_id);
    errdefer a.free(prompt);
    return .{
        .prompt = prompt,
        .text = try a.dupe(u8, text_s),
        .instruction = instruction,
        .use_spk_emb = in.use_spk_emb,
        .use_zero_spk_emb = in.use_zero_spk_emb,
        .prompt_wav = in.prompt_wav,
        .prompt_text = in.prompt_text,
        .cfg = in.cfg,
        .sigma = in.sigma,
        .temperature = in.temperature,
        .taskname = runtime.?,
        .text_language = text_language,
    };
}

// ─────────────────────────────────────────────────────────────────────────
// prompt_contract.build_input_ids (dense)
// ─────────────────────────────────────────────────────────────────────────

pub const TOK_SPEAKER_START: u32 = 151652; // <|vision_start|>
pub const TOK_SPEAKER_PAD: u32 = 151654; // <|vision_pad|>
pub const TOK_AUDIO_PATCH: u32 = 151705; // <audioPatch>
pub const TOK_AUDIO: u32 = 151706; // <audio>

pub const InputIds = struct {
    ids: []i32,
    speaker_positions: []usize,
    latent_position: ?usize,

    pub fn deinit(x: *const InputIds, a: std.mem.Allocator) void {
        a.free(x.ids);
        a.free(x.speaker_positions);
    }
};

/// `encode` is any tokenizer with `encode(allocator, text) ![]u32` (no BOS/EOS).
// ── NFC (the tokenizer's normalizer) ──

const nfc_table = @import("nfc_table.zig").pairs;

fn composePair(base: u21, mark: u21) ?u21 {
    // Hangul: L + V → LV, LV + T → LVT (Unicode §3.12).
    const S_BASE = 0xAC00;
    const L_BASE = 0x1100;
    const V_BASE = 0x1161;
    const T_BASE = 0x11A7;
    if (base >= L_BASE and base < L_BASE + 19 and mark >= V_BASE and mark < V_BASE + 21)
        return S_BASE + ((base - L_BASE) * 21 + (mark - V_BASE)) * 28;
    if (base >= S_BASE and base < S_BASE + 11172 and (base - S_BASE) % 28 == 0 and mark > T_BASE and mark < T_BASE + 28)
        return base + (mark - T_BASE);
    var lo: usize = 0;
    var hi: usize = nfc_table.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const p = nfc_table[mid];
        if (p.base == base and p.mark == mark) return p.composed;
        if (p.base < base or (p.base == base and p.mark < mark)) lo = mid + 1 else hi = mid;
    }
    return null;
}

/// Unicode NFC for text that is NFC already or canonically decomposed
/// (NFD), which is what keyboards and macOS paste produce. Caller frees.
/// Ceiling: no canonical reordering of several marks on one base and no
/// decomposition of precomposed input, so a non-canonical mark order or
/// mixed precomposed + combining text can differ from full NFC. Upgrade
/// path: generate the decomposition and combining-class tables too.
pub fn nfc(a: std.mem.Allocator, s: []const u8) ![]u8 {
    var view = std.unicode.Utf8View.init(s) catch return a.dupe(u8, s);
    var it = view.iterator();
    var cps: std.ArrayList(u21) = .empty;
    defer cps.deinit(a);
    var starter: ?usize = null;
    var changed = false;
    while (it.nextCodepoint()) |cp| {
        if (starter) |i| if (composePair(cps.items[i], cp)) |c| {
            cps.items[i] = c;
            changed = true;
            continue;
        };
        try cps.append(a, cp);
        if (!isCombiningMark(cp)) starter = cps.items.len - 1;
    }
    if (!changed) return a.dupe(u8, s);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    for (cps.items) |cp| {
        var buf: [4]u8 = undefined;
        const n = try std.unicode.utf8Encode(cp, &buf);
        try out.appendSlice(a, buf[0..n]);
    }
    return out.toOwnedSlice(a);
}

fn isCombiningMark(cp: u21) bool {
    return (cp >= 0x0300 and cp <= 0x036F) or (cp >= 0x1AB0 and cp <= 0x1AFF) or (cp >= 0x1DC0 and cp <= 0x1DFF) or
        (cp >= 0x20D0 and cp <= 0x20FF) or (cp >= 0xFE20 and cp <= 0xFE2F) or cp == 0x3099 or cp == 0x309A or
        (cp >= 0x0591 and cp <= 0x05C7) or (cp >= 0x0610 and cp <= 0x061A) or (cp >= 0x064B and cp <= 0x065F) or
        (cp >= 0x1161 and cp <= 0x11FF); // Hangul V/T jamo continue a syllable
}

test "ming_tts text: nfc composes French, Vietnamese stacking, kana and Hangul" {
    const a = std.testing.allocator;
    const cases = [_][2][]const u8{
        .{ "e\u{301}te\u{301} a\u{300} c\u{327}a o\u{302}", "été à ça ô" },
        .{ "e\u{323}\u{302}", "ệ" },
        .{ "か\u{3099}", "が" },
        .{ "\u{1112}\u{1161}\u{11AB}", "한" },
        .{ "déjà", "déjà" },
    };
    for (cases) |c| {
        const got = try nfc(a, c[0]);
        defer a.free(got);
        try std.testing.expectEqualStrings(c[1], got);
    }
}

pub fn buildInputIds(a: std.mem.Allocator, tok: anytype, prompt: []const u8, text_: []const u8, num_speakers: usize, instruction: ?[]const u8, prompt_text: ?[]const u8, prompt_latent_steps: usize) !InputIds {
    var ids: std.ArrayList(i32) = .empty;
    errdefer ids.deinit(a);
    const Enc = struct {
        fn add(al: std.mem.Allocator, t: anytype, list: *std.ArrayList(i32), s: []const u8) !void {
            // tokenizer.json declares an NFC normalizer that tokenizer.zig does
            // not apply; without it NFD input (e + U+0301, common from macOS
            // paste) tokenizes differently from the reference.
            const n = try nfc(al, s);
            defer al.free(n);
            const e = try t.encode(al, n);
            defer al.free(e);
            for (e) |x| try list.append(al, @intCast(x));
        }
    };
    try Enc.add(a, tok, &ids, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
    try Enc.add(a, tok, &ids, "<|im_start|>user\n");
    try Enc.add(a, tok, &ids, prompt);
    for (0..num_speakers) |i| {
        var b: [32]u8 = undefined;
        try Enc.add(a, tok, &ids, try std.fmt.bufPrint(&b, "  speaker_{d}:", .{i + 1}));
        try Enc.add(a, tok, &ids, "<|vision_start|>");
        try Enc.add(a, tok, &ids, "<|vision_pad|>");
        try Enc.add(a, tok, &ids, "<|vision_end|>\n");
    }
    const bgm = for ([_][]const u8{ "Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: " }) |mk| {
        if (std.mem.indexOf(u8, text_, mk) == null) break false;
    } else true;
    if (!bgm) try Enc.add(a, tok, &ids, " Text input:\n");
    const with_latent = prompt_latent_steps > 0 and prompt_text != null;
    if (with_latent) try Enc.add(a, tok, &ids, prompt_text.?);
    try Enc.add(a, tok, &ids, text_);
    try Enc.add(a, tok, &ids, "<|im_end|>\n");
    try Enc.add(a, tok, &ids, "<|im_start|>assistant\n");
    if (instruction) |ins| {
        try Enc.add(a, tok, &ids, ins);
        try Enc.add(a, tok, &ids, "<|im_end|>");
    }
    try Enc.add(a, tok, &ids, "<audio>");
    var pos: std.ArrayList(usize) = .empty;
    errdefer pos.deinit(a);
    for (ids.items, 0..) |x, i| if (x == TOK_SPEAKER_START) try pos.append(a, i + 1);
    if (pos.items.len != num_speakers) return error.SpeakerSlotMismatch;
    for (pos.items) |p| if (p >= ids.items.len or ids.items[p] != TOK_SPEAKER_PAD) return error.SpeakerSlotMismatch;
    var latent_pos: ?usize = null;
    if (with_latent) {
        latent_pos = ids.items.len;
        try ids.appendNTimes(a, @intCast(TOK_AUDIO_PATCH), prompt_latent_steps);
    }
    return .{ .ids = try ids.toOwnedSlice(a), .speaker_positions = try pos.toOwnedSlice(a), .latent_position = latent_pos };
}

test "ming_tts text: numberToWords matches inflect" {
    const a = std.testing.allocator;
    const cases = [_]struct { []const u8, bool, []const u8 }{
        .{ "1234", false, "one thousand, two hundred thirty-four" },
        .{ "2024", true, "two thousand and twenty-four" },
        .{ "1001000", true, "one million, one thousand" },
        .{ "123456789012", true, "one hundred and twenty-three billion, four hundred and fifty-six million, seven hundred and eighty-nine thousand and twelve" },
        .{ "0", true, "zero" },
    };
    for (cases) |c| {
        const w = try numberToWords(a, c[0], c[1]);
        defer a.free(w);
        try std.testing.expectEqualStrings(c[2], w);
    }
    const o = try ordinalWords(a, "twenty-one");
    defer a.free(o);
    try std.testing.expectEqualStrings("twenty-first", o);
}

// ── Text language guess (for routing French/German/Spanish/Japanese/Korean to MULTILINGUAL) ──

/// Best guess at the language of `text`: "ja", "ko", "fr", "de", "es", or
/// null for Chinese/English (what plain TTS handles) and anything unclear.
/// Script ranges decide ja/ko; Latin text is scored on stopwords and
/// language-specific letters. A heuristic: short or mixed text can be wrong,
/// so callers suggest rather than override when the user chose a task.
pub fn detectLanguage(text_utf8: []const u8) ?[]const u8 {
    var view = std.unicode.Utf8View.init(text_utf8) catch return null;
    var it = view.iterator();
    var kana: usize = 0;
    var hangul: usize = 0;
    var han: usize = 0;
    var score = [_]usize{ 0, 0, 0, 0 }; // en fr de es
    var word: [32]u8 = undefined;
    var wl: usize = 0;
    var latin_words: usize = 0;
    while (true) {
        const cp_opt = it.nextCodepoint();
        const cp: u21 = cp_opt orelse ' ';
        if ((cp >= 0x3040 and cp <= 0x30FF) or (cp >= 0x31F0 and cp <= 0x31FF)) kana += 1;
        if ((cp >= 0xAC00 and cp <= 0xD7A3) or (cp >= 0x1100 and cp <= 0x11FF) or (cp >= 0x3130 and cp <= 0x318F)) hangul += 1;
        if (cp >= 0x4E00 and cp <= 0x9FFF) han += 1;
        switch (cp) {
            'é', 'è', 'ê', 'à', 'ù', 'ç', 'œ', 'â', 'î', 'ô', 'û', 'ë', 'ï', 'É', 'È', 'À', 'Ç' => score[1] += 2,
            'ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü' => score[2] += 2,
            'ñ', 'á', 'í', 'ó', 'ú', '¿', '¡', 'Ñ' => score[3] += 2,
            0x0300...0x036F => {}, // NFD marks: the letters before them are scored as words below
            else => {},
        }
        const letter = (cp < 128 and std.ascii.isAlphabetic(@intCast(cp))) or (cp >= 0xC0 and cp <= 0x24F) or cp == '\'' or cp == 0x2019;
        if (letter) {
            if (cp < 128 and wl < word.len) {
                word[wl] = std.ascii.toLower(@intCast(cp));
                wl += 1;
            } else if (wl < word.len) {
                word[wl] = '*'; // non-ASCII letter inside the word
                wl += 1;
            }
        } else if (wl > 0) {
            latin_words += 1;
            const w = word[0..wl];
            for (stopwords, 0..) |list, li| for (list) |sw| if (std.mem.eql(u8, w, sw)) {
                score[li] += 1;
            };
            wl = 0;
        }
        if (cp_opt == null) break;
    }
    if (kana > 0) return "ja";
    if (hangul > 0) return "ko";
    if (han > 0 and han * 2 >= latin_words) return null;
    var best: usize = 0;
    for (score, 0..) |sc, i| if (sc > score[best]) {
        best = i;
    };
    if (best == 0 or score[best] < 3 or score[best] <= score[0]) return null;
    return ([_][]const u8{ "en", "fr", "de", "es" })[best];
}

const stopwords = [_][]const []const u8{
    &.{ "the", "and", "is", "are", "of", "to", "in", "it", "you", "that", "this", "with", "for", "was", "have", "not", "be", "on", "at", "we", "they", "what", "my", "your" },
    &.{ "le", "la", "les", "un", "une", "des", "et", "est", "je", "tu", "il", "elle", "nous", "vous", "ils", "pas", "ne", "que", "qui", "du", "de", "pour", "dans", "sur", "avec", "c'est", "mais", "ce", "cette", "au", "aux", "mon", "ma", "bonjour", "oui", "non", "j'ai", "l'", "d'", "qu'", "n'est" },
    &.{ "der", "die", "das", "und", "ist", "ich", "du", "er", "sie", "wir", "ihr", "nicht", "ein", "eine", "mit", "auf", "zu", "den", "dem", "von", "es", "guten", "heute", "auch", "sind", "wie" },
    &.{ "el", "los", "las", "y", "es", "yo", "tu", "que", "no", "una", "con", "por", "para", "del", "se", "lo", "como", "pero", "hola", "buenos", "hoy", "muy", "est*", "vamos", "hace" },
};

test "ming_tts text: detectLanguage" {
    const cases = [_]struct { []const u8, ?[]const u8 }{
        .{ "Bonjour à tous, voici un essai de voix en français.", "fr" },
        .{ "Il a soixante ans et il est arrivé hier.", "fr" },
        .{ "Guten Morgen, heute scheint die Sonne und wir gehen in den Park.", "de" },
        .{ "Buenos días, hoy hace buen tiempo y vamos a caminar por la playa.", "es" },
        .{ "おはようございます。今日はいい天気ですね。", "ja" },
        .{ "안녕하세요. 오늘은 날씨가 좋네요.", "ko" },
        .{ "The garden stays open until sunset, and visitors leave by the north gate.", null },
        .{ "今天天气很好，我们一起去公园散步吧。", null },
        .{ "Café latte.", null },
    };
    for (cases) |c| {
        const got = detectLanguage(c[0]);
        if (c[1]) |want| try std.testing.expectEqualStrings(want, got orelse "null") else try std.testing.expectEqual(@as(?[]const u8, null), got);
    }
}

test "ming_tts text: buildRequest routes detected fr/de/es/ja/ko text off the zh/en path" {
    const a = std.testing.allocator;
    const fr = "En 1895, 96 % des billets coûtaient 100 €. C'est l'été.";
    const Case = struct { in: TaskInput, lang: ?[]const u8, line: ?[]const u8 };
    const cases = [_]Case{
        // Captioned voices: the text only.
        .{ .in = .{ .task = "IP", .text = fr, .controls = .{ .ip = "UK_George" } }, .lang = "fr", .line = null },
        .{ .in = .{ .task = "FREEFORM", .text = fr, .instruction = "A calm voice." }, .lang = "fr", .line = null },
        .{ .in = .{ .task = "DIALECT", .text = fr, .controls = .{ .dialect = "cantonese" } }, .lang = "fr", .line = null },
        // Voice from a reference clip or none: the line too.
        .{ .in = .{ .task = "TTS", .text = fr }, .lang = "fr", .line = "French" },
        .{ .in = .{ .task = "EMOTION", .text = fr, .controls = .{ .emotion = "中性" } }, .lang = "fr", .line = "French" },
        .{ .in = .{ .task = "BASIC", .text = "Guten Morgen, heute scheint die Sonne und wir gehen in den Park.", .controls = .{ .speed = "中速" } }, .lang = "de", .line = "German" },
        // zh/en text and explicit languages are left alone.
        .{ .in = .{ .task = "TTS", .text = "The garden stays open until sunset, and visitors leave by the north gate." }, .lang = null, .line = null },
        .{ .in = .{ .task = "TTS", .text = "今天天气很好，我们一起去公园散步吧。" }, .lang = null, .line = null },
        .{ .in = .{ .task = "TTS", .text = fr, .language_id = "English" }, .lang = "English", .line = "English" },
    };
    for (cases) |c| {
        var d: Diag = .{};
        const req = try buildRequest(a, c.in, &d);
        defer req.deinit(a);
        try std.testing.expectEqualStrings(c.lang orelse "null", req.text_language orelse "null");
        const want = if (c.line) |l| try std.fmt.allocPrint(a, "{s}Target language: {s}.\n", .{ DEFAULT_TTS_PROMPT, l }) else try a.dupe(u8, DEFAULT_TTS_PROMPT);
        defer a.free(want);
        try std.testing.expectEqualStrings(want, req.prompt);
    }
    // The French path spells the numbers in French and keeps the accents.
    var d: Diag = .{};
    const req = try buildRequest(a, .{ .task = "IP", .text = fr, .controls = .{ .ip = "UK_George" } }, &d);
    defer req.deinit(a);
    var frags = try prepareText(a, req.text, req.text_language, 24);
    defer {
        for (frags.items) |f| a.free(f);
        frags.deinit(a);
    }
    const joined = try std.mem.join(a, " ", frags.items);
    defer a.free(joined);
    try std.testing.expectEqualStrings("En mille huit cent quatre-vingt-quinze, quatre-vingt-seize pour cent des billets coûtaient cent euros. C'est l'été.", joined);
}

test "ming_tts text: prepareLongText tags clause, sentence and paragraph boundaries" {
    const a = std.testing.allocator;
    const txt = "The garden will remain open until sunset, provided the weather stays clear, and visitors are asked to leave by the north gate. After that, the lights along the river path will switch on.\n\nChapter two\nIt was\nlate.";
    var frags = try prepareLongText(a, txt, null, 24);
    defer freeFragments(a, &frags);
    // One paragraph without line breaks: the same fragments as prepareText.
    var plain = try prepareText(a, txt[0..std.mem.indexOf(u8, txt, "\n").?], null, 24);
    defer {
        for (plain.items) |f| a.free(f);
        plain.deinit(a);
    }
    for (plain.items, 0..) |f, i| try std.testing.expectEqualStrings(f, frags.items[i].text);
    const n = plain.items.len;
    try std.testing.expectEqual(Boundary.sentence, frags.items[0].after);
    try std.testing.expectEqual(Boundary.paragraph, frags.items[n - 1].after);
    // "Chapter two" is its own paragraph; "It was\nlate." is a hard wrap.
    try std.testing.expectEqualStrings("Chapter two.", frags.items[n].text);
    try std.testing.expectEqual(Boundary.paragraph, frags.items[n].after);
    try std.testing.expectEqualStrings("It was late.", frags.items[n + 1].text);
    try std.testing.expectEqual(Boundary.none, frags.items[n + 1].after);
    try std.testing.expectEqual(n + 2, frags.items.len);
    // A long sentence cut inside: the first piece ends at a clause.
    var long = try prepareLongText(a, "雨后的山谷很安静，远处的云慢慢变成了金色，我们沿着小路走回了村子，路边的稻田在风里轻轻摇晃，狗在远处叫了几声。", null, 24);
    defer freeFragments(a, &long);
    try std.testing.expect(long.items.len > 1);
    try std.testing.expectEqual(Boundary.clause, long.items[0].after);
}
