//! English text → IPA phonemes for `src/kokoro.zig`.
//!
//! Kokoro's input is PHONEMES, not words, so this is not optional garnish —
//! without it `/v1/audio/speech` cannot accept text at all.
//!
//! Lookup ladder, cheapest first:
//!   1. exact dictionary hit (misaki `us_gold` → `us_silver`, ~180k entries)
//!   2. case variants (`Researchers` → `researchers`)
//!   3. suffix stemming with the English morphophonemic rules for -s/-ed
//!      (`completed` → `complete` + /d/), which is where most real misses land
//!   4. acronym spelling (`GPU` → "gee pee you") for all-caps tokens
//!   5. letter-to-sound rules
//!
//! LICENCE TRAP: upstream Kokoro's default path falls back to espeak-ng, which
//! is GPLv3 and would contaminate a shipped closed app. The misaki dictionaries
//! carry no such term. Never add an espeak fallback here.
//!
//! HETERONYMS: ~790 entries are objects keyed by part of speech
//! (`{"DEFAULT": "ˈæbz", "NOUN": null}`). With no POS tagger we always take
//! DEFAULT, so "read" and "lead" take their most common reading. That is a
//! known, bounded quality gap, not a correctness bug — see `resolveEntry`.

const std = @import("std");
const log = @import("log.zig");

pub const Phonemizer = struct {
    allocator: std.mem.Allocator,
    /// word → IPA. Owns both key and value bytes.
    map: std.StringHashMapUnmanaged([]const u8) = .{},
    arena: std.heap.ArenaAllocator,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Phonemizer {
        var self = Phonemizer{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
        errdefer self.arena.deinit();

        // gold first, silver second: `putNoClobber` semantics mean the
        // higher-confidence table wins on overlap.
        for ([_][]const u8{ "us_gold.json", "us_silver.json" }) |name| {
            self.loadTable(io, model_dir, name) catch |e| {
                // silver is a nice-to-have; gold is not.
                if (std.mem.eql(u8, name, "us_gold.json")) return e;
                log.warn("[kokoro-g2p] optional table {s} unavailable: {s}\n", .{ name, @errorName(e) });
            };
        }
        log.info("[kokoro-g2p] {d} pronunciations\n", .{self.map.count()});
        return self;
    }

    fn loadTable(self: *Phonemizer, io: std.Io, model_dir: []const u8, name: []const u8) !void {
        const a = self.arena.allocator();
        const path = try std.fmt.allocPrint(self.allocator, "{s}/g2p/{s}", .{ model_dir, name });
        defer self.allocator.free(path);

        const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer f.close(io);
        var rb: [4096]u8 = undefined;
        var rs = f.reader(io, &rb);
        const text = try rs.interface.allocRemaining(self.allocator, .limited(64 * 1024 * 1024));
        defer self.allocator.free(text);

        var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, text, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadG2pTable;

        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            const ipa = resolveEntry(e.value_ptr.*) orelse continue;
            const key = try a.dupe(u8, e.key_ptr.*);
            const val = try a.dupe(u8, ipa);
            _ = try self.map.getOrPutValue(a, key, val);
        }
    }

    pub fn deinit(self: *Phonemizer) void {
        self.arena.deinit();
    }

    /// A dictionary value is either a plain IPA string or a POS-keyed object.
    /// With no tagger we take DEFAULT; a null DEFAULT means "no general
    /// pronunciation", so the word falls through to the next rung rather than
    /// being voiced wrongly.
    fn resolveEntry(v: std.json.Value) ?[]const u8 {
        return switch (v) {
            .string => |s| s,
            .object => |o| blk: {
                const d = o.get("DEFAULT") orelse break :blk null;
                break :blk if (d == .string) d.string else null;
            },
            else => null,
        };
    }

    pub fn lookup(self: *const Phonemizer, word: []const u8) ?[]const u8 {
        return self.map.get(word);
    }

    /// Try the word as written, then lowercased, then capitalised.
    fn lookupCased(self: *const Phonemizer, allocator: std.mem.Allocator, word: []const u8) !?[]const u8 {
        if (self.lookup(word)) |p| return p;
        const lower = try std.ascii.allocLowerString(allocator, word);
        defer allocator.free(lower);
        if (self.lookup(lower)) |p| return p;
        if (lower.len > 0) {
            const cap = try allocator.dupe(u8, lower);
            defer allocator.free(cap);
            cap[0] = std.ascii.toUpper(cap[0]);
            if (self.lookup(cap)) |p| return p;
        }
        return null;
    }

    /// Phonemize a whole utterance. Returns an owned IPA string.
    pub fn phonemize(self: *const Phonemizer, allocator: std.mem.Allocator, text: []const u8) ![]u8 {
        const normalized = try normalize(allocator, text);
        defer allocator.free(normalized);

        var out: std.ArrayListUnmanaged(u8) = .empty;
        errdefer out.deinit(allocator);

        var it = TokenIter{ .text = normalized };
        var need_space = false;
        while (it.next()) |tok| {
            switch (tok.kind) {
                .word => {
                    if (need_space) try out.append(allocator, ' ');
                    try self.appendWord(allocator, &out, tok.text);
                    need_space = true;
                },
                .punct => {
                    // Kokoro's vocab carries these directly and they drive
                    // prosody, so they are kept, not stripped.
                    try out.appendSlice(allocator, tok.text);
                    need_space = true;
                },
            }
        }
        return out.toOwnedSlice(allocator);
    }

    fn appendWord(self: *const Phonemizer, allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), word: []const u8) !void {
        if (try self.lookupCased(allocator, word)) |p| {
            try out.appendSlice(allocator, p);
            return;
        }
        if (try self.stemmed(allocator, word)) |p| {
            defer allocator.free(p);
            try out.appendSlice(allocator, p);
            return;
        }
        if (isAcronym(word)) {
            try self.appendSpelled(allocator, out, word);
            return;
        }
        try appendLetterToSound(allocator, out, word);
    }

    /// Strip a known suffix, phonemize the stem, and re-attach the suffix's
    /// phonemes using the English voicing rules. This is the rung that recovers
    /// most real misses (`completed`, `quantizes`, `activations`).
    fn stemmed(self: *const Phonemizer, allocator: std.mem.Allocator, word: []const u8) !?[]u8 {
        const lower = try std.ascii.allocLowerString(allocator, word);
        defer allocator.free(lower);

        const Rule = struct { suffix: []const u8, restore: []const u8 };
        // Ordered longest-first so "-ings" cannot be eaten by "-s".
        const rules = [_]Rule{
            .{ .suffix = "'s", .restore = "" },
            .{ .suffix = "ing", .restore = "" },
            .{ .suffix = "ies", .restore = "y" },
            .{ .suffix = "es", .restore = "" },
            .{ .suffix = "ed", .restore = "" },
            .{ .suffix = "ly", .restore = "" },
            .{ .suffix = "s", .restore = "" },
            .{ .suffix = "d", .restore = "" },
        };

        for (rules) |r| {
            if (lower.len <= r.suffix.len + 1) continue;
            if (!std.mem.endsWith(u8, lower, r.suffix)) continue;

            const base_len = lower.len - r.suffix.len;
            const stem = try std.fmt.allocPrint(allocator, "{s}{s}", .{ lower[0..base_len], r.restore });
            defer allocator.free(stem);

            var found = try self.lookupCased(allocator, stem);
            // "-ing"/"-ed" often drop a silent e ("completing" → "complete").
            if (found == null and r.restore.len == 0) {
                const with_e = try std.fmt.allocPrint(allocator, "{s}e", .{lower[0..base_len]});
                defer allocator.free(with_e);
                found = try self.lookupCased(allocator, with_e);
            }
            const base = found orelse continue;

            var buf: std.ArrayListUnmanaged(u8) = .empty;
            errdefer buf.deinit(allocator);
            try buf.appendSlice(allocator, base);
            try appendSuffixPhonemes(allocator, &buf, r.suffix, base);
            return try buf.toOwnedSlice(allocator);
        }
        return null;
    }

    fn appendSpelled(self: *const Phonemizer, allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), word: []const u8) !void {
        for (word, 0..) |ch, i| {
            if (!std.ascii.isAlphabetic(ch)) continue;
            if (i > 0) try out.append(allocator, ' ');
            const name = letterName(ch);
            if (try self.lookupCased(allocator, name)) |p| {
                try out.appendSlice(allocator, p);
            } else {
                try out.appendSlice(allocator, letterIpa(ch));
            }
        }
    }
};

// ════════════════════════════════════════════════════════════════════════
// Suffix phonology
// ════════════════════════════════════════════════════════════════════════

/// Last phoneme of an IPA string, used to pick the voiced/voiceless allomorph.
fn lastPhoneme(ipa: []const u8) []const u8 {
    if (ipa.len == 0) return ipa;
    var i = ipa.len;
    while (i > 0) {
        i -= 1;
        // step back to a UTF-8 lead byte
        if (ipa[i] & 0xC0 != 0x80) {
            const seq = std.unicode.utf8ByteSequenceLength(ipa[i]) catch 1;
            const end = @min(i + seq, ipa.len);
            const c = ipa[i..end];
            // stress/length marks are not phonemes
            if (std.mem.eql(u8, c, "ˈ") or std.mem.eql(u8, c, "ˌ") or std.mem.eql(u8, c, "ː")) continue;
            return c;
        }
    }
    return ipa[0..0];
}

fn isIn(needle: []const u8, set: []const []const u8) bool {
    for (set) |x| if (std.mem.eql(u8, needle, x)) return true;
    return false;
}

const VOICELESS = [_][]const u8{ "p", "t", "k", "f", "θ", "s", "ʃ", "ʧ", "h" };
const SIBILANT = [_][]const u8{ "s", "z", "ʃ", "ʒ", "ʧ", "ʤ" };

/// Attach the phonemes for a stripped suffix, honouring the standard English
/// allomorphy: -s is /ɪz/ after a sibilant, /s/ after a voiceless consonant,
/// /z/ otherwise; -ed is /ɪd/ after /t d/, /t/ after voiceless, /d/ otherwise.
/// Getting this wrong is instantly audible ("cats" as "catz").
fn appendSuffixPhonemes(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), suffix: []const u8, base: []const u8) !void {
    const last = lastPhoneme(base);

    if (std.mem.eql(u8, suffix, "ing")) {
        try buf.appendSlice(allocator, "ɪŋ");
    } else if (std.mem.eql(u8, suffix, "ly")) {
        try buf.appendSlice(allocator, "li");
    } else if (std.mem.eql(u8, suffix, "ed") or std.mem.eql(u8, suffix, "d")) {
        if (isIn(last, &[_][]const u8{ "t", "d" })) {
            try buf.appendSlice(allocator, "ɪd");
        } else if (isIn(last, &VOICELESS)) {
            try buf.appendSlice(allocator, "t");
        } else {
            try buf.appendSlice(allocator, "d");
        }
    } else {
        // -s / -es / -ies / -'s
        if (isIn(last, &SIBILANT)) {
            try buf.appendSlice(allocator, "ɪz");
        } else if (isIn(last, &VOICELESS)) {
            try buf.appendSlice(allocator, "s");
        } else {
            try buf.appendSlice(allocator, "z");
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Fallbacks
// ════════════════════════════════════════════════════════════════════════

/// All-caps runs of 2+ letters read as initialisms ("GPU", "API").
fn isAcronym(word: []const u8) bool {
    if (word.len < 2) return false;
    var letters: usize = 0;
    for (word) |c| {
        if (std.ascii.isLower(c)) return false;
        if (std.ascii.isUpper(c)) letters += 1;
    }
    return letters >= 2;
}

fn letterName(ch: u8) []const u8 {
    return switch (std.ascii.toLower(ch)) {
        'a' => "ay",   'b' => "bee",  'c' => "see",  'd' => "dee",
        'e' => "ee",   'f' => "ef",   'g' => "gee",  'h' => "aitch",
        'i' => "eye",  'j' => "jay",  'k' => "kay",  'l' => "el",
        'm' => "em",   'n' => "en",   'o' => "oh",   'p' => "pee",
        'q' => "cue",  'r' => "ar",   's' => "ess",  't' => "tee",
        'u' => "you",  'v' => "vee",  'w' => "double-u", 'x' => "ex",
        'y' => "why",  'z' => "zee",
        else => "",
    };
}

fn letterIpa(ch: u8) []const u8 {
    return switch (std.ascii.toLower(ch)) {
        'a' => "ˈeɪ",  'b' => "bˈi",   'c' => "sˈi",   'd' => "dˈi",
        'e' => "ˈi",   'f' => "ˈɛf",   'g' => "ʤˈi",   'h' => "ˈeɪʧ",
        'i' => "ˈaɪ",  'j' => "ʤˈeɪ",  'k' => "kˈeɪ",  'l' => "ˈɛl",
        'm' => "ˈɛm",  'n' => "ˈɛn",   'o' => "ˈoʊ",   'p' => "pˈi",
        'q' => "kjˈu", 'r' => "ˈɑɹ",   's' => "ˈɛs",   't' => "tˈi",
        'u' => "jˈu",  'v' => "vˈi",   'w' => "dˈʌbəljˈu", 'x' => "ˈɛks",
        'y' => "wˈaɪ", 'z' => "zˈi",
        else => "",
    };
}

/// Last-resort grapheme→phoneme rules. Deliberately crude: it exists so an
/// unknown proper noun is voiced approximately rather than dropped in silence.
/// Anything reaching here is a dictionary gap worth logging, not a design goal.
fn appendLetterToSound(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), word: []const u8) !void {
    var i: usize = 0;
    const w = word;
    while (i < w.len) {
        const rest = w[i..];
        // digraphs first
        const Pair = struct { g: []const u8, p: []const u8 };
        const digraphs = [_]Pair{
            .{ .g = "sch", .p = "sk" }, .{ .g = "tch", .p = "ʧ" },
            .{ .g = "ch", .p = "ʧ" },   .{ .g = "sh", .p = "ʃ" },
            .{ .g = "th", .p = "θ" },   .{ .g = "ph", .p = "f" },
            .{ .g = "wh", .p = "w" },   .{ .g = "ck", .p = "k" },
            .{ .g = "ng", .p = "ŋ" },   .{ .g = "qu", .p = "kw" },
            .{ .g = "oo", .p = "u" },   .{ .g = "ee", .p = "i" },
            .{ .g = "ea", .p = "i" },   .{ .g = "ou", .p = "aʊ" },
            .{ .g = "ow", .p = "aʊ" },  .{ .g = "ai", .p = "eɪ" },
            .{ .g = "ay", .p = "eɪ" },  .{ .g = "oa", .p = "oʊ" },
            .{ .g = "oi", .p = "ɔɪ" },  .{ .g = "oy", .p = "ɔɪ" },
        };
        var matched = false;
        for (digraphs) |d| {
            if (rest.len >= d.g.len and std.ascii.eqlIgnoreCase(rest[0..d.g.len], d.g)) {
                try out.appendSlice(allocator, d.p);
                i += d.g.len;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        const single: []const u8 = switch (std.ascii.toLower(rest[0])) {
            'a' => "æ", 'b' => "b", 'c' => "k", 'd' => "d", 'e' => "ɛ",
            'f' => "f", 'g' => "ɡ", 'h' => "h", 'i' => "ɪ", 'j' => "ʤ",
            'k' => "k", 'l' => "l", 'm' => "m", 'n' => "n", 'o' => "ɑ",
            'p' => "p", 'q' => "k", 'r' => "ɹ", 's' => "s", 't' => "t",
            'u' => "ʌ", 'v' => "v", 'w' => "w", 'x' => "ks", 'y' => "i",
            'z' => "z",
            else => "",
        };
        try out.appendSlice(allocator, single);
        i += 1;
    }
}

// ════════════════════════════════════════════════════════════════════════
// Tokenization + normalization
// ════════════════════════════════════════════════════════════════════════

const TokenKind = enum { word, punct };
const Token = struct { text: []const u8, kind: TokenKind };

const TokenIter = struct {
    text: []const u8,
    i: usize = 0,

    fn next(self: *TokenIter) ?Token {
        while (self.i < self.text.len and self.text[self.i] == ' ') self.i += 1;
        if (self.i >= self.text.len) return null;

        const start = self.i;
        const c = self.text[self.i];
        if (std.ascii.isAlphanumeric(c) or c == '\'') {
            while (self.i < self.text.len and
                (std.ascii.isAlphanumeric(self.text[self.i]) or self.text[self.i] == '\'' or self.text[self.i] == '-'))
            {
                self.i += 1;
            }
            return .{ .text = self.text[start..self.i], .kind = .word };
        }
        self.i += 1;
        return .{ .text = self.text[start..self.i], .kind = .punct };
    }
};

/// Expand what a lexicon cannot hold: digits, currency and a few symbols.
/// Kokoro's vocab has no digits at all, so an unexpanded "42" is silently
/// dropped by the encoder — this is what stops numbers vanishing from speech.
pub fn normalize(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    var i: usize = 0;
    while (i < text.len) {
        const c = text[i];

        // "$42" reads as "forty two dollars", not "dollars forty two" — the
        // symbol PRECEDES the amount in text but FOLLOWS it in speech.
        if (c == '$' and i + 1 < text.len and std.ascii.isDigit(text[i + 1])) {
            var j = i + 1;
            while (j < text.len and (std.ascii.isDigit(text[j]) or text[j] == ',')) j += 1;
            const n = parseGrouped(allocator, text[i + 1 .. j]) catch null;
            if (n) |v| {
                try spaceIfNeeded(allocator, &out);
                try appendNumberWords(allocator, &out, v);
                try out.appendSlice(allocator, " dollars");
                i = j;
                continue;
            }
        }

        if (std.ascii.isDigit(c)) {
            // A digit run glued to letters ("int8", "H2O") must not fuse into
            // one pseudo-word — "inteight" is not in any lexicon and falls all
            // the way through to letter-to-sound as gibberish.
            try spaceIfNeeded(allocator, &out);

            var j = i;
            while (j < text.len and (std.ascii.isDigit(text[j]) or text[j] == ',')) j += 1;
            const n = parseGrouped(allocator, text[i..j]) catch {
                i = j;
                continue;
            };
            try appendNumberWords(allocator, &out, n);
            i = j;
            // "3.5" → "three point five"
            if (i + 1 < text.len and text[i] == '.' and std.ascii.isDigit(text[i + 1])) {
                try out.appendSlice(allocator, " point");
                i += 1;
                while (i < text.len and std.ascii.isDigit(text[i])) {
                    try out.append(allocator, ' ');
                    try appendNumberWords(allocator, &out, text[i] - '0');
                    i += 1;
                }
            }
            // and separate from any letters that follow ("8bit").
            if (i < text.len and std.ascii.isAlphabetic(text[i])) try out.append(allocator, ' ');
            continue;
        }

        const sym: ?[]const u8 = switch (c) {
            '$' => " dollars ",
            '%' => " percent ",
            '&' => " and ",
            '+' => " plus ",
            '=' => " equals ",
            '@' => " at ",
            '#' => " number ",
            '\n', '\t', '\r' => " ",
            else => null,
        };
        if (sym) |sy| {
            try out.appendSlice(allocator, sy);
        } else {
            try out.append(allocator, c);
        }
        i += 1;
    }
    return out.toOwnedSlice(allocator);
}

/// Separate a number from a letter that immediately precedes it.
fn spaceIfNeeded(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8)) !void {
    if (out.items.len > 0 and std.ascii.isAlphabetic(out.items[out.items.len - 1])) {
        try out.append(allocator, ' ');
    }
}

fn parseGrouped(allocator: std.mem.Allocator, raw: []const u8) !u64 {
    var digits: std.ArrayListUnmanaged(u8) = .empty;
    defer digits.deinit(allocator);
    for (raw) |d| if (d != ',') try digits.append(allocator, d);
    return std.fmt.parseInt(u64, digits.items, 10);
}

const ONES = [_][]const u8{ "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };
const TENS = [_][]const u8{ "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };

pub fn appendNumberWords(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), n: u64) !void {
    if (n < 20) {
        try out.appendSlice(allocator, ONES[@intCast(n)]);
        return;
    }
    if (n < 100) {
        try out.appendSlice(allocator, TENS[@intCast(n / 10)]);
        if (n % 10 != 0) {
            try out.append(allocator, ' ');
            try out.appendSlice(allocator, ONES[@intCast(n % 10)]);
        }
        return;
    }
    const scales = [_]struct { v: u64, name: []const u8 }{
        .{ .v = 1_000_000_000_000, .name = "trillion" },
        .{ .v = 1_000_000_000, .name = "billion" },
        .{ .v = 1_000_000, .name = "million" },
        .{ .v = 1_000, .name = "thousand" },
        .{ .v = 100, .name = "hundred" },
    };
    for (scales) |s| {
        if (n >= s.v) {
            try appendNumberWords(allocator, out, n / s.v);
            try out.append(allocator, ' ');
            try out.appendSlice(allocator, s.name);
            if (n % s.v != 0) {
                try out.append(allocator, ' ');
                try appendNumberWords(allocator, out, n % s.v);
            }
            return;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "kokoro-g2p: numbers become words because the vocab has no digits" {
    const a = testing.allocator;
    const cases = [_]struct { in: []const u8, want: []const u8 }{
        .{ .in = "0", .want = "zero" },
        .{ .in = "7", .want = "seven" },
        .{ .in = "13", .want = "thirteen" },
        .{ .in = "42", .want = "forty two" },
        .{ .in = "100", .want = "one hundred" },
        .{ .in = "365", .want = "three hundred sixty five" },
        .{ .in = "1000", .want = "one thousand" },
        .{ .in = "1,234", .want = "one thousand two hundred thirty four" },
        .{ .in = "2000000", .want = "two million" },
    };
    for (cases) |c| {
        const got = try normalize(a, c.in);
        defer a.free(got);
        try testing.expectEqualStrings(c.want, got);
    }
}

test "kokoro-g2p: decimals, currency and symbols expand" {
    const a = testing.allocator;
    const got = try normalize(a, "3.5");
    defer a.free(got);
    try testing.expectEqualStrings("three point five", got);

    // Currency reads AFTER the amount in speech even though the symbol comes
    // before it in text.
    const money = try normalize(a, "$20");
    defer a.free(money);
    try testing.expectEqualStrings("twenty dollars", money);

    const pct = try normalize(a, "50%");
    defer a.free(pct);
    try testing.expectEqualStrings("fifty percent ", pct);
}

test "kokoro-g2p: digits glued to letters are split, not fused" {
    const a = testing.allocator;
    // "int8" fusing into "inteight" is in no lexicon and falls through to
    // letter-to-sound as gibberish (measured: ɪntɛɪɡht).
    const got = try normalize(a, "int8");
    defer a.free(got);
    try testing.expectEqualStrings("int eight", got);

    const trailing = try normalize(a, "8bit");
    defer a.free(trailing);
    try testing.expectEqualStrings("eight bit", trailing);

    const both = try normalize(a, "H2O");
    defer a.free(both);
    try testing.expectEqualStrings("H two O", both);
}

test "kokoro-g2p: -s allomorphy follows the final phoneme, not the spelling" {
    const a = testing.allocator;
    // /s/ after voiceless, /z/ after voiced, /ɪz/ after a sibilant. Rendering
    // "cats" as "catz" is instantly audible.
    const cases = [_]struct { base: []const u8, want: []const u8 }{
        .{ .base = "kˈæt", .want = "kˈæts" }, // voiceless /t/
        .{ .base = "dˈɔɡ", .want = "dˈɔɡz" }, // voiced /ɡ/
        .{ .base = "bˈʌs", .want = "bˈʌsɪz" }, // sibilant /s/
        .{ .base = "bˈʌʒ", .want = "bˈʌʒɪz" }, // sibilant /ʒ/
    };
    for (cases) |c| {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(a);
        try buf.appendSlice(a, c.base);
        try appendSuffixPhonemes(a, &buf, "s", c.base);
        try testing.expectEqualStrings(c.want, buf.items);
    }
}

test "kokoro-g2p: -ed allomorphy" {
    const a = testing.allocator;
    const cases = [_]struct { base: []const u8, want: []const u8 }{
        .{ .base = "wˈɑnt", .want = "wˈɑntɪd" }, // after /t/
        .{ .base = "nˈid", .want = "nˈidɪd" }, // after /d/
        .{ .base = "wˈɔk", .want = "wˈɔkt" }, // voiceless /k/
        .{ .base = "kˈɔl", .want = "kˈɔld" }, // voiced /l/
    };
    for (cases) |c| {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(a);
        try buf.appendSlice(a, c.base);
        try appendSuffixPhonemes(a, &buf, "ed", c.base);
        try testing.expectEqualStrings(c.want, buf.items);
    }
}

test "kokoro-g2p: lastPhoneme skips stress and length marks" {
    // A naive "last byte" reading lands mid-codepoint on multi-byte IPA and
    // would pick the stress mark instead of the consonant.
    try testing.expectEqualStrings("t", lastPhoneme("kˈæt"));
    try testing.expectEqualStrings("ʒ", lastPhoneme("bˈʌʒ"));
    try testing.expectEqualStrings("i", lastPhoneme("bˈiː"));
    try testing.expectEqualStrings("ɡ", lastPhoneme("dˈɔɡ"));
}

test "kokoro-g2p: acronym detection" {
    try testing.expect(isAcronym("GPU"));
    try testing.expect(isAcronym("API"));
    try testing.expect(!isAcronym("Gpu"));
    try testing.expect(!isAcronym("hello"));
    try testing.expect(!isAcronym("A"));
}

test "kokoro-g2p: tokenizer keeps punctuation, which drives prosody" {
    var it = TokenIter{ .text = "hello, world!" };
    const t1 = it.next().?;
    try testing.expectEqualStrings("hello", t1.text);
    try testing.expectEqual(TokenKind.word, t1.kind);
    const t2 = it.next().?;
    try testing.expectEqualStrings(",", t2.text);
    try testing.expectEqual(TokenKind.punct, t2.kind);
    const t3 = it.next().?;
    try testing.expectEqualStrings("world", t3.text);
    const t4 = it.next().?;
    try testing.expectEqualStrings("!", t4.text);
    try testing.expectEqual(@as(?Token, null), it.next());
}

test "kokoro-g2p: hyphenated and apostrophe words stay one token" {
    var it = TokenIter{ .text = "don't state-of-the-art" };
    try testing.expectEqualStrings("don't", it.next().?.text);
    try testing.expectEqualStrings("state-of-the-art", it.next().?.text);
}

test "kokoro-g2p: letter-to-sound covers every ASCII letter without panicking" {
    const a = testing.allocator;
    var out: std.ArrayListUnmanaged(u8) = .empty;
    defer out.deinit(a);
    try appendLetterToSound(a, &out, "abcdefghijklmnopqrstuvwxyz");
    try testing.expect(out.items.len > 0);
    // Digraphs win over singles.
    out.clearRetainingCapacity();
    try appendLetterToSound(a, &out, "ship");
    try testing.expectEqualStrings("ʃɪp", out.items);
}

// ── Live (env-gated) ────────────────────────────────────────────────────

fn liveModelDir() ?[]const u8 {
    const p = std.c.getenv("KOKORO_TEST_MODEL") orelse return null;
    return std.mem.span(p);
}

fn testIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

test "kokoro-g2p: live dictionary phonemizes ordinary English" {
    const dir = liveModelDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    var p = try Phonemizer.load(testIo(), a, dir);
    defer p.deinit();

    try testing.expect(p.map.count() > 80_000);

    const ipa = try p.phonemize(a, "Hello world.");
    defer a.free(ipa);
    std.debug.print("\n[g2p] \"Hello world.\" -> {s}\n", .{ipa});
    try testing.expect(ipa.len > 0);
    try expectAllInVocab(a, dir, ipa);
}

/// THE invariant that matters: every symbol we emit must exist in Kokoro's
/// vocab, because `Vocab.encode` silently DROPS anything it does not know — an
/// un-phonemized word does not error, it just vanishes from the speech.
///
/// Note this cannot be "contains no ASCII letters": Kokoro's vocab genuinely
/// includes uppercase ASCII as diphthong symbols (A, I, O, W, Y, Q, S, T), so
/// `həlˈO` for "hello" is correct output, not a phonemizer failure.
fn expectAllInVocab(a: std.mem.Allocator, model_dir: []const u8, ipa: []const u8) !void {
    const kokoro = @import("kokoro.zig");
    const path = try std.fmt.allocPrint(a, "{s}/config.json", .{model_dir});
    defer a.free(path);
    const f = try std.Io.Dir.openFileAbsolute(testIo(), path, .{});
    defer f.close(testIo());
    var rb: [4096]u8 = undefined;
    var rs = f.reader(testIo(), &rb);
    const text = try rs.interface.allocRemaining(a, .limited(16 * 1024 * 1024));
    defer a.free(text);

    var vocab = try kokoro.Vocab.parse(a, text);
    defer vocab.deinit();

    var i: usize = 0;
    while (i < ipa.len) {
        const len = std.unicode.utf8ByteSequenceLength(ipa[i]) catch 1;
        const end = @min(i + len, ipa.len);
        const sym = ipa[i..end];
        if (vocab.get(sym) == null) {
            std.debug.print("\n[g2p] symbol {s} (bytes {any}) is NOT in the vocab\n", .{ sym, sym });
            return error.SymbolNotInVocab;
        }
        i = end;
    }
}

test "kokoro-g2p: live coverage on representative text" {
    const dir = liveModelDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    var p = try Phonemizer.load(testIo(), a, dir);
    defer p.deinit();

    const samples = [_][]const u8{
        "I'll check the server logs and restart the deployment if the health probe is failing.",
        "The transformer quantizes activations to int8 before dispatching the kernel to the GPU.",
        "Researchers announced Wednesday that the spacecraft completed its orbital insertion.",
        "Meet me at 3.30, it costs $42 and 50% is refundable.",
    };
    for (samples) |text| {
        const ipa = try p.phonemize(a, text);
        defer a.free(ipa);
        std.debug.print("[g2p] {s}\n   -> {s}\n", .{ text, ipa });
        try testing.expect(ipa.len > 0);
        try expectAllInVocab(a, dir, ipa);
    }
}
