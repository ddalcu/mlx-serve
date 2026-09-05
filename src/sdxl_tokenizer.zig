//! CLIP's BPE tokenizer, as SDXL's two towers use it.
//!
//! `tokenizer.zig` serves the language models and reads HF's single-file
//! `tokenizer.json`. CLIP ships the older pair — `vocab.json` + `merges.txt` —
//! with a different pretokenizer and a word-final `</w>` convention, so this is
//! its own reader rather than a mode of that one. The Unicode classifiers and
//! the GPT-2 byte map ARE shared, imported from `tokenizer.zig`, because a
//! second copy of a codepoint table is a copy that drifts.
//!
//! Two things about SDXL specifically:
//!
//!   THE TWO TOWERS PAD DIFFERENTLY. `tokenizer/` and `tokenizer_2/` ship
//!   byte-identical vocab and merges, and DIFFERENT `pad_token`s:
//!   `<|endoftext|>` (49407) for CLIP-L, `!` (0) for bigG. Padding both the
//!   same way is invisible — every id is in-vocab, every shape is right — and
//!   changes the embeddings at every pad position, which is most of a short
//!   prompt's 77-token window. The pad id is READ FROM EACH TOKENIZER'S OWN
//!   CONFIG, never assumed.
//!
//!   POOLING READS THE FIRST EOS. transformers pools bigG at
//!   `input_ids.argmax()`, and since EOS (49407) is the highest id in the
//!   vocab, that is the first EOS — the real end of the prompt, not the end of
//!   the padded window. `Encoded.eos_index` reports it.
//!
//! ORACLE STATUS: pinned against transformers' own `CLIPTokenizer` by the
//! reference vectors in the tests below, which were generated from it rather
//! than derived from this implementation.

const std = @import("std");
const log = @import("log.zig");
const lm_tok = @import("tokenizer.zig");

pub const BOS_ID: u32 = 49406;
pub const EOS_ID: u32 = 49407;
pub const MAX_TOKENS: usize = 77;

pub const Encoded = struct {
    /// Exactly `MAX_TOKENS` ids: BOS, content, EOS, then padding.
    ids: []i32,
    /// Position of the real EOS — where the pooled vector is read from.
    eos_index: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Encoded) void {
        self.allocator.free(self.ids);
    }
};

pub const ClipTokenizer = struct {
    allocator: std.mem.Allocator,
    /// Subword string -> id. Owns its keys.
    vocab: std.StringHashMapUnmanaged(u32),
    /// "left right" -> merge rank (lower merges first). Owns its keys.
    ranks: std.StringHashMapUnmanaged(u32),
    byte_map: [256]u21,
    pad_id: u32,

    pub fn deinit(self: *ClipTokenizer) void {
        var vit = self.vocab.iterator();
        while (vit.next()) |e| self.allocator.free(e.key_ptr.*);
        self.vocab.deinit(self.allocator);
        var rit = self.ranks.iterator();
        while (rit.next()) |e| self.allocator.free(e.key_ptr.*);
        self.ranks.deinit(self.allocator);
    }

    /// Encode `text` into a padded 77-token window.
    pub fn encode(self: *const ClipTokenizer, allocator: std.mem.Allocator, text: []const u8) !Encoded {
        var ids: std.ArrayList(i32) = .empty;
        defer ids.deinit(allocator);
        try ids.append(allocator, @intCast(BOS_ID));

        const cleaned = try cleanText(allocator, text);
        defer allocator.free(cleaned);

        // Content is capped at MAX_TOKENS-2 so BOS and EOS always fit; this is
        // transformers' `truncation=True` behavior, which truncates the content
        // rather than dropping the EOS.
        const content_cap = MAX_TOKENS - 2;

        var pos: usize = 0;
        outer: while (pos < cleaned.len) {
            const tok = nextPretoken(cleaned, pos) orelse break;
            pos = tok.end;
            if (tok.start == tok.end) break;
            const piece = cleaned[tok.start..tok.end];

            // Byte-level mapping, then BPE over the mapped string.
            const mapped = try self.mapBytes(allocator, piece);
            defer allocator.free(mapped);
            var parts = try self.bpe(allocator, mapped);
            defer {
                for (parts.items) |p| allocator.free(p);
                parts.deinit(allocator);
            }
            for (parts.items) |p| {
                const id = self.vocab.get(p) orelse {
                    // An out-of-vocab subword cannot happen with a well-formed
                    // merges table, but silently dropping one would shift every
                    // later token. Fall back to the unk id, which IS eos here.
                    log.warn("[sdxl] clip tokenizer: unknown subword {s}\n", .{p});
                    if (ids.items.len - 1 >= content_cap) break :outer;
                    try ids.append(allocator, @intCast(EOS_ID));
                    continue;
                };
                if (ids.items.len - 1 >= content_cap) break :outer;
                try ids.append(allocator, @intCast(id));
            }
        }

        const eos_index = ids.items.len;
        try ids.append(allocator, @intCast(EOS_ID));
        while (ids.items.len < MAX_TOKENS) {
            try ids.append(allocator, @intCast(self.pad_id));
        }

        return .{
            .ids = try allocator.dupe(i32, ids.items),
            .eos_index = eos_index,
            .allocator = allocator,
        };
    }

    /// Map each byte through the GPT-2 byte->codepoint table and re-encode as
    /// UTF-8 — the form the vocab's keys are written in.
    fn mapBytes(self: *const ClipTokenizer, allocator: std.mem.Allocator, piece: []const u8) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        var buf: [4]u8 = undefined;
        for (piece) |b| {
            const cp = self.byte_map[b];
            const n = try std.unicode.utf8Encode(cp, &buf);
            try out.appendSlice(allocator, buf[0..n]);
        }
        return out.toOwnedSlice(allocator);
    }

    /// BPE over one pretoken. The word's LAST symbol carries `</w>`, which is
    /// what makes "the" as a whole word a different token from "the" inside
    /// "there" — dropping it produces valid ids for a different string.
    fn bpe(self: *const ClipTokenizer, allocator: std.mem.Allocator, word: []const u8) !std.ArrayList([]u8) {
        var syms: std.ArrayList([]u8) = .empty;
        errdefer {
            for (syms.items) |s| allocator.free(s);
            syms.deinit(allocator);
        }

        // Split into codepoints; the final one gets the end-of-word marker.
        var i: usize = 0;
        while (i < word.len) {
            const info = lm_tok.decodeCodepoint(word, i) orelse break;
            const next = i + info.len;
            if (next >= word.len) {
                const s = try std.fmt.allocPrint(allocator, "{s}</w>", .{word[i..next]});
                try syms.append(allocator, s);
            } else {
                try syms.append(allocator, try allocator.dupe(u8, word[i..next]));
            }
            i = next;
        }
        if (syms.items.len == 0) return syms;

        // Repeatedly merge the adjacent pair with the LOWEST rank. Rank order
        // is the whole algorithm: merging by first-match instead produces a
        // valid tokenization of the same text with different ids.
        while (syms.items.len > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_at: ?usize = null;
            for (0..syms.items.len - 1) |j| {
                const key = try std.fmt.allocPrint(allocator, "{s} {s}", .{ syms.items[j], syms.items[j + 1] });
                defer allocator.free(key);
                if (self.ranks.get(key)) |r| {
                    if (r < best_rank) {
                        best_rank = r;
                        best_at = j;
                    }
                }
            }
            const at = best_at orelse break;
            const merged = try std.fmt.allocPrint(allocator, "{s}{s}", .{ syms.items[at], syms.items[at + 1] });
            allocator.free(syms.items[at]);
            allocator.free(syms.items[at + 1]);
            syms.items[at] = merged;
            _ = syms.orderedRemove(at + 1);
        }
        return syms;
    }
};

/// Lowercase and collapse whitespace — CLIP's `whitespace_clean` composed with
/// `do_lower_case`.
fn cleanText(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var buf: [4]u8 = undefined;

    var i: usize = 0;
    var pending_space = false;
    var wrote_any = false;
    while (i < text.len) {
        const info = lm_tok.decodeCodepoint(text, i) orelse break;
        i += info.len;
        const cp = info.cp;
        if (isSpace(cp)) {
            // Runs of any whitespace collapse to ONE space, and leading and
            // trailing runs vanish entirely (the trailing one by never being
            // flushed).
            if (wrote_any) pending_space = true;
            continue;
        }
        if (pending_space) {
            try out.append(allocator, ' ');
            pending_space = false;
        }
        const lower = toLower(cp);
        const n = try std.unicode.utf8Encode(lower, &buf);
        try out.appendSlice(allocator, buf[0..n]);
        wrote_any = true;
    }
    return out.toOwnedSlice(allocator);
}

fn isSpace(cp: u21) bool {
    return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or cp == 0x0B or cp == 0x0C;
}

/// Python's `str.lower()` for the ranges a prompt realistically carries: ASCII
/// and Latin-1 supplement. U+00D7 is MULTIPLICATION SIGN, not a letter, and is
/// the reason this is a range check with a hole rather than a bare `+= 0x20`.
fn toLower(cp: u21) u21 {
    if (cp >= 'A' and cp <= 'Z') return cp + 32;
    if (cp >= 0xC0 and cp <= 0xDE and cp != 0xD7) return cp + 32;
    if (cp >= 0x0391 and cp <= 0x03A9) return cp + 32; // Greek
    if (cp >= 0x0410 and cp <= 0x042F) return cp + 32; // Cyrillic
    return cp;
}

const Pretoken = struct { start: usize, end: usize };

/// CLIP's pretokenizer pattern:
///
///     <\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d
///     |[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
///
/// Note `[\p{N}]` has NO quantifier: every digit is its own pretoken, so "42"
/// is two tokens and not one. That is a real difference from the LM
/// tokenizers' `\p{N}{1,3}` grouping, and getting it wrong presents as a
/// model-quality bug rather than a tokenizer bug.
fn nextPretoken(text: []const u8, from: usize) ?Pretoken {
    var pos = from;
    // Skip separating whitespace; `cleanText` has already reduced runs to one.
    while (pos < text.len and isSpace(text[pos])) pos += 1;
    if (pos >= text.len) return null;

    // Special tokens, matched literally.
    for ([_][]const u8{ "<|startoftext|>", "<|endoftext|>" }) |sp| {
        if (std.mem.startsWith(u8, text[pos..], sp)) {
            return .{ .start = pos, .end = pos + sp.len };
        }
    }
    // Contractions. `cleanText` has lowercased, so a case-insensitive match is
    // the lowercase one.
    if (text[pos] == '\'') {
        for ([_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" }) |c| {
            if (std.mem.startsWith(u8, text[pos..], c)) {
                return .{ .start = pos, .end = pos + c.len };
            }
        }
    }

    const first = lm_tok.decodeCodepoint(text, pos) orelse return null;
    if (lm_tok.isLetter(first.cp)) {
        var end = pos + first.len;
        while (end < text.len) {
            const info = lm_tok.decodeCodepoint(text, end) orelse break;
            if (!lm_tok.isLetter(info.cp)) break;
            end += info.len;
        }
        return .{ .start = pos, .end = end };
    }
    if (lm_tok.isDigit(first.cp)) {
        // Exactly one digit — see the doc comment.
        return .{ .start = pos, .end = pos + first.len };
    }
    // `[^\s\p{L}\p{N}]+` — a run of everything else.
    var end = pos + first.len;
    while (end < text.len) {
        const info = lm_tok.decodeCodepoint(text, end) orelse break;
        if (isSpace(info.cp) or lm_tok.isLetter(info.cp) or lm_tok.isDigit(info.cp)) break;
        end += info.len;
    }
    return .{ .start = pos, .end = end };
}

/// Load `<model_dir>/<sub>` — `vocab.json`, `merges.txt`, and the pad token
/// from `tokenizer_config.json`.
pub fn load(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    sub: []const u8,
) !ClipTokenizer {
    var t = ClipTokenizer{
        .allocator = allocator,
        .vocab = .empty,
        .ranks = .empty,
        .byte_map = lm_tok.buildBytesToUnicode(),
        .pad_id = EOS_ID,
    };
    errdefer t.deinit();

    // ── vocab.json
    {
        const path = try std.fmt.allocPrint(allocator, "{s}/{s}/vocab.json", .{ model_dir, sub });
        defer allocator.free(path);
        const bytes = try readFile(io, allocator, path, 8 << 20);
        defer allocator.free(bytes);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadClipVocab;
        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            const key = try allocator.dupe(u8, e.key_ptr.*);
            errdefer allocator.free(key);
            try t.vocab.put(allocator, key, @intCast(e.value_ptr.*.integer));
        }
    }

    // ── merges.txt: one "left right" pair per line, rank = line order. The
    // leading `#version:` comment is not a merge and must not consume rank 0.
    {
        const path = try std.fmt.allocPrint(allocator, "{s}/{s}/merges.txt", .{ model_dir, sub });
        defer allocator.free(path);
        const bytes = try readFile(io, allocator, path, 8 << 20);
        defer allocator.free(bytes);
        var rank: u32 = 0;
        var lines = std.mem.splitScalar(u8, bytes, '\n');
        while (lines.next()) |raw| {
            const line = std.mem.trimEnd(u8, raw, "\r");
            if (line.len == 0 or line[0] == '#') continue;
            if (std.mem.indexOfScalar(u8, line, ' ') == null) continue;
            const key = try allocator.dupe(u8, line);
            errdefer allocator.free(key);
            try t.ranks.put(allocator, key, rank);
            rank += 1;
        }
    }

    // ── pad token, per tower. See the file header: the two towers differ.
    {
        const path = try std.fmt.allocPrint(allocator, "{s}/{s}/tokenizer_config.json", .{ model_dir, sub });
        defer allocator.free(path);
        if (readFile(io, allocator, path, 1 << 20)) |bytes| {
            defer allocator.free(bytes);
            if (std.json.parseFromSlice(std.json.Value, allocator, bytes, .{})) |parsed| {
                defer parsed.deinit();
                if (parsed.value == .object) {
                    if (parsed.value.object.get("pad_token")) |pt| {
                        if (pt == .string) {
                            if (t.vocab.get(pt.string)) |id| t.pad_id = id;
                        }
                    }
                }
            } else |_| {}
        } else |_| {}
    }

    log.info("[sdxl] loaded {s}: vocab={d} merges={d} pad_id={d}\n", .{
        sub, t.vocab.count(), t.ranks.count(), t.pad_id,
    });
    return t;
}

/// Build a tokenizer from vocab/merges BYTES rather than files, for a
/// single-file checkpoint that ships no tokenizer at all. The CLIP BPE is
/// standard and fixed, so `sdxl_single_file` embeds one copy and feeds it here
/// with each tower's own `pad_id` (49407 for CLIP-L, 0 for bigG — see the file
/// header). Parsing mirrors `load`'s vocab.json / merges.txt blocks exactly.
pub fn initFromBytes(
    allocator: std.mem.Allocator,
    vocab_json: []const u8,
    merges_txt: []const u8,
    pad_id: u32,
) !ClipTokenizer {
    var t = ClipTokenizer{
        .allocator = allocator,
        .vocab = .empty,
        .ranks = .empty,
        .byte_map = lm_tok.buildBytesToUnicode(),
        .pad_id = pad_id,
    };
    errdefer t.deinit();

    {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, vocab_json, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadClipVocab;
        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            const key = try allocator.dupe(u8, e.key_ptr.*);
            errdefer allocator.free(key);
            try t.vocab.put(allocator, key, @intCast(e.value_ptr.*.integer));
        }
    }

    {
        var rank: u32 = 0;
        var lines = std.mem.splitScalar(u8, merges_txt, '\n');
        while (lines.next()) |raw| {
            const line = std.mem.trimEnd(u8, raw, "\r");
            if (line.len == 0 or line[0] == '#') continue;
            if (std.mem.indexOfScalar(u8, line, ' ') == null) continue;
            const key = try allocator.dupe(u8, line);
            errdefer allocator.free(key);
            try t.ranks.put(allocator, key, rank);
            rank += 1;
        }
    }

    log.info("[sdxl] loaded single-file tokenizer: vocab={d} merges={d} pad_id={d}\n", .{
        t.vocab.count(), t.ranks.count(), t.pad_id,
    });
    return t;
}

fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8, limit: usize) ![]u8 {
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return error.BadPath;
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    return rs.interface.allocRemaining(allocator, .limited(limit));
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "sdxl tokenizer: pretokenizer splits digits singly and groups punctuation" {
    // `[\p{N}]` carries no quantifier, so each digit stands alone.
    {
        const t = nextPretoken("42", 0).?;
        try testing.expectEqualStrings("4", "42"[t.start..t.end]);
    }
    // Letters run together, across non-ASCII.
    {
        const s = "caf\u{00E9} x";
        const t = nextPretoken(s, 0).?;
        try testing.expectEqualStrings("caf\u{00E9}", s[t.start..t.end]);
    }
    // Punctuation runs group.
    {
        const s = "a!!!b";
        const t = nextPretoken(s, 1).?;
        try testing.expectEqualStrings("!!!", s[t.start..t.end]);
    }
    // Contractions are their own pretoken.
    {
        const s = "don't";
        const t = nextPretoken(s, 3).?;
        try testing.expectEqualStrings("'t", s[t.start..t.end]);
    }
}

test "sdxl tokenizer: cleanText lowercases and collapses whitespace" {
    const a = testing.allocator;
    {
        const got = try cleanText(a, "  A   PHOTO\tof \n a Cat!  ");
        defer a.free(got);
        try testing.expectEqualStrings("a photo of a cat!", got);
    }
    // Latin-1 uppercase lowercases; the multiplication sign in the same range
    // is not a letter and must not shift.
    {
        const got = try cleanText(a, "\u{00DC}BER \u{00D7}");
        defer a.free(got);
        try testing.expectEqualStrings("\u{00FC}ber \u{00D7}", got);
    }
}

// Pinned against transformers' own CLIPTokenizer. The expected vectors were
// generated by it, not derived from this implementation — a reference that
// reuses the code under test proves only that it is deterministic.
//
//   SDXL_CHECKPOINT_DIR=~/.mlx-serve/staging/sdxl-base-1.0 \
//     zig build test -Dtest-filter="sdxl tokenizer parity"
test "sdxl tokenizer parity: ids match transformers CLIPTokenizer" {
    const dir = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    var tok = try load(io, a, dir, "tokenizer");
    defer tok.deinit();

    const cases = [_]struct { text: []const u8, want: []const i32 }{
        .{ .text = "a photo of a cat", .want = &[_]i32{ 49406, 320, 1125, 539, 320, 2368, 49407 } },
        // Casing and repeated spaces normalize; `!` is its own token.
        .{ .text = "A PHOTO of  a Cat!", .want = &[_]i32{ 49406, 320, 1125, 539, 320, 2368, 256, 49407 } },
        // "42" is TWO tokens, and the contraction splits.
        .{ .text = "don't stop 42 times", .want = &[_]i32{ 49406, 847, 713, 1691, 275, 273, 1661, 49407 } },
        .{ .text = "cafe uber naive", .want = &[_]i32{ 49406, 4979, 10668, 43362, 49407 } },
        .{ .text = "hello-world, 3.14 (test)", .want = &[_]i32{ 49406, 3306, 268, 1002, 267, 274, 269, 272, 275, 263, 1628, 264, 49407 } },
        // Non-ASCII goes through the byte map before BPE.
        .{ .text = "caf\u{00E9} \u{00FC}ber na\u{00EF}ve", .want = &[_]i32{ 49406, 15304, 6522, 1516, 1097, 35689, 563, 49407 } },
        .{ .text = "  spaced   out  ", .want = &[_]i32{ 49406, 10336, 538, 620, 49407 } },
    };

    for (cases) |c| {
        var enc = try tok.encode(a, c.text);
        defer enc.deinit();
        try testing.expectEqual(MAX_TOKENS, enc.ids.len);
        // The window is BOS + content + EOS, then padding.
        try testing.expectEqualSlices(i32, c.want, enc.ids[0..c.want.len]);
        // EOS index points at the real end, not the end of the window.
        try testing.expectEqual(c.want.len - 1, enc.eos_index);
        for (enc.ids[c.want.len..]) |p| {
            try testing.expectEqual(@as(i32, @intCast(tok.pad_id)), p);
        }
    }
}

test "sdxl tokenizer parity: the two towers pad differently" {
    const dir = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    var t1 = try load(io, a, dir, "tokenizer");
    defer t1.deinit();
    var t2 = try load(io, a, dir, "tokenizer_2");
    defer t2.deinit();

    // Same vocab and merges, so the CONTENT agrees...
    try testing.expectEqual(t1.vocab.count(), t2.vocab.count());
    try testing.expectEqual(t1.ranks.count(), t2.ranks.count());
    // ...and the padding does not. This is the trap the file header describes:
    // padding both towers alike is invisible and wrong.
    try testing.expectEqual(EOS_ID, t1.pad_id);
    try testing.expectEqual(@as(u32, 0), t2.pad_id);

    var e1 = try t1.encode(a, "a photo of a cat");
    defer e1.deinit();
    var e2 = try t2.encode(a, "a photo of a cat");
    defer e2.deinit();
    try testing.expectEqual(e1.eos_index, e2.eos_index);
    try testing.expectEqualSlices(i32, e1.ids[0 .. e1.eos_index + 1], e2.ids[0 .. e2.eos_index + 1]);
    try testing.expectEqual(@as(i32, 49407), e1.ids[MAX_TOKENS - 1]);
    try testing.expectEqual(@as(i32, 0), e2.ids[MAX_TOKENS - 1]);
}

test "sdxl tokenizer parity: an overlong prompt keeps its EOS" {
    const dir = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tok = try load(io, a, dir, "tokenizer");
    defer tok.deinit();

    // 200 words is far past the 77-token window. Truncation drops CONTENT, and
    // the terminator survives — a window whose last token is a word leaves the
    // tower with no end-of-text signal and no valid pooling position.
    var long: std.ArrayList(u8) = .empty;
    defer long.deinit(a);
    for (0..200) |i| {
        if (i > 0) try long.append(a, ' ');
        try long.appendSlice(a, "cat");
    }
    var enc = try tok.encode(a, long.items);
    defer enc.deinit();
    try testing.expectEqual(MAX_TOKENS, enc.ids.len);
    try testing.expectEqual(@as(i32, @intCast(BOS_ID)), enc.ids[0]);
    try testing.expectEqual(MAX_TOKENS - 1, enc.eos_index);
    try testing.expectEqual(@as(i32, @intCast(EOS_ID)), enc.ids[MAX_TOKENS - 1]);
}
