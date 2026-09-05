//! T5's SentencePiece tokenizer, as SD 3.5 uses it (`tokenizer_3/`).
//!
//! A UNIGRAM model, not BPE. `sdxl_tokenizer.zig` next door merges symbol pairs
//! by rank; this one runs VITERBI over a scored vocabulary — every vocab entry
//! carries a log-probability, and the tokenization is the segmentation whose
//! piece scores sum highest. The two algorithms share a file layout and nothing
//! else, which is why this is its own reader rather than a mode of that one.
//!
//! The vocab comes from `tokenizer_3/tokenizer.json` (HF's fast-tokenizer
//! format: `model.vocab` is a list of `[piece, log_prob]` pairs, so a piece's
//! ID IS ITS INDEX). The `spiece.model` sitting beside it is a protobuf and is
//! deliberately not parsed.
//!
//! The pipeline, transcribed from that file's own `normalizer` /
//! `pre_tokenizer` blocks rather than guessed at:
//!
//!   normalizer: Sequence[ Precompiled(charsmap), Strip(right), Replace(/ {2,}/ -> U+2581) ]
//!   pre_tokenizer: Metaspace{ replacement: U+2581, prepend_scheme: always, split: true }
//!   post_processor: TemplateProcessing -> `$A </s>`   (</s> = id 1)
//!   model: Unigram{ unk_id: 2, byte_fallback: false }, 32100 pieces
//!
//! Two details that are easy to lose:
//!
//!   THE MULTI-SPACE RULE FIRES BEFORE METASPACE. A run of two or more spaces
//!   becomes ONE U+2581 during normalization, and Metaspace then does not
//!   prepend a second one because the string already starts with the
//!   replacement. Collapsing runs to a single ASCII space instead gives a
//!   different, valid-looking segmentation.
//!
//!   SD 3 PADS TO A FIXED LENGTH AND PASSES NO MASK. `encodePadded` emits
//!   exactly `max_len` ids: content, `</s>`, then `<pad>` (0) to the end. The
//!   content is capped at `max_len - 1` so the EOS always survives truncation
//!   — transformers truncates the CONTENT rather than dropping the special.
//!
//! ORACLE STATUS
//!
//!   VERIFIED against the reference tokenizer. The vectors in the tests below
//!   were produced by HuggingFace `tokenizers`' own `Tokenizer.from_file` over
//!   SD 3.5's real `tokenizer_3/tokenizer.json`, not derived from this code.
//!
//!   NOT VERIFIED / NOT IMPLEMENTED. The `Precompiled` charsmap — SentencePiece's
//!   NMT-NFKC table, a compiled trie shipped as a base64 blob — is treated as
//!   IDENTITY here. It is identity on printable ASCII, which is what the
//!   verified vectors cover; a prompt carrying full-width punctuation, exotic
//!   Unicode spaces or combining marks will tokenize differently from the
//!   reference. Likewise added tokens (`<extra_id_N>`, `</s>`, `<pad>`) written
//!   LITERALLY in a prompt are segmented as ordinary text rather than matched
//!   whole. Both are documented gaps, not claims.

const std = @import("std");
const log = @import("log.zig");

/// `</s>` and `<pad>` — ids 0 and 1 of the T5 vocab, read off the real
/// `tokenizer.json` (`model.vocab[0] == "<pad>"`, `[1] == "</s>"`).
pub const EOS_ID: i32 = 1;
pub const PAD_ID: i32 = 0;

/// SD 3.5's `tokenizer_max_length` for T5. The pipeline's default; a caller
/// may pass another to `encodePadded`.
pub const MAX_TOKENS: usize = 256;

/// SentencePiece's metaspace marker, U+2581 LOWER ONE EIGHTH BLOCK.
const METASPACE = "\u{2581}";

/// `tokenizers`' `K_UNK_PENALTY`. The unknown-piece node scores
/// `min_score - 10`, which makes unk strictly worse than any real piece
/// without being -inf (a -inf would make an unavoidable unk poison the whole
/// path score and lose the tie-breaks around it).
const UNK_PENALTY: f64 = 10.0;

const Entry = struct { id: u32, score: f64 };

pub const T5Tokenizer = struct {
    allocator: std.mem.Allocator,
    /// Piece -> {id, log-probability}. Owns its keys.
    pieces: std.StringHashMapUnmanaged(Entry),
    /// Longest piece in BYTES. The Viterbi scan tries every length up to this
    /// at each position; without it the scan is quadratic in the input.
    max_piece_bytes: usize,
    /// The lowest log-probability in the vocab — the base of the unk score.
    min_score: f64,
    unk_id: u32,

    pub fn deinit(self: *T5Tokenizer) void {
        var it = self.pieces.iterator();
        while (it.next()) |e| self.allocator.free(e.key_ptr.*);
        self.pieces.deinit(self.allocator);
    }

    /// Unigram encode + `</s>`, padded with `<pad>` to exactly `max_len`.
    /// Truncates the CONTENT at `max_len - 1` so the EOS always survives.
    pub fn encodePadded(self: *T5Tokenizer, a: std.mem.Allocator, text: []const u8, max_len: usize) ![]i32 {
        if (max_len == 0) return error.BadMaxLength;
        const out = try a.alloc(i32, max_len);
        errdefer a.free(out);
        @memset(out, PAD_ID);

        var ids: std.ArrayList(i32) = .empty;
        defer ids.deinit(a);
        try self.encodeContent(a, text, &ids, max_len - 1);

        @memcpy(out[0..ids.items.len], ids.items);
        out[ids.items.len] = EOS_ID;
        return out;
    }

    /// The content ids alone — no EOS, no padding. Split out because the
    /// padded window and the reference vectors are two different questions.
    pub fn encodeContent(
        self: *T5Tokenizer,
        a: std.mem.Allocator,
        text: []const u8,
        out: *std.ArrayList(i32),
        cap: usize,
    ) !void {
        const normalized = try normalize(a, text);
        defer a.free(normalized);
        // An empty normalized string produces NO pre-token splits at all, so
        // Metaspace never gets to prepend its marker — `""` encodes to just the
        // EOS, not to a lone `▁`.
        if (normalized.len == 0) return;

        const spaced = try metaspace(a, normalized);
        defer a.free(spaced);

        var start: usize = 0;
        while (start < spaced.len) {
            // `split: true` cuts BEFORE each marker and merges it with what
            // follows, so every segment after the first begins with U+2581.
            var end = spaced.len;
            var probe = start + METASPACE.len;
            while (probe + METASPACE.len <= spaced.len) : (probe += 1) {
                if (std.mem.startsWith(u8, spaced[probe..], METASPACE)) {
                    end = probe;
                    break;
                }
            }
            try self.viterbi(a, spaced[start..end], out, cap);
            start = end;
        }
    }

    /// Unigram's best-path segmentation of ONE pre-token.
    ///
    /// `best[j]` is the highest total score of any segmentation of `[0, j)`.
    /// Ties keep the FIRST candidate found (shortest piece from the earliest
    /// start), mirroring `tokenizers`' strict `>` over its lattice; with real
    /// float log-probabilities exact ties are vanishingly rare either way.
    fn viterbi(
        self: *T5Tokenizer,
        a: std.mem.Allocator,
        seg: []const u8,
        out: *std.ArrayList(i32),
        cap: usize,
    ) !void {
        if (seg.len == 0) return;
        const n = seg.len;
        const best = try a.alloc(f64, n + 1);
        defer a.free(best);
        const prev = try a.alloc(usize, n + 1);
        defer a.free(prev);
        const tok = try a.alloc(u32, n + 1);
        defer a.free(tok);

        const neg_inf = -std.math.inf(f64);
        @memset(best, neg_inf);
        best[0] = 0;
        const unk_score = self.min_score - UNK_PENALTY;

        var i: usize = 0;
        while (i < n) : (i += charLen(seg, i)) {
            if (best[i] == neg_inf) continue;
            const mblen = charLen(seg, i);
            var has_single = false;
            var l: usize = 1;
            const limit = @min(self.max_piece_bytes, n - i);
            while (l <= limit) : (l += 1) {
                const e = self.pieces.get(seg[i .. i + l]) orelse continue;
                if (l == mblen) has_single = true;
                const cand = best[i] + e.score;
                if (cand > best[i + l]) {
                    best[i + l] = cand;
                    prev[i + l] = i;
                    tok[i + l] = e.id;
                }
            }
            // No piece covers exactly the single character at `i`, so the
            // lattice gets an unk node for it — that is what keeps every
            // position reachable and the path finite.
            if (!has_single) {
                const cand = best[i] + unk_score;
                if (cand > best[i + mblen]) {
                    best[i + mblen] = cand;
                    prev[i + mblen] = i;
                    tok[i + mblen] = self.unk_id;
                }
            }
        }

        if (best[n] == neg_inf) return error.T5UnreachableSegment;

        // Backtrack, then emit forwards.
        var rev: std.ArrayList(u32) = .empty;
        defer rev.deinit(a);
        var p = n;
        while (p > 0) {
            try rev.append(a, tok[p]);
            p = prev[p];
        }
        var k = rev.items.len;
        while (k > 0) {
            k -= 1;
            if (out.items.len >= cap) return;
            try out.append(a, @intCast(rev.items[k]));
        }
    }
};

/// Byte length of the UTF-8 character starting at `i`. A malformed lead byte
/// advances one byte rather than stalling — a tokenizer that loops forever on
/// bad input is worse than one that mis-tokenizes it.
fn charLen(s: []const u8, i: usize) usize {
    const b = s[i];
    const n: usize = if (b < 0x80) 1 else if (b >= 0xF0) 4 else if (b >= 0xE0) 3 else if (b >= 0xC0) 2 else 1;
    return @min(n, s.len - i);
}

/// The normalizer Sequence, minus the `Precompiled` charsmap (see the header's
/// ORACLE STATUS): strip trailing whitespace, then collapse runs of TWO OR
/// MORE spaces into a single U+2581.
fn normalize(a: std.mem.Allocator, text: []const u8) ![]u8 {
    const stripped = std.mem.trimEnd(u8, text, " \t\n\r\u{000B}\u{000C}");
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < stripped.len) {
        if (stripped[i] == ' ') {
            var j = i;
            while (j < stripped.len and stripped[j] == ' ') j += 1;
            if (j - i >= 2) {
                try out.appendSlice(a, METASPACE);
            } else {
                try out.append(a, ' ');
            }
            i = j;
            continue;
        }
        try out.append(a, stripped[i]);
        i += 1;
    }
    return out.toOwnedSlice(a);
}

/// Metaspace with `prepend_scheme: always`: every space becomes U+2581, and the
/// marker is prepended UNLESS the string already starts with one. That
/// condition is what stops a normalized multi-space prefix from acquiring a
/// second marker.
fn metaspace(a: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    if (!std.mem.startsWith(u8, text, METASPACE)) try out.appendSlice(a, METASPACE);
    for (text) |b| {
        if (b == ' ') {
            try out.appendSlice(a, METASPACE);
        } else {
            try out.append(a, b);
        }
    }
    return out.toOwnedSlice(a);
}

// ── Loading ─────────────────────────────────────────────────────────────

/// Load `<model_dir>/<sub>/tokenizer.json` (`sub` is `tokenizer_3`).
pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, sub: []const u8) !T5Tokenizer {
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}/tokenizer.json", .{ model_dir, sub });
    defer allocator.free(path);
    // The real file is 2.3 MB.
    const bytes = try readFile(io, allocator, path, 32 << 20);
    defer allocator.free(bytes);
    var t = try initFromJson(allocator, bytes);
    errdefer t.deinit();
    log.info("[sd3-t5] loaded {s}: pieces={d} unk_id={d} max_piece={d}B\n", .{
        sub, t.pieces.count(), t.unk_id, t.max_piece_bytes,
    });
    return t;
}

/// Build from `tokenizer.json` BYTES — the half a test drives with a literal.
pub fn initFromJson(allocator: std.mem.Allocator, json: []const u8) !T5Tokenizer {
    var t = T5Tokenizer{
        .allocator = allocator,
        .pieces = .empty,
        .max_piece_bytes = 0,
        .min_score = 0,
        .unk_id = 2,
    };
    errdefer t.deinit();

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadT5Tokenizer;
    const model = parsed.value.object.get("model") orelse return error.BadT5Tokenizer;
    if (model != .object) return error.BadT5Tokenizer;

    // A BPE `tokenizer.json` has the same shape at this level and a completely
    // different algorithm below it, so the model type is CHECKED rather than
    // assumed — the failure would otherwise be a silently empty vocab.
    const kind = model.object.get("type") orelse return error.BadT5Tokenizer;
    if (kind != .string or !std.mem.eql(u8, kind.string, "Unigram")) return error.NotAUnigramTokenizer;

    if (model.object.get("unk_id")) |u| {
        if (u == .integer and u.integer >= 0) t.unk_id = @intCast(u.integer);
    }

    const vocab = model.object.get("vocab") orelse return error.BadT5Tokenizer;
    if (vocab != .array) return error.BadT5Tokenizer;
    t.min_score = std.math.inf(f64);
    for (vocab.array.items, 0..) |item, id| {
        if (item != .array or item.array.items.len < 2) return error.BadT5Tokenizer;
        const piece = item.array.items[0];
        if (piece != .string) return error.BadT5Tokenizer;
        const score: f64 = switch (item.array.items[1]) {
            .float => |f| f,
            .integer => |n| @floatFromInt(n),
            .number_string => |str| std.fmt.parseFloat(f64, str) catch return error.BadT5Tokenizer,
            else => return error.BadT5Tokenizer,
        };
        const key = try allocator.dupe(u8, piece.string);
        errdefer allocator.free(key);
        // A piece's ID IS ITS INDEX in this list.
        try t.pieces.put(allocator, key, .{ .id = @intCast(id), .score = score });
        t.max_piece_bytes = @max(t.max_piece_bytes, piece.string.len);
        t.min_score = @min(t.min_score, score);
    }
    if (t.pieces.count() == 0) return error.BadT5Tokenizer;
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

test "t5 tokenizer: normalization collapses multi-space runs to ONE metaspace" {
    const a = testing.allocator;
    {
        const got = try normalize(a, "a photo of a cat");
        defer a.free(got);
        try testing.expectEqualStrings("a photo of a cat", got);
    }
    {
        // Two-or-more spaces become a single U+2581 BEFORE metaspace runs;
        // trailing whitespace is stripped.
        const got = try normalize(a, "  double  spaces  here ");
        defer a.free(got);
        try testing.expectEqualStrings("\u{2581}double\u{2581}spaces\u{2581}here", got);
    }
}

test "t5 tokenizer: metaspace prepends only when the string lacks the marker" {
    const a = testing.allocator;
    {
        const got = try metaspace(a, "hello world");
        defer a.free(got);
        try testing.expectEqualStrings("\u{2581}hello\u{2581}world", got);
    }
    {
        // Already normalized to start with a marker — a second one here would
        // be an extra `▁` token at the head of every multi-space prompt.
        const got = try metaspace(a, "\u{2581}double\u{2581}spaces");
        defer a.free(got);
        try testing.expectEqualStrings("\u{2581}double\u{2581}spaces", got);
    }
}

/// A hand-built Unigram vocabulary in `tokenizer.json` shape, so the Viterbi
/// path can be reasoned about exactly. `ab` scores worse than `a` + `b`, so a
/// greedy longest-match tokenizer takes `ab` and Viterbi does not — which is
/// the entire difference between the two algorithms.
const TOY_JSON =
    \\{"model": {"type": "Unigram", "unk_id": 2, "vocab": [
    \\  ["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0],
    \\  ["▁", -2.0], ["a", -1.0], ["b", -1.0], ["ab", -5.0],
    \\  ["▁a", -1.5], ["▁ab", -20.0]
    \\]}}
;

test "t5 tokenizer: viterbi takes the best-SCORING path, not the longest match" {
    const a = testing.allocator;
    var t = try initFromJson(a, TOY_JSON);
    defer t.deinit();
    try testing.expectEqual(@as(u32, 2), t.unk_id);

    var out: std.ArrayList(i32) = .empty;
    defer out.deinit(a);
    try t.encodeContent(a, "ab", &out, 64);
    // "▁ab" (-20) loses to "▁a"(-1.5) + "b"(-1), and "ab"(-5) loses to
    // "a"(-1) + "b"(-1). A longest-match tokenizer would emit id 8.
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 5 }, out.items);
}

test "t5 tokenizer: an out-of-vocab character becomes unk, not a dropped byte" {
    const a = testing.allocator;
    var t = try initFromJson(a, TOY_JSON);
    defer t.deinit();
    var out: std.ArrayList(i32) = .empty;
    defer out.deinit(a);
    // 'z' is in no piece; the lattice must stay reachable through it. Silently
    // dropping it would shift every later id.
    try t.encodeContent(a, "azb", &out, 64);
    try testing.expectEqual(@as(usize, 3), out.items.len);
    try testing.expectEqual(@as(i32, 2), out.items[1]);
}

test "t5 tokenizer: padding is exactly max_len with the EOS always present" {
    const a = testing.allocator;
    var t = try initFromJson(a, TOY_JSON);
    defer t.deinit();

    {
        const ids = try t.encodePadded(a, "ab", 8);
        defer a.free(ids);
        try testing.expectEqualSlices(i32, &[_]i32{ 7, 5, 1, 0, 0, 0, 0, 0 }, ids);
    }
    {
        // Truncation cuts the CONTENT, never the EOS: with max_len 3 the two
        // content ids fill positions 0..1 and the EOS lands at 2.
        const ids = try t.encodePadded(a, "abab", 3);
        defer a.free(ids);
        try testing.expectEqual(@as(usize, 3), ids.len);
        try testing.expectEqual(EOS_ID, ids[2]);
    }
    {
        // An empty prompt is just the EOS — Metaspace never prepends its marker
        // to a string with no pre-token splits at all.
        const ids = try t.encodePadded(a, "", 4);
        defer a.free(ids);
        try testing.expectEqualSlices(i32, &[_]i32{ 1, 0, 0, 0 }, ids);
    }
}

test "t5 tokenizer: a BPE tokenizer.json is refused rather than read empty" {
    const a = testing.allocator;
    const bpe =
        \\{"model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}}
    ;
    try testing.expectError(error.NotAUnigramTokenizer, initFromJson(a, bpe));
}

// Reference vectors from HuggingFace `tokenizers` itself, over SD 3.5's real
// `tokenizer_3/tokenizer.json`:
//
//   from tokenizers import Tokenizer
//   Tokenizer.from_file(".../tokenizer_3/tokenizer.json").encode(text).ids
//
// Env-gated on the real tokenizer file because the vocab is 2.3 MB and does
// not belong in the tree:
//
//   T5_TOKENIZER_JSON=~/.mlx-serve/staging/sd3.5-large/tokenizer_3/tokenizer.json \
//     zig build test -Doptimize=ReleaseFast -Dtest-filter="t5 tokenizer reference"
test "t5 tokenizer reference: matches HuggingFace tokenizers on real prompts" {
    const path = std.mem.span(std.c.getenv("T5_TOKENIZER_JSON") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = try readFile(io, a, path, 32 << 20);
    defer a.free(bytes);
    var t = try initFromJson(a, bytes);
    defer t.deinit();

    const cases = [_]struct { text: []const u8, want: []const i32 }{
        // The `▁`+`a` split rather than `▁a` — the Viterbi choice, twice.
        .{ .text = "a photo of a cat", .want = &[_]i32{ 3, 9, 1202, 13, 3, 9, 1712, 1 } },
        .{
            .text = "A serene mountain lake at sunset, 8k, highly detailed",
            .want = &[_]i32{ 71, 28362, 4180, 6957, 44, 13744, 6, 505, 157, 6, 1385, 3117, 1 },
        },
        .{ .text = "hello", .want = &[_]i32{ 21820, 1 } },
        // The multi-space rule and the right-strip, together.
        .{ .text = "  double  spaces  here ", .want = &[_]i32{ 1486, 4856, 270, 1 } },
        .{ .text = "", .want = &[_]i32{1} },
    };

    for (cases) |c| {
        const ids = try t.encodePadded(a, c.text, MAX_TOKENS);
        defer a.free(ids);
        try testing.expectEqual(MAX_TOKENS, ids.len);
        try testing.expectEqualSlices(i32, c.want, ids[0..c.want.len]);
        for (ids[c.want.len..]) |v| try testing.expectEqual(PAD_ID, v);
    }
}
