//! T5 SentencePiece **Unigram** tokenizer — a new algorithm, needed only for
//! Anima's LLMAdapter `t5xxl_ids` (the shared `tokenizer.zig` only implements
//! sentencepiece_bpe/byte_level_bpe/wordpiece). Reference: HF `tokenizers`
//! `tokenizer.json` format for `comfy/text_encoders/t5_tokenizer` (T5TokenizerFast,
//! vocab 32100, model.type "Unigram"), and `comfy/sd1_clip.py SDTokenizer`
//! for how ComfyUI drives it (per-word encode, EOS stripped per word then
//! re-appended once at the end).
//!
//! Pipeline (normalizer Sequence[Precompiled, Strip(right), Replace(" {2,}"->"▁")]
//! + Metaspace pre-tokenizer(▁, prepend first) + Unigram model):
//!  1. Split the prompt on ASCII whitespace runs (mirrors ComfyUI calling the
//!     underlying HF tokenizer once per whitespace-separated word — a single
//!     word never contains a literal space, so the "collapse 2+ spaces" and
//!     "strip trailing whitespace" normalizer steps are no-ops per word and
//!     are not implemented separately).
//!  2. Prepend "▁" (U+2581) to each word (Metaspace add_prefix_space).
//!  3. Viterbi-tokenize the chunk against the 32100-piece vocabulary (DP over
//!     max cumulative log-score; a codepoint with no covering piece falls
//!     back to `unk_id` at a fixed penalty below the vocabulary's own worst
//!     score, matching sentencepiece's unk handling since `byte_fallback` is
//!     false for this model).
//!  4. Concatenate ids across words, append `eos_id` (1, `</s>`) exactly once.
//!
//! KNOWN GAP: step 0 (the `Precompiled` normalizer) is sentencepiece's own
//! charsmap-based NFKC-ish normalization. This is NOT implemented — Unicode
//! decomposition tables are a large undertaking for a component that mostly
//! sees plain prompt text. Text in the ASCII range round-trips exactly
//! (verified byte-for-byte against `tokenizers.Tokenizer.encode` — see
//! `tests/dump_anima_t5_fixtures.py`); exotic composed Unicode (fullwidth
//! forms, some quote/dash variants) may tokenize slightly differently than
//! the reference. Prompt-weighting syntax (`(word:1.5)`) is also not parsed —
//! every token gets weight 1.0, matching the model's own default.

const std = @import("std");

pub const EOS_ID: u32 = 1;
pub const UNK_ID: u32 = 2;
const UNK_PENALTY: f32 = 10.0;
const METASPACE = "\xe2\x96\x81"; // "▁" U+2581, UTF-8

const PieceInfo = struct { id: u32, score: f32 };

pub const T5Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    pieces: std.StringHashMap(PieceInfo),
    max_piece_len: usize = 0,
    min_score: f32 = 0.0,

    /// Load from a `tokenizer.json` (HF `tokenizers` format) with a top-level
    /// `model.vocab` array of `[piece, score]` pairs (Unigram model).
    pub fn loadFromFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !T5Tokenizer {
        const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer file.close(io);
        var read_buf: [4096]u8 = undefined;
        var reader_state = file.reader(io, &read_buf);
        const content = try reader_state.interface.allocRemaining(allocator, .limited(64 * 1024 * 1024));
        defer allocator.free(content);
        return loadFromSlice(allocator, content);
    }

    pub fn loadFromSlice(allocator: std.mem.Allocator, content: []const u8) !T5Tokenizer {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const aa = arena.allocator();

        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, aa, content, .{});
        if (parsed != .object) return error.InvalidT5Tokenizer;
        const model = parsed.object.get("model") orelse return error.InvalidT5Tokenizer;
        if (model != .object) return error.InvalidT5Tokenizer;
        const vocab_v = model.object.get("vocab") orelse return error.InvalidT5Tokenizer;
        if (vocab_v != .array) return error.InvalidT5Tokenizer;
        const vocab = vocab_v.array;

        var pieces = std.StringHashMap(PieceInfo).init(allocator);
        errdefer pieces.deinit();
        try pieces.ensureTotalCapacity(@intCast(vocab.items.len));

        var max_len: usize = 0;
        var min_score: f32 = std.math.inf(f32);
        for (vocab.items, 0..) |entry, i| {
            if (entry != .array or entry.array.items.len < 2) continue;
            const piece_v = entry.array.items[0];
            if (piece_v != .string) continue;
            const piece = try aa.dupe(u8, piece_v.string);
            const score_v = entry.array.items[1];
            const score: f32 = switch (score_v) {
                .float => |f| @floatCast(f),
                .integer => |n| @floatFromInt(n),
                else => 0.0,
            };
            pieces.putAssumeCapacity(piece, .{ .id = @intCast(i), .score = score });
            if (piece.len > max_len) max_len = piece.len;
            if (score < min_score) min_score = score;
        }
        if (!std.math.isFinite(min_score)) min_score = 0.0;

        return .{ .arena = arena, .pieces = pieces, .max_piece_len = max_len, .min_score = min_score };
    }

    pub fn deinit(self: *T5Tokenizer) void {
        self.pieces.deinit();
        self.arena.deinit();
    }

    /// Viterbi-tokenize one metaspace-prefixed chunk (no internal ASCII
    /// whitespace) against the piece vocabulary; appends ids to `out`.
    fn encodeChunk(self: *const T5Tokenizer, allocator: std.mem.Allocator, chunk: []const u8, out: *std.ArrayList(u32)) !void {
        if (chunk.len == 0) return;
        const n = chunk.len;
        const neg_inf = -std.math.inf(f32);
        const best_score = try allocator.alloc(f32, n + 1);
        defer allocator.free(best_score);
        const back_pos = try allocator.alloc(usize, n + 1);
        defer allocator.free(back_pos);
        const back_id = try allocator.alloc(u32, n + 1);
        defer allocator.free(back_id);
        @memset(best_score, neg_inf);
        best_score[0] = 0.0;

        const unk_score = self.min_score - UNK_PENALTY;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            if (best_score[i] == neg_inf) continue;
            // Unicode-codepoint fallback: always makes progress by >= 1 byte.
            const cp_len = std.unicode.utf8ByteSequenceLength(chunk[i]) catch 1;
            const next = @min(i + @as(usize, cp_len), n);
            const unk_total = best_score[i] + unk_score;
            if (unk_total > best_score[next]) {
                best_score[next] = unk_total;
                back_pos[next] = i;
                back_id[next] = UNK_ID;
            }
            // Vocabulary pieces starting at i.
            const max_len = @min(self.max_piece_len, n - i);
            var len: usize = 1;
            while (len <= max_len) : (len += 1) {
                const info = self.pieces.get(chunk[i .. i + len]) orelse continue;
                const total = best_score[i] + info.score;
                const j = i + len;
                if (total > best_score[j]) {
                    best_score[j] = total;
                    back_pos[j] = i;
                    back_id[j] = info.id;
                }
            }
        }

        // Backtrack n -> 0, then reverse-append.
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(allocator);
        var pos: usize = n;
        while (pos > 0) {
            try ids.append(allocator, back_id[pos]);
            pos = back_pos[pos];
        }
        var k: usize = ids.items.len;
        while (k > 0) {
            k -= 1;
            try out.append(allocator, ids.items[k]);
        }
    }

    /// Full pipeline: whitespace-split words, per-word metaspace+Viterbi
    /// encode (mirrors ComfyUI's per-word tokenizer call with the per-word
    /// EOS stripped), concatenate, append EOS once. Caller owns the result.
    pub fn encode(self: *const T5Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);

        var chunk: std.ArrayList(u8) = .empty;
        defer chunk.deinit(allocator);

        var it = std.mem.tokenizeAny(u8, text, " \t\n\r");
        while (it.next()) |word| {
            chunk.clearRetainingCapacity();
            try chunk.appendSlice(allocator, METASPACE);
            try chunk.appendSlice(allocator, word);
            try self.encodeChunk(allocator, chunk.items, &out);
        }
        try out.append(allocator, EOS_ID);
        return out.toOwnedSlice(allocator);
    }
};

const testing = std.testing;

fn testVocab(allocator: std.mem.Allocator) !T5Tokenizer {
    // A tiny hand-built vocab covering the "a cat" / unk cases below.
    const json =
        \\{"model":{"vocab":[
        \\["<pad>",0.0],["</s>",0.0],["<unk>",0.0],
        \\["\u2581",-2.0],["a",-4.0],["\u2581cat",-9.0],["c",-5.0],["at",-3.0],
        \\["t",-4.5],["\u2581a",-100.0]
        \\]}}
    ;
    return T5Tokenizer.loadFromSlice(allocator, json);
}

test "t5 tokenizer: loads a vocab and finds pieces by exact match" {
    var tok = try testVocab(testing.allocator);
    defer tok.deinit();
    try testing.expectEqual(@as(u32, 3), tok.pieces.get(METASPACE).?.id);
    try testing.expectEqual(@as(u32, 4), tok.pieces.get("a").?.id);
    try testing.expectEqual(@as(u32, 5), tok.pieces.get(METASPACE ++ "cat").?.id);
}

test "t5 tokenizer: single known word picks the single best-scoring piece" {
    var tok = try testVocab(testing.allocator);
    defer tok.deinit();
    const ids = try tok.encode(testing.allocator, "cat");
    defer testing.allocator.free(ids);
    // "▁cat" (id 5, score -9) beats "▁"+"c"+"at" (score -2-5-3=-10) or
    // "▁"+"c"+"a"+"t" etc — Viterbi picks the higher (less negative) total.
    try testing.expectEqualSlices(u32, &.{ 5, EOS_ID }, ids);
}

test "t5 tokenizer: falls back to per-char pieces when no merged piece wins" {
    var tok = try testVocab(testing.allocator);
    defer tok.deinit();
    // "▁a" exists but scores far worse than "▁"+"a" separately (-100 vs -2-4=-6).
    const ids = try tok.encode(testing.allocator, "a");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &.{ 3, 4, EOS_ID }, ids);
}

test "t5 tokenizer: unknown codepoint falls back to UNK_ID, one per codepoint" {
    var tok = try testVocab(testing.allocator);
    defer tok.deinit();
    const ids = try tok.encode(testing.allocator, "xyz");
    defer testing.allocator.free(ids);
    // "▁" matches, then x/y/z are all unknown -> UNK per codepoint.
    try testing.expectEqualSlices(u32, &.{ 3, UNK_ID, UNK_ID, UNK_ID, EOS_ID }, ids);
}

test "t5 tokenizer: multiple words concatenate with a single trailing EOS" {
    var tok = try testVocab(testing.allocator);
    defer tok.deinit();
    const ids = try tok.encode(testing.allocator, "cat cat");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &.{ 5, 5, EOS_ID }, ids);
}

test "t5 tokenizer: empty prompt is just EOS" {
    var tok = try testVocab(testing.allocator);
    defer tok.deinit();
    const ids = try tok.encode(testing.allocator, "");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &.{EOS_ID}, ids);
}

// Env-gated parity test against the real T5 tokenizer.json + a fixture
// produced by tests/dump_anima_t5_fixtures.py (HF `tokenizers` library).
// ANIMA_T5_TOKENIZER=/path/to/tokenizer.json ANIMA_T5_FIXTURE=/path/to/anima_t5_fixture.json
test "t5 tokenizer: parity vs reference fixture (real T5 vocab)" {
    const tok_path = std.mem.span(std.c.getenv("ANIMA_T5_TOKENIZER") orelse return error.SkipZigTest);
    const fixture_path = std.mem.span(std.c.getenv("ANIMA_T5_FIXTURE") orelse return error.SkipZigTest);

    const io = std.Io.Threaded.global_single_threaded.io();
    var tok = try T5Tokenizer.loadFromFile(io, testing.allocator, tok_path);
    defer tok.deinit();

    const file = try std.Io.Dir.openFileAbsolute(io, fixture_path, .{});
    defer file.close(io);
    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const content = try reader_state.interface.allocRemaining(testing.allocator, .limited(16 * 1024 * 1024));
    defer testing.allocator.free(content);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), content, .{});
    const cases = parsed.array.items;
    for (cases) |case| {
        const prompt = case.object.get("prompt").?.string;
        const expected = case.object.get("ids").?.array.items;
        const ids = try tok.encode(testing.allocator, prompt);
        defer testing.allocator.free(ids);
        try testing.expectEqual(expected.len, ids.len);
        for (expected, ids) |e, got| {
            try testing.expectEqual(@as(u32, @intCast(e.integer)), got);
        }
    }
}
