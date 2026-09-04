//! T5 (SentencePiece **Unigram**) tokenizer for the FLUX.1 T5-XXL text encoder.
//!
//! This is NOT the merge-based `sentencepiece_bpe` path in `tokenizer.zig`
//! (that is Gemma's BPE). T5 uses a **Unigram** language model: the vocab is a
//! list of (piece, log-prob score) and encoding is the Viterbi segmentation
//! that maximizes the summed score. See docs/reference.md (FLUX.1 section).
//!
//! Pre-tokenization mirrors HF `tokenizer.json`: WhitespaceSplit + Metaspace
//! (`▁`, prepend_scheme "always"). The Precompiled (NFKC) normalizer is
//! approximated as identity — exact for ASCII/most prompts; a future refinement
//! can fold the charsmap. Post-processing appends a single `</s>` (eos), no bos,
//! then right-pads with `<pad>` to `max_len` and builds the attention mask.

const std = @import("std");

pub const PAD_ID: i32 = 0;
pub const EOS_ID: i32 = 1;
const SPACE = "\xe2\x96\x81"; // ▁ U+2581 (Metaspace replacement)

pub const Encoded = struct {
    ids: []i32, // [max_len], right-padded with PAD_ID
    mask: []i32, // [max_len], 1 for real tokens (incl. eos), 0 for pad
    real_len: usize, // count of non-pad tokens (incl. eos)
    allocator: std.mem.Allocator,
    pub fn deinit(self: *Encoded) void {
        self.allocator.free(self.ids);
        self.allocator.free(self.mask);
    }
};

pub const T5Tokenizer = struct {
    allocator: std.mem.Allocator,
    /// piece bytes → id. Keys borrow `blob`.
    piece_to_id: std.StringHashMapUnmanaged(u32),
    scores: []f32, // by id
    unk_id: u32,
    max_piece_len: usize, // longest piece in bytes (Viterbi window bound)
    blob: []u8, // backing storage for all piece strings

    pub fn deinit(self: *T5Tokenizer) void {
        self.piece_to_id.deinit(self.allocator);
        self.allocator.free(self.scores);
        self.allocator.free(self.blob);
    }

    /// Load from `<model_dir>/tokenizer_2/tokenizer.json` (model_dir absolute).
    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !T5Tokenizer {
        const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer_2/tokenizer.json", .{model_dir});
        defer allocator.free(path);
        return loadFile(io, allocator, path);
    }

    /// Load from an explicit absolute `tokenizer.json` path.
    pub fn loadFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !T5Tokenizer {
        const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer file.close(io);
        var read_buf: [4096]u8 = undefined;
        var reader_state = file.reader(io, &read_buf);
        const bytes = try reader_state.interface.allocRemaining(allocator, .limited(64 * 1024 * 1024));
        defer allocator.free(bytes);
        return parse(allocator, bytes);
    }

    /// Parse a tokenizer.json blob. Extracts `model.vocab` (list of [piece,
    /// score]) and `model.unk_id`.
    pub fn parse(allocator: std.mem.Allocator, json_bytes: []const u8) !T5Tokenizer {
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        var parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, json_bytes, .{});
        const model = (parsed.object.get("model") orelse return error.NoModel).object;
        const mtype = (model.get("type") orelse return error.NoModelType).string;
        if (!std.mem.eql(u8, mtype, "Unigram")) return error.NotUnigram;
        const vocab = (model.get("vocab") orelse return error.NoVocab).array;
        const unk_id: u32 = if (model.get("unk_id")) |u| @intCast(u.integer) else 2;

        const n = vocab.items.len;
        var scores = try allocator.alloc(f32, n);
        errdefer allocator.free(scores);

        // First pass: total byte length of all pieces → one blob.
        var total: usize = 0;
        for (vocab.items) |entry| {
            const piece = entry.array.items[0].string;
            total += piece.len;
        }
        var blob = try allocator.alloc(u8, total);
        errdefer allocator.free(blob);

        var map: std.StringHashMapUnmanaged(u32) = .empty;
        errdefer map.deinit(allocator);
        try map.ensureTotalCapacity(allocator, @intCast(n));

        var off: usize = 0;
        var max_piece_len: usize = 1;
        for (vocab.items, 0..) |entry, id| {
            const piece = entry.array.items[0].string;
            const score: f64 = switch (entry.array.items[1]) {
                .float => |f| f,
                .integer => |iv| @floatFromInt(iv),
                else => 0,
            };
            scores[id] = @floatCast(score);
            @memcpy(blob[off .. off + piece.len], piece);
            const key = blob[off .. off + piece.len];
            off += piece.len;
            // First occurrence wins (SentencePiece ids are unique per piece).
            if (!map.contains(key)) map.putAssumeCapacity(key, @intCast(id));
            if (piece.len > max_piece_len) max_piece_len = piece.len;
        }

        return .{
            .allocator = allocator,
            .piece_to_id = map,
            .scores = scores,
            .unk_id = unk_id,
            .max_piece_len = max_piece_len,
            .blob = blob,
        };
    }

    /// Normalize `text` into the Metaspace byte string: split on ASCII
    /// whitespace, drop empties, join each word with a leading `▁`. Caller owns.
    fn metaspace(self: *const T5Tokenizer, text: []const u8) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(self.allocator);
        var it = std.mem.tokenizeAny(u8, text, " \t\r\n\x0b\x0c");
        while (it.next()) |word| {
            try out.appendSlice(self.allocator, SPACE);
            try out.appendSlice(self.allocator, word);
        }
        // Empty input still yields a leading ▁ (SentencePiece prepends always).
        if (out.items.len == 0) try out.appendSlice(self.allocator, SPACE);
        return out.toOwnedSlice(self.allocator);
    }

    /// Viterbi best-score segmentation of the normalized byte string → piece
    /// ids (no eos/pad). Caller owns.
    fn viterbi(self: *const T5Tokenizer, norm: []const u8) ![]u32 {
        const n = norm.len;
        const a = self.allocator;
        // dp[i] = best total score for norm[0..i]; back[i] = start of the piece
        // ending at i; back_id[i] = its id. i ranges 0..n (byte positions).
        const neg_inf = -std.math.inf(f32);
        const dp = try a.alloc(f32, n + 1);
        defer a.free(dp);
        const back = try a.alloc(usize, n + 1);
        defer a.free(back);
        const back_id = try a.alloc(u32, n + 1);
        defer a.free(back_id);
        @memset(dp, neg_inf);
        dp[0] = 0;
        // Unknown-piece penalty: worse than any real score so it only fires
        // when no vocab piece matches (rare; non-ASCII outside the vocab).
        const unk_penalty: f32 = -100.0;

        var end: usize = 1;
        while (end <= n) : (end += 1) {
            // Only consider ends on a UTF-8 char boundary.
            if (end < n and isUtf8Cont(norm[end])) continue;
            var start = if (end >= self.max_piece_len) end - self.max_piece_len else 0;
            var matched = false;
            while (start < end) : (start += 1) {
                if (dp[start] == neg_inf) continue;
                if (isUtf8Cont(norm[start])) continue; // start on boundary too
                if (self.piece_to_id.get(norm[start..end])) |id| {
                    const cand = dp[start] + self.scores[id];
                    if (cand > dp[end]) {
                        dp[end] = cand;
                        back[end] = start;
                        back_id[end] = id;
                        matched = true;
                    }
                }
            }
            // No piece ends here: consume one UTF-8 char as <unk>.
            if (!matched and dp[end] == neg_inf) {
                const cs = charStart(norm, end);
                if (dp[cs] != neg_inf) {
                    dp[end] = dp[cs] + unk_penalty;
                    back[end] = cs;
                    back_id[end] = self.unk_id;
                }
            }
        }

        // Walk back.
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(a);
        var pos = n;
        while (pos > 0) {
            try ids.append(a, back_id[pos]);
            pos = back[pos];
        }
        const out = try ids.toOwnedSlice(a);
        std.mem.reverse(u32, out);
        return out;
    }

    /// Encode `text` → padded ids + attention mask of length `max_len`
    /// (eos appended; truncated to leave room for eos if needed).
    pub fn encode(self: *const T5Tokenizer, text: []const u8, max_len: usize) !Encoded {
        const a = self.allocator;
        const norm = try self.metaspace(text);
        defer a.free(norm);
        const pieces = try self.viterbi(norm);
        defer a.free(pieces);

        const ids = try a.alloc(i32, max_len);
        errdefer a.free(ids);
        const mask = try a.alloc(i32, max_len);
        errdefer a.free(mask);
        @memset(ids, PAD_ID);
        @memset(mask, 0);

        // Room for a trailing eos.
        const keep = @min(pieces.len, max_len - 1);
        var w: usize = 0;
        while (w < keep) : (w += 1) {
            ids[w] = @intCast(pieces[w]);
            mask[w] = 1;
        }
        ids[w] = EOS_ID;
        mask[w] = 1;
        const real_len = w + 1;
        return .{ .ids = ids, .mask = mask, .real_len = real_len, .allocator = a };
    }
};

inline fn isUtf8Cont(b: u8) bool {
    return (b & 0xC0) == 0x80;
}
/// Start byte index of the UTF-8 char that ends at `end` (exclusive).
fn charStart(s: []const u8, end: usize) usize {
    var i = end;
    while (i > 0) {
        i -= 1;
        if (!isUtf8Cont(s[i])) return i;
    }
    return 0;
}

// ── Tests ──
const testing = std.testing;

test "metaspace normalization joins words with ▁ and prepends" {
    // Build a throwaway tokenizer with an empty vocab just to reach metaspace.
    var tk: T5Tokenizer = .{
        .allocator = testing.allocator,
        .piece_to_id = .empty,
        .scores = &.{},
        .unk_id = 2,
        .max_piece_len = 1,
        .blob = &.{},
    };
    const norm = try tk.metaspace("a  cat");
    defer testing.allocator.free(norm);
    // "a" and "cat" each get a leading ▁; the double space collapses.
    try testing.expectEqualStrings("\xe2\x96\x81a\xe2\x96\x81cat", norm);
}

// Full parity test — env-gated on a real tokenizer.json + reference ids fixture
// produced by tests/dump_t5_fixtures.py (HF tokenizers). Skips without them.
//   T5_TOK_JSON  = absolute path to tokenizer_2/tokenizer.json
//   T5_TOK_CASES = path to a JSONL of {"text":..., "ids":[...]} (pre-eos ids)
test "t5 unigram encode matches HF reference ids" {
    const json_p = std.c.getenv("T5_TOK_JSON") orelse return error.SkipZigTest;
    const cases_p = std.c.getenv("T5_TOK_CASES") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tk = try T5Tokenizer.loadFile(io, a, std.mem.span(json_p));
    defer tk.deinit();

    const cases_file = try std.Io.Dir.openFileAbsolute(io, std.mem.span(cases_p), .{});
    defer cases_file.close(io);
    var cases_buf: [4096]u8 = undefined;
    var cases_reader = cases_file.reader(io, &cases_buf);
    const cases = try cases_reader.interface.allocRemaining(a, .limited(4 * 1024 * 1024));
    defer a.free(cases);
    var lines = std.mem.tokenizeScalar(u8, cases, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var parsed = try std.json.parseFromSlice(std.json.Value, a, line, .{});
        defer parsed.deinit();
        const text = parsed.value.object.get("text").?.string;
        const want = parsed.value.object.get("ids").?.array;
        var enc = try tk.encode(text, 512);
        defer enc.deinit();
        // Compare pre-eos ids against the reference (reference excludes eos/pad).
        try testing.expectEqual(want.items.len + 1, enc.real_len); // +1 for eos
        for (want.items, 0..) |wid, i| {
            try testing.expectEqual(@as(i32, @intCast(wid.integer)), enc.ids[i]);
        }
        try testing.expectEqual(EOS_ID, enc.ids[enc.real_len - 1]);
    }
}
