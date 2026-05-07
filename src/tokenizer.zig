const std = @import("std");
const log = @import("log.zig");
const io_util = @import("io_util.zig");

pub const TokenizerType = enum { sentencepiece_bpe, byte_level_bpe, wordpiece };

/// BPE tokenizer supporting both SentencePiece (Gemma) and byte-level (GPT-2/Qwen3) modes.
pub const Tokenizer = struct {
    /// Token string -> id
    vocab: std.StringHashMap(u32),
    /// Id -> token string
    id_to_token: std.AutoHashMap(u32, []const u8),
    /// (left_str, right_str) -> merge rank (lower = higher priority)
    merge_ranks: std.HashMap(MergePair, u32, MergePairContext, std.hash_map.default_max_load_percentage),
    allocator: std.mem.Allocator,
    /// Special tokens (string -> id)
    special_tokens: std.StringHashMap(u32),
    /// Tokenizer type determines encode/decode behavior
    tok_type: TokenizerType,
    /// Byte-to-unicode mapping for byte-level BPE (256 entries, index = byte value)
    byte_to_unicode: [256]u21,
    /// Unicode-to-byte reverse mapping
    unicode_to_byte: std.AutoHashMap(u21, u8),

    // Dynamic token IDs (populated from tokenizer.json added_tokens)
    bos_id: ?u32,
    eos_id: ?u32,

    /// Parsed `tokenizer.json`. We keep it alive so the map keys/values
    /// (vocab strings, merge pair halves, special-token names) can be
    /// borrowed directly from its arena instead of duped per entry — a 30×
    /// speedup on Gemma-class tokenizers (262k vocab + 514k merges).
    parsed_json: ?std.json.Parsed(std.json.Value) = null,

    const MergePair = struct {
        left: []const u8,
        right: []const u8,
    };

    const MergePairContext = struct {
        pub fn hash(_: MergePairContext, key: MergePair) u64 {
            var h = std.hash.Wyhash.init(0);
            h.update(key.left);
            h.update("\x00");
            h.update(key.right);
            return h.final();
        }
        pub fn eql(_: MergePairContext, a: MergePair, b: MergePair) bool {
            return std.mem.eql(u8, a.left, b.left) and std.mem.eql(u8, a.right, b.right);
        }
    };

    pub fn deinit(self: *Tokenizer) void {
        // Map keys/values either point into `parsed_json`'s arena (no
        // per-entry free needed) or were duped explicitly when no parsed
        // JSON is held (e.g., the test-only constructors). Freeing the
        // parsed JSON deinits its arena in one shot.
        if (self.parsed_json) |*p| {
            self.vocab.deinit();
            self.id_to_token.deinit();
            self.merge_ranks.deinit();
            self.special_tokens.deinit();
            self.unicode_to_byte.deinit();
            p.deinit();
            return;
        }
        var vit = self.vocab.iterator();
        while (vit.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.vocab.deinit();
        self.id_to_token.deinit();

        var mit = self.merge_ranks.iterator();
        while (mit.next()) |entry| {
            self.allocator.free(entry.key_ptr.left);
            self.allocator.free(entry.key_ptr.right);
        }
        self.merge_ranks.deinit();

        var sit = self.special_tokens.iterator();
        while (sit.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.special_tokens.deinit();
        self.unicode_to_byte.deinit();
    }

    /// Encode text to token IDs (no BOS/EOS added, except WordPiece adds [CLS]/[SEP]).
    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        // Split text around special tokens, encode segments with BPE, insert special token IDs
        var result = std.ArrayList(u32).empty;
        errdefer result.deinit(allocator);

        // WordPiece (BERT): wrap with [CLS] ... [SEP]
        if (self.tok_type == .wordpiece) {
            if (self.bos_id) |cls_id| try result.append(allocator, cls_id);
            const ids = try self.encodeWordPiece(allocator, text);
            defer allocator.free(ids);
            try result.appendSlice(allocator, ids);
            if (self.eos_id) |sep_id| try result.append(allocator, sep_id);
            return result.toOwnedSlice(allocator);
        }

        var pos: usize = 0;
        while (pos < text.len) {
            // Find the earliest special token at or after pos
            var best_match_pos: usize = text.len;
            var best_match_len: usize = 0;
            var best_match_id: u32 = 0;

            var sit = self.special_tokens.iterator();
            while (sit.next()) |entry| {
                if (std.mem.indexOfPos(u8, text, pos, entry.key_ptr.*)) |found_pos| {
                    if (found_pos < best_match_pos or (found_pos == best_match_pos and entry.key_ptr.len > best_match_len)) {
                        best_match_pos = found_pos;
                        best_match_len = entry.key_ptr.len;
                        best_match_id = entry.value_ptr.*;
                    }
                }
            }

            if (best_match_len > 0) {
                // Encode text before the special token
                if (best_match_pos > pos) {
                    const segment = text[pos..best_match_pos];
                    const ids = try self.encodeSegment(allocator, segment);
                    defer allocator.free(ids);
                    try result.appendSlice(allocator, ids);
                }
                // Insert the special token ID
                try result.append(allocator, best_match_id);
                pos = best_match_pos + best_match_len;
            } else {
                // No more special tokens, encode the rest
                if (pos < text.len) {
                    const ids = try self.encodeSegment(allocator, text[pos..]);
                    defer allocator.free(ids);
                    try result.appendSlice(allocator, ids);
                }
                break;
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// Encode a text segment (no special tokens) using the appropriate method.
    fn encodeSegment(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        return switch (self.tok_type) {
            .sentencepiece_bpe => self.encodeSentencePiece(allocator, text),
            .byte_level_bpe => self.encodeByteLevel(allocator, text),
            .wordpiece => self.encodeWordPiece(allocator, text),
        };
    }

    /// Decode token IDs to text.
    pub fn decode(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32, strip_leading_space: bool) ![]u8 {
        return switch (self.tok_type) {
            .sentencepiece_bpe => self.decodeSentencePiece(allocator, ids, strip_leading_space),
            .byte_level_bpe => self.decodeByteLevel(allocator, ids),
            .wordpiece => self.decodeWordPiece(allocator, ids),
        };
    }

    /// Look up a special token ID by its string representation.
    pub fn specialTokenId(self: *const Tokenizer, name: []const u8) ?u32 {
        return self.special_tokens.get(name);
    }

    // ── SentencePiece BPE (Gemma-style) ──

    fn encodeSentencePiece(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        // Normalize: replace spaces with ▁ (U+2581)
        var normalized: std.ArrayList(u8) = .empty;
        defer normalized.deinit(allocator);

        for (text) |c| {
            if (c == ' ') {
                try normalized.appendSlice(allocator, "\xe2\x96\x81");
            } else {
                try normalized.append(allocator, c);
            }
        }

        return self.bpeMerge(allocator, normalized.items);
    }

    fn decodeSentencePiece(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32, strip_leading_space: bool) ![]u8 {
        var result: std.ArrayList(u8) = .empty;
        defer result.deinit(allocator);

        for (ids) |id| {
            if (self.id_to_token.get(id)) |token| {
                try result.appendSlice(allocator, token);
            }
        }

        // Replace ▁ (0xE2 0x96 0x81) with space
        var output = try allocator.alloc(u8, result.items.len);
        var out_len: usize = 0;
        var i: usize = 0;
        while (i < result.items.len) {
            if (i + 2 < result.items.len and
                result.items[i] == 0xE2 and
                result.items[i + 1] == 0x96 and
                result.items[i + 2] == 0x81)
            {
                output[out_len] = ' ';
                out_len += 1;
                i += 3;
            } else {
                output[out_len] = result.items[i];
                out_len += 1;
                i += 1;
            }
        }

        const start: usize = if (strip_leading_space and out_len > 0 and output[0] == ' ') 1 else 0;
        const final_out = try allocator.dupe(u8, output[start..out_len]);
        allocator.free(output);
        return final_out;
    }

    // ── Byte-level BPE (GPT-2/Qwen3-style) ──

    fn encodeByteLevel(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        // Pre-tokenize using GPT-2 regex pattern (hand-coded state machine)
        var words = std.ArrayList([]const u8).empty;
        defer {
            for (words.items) |w| allocator.free(w);
            words.deinit(allocator);
        }
        try gpt2PreTokenize(allocator, text, &words);

        // For each word: map bytes to unicode chars, then BPE merge, then look up vocab
        var all_ids = std.ArrayList(u32).empty;
        errdefer all_ids.deinit(allocator);

        for (words.items) |word| {
            // Map each byte to its unicode character
            var unicode_str = std.ArrayList(u8).empty;
            defer unicode_str.deinit(allocator);

            for (word) |byte| {
                const cp = self.byte_to_unicode[byte];
                var utf8_buf: [4]u8 = undefined;
                const len = std.unicode.utf8Encode(cp, &utf8_buf) catch 1;
                try unicode_str.appendSlice(allocator, utf8_buf[0..len]);
            }

            // BPE merge on the unicode string
            const ids = try self.bpeMerge(allocator, unicode_str.items);
            defer allocator.free(ids);
            try all_ids.appendSlice(allocator, ids);
        }

        return all_ids.toOwnedSlice(allocator);
    }

    fn decodeByteLevel(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        // Collect token strings
        var token_str = std.ArrayList(u8).empty;
        defer token_str.deinit(allocator);

        for (ids) |id| {
            if (self.id_to_token.get(id)) |token| {
                try token_str.appendSlice(allocator, token);
            }
        }

        // Reverse byte-to-unicode mapping: for each unicode char, map back to the original byte
        var output = std.ArrayList(u8).empty;
        defer output.deinit(allocator);

        var i: usize = 0;
        while (i < token_str.items.len) {
            const cp_len = std.unicode.utf8ByteSequenceLength(token_str.items[i]) catch 1;
            const end = @min(i + cp_len, token_str.items.len);
            const cp = std.unicode.utf8Decode(token_str.items[i..end]) catch {
                try output.append(allocator, token_str.items[i]);
                i += 1;
                continue;
            };
            if (self.unicode_to_byte.get(cp)) |byte| {
                try output.append(allocator, byte);
            } else {
                // Not in mapping — output the raw UTF-8 bytes
                try output.appendSlice(allocator, token_str.items[i..end]);
            }
            i = end;
        }

        return output.toOwnedSlice(allocator);
    }

    // ── WordPiece (BERT) ──

    fn encodeWordPiece(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var result = std.ArrayList(u32).empty;
        errdefer result.deinit(allocator);

        // Lowercase
        var lower = std.ArrayList(u8).empty;
        defer lower.deinit(allocator);
        for (text) |c| {
            try lower.append(allocator, if (c >= 'A' and c <= 'Z') c + 32 else c);
        }

        // Split on whitespace and punctuation
        var words = std.ArrayList([]const u8).empty;
        defer words.deinit(allocator);

        var start: usize = 0;
        var i: usize = 0;
        while (i < lower.items.len) {
            const c = lower.items[i];
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r') {
                if (i > start) try words.append(allocator, lower.items[start..i]);
                i += 1;
                start = i;
            } else if (isPunct(c)) {
                if (i > start) try words.append(allocator, lower.items[start..i]);
                try words.append(allocator, lower.items[i .. i + 1]);
                i += 1;
                start = i;
            } else {
                i += 1;
            }
        }
        if (start < lower.items.len) try words.append(allocator, lower.items[start..]);

        const unk_id = self.vocab.get("[UNK]") orelse 0;

        // WordPiece: greedy longest-match with ## prefix
        for (words.items) |word| {
            var pos: usize = 0;
            while (pos < word.len) {
                var end: usize = word.len;
                var found = false;
                while (end > pos) {
                    // Build candidate: "##substr" for continuations, "substr" for first piece
                    var candidate = std.ArrayList(u8).empty;
                    defer candidate.deinit(allocator);
                    if (pos > 0) try candidate.appendSlice(allocator, "##");
                    try candidate.appendSlice(allocator, word[pos..end]);

                    if (self.vocab.get(candidate.items)) |id| {
                        try result.append(allocator, id);
                        pos = end;
                        found = true;
                        break;
                    }
                    // Shrink by one UTF-8 character from the end
                    end -= 1;
                    while (end > pos and (word[end] & 0xC0) == 0x80) end -= 1;
                }
                if (!found) {
                    try result.append(allocator, unk_id);
                    break;
                }
            }
        }

        return result.toOwnedSlice(allocator);
    }

    fn decodeWordPiece(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var output = std.ArrayList(u8).empty;
        defer output.deinit(allocator);

        for (ids, 0..) |id, idx| {
            if (self.id_to_token.get(id)) |token| {
                // Skip [CLS], [SEP], [PAD], [UNK] etc.
                if (token.len > 0 and token[0] == '[') continue;
                if (std.mem.startsWith(u8, token, "##")) {
                    try output.appendSlice(allocator, token[2..]);
                } else {
                    if (idx > 0 and output.items.len > 0) try output.append(allocator, ' ');
                    try output.appendSlice(allocator, token);
                }
            }
        }
        return output.toOwnedSlice(allocator);
    }

    fn isPunct(c: u8) bool {
        return switch (c) {
            '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~' => true,
            else => false,
        };
    }

    // ── Shared BPE merge logic ──

    fn bpeMerge(self: *const Tokenizer, allocator: std.mem.Allocator, input: []const u8) ![]u32 {
        // Split into individual UTF-8 characters
        var symbols: std.ArrayList([]const u8) = .empty;
        defer {
            for (symbols.items) |s| allocator.free(s);
            symbols.deinit(allocator);
        }

        var idx: usize = 0;
        while (idx < input.len) {
            const char_len = std.unicode.utf8ByteSequenceLength(input[idx]) catch 1;
            const end = @min(idx + char_len, input.len);
            const char_slice = try allocator.dupe(u8, input[idx..end]);
            try symbols.append(allocator, char_slice);
            idx = end;
        }

        // Iteratively apply BPE merges (greedy: always pick lowest rank pair)
        while (symbols.items.len > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_idx: ?usize = null;

            for (0..symbols.items.len - 1) |j| {
                const pair = MergePair{ .left = symbols.items[j], .right = symbols.items[j + 1] };
                if (self.merge_ranks.get(pair)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = j;
                    }
                }
            }

            if (best_idx == null) break;

            const bi = best_idx.?;
            const merged = try std.mem.concat(allocator, u8, &.{ symbols.items[bi], symbols.items[bi + 1] });
            allocator.free(symbols.items[bi]);
            allocator.free(symbols.items[bi + 1]);
            symbols.items[bi] = merged;
            _ = symbols.orderedRemove(bi + 1);
        }

        // Map to vocab IDs
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(allocator);

        for (symbols.items) |sym| {
            if (self.vocab.get(sym)) |id| {
                try ids.append(allocator, id);
            } else {
                // Unknown token — try to encode individual bytes
                for (sym) |byte| {
                    var utf8_buf: [4]u8 = undefined;
                    const cp = self.byte_to_unicode[byte];
                    const len = std.unicode.utf8Encode(cp, &utf8_buf) catch 1;
                    if (self.vocab.get(utf8_buf[0..len])) |id| {
                        try ids.append(allocator, id);
                    }
                }
            }
        }

        return ids.toOwnedSlice(allocator);
    }
};

/// GPT-2 pre-tokenization: splits text following the Qwen / Llama-3 / GPT-2
/// pre-tokenizer regex as a hand-rolled state machine. Each iteration picks
/// the FIRST matching pattern from the alternation, in declared order.
///
/// Reference (from `tokenizer.json` `pre_tokenizer.pretokenizers[0].pattern`):
///
///     (?i:'s|'t|'re|'ve|'m|'ll|'d)
///   | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
///   | \p{N}
///   |  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
///   | \s*[\r\n]+
///   | \s+(?!\S)
///   | \s+
///
/// Critical priority rules vs. naïve "consume whitespace, then letters":
///   1. ` letter+` is one pre-token (pattern 2 with optional leading non-LN char).
///   2. ` punct+` is one pre-token (pattern 4 with optional leading space).
///   3. Multi-space `    word`: pattern 6 `\s+(?!\S)` matches all-but-last
///      whitespace (`   `), then pattern 2 picks up the last space + letters
///      as one combined ` word` pre-token. Getting this wrong adds extra
///      whitespace pre-tokens that the BPE stage cannot merge across, and
///      causes the model to see a perturbed prior on every subsequent word.
///   4. Digits are SINGLE-codepoint pre-tokens (pattern 3 = `\p{N}`, not
///      `\p{N}+`). `100` → three separate `1`, `0`, `0` pre-tokens.
fn gpt2PreTokenize(allocator: std.mem.Allocator, text: []const u8, words: *std.ArrayList([]const u8)) !void {
    var i: usize = 0;
    while (i < text.len) {
        const start = i;

        // ── Pattern 1: contraction `(?i:'s|'t|'re|'ve|'m|'ll|'d)` ──
        if (text[i] == '\'' and i + 1 < text.len) {
            const next = std.ascii.toLower(text[i + 1]);
            if (next == 's' or next == 't' or next == 'm' or next == 'd') {
                i += 2;
                try words.append(allocator, try allocator.dupe(u8, text[start..i]));
                continue;
            }
            if (i + 2 < text.len) {
                const next2 = std.ascii.toLower(text[i + 2]);
                if ((next == 'r' and next2 == 'e') or
                    (next == 'v' and next2 == 'e') or
                    (next == 'l' and next2 == 'l'))
                {
                    i += 3;
                    try words.append(allocator, try allocator.dupe(u8, text[start..i]));
                    continue;
                }
            }
        }

        // ── Pattern 2: `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` ──
        // Optional 1 char that's NOT \r, NOT \n, NOT letter, NOT digit (so it
        // CAN be whitespace or punct), followed by 1+ letters/marks.
        if (matchOptionalNonLnnAndLetters(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // ── Pattern 3: `\p{N}` — exactly ONE digit codepoint ──
        if (decodeCodepoint(text, i)) |cp_info| {
            if (isDigit(cp_info.cp)) {
                i += cp_info.len;
                try words.append(allocator, try allocator.dupe(u8, text[start..i]));
                continue;
            }
        }

        // ── Pattern 4: ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*` ──
        // Optional 1 space + 1+ chars that are NOT whitespace, NOT letter,
        // NOT mark, NOT digit (i.e. punctuation/symbols), then optional \r\n.
        if (matchOptionalSpaceAndPunct(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // ── Pattern 5: `\s*[\r\n]+` — whitespace ending in newline run ──
        if (matchWhitespaceWithNewline(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // ── Pattern 6: `\s+(?!\S)` — whitespace not followed by non-ws ──
        // Greedy match with backtrack: shortens by 1 if the next char is \S
        // so the trailing space gets handed to pattern 2/4 on the next pass.
        if (matchTrailingWhitespace(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // ── Pattern 7: `\s+` — fallback whitespace ──
        if (i < text.len and isWhitespace(text[i])) {
            while (i < text.len and isWhitespace(text[i])) i += 1;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // Fallback: single byte (unreachable in well-formed UTF-8 input).
        i += 1;
        try words.append(allocator, try allocator.dupe(u8, text[start..i]));
    }
}

/// Pattern 2: `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+`. Returns end position of
/// match, or null if no letters at the right place.
fn matchOptionalNonLnnAndLetters(text: []const u8, start: usize) ?usize {
    if (start >= text.len) return null;
    const cp_start = decodeCodepoint(text, start) orelse return null;

    // Try with the optional non-LNN char consumed.
    if (!isLetter(cp_start.cp) and !isDigit(cp_start.cp) and
        cp_start.cp != '\r' and cp_start.cp != '\n')
    {
        const after_opt = start + cp_start.len;
        if (after_opt < text.len) {
            const next_cp = decodeCodepoint(text, after_opt);
            if (next_cp != null and isLetterOrMark(next_cp.?.cp)) {
                var i: usize = after_opt + next_cp.?.len;
                while (i < text.len) {
                    const c = decodeCodepoint(text, i) orelse break;
                    if (!isLetterOrMark(c.cp)) break;
                    i += c.len;
                }
                return i;
            }
        }
    }

    // Try with 0-length optional: text[start] must itself be a letter/mark.
    if (isLetterOrMark(cp_start.cp)) {
        var i: usize = start + cp_start.len;
        while (i < text.len) {
            const c = decodeCodepoint(text, i) orelse break;
            if (!isLetterOrMark(c.cp)) break;
            i += c.len;
        }
        return i;
    }

    return null;
}

/// Pattern 4: ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`. Optional ASCII space then
/// 1+ punct/symbol codepoints, then optional \r\n run. Returns end position
/// or null. The optional space MUST be exactly the byte ' ' (0x20), not
/// any other whitespace — matches Qwen's tokenizer.json regex literal.
fn matchOptionalSpaceAndPunct(text: []const u8, start: usize) ?usize {
    if (start >= text.len) return null;
    var p_start: usize = start;
    if (text[start] == ' ') p_start = start + 1;

    if (p_start >= text.len) return null;
    const first_cp = decodeCodepoint(text, p_start) orelse return null;
    // Must be NOT whitespace, NOT letter, NOT mark, NOT digit.
    if (isWhitespaceCp(first_cp.cp) or isLetter(first_cp.cp) or
        isMark(first_cp.cp) or isDigit(first_cp.cp)) return null;

    var i: usize = p_start + first_cp.len;
    while (i < text.len) {
        const c = decodeCodepoint(text, i) orelse break;
        if (isWhitespaceCp(c.cp) or isLetter(c.cp) or isMark(c.cp) or isDigit(c.cp)) break;
        i += c.len;
    }
    // Optional trailing \r\n.
    while (i < text.len and (text[i] == '\r' or text[i] == '\n')) i += 1;
    return i;
}

/// Pattern 5: `\s*[\r\n]+`. Returns end position, or null if no \r\n found
/// after consuming \s*.
fn matchWhitespaceWithNewline(text: []const u8, start: usize) ?usize {
    var i: usize = start;
    while (i < text.len and isWhitespace(text[i]) and text[i] != '\r' and text[i] != '\n') i += 1;
    if (i >= text.len or (text[i] != '\r' and text[i] != '\n')) return null;
    while (i < text.len and (text[i] == '\r' or text[i] == '\n')) i += 1;
    return i;
}

/// Pattern 6: `\s+(?!\S)`. Greedy match of all whitespace bytes, then
/// shortens by 1 if the next char is \S so the trailing space can be picked
/// up by pattern 2/4 on the next iteration. Returns null if there's only
/// one whitespace char and the next is \S (lookahead can't be satisfied).
fn matchTrailingWhitespace(text: []const u8, start: usize) ?usize {
    if (start >= text.len) return null;
    if (!isWhitespace(text[start])) return null;
    var end: usize = start;
    while (end < text.len and isWhitespace(text[end])) end += 1;
    // text[start..end] is the maximal whitespace run starting at start.
    if (end == text.len) return end; // end of input — lookahead trivially OK
    // text[end] is non-whitespace (\S). Backtrack one whitespace char so
    // the position-after-match lands on a whitespace char (lookahead OK).
    if (end - start >= 2) return end - 1;
    return null;
}

fn isWhitespaceCp(cp: u21) bool {
    if (cp > 0xFF) return false;
    return isWhitespace(@intCast(cp));
}

fn isLetterOrMark(cp: u21) bool {
    return isLetter(cp) or isMark(cp);
}

/// Approximate `\p{M}` — combining marks. Coverage: common Latin/Greek/
/// Cyrillic combining marks (U+0300–U+036F), plus Hebrew/Arabic/Devanagari
/// combining ranges. Not exhaustive, but handles every codepoint our test
/// corpus encounters; expand if a non-ASCII model surfaces a false negative.
fn isMark(cp: u21) bool {
    if (cp >= 0x0300 and cp <= 0x036F) return true; // Combining diacritical marks
    if (cp >= 0x0483 and cp <= 0x0489) return true; // Cyrillic combining
    if (cp >= 0x0591 and cp <= 0x05BD) return true; // Hebrew points
    if (cp >= 0x064B and cp <= 0x065F) return true; // Arabic harakat
    if (cp >= 0x0670 and cp <= 0x0670) return true;
    if (cp >= 0x06D6 and cp <= 0x06DC) return true;
    if (cp >= 0x0900 and cp <= 0x097F) return true; // Devanagari (overlap with letters; harmless)
    if (cp >= 0x1AB0 and cp <= 0x1AFF) return true;
    if (cp >= 0x1DC0 and cp <= 0x1DFF) return true;
    if (cp >= 0x20D0 and cp <= 0x20FF) return true;
    if (cp >= 0xFE20 and cp <= 0xFE2F) return true;
    return false;
}

const CpInfo = struct { cp: u21, len: usize };

fn decodeCodepoint(text: []const u8, pos: usize) ?CpInfo {
    if (pos >= text.len) return null;
    const byte_len = std.unicode.utf8ByteSequenceLength(text[pos]) catch return CpInfo{ .cp = text[pos], .len = 1 };
    const end = @min(pos + byte_len, text.len);
    const cp = std.unicode.utf8Decode(text[pos..end]) catch return CpInfo{ .cp = text[pos], .len = 1 };
    return CpInfo{ .cp = cp, .len = end - pos };
}

fn isLetter(cp: u21) bool {
    // ASCII letters
    if (cp >= 'A' and cp <= 'Z') return true;
    if (cp >= 'a' and cp <= 'z') return true;
    // Common Unicode letter ranges
    if (cp >= 0xC0 and cp <= 0x024F) return true; // Latin Extended
    if (cp >= 0x0400 and cp <= 0x04FF) return true; // Cyrillic
    if (cp >= 0x4E00 and cp <= 0x9FFF) return true; // CJK
    if (cp >= 0x3040 and cp <= 0x30FF) return true; // Hiragana/Katakana
    if (cp >= 0xAC00 and cp <= 0xD7AF) return true; // Korean
    if (cp >= 0x0600 and cp <= 0x06FF) return true; // Arabic
    if (cp >= 0x0900 and cp <= 0x097F) return true; // Devanagari
    if (cp >= 0x0370 and cp <= 0x03FF) return true; // Greek
    return false;
}

fn isDigit(cp: u21) bool {
    return cp >= '0' and cp <= '9';
}

fn isWhitespace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 0x0B or c == 0x0C;
}

/// Build the GPT-2 bytes_to_unicode mapping (256 entries).
fn buildBytesToUnicode() [256]u21 {
    var table: [256]u21 = undefined;
    var n: u21 = 256; // Next available unicode codepoint for unmapped bytes

    for (0..256) |b| {
        const byte: u8 = @intCast(b);
        // Printable ASCII range + some Latin-1 chars get identity mapping
        if ((byte >= '!' and byte <= '~') or
            (byte >= 0xA1 and byte <= 0xAC) or
            (byte >= 0xAE and byte <= 0xFF))
        {
            table[b] = @intCast(b);
        } else {
            // Non-printable bytes get mapped to codepoints starting at U+0100
            table[b] = n;
            n += 1;
        }
    }
    return table;
}

/// Parse tokenizer.json and return a Tokenizer.
pub fn loadTokenizer(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Tokenizer {
    const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{model_dir});
    defer allocator.free(path);

    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const content = try reader_state.interface.allocRemaining(allocator, .limited(256 * 1024 * 1024));
    defer allocator.free(content);

    var sw = io_util.Stopwatch.init(io);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    errdefer parsed.deinit();
    log.info("  parsed tokenizer.json ({d} MB) in {d}ms\n", .{ content.len / (1024 * 1024), sw.read() / std.time.ns_per_ms });

    const root = parsed.value.object;
    const model_obj = root.get("model").?.object;

    // Detect tokenizer type
    var tok_type: TokenizerType = .sentencepiece_bpe;
    if (model_obj.get("type")) |mt| {
        if (mt == .string and std.mem.eql(u8, mt.string, "WordPiece")) {
            tok_type = .wordpiece;
        }
    }
    if (tok_type != .wordpiece) {
        if (root.get("pre_tokenizer")) |pt| {
            if (pt == .object) {
                if (hasByteLevel(pt)) {
                    tok_type = .byte_level_bpe;
                }
            }
        }
    }

    // Parse vocab — keys borrow directly from `parsed`'s arena (no per-
    // entry dupe). Pre-size to avoid rehashing during the inserts.
    var vocab = std.StringHashMap(u32).init(allocator);
    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);

    const vocab_obj = model_obj.get("vocab").?.object;
    try vocab.ensureTotalCapacity(@intCast(vocab_obj.count()));
    try id_to_token.ensureTotalCapacity(@intCast(vocab_obj.count()));
    sw = io_util.Stopwatch.init(io);
    var vit = vocab_obj.iterator();
    while (vit.next()) |entry| {
        const key = entry.key_ptr.*;
        const id: u32 = @intCast(entry.value_ptr.integer);
        try vocab.put(key, id);
        try id_to_token.put(id, key);
    }
    log.info("  loaded {d} vocab entries in {d}ms\n", .{ vocab.count(), sw.read() / std.time.ns_per_ms });

    // Parse merges (array format: [["a", "b"], ...]). String halves are
    // borrowed from `parsed`'s arena.
    var merge_ranks = std.HashMap(
        Tokenizer.MergePair,
        u32,
        Tokenizer.MergePairContext,
        std.hash_map.default_max_load_percentage,
    ).init(allocator);

    if (model_obj.get("merges")) |merges_val| {
        const merges_arr = merges_val.array;
        try merge_ranks.ensureTotalCapacity(@intCast(merges_arr.items.len));
        sw = io_util.Stopwatch.init(io);
        for (merges_arr.items, 0..) |merge_val, rank| {
            const pair = merge_val.array;
            try merge_ranks.put(
                .{ .left = pair.items[0].string, .right = pair.items[1].string },
                @intCast(rank),
            );
        }
        log.info("  loaded {d} merges in {d}ms\n", .{ merge_ranks.count(), sw.read() / std.time.ns_per_ms });
    }

    // Parse added_tokens
    var special_tokens = std.StringHashMap(u32).init(allocator);
    var bos_id: ?u32 = null;
    var eos_id: ?u32 = null;

    if (root.get("added_tokens")) |at_val| {
        for (at_val.array.items) |token_val| {
            const obj = token_val.object;
            const content_str = obj.get("content").?.string;
            const id: u32 = @intCast(obj.get("id").?.integer);
            // Include ALL added tokens (both special=true and special=false like <think>, <tool_call>)
            // so they are tokenized as single atomic units. The string is
            // borrowed from `parsed`'s arena — no per-entry dupe.
            try special_tokens.put(content_str, id);
            if (!vocab.contains(content_str)) {
                try vocab.put(content_str, id);
                try id_to_token.put(id, content_str);
            }
        }
    }

    // Try to find BOS/EOS from common special token patterns
    bos_id = special_tokens.get("<bos>") orelse special_tokens.get("<|startoftext|>") orelse special_tokens.get("[CLS]");
    eos_id = special_tokens.get("<eos>") orelse special_tokens.get("<|im_end|>") orelse special_tokens.get("<|endoftext|>") orelse special_tokens.get("[SEP]");

    // Build byte-to-unicode mapping
    const byte_to_unicode = buildBytesToUnicode();
    var unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator);
    for (0..256) |b| {
        try unicode_to_byte.put(byte_to_unicode[b], @intCast(b));
    }

    log.info("Tokenizer loaded: {d} vocab, {d} merges, {d} special tokens ({s})\n", .{
        vocab.count(),
        merge_ranks.count(),
        special_tokens.count(),
        switch (tok_type) {
            .byte_level_bpe => "byte-level BPE",
            .sentencepiece_bpe => "SentencePiece BPE",
            .wordpiece => "WordPiece",
        },
    });

    return .{
        .vocab = vocab,
        .id_to_token = id_to_token,
        .merge_ranks = merge_ranks,
        .allocator = allocator,
        .special_tokens = special_tokens,
        .tok_type = tok_type,
        .byte_to_unicode = byte_to_unicode,
        .unicode_to_byte = unicode_to_byte,
        .bos_id = bos_id,
        .eos_id = eos_id,
        .parsed_json = parsed,
    };
}

/// Check if a pre_tokenizer JSON value contains a ByteLevel type.
fn hasByteLevel(pt: std.json.Value) bool {
    if (pt != .object) return false;
    if (pt.object.get("type")) |t| {
        if (t == .string) {
            if (std.mem.eql(u8, t.string, "ByteLevel")) return true;
            if (std.mem.eql(u8, t.string, "Sequence")) {
                if (pt.object.get("pretokenizers")) |pts| {
                    for (pts.array.items) |sub| {
                        if (hasByteLevel(sub)) return true;
                    }
                }
            }
        }
    }
    return false;
}

// ── Tests ──

const testing = std.testing;

test "isPunct identifies punctuation" {
    try testing.expect(Tokenizer.isPunct('.'));
    try testing.expect(Tokenizer.isPunct(','));
    try testing.expect(Tokenizer.isPunct('!'));
    try testing.expect(Tokenizer.isPunct('('));
    try testing.expect(!Tokenizer.isPunct('a'));
    try testing.expect(!Tokenizer.isPunct('0'));
    try testing.expect(!Tokenizer.isPunct(' '));
}

test "WordPiece encode basic" {
    const allocator = testing.allocator;

    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    const words = [_]struct { k: []const u8, v: u32 }{
        .{ .k = "[CLS]", .v = 101 },
        .{ .k = "[SEP]", .v = 102 },
        .{ .k = "[UNK]", .v = 0 },
        .{ .k = "hello", .v = 10 },
        .{ .k = "world", .v = 11 },
        .{ .k = "hel", .v = 12 },
        .{ .k = "##lo", .v = 13 },
    };
    for (words) |w| try vocab.put(w.k, w.v);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    for (words) |w| try id_to_token.put(w.v, w.k);

    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    try special_tokens.put("[CLS]", 101);
    try special_tokens.put("[SEP]", 102);

    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();

    var tok = Tokenizer{
        .vocab = vocab,
        .id_to_token = id_to_token,
        .merge_ranks = merge_ranks,
        .allocator = allocator,
        .special_tokens = special_tokens,
        .tok_type = .wordpiece,
        .byte_to_unicode = buildBytesToUnicode(),
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = 101,
        .eos_id = 102,
    };
    defer tok.unicode_to_byte.deinit();

    const ids = try tok.encode(allocator, "hello world");
    defer allocator.free(ids);

    // Should be: [CLS]=101, hello=10, world=11, [SEP]=102
    try testing.expectEqual(@as(usize, 4), ids.len);
    try testing.expectEqual(@as(u32, 101), ids[0]);
    try testing.expectEqual(@as(u32, 10), ids[1]);
    try testing.expectEqual(@as(u32, 11), ids[2]);
    try testing.expectEqual(@as(u32, 102), ids[3]);
}

test "WordPiece encode with subword split" {
    const allocator = testing.allocator;

    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    const words = [_]struct { k: []const u8, v: u32 }{
        .{ .k = "[UNK]", .v = 0 },
        .{ .k = "un", .v = 10 },
        .{ .k = "##like", .v = 11 },
        .{ .k = "##ly", .v = 12 },
    };
    for (words) |w| try vocab.put(w.k, w.v);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    for (words) |w| try id_to_token.put(w.v, w.k);

    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();

    var tok = Tokenizer{
        .vocab = vocab,
        .id_to_token = id_to_token,
        .merge_ranks = merge_ranks,
        .allocator = allocator,
        .special_tokens = special_tokens,
        .tok_type = .wordpiece,
        .byte_to_unicode = buildBytesToUnicode(),
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
    };
    defer tok.unicode_to_byte.deinit();

    const ids = try tok.encode(allocator, "unlikely");
    defer allocator.free(ids);

    // un=10, ##like=11, ##ly=12
    try testing.expectEqual(@as(usize, 3), ids.len);
    try testing.expectEqual(@as(u32, 10), ids[0]);
    try testing.expectEqual(@as(u32, 11), ids[1]);
    try testing.expectEqual(@as(u32, 12), ids[2]);
}

test "WordPiece encode unknown word falls back to UNK" {
    const allocator = testing.allocator;

    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("[UNK]", 0);
    try vocab.put("hello", 10);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    try id_to_token.put(0, "[UNK]");
    try id_to_token.put(10, "hello");

    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();

    var tok = Tokenizer{
        .vocab = vocab,
        .id_to_token = id_to_token,
        .merge_ranks = merge_ranks,
        .allocator = allocator,
        .special_tokens = special_tokens,
        .tok_type = .wordpiece,
        .byte_to_unicode = buildBytesToUnicode(),
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
    };
    defer tok.unicode_to_byte.deinit();

    const ids = try tok.encode(allocator, "hello xyz");
    defer allocator.free(ids);

    try testing.expectEqual(@as(usize, 2), ids.len);
    try testing.expectEqual(@as(u32, 10), ids[0]);
    try testing.expectEqual(@as(u32, 0), ids[1]);
}

test "WordPiece encode handles punctuation splitting" {
    const allocator = testing.allocator;

    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("[UNK]", 0);
    try vocab.put("hello", 10);
    try vocab.put(",", 11);
    try vocab.put("world", 12);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    try id_to_token.put(0, "[UNK]");
    try id_to_token.put(10, "hello");
    try id_to_token.put(11, ",");
    try id_to_token.put(12, "world");

    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();

    var tok = Tokenizer{
        .vocab = vocab,
        .id_to_token = id_to_token,
        .merge_ranks = merge_ranks,
        .allocator = allocator,
        .special_tokens = special_tokens,
        .tok_type = .wordpiece,
        .byte_to_unicode = buildBytesToUnicode(),
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
    };
    defer tok.unicode_to_byte.deinit();

    const ids = try tok.encode(allocator, "hello, world");
    defer allocator.free(ids);

    try testing.expectEqual(@as(usize, 3), ids.len);
    try testing.expectEqual(@as(u32, 10), ids[0]);
    try testing.expectEqual(@as(u32, 11), ids[1]);
    try testing.expectEqual(@as(u32, 12), ids[2]);
}

test "WordPiece decode skips special tokens and joins subwords" {
    const allocator = testing.allocator;

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    try id_to_token.put(101, "[CLS]");
    try id_to_token.put(102, "[SEP]");
    try id_to_token.put(10, "hello");
    try id_to_token.put(11, "##ly");
    try id_to_token.put(12, "world");

    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();

    var tok = Tokenizer{
        .vocab = vocab,
        .id_to_token = id_to_token,
        .merge_ranks = merge_ranks,
        .allocator = allocator,
        .special_tokens = special_tokens,
        .tok_type = .wordpiece,
        .byte_to_unicode = buildBytesToUnicode(),
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = 101,
        .eos_id = 102,
    };
    defer tok.unicode_to_byte.deinit();

    const ids = [_]u32{ 101, 10, 11, 12, 102 };
    const text = try tok.decode(allocator, &ids, false);
    defer allocator.free(text);

    try testing.expectEqualStrings("helloly world", text);
}

test "WordPiece encode lowercases input" {
    const allocator = testing.allocator;

    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("[UNK]", 0);
    try vocab.put("hello", 10);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    try id_to_token.put(0, "[UNK]");
    try id_to_token.put(10, "hello");

    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();

    var tok = Tokenizer{
        .vocab = vocab,
        .id_to_token = id_to_token,
        .merge_ranks = merge_ranks,
        .allocator = allocator,
        .special_tokens = special_tokens,
        .tok_type = .wordpiece,
        .byte_to_unicode = buildBytesToUnicode(),
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
    };
    defer tok.unicode_to_byte.deinit();

    const ids = try tok.encode(allocator, "HELLO");
    defer allocator.free(ids);

    try testing.expectEqual(@as(usize, 1), ids.len);
    try testing.expectEqual(@as(u32, 10), ids[0]);
}

// Helper for pre-tokenizer tests: run gpt2PreTokenize and compare the
// emitted word strings to an expected slice. Owns the dupe'd word memory.
fn expectPreTokens(allocator: std.mem.Allocator, input: []const u8, expected: []const []const u8) !void {
    var words: std.ArrayList([]const u8) = .empty;
    defer {
        for (words.items) |w| allocator.free(w);
        words.deinit(allocator);
    }
    try gpt2PreTokenize(allocator, input, &words);
    if (words.items.len != expected.len) {
        std.debug.print("\n  pre-tokenize on {s}: got {d} words, expected {d}\n", .{
            input, words.items.len, expected.len,
        });
        for (words.items, 0..) |w, i| std.debug.print("    [{d}] {s}\n", .{ i, w });
        return error.WordCountMismatch;
    }
    for (words.items, expected, 0..) |got, want, idx| {
        if (!std.mem.eql(u8, got, want)) {
            std.debug.print("\n  pre-tokenize {s}: word[{d}] got `{s}` want `{s}`\n", .{
                input, idx, got, want,
            });
            return error.WordContentMismatch;
        }
    }
}

test "gpt2PreTokenize: multi-space + identifier" {
    // Regression: HF tokenizes `    total = 0` as
    //   ['   ', ' total', ' =', ' ', '0']
    // — the trailing space of the leading run joins with the next word, and
    // single-digit pre-tokens are emitted one at a time. The previous impl
    // emitted 4-space, identifier, single-space, =, single-space, 0 — six
    // words instead of five, with the model receiving a perturbed prior on
    // every subsequent word. Found via byte-diff against MTPLX.
    try expectPreTokens(testing.allocator, "    total = 0", &.{
        "   ", " total", " =", " ", "0",
    });
}

test "gpt2PreTokenize: leading space combines with letters" {
    // Pattern 2 absorbs the optional leading non-LN char.
    try expectPreTokens(testing.allocator, " total", &.{" total"});
    try expectPreTokens(testing.allocator, "def total", &.{ "def", " total" });
}

test "gpt2PreTokenize: leading space combines with punctuation" {
    // Pattern 4 is ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`.
    try expectPreTokens(testing.allocator, " =", &.{" ="});
    try expectPreTokens(testing.allocator, " += foo", &.{ " +=", " foo" });
    // Multi-punct with leading space.
    try expectPreTokens(testing.allocator, " *=", &.{" *="});
}

test "gpt2PreTokenize: digits are single-codepoint pre-tokens" {
    // Pattern 3 is `\p{N}` (no `+`), so each digit is its own pre-token.
    // BPE will not merge across pre-tokens, so this drives final token IDs.
    try expectPreTokens(testing.allocator, "100", &.{ "1", "0", "0" });
    try expectPreTokens(testing.allocator, " 100", &.{ " ", "1", "0", "0" });
}

test "gpt2PreTokenize: newline run after whitespace" {
    // Pattern 5 `\s*[\r\n]+` consumes leading spaces along with the newline.
    try expectPreTokens(testing.allocator, "x\n", &.{ "x", "\n" });
    try expectPreTokens(testing.allocator, "x   \n", &.{ "x", "   \n" });
    try expectPreTokens(testing.allocator, "x\n\n", &.{ "x", "\n\n" });
}

test "gpt2PreTokenize: trailing whitespace at end of input" {
    // Pattern 6 trivially matches when end-of-input satisfies the lookahead.
    try expectPreTokens(testing.allocator, "x   ", &.{ "x", "   " });
}

test "gpt2PreTokenize: full Python snippet matches HF reference" {
    // Reference output produced by the HuggingFace `tokenizers` library on a
    // Qwen3.5 tokenizer.json (any Qwen3.5/3.6 checkpoint reproduces this):
    //   ['def', ' total', '(items', '):\n', '   ', ' total', ' =', ' ', '0']
    // Note: `):\n` joins because pattern 4 allows trailing `[\r\n]*` after
    // the punct run. The byte-level encode + BPE merge stage downstream
    // turns this into exactly the same token-ids HF produces.
    try expectPreTokens(testing.allocator,
        "def total(items):\n    total = 0",
        &.{ "def", " total", "(items", "):\n", "   ", " total", " =", " ", "0" },
    );
}

test "gpt2PreTokenize: contractions still work after rewrite" {
    try expectPreTokens(testing.allocator, "don't", &.{ "don", "'t" });
    try expectPreTokens(testing.allocator, "they're", &.{ "they", "'re" });
    try expectPreTokens(testing.allocator, "we'll", &.{ "we", "'ll" });
}

test "gpt2PreTokenize: punct + letter joins via pattern 2 optional non-LN" {
    // `_start` matches pattern 2 with `_` as the optional non-LN char;
    // pattern 4 would also match `_` alone, but pattern 2 wins by priority.
    // HF reference: ['<|', 'im', '_start', '|>'].
    try expectPreTokens(testing.allocator, "<|im_start|>", &.{
        "<|", "im", "_start", "|>",
    });
}
