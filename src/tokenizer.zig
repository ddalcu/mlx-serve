const std = @import("std");

pub const TokenizerType = enum { sentencepiece_bpe, byte_level_bpe };

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

    /// Encode text to token IDs (no BOS/EOS added).
    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        return switch (self.tok_type) {
            .sentencepiece_bpe => self.encodeSentencePiece(allocator, text),
            .byte_level_bpe => self.encodeByteLevel(allocator, text),
        };
    }

    /// Decode token IDs to text.
    pub fn decode(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32, strip_leading_space: bool) ![]u8 {
        return switch (self.tok_type) {
            .sentencepiece_bpe => self.decodeSentencePiece(allocator, ids, strip_leading_space),
            .byte_level_bpe => self.decodeByteLevel(allocator, ids),
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

/// GPT-2 pre-tokenization: splits text using the GPT-2 regex pattern as a state machine.
/// Pattern: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
fn gpt2PreTokenize(allocator: std.mem.Allocator, text: []const u8, words: *std.ArrayList([]const u8)) !void {
    var i: usize = 0;
    while (i < text.len) {
        const start = i;

        // Try contraction: 's, 't, 're, 've, 'm, 'll, 'd (case insensitive)
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

        // Letter sequence (possibly starting with one non-letter non-digit non-newline char)
        const cp_start = decodeCodepoint(text, i);
        if (cp_start) |cp_info| {
            if (isLetter(cp_info.cp)) {
                // Pure letter sequence
                i += cp_info.len;
                while (i < text.len) {
                    const next_cp = decodeCodepoint(text, i) orelse break;
                    if (!isLetter(next_cp.cp)) break;
                    i += next_cp.len;
                }
                try words.append(allocator, try allocator.dupe(u8, text[start..i]));
                continue;
            }

            // Non-letter, non-digit, non-newline followed by letters?
            if (!isDigit(cp_info.cp) and cp_info.cp != '\r' and cp_info.cp != '\n') {
                const after_first = i + cp_info.len;
                if (after_first < text.len) {
                    const next_cp = decodeCodepoint(text, after_first);
                    if (next_cp != null and isLetter(next_cp.?.cp)) {
                        i = after_first + next_cp.?.len;
                        while (i < text.len) {
                            const lcp = decodeCodepoint(text, i) orelse break;
                            if (!isLetter(lcp.cp)) break;
                            i += lcp.len;
                        }
                        try words.append(allocator, try allocator.dupe(u8, text[start..i]));
                        continue;
                    }
                }
            }
        }

        // Digit sequence
        if (cp_start != null and isDigit(cp_start.?.cp)) {
            i += cp_start.?.len;
            while (i < text.len) {
                const dcp = decodeCodepoint(text, i) orelse break;
                if (!isDigit(dcp.cp)) break;
                i += dcp.len;
            }
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // Newline sequences: \s*[\r\n]+
        if (text[i] == '\r' or text[i] == '\n') {
            while (i < text.len and (text[i] == '\r' or text[i] == '\n')) {
                i += 1;
            }
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // Whitespace: consume spaces, handle [\r\n]* or trailing non-space
        if (isWhitespace(text[i])) {
            // Consume whitespace
            while (i < text.len and isWhitespace(text[i]) and text[i] != '\r' and text[i] != '\n') {
                i += 1;
            }
            // Check if all remaining is whitespace (the (?!\S) lookahead)
            if (i >= text.len) {
                // Trailing whitespace
                try words.append(allocator, try allocator.dupe(u8, text[start..i]));
                continue;
            }
            if (text[i] == '\r' or text[i] == '\n') {
                // Whitespace before newline
                while (i < text.len and (text[i] == '\r' or text[i] == '\n')) {
                    i += 1;
                }
                try words.append(allocator, try allocator.dupe(u8, text[start..i]));
                continue;
            }
            // Space before non-space: emit the space as part of the next token
            // (GPT-2 pattern: " ?[^\s\p{L}\p{N}]+..." or just \s+)
            // The space joins with the next word
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // [^\s\p{L}\p{N}]+ sequence (punctuation/symbols): optional leading space
        if (cp_start != null and !isLetter(cp_start.?.cp) and !isDigit(cp_start.?.cp) and !isWhitespace(text[i])) {
            i += cp_start.?.len;
            while (i < text.len) {
                if (isWhitespace(text[i])) break;
                const pcp = decodeCodepoint(text, i) orelse break;
                if (isLetter(pcp.cp) or isDigit(pcp.cp)) break;
                // Stop at contractions
                if (text[i] == '\'') break;
                i += pcp.len;
            }
            // Consume trailing \r\n
            while (i < text.len and (text[i] == '\r' or text[i] == '\n')) {
                i += 1;
            }
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // Fallback: single byte
        i += 1;
        try words.append(allocator, try allocator.dupe(u8, text[start..i]));
    }
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
pub fn loadTokenizer(allocator: std.mem.Allocator, model_dir: []const u8) !Tokenizer {
    const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{model_dir});
    defer allocator.free(path);

    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 256 * 1024 * 1024);
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    const model_obj = root.get("model").?.object;

    // Detect tokenizer type
    var tok_type: TokenizerType = .sentencepiece_bpe;
    if (root.get("pre_tokenizer")) |pt| {
        if (pt == .object) {
            // Check for ByteLevel pre-tokenizer (directly or in a Sequence)
            if (hasByteLevel(pt)) {
                tok_type = .byte_level_bpe;
            }
        }
    }

    // Parse vocab
    var vocab = std.StringHashMap(u32).init(allocator);
    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);

    const vocab_obj = model_obj.get("vocab").?.object;
    var vit = vocab_obj.iterator();
    while (vit.next()) |entry| {
        const key = try allocator.dupe(u8, entry.key_ptr.*);
        const id: u32 = @intCast(entry.value_ptr.integer);
        try vocab.put(key, id);
        try id_to_token.put(id, key);
    }

    // Parse merges (array format: [["a", "b"], ...])
    var merge_ranks = std.HashMap(
        Tokenizer.MergePair,
        u32,
        Tokenizer.MergePairContext,
        std.hash_map.default_max_load_percentage,
    ).init(allocator);

    const merges_arr = model_obj.get("merges").?.array;
    for (merges_arr.items, 0..) |merge_val, rank| {
        const pair = merge_val.array;
        const left = try allocator.dupe(u8, pair.items[0].string);
        const right = try allocator.dupe(u8, pair.items[1].string);
        try merge_ranks.put(
            .{ .left = left, .right = right },
            @intCast(rank),
        );
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
            const special = if (obj.get("special")) |s| s.bool else false;
            if (special) {
                const key = try allocator.dupe(u8, content_str);
                try special_tokens.put(key, id);
                // Also add to vocab/id_to_token so they can be decoded
                if (!vocab.contains(key)) {
                    const key2 = try allocator.dupe(u8, content_str);
                    try vocab.put(key2, id);
                    try id_to_token.put(id, key2);
                }
            }
        }
    }

    // Try to find BOS/EOS from common special token patterns
    bos_id = special_tokens.get("<bos>") orelse special_tokens.get("<|startoftext|>");
    eos_id = special_tokens.get("<eos>") orelse special_tokens.get("<|im_end|>") orelse special_tokens.get("<|endoftext|>");

    // Build byte-to-unicode mapping
    const byte_to_unicode = buildBytesToUnicode();
    var unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator);
    for (0..256) |b| {
        try unicode_to_byte.put(byte_to_unicode[b], @intCast(b));
    }

    std.debug.print("Tokenizer loaded: {d} vocab, {d} merges, {d} special tokens ({s})\n", .{
        vocab.count(),
        merge_ranks.count(),
        special_tokens.count(),
        if (tok_type == .byte_level_bpe) "byte-level BPE" else "SentencePiece BPE",
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
