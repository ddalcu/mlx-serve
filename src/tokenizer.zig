const std = @import("std");
const log = @import("log.zig");
const io_util = @import("io_util.zig");

pub const TokenizerType = enum { sentencepiece_bpe, byte_level_bpe, wordpiece };

/// An added token flagged `special: true` in tokenizer.json.
pub const FlaggedSpecial = struct { id: u32, content: []const u8 };

/// The ids a sampler must never draw: added tokens flagged `special: true`,
/// MINUS the ones that are legitimate output. Legitimacy is DERIVED, never a
/// hardcoded list (which would break thinking/tool-calling on the next arch):
///   - EOS/stop ids (`exempt_ids`) stay reachable, and
///   - any special whose literal text appears in the chat template source
///     stays reachable — thinking tags, tool markers and role markers all
///     ride their model's own template.
/// An empty `template_source` disables suppression entirely (returns no ids):
/// with no template to consult, "what can this model legitimately emit" has
/// no honest answer, and fallback-formatted models keep pre-change behavior.
/// A reserved marker like `<|fim_hole|>` in chat output is always a bug
/// regardless of what produced it (a collapsed distribution can rank one top-5 at a
/// degenerate position); the substring rule errs toward exemption, which can
/// only ever keep a token reachable.
pub fn reservedOutputIds(
    allocator: std.mem.Allocator,
    flagged: []const FlaggedSpecial,
    template_source: []const u8,
    exempt_ids: []const u32,
) ![]u32 {
    if (template_source.len == 0) return allocator.alloc(u32, 0);
    var out = std.ArrayList(u32).empty;
    errdefer out.deinit(allocator);
    outer: for (flagged) |sp| {
        for (exempt_ids) |e| {
            if (sp.id == e) continue :outer;
        }
        if (std.mem.indexOf(u8, template_source, sp.content) != null) continue;
        try out.append(allocator, sp.id);
    }
    return out.toOwnedSlice(allocator);
}

/// BPE tokenizer supporting both SentencePiece (Gemma) and byte-level (GPT-2/Qwen3) modes.
pub const PretokStyle = enum { gpt2, llama3 };

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
    /// Digits per pre-token for the byte-level path: 1 (`\p{N}`, Qwen/GPT-2)
    /// or 3 (`\p{N}{1,3}`, DeepSeek-V4/Llama-3) — parsed from tokenizer.json.
    digit_group: u8 = 1,
    /// Pre-tokenizer grammar: .gpt2 (Qwen/GPT-2-style) or .llama3
    /// (Muse-Glimmer/Llama-3-style: case-transition word splits, attached
    /// (?i) contractions, {1,3} digit groups, `/` in the punct tail).
    /// Parsed from the tokenizer.json Split regex.
    pretok_style: PretokStyle = .gpt2,
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

    /// Added tokens flagged `special: true` in tokenizer.json — the
    /// candidate set for reserved-output suppression. NOT the same as
    /// `special_tokens`, which deliberately holds ALL added tokens
    /// (`special: false` entries like `<think>` are atomic-encode units but
    /// perfectly legitimate output). Content slices borrow `parsed_json`'s
    /// arena; the slice itself is owned and freed in deinit.
    flagged_specials: []const FlaggedSpecial = &.{},

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

    /// A decode-capable tokenizer with EMPTY maps, for tests in other modules
    /// that need to turn ids into text without a checkpoint on disk. The caller
    /// owns every map and must `deinit` them; nothing here reads tokenizer.json,
    /// so encode is not meaningfully usable — populate `id_to_token` and decode.
    pub fn initEmptyForTests(allocator: std.mem.Allocator, tok_type: TokenizerType) Tokenizer {
        return .{
            .vocab = std.StringHashMap(u32).init(allocator),
            .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
            .merge_ranks = std.HashMap(MergePair, u32, MergePairContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
            .special_tokens = std.StringHashMap(u32).init(allocator),
            .tok_type = tok_type,
            .byte_to_unicode = buildBytesToUnicode(),
            .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
            .bos_id = null,
            .eos_id = null,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        // Map keys/values either point into `parsed_json`'s arena (no
        // per-entry free needed) or were duped explicitly when no parsed
        // JSON is held (e.g., the test-only constructors). Freeing the
        // parsed JSON deinits its arena in one shot.
        if (self.flagged_specials.len > 0) self.allocator.free(self.flagged_specials);
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

        // Special-token splitter. The old scan re-searched the ENTIRE
        // remaining text for EVERY special token per segment
        // (O(specials × text) — ~12 s per 66 KB prompt on gemma-3's
        // 6415-special vocabulary; gemma-4's 24 specials never noticed).
        // Instead: bucket the specials by first byte once per call (~µs),
        // then a single left-to-right pass tries only the candidates whose
        // first byte matches. Semantics unchanged — earliest occurrence
        // wins, longest special wins at the same position (buckets are
        // sorted by descending length, so the first hit is the longest).
        const n_special = self.special_tokens.count();
        const Cand = struct { bytes: []const u8, id: u32 };
        const cands = try allocator.alloc(Cand, n_special);
        defer allocator.free(cands);
        {
            var i: usize = 0;
            var sit = self.special_tokens.iterator();
            while (sit.next()) |entry| : (i += 1) {
                cands[i] = .{ .bytes = entry.key_ptr.*, .id = entry.value_ptr.* };
            }
        }
        std.mem.sort(Cand, cands, {}, struct {
            fn lessThan(_: void, a: Cand, b: Cand) bool {
                const ab: u8 = if (a.bytes.len > 0) a.bytes[0] else 0;
                const bb: u8 = if (b.bytes.len > 0) b.bytes[0] else 0;
                if (ab != bb) return ab < bb;
                return a.bytes.len > b.bytes.len;
            }
        }.lessThan);
        // bucket_start[b]..bucket_start[b+1] = candidates whose first byte is b.
        var bucket_start: [257]u32 = @splat(0);
        {
            var ci: usize = 0;
            for (0..256) |b| {
                bucket_start[b] = @intCast(ci);
                while (ci < cands.len and cands[ci].bytes.len > 0 and cands[ci].bytes[0] == b) ci += 1;
            }
            bucket_start[256] = @intCast(cands.len);
        }

        var pos: usize = 0;
        var seg_start: usize = 0;
        while (pos < text.len) {
            const b = text[pos];
            var ci = bucket_start[b];
            const cend = bucket_start[@as(usize, b) + 1];
            var matched: ?Cand = null;
            while (ci < cend) : (ci += 1) {
                const c = cands[ci];
                if (c.bytes.len <= text.len - pos and std.mem.startsWith(u8, text[pos..], c.bytes)) {
                    matched = c;
                    break;
                }
            }
            if (matched) |m| {
                if (pos > seg_start) {
                    const ids = try self.encodeSegment(allocator, text[seg_start..pos]);
                    defer allocator.free(ids);
                    try result.appendSlice(allocator, ids);
                }
                try result.append(allocator, m.id);
                pos += m.bytes.len;
                seg_start = pos;
            } else {
                pos += 1;
            }
        }
        if (seg_start < text.len) {
            const ids = try self.encodeSegment(allocator, text[seg_start..]);
            defer allocator.free(ids);
            try result.appendSlice(allocator, ids);
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

    /// Reserved-output ids for this tokenizer: `reservedOutputIds` over the
    /// `special: true` added tokens recorded at load. See that function for
    /// the derivation rules.
    pub fn reservedIds(self: *const Tokenizer, allocator: std.mem.Allocator, template_source: []const u8, exempt_ids: []const u32) ![]u32 {
        return reservedOutputIds(allocator, self.flagged_specials, template_source, exempt_ids);
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
        switch (self.pretok_style) {
            .gpt2 => try gpt2PreTokenize(allocator, text, self.digit_group, &words),
            .llama3 => try llama3PreTokenize(allocator, text, &words),
        }

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

    /// One symbol in the BPE merge worklist. Symbols are always contiguous
    /// slices `input[start..end]` — merging adjacent symbols just extends the
    /// left node's range and unlinks the right, so no bytes are ever copied.
    /// `ver` invalidates stale heap candidates: any merge touching a node
    /// bumps its version, and a candidate is only applied when both stored
    /// versions still match.
    const BpeNode = struct {
        start: u32,
        end: u32,
        prev: i32,
        next: i32,
        ver: u32,
    };

    /// Candidate pair in the merge heap, ordered by (rank, left node index).
    /// Node indices are creation order = text order and merges never create
    /// nodes, so the secondary key reproduces the naive scan's leftmost-wins
    /// tie-break exactly.
    const BpeCand = struct {
        rank: u32,
        left: u32,
        right: u32,
        lver: u32,
        rver: u32,

        fn order(_: void, a: BpeCand, b: BpeCand) std.math.Order {
            if (a.rank != b.rank) return std.math.order(a.rank, b.rank);
            return std.math.order(a.left, b.left);
        }
    };

    const BpeHeap = std.PriorityQueue(BpeCand, void, BpeCand.order);

    fn bpePushCand(self: *const Tokenizer, allocator: std.mem.Allocator, heap: *BpeHeap, nodes: []const BpeNode, input: []const u8, l: u32, r: u32) !void {
        const pair = MergePair{
            .left = input[nodes[l].start..nodes[l].end],
            .right = input[nodes[r].start..nodes[r].end],
        };
        if (self.merge_ranks.get(pair)) |rank| {
            try heap.push(allocator, .{ .rank = rank, .left = l, .right = r, .lver = nodes[l].ver, .rver = nodes[r].ver });
        }
    }

    /// Greedy BPE: repeatedly merge the lowest-ranked adjacent pair
    /// (leftmost on ties) until no pair has a rank. Heap + linked-list,
    /// O(n log n); `encodeSentencePiece` feeds entire prompts through here
    /// with no pre-tokenization, so the previous rescan-all-pairs loop was
    /// O(n²) and cost seconds on agent-sized (tens-of-KB) system prompts.
    fn bpeMerge(self: *const Tokenizer, allocator: std.mem.Allocator, input: []const u8) ![]u32 {
        // Split into individual UTF-8 characters.
        var nodes: std.ArrayList(BpeNode) = .empty;
        defer nodes.deinit(allocator);

        var idx: usize = 0;
        while (idx < input.len) {
            const char_len = std.unicode.utf8ByteSequenceLength(input[idx]) catch 1;
            const end = @min(idx + char_len, input.len);
            const i: i32 = @intCast(nodes.items.len);
            try nodes.append(allocator, .{
                .start = @intCast(idx),
                .end = @intCast(end),
                .prev = i - 1,
                .next = if (end < input.len) i + 1 else -1,
                .ver = 0,
            });
            idx = end;
        }

        var heap: BpeHeap = .initContext({});
        defer heap.deinit(allocator);

        if (nodes.items.len > 1) {
            for (0..nodes.items.len - 1) |j| {
                try self.bpePushCand(allocator, &heap, nodes.items, input, @intCast(j), @intCast(j + 1));
            }
        }

        while (heap.pop()) |c| {
            const ns = nodes.items;
            if (ns[c.left].ver != c.lver or ns[c.right].ver != c.rver) continue;

            // Merge right into left: extend range, unlink right, invalidate
            // every candidate that referenced either node's old content.
            ns[c.left].end = ns[c.right].end;
            ns[c.left].ver += 1;
            ns[c.right].ver += 1;
            ns[c.left].next = ns[c.right].next;
            if (ns[c.right].next >= 0) ns[@intCast(ns[c.right].next)].prev = @intCast(c.left);

            if (ns[c.left].prev >= 0) try self.bpePushCand(allocator, &heap, ns, input, @intCast(ns[c.left].prev), c.left);
            if (ns[c.left].next >= 0) try self.bpePushCand(allocator, &heap, ns, input, c.left, @intCast(ns[c.left].next));
        }

        // Map surviving symbols to vocab IDs.
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(allocator);

        var cur: i32 = if (nodes.items.len > 0) 0 else -1;
        while (cur >= 0) : (cur = nodes.items[@intCast(cur)].next) {
            const n = nodes.items[@intCast(cur)];
            const sym = input[n.start..n.end];
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
/// `digit_group`: how many consecutive digits form one pre-token. 1 = the
/// Qwen/GPT-2 `\p{N}` rule; 3 = the DeepSeek-V4/Llama-3 `\p{N}{1,3}` rule
/// (greedy left-to-right groups). Parsed per model from tokenizer.json —
/// feeding a {1,3}-trained model per-digit pre-tokens puts every number
/// off-distribution (the echo-precision slip class: `1o` for `10`, split
/// digits, o-for-0 near-ties).
fn gpt2PreTokenize(allocator: std.mem.Allocator, text: []const u8, digit_group: u8, words: *std.ArrayList([]const u8)) !void {
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

        // ── Pattern 3: `\p{N}` or `\p{N}{1,3}` — up to digit_group digits ──
        if (decodeCodepoint(text, i)) |cp_info| {
            if (isDigit(cp_info.cp)) {
                i += cp_info.len;
                var taken: u8 = 1;
                while (taken < digit_group) : (taken += 1) {
                    const next = decodeCodepoint(text, i) orelse break;
                    if (!isDigit(next.cp)) break;
                    i += next.len;
                }
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

/// Llama-3-style pre-tokenizer (Muse-Glimmer). Branch order mirrors the
/// tokenizer.json Split regex exactly:
///   1. [^\r\n\p{L}\p{N}]?[UPPER]*[LOWER]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
///   2. [^\r\n\p{L}\p{N}]?[UPPER]+[LOWER]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
///   3. \p{N}{1,3}
///   4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
///   5. \s*[\r\n]+   6. \s+(?!\S)   7. \s+
/// UPPER = \p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}, LOWER = \p{Ll}\p{Lm}\p{Lo}\p{M} —
/// caseless letters and marks belong to BOTH classes. Branches 1+2 collapse
/// into one deterministic scan (greedy upper-run then greedy lower-run, ≥1
/// letter total): for disjoint ASCII classes this reproduces the regex's
/// backtracking exactly, and for the overlap classes every reachable match
/// END coincides. Known gap: non-ASCII CASED letters (Cyrillic/Greek Lu/Ll)
/// are treated caseless, so a mid-word case transition there doesn't split —
/// same text, off-reference boundary; revisit with real Lu/Ll tables if a
/// checkpoint's traffic warrants it.
fn llama3PreTokenize(allocator: std.mem.Allocator, text: []const u8, words: *std.ArrayList([]const u8)) !void {
    var i: usize = 0;
    while (i < text.len) {
        const start = i;

        // ── Branches 1+2: optional non-LNN char + cased word + contraction ──
        if (llama3MatchWord(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // ── Branch 3: `\p{N}{1,3}` ──
        if (decodeCodepoint(text, i)) |cp_info| {
            if (isDigit(cp_info.cp)) {
                i += cp_info.len;
                var taken: u8 = 1;
                while (taken < 3) : (taken += 1) {
                    const next = decodeCodepoint(text, i) orelse break;
                    if (!isDigit(next.cp)) break;
                    i += next.len;
                }
                try words.append(allocator, try allocator.dupe(u8, text[start..i]));
                continue;
            }
        }

        // ── Branch 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*` (marks ride with punct,
        // and `/` joins the newline tail) ──
        if (llama3MatchPunct(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // ── Branches 5-7: identical whitespace grammar to gpt2 ──
        if (matchWhitespaceWithNewline(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }
        if (matchTrailingWhitespace(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }
        if (i < text.len and isWhitespace(text[i])) {
            while (i < text.len and isWhitespace(text[i])) i += 1;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        i += 1;
        try words.append(allocator, try allocator.dupe(u8, text[start..i]));
    }
}

/// UPPER class: \p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M} — ASCII exact, non-ASCII
/// letters/marks caseless (in both classes; see llama3PreTokenize doc).
fn llama3UpperClass(cp: u21) bool {
    if (cp < 128) return cp >= 'A' and cp <= 'Z';
    return isLetterOrMark(cp);
}

/// LOWER class: \p{Ll}\p{Lm}\p{Lo}\p{M}.
fn llama3LowerClass(cp: u21) bool {
    if (cp < 128) return cp >= 'a' and cp <= 'z';
    return isLetterOrMark(cp);
}

/// Branches 1+2 of the llama3 grammar. Returns the match end or null.
fn llama3MatchWord(text: []const u8, start: usize) ?usize {
    var i = start;
    const first = decodeCodepoint(text, i) orelse return null;
    // Optional single char that's NOT \r/\n/letter/digit (space and marks OK).
    if (!isLetter(first.cp) and !isDigit(first.cp) and first.cp != '\r' and first.cp != '\n') {
        // Consume it only if letters actually follow (regex optionality).
        const after = i + first.len;
        const next = decodeCodepoint(text, after) orelse return null;
        if (!llama3UpperClass(next.cp) and !llama3LowerClass(next.cp)) return null;
        i = after;
    }
    const word_start = i;
    // Greedy upper-run, then greedy lower-run; ≥1 letter total.
    while (decodeCodepoint(text, i)) |c| {
        if (!llama3UpperClass(c.cp)) break;
        i += c.len;
    }
    while (decodeCodepoint(text, i)) |c| {
        if (!llama3LowerClass(c.cp)) break;
        i += c.len;
    }
    if (i == word_start) return null;
    // Optional (?i) contraction suffix.
    if (i < text.len and text[i] == '\'' and i + 1 < text.len) {
        const n1 = std.ascii.toLower(text[i + 1]);
        if (n1 == 's' or n1 == 't' or n1 == 'm' or n1 == 'd') {
            i += 2;
        } else if (i + 2 < text.len) {
            const n2 = std.ascii.toLower(text[i + 2]);
            if ((n1 == 'r' and n2 == 'e') or (n1 == 'v' and n2 == 'e') or (n1 == 'l' and n2 == 'l')) {
                i += 3;
            }
        }
    }
    return i;
}

/// Branch 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*`. Unlike the gpt2 pattern, marks
/// belong to the punct class and `/` joins the trailing run.
fn llama3MatchPunct(text: []const u8, start: usize) ?usize {
    if (start >= text.len) return null;
    var p_start: usize = start;
    if (text[start] == ' ') p_start = start + 1;

    if (p_start >= text.len) return null;
    const first_cp = decodeCodepoint(text, p_start) orelse return null;
    if (isWhitespaceCp(first_cp.cp) or isLetter(first_cp.cp) or isDigit(first_cp.cp)) return null;

    var i: usize = p_start + first_cp.len;
    while (i < text.len) {
        const c = decodeCodepoint(text, i) orelse break;
        if (isWhitespaceCp(c.cp) or isLetter(c.cp) or isDigit(c.cp)) break;
        i += c.len;
    }
    while (i < text.len and (text[i] == '\r' or text[i] == '\n' or text[i] == '/')) i += 1;
    return i;
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
    return parseTokenizerContent(io, allocator, content);
}

/// Load a byte-level BPE tokenizer from the SLOW HF format (`vocab.json` +
/// `merges.txt`), for checkpoints that ship no `tokenizer.json` (e.g.
/// Qwen3-TTS). Synthesizes the equivalent `tokenizer.json` content and reuses
/// `parseTokenizerContent` so arena lifetime is handled identically.
pub fn loadTokenizerSlow(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Tokenizer {
    const vocab_path = try std.fmt.allocPrint(allocator, "{s}/vocab.json", .{model_dir});
    defer allocator.free(vocab_path);
    const vocab_json = try readFileAllocTok(io, allocator, vocab_path);
    defer allocator.free(vocab_json);

    const merges_path = try std.fmt.allocPrint(allocator, "{s}/merges.txt", .{model_dir});
    defer allocator.free(merges_path);
    const merges_txt = try readFileAllocTok(io, allocator, merges_path);
    defer allocator.free(merges_txt);

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, "{\"pre_tokenizer\":{\"type\":\"ByteLevel\"},\"model\":{\"type\":\"BPE\",\"vocab\":");
    try buf.appendSlice(allocator, std.mem.trim(u8, vocab_json, " \t\r\n"));
    try buf.appendSlice(allocator, ",\"merges\":[");
    var first = true;
    var lines = std.mem.splitScalar(u8, merges_txt, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        if (!first) try buf.append(allocator, ',');
        first = false;
        try buf.append(allocator, '"');
        try appendJsonEscapedTok(allocator, &buf, line);
        try buf.append(allocator, '"');
    }
    try buf.appendSlice(allocator, "]}}");
    return parseTokenizerContent(io, allocator, buf.items);
}

fn readFileAllocTok(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    return rs.interface.allocRemaining(allocator, .limited(256 * 1024 * 1024));
}

fn appendJsonEscapedTok(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), s: []const u8) !void {
    for (s) |c| switch (c) {
        '"' => try buf.appendSlice(allocator, "\\\""),
        '\\' => try buf.appendSlice(allocator, "\\\\"),
        else => {
            if (c < 0x20) {
                var tmp: [8]u8 = undefined;
                try buf.appendSlice(allocator, std.fmt.bufPrint(&tmp, "\\u{x:0>4}", .{c}) catch unreachable);
            } else try buf.append(allocator, c);
        },
    };
}

/// Try the fast `tokenizer.json` format, falling back to slow vocab.json+merges.txt.
pub fn loadTokenizerAny(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Tokenizer {
    return loadTokenizer(io, allocator, model_dir) catch |e| {
        if (e == error.FileNotFound) return loadTokenizerSlow(io, allocator, model_dir);
        return e;
    };
}

/// Parse a `tokenizer.json`-shaped JSON document into a `Tokenizer`. Split out
/// from `loadTokenizer` so the slow-format loader (`loadTokenizerSlow`) can feed
/// synthesized content. `parsed` (and the string keys borrowed from it) is owned
/// by the returned `Tokenizer.parsed_json`, so `content` may be freed after.
fn parseTokenizerContent(io: std.Io, allocator: std.mem.Allocator, content: []const u8) !Tokenizer {
    var sw = io_util.Stopwatch.init(io);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    errdefer parsed.deinit();
    log.info("  parsed tokenizer ({d} MB) in {d}ms\n", .{ content.len / (1024 * 1024), sw.read() / std.time.ns_per_ms });

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
            // Two on-disk formats: newer HF tokenizers store each merge as a
            // 2-element array `["a","b"]` (Qwen3, Gemma 4); older / GPT-2-style
            // exports store it as a single space-joined string `"a b"` (Qwen2.5,
            // many Llama/Mistral). `parseMergePair` handles both. Reading `.array`
            // off a string Value in ReleaseFast spins the loop on garbage, which
            // hung Qwen2.5 loads before this fix.
            const pair = parseMergePair(merge_val) orelse continue;
            try merge_ranks.put(pair, @intCast(rank));
        }
        log.info("  loaded {d} merges in {d}ms\n", .{ merge_ranks.count(), sw.read() / std.time.ns_per_ms });
    }

    // Parse added_tokens
    var special_tokens = std.StringHashMap(u32).init(allocator);
    var flagged = std.ArrayList(FlaggedSpecial).empty;
    errdefer flagged.deinit(allocator);
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
            // `special: true` entries additionally feed reserved-output
            // suppression (see `reservedOutputIds`).
            if (obj.get("special")) |sv| {
                if (sv == .bool and sv.bool) {
                    try flagged.append(allocator, .{ .id = id, .content = content_str });
                }
            }
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
        .flagged_specials = try flagged.toOwnedSlice(allocator),
        .tok_type = tok_type,
        .digit_group = if (root.get("pre_tokenizer")) |pt| digitGroupFromPreTokenizer(pt) else 1,
        .pretok_style = if (root.get("pre_tokenizer")) |pt| pretokStyleFromPreTokenizer(pt) else .gpt2,
        .byte_to_unicode = byte_to_unicode,
        .unicode_to_byte = unicode_to_byte,
        .bos_id = bos_id,
        .eos_id = eos_id,
        .parsed_json = parsed,
    };
}

/// Digits per pre-token from the tokenizer.json `pre_tokenizer` spec: a
/// `Split` rule (top-level or inside a `Sequence`) whose regex is exactly
/// `\p{N}{1,3}` selects 3-digit grouping; anything else keeps the GPT-2
/// single-digit rule. Deliberately exact-match — an unrecognized digit rule
/// keeps the conservative behavior rather than guessing.
fn digitGroupFromPreTokenizer(pt: std.json.Value) u8 {
    if (pt != .object) return 1;
    if (splitRegexIsDigits13(pt)) return 3;
    if (pt.object.get("pretokenizers")) |list| {
        if (list == .array) {
            for (list.array.items) |sub| {
                if (sub == .object and splitRegexIsDigits13(sub)) return 3;
            }
        }
    }
    return 1;
}

/// Pre-tokenizer grammar from the tokenizer.json Split regex. The muse /
/// Llama-3 family ships ONE combined pattern whose signature is the attached
/// contraction group PLUS the {1,3} digit rule — the exact-match digit
/// detection alone misses it because `\p{N}{1,3}` is an alternation branch,
/// not the whole regex (live 2026-08-10: "84" served per-digit, the model
/// echoed "8 4" — the DSV4 echo-precision class on a new spelling).
fn pretokStyleFromPreTokenizer(pt: std.json.Value) PretokStyle {
    if (pt != .object) return .gpt2;
    if (splitRegexIsLlama3(pt)) return .llama3;
    if (pt.object.get("pretokenizers")) |list| {
        if (list == .array) {
            for (list.array.items) |sub| {
                if (sub == .object and splitRegexIsLlama3(sub)) return .llama3;
            }
        }
    }
    return .gpt2;
}

fn splitRegexIsLlama3(node: std.json.Value) bool {
    const rx = splitRegexOf(node) orelse return false;
    return llama3StyleFromSplitRegex(rx);
}

fn llama3StyleFromSplitRegex(rx: []const u8) bool {
    return std.mem.indexOf(u8, rx, "'s|'t|'re|'ve|'m|'ll|'d") != null and
        std.mem.indexOf(u8, rx, "\\p{N}{1,3}") != null;
}

fn splitRegexOf(node: std.json.Value) ?[]const u8 {
    const obj = node.object;
    const ty = obj.get("type") orelse return null;
    if (ty != .string or !std.mem.eql(u8, ty.string, "Split")) return null;
    const pat = obj.get("pattern") orelse return null;
    if (pat != .object) return null;
    const rx = pat.object.get("Regex") orelse return null;
    if (rx != .string) return null;
    return rx.string;
}

/// The digit rule is a BRANCH, not necessarily the whole pattern: DeepSeek-V4
/// gives `\p{N}{1,3}` its own Split entry, while LFM2.5-VL / Llama-3 bury it in
/// one combined alternation. An exact-match test reads the combined spelling as
/// per-digit, which puts every number in the prompt off-distribution — and the
/// style detector cannot stand in for it, since a checkpoint can carry this
/// digit rule with a non-Llama-3 contraction group (LFM2.5-VL does).
fn splitRegexIsDigits13(node: std.json.Value) bool {
    const rx = splitRegexOf(node) orelse return false;
    return std.mem.indexOf(u8, rx, "\\p{N}{1,3}") != null;
}

/// Parse one BPE merge entry, accepting both on-disk formats:
///   - 2-element array `["left","right"]` (newer HF: Qwen3, Gemma 4)
///   - space-joined string `"left right"` (GPT-2-style: Qwen2.5, many Llama/Mistral)
/// Returns null for malformed entries (skip). The returned slices borrow from
/// `merge_val`'s backing arena, same lifetime as the array-format halves.
fn parseMergePair(merge_val: std.json.Value) ?Tokenizer.MergePair {
    switch (merge_val) {
        .array => |pair| {
            if (pair.items.len < 2) return null;
            if (pair.items[0] != .string or pair.items[1] != .string) return null;
            return .{ .left = pair.items[0].string, .right = pair.items[1].string };
        },
        .string => |s| {
            // Split on the FIRST space: byte-level BPE encodes spaces as 'Ġ',
            // so neither half contains a literal space separator.
            const sp = std.mem.indexOfScalar(u8, s, ' ') orelse return null;
            return .{ .left = s[0..sp], .right = s[sp + 1 ..] };
        },
        else => return null,
    }
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

test "encode special-token scan: earliest match, longest tie-break, adjacency" {
    // Characterization for the special-token splitter in `encode` — pins the
    // exact semantics (earliest occurrence wins; at the same position the
    // LONGEST special wins; adjacent specials produce no empty segments) so
    // the O(text) bucketed scan that replaced the per-special re-search
    // (12 s per 66 KB prompt on gemma-3's 6415-special vocab) can't drift.
    const allocator = testing.allocator;

    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("a", 1);
    try vocab.put("b", 2);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    try id_to_token.put(1, "a");
    try id_to_token.put(2, "b");

    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    try special_tokens.put("<tok>", 90);
    try special_tokens.put("<tok>x", 91);
    try special_tokens.put("<s>", 92);
    try special_tokens.put("<u>", 93);

    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();

    var tok = Tokenizer{
        .vocab = vocab,
        .id_to_token = id_to_token,
        .merge_ranks = merge_ranks,
        .allocator = allocator,
        .special_tokens = special_tokens,
        .tok_type = .sentencepiece_bpe,
        .byte_to_unicode = buildBytesToUnicode(),
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
    };
    defer tok.unicode_to_byte.deinit();

    // Longest tie-break: "<tok>x" and "<tok>" both match at position 1 —
    // the longer one must win.
    {
        const ids = try tok.encode(allocator, "a<tok>xb");
        defer allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 1, 91, 2 }, ids);
    }
    // Adjacent specials, special at start, trailing text.
    {
        const ids = try tok.encode(allocator, "<s><s>a");
        defer allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 92, 92, 1 }, ids);
    }
    // Earliest occurrence wins regardless of hashmap iteration order.
    {
        const ids = try tok.encode(allocator, "b<u>a<s>");
        defer allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 2, 93, 1, 92 }, ids);
    }
    // No specials present at all: whole text is one segment.
    {
        const ids = try tok.encode(allocator, "ab");
        defer allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 1, 2 }, ids);
    }
    // Special token at the very end, nothing after it.
    {
        const ids = try tok.encode(allocator, "a<s>");
        defer allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 1, 92 }, ids);
    }
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
    return expectPreTokensG(allocator, input, 1, expected);
}

fn expectPreTokensL3(allocator: std.mem.Allocator, input: []const u8, expected: []const []const u8) !void {
    var words: std.ArrayList([]const u8) = .empty;
    defer {
        for (words.items) |w| allocator.free(w);
        words.deinit(allocator);
    }
    try llama3PreTokenize(allocator, input, &words);
    if (words.items.len != expected.len) {
        std.debug.print("\n  llama3 pre-tokenize on {s}: got {d} words, expected {d}\n", .{
            input, words.items.len, expected.len,
        });
        for (words.items, 0..) |w, i| std.debug.print("    [{d}] '{s}'\n", .{ i, w });
        return error.WordCountMismatch;
    }
    for (words.items, expected, 0..) |got, want, idx| {
        if (!std.mem.eql(u8, got, want)) {
            std.debug.print("\n  llama3 pre-tokenize on {s}: word [{d}] got '{s}', expected '{s}'\n", .{ input, idx, got, want });
            return error.WordMismatch;
        }
    }
}

fn expectPreTokensG(allocator: std.mem.Allocator, input: []const u8, digit_group: u8, expected: []const []const u8) !void {
    var words: std.ArrayList([]const u8) = .empty;
    defer {
        for (words.items) |w| allocator.free(w);
        words.deinit(allocator);
    }
    try gpt2PreTokenize(allocator, input, digit_group, &words);
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
    // every subsequent word. Found via byte-diff against a reference tokenizer.
    try expectPreTokens(testing.allocator, "    total = 0", &.{
        "   ", " total", " =", " ", "0",
    });
}

test "gpt2PreTokenize: leading space combines with letters" {
    // Pattern 2 absorbs the optional leading non-LN char.
    try expectPreTokens(testing.allocator, " total", &.{" total"});
    try expectPreTokens(testing.allocator, "def total", &.{ "def", " total" });
}

test "llama3PreTokenize: boundaries match the reference regex (muse_glimmer class)" {
    // Expected boundaries generated with python `regex` findall of the exact
    // Llama-3-style Split pattern Muse-Glimmer-30B ships (case-transition
    // word splits, attached (?i) contractions, {1,3} digit groups, `/` in the
    // punct tail class). BPE cannot merge across pre-tokens, so these
    // boundaries drive final token ids — per-digit splitting here was the
    // DSV4 echo-precision class all over again ("8 4" echoed for "84", live
    // 2026-08-10 first boot).
    const a = testing.allocator;
    try expectPreTokensL3(a, "What is 84 * 3 / 2? In 1234 years.", &.{
        "What", " is", " ", "84", " *", " ", "3", " /", " ", "2", "?", " In", " ", "123", "4", " years", ".",
    });
    try expectPreTokensL3(a, "don't stop", &.{ "don't", " stop" });
    // (?i) contractions attach in ANY case; the pre-token stays whole.
    try expectPreTokensL3(a, "I'LL be DON'T", &.{ "I'LL", " be", " DON'T" });
    // Case-transition split (lower→upper) but ABCdef joins (upper*lower+).
    try expectPreTokensL3(a, "HelloWorld ABCdef", &.{ "Hello", "World", " ABCdef" });
    try expectPreTokensL3(a, "hello   world", &.{ "hello", "  ", " world" });
    try expectPreTokensL3(a, "x=1; a/b/ c", &.{ "x", "=", "1", ";", " a", "/b", "/", " c" });
    try expectPreTokensL3(a, " 12345.67", &.{ " ", "123", "45", ".", "67" });
    // `/` joins the punct tail (the [\r\n/]* quirk).
    try expectPreTokensL3(a, "https://a.b/c", &.{ "https", "://", "a", ".b", "/c" });
    try expectPreTokensL3(a, "foo\n\n  bar", &.{ "foo", "\n\n", " ", " bar" });
    try expectPreTokensL3(a, "It's 3.14", &.{ "It's", " ", "3", ".", "14" });
}

test "llama3 style detection: muse's combined Split regex selects it, others keep gpt2" {
    // The muse regex carries the contraction group AND the {1,3} digit rule
    // inside ONE combined Split — the exact-match digit detection alone
    // misses it (that was the live "8 4" bug).
    const muse_pattern = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    try testing.expect(llama3StyleFromSplitRegex(muse_pattern));
    // Qwen/GPT-2-style and the DSV4 standalone digit rule stay gpt2.
    try testing.expect(!llama3StyleFromSplitRegex("\\p{N}{1,3}"));
    try testing.expect(!llama3StyleFromSplitRegex("[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}"));
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

test "gpt2PreTokenize: {1,3} digit groups when the spec declares them (DSV4 class)" {
    // DeepSeek-V4's tokenizer.json isolates digit runs with `\p{N}{1,3}` —
    // greedy left-to-right groups of up to three. Our per-digit splitting fed
    // the model an alien segmentation of every number: the ROOT CAUSE of the
    // echo-precision slip class (`1o` for `10`, o-for-0, split digits) that
    // was chased through expert quantization for days. Reference HF ids for
    // "1048576": [104][857][6]; our per-digit split can never produce them.
    try expectPreTokensG(testing.allocator, "1048576", 3, &.{ "104", "857", "6" });
    try expectPreTokensG(testing.allocator, "100", 3, &.{"100"});
    try expectPreTokensG(testing.allocator, " 100 units", 3, &.{ " ", "100", " units" });
    try expectPreTokensG(testing.allocator, "26.7.12", 3, &.{ "26", ".", "7", ".", "12" });
    try expectPreTokensG(testing.allocator, "v2", 3, &.{ "v", "2" });
    // group 1 keeps the Qwen behavior byte-identical
    try expectPreTokensG(testing.allocator, "1048576", 1, &.{ "1", "0", "4", "8", "5", "7", "6" });
}

test "tokenizer.json digit-group parse: {1,3} Split rule sets digit_group 3" {
    const json_13 =
        \\{"type":"Sequence","pretokenizers":[
        \\  {"type":"Split","pattern":{"Regex":"\\p{N}{1,3}"},"behavior":"Isolated","invert":false},
        \\  {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}
    ;
    var p1 = try std.json.parseFromSlice(std.json.Value, testing.allocator, json_13, .{});
    defer p1.deinit();
    try testing.expectEqual(@as(u8, 3), digitGroupFromPreTokenizer(p1.value));

    const json_single =
        \\{"type":"ByteLevel"}
    ;
    var p2 = try std.json.parseFromSlice(std.json.Value, testing.allocator, json_single, .{});
    defer p2.deinit();
    try testing.expectEqual(@as(u8, 1), digitGroupFromPreTokenizer(p2.value));
}

test "tokenizer.json digit-group parse: {1,3} inside a COMBINED Split regex" {
    // LFM2.5-VL ships one alternation carrying the digit rule as a branch, so
    // an exact-match detector reads it as per-digit and every number in the
    // prompt goes off-distribution (live 2026-08-14: "973, 162" served as
    // '9','7','3' / '1','6','2' against HF's '973' / '162', measured on the
    // ScreenSpot-v2 grounding track). The contraction group is spelled
    // `'(?i:[sdmt]|ll|ve|re)`, not Llama-3's, so the style detector does not
    // cover it either — the digit rule is its own question.
    const json_combined =
        \\{"type":"Sequence","pretokenizers":[
        \\  {"type":"Split","pattern":{"Regex":"'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s"},"behavior":"Isolated","invert":false},
        \\  {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}
    ;
    var p3 = try std.json.parseFromSlice(std.json.Value, testing.allocator, json_combined, .{});
    defer p3.deinit();
    try testing.expectEqual(@as(u8, 3), digitGroupFromPreTokenizer(p3.value));

    // A tokenizer with no digit rule at all still groups per digit.
    const json_no_digits =
        \\{"type":"Split","pattern":{"Regex":"[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"},"behavior":"Isolated","invert":false}
    ;
    var p4 = try std.json.parseFromSlice(std.json.Value, testing.allocator, json_no_digits, .{});
    defer p4.deinit();
    try testing.expectEqual(@as(u8, 1), digitGroupFromPreTokenizer(p4.value));
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

// ── BPE merge characterization tests ──
// Pin the exact greedy semantics of `bpeMerge` (lowest rank first across the
// whole sequence; leftmost wins rank ties; unknown symbols fall back to
// byte-level pieces) so the merge algorithm can be swapped for a faster one
// without changing a single output id.

fn makeBpeTestTokenizer(
    allocator: std.mem.Allocator,
    vocab: *std.StringHashMap(u32),
    merge_ranks: *std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage),
    id_to_token: *std.AutoHashMap(u32, []const u8),
    special_tokens: *std.StringHashMap(u32),
) Tokenizer {
    return Tokenizer{
        .vocab = vocab.*,
        .id_to_token = id_to_token.*,
        .merge_ranks = merge_ranks.*,
        .allocator = allocator,
        .special_tokens = special_tokens.*,
        .tok_type = .sentencepiece_bpe,
        .byte_to_unicode = buildBytesToUnicode(),
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
    };
}

test "bpeMerge: lowest rank merges first regardless of position" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("a", 1);
    try vocab.put("b", 2);
    try vocab.put("c", 3);
    try vocab.put("d", 4);
    try vocab.put("ab", 5);
    try vocab.put("cd", 6);
    try vocab.put("abcd", 7);

    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    try merge_ranks.put(.{ .left = "c", .right = "d" }, 0);
    try merge_ranks.put(.{ .left = "a", .right = "b" }, 1);
    try merge_ranks.put(.{ .left = "ab", .right = "cd" }, 2);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();

    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);
    defer tok.unicode_to_byte.deinit();

    const ids = try tok.bpeMerge(allocator, "abcd");
    defer allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{7}, ids);
}

test "bpeMerge: leftmost pair wins rank ties" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("a", 1);
    try vocab.put("aa", 2);

    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    try merge_ranks.put(.{ .left = "a", .right = "a" }, 0);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();

    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);
    defer tok.unicode_to_byte.deinit();

    // "aaa": leftmost (a,a) merges first -> [aa, a]; (aa,a) has no rank.
    const ids = try tok.bpeMerge(allocator, "aaa");
    defer allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 2, 1 }, ids);
}

test "bpeMerge: cascading merges across merge sites" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("h", 1);
    try vocab.put("e", 2);
    try vocab.put("l", 3);
    try vocab.put("o", 4);
    try vocab.put("he", 5);
    try vocab.put("ll", 6);
    try vocab.put("hell", 7);
    try vocab.put("hello", 8);

    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    try merge_ranks.put(.{ .left = "h", .right = "e" }, 0);
    try merge_ranks.put(.{ .left = "l", .right = "l" }, 1);
    try merge_ranks.put(.{ .left = "he", .right = "ll" }, 2);
    try merge_ranks.put(.{ .left = "hell", .right = "o" }, 3);

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();

    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);
    defer tok.unicode_to_byte.deinit();

    const ids = try tok.bpeMerge(allocator, "hellohello");
    defer allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 8, 8 }, ids);
}

test "bpeMerge: symbols missing from vocab fall back to byte pieces" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("a", 1);
    // 'z' itself is NOT in vocab; its byte-to-unicode form is.
    const z_cp = buildBytesToUnicode()['z'];
    var z_buf: [4]u8 = undefined;
    const z_len = try std.unicode.utf8Encode(z_cp, &z_buf);
    try vocab.put(z_buf[0..z_len], 99);

    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();

    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();

    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);
    defer tok.unicode_to_byte.deinit();

    const ids = try tok.bpeMerge(allocator, "az");
    defer allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 99 }, ids);
}

test "parseMergePair handles both array and space-joined-string formats" {
    const alloc = testing.allocator;
    // Array format (Qwen3 / Gemma 4): ["left","right"].
    {
        var p = try std.json.parseFromSlice(std.json.Value, alloc, "[\"AB\",\"cd\"]", .{});
        defer p.deinit();
        const mp = parseMergePair(p.value).?;
        try testing.expectEqualStrings("AB", mp.left);
        try testing.expectEqualStrings("cd", mp.right);
    }
    // String format (Qwen2.5 / GPT-2-style): "left right" split on first space.
    {
        var p = try std.json.parseFromSlice(std.json.Value, alloc, "\"AB cd\"", .{});
        defer p.deinit();
        const mp = parseMergePair(p.value).?;
        try testing.expectEqualStrings("AB", mp.left);
        try testing.expectEqualStrings("cd", mp.right);
    }
    // Malformed → null (no crash): string without a space, short array.
    {
        var p = try std.json.parseFromSlice(std.json.Value, alloc, "\"nospace\"", .{});
        defer p.deinit();
        try testing.expect(parseMergePair(p.value) == null);
    }
    {
        var p = try std.json.parseFromSlice(std.json.Value, alloc, "[\"only\"]", .{});
        defer p.deinit();
        try testing.expect(parseMergePair(p.value) == null);
    }
}

test "reservedOutputIds: specials minus template markers minus eos" {
    // A reserved marker (`<|fim_hole|>` at a collapsed position) in
    // chat output is always a bug; the suppression set is every `special:
    // true` added token MINUS legitimate output. Legitimacy is DERIVED, never
    // hardcoded: EOS/stop ids stay, and so does any special whose literal
    // text appears in the chat template source (thinking tags, tool markers,
    // role markers) — a hardcoded list breaks thinking/tool-calling on the
    // next arch.
    const alloc = testing.allocator;
    const flagged = [_]FlaggedSpecial{
        .{ .id = 10, .content = "<|fim_hole|>" },
        .{ .id = 11, .content = "<|fim_begin|>" },
        .{ .id = 20, .content = "<think>" },
        .{ .id = 21, .content = "</think>" },
        .{ .id = 22, .content = "<tool_call>" },
        .{ .id = 30, .content = "<|endoftext|>" },
        .{ .id = 31, .content = "<|role_end|>" },
    };
    const template = "<role>HUMAN</role>{{ content }}<|role_end|>" ++
        "<role>ASSISTANT</role>\n<think></think>{% if tools %}<tool_call>{% endif %}";
    const eos = [_]u32{30};

    const ids = try reservedOutputIds(alloc, &flagged, template, &eos);
    defer alloc.free(ids);
    // Only the FIM markers survive: think/tool/role markers appear in the
    // template, <|endoftext|> is EOS.
    try testing.expectEqualSlices(u32, &[_]u32{ 10, 11 }, ids);

    // No template at all => suppression disabled entirely (fallback-formatted
    // models keep pre-change behavior; ChatML markers are not in any template
    // source we could consult).
    const none = try reservedOutputIds(alloc, &flagged, "", &eos);
    defer alloc.free(none);
    try testing.expectEqual(@as(usize, 0), none.len);
}

test "reservedOutputIds: every flagged special exempt yields empty set" {
    const alloc = testing.allocator;
    const flagged = [_]FlaggedSpecial{
        .{ .id = 5, .content = "<eos>" },
        .{ .id = 6, .content = "<think>" },
    };
    const ids = try reservedOutputIds(alloc, &flagged, "x<think>y", &[_]u32{5});
    defer alloc.free(ids);
    try testing.expectEqual(@as(usize, 0), ids.len);
}
