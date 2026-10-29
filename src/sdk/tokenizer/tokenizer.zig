const std = @import("std");

pub const isWhitespaceCp = @import("pretokenize.zig").isWhitespaceCp;
pub const isLetter = @import("pretokenize.zig").isLetter;
pub const isDigit = @import("pretokenize.zig").isDigit;
pub const loadTokenizer = @import("load.zig").loadTokenizer;
pub const loadTiktoken = @import("load.zig").loadTiktoken;
pub const loadTokenizerAny = @import("load.zig").loadTokenizerAny;
const isolatedRunsPreTokenize = @import("pretokenize.zig").isolatedRunsPreTokenize;
const buildBytesToUnicode = @import("pretokenize.zig").buildBytesToUnicode;
const MetaspacePrepend = @import("load.zig").MetaspacePrepend;
const byteTokenValue = @import("load.zig").byteTokenValue;
const flushByteRun = @import("load.zig").flushByteRun;

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
/// Specials a model legitimately EMITS but that no template ever RENDERS.
///
/// `reservedOutputIds` derives legitimacy from presence in the chat template,
/// which is right for every marker that appears on both sides of the
/// conversation (role markers, think tags, tool wrappers) — the model emits
/// what it was shown. An output-only marker breaks that symmetry: it exists
/// purely in generated text, so the template cannot vouch for it and the
/// derivation files it as reserved. Masking one does not merely drop it; the
/// model substitutes whatever it can still draw, and a malformed structure is
/// worse than a missing token.
///
/// Kept to markers that are unambiguous across families, since this is the one
/// place the derivation is overridden by name.
fn isOutputOnlySpecial(content: []const u8) bool {
    // harmony (gpt_oss): declares a tool call's argument content type inside
    // the header the MODEL writes — `to=functions.x <|constrain|>json`. The
    // template renders tool calls only when replaying history, and omits it.
    return std.mem.eql(u8, content, "<|constrain|>");
}

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
        if (isOutputOnlySpecial(sp.content)) continue;
        try out.append(allocator, sp.id);
    }
    return out.toOwnedSlice(allocator);
}

fn isAllWhitespace(bytes: []const u8) bool {
    const view = std.unicode.Utf8View.init(bytes) catch return false;
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| {
        if (!isWhitespaceCp(cp)) return false;
    }
    return true;
}

/// BPE tokenizer supporting both SentencePiece (Gemma) and byte-level (GPT-2/Qwen3) modes.
/// `.llama3` is Muse-Glimmer's CASED Llama-3 variant; the plain Llama-3 regex is `.gpt2`.
/// `.cased` is Flint's: the same word branches, a punctuation tail of newlines only (no `/`).
pub const PretokStyle = enum { gpt2, llama3, cased };

/// Runs that a tokenizer.json Split entry of their own isolates before the main regex runs (DeepSeek-V4,
/// Hunyuan, Spark-X2.5): `digits` = `\p{N}{1,3}`, `cjk` = their CJK class, U+4E00-9FA5 and U+3040-30FF.
pub const IsolatedRuns = struct { digits: bool = false, cjk: bool = false };

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
    isolated_runs: IsolatedRuns = .{},
    /// HF `Metaspace` pre-tokenizer (mmBERT / Gemma-2 class tokenizer.json):
    /// a leading ▁ is prepended when the text does not already start with
    /// one (`prepend_scheme`; `first` = only text at offset 0, never after a
    /// special token), and the text is split at every ▁ (kept with the
    /// following piece) before BPE (`split: true`).
    metaspace_prepend: MetaspacePrepend = .never,
    metaspace_split: bool = false,
    /// HF BPE `byte_fallback`: a character with no vocab entry becomes its
    /// UTF-8 bytes as `<0xNN>` tokens (all must exist, else `unk_token`);
    /// consecutive unks fuse when `fuse_unk`. Gemma / mmBERT class.
    byte_fallback: bool = false,
    fuse_unk: bool = false,
    /// HF BPE `ignore_merges`: a word the vocab holds whole is that one token, before any merge.
    ignore_merges: bool = false,
    /// HF WordPiece `max_input_chars_per_word`: a longer word is one `[UNK]` (the greedy match is cubic in its length).
    wordpiece_max_chars: usize = 100,
    unk_id: ?u32 = null,
    /// tokenizer.json decoder carries `ByteFallback`: runs of `<0xNN>`
    /// tokens decode as the bytes they spell (U+FFFD per byte when the run
    /// is not valid UTF-8).
    byte_fallback_decode: bool = false,
    /// The special a `TemplateProcessing` post-processor appends after a single sequence
    /// (Qwen3-Embedding's `<|endoftext|>`); `encode` never adds it, a last-token pooler must.
    appended_special: ?u32 = null,
    /// Decode-only marker aliases (K2-Horizon): the `<ifm|…>` think and tool
    /// markers decode as the canonical `<think>` / GLM tag spellings every
    /// downstream parser reads. Encoding keeps the checkpoint's own bytes.
    marker_aliases: ?std.AutoHashMap(u32, []const u8) = null,
    /// K2 think opener id -> its own closer id (three pairs, one per effort).
    marker_closers: ?std.AutoHashMap(u32, u32) = null,
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

    /// `special_tokens` bucketed for `encode`, built once by the loader. Null
    /// (a tokenizer assembled by hand): `encode` builds it per call.
    special_index: ?SpecialIndex = null,

    /// Special tokens sorted by first byte, longest first within a byte;
    /// `start[b]..start[b + 1]` are the candidates starting with byte `b`.
    pub const SpecialIndex = struct {
        cands: []Cand,
        start: [257]u32,

        pub const Cand = struct { bytes: []const u8, id: u32 };

        pub fn init(allocator: std.mem.Allocator, specials: *const std.StringHashMap(u32)) !SpecialIndex {
            const cands = try allocator.alloc(Cand, specials.count());
            var i: usize = 0;
            var sit = specials.iterator();
            while (sit.next()) |entry| : (i += 1) {
                cands[i] = .{ .bytes = entry.key_ptr.*, .id = entry.value_ptr.* };
            }
            std.mem.sort(Cand, cands, {}, struct {
                fn lessThan(_: void, a: Cand, b: Cand) bool {
                    const ab: u8 = if (a.bytes.len > 0) a.bytes[0] else 0;
                    const bb: u8 = if (b.bytes.len > 0) b.bytes[0] else 0;
                    if (ab != bb) return ab < bb;
                    return a.bytes.len > b.bytes.len;
                }
            }.lessThan);
            var start: [257]u32 = @splat(0);
            var ci: usize = 0;
            for (0..256) |b| {
                start[b] = @intCast(ci);
                while (ci < cands.len and cands[ci].bytes.len > 0 and cands[ci].bytes[0] == b) ci += 1;
            }
            start[256] = @intCast(cands.len);
            return .{ .cands = cands, .start = start };
        }

        pub fn deinit(self: *SpecialIndex, allocator: std.mem.Allocator) void {
            allocator.free(self.cands);
        }

        /// The specials starting with byte `b`, longest first.
        pub fn candidates(self: *const SpecialIndex, b: u8) []const Cand {
            return self.cands[self.start[b]..self.start[@as(usize, b) + 1]];
        }
    };

    pub const MergePair = struct {
        left: []const u8,
        right: []const u8,
    };

    pub const MergePairContext = struct {
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
        if (self.special_index) |*x| x.deinit(self.allocator);
        if (self.marker_aliases) |*m| m.deinit();
        if (self.marker_closers) |*m| m.deinit();
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
        return self.encodeFrom(allocator, text, true);
    }

    /// `encode`; `at_start` false for a piece in the middle of a text (SentencePiece adds no start prefix).
    fn encodeFrom(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, at_start: bool) ![]u32 {
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

        // Special-token splitter: specials bucketed by first byte (`SpecialIndex`, built once at load), one
        // left-to-right pass; earliest occurrence wins, longest special at the same position.
        var local_index: ?SpecialIndex = null;
        defer if (local_index) |*x| x.deinit(allocator);
        const index = try self.specialIndex(allocator, &local_index);

        var pos: usize = 0;
        var seg_start: usize = 0;
        while (pos < text.len) {
            const matched: ?SpecialIndex.Cand = for (index.candidates(text[pos])) |c| {
                if (std.mem.startsWith(u8, text[pos..], c.bytes)) break c;
            } else null;
            if (matched) |m| {
                if (pos > seg_start) {
                    const ids = try self.encodeSegment(allocator, text[seg_start..pos], at_start and seg_start == 0);
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
            const ids = try self.encodeSegment(allocator, text[seg_start..], at_start and seg_start == 0);
            defer allocator.free(ids);
            try result.appendSlice(allocator, ids);
        }

        return result.toOwnedSlice(allocator);
    }

    fn specialIndex(self: *const Tokenizer, allocator: std.mem.Allocator, local: *?SpecialIndex) !*const SpecialIndex {
        if (self.special_index) |*x| return x;
        local.* = try SpecialIndex.init(allocator, &self.special_tokens);
        return &local.*.?;
    }

    /// Plane-15 private-use code points `markControlTokens` wraps around a control token in message content:
    /// transformers' `mark_control_tokens` markers (`apply_chat_template(sanitize_control_tokens=True)`).
    pub const CONTROL_START = "\u{F0000}";
    pub const CONTROL_END = "\u{F0001}";

    /// A control token: flagged `special: true`, or an added token the chat template writes itself, unless it
    /// is all whitespace (Gemma's "\n" runs are ordinary text). Transformers' `_get_control_tokens` default.
    fn isControlToken(self: *const Tokenizer, token: SpecialIndex.Cand, template: []const u8) bool {
        for (self.flagged_specials) |f| {
            if (f.id == token.id) return true;
        }
        return !isAllWhitespace(token.bytes) and std.mem.indexOf(u8, template, token.bytes) != null;
    }

    /// `text` with every control token wrapped in CONTROL_START/CONTROL_END for `encodeMarked`, and the markers
    /// already in it dropped so content cannot forge a pair; null when that changes nothing.
    pub fn markControlTokens(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, template: []const u8) !?[]u8 {
        var local_index: ?SpecialIndex = null;
        defer if (local_index) |*x| x.deinit(allocator);
        const index = try self.specialIndex(allocator, &local_index);
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(allocator);
        var copied: usize = 0;
        var pos: usize = 0;
        scan: while (pos < text.len) {
            if (std.mem.startsWith(u8, text[pos..], CONTROL_START) or std.mem.startsWith(u8, text[pos..], CONTROL_END)) {
                try out.appendSlice(allocator, text[copied..pos]);
                pos += CONTROL_START.len;
                copied = pos;
                continue;
            }
            for (index.candidates(text[pos])) |c| {
                if (!std.mem.startsWith(u8, text[pos..], c.bytes) or !self.isControlToken(c, template)) continue;
                try out.appendSlice(allocator, text[copied..pos]);
                try out.appendSlice(allocator, CONTROL_START);
                try out.appendSlice(allocator, c.bytes);
                try out.appendSlice(allocator, CONTROL_END);
                pos += c.bytes.len;
                copied = pos;
                continue :scan;
            }
            pos += 1;
        }
        if (copied == 0) return null;
        try out.appendSlice(allocator, text[copied..]);
        return try out.toOwnedSlice(allocator);
    }

    /// `encode` of a chat rendered from `markControlTokens` content (transformers' `encode_marked_chat`): a
    /// marked token encodes as plain text, piece by piece; a lone marker (a template string op split the pair)
    /// is dropped but still ends the piece, so text on either side cannot join into a special.
    pub fn encodeMarked(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        const prefix = CONTROL_START[0..3]; // both markers share it; the 4th byte tells them apart
        var local_index: ?SpecialIndex = null;
        defer if (local_index) |*x| x.deinit(allocator);
        const index = try self.specialIndex(allocator, &local_index);
        var out = std.ArrayList(u32).empty;
        errdefer out.deinit(allocator);
        var piece: usize = 0;
        var pos: usize = 0;
        while (std.mem.indexOfPos(u8, text, pos, prefix)) |at| {
            pos = at + 1;
            if (at + CONTROL_START.len > text.len or (text[at + 3] != CONTROL_START[3] and text[at + 3] != CONTROL_END[3])) continue;
            try self.appendEncoded(allocator, &out, text[piece..at], piece == 0, false);
            pos = at + CONTROL_START.len;
            piece = pos;
            if (text[at + 3] != CONTROL_START[3] or pos == text.len) continue;
            const len = for (index.candidates(text[pos])) |c| {
                if (std.mem.startsWith(u8, text[pos..], c.bytes) and std.mem.startsWith(u8, text[pos + c.bytes.len ..], CONTROL_END)) break c.bytes.len;
            } else continue;
            try self.appendEncoded(allocator, &out, text[pos..][0..len], false, true);
            pos += len + CONTROL_END.len;
            piece = pos;
        }
        try self.appendEncoded(allocator, &out, text[piece..], piece == 0, false);
        return out.toOwnedSlice(allocator);
    }

    fn appendEncoded(self: *const Tokenizer, allocator: std.mem.Allocator, out: *std.ArrayList(u32), text: []const u8, at_start: bool, as_text: bool) !void {
        if (text.len == 0) return;
        const ids = if (as_text) try self.encodeSegment(allocator, text, at_start) else try self.encodeFrom(allocator, text, at_start);
        defer allocator.free(ids);
        try out.appendSlice(allocator, ids);
    }

    /// Encode a text segment (no special tokens) using the appropriate method.
    fn encodeSegment(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, at_start: bool) ![]u32 {
        return switch (self.tok_type) {
            .sentencepiece_bpe => self.encodeSentencePiece(allocator, text, at_start),
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

    /// K2-Horizon spells the think block and the GLM tool tags with an `ifm|`
    /// prefix, one special token each; three think variants (one per effort)
    /// all read as `<think>`. Installed when the vocabulary carries the family
    /// marker; false otherwise.
    pub fn installMarkerAliases(self: *Tokenizer) bool {
        const family_marker = "<ifm|think>";
        if (self.special_tokens.get(family_marker) == null) return false;
        var map = std.AutoHashMap(u32, []const u8).init(self.allocator);
        errdefer map.deinit();
        const pairs = [_][2][]const u8{
            .{ "<ifm|think>", "<think>" },              .{ "</ifm|think>", "</think>" },
            .{ "<ifm|think_fast>", "<think>" },         .{ "</ifm|think_fast>", "</think>" },
            .{ "<ifm|think_faster>", "<think>" },       .{ "</ifm|think_faster>", "</think>" },
            .{ "<ifm|tool_calls>", "<tool_calls>" },    .{ "</ifm|tool_calls>", "</tool_calls>" },
            .{ "<ifm|tool_call>", "<tool_call>" },      .{ "</ifm|tool_call>", "</tool_call>" },
            .{ "<ifm|arg_key>", "<arg_key>" },          .{ "</ifm|arg_key>", "</arg_key>" },
            .{ "<ifm|arg_value>", "<arg_value>" },      .{ "</ifm|arg_value>", "</arg_value>" },
        };
        for (pairs) |pair| {
            if (self.special_tokens.get(pair[0])) |id| map.put(id, pair[1]) catch return false;
        }
        var closers = std.AutoHashMap(u32, u32).init(self.allocator);
        errdefer closers.deinit();
        for ([_][2][]const u8{
            .{ "<ifm|think>", "</ifm|think>" },
            .{ "<ifm|think_fast>", "</ifm|think_fast>" },
            .{ "<ifm|think_faster>", "</ifm|think_faster>" },
        }) |pair| {
            const open = self.special_tokens.get(pair[0]) orelse continue;
            const close = self.special_tokens.get(pair[1]) orelse continue;
            closers.put(open, close) catch return false;
        }
        self.marker_aliases = map;
        self.marker_closers = closers;
        return true;
    }

    /// The atomic closer paired with a K2 think opener id; null for every other id.
    pub fn markerCloserFor(self: *const Tokenizer, opener_id: u32) ?u32 {
        const m = self.marker_closers orelse return null;
        return m.get(opener_id);
    }

    /// Highest DEFINED id + 1 — the real vocabulary, which is not the config's
    /// `vocab_size`: checkpoints pad the embedding/lm_head rows out to a
    /// friendly multiple (qwen4_exp: 248044 base + 33 added = 248077 defined
    /// against a declared 248320, so 243 rows decode to nothing at all). Those
    /// rows carry whatever the initializer left and a collapsed distribution
    /// can rank one top-1; `installSuppressMask` masks them out of sampling.
    /// 0 when nothing is defined (an `initEmptyForTests` tokenizer) — callers
    /// treat that as "no padding known", never as "suppress everything".
    pub fn definedVocabSize(self: *const Tokenizer) usize {
        var highest: ?u32 = null;
        var it = self.id_to_token.keyIterator();
        while (it.next()) |id| {
            if (highest == null or id.* > highest.?) highest = id.*;
        }
        return if (highest) |h| @as(usize, h) + 1 else 0;
    }

    // ── SentencePiece BPE (Gemma-style) ──

    pub fn encodeSentencePiece(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, at_start: bool) ![]u32 {
        // Normalize: replace spaces with ▁ (U+2581)
        var normalized: std.ArrayList(u8) = .empty;
        defer normalized.deinit(allocator);

        const sep = "\xe2\x96\x81";
        const prepend = switch (self.metaspace_prepend) {
            .never => false,
            .first => at_start,
            .always => true,
        };
        if (prepend and !std.mem.startsWith(u8, text, " ") and !std.mem.startsWith(u8, text, sep)) {
            try normalized.appendSlice(allocator, sep);
        }
        for (text) |c| {
            if (c == ' ') {
                try normalized.appendSlice(allocator, sep);
            } else {
                try normalized.append(allocator, c);
            }
        }

        if (!self.metaspace_split) return self.bpeMerge(allocator, normalized.items);

        // Metaspace `split: true` = SplitDelimiterBehavior::MergedWithNext on
        // ▁: every piece starts at a ▁ (a run of ▁ yields lone "▁" pieces) and
        // BPE never merges across pieces.
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(allocator);
        const n = normalized.items;
        var start: usize = 0;
        var i: usize = if (std.mem.startsWith(u8, n, sep)) sep.len else 0;
        while (i <= n.len) {
            const at_sep = i + sep.len <= n.len and std.mem.eql(u8, n[i .. i + sep.len], sep);
            if (i == n.len or at_sep) {
                if (i > start) {
                    const piece = try self.bpeMerge(allocator, n[start..i]);
                    defer allocator.free(piece);
                    try ids.appendSlice(allocator, piece);
                }
                start = i;
                if (i == n.len) break;
                i += sep.len;
            } else {
                i += 1;
            }
        }
        return ids.toOwnedSlice(allocator);
    }

    fn decodeSentencePiece(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32, strip_leading_space: bool) ![]u8 {
        var result: std.ArrayList(u8) = .empty;
        defer result.deinit(allocator);

        // `ByteFallback` decoder: a run of `<0xNN>` tokens is one byte string,
        // appended verbatim when it is valid UTF-8, else U+FFFD per byte.
        var pending: std.ArrayList(u8) = .empty;
        defer pending.deinit(allocator);
        for (ids) |id| {
            const token = self.id_to_token.get(id) orelse continue;
            if (self.byte_fallback_decode) {
                if (byteTokenValue(token)) |b| {
                    try pending.append(allocator, b);
                    continue;
                }
                try flushByteRun(allocator, &result, &pending);
            }
            try result.appendSlice(allocator, token);
        }
        try flushByteRun(allocator, &result, &pending);

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
        try isolatedRunsPreTokenize(allocator, text, self.pretok_style, self.digit_group, self.isolated_runs, &words);

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
            if (self.marker_aliases) |aliases| {
                if (aliases.get(id)) |alias| {
                    try token_str.appendSlice(allocator, alias);
                    continue;
                }
            }
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
            if ((std.unicode.utf8CountCodepoints(word) catch word.len) > self.wordpiece_max_chars) {
                try result.append(allocator, unk_id);
                continue;
            }
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

    pub fn isPunct(c: u8) bool {
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
    pub fn bpeMerge(self: *const Tokenizer, allocator: std.mem.Allocator, input: []const u8) ![]u32 {
        if (self.ignore_merges) if (self.vocab.get(input)) |id| {
            const out = try allocator.alloc(u32, 1);
            out[0] = id;
            return out;
        };
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
            } else if (self.byte_fallback) {
                try self.appendByteFallback(allocator, &ids, sym);
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

    /// `BPE::merge_word` with `byte_fallback`: `sym` (one character HF found
    /// no vocab entry for) becomes one `<0xNN>` id per byte when every byte
    /// token exists, else the unk id — fused with a directly preceding unk
    /// when `fuse_unk`.
    fn appendByteFallback(self: *const Tokenizer, allocator: std.mem.Allocator, ids: *std.ArrayList(u32), sym: []const u8) !void {
        var byte_ids: [4]u32 = undefined;
        var all = sym.len <= byte_ids.len;
        if (all) for (sym, 0..) |byte, i| {
            var name: [6]u8 = undefined;
            const key = std.fmt.bufPrint(&name, "<0x{X:0>2}>", .{byte}) catch unreachable;
            byte_ids[i] = self.vocab.get(key) orelse {
                all = false;
                break;
            };
        };
        if (all) {
            try ids.appendSlice(allocator, byte_ids[0..sym.len]);
            return;
        }
        const unk = self.unk_id orelse return;
        if (self.fuse_unk and ids.items.len > 0 and ids.items[ids.items.len - 1] == unk) return;
        try ids.append(allocator, unk);
    }
};

test {
    _ = @import("load.zig");
    _ = @import("pretokenize.zig");
    _ = @import("tokenizer_test.zig");
}
