//! Tests of the tokenizer: encode/decode, pre-tokenizers, loading.

const std = @import("std");
const env = @import("../../env.zig");
const testing = std.testing;

const FlaggedSpecial = @import("tokenizer.zig").FlaggedSpecial;
const reservedOutputIds = @import("tokenizer.zig").reservedOutputIds;
const PretokStyle = @import("tokenizer.zig").PretokStyle;
const IsolatedRuns = @import("tokenizer.zig").IsolatedRuns;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const isolatedRunsPreTokenize = @import("pretokenize.zig").isolatedRunsPreTokenize;
const gpt2PreTokenize = @import("pretokenize.zig").gpt2PreTokenize;
const llama3PreTokenize = @import("pretokenize.zig").llama3PreTokenize;
const buildBytesToUnicode = @import("pretokenize.zig").buildBytesToUnicode;
const loadTokenizer = @import("load.zig").loadTokenizer;
const loadTokenizerSlow = @import("load.zig").loadTokenizerSlow;
const tiktokenToTokenizerJson = @import("load.zig").tiktokenToTokenizerJson;
const parseTokenizerContent = @import("load.zig").parseTokenizerContent;
const digitGroupFromPreTokenizer = @import("load.zig").digitGroupFromPreTokenizer;
const pretokStyleFromPreTokenizer = @import("load.zig").pretokStyleFromPreTokenizer;
const isolatedRunsFromPreTokenizer = @import("load.zig").isolatedRunsFromPreTokenizer;
const llama3StyleFromSplitRegex = @import("load.zig").llama3StyleFromSplitRegex;
const parseMergePair = @import("load.zig").parseMergePair;
const MetaspaceOpts = @import("load.zig").MetaspaceOpts;
const byteTokenValue = @import("load.zig").byteTokenValue;
const decoderHasByteFallback = @import("load.zig").decoderHasByteFallback;
const appendedSpecialId = @import("load.zig").appendedSpecialId;
const metaspaceFromPreTokenizer = @import("load.zig").metaspaceFromPreTokenizer;

// ── tests ──

test "metaspaceFromPreTokenizer: Metaspace sets prepend+split, Split/ByteLevel do not" {
    var p1 = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always","split":true}
    , .{});
    defer p1.deinit();
    try testing.expectEqual(MetaspaceOpts{ .prepend = .always, .split = true }, metaspaceFromPreTokenizer(p1.value));
    var p2 = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false}
    , .{});
    defer p2.deinit();
    try testing.expectEqual(MetaspaceOpts{}, metaspaceFromPreTokenizer(p2.value));
    var p3 = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"never","split":false}
    , .{});
    defer p3.deinit();
    try testing.expectEqual(MetaspaceOpts{}, metaspaceFromPreTokenizer(p3.value));
}

test "encodeSentencePiece: Metaspace prepends ▁ once and splits pieces at ▁ before BPE" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("\xe2\x96\x81a", 1);
    try vocab.put("\xe2\x96\x81", 2);
    try vocab.put("\xe2\x96\x81b", 3);
    try vocab.put("a", 4);
    try vocab.put("b", 5);
    try vocab.put("\xe2\x96\x81a\xe2\x96\x81b", 6);
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    try merge_ranks.put(.{ .left = "\xe2\x96\x81", .right = "a" }, 0);
    try merge_ranks.put(.{ .left = "\xe2\x96\x81", .right = "b" }, 1);
    try merge_ranks.put(.{ .left = "\xe2\x96\x81a", .right = "\xe2\x96\x81b" }, 2);
    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    try special_tokens.put("[I]", 9);
    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);
    defer tok.unicode_to_byte.deinit();

    // Legacy path (no Metaspace): no prefix.
    const legacy = try tok.encodeSentencePiece(allocator, "a b", true);
    defer allocator.free(legacy);
    try testing.expectEqualSlices(u32, &[_]u32{ 4, 3 }, legacy);

    // Prefix without split: BPE merges straight across the ▁ boundary.
    tok.metaspace_prepend = .always;
    const unsplit = try tok.encodeSentencePiece(allocator, "a b", true);
    defer allocator.free(unsplit);
    try testing.expectEqualSlices(u32, &[_]u32{6}, unsplit);

    tok.metaspace_split = true;
    const split = try tok.encodeSentencePiece(allocator, "a b", true);
    defer allocator.free(split);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 3 }, split);
    // A leading space already IS the prefix; a double space yields a lone ▁ (HF: 'a  b' -> ▁a ▁ ▁b).
    const lead = try tok.encodeSentencePiece(allocator, " a  b", true);
    defer allocator.free(lead);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3 }, lead);

    // prepend_scheme "first" (Mistral v0.3): only text at offset 0 gets the ▁; HF `[INST]Use` -> `Use`.
    var first = try std.json.parseFromSlice(std.json.Value, allocator,
        \\{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"first","split":false}
    , .{});
    defer first.deinit();
    const opts = metaspaceFromPreTokenizer(first.value);
    tok.metaspace_prepend = opts.prepend;
    tok.metaspace_split = opts.split;
    const after_special = try tok.encode(allocator, "a[I]a");
    defer allocator.free(after_special);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 9, 4 }, after_special);
}

pub fn hasByteLevel(pt: std.json.Value) bool {
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

test "WordPiece: a word past max_input_chars_per_word is one [UNK] (HF), never a greedy split" {
    const allocator = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(allocator, .wordpiece);
    defer tok.deinit();
    for ([_][]const u8{ "[UNK]", "x", "##x" }, 0..) |t, i| try tok.vocab.put(try allocator.dupe(u8, t), @intCast(i));

    const x100: [100]u8 = @splat('x');
    const at_cap = try tok.encode(allocator, &x100);
    defer allocator.free(at_cap);
    try testing.expectEqual(@as(usize, 100), at_cap.len);

    const x101: [101]u8 = @splat('x');
    const past_cap = try tok.encode(allocator, "y " ++ x101);
    defer allocator.free(past_cap);
    try testing.expectEqualSlices(u32, &.{ 0, 0 }, past_cap);
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
    try llama3PreTokenize(allocator, input, false, &words);
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
    // first boot).
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

test "Flint's pre-tokenizer: the cased grammar over isolated digit and CJK runs, as HF splits it" {
    // RecursiveSI/flint's tokenizer.json: digits and CJK isolated first, then the cased regex
    // (contractions attached, no `/` in the punctuation tail). Want = HF `pre_tokenize_str`.
    const pre =
        \\{"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": false}, {"type": "Split", "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"}, "behavior": "Isolated", "invert": false}, {"type": "Split", "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, pre, .{});
    defer parsed.deinit();
    const style = pretokStyleFromPreTokenizer(parsed.value);
    const runs = isolatedRunsFromPreTokenizer(parsed.value);
    try testing.expectEqual(PretokStyle.cased, style);
    try testing.expect(runs.digits and runs.cjk);
    var words: std.ArrayList([]const u8) = .empty;
    defer {
        for (words.items) |w| testing.allocator.free(w);
        words.deinit(testing.allocator);
    }
    try isolatedRunsPreTokenize(testing.allocator, "Python's server can't 12345 你好世界 x.indexOf(UserDefaults)\n//! DON'T", style, digitGroupFromPreTokenizer(parsed.value), runs, &words);
    const want = [_][]const u8{ "Python's", " server", " can't", " ", "123", "45", " ", "你好世界", " x", ".index", "Of", "(User", "Defaults", ")\n", "//!", " DON'T" };
    try testing.expectEqual(want.len, words.items.len);
    for (want, words.items) |w, got| try testing.expectEqualStrings(w, got);
    // A format character (U+200B) is neither punctuation nor symbol: no branch takes it, and HF
    // keeps the unmatched character whole as its own piece (fixture fuzz_049).
    for (words.items) |w| testing.allocator.free(w);
    words.clearRetainingCapacity();
    try isolatedRunsPreTokenize(testing.allocator, "  \u{2028}\u{2003}\u{200b}(don't:}?]Z\u{2028}\u{1680}:\u{2003}", style, 3, runs, &words);
    const fuzz = [_][]const u8{ "  \u{2028}", "\u{2003}", "\u{200b}", "(don't", ":}?]", "Z", "\u{2028}", "\u{1680}", ":", "\u{2003}" };
    try testing.expectEqual(fuzz.len, words.items.len);
    for (fuzz, words.items) |w, got| try testing.expectEqualStrings(w, got);
    // The unmatched span ends where any branch matches again, never at the next non-`\p{C}` character.
    const gaps = [_]struct { in: []const u8, want: []const []const u8 }{
        .{ .in = "\u{200b}\tfoo", .want = &.{ "\u{200b}", "\tfoo" } },
        .{ .in = "\u{200b}\u{200b}Hello", .want = &.{ "\u{200b}", "\u{200b}Hello" } },
        .{ .in = "\u{feff}\nimport", .want = &.{ "\u{feff}", "\n", "import" } },
        .{ .in = "a\u{200b}\u{200b}", .want = &.{ "a", "\u{200b}\u{200b}" } },
    };
    for (gaps) |g| {
        for (words.items) |w| testing.allocator.free(w);
        words.clearRetainingCapacity();
        try isolatedRunsPreTokenize(testing.allocator, g.in, style, 3, runs, &words);
        try testing.expectEqual(g.want.len, words.items.len);
        for (g.want, words.items) |w, got| try testing.expectEqualStrings(w, got);
    }
}

test "the plain Llama-3 regex is the gpt2 grammar with 3-digit groups, never the cased one" {
    // GLM-5.3, Llama-3.2, LFM2.5 and K2 ship this pattern: `\p{L}+` keeps camelCase whole
    // and `//!\n` one pre-token, where the cased grammar split both against the reference.
    const pre =
        \\{"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, pre, .{});
    defer parsed.deinit();
    try testing.expectEqual(PretokStyle.gpt2, pretokStyleFromPreTokenizer(parsed.value));
    try testing.expectEqual(@as(u8, 3), digitGroupFromPreTokenizer(parsed.value));
    try testing.expectEqual(IsolatedRuns{}, isolatedRunsFromPreTokenizer(parsed.value));
    try expectPreTokensG(testing.allocator, "x.indexOf(UserDefaults)\n//!\n//! 12345", 3, &.{
        "x", ".indexOf", "(UserDefaults", ")\n", "//!\n", "//!", " ", "123", "45",
    });
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

test "tokenizer.json digit-group parse: a trailing Digits(individual) rule overrides {1,3} back to 1" {
    // Spark-X2.5: DeepSeek's {1,3} Split followed by a `Digits` pretokenizer
    // with individual_digits — the later rule re-splits every group.
    const json =
        \\{"type":"Sequence","pretokenizers":[
        \\  {"type":"Split","pattern":{"Regex":"\\p{N}{1,3}"},"behavior":"Isolated","invert":false},
        \\  {"type":"Digits","individual_digits":true},
        \\  {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(u8, 1), digitGroupFromPreTokenizer(parsed.value));
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
    // prompt goes off-distribution (live: "973, 162" served as
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

test "pre-tokenize: \\s*[\\r\\n]+ ends at the whitespace run's last newline (gpt2 + llama3)" {
    // Bar: the pre-tokens HF `tokenizers` gives for the Qwen3.8, Llama-3 and Muse split regexes.
    const a = testing.allocator;
    try expectPreTokens(a, "a\n    \nb", &.{ "a", "\n    \n", "b" });
    try expectPreTokens(a, "a\n  \n  b", &.{ "a", "\n  \n", " ", " b" });
    try expectPreTokens(a, "a  \n\n  \n b", &.{ "a", "  \n\n  \n", " b" });
    try expectPreTokens(a, "a\r\n   \r\nb", &.{ "a", "\r\n   \r\n", "b" });
    try expectPreTokens(a, "def f():\n    pass\n    \n    return 1", &.{
        "def", " f", "():\n", "   ", " pass", "\n    \n", "   ", " return", " ", "1",
    });
    try expectPreTokensL3(a, "x\n\t\n\ny", &.{ "x", "\n\t\n\n", "y" });
}

test "pre-tokenize: \\s is Unicode White_Space, zero-width characters are not (gpt2 + llama3)" {
    // Bar: the pre-tokens HF `tokenizers` gives for the Qwen3.8, Llama-3 and Muse split regexes.
    const a = testing.allocator;
    try expectPreTokens(a, "你好\u{3000}世界\u{3000}\u{3000}！", &.{ "你好", "\u{3000}世界", "\u{3000}", "\u{3000}", "！" });
    try expectPreTokens(a, "x\u{3000} \u{a0}\ny", &.{ "x", "\u{3000} \u{a0}\n", "y" });
    try expectPreTokens(a, "x \u{3000}\u{a0}y", &.{ "x", " \u{3000}", "\u{a0}y" });
    try expectPreTokens(a, "a\u{200b}b\u{feff}c", &.{ "a", "\u{200b}b", "\u{feff}c" });
    try expectPreTokensL3(a, "\u{300c}\u{3000}\u{3000}\u{300d}", &.{ "\u{300c}", "\u{3000}", "\u{3000}", "\u{300d}" });
}

test "pre-tokenize: DeepSeek's digit and CJK Splits run first, so a whitespace run before them stays whole" {
    // Bar: the pre-tokens HF `tokenizers` gives for DeepSeek-V4's pre_tokenizer (Hy-MT2 ships the same one).
    const pre =
        \\{"type":"Sequence","pretokenizers":[
        \\  {"type":"Split","pattern":{"Regex":"\\p{N}{1,3}"},"behavior":"Isolated","invert":false},
        \\  {"type":"Split","pattern":{"Regex":"[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+"},"behavior":"Isolated","invert":false},
        \\  {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}
    ;
    const a = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, pre, .{});
    defer parsed.deinit();
    const runs = isolatedRunsFromPreTokenizer(parsed.value);
    try testing.expectEqual(IsolatedRuns{ .digits = true, .cjk = true }, runs);
    const cases = [_]struct { []const u8, []const []const u8 }{
        .{ "\u{3000}\u{3000}长江发源于青藏高原。", &.{ "\u{3000}\u{3000}", "长江发源于青藏高原", "。" } },
        .{ "[\n    1,\n    22\n]", &.{ "[\n", "    ", "1", ",\n", "    ", "22", "\n", "]" } },
        .{ "x \u{2003}1", &.{ "x", " \u{2003}", "1" } },
        .{ "abc中文def", &.{ "abc", "中文", "def" } },
    };
    for (cases) |c| {
        var words: std.ArrayList([]const u8) = .empty;
        defer {
            for (words.items) |w| a.free(w);
            words.deinit(a);
        }
        try isolatedRunsPreTokenize(a, c[0], .gpt2, 3, runs, &words);
        try testing.expectEqual(c[1].len, words.items.len);
        for (c[1], words.items) |want, got| try testing.expectEqualStrings(want, got);
    }
}

test "tokenizer parity: encode() matches HF `tokenizers` ids (TOKENIZER_PARITY_MODEL)" {
    // Bar: encode() returns HF `tokenizers`' ids for every case tests/dump_tokenizer_parity_fixtures.py wrote.
    const model = env.get(.TOKENIZER_PARITY_MODEL) orelse return error.SkipZigTest;
    const cases_path = env.get(.TOKENIZER_PARITY_CASES) orelse return error.SkipZigTest;
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tok = try loadTokenizer(io, allocator, std.mem.span(model));
    defer tok.deinit();

    const f = try std.Io.Dir.openFileAbsolute(io, std.mem.span(cases_path), .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const text = try rs.interface.allocRemaining(allocator, .limited(16 << 20));
    defer allocator.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();

    var mismatches: usize = 0;
    for (parsed.value.array.items) |c| {
        const input = c.object.get("text").?.string;
        const want = c.object.get("ids").?.array.items;
        const ids = try tok.encode(allocator, input);
        defer allocator.free(ids);
        var same = ids.len == want.len;
        if (same) for (ids, want) |g, w| {
            if (g != w.integer) same = false;
        };
        if (same) continue;
        mismatches += 1;
        if (mismatches > 10) continue;
        std.debug.print("\n[parity] {s} {f}\n  got  {any}\n  want", .{ c.object.get("name").?.string, std.json.fmt(input, .{ .escape_unicode = true }), ids });
        for (want) |w| std.debug.print(" {d}", .{w.integer});
        std.debug.print("\n", .{});
    }
    const n = parsed.value.array.items.len;
    std.debug.print("\n[parity] {d}/{d} cases match HF tokenizers\n", .{ n - mismatches, n });
    try testing.expectEqual(@as(usize, 0), mismatches);
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

test "gpt2PreTokenize: every \\p{N} digit groups like an ASCII one (superscripts, fractions, Indic)" {
    // Reference: HF `tokenizers` on the plain Llama-3 regex; `²` sat outside `\p{N}{1,3}`.
    try expectPreTokensG(testing.allocator, "x4\u{b2} \u{bd} \u{663}\u{664}\u{665} \u{967}\u{968} \u{2460}", 3, &.{
        "x", "4\u{b2}", " ", "\u{bd}", " ", "\u{663}\u{664}\u{665}", " ", "\u{967}\u{968}", " ", "\u{2460}",
    });
}

test "ignore_merges: a word the vocab holds whole is one token, whatever the merges reach" {
    // HF BPE `ignore_merges` (Llama-3 / LFM2.5 / GLM class): LFM2.5's merges never build
    // `_tokens`, its vocab does, and the reference emits the vocab entry.
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    for ([_][]const u8{ "_", "t", "o", "k", "e", "n", "s", "_t", "ok", "ens", "_tokens" }, 0..) |w, i| try vocab.put(w, @intCast(i));
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    try merge_ranks.put(.{ .left = "_", .right = "t" }, 0);
    try merge_ranks.put(.{ .left = "o", .right = "k" }, 1);
    try merge_ranks.put(.{ .left = "e", .right = "n" }, 2);
    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);
    defer tok.unicode_to_byte.deinit();

    tok.ignore_merges = true;
    const whole = try tok.bpeMerge(allocator, "_tokens");
    defer allocator.free(whole);
    try testing.expectEqualSlices(u32, &[_]u32{10}, whole);

    tok.ignore_merges = false;
    const merged = try tok.bpeMerge(allocator, "_tokens");
    defer allocator.free(merged);
    try testing.expect(merged.len > 1);
}

test "byte_fallback: an unknown character becomes its UTF-8 bytes as <0xNN> tokens; missing byte token -> fused unk" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    try vocab.put("a", 1);
    try vocab.put("<unk>", 3);
    try vocab.put("<0xF0>", 10);
    try vocab.put("<0xA0>", 11);
    try vocab.put("<0x80>", 12);
    try vocab.put("<0x8B>", 13);
    try vocab.put("<0xE1>", 14);
    try vocab.put("<0xA8>", 15);
    // 'z' is NOT in vocab and neither is <0x7A>: unk.
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);
    defer tok.unicode_to_byte.deinit();
    tok.byte_fallback = true;
    tok.fuse_unk = true;
    tok.unk_id = 3;

    // U+2000B (CJK ext-B) = F0 A0 80 8B.
    const cjk = try tok.bpeMerge(allocator, "a\xf0\xa0\x80\x8ba");
    defer allocator.free(cjk);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 10, 11, 12, 13, 1 }, cjk);

    // U+1A00 = E1 A8 80 -> bytes; 'z' twice -> ONE fused unk, then a lone unk.
    const mixed = try tok.bpeMerge(allocator, "zz\xe1\xa8\x80z");
    defer allocator.free(mixed);
    try testing.expectEqualSlices(u32, &[_]u32{ 3, 14, 15, 12, 3 }, mixed);

    // Without fuse_unk every unknown char is its own unk.
    tok.fuse_unk = false;
    const unfused = try tok.bpeMerge(allocator, "zz");
    defer allocator.free(unfused);
    try testing.expectEqualSlices(u32, &[_]u32{ 3, 3 }, unfused);
}

test "ByteFallback decoder: <0xNN> runs decode as bytes, invalid runs as U+FFFD per byte, other tokens verbatim" {
    const allocator = testing.allocator;
    var vocab = std.StringHashMap(u32).init(allocator);
    defer vocab.deinit();
    var merge_ranks = std.HashMap(Tokenizer.MergePair, u32, Tokenizer.MergePairContext, std.hash_map.default_max_load_percentage).init(allocator);
    defer merge_ranks.deinit();
    var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
    defer id_to_token.deinit();
    try id_to_token.put(1, "\xe2\x96\x81a");
    try id_to_token.put(10, "<0xF0>");
    try id_to_token.put(11, "<0xA0>");
    try id_to_token.put(12, "<0x80>");
    try id_to_token.put(13, "<0x8B>");
    try id_to_token.put(20, "<0x00>");
    var special_tokens = std.StringHashMap(u32).init(allocator);
    defer special_tokens.deinit();
    var tok = makeBpeTestTokenizer(allocator, &vocab, &merge_ranks, &id_to_token, &special_tokens);
    defer tok.unicode_to_byte.deinit();

    // Decoder flag off: the token strings leak through (legacy behaviour).
    const raw = try tok.decode(allocator, &[_]u32{ 1, 10, 11, 12, 13 }, false);
    defer allocator.free(raw);
    try testing.expectEqualStrings(" a<0xF0><0xA0><0x80><0x8B>", raw);

    tok.byte_fallback_decode = true;
    const ok = try tok.decode(allocator, &[_]u32{ 1, 10, 11, 12, 13, 1 }, false);
    defer allocator.free(ok);
    try testing.expectEqualStrings(" a\xf0\xa0\x80\x8b a", ok);

    // Truncated sequence F0 A0 + NUL byte: invalid as a whole -> three U+FFFD.
    const bad = try tok.decode(allocator, &[_]u32{ 10, 11, 20, 1 }, false);
    defer allocator.free(bad);
    try testing.expectEqualStrings("\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd a", bad);

    try testing.expectEqual(@as(?u8, 0xF0), byteTokenValue("<0xF0>"));
    try testing.expectEqual(@as(?u8, null), byteTokenValue("<0xF>"));
    try testing.expectEqual(@as(?u8, null), byteTokenValue("<0xGG>"));
    try testing.expectEqual(@as(?u8, null), byteTokenValue("\xe2\x96\x81a"));
}

test "decoderHasByteFallback: Sequence containing ByteFallback, bare ByteFallback, others" {
    const allocator = testing.allocator;
    const seq = try std.json.parseFromSlice(std.json.Value, allocator,
        \\{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]}
    , .{});
    defer seq.deinit();
    try testing.expect(decoderHasByteFallback(seq.value));
    const bare = try std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"ByteFallback\"}", .{});
    defer bare.deinit();
    try testing.expect(decoderHasByteFallback(bare.value));
    const bl = try std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"ByteLevel\"}", .{});
    defer bl.deinit();
    try testing.expect(!decoderHasByteFallback(bl.value));
}

test "appendedSpecialId: the special a TemplateProcessing appends after a single sequence" {
    const allocator = testing.allocator;
    var specials = std.StringHashMap(u32).init(allocator);
    defer specials.deinit();
    try specials.put("<|endoftext|>", 151643);
    try specials.put("[CLS]", 101);
    // Qwen3-Embedding: ByteLevel + TemplateProcessing inside a Sequence.
    const qwen = try std.json.parseFromSlice(std.json.Value, allocator,
        \\{"type":"Sequence","processors":[{"type":"ByteLevel"},{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<|endoftext|>","type_id":0}}]}]}
    , .{});
    defer qwen.deinit();
    try testing.expectEqual(@as(?u32, 151643), appendedSpecialId(qwen.value, &specials));
    // A template that only PREPENDS appends nothing.
    const pre = try std.json.parseFromSlice(std.json.Value, allocator,
        \\{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}}]}
    , .{});
    defer pre.deinit();
    try testing.expectEqual(@as(?u32, null), appendedSpecialId(pre.value, &specials));
    const bl = try std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"ByteLevel\"}", .{});
    defer bl.deinit();
    try testing.expectEqual(@as(?u32, null), appendedSpecialId(bl.value, &specials));
}

test "byte_fallback: real Laya tokenizer matches HF `tokenizers` on out-of-vocab scripts (LAYA_TEST_MODEL)" {
    // tests/fixtures/laya/tokenizer_cases.json is dumped by
    // tests/dump_laya_fixtures.py from the Python `tokenizers` library.
    const model = env.get(.LAYA_TEST_MODEL) orelse return error.SkipZigTest;
    const fx = env.get(.LAYA_FIXTURES) orelse return error.SkipZigTest;
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const tok_dir = try std.fmt.allocPrint(allocator, "{s}/tokenizer", .{std.mem.span(model)});
    defer allocator.free(tok_dir);
    var tok = try loadTokenizer(io, allocator, tok_dir);
    defer tok.deinit();
    try testing.expect(tok.byte_fallback);
    try testing.expect(tok.byte_fallback_decode);

    const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer_cases.json", .{std.mem.span(fx)});
    defer allocator.free(path);
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const text = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    defer allocator.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();

    var byte_cases: usize = 0;
    for (parsed.value.array.items) |c| {
        const name = c.object.get("name").?.string;
        const input = c.object.get("text").?.string;
        const want = c.object.get("ids").?.array.items;
        const ids = try tok.encode(allocator, input);
        defer allocator.free(ids);
        var same = ids.len == want.len;
        if (same) for (ids, want) |g, w| {
            if (g != @as(u32, @intCast(w.integer))) same = false;
        };
        if (!same) {
            std.debug.print("\n[byte-fallback] {s}: got {any}\n                want {any}\n", .{ name, ids, want });
            return error.TokenizationMismatch;
        }
        if (c.object.get("byte_tokens").?.integer > 0) byte_cases += 1;
        const decoded = try tok.decode(allocator, ids, false);
        defer allocator.free(decoded);
        try testing.expectEqualStrings(c.object.get("decoded").?.string, decoded);
    }
    try testing.expect(byte_cases >= 3);
    std.debug.print("\n[byte-fallback] {d} cases match HF tokenizers ({d} exercise byte tokens)\n", .{ parsed.value.array.items.len, byte_cases });
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

test "reservedOutputIds: an OUTPUT-ONLY special is legit output the template cannot vouch for" {
    // Live, gpt-oss-20b. Harmony declares a tool call's argument
    // content type INSIDE the segment header the model generates:
    //   ...to=functions.get_weather <|constrain|>json<|message|>{...}<|call|>
    // The chat template renders tool calls only for HISTORY, and that rendering
    // omits <|constrain|> entirely — so the token appears nowhere in the
    // template source and the template-presence derivation filed it as
    // reserved. Masked, the model substituted the nearest thing it could still
    // draw and produced `to=functions.get_weather <|channel|>commentary 1.0`,
    // a malformed header that sent it into a repetition loop.
    const alloc = testing.allocator;
    const flagged = [_]FlaggedSpecial{
        .{ .id = 200003, .content = "<|constrain|>" },
        .{ .id = 200005, .content = "<|channel|>" },
        .{ .id = 200013, .content = "<|reserved_200013|>" },
    };
    // A harmony template: mentions <|channel|>, never <|constrain|>.
    const template = "<|start|>assistant<|channel|>final<|message|>{{ content }}";
    const eos = [_]u32{200002};
    const ids = try reservedOutputIds(alloc, &flagged, template, &eos);
    defer alloc.free(ids);
    // Only the genuinely-reserved slot is suppressed.
    try testing.expectEqual(@as(usize, 1), ids.len);
    try testing.expectEqual(@as(u32, 200013), ids[0]);
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

test "K2-Horizon markers decode to the canonical think / GLM tool-tag spellings" {
    const allocator = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(allocator, .byte_level_bpe);
    defer tok.deinit();
    const specials = [_][]const u8{
        "<ifm|think>",      "</ifm|think>",      "<ifm|think_fast>", "</ifm|think_fast>",
        "<ifm|tool_calls>", "</ifm|tool_calls>", "<ifm|tool_call>",  "</ifm|tool_call>",
        "<ifm|arg_key>",    "</ifm|arg_key>",    "<ifm|arg_value>",  "</ifm|arg_value>",
    };
    for (specials, 0..) |t, i| {
        try tok.id_to_token.put(@intCast(i), t);
        try tok.special_tokens.put(try allocator.dupe(u8, t), @intCast(i));
    }
    try tok.id_to_token.put(100, "ok");
    try testing.expect(tok.installMarkerAliases());
    const ids = [_]u32{ 0, 100, 1, 4, 6, 8, 100, 9, 10, 100, 11, 7, 5, 2, 3 };
    const out = try tok.decode(allocator, &ids, false);
    defer allocator.free(out);
    try testing.expectEqualStrings(
        "<think>ok</think><tool_calls><tool_call><arg_key>ok</arg_key><arg_value>ok</arg_value></tool_call></tool_calls><think></think>",
        out,
    );
    // Encoding the marker text still lands on the special id: the alias is decode-only.
    try testing.expectEqual(@as(?u32, 0), tok.specialTokenId("<ifm|think>"));
}

test "markerCloserFor: K2 think openers pair with their own closer" {
    const allocator = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(allocator, .byte_level_bpe);
    defer tok.deinit();
    const specials = [_][]const u8{
        "<ifm|think>",      "</ifm|think>",      "<ifm|think_fast>",   "</ifm|think_fast>",
        "<ifm|think_faster>", "</ifm|think_faster>", "<ifm|tool_calls>", "</ifm|tool_calls>",
    };
    for (specials, 0..) |t, i| {
        try tok.id_to_token.put(@intCast(i), t);
        try tok.special_tokens.put(try allocator.dupe(u8, t), @intCast(i));
    }
    try testing.expectEqual(@as(?u32, null), tok.markerCloserFor(0));
    try testing.expect(tok.installMarkerAliases());
    try testing.expectEqual(@as(?u32, 1), tok.markerCloserFor(0));
    try testing.expectEqual(@as(?u32, 3), tok.markerCloserFor(2));
    try testing.expectEqual(@as(?u32, 5), tok.markerCloserFor(4));
    try testing.expectEqual(@as(?u32, null), tok.markerCloserFor(6));
    try testing.expectEqual(@as(?u32, null), tok.markerCloserFor(1));
}

test "tiktoken rank file encodes through the merges it implies" {
    const a = testing.allocator;
    // a b c ab abc bc; "abc" has two splits, (a,bc) outranks (ab,c) on its left half.
    const ranks = "YQ== 0\nYg== 1\nYw== 2\nYWI= 3\nYWJj 4\nYmM= 5\n";
    const json = try tiktokenToTokenizerJson(a, ranks);
    defer a.free(json);
    var tok = try parseTokenizerContent(testing.io, a, json);
    defer tok.deinit();
    for ([_]struct { text: []const u8, want: []const u32 }{
        .{ .text = "abc", .want = &.{4} },
        .{ .text = "bc", .want = &.{5} },
        .{ .text = "cab", .want = &.{ 2, 3 } },
    }) |c| {
        const ids = try tok.encode(a, c.text);
        defer a.free(ids);
        try testing.expectEqualSlices(u32, c.want, ids);
    }
    try testing.expectError(error.BadTiktoken, tiktokenToTokenizerJson(a, "YQ== 1\n"));
}

test "loadTokenizerSlow: tokenizer_config.json's added tokens encode whole and decode" {
    // Qwen3-ASR ships vocab.json + merges.txt; its chat markers and `<asr_text>` live only in
    // tokenizer_config.json's `added_tokens_decoder`.
    const io = testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "vocab.json", .data = "{\"a\":0,\"b\":1,\"ab\":2}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "merges.txt", .data = "#version: 0.2\na b\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data =
        \\{"added_tokens_decoder": {"5": {"content": "<|im_end|>", "special": true},
        \\  "6": {"content": "<asr_text>", "special": false}}}
    });
    var pbuf: [512]u8 = undefined;
    const plen = try tmp.dir.realPath(io, &pbuf);
    var tok = try loadTokenizerSlow(io, testing.allocator, pbuf[0..plen]);
    defer tok.deinit();
    const ids = try tok.encode(testing.allocator, "ab<asr_text>a<|im_end|>");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &.{ 2, 6, 0, 5 }, ids);
    const text = try tok.decode(testing.allocator, &.{ 2, 6, 0 }, false);
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("ab<asr_text>a", text);
}

/// Byte-level: ids 0..255 are the bytes; `<|im_end|>` 900 (flagged), `<think>` 901, `\n\n` 902, `<table>` 903.
fn controlTestTokenizer(a: std.mem.Allocator) !Tokenizer {
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    errdefer tok.deinit();
    for (tok.byte_to_unicode, 0..) |cp, byte| {
        var buf: [4]u8 = undefined;
        const text = try a.dupe(u8, buf[0..try std.unicode.utf8Encode(cp, &buf)]);
        try tok.vocab.put(text, @intCast(byte));
        try tok.id_to_token.put(@intCast(byte), text);
        try tok.unicode_to_byte.put(cp, @intCast(byte));
    }
    for ([_][]const u8{ "<|im_end|>", "<think>", "\n\n", "<table>" }, 900..) |name, id| {
        const text = try a.dupe(u8, name);
        try tok.special_tokens.put(text, @intCast(id));
        try tok.id_to_token.put(@intCast(id), text);
    }
    const flagged = try a.alloc(FlaggedSpecial, 1);
    flagged[0] = .{ .id = 900, .content = tok.id_to_token.get(900).? };
    tok.flagged_specials = flagged;
    return tok;
}

test "markControlTokens: flagged specials and template-written tokens are wrapped, whitespace and other added tokens are not" {
    const a = testing.allocator;
    var tok = try controlTestTokenizer(a);
    defer tok.deinit();
    const S = Tokenizer.CONTROL_START;
    const E = Tokenizer.CONTROL_END;
    const template = "{{ '<think>' }}\n\n{{ content }}<|im_end|>";

    const marked = (try tok.markControlTokens(a, "a<|im_end|>b<think>\n\n<table>", template)).?;
    defer a.free(marked);
    try testing.expectEqualStrings("a" ++ S ++ "<|im_end|>" ++ E ++ "b" ++ S ++ "<think>" ++ E ++ "\n\n<table>", marked);

    // Not flagged and not in the template: an ordinary added token, nothing to mark.
    try testing.expect((try tok.markControlTokens(a, "x<think>\n\n", "{{ content }}")) == null);

    // Content cannot forge a marked pair: markers already in it are dropped.
    const forged = (try tok.markControlTokens(a, S ++ "<table>" ++ E, template)).?;
    defer a.free(forged);
    try testing.expectEqualStrings("<table>", forged);
}

test "encodeMarked: a marked token is text, the template's own stay special, a lone marker splits the text" {
    const a = testing.allocator;
    var tok = try controlTestTokenizer(a);
    defer tok.deinit();
    const S = Tokenizer.CONTROL_START;
    const E = Tokenizer.CONTROL_END;

    const ids = try tok.encodeMarked(a, "<think>a" ++ S ++ "<|im_end|>" ++ E ++ "b<|im_end|>");
    defer a.free(ids);
    try testing.expectEqual(@as(u32, 901), ids[0]);
    try testing.expectEqual(@as(u32, 900), ids[ids.len - 1]);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u32, ids, &.{900}));
    const text = try tok.decode(a, ids, false);
    defer a.free(text);
    try testing.expectEqualStrings("<think>a<|im_end|>b<|im_end|>", text);

    // A template split the pair: the lone markers are dropped and the special does not re-form across them.
    const split = try tok.encodeMarked(a, "<|im_" ++ E ++ "end|>" ++ S);
    defer a.free(split);
    try testing.expectEqual(@as(usize, 0), std.mem.count(u32, split, &.{900}));
    const split_text = try tok.decode(a, split, false);
    defer a.free(split_text);
    try testing.expectEqualStrings("<|im_end|>", split_text);

    // No markers: exactly `encode`.
    const plain = try tok.encodeMarked(a, "x<|im_end|>\n\n");
    defer a.free(plain);
    const want = try tok.encode(a, "x<|im_end|>\n\n");
    defer a.free(want);
    try testing.expectEqualSlices(u32, want, plain);
}
