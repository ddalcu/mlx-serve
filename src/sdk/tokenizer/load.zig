//! Tokenizer loading: tokenizer.json and tiktoken files, the pre-tokenizer regex probes.

const std = @import("std");
const log = @import("../util/log.zig");
const mlx_gguf = @import("../quant/gguf/claim.zig");
const io_util = @import("../util/io_util.zig");

const TokenizerType = @import("tokenizer.zig").TokenizerType;
const FlaggedSpecial = @import("tokenizer.zig").FlaggedSpecial;
const PretokStyle = @import("tokenizer.zig").PretokStyle;
const IsolatedRuns = @import("tokenizer.zig").IsolatedRuns;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const buildBytesToUnicode = @import("pretokenize.zig").buildBytesToUnicode;
const hasByteLevel = @import("tokenizer_test.zig").hasByteLevel;

/// Parse tokenizer.json and return a Tokenizer.
pub fn loadTokenizer(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Tokenizer {
    const content = (try mlx_gguf.sidecar(io, allocator, model_dir, .tokenizer)) orelse blk: {
        const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{model_dir});
        defer allocator.free(path);

        const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer file.close(io);

        var read_buf: [4096]u8 = undefined;
        var reader_state = file.reader(io, &read_buf);
        break :blk try reader_state.interface.allocRemaining(allocator, .limited(256 * 1024 * 1024));
    };
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
    try buf.appendSlice(allocator, "]},\"added_tokens\":");
    try appendAddedTokens(io, allocator, &buf, model_dir);
    try buf.append(allocator, '}');
    return parseTokenizerContent(io, allocator, buf.items);
}

/// The slow format's added tokens (chat markers, `<think>`, Qwen3-ASR's `<asr_text>`) live in
/// tokenizer_config.json's `added_tokens_decoder`, keyed by id; `[]` when it has none.
fn appendAddedTokens(io: std.Io, allocator: std.mem.Allocator, buf: *std.ArrayList(u8), model_dir: []const u8) !void {
    const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer_config.json", .{model_dir});
    defer allocator.free(path);
    const bytes = readFileAllocTok(io, allocator, path) catch |e| switch (e) {
        error.FileNotFound => return buf.appendSlice(allocator, "[]"),
        else => return e,
    };
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();
    try buf.append(allocator, '[');
    const decoder = if (parsed.value == .object) parsed.value.object.get("added_tokens_decoder") else null;
    if (decoder) |d| if (d == .object) {
        var it = d.object.iterator();
        var first = true;
        while (it.next()) |e| {
            const id = std.fmt.parseInt(u32, e.key_ptr.*, 10) catch continue;
            if (e.value_ptr.* != .object) continue;
            const content = e.value_ptr.object.get("content") orelse continue;
            if (content != .string) continue;
            const special = if (e.value_ptr.object.get("special")) |v| v == .bool and v.bool else false;
            if (!first) try buf.append(allocator, ',');
            first = false;
            try buf.print(allocator, "{{\"id\":{d},\"special\":{},\"content\":\"", .{ id, special });
            try appendJsonEscapedTok(allocator, buf, content.string);
            try buf.appendSlice(allocator, "\"}");
        }
    };
    try buf.append(allocator, ']');
}

/// Load an OpenAI tiktoken rank file (`<base64 bytes> <rank>` per line, ranks
/// dense from 0) as a byte-level BPE tokenizer, through the equivalent
/// synthesized `tokenizer.json`. No special tokens: callers add control ids.
pub fn loadTiktoken(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Tokenizer {
    const file_bytes = try readFileAllocTok(io, allocator, path);
    defer allocator.free(file_bytes);
    const json = try tiktokenToTokenizerJson(allocator, file_bytes);
    defer allocator.free(json);
    return parseTokenizerContent(io, allocator, json);
}

/// Every split of a token whose halves are both tokens is a merge, ordered by
/// the merged token's rank then its halves' (transformers' tiktoken converter).
pub fn tiktokenToTokenizerJson(allocator: std.mem.Allocator, file_bytes: []const u8) ![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var toks: std.ArrayList([]const u8) = .empty;
    var ranks = std.StringHashMap(u32).init(arena);
    var lines = std.mem.splitScalar(u8, file_bytes, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0) continue;
        const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return error.BadTiktoken;
        const dec = std.base64.standard.Decoder;
        const bytes = try arena.alloc(u8, dec.calcSizeForSlice(line[0..sp]) catch return error.BadTiktoken);
        dec.decode(bytes, line[0..sp]) catch return error.BadTiktoken;
        const rank = std.fmt.parseInt(u32, line[sp + 1 ..], 10) catch return error.BadTiktoken;
        if (rank != toks.items.len) return error.BadTiktoken;
        try toks.append(arena, bytes);
        try ranks.put(bytes, rank);
    }

    const b2u = buildBytesToUnicode();
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    const Emit = struct {
        fn token(a: std.mem.Allocator, o: *std.ArrayList(u8), map: *const [256]u21, bytes: []const u8) !void {
            var enc: std.ArrayList(u8) = .empty;
            defer enc.deinit(a);
            for (bytes) |b| {
                var utf8: [4]u8 = undefined;
                const n = std.unicode.utf8Encode(map[b], &utf8) catch unreachable;
                try enc.appendSlice(a, utf8[0..n]);
            }
            try appendJsonEscapedTok(a, o, enc.items);
        }
    };
    try out.appendSlice(allocator, "{\"pre_tokenizer\":{\"type\":\"ByteLevel\"},\"model\":{\"type\":\"BPE\",\"vocab\":{");
    for (toks.items, 0..) |t, rank| {
        if (rank > 0) try out.append(allocator, ',');
        try out.append(allocator, '"');
        try Emit.token(allocator, &out, &b2u, t);
        try out.print(allocator, "\":{d}", .{rank});
    }
    try out.appendSlice(allocator, "},\"merges\":[");
    var first = true;
    for (toks.items) |t| {
        const Cand = struct { l: u32, r: u32, at: usize };
        var cands: std.ArrayList(Cand) = .empty;
        for (1..t.len) |at| {
            const l = ranks.get(t[0..at]) orelse continue;
            const r = ranks.get(t[at..]) orelse continue;
            try cands.append(arena, .{ .l = l, .r = r, .at = at });
        }
        std.mem.sort(Cand, cands.items, {}, struct {
            fn lt(_: void, a: Cand, b: Cand) bool {
                return if (a.l != b.l) a.l < b.l else a.r < b.r;
            }
        }.lt);
        for (cands.items) |c| {
            if (!first) try out.append(allocator, ',');
            first = false;
            try out.append(allocator, '"');
            try Emit.token(allocator, &out, &b2u, t[0..c.at]);
            try out.append(allocator, ' ');
            try Emit.token(allocator, &out, &b2u, t[c.at..]);
            try out.append(allocator, '"');
        }
    }
    try out.appendSlice(allocator, "]}}");
    return out.toOwnedSlice(allocator);
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
pub fn parseTokenizerContent(io: std.Io, allocator: std.mem.Allocator, content: []const u8) !Tokenizer {
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

    var special_index = try Tokenizer.SpecialIndex.init(allocator, &special_tokens);
    errdefer special_index.deinit(allocator);
    var built: Tokenizer = .{
        .special_index = special_index,
        .vocab = vocab,
        .id_to_token = id_to_token,
        .merge_ranks = merge_ranks,
        .allocator = allocator,
        .special_tokens = special_tokens,
        .flagged_specials = try flagged.toOwnedSlice(allocator),
        .tok_type = tok_type,
        .digit_group = if (root.get("pre_tokenizer")) |pt| digitGroupFromPreTokenizer(pt) else 1,
        .pretok_style = if (root.get("pre_tokenizer")) |pt| pretokStyleFromPreTokenizer(pt) else .gpt2,
        .isolated_runs = if (root.get("pre_tokenizer")) |pt| isolatedRunsFromPreTokenizer(pt) else .{},
        .metaspace_prepend = if (root.get("pre_tokenizer")) |pt| metaspaceFromPreTokenizer(pt).prepend else .never,
        .metaspace_split = if (root.get("pre_tokenizer")) |pt| metaspaceFromPreTokenizer(pt).split else false,
        .byte_fallback = if (model_obj.get("byte_fallback")) |v| v == .bool and v.bool else false,
        .ignore_merges = if (model_obj.get("ignore_merges")) |v| v == .bool and v.bool else false,
        .wordpiece_max_chars = if (model_obj.get("max_input_chars_per_word")) |v| (if (v == .integer and v.integer > 0) @intCast(v.integer) else 100) else 100,
        .fuse_unk = if (model_obj.get("fuse_unk")) |v| v == .bool and v.bool else false,
        .unk_id = if (model_obj.get("unk_token")) |v| (if (v == .string) vocab.get(v.string) else null) else null,
        .byte_fallback_decode = if (root.get("decoder")) |d| decoderHasByteFallback(d) else false,
        .appended_special = if (root.get("post_processor")) |pp| appendedSpecialId(pp, &special_tokens) else null,
        .byte_to_unicode = byte_to_unicode,
        .unicode_to_byte = unicode_to_byte,
        .bos_id = bos_id,
        .eos_id = eos_id,
        .parsed_json = parsed,
    };
    if (built.installMarkerAliases()) log.info("Tokenizer: K2-Horizon markers alias to <think> / GLM tool tags\n", .{});
    return built;
}

/// Digits per pre-token from the tokenizer.json `pre_tokenizer` spec: a
/// `Split` rule (top-level or inside a `Sequence`) whose regex is exactly
/// `\p{N}{1,3}` selects 3-digit grouping; anything else keeps the GPT-2
/// single-digit rule. Deliberately exact-match — an unrecognized digit rule
/// keeps the conservative behavior rather than guessing.
pub fn digitGroupFromPreTokenizer(pt: std.json.Value) u8 {
    if (pt != .object) return 1;
    if (splitRegexIsDigits13(pt)) return 3;
    var group: u8 = 1;
    if (pt.object.get("pretokenizers")) |list| {
        if (list == .array) {
            for (list.array.items) |sub| {
                if (sub != .object) continue;
                if (splitRegexIsDigits13(sub)) group = 3;
                // A later `Digits(individual_digits)` rule re-splits every
                // group the {1,3} rule formed (Spark-X2.5).
                if (isIndividualDigitsRule(sub)) group = 1;
            }
        }
    }
    return group;
}

fn isIndividualDigitsRule(node: std.json.Value) bool {
    const t = node.object.get("type") orelse return false;
    if (t != .string or !std.mem.eql(u8, t.string, "Digits")) return false;
    const ind = node.object.get("individual_digits") orelse return false;
    return ind == .bool and ind.bool;
}

/// Pre-tokenizer grammar from the tokenizer.json Split regex. The muse /
/// Llama-3 family ships ONE combined pattern whose signature is the attached
/// contraction group PLUS the {1,3} digit rule — the exact-match digit
/// detection alone misses it because `\p{N}{1,3}` is an alternation branch,
/// not the whole regex (live: "84" served per-digit, the model
/// echoed "8 4" — the DSV4 echo-precision class on a new spelling).
pub fn pretokStyleFromPreTokenizer(pt: std.json.Value) PretokStyle {
    if (pt != .object) return .gpt2;
    if (casedStyleOf(pt)) |style| return style;
    if (pt.object.get("pretokenizers")) |list| {
        if (list == .array) {
            for (list.array.items) |sub| {
                if (sub != .object) continue;
                if (casedStyleOf(sub)) |style| return style;
            }
        }
    }
    return .gpt2;
}

/// Only a Sequence entry whose regex IS the class isolates it; the same class as a branch of a
/// combined regex (Llama-3's `\p{N}{1,3}`) does not.
pub fn isolatedRunsFromPreTokenizer(pt: std.json.Value) IsolatedRuns {
    var runs: IsolatedRuns = .{};
    if (pt != .object) return runs;
    const list = pt.object.get("pretokenizers") orelse return runs;
    if (list != .array) return runs;
    for (list.array.items) |sub| {
        if (sub != .object) continue;
        const rx = splitRegexOf(sub) orelse continue;
        if (std.mem.eql(u8, rx, "\\p{N}{1,3}")) runs.digits = true;
        if (std.mem.eql(u8, rx, "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+")) runs.cjk = true;
    }
    return runs;
}

fn casedStyleOf(node: std.json.Value) ?PretokStyle {
    const rx = splitRegexOf(node) orelse return null;
    if (!llama3StyleFromSplitRegex(rx)) return null;
    return if (std.mem.indexOf(u8, rx, "[\\r\\n/]") != null) .llama3 else .cased;
}

/// The cased grammar (Muse; Flint, whose digits are a Split of their own): the plain Llama-3
/// regex (`\p{L}+`, no case classes) is the gpt2 grammar with 3-digit groups.
pub fn llama3StyleFromSplitRegex(rx: []const u8) bool {
    return std.mem.indexOf(u8, rx, "'s|'t|'re|'ve|'m|'ll|'d") != null and
        std.mem.indexOf(u8, rx, "\\p{Lu}") != null;
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
pub fn parseMergePair(merge_val: std.json.Value) ?Tokenizer.MergePair {
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
pub const MetaspacePrepend = enum { never, first, always };
pub const MetaspaceOpts = struct { prepend: MetaspacePrepend = .never, split: bool = false };

/// `Metaspace` pre-tokenizer options from tokenizer.json (top-level or inside
/// a `Sequence`). Absent → both false, which is the pre-existing behaviour
/// (Gemma 4's `Split " " MergedWithPrevious` never reaches here).
/// `<0xNN>` -> NN, null for any other token.
pub fn byteTokenValue(token: []const u8) ?u8 {
    if (token.len != 6 or !std.mem.startsWith(u8, token, "<0x") or token[5] != '>') return null;
    return std.fmt.parseInt(u8, token[3..5], 16) catch null;
}

pub fn flushByteRun(allocator: std.mem.Allocator, out: *std.ArrayList(u8), pending: *std.ArrayList(u8)) !void {
    if (pending.items.len == 0) return;
    if (std.unicode.utf8ValidateSlice(pending.items)) {
        try out.appendSlice(allocator, pending.items);
    } else {
        for (pending.items) |_| try out.appendSlice(allocator, "\xef\xbf\xbd");
    }
    pending.clearRetainingCapacity();
}

/// True when the tokenizer.json `decoder` is `ByteFallback` or a `Sequence`
/// containing one.
pub fn decoderHasByteFallback(dec: std.json.Value) bool {
    if (dec != .object) return false;
    const t = dec.object.get("type") orelse return false;
    if (t == .string and std.mem.eql(u8, t.string, "ByteFallback")) return true;
    if (t == .string and std.mem.eql(u8, t.string, "Sequence")) {
        const subs = dec.object.get("decoders") orelse return false;
        if (subs != .array) return false;
        for (subs.array.items) |sub| if (decoderHasByteFallback(sub)) return true;
    }
    return false;
}

/// The id of the `SpecialToken` a `TemplateProcessing` (top-level or inside a `Sequence`) puts
/// LAST in its `single` template, or null when nothing is appended.
pub fn appendedSpecialId(pp: std.json.Value, specials: *const std.StringHashMap(u32)) ?u32 {
    if (pp != .object) return null;
    const t = pp.object.get("type") orelse return null;
    if (t != .string) return null;
    if (std.mem.eql(u8, t.string, "Sequence")) {
        const subs = pp.object.get("processors") orelse return null;
        if (subs != .array) return null;
        for (subs.array.items) |sub| if (appendedSpecialId(sub, specials)) |id| return id;
        return null;
    }
    if (!std.mem.eql(u8, t.string, "TemplateProcessing")) return null;
    const single = pp.object.get("single") orelse return null;
    if (single != .array or single.array.items.len == 0) return null;
    const last = single.array.items[single.array.items.len - 1];
    if (last != .object) return null;
    const st = last.object.get("SpecialToken") orelse return null;
    if (st != .object) return null;
    const name = st.object.get("id") orelse return null;
    return if (name == .string) specials.get(name.string) else null;
}

pub fn metaspaceFromPreTokenizer(pt: std.json.Value) MetaspaceOpts {
    if (pt != .object) return .{};
    const t = pt.object.get("type") orelse return .{};
    if (t != .string) return .{};
    if (std.mem.eql(u8, t.string, "Sequence")) {
        if (pt.object.get("pretokenizers")) |pts| {
            if (pts == .array) for (pts.array.items) |sub| {
                const o = metaspaceFromPreTokenizer(sub);
                if (o.prepend != .never or o.split) return o;
            };
        }
        return .{};
    }
    if (!std.mem.eql(u8, t.string, "Metaspace")) return .{};
    var opts = MetaspaceOpts{ .prepend = .always, .split = true };
    if (pt.object.get("prepend_scheme")) |ps| {
        if (ps == .string) opts.prepend = std.meta.stringToEnum(MetaspacePrepend, ps.string) orelse .always;
    }
    if (pt.object.get("split")) |sp| {
        if (sp == .bool) opts.split = sp.bool;
    }
    return opts;
}
