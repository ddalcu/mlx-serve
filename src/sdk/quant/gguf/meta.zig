//! Everything mlx-serve normally reads from sidecar JSON files, rebuilt in
//! memory from the GGUF metadata: config.json, tokenizer.json,
//! tokenizer_config.json, generation_config.json.
const std = @import("std");
const env = @import("../../../env.zig");
const gguf = @import("gguf.zig");
const Arch = @import("arch.zig").Arch;

pub const Error = error{ UnsupportedArch, UnsupportedTokenizer, UnsupportedTensor, MissingMetadata, BadMetadata } || std.mem.Allocator.Error;

/// Why we can't serve this file, null when we can. The caller falls back to
/// another engine on a non-null answer, so this has to be cheap and complete.
pub fn unsupportedReason(f: *const gguf.File, buf: []u8) ?[]const u8 {
    const arch = Arch.read(f) catch |err| return switch (err) {
        error.UnsupportedArch => std.fmt.bufPrint(buf, "arch {s}", .{f.getString("general.architecture").?}) catch "arch",
        else => "metadata",
    };
    if (TokenizerKind.of(f) == null) return "tokenizer";
    var name_buf: [160]u8 = undefined;
    var it = f.tensors.iterator();
    while (it.next()) |e| {
        if (!e.value_ptr.ty.supported()) return std.fmt.bufPrint(buf, "tensor type of {s}", .{e.key_ptr.*}) catch "tensor type";
        if (arch.mapTensor(&name_buf, e.key_ptr.*) == null) return std.fmt.bufPrint(buf, "tensor {s}", .{e.key_ptr.*}) catch "tensor";
    }
    return null;
}

pub fn configJson(allocator: std.mem.Allocator, f: *const gguf.File) ![]u8 {
    return (try Arch.read(f)).configJson(allocator);
}

/// The tokenizer.json blocks around the BPE vocab, by llama.cpp tokenizer
/// model + pre-tokenizer id. mlx-serve keys its encode path on the
/// pre_tokenizer (ByteLevel or not), the rest mirrors the HF files.
const TokenizerKind = struct {
    /// Everything before "model".
    head: []const u8,
    /// Extra BPE model fields, in front of "vocab".
    model_fields: []const u8 = "",
    /// A token HF lists as special that the GGUF types as a normal one.
    also_special: []const u8 = "",
    /// A SentencePiece vocab without merges: user-defined tokens keep their
    /// spaces as "\u2581" and the merges are derived from the scores the way
    /// HF's converter does it.
    spm: bool = false,

    const gemma_head =
        \\"normalizer":{"type":"Replace","pattern":{"String":" "},"content":"\u2581"},
        \\"pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false},
        \\"decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"\u2581"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},
    ;
    const gemma_model_fields =
        \\"unk_token":"<unk>","fuse_unk":true,"byte_fallback":true,
    ;

    fn of(f: *const gguf.File) ?TokenizerKind {
        const model = f.getString("tokenizer.ggml.model") orelse return null;
        if (std.mem.eql(u8, model, "gemma4")) return .{ .head = gemma_head, .model_fields = gemma_model_fields, .also_special = "<eos>" };
        const arch_name = f.getString("general.architecture") orelse return null;
        if (std.mem.eql(u8, model, "llama") and std.mem.eql(u8, arch_name, "gemma3")) return .{ .head = gemma_head, .model_fields = gemma_model_fields, .also_special = "<eos>", .spm = true };
        const pre = f.getString("tokenizer.ggml.pre") orelse return null;
        if (std.mem.eql(u8, model, "gpt2") and std.mem.eql(u8, pre, "gpt-4o")) return .{
            .head =
            \\"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
            \\"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
            ,
            .model_fields =
            \\"ignore_merges":true,
            ,
        };
        if (std.mem.eql(u8, model, "gpt2") and std.mem.eql(u8, pre, "pixtral")) return .{
            .head =
            \\"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
            \\"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
            ,
            .model_fields =
            \\"ignore_merges":true,
            ,
        };
        if (std.mem.eql(u8, model, "gpt2") and std.mem.eql(u8, pre, "qwen2")) return .{ .head =
        \\"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        };
        if (std.mem.eql(u8, model, "gpt2") and std.mem.eql(u8, pre, "lfm2")) return .{ .head =
        \\"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
        \\"decoder":{"type":"Sequence","decoders":[{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}]},
        };
        if (std.mem.eql(u8, model, "gpt2") and std.mem.eql(u8, pre, "qwen35")) return .{ .head =
        \\"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        };
        return null;
    }
};

// llama.cpp token types.
const TOKEN_CONTROL = 3;
const TOKEN_USER_DEFINED = 4;

/// An HF tokenizer.json equivalent: BPE vocab + merges, control and
/// user-defined tokens as added_tokens (control = special).
pub fn tokenizerJson(allocator: std.mem.Allocator, f: *const gguf.File) ![]u8 {
    const kind = TokenizerKind.of(f) orelse return error.UnsupportedTokenizer;
    const tokens = f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata;
    const types = f.getArray("tokenizer.ggml.token_type") orelse return error.MissingMetadata;
    const merges = f.getArray("tokenizer.ggml.merges");
    if (merges == null and !kind.spm) return error.MissingMetadata;
    if (types.len != tokens.len) return error.BadMetadata;

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var added: std.ArrayList(u8) = .empty;
    defer added.deinit(allocator);

    try out.append(allocator, '{');
    try out.appendSlice(allocator, kind.head);
    try out.appendSlice(allocator, "\"model\":{\"type\":\"BPE\",");
    try out.appendSlice(allocator, kind.model_fields);
    try out.appendSlice(allocator, "\"vocab\":{");
    var it = tokens.strings();
    var id: usize = 0;
    var first = true;
    while (it.next()) |tok| : (id += 1) {
        const ty = types.int(id);
        const special = ty == TOKEN_CONTROL or (kind.also_special.len > 0 and std.mem.eql(u8, tok, kind.also_special));
        const spaces = kind.spm and ty == TOKEN_USER_DEFINED;
        if (special or ty == TOKEN_USER_DEFINED) {
            if (added.items.len > 0) try added.append(allocator, ',');
            try added.print(allocator, "{{\"id\":{d},\"special\":{},\"content\":", .{ id, special });
            try appendJsonToken(allocator, &added, tok, spaces);
            try added.append(allocator, '}');
            continue;
        }
        if (!first) try out.append(allocator, ',');
        first = false;
        try appendJsonToken(allocator, &out, tok, spaces);
        try out.print(allocator, ":{d}", .{id});
    }
    try out.appendSlice(allocator, "},\"merges\":[");
    if (merges) |arr| {
        var mit = arr.strings();
        first = true;
        while (mit.next()) |m| {
            if (!first) try out.append(allocator, ',');
            first = false;
            try appendJsonString(allocator, &out, m);
        }
    } else try appendDerivedMerges(allocator, &out, tokens, types, f.getArray("tokenizer.ggml.scores") orelse return error.MissingMetadata);
    try out.appendSlice(allocator, "]},\"added_tokens\":[");
    try out.appendSlice(allocator, added.items);
    try out.appendSlice(allocator, "]}");
    return out.toOwnedSlice(allocator);
}

/// The tokenizer_config.json fields mlx-serve reads: template + named special tokens.
pub fn tokenizerConfigJson(allocator: std.mem.Allocator, f: *const gguf.File) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"chat_template\":");
    try appendJsonString(allocator, &out, f.getString("tokenizer.chat_template") orelse "");
    const named = [_]struct { []const u8, []const u8 }{
        .{ "eos_token", "tokenizer.ggml.eos_token_id" },
        .{ "bos_token", "tokenizer.ggml.bos_token_id" },
        .{ "pad_token", "tokenizer.ggml.padding_token_id" },
    };
    for (named) |n| {
        try out.print(allocator, ",\"{s}\":", .{n[0]});
        if (tokenById(f, f.getInt(n[1]))) |tok| try appendJsonString(allocator, &out, tok) else try out.appendSlice(allocator, "null");
    }
    try out.append(allocator, '}');
    return out.toOwnedSlice(allocator);
}

/// The sampling defaults the model author recommends, null when the GGUF has none.
pub fn generationConfigJson(allocator: std.mem.Allocator, f: *const gguf.File) !?[]u8 {
    const temp = f.getFloat("general.sampling.temp");
    const top_p = f.getFloat("general.sampling.top_p");
    const top_k = f.getInt("general.sampling.top_k");
    if (temp == null and top_p == null and top_k == null) return null;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"do_sample\":true");
    if (temp) |v| try out.print(allocator, ",\"temperature\":{d}", .{v});
    // f32 in the file: 0.95 reads back as 0.949999988, round it to what the author wrote.
    if (top_p) |v| try out.print(allocator, ",\"top_p\":{d}", .{@round(v * 1e4) / 1e4});
    if (top_k) |v| try out.print(allocator, ",\"top_k\":{d}", .{v});
    try out.append(allocator, '}');
    return try out.toOwnedSlice(allocator);
}

fn tokenById(f: *const gguf.File, id: ?i64) ?[]const u8 {
    const want = std.math.cast(usize, id orelse return null) orelse return null;
    const tokens = f.getArray("tokenizer.ggml.tokens") orelse return null;
    if (want >= tokens.len) return null;
    var it = tokens.strings();
    for (0..want) |_| _ = it.next();
    return it.next();
}

/// A token's text; user-defined SentencePiece tokens spell their spaces "\u2581" (HF's vocab form).
fn appendJsonToken(allocator: std.mem.Allocator, out: *std.ArrayList(u8), tok: []const u8, spaces_as_underscore: bool) !void {
    if (!spaces_as_underscore) return appendJsonString(allocator, out, tok);
    const mapped = try std.mem.replaceOwned(u8, allocator, tok, " ", "\xe2\x96\x81");
    defer allocator.free(mapped);
    return appendJsonString(allocator, out, mapped);
}

const Merge = struct { tok: u32, split: u32, l: u32, r: u32, score: f32, len_l: u32, len_r: u32 };

fn cpLen(s: []const u8) u32 {
    var n: u32 = 0;
    for (s) |c| n += @intFromBool(c & 0xC0 != 0x80);
    return n;
}

/// The BPE merges HF's SentencePiece converter derives from a vocab with
/// scores (transformers' `generate_merges` with `vocab_scores`): every split of
/// a token into two vocab pieces is a merge, listed per token by (left id,
/// right id), then all of them by score, longest left then right piece first
/// (stable). User-defined tokens rank as score 0 (the GGUF stores -1000 for
/// them). Checked against google/gemma-3's tokenizer.json: identical.
fn appendDerivedMerges(allocator: std.mem.Allocator, out: *std.ArrayList(u8), tokens: gguf.Array, types: gguf.Array, scores: gguf.Array) !void {
    if (scores.len != tokens.len) return error.BadMetadata;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const texts = try a.alloc([]const u8, tokens.len);
    var vocab = std.StringHashMap(u32).init(a);
    try vocab.ensureTotalCapacity(@intCast(tokens.len));
    var it = tokens.strings();
    var id: u32 = 0;
    while (it.next()) |tok| : (id += 1) {
        texts[id] = if (types.int(id) == TOKEN_USER_DEFINED) try std.mem.replaceOwned(u8, a, tok, " ", "\xe2\x96\x81") else tok;
        try vocab.put(texts[id], id);
    }
    var merges: std.ArrayList(Merge) = .empty;
    var local: std.ArrayList(Merge) = .empty;
    for (texts, 0..) |t, i| {
        const score: f32 = if (types.int(i) == TOKEN_USER_DEFINED) 0 else @floatCast(scores.float(i));
        local.clearRetainingCapacity();
        for (1..t.len) |k| {
            if (t[k] & 0xC0 == 0x80) continue;
            const l = vocab.get(t[0..k]) orelse continue;
            const r = vocab.get(t[k..]) orelse continue;
            try local.append(a, .{ .tok = @intCast(i), .split = @intCast(k), .l = l, .r = r, .score = score, .len_l = cpLen(t[0..k]), .len_r = cpLen(t[k..]) });
        }
        std.mem.sort(Merge, local.items, {}, struct {
            fn lt(_: void, x: Merge, y: Merge) bool {
                return if (x.l != y.l) x.l < y.l else x.r < y.r;
            }
        }.lt);
        try merges.appendSlice(a, local.items);
    }
    // Block sort is stable, so ties keep the per-token order above.
    std.mem.sort(Merge, merges.items, {}, struct {
        fn lt(_: void, x: Merge, y: Merge) bool {
            if (x.score != y.score) return x.score > y.score;
            if (x.len_l != y.len_l) return x.len_l > y.len_l;
            return x.len_r > y.len_r;
        }
    }.lt);
    var pair: std.ArrayList(u8) = .empty;
    for (merges.items, 0..) |m, j| {
        if (j > 0) try out.append(allocator, ',');
        pair.clearRetainingCapacity();
        try pair.appendSlice(a, texts[m.tok][0..m.split]);
        try pair.append(a, ' ');
        try pair.appendSlice(a, texts[m.tok][m.split..]);
        try appendJsonString(allocator, out, pair.items);
    }
}

test "derived merges follow HF's order: score, then longer pieces, user-defined tokens as score 0" {
    // vocab: a b ab abb bb "  " (user-defined, spaces) with scores
    const w = TestArrays{ .tokens = &.{ "a", "b", "\xe2\x96\x81", "ab", "abb", "bb", "  " }, .types = &.{ 1, 1, 1, 1, 1, 1, TOKEN_USER_DEFINED }, .scores = &.{ -1, -1, -1, -3, -2, -3, -1000 } };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    try appendDerivedMerges(std.testing.allocator, &out, w.tokens_arr(), w.types_arr(), w.scores_arr());
    // "  " -> "\u2581 \u2581" (score 0) first; abb (-2) splits: (a, bb) and (ab, b): longer left first; then ab and bb (-3) in token order.
    try std.testing.expectEqualStrings("\"\xe2\x96\x81 \xe2\x96\x81\",\"ab b\",\"a bb\",\"a b\",\"b b\"", out.items);
}

/// GGUF arrays built in memory for the test above.
const TestArrays = struct {
    tokens: []const []const u8,
    types: []const i64,
    scores: []const f32,
    buf_t: [256]u8 = undefined,
    buf_y: [64]u8 = undefined,
    buf_s: [64]u8 = undefined,

    fn tokens_arr(self: *const TestArrays) gguf.Array {
        const buf = @constCast(&self.buf_t);
        var n: usize = 0;
        for (self.tokens) |t| {
            std.mem.writeInt(u64, buf[n..][0..8], t.len, .little);
            @memcpy(buf[n + 8 ..][0..t.len], t);
            n += 8 + t.len;
        }
        return .{ .elem_type = .string, .len = self.tokens.len, .bytes = buf[0..n] };
    }
    fn types_arr(self: *const TestArrays) gguf.Array {
        const buf = @constCast(&self.buf_y);
        for (self.types, 0..) |t, i| std.mem.writeInt(i32, buf[4 * i ..][0..4], @intCast(t), .little);
        return .{ .elem_type = .i32, .len = self.types.len, .bytes = buf[0 .. 4 * self.types.len] };
    }
    fn scores_arr(self: *const TestArrays) gguf.Array {
        const buf = @constCast(&self.buf_s);
        for (self.scores, 0..) |v, i| std.mem.writeInt(u32, buf[4 * i ..][0..4], @bitCast(v), .little);
        return .{ .elem_type = .f32, .len = self.scores.len, .bytes = buf[0 .. 4 * self.scores.len] };
    }
};

fn appendJsonString(allocator: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    try out.append(allocator, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(allocator, "\\\""),
        '\\' => try out.appendSlice(allocator, "\\\\"),
        '\n' => try out.appendSlice(allocator, "\\n"),
        '\r' => try out.appendSlice(allocator, "\\r"),
        '\t' => try out.appendSlice(allocator, "\\t"),
        else => if (c < 0x20) try out.print(allocator, "\\u{x:0>4}", .{c}) else try out.append(allocator, c),
    };
    try out.append(allocator, '"');
}

test "appendJsonString round-trips through the JSON parser" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    const raw = "a\"b\\c\nd\x01\xc4\xa0";
    try appendJsonString(std.testing.allocator, &out, raw);
    const parsed = try std.json.parseFromSlice([]const u8, std.testing.allocator, out.items, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(raw, parsed.value);
}

test "real model: synthesized JSON parses and matches the GGUF (a real GGUF, env-gated)" {
    const path = std.mem.span(env.get(.MLX_SERVE_GGUF_TEST_MODEL) orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    var f = try gguf.File.open(a, path);
    defer f.deinit();
    var buf: [200]u8 = undefined;
    try std.testing.expectEqual(@as(?[]const u8, null), unsupportedReason(&f, &buf));

    const cfg = try configJson(a, &f);
    defer a.free(cfg);
    const cfg_parsed = try std.json.parseFromSlice(std.json.Value, a, cfg, .{});
    defer cfg_parsed.deinit();

    const tj = try tokenizerJson(a, &f);
    defer a.free(tj);
    const tok = try std.json.parseFromSlice(std.json.Value, a, tj, .{});
    defer tok.deinit();
    const n_vocab = tok.value.object.get("model").?.object.get("vocab").?.object.count();
    const n_added = tok.value.object.get("added_tokens").?.array.items.len;
    try std.testing.expectEqual(f.getArray("tokenizer.ggml.tokens").?.len, n_vocab + n_added);

    const tc = try tokenizerConfigJson(a, &f);
    defer a.free(tc);
    const tc_parsed = try std.json.parseFromSlice(std.json.Value, a, tc, .{});
    defer tc_parsed.deinit();
    try std.testing.expect(tc_parsed.value.object.get("eos_token").?.string.len > 0);
    try std.testing.expect(tc_parsed.value.object.get("chat_template").?.string.len > 100);

    if (try generationConfigJson(a, &f)) |gc| {
        defer a.free(gc);
        const gc_parsed = try std.json.parseFromSlice(std.json.Value, a, gc, .{});
        defer gc_parsed.deinit();
        try std.testing.expect(gc_parsed.value.object.get("do_sample").?.bool);
    }
}
