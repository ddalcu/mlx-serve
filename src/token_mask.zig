//! Token-level mask for grammar-constrained sampling.
//!
//! Given a `Grammar` (current state) and a `TokenBytes` table (decoded bytes for
//! every vocab id), `buildMask` simulates feeding each token's bytes through the
//! grammar and records which tokens would be accepted. The result is a
//! `[vocab_size]bool` slice the sampler uses to mask invalid logits.
//!
//! Performance
//! -----------
//! A step probes only tokens whose first two bytes the grammar can accept
//! (`by_first` + `second_off` under `allowedBytes`), and inside a string body
//! every quote-free token is admitted without a walk. The naive full-vocab
//! walk cost up to a second per generated token on a 248k vocabulary.

const std = @import("std");
const grammar_mod = @import("json_grammar.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;

pub const Grammar = grammar_mod.Grammar;

/// Decoded byte sequence for every token id in a tokenizer's vocabulary.
/// Built once per server load and reused across all requests.
pub const TokenBytes = struct {
    /// `bytes[id]` is the byte sequence the model would emit when sampling token
    /// `id`. `null` for tokens that have no plain-byte interpretation (special
    /// tokens like BOS/EOS, padding, and chat template tags). The grammar should
    /// never sample a `null`-bytes token while constrained — except for EOS,
    /// which is allowed only when the grammar is complete.
    bytes: []const ?[]const u8,
    /// EOS token id, if known. Allowed only when the grammar's root value is fully parsed.
    eos_id: ?u32,
    /// Probe-able token ids (non-null, non-empty, not EOS) grouped by first
    /// byte, each bucket ordered by `secondClass`.
    by_first: [256][]const u32,
    /// `by_first[b][second_off[b][k]..second_off[b][k+1]]` is the run of
    /// second-byte class `k` inside bucket `b`.
    second_off: [256][SECOND_CLASSES + 1]u32,
    /// Probe-able tokens legal anywhere in a JSON string body: no quote,
    /// backslash or control byte.
    string_plain: []const u32,
    /// The remaining probe-able tokens, walked individually inside a string.
    string_other: []const u32,
    arena: std.heap.ArenaAllocator,

    /// Index `bytes` (which must live in `arena`); takes ownership of the arena.
    pub fn init(arena: std.heap.ArenaAllocator, bytes: []const ?[]const u8, eos_id: ?u32) BuildError!TokenBytes {
        var self: TokenBytes = .{ .bytes = bytes, .eos_id = eos_id, .by_first = undefined, .second_off = undefined, .string_plain = &.{}, .string_other = &.{}, .arena = arena };
        errdefer self.arena.deinit();
        const a = self.arena.allocator();

        var counts: [256]u32 = @splat(0);
        var n_plain: usize = 0;
        var n_other: usize = 0;
        for (bytes, 0..) |maybe, id| {
            const b = probeable(maybe, id, eos_id) orelse continue;
            counts[b[0]] += 1;
            if (isStringPlain(b)) n_plain += 1 else n_other += 1;
        }
        var buckets: [256][]u32 = undefined;
        for (&buckets, counts) |*bucket, n| bucket.* = try a.alloc(u32, n);
        const plain = try a.alloc(u32, n_plain);
        const other = try a.alloc(u32, n_other);
        var fill: [256]usize = @splat(0);
        n_plain = 0;
        n_other = 0;
        for (bytes, 0..) |maybe, id| {
            const b = probeable(maybe, id, eos_id) orelse continue;
            const id32: u32 = @intCast(id);
            buckets[b[0]][fill[b[0]]] = id32;
            fill[b[0]] += 1;
            if (isStringPlain(b)) {
                plain[n_plain] = id32;
                n_plain += 1;
            } else {
                other[n_other] = id32;
                n_other += 1;
            }
        }
        for (buckets, 0..) |ids, b| {
            std.mem.sort(u32, ids, bytes, struct {
                fn lt(ctx: []const ?[]const u8, x: u32, y: u32) bool {
                    return secondClass(ctx[x].?) < secondClass(ctx[y].?);
                }
            }.lt);
            var i: u32 = 0;
            for (&self.second_off[b], 0..) |*off, k| {
                while (i < ids.len and secondClass(bytes[ids[i]].?) < k) i += 1;
                off.* = i;
            }
            self.by_first[b] = ids;
        }
        self.string_plain = plain;
        self.string_other = other;
        return self;
    }

    pub fn deinit(self: *TokenBytes) void {
        self.arena.deinit();
    }
};

/// Class 0 = single-byte token, 1 + c = second byte c.
const SECOND_CLASSES = 257;

fn probeable(maybe: ?[]const u8, id: usize, eos_id: ?u32) ?[]const u8 {
    const b = maybe orelse return null;
    if (b.len == 0 or (eos_id != null and id == eos_id.?)) return null;
    return b;
}

fn secondClass(b: []const u8) usize {
    return if (b.len == 1) 0 else 1 + @as(usize, b[1]);
}

fn isStringPlain(b: []const u8) bool {
    for (b) |c| if (c == '"' or c == '\\' or c < 0x20) return false;
    return true;
}

pub const BuildError = error{OutOfMemory};

/// Precompute the byte sequence for every token id in `tokenizer`.
///
/// For byte-level BPE, this reverses the GPT-2 byte→unicode mapping. For
/// SentencePiece, this swaps `▁` (U+2581) for spaces. Special tokens
/// (BOS/EOS/PAD/chat-template tags) decode to `null`.
pub fn build(gpa: std.mem.Allocator, tokenizer: *const Tokenizer) BuildError!TokenBytes {
    var arena = std.heap.ArenaAllocator.init(gpa);
    var arena_owned = true;
    errdefer if (arena_owned) arena.deinit();
    const a = arena.allocator();

    // Vocab size = max id + 1 (id_to_token may have gaps; treat missing ids as null).
    var max_id: u32 = 0;
    var it = tokenizer.id_to_token.iterator();
    while (it.next()) |entry| {
        if (entry.key_ptr.* > max_id) max_id = entry.key_ptr.*;
    }
    var sit = tokenizer.special_tokens.iterator();
    while (sit.next()) |entry| {
        if (entry.value_ptr.* > max_id) max_id = entry.value_ptr.*;
    }
    const vocab_size: usize = @as(usize, max_id) + 1;

    var bytes = try a.alloc(?[]const u8, vocab_size);
    @memset(bytes, null);

    // Build a set of special-token ids (always null bytes).
    var special_ids: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer special_ids.deinit(gpa);
    var sit2 = tokenizer.special_tokens.iterator();
    while (sit2.next()) |entry| {
        try special_ids.put(gpa, entry.value_ptr.*, {});
    }

    var tit = tokenizer.id_to_token.iterator();
    while (tit.next()) |entry| {
        const id = entry.key_ptr.*;
        if (special_ids.contains(id)) continue;
        const decoded = decodeSingle(a, tokenizer, id) catch continue;
        bytes[id] = decoded;
    }

    arena_owned = false;
    return TokenBytes.init(arena, bytes, tokenizer.eos_id);
}

fn decodeSingle(arena: std.mem.Allocator, tokenizer: *const Tokenizer, id: u32) ![]const u8 {
    const ids: [1]u32 = .{id};
    // The tokenizer's existing decode handles all three tokenizer types,
    // including byte-level unicode→byte reversal and SentencePiece ▁→space.
    return try tokenizer.decode(arena, &ids, false);
}

pub const MaskResult = struct { allowed: u32, probed: u32 };

/// Build the mask of allowed token ids given the grammar's current state.
///
/// `mask` must be `vocab_size` long; written entries are `true` (allowed) or
/// `false` (forbidden). Returns the allowed count and how many tokens were
/// walked through the grammar to get there.
///
/// Special handling:
///   * EOS token is allowed iff `grammar.isComplete()`.
///   * If `grammar.isDead()`, the entire vocab is permitted (graceful fallback,
///     matching the user-selected `ignore_mask` policy).
pub fn buildMask(
    grammar: *Grammar,
    token_bytes: *const TokenBytes,
    mask: []bool,
) std.mem.Allocator.Error!MaskResult {
    std.debug.assert(mask.len == token_bytes.bytes.len);

    if (grammar.isDead()) {
        @memset(mask, true);
        return .{ .allowed = @intCast(mask.len), .probed = 0 };
    }

    @memset(mask, false);
    var count: u32 = 0;

    if (token_bytes.eos_id) |eos| {
        if (eos < mask.len and grammar.isComplete()) {
            mask[eos] = true;
            count += 1;
        }
    }

    // Outer snapshot — the per-token loop restores from this each iteration so
    // we don't pay for per-byte snap/restore inside the grammar.
    const snap = try grammar.snapshot();
    defer grammar.discardSnapshot(snap);

    var probed: u32 = 0;
    if (grammar.stringBodyRoom()) |room| {
        for (token_bytes.string_plain) |id| {
            if (token_bytes.bytes[id].?.len <= room) {
                mask[id] = true;
                count += 1;
            }
        }
        for (token_bytes.string_other) |id| {
            probed += 1;
            if (try probeToken(grammar, snap, token_bytes.bytes[id].?)) {
                mask[id] = true;
                count += 1;
            }
        }
    } else {
        // Two-level trie walk: a token is legal only if its first byte is,
        // and then only if its second byte is legal in the state the first
        // byte leaves behind. Single-byte tokens are decided by the first
        // level alone.
        const first = try grammar.allowedBytes();
        for (token_bytes.by_first, 0..) |bucket, b| {
            if (!first.contains(@intCast(b))) continue;
            const off = token_bytes.second_off[b];
            for (bucket[off[0]..off[1]]) |id| {
                mask[id] = true;
                count += 1;
            }
            try grammar.restoreFrom(snap);
            _ = try grammar.acceptByteFast(@intCast(b));
            const second = try grammar.allowedBytes();
            for (1..SECOND_CLASSES) |k| {
                if (!second.contains(@intCast(k - 1))) continue;
                for (bucket[off[k]..off[k + 1]]) |id| {
                    probed += 1;
                    if (try probeToken(grammar, snap, token_bytes.bytes[id].?)) {
                        mask[id] = true;
                        count += 1;
                    }
                }
            }
        }
    }

    try grammar.restoreFrom(snap);
    return .{ .allowed = count, .probed = probed };
}

fn probeToken(grammar: *Grammar, snap: Grammar.Snapshot, bytes: []const u8) !bool {
    try grammar.restoreFrom(snap);
    for (bytes) |b| {
        if (!try grammar.acceptByteFast(b)) return false;
    }
    return true;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

const testing = std.testing;
const schema_mod = @import("json_schema.zig");

fn parseSchema(gpa: std.mem.Allocator, src: []const u8) !grammar_mod.Schema {
    const v = try std.json.parseFromSlice(std.json.Value, gpa, src, .{});
    defer v.deinit();
    return schema_mod.parse(gpa, v.value);
}

test "buildMask: only valid first bytes are allowed for object schema" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    const a = arena.allocator();

    // Vocab: 0={, 1=[, 2="hello", 3=" ", 4=EOS
    const vocab: [5]?[]const u8 = .{
        try a.dupe(u8, "{"),
        try a.dupe(u8, "["),
        try a.dupe(u8, "\"hello\""),
        try a.dupe(u8, " "),
        null, // EOS
    };
    var tb = try TokenBytes.init(arena, &vocab, 4);
    defer tb.deinit();

    var mask: [5]bool = undefined;
    const count = (try buildMask(&g, &tb, &mask)).allowed;

    try testing.expect(mask[0]); // `{` is the start of the object
    try testing.expect(!mask[1]); // `[` not allowed
    try testing.expect(!mask[2]); // `"hello"` not allowed (string, but root is object)
    try testing.expect(mask[3]); // whitespace allowed before value
    try testing.expect(!mask[4]); // EOS not allowed; grammar is incomplete
    try testing.expectEqual(@as(u32, 2), count);
}

test "buildMask: EOS is allowed when grammar is complete" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"boolean\"}");
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    _ = try g.acceptByte('t');
    _ = try g.acceptByte('r');
    _ = try g.acceptByte('u');
    _ = try g.acceptByte('e');
    try testing.expect(g.isComplete());

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    const a = arena.allocator();

    const vocab: [3]?[]const u8 = .{
        try a.dupe(u8, "x"),
        try a.dupe(u8, " "), // whitespace ok after value
        null, // EOS
    };
    var tb = try TokenBytes.init(arena, &vocab, 2);
    defer tb.deinit();

    var mask: [3]bool = undefined;
    _ = try buildMask(&g, &tb, &mask);

    try testing.expect(!mask[0]); // garbage byte rejected
    try testing.expect(mask[1]); // trailing whitespace allowed
    try testing.expect(mask[2]); // EOS allowed because root is accepted
}

test "buildMask: rejects multi-byte tokens that violate schema" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    _ = try g.acceptByte('{');

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    const a = arena.allocator();

    const vocab: [3]?[]const u8 = .{
        try a.dupe(u8, "\"name"), // valid: starts the only key
        try a.dupe(u8, "\"foo"), // invalid: `f` doesn't start any property
        try a.dupe(u8, "}"), // invalid: required `name` not yet seen
    };
    var tb = try TokenBytes.init(arena, &vocab, null);
    defer tb.deinit();

    var mask: [3]bool = undefined;
    _ = try buildMask(&g, &tb, &mask);

    try testing.expect(mask[0]);
    try testing.expect(!mask[1]);
    try testing.expect(!mask[2]);
}

test "buildMask: dead grammar permits everything (graceful fallback)" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"boolean\"}");
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    // Force a dead state by feeding garbage that doesn't match.
    g.dead = true;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    const a = arena.allocator();

    const vocab: [3]?[]const u8 = .{
        try a.dupe(u8, "anything"),
        try a.dupe(u8, "[]"),
        try a.dupe(u8, "garbage"),
    };
    var tb = try TokenBytes.init(arena, &vocab, null);
    defer tb.deinit();

    var mask: [3]bool = undefined;
    const count = (try buildMask(&g, &tb, &mask)).allowed;

    try testing.expect(mask[0]);
    try testing.expect(mask[1]);
    try testing.expect(mask[2]);
    try testing.expectEqual(@as(u32, 3), count);
}

/// Reference: the naive full-vocabulary walk the fast path must agree with.
fn bruteMask(grammar: *Grammar, tb: *const TokenBytes, mask: []bool) !void {
    @memset(mask, false);
    if (grammar.isDead()) {
        @memset(mask, true);
        return;
    }
    if (tb.eos_id) |eos| mask[eos] = grammar.isComplete();
    const snap = try grammar.snapshot();
    defer grammar.discardSnapshot(snap);
    for (tb.bytes, 0..) |maybe, id| {
        const bytes = maybe orelse continue;
        if (bytes.len == 0) continue;
        if (tb.eos_id) |eos| if (id == eos) continue;
        try grammar.restoreFrom(snap);
        var ok = true;
        for (bytes) |b| {
            if (!try grammar.acceptByteFast(b)) {
                ok = false;
                break;
            }
        }
        mask[id] = ok;
    }
    try grammar.restoreFrom(snap);
}

/// A vocabulary shaped like a byte-level BPE: every single byte, JSON
/// fragments, escapes, and deterministic multi-byte junk.
fn syntheticVocab(a: std.mem.Allocator) ![]?[]const u8 {
    var list: std.ArrayList(?[]const u8) = .empty;
    for (0..256) |b| try list.append(a, try a.dupe(u8, &[_]u8{@intCast(b)}));
    const frags = [_][]const u8{
        "{\"",     "\"name\"", "\":",     "\": \"", "\", \"", "\"}",    "}",  "],",  "[\"",  "\\\"", "\\n",  "\\u00", "12",  "3.5", "-7",
        "true",    "false",    "null",    "ab",     "hello",  " world",
        "ción",
        "日本",
        "\"age\"", "\"tags\"", "\"red\"", "red",    "blue",   "e+3",    "0.", ",\"", "\n  ", "a\"b", "\"\"", "\t",    "x\\",
    };
    for (frags) |f| try list.append(a, try a.dupe(u8, f));
    var seed: u32 = 0x9e3779b9;
    for (0..1500) |_| {
        seed = seed *% 1664525 +% 1013904223;
        const len = 2 + (seed >> 28) % 4;
        const buf = try a.alloc(u8, len);
        for (buf) |*c| {
            seed = seed *% 1664525 +% 1013904223;
            c.* = @intCast(0x20 + (seed >> 24) % 0x5f);
        }
        try list.append(a, buf);
    }
    try list.append(a, null); // EOS
    return list.toOwnedSlice(a);
}

test "buildMask: the fast path agrees with the full walk at every byte of a document" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string","maxLength":12},"age":{"type":"integer"},
        \\ "tags":{"type":"array","items":{"type":"string"}},"color":{"enum":["red","blue"]}},
        \\ "required":["name","age","tags","color"],"additionalProperties":false}
    );
    defer schema.deinit();
    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    const vocab = try syntheticVocab(arena.allocator());
    var tb = try TokenBytes.init(arena, vocab, @intCast(vocab.len - 1));
    defer tb.deinit();

    const fast = try testing.allocator.alloc(bool, vocab.len);
    defer testing.allocator.free(fast);
    const brute = try testing.allocator.alloc(bool, vocab.len);
    defer testing.allocator.free(brute);

    const doc = "{\"name\": \"a\\\"b \\u00e9\", \"age\": -12, \"tags\": [\"x\", \"日本\"], \"color\": \"blue\"} ";
    for (doc, 0..) |byte, i| {
        _ = try buildMask(&g, &tb, fast);
        try bruteMask(&g, &tb, brute);
        for (fast, brute, 0..) |f, b, id| {
            if (f != b) {
                std.debug.print("mismatch at doc[{d}] token {d} {s}: fast={} brute={}\n", .{ i, id, vocab[id].?, f, b });
                return error.MaskMismatch;
            }
        }
        try testing.expect(try g.acceptByte(byte));
    }
    try testing.expect(g.isComplete());
}

test "buildMask: a step walks only tokens that can start in the current state" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}
    );
    defer schema.deinit();
    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    const vocab = try syntheticVocab(arena.allocator());
    var tb = try TokenBytes.init(arena, vocab, @intCast(vocab.len - 1));
    defer tb.deinit();
    const mask = try testing.allocator.alloc(bool, vocab.len);
    defer testing.allocator.free(mask);

    // Structural state: only `{` and whitespace can start a token.
    var r = try buildMask(&g, &tb, mask);
    try testing.expect(r.probed < 64);
    // String body: the quote-free majority is admitted without a walk.
    for ("{\"name\": \"he") |b| try testing.expect(try g.acceptByte(b));
    r = try buildMask(&g, &tb, mask);
    try testing.expect(r.probed <= tb.string_other.len);
    try testing.expect(r.allowed > tb.string_plain.len);
}
