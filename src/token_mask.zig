//! Token-level mask for grammar-constrained sampling.
//!
//! Given a `Grammar` (current state) and a `TokenBytes` table (decoded bytes for
//! every vocab id), `buildMask` simulates feeding each token's bytes through the
//! grammar and records which tokens would be accepted. The result is a
//! `[vocab_size]bool` slice the sampler uses to mask invalid logits.
//!
//! Performance
//! -----------
//! Naive cost: O(vocab × avg_token_bytes). For a 100k-vocab byte-level BPE with
//! ~3 bytes per token, that's ~300k state transitions per generated LLM token.
//! On Apple Silicon, expect ~1–5 ms overhead per token. Future work: cache masks
//! keyed on a hashed grammar state.

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
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *TokenBytes) void {
        self.arena.deinit();
    }
};

pub const BuildError = error{OutOfMemory};

/// Precompute the byte sequence for every token id in `tokenizer`.
///
/// For byte-level BPE, this reverses the GPT-2 byte→unicode mapping. For
/// SentencePiece, this swaps `▁` (U+2581) for spaces. Special tokens
/// (BOS/EOS/PAD/chat-template tags) decode to `null`.
pub fn build(gpa: std.mem.Allocator, tokenizer: *const Tokenizer) BuildError!TokenBytes {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
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

    return .{
        .bytes = bytes,
        .eos_id = tokenizer.eos_id,
        .arena = arena,
    };
}

fn decodeSingle(arena: std.mem.Allocator, tokenizer: *const Tokenizer, id: u32) ![]const u8 {
    const ids: [1]u32 = .{id};
    // The tokenizer's existing decode handles all three tokenizer types,
    // including byte-level unicode→byte reversal and SentencePiece ▁→space.
    return try tokenizer.decode(arena, &ids, false);
}

/// Build the mask of allowed token ids given the grammar's current state.
///
/// `mask` must be `vocab_size` long; written entries are `true` (allowed) or
/// `false` (forbidden). Returns the count of allowed tokens.
///
/// Special handling:
///   * EOS token is allowed iff `grammar.isComplete()`.
///   * If `grammar.isDead()`, the entire vocab is permitted (graceful fallback,
///     matching the user-selected `ignore_mask` policy).
pub fn buildMask(
    grammar: *Grammar,
    token_bytes: *const TokenBytes,
    mask: []bool,
) std.mem.Allocator.Error!u32 {
    std.debug.assert(mask.len == token_bytes.bytes.len);

    if (grammar.isDead()) {
        @memset(mask, true);
        return @intCast(mask.len);
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

    for (token_bytes.bytes, 0..) |maybe_bytes, id| {
        const bytes = maybe_bytes orelse continue;
        if (bytes.len == 0) continue;
        if (token_bytes.eos_id) |eos| if (id == eos) continue;

        try grammar.restoreFrom(snap);

        var ok = true;
        for (bytes) |b| {
            if (!try grammar.acceptByteFast(b)) {
                ok = false;
                break;
            }
        }
        if (ok) {
            mask[id] = true;
            count += 1;
        }
    }

    try grammar.restoreFrom(snap);
    return count;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

const testing = std.testing;
const schema_mod = @import("json_schema.zig");

/// Minimal fake `TokenBytes` builder for unit tests — no real tokenizer needed.
fn fakeTokenBytes(arena: *std.heap.ArenaAllocator, tokens: []const ?[]const u8, eos_id: ?u32) TokenBytes {
    return .{
        .bytes = tokens,
        .eos_id = eos_id,
        .arena = arena.*, // moved
    };
}

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
    defer arena.deinit();
    const a = arena.allocator();

    // Vocab: 0={, 1=[, 2="hello", 3=" ", 4=EOS
    const vocab: [5]?[]const u8 = .{
        try a.dupe(u8, "{"),
        try a.dupe(u8, "["),
        try a.dupe(u8, "\"hello\""),
        try a.dupe(u8, " "),
        null, // EOS
    };
    const tb: TokenBytes = .{
        .bytes = &vocab,
        .eos_id = 4,
        .arena = std.heap.ArenaAllocator.init(testing.allocator), // dummy
    };
    var dummy = tb;
    defer dummy.arena.deinit();

    var mask: [5]bool = undefined;
    const count = try buildMask(&g, &tb, &mask);

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
    defer arena.deinit();
    const a = arena.allocator();

    const vocab: [3]?[]const u8 = .{
        try a.dupe(u8, "x"),
        try a.dupe(u8, " "), // whitespace ok after value
        null, // EOS
    };
    const tb: TokenBytes = .{
        .bytes = &vocab,
        .eos_id = 2,
        .arena = std.heap.ArenaAllocator.init(testing.allocator),
    };
    var dummy = tb;
    defer dummy.arena.deinit();

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
    defer arena.deinit();
    const a = arena.allocator();

    const vocab: [3]?[]const u8 = .{
        try a.dupe(u8, "\"name"), // valid: starts the only key
        try a.dupe(u8, "\"foo"), // invalid: `f` doesn't start any property
        try a.dupe(u8, "}"), // invalid: required `name` not yet seen
    };
    const tb: TokenBytes = .{
        .bytes = &vocab,
        .eos_id = null,
        .arena = std.heap.ArenaAllocator.init(testing.allocator),
    };
    var dummy = tb;
    defer dummy.arena.deinit();

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
    defer arena.deinit();
    const a = arena.allocator();

    const vocab: [3]?[]const u8 = .{
        try a.dupe(u8, "anything"),
        try a.dupe(u8, "[]"),
        try a.dupe(u8, "garbage"),
    };
    const tb: TokenBytes = .{
        .bytes = &vocab,
        .eos_id = null,
        .arena = std.heap.ArenaAllocator.init(testing.allocator),
    };
    var dummy = tb;
    defer dummy.arena.deinit();

    var mask: [3]bool = undefined;
    const count = try buildMask(&g, &tb, &mask);

    try testing.expect(mask[0]);
    try testing.expect(mask[1]);
    try testing.expect(mask[2]);
    try testing.expectEqual(@as(u32, 3), count);
}
