//! Reasoning-protocol state machine for grammar-constrained generation.
//!
//! A schema-constrained request may not start sampling JSON immediately: the
//! chat protocol can place generation inside (or in front of) a reasoning
//! segment first. This module is the ONE bounded recognizer shared by every
//! reasoning format the chat parser understands — per format it resolves the
//! opener (when the prompt did not already commit one), the close delimiter,
//! and the canonical forced-recovery sequence, and it tracks per-request
//! phase state:
//!
//!   choice    — the model may open reasoning or answer directly; the mask is
//!               the UNION of opener-prefix candidates and schema-legal JSON.
//!   reasoning — body text is unconstrained; a bounded incremental matcher
//!               watches for the close delimiter.
//!   json_body — the existing JSON grammar masks every token (unchanged).
//!
//! Direct answers are constrained from their FIRST byte: while the channel is
//! unresolved only opener prefixes and JSON-legal tokens are admissible, and
//! once opener bytes are in flight the opener must complete. A close
//! delimiter may share a token with the first payload bytes; the recognizer
//! returns that payload suffix so the caller can feed the grammar (an invalid
//! suffix degrades the same way an invalid sampled token does elsewhere).
//!
//! Model-level token candidate indexes live on `LoadedModel` (built once per
//! model); everything here is per-request with fixed-size buffers. `Protocol`
//! borrows its own internal arrays, so it must not be moved after resolution
//! (the same lifetime rule as `SchemaConstraint`).

const std = @import("std");
const json_grammar = @import("json_grammar.zig");
const token_mask = @import("token_mask.zig");

/// Longest reasoning delimiter this module recognizes (the 19-byte
/// `</think:opensource>` shape plus slack).
pub const MAX_MARKER_BYTES = 32;
/// Canonical token encoding of the longest delimiter.
pub const MAX_FORCED_TOKENS = MAX_MARKER_BYTES;
/// A full recovery transition: opener remainder + close delimiter.
pub const MAX_TRANSITION_TOKENS = 2 * MAX_FORCED_TOKENS;
/// Flat storage for one delimiter's per-suffix canonical encodings (the
/// remainder after each possible partial match). Worst case every suffix
/// needs the full token budget.
pub const SUFFIX_TABLE_BYTES = MAX_MARKER_BYTES * MAX_FORCED_TOKENS;

pub const SuffixRun = struct { offset: u32, len: u8 };

/// Where the constrained JSON payload begins in the generated stream, for
/// response routing: `token_index` is the index of the generated token that
/// carries the first payload byte, `byte_offset` the offset within that
/// token's decoded text. Generation is authoritative — response emitters use
/// this instead of re-parsing marker text that may legitimately appear inside
/// JSON string data.
pub const ConstraintSpan = struct { token_index: u32, byte_offset: u32 };

/// Byte offset of the payload start within the token that carried it: the
/// token's own byte length minus the payload suffix it carried. Special
/// tokens have no bytes — the caller passes null and the payload begins
/// AFTER the token (the caller expresses that in the token index).
pub fn payloadByteOffset(token_bytes: ?[]const u8, suffix: []const u8) u32 {
    const b = token_bytes orelse return 0;
    std.debug.assert(suffix.len <= b.len);
    return @intCast(b.len - suffix.len);
}

/// Whether `bytes` can extend or complete `opener` from SOME cursor: the
/// production candidate predicate. A token qualifies when it is a prefix of
/// some opener suffix (continuation) or some opener suffix is a prefix of it
/// (completion, possibly with payload). Shared by the model-level candidate
/// builder and the tests so the two can never drift.
pub fn openerCandidateBytesMatch(opener: []const u8, bytes: []const u8) bool {
    if (bytes.len == 0) return false;
    for (0..opener.len) |k| {
        const rest = opener[k..];
        if (std.mem.startsWith(u8, rest, bytes) or std.mem.startsWith(u8, bytes, rest)) return true;
    }
    return false;
}

fn isJsonWhitespaceRun(bytes: []const u8) bool {
    for (bytes) |c| {
        switch (c) {
            ' ', '\t', '\n', '\r' => {},
            else => return false,
        }
    }
    return true;
}

pub const Kind = enum {
    /// The delimited reasoning block Qwen/DeepSeek/LFM/Laguna share: an
    /// opening tag and a closing tag spelling the same base (see
    /// `chat.BARE_THINK_OPENER` / `BARE_THINK_CLOSER`).
    bare_think,
    /// `<think:S>...</think:S>` — the Hy3 suffixed family. The suffix is fixed
    /// by whichever opener resolved (prompt-injected) and the close must
    /// carry the SAME suffix; think tags with other suffixes are body text.
    suffixed_think,
};

pub const BARE_THINK_OPENER = "<think>";
pub const BARE_THINK_CLOSER = "</think>";

pub const Protocol = struct {
    kind: Kind,
    /// Special-token id whose text equals the opener, when the tokenizer has
    /// one. Special tokens keep their identity: `TokenBytes.bytes` is null
    /// for them by design, so they never ride the ordinary-byte candidates.
    opener_atomic: ?u32 = null,
    /// Atomic id for the close delimiter, when one exists.
    closer_atomic: ?u32 = null,
    /// Model-level candidate index (borrowed from `LoadedModel`): ordinary
    /// tokens whose bytes are a prefix of the opener or complete it. Only
    /// consulted in the choice phase.
    opener_candidates: []const u32 = &.{},

    opener_buf: [MAX_MARKER_BYTES]u8 = undefined,
    opener_len: u8 = 0,
    closer_buf: [MAX_MARKER_BYTES]u8 = undefined,
    closer_len: u8 = 0,
    forced_buf: [MAX_FORCED_TOKENS]u32 = undefined,
    forced_len: u8 = 0,
    /// Canonical remainder encodings per partial-match length: entry k spells
    /// `closer_buf[k..]` in ordinary bytes. `len == 0` marks an unencodable
    /// suffix — forced recovery from that partial state is refused (safe
    /// stop), never approximated.
    closer_suffix: [MAX_MARKER_BYTES]SuffixRun = @splat(.{ .offset = 0, .len = 0 }),
    closer_suffix_buf: [SUFFIX_TABLE_BYTES]u32 = undefined,
    /// Same for the opener (choice state): entry k spells `opener_buf[k..]`.
    opener_suffix: [MAX_MARKER_BYTES]SuffixRun = @splat(.{ .offset = 0, .len = 0 }),
    opener_suffix_buf: [SUFFIX_TABLE_BYTES]u32 = undefined,
    /// Model-level candidate set (borrowed from `LoadedModel`): ordinary
    /// tokens whose bytes CONTAIN the full close delimiter — the only tokens
    /// that can complete the close from match state 0. The reasoning-phase
    /// mask validates their payload suffixes before they can be sampled.
    closer_span_candidates: []const u32 = &.{},

    /// The opener literal; null when the prompt already opened the block and
    /// generation starts inside reasoning. Delimiters live in this struct's
    /// own buffers and are read through accessors, so the struct survives a
    /// copy (init must still run where the struct will live, or the caller
    /// must copy BEFORE the accessor-derived slices matter — everything here
    /// re-derives, so both are safe).
    pub fn openerText(self: *const Protocol) ?[]const u8 {
        if (self.opener_len == 0) return null;
        return self.opener_buf[0..self.opener_len];
    }

    pub fn closerText(self: *const Protocol) []const u8 {
        return self.closer_buf[0..self.closer_len];
    }

    pub fn forcedIds(self: *const Protocol) []const u32 {
        return self.forced_buf[0..self.forced_len];
    }

    /// Store `text` into `buf` and return the slice. Caller-bounded: text
    /// longer than the buffer cannot be a supported delimiter.
    fn storeMarker(buf: *[MAX_MARKER_BYTES]u8, len: *u8, text: []const u8) bool {
        if (text.len == 0 or text.len > MAX_MARKER_BYTES) return false;
        @memcpy(buf[0..text.len], text);
        len.* = @intCast(text.len);
        return true;
    }

    pub fn setOpener(self: *Protocol, text: []const u8) bool {
        return storeMarker(&self.opener_buf, &self.opener_len, text);
    }

    pub fn setCloser(self: *Protocol, text: []const u8) bool {
        return storeMarker(&self.closer_buf, &self.closer_len, text);
    }

    pub fn setForced(self: *Protocol, ids: []const u32) bool {
        if (ids.len == 0 or ids.len > MAX_FORCED_TOKENS) return false;
        @memcpy(self.forced_buf[0..ids.len], ids);
        self.forced_len = @intCast(ids.len);
        return true;
    }

    /// Tokens the forced transition still needs: the whole canonical
    /// sequence, or nothing when the close rides its atomic id. One entry per
    /// scheduler tick while draining.
    pub fn recoveryRemaining(self: *const Protocol) []const u32 {
        if (self.closer_atomic != null) return &.{};
        return self.forcedIds();
    }

    pub fn recoveryTokenCount(self: *const Protocol) usize {
        if (self.closer_atomic != null) return 1;
        return self.forced_len;
    }

    /// Compose the whole remaining forced transition into `out`: the opener
    /// remainder when choice-state opener bytes are in flight, then the close
    /// delimiter — its byte remainder after a partial match, else the atomic
    /// id, else the full canonical sequence. Returns the composed token count,
    /// or null when a needed suffix is unencodable (recovery is refused;
    /// delimiter bytes are never approximated).
    pub fn planRecovery(self: *const Protocol, state: *const State, out: *[MAX_TRANSITION_TOKENS]u32) ?usize {
        var n: usize = 0;
        if (state.phase == .choice and state.open_cursor > 0) {
            const cursor: usize = state.open_cursor;
            if (cursor < self.opener_len) {
                const run = self.opener_suffix[cursor];
                if (run.len == 0) return null;
                if (n + run.len > out.len) return null;
                @memcpy(out[n..][0..run.len], self.opener_suffix_buf[run.offset..][0..run.len]);
                n += run.len;
            }
        }
        if (state.close_match > 0 and state.close_match < self.closer_len) {
            const run = self.closer_suffix[state.close_match];
            if (run.len == 0) return null;
            if (n + run.len > out.len) return null;
            @memcpy(out[n..][0..run.len], self.closer_suffix_buf[run.offset..][0..run.len]);
            n += run.len;
        } else if (self.closer_atomic) |aid| {
            if (n + 1 > out.len) return null;
            out[n] = aid;
            n += 1;
        } else {
            const full = self.forcedIds();
            if (n + full.len > out.len) return null;
            @memcpy(out[n..][0..full.len], full);
            n += full.len;
        }
        return n;
    }
};

pub const Phase = enum { choice, reasoning, json_body };

/// Mutable per-request state. Bounded: memory does not grow with reasoning
/// length (the close matcher keeps only the last `MAX_MARKER_BYTES` bytes).
pub const State = struct {
    phase: Phase = .json_body,
    /// Bytes of the opener matched so far (choice phase).
    open_cursor: u8 = 0,
    /// Trailing bytes currently matching a prefix of the closer (reasoning).
    close_match: u8 = 0,
    /// Last `MAX_MARKER_BYTES` generated bytes — what close-match resets
    /// re-scan, so no full-text rescans ever happen.
    tail: [MAX_MARKER_BYTES]u8 = undefined,
    tail_len: u8 = 0,
    /// A forced transition (loop/EOS/padding recovery) is being drained: one
    /// canonical token per scheduler tick through the live forward path. The
    /// composed sequence (opener remainder + close delimiter) is planned once
    /// and consumed by cursor.
    recovering: bool = false,
    forced_cursor: u8 = 0,
    pending_len: u8 = 0,
    pending: [MAX_TRANSITION_TOKENS]u32 = undefined,

    pub fn initPromptOpened() State {
        return .{ .phase = .reasoning };
    }

    pub fn initChoice() State {
        return .{ .phase = .choice };
    }

    fn pushTail(self: *State, byte: u8) void {
        if (self.tail_len < MAX_MARKER_BYTES) {
            self.tail[self.tail_len] = byte;
            self.tail_len += 1;
        } else {
            std.mem.copyForwards(u8, self.tail[0 .. MAX_MARKER_BYTES - 1], self.tail[1..]);
            self.tail[MAX_MARKER_BYTES - 1] = byte;
        }
    }
};

// ── Choice-phase opener predicates ───────────────────────────────────────────

fn openerRemainingText(p: *const Protocol, cursor: u8) []const u8 {
    const o = p.openerText() orelse return &.{};
    if (cursor >= o.len) return &.{};
    return o[cursor..];
}

/// True when `bytes` is a STRICT prefix continuation of the opener from
/// `cursor` (the opener is still incomplete after them).
pub fn openerContinuesFrom(p: *const Protocol, cursor: u8, bytes: []const u8) bool {
    const rest = openerRemainingText(p, cursor);
    return bytes.len < rest.len and std.mem.startsWith(u8, rest, bytes);
}

/// True when `bytes` completes the opener from `cursor`. Bytes beyond the
/// opener are the segment that follows (reasoning); the caller routes by
/// phase, so no payload validation happens here.
pub fn openerCompletedBy(p: *const Protocol, cursor: u8, bytes: []const u8) bool {
    const rest = openerRemainingText(p, cursor);
    return rest.len > 0 and bytes.len >= rest.len and std.mem.startsWith(u8, bytes, rest);
}

/// One sampled token's effect on the choice phase.
pub const ChoiceOutcome = union(enum) {
    /// Token continued the opener; still ambiguous.
    progress,
    /// Token completed the opener; everything after it is reasoning.
    opened,
    /// Token is ordinary JSON: ALL its bytes are grammar input.
    json,
};

pub fn observeChoice(p: *const Protocol, state: *State, token_id: u32, bytes: ?[]const u8) ChoiceOutcome {
    if (p.opener_atomic) |aid| {
        if (token_id == aid) {
            state.phase = .reasoning;
            return .opened;
        }
    }
    const b = bytes orelse {
        state.phase = .json_body;
        return .json;
    };
    // JSON-legal whitespace does not resolve the choice: an opener or a JSON
    // value can both follow. Stay ambiguous (the cursor is untouched — the
    // opener match is byte-exact) and keep the grammar unfed.
    if (isJsonWhitespaceRun(b)) return .progress;
    if (openerCompletedBy(p, state.open_cursor, b)) {
        state.phase = .reasoning;
        return .opened;
    }
    if (openerContinuesFrom(p, state.open_cursor, b)) {
        state.open_cursor += @intCast(b.len);
        return .progress;
    }
    state.phase = .json_body;
    return .json;
}

/// Augment the base JSON mask with the choice-phase opener candidates.
///
/// At cursor 0 the mask is the union: schema-legal tokens (the direct answer
/// is constrained from its first byte) plus opener prefixes/completions. Once
/// opener bytes are in flight the opener must complete — only opener
/// continuations remain, and the JSON grammar never consumes protocol bytes.
/// Returns the allowed count.
pub fn applyChoiceMask(
    p: *const Protocol,
    state: *State,
    grammar: *json_grammar.Grammar,
    tb: *const token_mask.TokenBytes,
    mask: []bool,
) std.mem.Allocator.Error!u32 {
    var allowed: u32 = 0;
    if (state.open_cursor == 0) {
        const base = try token_mask.buildMask(grammar, tb, mask);
        allowed = base.allowed;
        if (p.opener_atomic) |aid| {
            if (aid < mask.len and !mask[aid]) {
                mask[aid] = true;
                allowed += 1;
            }
        }
    } else {
        @memset(mask, false);
    }
    const closer = p.closerText();
    for (p.opener_candidates) |id| {
        if (id >= mask.len or mask[id]) continue;
        const bytes = tb.bytes[id] orelse continue;
        if (tb.eos_id) |eos| {
            if (id == eos) continue;
        }
        // A candidate whose bytes contain the close delimiter would emit the
        // BOUNDARY as opener progress; it is never a legal choice token.
        if (std.mem.indexOf(u8, bytes, closer) != null) continue;
        const cursor = state.open_cursor;
        if (openerContinuesFrom(p, cursor, bytes) or openerCompletedBy(p, cursor, bytes)) {
            mask[id] = true;
            allowed += 1;
        }
    }
    return allowed;
}

// ── Reasoning-phase close recognition ────────────────────────────────────────

/// The ONE close-delimiter transition mechanism: per-byte matcher stepping
/// shared by post-sampling observation and the pre-sample candidate probe,
/// so recognition and masking can never drift apart.
const CloseSim = struct {
    closer: []const u8,
    match: u8,
    tail: [MAX_MARKER_BYTES]u8 = undefined,
    tail_len: u8 = 0,

    fn init(closer: []const u8, match: u8, tail_src: []const u8) CloseSim {
        var sim = CloseSim{ .closer = closer, .match = match };
        const n = @min(tail_src.len, MAX_MARKER_BYTES);
        @memcpy(sim.tail[0..n], tail_src[tail_src.len - n ..]);
        sim.tail_len = @intCast(n);
        return sim;
    }

    /// Feed one byte; true when the closer just completed.
    fn feed(self: *CloseSim, c: u8) bool {
        if (self.tail_len < MAX_MARKER_BYTES) {
            self.tail[self.tail_len] = c;
            self.tail_len += 1;
        } else {
            std.mem.copyForwards(u8, self.tail[0 .. MAX_MARKER_BYTES - 1], self.tail[1..]);
            self.tail[MAX_MARKER_BYTES - 1] = c;
        }
        if (c == self.closer[self.match]) {
            self.match += 1;
            if (self.match == self.closer.len) return true;
        } else {
            self.match = longestClosePrefix(self.tail[0..self.tail_len], self.closer);
        }
        return false;
    }
};

/// Longest length k such that the last k bytes of `text` equal a prefix of
/// `closer`. Bounded by the delimiter size; this is what lets the matcher
/// recover when reasoning prose contains a partial delimiter.
fn longestClosePrefix(tail: []const u8, closer: []const u8) u8 {
    var k: usize = @min(tail.len, closer.len -| 1);
    while (k > 0) : (k -= 1) {
        if (std.mem.eql(u8, tail[tail.len - k ..], closer[0..k])) return @intCast(k);
    }
    return 0;
}

/// Where a close completion would land inside `bytes` given the current
/// match state, WITHOUT mutating tracked state. One past the closer's last
/// byte, or null. Pure — this is the probe the reasoning-phase mask runs
/// over boundary-sensitive candidates before they can be sampled.
pub fn probeCloseCompletion(p: *const Protocol, state: *const State, bytes: []const u8) ?usize {
    var sim = CloseSim.init(p.closerText(), state.close_match, state.tail[0..state.tail_len]);
    for (bytes, 0..) |c, i| {
        if (sim.feed(c)) return i + 1;
    }
    return null;
}

/// Feed one reasoning token through the close-delimiter matcher.
/// Returns the payload suffix carried by this token when the close completed
/// inside it (may be empty), else null.
pub fn observeReasoningToken(p: *const Protocol, state: *State, token_id: u32, bytes: ?[]const u8) ?[]const u8 {
    if (p.closer_atomic) |aid| {
        // An atomic close has no bytes: the payload it carries is empty by
        // definition, whatever the tokenizer's decode says.
        if (token_id == aid) return "";
    }
    const b = bytes orelse return null;
    var sim = CloseSim.init(p.closerText(), state.close_match, state.tail[0..state.tail_len]);
    for (b, 0..) |c, i| {
        if (sim.feed(c)) {
            state.close_match = 0;
            return b[i + 1 ..];
        }
    }
    state.close_match = sim.match;
    @memcpy(state.tail[0..sim.tail_len], sim.tail[0..sim.tail_len]);
    state.tail_len = sim.tail_len;
    return null;
}

/// Upper bound on tokens masked out in one reasoning step (the candidate
/// sets are tiny; the cap only bounds a pathological vocabulary).
const MAX_MASKED_OUT = 256;

/// Reasoning-phase pre-sample validation: among the boundary-sensitive
/// candidates — tokens containing the full close delimiter, plus, while a
/// partial match is open, tokens beginning with the remaining delimiter
/// bytes — those whose payload suffix the grammar REJECTS are masked out
/// BEFORE sampling, so a close-sharing token can never bypass the schema by
/// degrading enforcement afterwards. Returns the number of masked-out ids;
/// 0 means nothing was excluded and the caller may sample unconstrained.
pub fn applyReasoningMask(
    p: *const Protocol,
    state: *const State,
    grammar: *json_grammar.Grammar,
    tb: *const token_mask.TokenBytes,
    mask: []bool,
) std.mem.Allocator.Error!usize {
    if (grammar.isDead()) return 0;
    std.debug.assert(mask.len == tb.bytes.len);
    var invalid: [MAX_MASKED_OUT]u32 = undefined;
    var n: usize = 0;

    const snap = try grammar.snapshot();
    defer grammar.discardSnapshot(snap);

    if (p.closer_span_candidates.len > 0) {
        for (p.closer_span_candidates) |id| {
            if (id >= mask.len) continue;
            const bytes = tb.bytes[id] orelse continue;
            if (tb.eos_id) |eos| {
                if (id == eos) continue;
            }
            if (probeCloseCompletion(p, state, bytes)) |end| {
                const suffix = bytes[end..];
                try grammar.restoreFrom(snap);
                var ok = true;
                for (suffix) |c| {
                    if (!try grammar.acceptByteFast(c)) {
                        ok = false;
                        break;
                    }
                }
                if (!ok and n < invalid.len) {
                    invalid[n] = id;
                    n += 1;
                }
            }
        }
    }
    if (state.close_match > 0 and state.close_match < p.closer_len) {
        const rest = p.closerText()[state.close_match..];
        // Any token beginning with the remaining delimiter bytes completes
        // the close at its first bytes; only a schema-valid suffix may ride
        // along. The by_first bucket bounds the scan to plausible starters.
        for (tb.by_first[rest[0]]) |id| {
            if (id >= mask.len) continue;
            const bytes = tb.bytes[id] orelse continue;
            if (!std.mem.startsWith(u8, bytes, rest)) continue;
            const suffix = bytes[rest.len..];
            try grammar.restoreFrom(snap);
            var ok = true;
            for (suffix) |c| {
                if (!try grammar.acceptByteFast(c)) {
                    ok = false;
                    break;
                }
            }
            if (!ok and n < invalid.len) {
                invalid[n] = id;
                n += 1;
            }
        }
    }
    try grammar.restoreFrom(snap);
    if (n == 0) return 0;
    @memset(mask, true);
    for (invalid[0..n]) |id| mask[id] = false;
    return n;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

const testing = std.testing;
const schema_mod = @import("json_schema.zig");

fn parseSchema(gpa: std.mem.Allocator, src: []const u8) !json_grammar.Schema {
    const v = try std.json.parseFromSlice(std.json.Value, gpa, src, .{});
    defer v.deinit();
    return schema_mod.parse(gpa, v.value);
}

/// Vocabulary shaped like a real BPE: opener fragments, JSON fragments,
/// junk, one atomic `</think>`, and EOS.
const Vocab = struct {
    const opener: u32 = 0; // "<think"
    const close_gt: u32 = 1; // ">"
    const close_full: u32 = 2; // "</think>"
    const close_atomic: u32 = 3; // special token `</think>`
    const json_ob: u32 = 4; // "{"
    const json_cb: u32 = 5; // "}"
    const json_ws: u32 = 6; // " "
    const prose: u32 = 7; // "hello"
    const lt: u32 = 8; // "<"
    const eos: u32 = 9;
    const span_close_json: u32 = 10; // close bytes followed by "{"
    const span_gt_reason: u32 = 11; // ">x": completes the opener plus reasoning
    const span_close_invalid: u32 = 12; // close bytes followed by "x" (schema-invalid)
    const ws_nl: u32 = 13; // "\n"
    const COUNT: u32 = 14;

    fn build(a: std.mem.Allocator) !token_mask.TokenBytes {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const al = arena.allocator();
        var list: std.ArrayList(?[]const u8) = .empty;
        const words = [_]?[]const u8{
            "<think", ">", BARE_THINK_CLOSER, null, "{", "}", " ", "hello", "<", null,
            BARE_THINK_CLOSER ++ "{", ">x", BARE_THINK_CLOSER ++ "x", "\n",
        };
        for (words) |w| try list.append(al, if (w) |s| try al.dupe(u8, s) else null);
        return token_mask.TokenBytes.init(arena, try list.toOwnedSlice(al), eos);
    }
};

fn makeProtocol(opener: ?[]const u8, closer: []const u8, candidates: []const u32) Protocol {
    var proto = Protocol{ .kind = .bare_think, .closer_atomic = 3, .opener_candidates = candidates };
    if (opener) |o| std.testing.expect(proto.setOpener(o)) catch unreachable;
    std.testing.expect(proto.setCloser(closer)) catch unreachable;
    return proto;
}

test "choice mask at cursor 0: opener candidates and JSON starts, nothing else" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}
    );
    defer schema.deinit();
    var g = try json_grammar.Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    var tb = try Vocab.build(testing.allocator);
    defer tb.deinit();

    var proto = makeProtocol(BARE_THINK_OPENER, BARE_THINK_CLOSER, &.{
        Vocab.opener, Vocab.lt, Vocab.close_gt, Vocab.span_gt_reason,
    });

    var state = State.initChoice();
    var mask: [Vocab.COUNT]bool = undefined;
    const allowed = try applyChoiceMask(&proto, &state, &g, &tb, &mask);

    try testing.expect(mask[Vocab.opener]); // "<think" starts the opener
    try testing.expect(mask[Vocab.lt]); // "<" is an opener prefix
    try testing.expect(mask[Vocab.json_ob]); // direct answer is JSON-legal
    try testing.expect(mask[Vocab.json_ws]); // JSON leading whitespace
    try testing.expect(!mask[Vocab.close_gt]); // ">" cannot START the opener
    try testing.expect(!mask[Vocab.span_gt_reason]); // ">x" is a completion, reachable only from cursor 6
    try testing.expect(!mask[Vocab.close_full]); // the CLOSE spelling never rides the choice mask
    try testing.expect(!mask[Vocab.close_atomic]); // the atomic CLOSER id is never an opener candidate
    try testing.expect(!mask[Vocab.prose]); // prose is neither
    try testing.expect(!mask[Vocab.eos]); // grammar incomplete → no EOS
    _ = allowed;
}

test "choice mask: once opener bytes are in flight only opener continuations remain" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"object\"}");
    defer schema.deinit();
    var g = try json_grammar.Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    var tb = try Vocab.build(testing.allocator);
    defer tb.deinit();

    var proto = makeProtocol(BARE_THINK_OPENER, BARE_THINK_CLOSER, &.{
        Vocab.opener, Vocab.lt, Vocab.close_gt, Vocab.span_gt_reason,
    });

    var state = State.initChoice();
    var mask: [Vocab.COUNT]bool = undefined;
    _ = try applyChoiceMask(&proto, &state, &g, &tb, &mask);

    // The model committed to "<think": JSON is no longer reachable.
    const out = observeChoice(&proto, &state, Vocab.opener, tb.bytes[Vocab.opener]);
    try testing.expectEqual(ChoiceOutcome.progress, out);
    try testing.expectEqual(@as(u8, 6), state.open_cursor);

    _ = try applyChoiceMask(&proto, &state, &g, &tb, &mask);
    try testing.expect(mask[Vocab.close_gt]); // ">" completes the opener
    try testing.expect(mask[Vocab.span_gt_reason]); // ">x" completes with reasoning payload
    try testing.expect(!mask[Vocab.json_ob]); // direct answer no longer reachable
    try testing.expect(!mask[Vocab.json_ws]);
    try testing.expect(!mask[Vocab.opener]); // "<think" no longer fits after cursor 6
    try testing.expect(!mask[Vocab.lt]); // "<" no longer fits either

    const done = observeChoice(&proto, &state, Vocab.close_gt, tb.bytes[Vocab.close_gt]);
    try testing.expectEqual(ChoiceOutcome.opened, done);
    try testing.expectEqual(Phase.reasoning, state.phase);

    // An opener-completing span's payload is REASONING, never grammar input.
    var state2 = State.initChoice();
    state2.open_cursor = 6;
    const spanned = observeChoice(&proto, &state2, Vocab.span_gt_reason, tb.bytes[Vocab.span_gt_reason]);
    try testing.expectEqual(ChoiceOutcome.opened, spanned);
    try testing.expectEqual(Phase.reasoning, state2.phase);
}

test "choice: direct JSON token enters the json body without opener bytes" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"object\"}");
    defer schema.deinit();
    var g = try json_grammar.Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    var tb = try Vocab.build(testing.allocator);
    defer tb.deinit();

    var proto = makeProtocol(BARE_THINK_OPENER, BARE_THINK_CLOSER, &.{
        Vocab.opener, Vocab.close_full, Vocab.lt,
    });

    var state = State.initChoice();
    const out = observeChoice(&proto, &state, Vocab.json_ob, tb.bytes[Vocab.json_ob]);
    try testing.expectEqual(ChoiceOutcome.json, out);
    try testing.expectEqual(Phase.json_body, state.phase);
    // The caller feeds the grammar; verify "{" advances it.
    try testing.expect(try g.acceptByte('{'));
}

test "choice: the atomic opener id resolves without bytes" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"object\"}");
    defer schema.deinit();
    var g = try json_grammar.Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    var tb = try Vocab.build(testing.allocator);
    defer tb.deinit();

    // The atomic opener rides the special-token identity: bytes stay null in
    // TokenBytes, so only the id can admit it.
    var proto = makeProtocol(BARE_THINK_OPENER, BARE_THINK_CLOSER, &.{});
    proto.opener_atomic = Vocab.close_atomic;

    var state = State.initChoice();
    var mask: [Vocab.COUNT]bool = undefined;
    _ = try applyChoiceMask(&proto, &state, &g, &tb, &mask);
    try testing.expect(mask[Vocab.close_atomic]);

    const out = observeChoice(&proto, &state, Vocab.close_atomic, null);
    try testing.expectEqual(ChoiceOutcome.opened, out);
    try testing.expectEqual(Phase.reasoning, state.phase);
}

test "reasoning: close recognized across split ordinary tokens and by atomic id" {
    var tb = try Vocab.build(testing.allocator);
    defer tb.deinit();
    var proto = makeProtocol(null, BARE_THINK_CLOSER, &.{});
    proto.closer_atomic = Vocab.close_atomic;

    // Byte-spelled close across fragments: "</th" holds the partial match,
    // "ink>" completes it with an empty payload.
    var state = State.initPromptOpened();
    try testing.expect(observeReasoningToken(&proto, &state, Vocab.prose, tb.bytes[Vocab.prose]) == null);
    try testing.expect(observeReasoningToken(&proto, &state, 100, "</th") == null);
    try testing.expectEqual(@as(u8, 4), state.close_match);
    const closed = observeReasoningToken(&proto, &state, 101, "ink>").?;
    try testing.expectEqualStrings("", closed);

    // A diverging partial resets: "</thi" + "X" then the real close.
    var state2 = State.initPromptOpened();
    try testing.expect(observeReasoningToken(&proto, &state2, 100, "</thi") == null);
    try testing.expectEqual(@as(u8, 5), state2.close_match);
    try testing.expect(observeReasoningToken(&proto, &state2, 102, "X hello") == null);
    try testing.expectEqual(@as(u8, 0), state2.close_match);
    try testing.expect(observeReasoningToken(&proto, &state2, 103, "more") == null);
    try testing.expect(observeReasoningToken(&proto, &state2, 104, ".</think>") != null);

    // Atomic identity wins even though TokenBytes has no bytes for it.
    var state3 = State.initPromptOpened();
    const atomic = observeReasoningToken(&proto, &state3, Vocab.close_atomic, null).?;
    try testing.expectEqualStrings("", atomic);
}

test "reasoning: a token that completes the close carries a validated payload suffix" {
    var tb = try Vocab.build(testing.allocator);
    defer tb.deinit();
    var proto = makeProtocol(null, BARE_THINK_CLOSER, &.{});
    proto.closer_atomic = null;

    var state = State.initPromptOpened();
    const closed = observeReasoningToken(&proto, &state, Vocab.span_close_json, tb.bytes[Vocab.span_close_json]).?;
    try testing.expectEqualStrings("{", closed); // "{" is the payload suffix; the caller feeds the grammar
    try testing.expectEqual(Phase.reasoning, state.phase); // the CALLER flips the phase after feeding
}

test "recovery composes the exact byte remainder after a partial close" {
    var proto = makeProtocol(null, "</think:opensource>", &.{});
    proto.closer_atomic = null;
    // Canonical per-suffix encodings (suffix k spells closer[k..]); the
    // server-side filler produces exactly this shape from the tokenizer.
    var buf: [SUFFIX_TABLE_BYTES]u32 = undefined;
    var table: [MAX_MARKER_BYTES]SuffixRun = @splat(.{ .offset = 0, .len = 0 });
    var off: u32 = 0;
    for (0.."</think:opensource>".len) |k| {
        const run_len: u32 = @intCast("</think:opensource>".len - k);
        for (0..run_len) |i| buf[off + i] = @intCast(100 + k + i);
        table[k] = .{ .offset = off, .len = @intCast(run_len) };
        off += run_len;
    }
    proto.closer_suffix = table;
    proto.closer_suffix_buf = buf;
    const ids = [_]u32{50};
    try testing.expect(proto.setForced(&ids));

    for (1.."</think:opensource>".len) |k| {
        var state = State.initPromptOpened();
        state.close_match = @intCast(k);
        var out: [MAX_TRANSITION_TOKENS]u32 = undefined;
        const n = proto.planRecovery(&state, &out).?;
        // One synthetic token per remaining byte.
        try testing.expectEqual("</think:opensource>".len - k, n);
        for (0.."</think:opensource>".len - k) |i| try testing.expectEqual(@as(u32, @intCast(100 + k + i)), out[i]);
    }

    // No partial bytes: the full canonical sequence.
    var state = State.initPromptOpened();
    var out: [MAX_TRANSITION_TOKENS]u32 = undefined;
    const n = proto.planRecovery(&state, &out).?;
    try testing.expectEqualSlices(u32, &ids, out[0..n]);
}

test "recovery after partial atomic-close bytes forces the byte remainder" {
    var proto = makeProtocol(null, BARE_THINK_CLOSER, &.{});
    proto.closer_atomic = 3;
    // Only the remainder after a 5-byte partial match is encodable.
    var buf: [SUFFIX_TABLE_BYTES]u32 = undefined;
    var table: [MAX_MARKER_BYTES]SuffixRun = @splat(.{ .offset = 0, .len = 0 });
    const k = 5; // "</thi"
    const rest = BARE_THINK_CLOSER[k..];
    for (0..rest.len) |i| buf[i] = @intCast(200 + i);
    table[k] = .{ .offset = 0, .len = @intCast(rest.len) };
    proto.closer_suffix = table;
    proto.closer_suffix_buf = buf;

    var state = State.initPromptOpened();
    state.close_match = @intCast(k);
    var out: [MAX_TRANSITION_TOKENS]u32 = undefined;
    const n = proto.planRecovery(&state, &out).?;
    try testing.expectEqual(rest.len, n);
    for (0..rest.len) |i| try testing.expectEqual(@as(u32, @intCast(200 + i)), out[i]);
    // The atomic id must NOT be appended after the remainder.
    try testing.expectEqual(@as(usize, rest.len), n);

    // Without partial bytes the atomic id is the whole transition.
    var state2 = State.initPromptOpened();
    const n2 = proto.planRecovery(&state2, &out).?;
    try testing.expectEqual(@as(usize, 1), n2);
    try testing.expectEqual(@as(u32, 3), out[0]);
}

test "recovery refuses an unencodable partial suffix instead of approximating" {
    var proto = makeProtocol(null, BARE_THINK_CLOSER, &.{});
    proto.closer_atomic = 3;
    // No table entry for the 2-byte partial: refuse rather than guess.
    var state = State.initPromptOpened();
    state.close_match = 2;
    var out: [MAX_TRANSITION_TOKENS]u32 = undefined;
    try testing.expect(proto.planRecovery(&state, &out) == null);
}

test "recovery completes a partial generated opener before closing" {
    var proto = makeProtocol(BARE_THINK_OPENER, BARE_THINK_CLOSER, &.{});
    proto.closer_atomic = 3;
    // Opener suffix table: cursor 6 -> ">" (token 9), cursor 5 -> "k>" (9,10).
    var obuf: [SUFFIX_TABLE_BYTES]u32 = undefined;
    var otable: [MAX_MARKER_BYTES]SuffixRun = @splat(.{ .offset = 0, .len = 0 });
    otable[6] = .{ .offset = 0, .len = 1 };
    obuf[0] = 9;
    otable[5] = .{ .offset = 1, .len = 2 };
    obuf[1] = 9;
    obuf[2] = 10;
    proto.opener_suffix = otable;
    proto.opener_suffix_buf = obuf;

    var state = State.initChoice();
    state.open_cursor = 6;
    var out: [MAX_TRANSITION_TOKENS]u32 = undefined;
    const n = proto.planRecovery(&state, &out).?;
    // ">" then the atomic close.
    try testing.expectEqual(@as(usize, 2), n);
    try testing.expectEqual(@as(u32, 9), out[0]);
    try testing.expectEqual(@as(u32, 3), out[1]);

    // Cursor 1 ("<"): "think>" then the atomic close.
    var state2 = State.initChoice();
    state2.open_cursor = 1;
    otable[1] = .{ .offset = 3, .len = 6 };
    for (0..6) |i| obuf[3 + i] = @intCast(20 + i);
    proto.opener_suffix = otable;
    proto.opener_suffix_buf = obuf;
    const n2 = proto.planRecovery(&state2, &out).?;
    try testing.expectEqual(@as(usize, 7), n2);
    try testing.expectEqual(@as(u32, 20), out[0]);
    try testing.expectEqual(@as(u32, 3), out[6]);
}

test "recovery plan: atomic closer is one token; byte closer drains canonically" {
    var proto = makeProtocol(null, BARE_THINK_CLOSER, &.{});
    proto.closer_atomic = 3;
    try testing.expectEqual(@as(usize, 1), proto.recoveryTokenCount());
    try testing.expectEqual(@as(usize, 0), proto.recoveryRemaining().len);

    var proto2 = makeProtocol(null, "</think:opensource>", &.{});
    proto2.closer_atomic = null;
    const ids = [_]u32{ 50, 51, 52 };
    try testing.expect(proto2.setForced(&ids));
    try testing.expectEqual(@as(usize, 3), proto2.recoveryTokenCount());
    try testing.expectEqualSlices(u32, &ids, proto2.recoveryRemaining());
}

test "suffixed protocol resolves opener and matching closer" {
    var proto = makeProtocol("<think:opensource>", "</think:opensource>", &.{});
    try testing.expectEqualStrings("<think:opensource>", proto.openerText().?);
    try testing.expectEqualStrings("</think:opensource>", proto.closerText());
    // A different-suffix close is body text, never the boundary.
    var state = State.initPromptOpened();
    try testing.expect(observeReasoningToken(&proto, &state, 200, "x</think:legacy>") == null);
    try testing.expectEqual(@as(u8, 0), state.close_match);
    const closed = observeReasoningToken(&proto, &state, 201, "y</think:opensource>").?;
    try testing.expectEqualStrings("", closed);
}

test "marker bytes beyond the matcher window cannot desync the tail" {
    var proto = makeProtocol(null, BARE_THINK_CLOSER, &.{});
    var state = State.initPromptOpened();
    // Flood past MAX_MARKER_BYTES with prose, then close normally.
    const flood: [64]u8 = @splat('x');
    try testing.expect(observeReasoningToken(&proto, &state, 300, &flood) == null);
    try testing.expect(state.tail_len == MAX_MARKER_BYTES);
    try testing.expect(observeReasoningToken(&proto, &state, 301, "</think>") != null);
}

test "payload offset is safe for atomic close tokens (no bytes)" {
    // The atomic close has null TokenBytes: the offset must come from the
    // bytes we actually observed, never a forced optional unwrap.
    try testing.expectEqual(@as(u32, 0), payloadByteOffset(null, ""));
    const bytes = BARE_THINK_CLOSER ++ "{x";
    try testing.expectEqual(@as(u32, bytes.len - 2), payloadByteOffset(bytes, "{x"));
    try testing.expectEqual(@as(u32, bytes.len), payloadByteOffset(bytes, ""));
}

test "production opener predicate admits continuation tokens" {
    const opener = BARE_THINK_OPENER; // "<think"
    try testing.expect(openerCandidateBytesMatch(opener, "<"));
    try testing.expect(openerCandidateBytesMatch(opener, "<th"));
    try testing.expect(openerCandidateBytesMatch(opener, "<think"));
    // Continuations past byte 6: ">" completes the opener.
    try testing.expect(openerCandidateBytesMatch(opener, ">"));
    // Completions carrying reasoning payload.
    try testing.expect(openerCandidateBytesMatch(opener, ">x"));
    // Unrelated tokens never match.
    try testing.expect(!openerCandidateBytesMatch(opener, "prose"));
    try testing.expect(!openerCandidateBytesMatch(opener, "{"));
    try testing.expect(!openerCandidateBytesMatch(opener, "x>"));
}

test "choice mask built from the PRODUCTION candidate builder admits continuations" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"object\"}");
    defer schema.deinit();
    var g = try json_grammar.Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    var tb = try Vocab.build(testing.allocator);
    defer tb.deinit();

    // Build the candidate list exactly the way LoadedModel does: scan the
    // vocabulary through the shared predicate. No hand-picked lists.
    var cands: std.ArrayList(u32) = .empty;
    defer cands.deinit(testing.allocator);
    for (tb.bytes, 0..) |maybe, id| {
        const b = maybe orelse continue;
        if (openerCandidateBytesMatch(BARE_THINK_OPENER, b)) {
            try cands.append(testing.allocator, @intCast(id));
        }
    }

    var proto = makeProtocol(BARE_THINK_OPENER, BARE_THINK_CLOSER, cands.items);
    proto.closer_atomic = null;

    var state = State.initChoice();
    var mask: [Vocab.COUNT]bool = undefined;
    // Cursor 0: opener starters are available.
    _ = try applyChoiceMask(&proto, &state, &g, &tb, &mask);
    try testing.expect(mask[Vocab.lt]);
    try testing.expect(mask[Vocab.opener]);
    // The close spelling is never opener progress even when the predicate
    // matches it (full-opener tokens are excluded at mask time).
    try testing.expect(!mask[Vocab.close_full]);

    // Commit "<think" (6 bytes), then ask again: the production list MUST
    // still offer a completion (">" or a ">x" span) — an empty legal set here
    // is the failure that disabled constraints in production.
    const out = observeChoice(&proto, &state, Vocab.opener, tb.bytes[Vocab.opener]);
    try testing.expectEqual(ChoiceOutcome.progress, out);
    _ = try applyChoiceMask(&proto, &state, &g, &tb, &mask);
    try testing.expect(mask[Vocab.close_gt]); // ">" completes
    try testing.expect(mask[Vocab.span_gt_reason]); // ">x" completes with reasoning

    const done = observeChoice(&proto, &state, Vocab.close_gt, tb.bytes[Vocab.close_gt]);
    try testing.expectEqual(ChoiceOutcome.opened, done);
}

test "choice: whitespace keeps the channel choice ambiguous" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"object\"}");
    defer schema.deinit();
    var g = try json_grammar.Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    var tb = try Vocab.build(testing.allocator);
    defer tb.deinit();

    var proto = makeProtocol(BARE_THINK_OPENER, BARE_THINK_CLOSER, &.{
        Vocab.opener, Vocab.lt, Vocab.close_gt, Vocab.span_gt_reason,
    });

    var state = State.initChoice();
    // A leading space (JSON-legal whitespace) must NOT commit the choice:
    // both the opener and JSON remain reachable afterwards.
    const ws = observeChoice(&proto, &state, Vocab.json_ws, tb.bytes[Vocab.json_ws]);
    try testing.expectEqual(ChoiceOutcome.progress, ws);
    try testing.expectEqual(Phase.choice, state.phase);
    try testing.expectEqual(@as(u8, 0), state.open_cursor);

    const ws2 = observeChoice(&proto, &state, Vocab.ws_nl, tb.bytes[Vocab.ws_nl]);
    try testing.expectEqual(ChoiceOutcome.progress, ws2);
    try testing.expectEqual(Phase.choice, state.phase);

    // The opener is still reachable after whitespace.
    var mask: [Vocab.COUNT]bool = undefined;
    _ = try applyChoiceMask(&proto, &state, &g, &tb, &mask);
    try testing.expect(mask[Vocab.opener]);
    try testing.expect(mask[Vocab.lt]);

    // And a direct JSON token still resolves the choice.
    const js = observeChoice(&proto, &state, Vocab.json_ob, tb.bytes[Vocab.json_ob]);
    try testing.expectEqual(ChoiceOutcome.json, js);
    try testing.expectEqual(Phase.json_body, state.phase);

    // A fresh state proves the whitespace never entered the grammar: "{" is
    // still the accepted first byte.
    var g2 = try json_grammar.Grammar.init(testing.allocator, &schema);
    defer g2.deinit();
    try testing.expect(try g2.acceptByte('{'));
}

test "reasoning mask rejects close-crossing tokens with schema-invalid payloads BEFORE sampling" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"object\"}");
    defer schema.deinit();
    var g = try json_grammar.Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    var tb = try Vocab.build(testing.allocator);
    defer tb.deinit();

    var proto = makeProtocol(null, BARE_THINK_CLOSER, &.{});
    proto.closer_atomic = null;
    // Production-style candidate set: every token whose bytes CONTAIN the
    // full closer (the only tokens that can complete it from match 0).
    var cands: std.ArrayList(u32) = .empty;
    defer cands.deinit(testing.allocator);
    for (tb.bytes, 0..) |maybe, id| {
        const b = maybe orelse continue;
        if (std.mem.indexOf(u8, b, BARE_THINK_CLOSER) != null) {
            try cands.append(testing.allocator, @intCast(id));
        }
    }
    proto.closer_span_candidates = cands.items;

    var state = State.initPromptOpened();
    var mask: [Vocab.COUNT]bool = undefined;
    const masked_out = try applyReasoningMask(&proto, &state, &g, &tb, &mask);

    // The invalid-payload crossing token ("close + x") must be excluded;
    // the valid one ("close + {") must stay; ordinary reasoning tokens and
    // EOS are untouched.
    try testing.expect(masked_out >= 1);
    try testing.expect(!mask[Vocab.span_close_invalid]);
    try testing.expect(mask[Vocab.span_close_json]);
    try testing.expect(mask[Vocab.prose]);
    try testing.expect(mask[Vocab.eos]);
    // With NO boundary-sensitive candidates and no partial match in flight,
    // the probe is a no-op and the caller keeps the unconstrained fast path.
    proto.closer_span_candidates = &.{};
    var state2 = State.initPromptOpened();
    const masked_out2 = try applyReasoningMask(&proto, &state2, &g, &tb, &mask);
    try testing.expectEqual(@as(usize, 0), masked_out2);
}

test "reasoning mask validates partial-close continuations from the live match state" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"object\"}");
    defer schema.deinit();
    var g = try json_grammar.Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    var tb = try Vocab.build(testing.allocator);
    defer tb.deinit();

    var proto = makeProtocol(null, BARE_THINK_CLOSER, &.{});
    proto.closer_atomic = null;
    // Production-style Set A: tokens whose bytes CONTAIN the full closer.
    var cands: std.ArrayList(u32) = .empty;
    defer cands.deinit(testing.allocator);
    for (tb.bytes, 0..) |maybe, id| {
        const b = maybe orelse continue;
        if (std.mem.indexOf(u8, b, BARE_THINK_CLOSER) != null) {
            try cands.append(testing.allocator, @intCast(id));
        }
    }
    proto.closer_span_candidates = cands.items;

    var mask: [Vocab.COUNT]bool = undefined;
    _ = try applyReasoningMask(&proto, &freshReasoningState(), &g, &tb, &mask);

    // Match state 8 of the 9-byte closer: the next byte must be ">".
    var state = State.initPromptOpened();
    state.close_match = @intCast(BARE_THINK_CLOSER.len - 1);
    _ = try applyReasoningMask(&proto, &state, &g, &tb, &mask);
    // ">x" completes with payload "x" — schema-invalid, masked out (Set B:
    // the token begins with the remaining delimiter bytes).
    try testing.expect(!mask[Vocab.span_gt_reason]);
    // ">" completes with an empty payload — legal.
    try testing.expect(mask[Vocab.close_gt]);
    // Prose that neither continues nor completes the close is untouched.
    try testing.expect(mask[Vocab.prose]);
    try testing.expect(mask[Vocab.eos]);

    // From an earlier partial (match 4, closer[4..] = "hink>"), candidates
    // are re-judged from the LIVE state: the invalid close-span token
    // re-completes after the diverging "<" resets the matcher and stays
    // masked (Set A probe through the shared simulator); a diverging ">"
    // is ordinary reasoning and stays legal.
    var state2 = State.initPromptOpened();
    state2.close_match = 4;
    _ = try applyReasoningMask(&proto, &state2, &g, &tb, &mask);
    try testing.expect(!mask[Vocab.span_close_invalid]);
    try testing.expect(mask[Vocab.close_gt]);
    try testing.expect(mask[Vocab.prose]);
}

fn freshReasoningState() State {
    return State.initPromptOpened();
}
