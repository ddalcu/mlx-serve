//! Bounded reasoning-protocol recognition for JSON-constrained generation.
//! Choice masks admit either an opener or JSON; header masks admit structural
//! channel transitions; reasoning is free except for tokens crossing into JSON.
//! Every payload suffix is validated before sampling. Generation publishes the
//! authoritative payload boundary for the shared streaming/batch Delivery router.
//! Tokenizer indexes and recovery encodings are immutable model-owned data.

const std = @import("std");
const json_grammar = @import("json_grammar.zig");
const token_mask = @import("token_mask.zig");

/// Bound for a delimiter or a complete assistant channel header.
pub const MAX_MARKER_BYTES = 128;
/// Canonical token encoding of the longest delimiter.
pub const MAX_FORCED_TOKENS = MAX_MARKER_BYTES;
/// Recovery: remaining header + close delimiter + final header.
pub const MAX_TRANSITION_TOKENS = 3 * MAX_FORCED_TOKENS;
/// Flat storage for one delimiter's per-suffix canonical encodings (the
/// remainder after each possible partial match). Worst case every suffix
/// needs the full token budget.
pub const SUFFIX_TABLE_BYTES = MAX_MARKER_BYTES * MAX_FORCED_TOKENS;

pub const SuffixRun = struct { offset: u32, len: u8 };
const empty_suffix: [MAX_MARKER_BYTES]SuffixRun = @splat(.{ .offset = 0, .len = 0 });

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
    const without_lead = std.mem.trimStart(u8, bytes, " \t\r\n");
    if (without_lead.len > 0 and (std.mem.startsWith(u8, opener, without_lead) or std.mem.startsWith(u8, without_lead, opener))) return true;
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
    gemma,
    inkling,
    harmony,
    muse,
};

pub const BARE_THINK_OPENER = "<think>";
pub const BARE_THINK_CLOSER = "</think>";

pub const Header = struct {
    text: []const u8,
    target: Phase,
    candidates: []const u32 = &.{},
    suffix: []const SuffixRun = &empty_suffix,
    tokens: []const u32 = &.{},
};
pub const Special = struct { id: u32, text: []const u8 };

pub const Protocol = struct {
    headers: [16]Header = undefined,
    header_len: u8 = 0,
    specials: [16]Special = undefined,
    special_len: u8 = 0,
    direct_json: bool = false,
    initial_phase: ?Phase = null,
    initial_cursor: u8 = 0,
    initial_candidates: u32 = 0,
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
    closer_suffix: []const SuffixRun = &empty_suffix,
    closer_suffix_buf: []const u32 = &.{},
    /// Same for the opener (choice state): entry k spells `opener_buf[k..]`.
    opener_suffix: []const SuffixRun = &empty_suffix,
    opener_suffix_buf: []const u32 = &.{},
    /// Model-level candidate set (borrowed from `LoadedModel`): ordinary
    /// tokens whose bytes CONTAIN the full close delimiter — the only tokens
    /// that can complete the close from match state 0. The reasoning-phase
    /// mask validates their payload suffixes before they can be sampled.
    closer_span_candidates: []const u32 = &.{},

    pub fn jsonOnly() Protocol {
        var p = Protocol{ .kind = .bare_think, .initial_phase = .json_body };
        _ = p.setCloser(BARE_THINK_CLOSER);
        return p;
    }

    /// Keep the rendered prefix but make reasoning unreachable. If a
    /// template already committed analysis, finish its structural header,
    /// close the empty segment, and enter a final header under the mask.
    /// The caller interns the temporary rule text in its model cache.
    pub fn finalOnly(self: *Protocol, buffer: []u8) bool {
        const state = self.startState();
        if (state.phase == .json_body) return true;
        if (state.phase == .choice) {
            self.initial_phase = .json_body;
            return true;
        }
        var final: []const u8 = "";
        for (self.headers[0..self.header_len]) |rule| {
            if (rule.target == .json_body) {
                final = rule.text;
                break;
            }
        }
        if (state.phase == .header) {
            var finals: u32 = 0;
            for (self.headers[0..self.header_len], 0..) |rule, i| {
                if (rule.target == .json_body) finals |= @as(u32, 1) << @intCast(i);
            }
            if (state.header_candidates & finals != 0) {
                self.initial_candidates = state.header_candidates & finals;
                return true;
            }
        }
        var prefix: []const u8 = "";
        if (state.phase == .header) {
            for (self.headers[0..self.header_len], 0..) |rule, i| {
                if (state.header_candidates & (@as(u32, 1) << @intCast(i)) != 0) {
                    prefix = rule.text;
                    break;
                }
            }
        }
        const text = std.fmt.bufPrint(buffer, "{s}{s}{s}", .{ prefix, self.closerText(), final }) catch return false;
        self.header_len = 0;
        self.addHeader(text, .json_body);
        self.initial_phase = .header;
        self.initial_cursor = if (state.phase == .header) state.header_cursor else 0;
        self.initial_candidates = 1;
        self.direct_json = false;
        return true;
    }

    pub fn configureChannels(self: *Protocol) void {
        switch (self.kind) {
            .bare_think, .suffixed_think => return,
            .gemma => {
                _ = self.setCloser("<channel|>");
                self.direct_json = true;
                self.addHeader("<|channel>thought\n", .reasoning);
                self.addHeader("<|channel>thought", .reasoning);
                self.addHeader("<|channel>\n", .json_body);
            },
            .inkling => {
                _ = self.setCloser("<|end_message|>");
                self.addHeader("<|message_model|><|content_thinking|>", .reasoning);
                self.addHeader("<|message_model|><|content_text|>", .json_body);
                self.addHeader("<|content_thinking|>", .reasoning);
                self.addHeader("<|content_text|>", .json_body);
            },
            .harmony => {
                _ = self.setCloser("<|end|>");
                inline for (.{ "<|start|>assistant", "" }) |prefix| {
                    self.addHeader(prefix ++ "<|channel|>analysis<|message|>", .reasoning);
                    self.addHeader(prefix ++ "<|channel|>final<|message|>", .json_body);
                    self.addHeader(prefix ++ "<|channel|>commentary<|message|>", .json_body);
                    self.addHeader(prefix ++ " to=user<|channel|>final<|message|>", .json_body);
                    self.addHeader(prefix ++ "<|channel|>final to=user<|message|>", .json_body);
                }
            },
            .muse => {
                _ = self.setCloser("<|eom|>");
                inline for (.{ "<|start|>assistant", "" }) |prefix| {
                    self.addHeader(prefix ++ " to=self<|message|>", .reasoning);
                    self.addHeader(prefix ++ " to=user<|message|>", .json_body);
                    self.addHeader(prefix ++ "<|message|>", .json_body);
                }
            },
        }
    }

    fn addHeader(self: *Protocol, text: []const u8, target: Phase) void {
        std.debug.assert(self.header_len < self.headers.len and text.len <= MAX_MARKER_BYTES);
        self.headers[self.header_len] = .{ .text = text, .target = target };
        self.header_len += 1;
    }

    fn allHeaders(self: *const Protocol) u32 {
        return (@as(u32, 1) << @intCast(self.header_len)) - 1;
    }

    pub fn startState(self: *const Protocol) State {
        if (self.initial_phase) |phase| return .{
            .phase = phase,
            .header_cursor = self.initial_cursor,
            .header_candidates = self.initial_candidates,
        };
        return if (self.openerText() == null) State.initPromptOpened() else State.initChoice();
    }

    /// Find the longest header prefix at the rendered prompt tail. Prompt
    /// decisions are request-local, while rule/tokenizer data is immutable.
    pub fn startFromPrompt(self: *Protocol, tail: []const u8) bool {
        var longest: usize = 0;
        var candidates: u32 = 0;
        var completed: ?Phase = null;
        for (self.headers[0..self.header_len], 0..) |rule, i| {
            var k: usize = 1;
            while (k <= rule.text.len) : (k += 1) {
                if (!std.mem.endsWith(u8, tail, rule.text[0..k])) continue;
                if (k > longest) {
                    longest = k;
                    candidates = 0;
                    completed = null;
                }
                if (k == longest) {
                    candidates |= @as(u32, 1) << @intCast(i);
                    if (k == rule.text.len) completed = rule.target;
                }
            }
        }
        if (longest == 0 and !self.direct_json) return false;
        self.initial_phase = completed orelse .header;
        self.initial_cursor = if (completed != null) 0 else @intCast(longest);
        self.initial_candidates = if (longest == 0) self.allHeaders() else candidates;
        return true;
    }

    pub fn tokenText(self: *const Protocol, id: u32, ordinary: ?[]const u8) ?[]const u8 {
        if (ordinary) |bytes| return bytes;
        for (self.specials[0..self.special_len]) |special| {
            if (id == special.id) return special.text;
        }
        return null;
    }

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
        if (state.phase == .header) {
            var selected: ?usize = null;
            for (self.headers[0..self.header_len], 0..) |rule, i| {
                if (state.header_candidates & (@as(u32, 1) << @intCast(i)) == 0) continue;
                if (selected == null or rule.target == .json_body) selected = i;
                if (rule.target == .json_body) break;
            }
            const rule = self.headers[selected orelse return null];
            n = appendRecovery(rule.suffix, rule.tokens, state.header_cursor, out, n) orelse return null;
            if (rule.target == .json_body) return n;
        }
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
        if (self.header_len > 0) {
            for (self.headers[0..self.header_len]) |rule| {
                if (rule.target == .json_body) {
                    return appendRecovery(rule.suffix, rule.tokens, 0, out, n);
                }
            }
            return null;
        }
        return n;
    }
};

fn appendRecovery(table: []const SuffixRun, tokens: []const u32, cursor: usize, out: []u32, n: usize) ?usize {
    if (cursor >= table.len) return null;
    const run = table[cursor];
    if (run.len == 0 or n + run.len > out.len) return null;
    @memcpy(out[n..][0..run.len], tokens[run.offset..][0..run.len]);
    return n + run.len;
}

pub const Phase = enum { choice, header, reasoning, json_body };

/// Mutable per-request state. Bounded: memory does not grow with reasoning
/// length (the close matcher keeps only the last `MAX_MARKER_BYTES` bytes).
pub const State = struct {
    phase: Phase = .json_body,
    header_cursor: u8 = 0,
    header_candidates: u32 = 0,
    invalid: bool = false,
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
    forced_cursor: u16 = 0,
    pending_len: u16 = 0,
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
    /// The opener and closer completed in this token; only this suffix is JSON.
    payload: []const u8,
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
    const raw = bytes orelse {
        state.phase = .json_body;
        return .json;
    };
    const b = if (state.open_cursor == 0) std.mem.trimStart(u8, raw, " \t\r\n") else raw;
    // JSON-legal whitespace does not resolve the choice: an opener or a JSON
    // value can both follow. Stay ambiguous (the cursor is untouched — the
    // opener match is byte-exact) and keep the grammar unfed.
    if (isJsonWhitespaceRun(b)) return .progress;
    if (openerCompletedBy(p, state.open_cursor, b)) {
        const rest = openerRemainingText(p, state.open_cursor);
        state.phase = .reasoning;
        if (observeReasoningToken(p, state, std.math.maxInt(u32), b[rest.len..])) |suffix| {
            state.phase = .json_body;
            return .{ .payload = suffix };
        }
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
    for (p.opener_candidates) |id| {
        if (id >= mask.len or mask[id]) continue;
        const raw = tb.bytes[id] orelse continue;
        const bytes = if (state.open_cursor == 0) std.mem.trimStart(u8, raw, " \t\r\n") else raw;
        if (bytes.len == 0) continue;
        if (tb.eos_id) |eos| {
            if (id == eos) continue;
        }
        const cursor = state.open_cursor;
        if (openerContinuesFrom(p, cursor, bytes) or openerCompletedBy(p, cursor, bytes)) {
            var trial = state.*;
            const outcome = observeChoice(p, &trial, id, raw);
            if (outcome == .payload and !try acceptsSuffix(grammar, outcome.payload)) continue;
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
        if (token_id == aid) return afterClose(p, state, "");
    }
    const b = bytes orelse return null;
    var sim = CloseSim.init(p.closerText(), state.close_match, state.tail[0..state.tail_len]);
    for (b, 0..) |c, i| {
        if (sim.feed(c)) {
            state.close_match = 0;
            state.tail_len = 0;
            return afterClose(p, state, b[i + 1 ..]);
        }
    }
    state.close_match = sim.match;
    @memcpy(state.tail[0..sim.tail_len], sim.tail[0..sim.tail_len]);
    state.tail_len = sim.tail_len;
    return null;
}

fn afterClose(p: *const Protocol, state: *State, suffix: []const u8) ?[]const u8 {
    state.close_match = 0;
    state.tail_len = 0;
    if (p.header_len == 0) {
        state.phase = .json_body;
        return suffix;
    }
    state.phase = .header;
    state.header_cursor = 0;
    state.header_candidates = p.allHeaders();
    return observeHeader(p, state, suffix);
}

fn observeHeader(p: *const Protocol, state: *State, bytes: []const u8) ?[]const u8 {
    for (bytes, 0..) |b, offset| {
        var matches: u32 = 0;
        var completed: ?Phase = null;
        for (p.headers[0..p.header_len], 0..) |rule, i| {
            const bit = @as(u32, 1) << @intCast(i);
            if (state.header_candidates & bit == 0) continue;
            if (state.header_cursor < rule.text.len and rule.text[state.header_cursor] == b) {
                matches |= bit;
                if (state.header_cursor + 1 == rule.text.len) completed = rule.target;
            }
        }
        if (matches == 0) {
            if (state.header_cursor == 0 and p.direct_json) {
                if (isJsonWhitespaceRun(bytes[offset..][0..1])) continue;
                state.phase = .json_body;
                return bytes[offset..];
            }
            state.invalid = true;
            return null;
        }
        state.header_candidates = matches;
        state.header_cursor += 1;
        if (completed) |phase| {
            state.phase = phase;
            state.header_cursor = 0;
            state.close_match = 0;
            state.tail_len = 0;
            if (phase == .json_body) return bytes[offset + 1 ..];
            return observeReasoningToken(p, state, std.math.maxInt(u32), bytes[offset + 1 ..]);
        }
    }
    return null;
}

/// The same transition drives candidate validation and selected-token
/// observation. Null means no payload; state.invalid rejects a malformed
/// header. Only the returned suffix belongs to the JSON grammar.
pub fn observeProtocolToken(p: *const Protocol, state: *State, id: u32, ordinary: ?[]const u8) ?[]const u8 {
    switch (state.phase) {
        .reasoning => return observeReasoningToken(p, state, id, ordinary),
        .header => {
            const bytes = p.tokenText(id, ordinary) orelse {
                state.invalid = true;
                return null;
            };
            return observeHeader(p, state, bytes);
        },
        .json_body => return ordinary,
        .choice => {
            const outcome = observeChoice(p, state, id, ordinary);
            return switch (outcome) {
                .json => ordinary,
                .payload => |bytes| bytes,
                else => null,
            };
        },
    }
}

fn allowHeaderToken(p: *const Protocol, state: *const State, grammar: *json_grammar.Grammar, tb: *const token_mask.TokenBytes, mask: []bool, id: u32) !bool {
    if (id >= mask.len or mask[id]) return false;
    var trial = state.*;
    const suffix = observeProtocolToken(p, &trial, id, tb.bytes[id]);
    if (trial.invalid) return false;
    if (suffix) |bytes| {
        if (!try acceptsSuffix(grammar, bytes)) return false;
    }
    mask[id] = true;
    return true;
}

pub fn applyHeaderMask(p: *const Protocol, state: *const State, grammar: *json_grammar.Grammar, tb: *const token_mask.TokenBytes, mask: []bool) !u32 {
    var allowed: u32 = 0;
    if (p.direct_json and state.header_cursor == 0) {
        const base = try token_mask.buildMask(grammar, tb, mask);
        allowed = base.allowed;
    } else @memset(mask, false);
    for (p.headers[0..p.header_len], 0..) |rule, i| {
        if (state.header_candidates & (@as(u32, 1) << @intCast(i)) == 0) continue;
        for (rule.candidates) |id| {
            if (try allowHeaderToken(p, state, grammar, tb, mask, id)) allowed += 1;
        }
    }
    for (p.specials[0..p.special_len]) |special| {
        if (try allowHeaderToken(p, state, grammar, tb, mask, special.id)) allowed += 1;
    }
    return allowed;
}

/// Trial only the bytes that enter the JSON body. Ordinary reasoning and
/// empty boundary suffixes take no grammar snapshot and allocate nothing.
fn acceptsSuffix(grammar: *json_grammar.Grammar, suffix: []const u8) !bool {
    if (suffix.len == 0) return true;
    const snap = try grammar.snapshot();
    defer grammar.discardSnapshot(snap);
    defer grammar.restoreFrom(snap) catch unreachable; // original stack capacity is retained
    for (suffix) |c| if (!try grammar.acceptByteFast(c)) return false;
    return true;
}

/// Mask boundary-crossing candidates in place, without a cap on exclusions.
/// The model index covers whole closers; the first-byte bucket covers the
/// remainder of a closer split across tokens. Both use the same recognizer.
pub fn applyReasoningMask(
    p: *const Protocol,
    state: *const State,
    grammar: *json_grammar.Grammar,
    tb: *const token_mask.TokenBytes,
    mask: []bool,
) std.mem.Allocator.Error!usize {
    std.debug.assert(mask.len == tb.bytes.len);
    var n: usize = 0;
    const rest = p.closerText()[state.close_match..];
    const partial = if (state.close_match > 0) tb.by_first[rest[0]] else &.{};
    for ([_][]const u32{ p.closer_span_candidates, partial }) |candidates| {
        for (candidates) |id| {
            if (id >= mask.len or (n > 0 and !mask[id])) continue;
            const bytes = tb.bytes[id] orelse continue;
            if (probeCloseCompletion(p, state, bytes) != null) {
                var trial = state.*;
                const payload = observeProtocolToken(p, &trial, id, bytes);
                if (trial.invalid or (payload != null and !try acceptsSuffix(grammar, payload.?))) {
                    if (n == 0) @memset(mask, true);
                    mask[id] = false;
                    n += 1;
                }
            }
        }
    }
    return n;
}

/// Shared constrained-response router. The generator's token/byte boundary
/// is authoritative; payload bytes are never scanned for markup. Only an
/// unresolved opener and a possible closing delimiter are held back.
/// Output buffers are reused across calls. Call noteToken before HTTP UTF8
/// carry handling, then feed only complete decoded chunks.
pub const Delivery = struct {
    proto: *const Protocol,
    header_state: State,
    token_index: u32 = 0,
    raw_bytes: usize = 0,
    consumed_bytes: usize = 0,
    payload_byte: ?usize = null,
    opener_cursor: usize = 0,
    opened: bool,
    pending: [MAX_MARKER_BYTES]u8 = undefined,
    pending_len: usize = 0,
    reasoning_started: bool = false,
    content_started: bool = false,
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,

    pub fn init(proto: *const Protocol) Delivery {
        return .{ .proto = proto, .header_state = proto.startState(), .opened = proto.openerText() == null };
    }

    pub fn deinit(self: *Delivery, allocator: std.mem.Allocator) void {
        self.reasoning.deinit(allocator);
        self.content.deinit(allocator);
    }

    pub fn noteToken(self: *Delivery, len: usize, span: ?ConstraintSpan) void {
        if (span) |s| {
            if (s.token_index == self.token_index) self.payload_byte = self.raw_bytes + s.byte_offset;
        }
        self.raw_bytes += len;
        self.token_index += 1;
    }

    fn emitReasoning(self: *Delivery, allocator: std.mem.Allocator, b: u8) !void {
        if (!self.reasoning_started and (b == ' ' or b == '\n')) return;
        self.reasoning_started = true;
        try self.reasoning.append(allocator, b);
    }

    fn flushPending(self: *Delivery, allocator: std.mem.Allocator) !void {
        if (!std.mem.eql(u8, self.pending[0..self.pending_len], self.proto.closerText())) {
            for (self.pending[0..self.pending_len]) |b| try self.emitReasoning(allocator, b);
        }
        self.pending_len = 0;
    }

    pub fn feed(self: *Delivery, allocator: std.mem.Allocator, bytes: []const u8) !void {
        self.reasoning.clearRetainingCapacity();
        self.content.clearRetainingCapacity();
        for (bytes) |b| {
            const is_content = if (self.payload_byte) |pb| self.consumed_bytes >= pb else false;
            self.consumed_bytes += 1;
            if (is_content) {
                try self.flushPending(allocator);
                if (!self.content_started and (b == ' ' or b == '\n')) continue;
                self.content_started = true;
                try self.content.append(allocator, b);
                continue;
            }
            if (self.proto.header_len > 0 and self.header_state.phase == .header) {
                _ = observeHeader(self.proto, &self.header_state, &.{b});
                if (self.header_state.invalid) return error.InvalidProtocolTransition;
                continue;
            }
            if (!self.opened) {
                const opener = self.proto.openerText().?;
                if (self.opener_cursor == 0 and (b == ' ' or b == '\n' or b == '\r' or b == '\t')) continue;
                // A direct answer's whitespace and a truncated header never
                // become reasoning. The generator admits only exact openers.
                if (b != opener[self.opener_cursor]) continue;
                self.opener_cursor += 1;
                self.opened = self.opener_cursor == opener.len;
                continue;
            }
            if (self.pending_len == self.pending.len) try self.flushPending(allocator);
            self.pending[self.pending_len] = b;
            self.pending_len += 1;
            while (self.pending_len > 0 and !std.mem.startsWith(u8, self.proto.closerText(), self.pending[0..self.pending_len])) {
                try self.emitReasoning(allocator, self.pending[0]);
                std.mem.copyForwards(u8, self.pending[0 .. self.pending_len - 1], self.pending[1..self.pending_len]);
                self.pending_len -= 1;
            }
            if (self.proto.header_len > 0 and std.mem.eql(u8, self.pending[0..self.pending_len], self.proto.closerText())) {
                self.pending_len = 0;
                _ = afterClose(self.proto, &self.header_state, "");
            }
        }
    }

    pub fn clearOutput(self: *Delivery) void {
        self.reasoning.clearRetainingCapacity();
        self.content.clearRetainingCapacity();
    }

    pub fn finish(self: *Delivery, allocator: std.mem.Allocator) !void {
        try self.flushPending(allocator);
    }
};

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
            "<think",                 ">",  BARE_THINK_CLOSER,        null, "{", "}", " ", "hello", "<", null,
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
    try testing.expectEqual(Phase.json_body, state.phase); // the transition owns the phase
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
    proto.closer_suffix = &table;
    proto.closer_suffix_buf = &buf;
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
    proto.closer_suffix = &table;
    proto.closer_suffix_buf = &buf;

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
    proto.opener_suffix = &otable;
    proto.opener_suffix_buf = &obuf;

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
    proto.opener_suffix = &otable;
    proto.opener_suffix_buf = &obuf;
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
    const flood: [2 * MAX_MARKER_BYTES]u8 = @splat('x');
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

test "all invalid crossing tokens must be masked" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"object\"}");
    defer schema.deinit();
    var g = try json_grammar.Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    const a = arena.allocator();
    const words = try a.alloc(?[]const u8, 257);
    var ids: [257]u32 = undefined;
    for (words, 0..) |*w, i| {
        w.* = try std.fmt.allocPrint(a, "</think>x{d}", .{i});
        ids[i] = @intCast(i);
    }
    var tb = try token_mask.TokenBytes.init(arena, words, null);
    defer tb.deinit();
    var p = makeProtocol(null, BARE_THINK_CLOSER, &.{});
    p.closer_atomic = null;
    p.closer_span_candidates = &ids;
    var s = State.initPromptOpened();
    var mask: [257]bool = undefined;
    _ = try applyReasoningMask(&p, &s, &g, &tb, &mask);
    try testing.expect(!mask[256]);
}

test "opener completion must observe its trailing partial closer" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"object\"}");
    defer schema.deinit();
    var g = try json_grammar.Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    const a = arena.allocator();
    const words = try a.alloc(?[]const u8, 2);
    words[0] = "<think>x</th";
    words[1] = "ink>{";
    var tb = try token_mask.TokenBytes.init(arena, words, null);
    defer tb.deinit();
    var p = makeProtocol(BARE_THINK_OPENER, BARE_THINK_CLOSER, &.{0});
    p.closer_atomic = null;
    var s = State.initChoice();
    var mask: [2]bool = undefined;
    _ = try applyChoiceMask(&p, &s, &g, &tb, &mask);
    try testing.expect(mask[0]);
    _ = observeChoice(&p, &s, 0, words[0]);
    const payload = observeReasoningToken(&p, &s, 1, words[1]);
    try testing.expect(payload != null);
}

test "delivery uses byte spans after whitespace and preserves marker strings" {
    var p = makeProtocol(BARE_THINK_OPENER, BARE_THINK_CLOSER, &.{});
    var d = Delivery.init(&p);
    defer d.deinit(testing.allocator);
    d.noteToken(1, null);
    try d.feed(testing.allocator, " ");
    try testing.expectEqualStrings("", d.reasoning.items);
    const json = "{\"tag\":\"<think>\"}";
    d.noteToken(json.len, .{ .token_index = 1, .byte_offset = 0 });
    try d.feed(testing.allocator, json);
    try testing.expectEqualStrings(json, d.content.items);
    try testing.expectEqualStrings("", d.reasoning.items);
}

test "delivery holds exact delimiters across chunks and honors partial completion" {
    var p = makeProtocol(null, "</think:opensource>", &.{});
    var d = Delivery.init(&p);
    defer d.deinit(testing.allocator);
    const body = "reason </think:legacy> still reasoning";
    d.noteToken(body.len, null);
    try d.feed(testing.allocator, body);
    try testing.expectEqualStrings(body, d.reasoning.items);
    d.noteToken(6, null);
    try d.feed(testing.allocator, "</thin");
    try testing.expectEqualStrings("", d.reasoning.items);
    const end = "k:opensource>{}";
    d.noteToken(end.len, .{ .token_index = 2, .byte_offset = end.len - 2 });
    try d.feed(testing.allocator, end);
    try testing.expectEqualStrings("{}", d.content.items);
    try testing.expectEqualStrings("", d.reasoning.items);
}

test "delivery token offsets survive a UTF8 carry and producer read-ahead" {
    var p = makeProtocol(null, BARE_THINK_CLOSER, &.{});
    var d = Delivery.init(&p);
    defer d.deinit(testing.allocator);
    // First byte is held by the HTTP decoder; the producer already knows a
    // future boundary. It must not switch the consumer's channel early.
    const span = ConstraintSpan{ .token_index = 1, .byte_offset = 9 };
    d.noteToken(1, span);
    d.noteToken(11, span); // second UTF8 byte + </think> + {}
    try d.feed(testing.allocator, "\xc3\xa9</think>{}");
    try testing.expectEqualStrings("\xc3\xa9", d.reasoning.items);
    try testing.expectEqualStrings("{}", d.content.items);
}

test "constrained split clips a stop before payload and preserves unrelated tags" {
    var p = makeProtocol(null, "</think:opensource>", &.{});
    const text = "r </think:legacy> then stop";
    var delivery = Delivery.init(&p);
    defer delivery.deinit(testing.allocator);
    delivery.payload_byte = text.len + 10;
    try delivery.feed(testing.allocator, text);
    try delivery.finish(testing.allocator);
    try testing.expectEqualStrings(text, delivery.reasoning.items);
    try testing.expectEqualStrings("", delivery.content.items);
    var choice = makeProtocol(BARE_THINK_OPENER, BARE_THINK_CLOSER, &.{});
    var partial = Delivery.init(&choice);
    defer partial.deinit(testing.allocator);
    try partial.feed(testing.allocator, " <thi");
    try partial.finish(testing.allocator);
    try testing.expectEqualStrings("", partial.reasoning.items);
    try testing.expectEqualStrings("", partial.content.items);
}

test "channel formats require their final header before JSON" {
    const cases = .{
        .{ Kind.gemma, "<|turn>model\n<|channel>thought\n", "work<channel|><|channel>\n{}" },
        .{ Kind.inkling, "<|message_model|><|content_thinking|>", "work<|end_message|><|message_model|><|content_text|>{}" },
        .{ Kind.harmony, "<|start|>assistant<|channel|>analysis<|message|>", "work<|end|><|start|>assistant<|channel|>final<|message|>{}" },
        .{ Kind.muse, "<|start|>assistant to=self<|message|>", "work<|eom|><|start|>assistant to=user<|message|>{}" },
    };
    inline for (cases) |case| {
        var p = Protocol{ .kind = case[0] };
        p.configureChannels();
        try testing.expect(p.startFromPrompt(case[1]));
        var state = p.startState();
        try testing.expectEqual(Phase.reasoning, state.phase);
        for (case[2], 0..) |_, i| {
            const bytes = case[2][i..][0..1];
            const suffix = observeProtocolToken(&p, &state, std.math.maxInt(u32), bytes);
            try testing.expect(!state.invalid);
            if (suffix) |json| {
                // Header completion may return an empty payload at its end.
                try testing.expect(json.len == 0 or json[0] == '{' or json[0] == '}');
            }
            if (i < case[2].len - 3) try testing.expect(state.phase != .json_body);
        }
        try testing.expectEqual(Phase.json_body, state.phase);
    }
}

fn indexTestProtocol(a: std.mem.Allocator, p: *Protocol, tb: *const token_mask.TokenBytes) !void {
    var closers: std.ArrayList(u32) = .empty;
    for (tb.bytes, 0..) |maybe, id| {
        const bytes = maybe orelse continue;
        if (std.mem.indexOf(u8, bytes, p.closerText()) != null) try closers.append(a, @intCast(id));
    }
    p.closer_span_candidates = try closers.toOwnedSlice(a);
    for (p.headers[0..p.header_len]) |*rule| {
        var candidates: std.ArrayList(u32) = .empty;
        for (tb.bytes, 0..) |maybe, id| {
            if (maybe) |bytes| {
                if (openerCandidateBytesMatch(rule.text, bytes)) try candidates.append(a, @intCast(id));
            }
        }
        rule.candidates = try candidates.toOwnedSlice(a);
    }
}

fn expectRoutedAtEverySplit(kind: Kind, prompt: []const u8, text: []const u8, expected_reasoning: []const u8, expected_json: []const u8) !void {
    for (1..text.len) |split| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var p = Protocol{ .kind = kind };
        p.configureChannels();
        try testing.expect(p.startFromPrompt(prompt));
        var token_arena = std.heap.ArenaAllocator.init(testing.allocator);
        const words = try token_arena.allocator().alloc(?[]const u8, 2);
        words[0] = text[0..split];
        words[1] = text[split..];
        var tb = try token_mask.TokenBytes.init(token_arena, words, null);
        defer tb.deinit();
        try indexTestProtocol(a, &p, &tb);
        var schema = try parseSchema(testing.allocator, "{\"type\":\"object\",\"properties\":{\"note\":{\"type\":\"string\"}}}");
        defer schema.deinit();
        var grammar = try json_grammar.Grammar.init(testing.allocator, &schema);
        defer grammar.deinit();
        var state = p.startState();
        var delivery = Delivery.init(&p);
        defer delivery.deinit(testing.allocator);
        var reasoning: std.ArrayList(u8) = .empty;
        var content: std.ArrayList(u8) = .empty;
        var span: ?ConstraintSpan = null;
        for (words, 0..) |word, i| {
            var mask: [2]bool = undefined;
            switch (state.phase) {
                .header => {
                    _ = try applyHeaderMask(&p, &state, &grammar, &tb, &mask);
                    if (!mask[i]) std.debug.print("header mask: kind={s} split={d} token={d} cursor={d} text={s}\n", .{ @tagName(kind), split, i, state.header_cursor, word.? });
                    try testing.expect(mask[i]);
                },
                .reasoning => {
                    const n = try applyReasoningMask(&p, &state, &grammar, &tb, &mask);
                    try testing.expect(n == 0 or mask[i]);
                },
                .json_body => {
                    _ = try token_mask.buildMask(&grammar, &tb, &mask);
                    try testing.expect(mask[i]);
                },
                .choice => unreachable,
            }
            const before = state.phase;
            const payload = observeProtocolToken(&p, &state, @intCast(i), word);
            try testing.expect(!state.invalid);
            if (payload) |bytes| {
                if (before != .json_body) span = .{ .token_index = @intCast(i), .byte_offset = @intCast(word.?.len - bytes.len) };
                for (bytes) |b| try testing.expect(try grammar.acceptByte(b));
            }
            delivery.noteToken(word.?.len, span);
            try delivery.feed(testing.allocator, word.?);
            try reasoning.appendSlice(a, delivery.reasoning.items);
            try content.appendSlice(a, delivery.content.items);
        }
        delivery.clearOutput();
        try delivery.finish(testing.allocator);
        try reasoning.appendSlice(a, delivery.reasoning.items);
        try testing.expectEqualStrings(expected_reasoning, reasoning.items);
        try testing.expectEqualStrings(expected_json, content.items);
        // The same router must reconstruct the same channels in one batch.
        var batch = Delivery.init(&p);
        defer batch.deinit(testing.allocator);
        batch.payload_byte = delivery.payload_byte;
        try batch.feed(testing.allocator, text);
        try batch.finish(testing.allocator);
        try testing.expectEqualStrings(expected_reasoning, batch.reasoning.items);
        try testing.expectEqualStrings(expected_json, batch.content.items);
    }
}

test "channel routing and masks agree at every token split, including repeated reasoning" {
    const json = "{\"note\":\"<think> and <|channel|> are data\"}";
    try expectRoutedAtEverySplit(.gemma, "<|turn>model\n", "<|channel>thought\nfirst<channel|><|channel>thought\nsecond<channel|><|channel>\n" ++ json, "first\nsecond", json);
    try expectRoutedAtEverySplit(.inkling, "<|message_model|>", "<|content_thinking|>first<|end_message|><|message_model|><|content_thinking|>second<|end_message|><|message_model|><|content_text|>" ++ json, "firstsecond", json);
    try expectRoutedAtEverySplit(.harmony, "<|start|>assistant", "<|channel|>analysis<|message|>first<|end|><|start|>assistant<|channel|>analysis<|message|>second<|end|><|start|>assistant<|channel|>final<|message|>" ++ json, "firstsecond", json);
    try expectRoutedAtEverySplit(.muse, "<|start|>assistant", " to=self<|message|>first<|eom|><|start|>assistant to=self<|message|>second<|eom|><|start|>assistant to=user<|message|>" ++ json, "firstsecond", json);
}

test "channel formats allow direct answers through their content headers" {
    try expectRoutedAtEverySplit(.gemma, "<|turn>model\n", "<|channel>\n{}", "", "{}");
    try expectRoutedAtEverySplit(.gemma, "<|turn>model\n", "  {}", "", "{}");
    try expectRoutedAtEverySplit(.inkling, "<|message_model|>", "<|content_text|>{}", "", "{}");
    try expectRoutedAtEverySplit(.harmony, "<|start|>assistant", "<|channel|>final<|message|>{}", "", "{}");
    try expectRoutedAtEverySplit(.harmony, "<|start|>assistant", "<|channel|>commentary<|message|>{}", "", "{}");
    try expectRoutedAtEverySplit(.muse, "<|start|>assistant", " to=user<|message|>{}", "", "{}");
    try expectRoutedAtEverySplit(.muse, "<|start|>assistant", "<|message|>{}", "", "{}");
}

fn asciiRecovery(a: std.mem.Allocator, text: []const u8) !struct { table: []SuffixRun, tokens: []u32 } {
    const table = try a.alloc(SuffixRun, text.len);
    var tokens: std.ArrayList(u32) = .empty;
    for (0..text.len) |k| {
        table[k] = .{ .offset = @intCast(tokens.items.len), .len = @intCast(text.len - k) };
        for (text[k..]) |b| try tokens.append(a, b);
    }
    return .{ .table = table, .tokens = try tokens.toOwnedSlice(a) };
}

test "every channel header prefix recovers to JSON without duplicating partial headers" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]Kind{ .gemma, .inkling, .harmony, .muse }) |kind| {
        var p = Protocol{ .kind = kind };
        p.configureChannels();
        const close = try asciiRecovery(a, p.closerText());
        p.closer_suffix = close.table;
        p.closer_suffix_buf = close.tokens;
        try testing.expect(p.setForced(close.tokens[0..p.closer_len]));
        for (p.headers[0..p.header_len]) |*rule| {
            const data = try asciiRecovery(a, rule.text);
            rule.suffix = data.table;
            rule.tokens = data.tokens;
        }
        for (p.headers[0..p.header_len]) |rule| {
            for (0..rule.text.len) |k| {
                var state = State{ .phase = .header, .header_candidates = p.allHeaders() };
                _ = observeProtocolToken(&p, &state, std.math.maxInt(u32), rule.text[0..k]);
                try testing.expect(!state.invalid);
                var plan: [MAX_TRANSITION_TOKENS]u32 = undefined;
                const count = p.planRecovery(&state, &plan) orelse return error.RecoveryRefused;
                for (plan[0..count]) |id| {
                    const bytes = [_]u8{@intCast(id)};
                    _ = observeProtocolToken(&p, &state, id, &bytes);
                    try testing.expect(!state.invalid);
                }
                try testing.expectEqual(Phase.json_body, state.phase);
            }
        }
        for (0..p.closer_len) |k| {
            var state = State.initPromptOpened();
            _ = observeProtocolToken(&p, &state, std.math.maxInt(u32), p.closerText()[0..k]);
            var plan: [MAX_TRANSITION_TOKENS]u32 = undefined;
            const count = p.planRecovery(&state, &plan) orelse return error.RecoveryRefused;
            for (plan[0..count]) |id| {
                _ = observeProtocolToken(&p, &state, id, &.{@intCast(id)});
                try testing.expect(!state.invalid);
            }
            try testing.expectEqual(Phase.json_body, state.phase);
        }
    }
}

test "thinking off constrains final headers even after a prompt committed analysis" {
    for ([_]Kind{ .gemma, .inkling, .harmony, .muse }) |kind| {
        var p = Protocol{ .kind = kind };
        p.configureChannels();
        const reasoning_header = p.headers[0].text;
        for (0..reasoning_header.len + 1) |k| {
            var configured = p;
            if (k == 0) {
                configured.initial_phase = .header;
                configured.initial_candidates = configured.allHeaders();
            } else try testing.expect(configured.startFromPrompt(reasoning_header[0..k]));
            var buffer: [MAX_MARKER_BYTES]u8 = undefined;
            try testing.expect(configured.finalOnly(&buffer));
            var state = configured.startState();
            if (state.phase == .json_body) continue;
            var final: ?Header = null;
            for (configured.headers[0..configured.header_len], 0..) |rule, i| {
                if (state.header_candidates & (@as(u32, 1) << @intCast(i)) != 0) {
                    try testing.expectEqual(Phase.json_body, rule.target);
                    final = rule;
                }
            }
            const rule = final orelse return error.NoFinalHeader;
            _ = observeProtocolToken(&configured, &state, std.math.maxInt(u32), rule.text[state.header_cursor..]);
            try testing.expectEqual(Phase.json_body, state.phase);
            try testing.expect(!state.invalid);
        }
    }
}

test "special-token channel headers preserve identity and never become JSON input" {
    var p = Protocol{ .kind = .harmony };
    p.configureChannels();
    try testing.expect(p.startFromPrompt("<|start|>assistant"));
    const specials = [_]Special{
        .{ .id = 0, .text = "<|channel|>" },
        .{ .id = 1, .text = "<|message|>" },
        .{ .id = 2, .text = "<|end|>" },
        .{ .id = 3, .text = "<|start|>" },
    };
    @memcpy(p.specials[0..specials.len], &specials);
    p.special_len = specials.len;
    p.closer_atomic = 2;
    var state = p.startState();
    _ = observeProtocolToken(&p, &state, 0, null);
    _ = observeProtocolToken(&p, &state, 100, "analysis");
    _ = observeProtocolToken(&p, &state, 1, null);
    try testing.expectEqual(Phase.reasoning, state.phase);
    _ = observeProtocolToken(&p, &state, 100, "reason");
    _ = observeProtocolToken(&p, &state, 2, null);
    try testing.expectEqual(Phase.header, state.phase);
    _ = observeProtocolToken(&p, &state, 3, null);
    _ = observeProtocolToken(&p, &state, 100, "assistant");
    _ = observeProtocolToken(&p, &state, 0, null);
    _ = observeProtocolToken(&p, &state, 100, "final");
    try testing.expectEqualStrings("", observeProtocolToken(&p, &state, 1, null).?);
    try testing.expectEqual(Phase.json_body, state.phase);
    try testing.expect(!state.invalid);
}

test "channel masks reject malformed transitions and invalid crossing payloads" {
    inline for (.{ Kind.gemma, Kind.inkling, Kind.harmony, Kind.muse }) |kind| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var p = Protocol{ .kind = kind };
        p.configureChannels();
        const final = for (p.headers[0..p.header_len]) |rule| {
            if (rule.target == .json_body) break rule.text;
        } else unreachable;
        const good = try std.fmt.allocPrint(a, "{s}{s}{{}}", .{ p.closerText(), final });
        const bad = try std.fmt.allocPrint(a, "{s}{s}garbage", .{ p.closerText(), final });
        const malformed = try std.fmt.allocPrint(a, "{s}<invalid-header>{{}}", .{p.closerText()});
        var ta = std.heap.ArenaAllocator.init(testing.allocator);
        const words = try ta.allocator().alloc(?[]const u8, 4);
        @memcpy(words, &[_]?[]const u8{ good, bad, malformed, "ordinary reasoning" });
        var tb = try token_mask.TokenBytes.init(ta, words, null);
        defer tb.deinit();
        try indexTestProtocol(a, &p, &tb);
        var schema = try parseSchema(testing.allocator, "{\"type\":\"object\"}");
        defer schema.deinit();
        var grammar = try json_grammar.Grammar.init(testing.allocator, &schema);
        defer grammar.deinit();
        var state = State.initPromptOpened();
        var mask: [4]bool = undefined;
        try testing.expectEqual(@as(u32, 2), try applyReasoningMask(&p, &state, &grammar, &tb, &mask));
        try testing.expectEqualSlices(bool, &.{ true, false, false, true }, &mask);
        try testing.expectEqual(Phase.reasoning, state.phase);
        // Candidate validation must not mutate the JSON grammar.
        try testing.expect(try grammar.acceptByte('{'));
        try testing.expect(try grammar.acceptByte('}'));
        try testing.expect(grammar.isComplete());
    }
}

test "Gemma prompt can end immediately after the thought channel name" {
    var p = Protocol{ .kind = .gemma };
    p.configureChannels();
    try testing.expect(p.startFromPrompt("<|turn>model\n<|channel>thought"));
    try testing.expectEqual(Phase.reasoning, p.startState().phase);
}
