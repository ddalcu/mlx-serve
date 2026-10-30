//! Streaming gates: `streamShouldBufferForTools`, `ThinkScan`, the think gate, and the
//! content lead.

const std = @import("std");
const testing = std.testing;

const TagAt = @import("think_tags.zig").TagAt;
const isPartialSuffixedTag = @import("think_tags.zig").isPartialSuffixedTag;
const indexOfThinkOpenTag = @import("think_tags.zig").indexOfThinkOpenTag;
const indexOfThinkCloseTag = @import("think_tags.zig").indexOfThinkCloseTag;
const partialThinkCloseSuffixLen = @import("think_tags.zig").partialThinkCloseSuffixLen;
const invokeOpenerHoldsForTools = @import("../tools/markup_trim.zig").invokeOpenerHoldsForTools;
const INKLING_THINKING_TAG = @import("inkling.zig").INKLING_THINKING_TAG;
const INKLING_TEXT_TAG = @import("inkling.zig").INKLING_TEXT_TAG;
const INKLING_END_TAG = @import("inkling.zig").INKLING_END_TAG;
const INKLING_MODEL_TAG = @import("inkling.zig").INKLING_MODEL_TAG;
const INKLING_INVOKE_TAG = @import("inkling.zig").INKLING_INVOKE_TAG;
const inklingStripMessageHead = @import("inkling.zig").inklingStripMessageHead;
const harmonyStreamVerdict = @import("harmony.zig").harmonyStreamVerdict;
const harmonyHeaderHoldsForTools = @import("harmony.zig").harmonyHeaderHoldsForTools;
const museStreamVerdict = @import("muse.zig").museStreamVerdict;
const museHeaderHoldsForTools = @import("muse.zig").museHeaderHoldsForTools;
const parseToolCalls = @import("../tools/parse.zig").parseToolCalls;
const inklingIsNameChar = @import("../tools/inkling.zig").inklingIsNameChar;
const miniCpm5AttrValue = @import("../tools/minicpm5.zig").miniCpm5AttrValue;

/// Streaming-only: true when the buffer TAIL is a partial prefix of a think
/// opener (`<think>` / `<|channel>thought`). The buffered-stream flush must
/// hold these bytes back until the tag completes or diverges — flushing them
/// leaks tag fragments as visible content (a pi session showed prose ending
/// in a glued "thought" because `<|channel>` flushed before "thought"
/// arrived and completed the opener).
pub fn endsWithPartialThinkOpen(buf: []const u8) bool {
    const tags = [_][]const u8{ "<|channel>thought", "<think>" };
    for (tags) |tag| {
        // Strictly-partial prefixes only — a COMPLETE opener in the buffer is
        // the caller's contains-check's job; ours is the growing tail.
        var l = @min(buf.len, tag.len - 1);
        while (l > 0) : (l -= 1) {
            if (std.mem.endsWith(u8, buf, tag[0..l])) return true;
        }
    }
    // Hy3-suffixed opener mid-stream (`…<think:opensou`): everything from the
    // last `<` must be a valid growing prefix of `<think:suffix>`.
    if (std.mem.lastIndexOfScalar(u8, buf, '<')) |lt| {
        const tail = buf[lt..];
        if (tail.len > "<think".len and isPartialSuffixedTag(tail, "<think")) return true;
    }
    return false;
}

/// Streaming-only: should the chat-completion SSE path defer flushing
/// because `buf` is on track to become a tool call?
///
/// True when:
///   * `buf` contains `<tool` followed by a valid tag terminator
///     (`>`, ` `, `\t`, `\n`, `_`, `|`) somewhere — that's any of the
///     accepted families: `<tool>`, `<tool …>`, `<tool_call…>`, `<tool_calls…>`,
///     `<tool_request…>`, `<tool_requests…>` (mirrors `parseToolCalls`).
///   * `buf` contains the Gemma 4 `<|tool_call` substring.
///   * `functionOpenerHoldsForTools(buf)` — a MiniCPM5 `<function name="…">`
///     opener still unresolved, or resolved into a real call. `<functional`,
///     `<function>` and `<function foo>` are not this dialect and keep flowing.
///     NOTE the bare equals-sign Hermes `<function=NAME>` form is deliberately
///     NOT held here: `parseToolCalls` accepts it wrapper-less, so the gate is
///     narrower than the parser for THAT dialect. Pre-existing on main, tracked
///     separately — two attempts at closing it inside this change were both
///     wrong, and real captures show non-adjacent `<parameter=` openers.
///   * `buf[0] == '{'` and `buf` contains `"name"` (raw JSON tool-call shape).
///   * `buf` ends with a partial prefix that could grow into one of the above
///     in the next token (`<`, `<t`, `<to`, `<too`, `<tool`, or any prefix
///     of `<|tool_call`).
///
/// Conservatively false on `<toolkit>`, `<toolbar>`, etc. — anything where
/// the char after `<tool` isn't a valid tag terminator. That keeps prose
/// that happens to mention HTML-ish tags flowing through normally.
pub fn streamShouldBufferForTools(buf: []const u8) bool {
    if (buf.len == 0) return false;

    // Raw JSON tool-call shape: starts with `{` and has `"name"` somewhere.
    if (buf[0] == '{' and std.mem.indexOf(u8, buf, "\"name\"") != null) return true;

    // Gemma 4 fully-formed canonical open.
    if (std.mem.indexOf(u8, buf, "<|tool_call") != null) return true;

    // Inkling: the invoke marker is a SINGLE special token (arrives whole) —
    // once present, everything from here on is call payload. Without this the
    // NAME + full JSON streamed as visible content deltas (live,
    // first real pi session; the leak landed in pi's transcript and
    // contaminated every later turn's history).
    if (std.mem.indexOf(u8, buf, INKLING_INVOKE_TAG) != null) return true;

    // DeepSeek-V4 DSML: everything from `<｜DSML｜` on is a tool-call block.
    if (std.mem.indexOf(u8, buf, "<｜DSML｜") != null) return true;
    if (std.mem.indexOf(u8, buf, "</｜DSML｜") != null) return true;

    // Muse-Glimmer ATEM: everything from the wrapper on is call payload.
    if (std.mem.indexOf(u8, buf, "<atem:") != null) return true;
    // Muse segment-header hold: between <|start|> and <|message|> only the
    // role+recipient text flows, and a non-self/user recipient names a TOOL.
    if (museHeaderHoldsForTools(buf)) return true;
    // gpt_oss/harmony segment-header hold: same shape, `<|channel|>`-keyed.
    if (harmonyHeaderHoldsForTools(buf)) return true;

    // Inkling NAME hold: with a message-boundary marker present (a family
    // signal no other template emits), a segment that is still a bare
    // identifier run may be the NAME directly before an invoke marker. Prose
    // disambiguates within a token or two (space/punctuation); a call
    // disambiguates at the invoke marker. An EMPTY segment never holds, so
    // thinking splits stay prompt and marker-only flushes keep flowing.
    if (inklingSegmentCouldBeToolName(buf)) return true;

    // `<tool…` family: walk every `<tool` occurrence, accept the first one
    // whose terminator is valid (mirrors parseToolCalls' acceptance rule).
    var scan: usize = 0;
    while (std.mem.indexOf(u8, buf[scan..], "<tool")) |rel| {
        const after = scan + rel + "<tool".len;
        if (after >= buf.len) return true; // truncated mid-prefix → in progress
        const c = buf[after];
        if (c == '>' or c == ' ' or c == '\t' or c == '\n' or c == '_' or c == '|') return true;
        scan = after;
    }

    // `<function…` family: MiniCPM5's `<function name="…">`.
    if (functionOpenerHoldsForTools(buf)) return true;
    // Anthropic-style `<function_calls>` / `<invoke name="…">`.
    if (invokeOpenerHoldsForTools(buf)) return true;

    // Trailing partial prefixes — the next streamed token could complete any
    // of these into a real tool open. Order doesn't matter; first endsWith
    // hit wins. Listed shortest-first for legibility.
    const tail_prefixes = [_][]const u8{
        "<",     "<t",     "<to",     "<too",     "<tool",
        "<|",    "<|t",    "<|to",    "<|too",    "<|tool",
        "<|tool_", "<|tool_c", "<|tool_ca", "<|tool_cal",
        // MiniCPM5 V3 `<function name="…">` — COMPLETE ladder. `<funct` is a
        // real decomposition in this vocabulary (`<f` id 54303 + `unct` id
        // 14185 decode to exactly `<funct`); omitting that one rung flushed the
        // fragment and leaked the rest of the tag. The rungs are DERIVED in the
        // test, so a future gap fails there instead of shipping.
        "<f",
        "<fu",      "<fun",      "<func",     "<funct",     "<functi",
        "<functio", "<function",
        // Muse ATEM (a fused multi-char BPE fragment can end mid-marker)
        "<a",        "<at",        "<ate",
        "<atem",    "<atem:",
        // DSML fullwidth-bar prefixes (`｜` = 3 bytes; cover mid-codepoint
        // splits too in case the tokenizer spells the marker in pieces)
           "<\xef",     "<\xef\xbd",
        "<｜",
        "<｜D",
        "<｜DS",
        "<｜DSM",
        "<｜DSML",
        "<｜DSML\xef",
        "<｜DSML\xef\xbd",
        // its closers: a lone one leaks the same way when the opener is gone
        "</",      "</\xef",    "</\xef\xbd",
        "</｜",     "</｜D",     "</｜DS",    "</｜DSM",    "</｜DSML",
        "</｜DSML\xef", "</｜DSML\xef\xbd",
    };
    for (tail_prefixes) |p| {
        if (std.mem.endsWith(u8, buf, p)) return true;
    }
    return false;
}

/// Inkling streaming NAME hold: true when the text after the last message
/// boundary (<|end_message|> / <|message_model|> / <|content_text|>) is a
/// NON-EMPTY bare identifier run — the shape of a tool NAME right before its
/// invoke marker. A thinking message never matches (no boundary marker →
/// whole buffer, which carries `<|`); other families never emit these
/// markers at all.
fn inklingSegmentCouldBeToolName(buf: []const u8) bool {
    var seg_start: usize = 0;
    var saw_marker = false;
    inline for (.{ INKLING_END_TAG, INKLING_MODEL_TAG, INKLING_TEXT_TAG }) |tag| {
        if (std.mem.lastIndexOf(u8, buf, tag)) |p| {
            saw_marker = true;
            seg_start = @max(seg_start, p + tag.len);
        }
    }
    if (!saw_marker) return false;
    const seg = buf[seg_start..];
    if (seg.len == 0 or seg.len > 64) return false;
    for (seg) |c| {
        if (!inklingIsNameChar(c)) return false;
    }
    return true;
}

/// Streaming-only: does a `<function …` opener anywhere in `buf` still hold?
/// Three arrival states, mirroring `museHeaderHoldsForTools` above:
///
///   unresolved (no `>` yet)        → HOLD; the next token may complete a call
///   resolved WITH a quoted name=   → HOLD; it is a MiniCPM5 V3 call
///   resolved WITHOUT one          → RELEASE and keep scanning; the parser
///                                    (parseMiniCpm5ToolCalls) will decline it
///
/// The `name=` test delegates to `miniCpm5AttrValue`, so the gate's acceptance
/// is DERIVED from the parser's instead of hand-mirrored. That direction
/// matters: the gate must stay a strict SUPERSET of the parser. A stricter gate
/// (e.g. one demanding `name=` immediately after the whitespace) would flush a
/// real call's opener as content — `<function foo name="x">` does parse — and
/// the end-of-stream parse would then emit the call anyway: leak AND duplicate.
///
/// Releasing the resolved-non-call case is what keeps prose usable. The hold is
/// monotonic over a buffer that never shrinks, so treating any `<function foo>`
/// as a call silences every remaining token of a tool-enabled turn.
fn functionOpenerHoldsForTools(buf: []const u8) bool {
    var scan: usize = 0;
    while (std.mem.indexOf(u8, buf[scan..], "<function")) |rel| {
        const after = scan + rel + "<function".len;
        // Truncated mid-marker — the next token decides what this is.
        if (after >= buf.len) return true;
        // `<functional`, `<function=NAME>` (the equals-sign Hermes form) and
        // `<function>` are not this dialect; skip and keep scanning so ordinary
        // prose keeps flowing. Gating the bare `<function=` dialect is a
        // SEPARATE problem with its own evidence — see the follow-up issue.
        if (!std.ascii.isWhitespace(buf[after])) {
            scan = after;
            continue;
        }
        const tag_rel = std.mem.indexOfScalar(u8, buf[after..], '>') orelse return true;
        const tag_end = after + tag_rel;
        // Empty `name=""` is rejected by the parser too, so it releases here.
        if (miniCpm5AttrValue(buf[after..tag_end], "name")) |v| {
            if (v.len > 0) return true;
        }
        scan = tag_end + 1;
    }
    return false;
}

/// Streaming-only: what should a tools-enabled SSE path do with the buffered
/// text so far, with respect to thinking? Shared by the chat-completions and
/// Anthropic /v1/messages stream handlers — the two paths drifted apart once
/// already: /v1/messages only recognized think OPENERS present in the output,
/// so Qwen-family template-opened thinking (opener injected into the PROMPT)
/// streamed as visible text and a raw `</think>` leaked into Claude Code
/// transcripts ( live).
///
///   .hold_thinking — inside an unclosed think block; buffer, stream only the
///                    reasoning not yet sent (`unstreamedReasoning`)
///   .split_think   — close tag arrived; splitThinkBlock once, emit
///                    reasoning + visible remainder, clear the buffer, and
///                    set think_closed for the rest of the turn
///   .flush_text    — plain visible prose; stream it
///
/// `think_closed` releases the enable_thinking hold after the one split —
/// without it the visible answer sits in the buffer until end-of-stream and
/// gets misfiled as reasoning (the pi hidden-answer bug).
pub const StreamThinkGate = enum { flush_text, hold_thinking, split_think };

/// The gate WITHOUT a prompt-opened think block — every family whose template
/// gates its `<think>` on `enable_thinking` renders the closed `<think></think>`
/// signature (or no tag at all) when thinking is off, so the flag is false for
/// them and this stays their exact behavior.
pub fn streamThinkGate(buf: []const u8, enable_thinking: bool, think_closed: bool) StreamThinkGate {
    return streamThinkGate2(buf, enable_thinking, think_closed, false);
}

/// `prompt_opened_think`: the rendered generation prompt ended with an OPEN
/// think tag (`server.promptOpensThink`). That is a fact about the prompt
/// BYTES, not about our request flag — LFM2.5's template pre-opens `<think>`
/// unconditionally, so with thinking off the model's reasoning arrives as
/// tag-free prose and the 3-arg gate flushed the whole chain-of-thought as the
/// visible answer. It can only ever ADD a hold: when
/// thinking is on `enable_thinking` already held, and once `think_closed` the
/// term is dropped so the answer after the block still flushes.
/// Memoized marker scan for the streaming think gate, owned BESIDE the stream
/// buffer and reset with it.
///
/// Without it `streamThinkGate2` re-scans the whole accumulated buffer on every
/// token for as long as thinking markup is present and no close has arrived —
/// O(buffer) per token, O(n²) over a reasoning block. Measured on a hermetic
/// 4000-token stream growing to 113 KB: 47.99 µs/token with no close in the
/// buffer (every scan runs to the end), 16.98 µs/token with a close near the
/// front (the close scan stops early). ~0.3% of a 15 ms/token model, ~1% at
/// 5 ms/token, and it GROWS with the thought.
///
/// Exactness rests on one property of the marker grammar: `tagSuffixChar`
/// excludes `<`, so every recognized marker contains exactly ONE `<`, at its
/// start. A marker straddling the scanned/unscanned boundary must therefore
/// begin at the LAST `<` in the buffer — and if that `<` cannot still grow into
/// a marker, nothing straddles the boundary at all. So the resume point is
/// exact and bounded, with no overlap constant to get wrong.
pub const ThinkScan = struct {
    /// Bytes already examined. Never trusted when the buffer is SHORTER than
    /// this (the loop reset the buffer without resetting the scan).
    checked_upto: usize = 0,
    /// Index of the last `<` seen, or null. The only position a marker can
    /// straddle the boundary from.
    last_lt: ?usize = null,
    saw_channel_thought: bool = false,
    saw_open: bool = false,
    saw_channel_close: bool = false,
    /// Tool markup anywhere in the buffer. This is what makes a found close
    /// UNSTABLE and therefore un-latchable: `thinkCloseIsToolCallPayload`
    /// reclassifies a close as an argument value once a `</tool_call` arrives
    /// AFTER it, so a close that was real at token N is payload at token N+k.
    /// With no tool markup the close is stable and latching is exact.
    saw_tool_marker: bool = false,
    /// First real close tag, once found and stable.
    close: ?TagAt = null,
    /// Bytes examined by the last call, INCLUDING the exact full re-scan the
    /// tool-markup case falls back to. Test-visible: the invariant is that this
    /// stays FLAT as the buffer grows for traffic with no tool markup — which
    /// is the arm the 47.99 µs/token bench measured.
    last_scan_span: usize = 0,

    /// Longest plain substring needle minus one — the overlap a windowed
    /// substring search needs so a needle split across two arrivals is caught.
    /// `<|channel>thought` is the longest at 17.
    const substr_overlap: usize = 16;

    pub fn reset(self: *ThinkScan) void {
        self.* = .{};
    }

    /// Where the next scan must start so no marker spanning the boundary is
    /// missed: the last `<` if it can still grow into a tag (see the type's
    /// doc comment for why no earlier `<` can), and far enough back for the
    /// plain substring needles either way.
    fn resumeFrom(self: *const ThinkScan, buf: []const u8) usize {
        const upto = @min(self.checked_upto, buf.len);
        const substr_from = upto -| substr_overlap;
        const lt = self.last_lt orelse return substr_from;
        if (lt >= upto) return substr_from;
        const tail = buf[lt..upto];
        const growing = isPartialSuffixedTag(tail, "<think") or
            isPartialSuffixedTag(tail, "</think") or
            (tail.len < "<|channel>thought".len and std.mem.startsWith(u8, "<|channel>thought", tail)) or
            (tail.len < "<channel|>".len and std.mem.startsWith(u8, "<channel|>", tail));
        return if (growing) @min(lt, substr_from) else substr_from;
    }

    fn advance(self: *ThinkScan, buf: []const u8) void {
        if (buf.len < self.checked_upto) self.reset(); // buffer was cleared under us
        if (buf.len == self.checked_upto and self.checked_upto != 0) {
            self.last_scan_span = 0;
        } else {
            const from = self.resumeFrom(buf);
            self.last_scan_span = buf.len - from;
            const win = buf[from..];
            if (std.mem.lastIndexOfScalar(u8, win, '<')) |rel| self.last_lt = from + rel;
            if (!self.saw_channel_thought and std.mem.indexOf(u8, win, "<|channel>thought") != null)
                self.saw_channel_thought = true;
            if (!self.saw_channel_close and std.mem.indexOf(u8, win, "<channel|>") != null)
                self.saw_channel_close = true;
            if (!self.saw_tool_marker and std.mem.indexOf(u8, win, "tool_call") != null)
                self.saw_tool_marker = true;
            // The open scan takes the FULL buffer with a start offset: it
            // consults nothing before the match, but a windowed slice would
            // shift the reported position. Monotone — a complete opener stays
            // one — so latching is exact.
            if (!self.saw_open and indexOfThinkOpenTag(buf, from) != null) self.saw_open = true;
            if (self.close == null and !self.saw_tool_marker)
                self.close = indexOfThinkCloseTag(buf, from);
            self.checked_upto = buf.len;
        }
        if (self.saw_tool_marker) {
            // Exact, and only here: the payload classification looks FORWARD
            // past the close for a `</tool_call`, so it must be recomputed on
            // the whole buffer. The caller latches `think_closed` right after a
            // split, so this runs for a handful of tokens, not a whole block.
            self.close = indexOfThinkCloseTag(buf, 0);
            self.last_scan_span = buf.len;
        }
    }
};

/// `streamThinkGate2` with the per-token rescan memoized. Byte-identical
/// verdicts — pinned prefix-by-prefix against the fresh gate over a corpus that
/// includes every marker family, a close buried in a tool argument, and a
/// suffix longer than the partial-scan's tail window.
pub fn streamThinkGateScan(
    buf: []const u8,
    enable_thinking: bool,
    think_closed: bool,
    prompt_opened_think: bool,
    scan: *ThinkScan,
) StreamThinkGate {
    // Harmony (gpt_oss) decides before muse — both emit <|message|>, and only
    // the <|channel|> marker tells them apart.
    if (harmonyStreamVerdict(buf)) |v| return v;
    if (museStreamVerdict(buf)) |v| return v;
    {
        const ihead = inklingStripMessageHead(buf);
        if (std.mem.startsWith(u8, ihead, INKLING_TEXT_TAG)) return .flush_text;
        if (std.mem.startsWith(u8, ihead, INKLING_THINKING_TAG)) {
            return if (std.mem.indexOf(u8, ihead, INKLING_END_TAG) != null) .split_think else .hold_thinking;
        }
    }
    scan.advance(buf);
    const has_thinking = ((enable_thinking or prompt_opened_think) and !think_closed) or
        scan.saw_channel_thought or
        scan.saw_open or
        (std.mem.startsWith(u8, buf, "<|channel>") and buf.len < 18) or
        (std.mem.startsWith(u8, buf, "<think") and buf.len < 7) or
        endsWithPartialThinkOpen(buf);
    if (!has_thinking) return .flush_text;
    return if (scan.saw_channel_close or scan.close != null) .split_think else .hold_thinking;
}

pub fn streamThinkGate2(buf: []const u8, enable_thinking: bool, think_closed: bool, prompt_opened_think: bool) StreamThinkGate {
    // Harmony (gpt_oss) channels decide first — the muse arm below shares the
    // <|message|> marker and would otherwise claim this traffic.
    if (harmonyStreamVerdict(buf)) |v| return v;
    // Muse channels decide early — the recipient header is ordinary text and
    // must never flush unresolved.
    if (museStreamVerdict(buf)) |v| return v;
    // Inkling channels decide early: every marker is a SINGLE special token,
    // so a leading <|content_text|> is a whole marker (the flush path's
    // marker-token skip keeps it out of visible deltas), and a thinking
    // message holds until its <|end_message|> arrives.
    {
        const ihead = inklingStripMessageHead(buf);
        if (std.mem.startsWith(u8, ihead, INKLING_TEXT_TAG)) return .flush_text;
        if (std.mem.startsWith(u8, ihead, INKLING_THINKING_TAG)) {
            return if (std.mem.indexOf(u8, ihead, INKLING_END_TAG) != null) .split_think else .hold_thinking;
        }
    }
    const has_thinking = ((enable_thinking or prompt_opened_think) and !think_closed) or
        std.mem.indexOf(u8, buf, "<|channel>thought") != null or
        indexOfThinkOpenTag(buf, 0) != null or
        (std.mem.startsWith(u8, buf, "<|channel>") and buf.len < 18) or
        (std.mem.startsWith(u8, buf, "<think") and buf.len < 7) or
        // A partial opener at the buffer TAIL (mid-text re-opened channel
        // arriving token by token) must hold the flush — flushing leaks tag
        // fragments like a glued "thought".
        endsWithPartialThinkOpen(buf);
    if (!has_thinking) return .flush_text;
    const has_close = std.mem.indexOf(u8, buf, "<channel|>") != null or
        indexOfThinkCloseTag(buf, 0) != null;
    return if (has_close) .split_think else .hold_thinking;
}

/// End-of-stream classification for a think buffer that never saw a close tag:
/// is it reasoning, or was it content all along?
///
/// The streaming loop starts `in_think_block` from `enable_thinking`, i.e. it
/// ASSUMES the model begins inside a think block whenever thinking was
/// requested. That holds only for templates that pre-inject the opener. Gemma 4
/// renders a bare `<|turn>model\n` and lets the MODEL decide, so a turn it
/// answers directly ("391") carries no think markup at all — and the whole
/// answer was flushed as `reasoning_content` with EMPTY content, while the
/// non-streaming path (`splitThinkBlock`, no opener + no close + not
/// template-opened ⇒ content) returned it correctly.
///
/// So require POSITIVE evidence a block was open: the PROMPT opened one, or the
/// model emitted a literal opener. This is what makes the two paths agree.
pub fn streamTailIsReasoning(in_think_block: bool, prompt_opened_think: bool, saw_think_open: bool) bool {
    return in_think_block and (prompt_opened_think or saw_think_open);
}

/// The reasoning bytes not yet streamed, or null when there is nothing new.
///
/// The tools streaming path emits reasoning INCREMENTALLY (see the
/// `.hold_thinking` arm), so every later site that emits reasoning for the same
/// Streaming counterpart to the leading `trimStart(content, "\n ")` every
/// non-streaming split already applies.
///
/// The two surfaces disagreed on the BYTES of the same answer: LFM2.5's template
/// opens `<think>` unconditionally, so its first generated token is the newline
/// pair that follows the closer, and `splitThinkBlock` drops it while a stream
/// flushing tokens verbatim shipped it — `'One, two…'` non-streaming,
/// `'\n\nOne, two…'` streaming. Any model whose first visible
/// token is whitespace is in this class, not just LFM2.5.
///
/// Leading whitespace is the ONE thing a stream can still withhold: nothing
/// visible has been sent yet, so this is a suppression, never a retraction.
///
/// It suppresses a chunk that is ENTIRELY whitespace, never a prefix of one. A
/// partial trim would ship a logprobs entry whose `token` no longer appears in
/// `content` — the collector describes whole tokens and cannot split one — so
/// the two contracts are kept consistent by never cutting inside a token. A
/// single token mixing leading whitespace with visible text is the residual
/// case, and it keeps its whitespace on both surfaces' terms: the stream ships
/// it, and the caller flips `content_started` so nothing later is touched.
///
/// `content_started` must flip on the first NON-EMPTY emission, so a suppressed
/// whitespace token leaves the next one still eligible.
pub fn streamContentLead(chunk: []const u8, content_started: bool) []const u8 {
    if (content_started) return chunk;
    if (std.mem.trimStart(u8, chunk, "\n ").len == 0) return "";
    return chunk;
}

/// turn must send only the remainder or the client sees the thought twice.
///
/// `already >= reasoning.len` returns null rather than underflowing: the split
/// can SHRINK when a tool marker appears mid-thought and `trimLeakedToolMarkup`
/// cuts the tail. An SSE delta cannot be retracted, so the honest behavior is to
/// send nothing further — never to resend from the top.
pub fn unstreamedReasoning(reasoning: []const u8, already: usize) ?[]const u8 {
    if (already >= reasoning.len) return null;
    return reasoning[already..];
}

/// The part of an unclosed thought safe to stream: a close tag still arriving
/// in pieces (`</thi`) is held back until the next token decides it.
pub fn streamableReasoning(so_far: []const u8) []const u8 {
    return so_far[0 .. so_far.len - partialThinkCloseSuffixLen(so_far)];
}

// ── tests ──

test "streamShouldBufferForTools: Inkling invoke marker and NAME hold" {
    // The invoke marker is a single special token — once present, everything
    // from here is call payload and must never flush as content.
    try testing.expect(streamShouldBufferForTools("<|message_model|>bash<|content_invoke_tool_json|>"));
    try testing.expect(streamShouldBufferForTools("<|content_invoke_tool_json|>{\"name\":\"bash\""));
    // NAME hold: an Inkling boundary marker plus a bare identifier segment
    // could be the NAME right before the invoke marker — hold until
    // disambiguated (prose adds a space/punct within a token or two).
    try testing.expect(streamShouldBufferForTools("<|message_model|><|content_text|>bash"));
    try testing.expect(streamShouldBufferForTools("<|message_model|>get_time"));
    // Prose disambiguates → flows.
    try testing.expect(!streamShouldBufferForTools("<|content_text|>The answer"));
    // Completed message (empty segment after the end marker) → flows.
    try testing.expect(!streamShouldBufferForTools("<|content_text|>4<|end_message|>"));
    // A thinking message never trips the NAME hold (the think gate owns it).
    try testing.expect(!streamShouldBufferForTools("<|content_thinking|>Need the time"));
    // No Inkling markers → other families completely unaffected.
    try testing.expect(!streamShouldBufferForTools("bash"));
    try testing.expect(!streamShouldBufferForTools("plain prose mentioning get_time here"));
}

test "endsWithPartialThinkOpen holds back partial opener tails (pi GGUF leak repro)" {
    // The exact failure shape: `<|channel>` flushed as content before
    // "thought" arrived; the buffer tail was a partial opener at every step.
    try testing.expect(endsWithPartialThinkOpen("prose ends here.<|channel>"));
    try testing.expect(endsWithPartialThinkOpen("prose ends here.<|chan"));
    try testing.expect(endsWithPartialThinkOpen("prose ends here.<|channel>thoug"));
    try testing.expect(endsWithPartialThinkOpen("prose <"));
    try testing.expect(endsWithPartialThinkOpen("prose <think"));
    try testing.expect(endsWithPartialThinkOpen("<th"));
    // Prose and HTML-ish tags keep flowing.
    try testing.expect(!endsWithPartialThinkOpen("just prose."));
    try testing.expect(!endsWithPartialThinkOpen("prose <table"));
    try testing.expect(!endsWithPartialThinkOpen("a > b"));
    try testing.expect(!endsWithPartialThinkOpen(""));
}

test "streamShouldBufferForTools: empty + plain prose" {
    try testing.expect(!streamShouldBufferForTools(""));
    try testing.expect(!streamShouldBufferForTools("hello world"));
    try testing.expect(!streamShouldBufferForTools("Let me look at this."));
}

test "streamShouldBufferForTools: trailing partial prefixes of <tool" {
    // The exact regression from the screenshot: model emits `<tool` as a
    // single BPE token, server must buffer instead of streaming it as
    // content (so the next token can either complete the open or be flushed
    // together as harmless prose).
    try testing.expect(streamShouldBufferForTools("<"));
    try testing.expect(streamShouldBufferForTools("<t"));
    try testing.expect(streamShouldBufferForTools("<to"));
    try testing.expect(streamShouldBufferForTools("<too"));
    try testing.expect(streamShouldBufferForTools("<tool"));
    try testing.expect(streamShouldBufferForTools("Let me look at the CLI help:\n<tool"));
}

test "streamShouldBufferForTools: trailing partial prefixes of <|tool_call" {
    try testing.expect(streamShouldBufferForTools("<|"));
    try testing.expect(streamShouldBufferForTools("<|t"));
    try testing.expect(streamShouldBufferForTools("<|too"));
    try testing.expect(streamShouldBufferForTools("<|tool"));
    try testing.expect(streamShouldBufferForTools("<|tool_"));
    try testing.expect(streamShouldBufferForTools("<|tool_c"));
    try testing.expect(streamShouldBufferForTools("<|tool_cal"));
}

test "streamShouldBufferForTools: bare <tool> open (DSV4)" {
    try testing.expect(streamShouldBufferForTools("<tool>"));
    try testing.expect(streamShouldBufferForTools("<tool>{\"name\":\"shell\""));
    try testing.expect(streamShouldBufferForTools("<tool name=\"shell\">"));
    try testing.expect(streamShouldBufferForTools("hello <tool>{}"));
}

test "streamShouldBufferForTools: canonical <tool_call*> opens" {
    try testing.expect(streamShouldBufferForTools("<tool_call>"));
    try testing.expect(streamShouldBufferForTools("<tool_call>{\"name\":"));
    try testing.expect(streamShouldBufferForTools("<tool_calls>"));
    try testing.expect(streamShouldBufferForTools("<tool_request>"));
    try testing.expect(streamShouldBufferForTools("prose first then <tool_call>"));
}

test "streamShouldBufferForTools: Gemma 4 <|tool_call" {
    try testing.expect(streamShouldBufferForTools("<|tool_call>"));
    try testing.expect(streamShouldBufferForTools("hello <|tool_call name=\"x\">"));
}

test "streamShouldBufferForTools: false positives — <toolkit> etc." {
    // `<toolkit>` and `<toolbar>` have `<tool` as a prefix but the char
    // after is `k` / `b` — not a tag terminator. Must NOT buffer.
    try testing.expect(!streamShouldBufferForTools("here is <toolkit>"));
    try testing.expect(!streamShouldBufferForTools("here is <toolbar>"));
    try testing.expect(!streamShouldBufferForTools("HTML: <toolset>"));
    // But a later real `<tool>` should still trip detection:
    try testing.expect(streamShouldBufferForTools("HTML: <toolset> then <tool>{"));
}

test "streamShouldBufferForTools: raw JSON shape" {
    try testing.expect(streamShouldBufferForTools("{\"name\":\"shell\",\"arguments\":{}}"));
    try testing.expect(streamShouldBufferForTools("{\"name\":\"shell\""));
    // Just `{` or JSON without `"name"` shouldn't false-trigger.
    try testing.expect(!streamShouldBufferForTools("{"));
    try testing.expect(!streamShouldBufferForTools("{\"foo\":1}"));
}

test "streamThinkGate: template-opened thinking holds tag-free prose" {
    // The Claude Code leak class: with thinking on and the opener injected by
    // the template, the model's prose has NO tags — it must still be held,
    // never flushed as visible text.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("The user is asking about their system specs.", true, false));
    // …and split exactly when the close tag arrives.
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate("The user is asking.\n</think>\n\nI don't have direct access.", true, false));
}

test "streamThinkGate: after the split, prose flushes (hidden-answer guard)" {
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate("The visible answer streams normally.", true, true));
}

test "streamThinkGate: explicit openers hold even with thinking off or closed" {
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("Sure.<|channel>thought\nlet me reconsider", true, true));
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("<think>hmm", false, false));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate("<|channel>thought\nplan<channel|>done", false, false));
}

test "streamThinkGate: partial opener at the buffer tail holds" {
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("The answer is 391.\n<|channel>", false, false));
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("Done. <thi", false, false));
}

test "streamThinkGate: plain prose with thinking off flushes" {
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate("17 × 23 = 391.", false, false));
}

test "streamThinkGate: a prompt-opened block is held even with thinking OFF" {
    // LFM2.5 renders `…assistant\n<think>` UNCONDITIONALLY — there is no
    // enable_thinking branch in its template, so the model always reasons and
    // the opener is never in the OUTPUT. The gate saw tag-free prose with
    // thinking off and flushed the entire chain-of-thought as the answer
    // (live, streaming only — non-streaming strips it because the
    // `</think>` is present by the time the text is split).
    //
    // Whether the prompt ends inside a think block is a fact about the
    // RENDERED BYTES, not about our request flag, which is why it needs its
    // own argument rather than riding `enable_thinking`.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate2("The user wants me to say hi in 5 words.", false, false, true));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate2("The user wants 5 words.</think>Hi there, how are you?", false, false, true));

    // Containment, both directions. Same buffers with the prompt flag FALSE
    // behave exactly as before — every other family's template renders the
    // closed `<think></think>` signature (or no tag) when thinking is off, so
    // `promptTailOpensThink` is false for them and this term can never fire.
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate2("The user wants me to say hi in 5 words.", false, false, false));
    // And once the block has closed, a prompt-opened flag must not re-hold the
    // visible answer — that would hide every reply after the first.
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate2("Hi there, how are you?", false, true, true));
}

test "unstreamedReasoning: only the remainder, and never a resend" {
    try testing.expectEqualStrings("cde", unstreamedReasoning("abcde", 2).?);
    try testing.expectEqualStrings("abcde", unstreamedReasoning("abcde", 0).?);
    // Nothing new since the last delta.
    try testing.expect(unstreamedReasoning("abcde", 5) == null);
    // The split SHRANK (a tool marker appeared mid-thought and the leaked-markup
    // cut took the tail). Deltas cannot be retracted, so send nothing more —
    // resending from the top would duplicate the whole thought.
    try testing.expect(unstreamedReasoning("ab", 5) == null);
    try testing.expect(unstreamedReasoning("", 0) == null);
}

test "streamTailIsReasoning: an unopened block was content all along" {
    // Gemma 4 with enable_thinking:true answering "391" directly — no opener
    // in the prompt (its template renders a bare `<|turn>model\n`) and none in
    // the output. Streamed, the whole answer arrived as reasoning_content with
    // EMPTY content while non-streaming returned it as content (live
    // ). Content is the correct reading: nothing ever opened.
    try testing.expect(!streamTailIsReasoning(true, false, false));

    // A template-opened block that never closed IS reasoning — the truncated
    // thought must not leak into content (Qwen/LFM2.5 pre-inject the opener).
    try testing.expect(streamTailIsReasoning(true, true, false));
    // A model-emitted opener is the same evidence (Gemma's `<|channel>thought`,
    // Inkling's `<|content_thinking|>`), even with no prompt opener.
    try testing.expect(streamTailIsReasoning(true, false, true));
    // Already closed and split ⇒ the tail is the visible answer.
    try testing.expect(!streamTailIsReasoning(false, true, true));
}

test "streamThinkGate: the 3-arg form is the 4-arg form with no prompt opener" {
    // The old signature is kept as the no-prompt-opener case so every existing
    // call site and test pins the SAME behavior it always did.
    const cases = [_][]const u8{
        "The user is asking about their system specs.",
        "The user is asking.\n</think>\n\nI don't have direct access.",
        "<think>hmm",
        "17 × 23 = 391.",
        "The answer is 391.\n<|channel>",
    };
    for (cases) |c| {
        for ([_]bool{ true, false }) |think| {
            for ([_]bool{ true, false }) |closed| {
                try testing.expectEqual(streamThinkGate(c, think, closed), streamThinkGate2(c, think, closed, false));
            }
        }
    }
}

test "hy3 think: streamThinkGate splits on suffixed close and holds on a partial suffixed re-open" {
    // Template-opened thinking: reasoning streams in with no literal opener.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("partial reasoning", true, false));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate("reasoning</think:opensource>ans", true, false));
    // Mid-text re-open arriving token by token: the partial tail must hold the
    // flush (leaking "<think:opensou" into visible text is the tag-leak class).
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("visible<think:opensou", false, true));
    // Prose that merely mentions "<think about it" is NOT a partial opener.
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate("let's <think about it", false, true));
}

test "streamShouldBufferForTools: DSML marker and partial fullwidth prefixes hold" {
    try testing.expect(streamShouldBufferForTools("Sure.\n\n<｜DSML｜tool_calls>"));
    try testing.expect(streamShouldBufferForTools("<｜DSML｜invoke name=\"x\">"));
    // partial fullwidth-bar prefix at the tail — next token may complete it
    try testing.expect(streamShouldBufferForTools("answer text <｜DSML"));
    try testing.expect(streamShouldBufferForTools("answer text <｜"));
    // plain prose with a fullwidth bar elsewhere must NOT hold
    try testing.expect(!streamShouldBufferForTools("the ｜ character is fun"));
}

test "streamShouldBufferForTools: a DSML closer with no opener holds, and so does its split lead-in (#980)" {
    try testing.expect(streamShouldBufferForTools("Step 3 done.\n\n</｜DSML｜ calls>"));
    try testing.expect(streamShouldBufferForTools("Step 3 done.\n\n</｜DSML｜"));
    try testing.expect(streamShouldBufferForTools("Step 3 done.\n\n</"));
    try testing.expect(!streamShouldBufferForTools("a closing </div> tag"));
}

test "streamShouldBufferForTools: every prefix of a bare <invoke> call holds; prose releases" {
    const text = "<function_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\">ls</parameter>\n</invoke>\n</function_calls>";
    for ([_]usize{ 0, std.mem.indexOf(u8, text, "<invoke").? }) |start| {
        var end = start + 2;
        while (end <= text.len) : (end += 1) {
            testing.expect(streamShouldBufferForTools(text[start..end])) catch |e| {
                std.debug.print("gate flushed a prefix of an accepted call: '{s}'\n", .{text[start..end]});
                return e;
            };
        }
    }
    try testing.expect(!streamShouldBufferForTools("I <invoked> the <function_callback> helper"));
}

test "muse streaming gate: headers hold, self holds then splits, user splits" {
    // Growing bare header — never flush the ` to=…` text. The sub-"to="
    // prefixes are the live leak (" to" shipped as a text delta
    // on a thinking-off /v1/messages stream).
    try testing.expect(museStreamVerdict(" t") == StreamThinkGate.hold_thinking);
    try testing.expect(museStreamVerdict(" to") == StreamThinkGate.hold_thinking);
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate2(" to", true, false, false));
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate2(" to=se", true, false, false));
    // Reasoning in progress streams via hold_thinking; <|eom|> splits.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate2(" to=self<|message|>thinking…", true, false, false));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate2(" to=self<|message|>done<|eom|>", true, false, false));
    // Direct answer resolves to a split (the handler strips the header once).
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate2(" to=user<|message|>Hi", true, false, false));
    // A next-segment header mid-stream holds again.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate2(" to=self<|message|>d<|eom|><|start|>assistant to=w", true, false, false));
    // Prose that merely starts with to= releases at the first non-recipient byte.
    try testing.expect(museStreamVerdict("to=5 grams of flour") == null);
    // The memoized gate agrees on the muse arms.
    var scan = ThinkScan{};
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGateScan(" to=self<|message|>done<|eom|>", true, false, false, &scan));
}

test "muse streamShouldBufferForTools: tool headers hold, self/user release" {
    // Unresolved header could still name a tool — including sub-"to="
    // prefixes (the live " to" text-delta leak).
    try testing.expect(streamShouldBufferForTools(" t"));
    try testing.expect(streamShouldBufferForTools(" to"));
    try testing.expect(streamShouldBufferForTools(" to=weath"));
    try testing.expect(streamShouldBufferForTools(" to=self<|message|>r<|eom|><|start|>assistant to=wea"));
    // Resolved tool recipient keeps holding (body is ATEM payload).
    try testing.expect(streamShouldBufferForTools(" to=weather.get<|message|>"));
    // ATEM markup anywhere holds.
    try testing.expect(streamShouldBufferForTools("<atem:function_calls>"));
    try testing.expect(streamShouldBufferForTools("prefix <atem"));
    // Resolved self/user releases so reasoning/content can stream.
    try testing.expect(!streamShouldBufferForTools(" to=self<|message|>thinking about tools"));
    try testing.expect(!streamShouldBufferForTools(" to=user<|message|>plain answer"));
    // Ordinary prose that starts with to= releases at the first space.
    try testing.expect(!streamShouldBufferForTools("to=5 grams"));
}

test "ThinkScan: incremental gate matches the fresh gate at every prefix length" {
    // The gate is O(buffer) per token today (measured 47.99 us/token at 113 KB
    // with no close in the buffer), i.e. O(n^2) over a reasoning block. A
    // cursor makes it O(new bytes) — but only if it is EXACT, and the failure
    // a naive cursor introduces is a marker split across two token arrivals.
    // Feeding prefix-by-prefix IS that case, at every possible split point.
    const corpus = [_][]const u8{
        "plain visible prose with no markup at all",
        "<think>reasoning here</think>the answer",
        "<think:opensource>hy3 reasoning</think:opensource>answer",
        // A </think> inside a tool ARGUMENT is payload, not a block close.
        "<think>plan<tool_call>f<arg_key>k</arg_key><arg_value>closes with </think> inside</arg_value></tool_call>done</think>visible",
        "<|channel>thought hidden<channel|>visible",
        "prose where a < b and nothing else happens",
        "trailing partial opener at the end <thi",
        "trailing partial close </thin",
        // A suffix longer than the 32-byte tail window the partial-scan uses:
        // the fresh scan still recognizes it, so the cursor must too.
        "<think>x</think:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>tail",
        "<|message_thinking|>inkling thought<|end_message|>after",
        "<|message_text|>inkling visible",
    };
    const flags = [_][2]bool{ .{ true, false }, .{ false, false }, .{ false, true } };
    for (corpus) |full| {
        for (flags) |f| {
            const enable_thinking = f[0];
            const prompt_opened = f[1];
            var scan: ThinkScan = .{};
            var n: usize = 0;
            while (n <= full.len) : (n += 1) {
                const buf = full[0..n];
                const want = streamThinkGate2(buf, enable_thinking, false, prompt_opened);
                const got = streamThinkGateScan(buf, enable_thinking, false, prompt_opened, &scan);
                if (want != got) {
                    std.debug.print("prefix {d} of \"{s}\" (think={}, opened={}): {any} != {any}\n", .{ n, full, enable_thinking, prompt_opened, got, want });
                    return error.IncrementalGateDiverged;
                }
            }
        }
    }
}

test "ThinkScan: per-token work is FLAT in buffer size, not linear" {
    // The whole point. A smaller constant is not the fix — the re-examined
    // span must not grow with the buffer, or a 32K-token thought pays 8x what
    // the 113 KB bench measured.
    var scan: ThinkScan = .{};
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(testing.allocator);
    var max_span: usize = 0;
    var i: usize = 0;
    while (i < 4000) : (i += 1) {
        // A think block that never closes — the expensive arm of the bench.
        try buf.appendSlice(testing.allocator, if (i == 0) "<think>" else "token words and more ");
        _ = streamThinkGateScan(buf.items, true, false, false, &scan);
        if (i > 0 and scan.last_scan_span > max_span) max_span = scan.last_scan_span;
    }
    try testing.expect(buf.items.len > 60_000);
    // Bounded by one token's bytes plus the trailing partial-marker candidate.
    try testing.expect(max_span <= 64);

    // A growing partial marker at the tail is re-examined, and only it.
    var scan2: ThinkScan = .{};
    var buf2 = std.ArrayList(u8).empty;
    defer buf2.deinit(testing.allocator);
    try buf2.appendSlice(testing.allocator, "<think>");
    var j: usize = 0;
    while (j < 500) : (j += 1) try buf2.appendSlice(testing.allocator, "filler ");
    _ = streamThinkGateScan(buf2.items, true, false, false, &scan2);
    for ("</think:opensou") |c| {
        try buf2.append(testing.allocator, c);
        _ = streamThinkGateScan(buf2.items, true, false, false, &scan2);
        try testing.expect(scan2.last_scan_span <= 64);
    }
}

test "ThinkScan: reset() re-arms the cursor when the stream buffer is cleared" {
    // The buffer is NOT append-only — the stream loop trims and resets it at
    // every emit. A stale cursor then points past the end or into different
    // bytes, which is the second failure mode a naive cursor introduces.
    var scan: ThinkScan = .{};
    const first = "<think>reasoning</think>";
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGateScan(first, true, false, false, &scan));
    scan.reset();
    // Same scan struct, fresh (shorter) buffer: must not carry the old close.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGateScan("<think>more", true, false, false, &scan));
}

test "bench: streaming think gate scans BYTES, memoized vs fresh" {
    if (!@import("build_options").slow_tests) return error.SkipZigTest;
    // The 5.10 acceptance bar is that per-token work is FLAT in buffer size,
    // not merely smaller. Wall clock is unavailable in a hermetic test under
    // Zig 0.17 (clocks live under std.Io) and would be noise anyway — bytes
    // examined is the thing that was quadratic, and it is deterministic.
    // Mirrors the 4b.4 harness: 4000 tokens growing to ~104 KB.
    const a = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(a);
    var scan: ThinkScan = .{};
    try buf.appendSlice(a, "<think>");
    var memo_bytes: u64 = 0;
    var fresh_bytes: u64 = 0;
    var i: usize = 0;
    while (i < 4000) : (i += 1) {
        try buf.appendSlice(a, "some reasoning words here ");
        const g = streamThinkGateScan(buf.items, true, false, false, &scan);
        try testing.expectEqual(StreamThinkGate.hold_thinking, g);
        memo_bytes += scan.last_scan_span;
        // What the unmemoized gate re-reads every token: the whole buffer, and
        // more than once (open scan + close scan + two substring scans).
        fresh_bytes += buf.items.len;
    }
    std.debug.print(
        "[think-gate] {d} tokens to {d} KB: fresh >= {d} KB scanned, memoized {d} KB ({d}x)\n",
        .{ 4000, buf.items.len / 1024, fresh_bytes / 1024, memo_bytes / 1024, fresh_bytes / @max(memo_bytes, 1) },
    );
    // Linear-in-total, not quadratic: the memoized scan reads each byte a
    // bounded number of times.
    try testing.expect(memo_bytes < buf.items.len * 4);
}

test "streamContentLead never touches whitespace inside the answer" {
    // Only the LEAD is suppressed: once anything visible has been sent, a
    // newline token is ordinary content (markdown paragraphs, code blocks).
    try testing.expectEqualStrings("\n\n", streamContentLead("\n\n", true));
    try testing.expectEqualStrings("  x", streamContentLead("  x", true));
    // And a lead that is not whitespace is untouched.
    try testing.expectEqualStrings("Hello", streamContentLead("Hello", false));
    // A token is never cut in half: one carrying whitespace AND text rides
    // whole, so its logprobs entry still describes bytes that reached content.
    try testing.expectEqualStrings("\nHello", streamContentLead("\nHello", false));
}

test "streamShouldBufferForTools: MiniCPM5 <function name=...> open" {
    try testing.expect(streamShouldBufferForTools("<function name=\"shell\">"));
    try testing.expect(streamShouldBufferForTools("<function name=\"shell\">\n  <param name=\"command\">pwd"));
    try testing.expect(streamShouldBufferForTools("<function name=\"shell\">\n  <param name=\"command\">pwd</param></functio"));
    try testing.expect(streamShouldBufferForTools("Sure, running that:\n<function name=\"shell\">"));
}

test "streamShouldBufferForTools: EVERY growing prefix of <function holds" {
    // DERIVED, not enumerated. The previous version of this test was a second
    // hand-written copy of `tail_prefixes` and inherited its hole: `<funct` was
    // missing from BOTH, so a token boundary landing exactly there flushed the
    // fragment and leaked the rest of the tag (`<f` id 54303 + `unct` id 14185
    // decode to exactly `<funct` in this vocabulary — a real decomposition, not
    // a hypothetical). Generating the assertions from the marker string means a
    // rung missing from the array fails here even if nobody remembers to add an
    // assertion for it.
    const marker = "<function";
    var i: usize = 2;
    while (i <= marker.len) : (i += 1) {
        try testing.expect(streamShouldBufferForTools(marker[0..i]));
    }
}

test "streamShouldBufferForTools: MiniCPM5 opener holds while unresolved, releases when resolved-and-not-a-call" {
    // HOLD — the opener is either unresolved (no `>` yet, so the next token
    // could still complete a real call) or resolved WITH a quoted name.
    const hold = [_][]const u8{
        "<f",                        "<fu",
        "<fun",                      "<func",
        "<funct",                    "<functi",
        "<functio",                  "<function",
        "<function ",                "<function n",
        "<function na",              "<function name",
        "<function name=",           "<function name=\"",
        "<function name=\"shell",    "<function name=\"shell\">",
        // Parser-ACCEPTED oddity: miniCpm5AttrValue finds `name=` anywhere in
        // the opener, so this IS a real call. The gate must therefore hold it —
        // a gate stricter than its parser flushes a genuine call's opener as
        // content and then the end-of-stream parse emits the call anyway: leak
        // AND duplicate, the exact class this work exists to fix.
        "<function foo name=\"x\">",
    };
    for (hold) |buf| {
        testing.expect(streamShouldBufferForTools(buf)) catch |e| {
            std.debug.print("expected HOLD, got flush: '{s}'\n", .{buf});
            return e;
        };
    }

    // RELEASE — resolved (`>` seen) and demonstrably not a MiniCPM5 call, or
    // never this dialect at all. Holding these buys nothing and costs the whole
    // remainder of the response: the hold is monotonic, so one `<function foo>`
    // in prose silences every later token of a tool-enabled turn.
    const release = [_][]const u8{
        "<functional",
        "<function-like>",
        "<function>",
        "<function foo>",
        "<function >",
        "<function name=unquoted>",
        // Empty name: the parser rejects it (`fn_name.len == 0` → skip), so the
        // gate releases too. The strictly-earlier prefix `<function name="` is
        // pinned in the hold set above, so the unresolved form still holds.
        "<function name=\"\">",
        "Use a <function foo> block to wrap it, then carry on writing prose.",
    };
    for (release) |buf| {
        testing.expect(!streamShouldBufferForTools(buf)) catch |e| {
            std.debug.print("expected RELEASE, got hold: '{s}'\n", .{buf});
            return e;
        };
    }
}

test "streamShouldBufferForTools: false positive — bare <function> with no attribute" {
    try testing.expect(!streamShouldBufferForTools("here is <function>"));
    try testing.expect(!streamShouldBufferForTools("Wrap it in a <functional> block"));
}

test "streamShouldBufferForTools: the gate is a strict SUPERSET of parseMiniCpm5ToolCalls" {
    // The load-bearing invariant, checked by DERIVATION rather than by a
    // hand-listed set: for every text the parser turns into a real call, every
    // prefix from the marker onward must hold. Any prefix that flushed would be
    // content already on the wire (SSE is append-only) followed by the same call
    // arriving properly at end of stream.
    const accepted = [_][]const u8{
        "<function name=\"shell\">\n  <param name=\"command\">pwd</param>\n</function>",
        "<function name=\"get_weather\">\n  <param name=\"city\">Paris</param>\n  <param name=\"unit\">c</param>\n</function>",
        "<function foo name=\"x\"><param name=\"a\">1</param></function>",
    };
    for (accepted) |text| {
        const calls = (try parseToolCalls(testing.allocator, text)) orelse {
            std.debug.print("fixture is not parser-accepted: '{s}'\n", .{text});
            return error.ExpectedToolCall;
        };
        defer {
            for (calls) |tc| {
                testing.allocator.free(tc.name);
                testing.allocator.free(tc.arguments);
            }
            testing.allocator.free(calls);
        }
        const start = std.mem.indexOf(u8, text, "<function").?;
        var end = start + 2; // shortest rung the ladder covers
        while (end <= text.len) : (end += 1) {
            testing.expect(streamShouldBufferForTools(text[start..end])) catch |e| {
                std.debug.print("gate flushed a prefix of an ACCEPTED call: '{s}'\n", .{text[start..end]});
                return e;
            };
        }
    }
}
