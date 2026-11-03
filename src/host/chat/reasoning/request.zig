//! reasoning from history, think bound, tool force, muse header, resolveReasoningProtocol; effort, thinking
//! resolution, schema thinking policy

const std = @import("std");
const tokenizer_mod = @import("../../../sdk/tokenizer/tokenizer.zig");
const request_mod = @import("../../../sdk/request/request.zig");
const generation_settings = @import("../../api/generation_settings.zig");
const chat_mod = @import("../chat.zig");
const rp_mod = @import("../../../sdk/grammar/protocol.zig");
const abi = @import("../../../abi/plugin.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const log = @import("../../../sdk/util/log.zig");
const responses_mod = @import("../../api/responses/model.zig");
const model_registry_mod = @import("../../registry/registry.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const LoadedModel = model_registry_mod.LoadedModel;
const http_server = @import("../../http/server.zig");

const decodeTokens = @import("../../api/tokenize.zig").decodeTokens;

/// Assistant-history reasoning field on an incoming chat message:
/// `reasoning_content` (our own SSE/vLLM field, what pi rounds-trips) with
/// `reasoning` (the vLLM request spelling laguna's template reads first) as
/// the fallback. Non-empty strings only — an empty string would render an
/// empty <think></think> block, which is exactly the nothink signature this
/// field exists to avoid.
pub fn messageReasoningFromObj(obj: std.json.ObjectMap) ?[]const u8 {
    inline for (.{ "reasoning_content", "reasoning" }) |key| {
        if (obj.get(key)) |v| {
            if (v == .string and v.string.len > 0) return v.string;
        }
    }
    return null;
}

/// True when the rendered generation prompt ends inside a template-opened
/// think block (Qwen 3.5/3.6 render `…assistant\n<think>\n` when thinking is
/// on). Decodes the last few prompt tokens — cheap, engine-agnostic, and
/// independent of the tokenize cache. Drives the unclosed-think split policy
/// in `chat.splitThinkBlock` so a length-truncated thought never leaks into
/// visible content.
pub fn promptOpensThink(
    allocator: std.mem.Allocator,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
) bool {
    if (prompt_ids.len == 0) return false;
    const n = @min(prompt_ids.len, 8);
    const tail = decodeTokens(allocator, lm, tok, prompt_ids[prompt_ids.len - n ..], false) catch return false;
    defer allocator.free(tail);
    return chat_mod.promptTailOpensThink(tail);
}

/// One token id for `text`, or null when the tokenizer spells it in pieces.
fn atomicTokenId(allocator: std.mem.Allocator, tok: *const Tokenizer, text: []const u8) ?u32 {
    const ids = tok.encode(allocator, text) catch return null;
    defer allocator.free(ids);
    return if (ids.len == 1) ids[0] else null;
}

/// The in-stream thinking bound needs atomic think markers (bare `<think>`
/// family, or a pack that aliases its own markers onto it).
fn thinkMarkersAtomic(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer) bool {
    if (lm.transformer == null) return false;
    return tok.marker_closers != null or atomicTokenId(allocator, tok, chat_mod.BARE_THINK_CLOSER) != null;
}

/// Qwen's own thinking-budget recipe: the model is told time is up, then the
/// closer is committed and the answer follows.
const THINK_BOUND_EARLY_STOP = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n";

/// No budget (< 0) still arms the bound at an unreachable limit: it exists to
/// close a thought the loop guard convicts (`scheduler.loopRecoveryAction`).
fn thinkBoundBudget(budget: i32) u32 {
    return if (budget < 0) std.math.maxInt(u32) else @intCast(budget);
}

/// Arm a decode-time thinking bound for this request, or null when it cannot
/// close a thought (thinking off, markers not atomic). `forced` is
/// allocated; the caller frees it after generation.
pub fn armThinkBound(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer, prompt_ids: []const u32, enable_thinking: bool, budget: i32) ?request_mod.ThinkBound {
    if (!enable_thinking or lm.transformer == null) return null;
    const closer = promptOpenerMarkerCloser(allocator, lm, tok, prompt_ids) orelse
        atomicTokenId(allocator, tok, chat_mod.BARE_THINK_CLOSER) orelse {
        if (budget >= 0) log.info("  reasoning budget {d}: think markers are not atomic tokens, delivery cap only\n", .{budget});
        return null;
    };
    const opener = atomicTokenId(allocator, tok, chat_mod.BARE_THINK_OPENER);
    const opened = promptOpensThink(allocator, lm, tok, prompt_ids);
    if (opener == null and !opened) return null;

    const stop_ids = tok.encode(allocator, THINK_BOUND_EARLY_STOP) catch return null;
    defer allocator.free(stop_ids);
    const sep_ids = tok.encode(allocator, "\n\n") catch return null;
    defer allocator.free(sep_ids);
    const forced = allocator.alloc(u32, stop_ids.len + 1 + sep_ids.len) catch return null;
    @memcpy(forced[0..stop_ids.len], stop_ids);
    forced[stop_ids.len] = closer;
    @memcpy(forced[stop_ids.len + 1 ..], sep_ids);
    if (budget >= 0) log.info("  reasoning budget {d}: enforced in-stream\n", .{budget});
    return .{ .budget = thinkBoundBudget(budget), .opener_id = opener, .closer_id = closer, .forced = forced, .in_think = opened };
}

/// Arm the forced tool-call opener for a `tool_choice` that obliges a call, or
/// null when the dialect has no opener we can spell or the thought cannot be
/// tracked. `forced` is allocated; the caller frees it after generation.
pub fn armToolForce(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer, prompt_ids: []const u32, forced_tool: ?chat_mod.ForcedTool) ?request_mod.ToolForce {
    const ft = forced_tool orelse return null;
    // Armed wherever the host's tokenizer is the model's (the Neural Engine too), so an engine that cannot force
    // a call refuses the request by name.
    if (lm.transformer == null and (lm.engine == null or lm.engine.?.caps().owns_tokenizer)) return null;
    const cc = lm.chat_config orelse return null;
    const text = (chat_mod.forcedToolOpener(allocator, cc.chat_template, ft) catch return null) orelse return null;
    defer allocator.free(text);
    const opened = promptOpensThink(allocator, lm, tok, prompt_ids);
    const closer = atomicTokenId(allocator, tok, chat_mod.BARE_THINK_CLOSER);
    if (opened and closer == null) return null;
    const opener = tok.encode(allocator, text) catch return null;
    defer allocator.free(opener);
    const sep = tok.encode(allocator, "\n\n") catch return null;
    defer allocator.free(sep);
    const head: usize = if (opened) 1 + sep.len else 0;
    const forced = allocator.alloc(u32, head + opener.len) catch return null;
    if (opened) {
        forced[0] = closer.?;
        @memcpy(forced[1..head], sep);
    }
    @memcpy(forced[head..], opener);
    log.info("  tool_choice: call opener enforced in-stream\n", .{});
    return .{ .closer_id = closer, .forced = forced, .opener_at = head, .in_think = opened };
}

/// The reasoning channels the model's plugin declares (`ChatFacts.reasoning_dialect`); unset off the plugin path.
pub fn reasoningDialect(c: *const ModelConfig) abi.ReasoningDialect {
    return if (c.plugin) |b| b.parsed.chat.reasoning_dialect else .unset;
}

pub fn promptOpensMuseHeader(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer, prompt_ids: []const u32) bool {
    const c = lm.config orelse return false;
    if (reasoningDialect(c) != .muse or prompt_ids.len == 0) return false;
    const n = @min(prompt_ids.len, 8);
    const tail = decodeTokens(allocator, lm, tok, prompt_ids[prompt_ids.len - n ..], false) catch return false;
    defer allocator.free(tail);
    return chat_mod.promptTailOpensMuseHeader(tail);
}

/// A generated tag needs one unambiguous literal spelling in the template.
/// Dynamic or conflicting spellings retain the unsupported-format fallback.
fn templateThinkOpener(template: []const u8) ?[]const u8 {
    var found: ?[]const u8 = null;
    var offset: usize = 0;
    while (std.mem.indexOfPos(u8, template, offset, "<think")) |start| {
        const len = chat_mod.thinkOpenTagLenAt(template[start..]) orelse return null;
        const tag = template[start..][0..len];
        if (found) |previous| {
            if (!std.mem.eql(u8, previous, tag)) return null;
        }
        found = tag;
        offset = start + len;
    }
    return found;
}

/// The aliased atomic closer for the last non-whitespace prompt token, when
/// that token is a K2 think opener (`Tokenizer.markerCloserFor`).
fn promptOpenerMarkerCloser(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer, prompt_ids: []const u32) ?u32 {
    if (tok.marker_closers == null) return null;
    var i = prompt_ids.len;
    while (i > 0 and prompt_ids.len - i < 8) {
        i -= 1;
        const id = prompt_ids[i];
        if (tok.markerCloserFor(id)) |closer| return closer;
        const text = decodeTokens(allocator, lm, tok, prompt_ids[i..][0..1], false) catch return null;
        defer allocator.free(text);
        if (std.mem.trim(u8, text, "\n\r\t ").len != 0) return null;
    }
    return null;
}

/// Resolve a reasoning protocol from the rendered prompt and template. Tokenizer
/// indexes and exact recovery suffixes are cached on the loaded model.
pub fn resolveReasoningProtocol(
    allocator: std.mem.Allocator,
    io: std.Io,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    prompt_ids: []const u32,
    proto_out: *rp_mod.Protocol,
    allow_reasoning: bool,
) bool {
    if (prompt_ids.len == 0 or lm.transformer == null) return false;
    const tail_n = @min(prompt_ids.len, 64);
    const tail = decodeTokens(allocator, lm, tok, prompt_ids[prompt_ids.len - tail_n ..], false) catch return false;
    defer allocator.free(tail);
    const trimmed = std.mem.trimEnd(u8, tail, "\n\r\t ");

    const template = chat_config.chat_template;
    const kind: rp_mod.Kind = if (std.mem.indexOf(u8, template, "<|content_thinking|>") != null)
        .inkling
    else if (std.mem.indexOf(u8, template, "<|channel|>") != null)
        .harmony
    else if (std.mem.indexOf(u8, template, "<|eom|>") != null)
        .muse
    else if (std.mem.indexOf(u8, template, "<|channel>") != null)
        .gemma
    else
        .bare_think;
    proto_out.* = .{ .kind = kind };
    const proto = proto_out;
    if (kind != .bare_think) {
        proto.configureChannels();
        if (!proto.startFromPrompt(tail)) return false;
    } else {
        switch (chat_mod.promptThinkTailClass(trimmed)) {
            .bare => {
                proto.kind = .bare_think;
                if (!proto.setCloser(chat_mod.BARE_THINK_CLOSER)) return false;
                // K2 decodes its openers as `<think>` (alias); the prompt's
                // opener token names the pack's own closer, and only that
                // spelling is the boundary (a literal `</think>` in the
                // reasoning is text).
                if (promptOpenerMarkerCloser(allocator, lm, tok, prompt_ids)) |id| {
                    const text = tok.id_to_token.get(id) orelse return false;
                    if (!proto.setCloser(text)) return false;
                }
            },
            .suffixed => |suffix| {
                proto.kind = .suffixed_think;
                var closer_buf: [32]u8 = undefined;
                const closer = std.fmt.bufPrint(&closer_buf, "</think:{s}>", .{suffix}) catch return false;
                if (!proto.setCloser(closer)) return false;
            },
            .none => {
                const opener = templateThinkOpener(template) orelse return false;
                proto.kind = if (std.mem.eql(u8, opener, rp_mod.BARE_THINK_OPENER)) .bare_think else .suffixed_think;
                var closer_buf: [rp_mod.MAX_MARKER_BYTES]u8 = undefined;
                const close = std.fmt.bufPrint(&closer_buf, "</{s}", .{opener[1..]}) catch return false;
                if (!proto.setOpener(opener) or !proto.setCloser(close)) return false;
            },
        }
    }

    var final_header: [rp_mod.MAX_MARKER_BYTES]u8 = undefined;
    if (!allow_reasoning and !proto.finalOnly(&final_header)) return false;

    const closer = lm.reasoningMarker(allocator, io, proto.closerText()) catch return false;
    proto.closer_atomic = closer.atomic;
    proto.closer_suffix = closer.suffix;
    proto.closer_suffix_buf = closer.tokens;
    proto.closer_span_candidates = closer.close_candidates;
    if (closer.atomic == null) {
        const run = closer.suffix[0];
        if (run.len == 0 or !proto.setForced(closer.tokens[run.offset..][0..run.len])) return false;
    }
    if (proto.openerText()) |text| {
        const opener = lm.reasoningMarker(allocator, io, text) catch return false;
        proto.opener_atomic = opener.atomic;
        proto.opener_suffix = opener.suffix;
        proto.opener_suffix_buf = opener.tokens;
        proto.opener_candidates = opener.open_candidates;
        if (opener.atomic == null and opener.open_candidates.len == 0) return false;
    }
    for (proto.headers[0..proto.header_len]) |*rule| {
        const marker = lm.reasoningMarker(allocator, io, rule.text) catch return false;
        if (marker.suffix[0].len == 0) return false;
        rule.text = marker.text;
        rule.candidates = marker.open_candidates;
        rule.suffix = marker.suffix;
        rule.tokens = marker.tokens;
    }
    // Structural special ids remain separate from JSON token bytes.
    if (proto.closer_atomic) |id| {
        proto.specials[proto.special_len] = .{ .id = id, .text = closer.text };
        proto.special_len += 1;
    }
    for ([_][]const u8{
        "<|start|>",         "<|channel|>",          "<|message|>",      "<|channel>",      "<channel|>",
        "<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|end_message|>", "<|end|>",
        "<|eom|>",
    }) |text| {
        if (tok.specialTokenId(text)) |id| {
            proto.specials[proto.special_len] = .{ .id = id, .text = text };
            proto.special_len += 1;
        }
    }
    return true;
}

/// `effort` is the client's raw string, borrowed from the parsed request JSON
/// (which outlives the handler) — dsv4-family templates map it into the
/// render via `chat.dsv4EffortFor`; every other consumer only reads
/// enable/budget.
const ReasoningEffort = struct { enable: bool, budget: i32, effort: ?[]const u8 = null };

/// OpenAI-standard `reasoning_effort` on chat/completions (none | minimal |
/// low | medium | high | xhigh — values are model-dependent, so unknown
/// strings still enable). "none" is an explicit off (the gpt-5.1 default
/// spelling). Absent or non-string → null: the vendor `enable_thinking` bool
/// stays in charge and existing clients see zero behavior change.
/// The effort word is the ONLY lever when the template reads it and no
/// decode-time bound can enforce a budget derived from it (a budget that is
/// only a delivery cap fights the word: `chat.templateConsumesEffort`).
pub fn effortWordOnly(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer) bool {
    const cc = lm.chat_config orelse return false;
    return chat_mod.templateConsumesEffort(cc.chat_template) and !thinkMarkersAtomic(allocator, lm, tok);
}

/// A request that thinks without naming an effort renders as "low" on the
/// Qwen3.8 family (`chat.qwen38EffortFor`), so it gets low's budget too; the
/// "keep it brief" preamble alone does not stop a model (a pi turn on a 2-bit
/// Bonsai thought for 6k tokens). `--reasoning-budget` still wins.
fn implicitEffortBudget(template: []const u8, markers_atomic: bool, default_budget: i32) i32 {
    if (default_budget >= 0 or !markers_atomic or !chat_mod.isQwen38EffortTemplate(template)) return default_budget;
    return responses_mod.effortBudget("low", default_budget);
}

pub fn implicitEffortBudgetFor(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer, default_budget: i32) i32 {
    const cc = lm.chat_config orelse return default_budget;
    return implicitEffortBudget(cc.chat_template, thinkMarkersAtomic(allocator, lm, tok), default_budget);
}

/// The request's `--reasoning-budget`: the generation-defaults budget rule replaces the flag,
/// and effort words map against it exactly as they map against the flag.
pub fn defaultReasoningBudget(gen: generation_settings.Profile) i32 {
    return gen.value(i32, .reasoning_budget) orelse http_server.serverState().server_config.default_reasoning_budget;
}

/// A forced budget rule replaces whatever the request and its effort word decided.
pub fn forcedReasoningBudget(gen: generation_settings.Profile, budget: i32) i32 {
    return if (gen.forced(.reasoning_budget)) gen.value(i32, .reasoning_budget).? else budget;
}

/// A request field at the top level, else inside vLLM's `chat_template_kwargs` object.
fn requestField(root: std.json.ObjectMap, name: []const u8) ?std.json.Value {
    if (root.get(name)) |v| return v;
    const kw = root.get("chat_template_kwargs") orelse return null;
    return if (kw == .object) kw.object.get(name) else null;
}

fn parseReasoningEffort(root: std.json.ObjectMap, default_budget: i32, template_consumes_effort: bool) ?ReasoningEffort {
    const v = requestField(root, "reasoning_effort") orelse return null;
    if (v != .string) return null;
    return reasoningEffortFromWord(v.string, default_budget, template_consumes_effort);
}

/// One effort word → one thinking config, whatever field carried the word —
/// OpenAI's flat `reasoning_effort` and Anthropic's `output_config.effort`
/// must not drift on what "low" means.
pub fn reasoningEffortFromWord(word: []const u8, default_budget: i32, template_consumes_effort: bool) ReasoningEffort {
    if (std.mem.eql(u8, word, "none")) return .{ .enable = false, .budget = default_budget, .effort = word };
    // Where the TEMPLATE reads the effort word, the word is the behavioral
    // lever and a budget derived from the same string is pure display
    // truncation on top of it (`chat.templateConsumesEffort`). An explicit
    // `--reasoning-budget` still rides `default_budget`.
    const budget = if (template_consumes_effort)
        default_budget
    else
        responses_mod.effortBudget(word, default_budget);
    return .{ .enable = true, .budget = budget, .effort = word };
}

/// Anthropic `output_config` — the 2026 spelling Claude Code sends: `effort`
/// in place of the older `thinking` budget object, and `format` where this
/// surface previously had no structured-output request at all. Ignored, it
/// served every Claude Code request at the arch default with an UNLIMITED
/// thinking budget (8-minute retries at 16k thinking tokens
/// each) and answered `json_schema` requests with markdown prose the client
/// rejects — which is what fed the retry loop.
const AnthropicOutputConfig = struct {
    effort: ?[]const u8 = null,
    /// `format.schema` when `format.type == "json_schema"`; null otherwise.
    schema: ?std.json.Value = null,
    /// `format.type == "json_schema"` with no object schema to enforce.
    schema_invalid: bool = false,
};

pub fn parseAnthropicOutputConfig(root: std.json.ObjectMap) AnthropicOutputConfig {
    var out: AnthropicOutputConfig = .{};
    const oc = root.get("output_config") orelse return out;
    if (oc != .object) return out;
    if (oc.object.get("effort")) |e| {
        if (e == .string) out.effort = e.string;
    }
    if (oc.object.get("format")) |f| {
        if (f == .object) {
            const ftype = if (f.object.get("type")) |t| (if (t == .string) t.string else "") else "";
            if (std.mem.eql(u8, ftype, "json_schema")) {
                out.schema = f.object.get("schema");
                if (out.schema == null or out.schema.? != .object) {
                    out.schema = null;
                    out.schema_invalid = true;
                }
            }
        }
    }
    return out;
}

fn thinkingFrom(enable: ?bool, effort: ?ReasoningEffort, fallback: bool) bool {
    if (enable == null and effort == null) return fallback;
    return (enable orelse false) or (if (effort) |e| e.enable else false);
}

/// The model's effort word fills in only when thinking ended up on and the request named none.
fn modelEffortIfThinking(model: ?ReasoningEffort, thinking: bool) ?ReasoningEffort {
    const m = model orelse return null;
    return if (thinking and m.enable) m else null;
}

const ChatThinking = struct { enable: bool, effort: ?ReasoningEffort };

/// An EXPLICIT client signal wins (`enable_thinking`, or an effort word, "none" being an
/// explicit OFF; the two stay OR'd). The generation-defaults rules (the model's
/// `chat_template_kwargs` thinking keys among them) act as a default request, consulted only
/// when the request names neither; then the arch. A forced thinking or effort rule replaces the
/// request's switches.
pub fn resolveThinking(gen: generation_settings.Profile, enable: ?bool, request: ?ReasoningEffort, arch_default: bool, default_budget: i32, word_only: bool) ChatThinking {
    const ignore = gen.forced(.enable_thinking) or gen.forced(.reasoning_effort);
    const req_effort = if (ignore) null else request;
    const model: ?ReasoningEffort = if (gen.value(generation_settings.Effort, .reasoning_effort)) |e|
        reasoningEffortFromWord(@tagName(e), default_budget, word_only)
    else
        null;
    const on = if (gen.forced(.enable_thinking))
        gen.value(bool, .enable_thinking).?
    else if (gen.forced(.reasoning_effort))
        model.?.enable
    else
        thinkingFrom(enable, req_effort, thinkingFrom(gen.value(bool, .enable_thinking), model, arch_default));
    return .{ .enable = on, .effort = req_effort orelse modelEffortIfThinking(model, on) };
}

pub fn resolveChatThinking(root: std.json.ObjectMap, gen: generation_settings.Profile, arch_default: bool, default_budget: i32, word_only: bool) ChatThinking {
    const enable: ?bool = if (requestField(root, "enable_thinking")) |v| (if (v == .bool) v.bool else null) else null;
    return resolveThinking(gen, enable, parseReasoningEffort(root, default_budget, word_only), arch_default, default_budget, word_only);
}

const SchemaThinkingPolicy = enum {
    no_mask,
    token_zero,
    deferred,
    fallback_thinking_off,
};

/// A schema grammar is safe with thinking only when generation can hand the
/// channel machinery to a resolved reasoning protocol (a prompt-opened think
/// block whose close the tokenizer can represent, or — for think-tag
/// templates — the unresolved opener choice). Every other masked surface
/// keeps the upstream thinking-off safety net. A finite response-side
/// reasoning budget also keeps that fallback because it can end the visible
/// think block before generation reaches the grammar boundary. Tools still
/// skip the mask.
pub fn schemaMasksThinking(
    has_schema: bool,
    has_tools: bool,
    enable_thinking: bool,
    has_finite_reasoning_budget: bool,
    protocol_ready: bool,
) SchemaThinkingPolicy {
    if (!has_schema or has_tools) return .no_mask;
    if (!enable_thinking) return .token_zero;
    if (has_finite_reasoning_budget) return .fallback_thinking_off;
    if (protocol_ready) return .deferred;
    return .fallback_thinking_off;
}

// ── tests ──

test "implicitEffortBudget: silence on the Qwen3.8 family gets low's budget" {
    const qwen38 = "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}";
    try std.testing.expectEqual(@as(i32, 2048), implicitEffortBudget(qwen38, true, -1));
    try std.testing.expectEqual(@as(i32, 512), implicitEffortBudget(qwen38, true, 512)); // --reasoning-budget wins
    try std.testing.expectEqual(@as(i32, -1), implicitEffortBudget(qwen38, false, -1)); // no bound to arm
    try std.testing.expectEqual(@as(i32, -1), implicitEffortBudget("{{ reasoning_effort }}", true, -1));
}

test "parseReasoningEffort: standard chat reasoning_effort opt-in maps to thinking + budget" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, expect: ?ReasoningEffort }{
        // Absent or wrong type → null: the vendor enable_thinking flag stays
        // in charge and nothing changes for existing clients.
        .{ .body = "{}", .expect = null },
        .{ .body = "{\"reasoning_effort\":42}", .expect = null },
        // "none" is an explicit OFF (gpt-5.1 default), never an enable.
        .{ .body = "{\"reasoning_effort\":\"none\"}", .expect = .{ .enable = false, .budget = -1 } },
        // Known efforts enable thinking with the shared Responses budget map.
        .{ .body = "{\"reasoning_effort\":\"minimal\"}", .expect = .{ .enable = true, .budget = 1024 } },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .expect = .{ .enable = true, .budget = 2048 } },
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .expect = .{ .enable = true, .budget = 8192 } },
        .{ .body = "{\"reasoning_effort\":\"high\"}", .expect = .{ .enable = true, .budget = -1 } },
        // Unknown efforts (xhigh, future values) enable with the default
        // budget — spec values are model-dependent, never reject them.
        .{ .body = "{\"reasoning_effort\":\"xhigh\"}", .expect = .{ .enable = true, .budget = -1 } },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = parseReasoningEffort(parsed.value.object, -1, false);
        if (case.expect) |want| {
            try std.testing.expectEqual(want.enable, got.?.enable);
            try std.testing.expectEqual(want.budget, got.?.budget);
            // The raw client string always rides along (dsv4 templates map it
            // into the render); it is the request body's own bytes.
            const v = parsed.value.object.get("reasoning_effort").?;
            try std.testing.expectEqualStrings(v.string, got.?.effort.?);
        } else {
            try std.testing.expect(got == null);
        }
    }
}

test "parseReasoningEffort: a template that READS the effort word gets no budget from it unless a decode-time bound can enforce it" {
    // Two levers, one word. Where the template acts on `reasoning_effort` the
    // word already shortens the THOUGHT; a budget that is only a delivery cap
    // fights it: pi asking Qwen3.8 for `medium` got the model's unguided (long)
    // thinking AND a 2048-token cut, followed by 25.9k tokens of invisible
    // generation. The third argument is `effortWordOnly`:
    // true when the template reads the word AND `armThinkBound` cannot close
    // the thought at the budget (non-atomic markers). With atomic markers the
    // budget is enforced in-stream and the word maps like everywhere else.
    // An EXPLICIT cap is someone asking on purpose and still applies — that is
    // `default_budget` here, which carries `--reasoning-budget`.
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, consumes: bool, default_budget: i32, want: i32 }{
        // Consuming template (qwen3.8 / dsv4 / inkling): no effort-derived cap.
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .consumes = true, .default_budget = -1, .want = -1 },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .consumes = true, .default_budget = -1, .want = -1 },
        .{ .body = "{\"reasoning_effort\":\"high\"}", .consumes = true, .default_budget = -1, .want = -1 },
        // ...but `--reasoning-budget` still binds there.
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .consumes = true, .default_budget = 4096, .want = 4096 },
        // Non-consuming template (gemma 4, LFM2.5, Qwen3.5/3.6, muse, laguna,
        // Ling): the effort word reaches no template, so the budget is the ONLY
        // thing it does and every mapping stays exactly as it shipped.
        .{ .body = "{\"reasoning_effort\":\"minimal\"}", .consumes = false, .default_budget = -1, .want = 1024 },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .consumes = false, .default_budget = -1, .want = 2048 },
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .consumes = false, .default_budget = -1, .want = 8192 },
        // high and xhigh are uncapped: pi's own ladder is 2048 / 8192 / 16384.
        .{ .body = "{\"reasoning_effort\":\"high\"}", .consumes = false, .default_budget = -1, .want = -1 },
        .{ .body = "{\"reasoning_effort\":\"xhigh\"}", .consumes = false, .default_budget = -1, .want = -1 },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = parseReasoningEffort(parsed.value.object, case.default_budget, case.consumes).?;
        try std.testing.expect(got.enable);
        try std.testing.expectEqual(case.want, got.budget);
        // The raw string rides along either way — the template still renders it.
        try std.testing.expectEqualStrings(parsed.value.object.get("reasoning_effort").?.string, got.effort.?);
    }
}

test "parseAnthropicOutputConfig: Claude Code's shape yields effort AND the schema" {
    const allocator = std.testing.allocator;
    // Verbatim shape from the live capture (schema abbreviated).
    const body =
        \\{"output_config":{"effort":"high","format":{"type":"json_schema","schema":{"type":"object","properties":{"title":{"type":"string"}}}}}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const oc = parseAnthropicOutputConfig(parsed.value.object);
    try std.testing.expectEqualStrings("high", oc.effort.?);
    try std.testing.expect(oc.schema != null);
    try std.testing.expect(oc.schema.? == .object);
}

test "parseAnthropicOutputConfig: absent, non-object, and non-schema shapes stay null" {
    const allocator = std.testing.allocator;
    const cases = [_][]const u8{
        "{}",
        \\{"output_config":"high"}
        ,
        // effort alone — no format means no schema, never a crash.
        \\{"output_config":{"effort":"low"}}
        ,
        // a format we don't serve is not a schema request.
        \\{"output_config":{"format":{"type":"text"}}}
        ,
    };
    for (cases) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
        defer parsed.deinit();
        const oc = parseAnthropicOutputConfig(parsed.value.object);
        try std.testing.expect(oc.schema == null);
        try std.testing.expect(!oc.schema_invalid);
    }
    // json_schema declared with nothing to enforce is flagged, never a silent fall-open.
    for ([_][]const u8{
        \\{"output_config":{"format":{"type":"json_schema"}}}
        ,
        \\{"output_config":{"format":{"type":"json_schema","schema":"nope"}}}
        ,
    }) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
        defer parsed.deinit();
        const oc = parseAnthropicOutputConfig(parsed.value.object);
        try std.testing.expect(oc.schema == null and oc.schema_invalid);
    }
    // The effort-alone case still carries its word.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator,
        \\{"output_config":{"effort":"low"}}
    , .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("low", parseAnthropicOutputConfig(parsed.value.object).effort.?);
}

test "reasoningEffortFromWord: none disables, words budget exactly like the OpenAI surface" {
    // "none" is an explicit off with the word kept for the template.
    const off = reasoningEffortFromWord("none", -1, false);
    try std.testing.expect(!off.enable);
    try std.testing.expectEqualStrings("none", off.effort.?);
    // A word maps through the ONE effortBudget table…
    const low = reasoningEffortFromWord("low", -1, false);
    try std.testing.expect(low.enable);
    try std.testing.expectEqual(@as(i32, 2048), low.budget);
    // …unless the template consumes the word (qwen3.8 class): then the word is
    // the lever and the budget stays the launch default.
    const consumed = reasoningEffortFromWord("low", -1, true);
    try std.testing.expect(consumed.enable);
    try std.testing.expectEqual(@as(i32, -1), consumed.budget);
}

test "resolveChatThinking: an explicit request value outranks the arch default, silence takes it" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, arch: bool, want: bool }{
        // Silent request → the arch default, either way.
        .{ .body = "{}", .arch = false, .want = false },
        .{ .body = "{}", .arch = true, .want = true },
        // An explicit false must survive a thinking-on arch — silently
        // ignoring a field the client set is the class this exists to avoid.
        .{ .body = "{\"enable_thinking\":false}", .arch = true, .want = false },
        .{ .body = "{\"reasoning_effort\":\"none\"}", .arch = true, .want = false },
        // An explicit enable works on an arch that defaults off.
        .{ .body = "{\"enable_thinking\":true}", .arch = false, .want = true },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .arch = false, .want = true },
        // Both present stay OR'd, as they always were.
        .{ .body = "{\"enable_thinking\":false,\"reasoning_effort\":\"high\"}", .arch = false, .want = true },
        .{ .body = "{\"enable_thinking\":true,\"reasoning_effort\":\"none\"}", .arch = false, .want = true },
        // A non-bool `enable_thinking` is not a signal; with nothing else in
        // the body the arch default still applies.
        .{ .body = "{\"enable_thinking\":\"yes\"}", .arch = true, .want = true },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        try std.testing.expectEqual(case.want, resolveChatThinking(parsed.value.object, .{}, case.arch, -1, false).enable);
    }
}

test "resolveChatThinking: the request decides, the generation-defaults rules fill its silence or force it, then the arch" {
    const allocator = std.testing.allocator;
    const Case = struct { body: []const u8, model_on: ?bool = null, model_effort: ?generation_settings.Effort = null, force: bool = false, arch: bool = false, want: bool, effort: ?[]const u8 = null };
    const cases = [_]Case{
        .{ .body = "{}", .arch = true, .want = true },
        .{ .body = "{}", .model_on = true, .want = true },
        .{ .body = "{}", .model_on = false, .arch = true, .want = false },
        .{ .body = "{}", .model_effort = .high, .want = true, .effort = "high" },
        .{ .body = "{}", .model_effort = .none, .arch = true, .want = false },
        .{ .body = "{\"enable_thinking\":false}", .model_effort = .high, .want = false },
        .{ .body = "{\"enable_thinking\":true}", .model_effort = .high, .want = true, .effort = "high" },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .model_effort = .high, .want = true, .effort = "low" },
        // A forced rule replaces the request's switches, effort word included.
        .{ .body = "{\"enable_thinking\":false,\"reasoning_effort\":\"none\"}", .model_on = true, .force = true, .want = true },
        .{ .body = "{\"enable_thinking\":true,\"reasoning_effort\":\"high\"}", .model_on = false, .force = true, .arch = true, .want = false },
        .{ .body = "{\"enable_thinking\":false}", .model_effort = .low, .force = true, .want = true, .effort = "low" },
        // vLLM's spelling; the top-level field wins over it.
        .{ .body = "{\"chat_template_kwargs\":{\"enable_thinking\":false}}", .model_on = true, .want = false },
        .{ .body = "{\"chat_template_kwargs\":{\"reasoning_effort\":\"medium\"}}", .want = true, .effort = "medium" },
        .{ .body = "{\"enable_thinking\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}", .want = true },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        var gen = generation_settings.Profile{};
        if (case.model_on) |on| gen.set(.enable_thinking, .{ .boolean = on }, case.force);
        if (case.model_effort) |e| gen.set(.reasoning_effort, .{ .effort = e }, case.force);
        const got = resolveChatThinking(parsed.value.object, gen, case.arch, -1, false);
        try std.testing.expectEqual(case.want, got.enable);
        if (case.effort) |w| try std.testing.expectEqualStrings(w, got.effort.?.effort.?) else try std.testing.expect(got.effort == null);
    }
}

test "resolveThinking: forced effort outranks unforced thinking defaults" {
    inline for (.{ false, true }) |arch_default| {
        inline for (.{ false, true }) |word_only| {
            var gen = generation_settings.Profile{};
            gen.set(.enable_thinking, .{ .boolean = true }, false);
            gen.set(.reasoning_effort, .{ .effort = .none }, true);
            const off = resolveThinking(gen, false, reasoningEffortFromWord("high", -1, word_only), arch_default, -1, word_only);
            try std.testing.expect(!off.enable);
            try std.testing.expect(off.effort == null);

            gen.set(.enable_thinking, .{ .boolean = false }, false);
            gen.set(.reasoning_effort, .{ .effort = .low }, true);
            const on = resolveThinking(gen, false, reasoningEffortFromWord("none", -1, word_only), arch_default, -1, word_only);
            try std.testing.expect(on.enable);
            try std.testing.expectEqualStrings("low", on.effort.?.effort.?);

            gen.set(.enable_thinking, .{ .boolean = false }, true);
            try std.testing.expect(!resolveThinking(gen, true, null, arch_default, -1, word_only).enable);
            gen.set(.enable_thinking, .{ .boolean = true }, true);
            gen.set(.reasoning_effort, .{ .effort = .none }, true);
            try std.testing.expect(resolveThinking(gen, false, null, arch_default, -1, word_only).enable);
        }
    }
}

test "schema thinking policy defers only a resolved reasoning protocol" {
    const decide = schemaMasksThinking;
    try std.testing.expectEqual(SchemaThinkingPolicy.no_mask, decide(false, false, true, false, true));
    try std.testing.expectEqual(SchemaThinkingPolicy.no_mask, decide(true, true, true, true, true));
    try std.testing.expectEqual(SchemaThinkingPolicy.token_zero, decide(true, false, false, true, false));
    try std.testing.expectEqual(SchemaThinkingPolicy.deferred, decide(true, false, true, false, true));
    try std.testing.expectEqual(SchemaThinkingPolicy.fallback_thinking_off, decide(true, false, true, true, true));
    try std.testing.expectEqual(SchemaThinkingPolicy.fallback_thinking_off, decide(true, false, true, false, false));

    // A suffixed think tail classifies as resolvable (the resolver owns the
    // capability checks); channel-family and plain ChatML tails do not, and
    // keep the fallback — capability is never the tail's alone to decide.
    try std.testing.expect(chat_mod.promptThinkTailClass("<|im_start|>assistant\n<think:opensource>") == .suffixed);
    const unsupported_tails = [_][]const u8{
        "<|channel|>analysis<|message|>",
        "<|im_start|>assistant\n",
    };
    for (unsupported_tails) |tail| {
        try std.testing.expectEqual(SchemaThinkingPolicy.fallback_thinking_off, decide(true, false, true, false, chat_mod.promptThinkTailClass(tail) != .none));
    }
}

test "reasoning budget: the rule stands in for --reasoning-budget, forced it replaces the request's" {
    var gen = generation_settings.Profile{};
    try std.testing.expectEqual(http_server.serverState().server_config.default_reasoning_budget, defaultReasoningBudget(gen));
    gen.set(.reasoning_budget, .{ .number = 1024 }, false);
    try std.testing.expectEqual(@as(i32, 1024), defaultReasoningBudget(gen));
    // xhigh maps to the default budget, so an unforced rule caps it; medium keeps its own.
    try std.testing.expectEqual(@as(i32, 1024), reasoningEffortFromWord("xhigh", defaultReasoningBudget(gen), false).budget);
    try std.testing.expectEqual(@as(i32, 8192), forcedReasoningBudget(gen, reasoningEffortFromWord("medium", 1024, false).budget));
    gen.set(.reasoning_budget, .{ .number = 16 }, true);
    try std.testing.expectEqual(@as(i32, 16), forcedReasoningBudget(gen, 32000));
}

test "messageReasoningFromObj: reasoning_content round-trip, reasoning fallback, empty dropped" {
    const t = std.testing;
    const allocator = t.allocator;
    const cases = [_]struct { body: []const u8, want: ?[]const u8 }{
        // pi/vLLM SSE spelling — the live laguna case.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning_content\":\"two plus two\"}", .want = "two plus two" },
        // vLLM request spelling.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning\":\"twice two\"}", .want = "twice two" },
        // reasoning_content wins when both are present.
        .{ .body = "{\"role\":\"assistant\",\"reasoning_content\":\"a\",\"reasoning\":\"b\"}", .want = "a" },
        // Empty string is NOT reasoning — it would render the empty
        // <think></think> nothink signature the field exists to avoid.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning_content\":\"\"}", .want = null },
        // Non-string shapes are ignored, never crash.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning_content\":{\"x\":1}}", .want = null },
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\"}", .want = null },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = messageReasoningFromObj(parsed.value.object);
        if (case.want) |w| {
            try t.expectEqualStrings(w, got.?);
        } else {
            try t.expect(got == null);
        }
    }
}

test "generated think tags require an unambiguous literal template opener" {
    try std.testing.expectEqualStrings("<think>", templateThinkOpener("{{ '<think>' }} <think>").?);
    try std.testing.expectEqualStrings("<think:opensource>", templateThinkOpener("{{ '<think:opensource>' }}").?);
    try std.testing.expect(templateThinkOpener("<think> <think:opensource>") == null);
    try std.testing.expect(templateThinkOpener("<think:{{ suffix }}>") == null);
    try std.testing.expect(templateThinkOpener("assistant") == null);
}

test "an absent reasoning budget arms an unreachable bound, a set one keeps its value" {
    try std.testing.expectEqual(@as(u32, std.math.maxInt(u32)), thinkBoundBudget(-1));
    try std.testing.expectEqual(@as(u32, 0), thinkBoundBudget(0));
    try std.testing.expectEqual(@as(u32, 2048), thinkBoundBudget(2048));
}
