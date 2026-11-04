//! Reasoning effort vocabularies per template family, template kwargs, and the extra context
//! (`serializeExtraContext`).

const std = @import("std");
const testing = std.testing;

const Message = @import("types.zig").Message;
const ChatConfig = @import("types.zig").ChatConfig;
const templateRequiresReasoningField = @import("serialize.zig").templateRequiresReasoningField;
const serializeMessagesJsonFor = @import("serialize.zig").serializeMessagesJsonFor;
const appendJsonString = @import("../../sdk/util/json_text.zig").appendJsonString;

/// Whether the template READS `reasoning_effort` — i.e. the word is a
/// BEHAVIORAL lever on this checkpoint (dsv4's preamble, Qwen3.8's and
/// Inkling's effort arms) rather than a string we hand over and it ignores.
/// Where it is, the server must NOT also derive a token budget from the same
/// word: the budget only truncates what the client is shown, so the two
/// levers fight — pi asking for `medium` on Qwen3.8 got the model's default
/// (unguided, long) thinking AND a 2048-token display cut, then 25.9k tokens
/// of invisible generation. An explicit
/// `reasoning_budget_tokens` or `--reasoning-budget` is someone asking for a
/// cap on purpose and still applies.
///
/// Substring on the KEY, so a family reading a different one is untouched:
/// muse's `reasoning_strength` is a near-miss the corpus pins.
/// Qwen3.8's effort vocabulary (xhigh|medium|low); `qwen38EffortFor` maps an
/// absent effort to low on this family.
pub fn isQwen38EffortTemplate(tpl: []const u8) bool {
    // Kolibri's effort list also spells 'xhigh' but has its own vocabulary.
    return std.mem.indexOf(u8, tpl, "'xhigh'") != null and !isKolibriEffortTemplate(tpl);
}

/// Kolibri-1's template: the one place that names its private effort variable.
pub fn isKolibriEffortTemplate(tpl: []const u8) bool {
    return std.mem.indexOf(u8, tpl, "_sv_reasoning_effort") != null;
}

pub fn templateConsumesEffort(tpl: []const u8) bool {
    return std.mem.indexOf(u8, tpl, "reasoning_effort") != null;
}

/// OpenAI's effort vocabulary → DeepSeek's low|high|max. `medium` maps to
/// low deliberately: DeepSeek's `high` text is "Absolute maximum with no
/// shortcuts permitted" and its own default is low, so nothing verbose is
/// injected unless the client explicitly asked for it. Unknown strings also
/// fall back to low for the same reason.
pub fn dsv4EffortFor(effort: ?[]const u8) []const u8 {
    const e = effort orelse return "low";
    if (std.mem.eql(u8, e, "high")) return "high";
    if (std.mem.eql(u8, e, "xhigh") or std.mem.eql(u8, e, "max") or std.mem.eql(u8, e, "ultra")) return "max";
    return "low";
}

/// OpenAI's effort vocabulary -> DeepSeek-V4.1's low|medium|high|max (encoder default high).
fn dsv41EffortFor(effort: ?[]const u8) []const u8 {
    const e = effort orelse return "high";
    if (std.mem.eql(u8, e, "none") or std.mem.eql(u8, e, "minimal") or std.mem.eql(u8, e, "low")) return "low";
    if (std.mem.eql(u8, e, "medium")) return "medium";
    if (std.mem.eql(u8, e, "xhigh") or std.mem.eql(u8, e, "max") or std.mem.eql(u8, e, "ultra")) return "max";
    return "high";
}

/// OpenAI's effort vocabulary -> GLM-5-Next's low|high|max (the template's default is max).
fn glm5EffortFor(effort: ?[]const u8) []const u8 {
    const e = effort orelse return "max";
    if (std.mem.eql(u8, e, "none") or std.mem.eql(u8, e, "minimal") or std.mem.eql(u8, e, "low")) return "low";
    if (std.mem.eql(u8, e, "medium") or std.mem.eql(u8, e, "high")) return "high";
    return "max";
}

/// OpenAI's effort vocabulary -> Qwen3.8's xhigh|medium|low. The template
/// raise_exception's on any other string and its own default is xhigh, so an
/// absent or unrecognized effort keeps the checkpoint default. Thinking-off
/// maps to low for the variants that REFUSE it (2.4T-A95B): the request is
/// served on the cheapest arm with the think block closed in the prompt by
/// `noThinkTailSuffix`. On a variant that accepts thinking-off (27B) the
/// template never reads the key on that arm, so the value is inert.
fn qwen38EffortFor(effort: ?[]const u8, enable_thinking: bool) []const u8 {
    if (!enable_thinking) return "low";
    // An ABSENT effort takes OUR default, not the checkpoint's. The template
    // defaults to xhigh, whose preamble asks the model to validate assumptions
    // and consider alternatives, and nothing bounds the result: a pi agent turn
    // that sent no effort ran 16803 reasoning tokens before the client gave up
    // The server-side `reasoning_effort` budget is NOT the
    // answer to that — it truncates rather than shortens, and on exhaustion the
    // model is still inside its block, so the rest of the thought streams as
    // CONTENT. Low is the only lever that makes the model itself stop early
    // ("keep your thinking brief and focused, moving directly to the
    // conclusion"). A client that wants the checkpoint default still gets it by
    // asking for it; only silence changed meaning.
    const e = effort orelse return "low";
    if (std.mem.eql(u8, e, "medium")) return "medium";
    if (std.mem.eql(u8, e, "low") or std.mem.eql(u8, e, "minimal") or std.mem.eql(u8, e, "none")) return "low";
    return "xhigh";
}

/// K2-Horizon reads `reasoning_effort` as high|medium|low (default high) and
/// raises on anything else; each word picks a different think opener.
fn k2EffortFor(effort: ?[]const u8) []const u8 {
    const e = effort orelse return "high";
    if (std.mem.eql(u8, e, "medium")) return "medium";
    if (std.mem.eql(u8, e, "low") or std.mem.eql(u8, e, "minimal") or std.mem.eql(u8, e, "none")) return "low";
    return "high";
}

/// `effort` is the client's raw `reasoning_effort` string (null when the
/// request didn't send one) — today only the dsv4 family maps it into the
/// template; other families keep their fixed vocabulary.
fn appendKwarg(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), key: []const u8, value: std.json.Value) !void {
    if (buf.items.len > 1) try buf.append(allocator, ',');
    try appendJsonString(allocator, buf, key);
    try buf.append(allocator, ':');
    const v = try std.json.Stringify.valueAlloc(allocator, value, .{});
    defer allocator.free(v);
    try buf.appendSlice(allocator, v);
}

/// A request's `chat_template_kwargs` over the model's: one object, request keys win.
pub fn mergeTemplateKwargs(allocator: std.mem.Allocator, model_kwargs: ?[]const u8, request: std.json.ObjectMap) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '{');
    var it = request.iterator();
    while (it.next()) |kv| try appendKwarg(allocator, &buf, kv.key_ptr.*, kv.value_ptr.*);
    if (model_kwargs) |raw| {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
        defer parsed.deinit();
        if (parsed.value == .object) {
            var mit = parsed.value.object.iterator();
            while (mit.next()) |kv| if (!request.contains(kv.key_ptr.*)) try appendKwarg(allocator, &buf, kv.key_ptr.*, kv.value_ptr.*);
        }
    }
    try buf.append(allocator, '}');
    return buf.toOwnedSlice(allocator);
}

/// Kolibri-1's template reads none|minimal|low|medium|high|xhigh|max and treats
/// "none" as thinking-off; any other word it does not know falls through to high.
fn kolibriEffortFor(effort: ?[]const u8, enable_thinking: bool) []const u8 {
    if (!enable_thinking) return "none";
    const e = effort orelse return "high";
    for ([_][]const u8{ "minimal", "low", "medium", "high", "xhigh", "max" }) |w| {
        if (std.mem.eql(u8, e, w)) return w;
    }
    return "high";
}

pub fn serializeExtraContext(allocator: std.mem.Allocator, chat_config: *const ChatConfig, enable_thinking: bool, effort: ?[]const u8) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.append(allocator, '{');
    var need_comma = false;

    if (chat_config.bos_token) |bos| {
        try buf.appendSlice(allocator, "\"bos_token\":");
        try appendJsonString(allocator, &buf, bos);
        need_comma = true;
    }
    if (chat_config.eos_token) |eos| {
        if (need_comma) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "\"eos_token\":");
        try appendJsonString(allocator, &buf, eos);
        need_comma = true;
    }
    // Qwen3.8's effort vocabulary, sniffed on the jinja STRING LITERAL 'xhigh'
    // — it appears in both the `reasoning_effort|default('xhigh')` line and the
    // accepted-values tuple, so a reworded raise message can't drift the
    // detection. EVERY 3.8 template raises on OpenAI's "high", so this keys the
    // mapping for the whole family.
    const qwen38_style = isQwen38EffortTemplate(chat_config.chat_template);
    // Whether the template REFUSES thinking-off is a SEPARATE question from the
    // effort vocabulary, and the two split inside one family: 2.4T-A95B raises
    // ("Disabling thinking is not supported"), the 27B answers that arm the 3.6
    // way with a closed `<think>\n\n</think>` and NO reasoning preamble. Keying
    // the override on the shared 'xhigh' literal forced the 27B's thinking-off
    // requests onto the thinking-on arm, injecting a system line the checkpoint
    // never renders there. Only a refusing template gets the flag withheld;
    // thinking-off is then committed in the prompt by `noThinkTailSuffix`.
    const refuses_nothink = std.mem.indexOf(u8, chat_config.chat_template, "Disabling thinking is not supported") != null;
    if (need_comma) try buf.append(allocator, ',');
    if (enable_thinking or refuses_nothink) {
        try buf.appendSlice(allocator, "\"enable_thinking\":true");
    } else {
        try buf.appendSlice(allocator, "\"enable_thinking\":false");
    }
    // Hy3 templates control thinking via `reasoning_effort` (high/low/
    // no_think), not enable_thinking — and default HIGH when the var is
    // absent. Always supply the mapped value so thinking-off requests
    // actually close the think block; templates that don't read it ignore it.
    //
    // Inkling's template ALSO reads `reasoning_effort` but with a different
    // vocabulary (none/minimal/low/medium/high/max) and raise_exception's on
    // anything else — hy3's "no_think" would fail the render and silently
    // swap in fallbackFormatChat (wrong-family tags). Its thinking-off word
    // is "none"; sniffed by the effort header string unique to that family.
    // Qwen3.8 reads it too, with a THIRD vocabulary (xhigh|medium|low,
    // default xhigh) and its own raise_exception — hy3's "high"/"no_think"
    // fail the render on every request, thinking on or off.
    // DeepSeek-V4-0731's encoder made `reasoning_effort` a THREE-level
    // vocabulary (low|high|max) whose high/max levels PREPEND a verbose
    // "Reasoning Effort: …" preamble to the whole conversation, with low (=
    // nothing) as the reference default. hy3's blanket "high" would inject
    // that preamble into every thinking request on this family, so dsv4 maps
    // the CLIENT's effort string through `dsv4EffortFor` — high/max opt in,
    // everything else (incl. absent) stays the reference default.
    const dsv4_style = std.mem.indexOf(u8, chat_config.chat_template, "thinking_mode") != null;
    const inkling_style = std.mem.indexOf(u8, chat_config.chat_template, "Thinking effort level") != null;
    if (templateRequiresReasoningField(chat_config.chat_template)) {
        try buf.appendSlice(allocator, ",\"reasoning_effort\":\"");
        try buf.appendSlice(allocator, k2EffortFor(effort));
        try buf.append(allocator, '"');
    } else if (dsv4_style) {
        // V4.1 renders a numeric budget for every thinking turn (its own ladder).
        const v41 = std.mem.indexOf(u8, chat_config.chat_template, "(range 1-100") != null;
        try buf.appendSlice(allocator, ",\"reasoning_effort\":\"");
        try buf.appendSlice(allocator, if (v41) dsv41EffortFor(effort) else dsv4EffortFor(effort));
        try buf.append(allocator, '"');
    } else if (qwen38_style) {
        try buf.appendSlice(allocator, ",\"reasoning_effort\":\"");
        try buf.appendSlice(allocator, qwen38EffortFor(effort, enable_thinking));
        try buf.append(allocator, '"');
    } else if (std.mem.indexOf(u8, chat_config.chat_template, "reasoning_effort in ['low', 'high']") != null) {
        try buf.appendSlice(allocator, ",\"reasoning_effort\":\"");
        try buf.appendSlice(allocator, glm5EffortFor(effort));
        try buf.append(allocator, '"');
    } else if (isKolibriEffortTemplate(chat_config.chat_template)) {
        try buf.appendSlice(allocator, ",\"reasoning_effort\":\"");
        try buf.appendSlice(allocator, kolibriEffortFor(effort, enable_thinking));
        try buf.append(allocator, '"');
    } else try buf.appendSlice(allocator, if (enable_thinking)
        ",\"reasoning_effort\":\"high\""
    else if (inkling_style)
        ",\"reasoning_effort\":\"none\""
    else
        ",\"reasoning_effort\":\"no_think\"");

    // DeepSeek-V4's template switches on `thinking_mode` ("chat"|"thinking",
    // the reference encoder's own vocabulary). Only emitted when the template
    // reads it — other families never see the key.
    if (std.mem.indexOf(u8, chat_config.chat_template, "thinking_mode") != null) {
        try buf.appendSlice(allocator, if (enable_thinking)
            ",\"thinking_mode\":\"thinking\""
        else
            ",\"thinking_mode\":\"chat\"");
    }

    // Muse-Glimmer reads `reasoning_strength` (its template default is
    // 'high' and it has no thinking-off switch — the model always emits a
    // to=self segment). Map the client's effort onto low|medium|high;
    // thinking-off drops to 'low'. Only emitted when the template reads it.
    if (std.mem.indexOf(u8, chat_config.chat_template, "reasoning_strength") != null) {
        const strength: []const u8 = if (!enable_thinking)
            "low"
        else if (effort) |e| blk: {
            if (std.mem.eql(u8, e, "none") or std.mem.eql(u8, e, "minimal") or std.mem.eql(u8, e, "low")) break :blk "low";
            if (std.mem.eql(u8, e, "medium")) break :blk "medium";
            break :blk "high";
        } else "high";
        try buf.appendSlice(allocator, ",\"reasoning_strength\":\"");
        try buf.appendSlice(allocator, strength);
        try buf.append(allocator, '"');
    }

    var kwargs: ?std.json.Parsed(std.json.Value) = null;
    defer if (kwargs) |*k| k.deinit();
    if (chat_config.chat_template_kwargs) |raw| {
        kwargs = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch null;
    }
    const kw_obj: ?std.json.ObjectMap = if (kwargs) |k| (if (k.value == .object) k.value.object else null) else null;

    if (kw_obj) |obj| {
        // Set above from resolved values, or the wrapper's own context.
        const reserved = [_][]const u8{ "bos_token", "eos_token", "enable_thinking", "reasoning_effort", "thinking_mode", "reasoning_strength", "messages", "tools", "add_generation_prompt" };
        var it = obj.iterator();
        while (it.next()) |kv| {
            var skip = false;
            for (reserved) |r| skip = skip or std.mem.eql(u8, r, kv.key_ptr.*);
            if (!skip) try appendKwarg(allocator, &buf, kv.key_ptr.*, kv.value_ptr.*);
        }
    }

    try buf.append(allocator, '}');
    return buf.toOwnedSlice(allocator);
}

// ── tests ──

test "templateConsumesEffort: the real templates that READ the effort word" {
    // The three families whose templates act on `reasoning_effort`. For these
    // the word is the lever and an effort-derived TOKEN budget on top is pure
    // truncation (see the fn doc).
    try testing.expect(templateConsumesEffort(@embedFile("../../fixtures/qwen38_chat_template.jinja")));
    try testing.expect(templateConsumesEffort(@embedFile("../../fixtures/qwen38_27b_chat_template.jinja")));
    try testing.expect(templateConsumesEffort(@embedFile("../../fixtures/dsv4_chat_template.jinja")));
    try testing.expect(templateConsumesEffort(@embedFile("../../fixtures/inkling_chat_template.jinja")));

    // Muse reads `reasoning_strength` — the near-miss. Matching it would strip
    // the budget from a family the word never reaches, silently uncapping it.
    try testing.expect(!templateConsumesEffort(@embedFile("../../fixtures/muse_chat_template.jinja")));
    // Everything else on disk (gemma 4, LFM2.5, Qwen3.5/3.6, laguna, Ling,
    // Nemotron, llama, mistral) reads neither: thinking is a bool there.
    try testing.expect(!templateConsumesEffort("{%- if enable_thinking %}<think>\n{%- endif %}"));
    try testing.expect(!templateConsumesEffort(""));
}

test "serializeExtraContext with thinking enabled" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;

    const result = try serializeExtraContext(allocator, &config, true, null);
    defer allocator.free(result);
    // Should contain enable_thinking:true
    try testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\":true") != null);
}

test "serializeExtraContext with thinking disabled" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;

    const result = try serializeExtraContext(allocator, &config, false, null);
    defer allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\":false") != null);
}

test "serializeExtraContext: glm5_next maps effort onto low|high|max" {
    // The template reads low|high and renders anything else as Max.
    const allocator = testing.allocator;
    var glm = ChatConfig{
        .chat_template = @embedFile("../../fixtures/glm5_next_chat_template.jinja"),
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const cases = [_]struct { in: ?[]const u8, out: []const u8 }{
        .{ .in = null, .out = "max" },
        .{ .in = "none", .out = "low" },
        .{ .in = "minimal", .out = "low" },
        .{ .in = "low", .out = "low" },
        .{ .in = "medium", .out = "high" },
        .{ .in = "high", .out = "high" },
        .{ .in = "xhigh", .out = "max" },
        .{ .in = "max", .out = "max" },
    };
    for (cases) |c| {
        const r = try serializeExtraContext(allocator, &glm, true, c.in);
        defer allocator.free(r);
        var want: [64]u8 = undefined;
        try testing.expect(std.mem.indexOf(u8, r, try std.fmt.bufPrint(&want, "\"reasoning_effort\":\"{s}\"", .{c.out})) != null);
    }
}

test "kolibriEffortFor: thinking-off is the word 'none', never hy3's no_think" {
    try testing.expectEqualStrings("none", kolibriEffortFor(null, false));
    try testing.expectEqualStrings("none", kolibriEffortFor("high", false));
    try testing.expectEqualStrings("high", kolibriEffortFor(null, true));
    try testing.expectEqualStrings("low", kolibriEffortFor("low", true));
    try testing.expectEqualStrings("medium", kolibriEffortFor("medium", true));
    try testing.expectEqualStrings("high", kolibriEffortFor("banana", true));
    // Its effort list spells 'xhigh' too, which must not read as Qwen3.8.
    try testing.expect(!isQwen38EffortTemplate("{%- set _sv_reasoning_effort = reasoning_effort -%}['high', 'xhigh', 'max']"));
    try testing.expect(isQwen38EffortTemplate("reasoning_effort|default('xhigh')"));
}

test "dsv4EffortFor: OpenAI effort vocabulary maps onto DeepSeek's low|high|max" {
    // medium deliberately maps LOW: DeepSeek's "high" text is a verbose
    // "absolute maximum, no shortcuts" preamble and the reference default is
    // low — nothing verbose is injected unless the client explicitly asked.
    try testing.expectEqualStrings("low", dsv4EffortFor(null));
    try testing.expectEqualStrings("low", dsv4EffortFor("none"));
    try testing.expectEqualStrings("low", dsv4EffortFor("minimal"));
    try testing.expectEqualStrings("low", dsv4EffortFor("low"));
    try testing.expectEqualStrings("low", dsv4EffortFor("medium"));
    try testing.expectEqualStrings("low", dsv4EffortFor("banana")); // unknown → default
    try testing.expectEqualStrings("high", dsv4EffortFor("high"));
    try testing.expectEqualStrings("max", dsv4EffortFor("xhigh"));
    try testing.expectEqualStrings("max", dsv4EffortFor("max"));
    // `ultra` is an EXPLICIT ask for the top tier — the unknown-string fallback would invert it into low.
    try testing.expectEqualStrings("max", dsv4EffortFor("ultra"));
}

test "serializeExtraContext: V4.1 maps effort onto its encoder's levels, default high" {
    const a = testing.allocator;
    var cfg = ChatConfig{
        .chat_template = @embedFile("../../fixtures/dsv41_chat_template.jinja"),
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = a,
    };
    const cases = [_]struct { in: ?[]const u8, out: []const u8 }{
        .{ .in = null, .out = "high" },
        .{ .in = "banana", .out = "high" },
        .{ .in = "none", .out = "low" },
        .{ .in = "minimal", .out = "low" },
        .{ .in = "low", .out = "low" },
        .{ .in = "medium", .out = "medium" },
        .{ .in = "high", .out = "high" },
        .{ .in = "xhigh", .out = "max" },
        .{ .in = "max", .out = "max" },
    };
    for (cases) |c| {
        const r = try serializeExtraContext(a, &cfg, true, c.in);
        defer a.free(r);
        const want = try std.fmt.allocPrint(a, "\"reasoning_effort\":\"{s}\"", .{c.out});
        defer a.free(want);
        try testing.expect(std.mem.indexOf(u8, r, want) != null);
        try testing.expect(std.mem.indexOf(u8, r, "\"thinking_mode\":\"thinking\"") != null);
    }
}

test "serializeExtraContext: dsv4 gets the reference's default reasoning effort" {
    // DeepSeek-V4-0731 turned `reasoning_effort` into low|high|max where
    // high/max PREPEND a verbose preamble to the conversation and low adds
    // nothing (the reference default). The hy3 convention of sending "high"
    // for every thinking-enabled request would silently attach that preamble
    // to every dsv4 thinking turn; the family is sniffed by `thinking_mode`,
    // which only this template reads.
    const allocator = testing.allocator;
    var dsv4 = ChatConfig{
        .chat_template = "{%- set mode = thinking_mode|default('chat') -%}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    for ([_]bool{ true, false }) |thinking| {
        const r = try serializeExtraContext(allocator, &dsv4, thinking, null);
        defer allocator.free(r);
        try testing.expect(std.mem.indexOf(u8, r, "\"reasoning_effort\":\"low\"") != null);
        // thinking_mode still carries the on/off switch for this family.
        try testing.expect(std.mem.indexOf(u8, r, if (thinking)
            "\"thinking_mode\":\"thinking\""
        else
            "\"thinking_mode\":\"chat\"") != null);
    }
    // A client-supplied effort string reaches the template MAPPED through
    // `dsv4EffortFor` — high/max opt into the verbose preamble, medium stays
    // at the reference default (never silently verbose).
    const mapped_cases = [_]struct { in: ?[]const u8, out: []const u8 }{
        .{ .in = "high", .out = "\"reasoning_effort\":\"high\"" },
        .{ .in = "max", .out = "\"reasoning_effort\":\"max\"" },
        .{ .in = "xhigh", .out = "\"reasoning_effort\":\"max\"" },
        .{ .in = "medium", .out = "\"reasoning_effort\":\"low\"" },
    };
    for (mapped_cases) |case| {
        const r = try serializeExtraContext(allocator, &dsv4, true, case.in);
        defer allocator.free(r);
        try testing.expect(std.mem.indexOf(u8, r, case.out) != null);
    }

    // Other families are untouched: hy3-style keeps high/no_think.
    var other = ChatConfig{
        .chat_template = "{{ reasoning_effort }}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const on = try serializeExtraContext(allocator, &other, true, null);
    defer allocator.free(on);
    try testing.expect(std.mem.indexOf(u8, on, "\"reasoning_effort\":\"high\"") != null);
    const off = try serializeExtraContext(allocator, &other, false, null);
    defer allocator.free(off);
    try testing.expect(std.mem.indexOf(u8, off, "\"reasoning_effort\":\"no_think\"") != null);
}

test "serializeExtraContext: muse maps effort onto reasoning_strength" {
    const allocator = testing.allocator;
    var muse = ChatConfig{
        .chat_template = "…{%- set rs = reasoning_strength if reasoning_strength is defined else 'high' -%}…",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const cases = [_]struct { thinking: bool, effort: ?[]const u8, want: []const u8 }{
        .{ .thinking = true, .effort = null, .want = "\"reasoning_strength\":\"high\"" },
        .{ .thinking = true, .effort = "low", .want = "\"reasoning_strength\":\"low\"" },
        .{ .thinking = true, .effort = "medium", .want = "\"reasoning_strength\":\"medium\"" },
        .{ .thinking = true, .effort = "xhigh", .want = "\"reasoning_strength\":\"high\"" },
        .{ .thinking = false, .effort = null, .want = "\"reasoning_strength\":\"low\"" },
    };
    for (cases) |case| {
        const r = try serializeExtraContext(allocator, &muse, case.thinking, case.effort);
        defer allocator.free(r);
        try testing.expect(std.mem.indexOf(u8, r, case.want) != null);
    }
    // Non-muse templates never see the key.
    var plain = ChatConfig{
        .chat_template = "{{ messages }}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const r = try serializeExtraContext(allocator, &plain, true, null);
    defer allocator.free(r);
    try testing.expect(std.mem.indexOf(u8, r, "reasoning_strength") == null);
}

test "serializeExtraContext: preserve_thinking keeps the template's default; chat_template_kwargs fill what the request did not decide" {
    // Qwen3.8 is trained to reuse prior-turn reasoning; only an explicit kwarg turns it off.
    const allocator = testing.allocator;
    var qwen38 = ChatConfig{
        .chat_template = "…{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}…",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const r = try serializeExtraContext(allocator, &qwen38, true, null);
    defer allocator.free(r);
    try testing.expect(std.mem.indexOf(u8, r, "preserve_thinking") == null);
    // The per-model kwargs can turn it off and carry any other key; a key the
    // request decides (enable_thinking) is not theirs.
    qwen38.chat_template_kwargs = "{\"preserve_thinking\":false,\"custom\":{\"n\":1},\"enable_thinking\":false}";
    const off = try serializeExtraContext(allocator, &qwen38, true, null);
    defer allocator.free(off);
    try testing.expect(std.mem.indexOf(u8, off, "\"preserve_thinking\":false") != null);
    try testing.expect(std.mem.indexOf(u8, off, "\"custom\":{\"n\":1}") != null);
    try testing.expect(std.mem.indexOf(u8, off, "\"enable_thinking\":true") != null);
    try testing.expect(std.mem.indexOf(u8, off, "\"enable_thinking\":false") == null);

    // A kwarg can never replace the conversation the wrapper puts in context.
    qwen38.chat_template_kwargs = "{\"messages\":[],\"tools\":[],\"add_generation_prompt\":false}";
    const core = try serializeExtraContext(allocator, &qwen38, true, null);
    defer allocator.free(core);
    for ([_][]const u8{ "messages", "tools", "add_generation_prompt" }) |k| try testing.expect(std.mem.indexOf(u8, core, k) == null);

    var plain = ChatConfig{
        .chat_template = "{{ messages }}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const p = try serializeExtraContext(allocator, &plain, true, null);
    defer allocator.free(p);
    try testing.expect(std.mem.indexOf(u8, p, "preserve_thinking") == null);
}

test "mergeTemplateKwargs: request keys win, the model's fill the rest" {
    const allocator = testing.allocator;
    const req = try std.json.parseFromSlice(std.json.Value, allocator, "{\"preserve_thinking\":true,\"x\":1}", .{});
    defer req.deinit();
    const merged = try mergeTemplateKwargs(allocator, "{\"preserve_thinking\":false,\"y\":\"m\"}", req.value.object);
    defer allocator.free(merged);
    const got = try std.json.parseFromSlice(std.json.Value, allocator, merged, .{});
    defer got.deinit();
    try testing.expectEqual(true, got.value.object.get("preserve_thinking").?.bool);
    try testing.expectEqual(@as(i64, 1), got.value.object.get("x").?.integer);
    try testing.expectEqualStrings("m", got.value.object.get("y").?.string);
    try testing.expectEqual(@as(usize, 3), got.value.object.count());
}

test "K2-Horizon: assistant history always carries a thinking field, and the effort word is high|medium|low" {
    const k2_tpl = "{%- set think_tag = 'ifm|think' -%}{%- set effort = reasoning_effort | default('high') -%}";
    const cfg = ChatConfig{ .chat_template = k2_tpl, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = testing.allocator };
    const messages = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "hello" },
        .{ .role = "assistant", .content = "again", .reasoning_content = "why" },
    };
    const json = try serializeMessagesJsonFor(testing.allocator, &messages, .null_literal, &cfg);
    defer testing.allocator.free(json);
    try testing.expect(std.mem.indexOf(u8, json, "\"content\":\"hello\",\"reasoning_content\":\"\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"reasoning_content\":\"why\"") != null);
    // Other families keep omitting the key.
    const plain = ChatConfig{ .chat_template = "{{ messages }}", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = testing.allocator };
    const json2 = try serializeMessagesJsonFor(testing.allocator, messages[0..2], .null_literal, &plain);
    defer testing.allocator.free(json2);
    try testing.expect(std.mem.indexOf(u8, json2, "reasoning_content") == null);

    const cases = [_]struct { in: ?[]const u8, want: []const u8 }{
        .{ .in = null, .want = "high" },     .{ .in = "high", .want = "high" },     .{ .in = "xhigh", .want = "high" },
        .{ .in = "max", .want = "high" },    .{ .in = "medium", .want = "medium" }, .{ .in = "low", .want = "low" },
        .{ .in = "minimal", .want = "low" }, .{ .in = "none", .want = "low" },      .{ .in = "bogus", .want = "high" },
    };
    for (cases) |c| try testing.expectEqualStrings(c.want, k2EffortFor(c.in));
    const extra = try serializeExtraContext(testing.allocator, &cfg, true, "xhigh");
    defer testing.allocator.free(extra);
    try testing.expect(std.mem.indexOf(u8, extra, "\"reasoning_effort\":\"high\"") != null);
}
