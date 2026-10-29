//! Tool-call parsing: `parseToolCalls` (the dialect dispatch) and `ParsedToolCall`.

const std = @import("std");
const log = @import("../../../sdk/util/log.zig");
const testing = std.testing;

const indexOfThinkCloseTag = @import("../reasoning/think_tags.zig").indexOfThinkCloseTag;
const INKLING_INVOKE_TAG = @import("../reasoning/inkling.zig").INKLING_INVOKE_TAG;
const HARMONY_CHANNEL_TAG = @import("../reasoning/harmony.zig").HARMONY_CHANNEL_TAG;
const stripParametersWrapper = @import("json_infer.zig").stripParametersWrapper;
const parseSelfClosingToolTag = @import("json_infer.zig").parseSelfClosingToolTag;
const truncatedJsonCallName = @import("json_infer.zig").truncatedJsonCallName;
const balancedJsonObject = @import("json_infer.zig").balancedJsonObject;
const parseXmlElementTagToolCall = @import("xml_element.zig").parseXmlElementTagToolCall;
const parseDsmlToolCalls = @import("dsml.zig").parseDsmlToolCalls;
const parseInklingToolCalls = @import("inkling.zig").parseInklingToolCalls;
const ATEM_INVOKE = @import("invoke.zig").ATEM_INVOKE;
const BARE_INVOKE = @import("invoke.zig").BARE_INVOKE;
const parseHarmonyToolCalls = @import("invoke.zig").parseHarmonyToolCalls;
const parseInvokeToolCalls = @import("invoke.zig").parseInvokeToolCalls;
const parsePythonicToolCalls = @import("pythonic.zig").parsePythonicToolCalls;
const parseHy3ToolCalls = @import("hy3.zig").parseHy3ToolCalls;
const parseXmlElementToolCall = @import("xml_element.zig").parseXmlElementToolCall;
const balancedJsonArray = @import("json_repair.zig").balancedJsonArray;
const appendJsonToolCallArray = @import("json_repair.zig").appendJsonToolCallArray;
const looseRepairToolCallJson = @import("json_repair.zig").looseRepairToolCallJson;
const tryParseJsonToolCall = @import("json_repair.zig").tryParseJsonToolCall;
const unwrapDoubleBraces = @import("json_repair.zig").unwrapDoubleBraces;
const parseGemma4ToolCall = @import("gemma4.zig").parseGemma4ToolCall;
const hermesParamSpanEnclosing = @import("hermes.zig").hermesParamSpanEnclosing;
const hermesSpecialOpenerRepair = @import("hermes.zig").hermesSpecialOpenerRepair;
const parseHermesToolCall = @import("hermes.zig").parseHermesToolCall;
const parseMiniCpm5ToolCalls = @import("minicpm5.zig").parseMiniCpm5ToolCalls;

/// Parse tool calls from model output text.
pub fn parseToolCalls(allocator: std.mem.Allocator, text: []const u8) !?[]ParsedToolCall {
    // Strip thinking blocks if present
    var effective_text = text;
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        effective_text = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
    } else if (indexOfThinkCloseTag(text, 0)) |close| {
        effective_text = std.mem.trimStart(u8, text[close.pos + close.len ..], "\n ");
    }

    var calls = std.ArrayList(ParsedToolCall).empty;
    errdefer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        calls.deinit(allocator);
    }

    // Look for `<tool…>…</tool…>` patterns. DSV4-Flash hallucinates the
    // spec aggressively; we accept the entire training-bias family:
    //   • `<tool_call>JSON</tool_call>`                      (canonical Hermes)
    //   • `<tool_call name="X">JSON</tool_call>`             (attribute style)
    //   • `<tool_call …>JSON</tool_request>`                 (mismatched close)
    //   • `<tool_calls>JSON</tool_calls>`                    (plural form)
    //   • `<tool_calls name="X">{args}</tool_calls>`         (plural + attr)
    //   • `<tool name="X">{args}</tool_calls>`               (bare-tool open, plural close — nested wrappers)
    //   • `<tool>{json}</tool>`                              (bare tag, body has name+args)
    //   • `<tool name="X" arguments="{...}"/>`               (SELF-CLOSING — DSV4 emits this routinely;
    //                                                          parseSelfClosingToolTag handles broken-quote
    //                                                          variants like `…"}'/>` too)
    // Approach: search for the prefix `<tool` (5 chars). Skip the Gemma 4
    // close marker `<tool_call|>`. Try the self-closing form first (cheap
    // attribute scan + balanced-JSON extraction; tolerates the model's
    // mismatched quotes). If that doesn't match, fall through to the
    // open/close-marker path: scan to the next `>`, optionally capturing
    // `name="X"` / `name='X'`, then locate the earliest of `</tool>`,
    // `</tool_call>`, `</tool_calls>`, `</tool_request>`, `</tool_requests>`.
    // When the body fails to parse, advance only past the OPENING `>` so any
    // inner `<tool…>` blocks get a chance — this is what makes the outer
    // `<tool_calls>` wrapper case work.
    //
    // Inkling (inkling_mm_model) is tried FIRST: its calls are MESSAGES —
    // `NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>` —
    // whose distinctive marker no other family emits; the generic `<tool`
    // scans never see the shape and other families never carry the marker.
    try parseInklingToolCalls(allocator, text, &calls);
    // Muse-Glimmer ATEM (`<atem:invoke name="X">`): distinctive namespace
    // prefix no other family emits; the generic `<tool` scan never sees it
    // (the bytes after `<` are `atem`, not `tool`). Runs on the FULL text —
    // tool segments arrive wrapped in channel headers the think-strip above
    // would misjudge.
    try parseInvokeToolCalls(allocator, text, &calls, ATEM_INVOKE);
    // gpt_oss / harmony (`to=functions.NAME<|channel|>commentary…<|message|>`):
    // the name lives in the header, not the payload, so no generic scan can
    // recover it. Runs on the FULL text for the same reason ATEM does — the
    // call arrives wrapped in channel headers the think-strip would misjudge.
    try parseHarmonyToolCalls(allocator, text, &calls);
    // DeepSeek-V4 native DSML (`<｜DSML｜invoke …>`): distinctive fullwidth-bar
    // marker no other family emits, and the generic `<tool` scan never sees
    // it (the byte before "tool_calls" is `｜`, not `<`).
    try parseDsmlToolCalls(allocator, effective_text, &calls);
    // LFM2.5 pythonic (`<|tool_call_start|>[fn(a='x')]<|tool_call_end|>`):
    // distinctive marker no other family emits, and the generic `<tool` scan
    // never sees it (the opener is `<|tool_call_start|>`, not `<|tool_call>`).
    try parsePythonicToolCalls(allocator, effective_text, &calls);
    // Hy3 (Hunyuan 3) SUFFIXED tag format is tried FIRST among the tag
    // families: its wrapper (`<tool_calls:opensource>`) would also trip the
    // generic `<tool` scan below, which would misread the non-JSON
    // arg_key/arg_value body. When it matched, the generic scans are skipped
    // but the shared safety net at the bottom still runs.
    try parseHy3ToolCalls(allocator, effective_text, &calls);
    var search_pos: usize = if (calls.items.len > 0) effective_text.len else 0;
    while (search_pos < effective_text.len) {
        const rel = std.mem.indexOf(u8, effective_text[search_pos..], "<tool") orelse break;
        const after_tool = search_pos + rel + "<tool".len;
        if (after_tool >= effective_text.len) break;

        // Reject anything that isn't actually a tool open tag — the char
        // right after `<tool` must be `>`, whitespace, `_` (suffix coming),
        // or `|` (Gemma 4 close marker). Anything else is text like
        // `<toolkit>` or `<toolbar>`; advance past `<tool` and keep scanning.
        const next = effective_text[after_tool];
        if (next != '>' and next != ' ' and next != '\t' and next != '\n' and next != '_' and next != '|') {
            search_pos = after_tool;
            continue;
        }
        // Gemma 4 close marker: `<tool_call|>`. Detected when after `<tool`
        // we see `_call|`. Let the Gemma 4 branch below pick this up.
        if (next == '_' and after_tool + 6 <= effective_text.len and std.mem.eql(u8, effective_text[after_tool .. after_tool + 6], "_call|")) {
            search_pos = after_tool + 6;
            continue;
        }
        // If suffix is `_`, the only accepted continuations are
        // `_call`, `_calls`, `_request`, `_requests`. Any OTHER suffix gets
        // one more chance as the XML-element-TAG form before being rejected:
        // DSV4 embeds the tool name in the tag itself —
        //   `<tool_read>\n<path>mlx.html</path>\n</tool_read>`
        // ( pi html-ds4 capture; both calls leaked as text and pi
        // scored 0/4). Conditions kept tight: tag must close with `>` right
        // after the name (no attributes), an EXACT `</tool_NAME>` close must
        // exist, the name must not be a result-ish marker (`<tool_output>`
        // is DSV4 hallucinating a result, not calling a tool named
        // "output"), and the body must be entirely `<key>value</key>` args.
        if (next == '_') {
            const accepted = [_][]const u8{ "_call", "_calls", "_request", "_requests" };
            var matched = false;
            for (accepted) |suf| {
                if (after_tool + suf.len > effective_text.len) continue;
                if (!std.mem.eql(u8, effective_text[after_tool .. after_tool + suf.len], suf)) continue;
                // Char right after the suffix must be tag-terminating; otherwise
                // `_call` could swallow `<tool_called_X>`.
                const post = after_tool + suf.len;
                if (post >= effective_text.len) break;
                const c = effective_text[post];
                if (c == '>' or c == ' ' or c == '\t' or c == '\n' or c == '/' or c == '|') {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                const origin = after_tool - "<tool".len;
                if (try parseXmlElementTagToolCall(allocator, effective_text[origin..])) |etc| {
                    try calls.append(allocator, .{ .name = etc.call.name, .arguments = etc.call.arguments });
                    search_pos = origin + etc.consumed;
                    continue;
                }
                search_pos = after_tool;
                continue;
            }
        }

        // Self-closing `<tool …/>` form first — DSV4 emits this routinely
        // and the open/close scan below won't find a `</tool*>` close, so
        // without this branch the call gets dropped silently.
        const tag_origin = after_tool - "<tool".len;
        if (parseSelfClosingToolTag(effective_text[tag_origin..])) |sc| {
            const name_owned = try allocator.dupe(u8, sc.name);
            errdefer allocator.free(name_owned);
            const args_owned = try allocator.dupe(u8, sc.arguments);
            try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
            search_pos = tag_origin + sc.consumed;
            continue;
        }

        // Scan the opening tag to its `>`, capturing optional name attr.
        var attr_name: ?[]const u8 = null;
        var content_start: usize = 0;
        {
            var i: usize = after_tool;
            // Bound the scan so a malformed input can't run wild.
            const limit = @min(effective_text.len, after_tool + 256);
            while (i < limit and effective_text[i] != '>') : (i += 1) {
                if (effective_text[i] != 'n') continue;
                if (i + 5 > effective_text.len) continue;
                if (!std.mem.eql(u8, effective_text[i .. i + 5], "name=")) continue;
                const q_pos = i + 5;
                if (q_pos >= effective_text.len) break;
                const quote = effective_text[q_pos];
                if (quote != '"' and quote != '\'') continue;
                const val_start = q_pos + 1;
                const val_end_rel = std.mem.indexOfScalar(u8, effective_text[val_start..], quote) orelse break;
                attr_name = effective_text[val_start .. val_start + val_end_rel];
                i = val_start + val_end_rel;
            }
            if (i >= limit or effective_text[i] != '>') {
                // Unclosed opening tag — skip past `<tool` and keep going.
                search_pos = after_tool;
                continue;
            }
            content_start = i + 1;
        }
        // Find the close marker. Prefer the close matching the OPENING tag's
        // name (`<tool_calls>` closes at `</tool_calls>`, never at a
        // `</tool_name>` child element inside the body — the XML-element
        // form puts `</tool…>`-shaped children first). When no exact match
        // exists, fall back to the earliest `</tool…>` with ANY suffix of
        // word characters: DSV4 hallucinates closes freely (`</tool_action>`
        // was captured live closing a `<tool_call>` open), and pinning an
        // exact-name list drops the whole call and leaks it as visible text.
        var close_rel: ?usize = null;
        var close_len: usize = 0;
        {
            const hay = effective_text[content_start..];
            // Opening tag name: "tool" + suffix up to the attr/`>` delimiter.
            var name_end = after_tool;
            while (name_end < effective_text.len and
                (std.ascii.isAlphanumeric(effective_text[name_end]) or effective_text[name_end] == '_')) : (name_end += 1)
            {}
            const open_name = effective_text[tag_origin + 1 .. name_end];
            var exact_buf: [40]u8 = undefined;
            if (open_name.len + 3 <= exact_buf.len) {
                const exact = std.fmt.bufPrint(&exact_buf, "</{s}>", .{open_name}) catch unreachable;
                var from: usize = 0;
                while (std.mem.indexOfPos(u8, hay, from, exact)) |found| {
                    // A `<parameter=…>` value may spell the wrapper's own close.
                    if (hermesParamSpanEnclosing(hay, found)) |past| {
                        from = past;
                        continue;
                    }
                    close_rel = found;
                    close_len = exact.len;
                    break;
                }
            }
            if (close_rel == null) {
                var cpos: usize = 0;
                while (std.mem.indexOf(u8, hay[cpos..], "</tool")) |found| {
                    const open_at = cpos + found;
                    var j = open_at + "</tool".len;
                    const jlimit = @min(hay.len, j + 24);
                    var word_only = true;
                    while (j < jlimit and hay[j] != '>') : (j += 1) {
                        const c = hay[j];
                        if (!std.ascii.isAlphanumeric(c) and c != '_') {
                            word_only = false;
                            break;
                        }
                    }
                    if (word_only and j < jlimit and hay[j] == '>') {
                        if (hermesParamSpanEnclosing(hay, open_at)) |past| {
                            cpos = past;
                            continue;
                        }
                        close_rel = open_at;
                        close_len = j + 1 - open_at;
                        break;
                    }
                    cpos = open_at + "</tool".len;
                }
            }
        }
        if (close_rel == null) {
            // No `</tool…>` close in sight. Two ways this happens in the
            // wild: (1) the model emitted EOS mid-tool-call (`<tool_call>\n{
            // …well-formed args…}` then nothing); (2) max_tokens truncated.
            // The args object itself is usually intact — snap a balanced
            // JSON object from right after the open tag and try to parse it.
            // Function-tag form first: a parameter VALUE is arbitrary bytes and never names the tool.
            const truncated_fn_tag = std.mem.indexOf(u8, effective_text[content_start..], "<function=") != null;
            const truncated_json: ?[]const u8 = if (truncated_fn_tag) null else balancedJsonObject(effective_text[content_start..]);
            if (truncated_json) |json_body| {
                const json_off = @intFromPtr(json_body.ptr) - @intFromPtr(effective_text[content_start..].ptr);
                const advance_to = content_start + json_off + json_body.len;
                // Hermes shape (top-level "name" + "arguments"): use as-is.
                if (tryParseJsonToolCall(allocator, json_body)) |tc| {
                    try calls.append(allocator, tc);
                    search_pos = advance_to;
                    continue;
                }
                // Attribute shape — body is JUST the args, take name from attr.
                if (attr_name) |an| {
                    const name_owned = try allocator.dupe(u8, an);
                    errdefer allocator.free(name_owned);
                    const args_owned = try allocator.dupe(u8, json_body);
                    try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                    search_pos = advance_to;
                    continue;
                }
            }
            // Truncated Hermes/XML tool call: the model emitted
            // `<tool_call><function=NAME><parameter=KEY>…` (Hermes function-tag)
            // or the XML-element `<tool_name>NAME</tool_name>…` shape and ran
            // out of tokens before ANY closing tag — so there's no balanced
            // JSON to snap. The OPENING tags still carry the tool NAME; recover
            // that (parseHermesToolCall breaks out of its parameter loop on a
            // missing `</parameter>`, yielding name + `{}` args) so the call is
            // recognized as a truncated writeFile instead of being DROPPED and
            // leaked into visible content (live JFK-novel capture,
            // a 19k-char writeFile cut off mid-content was silently lost and
            // the app fired the wrong "malformed tag" ghost nudge). Recovering
            // the name is enough for the client to fire the right chunk/append
            // nudge; we do NOT salvage the partial content (a half-written file
            // is worse than a re-issued chunked write). The text is truncated
            // to its end here, so advance past it to terminate the scan.
            if (parseHermesToolCall(allocator, effective_text[content_start..])) |tc| {
                try calls.append(allocator, tc);
                search_pos = effective_text.len;
                continue;
            }
            if (parseXmlElementToolCall(allocator, effective_text[content_start..])) |tc| {
                try calls.append(allocator, tc);
                search_pos = effective_text.len;
                continue;
            }
            // JSON dialect cut off INSIDE the object (EOS mid-string on a
            // large edit): nothing balances, but a top-level `"name"` key is
            // still the tool. NAME + `{}`, never a fragment.
            if (truncatedJsonCallName(effective_text[content_start..])) |name| {
                const name_owned = try allocator.dupe(u8, name);
                errdefer allocator.free(name_owned);
                const args_owned = try allocator.dupe(u8, "{}");
                try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                search_pos = effective_text.len;
                continue;
            }
            break;
        }
        const content = std.mem.trim(u8, effective_text[content_start .. content_start + close_rel.?], " \t\n\r");

        // Pre-clean the body before the parse attempts. Two DSV4 quirks:
        //   • The args object may be wrapped in `<parameters>…</parameters>`.
        //   • The model sometimes emits an extra trailing `}` after the
        //     proper JSON close. Find the first balanced JSON object via
        //     depth tracking and use just that.
        const unwrapped = stripParametersWrapper(content);
        // Some models (often when echoing a Jinja `{{ }}` example) wrap the args
        // object in an extra brace layer: `<tool_call>{{"name":…}}</tool_call>`.
        // Strip one layer so the body parses as JSON. Only fires on the `{{…}}`
        // shape (which otherwise fails to parse), so valid single-brace bodies are
        // untouched. Mirrors the unwrap in `parseGemma4ToolCall`.
        const balanced: []const u8 = unwrapDoubleBraces(balancedJsonObject(unwrapped) orelse unwrapped);

        // Try to extract a tool call from the body. Three shapes, tried in
        // priority order:
        //   1. Attribute form (`name="X"` captured, body is JUST args).
        //   2. Canonical Hermes JSON ({"name":"X","arguments":{...}}).
        //   3. Hermes function-tag format (parseHermesToolCall).
        var parsed_ok = false;
        // 0. Function-tag form FIRST when the body actually carries a
        // `<function=` opener. The dialect qwen 3.5/3.6's own template
        // mandates rides inside the SAME `<tool_call>` wrapper the JSON form
        // uses, and a `<parameter=…>` VALUE is arbitrary bytes — a written
        // package.json made `balancedJsonObject` snap the FILE CONTENT, whose
        // "name" key then became the tool name (pi answered
        // "Tool voxel-pagoda-garden not found" and the model looped on it).
        // The JSON shapes below can never contain that opener, so this only
        // reorders the case that was being read wrong.
        const repaired_opener = try hermesSpecialOpenerRepair(allocator, content);
        defer if (repaired_opener) |r| allocator.free(r);
        const body = repaired_opener orelse content;
        if (std.mem.indexOf(u8, body, "<function=") != null) {
            if (parseHermesToolCall(allocator, body)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (attr_name) |an| {
                if (std.json.parseFromSlice(std.json.Value, allocator, balanced, .{})) |parsed| {
                    defer parsed.deinit();
                    if (parsed.value == .object) {
                        const name_owned = try allocator.dupe(u8, an);
                        errdefer allocator.free(name_owned);
                        const args_owned = try allocator.dupe(u8, balanced);
                        try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                        parsed_ok = true;
                    }
                } else |_| {}
            }
        }
        if (!parsed_ok) {
            if (tryParseJsonToolCall(allocator, balanced)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (parseHermesToolCall(allocator, content)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (parseXmlElementToolCall(allocator, content)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }

        if (parsed_ok) {
            // Body consumed cleanly — advance past the close marker.
            search_pos = content_start + close_rel.? + close_len;
        } else {
            // Body wasn't a tool call. Advance only past the opening `>`
            // so inner `<tool…>` blocks (if any) still get parsed. This is
            // what makes the outer-wrapper case work: the outer
            // `<tool_calls>` has no JSON body, but its content contains
            // real `<tool name="X">` inner calls.
            search_pos = content_start;
        }
    }

    // Gemma 4 format: <|tool_call>call:name{args}<tool_call|>
    if (calls.items.len == 0) {
        search_pos = 0;
        while (search_pos < effective_text.len) {
            const tag_start = std.mem.indexOf(u8, effective_text[search_pos..], "<|tool_call>") orelse break;
            const content_start = search_pos + tag_start + "<|tool_call>".len;
            const tag_end_opt = std.mem.indexOf(u8, effective_text[content_start..], "<tool_call|>");
            const content = if (tag_end_opt) |tag_end|
                std.mem.trim(u8, effective_text[content_start .. content_start + tag_end], " \t\n\r")
            else
                // Incomplete tool call (model hit EOS before closing tag) — use rest of text
                std.mem.trim(u8, effective_text[content_start..], " \t\n\r");

            search_pos = if (tag_end_opt) |tag_end|
                content_start + tag_end + "<tool_call|>".len
            else
                effective_text.len;

            // No <tool_call|> close = the generation was CUT mid-call (EOS,
            // max_tokens, or the degenerate-tail-loop guard) — fragment values
            // are dropped inside the parse, only completed args ship.
            if (parseGemma4ToolCall(allocator, content, tag_end_opt == null)) |tc| {
                try calls.append(allocator, tc);
            } else {
                log.info("  [tool-parse] Gemma4 parse FAILED for: {s}\n", .{content[0..@min(content.len, 200)]});
            }
        }
    }

    // Bare Hermes function tag with NO <tool_call> wrapper. The outer scan
    // triggers on the substring `<tool`, which a lone `<function=…></function>`
    // (even with a trailing `</tool_call>`, since that is `</too…`, not `<tool`)
    // never provides — so a model that drops the OPENING <tool_call> would leak
    // the whole call as visible text. Only fire when a `<function=` opener AND a
    // `<parameter=` are both present, so prose that merely mentions the tag can't
    // false-fire (parseHermesToolCall needs a real parameter to recover args).
    if (calls.items.len == 0) {
        if (std.mem.indexOf(u8, effective_text, "<function=") != null and
            std.mem.indexOf(u8, effective_text, "<parameter=") != null)
        {
            if (parseHermesToolCall(allocator, effective_text)) |tc| {
                try calls.append(allocator, tc);
            }
        }
    }

    // MiniCPM5 V3 XML: `<function name="X"><param name="K">V</param>…</function>`,
    // zero or more consecutive calls, no outer wrapper. Tried only after every
    // other explicit tag format above has had its shot, and before the
    // raw-JSON guessing fallback below — an explicit recognized tag always
    // wins over heuristic inference.
    if (calls.items.len == 0) {
        // Never scan a gpt-oss/harmony transcript for this dialect. Harmony owns
        // its own format and its arm has already run; what is left is its
        // ANALYSIS channel — reasoning the user never sees as an answer. A
        // harmony model explaining `<function name="get_time"></function>` was
        // being turned into a real `get_time({})` call by this fallback, an
        // invocation nobody requested. A MiniCPM5 checkpoint never emits
        // `<|channel|>`, so this costs the dialect nothing.
        const harmony_owned = std.mem.indexOf(u8, effective_text, HARMONY_CHANNEL_TAG) != null;
        if (!harmony_owned and std.mem.indexOf(u8, effective_text, "<function") != null) {
            try parseMiniCpm5ToolCalls(allocator, effective_text, &calls);
        }
    }

    // Anthropic's `<invoke>` is another family's dialect, so it runs after every native one:
    // inside one of their parameter values it is payload, never the call.
    if (calls.items.len == 0) try parseInvokeToolCalls(allocator, effective_text, &calls, BARE_INVOKE);

    // If no <tool_call> tags, try to find raw JSON tool call(s). Never on
    // text carrying the Inkling invoke marker: that format OWNED the parse,
    // and a call it deliberately skipped (no recoverable name) must not be
    // resurrected from its payload bytes by the bare-JSON heuristic — the
    // comment below says "no tag syntax anywhere", and there IS tag syntax.
    if (calls.items.len == 0 and std.mem.indexOf(u8, text, INKLING_INVOKE_TAG) == null) {
        var trimmed = std.mem.trim(u8, effective_text, " \t\n\r");
        if (std.mem.startsWith(u8, trimmed, "</tool_call>")) {
            trimmed = std.mem.trim(u8, trimmed["</tool_call>".len..], " \t\n\r");
        }
        // JSON ARRAY of {name, arguments} objects — parallel tool calls from
        // models without a trained tool format. Gemma 3 emits a ```json fence
        // around the array; skip one leading fence line before the check so
        // the array (not the first object inside it) is what we parse —
        // otherwise only the first call survives and the rest silently drop.
        var array_probe = trimmed;
        if (std.mem.startsWith(u8, array_probe, "```")) {
            if (std.mem.indexOfScalar(u8, array_probe, '\n')) |nl| {
                array_probe = std.mem.trim(u8, array_probe[nl + 1 ..], " \t\n\r");
            }
        }
        if (array_probe.len > 0 and array_probe[0] == '[') {
            if (balancedJsonArray(array_probe)) |arr_body| {
                try appendJsonToolCallArray(allocator, arr_body, &calls);
            }
        }
        if (calls.items.len == 0) {
            if (std.mem.indexOf(u8, trimmed, "{")) |brace_pos| {
                const json_start = trimmed[brace_pos..];
                // Snap the first balanced object so trailing garbage can't poison
                // the parse — models without a trained tool format (Gemma 3) emit
                // the call inside a ```json fence, leaving "\n```" after the `}`.
                const json_body = balancedJsonObject(json_start) orelse json_start;
                if (tryParseJsonToolCall(allocator, json_body)) |tc| {
                    try calls.append(allocator, tc);
                }
            }
        }
        // Everything appended in this block was inferred from BARE JSON — no
        // tag syntax anywhere. Mark provenance so the request chokepoint
        // (server.parseToolCallsForRequest → filterInferredBySchema) can
        // validate the name against the declared tools: the first balanced
        // {"name": …} object in truncated prose/code is routinely a DATA
        // record, not a call (the George Washington class).
        for (calls.items) |*tc| tc.inferred = true;
    }

    if (calls.items.len == 0) return null;

    // Final safety net: EVERY emitted call carries valid-JSON arguments, whatever
    // converter built them. The direct-construction converters (Gemma custom
    // format, Hermes) can still emit invalid JSON on a pathological value — e.g. a
    // JSON-style string with a bad escape (`\q`) copied verbatim. std.json is the
    // arbiter: if the args don't parse, try the tolerant re-serialize (which
    // re-escapes lone backslashes / control bytes / inner quotes), and if THAT
    // still fails, fall back to `{}` — a client can retry a named call with empty
    // args, but invalid JSON it cannot parse at all. This makes the "args are
    // always valid JSON" invariant structural, not per-converter.
    for (calls.items) |*tc| {
        if (jsonIsValidObject(allocator, tc.arguments)) continue;
        if (looseRepairToolCallJson(allocator, tc.arguments)) |repaired| {
            if (jsonIsValidObject(allocator, repaired)) {
                allocator.free(tc.arguments);
                tc.arguments = repaired;
                continue;
            }
            allocator.free(repaired);
        }
        log.info("  [tool-parse] unrepairable args for '{s}' → empty object (never ship invalid JSON)\n", .{tc.name});
        const empty = try allocator.dupe(u8, "{}");
        allocator.free(tc.arguments);
        tc.arguments = empty;
    }

    return try calls.toOwnedSlice(allocator);
}

/// True when `s` strict-parses as a JSON object. Cheap arbiter for the
/// parseToolCalls safety net.
fn jsonIsValidObject(allocator: std.mem.Allocator, s: []const u8) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, s, .{}) catch return false;
    defer parsed.deinit();
    return parsed.value == .object;
}

pub const ParsedToolCall = struct {
    name: []const u8,
    arguments: []const u8, // JSON string
    /// True when the call was HEURISTICALLY inferred from bare JSON in the
    /// output (no tag syntax — the model never said "this is a tool call").
    /// The request chokepoint validates inferred names against the declared
    /// tools schema: a truncated DATA object like {"name": "George
    /// Washington", "num": 1, …} must never become a tool call (live pi
    /// capture). Explicit tag-format calls keep flowing even with
    /// unknown names — "tool not found" is feedback the model corrects from.
    inferred: bool = false,
};

pub fn earliestIndexOfAny(text: []const u8, from: usize, needles: []const []const u8) ?usize {
    var best: ?usize = null;
    for (needles) |n| {
        if (std.mem.indexOfPos(u8, text, from, n)) |q| {
            if (best == null or q < best.?) best = q;
        }
    }
    return best;
}

// ── tests ──

test "parseToolCalls JSON format" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    // arguments should be valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "parseToolCalls Qwen3.6 MoE nested-name garbage (real capture)" {
    // Captured verbatim from Qwen3.6-35B-A3B-6bit generating a tool call in
    // streaming mode with enable_thinking=false and tools present.
    // parseToolCalls previously returned null here because the top-level "name"
    // is an object, not a string — tokens then leaked as plain-text content.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": {"name": {"name":  "write", "arguments": {"path": "/tmp/x/app.test.js", "content": "hello"}}}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/x/app.test.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("hello", parsed.value.object.get("content").?.string);
}

test "parseToolCalls Qwen MoE double-nested name" {
    // Two layers of nesting — defensive test for a slightly different shape.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name":{"name":"shell","arguments":{"command":"ls"}}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls flat Qwen MoE shape (no arguments wrapper)" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"shell\", \"command\": \"mkdir -p 2ddungeon\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p 2ddungeon", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("name") == null);
}

test "parseToolCalls flat shape with multiple top-level args" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"writeFile\", \"path\": \"/tmp/a\", \"content\": \"hi\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/a", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("hi", parsed.value.object.get("content").?.string);
}

test "parseToolCalls flat shape ignores id/type metadata" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"id\": \"call_1\", \"type\": \"function\", \"name\": \"shell\", \"command\": \"ls\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("id") == null);
    try testing.expect(parsed.value.object.get("type") == null);
}

test "parseToolCalls leaves valid escaped content untouched (no regression)" {
    const allocator = testing.allocator;
    // Already-correct JSON must pass straight through — the recovery path only
    // runs after strict parse fails, so this never touches looseRepair.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"ok.js\",\"content\":\"a\\nb\\t\\\"q\\\"\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a\nb\t\"q\"", parsed.value.object.get("content").?.string);
}

test "parseToolCalls returns null for name-only object (no real args)" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"shell\"}\n</tool_call>";
    const result = try parseToolCalls(allocator, text);
    try testing.expect(result == null);
}

test "parseToolCalls marks raw-JSON fallback calls as inferred, tag calls as explicit" {
    const allocator = testing.allocator;
    // Tagged call → explicit (never schema-name-filtered).
    {
        const calls = (try parseToolCalls(allocator, "<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}</tool_call>")).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expect(!calls[0].inferred);
    }
    // Bare raw-JSON object → inferred (flat shape included).
    {
        const calls = (try parseToolCalls(allocator, "prefix text {\"name\": \"George Washington\", \"num\": 1, \"party\": \"None\"} suffix")).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expect(calls[0].inferred);
        try testing.expectEqualStrings("George Washington", calls[0].name);
    }
}

test "parseToolCalls multiple calls" {
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": "func_a", "arguments": {"x": 1}}
        \\</tool_call>
        \\<tool_call>
        \\{"name": "func_b", "arguments": {"y": 2}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("func_a", calls[0].name);
    try testing.expectEqualStrings("func_b", calls[1].name);
}

test "parseToolCalls returns null for no tool calls" {
    const allocator = testing.allocator;
    const result = try parseToolCalls(allocator, "Hello, how can I help you?");
    try testing.expect(result == null);
}

// ── DSV4 training-bias tool-call shapes ──
// DSV4-Flash via ds4 emits tool calls in two non-standard variants the
// reference parser previously dropped on the floor:
//   1. Attribute form: `<tool_call name="X">{args}</tool_call>` — name as an
//      XML attribute, body is JUST the args object.
//   2. Mismatched closing tag: `<tool_call>{full json}</tool_request>` — open
//      tag is `tool_call`, close is `tool_request`. Hermes-style trained on a
//      mixed corpus.
// These tests are the regression guard. Without the fix they fail because
// the parser hard-matched on `<tool_call>…</tool_call>` exactly.

test "parseToolCalls DSV4 attribute form: <tool_call name=\"X\">{args}</tool_call>" {
    const allocator = testing.allocator;
    const text = "<tool_call name=\"webSearch\">{\"query\":\"top open source inference applications\"}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("top open source inference applications", parsed.value.object.get("query").?.string);
}

test "parseToolCalls DSV4 mismatched close: <tool_call>{json}</tool_request>" {
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": "shell", "arguments": {"command": "mkdir -p prisma src/lib"}}
        \\</tool_request>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p prisma src/lib", parsed.value.object.get("command").?.string);
}

test "parseToolCalls DSV4 attribute + mismatched close combined" {
    // Worst-case combo observed from DSV4: attribute-style open, body is
    // just the args object (no top-level "name" key), and the closing tag
    // is `</tool_request>` instead of `</tool_call>`.
    const allocator = testing.allocator;
    const text = "<tool_call name=\"shell\">{\"command\":\"ls /tmp\"}</tool_request>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls /tmp", parsed.value.object.get("command").?.string);
}

test "parseToolCalls DSV4 attribute single-quoted: <tool_call name='X'>" {
    // Some captures used single quotes around the attribute value. Same
    // semantics — accept both quote styles.
    const allocator = testing.allocator;
    const text = "<tool_call name='browse'>{\"action\":\"navigate\",\"url\":\"https://example.com\"}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
}

test "parseToolCalls DSV4 plural-tag form: <tool_calls>{json}</tool_calls>" {
    // Real capture from DSV4-Flash in MLX Core ChatView:
    //   <tool_calls>
    //   {"name": "shell", "arguments": {"command": "df -h /"}}
    //   </tool_calls>
    // Plural open + plural close, single JSON object inside. Without the
    // fix the parser hard-matches `<tool_call` (the suffix `s>` makes the
    // `>`-scan walk past the tool name), the body never extracts, and the
    // raw XML leaks into the chat bubble.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\{"name": "shell", "arguments": {"command": "df -h /"}}
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("df -h /", parsed.value.object.get("command").?.string);
}

test "parseToolCalls DSV4 plural-tag with attribute: <tool_calls name=\"X\">{args}</tool_calls>" {
    const allocator = testing.allocator;
    const text = "<tool_calls name=\"webSearch\">{\"query\":\"hi\"}</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
}

test "parseToolCalls DSV4 nested <tool name=\"X\"> inside <tool_calls> wrapper" {
    // Real capture from DSV4-Flash:
    //   <tool_calls>
    //   <tool name="webSearch">{"query": "top 10 ..."}</tool_calls>
    //   <tool name="webSearch">{"query": "best ..."}</tool_calls>
    //   </tool_calls>
    // Two inner calls, each opening with `<tool name="X">` and closing with
    // `</tool_calls>` (the model conflates open and close tag names). The
    // outer `<tool_calls>` wrapper has no content of its own — it's a
    // decorative parent.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool name="webSearch">{"query": "top 10 open source AI inference engines 2024 2025"}</tool_calls>
        \\<tool name="webSearch">{"query": "best open source AI inference engines vllm tensorrt mlx comparison"}</tool_calls>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    // Both inner calls must be recovered.
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
    try testing.expectEqualStrings("webSearch", calls[1].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expect(std.mem.indexOf(u8, a0.value.object.get("query").?.string, "top 10") != null);
    const a1 = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer a1.deinit();
    try testing.expect(std.mem.indexOf(u8, a1.value.object.get("query").?.string, "best") != null);
}

test "parseToolCalls bare <tool>{json}</tool>" {
    // Hypothetical minimal form — `<tool>` open with no attribute, JSON
    // body carries the name. Belt-and-suspenders for future model output.
    const allocator = testing.allocator;
    const text = "<tool>{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}</tool>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls DSV4 <parameters> wrapper inside <tool name=\"X\">" {
    // Real capture from DSV4-Flash:
    //   <tool_calls>
    //   <tool name="browse"><parameters>{"action":"navigate","url":"…"}</parameters></tool>
    //   …
    //   </tool_calls>
    // The args object is wrapped in <parameters>…</parameters> before being
    // placed inside the <tool> body. Strip that wrapper before JSON-parsing.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool name="browse"><parameters>{"action":"navigate","url":"https://example.com"}</parameters></tool>
        \\<tool name="browse"><parameters>{"action":"navigate","url":"https://example.org"}</parameters></tool>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expectEqualStrings("navigate", a0.value.object.get("action").?.string);
    try testing.expectEqualStrings("https://example.com", a0.value.object.get("url").?.string);
    const a1 = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer a1.deinit();
    try testing.expectEqualStrings("https://example.org", a1.value.object.get("url").?.string);
}

test "parseToolCalls DSV4 trailing extra closing brace in args body" {
    // DSV4 occasionally emits an extra `}` after the args object — likely a
    // training-data artifact. Be tolerant: find the first balanced JSON
    // object and ignore trailing garbage.
    const allocator = testing.allocator;
    const text =
        "<tool name=\"browse\"><parameters>{\"action\":\"navigate\",\"url\":\"https://dev.to/agdex_ai/post\"}}</parameters></tool>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expectEqualStrings("navigate", a0.value.object.get("action").?.string);
}

test "parseToolCalls with think block" {
    const allocator = testing.allocator;
    const text = "<think>reasoning</think>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"test\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("search", calls[0].name);
}

test "parseToolCalls raw JSON without tags" {
    const allocator = testing.allocator;
    const text = "{\"name\": \"get_time\", \"arguments\": {}}";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_time", calls[0].name);
}

test "parseToolCalls: self-closing <tool name=... arguments=... />" {
    const allocator = testing.allocator;
    // Clean form. We use the literal byte sequence the model actually emits —
    // unescaped `"` inside the attribute value. That's XML-invalid but
    // perfectly JSON-parseable once we snap to the inner `{…}`.
    const text =
        \\<tool name="shell" arguments="{"command": "ls -la"}"/>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"command\"") != null);
}

test "parseToolCalls: DSV4 broken-quote self-closing fragment (regression repro)" {
    // Verbatim capture from the failing turn in test_agent_stop_repro.py.
    // The model opened arguments with `"`, closed with `'`, embedded
    // unescaped `"` inside the JSON, and finished with `'/>`. The parser
    // must still recover the JSON object and produce a single tool call.
    const allocator = testing.allocator;
    const text =
        "\n\n<tool_calls>\n" ++
        "<tool name=\"shell\" arguments=\"{\"command\": \"ls -la && cat package.json 2>/dev/null || echo 'no package.json yet'\"}'/>\n" ++
        "</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    // Args should be the balanced JSON object — starts with `{` and ends `}`.
    try testing.expect(calls[0].arguments.len > 0);
    try testing.expectEqual(@as(u8, '{'), calls[0].arguments[0]);
    try testing.expectEqual(@as(u8, '}'), calls[0].arguments[calls[0].arguments.len - 1]);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"command\"") != null);
}

test "parseToolCalls: multiple self-closing tool tags in one response" {
    const allocator = testing.allocator;
    const text =
        \\<tool name="cwd" arguments="{"path": "src"}"/>
        \\<tool name="listFiles" arguments="{"recursive": "true"}"/>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("cwd", calls[0].name);
    try testing.expectEqualStrings("listFiles", calls[1].name);
}

test "parseToolCalls: canonical <tool_call>JSON</tool_call> still parses" {
    // Regression guard — adding the self-closing branch must not break the
    // canonical Hermes form.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls: empty body + attribute-args + explicit </tool_call> close" {
    // Verbatim capture from the user's failed chat-history.json. The model
    // emitted args via XML attribute and used `></tool_call>` (not `/>`)
    // as the terminator, with empty body. Without explicit handling this
    // shape parses as "empty body" → no JSON to extract → no tool call.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool_call name="cwd" arguments="{"path": "/Users/david/.mlx-serve/workspace"}"></tool_call>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("cwd", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"path\"") != null);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, ".mlx-serve/workspace") != null);
}

test "parseToolCalls: <tool_call>{JSON} truncated before </tool_call>" {
    // The model emitted `<tool_call>` + well-formed args JSON, then hit
    // EOS before completing the close marker. Output looks like:
    //   "\n\n<tool_call>\n{\"name\": \"writeFile\", \"arguments\":{...}}\n</tool_cal"
    // Previously this fell through to the orphan branch and got silently
    // dropped — that's the second failure mode the agent-stops repro
    // surfaced. We snap to a balanced JSON body and parse it.
    const allocator = testing.allocator;
    const text = "\n\n<tool_call>\n" ++
        \\{"name": "writeFile", "arguments": {"path": "vite.config.js", "content": "ok"}}
        ++ "\n</tool_cal";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"path\"") != null);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"content\"") != null);
}

test "parseToolCalls: mismatched </tool_action> close still parses" {
    // Verbatim DSV4-Flash capture ( pi html-ds4 turn 2): opened
    // with <tool_call>, closed with the hallucinated </tool_action>. The
    // edit call must parse; pre-fix it leaked into visible text and the
    // agent executed nothing.
    const allocator = testing.allocator;
    const text = "<tool_call>\n" ++
        \\{"name": "edit", "arguments": {"path":"mlx.html", "edits":[{"oldText": "  </ul>\n</body>", "newText": "  </ul>\n  <button onclick=\"alert('Hello from MLX')\">Click me</button>\n</body>"}]}
        ++ "\n</tool_action>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("edit", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "mlx.html") != null);
}

test "parseToolCalls: fenced JSON array of parallel calls parses all" {
    const allocator = testing.allocator;
    const text = "```json\n[\n  {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris, France\"}},\n  {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo, Japan\"}}\n]\n```";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "Paris, France") != null);
    try testing.expect(std.mem.indexOf(u8, calls[1].arguments, "Tokyo, Japan") != null);
}

test "parseToolCalls: XML-element tool form (tool_name + arg children)" {
    // Verbatim DSV4-Flash capture (MLX Core agent chat): the
    // tool name and each argument arrive as XML child elements — no JSON.
    const allocator = testing.allocator;
    const text = "Let me check the available disk space on this device.\n\n<tool_calls>\n<tool_name>shell</tool_name>\n<command>df -h / | grep -v \"Filesystem\"</command>\n</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "df -h / | grep -v \\\"Filesystem\\\"") != null);
}

test "parseToolCalls: XML-element-TAG form (<tool_NAME> with arg children)" {
    // Verbatim DSV4-Flash capture ( pi html-ds4 turn 2): the tool
    // name rides in the tag itself — <tool_read>, <tool_edit> — with each
    // argument as an XML child element. Nested elements (the <edits> body)
    // stay verbatim as the arg's string value, matching what working models
    // send pi for the same tool.
    const allocator = testing.allocator;
    const text = "Let me read the current file first.\n\n<tool_read>\n<path>mlx.html</path>\n</tool_read>Now I'll add a button:\n\n<tool_edit>\n<path>mlx.html</path>\n<edits>\n  <oldText>    <h1>MLX</h1></oldText>\n  <newText>    <h1>MLX</h1>\n    <button onclick=\"alert('hi')\">Say Hello</button></newText>\n</edits>\n</tool_edit>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("read", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "mlx.html") != null);
    try testing.expectEqualStrings("edit", calls[1].name);
    // Args must be valid JSON with the nested XML preserved as a string value.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mlx.html", parsed.value.object.get("path").?.string);
    const edits = parsed.value.object.get("edits").?.string;
    try testing.expect(std.mem.indexOf(u8, edits, "<oldText>") != null);
    try testing.expect(std.mem.indexOf(u8, edits, "Say Hello") != null);
}

test "parseToolCalls: XML-element-TAG form with JSON args body" {
    // Verbatim-shape DSV4-Flash capture ( pi html-ds4, second
    // sampling): same name-in-tag form but the body is a bare JSON args
    // object instead of XML elements — `<tool_write>\n{…}\n</tool_write>`.
    const allocator = testing.allocator;
    const text = "Here's the HTML page:\n\n<tool_write>\n{\"path\": \"/tmp/ws/mlx.html\", \"content\": \"<!DOCTYPE html>\\n<html lang=\\\"en\\\">\\n<body>\\n</body>\\n</html>\"}\n</tool_write>\n\npage ready";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/ws/mlx.html", parsed.value.object.get("path").?.string);
    try testing.expect(std.mem.indexOf(u8, parsed.value.object.get("content").?.string, "<!DOCTYPE html>") != null);
}

test "parseToolCalls: <tool_NAME> with prose body containing braces is NOT a tool call" {
    // Prose with incidental {braces} inside an unknown tool-ish tag must not
    // produce arguments — the JSON-body variant requires the body to BE the
    // args object, not merely contain one.
    const allocator = testing.allocator;
    const text = "<tool_summary>The config {a: 1} was applied.</tool_summary>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: hallucinated <tool_output> result tag is NOT a tool call" {
    // Verbatim DSV4-Flash capture (same session, turn 1): the model invented
    // a tool RESULT without calling anything. A plain-text body must never
    // produce a call to a tool named "output".
    const allocator = testing.allocator;
    const text = "Here's the page I created for you:\n\n<tool_output>Page ready: mlx.html</tool_output>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: <tool_output> with element children is still NOT a tool call" {
    // A hallucinated result tag can carry element-shaped content (e.g. an
    // echoed HTML fragment). Result-ish names are denylisted outright.
    const allocator = testing.allocator;
    const text = "<tool_output>\n<status>ok</status>\n</tool_output>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: XML-element form without a tool_name is NOT a tool call" {
    // Prose-ish markup inside a <tool_calls> wrapper must not half-execute.
    const allocator = testing.allocator;
    const text = "<tool_calls>\n<note>this has no tool name element</note>\n</tool_calls>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: prose list array is NOT a tool call" {
    // All-or-nothing: an array whose elements aren't {name, arguments}
    // objects must pass through as text.
    const allocator = testing.allocator;
    const text = "[\"apples\", \"oranges\", \"pears\"]";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: <toolkit> is not a tool tag" {
    // The self-closing scan must not false-fire on HTML-ish prose. `<toolkit>`
    // has `<tool` as a prefix but the terminator (`k`) isn't valid, so the
    // outer scan rejects it before parseSelfClosingToolTag even runs.
    const allocator = testing.allocator;
    const text = "Here's a doc about <toolkit> and <toolbar>. No actual tool call.";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

// ── Schema-aware argument coercion (value-spelling type-inference class) ──

test "parseXmlElementToolCall is already dup-key + escape safe (characterization)" {
    // The DSV4 XML-element path builds args via std.json.ObjectMap.put + Stringify,
    // which dedups keys (last wins) and escapes values — so it is structurally
    // immune to the dup-key/raw-interpolation class that bit parseHermesToolCall
    // and convertGemma4Object. This characterization test pins that guarantee.
    const allocator = testing.allocator;
    const raw = "<tool_calls>\n<tool_name>shell</tool_name>\n<command>echo \"a\"</command>\n<command>echo \"b\"</command>\n</tool_calls>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    // Valid JSON despite the duplicate <command> and the inner quotes.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    try testing.expect(parsed.value.object.get("command") != null);
}
