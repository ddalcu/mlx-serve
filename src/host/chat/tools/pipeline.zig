//! parseToolCallsForRequest chokepoint, finish reason, tool deltas

const std = @import("std");
const crt = @import("../../../sdk/util/crt.zig");
const chat_mod = @import("../chat.zig");
const log = @import("../../../sdk/util/log.zig");
const http_server = @import("../../http/server.zig");

const jsonEscape = @import("../../api/format.zig").jsonEscape;

/// The auto budget's own tightness question: under a quarter of the window left.
pub fn autoBudgetWindowTight(remaining: u32, effective_ctx: u32) bool {
    return remaining < effective_ctx / 4;
}

/// Finish reason for a response whose generated text yielded tool calls.
/// A TRUNCATED generation ("length": max_tokens or the request timeout) must
/// keep reporting "length" even when the cut-off text salvaged a partial tool
/// call (the truncated-opener recovery in chat.parseToolCalls) — clients key
/// their truncation recovery on it (the app's chunk-and-retry nudge fires on
/// finish_reason "length" + tool calls present). Overriding to "tool_calls"
/// hides the cut: the client executes a half-argument call, blames the model
/// ("you sent no content"), and the model re-emits the same doomed mega-call.
pub fn toolCallFinishReason(pre_parse: []const u8) []const u8 {
    return if (std.mem.eql(u8, pre_parse, "length")) "length" else "tool_calls";
}

/// The `tool_calls` array of one streamed chat-completions delta: name, id and
/// the full arguments in ONE delta, every string escaped.
pub fn streamToolCallDelta(allocator: std.mem.Allocator, index: usize, id: []const u8, name: []const u8, arguments: []const u8) ![]u8 {
    const esc_name = try jsonEscape(allocator, name);
    defer allocator.free(esc_name);
    const esc_args = try jsonEscape(allocator, arguments);
    defer allocator.free(esc_args);
    return std.fmt.allocPrint(allocator,
        \\[{{"index":{d},"id":"{s}","type":"function","function":{{"name":{s},"arguments":{s}}}}}]
    , .{ index, id, esc_name, esc_args });
}

/// A repetition-loop cut may land inside an otherwise recognizable tool call.
/// Never emit that buffer as executable work: unlike a genuine max-token cut,
/// this intentional stop must not ask clients to recover by compacting/retrying.
pub fn shouldParseToolCalls(finish_details: ?[]const u8) bool {
    const d = finish_details orelse return true;
    return !std.mem.eql(u8, d, "repetition_loop");
}

/// The `finish_details` object emitted BESIDE `finish_reason` on a choice,
/// or "" when there is nothing to say. Comes with its leading comma so call
/// sites splice it straight into the choice literal.
///
/// The loop guard uses `finish_reason: "stop"`; the sibling preserves why the
/// server stopped without overloading "length" (which makes agents run output-
/// or context-exhaustion recovery). OpenAI's deprecated `finish_details` is the
/// closest precedent, and conforming clients ignore keys they don't know.
pub fn finishDetailsField(reason: []const u8, details: ?[]const u8) []const u8 {
    const d = details orelse return "";
    // Every emitter can override the slot's reason after the fact. Gate on the
    // reason actually being emitted so a cause never accompanies a genuine
    // length exhaustion, tool completion, or client disconnect.
    if (!std.mem.eql(u8, reason, "stop")) return "";
    // One known value today; a switch here keeps an unknown string from
    // reaching the wire as an unescaped literal.
    if (std.mem.eql(u8, d, "repetition_loop")) return ",\"finish_details\":{\"type\":\"repetition_loop\"}";
    return "";
}

/// Tokens a client should be shown after a loop cut: everything before the
/// degenerate span. `start` is an index into the SAME emitted-token sequence,
/// but it is computed from the generator's own list, so it is clamped rather
/// than trusted — a mismatch must degrade to "emit everything", never to a
/// slice out of bounds.
pub fn loopTrimmedIds(ids: []const u32, start: ?usize) []const u32 {
    const s = start orelse return ids;
    if (s >= ids.len) return ids;
    return ids[0..s];
}

/// Debug-only corpus-harvest aid (`MLX_SERVE_RAW_DUMP_FILE`). Appends ONE framed
/// record per tools request: the DECLARED TOOLS SCHEMA and the raw pre-parse text
/// together, written at the one site where both are in scope. Correlating them
/// any other way is unsound — the debug log truncates lines at 16 KB (so a large
/// request body never re-parses, and a harvester silently pairs a model output
/// with some EARLIER request's schema), and concurrent requests interleave.
///
/// Framing is by byte COUNT, never a delimiter: both payloads are arbitrary bytes
/// and any sentinel can occur inside them. libc `write(2)` like log.zig's sink
/// (callers here carry no `Io`); O_APPEND keeps conn threads from interleaving.
pub fn appendRawToolDump(path: []const u8, tools_json: ?[]const u8, text: []const u8) void {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len == 0 or path.len >= path_buf.len) return;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    const path_z: [*:0]const u8 = @ptrCast(&path_buf);

    const fd = crt.openWrite(path_z, false);
    if (fd < 0) return;
    defer crt.close(fd);

    const tools = tools_json orelse "";
    var hdr: [96]u8 = undefined;
    const h = std.fmt.bufPrint(&hdr, "\n===MLX_RAW_DUMP tools={d} raw={d}===\n", .{ tools.len, text.len }) catch return;
    _ = crt.write(fd, h);
    if (tools.len > 0) _ = crt.write(fd, tools);
    if (text.len > 0) _ = crt.write(fd, text);
}

/// The ONE tool-call extraction path for every HTTP surface: parse, fall back to
/// bare-JSON inference, then coerce the arguments to the schema the request
/// declared. Every surface must go through this — a site that calls
/// `chat_mod.parseToolCalls` directly silently reintroduces the value-spelling
/// type-inference bug (a boolean param arriving as the string "False") on that
/// surface alone, which no output-equality test can see.
pub fn parseToolCallsForRequest(
    allocator: std.mem.Allocator,
    text: []const u8,
    tools_json: ?[]const u8,
    /// OpenAI `parallel_tool_calls` (Anthropic spelling:
    /// !tool_choice.disable_parallel_tool_use). false = the client accepts AT
    /// MOST ONE call per response, so the chokepoint keeps the first parsed
    /// call and drops the rest (the model re-issues them next round after the
    /// first result). Deliberately NOT gated by --no-tool-autocorrect:
    /// client-requested behavior, not a repair heuristic.
    allow_parallel: bool,
) !?[]chat_mod.ParsedToolCall {
    var parsed_calls = try chat_mod.parseToolCalls(allocator, text);
    // Heuristically-inferred raw-JSON calls must name a DECLARED tool — a
    // truncated data object ({"name": "George Washington", …}) is not a call.
    // Deliberately NOT gated on g_tool_autocorrect: this corrects our own
    // heuristic's false positive, not the model's output.
    if (parsed_calls) |c| {
        if (tools_json) |tj| parsed_calls = try chat_mod.filterInferredBySchema(allocator, c, tj);
    }
    var calls = parsed_calls orelse
        (if (tools_json) |tj| try chat_mod.inferBareJsonToolCalls(allocator, text, tj) else null) orelse
        return null;
    if (!allow_parallel and calls.len > 1) {
        log.info("  parallel_tool_calls=false: clamping {d} parsed calls to the first\n", .{calls.len});
        for (calls[1..]) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        const kept = calls[0];
        allocator.free(calls);
        const one = try allocator.alloc(chat_mod.ParsedToolCall, 1);
        one[0] = kept;
        calls = one;
    }
    if (http_server.g_tool_autocorrect) {
        if (tools_json) |tj| {
            // Order matters: put a buried required param back where the schema
            // says it lives BEFORE coercing types, so the hoisted value is
            // type-checked like any other top-level arg.
            try chat_mod.hoistMisplacedRequiredParams(allocator, calls, tj);
            try chat_mod.coerceToolArgsToSchema(allocator, calls, tj);
        }
    }
    return calls;
}

// ── tests ──

test "streamed tool-call delta escapes the name and arguments (#748)" {
    const allocator = std.testing.allocator;
    const name = "read_file\n</parameter \"x\" \\";
    const args = "{\"path\":\"a\\nb\"}";
    const delta = try streamToolCallDelta(allocator, 0, "call_1_0", name, args);
    defer allocator.free(delta);
    try std.testing.expect(std.mem.indexOfScalar(u8, delta, '\n') == null);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, delta, .{});
    defer parsed.deinit();
    const function = parsed.value.array.items[0].object.get("function").?.object;
    try std.testing.expectEqualStrings(name, function.get("name").?.string);
    try std.testing.expectEqualStrings(args, function.get("arguments").?.string);
}

test "parseToolCallsForRequest coerces args to the schema (server-side chokepoint wiring)" {
    // The chokepoint every HTTP surface routes through must actually invoke the
    // schema coercion — not just parse. This pins the WIRING (a mis-wire that
    // parsed but skipped coerce would reintroduce the string-"False" bug on every
    // surface at once). Uses the exact Hermes-XML `replace_all=False` shape from
    // the live Claude Code failure.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const ra = parsed.value.object.get("replace_all").?;
    try std.testing.expect(ra == .bool); // coerced from the STRING "False"
    try std.testing.expectEqual(false, ra.bool);
    // Null tools_json path must still parse (no coercion, no crash).
    const calls2 = try parseToolCallsForRequest(allocator, text, null, true);
    defer if (calls2) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    };
    try std.testing.expect(calls2 != null);
}

test "parseToolCallsForRequest hoists a misplaced required param (server-side chokepoint wiring)" {
    // Live pi session, gemma-4-26B-A4B: the model buried `path` inside
    // each edits[] item and emitted none at the top level, so pi rejected three
    // consecutive multi-thousand-token generations with
    //   Validation failed for tool "edit": - path: must have required properties path
    // This pins the WIRING — every HTTP surface reaches the hoist through this one
    // function, and the hoist must run BEFORE coercion so the lifted value is
    // type-checked like any other top-level arg.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["oldText","newText"]}}},"required":["path","edits"]}}}]
    ;
    const text = "<tool_call>{\"name\":\"edit\",\"arguments\":{\"edits\":[{\"newText\":\"new\",\"oldText\":\"old\"," ++
        "\"path\":\"us_presidents/generate_site.sh\"}]}}</tool_call>";

    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try std.testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);
    try std.testing.expect(parsed.value.object.get("edits").?.array.items[0].object.get("path") == null);
    try std.testing.expect(chat_mod.toolCallConformsToSchema(allocator, calls[0], tools));

    // The escape hatch covers this correction too: it repairs the MODEL's output
    // (unlike the inferred-name filter, which corrects our own heuristic).
    http_server.g_tool_autocorrect = false;
    defer http_server.g_tool_autocorrect = true;
    const raw_calls = (try parseToolCallsForRequest(allocator, text, tools, true)).?;
    defer {
        for (raw_calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(raw_calls);
    }
    const raw_parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw_calls[0].arguments, .{});
    defer raw_parsed.deinit();
    try std.testing.expect(raw_parsed.value.object.get("path") == null); // verbatim: still buried
}

test "parseToolCallsForRequest: --no-tool-autocorrect leaves args verbatim" {
    // The escape hatch: with g_tool_autocorrect off, the schema coercion is
    // skipped and the model's value passes through EXACTLY (here the STRING
    // "False", which is what a strict client would then reject — the user's
    // choice). Parse-repair + valid-JSON safety net still run, so args stay
    // well-formed JSON regardless.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";

    http_server.g_tool_autocorrect = false;
    defer http_server.g_tool_autocorrect = true; // restore the default for other tests
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const ra = parsed.value.object.get("replace_all").?;
    try std.testing.expect(ra == .string); // NOT coerced — verbatim
    try std.testing.expectEqualStrings("False", ra.string);
}

test "parseToolCallsForRequest: truncated DATA object is not promoted to a tool call (George Washington class)" {
    // Live pi capture (Qwen3.6-35B-A3B distilled): generation hit
    // max_tokens midway through a presidents data script. The raw-JSON fallback
    // found the first balanced object — {"name": "George Washington", "num": 1,
    // …} — and the flat-shape synthesis promoted it to a TOOL CALL named
    // "George Washington" (args = every key but "name"). pi answered "Tool
    // George Washington not found", the model retried the identical mega-write,
    // and the session burned two full 16K-token turns making zero progress.
    // A HEURISTICALLY inferred call (no tag syntax — the model never said
    // "tool call") must carry a name the request actually declared; otherwise
    // the text stays visible and finish_reason="length" reaches the client
    // untouched, so its truncation recovery fires instead of a bogus tool loop.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    const text = "Now let me build a generator script that creates all 46 president pages with real historical data:\n\n" ++
        "presidents = [\n" ++
        "  {\"name\": \"George Washington\", \"num\": 1, \"party\": \"None (Federalist-leaning)\", \"term\": \"1789\u{2013}1797\", \"vice\": \"John Adams\"},\n" ++
        "  {\"name\": \"John Adams\", \"num\": 2, \"party\": \"Federalist\",";
    const calls = try parseToolCallsForRequest(allocator, text, tools, true);
    defer if (calls) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    };
    try std.testing.expect(calls == null);
}

test "parseToolCallsForRequest: raw-JSON call with a DECLARED name still parses" {
    // The counterweight to the George Washington guard: models without a
    // trained tool format (Gemma 3) emit fenced raw-JSON calls, and those must
    // keep working when the name matches a declared tool.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
    ;
    const text = "```json\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.txt\", \"content\": \"hi\"}}\n```";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("write", calls[0].name);
}

test "parseToolCallsForRequest: tag-format call with an UNDECLARED name is kept" {
    // An EXPLICIT tag-format call (<tool_call>…) to a name the request never
    // declared still goes to the client — "tool not found" is model-visible
    // feedback the model can correct from. Only HEURISTIC raw-JSON inference
    // gets schema-name validation; this pins the filter against over-reach.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    const text = "<tool_call>{\"name\":\"searchWeb\",\"arguments\":{\"q\":\"zig\"}}</tool_call>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("searchWeb", calls[0].name);
}

test "parseToolCallsForRequest: parallel_tool_calls=false clamps to the FIRST call" {
    // OpenAI contract: parallel_tool_calls=false means AT MOST ONE call per
    // response (the SDK sets it in strict structured-output mode; those
    // clients execute tool_calls[0] and assume length 1). We can't constrain
    // generation, so the chokepoint clamps post-parse: keep the first, drop
    // the rest — the model re-issues them next round after seeing the first
    // result. Whole valid calls only; nothing partial ships.
    const allocator = std.testing.allocator;
    const text = "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}</tool_call>\n" ++
        "<tool_call>{\"name\":\"write\",\"arguments\":{\"path\":\"b\"}}</tool_call>";

    // allow_parallel=true (default wiring): both calls survive.
    const both = (try parseToolCallsForRequest(allocator, text, null, true)).?;
    defer {
        for (both) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(both);
    }
    try std.testing.expectEqual(@as(usize, 2), both.len);

    // allow_parallel=false: only the FIRST survives, intact.
    const one = (try parseToolCallsForRequest(allocator, text, null, false)).?;
    defer {
        for (one) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(one);
    }
    try std.testing.expectEqual(@as(usize, 1), one.len);
    try std.testing.expectEqualStrings("read", one[0].name);

    // A single call with the flag off passes through untouched.
    const single_text = "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}</tool_call>";
    const single = (try parseToolCallsForRequest(allocator, single_text, null, false)).?;
    defer {
        for (single) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(single);
    }
    try std.testing.expectEqual(@as(usize, 1), single.len);
}

test "parseToolCallsForRequest: MiniCPM5 V3 XML through the real server chokepoint" {
    // Exercises the EXACT function every HTTP dispatch site in this file calls,
    // proving the new dialect is wired all the way through — not just reachable
    // from chat.parseToolCalls in isolation. Uses the Agent shell tool's real
    // declared schema shape.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"shell","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
    ;
    const text = "<function name=\"shell\">\n  <param name=\"command\">git status</param>\n</function>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("git status", parsed.value.object.get("command").?.string);
}

test "parseToolCallsForRequest: coercion fires across think on/off × qwen/gemma" {
    // The tool-call parse strips the leading think block BEFORE parsing args, and
    // the two families strip DIFFERENTLY (Qwen `<think>…</think>`, Gemma
    // `<|channel>thought…<channel|>`). Pin that a preceding reasoning block never
    // blocks the schema coercion, for BOTH families, WITH and WITHOUT thinking.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const gtools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;

    const Case = struct { name: []const u8, text: []const u8, tools: []const u8 };
    const cases = [_]Case{
        // Qwen3.6-MoE thinking OFF — Hermes XML, no reasoning block.
        .{ .name = "qwen think-off", .tools = tools, .text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>" },
        // Qwen3.6-MoE thinking ON — a `<think>…</think>` block precedes the call.
        .{ .name = "qwen think-on", .tools = tools, .text = "<think>\nThe user wants replace_all disabled, so I'll pass it false.\n</think>\n<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>" },
        // Gemma thinking OFF — custom `call:` form, no channel block.
        .{ .name = "gemma think-off", .tools = gtools, .text = "<|tool_call>call:Edit{file_path:<|\"|>/tmp/x<|\"|>,replace_all:False}<tool_call|>" },
        // Gemma thinking ON — a `<|channel>thought…<channel|>` block precedes.
        .{ .name = "gemma think-on", .tools = gtools, .text = "<|channel>thought\nI should disable replace_all for a single edit.<channel|>\n<|tool_call>call:Edit{file_path:<|\"|>/tmp/x<|\"|>,replace_all:False}<tool_call|>" },
    };

    for (cases) |c| {
        const calls = (try parseToolCallsForRequest(allocator, c.text, c.tools, true)) orelse {
            std.debug.print("\n[{s}] no tool call parsed\n", .{c.name});
            return error.NoToolCall;
        };
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try std.testing.expectEqualStrings("Edit", calls[0].name);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        const ra = parsed.value.object.get("replace_all") orelse {
            std.debug.print("\n[{s}] replace_all missing in {s}\n", .{ c.name, calls[0].arguments });
            return error.ArgMissing;
        };
        if (ra != .bool or ra.bool != false) {
            std.debug.print("\n[{s}] replace_all NOT coerced to bool false: {s}\n", .{ c.name, calls[0].arguments });
            return error.NotCoerced;
        }
    }
}

test "toolCallFinishReason preserves truncation over parsed tool calls" {
    // A truncated generation ("length": max_tokens OR request timeout) whose
    // cut-off text still salvaged a partial tool call must KEEP reporting
    // "length" — clients key their truncation recovery (chunk-and-retry
    // nudges) on it. Reporting "tool_calls" hides the cut: the client
    // executes a half-argument call and tells the model IT made a mistake.
    // Live capture Qwen3.6-27B 33KB writeFile guillotined by the
    // 300s default timeout arrived as {"path":...} with finish "tool_calls".
    try std.testing.expectEqualStrings("length", toolCallFinishReason("length"));
    // Everything else stays the normal tool-call finish.
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("stop"));
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("tool_calls"));
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("client_disconnect"));
}

test "repetition-loop cuts decline tool-call parsing" {
    // The loop detector can fire inside a tool argument. The response must
    // remain a safe terminal stop without emitting an executable fragment.
    try std.testing.expect(shouldParseToolCalls(null));
    try std.testing.expect(!shouldParseToolCalls("repetition_loop"));
    try std.testing.expect(shouldParseToolCalls("unknown_future_detail"));
}

test "finishDetailsField: the loop cause rides beside finish_reason, and only a known cause reaches the wire" {
    // Absent = the field is not emitted at all, so every ordinary response
    // is byte-identical to what it was before this existed.
    try std.testing.expectEqualStrings("", finishDetailsField("stop", null));
    try std.testing.expectEqualStrings(
        ",\"finish_details\":{\"type\":\"repetition_loop\"}",
        finishDetailsField("stop", "repetition_loop"),
    );
    // An unknown value is DROPPED rather than interpolated: this string is
    // spliced into a JSON literal, and a literal is arbitrary bytes too (the
    // media-gen `sendError` class). A future cause adds an arm here.
    try std.testing.expectEqualStrings("", finishDetailsField("stop", "something new"));
    try std.testing.expectEqualStrings("", finishDetailsField("stop", "\",\"x\":\""));
    // The cause belongs only beside the intentional loop stop.
    try std.testing.expectEqualStrings("", finishDetailsField("length", "repetition_loop"));
    try std.testing.expectEqualStrings("", finishDetailsField("tool_calls", "repetition_loop"));
    try std.testing.expectEqualStrings("", finishDetailsField("client_disconnect", "repetition_loop"));
}

test "loopTrimmedIds: the degenerate span is cut, and a bad index degrades to emitting everything" {
    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    // No cut → untouched (identity for every non-loop response).
    try std.testing.expectEqualSlices(u32, &ids, loopTrimmedIds(&ids, null));
    try std.testing.expectEqualSlices(u32, ids[0..3], loopTrimmedIds(&ids, 3));
    // A whole-generation loop leaves nothing: honest, and finish_reason
    // "length" already says the server cut it.
    try std.testing.expectEqualSlices(u32, ids[0..0], loopTrimmedIds(&ids, 0));
    // The index comes from the generator's own token list, so a mismatch
    // must emit everything rather than slice out of bounds.
    try std.testing.expectEqualSlices(u32, &ids, loopTrimmedIds(&ids, 5));
    try std.testing.expectEqualSlices(u32, &ids, loopTrimmedIds(&ids, 99));
}
