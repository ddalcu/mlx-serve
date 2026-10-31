//! The Harmony, ATEM, and bare `<invoke>` tool-call dialects.

const std = @import("std");
const testing = std.testing;
const log = @import("../../../sdk/util/log.zig");

const HARMONY_START_TAG = @import("../reasoning/harmony.zig").HARMONY_START_TAG;
const HARMONY_CHANNEL_TAG = @import("../reasoning/harmony.zig").HARMONY_CHANNEL_TAG;
const HARMONY_MSG_TAG = @import("../reasoning/harmony.zig").HARMONY_MSG_TAG;
const HARMONY_END_TAG = @import("../reasoning/harmony.zig").HARMONY_END_TAG;
const HARMONY_RETURN_TAG = @import("../reasoning/harmony.zig").HARMONY_RETURN_TAG;
const HARMONY_CALL_TAG = @import("../reasoning/harmony.zig").HARMONY_CALL_TAG;
const museIsRecipientChar = @import("../reasoning/muse.zig").museIsRecipientChar;
const parseToolCalls = @import("parse.zig").parseToolCalls;
const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const balancedJsonObject = @import("json_infer.zig").balancedJsonObject;
const inklingTrailingNameRun = @import("inkling.zig").inklingTrailingNameRun;
const dsmlAttr = @import("dsml.zig").dsmlAttr;
const freeParsedCalls = @import("dsml.zig").freeParsedCalls;

// ── ATEM tool calls (Muse-Glimmer / `muse_glimmer`) ────────────────────
//
//   <atem:function_calls>
//   <atem:invoke name="weather.get">
//   <atem:parameter name="city">Paris</atem:parameter>
//   <atem:parameter name="detailed">false</atem:parameter>
//   </atem:invoke>
//   </atem:function_calls>
//
// String parameters are RAW bytes (the reference explicitly does not strip
// spaces — the Hermes trim gap stays Hermes'); lists/objects/bools/null/
// numbers are spelled as JSON and keep their spelled type (the schema
// coercion chokepoint still gets the final word). The parser keys on the
// INVOKE tag so a dropped wrapper still parses (delimiter-drop class), and
// truncation salvages NAME + completed parameters — never fragment values.
const InvokeDialect = struct {
    invoke: []const u8,
    param: []const u8,
    param_close: []const u8,
    invoke_close: []const u8,
    /// Not the model's own dialect: kept only when it names a declared tool (`filterInferredBySchema`).
    inferred: bool,
};
pub const ATEM_INVOKE: InvokeDialect = .{ .invoke = "<atem:invoke", .param = "<atem:parameter", .param_close = "</atem:parameter>", .invoke_close = "</atem:invoke>", .inferred = false };
/// Anthropic's spelling of the same grammar (`<function_calls><invoke name="X"><parameter name="k">`),
/// which models trained on Claude transcripts emit in place of their own (qwen4_exp, #928).
pub const BARE_INVOKE: InvokeDialect = .{ .invoke = "<invoke", .param = "<parameter", .param_close = "</parameter>", .invoke_close = "</invoke>", .inferred = true };
pub const FUNCTION_CALLS_TAG = "<function_calls>";

/// gpt_oss / harmony tool calls. Structurally unlike every other family here:
/// the function NAME is not in the payload at all, it is the segment header's
/// recipient (`to=functions.NAME`), and the body is already a JSON arguments
/// object rather than a markup block to assemble.
///
///     <|start|>assistant to=functions.get_weather<|channel|>commentary
///     <|constrain|>json<|message|>{"location":"SF"}<|call|>
///
/// Keyed on `to=functions.` sitting in a resolved channel header, a shape no
/// other family emits. A body that is not a COMPLETE balanced JSON object is
/// a truncated call: ship the name with `{}` rather than a fragment, per the
/// same rule the ATEM and DSML parsers follow.
pub fn parseHarmonyToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    if (std.mem.indexOf(u8, text, HARMONY_CHANNEL_TAG) == null) return;
    var seg_start: usize = 0;
    while (true) {
        const next = std.mem.indexOfPos(u8, text, seg_start + 1, HARMONY_START_TAG);
        const seg_end = next orelse text.len;
        defer seg_start = next orelse text.len;

        const seg = text[seg_start..seg_end];
        // Only a RESOLVED header can name a call — an unterminated one is
        // still arriving and its recipient may yet change.
        const msg_rel = std.mem.indexOf(u8, seg, HARMONY_MSG_TAG) orelse {
            if (next == null) break;
            continue;
        };
        const header = seg[0..msg_rel];
        const fn_at = std.mem.indexOf(u8, header, "to=functions.") orelse {
            if (next == null) break;
            continue;
        };
        const raw_name = header[fn_at + "to=functions.".len ..];
        var n: usize = 0;
        while (n < raw_name.len and museIsRecipientChar(raw_name[n])) n += 1;
        // A tool name never carries a channel marker (corpus invariant).
        const name_slice = raw_name[0..n];
        if (name_slice.len == 0) {
            if (next == null) break;
            continue;
        }

        var body = seg[msg_rel + HARMONY_MSG_TAG.len ..];
        for ([_][]const u8{ HARMONY_CALL_TAG, HARMONY_END_TAG, HARMONY_RETURN_TAG }) |tag| {
            if (std.mem.indexOf(u8, body, tag)) |e| body = body[0..e];
        }
        const args_src = balancedJsonObject(std.mem.trim(u8, body, " \n\t\r")) orelse "{}";

        const name = try allocator.dupe(u8, name_slice);
        errdefer allocator.free(name);
        const arguments = try allocator.dupe(u8, args_src);
        try calls.append(allocator, .{ .name = name, .arguments = arguments, .inferred = false });

        if (next == null) break;
    }
}

pub fn parseInvokeToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall), comptime d: InvokeDialect) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, d.invoke)) |inv| {
        const after_tag = inv + d.invoke.len;
        const seg = text[after_tag..];
        // `<invoked>` is a word, not the tag.
        if (seg.len > 0 and !std.ascii.isWhitespace(seg[0])) {
            pos = after_tag;
            continue;
        }
        // Attribute scan bounded so a runaway unclosed quote can't eat the
        // rest of the output.
        const raw_name = dsmlAttr(seg[0..@min(seg.len, 512)], "name") orelse {
            pos = after_tag;
            continue;
        };
        // No real tool name carries a channel marker (corpus invariant —
        // the inkling marker-echo class).
        var name_slice = raw_name;
        if (std.mem.indexOf(u8, name_slice, "<|") != null) {
            name_slice = inklingTrailingNameRun(name_slice, name_slice.len);
        }
        if (name_slice.len == 0) {
            pos = after_tag;
            continue;
        }

        var args_json: ?[]const u8 = null;
        if (std.mem.indexOfScalar(u8, seg, '>')) |te| {
            const body_all = seg[te + 1 ..];
            const body_end = std.mem.indexOf(u8, body_all, d.invoke_close) orelse body_all.len;
            // An inferred opener with no parameter and no close is prose, as `bareInvokeMarkupCut` reads it.
            if (d.inferred and body_end == body_all.len and std.mem.indexOf(u8, body_all, d.param) == null) {
                pos = after_tag;
                continue;
            }
            args_json = try invokeParamsToJson(allocator, body_all[0..body_end], d);
            pos = if (body_end < body_all.len)
                after_tag + te + 1 + body_end + d.invoke_close.len
            else
                text.len; // truncated body — completed params already salvaged
        } else {
            if (d.inferred) break; // an unclosed opener alone is prose
            pos = text.len; // truncated inside the opening tag: NAME + {}
        }

        const name = try allocator.dupe(u8, name_slice);
        errdefer allocator.free(name);
        const arguments = args_json orelse try allocator.dupe(u8, "{}");
        try calls.append(allocator, .{ .name = name, .arguments = arguments, .inferred = d.inferred });
    }
}

/// Assemble `<atem:parameter name="k">v</atem:parameter>` runs into a JSON
/// object string. A value that parses as COMPLETE JSON keeps its spelled type;
/// everything else is a raw string, exact bytes. Keys are escaped + deduped
/// (last wins) via ObjectMap+Stringify, so emitted arguments are ALWAYS valid
/// JSON. Returns null when no complete parameter was found.
fn invokeParamsToJson(allocator: std.mem.Allocator, body: []const u8, comptime d: InvokeDialect) !?[]const u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var map: std.json.ObjectMap = .empty;
    var pos: usize = 0;
    var found = false;
    while (std.mem.indexOfPos(u8, body, pos, d.param)) |p| {
        const after = p + d.param.len;
        const seg = body[after..];
        const te = std.mem.indexOfScalar(u8, seg, '>') orelse break; // truncated open tag
        const pname = dsmlAttr(seg[0 .. te + 1], "name") orelse {
            pos = after;
            continue;
        };
        const vstart = after + te + 1;
        // The value runs to the CONFIRMED closing tag — rich values with
        // newlines/quotes/`<` survive; no close by end of text = fragment,
        // dropped.
        const vend_rel = std.mem.indexOf(u8, body[vstart..], d.param_close) orelse break;
        const raw = body[vstart .. vstart + vend_rel];
        const v: std.json.Value = std.json.parseFromSliceLeaky(std.json.Value, aa, raw, .{}) catch
            std.json.Value{ .string = raw };
        try map.put(aa, pname, v);
        found = true;
        pos = vstart + vend_rel + d.param_close.len;
    }
    if (!found) return null;
    return try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = map }, .{});
}

// ── tests ──

test "parseHarmonyToolCalls: name comes from the HEADER, arguments from the body" {
    // gpt-oss puts the tool name in the segment header's recipient
    // (`to=functions.NAME`) and the arguments — already JSON — in the body.
    // Nothing in the body names the function, so a body-only parser finds
    // an anonymous JSON object and infers nothing.
    const text = "<|start|>assistant to=functions.get_weather<|channel|>commentary " ++
        "<|constrain|>json<|message|>{\"location\": \"SF\", \"unit\": \"c\"}<|call|>";
    var calls: std.ArrayList(ParsedToolCall) = .empty;
    defer {
        for (calls.items) |tc| {
            std.testing.allocator.free(tc.name);
            std.testing.allocator.free(tc.arguments);
        }
        calls.deinit(std.testing.allocator);
    }
    try parseHarmonyToolCalls(std.testing.allocator, text, &calls);
    try std.testing.expectEqual(@as(usize, 1), calls.items.len);
    try std.testing.expectEqualStrings("get_weather", calls.items[0].name);
    try std.testing.expectEqualStrings("{\"location\": \"SF\", \"unit\": \"c\"}", calls.items[0].arguments);
    try std.testing.expect(!calls.items[0].inferred);
}

test "parseHarmonyToolCalls: recipient after the channel name parses the same" {
    const text = "<|start|>assistant<|channel|>commentary to=functions.ping<|message|>{\"n\":1}<|call|>";
    var calls: std.ArrayList(ParsedToolCall) = .empty;
    defer {
        for (calls.items) |tc| {
            std.testing.allocator.free(tc.name);
            std.testing.allocator.free(tc.arguments);
        }
        calls.deinit(std.testing.allocator);
    }
    try parseHarmonyToolCalls(std.testing.allocator, text, &calls);
    try std.testing.expectEqual(@as(usize, 1), calls.items.len);
    try std.testing.expectEqualStrings("ping", calls.items[0].name);
    try std.testing.expectEqualStrings("{\"n\":1}", calls.items[0].arguments);
}

test "parseHarmonyToolCalls: truncated arguments ship NAME + {}, never a fragment" {
    // Cut mid-value. The call is real (the header resolved), so the name must
    // survive, but a partial value must never be presented as an argument.
    const text = "<|start|>assistant to=functions.search<|channel|>commentary<|message|>{\"q\": \"how do I";
    var calls: std.ArrayList(ParsedToolCall) = .empty;
    defer {
        for (calls.items) |tc| {
            std.testing.allocator.free(tc.name);
            std.testing.allocator.free(tc.arguments);
        }
        calls.deinit(std.testing.allocator);
    }
    try parseHarmonyToolCalls(std.testing.allocator, text, &calls);
    try std.testing.expectEqual(@as(usize, 1), calls.items.len);
    try std.testing.expectEqualStrings("search", calls.items[0].name);
    try std.testing.expectEqualStrings("{}", calls.items[0].arguments);
}

test "parseHarmonyToolCalls: analysis and final segments are NOT tool calls" {
    const text = "<|channel|>analysis<|message|>thinking<|end|>" ++
        "<|start|>assistant<|channel|>final<|message|>done<|return|>";
    var calls: std.ArrayList(ParsedToolCall) = .empty;
    defer {
        for (calls.items) |tc| {
            std.testing.allocator.free(tc.name);
            std.testing.allocator.free(tc.arguments);
        }
        calls.deinit(std.testing.allocator);
    }
    try parseHarmonyToolCalls(std.testing.allocator, text, &calls);
    try std.testing.expectEqual(@as(usize, 0), calls.items.len);
}

test "parseHarmonyToolCalls: two tool segments both parse" {
    const text = "<|start|>assistant to=functions.a<|channel|>commentary<|message|>{\"x\":1}<|call|>" ++
        "<|start|>assistant to=functions.b<|channel|>commentary<|message|>{\"y\":2}<|call|>";
    var calls: std.ArrayList(ParsedToolCall) = .empty;
    defer {
        for (calls.items) |tc| {
            std.testing.allocator.free(tc.name);
            std.testing.allocator.free(tc.arguments);
        }
        calls.deinit(std.testing.allocator);
    }
    try parseHarmonyToolCalls(std.testing.allocator, text, &calls);
    try std.testing.expectEqual(@as(usize, 2), calls.items.len);
    try std.testing.expectEqualStrings("a", calls.items[0].name);
    try std.testing.expectEqualStrings("b", calls.items[1].name);
}

test "parseAtemToolCalls: canonical block — raw strings, spelled types, schema untouched" {
    const allocator = testing.allocator;
    const text = " to=weather.get<|message|><atem:function_calls>\n<atem:invoke name=\"weather.get\">\n<atem:parameter name=\"city\">Paris, France</atem:parameter>\n<atem:parameter name=\"detailed\">false</atem:parameter>\n<atem:parameter name=\"days\">3</atem:parameter>\n<atem:parameter name=\"tags\">[\"trip\", \"eu\"]</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("weather.get", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try testing.expectEqualStrings("Paris, France", obj.get("city").?.string);
    try testing.expectEqual(false, obj.get("detailed").?.bool);
    try testing.expectEqual(@as(i64, 3), obj.get("days").?.integer);
    try testing.expectEqual(@as(usize, 2), obj.get("tags").?.array.items.len);
}

test "parseAtemToolCalls: multiline string values keep exact bytes (no trimming)" {
    const allocator = testing.allocator;
    const text = "<atem:invoke name=\"fs.write\">\n<atem:parameter name=\"content\">line one\n  \"quoted\" line\n</atem:parameter>\n</atem:invoke>";
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
    // Exact bytes incl. the trailing newline — the reference does not strip.
    try testing.expectEqualStrings("line one\n  \"quoted\" line\n", parsed.value.object.get("content").?.string);
}

test "parseAtemToolCalls: truncation salvages NAME + completed params, never fragments" {
    const allocator = testing.allocator;
    // Cut mid-value: the completed param ships, the fragment is dropped.
    {
        const text = "<atem:invoke name=\"weather.get\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n<atem:parameter name=\"units\">celsi";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("weather.get", calls[0].name);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings("Paris", parsed.value.object.get("city").?.string);
        try testing.expect(parsed.value.object.get("units") == null);
    }
    // Cut right after the opener: NAME + {}.
    {
        const text = "<atem:function_calls>\n<atem:invoke name=\"weather.get\">";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("weather.get", calls[0].name);
        try testing.expectEqualStrings("{}", calls[0].arguments);
    }
}

test "parseAtemToolCalls: parallel invokes each parse; dropped wrapper still parses" {
    const allocator = testing.allocator;
    const text = "<atem:invoke name=\"a.one\">\n<atem:parameter name=\"x\">1</atem:parameter>\n</atem:invoke>\n<atem:invoke name=\"a.two\">\n<atem:parameter name=\"y\">two</atem:parameter>\n</atem:invoke>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("a.one", calls[0].name);
    try testing.expectEqualStrings("a.two", calls[1].name);
}

test "parseToolCalls: a bare <invoke> in prose or inside another dialect's value is not a call" {
    const allocator = testing.allocator;
    // The opener alone is prose, as the scrubber reads it.
    try testing.expect((try parseToolCalls(allocator, "Claude's format wraps each call in <invoke name=\"bash\"> tags.")) == null);
    // A Hermes parameter VALUE carrying invoke markup never decides the call.
    const text = "<tool_call>\n<function=write>\n<parameter=content>\nrun <invoke name=\"bash\"><parameter name=\"command\">ls</parameter></invoke>\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    try testing.expect(!calls[0].inferred);
}

// ── Collapsed <tool call><parameter name="TOOL"> recovery (DeepSeek-V4.1) ──
/// BARE_INVOKE with the `<invoke name=…>` layer dropped: a `<tool call>`/
/// `<tool_call>` wrapper whose first `<parameter name="X">` carries the TOOL NAME
/// (not a parameter) with the sole argument as its value. Unlike the other invoke
/// arms this is schema-aware (`.inferred`) — the arg key lives in the schema, not
/// the markup — so the request chokepoint calls it after every dialect returned
/// null, gated on the wrapper plus the parameter naming a declared tool (a
/// declared-tool name in a tool-call wrapper is the tell — a real parameter is
/// never the tool's own name).
pub fn inferCollapsedParamToolCalls(allocator: std.mem.Allocator, text: []const u8, tools_json: []const u8) !?[]ParsedToolCall {
    const w_space = std.mem.indexOf(u8, text, "<tool call");
    const w_under = std.mem.indexOf(u8, text, "<tool_call");
    const wrapper_pos = blk: {
        if (w_space) |a| break :blk if (w_under) |b| @min(a, b) else a;
        break :blk w_under orelse return null;
    };

    const tools_parsed = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return null;
    defer tools_parsed.deinit();
    if (tools_parsed.value != .array) return null;

    var pos: usize = wrapper_pos;
    while (std.mem.indexOfPos(u8, text, pos, "<parameter")) |p| {
        const after = p + "<parameter".len;
        if (after >= text.len) break;
        // Must be the attribute form `<parameter name=…>`, not `<parameters>`.
        if (!std.ascii.isWhitespace(text[after])) {
            pos = after;
            continue;
        }
        const seg = text[after..];
        const te = std.mem.indexOfScalar(u8, seg, '>') orelse break; // truncated open tag
        const pname = dsmlAttr(seg[0 .. te + 1], "name") orelse {
            pos = after + te + 1;
            continue;
        };
        // X must name a declared tool; capture its `parameters` for mapping.
        const params = findToolParameters(tools_parsed.value.array, pname) orelse {
            pos = after + te + 1;
            continue;
        };

        // Value runs to the confirmed `</parameter>` close; absent = truncated.
        const vstart = after + te + 1;
        var value: ?[]const u8 = null;
        if (std.mem.indexOf(u8, text[vstart..], "</parameter>")) |vend| {
            const raw = std.mem.trim(u8, text[vstart .. vstart + vend], " \n\t\r");
            if (raw.len > 0) value = raw;
        }

        const arguments = try buildCollapsedArgs(allocator, params, value);
        errdefer allocator.free(arguments);
        const name_owned = try allocator.dupe(u8, pname);
        errdefer allocator.free(name_owned);
        const calls = try allocator.alloc(ParsedToolCall, 1);
        // A heuristic recovery, like BARE_INVOKE: mark it inferred so the
        // chokepoint never treats a guessed call as an explicit one.
        calls[0] = .{ .name = name_owned, .arguments = arguments, .inferred = true };
        log.info("  [tool-parse] recovered collapsed tool-call markup into '{s}'\n", .{pname});
        return calls;
    }
    return null;
}

/// The `parameters` object of a declared tool, by function name (null if the
/// name is not a declared tool). A tool with no `parameters` key yields a null
/// Value so an argless collapsed call (`get_goal`) still resolves to NAME + {}.
fn findToolParameters(tools: std.json.Array, name: []const u8) ?std.json.Value {
    for (tools.items) |tool_val| {
        if (tool_val != .object) continue;
        const func_val = tool_val.object.get("function") orelse continue;
        if (func_val != .object) continue;
        const name_val = func_val.object.get("name") orelse continue;
        if (name_val != .string) continue;
        if (!std.mem.eql(u8, name_val.string, name)) continue;
        return func_val.object.get("parameters") orelse std.json.Value{ .null = {} };
    }
    return null;
}

/// The one schema parameter a collapsed value can safely fill: the sole
/// property, else the sole required param. null when the schema names no single
/// target — argless, or two-plus params with no single required one — so the
/// caller ships NAME + {} rather than guess a field (a wrong guess would write a
/// real argument into the wrong slot, e.g. a path into `content`).
fn pickMainParam(params: std.json.Value) ?[]const u8 {
    if (params != .object) return null;
    const props_v = params.object.get("properties") orelse return null;
    if (props_v != .object) return null;
    const props = props_v.object;
    if (props.count() == 1) return props.keys()[0];
    if (params.object.get("required")) |req_v| {
        if (req_v == .array and req_v.array.items.len == 1 and req_v.array.items[0] == .string) {
            const only = req_v.array.items[0].string;
            if (props.get(only) != null) return only;
        }
    }
    return null;
}

/// Build the arguments JSON for a collapsed call. The value ships as a RAW
/// STRING under the mapped parameter; types come from the schema at the coercion
/// chokepoint, never from the value's spelling (a command reading `true` or
/// `[1,2]` is a string, not a bool/array). No value or no single target → `{}`.
fn buildCollapsedArgs(allocator: std.mem.Allocator, params: std.json.Value, value: ?[]const u8) ![]const u8 {
    const v = value orelse return try allocator.dupe(u8, "{}");
    const key = pickMainParam(params) orelse return try allocator.dupe(u8, "{}");
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var map: std.json.ObjectMap = .empty;
    try map.put(aa, key, std.json.Value{ .string = v });
    return try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = map }, .{});
}

test "inferCollapsedParamToolCalls: complete <tool call><parameter name=bash> maps value to the command arg" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
        \\ {"type":"function","function":{"name":"get_goal","parameters":{"type":"object","properties":{}}}}]
    ;
    const text = "查 guest。\n\n</br>\n\n<tool call>\n<parameter name=\"bash\">cd /x && echo hi</parameter>\n</parameter>\n</invoke>";
    const calls = (try inferCollapsedParamToolCalls(allocator, text, tools)).?;
    defer {
        for (calls) |c| {
            allocator.free(c.name);
            allocator.free(c.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("cd /x && echo hi", parsed.value.object.get("command").?.string);
}

test "inferCollapsedParamToolCalls: truncated after the open tag ships NAME + {}" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
    ;
    const text = "x\n\n<tool call>\n<parameter name=\"bash\">";
    const calls = (try inferCollapsedParamToolCalls(allocator, text, tools)).?;
    defer {
        for (calls) |c| {
            allocator.free(c.name);
            allocator.free(c.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
}

test "inferCollapsedParamToolCalls: argless tool resolves to NAME + {}" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"get_goal","parameters":{"type":"object","properties":{}}}}]
    ;
    const text = "status.\n\n<tool call>\n<parameter name=\"get_goal\"></parameter>\n</invoke>";
    const calls = (try inferCollapsedParamToolCalls(allocator, text, tools)).?;
    defer {
        for (calls) |c| {
            allocator.free(c.name);
            allocator.free(c.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_goal", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
}

test "inferCollapsedParamToolCalls: sole required param wins over extra optionals" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query"]}}}]
    ;
    const text = "<tool call>\n<parameter name=\"search\">lions of massachusetts</parameter>";
    const calls = (try inferCollapsedParamToolCalls(allocator, text, tools)).?;
    defer {
        for (calls) |c| {
            allocator.free(c.name);
            allocator.free(c.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqualStrings("search", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("lions of massachusetts", parsed.value.object.get("query").?.string);
}

test "inferCollapsedParamToolCalls: a parameter NOT named after a tool is never promoted" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
    ;
    // `command` is a real param name, not a tool name — no tool is identifiable.
    const text = "<tool call>\n<parameter name=\"command\">ls -la</parameter>";
    try testing.expect((try inferCollapsedParamToolCalls(allocator, text, tools)) == null);
}

test "inferCollapsedParamToolCalls: no <tool call> wrapper is prose, never a call" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
    ;
    const text = "Claude wraps params as <parameter name=\"bash\"> in its docs.";
    try testing.expect((try inferCollapsedParamToolCalls(allocator, text, tools)) == null);
}

test "inferCollapsedParamToolCalls: a JSON-looking value stays a string (schema types it, not the spelling)" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
    ;
    // The command happens to read as a JSON literal; it must NOT ship as a bool.
    const text = "<tool call>\n<parameter name=\"bash\">true</parameter>";
    const calls = (try inferCollapsedParamToolCalls(allocator, text, tools)).?;
    defer {
        for (calls) |c| {
            allocator.free(c.name);
            allocator.free(c.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("command").? == .string);
    try testing.expectEqualStrings("true", parsed.value.object.get("command").?.string);
}

test "inferCollapsedParamToolCalls: two required params name no single target -> NAME + {}" {
    const allocator = testing.allocator;
    // write_file(path, content), both required: a value could be either, so the
    // schema names no single target and we must NOT guess a slot.
    const tools =
        \\[{"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
    ;
    const text = "<tool call>\n<parameter name=\"write_file\">/tmp/x</parameter>";
    const calls = (try inferCollapsedParamToolCalls(allocator, text, tools)).?;
    defer {
        for (calls) |c| {
            allocator.free(c.name);
            allocator.free(c.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqualStrings("write_file", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
}
