//! Bare JSON call inference, self-closing tool tags, and truncated JSON call names.

const std = @import("std");
const log = @import("../../../sdk/util/log.zig");
const testing = std.testing;

const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const dsmlAttr = @import("dsml.zig").dsmlAttr;

/// Last-resort tool-call inference for models that emit JUST the arguments
/// object — no tool name, no wrapper syntax. Observed live from Gemma 4 12B
/// via Claude Code: ```` ```json\n{"file_path": …, "content": …}\n``` ````
/// with the Write tool defined; parseToolCalls finds nothing (no "name" key)
/// and the un-executed JSON leaked into the visible text.
///
/// Conservative contract — fires ONLY when parseToolCalls returned null:
///   • the visible content (after a leading think block) must START with the
///     JSON object, optionally wrapped in a markdown fence;
///   • the object's keys must satisfy exactly ONE tool in `tools_json`
///     (OpenAI shape): required ⊆ keys AND keys ⊆ properties;
///   • zero or multiple matching tools → null (ambiguity never guesses).
pub fn inferBareJsonToolCalls(allocator: std.mem.Allocator, text: []const u8, tools_json: []const u8) !?[]ParsedToolCall {
    // Strip a leading think block, mirroring parseToolCalls.
    var effective_text = text;
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        effective_text = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
    } else if (std.mem.indexOf(u8, text, "</think>")) |think_end| {
        effective_text = std.mem.trimStart(u8, text[think_end + 8 ..], "\n ");
    }
    var trimmed = std.mem.trimStart(u8, effective_text, "\n \t");
    // Skip one opening markdown fence line (``` or ```json etc.).
    if (std.mem.startsWith(u8, trimmed, "```")) {
        const nl = std.mem.indexOfScalar(u8, trimmed, '\n') orelse return null;
        trimmed = std.mem.trimStart(u8, trimmed[nl + 1 ..], "\n \t");
    }
    // The object must LEAD the visible content — a JSON example mid-prose is
    // never promoted to a call.
    if (!std.mem.startsWith(u8, trimmed, "{")) return null;
    const obj_slice = balancedJsonObject(trimmed) orelse return null;

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, obj_slice, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const obj = parsed.value.object;
    if (obj.count() == 0) return null;

    const tools_parsed = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return null;
    defer tools_parsed.deinit();
    if (tools_parsed.value != .array) return null;

    var match_name: ?[]const u8 = null;
    for (tools_parsed.value.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func_val = tool_val.object.get("function") orelse continue;
        if (func_val != .object) continue;
        const name_val = func_val.object.get("name") orelse continue;
        if (name_val != .string) continue;
        const params_val = func_val.object.get("parameters") orelse continue;
        if (params_val != .object) continue;
        const props_val = params_val.object.get("properties") orelse continue;
        if (props_val != .object) continue;
        const props = props_val.object;
        if (props.count() == 0) continue;

        // keys ⊆ properties
        var keys_ok = true;
        var it = obj.iterator();
        while (it.next()) |kv| {
            if (props.get(kv.key_ptr.*) == null) {
                keys_ok = false;
                break;
            }
        }
        if (!keys_ok) continue;

        // required ⊆ keys
        if (params_val.object.get("required")) |req_val| {
            if (req_val == .array) {
                var req_ok = true;
                for (req_val.array.items) |r| {
                    if (r != .string) continue;
                    if (obj.get(r.string) == null) {
                        req_ok = false;
                        break;
                    }
                }
                if (!req_ok) continue;
            }
        }

        if (match_name != null) return null; // ambiguous — never guess
        match_name = name_val.string;
    }

    const name = match_name orelse return null;
    const calls = try allocator.alloc(ParsedToolCall, 1);
    errdefer allocator.free(calls);
    const name_owned = try allocator.dupe(u8, name);
    errdefer allocator.free(name_owned);
    calls[0] = .{ .name = name_owned, .arguments = try allocator.dupe(u8, obj_slice) };
    log.info("  [tool-parse] inferred bare-args call to '{s}' via unique schema match\n", .{name});
    return calls;
}

/// Last-resort inference for DeepSeek-V4.1's degraded tool-call markup (the
/// #928 Anthropic-markup family, DeepSeek-V4.1 signature; same class as #980).
/// Under prefill mode-lock the model drops the `<function=NAME>` layer and
/// collapses a call to the hybrid
///
///     <tool call>
///     <parameter name="bash">cd /x && …</parameter>
///
/// — a Hermes-ish `<tool call>`/`<tool_call>` wrapper (note the SPACE variant),
/// no `<invoke`/`<function=` middle layer, and the TOOL NAME carried as the
/// first `<parameter name="…">` (Anthropic spelling) whose value is the tool's
/// sole argument; often truncated right after the open tag. `parseHermesToolCall`
/// wants `<function=>` and `parseInvokeToolCalls` wants an `<invoke` opener, so
/// neither matches and the block leaks into `content` with finish "stop".
///
/// Conservative contract — fires ONLY when parseToolCalls returned null:
///   • text must carry the `<tool call>`/`<tool_call>` wrapper, and params are
///     scanned only AFTER it (a lone `<parameter>` in prose is out of scope);
///   • the first `<parameter name="X">` whose X names a DECLARED tool wins — a
///     parameter named after a tool, inside a tool-call wrapper, is the tell (a
///     real param is `command`/`path`/…, never the tool's own name); none → null;
///   • the value (to `</parameter>`, else truncated) maps to that tool's sole/
///     main parameter via the schema; no value or no mappable param → NAME + {}
///     (ship the call, never a fragment — the ATEM/Harmony truncation rule).
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
        calls[0] = .{ .name = name_owned, .arguments = arguments };
        log.info("  [tool-parse] recovered collapsed <tool call><parameter name=\"{s}\"> call\n", .{pname});
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

/// Pick the schema parameter a collapsed value should fill: the sole property,
/// else the sole required param, else a conventionally-named one. null = no
/// mappable parameter (argless tool, or ambiguous multi-param).
fn pickMainParam(params: std.json.Value) ?[]const u8 {
    if (params != .object) return null;
    const props_v = params.object.get("properties") orelse return null;
    if (props_v != .object) return null;
    const props = props_v.object;
    if (props.count() == 0) return null;
    if (props.count() == 1) return props.keys()[0];
    if (params.object.get("required")) |req_v| {
        if (req_v == .array and req_v.array.items.len == 1 and req_v.array.items[0] == .string)
            return req_v.array.items[0].string;
    }
    const conventional = [_][]const u8{ "command", "cmd", "input", "text", "content", "code", "script", "query", "path" };
    for (conventional) |c| if (props.get(c) != null) return c;
    return null;
}

/// Build the arguments JSON for a collapsed call. A present value maps to the
/// main parameter (spelled JSON keeps its type, else a raw string — the schema
/// coercion chokepoint still gets the final word); no value or no mappable
/// parameter ships `{}` so the call fires without a fabricated fragment.
fn buildCollapsedArgs(allocator: std.mem.Allocator, params: std.json.Value, value: ?[]const u8) ![]const u8 {
    const v = value orelse return try allocator.dupe(u8, "{}");
    const key = pickMainParam(params) orelse return try allocator.dupe(u8, "{}");
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var map: std.json.ObjectMap = .empty;
    const jv: std.json.Value = std.json.parseFromSliceLeaky(std.json.Value, aa, v, .{}) catch std.json.Value{ .string = v };
    try map.put(aa, key, jv);
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

/// Strip a leading `<parameters>` / trailing `</parameters>` pair if both
/// are present. DSV4 occasionally wraps the args object in this XML
/// container — leaving it in place would defeat the JSON parser. Trims
/// surrounding whitespace.
pub fn stripParametersWrapper(content: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const open = "<parameters>";
    const close = "</parameters>";
    if (trimmed.len < open.len + close.len) return trimmed;
    if (!std.mem.startsWith(u8, trimmed, open)) return trimmed;
    if (!std.mem.endsWith(u8, trimmed, close)) return trimmed;
    const inner = trimmed[open.len .. trimmed.len - close.len];
    return std.mem.trim(u8, inner, " \t\n\r");
}

/// Return the substring of `content` covering the first brace-balanced JSON
/// object, or null if no balanced object is present. Handles quoted strings
/// (so `}` inside a string doesn't close the object) and escape sequences.
/// Used to tolerate the DSV4 `{"…"}}` extra-trailing-`}` artifact.
/// Try to parse a tool tag whose body lives entirely in XML-ish
/// attributes. Two terminator shapes are accepted:
///
///   Self-closing /> :
///     <tool name="shell" arguments="{...}"/>
///     <tool name="X" arguments='{...}' />
///     <tool name="X" arguments="{...}'/>             (broken quote — DSV4)
///
///   Empty-body explicit close ></tool*> :
///     <tool_call name="cwd" arguments="{...}"></tool_call>
///     <tool name="X" arguments="{...}"></tool_calls>  (mismatched close OK)
///
/// `slice` MUST start with `<tool` (the caller has already validated the
/// terminator char after that prefix). Returns name + arguments slices into
/// `slice` plus the number of bytes consumed (up to and including the final
/// `>` or `</tool*>` close), or null if this isn't a shape we can parse.
///
/// We never trust the XML quoting around `arguments=` — the model frequently
/// uses `"` to open and `'` to close, or doesn't escape inner `"` at all.
/// Instead we expect a `{` immediately (optionally after a single opening
/// quote) and snap a balanced JSON object via `balancedJsonObject`. That
/// strategy is what makes the DSV4 fragment parse cleanly even though the
/// surrounding XML is malformed.
pub fn parseSelfClosingToolTag(slice: []const u8) ?struct { name: []const u8, arguments: []const u8, consumed: usize } {
    if (slice.len < "<tool".len) return null;
    if (!std.mem.startsWith(u8, slice, "<tool")) return null;
    // Bound the scan window so a runaway tag can't pin us forever.
    const limit = @min(slice.len, 8192);

    // 1. Locate `name="X"` (or `'X'`) attribute. We don't enforce ordering
    //    against `arguments=` — DSV4 sometimes interleaves them.
    var name: ?[]const u8 = null;
    var name_end: usize = "<tool".len;
    {
        var i: usize = "<tool".len;
        while (i + "name=".len < limit) : (i += 1) {
            if (slice[i] != 'n') continue;
            if (!std.mem.eql(u8, slice[i .. i + "name=".len], "name=")) continue;
            const q_pos = i + "name=".len;
            if (q_pos >= limit) break;
            const quote = slice[q_pos];
            if (quote != '"' and quote != '\'') continue;
            const val_start = q_pos + 1;
            const val_end_rel = std.mem.indexOfScalar(u8, slice[val_start..limit], quote) orelse break;
            name = slice[val_start .. val_start + val_end_rel];
            name_end = val_start + val_end_rel + 1;
            break;
        }
    }

    // 2. Locate `arguments=` and snap the next balanced JSON object as
    //    the value. We deliberately don't parse the XML quote — see
    //    function docs.
    var args: ?[]const u8 = null;
    var args_end: usize = 0;
    {
        var i: usize = "<tool".len;
        while (i + "arguments=".len < limit) : (i += 1) {
            if (slice[i] != 'a') continue;
            if (!std.mem.eql(u8, slice[i .. i + "arguments=".len], "arguments=")) continue;
            var j: usize = i + "arguments=".len;
            // Skip an opening quote if present.
            if (j < limit and (slice[j] == '"' or slice[j] == '\'')) j += 1;
            // Skip whitespace between the quote and the JSON object.
            while (j < limit and (slice[j] == ' ' or slice[j] == '\t')) j += 1;
            if (j >= limit or slice[j] != '{') break;
            const sub = slice[j..limit];
            const json = balancedJsonObject(sub) orelse break;
            args = json;
            const json_off = @intFromPtr(json.ptr) - @intFromPtr(sub.ptr);
            args_end = j + json_off + json.len;
            break;
        }
    }

    if (name == null or args == null) return null;

    // 3. Scan forward from the end of whichever attribute landed last to
    //    the terminator. Tolerate a stray closing quote (model often closes
    //    args with `'` even though the opener was `"`) and any intervening
    //    whitespace. We accept either:
    //      a) `/>` — self-closing.
    //      b) `>` followed (possibly across whitespace) by `</tool*>` —
    //         empty body, explicit close marker.
    var k: usize = @max(name_end, args_end);
    while (k < limit and (slice[k] == ' ' or slice[k] == '\t' or slice[k] == '\n' or slice[k] == '\r' or slice[k] == '"' or slice[k] == '\'')) k += 1;
    if (k >= limit) return null;

    if (slice[k] == '/') {
        k += 1;
        if (k >= limit or slice[k] != '>') return null;
        return .{ .name = name.?, .arguments = args.?, .consumed = k + 1 };
    }
    if (slice[k] == '>') {
        var w: usize = k + 1;
        while (w < slice.len and (slice[w] == ' ' or slice[w] == '\t' or slice[w] == '\n' or slice[w] == '\r')) w += 1;
        const close_markers = [_][]const u8{
            "</tool_call>",   "</tool_calls>",
            "</tool_request>", "</tool_requests>",
            "</tool>",
        };
        for (close_markers) |m| {
            if (w + m.len <= slice.len and std.mem.eql(u8, slice[w .. w + m.len], m)) {
                return .{ .name = name.?, .arguments = args.?, .consumed = w + m.len };
            }
        }
    }
    return null;
}

/// The value of a depth-1 `"name"` key in an UNBALANCED JSON object (a tool
/// call truncated mid-body). String-aware, so a `"name"` inside a nested
/// value or a string never counts. Null when the text is not an object or
/// carries no complete top-level name.
pub fn truncatedJsonCallName(content: []const u8) ?[]const u8 {
    const t = std.mem.trimStart(u8, content, " \t\n\r");
    if (!std.mem.startsWith(u8, t, "{")) return null;
    var depth: usize = 0;
    var i: usize = 0;
    var last_str: ?[]const u8 = null; // most recent complete string at depth 1
    var after_name_colon = false;
    while (i < t.len) : (i += 1) {
        const c = t[i];
        if (c == '"') {
            const start = i + 1;
            var j = start;
            while (j < t.len and t[j] != '"') : (j += 1) {
                if (t[j] == '\\') j += 1;
            }
            if (j >= t.len) return null; // unterminated string = the cut
            const str = t[start..j];
            if (depth == 1) {
                if (after_name_colon) return if (str.len > 0) str else null;
                last_str = str;
            }
            i = j;
            continue;
        }
        switch (c) {
            '{', '[' => {
                depth += 1;
                after_name_colon = false;
            },
            '}', ']' => {
                if (depth == 0) return null;
                depth -= 1;
            },
            ':' => if (depth == 1) {
                if (last_str) |k| after_name_colon = std.mem.eql(u8, k, "name");
            },
            ',' => {
                after_name_colon = false;
                if (depth == 1) last_str = null;
            },
            ' ', '\t', '\n', '\r' => {},
            else => after_name_colon = false,
        }
    }
    return null;
}

pub fn balancedJsonObject(content: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const start = std.mem.indexOfScalar(u8, trimmed, '{') orelse return null;
    var depth: i32 = 0;
    var in_string: bool = false;
    var escape: bool = false;
    var i: usize = start;
    while (i < trimmed.len) : (i += 1) {
        const c = trimmed[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            depth += 1;
        } else if (c == '}') {
            depth -= 1;
            if (depth == 0) return trimmed[start .. i + 1];
            if (depth < 0) return null; // mismatched — give up
        }
    }
    return null;
}

// ── tests ──

const test_tools_write_bash =
    \\[{"type":"function","function":{"name":"Write","description":"Write a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}},
    \\ {"type":"function","function":{"name":"Bash","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
;

test "inferBareJsonToolCalls maps bare-args fenced JSON to the unique matching tool (Claude Code capture)" {
    const allocator = testing.allocator;
    // Shape captured live from gemma-4-12b via Claude Code /v1/messages: thought
    // block, then a ```json fence holding ONLY the Write tool's arguments.
    const text = "<|channel>thought\nI will create this file using Write.<channel|>```json\n{\n  \"file_path\": \"/Users/david/mlx_info.html\",\n  \"content\": \"<h1>MLX</h1>\"\n}\n```\nI've created the file.";
    const calls = (try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("Write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/Users/david/mlx_info.html", parsed.value.object.get("file_path").?.string);
}

test "inferBareJsonToolCalls unfenced bare object at content start" {
    const allocator = testing.allocator;
    const text = "{\"command\": \"ls -la\"}";
    const calls = (try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqualStrings("Bash", calls[0].name);
}

test "inferBareJsonToolCalls refuses ambiguous and non-matching objects" {
    const allocator = testing.allocator;
    // Two tools share the same key → ambiguous → null.
    const dup_tools =
        \\[{"type":"function","function":{"name":"a","parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}},
        \\ {"type":"function","function":{"name":"b","parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}}]
    ;
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"x\": \"1\"}", dup_tools)) == null);
    // Keys not in any tool's properties → null.
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"zzz\": 1}", test_tools_write_bash)) == null);
    // Missing a required key → null.
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"file_path\": \"a.txt\"}", test_tools_write_bash)) == null);
}

test "inferBareJsonToolCalls ignores JSON that does not lead the content" {
    const allocator = testing.allocator;
    // Example object mid-prose must never be promoted to a call.
    const text = "Here is an example of the payload you could send: {\"command\": \"ls\"} — adjust as needed.";
    try testing.expect((try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) == null);
}
