//! The DSML tool-call dialect (DeepSeek V4).

const std = @import("std");
const testing = std.testing;

const parseToolCalls = @import("parse.zig").parseToolCalls;
const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const appendJsonString = @import("../../../sdk/util/json_text.zig").appendJsonString;

/// Inkling tool calls: `NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>`
/// per call (one message each; parallel calls are consecutive messages). The
/// payload's "name" is authoritative (the prefix NAME is the same string in
/// well-formed output, and is the fallback when the payload is truncated).
/// Truncation salvage per the hard rule: recover NAME + `{}` — never partial
/// argument values.
// DeepSeek-V4 native DSML markers. `｜` is the FULLWIDTH vertical bar
// (U+FF5C, 3 bytes) — the dsml_token "｜DSML｜" appears in no other family.
// V4.1 puts a space before every tag name (`<｜DSML｜ invoke`).
const DsmlTag = struct { open: []const u8, name: []const u8 };
const DSML_INVOKE_TAG: DsmlTag = .{ .open = "<｜DSML｜", .name = "invoke" };
const DSML_INVOKE_CLOSE: DsmlTag = .{ .open = "</｜DSML｜", .name = "invoke>" };
const DSML_PARAM_TAG: DsmlTag = .{ .open = "<｜DSML｜", .name = "parameter" };
const DSML_PARAM_CLOSE: DsmlTag = .{ .open = "</｜DSML｜", .name = "parameter>" };

const DsmlHit = struct { at: usize, end: usize };

fn dsmlFind(text: []const u8, from: usize, comptime tag: DsmlTag) ?DsmlHit {
    var p = from;
    while (std.mem.indexOfPos(u8, text, p, tag.open)) |at| {
        var q = at + tag.open.len;
        if (q < text.len and text[q] == ' ') q += 1;
        if (std.mem.startsWith(u8, text[q..], tag.name)) return .{ .at = at, .end = q + tag.name.len };
        p = at + 1;
    }
    return null;
}

/// Tolerant quoted-attribute extraction from a tag-attribute segment:
/// `name="X"` / `name='X'` / mangled unquoted `name=X` (to whitespace/`>`).
pub fn dsmlAttr(seg: []const u8, comptime key: []const u8) ?[]const u8 {
    const pat = key ++ "=";
    var start: usize = 0;
    while (std.mem.indexOfPos(u8, seg, start, pat)) |at| {
        if (at > 0 and (std.ascii.isAlphanumeric(seg[at - 1]) or seg[at - 1] == '_')) {
            start = at + pat.len;
            continue;
        }
        var v = at + pat.len;
        if (v >= seg.len) return null;
        const q = seg[v];
        if (q == '"' or q == '\'') {
            v += 1;
            const end = std.mem.indexOfScalarPos(u8, seg, v, q) orelse seg.len;
            return seg[v..end];
        }
        const end = std.mem.indexOfAnyPos(u8, seg, v, " \t\n>") orelse seg.len;
        return seg[v..end];
    }
    return null;
}

/// DeepSeek-V4 DSML tool calls:
///   `<｜DSML｜tool_calls>` wrapping
///   `<｜DSML｜invoke name="X">` +
///   `<｜DSML｜parameter name="k" string="true|false">v</｜DSML｜parameter>`.
/// string="true" values are raw text (JSON-escaped here); string="false"
/// values are already JSON (validated; invalid degrades to an honest string).
/// House conventions: the parser keys on the INVOKE marker (a dropped
/// wrapper or invoke-close still parses — delimiter-drop class), values run
/// to the CONFIRMED closing tag (rich values with `<`/newlines survive),
/// truncation salvages NAME + completed pairs (fragments dropped), keys are
/// escaped + deduped (last wins) so emitted arguments are ALWAYS valid JSON.
pub fn parseDsmlToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    const Param = struct { key: []const u8, value: []const u8, is_string: bool };
    var pos: usize = 0;
    while (dsmlFind(text, pos, DSML_INVOKE_TAG)) |inv| {
        var p = inv.end;
        const tag_end = std.mem.indexOfScalarPos(u8, text, p, '>') orelse break;
        const name = dsmlAttr(text[p..tag_end], "name") orelse {
            pos = tag_end + 1;
            continue;
        };
        p = tag_end + 1;

        var params = std.ArrayList(Param).empty;
        defer params.deinit(allocator);
        while (true) {
            const next_param = dsmlFind(text, p, DSML_PARAM_TAG);
            const next_close = dsmlFind(text, p, DSML_INVOKE_CLOSE);
            const next_invoke = dsmlFind(text, p, DSML_INVOKE_TAG);
            const np = if (next_param) |h| h.at else std.math.maxInt(usize);
            const nc = if (next_close) |h| h.at else std.math.maxInt(usize);
            const ni = if (next_invoke) |h| h.at else std.math.maxInt(usize);
            if (np == std.math.maxInt(usize) and nc == std.math.maxInt(usize) and ni == std.math.maxInt(usize)) {
                // truncated invoke: ship NAME + completed pairs
                p = text.len;
                break;
            }
            if (nc <= np and nc <= ni) {
                p = next_close.?.end;
                break;
            }
            if (ni < np) {
                // dropped invoke-close — the next call starts here; the outer
                // loop picks it up from `p`.
                p = ni;
                break;
            }
            // parameter
            var q = next_param.?.end;
            const ptag_end = std.mem.indexOfScalarPos(u8, text, q, '>') orelse {
                p = text.len;
                break;
            };
            const attrs = text[q..ptag_end];
            const pkey = dsmlAttr(attrs, "name");
            const pstring = dsmlAttr(attrs, "string");
            const vstart = ptag_end + 1;
            const vclose = dsmlFind(text, vstart, DSML_PARAM_CLOSE) orelse {
                // unterminated value = server-cut fragment; drop it, end scan
                p = text.len;
                break;
            };
            q = vclose.end;
            if (pkey) |k| {
                const value = text[vstart..vclose.at];
                // string attr absent (mangled opener): treat JSON-parseable
                // values as JSON, everything else as a string.
                const is_string = if (pstring) |sf|
                    !std.mem.eql(u8, sf, "false")
                else
                    !dsmlValueIsJson(allocator, value);
                // dedup: last wins
                var replaced = false;
                for (params.items) |*prm| {
                    if (std.mem.eql(u8, prm.key, k)) {
                        prm.value = value;
                        prm.is_string = is_string;
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) try params.append(allocator, .{ .key = k, .value = value, .is_string = is_string });
            }
            p = q;
        }

        // Build the arguments object: escaped keys, escaped string values,
        // raw (validated) JSON values.
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(allocator);
        try out.append(allocator, '{');
        for (params.items, 0..) |prm, i| {
            if (i > 0) try out.append(allocator, ',');
            try appendJsonString(allocator, &out, prm.key);
            try out.append(allocator, ':');
            if (!prm.is_string and dsmlValueIsJson(allocator, prm.value)) {
                try out.appendSlice(allocator, std.mem.trim(u8, prm.value, " \t\n\r"));
            } else {
                try appendJsonString(allocator, &out, prm.value);
            }
        }
        try out.append(allocator, '}');
        const name_owned = try allocator.dupe(u8, name);
        errdefer allocator.free(name_owned);
        try calls.append(allocator, .{ .name = name_owned, .arguments = try out.toOwnedSlice(allocator) });
        pos = p;
    }
}

fn dsmlValueIsJson(allocator: std.mem.Allocator, value: []const u8) bool {
    const trimmed = std.mem.trim(u8, value, " \t\n\r");
    if (trimmed.len == 0) return false;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{}) catch return false;
    parsed.deinit();
    return true;
}

// ── tests ──

pub fn freeParsedCalls(calls: []ParsedToolCall) void {
    for (calls) |tc| {
        testing.allocator.free(tc.name);
        testing.allocator.free(tc.arguments);
    }
    testing.allocator.free(calls);
}

test "parseToolCalls dsml: canonical two-call block, string and JSON params" {
    const raw = "I'll check the weather.\n\n<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"get_weather\">\n" ++
        "<｜DSML｜parameter name=\"city\" string=\"true\">San Francisco</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n" ++
        "<｜DSML｜invoke name=\"get_time\">\n" ++
        "<｜DSML｜parameter name=\"tz\" string=\"true\">PST</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n" ++
        "</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    try testing.expectEqualStrings("{\"city\":\"San Francisco\",\"days\":3}", calls[0].arguments);
    try testing.expectEqualStrings("get_time", calls[1].name);
    try testing.expectEqualStrings("{\"tz\":\"PST\"}", calls[1].arguments);
    try testing.expect(!calls[0].inferred);
}

test "parseToolCalls dsml: V4.1's spaced tag names parse like V4's" {
    const raw = "On it.\n\n<｜DSML｜ calls>\n" ++
        "<｜DSML｜ invoke name=\"get_weather\">\n" ++
        "<｜DSML｜ parameter name=\"city\" string=\"true\">San Francisco</｜DSML｜ parameter>\n" ++
        "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n" ++
        "</｜DSML｜ invoke>\n" ++
        "<｜DSML｜ invoke name=\"get_time\">\n" ++
        "</｜DSML｜ invoke>\n" ++
        "</｜DSML｜ calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    try testing.expectEqualStrings("{\"city\":\"San Francisco\",\"days\":3}", calls[0].arguments);
    try testing.expectEqualStrings("get_time", calls[1].name);
    try testing.expectEqualStrings("{}", calls[1].arguments);
}

test "parseToolCalls dsml: dropped invoke close before the next call still parses both" {
    // Delimiter-drop tolerance (hy3 class): the model omits </｜DSML｜invoke>
    // between back-to-back calls.
    const raw = "<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"first\">\n" ++
        "<｜DSML｜parameter name=\"a\" string=\"true\">x</｜DSML｜parameter>\n" ++
        "<｜DSML｜invoke name=\"second\">\n" ++
        "<｜DSML｜parameter name=\"b\" string=\"false\">true</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("first", calls[0].name);
    try testing.expectEqualStrings("{\"a\":\"x\"}", calls[0].arguments);
    try testing.expectEqualStrings("second", calls[1].name);
    try testing.expectEqualStrings("{\"b\":true}", calls[1].arguments);
}

test "parseToolCalls dsml: truncation salvages NAME + completed pairs, never fragments" {
    // Server-side cut mid-value: the unterminated pair is dropped (a half-
    // written file is worse than a re-issued write), completed pairs survive.
    const raw = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n" ++
        "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/a.txt</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"content\" string=\"true\">first half of a long fi";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    try testing.expectEqualStrings("{\"path\":\"/tmp/a.txt\"}", calls[0].arguments);
}

test "parseToolCalls dsml: rich string value runs to the confirmed close" {
    // Newlines, quotes, braces, `<` inside a string value — value ends ONLY
    // at the closing parameter tag (the Gemma dropped-quote class).
    const raw = "<｜DSML｜invoke name=\"bash\">\n" ++
        "<｜DSML｜parameter name=\"cmd\" string=\"true\">grep -r \"a < b\" src/\necho {done}</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("grep -r \"a < b\" src/\necho {done}", parsed.value.object.get("cmd").?.string);
}

test "parseToolCalls dsml: duplicate keys dedup (last wins) and invalid JSON value degrades to string" {
    const raw = "<｜DSML｜invoke name=\"t\">\n" ++
        "<｜DSML｜parameter name=\"k\" string=\"true\">old</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"k\" string=\"true\">new</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"n\" string=\"false\">not[json</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("new", parsed.value.object.get("k").?.string);
    try testing.expectEqualStrings("not[json", parsed.value.object.get("n").?.string);
    try testing.expectEqual(@as(usize, 2), parsed.value.object.count());
}

test "parseToolCalls dsml: zero-arg call and thinking prefix" {
    const raw = "thinking about it</think>Sure.\n\n<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"list_files\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("list_files", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
}
