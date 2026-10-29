//! The XML element tool-call dialect (`<tool_NAME>` / element-per-argument bodies).

const std = @import("std");

const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const balancedJsonObject = @import("json_infer.zig").balancedJsonObject;

/// DSV4-Flash XML-element tool form: the body of a `<tool_calls>`-style
/// wrapper carries the tool name and each argument as plain child elements —
/// no JSON anywhere (captured live):
///     <tool_name>shell</tool_name>
///     <command>df -h / | grep -v "Filesystem"</command>
/// The name comes from a <tool_name>, <name>, or <tool> element; every other
/// simple `<key>text</key>` pair becomes a string argument. Strict on shape:
/// non-whitespace between elements, an unclosed element, a duplicate name
/// element, or a missing name returns null so prose-ish markup can't
/// half-execute.
/// Tag suffixes after `tool_` that mark tool RESULTS or metadata, never tool
/// names. `<tool_output>…</tool_output>` is DSV4 hallucinating the result of
/// a call it never made — mapping it onto a tool named "output" would
/// fabricate a call out of thin air. `name` guards the bare XML-element-form
/// child (`<tool_name>shell</tool_name>`) appearing outside its wrapper.
const xml_tag_reserved_names = [_][]const u8{
    "output", "outputs", "result", "results", "response", "responses", "error", "errors", "name",
};

/// XML-element-TAG tool form: `<tool_NAME><key>value</key>…</tool_NAME>`,
/// the tool name embedded in the tag itself (DSV4 training-bias family;
/// captured live: `<tool_read>` / `<tool_edit>`). `slice` starts
/// at `<tool`. Returns the parsed call plus bytes consumed through the close
/// tag, or null when the shape doesn't hold (no attributes allowed, exact
/// close required, reserved names rejected, body must be all elements).
pub fn parseXmlElementTagToolCall(
    allocator: std.mem.Allocator,
    slice: []const u8,
) !?struct { call: ParsedToolCall, consumed: usize } {
    std.debug.assert(std.mem.startsWith(u8, slice, "<tool"));
    const name_start = "<tool_".len;
    if (slice.len <= name_start) return null;
    var name_end = name_start;
    while (name_end < slice.len and
        (std.ascii.isAlphanumeric(slice[name_end]) or slice[name_end] == '_')) : (name_end += 1)
    {}
    if (name_end == name_start) return null; // bare `<tool_>`
    if (name_end >= slice.len or slice[name_end] != '>') return null; // attributes → not this form
    const tool_name = slice[name_start..name_end];
    for (xml_tag_reserved_names) |reserved| {
        if (std.ascii.eqlIgnoreCase(tool_name, reserved)) return null;
    }
    var close_buf: [72]u8 = undefined;
    if (tool_name.len + "</tool_>".len > close_buf.len) return null;
    const close_tag = std.fmt.bufPrint(&close_buf, "</tool_{s}>", .{tool_name}) catch return null;
    const body_start = name_end + 1;
    const close_rel = std.mem.indexOf(u8, slice[body_start..], close_tag) orelse return null;
    const body = slice[body_start .. body_start + close_rel];
    const args = parseXmlElementArgsJson(allocator, body) orelse blk: {
        // JSON-args body variant of the same form, also captured live:
        // `<tool_write>\n{"path": …, "content": …}\n</tool_write>`. The whole
        // object IS the args. The trimmed body must START with `{` so prose
        // that merely contains braces never produces arguments.
        const trimmed = std.mem.trim(u8, body, " \t\n\r");
        if (trimmed.len == 0 or trimmed[0] != '{') return null;
        const json_body = balancedJsonObject(trimmed) orelse return null;
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_body, .{}) catch return null;
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        break :blk allocator.dupe(u8, json_body) catch return null;
    };
    const name_owned = allocator.dupe(u8, tool_name) catch {
        allocator.free(args);
        return null;
    };
    return .{
        .call = .{ .name = name_owned, .arguments = args },
        .consumed = body_start + close_rel + close_tag.len,
    };
}

/// Parse a body consisting ENTIRELY of `<key>value</key>` child elements into
/// a JSON args string. Values are kept verbatim, so nested markup (DSV4's
/// `<edits><oldText>…</oldText>…</edits>`) survives as the arg's string
/// value. Returns null unless every non-whitespace byte belongs to an element
/// and at least one element is present — a plain-text body must never
/// produce arguments.
fn parseXmlElementArgsJson(allocator: std.mem.Allocator, body: []const u8) ?[]u8 {
    var args_map: std.json.ObjectMap = .empty;
    defer args_map.deinit(allocator);
    var count: usize = 0;
    var i: usize = 0;
    while (i < body.len) {
        while (i < body.len and std.ascii.isWhitespace(body[i])) i += 1;
        if (i >= body.len) break;
        if (body[i] != '<') return null;
        const key_start = i + 1;
        var j = key_start;
        while (j < body.len and (std.ascii.isAlphanumeric(body[j]) or body[j] == '_')) j += 1;
        if (j == key_start or j >= body.len or body[j] != '>') return null;
        const key = body[key_start..j];
        const val_start = j + 1;
        var close_buf: [64]u8 = undefined;
        if (key.len + 3 > close_buf.len) return null;
        const close_tag = std.fmt.bufPrint(&close_buf, "</{s}>", .{key}) catch return null;
        const rel = std.mem.indexOf(u8, body[val_start..], close_tag) orelse return null;
        args_map.put(allocator, key, .{ .string = body[val_start .. val_start + rel] }) catch return null;
        count += 1;
        i = val_start + rel + close_tag.len;
    }
    if (count == 0) return null;
    return std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{}) catch null;
}

/// Hy3 (Hunyuan 3) tag-format tool calls (chat_template.jinja spec):
///   <tool_calls:SFX>
///   <tool_call:SFX>NAME<tool_sep:SFX>
///   <arg_key:SFX>KEY</arg_key:SFX>
///   <arg_value:SFX>VALUE</arg_value:SFX> …
///   </tool_call:SFX> …
///   </tool_calls:SFX>
/// Only the SUFFIXED form (`<tool_call:` …) is handled here — bare
/// `<tool_call>` is Hermes JSON and stays with the generic scan. Values are
/// kept as RAW STRINGS: the template `tojson`s non-string values on the way
/// in, and the schema-driven coercion at the server chokepoint types them on
/// the way back (types come from the SCHEMA, never the value's spelling).
/// Args build through ObjectMap + Stringify — keys escaped, duplicates
/// first-wins — per the tag-format-converter class rule. Truncation
/// (max_tokens mid-call, the big-file-write class): a call whose name is
/// delimited by `<tool_sep` recovers with its CLOSED key/value pairs only;
/// partial values are never salvaged.
/// Earliest position at/after `from` where any of `needles` occurs, or null.
pub fn isBareToolName(body: []const u8) bool {
    const name = std.mem.trim(u8, body, " \t\n\r");
    if (name.len == 0) return false;
    for (name) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '_' or c == '-' or c == '.')) return false;
    }
    return true;
}

pub fn parseXmlElementToolCall(allocator: std.mem.Allocator, body: []const u8) ?ParsedToolCall {
    var name: ?[]const u8 = null;
    var args_map: std.json.ObjectMap = .empty;
    defer args_map.deinit(allocator);

    var i: usize = 0;
    while (i < body.len) {
        while (i < body.len and std.ascii.isWhitespace(body[i])) i += 1;
        if (i >= body.len) break;
        if (body[i] != '<') return null;
        const key_start = i + 1;
        var j = key_start;
        while (j < body.len and (std.ascii.isAlphanumeric(body[j]) or body[j] == '_')) j += 1;
        if (j == key_start or j >= body.len or body[j] != '>') return null;
        const key = body[key_start..j];
        const val_start = j + 1;
        var close_buf: [64]u8 = undefined;
        if (key.len + 3 > close_buf.len) return null;
        const close_tag = std.fmt.bufPrint(&close_buf, "</{s}>", .{key}) catch return null;
        const rel = std.mem.indexOf(u8, body[val_start..], close_tag) orelse return null;
        const value = body[val_start .. val_start + rel];
        i = val_start + rel + close_tag.len;

        if (std.mem.eql(u8, key, "tool_name") or std.mem.eql(u8, key, "name") or std.mem.eql(u8, key, "tool")) {
            if (name != null) return null;
            const trimmed_name = std.mem.trim(u8, value, " \t\n\r");
            if (trimmed_name.len == 0) return null;
            name = trimmed_name;
        } else {
            args_map.put(allocator, key, .{ .string = value }) catch return null;
        }
    }

    const n = name orelse return null;
    const args_str = std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{}) catch return null;
    const name_owned = allocator.dupe(u8, n) catch {
        allocator.free(args_str);
        return null;
    };
    return .{ .name = name_owned, .arguments = args_str };
}
