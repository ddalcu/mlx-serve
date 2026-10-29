//! Loose JSON repair, `tryParseJsonToolCall`, and `sanitizeToolName`.

const std = @import("std");
const log = @import("../../../sdk/util/log.zig");
const testing = std.testing;

const parseToolCalls = @import("parse.zig").parseToolCalls;
const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const freeParsedCalls = @import("dsml.zig").freeParsedCalls;

/// Repair a JSON object whose trailing closer(s) were dropped — e.g. the
/// model wrote `{"name":"edit","arguments":{…,"edits":[{…}]}` and went
/// straight to its close tag, one `}` short. String/escape-aware bracket
/// stack; appends exactly the missing closers in nesting order. Returns null
/// when the text doesn't start with `{`, is already balanced (trailing
/// garbage is someone else's problem), ends mid-string, or nests deeper
/// than the stack (not worth guessing).
fn completeUnbalancedJsonObject(allocator: std.mem.Allocator, content: []const u8) ?[]u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    if (trimmed.len == 0 or trimmed[0] != '{') return null;
    var stack: [16]u8 = undefined;
    var depth: usize = 0;
    var in_string = false;
    var escape = false;
    for (trimmed) |c| {
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        switch (c) {
            '"' => in_string = true,
            '{' => {
                if (depth == stack.len) return null;
                stack[depth] = '}';
                depth += 1;
            },
            '[' => {
                if (depth == stack.len) return null;
                stack[depth] = ']';
                depth += 1;
            },
            '}', ']' => {
                if (depth == 0 or stack[depth - 1] != c) return null;
                depth -= 1;
                // Balanced before the end → trailing garbage, not a
                // truncated tail. balancedJsonObject owns that case.
                if (depth == 0) return null;
            },
            else => {},
        }
    }
    if (in_string or depth == 0) return null;
    var out = allocator.alloc(u8, trimmed.len + depth) catch return null;
    @memcpy(out[0..trimmed.len], trimmed);
    var i: usize = 0;
    while (i < depth) : (i += 1) {
        out[trimmed.len + i] = stack[depth - 1 - i];
    }
    return out;
}

/// Snap a balanced JSON array starting at the first `[`. Mirrors
/// balancedJsonObject: string/escape aware, tracks BOTH bracket kinds so
/// objects nested in the array can't fool the depth count.
pub fn balancedJsonArray(content: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const start = std.mem.indexOfScalar(u8, trimmed, '[') orelse return null;
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
        switch (c) {
            '"' => in_string = true,
            '[', '{' => depth += 1,
            ']', '}' => {
                depth -= 1;
                if (depth == 0) return trimmed[start .. i + 1];
                if (depth < 0) return null; // mismatched — give up
            },
            else => {},
        }
    }
    return null;
}

/// Parse a JSON array of tool-call objects (`[{"name":…,"arguments":…}, …]`)
/// and append EVERY call. All-or-nothing: if any element fails to parse as a
/// tool call, nothing is appended — a prose-ish array must pass through as
/// text rather than half-execute.
pub fn appendJsonToolCallArray(
    allocator: std.mem.Allocator,
    arr_text: []const u8,
    calls: *std.ArrayList(ParsedToolCall),
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, arr_text, .{}) catch return;
    defer parsed.deinit();
    if (parsed.value != .array) return;
    const items = parsed.value.array.items;
    if (items.len == 0) return;

    var pending = std.ArrayList(ParsedToolCall).empty;
    var ok = true;
    defer {
        for (pending.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        pending.deinit(allocator);
    }
    for (items) |item| {
        if (item != .object) {
            ok = false;
            break;
        }
        // Reuse tryParseJsonToolCall (leaf-name walk, flat-shape synthesis)
        // by round-tripping the element through its text form.
        const item_text = std.json.Stringify.valueAlloc(allocator, item, .{}) catch {
            ok = false;
            break;
        };
        defer allocator.free(item_text);
        const tc = tryParseJsonToolCall(allocator, item_text) orelse {
            ok = false;
            break;
        };
        try pending.append(allocator, tc);
    }
    if (!ok) return;
    try calls.appendSlice(allocator, pending.items);
    pending.clearRetainingCapacity();
}

/// Lenient recovery for tool-call argument JSON that small models mangle when
/// emitting large string values (file contents). Two failure modes, both of
/// which make strict std.json reject the WHOLE blob so the call would be dropped
/// and leak as visible text:
///   • raw control bytes (literal newlines/tabs) inside a string instead of
///     `\n`/`\t` — the dominant big-file failure;
///   • unescaped inner double-quotes (`<meta charset="UTF-8">`) and invalid
///     backslash escapes (Windows paths, regex) inside a string.
/// Re-serializes a CLEAN object by walking the input with a position-aware
/// tolerant parser (key vs value context): control bytes are re-escaped, lone/
/// invalid backslashes are escaped, and a `"` closes a string only at a
/// structural delimiter (`:` after a key; `,`/`}`/`]`/end after a value) — any
/// other `"` is an inner content quote and gets escaped. The CALLER strict-parses
/// the result, so a mis-recovery that yields invalid JSON is discarded; the only
/// residual risk is a value string closed early on pathological content (a
/// literal `"}` / `",` byte-sequence inside the file), which still beats dropping
/// the call. Does NOT tolerate truncation (no auto-close of open containers) —
/// that stays with completeUnbalancedJsonObject. Returns an allocator-owned
/// normalized JSON string, or null when no well-formed leading object recovers.
pub fn looseRepairToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    const obj_start = std.mem.indexOfScalar(u8, trimmed, '{') orelse return null;
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    _ = looseEmitObject(allocator, &out, trimmed, obj_start, 0) catch {
        out.deinit(allocator);
        return null;
    };
    return out.toOwnedSlice(allocator) catch null;
}

/// Tolerant re-serialize of a top-level JSON CONTAINER (`[…]` or `{…}`) — the
/// array-aware sibling of looseRepairToolCallJson. Used by the schema coercion to
/// salvage a container-typed argument the model mangled (e.g. a missing comma
/// inside an `edits` array). Same discipline: the caller strict-parses the
/// result, so a mis-repair yielding invalid JSON is discarded. Returns null when
/// the text doesn't start with `[`/`{` or can't be re-serialized.
pub fn looseRepairContainer(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0 or (trimmed[0] != '[' and trimmed[0] != '{')) return null;
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    _ = looseEmitValue(allocator, &out, trimmed, 0, 0, .object_value) catch {
        out.deinit(allocator);
        return null;
    };
    return out.toOwnedSlice(allocator) catch null;
}

const LooseStringCtx = enum { key, object_value, array_value };
const LooseError = error{ Malformed, OutOfMemory };
const loose_max_depth = 32;

fn looseSkipWs(body: []const u8, start: usize) usize {
    var i = start;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            ' ', '\t', '\n', '\r' => {},
            else => break,
        }
    }
    return i;
}

fn looseSkipWsCommas(body: []const u8, start: usize) usize {
    var i = start;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            ' ', '\t', '\n', '\r', ',' => {},
            else => break,
        }
    }
    return i;
}

fn looseIsHex4(s: []const u8) bool {
    if (s.len != 4) return false;
    for (s) |c| {
        const ok = (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F');
        if (!ok) return false;
    }
    return true;
}

fn looseEmitObject(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
) LooseError!usize {
    if (depth > loose_max_depth) return error.Malformed;
    var i = looseSkipWs(body, start);
    if (i >= body.len or body[i] != '{') return error.Malformed;
    i += 1;
    try out.append(allocator, '{');
    var first = true;
    while (true) {
        i = looseSkipWsCommas(body, i);
        if (i >= body.len) return error.Malformed; // unterminated (not truncation-tolerant)
        if (body[i] == '}') {
            try out.append(allocator, '}');
            return i + 1;
        }
        if (body[i] != '"') return error.Malformed; // key must be a string
        if (!first) try out.append(allocator, ',');
        first = false;
        i = try looseEmitString(allocator, out, body, i, .key);
        i = looseSkipWs(body, i);
        if (i >= body.len or body[i] != ':') return error.Malformed;
        i += 1;
        try out.append(allocator, ':');
        i = try looseEmitValue(allocator, out, body, i, depth, .object_value);
    }
}

fn looseEmitArray(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
) LooseError!usize {
    if (depth > loose_max_depth) return error.Malformed;
    var i = looseSkipWs(body, start);
    if (i >= body.len or body[i] != '[') return error.Malformed;
    i += 1;
    try out.append(allocator, '[');
    var first = true;
    while (true) {
        i = looseSkipWsCommas(body, i);
        if (i >= body.len) return error.Malformed;
        if (body[i] == ']') {
            try out.append(allocator, ']');
            return i + 1;
        }
        if (!first) try out.append(allocator, ',');
        first = false;
        i = try looseEmitValue(allocator, out, body, i, depth, .array_value);
    }
}

fn looseEmitValue(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    str_ctx: LooseStringCtx,
) LooseError!usize {
    const i = looseSkipWs(body, start);
    if (i >= body.len) return error.Malformed;
    return switch (body[i]) {
        '"' => try looseEmitString(allocator, out, body, i, str_ctx),
        '{' => try looseEmitObject(allocator, out, body, i, depth + 1),
        '[' => try looseEmitArray(allocator, out, body, i, depth + 1),
        else => try looseEmitScalar(allocator, out, body, i),
    };
}

fn looseEmitScalar(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
) LooseError!usize {
    var i = start;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            ',', '}', ']', ' ', '\t', '\n', '\r' => break,
            else => try out.append(allocator, body[i]),
        }
    }
    if (i == start) return error.Malformed;
    return i; // strict re-parse validates the token was a real number/bool/null
}

fn looseEmitString(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    ctx: LooseStringCtx,
) LooseError!usize {
    // body[start] == '"'
    try out.append(allocator, '"');
    var i = start + 1;
    while (i < body.len) {
        const c = body[i];
        if (c == '\\') {
            if (i + 1 >= body.len) {
                try out.appendSlice(allocator, "\\\\"); // trailing backslash → literal
                i += 1;
                continue;
            }
            const n = body[i + 1];
            switch (n) {
                '"', '\\', '/', 'b', 'f', 'n', 'r', 't' => {
                    try out.append(allocator, '\\');
                    try out.append(allocator, n);
                    i += 2;
                },
                'u' => {
                    if (i + 6 <= body.len and looseIsHex4(body[i + 2 .. i + 6])) {
                        try out.appendSlice(allocator, body[i .. i + 6]);
                        i += 6;
                    } else {
                        try out.appendSlice(allocator, "\\\\"); // invalid \u → literal backslash
                        i += 1;
                    }
                },
                else => {
                    try out.appendSlice(allocator, "\\\\"); // invalid escape → literal backslash
                    i += 1;
                },
            }
            continue;
        }
        if (c == '"') {
            const j = looseSkipWs(body, i + 1);
            const is_close = switch (ctx) {
                .key => j < body.len and body[j] == ':',
                .object_value => j >= body.len or body[j] == ',' or body[j] == '}',
                .array_value => j >= body.len or body[j] == ',' or body[j] == ']',
            };
            if (is_close) {
                try out.append(allocator, '"');
                return i + 1;
            }
            try out.appendSlice(allocator, "\\\""); // inner content quote → escape
            i += 1;
            continue;
        }
        if (c < 0x20) {
            switch (c) {
                '\n' => try out.appendSlice(allocator, "\\n"),
                '\t' => try out.appendSlice(allocator, "\\t"),
                '\r' => try out.appendSlice(allocator, "\\r"),
                0x08 => try out.appendSlice(allocator, "\\b"),
                0x0c => try out.appendSlice(allocator, "\\f"),
                else => {
                    var buf: [6]u8 = undefined;
                    const esc = std.fmt.bufPrint(&buf, "\\u{x:0>4}", .{c}) catch unreachable;
                    try out.appendSlice(allocator, esc);
                },
            }
            i += 1;
            continue;
        }
        try out.append(allocator, c);
        i += 1;
    }
    // Ran off the end with no structural close — unterminated. The enclosing
    // container won't find its close either → looseEmitObject returns Malformed
    // and the whole recovery is discarded (truncation handled elsewhere). Emit
    // a closing quote so the discarded buffer stays well-formed.
    try out.append(allocator, '"');
    return i;
}

pub fn tryParseJsonToolCall(allocator: std.mem.Allocator, text: []const u8) ?ParsedToolCall {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch blk: {
        // Strict parse failed. Try a chain of repairs for known shapes:
        //   1. {"name":"shell", {"command":"ls"}}           — missing `"arguments":` key entirely (Qwen MoE)
        //   2. {"name":"shell", arguments":{"command":..}}   — missing OPENING quote on `arguments` (Qwen MoE)
        //   3. {"name":"edit", "arguments": {…}]}            — truncated tail, final closer(s) dropped
        //      before the close tag (DSV4-Flash, captured live)
        // Repairs are cheap and run only on the parse-failure path.
        if (repairBrokenToolCallJson(allocator, text)) |repaired| {
            defer allocator.free(repaired);
            if (std.json.parseFromSlice(std.json.Value, allocator, repaired, .{})) |reparsed| {
                break :blk reparsed;
            } else |_| {}
        }
        //   4. Mangled big-file content: raw control bytes (literal newlines/
        //      tabs) and unescaped inner quotes inside a string value — what
        //      small models emit when writing a large file in one shot. The
        //      re-escaped copy is re-validated by the strict parse below, so a
        //      mis-recovery that yields invalid JSON is silently discarded.
        if (looseRepairToolCallJson(allocator, text)) |repaired| {
            defer allocator.free(repaired);
            if (std.json.parseFromSlice(std.json.Value, allocator, repaired, .{})) |reparsed| {
                log.info("  [tool-parse] loose-repair recovered mangled tool-call JSON (raw control bytes / unescaped quotes)\n", .{});
                break :blk reparsed;
            } else |_| {}
        }
        const completed = completeUnbalancedJsonObject(allocator, text) orelse return null;
        defer allocator.free(completed);
        const reparsed = std.json.parseFromSlice(std.json.Value, allocator, completed, .{}) catch return null;
        break :blk reparsed;
    };
    defer parsed.deinit();

    if (parsed.value != .object) return null;
    var obj = parsed.value.object;

    // Qwen 3.5/3.6 MoE sometimes emits nested-name garbage like
    //   {"name":{"name":{"name":"write","arguments":{...}}}}
    // Walk down through up to a few levels of nested "name" objects to find the
    // leaf object that has a string name + arguments/parameters. Observed in the
    // wild with pi + Qwen3.6-35B in non-thinking mode.
    {
        var depth: u8 = 0;
        while (depth < 4) : (depth += 1) {
            const nv = obj.get("name") orelse return null;
            switch (nv) {
                .string => break,
                .object => |inner| obj = inner,
                else => return null,
            }
        }
    }

    const name_val = obj.get("name") orelse return null;
    if (name_val != .string) return null;

    const args_str: []const u8 = blk: {
        if (obj.get("arguments") orelse obj.get("parameters")) |args_val| {
            break :blk switch (args_val) {
                .object => std.json.Stringify.valueAlloc(allocator, args_val, .{}) catch return null,
                .string => |s| allocator.dupe(u8, s) catch return null,
                else => return null,
            };
        }
        // Flat shape (e.g. Qwen MoE): {"name":"shell","command":"ls"} —
        // parameters live at top level. Synthesize an arguments object from
        // every non-metadata key.
        var flat_map: std.json.ObjectMap = .empty;
        defer flat_map.deinit(allocator);
        var it = obj.iterator();
        while (it.next()) |entry| {
            const k = entry.key_ptr.*;
            if (std.mem.eql(u8, k, "name") or
                std.mem.eql(u8, k, "id") or
                std.mem.eql(u8, k, "type"))
            {
                continue;
            }
            flat_map.put(allocator, k, entry.value_ptr.*) catch return null;
        }
        if (flat_map.count() == 0) return null;
        const flat_value = std.json.Value{ .object = flat_map };
        break :blk std.json.Stringify.valueAlloc(allocator, flat_value, .{}) catch return null;
    };

    return .{
        .name = allocator.dupe(u8, name_val.string) catch {
            allocator.free(args_str);
            return null;
        },
        .arguments = args_str,
    };
}

/// Run known Qwen-MoE-broken-JSON repairs in sequence; return the first
/// successfully-repaired string (allocator-owned), or null if none match.
fn repairBrokenToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    if (repairFlatBraceToolCallJson(allocator, text)) |s| return s;
    if (repairUnquotedArgsKey(allocator, text)) |s| return s;
    return null;
}

/// Repair `{"name":"x", arguments":{...}}` (missing OPENING quote on
/// `arguments`/`parameters`) by injecting the quote.
fn repairUnquotedArgsKey(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");
    // Look for the unquoted-key pattern. Both `arguments` and `parameters`
    // have been observed; both are short fixed strings, so direct search is fine.
    const candidates = [_][]const u8{ ", arguments\":", ",arguments\":", ", parameters\":", ",parameters\":" };
    for (candidates) |needle| {
        if (std.mem.indexOf(u8, trimmed, needle)) |at| {
            const insert_at = at + 1; // right after the comma
            // Skip leading whitespace after the comma so the injected `"` lands on the identifier.
            var p = insert_at;
            while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
            return std.fmt.allocPrint(allocator, "{s}\"{s}", .{ trimmed[0..p], trimmed[p..] }) catch null;
        }
    }
    return null;
}

/// Repair the Qwen-MoE-broken shape `{"name":"x", {"k":"v"}}` by injecting
/// the missing `"arguments":` key between the comma and the inner object.
/// Returns an allocator-owned repaired JSON string, or null if the pattern
/// doesn't match.
fn repairFlatBraceToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");
    if (trimmed.len < 4 or trimmed[0] != '{') return null;

    // Locate `"name"` key.
    const name_at = std.mem.indexOf(u8, trimmed, "\"name\"") orelse return null;
    var p = name_at + 6;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
    if (p >= trimmed.len or trimmed[p] != ':') return null;
    p += 1;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
    if (p >= trimmed.len or trimmed[p] != '"') return null;

    // Skip the string value.
    p += 1;
    while (p < trimmed.len and trimmed[p] != '"') {
        if (trimmed[p] == '\\' and p + 1 < trimmed.len) p += 2 else p += 1;
    }
    if (p >= trimmed.len) return null;
    p += 1;

    // Expect `, {` (any whitespace).
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t' or trimmed[p] == '\n')) p += 1;
    if (p >= trimmed.len or trimmed[p] != ',') return null;
    p += 1;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t' or trimmed[p] == '\n')) p += 1;
    if (p >= trimmed.len or trimmed[p] != '{') return null;

    return std.fmt.allocPrint(allocator, "{s}\"arguments\":{s}", .{ trimmed[0..p], trimmed[p..] }) catch null;
}

/// Strip one layer of outer braces from a `{{…}}`-wrapped JSON object. Models
/// sometimes emit args with a doubled brace layer (a literal-brace artifact from
/// Jinja `{{ }}` templates). Only touches strings that start with `{{` and end
/// with `}}`, so well-formed single-brace JSON passes through unchanged.
pub fn unwrapDoubleBraces(s: []const u8) []const u8 {
    if (s.len >= 4 and std.mem.startsWith(u8, s, "{{") and std.mem.endsWith(u8, s, "}}")) {
        return s[1 .. s.len - 1];
    }
    return s;
}

/// Strip artifacts models sometimes append to a tool name — surrounding
/// whitespace and a trailing ':' (Gemma 4 12B leaks one via its `call:NAME:`
/// format). A valid OpenAI tool name matches `^[A-Za-z0-9_-]{1,64}$`, so a
/// trailing colon is never legitimate and is always safe to drop.
pub fn sanitizeToolName(raw: []const u8) []const u8 {
    var name = std.mem.trim(u8, raw, " \t\n\r");
    while (name.len > 0 and name[name.len - 1] == ':') {
        name = std.mem.trim(u8, name[0 .. name.len - 1], " \t\n\r");
    }
    return name;
}

// ── tests ──

test "parseToolCalls repairs Qwen MoE missing-arguments-key shape" {
    const allocator = testing.allocator;
    // Real broken output observed from Qwen3.6-35B-A3B-6bit:
    // {"name": "shell",  {"command":"ls"}}  — `, {` instead of `, "arguments": {`
    const text = "<tool_call>\n{\"name\":  \"shell\",     {\"command\":\"ls -la\"}}\n</tool_call>";
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
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls repairs Qwen MoE missing-opening-quote on arguments key" {
    const allocator = testing.allocator;
    // Real broken output observed from Qwen3.6-35B-A3B-6bit:
    // {"name": "shell", arguments": {"command":"mkdir -p src/app"}}
    // — missing the OPENING `"` on the `arguments` key.
    const text = "<tool_call>\n{\"name\": \"shell\", arguments\": {\"command\": \"mkdir -p src/app\"}}\n</tool_call>";
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
    try testing.expectEqualStrings("mkdir -p src/app", parsed.value.object.get("command").?.string);
}

test "parseToolCalls recovers writeFile content with RAW newlines (small-model big-file escaping)" {
    const allocator = testing.allocator;
    // Small models writing a big file often emit literal newlines inside the
    // JSON `content` string instead of `\n` — strict JSON rejects raw control
    // bytes, so pre-fix the whole call was dropped and the file leaked as text.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"a.js\",\"content\":\"const x = 1;\nconst y = 2;\n\"}}</tool_call>";
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
    try testing.expectEqualStrings("a.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("const x = 1;\nconst y = 2;\n", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers writeFile content with RAW tab" {
    const allocator = testing.allocator;
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"m.py\",\"content\":\"def f():\n\treturn 1\"}}</tool_call>";
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
    try testing.expectEqualStrings("def f():\n\treturn 1", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers writeFile content with UNESCAPED inner quotes (HTML)" {
    const allocator = testing.allocator;
    // HTML/code with attribute quotes — `<meta charset="UTF-8">` — is the other
    // half of the escaping class: the model forgets to backslash the inner `"`.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"i.html\",\"content\":\"<meta charset=\"UTF-8\"><a href=\"x\">go</a>\"}}</tool_call>";
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
    try testing.expectEqualStrings("i.html", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("<meta charset=\"UTF-8\"><a href=\"x\">go</a>", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers writeFile content with BOTH raw newlines and unescaped quotes" {
    const allocator = testing.allocator;
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"p.html\",\"content\":\"<!DOCTYPE html>\n<meta charset=\"UTF-8\">\n<title>Hi</title>\"}}</tool_call>";
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
    try testing.expectEqualStrings("<!DOCTYPE html>\n<meta charset=\"UTF-8\">\n<title>Hi</title>", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers content with lone backslash (Windows path / regex)" {
    const allocator = testing.allocator;
    // `\U` and `\d` are invalid JSON escapes — strict parse rejects them.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"C:\\Users\\app.js\",\"content\":\"re.match(\\d+)\"}}</tool_call>";
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
    try testing.expectEqualStrings("C:\\Users\\app.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("re.match(\\d+)", parsed.value.object.get("content").?.string);
}

test "looseRepair does not fabricate a tool call from non-JSON prose" {
    const allocator = testing.allocator;
    // A stray `{` in prose must not become a tool call via the recovery path.
    const text = "Here is the plan {step one, step two} and we proceed.";
    const calls = try parseToolCalls(allocator, text);
    if (calls) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    }
    try testing.expect(calls == null);
}

test "sanitizeToolName strips trailing colons and whitespace" {
    try testing.expectEqualStrings("shell", sanitizeToolName("shell:"));
    try testing.expectEqualStrings("shell", sanitizeToolName("  shell  "));
    try testing.expectEqualStrings("shell", sanitizeToolName("shell : "));
    try testing.expectEqualStrings("shell", sanitizeToolName("shell::"));
    try testing.expectEqualStrings("cwd", sanitizeToolName("cwd"));
    try testing.expectEqualStrings("", sanitizeToolName(":"));
}

test "parseToolCalls: <tool_call>{JSON} truncated mid-string recovers NAME + {} (never a fragment)" {
    // A 4 KB edit call that hit EOS inside a string value. The object never
    // balances, so the balanced-snap arm declines; without a name salvage the
    // whole turn was dropped at the flush (empty `stop`, nothing for the client
    // to nudge on). Same rule as every other dialect: NAME + `{}`.
    const allocator = testing.allocator;
    const text = "Fixing it.\n<tool_call>\n{\"name\": \"edit\", \"arguments\": {\"path\": \"a.ts\", \"oldText\": \"function foo() {\\n  return 1;";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("edit", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
    // Reversed key order + whitespace variants still find the name.
    const text2 = "<tool_call>{ \"arguments\" : {\"name\": \"nope\", \"s\": \"a}b\"}, \"name\" : \"real\", \"extra\": \"unterm";
    const c2 = (try parseToolCalls(allocator, text2)) orelse return error.NoCalls;
    defer freeParsedCalls(c2);
    try testing.expectEqualStrings("real", c2[0].name);
    // No name anywhere: still nothing (no invented call).
    try testing.expect((try parseToolCalls(allocator, "<tool_call>\n{\"arguments\": {\"path\": \"a")) == null);
}

// ── MiniCPM5 V3 XML (`<function name="X"><param name="K">V</param></function>`) ──
