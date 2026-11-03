//! The pythonic tool-call dialect (LFM2/LFM2.5).

const std = @import("std");
const testing = std.testing;

const parseToolCalls = @import("parse.zig").parseToolCalls;
const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const freeParsedCalls = @import("dsml.zig").freeParsedCalls;

// ── Pythonic tool calls (LFM2.5 / `lfm2`) ──────────────────────────────
//
//   <|tool_call_start|>[get_weather(city='Paris', days=3, metric=True,
//                                   tags=['trip', 'eu'])]<|tool_call_end|>
//
// A CALL-EXPRESSION grammar, not a JSON one: values are Python literals. The
// model emits full repr — `True`, single-quoted lists — even though its own
// template renders history containers via `tojson`, so a JSON-only value
// reader mis-reads the two most common non-string types. Parsing the literals
// here (rather than shipping them as text for `coerceToolArgsToSchema` to
// rescue) is what keeps a declared `array`/`boolean` parameter from arriving
// as a string — the value types are knowable from the SPELLING in this
// grammar, so the schema never has to be consulted.
//
// Marker-gated: `<|tool_call_start|>` is emitted by no other family, and the
// generic `<tool` scan never sees it, so this arm cannot touch existing
// traffic. The Gemma 4 arm keys on the exact `<|tool_call>` (with `>`), which
// this never matches.
const PYTHONIC_CALL_START = "<|tool_call_start|>";
const PYTHONIC_CALL_END = "<|tool_call_end|>";

fn pythonicIsNameChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '.' or c == '-';
}

/// Index of the first byte in `s` at nesting depth 0 that satisfies `pred`,
/// skipping anything inside a quoted string or a nested ()/[]/{} group.
/// One scanner for every structural search in this grammar — the arg
/// separator, the `=`, the dict `:`, and the closing paren all need the same
/// blindness to separators that live inside a value.
fn pythonicScan(s: []const u8, start: usize, pred: *const fn (u8, i32) bool) ?usize {
    var depth: i32 = 0;
    var quote: u8 = 0;
    var i = start;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        if (quote != 0) {
            if (c == '\\') {
                i += 1;
            } else if (c == quote) {
                quote = 0;
            }
            continue;
        }
        if (c == '\'' or c == '"') {
            quote = c;
            continue;
        }
        if (c == '(' or c == '[' or c == '{') {
            depth += 1;
        } else if (c == ')' or c == ']' or c == '}') {
            if (pred(c, depth)) return i;
            depth -= 1;
            continue;
        }
        if (pred(c, depth)) return i;
    }
    return null;
}

fn pythonicIsCloser(c: u8, depth: i32) bool {
    return c == ')' and depth == 0;
}
fn pythonicIsComma(c: u8, depth: i32) bool {
    return c == ',' and depth == 0;
}
fn pythonicIsEquals(c: u8, depth: i32) bool {
    return c == '=' and depth == 0;
}
fn pythonicIsColon(c: u8, depth: i32) bool {
    return c == ':' and depth == 0;
}

/// Unescape a Python string literal body. Unknown escapes keep the backslash,
/// which is Python's own behavior (`'\d' == '\\d'`) and the forgiving reading
/// of a model that under-escaped a path.
fn pythonicUnescape(arena: std.mem.Allocator, body: []const u8) ![]const u8 {
    var out = std.ArrayList(u8).empty;
    var i: usize = 0;
    while (i < body.len) : (i += 1) {
        if (body[i] != '\\' or i + 1 >= body.len) {
            try out.append(arena, body[i]);
            continue;
        }
        i += 1;
        switch (body[i]) {
            'n' => try out.append(arena, '\n'),
            'r' => try out.append(arena, '\r'),
            't' => try out.append(arena, '\t'),
            '\\', '\'', '"' => try out.append(arena, body[i]),
            else => {
                try out.append(arena, '\\');
                try out.append(arena, body[i]);
            },
        }
    }
    return out.toOwnedSlice(arena);
}

/// A Python literal → its JSON value. Anything undecidable stays a string
/// VERBATIM rather than being guessed at, so a client's own validation error
/// stays honest (the `coerceToolArgsToSchema` contract, one layer down).
fn pythonicLiteral(arena: std.mem.Allocator, raw: []const u8) !std.json.Value {
    const s = std.mem.trim(u8, raw, " \t\n\r");
    if (s.len == 0) return .{ .string = "" };

    if ((s[0] == '\'' or s[0] == '"') and s.len >= 2 and s[s.len - 1] == s[0]) {
        return .{ .string = try pythonicUnescape(arena, s[1 .. s.len - 1]) };
    }
    if (std.mem.eql(u8, s, "True")) return .{ .bool = true };
    if (std.mem.eql(u8, s, "False")) return .{ .bool = false };
    if (std.mem.eql(u8, s, "None")) return .null;

    if (s[0] == '[' and s[s.len - 1] == ']') {
        var arr = std.json.Array.init(arena);
        var it = PythonicParts{ .s = s[1 .. s.len - 1] };
        while (it.next()) |part| {
            if (std.mem.trim(u8, part, " \t\n\r").len == 0) continue;
            try arr.append(try pythonicLiteral(arena, part));
        }
        return .{ .array = arr };
    }
    if (s[0] == '{' and s[s.len - 1] == '}') {
        var obj: std.json.ObjectMap = .empty;
        var it = PythonicParts{ .s = s[1 .. s.len - 1] };
        while (it.next()) |part| {
            const colon = pythonicScan(part, 0, pythonicIsColon) orelse continue;
            const key_v = try pythonicLiteral(arena, part[0..colon]);
            const key = switch (key_v) {
                .string => |k| k,
                else => continue,
            };
            if (key.len == 0 or obj.getEntry(key) != null) continue;
            try obj.put(arena, key, try pythonicLiteral(arena, part[colon + 1 ..]));
        }
        return .{ .object = obj };
    }

    if (std.fmt.parseInt(i64, s, 10)) |n| return .{ .integer = n } else |_| {}
    if (std.fmt.parseFloat(f64, s)) |f| return .{ .float = f } else |_| {}
    return .{ .string = s };
}

/// Iterator over top-level comma-separated parts.
const PythonicParts = struct {
    s: []const u8,
    i: usize = 0,

    fn next(self: *PythonicParts) ?[]const u8 {
        if (self.i > self.s.len) return null;
        if (self.i == self.s.len) {
            self.i += 1;
            return self.s[self.s.len..];
        }
        const start = self.i;
        if (pythonicScan(self.s, start, pythonicIsComma)) |c| {
            self.i = c + 1;
            return self.s[start..c];
        }
        self.i = self.s.len + 1;
        return self.s[start..];
    }
};

pub fn parsePythonicToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, PYTHONIC_CALL_START)) |p| {
        const body_start = p + PYTHONIC_CALL_START.len;
        const close = std.mem.indexOfPos(u8, text, body_start, PYTHONIC_CALL_END);
        pos = if (close) |c| c + PYTHONIC_CALL_END.len else text.len;

        // A dropped end marker (truncation, or the weak-model delimiter-drop
        // class) must not cost the call — the body simply runs to the end.
        var body = std.mem.trim(u8, text[body_start .. close orelse text.len], " \t\n\r");
        if (body.len > 0 and body[0] == '[') body = body[1..];
        if (body.len > 0 and body[body.len - 1] == ']') body = body[0 .. body.len - 1];

        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        var i: usize = 0;
        while (i < body.len) {
            while (i < body.len and (std.ascii.isWhitespace(body[i]) or body[i] == ',')) i += 1;
            const name_start = i;
            while (i < body.len and pythonicIsNameChar(body[i])) i += 1;
            const name = body[name_start..i];
            if (name.len == 0) break;

            while (i < body.len and std.ascii.isWhitespace(body[i])) i += 1;
            // No `(`, or no matching `)`: the cut landed inside the call. Ship
            // the NAME with EMPTY args — never a partial value (a half-written
            // file body is worse than a re-issued call).
            const args = blk: {
                if (i >= body.len or body[i] != '(') break :blk null;
                const arg_start = i + 1;
                const arg_end = pythonicScan(body, arg_start, pythonicIsCloser) orelse break :blk null;
                i = arg_end + 1;
                break :blk body[arg_start..arg_end];
            };

            var args_map: std.json.ObjectMap = .empty;
            if (args) |a| {
                var parts = PythonicParts{ .s = a };
                while (parts.next()) |part| {
                    const eq = pythonicScan(part, 0, pythonicIsEquals) orelse continue;
                    const key = std.mem.trim(u8, part[0..eq], " \t\n\r");
                    if (key.len == 0 or args_map.getEntry(key) != null) continue;
                    try args_map.put(arena, key, try pythonicLiteral(arena, part[eq + 1 ..]));
                }
            }

            const args_str = try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{});
            errdefer allocator.free(args_str);
            const name_owned = try allocator.dupe(u8, name);
            errdefer allocator.free(name_owned);
            try calls.append(allocator, .{ .name = name_owned, .arguments = args_str });

            if (args == null) break;
        }
    }
}

// ── tests ──

test "pythonic: the real LFM2.5 capture, with every literal type" {
    // Verbatim from mlx-community/LFM2.5-2.6B-8bit via /v1/completions
    // The model emits full Python repr — note `True` and the
    // SINGLE-quoted list, which is NOT the `tojson` form its own template
    // renders history with. A JSON-only value reader gets both wrong.
    const allocator = testing.allocator;
    const text = "...I should call the get_weather function.</think>" ++
        "<|tool_call_start|>[get_weather(city='Paris', days=3, metric=True, tags=['trip', 'eu'])]<|tool_call_end|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqualStrings("Paris", o.get("city").?.string);
    try testing.expectEqual(@as(i64, 3), o.get("days").?.integer);
    // A schema `boolean` must arrive as a real bool, not the string "True".
    try testing.expectEqual(true, o.get("metric").?.bool);
    // A schema `array` must arrive as a real array — the #94 defect in the
    // grammar where it is easiest to ship: the value is literal text.
    const tags = o.get("tags").?.array;
    try testing.expectEqual(@as(usize, 2), tags.items.len);
    try testing.expectEqualStrings("trip", tags.items[0].string);
    try testing.expectEqualStrings("eu", tags.items[1].string);
}

test "pythonic: several calls in one bracket list, and None/float/dict values" {
    const allocator = testing.allocator;
    const text = "<|tool_call_start|>[read_file(path='a.txt'), " ++
        "plot(scale=1.5, title=None, opts={'grid': True, 'dpi': 96})]<|tool_call_end|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("read_file", calls[0].name);
    try testing.expectEqualStrings("plot", calls[1].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqual(@as(f64, 1.5), o.get("scale").?.float);
    try testing.expectEqual(std.json.Value.null, o.get("title").?);
    try testing.expectEqual(true, o.get("opts").?.object.get("grid").?.bool);
    try testing.expectEqual(@as(i64, 96), o.get("opts").?.object.get("dpi").?.integer);
}

test "pythonic: separators inside a string value are not separators" {
    // `)` `,` `[` inside the quoted value must not end the arg or the call —
    // the delimiter-drop class in reverse (a naive scan cuts here).
    const allocator = testing.allocator;
    const text = "<|tool_call_start|>[shell(cmd='ls -la (tmp), [x]', dir='/a')]<|tool_call_end|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la (tmp), [x]", parsed.value.object.get("cmd").?.string);
    try testing.expectEqualStrings("/a", parsed.value.object.get("dir").?.string);
}

test "pythonic: escapes are unescaped, and a double-quoted string is accepted" {
    const allocator = testing.allocator;
    const text = "<|tool_call_start|>[write(body='line1\\nit\\'s \\\\ done', path=\"b.txt\")]<|tool_call_end|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("line1\nit's \\ done", parsed.value.object.get("body").?.string);
    try testing.expectEqualStrings("b.txt", parsed.value.object.get("path").?.string);
}

test "pythonic: a truncated call recovers the NAME with EMPTY args, never a partial value" {
    // max_tokens cut mid-argument. Shipping half a file body is worse than
    // re-issuing the call (the truncated-opener rule); the name is enough for
    // the client to see a tool_calls finish and retry.
    const allocator = testing.allocator;
    const text = "<|tool_call_start|>[write_file(path='a.html', content='<!DOCTYPE html>\\n<p>cut mid-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
}

test "pythonic: tolerates a dropped outer bracket and a dropped end marker" {
    const allocator = testing.allocator;
    // No `[`, no `<|tool_call_end|>` — one delimiter drop must not cost the call.
    const calls = (try parseToolCalls(allocator, "<|tool_call_start|>get_time(tz='UTC')")).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_time", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("UTC", parsed.value.object.get("tz").?.string);
}

test "pythonic: an unparseable value stays honest as a string, and keys dedup" {
    const allocator = testing.allocator;
    const text = "<|tool_call_start|>[run(mode=fast, mode=slow)]<|tool_call_end|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    // Bare identifier is not a Python literal — kept verbatim rather than
    // guessed at, so a client's own validation error stays honest. First wins.
    try testing.expectEqualStrings("fast", parsed.value.object.get("mode").?.string);
}

test "pythonic: the marker is required — prose that mentions a call is not one" {
    const allocator = testing.allocator;
    try testing.expect((try parseToolCalls(allocator, "Call get_weather(city='Paris') to do that.")) == null);
}
