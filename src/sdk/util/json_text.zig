//! JSON literals, UTF-8 sanitizing, and `appendJsonString`.

const std = @import("std");
const testing = std.testing;

/// A parameter name from a well-formed `<parameter=NAME>` tag is a short token
/// with no whitespace or markup. Reject names carrying a newline, angle bracket,
/// quote, or whitespace — those signal the `>`-scan spilled past a malformed tag.
pub fn isPlausibleParamName(name: []const u8) bool {
    if (name.len == 0 or name.len > 128) return false;
    for (name) |c| {
        switch (c) {
            '\n', '\r', '\t', ' ', '<', '>', '"' => return false,
            else => {},
        }
    }
    return true;
}

pub fn isJsonLiteral(s: []const u8) bool {
    if (s.len == 0) return false;
    if (std.mem.eql(u8, s, "true") or std.mem.eql(u8, s, "false") or std.mem.eql(u8, s, "null")) return true;
    // Objects/arrays are handled by brace-matching in the caller, not here
    return isJsonNumber(s);
}

/// JSON's number grammar; `parseFloat` is wider (`Infinity`, `0755`, `.5`) and the literal is spliced UNQUOTED.
pub fn isJsonNumber(s: []const u8) bool {
    var i: usize = 0;
    if (i < s.len and s[i] == '-') i += 1;
    if (i >= s.len or !std.ascii.isDigit(s[i])) return false;
    if (s[i] == '0') {
        i += 1;
    } else {
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    }
    if (i < s.len and s[i] == '.') {
        i += 1;
        if (i >= s.len or !std.ascii.isDigit(s[i])) return false;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    }
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        i += 1;
        if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
        if (i >= s.len or !std.ascii.isDigit(s[i])) return false;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    }
    return i == s.len;
}

/// One UTF-8 sequence starting at `i`: `valid` = well-formed, else `end` is the maximal
/// subpart to replace with U+FFFD (a character split across two tokens costs ONE replacement).
pub const Utf8Seq = struct { end: usize, valid: bool };

pub fn utf8Next(s: []const u8, i: usize) Utf8Seq {
    const len = std.unicode.utf8ByteSequenceLength(s[i]) catch return .{ .end = i + 1, .valid = false };
    if (i + len <= s.len and std.unicode.utf8ValidateSlice(s[i .. i + len])) return .{ .end = i + len, .valid = true };
    var j = i + 1;
    while (j < s.len and j < i + len and s[j] & 0xC0 == 0x80) j += 1;
    return .{ .end = j, .valid = false };
}

/// The lossy view of bytes that are not valid UTF-8: one U+FFFD per invalid sequence.
/// A token is a BPE fragment, so ANY string built from model bytes can end mid-character;
/// raw, that makes a response body unparseable and a prompt render fall back silently.
pub fn utf8Sanitize(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var clean = std.ArrayList(u8).empty;
    errdefer clean.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        const seq = utf8Next(input, i);
        try clean.appendSlice(allocator, if (seq.valid) input[i..seq.end] else "\u{FFFD}");
        i = seq.end;
    }
    return clean.toOwnedSlice(allocator);
}

/// THE JSON string escaper (quoted, control bytes \u-escaped). Public so other
/// request builders reuse it instead of hand-rolling a second one — a duplicate
/// escaper is exactly how the control-byte class shipped twice before.
pub fn appendJsonString(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), s: []const u8) !void {
    try buf.append(allocator, '"');
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c >= 0x80) {
            // nlohmann rejects ill-formed UTF-8 like a raw control byte.
            const seq = utf8Next(s, i);
            try buf.appendSlice(allocator, if (seq.valid) s[i..seq.end] else "\u{FFFD}");
            i = seq.end;
            continue;
        }
        i += 1;
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            // Every other control char (e.g. ESC from ANSI codes in tool
            // results) must be \u-escaped — nlohmann inside jinja_render_chat
            // rejects raw control bytes, and the render failure silently
            // downgrades the prompt to fallbackFormatChat.
            0...8, 0x0B, 0x0C, 0x0E...0x1F => {
                var esc: [6]u8 = undefined;
                const n = std.fmt.bufPrint(&esc, "\\u{x:0>4}", .{c}) catch unreachable;
                try buf.appendSlice(allocator, n);
            },
            else => try buf.append(allocator, c),
        }
    }
    try buf.append(allocator, '"');
}

// ── tests ──

test "isJsonLiteral" {
    try testing.expect(isJsonLiteral("true"));
    try testing.expect(isJsonLiteral("false"));
    try testing.expect(isJsonLiteral("null"));
    // Objects/arrays are now handled by brace-matching in convertGemma4ArgsToJson, not isJsonLiteral
    try testing.expect(!isJsonLiteral("{\"key\":1}"));
    try testing.expect(!isJsonLiteral("[1,2,3]"));
    try testing.expect(isJsonLiteral("42"));
    try testing.expect(isJsonLiteral("3.14"));
    try testing.expect(!isJsonLiteral("hello"));
    try testing.expect(!isJsonLiteral(""));
}

test "appendJsonString escapes special chars" {
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    try appendJsonString(allocator, &buf, "hello \"world\"\nnew\\line");
    try testing.expectEqualStrings("\"hello \\\"world\\\"\\nnew\\\\line\"", buf.items);
}

test "appendJsonString empty string" {
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    try appendJsonString(allocator, &buf, "");
    try testing.expectEqualStrings("\"\"", buf.items);
}

test "appendJsonString escapes ALL control characters (2026-06-11 ESC-byte regression)" {
    // Live failure: a tool result carrying raw ANSI terminal codes (ESC = 0x1B
    // from `\x1b[?25l`) passed through unescaped, nlohmann::json rejected the
    // messages JSON inside jinja_render_chat, and the prompt silently fell
    // back to the wrong-format fallbackFormatChat — gemma-4-31b then
    // hallucinated entire conversations. JSON strings must escape EVERY
    // control char < 0x20, not just \n \r \t.
    const allocator = testing.allocator;

    var input: [0x20]u8 = undefined;
    for (&input, 0..) |*c, i| c.* = @intCast(i);

    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try appendJsonString(allocator, &buf, &input);

    // No raw control byte may survive in the serialized form.
    for (buf.items) |c| {
        try testing.expect(c >= 0x20);
    }

    // And the result must round-trip through a strict JSON parser byte-exact.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf.items, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings(&input, parsed.value.string);
}

test "appendJsonString: invalid UTF-8 becomes U+FFFD, valid multibyte is byte-exact" {
    // Bar: the escaper's output is always well-formed UTF-8 (nlohmann rejects
    // anything else), and every valid sequence survives unchanged.
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try appendJsonString(allocator, &buf, "\xff\xfe\x80 binary");
    try testing.expect(std.unicode.utf8ValidateSlice(buf.items));
    try testing.expectEqualStrings("\"\u{FFFD}\u{FFFD}\u{FFFD} binary\"", buf.items);

    buf.clearRetainingCapacity();
    const valid = "héllo \u{4e2d}\u{6587} \u{1F600} ok";
    try appendJsonString(allocator, &buf, valid);
    try testing.expectEqualStrings("\"" ++ valid ++ "\"", buf.items);

    // A truncated multibyte sequence at end-of-input is one replacement.
    buf.clearRetainingCapacity();
    try appendJsonString(allocator, &buf, "a\xe4\xb8");
    try testing.expectEqualStrings("\"a\u{FFFD}\"", buf.items);
}
