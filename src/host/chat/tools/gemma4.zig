//! The Gemma 4 tool-call dialect.

const std = @import("std");
const log = @import("../../../sdk/util/log.zig");
const testing = std.testing;

const parseToolCalls = @import("parse.zig").parseToolCalls;
const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const looseRepairToolCallJson = @import("json_repair.zig").looseRepairToolCallJson;
const unwrapDoubleBraces = @import("json_repair.zig").unwrapDoubleBraces;
const sanitizeToolName = @import("json_repair.zig").sanitizeToolName;
const isJsonLiteral = @import("../../../sdk/util/json_text.zig").isJsonLiteral;
const appendJsonString = @import("../../../sdk/util/json_text.zig").appendJsonString;

/// Parse Gemma 4 tool call format: "call:function_name{json_args}"
/// `input_truncated`: the call had NO `<tool_call|>` close tag — the
/// generation was cut (EOS mid-call, max_tokens, or the scheduler's
/// degenerate-tail-loop guard). Under truncation, a value whose scan runs to
/// end-of-body without its terminator is a FRAGMENT and is dropped (the
/// Hermes-truncation rule: a half-written file is worse than a re-issued
/// write); with the close tag present, behavior is byte-identical to before.
pub fn parseGemma4ToolCall(allocator: std.mem.Allocator, content: []const u8, input_truncated: bool) ?ParsedToolCall {
    const prefix = "call:";
    if (!std.mem.startsWith(u8, content, prefix)) return null;
    const after_prefix = content[prefix.len..];

    // Find the opening brace
    const brace_pos = std.mem.indexOf(u8, after_prefix, "{") orelse return null;
    const name = sanitizeToolName(after_prefix[0..brace_pos]);
    if (name.len == 0) return null;

    var args_str = after_prefix[brace_pos..];

    // Gemma 4 uses {{ }} (double braces) for literal braces in Jinja templates.
    // The model often generates {{"key":"value"}} — unwrap the outer braces.
    args_str = unwrapDoubleBraces(args_str);

    // Try JSON first (model sometimes outputs valid JSON arguments)
    if (std.json.parseFromSlice(std.json.Value, allocator, args_str, .{})) |parsed| {
        defer parsed.deinit();
        if (parsed.value == .object) {
            return .{
                .name = allocator.dupe(u8, name) catch return null,
                .arguments = allocator.dupe(u8, args_str) catch return null,
            };
        }
    } else |_| {}

    // Strict JSON failed — but the model may have emitted standard JSON with
    // mangled escaping (raw newlines / unescaped quotes in big content) rather
    // than the custom <|"|> format. Recover that BEFORE the custom-format
    // converter, which would garble standard JSON. looseRepair returns null for
    // the bare-key `{key:<|"|>v<|"|>}` form, so the custom path still runs.
    if (looseRepairToolCallJson(allocator, args_str)) |repaired| {
        var keep = false;
        defer if (!keep) allocator.free(repaired);
        if (std.json.parseFromSlice(std.json.Value, allocator, repaired, .{})) |reparsed| {
            defer reparsed.deinit();
            if (reparsed.value == .object) {
                const name_owned = allocator.dupe(u8, name) catch return null;
                keep = true;
                log.info("  [tool-parse] loose-repair recovered mangled Gemma 4 tool-call JSON\n", .{});
                return .{ .name = name_owned, .arguments = repaired };
            }
        } else |_| {}
    }

    // Convert Gemma 4 custom format to JSON:
    // {key:<|"|>value<|"|>,key2:<|"|>value2<|"|>} → {"key":"value","key2":"value2"}
    const json = convertGemma4ArgsToJson(allocator, args_str, input_truncated) orelse {
        log.info("  [tool-parse] convertGemma4ArgsToJson FAILED for: {s}\n", .{args_str[0..@min(args_str.len, 200)]});
        return null;
    };
    return .{
        .name = allocator.dupe(u8, name) catch return null,
        .arguments = json,
    };
}

/// Convert Gemma 4's custom key-value format to JSON.
/// Input:  {key:<|"|>value<|"|>,nested:{k:<|"|>v<|"|>},arr:[<|"|>a<|"|>]}
/// Output: {"key":"value","nested":{"k":"v"},"arr":["a"]}
///
/// Gemma 4 emits its own object/array/string syntax (bare keys, `<|"|>…<|"|>`
/// string delimiters) that can nest arbitrarily, so every value is converted
/// recursively — see `convertGemma4Value`. Nested structures that already use
/// valid JSON (regular `"…"` strings, numbers, bools, null) pass through
/// unchanged. The args are always an object, with or without a literal `{…}`.
fn convertGemma4ArgsToJson(allocator: std.mem.Allocator, input: []const u8, input_truncated: bool) ?[]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);
    // `cut` = the truncation landed INSIDE a value somewhere below; the pair
    // holding it has already been rolled back (fragment values never ship).
    var cut = false;
    _ = convertGemma4Object(allocator, &result, input, 0, 0, input_truncated, &cut) orelse return null;
    return result.toOwnedSlice(allocator) catch return null;
}

const gemma4_str_delim = "<|\"|>";
/// Recursion-depth guard against adversarial deeply-nested model output.
const gemma4_max_depth = 64;

fn gemma4SkipWs(body: []const u8, start: usize) usize {
    var pos = start;
    while (pos < body.len) : (pos += 1) {
        switch (body[pos]) {
            ' ', '\t', '\n', '\r' => {},
            else => break,
        }
    }
    return pos;
}

fn gemma4SkipWsCommas(body: []const u8, start: usize) usize {
    var pos = start;
    while (pos < body.len) : (pos += 1) {
        switch (body[pos]) {
            ' ', '\t', '\n', '\r', ',' => {},
            else => break,
        }
    }
    return pos;
}

/// Parse a Gemma 4 object beginning at `start` (an optional leading `{` is
/// consumed; absent braces are tolerated so a brace-less body still parses).
/// Appends the JSON object to `result`; returns the index just past it.
fn convertGemma4Object(
    allocator: std.mem.Allocator,
    result: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    input_truncated: bool,
    cut: *bool,
) ?usize {
    if (depth >= gemma4_max_depth) return null;
    var pos = gemma4SkipWs(body, start);
    if (pos < body.len and body[pos] == '{') pos += 1;
    result.append(allocator, '{') catch return null;

    // Dedup keys within this object (a repeated key would make std.json reject
    // the args with error.DuplicateField — same class as parseHermesToolCall's
    // dup-param bug). Keys are slices into the stable input `body`.
    var seen = std.ArrayList([]const u8).empty;
    defer seen.deinit(allocator);

    var closed = false;
    var first = true;
    while (pos < body.len) {
        pos = gemma4SkipWsCommas(body, pos);
        if (pos >= body.len) break;
        if (body[pos] == '}') {
            pos += 1;
            closed = true;
            break;
        }

        // Key: everything up to the first ':'.
        const colon = std.mem.indexOfScalar(u8, body[pos..], ':') orelse break;
        const key_raw = std.mem.trim(u8, body[pos .. pos + colon], " \t\n\r");
        // Strip surrounding quotes if present (model sometimes quotes keys).
        const key = if (key_raw.len >= 2 and key_raw[0] == '"' and key_raw[key_raw.len - 1] == '"')
            key_raw[1 .. key_raw.len - 1]
        else
            key_raw;
        pos = pos + colon + 1;

        // Tentatively emit comma+key+value; roll back if this key is a dup. The
        // value is still consumed (pos advances) so the dup's payload is skipped.
        const mark = result.items.len;
        if (!first) result.append(allocator, ',') catch return null;
        // Escape the KEY — a raw `"`/control byte in the key would emit invalid
        // JSON (the Gemma converter's output is not strict-re-validated).
        appendJsonString(allocator, result, key) catch return null;
        result.append(allocator, ':') catch return null;

        pos = convertGemma4Value(allocator, result, body, pos, depth, input_truncated, cut) orelse return null;

        // The truncation landed INSIDE this value — it is a fragment (partial
        // file content, partial url) and must never ship as a real argument
        // (the Hermes-truncation rule, extended to the Gemma arm
        // php.html loop-stop capture). Roll the whole pair back; the scan
        // consumed to end-of-body, so this object is done.
        if (cut.*) {
            result.shrinkRetainingCapacity(mark); // undo comma+key+value; keep `first`
            break;
        }

        var is_dup = false;
        for (seen.items) |s| {
            if (std.mem.eql(u8, s, key)) {
                is_dup = true;
                break;
            }
        }
        if (is_dup) {
            result.shrinkRetainingCapacity(mark); // undo comma+key+value; keep `first`
            continue;
        }
        seen.append(allocator, key) catch return null;
        first = false;
    }

    // A NESTED object the cut landed inside (it never saw its `}`) is itself a
    // fragment — even when every pair inside completed, more were coming; the
    // parent rolls the whole container back. The ROOT object (depth 0) is
    // exempt: pairs completed before the cut are intact values and survive —
    // only the value holding the cut drops.
    if (input_truncated and depth > 0 and !closed) cut.* = true;

    result.append(allocator, '}') catch return null;
    return pos;
}

/// Parse a Gemma 4 array beginning at `start` (`body[start] == '['`). Appends
/// the JSON array to `result`; returns the index just past it.
fn convertGemma4Array(
    allocator: std.mem.Allocator,
    result: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    input_truncated: bool,
    cut: *bool,
) ?usize {
    if (depth >= gemma4_max_depth) return null;
    var pos = start + 1; // consume '['
    result.append(allocator, '[') catch return null;

    var closed = false;
    var first = true;
    while (pos < body.len) {
        pos = gemma4SkipWsCommas(body, pos);
        if (pos >= body.len) break;
        if (body[pos] == ']') {
            pos += 1;
            closed = true;
            break;
        }
        if (!first) result.append(allocator, ',') catch return null;
        first = false;
        pos = convertGemma4Value(allocator, result, body, pos, depth, input_truncated, cut) orelse return null;
        // Fragment below: stop consuming; the parent object rolls the whole
        // container pair back, so no local cleanup is needed.
        if (cut.*) break;
    }

    // Same fragment rule as nested objects: an unclosed array under truncation
    // is a partial LIST — more elements were coming (a partial edits[] applies
    // fragmentary work, the same hazard as partial file content).
    if (input_truncated and !closed) cut.* = true;

    result.append(allocator, ']') catch return null;
    return pos;
}

/// Convert one Gemma 4 value (custom string / JSON string / object / array /
/// bare literal) beginning at `start`. String branches are checked first so
/// braces inside a string stay string content. Returns the index past it.
fn convertGemma4Value(
    allocator: std.mem.Allocator,
    result: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    input_truncated: bool,
    cut: *bool,
) ?usize {
    var pos = gemma4SkipWs(body, start);
    if (pos >= body.len) {
        // `key:` then end-of-body. Under truncation the value is a fragment
        // that never arrived — drop the pair (pre-fix this shipped
        // {"path":""} and a client would act on the bogus empty value).
        if (input_truncated) {
            cut.* = true;
            return pos;
        }
        result.appendSlice(allocator, "\"\"") catch return null;
        return pos;
    }

    // Gemma custom string: <|"|>…<|"|> (closing delimiter may be missing if
    // the model output was truncated).
    if (pos + gemma4_str_delim.len <= body.len and
        std.mem.eql(u8, body[pos .. pos + gemma4_str_delim.len], gemma4_str_delim))
    {
        pos += gemma4_str_delim.len;
        const end_idx = std.mem.indexOf(u8, body[pos..], gemma4_str_delim);
        const value = if (end_idx) |e| body[pos .. pos + e] else blk: {
            // Closing <|"|> missing.
            if (input_truncated) {
                // The call itself was cut (no <tool_call|>): the truncation
                // landed INSIDE this string — a fragment (partial file
                // content, partial url) that must never ship as a real
                // argument (live php.html: the loop-stop guard cut
                // a write call mid-`content` and the 1.1 KB fragment shipped).
                cut.* = true;
                return body.len;
            }
            // Complete call (close tag present), delimiter just dropped: a
            // plain to-end-of-body scan swallows the args object's own `}`
            // and any stray fence garbage the model tacked on — a real write
            // call reached disk as "mlx_pi1.html`}". Trim one trailing `}`
            // (the enclosing object's closer) plus backtick/whitespace junk.
            var v = std.mem.trimEnd(u8, body[pos..], " \t\n\r");
            if (std.mem.endsWith(u8, v, "}")) v = v[0 .. v.len - 1];
            v = std.mem.trimEnd(u8, v, "` \t\n\r");
            break :blk v;
        };
        pos = if (end_idx) |e|
            pos + e + gemma4_str_delim.len
        else
            body.len;
        appendJsonString(allocator, result, value) catch return null;
        return pos;
    }

    // Regular JSON string: already valid, copy verbatim (respecting \" escapes).
    if (body[pos] == '"') {
        var end = pos + 1;
        var closed_quote = false;
        while (end < body.len) : (end += 1) {
            if (body[end] == '\\' and end + 1 < body.len) {
                end += 1;
                continue;
            }
            if (body[end] == '"') {
                closed_quote = true;
                end += 1;
                break;
            }
        }
        // Unclosed JSON string at end-of-body under truncation: same fragment
        // rule — and copying it verbatim would emit an UNCLOSED string, i.e.
        // invalid JSON that only the {}-fallback safety net could catch.
        if (!closed_quote and input_truncated) {
            cut.* = true;
            return body.len;
        }
        result.appendSlice(allocator, body[pos..end]) catch return null;
        return end;
    }

    if (body[pos] == '{') return convertGemma4Object(allocator, result, body, pos, depth + 1, input_truncated, cut);
    if (body[pos] == '[') return convertGemma4Array(allocator, result, body, pos, depth + 1, input_truncated, cut);

    // Bare value — a JSON literal (number/bool/null) terminates at the
    // enclosing separator and is emitted verbatim.
    const sep_rel = std.mem.indexOfAny(u8, body[pos..], ",}]");
    const first_sep = sep_rel orelse (body.len - pos);
    const head = std.mem.trim(u8, body[pos .. pos + first_sep], " \t\n\r");
    if (isJsonLiteral(head)) {
        // A literal that ran to end-of-body under truncation has no
        // terminator either — uniform fragment rule (a cut `5` may have been
        // `50`; the client re-asks and gets the real value).
        if (sep_rel == null and input_truncated) {
            cut.* = true;
            return body.len;
        }
        result.appendSlice(allocator, head) catch return null;
        return pos + first_sep;
    }

    // Unquoted STRING. The bare scan above stops at the first `,`/`}`/`]`, which
    // truncates rich content (HTML/markdown) that legitimately contains those
    // bytes. Observed live on gemma-4-e4b-it writing a full page: it dropped the
    // OPENING <|"|> on `content` but kept the CLOSING one
    // (`content:<!DOCTYPE…>…</html><|"|>,path:…`), so content got cut at the
    // viewport meta's comma and the rest became bogus keys → invalid args. Only
    // rich content (multi-line or markup) gets the wider scan; a plain short
    // bare token keeps the first-separator behavior so it can't swallow a
    // sibling field (e.g. `command:ls -la`).
    const rich = std.mem.indexOfScalar(u8, head, '\n') != null or
        std.mem.indexOfScalar(u8, head, '<') != null;
    if (rich) {
        // Prefer a CLOSING <|"|> as the boundary (dropped-opener case). Confirm
        // it's a closer — the byte after it (past whitespace) must be a
        // separator/closer — so a LATER field's OPENING delimiter isn't grabbed.
        if (std.mem.indexOf(u8, body[pos..], gemma4_str_delim)) |close_rel| {
            const after = gemma4SkipWs(body, pos + close_rel + gemma4_str_delim.len);
            const is_closer = after >= body.len or body[after] == ',' or body[after] == '}' or body[after] == ']';
            if (is_closer) {
                const value = std.mem.trim(u8, body[pos .. pos + close_rel], " \t\n\r");
                appendJsonString(allocator, result, value) catch return null;
                return pos + close_rel + gemma4_str_delim.len;
            }
        }
        // No usable closing delimiter at all. Under truncation the rest of the
        // body IS the fragment: consuming to the first comma would ship a
        // partial value AND shred the remainder into bogus keys, and the
        // last-`}` heuristic below would cut a truncated page at some interior
        // CSS brace — drop the whole value instead.
        if (input_truncated) {
            cut.* = true;
            return body.len;
        }
        // Complete call, both delimiters dropped. At the TOP level the value
        // runs to the object's closing `}` (the last one); nested values keep
        // the narrow scan since the outer `}` isn't theirs.
        if (depth == 0) {
            if (std.mem.lastIndexOfScalar(u8, body, '}')) |last_brace| {
                if (last_brace > pos) {
                    const value = std.mem.trim(u8, body[pos..last_brace], " \t\n\r");
                    appendJsonString(allocator, result, value) catch return null;
                    return last_brace;
                }
            }
        }
    }

    // A short bare string that ran to end-of-body under truncation is a
    // fragment too (`command:ls -la` cut mid-flags) — uniform rule.
    if (sep_rel == null and input_truncated) {
        cut.* = true;
        return body.len;
    }

    // Plain short bare string — terminate at the first separator (unchanged).
    appendJsonString(allocator, result, head) catch return null;
    return pos + first_sep;
}

// ── tests ──

test "parseToolCalls recovers Gemma 4 call:NAME{json} with raw newline content" {
    const allocator = testing.allocator;
    // Gemma 4's JSON-first branch hits the same strict-parse wall on raw bytes.
    const text = "<|tool_call>call:writeFile{\"path\":\"g.txt\",\"content\":\"line1\nline2\"}<tool_call|>";
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
    try testing.expectEqualStrings("line1\nline2", parsed.value.object.get("content").?.string);
}

test "parseToolCalls Gemma 4 format" {
    const allocator = testing.allocator;
    const text = "<|tool_call>call:get_weather{\"location\": \"Tokyo\"}<tool_call|>";
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
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "parseToolCalls Gemma 4 with channel thinking" {
    const allocator = testing.allocator;
    const text = "<|channel>thought\nLet me check the weather<channel|>\n<|tool_call>call:get_weather{\"city\": \"Paris\"}<tool_call|>";
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
}

test "parseToolCalls Gemma 4 truncated (no closing tag)" {
    const allocator = testing.allocator;
    // Model hit EOS before generating <tool_call|>
    const text = "<|tool_call>call:browse{action:<|\"|>browse<|\"|>,url:<|\"|>https://finance.yahoo.com<|\"|>}";
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
    // Should have parsed at least the action argument
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("browse", parsed.value.object.get("action").?.string);
}

test "parseToolCalls Gemma 4 truncated mid-value drops the partial value" {
    const allocator = testing.allocator;
    // Model stopped mid-URL: no closing <|"|>, no `}`, no <tool_call|>. POLICY
    // (aligned with the Hermes truncation rule): a value whose scan
    // runs to end-of-body without its terminator is a truncation FRAGMENT —
    // partial bytes must never ship as a real argument (a client executes them:
    // a fragmentary url browses garbage, a fragmentary content writes a corrupt
    // file "successfully"). The completed pair before it survives, and the tool
    // NAME is always recovered so the client's steering fires.
    const text = "<|tool_call>call:browse{action:<|\"|>navigate<|\"|>,url:<|\"|>https://finance.";
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
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("navigate", parsed.value.object.get("action").?.string);
    // The truncated URL must be DROPPED, never shipped as partial bytes.
    try testing.expect(parsed.value.object.get("url") == null);
}

test "parseToolCalls Gemma 4 loop-stop truncated content dropped, name recovered (php.html capture)" {
    const allocator = testing.allocator;
    // Live (pi → gemma-4-26B-A4B-it-qat-4bit, plang/php.html): the
    // model repetition-looped INSIDE the write call's content string and the
    // scheduler's degenerate-tail-loop guard cut generation mid-word — the
    // buffered call arrived with NO closing delimiter, NO `}`, NO <tool_call|>,
    // and `path` never emitted at all. Pre-fix the salvage shipped
    // {"content":"<1.1 KB of loop garbage>"}; pi rejected the call on the
    // missing path and echoed the garbage back into the model's context.
    const text = "<|tool_call>call:write{content:<|\"|><!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <title>PHP</title>\n</head>\n<body>\n    <p>PHP is a widely-used general-purpose scripting language. It is a server-side scripting language, server-side scripting language, server-side scripting language, server-";
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
    try testing.expect(parsed.value == .object);
    try testing.expect(parsed.value.object.get("content") == null);
    try testing.expect(parsed.value.object.get("path") == null);
}

test "parseToolCalls Gemma 4 truncated content after a complete path keeps path, drops fragment" {
    const allocator = testing.allocator;
    // The ordering that would CORRUPT a file: `path` completed BEFORE the cut.
    // Pre-fix salvage = {path, <partial garbage content>} — a schema-valid call
    // the client executes, writing a fragment to a real path and reporting
    // success. The completed path survives (the client's missing-content error
    // steers a clean re-issue); the fragment never ships.
    const text = "<|tool_call>call:write{path:<|\"|>plang/php.html<|\"|>,content:<|\"|><!DOCTYPE html>\n<p>PHP is a server-side scripting language, server-side scripting language, server-";
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
    try testing.expectEqualStrings("plang/php.html", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls Gemma 4 bare rich value truncated at end-of-body is dropped" {
    const allocator = testing.allocator;
    // Both <|"|> delimiters dropped AND the generation truncated (no `}`, no
    // close tag anywhere in the rest): the run-to-end scan must not ship the
    // fragment as content.
    const text = "<|tool_call>call:write{content:<!DOCTYPE html>\n<h1>Hi</h1>\n<p>cut mid-";
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
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls Gemma 4 truncated edits container dropped, complete path kept" {
    const allocator = testing.allocator;
    // Cut INSIDE a container arg (pi-style edits array): the list is a fragment
    // — more edits were coming — and applying a partial edit list is the same
    // fragmentary-work hazard as partial content. The whole container drops;
    // the completed scalar before it survives.
    const text = "<|tool_call>call:edit{path:<|\"|>a.sh<|\"|>,edits:[{oldText:<|\"|>x<|\"|>,newText:<|\"|>y<|\"|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("edit", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a.sh", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("edits") == null);
}

test "parseToolCalls Gemma 4 EOS after a complete pair keeps the pair (no over-drop)" {
    const allocator = testing.allocator;
    // Truncation landing BETWEEN pairs (after a completed value + comma) must
    // not over-drop: only the value the cut landed INSIDE is a fragment.
    const text = "<|tool_call>call:write{path:<|\"|>a.html<|\"|>,";
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
    try testing.expectEqualStrings("a.html", parsed.value.object.get("path").?.string);
}

test "parseToolCalls Gemma 4 EOS right after the key colon drops the key" {
    const allocator = testing.allocator;
    // `path:` then nothing — pre-fix this shipped {"path":""} and a client
    // would create a file with an EMPTY name (or reject on a bogus value).
    // An absent value is a fragment; drop the key, recover the name.
    const text = "<|tool_call>call:write{path:";
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
    try testing.expect(parsed.value.object.get("path") == null);
}

test "parseToolCalls Gemma 4 JSON-style truncated string value is dropped" {
    const allocator = testing.allocator;
    // Same truncation class through the JSON-quoted spelling: the closing `"`
    // never arrived. Pre-fix the verbatim copy emitted an UNCLOSED JSON string
    // (invalid args, caught only by the {}-fallback safety net); the complete
    // first pair must survive and the fragment must drop.
    const text = "<|tool_call>call:write{\"path\":\"plang/php.html\",\"content\":\"<!DOCTYPE html>\n<p>cut mid-";
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
    try testing.expectEqualStrings("plang/php.html", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls Gemma 4 unterminated string must not swallow the closing brace (pi write regression)" {
    const allocator = testing.allocator;
    // Gemma 4 12B emitted a write call whose LAST string value was missing
    // its closing <|"|> delimiter and carried a stray markdown backtick
    // before the args object's `}`. The unterminated-string scan ran to end
    // of body, so the parsed path was literally "mlx_pi1.html`}" — and pi
    // created a file with that name on disk.
    const text = "<|tool_call>call:write{content:<|\"|><!DOCTYPE html><html></html><|\"|>,path:<|\"|>mlx_pi1.html`}<tool_call|>";
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
    try testing.expectEqualStrings("mlx_pi1.html", parsed.value.object.get("path").?.string);

    // Plain missing-delimiter variant (no backtick): same brace exclusion.
    const text2 = "<|tool_call>call:write{content:<|\"|>X<|\"|>,path:<|\"|>out.html}<tool_call|>";
    const calls2 = (try parseToolCalls(allocator, text2)).?;
    defer {
        for (calls2) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls2);
    }
    const parsed2 = try std.json.parseFromSlice(std.json.Value, allocator, calls2[0].arguments, .{});
    defer parsed2.deinit();
    try testing.expectEqualStrings("out.html", parsed2.value.object.get("path").?.string);
}

test "parseToolCalls Gemma 4 quoted keys with custom delimiters" {
    const allocator = testing.allocator;
    // Model mixes JSON-style quoted keys with <|"|> delimiters — JSON parse fails,
    // convertGemma4ArgsToJson must strip quotes from keys to produce valid JSON.
    const text =
        \\<|tool_call>call:shell{"command":<|"|>ls -la<|"|>}<tool_call|>
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
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls Gemma 4 quoted keys with bare values" {
    const allocator = testing.allocator;
    // Model uses quoted keys but bare (non-JSON, non-delimited) values
    const text = "<|tool_call>call:shell{\"command\":ls -la}<tool_call|>";
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

test "parseToolCalls Gemma 4 dropped OPENING delimiter on big content (E4B live)" {
    const allocator = testing.allocator;
    // Verbatim shape captured from gemma-4-e4b-it-4bit writing a full HTML page
    // (test_tool_matrix_small.sh): the model dropped the OPENING <|"|> on
    // `content` but kept the CLOSING one, so the bare-value scan cut content at
    // the FIRST comma (inside `<meta ... content="width=device-width, ...">`)
    // and shredded the rest of the markup into bogus keys → invalid args. The
    // closing <|"|> (followed by `,path`) is the real boundary.
    const text = "<|tool_call>call:write_file{content:<!DOCTYPE html>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n<style>body{margin:0}</style>\n</html><|\"|>,path:<|\"|>mars.html<|\"|>}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mars.html", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings(
        "<!DOCTYPE html>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n<style>body{margin:0}</style>\n</html>",
        parsed.value.object.get("content").?.string,
    );
}

test "parseToolCalls Gemma 4 dropped BOTH delimiters on single content field" {
    const allocator = testing.allocator;
    // Both delimiters dropped, content is the only/last field → run to the
    // object's closing brace.
    const text = "<|tool_call>call:write_file{content:<h1>Hi</h1>\n<p>a, b, c</p>}<tool_call|>";
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
    try testing.expectEqualStrings("<h1>Hi</h1>\n<p>a, b, c</p>", parsed.value.object.get("content").?.string);
}

test "convertGemma4ArgsToJson nested braces in value" {
    const allocator = testing.allocator;
    // Content value contains JSON-like structures (e.g., JavaScript code with objects)
    const input = "{path:<|\"|>server.js<|\"|>,content:<|\"|>const x = {a: 1, b: {c: 2}};<|\"|>}";
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    // Verify it produces valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("server.js", parsed.value.object.get("path").?.string);
    const content = parsed.value.object.get("content").?.string;
    try testing.expect(std.mem.indexOf(u8, content, "{a: 1") != null);
}

test "convertGemma4ArgsToJson bare array value" {
    const allocator = testing.allocator;
    // Bare array value should be preserved via brace-matching
    const input = "{stops:[\"Rome\",\"Venice\",\"Athens\"],price:1200}";
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 3), parsed.value.object.get("stops").?.array.items.len);
    try testing.expectEqual(@as(i64, 1200), parsed.value.object.get("price").?.integer);
}

test "convertGemma4ArgsToJson bare nested object value" {
    const allocator = testing.allocator;
    // Bare nested object should be preserved via brace-matching
    const input = "{name:test,config:{\"port\":3000,\"host\":\"localhost\"}}";
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    const config = parsed.value.object.get("config").?.object;
    try testing.expectEqual(@as(i64, 3000), config.get("port").?.integer);
}

// Issue #16: nested objects/arrays that themselves use Gemma 4's custom
// `<|"|>`-delimited string format (or bare keys) must be converted recursively,
// not passed through verbatim.
test "convertGemma4ArgsToJson issue#16 nested array of custom strings" {
    const allocator = testing.allocator;
    const input =
        \\{nested_array:[<|"|>foo<|"|>,<|"|>bar<|"|>]}
    ;
    const expected =
        \\{"nested_array":["foo","bar"]}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

test "convertGemma4ArgsToJson issue#16 nested object custom format" {
    const allocator = testing.allocator;
    const input =
        \\{nested_object:{foo:<|"|>bar<|"|>}}
    ;
    const expected =
        \\{"nested_object":{"foo":"bar"}}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

test "convertGemma4ArgsToJson issue#16 nested object inside array" {
    const allocator = testing.allocator;
    const input =
        \\{nested_object_in_array:[{foo:<|"|>bar<|"|>}]}
    ;
    const expected =
        \\{"nested_object_in_array":[{"foo":"bar"}]}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

test "convertGemma4ArgsToJson issue#16 nested array inside object" {
    const allocator = testing.allocator;
    const input =
        \\{nested_array_in_object:{foo:[<|"|>bar<|"|>]}}
    ;
    const expected =
        \\{"nested_array_in_object":{"foo":["bar"]}}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

// Issue #16, real capture: this is the verbatim output of gemma-4-e4b-it-8bit
// for a tool with nested-object + nested-array params. The model nests its
// custom `<|"|>`/bare-key format, so the client must still receive valid,
// fully-converted JSON arguments (no surviving <|"|> delimiters / bare keys).
test "parseToolCalls Gemma 4 nested object + array args (issue#16 real capture)" {
    const allocator = testing.allocator;
    const text =
        \\<|tool_call>call:send_notification{message:<|"|>Deploy complete<|"|>,metadata:{priority:<|"|>high<|"|>,tags:[<|"|>ci<|"|>,<|"|>release<|"|>]},recipients:[<|"|>alice<|"|>,<|"|>bob<|"|>]}<tool_call|>
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
    try testing.expectEqualStrings("send_notification", calls[0].name);
    // The arguments must be valid JSON (would fail to parse with raw <|"|> in them).
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Deploy complete", parsed.value.object.get("message").?.string);
    const metadata = parsed.value.object.get("metadata").?.object;
    try testing.expectEqualStrings("high", metadata.get("priority").?.string);
    const tags = metadata.get("tags").?.array;
    try testing.expectEqual(@as(usize, 2), tags.items.len);
    try testing.expectEqualStrings("ci", tags.items[0].string);
    try testing.expectEqualStrings("release", tags.items[1].string);
    const recipients = parsed.value.object.get("recipients").?.array;
    try testing.expectEqual(@as(usize, 2), recipients.items.len);
    try testing.expectEqualStrings("alice", recipients.items[0].string);
    try testing.expectEqualStrings("bob", recipients.items[1].string);
}

// Gemma 4 12B leaks a trailing ':' into the function name (`call:shell:{…}`),
// producing an unresolvable tool name `shell:`. The parser must strip it so the
// client resolves a real tool instead of looping on "Unknown tool 'shell:'".
test "parseToolCalls Gemma 4 strips trailing colon from tool name" {
    const allocator = testing.allocator;
    const text =
        \\<|tool_call>call:shell:{"command":<|"|>ls<|"|>}<tool_call|>
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
    try testing.expectEqualStrings("ls", parsed.value.object.get("command").?.string);
}

test "convertGemma4 dedups a repeated key + escapes a key with special chars (valid JSON)" {
    // Same class as the parseHermesToolCall dup-key / raw-name bugs, but in the
    // Gemma `call:name{...}` converter: a repeated key produced a DUPLICATE JSON
    // key (std.json → error.DuplicateField), and a key with a `"` was appended
    // raw → invalid JSON. Both must yield a valid JSON object.
    const allocator = testing.allocator;
    // Duplicate key.
    {
        const raw = "<|tool_call>call:shell{command:<|\"|>ls<|\"|>,command:<|\"|>pwd<|\"|>}<tool_call|>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expect(parsed.value == .object);
        try testing.expect(parsed.value.object.get("command") != null);
    }
    // Key containing a double-quote must not break the JSON.
    {
        const raw = "<|tool_call>call:shell{a\"b:<|\"|>v<|\"|>}<tool_call|>";
        const calls = try parseToolCalls(allocator, raw);
        defer if (calls) |cs| {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        };
        if (calls) |cs| {
            const parsed = try std.json.parseFromSlice(std.json.Value, allocator, cs[0].arguments, .{});
            defer parsed.deinit();
            try testing.expect(parsed.value == .object);
        }
    }
}

test "parseToolCalls: NO path emits invalid JSON args (final safety net)" {
    // Bullet-proof invariant: whatever a converter builds, the arguments a client
    // receives are ALWAYS valid JSON. convertGemma4Value copies a JSON-style
    // string verbatim, so a bad escape (`\q`) would otherwise emit invalid JSON.
    const allocator = testing.allocator;
    const cases = [_][]const u8{
        // Gemma regular-JSON-string value with an INVALID escape.
        "<|tool_call>call:write{path:\"a\\qb\"}<tool_call|>",
        // Gemma custom-string is fine, but mix a bad JSON string sibling.
        "<|tool_call>call:write{path:<|\"|>ok<|\"|>,x:\"y\\z\"}<tool_call|>",
    };
    for (cases) |text| {
        const calls = try parseToolCalls(allocator, text);
        defer if (calls) |cs| {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        };
        if (calls) |cs| {
            for (cs) |tc| {
                const parsed = std.json.parseFromSlice(std.json.Value, allocator, tc.arguments, .{}) catch {
                    std.debug.print("\nINVALID JSON args emitted: {s}\n", .{tc.arguments});
                    return error.InvalidJsonEmitted;
                };
                parsed.deinit();
            }
        }
    }
    // The `\q` case is REPAIRABLE (looseRepair doubles the bad backslash), so the
    // path survives as the literal `a\qb` — the safety net repairs, only falling
    // back to `{}` when even tolerant repair fails.
    {
        const calls = (try parseToolCalls(allocator, "<|tool_call>call:write{path:\"a\\qb\"}<tool_call|>")).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings("a\\qb", parsed.value.object.get("path").?.string);
    }
}
