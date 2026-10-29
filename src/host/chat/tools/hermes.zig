//! The Hermes tool-call dialect (`<function=…>` / `<parameter=…>`).

const std = @import("std");
const testing = std.testing;

const parseToolCalls = @import("parse.zig").parseToolCalls;
const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const isPlausibleParamName = @import("../../../sdk/util/json_text.zig").isPlausibleParamName;
const isJsonLiteral = @import("../../../sdk/util/json_text.zig").isJsonLiteral;
const appendJsonString = @import("../../../sdk/util/json_text.zig").appendJsonString;
const freeParsedCalls = @import("dsml.zig").freeParsedCalls;

/// End of a `<parameter=…>` value: the LAST `</parameter>` before the next opener (a value may
/// spell its own close tag); an unclosed parameter falls back to the first close.
fn hermesValueEnd(body: []const u8, val_start: usize) ?usize {
    const bound = std.mem.indexOfPos(u8, body, val_start, "<parameter=") orelse body.len;
    var best: ?usize = null;
    var i = val_start;
    while (std.mem.indexOfPos(u8, body, i, "</parameter>")) |at| {
        if (at >= bound) break;
        best = at;
        i = at + 1;
    }
    return best orelse std.mem.indexOfPos(u8, body, val_start, "</parameter>");
}

/// If `at` falls inside a `<parameter=…>` value, the position just past that value's close; else null.
pub fn hermesParamSpanEnclosing(body: []const u8, at: usize) ?usize {
    const open = std.mem.lastIndexOf(u8, body[0..at], "<parameter=") orelse return null;
    const gt = std.mem.indexOfScalarPos(u8, body, open, '>') orelse return null;
    if (gt >= at) return null;
    const end = hermesValueEnd(body, gt + 1) orelse return null;
    if (end <= at) return null;
    return end + "</parameter>".len;
}

/// A `<tool_call>` body whose `<function=` opener lost its `<` to a sampled special token
/// (`<|im_start|>function=read_file>`, live on Qwen3.8-27B under MTP): one mangled delimiter,
/// name and arguments intact. Returns the body with that special spelled as `<`, null when the
/// body already has a `<function=` or no `<|…|>function=` run; the caller owns the copy.
pub fn hermesSpecialOpenerRepair(allocator: std.mem.Allocator, body: []const u8) !?[]u8 {
    if (std.mem.indexOf(u8, body, "<function=") != null) return null;
    const fn_at = std.mem.indexOf(u8, body, "function=") orelse return null;
    if (fn_at < 4 or !std.mem.endsWith(u8, body[0..fn_at], "|>")) return null;
    const open = std.mem.lastIndexOf(u8, body[0..fn_at], "<|") orelse return null;
    if (std.mem.indexOfAny(u8, body[open + 2 .. fn_at - 2], " \t\r\n<>") != null) return null;
    const out = try allocator.alloc(u8, body.len - (fn_at - open) + 1);
    @memcpy(out[0..open], body[0..open]);
    out[open] = '<';
    @memcpy(out[open + 1 ..], body[fn_at..]);
    return out;
}

pub fn parseHermesToolCall(allocator: std.mem.Allocator, block: []const u8) ?ParsedToolCall {
    const fn_start_tag = "<function=";
    const fn_start = std.mem.indexOf(u8, block, fn_start_tag) orelse return null;
    var name_start = fn_start + fn_start_tag.len;
    while (name_start < block.len and std.ascii.isWhitespace(block[name_start])) name_start += 1;
    // A name never spans a line or a tag: a newline or `<` before the `>` means
    // the model dropped the `>`, so the name ends there and the body starts there.
    const name_end = std.mem.indexOfAnyPos(u8, block, name_start, ">\n\r<") orelse return null;
    const fn_name = std.mem.trim(u8, block[name_start..name_end], " \t");
    if (fn_name.len == 0) return null;

    var args_map = std.ArrayList(u8).empty;
    defer args_map.deinit(allocator);
    args_map.append(allocator, '{') catch return null;

    const fn_body_start = if (block[name_end] == '>') name_end + 1 else name_end;
    const fn_end = blk: {
        var from = fn_body_start;
        while (std.mem.indexOfPos(u8, block, from, "</function>")) |found| {
            if (hermesParamSpanEnclosing(block, found)) |past| {
                from = past;
                continue;
            }
            break :blk found - fn_body_start;
        }
        break :blk block.len - fn_body_start;
    };
    const fn_body = block[fn_body_start .. fn_body_start + fn_end];

    // Track emitted parameter names so a repeated `<parameter=NAME>` can't
    // produce a duplicate JSON key (std.json rejects those with DuplicateField —
    // invalid JSON to the client; live soak record 443, a 0.8B model emitting two
    // `<parameter=edits>` blocks). First occurrence wins.
    var seen_names = std.ArrayList([]const u8).empty;
    defer seen_names.deinit(allocator);

    var param_search: usize = 0;
    var first_param = true;
    while (std.mem.indexOf(u8, fn_body[param_search..], "<parameter=")) |ps| {
        const p_name_start = param_search + ps + "<parameter=".len;
        const p_name_end = std.mem.indexOf(u8, fn_body[p_name_start..], ">") orelse break;
        const p_name = std.mem.trim(u8, fn_body[p_name_start .. p_name_start + p_name_end], " \n");

        // A well-formed `<parameter=NAME>` name is a clean token. When the model
        // malforms the tag — e.g. `<parameter=limit=1` with no closing `>` — the
        // `>`-scan spills across a newline into `</parameter`, and interpolating
        // that raw would emit INVALID JSON (live soak record 317). Detect the
        // malformed name and skip just this opener, so the well-formed sibling
        // parameter after it is still recovered. Advancing to p_name_start (past
        // the `<parameter=` we matched) guarantees forward progress.
        if (!isPlausibleParamName(p_name)) {
            param_search = p_name_start;
            continue;
        }

        const p_val_start = p_name_start + p_name_end + 1;
        const p_val_end = (hermesValueEnd(fn_body, p_val_start) orelse break) - p_val_start;
        const p_val = stripHermesValueFraming(fn_body[p_val_start .. p_val_start + p_val_end]);

        // Skip a duplicate name (first wins); still advance past its block.
        var dup = false;
        for (seen_names.items) |seen| {
            if (std.mem.eql(u8, seen, p_name)) {
                dup = true;
                break;
            }
        }
        if (dup) {
            param_search = p_val_start + p_val_end + "</parameter>".len;
            continue;
        }
        seen_names.append(allocator, p_name) catch return null;

        if (!first_param) args_map.append(allocator, ',') catch return null;
        first_param = false;

        // Escape the NAME too — belt-and-braces so no parameter name can ever
        // produce invalid JSON, even if a future malformed shape slips the guard.
        appendJsonString(allocator, &args_map, p_name) catch return null;
        args_map.append(allocator, ':') catch return null;

        // Whitespace around a NUMBER/BOOLEAN/null carries no meaning, so a
        // padded scalar is still typed from its spelling. A STRING's own
        // whitespace is PAYLOAD and ships verbatim: an `old_string` must match
        // the file "exactly, including indentation", and trimming it either
        // failed the edit or — worse — matched a different, un-indented
        // occurrence mid-line. Only the template framing is removed.
        const literal_val = std.mem.trim(u8, p_val, " \t\r\n");
        if (isJsonLiteral(literal_val)) {
            args_map.appendSlice(allocator, literal_val) catch return null;
        } else {
            appendJsonString(allocator, &args_map, p_val) catch return null;
        }

        param_search = p_val_start + p_val_end + "</parameter>".len;
    }

    args_map.append(allocator, '}') catch return null;

    return .{
        .name = allocator.dupe(u8, fn_name) catch return null,
        .arguments = allocator.dupe(u8, args_map.items) catch return null,
    };
}

/// The Hermes/Qwen XML tool-call template frames a parameter value with EXACTLY
/// one newline on each side (`<parameter=NAME>\nVALUE\n</parameter>` — pinned by the
/// Qwen3.8 render test in this file), so exactly that framing is what the parser
/// may remove. Everything else between the tags is the value's own bytes: eating
/// a run of newlines, or the spaces an `old_string` needs to match a file
/// "exactly, including indentation", silently mis-edits.
pub fn stripHermesValueFraming(raw: []const u8) []const u8 {
    var v = raw;
    if (std.mem.startsWith(u8, v, "\r\n")) {
        v = v[2..];
    } else if (std.mem.startsWith(u8, v, "\n")) {
        v = v[1..];
    }
    if (std.mem.endsWith(u8, v, "\r\n")) {
        v = v[0 .. v.len - 2];
    } else if (std.mem.endsWith(u8, v, "\n")) {
        v = v[0 .. v.len - 1];
    }
    return v;
}

// ── tests ──

test "parseToolCalls recovers truncated <function=writeFile> (max_tokens mid-content)" {
    const allocator = testing.allocator;
    // Live JFK-novel capture: a Hermes-format writeFile dumped a
    // 19k-char novel into one <parameter=content> and hit the token cap before
    // any closing tag. Pre-fix the close_rel==null branch only tried JSON, so
    // the whole call was DROPPED and leaked as visible text. We must recover at
    // least the tool NAME (args may be empty — content is truncated, not
    // salvaged) so the client fires the chunk/append nudge instead of "use JSON".
    const text = "<tool_call>\n<function=writeFile>\n<parameter=content>\n# THE LION OF MASSACHUSETTS\n\nChapter 1. The young senator rose before dawn";
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
    // Args are valid JSON (empty object) — the parameter never closed.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
}

test "parseToolCalls recovers EOS-before-close-tag <function=> with full args" {
    const allocator = testing.allocator;
    // Bonus from the same fix: a Hermes call that closed </parameter></function>
    // but hit EOS before </tool_call> now recovers WITH its args, not just the
    // name (parseHermesToolCall reads the closed parameter).
    const text = "<tool_call>\n<function=shell>\n<parameter=command>ls -la</parameter>\n</function>";
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

test "parseToolCalls recovers a bare <function=> call with NO <tool_call> opener" {
    const allocator = testing.allocator;
    // Live capture (soak, Qwen3.6-class via Claude Code): the model
    // emitted the CLOSING </tool_call> but DROPPED the opening <tool_call>, so the
    // whole call arrived as `<function=Write>…</function></tool_call>`. The outer
    // scan triggers on the substring `<tool` — which `</tool_call>` does NOT
    // contain (it is `</too…`) — so parseHermesToolCall was never reached and the
    // entire Write (with its raw <function=/<parameter= markup) leaked into the
    // user-visible text. Same big-file-write dropped-delimiter CLASS as the
    // truncated-opener bugs; recover the call so the markup never leaks.
    const text = "\n\n<function=Write>\n<parameter=file_path>\n/tmp/index.html\n</parameter>\n" ++
        "<parameter=content>\n<!DOCTYPE html>\n<html></html>\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
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
    try testing.expectEqualStrings("/tmp/index.html", parsed.value.object.get("file_path").?.string);
    try testing.expectEqualStrings("<!DOCTYPE html>\n<html></html>", parsed.value.object.get("content").?.string);
}

test "parseHermesToolCall never emits invalid JSON on a malformed <parameter=> tag" {
    // Live soak capture (record 317): the model wrote `<parameter=limit=1` — using
    // `=1` instead of closing the tag with `>` and a value. The `>`-scan then
    // spilled the 'name' across a newline into `</parameter`, and the raw
    // (unescaped) name interpolation produced INVALID JSON args that flowed to the
    // client. Two guarantees: (1) args are ALWAYS valid JSON; (2) the well-formed
    // sibling `<parameter=path>` is still recovered.
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=cat>\n<parameter=limit=1\n</parameter>\n<parameter=path>\n./parse.py\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("cat", calls[0].name);
    // Hard invariant: valid JSON object, no matter how malformed the tag was.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    // The clean sibling parameter is recovered.
    try testing.expectEqualStrings("./parse.py", parsed.value.object.get("path").?.string);
}

test "parseHermesToolCall ends a function name whose > is missing at the line break (#748)" {
    const allocator = testing.allocator;
    const cases = [_]struct { text: []const u8, path: ?[]const u8 }{
        .{ .text = "<tool_call>\n<function=read_file\n</parameter>\n</function>\n</tool_call>", .path = null },
        .{ .text = "<tool_call>\n<function=read_file\n<parameter=path>\n./a.txt\n</parameter>\n</function>\n</tool_call>", .path = "./a.txt" },
        .{ .text = "<tool_call>\n<function=read_file<parameter=path>\n./a.txt\n</parameter>\n</function>\n</tool_call>", .path = "./a.txt" },
    };
    for (cases) |case| {
        const calls = (try parseToolCalls(allocator, case.text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("read_file", calls[0].name);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        if (case.path) |p| {
            try testing.expectEqualStrings(p, parsed.value.object.get("path").?.string);
        } else {
            try testing.expectEqual(@as(usize, 0), parsed.value.object.count());
        }
    }
}

test "parseHermesToolCall dedups a repeated <parameter=> name (no duplicate JSON key)" {
    // Live soak capture (record 443, Qwen3.5-0.8B): the model emitted TWO
    // `<parameter=edits>` blocks in one call. Pre-fix that produced
    // `{"edits":…,"edits":…,"path":…}`, which std.json rejects with
    // error.DuplicateField — invalid JSON to the client. Dedup so the args are
    // always a valid object.
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=edit>\n<parameter=edits>\nfirst\n</parameter>\n" ++
        "<parameter=edits>\nsecond\n</parameter>\n<parameter=path>\n./notes.md\n</parameter>\n</function>\n</tool_call>";
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
    try testing.expect(parsed.value == .object);
    try testing.expectEqualStrings("./notes.md", parsed.value.object.get("path").?.string);
    // Exactly one `edits` key survives.
    try testing.expect(parsed.value.object.get("edits") != null);
}

test "parseHermesToolCall keeps leading/trailing whitespace inside a parameter value" {
    // The Hermes/Qwen XML template frames a value with EXACTLY one newline on
    // each side (`<parameter=NAME>\nVALUE\n</parameter>` — pinned by the
    // Qwen3.8 render test above), so that framing is the only thing the parser
    // may remove. It also did `std.mem.trim(u8, p_val, " ")`, which ate an
    // `old_string`/`new_string`'s own indentation: an Edit whose needle must
    // match the file "exactly, including indentation" then either failed, or —
    // worse — matched a DIFFERENT, un-indented occurrence mid-line and wrote
    // the replacement at the wrong nesting. Trailing whitespace is payload too
    // (a `new_string` that must end in two spaces, a markdown hard line break).
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=Edit>\n<parameter=file_path>\n/tmp/app.py\n</parameter>\n" ++
        "<parameter=old_string>\n        return None\n</parameter>\n" ++
        "<parameter=new_string>\n        return self.value  \n</parameter>\n" ++
        "</function>\n</tool_call>";
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
    try testing.expectEqualStrings("/tmp/app.py", parsed.value.object.get("file_path").?.string);
    try testing.expectEqualStrings("        return None", parsed.value.object.get("old_string").?.string);
    try testing.expectEqualStrings("        return self.value  ", parsed.value.object.get("new_string").?.string);
}

test "parseHermesToolCall strips only the ONE framing newline on each side" {
    // A value that legitimately begins or ends with a blank line keeps it: the
    // template adds one newline, not "any run of newlines". A multi-line
    // `content` written into a file is byte-exact or the file is wrong.
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=write>\n<parameter=path>\n./a.md\n</parameter>\n" ++
        "<parameter=content>\n\n# Title\n\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("\n# Title\n", parsed.value.object.get("content").?.string);
}

test "parseHermesToolCall still types a space-padded scalar from its spelling" {
    // The other half: whitespace around a NUMBER or BOOLEAN carries no meaning,
    // so a one-line `<parameter=limit> 5 </parameter>` must still ship JSON 5,
    // not the string " 5 " — fixing the string class must not break the literal
    // class (`isJsonLiteral` never matches a padded token).
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=ls>\n<parameter=path>\n./src\n</parameter>\n" ++
        "<parameter=limit> 5 </parameter>\n<parameter=recursive>\n true \n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 5), parsed.value.object.get("limit").?.integer);
    try testing.expectEqual(true, parsed.value.object.get("recursive").?.bool);
}

test "parseHermesToolCall strips a CRLF framing pair as one unit" {
    // A model that emits Windows line endings frames the value with
    // `>\r\nVALUE\r\n</parameter>`. Dropping the "\n" half alone would leave a
    // stray CR at both ends of every value — the framing is one unit.
    const allocator = testing.allocator;
    const text = "<tool_call>\r\n<function=write>\r\n<parameter=content>\r\n" ++
        "  hi\r\n</parameter>\r\n</function>\r\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("  hi", parsed.value.object.get("content").?.string);
}

test "parseToolCalls does NOT treat prose mentioning <function=> as a call" {
    const allocator = testing.allocator;
    // The bare-<function=> fallback must not fire on prose. Without a matching
    // </function> AND a real <parameter=, parseHermesToolCall recovers nothing
    // useful — but guard against a false positive on a lone mention.
    const text = "To define one you write `<function=name>` in the Hermes format.";
    const calls = try parseToolCalls(allocator, text);
    if (calls) |cs| {
        defer {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        }
        // If it parsed anything, it must at least be well-formed JSON args —
        // never leak the raw prose as a tool name.
        try testing.expect(cs.len == 0);
    }
}

test "parseToolCalls Hermes double-brace body: <tool_call>{{json}}</tool_call>" {
    // Some models (seen on small GGUF instruct models echoing a Jinja `{{ }}`
    // example) wrap the args object in an extra brace layer. The body must still
    // parse into a tool call rather than leaking through as raw content.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{{"name": "get_weather", "arguments": {"city": "Paris"}}}
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
    try testing.expectEqualStrings("get_weather", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Paris", parsed.value.object.get("city").?.string);
}

test "qwen xml: the live <function=…>/<parameter=…> capture (package.json class)" {
    // Verbatim from ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve via pi.
    // Qwen 3.5/3.6's OWN template dictates this dialect ("ONLY reply in the
    // following format"), so it is the checkpoint's contract, not a mangle.
    // The JSON-only reader mined the first balanced object out of the body —
    // the file CONTENT — and promoted its "name" key to the tool name, so pi
    // answered "Tool voxel-pagoda-garden not found" and the model looped.
    const allocator = testing.allocator;
    const text =
        "<tool_call>\n<function=write>\n<parameter=path>\n" ++
        "/Users/d/pi-lfm/package.json\n</parameter>\n<parameter=content>\n" ++
        "{\n  \"name\": \"voxel-pagoda-garden\",\n  \"version\": \"1.0.0\"\n}\n" ++
        "</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqualStrings("/Users/d/pi-lfm/package.json", o.get("path").?.string);
    // The value is RAW BYTES: a JSON file stays a STRING (typing by spelling
    // is what turned this call into `{"name": "voxel-pagoda-garden", …}`).
    // Types come from the schema at the request chokepoint, never from here.
    try testing.expectEqualStrings(
        "{\n  \"name\": \"voxel-pagoda-garden\",\n  \"version\": \"1.0.0\"\n}",
        o.get("content").?.string,
    );
}

test "qwen xml: truncated mid-parameter ships NAME + the completed params only" {
    const allocator = testing.allocator;
    const text =
        "<tool_call>\n<function=write>\n<parameter=path>\n/tmp/a.ts\n</parameter>\n" ++
        "<parameter=content>\nexport const half = ";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/a.ts", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls Hermes format" {
    const allocator = testing.allocator;
    const text = "<tool_call><function=get_weather><parameter=location>Tokyo</parameter></function></tool_call>";
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
    // Should have {"location":"Tokyo"}
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "parseToolCalls qwen: a truncated <function= call is never renamed by a JSON parameter value" {
    // Bar: a call cut before any close tag keeps the name its `<function=`
    // opener declared, whatever the parameter value looks like.
    const allocator = testing.allocator;
    const raw =
        "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/package.json\n</parameter>\n" ++
        "<parameter=content>\n{\"name\": \"voxel-pagoda-garden\", \"version\": \"1.0.0\"}";
    const calls = (try parseToolCalls(allocator, raw)) orelse return error.NoToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqualStrings("write_file", calls[0].name);
}

test "parseToolCalls qwen: a parameter value carrying </parameter> is not emptied" {
    // Bar: the value ends at the close tag that actually ends it, byte-exact.
    const allocator = testing.allocator;
    const raw =
        "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n" ++
        "<parameter=content>\nsee </parameter> in the docs\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)) orelse return error.NoToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("see </parameter> in the docs", parsed.value.object.get("content").?.string);
}

test "parseToolCalls qwen: a parameter value carrying </tool_call> is not dropped" {
    // Bar: a required parameter survives a value that spells the wrapper close.
    const allocator = testing.allocator;
    const raw =
        "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n" ++
        "<parameter=content>\nemit </tool_call> here\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)) orelse return error.NoToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("emit </tool_call> here", parsed.value.object.get("content").?.string);
}

test "parseToolCalls qwen: a parameter value types from JSON's number grammar, not parseFloat" {
    // Bar: only a spelling JSON itself calls a number stays unquoted.
    const allocator = testing.allocator;
    const Case = struct { spelling: []const u8, number: bool };
    const cases = [_]Case{
        .{ .spelling = "Infinity", .number = false },
        .{ .spelling = "nan", .number = false },
        .{ .spelling = "+1", .number = false },
        .{ .spelling = "0755", .number = false },
        .{ .spelling = ".5", .number = false },
        .{ .spelling = "5.", .number = false },
        .{ .spelling = "1.5", .number = true },
        .{ .spelling = "-2", .number = true },
        .{ .spelling = "3e5", .number = true },
        .{ .spelling = "0", .number = true },
    };
    for (cases) |c| {
        const raw = try std.mem.concat(allocator, u8, &.{
            "<tool_call>\n<function=edit>\n<parameter=mode>\n", c.spelling, "\n</parameter>\n</function>\n</tool_call>",
        });
        defer allocator.free(raw);
        const calls = (try parseToolCalls(allocator, raw)) orelse return error.NoToolCall;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{}) catch {
            std.debug.print("\n{s}: args not valid JSON: {s}\n", .{ c.spelling, calls[0].arguments });
            return error.ArgsNotValidJson;
        };
        defer parsed.deinit();
        const v = parsed.value.object.get("mode") orelse return error.MissingParam;
        if (c.number) {
            if (v == .string) {
                std.debug.print("\n{s}: shipped as a string\n", .{c.spelling});
                return error.NumberQuoted;
            }
        } else {
            if (v != .string) {
                std.debug.print("\n{s}: shipped unquoted as {s}\n", .{ c.spelling, @tagName(v) });
                return error.NonNumberUnquoted;
            }
            try testing.expectEqualStrings(c.spelling, v.string);
        }
    }
}

test "a sampled special token where <function='s `<` belongs still names the call" {
    const allocator = std.testing.allocator;
    const raw = "<tool_call>\n<|im_start|>function=read_file>\n<parameter=path>\nsrc/game.js\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)) orelse return error.TestUnexpectedResult;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("read_file", calls[0].name);
    try std.testing.expectEqualStrings("{\"path\":\"src/game.js\"}", calls[0].arguments);
    // A body with its real opener is left alone; a special that is not glued to `function=` is not touched.
    try std.testing.expect((try hermesSpecialOpenerRepair(allocator, "<function=x>\n</function>")) == null);
    try std.testing.expect((try hermesSpecialOpenerRepair(allocator, "<|im_start|>\nfunction=x>")) == null);
}
