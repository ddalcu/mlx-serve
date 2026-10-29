//! The MiniCPM5 tool-call dialect.

const std = @import("std");
const testing = std.testing;

const parseToolCalls = @import("parse.zig").parseToolCalls;
const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const stripHermesValueFraming = @import("hermes.zig").stripHermesValueFraming;
const freeParsedCalls = @import("dsml.zig").freeParsedCalls;

/// MiniCPM5 V3 XML tool calls — attribute-quoted, one call per `<function>`
/// tag, no outer wrapper:
///   <function name="shell">
///     <param name="command">pwd</param>
///   </function>
/// Multiple calls appear as consecutive `<function name="…">…</function>`
/// blocks. A `<param>` value may be CDATA-wrapped
/// (`<param name="x"><![CDATA[ … ]]></param>`) — the payload is used
/// verbatim. This is its own scan, not part of the generic `<tool` family or
/// the equals-sign Hermes function-tag scan just above (`<function=NAME>`):
/// the marker is `<function` followed by WHITESPACE (the attribute form),
/// which the equals-sign form never produces (its char right after
/// `<function` is always `=`), so the two can never false-fire on each
/// other. Duplicate `<param name="K">` — first occurrence wins, mirroring
/// `parseHermesToolCall`'s `<parameter=>` dedup and `parseHy3ToolCalls`'s
/// `<arg_key>` dedup (both exist because std.json rejects duplicate object
/// keys). An unrecognized function name or an undeclared parameter is never
/// rejected here — schema validation is centralized downstream
/// (`toolCallConformsToSchema` et al., `src/host/http/server.zig`'s
/// `parseToolCallsForRequest`), exactly as for every other tag-format
/// dialect. Truncation (EOS or max_tokens mid-call): when no `</function>`
/// close is found, the param scan is bounded by end-of-text instead — any
/// COMPLETE `<param>…</param>` pairs before the cut still recover (mirrors
/// `parseHy3ToolCalls`'s `call_end` fallback); a partial trailing value is
/// never salvaged.
pub fn parseMiniCpm5ToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, "<function")) |p| {
        const after = p + "<function".len;
        // The attribute form requires whitespace right after `<function` —
        // anything else is the equals-sign Hermes form or unrelated text
        // (e.g. `<functional>`), left for that scan / plain content.
        if (after >= text.len or !std.ascii.isWhitespace(text[after])) {
            pos = after;
            continue;
        }
        const tag_end = std.mem.indexOfScalarPos(u8, text, after, '>') orelse {
            pos = text.len; // Unterminated opening tag — nothing left to recover.
            break;
        };
        const open_tag = text[after..tag_end];
        const fn_name = miniCpm5AttrValue(open_tag, "name") orelse {
            // No `name="…"` attribute — not a recognizable MiniCPM5 call.
            pos = tag_end + 1;
            continue;
        };
        if (fn_name.len == 0) {
            pos = tag_end + 1;
            continue;
        }

        const body_start = tag_end + 1;
        const close_pos = miniCpm5FindCloseTag(text, body_start, "</function>");
        const body_end = if (close_pos) |cp| cp else text.len;
        const body = text[body_start..body_end];

        var args_map: std.json.ObjectMap = .empty;
        defer args_map.deinit(allocator);

        var pscan: usize = 0;
        while (std.mem.indexOfPos(u8, body, pscan, "<param")) |pp| {
            const p_after = pp + "<param".len;
            if (p_after >= body.len or !std.ascii.isWhitespace(body[p_after])) {
                pscan = p_after;
                continue;
            }
            const p_tag_end = std.mem.indexOfScalarPos(u8, body, p_after, '>') orelse break;
            const p_open_tag = body[p_after..p_tag_end];
            const p_name = miniCpm5AttrValue(p_open_tag, "name") orelse {
                pscan = p_tag_end + 1;
                continue;
            };
            const p_val_start = p_tag_end + 1;
            const p_close_pos = miniCpm5FindCloseTag(body, p_val_start, "</param>") orelse break;
            const value = miniCpm5ParamValue(body[p_val_start..p_close_pos]);

            if (p_name.len > 0 and args_map.getEntry(p_name) == null) {
                try args_map.put(allocator, p_name, .{ .string = value });
            }
            pscan = p_close_pos + "</param>".len;
        }

        // An UNCLOSED function is salvaged only with EVIDENCE that a call was
        // really under way: a completed `<param>…</param>`, or at minimum a
        // `<param` opener. Without that, a model merely WRITING ABOUT the
        // syntax ("you write <function name=\"shell\"> to open a call") is
        // promoted into an executable `shell({})` — an unintended invocation
        // for a zero-arg tool, a bogus call replacing the answer otherwise.
        //
        // The `<param` opener half is NOT belt-and-braces; it is required by a
        // real capture. MiniCPM5-1B-OptiQ-4bit at max_tokens=12 emits exactly:
        //     <function name="shell"><param name="command">git status
        // — a genuine truncated call with ZERO completed params. Demanding a
        // COMPLETE pair discarded it. Prose naming a function carries no
        // `<param` at all, so the two stay separable.
        //
        // A properly CLOSED call is untouched either way, including the
        // zero-argument form `<function name="get_time"></function>`, which the
        // same model emits verbatim: its `</function>` is the model's own
        // commitment that the call is complete.
        if (close_pos == null and args_map.count() == 0 and
            std.mem.indexOf(u8, body, "<param") == null)
        {
            pos = text.len;
            continue;
        }

        const args_str = try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{});
        errdefer allocator.free(args_str);
        const name_owned = try allocator.dupe(u8, fn_name);
        errdefer allocator.free(name_owned);
        try calls.append(allocator, .{ .name = name_owned, .arguments = args_str });

        pos = if (close_pos) |cp| cp + "</function>".len else text.len;
    }
}

/// Find `close_tag` (`</param>` or `</function>`) starting at `from`,
/// skipping over the payload of any `<![CDATA[ … ]]>` span encountered along
/// the way. A CDATA payload may legitimately contain the literal closing-tag
/// text — e.g. a `write_file` call whose content documents this very XML
/// format, or writes an unrelated `</param>`-shaped snippet — and CDATA's
/// whole purpose is to carry such content safely; a naive substring search
/// would truncate the value/call at that inner occurrence. An unterminated
/// CDATA open (no matching `]]>`) has no knowable end, so the search gives
/// up (null) rather than guess — the caller's existing truncation path
/// already treats "no close found" as EOS/max_tokens truncation.
fn miniCpm5FindCloseTag(haystack: []const u8, from: usize, close_tag: []const u8) ?usize {
    var i = from;
    while (true) {
        const cdata_at = std.mem.indexOfPos(u8, haystack, i, "<![CDATA[");
        const close_at = std.mem.indexOfPos(u8, haystack, i, close_tag);
        if (cdata_at) |cd| {
            if (close_at == null or cd < close_at.?) {
                const payload_start = cd + "<![CDATA[".len;
                const end_pos = std.mem.indexOfPos(u8, haystack, payload_start, "]]>") orelse return null;
                i = end_pos + "]]>".len;
                continue;
            }
        }
        return close_at;
    }
}

/// Extract `key="value"` or `key='value'` from an opening tag's attribute
/// span (the text strictly between the tag name and its closing `>`).
/// Requires a word boundary before `key` so `data-name=` can't match `name=`.
/// An unquoted `key=value` is malformed (MiniCPM5 always quotes) and yields
/// null, same as a missing key.
pub fn miniCpm5AttrValue(tag_attrs: []const u8, key: []const u8) ?[]const u8 {
    var buf: [40]u8 = undefined;
    if (key.len + 1 > buf.len) return null;
    const needle = std.fmt.bufPrint(&buf, "{s}=", .{key}) catch return null;
    var search_from: usize = 0;
    while (std.mem.indexOf(u8, tag_attrs[search_from..], needle)) |rel| {
        const at = search_from + rel;
        if (at > 0) {
            const before = tag_attrs[at - 1];
            if (std.ascii.isAlphanumeric(before) or before == '_' or before == '-') {
                search_from = at + needle.len;
                continue;
            }
        }
        const q_pos = at + needle.len;
        if (q_pos >= tag_attrs.len) return null;
        const quote = tag_attrs[q_pos];
        if (quote != '"' and quote != '\'') {
            search_from = q_pos;
            continue;
        }
        const val_start = q_pos + 1;
        const val_end_rel = std.mem.indexOfScalar(u8, tag_attrs[val_start..], quote) orelse return null;
        return tag_attrs[val_start .. val_start + val_end_rel];
    }
    return null;
}

/// Resolve a `<param>` body to its final value. CDATA form
/// (`<![CDATA[ … ]]>`, detected after trimming incidental surrounding
/// whitespace) yields its payload byte-exact and unfiltered — CDATA's whole
/// purpose is to carry content verbatim. Otherwise it reuses
/// `stripHermesValueFraming`, removing at most ONE newline per side.
/// This is a deliberately CONSERVATIVE choice. Captured MiniCPM5-1B-OptiQ-4bit
/// output is VALUE-ADJACENT — `<param name="command">git status</param>`, no
/// framing newlines at all — so on real emissions this strip is a no-op and the
/// question is moot. It exists for the newline-framed layout, which the fixtures
/// use but which no capture has yet shown; removing at most one newline per side
/// is the least-lossy reading if that layout is ever emitted. Trimming a RUN, which is what
/// this did first, eats blank lines the value itself carries - the
/// indentation-destroying class fixed upstream for `<parameter=>`. Interior and
/// space-only padding (shell flags, indentation) are kept either way.
fn miniCpm5ParamValue(raw: []const u8) []const u8 {
    const probe = std.mem.trim(u8, raw, " \t\n\r");
    const cdata_open = "<![CDATA[";
    const cdata_close = "]]>";
    if (std.mem.startsWith(u8, probe, cdata_open) and std.mem.endsWith(u8, probe, cdata_close)) {
        // The close must END the probe. Keying on the LAST `]]>` anywhere let
        // `<![CDATA[x]]>tail` return just `x` — silently discarding character
        // data the model sent. A section that does not end the value is not a
        // CDATA-wrapped value; fall through and treat the whole thing as text.
        const inner = probe[cdata_open.len .. probe.len - cdata_close.len];
        return inner;
    }
    return stripHermesValueFraming(raw);
}

// ── tests ──

test "parseToolCalls minicpm5: single string arg (shell pwd)" {
    const raw = "<function name=\"shell\">\n  <param name=\"command\">pwd</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("pwd", parsed.value.object.get("command").?.string);
}

test "parseToolCalls minicpm5: shell echo hello" {
    const raw = "<function name=\"shell\">\n  <param name=\"command\">echo hello</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("echo hello", parsed.value.object.get("command").?.string);
}

test "parseToolCalls minicpm5: git status" {
    const raw = "<function name=\"shell\">\n  <param name=\"command\">git status</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("git status", parsed.value.object.get("command").?.string);
}

test "parseToolCalls minicpm5: git log -1" {
    const raw = "<function name=\"shell\">\n  <param name=\"command\">git log -1</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("git log -1", parsed.value.object.get("command").?.string);
}

test "parseToolCalls minicpm5: multiple arguments in one call" {
    const raw = "<function name=\"write_file\">\n" ++
        "  <param name=\"path\">notes.txt</param>\n" ++
        "  <param name=\"content\">hello world</param>\n" ++
        "</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 2), parsed.value.object.count());
    try testing.expectEqualStrings("notes.txt", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("hello world", parsed.value.object.get("content").?.string);
}

test "parseToolCalls minicpm5: a param value keeps its own leading/trailing blank lines" {
    // Upstream #295 (stripHermesValueFraming) established the rule for the
    // Hermes dialect: the template frames a value with EXACTLY one newline per
    // side, so exactly one is what the parser may remove. Eating a RUN destroys
    // content the model meant to send - an old_string that must match a file
    // "exactly, including indentation" then edits at the wrong place (#294).
    //
    // NOTE the captured model output is value-adjacent, so this framing case is
    // NOT yet observed in the wild; it is the conservative handling for the
    // newline-framed layout rather than a claim that MiniCPM5 emits it. This pins it - the value here is
    // framed by one newline per side AND legitimately begins and ends with a
    // blank line of its own, which must survive.
    const text = "<function name=\"write\">\n<param name=\"content\">\n\nhi\n\n</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, text)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(calls);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("\nhi\n", parsed.value.object.get("content").?.string);
}

test "parseToolCalls minicpm5: CDATA param value kept verbatim" {
    const raw = "<function name=\"write_file\">\n" ++
        "  <param name=\"path\"><![CDATA[notes.txt]]></param>\n" ++
        "  <param name=\"content\"><![CDATA[line one\nline <two> & \"three\"]]></param>\n" ++
        "</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("line one\nline <two> & \"three\"", parsed.value.object.get("content").?.string);
}

test "parseToolCalls minicpm5: character data AFTER a CDATA section is not discarded" {
    // `<![CDATA[x]]>tail` used to return just `x`: the scan took the LAST
    // `]]>` anywhere in the probe, so anything following the section was
    // silently dropped. Silent truncation of a tool argument is the worst
    // failure mode here - the call still looks well-formed to the client.
    const text = "<function name=\"w\"><param name=\"c\"><![CDATA[x]]>tail</param></function>";
    const calls = (try parseToolCalls(testing.allocator, text)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(calls);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    // Not a CDATA-wrapped VALUE (the section does not end it), so it is text.
    try testing.expectEqualStrings("<![CDATA[x]]>tail", parsed.value.object.get("c").?.string);
}

test "parseToolCalls minicpm5: CDATA payload containing a literal </param> is not truncated" {
    // A write_file call documenting this very XML format — the CDATA payload
    // legitimately contains the substring `</param>`. A naive substring
    // search for the closing tag would cut the value there; CDATA exists
    // precisely so this content is carried verbatim.
    const raw = "<function name=\"write_file\">\n" ++
        "  <param name=\"path\">docs.txt</param>\n" ++
        "  <param name=\"content\"><![CDATA[Example: <param name=\"x\">y</param> then more text]]></param>\n" ++
        "</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings(
        "Example: <param name=\"x\">y</param> then more text",
        parsed.value.object.get("content").?.string,
    );
}

test "parseToolCalls minicpm5: CDATA payload containing a literal </function> is not truncated, and a second real call still parses" {
    const raw = "<function name=\"write_file\">\n" ++
        "  <param name=\"path\">docs.txt</param>\n" ++
        "  <param name=\"content\"><![CDATA[Close a call with </function> like this]]></param>\n" ++
        "</function>\n" ++
        "<function name=\"shell\">\n  <param name=\"command\">pwd</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    const first = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer first.deinit();
    try testing.expectEqualStrings("Close a call with </function> like this", first.value.object.get("content").?.string);
    try testing.expectEqualStrings("shell", calls[1].name);
    const second = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[1].arguments, .{});
    defer second.deinit();
    try testing.expectEqualStrings("pwd", second.value.object.get("command").?.string);
}

test "parseToolCalls minicpm5: two sequential calls, no wrapper" {
    const raw = "<function name=\"shell\">\n  <param name=\"command\">pwd</param>\n</function>\n" ++
        "<function name=\"shell\">\n  <param name=\"command\">ls -la</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    const first = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer first.deinit();
    try testing.expectEqualStrings("pwd", first.value.object.get("command").?.string);
    const second = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[1].arguments, .{});
    defer second.deinit();
    try testing.expectEqualStrings("ls -la", second.value.object.get("command").?.string);
}

test "parseToolCalls minicpm5: duplicate param — first occurrence wins" {
    const raw = "<function name=\"shell\">\n" ++
        "  <param name=\"command\">pwd</param>\n" ++
        "  <param name=\"command\">ls -la</param>\n" ++
        "</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("pwd", parsed.value.object.get("command").?.string);
}

test "parseToolCalls minicpm5: undeclared function name is still parsed (kept, not guessed away)" {
    const raw = "<function name=\"delete_everything\">\n  <param name=\"path\">/</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("delete_everything", calls[0].name);
    try testing.expectEqual(false, calls[0].inferred);
}

test "parseToolCalls minicpm5: missing param is never fabricated" {
    const raw = "<function name=\"write\">\n  <param name=\"path\">notes.txt</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 1), parsed.value.object.count());
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls minicpm5: unrecognized param passes through untouched" {
    const raw = "<function name=\"shell\">\n" ++
        "  <param name=\"command\">pwd</param>\n" ++
        "  <param name=\"timeout_ms\">5000</param>\n" ++
        "</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("5000", parsed.value.object.get("timeout_ms").?.string);
}

test "parseToolCalls minicpm5: a harmony ANALYSIS channel is never scanned for this dialect" {
    // gpt-oss/harmony (upstream #247) landed a reasoning transcript format whose
    // analysis channel carries the model's private thinking. This dialect's
    // fallback scanned that text too, so a harmony model merely EXPLAINING the
    // MiniCPM5 syntax produced a real, unrequested `get_time({})` call.
    // Introduced by adding this arm - before it, `<function name=` parsed nowhere.
    const analysis = "<|start|>assistant<|channel|>analysis<|message|>The user asks about <function name=\"get_time\"></function> syntax.<|end|>";
    if (try parseToolCalls(testing.allocator, analysis)) |calls| {
        defer freeParsedCalls(calls);
        std.debug.print("harmony analysis became a call: {s}({s})\n", .{ calls[0].name, calls[0].arguments });
        return error.HarmonyAnalysisBecameToolCall;
    }

    // The dialect itself is untouched: no harmony markers, still parses.
    const plain = "<function name=\"get_time\"></function>";
    const c = (try parseToolCalls(testing.allocator, plain)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(c);
    try testing.expectEqualStrings("get_time", c[0].name);
}

test "parseToolCalls minicpm5: an UNCLOSED named opener in prose is not a call" {
    // The truncation salvage recovers NAME + complete params when EOS cut a
    // real call. It cannot, on its own, tell that apart from a model WRITING
    // ABOUT the syntax - and an unclosed opener with no params was being
    // promoted into an executable `shell({})`. For a zero-arg tool that is an
    // unintended invocation; otherwise it replaces the answer with a bogus call.
    //
    // Rule: an unclosed function must carry structural evidence that a call was
    // under way - at minimum a `<param` opener. (It was briefly a COMPLETE
    // `<param>...</param>`; a real max_tokens capture has zero completed pairs,
    // so that discarded genuine calls.) A properly CLOSED zero-arg call is
    // still a call - the close tag is the model's own commitment.
    const prose = [_][]const u8{
        "Document the syntax as <function name=\"shell\"> before discussing parameters.",
        "You write <function name=\"get_time\"> to open a call.",
    };
    for (prose) |t| {
        if (try parseToolCalls(testing.allocator, t)) |calls| {
            defer freeParsedCalls(calls);
            std.debug.print("PROSE became a call: '{s}' -> {s}({s})\n", .{ t, calls[0].name, calls[0].arguments });
            return error.ProseBecameToolCall;
        }
    }

    // A CLOSED zero-arg call still parses.
    const closed = "<function name=\"get_time\"></function>";
    const c1 = (try parseToolCalls(testing.allocator, closed)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(c1);
    try testing.expectEqualStrings("get_time", c1[0].name);
    try testing.expectEqualStrings("{}", c1[0].arguments);

    // A genuinely TRUNCATED call that got one complete param through still
    // salvages - that is the case the branch exists for.
    const cut = "<function name=\"shell\">\n<param name=\"command\">ls</param>\n<param name=\"cwd\">/tm";
    const c2 = (try parseToolCalls(testing.allocator, cut)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(c2);
    try testing.expectEqualStrings("shell", c2[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c2[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("cwd") == null);
}

test "parseToolCalls minicpm5: malformed open tag (dropped quote) never guesses a call" {
    const raw = "<function name=\"shell>\n  <param name=\"command\">pwd</param>\n</function>\nI'll run that now.";
    const calls = try parseToolCalls(testing.allocator, raw);
    try testing.expect(calls == null);
}

test "parseToolCalls minicpm5: prose before and after a call, and function-like false positives" {
    const raw = "Sure, let me check the working directory.\n" ++
        "<function name=\"shell\">\n  <param name=\"command\">pwd</param>\n</function>\n" ++
        "Done — see the result above.";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);

    // `<functional>`/`<function-like>` are prose, not the attribute-quoted
    // MiniCPM5 form (no whitespace right after `<function`) — no call.
    const prose = "Wrap it in a <functional> or <function-like> block — just prose, no call here.";
    try testing.expect(try parseToolCalls(testing.allocator, prose) == null);
}

test "parseToolCalls minicpm5: no tool call in plain prose" {
    const raw = "The `shell` function name attribute isn't used here at all — just chatting.";
    try testing.expect(try parseToolCalls(testing.allocator, raw) == null);
}

test "parseToolCalls minicpm5: existing bare Hermes <function=...> form is unaffected" {
    const raw = "<function=shell><parameter=command>pwd</parameter></function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("pwd", parsed.value.object.get("command").?.string);
}

test "parseToolCalls minicpm5: adversarial malformed input never hangs or crashes" {
    // 200 back-to-back unterminated openers — the scan must advance past each
    // one rather than re-finding the same offset forever. Built with a
    // comptime loop because Zig 0.17 no longer has the `**` repeat operator.
    const many_openers = comptime blk: {
        var s: []const u8 = "";
        for (0..200) |_| s = s ++ "<function name=\"a\">";
        break :blk s;
    };
    const inputs = [_][]const u8{
        "<function",
        "<function ",
        "<function name=",
        "<function name=\"",
        "<function name=\"a\">",
        "<function name=\"a\"><param",
        "<function name=\"a\"><param name=",
        "<function name=\"a\"><param name=\"\"></param></function>",
        "<function name=\"\"></function>",
        "<function><function><function>",
        many_openers,
    };
    for (inputs) |raw| {
        const calls = try parseToolCalls(testing.allocator, raw);
        if (calls) |cs| {
            for (cs) |tc| {
                testing.allocator.free(tc.name);
                testing.allocator.free(tc.arguments);
            }
            testing.allocator.free(cs);
        }
    }
}

test "parseToolCalls minicpm5: the three <function arms stay disjoint" {
    // MiniCPM5 and the two equals-sign Hermes paths all key on the SAME
    // `<function` marker and are discriminated ONLY by the byte after it
    // (`=` vs whitespace). Upstream's standing rule — "a `<tool_call>` body
    // carrying `<function=` is the XML dialect and is read FIRST (the qwen 3.5+
    // template mandates it)" — means the new arm must not be able to steal
    // those bytes. This pins the ordering against the real arm chain rather
    // than assuming it: the MiniCPM5 arm is gated on `calls.items.len == 0` and
    // sits after both Hermes paths, so it only ever sees text they declined.
    const allocator = testing.allocator;
    const Case = struct { text: []const u8, name: []const u8, key: []const u8, value: []const u8 };
    const cases = [_]Case{
        // 1. WRAPPED `<function=` — the <tool_call> arm owns it.
        .{
            .text = "<tool_call><function=shell><parameter=command>ls -la</parameter></function></tool_call>",
            .name = "shell",
            .key = "command",
            .value = "ls -la",
        },
        // 2. BARE `<function=` + `<parameter=` — the bare-Hermes arm owns it.
        .{
            .text = "<function=shell><parameter=command>ls -la</parameter></function>",
            .name = "shell",
            .key = "command",
            .value = "ls -la",
        },
        // 3. `<function name="…">` — only this one reaches the MiniCPM5 arm.
        .{
            .text = "<function name=\"shell\">\n  <param name=\"command\">ls -la</param>\n</function>",
            .name = "shell",
            .key = "command",
            .value = "ls -la",
        },
    };
    for (cases) |c| {
        const calls = (try parseToolCalls(allocator, c.text)) orelse {
            std.debug.print("no call parsed for: '{s}'\n", .{c.text});
            return error.ExpectedToolCall;
        };
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings(c.name, calls[0].name);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings(c.value, parsed.value.object.get(c.key).?.string);
    }
}
