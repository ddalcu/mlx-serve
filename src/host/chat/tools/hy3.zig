//! The hy3 tool-call dialect (Hunyuan 3).

const std = @import("std");
const testing = std.testing;

const suffixedTagLenAt = @import("../reasoning/think_tags.zig").suffixedTagLenAt;
const parseToolCalls = @import("parse.zig").parseToolCalls;
const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const isBareToolName = @import("xml_element.zig").isBareToolName;
const earliestIndexOfAny = @import("parse.zig").earliestIndexOfAny;

pub fn parseHy3ToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, "<tool_call")) |p| {
        const after_base = p + "<tool_call".len;
        var name_start: usize = undefined;
        if (after_base < text.len and text[after_base] == ':') {
            // Canonical singular per-call opener `<tool_call:sfx>`.
            const open_len = suffixedTagLenAt(text[p..], "<tool_call") orelse {
                pos = after_base;
                continue;
            };
            name_start = p + open_len;
        } else if (std.mem.startsWith(u8, text[p..], "<tool_calls:")) {
            // Suffixed plural WRAPPER `<tool_calls:sfx>` (hy3 only — the BARE
            // DSV4/generic `<tool_calls>` wrapper is a different format and must
            // fall through to its own parser). Normally an inner singular
            // `<tool_call:sfx>` opener follows — defer to it (the next loop
            // iteration parses that). A weak model (Hy3-REAP62, live
            // raw capture) DROPS the singular opener and jumps straight to the
            // NAME, leaking the whole (otherwise well-formed) call; recover by
            // treating the wrapper's end as the opener. Same weak-model
            // delimiter-drop class as the <tool_sep> drop below, one delimiter over.
            const wrap_len = suffixedTagLenAt(text[p..], "<tool_calls") orelse {
                pos = after_base;
                continue;
            };
            var probe = p + wrap_len;
            while (probe < text.len and std.ascii.isWhitespace(text[probe])) probe += 1;
            const inner_singular = std.mem.startsWith(u8, text[probe..], "<tool_call") and
                probe + "<tool_call".len < text.len and text[probe + "<tool_call".len] == ':';
            if (inner_singular) {
                pos = p + wrap_len;
                continue;
            }
            name_start = p + wrap_len;
        } else if (after_base < text.len and text[after_base] == '>' and blk: {
            // Bare `<tool_call>` has two sub-formats sharing this opener:
            //   • Hermes JSON `<tool_call>{…}` / function-tag `<tool_call><function=…>`
            //     → the generic scan reads those; fall through.
            //   • GLM (Laguna, tokenizer tool_parser_type "glm47"):
            //     `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…`
            //     — bare opener, NAME, then arg_key/arg_value pairs, NO plural
            //     wrapper. Route HERE only when an `<arg_key` precedes this
            //     call's close (the unambiguous GLM signal a JSON/function body
            //     never has), so the Hermes path is untouched.
            const body = after_base + 1;
            const this_close = std.mem.indexOfPos(u8, text, body, "</tool_call") orelse text.len;
            const ak_at = std.mem.indexOfPos(u8, text, body, "<arg_key") orelse text.len;
            // A parameterless GLM call has no <arg_key>: its whole body is
            // the bare NAME, a shape neither the JSON nor `<function=` arms read.
            break :blk ak_at < this_close or isBareToolName(text[body..this_close]);
        }) {
            name_start = after_base + 1;
        } else {
            // Bare `<tool_call>` Hermes JSON / function-tag — not this format.
            pos = after_base;
            continue;
        }

        // NAME runs to the first structural marker after the opener. Canonically
        // that's <tool_sep:sfx>; a mangled call (weak model dropped <tool_sep> —
        // live, Hy3-REAP62 via pi) instead closes the name with
        // </arg_value>/<arg_key>/</tool_call>, so accept any of them. No marker
        // at all → the cut happened inside the name; nothing to recover.
        const name_end = earliestIndexOfAny(text, name_start, &.{
            "<tool_sep", "<arg_key", "<arg_value", "</tool_call", "</arg_key", "</arg_value",
        }) orelse {
            pos = name_start;
            continue;
        };
        const name = std.mem.trim(u8, text[name_start..name_end], " \t\n\r");
        if (name.len == 0) {
            pos = name_start;
            continue;
        }

        var args_map: std.json.ObjectMap = .empty;
        defer args_map.deinit(allocator);

        // Consume a valid <tool_sep:sfx> if present; a mangled call that dropped
        // it (weak model — live REAP capture) starts the arg scan right at the
        // name's (wrong) close tag instead.
        var i: usize = name_end;
        if (suffixedTagLenAt(text[name_end..], "<tool_sep")) |sep_len| {
            i = name_end + sep_len;
        }
        // Parse <arg_key>/<arg_value> pairs, bounded by this call's </tool_call>.
        // SCAN to the next <arg_key> (rather than requiring it right here) so a
        // stray name-close tag between the name and the first key is skipped, and
        // match the KEY block's close TOLERANTLY (</arg_key> OR </arg_value>) —
        // REAP closes it with </arg_value> (live raw capture: bash /
        // command / "ls -la"). The well-formed path is unchanged: its <tool_sep>
        // is consumed above, <arg_key> is found immediately, and </arg_key> is the
        // earliest close.
        const call_end = std.mem.indexOfPos(u8, text, i, "</tool_call") orelse text.len;
        while (true) {
            const ak_at = std.mem.indexOfPos(u8, text, i, "<arg_key") orelse break;
            if (ak_at >= call_end) break;
            const ak_len = suffixedTagLenAt(text[ak_at..], "<arg_key") orelse {
                i = ak_at + "<arg_key".len;
                continue;
            };
            const key_start = ak_at + ak_len;
            const ak_close = earliestIndexOfAny(text, key_start, &.{ "</arg_key", "</arg_value" }) orelse break;
            const ak_close_len = suffixedTagLenAt(text[ak_close..], "</arg_key") orelse
                suffixedTagLenAt(text[ak_close..], "</arg_value") orelse break;
            // A doubled opener (`<arg_key><arg_key>K`, Flint) keeps the key after the last one.
            var key_from = key_start;
            if (std.mem.lastIndexOf(u8, text[key_start..ak_close], "<arg_key")) |rel| {
                if (suffixedTagLenAt(text[key_start + rel ..], "<arg_key")) |l| key_from = key_start + rel + l;
            }
            const key = std.mem.trim(u8, text[key_from..ak_close], " \t\n\r");
            i = ak_close + ak_close_len;
            while (i < text.len and std.ascii.isWhitespace(text[i])) i += 1;
            const av_len = suffixedTagLenAt(text[i..], "<arg_value") orelse break;
            const val_start = i + av_len;
            const av_close = std.mem.indexOfPos(u8, text, val_start, "</arg_value") orelse break;
            const av_close_len = suffixedTagLenAt(text[av_close..], "</arg_value") orelse break;
            const value = text[val_start..av_close];
            i = av_close + av_close_len;
            if (key.len > 0 and args_map.getEntry(key) == null) {
                try args_map.put(allocator, key, .{ .string = value });
            }
        }

        const args_str = try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{});
        errdefer allocator.free(args_str);
        const name_owned = try allocator.dupe(u8, name);
        errdefer allocator.free(name_owned);
        try calls.append(allocator, .{ .name = name_owned, .arguments = args_str });

        if (std.mem.indexOfPos(u8, text, i, "</tool_call")) |cl| {
            const cl_len = suffixedTagLenAt(text[cl..], "</tool_call") orelse 0;
            pos = if (cl_len > 0) cl + cl_len else cl + "</tool_call".len;
        } else {
            pos = i;
        }
    }
}

// ── tests ──

test "parseToolCalls hy3 arg_key/arg_value format (single call, hostile value bytes)" {
    const raw = "<tool_calls:opensource>\n" ++
        "<tool_call:opensource>write_file<tool_sep:opensource>\n" ++
        "<arg_key:opensource>path</arg_key:opensource>\n" ++
        "<arg_value:opensource>jfk.txt</arg_value:opensource>\n" ++
        "<arg_key:opensource>content</arg_key:opensource>\n" ++
        "<arg_value:opensource>Hello \"world\"\nline2\ttabbed</arg_value:opensource>\n" ++
        "</tool_call:opensource>\n</tool_calls:opensource>";
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
    try testing.expect(!calls[0].inferred);
    // Arguments must be VALID JSON with both keys, hostile bytes escaped.
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("jfk.txt", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("Hello \"world\"\nline2\ttabbed", parsed.value.object.get("content").?.string);
}

test "parseToolCalls hy3: two calls in one wrapper" {
    const raw = "<tool_calls:opensource>\n" ++
        "<tool_call:opensource>alpha<tool_sep:opensource>\n" ++
        "<arg_key:opensource>a</arg_key:opensource>\n<arg_value:opensource>1</arg_value:opensource>\n" ++
        "</tool_call:opensource>\n" ++
        "<tool_call:opensource>beta<tool_sep:opensource>\n" ++
        "<arg_key:opensource>b</arg_key:opensource>\n<arg_value:opensource>two</arg_value:opensource>\n" ++
        "</tool_call:opensource>\n</tool_calls:opensource>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("alpha", calls[0].name);
    try testing.expectEqualStrings("beta", calls[1].name);
}

test "parseToolCalls hy3: duplicate arg key — first wins, args stay valid JSON" {
    const raw = "<tool_call:opensource>edit<tool_sep:opensource>\n" ++
        "<arg_key:opensource>path</arg_key:opensource>\n<arg_value:opensource>a.txt</arg_value:opensource>\n" ++
        "<arg_key:opensource>path</arg_key:opensource>\n<arg_value:opensource>b.txt</arg_value:opensource>\n" ++
        "</tool_call:opensource>";
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
    try testing.expectEqualStrings("a.txt", parsed.value.object.get("path").?.string);
}

test "parseToolCalls hy3: truncated mid-value recovers name + closed args only" {
    // max_tokens cut the generation inside the second value — recover the call
    // with the CLOSED pair only; never salvage the partial content (the
    // truncated-opener class: a half-written file is worse than a retry).
    const raw = "<tool_calls:opensource>\n" ++
        "<tool_call:opensource>write_file<tool_sep:opensource>\n" ++
        "<arg_key:opensource>path</arg_key:opensource>\n<arg_value:opensource>x.txt</arg_value:opensource>\n" ++
        "<arg_key:opensource>content</arg_key:opensource>\n<arg_value:opensource>a very long novel that never clos";
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
    try testing.expectEqualStrings("x.txt", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls hy3: dropped <tool_sep> + mangled key-close still recovers name AND args" {
    // Live RAW capture (pipenetwork/Hy3-REAP62 via the running server,
    // MLX_SERVE_RAW_DUMP_FILE): the pruned model drops <tool_sep> (closes the NAME
    // with </arg_value:opensource>) and closes the arg KEY block with
    // </arg_value:opensource> instead of </arg_key:opensource> — the VALUE block is
    // well-formed. Before the fix the strict parser bailed at the missing
    // <tool_sep> and the whole call LEAKED as content (finish_reason stop), so pi
    // saw bash({}) → "command required" and looped. The corruption is small and
    // regular, so recover the FULL call: name=bash, {"command":"ls -la"}.
    const raw = "<tool_calls:opensource>\n" ++
        "<tool_call:opensource>bash</arg_value:opensource>\n" ++
        "<arg_key:opensource>command</arg_value:opensource>\n" ++
        "<arg_value:opensource>ls -la</arg_value:opensource>\n" ++
        "</tool_call:opensource>\n</tool_calls:opensource>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls hy3: dropped singular <tool_call> opener (plural wrapper only) still recovers" {
    // Live RAW capture (pipenetwork/Hy3-REAP62 via the running server,
    // MLX_SERVE_RAW_DUMP_FILE): the pruned model emitted the PLURAL wrapper
    // <tool_calls:opensource> and jumped STRAIGHT to the NAME, dropping the
    // singular per-call <tool_call:opensource> opener the parser keys on — so the
    // whole (well-structured, complete) call LEAKED as content (finish_reason
    // stop). Same weak-model delimiter-drop class as the <tool_sep> drop above,
    // one delimiter over. The call is otherwise regular (name/key closed with
    // </arg_value>), so recover the FULL call incl. the quote-bearing content.
    const raw = "<tool_calls:opensource>\n" ++
        "write_file</arg_value:opensource>\n" ++
        "<arg_key:opensource>path</arg_value:opensource>\n" ++
        "<arg_value:opensource>page.html</arg_value:opensource>\n" ++
        "<arg_key:opensource>content</arg_value:opensource>\n" ++
        "<arg_value:opensource><meta charset=\"UTF-8\"><a href=\"/x\">L</a><div class=\"hero\">Hi</div></arg_value:opensource>\n" ++
        "</tool_call:opensource>\n</tool_calls:opensource>";
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
    try testing.expectEqualStrings("page.html", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings(
        "<meta charset=\"UTF-8\"><a href=\"/x\">L</a><div class=\"hero\">Hi</div>",
        parsed.value.object.get("content").?.string,
    );
}

test "parseToolCalls hy3: suffixed think close before the wrapper still parses" {
    const raw = "planning the call</think:opensource>\n<tool_calls:opensource>\n" ++
        "<tool_call:opensource>list_files<tool_sep:opensource>\n" ++
        "<arg_key:opensource>dir</arg_key:opensource>\n<arg_value:opensource>.</arg_value:opensource>\n" ++
        "</tool_call:opensource>\n</tool_calls:opensource>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("list_files", calls[0].name);
}

// ── DeepSeek-V4 native DSML tool calls ─────────────────────────────────
