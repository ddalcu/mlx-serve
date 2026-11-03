//! Tool markup openers and `trimLeakedToolMarkup`: unparsed tool markup never rides out.

const std = @import("std");
const testing = std.testing;

const INKLING_INVOKE_TAG = @import("../reasoning/inkling.zig").INKLING_INVOKE_TAG;
const parseToolCalls = @import("parse.zig").parseToolCalls;
const inklingIsNameChar = @import("inkling.zig").inklingIsNameChar;
const BARE_INVOKE = @import("invoke.zig").BARE_INVOKE;
const FUNCTION_CALLS_TAG = @import("invoke.zig").FUNCTION_CALLS_TAG;
const freeParsedCalls = @import("dsml.zig").freeParsedCalls;

/// Tool-call WRAPPER openers (and DSML's closer, which no prose spells) that must never survive into text a client
/// renders. Every entry is the opening marker of a call block, so whatever
/// follows it is call payload, not prose — parseToolCalls has already run by
/// the time these matter, so a marker still present means the block did not
/// parse (a mangled opener, a bare marker with no call behind it, or a call
/// the model emitted inside its think block, which the parser deliberately
/// skips). The whole tail is cut rather than excised: a wrapper we could not
/// parse has no reliable end, and shipping half of it is the same leak.
const tool_markup_openers = [_][]const u8{
    "<｜DSML｜", // DeepSeek-V4 native (covers tool_calls/invoke/parameter)
    "</｜DSML｜", // its closers, once the opener is gone (a call lost to the think split or never written)
    "<|tool_call", // Gemma 4
    "<tool_call", // Hermes XML + the plural/suffixed variants
    "<tool_calls:", // Hy3 suffixed wrapper
    "<atem:", // Muse-Glimmer ATEM (covers function_calls/invoke/parameter)
    INKLING_INVOKE_TAG,
};

/// Earliest index of Anthropic-style `<function_calls>` / `<invoke …>` markup, or null. Like MiniCPM5's,
/// the opener alone is prose ("wrap it in <invoke name=…>"): a `<parameter ` or `</invoke>` must be there too.
fn bareInvokeMarkupCut(text: []const u8) ?usize {
    if (std.mem.indexOf(u8, text, "<parameter ") == null and
        std.mem.indexOf(u8, text, BARE_INVOKE.invoke_close) == null) return null;
    var cut = std.mem.indexOf(u8, text, FUNCTION_CALLS_TAG);
    var scan: usize = 0;
    while (std.mem.indexOfPos(u8, text, scan, BARE_INVOKE.invoke)) |p| {
        const after = p + BARE_INVOKE.invoke.len;
        if (after >= text.len or std.ascii.isWhitespace(text[after])) {
            if (cut == null or p < cut.?) cut = p;
            break;
        }
        scan = after;
    }
    return cut;
}

/// True while `buf` holds, or ends in a growing prefix of, a `<function_calls>` / `<invoke …` opener.
pub fn invokeOpenerHoldsForTools(buf: []const u8) bool {
    if (std.mem.indexOf(u8, buf, FUNCTION_CALLS_TAG) != null) return true;
    var scan: usize = 0;
    while (std.mem.indexOfPos(u8, buf, scan, BARE_INVOKE.invoke)) |p| {
        const after = p + BARE_INVOKE.invoke.len;
        if (after >= buf.len or std.ascii.isWhitespace(buf[after])) return true;
        scan = after;
    }
    inline for (.{ FUNCTION_CALLS_TAG, BARE_INVOKE.invoke }) |marker| {
        var k: usize = 2;
        while (k < marker.len) : (k += 1) if (std.mem.endsWith(u8, buf, marker[0..k])) return true;
    }
    return false;
}

/// Earliest index of MiniCPM5 attribute-form tool markup, or null.
/// Discriminated exactly as the streaming gate does it — `<function` followed
/// by WHITESPACE, with a `name=` inside the opener region — so `<functional`,
/// `<function-like>` and `<function>` prose are never cut. A malformed opener
/// (dropped quote) still matches, which is the point: `parseToolCalls` has
/// already run by the time this matters, so markup still present is markup that
/// did NOT parse, and shipping it renders raw XML to the user.
fn miniCpm5MarkupCut(text: []const u8) ?usize {
    // The opener ALONE is not evidence of leaked markup. Prose names functions
    // ("Document the syntax as <function name=\"shell\"> before …") and cutting
    // there deletes the rest of the answer — the exact contradiction that let a
    // green parser test and a green scrubber test disagree about one string.
    // Require the same structural evidence the parser requires: a `<param`
    // element or a `</function>` close somewhere in the text.
    if (std.mem.indexOf(u8, text, "<param") == null and
        std.mem.indexOf(u8, text, "</function>") == null) return null;
    var scan: usize = 0;
    while (std.mem.indexOfPos(u8, text, scan, "<function")) |p| {
        const after = p + "<function".len;
        if (after >= text.len) return p; // truncated mid-opener
        if (!std.ascii.isWhitespace(text[after])) {
            scan = after;
            continue;
        }
        const gt_rel = std.mem.indexOfScalar(u8, text[after..], '>');
        const region_end = if (gt_rel) |g| after + g else text.len;
        if (std.mem.indexOf(u8, text[after..region_end], "name=") != null) return p;
        scan = region_end;
    }
    return null;
}

/// Cut visible text at the first unparsed tool-call marker. Returns a prefix
/// slice (no alloc); text with no marker is returned untouched.
pub fn trimLeakedToolMarkup(text: []const u8) []const u8 {
    var cut: usize = text.len;
    var is_inkling = false;
    for (tool_markup_openers) |m| {
        if (std.mem.indexOf(u8, text, m)) |p| {
            if (p < cut) {
                cut = p;
                is_inkling = std.mem.eql(u8, m, INKLING_INVOKE_TAG);
            }
        }
    }
    // MiniCPM5 attribute form. Same reasoning, different discriminator: the
    // dialect is `<function` + whitespace + a `name=` attribute, so prose words
    // beginning `<function…` are left alone.
    if (miniCpm5MarkupCut(text)) |p| {
        if (p < cut) {
            cut = p;
            is_inkling = false;
        }
    }
    if (bareInvokeMarkupCut(text)) |p| {
        if (p < cut) {
            cut = p;
            is_inkling = false;
        }
    }
    if (cut == text.len) return text;
    // Inkling puts the tool NAME immediately BEFORE its invoke marker, so the
    // cut takes the trailing identifier run with it — otherwise the bare name
    // is what the client renders as the answer.
    if (is_inkling) {
        while (cut > 0 and inklingIsNameChar(text[cut - 1])) cut -= 1;
    }
    return std.mem.trimEnd(u8, text[0..cut], "\n\r\t ");
}

// NOTE: there is deliberately NO strip-think helper anymore. Reasoning the
// model generated is ALWAYS delivered (reasoning_content / thinking blocks) —
// thinking-off is enforced on the PROMPT side (`noThinkTailSuffix`), so every
// delivery site takes `splitThinkBlock(text, true, …)` and ships both halves.

// ── tests ──

test "trimLeakedToolMarkup cuts at the first unparsed tool-call opener" {
    // Untouched when there is no marker (the overwhelmingly common case —
    // the returned slice must be the input itself, not a trimmed copy).
    try testing.expectEqualStrings("plain prose", trimLeakedToolMarkup("plain prose"));
    try testing.expectEqualStrings("ends in a tag <b>", trimLeakedToolMarkup("ends in a tag <b>"));

    // Every wrapper family cuts, and the trailing whitespace before the
    // marker goes with it.
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"x\">"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<|tool_call>{}"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<tool_call>{}"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<tool_calls:opensource>"));
    // Inkling's NAME sits before the marker, so it goes with the cut.
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done. shell<|content_invoke_tool_json|>{}"));

    // A marker with nothing before it leaves nothing behind — a bare marker
    // is not an answer (live: DSV4 answered a question and then emitted a
    // lone `<｜DSML｜` with no call behind it).
    try testing.expectEqualStrings("", trimLeakedToolMarkup("\n\n<｜DSML｜\n"));

    // Earliest marker wins when several families appear.
    try testing.expectEqualStrings("a", trimLeakedToolMarkup("a<tool_call>b<｜DSML｜c"));
}

test "trimLeakedToolMarkup cuts a DSML closer whose opener is gone (#980)" {
    try testing.expectEqualStrings("Task 2 green. Running the full check.", trimLeakedToolMarkup("Task 2 green. Running the full check.\n\n</｜DSML｜ calls>"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n\n</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"));
}

test "a tool marker split across tokens is only visible once concatenated" {
    // The streaming tools flush holds a tail that looked like a tool call and
    // ships it when nothing parses. DSV4 spells the DSML marker as `<` then
    // `｜DSML｜` (two tokens), so a per-token filter sees neither piece as a
    // marker and both went out as visible content. The
    // flush concatenates first for exactly this reason.
    const pieces = [_][]const u8{ "My name is", "<", "｜DSML｜", "\n" };
    var joined: [64]u8 = undefined;
    var n: usize = 0;
    for (pieces) |p| {
        @memcpy(joined[n .. n + p.len], p);
        n += p.len;
    }
    for (pieces) |p| try testing.expectEqualStrings(p, trimLeakedToolMarkup(p));
    try testing.expectEqualStrings("My name is", trimLeakedToolMarkup(joined[0..n]));
}

test "parseToolCalls: Anthropic-style bare <invoke> markup is an inferred call (#928)" {
    const allocator = testing.allocator;
    const text = "<function_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\">cd /x && ls -la</parameter>\n<parameter name=\"timeout\">30</parameter>\n</invoke>\n</function_calls>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    // Not the model's own dialect: it must name a declared tool to survive.
    try testing.expect(calls[0].inferred);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("cd /x && ls -la", parsed.value.object.get("command").?.string);
    try testing.expectEqual(@as(i64, 30), parsed.value.object.get("timeout").?.integer);
    // Prose words are not the tag.
    try testing.expect((try parseToolCalls(allocator, "I <invoked> it with <parameters> set.")) == null);
    // Nothing of an unparsed block reaches content; prose naming the tag stays.
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<function_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\">ls"));
    const prose = "Claude's format wraps each call in <invoke name=\"x\"> tags.";
    try testing.expectEqualStrings(prose, trimLeakedToolMarkup(prose));
}

test "minicpm5 COMBINED: parser and scrubber agree on the same bytes" {
    // The earlier defect was two individually-green tests that CONTRADICTED
    // each other: the parser called a string prose while the scrubber cut it as
    // markup. Neither test could see it because neither ran both layers. This
    // one does, so that shape cannot recur silently.
    //
    // Fixtures marked LIVE are verbatim raw emissions of
    // mlx-community/MiniCPM5-1B-OptiQ-4bit captured via MLX_SERVE_RAW_DUMP_FILE.
    const Case = struct {
        raw: []const u8,
        is_call: bool, // parser must produce a call
        keeps_text: bool, // scrubber must leave the text intact
        args: ?[]const u8 = null, // when set, the exact arguments JSON
    };
    const cases = [_]Case{
        // LIVE: normal parameterised call - value-adjacent, no framing newlines.
        .{ .raw = "<function name=\"shell\"><param name=\"command\">git status</param></function>", .is_call = true, .keeps_text = false },
        // LIVE: zero-argument call, closed with an empty body.
        .{ .raw = "<function name=\"get_time\"></function>", .is_call = true, .keeps_text = false },
        // LIVE: truncated at max_tokens - a real call, ZERO completed params.
        // Salvages the NAME with empty args; the unterminated value never ships
        // (a fragment argument is worse than none - the call looks complete).
        .{ .raw = "<function name=\"shell\"><param name=\"command\">git status", .is_call = true, .keeps_text = false, .args = "{}" },
        // Prose naming a function: no <param, no close. Neither layer may act.
        .{ .raw = "Document the syntax as <function name=\"shell\"> before discussing parameters.", .is_call = false, .keeps_text = true },
        .{ .raw = "Wrap it in a <functional> block - just prose.", .is_call = false, .keeps_text = true },
        // Malformed (dropped quote) WITH structural evidence: not a call, but
        // it is markup, so it must be scrubbed rather than rendered.
        .{ .raw = "<function name=\"shell>\n  <param name=\"command\">pwd</param>\n</function>", .is_call = false, .keeps_text = false },
    };
    for (cases) |c| {
        const parsed = try parseToolCalls(testing.allocator, c.raw);
        if (parsed) |calls| {
            if (c.args) |want| try testing.expectEqualStrings(want, calls[0].arguments);
            freeParsedCalls(calls);
        }
        const got_call = parsed != null;
        testing.expectEqual(c.is_call, got_call) catch |e| {
            std.debug.print("parser disagreed on: '{s}' (got call={})\n", .{ c.raw, got_call });
            return e;
        };
        const kept = std.mem.eql(u8, std.mem.trimEnd(u8, c.raw, "\n\r\t "), trimLeakedToolMarkup(c.raw));
        testing.expectEqual(c.keeps_text, kept) catch |e| {
            std.debug.print("scrubber disagreed on: '{s}' -> '{s}'\n", .{ c.raw, trimLeakedToolMarkup(c.raw) });
            return e;
        };
        // THE invariant: text the parser calls prose is text the scrubber keeps.
        if (!got_call and !kept) {
            const still_markup = std.mem.indexOf(u8, c.raw, "<param") != null or
                std.mem.indexOf(u8, c.raw, "</function>") != null;
            testing.expect(still_markup) catch |e| {
                std.debug.print("CONTRADICTION: parser says prose, scrubber cut it: '{s}'\n", .{c.raw});
                return e;
            };
        }
    }
}
