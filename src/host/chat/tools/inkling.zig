//! The Inkling tool-call dialect and its name helpers.

const std = @import("std");
const testing = std.testing;

const INKLING_END_TAG = @import("../reasoning/inkling.zig").INKLING_END_TAG;
const INKLING_INVOKE_TAG = @import("../reasoning/inkling.zig").INKLING_INVOKE_TAG;
const parseToolCalls = @import("parse.zig").parseToolCalls;
const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const balancedJsonObject = @import("json_infer.zig").balancedJsonObject;

/// A tool-name character per the identifier alphabet every template family's
/// names draw from. The Inkling NAME is glued directly to the invoke marker,
/// so anything outside this set (a channel marker's `>`, a payload's `}`,
/// prose punctuation) terminates the name run.
pub fn inklingIsNameChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '.' or c == '-';
}

/// The trailing identifier run ending at `end` (exclusive). Live:
/// the boundary-based prefix scan swallowed the `<|content_text|>` marker
/// (name `<|content_text|>bash`), and a dropped `<|end_message|>` between
/// calls produced `}}write` — both resolve to the bare NAME here.
pub fn inklingTrailingNameRun(text: []const u8, end: usize) []const u8 {
    var start = end;
    while (start > 0 and inklingIsNameChar(text[start - 1])) start -= 1;
    return text[start..end];
}

pub fn parseInklingToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, INKLING_INVOKE_TAG)) |inv| {
        const prefix_name = inklingTrailingNameRun(text, inv);

        // Body = the balanced JSON object right after the marker (string-aware
        // brace scan). The next <|end_message|> is NOT a reliable terminator:
        // the model drops it between back-to-back calls (live,
        // `{…}}write<|content_invoke_tool_json|>{…}`), and an end-tag-bounded
        // body then swallows the following call whole. No balance by end of
        // text = truncation → the NAME + `{}` salvage below.
        const body_start = inv + INKLING_INVOKE_TAG.len;
        var body: []const u8 = "";
        const after = std.mem.trimStart(u8, text[body_start..], "\n ");
        if (std.mem.startsWith(u8, after, "{")) {
            if (balancedJsonObject(after)) |obj| {
                body = obj;
                pos = (body_start + (text.len - body_start - after.len)) + obj.len;
            } else {
                pos = text.len; // truncated object — nothing further to scan
            }
        } else {
            // No object after the marker at all: keep the old end-tag-bounded
            // read so a non-JSON payload still gets a parse attempt.
            const body_end = std.mem.indexOfPos(u8, text, body_start, INKLING_END_TAG) orelse text.len;
            body = std.mem.trim(u8, text[body_start..body_end], "\n ");
            pos = body_end;
        }

        var name: []const u8 = prefix_name;
        var args_json: ?[]const u8 = null; // allocated when set
        if (std.json.parseFromSlice(std.json.Value, allocator, body, .{})) |parsed| {
            defer parsed.deinit();
            if (parsed.value == .object) {
                if (parsed.value.object.get("name")) |nv| {
                    if (nv == .string and nv.string.len > 0) {
                        name = nv.string;
                        // Marker-echo repair: pi's "Tool <|content_text|>bash
                        // not found" reply taught the model to copy the garbage
                        // name INTO its payloads. No real
                        // tool name contains `<|` — keep the trailing
                        // identifier run; empty → fall back to the prefix name.
                        if (std.mem.indexOf(u8, name, "<|") != null) {
                            name = inklingTrailingNameRun(name, name.len);
                            if (name.len == 0) name = prefix_name;
                        }
                    }
                }
                if (parsed.value.object.get("args")) |av| {
                    if (av == .object) {
                        // Re-serialize through Stringify: escaping + key dedup
                        // by construction (the tag-converter invariant).
                        args_json = std.json.Stringify.valueAlloc(allocator, av, .{}) catch null;
                    }
                }
                // `name` borrows from `parsed` — dupe before deinit.
                name = try allocator.dupe(u8, name);
            } else {
                name = try allocator.dupe(u8, name);
            }
        } else |_| {
            // Truncated / mangled payload: NAME + {} (never ship fragments).
            name = try allocator.dupe(u8, name);
        }
        errdefer allocator.free(name);
        if (name.len == 0) {
            allocator.free(name);
            if (args_json) |a| allocator.free(a);
            continue;
        }
        const arguments = args_json orelse try allocator.dupe(u8, "{}");
        try calls.append(allocator, .{ .name = name, .arguments = arguments, .inferred = false });
    }
}

// ── tests ──

test "parseToolCalls: Inkling invoke_tool_json family" {
    const allocator = testing.allocator;
    // Canonical shape from the chat template: NAME<|content_invoke_tool_json|>
    // {"name":...,"args":{...}}<|end_message|>, optionally after a thinking message.
    {
        const text = "<|content_thinking|>Need the time.<|end_message|><|message_model|>get_time<|content_invoke_tool_json|>{\"name\":\"get_time\",\"args\":{\"timezone\":\"UTC\"}}<|end_message|>";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("get_time", calls[0].name);
        try testing.expect(!calls[0].inferred);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings("UTC", parsed.value.object.get("timezone").?.string);
    }
    // Two calls in one turn (parallel) — each its own message.
    {
        const text = "<|message_model|>alpha<|content_invoke_tool_json|>{\"name\":\"alpha\",\"args\":{\"a\":1}}<|end_message|><|message_model|>beta<|content_invoke_tool_json|>{\"name\":\"beta\",\"args\":{}}<|end_message|>";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 2), calls.len);
        try testing.expectEqualStrings("alpha", calls[0].name);
        try testing.expectEqualStrings("beta", calls[1].name);
    }
    // Truncated mid-args: salvage NAME + {} — never partial values.
    {
        const text = "<|message_model|>write_file<|content_invoke_tool_json|>{\"name\":\"write_file\",\"args\":{\"path\":\"/tmp/x\",\"content\":\"half-writ";
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
        try testing.expectEqualStrings("{}", calls[0].arguments);
    }
}

test "parseToolCalls: Inkling name is the trailing identifier run" {
    const allocator = testing.allocator;
    // Live pi session: the tool message opens with the
    // <|content_text|> marker glued to the NAME. When the payload is
    // truncated, the salvage name must be the bare identifier — never
    // `<|content_text|>bash` (pi's "Tool <|content_text|>bash not found"
    // reply taught the model to echo that garbage name back).
    {
        const text = "<|message_model|><|content_text|>bash<|content_invoke_tool_json|>{\"name\":\"bash\",\"args\":{\"command\":\"ls";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("bash", calls[0].name);
        try testing.expectEqualStrings("{}", calls[0].arguments);
    }
    // Marker echoed INTO the payload's "name" (the self-reinforcing loop's
    // second stage): sanitize to the trailing identifier run.
    {
        const text = "<|message_model|>bash<|content_invoke_tool_json|>{\"name\":\"<|content_text|>bash\",\"args\":{}}<|end_message|>";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("bash", calls[0].name);
    }
    // Payload name that is ONLY a marker, no prefix name either → the call
    // has no recoverable name and is skipped.
    {
        const text = "<|message_model|><|content_invoke_tool_json|>{\"name\":\"<|content_text|>\",\"args\":{}}<|end_message|>";
        const calls = try parseToolCalls(allocator, text);
        try testing.expect(calls == null);
    }
}

test "parseToolCalls: Inkling back-to-back invokes without <|end_message|>" {
    const allocator = testing.allocator;
    // The model drops the <|end_message|> between two calls
    // (`{…}}write<|content_invoke_tool_json|>{…}`). Body extraction must stop
    // at the balanced object — an end-tag-bounded body swallows call 2 whole.
    const text = "<|message_model|><|content_text|>write<|content_invoke_tool_json|>{\"name\":\"write\",\"args\":{\"path\":\"a.js\",\"content\":\"x\"}}write<|content_invoke_tool_json|>{\"name\":\"write\",\"args\":{\"path\":\"b.js\",\"content\":\"y\"}}<|end_message|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    try testing.expectEqualStrings("write", calls[1].name);
    const p0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer p0.deinit();
    try testing.expectEqualStrings("a.js", p0.value.object.get("path").?.string);
    const p1 = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer p1.deinit();
    try testing.expectEqualStrings("b.js", p1.value.object.get("path").?.string);
}
