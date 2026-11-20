//! Authenticate a replayed assistant body before substituting its retained session tokens.
//! The chat imports below are pure format helpers, not the host request pipeline; this module never calls MLX.

const std = @import("std");
const reasoning = @import("../../host/chat/reasoning/split.zig");
const think_tags = @import("../../host/chat/reasoning/think_tags.zig");
const tools = @import("../../host/chat/tools/parse.zig");
const hermes = @import("../../host/chat/tools/hermes.zig");
const hy3 = @import("../../host/chat/tools/hy3.zig");
const balancedJsonObject = @import("../../host/chat/tools/json_infer.zig").balancedJsonObject;
const isBareToolName = @import("../../host/chat/tools/xml_element.zig").isBareToolName;
const isPlausibleParamName = @import("../../sdk/util/json_text.zig").isPlausibleParamName;

pub const Reply = struct {
    /// The complete decoded assistant body, excluding role and end-of-turn tokens. Never a suffix window.
    text: []const u8,
    /// True only when this body's preceding generation prompt actually opened a thought block.
    opened_by_template: bool = false,
};

/// Visible prose and ordered calls must agree in full. A replay may omit reasoning; reasoning it retains must agree.
/// Only native complete Qwen/GLM calls and strict Hermes JSON are normalized. Ambiguity is a cache miss.
pub fn equivalent(gpa: std.mem.Allocator, sampled: Reply, replay: Reply) error{OutOfMemory}!bool {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    return compare(arena.allocator(), sampled, replay) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidReply => false,
    };
}

const Error = error{ OutOfMemory, InvalidReply };
const Body = struct { content: []const u8, thought: ?[]const u8 = null };
const Call = struct { name: []const u8, args: std.json.Value };
const Part = struct { prose: []const u8, call: ?Call = null };
const open_call = "<tool_call>";
const close_call = "</tool_call>";
const ws = " \t\r\n";

fn body(reply: Reply) Error!Body {
    const text = reply.text;
    const leading = std.mem.trimStart(u8, text, ws);
    if (!reply.opened_by_template and think_tags.thinkOpenTagLenAt(leading) == null)
        return .{ .content = text };
    // The delivery splitter also recovers truncated thoughts; authentication requires their real closing boundary.
    const close = think_tags.indexOfThinkCloseTag(leading, 0) orelse return error.InvalidReply;
    const split = reasoning.splitThinkBlockKeepingMarkup(leading, true, reply.opened_by_template);
    // Delivery may scrub trailing markup; authentication must retain every byte after the leading thought.
    return .{ .content = std.mem.trimStart(u8, leading[close.pos + close.len ..], "\n "), .thought = split.reasoning_content };
}

fn compare(a: std.mem.Allocator, sampled: Reply, replay: Reply) Error!bool {
    const s = try body(sampled);
    const r = try body(replay);
    if (r.thought) |thought| {
        if (!std.mem.eql(u8, thought, s.thought orelse return false)) return false;
    }
    var si: usize = 0;
    var ri: usize = 0;
    var meaningful = false;
    while (true) {
        const sp = try next(a, s.content, &si);
        const rp = try next(a, r.content, &ri);
        if (!std.mem.eql(u8, sp.prose, rp.prose)) return false;
        meaningful = meaningful or sp.prose.len != 0;
        if (sp.call) |sc| {
            const rc = rp.call orelse return false;
            if (!std.mem.eql(u8, sc.name, rc.name) or !try valuesEqual(a, sc.args, rc.args)) return false;
            meaningful = true;
        } else return meaningful and rp.call == null;
    }
}

fn next(a: std.mem.Allocator, text: []const u8, at: *usize) Error!Part {
    const start = std.mem.indexOfPos(u8, text, at.*, open_call) orelse {
        const rest = text[at.*..];
        at.* = text.len;
        return .{ .prose = rest };
    };
    const prose = std.mem.trimEnd(u8, text[at.*..start], ws);
    const begin = start + open_call.len;
    const first = skipWs(text, begin);
    if (first == text.len) return error.InvalidReply;
    var end: usize = undefined;
    var call: Call = undefined;
    if (text[first] == '{') {
        const raw = balancedJsonObject(text[first..]) orelse return error.InvalidReply;
        const obj = try json(a, raw);
        if (obj != .object or obj.object.count() != 2) return error.InvalidReply;
        const name = obj.object.get("name") orelse return error.InvalidReply;
        const args = obj.object.get("arguments") orelse return error.InvalidReply;
        if (name != .string or name.string.len == 0 or args != .object) return error.InvalidReply;
        end = skipWs(text, first + raw.len);
        call = .{ .name = name.string, .args = args };
    } else {
        const checked = try nativeEnd(text, first);
        end = checked.end;
        if (!std.mem.startsWith(u8, text[end..], close_call)) return error.InvalidReply;
        const parsed = if (checked.qwen)
            hermes.parseHermesToolCall(a, text[first..end]) orelse return error.InvalidReply
        else blk: {
            var calls: std.ArrayList(tools.ParsedToolCall) = .empty;
            hy3.parseHy3ToolCalls(a, text[start .. end + close_call.len], &calls) catch |err|
                return parseError(err);
            if (calls.items.len != 1) return error.InvalidReply;
            break :blk calls.items[0];
        };
        if (!std.mem.eql(u8, parsed.name, checked.name)) return error.InvalidReply;
        const args = try json(a, parsed.arguments);
        // Native parsers intentionally recover partial calls and deduplicate keys. Neither is proof of equivalence.
        if (args != .object or args.object.count() != checked.fields) return error.InvalidReply;
        call = .{ .name = parsed.name, .args = args };
    }
    if (!std.mem.startsWith(u8, text[end..], close_call)) return error.InvalidReply;
    at.* = skipWs(text, end + close_call.len);
    return .{ .prose = prose, .call = call };
}

/// Validate full native framing before using the existing, intentionally tolerant semantic parser.
fn nativeEnd(text: []const u8, begin: usize) Error!struct { end: usize, fields: usize, name: []const u8, qwen: bool } {
    const qwen = std.mem.startsWith(u8, text[begin..], "<function=");
    var at = begin + if (qwen) "<function=".len else 0;
    const name_end = std.mem.indexOfAnyPos(u8, text, at, if (qwen) ">" else "<") orelse return error.InvalidReply;
    const name = std.mem.trim(u8, text[at..name_end], ws);
    if (!isBareToolName(name)) return error.InvalidReply;
    at = skipWs(text, name_end + @intFromBool(qwen));
    var fields: usize = 0;
    while (true) {
        if (qwen and std.mem.startsWith(u8, text[at..], "</function>")) {
            at = skipWs(text, at + "</function>".len);
            break;
        }
        if (!qwen and std.mem.startsWith(u8, text[at..], close_call)) break;
        const key_open = if (qwen) "<parameter=" else "<arg_key>";
        if (!std.mem.startsWith(u8, text[at..], key_open)) return error.InvalidReply;
        const key_start = at + key_open.len;
        const key_close = if (qwen) ">" else "</arg_key>";
        const key_end = std.mem.indexOfPos(u8, text, key_start, key_close) orelse return error.InvalidReply;
        if (!isPlausibleParamName(text[key_start..key_end])) return error.InvalidReply;
        at = key_end + key_close.len;
        if (qwen) {
            const relative = hermes.hermesParamSpanEnclosing(text[begin..], at - begin) orelse return error.InvalidReply;
            at = begin + relative;
        } else {
            at = skipWs(text, at);
            if (!std.mem.startsWith(u8, text[at..], "<arg_value>")) return error.InvalidReply;
            at += "<arg_value>".len;
            const end = std.mem.indexOfPos(u8, text, at, "</arg_value>") orelse return error.InvalidReply;
            at = end + "</arg_value>".len;
        }
        fields += 1;
        at = skipWs(text, at);
    }
    return .{ .end = at, .fields = fields, .name = name, .qwen = qwen };
}

fn skipWs(text: []const u8, from: usize) usize {
    var at = from;
    while (at < text.len and std.mem.indexOfScalar(u8, ws, text[at]) != null) : (at += 1) {}
    return at;
}

fn parseError(err: anyerror) Error {
    return if (err == error.OutOfMemory) error.OutOfMemory else error.InvalidReply;
}

// All parsed values live in the comparison's arena. Keep numbers as text so large integers never round through f64.
fn json(a: std.mem.Allocator, text: []const u8) Error!std.json.Value {
    const parsed = std.json.parseFromSlice(std.json.Value, a, text, .{ .parse_numbers = false }) catch |err|
        return parseError(err);
    return parsed.value;
}

fn valuesEqual(a: std.mem.Allocator, left: std.json.Value, right: std.json.Value) Error!bool {
    const Pair = struct { l: std.json.Value, r: std.json.Value };
    var pending: std.ArrayList(Pair) = .empty;
    try pending.append(a, .{ .l = left, .r = right });
    while (pending.pop()) |p| {
        if (std.meta.activeTag(p.l) != std.meta.activeTag(p.r)) return false;
        switch (p.l) {
            .null => {},
            .bool => |v| if (v != p.r.bool) return false,
            .string => |v| if (!std.mem.eql(u8, v, p.r.string)) return false,
            .number_string => |v| if (!try numbersEqual(a, v, p.r.number_string)) return false,
            .array => |v| {
                if (v.items.len != p.r.array.items.len) return false;
                for (v.items, p.r.array.items) |l, r| try pending.append(a, .{ .l = l, .r = r });
            },
            .object => |v| {
                if (v.count() != p.r.object.count()) return false;
                var it = v.iterator();
                while (it.next()) |entry| {
                    const r = p.r.object.get(entry.key_ptr.*) orelse return false;
                    try pending.append(a, .{ .l = entry.value_ptr.*, .r = r });
                }
            },
            else => return error.InvalidReply,
        }
    }
    return true;
}

const Number = struct { negative: bool, digits: []const u8, exponent: i128 };

fn number(a: std.mem.Allocator, raw: []const u8) Error!?Number {
    const negative = raw[0] == '-';
    const exp_at = std.mem.indexOfAny(u8, raw, "eE") orelse raw.len;
    const exponent = if (exp_at == raw.len) 0 else std.fmt.parseInt(i128, raw[exp_at + 1 ..], 10) catch return null;
    const mantissa = raw[@intFromBool(negative)..exp_at];
    const dot = std.mem.indexOfScalar(u8, mantissa, '.');
    const fractional: i128 = if (dot) |d| @intCast(mantissa.len - d - 1) else 0;
    var digits: std.ArrayList(u8) = .empty;
    for (mantissa) |c| if (c != '.') try digits.append(a, c);
    const significant = std.mem.trimStart(u8, digits.items, "0");
    if (significant.len == 0) return .{ .negative = false, .digits = "0", .exponent = 0 };
    const trimmed = std.mem.trimEnd(u8, significant, "0");
    const power = std.math.sub(i128, exponent, fractional) catch return null;
    const adjusted = std.math.add(i128, power, @intCast(significant.len - trimmed.len)) catch return null;
    return .{ .negative = negative, .digits = trimmed, .exponent = adjusted };
}

fn numbersEqual(a: std.mem.Allocator, left: []const u8, right: []const u8) Error!bool {
    if (std.mem.eql(u8, left, right)) return true;
    const l = (try number(a, left)) orelse return false;
    const r = (try number(a, right)) orelse return false;
    return l.negative == r.negative and l.exponent == r.exponent and std.mem.eql(u8, l.digits, r.digits);
}

const testing = std.testing;

fn expectEquivalent(want: bool, sampled: []const u8, replay: []const u8) !void {
    try testing.expectEqual(want, try equivalent(testing.allocator, .{ .text = sampled }, .{ .text = replay }));
}

test "reply: the entire answer is authenticated, including an identical long suffix" {
    try expectEquivalent(true, "The answer is 42.", "The answer is 42.");
    try expectEquivalent(false, "Allow access. This ending is identical.", "Deny access. This ending is identical.");
    try expectEquivalent(false, "One\n two", "One\ntwo");
    try expectEquivalent(false, "", "");
    try expectEquivalent(false, "A full answer", "answer");
    try expectEquivalent(false, "answer", "edited</think>answer");
}

test "reply: native Qwen and GLM reasoning may be omitted, never a different visible answer" {
    const sampled = "<think>private plan</think>\n\nThe answer is 42.";
    try expectEquivalent(true, sampled, "<think>\n\n</think>\n\nThe answer is 42.");
    try expectEquivalent(true, sampled, "The answer is 42.");
    try expectEquivalent(true, sampled, sampled);
    try expectEquivalent(false, sampled, "<think>changed plan</think>The answer is 42.");
    try expectEquivalent(false, sampled, "<think></think>The answer is 43.");
    try expectEquivalent(false, sampled, "<think></think>The answer is 42.<think>extra");
    try expectEquivalent(false, "<think>unfinished", "<think>another unfinished");
    try testing.expect(try equivalent(testing.allocator, .{ .text = "private plan</think>answer", .opened_by_template = true }, .{ .text = "<think></think>answer" }));
}

test "reply: Qwen native parameters match JSON calls with reordered typed arguments" {
    const native = "<tool_call>\n<function=edit>\n<parameter=path>\na.txt\n</parameter>\n" ++
        "<parameter=count>\n2\n</parameter>\n<parameter=enabled>\ntrue\n</parameter>\n</function>\n</tool_call>";
    try expectEquivalent(true, native, "<tool_call>{\"arguments\":{\"enabled\":true,\"count\":2,\"path\":\"a.txt\"},\"name\":\"edit\"}</tool_call>");
    try expectEquivalent(false, native, "<tool_call>{\"name\":\"edit\",\"arguments\":{\"path\":\"b.txt\",\"count\":2,\"enabled\":true}}</tool_call>");
    try expectEquivalent(false, native, "<tool_call>{\"name\":\"edit\",\"arguments\":{\"path\":\"a.txt\",\"count\":\"2\",\"enabled\":true}}</tool_call>");
}

test "reply: GLM native arguments keep their string type without a schema" {
    const native = "<tool_call>edit<arg_key>path</arg_key><arg_value>a.txt</arg_value>" ++
        "<arg_key>count</arg_key><arg_value>2</arg_value></tool_call>";
    try expectEquivalent(true, native, "<tool_call>edit\n<arg_key>count</arg_key><arg_value>2</arg_value>\n" ++
        "<arg_key>path</arg_key><arg_value>a.txt</arg_value></tool_call>");
    try expectEquivalent(true, native, "<tool_call>{\"name\":\"edit\",\"arguments\":{\"count\":\"2\",\"path\":\"a.txt\"}}</tool_call>");
    try expectEquivalent(false, native, "<tool_call>{\"name\":\"edit\",\"arguments\":{\"count\":2,\"path\":\"a.txt\"}}</tool_call>");
    try expectEquivalent(false, native, "<tool_call>edit<arg_key>path</arg_key><arg_value>b.txt</arg_value>" ++
        "<arg_key>count</arg_key><arg_value>2</arg_value></tool_call>");
}

test "reply: all calls, names, ordering, and surrounding visible prose matter" {
    const first = "<tool_call>{\"name\":\"read\",\"arguments\":{}}</tool_call>";
    const second = "<tool_call>{\"name\":\"write\",\"arguments\":{}}</tool_call>";
    try expectEquivalent(true, "Checking.\n\n" ++ first ++ "\n" ++ second, "Checking.\n" ++ first ++ second);
    try expectEquivalent(false, first ++ second, second ++ first);
    try expectEquivalent(false, first ++ second, first);
    try expectEquivalent(false, first, first ++ second);
    try expectEquivalent(false, "Checking." ++ first, "Edited." ++ first);
    try expectEquivalent(false, first ++ "Done.", first ++ "Edited.");
}

test "reply: strict JSON values preserve structure, scalar types, strings, and large numbers" {
    const original = "<tool_call>{\"name\":\"edit\",\"arguments\":{\"x\":{\"a\":[true,null,2],\"b\":\"x\\ny\"}}}</tool_call>";
    try expectEquivalent(true, original, "<tool_call>{\"arguments\":{\"x\":{\"b\":\"x\\u000ay\",\"a\":[true,null,2.0]}},\"name\":\"edit\"}</tool_call>");
    const variants = [_][]const u8{ "[true,null,3]", "[null,true,2]", "[\"true\",null,2]", "{\"a\":true}", "[true,null]" };
    for (variants) |value| {
        const edited = try std.fmt.allocPrint(testing.allocator, "<tool_call>{{\"name\":\"edit\",\"arguments\":{{\"x\":{{\"a\":{s},\"b\":\"x\\ny\"}}}}}}</tool_call>", .{value});
        defer testing.allocator.free(edited);
        try expectEquivalent(false, original, edited);
    }
    try expectEquivalent(false, "<tool_call>{\"name\":\"x\",\"arguments\":{\"n\":9007199254740992.0}}</tool_call>", "<tool_call>{\"name\":\"x\",\"arguments\":{\"n\":9007199254740993.0}}</tool_call>");
    try expectEquivalent(true, "<tool_call>{\"name\":\"x\",\"arguments\":{\"n\":12300e-2}}</tool_call>", "<tool_call>{\"name\":\"x\",\"arguments\":{\"n\":123}}</tool_call>");
}

test "reply: malformed or duplicate native parameters cannot authenticate via parser recovery" {
    const empty = "<tool_call><function=edit></function></tool_call>";
    try expectEquivalent(true, empty, "<tool_call>edit</tool_call>");
    for ([_][]const u8{
        "<tool_call><function=edit><parameter=path>unfinished</function></tool_call>",
        "<tool_call><function=edit>ignored</function></tool_call>",
        "<tool_call>edit<arg_key>path</arg_key><arg_value>unfinished</tool_call>",
        "<tool_call>edit junk</tool_call>",
        "<tool_call>{\"name\":\"edit\",\"arguments\":{}}ignored</tool_call>",
        "<tool_call>{\"name\":\"edit\",\"arguments\":{\"x\":1,\"x\":2}}</tool_call>",
        "<tool_call>{\"name\":\"edit\",\"arguments\":{}",
    }) |malformed| try expectEquivalent(false, empty, malformed);
    const duplicate = "<tool_call>edit<arg_key>x</arg_key><arg_value>1</arg_value><arg_key>x</arg_key><arg_value>2</arg_value></tool_call>";
    try expectEquivalent(false, duplicate, duplicate);
}

test "reply: native parameter payload whitespace and literal protocol bytes are not discarded" {
    const raw = "<tool_call><function=write><parameter=content>\n  keep </think> and </tool_call>\n</parameter></function></tool_call>";
    try expectEquivalent(true, raw, "<tool_call>{\"name\":\"write\",\"arguments\":{\"content\":\"  keep </think> and </tool_call>\"}}</tool_call>");
    try expectEquivalent(false, raw, "<tool_call>{\"name\":\"write\",\"arguments\":{\"content\":\"keep </think> and </tool_call>\"}}</tool_call>");
    const embedded = "<tool_call><function=write><parameter=content>\n<arg_key>x</arg_key><arg_value>1</arg_value>\n</parameter></function></tool_call>";
    try expectEquivalent(true, embedded, "<tool_call>{\"name\":\"write\",\"arguments\":{\"content\":\"<arg_key>x</arg_key><arg_value>1</arg_value>\"}}</tool_call>");
}

test "reply: literal protocol-looking prose requires the whole decoded sequence" {
    try expectEquivalent(true, "<50><50><50>", "<50><50><50>");
    try expectEquivalent(false, "<50><50><50>", "<9><50><50><50>");
}
