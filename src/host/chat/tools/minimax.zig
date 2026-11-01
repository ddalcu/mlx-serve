//! Native MiniMax namespaced XML tool calls.

const std = @import("std");
const ParsedToolCall = @import("parse.zig").ParsedToolCall;
const dsmlAttr = @import("dsml.zig").dsmlAttr;

pub const MINIMAX_NAMESPACE = "]<]minimax[>[";
const MINIMAX_INVOKE = MINIMAX_NAMESPACE ++ "<invoke";
const MINIMAX_INVOKE_CLOSE = MINIMAX_NAMESPACE ++ "</invoke>";

/// Native MiniMax XML marks every structural tag with its namespace, leaving
/// markup in string argument values untouched. Schema coercion types leaves.
pub fn parseMiniMaxToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, MINIMAX_INVOKE)) |inv| {
        const after = inv + MINIMAX_INVOKE.len;
        pos = after;
        if (after == text.len or !std.ascii.isWhitespace(text[after])) continue;
        const te = std.mem.indexOfScalarPos(u8, text, after, '>') orelse break;
        const name = dsmlAttr(text[after..te], "name") orelse continue;
        if (name.len == 0 or std.mem.indexOfAny(u8, name, "\r\n<>") != null) continue;
        const body_start = te + 1;
        const close = std.mem.indexOfPos(u8, text, body_start, MINIMAX_INVOKE_CLOSE) orelse break;
        pos = close + MINIMAX_INVOKE_CLOSE.len;
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const aa = arena.allocator();
        const value = (try miniMaxXmlValue(aa, text[body_start..close], 0, true)) orelse continue;
        const name_owned = try allocator.dupe(u8, name);
        errdefer allocator.free(name_owned);
        const arguments = try std.json.Stringify.valueAlloc(allocator, value, .{});
        errdefer allocator.free(arguments);
        try calls.append(allocator, .{ .name = name_owned, .arguments = arguments, .inferred = false });
    }
}

fn miniMaxXmlValue(a: std.mem.Allocator, body: []const u8, depth: u32, object_required: bool) std.mem.Allocator.Error!?std.json.Value {
    if (depth > 32) return null;
    const t = std.mem.trim(u8, body, " \t\r\n");
    if (!object_required and !std.mem.startsWith(u8, t, MINIMAX_NAMESPACE ++ "<")) return .{ .string = body };
    var map: std.json.ObjectMap = .empty;
    var list = std.json.Array.init(a);
    var array: ?bool = null;
    var pos: usize = 0;
    while (pos < t.len) {
        while (pos < t.len and std.ascii.isWhitespace(t[pos])) pos += 1;
        if (pos == t.len) break;
        if (!std.mem.startsWith(u8, t[pos..], MINIMAX_NAMESPACE ++ "<")) return null;
        const key_start = pos + MINIMAX_NAMESPACE.len + 1;
        const key_end = std.mem.indexOfScalarPos(u8, t, key_start, '>') orelse return null;
        const key = t[key_start..key_end];
        if (key.len == 0 or key.len > 256) return null;
        for (key) |c| if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-' and c != '.') return null;
        const is_item = !object_required and std.mem.eql(u8, key, "item");
        if (array) |previous| {
            if (previous != is_item) return null;
        } else array = is_item;
        const close_tag = try std.fmt.allocPrint(a, MINIMAX_NAMESPACE ++ "</{s}>", .{key});
        const open_tag = try std.fmt.allocPrint(a, MINIMAX_NAMESPACE ++ "<{s}>", .{key});
        const value_start = key_end + 1;
        var scan = value_start;
        var nested: usize = 0;
        var value_end: ?usize = null;
        while (std.mem.indexOfPos(u8, t, scan, close_tag)) |candidate| {
            if (std.mem.indexOfPos(u8, t, scan, open_tag)) |next_open| {
                if (next_open < candidate) {
                    nested += 1;
                    scan = next_open + open_tag.len;
                    continue;
                }
            }
            if (nested == 0) {
                value_end = candidate;
                break;
            }
            nested -= 1;
            scan = candidate + close_tag.len;
        }
        const end = value_end orelse return null;
        const value = (try miniMaxXmlValue(a, t[value_start..end], depth + 1, false)) orelse return null;
        if (is_item) try list.append(value) else try map.put(a, key, value);
        pos = end + close_tag.len;
    }
    if (array orelse false) return .{ .array = list };
    return .{ .object = map };
}
