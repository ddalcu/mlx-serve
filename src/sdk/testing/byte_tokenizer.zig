//! A fixture's `tokenizer.json`: GPT-2 byte-level BPE over the 256 bytes and no merges (ids 0..255 are the bytes),
//! plus added special tokens at the ids a pack expects. MLX-free.

const std = @import("std");

pub const Special = struct { id: u32, content: []const u8 };

/// The tokenizer.json bytes (owned by `a`).
pub fn json(a: std.mem.Allocator, specials: []const Special) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "{\"added_tokens\":[");
    for (specials, 0..) |sp, i| try out.print(a, "{s}{{\"id\":{d},\"content\":\"{s}\",\"special\":true}}", .{ if (i == 0) "" else ",", sp.id, sp.content });
    try out.appendSlice(a, "],\"pre_tokenizer\":{\"type\":\"ByteLevel\"},\"decoder\":{\"type\":\"ByteLevel\"},\"model\":{\"type\":\"BPE\",\"vocab\":{");
    for (0..256) |b| {
        var utf8: [4]u8 = undefined;
        const n = try std.unicode.utf8Encode(byteChar(@intCast(b)), &utf8);
        const ch = utf8[0..n];
        try out.appendSlice(a, if (b == 0) "\"" else ",\"");
        if (ch.len == 1 and (ch[0] == '"' or ch[0] == '\\')) try out.append(a, '\\');
        try out.print(a, "{s}\":{d}", .{ ch, b });
    }
    try out.appendSlice(a, "},\"merges\":[]}}");
    return out.items;
}

/// GPT-2's bytes_to_unicode: printable Latin-1 bytes map to themselves, the rest to 256 + their rank.
fn byteChar(b: u8) u21 {
    if ((b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or b >= 0xAE) return b;
    var n: u21 = 0;
    var x: u16 = 0;
    while (x < b) : (x += 1) {
        const c: u8 = @intCast(x);
        if (!((c >= '!' and c <= '~') or (c >= 0xA1 and c <= 0xAC) or c >= 0xAE)) n += 1;
    }
    return 256 + n;
}

// ── tests ──

test "byte tokenizer: space maps to Ġ, printable bytes to themselves" {
    try std.testing.expectEqual(@as(u21, 'Ġ'), byteChar(' '));
    try std.testing.expectEqual(@as(u21, 'a'), byteChar('a'));
    try std.testing.expectEqual(@as(u21, 256), byteChar(0));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const j = try json(arena.allocator(), &.{.{ .id = 300, .content = "<|x|>" }});
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), j, .{});
    try std.testing.expectEqual(@as(usize, 256), parsed.object.get("model").?.object.get("vocab").?.object.count());
}
