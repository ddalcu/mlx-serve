//! Request media owned across an engine's asynchronous submit boundary.
const std = @import("std");
const abi = @import("../../abi/engine.zig");

pub fn dupe(a: std.mem.Allocator, media: []const abi.Media) ![]abi.Media {
    const out = try a.alloc(abi.Media, media.len);
    var copied: usize = 0;
    errdefer {
        for (out[0..copied]) |m| a.free(abi.items(u8, m.data));
        a.free(out);
    }
    for (media, out) |m, *dst| {
        dst.* = m;
        dst.data = abi.span(u8, try a.dupe(u8, abi.items(u8, m.data)));
        copied += 1;
    }
    return out;
}

pub fn deinit(a: std.mem.Allocator, media: []abi.Media) void {
    for (media) |m| a.free(abi.items(u8, m.data));
    a.free(media);
}

test "engine media: the asynchronous copy owns pixels and preserves descriptors" {
    const a = std.testing.allocator;
    var rgb = [_]u8{ 1, 2, 3 };
    const input = [_]abi.Media{.{ .kind = .image, .at = 4, .width = 1, .height = 1, .hash = 7, .data = .of(&rgb) }};
    const owned = try dupe(a, &input);
    defer deinit(a, owned);
    rgb[0] = 99;
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3 }, abi.items(u8, owned[0].data));
    try std.testing.expectEqual(@as(u32, 4), owned[0].at);
    try std.testing.expectEqual(@as(u64, 7), owned[0].hash);
}
