//! A prompt boundary's identity: its prefix key, its tokens, the images inside them, and the cache configuration the
//! entry was committed under (tool-call flag, KV quantization), the fields ordinary matching also tells apart. The key
//! alone is zero for every text-only prompt of some plugins, so two different prompts of one length share it, and a
//! plugin's key may not cover the pixels of an image its tokens only name.

const std = @import("std");
const MediaSpan = @import("../../../sdk/kv/media_span.zig").MediaSpan;
const KVQuantConfig = @import("kv_quant.zig").KVQuantConfig;

pub fn id(prefix_key: u64, tokens: []const u32, media: []const MediaSpan, has_tools: bool, quant: KVQuantConfig) u64 {
    var h = std.hash.Wyhash.init(prefix_key);
    h.update(std.mem.sliceAsBytes(tokens));
    h.update(std.mem.asBytes(&@as(u64, tokens.len)));
    for (media) |span| {
        h.update(std.mem.asBytes(&@as(u64, span.start)));
        h.update(std.mem.asBytes(&span.key));
    }
    h.update(&.{ @intFromBool(has_tools), @intFromEnum(quant.scheme), quant.bits });
    h.update(std.mem.asBytes(&quant.group_size));
    return h.final();
}

test "boundary id: equal-length prompts under one key differ, and a key, a length, an image or a cache configuration changes it" {
    const dense = KVQuantConfig.dense;
    const a = id(0, &.{ 1, 2, 3, 4 }, &.{}, false, dense);
    try std.testing.expect(a != id(0, &.{ 5, 6, 7, 8 }, &.{}, false, dense));
    try std.testing.expect(a != id(9, &.{ 1, 2, 3, 4 }, &.{}, false, dense));
    try std.testing.expect(a != id(0, &.{ 1, 2, 3 }, &.{}, false, dense));
    try std.testing.expectEqual(a, id(0, &.{ 1, 2, 3, 4 }, &.{}, false, dense));
    const first = [_]MediaSpan{.{ .start = 1, .key = 7 }};
    const second = [_]MediaSpan{.{ .start = 1, .key = 8 }};
    const moved = [_]MediaSpan{.{ .start = 2, .key = 7 }};
    const with_first = id(0, &.{ 1, 2, 3, 4 }, &first, false, dense);
    try std.testing.expect(a != with_first);
    try std.testing.expect(with_first != id(0, &.{ 1, 2, 3, 4 }, &second, false, dense));
    try std.testing.expect(with_first != id(0, &.{ 1, 2, 3, 4 }, &moved, false, dense));
    try std.testing.expect(a != id(0, &.{ 1, 2, 3, 4 }, &.{}, true, dense));
    try std.testing.expect(a != id(0, &.{ 1, 2, 3, 4 }, &.{}, false, KVQuantConfig.affine(8)));
    try std.testing.expect(id(0, &.{ 1, 2, 3, 4 }, &.{}, false, KVQuantConfig.affine(4)) != id(0, &.{ 1, 2, 3, 4 }, &.{}, false, KVQuantConfig.affine(8)));
}
