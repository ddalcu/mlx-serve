//! One media item's rows in a prompt and the bounds a prefix-cache match clamps at: the host lays a prompt's media out
//! in these, the engine's caches key state on them.

const std = @import("std");

/// One media item's rows in a prompt: its first placeholder row and its
/// pixel/PCM hash. Placeholder ids are identical across items, so state at and
/// after `start` is keyed on `key`.
pub const MediaSpan = struct { start: u32, key: u64 };

/// First position where state under two span lists can differ; maxInt when
/// they agree. Rows before the first span that differs are text.
pub fn mediaSharedBound(a: []const MediaSpan, b: []const MediaSpan) usize {
    const n = @min(a.len, b.len);
    for (a[0..n], b[0..n]) |x, y| {
        if (x.start != y.start or x.key != y.key) return @min(x.start, y.start);
    }
    if (a.len > n) return a[n].start;
    if (b.len > n) return b[n].start;
    return std.math.maxInt(usize);
}

/// The spans whose rows lie inside `[0, len)`.
pub fn spansBelow(spans: []const MediaSpan, len: usize) []const MediaSpan {
    var n: usize = 0;
    while (n < spans.len and spans[n].start < len) n += 1;
    return spans[0..n];
}

// ── tests

const testing = std.testing;

test "mediaSharedBound: a repeated pixel key in a superset does not collapse the bound" {
    // A pixel key may legally repeat across spans: the comparison is
    // positional, so it must walk past the reuse and clamp only where a
    // start/key pair actually differs.
    const K1: u64 = 8_376_670_522_757_638_769;
    const K2: u64 = 0xB2;
    const a = [_]MediaSpan{ .{ .start = 100, .key = K1 }, .{ .start = 300, .key = K2 } };
    const b = [_]MediaSpan{ .{ .start = 100, .key = K1 }, .{ .start = 300, .key = K2 }, .{ .start = 500, .key = K1 } };

    // The superset's extra item REUSES an earlier key: the bound is that item's
    // start, not a collapse back to the first K1 occurrence.
    try testing.expectEqual(@as(usize, 500), mediaSharedBound(&a, &b));
    try testing.expectEqual(@as(usize, 500), mediaSharedBound(&b, &a));
    // Identical lists agree everywhere.
    try testing.expectEqual(std.math.maxInt(usize), mediaSharedBound(&a, &a));
    try testing.expectEqual(std.math.maxInt(usize), mediaSharedBound(&b, &b));
    // A repeat that sits behind a real divergence still clamps at the divergence.
    const c = [_]MediaSpan{ .{ .start = 100, .key = K1 }, .{ .start = 300, .key = 0xB9 }, .{ .start = 500, .key = K1 } };
    try testing.expectEqual(@as(usize, 300), mediaSharedBound(&a, &c));
    // A text turn still clamps at the first item.
    try testing.expectEqual(@as(usize, 100), mediaSharedBound(&a, &.{}));
}
