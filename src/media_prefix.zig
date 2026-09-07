const std = @import("std");

/// Identity of one rendered media block, including its opening marker.
pub const Span = struct {
    start: usize,
    end: usize,
    digest: [32]u8,
};

/// Null spans retain the legacy whole-request media-key contract.
pub const State = struct {
    key: u64 = 0,
    start: ?usize = null,
    spans: ?[]const Span = null,
};

pub const Block = struct {
    insert_at: usize,
    rows: usize,
    token: u32,
    digest: [32]u8,
};

pub const Layout = struct {
    tokens: []u32,
    spans: []Span,

    pub fn deinit(self: Layout, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        allocator.free(self.spans);
    }
};

/// A raw text placeholder has no pixel identity/span. Never let it consume
/// rows belonging to a later real image while remaining outside that span.
pub fn validateRaw(tokens: []const u32, reserved: []const u32) !void {
    for (tokens) |token| {
        for (reserved) |marker| {
            if (marker != 0 and token == marker) return error.UntrackedMediaToken;
        }
    }
}

/// Replace owned one-pad blocks emitted through each source message's
/// template content. Role markers elsewhere in the prompt are irrelevant.
pub fn expandInline(allocator: std.mem.Allocator, raw: []const u32, specs: []const Block, open: u32, close: u32, image: u32, video: u32) !Layout {
    var text = std.ArrayList(u32).empty;
    defer text.deinit(allocator);
    var bound = std.ArrayList(Block).empty;
    defer bound.deinit(allocator);
    var i: usize = 0;
    while (i < raw.len) {
        if (open != 0 and raw[i] == open) {
            if (bound.items.len >= specs.len or i + 2 >= raw.len or raw[i + 2] != close or raw[i + 1] != specs[bound.items.len].token)
                return error.InvalidMediaLayout;
            var block = specs[bound.items.len];
            block.insert_at = text.items.len;
            try bound.append(allocator, block);
            i += 3;
        } else {
            try text.append(allocator, raw[i]);
            i += 1;
        }
    }
    if (bound.items.len != specs.len) return error.InvalidMediaLayout;
    try validateRaw(text.items, &.{ open, close, image, video });
    return expand(allocator, text.items, bound.items, open, close);
}

test "media inline provenance ignores appended user headers" {
    const a = std.testing.allocator;
    const spec = Block{ .insert_at = 0, .rows = 2, .token = 900, .digest = @splat(1) };
    const raw = [_]u32{ 100, 101, 800, 900, 801, 7, 102, 100, 101, 8 };
    const old = try expandInline(a, raw[0..7], &.{spec}, 800, 801, 900, 901);
    defer old.deinit(a);
    const next = try expandInline(a, &raw, &.{spec}, 800, 801, 900, 901);
    defer next.deinit(a);
    try std.testing.expectEqualSlices(u32, old.tokens, next.tokens[0..old.tokens.len]);
    try std.testing.expectEqual(@as(usize, 2), next.spans[0].start);
    try std.testing.expectError(error.UntrackedMediaToken, expandInline(a, &.{900}, &.{}, 800, 801, 900, 901));
    try std.testing.expectError(error.InvalidMediaLayout, expandInline(a, &.{ 800, 900, 801, 800, 900, 801 }, &.{spec}, 800, 801, 900, 901));
    try std.testing.expectError(error.InvalidMediaLayout, expandInline(a, &.{1}, &.{spec}, 800, 801, 900, 901));
}

test "media layout rejects untracked raw media markers" {
    try validateRaw(&.{ 1, 2, 3 }, &.{ 0, 800, 801, 900, 901 });
    try validateRaw(&.{ 0, 1 }, &.{0}); // zero is an unset config field
    for ([_]u32{ 800, 801, 900, 901 }) |token| {
        try std.testing.expectError(error.UntrackedMediaToken, validateRaw(&.{ 1, token, 2 }, &.{ 800, 801, 900, 901 }));
    }
}

pub fn expand(allocator: std.mem.Allocator, tokens: []const u32, blocks: []const Block, open: u32, close: u32) !Layout {
    var size = tokens.len;
    var prev: usize = 0;
    for (blocks) |block| {
        if (block.insert_at < prev or block.insert_at > tokens.len or block.rows == 0)
            return error.InvalidMediaLayout;
        size = try std.math.add(usize, size, try std.math.add(usize, block.rows, 2));
        prev = block.insert_at;
    }
    const output = try allocator.alloc(u32, size);
    errdefer allocator.free(output);
    const spans = try allocator.alloc(Span, blocks.len);
    var read: usize = 0;
    var write: usize = 0;
    for (blocks, 0..) |block, i| {
        const text_len = block.insert_at - read;
        @memcpy(output[write..][0..text_len], tokens[read..block.insert_at]);
        write += text_len;
        spans[i] = .{ .start = write, .end = write + block.rows + 2, .digest = block.digest };
        output[write] = open;
        @memset(output[write + 1 ..][0..block.rows], block.token);
        output[write + block.rows + 1] = close;
        write += block.rows + 2;
        read = block.insert_at;
    }
    @memcpy(output[write..], tokens[read..]);
    return .{ .tokens = output, .spans = spans };
}

test "media layout preserves historical blocks and individual image boundaries" {
    const a = std.testing.allocator;
    const one = Block{ .insert_at = 2, .rows = 2, .token = 900, .digest = @splat(1) };
    const two = Block{ .insert_at = 6, .rows = 1, .token = 900, .digest = @splat(2) };
    const old = try expand(a, &.{ 1, 2, 3, 4, 5 }, &.{one}, 800, 801);
    defer old.deinit(a);
    const next = try expand(a, &.{ 1, 2, 3, 4, 5, 6, 7 }, &.{ one, two, two }, 800, 801);
    defer next.deinit(a);
    try std.testing.expectEqualSlices(u32, old.tokens, next.tokens[0..old.tokens.len]);
    try std.testing.expectEqualSlices(u32, &.{ 800, 900, 801, 800, 900, 801 }, next.tokens[10..16]);
    try std.testing.expectEqual(@as(usize, 2), next.spans[0].start);
    try std.testing.expectEqual(@as(usize, 10), next.spans[1].start);
    try std.testing.expectEqual(@as(usize, 13), next.spans[2].start);
    try std.testing.expectEqual(@as(usize, 10), sharedLimit(.{ .spans = old.spans }, .{ .spans = next.spans }, next.tokens.len));
}

/// Exclusive upper bound on state reusable between two media histories.
pub fn sharedLimit(entry: State, request: State, token_limit: usize) usize {
    if (entry.spans) |a| {
        if (request.spans) |b| {
            for (0..@min(a.len, b.len)) |i| {
                if (a[i].start != b[i].start or a[i].end != b[i].end or
                    !std.mem.eql(u8, &a[i].digest, &b[i].digest))
                    return @min(token_limit, @min(a[i].start, b[i].start));
            }
            if (a.len > b.len) return @min(token_limit, a[b.len].start);
            if (b.len > a.len) return @min(token_limit, b[a.len].start);
            return token_limit;
        }
    }
    if (entry.key == request.key) return token_limit;
    const boundary = if (entry.start) |a|
        if (request.start) |b| @min(a, b) else a
    else
        request.start orelse return 0;
    return @min(boundary, token_limit);
}

test "media prefix append keeps text after unchanged images" {
    const a = Span{ .start = 100, .end = 200, .digest = @splat(1) };
    const b = Span{ .start = 30000, .end = 30100, .digest = @splat(2) };
    const old = State{ .key = 1, .start = 100, .spans = &.{a} };
    const next = State{ .key = 2, .start = 100, .spans = &.{ a, b } };
    try std.testing.expectEqual(@as(usize, 29970), sharedLimit(old, next, 29970));
    try std.testing.expectEqual(@as(usize, 30000), sharedLimit(old, next, 31000));
}

test "media prefix edit reorder removal and layout changes stop at first divergence" {
    const a = Span{ .start = 100, .end = 200, .digest = @splat(1) };
    const b = Span{ .start = 30000, .end = 30100, .digest = @splat(2) };
    const old = State{ .key = 1, .start = 100, .spans = &.{ a, b } };
    var edited = b;
    edited.digest[0] = 3;
    try std.testing.expectEqual(@as(usize, 30000), sharedLimit(old, .{ .key = 2, .start = 100, .spans = &.{ a, edited } }, 40000));
    edited = a;
    edited.digest = b.digest;
    try std.testing.expectEqual(@as(usize, 100), sharedLimit(old, .{ .key = 2, .start = 100, .spans = &.{ edited, b } }, 40000));
    try std.testing.expectEqual(@as(usize, 30000), sharedLimit(old, .{ .key = 2, .start = 100, .spans = &.{a} }, 40000));
    edited = b;
    edited.start -= 10;
    try std.testing.expectEqual(@as(usize, 29990), sharedLimit(old, .{ .key = 2, .start = 100, .spans = &.{ a, edited } }, 40000));
}

test "media prefix empty and legacy identities fail conservatively" {
    try std.testing.expectEqual(@as(usize, 900), sharedLimit(.{}, .{}, 900));
    try std.testing.expectEqual(@as(usize, 0), sharedLimit(.{ .key = 1 }, .{}, 900));
    try std.testing.expectEqual(@as(usize, 100), sharedLimit(.{ .key = 1, .start = 100 }, .{ .key = 2, .start = 200 }, 900));
}

test "media prefix spans override equal whole keys and permit partial blocks" {
    const a = Span{ .start = 100, .end = 200, .digest = @splat(1) };
    var b = a;
    b.end = 201;
    const old = State{ .key = 1, .spans = &.{a} };
    try std.testing.expectEqual(@as(usize, 100), sharedLimit(old, .{ .key = 1, .spans = &.{b} }, 900));
    try std.testing.expectEqual(@as(usize, 150), sharedLimit(old, old, 150));
    try std.testing.expectEqual(@as(usize, 100), sharedLimit(old, .{ .spans = &.{} }, 900));
}
