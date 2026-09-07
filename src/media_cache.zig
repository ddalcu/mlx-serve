const std = @import("std");
const mlx = @import("mlx.zig");
const chat = @import("chat.zig");

pub const Key = [32]u8;

/// Caller serializes access; values returned by get are independently owned.
pub fn Cache(comptime Value: type) type {
    return struct {
        const Self = @This();
        const Entry = struct { key: Key, value: Value };
        entries: std.ArrayList(Entry) = .empty,
        bytes: usize = 0,
        max_bytes: usize = 256 * 1024 * 1024,
        max_entries: usize = 64,

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            for (self.entries.items) |entry| entry.value.deinit(allocator);
            self.entries.deinit(allocator);
            self.entries = .empty;
            self.bytes = 0;
        }

        pub fn get(self: *Self, allocator: std.mem.Allocator, key: Key) !?Value {
            for (self.entries.items, 0..) |entry, i| {
                if (!std.mem.eql(u8, &entry.key, &key)) continue;
                const value = try entry.value.clone(allocator);
                const touched = self.entries.orderedRemove(i);
                self.entries.appendAssumeCapacity(touched);
                return value;
            }
            return null;
        }

        pub fn put(self: *Self, allocator: std.mem.Allocator, key: Key, value: Value) !void {
            const size = value.size();
            if (size > self.max_bytes or self.max_entries == 0) return;
            const owned = try value.clone(allocator);
            errdefer owned.deinit(allocator);
            try self.entries.ensureTotalCapacity(allocator, @min(self.max_entries, self.entries.items.len + 1));
            for (self.entries.items, 0..) |entry, i| {
                if (!std.mem.eql(u8, &entry.key, &key)) continue;
                self.remove(allocator, i);
                break;
            }
            while (self.entries.items.len >= self.max_entries or self.bytes > self.max_bytes - size)
                self.remove(allocator, 0);
            self.entries.appendAssumeCapacity(.{ .key = key, .value = owned });
            self.bytes += size;
        }

        fn remove(self: *Self, allocator: std.mem.Allocator, i: usize) void {
            const entry = self.entries.orderedRemove(i);
            self.bytes -= entry.value.size();
            entry.value.deinit(allocator);
        }
    };
}

pub const Pixels = struct {
    image: chat.ImageData,

    pub fn size(self: Pixels) usize {
        return self.image.pixels.len;
    }
    pub fn clone(self: Pixels, allocator: std.mem.Allocator) !Pixels {
        var result = self;
        result.image.pixels = try allocator.dupe(u8, self.image.pixels);
        return result;
    }
    pub fn deinit(self: Pixels, allocator: std.mem.Allocator) void {
        allocator.free(self.image.pixels);
    }
};

pub const Embedding = struct {
    array: mlx.mlx_array,
    bytes: usize,

    pub fn size(self: Embedding) usize {
        return self.bytes;
    }
    pub fn clone(self: Embedding, _: std.mem.Allocator) !Embedding {
        var result = self;
        result.array = mlx.mlx_array_new();
        if (mlx.mlx_array_set(&result.array, self.array) != 0) {
            _ = mlx.mlx_array_free(result.array);
            return error.ArrayShareFailed;
        }
        return result;
    }
    pub fn deinit(self: Embedding, _: std.mem.Allocator) void {
        _ = mlx.mlx_array_free(self.array);
    }
};

pub fn pixelKey(kind: u8, dimensions: []const u32, pixels: []const u8) Key {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(&.{kind});
    hash.update(std.mem.sliceAsBytes(dimensions));
    hash.update(pixels);
    return hash.finalResult();
}

test "media cache owns hits and evicts LRU within byte and count limits" {
    const a = std.testing.allocator;
    var cache = Cache(Pixels){ .max_bytes = 4, .max_entries = 2 };
    defer cache.deinit(a);
    const value = Pixels{ .image = .{ .pixels = "ab", .width = 1, .height = 1 } };
    try cache.put(a, @splat(1), value);
    try cache.put(a, @splat(2), value);
    const hit = (try cache.get(a, @splat(1))) orelse return error.ExpectedCacheHit;
    defer hit.deinit(a);
    try cache.put(a, @splat(3), value);
    try std.testing.expect((try cache.get(a, @splat(2))) == null);
    try std.testing.expectEqual(@as(usize, 4), cache.bytes);
    cache.deinit(a);
    try std.testing.expectEqualSlices(u8, "ab", hit.image.pixels);
    const oversized = Pixels{ .image = .{ .pixels = "abcde", .width = 1, .height = 1 } };
    try cache.put(a, @splat(4), oversized);
    try std.testing.expectEqual(@as(usize, 0), cache.bytes);
}

test "media digest distinguishes pixels geometry and modality" {
    const original = pixelKey(1, &.{ 2, 3 }, "pixels");
    try std.testing.expect(!std.mem.eql(u8, &original, &pixelKey(1, &.{ 3, 2 }, "pixels")));
    try std.testing.expect(!std.mem.eql(u8, &original, &pixelKey(2, &.{ 2, 3 }, "pixels")));
    try std.testing.expect(!std.mem.eql(u8, &original, &pixelKey(1, &.{ 2, 3 }, "edited")));
}
