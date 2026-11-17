//! `readStubMeta` per model dir, kept until that dir's files change: `/v1/models` asks it of every row that is
//! not resident, on every call.

const std = @import("std");
const discovery = @import("discovery.zig");
const StubMeta = discovery.StubMeta;

pub const Cache = struct {
    gpa: std.mem.Allocator = std.heap.c_allocator,
    mutex: std.Io.Mutex = .init,
    rows: std.StringHashMapUnmanaged(Row) = .empty,
    /// Answers that went to the files.
    reads: u64 = 0,

    const Row = struct { stamp: u64, meta: StubMeta };

    pub fn deinit(self: *Cache) void {
        var it = self.rows.keyIterator();
        while (it.next()) |k| self.gpa.free(k.*);
        self.rows.deinit(self.gpa);
    }

    /// `scratch` serves one read's parses.
    pub fn get(self: *Cache, io: std.Io, scratch: std.mem.Allocator, abs_path: []const u8) StubMeta {
        const stamp = stampOf(io, abs_path) orelse return discovery.readStubMeta(io, scratch, abs_path);
        {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            if (self.rows.get(abs_path)) |row| if (row.stamp == stamp) return row.meta;
        }
        // Stamped before the read: a file changed in between leaves a stale stamp, and the next call reads again.
        const meta = discovery.readStubMeta(io, scratch, abs_path);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.reads += 1;
        if (self.rows.getPtr(abs_path)) |row| {
            row.* = .{ .stamp = stamp, .meta = meta };
            return meta;
        }
        const key = self.gpa.dupe(u8, abs_path) catch return meta;
        self.rows.put(self.gpa, key, .{ .stamp = stamp, .meta = meta }) catch self.gpa.free(key);
        return meta;
    }
};

/// Every top-level entry's name, size and mtime, in any order: a file edited, added or removed moves it, and so
/// does one added to or removed from a subfolder (the subfolder's own mtime). Null when the dir cannot be listed.
fn stampOf(io: std.Io, abs_path: []const u8) ?u64 {
    if (!std.fs.path.isAbsolute(abs_path)) return null;
    var dir = std.Io.Dir.openDirAbsolute(io, abs_path, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    var stamp: u64 = 0;
    var it = dir.iterate();
    while (it.next(io) catch return null) |entry| {
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        var h = std.hash.Wyhash.init(0);
        h.update(entry.name);
        h.update(std.mem.asBytes(&st.size));
        h.update(std.mem.asBytes(&st.mtime.nanoseconds));
        stamp +%= h.final();
    }
    return stamp;
}

var shared: Cache = .{};

pub fn stubMeta(io: std.Io, scratch: std.mem.Allocator, abs_path: []const u8) StubMeta {
    return shared.get(io, scratch, abs_path);
}

// ── tests ──

const testing = std.testing;

const dense = "{\"model_type\":\"qwen3\",\"hidden_size\":1024,\"num_hidden_layers\":2,\"vocab_size\":100}";

fn tmpPath(tmp: *std.testing.TmpDir, buf: []u8) ![]const u8 {
    return buf[0..try tmp.dir.realPath(testing.io, buf)];
}

test "stub cache: a dir nobody touched is read once" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = dense });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    var c: Cache = .{ .gpa = testing.allocator };
    defer c.deinit();
    const first = c.get(testing.io, testing.allocator, path);
    try testing.expect(first.found);
    try testing.expectEqual(first, c.get(testing.io, testing.allocator, path));
    try testing.expectEqual(discovery.readStubMeta(testing.io, testing.allocator, path), first);
    try testing.expectEqual(@as(u64, 1), c.reads);
}

test "stub cache: an edited, added or removed file is read again, in a subfolder too" {
    const io = testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = dense });
    try tmp.dir.createDirPath(io, "1_Pooling");
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    var c: Cache = .{ .gpa = testing.allocator };
    defer c.deinit();
    const first = c.get(io, testing.allocator, path);
    try testing.expectEqual(@as(u32, 1024), first.hidden_size);
    try testing.expect(!first.has_chat and !first.has_embedding);

    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3\",\"hidden_size\":64,\"num_hidden_layers\":2,\"vocab_size\":100}" });
    try testing.expectEqual(@as(u32, 64), c.get(io, testing.allocator, path).hidden_size);

    try tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = "{{ messages }}" });
    try testing.expect(c.get(io, testing.allocator, path).has_chat);

    try tmp.dir.writeFile(io, .{ .sub_path = "1_Pooling/config.json", .data = "{}" });
    try testing.expect(c.get(io, testing.allocator, path).has_embedding);

    try tmp.dir.deleteFile(io, "chat_template.jinja");
    try testing.expect(!c.get(io, testing.allocator, path).has_chat);
    try testing.expectEqual(@as(u64, 5), c.reads);
    _ = c.get(io, testing.allocator, path);
    try testing.expectEqual(@as(u64, 5), c.reads);
}

test "stub cache: a dir that is gone answers not found and is not kept" {
    var c: Cache = .{ .gpa = testing.allocator };
    defer c.deinit();
    try testing.expect(!c.get(testing.io, testing.allocator, "/nonexistent/mlx-serve-stub-cache").found);
    try testing.expect(!c.get(testing.io, testing.allocator, "").found);
    try testing.expectEqual(@as(u32, 0), c.rows.count());
}
