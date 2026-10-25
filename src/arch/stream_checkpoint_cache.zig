const std = @import("std");
const disk = @import("../kv_disk_cache.zig");

const max_metadata = 16 << 20;
const max_tokens = 2 << 20;
const max_entries = 4096;
const max_index_bytes = 128 << 20;
const Meta = struct {
    version: u32 = 1,
    identity: [32]u8,
    primary: []const u32,
    fallback: []const u32,
    bytes: u64,
};
const Entry = struct { name: []u8, primary: []u32, fallback: []u32, bytes: u64, touched: i128 };

pub fn prefixKeep(saved: []const u32, prefix: []const u32) u64 {
    if (saved.len == 0) return 0;
    const keep = @min(saved.len, prefix.len);
    if (keep + 1 < saved.len or !std.mem.eql(u32, saved[0..keep], prefix[0..keep])) return 0;
    return keep;
}

/// The plugin owns checkpoint contents. Publishing a directory is the commit point;
/// incomplete writes never become candidates and active request state never belongs to this LRU.
pub const Cache = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    base: []u8,
    root: []u8,
    identity: [32]u8,
    budget: u64,
    entries: std.ArrayList(Entry) = .empty,
    serial: u64 = 0,
    unindexed_bytes: u64 = 0,
    last_saved_bytes: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, model_path: []const u8, layout: []const u8, budget: u64) !Cache {
        const base = try disk.defaultBaseDir(gpa);
        defer gpa.free(base);
        return initAt(gpa, io, base, try modelIdentity(gpa, io, model_path, layout), budget);
    }

    pub fn initAt(gpa: std.mem.Allocator, io: std.Io, base: []const u8, identity: [32]u8, budget: u64) !Cache {
        if (budget == 0) return error.CheckpointCacheDisabled;
        const hex = std.fmt.bytesToHex(identity, .lower);
        var c: Cache = .{ .gpa = gpa, .io = io, .base = try gpa.dupe(u8, base), .root = undefined, .identity = identity, .budget = budget };
        errdefer gpa.free(c.base);
        c.root = try std.fmt.allocPrint(gpa, "{s}/stream-{s}", .{ base, hex });
        errdefer gpa.free(c.root);
        try std.Io.Dir.cwd().createDirPath(io, c.root);
        errdefer c.clearEntries();
        disk.sweepBase(gpa, io, base, "", budget);
        try c.scan();
        c.evictTo(budget);
        c.sweepSiblings();
        return c;
    }

    pub fn deinit(c: *Cache) void {
        c.clearEntries();
        c.gpa.free(c.root);
        c.gpa.free(c.base);
    }

    fn clearEntries(c: *Cache) void {
        for (c.entries.items) |e| c.freeEntry(e);
        c.entries.deinit(c.gpa);
    }

    fn freeEntry(c: *Cache, e: Entry) void {
        c.gpa.free(e.name);
        c.gpa.free(e.primary);
        c.gpa.free(e.fallback);
    }

    fn path(c: *const Cache, name: []const u8) ![]u8 {
        return std.fmt.allocPrint(c.gpa, "{s}/{s}", .{ c.root, name });
    }

    fn scan(c: *Cache) !void {
        var dir = try std.Io.Dir.openDirAbsolute(c.io, c.root, .{ .iterate = true });
        defer dir.close(c.io);
        var it = dir.iterate();
        while (try it.next(c.io)) |item| {
            if (item.kind != .directory or item.name.len < 2 or item.name[0] != 'e') continue;
            const entry_path = try c.path(item.name);
            defer c.gpa.free(entry_path);
            if (std.mem.endsWith(u8, item.name, ".tmp")) {
                c.unindexed_bytes +|= treeBytes(c.gpa, c.io, entry_path) catch 0;
                continue;
            }
            const meta_path = try std.fmt.allocPrint(c.gpa, "{s}/meta.json", .{entry_path});
            defer c.gpa.free(meta_path);
            const raw = readFile(c.gpa, c.io, meta_path, max_metadata) catch {
                c.discardPath(entry_path);
                continue;
            };
            defer c.gpa.free(raw);
            var p = std.json.parseFromSlice(Meta, c.gpa, raw, .{ .allocate = .alloc_always }) catch {
                c.discardPath(entry_path);
                continue;
            };
            defer p.deinit();
            const m = p.value;
            if (m.version != 1 or !std.mem.eql(u8, &m.identity, &c.identity) or
                m.primary.len == 0 or m.primary.len > max_tokens or m.fallback.len > max_tokens or
                c.entries.items.len >= max_entries)
            {
                c.discardPath(entry_path);
                continue;
            }
            const st = std.Io.Dir.cwd().statFile(c.io, meta_path, .{}) catch {
                c.discardPath(entry_path);
                continue;
            };
            const bytes = treeBytes(c.gpa, c.io, entry_path) catch {
                c.discardPath(entry_path);
                continue;
            };
            c.append(item.name, m.primary, m.fallback, bytes, st.mtime.nanoseconds) catch |err| {
                if (err == error.CheckpointIndexFull) {
                    c.discardPath(entry_path);
                    continue;
                }
                return err;
            };
        }
    }

    fn append(c: *Cache, name: []const u8, primary: []const u32, fallback: []const u32, bytes: u64, touched: i128) !void {
        var index_bytes: usize = name.len + (primary.len + fallback.len) * @sizeOf(u32);
        for (c.entries.items) |e| index_bytes += e.name.len + (e.primary.len + e.fallback.len) * @sizeOf(u32);
        if (index_bytes > max_index_bytes) return error.CheckpointIndexFull;
        const n = try c.gpa.dupe(u8, name);
        errdefer c.gpa.free(n);
        const p = try c.gpa.dupe(u32, primary);
        errdefer c.gpa.free(p);
        const f = try c.gpa.dupe(u32, fallback);
        errdefer c.gpa.free(f);
        try c.entries.append(c.gpa, .{ .name = n, .primary = p, .fallback = f, .bytes = bytes, .touched = touched });
    }

    fn held(c: *const Cache) u64 {
        var total: u64 = c.unindexed_bytes;
        for (c.entries.items) |e| total +|= e.bytes;
        return total;
    }

    fn remove(c: *Cache, index: usize) void {
        const e = c.entries.orderedRemove(index);
        const abs = c.path(e.name) catch {
            c.unindexed_bytes +|= e.bytes;
            c.freeEntry(e);
            return;
        };
        defer c.gpa.free(abs);
        std.Io.Dir.cwd().deleteTree(c.io, abs) catch {
            c.unindexed_bytes +|= e.bytes;
        };
        c.freeEntry(e);
    }

    fn discardPath(c: *Cache, abs: []const u8) void {
        const bytes = treeBytes(c.gpa, c.io, abs) catch 0;
        std.Io.Dir.cwd().deleteTree(c.io, abs) catch {
            c.unindexed_bytes +|= bytes;
        };
    }

    fn evictTo(c: *Cache, limit: u64) void {
        while (c.held() > limit and c.entries.items.len > 0) {
            var oldest: usize = 0;
            for (c.entries.items, 0..) |e, i| if (e.touched < c.entries.items[oldest].touched) {
                oldest = i;
            };
            c.remove(oldest);
        }
    }

    fn sweepSiblings(c: *Cache) void {
        disk.sweepBase(c.gpa, c.io, c.base, c.root, c.budget -| c.held());
    }

    pub fn save(c: *Cache, primary: []const u32, fallback: []const u32, ctx: *anyopaque, write: *const fn (*anyopaque, []const u8, [32]u8) anyerror!void) !void {
        c.last_saved_bytes = 0;
        if (primary.len == 0 or primary.len > max_tokens or fallback.len > max_tokens) return error.CheckpointTokenLimit;
        const space = disk.volumeSpace(c.root) orelse return error.CheckpointSpaceUnknown;
        const limit = disk.diskBudgetFromFreeSpace(c.budget, space.free, space.total) orelse return error.CheckpointDiskFull;
        c.serial +%= 1;
        const stamp = std.Io.Timestamp.now(c.io, .real).nanoseconds;
        const name = try std.fmt.allocPrint(c.gpa, "e{d}-{d}", .{ stamp, c.serial });
        defer c.gpa.free(name);
        const dest = try c.path(name);
        defer c.gpa.free(dest);
        const staging = try std.fmt.allocPrint(c.gpa, "{s}.tmp", .{dest});
        defer c.gpa.free(staging);
        try std.Io.Dir.cwd().createDir(c.io, staging, .default_dir);
        defer std.Io.Dir.cwd().deleteTree(c.io, staging) catch {};
        try write(ctx, staging, c.identity);
        const payload_bytes = try treeBytes(c.gpa, c.io, staging);
        const metadata = try std.json.Stringify.valueAlloc(c.gpa, Meta{
            .identity = c.identity,
            .primary = primary,
            .fallback = fallback,
            .bytes = payload_bytes,
        }, .{});
        defer c.gpa.free(metadata);
        if (metadata.len > max_metadata) return error.CheckpointTokenLimit;
        const total = std.math.add(u64, payload_bytes, metadata.len) catch return error.CheckpointDiskFull;
        if (total > limit) return error.CheckpointDiskFull;
        const meta_path = try std.fmt.allocPrint(c.gpa, "{s}/meta.json", .{staging});
        defer c.gpa.free(meta_path);
        try std.Io.Dir.cwd().writeFile(c.io, .{ .sub_path = meta_path, .data = metadata });
        c.evictTo(limit - total);
        if (c.held() > limit - total) return error.CheckpointDiskFull;
        while (c.entries.items.len >= max_entries) c.remove(0);
        try std.Io.Dir.renameAbsolute(staging, dest, c.io);
        errdefer std.Io.Dir.cwd().deleteTree(c.io, dest) catch {};
        try c.append(name, primary, fallback, total, stamp);
        c.last_saved_bytes = total;
        // Superseded checkpoints with the same complete prefix only consume the disk budget.
        var i: usize = 0;
        while (i + 1 < c.entries.items.len) {
            if (std.mem.eql(u32, c.entries.items[i].primary, primary) and
                std.mem.eql(u32, c.entries.items[i].fallback, fallback)) c.remove(i) else i += 1;
        }
        c.sweepSiblings();
    }

    /// Only CheckpointNoCompatiblePrefix guarantees the read left the destination untouched.
    /// Other failures may follow eviction or GPU work and must terminate this restore attempt.
    pub fn restoreBest(c: *Cache, prefix: []const u32, min_keep: u64, ctx: *anyopaque, read: *const fn (*anyopaque, []const u8, []const u32, [32]u8) anyerror!u64) !?u64 {
        const Candidate = struct { index: usize, keep: u64 };
        var candidates: std.ArrayList(Candidate) = .empty;
        defer candidates.deinit(c.gpa);
        for (c.entries.items, 0..) |e, i| {
            const n = @max(prefixKeep(e.primary, prefix), prefixKeep(e.fallback, prefix));
            if (n > min_keep) try candidates.append(c.gpa, .{ .index = i, .keep = n });
        }
        std.mem.sort(Candidate, candidates.items, {}, struct {
            fn less(_: void, left: Candidate, right: Candidate) bool {
                return if (left.keep == right.keep) left.index < right.index else left.keep > right.keep;
            }
        }.less);
        for (candidates.items) |candidate| {
            const i = candidate.index;
            const dir = try c.path(c.entries.items[i].name);
            defer c.gpa.free(dir);
            const actual = read(ctx, dir, prefix, c.identity) catch |err| {
                if (err == error.CheckpointNoCompatiblePrefix) continue;
                return err;
            };
            if (actual == 0 or actual > candidate.keep or actual > prefix.len)
                return error.CheckpointRestoreMismatch;
            c.entries.items[i].touched = std.Io.Timestamp.now(c.io, .real).nanoseconds;
            const meta = std.fmt.allocPrint(c.gpa, "{s}/meta.json", .{dir}) catch return actual;
            defer c.gpa.free(meta);
            std.Io.Dir.cwd().setTimestamps(c.io, meta, .{ .modify_timestamp = .now }) catch {};
            return actual;
        }
        return null;
    }

};

fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8, limit: usize) ![]u8 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var buf: [8192]u8 = undefined;
    var reader = f.reader(io, &buf);
    return reader.interface.allocRemaining(gpa, .limited(limit));
}

fn treeBytes(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !u64 {
    var dir = try std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(gpa);
    defer walker.deinit();
    var bytes: u64 = 0;
    while (try walker.next(io)) |e| {
        if (e.kind == .directory) continue;
        if (e.kind != .file) return error.CheckpointUnexpectedFile;
        const stat = try e.dir.statFile(io, e.basename, .{});
        bytes = try std.math.add(u64, bytes, stat.size);
    }
    return bytes;
}

/// Contents pin configuration/manifests; file metadata detects ordinary weight replacement
/// without reading the entire expert bank. The plugin also validates its numerical layout.
fn modelIdentity(gpa: std.mem.Allocator, io: std.Io, model_path: []const u8, layout: []const u8) ![32]u8 {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_path, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(gpa);
    defer walker.deinit();
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    while (try walker.next(io)) |e| {
        if (e.kind != .file and e.kind != .sym_link) continue;
        const n = try gpa.dupe(u8, e.path);
        errdefer gpa.free(n);
        try names.append(gpa, n);
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn less(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.less);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("mlx-stream-checkpoint-v1\x00");
    h.update(model_path);
    h.update(layout);
    for (names.items) |name| {
        const st = try dir.statFile(io, name, .{});
        if (st.kind != .file) continue;
        h.update(name);
        h.update(std.mem.asBytes(&st.size));
        h.update(std.mem.asBytes(&st.mtime.nanoseconds));
        if (std.mem.endsWith(u8, name, ".json")) {
            const abs = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ model_path, name });
            defer gpa.free(abs);
            const bytes = try readFile(gpa, io, abs, 64 << 20);
            defer gpa.free(bytes);
            h.update(bytes);
        }
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

test "stream checkpoint cache: full and one-token-short matches preserve prompt fallback" {
    try std.testing.expectEqual(@as(u64, 4), prefixKeep(&.{ 1, 2, 3, 4 }, &.{ 1, 2, 3, 4, 9 }));
    try std.testing.expectEqual(@as(u64, 3), prefixKeep(&.{ 1, 2, 3, 4 }, &.{ 1, 2, 3 }));
    try std.testing.expectEqual(@as(u64, 0), prefixKeep(&.{ 1, 2, 3, 4 }, &.{ 1, 2, 8, 4 }));
    try std.testing.expectEqual(@as(u64, 0), prefixKeep(&.{}, &.{1}));
}

const Fixture = struct {
    io: std.Io,
    fail_write: bool = false,
    fail_read: bool = false,
    reads: usize = 0,

    fn write(raw: *anyopaque, dir: []const u8, _: [32]u8) !void {
        const self: *Fixture = @ptrCast(@alignCast(raw));
        var d = try std.Io.Dir.openDirAbsolute(self.io, dir, .{});
        defer d.close(self.io);
        try d.writeFile(self.io, .{ .sub_path = "tensor.bin", .data = "checkpoint payload" });
        if (self.fail_write) return error.InjectedWriteFailure;
    }

    fn read(raw: *anyopaque, _: []const u8, prefix: []const u32, _: [32]u8) !u64 {
        const self: *Fixture = @ptrCast(@alignCast(raw));
        self.reads += 1;
        if (self.fail_read) return error.InjectedCorruption;
        return @min(prefix.len, if (prefix.len > 2 and prefix[2] == 99) @as(usize, 2) else 4);
    }
};

test "stream checkpoint cache: restart restores both prefixes and read failures preserve disk entries" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [4096]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var f: Fixture = .{ .io = io };
    var cache = try Cache.initAt(a, io, root, @splat(7), 2 << 30);
    try cache.save(&.{ 1, 2, 3, 4 }, &.{ 1, 2 }, &f, Fixture.write);
    try std.testing.expectEqual(@as(usize, 1), cache.entries.items.len);
    cache.deinit();
    cache = try Cache.initAt(a, io, root, @splat(7), 2 << 30);
    defer cache.deinit();
    try std.testing.expectEqual(@as(?u64, 4), try cache.restoreBest(&.{ 1, 2, 3, 4, 5 }, 0, &f, Fixture.read));
    try std.testing.expectEqual(@as(?u64, 2), try cache.restoreBest(&.{ 1, 2, 99 }, 0, &f, Fixture.read));
    try std.testing.expectEqual(@as(?u64, null), try cache.restoreBest(&.{ 1, 2, 3, 4 }, 4, &f, Fixture.read));
    try std.testing.expectEqual(@as(usize, 2), f.reads);
    f.fail_read = true;
    try std.testing.expectError(error.InjectedCorruption, cache.restoreBest(&.{ 1, 2, 3, 4 }, 0, &f, Fixture.read));
    try std.testing.expectEqual(@as(usize, 1), cache.entries.items.len);
}

test "stream checkpoint cache: failed writes are invisible and LRU eviction respects held bytes" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [4096]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var f: Fixture = .{ .io = io, .fail_write = true };
    var cache = try Cache.initAt(a, io, root, @splat(8), 2 << 30);
    defer cache.deinit();
    try std.testing.expectError(error.InjectedWriteFailure, cache.save(&.{ 1, 2, 3, 4 }, &.{ 1, 2 }, &f, Fixture.write));
    try std.testing.expectEqual(@as(u64, 0), cache.held());
    f.fail_write = false;
    try cache.save(&.{ 1, 2, 3, 4 }, &.{ 1, 2 }, &f, Fixture.write);
    try cache.save(&.{ 7, 8, 9, 10 }, &.{ 7, 8 }, &f, Fixture.write);
    const keep_bytes = cache.entries.items[0].bytes;
    cache.entries.items[0].touched = std.math.maxInt(i128);
    cache.evictTo(keep_bytes);
    try std.testing.expectEqual(@as(usize, 1), cache.entries.items.len);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, cache.entries.items[0].primary);
    try std.testing.expect(cache.held() <= keep_bytes);
}

test "stream checkpoint cache: identity changes with model contents and numerical layout" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [4096]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"version\":1}" });
    const first = try modelIdentity(a, io, root, "layout-v1");
    const changed_layout = try modelIdentity(a, io, root, "layout-v2");
    try std.testing.expect(!std.mem.eql(u8, &first, &changed_layout));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"version\":2}" });
    const changed_model = try modelIdentity(a, io, root, "layout-v1");
    try std.testing.expect(!std.mem.eql(u8, &first, &changed_model));
}

test "stream checkpoint cache: an unpublished staging directory cannot restore after restart" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [4096]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var f: Fixture = .{ .io = io };
    var cache = try Cache.initAt(a, io, root, @splat(9), 2 << 30);
    try cache.save(&.{ 1, 2, 3, 4 }, &.{ 1, 2 }, &f, Fixture.write);
    const published = try cache.path(cache.entries.items[0].name);
    defer a.free(published);
    const staging = try std.fmt.allocPrint(a, "{s}.tmp", .{published});
    defer a.free(staging);
    cache.deinit();
    try std.Io.Dir.renameAbsolute(published, staging, io);
    cache = try Cache.initAt(a, io, root, @splat(9), 2 << 30);
    defer cache.deinit();
    try std.testing.expectEqual(@as(usize, 0), cache.entries.items.len);
    try std.testing.expectEqual(@as(?u64, null), try cache.restoreBest(&.{ 1, 2, 3, 4 }, 0, &f, Fixture.read));
    try std.testing.expectEqual(@as(usize, 0), f.reads);
}

test "stream checkpoint cache: incompatible candidates fall back without deleting other configurations" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [4096]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var cache = try Cache.initAt(a, io, root, @splat(12), 2 << 30);
    defer cache.deinit();
    try cache.append("old-long", &.{ 1, 2, 3, 4, 5 }, &.{}, 1, 0);
    try cache.append("old-short", &.{ 1, 2, 3 }, &.{}, 1, 0);
    try cache.append("new-short", &.{ 1, 2, 3 }, &.{}, 1, 0);
    const Read = struct {
        calls: usize = 0,
        fail: ?anyerror = null,
        all_miss: bool = false,
        zero: bool = false,
        fn call(raw: *anyopaque, dir: []const u8, _: []const u32, _: [32]u8) !u64 {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.calls += 1;
            if (self.fail) |err| return err;
            if (self.zero) return 0;
            if (self.all_miss or !std.mem.endsWith(u8, dir, "/new-short")) return error.CheckpointNoCompatiblePrefix;
            return 3;
        }
    };
    var read: Read = .{};
    try std.testing.expectEqual(@as(?u64, 3), try cache.restoreBest(&.{ 1, 2, 3, 4, 5, 6 }, 2, &read, Read.call));
    try std.testing.expectEqual(@as(usize, 3), read.calls);
    try std.testing.expectEqual(@as(usize, 3), cache.entries.items.len);
    read = .{ .all_miss = true };
    try std.testing.expectEqual(@as(?u64, null), try cache.restoreBest(&.{ 1, 2, 3, 4, 5, 6 }, 2, &read, Read.call));
    try std.testing.expectEqual(@as(usize, 3), read.calls);
    try std.testing.expectEqual(@as(usize, 3), cache.entries.items.len);
    read = .{ .fail = error.OutOfMemory };
    try std.testing.expectError(error.OutOfMemory, cache.restoreBest(&.{ 1, 2, 3, 4, 5, 6 }, 2, &read, Read.call));
    try std.testing.expectEqual(@as(usize, 1), read.calls);
    try std.testing.expectEqual(@as(usize, 3), cache.entries.items.len);
    read = .{ .zero = true };
    try std.testing.expectError(error.CheckpointRestoreMismatch, cache.restoreBest(&.{ 1, 2, 3, 4, 5, 6 }, 2, &read, Read.call));
    try std.testing.expectEqual(@as(usize, 1), read.calls);
    try std.testing.expectEqual(@as(usize, 3), cache.entries.items.len);
}
