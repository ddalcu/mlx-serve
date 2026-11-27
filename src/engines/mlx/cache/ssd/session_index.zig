//! The tip of each session, persisted by itself: a prefix entry is shared between sessions, so its record cannot
//! carry one session's name. A header, then an append-only run of fixed records; a damaged or half-written record is
//! skipped. A file without the current header holds identities of another scheme: it is set aside on open.

const std = @import("std");
const log = @import("../../../../sdk/util/log.zig");

pub const Tip = struct { session: u64, id: u64, n: u64, variant: u64, seq: u64 };

pub const RECORD_BYTES: usize = 48;
const HEADER_BYTES: usize = 16;
const MAGIC: u64 = 0x7365_7373_6964_7831;
/// Bumped whenever what a record's `id` hashes changes.
const FORMAT_VERSION: u64 = 1;
pub const DEFAULT_CAP: usize = 65536;
/// How many tips of distinct boundaries the file keeps for one session: its newest and the ones a lookup falls back to.
pub const TIPS_PER_SESSION: usize = 4;
/// The file is rewritten, newest `cap` sessions only, once it holds this many records per cap.
const COMPACT_FACTOR: usize = 2 * TIPS_PER_SESSION;
const MAX_READ_BYTES: usize = 256 * 1024 * 1024;
const CHECK_SEED: u64 = 0x7365_7373_7469_7073;

fn checksum(body: []const u8) u64 {
    return std.hash.Wyhash.hash(CHECK_SEED, body);
}

fn encode(tip: Tip, out: *[RECORD_BYTES]u8) void {
    const words = [5]u64{ tip.session, tip.id, tip.n, tip.variant, tip.seq };
    for (words, 0..) |w, i| std.mem.writeInt(u64, out[i * 8 ..][0..8], w, .little);
    std.mem.writeInt(u64, out[40..48], checksum(out[0..40]), .little);
}

fn decode(record: *const [RECORD_BYTES]u8) ?Tip {
    if (std.mem.readInt(u64, record[40..48], .little) != checksum(record[0..40])) return null;
    var words: [5]u64 = undefined;
    for (&words, 0..) |*w, i| w.* = std.mem.readInt(u64, record[i * 8 ..][0..8], .little);
    return .{ .session = words[0], .id = words[1], .n = words[2], .variant = words[3], .seq = words[4] };
}

fn headerBytes() [HEADER_BYTES]u8 {
    var out: [HEADER_BYTES]u8 = undefined;
    std.mem.writeInt(u64, out[0..8], MAGIC, .little);
    std.mem.writeInt(u64, out[8..16], FORMAT_VERSION, .little);
    return out;
}

/// The records of a file with the current header; null for any other bytes.
fn bodyOf(bytes: []const u8) ?[]const u8 {
    const header = headerBytes();
    if (bytes.len < HEADER_BYTES or !std.mem.eql(u8, bytes[0..HEADER_BYTES], &header)) return null;
    return bytes[HEADER_BYTES..];
}

fn bySeq(_: void, a: Tip, b: Tip) bool {
    return if (a.seq != b.seq) a.seq < b.seq else a.session < b.session;
}

fn newerFirst(_: void, a: Tip, b: Tip) bool {
    return a.seq > b.seq;
}

/// One session's newest tip of each of its newest `TIPS_PER_SESSION` boundaries, newest first (a boundary claimed
/// again is one tip with a newer `seq`).
const Kept = struct {
    tips: [TIPS_PER_SESSION]Tip = undefined,
    len: usize = 0,

    fn add(self: *Kept, tip: Tip) void {
        if (self.indexOf(tip.id)) |i| {
            if (tip.seq >= self.tips[i].seq) self.tips[i] = tip;
        } else if (self.len < TIPS_PER_SESSION) {
            self.tips[self.len] = tip;
            self.len += 1;
        } else if (tip.seq >= self.tips[self.len - 1].seq) {
            self.tips[self.len - 1] = tip;
        }
        std.mem.sort(Tip, self.tips[0..self.len], {}, newerFirst);
    }

    fn indexOf(self: *const Kept, id: u64) ?usize {
        for (self.tips[0..self.len], 0..) |held, i| if (held.id == id) return i;
        return null;
    }
};

pub const Index = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []u8,
    cap: usize,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, path: []const u8, cap: usize) !Index {
        var index: Index = .{ .allocator = allocator, .io = io, .path = try allocator.dupe(u8, path), .cap = cap };
        errdefer index.deinit();
        try index.setAsideStale();
        return index;
    }

    fn setAsideStale(self: *Index) !void {
        const bytes = self.readAll() orelse return;
        defer self.allocator.free(bytes);
        if (bytes.len == 0 or bodyOf(bytes) != null) return;
        const aside = try std.fmt.allocPrint(self.allocator, "{s}.old", .{self.path});
        defer self.allocator.free(aside);
        try std.Io.Dir.renameAbsolute(self.path, aside, self.io);
        log.warn("[session-index] {s} is not the current format: set aside as {s}, sessions start empty\n", .{ self.path, aside });
    }

    pub fn deinit(self: *Index) void {
        self.allocator.free(self.path);
    }

    /// The newest tip of each of the newest `cap` sessions, oldest first. The file keeps each of those sessions' newest
    /// tip of each of `TIPS_PER_SESSION` boundaries, so a session whose newest record is gone has others to fall back
    /// to (`findExcluding`); a file holding anything else (older tips, sessions past the cap, a damaged record) is
    /// rewritten to just those.
    pub fn load(self: *Index, out: *std.ArrayList(Tip)) !void {
        const file = self.readAll() orelse return;
        defer self.allocator.free(file);
        const bytes = bodyOf(file) orelse return;
        var sessions = std.AutoHashMap(u64, Kept).init(self.allocator);
        defer sessions.deinit();
        var clean = bytes.len % RECORD_BYTES == 0;
        var records: usize = 0;
        var at: usize = 0;
        while (at + RECORD_BYTES <= bytes.len) : (at += RECORD_BYTES) {
            const tip = decode(bytes[at..][0..RECORD_BYTES]) orelse {
                clean = false;
                continue;
            };
            records += 1;
            const held = try sessions.getOrPut(tip.session);
            if (!held.found_existing) held.value_ptr.* = .{};
            held.value_ptr.add(tip);
        }
        var newest = std.ArrayList(Tip).empty;
        defer newest.deinit(self.allocator);
        var it = sessions.valueIterator();
        while (it.next()) |session| try newest.append(self.allocator, session.tips[0]);
        std.mem.sort(Tip, newest.items, {}, bySeq);
        const kept = newest.items[newest.items.len -| self.cap..];
        var survivors = std.ArrayList(Tip).empty;
        defer survivors.deinit(self.allocator);
        for (kept) |head| {
            const session = sessions.getPtr(head.session).?;
            try survivors.appendSlice(self.allocator, session.tips[0..session.len]);
        }
        std.mem.sort(Tip, survivors.items, {}, bySeq);
        try out.appendSlice(self.allocator, kept);
        if (!clean or records != survivors.items.len) self.rewrite(survivors.items);
    }

    /// The newest tip the file holds for `session` whose boundary is not in `tried`: where a lookup falls back to when
    /// the records of the tips it has tried are gone.
    pub fn findExcluding(self: *Index, session: u64, tried: []const u64) ?Tip {
        const file = self.readAll() orelse return null;
        defer self.allocator.free(file);
        const bytes = bodyOf(file) orelse return null;
        var best: ?Tip = null;
        var at: usize = 0;
        while (at + RECORD_BYTES <= bytes.len) : (at += RECORD_BYTES) {
            const tip = decode(bytes[at..][0..RECORD_BYTES]) orelse continue;
            if (tip.session != session or std.mem.indexOfScalar(u64, tried, tip.id) != null) continue;
            if (best == null or best.?.seq <= tip.seq) best = tip;
        }
        return best;
    }

    /// The newest tip the file holds for `session`.
    pub fn find(self: *Index, session: u64) ?Tip {
        const file = self.readAll() orelse return null;
        defer self.allocator.free(file);
        const bytes = bodyOf(file) orelse return null;
        var best: ?Tip = null;
        var at: usize = 0;
        while (at + RECORD_BYTES <= bytes.len) : (at += RECORD_BYTES) {
            const tip = decode(bytes[at..][0..RECORD_BYTES]) orelse continue;
            if (tip.session != session) continue;
            if (best == null or best.?.seq <= tip.seq) best = tip;
        }
        return best;
    }

    pub fn append(self: *Index, tip: Tip) !void {
        if (self.recordCount() >= COMPACT_FACTOR * self.cap) self.compact();
        const f = try std.Io.Dir.createFileAbsolute(self.io, self.path, .{ .truncate = false });
        defer f.close(self.io);
        var len = try f.length(self.io);
        if (len < HEADER_BYTES) {
            try f.writePositionalAll(self.io, &headerBytes(), 0);
            len = HEADER_BYTES;
        }
        var record: [RECORD_BYTES]u8 = undefined;
        encode(tip, &record);
        try f.writePositionalAll(self.io, &record, HEADER_BYTES + (len - HEADER_BYTES) / RECORD_BYTES * RECORD_BYTES);
    }

    fn recordCount(self: *Index) usize {
        const f = std.Io.Dir.openFileAbsolute(self.io, self.path, .{}) catch return 0;
        defer f.close(self.io);
        return @intCast(((f.length(self.io) catch return 0) -| HEADER_BYTES) / RECORD_BYTES);
    }

    fn compact(self: *Index) void {
        var tips = std.ArrayList(Tip).empty;
        defer tips.deinit(self.allocator);
        self.load(&tips) catch |err| log.warn("[session-index] compaction failed: {s}\n", .{@errorName(err)});
    }

    fn readAll(self: *Index) ?[]u8 {
        const f = std.Io.Dir.openFileAbsolute(self.io, self.path, .{}) catch return null;
        defer f.close(self.io);
        var buffer: [8192]u8 = undefined;
        var reader = f.reader(self.io, &buffer);
        return reader.interface.allocRemaining(self.allocator, .limited(MAX_READ_BYTES)) catch null;
    }

    fn rewrite(self: *Index, tips: []const Tip) void {
        self.rewriteFile(tips) catch |err| log.warn("[session-index] rewrite failed: {s}\n", .{@errorName(err)});
    }

    fn rewriteFile(self: *Index, tips: []const Tip) !void {
        const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}.tmp", .{self.path});
        defer self.allocator.free(tmp_path);
        {
            const f = try std.Io.Dir.createFileAbsolute(self.io, tmp_path, .{});
            defer f.close(self.io);
            var buffer: [8192]u8 = undefined;
            var writer = f.writer(self.io, &buffer);
            try writer.interface.writeAll(&headerBytes());
            var record: [RECORD_BYTES]u8 = undefined;
            for (tips) |tip| {
                encode(tip, &record);
                try writer.interface.writeAll(&record);
            }
            try writer.interface.flush();
        }
        try std.Io.Dir.renameAbsolute(tmp_path, self.path, self.io);
    }
};

const testing = std.testing;

fn tmpPath(tmp: *testing.TmpDir, buf: []u8) ![]const u8 {
    const n = try tmp.dir.realPath(testing.io, buf);
    const tail = try std.fmt.bufPrint(buf[n..], "/session-tips.bin", .{});
    return buf[0 .. n + tail.len];
}

fn tipOf(session: u64, seq: u64) Tip {
    return .{ .session = session, .id = session * 100 + seq, .n = 512 + seq, .variant = 0, .seq = seq };
}

fn loaded(index: *Index) !std.ArrayList(Tip) {
    var out = std.ArrayList(Tip).empty;
    errdefer out.deinit(testing.allocator);
    try index.load(&out);
    return out;
}

test "a session index returns each session's newest tip, oldest first, after a reopen" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    {
        var index = try Index.init(testing.allocator, testing.io, path, 16);
        defer index.deinit();
        try index.append(tipOf(5, 1));
        try index.append(tipOf(6, 2));
        try index.append(tipOf(5, 3));
    }
    var index = try Index.init(testing.allocator, testing.io, path, 16);
    defer index.deinit();
    var out = try loaded(&index);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), out.items.len);
    try testing.expectEqual(tipOf(6, 2), out.items[0]);
    try testing.expectEqual(tipOf(5, 3), out.items[1]);
    try testing.expectEqual(tipOf(5, 3), index.find(5).?);
    try testing.expect(index.find(7) == null);
    try testing.expectEqual(tipOf(5, 1), index.findExcluding(5, &.{tipOf(5, 3).id}).?);
    try testing.expect(index.findExcluding(5, &.{ tipOf(5, 3).id, tipOf(5, 1).id }) == null);
}

test "a session index keeps a session's newest tips of distinct boundaries through compaction" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    var index = try Index.init(testing.allocator, testing.io, path, 4);
    defer index.deinit();
    for (1..7) |seq| try index.append(tipOf(5, seq));
    var out = try loaded(&index);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), out.items.len);
    try testing.expectEqual(tipOf(5, 6), out.items[0]);
    var tried: [TIPS_PER_SESSION]u64 = undefined;
    for (0..TIPS_PER_SESSION - 1) |i| {
        tried[i] = tipOf(5, 6 - i).id;
        try testing.expectEqual(tipOf(5, 5 - i), index.findExcluding(5, tried[0 .. i + 1]).?);
    }
    tried[TIPS_PER_SESSION - 1] = tipOf(5, 6 - (TIPS_PER_SESSION - 1)).id;
    try testing.expect(index.findExcluding(5, &tried) == null);
    const f = try std.Io.Dir.openFileAbsolute(testing.io, path, .{});
    defer f.close(testing.io);
    try testing.expectEqual(@as(u64, HEADER_BYTES + TIPS_PER_SESSION * RECORD_BYTES), try f.length(testing.io));
}

test "a session index that holds C, A, B, A keeps C once compacted, and findExcluding walks A, B, C" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    var index = try Index.init(testing.allocator, testing.io, path, 4);
    defer index.deinit();
    const written = [_]u64{ 100, 200, 300, 200 };
    for (written, 1..) |id, seq| try index.append(.{ .session = 5, .id = id, .n = 512, .variant = 0, .seq = seq });
    var out = try loaded(&index);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(u64, 200), index.findExcluding(5, &.{}).?.id);
    try testing.expectEqual(@as(u64, 300), index.findExcluding(5, &.{200}).?.id);
    try testing.expectEqual(@as(u64, 100), index.findExcluding(5, &.{ 200, 300 }).?.id);
    try testing.expect(index.findExcluding(5, &.{ 200, 300, 100 }) == null);
}

test "a session index skips a half-written tail and a damaged record, and appends after them" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    {
        var index = try Index.init(testing.allocator, testing.io, path, 16);
        defer index.deinit();
        try index.append(tipOf(1, 1));
        try index.append(tipOf(2, 2));
        try index.append(tipOf(3, 3));
    }
    {
        const f = try std.Io.Dir.openFileAbsolute(testing.io, path, .{ .mode = .read_write });
        defer f.close(testing.io);
        try f.writePositionalAll(testing.io, "\xff", HEADER_BYTES + RECORD_BYTES + 9);
        try f.setLength(testing.io, HEADER_BYTES + 3 * RECORD_BYTES - 10);
    }
    var index = try Index.init(testing.allocator, testing.io, path, 16);
    defer index.deinit();
    {
        var out = try loaded(&index);
        defer out.deinit(testing.allocator);
        try testing.expectEqual(@as(usize, 1), out.items.len);
        try testing.expectEqual(tipOf(1, 1), out.items[0]);
    }
    try index.append(tipOf(4, 4));
    var again = try Index.init(testing.allocator, testing.io, path, 16);
    defer again.deinit();
    var out = try loaded(&again);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), out.items.len);
    try testing.expectEqual(tipOf(4, 4), out.items[1]);
}

test "a session index counts a tip of one boundary once, so the fallback is another boundary" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    var index = try Index.init(testing.allocator, testing.io, path, 8);
    defer index.deinit();
    const first: Tip = .{ .session = 5, .id = 100, .n = 512, .variant = 0, .seq = 1 };
    for ([_]u64{ 2, 3, 4 }) |seq| try index.append(.{ .session = 5, .id = 200, .n = 513, .variant = 0, .seq = seq });
    try index.append(first);
    var out = try loaded(&index);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), out.items.len);
    try testing.expectEqual(@as(u64, 4), out.items[0].seq);
    try testing.expectEqual(first, index.findExcluding(5, &.{200}).?);
    const f = try std.Io.Dir.openFileAbsolute(testing.io, path, .{});
    defer f.close(testing.io);
    try testing.expectEqual(@as(u64, HEADER_BYTES + 2 * RECORD_BYTES), try f.length(testing.io));
}

test "a session index written before the format header is set aside on open, and new tips persist and reload" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    {
        const f = try std.Io.Dir.createFileAbsolute(testing.io, path, .{});
        defer f.close(testing.io);
        var record: [RECORD_BYTES]u8 = undefined;
        encode(tipOf(5, 1), &record);
        try f.writePositionalAll(testing.io, &record, 0);
        encode(tipOf(6, 2), &record);
        try f.writePositionalAll(testing.io, &record, RECORD_BYTES);
    }
    {
        var index = try Index.init(testing.allocator, testing.io, path, 16);
        defer index.deinit();
        try testing.expect(index.find(5) == null);
        var out = try loaded(&index);
        defer out.deinit(testing.allocator);
        try testing.expectEqual(@as(usize, 0), out.items.len);
        try index.append(tipOf(7, 3));
    }
    var aside_buf: [520]u8 = undefined;
    const aside = try std.fmt.bufPrint(&aside_buf, "{s}.old", .{path});
    const kept = try std.Io.Dir.openFileAbsolute(testing.io, aside, .{});
    defer kept.close(testing.io);
    try testing.expectEqual(@as(u64, 2 * RECORD_BYTES), try kept.length(testing.io));
    var index = try Index.init(testing.allocator, testing.io, path, 16);
    defer index.deinit();
    var out = try loaded(&index);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), out.items.len);
    try testing.expectEqual(tipOf(7, 3), out.items[0]);
}

test "a session index whose first header write stopped short rewrites the header on the next append" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    var index = try Index.init(testing.allocator, testing.io, path, 16);
    defer index.deinit();
    for ([_]usize{ 1, HEADER_BYTES / 2, HEADER_BYTES - 1 }) |partial| {
        {
            const f = try std.Io.Dir.createFileAbsolute(testing.io, path, .{});
            defer f.close(testing.io);
            try f.writePositionalAll(testing.io, headerBytes()[0..partial], 0);
        }
        try index.append(tipOf(7, partial));
        try testing.expectEqual(tipOf(7, partial), index.find(7).?);
        var out = try loaded(&index);
        defer out.deinit(testing.allocator);
        try testing.expectEqual(@as(usize, 1), out.items.len);
        try testing.expectEqual(tipOf(7, partial), out.items[0]);
        const f = try std.Io.Dir.openFileAbsolute(testing.io, path, .{});
        defer f.close(testing.io);
        try testing.expectEqual(@as(u64, HEADER_BYTES + RECORD_BYTES), try f.length(testing.io));
    }
}

test "a session index keeps the newest sessions at its cap, and find reads the file" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    var index = try Index.init(testing.allocator, testing.io, path, 3);
    defer index.deinit();
    for (1..15) |s| try index.append(tipOf(s, s));
    try testing.expectEqual(tipOf(14, 14), index.find(14).?);
    var out = try loaded(&index);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 3), out.items.len);
    try testing.expectEqual(tipOf(12, 12), out.items[0]);
    try testing.expectEqual(tipOf(14, 14), out.items[2]);
    try testing.expectEqual(tipOf(12, 12), index.find(12).?);
    try testing.expect(index.find(2) == null);
    const f = try std.Io.Dir.openFileAbsolute(testing.io, path, .{});
    defer f.close(testing.io);
    try testing.expectEqual(@as(u64, HEADER_BYTES + 3 * RECORD_BYTES), try f.length(testing.io));
}
