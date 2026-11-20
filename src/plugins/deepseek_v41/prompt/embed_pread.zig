//! The token embedding table (`embed.weight`, bf16 [vocab, hidden]: 1.32 GB in the full pack) read from
//! the pack by pread at each lookup instead of held resident. The model leaves it out of the load and reads its
//! rows here, so the streamed bank, sized from the memory in use when it
//! opens, takes the bytes as slot rows. A decode round looks up 6 to 7 rows (the verify's ids and the draft's first id), a
//! 16K prompt its distinct ids; the preads run in parallel and the OS page cache keeps the hot rows. The rows are the
//! table's bytes, so the served bytes are the same.

const std = @import("std");
const builtin = @import("builtin");

/// The table's tensor name in the pack (its index's `weight_map` names the shard).
pub const tensor = "embed.weight";

/// `tensor` in its shard: bf16 rows of `hidden` values, read by pread.
pub const Table = struct {
    fd: c_int,
    /// The tensor's first byte in its shard.
    off: u64,
    rows: u32,
    hidden: u32,
    row_bytes: usize,

    pub const OpenError = error{ EmbedPreadIndex, EmbedPreadOpen, EmbedPreadHeader, EmbedPreadTensor, OutOfMemory };

    /// The pack in `dir`: its index names the shard holding `tensor`, whose header must give it as BF16 [rows, hidden].
    pub fn open(gpa: std.mem.Allocator, dir: []const u8, rows: u32, hidden: u32) OpenError!Table {
        // Its reads are POSIX `pread`s: Windows keeps the embedding in memory.
        if (comptime builtin.os.tag == .windows) return error.EmbedPreadOpen;
        const shard = try shardOf(gpa, dir);
        defer gpa.free(shard);
        const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, shard }, 0);
        defer gpa.free(path);
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.EmbedPreadOpen;
        errdefer _ = std.c.close(fd);
        // Rows are read at random: read-ahead would pull their neighbours in.
        if (comptime builtin.os.tag.isDarwin()) _ = std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0));
        var hl: [8]u8 = undefined;
        if (!preadAll(fd, &hl, 0)) return error.EmbedPreadHeader;
        const hlen = std.mem.readInt(u64, &hl, .little);
        if (hlen == 0 or hlen > 256 << 20) return error.EmbedPreadHeader;
        const hbuf = try gpa.alloc(u8, @intCast(hlen));
        defer gpa.free(hbuf);
        if (!preadAll(fd, hbuf, 8)) return error.EmbedPreadHeader;
        const parsed = std.json.parseFromSlice(std.json.Value, gpa, hbuf, .{}) catch return error.EmbedPreadHeader;
        defer parsed.deinit();
        if (parsed.value != .object) return error.EmbedPreadHeader;
        const t = parsed.value.object.get(tensor) orelse return error.EmbedPreadTensor;
        if (t != .object) return error.EmbedPreadTensor;
        const dt = t.object.get("dtype") orelse return error.EmbedPreadTensor;
        if (dt != .string or !std.mem.eql(u8, dt.string, "BF16")) return error.EmbedPreadTensor;
        const shape = t.object.get("shape") orelse return error.EmbedPreadTensor;
        if (shape != .array or shape.array.items.len != 2) return error.EmbedPreadTensor;
        const shape_rows = intOf(shape.array.items[0]) orelse return error.EmbedPreadTensor;
        const shape_cols = intOf(shape.array.items[1]) orelse return error.EmbedPreadTensor;
        if (shape_rows != rows or shape_cols != hidden) return error.EmbedPreadTensor;
        const offs = t.object.get("data_offsets") orelse return error.EmbedPreadTensor;
        if (offs != .array or offs.array.items.len != 2) return error.EmbedPreadTensor;
        const a = intOf(offs.array.items[0]) orelse return error.EmbedPreadTensor;
        const b = intOf(offs.array.items[1]) orelse return error.EmbedPreadTensor;
        const row_bytes: usize = @as(usize, hidden) * 2;
        if (b < a or b - a != @as(u64, rows) * row_bytes) return error.EmbedPreadTensor;
        const off = 8 + hlen + a;
        const size = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (size < 0 or @as(u64, @intCast(size)) < off + (b - a)) return error.EmbedPreadTensor;
        return .{ .fd = fd, .off = off, .rows = rows, .hidden = hidden, .row_bytes = row_bytes };
    }

    pub fn close(self: *Table) void {
        if (comptime builtin.os.tag == .windows) return;
        _ = std.c.close(self.fd);
        self.fd = -1;
    }

    pub const ReadError = error{ EmbedPreadShape, EmbedPreadTokenPastVocab, EmbedPreadRead, OutOfMemory };

    /// Rows `ids` into `out` (`ids.len` rows of `row_bytes`, in `ids`' order): each distinct id read once, the reads in
    /// parallel, then copied to its repeats.
    pub fn read(self: *const Table, gpa: std.mem.Allocator, ids: []const u32, out: []u8) ReadError!void {
        if (out.len != ids.len * self.row_bytes) return error.EmbedPreadShape;
        for (ids) |v| if (v >= self.rows) return error.EmbedPreadTokenPastVocab;
        if (ids.len == 0) return;
        const order = try gpa.alloc(u64, ids.len);
        defer gpa.free(order);
        for (order, ids, 0..) |*o, v, i| o.* = (@as(u64, v) << 32) | i;
        std.mem.sort(u64, order, {}, std.sort.asc(u64));
        // the first position of each distinct id
        const firsts = try gpa.alloc(u32, ids.len);
        defer gpa.free(firsts);
        var n_first: usize = 0;
        for (order, 0..) |o, k| {
            if (k > 0 and (order[k - 1] >> 32) == (o >> 32)) continue;
            firsts[n_first] = @truncate(o);
            n_first += 1;
        }
        var job: Job = .{ .t = self, .ids = ids, .pos = firsts[0..n_first], .out = out };
        job.runAll();
        if (job.bad.load(.acquire)) return error.EmbedPreadRead;
        // the repeats copy their id's first row
        var src: usize = 0;
        for (order, 0..) |o, k| {
            const p: usize = @as(u32, @truncate(o));
            if (k == 0 or (order[k - 1] >> 32) != (o >> 32)) {
                src = p;
                continue;
            }
            @memcpy(out[p * self.row_bytes ..][0..self.row_bytes], out[src * self.row_bytes ..][0..self.row_bytes]);
        }
    }

    const Job = struct {
        t: *const Table,
        ids: []const u32,
        pos: []const u32,
        out: []u8,
        bad: std.atomic.Value(bool) = .init(false),

        fn runAll(self: *Job) void {
            if (comptime builtin.os.tag.isDarwin()) {
                // GCD's pooled workers: a round's few rows and a prompt's thousands alike, no thread to spawn
                dispatch_apply_f(self.pos.len, dispatch_get_global_queue(qos_user_initiated, 0), self, applyOne);
            } else for (0..self.pos.len) |i| self.one(i);
        }

        fn applyOne(ctx: ?*anyopaque, i: usize) callconv(.c) void {
            const self: *Job = @ptrCast(@alignCast(ctx.?));
            self.one(i);
        }

        fn one(self: *Job, i: usize) void {
            const t = self.t;
            const p: usize = self.pos[i];
            const row: u64 = self.ids[p];
            if (!preadAll(t.fd, self.out[p * t.row_bytes ..][0..t.row_bytes], t.off + row * t.row_bytes)) self.bad.store(true, .release);
        }
    };
};

const qos_user_initiated: isize = 0x19;
extern "c" fn dispatch_get_global_queue(identifier: isize, flags: usize) ?*anyopaque;
extern "c" fn dispatch_apply_f(iterations: usize, queue: ?*anyopaque, context: ?*anyopaque, work: *const fn (?*anyopaque, usize) callconv(.c) void) void;

fn intOf(v: std.json.Value) ?u64 {
    return switch (v) {
        .integer => |x| if (x >= 0) @intCast(x) else null,
        else => null,
    };
}

/// `dst` filled from `off` (pread until done; false on an error or the end of the file).
fn preadAll(fd: c_int, dst: []u8, off: u64) bool {
    if (comptime builtin.os.tag == .windows) return false;
    var got: usize = 0;
    while (got < dst.len) {
        const r = std.c.pread(fd, dst[got..].ptr, dst.len - got, @intCast(off + got));
        if (r <= 0) return false;
        got += @intCast(r);
    }
    return true;
}

/// The shard the pack's index names for `tensor` (caller frees).
fn shardOf(gpa: std.mem.Allocator, dir: []const u8) error{ EmbedPreadIndex, OutOfMemory }![]u8 {
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/model.safetensors.index.json", .{dir}, 0);
    defer gpa.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.EmbedPreadIndex;
    defer _ = std.c.close(fd);
    const size = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (size <= 0 or size > 256 << 20) return error.EmbedPreadIndex;
    const raw = try gpa.alloc(u8, @intCast(size));
    defer gpa.free(raw);
    if (!preadAll(fd, raw, 0)) return error.EmbedPreadIndex;
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, raw, .{}) catch return error.EmbedPreadIndex;
    defer parsed.deinit();
    if (parsed.value != .object) return error.EmbedPreadIndex;
    const wm = parsed.value.object.get("weight_map") orelse return error.EmbedPreadIndex;
    if (wm != .object) return error.EmbedPreadIndex;
    const shard = wm.object.get(tensor) orelse return error.EmbedPreadIndex;
    if (shard != .string or shard.string.len == 0 or std.mem.indexOfScalar(u8, shard.string, '/') != null) return error.EmbedPreadIndex;
    return gpa.dupe(u8, shard.string);
}

/// A test pack in a fresh directory under /tmp: an index naming `shard` for `tensor`, and `shard` holding a header, a
/// leading tensor and `tensor` as BF16 [rows, hidden] whose row r, value j holds the bits `valueAt(r, j)`.
const TestPack = struct {
    dir: [:0]u8,
    shard_path: [:0]u8,
    index_path: [:0]u8,

    fn make(gpa: std.mem.Allocator, rows: u32, hidden: u32, dtype: []const u8) !TestPack {
        const dir = try std.fmt.allocPrintSentinel(gpa, "/tmp/embed-pread-test-{d}-{d}", .{ std.c.getpid(), rows * 7919 + hidden }, 0);
        errdefer gpa.free(dir);
        _ = std.c.mkdir(dir.ptr, @as(std.c.mode_t, 0o755));
        const shard_path = try std.fmt.allocPrintSentinel(gpa, "{s}/shard-1.safetensors", .{dir}, 0);
        errdefer gpa.free(shard_path);
        const index_path = try std.fmt.allocPrintSentinel(gpa, "{s}/model.safetensors.index.json", .{dir}, 0);
        errdefer gpa.free(index_path);
        const lead: usize = 24; // a tensor before the table, so its data offset is not 0
        const tbytes: usize = @as(usize, rows) * hidden * 2;
        const header = try std.fmt.allocPrint(gpa, "{{\"lead\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}},\"" ++ tensor ++ "\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}}}", .{ lead, lead, dtype, rows, hidden, lead, lead + tbytes });
        defer gpa.free(header);
        const file = try gpa.alloc(u8, 8 + header.len + lead + tbytes);
        defer gpa.free(file);
        std.mem.writeInt(u64, file[0..8], header.len, .little);
        @memcpy(file[8..][0..header.len], header);
        @memset(file[8 + header.len ..][0..lead], 0xEE);
        const body = file[8 + header.len + lead ..];
        for (0..rows) |r| for (0..hidden) |j| std.mem.writeInt(u16, body[(r * hidden + j) * 2 ..][0..2], valueAt(r, j), .little);
        try writeFile(shard_path, file);
        const index = "{\"metadata\":{},\"weight_map\":{\"lead\":\"shard-1.safetensors\",\"" ++ tensor ++ "\":\"shard-1.safetensors\"}}";
        try writeFile(index_path, index);
        return .{ .dir = dir, .shard_path = shard_path, .index_path = index_path };
    }

    fn deinit(self: *TestPack, gpa: std.mem.Allocator) void {
        _ = std.c.unlink(self.shard_path.ptr);
        _ = std.c.unlink(self.index_path.ptr);
        _ = std.c.rmdir(self.dir.ptr);
        gpa.free(self.shard_path);
        gpa.free(self.index_path);
        gpa.free(self.dir);
    }

    /// Row `r`, value `j`'s bits: distinct for every row below 8192 (1000 r + j modulo 2^16, j < 1000).
    fn valueAt(r: usize, j: usize) u16 {
        return @truncate(r * 1000 + j);
    }

    fn writeFile(path: [:0]const u8, bytes: []const u8) !void {
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return error.TestPackWrite;
        defer _ = std.c.close(fd);
        var done: usize = 0;
        while (done < bytes.len) {
            const w = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
            if (w <= 0) return error.TestPackWrite;
            done += @intCast(w);
        }
    }
};

test "embed_pread table: rows by id in the request's order, repeats copied, at a few rows and at a prompt's thousands" {
    const t = std.testing;
    const gpa = t.allocator;
    const rows: u32 = 600;
    const hidden: u32 = 48;
    var pack = try TestPack.make(gpa, rows, hidden, "BF16");
    defer pack.deinit(gpa);
    var tab = try Table.open(gpa, pack.dir, rows, hidden);
    defer tab.close();
    try t.expectEqual(@as(usize, hidden * 2), tab.row_bytes);
    var prng = std.Random.DefaultPrng.init(0xE3BED);
    const rnd = prng.random();
    for ([_]usize{ 1, 7, 3000 }) |n| {
        const ids = try gpa.alloc(u32, n);
        defer gpa.free(ids);
        for (ids, 0..) |*v, i| v.* = if (n > 1 and i % 5 == 4) ids[i - 1] else rnd.uintLessThan(u32, rows);
        if (n > 2) ids[n - 1] = rows - 1;
        const out = try gpa.alloc(u8, n * tab.row_bytes);
        defer gpa.free(out);
        @memset(out, 0);
        try tab.read(gpa, ids, out);
        for (ids, 0..) |v, i| for (0..hidden) |j| {
            try t.expectEqual(TestPack.valueAt(v, j), std.mem.readInt(u16, out[(i * hidden + j) * 2 ..][0..2], .little));
        };
    }
    // an id past the table, and a buffer of the wrong size, are refused
    var one: [96]u8 = undefined;
    try t.expectError(error.EmbedPreadTokenPastVocab, tab.read(gpa, &.{rows}, &one));
    try t.expectError(error.EmbedPreadShape, tab.read(gpa, &.{ 1, 2 }, &one));
}

test "embed_pread table: a pack whose table is not BF16 [rows, hidden] is refused" {
    const t = std.testing;
    const gpa = t.allocator;
    var f16_pack = try TestPack.make(gpa, 16, 8, "F16");
    defer f16_pack.deinit(gpa);
    try t.expectError(error.EmbedPreadTensor, Table.open(gpa, f16_pack.dir, 16, 8));
    var pack = try TestPack.make(gpa, 17, 8, "BF16");
    defer pack.deinit(gpa);
    try t.expectError(error.EmbedPreadTensor, Table.open(gpa, pack.dir, 18, 8));
    try t.expectError(error.EmbedPreadTensor, Table.open(gpa, pack.dir, 17, 16));
    try t.expectError(error.EmbedPreadIndex, Table.open(gpa, "/tmp/embed-pread-test-no-such-pack", 17, 8));
}
