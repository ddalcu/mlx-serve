//! The row tables DeepSeek-V4.1 reads past the page cache: the Engram banks' fixed-width records (`openRecords`, with
//! the byte-budgeted row cache and a poster thread) and the input embedding's raw BF16 rows inside a checkpoint shard
//! (`openTensor`, `gatherRaw`). Derived from upstream mlx-serve's n-gram table (`src/qwen4_exp.zig`); the Qwen PLE
//! paths (the mmapped quantized table, its prefetch knobs and warm thread, the n-gram hash) are not carried.

const std = @import("std");
const log = @import("sdk").log;
const io_util = @import("sdk").io_util;

pub const NgramTable = struct {
    /// The host's persistent pread workers (`io_util.PrefetchPool`), over this table's regions.
    pub const Pool = io_util.PrefetchPool(NgramTable);

    map: []align(std.heap.page_size_min) const u8,
    rows: u64,
    dim: u32,
    bits: u32,
    group_size: u32,
    w_off: usize,
    s_off: usize,
    b_off: usize,
    wcols: u32,
    scols: u32,
    /// Kept open for the pool's `pread` gather (page faults on one mapping
    /// serialize on the VM map lock; preads run in parallel).
    fd: std.c.fd_t = -1,
    pool: ?*Pool = null,
    /// Rows come through `pread` on `fd`, opened past the page cache
    /// (F_NOCACHE, read-ahead off), never through a mapping (`openTensor`).
    nocache: bool = false,
    /// A record table's resident rows (`attachCache`); without it every row is read.
    cache: ?*RowCache = null,
    /// The table's posted gathers (`enablePosting`): one thread runs them in post order.
    poster: ?*Poster = null,
    /// A raw BF16 table's aligned parallel row reads (`openTensor`; `gatherRaw`): its stages and readers, built once.
    row_gather: ?*io_util.RowGather = null,

    /// `gatherRaw`'s readers: 15 helpers and the caller; pieces of 1,024 ids; the caller alone below 32 ids (a decode
    /// or verify block). Built once per table.
    pub const raw_gather_helpers: usize = 15;
    pub const raw_gather_max_ids: usize = 1024;
    pub const raw_gather_parallel_min: usize = 32;

    /// A gather handed to the table's poster thread: exactly `gatherRecords(rows, out)`, run in post
    /// order (the cache included), ready once `wait` returns. The caller keeps `rows` and `out` alive and
    /// makes no other gather on the table until every posted one is waited.
    pub const Posted = struct {
        rows: []const i64,
        out: []u8,
        state: enum { queued, done, failed } = .queued,
        err: anyerror = error.RecordRead,
        next: ?*Posted = null,
    };

    /// `bits` of a record table: raw fixed-width records, never dequantized here.
    pub const records_bits: u32 = 0;

    /// `rows` records of `record_bytes` at `data_offset` of `path` (an Engram bank, an
    /// embedding tensor inside a safetensors shard), read past the page cache, not mapped.
    pub fn openRecords(path: [:0]const u8, record_bytes: u32, rows: u64, data_offset: u64) !NgramTable {
        if (record_bytes == 0) return error.NgramTableRegion;
        const fd = try io_util.openNoCache(path.ptr, .{});
        errdefer _ = std.c.close(fd);
        const size = std.c.lseek(fd, 0, std.c.SEEK.END);
        const need = std.math.add(u64, data_offset, std.math.mul(u64, rows, record_bytes) catch return error.NgramTableRegion) catch return error.NgramTableRegion;
        if (size < 0 or @as(u64, @intCast(size)) < need) return error.NgramTableTruncated;
        var t: NgramTable = .{ .map = &empty_map, .rows = rows, .dim = record_bytes, .bits = records_bits, .group_size = 0, .w_off = @intCast(data_offset), .s_off = 0, .b_off = 0, .wcols = 0, .scols = 0, .fd = fd, .nocache = true };
        if (record_bytes <= Pool.ROW_BUF) t.pool = try Pool.create();
        return t;
    }

    /// The records of `row_ids` read past the cache (a check's independent read).
    pub fn readUncached(self: *const NgramTable, row_ids: []const i64, out: []u8) !void {
        std.debug.assert(self.bits == records_bits and out.len == row_ids.len * self.dim);
        for (row_ids) |r| if (r < 0 or @as(u64, @intCast(r)) >= self.rows) return error.RowOutOfRange;
        return self.readRecords(row_ids, out);
    }

    /// Start the table's poster thread (`post` / `wait`). Call once the table sits at its final address.
    pub fn enablePosting(self: *NgramTable) !void {
        std.debug.assert(self.poster == null);
        self.poster = try Poster.create(self);
    }

    /// Queue `job` on the poster thread; the gather starts once the jobs before it finish.
    pub fn post(self: *NgramTable, job: *Posted) void {
        self.poster.?.push(job);
    }

    /// Block until `job` ran; its gather's error, if any.
    pub fn wait(self: *NgramTable, job: *Posted) !void {
        return self.poster.?.await(job);
    }

    /// Keep up to `budget_bytes` of this record table's rows resident (`RowCache`).
    pub fn attachCache(self: *NgramTable, gpa: std.mem.Allocator, budget_bytes: u64) !void {
        std.debug.assert(self.bits == records_bits and self.cache == null);
        const c = try gpa.create(RowCache);
        errdefer gpa.destroy(c);
        c.* = try RowCache.init(gpa, self.dim, budget_bytes);
        self.cache = c;
    }

    /// Row reads per row: the three quantized regions, or the one record.
    pub fn regions(self: *const NgramTable) usize {
        return if (self.bits == records_bits) 1 else 3;
    }

    /// The records of `row_ids`, in order, into `out` (`row_ids.len * record_bytes`),
    /// through the cache when one is attached; reads ride the pool.
    pub fn gatherRecords(self: *const NgramTable, row_ids: []const i64, out: []u8) !void {
        std.debug.assert(self.bits == records_bits and out.len == row_ids.len * self.dim);
        for (row_ids) |r| if (r < 0 or @as(u64, @intCast(r)) >= self.rows) return error.RowOutOfRange;
        if (self.cache) |c| return c.gather(self, row_ids, out);
        return self.readRecords(row_ids, out);
    }

    /// Read `rows`' records into `out`, the pool's rounds when it has one.
    fn readRecords(self: *const NgramTable, rows: []const i64, out: []u8) !void {
        const rb: usize = self.dim;
        if (self.pool) |p| {
            var start: usize = 0;
            while (start < rows.len) : (start += Pool.MAX_ROWS) {
                const end = @min(start + Pool.MAX_ROWS, rows.len);
                if (!p.run(self, rows[start..end])) return error.RecordRead;
                for (start..end) |i| @memcpy(out[i * rb ..][0..rb], p.bufs[i - start][0..rb]);
            }
            return;
        }
        for (rows, 0..) |r, i| if (!self.preadSite(@intCast(r), 0, out[i * rb ..][0..rb])) return error.RecordRead;
    }

    /// A raw BF16 `[rows, dim]` tensor named `name` inside any safetensors file
    /// (a checkpoint shard, no `mlx-serve-ngram` metadata), read past the page
    /// cache: the descriptor is F_NOCACHE with read-ahead off, rows come
    /// through `pread` (`gatherRaw`), nothing is mapped and no warm thread runs.
    pub fn openTensor(path: [:0]const u8, name: []const u8) !NgramTable {
        const fd = io_util.openNoCache(path.ptr, .{}) catch |e| return switch (e) {
            error.NoCacheFcntl, error.NoCacheUnsupported => error.NgramTableNoCache,
            else => error.FileNotFound,
        };
        errdefer _ = std.c.close(fd);
        const size: usize = @intCast(@max(std.c.lseek(fd, 0, std.c.SEEK.END), 0));
        if (size < 8) return error.NgramTableTruncated;
        var len_bytes: [8]u8 = undefined;
        try io_util.readAligned(fd, &len_bytes, 0);
        const hlen: usize = @intCast(std.mem.readInt(u64, &len_bytes, .little));
        if (hlen > size - 8) return error.NgramTableTruncated;
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const header = try a.alloc(u8, hlen);
        try io_util.readAligned(fd, header, 8);
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, header, .{}) catch return error.NgramTableHeader;
        if (parsed != .object) return error.NgramTableHeader;
        const w = try io_util.headerRegion(parsed.object, name, "BF16", 2, size, 8 + hlen);
        var t: NgramTable = .{
            .map = &empty_map,
            .rows = w.rows,
            .dim = @intCast(w.cols),
            .bits = 16,
            .group_size = 0,
            .w_off = 8 + hlen + @as(usize, @intCast(w.start)),
            .s_off = 0,
            .b_off = 0,
            .wcols = 0,
            .scols = 0,
            .fd = fd,
            .nocache = true,
        };
        t.row_gather = try io_util.RowGather.init(fd, t.w_off, @as(usize, t.dim) * 2, t.rows, raw_gather_helpers, raw_gather_max_ids, raw_gather_parallel_min);
        return t;
    }

    const empty_map: [0]u8 align(std.heap.page_size_min) = .{};

    /// Raw BF16 rows `row_ids`, in order, into `out` (`row_ids.len * dim * 2`
    /// bytes): the table's bytes, no conversion (a bits-16 table). A no-cache table reads through its row gather:
    /// whole aligned pages only (an unaligned read would leave its pages in the page cache), each distinct row once,
    /// in parallel at prompt widths.
    pub fn gatherRaw(self: *const NgramTable, row_ids: []const u32, out: []u8) !void {
        std.debug.assert(self.bits == 16);
        const rb: usize = @as(usize, self.dim) * 2;
        if (out.len != row_ids.len * rb) return error.NgramTableRegion;
        for (row_ids) |r| if (r >= self.rows) return error.NgramTableRegion;
        if (self.nocache) return self.row_gather.?.gather(row_ids, out);
        for (row_ids, 0..) |r, i| {
            const off = self.w_off + @as(usize, r) * rb;
            @memcpy(out[i * rb ..][0..rb], self.map[off..][0..rb]);
        }
    }

    pub fn close(self: *NgramTable) void {
        if (self.poster) |p| p.destroy();
        self.poster = null;
        if (self.pool) |p| p.destroy();
        self.pool = null;
        if (self.row_gather) |g| g.deinit();
        self.row_gather = null;
        if (self.fd >= 0) _ = std.c.close(self.fd);
        self.fd = -1;
        if (self.cache) |c| {
            const gpa = c.gpa;
            c.deinit();
            gpa.destroy(c);
            self.cache = null;
        }
        if (!self.nocache) std.posix.munmap(self.map);
    }

    /// One (row, region) pread into the pool's row buffer. False on a short read.
    pub fn preadSite(self: *const NgramTable, r: u64, region: usize, buf: []u8) bool {
        const wl: usize = if (self.bits == records_bits) self.dim else self.wcols * 4;
        const sl: usize = self.scols * 2;
        const off: usize, const dst: []u8 = switch (region) {
            0 => .{ self.w_off + r * wl, buf[0..wl] },
            1 => .{ self.s_off + r * sl, buf[wl .. wl + sl] },
            else => .{ self.b_off + r * sl, buf[wl + sl .. wl + 2 * sl] },
        };
        return std.c.pread(self.fd, dst.ptr, dst.len, @intCast(off)) == @as(isize, @intCast(dst.len));
    }

};

/// A byte-budgeted LRU of a record table's rows, Python's `NGramRowCache` rule for rule: a
/// gather's distinct rows in first-appearance order, a hit made newest, the misses read and
/// entered sorted (sub-runs capped at the slot count), each taking the highest free slot or the oldest row's.
pub const RowCache = struct {
    pub const Stats = struct { hits: u64 = 0, misses: u64 = 0, evictions: u64 = 0, reads: u64 = 0, rows_read: u64 = 0, gathers: u64 = 0 };
    pub const nil: u32 = std.math.maxInt(u32);
    const Miss = struct { row: u64, d: u32 };

    gpa: std.mem.Allocator,
    record_bytes: u32,
    slot_count: u32,
    arena: []u8,
    slot_row: []u64,
    prev: []u32,
    next: []u32,
    free: []u32,
    n_free: u32,
    oldest: u32 = nil,
    newest: u32 = nil,
    index: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    /// Removals since the index was last rehashed (its tombstones).
    removed: u32 = 0,
    stats: Stats = .{},
    // One gather's scratch, kept across gathers.
    first: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    distinct: std.ArrayList(u64) = .empty,
    chain_head: std.ArrayList(u32) = .empty,
    chain_tail: std.ArrayList(u32) = .empty,
    chain_next: std.ArrayList(u32) = .empty,
    misses: std.ArrayList(Miss) = .empty,
    miss_rows: std.ArrayList(i64) = .empty,
    miss_recs: std.ArrayList(u8) = .empty,

    /// Python's slot rule: `max(1, max(record_bytes, budget) // record_bytes)`.
    pub fn slotCount(record_bytes: u32, budget_bytes: u64) u32 {
        return @intCast(@max(1, @max(record_bytes, budget_bytes) / record_bytes));
    }

    /// The row index's capacity for `slots` rows (the std map's 80 % load rule, minimum 8).
    pub fn indexCapacity(slots: u32) u32 {
        return @max(8, std.math.ceilPowerOfTwo(u32, @intCast(@as(u64, slots) * 100 / 80 + 1)) catch unreachable);
    }

    /// Host bytes the cache holds for its whole life (a memory bill's term): the slot
    /// arena, the per-slot links and the row index.
    pub fn hostBytes(record_bytes: u32, budget_bytes: u64) u64 {
        const slots = slotCount(record_bytes, budget_bytes);
        return @as(u64, slots) * (record_bytes + @sizeOf(u64) + 3 * @sizeOf(u32)) + @as(u64, indexCapacity(slots)) * (@sizeOf(u64) + @sizeOf(u32) + 1);
    }

    pub fn init(gpa: std.mem.Allocator, record_bytes: u32, budget_bytes: u64) !RowCache {
        const slots = slotCount(record_bytes, budget_bytes);
        const arena = try gpa.alloc(u8, @as(usize, slots) * record_bytes);
        errdefer gpa.free(arena);
        const slot_row = try gpa.alloc(u64, slots);
        errdefer gpa.free(slot_row);
        const prev = try gpa.alloc(u32, slots);
        errdefer gpa.free(prev);
        const next = try gpa.alloc(u32, slots);
        errdefer gpa.free(next);
        const free = try gpa.alloc(u32, slots);
        errdefer gpa.free(free);
        for (free, 0..) |*f, i| f.* = @intCast(i);
        var self: RowCache = .{ .gpa = gpa, .record_bytes = record_bytes, .slot_count = slots, .arena = arena, .slot_row = slot_row, .prev = prev, .next = next, .free = free, .n_free = slots };
        try self.index.ensureTotalCapacity(gpa, slots);
        return self;
    }

    pub fn deinit(self: *RowCache) void {
        const a = self.gpa;
        a.free(self.arena);
        a.free(self.slot_row);
        a.free(self.prev);
        a.free(self.next);
        a.free(self.free);
        self.index.deinit(a);
        self.first.deinit(a);
        self.distinct.deinit(a);
        self.chain_head.deinit(a);
        self.chain_tail.deinit(a);
        self.chain_next.deinit(a);
        self.misses.deinit(a);
        self.miss_rows.deinit(a);
        self.miss_recs.deinit(a);
    }

    fn unlink(self: *RowCache, s: u32) void {
        const p = self.prev[s];
        const n = self.next[s];
        if (p != nil) self.next[p] = n else self.oldest = n;
        if (n != nil) self.prev[n] = p else self.newest = p;
    }

    fn linkNewest(self: *RowCache, s: u32) void {
        self.prev[s] = self.newest;
        self.next[s] = nil;
        if (self.newest != nil) self.next[self.newest] = s else self.oldest = s;
        self.newest = s;
    }

    /// Python `_alloc_slot`: `list.pop()` of the free list, else the oldest row is evicted.
    fn allocSlot(self: *RowCache) u32 {
        if (self.n_free > 0) {
            self.n_free -= 1;
            return self.free[self.n_free];
        }
        const s = self.oldest;
        _ = self.index.remove(self.slot_row[s]);
        self.removed += 1;
        self.unlink(s);
        self.stats.evictions += 1;
        return s;
    }

    fn lessRow(_: void, x: Miss, y: Miss) bool {
        return x.row < y.row;
    }

    /// `rec` into every position of distinct row `d` in `out`.
    fn put(self: *const RowCache, out: []u8, d: u32, rec: []const u8) void {
        const rb: usize = self.record_bytes;
        var i = self.chain_head.items[d];
        while (i != nil) : (i = self.chain_next.items[i]) @memcpy(out[@as(usize, i) * rb ..][0..rb], rec);
    }

    /// Python `gather_bytes` over `table` (rows already range-checked).
    fn gather(self: *RowCache, table: *const NgramTable, rows: []const i64, out: []u8) !void {
        const a = self.gpa;
        const rb: usize = self.record_bytes;
        self.first.clearRetainingCapacity();
        self.distinct.clearRetainingCapacity();
        self.chain_head.clearRetainingCapacity();
        self.chain_tail.clearRetainingCapacity();
        try self.chain_next.resize(a, rows.len);
        for (rows, 0..) |r, i| {
            const gop = try self.first.getOrPut(a, @intCast(r));
            self.chain_next.items[i] = nil;
            if (gop.found_existing) {
                const d = gop.value_ptr.*;
                self.chain_next.items[self.chain_tail.items[d]] = @intCast(i);
                self.chain_tail.items[d] = @intCast(i);
            } else {
                gop.value_ptr.* = @intCast(self.distinct.items.len);
                try self.distinct.append(a, @intCast(r));
                try self.chain_head.append(a, @intCast(i));
                try self.chain_tail.append(a, @intCast(i));
            }
        }
        self.misses.clearRetainingCapacity();
        for (self.distinct.items, 0..) |row, d| {
            if (self.index.get(row)) |slot| {
                self.unlink(slot);
                self.linkNewest(slot);
                self.stats.hits += 1;
                self.put(out, @intCast(d), self.arena[@as(usize, slot) * rb ..][0..rb]);
            } else try self.misses.append(a, .{ .row = row, .d = @intCast(d) });
        }
        const ms = self.misses.items;
        if (ms.len > 0) {
            std.mem.sort(Miss, ms, {}, lessRow);
            try self.miss_rows.resize(a, ms.len);
            for (ms, self.miss_rows.items) |m, *r| r.* = @intCast(m.row);
            try self.miss_recs.resize(a, ms.len * rb);
            try table.readRecords(self.miss_rows.items, self.miss_recs.items);
        }
        var i: usize = 0;
        while (i < ms.len) {
            var j = i + 1;
            while (j < ms.len and ms[j].row == ms[j - 1].row + 1) j += 1;
            var off = i;
            while (off < j) {
                const n: usize = @min(j - off, self.slot_count);
                self.stats.reads += 1;
                self.stats.rows_read += n;
                self.stats.misses += n;
                for (ms[off..][0..n], off..) |m, k| {
                    const slot = self.allocSlot();
                    const rec = self.miss_recs.items[k * rb ..][0..rb];
                    @memcpy(self.arena[@as(usize, slot) * rb ..][0..rb], rec);
                    self.slot_row[slot] = m.row;
                    self.index.putAssumeCapacity(m.row, slot);
                    self.linkNewest(slot);
                    self.put(out, m.d, rec);
                }
                off += n;
            }
            i = j;
        }
        self.stats.gathers += 1;
        // Evictions leave tombstones in the index: clear them once they reach half the slots.
        if (self.removed >= self.slot_count / 2 + 1) {
            self.index.rehash(std.hash_map.AutoContext(u64){});
            self.removed = 0;
        }
    }
};

/// One table's posted gathers: a thread that runs them in post order (each is `gatherRecords`, so the
/// pool's rounds and the cache's order are the blocking path's). Jobs are drained before it stops.
const Poster = struct {
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    head: ?*NgramTable.Posted = null,
    tail: ?*NgramTable.Posted = null,
    quit: bool = false,
    table: *NgramTable,
    thread: std.Thread = undefined,

    fn create(table: *NgramTable) !*Poster {
        const a = std.heap.page_allocator;
        const p = try a.create(Poster);
        errdefer a.destroy(p);
        p.* = .{ .table = table };
        p.thread = try std.Thread.spawn(.{ .stack_size = 256 * 1024 }, run, .{p});
        return p;
    }

    fn destroy(self: *Poster) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.quit = true;
        self.cv.broadcast(io);
        self.mu.unlock(io);
        self.thread.join();
        std.heap.page_allocator.destroy(self);
    }

    fn push(self: *Poster, job: *NgramTable.Posted) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        job.state = .queued;
        job.next = null;
        self.mu.lockUncancelable(io);
        if (self.tail) |t| t.next = job else self.head = job;
        self.tail = job;
        self.cv.broadcast(io);
        self.mu.unlock(io);
    }

    fn await(self: *Poster, job: *NgramTable.Posted) !void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        while (job.state == .queued) self.cv.wait(io, &self.mu) catch {};
        const failed = job.state == .failed;
        self.mu.unlock(io);
        if (failed) return job.err;
    }

    fn run(self: *Poster) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        while (true) {
            self.mu.lockUncancelable(io);
            while (self.head == null and !self.quit) self.cv.wait(io, &self.mu) catch {};
            const job = self.head orelse {
                self.mu.unlock(io);
                return;
            };
            self.head = job.next;
            if (self.head == null) self.tail = null;
            self.mu.unlock(io);
            const r = self.table.gatherRecords(job.rows, job.out);
            self.mu.lockUncancelable(io);
            if (r) |_| job.state = .done else |e| {
                job.err = e;
                job.state = .failed;
            }
            self.cv.broadcast(io);
            self.mu.unlock(io);
        }
    }
};

const testing = std.testing;

test "dsv41 ngram table: a BF16 tensor inside a checkpoint shard gathers its raw rows past the page cache" {
    // Two tensors, the second a BF16 [5, 3] table after a U8 one; no mlx-serve-ngram metadata.
    const header = "{\"other\":{\"dtype\":\"U8\",\"shape\":[1,7],\"data_offsets\":[0,7]},\"embed.weight\":{\"dtype\":\"BF16\",\"shape\":[5,3],\"data_offsets\":[7,37]}}";
    var image: [8 + header.len + 37 + 3]u8 = undefined;
    std.mem.writeInt(u64, image[0..8], header.len, .little);
    @memcpy(image[8..][0..header.len], header);
    const data = image[8 + header.len ..];
    for (data, 0..) |*b, i| b.* = @truncate(i *% 37 +% 11);
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try td.dir.writeFile(testing.io, .{ .sub_path = "shard.safetensors", .data = &image });
    var root: [512]u8 = undefined;
    var pbuf: [700]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/shard.safetensors", .{root[0..try td.dir.realPath(testing.io, &root)]}, 0);
    var t = try NgramTable.openTensor(path, "embed.weight");
    defer t.close();
    try testing.expect(t.nocache and t.map.len == 0);
    try testing.expectEqual(@as(u64, 5), t.rows);
    try testing.expectEqual(@as(u32, 3), t.dim);
    var out: [4 * 6]u8 = undefined;
    const ids = [_]u32{ 4, 0, 4, 2 };
    try t.gatherRaw(&ids, &out);
    for (ids, 0..) |r, i| try testing.expectEqualSlices(u8, data[7 + r * 6 ..][0..6], out[i * 6 ..][0..6]);
    try testing.expectError(error.NgramTableRegion, t.gatherRaw(&.{5}, out[0..6]));
    try testing.expectError(error.NgramTableRegion, t.gatherRaw(&.{0}, out[0..5]));
    // By name only: another dtype or a missing name is refused.
    try testing.expectError(error.NgramTableHeader, NgramTable.openTensor(path, "other"));
    try testing.expectError(error.NgramTableHeader, NgramTable.openTensor(path, "missing"));
    try testing.expectError(error.FileNotFound, NgramTable.openTensor("/nonexistent/shard.safetensors", "embed.weight"));
}
