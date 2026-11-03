//! Reads past the page cache (`openNoCache`, `noCache`, `readAligned`), the on-disk row gather and its
//! prefetch pool, file residency, and bf16 rounding: the plugin's own I/O helpers (from #749's host io_util).

const std = @import("std");
const builtin = @import("builtin");

pub const NoCacheOptions = struct {
    /// Keep the kernel's read-ahead (a file streamed front to back).
    read_ahead: bool = false,
    /// Open a symlink's target; false refuses a symlink (ELOOP).
    follow_symlinks: bool = true,
};

/// Reads of `fd` bypass the page cache (F_NOCACHE); read-ahead off unless asked.
/// Darwin only: elsewhere it refuses, so no caller silently reads through the cache.
pub fn noCache(fd: std.c.fd_t, opts: NoCacheOptions) error{ NoCacheFcntl, NoCacheUnsupported }!void {
    if (comptime !builtin.os.tag.isDarwin()) return error.NoCacheUnsupported;
    if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0) return error.NoCacheFcntl;
    if (!opts.read_ahead and std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0)) != 0) return error.NoCacheFcntl;
}

/// `path` read-only and close-on-exec, then `noCache`. `error.OpenFailed` leaves
/// errno as `open` set it.
pub fn openNoCache(path: [*:0]const u8, opts: NoCacheOptions) error{ FileNotFound, OpenFailed, NoCacheFcntl, NoCacheUnsupported }!std.c.fd_t {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .NOFOLLOW = !opts.follow_symlinks, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return if (std.c._errno().* == @intFromEnum(std.posix.E.NOENT)) error.FileNotFound else error.OpenFailed;
    errdefer _ = std.c.close(fd);
    try noCache(fd, opts);
    return fd;
}

/// A whole file read past the page cache (`openNoCache`), at most `limit` bytes; caller frees.
pub fn readAllNoCache(a: std.mem.Allocator, path: []const u8, limit: usize) ![]u8 {
    const z = try std.fmt.allocPrintSentinel(a, "{s}", .{path}, 0);
    defer a.free(z);
    const fd = try openNoCache(z.ptr, .{});
    defer _ = std.c.close(fd);
    const size = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (size < 0) return error.OpenFailed;
    if (@as(u64, @intCast(size)) > limit) return error.FileTooBig;
    const buf = try a.alloc(u8, @intCast(size));
    errdefer a.free(buf);
    var done: usize = 0;
    while (done < buf.len) {
        const n = std.c.pread(fd, buf[done..].ptr, buf.len - done, @intCast(done));
        if (n <= 0) return error.ReadFailed;
        done += @intCast(n);
    }
    return buf;
}

// ── Reads past the page cache: aligned reads, a table's row gather, residency ──

/// readAligned's staging buffer: page-aligned (the page allocator), a multiple of every page size.
const stage_bytes: usize = 8 << 20;

/// `buf.len` bytes at `off` of an F_NOCACHE descriptor, through whole aligned pages into a page-aligned stage (the
/// construction's header reads of a table read past the page cache, `NgramTable.openTensor`); errors by name.
pub fn readAligned(fd: std.c.fd_t, buf: []u8, off: u64) !void {
    if (buf.len == 0) return;
    const page: u64 = std.heap.pageSize();
    const span = std.mem.alignForward(u64, off % page + buf.len, page);
    const stage = try std.heap.page_allocator.alloc(u8, @intCast(@min(span, stage_bytes)));
    defer std.heap.page_allocator.free(stage);
    var pos = off;
    const end = off + buf.len;
    var out: usize = 0;
    while (pos < end) {
        const a0 = pos - pos % page;
        const want_end = @min(end, a0 + stage.len);
        const need = want_end - a0;
        const len = std.mem.alignForward(u64, need, page);
        var got: u64 = 0;
        while (got < need) {
            if (got % page != 0) return error.ReadShort;
            const n = std.c.pread(fd, stage[@intCast(got)..].ptr, @intCast(len - got), @intCast(a0 + got));
            if (n < 0) {
                if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
                return error.ReadFailed;
            }
            if (n == 0) return error.ReadShort;
            got += @intCast(n);
        }
        const s0: usize = @intCast(pos - a0);
        const take: usize = @intCast(need - (pos - a0));
        @memcpy(buf[out..][0..take], stage[s0..][0..take]);
        out += take;
        pos = want_end;
    }
}

// ── Row gather: a table's rows past the page cache ──

/// A table's rows past the page cache (the input embedding's host rows): `row_bytes` at `base + id * row_bytes` of an
/// F_NOCACHE descriptor, gathered into the caller's id order. Every read is page-aligned in offset, length and
/// destination, and checked, since macOS keeps only aligned reads out of the page cache (nocache_reader's `Desc.readAt`). Each distinct
/// row is read once; rows whose aligned pages touch are read as one run; `helpers` threads and the caller take the runs
/// in parallel. `init` allocates everything once (the page-aligned stages, the sort and run scratch, the threads): a
/// gather allocates nothing. One gather at a time (the model's thread); the descriptor stays its owner's.
pub const RowGather = struct {
    /// One reader's stage: the widest run it reads (whole pages).
    pub const stage_len: usize = 128 << 10;
    pub const Item = struct { id: u32, at: u32 };
    /// `len` aligned bytes at `off` (aligned) hold `items[first..end]`'s rows; the last of them ends `need` bytes in.
    pub const Run = struct { off: u64, len: u64, need: u64, first: u32, end: u32 };

    fd: std.c.fd_t,
    base: u64,
    row_bytes: usize,
    rows: u64,
    page: usize,
    /// Ids per piece (a longer call runs in pieces of this many).
    max_ids: usize,
    /// Below this many ids the caller reads alone (no helper woken).
    parallel_min: usize,
    /// `helpers + 1` stages; `[0]` is the caller's.
    stages: [][]u8,
    items: []Item,
    runs: []Run,
    threads: []std.Thread,
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    gen: u64 = 0,
    quit: bool = false,
    /// The piece in flight (set before `gen` moves, read after it).
    out: []u8 = &.{},
    n_runs: usize = 0,
    next: std.atomic.Value(usize) = .init(0),
    pending: std.atomic.Value(u32) = .init(0),
    failed: std.atomic.Value(u32) = .init(0),
    unaligned: std.atomic.Value(u32) = .init(0),

    pub fn init(fd: std.c.fd_t, base: u64, row_bytes: usize, rows: u64, helpers: usize, max_ids: usize, parallel_min: usize) !*RowGather {
        const page = std.heap.pageSize();
        if (row_bytes == 0 or max_ids == 0 or row_bytes + 2 * page > stage_len or stage_len % page != 0) return error.GatherGeometry;
        const a = std.heap.c_allocator;
        const self = try a.create(RowGather);
        errdefer a.destroy(self);
        self.* = .{ .fd = fd, .base = base, .row_bytes = row_bytes, .rows = rows, .page = page, .max_ids = max_ids, .parallel_min = parallel_min, .stages = &.{}, .items = &.{}, .runs = &.{}, .threads = &.{} };
        self.stages = try a.alloc([]u8, helpers + 1);
        errdefer a.free(self.stages);
        var made: usize = 0;
        errdefer for (self.stages[0..made]) |st| std.heap.page_allocator.free(st);
        for (self.stages) |*st| {
            st.* = try std.heap.page_allocator.alloc(u8, stage_len);
            made += 1;
            if (@intFromPtr(st.*.ptr) % page != 0) return error.GatherGeometry;
        }
        self.items = try a.alloc(Item, max_ids);
        errdefer a.free(self.items);
        self.runs = try a.alloc(Run, max_ids);
        errdefer a.free(self.runs);
        self.threads = try a.alloc(std.Thread, helpers);
        errdefer a.free(self.threads);
        var started: usize = 0;
        errdefer self.stop(started);
        for (self.threads, 0..) |*t, i| {
            t.* = try std.Thread.spawn(.{ .stack_size = 64 * 1024 }, helper, .{ self, i });
            started += 1;
        }
        return self;
    }

    pub fn deinit(self: *RowGather) void {
        self.stop(self.threads.len);
        const a = std.heap.c_allocator;
        for (self.stages) |st| std.heap.page_allocator.free(st);
        a.free(self.stages);
        a.free(self.items);
        a.free(self.runs);
        a.free(self.threads);
        a.destroy(self);
    }

    fn stop(self: *RowGather, started: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.quit = true;
        self.cv.broadcast(io);
        self.mu.unlock(io);
        for (self.threads[0..started]) |t| t.join();
    }

    /// The rows of `ids` (any order, repeats allowed) into `out` (`ids.len * row_bytes`), in `ids` order.
    pub fn gather(self: *RowGather, ids: []const u32, out: []u8) !void {
        if (out.len != ids.len * self.row_bytes) return error.GatherShape;
        for (ids) |id| if (id >= self.rows) return error.RowOutOfRange;
        var start: usize = 0;
        while (start < ids.len) : (start += self.max_ids) {
            const end = @min(start + self.max_ids, ids.len);
            try self.piece(ids[start..end], out[start * self.row_bytes .. end * self.row_bytes]);
        }
    }

    fn lessId(_: void, x: Item, y: Item) bool {
        return x.id < y.id;
    }

    fn alignUp(self: *const RowGather, x: u64) u64 {
        return std.mem.alignForward(u64, x, self.page);
    }

    /// The sorted items' runs: each distinct row once, rows whose aligned pages touch in one run up to a stage.
    fn plan(self: *RowGather, n: usize) usize {
        const p: u64 = self.page;
        const rb: u64 = self.row_bytes;
        const items = self.items[0..n];
        var nr: usize = 0;
        var i: usize = 0;
        while (i < n) {
            const o0 = self.base + @as(u64, items[i].id) * rb;
            const start = o0 - o0 % p;
            var end_row = o0 + rb;
            var j = i + 1;
            while (j < n) : (j += 1) {
                if (items[j].id == items[j - 1].id) continue;
                const o = self.base + @as(u64, items[j].id) * rb;
                if (o - o % p > self.alignUp(end_row)) break;
                if (self.alignUp(o + rb) - start > stage_len) break;
                end_row = o + rb;
            }
            self.runs[nr] = .{ .off = start, .len = self.alignUp(end_row) - start, .need = end_row - start, .first = @intCast(i), .end = @intCast(j) };
            nr += 1;
            i = j;
        }
        return nr;
    }

    fn piece(self: *RowGather, ids: []const u32, out: []u8) !void {
        for (self.items[0..ids.len], ids, 0..) |*it, id, i| it.* = .{ .id = id, .at = @intCast(i) };
        std.mem.sort(Item, self.items[0..ids.len], {}, lessId);
        self.n_runs = self.plan(ids.len);
        self.out = out;
        self.next.store(0, .release);
        self.failed.store(0, .release);
        self.unaligned.store(0, .release);
        if (self.threads.len == 0 or ids.len < self.parallel_min or self.n_runs == 1) {
            self.work(self.stages[0]);
        } else {
            const io = std.Io.Threaded.global_single_threaded.io();
            self.mu.lockUncancelable(io);
            self.pending.store(@intCast(self.threads.len), .release);
            self.gen += 1;
            self.cv.broadcast(io);
            self.mu.unlock(io);
            self.work(self.stages[0]);
            while (self.pending.load(.acquire) != 0) std.atomic.spinLoopHint();
        }
        if (self.unaligned.load(.acquire) != 0) return error.GatherUnaligned;
        if (self.failed.load(.acquire) != 0) return error.GatherRead;
    }

    fn helper(self: *RowGather, idx: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        var seen: u64 = 0;
        while (true) {
            self.mu.lockUncancelable(io);
            while (self.gen == seen and !self.quit) self.cv.wait(io, &self.mu) catch {};
            if (self.quit) {
                self.mu.unlock(io);
                return;
            }
            seen = self.gen;
            self.mu.unlock(io);
            self.work(self.stages[idx + 1]);
            _ = self.pending.fetchSub(1, .acq_rel);
        }
    }

    /// Runs taken in turn: each read into `stage`, its rows copied to their places in the piece's `out`.
    fn work(self: *RowGather, stage: []u8) void {
        while (true) {
            const r = self.next.fetchAdd(1, .acq_rel);
            if (r >= self.n_runs) return;
            const run = self.runs[r];
            if (!self.aligned(run, stage)) {
                _ = self.unaligned.fetchAdd(1, .acq_rel);
                continue;
            }
            if (!self.readRun(run, stage)) {
                _ = self.failed.fetchAdd(1, .acq_rel);
                continue;
            }
            for (self.items[run.first..run.end]) |it| {
                const off: usize = @intCast(self.base + @as(u64, it.id) * self.row_bytes - run.off);
                @memcpy(self.out[@as(usize, it.at) * self.row_bytes ..][0..self.row_bytes], stage[off..][0..self.row_bytes]);
            }
        }
    }

    /// A run's read is page-aligned in offset, length and destination, and fits its stage.
    pub fn aligned(self: *const RowGather, run: Run, stage: []const u8) bool {
        return run.off % self.page == 0 and run.len % self.page == 0 and @intFromPtr(stage.ptr) % self.page == 0 and run.len <= stage.len and run.need <= run.len;
    }

    /// `run.len` bytes at `run.off` into `stage`, until its rows are in (the file may end inside the last page). Each
    /// pread stays aligned: a short read that is not whole pages is the file's end, never continued unaligned.
    fn readRun(self: *const RowGather, run: Run, stage: []u8) bool {
        var got: u64 = 0;
        while (got < run.need) {
            if (got % self.page != 0) return false;
            const k = std.c.pread(self.fd, stage[@intCast(got)..].ptr, @intCast(run.len - got), @intCast(run.off + got));
            if (k < 0) {
                if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
                return false;
            }
            if (k == 0) return false;
            got += @intCast(k);
        }
        return true;
    }
};

// ── Residency probes (proof tests) ──

/// Bytes of `path` resident in the page cache (mincore over a read-only map:
/// the map faults nothing in).
pub fn residentBytes(path: [:0]const u8) !u64 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
    const size: usize = @intCast(st.size);
    if (size == 0) return 0;
    const map = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
    defer std.posix.munmap(map);
    const page = std.heap.pageSize();
    const n_pages = (size + page - 1) / page;
    const vec = try std.heap.c_allocator.alloc(u8, n_pages);
    defer std.heap.c_allocator.free(vec);
    if (std.c.mincore(@ptrCast(map.ptr), size, vec.ptr) != 0) return error.MincoreFailed;
    var resident: u64 = 0;
    for (vec) |v| resident += @intFromBool(v & 1 != 0);
    return resident * page;
}

// ── Row prefetch, safetensors header entries, bf16 rounding (shared with plugins through sdk.io_util) ──

/// Persistent gather workers. Every row's regions are one SSD read each on a
/// cold table (~100 us), 48 per token: serial mmap faults were ~5 ms of every
/// decode step, 16 fault threads ~0.7 ms, and more threads got SLOWER (faults on
/// one mapping serialize on the VM map lock), so workers `pread` instead and
/// dequantize their rows in place. Workers wake on a generation bump and count
/// themselves down; the caller spins (the job is ~100 us). `Table` declares
/// `pub fn regions(*const Table) usize` and
/// `pub fn preadSite(*const Table, row: u64, region: usize, buf: []u8) bool`.
pub fn PrefetchPool(comptime Table: type) type {
    return struct {
        const Self = @This();
        pub const N = 48;
        pub const MAX_ROWS = 64;
        pub const ROW_BUF = 512;
        mu: std.Io.Mutex = .init,
        cv: std.Io.Condition = .init,
        gen: u64 = 0,
        quit: bool = false,
        table: ?*const Table = null,
        rows: []const i64 = &.{},
        bufs: [MAX_ROWS][ROW_BUF]u8 = undefined,
        pending: std.atomic.Value(u32) = .init(0),
        failed: std.atomic.Value(u32) = .init(0),
        /// Fan-out rounds issued; the engagement counter a test reads.
        runs: std.atomic.Value(u64) = .init(0),
        threads: [N]std.Thread = undefined,

        pub fn create() !*Self {
            const a = std.heap.page_allocator;
            const p = try a.create(Self);
            p.* = .{};
            var started: usize = 0;
            errdefer {
                p.shutdown(started);
                a.destroy(p);
            }
            for (0..N) |i| {
                p.threads[i] = try std.Thread.spawn(.{ .stack_size = 64 * 1024 }, worker, .{ p, i });
                started += 1;
            }
            return p;
        }

        pub fn destroy(self: *Self) void {
            self.shutdown(N);
            std.heap.page_allocator.destroy(self);
        }

        fn shutdown(self: *Self, started: usize) void {
            const io = std.Io.Threaded.global_single_threaded.io();
            self.mu.lockUncancelable(io);
            self.quit = true;
            self.cv.broadcast(io);
            self.mu.unlock(io);
            for (self.threads[0..started]) |t| t.join();
        }

        /// Fan the `regions * rows.len` preads over the workers; rows land in `bufs`.
        pub fn run(self: *Self, table: *const Table, rows: []const i64) bool {
            _ = self.runs.fetchAdd(1, .monotonic);
            const io = std.Io.Threaded.global_single_threaded.io();
            self.mu.lockUncancelable(io);
            self.table = table;
            self.rows = rows;
            self.failed.store(0, .release);
            self.pending.store(N, .release);
            self.gen += 1;
            self.cv.broadcast(io);
            self.mu.unlock(io);
            while (self.pending.load(.acquire) != 0) std.atomic.spinLoopHint();
            return self.failed.load(.acquire) == 0;
        }

        fn worker(self: *Self, idx: usize) void {
            const io = std.Io.Threaded.global_single_threaded.io();
            var seen: u64 = 0;
            while (true) {
                self.mu.lockUncancelable(io);
                while (self.gen == seen and !self.quit) self.cv.wait(io, &self.mu) catch {};
                if (self.quit) {
                    self.mu.unlock(io);
                    return;
                }
                seen = self.gen;
                const table = self.table.?;
                const rows = self.rows;
                self.mu.unlock(io);
                const k = table.regions();
                var i = idx;
                while (i < rows.len * k) : (i += N) {
                    if (!table.preadSite(@intCast(rows[i / k]), i % k, &self.bufs[i / k])) _ = self.failed.fetchAdd(1, .acq_rel);
                }
                _ = self.pending.fetchSub(1, .acq_rel);
            }
        }
    };
}

/// One safetensors header entry's 2-D region, as the header spells it (`start` / `end` relative to the data section).
pub const HeaderRegion = struct {
    rows: u64,
    cols: u64,
    start: u64,
    end: u64,
};

/// One header entry, every access checked and the region proven to hold exactly
/// `rows x cols x elem` bytes inside the mapping.
pub fn headerRegion(
    obj: std.json.ObjectMap,
    key: []const u8,
    dtype: []const u8,
    elem: u64,
    map_len: usize,
    data_off: usize,
) !HeaderRegion {
    const v = obj.get(key) orelse return error.NgramTableHeader;
    if (v != .object) return error.NgramTableHeader;
    const o = v.object;
    const dt = o.get("dtype") orelse return error.NgramTableHeader;
    if (dt != .string or !std.mem.eql(u8, dt.string, dtype)) return error.NgramTableHeader;
    const shape = o.get("shape") orelse return error.NgramTableHeader;
    if (shape != .array or shape.array.items.len != 2) return error.NgramTableHeader;
    if (shape.array.items[0] != .integer or shape.array.items[1] != .integer) return error.NgramTableHeader;
    const dofs = o.get("data_offsets") orelse return error.NgramTableHeader;
    if (dofs != .array or dofs.array.items.len != 2) return error.NgramTableHeader;
    if (dofs.array.items[0] != .integer or dofs.array.items[1] != .integer) return error.NgramTableHeader;

    const rows_i = shape.array.items[0].integer;
    const cols_i = shape.array.items[1].integer;
    const start_i = dofs.array.items[0].integer;
    const end_i = dofs.array.items[1].integer;
    if (rows_i <= 0 or cols_i <= 0 or start_i < 0 or end_i < start_i) return error.NgramTableRegion;
    const r: HeaderRegion = .{
        .rows = @intCast(rows_i),
        .cols = @intCast(cols_i),
        .start = @intCast(start_i),
        .end = @intCast(end_i),
    };
    if (r.cols > std.math.maxInt(u32) or r.rows > std.math.maxInt(u32)) return error.NgramTableRegion;
    const need = std.math.mul(u64, r.rows, r.cols * elem) catch return error.NgramTableRegion;
    if (r.end - r.start != need) return error.NgramTableRegion;
    const abs_end = std.math.add(u64, data_off, r.end) catch return error.NgramTableTruncated;
    if (abs_end > map_len) return error.NgramTableTruncated;
    return r;
}

/// Round-to-nearest-even f32 -> bf16 bits.
pub fn bf16Rne(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u +% 0x7FFF +% ((u >> 16) & 1)) >> 16);
}

// ── Tests (temp files; the reads checked against a plain pread of the same bytes) ──

const testing = std.testing;

/// A temp file of `len` pattern bytes and its absolute path.
const TmpFile = struct {
    dir: std.testing.TmpDir,
    path: [:0]u8,
    image: []u8,

    fn init(len: usize) !TmpFile {
        var dir = std.testing.tmpDir(.{});
        errdefer dir.cleanup();
        const image = try testing.allocator.alloc(u8, len);
        errdefer testing.allocator.free(image);
        for (image, 0..) |*b, i| b.* = @truncate((i *% 2654435761) >> 11);
        try dir.dir.writeFile(testing.io, .{ .sub_path = "f.bin", .data = image });
        var root: [512]u8 = undefined;
        const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/f.bin", .{root[0..try dir.dir.realPath(testing.io, &root)]}, 0);
        return .{ .dir = dir, .path = path, .image = image };
    }

    fn deinit(t: *TmpFile) void {
        testing.allocator.free(t.path);
        testing.allocator.free(t.image);
        t.dir.cleanup();
    }
};

test "io_util: readAligned equals the file's bytes at unaligned offsets and lengths, across stages, zero-length and to the last byte" {
    const page = std.heap.pageSize();
    // Past one 8 MiB stage, ending inside a page.
    var t = try TmpFile.init(stage_bytes + 3 * page + 123);
    defer t.deinit();
    const fd = try openNoCache(t.path.ptr, .{});
    defer _ = std.c.close(fd);
    const buf = try testing.allocator.alloc(u8, stage_bytes + 2 * page);
    defer testing.allocator.free(buf);
    const Case = struct { off: u64, len: usize };
    for ([_]Case{
        .{ .off = 0, .len = 1 },
        .{ .off = 1, .len = page },
        .{ .off = page - 1, .len = 2 },
        .{ .off = 7, .len = stage_bytes + page + 5 }, // two stages, both ends unaligned
        .{ .off = t.image.len - 10, .len = 10 }, // the last bytes (the file ends inside the page)
        .{ .off = 5, .len = 0 },
    }) |cs| {
        @memset(buf, 0xAA);
        try readAligned(fd, buf[0..cs.len], cs.off);
        try testing.expectEqualSlices(u8, t.image[@intCast(cs.off)..][0..cs.len], buf[0..cs.len]);
        if (cs.len < buf.len) try testing.expectEqual(@as(u8, 0xAA), buf[cs.len]);
    }
    // Past the end: short, by name.
    try testing.expectError(error.ReadShort, readAligned(fd, buf[0..20], t.image.len - 10));
    try testing.expectError(error.ReadShort, readAligned(fd, buf[0..1], t.image.len + page));
    // A bad descriptor: the read fails by name.
    try testing.expectError(error.ReadFailed, readAligned(-1, buf[0..1], 0));
}

test "io_util: the past-the-cache open refuses a missing file and a symlink it may not follow; the whole-file read keeps its limit" {
    var t = try TmpFile.init(5000);
    defer t.deinit();
    try testing.expectError(error.FileNotFound, openNoCache("/nonexistent/cov-d/f.bin", .{}));
    var lbuf: [700]u8 = undefined;
    const link = try std.fmt.bufPrintSentinel(&lbuf, "{s}.link", .{t.path}, 0);
    try testing.expectEqual(@as(c_int, 0), std.c.symlink(t.path.ptr, link.ptr));
    try testing.expectError(error.OpenFailed, openNoCache(link.ptr, .{ .follow_symlinks = false }));
    const fd = try openNoCache(link.ptr, .{ .read_ahead = true });
    _ = std.c.close(fd);
    const all = try readAllNoCache(testing.allocator, t.path, 5000);
    defer testing.allocator.free(all);
    try testing.expectEqualSlices(u8, t.image, all);
    try testing.expectError(error.FileTooBig, readAllNoCache(testing.allocator, t.path, 4999));
    try testing.expectError(error.FileNotFound, readAllNoCache(testing.allocator, "/nonexistent/cov-d/f.bin", 10));
    try testing.expectError(error.NoCacheFcntl, noCache(-1, .{}));
}

test "io_util: the row gather refuses its geometry by name, and a table shorter than its rows fails the gather (caller alone and with helpers)" {
    const page = std.heap.pageSize();
    try testing.expectError(error.GatherGeometry, RowGather.init(-1, 0, 0, 10, 0, 8, 2));
    try testing.expectError(error.GatherGeometry, RowGather.init(-1, 0, 100, 10, 0, 0, 2));
    try testing.expectError(error.GatherGeometry, RowGather.init(-1, 0, RowGather.stage_len - page, 10, 0, 8, 2));
    // 40 rows declared, the file holds 10 rows of 1000 B after a 100 B prefix.
    var t = try TmpFile.init(100 + 10 * 1000);
    defer t.deinit();
    const fd = try openNoCache(t.path.ptr, .{});
    defer _ = std.c.close(fd);
    for ([_]usize{ 0, 3 }) |helpers| {
        const rg = try RowGather.init(fd, 100, 1000, 40, helpers, 16, 2);
        defer rg.deinit();
        var out: [3 * 1000]u8 = undefined;
        try rg.gather(&.{ 9, 0, 9 }, &out);
        for ([_]u32{ 9, 0, 9 }, 0..) |r, i| try testing.expectEqualSlices(u8, t.image[100 + r * 1000 ..][0..1000], out[i * 1000 ..][0..1000]);
        // Rows far past the file's end: their runs read nothing, the gather fails by name.
        try testing.expectError(error.GatherRead, rg.gather(&.{ 39, 0, 30 }, &out));
        // The row gather's reads are aligned by construction; a hand-made unaligned run is refused.
        try testing.expect(!rg.aligned(.{ .off = 0, .len = page, .need = page + 1, .first = 0, .end = 1 }, rg.stages[0]));
        try testing.expect(!rg.aligned(.{ .off = 0, .len = 2 * RowGather.stage_len, .need = 1, .first = 0, .end = 1 }, rg.stages[0]));
    }
}

test "io_util: residency of an empty file is 0, a missing file refuses, and a file's resident bytes never exceed its pages" {
    const page = std.heap.pageSize();
    var t = try TmpFile.init(3 * page + 1);
    defer t.deinit();
    const r = try residentBytes(t.path);
    try testing.expect(r <= 4 * page);
    try testing.expectError(error.OpenFailed, residentBytes("/nonexistent/cov-d/f.bin"));
    var e = try TmpFile.init(0);
    defer e.deinit();
    try testing.expectEqual(@as(u64, 0), try residentBytes(e.path));
}

