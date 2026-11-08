//! Engram rows kept across a conversation's turns in an LRU (`MLX_SERVE_DSV41_ENGRAM_LRU_MB`, read at load: its size in
//! MiB, 1 to 16384; unset: 256), a decode read's misses read at high parallelism.
//!
//! A DSpark verify's 6 rows read 24 n-gram ids a table, past the page cache (the direct Engram reads; a pread a
//! half-row in oMLX's layout, one a record in a repacked table), on a prefetch thread from the forward's start, a row a
//! GCD work item (about 9-10 reads in flight); the first Engram layer waits for the first table's: ~2.0 ms a round, the
//! GPU idle ~1.0 ms of it (tp1, w-20261101-1811: 0.98 ms median at layer 1's entry, the verify's largest gap). Instead:
//! - one LRU of rows over the tables (keyed by table and id, `lru_mb` MiB with its index), kept across requests: a
//!   prompt grid's distinct rows go in as the whole-table read hands its spans over (ENGRAM_DIRECT; 310,174 rows over
//!   the two tables for a 16K-token prompt), and so does every row a decode read reads;
//! - a decode read takes every id the LRU holds from it and reads the others on a pool of `pool_threads` threads, each
//!   read (a repacked table's record, else a row's codes or its scales) a work item;
//! - a round's bonus token (`t1`, the verify's first row, known before its draft runs) has its rows read when the round
//!   starts, on a GCD worker, into the LRU, where the verify after the draft finds them.
//! The LRU's bytes are billed at the load beside the host's own (`bill.engram_store_bytes`): fewer slot rows where they
//! do not fit beside them; the load line says how many. Exact: a row's bytes are the table's whichever way they come.

const std = @import("std");
const builtin = @import("builtin");

/// The LRU's bytes in MiB (`MLX_SERVE_DSV41_ENGRAM_LRU_MB`), set at load.
/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    lru_mb: u32 = default_mb,
    lru: Lru = .{},
    mutex: Lock = .{},
    pool: Pool = .{},
    early_jobs: [max_tables]Early = @splat(.{}),
};
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(State, "dsv41_engram_decode", .{});
pub inline fn ps() *State {
    return shared_state.get();
}
pub const default_mb: u32 = 256;
pub const max_mb: u32 = 16384;
/// The decode reads' pool (the caller reads beside it).
pub const pool_threads = 64;
/// The most ids one early read takes (engram.zig's `MAX_COLS`), and the tables it may serve.
pub const max_early = 112;
pub const max_tables = 4;
/// The longest row part (codes, scales or biases) an early read holds.
pub const max_part = 512;

pub const Error = error{EngramLruSetting};

/// `value`: null or empty `default_mb`; a whole number of MiB in [1, `max_mb`]; anything else refused by name.
pub fn parseMb(value: ?[]const u8) Error!u32 {
    const v = value orelse return default_mb;
    if (v.len == 0) return default_mb;
    const n = std.fmt.parseInt(u32, v, 10) catch return error.EngramLruSetting;
    if (n == 0 or n > max_mb) return error.EngramLruSetting;
    return n;
}

/// What the load bills for the LRU (`bill.engram_store_bytes`): the bytes `setting` (`MLX_SERVE_DSV41_ENGRAM_LRU_MB`)
/// names, read here because the bill runs before the load sets `lru_mb`; a refused value bills the default (the load
/// refuses it by name).
pub fn billBytes(setting: ?[]const u8) u64 {
    return @as(u64, parseMb(setting) catch default_mb) << 20;
}

/// A pthread mutex: the LRU's users are the prefetch threads, GCD workers and the model's thread, with no `Io` among them.
/// None on Windows, which opens no Engram table (`engram.Table.open`).
const Lock = struct {
    m: if (windows) struct {} else std.c.pthread_mutex_t = .{},
    const windows = builtin.os.tag == .windows;
    fn lock(l: *Lock) void {
        if (comptime !windows) _ = std.c.pthread_mutex_lock(&l.m);
    }
    fn unlock(l: *Lock) void {
        if (comptime !windows) _ = std.c.pthread_mutex_unlock(&l.m);
    }
};

const alloc = std.heap.c_allocator;
const nil: u32 = std.math.maxInt(u32);

/// The LRU: `cap` slots of a row each (codes, scales, biases: `wlen + slen + blen` B, the geometry of the first table it
/// serves; a table of another geometry is read as before), found by key through chained buckets, ordered by use in a
/// list (head: the most recent). Its arrays come to at most `lru_mb` MiB.
const Lru = struct {
    ready: bool = false,
    wlen: usize = 0,
    slen: usize = 0,
    blen: usize = 0,
    row: usize = 0,
    cap: u32 = 0,
    used: u32 = 0,
    keys: []u64 = &.{},
    data: []u8 = &.{},
    hnext: []u32 = &.{},
    prev: []u32 = &.{},
    next: []u32 = &.{},
    buckets: []u32 = &.{},
    shift: u6 = 0,
    head: u32 = nil,
    tail: u32 = nil,

    /// The arrays for `budget` B at this geometry: `cap` rows of `row + 20` B and a bucket for each (rounded up to a
    /// power of two: at most 8 B a row more). False (nothing kept) where they cannot be had.
    fn init(l: *Lru, budget: u64, wlen: usize, slen: usize, blen: usize) bool {
        const row = wlen + slen + blen;
        const n64 = budget / (row + 28);
        if (row == 0 or n64 < 2) return false;
        const n: u32 = @intCast(@min(n64, nil - 1));
        var nb: u64 = 2;
        var bits: u6 = 1;
        while (nb < n) {
            nb <<= 1;
            bits += 1;
        }
        l.* = .{ .wlen = wlen, .slen = slen, .blen = blen, .row = row, .cap = n, .shift = @intCast(64 - @as(u7, bits)) };
        l.keys = alloc.alloc(u64, n) catch return l.drop();
        l.data = alloc.alloc(u8, @as(usize, n) * row) catch return l.drop();
        l.hnext = alloc.alloc(u32, n) catch return l.drop();
        l.prev = alloc.alloc(u32, n) catch return l.drop();
        l.next = alloc.alloc(u32, n) catch return l.drop();
        l.buckets = alloc.alloc(u32, @intCast(nb)) catch return l.drop();
        @memset(l.buckets, nil);
        l.ready = true;
        return true;
    }

    /// Everything let go of; false.
    fn drop(l: *Lru) bool {
        alloc.free(l.keys);
        alloc.free(l.data);
        alloc.free(l.hnext);
        alloc.free(l.prev);
        alloc.free(l.next);
        alloc.free(l.buckets);
        l.* = .{};
        return false;
    }

    fn fits(l: *const Lru, wlen: usize, slen: usize, blen: usize) bool {
        return l.ready and l.wlen == wlen and l.slen == slen and l.blen == blen;
    }

    fn bucketOf(l: *const Lru, key: u64) usize {
        return @intCast((key *% 0x9E3779B97F4A7C15) >> l.shift);
    }

    fn find(l: *const Lru, key: u64) ?u32 {
        var s = l.buckets[l.bucketOf(key)];
        while (s != nil) : (s = l.hnext[s]) {
            if (l.keys[s] == key) return s;
        }
        return null;
    }

    fn unlinkList(l: *Lru, s: u32) void {
        const p = l.prev[s];
        const n = l.next[s];
        if (p != nil) l.next[p] = n else l.head = n;
        if (n != nil) l.prev[n] = p else l.tail = p;
    }

    fn pushFront(l: *Lru, s: u32) void {
        l.prev[s] = nil;
        l.next[s] = l.head;
        if (l.head != nil) l.prev[l.head] = s;
        l.head = s;
        if (l.tail == nil) l.tail = s;
    }

    fn touch(l: *Lru, s: u32) void {
        if (l.head == s) return;
        l.unlinkList(s);
        l.pushFront(s);
    }

    fn unlinkHash(l: *Lru, s: u32) void {
        const bk = l.bucketOf(l.keys[s]);
        var p: u32 = nil;
        var x = l.buckets[bk];
        while (x != nil and x != s) : (x = l.hnext[x]) p = x;
        if (x == nil) return;
        if (p == nil) l.buckets[bk] = l.hnext[s] else l.hnext[p] = l.hnext[s];
    }

    /// Row `key`'s bytes in: an id already held only made the most recent (its bytes are the table's either way), a new
    /// one in a free slot or the least recently used one's.
    fn put(l: *Lru, key: u64, w: []const u8, s: []const u8, b: []const u8) void {
        if (l.find(key)) |x| {
            l.touch(x);
            return;
        }
        var x: u32 = undefined;
        if (l.used < l.cap) {
            x = l.used;
            l.used += 1;
        } else {
            x = l.tail;
            l.unlinkList(x);
            l.unlinkHash(x);
        }
        l.keys[x] = key;
        const bk = l.bucketOf(key);
        l.hnext[x] = l.buckets[bk];
        l.buckets[bk] = x;
        l.pushFront(x);
        const d = l.data[@as(usize, x) * l.row ..][0..l.row];
        @memcpy(d[0..l.wlen], w);
        @memcpy(d[l.wlen..][0..l.slen], s);
        @memcpy(d[l.wlen + l.slen ..][0..l.blen], b);
    }

    /// Row `key`'s bytes copied out when held (it made the most recent).
    fn get(l: *Lru, key: u64, w: []u8, s: []u8, b: []u8) bool {
        const x = l.find(key) orelse return false;
        l.touch(x);
        const d = l.data[@as(usize, x) * l.row ..][0..l.row];
        @memcpy(w, d[0..l.wlen]);
        @memcpy(s, d[l.wlen..][0..l.slen]);
        @memcpy(b, d[l.wlen + l.slen ..][0..l.blen]);
        return true;
    }

    /// The rows held now and the arrays' bytes.
    fn bytes(l: *const Lru) u64 {
        return @as(u64, l.cap) * (l.row + 20) + l.buckets.len * @sizeOf(u32);
    }
};


fn keyOf(fd: c_int, id: u32) u64 {
    return (@as(u64, @as(u32, @bitCast(fd))) << 32) | id;
}

/// The LRU at this geometry: made at its first use (`lru_mb` MiB), null where it serves another geometry or cannot be
/// had. Under the mutex.
fn lruFor(wlen: usize, slen: usize, blen: usize) ?*Lru {
    if (!ps().lru.ready and !ps().lru.init(@as(u64, ps().lru_mb) << 20, wlen, slen, blen)) return null;
    return if (ps().lru.fits(wlen, slen, blen)) &ps().lru else null;
}

/// A prompt grid's table read whole: its distinct ids and their rows (`w`, `s`, `b` in the ids' order) into the LRU.
pub fn keep(fd: c_int, wlen: usize, slen: usize, blen: usize, ids: []const u32, w: []const u8, s: []const u8, b: []const u8) void {
    if (ids.len == 0) return;
    ps().mutex.lock();
    defer ps().mutex.unlock();
    const l = lruFor(wlen, slen, blen) orelse return;
    for (ids, 0..) |id, i| l.put(keyOf(fd, id), w[i * wlen ..][0..wlen], s[i * slen ..][0..slen], b[i * blen ..][0..blen]);
}

/// A decode read of table `fd`'s `ids`: each one the LRU holds copied into its place in `w`, `s` and `b` (rows of `wlen`,
/// `slen`, `blen` B), `held` saying which; the count held.
pub fn take(fd: c_int, ids: []const u32, wlen: usize, slen: usize, blen: usize, w: []u8, s: []u8, b: []u8, held: []bool) usize {
    @memset(held, false);
    ps().mutex.lock();
    defer ps().mutex.unlock();
    const l = lruFor(wlen, slen, blen) orelse return 0;
    var n: usize = 0;
    for (ids, 0..) |id, i| {
        if (!l.get(keyOf(fd, id), w[i * wlen ..][0..wlen], s[i * slen ..][0..slen], b[i * blen ..][0..blen])) continue;
        held[i] = true;
        n += 1;
    }
    return n;
}

/// The rows a decode read read (those `held` does not mark) into the LRU.
pub fn putMisses(fd: c_int, ids: []const u32, held: []const bool, wlen: usize, slen: usize, blen: usize, w: []const u8, s: []const u8, b: []const u8) void {
    ps().mutex.lock();
    defer ps().mutex.unlock();
    const l = lruFor(wlen, slen, blen) orelse return;
    for (ids, 0..) |id, i| {
        if (i < held.len and held[i]) continue;
        l.put(keyOf(fd, id), w[i * wlen ..][0..wlen], s[i * slen ..][0..slen], b[i * blen ..][0..blen]);
    }
}

// ── the decode reads' pool ──

/// One parallel job: `f(ctx, i)` for each `i` below `n`, claimed one at a time.
const Job = struct {
    ctx: *const anyopaque,
    f: *const fn (ctx: *const anyopaque, i: usize) void,
    n: usize,
    claim: std.atomic.Value(usize) = .init(0),
    done: std.atomic.Value(usize) = .init(0),

    fn drain(j: *Job) void {
        while (true) {
            const i = j.claim.fetchAdd(1, .monotonic);
            if (i >= j.n) return;
            j.f(j.ctx, i);
            _ = j.done.fetchAdd(1, .release);
        }
    }
};

/// `pool_threads` threads parked on a GCD semaphore; the job they help with (`current`, on its caller's stack), and how
/// many of them are looking at it (the caller lets it go only when none is).
const Pool = struct {
    started: bool = false,
    width: usize = 0,
    wake: ?*anyopaque = null,
    current: std.atomic.Value(?*Job) = .init(null),
    inside: std.atomic.Value(u32) = .init(0),
    gate: Lock = .{},
};

const dispatch_time_forever: u64 = std.math.maxInt(u64);
extern "c" fn dispatch_semaphore_create(value: isize) ?*anyopaque;
extern "c" fn dispatch_semaphore_wait(dsema: ?*anyopaque, timeout: u64) isize;
extern "c" fn dispatch_semaphore_signal(dsema: ?*anyopaque) isize;

fn worker() void {
    while (true) {
        _ = dispatch_semaphore_wait(ps().pool.wake, dispatch_time_forever);
        _ = ps().pool.inside.fetchAdd(1, .seq_cst);
        if (ps().pool.current.load(.seq_cst)) |j| j.drain();
        _ = ps().pool.inside.fetchSub(1, .seq_cst);
    }
}

/// The pool's threads, at its first job (as many as start; none: the caller reads alone). Under `pool.gate`.
fn startPool() void {
    ps().pool.started = true;
    if (comptime !builtin.os.tag.isDarwin()) return;
    ps().pool.wake = dispatch_semaphore_create(0) orelse return;
    for (0..pool_threads) |_| {
        const t = std.Thread.spawn(.{ .stack_size = 512 * 1024 }, worker, .{}) catch break;
        t.detach();
        ps().pool.width += 1;
    }
}

/// `f(ctx, i)` for every `i` below `n`, on the pool's threads and the caller's; returns when all are done. One job at a
/// time.
pub fn parallel(ctx: *const anyopaque, n: usize, f: *const fn (ctx: *const anyopaque, i: usize) void) void {
    if (n == 0) return;
    ps().pool.gate.lock();
    defer ps().pool.gate.unlock();
    if (!ps().pool.started) startPool();
    var job: Job = .{ .ctx = ctx, .f = f, .n = n };
    const k = @min(n - 1, ps().pool.width);
    if (k > 0) {
        ps().pool.current.store(&job, .seq_cst);
        for (0..k) |_| _ = dispatch_semaphore_signal(ps().pool.wake);
    }
    job.drain();
    while (job.done.load(.acquire) < n) std.atomic.spinLoopHint();
    if (k > 0) {
        ps().pool.current.store(null, .seq_cst);
        while (ps().pool.inside.load(.seq_cst) != 0) std.atomic.spinLoopHint();
    }
}

// ── a round's bonus row, read early ──

/// An early read's reader: `ids`' rows of the table at `ctx` into `w`, `s` and `b` (`Table.read`'s layout); false on a
/// failed read.
pub const ReadFn = *const fn (ctx: *const anyopaque, ids: []const u32, w: []u8, s: []u8, b: []u8) bool;

/// A table's early read in flight: what it reads and where.
const Early = struct {
    busy: std.atomic.Value(bool) = .init(false),
    ctx: *const anyopaque = undefined,
    read: ReadFn = undefined,
    fd: c_int = 0,
    wlen: usize = 0,
    slen: usize = 0,
    blen: usize = 0,
    n: usize = 0,
    ids: [max_early]u32 = undefined,
    w: [max_early * max_part]u8 = undefined,
    s: [max_early * max_part]u8 = undefined,
    b: [max_early * max_part]u8 = undefined,

    fn run(ctx: ?*anyopaque) callconv(.c) void {
        const e: *Early = @ptrCast(@alignCast(ctx.?));
        defer e.busy.store(false, .release);
        const n = e.n;
        const w = e.w[0 .. n * e.wlen];
        const s = e.s[0 .. n * e.slen];
        const b = e.b[0 .. n * e.blen];
        if (!e.read(e.ctx, e.ids[0..n], w, s, b)) return;
        putMisses(e.fd, e.ids[0..n], &.{}, e.wlen, e.slen, e.blen, w, s, b);
    }
};

const qos_user_initiated: isize = 0x19;
extern "c" fn dispatch_get_global_queue(identifier: isize, flags: usize) ?*anyopaque;
extern "c" fn dispatch_async_f(queue: ?*anyopaque, context: ?*anyopaque, work: *const fn (?*anyopaque) callconv(.c) void) void;

/// The bonus token's ids of table `slot` (descriptor `fd`, the table at `ctx`, read by `read`): those the LRU does not
/// hold read now on a GCD worker into it; returns at once. Skipped while the table's last one is still in flight.
pub fn early(slot: usize, ctx: *const anyopaque, read: ReadFn, fd: c_int, wlen: usize, slen: usize, blen: usize, ids: []const u32) void {
    if (comptime !builtin.os.tag.isDarwin()) return;
    if (slot >= max_tables or ids.len == 0 or ids.len > max_early or wlen > max_part or slen > max_part or blen > max_part) return;
    const e = &ps().early_jobs[slot];
    if (e.busy.swap(true, .acquire)) return;
    var n: usize = 0;
    {
        ps().mutex.lock();
        defer ps().mutex.unlock();
        const l = lruFor(wlen, slen, blen);
        for (ids) |id| {
            if (l) |x| if (x.find(keyOf(fd, id)) != null) continue;
            if (std.mem.indexOfScalar(u32, e.ids[0..n], id) != null) continue;
            e.ids[n] = id;
            n += 1;
        }
    }
    if (n == 0) {
        e.busy.store(false, .release);
        return;
    }
    e.ctx = ctx;
    e.read = read;
    e.fd = fd;
    e.wlen = wlen;
    e.slen = slen;
    e.blen = blen;
    e.n = n;
    dispatch_async_f(dispatch_get_global_queue(qos_user_initiated, 0), e, Early.run);
}

/// At a table's close: the early reads in flight waited for, then the LRU let go of (the next load makes a new one; a
/// later table may take the same descriptor).
pub fn forget(fd: c_int) void {
    _ = fd;
    for (&ps().early_jobs) |*e| while (e.busy.load(.acquire)) std.Thread.yield() catch {};
    ps().mutex.lock();
    defer ps().mutex.unlock();
    if (ps().lru.ready) _ = ps().lru.drop();
}

/// The LRU's rows held, its slots and its arrays' bytes now.
pub fn size() struct { rows: u64, cap: u64, bytes: u64 } {
    ps().mutex.lock();
    defer ps().mutex.unlock();
    return .{ .rows = ps().lru.used, .cap = ps().lru.cap, .bytes = if (ps().lru.ready) ps().lru.bytes() else 0 };
}

// ── tests ──

const testing = std.testing;

fn resetForTest() void {
    if (ps().lru.ready) _ = ps().lru.drop();
}

test "engram decode setting: the LRU's MiB (unset 256; 1 to 16384), anything else refused by name" {
    try testing.expectEqual(@as(u32, 256), try parseMb(null));
    try testing.expectEqual(@as(u32, 256), try parseMb(""));
    try testing.expectEqual(@as(u32, 1), try parseMb("1"));
    try testing.expectEqual(@as(u32, 16384), try parseMb("16384"));
    for ([_][]const u8{ "0", "16385", "-1", "256M", " 256", "1.5" }) |v| try testing.expectError(error.EngramLruSetting, parseMb(v));
    // Billed at the size its setting names, before a load sets `lru_mb`; a refused value bills the default.
    try testing.expectEqual(@as(u64, 300 << 20), billBytes("300"));
    try testing.expectEqual(@as(u64, default_mb) << 20, billBytes("lots"));
}

test "engram decode LRU: a grid's rows taken by id across tables, the least recently used giving way, a used row kept" {
    const saved = ps().lru_mb;
    defer {
        ps().lru_mb = saved;
        resetForTest();
    }
    ps().lru_mb = 1;
    resetForTest();
    // 1 MiB at 8 + 4 B rows (40 B with the index): 26,214 slots
    const a: c_int = 9999;
    const z: c_int = 9998;
    const w37 = [16]u8{ 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 };
    const s37 = [8]u8{ 10, 10, 10, 10, 20, 20, 20, 20 };
    const w7z: [8]u8 = @splat(5);
    const s7z: [4]u8 = @splat(50);
    keep(a, 8, 4, 0, &.{ 3, 7 }, &w37, &s37, &.{});
    keep(z, 8, 4, 0, &.{7}, &w7z, &s7z, &.{});
    try testing.expectEqual(@as(u64, 3), size().rows);
    try testing.expectEqual(@as(u64, (1 << 20) / 40), size().cap);
    try testing.expect(size().bytes <= 1 << 20);
    var w: [24]u8 = @splat(0);
    var s: [12]u8 = @splat(0);
    var held: [3]bool = undefined;
    // table a's 7 and 3 held, 5 not; table z's 7 is its own row
    try testing.expectEqual(@as(usize, 2), take(a, &.{ 7, 5, 3 }, 8, 4, 0, &w, &s, &.{}, &held));
    try testing.expectEqualSlices(bool, &.{ true, false, true }, &held);
    try testing.expectEqualSlices(u8, &[24]u8{ 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 }, &w);
    try testing.expectEqualSlices(u8, &[12]u8{ 20, 20, 20, 20, 0, 0, 0, 0, 10, 10, 10, 10 }, &s);
    try testing.expectEqual(@as(usize, 1), take(z, &.{7}, 8, 4, 0, w[0..8], s[0..4], &.{}, held[0..1]));
    try testing.expectEqual(@as(u8, 5), w[0]);
    // the read row (5) put in; held rows are not put again
    @memset(w[8..16], 9);
    putMisses(a, &.{ 7, 5, 3 }, &.{ true, false, true }, 8, 4, 0, &w, &s, &.{});
    try testing.expectEqual(@as(u64, 4), size().rows);
    try testing.expectEqual(@as(usize, 1), take(a, &.{5}, 8, 4, 0, w[0..8], s[0..4], &.{}, held[0..1]));
    try testing.expectEqual(@as(u8, 9), w[0]);
    // another geometry is not served
    try testing.expectEqual(@as(usize, 0), take(a, &.{7}, 16, 4, 0, w[0..16], s[0..4], &.{}, held[0..1]));
    // a grid of cap - 1 new rows: three of the four held give way in order of use, table a's 3 (used just now) stays
    const cap: usize = @intCast(size().cap);
    try testing.expectEqual(@as(usize, 1), take(a, &.{3}, 8, 4, 0, w[0..8], s[0..4], &.{}, held[0..1]));
    const ids = try testing.allocator.alloc(u32, cap - 1);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*x, i| x.* = @intCast(1000 + i);
    const wb = try testing.allocator.alloc(u8, ids.len * 8);
    defer testing.allocator.free(wb);
    @memset(wb, 7);
    const sb = try testing.allocator.alloc(u8, ids.len * 4);
    defer testing.allocator.free(sb);
    @memset(sb, 7);
    keep(a, 8, 4, 0, ids, wb, sb, &.{});
    try testing.expectEqual(@as(u64, cap), size().rows);
    // given way: z's 7, a's 7 and 5 (used before a's 3); a's 3 and the grid's rows held
    try testing.expectEqual(@as(usize, 0), take(z, &.{7}, 8, 4, 0, w[0..8], s[0..4], &.{}, held[0..1]));
    try testing.expectEqual(@as(usize, 0), take(a, &.{ 7, 5 }, 8, 4, 0, w[0..16], s[0..8], &.{}, held[0..2]));
    try testing.expectEqual(@as(usize, 2), take(a, &.{ 3, 1000 }, 8, 4, 0, w[0..16], s[0..8], &.{}, held[0..2]));
    try testing.expectEqual(@as(u8, 1), w[0]);
    try testing.expectEqual(@as(u8, 7), w[8]);
    // dropped at a table's close
    forget(a);
    try testing.expectEqual(@as(u64, 0), size().rows);
}

test "engram decode pool: every item once, on the pool's threads and the caller's, job after job" {
    const Count = struct {
        hits: [1000]std.atomic.Value(u32) = @splat(.init(0)),
        fn one(ctx: *const anyopaque, i: usize) void {
            const c: *const @This() = @ptrCast(@alignCast(ctx));
            _ = @constCast(&c.hits[i]).fetchAdd(1, .monotonic);
        }
    };
    for ([_]usize{ 1, 2, 65, 1000, 7, 1000 }) |n| {
        var c: Count = .{};
        parallel(&c, n, Count.one);
        for (c.hits[0..n]) |*h| try testing.expectEqual(@as(u32, 1), h.load(.monotonic));
        for (c.hits[n..]) |*h| try testing.expectEqual(@as(u32, 0), h.load(.monotonic));
    }
    if (comptime builtin.os.tag.isDarwin()) try testing.expect(ps().pool.width > 0);
}

test "engram decode early read: the ids the LRU does not hold read on a GCD worker into it" {
    if (comptime !builtin.os.tag.isDarwin()) return error.SkipZigTest;
    const saved = ps().lru_mb;
    defer {
        ps().lru_mb = saved;
        resetForTest();
    }
    ps().lru_mb = 1;
    resetForTest();
    const fd: c_int = 4444;
    keep(fd, 2, 1, 0, &.{5}, &.{ 5, 5 }, &.{50}, &.{});
    const Fake = struct {
        var calls: usize = 0;
        var got: [8]u32 = undefined;
        var n: usize = 0;
        fn read(ctx: *const anyopaque, ids: []const u32, w: []u8, s: []u8, b: []u8) bool {
            _ = ctx;
            _ = b;
            calls += 1;
            n = ids.len;
            for (ids, 0..) |id, i| {
                got[i] = id;
                w[2 * i] = @truncate(id);
                w[2 * i + 1] = @truncate(id);
                s[i] = @truncate(id * 10);
            }
            return true;
        }
    };
    const tag: u8 = 0;
    // 5 is held, 6 repeats: 6 and 8 read
    early(1, &tag, Fake.read, fd, 2, 1, 0, &.{ 6, 5, 8, 6 });
    while (ps().early_jobs[1].busy.load(.acquire)) std.Thread.yield() catch {};
    try testing.expectEqual(@as(usize, 1), Fake.calls);
    try testing.expectEqualSlices(u32, &.{ 6, 8 }, Fake.got[0..Fake.n]);
    var w: [6]u8 = undefined;
    var s: [3]u8 = undefined;
    var held: [3]bool = undefined;
    try testing.expectEqual(@as(usize, 3), take(fd, &.{ 8, 5, 6 }, 2, 1, 0, &w, &s, &.{}, &held));
    try testing.expectEqualSlices(u8, &.{ 8, 8, 5, 5, 6, 6 }, &w);
    try testing.expectEqualSlices(u8, &.{ 80, 50, 60 }, &s);
    // nothing new: no read
    early(1, &tag, Fake.read, fd, 2, 1, 0, &.{ 5, 6 });
    while (ps().early_jobs[1].busy.load(.acquire)) std.Thread.yield() catch {};
    try testing.expectEqual(@as(usize, 1), Fake.calls);
}
