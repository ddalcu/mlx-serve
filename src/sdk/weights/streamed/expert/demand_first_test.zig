//! The demand-first order's byte gate (df1), MLX-free: the read pool's C alone, built with its test hooks (-DQ3LD_INJECT: scripted per-offset preadv faults and sleeps, a
//! delay before each of the lookahead's reads and before each demand range), reading the records of a temp file through
//! the lookahead (the speculative class) and the demand path at each mode (0, 1, 2), the gate/up split on as served.
//! Each scenario checks that every slot row holds the file's bytes and every ticket ends OK:
//!   - claimed in flight: a record claimed while its lookahead read is slowed. Modes 0 and 1 wait for the lookahead
//!     thread; mode 2's demand workers read the untaken pieces themselves.
//!   - demand beside the lookahead: a demand read (held 200 ms) while an unclaimed record reads. In modes 1 and 2 no
//!     unclaimed piece starts while demand runs: the record stops at its next piece and lands after.
//!   - a failed piece: the record's second piece fails (EIO) once. The record fails and its ranges read the stock way.
//!   - rounds: lookahead steps and demand submits with random delays (claims in flight, landed records, records the
//!     lookahead never read, settles dropping older records).
//! The records are 9,000,000 B (gate, up, down: 3,000,000, 3,100,000, 2,900,000) at a 4,096 B stride, so record bases sit
//! off the pool's page.

const std = @import("std");
const testing = std.testing;

extern fn q3ld_spec_config(nthreads: i32, bufs: ?[*]const u64, nslots: i32, slot_bytes: i64, rec_len: i64, chunk: i64, counters: ?[*]i64) c_int;
extern fn q3ld_spec_streams(idle_busy: i32) c_int;
extern fn q3ld_direct_config(on: i32) c_int;
extern fn q3ld_parts_config(block: i64) c_int;
extern fn q3ld_start(nw: i32, staging_ptrs: [*]const u64, sbytes: i64, psize: i64, res: [*]i64, n_tickets: i64, log: [*]i64, n_log: i64, gauge: *[6]i64) c_int;
extern fn q3ld_submit(fd: i32, file_size: i64, deadline: i64, n: i32, ngu: i32, ndown: i32, offsets: [*]const i64, rows: [*]const [*]const u64, lens: [*]const i64, first: i64) c_int;
extern fn q3ld_spec_step_len(fd: i32, file_size: i64, cur: i64, n: i32, bases: ?[*]const i64, len: i64) i32;
extern fn q3ld_seq() i64;
extern fn q3ld_wait(seen: i64, timeout_ns: i64) i64;
extern fn q3ld_quiesce(timeout_ns: i64) c_int;
extern fn q3ld_stop() c_int;
extern fn q3ld_gu_split(mode: i32) c_int;
extern fn q3ld_demand_first(mode: i32) c_int;
extern fn q3ld_df_stats(out: *[df_n]i64, n: i32) i32;
extern fn q3ld_abi() i32;
/// The test hooks (-DQ3LD_INJECT): per-offset rules (code 2 EIO, 5 sleep `arg` ns, each used once), a uniform delay in
/// [0, max) before each demand range, a delay in [max / 2, max) before each lookahead read.
extern fn q3ld_test_rules(n: i32, off: [*]const i64, code: [*]const i64, arg: [*]const i64) void;
extern fn q3ld_test_delay(seed: u64, max_ns: i64) void;
extern fn q3ld_test_spec_delay(max_ns: i64) void;


const counters_n = 82;
const sc_started = 3;
const sc_landed = 4;
const sc_failed = 5;
const sc_claimed = 8;
const sc_adopt_ranges = 13;
const sc_max_busy_at_start = 20;
const sc_max_qlen_at_start = 21;
/// The pool's DF_* counters (q3ld_df_stats).
const df_n = 10;
const df_pieces = 0;
const df_stops = 1;
const df_help_runs = 3;
const df_failed = 9;
const res_w = 10;
const st_ok = 0;
const st_pending = -1;

const n_comp = 3;
const ngu = 2;
const comp_len = [n_comp]u64{ 3_000_000, 3_100_000, 2_900_000 };
const rec_len: u64 = 9_000_000;
const stride: u64 = 9_003_008; // rec_len rounded up to 4,096
const n_rec = 8;
const workers = 2;
const spec_threads = 2;
const n_spec = 4;
const staging_bytes: u64 = 9 << 20;
const tickets = 64;
const max_items = 3;
const ms: i64 = std.time.ns_per_ms;

const page = std.heap.page_size_min;
const Map = []align(page) u8;

fn mapBytes(n: usize) !Map {
    return std.posix.mmap(null, std.mem.alignForward(usize, n + page, page), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
}

fn base(r: usize) u64 {
    return r * stride;
}

/// The records' file: random bytes, opened as the reader opens (read-only, the page cache bypassed, no read-ahead).
const TempFile = struct {
    path: [:0]u8,
    fd: c_int,
    bytes: []u8,
    fn create(a: std.mem.Allocator, seed: u64) !TempFile {
        const bytes = try a.alloc(u8, n_rec * stride);
        errdefer a.free(bytes);
        var prng = std.Random.DefaultPrng.init(seed);
        prng.random().bytes(bytes);
        const path = try std.fmt.allocPrintSentinel(a, "/tmp/turboexl3_ds41-df1-{d}.bin", .{std.c.getpid()}, 0);
        errdefer a.free(path);
        const w = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (w < 0) return error.TempFile;
        var done: usize = 0;
        while (done < bytes.len) {
            const r = std.c.write(w, bytes[done..].ptr, bytes.len - done);
            if (r <= 0) {
                _ = std.c.close(w);
                return error.TempFile;
            }
            done += @intCast(r);
        }
        _ = std.c.close(w);
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.Open;
        if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0 or std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0)) != 0) {
            _ = std.c.close(fd);
            return error.Open;
        }
        return .{ .path = path, .fd = fd, .bytes = bytes };
    }
    fn destroy(f: TempFile, a: std.mem.Allocator) void {
        _ = std.c.close(f.fd);
        _ = std.c.unlink(f.path);
        a.free(f.path);
        a.free(f.bytes);
    }
    fn size(f: TempFile) i64 {
        return @intCast(f.bytes.len);
    }
};

/// No scripted fault or sleep, no delay (the hooks are the C file's statics: they outlive a pool).
fn hooksOff() void {
    const none = [1]i64{0};
    q3ld_test_rules(0, &none, &none, &none);
    q3ld_test_delay(1, 0);
    q3ld_test_spec_delay(0);
}

/// One pool at `mode` over its own staging, slots, status words and counters (each kept until the pool stops). `release`
/// (deferred before `start`) stops a pool still running and unmaps what `start` mapped; `finish` stops it checked.
const Harness = struct {
    counters: [counters_n]i64 = @splat(0),
    res: [tickets * res_w]i64 = @splat(0),
    log: [tickets]i64 = @splat(0),
    gauge: [6]i64 = @splat(0),
    /// The staging buffers, the lookahead's slots, then `max_items` records' rows (a map a component).
    maps: [2 + max_items * n_comp]Map = undefined,
    n_maps: usize = 0,
    next: usize = 0,
    running: bool = false,

    fn map(h: *Harness, n: usize) !Map {
        const m = try mapBytes(n);
        h.maps[h.n_maps] = m;
        h.n_maps += 1;
        return m;
    }

    fn row(h: *const Harness, i: usize, c: usize) Map {
        return h.maps[2 + i * n_comp + c];
    }

    fn start(h: *Harness, mode: i32) !void {
        const psize: u64 = std.heap.pageSize();
        const slot = std.mem.alignForward(u64, rec_len, psize) + 2 * psize;
        const stage = try h.map(workers * staging_bytes);
        const slots = try h.map(n_spec * slot);
        for (0..max_items) |_| for (comp_len) |len| {
            _ = try h.map(len);
        };
        var bufs: [n_spec]u64 = undefined;
        for (&bufs, 0..) |*b, i| b.* = @intFromPtr(slots.ptr) + i * slot;
        const chunk = std.mem.alignForward(u64, slot / 4, psize);
        try testing.expectEqual(@as(c_int, 0), q3ld_spec_config(spec_threads, &bufs, n_spec, @intCast(slot), @intCast(rec_len), @intCast(chunk), &h.counters));
        try testing.expectEqual(@as(c_int, 0), q3ld_spec_streams(0));
        try testing.expectEqual(@as(c_int, 0), q3ld_direct_config(0));
        try testing.expectEqual(@as(c_int, 0), q3ld_parts_config(0));
        var ptrs: [workers]u64 = undefined;
        for (&ptrs, 0..) |*p, w| p.* = @intFromPtr(stage.ptr) + w * staging_bytes;
        try testing.expectEqual(@as(c_int, 0), q3ld_start(workers, &ptrs, staging_bytes, @intCast(psize), &h.res, tickets, &h.log, tickets, &h.gauge));
        h.running = true;
        try testing.expectEqual(@as(c_int, 0), q3ld_gu_split(1));
        try testing.expectEqual(@as(c_int, 0), q3ld_demand_first(mode));
    }

    /// Quiesced (every read done) and stopped: the df1 counters at the end (the counters stay in `counters`).
    fn finish(h: *Harness) ![df_n]i64 {
        try testing.expectEqual(@as(c_int, 0), q3ld_quiesce(20 * std.time.ns_per_s));
        var df: [df_n]i64 = @splat(0);
        try testing.expectEqual(@as(i32, df_n), q3ld_df_stats(&df, df_n));
        try testing.expectEqual(@as(c_int, 0), q3ld_stop());
        h.running = false;
        return df;
    }

    fn release(h: *Harness) void {
        if (h.running) {
            _ = q3ld_quiesce(20 * std.time.ns_per_s);
            _ = q3ld_stop();
            h.running = false;
        }
        for (h.maps[0..h.n_maps]) |m| std.posix.munmap(m);
        h.n_maps = 0;
    }

    /// Demand for records `recs` (one submit, at most `max_items`): every row poisoned first, then the pool's ranges
    /// waited for; every ticket OK and every row the file's bytes.
    fn demand(h: *Harness, f: TempFile, recs: []const usize) !void {
        const n = recs.len;
        if (h.next + 2 * n > tickets) h.next = 0;
        var offsets: [2 * max_items]i64 = undefined;
        var addr: [max_items][n_comp]u64 = undefined;
        var item: [max_items][*]const u64 = undefined;
        const lens = [n_comp]i64{ @intCast(comp_len[0]), @intCast(comp_len[1]), @intCast(comp_len[2]) };
        for (recs, 0..) |r, i| {
            offsets[i] = @intCast(base(r));
            offsets[n + i] = @intCast(base(r) + comp_len[0] + comp_len[1]);
            for (0..n_comp) |c| {
                @memset(h.row(i, c)[0..comp_len[c]], 0xa5);
                addr[i][c] = @intFromPtr(h.row(i, c).ptr);
            }
            item[i] = &addr[i];
        }
        const first = h.next;
        try testing.expectEqual(@as(c_int, 0), q3ld_submit(f.fd, f.size(), -1, @intCast(n), ngu, n_comp - ngu, &offsets, &item, &lens, @intCast(first)));
        h.next += 2 * n;
        var seen = q3ld_seq();
        var rounds: u32 = 0;
        while (true) {
            var done = true;
            for (first..first + 2 * n) |t| {
                if (@atomicLoad(i64, &h.res[t * res_w], .acquire) == st_pending) done = false;
            }
            if (done) break;
            rounds += 1;
            if (rounds > 300) return error.PoolTimeout;
            seen = q3ld_wait(seen, 100 * std.time.ns_per_ms);
        }
        for (first..first + 2 * n) |t| try testing.expectEqual(@as(i64, st_ok), h.res[t * res_w]);
        for (recs, 0..) |r, i| {
            var at: u64 = base(r);
            for (0..n_comp) |c| {
                try testing.expectEqualSlices(u8, f.bytes[at..][0..comp_len[c]], h.row(i, c)[0..comp_len[c]]);
                at += comp_len[c];
            }
        }
    }

    fn ahead(h: *Harness, f: TempFile, cur: i64, recs: []const usize) i32 {
        _ = h;
        var bases: [max_items]i64 = undefined;
        for (recs, 0..) |r, i| bases[i] = @intCast(base(r));
        return q3ld_spec_step_len(f.fd, f.size(), cur, @intCast(recs.len), &bases, @intCast(rec_len));
    }

    fn waitCounter(h: *Harness, k: usize, want: i64) !void {
        var i: u32 = 0;
        const ts = std.c.timespec{ .sec = 0, .nsec = 2_000_000 };
        while (@atomicLoad(i64, &h.counters[k], .acquire) < want) : (i += 1) {
            if (i > 10_000) return error.PoolTimeout;
            _ = std.c.nanosleep(&ts, null);
        }
    }
};

fn sleepMs(n: i64) void {
    const ts = std.c.timespec{ .sec = 0, .nsec = @intCast(n * ms) };
    _ = std.c.nanosleep(&ts, null);
}

test "demand_first: the setting's modes, the pool's ABI and its counters' count" {
    try testing.expectEqual(@as(i32, 2026110203), q3ld_abi());
    for ([_]i32{ -1, 3, 7 }) |m| try testing.expectEqual(@as(c_int, -1), q3ld_demand_first(m));
    var df: [df_n]i64 = @splat(0);
    try testing.expectEqual(@as(i32, df_n), q3ld_df_stats(&df, 0));
    try testing.expectEqual(@as(c_int, 0), q3ld_demand_first(0));
}

test "demand_first byte gate: a record claimed while its lookahead read is slowed; mode 2's demand workers read its untaken pieces" {
    const a = testing.allocator;
    const f = try TempFile.create(a, 0xdf1_c1a1);
    defer f.destroy(a);
    for ([_]i32{ 0, 1, 2 }) |mode| {
        hooksOff();
        defer hooksOff();
        q3ld_test_spec_delay(40 * ms);
        var h: Harness = .{};
        defer h.release();
        try h.start(mode);
        try testing.expectEqual(@as(i32, 1), h.ahead(f, 0, &.{1}));
        try h.waitCounter(sc_started, 1);
        try h.demand(f, &.{1});
        const df = try h.finish();
        const cnt = h.counters;
        std.debug.print("demand_first claimed in flight, mode {d}: claimed {d} adopted {d}; df1 {any}\n", .{ mode, cnt[sc_claimed], cnt[sc_adopt_ranges], df });
        try testing.expectEqual(@as(i64, 1), cnt[sc_claimed]);
        // the gate/up range's two halves and the down range, all three out of the record
        try testing.expectEqual(@as(i64, 3), cnt[sc_adopt_ranges]);
        if (mode == 2) try testing.expect(df[df_help_runs] >= 1) else try testing.expectEqual(@as(i64, 0), df[df_help_runs]);
        if (mode == 0) try testing.expectEqual(@as(i64, 0), df[df_pieces]);
    }
}

test "demand_first byte gate: a demand read beside an unclaimed lookahead record; modes 1 and 2 start no unclaimed piece while demand runs" {
    const a = testing.allocator;
    const f = try TempFile.create(a, 0xdf1_5e5e);
    defer f.destroy(a);
    const psize: u64 = std.heap.pageSize();
    for ([_]i32{ 0, 1, 2 }) |mode| {
        hooksOff();
        defer hooksOff();
        q3ld_test_spec_delay(30 * ms);
        // the demand's first gate/up half (record 3's page-aligned start) held 200 ms
        const off = [1]i64{@intCast(base(3) / psize * psize)};
        const code = [1]i64{5};
        const arg = [1]i64{200 * ms};
        q3ld_test_rules(1, &off, &code, &arg);
        var h: Harness = .{};
        defer h.release();
        try h.start(mode);
        try testing.expectEqual(@as(i32, 1), h.ahead(f, 0, &.{0}));
        try h.waitCounter(sc_started, 1);
        try h.demand(f, &.{3});
        try h.waitCounter(sc_landed, 1);
        try h.demand(f, &.{0});
        const df = try h.finish();
        const cnt = h.counters;
        std.debug.print("demand_first beside, mode {d}: max busy at an unclaimed start {d}; df1 {any}\n", .{ mode, cnt[sc_max_busy_at_start], df });
        try testing.expectEqual(@as(i64, 0), cnt[sc_max_qlen_at_start]);
        if (mode != 0) {
            try testing.expectEqual(@as(i64, 0), cnt[sc_max_busy_at_start]);
            try testing.expect(df[df_stops] >= 1);
            // a 9,000,000 B record off the page: ten 1 MiB pieces at most, nine at least
            try testing.expect(df[df_pieces] >= 9);
        } else {
            for (df) |x| try testing.expectEqual(@as(i64, 0), x);
        }
    }
}

test "demand_first byte gate: a lookahead piece that fails once: the record fails, its ranges read the stock way" {
    const a = testing.allocator;
    const f = try TempFile.create(a, 0xdf1_e10);
    defer f.destroy(a);
    for ([_]i32{ 0, 1, 2 }) |mode| {
        hooksOff();
        defer hooksOff();
        q3ld_test_spec_delay(40 * ms);
        // record 0's second piece (1 MiB in: the spec thread's next read in mode 1, a demand worker's in mode 2) fails once
        const off = [1]i64{1 << 20};
        const code = [1]i64{2};
        const arg = [1]i64{0};
        q3ld_test_rules(1, &off, &code, &arg);
        var h: Harness = .{};
        defer h.release();
        try h.start(mode);
        try testing.expectEqual(@as(i32, 1), h.ahead(f, 0, &.{0}));
        try h.waitCounter(sc_started, 1);
        try h.demand(f, &.{0});
        const df = try h.finish();
        const cnt = h.counters;
        std.debug.print("demand_first failed piece, mode {d}: failed {d}; df1 {any}\n", .{ mode, cnt[sc_failed], df });
        if (mode != 0) try testing.expectEqual(@as(i64, 1), cnt[sc_failed]);
        if (mode == 2) try testing.expectEqual(@as(i64, 1), df[df_failed]);
    }
}

test "demand_first byte gate: rounds of lookahead steps and demand submits with random delays, every row the file's" {
    const a = testing.allocator;
    const f = try TempFile.create(a, 0xdf1_0b0b);
    defer f.destroy(a);
    for ([_]i32{ 0, 1, 2 }) |mode| {
        hooksOff();
        defer hooksOff();
        const seed: u64 = 0x5eed_0df1 + @as(u64, @intCast(mode));
        q3ld_test_delay(seed, 3 * ms);
        q3ld_test_spec_delay(6 * ms);
        var h: Harness = .{};
        defer h.release();
        try h.start(mode);
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        for (0..24) |r| {
            const a0 = rnd.uintLessThan(usize, n_rec);
            const a1 = (a0 + 1 + rnd.uintLessThan(usize, n_rec - 1)) % n_rec;
            _ = h.ahead(f, @intCast(r), &.{ a0, a1 });
            sleepMs(@intCast(rnd.uintLessThan(u32, 7)));
            var recs: [max_items]usize = undefined;
            const k = 1 + rnd.uintLessThan(usize, max_items);
            var n: usize = 0;
            if (rnd.boolean()) {
                recs[0] = a0;
                n = 1;
            }
            while (n < k) {
                const x = rnd.uintLessThan(usize, n_rec);
                if (std.mem.indexOfScalar(usize, recs[0..n], x) == null) {
                    recs[n] = x;
                    n += 1;
                }
            }
            try h.demand(f, recs[0..n]);
        }
        const df = try h.finish();
        const cnt = h.counters;
        std.debug.print("demand_first rounds, mode {d}: claimed {d} adopted {d} failed {d}; df1 {any}\n", .{ mode, cnt[sc_claimed], cnt[sc_adopt_ranges], cnt[sc_failed], df });
        try testing.expectEqual(@as(i64, 0), cnt[sc_failed]);
        try testing.expectEqual(@as(i64, 0), df[df_failed]);
    }
}
