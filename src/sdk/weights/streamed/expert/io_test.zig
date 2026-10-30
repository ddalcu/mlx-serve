//! The reader's tests (`sdk_ext.expert.io`), kept in the package's test root under their names: the C pool and its
//! fault injection compile into the test binary, never into the SDK's own tests.

const std = @import("std");
const expert_bank = @import("bank.zig");
const io = @import("../sdk_ext.zig").expert.io;
const c = io.test_abi.abi;
const Pool = io.Pool;
const Status = io.Status;
const Spec = io.Spec;
const max_items = io.max_items;
const max_spec = io.max_spec;
const max_pre = io.max_pre;
const max_gate_tickets = io.max_gate_tickets;
const res_w = io.test_abi.status_words;
const spec_state_w = io.spec_state_w;
const pre_state_w = io.pre_state_w;
const abi_version = io.abi_version;
const slotBytes = io.slotBytes;
const chunkBytes = io.chunkBytes;
const injectFault = io.injectFault;
const injectFaults = io.injectFaults;
const clearFaults = io.clearFaults;
const n_components = expert_bank.n_components;
const gu_components = expert_bank.gu_components;
/// EXL3's record topology (the bank of record's).
const R96 = io.Records(n_components, gu_components);

const testing = std.testing;

/// A temp file whose byte at offset o is a function of o, plus its image.
const PatternFile = struct {
    tmp: std.testing.TmpDir,
    image: []u8,
    ufd: io.UncachedFd,

    fn init(len: usize) !PatternFile {
        return initSeeded(len, 11);
    }

    fn initSeeded(len: usize, seed: u64) !PatternFile {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const image = try testing.allocator.alloc(u8, len);
        errdefer testing.allocator.free(image);
        expert_bank.fillPattern(image, seed);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "pattern.bin", .data = image });
        var root: [512]u8 = undefined;
        var pbuf: [600]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/pattern.bin", .{root[0..try tmp.dir.realPath(std.testing.io, &root)]}, 0);
        const ufd = try io.openUncached(path.ptr, null);
        return .{ .tmp = tmp, .image = image, .ufd = ufd };
    }

    fn deinit(self: *PatternFile) void {
        self.ufd.close();
        testing.allocator.free(self.image);
        self.tmp.cleanup();
    }
};

/// Nine destination rows per record, `lens[c]` bytes each, in one buffer: the test allocator's (`init`, the caller
/// frees `buf`), or whole pages with the rows from byte `skew` on (`initPages`, freed by `deinit`).
const Dests = struct {
    buf: []u8,
    rows: [max_items][n_components]u64,
    at: [max_items][n_components]usize,
    pages: []align(std.heap.page_size_min) u8 = &.{},

    fn init(n: usize, lens: *const [n_components]u64) !Dests {
        return layout(try testing.allocator.alloc(u8, totalOf(lens) * n), n, lens, 0);
    }

    fn initPages(n: usize, lens: *const [n_components]u64, skew: usize) !Dests {
        const pages = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), totalOf(lens) * n + std.heap.pageSize());
        var d = layout(pages, n, lens, skew);
        d.pages = pages;
        return d;
    }

    fn deinit(self: *Dests) void {
        std.heap.page_allocator.free(self.pages);
    }

    fn totalOf(lens: *const [n_components]u64) usize {
        var total: usize = 0;
        for (lens.*) |l| total += l;
        return total;
    }

    fn layout(buf: []u8, n: usize, lens: *const [n_components]u64, skew: usize) Dests {
        var d: Dests = .{ .buf = buf, .rows = undefined, .at = undefined };
        @memset(d.buf, 0xAA);
        var off: usize = skew;
        for (0..n) |i| for (lens.*, 0..) |l, k| {
            d.at[i][k] = off;
            d.rows[i][k] = @intFromPtr(d.buf.ptr) + off;
            off += l;
        };
        return d;
    }

    fn part(self: *const Dests, i: usize, k: usize, lens: *const [n_components]u64) []const u8 {
        return self.buf[self.at[i][k]..][0..lens[k]];
    }

    /// Record i's nine rows hold the file's bytes at `gu` (six parts) and `down` (three).
    fn expectRecord(self: *const Dests, i: usize, image: []const u8, gu: u64, down: u64, lens: *const [n_components]u64) !void {
        var at = gu;
        for (0..n_components) |k| {
            if (k == gu_components) at = down;
            try testing.expectEqualSlices(u8, image[at..][0..lens[k]], self.part(i, k, lens));
            at += lens[k];
        }
    }
};

test "dsv41 io: pool scatters a synthetic file" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page + 321);
    defer f.deinit();
    // One-page staging: every range needs several aligned reads.
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    // Unequal, page-unaligned parts; gate/up parts contiguous, down parts contiguous.
    const lens = [n_components]u64{ page + 3, 17, 4099, 1, 777, 2 * page + 5, 1000, page - 1, 33 };
    var gu_len: u64 = 0;
    for (lens[0..gu_components]) |l| gu_len += l;
    for ([_]usize{ 1, 3, 8 }, 0..) |n, round| {
        var d = try Dests.init(n, &lens);
        defer testing.allocator.free(d.buf);
        var gu: [max_items]u64 = undefined;
        var down: [max_items]u64 = undefined;
        for (0..n) |i| {
            gu[i] = 7 + round * 13 + i * (5 * page + 11);
            down[i] = 40 * page + 3 + i * (2 * page + 7);
        }
        const seq0 = c.q3ld_seq();
        const first = try R96.submit(pool, f.ufd, gu[0..n], down[0..n], d.rows[0..n], &lens);
        const count: u32 = @intCast(2 * n);
        try pool.wait(first, count, 10 * std.time.ns_per_s);
        for (0..n) |i| {
            try d.expectRecord(i, f.image, gu[i], down[i], &lens);
            const r_gu = pool.result(first + @as(u32, @intCast(i)));
            try testing.expectEqual(Status.ok, r_gu.status);
            try testing.expectEqual(@as(i64, @intCast(gu_len)), r_gu.payload);
            try testing.expect(r_gu.preadv_calls >= @as(i64, @intCast(gu_len / page)));
            try testing.expectEqual(Status.ok, pool.result(first + @as(u32, @intCast(n + i))).status);
        }
        // A job per range: every ticket lands once (two workers may land them out of order).
        var order: [2 * max_items]u32 = undefined;
        const got = pool.logOrder(seq0, c.q3ld_seq(), &order);
        try testing.expectEqual(@as(usize, count), got.len);
        std.mem.sort(u32, got, {}, std.sort.asc(u32));
        for (got, 0..) |t, k| try testing.expectEqual(first + @as(u32, @intCast(k)), t);
    }
}

test "io scatter: each part of a record lands from its own file and offset; a part past its file's end fails its range" {
    const page = std.heap.pageSize();
    var f0 = try PatternFile.initSeeded(16 * page + 77, 3);
    defer f0.deinit();
    var f1 = try PatternFile.initSeeded(9 * page + 5, 4);
    defer f1.deinit();
    const files = [2]*const PatternFile{ &f0, &f1 };
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    const lens = [n_components]u64{ page + 3, 17, 4099, 1, 777, 2 * page + 5, 1000, page - 1, 33 };
    const n = 3;
    var d = try Dests.init(n, &lens);
    defer testing.allocator.free(d.buf);
    var parts: [n][n_components]R96.Part = undefined;
    for (&parts, 0..) |*rec, i| for (rec, 0..) |*p, k| {
        const f = files[(i + k) % 2];
        p.* = .{ .fd = f.ufd, .offset = (i * 977 + k * 1301) % (f.image.len - lens[k]) };
    };
    const first = try R96.submitScatter(pool, &parts, d.rows[0..n], &lens);
    try pool.wait(first, 2 * n, 10 * std.time.ns_per_s);
    for (0..n) |i| {
        for (0..n_components) |k| {
            const f = files[(i + k) % 2];
            try testing.expectEqualSlices(u8, f.image[parts[i][k].offset..][0..lens[k]], d.part(i, k, &lens));
        }
        try testing.expectEqual(Status.ok, pool.result(first + @as(u32, @intCast(i))).status);
        try testing.expectEqual(Status.ok, pool.result(first + @as(u32, @intCast(n + i))).status);
    }
    // A down part one byte past the end of its file: that record's down range is short, the rest land.
    parts[1][n_components - 1] = .{ .fd = f1.ufd, .offset = f1.image.len - lens[n_components - 1] + 1 };
    const again = try R96.submitScatter(pool, &parts, d.rows[0..n], &lens);
    try pool.wait(again, 2 * n, 10 * std.time.ns_per_s);
    try testing.expectEqual(Status.short, pool.result(again + n + 1).status);
    try testing.expectEqual(Status.ok, pool.result(again + 1).status);
    try testing.expectEqual(Status.ok, pool.result(again + n + 2).status);
}

test "dsv41 io: injected faults map to status words" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(40 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    defer clearFaults();
    const lens = [n_components]u64{ 100, 100, 100, 100, 100, 100, 50, 50, 50 };
    // Every range starts 500 bytes into its own page, so its first read skips 500.
    const Case = struct { code: i64, arg: i64, hit: enum { gu1, down0 }, want: [4]Status, err: i64 = 0, hit_calls: i64 = 1 };
    const cases = [_]Case{
        // 1 EINTR: retried inside the call, uncounted.
        .{ .code = 1, .arg = 0, .hit = .gu1, .want = .{ .ok, .ok, .ok, .ok } },
        // 2 EIO: that range fails with errno; every other range is its own job and lands.
        .{ .code = 2, .arg = 0, .hit = .gu1, .want = .{ .ok, .os_error, .ok, .ok }, .err = @intFromEnum(std.posix.E.IO) },
        // 3 zero return: a short read.
        .{ .code = 3, .arg = 0, .hit = .down0, .want = .{ .ok, .ok, .short, .ok } },
        // 4 truncated past the skip: 200 bytes land, the range continues.
        .{ .code = 4, .arg = 700, .hit = .gu1, .want = .{ .ok, .ok, .ok, .ok }, .hit_calls = 2 },
        // 4 truncated inside the skip: no byte past it, the range ends short (never a retry of the same request).
        .{ .code = 4, .arg = 300, .hit = .gu1, .want = .{ .ok, .short, .ok, .ok } },
    };
    for (cases, 0..) |cs, round| {
        // Each range on its own page, 500 bytes in, so a rule's aligned offset names one range.
        const base = round * 8 * page;
        const gu = [2]u64{ base + 500, base + 2 * page + 500 };
        const down = [2]u64{ base + 4 * page + 500, base + 6 * page + 500 };
        const hit_off: u64 = switch (cs.hit) {
            .gu1 => base + 2 * page,
            .down0 => base + 4 * page,
        };
        injectFault(hit_off, cs.code, cs.arg);
        var d = try Dests.init(2, &lens);
        defer testing.allocator.free(d.buf);
        const first = try R96.submit(pool, f.ufd, &gu, &down, d.rows[0..2], &lens);
        try pool.wait(first, 4, 10 * std.time.ns_per_s);
        for (cs.want, 0..) |w, k| {
            const r = pool.result(first + @as(u32, @intCast(k)));
            testing.expectEqual(w, r.status) catch |e| {
                std.debug.print("case {d} range {d}\n", .{ round, k });
                return e;
            };
            if (w == .os_error) try testing.expectEqual(cs.err, r.errno);
            if (w == .ok) try testing.expectEqual(@as(i64, if (k < 2) 600 else 150), r.payload);
            const hit_k: usize = if (cs.hit == .gu1) 1 else 2;
            if (k == hit_k and w == .ok) try testing.expectEqual(cs.hit_calls, r.preadv_calls);
        }
        if (cs.want[1] == .ok) try testing.expectEqualSlices(u8, f.image[gu[1]..][0..100], d.part(1, 0, &lens));
    }
}

test "glm io: an id's 4 minis (one submit, GLM-5.3's shape: 4 gate/up and 4 down ranges) read on several workers, each range's status its own" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(40 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 8, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    defer clearFaults();
    const minis = 4;
    const lens = [n_components]u64{ 100, 100, 100, 100, 100, 100, 50, 50, 50 };
    var gu: [minis]u64 = undefined;
    var down: [minis]u64 = undefined;
    for (0..minis) |m| {
        gu[m] = (2 * m) * page + 500;
        down[m] = (2 * m + 1) * page + 500;
    }
    // Mini 0's gate/up sleeps 200 ms first: the other seven ranges land on other workers before it; mini 1's gate/up
    // fails alone.
    injectFaults(&.{ 0, 2 * @as(i64, @intCast(page)) }, &.{ 5, 2 }, &.{ 200 * std.time.ns_per_ms, 0 });
    var d = try Dests.init(minis, &lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &gu, &down, d.rows[0..minis], &lens);
    try pool.wait(first, 2 * minis, 10 * std.time.ns_per_s);
    const r0 = pool.result(first);
    try testing.expectEqual(Status.ok, r0.status);
    for (1..2 * minis) |k| {
        const r = pool.result(first + @as(u32, @intCast(k)));
        try testing.expectEqual(if (k == 1) Status.os_error else Status.ok, r.status);
        try testing.expect(r.worker != r0.worker);
        try testing.expect(r.t_end_ns < r0.t_end_ns);
    }
    for (0..minis) |m| if (m != 1) try d.expectRecord(m, f.image, gu[m], down[m], &lens);
    // Mini 1's gate/up failed; its down range landed.
    var at = down[1];
    for (gu_components..n_components) |k| {
        try testing.expectEqualSlices(u8, f.image[at..][0..lens[k]], d.part(1, k, &lens));
        at += lens[k];
    }
}

test "dsv41 io: stop joins and restarts" {
    try testing.expectEqual(@as(i32, abi_version), c.q3ld_abi());
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = std.heap.pageSize(), .tickets = 16 });
    try testing.expectError(error.PoolUnavailable, Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = std.heap.pageSize(), .tickets = 16 }));
    pool.stop();
    pool = try Pool.start(testing.allocator, .{ .workers = 4, .staging_bytes = 9 << 20, .tickets = 16 });
    pool.stop();
    try testing.expectError(error.InvalidOptions, Pool.start(testing.allocator, .{ .staging_bytes = std.heap.pageSize() + 1 }));
    const bad_spec: Spec = .{ .threads = 1, .slots = max_spec + 1, .record_bytes = 100, .chunk_bytes = std.heap.pageSize() };
    try testing.expectError(error.InvalidOptions, Pool.start(testing.allocator, .{ .spec = bad_spec }));
}

test "dsv41 io: speculative slots and chunks follow the lane's sizes" {
    // q3_lookahead4_candidate.slot_bytes / chunk_bytes on the 3.0 bank's 13,315,584-byte record, 16 KiB pages.
    try testing.expectEqual(@as(u64, 13_352_960), slotBytes(13_315_584, 16384));
    try testing.expectEqual(@as(u64, 3_342_336), chunkBytes(4, 13_315_584, 16384));
    try testing.expectEqual(@as(u64, 13_352_960), chunkBytes(1, 13_315_584, 16384));
    try testing.expectEqual(@as(u64, 49_152), slotBytes(2880, 16384));
    try testing.expectEqual(@as(u64, 16_384), chunkBytes(4, 2880, 16384));
}

// Speculative class. One record = a gate/up range then a down range, back to back.
const spec_lens = [n_components]u64{ 3000, 100, 200, 3000, 100, 200, 3000, 200, 100 };
const spec_gu_len = 6600;
const spec_rec_len = 9900;

fn specPool(workers: u32, slots: u32) !*Pool {
    const page = std.heap.pageSize();
    return Pool.start(testing.allocator, .{ .workers = workers, .staging_bytes = 4 * page, .tickets = 128, .spec = .{
        .threads = 1,
        .slots = slots,
        .record_bytes = spec_rec_len,
        .chunk_bytes = page,
    } });
}

const SpecSlot = struct { state: i64, tag: i64, base: i64, landed: i64, claimed: i64 };

fn specSlots(out: *[max_spec]SpecSlot) []SpecSlot {
    var raw: [spec_state_w * max_spec]i64 = undefined;
    const n: usize = @intCast(c.q3ld_spec_state(&raw));
    for (out[0..n], 0..) |*s, i| {
        const w = raw[i * spec_state_w ..][0..spec_state_w];
        s.* = .{ .state = w[0], .tag = w[1], .base = w[2], .landed = w[3], .claimed = w[6] };
    }
    return out[0..n];
}

/// Polls until `pred` holds (10 s).
fn waitFor(ctx: anytype, comptime pred: fn (@TypeOf(ctx)) bool) !void {
    var t: u32 = 0;
    while (!pred(ctx)) : (t += 1) {
        if (t > 10_000) return error.Timeout;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
}

fn landedAt(base: i64) bool {
    var slots: [max_spec]SpecSlot = undefined;
    for (specSlots(&slots)) |s| if (s.base == base and s.state == 3) return true;
    return false;
}

test "dsv41 io: a landed speculative record serves the demand read with no preadv" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(32 * page);
    defer f.deinit();
    var pool = try specPool(2, 2);
    defer pool.stop();
    const base: u64 = 3 * page + 100;
    try testing.expectEqual(@as(u32, 1), try pool.specStep(f.ufd, 1, &.{@intCast(base)}, spec_rec_len));
    // A second issue of a live record only refreshes it.
    try testing.expectEqual(@as(u32, 0), try pool.specStep(f.ufd, 1, &.{@intCast(base)}, spec_rec_len));
    try waitFor(@as(i64, @intCast(base)), landedAt);
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &.{base}, &.{base + spec_gu_len}, d.rows[0..1], &spec_lens);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try d.expectRecord(0, f.image, base, base + spec_gu_len, &spec_lens);
    for (0..2) |k| {
        const r = pool.result(first + @as(u32, @intCast(k)));
        try testing.expectEqual(Status.ok, r.status);
        try testing.expectEqual(@as(i64, 0), r.preadv_calls);
    }
    try testing.expectEqual(@as(i64, 1), pool.counter(.claimed));
    try testing.expectEqual(@as(i64, 2), pool.counter(.adopt_ranges));
    try testing.expectEqual(@as(i64, spec_rec_len), pool.counter(.adopt_bytes));
    try testing.expectEqual(@as(i64, 1), pool.counter(.refreshed));
    try testing.expect(pool.counter(.spec_bytes) >= spec_rec_len);
}

test "dsv41 io: a queued speculative record is cancelled by the demand read; settle drops the unclaimed" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(2, 4);
    defer pool.stop();
    defer clearFaults();
    // Two slow demand jobs keep both workers busy, so no unclaimed record may start.
    const slow = [2]u64{ 20 * page, 30 * page };
    injectFaults(&.{ @intCast(slow[0]), @intCast(slow[1]) }, &.{ 5, 5 }, &.{ 300 * std.time.ns_per_ms, 300 * std.time.ns_per_ms });
    var busy = try Dests.init(2, &spec_lens);
    defer testing.allocator.free(busy.buf);
    const j0 = try R96.submit(pool, f.ufd, slow[0..1], &.{slow[0] + spec_gu_len}, busy.rows[0..1], &spec_lens);
    const j1 = try R96.submit(pool, f.ufd, slow[1..2], &.{slow[1] + spec_gu_len}, busy.rows[1..2], &spec_lens);
    const a: u64 = 2 * page + 7;
    const b: u64 = 6 * page + 9;
    try testing.expectEqual(@as(u32, 2), try pool.specStep(f.ufd, 1, &.{ @intCast(a), @intCast(b) }, spec_rec_len));
    // The demand read of `a` cancels its queued record and reads it itself.
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &.{a}, &.{a + spec_gu_len}, d.rows[0..1], &spec_lens);
    try testing.expectEqual(@as(i64, 1), pool.counter(.cancelled_by_demand));
    // The next layer call's settle drops `b`, never claimed.
    _ = try pool.specStep(f.ufd, 2, &.{}, spec_rec_len);
    try testing.expectEqual(@as(i64, 1), pool.counter(.expired));
    var slots: [max_spec]SpecSlot = undefined;
    for (specSlots(&slots)) |s| try testing.expectEqual(@as(i64, 0), s.state);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try pool.wait(j0, 2, 10 * std.time.ns_per_s);
    try pool.wait(j1, 2, 10 * std.time.ns_per_s);
    try d.expectRecord(0, f.image, a, a + spec_gu_len, &spec_lens);
    try testing.expect(pool.result(first).preadv_calls >= 1);
    try testing.expectEqual(@as(i64, 0), pool.counter(.adopt_ranges));
    try testing.expectEqual(@as(i64, 0), pool.counter(.started));
}

fn preCount(p: *Pool) bool {
    return p.counter(.pre_started) >= 4;
}

fn preTwo(p: *Pool) bool {
    return p.counter(.pre_started) >= 2;
}

fn preIdle(p: *Pool) bool {
    _ = p;
    var raw: [pre_state_w * max_pre]i64 = undefined;
    _ = c.q3ld_pre_state(&raw);
    for (0..max_pre) |i| if (raw[i * pre_state_w] != 0) return false;
    return true;
}

test "dsv41 io: pre-read ranges bind to the demand submit; unbound ones expire at the settle" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(4, 2);
    defer pool.stop();
    try R96.armPreRead(pool, &spec_lens);
    // Bound: two records pre-read (4 ranges, each worker holds one at its first copy), then demanded.
    const recs = [2]u64{ 5 * page + 11, 9 * page + 13 };
    try testing.expectEqual(@as(u32, 4), try R96.preRead(pool, f.ufd, 1, &.{ @intCast(recs[0]), @intCast(recs[1]) }, &spec_lens));
    try waitFor(pool, preCount);
    var d = try Dests.init(2, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &recs, &.{ recs[0] + spec_gu_len, recs[1] + spec_gu_len }, d.rows[0..2], &spec_lens);
    try pool.wait(first, 4, 10 * std.time.ns_per_s);
    for (0..2) |i| try d.expectRecord(i, f.image, recs[i], recs[i] + spec_gu_len, &spec_lens);
    for (0..4) |k| try testing.expect(pool.result(first + @as(u32, @intCast(k))).preadv_calls >= 1);
    try testing.expectEqual(@as(i64, 4), pool.counter(.pre_bound));
    try testing.expectEqual(@as(i64, 4), pool.counter(.pre_served));
    try testing.expectEqual(@as(i64, 0), pool.counter(.pre_cancelled));
    _ = try pool.specStep(f.ufd, 1, &.{}, spec_rec_len);
    // Unbound: pre-read, never demanded; the call's settle expires both ranges and frees their workers.
    const lone: u64 = 20 * page + 5;
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 2, &.{@intCast(lone)}, &spec_lens));
    try waitFor(pool, preCount6);
    _ = try pool.specStep(f.ufd, 2, &.{}, spec_rec_len);
    try waitFor(pool, preIdle);
    try testing.expectEqual(@as(i64, 2), pool.counter(.pre_expired));
    try testing.expectEqual(@as(i64, 4), pool.counter(.pre_served));
}

fn preCount6(p: *Pool) bool {
    return p.counter(.pre_started) >= 6;
}

// Event gates on a host word. Gate/up spans cross a page, so a down range's
// first preadv never shares an aligned offset with its record's gate/up read.
const ev_lens = [n_components]u64{ 8000, 100, 200, 8000, 100, 200, 3000, 200, 100 };
const ev_gu_len = 16600;

const EvRig = struct {
    f: PatternFile,
    pool: *Pool,
    word: *i64,
    dests: std.ArrayList(Dests) = .empty,
    value: u64 = 0,

    fn init(timeout_ms: i64, workers: u32) !EvRig {
        const page = std.heap.pageSize();
        var f = try PatternFile.init(96 * page);
        errdefer f.deinit();
        const word = try testing.allocator.create(i64);
        errdefer testing.allocator.destroy(word);
        word.* = 0;
        const pool = try Pool.start(testing.allocator, .{ .workers = workers, .staging_bytes = 4 * page, .tickets = 512 });
        errdefer pool.stop();
        try pool.armEvent(.host, @intFromPtr(word), timeout_ms * std.time.ns_per_ms, 0);
        return .{ .f = f, .pool = pool, .word = word };
    }

    fn deinit(self: *EvRig) void {
        // The pool writes the word until it stops.
        self.pool.stop();
        for (self.dests.items) |d| testing.allocator.free(d.buf);
        self.dests.deinit(testing.allocator);
        testing.allocator.destroy(self.word);
        self.f.deinit();
    }

    fn base(r: u64) u64 {
        return 2 * std.heap.pageSize() + r * 3 * std.heap.pageSize() + 17 * r;
    }

    /// One job of records `recs`; returns its first ticket.
    fn job(self: *EvRig, recs: []const u64) !u32 {
        var d = try Dests.init(recs.len, &ev_lens);
        errdefer testing.allocator.free(d.buf);
        var gu: [max_items]u64 = undefined;
        var down: [max_items]u64 = undefined;
        for (recs, 0..) |r, i| {
            gu[i] = base(r);
            down[i] = base(r) + ev_gu_len;
        }
        const first = try R96.submit(self.pool, self.f.ufd, gu[0..recs.len], down[0..recs.len], d.rows[0..recs.len], &ev_lens);
        try self.dests.append(testing.allocator, d);
        return first;
    }

    /// Gates over ticket groups, values continuing from the last one.
    fn register(self: *EvRig, groups: []const []const i64) !void {
        var values: [16]u64 = undefined;
        var counts: [16]i32 = undefined;
        var tickets: [io.max_gate_tickets]i64 = undefined;
        var k: usize = 0;
        for (groups, 0..) |g, i| {
            values[i] = self.value + 1 + i;
            counts[i] = @intCast(g.len);
            @memcpy(tickets[k..][0..g.len], g);
            k += g.len;
        }
        try self.pool.registerGates(values[0..groups.len], counts[0..groups.len], tickets[0..k]);
        self.value += groups.len;
    }

    fn waitWord(self: *EvRig, value: u64, timeout_ms: u32) !void {
        var t: u32 = 0;
        while (@as(u64, @intCast(@atomicLoad(i64, self.word, .acquire))) < value) : (t += 1) {
            if (t > timeout_ms * 2) return error.Timeout;
            std.Io.sleep(std.testing.io, .fromMicroseconds(500), .awake) catch {};
        }
    }
};

test "dsv41 io: event gates hand the satisfied prefix to the event in order" {
    var rig = try EvRig.init(10_000, 2);
    defer rig.deinit();
    var log: [4 * 64]i64 = undefined;
    _ = c.q3ld_test_ev_log(&log, 64);
    defer _ = c.q3ld_test_ev_log(null, -1);
    c.q3ld_test_delay(11, 2 * std.time.ns_per_ms);
    defer c.q3ld_test_delay(0, 0);
    // Three jobs; gate 1 = every gate/up ticket, then one gate per job's down tickets.
    var firsts: [3]u32 = undefined;
    for (&firsts, 0..) |*fst, j| {
        const r: u64 = 2 * j;
        fst.* = try rig.job(&.{ r, r + 1 });
    }
    var gu: [6]i64 = undefined;
    var downs: [3][2]i64 = undefined;
    for (firsts, 0..) |fst, j| {
        gu[2 * j] = fst;
        gu[2 * j + 1] = fst + 1;
        downs[j] = .{ fst + 2, fst + 3 };
    }
    const groups = [_][]const i64{ &gu, &downs[0], &downs[1], &downs[2] };
    try rig.register(&groups);
    try rig.waitWord(4, 10_000);
    for (firsts) |fst| try rig.pool.wait(fst, 4, 10 * std.time.ns_per_s);
    // Every signal value is new and higher, and every gate at or below it had all its tickets published before the call.
    const n: usize = @intCast(c.q3ld_test_ev_log(null, 0));
    try testing.expect(n >= 1);
    var prev: i64 = 0;
    for (0..n) |i| {
        const e = log[4 * i ..][0..4];
        try testing.expect(e[0] > prev);
        prev = e[0];
        for (groups, 1..) |g, v| {
            if (@as(i64, @intCast(v)) > e[0]) break;
            for (g) |t| {
                const r = rig.pool.result(@intCast(t));
                try testing.expect(r.status != .pending and r.t_end_ns <= e[1]);
            }
        }
    }
    try testing.expectEqual(@as(i64, 4), prev);
    for (rig.dests.items, 0..) |d, j| for (0..2) |i| {
        const r: u64 = 2 * j + i;
        try d.expectRecord(i, rig.f.image, EvRig.base(r), EvRig.base(r) + ev_gu_len, &ev_lens);
    };
    try testing.expectEqual(@as(i64, 4), rig.pool.counter(.ev_gates));
    try testing.expectEqual(@as(i64, 0), rig.pool.counter(.ev_wd_forced));
    // Gates whose tickets already landed are handed over inside the registration.
    const late = try rig.job(&.{7});
    try rig.pool.wait(late, 2, 10 * std.time.ns_per_s);
    try rig.register(&.{ &.{late}, &.{late + 1} });
    try testing.expectEqual(@as(i64, 6), @atomicLoad(i64, rig.word, .acquire));
    try testing.expectEqual(@as(i64, 2), rig.pool.counter(.ev_immediate));
}

test "dsv41 io: a failed read is terminal for its gate" {
    var rig = try EvRig.init(10_000, 2);
    defer rig.deinit();
    defer clearFaults();
    const page = std.heap.pageSize();
    injectFault(EvRig.base(4) / page * page, 2, 0);
    const f0 = try rig.job(&.{ 4, 5 });
    const f1 = try rig.job(&.{6});
    try rig.register(&.{ &.{ f0, f0 + 1, f1 }, &.{ f0 + 2, f0 + 3 }, &.{f1 + 1} });
    try rig.waitWord(3, 10_000);
    try rig.pool.wait(f0, 4, 10 * std.time.ns_per_s);
    try rig.pool.wait(f1, 2, 10 * std.time.ns_per_s);
    const want = [4]Status{ .os_error, .ok, .ok, .ok };
    for (want, 0..) |w, k| try testing.expectEqual(w, rig.pool.result(f0 + @as(u32, @intCast(k))).status);
    try testing.expectEqual(@as(i64, 0), rig.pool.counter(.ev_wd_forced));
}

test "dsv41 io: the watchdog forces a gate whose bytes never land; stop and the host release the rest" {
    var rig = try EvRig.init(100, 2);
    defer rig.deinit();
    defer clearFaults();
    const page = std.heap.pageSize();
    // Record 8's down range is held 600 ms: its gate is forced after ~100 ms, record 9's still waits for its bytes.
    injectFault((EvRig.base(8) + ev_gu_len) / page * page, 5, 600 * std.time.ns_per_ms);
    const f0 = try rig.job(&.{8});
    const f1 = try rig.job(&.{9});
    const t_reg = c.q3ld_monotonic_ns();
    try rig.register(&.{ &.{ f0, f1 }, &.{f0 + 1}, &.{f1 + 1} });
    try rig.waitWord(2, 5_000);
    const forced_ms = @divTrunc(c.q3ld_monotonic_ns() - t_reg, std.time.ns_per_ms);
    try testing.expectEqual(Status.pending, rig.pool.result(f0 + 1).status);
    try testing.expect(forced_ms >= 90 and forced_ms < 450);
    try testing.expectEqual(@as(i64, 1), rig.pool.counter(.ev_wd_forced));
    try testing.expectEqual(@as(i64, 2), rig.pool.counter(.ev_wd_last_value));
    try rig.waitWord(3, 5_000);
    try rig.pool.wait(f0, 2, 10 * std.time.ns_per_s);
    // Host release: a gate over a ticket that never publishes is forced by value.
    const phantom: u32 = 500;
    rig.pool.res[phantom * res_w] = @intFromEnum(Status.pending);
    try rig.register(&.{&.{phantom}});
    rig.pool.releaseGates(4);
    try testing.expectEqual(@as(i64, 4), @atomicLoad(i64, rig.word, .acquire));
    try testing.expectEqual(@as(i64, 1), rig.pool.counter(.ev_host_released));
    // Stop: a live gate is released so no GPU wait outlives the pool.
    const phantom2: u32 = 501;
    rig.pool.res[phantom2 * res_w] = @intFromEnum(Status.pending);
    try rig.register(&.{&.{phantom2}});
    try testing.expectEqual(@as(i64, 4), @atomicLoad(i64, rig.word, .acquire));
    _ = c.q3ld_quiesce(10 * std.time.ns_per_s);
    _ = c.q3ld_stop();
    try testing.expectEqual(@as(i64, 5), @atomicLoad(i64, rig.word, .acquire));
    try testing.expectEqual(@as(i64, 1), rig.pool.counter(.ev_stop_released));
    rig.pool.res[phantom * res_w] = 0;
    rig.pool.res[phantom2 * res_w] = 0;
}

test "dsv41 io: event-gate refusals" {
    var rig = try EvRig.init(60_000, 1);
    defer rig.deinit();
    try testing.expectError(error.EventRefused, rig.pool.armEvent(.host, @intFromPtr(rig.word), std.time.ns_per_s, 0));
    const f = try rig.job(&.{11});
    try rig.pool.wait(f, 2, 10 * std.time.ns_per_s);
    try rig.register(&.{&.{f}});
    // Not above the last value; not increasing; out-of-range and repeated tickets; too many tickets.
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{1}, &.{0}, &.{}));
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{ 4, 3 }, &.{ 0, 0 }, &.{}));
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{2}, &.{1}, &.{512}));
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{2}, &.{1}, &.{-1}));
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{2}, &.{2}, &.{ f + 1, f + 1 }));
    var many: [max_gate_tickets + 1]i64 = undefined;
    for (&many, 0..) |*t, i| t.* = @intCast(i);
    try testing.expectError(error.GateInvalid, rig.pool.registerGates(&.{2}, &.{max_gate_tickets + 1}, &many));
    // Nothing was registered by a refused call.
    try testing.expectEqual(@as(i64, 1), rig.pool.counter(.ev_calls));
}

test "dsv41 io: the event class refuses a pool without it and bad arguments" {
    const page = std.heap.pageSize();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    try testing.expectError(error.GateRefused, pool.registerGates(&.{1}, &.{0}, &.{}));
    var word: i64 = 0;
    try testing.expectError(error.EventRefused, pool.armEvent(.host, @intFromPtr(&word), 999_999, 0));
    try testing.expectError(error.EventRefused, pool.armEvent(.host, 0, std.time.ns_per_s, 0));
    // The pre-read class needs the speculative class.
    try testing.expectError(error.PreReadRefused, R96.armPreRead(pool, &spec_lens));
}

/// The pool threads as the kernel sees them (tests): how many carry a q3ld-* name.
const ThreadProbe = struct {
    const mach_port_t = u32;
    extern "c" var mach_task_self_: mach_port_t;
    extern "c" fn task_threads(task: mach_port_t, list: *[*]mach_port_t, count: *u32) c_int;
    extern "c" fn pthread_from_mach_thread_np(port: mach_port_t) ?std.c.pthread_t;
    extern "c" fn pthread_getname_np(t: std.c.pthread_t, name: [*]u8, len: usize) c_int;

    fn named() u32 {
        var list: [*]mach_port_t = undefined;
        var n: u32 = 0;
        var count: u32 = 0;
        if (task_threads(mach_task_self_, &list, &n) != 0) return 0;
        for (list[0..n]) |port| {
            const t = pthread_from_mach_thread_np(port) orelse continue;
            var name: [64]u8 = @splat(0);
            _ = pthread_getname_np(t, &name, name.len);
            if (std.mem.startsWith(u8, std.mem.sliceTo(&name, 0), "q3ld-")) count += 1;
        }
        return count;
    }
};

test "dsv41 io: the pool's threads keep the creating thread's class and are left unnamed" {
    const page = std.heap.pageSize();
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 64, .spec = .{ .threads = 1, .slots = 1, .record_bytes = page, .chunk_bytes = page } });
    defer pool.stop();
    var word: i64 align(8) = 0;
    try pool.armEvent(.host, @intFromPtr(&word), std.time.ns_per_s, 0);
    try testing.expectEqual(@as(u32, 0), ThreadProbe.named());
}

// ── The reader's C ABI at its edges. Deterministic by construction: a worker is held at a pre-range's bind point (a
// pre-read nobody demands), a ticket is pending through its status word, a deadline of 0 is already past. ──

/// The C pool's entry points the Pool does not wrap: the stock and horizon-N steps, the stock pre-read, the snapshots.
const q3raw = struct {
    extern fn q3ld_spec_step(fd: i32, file_size: i64, cur: i64, n: i32, bases: ?[*]const i64) i32;
    extern fn q3ld_spec_stepn(fd: i32, file_size: i64, cur: i64, nh: i32, nval: [*]const i32, val: [*]const i64, nissue: [*]const i32, iss: [*]const i64) i32;
    extern fn q3ld_pre_read(fd: i32, file_size: i64, tag: i64, n: i32, bases: [*]const i64) i32;
    extern fn q3ld_ev_state(out: *[10]i64) i32;
    extern fn q3ld_test_events(buf: ?[*]i64, cap: i64) i64;
    extern fn q3ld_test_spec_delay(max_ns: i64) void;
    extern fn q3ld_spec_idle_busy() i32;
    extern fn q3ld_max_h() i32;
};

/// One record's job on the raw ABI at ticket `first`: gate/up parts `gl` at `gu`, down parts `dl` at `down`, back to
/// back into `dst`. The Pool's published marks of its two tickets are cleared first (Pool.wait reads them).
fn rawSubmit(pool: *Pool, fd: i32, size: i64, deadline: i64, gu: u64, down: u64, gl: []const i64, dl: []const i64, dst: []u8, first: u32) c_int {
    var lens: [12]i64 = undefined;
    var row: [12]u64 = undefined;
    var at: usize = 0;
    var k: usize = 0;
    for ([_][]const i64{ gl, dl }) |part| for (part) |l| {
        lens[k] = l;
        row[k] = @intFromPtr(dst.ptr) + at;
        at += @intCast(l);
        k += 1;
    };
    std.debug.assert(at <= dst.len);
    const offs = [2]i64{ @intCast(gu), @intCast(down) };
    const rows = [1][*]const u64{&row};
    @memset(pool.published[first..][0..2], false);
    return c.q3ld_submit(fd, size, deadline, 1, @intCast(gl.len), @intCast(dl.len), &offs, &rows, &lens, first);
}

fn counterAt(p: *const Pool, i: usize) i64 {
    return @atomicLoad(i64, &p.counters[i], .monotonic);
}

fn preStarted1(p: *Pool) bool {
    return p.counter(.pre_started) >= 1;
}

fn preStarted2(p: *Pool) bool {
    return p.counter(.pre_started) >= 2;
}

/// A started-and-stopped raw pool: the C state's configuration flag cleared, whatever an earlier test left.
fn resetPoolState() !void {
    const page = std.heap.pageSize();
    const mem = try std.heap.page_allocator.alloc(u8, page);
    defer std.heap.page_allocator.free(mem);
    var res_arr: [16 * res_w]i64 = @splat(0);
    var log_arr: [16]i64 = @splat(0);
    var gauge: [6]i64 = @splat(0);
    const staging = [1]u64{@intFromPtr(mem.ptr)};
    try testing.expectEqual(@as(c_int, 0), c.q3ld_spec_config(0, null, 0, 0, 0, 0, null));
    try testing.expectEqual(@as(c_int, 0), c.q3ld_start(1, &staging, @intCast(page), @intCast(page), &res_arr, 16, &log_arr, 16, &gauge));
    try testing.expectEqual(@as(c_int, 0), c.q3ld_stop());
}

test "dsv41 io cov: with no pool running every entry point refuses, and the snapshots read the stopped state" {
    const lens = [2]i64{ 10, 10 };
    const offs = [2]i64{ 0, 100 };
    const row = [2]u64{ 0, 0 };
    const rows = [1][*]const u64{&row};
    const bases = [1]i64{0};
    const one = [1]i32{1};
    const vals = [1]u64{1};
    const tix = [1]i64{0};
    var word: i64 = 0;
    try testing.expectEqual(@as(c_int, -1), c.q3ld_stop());
    try testing.expectEqual(@as(c_int, 0), c.q3ld_quiesce(std.time.ns_per_ms));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_submit(3, 1000, -1, 1, 1, 1, &offs, &rows, &lens, 0));
    try testing.expectEqual(@as(i32, -1), c.q3ld_spec_step_len(3, 1000, 0, 1, &bases, 10));
    try testing.expectEqual(@as(i32, -1), q3raw.q3ld_spec_step(3, 1000, 0, 1, &bases));
    for ([_]i32{ -1, 1, 5 }) |nh| try testing.expectEqual(@as(i32, -1), q3raw.q3ld_spec_stepn(3, 1000, 0, nh, &one, &bases, &one, &bases));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_pre_config(1, 1, &lens));
    try testing.expectEqual(@as(i32, -1), c.q3ld_pre_read_lens(3, 1000, 1, 1, &bases, &lens));
    try testing.expectEqual(@as(i32, -1), q3raw.q3ld_pre_read(3, 1000, 1, 1, &bases));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_ev_config(2, @intFromPtr(&word), std.time.ns_per_s, 0));
    for ([_]i32{ 0, 1, io.max_gates + 1 }) |n| try testing.expectEqual(@as(i32, -1), c.q3ld_ev_gates(n, &vals, &one, &tix));
    try testing.expectEqual(@as(i32, -1), c.q3ld_ev_release(1));
    var ev: [10]i64 = undefined;
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_ev_state(&ev));
    try testing.expectEqual(@as(i64, 0), ev[0]);
    try testing.expectEqual(@as(i64, 0), ev[8]);
    var pre: [pre_state_w * max_pre]i64 = undefined;
    try testing.expectEqual(@as(i32, 0), c.q3ld_pre_state(&pre));
    var sp: [spec_state_w * max_spec]i64 = undefined;
    try testing.expectEqual(@as(i32, 0), c.q3ld_spec_state(&sp));
    // A stop restores the stock idle rule; the horizon count is the header's.
    try testing.expectEqual(@as(i32, 1), q3raw.q3ld_spec_idle_busy());
    try testing.expectEqual(@as(i32, 4), q3raw.q3ld_max_h());
    try testing.expectEqual(@as(i64, 0), word);
}

test "dsv41 io cov: the stopped pool's configuration refuses bad arguments; the start refuses an unconfigured pool and misaligned buffers" {
    try resetPoolState();
    const page = std.heap.pageSize();
    const pg: i64 = @intCast(page);
    var counters: [io.counters_n]i64 = @splat(0);
    const mem = try std.heap.page_allocator.alloc(u8, 2 * page);
    defer std.heap.page_allocator.free(mem);
    const p0: u64 = @intFromPtr(mem.ptr);
    const bufs = [1]u64{p0};
    const Bad = struct { t: i32 = 1, n: i32 = 1, slot: i64, rec: i64 = 100, chunk: i64, ctr: bool = true };
    for ([_]Bad{
        .{ .t = -1, .slot = pg, .chunk = pg },
        .{ .t = io.max_spec_threads + 1, .slot = pg, .chunk = pg },
        .{ .n = -1, .slot = pg, .chunk = pg },
        .{ .n = max_spec + 1, .slot = pg, .chunk = pg },
        .{ .n = 0, .slot = pg, .chunk = pg },
        .{ .rec = 0, .slot = pg, .chunk = pg },
        .{ .rec = pg + 1, .slot = pg, .chunk = pg },
        .{ .slot = pg, .chunk = 0 },
        .{ .slot = pg, .chunk = pg, .ctr = false },
    }) |b| try testing.expectEqual(@as(c_int, -1), c.q3ld_spec_config(b.t, &bufs, b.n, b.slot, b.rec, b.chunk, if (b.ctr) &counters else null));
    for ([_]i32{ -1, 2 }) |v| try testing.expectEqual(@as(c_int, -1), c.q3ld_spec_streams(v));

    var res_arr: [16 * res_w]i64 = @splat(0);
    var log_arr: [16]i64 = @splat(0);
    var gauge: [6]i64 = @splat(0);
    const staging = [1]u64{p0};
    // No q3ld_spec_config since the last stop.
    try testing.expectEqual(@as(c_int, -1), c.q3ld_start(1, &staging, pg, pg, &res_arr, 16, &log_arr, 16, &gauge));
    // Configured without the speculative class (counters optional): each start argument out of range.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_spec_config(0, null, 0, 0, 0, 0, null));
    for ([_][3]i64{ .{ 0, pg, pg }, .{ io.max_workers + 1, pg, pg }, .{ 1, pg, 0 }, .{ 1, pg + 1, pg } }) |a|
        try testing.expectEqual(@as(c_int, -1), c.q3ld_start(@intCast(a[0]), &staging, a[1], a[2], &res_arr, 16, &log_arr, 16, &gauge));
    // A speculative slot off its page, a slot size or a chunk that is not a page multiple.
    const odd = [1]u64{p0 + 8};
    for ([_]struct { b: *const [1]u64, slot: i64, chunk: i64 }{ .{ .b = &odd, .slot = pg, .chunk = pg }, .{ .b = &bufs, .slot = pg + 8, .chunk = pg }, .{ .b = &bufs, .slot = pg, .chunk = 100 } }) |s| {
        try testing.expectEqual(@as(c_int, 0), c.q3ld_spec_config(1, s.b, 1, s.slot, 100, s.chunk, &counters));
        try testing.expectEqual(@as(c_int, -1), c.q3ld_start(1, &staging, pg, pg, &res_arr, 16, &log_arr, 16, &gauge));
    }
    // A running pool refuses every configuration call and a second start; its stop returns it to unconfigured.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_spec_config(0, null, 0, 0, 0, 0, &counters));
    try testing.expectEqual(@as(c_int, 0), c.q3ld_start(1, &staging, pg, pg, &res_arr, 16, &log_arr, 16, &gauge));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_start(1, &staging, pg, pg, &res_arr, 16, &log_arr, 16, &gauge));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_spec_config(0, null, 0, 0, 0, 0, &counters));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_spec_streams(0));
    try testing.expectEqual(@as(c_int, 0), c.q3ld_stop());
    try testing.expectEqual(@as(c_int, -1), c.q3ld_start(1, &staging, pg, pg, &res_arr, 16, &log_arr, 16, &gauge));
}

test "dsv41 io cov: a running pool refuses a malformed job and a pending ticket by their codes" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(8 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 32 });
    defer pool.stop();
    const fd = f.ufd.fd;
    const size: i64 = @intCast(f.ufd.size);
    var dst: [20]u8 = @splat(0xAA);
    const lens = [2]i64{ 10, 10 };
    const row = [2]u64{ @intFromPtr(&dst), @intFromPtr(&dst) + 10 };
    const rows = [1][*]const u64{&row};
    const offs = [2]i64{ 0, 100 };
    const Shape = struct { n: i32 = 1, ngu: i32 = 1, ndown: i32 = 1, first: i64 = 0 };
    for ([_]Shape{ .{ .n = 0 }, .{ .n = max_items + 1 }, .{ .ngu = 0 }, .{ .ngu = 7 }, .{ .ndown = 0 }, .{ .ndown = 7 }, .{ .first = -1 }, .{ .first = 31 } }) |s| {
        try testing.expectEqual(@as(c_int, -1), c.q3ld_submit(fd, size, -1, s.n, s.ngu, s.ndown, &offs, &rows, &lens, s.first));
    }
    // A ticket still pending (its status word) refuses the submit, raw and through Records.
    pool.res[5 * res_w] = @intFromEnum(Status.pending);
    try testing.expectEqual(@as(c_int, -2), rawSubmit(pool, fd, size, -1, 0, 100, lens[0..1], lens[1..2], &dst, 4));
    pool.res[5 * res_w] = 0;
    const small = [n_components]u64{ 10, 10, 10, 10, 10, 10, 10, 10, 10 };
    var d = try Dests.init(1, &small);
    defer testing.allocator.free(d.buf);
    pool.res[0] = @intFromEnum(Status.pending);
    try testing.expectError(error.TicketsBusy, R96.submit(pool, f.ufd, &.{0}, &.{1000}, d.rows[0..1], &small));
    pool.res[0] = 0;
    // Shapes Records refuses before the pool: no record, offsets that do not match the rows.
    try testing.expectError(error.InvalidJob, R96.submit(pool, f.ufd, &.{}, &.{}, d.rows[0..0], &small));
    try testing.expectError(error.InvalidJob, R96.submit(pool, f.ufd, &.{ 0, 1 }, &.{1000}, d.rows[0..1], &small));
    // A well-formed raw job lands: its two one-part ranges, byte for byte.
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, size, -1, 3, page + 7, lens[0..1], lens[1..2], &dst, 10));
    try pool.wait(10, 2, 10 * std.time.ns_per_s);
    try testing.expectEqualSlices(u8, f.image[3..13], dst[0..10]);
    try testing.expectEqualSlices(u8, f.image[page + 7 ..][0..10], dst[10..20]);
    for (10..12) |t| try testing.expectEqual(Status.ok, pool.result(@intCast(t)).status);
}

test "dsv41 io cov: a range past its deadline publishes deadline; a zero-length range finishes before any check; a far deadline reads" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(8 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 32 });
    defer pool.stop();
    const fd = f.ufd.fd;
    const size: i64 = @intCast(f.ufd.size);
    var dst: [500]u8 = @splat(0xAA);
    // Deadline 0 is past at the range's first check: nothing read, nothing written.
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, size, 0, 500, 3 * page + 7, &.{300}, &.{200}, &dst, 0));
    try pool.wait(0, 2, 10 * std.time.ns_per_s);
    const late = pool.result(0);
    try testing.expectEqual(Status.deadline, late.status);
    try testing.expectEqual(@as(i64, 0), late.payload);
    try testing.expectEqual(@as(i64, 0), late.preadv_calls);
    try testing.expectEqual(Status.deadline, pool.result(1).status);
    try testing.expect(std.mem.allEqual(u8, &dst, 0xAA));
    // A zero-length gate/up range has nothing to read: OK past the deadline; its down range is late.
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, size, 0, 500, 3 * page + 7, &.{0}, &.{200}, &dst, 2));
    try pool.wait(2, 2, 10 * std.time.ns_per_s);
    try testing.expectEqual(Status.ok, pool.result(2).status);
    try testing.expectEqual(@as(i64, 0), pool.result(2).payload);
    try testing.expectEqual(Status.deadline, pool.result(3).status);
    // A deadline an hour away reads as no deadline.
    const far = c.q3ld_monotonic_ns() + 3600 * std.time.ns_per_s;
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, size, far, 500, 3 * page + 7, &.{300}, &.{200}, &dst, 4));
    try pool.wait(4, 2, 10 * std.time.ns_per_s);
    try testing.expectEqual(Status.ok, pool.result(4).status);
    try testing.expectEqual(Status.ok, pool.result(5).status);
    try testing.expectEqualSlices(u8, f.image[500..800], dst[0..300]);
    try testing.expectEqualSlices(u8, f.image[3 * page + 7 ..][0..200], dst[300..500]);
}

test "dsv41 io cov: ranges at and past the end of file, zero-length parts, a short file size, a range over many stagings and a bad descriptor" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(6 * page + 100);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 64 });
    defer pool.stop();
    const fd = f.ufd.fd;
    const size: i64 = @intCast(f.ufd.size);
    const S: u64 = f.ufd.size;
    const dst = try testing.allocator.alloc(u8, 4 * page);
    defer testing.allocator.free(dst);
    const Case = struct { gu: u64, gl: []const i64, dl: []const i64 = &.{10}, size: i64, fd: i32, want: Status, payload: i64, calls: ?i64 = null };
    const cases = [_]Case{
        // Straddles the end: the 50 bytes that exist land, then the range is short; its down range is its own read.
        .{ .gu = S - 50, .gl = &.{100}, .size = size, .fd = fd, .want = .short, .payload = 50, .calls = 1 },
        // Starts at the end, or a page past it: short, nothing read.
        .{ .gu = S, .gl = &.{10}, .size = size, .fd = fd, .want = .short, .payload = 0, .calls = 0 },
        .{ .gu = S + page, .gl = &.{10}, .size = size, .fd = fd, .want = .short, .payload = 0, .calls = 0 },
        // The caller's file size bounds the read, whatever the file holds.
        .{ .gu = 0, .gl = &.{200}, .size = 100, .fd = fd, .want = .short, .payload = 100 },
        // Zero-length parts (a leading one, one between, a trailing one) are skipped; the rest lands.
        .{ .gu = 3, .gl = &.{ 0, 7, 0 }, .dl = &.{ 0, 5 }, .size = size, .fd = fd, .want = .ok, .payload = 7 },
        // Unaligned, three stagings and a page wide: four or more aligned reads.
        .{ .gu = page - 1, .gl = &.{ page + 2, 2 * page + 1 }, .size = size, .fd = fd, .want = .ok, .payload = 3 * page + 3 },
        // A bad descriptor: the preadv's errno.
        .{ .gu = 0, .gl = &.{10}, .size = size, .fd = -1, .want = .os_error, .payload = 0, .calls = 1 },
    };
    for (cases, 0..) |cs, i| {
        @memset(dst, 0xAA);
        const t: u32 = @intCast(2 * i);
        const down: u64 = 2 * page + 9;
        try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, cs.fd, cs.size, -1, cs.gu, down, cs.gl, cs.dl, dst, t));
        try pool.wait(t, 2, 10 * std.time.ns_per_s);
        const r = pool.result(t);
        testing.expectEqual(cs.want, r.status) catch |e| {
            std.debug.print("case {d}\n", .{i});
            return e;
        };
        try testing.expectEqual(cs.payload, r.payload);
        if (cs.calls) |n| try testing.expectEqual(n, r.preadv_calls);
        var gl_total: usize = 0;
        for (cs.gl) |l| gl_total += @intCast(l);
        const got: usize = @intCast(r.payload);
        // The landed bytes are the file's (zero-length parts take no room), the rest of the rows untouched.
        if (got > 0) try testing.expectEqualSlices(u8, f.image[cs.gu..][0..got], dst[0..got]);
        try testing.expect(std.mem.allEqual(u8, dst[got..gl_total], 0xAA));
        // The down range is its own job: read in full, short past the caller's file size, the descriptor's errno.
        var dl_total: usize = 0;
        for (cs.dl) |l| dl_total += @intCast(l);
        const down_want: Status = if (cs.fd < 0) .os_error else if (down + dl_total > @as(u64, @intCast(cs.size))) .short else .ok;
        try testing.expectEqual(down_want, pool.result(t + 1).status);
        if (down_want == .ok) try testing.expectEqualSlices(u8, f.image[down..][0..dl_total], dst[gl_total..][0..dl_total]);
        if (cs.want == .os_error) try testing.expectEqual(@as(i64, @intFromEnum(std.posix.E.BADF)), r.errno);
        if (cs.gu == page - 1) try testing.expect(r.preadv_calls >= 4);
    }
}

test "dsv41 io cov: jobs queued behind a held worker stay pending (a resubmit is refused), then drain in order once a quiesce frees it" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(1, 2);
    defer pool.stop();
    try R96.armPreRead(pool, &spec_lens);
    // The only worker takes the pre-read's gate/up range and waits at its bind point; its down range stays queued.
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 100, &.{@intCast(40 * page + 3)}, &spec_lens));
    try waitFor(pool, preStarted1);
    var d = try Dests.init(3, &spec_lens);
    defer testing.allocator.free(d.buf);
    var firsts: [3]u32 = undefined;
    const bases = [3]u64{ 2 * page + 1, 10 * page + 2, 20 * page + 3 };
    for (&firsts, bases, 0..) |*fst, b, i| fst.* = try R96.submit(pool, f.ufd, &.{b}, &.{b + spec_gu_len}, d.rows[i..][0..1], &spec_lens);
    for (firsts) |fst| for (0..2) |k| try testing.expectEqual(Status.pending, pool.result(fst + @as(u32, @intCast(k))).status);
    var scratch: [20]u8 = undefined;
    try testing.expectEqual(@as(c_int, -2), rawSubmit(pool, f.ufd.fd, @intCast(f.ufd.size), -1, 0, 100, &.{10}, &.{10}, &scratch, firsts[1]));
    // Quiesce expires the unbound ranges (the queued one freed, the held one cancelled at its bind point) and drains.
    const seq0 = c.q3ld_seq();
    try testing.expectEqual(@as(c_int, 0), c.q3ld_quiesce(10 * std.time.ns_per_s));
    for (firsts) |fst| try pool.wait(fst, 2, 10 * std.time.ns_per_s);
    for (bases, 0..) |b, i| try d.expectRecord(i, f.image, b, b + spec_gu_len, &spec_lens);
    var order: [6]u32 = undefined;
    const got = pool.logOrder(seq0, c.q3ld_seq(), &order);
    try testing.expectEqualSlices(u32, &.{ firsts[0], firsts[0] + 1, firsts[1], firsts[1] + 1, firsts[2], firsts[2] + 1 }, got);
    try testing.expectEqual(@as(i64, 2), pool.counter(.pre_expired));
    try testing.expectEqual(@as(i64, 1), pool.counter(.pre_started));
    try testing.expectEqual(@as(i64, 0), pool.counter(.pre_served));
    try testing.expectEqual(@as(i64, 1), pool.readGauge()[1]);
    try testing.expect(preIdle(pool));
}

fn readInFlight(p: *Pool) bool {
    return p.readGauge()[0] >= 1;
}

/// X21's withdraw test: the event log it reads (kind, qlen, busy, t, arg per entry).
const WithdrawLog = struct {
    var buf: [5 * 32]i64 = undefined;
    var off: [5]i64 = undefined;
};

test "dsv41 io cov X21: a queued job is withdrawn unread (its tickets published as withdrawn, its rows untouched, no start in the event log, its tickets free again); a taken job runs as submitted; a gated job is refused" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = 4 * page, .tickets = 64 });
    defer pool.stop();
    defer clearFaults();
    var word: i64 align(8) = 0;
    try pool.armEvent(.host, @intFromPtr(&word), 10 * std.time.ns_per_s, 0);
    _ = q3raw.q3ld_test_events(&WithdrawLog.buf, 32);
    defer _ = q3raw.q3ld_test_events(&WithdrawLog.off, 0);
    const bases = [4]u64{ 3 * page + 11, 12 * page + 5, 22 * page + 7, 32 * page + 9 };
    var d = try Dests.init(4, &spec_lens);
    defer testing.allocator.free(d.buf);
    // The only worker takes job 0 and sleeps 300 ms at its first read: jobs 1-3 queue behind it.
    const pg: u64 = page;
    injectFault(bases[0] / pg * pg, 5, 300 * std.time.ns_per_ms);
    var firsts: [4]u32 = undefined;
    firsts[0] = try R96.submit(pool, f.ufd, bases[0..1], &.{bases[0] + spec_gu_len}, d.rows[0..1], &spec_lens);
    try waitFor(pool, readInFlight);
    for (1..4) |i| firsts[i] = try R96.submit(pool, f.ufd, bases[i..][0..1], &.{bases[i] + spec_gu_len}, d.rows[i..][0..1], &spec_lens);
    // Job 0 was taken: it runs as submitted.
    try testing.expect(!(try pool.withdraw(firsts[0])));
    // Job 2, queued: withdrawn unread, its tickets terminal at once.
    const seq0 = c.q3ld_seq();
    try testing.expect(try pool.withdraw(firsts[2]));
    try pool.wait(firsts[2], 2, std.time.ns_per_s);
    for (0..2) |k| {
        const r = pool.result(firsts[2] + @as(u32, @intCast(k)));
        try testing.expectEqual(Status.withdrawn, r.status);
        try testing.expectEqual(@as(i64, 0), r.payload);
        try testing.expectEqual(@as(i64, 0), r.preadv_calls);
    }
    // Withdrawn once, it is no longer queued; a ticket that starts no queued job withdraws nothing.
    try testing.expect(!(try pool.withdraw(firsts[2])));
    try testing.expect(!(try pool.withdraw(firsts[1] + 1)));
    // Job 3's tickets waited on by a live event gate (its bytes promised to the GPU): refused, it runs as submitted.
    try pool.registerGates(&.{1}, &.{2}, &.{ firsts[3], firsts[3] + 1 });
    try testing.expectEqual(@as(c_int, -2), c.q3ld_withdraw(firsts[3]));
    try testing.expect(!(try pool.withdraw(firsts[3])));
    // The worker wakes and runs jobs 0, 1 and 3 in order; job 2's rows were never written.
    for ([_]usize{ 0, 1, 3 }) |i| try pool.wait(firsts[i], 2, 10 * std.time.ns_per_s);
    for ([_]usize{ 0, 1, 3 }) |i| try d.expectRecord(i, f.image, bases[i], bases[i] + spec_gu_len, &spec_lens);
    for (0..n_components) |k| try testing.expect(std.mem.allEqual(u8, d.part(2, k, &spec_lens), 0xAA));
    var order: [8]u32 = undefined;
    const got = pool.logOrder(seq0, c.q3ld_seq(), &order);
    try testing.expectEqualSlices(u32, &.{ firsts[2], firsts[2] + 1, firsts[0], firsts[0] + 1, firsts[1], firsts[1] + 1, firsts[3], firsts[3] + 1 }, got);
    try testing.expectEqual(@as(i64, 1), @atomicLoad(i64, &word, .acquire));
    // The event log: four submits, job 2's withdrawal (kind 11), and a start (kind 2) for jobs 0, 1 and 3 only.
    const n: usize = @intCast(q3raw.q3ld_test_events(null, 0));
    var submits: u32 = 0;
    var withdrawals: u32 = 0;
    var started = [4]bool{ false, false, false, false };
    for (0..n) |i| {
        const e = WithdrawLog.buf[5 * i ..][0..5];
        switch (e[0]) {
            1 => submits += 1,
            11 => {
                withdrawals += 1;
                try testing.expectEqual(@as(i64, firsts[2]), e[4]);
            },
            2 => {
                for (firsts, 0..) |fst, j| {
                    if (e[4] == fst) started[j] = true;
                }
            },
            else => {},
        }
    }
    try testing.expectEqual(@as(u32, 4), submits);
    try testing.expectEqual(@as(u32, 1), withdrawals);
    try testing.expectEqualSlices(bool, &.{ true, true, false, true }, &started);
    // A withdrawn ticket is not pending: the ring submits on it again, and that job reads.
    var lens_i: [n_components]i64 = undefined;
    for (spec_lens, &lens_i) |l, *x| x.* = @intCast(l);
    var again: [spec_rec_len]u8 = @splat(0xAA);
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, f.ufd.fd, @intCast(f.ufd.size), -1, bases[2], bases[2] + spec_gu_len, lens_i[0..gu_components], lens_i[gu_components..], &again, firsts[2]));
    try pool.wait(firsts[2], 2, 10 * std.time.ns_per_s);
    try testing.expectEqual(Status.ok, pool.result(firsts[2]).status);
    try testing.expectEqualSlices(u8, f.image[bases[2]..][0..spec_rec_len], &again);
}

test "dsv41 io cov: the pre-read class refuses bad geometry and a re-arm under live ranges; it caps ranges at a staging buffer and its table" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(160 * page);
    defer f.deinit();
    var pool = try specPool(2, 2);
    defer pool.stop();
    const fd = f.ufd.fd;
    const size: i64 = @intCast(f.ufd.size);
    var l: [n_components]i64 = undefined;
    for (spec_lens, &l) |x, *y| y.* = @intCast(x);
    // Disarming an unarmed class is a no-op; bad counts and a missing length list are refused.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_pre_config(0, 0, null));
    for ([_][2]i32{ .{ 0, 1 }, .{ 1, 0 }, .{ 7, 1 }, .{ 1, 7 }, .{ -1, 1 } }) |g| try testing.expectEqual(@as(c_int, -1), c.q3ld_pre_config(g[0], g[1], &l));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_pre_config(6, 3, null));
    try R96.armPreRead(pool, &spec_lens);
    // Malformed calls: a negative count, a negative length.
    const b0 = [1]i64{@intCast(4 * page)};
    try testing.expectEqual(@as(i32, -1), c.q3ld_pre_read_lens(fd, size, 1, -1, &b0, &l));
    var neg = l;
    neg[7] = -5;
    try testing.expectEqual(@as(i32, -1), c.q3ld_pre_read_lens(fd, size, 1, 1, &b0, &neg));
    // Two ranges, both workers held at their bind points: a re-arm is refused while they live.
    const hold = [1]i64{@intCast(150 * page + 3)};
    try testing.expectEqual(@as(i32, 2), c.q3ld_pre_read_lens(fd, size, 100, 1, &hold, &l));
    try waitFor(pool, preStarted2);
    try testing.expectEqual(@as(c_int, -1), c.q3ld_pre_config(6, 3, &l));
    // The stock entry point (the armed lengths) on the same record refreshes the live ranges: nothing new.
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_pre_read(fd, size, 100, 1, &hold));
    // A gate/up range wider than a staging buffer (4 pages) is not pre-read; its down range is.
    var wide = l;
    wide[0] = @intCast(5 * page);
    try testing.expectEqual(@as(i32, 1), c.q3ld_pre_read_lens(fd, size, 100, 1, &b0, &wide));
    try testing.expectEqual(@as(i64, 1), pool.counter(.pre_noslot));
    // 20 records want 40 ranges; the table has 29 entries left: every gate/up, then 9 downs.
    var many: [20]i64 = undefined;
    for (&many, 0..) |*b, i| b.* = @intCast(10 * page + i * 5 * page + 1);
    try testing.expectEqual(@as(i32, 29), c.q3ld_pre_read_lens(fd, size, 100, 20, &many, &l));
    try testing.expectEqual(@as(i64, 12), pool.counter(.pre_noslot));
    try testing.expectEqual(@as(i64, 32), pool.counter(.pre_issued));
    var st: [pre_state_w * max_pre]i64 = undefined;
    try testing.expectEqual(@as(i32, 1), c.q3ld_pre_state(&st));
    var queued: usize = 0;
    var inflight: usize = 0;
    for (0..max_pre) |i| {
        queued += @intFromBool(st[i * pre_state_w] == 1);
        inflight += @intFromBool(st[i * pre_state_w] == 2);
    }
    try testing.expectEqual(@as(usize, 30), queued);
    try testing.expectEqual(@as(usize, 2), inflight);
    // The call's settle expires all 32: the held two cancelled at their bind points, the queued freed.
    _ = try pool.specStep(f.ufd, 100, &.{}, 0);
    try waitFor(pool, preIdle);
    try testing.expectEqual(@as(i64, 32), pool.counter(.pre_expired));
    try testing.expectEqual(@as(i64, 2), pool.counter(.pre_started));
    // Disarmed, the class refuses a pre-read.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_pre_config(0, 0, null));
    try testing.expectEqual(@as(i32, -1), c.q3ld_pre_read_lens(fd, size, 101, 1, &b0, &l));
}

test "dsv41 io cov: horizon-N steps keep, promote and drop queued records; a full table refuses; a live base refreshes; a pre-read cancels a queued record" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(160 * page);
    defer f.deinit();
    const fd = f.ufd.fd;
    const size: i64 = @intCast(f.ufd.size);
    var pool = try specPool(2, 4);
    defer pool.stop();
    try R96.armPreRead(pool, &spec_lens);
    // Both workers held at pre-range bind points (tag 100, past every step below): demand busy 2, so the queue rule
    // keeps every unclaimed record QUEUED.
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 100, &.{@intCast(150 * page + 3)}, &spec_lens));
    try waitFor(pool, preStarted2);
    const rec = struct {
        fn at(i: u64) i64 {
            return @intCast(2 * std.heap.pageSize() + i * 6 * std.heap.pageSize() + i);
        }
    }.at;
    const A = rec(0);
    const B = rec(1);
    const C = rec(2);
    const D = rec(3);
    const E = rec(4);
    // Horizon 1 (A, B: tag 1, class 0); horizon 2 (C, D: tag 2, class 1).
    try testing.expectEqual(@as(u32, 2), try pool.specStep(f.ufd, 0, &.{ A, B }, 0));
    const no_val = [1]i64{0};
    try testing.expectEqual(@as(i32, 2), q3raw.q3ld_spec_stepn(fd, size, 0, 2, &[2]i32{ 0, 0 }, &no_val, &[2]i32{ 0, 2 }, &[2]i64{ C, D }));
    // The next call (cur 1): A and B settle; C is still a candidate for the next layer (kept, promoted to class 0),
    // D is not (dropped); E is issued two ahead.
    try testing.expectEqual(@as(i32, 1), q3raw.q3ld_spec_stepn(fd, size, 1, 2, &[2]i32{ 1, 0 }, &[1]i64{C}, &[2]i32{ 0, 1 }, &[1]i64{E}));
    try testing.expectEqual(@as(i64, 3), pool.counter(.expired));
    try testing.expectEqual(@as(i64, 1), counterAt(pool, @intFromEnum(io.Counter.h_kept) + 1));
    try testing.expectEqual(@as(i64, 1), counterAt(pool, @intFromEnum(io.Counter.h_dropped) + 1));
    try testing.expectEqual(@as(i64, 2), counterAt(pool, @intFromEnum(io.Counter.h_submitted)));
    try testing.expectEqual(@as(i64, 3), counterAt(pool, @intFromEnum(io.Counter.h_submitted) + 1));
    var raw_st: [spec_state_w * max_spec]i64 = undefined;
    try testing.expectEqual(@as(i32, 4), c.q3ld_spec_state(&raw_st));
    var seen: u32 = 0;
    for (0..4) |i| {
        const w = raw_st[i * spec_state_w ..][0..spec_state_w];
        if (w[0] == 0) continue;
        try testing.expectEqual(@as(i64, 1), w[0]);
        if (w[2] == C) {
            try testing.expectEqualSlices(i64, &.{ 2, 0, 2 }, &.{ w[1], w[8], w[10] });
            seen += 1;
        } else if (w[2] == E) {
            try testing.expectEqualSlices(i64, &.{ 3, 1, 2 }, &.{ w[1], w[8], w[10] });
            seen += 1;
        }
    }
    try testing.expectEqual(@as(u32, 2), seen);
    // Four slots, C and E live: two of three new records find a slot, the third none (no landed slot to reclaim).
    try testing.expectEqual(@as(u32, 2), try pool.specStep(f.ufd, 1, &.{ rec(5), rec(6), rec(7) }, 0));
    try testing.expectEqual(@as(i64, 1), pool.counter(.noslot));
    // A live base is refreshed, never queued twice (both h1 entry points).
    try testing.expectEqual(@as(u32, 0), try pool.specStep(f.ufd, 1, &.{C}, 0));
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_spec_step(fd, size, 1, 1, &[1]i64{C}));
    try testing.expectEqual(@as(i64, 2), pool.counter(.refreshed));
    // A record length past the configured one is refused.
    try testing.expectError(error.SpecRefused, pool.specStep(f.ufd, 1, &.{A}, spec_rec_len + 1));
    // A pre-read of the queued C cancels it (its bytes are pre-read instead): two ranges queued.
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 100, &.{C}, &spec_lens));
    try testing.expectEqual(@as(i64, 1), pool.counter(.cancelled_by_demand));
    try testing.expectEqual(@as(i64, 0), pool.counter(.started));
    // The stop's quiesce frees the queued records and the held workers.
}

test "dsv41 io cov: a failed range leaves the pre-ranges bound to its submit's other ranges served" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(2, 2);
    defer pool.stop();
    defer clearFaults();
    try R96.armPreRead(pool, &spec_lens);
    // ra's gate/up range starts in page 2 and ends in page 3, its down range in page 3: the fault on page 2 hits the
    // gate/up range alone.
    const ra: u64 = 3 * page - 6000;
    const rb: u64 = 20 * page + 9;
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 1, &.{@intCast(rb)}, &spec_lens));
    try waitFor(pool, preStarted2);
    injectFault(ra / page * page, 2, 0);
    var d = try Dests.init(2, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &.{ ra, rb }, &.{ ra + spec_gu_len, rb + spec_gu_len }, d.rows[0..2], &spec_lens);
    try pool.wait(first, 4, 10 * std.time.ns_per_s);
    // ra's gate/up fails; its down range and rb's two (served by their pre-ranges) land.
    const want = [4]Status{ .os_error, .ok, .ok, .ok };
    for (want, 0..) |w, k| try testing.expectEqual(w, pool.result(first + @as(u32, @intCast(k))).status);
    try d.expectRecord(1, f.image, rb, rb + spec_gu_len, &spec_lens);
    try testing.expectEqual(@as(i64, 2), pool.counter(.pre_bound));
    try testing.expectEqual(@as(i64, 2), pool.counter(.pre_served));
    try waitFor(pool, preIdle);
}

test "dsv41 io cov: the event class's arming refusals, the gate ring's bounds, and a host release past every gate" {
    const page = std.heap.pageSize();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 128 });
    defer pool.stop();
    var word: i64 align(8) = 0;
    const w: u64 = @intFromPtr(&word);
    var st: [10]i64 = undefined;
    // Off while off is a no-op; a bad kind, a null object, timeouts outside 1 ms .. 600 s.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_ev_config(0, 0, 0, 0));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_ev_config(3, w, std.time.ns_per_s, 0));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_ev_config(2, 0, std.time.ns_per_s, 0));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_ev_config(2, w, std.time.ns_per_ms - 1, 0));
    try testing.expectEqual(@as(c_int, -1), c.q3ld_ev_config(2, w, 600 * std.time.ns_per_s + 1, 0));
    // Armed at 5: the event's value is taken as handed over; a second arm is refused.
    try pool.armEvent(.host, w, 60 * std.time.ns_per_s, 5);
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_ev_state(&st));
    try testing.expectEqualSlices(i64, &.{ 2, 0, 0, 0, 0, 5, 5, 5, 1, 60 * std.time.ns_per_s }, &st);
    try testing.expectError(error.EventRefused, pool.armEvent(.host, w, std.time.ns_per_s, 0));
    try testing.expectEqual(@as(i64, 0), word);
    // Disarmed (the watchdog joined): release and gates are refused; it arms again.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_ev_config(0, 0, 0, 0));
    _ = q3raw.q3ld_ev_state(&st);
    try testing.expectEqual(@as(i64, 0), st[0]);
    try testing.expectEqual(@as(i64, 0), st[8]);
    try testing.expectEqual(@as(i32, -1), c.q3ld_ev_release(1));
    try testing.expectError(error.GateRefused, pool.registerGates(&.{1}, &.{0}, &.{}));
    try pool.armEvent(.host, w, 60 * std.time.ns_per_s, 0);
    // A gate over a pending ticket heads the ring; a second gate may not wait for that ticket, nor for a negative count.
    pool.res[60 * res_w] = @intFromEnum(Status.pending);
    defer pool.res[60 * res_w] = 0;
    try pool.registerGates(&.{1}, &.{1}, &.{60});
    try testing.expectError(error.GateInvalid, pool.registerGates(&.{2}, &.{1}, &.{60}));
    try testing.expectError(error.GateInvalid, pool.registerGates(&.{2}, &.{-1}, &.{}));
    // max_gates - 1 satisfied gates queue behind the unsatisfied head: the ring is full at max_gates live.
    var values: [io.max_gates - 1]u64 = undefined;
    for (&values, 0..) |*v, i| v.* = 2 + i;
    const zeros: [io.max_gates - 1]i32 = @splat(0);
    try pool.registerGates(&values, &zeros, &.{});
    try testing.expectEqual(@as(i32, io.max_gates), q3raw.q3ld_ev_state(&st));
    try testing.expectEqualSlices(i64, &.{ 1, 1, 0 }, st[2..5]);
    try testing.expectEqual(@as(i64, 0), word);
    try testing.expectError(error.GatesFull, pool.registerGates(&.{io.max_gates + 1}, &.{0}, &.{}));
    // Live gates refuse a disarm.
    try testing.expectEqual(@as(c_int, -1), c.q3ld_ev_config(0, 0, 0, 0));
    // The host release forces the head (the only gate still waiting): the prefix runs to the last gate.
    try testing.expectEqual(@as(i32, 1), c.q3ld_ev_release(io.max_gates));
    try testing.expectEqual(@as(i64, io.max_gates), word);
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_ev_state(&st));
    try testing.expectEqual(@as(i64, 1), pool.counter(.ev_host_released));
    try testing.expectEqual(@as(i64, io.max_gates - 1), pool.counter(.ev_immediate));
    // With no live gate, a released value goes to the event too, and later gates must rise above it.
    const past = io.max_gates + 44;
    try testing.expectEqual(@as(i32, 0), c.q3ld_ev_release(past));
    try testing.expectEqual(@as(i64, past), word);
    try testing.expectError(error.GateInvalid, pool.registerGates(&.{past}, &.{0}, &.{}));
    try pool.registerGates(&.{past + 1}, &.{0}, &.{});
    try testing.expectEqual(@as(i64, past + 1), word);
}

test "dsv41 io cov: a gate forced before its ticket lands ignores the late publish; the next gate's ticket still satisfies it" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(1, 2);
    defer pool.stop();
    var word: i64 align(8) = 0;
    try pool.armEvent(.host, @intFromPtr(&word), 60 * std.time.ns_per_s, 0);
    try R96.armPreRead(pool, &spec_lens);
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 100, &.{@intCast(40 * page + 3)}, &spec_lens));
    try waitFor(pool, preStarted1);
    // The job queues behind the held worker: both tickets pending, each under a gate.
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const base: u64 = 4 * page + 1;
    const first = try R96.submit(pool, f.ufd, &.{base}, &.{base + spec_gu_len}, d.rows[0..1], &spec_lens);
    try pool.registerGates(&.{ 1, 2 }, &.{ 1, 1 }, &.{ first, first + 1 });
    try testing.expectEqual(@as(i32, 1), c.q3ld_ev_release(1));
    try testing.expectEqual(@as(i64, 1), word);
    // The quiesce frees the worker: the forced gate's ticket publishes into nothing, the second gate is satisfied.
    try testing.expectEqual(@as(c_int, 0), c.q3ld_quiesce(10 * std.time.ns_per_s));
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try testing.expectEqual(@as(i64, 2), @atomicLoad(i64, &word, .acquire));
    try d.expectRecord(0, f.image, base, base + spec_gu_len, &spec_lens);
    try testing.expectEqual(@as(i64, 1), pool.counter(.ev_host_released));
    try testing.expectEqual(@as(i64, 2), pool.counter(.ev_signals));
    try testing.expectEqual(@as(i64, 0), pool.counter(.ev_wd_forced));
}

test "dsv41 io cov: a pool reads a record; the test event log names submit, start and the spec delay hook" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(16 * page);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = 4 * page, .tickets = 32 });
    defer pool.stop();
    const Log = struct {
        var buf: [5 * 16]i64 = undefined;
        var off: [5]i64 = undefined;
    };
    _ = q3raw.q3ld_test_events(&Log.buf, 16);
    defer _ = q3raw.q3ld_test_events(&Log.off, 0);
    q3raw.q3ld_test_spec_delay(0);
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const base: u64 = 3 * page + 11;
    const first = try R96.submit(pool, f.ufd, &.{base}, &.{base + spec_gu_len}, d.rows[0..1], &spec_lens);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try d.expectRecord(0, f.image, base, base + spec_gu_len, &spec_lens);
    // Submit (kind 1: its two ranges queued, no worker busy), then the worker's start of each (kind 2), in ticket order.
    const n: usize = @intCast(q3raw.q3ld_test_events(null, 0));
    try testing.expectEqual(@as(usize, 3), n);
    try testing.expectEqualSlices(i64, &.{ 1, 2, 0 }, Log.buf[0..3]);
    try testing.expectEqualSlices(i64, &.{ 2, 1, 0 }, Log.buf[5..8]);
    try testing.expectEqualSlices(i64, &.{ 2, 0, 0 }, Log.buf[10..13]);
    try testing.expectEqual(@as(i64, first), Log.buf[4]);
    try testing.expectEqual(@as(i64, first), Log.buf[9]);
    try testing.expectEqual(@as(i64, first + 1), Log.buf[14]);
    try testing.expect(Log.buf[8] >= Log.buf[3] and Log.buf[13] >= Log.buf[8]);
}

fn specStateIs(base: i64, state: i64) bool {
    var slots: [max_spec]SpecSlot = undefined;
    for (specSlots(&slots)) |s| if (s.base == base and s.state == state) return true;
    return false;
}

fn failedAt(base: i64) bool {
    return specStateIs(base, 4);
}

test "dsv41 io cov: a speculative record past the end fails; one the file ends inside lands short and serves only what landed" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(20 * page);
    defer f.deinit();
    var pool = try specPool(1, 4);
    defer pool.stop();
    const fd = f.ufd.fd;
    const S: i64 = @intCast(f.ufd.size);
    const pg: i64 = @intCast(page);
    // Past the end of the file: no span to read, the record fails.
    try testing.expectEqual(@as(i32, 1), c.q3ld_spec_step_len(fd, S, 0, 1, &[1]i64{S + pg}, spec_rec_len));
    try waitFor(S + pg, failedAt);
    // A caller's size past the real end: the record lands the 3000 bytes the file holds.
    const short = S - 3000;
    const big = S + 10 * pg;
    try testing.expectEqual(@as(i32, 1), c.q3ld_spec_step_len(fd, big, 0, 1, &[1]i64{short}, spec_rec_len));
    try waitFor(short, landedAt);
    var slots: [max_spec]SpecSlot = undefined;
    for (specSlots(&slots)) |s| if (s.base == short) try testing.expectEqual(@as(i64, 3000), s.landed);
    try testing.expectEqual(@as(i64, 1), pool.counter(.failed));
    // A range inside what landed is copied out (no preadv); one past it is read the stock way and is short.
    var dst: [4010]u8 = @splat(0xAA);
    const s_u: u64 = @intCast(short);
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, big, -1, s_u, s_u + 1000, &.{1000}, &.{500}, &dst, 0));
    try pool.wait(0, 2, 10 * std.time.ns_per_s);
    for (0..2) |k| {
        try testing.expectEqual(Status.ok, pool.result(@intCast(k)).status);
        try testing.expectEqual(@as(i64, 0), pool.result(@intCast(k)).preadv_calls);
    }
    try testing.expectEqualSlices(u8, f.image[s_u..][0..1500], dst[0..1500]);
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, big, -1, s_u, s_u + 1000, &.{4000}, &.{10}, &dst, 2));
    try pool.wait(2, 2, 10 * std.time.ns_per_s);
    try testing.expectEqual(Status.short, pool.result(2).status);
    try testing.expectEqual(@as(i64, 3000), pool.result(2).payload);
    try testing.expect(pool.result(2).preadv_calls >= 1);
    // Its down range is its own job, inside what landed: copied out.
    try testing.expectEqual(Status.ok, pool.result(3).status);
    try testing.expectEqual(@as(i64, 0), pool.result(3).preadv_calls);
    try testing.expectEqual(@as(i64, 3), pool.counter(.adopt_ranges));
}

fn specStarted(n: i64) type {
    return struct {
        fn f(p: *Pool) bool {
            return p.counter(.started) >= n;
        }
    };
}

fn parked1(p: *Pool) bool {
    return p.counter(.parks) >= 1;
}

fn paused1(p: *Pool) bool {
    return p.counter(.pauses) >= 1;
}

fn abandoned2(p: *Pool) bool {
    return p.counter(.abandoned) >= 2;
}

fn preStarted4(p: *Pool) bool {
    return p.counter(.pre_started) >= 4;
}

test "dsv41 io cov: an in-flight record parks (class 1) or pauses (class 0) when demand turns busy; a drop or settle abandons it; a claim waits for its chunk" {
    // Timing-guarded: each record's first chunk sleeps 400 ms in the pool (an injected rule), and the test moves demand
    // busy (two workers held at pre-range bind points) inside that window. A box that stalls the test thread past it
    // skips the step, never asserts on a race.
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    const fd = f.ufd.fd;
    const size: i64 = @intCast(f.ufd.size);
    const pg: i64 = @intCast(page);
    const rec: u64 = 3 * page;
    var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = 4 * page, .tickets = 128, .spec = .{ .threads = 1, .slots = 4, .record_bytes = rec, .chunk_bytes = page } });
    defer pool.stop();
    defer clearFaults();
    try R96.armPreRead(pool, &spec_lens);
    var l: [n_components]i64 = undefined;
    for (spec_lens, &l) |x, *y| y.* = @intCast(x);
    const none = [1]i64{0};
    const hold_ms = 400 * std.time.ns_per_ms;

    // Park: R (horizon 2, class 1) starts with demand idle; demand turns busy during its first chunk.
    const R: i64 = 10 * pg;
    injectFault(@intCast(R), 5, hold_ms);
    try testing.expectEqual(@as(i32, 1), q3raw.q3ld_spec_stepn(fd, size, 0, 2, &[2]i32{ 0, 0 }, &none, &[2]i32{ 0, 1 }, &[1]i64{R}));
    try waitFor(pool, specStarted(1).f);
    try testing.expectEqual(@as(i32, 2), c.q3ld_pre_read_lens(fd, size, 100, 1, &[1]i64{40 * pg + 3}, &l));
    try waitFor(pool, preStarted2);
    if (pool.counter(.spec_chunks) != 0) {
        std.debug.print("SPECPROBE skipped: the first chunk ended before demand turned busy (box loaded)\n", .{});
        return error.SkipZigTest;
    }
    try waitFor(pool, parked1);
    // The next call no longer predicts R two ahead: the parked record is dropped with the chunk it read.
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_spec_stepn(fd, size, 1, 2, &[2]i32{ 0, 0 }, &none, &[2]i32{ 0, 0 }, &none));
    try testing.expectEqual(@as(i64, 1), pool.counter(.abandoned));
    try testing.expectEqual(pg, pool.counter(.abandoned_bytes));
    try testing.expectEqual(@as(i64, 1), counterAt(pool, @intFromEnum(io.Counter.h_dropped) + 1));
    // Demand idle again (a later pre-read call expires the held ranges).
    try testing.expectEqual(@as(i32, 0), c.q3ld_pre_read_lens(fd, size, 101, 0, &none, &l));
    try waitFor(pool, preIdle);

    // Pause: P (horizon 1, class 0) starts; demand turns busy during its first chunk; it waits at the boundary.
    const P: i64 = 20 * pg;
    injectFault(@intCast(P), 5, hold_ms);
    try testing.expectEqual(@as(u32, 1), try pool.specStep(f.ufd, 1, &.{P}, rec));
    try waitFor(pool, specStarted(2).f);
    try testing.expectEqual(@as(i32, 2), c.q3ld_pre_read_lens(fd, size, 200, 1, &[1]i64{44 * pg + 5}, &l));
    try waitFor(pool, preStarted4);
    if (pool.counter(.spec_chunks) != 1) {
        std.debug.print("SPECPROBE skipped: P's first chunk ended before demand turned busy (box loaded)\n", .{});
        return error.SkipZigTest;
    }
    try waitFor(pool, paused1);
    // Its call's settle flags the running record: it is abandoned at the boundary it waits at.
    _ = try pool.specStep(f.ufd, 2, &.{}, rec);
    try waitFor(pool, abandoned2);
    try testing.expectEqual(@as(i32, 0), c.q3ld_pre_read_lens(fd, size, 201, 0, &none, &l));
    try waitFor(pool, preIdle);

    // A claim of an in-flight record: a deadline inside its first chunk expires the wait; a later one is served.
    const Q: i64 = 30 * pg;
    injectFault(@intCast(Q), 5, hold_ms);
    try testing.expectEqual(@as(u32, 1), try pool.specStep(f.ufd, 2, &.{Q}, rec));
    try waitFor(pool, specStarted(3).f);
    var dst: [200]u8 = @splat(0xAA);
    const q: u64 = @intCast(Q);
    const t0 = c.q3ld_monotonic_ns();
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, size, t0 + 50 * std.time.ns_per_ms, q, q + 200, &.{100}, &.{100}, &dst, 0));
    try pool.wait(0, 2, 10 * std.time.ns_per_s);
    if (pool.result(0).status == .ok) {
        std.debug.print("SPECPROBE skipped: Q landed before its claim's deadline (box loaded)\n", .{});
        return error.SkipZigTest;
    }
    try testing.expectEqual(Status.deadline, pool.result(0).status);
    try testing.expectEqual(Status.deadline, pool.result(1).status);
    try testing.expect(std.mem.allEqual(u8, &dst, 0xAA));
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, size, t0 + 60 * std.time.ns_per_s, q, q + 200, &.{100}, &.{100}, &dst, 2));
    try pool.wait(2, 2, 10 * std.time.ns_per_s);
    for (2..4) |k| {
        try testing.expectEqual(Status.ok, pool.result(@intCast(k)).status);
        try testing.expectEqual(@as(i64, 0), pool.result(@intCast(k)).preadv_calls);
    }
    try testing.expectEqualSlices(u8, f.image[q..][0..100], dst[0..100]);
    try testing.expectEqualSlices(u8, f.image[q + 200 ..][0..100], dst[100..200]);
    try testing.expect(pool.counter(.adopt_waits) >= 1);
    try testing.expectEqual(@as(i64, 1), pool.counter(.claimed_inflight));
}

test "dsv41 io cov: q3ld_stop alone wakes and joins a worker held at a pre-range's bind point" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(64 * page);
    defer f.deinit();
    var pool = try specPool(1, 2);
    defer pool.stop();
    try R96.armPreRead(pool, &spec_lens);
    try testing.expectEqual(@as(u32, 2), try R96.preRead(pool, f.ufd, 100, &.{@intCast(40 * page + 3)}, &spec_lens));
    try waitFor(pool, preStarted1);
    // The header's contract: q3ld_stop drains, joins and returns. Run it on a thread and give it 2 s.
    const Stopper = struct {
        var done = std.atomic.Value(bool).init(false);
        fn run() void {
            _ = c.q3ld_stop();
            done.store(true, .release);
        }
    };
    Stopper.done.store(false, .release);
    const th = try std.Thread.spawn(.{}, Stopper.run, .{});
    var t: u32 = 0;
    while (!Stopper.done.load(.acquire) and t < 2000) : (t += 1) std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
    const hung = !Stopper.done.load(.acquire);
    // Rescue: a quiesce cancels the unbound range at its bind point, so the stop's join returns.
    if (hung) _ = c.q3ld_quiesce(10 * std.time.ns_per_s);
    th.join();
    try testing.expect(!hung);
}

test "dsv41 io cov: a refused pre-read re-arm (a negative length) leaves the armed class as it was" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(16 * page);
    defer f.deinit();
    var pool = try specPool(1, 1);
    defer pool.stop();
    try R96.armPreRead(pool, &spec_lens);
    var l: [n_components]i64 = undefined;
    for (spec_lens, &l) |x, *y| y.* = @intCast(x);
    var neg = l;
    neg[7] = -1;
    try testing.expectEqual(@as(c_int, -1), c.q3ld_pre_config(6, 3, &neg));
    // A refused call leaves the armed class as it was: the stock pre-read's ranges carry the record's geometry, so the
    // demand submit of that record binds them (with the defect they hold no component and never bind).
    const b: u64 = 4 * page + 1;
    try testing.expectEqual(@as(i32, 2), q3raw.q3ld_pre_read(f.ufd.fd, @intCast(f.ufd.size), 1, 1, &[1]i64{@intCast(b)}));
    try waitFor(pool, preStarted1);
    var d = try Dests.init(1, &spec_lens);
    defer testing.allocator.free(d.buf);
    const first = try R96.submit(pool, f.ufd, &.{b}, &.{b + spec_gu_len}, d.rows[0..1], &spec_lens);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try d.expectRecord(0, f.image, b, b + spec_gu_len, &spec_lens);
    try testing.expect(pool.counter(.pre_bound) >= 1);
}

test "dsv41 io cov: a file shorter than its job's size, ending inside a range's page, ends the range short (no retry spin)" {
    const page = std.heap.pageSize();
    // The file ends 300 bytes into its second page; the job carries the four pages it had when opened.
    var f = try PatternFile.init(page + 300);
    defer f.deinit();
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 16 });
    defer pool.stop();
    const fd = std.c.dup(f.ufd.fd);
    try testing.expect(fd >= 0);
    var closed = false;
    defer if (!closed) {
        _ = std.c.close(fd);
    };
    var dst: [510]u8 = @splat(0xAA);
    try testing.expectEqual(@as(c_int, 0), rawSubmit(pool, fd, @intCast(4 * page), -1, page + 100, 0, &.{500}, &.{10}, &dst, 0));
    // 200 bytes exist past the skip: they land, then the end lies inside the same page. The range should be short.
    pool.wait(0, 2, 2 * std.time.ns_per_s) catch |e| {
        // Rescue the spinning worker: its next preadv fails on the closed descriptor.
        _ = std.c.close(fd);
        closed = true;
        try pool.wait(0, 2, 10 * std.time.ns_per_s);
        return e;
    };
    try testing.expectEqual(Status.short, pool.result(0).status);
    try testing.expectEqual(@as(i64, 200), pool.result(0).payload);
}

test "glm io: with direct reads every range lands straight in its rows (aligned or not: the route tests no range, the owner proves alignment at construction); without them the staging loop; every range lands the file's bytes" {
    const page = std.heap.pageSize();
    var f = try PatternFile.init(96 * page);
    defer f.deinit();
    // Whole pages per part (the GLM affine records' geometry), page-aligned records.
    const lens = [n_components]u64{ 3 * page, page, page, 3 * page, page, page, 2 * page, page, page };
    const n = 3;
    var gu: [n]u64 = undefined;
    var down: [n]u64 = undefined;
    for (0..n) |i| {
        gu[i] = (2 + 13 * i) * page;
        down[i] = (60 + 5 * i) * page;
    }
    const Case = struct { direct: bool, skew: usize, want: i64 };
    for ([_]Case{ .{ .direct = true, .skew = 0, .want = 2 * n }, .{ .direct = true, .skew = 64, .want = 2 * n }, .{ .direct = false, .skew = 0, .want = 0 } }) |cs| {
        var pool = try Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = page, .tickets = 64, .direct = cs.direct });
        defer pool.stop();
        var d = try Dests.initPages(n, &lens, cs.skew);
        defer d.deinit();
        const first = try R96.submit(pool, f.ufd, &gu, &down, d.rows[0..n], &lens);
        try pool.wait(first, 2 * n, 10 * std.time.ns_per_s);
        for (0..n) |i| {
            try d.expectRecord(i, f.image, gu[i], down[i], &lens);
            try testing.expectEqual(Status.ok, pool.result(first + @as(u32, @intCast(i))).status);
            try testing.expectEqual(Status.ok, pool.result(first + @as(u32, @intCast(n + i))).status);
        }
        try testing.expectEqual(cs.want, pool.counter(.direct_ranges));
        // A direct gate/up range is one preadv of its six parts.
        if (cs.want > 0) try testing.expectEqual(@as(i64, 1), pool.result(first).preadv_calls);
    }
    // A short read inside a part (the first preadv returns 5 pages and a half): the rest lands after it.
    var pool = try Pool.start(testing.allocator, .{ .workers = 1, .staging_bytes = page, .tickets = 64, .direct = true });
    defer pool.stop();
    defer clearFaults();
    var d = try Dests.initPages(n, &lens, 0);
    defer d.deinit();
    injectFault(gu[1], 4, @intCast(5 * page / 2));
    const first = try R96.submit(pool, f.ufd, &gu, &down, d.rows[0..n], &lens);
    try pool.wait(first, 2 * n, 10 * std.time.ns_per_s);
    for (0..n) |i| try d.expectRecord(i, f.image, gu[i], down[i], &lens);
    try testing.expectEqual(@as(i64, 2 * n), pool.counter(.direct_ranges));
    try testing.expectEqual(@as(i64, 2), pool.result(first + 1).preadv_calls);
}
