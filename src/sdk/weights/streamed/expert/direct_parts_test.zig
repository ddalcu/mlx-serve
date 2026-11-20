//! The direct parts route's byte gate: the read pool (its C alone, no MLX, no stream) reads records laid out as the pack lays them, K = 3 and K = 2 (hidden 5,120,
//! intermediate 2,304, each record padded to 4,096 B; two layers of two experts, two records a submit), through the
//! PARTS route and through the staging route. Both land the file's bytes in every slot row with the same status words;
//! PARTS' counters are exactly the gate codes (the only parts on the block). A row off the block stages its part, still
//! the file's bytes; a file that ends inside a record ends both routes' ranges short with the same payloads.

const std = @import("std");
const testing = std.testing;

extern fn q3ld_spec_config(nthreads: i32, bufs: ?[*]const u64, nslots: i32, slot_bytes: i64, rec_len: i64, chunk: i64, counters: ?[*]i64) c_int;
extern fn q3ld_direct_config(on: i32) c_int;
extern fn q3ld_parts_config(block: i64) c_int;
extern fn q3ld_parts_live(on: i32) c_int;
extern fn q3ld_start(nw: i32, staging_ptrs: [*]const u64, sbytes: i64, psize: i64, res: [*]i64, n_tickets: i64, log: [*]i64, n_log: i64, gauge: *[6]i64) c_int;
extern fn q3ld_submit(fd: i32, file_size: i64, deadline: i64, n: i32, ngu: i32, ndown: i32, offsets: [*]const i64, rows: [*]const [*]const u64, lens: [*]const i64, first: i64) c_int;
extern fn q3ld_seq() i64;
extern fn q3ld_wait(seen: i64, timeout_ns: i64) i64;
extern fn q3ld_quiesce(timeout_ns: i64) c_int;
extern fn q3ld_stop() c_int;
extern fn q3ld_abi() i32;
extern fn q3ld_counters_n() i32;


const abi = 2026110203;
const counters_n = 82;
/// The pool's SC_PARTS_DIRECT, SC_PARTS_DIRECT_BYTES, SC_PARTS_STAGED_BYTES.
const sc_parts = 72;
const res_w = 10;
const st_ok = 0;
const st_short = 1;
const st_pending = -1;

const hidden: u64 = 5120;
const inter: u64 = 2304;
const block: u64 = 4096;
const n_comp = 9;
const ngu = 6;
const experts = 2;
const workers = 2;
const staging_bytes: u64 = 9 << 20;
const tickets = 16;

/// The bank's record geometry at K (expert/bank.zig layerSegments): per projection code, rout, rin.
const Geom = struct {
    off: [n_comp]u64,
    len: [n_comp]u64,
    logical: u64,
    record: u64,
    fn of(k: u64) Geom {
        var g: Geom = .{ .off = undefined, .len = undefined, .logical = 0, .record = 0 };
        var at: u64 = 0;
        for (0..3) |p| {
            const in: u64 = if (p == 2) inter else hidden;
            const out: u64 = if (p == 2) hidden else inter;
            const parts = [3]u64{ (in / 16) * (out / 16) * 16 * k * 2, out * 2, in * 2 };
            for (parts, 0..) |l, j| {
                g.off[p * 3 + j] = at;
                g.len[p * 3 + j] = l;
                at += l;
            }
        }
        g.logical = at;
        g.record = std.mem.alignForward(u64, at, block);
        return g;
    }
    fn guLen(g: Geom) u64 {
        return g.off[ngu];
    }
};

const geoms = [2]Geom{ Geom.of(3), Geom.of(2) };

fn layerBase(layer: usize) u64 {
    return if (layer == 0) 0 else experts * geoms[0].record;
}

fn recordOffset(layer: usize, e: usize) u64 {
    return layerBase(layer) + e * geoms[layer].record;
}

const page = std.heap.page_size_min;
const Map = []align(page) u8;

fn mapBytes(n: usize) !Map {
    return std.posix.mmap(null, std.mem.alignForward(usize, n + page, page), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
}

const TempFile = struct {
    path: [:0]u8,
    fd: c_int,
    size: u64,
    fn create(a: std.mem.Allocator, tag: []const u8, bytes: []const u8) !TempFile {
        const path = try std.fmt.allocPrintSentinel(a, "/tmp/turboexl3_ds41-direct-parts-{s}-{d}.bin", .{ tag, std.c.getpid() }, 0);
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
        // The reader's descriptor: read-only, the page cache bypassed, no read-ahead (expert/io.zig openUncached).
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.TempFile;
        if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0 or std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0)) != 0) {
            _ = std.c.close(fd);
            return error.TempFile;
        }
        return .{ .path = path, .fd = fd, .size = bytes.len };
    }
    fn destroy(f: TempFile, a: std.mem.Allocator) void {
        _ = std.c.close(f.fd);
        _ = std.c.unlink(f.path);
        a.free(f.path);
    }
};

const Route = enum { staged, parts };

/// One pool run's results: per ticket (layer l's gate/up of expert e = 4l + e, its down 4l + 2 + e) the status and the
/// payload, and the PARTS counters.
const Run = struct {
    status: [2 * 2 * experts]i64 = @splat(0),
    payload: [2 * 2 * experts]i64 = @splat(0),
    parts: [3]i64 = @splat(0),
};

/// Each record's rows: one map a component, the row at `shift` bytes into it (0: on the block).
const Rows = struct {
    maps: [2][experts][n_comp]Map = undefined,
    ptr: [2][experts][n_comp]u64 = undefined,
    fn init(shift_gate_code: usize) !Rows {
        var r: Rows = .{};
        var made: usize = 0;
        errdefer for (0..made) |i| std.posix.munmap(r.maps[i / (experts * n_comp)][(i / n_comp) % experts][i % n_comp]);
        for (0..2) |l| for (0..experts) |e| for (0..n_comp) |c| {
            const m = try mapBytes(geoms[l].len[c]);
            r.maps[l][e][c] = m;
            made += 1;
            const shift: usize = if (c == 0) shift_gate_code else 0;
            r.ptr[l][e][c] = @intFromPtr(m.ptr) + shift;
        };
        return r;
    }
    fn deinit(r: *Rows) void {
        for (0..2) |l| for (0..experts) |e| for (0..n_comp) |c| std.posix.munmap(r.maps[l][e][c]);
    }
    fn row(r: *const Rows, l: usize, e: usize, c: usize) []const u8 {
        const p: [*]const u8 = @ptrFromInt(r.ptr[l][e][c]);
        return p[0..geoms[l].len[c]];
    }
};

fn waitAll(res: []const i64, first: usize, count: usize) !void {
    var seen = q3ld_seq();
    var rounds: u32 = 0;
    while (true) {
        var done = true;
        for (first..first + count) |t| {
            if (@atomicLoad(i64, &res[t * res_w], .acquire) == st_pending) done = false;
        }
        if (done) return;
        rounds += 1;
        if (rounds > 300) return error.PoolTimeout;
        seen = q3ld_wait(seen, 100 * std.time.ns_per_ms);
    }
}

/// The pool over `file` through `route`, the layers in `layers` (one submit a layer, its two experts, each landed before
/// the next; `live[i]` the PARTS phase switch for layer i's, all on when null), into `rows`.
fn run(file: TempFile, route: Route, layers: []const usize, rows: *const Rows) !Run {
    return runLive(file, route, layers, null, rows);
}

fn runLive(file: TempFile, route: Route, layers: []const usize, live: ?[]const bool, rows: *const Rows) !Run {
    var counters: [counters_n]i64 = @splat(0);
    var res: [tickets * res_w]i64 = @splat(0);
    var log: [tickets]i64 = @splat(0);
    var gauge: [6]i64 = @splat(0);
    const stage = try std.posix.mmap(null, workers * staging_bytes, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(stage);
    var ptrs: [workers]u64 = undefined;
    for (&ptrs, 0..) |*p, w| p.* = @intFromPtr(stage.ptr) + w * staging_bytes;
    try testing.expectEqual(@as(c_int, 0), q3ld_spec_config(0, null, 0, 0, 0, 0, &counters));
    try testing.expectEqual(@as(c_int, 0), q3ld_direct_config(0));
    try testing.expectEqual(@as(c_int, 0), q3ld_parts_config(if (route == .parts) @intCast(block) else 0));
    try testing.expectEqual(@as(c_int, 0), q3ld_start(workers, &ptrs, staging_bytes, @intCast(std.heap.pageSize()), &res, tickets, &log, tickets, &gauge));
    var stopped = false;
    defer if (!stopped) {
        _ = q3ld_quiesce(10 * std.time.ns_per_s);
        _ = q3ld_stop();
    };
    for (layers, 0..) |l, li| {
        if (live) |lv| try testing.expectEqual(@as(c_int, 0), q3ld_parts_live(@intFromBool(lv[li])));
        const g = geoms[l];
        var offsets: [2 * experts]i64 = undefined;
        var item: [experts][*]const u64 = undefined;
        for (0..experts) |e| {
            offsets[e] = @intCast(recordOffset(l, e) + g.off[0]);
            offsets[experts + e] = @intCast(recordOffset(l, e) + g.off[ngu]);
            item[e] = &rows.ptr[l][e];
        }
        var lens: [n_comp]i64 = undefined;
        for (&lens, g.len) |*d, s| d.* = @intCast(s);
        try testing.expectEqual(@as(c_int, 0), q3ld_submit(file.fd, @intCast(file.size), -1, experts, ngu, n_comp - ngu, &offsets, &item, &lens, @intCast(4 * l)));
        try waitAll(&res, 4 * l, 2 * experts);
    }
    try testing.expectEqual(@as(c_int, 0), q3ld_quiesce(10 * std.time.ns_per_s));
    var out: Run = .{};
    for (layers) |l| for (0..2 * experts) |i| {
        out.status[4 * l + i] = res[(4 * l + i) * res_w];
        out.payload[4 * l + i] = res[(4 * l + i) * res_w + 3];
    };
    out.parts = .{ counters[sc_parts], counters[sc_parts + 1], counters[sc_parts + 2] };
    try testing.expectEqual(@as(c_int, 0), q3ld_stop());
    stopped = true;
    return out;
}

/// Every row of the layers' records holds the file's bytes (up to `limit`, the file's end).
fn expectRowsHoldFile(rows: *const Rows, bytes: []const u8, layers: []const usize) !void {
    for (layers) |l| for (0..experts) |e| for (0..n_comp) |c| {
        const at = recordOffset(l, e) + geoms[l].off[c];
        const want = bytes[@min(at, bytes.len)..@min(at + geoms[l].len[c], bytes.len)];
        try testing.expectEqualSlices(u8, want, rows.row(l, e, c)[0..want.len]);
    };
}

fn fileBytes(a: std.mem.Allocator, n: usize, seed: u64) ![]u8 {
    const b = try a.alloc(u8, n);
    var prng = std.Random.DefaultPrng.init(seed);
    prng.random().bytes(b);
    return b;
}

test "direct parts: the pool's ABI and the route's setting (0 or a power of two from 512, on a stopped pool)" {
    try testing.expectEqual(@as(i32, abi), q3ld_abi());
    try testing.expectEqual(@as(i32, counters_n), q3ld_counters_n());
    for ([_]i64{ -1, 1, 256, 3000, 4097 }) |b| try testing.expectEqual(@as(c_int, -1), q3ld_parts_config(b));
    for ([_]i64{ 4096, 512, 16384, 0 }) |b| try testing.expectEqual(@as(c_int, 0), q3ld_parts_config(b));
    // The pack's geometry: only the gate's code lies on the block.
    for (geoms) |g| {
        try testing.expectEqual(@as(u64, 0), g.len[0] % block);
        try testing.expect(g.off[3] % block != 0 and g.off[6] % block != 0);
        try testing.expect(g.guLen() <= staging_bytes and g.logical - g.guLen() <= staging_bytes);
    }
    try testing.expectEqual(@as(u64, 4_423_680), geoms[0].len[0]);
    try testing.expectEqual(@as(u64, 13_316_096), geoms[0].record);
    try testing.expectEqual(@as(u64, 8_891_904), geoms[1].logical);
}

test "direct parts: the same slot bytes and status words read direct or staged; the counters exactly the gate codes" {
    const a = testing.allocator;
    const n = experts * (geoms[0].record + geoms[1].record);
    const bytes = try fileBytes(a, n, 0x5eed_d1_7ec7);
    defer a.free(bytes);
    const file = try TempFile.create(a, "full", bytes);
    defer file.destroy(a);
    const layers = [_]usize{ 0, 1 };
    var gate_code: i64 = 0;
    var total: i64 = 0;
    for (0..2) |l| {
        gate_code += @intCast(experts * geoms[l].len[0]);
        total += @intCast(experts * geoms[l].logical);
    }
    var runs: [3]Run = undefined;
    for (&runs, [_]Route{ .parts, .staged, .parts }, [_]usize{ 0, 0, 512 }) |*r, route, shift| {
        var rows = try Rows.init(shift);
        defer rows.deinit();
        r.* = try run(file, route, &layers, &rows);
        try expectRowsHoldFile(&rows, bytes, &layers);
        for (0..2) |l| for (0..experts) |e| {
            try testing.expectEqual(@as(i64, st_ok), r.status[4 * l + e]);
            try testing.expectEqual(@as(i64, st_ok), r.status[4 * l + experts + e]);
            try testing.expectEqual(@as(i64, @intCast(geoms[l].guLen())), r.payload[4 * l + e]);
            try testing.expectEqual(@as(i64, @intCast(geoms[l].logical - geoms[l].guLen())), r.payload[4 * l + experts + e]);
        };
    }
    // PARTS on rows on the block: every gate code direct (4 parts), the rest staged.
    try testing.expectEqual([3]i64{ 2 * experts, gate_code, total - gate_code }, runs[0].parts);
    // The staging route counts none.
    try testing.expectEqual([3]i64{ 0, 0, 0 }, runs[1].parts);
    // A gate code row 512 B off the block: that part stages too.
    try testing.expectEqual([3]i64{ 0, 0, total }, runs[2].parts);
}

test "direct parts: a file that ends inside a record ends both routes' ranges short with the same payloads" {
    const a = testing.allocator;
    const cut: usize = @intCast(geoms[0].record + 1_000_000);
    const bytes = try fileBytes(a, cut, 0x0dd_c0de);
    defer a.free(bytes);
    const file = try TempFile.create(a, "short", bytes);
    defer file.destroy(a);
    const layers = [_]usize{0};
    var runs: [2]Run = undefined;
    for (&runs, [_]Route{ .parts, .staged }) |*r, route| {
        var rows = try Rows.init(0);
        defer rows.deinit();
        r.* = try run(file, route, &layers, &rows);
        try expectRowsHoldFile(&rows, bytes, &layers);
    }
    // Expert 0 whole; expert 1's gate/up span ends at the file's end, its down span starts past it.
    const want_status = [4]i64{ st_ok, st_short, st_ok, st_short };
    const want_payload = [4]i64{ @intCast(geoms[0].guLen()), 1_000_000, @intCast(geoms[0].logical - geoms[0].guLen()), 0 };
    for (runs) |r| {
        try testing.expectEqualSlices(i64, &want_status, r.status[0..4]);
        try testing.expectEqualSlices(i64, &want_payload, r.payload[0..4]);
    }
}

test "direct parts: the phase switch: off, a running pool's reads stage (none counted); on again, they read direct" {
    const a = testing.allocator;
    const n = experts * (geoms[0].record + geoms[1].record);
    const bytes = try fileBytes(a, n, 0x7ab_5e7);
    defer a.free(bytes);
    const file = try TempFile.create(a, "live", bytes);
    defer file.destroy(a);
    const layers = [_]usize{ 0, 1 };
    var rows = try Rows.init(0);
    defer rows.deinit();
    // Layer 0 with the switch off (the decode phase's), layer 1 with it on (the prompt's).
    const r = try runLive(file, .parts, &layers, &.{ false, true }, &rows);
    try expectRowsHoldFile(&rows, bytes, &layers);
    for (0..2 * 2 * experts) |t| try testing.expectEqual(@as(i64, st_ok), r.status[t]);
    const g1: i64 = @intCast(experts * geoms[1].len[0]);
    const t1: i64 = @intCast(experts * geoms[1].logical);
    try testing.expectEqual([3]i64{ experts, g1, t1 - g1 }, r.parts);
    // The stop leaves the switch on for the next pool.
    var rows2 = try Rows.init(0);
    defer rows2.deinit();
    const r2 = try run(file, .parts, &.{1}, &rows2);
    try testing.expectEqual([3]i64{ experts, g1, t1 - g1 }, r2.parts);
}
