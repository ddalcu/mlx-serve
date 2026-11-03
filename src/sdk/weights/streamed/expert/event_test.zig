//! The event gate's tests (`sdk_ext.expert.event`), kept in the package's test root under their names (the 0b ones run
//! on a GPU machine only: DSV41_PHASE0B_MLX=1; creating any MLX array creates the Metal device).

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const expert_bank = @import("bank.zig");
const expert_io = @import("../sdk_ext.zig").expert.io;
const event = @import("../sdk_ext.zig").expert.event;
const c = event.test_abi.abi;
const abi_version = event.abi_version;
const Event = event.Event;
const createHost = event.createHost;
const createMetal = event.createMetal;
const signaledValue = event.signaledValue;
const wait = event.wait;
const signal = event.signal;
const allFresh = event.allFresh;
const stats = event.stats;

const testing = std.testing;
const n_components = expert_bank.n_components;

test "dsv41 event: the shim's ABI" {
    try testing.expectEqual(@as(i32, abi_version), c.dsv41ev_abi());
    var word: i64 = 0;
    try testing.expectError(error.EventUnavailable, createHost(&word, 0));
    const ev = try createHost(&word, std.time.ns_per_s);
    try testing.expect(ev.id >= 1);
    word = 7;
    try testing.expectEqual(@as(u64, 7), signaledValue(ev));
}

test "dsv41 event: a wait's and a signal's outputs must be fresh handles; stack garbage is refused in safe builds" {
    // What an `undefined` [3]mlx_array holds in a safe build (0xaa bytes), and what mlx_array_new returns.
    var outs: [3]mlx.mlx_array = undefined;
    for (&outs) |*o| o.ctx = @ptrFromInt(0xaaaa_aaaa_aaaa_aaa8);
    try testing.expect(!allFresh(&outs));
    for (&outs) |*o| o.* = mlx.mlx_array_new();
    try testing.expect(allFresh(&outs));
    try testing.expect(allFresh(&@as([2]mlx.mlx_array, @splat(.{}))));
}

/// A pattern file and one record's nine destinations inside one uint8 MLX array.
const Rig = struct {
    tmp: std.testing.TmpDir,
    image: []u8,
    ufd: expert_io.UncachedFd,
    arr: mlx.mlx_array,
    rows: [1][n_components]u64,
    const lens = [n_components]u64{ 8000, 100, 200, 8000, 100, 200, 3000, 200, 100 };
    const gu_len = 16600;
    const total = 19900;
    const base = 3 * 16384 + 40;

    fn init(stream: mlx.mlx_stream) !Rig {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const image = try testing.allocator.alloc(u8, 8 * 16384);
        errdefer testing.allocator.free(image);
        expert_bank.fillPattern(image, 5);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "p.bin", .data = image });
        var root: [512]u8 = undefined;
        var pbuf: [600]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/p.bin", .{root[0..try tmp.dir.realPath(std.testing.io, &root)]}, 0);
        const ufd = try expert_io.openUncached(path.ptr, null);
        errdefer ufd.close();
        var arr = mlx.mlx_array_new();
        const shape = [_]c_int{total};
        try mlx.check(mlx.mlx_zeros(&arr, &shape, 1, .uint8, stream));
        try mlx.check(mlx.mlx_array_eval(arr));
        const p = @intFromPtr(mlx.mlx_array_data_uint8(arr) orelse return error.MlxNoData);
        var rig: Rig = .{ .tmp = tmp, .image = image, .ufd = ufd, .arr = arr, .rows = undefined };
        var off: u64 = 0;
        for (lens, 0..) |l, k| {
            rig.rows[0][k] = p + off;
            off += l;
        }
        return rig;
    }

    fn deinit(self: *Rig) void {
        _ = mlx.mlx_array_free(self.arr);
        self.ufd.close();
        testing.allocator.free(self.image);
        self.tmp.cleanup();
    }

    /// The record's read (its gate/up read held `hold_ns`), gated by values 1 (gate/up) and 2 (down).
    fn submitGated(self: *Rig, pool: *expert_io.Pool, hold_ns: i64) !u32 {
        expert_io.injectFault(base / 16384 * 16384, 5, hold_ns);
        const first = try expert_io.Records(n_components, expert_bank.gu_components).submit(pool, self.ufd, &.{base}, &.{base + gu_len}, &self.rows, &lens);
        try pool.registerGates(&.{ 1, 2 }, &.{ 1, 1 }, &.{ first, first + 1 });
        return first;
    }

    /// astype(uint32) of the gated alias, evaluated: ms taken, and whether it equals the file's record.
    fn consume(self: *Rig, ev: Event, stream: mlx.mlx_stream) !struct { ms: i64, equal: bool } {
        var out = [1]mlx.mlx_array{mlx.mlx_array_new()};
        try wait(&.{self.arr}, ev, 2, &.{}, false, stream, &out);
        defer _ = mlx.mlx_array_free(out[0]);
        var wide = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wide);
        try mlx.check(mlx.mlx_astype(&wide, out[0], .uint32, stream));
        const t0 = std.Io.Timestamp.now(std.testing.io, .boot);
        try mlx.check(mlx.mlx_array_eval(wide));
        const ms: i64 = @intCast(@divTrunc(t0.untilNow(std.testing.io, .boot).nanoseconds, std.time.ns_per_ms));
        const got = (mlx.mlx_array_data_uint32(wide) orelse return error.MlxNoData)[0..total];
        const want = self.image[base..][0..total];
        var equal = true;
        for (got, want) |g, w| equal = equal and g == w;
        return .{ .ms = ms, .equal = equal };
    }
};

// DSV41_PHASE0B_MLX=1, on a GPU machine.
test "dsv41 event 0b: a host event holds a CPU-stream consumer until the pool publishes the bytes" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var rig = try Rig.init(stream);
    defer rig.deinit();
    const word = try testing.allocator.create(i64);
    defer testing.allocator.destroy(word);
    word.* = 0;
    var pool = try expert_io.Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = 4 * 16384, .tickets = 64 });
    defer pool.stop();
    defer expert_io.clearFaults();
    try pool.armEvent(.host, @intFromPtr(word), 10 * std.time.ns_per_s, 0);
    const ev = try createHost(word, 20 * std.time.ns_per_s);
    const first = try rig.submitGated(pool, 300 * std.time.ns_per_ms);
    const r = try rig.consume(ev, stream);
    std.debug.print("host event: consumer evaluated after {d} ms, bytes equal {}\n", .{ r.ms, r.equal });
    try testing.expect(r.equal and r.ms >= 250);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try testing.expectEqual(@as(i64, 0), pool.counter(.ev_wd_forced));
}

test "dsv41 event 0b: an MTLSharedEvent holds GPU work until the pool signals it" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var rig = try Rig.init(stream);
    defer rig.deinit();
    const ev = try createMetal();
    var pool = try expert_io.Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = 4 * 16384, .tickets = 64 });
    defer pool.stop();
    defer expert_io.clearFaults();
    try pool.armEvent(.metal, ev.object, 10 * std.time.ns_per_s, 0);
    const waits0 = stats().gpu_waits;
    const first = try rig.submitGated(pool, 300 * std.time.ns_per_ms);
    const r = try rig.consume(ev, stream);
    std.debug.print("metal event: GPU consumer evaluated after {d} ms, bytes equal {}, event value {d}\n", .{ r.ms, r.equal, signaledValue(ev) });
    try testing.expect(r.equal and r.ms >= 250);
    try testing.expectEqual(waits0 + 1, stats().gpu_waits);
    try pool.wait(first, 2, 10 * std.time.ns_per_s);
    try testing.expect(pool.counter(.ev_signals) >= 1);
    try testing.expectEqual(@as(i64, 0), pool.counter(.ev_wd_forced));
    try testing.expectEqual(@as(u64, 2), signaledValue(ev));
}

test "dsv41 event 0b: a gate whose bytes never land is forced, so the GPU never hangs" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var rig = try Rig.init(stream);
    defer rig.deinit();
    const ev = try createMetal();
    var pool = try expert_io.Pool.start(testing.allocator, .{ .workers = 2, .staging_bytes = 4 * 16384, .tickets = 64 });
    defer pool.stop();
    try pool.armEvent(.metal, ev.object, 200 * std.time.ns_per_ms, 0);
    // Two gates over a ticket that never publishes.
    pool.res[60 * 8] = @intFromEnum(expert_io.Status.pending);
    defer pool.res[60 * 8] = 0;
    try pool.registerGates(&.{ 1, 2 }, &.{ 1, 0 }, &.{60});
    const r = try rig.consume(ev, stream);
    std.debug.print("forced gate: GPU consumer released after {d} ms\n", .{r.ms});
    try testing.expect(r.ms >= 150 and r.ms < 5000);
    try testing.expectEqual(@as(i64, 1), pool.counter(.ev_wd_forced));
}

test "dsv41 event: a refused host event names its reason; the shim's counters count every event made (no device)" {
    var word: i64 align(8) = 0;
    const before = stats();
    try testing.expectError(error.EventUnavailable, createHost(&word, -1));
    try testing.expect(std.mem.indexOf(u8, event.lastError(), "positive timeout") != null);
    const ev = try createHost(&word, std.time.ns_per_s);
    const after = stats();
    try testing.expectEqual(before.events + 1, after.events);
    // No wait or signal ran: the device-side counters did not move.
    try testing.expectEqual(before.gpu_waits, after.gpu_waits);
    try testing.expectEqual(before.gpu_signals, after.gpu_signals);
    try testing.expectEqual(@as(u64, 0), signaledValue(ev));
    // An unknown id reads 0.
    try testing.expectEqual(@as(u64, 0), signaledValue(.{ .id = 1 << 30, .object = 0 }));
}
