//! A `Host` table for sdk and plugin unit tests: no MLX, no environment, no process state. Its log lines, latch,
//! levers, counters, ceiling and kernel cache live in the `FakeHost` a test owns; `HostGpu` refuses every call.

const std = @import("std");
const abi = @import("../../abi/host.zig");

pub const Options = struct {
    allocator: std.mem.Allocator = std.testing.allocator,
    level: abi.LogLevel = .info,
    levers: []const struct { []const u8, []const u8 } = &.{},
    /// Counter names that resolve; empty = metrics off.
    counters: []const []const u8 = &.{},
    ceiling: u64 = 1 << 30,
};

pub const max_counters = 16;
pub const max_kernels = 16;

pub const FakeHost = struct {
    opts: Options,
    /// Lines that passed the level, and the last one's bytes (cut at the buffer).
    lines: usize = 0,
    last_buf: [4096]u8 = undefined,
    last_len: usize = 0,
    /// Set it to simulate an MLX failure; a check consumes it.
    latch: abi.Latched = .none,
    context_lost: bool = false,
    counter_words: [max_counters]u64 = @splat(0),
    /// One (backend, name hash, digest) per kernel made; a handle points at its slot.
    kernels: [max_kernels]u64 = undefined,
    kernels_made: usize = 0,
    table: abi.Host,
    gpu_table: abi.HostGpu,

    pub fn init(self: *FakeHost, opts: Options) void {
        std.debug.assert(opts.counters.len <= max_counters);
        self.* = .{ .opts = opts, .table = undefined, .gpu_table = refusing_gpu };
        self.table = .{
            .ctx = self,
            .alloc = alloc,
            .resize = resize,
            .remap = remap,
            .free = free,
            .log_enabled = logEnabled,
            .log = logLine,
            .latch_pending = latchPending,
            .latch_check = latchCheck,
            .latch_drop_unless = latchDropUnless,
            .gpu_context_lost = gpuContextLost,
            .kernel = kernel,
            .lever = lever,
            .counter = counter,
            .memory_ceiling = memoryCeiling,
            .gpu = &self.gpu_table,
        };
    }

    pub fn host(self: *const FakeHost) *const abi.Host {
        return &self.table;
    }

    pub fn last(self: *const FakeHost) []const u8 {
        return self.last_buf[0..self.last_len];
    }

    pub fn count(self: *const FakeHost, name: []const u8) u64 {
        for (self.opts.counters, 0..) |n, i| if (std.mem.eql(u8, n, name)) return self.counter_words[i];
        return 0;
    }
};

fn self_(ctx: ?*anyopaque) *FakeHost {
    return @ptrCast(@alignCast(ctx.?));
}
fn alignment(align_log2: u8) std.mem.Alignment {
    return @fromBackingInt(@intCast(@as(std.math.Log2Int(usize), @intCast(align_log2))));
}
fn alloc(ctx: ?*anyopaque, len: usize, a: u8) callconv(.c) ?[*]u8 {
    return self_(ctx).opts.allocator.rawAlloc(len, alignment(a), 0);
}
fn resize(ctx: ?*anyopaque, p: [*]u8, len: usize, a: u8, new_len: usize) callconv(.c) bool {
    return self_(ctx).opts.allocator.rawResize(p[0..len], alignment(a), new_len, 0);
}
fn remap(ctx: ?*anyopaque, p: [*]u8, len: usize, a: u8, new_len: usize) callconv(.c) ?[*]u8 {
    return self_(ctx).opts.allocator.rawRemap(p[0..len], alignment(a), new_len, 0);
}
fn free(ctx: ?*anyopaque, p: [*]u8, len: usize, a: u8) callconv(.c) void {
    self_(ctx).opts.allocator.rawFree(p[0..len], alignment(a), 0);
}

fn logEnabled(ctx: ?*anyopaque, l: abi.LogLevel) callconv(.c) bool {
    return @backingInt(l) <= @backingInt(self_(ctx).opts.level);
}
fn logLine(ctx: ?*anyopaque, l: abi.LogLevel, line: abi.Str) callconv(.c) void {
    const self = self_(ctx);
    if (!logEnabled(ctx, l)) return;
    const s = line.slice();
    self.last_len = @min(s.len, self.last_buf.len);
    @memcpy(self.last_buf[0..self.last_len], s[0..self.last_len]);
    self.lines += 1;
}

fn latchPending(ctx: ?*anyopaque) callconv(.c) bool {
    return self_(ctx).latch != .none;
}
fn latchCheck(ctx: ?*anyopaque, _: abi.CheckSite) callconv(.c) abi.Latched {
    const self = self_(ctx);
    defer self.latch = .none;
    return self.latch;
}
fn latchDropUnless(ctx: ?*anyopaque, had_error: bool) callconv(.c) void {
    if (!had_error) self_(ctx).latch = .none;
}
fn gpuContextLost(ctx: ?*anyopaque) callconv(.c) bool {
    return self_(ctx).context_lost;
}

fn kernel(ctx: ?*anyopaque, desc: *const abi.KernelDesc) callconv(.c) ?*anyopaque {
    const self = self_(ctx);
    const key = std.hash.Wyhash.hash(desc.digest +% @backingInt(desc.backend), desc.name.slice());
    for (self.kernels[0..self.kernels_made]) |*k| if (k.* == key) return k;
    if (self.kernels_made == max_kernels) return null;
    self.kernels[self.kernels_made] = key;
    self.kernels_made += 1;
    return &self.kernels[self.kernels_made - 1];
}

fn lever(ctx: ?*anyopaque, name: abi.Str) callconv(.c) abi.Str {
    for (self_(ctx).opts.levers) |kv| if (std.mem.eql(u8, kv[0], name.slice())) return .{ .ptr = kv[1].ptr, .len = kv[1].len };
    return .{};
}

fn counter(ctx: ?*anyopaque, name: abi.Str) callconv(.c) ?*u64 {
    const self = self_(ctx);
    for (self.opts.counters, 0..) |n, i| if (std.mem.eql(u8, n, name.slice())) return &self.counter_words[i];
    return null;
}

fn memoryCeiling(ctx: ?*anyopaque) callconv(.c) u64 {
    return self_(ctx).opts.ceiling;
}

const refused = abi.Verdict.refuse(.unsupported, "the fake host has no GPU");

const refusing_gpu: abi.HostGpu = .{
    .ctx = null,
    .event_metal = struct {
        fn f(_: ?*anyopaque, _: u64, _: *abi.Event) callconv(.c) abi.Verdict {
            return refused;
        }
    }.f,
    .event_host = struct {
        fn f(_: ?*anyopaque, _: *i64, _: i64, _: *abi.Event) callconv(.c) abi.Verdict {
            return refused;
        }
    }.f,
    .wait = struct {
        fn f(_: ?*anyopaque, _: [*]const abi.Array, _: usize, _: i32, _: u64, _: ?[*]const abi.Array, _: usize, _: bool, _: abi.Stream, _: [*]abi.Array) callconv(.c) abi.Verdict {
            return refused;
        }
    }.f,
    .signal = struct {
        fn f(_: ?*anyopaque, _: [*]const abi.Array, _: usize, _: i32, _: u64, _: abi.Stream, _: [*]abi.Array) callconv(.c) abi.Verdict {
            return refused;
        }
    }.f,
    .event_set = struct {
        fn f(_: ?*anyopaque, _: abi.Event, _: u64) callconv(.c) void {}
    }.f,
    .event_value = struct {
        fn f(_: ?*anyopaque, _: i32) callconv(.c) u64 {
            return 0;
        }
    }.f,
    .event_stats = struct {
        fn f(_: ?*anyopaque, out: *abi.EventStats) callconv(.c) void {
            out.* = .{};
        }
    }.f,
    .alloc_uninit = struct {
        fn f(_: ?*anyopaque, _: [*]const i32, _: usize, _: i32, _: *abi.Array) callconv(.c) abi.Verdict {
            return refused;
        }
    }.f,
    .launch = struct {
        fn f(_: ?*anyopaque, _: *abi.VectorArray, _: ?*anyopaque, _: abi.VectorArray, _: abi.KernelConfig, _: abi.Stream) callconv(.c) abi.Verdict {
            return refused;
        }
    }.f,
    .buffer_addresses = struct {
        fn f(_: ?*anyopaque, _: [*]const abi.Array, _: usize, _: [*]u64) callconv(.c) abi.Verdict {
            return refused;
        }
    }.f,
};

const testing = std.testing;
const sdk = @import("../host.zig");

test "fake host: every member works without MLX through the sdk accessor" {
    var fake: FakeHost = undefined;
    fake.init(.{ .counters = &.{"steps"}, .ceiling = 99 });
    const h = try sdk.Host.of(fake.host());

    const buf = try h.allocator().alloc(u8, 32);
    h.allocator().free(buf);

    try testing.expect(!h.latchPending());
    fake.latch = .out_of_memory;
    try testing.expectError(error.OutOfMemory, h.check(.decode_step));
    fake.latch = .failure;
    h.dropLatchedErrorUnless(true);
    try testing.expect(h.latchPending());
    h.dropLatchedErrorUnless(false);
    try testing.expect(!h.latchPending());

    h.counter("steps").add(2);
    h.counter("other").add(5);
    try testing.expectEqual(@as(u64, 2), fake.count("steps"));
    try testing.expectEqual(@as(u64, 99), h.memoryCeiling());

    var desc: abi.KernelDesc = .{ .digest = 1, .name = abi.Str.of("k"), .source = abi.Str.of("s"), .inputs = .{}, .outputs = .{} };
    const k = h.kernel(&desc).?;
    try testing.expectEqual(k, h.kernel(&desc).?);
    desc.digest = 2;
    try testing.expect(h.kernel(&desc).? != k);
    try testing.expectEqual(@as(usize, 2), fake.kernels_made);

    var ev: abi.Event = .{};
    var word: i64 = 0;
    try testing.expectEqual(abi.Verdict.Code.unsupported, h.gpu().event_host(null, &word, 1, &ev).code);
}
