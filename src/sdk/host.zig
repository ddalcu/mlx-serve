//! How sdk code reaches the `Host` table it was handed (`LoadCtx.host`): `Host.of` checks the table once, and the
//! plugin keeps the value in its own per-load state and passes it down. Nothing here is global: each dylib compiles
//! its own copy of sdk, so anything process-wide is the host's and arrives through the table.

const std = @import("std");
pub const abi = @import("../abi/host.zig");

pub const Host = struct {
    t: *const abi.Host,

    /// The table a plugin was handed, refused by name when absent or older than this sdk's (a newer one is a superset).
    pub fn of(raw: ?*const anyopaque) error{ NoHost, HostTooOld }!Host {
        const t: *const abi.Host = @ptrCast(@alignCast(raw orelse return error.NoHost));
        if (t.size < @sizeOf(abi.Host) or t.gpu.size < @sizeOf(abi.HostGpu)) return error.HostTooOld;
        return .{ .t = t };
    }

    pub fn gpu(self: Host) *const abi.HostGpu {
        return self.t.gpu;
    }

    // ── Allocator ──

    pub fn allocator(self: Host) std.mem.Allocator {
        return .{ .ptr = @constCast(self.t), .vtable = &vtable };
    }

    const vtable: std.mem.Allocator.VTable = .{ .alloc = vAlloc, .resize = vResize, .remap = vRemap, .free = vFree };

    fn table(p: *anyopaque) *const abi.Host {
        return @ptrCast(@alignCast(p));
    }
    fn vAlloc(p: *anyopaque, len: usize, a: std.mem.Alignment, _: usize) ?[*]u8 {
        const t = table(p);
        return t.alloc(t.ctx, len, @backingInt(a));
    }
    fn vResize(p: *anyopaque, m: []u8, a: std.mem.Alignment, new_len: usize, _: usize) bool {
        const t = table(p);
        return t.resize(t.ctx, m.ptr, m.len, @backingInt(a), new_len);
    }
    fn vRemap(p: *anyopaque, m: []u8, a: std.mem.Alignment, new_len: usize, _: usize) ?[*]u8 {
        const t = table(p);
        return t.remap(t.ctx, m.ptr, m.len, @backingInt(a), new_len);
    }
    fn vFree(p: *anyopaque, m: []u8, a: std.mem.Alignment, _: usize) void {
        const t = table(p);
        t.free(t.ctx, m.ptr, m.len, @backingInt(a));
    }

    // ── Log ──

    pub fn logEnabled(self: Host, level: abi.LogLevel) bool {
        return self.t.log_enabled(self.t.ctx, level);
    }

    /// `log.zig`'s shape: the caller writes the newline. Formatted only when the level is on; a line past 4 KiB is cut.
    pub fn log(self: Host, level: abi.LogLevel, comptime fmt: []const u8, args: anytype) void {
        if (!self.logEnabled(level)) return;
        var buf: [4096]u8 = undefined;
        var w = std.Io.Writer.fixed(&buf);
        w.print(fmt, args) catch {};
        self.t.log(self.t.ctx, level, abi.Str.of(w.buffered()));
    }

    // ── MLX error latch ──

    pub fn latchPending(self: Host) bool {
        return self.t.latch_pending(self.t.ctx);
    }

    /// The checkpoint: the same errors as the host's `checkError` (`OutOfMemory` for the memory class).
    pub fn check(self: Host, site: abi.CheckSite) error{ OutOfMemory, MlxFailure }!void {
        return switch (self.t.latch_check(self.t.ctx, site)) {
            .none => {},
            .out_of_memory => error.OutOfMemory,
            .failure => error.MlxFailure,
        };
    }

    pub fn dropLatchedErrorUnless(self: Host, had_error: bool) void {
        self.t.latch_drop_unless(self.t.ctx, had_error);
    }

    pub fn contextLost(self: Host) bool {
        return self.t.gpu_context_lost(self.t.ctx);
    }

    // ── Kernel cache, levers, metrics, memory ──

    /// The cached kernel's handle (the mlx-c kernel's `ctx`), or null when this build cannot make it.
    pub fn kernel(self: Host, desc: *const abi.KernelDesc) ?*anyopaque {
        return self.t.kernel(self.t.ctx, desc);
    }

    pub fn leverRaw(self: Host, name: []const u8) ?[]const u8 {
        const v = self.t.lever(self.t.ctx, abi.Str.of(name));
        return if (v.ptr == null) null else v.slice();
    }

    /// A diagnostic switch: set and not `0`.
    pub fn leverOn(self: Host, name: []const u8) bool {
        const v = self.leverRaw(name) orelse return false;
        return v.len > 0 and v[0] != '0';
    }

    /// Resolve once at load; `add` is one relaxed atomic add, nothing when metrics are off.
    pub fn counter(self: Host, name: []const u8) Counter {
        return .{ .word = self.t.counter(self.t.ctx, abi.Str.of(name)) };
    }

    pub fn memoryCeiling(self: Host) u64 {
        return self.t.memory_ceiling(self.t.ctx);
    }
};

pub const Counter = struct {
    word: ?*u64 = null,

    pub fn add(self: Counter, n: u64) void {
        if (self.word) |w| _ = @atomicRmw(u64, w, .Add, n, .monotonic);
    }
};

/// A kernel's cache digest over its source and header; comptime for a comptime source.
pub fn digest(source: []const u8, header: []const u8) u64 {
    return std.hash.Wyhash.hash(std.hash.Wyhash.hash(0, header), source);
}

const testing = std.testing;
const FakeHost = @import("testing/fake_host.zig").FakeHost;

test "sdk host: a missing or older table is refused by name" {
    try testing.expectError(error.NoHost, Host.of(null));
    var fake: FakeHost = undefined;
    fake.init(.{});
    var old = fake.table;
    old.size -= 8;
    try testing.expectError(error.HostTooOld, Host.of(&old));
    var old_gpu = fake.gpu_table;
    old_gpu.size = 16;
    var t = fake.table;
    t.gpu = &old_gpu;
    try testing.expectError(error.HostTooOld, Host.of(&t));
    _ = try Host.of(fake.host());
}

test "sdk host: a lever is on only when set and not 0; a digest names source and header" {
    var fake: FakeHost = undefined;
    fake.init(.{ .levers = &.{ .{ "A", "1" }, .{ "B", "0" }, .{ "C", "" }, .{ "D", "yes" } } });
    const h = try Host.of(fake.host());
    try testing.expect(h.leverOn("A"));
    try testing.expect(!h.leverOn("B"));
    try testing.expect(!h.leverOn("C"));
    try testing.expect(h.leverOn("D"));
    try testing.expect(!h.leverOn("E"));
    try testing.expectEqualStrings("", h.leverRaw("C").?);
    try testing.expectEqual(@as(?[]const u8, null), h.leverRaw("E"));

    try testing.expect(digest("a", "b") != digest("b", "a"));
    try testing.expect(digest("ab", "") != digest("a", "b"));
    comptime std.debug.assert(digest("x", "") == digest("x", ""));
}

test "sdk host: a log line is formatted only when its level is on, and cut at 4 KiB" {
    var fake: FakeHost = undefined;
    fake.init(.{ .level = .info });
    const h = try Host.of(fake.host());
    h.log(.info, "n={d}\n", .{7});
    h.log(.debug, "{d}\n", .{8});
    try testing.expectEqual(@as(usize, 1), fake.lines);
    try testing.expectEqualStrings("n=7\n", fake.last());
    const long: [5000]u8 = @splat('x');
    h.log(.warn, "{s}", .{long});
    try testing.expectEqual(@as(usize, 4096), fake.last().len);
}
