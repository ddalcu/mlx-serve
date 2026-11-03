//! G7, profile builds: a plugin's profile instruments reach the code of its other kinds as a hook its arch injects at
//! comptime, never as an import, so no kind imports another's profiling. The SDK declares the probes' shapes, the
//! hooks that compile to nothing and the probes' one clock (`Clock`); it carries no timer or counter of its own.
//!
//! The backend type an arch binds its quant and its source with declares the hook (`pub const profile_hook: Hook`);
//! the generic code they share reads it with `of(Backend)`, once, at comptime. A backend that declares none (a host
//! test backend, another arch's) gets `off`, and so does every probe of a build without the plugin's profile options:
//! a served build carries no clock read, counter, field or branch for them.

const std = @import("std");
const mlx = @import("sdk").mlx;

/// Where the prompt pass's routed call spends its host time: the routing barrier (the ids to the host), the source's
/// route (reads issued), the read waits, the waves' encode (graphs, host tables, submission), the drains (the host
/// blocked on GPU compute), the join (the outputs' concatenate / take).
pub const PrefillBucket = enum { barrier, route, read_wait, encode, drain, join };

/// One plugin's probes, each a namespace.
pub const Hook = struct {
    /// The prompt pass's routed-call split: `enabled: bool`, `Stamp: type`, `now() Stamp`,
    /// `charge(PrefillBucket, Stamp) void`, `count(waves: u64, launches: u64) void`, `countCall() void`.
    prefill: type = prefill_off,
    /// A pinned kernel's dispatch, seen by the kernel registry's launcher: `enabled: bool`,
    /// `kernel(name: []const u8, key: u64, inputs: []const mlx.mlx_array) void` (`key`: the launch config's template).
    launch: type = launch_off,
};

/// The hook of a backend that declares none.
pub const off: Hook = .{};

/// A probe file's clock: `Stamp` is the monotonic clock in ns where its probes are compiled in (`enabled`), else
/// `void`, and `now` reads nothing.
pub fn Clock(comptime enabled: bool) type {
    return struct {
        pub const Stamp = if (enabled) u64 else void;

        pub inline fn now() Stamp {
            if (comptime !enabled) return {};
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &ts);
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        }
    };
}

pub const prefill_off = struct {
    pub const enabled = false;
    pub const Stamp = void;
    pub inline fn now() Stamp {}
    pub inline fn charge(_: PrefillBucket, _: Stamp) void {}
    pub inline fn count(_: u64, _: u64) void {}
    pub inline fn countCall() void {}
};

pub const launch_off = struct {
    pub const enabled = false;
    pub fn kernel(_: []const u8, _: u64, _: []const mlx.mlx_array) void {}
};

/// The hook `B` declares (`B.profile_hook`), else its wrapped backend's (`B.Inner`: a profiling wrapper), else `off`.
/// A declared hook whose probes are missing or mistyped is a compile error that names them.
pub fn of(comptime B: type) Hook {
    if (@hasDecl(B, "profile_hook")) {
        const h: Hook = B.profile_hook;
        check(h, @typeName(B));
        return h;
    }
    if (@hasDecl(B, "Inner")) return of(B.Inner);
    return off;
}

fn check(comptime h: Hook, comptime where: []const u8) void {
    const P = h.prefill;
    inline for ([_][]const u8{ "enabled", "Stamp", "now", "charge", "count", "countCall" }) |d| {
        if (!@hasDecl(P, d)) @compileError("profile hook of " ++ where ++ ": the prefill probes lack " ++ d);
    }
    if (@TypeOf(P.enabled) != bool) @compileError("profile hook of " ++ where ++ ": prefill.enabled is not a bool");
    const L = h.launch;
    inline for ([_][]const u8{ "enabled", "kernel" }) |d| {
        if (!@hasDecl(L, d)) @compileError("profile hook of " ++ where ++ ": the launch probes lack " ++ d);
    }
    if (@TypeOf(L.enabled) != bool) @compileError("profile hook of " ++ where ++ ": launch.enabled is not a bool");
    if (L.enabled and @TypeOf(&L.kernel) != *const fn ([]const u8, u64, []const mlx.mlx_array) void)
        @compileError("profile hook of " ++ where ++ ": launch.kernel does not take (name, key: u64, inputs)");
}

const testing = std.testing;

test "sdk profile: a backend without a hook gets off; a declared hook and a wrapper's inner one are found" {
    const Plain = struct {};
    try testing.expect(of(Plain).prefill == prefill_off and of(Plain).launch == launch_off);
    const Timers = struct {
        pub const enabled = true;
        pub const Stamp = u64;
        pub var charged: u64 = 0;
        pub fn now() Stamp {
            return 7;
        }
        pub fn charge(_: PrefillBucket, t: Stamp) void {
            charged += t;
        }
        pub fn count(_: u64, _: u64) void {}
        pub fn countCall() void {}
    };
    const Backend = struct {
        pub const profile_hook: Hook = .{ .prefill = Timers };
    };
    const Wrapper = struct {
        pub const Inner = Backend;
    };
    try testing.expect(of(Backend).prefill == Timers and of(Backend).launch == launch_off);
    try testing.expect(of(Wrapper).prefill == Timers);
    const p = of(Wrapper).prefill;
    p.charge(.drain, p.now());
    try testing.expectEqual(@as(u64, 7), Timers.charged);
    // the off probes compile to nothing
    const o = of(Plain).prefill;
    o.charge(.join, o.now());
    o.count(1, 5);
    o.countCall();
    of(Plain).launch.kernel("k", 0, &.{});
}
