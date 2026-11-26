//! The toolchain's process spawn on macOS: a child holds only the descriptors its spawner named
//! (`patches/zig-0.17.0-darwin-spawn-cloexec-default.patch`, applied by `scripts/fetch-zig.sh`). macOS has no pipe2,
//! so Zig makes a child's pipes with pipe() and then sets CLOEXEC with fcntl(); a child spawned by another thread in
//! between inherits them. The parent then reads no end-of-file on the first child's stdout until the second child
//! exits, and the build runner fails a test shard that had finished ("test runner failed to respond"), dropping its
//! results. The first test fails on the plain 0.17.0 release every run; the third drives the real concurrent spawn
//! path, where the race itself hits only now and then.

const std = @import("std");
const builtin = @import("builtin");

const testing = std.testing;
const Io = std.Io;

fn nowMs() i64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
    return @as(i64, @intCast(ts.sec)) * std.time.ms_per_s + @divTrunc(@as(i64, @intCast(ts.nsec)), std.time.ns_per_ms);
}

/// Reads and drops what arrives on `fd` until end-of-file; false when `ms` pass first (a writer is still open).
fn eofWithin(fd: std.c.fd_t, ms: i64) bool {
    var buf: [256]u8 = undefined;
    const deadline = nowMs() + ms;
    while (true) {
        const left = deadline - nowMs();
        if (left <= 0) return false;
        var pfd = [1]std.c.pollfd{.{ .fd = fd, .events = std.c.POLL.IN, .revents = 0 }};
        const ready = std.c.poll(&pfd, 1, @intCast(left));
        if (ready < 0) continue;
        if (ready == 0) return false;
        const n = std.c.read(fd, &buf, buf.len);
        if (n == 0) return true;
    }
}

/// A child that lives long and names no descriptor of its own: stdio all on /dev/null.
fn spawnHolder(io: Io) !std.process.Child {
    return std.process.spawn(io, .{ .argv = &.{ "/bin/sleep", "60" }, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore });
}

test "spawn: a child does not inherit a descriptor its spawner holds without CLOEXEC" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const io = testing.io;
    // A pipe as another thread's spawn leaves it between pipe() and fcntl(FD_CLOEXEC).
    var fds: [2]std.c.fd_t = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.pipe(&fds));
    defer _ = std.c.close(fds[0]);
    var write_open = true;
    defer if (write_open) {
        _ = std.c.close(fds[1]);
    };
    var holder = try spawnHolder(io);
    defer holder.kill(io);
    _ = std.c.close(fds[1]);
    write_open = false;
    // The child named no descriptor: with the write end closed here, the reader sees end-of-file at once.
    try testing.expect(eofWithin(fds[0], 2000));
}

test "spawn: a child keeps inherited stdio and the descriptors in inherit_files" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const io = testing.io;
    var fds: [2]std.c.fd_t = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.pipe(&fds));
    defer _ = std.c.close(fds[0]);
    _ = std.c.fcntl(fds[0], std.c.F.SETFD, @as(c_int, std.c.FD_CLOEXEC));
    _ = std.c.fcntl(fds[1], std.c.F.SETFD, @as(c_int, std.c.FD_CLOEXEC));
    var script_buf: [128]u8 = undefined;
    // `exec 9<&0` fails when the child has no stdin; the echo then goes to the inherited descriptor by its number.
    const script = try std.fmt.bufPrint(&script_buf, "exec 9<&0 && echo stdin-open >&{d}", .{fds[1]});
    const out: Io.File = .{ .handle = fds[1], .flags = .{ .nonblocking = false } };
    var child = std.process.spawn(io, .{
        .argv = &.{ "/bin/sh", "-c", script },
        .stdin = .inherit,
        .stdout = .ignore,
        .stderr = .ignore,
        .inherit_files = &.{out},
    }) catch |e| {
        _ = std.c.close(fds[1]);
        return e;
    };
    _ = std.c.close(fds[1]);
    const term = try child.wait(io);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, term);
    var buf: [64]u8 = undefined;
    const n = std.c.read(fds[0], &buf, buf.len);
    try testing.expect(n > 0);
    try testing.expectEqualStrings("stdin-open\n", buf[0..@intCast(n)]);
}

test "spawn: beside children spawned at the same time that live on, a child's stdout ends when it exits" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const short_threads = 8;
    const short_each = 48;
    const holder_threads = 2;
    const holder_count = 96;
    const Burst = struct {
        io: Io,
        go: std.atomic.Value(bool) = .init(false),
        leaked: std.atomic.Value(u32) = .init(0),
        spawn_failed: std.atomic.Value(u32) = .init(0),
        holders: [holder_count]?std.process.Child = @splat(null),

        fn waitGo(b: *@This()) void {
            while (!b.go.load(.acquire)) std.atomic.spinLoopHint();
        }

        /// Short children with a stdout pipe each: its end-of-file must come within 2 s of the spawn.
        fn short(b: *@This()) void {
            b.waitGo();
            for (0..short_each) |_| {
                var child = std.process.spawn(b.io, .{ .argv = &.{"/usr/bin/true"}, .stdin = .ignore, .stdout = .pipe, .stderr = .ignore }) catch {
                    _ = b.spawn_failed.fetchAdd(1, .monotonic);
                    continue;
                };
                if (!eofWithin(child.stdout.?.handle, 2000)) _ = b.leaked.fetchAdd(1, .monotonic);
                _ = child.wait(b.io) catch {};
            }
        }

        /// Long-lived children spawned through the same burst.
        fn holder(b: *@This(), first: usize) void {
            b.waitGo();
            var i = first;
            while (i < holder_count) : (i += holder_threads) {
                b.holders[i] = spawnHolder(b.io) catch {
                    _ = b.spawn_failed.fetchAdd(1, .monotonic);
                    continue;
                };
            }
        }
    };
    var burst: Burst = .{ .io = testing.io };
    defer for (&burst.holders) |*h| if (h.*) |*c| c.kill(testing.io);
    var threads: [short_threads + holder_threads]std.Thread = undefined;
    for (threads[0..short_threads]) |*t| t.* = try std.Thread.spawn(.{}, Burst.short, .{&burst});
    for (threads[short_threads..], 0..) |*t, k| t.* = try std.Thread.spawn(.{}, Burst.holder, .{ &burst, k });
    burst.go.store(true, .release);
    for (threads) |t| t.join();
    try testing.expectEqual(@as(u32, 0), burst.spawn_failed.load(.monotonic));
    const leaked = burst.leaked.load(.monotonic);
    if (leaked != 0) {
        std.debug.print("spawn burst: {d} of {d} short children's stdout stayed open past their exit (a sibling held it)\n", .{ leaked, short_threads * short_each });
        return error.TestUnexpectedResult;
    }
}
