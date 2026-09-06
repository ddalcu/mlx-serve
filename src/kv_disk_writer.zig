//! SSD-first background writer. The inference thread keeps the device->host readback and
//! hands one writer thread a host byte buffer per file; no mlx handle crosses. Files land
//! `tmp` + `rename`, FIFO, so an entry's `meta.json` (enqueued last) is the last to land.
//! A host-byte permit blocks `submit` past ~1 GiB unwritten; an epoch fence drops staged
//! bytes for a directory about to be removed. POSIX syscalls: this runs off the main thread.

const std = @import("std");
const log = @import("log.zig");

/// One staged file; both buffers are owned by the queue once `submit` accepts them.
pub const Blob = struct {
    path: []u8,
    bytes: []u8,
    epoch: u64,
};

pub const DEFAULT_PERMIT_BYTES: u64 = 1024 * 1024 * 1024;

/// One blob the writer could not write; the tier invalidates the entry it belonged to.
/// `path` is owned by whoever takes it out of `takeFailures`.
pub const Failure = struct { path: []u8, err_name: []const u8 };

/// Where an injected failure strikes. See `Writer.fail_at`.
pub const FailAt = enum { write, submit };

/// Failed paths kept for attribution; past this `unattributed_failure` is raised.
pub const MAX_RECORDED_FAILURES: usize = 256;

pub const Writer = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    work: std.Io.Condition = .init,
    done: std.Io.Condition = .init,
    queue: std.ArrayList(Blob) = .empty,
    pending_bytes: u64 = 0,
    inflight_bytes: u64 = 0,
    /// Valid only while `inflight_bytes > 0`; the blob owns the memory.
    inflight_path: ?[]const u8 = null,
    permit_bytes: u64 = DEFAULT_PERMIT_BYTES,
    epoch: u64 = 1,
    running: bool = false,
    /// Test-only: hold the queue so submission order can be inspected.
    paused: bool = false,
    deinited: bool = false,
    thread: ?std.Thread = null,
    /// Failed blobs since the last `takeFailures`.
    failures: std.ArrayList(Failure) = .empty,
    /// A failure that could not be recorded; read-and-clear via `takeUnattributed`.
    unattributed_failure: bool = false,
    /// Test-only: every blob whose path contains this substring fails like a full volume. Owned.
    fail_substr: ?[]u8 = null,
    fail_at: FailAt = .write,
    /// Diagnostics / test bars. Written under the mutex.
    files_written: u64 = 0,
    bytes_written: u64 = 0,
    files_dropped: u64 = 0,
    write_errors: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) Writer {
        return .{ .allocator = allocator, .io = io };
    }

    pub fn start(self: *Writer) !void {
        self.mutex.lockUncancelable(self.io);
        if (self.running) {
            self.mutex.unlock(self.io);
            return;
        }
        self.running = true;
        self.mutex.unlock(self.io);
        self.thread = std.Thread.spawn(.{}, loop, .{self}) catch |err| {
            self.mutex.lockUncancelable(self.io);
            self.running = false;
            self.mutex.unlock(self.io);
            return err;
        };
    }

    /// Drain, stop the thread, free anything left. Safe to call twice.
    pub fn deinit(self: *Writer) void {
        self.mutex.lockUncancelable(self.io);
        if (self.deinited) {
            self.mutex.unlock(self.io);
            return;
        }
        self.deinited = true;
        // Lift a pause and wake both condvars before stopping the loop, or `drain` parks forever.
        self.paused = false;
        self.work.broadcast(self.io);
        self.done.broadcast(self.io);
        self.mutex.unlock(self.io);
        self.mutex.lockUncancelable(self.io);
        self.running = false;
        self.paused = false;
        self.work.broadcast(self.io);
        self.done.broadcast(self.io);
        self.mutex.unlock(self.io);
        if (self.thread) |t| t.join();
        self.thread = null;
        self.mutex.lockUncancelable(self.io);
        for (self.queue.items) |*b| self.freeBlob(b);
        self.queue.clearRetainingCapacity();
        self.pending_bytes = 0;
        self.queue.deinit(self.allocator);
        for (self.failures.items) |f| self.allocator.free(f.path);
        self.failures.deinit(self.allocator);
        if (self.fail_substr) |fs| self.allocator.free(fs);
        self.fail_substr = null;
        self.mutex.unlock(self.io);
    }

    fn freeBlob(self: *Writer, b: *Blob) void {
        self.allocator.free(b.path);
        self.allocator.free(b.bytes);
    }

    /// Stage one file. Takes ownership of both slices on every path, including errors.
    /// Blocks while the unwritten queue is over the permit (the only place the inference
    /// thread waits on the writer). Single producer: the inference thread.
    pub fn submit(self: *Writer, path: []u8, bytes: []u8) void {
        std.debug.assert(!self.deinited);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (!self.running) {
            // No writer: dropping is correct, the index file rides the same queue.
            self.allocator.free(path);
            self.allocator.free(bytes);
            self.files_dropped += 1;
            return;
        }
        if (self.injectedLocked(path, .submit)) {
            log.warn("  [disk-cache] background write failed: {s} ({s})\n", .{ "InjectedSubmitFailure", path });
            self.noteFailureLocked(path, "InjectedSubmitFailure");
            self.allocator.free(path);
            self.allocator.free(bytes);
            self.files_dropped += 1;
            return;
        }
        while (self.pending_bytes + self.inflight_bytes + bytes.len > self.permit_bytes and
            (self.queue.items.len > 0 or self.inflight_bytes > 0))
        {
            self.done.waitUncancelable(self.io, &self.mutex);
        }
        self.queue.append(self.allocator, .{
            .path = path,
            .bytes = bytes,
            .epoch = self.epoch,
        }) catch {
            self.allocator.free(path);
            self.allocator.free(bytes);
            self.files_dropped += 1;
            return;
        };
        self.pending_bytes += bytes.len;
        self.work.signal(self.io);
    }

    /// Wait until the files staged for `path_prefix` have been written (or dropped); null = all.
    pub fn drainPrefix(self: *Writer, path_prefix: ?[]const u8) void {
        const pre = path_prefix orelse {
            self.drain();
            return;
        };
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        while (self.running) {
            var waiting = false;
            if (self.inflight_path) |p| {
                if (std.mem.startsWith(u8, p, pre)) waiting = true;
            }
            if (!waiting) {
                for (self.queue.items) |b| {
                    if (std.mem.startsWith(u8, b.path, pre)) {
                        waiting = true;
                        break;
                    }
                }
            }
            if (!waiting) return;
            self.done.waitUncancelable(self.io, &self.mutex);
        }
    }

    /// Non-blocking twin of `drainPrefix`: is any blob for `path_prefix` still staged or in flight?
    pub fn pendingPrefix(self: *Writer, path_prefix: []const u8) bool {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.inflight_path) |p| {
            if (std.mem.startsWith(u8, p, path_prefix)) return true;
        }
        for (self.queue.items) |b| {
            if (std.mem.startsWith(u8, b.path, path_prefix)) return true;
        }
        return false;
    }

    /// Wait until every staged file has been written (or dropped).
    pub fn drain(self: *Writer) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        while (self.running and (self.queue.items.len > 0 or self.inflight_bytes > 0)) {
            self.done.waitUncancelable(self.io, &self.mutex);
        }
    }

    /// Epoch fence: staged bytes for `path_prefix` (null = everything) are discarded rather than
    /// written, and anything in flight is waited out, so the caller can remove the directory.
    pub fn fence(self: *Writer, path_prefix: ?[]const u8) void {
        self.mutex.lockUncancelable(self.io);
        if (path_prefix == null) self.epoch += 1;
        var i: usize = 0;
        while (i < self.queue.items.len) {
            const b = &self.queue.items[i];
            const doomed = if (path_prefix) |pre| std.mem.startsWith(u8, b.path, pre) else true;
            if (!doomed) {
                i += 1;
                continue;
            }
            self.pending_bytes -|= b.bytes.len;
            var owned = self.queue.orderedRemove(i);
            self.freeBlob(&owned);
            self.files_dropped += 1;
        }
        self.done.broadcast(self.io);
        while (self.running and self.inflight_bytes > 0) self.done.waitUncancelable(self.io, &self.mutex);
        self.mutex.unlock(self.io);
    }

    /// Test-only: hold / release the writer thread.
    /// Is a write to `path` still queued or in flight? Read-only on the queue.
    pub fn isPending(self: *Writer, path: []const u8) bool {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.inflight_path) |p| {
            if (std.mem.eql(u8, p, path)) return true;
        }
        for (self.queue.items) |b| {
            if (std.mem.eql(u8, b.path, path)) return true;
        }
        return false;
    }

    pub fn setPaused(self: *Writer, v: bool) void {
        self.mutex.lockUncancelable(self.io);
        self.paused = v;
        self.work.broadcast(self.io);
        self.mutex.unlock(self.io);
    }

    /// Test-only: the staged paths in write order, duped into `a`. Caller frees each item.
    pub fn stagedPaths(self: *Writer, out: *std.ArrayList([]const u8), a: std.mem.Allocator) !void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        for (self.queue.items) |b| try out.append(a, try a.dupe(u8, b.path));
    }

    pub fn pendingBytes(self: *Writer) u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.pending_bytes + self.inflight_bytes;
    }

    /// Hand over every failure recorded since the last call; the caller owns the slice and each `path`.
    pub fn takeFailures(self: *Writer) []Failure {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.failures.toOwnedSlice(self.allocator) catch blk: {
            self.unattributed_failure = true;
            break :blk &[_]Failure{};
        };
    }

    /// Read-and-clear: did a failure go unrecorded since the last call?
    pub fn takeUnattributed(self: *Writer) bool {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const v = self.unattributed_failure;
        self.unattributed_failure = false;
        return v;
    }

    /// Test-only: fail every blob whose path contains `substr` (null clears), the way an ENOSPC does.
    pub fn injectFailure(self: *Writer, substr: ?[]const u8, at: FailAt) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.fail_substr) |fs| self.allocator.free(fs);
        self.fail_substr = if (substr) |sub| (self.allocator.dupe(u8, sub) catch null) else null;
        self.fail_at = at;
    }

    /// Caller holds the mutex.
    fn injectedLocked(self: *Writer, path: []const u8, at: FailAt) bool {
        if (self.fail_at != at) return false;
        const fs = self.fail_substr orelse return false;
        return std.mem.indexOf(u8, path, fs) != null;
    }

    /// Count + record one failed blob. Caller must NOT hold the mutex.
    fn noteFailure(self: *Writer, path: []const u8, err_name: []const u8) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.noteFailureLocked(path, err_name);
    }

    fn noteFailureLocked(self: *Writer, path: []const u8, err_name: []const u8) void {
        self.write_errors += 1;
        if (self.failures.items.len >= MAX_RECORDED_FAILURES) {
            self.unattributed_failure = true;
            return;
        }
        const p = self.allocator.dupe(u8, path) catch {
            self.unattributed_failure = true;
            return;
        };
        self.failures.append(self.allocator, .{ .path = p, .err_name = err_name }) catch {
            self.allocator.free(p);
            self.unattributed_failure = true;
        };
    }

    pub fn writeErrorCount(self: *Writer) u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.write_errors;
    }

    pub fn filesWritten(self: *Writer) u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.files_written;
    }

    fn loop(self: *Writer) void {
        while (true) {
            self.mutex.lockUncancelable(self.io);
            while (self.running and (self.paused or self.queue.items.len == 0)) self.work.waitUncancelable(self.io, &self.mutex);
            if (!self.running and self.queue.items.len == 0) {
                self.mutex.unlock(self.io);
                return;
            }
            var blob = self.queue.orderedRemove(0);
            self.pending_bytes -|= blob.bytes.len;
            self.inflight_bytes = blob.bytes.len;
            self.inflight_path = blob.path;
            self.mutex.unlock(self.io);

            // Re-read the epoch under the lock, immediately before the write.
            self.mutex.lockUncancelable(self.io);
            const live_epoch = self.epoch;
            const inject = self.injectedLocked(blob.path, .write);
            self.mutex.unlock(self.io);

            var dropped = false;
            if (blob.epoch != live_epoch) {
                dropped = true;
            } else if (inject) {
                log.warn("  [disk-cache] background write failed: {s} ({s})\n", .{ "InjectedWriteFailure", blob.path });
                self.noteFailure(blob.path, "InjectedWriteFailure");
                dropped = true;
            } else if (writeAtomic(blob.path, blob.bytes)) |_| {} else |err| {
                log.warn("  [disk-cache] background write failed: {s} ({s})\n", .{ @errorName(err), blob.path });
                self.noteFailure(blob.path, @errorName(err));
                dropped = true;
            }

            self.mutex.lockUncancelable(self.io);
            if (dropped) {
                self.files_dropped += 1;
            } else {
                self.files_written += 1;
                self.bytes_written += blob.bytes.len;
            }
            self.inflight_bytes = 0;
            self.inflight_path = null;
            self.freeBlob(&blob);
            self.done.broadcast(self.io);
            self.mutex.unlock(self.io);
        }
    }
};

/// `<path>.tmp` then rename.
fn writeAtomic(path: []const u8, bytes: []const u8) !void {
    var tmp_buf: [std.fs.max_path_bytes + 8]u8 = undefined;
    if (path.len + 6 >= tmp_buf.len) return error.NameTooLong;
    @memcpy(tmp_buf[0..path.len], path);
    @memcpy(tmp_buf[path.len .. path.len + 4], ".tmp");
    tmp_buf[path.len + 4] = 0;
    const tmp: [:0]const u8 = tmp_buf[0 .. path.len + 4 :0];

    const fd = std.c.open(tmp.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    var off: usize = 0;
    while (off < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + off, bytes.len - off);
        if (n < 0) {
            const e = std.c._errno().*;
            if (e == @intFromEnum(std.c.E.INTR) or e == @intFromEnum(std.c.E.AGAIN)) continue;
            return error.WriteFailed;
        }
        if (n == 0) return error.WriteFailed;
        off += @intCast(n);
    }

    var final_buf: [std.fs.max_path_bytes + 1]u8 = undefined;
    if (path.len >= final_buf.len) return error.NameTooLong;
    @memcpy(final_buf[0..path.len], path);
    final_buf[path.len] = 0;
    const final: [:0]const u8 = final_buf[0..path.len :0];
    if (std.c.rename(tmp.ptr, final.ptr) != 0) return error.RenameFailed;
}

// ── Tests ──

const testing = std.testing;

test "kv_disk_writer: files land off-thread, in FIFO order, and atomically" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];

    var w = Writer.init(testing.allocator, std.testing.io);
    try w.start();
    defer w.deinit();

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const path = try std.fmt.allocPrint(testing.allocator, "{s}/f{d}.bin", .{ root, i });
        const bytes = try testing.allocator.alloc(u8, 4096);
        @memset(bytes, @intCast(i));
        w.submit(path, bytes);
    }
    w.drain();
    try testing.expectEqual(@as(u64, 10), w.filesWritten());
    try testing.expectEqual(@as(u64, 0), w.pendingBytes());

    i = 0;
    while (i < 10) : (i += 1) {
        var name: [64]u8 = undefined;
        const n = try std.fmt.bufPrint(&name, "f{d}.bin", .{i});
        const got = try tmp.dir.readFileAlloc(std.testing.io, n, testing.allocator, .limited(1 << 20));
        defer testing.allocator.free(got);
        try testing.expectEqual(@as(usize, 4096), got.len);
        try testing.expectEqual(@as(u8, @intCast(i)), got[0]);
        var tname: [72]u8 = undefined;
        const tn = try std.fmt.bufPrint(&tname, "f{d}.bin.tmp", .{i});
        try testing.expectError(error.FileNotFound, tmp.dir.statFile(std.testing.io, tn, .{}));
    }
}

test "kv_disk_writer: the epoch fence drops staged bytes instead of writing them" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];

    var w = Writer.init(testing.allocator, std.testing.io);
    const p0 = try std.fmt.allocPrint(testing.allocator, "{s}/never.bin", .{root});
    const b0 = try testing.allocator.alloc(u8, 16);
    w.submit(p0, b0);
    try testing.expectEqual(@as(u64, 1), w.files_dropped);

    try w.start();
    defer w.deinit();
    w.fence(null);
    const after_fence = w.epoch;
    try testing.expect(after_fence > 1);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(std.testing.io, "never.bin", .{}));
}

test "kv_disk_writer: the host-byte permit bounds staged bytes" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];

    var w = Writer.init(testing.allocator, std.testing.io);
    w.permit_bytes = 64 * 1024;
    try w.start();
    defer w.deinit();

    var i: usize = 0;
    while (i < 32) : (i += 1) {
        const path = try std.fmt.allocPrint(testing.allocator, "{s}/p{d}.bin", .{ root, i });
        const bytes = try testing.allocator.alloc(u8, 16 * 1024);
        @memset(bytes, 7);
        w.submit(path, bytes);
        try testing.expect(w.pendingBytes() <= w.permit_bytes + 16 * 1024);
    }
    w.drain();
    try testing.expectEqual(@as(u64, 32), w.filesWritten());
}

test "kv_disk_writer: a PAUSED writer deinits without blocking" {
    // A test that pauses the writer and then fails must not hang the suite.
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];

    var w = Writer.init(testing.allocator, std.testing.io);
    try w.start();
    w.setPaused(true);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/held.bin", .{root});
    const bytes = try testing.allocator.alloc(u8, 4096);
    @memset(bytes, 3);
    w.submit(path, bytes);
    try testing.expect(w.pendingBytes() > 0);

    w.deinit();
    try testing.expect(w.thread == null);
    w.deinit();
}
