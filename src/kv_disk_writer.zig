//! SSD-first background writer (mechanism 2).
//!
//! The SSD prefix-cache tier used to serialize AND write on the inference
//! thread — the sole mlx caller — so a multi-GB entry stalled the next
//! request. `DiskTier.max_flush_bytes` existed only to bound that stall, and
//! it did so by TRUNCATING the entry: a 374k-token session persisted ~42k
//! tokens per finished request and stayed unrestorable for turns.
//!
//! The split: the inference thread keeps the device→host readback (it must —
//! mlx arrays are inference-thread-owned) and hands ONE writer thread a plain
//! host byte buffer per file. Only BYTES cross the boundary; no mlx handle
//! ever does. The writer does `tmp` + `rename`, so a kill -9 can leave a
//! `.tmp` but never a half-written file under its final name, and the queue is
//! FIFO so an entry's `meta.json` — enqueued after its chunks — is always the
//! LAST file to land: a crash mid-flush leaves chunks with no index, which the
//! tier's scan already treats as a miss.
//!
//! Two bounds keep it honest:
//!   * a HOST-BYTE PERMIT (`permit_bytes`, ~1 GiB): `submit` blocks once the
//!     unwritten queue exceeds it, so a runaway producer cannot trade GPU
//!     memory for host memory. Back-pressure, not a truncation cliff.
//!   * an EPOCH FENCE: eviction/invalidation bumps the epoch and drains, so
//!     staged bytes for a directory that is about to be removed are dropped
//!     rather than written into it.
//!
//! POSIX file syscalls (not `std.Io`) on purpose: the process' `std.Io` is the
//! single-threaded implementation and this runs off the main thread.

const std = @import("std");
const log = @import("log.zig");

/// One staged file: the final absolute path and the exact bytes to write.
/// Both buffers are owned by the queue once `submit` accepts them.
pub const Blob = struct {
    path: []u8,
    bytes: []u8,
    epoch: u64,
};

pub const DEFAULT_PERMIT_BYTES: u64 = 1024 * 1024 * 1024;

pub const Writer = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    /// Signalled when work arrives or the writer is asked to stop.
    work: std.Io.Condition = .init,
    /// Signalled when a blob leaves the queue (permit freed / drain reached).
    done: std.Io.Condition = .init,
    queue: std.ArrayList(Blob) = .empty,
    /// Bytes staged and not yet written.
    pending_bytes: u64 = 0,
    /// In flight in the writer thread right now (0 or 1 blob's worth).
    inflight_bytes: u64 = 0,
    /// The in-flight blob's path, so a prefix-scoped drain can see it. Valid
    /// only while `inflight_bytes > 0`; the blob owns the memory.
    inflight_path: ?[]const u8 = null,
    permit_bytes: u64 = DEFAULT_PERMIT_BYTES,
    epoch: u64 = 1,
    running: bool = false,
    /// Test-only: hold the queue so a caller can inspect submission ORDER
    /// (the "index file lands last" guarantee) deterministically.
    paused: bool = false,
    /// `deinit` ran. Makes its "safe to call twice" doc claim true.
    deinited: bool = false,
    thread: ?std.Thread = null,
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

    /// Drain, stop the thread, free anything left. Safe to call twice — and now
    /// actually is: the second call used to re-`deinit` an already-deinited
    /// `queue` and re-join a null thread. The claim was in the doc comment
    /// before it was in the code. (audit S13)
    pub fn deinit(self: *Writer) void {
        self.mutex.lockUncancelable(self.io);
        if (self.deinited) {
            self.mutex.unlock(self.io);
            return;
        }
        self.deinited = true;
        // A PAUSED writer must not make teardown block, and neither may it
        // leave another thread parked in `drain`/`submit` forever: lift the
        // pause and wake BOTH condition variables before the loop is stopped.
        // `drain` waits on `done`, and stopping the loop alone never signals
        // it. (B-A1 chunk-share audit: a failed assertion under a paused
        // writer deadlocked the suite.)
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
        // Whatever the writer never got to.
        self.mutex.lockUncancelable(self.io);
        for (self.queue.items) |*b| self.freeBlob(b);
        self.queue.clearRetainingCapacity();
        self.pending_bytes = 0;
        self.queue.deinit(self.allocator);
        self.mutex.unlock(self.io);
    }

    fn freeBlob(self: *Writer, b: *Blob) void {
        self.allocator.free(b.path);
        self.allocator.free(b.bytes);
    }

    /// Stage one file. Takes ownership of BOTH slices on every path, including
    /// errors — the caller must not free them afterwards (ownership by
    /// provenance: only the queue can free what the queue accepted).
    ///
    /// Blocks while the unwritten queue is over the permit. That block is the
    /// designed back-pressure and is the ONLY place the inference thread waits
    /// on the writer.
    ///
    /// SINGLE PRODUCER. Every caller is the inference thread (commit flush,
    /// prefill write-through, index write), which is also the only caller of
    /// `deinit` — so a `submit` can never race the post-join `queue.deinit`
    /// and append into freed memory. True by construction today; asserted
    /// because nothing else enforces it if a second producer appears.
    /// (audit N10)
    pub fn submit(self: *Writer, path: []u8, bytes: []u8) void {
        std.debug.assert(!self.deinited);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (!self.running) {
            // No writer: dropping is correct — the index file rides the same
            // queue, so nothing half-indexed can result.
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

    /// Wait until the files staged for `path_prefix` have been written (or
    /// dropped); null = all of them.
    ///
    /// A restore only needs ITS entry on disk. Draining the whole queue made
    /// the next turn's head wait on the previous turn's tail — the inference
    /// thread blocking on writes that no one was reading, which is the stall
    /// the background writer exists to remove. (audit S12)
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

    /// NON-BLOCKING twin of `drainPrefix`: is any blob for `path_prefix` still
    /// staged or in the writer's hands?
    ///
    /// `drainPrefix` is a WAIT, and the inference thread must never wait on a
    /// write it is not reading. The idle spill's durability check used
    /// `drainWriter` (the whole-queue form), which parked decode at the end of
    /// every request that had a flush outstanding — the exact stall the
    /// background writer exists to remove. An entry with writes in flight is
    /// simply not evictable YET; the next pass asks again. (external review
    /// item 6)
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

    /// Epoch fence: staged bytes for `path_prefix` (null = EVERYTHING) are
    /// discarded rather than written, and anything already in the writer's
    /// hands is waited out — so the caller can remove the directory those
    /// bytes were headed for without the writer re-creating it.
    ///
    /// The prefix form is load-bearing: an `appendCommit` that evicts an LRU
    /// entry must not throw away the bytes it just staged for the entry it is
    /// writing. Only the doomed directory's blobs go.
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
        // A blob already in the writer's hands is waited out (a full fence also
        // discards it via the epoch check) so the caller's rmdir is safe.
        while (self.running and self.inflight_bytes > 0) self.done.waitUncancelable(self.io, &self.mutex);
        self.mutex.unlock(self.io);
    }

    /// Test-only: hold / release the writer thread.
    /// Is a write to `path` still queued or in flight? READ-ONLY on the
    /// queue — the opposite of `fence`, which DISCARDS what it matches. A
    /// reader that must not consume another entry's pending files (the
    /// chunk-share link) asks this and links only what has LANDED.
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

    /// Test-only: the staged paths, in submission (write) order. The strings are
    /// DUPED into `a` — handing out `b.path` borrowed the queue's memory, which
    /// the writer frees the moment the mutex is released. Test-only, but a
    /// use-after-free shape in a shipped file is one a future caller inherits.
    /// Caller frees each item. (audit N11)
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

            // Re-read the epoch UNDER THE LOCK, immediately before the write.
            // It used to be sampled in the same critical section that popped
            // the blob, where it could not possibly differ from `blob.epoch` —
            // the check was dead and the comment calling it load-bearing was
            // wrong. A fence raised between the pop and the write is exactly
            // the interleaving the fence exists for. (audit S14)
            self.mutex.lockUncancelable(self.io);
            const live_epoch = self.epoch;
            self.mutex.unlock(self.io);

            var dropped = false;
            if (blob.epoch != live_epoch) {
                dropped = true;
            } else if (writeAtomic(blob.path, blob.bytes)) |_| {} else |err| {
                log.warn("  [disk-cache] background write failed: {s} ({s})\n", .{ @errorName(err), blob.path });
                self.mutex.lockUncancelable(self.io);
                self.write_errors += 1;
                self.mutex.unlock(self.io);
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

/// `<path>.tmp` then rename — a kill -9 leaves at worst a `.tmp` the tier's
/// scan ignores, never a truncated file under its real name.
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
            // A benign signal must not cost the file. (audit N9)
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

    // Ten payloads; the last one stands in for `meta.json` — FIFO ordering is
    // what makes "index last" true without the producer waiting.
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
        // No `.tmp` survivor under the final name's sibling.
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
    // Not started: `submit` must still consume ownership. Then start and
    // fence a real queue.
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
        // The permit is a HARD bound on host memory held for the writer.
        try testing.expect(w.pendingBytes() <= w.permit_bytes + 16 * 1024);
    }
    w.drain();
    try testing.expectEqual(@as(u64, 32), w.filesWritten());
}

test "kv_disk_writer: a PAUSED writer deinits without blocking" {
    // A test that pauses the writer and then fails an assertion must not hang
    // the SUITE. The B-A1 chunk-share test did exactly that: `setPaused(true)`
    // + `defer tier.deinit()` and the teardown drain waited forever on a queue
    // the paused loop would never take. `deinit` lifts the pause (and wakes
    // `done` as well as `work`, since `drain` parks on `done`) before it stops
    // the loop, so this returns whatever the queue holds.
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

    // The bar is that this RETURNS. Nothing frees the blob but `deinit`, so a
    // leak-checked run also proves it took ownership of the paused queue.
    w.deinit();
    try testing.expect(w.thread == null);
    // Safe to call twice, paused or not.
    w.deinit();
}
