const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

pub const align_bytes: usize = std.heap.page_size_max;

pub fn pageSize() usize {
    return std.heap.pageSize();
}

pub fn pageRoundUp(n: u64) u64 {
    const p: u64 = @intCast(pageSize());
    return (n + p - 1) / p * p;
}

pub fn pageRoundDown(n: u64) u64 {
    const p: u64 = @intCast(pageSize());
    return n / p * p;
}

pub const ReadHints = struct {
    nocache: bool = true,
    readahead_off: bool = true,
};

var hint_failure_logged: bool = false;

pub fn applyReadHints(fd: std.c.fd_t, hints: ReadHints) void {
    var failed = false;
    const want: c_int = if (hints.nocache) 1 else 0;
    if (std.c.fcntl(fd, std.c.F.NOCACHE, want) != 0) failed = true;
    if (hints.readahead_off and std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0)) != 0) failed = true;
    if (failed and !hint_failure_logged) {
        hint_failure_logged = true;
        log.warn("[expert-io] read hints declined (fd {d})", .{fd});
    }
}

pub const FileCache = struct {
    pub const CAPACITY: usize = 64;

    const Entry = struct {
        path: []u8,
        fd: std.c.fd_t,
        dev: i64,
        ino: u64,
        size: u64,
        mtime_sec: i64,
        mtime_nsec: i64,
        clock: u64,
    };

    allocator: std.mem.Allocator,
    hints: ReadHints,
    capacity: usize = CAPACITY,
    entries: std.ArrayList(Entry) = .empty,
    tick: u64 = 0,
    opens: u64 = 0,
    hits: u64 = 0,
    revalidations: u64 = 0,
    evictions: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, hints: ReadHints) FileCache {
        return .{ .allocator = allocator, .hints = hints };
    }

    pub fn initCapacity(allocator: std.mem.Allocator, hints: ReadHints, capacity: usize) FileCache {
        return .{ .allocator = allocator, .hints = hints, .capacity = @max(capacity, 1) };
    }

    pub fn deinit(self: *FileCache) void {
        for (self.entries.items) |e| {
            _ = std.c.close(e.fd);
            self.allocator.free(e.path);
        }
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    fn identityOf(path: [:0]const u8) !Entry {
        var st: std.c.Stat = undefined;
        if (std.c.stat(path.ptr, &st) != 0) return error.FillStatFailed;
        const mt = st.mtime();
        return .{
            .path = &.{},
            .fd = -1,
            .dev = @intCast(st.dev),
            .ino = @intCast(st.ino),
            .size = @intCast(st.size),
            .mtime_sec = @intCast(mt.sec),
            .mtime_nsec = @intCast(mt.nsec),
            .clock = 0,
        };
    }

    fn sameIdentity(a: Entry, b: Entry) bool {
        return a.dev == b.dev and a.ino == b.ino and a.size == b.size and
            a.mtime_sec == b.mtime_sec and a.mtime_nsec == b.mtime_nsec;
    }

    pub fn get(self: *FileCache, path: []const u8) !std.c.fd_t {
        if (path.len == 0 or path.len >= std.fs.max_path_bytes) return error.FillOpenFailed;
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        @memcpy(buf[0..path.len], path);
        buf[path.len] = 0;
        const zpath = buf[0..path.len :0];
        const fresh = try identityOf(zpath);
        self.tick += 1;
        for (self.entries.items) |*e| {
            if (!std.mem.eql(u8, e.path, path)) continue;
            if (sameIdentity(e.*, fresh)) {
                e.clock = self.tick;
                self.hits += 1;
                return e.fd;
            }
            _ = std.c.close(e.fd);
            const fd = try openHinted(zpath, self.hints);
            e.fd = fd;
            e.dev = fresh.dev;
            e.ino = fresh.ino;
            e.size = fresh.size;
            e.mtime_sec = fresh.mtime_sec;
            e.mtime_nsec = fresh.mtime_nsec;
            e.clock = self.tick;
            self.revalidations += 1;
            return fd;
        }
        if (self.entries.items.len >= self.capacity) {
            var victim: usize = 0;
            for (self.entries.items, 0..) |e, i| {
                if (e.clock < self.entries.items[victim].clock) victim = i;
            }
            _ = std.c.close(self.entries.items[victim].fd);
            self.allocator.free(self.entries.items[victim].path);
            _ = self.entries.swapRemove(victim);
            self.evictions += 1;
        }
        const fd = try openHinted(zpath, self.hints);
        errdefer _ = std.c.close(fd);
        const owned = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned);
        try self.entries.append(self.allocator, .{
            .path = owned,
            .fd = fd,
            .dev = fresh.dev,
            .ino = fresh.ino,
            .size = fresh.size,
            .mtime_sec = fresh.mtime_sec,
            .mtime_nsec = fresh.mtime_nsec,
            .clock = self.tick,
        });
        self.opens += 1;
        return fd;
    }
};

pub fn openHinted(path: [:0]const u8, hints: ReadHints) !std.c.fd_t {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FillOpenFailed;
    applyReadHints(fd, hints);
    return fd;
}

pub const FillSpan = struct {
    file: u16,
    offset: u64,
    len: u64,
    dst: [*]u8,
};

pub const ReadJob = struct {
    file: u16,
    offset: u64,
    len: u64,
    first: usize,
    count: usize,
    direct: bool = false,
};

pub const FailureReason = enum {
    read_failed,
    short_read,
    past_eof,
    bounce_unavailable,
};

pub const FillFailure = struct {
    span: usize,
    file: u16,
    offset: u64,
    len: u64,
    dst: [*]u8,
    reason: FailureReason,
};

pub fn failureError(reason: FailureReason) anyerror {
    return switch (reason) {
        .read_failed => error.FillReadFailed,
        .short_read => error.FillShortRead,
        .past_eof => error.FillSpanPastEof,
        .bounce_unavailable => error.FillBounceUnavailable,
    };
}

fn spanLess(_: void, a: FillSpan, b: FillSpan) bool {
    if (a.file != b.file) return a.file < b.file;
    return a.offset < b.offset;
}

pub fn coalesceSpans(allocator: std.mem.Allocator, spans: []const FillSpan, max_bytes: u64) ![]ReadJob {
    if (max_bytes == 0) return error.InvalidCoalesceBound;
    var jobs: std.ArrayList(ReadJob) = .empty;
    errdefer jobs.deinit(allocator);
    var i: usize = 0;
    while (i < spans.len) {
        const first = i;
        const file = spans[i].file;
        const offset = spans[i].offset;
        var len = spans[i].len;
        if (len == 0 or len > max_bytes) return error.InvalidFillSpan;
        i += 1;
        while (i < spans.len) : (i += 1) {
            const next = spans[i];
            if (next.len == 0) return error.InvalidFillSpan;
            if (next.file != file or next.offset != offset + len) break;
            if (len + next.len > max_bytes) break;
            len += next.len;
        }
        try jobs.append(allocator, .{ .file = file, .offset = offset, .len = len, .first = first, .count = i - first });
    }
    return jobs.toOwnedSlice(allocator);
}

fn jobIsDirect(spans: []const FillSpan, job: ReadJob, page: u64) bool {
    var expect = spans[job.first].dst;
    var k: usize = 0;
    while (k < job.count) : (k += 1) {
        const s = spans[job.first + k];
        if (s.dst != expect) return false;
        expect = s.dst + @as(usize, @intCast(s.len));
    }
    if (job.offset % page != 0) return false;
    if (job.len % page != 0) return false;
    if (@intFromPtr(spans[job.first].dst) % page != 0) return false;
    return true;
}

pub fn partitionByMidpoint(allocator: std.mem.Allocator, jobs: []const ReadJob, workers: usize) ![]usize {
    if (workers == 0) return error.InvalidFillPool;
    const bounds = try allocator.alloc(usize, workers + 1);
    errdefer allocator.free(bounds);
    var total: u64 = 0;
    for (jobs) |j| total += j.len;
    for (bounds) |*b| b.* = jobs.len;
    bounds[0] = 0;
    if (total == 0 or jobs.len == 0) {
        for (bounds) |*b| b.* = 0;
        return bounds;
    }
    var cursor: u64 = 0;
    var owner = try allocator.alloc(usize, jobs.len);
    defer allocator.free(owner);
    for (jobs, 0..) |j, i| {
        const mid = cursor + j.len / 2;
        var w = @as(usize, @intCast(mid * @as(u64, workers) / total));
        if (w >= workers) w = workers - 1;
        owner[i] = w;
        cursor += j.len;
    }
    var w: usize = 0;
    var i: usize = 0;
    while (w < workers) : (w += 1) {
        bounds[w] = i;
        while (i < jobs.len and owner[i] <= w) i += 1;
    }
    bounds[workers] = jobs.len;
    return bounds;
}

pub const FillPlan = struct {
    allocator: std.mem.Allocator,
    spans: []FillSpan,
    jobs: []ReadJob,
    bounds: []usize,

    pub fn init(
        allocator: std.mem.Allocator,
        spans_in: []const FillSpan,
        max_bytes: u64,
        workers: usize,
        force_bounce: bool,
    ) !FillPlan {
        const spans = try allocator.dupe(FillSpan, spans_in);
        errdefer allocator.free(spans);
        std.mem.sort(FillSpan, spans, {}, spanLess);
        const jobs = try coalesceSpans(allocator, spans, max_bytes);
        errdefer allocator.free(jobs);
        const page: u64 = @intCast(pageSize());
        for (jobs) |*j| j.direct = !force_bounce and jobIsDirect(spans, j.*, page);
        const bounds = try partitionByMidpoint(allocator, jobs, workers);
        return .{ .allocator = allocator, .spans = spans, .jobs = jobs, .bounds = bounds };
    }

    pub fn deinit(self: *FillPlan) void {
        self.allocator.free(self.spans);
        self.allocator.free(self.jobs);
        self.allocator.free(self.bounds);
        self.* = undefined;
    }
};

pub const FillStats = struct {
    bytes_read: u64 = 0,
    bytes_issued: u64 = 0,
    bytes_bounced: u64 = 0,
    reads_issued: u64 = 0,
    pread_ns: u64 = 0,
    max_worker_bytes: u64 = 0,
    min_worker_bytes: u64 = 0,

    pub fn imbalance(self: FillStats, workers: usize) f64 {
        if (workers == 0 or self.bytes_read == 0) return 1.0;
        const mean = @as(f64, @floatFromInt(self.bytes_read)) / @as(f64, @floatFromInt(workers));
        return @as(f64, @floatFromInt(self.max_worker_bytes)) / mean;
    }
};

pub const FillOptions = struct {
    workers: usize = 8,
    queue_capacity: usize = 1,
    bounce_cap: usize = 64 * 1024 * 1024,
    coalesce_max: u64 = 64 * 1024 * 1024,
    force_bounce: bool = false,
};

const Worker = struct {
    end: usize = 0,
    pushed: usize = 0,
    taken: usize = 0,
    completed: usize = 0,
    bytes: u64 = 0,
    bounce: []align(align_bytes) u8 = &.{},
};

pub const FillPool = struct {
    allocator: std.mem.Allocator,
    opts: FillOptions,
    threads: []std.Thread,
    workers: []Worker,
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    quit: bool = false,
    cancel: bool = false,
    files: []const std.c.fd_t = &.{},
    plan: ?*const FillPlan = null,
    failure: ?FillFailure = null,
    accepted: u64 = 0,
    max_depth: usize = 0,
    stats_acc: FillStats = .{},

    pub fn create(allocator: std.mem.Allocator, opts: FillOptions) !*FillPool {
        if (opts.workers == 0 or opts.workers > 64) return error.InvalidFillPool;
        if (opts.queue_capacity == 0) return error.InvalidFillPool;
        if (opts.bounce_cap == 0 or opts.coalesce_max == 0) return error.InvalidFillPool;
        const pool = try allocator.create(FillPool);
        errdefer allocator.destroy(pool);
        const threads = try allocator.alloc(std.Thread, opts.workers);
        errdefer allocator.free(threads);
        const workers = try allocator.alloc(Worker, opts.workers);
        errdefer allocator.free(workers);
        for (workers) |*w| w.* = .{};
        pool.* = .{ .allocator = allocator, .opts = opts, .threads = threads, .workers = workers };
        var started: usize = 0;
        errdefer pool.shutdown(started);
        for (threads, 0..) |*thread, i| {
            thread.* = try std.Thread.spawn(.{ .stack_size = 128 * 1024 }, workerMain, .{ pool, i });
            started += 1;
        }
        return pool;
    }

    fn shutdown(self: *FillPool, started: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.quit = true;
        self.cv.broadcast(io);
        self.mu.unlock(io);
        for (self.threads[0..started]) |thread| thread.join();
    }

    pub fn destroy(self: *FillPool) void {
        const allocator = self.allocator;
        self.shutdown(self.threads.len);
        for (self.workers) |w| if (w.bounce.len != 0) allocator.free(w.bounce);
        allocator.free(self.workers);
        allocator.free(self.threads);
        allocator.destroy(self);
    }

    pub fn resetStats(self: *FillPool) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.stats_acc = .{};
        for (self.workers) |*w| w.bytes = 0;
        self.mu.unlock(io);
    }

    pub fn stats(self: *FillPool) FillStats {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        var out = self.stats_acc;
        var max_b: u64 = 0;
        var min_b: u64 = std.math.maxInt(u64);
        for (self.workers) |w| {
            if (w.bytes > max_b) max_b = w.bytes;
            if (w.bytes < min_b) min_b = w.bytes;
        }
        out.max_worker_bytes = max_b;
        out.min_worker_bytes = if (min_b == std.math.maxInt(u64)) 0 else min_b;
        return out;
    }

    pub fn maxQueueDepth(self: *FillPool) usize {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        return self.max_depth;
    }

    pub fn acceptedCount(self: *FillPool) u64 {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        return self.accepted;
    }

    pub fn firstFailure(self: *FillPool) ?FillFailure {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        return self.failure;
    }

    pub fn requestCancel(self: *FillPool) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.cancel = true;
        self.cv.broadcast(io);
        self.mu.unlock(io);
    }

    pub fn submit(self: *FillPool, files: []const std.c.fd_t, plan: *const FillPlan) !void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.files = files;
        self.plan = plan;
        self.cancel = false;
        self.failure = null;
        self.accepted = 0;
        self.max_depth = 0;
        for (self.workers, 0..) |*w, i| {
            w.end = plan.bounds[i + 1];
            w.pushed = plan.bounds[i];
            w.taken = plan.bounds[i];
            w.completed = plan.bounds[i];
        }
        self.cv.broadcast(io);
        while (!self.cancel) {
            var remaining = false;
            var pushed_any = false;
            for (self.workers) |*w| {
                if (w.pushed >= w.end) continue;
                remaining = true;
                if (w.pushed - w.taken < self.opts.queue_capacity) {
                    w.pushed += 1;
                    pushed_any = true;
                    const depth = w.pushed - w.completed;
                    if (depth > self.max_depth) self.max_depth = depth;
                }
            }
            if (!remaining) break;
            if (pushed_any) {
                self.cv.broadcast(io);
            } else {
                self.cv.wait(io, &self.mu) catch {};
            }
        }
        self.mu.unlock(io);
    }

    pub fn join(self: *FillPool) !void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        while (true) {
            var done = true;
            for (self.workers) |w| {
                const wanted = if (self.cancel) w.taken else w.pushed;
                if (w.completed < wanted) done = false;
            }
            if (done) break;
            self.cv.wait(io, &self.mu) catch {};
        }
        const failure = self.failure;
        var incomplete = false;
        for (self.workers) |w| {
            if (w.completed < w.end) incomplete = true;
        }
        const cancelled = self.cancel;
        self.plan = null;
        self.mu.unlock(io);
        if (failure) |f| return failureError(f.reason);
        if (cancelled and incomplete) return error.FillCancelled;
    }

    pub fn run(self: *FillPool, files: []const std.c.fd_t, plan: *const FillPlan) !void {
        var submit_err: ?anyerror = null;
        self.submit(files, plan) catch |e| {
            submit_err = e;
        };
        var join_err: ?anyerror = null;
        self.join() catch |e| {
            join_err = e;
        };
        if (submit_err) |e| return e;
        if (join_err) |e| return e;
    }

    fn recordFailure(self: *FillPool, f: FillFailure) void {
        if (self.failure) |old| {
            if (old.span <= f.span) return;
        }
        self.failure = f;
    }

    fn bounceFor(self: *FillPool, worker: usize, len: usize) ?[]align(align_bytes) u8 {
        const w = &self.workers[worker];
        if (w.bounce.len >= len) return w.bounce;
        if (len > self.opts.bounce_cap) return null;
        const want: usize = @intCast(pageRoundUp(len));
        const fresh = self.allocator.alignedAlloc(u8, .fromByteUnits(align_bytes), want) catch return null;
        if (w.bounce.len != 0) self.allocator.free(w.bounce);
        w.bounce = fresh;
        return w.bounce;
    }

    const ReadResult = struct { done: usize, reason: ?FailureReason };

    fn preadInto(fd: std.c.fd_t, dst: []u8, offset: u64, required: usize) ReadResult {
        var done: usize = 0;
        while (done < dst.len) {
            const got = std.c.pread(fd, dst[done..].ptr, dst.len - done, @intCast(offset + done));
            if (got < 0) {
                if (std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
                return .{ .done = done, .reason = if (done < required) .read_failed else null };
            }
            if (got == 0) return .{ .done = done, .reason = if (done < required) .past_eof else null };
            done += @intCast(got);
        }
        return .{ .done = done, .reason = null };
    }

    fn runJob(self: *FillPool, worker: usize, files: []const std.c.fd_t, plan: *const FillPlan, index: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        const job = plan.jobs[index];
        const len: usize = @intCast(job.len);
        var head: usize = 0;
        var read_offset: u64 = job.offset;
        var target: []u8 = undefined;
        if (job.direct) {
            target = plan.spans[job.first].dst[0..len];
        } else {
            if (job.len > self.opts.bounce_cap) {
                self.finishFailed(worker, plan, job, .bounce_unavailable);
                return;
            }
            read_offset = pageRoundDown(job.offset);
            head = @intCast(job.offset - read_offset);
            const want: usize = @intCast(pageRoundUp(@as(u64, head) + job.len));
            self.mu.lockUncancelable(io);
            const buf = self.bounceFor(worker, want);
            self.mu.unlock(io);
            if (buf == null) {
                self.finishFailed(worker, plan, job, .bounce_unavailable);
                return;
            }
            target = buf.?[0..want];
        }
        const required = head + len;
        const started = std.Io.Timestamp.now(io, .boot);
        const result = preadInto(files[job.file], target, read_offset, required);
        const elapsed: u64 = @intCast(@max(@as(i96, 0), started.untilNow(io, .boot).nanoseconds));
        const delivered: usize = if (result.done > head) result.done - head else 0;
        if (!job.direct) {
            var written: usize = 0;
            var k: usize = 0;
            while (k < job.count and written < delivered) : (k += 1) {
                const s1 = plan.spans[job.first + k];
                const sl: usize = @intCast(s1.len);
                const take = @min(sl, delivered - written);
                @memcpy(s1.dst[0..take], target[head + written .. head + written + take]);
                written += sl;
            }
        }
        self.mu.lockUncancelable(io);
        self.stats_acc.bytes_read += delivered;
        self.stats_acc.bytes_issued += result.done;
        self.stats_acc.bytes_bounced += if (job.direct) 0 else delivered;
        self.stats_acc.reads_issued += 1;
        self.stats_acc.pread_ns += elapsed;
        self.workers[worker].bytes += delivered;
        if (result.reason) |reason| {
            var acc: u64 = 0;
            var span_index = job.first;
            var k: usize = 0;
            while (k < job.count) : (k += 1) {
                const s1 = plan.spans[job.first + k];
                if (acc + s1.len > delivered) {
                    span_index = job.first + k;
                    break;
                }
                acc += s1.len;
            }
            self.recordFailure(.{
                .span = span_index,
                .file = job.file,
                .offset = plan.spans[span_index].offset,
                .len = plan.spans[span_index].len,
                .dst = plan.spans[span_index].dst,
                .reason = reason,
            });
        }
        self.workers[worker].completed += 1;
        self.cv.broadcast(io);
        self.mu.unlock(io);
    }

    fn finishFailed(self: *FillPool, worker: usize, plan: *const FillPlan, job: ReadJob, reason: FailureReason) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.recordFailure(.{
            .span = job.first,
            .file = job.file,
            .offset = plan.spans[job.first].offset,
            .len = plan.spans[job.first].len,
            .dst = plan.spans[job.first].dst,
            .reason = reason,
        });
        self.workers[worker].completed += 1;
        self.cv.broadcast(io);
        self.mu.unlock(io);
    }

    fn workerMain(self: *FillPool, worker: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        while (true) {
            self.mu.lockUncancelable(io);
            var index: usize = 0;
            while (true) {
                if (self.quit) {
                    self.mu.unlock(io);
                    return;
                }
                const w = &self.workers[worker];
                if (!self.cancel and w.taken < w.pushed) {
                    index = w.taken;
                    w.taken += 1;
                    self.accepted += 1;
                    self.cv.broadcast(io);
                    break;
                }
                self.cv.broadcast(io);
                self.cv.wait(io, &self.mu) catch {};
            }
            const files = self.files;
            const plan = self.plan.?;
            self.mu.unlock(io);
            self.runJob(worker, files, plan, index);
        }
    }
};

pub const SlabState = enum {
    free,
    filling,
    ready,
    leased,
    readers_complete,
    reclaimable,
};

pub const Lease = struct {
    epoch: u64,
    id: u64,
};

pub const PageSlab = struct {
    allocator: std.mem.Allocator,
    bytes: []align(align_bytes) u8,
    state: SlabState = .free,
    epoch: u64 = 0,
    next_id: u64 = 1,
    live: std.ArrayList(u64) = .empty,
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,

    pub fn create(allocator: std.mem.Allocator, len: usize) !*PageSlab {
        if (len == 0) return error.SlabEmpty;
        const slab = try allocator.create(PageSlab);
        errdefer allocator.destroy(slab);
        const mapped = try std.posix.mmap(
            null,
            @intCast(pageRoundUp(len)),
            .{ .READ = true, .WRITE = true },
            .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1,
            0,
        );
        slab.* = .{ .allocator = allocator, .bytes = @alignCast(mapped) };
        return slab;
    }

    pub fn destroy(self: *PageSlab) void {
        const allocator = self.allocator;
        self.live.deinit(allocator);
        std.posix.munmap(self.bytes);
        allocator.destroy(self);
    }

    pub fn stateOf(self: *PageSlab) SlabState {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        return self.state;
    }

    pub fn currentEpoch(self: *PageSlab) u64 {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        return self.epoch;
    }

    fn startFillLocked(self: *PageSlab) u64 {
        self.state = .filling;
        self.epoch += 1;
        return self.epoch;
    }

    pub fn tryBeginFill(self: *PageSlab) !u64 {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        switch (self.state) {
            .free, .reclaimable => return self.startFillLocked(),
            .filling => return error.SlabFilling,
            .ready, .readers_complete => return error.SlabPublished,
            .leased => return error.SlabLeased,
        }
    }

    pub fn beginFillTimeout(self: *PageSlab, nanoseconds: u64) !u64 {
        const io = std.Io.Threaded.global_single_threaded.io();
        const deadline = std.Io.Timestamp.now(io, .awake);
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        while (true) {
            switch (self.state) {
                .free, .reclaimable => return self.startFillLocked(),
                .filling => return error.SlabFilling,
                .ready, .readers_complete => return error.SlabPublished,
                .leased => {},
            }
            const spent: i96 = deadline.untilNow(io, .awake).nanoseconds;
            if (spent >= @as(i96, @intCast(nanoseconds))) return error.SlabRefillTimeout;
            const left: u64 = nanoseconds - @as(u64, @intCast(@max(@as(i96, 0), spent)));
            self.cv.waitTimeout(io, &self.mu, .{ .duration = .{
                .raw = .fromNanoseconds(@intCast(left)),
                .clock = .awake,
            } }) catch {};
        }
    }

    pub fn writable(self: *PageSlab) ![]u8 {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        switch (self.state) {
            .filling => return self.bytes,
            .free, .reclaimable => return error.SlabNotFilling,
            .ready, .leased, .readers_complete => return error.SlabPublished,
        }
    }

    pub fn publish(self: *PageSlab) !u64 {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        switch (self.state) {
            .filling => {
                self.state = .ready;
                self.cv.broadcast(io);
                return self.epoch;
            },
            .free, .reclaimable => return error.SlabNotFilling,
            .ready, .leased, .readers_complete => return error.SlabPublished,
        }
    }

    pub fn lease(self: *PageSlab) !Lease {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        switch (self.state) {
            .ready, .leased => {
                const id = self.next_id;
                try self.live.append(self.allocator, id);
                self.next_id += 1;
                self.state = .leased;
                return .{ .epoch = self.epoch, .id = id };
            },
            .free, .reclaimable => return error.SlabNotReady,
            .filling => return error.SlabFilling,
            .readers_complete => return error.SlabRetired,
        }
    }

    pub fn release(self: *PageSlab, held: Lease) !void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        if (held.epoch != self.epoch) return error.SlabEpochMismatch;
        switch (self.state) {
            .leased => {},
            .free, .filling, .ready, .readers_complete, .reclaimable => return error.SlabNotLeased,
        }
        for (self.live.items, 0..) |id, i| {
            if (id != held.id) continue;
            _ = self.live.swapRemove(i);
            if (self.live.items.len == 0) {
                self.state = .readers_complete;
                self.cv.broadcast(io);
            }
            return;
        }
        return error.SlabLeaseUnknown;
    }

    pub fn retire(self: *PageSlab) !void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        switch (self.state) {
            .ready, .readers_complete => {
                self.state = .reclaimable;
                self.cv.broadcast(io);
            },
            .reclaimable, .free => {},
            .filling => return error.SlabFilling,
            .leased => return error.SlabLeased,
        }
    }

    pub fn readerCount(self: *PageSlab) usize {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        return self.live.items.len;
    }
};

pub const ImportPayload = struct {
    released: std.atomic.Value(u32) = .init(0),
    epoch: u64 = 0,
};

pub const ImportedOperand = struct {
    array: mlx.mlx_array,
    aliased: bool,
    data_ptr: ?*const anyopaque,
    host_ptr: *const anyopaque,
};

pub var aliased_imports: std.atomic.Value(u64) = .init(0);
pub var fallback_imports: std.atomic.Value(u64) = .init(0);

fn payloadRelease(raw: ?*anyopaque) callconv(.c) void {
    const payload: *ImportPayload = @ptrCast(@alignCast(raw orelse return));
    _ = payload.released.fetchAdd(1, .release);
}

fn dtypeBytes(dtype: mlx.mlx_dtype) !usize {
    return switch (dtype) {
        .uint8, .int8, .bool_ => 1,
        .bfloat16, .float16, .uint16, .int16 => 2,
        .float32, .uint32, .int32 => 4,
        .float64, .uint64, .int64 => 8,
        else => error.UnsupportedImportDtype,
    };
}

fn dataPointerOf(array: mlx.mlx_array, dtype: mlx.mlx_dtype) ?*const anyopaque {
    return switch (dtype) {
        .bfloat16 => @ptrCast(mlx.mlx_array_data_bfloat16(array)),
        .uint8 => @ptrCast(mlx.mlx_array_data_uint8(array)),
        .uint32 => @ptrCast(mlx.mlx_array_data_uint32(array)),
        .float32 => @ptrCast(mlx.mlx_array_data_float32(array)),
        else => null,
    };
}

pub const ImportOptions = struct {
    copy: bool = false,
};

pub fn importHostBytes(
    bytes: []u8,
    shape: []const c_int,
    dtype: mlx.mlx_dtype,
    payload: *ImportPayload,
    opts: ImportOptions,
) !ImportedOperand {
    if (shape.len == 0) return error.InvalidImportShape;
    var elements: usize = 1;
    for (shape) |d| {
        if (d <= 0) return error.InvalidImportShape;
        elements *= @intCast(d);
    }
    if (elements * try dtypeBytes(dtype) != bytes.len) return error.ImportShapeMismatch;
    const array = if (opts.copy)
        mlx.mlx_array_new_data(@ptrCast(bytes.ptr), shape.ptr, @intCast(shape.len), dtype)
    else
        mlx.mlx_array_new_data_managed_payload(
            @ptrCast(bytes.ptr),
            shape.ptr,
            @intCast(shape.len),
            dtype,
            @ptrCast(payload),
            payloadRelease,
        );
    if (array.ctx == null) return error.ImportFailed;
    errdefer _ = mlx.mlx_array_free(array);
    if (mlx.mlx_array_eval(array) != 0) return error.ImportFailed;
    const got = dataPointerOf(array, dtype);
    const aliased = !opts.copy and got != null and @intFromPtr(got.?) == @intFromPtr(bytes.ptr);
    if (aliased) {
        _ = aliased_imports.fetchAdd(1, .monotonic);
    } else {
        _ = fallback_imports.fetchAdd(1, .monotonic);
        if (opts.copy) payloadRelease(@ptrCast(payload));
    }
    return .{ .array = array, .aliased = aliased, .data_ptr = got, .host_ptr = @ptrCast(bytes.ptr) };
}

pub fn importSlab(
    slab: *PageSlab,
    shape: []const c_int,
    dtype: mlx.mlx_dtype,
    payload: *ImportPayload,
    opts: ImportOptions,
) !ImportedOperand {
    if (slab.stateOf() == .filling) return error.SlabFilling;
    var elements: usize = 1;
    for (shape) |d| {
        if (d <= 0) return error.InvalidImportShape;
        elements *= @intCast(d);
    }
    const needed = elements * try dtypeBytes(dtype);
    if (needed > slab.bytes.len) return error.ImportShapeMismatch;
    payload.epoch = slab.currentEpoch();
    return importHostBytes(slab.bytes[0..needed], shape, dtype, payload, opts);
}

const TestFile = struct {
    dir: std.testing.TmpDir,
    path: []u8,
    bytes: []u8,

    fn make(allocator: std.mem.Allocator, name: []const u8, len: usize, seed: u64) !TestFile {
        const io = std.testing.io;
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const bytes = try allocator.alloc(u8, len);
        errdefer allocator.free(bytes);
        var prng = std.Random.DefaultPrng.init(seed);
        prng.random().bytes(bytes);
        try tmp.dir.writeFile(io, .{ .sub_path = name, .data = bytes });
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_len = try tmp.dir.realPath(io, &buf);
        const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ buf[0..dir_len], name });
        return .{ .dir = tmp, .path = path, .bytes = bytes };
    }

    fn deinit(self: *TestFile, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.bytes);
        self.dir.cleanup();
    }

    fn open(self: *const TestFile, allocator: std.mem.Allocator) !std.c.fd_t {
        const z = try std.fmt.allocPrintSentinel(allocator, "{s}", .{self.path}, 0);
        defer allocator.free(z);
        return openHinted(z, .{});
    }
};

fn alignedDest(allocator: std.mem.Allocator, len: usize) ![]align(align_bytes) u8 {
    const buf = try allocator.alignedAlloc(u8, .fromByteUnits(align_bytes), pageRoundUp(len));
    @memset(buf, 0);
    return buf;
}

test "expert io fill reads identical bytes for aligned and unaligned spans" {
    const t = std.testing;
    const page = pageSize();
    var file = try TestFile.make(t.allocator, "blob.bin", page * 6 + 777, 0x51ee);
    defer file.deinit(t.allocator);
    const fd = try file.open(t.allocator);
    defer _ = std.c.close(fd);

    const aligned = try alignedDest(t.allocator, page * 2);
    defer t.allocator.free(aligned);
    const unaligned_buf = try t.allocator.alloc(u8, 4096 + 1000);
    defer t.allocator.free(unaligned_buf);
    @memset(unaligned_buf, 0);
    const unaligned = unaligned_buf[3..];

    const spans = [_]FillSpan{
        .{ .file = 0, .offset = @intCast(page), .len = @intCast(page * 2), .dst = aligned.ptr },
        .{ .file = 0, .offset = @intCast(page * 4 + 13), .len = 1000, .dst = unaligned.ptr },
    };
    var plan = try FillPlan.init(t.allocator, &spans, 64 * 1024 * 1024, 2, false);
    defer plan.deinit();
    const pool = try FillPool.create(t.allocator, .{ .workers = 2 });
    defer pool.destroy();
    try pool.run(&.{fd}, &plan);

    try t.expectEqualSlices(u8, file.bytes[page .. page * 3], aligned[0 .. page * 2]);
    try t.expectEqualSlices(u8, file.bytes[page * 4 + 13 .. page * 4 + 13 + 1000], unaligned[0..1000]);
}

test "expert io fill coalesces only adjacent spans up to the bound" {
    const t = std.testing;
    var sink: [1]u8 = undefined;
    const input = [_]FillSpan{
        .{ .file = 0, .offset = 0, .len = 8, .dst = &sink },
        .{ .file = 0, .offset = 8, .len = 8, .dst = &sink },
        .{ .file = 0, .offset = 16, .len = 8, .dst = &sink },
        .{ .file = 1, .offset = 24, .len = 8, .dst = &sink },
    };
    const jobs = try coalesceSpans(t.allocator, &input, 16);
    defer t.allocator.free(jobs);
    try t.expectEqual(@as(usize, 3), jobs.len);
    try t.expectEqual(@as(u64, 16), jobs[0].len);
    try t.expectEqual(@as(usize, 2), jobs[0].count);
    try t.expectEqual(@as(u64, 16), jobs[1].offset);
    try t.expectEqual(@as(usize, 1), jobs[1].count);
    try t.expectEqual(@as(u16, 1), jobs[2].file);
}

test "expert io fill scatters one coalesced read into disjoint destinations" {
    const t = std.testing;
    var file = try TestFile.make(t.allocator, "pair.bin", 4096, 0x1234);
    defer file.deinit(t.allocator);
    const fd = try file.open(t.allocator);
    defer _ = std.c.close(fd);

    var left: [128]u8 = @splat(0);
    var right: [128]u8 = @splat(0);
    const spans = [_]FillSpan{
        .{ .file = 0, .offset = 512, .len = 128, .dst = &left },
        .{ .file = 0, .offset = 640, .len = 128, .dst = &right },
    };
    var plan = try FillPlan.init(t.allocator, &spans, 64 * 1024 * 1024, 1, false);
    defer plan.deinit();
    try t.expectEqual(@as(usize, 1), plan.jobs.len);
    const pool = try FillPool.create(t.allocator, .{ .workers = 1 });
    defer pool.destroy();
    try pool.run(&.{fd}, &plan);
    try t.expectEqualSlices(u8, file.bytes[512..640], &left);
    try t.expectEqualSlices(u8, file.bytes[640..768], &right);
}

test "expert io fill attributes a span past eof while other spans complete" {
    const t = std.testing;
    var file = try TestFile.make(t.allocator, "short.bin", 8192, 0xabc);
    defer file.deinit(t.allocator);
    const fd = try file.open(t.allocator);
    defer _ = std.c.close(fd);

    var good: [256]u8 = @splat(0);
    var bad: [256]u8 = @splat(0);
    const spans = [_]FillSpan{
        .{ .file = 0, .offset = 0, .len = 256, .dst = &good },
        .{ .file = 0, .offset = 8000, .len = 256, .dst = &bad },
    };
    var plan = try FillPlan.init(t.allocator, &spans, 64 * 1024 * 1024, 2, false);
    defer plan.deinit();
    const pool = try FillPool.create(t.allocator, .{ .workers = 2 });
    defer pool.destroy();
    try t.expectError(error.FillSpanPastEof, pool.run(&.{fd}, &plan));
    const failure = pool.firstFailure().?;
    try t.expectEqual(@as(u64, 8000), failure.offset);
    try t.expectEqual(@as(*u8, &bad[0]), @as(*u8, @ptrCast(failure.dst)));
    try t.expectEqualSlices(u8, file.bytes[0..256], &good);
}

const Canceller = struct {
    pool: *FillPool,
    fn main(self: *Canceller) void {
        var spins: usize = 0;
        while (self.pool.acceptedCount() == 0 and spins < 100_000_000) : (spins += 1) std.atomic.spinLoopHint();
        self.pool.requestCancel();
    }
};

test "expert io fill cancel drains accepted work and reports" {
    const t = std.testing;
    const chunk: usize = 16 * 1024 * 1024;
    var file = try TestFile.make(t.allocator, "big.bin", chunk * 4, 0x777);
    defer file.deinit(t.allocator);
    const fd = try file.open(t.allocator);
    defer _ = std.c.close(fd);

    const dst = try alignedDest(t.allocator, chunk * 4);
    defer t.allocator.free(dst);
    var spans: [4]FillSpan = undefined;
    for (&spans, 0..) |*s, i| {
        s.* = .{ .file = 0, .offset = @intCast(i * chunk), .len = @intCast(chunk), .dst = dst.ptr + i * chunk };
    }
    var plan = try FillPlan.init(t.allocator, &spans, @intCast(chunk), 1, false);
    defer plan.deinit();
    try t.expectEqual(@as(usize, 4), plan.jobs.len);
    const pool = try FillPool.create(t.allocator, .{ .workers = 1, .queue_capacity = 1 });
    defer pool.destroy();
    var canceller = Canceller{ .pool = pool };
    const thread = try std.Thread.spawn(.{}, Canceller.main, .{&canceller});
    const outcome = pool.run(&.{fd}, &plan);
    thread.join();
    try t.expectError(error.FillCancelled, outcome);
    try t.expectEqualSlices(u8, file.bytes[0..chunk], dst[0..chunk]);
    try t.expect(pool.stats().bytes_read < @as(u64, chunk) * 4);
}

test "expert io fill keeps a worker fed while it is reading" {
    const t = std.testing;
    const chunk: usize = 32 * 1024 * 1024;
    var file = try TestFile.make(t.allocator, "fed.bin", chunk * 2, 0x5eed);
    defer file.deinit(t.allocator);
    const fd = try file.open(t.allocator);
    defer _ = std.c.close(fd);

    const dst = try alignedDest(t.allocator, chunk * 2);
    defer t.allocator.free(dst);
    const spans = [_]FillSpan{
        .{ .file = 0, .offset = 0, .len = @intCast(chunk), .dst = dst.ptr },
        .{ .file = 0, .offset = @intCast(chunk), .len = @intCast(chunk), .dst = dst.ptr + chunk },
    };
    var plan = try FillPlan.init(t.allocator, &spans, @intCast(chunk), 1, false);
    defer plan.deinit();
    try t.expectEqual(@as(usize, 2), plan.jobs.len);
    const pool = try FillPool.create(t.allocator, .{ .workers = 1, .queue_capacity = 1 });
    defer pool.destroy();
    try pool.run(&.{fd}, &plan);
    try t.expectEqualSlices(u8, file.bytes, dst[0 .. chunk * 2]);
    try t.expectEqual(@as(usize, 2), pool.maxQueueDepth());
}

test "expert io file cache reopens a replaced shard and reuses a stable one" {
    const t = std.testing;
    const io = t.io;
    var file = try TestFile.make(t.allocator, "shard.bin", 64, 0x9);
    defer file.deinit(t.allocator);
    var cache = FileCache.init(t.allocator, .{});
    defer cache.deinit();
    const first = try cache.get(file.path);
    const again = try cache.get(file.path);
    try t.expectEqual(first, again);
    try t.expectEqual(@as(u64, 1), cache.opens);
    try t.expectEqual(@as(u64, 1), cache.hits);

    const replacement = try t.allocator.alloc(u8, 128);
    defer t.allocator.free(replacement);
    @memset(replacement, 7);
    try file.dir.dir.writeFile(io, .{ .sub_path = "shard.bin", .data = replacement });
    const third = try cache.get(file.path);
    try t.expectEqual(@as(u64, 1), cache.revalidations);
    var got: [128]u8 = undefined;
    const r = FillPool.preadInto(third, &got, 0, 128);
    try t.expectEqual(@as(usize, 128), r.done);
    try t.expectEqualSlices(u8, replacement, &got);
}

pub const Dtype = enum { bf16, u32, other };

pub const TensorRegion = struct {
    data_offset: u64,
    tensor_offset: u64,
    tensor_bytes: u64,
    shape: [3]u64,
    rank: u8,
    dtype: Dtype,
};

pub fn readExact(fd: std.c.fd_t, dst: []u8, offset: u64) !void {
    var done: usize = 0;
    while (done < dst.len) {
        const got = std.c.pread(fd, dst[done..].ptr, dst.len - done, @intCast(offset + done));
        if (got < 0) {
            if (std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
            return error.FillReadFailed;
        }
        if (got == 0) return error.FillShortRead;
        done += @intCast(got);
    }
}

pub fn tensorRegion(allocator: std.mem.Allocator, fd: std.c.fd_t, key: []const u8) !TensorRegion {
    var len_bytes: [8]u8 = undefined;
    try readExact(fd, &len_bytes, 0);
    const header_len = std.mem.readInt(u64, &len_bytes, .little);
    if (header_len == 0 or header_len > 128 * 1024 * 1024) return error.InvalidSafetensorsHeader;
    const header = try allocator.alloc(u8, @intCast(header_len));
    defer allocator.free(header);
    try readExact(fd, header, 8);
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, header, .{}) catch return error.InvalidSafetensorsHeader;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSafetensorsHeader;
    const value = parsed.value.object.get(key) orelse return error.MissingSafetensorsTensor;
    if (value != .object) return error.InvalidSafetensorsTensor;
    const object = value.object;
    const dtype = object.get("dtype") orelse return error.InvalidSafetensorsTensor;
    if (dtype != .string) return error.InvalidSafetensorsTensor;
    const dt: Dtype = if (std.mem.eql(u8, dtype.string, "BF16"))
        .bf16
    else if (std.mem.eql(u8, dtype.string, "U32") or std.mem.eql(u8, dtype.string, "UINT32"))
        .u32
    else
        .other;
    const shape = object.get("shape") orelse return error.InvalidSafetensorsTensor;
    if (shape != .array or shape.array.items.len < 2 or shape.array.items.len > 3) return error.InvalidSafetensorsTensor;
    var dimensions: [3]u64 = .{ 0, 0, 0 };
    var elements: u64 = 1;
    for (shape.array.items, 0..) |dim, i| {
        if (dim != .integer or dim.integer <= 0) return error.InvalidSafetensorsTensor;
        dimensions[i] = @intCast(dim.integer);
        elements = std.math.mul(u64, elements, dimensions[i]) catch return error.InvalidSafetensorsTensor;
    }
    const offsets = object.get("data_offsets") orelse return error.InvalidSafetensorsTensor;
    if (offsets != .array or offsets.array.items.len != 2) return error.InvalidSafetensorsTensor;
    const start = offsets.array.items[0];
    const end = offsets.array.items[1];
    if (start != .integer or end != .integer or start.integer < 0 or end.integer < start.integer) return error.InvalidSafetensorsTensor;
    const start_u: u64 = @intCast(start.integer);
    const end_u: u64 = @intCast(end.integer);
    const bytes = end_u - start_u;
    const data_offset = std.math.add(u64, 8, header_len) catch return error.InvalidSafetensorsTensor;
    var st: std.c.Stat = undefined;
    const absolute_end = std.math.add(u64, data_offset, end_u) catch return error.InvalidSafetensorsTensor;
    if (std.c.fstat(fd, &st) != 0 or st.size < 0 or absolute_end > @as(u64, @intCast(st.size))) return error.SafetensorsTensorOutOfBounds;
    return .{
        .data_offset = data_offset,
        .tensor_offset = start_u,
        .tensor_bytes = bytes,
        .shape = dimensions,
        .rank = @intCast(shape.array.items.len),
        .dtype = dt,
    };
}

pub fn expertSpanOf(region: TensorRegion, expert: u64) !struct { offset: u64, len: u64 } {
    const experts = region.shape[0];
    if (experts == 0 or expert >= experts or region.tensor_bytes % experts != 0) return error.InvalidExpertTensor;
    const per = region.tensor_bytes / experts;
    return .{ .offset = region.data_offset + region.tensor_offset + expert * per, .len = per };
}

pub const LayerTensors = struct {
    gate_up_file: u16,
    down_file: u16,
    gate_up: TensorRegion,
    down: TensorRegion,
};

pub const ShardIndex = struct {
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    parsed: std.json.Parsed(std.json.Value),

    pub fn open(allocator: std.mem.Allocator, model_dir: []const u8) !ShardIndex {
        const io = std.Io.Threaded.global_single_threaded.io();
        var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
        defer dir.close(io);
        const raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 * 1024 * 1024));
        defer allocator.free(raw);
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch return error.InvalidSafetensorsIndex;
        errdefer parsed.deinit();
        if (parsed.value != .object) return error.InvalidSafetensorsIndex;
        if (parsed.value.object.get("weight_map") == null) return error.InvalidSafetensorsIndex;
        return .{ .allocator = allocator, .model_dir = model_dir, .parsed = parsed };
    }

    pub fn deinit(self: *ShardIndex) void {
        self.parsed.deinit();
        self.* = undefined;
    }

    pub fn shardOf(self: *const ShardIndex, key: []const u8) ![]const u8 {
        const map = self.parsed.value.object.get("weight_map").?.object;
        const v = map.get(key) orelse return error.MissingExpertTensor;
        if (v != .string) return error.InvalidSafetensorsIndex;
        return v.string;
    }

    pub fn layerTensors(
        self: *const ShardIndex,
        layer: u16,
        cache: *FileCache,
        files: *std.ArrayList(std.c.fd_t),
    ) !LayerTensors {
        var key_buf: [192]u8 = undefined;
        const gate_key = try std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.mlp.experts.gate_up_proj", .{layer});
        const gate_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.model_dir, try self.shardOf(gate_key) });
        defer self.allocator.free(gate_path);
        const gate_fd = try cache.get(gate_path);
        const gate_index = try fileIndex(self.allocator, files, gate_fd);
        const gate = try tensorRegion(self.allocator, gate_fd, gate_key);

        var down_buf: [192]u8 = undefined;
        const down_key = try std.fmt.bufPrint(&down_buf, "model.language_model.layers.{d}.mlp.experts.down_proj", .{layer});
        const down_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.model_dir, try self.shardOf(down_key) });
        defer self.allocator.free(down_path);
        const down_fd = try cache.get(down_path);
        const down_index = try fileIndex(self.allocator, files, down_fd);
        const down = try tensorRegion(self.allocator, down_fd, down_key);
        return .{ .gate_up_file = gate_index, .down_file = down_index, .gate_up = gate, .down = down };
    }
};

fn fileIndex(allocator: std.mem.Allocator, files: *std.ArrayList(std.c.fd_t), fd: std.c.fd_t) !u16 {
    for (files.items, 0..) |f, i| {
        if (f == fd) return @intCast(i);
    }
    try files.append(allocator, fd);
    return @intCast(files.items.len - 1);
}

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;

fn processDiskBytesRead() u64 {
    var buf: [1024]u8 align(8) = @splat(0);
    if (proc_pid_rusage(std.c.getpid(), 2, @ptrCast(&buf)) != 0) return 0;
    return std.mem.readInt(u64, buf[144..152], .little);
}

const BenchArm = struct {
    workers: usize,
    force_bounce: bool,
    nocache: bool,
};

fn benchPrint(comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = std.c.write(2, line.ptr, line.len);
}

fn unixSeconds() f64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const ts = std.Io.Timestamp.now(io, .real);
    return @as(f64, @floatFromInt(@as(i64, @intCast(@divTrunc(ts.nanoseconds, 1000))))) / 1_000_000.0;
}

const LayerPlan = struct {
    gate_path: usize,
    down_path: usize,
    gate: TensorRegion,
    down: TensorRegion,
};

test "expert io ssd microbench" {
    const model_dir = std.mem.span(std.c.getenv("QWEN4_BF16_STREAM_MODEL") orelse return error.SkipZigTest);
    const allocator = std.heap.page_allocator;
    const arm_bytes: u64 = 8 * 1024 * 1024 * 1024;
    const layer_count: u16 = 48;

    var index = try ShardIndex.open(allocator, model_dir);
    defer index.deinit();

    var paths: std.ArrayList([]u8) = .empty;
    defer {
        for (paths.items) |x| allocator.free(x);
        paths.deinit(allocator);
    }
    const pathIndex = struct {
        fn call(a: std.mem.Allocator, list: *std.ArrayList([]u8), path: []const u8) !usize {
            for (list.items, 0..) |p, i| {
                if (std.mem.eql(u8, p, path)) return i;
            }
            try list.append(a, try a.dupe(u8, path));
            return list.items.len - 1;
        }
    }.call;

    var probe_cache = FileCache.initCapacity(allocator, .{}, 256);
    defer probe_cache.deinit();
    var layer_plans: [64]LayerPlan = undefined;
    var key_buf: [192]u8 = undefined;
    var l: u16 = 0;
    while (l < layer_count) : (l += 1) {
        const gate_key = try std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.mlp.experts.gate_up_proj", .{l});
        const gate_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, try index.shardOf(gate_key) });
        defer allocator.free(gate_path);
        const gate_fd = try probe_cache.get(gate_path);
        const gate = try tensorRegion(allocator, gate_fd, gate_key);
        var down_buf: [192]u8 = undefined;
        const down_key = try std.fmt.bufPrint(&down_buf, "model.language_model.layers.{d}.mlp.experts.down_proj", .{l});
        const down_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, try index.shardOf(down_key) });
        defer allocator.free(down_path);
        const down_fd = try probe_cache.get(down_path);
        const down = try tensorRegion(allocator, down_fd, down_key);
        layer_plans[l] = .{
            .gate_path = try pathIndex(allocator, &paths, gate_path),
            .down_path = try pathIndex(allocator, &paths, down_path),
            .gate = gate,
            .down = down,
        };
    }

    const experts: u64 = layer_plans[0].gate.shape[0];
    const gate_bytes = layer_plans[0].gate.tensor_bytes / experts;
    const down_bytes = layer_plans[0].down.tensor_bytes / experts;
    const pair_bytes = gate_bytes + down_bytes;
    benchPrint("[ssd-bench] experts={d} layers={d} shards={d} gate_up={d} down={d} pair={d} page={d}\n", .{ experts, layer_count, paths.items.len, gate_bytes, down_bytes, pair_bytes, pageSize() });

    var arms: std.ArrayList(BenchArm) = .empty;
    defer arms.deinit(allocator);
    for ([_]bool{ false, true }) |bounce| {
        for ([_]usize{ 1, 2, 4, 8, 16 }) |w| {
            try arms.append(allocator, .{ .workers = w, .force_bounce = bounce, .nocache = true });
        }
    }
    try arms.append(allocator, .{ .workers = 8, .force_bounce = false, .nocache = false });
    try arms.append(allocator, .{ .workers = 8, .force_bounce = true, .nocache = false });

    var prng = std.Random.DefaultPrng.init(@intCast(@as(i64, @intCast(@divTrunc(std.Io.Timestamp.now(std.Io.Threaded.global_single_threaded.io(), .real).nanoseconds, 1)))));
    const rand = prng.random();
    const Pick = struct { layer: u16, expert: u16 };
    var pool_picks: std.ArrayList(Pick) = .empty;
    defer pool_picks.deinit(allocator);
    l = 0;
    while (l < layer_count) : (l += 1) {
        var e: u16 = 0;
        while (e < experts) : (e += 1) try pool_picks.append(allocator, .{ .layer = l, .expert = @intCast(e) });
    }
    rand.shuffle(Pick, pool_picks.items);
    const pairs_per_arm: usize = @intCast(arm_bytes / pair_bytes);
    if (pool_picks.items.len < pairs_per_arm * arms.items.len) return error.NotEnoughDistinctExperts;

    const dest_len: usize = @intCast(pageRoundUp(arm_bytes + pair_bytes));
    const dest = try std.posix.mmap(null, dest_len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(dest);
    @memset(dest[0..1], 0);

    var cursor: usize = 0;
    for (arms.items) |arm| {
        var cache = FileCache.initCapacity(allocator, .{ .nocache = arm.nocache, .readahead_off = arm.nocache }, paths.items.len);
        defer cache.deinit();
        var files = try allocator.alloc(std.c.fd_t, paths.items.len);
        defer allocator.free(files);
        for (paths.items, 0..) |p, i| files[i] = try cache.get(p);

        var spans: std.ArrayList(FillSpan) = .empty;
        defer spans.deinit(allocator);
        var dst_off: usize = 0;
        var requested: u64 = 0;
        var i: usize = 0;
        while (i < pairs_per_arm) : (i += 1) {
            const pick = pool_picks.items[cursor + i];
            const lp = layer_plans[pick.layer];
            const g = try expertSpanOf(lp.gate, pick.expert);
            const d = try expertSpanOf(lp.down, pick.expert);
            const g_off = if (arm.force_bounce) g.offset else pageRoundDown(g.offset);
            const d_off = if (arm.force_bounce) d.offset else pageRoundDown(d.offset);
            try spans.append(allocator, .{ .file = @intCast(lp.gate_path), .offset = g_off, .len = g.len, .dst = dest.ptr + dst_off });
            dst_off += @intCast(g.len);
            try spans.append(allocator, .{ .file = @intCast(lp.down_path), .offset = d_off, .len = d.len, .dst = dest.ptr + dst_off });
            dst_off += @intCast(d.len);
            requested += g.len + d.len;
        }
        cursor += pairs_per_arm;

        var plan = try FillPlan.init(allocator, spans.items, 64 * 1024 * 1024, arm.workers, arm.force_bounce);
        defer plan.deinit();
        var direct_jobs: usize = 0;
        for (plan.jobs) |j| {
            if (j.direct) direct_jobs += 1;
        }
        const pool = try FillPool.create(allocator, .{ .workers = arm.workers, .force_bounce = arm.force_bounce });
        defer pool.destroy();
        pool.resetStats();
        const phys0 = processDiskBytesRead();
        const t0 = unixSeconds();
        try pool.run(files, &plan);
        const t1 = unixSeconds();
        const phys1 = processDiskBytesRead();
        const st = pool.stats();
        const secs = t1 - t0;
        const phys = phys1 - phys0;
        benchPrint("[ssd-bench] arm workers={d} mode={s} nocache={d} jobs={d} direct_jobs={d} bytes={d} secs={d:.3} app_GBs={d:.2} phys_bytes={d} phys_GBs={d:.2} pread_ms={d:.1} reads={d} imbalance={d:.2} t0={d:.3} t1={d:.3}\n", .{
            arm.workers,
            if (arm.force_bounce) "bounce" else "direct",
            @as(u8, if (arm.nocache) 1 else 0),
            plan.jobs.len,
            direct_jobs,
            requested,
            secs,
            @as(f64, @floatFromInt(requested)) / secs / 1_000_000_000.0,
            phys,
            @as(f64, @floatFromInt(phys)) / secs / 1_000_000_000.0,
            @as(f64, @floatFromInt(st.pread_ns)) / 1_000_000.0,
            st.reads_issued,
            st.imbalance(arm.workers),
            t0,
            t1,
        });
    }

    var seq_cache = FileCache.init(allocator, .{});
    defer seq_cache.deinit();
    const seq_fd = try seq_cache.get(paths.items[paths.items.len - 1]);
    var seq_st: std.c.Stat = undefined;
    _ = std.c.fstat(seq_fd, &seq_st);
    const shard_size: u64 = @intCast(seq_st.size);
    const run_bytes: u64 = 64 * 1024 * 1024;
    if (shard_size > run_bytes * 4) {
        const runs: usize = @intCast(@min(arm_bytes, shard_size - run_bytes) / run_bytes);
        var seq_spans: std.ArrayList(FillSpan) = .empty;
        defer seq_spans.deinit(allocator);
        var used: u64 = 0;
        var r: usize = 0;
        while (r < runs) : (r += 1) {
            try seq_spans.append(allocator, .{ .file = 0, .offset = used, .len = run_bytes, .dst = dest.ptr + @as(usize, @intCast(used)) });
            used += run_bytes;
        }
        var seq_plan = try FillPlan.init(allocator, seq_spans.items, run_bytes, 8, false);
        defer seq_plan.deinit();
        const seq_pool = try FillPool.create(allocator, .{ .workers = 8 });
        defer seq_pool.destroy();
        seq_pool.resetStats();
        const phys0 = processDiskBytesRead();
        const t0 = unixSeconds();
        try seq_pool.run(&.{seq_fd}, &seq_plan);
        const t1 = unixSeconds();
        const phys1 = processDiskBytesRead();
        const st = seq_pool.stats();
        benchPrint("[ssd-bench] arm workers=8 mode=seq64 nocache=1 jobs={d} bytes={d} secs={d:.3} app_GBs={d:.2} phys_bytes={d} phys_GBs={d:.2} pread_ms={d:.1} reads={d} t0={d:.3} t1={d:.3}\n", .{
            seq_plan.jobs.len,
            used,
            t1 - t0,
            @as(f64, @floatFromInt(used)) / (t1 - t0) / 1_000_000_000.0,
            phys1 - phys0,
            @as(f64, @floatFromInt(phys1 - phys0)) / (t1 - t0) / 1_000_000_000.0,
            @as(f64, @floatFromInt(st.pread_ns)) / 1_000_000.0,
            st.reads_issued,
            t0,
            t1,
        });
    }
}


test "expert io slab blocks a refill while a reader holds the epoch" {
    const t = std.testing;
    const slab = try PageSlab.create(t.allocator, 4096);
    defer slab.destroy();
    _ = try slab.tryBeginFill();
    const w = try slab.writable();
    @memset(w[0..16], 0xAB);
    const epoch = try slab.publish();
    const held = try slab.lease();
    try t.expectEqual(epoch, held.epoch);
    try t.expectEqual(SlabState.leased, slab.stateOf());
    try t.expectError(error.SlabLeased, slab.tryBeginFill());
    try t.expectError(error.SlabRefillTimeout, slab.beginFillTimeout(20 * std.time.ns_per_ms));
    try t.expectError(error.SlabLeased, slab.retire());
    try slab.release(held);
    try t.expectEqual(SlabState.readers_complete, slab.stateOf());
    try slab.retire();
    const next = try slab.beginFillTimeout(20 * std.time.ns_per_ms);
    try t.expectEqual(epoch + 1, next);
}

test "expert io slab releases leases out of order and rejects a double release" {
    const t = std.testing;
    const slab = try PageSlab.create(t.allocator, 4096);
    defer slab.destroy();
    _ = try slab.tryBeginFill();
    _ = try slab.publish();
    const a = try slab.lease();
    const b = try slab.lease();
    const c = try slab.lease();
    try t.expectEqual(@as(usize, 3), slab.readerCount());
    try slab.release(b);
    try slab.release(a);
    try t.expectEqual(SlabState.leased, slab.stateOf());
    try t.expectError(error.SlabLeaseUnknown, slab.release(b));
    try slab.release(c);
    try t.expectEqual(SlabState.readers_complete, slab.stateOf());
    try t.expectError(error.SlabNotLeased, slab.release(c));
}

test "expert io slab names an error for every illegal transition" {
    const t = std.testing;
    const slab = try PageSlab.create(t.allocator, 4096);
    defer slab.destroy();

    try t.expectEqual(SlabState.free, slab.stateOf());
    try t.expectError(error.SlabNotFilling, slab.writable());
    try t.expectError(error.SlabNotFilling, slab.publish());
    try t.expectError(error.SlabNotReady, slab.lease());
    try slab.retire();

    _ = try slab.tryBeginFill();
    try t.expectEqual(SlabState.filling, slab.stateOf());
    try t.expectError(error.SlabFilling, slab.tryBeginFill());
    try t.expectError(error.SlabFilling, slab.lease());
    try t.expectError(error.SlabFilling, slab.retire());

    _ = try slab.publish();
    try t.expectEqual(SlabState.ready, slab.stateOf());
    try t.expectError(error.SlabPublished, slab.writable());
    try t.expectError(error.SlabPublished, slab.publish());
    try t.expectError(error.SlabPublished, slab.tryBeginFill());

    const held = try slab.lease();
    try t.expectError(error.SlabPublished, slab.writable());
    try t.expectError(error.SlabEpochMismatch, slab.release(.{ .epoch = held.epoch + 7, .id = held.id }));
    try slab.release(held);

    try t.expectEqual(SlabState.readers_complete, slab.stateOf());
    try t.expectError(error.SlabRetired, slab.lease());
    try t.expectError(error.SlabPublished, slab.tryBeginFill());

    try slab.retire();
    try t.expectEqual(SlabState.reclaimable, slab.stateOf());
    try t.expectError(error.SlabNotReady, slab.lease());
    try t.expectError(error.SlabNotFilling, slab.writable());
    _ = try slab.tryBeginFill();
}

test "expert io import aliases a page slab and sees the bytes written while filling" {
    const t = std.testing;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const elements: usize = 512;
    const slab = try PageSlab.create(t.allocator, elements * 2);
    defer slab.destroy();
    _ = try slab.tryBeginFill();
    const w = try slab.writable();
    const one_bf16: u16 = 0x3F80;
    var i: usize = 0;
    while (i < elements) : (i += 1) std.mem.writeInt(u16, w[i * 2 ..][0..2], one_bf16, .little);
    _ = try slab.publish();

    var payload: ImportPayload = .{};
    const shape = [_]c_int{@intCast(elements)};
    const before = aliased_imports.load(.monotonic);
    const operand = try importSlab(slab, &shape, .bfloat16, &payload, .{});
    defer _ = mlx.mlx_array_free(operand.array);
    if (!operand.aliased) {
        benchPrint("[expert-io] import did not alias: slab {x} mlx {x}\n", .{ @intFromPtr(operand.host_ptr), @intFromPtr(operand.data_ptr orelse @as(*const anyopaque, operand.host_ptr)) });
    }
    try t.expect(operand.aliased);
    try t.expectEqual(before + 1, aliased_imports.load(.monotonic));

    const s = mlx.gpuStream();
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try t.expectEqual(@as(c_int, 0), mlx.mlx_sum(&sum, operand.array, false, s));
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try t.expectEqual(@as(c_int, 0), mlx.mlx_astype(&wide, sum, .float32, s));
    var out: f32 = 0;
    _ = mlx.mlx_array_eval(wide);
    _ = mlx.mlx_array_item_float32(&out, wide);
    try t.expectApproxEqAbs(@as(f32, @floatFromInt(elements)), out, 1.0);
}

test "expert io import counts a fallback copy and still yields the bytes" {
    const t = std.testing;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const elements: usize = 256;
    const slab = try PageSlab.create(t.allocator, 4096 + elements * 2);
    defer slab.destroy();
    _ = try slab.tryBeginFill();
    const w = try slab.writable();
    const unaligned = w[2 .. 2 + elements * 2];
    const one_bf16: u16 = 0x3F80;
    var i: usize = 0;
    while (i < elements) : (i += 1) std.mem.writeInt(u16, unaligned[i * 2 ..][0..2], one_bf16, .little);
    _ = try slab.publish();

    var payload: ImportPayload = .{};
    const shape = [_]c_int{@intCast(elements)};
    const before = fallback_imports.load(.monotonic);
    const operand = try importHostBytes(unaligned, &shape, .bfloat16, &payload, .{ .copy = true });
    defer _ = mlx.mlx_array_free(operand.array);
    try t.expect(!operand.aliased);
    try t.expectEqual(before + 1, fallback_imports.load(.monotonic));
    try t.expectEqual(@as(u32, 1), payload.released.load(.acquire));

    const s = mlx.gpuStream();
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try t.expectEqual(@as(c_int, 0), mlx.mlx_sum(&sum, operand.array, false, s));
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try t.expectEqual(@as(c_int, 0), mlx.mlx_astype(&wide, sum, .float32, s));
    var out: f32 = 0;
    _ = mlx.mlx_array_eval(wide);
    _ = mlx.mlx_array_item_float32(&out, wide);
    try t.expectApproxEqAbs(@as(f32, @floatFromInt(elements)), out, 1.0);
}

test "expert io import runs the payload release callback exactly once" {
    const t = std.testing;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const elements: usize = 128;
    const slab = try PageSlab.create(t.allocator, elements * 2);
    defer slab.destroy();
    _ = try slab.tryBeginFill();
    _ = try slab.publish();
    var payload: ImportPayload = .{};
    const shape = [_]c_int{@intCast(elements)};
    const operand = try importSlab(slab, &shape, .bfloat16, &payload, .{});
    try t.expectEqual(@as(u32, 0), payload.released.load(.acquire));
    _ = mlx.mlx_array_free(operand.array);
    const s = mlx.gpuStream();
    _ = mlx.mlx_synchronize(s);
    try t.expectEqual(@as(u32, 1), payload.released.load(.acquire));
    try t.expectEqual(slab.currentEpoch(), payload.epoch);
}

test "expert io import active memory counts an aliased slab once" {
    const t = std.testing;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const bytes: usize = 4 * 1024 * 1024;
    const slab = try PageSlab.create(t.allocator, bytes);
    defer slab.destroy();
    _ = try slab.tryBeginFill();
    _ = try slab.publish();
    var payload: ImportPayload = .{};
    const shape = [_]c_int{@intCast(bytes / 2)};
    var before: usize = 0;
    _ = mlx.mlx_get_active_memory(&before);
    const operand = try importSlab(slab, &shape, .bfloat16, &payload, .{});
    var after: usize = 0;
    _ = mlx.mlx_get_active_memory(&after);
    defer _ = mlx.mlx_array_free(operand.array);
    const delta = after -| before;
    benchPrint("[expert-io] aliased import active memory delta {d} for {d} bytes\n", .{ delta, bytes });
    try t.expect(operand.aliased);
    try t.expect(delta < 2 * bytes);
}
