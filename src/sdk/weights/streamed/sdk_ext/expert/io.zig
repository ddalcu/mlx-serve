//! Lookahead read pool (lib/expert_io/q3_lookahead4_exl3.c). Its demand path is
//! the native-issue pool's: worker threads read each record's gate/up span,
//! then its down span (page-aligned F_NOCACHE preadv into a staging buffer, then
//! a scatter copy into slot rows) and publish every range's status words, then
//! its ticket in an ordered log. Three classes ride on it, each chosen at
//! construction: speculative whole-record reads (a demand submit claims the
//! record holding its range and copies it out), pre-read ranges (queued before
//! a plan, bound by the submit that needs them) and event gates (the satisfied
//! prefix goes to an MTLSharedEvent or a host word; a watchdog forces a gate
//! whose bytes never land). Workers never call MLX. The C state is static: one
//! pool per process, stopped before the next.
//!
//! A source reads through one record topology fixed when it is built (`Records(components, gate_up)`: EXL3 (9, 6),
//! MXFP4 (6, 4)), from an fd `openUncached` made: there is no second submit path and no page-cached fd.

const std = @import("std");
const builtin = @import("builtin");
const io_util = @import("sdk").io_util;
const policy = @import("policy.zig");

// 1:1 mirror of lib/expert_io/q3_lookahead4.h. The C pool and the event shims compile on macOS graphs only (the host's
// `macos_engines` graphs: the macOS exe and tests); the Linux exe and the iOS lib get the refusing stand-ins, whose
// q3ld_abi answers 0 so Pool.start refuses before any read. A macOS artifact that does not link the C objects (the
// SDK's own tests) references no Pool. `macos_engines` stays the host's switch for its own engines.
const c = if (builtin.os.tag == .macos) struct {
    pub extern fn q3ld_spec_config(nthreads: i32, bufs: ?[*]const u64, nslots: i32, slot_bytes: i64, rec_len: i64, chunk: i64, counters: ?[*]i64) c_int;
    pub extern fn q3ld_spec_streams(idle_busy: i32) c_int;
    pub extern fn q3ld_direct_config(on: i32) c_int;
    pub extern fn q3ld_start(nw: i32, staging_ptrs: [*]const u64, sbytes: i64, psize: i64, res: [*]i64, n_tickets: i64, log: [*]i64, n_log: i64, gauge: *[6]i64) c_int;
    pub extern fn q3ld_submit(fd: i32, file_size: i64, deadline: i64, n: i32, ngu: i32, ndown: i32, offsets: [*]const i64, rows: [*]const [*]const u64, lens: [*]const i64, first: i64) c_int;
    pub extern fn q3ld_submit_scatter(n: i32, ngu: i32, ndown: i32, fds: [*]const i32, sizes: [*]const i64, offsets: [*]const i64, rows: [*]const [*]const u64, lens: [*]const i64, first: i64) c_int;
    pub extern fn q3ld_withdraw(first: i64) c_int;
    pub extern fn q3ld_spec_step_len(fd: i32, file_size: i64, cur: i64, n: i32, bases: ?[*]const i64, len: i64) i32;
    pub extern fn q3ld_spec_step_lens(fd: i32, file_size: i64, cur: i64, n: i32, bases: ?[*]const i64, lens: ?[*]const i64) i32;
    pub extern fn q3ld_spec_state(out: *[spec_state_w * max_spec]i64) i32;
    pub extern fn q3ld_seq() i64;
    pub extern fn q3ld_wait(seen: i64, timeout_ns: i64) i64;
    pub extern fn q3ld_gauge(out: *[6]i64) void;
    pub extern fn q3ld_quiesce(timeout_ns: i64) c_int;
    pub extern fn q3ld_stop() c_int;
    pub extern fn q3ld_pre_config(ngu: i32, ndown: i32, lens: ?[*]const i64) c_int;
    pub extern fn q3ld_pre_read_lens(fd: i32, file_size: i64, tag: i64, n: i32, bases: [*]const i64, lens: [*]const i64) i32;
    pub extern fn q3ld_pre_state(out: *[pre_state_w * max_pre]i64) i32;
    pub extern fn q3ld_ev_config(kind: i32, obj: u64, timeout_ns: i64, start: u64) c_int;
    pub extern fn q3ld_ev_gates(n: i32, values: [*]const u64, counts: [*]const i32, tickets: [*]const i64) i32;
    pub extern fn q3ld_ev_release(value: u64) i32;
    pub extern fn q3ld_ev_state(out: *[10]i64) i32;
    pub extern fn q3ld_monotonic_ns() i64;
    pub extern fn q3ld_abi() i32;
    pub extern fn q3ld_counters_n() i32;
    pub extern fn q3ld_max_spec() i32;
    pub extern fn q3ld_max_pre() i32;
    pub extern fn q3ld_max_gates() i32;
    pub extern fn q3ld_max_gate_tickets() i32;
    /// Q3LD_INJECT builds (the test module) only.
    pub extern fn q3ld_test_rules(n: i32, off: [*]const i64, code: [*]const i64, arg: [*]const i64) void;
    pub extern fn q3ld_test_delay(seed: u64, max_ns: i64) void;
    pub extern fn q3ld_test_ev_log(buf: ?[*]i64, cap: i64) i64;
} else @import("io_stub.zig").q3ld;

/// Test builds only: the reader's C ABI and its status-word stride, for its own tests (its state words and scripted
/// faults). Served code goes through the Pool.
pub const test_abi = if (builtin.is_test) struct {
    pub const abi = c;
    pub const status_words = res_w;
} else struct {};

/// The C pool's ABI (`q3ld_abi`): 2026101401, DIRECT reads as a construction-time route (2026101201: the submit-wide
/// WITHDRAW over the range-job queue).
pub const abi_version = 2026102401;
pub const max_workers = 8;
/// Records per job (one fill unit).
pub const max_items = 8;
pub const max_spec = 16;
pub const max_spec_threads = 3;
pub const max_pre = 32;
/// Live gates (the C pool's MAX_GATES): every route a stream holds at once, each the gate/up gate and a down gate per
/// part (the stream asserts the bound).
pub const max_gates = 512;
/// Tickets of one gate registration (the C pool's MAX_GATE_TICKETS; the start's ABI check ties them): a gate/up and a
/// down ticket per record of the widest route, its ids each of at most `max_items` minis.
pub const max_gate_tickets = 2 * policy.max_route_ids * max_items;
/// Status words per ticket (`Result`'s fields, in order).
const res_w = 8;
pub const spec_state_w = 11;
pub const pre_state_w = 5;
/// `q3ld_submit` reads -1 as no deadline; 0 would expire every range at once.
const no_deadline: i64 = -1;

/// `withdrawn`: the job left the queue unread (`Pool.withdraw`): no preadv, no row written.
pub const Status = enum(i64) { ok = 0, short = 1, os_error = 2, deadline = 3, skipped = 4, withdrawn = 5, pending = -1, _ };

/// One range's status words, as the worker published them.
pub const Result = struct {
    status: Status,
    preadv_calls: i64,
    bytes_returned: i64,
    payload: i64,
    errno: i64,
    t_start_ns: i64,
    t_end_ns: i64,
    worker: i64,
};

/// The pool's counter words (q3_lookahead4_exl3.c `SC_*`), written under its
/// mutex; the h_* entries are the first of four horizon classes.
pub const Counter = enum(u8) {
    submitted = 0,
    refreshed = 1,
    noslot = 2,
    started = 3,
    landed = 4,
    failed = 5,
    expired = 6,
    cancelled_by_demand = 7,
    claimed = 8,
    claimed_inflight = 9,
    abandoned = 10,
    abandoned_bytes = 11,
    discarded = 12,
    adopt_ranges = 13,
    adopt_bytes = 14,
    adopt_waits = 15,
    adopt_wait_ns = 16,
    spec_bytes = 17,
    spec_chunks = 18,
    pauses = 19,
    max_busy_at_start = 20,
    max_qlen_at_start = 21,
    parks = 22,
    max_nearer_at_start = 23,
    h_submitted = 24,
    h_kept = 28,
    h_dropped = 32,
    h_claimed = 36,
    h_adopt_ranges = 40,
    idle_chunks = 44,
    max_busy_at_start_idle = 45,
    pre_calls = 46,
    pre_issued = 47,
    pre_noslot = 48,
    pre_claims = 49,
    pre_started = 50,
    pre_bound = 51,
    pre_cancelled = 52,
    pre_expired = 53,
    pre_served = 54,
    pre_waits = 55,
    pre_wait_ns = 56,
    pre_bind_wait_ns = 57,
    pre_max_inflight = 58,
    ev_calls = 59,
    ev_gates = 60,
    ev_tickets = 61,
    ev_immediate = 62,
    ev_signals = 63,
    ev_signal_ns = 64,
    ev_lag_ns = 65,
    ev_max_live = 66,
    ev_wd_forced = 67,
    ev_wd_last_value = 68,
    ev_host_released = 69,
    ev_stop_released = 70,
    direct_ranges = 71,
};
pub const counters_n = 72;

/// The speculative class: `slots` staging slots of `slotBytes(record_bytes)`,
/// read by `threads` threads in `chunk_bytes` preadv steps.
pub const Spec = struct {
    threads: u32,
    slots: u32,
    /// The largest record's payload; a speculative read covers one record.
    record_bytes: u64,
    chunk_bytes: u64,
    /// Queue rule of speculative threads >= 1: an unclaimed chunk starts only
    /// while at most this many demand jobs execute (0 = demand idle).
    idle_busy: u32 = 0,
};

pub const Options = struct {
    workers: u32 = 4,
    /// One page-aligned staging buffer per worker; 9 MiB holds a whole
    /// 8,877,056-byte gate/up span plus its page alignment. A stream over a bank with
    /// wider spans sizes it from the bank (`stream.Options.staging_from_bank`).
    staging_bytes: u64 = 9 << 20,
    tickets: u32 = 256,
    /// The demand ranges' read route, fixed at start: true reads every demand range straight into its rows (one preadv
    /// iovec per part, no staging, no copy); false (the default) takes the staging loop and the copy. The pool tests no
    /// range: the owner proves the alignment once at construction (`checkDirectBlock` + the stream's
    /// `checkDirectReads`), so the uncached preadv takes the drive's direct path.
    direct: bool = false,
    spec: ?Spec = null,
};

/// A speculative staging slot: the page-rounded record plus two pages.
pub fn slotBytes(record_bytes: u64, page: u64) u64 {
    return std.mem.alignForward(u64, record_bytes, page) + 2 * page;
}

/// The host bytes `Pool.start` maps and allocates for `opt` (a bill's read pool term): each worker's staging buffer,
/// the speculative slots, and each ticket's result words, log word and published flag.
pub fn poolBytes(opt: Options, page: u64) u64 {
    var n = @as(u64, opt.workers) * opt.staging_bytes + @as(u64, opt.tickets) * (res_w * @sizeOf(i64) + @sizeOf(i64) + @sizeOf(bool));
    if (opt.spec) |s| n += @as(u64, s.slots) * slotBytes(s.record_bytes, page);
    return n;
}

/// Page-aligned chunk so `chunks` chunks cover a staging slot.
pub fn chunkBytes(chunks: u32, record_bytes: u64, page: u64) u64 {
    return (std.math.divCeil(u64, slotBytes(record_bytes, page), chunks * page) catch unreachable) * page;
}

pub const EventKind = enum(i32) { metal = 1, host = 2 };

/// The reader's per-range component limit (q3_lookahead4_exl3.c MAX_COMP): a record's gate/up range and its down
/// range each hold 1..6 components.
pub const max_range_components = 6;

/// A record topology the reader serves, else a compile error: `gate_up` components in the gate/up range, the rest in
/// the down range, each range 1..`max_range_components`.
pub fn checkTopology(comptime components: usize, comptime gate_up: usize) void {
    if (gate_up < 1 or gate_up > max_range_components or components <= gate_up or components - gate_up > max_range_components)
        @compileError(std.fmt.comptimePrint("expert reader: {d} components with {d} gate/up (each range holds 1..{d})", .{ components, gate_up, max_range_components }));
}

/// An fd only `openUncached` makes: a regular file opened read-only with O_NOFOLLOW, the page cache bypassed
/// (F_NOCACHE) and read-ahead off; `size` is the file's at open. The reader takes nothing else.
pub const UncachedFd = struct {
    fd: std.c.fd_t,
    size: u64,

    pub fn close(f: UncachedFd) void {
        _ = std.c.close(f.fd);
    }
};

/// Why a direct-read geometry is refused (`checkDirectBlock`, the stream's `checkDirectReads`), each by name.
pub const DirectError = error{ DirectReadsBlock, DirectReadsOffset, DirectReadsLength, DirectReadsRow };

/// The file system block of `fd` (fstatfs `f_bsize`: APFS 4,096 B): the alignment at which an F_NOCACHE preadv reads
/// straight from the drive into the caller's pages. Read once, at construction.
pub fn fileBlockSize(fd: UncachedFd) error{StatFailed}!u64 {
    if (comptime builtin.os.tag != .macos) return error.StatFailed;
    var st: Statfs = undefined;
    if (fstatfs(fd.fd, &st) != 0 or st.f_bsize == 0) return error.StatFailed;
    return st.f_bsize;
}

/// Darwin's `struct statfs` (64-bit inodes, arm64's only layout): only `f_bsize` is read.
const Statfs = extern struct { f_bsize: u32, f_iosize: i32, rest: [2160]u8 };
extern "c" fn fstatfs(fd: c_int, buf: *Statfs) c_int;

comptime {
    if (@sizeOf(Statfs) != 2168) @compileError("expert io: struct statfs is 2,168 bytes on Darwin");
}

/// A direct-read block: a power of two of at least 512 B and at most `page` (the pool's staging page).
pub fn checkDirectBlock(block: u64, page: u64) DirectError!void {
    if (block < 512 or !std.math.isPowerOfTwo(block) or block > page or page % block != 0) return error.DirectReadsBlock;
}

pub const OpenError = error{ NotFound, OpenFailed, StatFailed, NotRegularFile, NoCacheRefused };

/// `path` for the reader: open(O_RDONLY | O_NOFOLLOW), a regular file, F_NOCACHE and read-ahead off. `errno`
/// receives the open's errno when it fails.
pub fn openUncached(path: [*:0]const u8, errno: ?*c_int) OpenError!UncachedFd {
    return openUncachedWith(path, errno, false);
}

/// `openUncached` through a symlink (a model directory's safetensors shard may be one): the target is opened, still
/// a regular file, F_NOCACHE and read-ahead off.
pub fn openUncachedFollowing(path: [*:0]const u8, errno: ?*c_int) OpenError!UncachedFd {
    return openUncachedWith(path, errno, true);
}

fn openUncachedWith(path: [*:0]const u8, errno: ?*c_int, follow: bool) OpenError!UncachedFd {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .NOFOLLOW = !follow, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) {
        const e = std.c._errno().*;
        if (errno) |p| p.* = e;
        return if (e == @intFromEnum(std.posix.E.NOENT)) error.NotFound else error.OpenFailed;
    }
    errdefer _ = std.c.close(fd);
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
    if (st.mode & std.c.S.IFMT != std.c.S.IFREG) return error.NotRegularFile;
    io_util.noCache(fd, .{}) catch return error.NoCacheRefused;
    return .{ .fd = fd, .size = @intCast(st.size) };
}

pub const Pool = struct {
    allocator: std.mem.Allocator,
    staging: []align(std.heap.page_size_min) u8,
    spec_staging: ?[]align(std.heap.page_size_min) u8 = null,
    /// [ticket * res_w + k], written by the workers; zeroed before start.
    res: []i64,
    /// Completion order: log[seq % tickets] = ticket.
    log: []i64,
    gauge: [6]i64 = @splat(0),
    /// Written by the pool under its mutex until `stop`.
    counters: [counters_n]i64 = @splat(0),
    /// Tickets seen in the log since their last submit.
    published: []bool,
    /// Log entries consumed so far.
    seen: i64 = 0,
    next_ticket: u32 = 0,
    /// Demand's tickets end here.
    demand_tickets: u32 = 0,
    record_bytes: u64 = 0,

    /// Starts the process's pool (the speculative class, when given, is
    /// configured first). Staging, status words, counters and the log live
    /// here and stay put until `stop`.
    pub fn start(allocator: std.mem.Allocator, opt: Options) !*Pool {
        const page = std.heap.pageSize();
        if (opt.workers == 0 or opt.workers > max_workers or opt.staging_bytes == 0 or opt.staging_bytes % page != 0 or opt.tickets < 2 * max_items)
            return error.InvalidOptions;
        if (opt.spec) |s| if (s.threads == 0 or s.threads > max_spec_threads or s.slots == 0 or s.slots > max_spec or s.record_bytes == 0 or
            s.chunk_bytes == 0 or s.chunk_bytes % page != 0 or s.idle_busy > 1) return error.InvalidOptions;
        if (c.q3ld_abi() != abi_version or c.q3ld_counters_n() != counters_n or c.q3ld_max_spec() != max_spec or c.q3ld_max_pre() != max_pre or
            c.q3ld_max_gates() != max_gates or c.q3ld_max_gate_tickets() != max_gate_tickets) return error.PoolAbi;
        const self = try allocator.create(Pool);
        errdefer allocator.destroy(self);
        const staging = try std.posix.mmap(null, @intCast(opt.workers * opt.staging_bytes), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        errdefer std.posix.munmap(staging);
        const res = try allocator.alloc(i64, @as(usize, opt.tickets) * res_w);
        errdefer allocator.free(res);
        const log_arr = try allocator.alloc(i64, opt.tickets);
        errdefer allocator.free(log_arr);
        const published = try allocator.alloc(bool, opt.tickets);
        errdefer allocator.free(published);
        @memset(res, 0);
        @memset(log_arr, 0);
        @memset(published, false);
        const demand: u32 = opt.tickets;
        self.* = .{ .allocator = allocator, .staging = staging, .res = res, .log = log_arr, .published = published, .demand_tickets = demand };
        var bufs: [max_spec]u64 = undefined;
        var threads: i32 = 0;
        var slot_bytes: u64 = 0;
        if (opt.spec) |s| {
            slot_bytes = slotBytes(s.record_bytes, page);
            const spec = try std.posix.mmap(null, @intCast(s.slots * slot_bytes), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
            self.spec_staging = spec;
            for (0..s.slots) |i| bufs[i] = @intFromPtr(spec.ptr) + i * slot_bytes;
            threads = @intCast(s.threads);
            self.record_bytes = s.record_bytes;
        }
        errdefer if (self.spec_staging) |s| std.posix.munmap(s);
        // Arguments are valid here, so a refusal means a pool is running.
        const nslots: i32 = if (opt.spec) |s| @intCast(s.slots) else 0;
        const rec_len: i64 = @intCast(self.record_bytes);
        if (c.q3ld_spec_config(threads, &bufs, nslots, @intCast(slot_bytes), rec_len, @intCast(if (opt.spec) |s| s.chunk_bytes else 0), &self.counters) != 0)
            return error.PoolUnavailable;
        if (opt.spec) |s| if (c.q3ld_spec_streams(@intCast(s.idle_busy)) != 0) return error.PoolUnavailable;
        if (c.q3ld_direct_config(@intFromBool(opt.direct)) != 0) return error.PoolUnavailable;
        var ptrs: [max_workers]u64 = undefined;
        for (0..opt.workers) |w| ptrs[w] = @intFromPtr(staging.ptr) + w * opt.staging_bytes;
        const rc = c.q3ld_start(@intCast(opt.workers), &ptrs, @intCast(opt.staging_bytes), @intCast(page), res.ptr, opt.tickets, log_arr.ptr, opt.tickets, &self.gauge);
        if (rc == -2) _ = c.q3ld_stop(); // fewer threads than asked: join the ones that started
        if (rc != 0) return if (rc == -1) error.PoolUnavailable else error.PoolStart;
        return self;
    }

    /// After the owned consumers settle: expire unbound reads and wait for the pool's activity to stop.
    /// Ready speculative records and their tags stay intact. A timeout refuses the phase change.
    pub fn quiesce(self: *Pool, timeout_ns: i64) error{ InvalidTimeout, Timeout }!void {
        _ = self;
        if (timeout_ns <= 0) return error.InvalidTimeout;
        if (c.q3ld_quiesce(timeout_ns) != 0) return error.Timeout;
    }

    /// Drains the queue, joins the workers (releasing every live gate), then
    /// frees what they wrote into.
    pub fn stop(self: *Pool) void {
        _ = c.q3ld_quiesce(10 * std.time.ns_per_s);
        _ = c.q3ld_stop();
        const a = self.allocator;
        std.posix.munmap(self.staging);
        if (self.spec_staging) |s| std.posix.munmap(s);
        a.free(self.res);
        a.free(self.log);
        a.free(self.published);
        a.destroy(self);
    }

    /// Arms the event-gate class: the satisfied prefix goes to `object` (an
    /// id<MTLSharedEvent>, or an 8-aligned int64 host word) from the
    /// publishing worker; a gate unsatisfied after `timeout_ns` is forced.
    pub fn armEvent(self: *Pool, kind: EventKind, object: u64, timeout_ns: i64, start_value: u64) !void {
        _ = self;
        if (c.q3ld_ev_config(@intFromEnum(kind), object, timeout_ns, start_value) != 0) return error.EventRefused;
    }

    /// A layer call's speculative step: settles every unclaimed record tagged
    /// <= `cur`, then queues `bases` (whole records of `len` bytes) tagged
    /// `cur + 1`. Returns how many were newly queued.
    pub fn specStep(self: *Pool, fd: UncachedFd, cur: i64, bases: []const i64, len: u64) !u32 {
        const len_i: i64 = @intCast(if (len == 0) self.record_bytes else len);
        const rc = c.q3ld_spec_step_len(fd.fd, @intCast(fd.size), cur, @intCast(bases.len), bases.ptr, len_i);
        if (rc < 0) return error.SpecRefused;
        return @intCast(rc);
    }

    /// `specStep` whose record k is `lens[k]` bytes (records of layers of different lengths in one step).
    pub fn specStepLens(self: *Pool, fd: UncachedFd, cur: i64, bases: []const i64, lens: []const i64) !u32 {
        _ = self;
        std.debug.assert(bases.len == lens.len);
        const rc = c.q3ld_spec_step_lens(fd.fd, @intCast(fd.size), cur, @intCast(bases.len), bases.ptr, lens.ptr);
        if (rc < 0) return error.SpecRefused;
        return @intCast(rc);
    }

    /// Takes a submit that no worker has started back out of the queue, by its first ticket: every range job of it
    /// (all or none). true = withdrawn (its tickets are published as `withdrawn`: nothing read, no row written; the jobs
    /// behind it keep their order); false = it runs as submitted: a worker started it (or it is done, or no submit
    /// starts at `first`), or its bytes are promised elsewhere (a ticket bound to a pre-range or waited on by a live
    /// event gate: the C call's -2).
    pub fn withdraw(self: *Pool, first: u32) error{PoolUnavailable}!bool {
        _ = self;
        return switch (c.q3ld_withdraw(first)) {
            1 => true,
            0, -2 => false,
            else => error.PoolUnavailable,
        };
    }

    /// Registers gates in the GPU's encode order: gate i waits for `counts[i]`
    /// of `tickets` (in order); values strictly increase above every earlier one.
    pub fn registerGates(self: *Pool, values: []const u64, counts: []const i32, tickets: []const i64) !void {
        _ = self;
        std.debug.assert(values.len == counts.len and values.len > 0);
        switch (c.q3ld_ev_gates(@intCast(values.len), values.ptr, counts.ptr, tickets.ptr)) {
            -2 => return error.GateInvalid,
            -3 => return error.GatesFull,
            else => |rc| if (rc != @as(i32, @intCast(values.len))) return error.GateRefused,
        }
    }

    /// Forces every live gate <= `value` (error paths: nothing may wait for it).
    pub fn releaseGates(self: *Pool, value: u64) void {
        _ = self;
        _ = c.q3ld_ev_release(value);
    }

    /// The pool's clock (a `Result`'s `t_start_ns` / `t_end_ns`).
    pub fn nowNs() i64 {
        return c.q3ld_monotonic_ns();
    }

    pub fn counter(self: *const Pool, which: Counter) i64 {
        return @atomicLoad(i64, &self.counters[@intFromEnum(which)], .monotonic);
    }

    /// Blocks until every ticket in [first, first + count) is in the log;
    /// `timeout_ns` bounds each wait for the next completion.
    pub fn wait(self: *Pool, first: u32, count: u32, timeout_ns: i64) !void {
        while (true) {
            self.drain();
            if (std.mem.allEqual(bool, self.published[first..][0..count], true)) return;
            if (c.q3ld_wait(self.seen, timeout_ns) == self.seen) return error.Timeout;
        }
    }

    /// A copy of the read gauge (taken under the pool mutex): 0 in flight, 1 max in flight, 2 depth sum,
    /// 3 samples, 4 wall ns with a read in flight, 5 busy-since ns.
    pub fn readGauge(self: *Pool) [6]i64 {
        _ = self;
        var out: [6]i64 = undefined;
        c.q3ld_gauge(&out);
        return out;
    }

    pub fn result(self: *const Pool, ticket: u32) Result {
        const w = self.res[@as(usize, ticket) * res_w ..][0..res_w];
        return .{ .status = @enumFromInt(w[0]), .preadv_calls = w[1], .bytes_returned = w[2], .payload = w[3], .errno = w[4], .t_start_ns = w[5], .t_end_ns = w[6], .worker = w[7] };
    }

    /// Tickets in publication order for log positions [from, to).
    pub fn logOrder(self: *const Pool, from: i64, to: i64, out: []u32) []u32 {
        var n: usize = 0;
        var s = from;
        while (s < to and n < out.len) : (s += 1) {
            out[n] = @intCast(self.log[@intCast(@mod(s, @as(i64, @intCast(self.log.len))))]);
            n += 1;
        }
        return out[0..n];
    }

    /// Marks what the log published since the last call (ACQUIRE on the
    /// sequence, so the status words and slot bytes of those tickets are visible).
    fn drain(self: *Pool) void {
        const s = c.q3ld_seq();
        while (self.seen < s) : (self.seen += 1) {
            const t = self.log[@intCast(@mod(self.seen, @as(i64, @intCast(self.log.len))))];
            self.published[@intCast(t)] = true;
        }
    }

    /// The first of `count` tickets in [lo, hi) for the next submit, wrapping to `lo` (the log drained first).
    fn claimTickets(self: *Pool, lo: u32, hi: u32, next: *u32, count: u32) u32 {
        self.drain();
        if (next.* + count > hi) next.* = lo;
        return next.*;
    }
};

test "mimo sdk2 stageB shared: phase quiescence rejects unbounded deadlines before touching the native pool" {
    // This inert host object owns no mappings or tickets. Invalid deadlines must return before any C call.
    var unstarted: Pool = .{ .allocator = std.testing.allocator, .staging = undefined, .res = &.{}, .log = &.{}, .published = &.{} };
    try std.testing.expectError(error.InvalidTimeout, unstarted.quiesce(0));
    try std.testing.expectError(error.InvalidTimeout, unstarted.quiesce(-1));
}

/// Test builds (-DQ3LD_INJECT): one scripted preadv fault at an aligned file
/// offset (code 1 EINTR, 2 EIO, 3 zero return, 4 truncate to `arg` bytes, 5
/// sleep `arg` ns first).
pub fn injectFault(aligned_offset: u64, code: i64, arg: i64) void {
    c.q3ld_test_rules(1, &[_]i64{@intCast(aligned_offset)}, &[_]i64{code}, &[_]i64{arg});
}

pub fn injectFaults(aligned_offsets: []const i64, codes: []const i64, args: []const i64) void {
    c.q3ld_test_rules(@intCast(aligned_offsets.len), aligned_offsets.ptr, codes.ptr, args.ptr);
}

pub fn clearFaults() void {
    c.q3ld_test_rules(0, &[_]i64{0}, &[_]i64{0}, &[_]i64{0});
}

/// One record topology over the process's pool, fixed when its source is built: `components` components per record,
/// the first `gate_up` in its gate/up range. Every demand and pre-read submit goes through it: gate/up of
/// record i is ticket `first + i`, its down `first + n + i`, whatever the topology.
pub fn Records(comptime components: usize, comptime gate_up: usize) type {
    comptime checkTopology(components, gate_up);
    return struct {
        /// One record's destinations, a row address per component.
        pub const Rows = [components]u64;

        /// One submit: `rows.len` records, every gate/up span (`gu_offsets`, `gate_up` parts) then every down span, each
        /// part `lens[c]` bytes into `rows[i][c]`, each span a job of its own (idle workers read them side by side, each
        /// its own status). Returns the first of its 2n tickets.
        pub fn submit(pool: *Pool, fd: UncachedFd, gu_offsets: []const u64, down_offsets: []const u64, rows: []const Rows, lens: *const Rows) !u32 {
            return submitOn(pool, 0, pool.demand_tickets, &pool.next_ticket, fd, gu_offsets, down_offsets, rows, lens);
        }

        /// A part's place in a pack whose parts live apart (`submitScatter`): its file and offset.
        pub const Part = struct { fd: UncachedFd, offset: u64 };

        /// `submit` for records whose parts are not contiguous: part `c` of record `i` read from `parts[i][c]`. The
        /// tickets are `submit`'s; no speculative or pre-read class serves them.
        pub fn submitScatter(pool: *Pool, parts: []const [components]Part, rows: []const Rows, lens: *const Rows) !u32 {
            const n = rows.len;
            if (n == 0 or n > max_items or parts.len != n) return error.InvalidJob;
            var fds: [max_items * components]i32 = undefined;
            var sizes: [max_items * components]i64 = undefined;
            var offsets: [max_items * components]i64 = undefined;
            var row_ptrs: [max_items][*]const u64 = undefined;
            for (parts, 0..) |rec, i| {
                for (rec, 0..) |p, k| {
                    fds[i * components + k] = p.fd.fd;
                    sizes[i * components + k] = @intCast(p.fd.size);
                    offsets[i * components + k] = @intCast(p.offset);
                }
                row_ptrs[i] = &rows[i];
            }
            var lens_i: [components]i64 = undefined;
            for (lens.*, &lens_i) |l, *li| li.* = @intCast(l);
            const first = pool.claimTickets(0, pool.demand_tickets, &pool.next_ticket, @intCast(2 * n));
            try submitted(pool, first, @intCast(2 * n), c.q3ld_submit_scatter(@intCast(n), gate_up, components - gate_up, &fds, &sizes, &offsets, &row_ptrs, &lens_i, first), &pool.next_ticket);
            return first;
        }

        fn submitOn(pool: *Pool, lo: u32, hi: u32, next: *u32, fd: UncachedFd, gu_offsets: []const u64, down_offsets: []const u64, rows: []const Rows, lens: *const Rows) !u32 {
            const n = rows.len;
            if (n == 0 or n > max_items or gu_offsets.len != n or down_offsets.len != n) return error.InvalidJob;
            const count: u32 = @intCast(2 * n);
            const first = pool.claimTickets(lo, hi, next, count);
            var offsets: [2 * max_items]i64 = undefined;
            var row_ptrs: [max_items][*]const u64 = undefined;
            for (0..n) |i| {
                offsets[i] = @intCast(gu_offsets[i]);
                offsets[n + i] = @intCast(down_offsets[i]);
                row_ptrs[i] = &rows[i];
            }
            var lens_i: [components]i64 = undefined;
            for (lens.*, &lens_i) |l, *li| li.* = @intCast(l);
            try submitted(pool, first, count, c.q3ld_submit(fd.fd, @intCast(fd.size), no_deadline, @intCast(n), gate_up, components - gate_up, &offsets, &row_ptrs, &lens_i, first), next);
            return first;
        }

        fn submitted(pool: *Pool, first: u32, count: u32, rc: c_int, next: *u32) !void {
            switch (rc) {
                0 => {},
                -2 => return error.TicketsBusy,
                -3 => return error.QueueFull,
                else => return error.SubmitRefused,
            }
            @memset(pool.published[first..][0..count], false);
            next.* = first + count;
        }

        /// Arms the pre-read class with one geometry's component lengths (needs the speculative class). The pool
        /// binds a demand range to a pre-range only at this topology and these lengths.
        pub fn armPreRead(pool: *Pool, lens: *const Rows) !void {
            _ = pool;
            var l: [components]i64 = undefined;
            for (lens, &l) |x, *y| y.* = @intCast(x);
            if (c.q3ld_pre_config(gate_up, components - gate_up, &l) != 0) return error.PreReadRefused;
        }

        /// A layer call's certain misses (record bases, route order) before its plan, as gate/up and down ranges of
        /// `lens`; `tag` = the call's settle value. Returns how many ranges were queued.
        pub fn preRead(pool: *Pool, fd: UncachedFd, tag: i64, bases: []const i64, lens: *const Rows) !u32 {
            _ = pool;
            var l: [components]i64 = undefined;
            for (lens, &l) |x, *y| y.* = @intCast(x);
            const rc = c.q3ld_pre_read_lens(fd.fd, @intCast(fd.size), tag, @intCast(bases.len), bases.ptr, &l);
            if (rc < 0) return error.PreReadRefused;
            return @intCast(rc);
        }
    };
}
