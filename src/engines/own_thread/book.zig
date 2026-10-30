//! What the own-thread engine adapters (ds4, llama.cpp, ANE) share: a request copied out of `submit`, the book of
//! one opened model's requests (queue order, cancel, status, close), and the events one request emits under the
//! contract of `abi/engine.zig`. The book's lock is never held across a `Sink.event`.

const std = @import("std");
const abi = @import("../../abi/engine.zig");

/// The adapters' Io: the C ABI hands an engine none, and mutexes, conditions and clocks need no more than this.
pub const io = std.Io.Threaded.global_single_threaded.io();

pub const Reason = @FieldType(abi.Event, "reason");

pub fn nowNs() u64 {
    return @intCast(std.Io.Clock.awake.now(io).nanoseconds);
}

/// One request, owned by its book from `add` to `done`.
pub const Req = struct {
    id: u64,
    sink: abi.Sink,
    prompt: []u32,
    eos: []u32,
    bias: []abi.Bias,
    sampling: abi.Sampling, // `logit_bias` points into `bias`
    max_tokens: u32,
    logprobs: u8,
    cancelled: std.atomic.Value(bool) = .init(false),
    running: bool = false, // under the book's lock
    // The engine thread's alone:
    emitted: u32 = 0,
    stats: abi.Stats = .{},
    t0: u64 = 0,

    pub fn isCancelled(r: *const Req) bool {
        return r.cancelled.load(.acquire);
    }

    /// The request's draw seed: its own, else the clock.
    pub fn seed(r: *const Req) u64 {
        if (r.sampling.seeded) return r.sampling.seed;
        return @truncate(@as(u128, @bitCast(@as(i128, std.Io.Clock.real.now(io).nanoseconds))));
    }

    pub fn isEos(r: *const Req, t: u32) bool {
        return std.mem.indexOfScalar(u32, r.eos, t) != null;
    }

    /// The prompt is in the engine, `cached` of it kept from before. Null: decode on; else the reason it ends.
    pub fn prefilled(r: *Req, cached: u32) ?Reason {
        if (r.isCancelled()) return .cancelled;
        r.stats.prefill_ns = nowNs() - r.t0;
        const ev: abi.Event = .{ .kind = .prefilled, .cached = @min(cached, @as(u32, @intCast(r.prompt.len))) };
        if (r.sink.event(r.sink.ctx, r.id, &ev) == .stop) return .stop;
        return if (r.max_tokens == 0) .length else null;
    }

    /// One committed round. An EOS id ends it unemitted; nothing past `max_tokens` goes out; `lps` holds one entry
    /// per token when the request asked for logprobs. Null: decode on; else the reason the request ends.
    pub fn commit(r: *Req, toks: []const u32, lps: []const abi.Logprob) ?Reason {
        if (r.isCancelled()) return .cancelled;
        var n = toks.len;
        var eos = false;
        for (toks, 0..) |t, i| if (r.isEos(t)) {
            n = i;
            eos = true;
            break;
        };
        const room = r.max_tokens - r.emitted;
        if (n > room) {
            n = room;
            eos = false;
        }
        var action: abi.Action = .go;
        if (n > 0) {
            if (r.logprobs > 0) std.debug.assert(lps.len >= n);
            const ev: abi.Event = .{
                .kind = .tokens,
                .tokens = abi.span(u32, toks[0..n]),
                .logprobs = if (r.logprobs > 0) abi.span(abi.Logprob, lps[0..n]) else .{},
            };
            action = r.sink.event(r.sink.ctx, r.id, &ev);
            r.emitted += @intCast(n);
        }
        if (eos or action == .stop) return .stop;
        return if (r.emitted >= r.max_tokens) .length else null;
    }

    /// The last event. `why` names a failure (static text); ignored otherwise.
    pub fn finish(r: *Req, reason: Reason, why: []const u8) void {
        if (r.stats.prefill_ns > 0) r.stats.decode_ns = nowNs() - r.t0 - r.stats.prefill_ns;
        const ev: abi.Event = .{
            .kind = .finished,
            .reason = reason,
            .stats = &r.stats,
            .verdict = if (reason == .failed) .{ .code = .failed, .why = abi.span(u8, why) } else .{},
        };
        _ = r.sink.event(r.sink.ctx, r.id, &ev);
    }
};

/// One opened model's requests, in submit order: waiting until the engine takes them, running until `done`.
pub const Book = struct {
    gpa: std.mem.Allocator,
    /// Requests that may wait beside the running ones; past it `add` answers `busy`.
    max_waiting: u32,
    mu: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    reqs: std.ArrayList(*Req) = .empty,
    closing: bool = false,
    decode_tps: f64 = 0,
    prefill_tps: f64 = 0,

    pub fn deinit(b: *Book) void {
        std.debug.assert(b.reqs.items.len == 0);
        b.reqs.deinit(b.gpa);
    }

    /// Copies `r` into a new waiting request; null with `out` set when refused (`caps`, a full queue, closing).
    pub fn add(b: *Book, caps: abi.Caps, id: u64, r: *const abi.Request, sink: abi.Sink, out: *abi.Verdict) ?*Req {
        if (abi.refusal(caps, abi.needs(r))) |v| {
            out.* = v;
            return null;
        }
        const req = b.copy(id, r, sink) catch {
            out.* = abi.Verdict.refuse(.failed, "out of memory");
            return null;
        };
        b.mu.lockUncancelable(io);
        defer b.mu.unlock(io);
        if (b.closing or b.waitingLocked() >= b.max_waiting) {
            b.free(req);
            out.* = if (b.closing) abi.Verdict.refuse(.busy, "the engine is closing") else abi.Verdict.refuse(.busy, "the engine's queue is full");
            return null;
        }
        b.reqs.append(b.gpa, req) catch {
            b.free(req);
            out.* = abi.Verdict.refuse(.failed, "out of memory");
            return null;
        };
        b.cond.broadcast(io);
        out.* = .{};
        return req;
    }

    fn copy(b: *Book, id: u64, r: *const abi.Request, sink: abi.Sink) !*Req {
        const req = try b.gpa.create(Req);
        errdefer b.gpa.destroy(req);
        const prompt = try b.gpa.dupe(u32, abi.items(u32, r.prompt));
        errdefer b.gpa.free(prompt);
        const eos = try b.gpa.dupe(u32, abi.items(u32, r.eos));
        errdefer b.gpa.free(eos);
        const bias = try b.gpa.dupe(abi.Bias, abi.items(abi.Bias, r.sampling.logit_bias));
        var sampling = r.sampling;
        sampling.logit_bias = abi.span(abi.Bias, bias);
        req.* = .{
            .id = id,
            .sink = sink,
            .prompt = prompt,
            .eos = eos,
            .bias = bias,
            .sampling = sampling,
            .max_tokens = r.max_tokens,
            .logprobs = r.logprobs,
            .t0 = nowNs(),
        };
        return req;
    }

    fn free(b: *Book, r: *Req) void {
        b.gpa.free(r.prompt);
        b.gpa.free(r.eos);
        b.gpa.free(r.bias);
        b.gpa.destroy(r);
    }

    fn waitingLocked(b: *const Book) u32 {
        var n: u32 = 0;
        for (b.reqs.items) |r| n += @intFromBool(!r.running);
        return n;
    }

    /// The oldest waiting request, now running. With `block`, waits for one; null when there is none (closing,
    /// when blocking). After `close` the waiting ones still come out, cancelled, to be finished.
    pub fn next(b: *Book, block: bool) ?*Req {
        b.mu.lockUncancelable(io);
        defer b.mu.unlock(io);
        while (true) {
            for (b.reqs.items) |r| if (!r.running) {
                r.running = true;
                return r;
            };
            if (!block or b.closing) return null;
            b.cond.waitUncancelable(io, &b.mu);
        }
    }

    /// Wakes a `next(true)` waiter when the owner has more to do than the book knows of.
    pub fn wake(b: *Book) void {
        b.mu.lockUncancelable(io);
        defer b.mu.unlock(io);
        b.cond.broadcast(io);
    }

    pub fn cancel(b: *Book, id: u64) void {
        b.mu.lockUncancelable(io);
        defer b.mu.unlock(io);
        for (b.reqs.items) |r| if (r.id == id) r.cancelled.store(true, .release);
    }

    /// Ends `r` with `reason` (its last event) and frees it.
    pub fn done(b: *Book, r: *Req, reason: Reason, why: []const u8) void {
        r.finish(reason, why);
        b.mu.lockUncancelable(io);
        defer b.mu.unlock(io);
        if (r.stats.prefill_ns > 0) b.prefill_tps = tps(r.prompt.len, r.stats.prefill_ns);
        if (r.stats.decode_ns > 0 and r.emitted > 0) b.decode_tps = tps(r.emitted, r.stats.decode_ns);
        if (std.mem.indexOfScalar(*Req, b.reqs.items, r)) |i| _ = b.reqs.orderedRemove(i);
        b.free(r);
    }

    fn tps(n: usize, ns: u64) f64 {
        return @as(f64, @floatFromInt(n)) * 1e9 / @as(f64, @floatFromInt(ns));
    }

    /// No more requests; every one in the book is cancelled. The owner then drains them through `done`.
    pub fn close(b: *Book) void {
        b.mu.lockUncancelable(io);
        defer b.mu.unlock(io);
        b.closing = true;
        for (b.reqs.items) |r| r.cancelled.store(true, .release);
        b.cond.broadcast(io);
    }

    pub fn status(b: *Book, out: *abi.Status) void {
        b.mu.lockUncancelable(io);
        defer b.mu.unlock(io);
        const waiting = b.waitingLocked();
        out.* = .{
            .running = @as(u32, @intCast(b.reqs.items.len)) - waiting,
            .waiting = waiting,
            .decode_tps = b.decode_tps,
            .prefill_tps = b.prefill_tps,
        };
    }
};

// ── tests ──

const testing = std.testing;

const Seen = struct {
    kinds: [8]@FieldType(abi.Event, "kind") = undefined,
    tokens: [8]u32 = undefined,
    n_events: usize = 0,
    n_tokens: usize = 0,
    reason: Reason = .stop,
    stop_on_tokens: bool = false,

    fn sink(s: *Seen) abi.Sink {
        return .{ .ctx = s, .event = onEvent };
    }

    fn onEvent(ctx: *anyopaque, _: u64, ev: *const abi.Event) callconv(.c) abi.Action {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        s.kinds[s.n_events] = ev.kind;
        s.n_events += 1;
        for (abi.items(u32, ev.tokens)) |t| {
            s.tokens[s.n_tokens] = t;
            s.n_tokens += 1;
        }
        if (ev.kind == .finished) s.reason = ev.reason;
        return if (ev.kind == .tokens and s.stop_on_tokens) .stop else .go;
    }
};

test "book: a round stops at EOS unemitted, and never passes max_tokens" {
    var b: Book = .{ .gpa = testing.allocator, .max_waiting = 4 };
    defer b.deinit();
    var seen: Seen = .{};
    const eos = [_]u32{9};
    var v: abi.Verdict = .{};
    const r = b.add(.{}, 1, &.{ .prompt = abi.span(u32, &.{ 1, 2 }), .max_tokens = 3, .eos = abi.span(u32, &eos) }, seen.sink(), &v).?;
    try testing.expectEqual(r, b.next(false).?);
    try testing.expectEqual(@as(?Reason, null), r.prefilled(0));
    try testing.expectEqual(@as(?Reason, .stop), r.commit(&.{ 4, 9, 5 }, &.{}));
    b.done(r, .stop, "");
    try testing.expectEqualSlices(u32, &.{4}, seen.tokens[0..seen.n_tokens]);

    var seen2: Seen = .{};
    const r2 = b.add(.{}, 2, &.{ .prompt = abi.span(u32, &.{1}), .max_tokens = 3 }, seen2.sink(), &v).?;
    _ = b.next(false);
    _ = r2.prefilled(0);
    try testing.expectEqual(@as(?Reason, null), r2.commit(&.{ 4, 5 }, &.{}));
    try testing.expectEqual(@as(?Reason, .length), r2.commit(&.{ 6, 7 }, &.{}));
    b.done(r2, .length, "");
    try testing.expectEqualSlices(u32, &.{ 4, 5, 6 }, seen2.tokens[0..seen2.n_tokens]);
    try testing.expectEqual(Reason.length, seen2.reason);
}

test "book: a sink's stop ends the request after its round; cancel and close cancel" {
    var b: Book = .{ .gpa = testing.allocator, .max_waiting = 4 };
    defer b.deinit();
    var v: abi.Verdict = .{};
    var seen: Seen = .{ .stop_on_tokens = true };
    const r = b.add(.{}, 1, &.{ .prompt = abi.span(u32, &.{1}), .max_tokens = 10 }, seen.sink(), &v).?;
    _ = b.next(false);
    _ = r.prefilled(0);
    try testing.expectEqual(@as(?Reason, .stop), r.commit(&.{ 4, 5 }, &.{}));
    b.done(r, .stop, "");

    var seen2: Seen = .{};
    const r2 = b.add(.{}, 2, &.{ .prompt = abi.span(u32, &.{1}), .max_tokens = 10 }, seen2.sink(), &v).?;
    b.cancel(2);
    try testing.expectEqual(@as(?Reason, .cancelled), r2.prefilled(0));
    b.close();
    try testing.expectEqual(r2, b.next(true).?); // still drained after close
    try testing.expectEqual(@as(?*Req, null), b.next(true));
    b.done(r2, .cancelled, "");
    try testing.expectEqual(@as(usize, 1), seen2.n_events); // finished alone
    try testing.expectEqual(@as(?*Req, null), b.add(.{}, 3, &.{ .prompt = abi.span(u32, &.{1}), .max_tokens = 1 }, seen2.sink(), &v));
    try testing.expectEqual(abi.Verdict.Code.busy, v.code);
}

test "book: an undeclared cap is refused by name; past the queue is busy; status counts both" {
    var b: Book = .{ .gpa = testing.allocator, .max_waiting = 1 };
    defer b.deinit();
    var seen: Seen = .{};
    var v: abi.Verdict = .{};
    var req: abi.Request = .{ .prompt = abi.span(u32, &.{1}), .max_tokens = 1, .logprobs = 2 };
    try testing.expectEqual(@as(?*Req, null), b.add(.{}, 1, &req, seen.sink(), &v));
    try testing.expect(std.mem.indexOf(u8, abi.items(u8, v.why), "logprobs") != null);
    req.logprobs = 0;
    const a = b.add(.{}, 1, &req, seen.sink(), &v).?;
    _ = b.next(false);
    const w = b.add(.{}, 2, &req, seen.sink(), &v).?;
    try testing.expectEqual(@as(?*Req, null), b.add(.{}, 3, &req, seen.sink(), &v));
    try testing.expectEqual(abi.Verdict.Code.busy, v.code);
    var st: abi.Status = .{};
    b.status(&st);
    try testing.expectEqual(@as(u32, 1), st.running);
    try testing.expectEqual(@as(u32, 1), st.waiting);
    b.done(a, .cancelled, "");
    b.done(w, .cancelled, "");
}
