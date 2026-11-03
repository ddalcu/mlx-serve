//! A request's output as the connection thread drains it: the tokens, the logprobs, the terminator and the request's
//! numbers. Every engine fills it through `Sink.event` (`onEvent`) from its own thread or the GPU thread; the
//! connection thread reads it. Two owners hold it: the connection thread and, from submit to `finished`, the engine;
//! the last to let go frees it (`release`).

const std = @import("std");
const abi = @import("../../abi/engine.zig");
const logprob = @import("../../sdk/request/logprob.zig");
const ConstraintSpan = @import("../../sdk/grammar/protocol.zig").ConstraintSpan;

pub const LogprobResult = logprob.LogprobResult;
pub const TokenLogprob = logprob.TokenLogprob;

pub const State = enum { pending_prefill, decoding, finished, errored };

/// One `waitNext` result: a token, then exactly one terminator.
pub const NextResult = union(enum) {
    token: u32,
    /// Generation completed; `finish_reason` is set.
    done: void,
    /// Generation failed; `error_code` is set.
    err: void,
};

pub const Stream = struct {
    allocator: std.mem.Allocator,
    io: std.Io,

    state: State = .pending_prefill,
    out_mu: std.Io.Mutex = .init,
    out_cond: std.Io.Condition = .init,
    /// Set beside every `out_cond` broadcast: events take a timed wait, which lets the connection thread probe the
    /// peer socket during a long prefill (`waitNextTimeout`).
    out_event: std.Io.Event = .unset,
    out_buf: std.ArrayList(u32) = .empty,
    out_idx: usize = 0,
    finished: bool = false,
    error_code: ?[]const u8 = null,
    /// {needed, available} bytes the engine compared when it refused this prefill (`PrefillDoesNotFit`).
    refused_bill: ?[2]u64 = null,
    finish_reason: []const u8 = "stop",
    /// Set only by the degenerate-tail guard: the wire reason stays "stop", this names the cause. Static.
    finish_details: ?[]const u8 = null,
    /// Where the degenerate span begins in the generated tokens; a non-streamed reply is cut there.
    loop_trim_start: ?usize = null,
    cancelled: std.atomic.Value(bool) = .init(false),
    /// Owners: the connection thread, plus the engine from submit to its `finished` (`engineHolds`).
    refs: std.atomic.Value(u32) = .init(1),
    /// Reasoning-protocol payload boundary, published before the token that carries it.
    constraint_payload_at: std.atomic.Value(u32) = .init(std.math.maxInt(u32)),
    constraint_payload_offset: std.atomic.Value(u32) = .init(0),

    /// The whole prompt's length as submitted: a refusal before prefill quotes it.
    prompt_len: usize = 0,
    prompt_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    cached_tokens: u32 = 0,
    /// Monotonic arrival time: anchors the time to first token.
    request_start_ts: std.Io.Timestamp,
    /// Arrival to the end of prefill (queue wait included).
    first_token_ns: u64 = 0,
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,
    /// The whole generation was token id 0 (the server drops its prefix-cache state).
    was_pad_only: bool = true,
    logprobs_buf: std.ArrayList(LogprobResult) = .empty,
    /// A spoken reply (`.audio` events): its frames' codes and its samples, at `sample_rate`, under `out_mu`.
    audio_codes: std.ArrayList(u32) = .empty,
    audio_samples: std.ArrayList(f32) = .empty,
    sample_rate: u32 = 0,
    prompt_logprobs_buf: std.ArrayList(LogprobResult) = .empty,
    /// Forwards `cancel` to the engine running the request.
    canceller: ?Canceller = null,

    pub const Canceller = struct { ctx: *anyopaque, cancel: *const fn (ctx: *anyopaque) void };

    pub fn create(allocator: std.mem.Allocator, io: std.Io) !*Stream {
        const s = try allocator.create(Stream);
        s.* = .{ .allocator = allocator, .io = io, .request_start_ts = .now(io, .boot) };
        return s;
    }

    /// The engine takes its hold before it is handed the sink; its `finished` lets go.
    pub fn engineHolds(self: *Stream) void {
        _ = self.refs.fetchAdd(1, .acq_rel);
    }

    /// Lets go of one hold; the last frees the stream.
    pub fn release(self: *Stream) void {
        if (self.refs.fetchSub(1, .acq_rel) == 1) self.free();
    }

    fn free(self: *Stream) void {
        if (self.error_code) |code| self.allocator.free(code);
        for (self.logprobs_buf.items) |*lp| self.allocator.free(lp.top_logprobs);
        self.logprobs_buf.deinit(self.allocator);
        for (self.prompt_logprobs_buf.items) |*lp| self.allocator.free(lp.top_logprobs);
        self.prompt_logprobs_buf.deinit(self.allocator);
        self.audio_codes.deinit(self.allocator);
        self.audio_samples.deinit(self.allocator);
        self.out_buf.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// The sink an engine fills this stream through.
    pub fn sink(self: *Stream) abi.Sink {
        return .{ .ctx = self, .event = onEvent };
    }

    /// Record the reasoning→payload boundary, visible no later than the token that carries it.
    pub fn publishConstraintSpan(self: *Stream, token_index: u32, byte_offset: u32) void {
        self.constraint_payload_offset.store(byte_offset, .release);
        self.constraint_payload_at.store(token_index, .release);
    }

    /// Where the constrained payload begins in the generated stream, or null while it has not begun.
    pub fn constraintPayloadStart(self: *const Stream) ?ConstraintSpan {
        const at = self.constraint_payload_at.load(.acquire);
        if (at == std.math.maxInt(u32)) return null;
        return .{ .token_index = at, .byte_offset = self.constraint_payload_offset.load(.acquire) };
    }

    pub fn pushToken(self: *Stream, t: u32) void {
        self.pushTokenWithLogprob(t, null);
    }

    /// A token and its logprob entry in ONE critical section: a streaming reader reads entry i as soon as it has
    /// token i, so an append outside the lock can be invisible or race a reallocating grow.
    pub fn pushTokenWithLogprob(self: *Stream, t: u32, lp: ?LogprobResult) void {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (lp) |entry| {
            // Degrade to an error like the token append below, and still broadcast: an early return hangs the reader.
            self.logprobs_buf.append(self.allocator, entry) catch |err| self.latchError(@errorName(err));
        }
        self.out_buf.append(self.allocator, t) catch |err| self.latchError(@errorName(err));
        self.completion_tokens += 1;
        if (t != 0) self.was_pad_only = false;
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    fn latchError(self: *Stream, name: []const u8) void {
        if (self.error_code == null) self.error_code = self.allocator.dupe(u8, name) catch null;
        self.state = .errored;
    }

    /// Copies the logprob entries produced since `cursor` (shallow: each `top_logprobs` stays the stream's),
    /// returning the new cursor. Under `out_mu`, so it never sees a half-written entry or a reallocating buffer.
    pub fn copyLogprobsFrom(self: *Stream, allocator: std.mem.Allocator, cursor: usize, out: *std.ArrayList(LogprobResult)) !usize {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        const items = self.logprobs_buf.items;
        if (cursor >= items.len) return cursor;
        try out.appendSlice(allocator, items[cursor..]);
        return items.len;
    }

    /// Normal completion; idempotent.
    pub fn markFinished(self: *Stream, reason: []const u8) void {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (self.finished) return;
        self.finished = true;
        self.state = .finished;
        self.finish_reason = reason;
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    /// Whether an error name names a memory failure (`OutOfMemory` from the MLX latch or Zig's allocator).
    pub fn errorNameIsMemory(name: []const u8) bool {
        return std.mem.eql(u8, name, "OutOfMemory") or std.mem.eql(u8, name, "InsufficientMemory");
    }

    pub fn errorNameIs(self: *Stream, name: []const u8) bool {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        const have = self.error_code orelse return false;
        return std.mem.eql(u8, have, name);
    }

    pub fn errorIsMemory(self: *Stream) bool {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        const name = self.error_code orelse return false;
        return errorNameIsMemory(name);
    }

    /// The request failed; `name` is copied.
    pub fn markError(self: *Stream, name: []const u8) void {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (self.error_code != null or self.finished) return;
        self.error_code = self.allocator.dupe(u8, name) catch null;
        self.state = .errored;
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    /// Blocks until the next token or the terminator. Buffered tokens drain before a terminator; a cancelled
    /// stream reads as done, so a disconnect or shutdown never sleeps out the generation.
    pub fn waitNext(self: *Stream) NextResult {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        while (true) {
            if (self.next()) |r| return r;
            self.out_cond.waitUncancelable(self.io, &self.out_mu);
        }
    }

    /// Like `waitNext`, but null (idle) after `timeout_ms` with nothing new: the caller probes the peer socket and
    /// sends keepalives during a long prefill.
    pub fn waitNextTimeout(self: *Stream, timeout_ms: i64) ?NextResult {
        while (true) {
            self.out_mu.lockUncancelable(self.io);
            if (self.next()) |r| {
                self.out_mu.unlock(self.io);
                return r;
            }
            // Armed under the lock: a producer sets before it unlocks, so no wakeup is lost.
            self.out_event.reset();
            self.out_mu.unlock(self.io);
            self.out_event.waitTimeout(self.io, .{ .duration = .{ .raw = .fromMilliseconds(timeout_ms), .clock = .awake } }) catch return null;
        }
    }

    /// Under `out_mu`.
    fn next(self: *Stream) ?NextResult {
        if (self.out_idx < self.out_buf.items.len) {
            const t = self.out_buf.items[self.out_idx];
            self.out_idx += 1;
            return .{ .token = t };
        }
        if (self.error_code != null) return .{ .err = {} };
        if (self.finished) return .{ .done = {} };
        if (self.cancelled.load(.acquire)) return .{ .done = {} };
        return null;
    }

    /// Ends the request: the reader unblocks now, the engine stops at its next round.
    pub fn cancel(self: *Stream) void {
        self.cancelled.store(true, .release);
        if (self.canceller) |c| c.cancel(c.ctx);
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }
};

/// The engine's thread: writes the event into the stream. `prefilled` and `tokens` before `finished`, which is last.
pub fn onEvent(ctx: *anyopaque, _: u64, ev: *const abi.Event) callconv(.c) abi.Action {
    const s: *Stream = @ptrCast(@alignCast(ctx));
    switch (ev.kind) {
        .prefilled => {
            s.prompt_tokens = ev.prompt_tokens;
            s.cached_tokens = ev.cached;
            s.first_token_ns = @intCast(s.request_start_ts.untilNow(s.io, .boot).nanoseconds);
            s.state = .decoding;
            copyLogprobs(s, &s.prompt_logprobs_buf, abi.items(abi.Logprob, ev.prompt_logprobs));
        },
        .tokens => {
            if (ev.constraint_at != std.math.maxInt(u32)) s.publishConstraintSpan(ev.constraint_at, ev.constraint_offset);
            const lps = abi.items(abi.Logprob, ev.logprobs);
            for (abi.items(u32, ev.tokens), 0..) |t, i| s.pushTokenWithLogprob(t, if (i < lps.len) fromEvent(s.allocator, lps[i]) else null);
        },
        .audio => {
            s.out_mu.lockUncancelable(s.io);
            defer s.out_mu.unlock(s.io);
            s.sample_rate = ev.sample_rate;
            s.audio_codes.appendSlice(s.allocator, abi.items(u32, ev.codes)) catch return .stop;
            s.audio_samples.appendSlice(s.allocator, abi.items(f32, ev.samples)) catch return .stop;
            s.out_cond.broadcast(s.io);
            s.out_event.set(s.io);
        },
        .finished => {
            defer s.release();
            if (ev.stats) |st| {
                s.prefill_ns = st.prefill_ns;
                s.decode_ns = st.decode_ns;
            }
            s.finish_details = if (ev.finish_details.len > 0) abi.items(u8, ev.finish_details) else null;
            s.loop_trim_start = if (ev.loop_trim_start != std.math.maxInt(u64)) @intCast(ev.loop_trim_start) else null;
            if (ev.refused_needed != 0) s.refused_bill = .{ ev.refused_needed, ev.refused_available };
            if (ev.reason == .failed) {
                s.markError(abi.items(u8, ev.verdict.why));
                return .go;
            }
            // A cancelled request reads as vLLM's and SGLang's "abort" (OpenAI names none).
            s.markFinished(if (s.cancelled.load(.acquire)) "abort" else switch (ev.reason) {
                .stop => "stop",
                .length => "length",
                .cancelled, .failed => "abort",
            });
            return .go;
        },
    }
    return if (s.cancelled.load(.acquire)) .stop else .go;
}

fn copyLogprobs(s: *Stream, out: *std.ArrayList(LogprobResult), lps: []const abi.Logprob) void {
    for (lps) |lp| if (fromEvent(s.allocator, lp)) |e| out.append(s.allocator, e) catch {};
}

/// An event's logprob as the API surfaces read it; its top list is the stream's to free.
fn fromEvent(gpa: std.mem.Allocator, lp: abi.Logprob) ?LogprobResult {
    const top = abi.items(abi.Bias, lp.top);
    const out = gpa.alloc(TokenLogprob, top.len) catch return null;
    for (top, out) |b, *o| o.* = .{ .token_id = b.token, .logprob = b.bias };
    return .{ .token_logprob = lp.logprob, .top_logprobs = out };
}

// ── tests ──

const testing = std.testing;

fn send(s: *Stream, ev: abi.Event) abi.Action {
    return onEvent(s, 0, &ev);
}

test "stream: prefilled, tokens with their logprobs, then finished reach the reader in order" {
    const s = try Stream.create(testing.allocator, testing.io);
    defer s.release();
    s.engineHolds();
    _ = send(s, .{ .kind = .prefilled, .prompt_tokens = 7, .cached = 3 });
    var top = [_]abi.Bias{.{ .token = 9, .bias = -0.5 }};
    const lps = [_]abi.Logprob{ .{ .token = 4, .logprob = -0.1, .top = abi.span(abi.Bias, &top) }, .{ .token = 5, .logprob = -0.2, .top = .{} } };
    _ = send(s, .{ .kind = .tokens, .tokens = abi.span(u32, &.{ 4, 5 }), .logprobs = abi.span(abi.Logprob, &lps) });
    const st: abi.Stats = .{ .prefill_ns = 11, .decode_ns = 22 };
    _ = send(s, .{ .kind = .finished, .reason = .length, .stats = &st });
    try testing.expectEqual(NextResult{ .token = 4 }, s.waitNext());
    try testing.expectEqual(NextResult{ .token = 5 }, s.waitNext());
    try testing.expectEqual(NextResult.done, s.waitNext());
    try testing.expectEqualStrings("length", s.finish_reason);
    try testing.expectEqual(@as(u32, 7), s.prompt_tokens);
    try testing.expectEqual(@as(u32, 3), s.cached_tokens);
    try testing.expectEqual(@as(u32, 2), s.completion_tokens);
    try testing.expectEqual(@as(u64, 22), s.decode_ns);
    try testing.expectEqual(@as(usize, 2), s.logprobs_buf.items.len);
    try testing.expectEqual(@as(u32, 9), s.logprobs_buf.items[0].top_logprobs[0].token_id);
    try testing.expectEqual(@as(u32, 1), s.refs.load(.acquire));
}

test "stream: a failed finish carries its name, the refused bill and the loop cut; a cancel reads as done" {
    const s = try Stream.create(testing.allocator, testing.io);
    defer s.release();
    s.engineHolds();
    _ = send(s, .{ .kind = .finished, .reason = .failed, .verdict = .{ .code = .failed, .why = abi.Str.of("PrefillDoesNotFit") }, .refused_needed = 10, .refused_available = 4 });
    try testing.expectEqual(NextResult.err, s.waitNext());
    try testing.expect(s.errorNameIs("PrefillDoesNotFit"));
    try testing.expectEqual([2]u64{ 10, 4 }, s.refused_bill.?);

    const c = try Stream.create(testing.allocator, testing.io);
    defer c.release();
    c.cancel();
    try testing.expectEqual(NextResult.done, c.waitNext());
    try testing.expectEqual(abi.Action.stop, send(c, .{ .kind = .tokens }));
}

test "stream: an engine-cancelled request finishes as \"abort\", vLLM's word, never \"length\"" {
    const s = try Stream.create(testing.allocator, testing.io);
    defer s.release();
    s.engineHolds();
    _ = send(s, .{ .kind = .finished, .reason = .cancelled });
    try testing.expectEqual(NextResult.done, s.waitNext());
    try testing.expectEqualStrings("abort", s.finish_reason);
}

test "stream: a constraint span is visible with the token that carries it; the loop details ride the finish" {
    const s = try Stream.create(testing.allocator, testing.io);
    defer s.release();
    s.engineHolds();
    _ = send(s, .{ .kind = .tokens, .tokens = abi.span(u32, &.{ 1, 2 }), .constraint_at = 1, .constraint_offset = 3 });
    try testing.expectEqual(ConstraintSpan{ .token_index = 1, .byte_offset = 3 }, s.constraintPayloadStart().?);
    _ = send(s, .{ .kind = .finished, .finish_details = abi.Str.of("repetition_loop"), .loop_trim_start = 1 });
    try testing.expectEqualStrings("repetition_loop", s.finish_details.?);
    try testing.expectEqual(@as(usize, 1), s.loop_trim_start.?);
    try testing.expect(!s.was_pad_only);
}
