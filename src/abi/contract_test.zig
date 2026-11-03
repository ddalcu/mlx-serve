//! The engine contract: `run` opens fresh instances of one engine table and checks the rules of `engine.zig`
//! on each. Every engine (ours, ds4, llama.cpp, ANE, the fakes) passes it; a failure is a named error.

const std = @import("std");
const abi = @import("engine.zig");

const io = std.testing.io;
const gpa = std.testing.allocator;

pub const Fixture = struct {
    model_dir: []const u8,
    model_type: []const u8,
    config_json: []const u8,
    prompt: []const u32, // a prompt the model serves; the suite asks for its own reply lengths
    settings_json: []const u8 = "",
    options_json: []const u8 = "",
    memory_budget: u64 = 1 << 30,
    timeout_ms: u32 = 10_000, // the longest wait for an event
    gate_ms: u32 = 2_000, // how long a callback waits for `status`/`memory` from another thread
};

pub const Error = error{
    TableHeader,
    Probe,
    LoadBytes,
    Open,
    InfoHeader,
    ThreadingMismatch,
    CapsEntryMismatch,
    SubmitRefused,
    UnknownId,
    EventSize,
    EventAfterFinished,
    TwoPrefilled,
    TokensBeforePrefilled,
    EmptyTokens,
    CachedPastPrompt,
    LogprobsMismatch,
    MaxTokensExceeded,
    MaxTokensShort,
    LengthNotReported,
    FinishedBeforePrefilled,
    FinishedVerdict,
    StopIgnored,
    StopNotHonoured,
    NotCancelled,
    UndeclaredCapAccepted,
    RefusalUnnamed,
    RefusedSubmitEmitted,
    ConcurrentRefused,
    StatusBlocked,
    StatusInsane,
    StepNoProgress,
    StepNeverIdle,
    CloseDroppedFinished,
    TimedOut,
};

pub fn run(engine: *const abi.Engine, fx: Fixture) !void {
    std.debug.assert(fx.prompt.len > 0);
    try checkTable(engine, fx);
    try checkOrderAndLength(engine, fx);
    try checkStop(engine, fx);
    try checkCancel(engine, fx);
    try checkRefusals(engine, fx);
    try checkConcurrent(engine, fx);
    try checkStatusDuringCallback(engine, fx);
    try checkStep(engine, fx);
    try checkClose(engine, fx);
}

// ── one open instance and what the host saw of it

const max_ids = 32;
const unlimited = std.math.maxInt(u32);

const Track = struct {
    submitted: bool = false,
    prompt_len: usize = 0,
    max_tokens: u32 = 0,
    logprobs: u8 = 0,
    stop_at: u32 = 0, // answer `stop` on this tokens event (1-based); 0 = never
    gate: bool = false, // the first tokens event waits for status/memory from another thread
    events: u32 = 0,
    prefilled: u32 = 0,
    token_events: u32 = 0,
    tokens: u64 = 0,
    stop_answered: bool = false,
    finished: u32 = 0,
    reason: @FieldType(abi.Event, "reason") = .stop,
};

const Session = struct {
    engine: *const abi.Engine,
    fx: Fixture,
    host: abi.Host,
    handle: *anyopaque,
    info: abi.Info,
    closed: bool = false,
    mu: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    tracks: [max_ids]Track = @splat(.{}),
    violation: ?Error = null,
    gate_go: std.atomic.Value(bool) = .init(false),
    gate_done: std.atomic.Value(bool) = .init(false),

    fn open(s: *Session, engine: *const abi.Engine, fx: Fixture) !void {
        s.* = .{ .engine = engine, .fx = fx, .host = .{ .ctx = s, .log = hostLog, .wake = hostWake }, .handle = undefined, .info = undefined };
        const o = s.openArgs();
        var bytes: u64 = 0;
        if (engine.load_bytes(&o, &bytes).code != .ok) return error.LoadBytes;
        var info: abi.Info = .{ .name = .{}, .version = .{}, .caps = .{}, .threading = .own_thread, .context_window = 0, .max_concurrent = 0, .embedding_dims = 0, .eos = .{}, .startup = .{} };
        var v: abi.Verdict = .{};
        s.handle = engine.open(&o, &info, &v) orelse return error.Open;
        s.info = info;
    }

    fn openArgs(s: *Session) abi.Open {
        return .{
            .model_dir = abi.span(u8, s.fx.model_dir),
            .model_type = abi.span(u8, s.fx.model_type),
            .config_json = abi.span(u8, s.fx.config_json),
            .context = 0,
            .max_concurrent = 4,
            .memory_budget = s.fx.memory_budget,
            .settings_json = abi.span(u8, s.fx.settings_json),
            .options_json = abi.span(u8, s.fx.options_json),
            .host = &s.host,
        };
    }

    fn close(s: *Session) void {
        if (s.closed) return;
        s.closed = true;
        s.engine.close(s.handle);
    }

    /// The first violation wins; the caller holds `mu`.
    fn fail(s: *Session, e: Error) void {
        if (s.violation == null) s.violation = e;
    }

    /// Closes the instance, then reports the first violation: `close` returns after every event is delivered.
    fn end(s: *Session) Error!void {
        s.close();
        return s.check();
    }

    fn check(s: *Session) Error!void {
        s.mu.lockUncancelable(io);
        defer s.mu.unlock(io);
        if (s.violation) |e| return e;
    }

    fn track(s: *Session, id: u64) Track {
        s.mu.lockUncancelable(io);
        defer s.mu.unlock(io);
        return s.tracks[id];
    }

    /// Submits from a copy of the request's memory that is poisoned once `submit` returns: the engine copies
    /// what it keeps.
    fn submit(s: *Session, id: u64, r: abi.Request, t: Track) !abi.Verdict {
        std.debug.assert(id < max_ids);
        s.mu.lockUncancelable(io);
        s.tracks[id] = t;
        s.tracks[id].submitted = true;
        s.tracks[id].prompt_len = s.fx.prompt.len;
        s.tracks[id].max_tokens = r.max_tokens;
        s.tracks[id].logprobs = r.logprobs;
        s.mu.unlock(io);
        const prompt = try gpa.dupe(u32, s.fx.prompt);
        defer gpa.free(prompt);
        var req = r;
        req.prompt = abi.span(u32, prompt);
        const v = s.engine.submit(s.handle, id, &req, .{ .ctx = s, .event = onEvent });
        @memset(prompt, 0xAAAA_AAAA);
        return v;
    }

    fn plain(s: *Session, max_tokens: u32) abi.Request {
        return .{ .prompt = abi.span(u32, s.fx.prompt), .max_tokens = max_tokens };
    }

    /// Drives the engine (steps it on this thread when it runs on the host's) until `done` holds under `mu`,
    /// a violation is recorded, or the fixture's timeout passes.
    fn waitFor(s: *Session, ctx: anytype, comptime done: fn (*Session, @TypeOf(ctx)) bool) Error!void {
        const deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromMilliseconds(s.fx.timeout_ms), .clock = .awake });
        while (true) {
            s.mu.lockUncancelable(io);
            const ok = done(s, ctx);
            const bad = s.violation;
            s.mu.unlock(io);
            if (bad) |e| return e;
            if (ok) return;
            if (deadline.untilNow(io).raw.nanoseconds >= 0) return error.TimedOut;
            if (s.engine.step) |step| {
                if (step(s.handle, nowNs() + 2 * std.time.ns_per_ms)) continue;
            }
            s.mu.lockUncancelable(io);
            s.cond.waitTimeout(io, &s.mu, .{ .duration = .{ .raw = .fromMilliseconds(2), .clock = .awake } }) catch {};
            s.mu.unlock(io);
        }
    }

    fn finished(s: *Session, ids: []const u64) Error!void {
        return s.waitFor(ids, struct {
            fn f(ss: *Session, xs: []const u64) bool {
                for (xs) |id| if (ss.tracks[id].finished == 0) return false;
                return true;
            }
        }.f);
    }

    fn started(s: *Session, ids: []const u64) Error!void {
        return s.waitFor(ids, struct {
            fn f(ss: *Session, xs: []const u64) bool {
                for (xs) |id| if (ss.tracks[id].token_events == 0) return false;
                return true;
            }
        }.f);
    }
};

fn nowNs() u64 {
    return @intCast(std.Io.Clock.awake.now(io).nanoseconds);
}

fn hostLog(_: *anyopaque, _: u8, _: abi.Str) callconv(.c) void {}

fn hostWake(ctx: *anyopaque) callconv(.c) void {
    const s: *Session = @ptrCast(@alignCast(ctx));
    s.cond.broadcast(io);
}

fn onEvent(ctx: *anyopaque, id: u64, ev: *const abi.Event) callconv(.c) abi.Action {
    const s: *Session = @ptrCast(@alignCast(ctx));
    var action: abi.Action = .go;
    var gate = false;
    s.mu.lockUncancelable(io);
    record: {
        if (id >= max_ids or !s.tracks[id].submitted) {
            s.fail(error.UnknownId);
            break :record;
        }
        const t = &s.tracks[id];
        t.events += 1;
        if (ev.size < @sizeOf(abi.Event)) s.fail(error.EventSize);
        if (t.finished > 0) s.fail(error.EventAfterFinished);
        switch (ev.kind) {
            .prefilled => {
                if (t.prefilled > 0) s.fail(error.TwoPrefilled);
                if (t.token_events > 0) s.fail(error.TokensBeforePrefilled);
                if (ev.cached > t.prompt_len) s.fail(error.CachedPastPrompt);
                t.prefilled += 1;
            },
            .audio => if (t.prefilled == 0) s.fail(error.TokensBeforePrefilled),
            .tokens => {
                if (t.prefilled == 0) s.fail(error.TokensBeforePrefilled);
                if (ev.tokens.len == 0) s.fail(error.EmptyTokens);
                if (t.stop_answered) s.fail(error.StopIgnored);
                if (ev.logprobs.len != (if (t.logprobs > 0) ev.tokens.len else 0)) s.fail(error.LogprobsMismatch);
                for (abi.items(abi.Logprob, ev.logprobs)) |lp| if (lp.top.len > t.logprobs) s.fail(error.LogprobsMismatch);
                t.token_events += 1;
                t.tokens += ev.tokens.len;
                if (t.tokens > t.max_tokens) s.fail(error.MaxTokensExceeded);
                if (t.stop_at == t.token_events) {
                    action = .stop;
                    t.stop_answered = true;
                }
                gate = t.gate and t.token_events == 1;
            },
            .finished => {
                t.finished += 1;
                t.reason = ev.reason;
                const failed = ev.reason == .failed;
                if (failed != (ev.verdict.code != .ok) or (failed and ev.verdict.why.len == 0)) s.fail(error.FinishedVerdict);
                if ((ev.reason == .stop or ev.reason == .length) and t.prefilled == 0) s.fail(error.FinishedBeforePrefilled);
                if (t.stop_answered and ev.reason != .stop) s.fail(error.StopNotHonoured);
            },
        }
    }
    s.cond.broadcast(io);
    s.mu.unlock(io);
    if (gate) awaitGate(s);
    return action;
}

/// Inside a callback: hand the poller the go and wait for its `status` and `memory` to return.
fn awaitGate(s: *Session) void {
    s.gate_go.store(true, .release);
    const deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromMilliseconds(s.fx.gate_ms), .clock = .awake });
    while (!s.gate_done.load(.acquire)) {
        if (deadline.untilNow(io).raw.nanoseconds >= 0) {
            s.mu.lockUncancelable(io);
            s.fail(error.StatusBlocked);
            s.mu.unlock(io);
            return;
        }
        io.sleep(.fromMicroseconds(200), .awake) catch {};
    }
}

fn gatePoller(s: *Session) void {
    while (!s.gate_go.load(.acquire)) io.sleep(.fromMicroseconds(200), .awake) catch {};
    var st: abi.Status = .{};
    s.engine.status(s.handle, &st);
    var m: abi.Memory = .{};
    s.engine.memory(s.handle, &m);
    s.gate_done.store(true, .release);
}

// ── the checks

fn checkTable(engine: *const abi.Engine, fx: Fixture) !void {
    if (engine.size < @sizeOf(abi.Engine) or engine.version != abi.version) return error.TableHeader;
    var v: abi.Verdict = .{};
    if (!engine.probe(abi.span(u8, fx.model_dir), abi.span(u8, fx.config_json), &v)) return error.Probe;
    var s: Session = undefined;
    try s.open(engine, fx);
    defer s.close();
    const i = s.info;
    if (i.size < @sizeOf(abi.Info) or i.name.len == 0 or i.context_window == 0 or i.max_concurrent == 0) return error.InfoHeader;
    if ((i.threading == .host_gpu_thread) != (engine.step != null)) return error.ThreadingMismatch;
    const c = i.caps;
    const media = c.images or c.video or c.audio;
    if (c.embed != (engine.embed != null) or c.decide != (engine.decide != null) or
        media != (engine.media_rows != null) or
        (c.owns_tokenizer and (engine.tokenize == null or engine.token_bytes == null)) or
        (c.owns_template and engine.render == null)) return error.CapsEntryMismatch;
}

fn checkOrderAndLength(engine: *const abi.Engine, fx: Fixture) !void {
    var s: Session = undefined;
    try s.open(engine, fx);
    defer s.close();
    if ((try s.submit(1, s.plain(7), .{})).code != .ok) return error.SubmitRefused;
    var ids: [2]u64 = .{ 1, 2 };
    var n: usize = 1;
    if (s.info.caps.logprobs) {
        var r = s.plain(5);
        r.logprobs = 2;
        if ((try s.submit(2, r, .{})).code != .ok) return error.SubmitRefused;
        n = 2;
    }
    try s.finished(ids[0..n]);
    try s.end();
    for (ids[0..n]) |id| try expectLength(s.track(id));
}

fn expectLength(t: Track) Error!void {
    if (t.reason != .length) return error.LengthNotReported;
    if (t.tokens != t.max_tokens) return error.MaxTokensShort;
}

fn checkStop(engine: *const abi.Engine, fx: Fixture) !void {
    var s: Session = undefined;
    try s.open(engine, fx);
    defer s.close();
    if ((try s.submit(1, s.plain(1000), .{ .stop_at = 1 })).code != .ok) return error.SubmitRefused;
    try s.finished(&.{1});
    try s.end();
    if (s.track(1).reason != .stop) return error.StopNotHonoured;
}

fn checkCancel(engine: *const abi.Engine, fx: Fixture) !void {
    var s: Session = undefined;
    try s.open(engine, fx);
    defer s.close();
    if ((try s.submit(1, s.plain(unlimited), .{})).code != .ok) return error.SubmitRefused;
    try s.started(&.{1});
    engine.cancel(s.handle, 1);
    try s.finished(&.{1});
    try s.end();
    if (s.track(1).reason != .cancelled) return error.NotCancelled;
}

/// The Caps a request can ask for; each one the opened model does not declare must be refused by name.
const request_caps = .{ "penalties", "logit_bias", "logprobs", "grammar", "think_budget", "tool_gate", "images", "video", "audio" };

fn checkRefusals(engine: *const abi.Engine, fx: Fixture) !void {
    var s: Session = undefined;
    try s.open(engine, fx);
    defer s.close();
    const bias = [_]abi.Bias{.{ .token = 0, .bias = 1 }};
    const grammar: abi.Grammar = .{ .kind = .json_object, .text = .{} };
    const think: abi.Think = .{ .budget = 4, .open = .{}, .opened_in_prompt = false, .close = .{}, .end = 0 };
    const gate: abi.ToolGate = .{ .opener = .{}, .names = .{} };
    const pixel = [_]u8{ 0, 0, 0, 0 };
    var refused: [request_caps.len]u64 = undefined;
    var n: usize = 0;
    inline for (request_caps, 0..) |name, i| {
        if (!@field(s.info.caps, name)) {
            var r = s.plain(4);
            var media = [_]abi.Media{.{ .kind = .image, .at = 0, .width = 1, .height = 1, .data = abi.span(u8, &pixel), .hash = 1 }};
            if (comptime std.mem.eql(u8, name, "penalties")) r.sampling.repeat_penalty = 1.1;
            if (comptime std.mem.eql(u8, name, "logit_bias")) r.sampling.logit_bias = abi.span(abi.Bias, &bias);
            if (comptime std.mem.eql(u8, name, "logprobs")) r.logprobs = 1;
            if (comptime std.mem.eql(u8, name, "grammar")) r.grammar = &grammar;
            if (comptime std.mem.eql(u8, name, "think_budget")) r.think = &think;
            if (comptime std.mem.eql(u8, name, "tool_gate")) r.tool_gate = &gate;
            if (comptime std.mem.eql(u8, name, "video")) media[0].kind = .video_frame;
            if (comptime std.mem.eql(u8, name, "audio")) media[0].kind = .audio;
            if (comptime (std.mem.eql(u8, name, "images") or std.mem.eql(u8, name, "video") or std.mem.eql(u8, name, "audio"))) r.media = abi.span(abi.Media, &media);
            std.debug.assert(@field(abi.needs(&r), name));
            const id: u64 = 1 + i;
            const v = try s.submit(id, r, .{});
            if (v.code == .ok) {
                s.finished(&.{id}) catch {};
                return error.UndeclaredCapAccepted;
            }
            if (v.code != .unsupported or std.mem.indexOf(u8, abi.items(u8, v.why), name) == null) return error.RefusalUnnamed;
            refused[n] = id;
            n += 1;
        }
    }
    // A request served after the refusals: by its end a refused one would have spoken.
    if ((try s.submit(0, s.plain(3), .{})).code != .ok) return error.SubmitRefused;
    try s.finished(&.{0});
    try s.end();
    for (refused[0..n]) |id| if (s.track(id).events != 0) return error.RefusedSubmitEmitted;
}

const Concurrent = struct {
    s: *Session,
    k: usize,
    verdicts: [4]abi.Verdict = @splat(.{}),
    done: std.atomic.Value(bool) = .init(false),
    insane: std.atomic.Value(bool) = .init(false),
    threads: [5]?std.Thread = @splat(null), // the status poller, then one submitter per request

    fn submitOne(c: *Concurrent, i: usize) void {
        c.verdicts[i] = c.s.submit(10 + i, c.s.plain(@intCast(5 + i)), .{}) catch .{ .code = .failed };
    }

    fn hammer(c: *Concurrent) void {
        while (!c.done.load(.acquire)) {
            var st: abi.Status = .{};
            c.s.engine.status(c.s.handle, &st);
            var m: abi.Memory = .{};
            c.s.engine.memory(c.s.handle, &m);
            if (st.running + st.waiting > c.k) c.insane.store(true, .release);
        }
    }

    fn join(c: *Concurrent) void {
        c.done.store(true, .release);
        for (&c.threads) |*t| if (t.*) |th| {
            th.join();
            t.* = null;
        };
    }
};

fn checkConcurrent(engine: *const abi.Engine, fx: Fixture) !void {
    var s: Session = undefined;
    try s.open(engine, fx);
    defer s.close();
    var c: Concurrent = .{ .s = &s, .k = @min(4, s.info.max_concurrent) };
    defer c.join();
    c.threads[0] = try std.Thread.spawn(.{}, Concurrent.hammer, .{&c});
    for (0..c.k) |i| c.threads[1 + i] = try std.Thread.spawn(.{}, Concurrent.submitOne, .{ &c, i });
    const ids = [_]u64{ 10, 11, 12, 13 };
    try s.finished(ids[0..c.k]);
    c.join();
    try s.end();
    for (c.verdicts[0..c.k]) |v| if (v.code != .ok) return error.ConcurrentRefused;
    for (ids[0..c.k]) |id| try expectLength(s.track(id));
    if (c.insane.load(.acquire)) return error.StatusInsane;
}

fn checkStatusDuringCallback(engine: *const abi.Engine, fx: Fixture) !void {
    var s: Session = undefined;
    try s.open(engine, fx);
    defer s.close();
    var poller: ?std.Thread = try std.Thread.spawn(.{}, gatePoller, .{&s});
    defer if (poller) |p| {
        s.gate_go.store(true, .release);
        p.join();
    };
    if ((try s.submit(1, s.plain(4), .{ .gate = true })).code != .ok) return error.SubmitRefused;
    try s.finished(&.{1});
    s.gate_go.store(true, .release);
    poller.?.join();
    poller = null;
    try s.end();
}

fn checkStep(engine: *const abi.Engine, fx: Fixture) !void {
    const step = engine.step orelse return;
    var s: Session = undefined;
    try s.open(engine, fx);
    defer s.close();
    try stepUntilIdle(&s, step);
    if ((try s.submit(1, s.plain(4), .{})).code != .ok) return error.SubmitRefused;
    if (!step(s.handle, nowNs() + 2 * std.time.ns_per_ms)) return error.StepNoProgress;
    var n: u32 = 0;
    while (s.track(1).finished == 0) : (n += 1) {
        if (n == 10_000 or !step(s.handle, nowNs() + 2 * std.time.ns_per_ms)) return error.StepNoProgress;
    }
    try stepUntilIdle(&s, step);
    try s.end();
}

/// `false` from `step` tells the host's GPU thread to sleep; an engine with nothing to do must say so.
fn stepUntilIdle(s: *Session, step: anytype) !void {
    for (0..10_000) |_| if (!step(s.handle, nowNs() + 2 * std.time.ns_per_ms)) return;
    return error.StepNeverIdle;
}

fn checkClose(engine: *const abi.Engine, fx: Fixture) !void {
    var s: Session = undefined;
    try s.open(engine, fx);
    defer s.close();
    const ids = [_]u64{ 1, 2, 3 };
    const k = @min(ids.len, s.info.max_concurrent);
    for (ids[0..k]) |id| if ((try s.submit(id, s.plain(unlimited), .{})).code != .ok) return error.SubmitRefused;
    try s.started(ids[0..k]);
    try s.end();
    for (ids[0..k]) |id| {
        const t = s.track(id);
        if (t.finished != 1) return error.CloseDroppedFinished;
        if (t.reason != .cancelled) return error.NotCancelled;
    }
}
