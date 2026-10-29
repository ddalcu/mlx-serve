//! The three exactness contracts, checked on the engine interface: drafted output equals plain, a prompt resumed
//! from the prefix cache (and from its SSD tier, across a reopen) equals it served cold, and every stream of a
//! concurrent group equals its solo run. Token for token, greedy and seeded-sampled; a break is a named error.
//! `expect` runs the contracts a model declares (`plugin.Caps.exact_*`); the planted controls run with the fakes.

const std = @import("std");
const abi = @import("engine.zig");
const plugin = @import("plugin.zig");

const io = std.testing.io;

pub const Contracts = struct {
    drafted: bool = false,
    resumed: bool = false,
    concurrent: bool = false,

    pub fn declaredBy(c: plugin.Caps) Contracts {
        return .{ .drafted = c.exact_drafted, .resumed = c.exact_resumed, .concurrent = c.exact_concurrent };
    }
};

pub const Fixture = struct {
    model_dir: []const u8,
    model_type: []const u8,
    config_json: []const u8,
    settings_json: []const u8 = "", // arms the spec mode and the cache tiers the contracts run on
    options_json: []const u8 = "",
    memory_budget: u64 = 1 << 30,
    prompts: []const []const u32, // at most `max_group`; the concurrent group submits them together
    max_tokens: u32 = 24, // replies ignore EOS, so each runs to this length
    ssd: bool = false, // the settings put an SSD tier under the prefix cache: a reopened instance resumes from it
    timeout_ms: u32 = 60_000,
    gate_ms: u32 = 2_000, // how long a group's first event waits for the rest of the group's submits
};

pub const Error = error{
    Open,
    SubmitRefused,
    RequestFailed,
    ReplyOverflow,
    TimedOut,
    NoSpecMode,
    NotDrafted,
    PlainDrafted,
    DraftedDiffers,
    NoPrefixCache,
    UnexpectedCacheHit,
    NotResumed,
    NotResumedFromDisk,
    ResumedDiffers,
    ConcurrentDiffers,
};

/// Greedy, then seeded sampling: a sampled stream draws from the same keys on both sides of every contract.
pub const arms = [_]abi.Sampling{
    .{ .temperature = 0 },
    .{ .temperature = 0.8, .top_p = 0.95, .seed = 0x5eed, .seeded = true },
};

pub const max_group = 8;
const max_reply = 256;

pub fn expect(engine: *const abi.Engine, fx: Fixture, c: Contracts) Error!void {
    std.debug.assert(fx.prompts.len > 0 and fx.prompts.len <= max_group and fx.max_tokens <= max_reply);
    if (c.drafted) try drafted(engine, fx);
    if (c.resumed) try resumed(engine, fx);
    if (c.concurrent) try concurrent(engine, fx);
}

/// Each prompt with drafts off (`Request.no_drafts`), then with the engine's spec mode; both cold.
pub fn drafted(engine: *const abi.Engine, fx: Fixture) Error!void {
    var inst: Instance = undefined;
    try inst.open(engine, fx);
    defer inst.close();
    if (!inst.info.caps.drafts) return error.NoSpecMode;
    var engaged = false;
    for (arms, 0..) |arm, a| for (fx.prompts, 0..) |p, i| {
        const plain = try inst.one(.{ .prompt = p, .sampling = arm, .no_drafts = true });
        const spec = try inst.one(.{ .prompt = p, .sampling = arm });
        if (plain.cached != 0 or spec.cached != 0) return error.UnexpectedCacheHit;
        if (plain.drafted != 0) return error.PlainDrafted;
        engaged = engaged or spec.drafted > 0;
        try expectSame(error.DraftedDiffers, &plain, &spec, a, i);
    };
    if (!engaged) return error.NotDrafted;
}

/// Each prompt cold (`CacheHint.cold`), then retained once and served again from the cache; with an SSD tier,
/// served again by a reopened instance per arm.
pub fn resumed(engine: *const abi.Engine, fx: Fixture) Error!void {
    var cold: [arms.len][max_group]Reply = undefined;
    {
        var inst: Instance = undefined;
        try inst.open(engine, fx);
        defer inst.close();
        if (!inst.info.caps.prefix_cache) return error.NoPrefixCache;
        for (arms, 0..) |arm, a| for (fx.prompts, 0..) |p, i| {
            cold[a][i] = try inst.one(.{ .prompt = p, .sampling = arm });
            if (cold[a][i].cached != 0) return error.UnexpectedCacheHit;
        };
        for (fx.prompts, 0..) |p, i| {
            _ = try inst.one(.{ .prompt = p, .sampling = arms[0], .warm = true });
            for (arms, 0..) |arm, a| {
                const warm = try inst.one(.{ .prompt = p, .sampling = arm, .warm = true });
                if (warm.cached == 0) return error.NotResumed;
                try expectSame(error.ResumedDiffers, &cold[a][i], &warm, a, i);
            }
        }
    }
    if (!fx.ssd) return;
    for (arms, 0..) |arm, a| {
        var inst: Instance = undefined;
        try inst.open(engine, fx);
        defer inst.close();
        for (fx.prompts, 0..) |p, i| {
            const warm = try inst.one(.{ .prompt = p, .sampling = arm, .warm = true });
            if (warm.cached == 0) return error.NotResumedFromDisk;
            try expectSame(error.ResumedDiffers, &cold[a][i], &warm, a, i);
        }
    }
}

/// Each stream alone, then the streams submitted together (at least two: a lone prompt runs twice).
pub fn concurrent(engine: *const abi.Engine, fx: Fixture) Error!void {
    var inst: Instance = undefined;
    try inst.open(engine, fx);
    defer inst.close();
    const n = std.math.clamp(fx.prompts.len, 2, max_group);
    var asks: [max_group]Ask = undefined;
    var solo: [max_group]Reply = undefined;
    for (arms, 0..) |arm, a| {
        for (asks[0..n], solo[0..n], 0..) |*ask, *r, i| {
            ask.* = .{ .prompt = fx.prompts[i % fx.prompts.len], .sampling = arm };
            r.* = try inst.one(ask.*);
        }
        const group = try inst.serve(asks[0..n]);
        for (solo[0..n], group, 0..) |*want, *got, i| {
            if (want.cached != 0 or got.cached != 0) return error.UnexpectedCacheHit;
            try expectSame(error.ConcurrentDiffers, want, got, a, i);
        }
    }
}

/// The first index where two replies differ (a length difference counts at the shorter end); null when equal.
pub fn firstDifference(a: []const u32, b: []const u32) ?usize {
    for (a[0..@min(a.len, b.len)], b[0..@min(a.len, b.len)], 0..) |x, y, i| if (x != y) return i;
    return if (a.len == b.len) null else @min(a.len, b.len);
}

fn expectSame(comptime err: Error, want: *const Reply, got: *const Reply, arm: usize, stream: usize) Error!void {
    const k = firstDifference(want.got(), got.got()) orelse if (want.reason == got.reason) return else want.len;
    std.debug.print("exactness: {s} at token {d} ({s} arm, stream {d})\n", .{ @errorName(err), k, if (arm == 0) "greedy" else "seeded", stream });
    return err;
}

// ── one open instance

const Ask = struct {
    prompt: []const u32,
    sampling: abi.Sampling,
    no_drafts: bool = false,
    warm: bool = false, // retains its state and may read kept state; every other request is served cold
};

const Reply = struct {
    buf: [max_reply]u32 = undefined,
    len: usize = 0,
    cached: u32 = 0,
    drafted: u64 = 0,
    reason: @FieldType(abi.Event, "reason") = .stop,
    finished: bool = false,
    overflow: bool = false,

    fn got(r: *const Reply) []const u32 {
        return r.buf[0..r.len];
    }
};

const Instance = struct {
    engine: *const abi.Engine,
    fx: Fixture,
    host: abi.Host,
    handle: *anyopaque,
    info: abi.Info,
    mu: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    replies: [max_group]Reply = undefined,
    base: u64 = 0, // ids of the current serve: base + index
    group: usize = 0,
    submitted: std.atomic.Value(usize) = .init(0),
    want: std.atomic.Value(usize) = .init(0),

    fn open(s: *Instance, engine: *const abi.Engine, fx: Fixture) Error!void {
        s.* = .{ .engine = engine, .fx = fx, .host = .{ .ctx = s, .log = hostLog, .wake = hostWake }, .handle = undefined, .info = undefined };
        const o: abi.Open = .{
            .model_dir = abi.span(u8, fx.model_dir),
            .model_type = abi.span(u8, fx.model_type),
            .config_json = abi.span(u8, fx.config_json),
            .context = 0,
            .max_concurrent = max_group,
            .memory_budget = fx.memory_budget,
            .settings_json = abi.span(u8, fx.settings_json),
            .options_json = abi.span(u8, fx.options_json),
            .host = &s.host,
        };
        var info: abi.Info = .{ .name = .{}, .version = .{}, .caps = .{}, .threading = .own_thread, .context_window = 0, .max_concurrent = 0, .embedding_dims = 0, .eos = .{}, .startup = .{} };
        var v: abi.Verdict = .{};
        s.handle = engine.open(&o, &info, &v) orelse return error.Open;
        s.info = info;
    }

    fn close(s: *Instance) void {
        s.engine.close(s.handle);
    }

    fn one(s: *Instance, ask: Ask) Error!Reply {
        return (try s.serve(&.{ask}))[0];
    }

    /// Submits `asks` together and waits for every reply; the group's events hold until every submit returned, so
    /// an engine on its own thread cannot finish the first request before the last arrives.
    fn serve(s: *Instance, asks: []const Ask) Error![]const Reply {
        s.mu.lockUncancelable(io);
        s.base += max_group;
        s.group = asks.len;
        for (s.replies[0..asks.len]) |*r| r.* = .{};
        s.mu.unlock(io);
        s.submitted.store(0, .release);
        s.want.store(asks.len, .release);
        for (asks, 0..) |ask, i| {
            const r: abi.Request = .{
                .prompt = abi.span(u32, ask.prompt),
                .max_tokens = s.fx.max_tokens,
                .sampling = ask.sampling,
                .cache = .{ .keep = ask.warm, .cold = !ask.warm },
                .no_drafts = ask.no_drafts,
            };
            if (s.engine.submit(s.handle, s.base + i, &r, .{ .ctx = s, .event = onEvent }).code != .ok) {
                for (0..i) |j| s.engine.cancel(s.handle, s.base + j);
                s.want.store(i, .release);
                s.waitAll(i) catch {};
                return error.SubmitRefused;
            }
            s.submitted.store(i + 1, .release);
        }
        try s.waitAll(asks.len);
        for (s.replies[0..asks.len]) |*r| {
            if (r.overflow) return error.ReplyOverflow;
            if (r.reason != .stop and r.reason != .length) return error.RequestFailed;
        }
        return s.replies[0..asks.len];
    }

    fn waitAll(s: *Instance, n: usize) Error!void {
        const deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromMilliseconds(s.fx.timeout_ms), .clock = .awake });
        while (true) {
            s.mu.lockUncancelable(io);
            var done = true;
            for (s.replies[0..n]) |r| done = done and r.finished;
            s.mu.unlock(io);
            if (done) return;
            if (deadline.untilNow(io).raw.nanoseconds >= 0) return error.TimedOut;
            if (s.engine.step) |step| {
                if (step(s.handle, nowNs() + 2 * std.time.ns_per_ms)) continue;
            }
            s.mu.lockUncancelable(io);
            s.cond.waitTimeout(io, &s.mu, .{ .duration = .{ .raw = .fromMilliseconds(2), .clock = .awake } }) catch {};
            s.mu.unlock(io);
        }
    }

    /// On the engine's thread: wait (bounded) until the whole group is submitted.
    fn hold(s: *Instance) void {
        const deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromMilliseconds(s.fx.gate_ms), .clock = .awake });
        while (s.submitted.load(.acquire) < s.want.load(.acquire)) {
            if (deadline.untilNow(io).raw.nanoseconds >= 0) return;
            io.sleep(.fromMicroseconds(200), .awake) catch {};
        }
    }
};

fn onEvent(ctx: *anyopaque, id: u64, ev: *const abi.Event) callconv(.c) abi.Action {
    const s: *Instance = @ptrCast(@alignCast(ctx));
    s.hold();
    s.mu.lockUncancelable(io);
    defer s.mu.unlock(io);
    defer s.cond.broadcast(io);
    if (id < s.base or id - s.base >= s.group) return .go;
    const r = &s.replies[@intCast(id - s.base)];
    switch (ev.kind) {
        .prefilled => r.cached = ev.cached,
        .audio => {},
        .tokens => for (abi.items(u32, ev.tokens)) |t| {
            if (r.len == r.buf.len) {
                r.overflow = true;
                break;
            }
            r.buf[r.len] = t;
            r.len += 1;
        },
        .finished => {
            r.finished = true;
            r.reason = ev.reason;
            if (ev.stats) |st| r.drafted = st.drafted;
        },
    }
    return .go;
}

fn nowNs() u64 {
    return @intCast(std.Io.Clock.awake.now(io).nanoseconds);
}

fn hostLog(_: *anyopaque, _: u8, _: abi.Str) callconv(.c) void {}

fn hostWake(ctx: *anyopaque) callconv(.c) void {
    const s: *Instance = @ptrCast(@alignCast(ctx));
    s.cond.broadcast(io);
}

// ── tests ──

const testing = std.testing;

test "exactness: the first difference counts a length difference at the shorter end" {
    try testing.expectEqual(@as(?usize, null), firstDifference(&.{ 1, 2, 3 }, &.{ 1, 2, 3 }));
    try testing.expectEqual(@as(?usize, 1), firstDifference(&.{ 1, 2, 3 }, &.{ 1, 9, 3 }));
    try testing.expectEqual(@as(?usize, 2), firstDifference(&.{ 1, 2 }, &.{ 1, 2, 3 }));
}

test "exactness: a plugin's declared caps are the contracts that run" {
    const c = Contracts.declaredBy(.{ .exact_drafted = true, .exact_concurrent = true });
    try testing.expect(c.drafted and !c.resumed and c.concurrent);
    try testing.expectEqual(Contracts{}, Contracts.declaredBy(.{}));
}
