//! Deterministic replay of MTP draft-depth policies over recorded acceptance traces, through
//! the real planner functions. Greedy text diverges with draft depth, so live tok/s cannot
//! rank two policies on the same content; this can, in milliseconds.
//!
//! A trace is one forced-depth-6 request: per round, how many drafts landed. It is unrolled
//! into one flag per generated token, and a policy drafting a position at another chain
//! index than the trace did lands it with the probabilities in `hitProb`.

const std = @import("std");
const testing = std.testing;
const generate = @import("generate.zig");
const round_cost = @import("round_cost.zig");
const mtp_mod = @import("mtp.zig");
const Generator = generate.Generator;

pub const TRACE_DEPTH: u32 = 6;
pub const MAX_TOKENS: u32 = 1024;

pub const Kind = enum { hit, miss, bonus_streak, bonus };
pub const Flag = struct { kind: Kind, idx: u8 };

/// P(draft lands) for a position the trace saw as `f`, drafted now at chain index `at`.
/// Measured by aligning forced-depth 1 and 3 runs with the depth-6 run over their common
/// text (Qwen3.8-27B, 5k positions): a hit lands anywhere, a miss stays one at its own index
/// or deeper, and a DEEP miss is mostly chain degradation, so it lands at a shallower index.
pub fn hitProb(f: Flag, at: u32) f32 {
    const shallower = [_]f32{ 0.71, 0.52, 0.24, 0.15, 0.10, 0.0 };
    return switch (f.kind) {
        .hit => if (at <= f.idx) 1.0 else 0.96,
        .miss => if (at == f.idx) 0.0 else if (at < f.idx) shallower[@min(at, shallower.len - 1)] else 0.12,
        // The token after a full accept was never drafted: easy inside a streak.
        .bonus_streak => 1.0,
        .bonus => 0.85,
    };
}

pub const Request = struct {
    id: u32,
    flags: []Flag,

    fn uniform(self: Request, pos: usize) f32 {
        const h = std.hash.Wyhash.hash(self.id, std.mem.asBytes(&@as(u64, pos)));
        return @as(f32, @floatFromInt(h >> 40)) / @as(f32, @floatFromInt(@as(u64, 1) << 24));
    }

    /// Drafts landed by an `m`-deep chain starting after token `pos`. One uniform per position,
    /// so a deeper chain from the same start never lands fewer.
    pub fn accepted(self: Request, pos: usize, m: u32) u32 {
        var acc: u32 = 0;
        while (acc < m and pos + acc < self.flags.len) : (acc += 1) {
            if (!(self.uniform(pos + acc) < hitProb(self.flags[pos + acc], acc))) break;
        }
        return acc;
    }
};

pub const Corpus = struct {
    arena: std.heap.ArenaAllocator,
    code: []Request = &.{},
    prose: []Request = &.{},
    echo: []Request = &.{},

    pub fn load(gpa: std.mem.Allocator) !Corpus {
        var self = Corpus{ .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        var lists: [3]std.ArrayList(Request) = .{ .empty, .empty, .empty };
        var lines = std.mem.tokenizeScalar(u8, @embedFile("fixtures/mtp_accept_traces.txt"), '\n');
        var id: u32 = 0;
        while (lines.next()) |line| {
            if (line[0] == '#') continue;
            const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return error.BadTrace;
            const class: usize = if (std.mem.eql(u8, line[0..sp], "code")) 0 else if (std.mem.eql(u8, line[0..sp], "prose")) 1 else if (std.mem.eql(u8, line[0..sp], "echo")) 2 else return error.BadTrace;
            const digits = line[sp + 1 ..];
            var flags: std.ArrayList(Flag) = .empty;
            for (digits, 0..) |d, i| {
                const acc: u32 = d - '0';
                if (acc > TRACE_DEPTH) return error.BadTrace;
                for (0..acc) |j| try flags.append(a, .{ .kind = .hit, .idx = @intCast(j) });
                const next_full = i + 1 < digits.len and digits[i + 1] - '0' == TRACE_DEPTH;
                try flags.append(a, if (acc < TRACE_DEPTH) .{ .kind = .miss, .idx = @intCast(acc) } else .{ .kind = if (next_full) .bonus_streak else .bonus, .idx = @intCast(TRACE_DEPTH) });
            }
            try lists[class].append(a, .{ .id = id, .flags = flags.items });
            id += 1;
        }
        self.code = lists[0].items;
        self.prose = lists[1].items;
        self.echo = lists[2].items;
        return self;
    }

    pub fn deinit(self: *Corpus) void {
        self.arena.deinit();
    }
};

/// Wall ms between round ends by drafted depth, and the chunk-A confidence sync.
pub const Machine = struct {
    round_ms: [TRACE_DEPTH + 1]f32,
    sync_ms: f32,
};

/// M4 Max, Qwen3.8-27B 4-bit, short context: decode window / rounds under forced depth.
pub const M4_MAX_27B = Machine{ .round_ms = .{ 0, 36.71, 41.87, 48.57, 56.06, 63.77, 77.31 }, .sync_ms = 6.5 };

/// What a policy asks of one round: draft `m_lo`; when `m_hi > m_lo` the round pays the
/// confidence sync and extends to `m_hi` iff chunk A landed whole (the recorded traces carry
/// no confidences, so a fully accepted chunk A stands in for a confident one).
pub const Plan = Generator.MtpRoundPlan;

pub const Outcome = struct { plan: Plan, drafted: u32, accepted: u32, wall_ms: f32 };

pub const Stats = struct {
    tok_s: f32 = 0,
    depth_rounds: [TRACE_DEPTH + 1]u32 = @splat(0),
    sync_rounds: u32 = 0,
    rounds: u32 = 0,
};

/// Mean per-request tok/s of `policy` over `reqs`, one request at a time, state carried across
/// requests the way a boot carries it. `policy` needs `beginRequest`, `plan`, `observe`, `endRequest`.
pub fn replay(policy: anytype, reqs: []const Request, machine: Machine) Stats {
    var st = Stats{};
    var rate_sum: f32 = 0;
    for (reqs) |req| {
        policy.beginRequest();
        var pos: usize = 0;
        var tok: u32 = 0;
        var ms: f32 = 0;
        while (pos < req.flags.len and tok < MAX_TOKENS) {
            const plan: Plan = policy.plan();
            const two = plan.m_hi > plan.m_lo;
            var m = plan.m_lo;
            var acc = req.accepted(pos, m);
            if (two and acc == m) {
                m = plan.m_hi;
                acc = req.accepted(pos, m);
            }
            const wall = machine.round_ms[m] + (if (two) machine.sync_ms else 0);
            tok += 1 + acc;
            ms += wall;
            pos += acc + 1;
            st.depth_rounds[m] += 1;
            st.rounds += 1;
            if (two) st.sync_rounds += 1;
            policy.observe(.{ .plan = plan, .drafted = m, .accepted = acc, .wall_ms = wall });
        }
        policy.endRequest();
        rate_sum += @as(f32, @floatFromInt(tok)) / ms * 1000.0;
    }
    st.tok_s = rate_sum / @as(f32, @floatFromInt(reqs.len));
    return st;
}

pub const Fixed = struct {
    depth: u32,
    pub fn beginRequest(_: *Fixed) void {}
    pub fn endRequest(_: *Fixed) void {}
    pub fn plan(self: *Fixed) Plan {
        return .{ .m_lo = self.depth, .m_hi = self.depth, .tau_ln = 0 };
    }
    pub fn observe(_: *Fixed, _: Outcome) void {}
};

/// The live controller under a depth policy, built from the same pure functions
/// `mtpRoundPlanInner` and `mtpRoundEndObserve` call, in their order: EV update, NEXT round's
/// plan, then this round's regime and table samples.
pub const Controller = struct {
    const KV: u32 = 1000;
    const WARMUP = Generator.MTP_EV_WARMUP_ROUNDS;

    cap: u32 = Generator.MTP_ADAPTIVE_DEFAULT_CAP,
    costs: Generator.MtpEvCosts = Generator.MTP_EV_DEFAULT_COSTS,
    policy: Generator.MtpDepthPolicy = .accept,
    table: round_cost.Table = .{},
    seed: ?struct { a: [mtp_mod.MAX_DEPTH]f32, m_lo: u32 } = null,

    a: [mtp_mod.MAX_DEPTH]f32 = @splat(Generator.MTP_EV_PRIOR),
    rounds: u32 = 0,
    m_lo_prev: u32 = 1,
    streak: u32 = 0,
    dry_streak: u32 = 0,
    cooldown: u32 = 0,
    regime: Generator.MtpRegime = .{},
    wt: Generator.MtpWidthTrial = .{},
    sync_ms: f32 = 0,
    round_ms: f32 = 0,
    prev_w: ?u32 = null,
    prev_w2: ?u32 = null,
    prev_two: bool = false,
    prev_two2: bool = false,
    pending: ?Outcome = null,

    pub fn beginRequest(self: *Controller) void {
        const keep_table = self.table;
        const keep_seed = self.seed;
        self.* = .{ .cap = self.cap, .costs = self.costs, .policy = self.policy, .table = keep_table, .seed = keep_seed };
        if (self.seed) |s| {
            self.a = s.a;
            self.m_lo_prev = s.m_lo;
            self.rounds = WARMUP;
        }
    }

    pub fn endRequest(self: *Controller) void {
        self.flushPending();
        if (self.rounds >= 8) self.seed = .{ .a = self.a, .m_lo = self.m_lo_prev };
    }

    pub fn plan(self: *Controller) Plan {
        const p = self.planInner();
        self.flushPending();
        return p;
    }

    fn planInner(self: *Controller) Plan {
        if (self.rounds < WARMUP) {
            self.m_lo_prev = 1;
            return .{ .m_lo = 1, .m_hi = 1, .tau_ln = 0 };
        }
        const src = Generator.MtpCostSource.init(self.costs, KV, &self.table);
        var cap = self.cap;
        if (src.fromTable()) cap = @min(self.cap, @max(self.cap, self.table.widestMeasured(src.bucket) orelse self.cap));
        var p = Generator.mtpBasePlan(self.policy, self.a[0..cap], cap, src, self.m_lo_prev + 1);
        if (p.m_lo == self.m_lo_prev) self.streak +|= 1 else self.streak = 0;
        self.m_lo_prev = p.m_lo;
        const dry_threshold = Generator.mtpExtDryThresholdFor(self.sync_ms, self.round_ms);
        if (p.m_hi > p.m_lo and !Generator.mtpExtDryAllows(&self.dry_streak, &self.cooldown, dry_threshold)) {
            p.m_hi = p.m_lo;
            p.tau_ln = 0;
        }
        if (p.m_hi > p.m_lo) {
            if (Generator.mtpRegimeForce(&self.regime, self.rounds) == false) {
                p.m_hi = p.m_lo;
                p.tau_ln = 0;
            }
        }
        if (self.rounds >= self.regime.trial_end) {
            if (Generator.mtpWidthTrialTarget(&self.table, KV, p, self.cap, self.streak >= 2)) |target| {
                if (self.policy == .accept and self.table.rawMs(target, src.bucket) == null) self.wt.startAt(self.rounds);
                const period = if (self.policy == .accept) Generator.mtpProbePeriod(&self.a, src, p.m_lo) else Generator.mtpWidthTrialPeriod(&self.table, KV, p.m_lo);
                if (Generator.mtpWidthTrialForce(&self.wt, self.rounds, period, round_cost.schedulePeriodReread(self.table.layout))) {
                    p = Generator.mtpWidthTrialPlan(target);
                }
            }
        }
        return p;
    }

    pub fn observe(self: *Controller, o: Outcome) void {
        Generator.mtpEvObserve(&self.a, o.drafted, o.accepted, Generator.MTP_EV_EMA_BETA);
        self.rounds += 1;
        const two = o.plan.m_hi > o.plan.m_lo;
        if (two) {
            self.sync_ms = emaMs(self.sync_ms, M4_MAX_27B.sync_ms);
            if (o.drafted > o.plan.m_lo) self.dry_streak = 0 else self.dry_streak +|= 1;
        }
        self.round_ms = emaMs(self.round_ms, o.wall_ms);
        self.pending = o;
    }

    fn emaMs(prev: f32, sample: f32) f32 {
        return if (prev <= 0) sample else prev + Generator.MTP_EV_COST_BETA * (sample - prev);
    }

    fn flushPending(self: *Controller) void {
        const o = self.pending orelse return;
        self.pending = null;
        const two = o.plan.m_hi > o.plan.m_lo;
        const tok: f32 = @floatFromInt(o.accepted + 1);
        // `rounds` was already bumped for this round, as in `mtpRoundEndObserve`.
        const post_warmup = self.rounds >= WARMUP;
        const ev_planned = self.rounds > WARMUP;
        if (post_warmup and !o.plan.width_trial) Generator.mtpRegimeObserve(&self.regime, two, o.plan.m_lo, o.wall_ms, tok);
        if (o.plan.width_trial) self.regime.last_two = false;
        const shape_changed = self.prev_two != two or self.prev_two2 != two;
        self.prev_two2 = self.prev_two;
        self.prev_two = two;
        const transition = shape_changed or (if (self.prev_w) |w| w != o.drafted else true) or (if (self.prev_w2) |w| w != o.drafted else true);
        self.prev_w2 = self.prev_w;
        self.prev_w = o.drafted;
        if (ev_planned and !two) _ = self.table.observe(o.drafted, KV, o.wall_ms, tok, true, transition);
    }
};

fn reportEnabled() bool {
    return std.c.getenv("MTP_REPLAY_REPORT") != null;
}

fn report(comptime fmt: []const u8, args: anytype) void {
    if (reportEnabled()) std.debug.print(fmt, args);
}

/// Live decode tok/s under `MLX_SERVE_MTP_FORCE_DEPTH` 1..6, the 8 code prompts, one boot each
/// (the run `M4_MAX_27B.round_ms` came from).
const LIVE_CODE_FIXED = [_]f32{ 52.31, 64.78, 68.71, 68.23, 65.60, 57.29 };

fn oracleFixed(reqs: []const Request, machine: Machine) struct { depth: u32, tok_s: f32 } {
    var best_d: u32 = 1;
    var best: f32 = 0;
    var d: u32 = 1;
    while (d <= TRACE_DEPTH) : (d += 1) {
        var f = Fixed{ .depth = d };
        const r = replay(&f, reqs, machine).tok_s;
        if (r > best) {
            best = r;
            best_d = d;
        }
    }
    return .{ .depth = best_d, .tok_s = best };
}

test "mtp replay: fixed depths reproduce the live forced-depth ladder on code" {
    var corpus = try Corpus.load(testing.allocator);
    defer corpus.deinit();
    try testing.expectEqual(@as(usize, 8), corpus.code.len);
    var rates: [TRACE_DEPTH]f32 = undefined;
    for (&rates, 1..) |*r, d| {
        var f = Fixed{ .depth = @intCast(d) };
        r.* = replay(&f, corpus.code, M4_MAX_27B).tok_s;
        report("code fixed {d}: replay {d:.2} live {d:.2} ({d:.1}%)\n", .{ d, r.*, LIVE_CODE_FIXED[d - 1], 100.0 * (r.* / LIVE_CODE_FIXED[d - 1] - 1.0) });
        try testing.expect(@abs(r.* / LIVE_CODE_FIXED[d - 1] - 1.0) < 0.02);
    }
    // Same ordering as live: 3 > 4 > 5 > 2 > 6 > 1.
    const order = [_]usize{ 3, 4, 5, 2, 6, 1 };
    for (order[0 .. order.len - 1], order[1..]) |hi, lo| try testing.expect(rates[hi - 1] > rates[lo - 1]);
}

test "mtp replay: the shipped depth policy stays near the best fixed depth on every recorded class" {
    var corpus = try Corpus.load(testing.allocator);
    defer corpus.deinit();
    // One boot serving interleaved content: what the seed and the table carry across a shift.
    var mixed: [24]Request = undefined;
    for (0..8) |i| {
        mixed[3 * i] = corpus.code[i];
        mixed[3 * i + 1] = corpus.echo[i];
        mixed[3 * i + 2] = corpus.prose[i];
    }
    const classes = [_]struct { name: []const u8, reqs: []const Request, floor: f32 }{
        .{ .name = "code", .reqs = corpus.code, .floor = 0.99 },
        .{ .name = "prose", .reqs = corpus.prose, .floor = 0.98 },
        .{ .name = "echo", .reqs = corpus.echo, .floor = 0.975 },
        .{ .name = "mixed", .reqs = &mixed, .floor = 0.965 },
    };
    var oracle_sum: f32 = 0;
    for (classes, 0..) |c, ci| {
        report("{s}:", .{c.name});
        var d: u32 = 1;
        while (d <= TRACE_DEPTH) : (d += 1) {
            var f = Fixed{ .depth = d };
            report(" d{d} {d:.2}", .{ d, replay(&f, c.reqs, M4_MAX_27B).tok_s });
        }
        var o = oracleFixed(c.reqs, M4_MAX_27B);
        // Mixed has no single best depth: its bar is each class at its own.
        if (ci < 3) oracle_sum += o.tok_s else o.tok_s = oracle_sum / 3.0;
        var legacy = Controller{ .policy = .legacy };
        const l = replay(&legacy, c.reqs, M4_MAX_27B);
        var shipped = Controller{};
        const r = replay(&shipped, c.reqs, M4_MAX_27B);
        report("\n    oracle d{d} {d:.2} | legacy {d:.2} ({d:.1}%) depths {any} sync {d}/{d} | accept {d:.2} ({d:.1}%) depths {any}\n", .{ o.depth, o.tok_s, l.tok_s, 100.0 * l.tok_s / o.tok_s, l.depth_rounds[1..], l.sync_rounds, l.rounds, r.tok_s, 100.0 * r.tok_s / o.tok_s, r.depth_rounds[1..] });
        try testing.expectEqual(@as(u32, 0), r.sync_rounds);
        try testing.expect(r.tok_s >= c.floor * o.tok_s);
        try testing.expect(r.tok_s >= l.tok_s);
    }
}
