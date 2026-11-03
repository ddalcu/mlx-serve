//! Adaptive MTP serial switching: price window, votes, probes, head release.
const std = @import("std");
const env = @import("../../../../env.zig");
const log = @import("../../../../sdk/util/log.zig");
const round_cost = @import("../round_cost.zig");
const testing = std.testing;
const Generator = @import("../../generate/generator.zig").Generator;
const MTP_ADAPTIVE_CONFIRM = @import("regime.zig").MTP_ADAPTIVE_CONFIRM;
const MTP_ADAPTIVE_MARGIN = @import("regime.zig").MTP_ADAPTIVE_MARGIN;
const MTP_ADAPTIVE_PROBE_TOKENS = @import("regime.zig").MTP_ADAPTIVE_PROBE_TOKENS;
const MtpHeadRef = @import("../mtp_head_ref.zig").MtpHeadRef;
const mtpRoundOff0 = @import("chain.zig").mtpRoundOff0;

pub const MTP_ADAPTIVE_PROBE_WARM: u32 = 2;
/// KV below which the switch does not exist: no vote, no probe. Rounds lose to serial from
/// the 32-64k bucket up; below that the probes cost more than a switch could buy.
pub const MTP_ADAPTIVE_MIN_KV: u32 = 32768;

/// May the adaptive switch run at this context at all? Read before the vote and the probe.
pub fn mtpAdaptiveKvEligible(kv_len: u32, min_kv: u32) bool {
    return kv_len >= min_kv;
}

/// The bucket the adaptive switch speaks in. Every site resolves it the same way (the
/// decision and the re-entry once disagreed and oscillated); the fallback is the table's
/// own grid, never the free `bucketFor`.
pub fn mtpAdaptiveBucketOf(read: ?usize, kv_len: u32, layout: round_cost.Layout) usize {
    return read orelse round_cost.bucketForLayout(kv_len, layout);
}

pub fn mtpAdaptiveBucket(self: *const Generator, kv_len: u32) usize {
    const t = &self.xfm.round_cost;
    return mtpAdaptiveBucketOf(t.bucketToRead(kv_len), kv_len, t.layout);
}

/// Must the realized price window be dropped? It prices one arm in one bucket.
pub fn mtpAdaptiveRegimeMoved(
    prev_bucket: ?usize,
    prev_arm: MtpAdaptiveArm,
    bucket: ?usize,
    arm: MtpAdaptiveArm,
) bool {
    if (prev_arm != arm) return true;
    if (prev_bucket == null and bucket == null) return false;
    if (prev_bucket == null or bucket == null) return true;
    return prev_bucket.? != bucket.?;
}

/// Apply the rule above around a `MtpAdaptive` transition (both call sites).
pub fn mtpAdaptiveSyncWindow(self: *Generator, prev_bucket: ?usize, prev_arm: MtpAdaptiveArm) void {
    if (mtpAdaptiveRegimeMoved(prev_bucket, prev_arm, self.mtp_adaptive.bucket, self.mtp_adaptive.arm)) {
        self.mtp_price.reset();
    }
}

/// Rounds in the realized price window. Full-window-only: a restore's warmup rounds must not vote.
pub const MTP_PRICE_WINDOW: u32 = 16;

/// This request's realized ms per emitted token over a trailing window of speculative
/// rounds. Width trials are skipped, extension rounds kept. A new request starts empty.
pub const MtpPriceWindow = struct {
    ms: [MTP_PRICE_WINDOW]f32 = @splat(0),
    tok: [MTP_PRICE_WINDOW]f32 = @splat(0),
    idx: u32 = 0,
    filled: u32 = 0,

    pub fn reset(self: *MtpPriceWindow) void {
        self.idx = 0;
        self.filled = 0;
    }

    pub fn observe(self: *MtpPriceWindow, round_ms: f32, emitted: f32, width_trial: bool) void {
        if (width_trial) return;
        if (!std.math.isFinite(round_ms) or !(round_ms > 0)) return;
        if (!std.math.isFinite(emitted) or !(emitted > 0)) return;
        const i = self.idx % MTP_PRICE_WINDOW;
        self.ms[i] = round_ms;
        self.tok[i] = emitted;
        self.idx +%= 1;
        if (self.filled < MTP_PRICE_WINDOW) self.filled += 1;
    }

    /// Realized ms per emitted token, or null until the window is full. Summed, not averaged per round.
    pub fn msPerTok(self: *const MtpPriceWindow) ?f32 {
        if (self.filled < MTP_PRICE_WINDOW) return null;
        var ms: f32 = 0;
        var tok: f32 = 0;
        for (self.ms, self.tok) |m, t| {
            ms += m;
            tok += t;
        }
        if (!(ms > 0) or !(tok > 0)) return null;
        const v = ms / tok;
        return if (std.math.isFinite(v)) v else null;
    }
};

pub const MtpAdaptiveVote = enum { undecided, mtp, serial };
pub const MtpAdaptiveArm = enum { undecided, mtp, serial };
pub const MtpAdaptiveAction = enum { none, to_serial, to_mtp };
/// Where a serial block sits in its ramp back to `nextMtp`'s entry invariant.
pub const MtpSerialExit = enum { none, drain, capture };

/// One round's vote. Both prices must be measured: the table's cell (cross-request, but
/// its `tok` column is a workload mixture) and this request's window (per-request, empty
/// until full). A modeled denominator under-predicted committed tokens by 12-31% and
/// switched where MTP was ~10% faster. `serial` needs BOTH; anything missing = `.undecided`.
pub fn mtpAdaptiveVoteFor(
    table_ms_tok: ?f32,
    window_ms_tok: ?f32,
    serial_ms_tok: ?f32,
    margin: f32,
) MtpAdaptiveVote {
    const tbl = table_ms_tok orelse return .undecided;
    const win = window_ms_tok orelse return .undecided;
    const ser = serial_ms_tok orelse return .undecided;
    if (!std.math.isFinite(tbl) or !(tbl > 0)) return .undecided;
    if (!std.math.isFinite(win) or !(win > 0)) return .undecided;
    if (!std.math.isFinite(ser) or !(ser > 0)) return .undecided;
    if (!std.math.isFinite(margin) or margin < 0) return .undecided;
    const bar = ser * (1.0 + margin);
    return if (tbl > bar and win > bar) .serial else .mtp;
}

/// Arm a bounded serial probe, returning the bucket it will measure. One bucket parameter
/// (the flag and the "already measured?" check once keyed on different buckets). The
/// sample lands in the bucket the request is IN.
/// A probe buys the LAST missing input: a request that cannot fill its price window can
/// never use what the probe teaches it and just pays 8 serial tokens.
pub fn mtpSerialProbeUseful(window_ms_tok: ?f32) bool {
    return window_ms_tok != null;
}

pub fn mtpSerialProbeArm(
    t: *round_cost.Table,
    bucket: usize,
    solo: bool,
    idle: bool,
    may_resume: bool,
    useful: bool,
) ?usize {
    if (!solo or !idle or !may_resume or !useful) return null;
    if (bucket >= round_cost.N_BUCKETS) return null;
    if (t.serialMsPerTok(bucket) != null) return null;
    // Bounded retries, not a one-shot flag: an interrupted probe must not burn the bucket's only chance.
    if (t.serial_probes[bucket] >= round_cost.MAX_SERIAL_PROBES) return null;
    t.serial_probes[bucket] += 1;
    return bucket;
}

/// The standing choice and the streak behind it. Pure.
pub const MtpAdaptive = struct {
    arm: MtpAdaptiveArm = .undecided,
    confirm: u32 = 0,
    bucket: ?usize = null,
    /// A re-entry was refused because the head's position bookkeeping is out of sync; latched.
    reentry_declined: bool = false,
    /// The switch left MTP on a module-owned head, so the arm is sticky: no re-entry, probe
    /// or round for the rest of the request. That is what buys the head's release.
    sticky_serial: bool = false,
    /// Round index of the last `round()`: `mtpRoundPlan` has two call sites per round.
    last_round: ?u32 = null,
    switches: u32 = 0,

    pub fn round(self: *MtpAdaptive, round_idx: u32, bucket: usize, vote: MtpAdaptiveVote, need: u32) MtpAdaptiveAction {
        if (self.sticky_serial) return .none;
        if (self.last_round) |r| {
            if (r == round_idx) return .none;
        }
        self.last_round = round_idx;
        // A crossing invalidates the streak.
        if (self.bucket == null or self.bucket.? != bucket) {
            self.bucket = bucket;
            self.confirm = 0;
        }
        switch (vote) {
            .undecided => {
                self.confirm = 0;
                return .none;
            },
            .mtp => {
                self.confirm = 0;
                self.arm = .mtp;
                return .none;
            },
            .serial => {
                self.confirm += 1;
                if (self.confirm < @max(@as(u32, 1), need)) return .none;
                self.confirm = 0;
                self.arm = .serial;
                self.switches += 1;
                return .to_serial;
            },
        }
    }

    /// Latch a refused re-entry: back to the serial arm, never ask again for this request.
    pub fn declineReentry(self: *MtpAdaptive) void {
        self.arm = .serial;
        self.reentry_declined = true;
    }

    /// Make the serial arm permanent for this request (a `to_serial` on a module-owned head).
    pub fn stickSerial(self: *MtpAdaptive) void {
        self.arm = .serial;
        self.sticky_serial = true;
    }

    /// Re-open the decision once the context crosses into another bucket; never inside one: a
    /// re-entry there resumes a head whose history did not grow across the serial block.
    pub fn serialTick(self: *MtpAdaptive, bucket: usize) MtpAdaptiveAction {
        // Sticky outranks the bucket crossing.
        if (self.sticky_serial) return .none;
        if (self.reentry_declined) return .none;
        if (self.arm != .serial) return .none;
        if (self.bucket) |b| if (b == bucket) return .none;
        return self.reopen(bucket);
    }

    fn reopen(self: *MtpAdaptive, bucket: usize) MtpAdaptiveAction {
        self.bucket = bucket;
        self.confirm = 0;
        self.arm = .undecided;
        self.switches += 1;
        return .to_mtp;
    }
};

/// Drift between the head's position bookkeeping and what its next forward demands
/// (`pos_offset == pos_base + seq_offset`; `seq_offset == 0` re-seeds). 0 = in sync.
pub fn mtpHeadPositionDrift(pos_base: i64, seq_offset: usize, next_pos_offset: i64) i64 {
    if (seq_offset == 0) return 0;
    return next_pos_offset - (pos_base + @as(i64, @intCast(seq_offset)));
}

/// The drift the next round would hit. A serial block advances the trunk while the head's
/// `seq_offset` stays put. Null = no head state.
pub fn mtpHeadDriftNow(self: *const Generator) ?i64 {
    if (self.mtp_cache == null) return null;
    const mc = &self.mtp_cache.?;
    return switch (mc.*) {
        .qwen, .plugin => 0,
        .qwen4 => |r| blk: {
            r.t.qwen4MtpActivate(r.st);
            const m = &(r.t.qwen4_mtp orelse break :blk 0);
            const off0 = mtpRoundOff0(self.mtp_hist_stash, mc.step());
            break :blk mtpHeadPositionDrift(@intCast(m.pos_base), m.seq_offset, @intCast(off0 + 1));
        },
    };
}

/// May this request resume MTP after a serial block? An M-RoPE turn ropes the head at an
/// absolute position where the content gap is a wrong answer, so vision turns never come back.
pub fn mtpAdaptiveHeadMayResume(self: *const Generator) bool {
    // The one predicate all three round-start doors ask (re-entry, `apply_stash`, the probe).
    if (self.mtp_adaptive.sticky_serial) return false;
    return self.ctx.mrope_pos == null;
}

/// Does a `to_serial` on this head make the arm sticky? Only a module-owned head.
pub fn stickyOnSerialSwitch(head: ?MtpHeadRef) bool {
    const h = head orelse return false;
    return h.moduleOwned();
}

/// Does this tick newly release the module head? True exactly once per request.
pub fn stickyReleaseNow(sticky: bool, already_released: bool) bool {
    return sticky and !already_released;
}

/// Has this request released the module-owned MTP head?
pub fn mtpModuleHeadReleased(self: *const Generator) bool {
    return self.mtp_head_released;
}

/// Armed but not landed. The release lives in `nextMtp`'s serial branch and the batched
/// tick never calls `nextMtp`, so `scheduler.batchable` keeps the slot out of a group for the
/// one tick the release needs.
pub fn mtpReleasePending(self: *const Generator) bool {
    return self.mtp_adaptive.sticky_serial and !self.mtp_head_released;
}

/// Release the module-owned head at a serial block boundary, after `mtpDetachHead`; never
/// where the switch is decided (that round still runs).
pub fn mtpMaybeReleaseModuleHead(self: *Generator) void {
    if (!stickyReleaseNow(self.mtp_adaptive.sticky_serial, self.mtp_head_released)) return;
    self.mtp_head_released = true;
    log.info(
        "  [mtp] sticky-serial: module head released for the rest of the request (kv {d}) — other slots may claim it\n",
        .{self.mtpKvLen()},
    );
}

pub fn mtpAdaptiveArchEligible(self: *const Generator) bool {
    return self.mtp != null;
}

/// Rounds in a row with the tokens-per-round EMA (bonus included) under this before a
/// sidecar head may pay a serial probe.
pub const MTP_PROBE_LOW_TOK: f32 = 2.0;
pub const MTP_PROBE_LOW_ROUNDS: u32 = 16;

pub fn mtpLowAcceptStreak(streak: u32, round_tokens: f32) u32 {
    return if (round_tokens < MTP_PROBE_LOW_TOK) streak +| 1 else 0;
}

/// The module head's switch was calibrated with an immediate probe; a sidecar head wins
/// wherever it accepts, so only a run of near-serial rounds buys its probe.
pub fn mtpSerialProbeEarned(module_head: bool, low_streak: u32) bool {
    return module_head or low_streak >= MTP_PROBE_LOW_ROUNDS;
}

pub fn mtpAdaptiveSerialStep(self: *Generator, m_lo: u32, kv_len: u32) bool {
    // Read after the plan (m_lo is the width it prices) and before the width trial.
    if (        self.mtpAdaptiveArchEligible() and
        mtpAdaptiveKvEligible(kv_len, MTP_ADAPTIVE_MIN_KV))
    {
        const t = &self.xfm.round_cost;
        const b = self.mtpAdaptiveBucket(kv_len);
        const table_ms_tok = t.msPerTok(m_lo, b);
        const window_ms_tok = self.mtp_price.msPerTok();
        const serial_ms = t.serialMsPerTok(b);
        const vote = mtpAdaptiveVoteFor(table_ms_tok, window_ms_tok, serial_ms, MTP_ADAPTIVE_MARGIN);
        const prev_bucket = self.mtp_adaptive.bucket;
        const prev_arm = self.mtp_adaptive.arm;
        const action = self.mtp_adaptive.round(self.mtp_ev_rounds, b, vote, MTP_ADAPTIVE_CONFIRM);
        self.mtpAdaptiveSyncWindow(prev_bucket, prev_arm);
        if (action == .to_serial) {
            // The round under construction still runs; the switch takes effect next tick.
            log.info(
                "  [mtp] adaptive: bucket {s} mtp table {d:.2} / window {d:.2} ms/tok (w{d}) vs serial {d:.2} ms/tok -> serial (from the next round)\n",
                .{ round_cost.bucketName(self.xfm.round_cost.layout, b), table_ms_tok.?, window_ms_tok.?, m_lo, serial_ms.? },
            );
            self.spec_disabled_runtime = true;
            self.spec_disable_reason = .adaptive;
            // On a module-owned head the switch is one-way; the latch is consumed at the block boundary.
            if (stickyOnSerialSwitch(self.mtp)) self.mtp_adaptive.stickSerial();
            return true;
        }
        // Nothing to decide with: teach the bucket a serial token, once.
        const idle = self.mtp_serial_left == 0 and self.mtp_serial_exit == .none;
        const useful = mtpSerialProbeUseful(window_ms_tok) and mtpSerialProbeEarned(if (self.xfm.qwen4Trunk()) |trunk| trunk.qwen4_mtp != null else false, self.mtp_low_accept);
        if (mtpSerialProbeArm(t, b, self.spec_cost_solo, idle, self.mtpAdaptiveHeadMayResume(), useful)) |own| {
            self.mtp_serial_left = MTP_ADAPTIVE_PROBE_TOKENS;
            log.info(
                "  [mtp] adaptive: bucket {s} has no serial cell -> probing {d} serial tokens\n",
                .{ round_cost.bucketName(self.xfm.round_cost.layout, own), MTP_ADAPTIVE_PROBE_TOKENS },
            );
            return true;
        }
    }
    return false;
}

// ── tests ──


test "mtpAdaptiveVoteFor: a switch needs BOTH measured prices past the margin" {
    const G = Generator;
    const margin = G.MTP_ADAPTIVE_MARGIN;
    try testing.expectEqual(G.MtpAdaptiveVote.serial, G.mtpAdaptiveVoteFor(12.0, 12.0, 10.0, margin));
    // Either price inside the bar keeps speculation.
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(12.0, 9.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(9.0, 12.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(9.0, 9.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(12.0, 12.0, 11.5, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.serial, G.mtpAdaptiveVoteFor(12.0, 12.0, 11.4, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(12.0, 12.0, 12.0, 0.0));

    const nan = std.math.nan(f32);
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(null, 12.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, null, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 12.0, null, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(0.0, 12.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 0.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 12.0, 0.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(nan, 12.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, nan, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 12.0, nan, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 12.0, 10.0, nan));
}

test "mtpAdaptiveVoteFor: four real switch events of the qwen4 A/B" {
    // Four switches the v1 controller made on a 62.7k prose prompt where MTP was the faster
    // arm (56.8 vs 51.6 tok/s): the modeled token count ran 12-31% under the cell's own `tok`.
    const G = Generator;
    const margin = G.MTP_ADAPTIVE_MARGIN;
    const Event = struct {
        name: []const u8,
        table_ms_tok: f32, // Table.msPerTok(m_lo, bucket), from `table=` in [spec-stats]
        serial_ms_tok: f32, // serial_cell= at the moment of the switch
        window_ms_tok: f32, // round_ms EMA / (avg_per_round + 1)
        window_full: bool, // rounds run before the vote >= MTP_PRICE_WINDOW
    };
    const events = [_]Event{
        .{ .name = "b1 long1", .table_ms_tok = 16.12, .serial_ms_tok = 19.20, .window_ms_tok = 15.40, .window_full = true },
        // Voted at round ~3 on prefix-cache-restore warmup rounds; the full-window gate saves it.
        .{ .name = "b1 long2", .table_ms_tok = 15.85, .serial_ms_tok = 18.16, .window_ms_tok = 24.17, .window_full = false },
        .{ .name = "b4 long1", .table_ms_tok = 17.28, .serial_ms_tok = 19.53, .window_ms_tok = 16.63, .window_full = true },
        .{ .name = "b4 long2", .table_ms_tok = 17.91, .serial_ms_tok = 17.08, .window_ms_tok = 15.97, .window_full = true },
    };
    for (events) |e| {
        try testing.expectEqual(
            G.MtpAdaptiveVote.mtp,
            G.mtpAdaptiveVoteFor(e.table_ms_tok, e.table_ms_tok, e.serial_ms_tok, margin),
        );
        // Must not vote serial on any row; an unfilled window is `.undecided`.
        const win: ?f32 = if (e.window_full) e.window_ms_tok else null;
        const got = G.mtpAdaptiveVoteFor(e.table_ms_tok, win, e.serial_ms_tok, margin);
        try testing.expect(got != .serial);
        try testing.expectEqual(
            if (e.window_full) G.MtpAdaptiveVote.mtp else G.MtpAdaptiveVote.undecided,
            got,
        );
    }
    // Rule B ungated reproduces the b1-long2 mistake.
    const b1l2 = events[1];
    try testing.expectEqual(
        G.MtpAdaptiveVote.serial,
        G.mtpAdaptiveVoteFor(30.0, b1l2.window_ms_tok, b1l2.serial_ms_tok, margin),
    );
}

test "mtpAdaptiveBucketOf: the decision and the re-entry resolve ONE bucket (H6 oscillation)" {
    const G = Generator;
    const kv: u32 = 40_000;
    const own = round_cost.bucketFor(kv);

    try testing.expectEqual(own, G.mtpAdaptiveBucketOf(null, kv, .long));
    try testing.expectEqual(@as(usize, own + 1), G.mtpAdaptiveBucketOf(own + 1, kv, .long));
    try testing.expect(G.mtpAdaptiveBucketOf(own + 1, kv, .long) != round_cost.bucketFor(kv));

    // Two resolvers oscillated: switch, crossing, re-enter, switch again.
    var osc = G.MtpAdaptive{};
    _ = osc.round(0, G.mtpAdaptiveBucketOf(own + 1, kv, .long), .serial, 1);
    try testing.expectEqual(G.MtpAdaptiveArm.serial, osc.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.to_mtp, osc.serialTick(round_cost.bucketFor(kv)));

    var ok = G.MtpAdaptive{};
    const b = G.mtpAdaptiveBucketOf(own + 1, kv, .long);
    _ = ok.round(0, b, .serial, 1);
    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, ok.serialTick(G.mtpAdaptiveBucketOf(own + 1, kv, .long)));
    }
    try testing.expectEqual(G.MtpAdaptiveArm.serial, ok.arm);
}

test "mtpAdaptiveRegimeMoved: a switch either way, or a crossing, drops the price window" {
    const G = Generator;
    const A = G.MtpAdaptiveArm;
    try testing.expect(!G.mtpAdaptiveRegimeMoved(null, .undecided, null, .undecided));
    try testing.expect(!G.mtpAdaptiveRegimeMoved(3, .mtp, 3, .mtp));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.mtp, 3, A.serial));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.serial, 3, A.undecided));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.undecided, 3, A.mtp));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.mtp, 4, A.mtp));
    try testing.expect(G.mtpAdaptiveRegimeMoved(null, A.mtp, 4, A.mtp));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.mtp, null, A.mtp));

    var a = G.MtpAdaptive{};
    var pb = a.bucket;
    var pa = a.arm;
    _ = a.round(0, 5, .serial, 1); // first vote: bucket null -> 5, arm -> serial
    try testing.expect(G.mtpAdaptiveRegimeMoved(pb, pa, a.bucket, a.arm));
    pb = a.bucket;
    pa = a.arm;
    _ = a.serialTick(6); // crossing: re-opens the decision
    try testing.expect(G.mtpAdaptiveRegimeMoved(pb, pa, a.bucket, a.arm));
    // A tick that changes nothing must not drop the window, or it can never fill.
    pb = a.bucket;
    pa = a.arm;
    _ = a.serialTick(6);
    try testing.expect(!G.mtpAdaptiveRegimeMoved(pb, pa, a.bucket, a.arm));
}

test "mtpAdaptiveKvEligible: short context is below the floor, and the floor is the default" {
    const G = Generator;
    const floor = G.MTP_ADAPTIVE_MIN_KV;
    try testing.expectEqual(@as(u32, 32768), floor);

    var reached: u32 = 0;
    var i: u32 = 0;
    while (i < 30) : (i += 1) {
        const kv: u32 = 60 + i * 7 + 200;
        if (G.mtpAdaptiveKvEligible(kv, floor)) reached += 1;
    }
    try testing.expectEqual(@as(u32, 0), reached);

    try testing.expect(!G.mtpAdaptiveKvEligible(0, floor));
    try testing.expect(!G.mtpAdaptiveKvEligible(floor - 1, floor));
    try testing.expect(!G.mtpAdaptiveKvEligible(8_200, floor));
    try testing.expect(!G.mtpAdaptiveKvEligible(16_400, floor));
    try testing.expect(G.mtpAdaptiveKvEligible(floor, floor));
    try testing.expect(G.mtpAdaptiveKvEligible(62_755, floor)); // the A/B prose prompt
    try testing.expect(G.mtpAdaptiveKvEligible(374_000, floor)); // the ladder's top rung

    try testing.expect(G.mtpAdaptiveKvEligible(1, 0));
    try testing.expect(!G.mtpAdaptiveKvEligible(62_755, 131_072));
}

test "MtpPriceWindow: reads nothing until FULL, skips width trials, prices per TOKEN" {
    const G = Generator;
    const N = G.MTP_PRICE_WINDOW;
    var w = G.MtpPriceWindow{};
    try testing.expect(w.msPerTok() == null);

    var i: u32 = 0;
    while (i < N - 1) : (i += 1) {
        w.observe(20.0, 2.0, false);
        try testing.expect(w.msPerTok() == null); // still not full
    }
    w.observe(20.0, 2.0, false);
    try testing.expectApproxEqAbs(@as(f32, 10.0), w.msPerTok().?, 1e-4);

    var t = G.MtpPriceWindow{};
    i = 0;
    while (i < N * 2) : (i += 1) t.observe(999.0, 1.0, true);
    try testing.expect(t.msPerTok() == null);
    try testing.expectEqual(@as(u32, 0), t.filled);

    var d = G.MtpPriceWindow{};
    d.observe(std.math.nan(f32), 2.0, false);
    d.observe(20.0, 0.0, false);
    d.observe(-1.0, 2.0, false);
    try testing.expectEqual(@as(u32, 0), d.filled);

    // Summed over the window: 8 rounds at 1 tok/30 ms + 8 at 4 tok/60 ms = 18.0 ms/tok, not 22.5.
    var m = G.MtpPriceWindow{};
    i = 0;
    while (i < N / 2) : (i += 1) m.observe(30.0, 1.0, false);
    while (i < N) : (i += 1) m.observe(60.0, 4.0, false);
    try testing.expectApproxEqAbs(@as(f32, 18.0), m.msPerTok().?, 1e-4);

    i = 0;
    while (i < N) : (i += 1) m.observe(10.0, 2.0, false);
    try testing.expectApproxEqAbs(@as(f32, 5.0), m.msPerTok().?, 1e-4);
    m.reset();
    try testing.expect(m.msPerTok() == null);
}

test "MtpAdaptive: the switch needs CONFIRM consecutive rounds, counted once per round" {
    const G = Generator;
    const need = G.MTP_ADAPTIVE_CONFIRM;
    var a = G.MtpAdaptive{};
    var r: u32 = 0;
    while (r < need - 1) : (r += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, a.round(r, 3, .serial, need));
    }
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, a.round(need - 1, 3, .serial, need));
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);
    try testing.expectEqual(@as(u32, 1), a.switches);

    // `mtpRoundPlan` has two call sites per round: the second call at the same index must not advance the streak.
    var b = G.MtpAdaptive{};
    var i: u32 = 0;
    while (i < need - 1) : (i += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, b.round(i, 3, .serial, need));
        try testing.expectEqual(G.MtpAdaptiveAction.none, b.round(i, 3, .serial, need));
    }
    try testing.expectEqual(G.MtpAdaptiveArm.undecided, b.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, b.round(need - 1, 3, .serial, need));

    var c = G.MtpAdaptive{};
    _ = c.round(0, 3, .serial, need);
    _ = c.round(1, 3, .serial, need);
    try testing.expectEqual(G.MtpAdaptiveAction.none, c.round(2, 3, .mtp, need));
    try testing.expectEqual(G.MtpAdaptiveArm.mtp, c.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.none, c.round(3, 3, .serial, need));
    try testing.expectEqual(G.MtpAdaptiveAction.none, c.round(4, 3, .serial, need));
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, c.round(5, 3, .serial, need));

    var d = G.MtpAdaptive{};
    _ = d.round(0, 3, .serial, need);
    _ = d.round(1, 3, .serial, need);
    try testing.expectEqual(G.MtpAdaptiveAction.none, d.round(2, 3, .undecided, need));
    try testing.expectEqual(G.MtpAdaptiveAction.none, d.round(3, 3, .serial, need));
    try testing.expectEqual(G.MtpAdaptiveArm.undecided, d.arm);

    var e = G.MtpAdaptive{};
    _ = e.round(0, 3, .serial, need);
    _ = e.round(1, 3, .serial, need);
    try testing.expectEqual(G.MtpAdaptiveAction.none, e.round(2, 4, .serial, need));
    try testing.expectEqual(@as(u32, 1), e.confirm);

    var f = G.MtpAdaptive{};
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, f.round(0, 3, .serial, 0));
}

test "MtpAdaptive: crossing re-opens, staying in the bucket never does" {
    const G = Generator;
    var off = G.MtpAdaptive{};
    _ = off.round(0, 3, .serial, 1);
    var i: u32 = 0;
    while (i < 1024) : (i += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, off.serialTick(3));
    }
    try testing.expectEqual(G.MtpAdaptiveArm.serial, off.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.to_mtp, off.serialTick(4));
    try testing.expectEqual(G.MtpAdaptiveArm.undecided, off.arm);
}

test "mtpHeadPositionDrift: re-entry may only resume a head that is provably in sync" {
    const G = Generator;
    try testing.expectEqual(@as(i64, 0), G.mtpHeadPositionDrift(100, 20, 120));
    try testing.expectEqual(@as(i64, 7), G.mtpHeadPositionDrift(100, 20, 127));
    try testing.expectEqual(@as(i64, -3), G.mtpHeadPositionDrift(100, 20, 117));
    try testing.expectEqual(@as(i64, 0), G.mtpHeadPositionDrift(100, 0, 999));
    try testing.expectEqual(@as(i64, 0), G.mtpHeadPositionDrift(0, 0, 0));

    var a = G.MtpAdaptive{};
    _ = a.round(0, 3, .serial, 1);
    a.declineReentry();
    try testing.expect(a.reentry_declined);
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(4)); // crossing
    var i: u32 = 0;
    while (i < 2000) : (i += 1) try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(3));
}

test "mtpSerialProbeArm: bounded RETRIES per bucket, and none once the cell is trusted" {
    const G = Generator;
    var t = round_cost.Table{};
    const b = round_cost.bucketFor(20_000);

    var fired: u32 = 0;
    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        if (G.mtpSerialProbeArm(&t, b, true, true, true, true) != null) fired += 1;
    }
    try testing.expectEqual(@as(u32, round_cost.MAX_SERIAL_PROBES), fired);
    try testing.expect(round_cost.MAX_SERIAL_PROBES > 1); // else it is the old flag

    var u = round_cost.Table{};
    const kv: u32 = 20_000;
    try testing.expect(G.mtpSerialProbeArm(&u, b, true, true, true, true) != null);
    var k: u32 = 0;
    while (k < round_cost.MIN_SAMPLES) : (k += 1) _ = u.observeSerial(kv, 16.0, true, false);
    try testing.expect(u.serialMsPerTok(b) != null);
    try testing.expect(G.mtpSerialProbeArm(&u, b, true, true, true, true) == null);

    var v = round_cost.Table{};
    try testing.expect(G.mtpSerialProbeArm(&v, round_cost.bucketFor(1000), true, true, true, true) != null);
    try testing.expect(G.mtpSerialProbeArm(&v, b, true, true, true, true) != null);

    var w = round_cost.Table{};
    try testing.expect(G.mtpSerialProbeArm(&w, b, false, true, true, true) == null);
    try testing.expect(G.mtpSerialProbeArm(&w, b, true, false, true, true) == null);
    try testing.expect(G.mtpSerialProbeArm(&w, b, true, true, false, true) == null);
    try testing.expect(G.mtpSerialProbeArm(&w, round_cost.N_BUCKETS, true, true, true, true) == null);
    try testing.expectEqual(@as(u8, 0), w.serial_probes[b]);
}

test "S21: a to_serial on a per-request head is not sticky; the release stays a one-shot" {
    const G = Generator;
    // The qwen4 head keeps per-request state now (`Qwen4MtpState`), so a serial switch
    // no longer has to give the module back.
    try testing.expect(!G.stickyOnSerialSwitch(MtpHeadRef{ .qwen4 = undefined }));

    var a = G.MtpAdaptive{};
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, a.round(0, 3, .serial, 1));
    a.stickSerial();
    try testing.expect(a.sticky_serial);
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);

    try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(4));
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(5));
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.round(1, 3, .mtp, 1));
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.round(2, 4, .serial, 1));
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);

    try testing.expect(G.stickyReleaseNow(true, false));
    try testing.expect(!G.stickyReleaseNow(true, true));
    try testing.expect(!G.stickyReleaseNow(false, false));
    try testing.expect(!G.stickyReleaseNow(false, true));
}

test "S21: a KV-only sidecar head keeps its re-arm, and nothing about it changes" {
    const G = Generator;
    try testing.expect(!G.stickyOnSerialSwitch(MtpHeadRef{ .qwen = undefined }));
    try testing.expect(!G.stickyOnSerialSwitch(null));

    var b = G.MtpAdaptive{};
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, b.round(0, 3, .serial, 1));
    try testing.expect(!b.sticky_serial);
    try testing.expectEqual(G.MtpAdaptiveAction.to_mtp, b.serialTick(4));
    try testing.expectEqual(G.MtpAdaptiveArm.undecided, b.arm);
}

test "mtpSerialCaptureReady: the capture step's entry invariant is a RUNTIME check, not an assert" {
    const G = Generator;
    try testing.expect(G.mtpSerialCaptureReady(false, false));
    try testing.expect(!G.mtpSerialCaptureReady(true, false));
    try testing.expect(!G.mtpSerialCaptureReady(false, true));
    try testing.expect(!G.mtpSerialCaptureReady(true, true));
}

test "mtpSerialProbeEarned: a sidecar head probes only after a run of low-acceptance rounds" {
    const G = Generator;
    try testing.expect(G.mtpSerialProbeEarned(true, 0)); // the calibrated module head probes at once
    try testing.expect(!G.mtpSerialProbeEarned(false, G.MTP_PROBE_LOW_ROUNDS - 1));
    try testing.expect(G.mtpSerialProbeEarned(false, G.MTP_PROBE_LOW_ROUNDS));
    var streak: u32 = 0;
    for (0..20) |_| streak = G.mtpLowAcceptStreak(streak, 1.0);
    try testing.expectEqual(@as(u32, 20), streak);
    try testing.expectEqual(@as(u32, 0), G.mtpLowAcceptStreak(streak, 3.0)); // code/echo never earn it
}

test "mtpSerialProbeUseful: a probe buys the LAST missing input, never the first" {
    const G = Generator;
    try testing.expect(!G.mtpSerialProbeUseful(null));
    try testing.expect(G.mtpSerialProbeUseful(12.5));

    var t = round_cost.Table{ .layout = .long };
    const b = round_cost.bucketFor(20_000);
    try testing.expect(G.mtpSerialProbeArm(&t, b, true, true, true, G.mtpSerialProbeUseful(null)) == null);
    try testing.expectEqual(@as(u8, 0), t.serial_probes[b]);
    try testing.expectEqual(@as(?usize, b), G.mtpSerialProbeArm(&t, b, true, true, true, G.mtpSerialProbeUseful(12.5)));
    try testing.expectEqual(@as(u8, 1), t.serial_probes[b]);
}
