//! MTP round observation: exact-depth acceptance and round-end updates.
const std = @import("std");
const log = @import("../../../../sdk/util/log.zig");
const io_util = @import("../../../../sdk/util/io_util.zig");
const round_cost = @import("../round_cost.zig");
const group_planner = @import("../mtp_group_planner.zig");
const Generator = @import("../../generate/generator.zig").Generator;
const MTP_EV_WARMUP_ROUNDS = @import("depth.zig").MTP_EV_WARMUP_ROUNDS;
const MtpCostSource = @import("ev_plan.zig").MtpCostSource;
const mtpLowAcceptStreak = @import("adaptive.zig").mtpLowAcceptStreak;
const mtpProbePeriod = @import("planner.zig").mtpProbePeriod;
const mtpRegimeObserve = @import("regime.zig").mtpRegimeObserve;


/// Minimum kv separation before two anchors can identify B. Below it the
/// difference is dominated by round-to-round noise, not by the KV read.
/// Round-end bookkeeping shared by the full- and partial-accept paths:
/// the kv term, the regime gate and the round-cost table all read ONE
/// inter-round wall clock (tok/s is measured between round ends, so
/// per-round work outside the round stopwatch belongs to the width).
/// The depth (1 or 2) with the most expected tokens per ms: this request's
/// acceptance against the model's measured round ms per depth. A depth
/// never measured on this model runs first; every MTP_EXACT_PROBE rounds
/// on the model the other depth runs, so both costs stay current.
pub fn mtpExactDepth(self: *Generator, cap: u32) u32 {
    const ms = &self.xfm.mtp_depth_ms;
    var d: u32 = 1;
    while (d <= cap) : (d += 1) if (ms[d] == 0) return d;
    if (cap < 2) return 1;
    const a = self.mtpExactAcceptance();
    const tok1 = 1 + a;
    const tok2 = tok1 + a * a;
    const best: u32 = if (tok2 / ms[2] > tok1 / ms[1]) 2 else 1;
    self.xfm.mtp_depth_rounds +%= 1;
    return if (self.xfm.mtp_depth_rounds % MTP_EXACT_PROBE == 0) 3 - best else best;
}
pub const MTP_EXACT_PROBE = 64;

/// This request's first-draft acceptance: its own counts over a prior of
/// MTP_EXACT_PRIOR_ROUNDS rounds at the model's rate. A second draft is
/// kept about as often as a first on the same text, so it stands for both.
pub fn mtpExactAcceptance(self: *const Generator) f32 {
    const kept: f32 = @floatFromInt(self.mtp_kept);
    const tried: f32 = @floatFromInt(self.mtp_tried);
    return (kept + MTP_EXACT_PRIOR_ROUNDS * self.xfm.mtp_depth_acc) / (tried + MTP_EXACT_PRIOR_ROUNDS);
}
pub const MTP_EXACT_PRIOR_ROUNDS = 8;

/// `m` drafts, `accepted` kept. A round's cost is the wall time since the
/// previous round ended: everything a round adds to the request, streaming included.
pub fn mtpExactObserve(self: *Generator, m: u32, accepted: u32) void {
    if (m == 0 or m >= self.xfm.mtp_depth_ms.len) return;
    if (self.mtp_cycle_watch) |*w| {
        const cycle_ms = @as(f32, @floatFromInt(w.read())) / std.time.ns_per_ms;
        const ms = &self.xfm.mtp_depth_ms[m];
        // A stall (streaming, another request) is not this depth's cost.
        if (ms.* == 0) ms.* = cycle_ms else if (cycle_ms < 2 * ms.*) ms.* += 0.2 * (cycle_ms - ms.*);
        w.reset();
    } else self.mtp_cycle_watch = io_util.Stopwatch.init(self.timer.io);
    self.mtp_tried += 1;
    self.mtp_kept += @intFromBool(accepted >= 1);
    self.xfm.mtp_depth_acc += 0.02 * (@as(f32, @floatFromInt(@intFromBool(accepted >= 1))) - self.xfm.mtp_depth_acc);
}

pub fn mtpRoundEndObserve(self: *Generator, m: u32, tokens: u32, two_chunk: bool, m_lo: u32, width_trial: bool, round_ms: f32) void {
    if (self.mtp_serial_accept) self.mtpExactObserve(m, tokens - 1);
    if (group_planner.enabled() and self.mtp_planner_width != null) return;
    // An interval spanning a speculative round is neither arm's number.
    self.mtp_serial_clock = null;
    self.mtp_serial_warm = 0;
    const post_warmup = self.mtp_ev_rounds >= MTP_EV_WARMUP_ROUNDS;
    // The table wants rounds the EV controller PLANNED: the round that
    // ends warmup was still the legacy controller's (a w2 sample there
    // anchored the table at 2 on the M4 base 9B).
    const ev_planned = self.mtp_ev_rounds > MTP_EV_WARMUP_ROUNDS;
    const wall = if (post_warmup) self.mtpRegimeWallMs(round_ms) else round_ms;
    const tok: f32 = @floatFromInt(tokens);
    if (post_warmup and self.spec_cost_solo and !width_trial) mtpRegimeObserve(&self.mtp_regime, two_chunk, m_lo, wall, tok);
    // A trial round was a single-chunk shape at another depth: not a
    // regime sample, but the shape its successor transitions from.
    if (width_trial) self.mtp_regime.last_two = false;
    // The table is the cost of the SINGLE-CHUNK shape at each width — the
    // quantity the m_lo loop compares. A two-chunk round chose its width
    // by its own confidence gate (tokens biased high) and paid a sync
    // (ms biased high); observed as width m_hi it read single-chunk 6
    // as better than the 5 -> 6 two-chunk it was measured FROM (M4 Max
    // 27B @16k, -4.5%). The shape question stays the regime gate's.
    const shape_changed = self.spec_round_prev_two_chunk != two_chunk or self.spec_round_prev_two_chunk2 != two_chunk;
    self.spec_round_prev_two_chunk2 = self.spec_round_prev_two_chunk;
    self.spec_round_prev_two_chunk = two_chunk;
    // The realized price of speculation for this request (every non-trial round, extension
    // included). Gated with its one consumer, `mtpAdaptiveVoteFor`.
    if (post_warmup and self.spec_cost_solo and self.mtpAdaptiveModelOk()) self.mtp_price.observe(wall, tok, width_trial);
    if (!width_trial) {
        self.mtp_tok_ema = if (self.mtp_tok_ema == 0) tok else self.mtp_tok_ema + 0.125 * (tok - self.mtp_tok_ema);
        self.mtp_low_accept = mtpLowAcceptStreak(self.mtp_low_accept, self.mtp_tok_ema);
    }
    self.specObserveRound(m, wall, tok, ev_planned and !two_chunk, shape_changed);
}

/// Feed the model's round-cost table (`Transformer.round_cost`, shared
/// by every request and both block decoders). Width = drafts this round
/// (0 = serial). The previous width is tracked on EVERY round so the
/// first observed one is not a transition by accident; a rejected
/// sample that is not an expected drop (contended, transition) logs its
/// numbers — a silent reject is the probe's anti-pattern.
pub fn specObserveRound(self: *Generator, width: u32, wall_ms: f32, tokens: f32, observe: bool, shape_changed: bool) void {
    const transition = shape_changed or
        (if (self.spec_round_prev_width) |p| p != width else true) or
        (if (self.spec_round_prev_width2) |p| p != width else true);
    self.spec_round_prev_width2 = self.spec_round_prev_width;
    self.spec_round_prev_width = width;
    if (!observe) {
        log.debug("  [spec-cost] w{d} ms={d:.2} tok={d:.0} -> unobserved\n", .{ width, wall_ms, tokens });
        return;
    }
    const kv = self.mtpKvLen();
    const v = self.xfm.round_cost.observe(width, kv, wall_ms, tokens, self.spec_cost_solo, transition);
    log.debug("  [spec-cost] w{d} kv={d} ms={d:.2} tok={d:.0} -> {s}\n", .{ width, kv, wall_ms, tokens, @tagName(v) });
    if (v == .bad_sample or v == .out_of_range) {
        log.warn("[spec-cost] table rejected sample ({s}): width={d} kv={d} ms={d:.2} tokens={d:.1}\n", .{ @tagName(v), width, kv, wall_ms, tokens });
    }
}
