//! MTP draft depth and proposal policy (greedy tail, draft temperature, depth decisions).
const depth_bounds = @import("../mtp_depth_bounds.zig");
const std = @import("std");
const env = @import("../../../../env.zig");
const mlx = @import("../../../../mlx/ffi.zig");
const transformer_mod = @import("../../transformer.zig");
const model_mod = @import("../../../../sdk/config/model.zig");
const log = @import("../../../../sdk/util/log.zig");
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");
const Transformer = transformer_mod.Transformer;
const testing = std.testing;
const Generator = @import("../../generate/generator.zig").Generator;
const MtpHeadRef = @import("../mtp_head_ref.zig").MtpHeadRef;
const SamplingParams = @import("../../sampling/params.zig").SamplingParams;
const logitsStepProposal = @import("../../sampling/spec_sampling.zig").logitsStepProposal;
const proposalTestRead = @import("../../sampling/spec_sampling.zig").proposalTestRead;
const shortlistStepProposal = @import("../../sampling/spec_sampling.zig").shortlistStepProposal;


// ── MTP adaptive depth ──
// Unlike the drafter's binary gate, the MTP head has a useful fallback
// BETWEEN "full depth" and "off": depth 1. Measured on Qwen3.6-27B
// (M4 Max, greedy): creative content runs 48% per-draft at depth 2
// (a regression vs AR) but 73% at depth 1 (1.11× AR); code runs 89% at
// depth 2 (1.45× AR). A windowed controller demotes/promotes between
// depths and only disables outright when even depth 1 can't pay for its
// verify overhead.
pub const MTP_DEPTH_WINDOW: u32 = 16; // rounds in the moving window
pub const MTP_DEPTH_SWITCH_WARMUP: u32 = 5; // rounds before re-evaluating after a switch
// Thresholds assume the capture-based rollback (no re-forward on partial
// accept): a rejected draft costs ONLY its own MTP-layer + draft-head
// pass (~2 ms), not a second trunk forward (~30-50 ms). Extra depth pays
// whenever the marginal accept probability clears draft-cost/trunk-cost
// ≈ 0.05-0.10, so the demote floor sits far lower than the old
// re-forward-era 0.60 — hysteresis band keeps switch churn down.
pub const MTP_DEMOTE_BELOW: f32 = 0.40; // per-draft rate at depth > 1 → step down
pub const MTP_PROMOTE_ABOVE: f32 = 0.60; // per-draft rate below configured depth → step up
// Disable floor = the MEASURED depth-1 breakeven plus margin, not a
// quality judgment: a d1 round costs ~AR+6 ms (44 vs 38.4 ms at 8K on
// the 27B, mtp-trace) and yields (1+p) tokens, so speculation
// pays down to p ≈ 0.15 — at p=0.45 it is +27% over AR. The old 0.50
// floor sticky-disabled mid-request on the oQ4e head at long context
// (window rate dips 0.45-0.55) and cratered 16K/32K ladder decode to
// bare AR (24-26 tok/s vs oMLX's 41-47, which never fully disables).
pub const MTP_DISABLE_BELOW: f32 = 0.20; // per-draft rate at depth 1 → disable (sticky)
pub const MTP_PROMOTE_COOLDOWN: u32 = 32; // rounds promotion stays blocked after a demotion

/// Target temperature at or below which the GREEDY proposal still wins:
/// the argmax accepts with `p_target[argmax]`, high only while the target
/// row is sharp. Measured crossover between 0.3 and 0.7 on Flash Next.
pub const MTP_SAMPLED_PROPOSAL_MIN_TEMP: f32 = 0.5;

/// Draft-proposal mode is a property of the REQUEST: a greedy target keeps the argmax proposal (the temp-0 identity
/// contract); a sampled target proposes from the draft sampler, where `min(1, p[argmax])` no longer collapses on a
/// flat row. `top_k == 1` is a deterministic target whose filtered p is one-hot, so its argmax proposal is optimal.
pub fn mtpDraftGreedyFor(target: SamplingParams) bool {
    return target.temperature <= MTP_SAMPLED_PROPOSAL_MIN_TEMP or target.top_k == 1;
}

/// Greedy drafts at chain depth `step` (0 = the round's first draft). The
/// greedy tail samples depth 0 of a sampled request and takes the argmax after it.
pub fn mtpDraftGreedyAt(target: SamplingParams, greedy_tail: bool, step: u32) bool {
    return mtpDraftGreedyFor(target) or (greedy_tail and step > 0);
}

// ── Sharpened stochastic draft proposals (Lightning-class acceptance) ──
// Drafts for a stochastic target are SAMPLED from a fixed sharper
// distribution (constants mirror oMLX's _DRAFT_SAMPLER_*: their comment —
// matched-temp drafting "collapses to ~10-20% on high-entropy content"),
// acceptance is the full Leviathan/Chen ratio min(1, p/q) with q = the
// draft sampler's own filtered distribution, and rejection re-samples
// from normalize(max(p-q, 0)). Output distribution provably equals the
// target's filtered p for ANY proposal q (pinned by the toy-vocab
// exactness test); q only moves the ACCEPTANCE RATE, which is why the
// draft head's quantization never affects correctness.
pub const MTP_DRAFT_TEMP: f32 = 0.6;
pub const MTP_DRAFT_TOP_P: f32 = 0.95;
pub const MTP_DRAFT_TOP_K: u32 = 20;

/// Draft temperature is per FAMILY (`SpecPolicy.draft_temperature`): the sidecar head's 0.6 does not carry over.
pub fn mtpDraftTempFor(config: *const model_mod.ModelConfig) f32 {
    const b = config.plugin orelse return MTP_DRAFT_TEMP;
    const t = b.parsed.spec.policy.draft_temperature;
    return if (std.math.isNan(t)) MTP_DRAFT_TEMP else t;
}

/// Draft-proposal sampler for a round: greedy targets keep greedy drafts
/// (temp-0 identity contract); stochastic targets draft from the fixed
/// sharpened distribution unless greedy is forced. The constants are not
/// tunable: swept on the oQ4e head (prose@0.6, forced depth 3,
/// sampled drafts) at draft temp 0.6/0.7/0.85/1.0 and per-draft
/// acceptance was flat at 49.7-56.9% — no arm beat 0.6.
pub fn mtpDraftSamplingFor(target: SamplingParams, force_greedy: bool, draft_temp: f32) SamplingParams {
    var d = target;
    if (force_greedy or target.temperature <= 0.01) {
        d.temperature = 0.0;
        return d;
    }
    d.temperature = draft_temp;
    d.top_p = MTP_DRAFT_TOP_P;
    d.top_k = MTP_DRAFT_TOP_K;
    return d;
}

/// This request's draft sampler at chain depth `step`: the per-request
/// proposal mode, the greedy tail and the loaded family's draft temperature,
/// resolved in ONE place so no call site can read a different proposal than
/// the one the verify's q came from.
pub fn mtpDraftSampling(self: *const Generator, step: u32) SamplingParams {
    return mtpDraftSamplingFor(
        self.sampling,
        self.mtp_serial_accept or mtpDraftGreedyAt(self.sampling, self.mtp_greedy_tail, step),
        mtpDraftTempFor(&self.xfm.config),
    );
}

/// Full Leviathan acceptance ratio; q clamped so a sampled draft (q > 0
/// by construction) can never divide by an underflowed zero.
pub fn specAcceptProb(p: f32, q_draft: f32) f32 {
    return @min(1.0, p / @max(q_draft, 1e-12));
}

/// Pure depth-policy step. `rate` is the windowed per-draft acceptance
/// probability. Returns the new depth; 0 means "disable speculation".
pub fn mtpNextDepth(current: u32, configured: u32, rate: f32) u32 {
    if (current > 1 and rate < MTP_DEMOTE_BELOW) return current - 1;
    if (current <= 1 and rate < MTP_DISABLE_BELOW) return 0;
    if (current < configured and rate > MTP_PROMOTE_ABOVE) return current + 1;
    return current;
}

/// Confidence-gated depth decision. Demoting is cheap (still
/// speculating) so it reacts on a small sample; DISABLING is sticky and
/// PROMOTING raises verify cost, so both require more evidence. A
/// 16-round window at a true 73% per-draft rate essentially never dips
/// below the 0.50 disable floor; a 5-round window does (observed live:
/// an early-story cold streak disabled a request that would have run
/// 1.11x at depth 1).
pub fn mtpDepthDecision(current: u32, configured: u32, rate: f32, window_rounds: u32, promote_blocked: bool) u32 {
    const next_depth = mtpNextDepth(current, configured, rate);
    if (next_depth == 0 and window_rounds < MTP_DEPTH_WINDOW) return current;
    if (next_depth > current and (window_rounds < 8 or promote_blocked)) return current;
    return next_depth;
}

/// Windowed adaptive-depth update, called once per nextMtp round with
/// that round's (drafted, accepted) counts.
pub fn updateMtpDepth(self: *Generator, drafted: u32, accepted: u32) void {
    const idx = self.mtp_window_idx % MTP_DEPTH_WINDOW;
    self.mtp_window_drafted[idx] = @intCast(drafted);
    self.mtp_window_accepted[idx] = @intCast(accepted);
    self.mtp_window_idx += 1;
    if (self.mtp_promote_cooldown > 0) self.mtp_promote_cooldown -= 1;
    if (self.mtp_rounds_since_switch < MTP_DEPTH_SWITCH_WARMUP) {
        self.mtp_rounds_since_switch += 1;
        return;
    }
    const n = @min(self.mtp_window_idx, MTP_DEPTH_WINDOW);
    var drafted_sum: u32 = 0;
    var accepted_sum: u32 = 0;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        drafted_sum += self.mtp_window_drafted[i];
        accepted_sum += self.mtp_window_accepted[i];
    }
    if (drafted_sum == 0) return;
    const rate = @as(f32, @floatFromInt(accepted_sum)) / @as(f32, @floatFromInt(drafted_sum));
    const decided = mtpDepthDecision(self.mtp_depth_current, self.mtp_depth, rate, n, self.mtp_promote_cooldown > 0);
    const next_depth = if (decided == 0) 0 else depth_bounds.clampWidth(decided, depth_bounds.floorFor(depth_bounds.active, self.mtp_depth), self.mtp_depth);
    if (next_depth == self.mtp_depth_current) return;
    if (next_depth == 0) {
        log.info(
            "  mtp=disabled (windowed per-draft rate {d:.2} < {d:.2} at depth 1)\n",
            .{ rate, MTP_DISABLE_BELOW },
        );
        self.spec_disabled_runtime = true;
        self.spec_disable_reason = .acceptance;
        return;
    }
    log.debug("  [mtp-depth] {d} -> {d} (windowed per-draft rate {d:.2})\n", .{ self.mtp_depth_current, next_depth, rate });
    if (next_depth < self.mtp_depth_current) self.mtp_promote_cooldown = MTP_PROMOTE_COOLDOWN;
    self.mtp_depth_current = next_depth;
    self.mtp_rounds_since_switch = 0;
    // Reset both windows so the new depth is judged on its own rounds.
    self.mtp_window_idx = 0;
    self.mtp_price.reset();
}

// ── MTP EV (expected-value) adaptive controller ──
// Fixed-depth drafting is the warm-decode ceiling: at ~77% per-draft the
// marginal chain decays with index, so one global depth wastes verify
// width on hard stretches and leaves easy stretches (code boilerplate
// where 8/8 accept) under-drafted. The EV controller tracks CONDITIONAL
// per-index acceptance EMAs a[i] = P(draft i accepted | i-1 accepted) and
// plans each round as two chunks: a base chunk `m_lo` (the static EV
// optimum), then — when the head's own confidence on chunk A clears a
// cost-derived threshold tau — an extension to `m_hi`. Only rounds that
// CONSIDER extension pay the one bounded chunk-A sync; when the plan
// collapses to m_lo == m_hi the round is byte-identical in shape to the
// fixed-depth path (no confidence graph, no sync).

/// Default depth cap when `--mtp-max-depth` is not passed (0 = auto) and the
/// EV controller is active. 6 keeps the verify forward at seq 1+6 = 7,
/// the split-K verify-qmm kernel's ceiling on M1-M4. Eligible M5/G17
/// targets use MTP_ADAPTIVE_NAX_CAP instead: their measured NAX round-cost
/// surface makes depths 7/8 profitable. Explicit depths always win.
/// This is the DEFAULT ROW only: `mtp.adaptiveDepthCapForMachine` lowers
/// it on chips whose verify-width cliff was measured (M1 Pro: 4).
pub const MTP_ADAPTIVE_DEFAULT_CAP: u32 = 6;
pub const MTP_ADAPTIVE_NAX_CAP: u32 = 8;
/// Fraction by which a width's cost-per-position may exceed the running
/// best before the measured ladder calls it past the cliff. 5% is well
/// inside the gap the tile cliff opens (the M4 ladder turns up 12% at
/// width 8) and well outside probe rep-to-rep noise, which is bounded
/// below by keeping the MIN of the reps.
pub const MTP_CLIFF_TOLERANCE: f32 = 0.05;
/// One-shot guard for the per-silicon cap log (the row is a property of
/// the machine, so it is worth saying once per process, not per request).
pub var mtp_depth_cap_logged: bool = false;
/// Rounds of legacy (fixed-depth windowed) behavior while the EMAs fill.
/// Warmup, but converges in ROUNDS, not 43 s of offline calibration.
pub const MTP_EV_WARMUP_ROUNDS: u32 = 10;
/// EMA step for the per-index acceptance estimates. 0.15 demotes fast on
/// cold streaks (5 consecutive rejects: 0.72 -> 0.32) without letting a
/// single unlucky round move the plan.
pub const MTP_EV_EMA_BETA: f32 = 0.15;
/// Optimistic prior for unobserved indices. Deliberately ABOVE the
/// measured average per-draft rate (~77%): a deep index is only ever
/// observed when extension fires, and on this cost surface the
/// break-even conditional acceptance for a ramp position is ~0.78 — a
/// realistic prior would sit razor-under it and extension would never
/// get its first trial (measured live: ext_rounds=0 on a pure-echo
/// workload). The tau gate (only near-perfect-confidence rounds extend)
/// plus demote-fast EMAs bound the cost of an optimistic trial to a few
/// rounds.
pub const MTP_EV_PRIOR: f32 = 0.85;
/// Clamp band for the extension confidence threshold.
pub const MTP_EV_TAU_MIN: f32 = 0.05;
pub const MTP_EV_TAU_MAX: f32 = 0.95;
/// Minimum base-round EV (tokens per round-cost) for the exploration
/// horizon: below this the base itself barely beats AR, so keeping an
/// extension position reachable would only add sync tax to a dying
/// speculation (the sticky-disable floor is likely next anyway). 1.10
/// keeps fully-cold EMAs collapsed on every cost profile (cold best_r:
/// generic 1.00, G17-NAX 1.07/1.08).
pub const MTP_EV_EXPLORE_MIN_R: f32 = 1.10;

// ── tests ──


test "greedy tail: depth 0 hands the verify its sampled density, a later depth the one-hot at its argmax" {
    // Bar: off the shortlist and off the full logits alike, each depth's q is the proposal its `[1]` draft came from.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 4096;
    const N: usize = 32;
    var ids: [N]u32 = undefined;
    var logits: [N]f32 = undefined;
    for (0..N) |i| {
        ids[i] = @intCast(37 + i * 101);
        logits[i] = @as(f32, @floatFromInt(i)) * 0.25;
    }
    var sl = mtp_mod.Shortlist{
        .cands = mlx.mlx_array_new_data(&ids, &[_]c_int{@intCast(N)}, 1, .uint32),
        .exact = mlx.mlx_array_new_data(&logits, &[_]c_int{ 1, 1, @intCast(N) }, 3, .float32),
        .rows = V,
    };
    defer sl.deinit();
    const full = try allocator.alloc(f32, @intCast(V));
    defer allocator.free(full);
    @memset(full, -30);
    for (ids, logits) |id, l| full[id] = l;
    const full_logits = mlx.mlx_array_new_data(full.ptr, &[_]c_int{ 1, 1, V }, 3, .float32);
    defer _ = mlx.mlx_array_free(full_logits);

    const target = SamplingParams{ .temperature = 1.0 };
    for (0..2) |d| {
        const sampling = Generator.mtpDraftSamplingFor(target, Generator.mtpDraftGreedyAt(target, true, @intCast(d)), Generator.MTP_DRAFT_TEMP);
        for (0..2) |path| {
            const prop = if (path == 0) try shortlistStepProposal(sl, sampling, s) else try logitsStepProposal(full_logits, sampling, s);
            defer {
                _ = mlx.mlx_array_free(prop.id);
                _ = mlx.mlx_array_free(prop.q);
            }
            try testing.expectEqualSlices(c_int, &.{1}, mlx.getShape(prop.id));
            const q = try proposalTestRead(allocator, prop.q, @intCast(V), s);
            defer allocator.free(q);
            const id_read = try proposalTestRead(allocator, prop.id, 1, s);
            defer allocator.free(id_read);
            const id: usize = @intFromFloat(id_read[0]);
            var sum: f64 = 0;
            var support: usize = 0;
            for (q) |v| {
                sum += v;
                if (v > 0) support += 1;
            }
            try testing.expect(@abs(sum - 1.0) < 1e-4 and q[id] > 0);
            if (d == 0) {
                try testing.expect(support > 1);
            } else {
                try testing.expectEqual(@as(usize, ids[N - 1]), id);
                try testing.expectEqual(@as(usize, 1), support);
            }
        }
    }
}

test "mtpDraftStepPath: a sampled proposal keeps the rerank shortlist" {
    // Bar: a sampled proposal reads the rerank shortlist; only the chunk-A
    // confidence gate needs the full lm_head.
    const P = Generator.DraftStepPath;
    try testing.expectEqual(P.rerank_sampled, Generator.mtpDraftStepPath(true, true, false));
    try testing.expectEqual(P.rerank_greedy, Generator.mtpDraftStepPath(true, false, false));
    // The extension gate wants ln p_head over the whole vocabulary.
    try testing.expectEqual(P.full_logits, Generator.mtpDraftStepPath(true, true, true));
    try testing.expectEqual(P.full_logits, Generator.mtpDraftStepPath(true, false, true));
    // No coarse head: nothing to shortlist.
    try testing.expectEqual(P.full_logits, Generator.mtpDraftStepPath(false, true, false));
    try testing.expectEqual(P.full_logits, Generator.mtpDraftStepPath(false, false, false));
}

test "mtpDraftSamplingFor: sharpened fixed proposal for stochastic targets, greedy stays greedy" {
    // Stochastic target: drafts sample from the FIXED sharpened distribution
    // (temp 0.6 / top_p 0.95 / top_k 20 — oMLX Lightning's _DRAFT_SAMPLER_*
    // constants; matched-temp drafting collapses on high-entropy content).
    const target = SamplingParams{ .temperature = 1.0, .top_p = 1.0, .top_k = 0, .min_p = 0.1, .repeat_penalty = 1.1 };
    const d = Generator.mtpDraftSamplingFor(target, false, Generator.MTP_DRAFT_TEMP);
    try testing.expectEqual(@as(f32, 0.6), d.temperature);
    try testing.expectEqual(@as(f32, 0.95), d.top_p);
    try testing.expectEqual(@as(u32, 20), d.top_k);
    // Non-sampler fields ride through untouched.
    try testing.expectEqual(@as(f32, 1.1), d.repeat_penalty);
    try testing.expectEqual(@as(?f32, 0.1), d.min_p);

    // Greedy target keeps greedy drafts (the temp-0 identity contract).
    const greedy = SamplingParams{ .temperature = 0.0 };
    try testing.expectEqual(@as(f32, 0.0), Generator.mtpDraftSamplingFor(greedy, false, Generator.MTP_DRAFT_TEMP).temperature);
    const near_greedy = SamplingParams{ .temperature = 0.005 };
    try testing.expectEqual(@as(f32, 0.0), Generator.mtpDraftSamplingFor(near_greedy, false, Generator.MTP_DRAFT_TEMP).temperature);

    // Explicit greedy override wins.
    try testing.expectEqual(@as(f32, 0.0), Generator.mtpDraftSamplingFor(target, true, Generator.MTP_DRAFT_TEMP).temperature);

    // The draft temperature is the FAMILY's, passed in, never a global.
    try testing.expectEqual(@as(f32, 1.0), Generator.mtpDraftSamplingFor(target, false, 1.0).temperature);
}

test "mtpDraftTempFor: the plugin's declared draft temperature, else the sidecar head's" {
    var b: @import("../../plugin/config.zig").Binding = .{ .table = undefined };
    var c: model_mod.ModelConfig = .{ .plugin = &b };
    try testing.expectEqual(Generator.MTP_DRAFT_TEMP, mtpDraftTempFor(&c));
    b.parsed.spec.policy.draft_temperature = 1.0;
    try testing.expectEqual(@as(f32, 1.0), mtpDraftTempFor(&c));
    c.plugin = null;
    try testing.expectEqual(Generator.MTP_DRAFT_TEMP, mtpDraftTempFor(&c));
}

test "mtpDraftGreedyFor: the proposal mode is a property of the REQUEST" {
    const sampled = SamplingParams{ .temperature = 1.0 };
    const warm = SamplingParams{ .temperature = 0.7 };
    const cold = SamplingParams{ .temperature = 0.0 };
    const near_cold = SamplingParams{ .temperature = 0.005 };
    // top_k 1 is a deterministic target: its filtered p is one-hot, so the
    // argmax proposal already accepts with probability 1.
    const one_way = SamplingParams{ .temperature = 1.0, .top_k = 1 };

    // Sampled targets propose from the shortlist, greedy ones do not.
    try testing.expect(!Generator.mtpDraftGreedyFor(sampled));
    try testing.expect(!Generator.mtpDraftGreedyFor(warm));
    try testing.expect(Generator.mtpDraftGreedyFor(cold));
    // Bar: a sharp target keeps the argmax, a flat one samples.
    try testing.expect(Generator.mtpDraftGreedyFor(.{ .temperature = 0.3 }));
    try testing.expect(Generator.mtpDraftGreedyFor(near_cold));
    try testing.expect(Generator.mtpDraftGreedyFor(one_way));

    // The temp-0 guard in `mtpDraftSamplingFor` keeps a greedy target greedy.
    try testing.expectEqual(
        @as(f32, 0.0),
        Generator.mtpDraftSamplingFor(cold, Generator.mtpDraftGreedyFor(cold), 1.0).temperature,
    );
}

test "mtpDraftGreedyAt: the greedy tail samples depth 0 only; a greedy request drafts the same way with it on" {
    const sampled = SamplingParams{ .temperature = 1.0, .top_p = 0.95, .top_k = 20 };
    for (0..4) |d| {
        const step: u32 = @intCast(d);
        try testing.expect(!Generator.mtpDraftGreedyAt(sampled, false, step));
        try testing.expectEqual(step > 0, Generator.mtpDraftGreedyAt(sampled, true, step));
    }
    // Only a sampled chain's depths past 0 change: a greedy chain drafts the same sampler at every depth.
    for ([_]f32{ 0.0, 0.005, 0.3, 1.0 }) |t| {
        const target = SamplingParams{ .temperature = t, .top_p = 0.95, .top_k = 20, .seed = 7 };
        const greedy_chain = t <= 0.01 or Generator.mtpDraftGreedyFor(target);
        for (0..mtp_mod.MAX_DEPTH) |d| {
            const step: u32 = @intCast(d);
            const on = Generator.mtpDraftSamplingFor(target, Generator.mtpDraftGreedyAt(target, true, step), 1.0);
            const off = Generator.mtpDraftSamplingFor(target, Generator.mtpDraftGreedyAt(target, false, step), 1.0);
            try testing.expectEqual(greedy_chain or step == 0, std.meta.eql(on, off));
        }
    }
}

test "greedy tail: an argmax draft consumes no draw index (mtpSamplingDraw)" {
    var g: Generator = undefined;
    g.sampling = .{ .temperature = 1.0, .seed = 3, .draw = 5 };
    const greedy = g.mtpSamplingDraw(.{ .temperature = 0.0, .seed = 3 });
    try testing.expectEqual(@as(u64, 0), greedy.draw);
    try testing.expectEqual(@as(u64, 5), g.sampling.draw);
    const sampled = g.mtpSamplingDraw(.{ .temperature = 0.6, .seed = 3 });
    try testing.expectEqual(@as(u64, 5), sampled.draw);
    try testing.expectEqual(@as(u64, 6), g.sampling.draw);
}

test "mtpNextDepth: adaptive depth policy transitions" {
    const configured: u32 = 3;
    // Hot at configured depth: stay.
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(3, configured, 0.9));
    // Sagging at depth > 1: step down (one level at a time). The demote
    // floor is 0.40 under capture-based rollback (a rejected draft costs
    // only its own head pass, not a trunk re-forward).
    try testing.expectEqual(@as(u32, 2), Generator.mtpNextDepth(3, configured, 0.35));
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(2, configured, 0.30));
    // Mid-band (0.40..0.60): hold.
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(3, configured, 0.48));
    try testing.expectEqual(@as(u32, 2), Generator.mtpNextDepth(2, configured, 0.55));
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(1, configured, 0.55));
    // Hot below configured depth: promote (band top is 0.60).
    try testing.expectEqual(@as(u32, 2), Generator.mtpNextDepth(1, configured, 0.73));
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(2, configured, 0.70));
    // Never exceeds configured.
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(3, configured, 0.99));
    // Depth 1 at 0.40: speculation still pays (~+27% over AR at current
    // round costs — the disable floor sits at the measured breakeven, not
    // at "acceptance feels low"). The old 0.50 floor DISABLED here and
    // cratered the oQ4e 16K/32K ladder cells to bare AR (24-26 tok/s).
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(1, configured, 0.40));
    // Depth 1 below the true breakeven (+margin): disable (0).
    try testing.expectEqual(@as(u32, 0), Generator.mtpNextDepth(1, configured, 0.15));
    // Demote-before-disable: a terrible rate at depth 2 still goes through 1.
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(2, configured, 0.10));
}

test "mtpDepthDecision: confidence gates on disable, promote, cooldown" {
    const W = Generator.MTP_DEPTH_WINDOW;
    // Disable needs a FULL window of evidence; small samples hold at depth 1.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.15, 5, false));
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.15, W - 1, false));
    try testing.expectEqual(@as(u32, 0), Generator.mtpDepthDecision(1, 3, 0.15, W, false));
    // Rates the old 0.50 floor killed keep speculating at any window size.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.40, W, false));
    // Promote needs >= 8 rounds AND no active cooldown.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.95, 7, false));
    try testing.expectEqual(@as(u32, 2), Generator.mtpDepthDecision(1, 3, 0.95, 8, false));
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.95, 8, true));
    // Demote reacts on a small sample, even during cooldown.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(2, 3, 0.30, 5, true));
}

test "MTP EV seed round-trips on the qwen4 head; a fresh or absent head reads null" {
    // The in-checkpoint head stores the seed on `Qwen4Mtp` — per loaded model,
    // exactly like the sidecar's `MtpModel.ev_seed_*`. Before this it was a
    // null/no-op stub, so every qwen4 request re-warmed the controller from
    // scratch. Only the two seed fields are touched here; the rest of the head
    // (and of the Transformer) is never read on this path.
    var t: @import("../../../../plugins/qwen4_exp/trunk.zig").Trunk = undefined;
    t.qwen4_mtp = null;
    const ref = MtpHeadRef{ .qwen4 = &t };

    // `--no-mtp`: no head to seed, and setting one must not fault.
    try testing.expect(ref.evSeed() == null);
    var zeros: [mtp_mod.MAX_DEPTH]f32 = @splat(0.5);
    ref.setEvSeed(zeros, 4);
    try testing.expect(ref.evSeed() == null);

    var head: transformer_mod.Qwen4Mtp = undefined;
    head.ev_seed_accept = null;
    head.ev_seed_m_lo = 1;
    t.qwen4_mtp = head;
    // A head that has never published reads null — not a zeroed surface.
    try testing.expect(ref.evSeed() == null);

    var accept: [mtp_mod.MAX_DEPTH]f32 = @splat(0.0);
    accept[0] = 0.91;
    accept[1] = 0.80;
    accept[2] = 0.68;
    ref.setEvSeed(accept, 5);
    const got = ref.evSeed() orelse return error.SeedMissing;
    try testing.expectEqual(@as(u32, 5), got.m_lo);
    try testing.expectEqualSlices(f32, &accept, &got.accept);

    // Overwrite, not accumulate: the LAST healthy request wins.
    zeros[0] = 0.10;
    ref.setEvSeed(zeros, 2);
    const again = ref.evSeed() orelse return error.SeedMissing;
    try testing.expectEqual(@as(u32, 2), again.m_lo);
    try testing.expectEqualSlices(f32, &zeros, &again.accept);
}

test "MTP cross-round pre-draft bills its own trace phase" {
    // The tail-side pre-draft bills its own trace phase; the round total
    // stays the honest sum of all phases.
    var t = Generator.MtpTrace{};
    t.add(.predraft, 2_000_000);
    t.add(.eval, 8_000_000);
    _ = t.endRound(1, 1, false);
    try testing.expectApproxEqAbs(@as(f64, 2.0), t.avgMs(.predraft), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 10.0), t.totalAvgMs(), 1e-9);
}
