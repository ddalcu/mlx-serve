//! DSpark rounds: greedy copy and stochastic arms, the Markov-head policy.
const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const dsv4_mod = @import("../../../plugins/deepseek_v4/deepseek_v4.zig");
const dsv41_mod = @import("../plugin/dsv41_link.zig").deepseek_v41;
const tokenizer_mod = @import("../../../sdk/tokenizer/tokenizer.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const log = @import("../../../sdk/util/log.zig");
const dflash_policy = @import("dflash/policy.zig");
const spec_rounds = @import("../../../sdk/spec/rounds.zig");
const DflashPol = @import("dflash.zig").DflashPol;
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const testing = std.testing;
const Constraint = @import("../grammar/constraint.zig").Constraint;
const DrafterStepResult = @import("drafter.zig").DrafterStepResult;
const Generator = @import("../generate/generator.zig").Generator;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const modules = @import("../modules/call.zig");
const plugin_settle = @import("../plugin/settle.zig");
const SamplingParams = @import("../sampling/params.zig").SamplingParams;
const capAcceptedForTokenBudget = @import("verify.zig").capAcceptedForTokenBudget;
const dsparkCopyLen = @import("verify.zig").dsparkCopyLen;
const mtp_lookup = @import("mtp_lookup.zig");
const mtpBatchedAcceptGraph = @import("mtp/batched_graph.zig").mtpBatchedAcceptGraph;
const mtpBatchedTypicalGraph = @import("mtp/batched_graph.zig").mtpBatchedTypicalGraph;
const mtp_acceptance = @import("../../../sdk/spec/mtp_acceptance.zig");
const MtpBatchedGraph = @import("mtp/batched_graph.zig").MtpBatchedGraph;
const mtpSerialOneToken = @import("mtp/serial.zig").mtpSerialOneToken;
const probsAllPositions = @import("../sampling/spec_sampling.zig").probsAllPositions;
const groupProbsBlock = @import("../sampling/spec_sampling.zig").groupProbsBlock;
const groupProbsRow = @import("../sampling/spec_sampling.zig").groupProbsRow;
const sampleRowsHomogeneous = @import("../sampling/lazy.zig").sampleRowsHomogeneous;
const verifyArgmax = @import("verify.zig").verifyArgmax;
const seedKey = @import("../sampling/lazy.zig").seedKey;
const sampleTokenLazy = @import("../sampling/lazy.zig").sampleTokenLazy;
const CommittedArgmax = @import("verify.zig").CommittedArgmax;
const specDecodeUnsupported = @import("verify.zig").specDecodeUnsupported;
const intree = @import("../plugin/intree.zig");
const abi = @import("../../../abi/plugin.zig");
const plugin_driver = @import("../plugin/driver.zig");


/// Which DSpark accept rule (if any) the dsv4 chokepoint may arm for a
/// request. Pure over its inputs so every arm is unit-testable.
pub const DsparkArm = enum { off, greedy, stochastic };

/// `clean` = nothing consumes logits beyond plain sampling: penalties,
/// grammar and logprobs stay serial on BOTH arms (matching the greedy-only
/// contract this generalizes). A clean greedy request (temp < 0.01 or
/// top_k == 1) gets the raw argmax-equality accept; a clean SAMPLED
/// request gets the stochastic arm (MTP one-hot Leviathan acceptance over
/// filtered target probs).
pub fn dsparkArmFor(sampling: SamplingParams, logprobs_n: u32) DsparkArm {
    const clean = !sampling.penalized() and
        sampling.constraint == null and
        logprobs_n == 0;
    if (!clean) return .off;
    const greedy = sampling.temperature < 0.01 or sampling.top_k == 1;
    if (greedy) return .greedy;
    return .stochastic;
}

/// A sampled round's accept rule. `exact` (the default, lossless): draft k kept with probability min(1, p_k / q_k), the
/// output distribution serial sampling's. `typical` (opt-in, LOSSY: the request's `mtp_acceptance`, `--mtp-typical
/// <delta>` / `--mtp-typical-eps <eps>`): draft k kept iff p_k > min(eps, delta * e^-H(p)), Medusa's form, deterministic,
/// the first reject resampled from p (`mtpBatchedTypicalGraph`, the MTP head's graph). Greedy requests never reach
/// either (argmax equality).
pub const Rule = union(enum) { exact, typical: mtp_acceptance.Typical };

/// The one owner of a sampled round's rule: typical only where the request asked for it (`typical` set) and the drafts
/// are the model's own proposals. A draft copied from the context (DSpark's copy round, as a prompt lookup's) is a point
/// mass the model did not propose: typical acceptance keeps it whenever the target clears its floor, so an echo of the
/// context would feed itself; it keeps the exact rule.
pub fn draftRule(typical: ?mtp_acceptance.Typical, context_draft: bool) Rule {
    if (context_draft) return .exact;
    return if (typical) |t| .{ .typical = t } else .exact;
}

/// The planted bug of the typical verdict's test: the floor's comparison inverted.
pub var typical_plant = false;

/// Shared by dispatch and /props; this disables drafting, not resident stages.
pub fn dsparkEnabled() bool {
    return if (env.get(.MLX_SERVE_DSV4_DSPARK)) |v| v[0] != '0' else true;
}

/// PLD draft+verify decode step. The draft comes from an n-gram lookup
/// over `prompt_ids_owned ++ generated_ids`, NOT a model call — that's
/// what makes PLD model-agnostic and cheap.
///
/// `key_len` is the n-gram size used for matching (default 3). `draft_len`
/// is the maximum number of speculative tokens to verify per step (default
/// 5). Both are clamped to safe upper bounds internally.
///
/// Returns `null` only when generation is already done. When no n-gram
/// match exists (cold start, novel output), falls back to the regular
/// `next()` path and returns a single-token result with `used_lookup=false`.
/// DeepSeek-V4 DSpark step (the arch's OWN block-parallel spec decode).
/// Entry/exit share the v2 spec invariant: module state = prompt +
/// emitted positions, t1 = `next_token_id` NOT in state, pending empty
/// (init's spec branch establishes it; every exit restores it). The
/// heavy lifting — draft, batched verify, snapshot rollback — lives in
/// `deepseek_v4.dsparkRound`; this wrapper only keeps the Generator's
/// bookkeeping (generated_ids, step accounting, the shell cache.step
/// that forwardDsv4WithImpl keys fresh-vs-decode on) in sync.
pub fn nextDspark(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    if (self.done) return null;
    if (!self.dspark_enabled) {
        // Same defensive fallback as nextPld's disarmed arm: the
        // dispatching caller's flag alone must never run a draft.
        const tok_opt = try self.next(allocator);
        if (tok_opt == null) return null;
        const tokens = try allocator.alloc(u32, 1);
        tokens[0] = tok_opt.?;
        return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
    }
    if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;
    // A skipped round is a tick of the serial pipeline; the first drafted round after it drains the pipeline.
    if (self.dspark_pol) |*pol| {
        if (pol.timing) |*t| {
            // Tick to tick: what the round before this one took, with all it left queued.
            if (pol.timing_watch) |*w| t.took(@as(f32, @floatFromInt(w.read())) / @as(f32, std.time.ns_per_ms));
            pol.timing_watch = io_util.Stopwatch.init(self.timer.io);
            if (t.next() == null) {
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        }
        if (try staysPlain(self, pol, allocator)) return try self.dsparkPlainPipelined(allocator);
        pol.plain_watch = null;
        if (self.has_pending_logits or self.has_pending_token) switch (try self.drainPipelineForSpec(allocator)) {
            .drained => |tok| return try mtpSerialOneToken(allocator, tok),
            .stopped => return null,
            .already_clean, .stay_disabled => return error.PipelineHalfState,
        };
    }
    if (try self.checkStop()) return null; // t1 is this block's first emit: stop before drafting
    if (plugin_driver.nativeDrafter(self.xfm) != null) return nativeRound(self, allocator);
    return modules.call("dspark_round", .{ self, allocator });
}

/// Whether this tick is a plain step of the serial pipeline: the policy owes one, or no round pays.
fn staysPlain(self: *Generator, pol: *DflashPol, allocator: std.mem.Allocator) !bool {
    if (pol.pin_rows != 0) return false;
    if (plugin_driver.nativeDrafter(self.xfm) == null) return !policyDrafts(pol) and self.dflashCanPipeline();
    if (!self.dflashCanPipeline()) return false;
    if (pol.timing) |*t| return t.next().?.kind == .plain;
    const ctl = if (pol.rounds) |*c| c else return true;
    if (try copyAhead(self, allocator)) return false;
    return ctl.drafted(nativeBlock(self)).rows == 1;
}

/// Makes this request the one its model's load times the ladder with: it runs `rounds.Timing`'s rounds and ends.
pub fn timeRounds(self: *Generator) void {
    if (plugin_driver.nativeDrafter(self.xfm) == null) return;
    if (self.dspark_pol) |*pol| {
        pol.rounds = null;
        pol.timing = .{ .block = nativeBlock(self) };
    }
}

/// The timing request has ended: the model's rounds are priced on what it measured.
pub fn keepLadder(self: *Generator) void {
    const t = if (self.dspark_pol) |*pol| (if (pol.timing) |*x| x else return) else return;
    const l = t.ladder() orelse return log.warn("[rounds] the round timing did not finish; the drafter stays off\n", .{});
    self.xfm.plugin.?.ladder = l;
    var copies: [160]u8 = undefined;
    var drafts: [160]u8 = undefined;
    log.info("[rounds] plain step {d:.1} ms; a round by its drafts: copied {s}; drafted {s}\n", .{ l.plain, joinMs(&copies, &l.copies), joinMs(&drafts, l.drafts[0 .. l.block + 1]) });
}

fn joinMs(buf: []u8, ms: []const f32) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    for (ms, 0..) |x, i| w.print("{s}{d:.1}", .{ if (i > 0) "/" else "", x }) catch break;
    return w.buffered();
}

/// Whether the committed tail repeats the context far enough that the next round likely copies: a served drafter's
/// round then runs whatever the acceptance counts say, since a copy is looked up only where a round runs.
fn copyAhead(self: *Generator, allocator: std.mem.Allocator) !bool {
    return (try self.mtpLookupIndex(allocator)).tailAgreement() >= mtp_lookup.STRONG_SUFFIX;
}

/// A copy round's bookkeeping (`[spec-stats] copy_rounds=`).
pub fn noteCopy(self: *Generator, drafted: u64, accepted: u64) void {
    self.dspark_copies += 1;
    self.dspark_copy_drafted += drafted;
    self.dspark_copy_accepted += accepted;
}

/// The drafts a served drafter's round holds: its block, inside what a ladder prices.
fn nativeBlock(self: *const Generator) u32 {
    return @min(self.xfm.config.plugin.?.parsed.spec.native_block -| 1, spec_rounds.MAX_DRAFTS);
}

/// What a native round runs: a copy of the context (`copy`, which agreed `suffix` tokens back), the drafter's block
/// of at most `drafts`, or a plain step.
const NativePlan = struct { kind: spec_rounds.Timing.Kind, drafts: u32 = 0, copy: []const u32 = &.{}, suffix: u32 = 0 };

/// The round after `t1`, of at most `budget` drafts: the one the request's controller prices highest, the one the
/// load's timing runs next, or (no policy) the drafter's block.
fn nativePlan(self: *Generator, allocator: std.mem.Allocator, budget: u32, garbage: *[spec_rounds.MAX_DRAFTS]u32) !NativePlan {
    const block = @min(nativeBlock(self), budget);
    const pol = if (self.dspark_pol) |*p| p else return .{ .kind = .draft, .drafts = block };
    if (pol.timing) |*t| {
        const r = t.next().?;
        // A round costs what its rows do, whatever their ids.
        @memset(garbage[0..r.drafts], self.next_token_id);
        return .{ .kind = r.kind, .drafts = r.drafts, .copy = if (r.kind == .copy) garbage[0..r.drafts] else &.{} };
    }
    const ctl = if (pol.rounds) |*c| c else return .{ .kind = .plain };
    var plan: NativePlan = .{ .kind = .plain };
    if ((try self.mtpLookupIndex(allocator)).matchBest(self.next_token_id, @min(spec_rounds.MAX_DRAFTS, budget))) |m| if (m.suffix >= mtp_lookup.STRONG_SUFFIX) {
        plan.copy = m.draft;
        plan.suffix = m.suffix;
    };
    const r = ctl.choose(block, plan.suffix, @intCast(plan.copy.len));
    plan.kind = r.kind;
    plan.drafts = r.drafts;
    plan.copy = if (r.kind == .copy) plan.copy[0..r.drafts] else &.{};
    return plan;
}

/// What the round `plan` verified and kept teaches the request's counts.
fn noteNative(self: *Generator, plan: NativePlan, verified: u32, accepted: u32) void {
    self.dspark_stats.record(verified, accepted);
    if (plan.kind == .copy and verified > 0) self.noteCopy(verified, accepted);
    const ctl = if (self.dspark_pol) |*pol| (if (pol.rounds) |*c| c else return) else return;
    switch (plan.kind) {
        .copy => ctl.copy.observe(plan.suffix, verified, accepted),
        // A block its drafter was not sure enough of to verify a draft is a miss.
        .draft => ctl.acc.observe(@max(verified, 1), accepted),
        .plain => ctl.acc.idle(),
    }
}

/// A sampled round of the served plugin's drafter: the plugin drafts from its own law (or verifies the copy, point
/// masses), the engine accepts under the request's sampling, the plugin keeps what was accepted.
fn nativeSampledRound(self: *Generator, allocator: std.mem.Allocator, plan: NativePlan, ask: plugin_driver.RoundAsk) !?DrafterStepResult {
    var asked = ask;
    if (plan.kind == .draft) asked.law = .{ .temperature = self.sampling.temperature, .seed = self.prng.random().int(u64) };
    var v = try plugin_driver.nativeVerify(self.xfm, &self.ctx, self.next_token_id, asked);
    defer _ = mlx.mlx_array_free(v.logits);
    defer if (v.law.ctx != null) {
        _ = mlx.mlx_array_free(v.law);
    };
    const ids = v.round.tokens[0..v.round.n];
    // A failure below leaves the plugin's verify open, which is the model's, not this request's: close it.
    var closed = false;
    errdefer if (!closed) {
        _ = plugin_driver.nativeFinish(self.xfm, &self.ctx, 0) catch 0;
    };
    // The plugin's round does not say which drafts it copied from the context: the exact rule.
    const drawn = try stochasticVerdict(self, v.logits, ids, self.xfm.config.vocab_size, ask.cap, if (v.law.ctx != null) v.law else null, .exact);
    const verdict = stopPending(drawn, ids, ask.stops);
    closed = true;
    const position = try plugin_driver.nativeFinish(self.xfm, &self.ctx, verdict.accepted);
    const tokens = try allocator.dupe(u32, ids[0 .. 1 + verdict.accepted]);
    errdefer allocator.free(tokens);
    noteNative(self, plan, v.round.rows -| 1, drawn.accepted);
    return self.commitDsparkRound(allocator, .{ .tokens = tokens, .accepted = verdict.accepted, .next_token = verdict.next_token }, @intCast(position));
}

/// Whether `self`'s next round can run in a group round of a served drafter: a request at the spec invariant that is no
/// timing request and has nothing pending (a slot the controller sends plain rides the group on its one row). Anything
/// else ticks solo.
pub fn groupReady(self: *Generator) !bool {
    if (self.done or !self.dspark_enabled) return false;
    if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return false;
    if (self.has_pending_logits or self.has_pending_token or self.ctx.ablation != null) return false;
    if (env.on(.MLX_SERVE_KEYED_SAMPLING) and keyedSerial(self.sampling)) return false;
    const pol = if (self.dspark_pol) |*p| p else return false;
    if (pol.timing != null) return false;
    return !(try self.checkStop());
}

/// One round for each of `gens` (two or more slots of one served-drafter model, each `groupReady`): the slots draft and their
/// blocks are verified as one forward, every accept is started on the GPU together (a sampled slot's under its own sampling,
/// the greedy slots' as one argmax over their rows), and each slot commits what it kept. `out[k]` is `gens[k]`'s committed
/// block, its tokens owned through `allocs[k]`.
pub fn nativeGroupRound(gens: []const *Generator, allocs: []const std.mem.Allocator, out: []DrafterStepResult) !void {
    const max = plugin_driver.max_native_group;
    const m = gens.len;
    std.debug.assert(m >= 2 and m <= max);
    var plans: [max]NativePlan = undefined;
    var garbage: [max][spec_rounds.MAX_DRAFTS]u32 = undefined;
    var ctxs: [max]*ForwardCtx = undefined;
    var t1s: [max]u32 = undefined;
    var asks: [max]plugin_driver.RoundAsk = undefined;
    for (gens, allocs, 0..) |gen, alloc, i| {
        try plugin_settle.activate(gen.xfm, &gen.ctx);
        const budget = @min(capAcceptedForTokenBudget(std.math.maxInt(u32), gen.completion_tokens, gen.max_tokens), abi.max_round_tokens - 1);
        plans[i] = try nativePlan(gen, alloc, budget, &garbage[i]);
        // A slot the controller sends plain rides the group on its one row.
        if (plans[i].kind == .plain) plans[i].drafts = 0;
        const ladder: ?*const spec_rounds.Ladder = if (gen.dspark_pol) |*pol| (if (pol.rounds) |*c| &c.ladder else null) else null;
        // A sampled block is as wide as the counts chose; a greedy one is cut by the drafter (`nativeRound`).
        const cap = if (plans[i].kind == .draft and !gen.dspark_stochastic) @min(nativeBlock(gen), budget) else plans[i].drafts;
        var ask: plugin_driver.RoundAsk = .{ .cap = cap, .stops = gen.eos_token_ids, .copy = plans[i].copy, .round_ms = if (ladder) |l| l.drafts[0 .. l.block + 1] else &.{} };
        if (plans[i].kind == .draft and gen.dspark_stochastic) ask.law = .{ .temperature = gen.sampling.temperature, .seed = gen.prng.random().int(u64) };
        ctxs[i] = &gen.ctx;
        t1s[i] = gen.next_token_id;
        asks[i] = ask;
    }
    const xfm = gens[0].xfm;
    // Fused GPU-side draft tree + sampling + accept path (zero host-side dense prob scatter or mid-round sync).
    if (plugin_driver.nativeStepGroupAvail(xfm)) {
        var step_asks = asks;
        var copy_slices: [max][]const u32 = @splat(&.{});
        for (gens, allocs, 0..) |gen, alloc, i| {
            const budget = @min(capAcceptedForTokenBudget(std.math.maxInt(u32), gen.completion_tokens, gen.max_tokens), abi.max_round_tokens - 1);
            step_asks[i].cap = budget;
            step_asks[i].top_k = if (gen.dspark_stochastic) gen.sampling.top_k else 0;
            step_asks[i].top_p = if (gen.dspark_stochastic) gen.sampling.top_p else 1.0;
            step_asks[i].min_p = if (gen.dspark_stochastic) (gen.sampling.min_p orelse 0.0) else 0.0;
            if (gen.dspark_stochastic and step_asks[i].law == null) {
                step_asks[i].law = .{ .temperature = gen.sampling.temperature, .seed = gen.prng.random().int(u64) };
            }
            if (step_asks[i].copy.len == 0) {
                if ((try gen.mtpLookupIndex(alloc)).matchBest(gen.next_token_id, @min(spec_rounds.MAX_DRAFTS, budget))) |match| {
                    if (match.suffix >= mtp_lookup.STRONG_SUFFIX) copy_slices[i] = match.draft;
                }
                step_asks[i].copy = copy_slices[i];
            }
        }
        var rounds_out: [max]abi.Round = undefined;
        const used_fused = blk: {
            plugin_driver.nativeStepGroup(xfm, ctxs[0..m], t1s[0..m], step_asks[0..m], rounds_out[0..m]) catch |e| switch (e) {
                error.PluginStepGroupUnsupported => break :blk false,
                else => return e,
            };
            break :blk true;
        };
        if (used_fused) {
            for (gens, allocs, rounds_out[0..m], out, 0..) |gen, alloc, r, *res, k| {
                const tokens = try alloc.dupe(u32, r.tokens[0..r.n]);
                errdefer alloc.free(tokens);
                noteNative(gen, plans[k], r.rows -| 1, r.accepted);
                gen.dspark_group_rounds += 1;
                res.* = (try gen.commitDsparkRound(alloc, .{ .tokens = tokens, .accepted = r.accepted, .next_token = r.next_token }, @intCast(r.position))).?;
            }
            return;
        }
    }
    const vocab = xfm.config.vocab_size;
    var verified: [max]plugin_driver.Verified = undefined;
    try plugin_driver.nativeVerifyGroup(xfm, ctxs[0..m], t1s[0..m], asks[0..m], verified[0..m]);
    defer for (verified[0..m]) |v| {
        _ = mlx.mlx_array_free(v.logits);
        if (v.law.ctx != null) _ = mlx.mlx_array_free(v.law);
    };
    // A failure before the finish leaves the plugin's verifies open, which are the model's, not these requests': close them.
    var closed = false;
    errdefer if (!closed) {
        var none: [max]u32 = @splat(0);
        plugin_driver.nativeFinishGroup(xfm, ctxs[0..m], none[0..m]) catch {};
    };
    var sampled: [max]usize = undefined;
    var greedy: [max]usize = undefined;
    var ns: usize = 0;
    var ng: usize = 0;
    for (gens, 0..) |gen, k| {
        if (gen.dspark_stochastic) {
            sampled[ns] = k;
            ns += 1;
        } else {
            greedy[ng] = k;
            ng += 1;
        }
    }
    var pend: [max]?PendingAccept = @splat(null);
    defer for (pend[0..m]) |*p| if (p.*) |*x| x.deinit();
    // Sampled slots that filter alike send their rows through the filter as one block: it costs a sort a row and a fixed
    // price a call.
    var params: [max]SamplingParams = undefined;
    for (sampled[0..ns], params[0..ns]) |k, *x| x.* = gens[k].sampling;
    const alike = ns > 0 and sampleRowsHomogeneous(params[0..ns]);
    var filtered: mlx.mlx_array = .{ .ctx = null };
    defer if (filtered.ctx != null) {
        _ = mlx.mlx_array_free(filtered);
    };
    if (alike) {
        var logits: [max]mlx.mlx_array = undefined;
        for (sampled[0..ns], logits[0..ns]) |k, *x| x.* = verified[k].logits;
        const stacked = try stackRows(logits[0..ns], vocab, xfm.s);
        defer _ = mlx.mlx_array_free(stacked);
        filtered = try groupProbsBlock(stacked, gens[sampled[0]].sampling, xfm.s);
    }
    var at: c_int = 0;
    for (sampled[0..ns]) |k| {
        const v = verified[k];
        const ids = v.round.tokens[0..v.round.n];
        defer at += @intCast(ids.len);
        if (ids.len < 2) continue;
        const law: ?mlx.mlx_array = if (v.law.ctx != null) v.law else null;
        if (alike) {
            const mine = try groupProbsRow(filtered, at, @intCast(ids.len), xfm.s);
            defer _ = mlx.mlx_array_free(mine);
            pend[k] = try acceptStartProbs(gens[k], mine, gens[k].nextDraw(), ids, vocab, law);
        } else pend[k] = try acceptStart(gens[k], v.logits, ids, vocab, law);
    }
    // The greedy slots' rows pick as one block, the trunk's own argmax under the serial sampler's suppress mask.
    var picked: ?CommittedArgmax = null;
    defer if (picked) |*p| p.deinit();
    var greedy_rows: usize = 0;
    if (ng > 0) {
        var logits: [max]mlx.mlx_array = undefined;
        for (greedy[0..ng], logits[0..ng]) |k, *x| {
            x.* = verified[k].logits;
            greedy_rows += verified[k].round.n;
        }
        const stacked = try stackRows(logits[0..ng], vocab, xfm.s);
        defer _ = mlx.mlx_array_free(stacked);
        picked = try verifyArgmax(stacked, xfm.suppress_mask, xfm.s);
    }
    // Every slot's verify and accept in one evaluation.
    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        for (pend[0..m]) |*p| if (p.*) |*x| {
            _ = mlx.mlx_vector_array_append_value(ev, x.bg.accept_p);
            _ = mlx.mlx_vector_array_append_value(ev, x.bg.corr_samples);
        };
        if (picked) |p| _ = mlx.mlx_vector_array_append_value(ev, p.lazy());
        try mlx.check(mlx.mlx_async_eval(ev));
    }
    const picks: []const i32 = if (picked) |p| blk: {
        try mlx.check(mlx.mlx_array_eval(p.lazy()));
        break :blk try p.ids(greedy_rows);
    } else &.{};
    var drawn: [max]Verdict = undefined;
    var verdicts: [max]Verdict = undefined;
    var accepted: [max]u32 = undefined;
    var row: usize = 0;
    for (gens, verified[0..m], pend[0..m], 0..) |gen, v, *p, k| {
        const ids = v.round.tokens[0..v.round.n];
        if (gen.dspark_stochastic) {
            drawn[k] = if (p.*) |*x| try acceptRead(gen, x, asks[k].cap) else try stochasticVerdict(gen, v.logits, ids, vocab, asks[k].cap, null, .exact);
            verdicts[k] = stopPending(drawn[k], ids, asks[k].stops);
        } else {
            drawn[k] = greedyVerdict(ids, picks[row .. row + ids.len]);
            verdicts[k] = drawn[k];
            row += ids.len;
        }
        accepted[k] = verdicts[k].accepted;
    }
    closed = true;
    try plugin_driver.nativeFinishGroup(xfm, ctxs[0..m], accepted[0..m]);
    for (gens, allocs, verified[0..m], out, 0..) |gen, alloc, v, *res, k| {
        const ids = v.round.tokens[0..v.round.n];
        const tokens = try alloc.dupe(u32, ids[0 .. 1 + verdicts[k].accepted]);
        errdefer alloc.free(tokens);
        noteNative(gen, plans[k], v.round.rows -| 1, drawn[k].accepted);
        gen.dspark_group_rounds += 1;
        res.* = (try gen.commitDsparkRound(alloc, .{ .tokens = tokens, .accepted = verdicts[k].accepted, .next_token = verdicts[k].next_token }, @intCast(ctxs[k].cache.step))).?;
    }
}

/// What a greedy slot's round keeps: the drafts (`ids[1..]`) that are the trunk's pick for their row (`want[k]` follows
/// `ids[k]`), and the pick after them.
fn greedyVerdict(ids: []const u32, want: []const i32) Verdict {
    var a: u32 = 0;
    while (a + 1 < ids.len and want[a] == @as(i32, @intCast(ids[1 + a]))) a += 1;
    return .{ .accepted = a, .next_token = @intCast(want[a]) };
}

/// `logits` `[rows_k, V]` blocks (the verify rows of several slots) stacked as one `[1, sum of rows, V]` block.
fn stackRows(logits: []const mlx.mlx_array, vocab: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new_data(logits.ptr, logits.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_concatenate_axis(&all, vec, 0, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const shape = [_]c_int{ 1, mlx.getShape(all)[0], @intCast(vocab) };
    try mlx.check(mlx.mlx_reshape(&out, all, &shape, 3, s));
    return out;
}

/// One round of the served plugin's own drafter (`NativeDrafter.step`): its tokens and the trunk token after them.
fn nativeRound(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    try plugin_settle.activate(self.xfm, &self.ctx);
    // A forced commit (`commitForcedTokens`) leaves its last token in the model's state, whatever `dspark_pol` is:
    // publish it and sample `t1` from its logits, so the drafter never takes it a second time.
    if (self.has_pending_logits or self.has_pending_token) switch (try self.drainPipelineForSpec(allocator)) {
        .drained => |tok| return try mtpSerialOneToken(allocator, tok),
        .stopped => return null,
        .already_clean, .stay_disabled => return error.PipelineHalfState,
    };
    const budget = @min(capAcceptedForTokenBudget(std.math.maxInt(u32), self.completion_tokens, self.max_tokens), abi.max_round_tokens - 1);
    var garbage: [spec_rounds.MAX_DRAFTS]u32 = undefined;
    const plan = try nativePlan(self, allocator, budget, &garbage);
    if (plan.kind == .plain and self.dflashCanPipeline()) return self.dsparkPlainPipelined(allocator);
    // A sampled block is as wide as the counts chose; a greedy one is cut by the drafter, sure of each draft or not.
    const ladder: ?*const spec_rounds.Ladder = if (self.dspark_pol) |*pol| (if (pol.rounds) |*c| &c.ladder else null) else null;
    const ask: plugin_driver.RoundAsk = .{
        .cap = if (plan.kind == .draft and !self.dspark_stochastic) @min(nativeBlock(self), budget) else plan.drafts,
        .stops = self.eos_token_ids,
        .copy = plan.copy,
        .round_ms = if (ladder) |l| l.drafts[0 .. l.block + 1] else &.{},
    };
    // Fused GPU-side draft tree + accept path for solo greedy & sampled requests (bypassed for load timing and keyed serial).
    const is_timing = if (self.dspark_pol) |*p| p.timing != null else false;
    const is_keyed = self.dspark_stochastic and env.on(.MLX_SERVE_KEYED_SAMPLING) and keyedSerial(self.sampling);
    if (!is_timing and !is_keyed and plugin_driver.nativeStepGroupAvail(self.xfm)) {
        var copy_slice = plan.copy;
        if (copy_slice.len == 0) {
            if ((try self.mtpLookupIndex(allocator)).matchBest(self.next_token_id, @min(spec_rounds.MAX_DRAFTS, budget))) |match| {
                if (match.suffix >= mtp_lookup.STRONG_SUFFIX) copy_slice = match.draft;
            }
        }
        var fused_ask: plugin_driver.RoundAsk = .{
            .cap = budget,
            .stops = self.eos_token_ids,
            .copy = copy_slice,
            .round_ms = if (ladder) |l| l.drafts[0 .. l.block + 1] else &.{},
            .top_k = if (self.dspark_stochastic) self.sampling.top_k else 0,
            .top_p = if (self.dspark_stochastic) self.sampling.top_p else 1.0,
            .min_p = if (self.dspark_stochastic) (self.sampling.min_p orelse 0.0) else 0.0,
        };
        if (self.dspark_stochastic) fused_ask.law = .{ .temperature = self.sampling.temperature, .seed = self.prng.random().int(u64) };
        var r_out: [1]abi.Round = undefined;
        const used_fused = blk: {
            plugin_driver.nativeStepGroup(self.xfm, &.{&self.ctx}, &.{self.next_token_id}, &.{fused_ask}, &r_out) catch |e| switch (e) {
                error.PluginStepGroupUnsupported => break :blk false,
                else => return e,
            };
            break :blk true;
        };
        if (used_fused) {
            const r = r_out[0];
            const tokens = try allocator.dupe(u32, r.tokens[0..r.n]);
            errdefer allocator.free(tokens);
            noteNative(self, plan, r.rows -| 1, r.accepted);
            return self.commitDsparkRound(allocator, .{ .tokens = tokens, .accepted = r.accepted, .next_token = r.next_token }, @intCast(r.position));
        }
    }
    if (self.dspark_stochastic) return nativeSampledRound(self, allocator, plan, ask);
    const r = try plugin_driver.nativeRound(self.xfm, &self.ctx, self.next_token_id, ask);
    const tokens = try allocator.dupe(u32, r.tokens[0..r.n]);
    errdefer allocator.free(tokens);
    noteNative(self, plan, r.rows -| 1, r.accepted);
    return self.commitDsparkRound(allocator, .{ .tokens = tokens, .accepted = r.accepted, .next_token = r.next_token }, @intCast(r.position));
}

/// One DSpark round on the DeepSeek-V4/V4.1 module itself (`modules.call`).
pub fn moduleRound(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    if (intree.dsv41(self.xfm)) |mdl| {
        try plugin_settle.activate(self.xfm, &self.ctx);
        // An exact greedy accept masks what the serial sampler masks.
        mdl.suppress = self.xfm.suppress_mask orelse .{ .ctx = null };
        // A draft vocabulary takes the request's prompt ids before its first round; a sampled request drafts every id.
        if (!self.dspark_prompt_vocab) {
            self.dspark_prompt_vocab = true;
            self.dspark_vocab_mode = true;
            try dsv41_mod.draftPromptVocab(mdl, &mdl.dec_state.?, self.prompt_ids_owned, !self.dspark_stochastic);
        }
        if (try self.dsparkCopyStep(allocator, mdl)) |res| return res;
        return self.dsparkPolicyStep(allocator, mdl);
    }
    return self.dsparkStep(allocator, dsv4_mod, intree.dsv4(self.xfm).?);
}

/// A DSpark round whose drafts are the context's own continuation after `t1` (an echo, an edit):
/// no stage forward, and up to `VERIFY_MAX_DRAFTS` drafts where the block holds `ds_block`.
/// Null when the context holds no copy worth verifying.
pub fn dsparkCopyStep(self: *Generator, allocator: std.mem.Allocator, mdl: *dsv41_mod.Dsv41Model) !?DrafterStepResult {
    const t1 = self.next_token_id;
    const idx = try self.mtpLookupIndex(allocator);
    const found = idx.matchBest(t1, @intCast(dsv41_mod.maxVerifyDrafts(mdl)));
    const n = dsparkCopyLen(found, self.max_tokens -| self.completion_tokens -| 1, dsv41_mod.copyBlock(mdl));
    if (n == 0) return null;
    const ids = try allocator.alloc(u32, n + 1);
    ids[0] = t1;
    @memcpy(ids[1..], found.?.draft[0..n]);
    var copy: dsv41_mod.DsparkDraft = .{ .ids = ids, .len = n, .logits = &.{}, .confidence = &.{} };
    defer copy.deinit(allocator);
    const st = &mdl.dec_state.?;
    const cap = capAcceptedForTokenBudget(std.math.maxInt(u32), self.completion_tokens, self.max_tokens);
    const round = if (self.dspark_stochastic)
        try self.dsparkStochasticRound(allocator, dsv41_mod, mdl, t1, cap, &copy, null, .context)
    else
        try dsv41_mod.dsparkRoundWith(mdl, allocator, st, t1, &copy, cap);
    self.noteCopy(n, round.accepted);
    return self.commitDsparkRound(allocator, round, st.n);
}

/// One DSpark round on module `mod` (deepseek_v4 or deepseek_v41, which
/// share the round API) and its bookkeeping on the Generator.
pub fn dsparkStep(self: *Generator, allocator: std.mem.Allocator, comptime mod: type, mdl: anytype) !?DrafterStepResult {
    const t1 = self.next_token_id;
    const accepted_cap = capAcceptedForTokenBudget(
        std.math.maxInt(u32),
        self.completion_tokens,
        self.max_tokens,
    );
    const round = if (self.dspark_stochastic)
        try self.dsparkStochasticRound(allocator, mod, mdl, t1, accepted_cap, null, null, .model)
    else
        try mod.dsparkRound(mdl, allocator, &mdl.dec_state.?, t1, accepted_cap);
    return self.commitDsparkRound(allocator, round, mdl.dec_state.?.n);
}

/// A round the drafting backoff skipped: one tick of the serial pipeline. Consecutive ticks time the plain step.
pub fn dsparkPlainPipelined(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    const tok = (try self.next(allocator)) orelse return null;
    self.dspark_plain += 1;
    const pol = &self.dspark_pol.?;
    // A served drafter's plain steps fade its counts; a module's time the plain step.
    if (pol.rounds) |*ctl| ctl.acc.idle();
    if (plugin_driver.nativeDrafter(self.xfm) == null) {
        if (pol.plain_watch) |*w| {
            const ms = @as(f32, @floatFromInt(w.read())) / @as(f32, std.time.ns_per_ms);
            pol.plain.sample(ms);
            if (pol.p2) |*p| p.plainDone(pol.plain.ms, ms);
        }
        pol.plain_watch = io_util.Stopwatch.init(self.timer.io);
    }
    return try mtpSerialOneToken(allocator, tok);
}

/// A test hook, never set by serving: N > 0 pins every DSpark round at N verify rows (t1 and N - 1 drafts), the
/// width sweep's arm. 0 leaves the adaptive policy.
pub var test_pin_rows: u32 = 0;

/// The DSpark policy's pinned verify rows: a streamed EXL3 bank's greedy request verifies the whole block (`whole`);
/// else `test_pin_rows`, which serving leaves at 0 (the adaptive policy).
pub fn dsparkPinRows(whole: bool) u32 {
    if (whole) return std.math.maxInt(u32);
    return test_pin_rows;
}

/// Drafts a pinned round verifies: `pin_rows` rows are t1 and `pin_rows - 1` drafts, at most the `len` drafted.
pub fn pinnedDrafts(len: usize, pin_rows: u32) usize {
    return @min(len, pin_rows - 1);
}

test "dspark pin rows: test_pin_rows = N verifies N rows a round, 0 is adaptive, an EXL3 bank the block" {
    defer test_pin_rows = 0;
    try testing.expectEqual(@as(u32, 0), dsparkPinRows(false));
    test_pin_rows = 3;
    try testing.expectEqual(std.math.maxInt(u32), dsparkPinRows(true));
    for (2..7) |n| {
        test_pin_rows = @intCast(n);
        const pin = dsparkPinRows(false);
        try testing.expectEqual(@as(u32, @intCast(n)), pin);
        // A 5-draft block: N rows are t1 and N - 1 drafts.
        try testing.expectEqual(n - 1, pinnedDrafts(5, pin));
    }
    try testing.expectEqual(@as(usize, 5), pinnedDrafts(5, 8));
    try testing.expectEqual(@as(usize, 5), pinnedDrafts(5, std.math.maxInt(u32)));
}

/// One DeepSeek-V4.1 stage round under `dspark_pol`: draft, verify the prefix the calibrated confidence pays
/// for (none = the trunk token alone), and teach the calibration and the drafting backoff what it bought.
pub fn dsparkPolicyStep(self: *Generator, allocator: std.mem.Allocator, mdl: *dsv41_mod.Dsv41Model) !?DrafterStepResult {
    const watch = io_util.Stopwatch.init(self.timer.io);
    const pol = &self.dspark_pol.?;
    const st = &mdl.dec_state.?;
    const t1 = self.next_token_id;
    // MLX_SERVE_DSV41_DRAFT_SAMPLE: a sampled round's drafts drawn from the draft's own distribution q.
    var q: mlx.mlx_array = .{ .ctx = null };
    defer if (q.ctx != null) {
        _ = mlx.mlx_array_free(q);
    };
    var d = if (self.dspark_stochastic and dsv41_mod.draftSampleServes(mdl)) blk: {
        const sd = try draftSampled(self, allocator, mdl, st, t1);
        q = sd.q;
        break :blk sd.d;
    } else try dsv41_mod.dsparkDraft(mdl, allocator, st, t1);
    defer d.deinit(allocator);
    const drafted = d.len;
    d.len = if (pol.pin_rows > 0)
        pinnedDrafts(d.len, pol.pin_rows)
    else if (pol.p2) |*p|
        p.verifyDrafts(d.confidence[0..d.len], pol.plain.ms)
    else
        dsparkVerifyDrafts(d.confidence[0..d.len], &pol.calib, pol.plain.ms);
    // An exact verify stays within the decode-width rows (`maxVerifyDrafts`; the block itself otherwise).
    d.len = @min(d.len, dsv41_mod.maxVerifyDrafts(mdl));
    const cap = capAcceptedForTokenBudget(std.math.maxInt(u32), self.completion_tokens, self.max_tokens);
    const round = if (self.dspark_stochastic)
        try self.dsparkStochasticRound(allocator, dsv41_mod, mdl, t1, cap, &d, if (q.ctx != null) q else null, .model)
    else
        try dsv41_mod.dsparkRoundWith(mdl, allocator, st, t1, &d, cap);
    self.dspark_stats.record(@intCast(d.len), round.accepted);
    // `MLX_SERVE_DSV41_SPEC_SYNC`: the round's own draft and verify (the next round's draft, built here, is its).
    const wall_ms = @max(0, @as(f32, @floatFromInt(watch.read())) / @as(f32, std.time.ns_per_ms) - dsv41_mod.takeRoundShiftMs(mdl));
    if (pol.p2) |*p| {
        p.learn(d.confidence[0..drafted], d.len, round.accepted, nextDecides(&d, drafted, round.accepted, round.next_token, self.dspark_stochastic));
        p.roundDone(pol.plain.ms, d.len + 1, round.tokens.len, wall_ms);
        return self.commitDsparkRound(allocator, round, st.n);
    }
    for (d.confidence[0..d.len], 0..) |c, k| {
        if (k > round.accepted) break;
        pol.calib.observe(sigmoid(c), k < round.accepted);
    }
    pol.drafting.record(@as(f32, @floatFromInt(round.tokens.len)) * pol.plain.ms, wall_ms);
    return self.commitDsparkRound(allocator, round, st.n);
}

/// Whether this round drafts at all, under the request's backoff.
pub fn policyDrafts(pol: *DflashPol) bool {
    return if (pol.p2) |*p| p.drafting.shouldDraft() else pol.drafting.shouldDraft();
}

/// A greedy round that kept every verified draft of `drafted` knows the first one it did not verify: the trunk's
/// `next_token` stands where it would. Null when there is none, or the round was sampled (its next token is a draw).
pub fn nextDecides(d: *const dsv41_mod.DsparkDraft, drafted: usize, accepted: usize, next_token: u32, stochastic: bool) ?bool {
    if (stochastic or accepted != d.len or d.len >= drafted or d.len + 1 >= d.ids.len) return null;
    return d.ids[d.len + 1] == next_token;
}

pub fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

/// Drafts a DSpark round verifies: the prefix whose calibrated acceptance (the confidence head's logits through
/// `calib`) buys the most tokens per ms of verify, on the verify ladder scaled to this request's plain step.
pub fn dsparkVerifyDrafts(conf_logits: []const f32, calib: *const dflash_policy.Calibration, plain_ms: f32) usize {
    var p: [dflash_policy.MAX_DRAFTS]f32 = undefined;
    const n = @min(conf_logits.len, p.len);
    for (conf_logits[0..n], p[0..n]) |c, *o| o.* = calib.probability(sigmoid(c));
    const costs = dflash_policy.priorCosts(plain_ms);
    return dflash_policy.chooseRows(p[0..n], &costs).rows - 1;
}

/// A draft round's bookkeeping; the module stands at `position`.
pub fn commitDsparkRound(self: *Generator, allocator: std.mem.Allocator, round_in: dsv4_mod.DsparkRound, position: usize) !?DrafterStepResult {
    var round = round_in;
    errdefer round.deinit(allocator);
    // The round advanced the module state — mirror it on the shell
    // cache verbatim so a later serial fallback (or the fresh-request
    // check keying on step==0) sees a consistent position. Generator.step
    // itself moves through advanceStep below (the clear-cadence clock).
    self.ctx.cache.step = position;
    self.dspark_attempted += 1;
    self.dspark_accepted_tokens += round.accepted;
    try self.generated_ids.appendSlice(allocator, round.tokens);
    self.advanceStep(@intCast(round.tokens.len));
    self.next_token_id = round.next_token;
    // tokens ownership transfers to the caller (scheduler frees).
    return DrafterStepResult{ .tokens = round.tokens, .accepted_tokens = round.accepted };
}

/// This request's sampling for one more draw: a seeded request keys each draw apart (`seedKey`).
pub fn nextDraw(self: *Generator) SamplingParams {
    const sp = self.sampling;
    self.sampling.draw +%= 1;
    return sp;
}

/// Where a sampled round's drafts came from: the model's stages (`model`), or the context (`context`, a copy round).
pub const DraftSource = enum { model, context };

/// One stochastic DSpark round: dsv4's own greedy stage draft (a one-hot
/// proposal) verified with the MTP acceptance machinery — filtered target
/// probs at EVERY verify position (`probsAllPositions`, the request's own
/// temperature/top-k/top-p), accept draft k with prob `min(1, p_k)`,
/// first reject at `a` corrected from `normalize(max(p_a − onehot, 0))`,
/// full accept sampled from the bonus row — corrections pre-sampled in
/// ONE batched graph (`mtpBatchedAcceptGraph`, one-hot arm) so the round
/// pays ONE bounded sync. The output distribution equals serial sampling
/// (the toy-vocab exactness test's invariant), and the correction always
/// derives from the ORIGINAL verify logits at the acceptance point — the
/// house partial-accept invariant in sampled form.
/// `copy`, when set, is the draft instead of the stages' (a one-hot proposal too): the policy step's own stage draft, or
/// the copy step's draft from the context. `source` says which, for the accept rule (`draftRule`): a draft the caller
/// passes is not by itself a copy.
pub fn dsparkStochasticRound(self: *Generator, allocator: std.mem.Allocator, comptime mod: type, mdl: anytype, t1: u32, accepted_cap: u32, copy: ?*const mod.DsparkDraft, q: ?mlx.mlx_array, source: DraftSource) !mod.DsparkRound {
    var pending = if (copy) |d| try mod.dsparkBeginWith(mdl, allocator, &mdl.dec_state.?, t1, d) else try mod.dsparkBegin(mdl, allocator, &mdl.dec_state.?, t1);
    defer pending.deinit();
    const verdict = try stochasticVerdict(self, pending.vl_g, pending.verify[0 .. pending.b + 1], mdl.vocab, accepted_cap, q, draftRule(self.dspark_typical, source == .context));
    pending.lapVerify(mdl);

    const round = try mod.dsparkFinish(mdl, allocator, &mdl.dec_state.?, &pending, verdict.accepted, verdict.next_token);
    mod.dsparkObserve(mdl, round.phases);
    return round;
}

/// A sampled round's outcome: the drafts kept and the token pending after them.
pub const Verdict = struct { accepted: u32, next_token: u32 };

/// A drawn block ends on its stop id (`ids` = `t1`, then the drafts): accepted, the stop is the token pending after
/// the drafts before it, so its row is never committed and the turn ends as often as the trunk ends it.
fn stopPending(v: Verdict, ids: []const u32, stops: []const u32) Verdict {
    for (ids[1 .. 1 + v.accepted], 0..) |id, k| {
        if (std.mem.indexOfScalar(u32, stops, id) != null) return .{ .accepted = @intCast(k), .next_token = id };
    }
    return v;
}

/// The sampled accept of a verify chunk: `verify` is `[t1, drafts...]` and `vl_g` its logits `[rows, vocab]`, `q` the
/// drafts' law `[drafts, vocab]` when they were drawn (a sampled draft's, a served drafter's) or null (each draft a
/// point mass): `verdictOf` over the request's filtered density under `rule`. With MLX_SERVE_KEYED_SAMPLING a keyed
/// request takes the serial verdict instead.
pub fn stochasticVerdict(self: *Generator, vl_g: mlx.mlx_array, verify: []const u32, vocab: usize, accepted_cap: u32, q: ?mlx.mlx_array, rule: Rule) !Verdict {
    if (env.on(.MLX_SERVE_KEYED_SAMPLING) and keyedSerial(self.sampling)) return serialVerdict(self, vl_g, verify, vocab, accepted_cap);
    // The rounds the typical rule decided (`[spec-stats] dspark typical`): the proof it engaged, not only its setting.
    if (rule == .typical) self.dspark_typical_rounds += 1;
    const s = self.xfm.s;
    // Filtered target probs over every verify row: [b+1, V] -> [1, b+1, V].
    const vshape = [_]c_int{ 1, @intCast(verify.len), @intCast(vocab) };
    var vl3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vl3);
    try mlx.check(mlx.mlx_reshape(&vl3, vl_g, &vshape, 3, s));
    const sp = self.nextDraw();
    const probs_all = try probsAllPositions(vl3, sp, s);
    defer _ = mlx.mlx_array_free(probs_all);
    return verdictOf(probs_all, verify, q, accepted_cap, sp, self.prng.random(), s, rule);
}

/// A sampled draft (`MLX_SERVE_DSV41_DRAFT_SAMPLE`): the block's uniforms from a fresh draw of the request's seed, its
/// tokens drawn from softmax(row / T) in the fused Markov chain, and their proposal rows `q`.
fn draftSampled(self: *Generator, allocator: std.mem.Allocator, mdl: *dsv41_mod.Dsv41Model, st: *dsv41_mod.State, t1: u32) !dsv41_mod.SampledDraft {
    const s = self.xfm.s;
    const sp = self.nextDraw();
    const key = seedKey(sp);
    defer _ = mlx.mlx_array_free(key);
    const lo = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(1);
    defer _ = mlx.mlx_array_free(hi);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, lo, hi, &[_]c_int{ @intCast(mdl.ds_block), @intCast(mdl.vocab) }, 2, .float32, key, s));
    return dsv41_mod.dsparkDraftSampled(mdl, allocator, st, t1, u, 1 / @max(sp.temperature, 1e-5));
}

/// The verdict of a verify over `verify` = `[t1, drafts...]` from the filtered target probs `probs_all` [1, rows, V]:
/// draft k kept with probability min(1, p_k / q_k), q the drafts' proposal rows ([drafts, V], drawn from them) or a
/// point mass on each draft (null: an argmax or copy draft); the token after the kept prefix is the residual sample
/// max(0, p - q) at the first reject, or the bonus row's draw on a full accept. Under `rule` `.typical` draft k is kept
/// iff p_k clears min(eps, delta * e^-H(p_k's row)), no uniform drawn, and the first reject is a draw from p itself.
pub fn verdictOf(probs_all: mlx.mlx_array, verify: []const u32, q: ?mlx.mlx_array, accepted_cap: u32, sp: SamplingParams, rnd: std.Random, s: mlx.mlx_stream, rule: Rule) !Verdict {
    if (verify.len > 1) {
        var p = try acceptStartRule(probs_all, sp, verify, q, if (q) |qa| mlx.getShape(qa)[1] else 0, rule, s);
        defer p.deinit();
        acceptKick(&p);
        return acceptReadWith(&p, accepted_cap, rnd);
    }
    // The confidence gate submitted nothing: this round verifies t1 alone and row 0 IS the bonus row: sample the next
    // trunk token from it directly.
    var log_p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_p);
    try mlx.check(mlx.mlx_log(&log_p, probs_all, s));
    const key = seedKey(sp);
    defer _ = mlx.mlx_array_free(key);
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, log_p, -1, key, s));
    var samp_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(samp_i);
    try mlx.check(mlx.mlx_astype(&samp_i, sampled, .int32, s));
    try mlx.check(mlx.mlx_array_eval(samp_i));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, samp_i));
    return .{ .accepted = 0, .next_token = @intCast(v) };
}

/// A sampled accept of a block (one draft or more) built but not read: the accept probabilities and the pre-sampled
/// corrections, with the verify graph beneath them, so several slots' accepts can be started before any is read.
pub const PendingAccept = struct {
    bg: MtpBatchedGraph,
    draft_arrs: [16]mlx.mlx_array = undefined,
    n_arrs: usize = 0,
    q_rows: [16]mlx.mlx_array = undefined,
    n_q: usize = 0,
    b: u32,
    rule: Rule = .exact,

    pub fn deinit(p: *PendingAccept) void {
        p.bg.deinit();
        for (p.draft_arrs[0..p.n_arrs]) |a| _ = mlx.mlx_array_free(a);
        for (p.q_rows[0..p.n_q]) |a| _ = mlx.mlx_array_free(a);
    }
};

/// `stochasticVerdict`'s accept graph for `verify` (`[t1, drafts...]`, two rows or more) over its logits `vl_g`, under
/// the exact rule.
pub fn acceptStart(self: *Generator, vl_g: mlx.mlx_array, verify: []const u32, vocab: usize, law: ?mlx.mlx_array) !PendingAccept {
    const s = self.xfm.s;
    // Filtered target probs over every verify row: [b+1, V] -> [1, b+1, V].
    const vshape = [_]c_int{ 1, @intCast(verify.len), @intCast(vocab) };
    var vl3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vl3);
    try mlx.check(mlx.mlx_reshape(&vl3, vl_g, &vshape, 3, s));
    const sp = self.nextDraw();
    const probs_all = try probsAllPositions(vl3, sp, s);
    defer _ = mlx.mlx_array_free(probs_all);
    return acceptStartProbs(self, probs_all, sp, verify, vocab, law);
}

/// `acceptStart` over the rows' filtered probs `probs_all` `[1, b+1, V]` (`probsAllPositions` under `sp`).
pub fn acceptStartProbs(self: *Generator, probs_all: mlx.mlx_array, sp: SamplingParams, verify: []const u32, vocab: usize, law: ?mlx.mlx_array) !PendingAccept {
    return acceptStartRule(probs_all, sp, verify, law, @intCast(vocab), .exact, self.xfm.s);
}

/// The accept graph of `verify` under `rule`, `law`'s rows read `width` wide: `.exact` keeps draft k with min(1, p/q)
/// (p for a point mass), `.typical` gives each draft's p and its row's floor min(eps, delta * e^-H) and corrections
/// drawn from p.
fn acceptStartRule(probs_all: mlx.mlx_array, sp: SamplingParams, verify: []const u32, law: ?mlx.mlx_array, width: c_int, rule: Rule, s: mlx.mlx_stream) !PendingAccept {
    const b: u32 = @intCast(verify.len - 1);
    if (law != null and b > 16) return error.SpecDraftLawTooWide;
    var p: PendingAccept = .{ .bg = undefined, .b = b, .rule = rule };
    errdefer {
        for (p.draft_arrs[0..p.n_arrs]) |a| _ = mlx.mlx_array_free(a);
        for (p.q_rows[0..p.n_q]) |a| _ = mlx.mlx_array_free(a);
    }
    // [1] int32 arrays of the draft ids, realized host values.
    const idshape = [_]c_int{1};
    for (0..b) |k| {
        const idv = [_]i32{@intCast(verify[k + 1])};
        p.draft_arrs[k] = mlx.mlx_array_new_data(&idv, &idshape, 1, .int32);
        p.n_arrs += 1;
    }
    if (law) |q| for (0..b) |k| {
        const ki: c_int = @intCast(k);
        p.q_rows[k] = mlx.mlx_array_new();
        p.n_q += 1;
        try mlx.check(mlx.mlx_slice(&p.q_rows[k], q, &[_]c_int{ ki, 0 }, 2, &[_]c_int{ ki + 1, width }, 2, &[_]c_int{ 1, 1 }, 2, s));
    };
    const q_in: ?[]const mlx.mlx_array = if (law != null) p.q_rows[0..b] else null;
    p.bg = switch (rule) {
        .exact => try mtpBatchedAcceptGraph(probs_all, p.draft_arrs[0..b], q_in, b, sp, s),
        // accept_p = p(draft), accept_q = the row's floor min(eps, delta * e^-H); the corrections draw from p.
        .typical => |t| try mtpBatchedTypicalGraph(probs_all, p.draft_arrs[0..b], q_in, b, mtp_acceptance.param(.{ .typical = t }), sp, s),
    };
    return p;
}

/// Starts `p`'s accept on the GPU: the accept vector and pre-sampled corrections, and the whole verify graph beneath them.
pub fn acceptKick(p: *const PendingAccept) void {
    const ev = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ev);
    _ = mlx.mlx_vector_array_append_value(ev, p.bg.accept_p);
    if (p.bg.accept_q.ctx != null) _ = mlx.mlx_vector_array_append_value(ev, p.bg.accept_q);
    _ = mlx.mlx_vector_array_append_value(ev, p.bg.corr_samples);
    _ = mlx.mlx_async_eval(ev);
}

/// Reads `p`: each draft kept with its probability under the request's draws, the token pending after the kept prefix
/// the correction at the first reject (or the bonus draw on a full accept), cut at the request's token budget.
pub fn acceptRead(self: *Generator, p: *PendingAccept, accepted_cap: u32) !Verdict {
    return acceptReadWith(p, accepted_cap, self.prng.random());
}

/// `acceptRead` with the uniforms from `rnd`; a typical rule draws none.
fn acceptReadWith(p: *PendingAccept, accepted_cap: u32, rnd: std.Random) !Verdict {
    const bg = &p.bg;
    try mlx.check(mlx.mlx_array_eval(bg.accept_p));
    const p_data = mlx.mlx_array_data_float32(bg.accept_p) orelse return error.MlxArrayDataNull;
    const q_data: ?[*]const f32 = if (bg.accept_q.ctx != null) blk: {
        try mlx.check(mlx.mlx_array_eval(bg.accept_q));
        break :blk mlx.mlx_array_data_float32(bg.accept_q) orelse return error.MlxArrayDataNull;
    } else null;
    var accepted: u32 = 0;
    switch (p.rule) {
        .exact => while (accepted < p.b) {
            const accept_prob = acceptProb(p_data[accepted], if (q_data) |q| q[accepted] else null);
            const u: f32 = rnd.float(f32);
            if (u >= accept_prob) break;
            accepted += 1;
        },
        .typical => {
            const floor = q_data orelse return error.MlxArrayDataNull;
            while (accepted < p.b and mtp_acceptance.typicalAccept(p_data[accepted], floor[accepted]) != typical_plant) accepted += 1;
        },
    }
    // Pick the correction at the request-budget boundary and let the model's state roll back to that same point.
    accepted = @min(accepted, accepted_cap);
    try mlx.check(mlx.mlx_array_eval(bg.corr_samples));
    const corr = mlx.mlx_array_data_int32(bg.corr_samples) orelse return error.MlxArrayDataNull;
    return .{ .accepted = accepted, .next_token = @intCast(corr[accepted]) };
}

/// A draft's acceptance: min(1, p/q) for one drawn from law q (0 where q gave it none), p for a point mass.
pub fn acceptProb(p: f32, q: ?f32) f32 {
    const law = q orelse return @min(1.0, p);
    return if (law > 0) @min(1.0, p / law) else 0;
}

/// Whether a sampled request's rounds accept serially: its draws are keyed (`sampling.keyed`, its seed set), as
/// DFlash's and MTP's `serial` arms read it.
pub fn keyedSerial(sampling: SamplingParams) bool {
    return sampling.keyed and sampling.temperature > 0.01 and sampling.seed != null;
}

/// MLX_SERVE_KEYED_SAMPLING's verdict: verify row i draws the keyed token at generated index G + 1 + i (t1 sits at G),
/// a draft stays while it equals the draw before it, and the draw after the kept prefix is the next token: what serial
/// decoding draws from the same rows' logits. The draw counter then stands where serial decoding's would.
fn serialVerdict(self: *Generator, vl_g: mlx.mlx_array, verify: []const u32, vocab: usize, accepted_cap: u32) !Verdict {
    const s = self.xfm.s;
    const rows = verify.len;
    const vshape = [_]c_int{ 1, @intCast(rows), @intCast(vocab) };
    var vl3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vl3);
    try mlx.check(mlx.mlx_reshape(&vl3, vl_g, &vshape, 3, s));
    var draws = try keyedDraws(self.sampling, self.generated_ids.items.len + 1, vl3, rows, s);
    defer draws.deinit();
    try mlx.check(mlx.mlx_array_eval(draws.lazy()));
    const v = serialAccept(verify[1..], try draws.ids(rows), accepted_cap);
    self.sampling.draw = self.generated_ids.items.len + v.accepted + 2;
    return v;
}

/// Row i of `logits` `[1, rows, V]` drawn as serial decoding draws generated index `base + i` (`sampleTokenLazy` on the
/// row alone): `verifySerialSamples` (spec/mtp/chain.zig) without a Generator; `[1, rows]` ids.
pub fn keyedDraws(sampling: SamplingParams, base: usize, logits: mlx.mlx_array, rows: usize, s: mlx.mlx_stream) !CommittedArgmax {
    const vocab = mlx.getShape(logits)[2];
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (0..rows) |i| {
        const ri: c_int = @intCast(i);
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, logits, &[_]c_int{ 0, ri, 0 }, 3, &[_]c_int{ 1, ri + 1, vocab }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var sp = sampling;
        sp.draw = base + i;
        const tok = sampleTokenLazy(row, sp, s);
        defer _ = mlx.mlx_array_free(tok);
        if (tok.ctx == null) return error.KeyedDrawFailed;
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, tok, &[_]c_int{1}, 1, s));
        _ = mlx.mlx_vector_array_append_value(vec, flat);
    }
    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
    var ids32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids32);
    try mlx.check(mlx.mlx_astype(&ids32, cat, .int32, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, ids32, &[_]c_int{ 1, @intCast(rows) }, 2, s));
    return .{ .arr = out };
}

/// The serial accept over realized draws: `drafts` d1..dk, `draws` x0..xk (row i's draw); d(i+1) stays while it equals
/// x(i), up to `cap` kept; the next token is the draw after the kept prefix.
pub fn serialAccept(drafts: []const u32, draws: []const i32, cap: u32) Verdict {
    var a: u32 = 0;
    while (a < drafts.len and a < cap and @as(i64, drafts[a]) == @as(i64, draws[a])) a += 1;
    return .{ .accepted = a, .next_token = @intCast(draws[a]) };
}

// ── tests ──

test "keyed DSpark: the serial accept keeps drafts while they equal the draws; the next token is the draw after them" {
    try testing.expectEqual(Verdict{ .accepted = 3, .next_token = 9 }, serialAccept(&.{ 4, 5, 6 }, &.{ 4, 5, 6, 9 }, 16));
    try testing.expectEqual(Verdict{ .accepted = 1, .next_token = 7 }, serialAccept(&.{ 4, 5, 6 }, &.{ 4, 7, 6, 9 }, 16));
    try testing.expectEqual(Verdict{ .accepted = 0, .next_token = 8 }, serialAccept(&.{ 4, 5 }, &.{ 8, 5, 1 }, 16));
    // the token budget: the draw at the cap is the target's own, a kept draft's equal
    try testing.expectEqual(Verdict{ .accepted = 2, .next_token = 6 }, serialAccept(&.{ 4, 5, 6 }, &.{ 4, 5, 6, 9 }, 2));
    try testing.expectEqual(Verdict{ .accepted = 0, .next_token = 4 }, serialAccept(&.{ 4, 5, 6 }, &.{ 4, 5, 6, 9 }, 0));
    // no drafts (the confidence gate's single-row round): the bonus draw
    try testing.expectEqual(Verdict{ .accepted = 0, .next_token = 3 }, serialAccept(&.{}, &.{3}, 16));
    // a keyed request accepts serially only when sampled and seeded
    try testing.expect(keyedSerial(.{ .temperature = 1.0, .seed = 7, .keyed = true }));
    try testing.expect(!keyedSerial(.{ .temperature = 1.0, .seed = 7, .keyed = false }));
    try testing.expect(!keyedSerial(.{ .temperature = 1.0, .seed = null, .keyed = true }));
    try testing.expect(!keyedSerial(.{ .temperature = 0.0, .seed = 7, .keyed = true }));
}

// Bit for bit over seeds, prompt positions, widths, filters and f32/bf16 rows: a verify's keyed draws are 1-row decodes'.
test "keyed DSpark (GPU): k-row verify draws equal k sequential 1-row draws over seeds and positions" {
    if (env.get(._GPU_WINDOW_LOCKED) == null) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const V: c_int = 129280;
    const seeds = [_]u64{ 0, 1, 20260829, 0x9E3779B97F4A7C15, 0xFFFF_FFFF_FFFF_FFFE, 31337 };
    const bases = [_]u64{ 0, 1, 16384, 16399, 1 << 20, 4_000_000_000 };
    const Cfg = struct { t: f32, p: f32, k: u32, min_p: ?f32 = null };
    const cfgs = [_]Cfg{ .{ .t = 1.0, .p = 0.95, .k = 20 }, .{ .t = 0.6, .p = 0.95, .k = 0 }, .{ .t = 1.3, .p = 1.0, .k = 0 }, .{ .t = 0.8, .p = 0.9, .k = 40, .min_p = 0.05 } };
    const widths = [_]usize{ 1, 2, 6, 8 };
    var flips: usize = 0;
    var case_i: u64 = 0;
    for (seeds) |seed| for (bases) |pb| for (cfgs) |cf| for (widths) |k| for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dt| {
        case_i += 1;
        const R: usize = k + 1;
        // the chunk's logits: normal rows, every other one peaked (one id raised by 8 / T), in dt
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, case_i));
        var base_l = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(base_l);
        try mlx.check(mlx.mlx_random_normal(&base_l, &[_]c_int{ 1, @intCast(R), V }, 3, .float32, 0.0, 2.5, key, s));
        const bump = try std.testing.allocator.alloc(f32, R * @as(usize, @intCast(V)));
        defer std.testing.allocator.free(bump);
        @memset(bump, 0);
        const Vu: usize = @intCast(V);
        for (0..R) |r| if (r % 2 == 1) {
            const at: usize = @intCast((case_i * 7919 + @as(u64, @intCast(r)) * 104729) % @as(u64, @intCast(V)));
            bump[r * Vu + at] = 8.0;
        };
        const bump_a = mlx.mlx_array_new_data(bump.ptr, &[_]c_int{ 1, @intCast(R), V }, 3, .float32);
        defer _ = mlx.mlx_array_free(bump_a);
        var l32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(l32);
        try mlx.check(mlx.mlx_add(&l32, base_l, bump_a, s));
        var chunk = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(chunk);
        try mlx.check(mlx.mlx_astype(&chunk, l32, dt, s));
        try mlx.check(mlx.mlx_array_eval(chunk));
        const G: usize = @intCast(case_i % 97);
        const sp: SamplingParams = .{ .temperature = cf.t, .top_p = cf.p, .top_k = cf.k, .min_p = cf.min_p, .seed = seed, .keyed = true, .position_base = pb };
        // the verify's draws, as `serialVerdict` takes them
        var draws = try keyedDraws(sp, G + 1, chunk, R, s);
        defer draws.deinit();
        try mlx.check(mlx.mlx_array_eval(draws.lazy()));
        const vd = try draws.ids(R);
        // each row on its own, as a 1-row decode's logits
        const host = try std.testing.allocator.alloc(u8, R * @as(usize, @intCast(V)) * mlx.mlx_array_itemsize(chunk));
        defer std.testing.allocator.free(host);
        {
            const p: [*]const u8 = switch (dt) {
                .float32 => @ptrCast(mlx.mlx_array_data_float32(chunk) orelse return error.MlxArrayDataNull),
                else => @ptrCast(mlx.mlx_array_data_bfloat16(chunk) orelse return error.MlxArrayDataNull),
            };
            @memcpy(host, p[0..host.len]);
        }
        const rb = @as(usize, @intCast(V)) * mlx.mlx_array_itemsize(chunk);
        for (0..R) |r| {
            const one = mlx.mlx_array_new_data(host[r * rb ..].ptr, &[_]c_int{ 1, 1, V }, 3, dt);
            defer _ = mlx.mlx_array_free(one);
            var sp1 = sp;
            sp1.draw = G + 1 + r;
            const tok = sampleTokenLazy(one, sp1, s);
            defer _ = mlx.mlx_array_free(tok);
            try testing.expect(tok.ctx != null);
            var t32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(t32);
            try mlx.check(mlx.mlx_astype(&t32, tok, .int32, s));
            try mlx.check(mlx.mlx_array_eval(t32));
            const x = (mlx.mlx_array_data_int32(t32) orelse return error.MlxArrayDataNull)[0];
            if (x != vd[r]) {
                flips += 1;
                std.debug.print("keyed DSpark: seed {d} base {d} T {d} top_p {d} top_k {d} width {d} {s} row {d}: verify {d} vs 1-row {d}\n", .{ seed, pb, cf.t, cf.p, cf.k, k, @tagName(dt), r, vd[r], x });
            }
        }
        // the accept over drafts equal to the draws up to a mismatch at j
        var drafts: [8]u32 = undefined;
        for (0..k + 1) |j| {
            for (0..k) |i| drafts[i] = @intCast(vd[i]);
            if (j < k) drafts[j] = @intCast((@as(u64, @intCast(vd[j])) + 1) % @as(u64, @intCast(V)));
            const v = serialAccept(drafts[0..k], vd, 16);
            try testing.expectEqual(@as(u32, @intCast(j)), v.accepted);
            try testing.expectEqual(@as(u32, @intCast(vd[j])), v.next_token);
        }
    };
    try testing.expectEqual(@as(usize, 0), flips);
}


test "native speculation: shared DSpark environment policy" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    try testing.expect(Generator.dsparkEnabled());
    for ([_][:0]const u8{ "", "1", "0", "0disabled" }) |value| {
        env.set(.MLX_SERVE_DSV4_DSPARK, value);
        try testing.expectEqual(value.len == 0 or value[0] != '0', Generator.dsparkEnabled());
    }
}

/// The DSV4_MINI pack at `path` served as the scheduler loads it: its plugin behind the driver's shell, `cfg` the
/// engine's parse (freed after the shell).
fn servedMini(io: std.Io, a: std.mem.Allocator, path: []const u8, cfg: *model_mod.ModelConfig) !Transformer {
    cfg.* = try model_mod.parseConfig(io, a, path);
    errdefer cfg.deinit(a);
    return plugin_driver.load(io, a, cfg, path, mlx.gpuStream());
}

test "dsv4: nextPld on a chokepoint-disabled generator stays serial (DSV4_MINI)" {
    // DSpark is opt-in at load; the nextDspark arm below needs it armed.
    env.set(.MLX_SERVE_DSV4_DSPARK, "1");
    // The live corruption path (log 166348-166361): the scheduler
    // decode tick dispatched on `slot.enable_pld` alone, so it called
    // `nextPld` on a generator whose init the dsv4 guard had already flipped
    // to pld_enabled=false — and nextPld trusted its caller, ran lookup +
    // verify forwards, and the rejected drafts left dsv4's module-owned
    // state (rings + kv/comp caches) permanently ahead of the rolled-back
    // KVCache shell. This test reproduces the bypassing caller directly:
    // nextPld on such a generator must (a) report the chokepoint flip via
    // `gen.pld_enabled == false`, (b) never run a verify forward
    // (`pld_attempted == 0`), and (c) emit the exact serial-decode tokens.
    //
    // Fabricate the mini with:
    //   python3 tests/dsv4_mlx_ref.py --fabricate /tmp/dsv4-mini
    //   DSV4_MINI=/tmp/dsv4-mini zig build test -Dtest-filter=DSV4_MINI
    const path_z = env.get(.DSV4_MINI) orelse return;
    if (mlx.noGpuBackend()) return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();


    // Never read by the Generator (stored only) — see the field comment.
    var tok_dummy: Tokenizer = undefined;

    // Prompt = every vocab id once. With key_len=1 any sampled t1 < V has an
    // earlier occurrence, so PLD's lookup ALWAYS proposes a draft — on the
    // pre-fix code that guarantees a verify forward (and the corruption);
    // random-content prompts can idle in the cold path and mask the bug
    // (exactly how the first two live requests read as "guards held").
    var prompt: [64]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(i);
    const greedy = SamplingParams{ .temperature = 0.0 };
    const want: usize = 10;

    // Serial baseline: the regular scheduler shape (skip_lazy_preforward).
    var serial: [want]u32 = undefined;
    {
        var xfm_cfg: model_mod.ModelConfig = undefined;
        var xfm = try servedMini(io, allocator, path, &xfm_cfg);
        defer xfm_cfg.deinit(allocator);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 16, greedy, &.{}, .{
            .skip_lazy_preforward = true,
        });
        defer gen.deinit(allocator);
        var n: usize = 0;
        while (n < want) {
            const t = (try gen.next(allocator)) orelse break;
            serial[n] = t;
            n += 1;
        }
        try testing.expectEqual(want, n);
    }

    // Bypass arm: init asks for PLD (the app's always-on flags), the dsv4
    // chokepoint flips it off, and the caller drives nextPld anyway — the
    // scheduler tick's exact live behavior before specTickMode grew the
    // generator-state conjunct.
    {
        var xfm_cfg: model_mod.ModelConfig = undefined;
        var xfm = try servedMini(io, allocator, path, &xfm_cfg);
        defer xfm_cfg.deinit(allocator);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 16, greedy, &.{}, .{
            .pld_enabled = true,
            .skip_lazy_preforward = true,
            .lookup_prompt = &prompt,
        });
        defer gen.deinit(allocator);
        try testing.expect(!gen.pld_enabled);

        var pld_toks: [want]u32 = undefined;
        var n: usize = 0;
        while (n < want) {
            const r = (try gen.nextPld(allocator, 4, 1)) orelse break;
            defer allocator.free(r.tokens);
            for (r.tokens) |t| {
                if (n < want) {
                    pld_toks[n] = t;
                    n += 1;
                }
            }
        }
        try testing.expectEqual(want, n);
        try testing.expectEqual(@as(u64, 0), gen.pld_attempted);
        try testing.expectEqualSlices(u32, serial[0..], pld_toks[0..]);
    }

    // DSpark arm: the SAME app-shaped init (spec flags on, greedy) now arms
    // dsv4's own draft mode on a stage-bearing checkpoint. The tick driver
    // is nextDspark; the sequence must match serial (the batch-verify vs
    // single-token kernel-choice class allows a late near-tie flip on the
    // random mini — the module-level dsparkRound gate pins the loop itself,
    // this pins the GENERATOR wiring: engagement, step accounting, and the
    // shell-cache mirror).
    {
        var xfm_cfg: model_mod.ModelConfig = undefined;
        var xfm = try servedMini(io, allocator, path, &xfm_cfg);
        defer xfm_cfg.deinit(allocator);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 16, greedy, &.{}, .{
            .pld_enabled = true,
            .mtp_enabled = true,
            .skip_lazy_preforward = true,
            .lookup_prompt = &prompt,
        });
        defer gen.deinit(allocator);
        try testing.expect(!gen.pld_enabled);
        try testing.expect(gen.mtp == null);
        try testing.expect(gen.dspark_enabled);

        var ds_toks: [want]u32 = undefined;
        var n: usize = 0;
        while (n < want) {
            const r = (try gen.nextDspark(allocator)) orelse break;
            defer allocator.free(r.tokens);
            for (r.tokens) |t| {
                if (n < want) {
                    ds_toks[n] = t;
                    n += 1;
                }
            }
        }
        try testing.expectEqual(want, n);
        // ENGAGEMENT: silent serial fallback is output-identical — count rounds.
        try testing.expect(gen.dspark_attempted >= 1);
        // Shell cache mirrors the module state exactly.
        try testing.expectEqual(intree.dsv4(&xfm).?.dec_state.?.n, gen.ctx.cache.step);
        // Sequence agreement: exact up to the sanctioned near-tie window.
        var first_div: usize = want;
        for (0..want) |k| {
            if (serial[k] != ds_toks[k]) {
                first_div = k;
                break;
            }
        }
        if (first_div < 4) {
            std.debug.print("nextDspark serial={any} dspark={any}\n", .{ serial, ds_toks });
            try testing.expect(false);
        }
        std.debug.print("dsv4 nextDspark (generator): first_div={d}/{d}, rounds={d}, accepts={d}\n", .{ first_div, want, gen.dspark_attempted, gen.dspark_accepted_tokens });
    }
}

test "nextDraw: each stochastic DSpark round takes a fresh draw of the request's seed" {
    var gen: Generator = undefined;
    gen.sampling = .{ .temperature = 0.6, .seed = 7, .draw = 5 };
    const a = gen.nextDraw();
    const b = gen.nextDraw();
    try testing.expectEqual(@as(u64, 5), a.draw);
    try testing.expectEqual(@as(u64, 6), b.draw);
    try testing.expectEqual(@as(?u64, 7), b.seed);
    try testing.expectEqual(@as(u64, 7), gen.sampling.draw);
}

test "dsparkArmFor: greedy and stochastic arms gate on clean sampling" {
    // Greedy clean → the argmax-equality arm (temp 0 or top_k 1).
    try testing.expectEqual(Generator.DsparkArm.greedy, Generator.dsparkArmFor(.{ .temperature = 0.0 }, 0));
    try testing.expectEqual(Generator.DsparkArm.greedy, Generator.dsparkArmFor(.{ .temperature = 0.6, .top_k = 1 }, 0));
    // Sampled clean → stochastic (the checkpoint-default temp 0.6 agent
    // shape: pi and friends omit temperature, generation_config fills 0.6).
    try testing.expectEqual(Generator.DsparkArm.stochastic, Generator.dsparkArmFor(.{ .temperature = 0.6, .top_p = 0.95 }, 0));
    try testing.expectEqual(Generator.DsparkArm.stochastic, Generator.dsparkArmFor(.{ .temperature = 1.0, .top_k = 40 }, 0));
    // Penalties / grammar / logprobs stay serial on BOTH arms — the
    // pre-stochastic contract, unchanged.
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.0, .repeat_penalty = 1.1 }, 0));
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.6, .presence_penalty = 0.5 }, 0));
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.6 }, 5));
    var c: Constraint = undefined;
    try testing.expectEqual(Generator.DsparkArm.off, Generator.dsparkArmFor(.{ .temperature = 0.6, .constraint = &c }, 0));
}

test "dsv4: stochastic dspark engages at sampled temperature and keeps the exit invariant (DSV4_MINI)" {
    // The motivating traffic shape: the checkpoint ships generation_config
    // temp 0.6 and agent CLIs omit temperature, so every real agent request
    // ran serial while only pinned `--temp 0` earned DSpark. The stochastic
    // arm ports the MTP probsAllPositions acceptance (one-hot Leviathan over
    // filtered target probs) onto dsv4's own draft stages. Engagement is
    // COUNTED (dspark_attempted) — a silent serial fallback emits perfectly
    // plausible tokens.
    env.set(.MLX_SERVE_DSV4_DSPARK, "1");
    const path_z = env.get(.DSV4_MINI) orelse return;
    if (mlx.noGpuBackend()) return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();


    // Never read by the Generator (stored only) — see the field comment.
    var tok_dummy: Tokenizer = undefined;

    var prompt: [64]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(i);
    // The agent-default request shape, seeded so the run is reproducible.
    const sampled = SamplingParams{ .temperature = 0.6, .top_p = 0.95, .seed = 0xD54A };

    var xfm_cfg: model_mod.ModelConfig = undefined;
    var xfm = try servedMini(io, allocator, path, &xfm_cfg);
    defer xfm_cfg.deinit(allocator);
    defer xfm.deinit();
    var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 64, sampled, &.{}, .{
        .pld_enabled = true,
        .mtp_enabled = true,
        .skip_lazy_preforward = true,
        .lookup_prompt = &prompt,
    });
    defer gen.deinit(allocator);

    try testing.expect(gen.dspark_enabled);
    try testing.expect(gen.dspark_stochastic);

    const mdl = intree.dsv4(&xfm).?;
    const want: usize = 10;
    var n: usize = 0;
    while (n < want) {
        const t1 = gen.next_token_id;
        const r = (try gen.nextDspark(allocator)) orelse break;
        defer allocator.free(r.tokens);
        try testing.expect(r.tokens.len >= 1);
        try testing.expectEqual(t1, r.tokens[0]);
        try testing.expect(r.accepted_tokens <= mdl.ds_block);
        for (r.tokens) |tok| {
            try testing.expect(tok < mdl.vocab);
            n += 1;
        }
        // Module state and the shell-cache mirror stay in lockstep with
        // exactly what was committed (the v2 spec exit invariant).
        try testing.expectEqual(prompt.len + n, mdl.dec_state.?.n);
        try testing.expectEqual(mdl.dec_state.?.n, gen.ctx.cache.step);
    }
    try testing.expect(n >= want);
    // ENGAGEMENT: count rounds, never output shape.
    try testing.expect(gen.dspark_attempted >= 1);
    // Exit invariant survives a hand-off: a following serial step decodes a
    // valid token (finite logits) from the state the rounds left behind.
    const t = (try gen.next(allocator)) orelse return error.TestExpectedResult;
    try testing.expect(t < mdl.vocab);
    std.debug.print("dsv4 stochastic dspark (generator): {d} tokens over {d} rounds, {d} drafts accepted\n", .{ n, gen.dspark_attempted, gen.dspark_accepted_tokens });

    // b==0 arm: an over-threshold confidence gate submits NOTHING — the round
    // verifies t1 alone, commits it, and samples the next trunk token from
    // row 0's filtered probs (the stochastic sibling of the greedy
    // confidence-gate test), on a fresh Transformer whose gate is set past any confidence.
    var xfm2_cfg: model_mod.ModelConfig = undefined;
    var xfm2 = try servedMini(io, allocator, path, &xfm2_cfg);
    defer xfm2_cfg.deinit(allocator);
    defer xfm2.deinit();
    intree.dsv4(&xfm2).?.ds_conf_thr = 999999;
    var gen2 = try Generator.initWithOptions(io, allocator, &xfm2, &tok_dummy, &prompt, 64, sampled, &.{}, .{
        .pld_enabled = true,
        .mtp_enabled = true,
        .skip_lazy_preforward = true,
        .lookup_prompt = &prompt,
    });
    defer gen2.deinit(allocator);
    try testing.expect(gen2.dspark_enabled);
    var n2: usize = 0;
    while (n2 < 4) {
        const r = (try gen2.nextDspark(allocator)) orelse break;
        defer allocator.free(r.tokens);
        try testing.expectEqual(@as(u32, 0), r.accepted_tokens);
        try testing.expectEqual(@as(usize, 1), r.tokens.len);
        try testing.expect(r.tokens[0] < mdl.vocab);
        n2 += 1;
    }
    try testing.expectEqual(@as(usize, 4), n2);
    try testing.expect(gen2.dspark_attempted >= 4);
    try testing.expectEqual(intree.dsv4(&xfm2).?.dec_state.?.n, gen2.ctx.cache.step);
    std.debug.print("dsv4 stochastic dspark (b==0 gate): {d} single-token rounds\n", .{n2});
}

test "dsparkVerifyDrafts: a sure head verifies the block, an unsure one t1 alone, a doubt mid-block cuts there" {
    const G = Generator;
    var calib = dflash_policy.Calibration{};
    try testing.expectEqual(@as(usize, 5), G.dsparkVerifyDrafts(&.{ 9, 9, 9, 9, 9 }, &calib, 17.6));
    try testing.expectEqual(@as(usize, 0), G.dsparkVerifyDrafts(&.{ 0, 0, 0, 0, 0 }, &calib, 17.6));
    try testing.expectEqual(@as(usize, 2), G.dsparkVerifyDrafts(&.{ 9, 9, -4, 9, 9 }, &calib, 17.6));
    try testing.expectEqual(@as(usize, 0), G.dsparkVerifyDrafts(&.{}, &calib, 17.6));
    // A head that keeps claiming certainty the trunk rejects stops buying rows.
    for (0..40) |_| calib.observe(G.sigmoid(9), false);
    try testing.expectEqual(@as(usize, 0), G.dsparkVerifyDrafts(&.{ 9, 9, 9, 9, 9 }, &calib, 17.6));
}

test "dspark nextDecides: a greedy round that kept every verified draft learns the first unverified one from its next token" {
    var ids = [_]u32{ 10, 11, 12, 13, 14, 15 };
    var conf = [_]f32{ 1, 1, 1, 1, 1 };
    var d: dsv41_mod.DsparkDraft = .{ .ids = &ids, .len = 2, .logits = &.{}, .confidence = &conf };
    // Drafts 0 and 1 (ids 11, 12) kept: the trunk's next token is draft 2 (id 13), a hit; any other id, a miss.
    try testing.expectEqual(@as(?bool, true), nextDecides(&d, 5, 2, 13, false));
    try testing.expectEqual(@as(?bool, false), nextDecides(&d, 5, 2, 99, false));
    // A reject among the verified drafts, a sampled round, or no draft past the verified ones: nothing to learn.
    try testing.expectEqual(@as(?bool, null), nextDecides(&d, 5, 1, 12, false));
    try testing.expectEqual(@as(?bool, null), nextDecides(&d, 5, 2, 13, true));
    try testing.expectEqual(@as(?bool, null), nextDecides(&d, 2, 2, 13, false));
    // A round that verified t1 alone learns draft 0 (id 11).
    d.len = 0;
    try testing.expectEqual(@as(?bool, true), nextDecides(&d, 5, 0, 11, false));
}

test "dspark drafting backoff: prose rounds that lose to the plain step are skipped, copy-like rounds are not" {
    // A round that commits few tokens for its cost loses to the plain step (costs in ms).
    var prose = dflash_policy.Drafting{};
    try testing.expect(prose.shouldDraft());
    prose.record(3.2 * 17.6, 66);
    try testing.expect(!prose.shouldDraft());
    var echo = dflash_policy.Drafting{};
    try testing.expect(echo.shouldDraft());
    echo.record(5.5 * 17.6, 70);
    try testing.expect(echo.shouldDraft());
}

/// Pearson's chi-square of `counts` against `n` draws from `p`.
fn chiSquare(counts: []const u64, p: []const f64, n: usize) f64 {
    var x: f64 = 0;
    for (counts, p) |c, pi| {
        const e = pi * @as(f64, @floatFromInt(n));
        const d = @as(f64, @floatFromInt(c)) - e;
        x += d * d / e;
    }
    return x;
}

test "dspark sampled verdict (GPU): drafts drawn from q, kept at min(1, p / q) and resampled from max(0, p - q), emit p over fixed seeds; a flat proposal in the ratio for the same draws fails" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const V = 6;
    // The target's filtered rows at the draft's position and the bonus row, and the draft's proposal.
    const p = [_]f64{ 0.05, 0.10, 0.15, 0.20, 0.20, 0.30 };
    const q = [_]f64{ 0.30, 0.25, 0.20, 0.10, 0.10, 0.05 };
    var pv: [2 * V]f32 = undefined;
    for (0..V) |i| {
        pv[i] = @floatCast(p[i]);
        pv[V + i] = @floatCast(p[V - 1 - i]);
    }
    var qv: [V]f32 = undefined;
    for (&qv, q) |*o, v| o.* = @floatCast(v);
    const probs = mlx.mlx_array_new_data(&pv, &[_]c_int{ 1, 2, V }, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const qa = mlx.mlx_array_new_data(&qv, &[_]c_int{ 1, V }, 2, .float32);
    defer _ = mlx.mlx_array_free(qa);
    // planted: the ratio and the residual read a flat proposal, not the q the drafts came from
    const flat: [V]f32 = @splat(1.0 / @as(f32, V));
    const qf = mlx.mlx_array_new_data(&flat, &[_]c_int{ 1, V }, 2, .float32);
    defer _ = mlx.mlx_array_free(qf);
    const N = 4000;
    var got: [2][V]u64 = .{ @splat(0), @splat(0) };
    for (0..2) |arm| {
        var prng = std.Random.DefaultPrng.init(0xD5A2);
        const rnd = prng.random();
        for (0..N) |trial| {
            var u = rnd.float(f64);
            var x: u32 = V - 1;
            for (q, 0..) |qi, i| {
                if (u < qi) {
                    x = @intCast(i);
                    break;
                }
                u -= qi;
            }
            const v = try verdictOf(probs, &.{ 0, x }, if (arm == 0) qa else qf, 1, .{ .temperature = 1, .seed = 1234 + trial }, rnd, s, .exact);
            got[arm][if (v.accepted == 1) x else v.next_token] += 1;
        }
    }
    const exact = chiSquare(&got[0], &p, N);
    const planted = chiSquare(&got[1], &p, N);
    std.debug.print("[dspark sample] chi-square of the emitted token against p over {d} seeded rounds (5 dof): q in the ratio {d:.1}, a flat proposal {d:.1}\n", .{ N, exact, planted });
    // 5 degrees of freedom: 30 is past the 1e-5 tail; the flat proposal's bias is about 345 at this count.
    try testing.expect(exact < 30);
    try testing.expect(planted > 150);
}

test "dspark sampled verdict (GPU): a draft vocabulary's q (0 past it, renormalized over it) in the ratio and the residual emits p; the whole row's q for the same draws fails" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const V = 6;
    const p = [_]f64{ 0.05, 0.10, 0.15, 0.20, 0.20, 0.30 };
    // The draft's whole-row softmax, and its vocabulary {0, 2, 3, 5}: the drafts come from the row over those ids alone
    // (`dsv41.draftChain` with `MLX_SERVE_DSV41_DRAFT_VOCAB`), the renormalized restriction, 0 at ids 1 and 4.
    const whole = [_]f64{ 0.30, 0.25, 0.20, 0.10, 0.10, 0.05 };
    const in_vocab = [_]bool{ true, false, true, true, false, true };
    var z: f64 = 0;
    for (whole, in_vocab) |w, in| z += if (in) w else 0;
    var q: [V]f64 = undefined;
    for (&q, whole, in_vocab) |*o, w, in| o.* = if (in) w / z else 0;
    var pv: [2 * V]f32 = undefined;
    for (0..V) |i| {
        pv[i] = @floatCast(p[i]);
        pv[V + i] = @floatCast(p[V - 1 - i]);
    }
    var qv: [V]f32 = undefined;
    for (&qv, q) |*o, v| o.* = @floatCast(v);
    var wv: [V]f32 = undefined;
    for (&wv, whole) |*o, v| o.* = @floatCast(v);
    const probs = mlx.mlx_array_new_data(&pv, &[_]c_int{ 1, 2, V }, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const qa = mlx.mlx_array_new_data(&qv, &[_]c_int{ 1, V }, 2, .float32);
    defer _ = mlx.mlx_array_free(qa);
    // planted: the ratio and the residual read the whole row's softmax, not the restricted q the drafts came from
    const wa = mlx.mlx_array_new_data(&wv, &[_]c_int{ 1, V }, 2, .float32);
    defer _ = mlx.mlx_array_free(wa);
    const N = 4000;
    var got: [2][V]u64 = .{ @splat(0), @splat(0) };
    for (0..2) |arm| {
        var prng = std.Random.DefaultPrng.init(0xD5A3);
        const rnd = prng.random();
        for (0..N) |trial| {
            var u = rnd.float(f64);
            var x: u32 = V - 1;
            for (q, 0..) |qi, i| {
                if (u < qi) {
                    x = @intCast(i);
                    break;
                }
                u -= qi;
            }
            try testing.expect(in_vocab[x]);
            const v = try verdictOf(probs, &.{ 0, x }, if (arm == 0) qa else wa, 1, .{ .temperature = 1, .seed = 4321 + trial }, rnd, s, .exact);
            got[arm][if (v.accepted == 1) x else v.next_token] += 1;
        }
    }
    const exact = chiSquare(&got[0], &p, N);
    const planted = chiSquare(&got[1], &p, N);
    std.debug.print("[dspark sample] draft vocabulary: chi-square of the emitted token against p over {d} seeded rounds (5 dof): the restricted q {d:.1}, the whole row's {d:.1}\n", .{ N, exact, planted });
    // The whole row's q never resamples id 1 (p 0.10 < q 0.25): its expected chi-square is about 900 at this count.
    try testing.expect(exact < 30);
    try testing.expect(planted > 150);
}

test "draftRule: typical only where the request asked for it and the drafts are the model's; a copied draft stays exact" {
    const t: mtp_acceptance.Typical = .{ .delta = 0.3, .eps = 0.09 };
    try testing.expectEqual(Rule.exact, draftRule(null, false));
    try testing.expectEqual(Rule.exact, draftRule(null, true));
    try testing.expectEqual(Rule{ .typical = t }, draftRule(t, false));
    try testing.expectEqual(Rule.exact, draftRule(t, true));
}

test "dspark typical verdict (GPU): p(draft) against min(eps, delta * e^-H) at eps 1 decides with no uniform, the reject draws from p; a floor of 1 keeps nothing (serial sampling), a floor near 0 keeps any draft; an inverted floor fails" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const V = 4;
    const p = [V]f64{ 0.7, 0.2, 0.05, 0.05 };
    // The draft's row is p; the bonus row (never reached: one draft, cap 1, reads it only on a full accept) is p too.
    var pv: [2 * V]f32 = undefined;
    for (0..V) |i| {
        pv[i] = @floatCast(p[i]);
        pv[V + i] = @floatCast(p[i]);
    }
    const probs = mlx.mlx_array_new_data(&pv, &[_]c_int{ 1, 2, V }, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    // The CPU oracle's floor at delta 0.2: H(p) = 0.871, e^-H = 0.418, floor 0.0837.
    const floor = mtp_acceptance.typicalThreshold(&p, 0.2, 1.0);
    try testing.expect(floor > 0.05 and floor < 0.2);
    const N = 2000;
    // kept[case]: rounds that kept the draft; emitted: the token each rejecting round drew.
    const Case = struct { x: u32, delta: f32 };
    const cases = [_]Case{ .{ .x = 0, .delta = 0.2 }, .{ .x = 2, .delta = 0.2 }, .{ .x = 0, .delta = 1e30 }, .{ .x = 2, .delta = 1e-9 } };
    var kept: [cases.len]u64 = @splat(0);
    var drawn: [cases.len][V]u64 = @splat(@as([V]u64, @splat(0)));
    var prng = std.Random.DefaultPrng.init(0x7E51);
    for (cases, 0..) |c, ci| for (0..N) |trial| {
        const v = try verdictOf(probs, &.{ 0, c.x }, null, 1, .{ .temperature = 1, .seed = 1234 + trial }, prng.random(), s, .{ .typical = .{ .delta = c.delta, .eps = 1.0 } });
        if (v.accepted == 1) kept[ci] += 1 else drawn[ci][v.next_token] += 1;
    };
    std.debug.print("[dspark typical] kept of {d}: p 0.7 at delta 0.2 {d}, p 0.05 at 0.2 {d}, p 0.7 at a floor of 1 {d}, p 0.05 at a floor near 0 {d}\n", .{ N, kept[0], kept[1], kept[2], kept[3] });
    // Deterministic: the draft over the floor is always kept, the one under it never (exact keeps them 70% and 5%).
    try testing.expectEqual(@as(u64, N), kept[0]);
    try testing.expectEqual(@as(u64, 0), kept[1]);
    // A floor of 1 keeps nothing: every round draws from p, which is serial sampling's distribution.
    try testing.expectEqual(@as(u64, 0), kept[2]);
    try testing.expect(chiSquare(&drawn[2], &p, N) < 30);
    // The reject's draw is p itself (not max(0, p - onehot)): the rejected draft can come back.
    try testing.expect(chiSquare(&drawn[1], &p, N) < 30);
    // A floor near 0 keeps any draft with mass: the lossy end.
    try testing.expectEqual(@as(u64, N), kept[3]);
    // Planted: the floor's comparison inverted keeps the draft under the floor and drops the one over it.
    typical_plant = true;
    defer typical_plant = false;
    var planted: [2]u64 = .{ 0, 0 };
    for (0..2) |ci| for (0..N) |trial| {
        const v = try verdictOf(probs, &.{ 0, cases[ci].x }, null, 1, .{ .temperature = 1, .seed = 1234 + trial }, prng.random(), s, .{ .typical = .{ .delta = 0.2, .eps = 1.0 } });
        planted[ci] += v.accepted;
    };
    try testing.expect(planted[0] != N or planted[1] != 0);
}

test "an accepted stop id is the pending token, and the drafts before it are what the round keeps" {
    const ids = [_]u32{ 11, 5, 9, 2 };
    const whole: Verdict = .{ .accepted = 3, .next_token = 40 };
    try testing.expectEqual(whole, stopPending(whole, &ids, &.{}));
    try testing.expectEqual(whole, stopPending(whole, &ids, &.{ 11, 40 }));
    try testing.expectEqual(Verdict{ .accepted = 2, .next_token = 2 }, stopPending(whole, &ids, &.{2}));
    try testing.expectEqual(Verdict{ .accepted = 0, .next_token = 5 }, stopPending(whole, &ids, &.{ 5, 2 }));
    // A stop the accept refused was never kept.
    const cut: Verdict = .{ .accepted = 1, .next_token = 40 };
    try testing.expectEqual(cut, stopPending(cut, &ids, &.{9}));
}

test "a drawn draft is accepted with min(1, p/q), a point mass with p" {
    try testing.expectEqual(@as(f32, 0.3), acceptProb(0.3, null));
    try testing.expectEqual(@as(f32, 1), acceptProb(1.4, null));
    try testing.expectApproxEqAbs(@as(f32, 0.5), acceptProb(0.2, 0.4), 1e-7);
    try testing.expectEqual(@as(f32, 1), acceptProb(0.4, 0.2));
    try testing.expectEqual(@as(f32, 0), acceptProb(0.4, 0));
}

// Bar: a greedy slot keeps the drafts that are the trunk's picks, and the pick after them is its next token.
test "dspark: a greedy verdict keeps the agreeing prefix and the pick after it" {
    const ids = [_]u32{ 10, 5, 6, 7 };
    try testing.expectEqual(Verdict{ .accepted = 2, .next_token = 9 }, greedyVerdict(&ids, &.{ 5, 6, 9, 1 }));
    try testing.expectEqual(Verdict{ .accepted = 3, .next_token = 3 }, greedyVerdict(&ids, &.{ 5, 6, 7, 3 }));
    try testing.expectEqual(Verdict{ .accepted = 0, .next_token = 4 }, greedyVerdict(&ids, &.{ 4, 6, 7, 3 }));
    try testing.expectEqual(Verdict{ .accepted = 0, .next_token = 8 }, greedyVerdict(ids[0..1], &.{8}));
}
