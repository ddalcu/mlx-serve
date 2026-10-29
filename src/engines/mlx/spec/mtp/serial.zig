//! MTP serial-mode switching: pre-draft open, head detach, serial ticks and clocks.
const std = @import("std");
const env = @import("../../../../env.zig");
const mlx = @import("../../../../mlx/ffi.zig");
const log = @import("../../../../sdk/util/log.zig");
const io_util = @import("../../../../sdk/util/io_util.zig");
const DrafterStepResult = @import("../drafter.zig").DrafterStepResult;
const Generator = @import("../../generate/generator.zig").Generator;
const MTP_ADAPTIVE_PROBE_WARM = @import("adaptive.zig").MTP_ADAPTIVE_PROBE_WARM;
const MTP_ADAPTIVE_MIN_KV = @import("adaptive.zig").MTP_ADAPTIVE_MIN_KV;
const MTP_GROUP_ROWS_MAX = @import("round.zig").MTP_GROUP_ROWS_MAX;
const MtpPreDraft = @import("chain.zig").MtpPreDraft;
const SubLap = @import("trace.zig").SubLap;
const isEosId = @import("../../../../sdk/request/stops.zig").isEosId;
const mtpAdaptiveKvEligible = @import("adaptive.zig").mtpAdaptiveKvEligible;
const mtpChainBuildBatched = @import("chain_build.zig").mtpChainBuildBatched;
const mtpChainDispatch = @import("chain_build.zig").mtpChainDispatch;
const mtpChainDispatchBatched = @import("chain_build.zig").mtpChainDispatchBatched;
const mtpPredraftEnabled = @import("trace.zig").mtpPredraftEnabled;
const mtpTraceEnabled = @import("trace.zig").mtpTraceEnabled;


/// Cross-round pre-draft (round pipelining): at the round's tail — the
/// accept decision made, trunk committed/rolled back, EV updated,
/// last_hidden/next_token_id already pointing at the next round — build
/// the NEXT round's chunk-A draft chain (consuming the history stash
/// Phase 5a just created, exactly as the next round's Phase 1 would)
/// and fire it at the GPU. The CPU then returns to the scheduler for
/// emit/SSE work while the head chain runs; the next nextMtp call finds
/// its drafts already materialized. The plan is computed AFTER this
/// round's EV update, so it is byte-identical to the one the next
/// round's entry would compute.
pub fn mtpMaybePreDraft(self: *Generator, allocator: std.mem.Allocator) !void {
    if (self.mtp_batch_head) return;
    var chain = (try self.mtpPreDraftOpen(allocator, true)) orelse return;
    errdefer chain.deinit(allocator);
    if (chain.lookup) {
        self.mtp_pre_draft = chain;
        return;
    }
    try self.mtpChainBuild(&chain, 0, chain.plan.m_lo);
    try mtpChainDispatch(&chain, 0, chain.plan.m_lo);
    self.mtp_pre_draft = chain;
}

/// Whether the eager pre-draft may build the next round's head chain. A
/// chain is only worth its cost when a later round reads it: a request
/// whose budget is spent or whose pending token ends it keeps no reader.
pub fn mtpPreDraftOpenAllows(completion_tokens: u32, max_tokens: u32, pending_is_eos: bool) bool {
    return completion_tokens < max_tokens and !pending_is_eos;
}

/// The next round's empty chain, or null when this request must not pre-draft.
/// `allow_lookup`: false for group pre-drafts, which build every chain batched.
pub fn mtpPreDraftOpen(self: *Generator, allocator: std.mem.Allocator, allow_lookup: bool) !?MtpPreDraft {
    if (!mtpPreDraftOpenAllows(self.completion_tokens, self.max_tokens, isEosId(self.next_token_id, self.eos_token_ids))) return null;
    if (!mtpPredraftEnabled() or self.spec_disabled_runtime or self.mtp_planner_pending) return null;
    std.debug.assert(self.mtp_pre_draft == null);
    const plan = self.mtpRoundPlan();
    // The plan itself can end speculation or park the request on a serial probe.
    if (self.spec_disabled_runtime or self.mtp_serial_left > 0) return null;
    if (allow_lookup) if (try self.mtpLookupChain(allocator, plan, self.next_token_id)) |chain| return chain;
    return try self.mtpChainInit(allocator, plan, self.next_token_id);
}

/// `mtpMaybePreDraft` for a group that just finished its round: ONE batched chain and
/// one dispatch, so the head runs while the scheduler publishes the round.
pub fn mtpGroupPreDraft(gens: []const *Generator, allocator: std.mem.Allocator) !void {
    var rows: [MTP_GROUP_ROWS_MAX]*Generator = undefined;
    var chains: [MTP_GROUP_ROWS_MAX]MtpPreDraft = undefined;
    var n: usize = 0;
    errdefer for (chains[0..n]) |*c| c.deinit(allocator);
    const lap = SubLap.start(mtpTraceEnabled(), gens[0].timer.io);
    var depth: u32 = 0;
    for (gens) |g| {
        chains[n] = (try g.mtpPreDraftOpen(allocator, false)) orelse continue;
        rows[n] = g;
        depth = @max(depth, chains[n].m);
        n += 1;
    }
    if (n == 0) return;
    try mtpChainBuildBatched(rows[0..n], chains[0..n], 0, depth);
    if (lap.read()) |ns| for (gens) |g| g.mtp_trace.addSub(.chain, ns);
    try mtpChainDispatchBatched(chains[0..n]);
    for (rows[0..n], chains[0..n]) |g, c| g.mtp_pre_draft = c;
    n = 0;
    if (lap.read()) |ns| for (gens) |g| {
        g.mtp_trace.add(.predraft, ns);
        g.mtp_gap_watch = io_util.Stopwatch.init(g.timer.io);
    };
}

// Serial blocks inside nextMtp. Leaving MTP is cheap; coming back needs `t1 NOT in cache`
// and an `h_prev` for that position: `drainPipelineForSpec` lands the first, the capture
// tick the second. The head's committed history does not grow across a serial block
// (a content gap that costs acceptance for a while); M-RoPE turns never come back.

/// Will this serial block be left behind? Only then is applying the deferred history stash worth a head forward.
pub fn mtpSerialMayResume(self: *const Generator) bool {
    if (self.mtp_serial_left > 0 or self.mtp_serial_exit != .none) return true;
    return self.spec_disable_reason == .adaptive and self.mtpAdaptiveHeadMayResume();
}

/// Leave the head cleanly for a serial block; `apply_stash` applies the deferred history
/// stash so the head history is complete up to the block. Idempotent.
pub fn mtpDetachHead(self: *Generator, allocator: std.mem.Allocator, apply_stash: bool) !void {
    if (self.mtp_pre_draft) |*pd| {
        // A placed chain left dead history rows past the committed end: never keep them.
        if (self.mtp_planner_owned or pd.place != null) try self.mtp_cache.?.truncate(self.mtpCommittedHistoryLen(), self.xfm.s);
        pd.deinit(allocator);
        self.mtp_pre_draft = null;
    }
    if (apply_stash) try self.mtpApplyStash();
}

/// Append the deferred history stash to the head cache now, as a
/// history-only head forward (lazy; the next head forward realizes it).
pub fn mtpApplyStash(self: *Generator) !void {
    var st = self.mtp_hist_stash orelse return;
    self.mtp_hist_stash = null;
    defer st.deinit();
    const mc = &self.mtp_cache.?;
    try mc.truncate(st.off0, self.xfm.s);
    // A committed-history append: the mixer and the vocab projection are dead work.
    const out = try self.mtp.?.forward(
        self.xfm,
        mc,
        st.ids,
        st.hidden,
        @intCast(st.off0),
        .none,
        self.mtpMropeContext(),
        st.host_ids[0..st.n],
    );
    if (out.logits.ctx != null) _ = mlx.mlx_array_free(out.logits);
    if (out.hidden_next.ctx != null) _ = mlx.mlx_array_free(out.hidden_next);
    if (out.rerank_x.ctx != null) _ = mlx.mlx_array_free(out.rerank_x);
}

/// A forced commit (`commitForcedTokens`) ran plain through the trunk, so the head's history
/// has a hole. Re-seed a qwen4 head (its positions refuse a gap) and take the serial exit
/// ramp back to MTP. False = not an MTP slot that may resume: the caller disables spec.
pub fn mtpResumeAfterForced(self: *Generator, allocator: std.mem.Allocator) !bool {
    if (self.mtp == null or self.mtp_cache == null or !self.mtpAdaptiveHeadMayResume()) return false;
    try self.mtpDetachHead(allocator, false);
    switch (self.mtp_cache.?) {
        .qwen4 => |r| {
            if (self.mtp_hist_stash) |*st| st.deinit();
            self.mtp_hist_stash = null;
            r.t.qwen4MtpActivate(r.st);
            try r.t.qwen4MtpReset();
        },
        else => {},
    }
    self.spec_disabled_runtime = false;
    self.spec_disable_reason = .none;
    self.mtp_serial_exit = .drain;
    return true;
}

pub fn mtpSerialOneToken(allocator: std.mem.Allocator, tok: u32) !DrafterStepResult {
    const tokens = try allocator.alloc(u32, 1);
    tokens[0] = tok;
    return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
}

/// Inter-tick wall clock for the table's serial cell (twin of `mtpRegimeWallMs`). Null on the first read of a block.
pub fn mtpSerialWallMs(self: *Generator) ?f32 {
    if (self.mtp_serial_clock) |*c| {
        const ns = c.read();
        c.reset();
        return @as(f32, @floatFromInt(ns)) / @as(f32, std.time.ns_per_ms);
    }
    self.mtp_serial_clock = io_util.Stopwatch.init(self.timer.io);
    return null;
}

pub fn mtpAdaptiveModelOk(self: *const Generator) bool {
    return self.model_has_mtp;
}

/// Will anyone read a serial cell for this model? Gated on the model, not the request: a
/// per-request `enable_mtp:false` on a calibrated checkpoint is the cleanest serial sample.
pub fn serialCellWanted(self: *const Generator) bool {
    if (!self.mtpAdaptiveModelOk()) return false;
    // Not `xfm.qwen4_mtp != null`: the weights load with the trunk even under `--no-mtp`.
    if (!self.model_has_mtp) return false;
    return mtpAdaptiveKvEligible(self.mtpKvLen(), MTP_ADAPTIVE_MIN_KV);
}

/// Drop the serial cell's pending interval: something ran between two decode ticks that
/// was not decode (a prefill chunk under `interleaveDecodeTick`).
pub fn invalidateSerialClock(self: *Generator) void {
    self.mtp_serial_clock = null;
    self.mtp_serial_warm = 0;
}

/// Same for the spec round's clock: the interval since the previous round end held a
/// prefill chunk. The next round times itself and is dropped as a transition.
pub fn invalidateRoundClock(self: *Generator) void {
    self.mtp_regime_clock = null;
    self.spec_round_prev_width = null;
}

/// Fold one plain serial decode token into the model's serial cell. The first
/// `MTP_ADAPTIVE_PROBE_WARM` ticks of a block are transitions; contention restarts the clock.
pub fn observeSerialTick(self: *Generator) void {
    if (!self.serialCellWanted()) {
        self.mtp_serial_clock = null;
        self.mtp_serial_warm = 0;
        return;
    }
    if (!self.spec_cost_solo) {
        self.mtp_serial_clock = null;
        self.mtp_serial_warm = 0;
        return;
    }
    const ms = self.mtpSerialWallMs() orelse return;
    const warm = self.mtp_serial_warm < MTP_ADAPTIVE_PROBE_WARM;
    if (warm) self.mtp_serial_warm += 1;
    _ = self.xfm.round_cost.observeSerial(self.mtpKvLen(), ms, true, warm);
}

/// May `mtpSerialCaptureTick` run? It forwards the one pending token with capture, so
/// pipelined logits or a pipelined token would publish a hidden for an uncommitted row.
/// A runtime check: `std.debug.assert` compiles to nothing in ReleaseFast.
pub fn mtpSerialCaptureReady(has_pending_logits: bool, has_pending_token: bool) bool {
    return !has_pending_logits and !has_pending_token;
}

/// Abandon the ramp back to MTP and finish this request on the serial arm. Logged once per request.
pub fn mtpSerialGiveUp(self: *Generator, allocator: std.mem.Allocator, why: []const u8) !?DrafterStepResult {
    if (!self.mtp_serial_giveup_logged) {
        self.mtp_serial_giveup_logged = true;
        log.info("  [mtp] adaptive: {s} at the serial exit -> serial for the rest of the request\n", .{why});
    }
    self.mtp_serial_exit = .none;
    self.mtp_serial_left = 0;
    self.spec_disabled_runtime = true;
    self.spec_disable_reason = .adaptive;
    const tok = try self.next(allocator) orelse return null;
    return try mtpSerialOneToken(allocator, tok);
}

/// One tick of a serial block. `.none` is the real serial arm; `.drain` and `.capture` are the exit ramp.
pub fn mtpSerialTick(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    if (self.mtp_serial_exit == .capture) return try self.mtpSerialCaptureTick(allocator);
    if (self.mtp_serial_exit == .drain) {
        switch (try self.drainPipelineForSpec(allocator)) {
            .stopped => return null,
            .stay_disabled => return try self.mtpSerialGiveUp(allocator, "unexpected pipeline state"),
            .already_clean => {
                self.mtp_serial_exit = .capture;
                return try self.mtpSerialCaptureTick(allocator);
            },
            .drained => |tok| {
                self.mtp_serial_exit = .capture;
                return try mtpSerialOneToken(allocator, tok);
            },
        }
    }
    const tok = try self.next(allocator) orelse return null;
    self.observeSerialTick();
    if (self.mtp_serial_left > 0) {
        self.mtp_serial_left -= 1;
        if (self.mtp_serial_left == 0) self.mtp_serial_exit = .drain;
    }
    return try mtpSerialOneToken(allocator, tok);
}

/// The one synchronous step that ends a serial block: forward the pending token with
/// capture and sample its successor without forwarding it (`nextMtp`'s entry invariant).
pub fn mtpSerialCaptureTick(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    if (!mtpSerialCaptureReady(self.has_pending_logits, self.has_pending_token)) {
        return try self.mtpSerialGiveUp(allocator, "pipeline not drained");
    }
    if (try self.checkStop()) return null;
    const token = self.next_token_id;
    const tok_i32: i32 = @intCast(token);
    const tok_shape = [_]c_int{ 1, 1 };
    const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(tok_input);

    var hidden = mlx.mlx_array_new();
    const logits = self.xfm.forwardWithCapture(&self.ctx, tok_input, &hidden) catch |e| {
        _ = mlx.mlx_array_free(hidden);
        return e;
    };
    defer _ = mlx.mlx_array_free(logits);
    if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
    self.last_hidden = hidden;
    self.has_last_hidden = true;
    self.advanceStep(1);
    try self.generated_ids.append(allocator, token);

    // Both checks below return on a Metal abort instead of ending the process, so the
    // handle needs an owner on the error path; a scoped `defer`, not errdefer + manual free.
    const val: i32 = blk: {
        const lazy = self.sampleLazy(logits);
        defer _ = mlx.mlx_array_free(lazy);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var v: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
        break :blk v;
    };
    self.next_token_id = @intCast(val);

    self.mtp_serial_exit = .none;
    self.mtp_serial_clock = null;
    self.mtp_serial_warm = 0;
    self.mtp_regime_clock = null;
    return try mtpSerialOneToken(allocator, token);
}
