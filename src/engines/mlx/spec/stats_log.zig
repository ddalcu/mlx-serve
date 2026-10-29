//! The [spec-stats] and QSA arm log lines.
const std = @import("std");
const mlx_stream = @import("../modules/stream.zig");
const log = @import("../../../sdk/util/log.zig");
const plugin_settle = @import("../plugin/settle.zig");
const round_cost = @import("round_cost.zig");
const Generator = @import("../generate/generator.zig").Generator;
const StepTrace = @import("../generate/step.zig").StepTrace;
const mtpDepthPolicy = @import("mtp/ev_plan.zig").mtpDepthPolicy;
const mtpDepthPolicyFor = @import("mtp/ev_plan.zig").mtpDepthPolicyFor;


/// The `[spec-stats] mode=` word of an mlx-stream lane: the first word of the plugin's lane name.
fn laneMode(lane: []const u8) []const u8 {
    return lane[0 .. std.mem.indexOfScalar(u8, lane, ' ') orelse lane.len];
}

/// Emit a stable, easy-to-grep one-line summary of spec-decode acceptance
/// for this request. External tooling parses the `[spec-stats]` prefix;
/// keep the format stable.
///
/// No-op when this Generator never ran a speculative path. Drafter and
/// PLD are mutually exclusive within a single request (drafter > PLD per
/// dispatch), so the branching here is unambiguous.
///
/// Field semantics:
/// - `attempts` = number of speculative rounds (one verify forward each).
/// - `accepts` = total drafted tokens accepted across all rounds (excludes
///   the always-committed t1 token at the start of each round).
/// - `avg_per_round` = accepts/attempts. Bounded by `(block_size - 1)` for
///   drafter and `pld_draft_len` for PLD. Equals the metric the runtime
///   gate compares against `RUNTIME_GATE_MIN_RATE`.
/// - `per_draft_pct` (drafter only) = accepts / (attempts × (block_size-1)),
///   the per-draft acceptance probability comparable to vLLM's reported
///   "62% acceptance rate" metric.
/// Which QSA arm served this request's attention (qwen4_exp only; no-op
/// elsewhere). The engaged/declined meters are one-shots for the whole
/// PROCESS, so they cannot say what ran on THIS request at THIS context
/// length — a gather that quietly hands every call back to the dense mask
/// (a full kv8 dequant per layer per forward) otherwise reads like a
/// healthy run. Emitted next to `[spec-stats]`; resets the tally.
pub fn logQsaArms(self: *Generator) void {
    // The tally rides THIS request's ForwardCtx — the one every forward
    // of this slot was handed. A Transformer-level counter printed the
    // union of every slot that decoded or prefilled since the last
    // finisher reset it.
    plugin_settle.logMeters(self.xfm, &self.ctx);
    const c = self.ctx.qsa_arms.take() orelse return;
    const arm = c.majority() orelse return;
    log.info(
        "  [qsa-arms] qsa={s} calls={d} mask={d} decode={d} verify={d} prefill={d}\n",
        .{ @tagName(arm), c.total(), c.mask, c.decode, c.verify, c.prefill },
    );
}

pub fn logSpecStats(self: *const Generator) void {
    StepTrace.report();
    var table_buf: [256]u8 = undefined;
    var acc_buf: [128]u8 = undefined;
    var rows_buf: [160]u8 = undefined;
    var lookup_buf: [256]u8 = undefined;
    const table_bucket = self.xfm.round_cost.bucketOf(self.mtpKvLen());
    if (self.dspark_enabled and self.dspark_attempted > 0) {
        const avg_per_round: f64 = @as(f64, @floatFromInt(self.dspark_accepted_tokens)) /
            @as(f64, @floatFromInt(self.dspark_attempted));
        if (self.xfm.mlx_stream_ext) |m| {
            // The plugin's lane names the mode (DeepSeek-V4.1: dspark, MiMo: mtp).
            const lane = mlx_stream.laneName(m);
            log.info(
                "  [spec-stats] mode={s} attempts={d} accepts={d} avg_per_round={d:.2} plugin=mlx-stream lane=\"{s}\"\n",
                .{ laneMode(lane), self.dspark_attempted, self.dspark_accepted_tokens, avg_per_round, lane },
            );
            var lane_buf: [512]u8 = undefined;
            if (mlx_stream.laneReceipt(m, &lane_buf)) |line| if (line.len > 0) log.info("  [spec-stats] {s}\n", .{line});
            return;
        }
        log.info(
            "  [spec-stats] mode=dspark attempts={d} accepts={d} avg_per_round={d:.2} stage_rounds={d} copy_rounds={d} copy_drafted={d} copy_accepted={d}\n",
            .{ self.dspark_attempted, self.dspark_accepted_tokens, avg_per_round, self.dspark_attempted - self.dspark_copies, self.dspark_copies, self.dspark_copy_drafted, self.dspark_copy_accepted },
        );
        if (self.dspark_pol) |pol| log.info(
            "  [spec-stats] dspark policy: plain_ticks={d} skipped={d} probes={d} plain_ms={d:.2} acc_pos={s} rows_hist={s}\n",
            .{ self.dspark_plain, pol.drafting.skipped, pol.drafting.probes, pol.plain.ms, self.dspark_stats.formatAcc(&acc_buf), self.dspark_stats.formatRows(&rows_buf) },
        );
        return;
    }
    if (self.mtp != null and (self.mtp_attempted > 0 or self.mtp_lookup_rounds > 0)) {
        const avg_per_round: f64 = @as(f64, @floatFromInt(self.mtp_accepted_tokens)) /
            @as(f64, @floatFromInt(@max(self.mtp_attempted, 1)));
        // Depth varies per round under the EV controller — the honest
        // denominator is the DRAFTED count, not attempts x cap.
        const drafts_proposed: u64 = if (self.mtp_drafted_tokens > 0)
            self.mtp_drafted_tokens
        else
            self.mtp_attempted * @as(u64, self.mtp_depth);
        const per_draft_pct: f64 = if (drafts_proposed > 0)
            100.0 * @as(f64, @floatFromInt(self.mtp_accepted_tokens)) /
                @as(f64, @floatFromInt(drafts_proposed))
        else
            0.0;
        log.info(
            "  [spec-stats] mode=mtp attempts={d} accepts={d} avg_per_round={d:.2} per_draft_pct={d:.1}% depth={d} drafted={d} ext_rounds={d} partial_rounds={d} runtime_disabled={s} reason={s} adaptive={s} serial_cell={d:.2} sync_ms={d:.2} round_ms={d:.2} two_ms_tok={d:.2} one_ms_tok={d:.2} verdict_round={d} trials={d} width_trials={d} table={s}:{s} table_drops=t{d}/c{d}/b{d}/i{d} serial_drops=t{d}/c{d}/b{d} depth_policy={s} lookup={d}/{d}/{d}\n",
            .{
                self.mtp_attempted,
                self.mtp_accepted_tokens,
                avg_per_round,
                per_draft_pct,
                self.mtp_depth,
                self.mtp_drafted_tokens,
                self.mtp_ext_rounds,
                self.partial_rounds,
                if (self.spec_disabled_runtime) "true" else "false",
                @tagName(self.spec_disable_reason),
                @tagName(self.mtp_adaptive.arm),
                self.xfm.round_cost.serialMsPerTok(table_bucket) orelse 0.0,
                self.mtp_ev_sync_ms,
                self.mtp_ev_round_ms,
                if (self.mtp_regime.two_tok > 0) self.mtp_regime.two_ms / self.mtp_regime.two_tok else 0.0,
                if (self.mtp_regime.one_tok > 0) self.mtp_regime.one_ms / self.mtp_regime.one_tok else 0.0,
                self.mtp_regime.verdict_round,
                self.mtp_regime.trials,
                self.mtp_width_trial.trials,
                round_cost.bucketName(self.xfm.round_cost.layout, table_bucket),
                self.xfm.round_cost.formatBucket(table_bucket, &table_buf),
                self.xfm.round_cost.dropped_transition,
                self.xfm.round_cost.dropped_contended,
                self.xfm.round_cost.dropped_bad,
                self.xfm.round_cost.dropped_implausible,
                self.xfm.round_cost.serial_dropped_transition,
                self.xfm.round_cost.serial_dropped_contended,
                self.xfm.round_cost.serial_dropped_bad,
                @tagName(mtpDepthPolicyFor(mtpDepthPolicy(), self.mtpKvLen())),
                self.mtp_lookup_rounds,
                self.mtp_lookup_drafted,
                self.mtp_lookup_accepted,
            },
        );
        log.info("  [spec-stats] lazy={d}/{d}\n", .{ self.mtp_lazy_built, self.mtp_lazy_kept });
        if (self.mtp_lookup_rounds > 0) log.info("  [spec-stats] lookup_table={s}:{s}\n", .{
            round_cost.bucketName(self.xfm.round_cost.layout, table_bucket),
            self.xfm.round_cost.formatLookupBucket(table_bucket, &lookup_buf),
        });
        return;
    }
    if (self.dflash != null and self.dflash_attempted + self.dflash_plain > 0) {
        const avg_per_round: f64 = @as(f64, @floatFromInt(self.dflash_accepted_tokens)) /
            @as(f64, @floatFromInt(@max(self.dflash_attempted, 1)));
        const drafts_per_round: u32 = if (self.dflash_block_size >= 1) self.dflash_block_size - 1 else 0;
        const drafts_proposed: u64 = self.dflash_attempted * @as(u64, drafts_per_round);
        const per_draft_pct: f64 = if (drafts_proposed > 0)
            100.0 * @as(f64, @floatFromInt(self.dflash_accepted_tokens)) /
                @as(f64, @floatFromInt(drafts_proposed))
        else
            0.0;
        log.info(
            "  [spec-stats] mode=dflash attempts={d} accepts={d} avg_per_round={d:.2} gate_min={d:.2} per_draft_pct={d:.1}% block_size={d} partial_rounds={d} runtime_disabled={s} table={s}:{s} table_drops=t{d}/c{d}/b{d}/i{d} plain={d} skipped={d} probes={d} copies={d}/{d}/{d} acc_pos={s} rows_hist={s}\n",
            .{
                self.dflash_attempted,
                self.dflash_accepted_tokens,
                avg_per_round,
                self.dflash_min_accepted_per_round,
                per_draft_pct,
                self.dflash_block_size,
                self.partial_rounds,
                if (self.spec_disabled_runtime) "true" else "false",
                round_cost.bucketName(self.xfm.round_cost.layout, table_bucket),
                self.xfm.round_cost.formatBucket(table_bucket, &table_buf),
                self.xfm.round_cost.dropped_transition,
                self.xfm.round_cost.dropped_contended,
                self.xfm.round_cost.dropped_bad,
                self.xfm.round_cost.dropped_implausible,
                self.dflash_plain,
                if (self.dflash_pol) |pol| pol.drafting.skipped else 0,
                if (self.dflash_pol) |pol| pol.drafting.probes else 0,
                self.dflash_copies,
                self.dflash_copy_drafted,
                self.dflash_copy_accepted,
                self.dflash_stats.formatAcc(&acc_buf),
                self.dflash_stats.formatRows(&rows_buf),
            },
        );
        return;
    }
    if (self.drafter != null and self.drafter_attempted > 0) {
        const avg_per_round: f64 = @as(f64, @floatFromInt(self.drafter_accepted_tokens)) /
            @as(f64, @floatFromInt(self.drafter_attempted));
        const drafts_per_round: u32 = if (self.drafter_block_size >= 1) self.drafter_block_size - 1 else 0;
        const drafts_proposed: u64 = self.drafter_attempted * @as(u64, drafts_per_round);
        const per_draft_pct: f64 = if (drafts_proposed > 0)
            100.0 * @as(f64, @floatFromInt(self.drafter_accepted_tokens)) /
                @as(f64, @floatFromInt(drafts_proposed))
        else
            0.0;
        log.info(
            "  [spec-stats] mode=drafter attempts={d} accepts={d} avg_per_round={d:.2} per_draft_pct={d:.1}% block_size={d} runtime_disabled={s}\n",
            .{
                self.drafter_attempted,
                self.drafter_accepted_tokens,
                avg_per_round,
                per_draft_pct,
                self.drafter_block_size,
                if (self.spec_disabled_runtime) "true" else "false",
            },
        );
    } else if (self.pld_attempted > 0) {
        const avg_per_round: f64 = @as(f64, @floatFromInt(self.pld_accepted_tokens)) /
            @as(f64, @floatFromInt(self.pld_attempted));
        log.info(
            "  [spec-stats] mode=pld attempts={d} accepts={d} avg_per_round={d:.2} runtime_disabled={s}\n",
            .{
                self.pld_attempted,
                self.pld_accepted_tokens,
                avg_per_round,
                if (self.spec_disabled_runtime) "true" else "false",
            },
        );
    }
}

// ── tests ──


test "laneMode: the spec-stats mode is the first word of the lane name" {
    try std.testing.expectEqualStrings("dspark", laneMode("dspark typical 0.3"));
    try std.testing.expectEqualStrings("mtp", laneMode("mtp typical 0.5 depth 2"));
    try std.testing.expectEqualStrings("serial", laneMode("serial"));
}
