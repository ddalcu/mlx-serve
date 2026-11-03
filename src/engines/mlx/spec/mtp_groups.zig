//! Batched MTP groups: planner, sub-groups and the merged verify.

const std = @import("std");
const env = @import("../../../env.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const generate_mod = @import("../generate.zig");
const model_registry_mod = @import("../../../host/registry/registry.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const ForwardCtx = transformer_mod.ForwardCtx;
const Generator = generate_mod.Generator;
const round_cost_mod = @import("round_cost.zig");
const group_cost_mod = @import("mtp_group_cost.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const testing = std.testing;

const runBatchedDecodeTick = @import("../batching/tick.zig").runBatchedDecodeTick;
const MAX_BATCH_GROUP = @import("../batching/verdict.zig").MAX_BATCH_GROUP;
const noteSerial = @import("../batching/verdict.zig").noteSerial;
const batchedKvKeepCount = @import("../batching/verdict.zig").batchedKvKeepCount;
const groupKeepCount = @import("../batching/verdict.zig").groupKeepCount;
const groupCtxMax = @import("../batching/verdict.zig").groupCtxMax;
const groupAttendsPerSlot = @import("../batching/verdict.zig").groupAttendsPerSlot;
const fillGroupPadWasteKvLens = @import("../batching/verdict.zig").fillGroupPadWasteKvLens;
const runDecodeTick = @import("../decode.zig").runDecodeTick;
const runSingleDecodeTick = @import("../decode.zig").runSingleDecodeTick;
const finishSlot = @import("../finish.zig").finishSlot;
const Scheduler = @import("../scheduler.zig").Scheduler;
const Slot = @import("../slot.zig").Slot;
const specTickMode = @import("wiring.zig").specTickMode;
const publishSpeculativeBlock = @import("wiring.zig").publishSpeculativeBlock;
const loopGuardTick = @import("../loop_stop.zig").loopGuardTick;

/// Per-slot follow-up to a batched forward: `record` moves the id ledger, `publish` samples and streams.
const BatchedTickAction = struct { record: bool, publish: bool };

pub fn batchedTickAction(cancelled: bool) BatchedTickAction {
    // The KV row is already written, so the ledger records it even when cancelled.
    return .{ .record = true, .publish = !cancelled };
}

/// Batched decode kernel for >=2 active slots. All slots must have already
/// done a non-spec prefill (`skip_lazy_preforward = true`) so cache.step is
/// at prompt_len with `next_token_id` carrying t1. We forward those N tokens
/// in one kernel pass, sample per-slot, push the OLD next_token_id (= the
/// token we just committed to cache via the forward), and load the new
/// sampled id back into next_token_id.
/// Batched sibling: a batched group shares one forward, so a failure belongs to every slot in it.
/// Can this MTP slot's verify ride one batched trunk forward with its neighbours?
/// Per-request head (never qwen4's module-owned one), a GDN trunk with per-slot state
/// the batched path merges, and a round that is actually speculating this tick.
pub fn slotMtpGroupable(slot: *const Slot) bool {
    const gen = if (slot.legacy_gen) |*g| g else return false;
    if (!slot.enable_mtp or gen.mtp == null or gen.mtp_cache == null or !gen.has_last_hidden) return false;
    if (gen.mtp.?.moduleOwned()) return false;
    if (gen.spec_disabled_runtime or gen.mtp_serial_left > 0 or gen.mtp_serial_exit != .none) return false;
    if (slot.sampling.constraint != null or slot.logprobs_n > 0) return false;
    const t = slot.model.transformer orelse return false;
    if (!t.supportsBatchedGdnDecode()) return false;
    if (gen.ctx.ssm_entries == null and t.hasRecurrentLayers()) return false;
    return specTickMode(slot.enable_mtp, true, slot.enable_drafter, gen.drafter != null, gen.dflash != null, slot.enable_pld, gen.pld_enabled, gen.dspark_enabled) == .mtp;
}

pub const Planner = @import("mtp_group_planner.zig");
const qmv_rows = @import("../../../sdk/blocks/linear/qmv_rows.zig");

pub fn plannerShape(slots: []const *Slot, widths: []const u8) group_cost_mod.GroupShape {
    const xfm = slots[0].model.transformer.?;
    var shape = group_cost_mod.GroupShape{ .n = @intCast(slots.len), .kv_format = cacheCostFormat(slots[0].cache.config) };
    for (slots, widths, 0..) |slot, width, i| {
        const gen = &slot.legacy_gen.?;
        const kv: u8 = @intCast(xfm.round_cost.bucketOf(@intCast(slot.moe_seq_offset)));
        if (width == 0) {
            shape.rows[i] = group_cost_mod.GroupShape.row(0, 0, 0, kv, 0, 0, group_cost_mod.GroupShape.samplingMode(false, false, false, gen.sampling.temperature > 0.01));
        } else {
            const off = Generator.mtpRoundOff0(gen.mtp_hist_stash, gen.mtp_cache.?.step());
            const history = 1 + if (gen.mtp_hist_stash) |stash| stash.n else @as(usize, 0);
            shape.head_format = cacheCostFormat(xfm.qwen4Trunk().?.qwen4_mtp.?.cache.config);
            shape.rows[i] = group_cost_mod.GroupShape.row(width, width, width, kv, @intCast(xfm.round_cost.bucketOf(@intCast(off))), @intCast(@min(history, round_cost_mod.MAX_WIDTH + 2)), gen.mtpPlannerMode());
        }
    }
    return shape;
}

fn plannerInputSame(a: group_cost_mod.GroupShape, b: group_cost_mod.GroupShape) bool {
    var aa = a;
    var bb = b;
    aa.attention = 0;
    aa.moe = 0;
    aa.head_calls = 0;
    bb.attention = 0;
    bb.moe = 0;
    bb.head_calls = 0;
    return std.meta.eql(aa, bb);
}

fn plannerPriceTransition(previous: u8, width: u8, pending: bool) bool {
    return width > 0 and (pending or previous != width);
}

fn plannerRecurringInputSame(a: group_cost_mod.GroupShape, b: group_cost_mod.GroupShape) bool {
    if (a.n != b.n or a.width() == 0 or b.width() == 0) return false;
    inline for (.{ @as(u6, 24), @as(u6, 32) }) |shift| {
        var a_max: u8 = 0;
        var b_max: u8 = 0;
        for (a.rows[0..a.n], b.rows[0..b.n]) |arow, brow| {
            a_max = @max(a_max, group_cost_mod.GroupShape.part(arow, shift));
            b_max = @max(b_max, group_cost_mod.GroupShape.part(brow, shift));
        }
        if (a_max != b_max) return false;
    }
    var aa = a;
    var bb = b;
    const recurring_mask = ~((@as(u64, 0xff) << 24) | (@as(u64, 0xff) << 32) | (@as(u64, 0xff) << 40));
    for (aa.rows[0..aa.n], bb.rows[0..bb.n]) |*arow, *brow| {
        if (group_cost_mod.GroupShape.part(arow.*, 40) < 2 or group_cost_mod.GroupShape.part(brow.*, 40) < 2) return false;
        arow.* &= recurring_mask;
        brow.* &= recurring_mask;
    }
    return plannerInputSame(aa.canonical(), bb.canonical());
}

const PlannerPrices = struct {
    slots: []const *Slot,
    geometry: group_cost_mod.GroupShape,

    fn init(slots: []const *Slot) PlannerPrices {
        var widths: [Planner.MAX_ROWS]u8 = @splat(0);
        for (slots, 0..) |slot, i| widths[i] = @intFromBool(slot.legacy_gen.?.mtp != null and slot.legacy_gen.?.mtp_cache != null);
        return .{ .slots = slots, .geometry = plannerShape(slots, widths[0..slots.len]) };
    }

    fn keyFor(self: *const PlannerPrices, slots: []const *Slot, widths: []const u8) group_cost_mod.GroupShape {
        var key = self.geometry;
        key.n = @intCast(slots.len);
        key.rows = @splat(0);
        key.head_format = 0;
        for (slots, widths, 0..) |slot, width, i| {
            for (self.slots, 0..) |original, j| {
                if (original != slot) continue;
                const code = self.geometry.rows[j];
                key.rows[i] = if (width == 0) group_cost_mod.GroupShape.row(0, 0, 0, group_cost_mod.GroupShape.part(code, 24), 0, 0, group_cost_mod.GroupShape.part(code, 48) & 8) else (code & ~@as(u64, 0xffffff)) | @as(u64, width) * 0x10101;
                if (width > 0) key.head_format = self.geometry.head_format;
                break;
            }
        }
        return key.canonical();
    }

    fn sample(self: *const PlannerPrices, slots: []const *Slot, widths: []const u8, trusted: bool) ?Planner.Price {
        if (slots.len == 0) return .{ .ms = 0, .samples = std.math.maxInt(u32) };
        const key = self.keyFor(slots, widths);
        const table = &slots[0].model.transformer.?.mtp_group_cost;
        var selected: ?Planner.Price = null;
        var score: f32 = 0;
        var max_gap: f32 = 0;
        var mean_upper: f32 = 0;
        var mean_lower: f32 = std.math.inf(f32);
        const plain = key.width() == 0;
        for (table.shapes) |entry| {
            if (entry.cell.n == 0 or !plannerInputSame(key, entry.shape)) continue;
            if (trusted and entry.cell.n < Planner.MIN_SAMPLES) continue;
            const deviation = 2 * @sqrt(entry.variance);
            const mean_deviation = 2 * @sqrt(entry.variance * entry.mean_weight_sq);
            mean_upper = @max(mean_upper, entry.cell.ms + mean_deviation);
            mean_lower = @min(mean_lower, entry.cell.ms - mean_deviation);
            const value = if (plain) entry.cell.ms - deviation else entry.cell.ms + deviation;
            for (entry.gap_ms[0..key.n]) |gap| max_gap = @max(max_gap, gap);
            if (selected == null or (if (plain) value < score else value > score)) {
                score = value;
                selected = .{ .ms = entry.cell.ms, .variance = entry.variance, .samples = entry.cell.n };
            }
        }
        if (selected) |*value| {
            value.max_gap_ms = max_gap;
            if (!plain) value.mean_upper_ms = mean_upper;
            if (plain) value.mean_lower_ms = mean_lower;
            return selected;
        }
        if (plain) return null;

        var pooled_n: u32 = 0;
        var sum: f64 = 0;
        var sum_sq: f64 = 0;
        var mean_sum_sq: f64 = 0;
        max_gap = 0;
        for (table.shapes) |entry| {
            if (entry.cell.n == 0 or !plannerRecurringInputSame(key, entry.shape)) continue;
            const n: f64 = @floatFromInt(entry.cell.n);
            const mean: f64 = entry.cell.ms;
            pooled_n +|= entry.cell.n;
            sum += n * mean;
            sum_sq += n * (@as(f64, entry.variance) + mean * mean);
            // History-to-history differences remain uncertainty in a pooled prediction.
            mean_sum_sq += n * (@as(f64, entry.variance) * entry.mean_weight_sq + mean * mean);
            for (entry.gap_ms[0..key.n]) |gap| max_gap = @max(max_gap, gap);
        }
        const required: u32 = if (trusted) Planner.MIN_SAMPLES else 1;
        if (pooled_n < required) return null;
        const n: f64 = @floatFromInt(pooled_n);
        const mean = sum / n;
        const variance = @max(0, sum_sq / n - mean * mean);
        const mean_variance = @max(0, mean_sum_sq / n - mean * mean);
        return .{ .ms = @floatCast(mean), .variance = @floatCast(variance), .samples = pooled_n, .max_gap_ms = max_gap, .mean_upper_ms = @floatCast(mean + 2 * @sqrt(mean_variance)) };
    }

    pub fn price(self: *PlannerPrices, widths: []const u8) ?Planner.Price {
        var positive: [Planner.MAX_ROWS]*Slot = undefined;
        var drafts: [Planner.MAX_ROWS]u8 = undefined;
        var zeros: [Planner.MAX_ROWS]*Slot = undefined;
        const zero_widths: [Planner.MAX_ROWS]u8 = @splat(0);
        var pn: usize = 0;
        var zn: usize = 0;
        for (self.slots, widths) |slot, width| {
            if (width == 0) {
                zeros[zn] = slot;
                zn += 1;
            } else {
                positive[pn] = slot;
                drafts[pn] = width;
                pn += 1;
            }
        }
        const plain = self.sample(zeros[0..zn], zero_widths[0..zn], true) orelse return null;
        const spec = self.sample(positive[0..pn], drafts[0..pn], true) orelse return null;
        const deviation = @sqrt(plain.variance) + @sqrt(spec.variance);
        return .{ .ms = plain.ms + spec.ms, .variance = deviation * deviation, .samples = @min(plain.samples, spec.samples), .max_gap_ms = @max(plain.max_gap_ms, spec.max_gap_ms), .mean_upper_ms = if (spec.mean_upper_ms) |bound| plain.ms + 2 * @sqrt(plain.variance) + bound else null, .mean_lower_ms = if (pn == 0) plain.mean_lower_ms else null };
    }

    fn probeEstimate(self: *PlannerPrices, widths: []const u8, plain: Planner.Price) f32 {
        if (self.sample(self.slots, widths, false)) |value| return value.ms + 2 * @sqrt(value.variance);
        var widest: u8 = 0;
        for (widths) |width| widest = @max(widest, width);
        var lower: [Planner.MAX_ROWS]u8 = undefined;
        for ([_]u8{ 2, 1 }) |cap| {
            if (cap >= widest) continue;
            for (widths, 0..) |width, i| lower[i] = @min(width, cap);
            if (self.sample(self.slots, lower[0..widths.len], false)) |value| return scaledProbeEstimate(value, widest, cap);
        }
        if (self.slots.len >= 4 and self.slots.len % 2 == 0) {
            const half = self.slots.len / 2;
            if (self.sample(self.slots[0..half], widths[0..half], false)) |a| {
                if (self.sample(self.slots[half..], widths[half..], false)) |b| return a.ms + b.ms + 2 * (@sqrt(a.variance) + @sqrt(b.variance));
            }
        }
        return self.probeFallbackEstimate(widths, plain);
    }

    fn probeFallbackEstimate(_: *PlannerPrices, widths: []const u8, plain: Planner.Price) f32 {
        var widest: u8 = 0;
        for (widths) |width| widest = @max(widest, width);
        const upper = plain.ms + 2 * @sqrt(plain.variance);
        if (widest <= 2) return 2 * upper;
        return 1.5 * upper * @as(f32, @floatFromInt(widest + 1));
    }

    fn scaledProbeEstimate(value: Planner.Price, widest: u8, measured: u8) f32 {
        const ratio = @as(f32, @floatFromInt(widest + 1)) / @as(f32, @floatFromInt(measured + 1));
        return value.ms * ratio;
    }
};

var planner_engaged_logged = false;
var planner_calibration_block_logged = false;
var planner_depth_two_reject_logged = false;

const ProbeSource = enum { calibrate, recovery };

/// Widths for a calibration or recovery probe round, priced against the plain
/// tick. False leaves `decision` untouched: no candidate, no price, or over budget.
fn planProbe(
    prices: *PlannerPrices,
    active: []*Slot,
    rows: []const Planner.Row,
    decision: *Planner.Decision,
    comptime source: ProbeSource,
) bool {
    const zero_widths: [Planner.MAX_ROWS]u8 = @splat(0);
    var candidate: [Planner.MAX_ROWS]u8 = @splat(0);
    for (active, rows, 0..) |slot, row, i| {
        const gen = &slot.legacy_gen.?;
        candidate[i] = switch (source) {
            .calibrate => Planner.probeWidth(gen.mtp_planner_probes, gen.max_tokens -| gen.completion_tokens, row.cap),
            .recovery => gen.mtp_planner_recovery.width(gen.completion_tokens, gen.max_tokens -| gen.completion_tokens, row.cap, gen.mtp_planner_probes),
        } orelse return false;
    }
    const plain = prices.price(zero_widths[0..active.len]) orelse return false;
    const estimate = prices.probeEstimate(candidate[0..active.len], plain);
    var latency_limit: f32 = std.math.floatMax(f32);
    var available = true;
    for (rows) |row| {
        latency_limit = @min(latency_limit, row.latency_ms);
        if (estimate > row.latency_ms) available = false;
    }
    if (!available) {
        if (source == .calibrate and !planner_calibration_block_logged) {
            planner_calibration_block_logged = true;
            log.info("[mtp-planner] calibration blocked widths={any} predicted_ms={d:.2} latency_ms={d:.2}\n", .{ candidate[0..active.len], estimate, latency_limit });
        }
        return false;
    }
    decision.widths = candidate;
    decision.upper_ms = estimate;
    return true;
}

pub fn tryPlannerTick(sch: *Scheduler, active: []*Slot) anyerror!bool {
    if (!Planner.enabled() or active.len == 0 or active.len > Planner.MAX_ROWS) return false;
    const xfm = active[0].model.transformer orelse return false;
    if ((xfm.qwen4Trunk() orelse return false).qwen4_mtp == null) return false;
    if (active.len == 1) {
        const gen = if (active[0].legacy_gen) |*g| g else return false;
        if (!gen.mtp_planner_owned) return false;
    }
    var rows: [Planner.MAX_ROWS]Planner.Row = undefined;
    var lenses: [Planner.MAX_ROWS]u32 = undefined;
    var has_mtp = false;
    var pending_pipeline = false;
    var pending_draft = false;
    for (active, 0..) |slot, i| {
        if (slot.model != active[0].model or !slot.allow_batch_mtp) return false;
        const gen = if (slot.legacy_gen) |*g| g else return false;
        if (gen.ctx.mrope_pos != null) return false;
        pending_pipeline = pending_pipeline or gen.has_pending_logits or gen.has_pending_token;
        pending_draft = pending_draft or gen.mtp_pre_draft != null;
        const why = @import("../scheduler.zig").batchVerdict(sch, slot);
        const mtp_row = why == .spec_active and slotMtpGroupable(slot);
        if (why != .ok and !mtp_row) return false;
        has_mtp = has_mtp or mtp_row;
        const cap = if (mtp_row) @min(@min(if (gen.mtp_depth > 0) gen.mtp_depth else @max(1, gen.mtp_depth_current), Planner.MAX_DEPTH), gen.max_tokens -| gen.completion_tokens -| 1) else 0;
        rows[i] = .{ .history = &gen.mtp_planner_history, .cap = @intCast(cap), .latency_ms = Planner.LATENCY_MS };
        lenses[i] = @intCast(slot.moe_seq_offset);
    }
    if (!has_mtp) return false;
    std.sort.insertion(u32, lenses[0..active.len], {}, std.sort.asc(u32));
    if (active.len > 1 and batchedKvKeepCount(lenses[0..active.len]) != active.len) return false;
    const entry = Planner.entry(pending_pipeline, pending_draft);
    if (entry == .wait_predraft) {
        // Consume the pending rounds without perpetually replacing their successors.
        for (active) |slot| slot.legacy_gen.?.mtp_planner_pending = true;
        return false;
    }
    if (entry == .drain) {
        for (active, rows[0..active.len]) |slot, row| {
            const gen = &slot.legacy_gen.?;
            if (gen.mtp != null) try gen.mtpDetachHead(slot.allocator, true);
            if (row.cap > 0) gen.mtp_planner_owned = true;
            slot.planner_plain_transition = true;
            slot.mtp_plain_tick = row.cap > 0;
        }
        log.info("[mtp-planner] rows={d} action=drain\n", .{active.len});
        try runBatchedDecodeTick(sch, active);
        return true;
    }

    var live: [Planner.MAX_ROWS]*Slot = undefined;
    var count: usize = 0;
    for (active) |slot| {
        const gen = &slot.legacy_gen.?;
        if (try loopGuardTick(sch, slot, gen)) continue;
        if (try gen.checkStop()) {
            finishSlot(sch, slot, gen.finish_reason);
            continue;
        }
        live[count] = slot;
        count += 1;
    }
    if (count != active.len) {
        if (count > 0) try runDecodeTick(sch, live[0..count]);
        return true;
    }
    for (active, rows[0..active.len]) |slot, row| if (row.cap > 0) {
        slot.legacy_gen.?.mtp_planner_owned = true;
    };
    var prices = PlannerPrices.init(active);
    var decision = Planner.choose(rows[0..active.len], &prices);
    var probe = false;
    var recovering = false;
    var probe_rounds: u8 = std.math.maxInt(u8);
    for (active, rows[0..active.len]) |slot, row| if (row.cap > 0) {
        probe_rounds = @min(probe_rounds, slot.legacy_gen.?.mtp_planner_probes);
    };
    if (!planner_depth_two_reject_logged and probe_rounds >= 12 and Planner.shouldProbe(decision, probe_rounds)) {
        var depth_two: [Planner.MAX_ROWS]u8 = @splat(0);
        var expected: f32 = 0;
        for (rows[0..active.len], 0..) |row, i| {
            depth_two[i] = @min(2, row.cap);
            expected += row.history.expected(depth_two[i]);
        }
        planner_depth_two_reject_logged = true;
        log.info("[mtp-planner] depth-two rejected price={any} expected_tokens={d:.2} plain_rate={d:.1}\n", .{ prices.price(depth_two[0..active.len]), expected, decision.plain_rate });
    }
    if (Planner.shouldProbe(decision, probe_rounds)) {
        probe = planProbe(&prices, active, rows[0..active.len], &decision, .calibrate);
    }
    if (!probe and Planner.shouldProbe(decision, probe_rounds) and planProbe(&prices, active, rows[0..active.len], &decision, .recovery)) {
        probe = true;
        recovering = true;
    }
    // A width the range moves no longer matches what a probe was meant to price.
    for (rows[0..active.len], 0..) |row, i| {
        const width: u8 = @intCast(depth_bounds.plannerWidth(decision.widths[i], row.cap, depth_bounds.active));
        if (width != decision.widths[i] or depth_bounds.active.pinned() != null) {
            decision.widths[i] = width;
            probe = false;
            recovering = false;
        }
    }
    var stale: [Planner.MAX_ROWS]bool = undefined;
    for (active, 0..) |slot, row| stale[row] = slot.legacy_gen.?.mtp_hidden_stale;
    const execution = Planner.execution(decision.widths[0..active.len], stale[0..active.len]);
    const prime = execution.prime;
    var positives: [Planner.MAX_ROWS]*Slot = undefined;
    var plains: [Planner.MAX_ROWS]*Slot = undefined;
    const pn = execution.speculative_n;
    const zn = execution.plain_n;
    var changed = false;
    for (active, decision.widths[0..active.len]) |slot, width| {
        const row_changed = slot.planner_last_width != width;
        changed = changed or row_changed;
        slot.planner_price_transition = plannerPriceTransition(slot.planner_last_width, width, slot.planner_price_transition);
        slot.planner_last_width = width;
    }
    for (execution.plain[0..zn], 0..) |row, i| {
        const slot = active[row];
        const gen = &slot.legacy_gen.?;
        slot.planner_plain_transition = gen.mtp_hist_stash != null or gen.mtp_pre_draft != null or prime;
        if (gen.mtp != null) try gen.mtpDetachHead(slot.allocator, true);
        slot.mtp_plain_tick = execution.capture[row];
        plains[i] = slot;
    }
    for (execution.speculative[0..pn], 0..) |row, i| {
        const slot = active[row];
        const gen = &slot.legacy_gen.?;
        gen.mtp_planner_width = decision.widths[row];
        gen.mtp_planner_probe = probe;
        gen.mtp_planner_recovering = recovering;
        positives[i] = slot;
    }
    if (!planner_engaged_logged) {
        planner_engaged_logged = true;
        log.info("[mtp-planner] engaged: margin=5% noise=2sigma latency_ms={d:.1} max_rows=8\n", .{Planner.LATENCY_MS});
    }
    if (changed or prime or probe) log.info("[mtp-planner] rows={d} widths={any} action={s} predicted_ms={d:.2} rate={d:.1} plain={d:.1}\n", .{ active.len, decision.widths[0..active.len], if (prime) "prime" else if (recovering) "recover" else if (probe) "calibrate" else "choose", decision.upper_ms, decision.rate, decision.plain_rate });
    if (pn > 0) {
        try runBatchedMtpHeadTick(sch, positives[0..pn]);
    }
    if (zn > 0) try runBatchedDecodeTick(sch, plains[0..zn]);
    return true;
}


/// Why this group cannot take the merged `[N, S]` verify forward, or null for a shape that
/// may: a qwen4 trunk's batched sub-paths are decode-shaped and uncertified past width 1.
pub fn mergedVerifyDeclineReason(has_qwen4: bool, width: u32) ?[]const u8 {
    if (has_qwen4 and width > 1) return "qwen4 trunk past verify width 1";
    return null;
}

pub const VerifyShape = enum { solo, row_axis, merged };

pub fn verifyShapeFor(rows: usize, has_qwen4: bool, width: u32) VerifyShape {
    if (rows <= 1) return .solo;
    if (mergedVerifyDeclineReason(has_qwen4, width) != null) return .row_axis;
    return .merged;
}

fn mtpRoundsStaySolo(slot: *const Slot) bool {
    const t = slot.model.transformer orelse return true;
    // A stateless trunk (MiMo) has no grouped verify: its MTP rounds run alone.
    if (!t.hasRecurrentLayers()) return true;
    // A model whose grouped verify measured no better than solo rounds says so.
    const b = t.config.plugin orelse return false;
    return b.parsed.spec.policy.mtp_group_solo;
}

pub fn mtpCrowdThresholdFor(slot: *const Slot) usize {
    // Solo MTP rounds interleave one slot at a time, so any company decodes
    // faster as plain batched ticks.
    if (slot.model.transformer) |t| if (!t.hasRecurrentLayers()) return 2;
    return if (mtpRoundsStaySolo(slot)) 3 else mtpCrowdThreshold();
}

fn runBatchedMtpHeadTick(sch: *Scheduler, group: []*Slot) !void {
    const cost_xfm = group[0].model.transformer.?;
    // The forward records its attention and MoE arms on qwen4_exp's trunk.
    const cost_trunk = cost_xfm.qwen4Trunk();
    if (cost_trunk) |t| {
        t.cost_trace_active = true;
        t.cost_attention = 0;
        t.cost_moe = 0;
    }
    const head_calls_before = qmv_rows.mtp_head_row_dispatches;
    defer if (cost_trunk) |t| {
        t.cost_trace_active = false;
    };
    defer for (group) |slot| {
        if (slot.legacy_gen) |*gen| gen.mtp_batch_head = false;
    };
    var tick_sw = io_util.Stopwatch.init(group[0].io);
    var gens: [MAX_BATCH_GROUP]*generate_mod.Generator = undefined;
    var chains: [MAX_BATCH_GROUP]generate_mod.Generator.MtpPreDraft = undefined;
    var opens: [MAX_BATCH_GROUP]generate_mod.Generator.MtpRoundOpen = undefined;
    var live: [MAX_BATCH_GROUP]*Slot = undefined;
    var n: usize = 0;
    var n_chain: usize = 0;
    defer for (opens[0..n]) |*o| {
        if (o.chain.drafts.len != 0) o.chain.deinit(live[0].allocator);
    };
    defer for (chains[0..n_chain]) |*c| c.deinit(live[0].allocator);

    for (group) |slot| {
        const gen = &slot.legacy_gen.?;
        if (try loopGuardTick(sch, slot, gen)) continue;
        gen.mtp_batch_head = true;
        const begun = gen.mtpRoundBegin(slot.allocator) catch |e| {
            gen.mtp_batch_head = false;
            slot.markError(@errorName(e));
            continue;
        };
        switch (begun) {
            .done => |r| publishMtpResult(sch, slot, gen, r),
            .verify => |st_in| {
                var state = st_in;
                defer state.deinit(slot.allocator);
                gen.mtpRoundVerify(&state) catch |e| {
                    slot.markError(@errorName(e));
                    continue;
                };
                mtpFinishPublish(sch, slot, gen, &state);
            },
            .open => |o| {
                opens[n] = o;
                gens[n] = gen;
                live[n] = slot;
                n += 1;
            },
        }
    }
    if (n == 0) return;

    var depth: u32 = 1;
    for (opens[0..n], 0..) |*o, i| {
        depth = @max(depth, o.chain.m);
        chains[i] = o.chain;
        o.chain = .{
            .plan = chains[i].plan,
            .off0 = 0,
            .t1 = 0,
            .t1_arr = .{ .ctx = null },
            .drafts = &.{},
            .draft_arrs = &.{},
            .n_drafted = 0,
            .conf_arrs = null,
            .n_conf = 0,
            .q_probs = null,
            .n_qp = 0,
            .h_chain = null,
            .m = 0,
        };
        n_chain = i + 1;
    }

    const chain_lap = generate_mod.Generator.SubLap.start(generate_mod.Generator.mtpTraceOn(), live[0].io);
    generate_mod.Generator.mtpChainBuildBatched(gens[0..n], chains[0..n], 0, depth) catch |e| {
        for (live[0..n]) |slot| slot.markError(@errorName(e));
        return;
    };
    // Dispatched before the verify build so the chain's GPU time covers the build.
    generate_mod.Generator.mtpChainDispatchBatched(chains[0..n]) catch |e| {
        for (live[0..n]) |slot| slot.markError(@errorName(e));
        return;
    };
    if (chain_lap.read()) |chain_ns| {
        for (gens[0..n]) |gen| gen.mtpTraceSub(.chain, chain_ns);
    }

    const taken = n;
    n = 0;
    n_chain = 0;
    var states: [MAX_BATCH_GROUP]generate_mod.Generator.MtpRoundState = undefined;
    var vgens: [MAX_BATCH_GROUP]*generate_mod.Generator = undefined;
    var vlive: [MAX_BATCH_GROUP]*Slot = undefined;
    var vn: usize = 0;
    defer for (states[0..vn], vlive[0..vn]) |*st, slot| {
        st.deinit(slot.allocator);
    };
    for (0..taken) |i| {
        opens[i].chain = chains[i];
        states[vn] = gens[i].mtpRoundContinue(live[i].allocator, opens[i]) catch |e| {
            live[i].markError(@errorName(e));
            continue;
        };
        vgens[vn] = gens[i];
        vlive[vn] = live[i];
        vn += 1;
    }
    if (vn == 0) return;

    var sts: [MAX_BATCH_GROUP]*generate_mod.Generator.MtpRoundState = undefined;
    for (states[0..vn], 0..) |*st, i| sts[i] = st;
    if (generate_mod.Generator.mtpGroupVerify(vgens[0..vn], sts[0..vn])) |_| {
        generate_mod.Generator.mtpGroupAccept(vgens[0..vn], sts[0..vn]) catch |e| {
            for (vlive[0..vn]) |slot| slot.markError(@errorName(e));
            return;
        };
    } else |e| {
        for (vlive[0..vn]) |slot| slot.markError(@errorName(e));
        return;
    }
    for (states[0..vn], vgens[0..vn], vlive[0..vn]) |*st, gen, slot| {
        if (slot.state == .errored) continue;
        mtpFinishPublish(sch, slot, gen, st);
    }
    observeGroupRound(vlive[0..vn], states[0..vn], group.len, &tick_sw, head_calls_before);
}

/// Group MTP slots by model (pad-waste capped like the plain group) and run one
/// batched verify per group; a group of one takes the solo round.
pub fn runMtpGroups(sch: *Scheduler, slots: []*Slot) !void {
    if (slots.len == 0) return;
    std.sort.pdq(*Slot, slots, {}, struct {
        fn lt(_: void, a: *Slot, b: *Slot) bool {
            return @intFromPtr(a.model) < @intFromPtr(b.model);
        }
    }.lt);
    var start: usize = 0;
    while (start < slots.len) {
        var end = start + 1;
        while (end < slots.len and slots[end].model == slots[start].model) end += 1;
        var group = slots[start..end];
        if (mtpRoundsStaySolo(group[0])) {
            for (group) |slot| try runSingleDecodeTick(sch, slot);
            start = end;
            continue;
        }
        if (group.len >= 2) {
            var kv_lens: [MAX_BATCH_GROUP]u32 = undefined;
            var caches_buf: [MAX_BATCH_GROUP]*const KVCache = undefined;
            for (group, 0..) |g, i| {
                caches_buf[i] = &g.cache;
            }
            fillGroupPadWasteKvLens(caches_buf[0..group.len], group[0].model.config, 2, kv_lens[0..group.len]);
            const ctx_max = groupCtxMax(caches_buf[0..group.len]);
            var i: usize = 1;
            while (i < group.len) : (i += 1) {
                const slot_i = group[i];
                const len_i = kv_lens[i];
                var j = i;
                while (j > 0 and kv_lens[j - 1] > len_i) : (j -= 1) {
                    group[j] = group[j - 1];
                    kv_lens[j] = kv_lens[j - 1];
                }
                group[j] = slot_i;
                kv_lens[j] = len_i;
            }
            const keep = groupKeepCount(kv_lens[0..group.len], groupAttendsPerSlot(group[0]), ctx_max);
            for (group[keep..]) |s| {
                noteSerial(sch, s, .pad_waste);
                try runSingleDecodeTick(sch, s);
            }
            group = group[0..keep];
        }
        // Sub-groups sized to the verify lane's row budget; the leftover slot rounds solo.
        var g0: usize = 0;
        while (g0 < group.len) {
            const rem = group.len - g0;
            const plan = mtpSubGroupPlan(rem, transformer_mod.verifyQmmTile());
            const size = plan.size;
            const sub = group[g0 .. g0 + size];
            if (size >= 2) {
                const cap = plan.depthCap();
                for (sub) |slot| {
                    slot.legacy_gen.?.mtp_group_cap = cap;
                    slot.legacy_gen.?.mtp_group_fill = plan.fill;
                }
                try runBatchedMtpTick(sch, sub);
            } else {
                sub[0].legacy_gen.?.mtp_group_cap = 0;
                try runSingleDecodeTick(sch, sub[0]);
            }
            g0 += size;
        }
        start = end;
    }
}

/// MTP slots on one model at or past this count decode on the plain batched tick instead:
/// sub-grouped verify rounds lose to one plain tick there.
fn mtpCrowdThreshold() usize {
    return mtpCrowdThresholdForTile(transformer_mod.verifyQmmTile());
}

pub fn mtpCrowdThresholdForTile(tile: transformer_mod.VqmmTile) usize {
    return mtpSubGroupPlan(std.math.maxInt(usize), tile).size + 1;
}

pub const SubGroup = struct {
    size: usize,
    /// Rows the batched verify may carry: 7 keeps the projections on the split-K lane,
    /// 16 is one matmul2d tile.
    row_cap: u32,
    /// The rows are one fixed-cost tile: every lane drafts to `depthCap`.
    fill: bool = false,

    pub fn depthCap(self: SubGroup) u32 {
        return self.row_cap / @as(u32, @intCast(@max(self.size, 1))) - 1;
    }
};

/// The next sub-group. The M5 tile takes up to four lanes at any count. The M4 shader tile
/// costs a fixed 16 rows, which two or three lanes cannot fill with drafts worth verifying,
/// so it starts at four lanes and carries up to eight at depth 1.
pub fn mtpSubGroupPlan(remaining: usize, tile: transformer_mod.VqmmTile) SubGroup {
    switch (tile) {
        .nax => return .{ .size = mtpSubGroupSize(remaining, 16), .row_cap = 16 },
        .shader => if (remaining >= 4) return .{ .size = @min(remaining, 8), .row_cap = 16, .fill = true },
        .off => {},
    }
    return .{ .size = mtpSubGroupSize(remaining, 7), .row_cap = 7 };
}

/// Slots in the next sub-group: two at depth 2 or three at depth 1 fill 7 rows; four is
/// better as 2+2 than 3+1.
pub fn mtpSubGroupSize(remaining: usize, row_cap: u32) usize {
    if (remaining < 2) return remaining;
    if (row_cap >= 16) return @min(remaining, 4);
    if (remaining == 3 or remaining >= 5) return 3;
    return 2;
}

fn runBatchedMtpTick(sch: *Scheduler, group: []*Slot) !void {
    var inner_err: ?anyerror = null;
    runBatchedMtpTickInner(sch, group) catch |e| {
        inner_err = e;
    };
    mlx.checkErrorDecode() catch |mlx_err| {
        log.err("[scheduler] batched verify aborted: MLX failure mid-generation ({s}) — failing these requests, the server keeps serving\n", .{@errorName(mlx_err)});
        for (group) |slot| slot.markError(@errorName(mlx_err));
        return;
    };
    if (inner_err) |e| return e;
}

/// One speculative round for a group: the lanes draft as rows of one head forward, the
/// trunk verifies all of them in ONE `[N, S]` forward (rows padded to the widest draft),
/// every slot accepts and rolls back on its own row, and the next round's chain is on the
/// GPU before anything is published.
fn runBatchedMtpTickInner(sch: *Scheduler, group: []*Slot) !void {
    const allocator = sch.allocator;
    var states: [MAX_BATCH_GROUP]Generator.MtpRoundState = undefined;
    var live: [MAX_BATCH_GROUP]*Slot = undefined;
    var n: usize = 0;
    defer for (states[0..n], live[0..n]) |*st, slot| {
        st.deinit(slot.allocator);
    };
    var opens: [MAX_BATCH_GROUP]Generator.MtpRoundOpen = undefined;
    var open_slots: [MAX_BATCH_GROUP]*Slot = undefined;
    var open_n: usize = 0;
    defer for (opens[0..open_n], open_slots[0..open_n]) |*o, slot| {
        o.chain.deinit(slot.allocator);
    };
    // Set for the whole tick: a begin without a pre-draft hands its chain back to be
    // built with the group's, and no finish pre-drafts on its own.
    defer for (group) |slot| {
        slot.legacy_gen.?.mtp_batch_head = false;
    };

    for (group) |slot| {
        const gen = &slot.legacy_gen.?;
        if (try loopGuardTick(sch, slot, gen)) continue;
        gen.mtp_batch_head = true;
        const begun = gen.mtpRoundBegin(slot.allocator) catch |e| {
            slot.markError(@errorName(e));
            continue;
        };
        switch (begun) {
            .done => |r| publishMtpResult(sch, slot, gen, r),
            .verify => |st| {
                states[n] = st;
                live[n] = slot;
                n += 1;
            },
            .open => |o| {
                opens[open_n] = o;
                open_slots[open_n] = slot;
                open_n += 1;
            },
        }
    }
    if (open_n > 0) {
        var gens: [MAX_BATCH_GROUP]*Generator = undefined;
        var chains: [MAX_BATCH_GROUP]Generator.MtpPreDraft = undefined;
        var depth: u32 = 0;
        for (opens[0..open_n], open_slots[0..open_n], 0..) |*o, slot, i| {
            gens[i] = &slot.legacy_gen.?;
            chains[i] = o.chain;
            depth = @max(depth, o.chain.m);
        }
        const chain_lap = Generator.SubLap.start(Generator.mtpTraceOn(), open_slots[0].io);
        const built = Generator.mtpChainBuildBatched(gens[0..open_n], chains[0..open_n], 0, depth);
        for (opens[0..open_n], chains[0..open_n]) |*o, c| o.chain = c;
        const dispatched = if (built) |_| Generator.mtpChainDispatchBatched(chains[0..open_n]) else |e| e;
        if (chain_lap.read()) |ns| for (gens[0..open_n]) |gen| gen.mtpTraceSub(.chain, ns);
        var taken = open_n;
        dispatched catch |e| {
            for (open_slots[0..open_n]) |slot| slot.markError(@errorName(e));
            taken = 0;
        };
        if (taken > 0) open_n = 0;
        for (opens[0..taken], open_slots[0..taken], 0..) |o, slot, i| {
            // `mtpRoundContinue` owns the chain from here, on success and on error.
            states[n] = gens[i].mtpRoundContinue(slot.allocator, o) catch |e| {
                slot.markError(@errorName(e));
                continue;
            };
            live[n] = slot;
            n += 1;
        }
    }
    if (n == 0) return;

    var results: [MAX_BATCH_GROUP]?Generator.DrafterStepResult = @splat(null);
    const merged = n > 1 and try verifyGroupMerged(sch, states[0..n], live[0..n]);
    var gens: [MAX_BATCH_GROUP]*Generator = undefined;
    var running: usize = 0;
    for (states[0..n], live[0..n], 0..) |*st, slot, i| {
        const gen = &slot.legacy_gen.?;
        if (!merged) gen.mtpRoundVerify(st) catch |e| {
            slot.markError(@errorName(e));
            continue;
        };
        results[i] = gen.mtpRoundFinish(slot.allocator, st) catch |e| {
            slot.markError(@errorName(e));
            continue;
        };
        if (results[i] != null) {
            gens[running] = gen;
            running += 1;
        }
    }
    if (running > 0) Generator.mtpGroupPreDraft(gens[0..running], allocator) catch |e| {
        for (live[0..n]) |slot| slot.markError(@errorName(e));
    };
    for (live[0..n], results[0..n]) |slot, r| {
        if (slot.state == .errored) {
            if (r) |res| slot.allocator.free(res.tokens);
            continue;
        }
        publishMtpResult(sch, slot, &slot.legacy_gen.?, r);
    }
}

/// The group's verify as ONE `[N, S]` trunk forward. False = no recurrent state to merge
/// yet, every row verifies solo this tick.
fn verifyGroupMerged(sch: *Scheduler, states: []Generator.MtpRoundState, live: []*Slot) !bool {
    const allocator = sch.allocator;
    const n = states.len;
    const xfm = live[0].model.transformer.?;
    const s_stream = xfm.s;
    var width: u32 = 0;
    for (states) |*st| width = @max(width, st.verify_len);
    const ctxs = try allocator.alloc(*ForwardCtx, n);
    defer allocator.free(ctxs);
    const rope_offsets = try allocator.alloc(u32, n);
    defer allocator.free(rope_offsets);
    for (live, 0..) |slot, i| {
        ctxs[i] = &slot.legacy_gen.?.ctx;
        rope_offsets[i] = @intCast(slot.moe_seq_offset);
    }
    if (!xfm.batchedGdnReady(ctxs)) return false;

    // [N, width] rows: each verify input right-padded with token 0 (never read past 1+m).
    const rows_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(rows_vec);
    var padded: [MAX_BATCH_GROUP]mlx.mlx_array = undefined;
    var padded_n: usize = 0;
    defer for (padded[0..padded_n]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (states) |*st| {
        const pad: c_int = @intCast(width - st.verify_len);
        var row = mlx.mlx_array_new();
        if (pad > 0) {
            const axes = [_]c_int{1};
            const low = [_]c_int{0};
            const high = [_]c_int{pad};
            const zero = mlx.mlx_array_new_int(0);
            defer _ = mlx.mlx_array_free(zero);
            try mlx.check(mlx.mlx_pad(&row, st.verify_input, &axes, 1, &low, 1, &high, 1, zero, "constant", s_stream));
        } else {
            try mlx.check(mlx.mlx_array_set(&row, st.verify_input));
        }
        padded[padded_n] = row;
        padded_n += 1;
        _ = mlx.mlx_vector_array_append_value(rows_vec, row);
    }
    var token_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(token_arr);
    try mlx.check(mlx.mlx_concatenate_axis(&token_arr, rows_vec, 0, s_stream));

    var hidden_last = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden_last);
    var hidden_all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden_all);
    const logits = try xfm.forwardMoeBatchedVerify(token_arr, ctxs, rope_offsets, &hidden_last, &hidden_all);
    defer _ = mlx.mlx_array_free(logits);
    // The batched forward moved only its scratch offset; the solo forward would have
    // advanced every slot by the rows it ran, and the finish rolls back from there.
    for (live) |slot| slot.moe_seq_offset += width;

    const logit_rows = try Transformer.sliceBatchRows(allocator, s_stream, logits, n);
    defer allocator.free(logit_rows);
    const last_rows = try Transformer.sliceBatchRows(allocator, s_stream, hidden_last, n);
    defer allocator.free(last_rows);
    const all_rows = try Transformer.sliceBatchRows(allocator, s_stream, hidden_all, n);
    defer allocator.free(all_rows);
    for (states, 0..) |*st, i| {
        st.verify_logits = logit_rows[i];
        st.new_hidden = last_rows[i];
        st.verify_hidden_all = all_rows[i];
        st.verify_len = width;
    }
    if (sch.metrics) |m| m.batched_group_size.set(n);
    return true;
}

fn groupCostSampleComplete(expected: usize, retained: []const u32, published: []const bool) bool {
    if (expected < 1 or expected > group_cost_mod.MAX_GROUP_ROWS or retained.len != expected or published.len != expected) return false;
    for (retained, published) |count, complete| if (count == 0 or !complete) return false;
    return true;
}

fn cacheCostFormat(config: transformer_mod.KVQuantConfig) u64 {
    return (@as(u64, config.group_size) << 16) | (@as(u64, config.bits) << 8) | @backingInt(config.scheme);
}

/// One wall for the group grid: from the tick's first round-begin to its last publish.
/// Both group sites hand their tick stopwatch here rather than starting one of their own.
fn observeGroupRound(live: []*Slot, states: []const Generator.MtpRoundState, expected: usize, tick_sw: *io_util.Stopwatch, head_calls_before: u64) void {
    const xfm = live[0].model.transformer orelse return;
    if (live.len > group_cost_mod.MAX_GROUP_ROWS) return;
    var price_transition = false;
    for (live) |slot| price_transition = price_transition or slot.planner_price_transition;
    defer {
        for (live) |slot| slot.planner_price_transition = false;
    }
    var emitted: [group_cost_mod.MAX_GROUP_ROWS]u32 = undefined;
    var published: [group_cost_mod.MAX_GROUP_ROWS]bool = undefined;
    var gaps: [group_cost_mod.MAX_GROUP_ROWS]f32 = undefined;
    for (live, states, 0..) |slot, st, i| {
        emitted[i] = st.retained;
        published[i] = slot.state == .decoding and !slot.cancelled.load(.acquire);
        gaps[i] = slot.mtp_publish_gap_ms;
    }
    if (!groupCostSampleComplete(expected, emitted[0..live.len], published[0..live.len])) return;
    if (price_transition) return;
    var key = group_cost_mod.GroupShape{
        .n = @intCast(live.len),
        .kv_format = cacheCostFormat(live[0].cache.config),
        .head_format = if (xfm.qwen4Trunk()) |t| if (t.qwen4_mtp) |head| cacheCostFormat(head.cache.config) else 0 else 0,
        .attention = if (xfm.qwen4Trunk()) |t| t.cost_attention else 0,
        .moe = if (xfm.qwen4Trunk()) |t| t.cost_moe else 0,
        .head_calls = @intCast(@min(std.math.maxInt(u16), qmv_rows.mtp_head_row_dispatches -% head_calls_before)),
    };
    for (live, states, 0..) |slot, st, i| {
        const chain = st.chain;
        if (chain.m > round_cost_mod.MAX_WIDTH or chain.n_drafted > round_cost_mod.MAX_WIDTH or chain.head_input_rows > round_cost_mod.MAX_WIDTH + 2) return;
        const mode = group_cost_mod.GroupShape.samplingMode(slot.legacy_gen.?.mtp.?.canRerankDrafts(), chain.conf_arrs != null, chain.q_probs != null, slot.legacy_gen.?.sampling.temperature > 0.01);
        key.rows[i] = group_cost_mod.GroupShape.row(@intCast(chain.m), @intCast(chain.n_drafted), @intCast(@min(chain.plan.m_lo, chain.n_drafted)), @intCast(xfm.round_cost.bucketOf(@intCast(st.moe_seq_offset_snap))), @intCast(xfm.round_cost.bucketOf(@intCast(chain.off0))), @intCast(chain.head_input_rows), mode);
        key.padding += @intCast(st.verify_len - (1 + chain.m));
    }
    const ms = @as(f32, @floatFromInt(tick_sw.read())) / @as(f32, std.time.ns_per_ms);
    _ = xfm.mtp_group_cost.observeShape(key, ms, emitted[0..live.len], gaps[0..live.len], .round);
}

/// One verify forward for a whole group: ragged rows ride the row-axis form, equal-lane
/// trunks the merged padded forward, one row a solo round.
/// Finish one row's round and publish it; a failed finish errors the slot instead.
fn mtpFinishPublish(sch: *Scheduler, slot: *Slot, gen: *Generator, st: *Generator.MtpRoundState) void {
    const r = gen.mtpRoundFinish(slot.allocator, st) catch |e| {
        slot.markError(@errorName(e));
        return;
    };
    publishMtpResult(sch, slot, gen, r);
}

fn publishMtpResult(sch: *Scheduler, slot: *Slot, gen: *Generator, result: ?Generator.DrafterStepResult) void {
    const r = result orelse {
        finishSlot(sch, slot, gen.finish_reason);
        return;
    };
    defer slot.allocator.free(r.tokens);
    publishSpeculativeBlock(sch, slot, gen, r.tokens);
}

// ── tests ──

test "a cancel mid batched tick keeps generated_ids level with the KV rows" {
    // Bar: kv rows == full_prompt + generated_ids even when the slot cancels mid tick.
    const prompt_len: usize = 7;
    var kv_rows = prompt_len;
    var ids: usize = 0;
    for ([_]bool{ false, false, false, true }) |cancelled| {
        kv_rows += 1;
        const act = batchedTickAction(cancelled);
        if (act.record) ids += 1;
        if (cancelled) try testing.expect(!act.publish);
    }
    try testing.expectEqual(kv_rows, prompt_len + ids);
}

test "mtpSubGroupSize fills the verify lane's row budget" {
    // 7 rows off-NAX: pairs at depth 2, triples at depth 1, four as 2+2, one rounds solo.
    try testing.expectEqual(@as(usize, 1), mtpSubGroupSize(1, 7));
    try testing.expectEqual(@as(usize, 2), mtpSubGroupSize(2, 7));
    try testing.expectEqual(@as(usize, 3), mtpSubGroupSize(3, 7));
    try testing.expectEqual(@as(usize, 2), mtpSubGroupSize(4, 7));
    try testing.expectEqual(@as(usize, 3), mtpSubGroupSize(5, 7));
    try testing.expectEqual(@as(usize, 4), mtpSubGroupSize(9, 16));
}

test "mtpSubGroupPlan: four or more lanes share one 16-row verify on the M4 shader tile" {
    // No tile: the split-K budget, 2+2 at four lanes, plain tick from four.
    try testing.expectEqual(SubGroup{ .size = 2, .row_cap = 7 }, mtpSubGroupPlan(4, .off));
    try testing.expectEqual(@as(usize, 4), mtpCrowdThresholdForTile(.off));
    // M5 tile: unchanged.
    try testing.expectEqual(SubGroup{ .size = 4, .row_cap = 16 }, mtpSubGroupPlan(9, .nax));
    try testing.expectEqual(SubGroup{ .size = 2, .row_cap = 16 }, mtpSubGroupPlan(2, .nax));
    try testing.expectEqual(@as(usize, 5), mtpCrowdThresholdForTile(.nax));
    // Shader tile: a 16-row tile costs more than two lanes are worth, so 2..3 lanes stay narrow.
    try testing.expectEqual(SubGroup{ .size = 2, .row_cap = 7 }, mtpSubGroupPlan(2, .shader));
    try testing.expectEqual(SubGroup{ .size = 3, .row_cap = 7 }, mtpSubGroupPlan(3, .shader));
    try testing.expectEqual(SubGroup{ .size = 4, .row_cap = 16, .fill = true }, mtpSubGroupPlan(4, .shader));
    try testing.expectEqual(SubGroup{ .size = 8, .row_cap = 16, .fill = true }, mtpSubGroupPlan(11, .shader));
    try testing.expectEqual(@as(usize, 9), mtpCrowdThresholdForTile(.shader));
    // Depth per lane: 16 rows over four lanes is [t1 + 3 drafts] each.
    try testing.expectEqual(@as(u32, 3), mtpSubGroupPlan(4, .shader).depthCap());
    try testing.expectEqual(@as(u32, 1), mtpSubGroupPlan(8, .shader).depthCap());
    try testing.expectEqual(@as(u32, 2), mtpSubGroupPlan(2, .shader).depthCap());
    // The depth each sub-group gets: floor(rows / size) - 1.
    try testing.expectEqual(@as(u32, 2), 7 / @as(u32, 2) - 1);
    try testing.expectEqual(@as(u32, 1), 7 / @as(u32, 3) - 1);
}

test "retained position: a padded row can never take the full-accept arm" {
    // Unpadded full accept commits 1+m and keeps the trunk state the forward left.
    try testing.expect(generate_mod.Generator.mtpFullAccept(2, 2, 3));
    try testing.expectEqual(@as(u32, 3), generate_mod.Generator.mtpRetained(2));
    // Same accept count, row padded to a neighbour's width: rollback arm.
    try testing.expect(!generate_mod.Generator.mtpFullAccept(2, 2, 5));
    try testing.expectEqual(@as(u32, 3), generate_mod.Generator.mtpRetained(2));
    // Partial and zero accepts are rollback at any width.
    try testing.expect(!generate_mod.Generator.mtpFullAccept(1, 2, 3));
    try testing.expect(!generate_mod.Generator.mtpFullAccept(0, 2, 3));
    try testing.expectEqual(@as(u32, 1), generate_mod.Generator.mtpRetained(0));
}

test "merged verify: an uncertified [N,S] shape is declined by name, never dispatched" {
    try testing.expect(mergedVerifyDeclineReason(true, 5) != null);
    try testing.expect(mergedVerifyDeclineReason(true, 2) != null);
    // Width 1 is a plain batched decode row, not a verify.
    try testing.expect(mergedVerifyDeclineReason(true, 1) == null);
    // Other GDN trunks keep the padded merged forward they had.
    try testing.expect(mergedVerifyDeclineReason(false, 5) == null);
}

test "group verify: ragged widths keep ONE forward, never a solo cohort per width" {
    // The row-axis form runs each row at its own width, so a mixed group is never split.
    try testing.expectEqual(VerifyShape.row_axis, verifyShapeFor(4, true, 9));
    try testing.expectEqual(VerifyShape.row_axis, verifyShapeFor(2, true, 2));
    try testing.expectEqual(VerifyShape.merged, verifyShapeFor(4, false, 9));
    try testing.expectEqual(VerifyShape.merged, verifyShapeFor(2, false, 3));
    // One row is a solo round on every trunk.
    try testing.expectEqual(VerifyShape.solo, verifyShapeFor(1, true, 5));
    try testing.expectEqual(VerifyShape.solo, verifyShapeFor(1, false, 5));
}

test "group cost geometry rejects partial rounds and keeps complete cache formats" {
    try testing.expect(groupCostSampleComplete(2, &.{ 1, 3 }, &.{ true, true }));
    try testing.expect(!groupCostSampleComplete(3, &.{ 1, 3 }, &.{ true, true }));
    try testing.expect(!groupCostSampleComplete(2, &.{ 1, 0 }, &.{ true, true }));
    try testing.expect(!groupCostSampleComplete(2, &.{ 1, 3 }, &.{ true, false }));
    const a = transformer_mod.KVQuantConfig.affine(8);
    var b = a;
    b.group_size = 128;
    try testing.expect(cacheCostFormat(a) != cacheCostFormat(b));
    try testing.expect(cacheCostFormat(a) != cacheCostFormat(transformer_mod.KVQuantConfig.affine(4)));
}

test "scheduler prices each shared execution once and preserves row sampling geometry" {
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.round_cost = .{ .layout = .long };
    xfm.mtp_group_cost = .{};
    var model: LoadedModel = undefined;
    model.transformer = &xfm;
    var a: Slot = undefined;
    var b: Slot = undefined;
    a.model = &model;
    b.model = &model;
    const slots = [_]*Slot{ &a, &b };
    var geometry = group_cost_mod.GroupShape{ .n = 2, .kv_format = 8, .head_format = 8 };
    geometry.rows[0] = group_cost_mod.GroupShape.row(1, 1, 1, 2, 1, 3, 1);
    geometry.rows[1] = group_cost_mod.GroupShape.row(1, 1, 1, 5, 2, 2, 9);
    var prices = PlannerPrices{ .slots = &slots, .geometry = geometry };
    const speculative = prices.keyFor(&slots, &.{ 2, 4 });
    const swapped = prices.keyFor(&.{ &b, &a }, &.{ 4, 2 });
    try testing.expectEqual(speculative, swapped);
    for (0..3) |_| _ = xfm.mtp_group_cost.observeShape(speculative, 30, &.{ 2, 4 }, &.{ 30, 30 }, .round);
    try testing.expectEqual(@as(f32, 30), prices.price(&.{ 2, 4 }).?.ms);
    const plain_a = prices.keyFor(&.{&a}, &.{0});
    const spec_b = prices.keyFor(&.{&b}, &.{4});
    for (0..3) |_| {
        _ = xfm.mtp_group_cost.observeShape(plain_a, 10, &.{1}, &.{10}, .round);
        _ = xfm.mtp_group_cost.observeShape(spec_b, 25, &.{4}, &.{25}, .round);
    }
    try testing.expectEqual(@as(f32, 35), prices.price(&.{ 0, 4 }).?.ms);
    try testing.expect(prices.price(&.{ 4, 0 }) == null);
    const plain_b = prices.keyFor(&.{&b}, &.{0});
    try testing.expectEqual(@as(u8, 8), group_cost_mod.GroupShape.part(plain_b.rows[0], 48));
    try testing.expectEqual(@as(u8, 5), group_cost_mod.GroupShape.part(plain_b.rows[0], 24));
    try testing.expectEqual(@as(u8, 0), group_cost_mod.GroupShape.part(plain_b.rows[0], 40));
    try testing.expectEqual(@as(u64, 0), plain_b.head_format);
}

test "planner pools recurring head histories without mixing the cold first history" {
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.round_cost = .{ .layout = .long };
    xfm.mtp_group_cost = .{};
    var model: LoadedModel = undefined;
    model.transformer = &xfm;
    var slots_storage: [4]Slot = undefined;
    var slots: [4]*Slot = undefined;
    for (&slots_storage, 0..) |*slot, i| {
        slot.model = &model;
        slots[i] = slot;
    }
    var geometry = group_cost_mod.GroupShape{ .n = 4, .kv_format = 8, .head_format = 8 };
    @memset(geometry.rows[0..4], group_cost_mod.GroupShape.row(2, 2, 2, 2, 1, 4, 1));
    var prices = PlannerPrices{ .slots = &slots, .geometry = geometry };

    var cold = geometry;
    cold.rows[0] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 1, 1, 1);
    _ = xfm.mtp_group_cost.observeShape(cold, 180, &.{ 2, 2, 2, 2 }, &.{ 180, 180, 180, 180 }, .round);
    var h2 = geometry;
    h2.rows[0] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 1, 2, 1);
    _ = xfm.mtp_group_cost.observeShape(h2, 74, &.{ 2, 3, 3, 2 }, &.{ 74, 74, 74, 74 }, .round);
    const probe_seed = prices.sample(&slots, &.{ 2, 2, 2, 2 }, false).?;
    try testing.expectEqual(@as(u32, 1), probe_seed.samples);
    try testing.expectEqual(@as(f32, 74), probe_seed.ms);
    var h3 = geometry;
    h3.rows[1] = group_cost_mod.GroupShape.row(2, 2, 2, 1, 1, 3, 1);
    _ = xfm.mtp_group_cost.observeShape(h3, 76, &.{ 3, 2, 3, 2 }, &.{ 76, 76, 76, 76 }, .round);
    var other_head_kv = geometry;
    other_head_kv.rows[1] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 2, 3, 1);
    _ = xfm.mtp_group_cost.observeShape(other_head_kv, 20, &.{ 3, 3, 3, 3 }, &.{ 20, 20, 20, 20 }, .round);
    var other_mode = geometry;
    other_mode.rows[1] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 1, 3, 9);
    _ = xfm.mtp_group_cost.observeShape(other_mode, 20, &.{ 3, 3, 3, 3 }, &.{ 20, 20, 20, 20 }, .round);
    try testing.expect(prices.price(&.{ 2, 2, 2, 2 }) == null);

    var h4 = geometry;
    h4.rows[2] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 0, 2, 1);
    _ = xfm.mtp_group_cost.observeShape(h4, 78, &.{ 3, 3, 2, 2 }, &.{ 78, 78, 78, 78 }, .round);
    const pooled = prices.price(&.{ 2, 2, 2, 2 }).?;
    try testing.expectEqual(@as(u32, 3), pooled.samples);
    try testing.expect(pooled.ms >= 74 and pooled.ms <= 78);
    try testing.expect(pooled.variance > 0);
    try testing.expect(pooled.max_gap_ms >= 78);

    const plain = prices.keyFor(&slots, &.{ 0, 0, 0, 0 });
    for (0..3) |_| _ = xfm.mtp_group_cost.observeShape(plain, 37, &.{ 1, 1, 1, 1 }, &.{ 37, 37, 37, 37 }, .round);
    var histories: [4]Planner.History = @splat(.{});
    var rows: [4]Planner.Row = undefined;
    for (&histories, 0..) |*history, i| {
        for (0..3) |_| history.observe(2, 2);
        rows[i] = .{ .history = history };
    }
    const decision = Planner.choose(&rows, &prices);
    try testing.expectEqualSlices(u8, &.{ 2, 2, 2, 2 }, decision.widths[0..4]);
}

test "mean cost pooling retains differences between histories and both worst-case bounds" {
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.round_cost = .{ .layout = .long };
    xfm.mtp_group_cost = .{};
    var model: LoadedModel = undefined;
    model.transformer = &xfm;
    var storage: [2]Slot = undefined;
    var slots: [2]*Slot = undefined;
    for (&storage, 0..) |*slot, i| {
        slot.model = &model;
        slots[i] = slot;
    }
    var geometry = group_cost_mod.GroupShape{ .n = 2 };
    @memset(geometry.rows[0..2], group_cost_mod.GroupShape.row(2, 2, 2, 4, 4, 4, 1));
    var prices = PlannerPrices{ .slots = &slots, .geometry = geometry };
    var a = geometry;
    var b = geometry;
    a.rows[0] = group_cost_mod.GroupShape.row(2, 2, 2, 4, 4, 2, 1);
    b.rows[0] = group_cost_mod.GroupShape.row(2, 2, 2, 4, 4, 3, 1);
    for (0..20) |_| {
        _ = xfm.mtp_group_cost.observeShape(a, 64, &.{ 3, 3 }, &.{ 64, 64 }, .round);
        _ = xfm.mtp_group_cost.observeShape(b, 80, &.{ 3, 3 }, &.{ 80, 80 }, .round);
    }
    const pooled = prices.price(&.{ 2, 2 }).?;
    try testing.expectApproxEqAbs(@as(f32, 88), pooled.mean_upper_ms.?, 0.001);
    xfm.round_cost = .{ .layout = .long };
    xfm.mtp_group_cost = .{};
    a = geometry;
    b = geometry;
    b.head_calls = 2;
    xfm.mtp_group_cost.shapes[0] = .{ .shape = a, .cell = .{ .ms = 60, .n = 100 }, .variance = 225, .mean_weight_sq = 1.0 / 19.0 };
    xfm.mtp_group_cost.shapes[1] = .{ .shape = b, .cell = .{ .ms = 75, .n = 3 }, .variance = 9, .mean_weight_sq = 1.0 / 3.0 };
    const exact = prices.price(&.{ 2, 2 }).?;
    try testing.expectApproxEqAbs(@as(f32, 90), exact.ms + 2 * @sqrt(exact.variance), 0.001);
    try testing.expectApproxEqAbs(@as(f32, 75) + 2 * @sqrt(@as(f32, 3)), exact.mean_upper_ms.?, 0.001);
    a = prices.keyFor(&slots, &.{ 0, 0 });
    b = a;
    b.head_calls = 2;
    xfm.mtp_group_cost.shapes[0] = .{ .shape = a, .cell = .{ .ms = 40, .n = 100 }, .variance = 25, .mean_weight_sq = 1.0 / 19.0 };
    xfm.mtp_group_cost.shapes[1] = .{ .shape = b, .cell = .{ .ms = 35, .n = 3 }, .variance = 1, .mean_weight_sq = 1.0 / 3.0 };
    const plain = prices.price(&.{ 0, 0 }).?;
    try testing.expectApproxEqAbs(@as(f32, 35) - 2 * @sqrt(@as(f32, 1.0 / 3.0)), plain.mean_lower_ms.?, 0.001);
}

test "cold grouped depths one and two use a bounded two-tick bootstrap" {
    var prices: PlannerPrices = undefined;
    prices.slots = &.{};
    const plain = Planner.Price{ .ms = 40, .variance = 4, .samples = 10 };
    try testing.expectEqual(@as(f32, 88), prices.probeFallbackEstimate(&.{ 1, 1, 1, 1 }, plain));
    try testing.expectEqual(@as(f32, 88), prices.probeFallbackEstimate(&.{ 2, 2, 2, 2 }, plain));
    try testing.expectEqual(@as(f32, 330), prices.probeFallbackEstimate(&.{ 4, 4, 4, 4 }, plain));
    const lower = Planner.Price{ .ms = 60, .variance = 100, .samples = 2, .max_gap_ms = 170 };
    try testing.expectEqual(@as(f32, 90), PlannerPrices.scaledProbeEstimate(lower, 2, 1));
}

test "transition pricing skips the first realized round after prime or width change" {
    try testing.expect(plannerPriceTransition(255, 1, false));
    try testing.expect(plannerPriceTransition(1, 1, true));
    try testing.expect(!plannerPriceTransition(1, 1, false));
    try testing.expect(plannerPriceTransition(1, 2, false));
    try testing.expect(!plannerPriceTransition(2, 0, true));
}
