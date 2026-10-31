//! DeepSeek-V4.1 DSpark rounds for several slots of one model at once: each slot drafts (its context's continuation,
//! or the stages, every slot's block in one pass), the group's verify rows are priced together
//! (`dflash_policy.chooseGroupRows`), one batched verify runs them, and each slot accepts under its own sampling and
//! keeps its accepted prefix. A slot alone ticks its solo round (`nextDspark`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const dsv41_mod = @import("../../../plugins/deepseek_v41/deepseek_v41.zig");
const dflash_policy = @import("dflash/policy.zig");
const intree = @import("../plugin/intree.zig");
const modules = @import("../modules/call.zig");
const dspark = @import("dspark.zig");
const Generator = @import("../generate/generator.zig").Generator;
const DrafterStepResult = @import("drafter.zig").DrafterStepResult;
const Transformer = @import("../transformer.zig").Transformer;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Scheduler = @import("../scheduler.zig").Scheduler;
const Slot = @import("../slot.zig").Slot;
const finishSlot = @import("../finish.zig").finishSlot;
const runSingleDecodeTick = @import("../decode.zig").runSingleDecodeTick;
const loopGuardTick = @import("../loop_stop.zig").loopGuardTick;
const publishSpeculativeBlock = @import("wiring.zig").publishSpeculativeBlock;
const slotSpecMode = @import("wiring.zig").slotSpecMode;
const capAcceptedForTokenBudget = @import("verify.zig").capAcceptedForTokenBudget;
const dsparkCopyLen = @import("verify.zig").dsparkCopyLen;
const specDecodeUnsupported = @import("verify.zig").specDecodeUnsupported;
const SamplingParams = @import("../sampling/params.zig").SamplingParams;
const sampleRows = @import("../sampling/lazy.zig").sampleRows;

/// Slots one group round carries.
pub const MAX_SLOTS: usize = dflash_policy.MAX_GROUP_ROWS;

/// Whether `xfm` runs DSpark group rounds: a DeepSeek-V4.1 module with its stages that batches its states (resident,
/// unpaired). Its DSpark slots keep drafting beside company instead of yielding to the plain batch.
pub fn groupable(xfm: *const Transformer) bool {
    const m = intree.dsv41(xfm) orelse return false;
    return m.dspark != null and m.pair == null and m.streamed == null;
}

/// Whether `slot` ticks DSpark rounds on a groupable model.
pub fn slotGroupable(slot: *const Slot) bool {
    const gen = if (slot.legacy_gen) |*g| g else return false;
    if (gen.spec_disabled_runtime or slotSpecMode(slot, gen) != .dspark) return false;
    return groupable(slot.model.transformer orelse return false);
}

/// The decode tick's DSpark slots on groupable models: one group round per model, `MAX_SLOTS` slots at a time; a slot
/// alone ticks solo.
pub fn runDsparkGroups(sch: *Scheduler, slots: []*Slot) !void {
    std.sort.pdq(*Slot, slots, {}, struct {
        fn lt(_: void, a: *Slot, b: *Slot) bool {
            return @intFromPtr(a.model) < @intFromPtr(b.model);
        }
    }.lt);
    var start: usize = 0;
    while (start < slots.len) {
        var end = start + 1;
        while (end < slots.len and slots[end].model == slots[start].model) end += 1;
        var g0 = start;
        while (g0 < end) {
            const sub = slots[g0..@min(end, g0 + MAX_SLOTS)];
            if (sub.len == 1) try runSingleDecodeTick(sch, sub[0]) else try runGroupTick(sch, sub);
            g0 += sub.len;
        }
        start = end;
    }
}

fn runGroupTick(sch: *Scheduler, group: []*Slot) !void {
    var inner_err: ?anyerror = null;
    runGroupTickInner(sch, group) catch |e| {
        inner_err = e;
    };
    mlx.checkErrorDecode() catch |mlx_err| {
        log.err("[scheduler] DSpark group round aborted: MLX failure mid-generation ({s}) — failing these requests, the server keeps serving\n", .{@errorName(mlx_err)});
        for (group) |slot| slot.markError(@errorName(mlx_err));
        return;
    };
    if (inner_err) |e| return e;
}

fn runGroupTickInner(sch: *Scheduler, group: []*Slot) !void {
    var live: [MAX_SLOTS]*Slot = undefined;
    var gens: [MAX_SLOTS]*Generator = undefined;
    var allocs: [MAX_SLOTS]std.mem.Allocator = undefined;
    var n: usize = 0;
    for (group) |slot| {
        const gen = &slot.legacy_gen.?;
        if (try loopGuardTick(sch, slot, gen)) continue;
        const entry = enter(gen, slot.allocator) catch |e| {
            slot.markError(@errorName(e));
            continue;
        };
        switch (entry) {
            .ready => {
                live[n] = slot;
                gens[n] = gen;
                allocs[n] = slot.allocator;
                n += 1;
            },
            .emitted => |tok| publishSpeculativeBlock(sch, slot, gen, &.{tok}),
            .done => finishSlot(sch, slot, gen.finish_reason),
        }
    }
    if (n == 0) return;
    if (n == 1) return runSingleDecodeTick(sch, live[0]);
    var results: [MAX_SLOTS]DrafterStepResult = undefined;
    try modules.call("dspark_group_round", .{ gens[0..n], allocs[0..n], results[0..n] });
    for (live[0..n], gens[0..n], results[0..n]) |slot, gen, r| {
        defer slot.allocator.free(r.tokens);
        publishSpeculativeBlock(sch, slot, gen, r.tokens);
    }
}

const Entry = union(enum) { ready, emitted: u32, done };

/// `nextDspark`'s way in: a pipelined token still in flight is published first (the slot joins the next round), a
/// stopped request ends; else the slot stands at the spec invariant.
fn enter(gen: *Generator, allocator: std.mem.Allocator) !Entry {
    if (gen.done) return .done;
    if (specDecodeUnsupported(gen.sampling, gen.logprobs_n)) return error.SpecDecodeUnsupported;
    if (gen.has_pending_logits or gen.has_pending_token) switch (try gen.drainPipelineForSpec(allocator)) {
        .drained => |tok| return .{ .emitted = tok },
        .stopped => return .done,
        .already_clean, .stay_disabled => return error.PipelineHalfState,
    };
    if (try gen.checkStop()) return .done;
    return .ready;
}

const Kind = enum { plain, copy, stage };

/// One DSpark round for each of `gens` (slots of one DeepSeek-V4.1 model, each at the spec invariant): its drafts, the
/// verify rows the group's pricing buys, one batched verify, its accept under its own sampling, its kept prefix.
/// `out[k]` is `gens[k]`'s committed block, its tokens owned through `allocs[k]`.
pub fn moduleGroupRound(gens: []const *Generator, allocs: []const std.mem.Allocator, out: []DrafterStepResult) !void {
    const n = gens.len;
    const xfm = gens[0].xfm;
    const mdl = intree.dsv41(xfm).?;
    const gpa = allocs[0];
    var ctxs: [MAX_SLOTS]*ForwardCtx = undefined;
    for (gens, ctxs[0..n]) |g, *c| c.* = &g.ctx;
    var states: [MAX_SLOTS]*dsv41_mod.State = undefined;
    var ids: [MAX_SLOTS]u64 = undefined;
    if (!intree.dsv41States(xfm, ctxs[0..n], states[0..n], ids[0..n])) return error.Dsv41BatchUnsupported;

    // Drafts: the context's continuation where it holds one, else the stages for every slot that drafts, in one pass.
    var kinds: [MAX_SLOTS]Kind = undefined;
    var drafts: [MAX_SLOTS]dsv41_mod.DsparkDraft = undefined;
    var caps: [MAX_SLOTS]u32 = undefined;
    var suffix: [MAX_SLOTS]u32 = @splat(0);
    var made: usize = 0;
    defer for (drafts[0..made]) |*d| d.deinit(gpa);
    var staged: [MAX_SLOTS]usize = undefined;
    var n_stage: usize = 0;
    for (gens, 0..) |gen, k| {
        const t1 = gen.next_token_id;
        caps[k] = capAcceptedForTokenBudget(std.math.maxInt(u32), gen.completion_tokens, gen.max_tokens);
        const idx = try gen.mtpLookupIndex(allocs[k]);
        const found = idx.matchBest(t1, @intCast(dsv41_mod.maxVerifyDrafts(mdl)));
        const copy = dsparkCopyLen(found, gen.max_tokens -| gen.completion_tokens -| 1, mdl.ds_block);
        // A group with no row to spare verifies no stage draft: it drafts none.
        const stage = copy == 0 and n < dflash_policy.MAX_GROUP_ROWS and if (gen.dspark_pol) |*pol| pol.drafting.shouldDraft() else true;
        kinds[k] = if (copy > 0) .copy else if (stage) .stage else .plain;
        if (stage) {
            staged[n_stage] = k;
            n_stage += 1;
        }
        const ids_k = try gpa.alloc(u32, copy + 1);
        ids_k[0] = t1;
        if (copy > 0) {
            @memcpy(ids_k[1..], found.?.draft[0..copy]);
            suffix[k] = found.?.suffix;
        }
        drafts[k] = .{ .ids = ids_k, .len = copy, .logits = &.{}, .confidence = &.{} };
        made += 1;
    }
    var draft_ms: f32 = 0;
    if (n_stage > 0) {
        const draft_watch = io_util.Stopwatch.init(gens[0].timer.io);
        defer draft_ms = msOf(draft_watch.read());
        var sts: [MAX_SLOTS]*dsv41_mod.State = undefined;
        var t1s: [MAX_SLOTS]u32 = undefined;
        for (staged[0..n_stage], 0..) |k, i| {
            sts[i] = states[k];
            t1s[i] = gens[k].next_token_id;
        }
        var blocks: [MAX_SLOTS]dsv41_mod.DsparkDraft = undefined;
        try dsv41_mod.dsparkDraftBatched(mdl, gpa, sts[0..n_stage], t1s[0..n_stage], false, blocks[0..n_stage]);
        for (staged[0..n_stage], blocks[0..n_stage]) |k, b| {
            drafts[k].deinit(gpa);
            drafts[k] = b;
        }
    }

    // The rows the group buys: each slot's drafts priced by its acceptance (the confidence head's through the slot's
    // calibration, a copy's survival), never past its token budget.
    var probs: [MAX_SLOTS][dflash_policy.MAX_DRAFTS]f32 = undefined;
    var p: [MAX_SLOTS][]const f32 = undefined;
    for (gens, drafts[0..n], kinds[0..n], 0..) |gen, d, kind, k| {
        const avail = @min(d.len, caps[k], dflash_policy.MAX_DRAFTS);
        for (probs[k][0..avail], 0..) |*o, j| o.* = switch (kind) {
            .stage => if (gen.dspark_pol) |*pol| pol.calib.probability(dspark.sigmoid(d.confidence[j])) else 1,
            .copy => if (gen.dspark_pol) |*pol| pol.copy.probability(suffix[k]) else 1,
            .plain => unreachable,
        };
        p[k] = probs[k][0..avail];
    }
    const cost = &xfm.dspark_group_cost;
    const costs = cost.costs(n);
    var rows_k: [MAX_SLOTS]u32 = undefined;
    _ = dflash_policy.chooseGroupRows(p[0..n], &costs, rows_k[0..n]);
    var chunks: [MAX_SLOTS][]const u32 = undefined;
    var rows: usize = 0;
    for (drafts[0..n], rows_k[0..n], chunks[0..n]) |*d, b, *ch| {
        d.len = b;
        ch.* = d.ids[0 .. b + 1];
        rows += b + 1;
    }

    // One verify: a round that verifies no draft is the group's plain step, with nothing to roll back.
    const verify_watch = io_util.Stopwatch.init(gens[0].timer.io);
    const drafted = rows > n;
    const logits = try dsv41_mod.extendBatched(mdl, gpa, states[0..n], ids[0..n], chunks[0..n], drafted);
    defer _ = mlx.mlx_array_free(logits);
    // Each slot's accept on its own rows. A slot's `t1` alone draws its next token with the others' in one batch, as the
    // plain batched tick does.
    var greedy = false;
    for (gens, rows_k[0..n]) |g, b| greedy = greedy or (b > 0 and !g.dspark_stochastic);
    const best: []const i32 = if (greedy) try argmaxRows(gpa, logits, mdl.s) else &.{};
    defer gpa.free(best);
    var accepted: [MAX_SLOTS]u32 = @splat(0);
    var next: [MAX_SLOTS]u32 = undefined;
    var draw_rows: [MAX_SLOTS]mlx.mlx_array = undefined;
    var draw_params: [MAX_SLOTS]SamplingParams = undefined;
    var draw_slot: [MAX_SLOTS]usize = undefined;
    var n_draw: usize = 0;
    defer for (draw_rows[0..n_draw]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    var off: usize = 0;
    for (gens, chunks[0..n], 0..) |gen, ch, k| {
        defer off += ch.len;
        if (ch.len == 1) {
            const row = try rowsOf(logits, off, 1, mdl.vocab, mdl.s);
            defer _ = mlx.mlx_array_free(row);
            var r = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(r);
            try mlx.check(mlx.mlx_reshape(&r, row, &[_]c_int{ 1, 1, @intCast(mdl.vocab) }, 3, mdl.s));
            draw_rows[n_draw] = r;
            draw_params[n_draw] = gen.sampling;
            draw_slot[n_draw] = k;
            n_draw += 1;
        } else if (gen.dspark_stochastic) {
            const rows_l = try rowsOf(logits, off, ch.len, mdl.vocab, mdl.s);
            defer _ = mlx.mlx_array_free(rows_l);
            const v = try dspark.stochasticVerdict(gen, rows_l, ch, mdl.vocab, caps[k]);
            accepted[k] = v.accepted;
            next[k] = v.next_token;
        } else {
            var a: u32 = 0;
            while (a + 1 < ch.len and @as(u32, @intCast(best[off + a])) == ch[a + 1]) a += 1;
            accepted[k] = @min(a, caps[k]);
            next[k] = @intCast(best[off + accepted[k]]);
        }
    }
    if (n_draw > 0) {
        var drawn: [MAX_SLOTS]i32 = undefined;
        try sampleRows(drawn[0..n_draw], draw_rows[0..n_draw], draw_params[0..n_draw], mdl.s);
        for (draw_slot[0..n_draw], drawn[0..n_draw], draw_params[0..n_draw]) |k, t, sp| {
            next[k] = @intCast(t);
            gens[k].sampling.draw = sp.draw;
        }
    }
    cost.sample(n, rows, msOf(verify_watch.read()));

    // Each slot keeps its prefix, and teaches its drafting backoff what drafting cost the round (its share of the draft,
    // its rows) against what its accepted drafts saved (the group's plain step a token).
    var done: usize = 0;
    errdefer for (out[0..done], allocs[0..done]) |r, a| a.free(r.tokens);
    for (gens, states[0..n], drafts[0..n], kinds[0..n], 0..) |gen, st, d, kind, k| {
        if (drafted) try dsv41_mod.finishVerify(mdl, gpa, st, accepted[k] + 1);
        const tokens = try allocs[k].dupe(u32, d.ids[0 .. accepted[k] + 1]);
        out[k] = (try gen.commitDsparkRound(allocs[k], .{ .tokens = tokens, .accepted = accepted[k], .next_token = next[k] }, st.n)).?;
        done += 1;
        switch (kind) {
            .stage => {
                if (gen.dspark_pol) |*pol| for (d.confidence[0..d.len], 0..) |c, j| {
                    if (j > accepted[k]) break;
                    pol.calib.observe(dspark.sigmoid(c), j < accepted[k]);
                };
                gen.dspark_stats.record(@intCast(d.len), accepted[k]);
            },
            .copy => {
                gen.dspark_copies += 1;
                gen.dspark_copy_drafted += d.len;
                gen.dspark_copy_accepted += accepted[k];
                if (gen.dspark_pol) |*pol| pol.copy.observe(suffix[k], @intCast(d.len), accepted[k]);
            },
            .plain => gen.dspark_plain += 1,
        }
        gen.dspark_group_rounds += 1;
    }
    const token_ms = cost.ms(n, n) / @as(f32, @floatFromInt(n));
    for (gens, kinds[0..n], accepted[0..n], rows_k[0..n]) |gen, kind, a, b| if (kind == .stage) if (gen.dspark_pol) |*pol| {
        const spent = draft_ms / @as(f32, @floatFromInt(n_stage)) + @as(f32, @floatFromInt(b)) * cost.rowMs();
        pol.drafting.record(@as(f32, @floatFromInt(a)) * token_ms, spent);
    };
}

/// Rows [lo, lo + n) of `logits` `[R, V]` (caller frees).
fn rowsOf(logits: mlx.mlx_array, lo: usize, n: usize, vocab: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_slice(&o, logits, &[_]c_int{ @intCast(lo), 0 }, 2, &[_]c_int{ @intCast(lo + n), @intCast(vocab) }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return o;
}

/// Every row's argmax of `logits` `[R, V]` on the host (caller frees).
fn argmaxRows(gpa: std.mem.Allocator, logits: mlx.mlx_array, s: mlx.mlx_stream) ![]i32 {
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, logits, -1, false, s));
    var am32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am32);
    try mlx.check(mlx.mlx_astype(&am32, am, .int32, s));
    try mlx.check(mlx.mlx_array_eval(am32));
    const data = mlx.mlx_array_data_int32(am32) orelse return error.NoData;
    return gpa.dupe(i32, data[0..mlx.mlx_array_size(am32)]);
}

fn msOf(ns: u64) f32 {
    return @as(f32, @floatFromInt(ns)) / std.time.ns_per_ms;
}
