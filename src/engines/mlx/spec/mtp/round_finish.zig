//! Generator.mtpRoundFinish: committing an MTP round.
const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const transformer_mod = @import("../../transformer.zig");
const log = @import("../../../../sdk/util/log.zig");
const io_util = @import("../../../../sdk/util/io_util.zig");
const group_planner = @import("../mtp_group_planner.zig");
const Transformer = transformer_mod.Transformer;
const chain_mod = @import("chain.zig");
const DrafterStepResult = @import("../drafter.zig").DrafterStepResult;
const Generator = @import("../../generate/generator.zig").Generator;
const MtpPreDraft = @import("chain.zig").MtpPreDraft;
const MtpRoundState = @import("round.zig").MtpRoundState;
const SubLap = @import("trace.zig").SubLap;
const capAcceptedForTokenBudget = @import("../verify.zig").capAcceptedForTokenBudget;
const mtpAcceptRowGreedy = @import("round.zig").mtpAcceptRowGreedy;
const mtpEmaMs = @import("regime.zig").mtpEmaMs;
const mtpFullAccept = @import("round.zig").mtpFullAccept;
const mtpGreedyPrefix = @import("round.zig").mtpGreedyPrefix;
const mtpLazyPredraftAllowedFor = @import("chain.zig").mtpLazyPredraftAllowedFor;
const mtpRetained = @import("round.zig").mtpRetained;
const mtpStopPrefix = @import("round.zig").mtpStopPrefix;
const probsAllPositions = @import("../../sampling/spec_sampling.zig").probsAllPositions;
const seedKey = @import("../../sampling/lazy.zig").seedKey;
const verifyArgmax = @import("../verify.zig").verifyArgmax;


pub fn mtpRoundFinish(self: *Generator, allocator: std.mem.Allocator, st: *MtpRoundState) !?DrafterStepResult {
    var failed = false;
    defer if (!failed) {
        if (st.verify_rollback) |*saved| saved.deinit();
        st.verify_rollback = null;
        st.verify_ctx = null;
    };
    errdefer failed = true;
    const xfm = self.xfm;
    const s = xfm.s;
    const tracing = st.tracing;
    var ph = st.ph;
    var round_watch = st.round_watch;
    const chain = &st.chain;
    const plan = chain.plan;
    const m_lo: u32 = plan.m_lo;
    const m_max: u32 = plan.m_hi;
    const m: u32 = chain.m;
    const observed_limit = @min(m, self.max_tokens -| self.completion_tokens -| 1);
    const t1: u32 = chain.t1;
    const mtp_off0: usize = chain.off0;
    const drafts = chain.drafts;
    const draft_arrs = chain.draft_arrs;
    const q_probs = chain.q_probs;
    const kv_step_snap = st.kv_step_snap;
    const moe_seq_offset_snap = st.moe_seq_offset_snap;
    var verify_logits = st.verify_logits;
    st.verify_logits = .{ .ctx = null };
    errdefer _ = mlx.mlx_array_free(verify_logits);
    var new_hidden = st.new_hidden;
    st.new_hidden = .{ .ctx = null };
    errdefer _ = mlx.mlx_array_free(new_hidden);
    const verify_hidden_all = st.verify_hidden_all;
    // A GPU-arm verify still owes its PLE history (`mtpRoundVerify`); a round that
    // exits before the draft read drops it with the verify.
    defer xfm.discardDeferredPle(&self.ctx);
    // Always free the transient capture buffers before returning, however
    // we exit this round (full accept, partial accept, or error).
    defer if (self.ctx.ssm_entries) |entries| {
        for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
    };
    // A lookup round is no MTP width sample, nor is the MTP round after one:
    // its first head step consumes a wider history stash than MTP makes.
    const times_round = !chain.lookup and !self.mtp_after_lookup;
    self.mtp_after_lookup = chain.lookup;
    if (chain.lookup) {
        self.mtp_lookup_rounds += 1;
        self.mtp_lookup_drafted += m;
    } else {
        self.mtp_attempted += 1;
        self.mtp_drafted_tokens += m;
    }

    // ── Phase 4: decide longest accepted prefix ──
    // Stochastic path is fully BATCHED: accept probabilities for every
    // draft AND a candidate correction token for every possible reject
    // position are built lazily (draft ids stay lazy arrays — never read
    // on the CPU inside a graph-building loop), then ONE async eval
    // realizes the whole round. The old per-draft probAt()/sampleResidual()
    // calls cost one GPU round-trip sync EACH — 3-5 syncs per round that
    // stalled the pipeline for milliseconds while the GPU sat idle.
    const stochastic = self.sampling.temperature > 0.01 and !self.mtp_serial_accept;
    const vl_shape = mlx.getShape(verify_logits);

    var per_pos_probs: ?[]mlx.mlx_array = null;
    var per_pos_probs_n: usize = 0;
    defer if (per_pos_probs) |slots| {
        for (slots[0..per_pos_probs_n]) |arr| _ = mlx.mlx_array_free(arr);
        allocator.free(slots);
    };
    var accept_p_vec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(accept_p_vec);
    var accept_q_vec: mlx.mlx_array = .{ .ctx = null };
    defer if (accept_q_vec.ctx != null) {
        _ = mlx.mlx_array_free(accept_q_vec);
    };
    var accept_defer_vec: mlx.mlx_array = .{ .ctx = null };
    defer if (accept_defer_vec.ctx != null) {
        _ = mlx.mlx_array_free(accept_defer_vec);
    };
    var corr_samples: ?[]mlx.mlx_array = null;
    var corr_samples_n: usize = 0;
    defer if (corr_samples) |slots| {
        for (slots[0..corr_samples_n]) |arr| _ = mlx.mlx_array_free(arr);
        allocator.free(slots);
    };
    // Batched-corrections path: ONE [1+m] pre-sampled correction array
    // instead of the per-position corr_samples slots.
    var corr_batch: mlx.mlx_array = .{ .ctx = null };
    defer if (corr_batch.ctx != null) {
        _ = mlx.mlx_array_free(corr_batch);
    };

    // The group already built and dispatched this row's Phase 4 with the rest
    // of the round (`mtpGroupSampledAccept`); nothing here but the handover.
    const prepared = stochastic and st.accept_p.ctx != null;
    if (prepared) {
        _ = mlx.mlx_array_free(accept_p_vec);
        accept_p_vec = st.accept_p;
        accept_q_vec = st.accept_q;
        accept_defer_vec = st.accept_defer;
        corr_batch = st.corr;
        st.accept_p = .{ .ctx = null };
        st.accept_q = .{ .ctx = null };
        st.accept_defer = .{ .ctx = null };
        st.corr = .{ .ctx = null };
    } else if (stochastic) {
        // A group-padded row filters only the 1+m positions it reads.
        var live_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(live_logits);
        try mlx.check(mlx.mlx_slice(&live_logits, verify_logits, &.{ 0, 0, 0 }, 3, &.{ vl_shape[0], @intCast(1 + m), vl_shape[2] }, 3, &.{ 1, 1, 1 }, 3, s));
        const probs_all = try probsAllPositions(live_logits, self.sampling, s);
        defer _ = mlx.mlx_array_free(probs_all);
        const bg = try self.acceptGraphFor(chain.lookup)(
            probs_all,
            draft_arrs[0..m],
            if (q_probs) |qs| qs[0..m] else null,
            m,
            self.mtp_accept_param,
            self.mtpSamplingDraw(self.sampling),
            s,
        );
        corr_batch = bg.corr_samples;
        _ = mlx.mlx_array_free(accept_p_vec);
        accept_p_vec = bg.accept_p;
        accept_q_vec = bg.accept_q;
        accept_defer_vec = bg.accept_defer;
    } else if (stochastic) {
        const slice_strides = [_]c_int{ 1, 1, 1 };
        // Filtered + softmaxed target probs for ALL 1+m positions in one
        // batched kernel set, then per-position slice VIEWS (no copies).
        const probs_all = try probsAllPositions(verify_logits, self.sampling, s);
        defer _ = mlx.mlx_array_free(probs_all);
        const slots = try allocator.alloc(mlx.mlx_array, 1 + m);
        per_pos_probs = slots;
        for (slots, 0..) |*slot, idx| {
            slot.* = mlx.mlx_array_new();
            per_pos_probs_n = idx + 1;
            const start = [_]c_int{ 0, @intCast(idx), 0 };
            const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
            try mlx.check(mlx.mlx_slice(slot, probs_all, &start, 3, &stop, 3, &slice_strides, 3, s));
        }

        // accept_p_vec[k] = target_p[k][draft_k], gathered with the LAZY
        // draft id array → [m] f32 after one eval.
        {
            const taken = try allocator.alloc(mlx.mlx_array, m);
            var taken_n: usize = 0;
            defer {
                for (taken[0..taken_n]) |arr| _ = mlx.mlx_array_free(arr);
                allocator.free(taken);
            }
            for (0..m) |k| {
                taken[k] = mlx.mlx_array_new();
                taken_n = k + 1;
                try mlx.check(mlx.mlx_take_axis(&taken[k], slots[k], draft_arrs[k], -1, s));
            }
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (taken) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
            var cat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cat);
            try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
            try mlx.check(mlx.mlx_astype(&accept_p_vec, cat, .float32, s));
        }

        // accept_q_vec[k] = q_k[draft_k] — the proposal's own density at
        // the sampled draft, for the Leviathan ratio (sharp drafts only).
        if (q_probs) |qslots| {
            const taken = try allocator.alloc(mlx.mlx_array, m);
            var taken_n: usize = 0;
            defer {
                for (taken[0..taken_n]) |arr| _ = mlx.mlx_array_free(arr);
                allocator.free(taken);
            }
            for (0..m) |k| {
                taken[k] = mlx.mlx_array_new();
                taken_n = k + 1;
                try mlx.check(mlx.mlx_take_axis(&taken[k], qslots[k], draft_arrs[k], -1, s));
            }
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (taken) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
            var cat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cat);
            try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
            accept_q_vec = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&accept_q_vec, cat, .float32, s));
        }

        // Candidate correction for every possible reject position a<m
        // (residual sample) plus the full-accept bonus at a=m. Only the
        // one at the realized `accepted` is read; the rest are a few
        // vocab-length ops of throwaway GPU work — far cheaper than a
        // second synchronous softmax+categorical round-trip.
        var indices = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(indices);
        try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vl_shape[2])), 1, .int32, s));

        const corrs = try allocator.alloc(mlx.mlx_array, 1 + m);
        corr_samples = corrs;
        for (corrs, 0..) |*slot, a| {
            slot.* = mlx.mlx_array_new();
            corr_samples_n = a + 1;
            if (a < m) {
                // residual = max(target_p − proposal, 0): the proposal is
                // the FULL sharpened q distribution under sharp drafts
                // (exact Leviathan residual), the one-hot of the lazy
                // draft id under greedy drafts (arange == id).
                var diff = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(diff);
                if (q_probs) |qslots| {
                    try mlx.check(mlx.mlx_subtract(&diff, per_pos_probs.?[a], qslots[a], s));
                } else {
                    var onehot_b = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(onehot_b);
                    try mlx.check(mlx.mlx_equal(&onehot_b, indices, draft_arrs[a], s));
                    var onehot = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(onehot);
                    try mlx.check(mlx.mlx_astype(&onehot, onehot_b, .float32, s));
                    try mlx.check(mlx.mlx_subtract(&diff, per_pos_probs.?[a], onehot, s));
                }
                const zero = mlx.mlx_array_new_float(0.0);
                defer _ = mlx.mlx_array_free(zero);
                var residual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(residual);
                try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));
                var log_res = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(log_res);
                try mlx.check(mlx.mlx_log(&log_res, residual, s));
                const key = seedKey(self.mtpSamplingDraw(self.sampling));
                defer _ = mlx.mlx_array_free(key);
                try mlx.check(mlx.mlx_random_categorical(slot, log_res, -1, key, s));
            } else {
                var log_p = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(log_p);
                try mlx.check(mlx.mlx_log(&log_p, per_pos_probs.?[m], s));
                const key = seedKey(self.mtpSamplingDraw(self.sampling));
                defer _ = mlx.mlx_array_free(key);
                try mlx.check(mlx.mlx_random_categorical(slot, log_p, -1, key, s));
            }
        }
    }

    const group_argmax = st.verify_argmax.present();
    var verify_argmax = st.verify_argmax;
    st.verify_argmax = .{};
    defer verify_argmax.deinit();
    if (!group_argmax and !stochastic) {
        verify_argmax = if (self.mtp_serial_accept and self.sampling.temperature > 0.01)
            try self.verifySerialSamples(verify_logits, st.verify_len)
        else
            try verifyArgmax(verify_logits, self.sampling.suppress_mask, s);
    }
    _ = mlx.mlx_array_free(verify_logits);
    verify_logits = .{ .ctx = null };
    if (tracing) {
        self.mtp_trace.add(.corr, ph.read());
        ph.reset();
    }

    // ── Phase 4b: one batched async eval for the whole round ──
    // A prepared row rode the group's eval; everything here is already on it.
    const dispatch_lap = SubLap.start(tracing, self.timer.io);
    if (!prepared) {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        for (draft_arrs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
        if (stochastic) {
            _ = mlx.mlx_vector_array_append_value(eval_vec, accept_p_vec);
            if (accept_q_vec.ctx != null) _ = mlx.mlx_vector_array_append_value(eval_vec, accept_q_vec);
            if (accept_defer_vec.ctx != null) _ = mlx.mlx_vector_array_append_value(eval_vec, accept_defer_vec);
            if (corr_batch.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(eval_vec, corr_batch);
            } else {
                for (corr_samples.?) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
            }
        } else {
            _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax.lazy());
        }
        _ = mlx.mlx_vector_array_append_value(eval_vec, new_hidden);
        _ = mlx.mlx_vector_array_append_value(eval_vec, verify_hidden_all);
        try mlx.check(mlx.mlx_async_eval(eval_vec));
    }
    dispatch_lap.stop(&self.mtp_trace, .dispatch);
    // Round N+1's chain queues behind this round's verify; the host read below settles it.
    var lazy: ?MtpPreDraft = null;
    var lazy_planned = false;
    errdefer if (lazy) |*c| {
        c.deinit(allocator);
        self.mtp_cache.?.truncate(mtp_off0, s) catch {};
    };
    if (mtpLazyPredraftAllowedFor(self.mtpLazyGate(st, !stochastic, group_argmax or prepared))) {
        const lazy_lap = SubLap.start(tracing, self.timer.io);
        lazy_planned = true;
        lazy = try self.mtpLazyPreDraft(allocator, chain, verify_argmax.lazy(), verify_hidden_all);
        lazy_lap.stop(&self.mtp_trace, .chain);
        if (chain_mod.mtp_lazy_fail_after_dispatch and lazy != null) return error.MtpLazyInjected;
    }
    const wait_lap = SubLap.start(tracing, self.timer.io);
    for (draft_arrs[0..m], 0..) |arr, idx| {
        try mlx.check(mlx.mlx_array_eval(arr));
        var v: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&v, arr));
        drafts[idx] = @intCast(v);
    }
    if (Transformer.deferredPleOnGpu(&self.ctx)) {
        // The verify ids are `[t1, drafts]`, read just above: the PLE history moves here,
        // before Phase 5 rolls it back.
        const verify_ids = try allocator.alloc(u32, 1 + m);
        defer allocator.free(verify_ids);
        verify_ids[0] = t1;
        @memcpy(verify_ids[1..], drafts[0..m]);
        try xfm.settleDeferredPle(&self.ctx, verify_ids);
    }
    if (!stochastic) {
        // Separate graph branch from the draft chain — force it before
        // bulk-reading (see the v26.5.6 0%-acceptance note in nextDrafter).
        try mlx.check(mlx.mlx_array_eval(verify_argmax.lazy()));
    }
    wait_lap.stop(&self.mtp_trace, .wait);

    var accepted: u32 = 0;
    var greedy_next_pending: u32 = 0;
    var greedy_raw: u32 = 0;
    if (stochastic) {
        // Sharp drafts: full Leviathan ratio min(1, p/q) against the
        // proposal's own density. Greedy-forced drafts keep the exact
        // one-hot rule min(1, target_p).
        try mlx.check(mlx.mlx_array_eval(accept_p_vec));
        const p_data = mlx.mlx_array_data_float32(accept_p_vec) orelse {
            return error.MlxArrayDataNull;
        };
        var q_data: ?[]const f32 = null;
        if (accept_q_vec.ctx != null) {
            try mlx.check(mlx.mlx_array_eval(accept_q_vec));
            const q_ptr = mlx.mlx_array_data_float32(accept_q_vec) orelse {
                return error.MlxArrayDataNull;
            };
            q_data = q_ptr[0..m];
        }
        var defer_data: ?[]const bool = null;
        if (accept_defer_vec.ctx != null) {
            try mlx.check(mlx.mlx_array_eval(accept_defer_vec));
            const defer_ptr = mlx.mlx_array_data_bool(accept_defer_vec) orelse return error.MlxArrayDataNull;
            defer_data = defer_ptr[0..m];
        }
        accepted = self.acceptPrefixFor(chain.lookup)(p_data[0..m], q_data, defer_data, &self.prng);
        accepted = capAcceptedForTokenBudget(
            accepted,
            self.completion_tokens,
            self.max_tokens,
        );
    } else {
        const verdict = try mtpAcceptRowGreedy(verify_argmax, st.verify_len, drafts[0..m], self.completion_tokens, self.max_tokens);
        accepted = verdict.accepted;
        greedy_next_pending = verdict.next_pending;
        greedy_raw = mtpGreedyPrefix(try verify_argmax.ids(st.verify_len), drafts[0..m]);
    }

    const stop_prefix = mtpStopPrefix(drafts, accepted, if (self.mtp_planner_owned) self.eos_token_ids else &.{});
    // t1 sits at generated index G, the kept drafts after it, the pending
    // token next: the serial sampler's next draw is the one after that.
    if (self.mtp_serial_accept) self.sampling.draw = self.generated_ids.items.len + stop_prefix.accepted + 2;
    if (stop_prefix.stop) |token| log.debug("[mtp-stop] accepted EOS token={d} accepted={d}->{d}\n", .{ token, accepted, stop_prefix.accepted });
    accepted = stop_prefix.accepted;
    const next_pending: u32 = blk: {
        if (stop_prefix.stop) |stop| break :blk stop;
        if (stochastic) {
            if (corr_batch.ctx != null) {
                // Pre-sampled [1+m] batch; index at the realized accept.
                try mlx.check(mlx.mlx_array_eval(corr_batch));
                const d = mlx.mlx_array_data_int32(corr_batch) orelse {
                    return error.MlxArrayDataNull;
                };
                break :blk @intCast(d[accepted]);
            }
            // Pre-sampled in the round batch; realized already.
            const corr = corr_samples.?[accepted];
            try mlx.check(mlx.mlx_array_eval(corr));
            var v: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&v, corr));
            break :blk @intCast(v);
        } else {
            break :blk greedy_next_pending;
        }
    };

    if (tracing) {
        self.mtp_trace.add(.eval, ph.read());
        ph.reset();
    }
    log.debug("  [mtp-round] off0={d} t1={d} m={d}/{d} drafts={any} accepted={d}\n", .{ mtp_off0, t1, m, m_max, drafts[0..m], accepted });

    // ── Phase 5a: stash the committed history for a DEFERRED append ──
    // The old shape paid a second head forward (appendHistory) here every
    // round to rebuild history from true verify hiddens — then the next
    // round's first draft re-entered the head anyway. Instead, stash the
    // (tokens, hiddens) pair — tokens as [t1, drafts[0..accepted]], the
    // hiddens the SAME concat of h_prev + ORIGINAL verify hiddens the old
    // appendHistory received — and fold the append into that first draft
    // step (the i==0 merged branch above). The head cache keeps this
    // round's draft tail past mtp_off0 until the consume-time truncate;
    // nothing reads it in between. Rounds with no successor (EOS/length/
    // runtime disable) never pay for the append; deinit frees the stash.
    {
        std.debug.assert(self.mtp_hist_stash == null);
        const n_commit: usize = accepted;
        const ids_i32 = try allocator.alloc(i32, 1 + n_commit);
        defer allocator.free(ids_i32);
        ids_i32[0] = @intCast(t1);
        for (drafts[0..n_commit], 0..) |d, idx| ids_i32[1 + idx] = @intCast(d);
        const id_shape = [_]c_int{@intCast(1 + n_commit)};
        const stash_ids = mlx.mlx_array_new_data(ids_i32.ptr, &id_shape, 1, .int32);
        errdefer _ = mlx.mlx_array_free(stash_ids);
        const pad = if (self.mtpPaddedHistory()) try self.mtpPaddedStash(allocator, t1, drafts[0..m], verify_hidden_all) else null;
        errdefer if (pad) |p| {
            _ = mlx.mlx_array_free(p.ids);
            _ = mlx.mlx_array_free(p.hidden);
        };

        var hist_hidden = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(hist_hidden);
        if (n_commit == 0) {
            try mlx.check(mlx.mlx_array_set(&hist_hidden, self.last_hidden));
        } else {
            const vh_shape = mlx.getShape(verify_hidden_all);
            var vh_slice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vh_slice);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ 1, @intCast(n_commit), vh_shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&vh_slice, verify_hidden_all, &start, 3, &stop, 3, &strides, 3, s));
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, self.last_hidden);
            _ = mlx.mlx_vector_array_append_value(vec, vh_slice);
            try mlx.check(mlx.mlx_concatenate_axis(&hist_hidden, vec, 1, s));
        }
        self.mtp_hist_stash = .{
            .ids = stash_ids,
            .hidden = hist_hidden,
            .n = 1 + n_commit,
            .off0 = mtp_off0,
            .pad = pad,
        };
        for (ids_i32, 0..) |id, idx| self.mtp_hist_stash.?.host_ids[idx] = @intCast(id);
    }
    if (tracing) {
        self.mtp_trace.add(.hist, ph.read());
        ph.reset();
    }

    // ── Phase 5b: commit / rollback the trunk ──
    st.retained = mtpRetained(accepted);
    if (mtpFullAccept(accepted, m, st.verify_len)) {
        const tokens = try allocator.alloc(u32, 1 + m);
        tokens[0] = t1;
        for (drafts[0..m], 0..) |d, idx| tokens[1 + idx] = d;

        try self.generated_ids.append(allocator, t1);
        for (drafts[0..m]) |d| try self.generated_ids.append(allocator, d);

        if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
        self.last_hidden = new_hidden;
        self.has_last_hidden = true;
        new_hidden = .{ .ctx = null };

        self.mtpRoundAcceptObserve(chain.lookup, m, m);
        self.next_token_id = next_pending;
        self.advanceStep(1 + m);

        if (!chain.lookup) {
            self.mtpPlannerObserve(observed_limit, m);
            if (!group_planner.enabled() or self.mtp_planner_width == null) {
                self.updateMtpEvRound(m, m);
            }
        }
        if (tracing) {
            self.mtp_trace.add(.commit, ph.read());
            ph.reset();
        }
        try self.mtpPreDraftResolve(allocator, &lazy, accepted == greedy_raw, lazy_planned);
        if (tracing) {
            self.mtp_trace.add(.predraft, ph.read());
            self.mtp_gap_watch = io_util.Stopwatch.init(self.timer.io);
        }
        self.mtpTraceRoundEnd(m, m, m_lo);
        if (times_round) {
            self.mtpRoundEndObserve(m, m + 1, m_max > m_lo, m_lo, plan.width_trial, @as(f32, @floatFromInt(round_watch.read())) / @as(f32, std.time.ns_per_ms));
            self.mtp_ev_round_ms = mtpEmaMs(self.mtp_ev_round_ms, round_watch.read());
        } else {
            if (chain.lookup) self.mtpLookupObserve(m, m + 1);
            self.mtpRoundUntimed();
        }
        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = m,
        };
    }

    // Partial accept: roll back the trunk. On a GDN trunk the verify pass
    // captured per-position SSM/conv state, so roll back by truncating the
    // KV cache to the accepted length and slicing the capture — NO
    // re-forward of the accepted prefix (mirrors nextPld's fast path; the
    // re-forward is a full trunk weight read, and at depth > 1 most rounds
    // are partial, so it dominated the round cost). The next round's
    // h_prev is the TRUE verify hidden at the last committed position
    // (input index `accepted`), which forwardWithCaptureAll captured.
    // Non-GDN archs keep the proven restore + re-forward fallback.
    _ = mlx.mlx_array_free(new_hidden);
    new_hidden = .{ .ctx = null };

    const gdn_captured = if (self.ctx.ssm_entries) |entries|
        entries.len > 0 and entries[0].spec_state_seq.ctx != null
    else
        false;

    var re_new_hidden = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(re_new_hidden);
    if (gdn_captured or xfm.config.caps().verify_truncates) {
        const accepted_len: usize = 1 + @as(usize, accepted);
        // `truncate` overwrites cache.step with its length arg. On a GDN
        // trunk cache.step is a stale counter the model never reads
        // (positioning is moe_seq_offset), so keep the pre-verify value
        // (same rule as nextPld); MiMo's attention trunk keeps a LIVE step,
        // which truncate leaves where the re-forward fallback would.
        try self.ctx.cache.truncate(moe_seq_offset_snap + accepted_len, s);
        if (gdn_captured) self.ctx.cache.step = kv_step_snap;
        if (gdn_captured) try self.rollbackSsmFromCapture(self.ctx.ssm_entries.?, accepted, st.verify_len, s);
        self.ctx.moe_seq_offset.* = moe_seq_offset_snap + accepted_len;

        const vh_shape = mlx.getShape(verify_hidden_all);
        const start = [_]c_int{ 0, @intCast(accepted), 0 };
        const stop = [_]c_int{ 1, @as(c_int, @intCast(accepted)) + 1, vh_shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&re_new_hidden, verify_hidden_all, &start, 3, &stop, 3, &strides, 3, s));
    } else if (st.kv_snap) |*snap| {
        try self.ctx.cache.restore(snap);
        self.ctx.moe_seq_offset.* = moe_seq_offset_snap;

        const re_seq_len: c_int = @intCast(1 + accepted);
        const re_input_buf = try allocator.alloc(i32, 1 + accepted);
        defer allocator.free(re_input_buf);
        re_input_buf[0] = @intCast(t1);
        for (drafts[0..accepted], 0..) |d, idx| re_input_buf[1 + idx] = @intCast(d);
        const re_shape = [_]c_int{ 1, re_seq_len };
        const re_input = mlx.mlx_array_new_data(re_input_buf.ptr, &re_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(re_input);

        const re_logits = try xfm.forwardWithCapture(&self.ctx, re_input, &re_new_hidden);
        _ = mlx.mlx_array_free(re_logits);
    } else {
        // GDN trunk whose verify pass produced no capture — cannot roll
        // back safely. Unreachable on real targets (every qwen3_5-family
        // GDN layer populates the capture when capture_ssm_seq is set);
        // pinned by tests/test_mtp_equivalence.sh.
        _ = mlx.mlx_array_free(re_new_hidden);
        re_new_hidden = .{ .ctx = null };
        return error.MtpRollbackUnavailable;
    }

    const tokens = try allocator.alloc(u32, 1 + accepted);
    tokens[0] = t1;
    for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;

    try self.generated_ids.append(allocator, t1);
    for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);

    if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
    self.last_hidden = re_new_hidden;
    self.has_last_hidden = true;
    re_new_hidden = .{ .ctx = null };

    self.mtpRoundAcceptObserve(chain.lookup, m, accepted);
    self.next_token_id = next_pending;
    self.advanceStep(1 + accepted);

    if (!chain.lookup) {
        self.mtpPlannerObserve(if (stop_prefix.stop != null) accepted else observed_limit, accepted);
        if (!group_planner.enabled() or self.mtp_planner_width == null) {
            self.updateMtpEvRound(m, accepted);
        }
    }
    if (tracing) {
        self.mtp_trace.add(.commit, ph.read());
        ph.reset();
    }
    try self.mtpPreDraftResolve(allocator, &lazy, accepted == greedy_raw, lazy_planned);
    if (tracing) {
        self.mtp_trace.add(.predraft, ph.read());
        self.mtp_gap_watch = io_util.Stopwatch.init(self.timer.io);
    }
    self.mtpTraceRoundEnd(m, accepted, m_lo);
    if (times_round) {
        self.mtpRoundEndObserve(m, accepted + 1, m_max > m_lo, m_lo, plan.width_trial, @as(f32, @floatFromInt(round_watch.read())) / @as(f32, std.time.ns_per_ms));
        self.mtp_ev_round_ms = mtpEmaMs(self.mtp_ev_round_ms, round_watch.read());
    } else {
        if (chain.lookup) self.mtpLookupObserve(m, accepted + 1);
        self.mtpRoundUntimed();
    }
    return DrafterStepResult{
        .tokens = tokens,
        .accepted_tokens = accepted,
    };
}
