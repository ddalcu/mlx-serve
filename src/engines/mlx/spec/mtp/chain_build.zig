//! Building and dispatching MTP draft chains (solo, batched, lanes).
const std = @import("std");
const modules = @import("../../modules/call.zig");
const Qwen4MtpBatchRow = @import("../../../../plugins/qwen4_exp/mtp.zig").Qwen4MtpBatchRow;
const mlx = @import("../../../../mlx/ffi.zig");
const transformer_mod = @import("../../transformer.zig");
const log = @import("../../../../sdk/util/log.zig");
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");
const Transformer = transformer_mod.Transformer;
const Generator = @import("../../generate/generator.zig").Generator;
const MAX_ROUND_DRAFTS = @import("../mtp_head_ref.zig").MAX_ROUND_DRAFTS;
const MtpHeadRef = @import("../mtp_head_ref.zig").MtpHeadRef;
const MtpPreDraft = @import("chain.zig").MtpPreDraft;
const SamplingParams = @import("../../sampling/params.zig").SamplingParams;
const argmaxStepProposal = @import("../../sampling/spec_sampling.zig").argmaxStepProposal;
const draftConfidenceGraph = @import("../../sampling/spec_sampling.zig").draftConfidenceGraph;
const logitsStepProposal = @import("../../sampling/spec_sampling.zig").logitsStepProposal;
const sampleRowsAnySeed = @import("../../sampling/lazy.zig").sampleRowsAnySeed;
const sampleRowsHomogeneous = @import("../../sampling/lazy.zig").sampleRowsHomogeneous;
const sampleTokenLazy = @import("../../sampling/lazy.zig").sampleTokenLazy;
const shortlistProposal = @import("../../sampling/spec_sampling.zig").shortlistProposal;
const shortlistProposalRows = @import("../../sampling/spec_sampling.zig").shortlistProposalRows;
const shortlistStepProposal = @import("../../sampling/spec_sampling.zig").shortlistStepProposal;


/// Build draft steps [from..to) of `chain` — graph construction only, no
/// sync; the caller dispatches. Each step's sampled token ([1] lazy
/// array) feeds the next step's embedding lookup; the MTP post-norm
/// hidden chains as the next h_prev. Step 0 merges the deferred history
/// append (consumes `self.mtp_hist_stash`); a chunk-B call resumes from
/// `chain.h_chain`.
pub fn mtpChainBuild(self: *Generator, chain: *MtpPreDraft, from: u32, to: u32) !void {
    const xfm = self.xfm;
    const s = xfm.s;
    const head = self.mtp.?;
    const mc = &self.mtp_cache.?;
    const mtp_mrope_ctx = self.mtpMropeContext();
    const rerank_drafts = self.mtpRerankDrafts();
    std.debug.assert(chain.n_drafted == from);
    var i: u32 = from;
    while (i < to) : (i += 1) {
        const draft_sampling = self.mtpDraftSampling(i);
        const h_prev_arg: mlx.mlx_array = if (chain.h_chain) |h| h else self.last_hidden;
        const prev_tok_arr: mlx.mlx_array = if (i == 0) chain.t1_arr else chain.draft_arrs[i - 1];
        // Rerank drafts skip the head's own logits projection entirely:
        // the token comes off the coarse readout's re-scored shortlist,
        // greedily (argmax) or sampled from a q over those 32 rows.
        const path = mtpDraftStepPath(
            rerank_drafts,
            draft_sampling.temperature > 0.01,
            chain.conf_arrs != null and i < chain.plan.m_lo,
        );
        const use_rerank = path != .full_logits;
        // `want_logits: bool` could not tell a rerank draft from a history
        // append: both skip the vocab projection, but the draft still needs
        // the vector that projection would have consumed.
        const want: mtp_mod.StepWant = if (use_rerank) .mixed else .logits;
        if (i == 0 and self.mtp_hist_stash != null and self.mtp_hist_stash.?.pad != null and self.mtpPaddedHistory()) {
            try self.mtpPaddedHistoryAppend(chain);
        }
        var step_out = if (chain.place) |*place|
            try head.forwardPlaced(mc, prev_tok_arr, h_prev_arg, place, want)
        else if (i == 0 and self.mtp_hist_stash != null) blk: {
            // Deferred history append (stashed at the END of the
            // previous round, Phase 5a) merged into this chain's first
            // draft: ONE (n+1)-row head forward appends the
            // committed-history entries AND the first draft entry,
            // replacing the old per-round appendHistory forward. RoPE
            // offsets and cache-append order are byte-identical to the
            // appendHistory-then-stepArr sequence (pinned by the merged-
            // forward equivalence test in mtp.zig).
            var st = self.mtp_hist_stash.?;
            self.mtp_hist_stash = null;
            defer st.deinit();
            // Drop the previous round's stale draft tail — the old
            // Phase 5a truncate, moved to consume time (offset-only).
            try mc.truncate(st.off0, s);
            var merged_ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(merged_ids);
            var merged_hidden = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(merged_hidden);
            {
                const idv = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(idv);
                _ = mlx.mlx_vector_array_append_value(idv, st.ids);
                _ = mlx.mlx_vector_array_append_value(idv, prev_tok_arr);
                try mlx.check(mlx.mlx_concatenate_axis(&merged_ids, idv, 0, s));
                const hv = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(hv);
                _ = mlx.mlx_vector_array_append_value(hv, st.hidden);
                _ = mlx.mlx_vector_array_append_value(hv, h_prev_arg);
                try mlx.check(mlx.mlx_concatenate_axis(&merged_hidden, hv, 1, s));
            }
            var merged_host: [MAX_ROUND_DRAFTS + 2]u32 = undefined;
            @memcpy(merged_host[0..st.n], st.host_ids[0..st.n]);
            merged_host[st.n] = chain.t1;
            break :blk try head.forward(xfm, mc, merged_ids, merged_hidden, @intCast(st.off0), want, mtp_mrope_ctx, merged_host[0 .. st.n + 1]);
        } else try head.forward(xfm, mc, prev_tok_arr, h_prev_arg, @intCast(chain.off0 + i), want, mtp_mrope_ctx, if (i == 0) &.{chain.t1} else null);
        errdefer {
            if (step_out.logits.ctx != null) _ = mlx.mlx_array_free(step_out.logits);
            _ = mlx.mlx_array_free(step_out.hidden_next);
        }
        // A `.mixed` step that publishes `rerank_x` means `hidden_next` is
        // NOT what the lm_head consumes — on qwen4_exp it is the pre-mixer
        // `[B,S,hc*H]` stream and the mixer output is the rerank input.
        // Feeding the wrong one is a shape error, not a wrong answer.
        defer if (step_out.rerank_x.ctx != null) {
            _ = mlx.mlx_array_free(step_out.rerank_x);
        };
        if (use_rerank and chain.q_probs != null) {
            const rerank_x = if (step_out.rerank_x.ctx != null) step_out.rerank_x else step_out.hidden_next;
            const slots = chain.q_probs.?;
            if (try head.draftShortlist(xfm, rerank_x, draft_sampling.suppress_mask)) |shortlist| {
                var sl = shortlist;
                defer sl.deinit();
                const prop = try shortlistStepProposal(sl, self.mtpSamplingDraw(draft_sampling), s);
                chain.draft_arrs[i] = prop.id;
                slots[i] = prop.q;
            } else {
                // The coarse head just retired: propose greedily and hand
                // the verify the one-hot q that proposal really has.
                const id = try head.draftSelect(xfm, rerank_x, draft_sampling.suppress_mask);
                const prop = try argmaxStepProposal(id, @intCast(mlx.getShape(xfm.lm_head_w)[0]), s);
                chain.draft_arrs[i] = prop.id;
                slots[i] = prop.q;
            }
            chain.n_qp = i + 1;
        } else if (use_rerank) {
            const rerank_x = if (step_out.rerank_x.ctx != null) step_out.rerank_x else step_out.hidden_next;
            chain.draft_arrs[i] = try head.draftSelect(xfm, rerank_x, draft_sampling.suppress_mask);
        } else if (chain.q_probs) |slots| {
            const prop = try logitsStepProposal(step_out.logits, self.mtpSamplingDraw(draft_sampling), s);
            slots[i] = prop.q;
            chain.n_qp = i + 1;
            chain.draft_arrs[i] = prop.id;
        } else if (self.mtpCoupledDrafts()) {
            // The verify row's own keyed draw over the head's logits.
            var sp = self.sampling;
            sp.draw = chain.pos0 + i;
            chain.draft_arrs[i] = sampleTokenLazy(step_out.logits, sp, s);
        } else {
            chain.draft_arrs[i] = sampleTokenLazy(step_out.logits, draft_sampling, s);
        }
        chain.n_drafted = i + 1;
        if (chain.conf_arrs != null and i < chain.plan.m_lo) {
            // Chunk-A confidence: log p_head(draft) — built from the
            // step's own logits BEFORE they're freed (lazy graphs
            // hold their inputs internally).
            chain.conf_arrs.?[i] = try draftConfidenceGraph(step_out.logits, chain.draft_arrs[i], s);
            chain.n_conf = i + 1;
        }
        if (step_out.logits.ctx != null) _ = mlx.mlx_array_free(step_out.logits);
        step_out.logits = .{ .ctx = null };
        if (chain.h_chain) |h_old| _ = mlx.mlx_array_free(h_old);
        chain.h_chain = step_out.hidden_next;
        step_out.hidden_next = .{ .ctx = null };
    }
}

/// Fire the chain's built graphs [from..to) at the GPU (async, no
/// sync) so the head chain computes while the CPU builds the round's
/// remaining graphs. Chunk-A dispatches (from == 0) on two-chunk plans
/// also carry the confidence graphs and the chain hidden, so a
/// consume-time extension decision reads materialized arrays (near-free
/// sync) and chunk B resumes from a realized buffer. Dispatch timing
/// only — lazy sampling ops bind their PRNG key at graph BUILD time, so
/// values are identical.
pub fn mtpChainDispatch(chain: *const MtpPreDraft, from: u32, to: u32) !void {
    const ev = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ev);
    for (chain.draft_arrs[from..to]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
    if (chain.q_probs) |slots| {
        for (slots[from..to]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
    }
    if (from == 0) {
        if (chain.conf_arrs) |slots| {
            for (slots[0..chain.n_conf]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
            if (chain.h_chain) |h| _ = mlx.mlx_vector_array_append_value(ev, h);
        }
    }
    try mlx.check(mlx.mlx_async_eval(ev));
}

// One async eval over every row's built draft graphs so the chain runs while the CPU builds
// the verify; a lazy sampling op binds its PRNG key at graph build, so the drafts do not move.
pub fn mtpChainDispatchBatched(chains: []const MtpPreDraft) !void {
    const ev = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ev);
    var any = false;
    for (chains) |chain| {
        const to = chain.n_drafted;
        if (to == 0) continue;
        any = true;
        for (chain.draft_arrs[0..to]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
        if (chain.q_probs) |slots| {
            for (slots[0..@min(to, chain.n_qp)]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
        }
        if (chain.conf_arrs) |slots| {
            for (slots[0..chain.n_conf]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
            if (chain.h_chain) |h| _ = mlx.mlx_vector_array_append_value(ev, h);
        }
    }
    if (!any) return;
    try mlx.check(mlx.mlx_async_eval(ev));
}

pub fn mtpDraftActiveRows(chains: []const MtpPreDraft, step: u32, indices: []usize) usize {
    var n: usize = 0;
    for (chains, 0..) |chain, row| {
        if (step >= chain.m) continue;
        indices[n] = row;
        n += 1;
    }
    return n;
}

/// What one draft step reads. Both rerank paths skip the head's own vocab
/// projection (the token comes off the coarse readout's exactly re-scored
/// top-32); `full_logits` pays the whole lm_head because something
/// downstream needs a density over the vocabulary. A SAMPLED proposal does
/// not: q over the shortlist is the proposal's true density, zero off it,
/// and the verify's residual covers the rest.
pub const DraftStepPath = enum { rerank_greedy, rerank_sampled, full_logits };

pub fn mtpDraftStepPath(can_rerank: bool, sampled: bool, conf_needed: bool) DraftStepPath {
    // The chunk-A extension gate wants ln p_head(draft) over the whole
    // head, which the shortlist cannot answer.
    if (conf_needed or !can_rerank) return .full_logits;
    return if (sampled) .rerank_sampled else .rerank_greedy;
}

/// Whether this row drafts through the rerank shortlist at `step`. Greedy and
/// sampled rerank rows share the group: same coarse readout, same
/// `.mixed_last_row` head forward, only the tail over the 32 candidates
/// differs — so a mixed sampled/greedy workload does not split the step into
/// two head forwards.
/// Row-exact rounds draft from the head's full logits: the coarse
/// shortlist costs more than it saves there, and a sampled draft needs the
/// verify row's keyed draw over the full row.
pub fn mtpRerankDrafts(self: *const Generator) bool {
    return !self.mtp_serial_accept and self.mtp.?.canRerankDrafts();
}

pub fn mtpRowRerankMode(g: *Generator, c: *const MtpPreDraft, step: u32) bool {
    const conf_needed = c.conf_arrs != null and step < c.plan.m_lo;
    return mtpDraftStepPath(g.mtpRerankDrafts(), c.q_probs != null, conf_needed) != .full_logits;
}

pub fn mtpChainBuildBatched(gens: []const *Generator, chains: []MtpPreDraft, from: u32, to: u32) !void {
    if (gens.len != chains.len) return error.MtpBatchLengthMismatch;
    if (gens.len > 32) return error.MtpBatchTooWide;
    var step = from;
    while (step < to) : (step += 1) {
        var indices: [32]usize = undefined;
        const n = mtpDraftActiveRows(chains, step, &indices);
        if (n == 0) break;
        var modes: [32]bool = undefined;
        for (indices[0..n], 0..) |row, i| {
            modes[i] = mtpRowRerankMode(gens[row], &chains[row], step);
        }
        for ([_]bool{ true, false }) |mode| {
            var active_gens: [32]*Generator = undefined;
            var active_chains: [32]MtpPreDraft = undefined;
            var row_map: [32]usize = undefined;
            var count: usize = 0;
            for (indices[0..n], 0..) |row, i| {
                if (modes[i] != mode) continue;
                active_gens[count] = gens[row];
                active_chains[count] = chains[row];
                row_map[count] = row;
                count += 1;
            }
            if (count == 0) continue;
            defer for (row_map[0..count], 0..) |row, i| {
                chains[row] = active_chains[i];
            };
            try mtpChainBuildBatchedActive(active_gens[0..count], active_chains[0..count], step, step + 1);
        }
    }
}

/// One draft step for a group whose rerank rows include sampled chains:
/// per-row shortlists off ONE coarse readout, then one batched proposal
/// over the rows that sample this depth when they share a sampler and none
/// is seeded (the `sampleRowsLazy` gate), else one proposal per row. Every
/// other row takes its argmax.
pub fn mtpGroupSampledDrafts(
    gens: []const *Generator,
    chains: []MtpPreDraft,
    stacked_x: mlx.mlx_array,
    step: u32,
) !void {
    const xfm = gens[0].xfm;
    const s = xfm.s;
    const n = gens.len;
    var shortlists: [32]mtp_mod.Shortlist = undefined;
    const have = try gens[0].mtp.?.draftShortlistsBatched(xfm, stacked_x, gens[0].sampling.suppress_mask, shortlists[0..n]);
    if (!have) {
        // No coarse head: every row proposes greedily and hands the verify
        // the one-hot q that proposal really has.
        const vocab: c_int = @intCast(mlx.getShape(xfm.lm_head_w)[0]);
        for (gens, chains, 0..) |g, *c, k| {
            const k_c: c_int = @intCast(k);
            var xk = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xk);
            const xsh = mlx.getShape(stacked_x);
            try mlx.check(mlx.mlx_slice(&xk, stacked_x, &[_]c_int{ k_c, 0, 0 }, 3, &[_]c_int{ k_c + 1, 1, xsh[xsh.len - 1] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            const id = try g.mtp.?.draftSelect(xfm, xk, g.sampling.suppress_mask);
            if (c.q_probs) |slots| {
                const prop = try argmaxStepProposal(id, vocab, s);
                c.draft_arrs[step] = prop.id;
                slots[step] = prop.q;
                c.n_qp = step + 1;
            } else c.draft_arrs[step] = id;
            c.n_drafted = step + 1;
        }
        return;
    }
    defer for (shortlists[0..n]) |*sl| sl.deinit();

    // Which rows sample, and whether one batched draw can serve them all.
    var sampled: [32]usize = undefined;
    var params: [32]SamplingParams = undefined;
    var samples: [32]bool = @splat(false);
    var k_sampled: usize = 0;
    for (gens, chains, 0..) |g, *c, row| {
        const row_params = g.mtpDraftSampling(step);
        if (c.q_probs == null or row_params.temperature <= 0.01) continue;
        samples[row] = true;
        sampled[k_sampled] = row;
        params[k_sampled] = row_params;
        k_sampled += 1;
    }
    const batched = k_sampled > 1 and
        sampleRowsHomogeneous(params[0..k_sampled]) and
        !sampleRowsAnySeed(params[0..k_sampled]);

    for (gens, chains, 0..) |g, *c, row| {
        if (samples[row]) continue;
        if (c.q_probs) |slots| {
            const prop = try shortlistStepProposal(shortlists[row], g.mtpDraftSampling(step), s);
            c.draft_arrs[step] = prop.id;
            slots[step] = prop.q;
            c.n_qp = step + 1;
        } else c.draft_arrs[step] = try mtp_mod.shortlistArgmax(s, shortlists[row]);
        c.n_drafted = step + 1;
    }
    if (!batched) {
        for (sampled[0..k_sampled], params[0..k_sampled]) |row, row_params| {
            const prop = try shortlistProposal(shortlists[row], gens[row].mtpSamplingDraw(row_params), s);
            chains[row].draft_arrs[step] = prop.id;
            chains[row].n_drafted = step + 1;
            chains[row].q_probs.?[step] = prop.q;
            chains[row].n_qp = step + 1;
        }
        return;
    }
    const group = try shortlistProposalRows(shortlists[0..n], sampled[0..k_sampled], params[0], s);
    for (sampled[0..k_sampled], 0..) |row, j| {
        chains[row].draft_arrs[step] = group.ids[j];
        chains[row].n_drafted = step + 1;
        chains[row].q_probs.?[step] = group.qs[j];
        chains[row].n_qp = step + 1;
    }
}

var mtp_lanes_logged: bool = false;

/// Draft steps [from..to) for a group on one head (`MtpHeadRef.lanesWith`): every lane is a row
/// of ONE head forward per step, and one exact trunk-head readout proposes for all of them.
pub fn mtpChainBuildLanes(gens: []const *Generator, chains: []MtpPreDraft, from: u32, to: u32) !void {
    const xfm = gens[0].xfm;
    const s = xfm.s;
    const head = gens[0].mtp.?;
    const n = gens.len;
    if (!mtp_lanes_logged) {
        mtp_lanes_logged = true;
        log.info("[batched] mtp head engaged (slots={d} depth={d})\n", .{ n, to });
    }
    var i: u32 = from;
    while (i < to) : (i += 1) {
        var lanes: [mtp_mod.MAX_LANES]MtpHeadRef.Lane = undefined;
        for (gens, chains, 0..) |g, *c, k| {
            std.debug.assert(c.n_drafted == i);
            const mc = &g.mtp_cache.?;
            if (i == 0) if (g.mtp_hist_stash) |pending| {
                var stash = pending;
                g.mtp_hist_stash = null;
                defer stash.deinit();
                try mc.truncate(stash.off0, s);
                try head.appendRows(xfm, mc, stash.ids, stash.hidden, @intCast(stash.off0));
            };
            lanes[k] = .{
                .cache = mc,
                .id = if (i == 0) c.t1_arr else c.draft_arrs[i - 1],
                .hidden = if (c.h_chain) |h| h else g.last_hidden,
                .rope_offset = @intCast(c.off0 + i),
            };
        }
        const post = try head.forwardLanes(xfm, lanes[0..n]);
        defer _ = mlx.mlx_array_free(post);
        const hidden_size = mlx.getShape(post)[2];
        var stacked_x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stacked_x);
        try mlx.check(mlx.mlx_reshape(&stacked_x, post, &[_]c_int{ @intCast(n), 1, hidden_size }, 3, s));

        var any_sampled = false;
        for (chains) |*c| {
            if (c.q_probs != null) any_sampled = true;
        }
        if (any_sampled) {
            try mtpGroupSampledDrafts(gens, chains, stacked_x, i);
        } else {
            const ids = try head.draftSelectBatched(xfm, stacked_x, gens[0].sampling.suppress_mask);
            defer _ = mlx.mlx_array_free(ids);
            for (chains, 0..) |*c, k| {
                c.draft_arrs[i] = try Transformer.axisView(s, ids, 0, k);
                c.n_drafted = i + 1;
            }
        }
        for (chains, 0..) |*c, k| {
            const h_next = try Transformer.axisView(s, stacked_x, 0, k);
            if (c.h_chain) |h_old| _ = mlx.mlx_array_free(h_old);
            c.h_chain = h_next;
        }
    }
}

pub fn mtpBatchHeadPosition(pos_base: c_int, seq_offset: usize) c_int {
    return if (seq_offset == 0 and pos_base == -1) 1 else pos_base + @as(c_int, @intCast(seq_offset));
}

pub fn mtpChainBuildBatchedActive(gens: []const *Generator, chains: []MtpPreDraft, from: u32, to: u32) !void {
    if (gens.len == 0) return;
    std.debug.assert(gens.len == chains.len);
    var any_mrope = false;
    var all_qwen4 = true;
    var one_head = true;
    for (gens) |g| {
        if (g.mtpMropeContext() != null) any_mrope = true;
        if (g.mtp.? != .qwen4) all_qwen4 = false;
        if (!gens[0].mtp.?.lanesWith(g.mtp.?)) one_head = false;
    }
    // The lanes step hands back no head logits: a row that needs them builds alone.
    if (one_head and !any_mrope and gens.len > 1 and mtpRowRerankMode(gens[0], &chains[0], from)) {
        return mtpChainBuildLanes(gens, chains, from, to);
    }
    if (!all_qwen4 or any_mrope or gens.len == 1) {
        for (gens, chains) |g, *c| try g.mtpChainBuild(c, from, to);
        return;
    }
    const xfm = gens[0].mtp.?.qwen4;
    const n = gens.len;
    var i: u32 = from;
    while (i < to) : (i += 1) {
        var rows: [32]Qwen4MtpBatchRow = undefined;
        var merged_ids: [32]mlx.mlx_array = undefined;
        var merged_hidden: [32]mlx.mlx_array = undefined;
        var own_ids: [32]bool = @splat(false);
        var own_h: [32]bool = @splat(false);
        for (merged_ids[0..n], 0..) |*p, k| {
            p.* = .{ .ctx = null };
            merged_hidden[k] = .{ .ctx = null };
        }
        defer for (0..n) |k| {
            if (own_ids[k] and merged_ids[k].ctx != null) _ = mlx.mlx_array_free(merged_ids[k]);
            if (own_h[k] and merged_hidden[k].ctx != null) _ = mlx.mlx_array_free(merged_hidden[k]);
        };
        var modes: [32]bool = undefined;
        for (gens, chains, 0..) |g, *c, row| modes[row] = mtpRowRerankMode(g, c, i);
        const use_rerank = modes[0];
        const project: @import("../../../../plugins/qwen4_exp/mtp.zig").Qwen4MtpProject = if (use_rerank) .mixed_last_row else .last_row;
        for (gens, chains, 0..) |g, *c, k| {
            const qref = g.mtp_cache.?.qwen4;
            const st = g.mtp_cache.?.qwen4StateForBatch();
            const h_prev_arg: mlx.mlx_array = if (c.h_chain) |h| h else g.last_hidden;
            const prev_tok_arr: mlx.mlx_array = if (i == 0) c.t1_arr else c.draft_arrs[i - 1];
            if (i == 0 and g.mtp_hist_stash != null) {
                var stash = g.mtp_hist_stash.?;
                g.mtp_hist_stash = null;
                defer stash.deinit();
                try qref.t.qwen4MtpTruncateOn(st, stash.off0);
                var mids = mlx.mlx_array_new();
                var mh = mlx.mlx_array_new();
                {
                    const idv = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(idv);
                    _ = mlx.mlx_vector_array_append_value(idv, stash.ids);
                    _ = mlx.mlx_vector_array_append_value(idv, prev_tok_arr);
                    try mlx.check(mlx.mlx_concatenate_axis(&mids, idv, 0, xfm.s));
                    const hv = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(hv);
                    _ = mlx.mlx_vector_array_append_value(hv, stash.hidden);
                    _ = mlx.mlx_vector_array_append_value(hv, h_prev_arg);
                    try mlx.check(mlx.mlx_concatenate_axis(&mh, hv, 1, xfm.s));
                }
                merged_ids[k] = mids;
                merged_hidden[k] = mh;
                own_ids[k] = true;
                own_h[k] = true;
            } else {
                merged_ids[k] = prev_tok_arr;
                merged_hidden[k] = h_prev_arg;
            }
            rows[k] = .{
                .st = st,
                .stream = merged_hidden[k],
                .token_ids = merged_ids[k],
                .pos_offset = mtpBatchHeadPosition(st.pos_base, st.seq_offset),
            };
        }
        const outs = try modules.call("qwen4_mtp_forward_batched", .{ xfm, rows[0..n], project, to });
        defer {
            for (outs) |out| {
                if (out.logits.ctx != null) _ = mlx.mlx_array_free(out.logits);
                if (out.stream.ctx != null) _ = mlx.mlx_array_free(out.stream);
                if (out.mixed.ctx != null) _ = mlx.mlx_array_free(out.mixed);
            }
            xfm.allocator.free(outs);
        }
        var mixed_parts: [32]mlx.mlx_array = undefined;
        if (use_rerank) {
            for (outs, 0..) |o, k| mixed_parts[k] = o.mixed;
            const mvec = mlx.mlx_vector_array_new_data(mixed_parts[0..n].ptr, n);
            defer _ = mlx.mlx_vector_array_free(mvec);
            var stacked_x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(stacked_x);
            try mlx.check(mlx.mlx_concatenate_axis(&stacked_x, mvec, 0, xfm.s));
            var any_sampled = false;
            for (chains) |*c| {
                if (c.q_probs != null) any_sampled = true;
            }
            if (any_sampled) {
                try mtpGroupSampledDrafts(gens, chains, stacked_x, i);
            } else {
                const ids = try modules.call("qwen4_draft_select_batched", .{ xfm, stacked_x, gens[0].sampling.suppress_mask });
                defer _ = mlx.mlx_array_free(ids);
                for (chains, 0..) |*c, k| {
                    const k_c: c_int = @intCast(k);
                    var one = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_slice(&one, ids, &[_]c_int{ k_c, 0 }, 2, &[_]c_int{ k_c + 1, 1 }, 2, &[_]c_int{ 1, 1 }, 2, xfm.s));
                    c.draft_arrs[i] = one;
                }
            }
            for (chains, 0..) |*c, k| {
                c.n_drafted = i + 1;
                if (c.h_chain) |h_old| _ = mlx.mlx_array_free(h_old);
                c.h_chain = outs[k].stream;
                outs[k].stream = .{ .ctx = null };
            }
        } else {
            for (gens, chains, 0..) |g, *c, k| {
                const draft_sampling = g.mtpDraftSampling(i);
                if (c.q_probs) |slots| {
                    const prop = try logitsStepProposal(outs[k].logits, g.mtpSamplingDraw(draft_sampling), xfm.s);
                    slots[i] = prop.q;
                    c.n_qp = i + 1;
                    c.draft_arrs[i] = prop.id;
                } else {
                    c.draft_arrs[i] = sampleTokenLazy(outs[k].logits, draft_sampling, xfm.s);
                }
                c.n_drafted = i + 1;
                if (c.conf_arrs != null and i < c.plan.m_lo) {
                    c.conf_arrs.?[i] = try draftConfidenceGraph(outs[k].logits, c.draft_arrs[i], xfm.s);
                    c.n_conf = i + 1;
                }
                if (c.h_chain) |h_old| _ = mlx.mlx_array_free(h_old);
                c.h_chain = outs[k].stream;
                outs[k].stream = .{ .ctx = null };
            }
        }
    }
}
