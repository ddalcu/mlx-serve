//! DFlash block-drafter rounds: verify, copy, plain/pipelined policy rounds, parked taps.
const std = @import("std");
const DFLASH_GATE_WARMUP = @import("gates.zig").DFLASH_GATE_WARMUP;
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const tokenizer_mod = @import("../../../sdk/tokenizer/tokenizer.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const pld_index = @import("../../../sdk/spec/pld_index.zig");
const mtp_lookup = @import("mtp_lookup.zig");
const round_cost = @import("round_cost.zig");
const dflash_policy = @import("dflash/policy.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const dflash_mod = @import("dflash/dflash.zig");
const plugin_driver = @import("../plugin/driver.zig");
const lilicorr_mod = @import("dflash/lilicorr.zig");
const testing = std.testing;
const CommittedArgmax = @import("verify.zig").CommittedArgmax;
const DFLASH_COPY_MATCH = @import("dflash_tree.zig").DFLASH_COPY_MATCH;
const DFLASH_COPY_WINDOW = @import("dflash_tree.zig").DFLASH_COPY_WINDOW;
const DrafterStepResult = @import("drafter.zig").DrafterStepResult;
const Generator = @import("../generate/generator.zig").Generator;
const MTP_EV_EMA_BETA = @import("mtp/depth.zig").MTP_EV_EMA_BETA;
const SamplingParams = @import("../sampling/params.zig").SamplingParams;
const capAcceptedForTokenBudget = @import("verify.zig").capAcceptedForTokenBudget;
const dflashTraceEnabled = @import("mtp/trace.zig").dflashTraceEnabled;
const dflashTreeNodes = @import("dflash_tree.zig").dflashTreeNodes;
const filteredProbsBlock = @import("../sampling/spec_sampling.zig").filteredProbsBlock;
const mtpDepthPolicy = @import("mtp/ev_plan.zig").mtpDepthPolicy;
const pldOneHotRow = @import("../sampling/spec_sampling.zig").pldOneHotRow;
const probAt = @import("../sampling/spec_sampling.zig").probAt;
const probsAtLastPos = @import("../sampling/spec_sampling.zig").probsAtLastPos;
const sampleFromProbs = @import("../sampling/spec_sampling.zig").sampleFromProbs;
const sampleResidual = @import("../sampling/spec_sampling.zig").sampleResidual;
const selectorQRow = @import("../sampling/spec_sampling.zig").selectorQRow;
const sliceProbRow = @import("../sampling/spec_sampling.zig").sliceProbRow;
const specAcceptProb = @import("mtp/depth.zig").specAcceptProb;
const specDecodeUnsupported = @import("verify.zig").specDecodeUnsupported;
const verifyArgmax = @import("verify.zig").verifyArgmax;


fn embedTrunk(ctx: *const anyopaque, ids: mlx.mlx_array) anyerror!mlx.mlx_array {
    const xfm: *const transformer_mod.Transformer = @ptrCast(@alignCast(ctx));
    return xfm.rawEmbedding(ids);
}

/// Step-based generator. Call `init` to prefill, then `next` per token.
/// Uses a fully-lazy async pipeline matching mlx-lm: sample + next forward are
/// built as a single lazy computation graph, async_eval'd together. The GPU
/// never idles between token generation steps.
/// A request's round-policy state (`dflash_policy.zig`).
pub const DflashPol = struct {
    calib: dflash_policy.Calibration = .{},
    drafting: dflash_policy.Drafting = .{},
    /// Verify exactly this many rows (t1 included) whenever the draft has them; 0 = the policy chooses.
    pin_rows: u32 = 0,
    /// What a plain tick costs on this machine at this context, from this request's own pipelined ticks.
    /// The shared round-cost table is no source: its plain cell folds transition ticks and, once stale,
    /// takes one outlier at half weight.
    plain: dflash_policy.PlainCost = .{},
    /// How copied continuations have fared.
    copy: dflash_policy.CopyStats = .{},
    /// Since the last pipelined plain tick ended; null once any other kind of round has run.
    plain_watch: ?io_util.Stopwatch = null,
};

/// DFlash block-drafter step. ONE assistant forward proposes
/// `block_size - 1` drafts, conditioned on the trunk's cached
/// target_layer_ids hiddens (the per-request `dflash_ctx`); the trunk
/// verify over `[t1, drafts...]` follows the standard spec invariant
/// (`cache.step = prompt_len + emitted`, t1 NOT in cache on entry,
/// correction from ORIGINAL `verify_logits[accepted]`).
///
/// Rollback is an offset-only `cache.truncate` — legal because
/// `DflashModel.bind` restricts targets to the pure-KVCache standard
/// path (no SSM, no module-owned state), where a position's K/V depend
/// only on earlier positions (causal) and are identical whether computed
/// in a width-16 verify or any re-forward. No re-forward runs at all:
/// the next round's context comes from slicing THIS round's verify
/// captures to the committed prefix — exactly the reference's
/// `hidden_states[i+1][:, :n_accepted]`.
///
/// Drafts are greedy (argmax over the trunk lm_head on assistant
/// hiddens, anchor row DROPPED — reference `[:, 1:]`); sampled requests
/// use the same one-hot Leviathan acceptance the drafter/PLD paths use.
pub fn nextDflash(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    // The budget's last token is the previous verify's: emit it without a round. (A pipelined plain
    // stretch has no such token: its last one comes out of `next`.)
    if (!self.done and !self.spec_disabled_runtime and self.max_tokens -| self.completion_tokens == 1 and !self.has_pending_logits and !self.has_pending_token) {
        if (try self.checkStop()) return null;
        const tokens = try allocator.alloc(u32, 1);
        errdefer allocator.free(tokens);
        tokens[0] = self.next_token_id;
        try self.generated_ids.append(allocator, tokens[0]);
        self.advanceStep(1);
        self.unforwarded_tail = 1;
        self.done = true;
        self.finish_reason = "length";
        return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
    }
    // The kv term is the same physics for either block decoder (one
    // forward, one KV read, shared across the block's rows), so a DFlash
    // round is an observation for it too — and on a DFlash-only server
    // it is the ONLY source.
    const dflash_kv_watch = io_util.Stopwatch.init(self.timer.io);
    // The round-cost table sees the same round: drafts = block - 1, or
    // 0 when the runtime gate fell back to serial (a serial sample is
    // the "no spec" candidate a width chooser needs).
    const dflash_gen_before = self.generated_ids.items.len;
    self.dflash_round_width = 0;
    self.dflash_unobserved = false;
    const dflash_rounds_before: u64 = self.dflash_attempted + self.dflash_plain;
    defer {
        const ms = @as(f32, @floatFromInt(dflash_kv_watch.read())) / @as(f32, std.time.ns_per_ms);
        const emitted = self.generated_ids.items.len - dflash_gen_before;
        const post_warmup = dflash_rounds_before >= DFLASH_GATE_WARMUP;
        const wall = if (post_warmup) self.mtpRegimeWallMs(ms) else ms;
        self.specObserveRound(self.dflash_round_width, wall, @floatFromInt(emitted), post_warmup and emitted > 0 and !self.dflash_unobserved, false);
    }
    if (self.done) return null;
    std.debug.assert(self.dflash != null);
    std.debug.assert(self.dflash_ctx != null);
    if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

    if (self.completion_tokens >= self.max_tokens) {
        self.done = true;
        self.finish_reason = "length";
        return null;
    }

    if (self.spec_disabled_runtime) {
        const tok_opt = try self.next(allocator);
        if (tok_opt == null) return null;
        const tokens = try allocator.alloc(u32, 1);
        tokens[0] = tok_opt.?;
        return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
    }

    // The round policy: a plain tick runs on the serial pipeline; every other round starts from the drained state.
    var policy_draft = true;
    if (self.dflash_pol) |*pol| {
        policy_draft = pol.drafting.shouldDraft();
        if (!policy_draft and self.dflashCanPipeline()) return try self.dflashPlainPipelined(allocator);
        pol.plain_watch = null;
        if (self.has_pending_logits or self.has_pending_token) return try self.dflashLeavePipeline(allocator);
    } else if (self.has_pending_logits or self.has_pending_token) {
        return try self.dflashLeavePipeline(allocator);
    }

    var round_width: u32 = @max(self.dflash_block_size, 2) - 1;
    // The round policy replaces the serial fallback: a round with no drafts is the verify
    // forward at one row, so DFlash resumes whenever drafting pays again.
    if (self.dflash_pol != null and !policy_draft) round_width = 0;
    self.dflash_round_width = round_width;
    if (round_width == 0 and self.dflash_pol == null) {
        const tok_opt = try self.next(allocator);
        if (tok_opt == null) return null;
        const tokens = try allocator.alloc(u32, 1);
        tokens[0] = tok_opt.?;
        return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
    }

    if (try self.checkStop()) return null; // t1 is this block's first emit: stop before drafting

    const xfm = self.xfm;
    const s = xfm.s;
    const model = self.dflash.?;
    const dctx = &self.dflash_ctx.?;
    // Drafts the assistant proposes (0 = a plain round) and drafts the verify carries: the policy may keep fewer.
    const m_draft: u32 = round_width;
    var m: u32 = m_draft;
    const t1: u32 = self.next_token_id;
    // On the moe/GDN path positions come from moe_seq_offset (cache.step
    // is a bookkeeping counter the model never reads there — same rule as
    // nextMtp); the standard path keeps cache.step as the anchor.
    const moe_path = self.xfm.moe_layers != null or plugin_driver.positionsOf(self.xfm) == .gdn;
    // A hybrid trunk (LFM2 DSpark) positions from moe_seq_offset too, but
    // its cache.step IS genuine (every token passes the attention layers),
    // so it keeps the truncate and only needs the offset + conv-state
    // rollback. Deciding the two independently is what keeps a partial
    // accept from either mis-positioning or double-counting.
    const hybrid_path = plugin_driver.positionsOf(self.xfm) == .hybrid;
    const anchor_pos: usize = if (moe_path or hybrid_path) self.ctx.moe_seq_offset.* else self.ctx.cache.step;
    const kv_step_snap = self.ctx.cache.step;
    // A drafted round reads the whole context: whatever the plain rounds parked joins it first.
    if (m_draft > 0) try self.dflashFlushPending(allocator);
    std.debug.assert(dctx.absLen() + self.dflash_pending_rows == anchor_pos);

    const tracing = dflashTraceEnabled();
    var ph: io_util.Stopwatch = undefined;
    if (tracing) {
        ph = io_util.Stopwatch.init(self.timer.io);
        if (self.dflash_gap_watch) |*gw| self.dflash_trace.add(.gap, gw.read());
        self.dflash_gap_watch = null;
    }

    // Sampled requests accept through the full Leviathan ratio
    // min(1, p/q), so a GREEDY draft is a one-hot q — and at temperature
    // the target row is flat, which is exactly where min(1, p(argmax))
    // collapses. Drafting FROM the request's own distribution makes q
    // track p and keeps acceptance flat across temperature. Greedy
    // requests keep the argmax path untouched, so the byte-equality
    // guard is unaffected.
    // A row-exact trunk samples with the serial sampler's keys: drafts
    // drawn with the verify row's key, rows kept only while they match.
    const serial = self.sampling.keyed and self.sampling.temperature > 0.01 and self.sampling.seed != null;
    const stochastic = self.sampling.temperature > 0.01 and !serial;
    // Prefix snapshots keep assistant K/V without the LiLiCorr anchor.
    // A plain round supplies a fresh committed row for the next draft.
    if (m_draft == 0 or (model.lilicorr != null and dctx.anchor.ctx == null))
        return try self.dflashPlainRound(allocator, t1, serial, stochastic);
    if (self.dflash_pol != null) if (try self.dflashCopyRound(allocator, t1, anchor_pos, kv_step_snap, serial, stochastic, tracing, &ph, dflash_kv_watch)) |res| return res;
    // DFlash2 path selector: when the sidecar ships one, drafts come from
    // the pairwise-scored path trace instead of per-position argmax /
    // block sampling. Greedy requests keep the byte-equality bar (a
    // selector draft only survives verify if it IS the trunk argmax);
    // stochastic requests sample the selector's own candidate softmax and
    // accept through min(1, p/q) with q read off the traced path —
    // exact by construction.
    const use_selector = model.selector != null;
    // DSpark: the block's base logits are position-parallel, but each
    // position's draft is picked from logits CORRECTED by the token
    // drafted at the previous one (the Markov bigram bias). Chaining it
    // is what the head is for — dropping it drafts every position from
    // an uncorrected distribution the sidecar was never trained to emit.
    const use_markov = model.markov != null;
    const tree_round = use_selector and !use_markov and (serial or !stochastic) and xfm.specTreeSupported() and self.dflash_pol == null;
    // A continuation the context backs long enough to fill the tree is
    // verified as the round's chain, with no assistant forward.
    if (tree_round and self.dflash_copy_silent > 0) {
        self.dflash_copy_silent -= 1;
    } else if (tree_round) {
        const prompt = self.prompt_ids_owned;
        const gen = self.generated_ids.items;
        const committed = try allocator.alloc(u32, @min(prompt.len + gen.len + 1, DFLASH_COPY_WINDOW));
        defer allocator.free(committed);
        pld_index.tailWindow(committed, prompt, gen, t1);
        if (pld_index.backedCopy(committed, dflashTreeNodes(m), DFLASH_COPY_MATCH)) |c| if (c.draft.len == dflashTreeNodes(m))
            return try self.dflashTreeRound(allocator, model, dctx, null, null, t1, m, anchor_pos, kv_step_snap, serial, c.draft);
    }

    // ── Phase 1: one assistant forward drafts all m tokens ──
    // Row mapping is the export's convention (`anchor_row_drafts`):
    // DFlash reads mask rows 1..bs-1 (anchor row dropped), DSpark reads
    // ALL rows starting at the anchor — so DSpark needs only m noise rows
    // for the same m drafts and the same verify width.
    const noise_rows: u32 = if (model.lilicorr != null)
        model.config.block_size
    else if (model.config.anchor_row_drafts)
        m_draft
    else
        m_draft + 1;
    const noise_ids = try allocator.alloc(i32, noise_rows);
    defer allocator.free(noise_ids);
    noise_ids[0] = @intCast(t1);
    for (noise_ids[1..]) |*v| v.* = @intCast(model.config.mask_token_id);
    const noise_shape = [_]c_int{ 1, @intCast(noise_rows) };
    const noise_input = mlx.mlx_array_new_data(noise_ids.ptr, &noise_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(noise_input);
    // RAW table rows — no embed norm, no scale (the DFlash contract).
    const noise_embeds = try xfm.rawEmbedding(noise_input);
    defer _ = mlx.mlx_array_free(noise_embeds);

    const blk_hidden = try dflash_mod.forwardBlock(model, dctx, noise_embeds, anchor_pos);
    defer _ = mlx.mlx_array_free(blk_hidden);
    if (tracing) {
        try mlx.check(mlx.mlx_array_eval(blk_hidden));
        self.dflash_trace.add(.assist, ph.read());
        ph.reset();
    }
    const draft_logits_all = try model.draftLogits(xfm, blk_hidden);
    defer _ = mlx.mlx_array_free(draft_logits_all);

    if (model.lilicorr) |*hd| {
        const slots: u32 = model.config.block_size - 1;
        if (m > slots) m = slots;
        var slot_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(slot_logits);
        const vocab = mlx.getShape(draft_logits_all)[2];
        const start = [_]c_int{ 0, 1, 0 };
        const stop = [_]c_int{ 1, 1 + @as(c_int, @intCast(slots)), vocab };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&slot_logits, draft_logits_all, &start, 3, &stop, 3, &strides, 3, s));
        const ids = try hd.draft(allocator, .{ .ctx = xfm, .call = embedTrunk }, dctx.anchor, blk_hidden, slot_logits, m);
        defer allocator.free(ids);
        const ids_i32 = try allocator.alloc(i32, m);
        defer allocator.free(ids_i32);
        for (ids, ids_i32) |v, *dst| dst.* = @intCast(v);
        const row_shape = [_]c_int{ 1, @as(c_int, @intCast(m)) };
        const host_arr = mlx.mlx_array_new_data(ids_i32.ptr, &row_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(host_arr);
        var lili_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lili_ids);
        try mlx.check(mlx.mlx_array_set(&lili_ids, host_arr));
        if (tracing) {
            try mlx.check(mlx.mlx_array_eval(lili_ids));
            self.dflash_trace.add(.head, ph.read());
            ph.reset();
        }
        return try self.dflashVerify(allocator, .{
            .t1 = t1,
            .m = m,
            .m_draft = m_draft,
            .anchor_pos = anchor_pos,
            .kv_step_snap = kv_step_snap,
            .serial = serial,
            .stochastic = stochastic,
            .tracing = tracing,
            .ph = &ph,
            .watch = dflash_kv_watch,
            .draft_ids = lili_ids,
            .sel_path = null,
            .draft_q = .{ .ctx = null },
        });
    }

    // Slice the m draft rows: DFlash drops the anchor row (1..bs-1),
    // DSpark reads every row (0..m) — the anchor row IS draft 0.
    const dl_shape = mlx.getShape(draft_logits_all);
    var draft_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(draft_logits);
    {
        const row0: c_int = if (model.config.anchor_row_drafts) 0 else 1;
        const start = [_]c_int{ 0, row0, 0 };
        const stop = [_]c_int{ 1, row0 + @as(c_int, @intCast(m_draft)), dl_shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&draft_logits, draft_logits_all, &start, 3, &stop, 3, &strides, 3, s));
    }
    if (tree_round) {
        return try self.dflashTreeRound(allocator, model, dctx, blk_hidden, draft_logits, t1, m, anchor_pos, kv_step_snap, serial, &.{});
    }
    var sel_path: ?dflash_mod.SelectedPath = null;
    defer if (sel_path) |*sp| sp.deinit(allocator);
    var draft_q: mlx.mlx_array = .{ .ctx = null }; // [m, V] proposal density
    defer if (draft_q.ctx != null) {
        _ = mlx.mlx_array_free(draft_q);
    };
    var draft_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(draft_ids);
    if (use_markov) {
        const mh = &model.markov.?;
        const ids_i32 = try allocator.alloc(i32, m);
        defer allocator.free(ids_i32);
        var q_rows: ?[]mlx.mlx_array = null;
        defer if (q_rows) |rows| {
            for (rows) |r| _ = mlx.mlx_array_free(r);
            allocator.free(rows);
        };
        if (stochastic) {
            const rows = try allocator.alloc(mlx.mlx_array, m);
            for (rows) |*r| r.* = .{ .ctx = null };
            q_rows = rows;
        }
        var prev: u32 = t1;
        var step: u32 = 0;
        while (step < m) : (step += 1) {
            var base_row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(base_row);
            {
                const start = [_]c_int{ 0, @intCast(step), 0 };
                const stop = [_]c_int{ 1, @as(c_int, @intCast(step)) + 1, dl_shape[2] };
                const strides = [_]c_int{ 1, 1, 1 };
                try mlx.check(mlx.mlx_slice(&base_row, draft_logits, &start, 3, &stop, 3, &strides, 3, s));
            }
            const corrected = try mh.stepLogits(base_row, prev, s);
            defer _ = mlx.mlx_array_free(corrected);
            if (stochastic) {
                // Sample the step from the request's own filtered
                // distribution and keep the row as q — the accept ratio
                // and the reject residual both need the density the
                // draft was actually drawn from.
                const qrow = try filteredProbsBlock(corrected, self.sampling, s);
                var logq = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(logq);
                try mlx.check(mlx.mlx_log(&logq, qrow, s));
                const null_key = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(null_key);
                var sampled = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(sampled);
                try mlx.check(mlx.mlx_random_categorical(&sampled, logq, -1, null_key, s));
                try mlx.check(mlx.mlx_array_eval(sampled));
                const sd = mlx.mlx_array_data_int32(sampled) orelse return error.MlxArrayDataNull;
                prev = @intCast(sd[0]);
                q_rows.?[step] = qrow;
            } else {
                var amax = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(amax);
                try mlx.check(mlx.mlx_argmax_axis(&amax, corrected, 2, false, s));
                var as_i32 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(as_i32);
                try mlx.check(mlx.mlx_astype(&as_i32, amax, .int32, s));
                try mlx.check(mlx.mlx_array_eval(as_i32));
                const ad = mlx.mlx_array_data_int32(as_i32) orelse return error.MlxArrayDataNull;
                prev = @intCast(ad[0]);
            }
            ids_i32[step] = @intCast(prev);
        }
        if (q_rows) |rows| {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (rows) |r| _ = mlx.mlx_vector_array_append_value(vec, r);
            draft_q = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&draft_q, vec, 0, s));
        }
        const row_shape = [_]c_int{ 1, @as(c_int, @intCast(m)) };
        const host_arr = mlx.mlx_array_new_data(ids_i32.ptr, &row_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(host_arr);
        try mlx.check(mlx.mlx_array_set(&draft_ids, host_arr));
    } else if (use_selector) {
        // A serial-sampled request drafts the greedy path: over 20 seeds it
        // kept more drafts than paths scored with the verify rows' noise.
        const sel_temp: f32 = if (stochastic) self.sampling.temperature else 0.0;
        sel_path = try dflash_mod.selectPath(
            allocator,
            &model.selector.?,
            model.config.selector_top_k,
            blk_hidden,
            draft_logits,
            t1,
            sel_temp,
            self.prng.random(),
            s,
        );
        // The policy verifies the rows the selector's confidence pays for; none at all makes this a plain round.
        if (self.dflash_pol != null) {
            m = self.dflashPolicyRows(&sel_path.?, m_draft) - 1;
            self.dflash_round_width = m;
            if (m == 0) {
                self.dflash_unobserved = true;
                const plain = try self.dflashPlainRound(allocator, t1, serial, stochastic);
                self.dflashRecordDrafted(1, dflash_kv_watch.read());
                return plain;
            }
        }
        const ids_i32 = try allocator.alloc(i32, m);
        defer allocator.free(ids_i32);
        for (sel_path.?.ids[0..m], ids_i32) |v, *d| d.* = @intCast(v);
        const row_shape = [_]c_int{ 1, @intCast(m) };
        const host_arr = mlx.mlx_array_new_data(ids_i32.ptr, &row_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(host_arr);
        try mlx.check(mlx.mlx_array_set(&draft_ids, host_arr));
    } else {
        var draft_amax = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(draft_amax);
        try mlx.check(mlx.mlx_argmax_axis(&draft_amax, draft_logits, 2, false, s));
        try mlx.check(mlx.mlx_astype(&draft_ids, draft_amax, .int32, s));
    }
    if (tracing) {
        try mlx.check(mlx.mlx_array_eval(draft_ids));
        self.dflash_trace.add(.head, ph.read());
        ph.reset();
    }

    return try self.dflashVerify(allocator, .{
        .t1 = t1,
        .m = m,
        .m_draft = m_draft,
        .anchor_pos = anchor_pos,
        .kv_step_snap = kv_step_snap,
        .serial = serial,
        .stochastic = stochastic,
        .tracing = tracing,
        .ph = &ph,
        .watch = dflash_kv_watch,
        .draft_ids = draft_ids,
        .sel_path = if (sel_path) |*sp| sp else null,
        .draft_q = draft_q,
    });
}

/// What a DFlash round's verify needs from its draft: the drafts as an int32 `[1, m]` array, the selector's
/// path (its confidences and q) when there is one, and the density they were drawn from when sampled.
pub const VerifyIn = struct {
    t1: u32,
    m: u32,
    m_draft: u32,
    anchor_pos: usize,
    kv_step_snap: usize,
    serial: bool,
    stochastic: bool,
    tracing: bool,
    ph: *io_util.Stopwatch,
    watch: io_util.Stopwatch,
    draft_ids: mlx.mlx_array,
    sel_path: ?*dflash_mod.SelectedPath,
    draft_q: mlx.mlx_array,
};

/// Phases 2-4 of a DFlash round: verify `[t1, drafts]` with the layer taps, accept, roll back and commit.
pub fn dflashVerify(self: *Generator, allocator: std.mem.Allocator, r: VerifyIn) !DrafterStepResult {
    const xfm = self.xfm;
    const s = xfm.s;
    const model = self.dflash.?;
    const dctx = &self.dflash_ctx.?;
    const moe_path = xfm.moe_layers != null or plugin_driver.positionsOf(xfm) == .gdn;
    const hybrid_path = plugin_driver.positionsOf(xfm) == .hybrid;
    const t1 = r.t1;
    const m = r.m;
    const m_draft = r.m_draft;
    const anchor_pos = r.anchor_pos;
    const kv_step_snap = r.kv_step_snap;
    const serial = r.serial;
    const stochastic = r.stochastic;
    const tracing = r.tracing;
    const ph = r.ph;
    const dflash_kv_watch = r.watch;
    const draft_ids = r.draft_ids;
    const sel_path = r.sel_path;
    const draft_q = r.draft_q;
    const bs: u32 = m + 1;

    // ── Phase 2: verify input [t1, drafts...] — [1, bs] int32, lazy ──
    const t1_i32: i32 = @intCast(t1);
    const t1_shape = [_]c_int{ 1, 1 };
    const t1_arr = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(t1_arr);
    var verify_input = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(verify_input);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, t1_arr);
        _ = mlx.mlx_vector_array_append_value(vec, draft_ids);
        try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
    }

    // Verify forward with layer captures — this round's verify IS the
    // next round's context producer.
    const cap_out = try allocator.alloc(mlx.mlx_array, model.config.target_layer_ids.len);
    defer {
        for (cap_out) |a| _ = mlx.mlx_array_free(a);
        allocator.free(cap_out);
    }
    for (cap_out) |*a| a.* = mlx.mlx_array_new();
    var cl = transformer_mod.CaptureLayers{ .ids = model.config.captureIds(), .out = cap_out };
    self.ctx.capture_layers = &cl;
    defer self.ctx.capture_layers = null;
    // Per-position SSM capture on a GDN trunk so partial accept can roll
    // back the recurrent state without re-forwarding (mirrors nextMtp).
    self.ctx.capture_ssm_seq = self.ctx.ssm_entries != null;
    var verify_logits = try xfm.forwardWith(&self.ctx, verify_input);
    errdefer _ = mlx.mlx_array_free(verify_logits);
    self.ctx.capture_ssm_seq = false;
    defer if (self.ctx.ssm_entries) |entries| {
        for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
    };
    self.dflash_attempted += 1;
    if (tracing) {
        // Captures ride the same forward — eval them here or their cost
        // is billed to `append`.
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, verify_logits);
        for (cap_out) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
        try mlx.check(mlx.mlx_eval(vec));
        self.dflash_trace.add(.verify, ph.read());
        ph.reset();
    }

    // ── Phase 3: decide the longest accepted prefix ──
    const vl_shape = mlx.getShape(verify_logits);
    var per_pos_logits: ?[]mlx.mlx_array = null;
    var per_pos_logits_n: usize = 0;
    defer if (per_pos_logits) |slots| {
        for (slots[0..per_pos_logits_n]) |arr| _ = mlx.mlx_array_free(arr);
        allocator.free(slots);
    };
    if (stochastic) {
        const slots = try allocator.alloc(mlx.mlx_array, bs);
        per_pos_logits = slots;
        const slice_strides = [_]c_int{ 1, 1, 1 };
        for (slots, 0..) |*slot, idx| {
            slot.* = mlx.mlx_array_new();
            per_pos_logits_n = idx + 1;
            const start = [_]c_int{ 0, @intCast(idx), 0 };
            const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
            try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
        }
    }
    // Committed entries: masked like every other block decoder's.
    var verify_argmax: CommittedArgmax = .{};
    defer verify_argmax.deinit();
    if (!stochastic) {
        verify_argmax = if (serial)
            try self.verifySerialSamples(verify_logits, 1 + m)
        else
            try verifyArgmax(verify_logits, self.sampling.suppress_mask, s);
    }
    _ = mlx.mlx_array_free(verify_logits);
    verify_logits = .{ .ctx = null };

    // One batched dispatch, then read (drafts branch + verify branch are
    // separate graphs — eval BOTH before reading either; the v26.5.6
    // 0%-acceptance class).
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        _ = mlx.mlx_vector_array_append_value(eval_vec, draft_ids);
        if (!stochastic) _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax.lazy());
        try mlx.check(mlx.mlx_async_eval(eval_vec));
    }
    var drafts = try allocator.alloc(u32, m);
    errdefer allocator.free(drafts);
    if (sel_path) |sp| {
        @memcpy(drafts, sp.ids[0..m]);
    } else {
        try mlx.check(mlx.mlx_array_eval(draft_ids));
        const draft_data = mlx.mlx_array_data_int32(draft_ids) orelse return error.MlxArrayDataNull;
        for (drafts, 0..) |*d, idx| d.* = @intCast(draft_data[idx]);
    }
    if (!stochastic) try mlx.check(mlx.mlx_array_eval(verify_argmax.lazy()));

    var accepted: u32 = 0;
    if (stochastic) {
        var k: u32 = 0;
        while (k < m) : (k += 1) {
            const target_p = try probsAtLastPos(per_pos_logits.?[k], self.sampling, s);
            defer _ = mlx.mlx_array_free(target_p);
            const p_draft = try probAt(target_p, drafts[k], s);
            const accept_prob: f32 = if (draft_q.ctx != null) blk: {
                const q_row = try sliceProbRow(draft_q, k, s);
                defer _ = mlx.mlx_array_free(q_row);
                break :blk specAcceptProb(p_draft, try probAt(q_row, drafts[k], s));
            } else if (sel_path) |sp| blk: {
                // Selector-sampled draft: q is the traced step's own
                // candidate softmax — exact, no GPU read.
                const kk = sp.cand_ids.len / @as(usize, m_draft);
                break :blk specAcceptProb(p_draft, sp.q.?[@as(usize, k) * kk + sp.chosen_idx[k]]);
            } else @min(1.0, p_draft);
            const u: f32 = self.prng.random().float(f32);
            if (u >= accept_prob) break;
            accepted += 1;
        }
    } else {
        const argmax_data = try verify_argmax.ids(1 + m);
        var k: u32 = 0;
        while (k < m) : (k += 1) {
            if (@as(u32, @intCast(argmax_data[k])) != drafts[k]) break;
            accepted += 1;
        }
    }

    // A speculative verify may accept the whole block even when only a
    // few output positions remain. Cap the accepted prefix BEFORE cache
    // and assistant-context commit so every representation of the turn —
    // returned tokens, generated_ids, usage, KV and DFlash context — ends
    // at exactly max_tokens.
    accepted = capAcceptedForTokenBudget(
        accepted,
        self.completion_tokens,
        self.max_tokens,
    );
    if (serial) self.sampling.draw = self.generated_ids.items.len + accepted + 2;

    const next_pending: u32 = blk: {
        if (stochastic) {
            const correction_logits = per_pos_logits.?[accepted];
            const probs = try probsAtLastPos(correction_logits, self.sampling, s);
            defer _ = mlx.mlx_array_free(probs);
            if (accepted < m) {
                // norm(max(0, p − q)) against the q the draft was drawn
                // from — a one-hot here is the WRONG residual for a
                // sampled draft, and wrong silently.
                const q_row = if (draft_q.ctx != null)
                    try sliceProbRow(draft_q, accepted, s)
                else if (sel_path) |sp|
                    try selectorQRow(sp, accepted, m_draft, vl_shape[2], s)
                else
                    try pldOneHotRow(drafts[accepted], vl_shape[2], s);
                defer _ = mlx.mlx_array_free(q_row);
                break :blk try sampleResidual(probs, q_row, self.correctionDraw(accepted), s);
            } else {
                break :blk try sampleFromProbs(probs, self.correctionDraw(accepted), s);
            }
        } else {
            const argmax_data = try verify_argmax.ids(1 + m);
            break :blk @intCast(argmax_data[accepted]);
        }
    };

    if (tracing) {
        self.dflash_trace.add(.accept, ph.read());
        ph.reset();
    }

    // ── Phase 4: commit — truncate on partial accept, then grow the
    // assistant context by exactly the committed positions ──
    const n_commit: usize = 1 + @as(usize, accepted);
    if (accepted < m) {
        try self.ctx.cache.truncate(anchor_pos + n_commit, s);
        if (moe_path) {
            // Same bookkeeping as nextMtp's GDN arm: preserve the
            // pre-verify cache.step (prefix-cache kv_step contract),
            // roll every linear layer's recurrent state back to the
            // accepted position from the verify pass's capture, and
            // re-point moe_seq_offset at the committed length.
            self.ctx.cache.step = kv_step_snap;
            if (self.ctx.ssm_entries) |entries| {
                const gdn_captured = entries.len > 0 and entries[0].spec_state_seq.ctx != null;
                if (!gdn_captured) return error.SpecRollbackUnavailable;
                try self.rollbackSsmFromCapture(entries, accepted, 1 + m, s);
            }
            self.ctx.moe_seq_offset.* = anchor_pos + n_commit;
        } else if (hybrid_path) {
            if (self.ctx.ssm_entries) |entries| {
                // On a hybrid trunk only the CONV layers hold state, so
                // entry 0 may legitimately be an attention layer with no
                // capture at all — ask whether ANY layer recorded one.
                var captured = false;
                for (entries) |*e| {
                    if (e.spec_conv_input.ctx != null or e.spec_state_seq.ctx != null) {
                        captured = true;
                        break;
                    }
                }
                if (!captured) return error.SpecRollbackUnavailable;
                try self.rollbackSsmFromCapture(entries, accepted, bs, s);
            }
            self.ctx.moe_seq_offset.* = anchor_pos + n_commit;
        }
    }
    if (accepted == m) {
        try dflash_mod.appendContext(model, dctx, cap_out, anchor_pos);
    } else {
        const sliced = try allocator.alloc(mlx.mlx_array, cap_out.len);
        var sliced_n: usize = 0;
        defer {
            for (sliced[0..sliced_n]) |a| _ = mlx.mlx_array_free(a);
            allocator.free(sliced);
        }
        for (cap_out, sliced) |full, *out| {
            out.* = mlx.mlx_array_new();
            sliced_n += 1;
            const fsh = mlx.getShape(full);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ 1, @intCast(n_commit), fsh[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(out, full, &start, 3, &stop, 3, &strides, 3, s));
        }
        try dflash_mod.appendContext(model, dctx, sliced, anchor_pos);
    }
    // Materialize the appended context off the critical path so next
    // round's assistant forward doesn't drag this round's verify graph.
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        dctx.appendEvalArrays(eval_vec);
        if (tracing) {
            try mlx.check(mlx.mlx_eval(eval_vec));
            self.dflash_trace.add(.append, ph.read());
            ph.reset();
        } else {
            try mlx.check(mlx.mlx_async_eval(eval_vec));
        }
    }

    const tokens = try allocator.alloc(u32, n_commit);
    tokens[0] = t1;
    for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;
    try self.generated_ids.append(allocator, t1);
    for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);
    allocator.free(drafts);

    self.dflash_accepted_tokens += accepted;
    self.next_token_id = next_pending;
    self.advanceStep(@intCast(n_commit));
    self.dflash_stats.record(m, accepted);
    if (self.dflash_pol) |*pol| {
        if (sel_path) |sp| {
            for (sp.conf[0..m], 0..) |c, k| {
                if (k > accepted) break;
                pol.calib.observe(c, k < accepted);
            }
            if (log.enabled(.debug)) self.logDflashRound(sp.conf[0..m_draft], m, accepted);
        }
    }
    self.dflashRecordDrafted(n_commit, dflash_kv_watch.read());
    // The calibrated sticky gate stays as the bootstrap that gets serial
    // MEASURED (the chooser picks serial only from a measured w0 cell).
    self.checkDflashRuntimeGate();
    if (self.completion_tokens >= self.max_tokens) {
        self.done = true;
        self.finish_reason = "length";
    }

    if (tracing) {
        self.dflashTraceRoundEnd(accepted);
        self.dflash_gap_watch = io_util.Stopwatch.init(self.timer.io);
    }

    return DrafterStepResult{
        .tokens = tokens,
        .accepted_tokens = accepted,
    };
}

/// Longest copied continuation a round verifies: the fused row kernels reach 16 rows.
pub const DFLASH_COPY_MAX: u32 = dflash_policy.MAX_DRAFTS;

/// A round whose drafts are the context's own continuation after `t1` (an echo, an edit, a repeated block):
/// no assistant forward, and as many rows as the cost ladder and the copies' record pay for.
pub fn dflashCopyRound(self: *Generator, allocator: std.mem.Allocator, t1: u32, anchor_pos: usize, kv_step_snap: usize, serial: bool, stochastic: bool, tracing: bool, ph: *io_util.Stopwatch, watch: io_util.Stopwatch) !?DrafterStepResult {
    const pol = &self.dflash_pol.?;
    const idx = try self.mtpLookupIndex(allocator);
    const found = idx.matchBest(t1, DFLASH_COPY_MAX) orelse return null;
    if (found.suffix < mtp_lookup.MIN_SUFFIX or found.draft.len == 0) return null;
    var p: [dflash_policy.MAX_DRAFTS]f32 = undefined;
    const avail: usize = @min(found.draft.len, p.len, @as(usize, self.max_tokens -| self.completion_tokens -| 1));
    if (avail == 0) return null;
    @memset(p[0..avail], pol.copy.probability(found.suffix));
    const costs = dflash_policy.priorCosts(pol.plain.ms);
    const rows = self.dflashRowsFor(p[0..avail], &costs);
    if (rows < 2) return null;
    const m: u32 = rows - 1;
    var ids: [dflash_policy.MAX_DRAFTS]i32 = undefined;
    for (found.draft[0..m], ids[0..m]) |t, *d| d.* = @intCast(t);
    const draft_ids = mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, @intCast(m) }, 2, .int32);
    defer _ = mlx.mlx_array_free(draft_ids);
    self.dflash_round_width = m;
    const suffix = found.suffix;
    const res = try self.dflashVerify(allocator, .{
        .t1 = t1,
        .m = m,
        .m_draft = m,
        .anchor_pos = anchor_pos,
        .kv_step_snap = kv_step_snap,
        .serial = serial,
        .stochastic = stochastic,
        .tracing = tracing,
        .ph = ph,
        .watch = watch,
        .draft_ids = draft_ids,
        .sel_path = null,
        .draft_q = .{ .ctx = null },
    });
    pol.copy.observe(suffix, m, res.accepted_tokens);
    log.debug("  [dflash-copy] suffix={d} avail={d} drafts={d} accepted={d}\n", .{ suffix, avail, m, res.accepted_tokens });
    self.dflash_copies += 1;
    self.dflash_copy_drafted += m;
    self.dflash_copy_accepted += res.accepted_tokens;
    return res;
}

/// Parked captures per target layer before a plain stretch appends them anyway (a stretch the
/// policy's backoff allows is at most 32 rounds; this bounds the parked arrays whatever the caller does).
pub const DFLASH_PENDING_MAX: u32 = 32;

/// A plain DFlash round: t1 through the trunk alone, the verify forward at one row. Its captures
/// are parked in `dflash_pending` and join the assistant context with the next drafted round's append.
pub fn dflashPlainRound(self: *Generator, allocator: std.mem.Allocator, t1: u32, serial: bool, stochastic: bool) !DrafterStepResult {
    const xfm = self.xfm;
    const s = xfm.s;
    const model = self.dflash.?;
    self.dflash_round_width = 0;

    const t1_i32: i32 = @intCast(t1);
    const t1_arr = mlx.mlx_array_new_data(&t1_i32, &[_]c_int{ 1, 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(t1_arr);
    const caps = try allocator.alloc(mlx.mlx_array, model.config.target_layer_ids.len);
    defer {
        for (caps) |a| _ = mlx.mlx_array_free(a);
        allocator.free(caps);
    }
    for (caps) |*a| a.* = mlx.mlx_array_new();
    var cl = transformer_mod.CaptureLayers{ .ids = model.config.captureIds(), .out = caps };
    self.ctx.capture_layers = &cl;
    self.ctx.capture_ssm_seq = false; // one row never rolls back
    const forward = xfm.forwardWith(&self.ctx, t1_arr);
    self.ctx.capture_layers = null;
    const logits = try forward;
    defer _ = mlx.mlx_array_free(logits);

    // The captures are no dependency of the logits: evaluate them with the next token, or the
    // forward's streams stay alive behind them.
    const eval_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(eval_vec);
    for (caps) |c| _ = mlx.mlx_vector_array_append_value(eval_vec, c);
    var picked: CommittedArgmax = .{};
    defer picked.deinit();
    const next_pending: u32 = blk: {
        if (stochastic) {
            const probs = try probsAtLastPos(logits, self.sampling, s);
            defer _ = mlx.mlx_array_free(probs);
            try mlx.check(mlx.mlx_async_eval(eval_vec));
            break :blk try sampleFromProbs(probs, self.correctionDraw(0), s);
        }
        picked = if (serial) try self.verifySerialSamples(logits, 1) else try verifyArgmax(logits, self.sampling.suppress_mask, s);
        _ = mlx.mlx_vector_array_append_value(eval_vec, picked.lazy());
        try mlx.check(mlx.mlx_async_eval(eval_vec));
        try mlx.check(mlx.mlx_array_eval(picked.lazy()));
        break :blk @intCast((try picked.ids(1))[0]);
    };
    if (serial) self.sampling.draw = self.generated_ids.items.len + 2;

    try self.dflashPark(allocator, caps, 1);

    const tokens = try allocator.alloc(u32, 1);
    errdefer allocator.free(tokens);
    tokens[0] = t1;
    try self.generated_ids.append(allocator, t1);
    self.dflash_plain += 1;
    self.dflash_stats.record(0, 0);
    self.next_token_id = next_pending;
    self.advanceStep(1);
    if (self.completion_tokens >= self.max_tokens) {
        self.done = true;
        self.finish_reason = "length";
    }
    return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
}

/// Park one forward's captures (and take them over: the caller's handles are left empty).
pub fn dflashPark(self: *Generator, allocator: std.mem.Allocator, caps: []mlx.mlx_array, rows: u32) !void {
    try self.dflash_pending.appendSlice(allocator, caps);
    for (caps) |*a| a.* = .{ .ctx = null };
    self.dflash_pending_rounds += 1;
    self.dflash_pending_rows += rows;
    if (self.dflash_pending_rounds >= DFLASH_PENDING_MAX) try self.dflashFlushPending(allocator);
}

/// Whether a plain tick can ride the serial pipeline (`next`'s fast path).
pub fn dflashCanPipeline(self: *const Generator) bool {
    return self.logprobs_n == 0 and !self.sampling.penalized() and self.sampling.constraint == null and self.step + 1 < self.max_tokens;
}

/// A plain tick on the serial pipeline: `next` overlaps the following forward with the host's work,
/// which a round of its own cannot. Its forwards carry the DFlash taps, parked like a plain round's.
pub fn dflashPlainPipelined(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    const model = self.dflash.?;
    const caps = try allocator.alloc(mlx.mlx_array, model.config.target_layer_ids.len);
    defer {
        for (caps) |a| _ = mlx.mlx_array_free(a);
        allocator.free(caps);
    }
    for (caps) |*a| a.* = mlx.mlx_array_new();
    var cl = transformer_mod.CaptureLayers{ .ids = model.config.captureIds(), .out = caps };
    self.ctx.capture_layers = &cl;
    defer self.ctx.capture_layers = null;
    self.ctx.capture_ssm_seq = false; // one row never rolls back
    // The stretch starts from the DFlash invariant (t1 not in the cache): forward it as `next`'s own bootstrap
    // would, here with the taps, so `next` finds the pipeline seeded and runs only its lookahead.
    if (!self.has_pending_logits and !self.has_pending_token) {
        const t1_i32: i32 = @intCast(self.next_token_id);
        const t1_arr = mlx.mlx_array_new_data(&t1_i32, &[_]c_int{ 1, 1 }, 2, .int32);
        defer _ = mlx.mlx_array_free(t1_arr);
        self.pending_logits = try self.xfm.forwardWith(&self.ctx, t1_arr);
        self.has_pending_logits = true;
        try self.dflashEvalPark(allocator, caps);
    }
    const tok = (try self.next(allocator)) orelse return null;
    // The lookahead's forward (none on the budget's last token) is already queued; its taps are small work behind it.
    if (caps[0].ctx != null) try self.dflashEvalPark(allocator, caps);
    const tokens = try allocator.alloc(u32, 1);
    tokens[0] = tok;
    self.dflash_plain += 1;
    self.dflash_stats.record(0, 0);
    // Consecutive pipelined ticks are the machine's plain period; the first of a stretch has no predecessor.
    const pol = &self.dflash_pol.?;
    if (pol.plain_watch) |*w| pol.plain.sample(@as(f32, @floatFromInt(w.read())) / @as(f32, std.time.ns_per_ms));
    pol.plain_watch = io_util.Stopwatch.init(self.timer.io);
    return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
}

/// Evaluate a forward's taps (they are no dependency of its logits) and park them.
pub fn dflashEvalPark(self: *Generator, allocator: std.mem.Allocator, caps: []mlx.mlx_array) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (caps) |c| _ = mlx.mlx_vector_array_append_value(vec, c);
    try mlx.check(mlx.mlx_async_eval(vec));
    try self.dflashPark(allocator, caps, 1);
    for (caps) |*a| a.* = mlx.mlx_array_new();
}

/// A drafted round after a pipelined stretch: land on the spec entry invariant, one token of the stretch emitted.
pub fn dflashLeavePipeline(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    switch (try self.drainPipelineForSpec(allocator)) {
        .drained => |tok| {
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok;
            self.dflash_unobserved = true;
            return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
        },
        .stopped => return null,
        .already_clean, .stay_disabled => return error.PipelineHalfState,
    }
}

/// Append the plain rounds' parked captures to the assistant context as one block.
pub fn dflashFlushPending(self: *Generator, allocator: std.mem.Allocator) !void {
    const rounds = self.dflash_pending_rounds;
    if (rounds == 0) return;
    const model = self.dflash.?;
    const dctx = &self.dflash_ctx.?;
    const n_t = model.config.target_layer_ids.len;
    const merged = try allocator.alloc(mlx.mlx_array, n_t);
    defer {
        for (merged) |a| _ = mlx.mlx_array_free(a);
        allocator.free(merged);
    }
    for (merged) |*a| a.* = mlx.mlx_array_new();
    for (merged, 0..) |*out, t| {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (0..rounds) |r| _ = mlx.mlx_vector_array_append_value(vec, self.dflash_pending.items[r * n_t + t]);
        try mlx.check(mlx.mlx_concatenate_axis(out, vec, 1, self.xfm.s));
    }
    try dflash_mod.appendContext(model, dctx, merged, dctx.absLen());
    for (self.dflash_pending.items) |a| _ = mlx.mlx_array_free(a);
    self.dflash_pending.clearRetainingCapacity();
    self.dflash_pending_rounds = 0;
    self.dflash_pending_rows = 0;
    const eval_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(eval_vec);
    dctx.appendEvalArrays(eval_vec);
    try mlx.check(mlx.mlx_async_eval(eval_vec));
}

/// The assistant context's length once the parked captures are counted: the trunk's cursor, always.
pub fn dflashContextLen(self: *const Generator) usize {
    return self.dflash_ctx.?.absLen() + self.dflash_pending_rows;
}

/// Bring the assistant context to the position `covers` (the committed prefix): parked captures appended, and a
/// pipelined lookahead's position dropped. A prefix-cache commit pairs the two only when they end together.
pub fn dflashSettle(self: *Generator, allocator: std.mem.Allocator, covers: usize) !void {
    const dctx = if (self.dflash_ctx) |*d| d else return;
    try self.dflashFlushPending(allocator);
    if (dctx.absLen() > covers and covers >= dctx.base_pos) try dctx.cache.truncate(covers - dctx.base_pos, self.xfm.s);
}

/// Tell the drafting backoff what a drafted round bought: the plain rounds its tokens would have cost, and what it took.
pub fn dflashRecordDrafted(self: *Generator, tokens: usize, wall_ns: u64) void {
    const pol = if (self.dflash_pol) |*p| p else return;
    const wall_ms = @as(f32, @floatFromInt(wall_ns)) / @as(f32, std.time.ns_per_ms);
    pol.drafting.record(@as(f32, @floatFromInt(tokens)) * self.dflashPlainMs(), wall_ms);
}

/// ms of a plain round for this request (policy rounds only).
pub fn dflashPlainMs(self: *const Generator) f32 {
    return self.dflash_pol.?.plain.ms;
}

/// Rows to verify (t1 included) for drafts with acceptance probabilities `p`: the pin, else the best tokens per ms.
pub fn dflashRowsFor(self: *const Generator, p: []const f32, costs: *const dflash_policy.Costs) u32 {
    const pol = &self.dflash_pol.?;
    if (pol.pin_rows > 0) return @min(pol.pin_rows, @as(u32, @intCast(p.len)) + 1);
    return dflash_policy.chooseRows(p, costs).rows;
}

/// Rows to verify (t1 included) for a draft whose per-position confidence the selector reports.
pub fn dflashPolicyRows(self: *Generator, sel: *const dflash_mod.SelectedPath, m_draft: u32) u32 {
    const pol = &self.dflash_pol.?;
    var p: [dflash_policy.MAX_DRAFTS]f32 = undefined;
    for (sel.conf[0..m_draft], p[0..m_draft]) |c, *out| out.* = pol.calib.probability(c);
    const costs = dflash_policy.priorCosts(self.dflashPlainMs());
    return self.dflashRowsFor(p[0..m_draft], &costs);
}

/// `[dflash-round]`: what the selector was sure of against what the target kept (the calibration's raw material).
pub fn logDflashRound(self: *const Generator, conf: []const f32, rows_drafts: u32, accepted: u32) void {
    _ = self;
    var buf: [160]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    for (conf, 0..) |c, k| w.print("{s}{d:.2}", .{ if (k == 0) "" else "/", c }) catch break;
    log.debug("  [dflash-round] verified={d}/{d} accepted={d} conf={s}\n", .{ rows_drafts, conf.len, accepted, w.buffered() });
}

// ── tests ──


test "lilicorr missing restored anchor takes a plain round and recovers it" {
    if (mlx.noGpuBackend()) return;
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var trunk_tmp = testing.tmpDir(.{});
    defer trunk_tmp.cleanup();
    var asst_tmp = testing.tmpDir(.{});
    defer asst_tmp.cleanup();
    var trunk_buf: [512]u8 = undefined;
    const trunk_len = try trunk_tmp.dir.realPath(io, &trunk_buf);
    const trunk_path = trunk_buf[0..trunk_len];
    var asst_buf: [512]u8 = undefined;
    const asst_len = try asst_tmp.dir.realPath(io, &asst_buf);
    const asst_path = asst_buf[0..asst_len];
    try dflash_mod.TinyFix.writeTrunk(io, trunk_tmp.dir, trunk_path, s);
    try dflash_mod.TinyFix.writeAssistant(io, asst_tmp.dir, asst_path, s);
    var cfg = try model_mod.parseConfig(io, a, trunk_path);
    defer cfg.deinit(a);
    var xfm = try plugin_driver.load(io, a, &cfg, trunk_path, mlx.gpuStream());
    defer xfm.deinit();
    var dm = try dflash_mod.loadDflash(io, a, s, asst_path);
    defer dm.deinit();
    try dm.bind(&xfm);
    var tok: Tokenizer = undefined;
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5 };
    var gen = try Generator.initWithOptions(io, a, &xfm, &tok, &prompt, 8, .{ .temperature = 0 }, &.{}, .{
        .dflash_enabled = true,
        .dflash = &dm,
        .dflash_min_accepted_per_round = 0,
    });
    defer gen.deinit(a);
    // A restored context has its K/V but no LiLiCorr anchor row.
    try asst_tmp.dir.writeFile(io, .{ .sub_path = "fixture.safetensors", .data = @embedFile("../../../fixtures/lilicorr_fp32.safetensors") });
    const path = try std.fmt.allocPrint(a, "{s}/fixture.safetensors", .{asst_path});
    defer a.free(path);
    var fx = try model_mod.loadWeightsSingleFile(a, path);
    defer fx.deinit();
    dm.lilicorr = try lilicorr_mod.load(a, &fx, s, .{ .top_k = 3, .hidden = 8, .trunk = 12, .heads = 2, .slots = 3, .layers = 2, .factor = 6, .vector_eps = 0.0001, .logit_scale = 8, .rms_eps = 1e-6 });
    try testing.expect(gen.dflash_ctx.?.anchor.ctx == null);
    const pending = gen.next_token_id;
    const step = (try gen.nextDflash(a)).?;
    defer a.free(step.tokens);
    try testing.expectEqualSlices(u32, &[_]u32{pending}, step.tokens);
    try testing.expectEqual(@as(u32, 0), step.accepted_tokens);
    try testing.expectEqual(@as(u32, 1), gen.dflash_plain);
    try testing.expect(!gen.spec_disabled_runtime);
    try gen.dflashFlushPending(a);
    try testing.expect(gen.dflash_ctx.?.anchor.ctx != null);
    try testing.expectEqual(gen.ctx.cache.step, gen.dflash_ctx.?.absLen());
}

test "dflash: nextDflash greedy equals serial decode, invariants exact each round" {
    // Hermetic end-to-end round loop on the tiny llama trunk + tiny DFlash
    // assistant (dflash.TinyFix). The bar is the spec-decode contract:
    //   - greedy dflash-on emits EXACTLY the serial greedy tokens (any bug in
    //     verify alignment, rollback truncate, correction indexing, or the
    //     anchor-row drop shifts the stream);
    //   - after EVERY round: trunk cache.step == prompt_len + emitted and the
    //     assistant context tracks it exactly (base + step == cache.step) at
    //     whatever accepted counts the round produced.
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_trunk = std.testing.tmpDir(.{});
    defer tmp_trunk.cleanup();
    var trunk_buf: [512]u8 = undefined;
    const trunk_len = try tmp_trunk.dir.realPath(io, &trunk_buf);
    const trunk_path = trunk_buf[0..trunk_len];
    try dflash_mod.TinyFix.writeTrunk(io, tmp_trunk.dir, trunk_path, s);

    var tmp_asst = std.testing.tmpDir(.{});
    defer tmp_asst.cleanup();
    var asst_buf: [512]u8 = undefined;
    const asst_len = try tmp_asst.dir.realPath(io, &asst_buf);
    const asst_path = asst_buf[0..asst_len];
    try dflash_mod.TinyFix.writeAssistant(io, tmp_asst.dir, asst_path, s);

    var config = try model_mod.parseConfig(io, allocator, trunk_path);
    defer config.deinit(allocator);

    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
    const greedy = SamplingParams{ .temperature = 0.0 };
    const want: usize = 16;

    // Serial baseline.
    var serial: [want]u32 = undefined;
    {
        var xfm = try plugin_driver.load(io, allocator, &config, trunk_path, mlx.gpuStream());
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, @intCast(want), greedy, &.{}, .{
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

    // DFlash arm.
    {
        var xfm = try plugin_driver.load(io, allocator, &config, trunk_path, mlx.gpuStream());
        defer xfm.deinit();
        var dm = try dflash_mod.loadDflash(io, allocator, s, asst_path);
        defer dm.deinit();
        try dm.bind(&xfm);

        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, @intCast(want), greedy, &.{}, .{
            .dflash_enabled = true,
            .dflash = &dm,
            // This is the numeric DFlash guard, not the economics-gate test:
            // every emitted token must pass through nextDflash itself.
            .dflash_min_accepted_per_round = 0,
        });
        defer gen.deinit(allocator);
        try testing.expectEqual(@as(u32, 4), gen.dflash_block_size);
        // Prefill built context for the whole prompt.
        try testing.expectEqual(prompt.len, gen.dflash_ctx.?.absLen());
        try testing.expectEqual(prompt.len, gen.ctx.cache.step);

        var got = std.ArrayList(u32).empty;
        defer got.deinit(allocator);
        while (true) {
            const attempts_before = gen.dflash_attempted;
            // The budget's last token is the previous verify's: it leaves without a round.
            const last = gen.max_tokens -| gen.completion_tokens == 1;
            const res = (try gen.nextDflash(allocator)) orelse break;
            defer allocator.free(res.tokens);
            try got.appendSlice(allocator, res.tokens);
            // A real DFlash round commits trunk and assistant context to the
            // same exact boundary at every accepted count. Falling back here
            // would make the remaining equality checks compare serial decode
            // with itself and gut the default-on draft-quantization guard.
            try testing.expect(last or gen.dflash_attempted != attempts_before);
            try testing.expectEqual(prompt.len + gen.generated_ids.items.len - gen.unforwarded_tail, gen.ctx.cache.step);
            try testing.expectEqual(gen.ctx.cache.step, gen.dflash_ctx.?.absLen());
        }
        try testing.expect(gen.dflash_attempted > 0);
        try testing.expect(!gen.spec_disabled_runtime);
        try testing.expectEqual(want, got.items.len);
        try testing.expectEqual(@as(u32, @intCast(want)), gen.completion_tokens);
        try testing.expectEqual(want, gen.generated_ids.items.len);
        for (serial, got.items) |a, b| try testing.expectEqual(a, b);
    }

    // DFlash2 arm: selector-traced greedy drafts + dyn-conv forward. The bar
    // is unchanged — a selector draft only survives verify when it IS the
    // trunk argmax, so the emitted stream must still be byte-identical to
    // serial at every accepted count.
    {
        var tmp_a2 = std.testing.tmpDir(.{});
        defer tmp_a2.cleanup();
        var a2_buf: [512]u8 = undefined;
        const a2_path = a2_buf[0..try tmp_a2.dir.realPath(io, &a2_buf)];
        try dflash_mod.TinyFix.writeAssistant2(io, tmp_a2.dir, a2_path, s, false);

        var xfm = try plugin_driver.load(io, allocator, &config, trunk_path, mlx.gpuStream());
        defer xfm.deinit();
        var dm = try dflash_mod.loadDflash(io, allocator, s, a2_path);
        defer dm.deinit();
        try testing.expect(dm.selector != null);
        try dm.bind(&xfm);

        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, @intCast(want), greedy, &.{}, .{
            .dflash_enabled = true,
            .dflash = &dm,
            .dflash_min_accepted_per_round = 0,
        });
        defer gen.deinit(allocator);

        var got = std.ArrayList(u32).empty;
        defer got.deinit(allocator);
        while (true) {
            const attempts_before = gen.dflash_attempted;
            // The budget's last token is the previous verify's: it leaves without a round.
            const last = gen.max_tokens -| gen.completion_tokens == 1;
            const res = (try gen.nextDflash(allocator)) orelse break;
            defer allocator.free(res.tokens);
            try got.appendSlice(allocator, res.tokens);
            try testing.expect(last or gen.dflash_attempted != attempts_before);
            try testing.expectEqual(prompt.len + gen.generated_ids.items.len - gen.unforwarded_tail, gen.ctx.cache.step);
            try testing.expectEqual(gen.ctx.cache.step, gen.dflash_ctx.?.absLen());
        }
        try testing.expect(gen.dflash_attempted > 0);
        try testing.expectEqual(want, got.items.len);
        for (serial, got.items) |a, b| try testing.expectEqual(a, b);
    }
}

const PolicyArm = struct {
    /// A DFlash2 assistant (selector confidence) instead of the v1 block drafter.
    v2: bool,
    /// Verify exactly this many rows whenever the draft has them (0 = the policy chooses).
    pin_rows: u32 = 0,
    /// Pre-train the calibration to trust every draft, so the policy verifies the whole block.
    trust: bool = false,
    sampled: bool = false,
    /// A prompt that ends mid-repeat, so the first round copies the context's continuation.
    copy: bool = false,
};

/// Ten distinct ids repeated, then the first nine again: `COPY_PATTERN[9]` as t1 continues the repeat.
const COPY_PATTERN = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17 };

/// One request through `nextDflash` under the round policy on the tiny fixture pair, three plain
/// rounds first: the trunk and assistant cursors agree after EVERY round whatever mix of plain and
/// drafted rounds ran, and a greedy stream is the serial stream. Returns the generator's stats.
fn policyArm(arm: PolicyArm, serial: []const u32) !dflash_policy.PositionStats {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_trunk = std.testing.tmpDir(.{});
    defer tmp_trunk.cleanup();
    var trunk_buf: [512]u8 = undefined;
    const trunk_path = trunk_buf[0..try tmp_trunk.dir.realPath(io, &trunk_buf)];
    try dflash_mod.TinyFix.writeTrunk(io, tmp_trunk.dir, trunk_path, s);
    var config = try model_mod.parseConfig(io, allocator, trunk_path);
    defer config.deinit(allocator);

    var tmp_asst = std.testing.tmpDir(.{});
    defer tmp_asst.cleanup();
    var asst_buf: [512]u8 = undefined;
    const asst_path = asst_buf[0..try tmp_asst.dir.realPath(io, &asst_buf)];
    if (arm.v2) try dflash_mod.TinyFix.writeAssistant2(io, tmp_asst.dir, asst_path, s, false) else try dflash_mod.TinyFix.writeAssistant(io, tmp_asst.dir, asst_path, s);

    var xfm = try plugin_driver.load(io, allocator, &config, trunk_path, mlx.gpuStream());
    defer xfm.deinit();
    var dm = try dflash_mod.loadDflash(io, allocator, s, asst_path);
    defer dm.deinit();
    try dm.bind(&xfm);

    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const copy_prompt = COPY_PATTERN ++ COPY_PATTERN ++ COPY_PATTERN[0..9].*;
    const plain_prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
    const prompt: []const u32 = if (arm.copy) &copy_prompt else &plain_prompt;
    var copy_serial: [16]u32 = undefined;
    if (arm.copy) {
        var sx = try plugin_driver.load(io, allocator, &config, trunk_path, mlx.gpuStream());
        defer sx.deinit();
        var sg = try Generator.initWithOptions(io, allocator, &sx, &tok_dummy, prompt, 16, SamplingParams{ .temperature = 0.0 }, &.{}, .{ .skip_lazy_preforward = true });
        defer sg.deinit(allocator);
        sg.next_token_id = COPY_PATTERN[9];
        for (&copy_serial) |*t| t.* = (try sg.next(allocator)).?;
    }
    const reference: []const u32 = if (arm.copy) &copy_serial else serial;
    const sampling = if (arm.sampled) SamplingParams{ .temperature = 1.0, .seed = 7 } else SamplingParams{ .temperature = 0.0 };
    const want: u32 = 16;
    var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, prompt, want, sampling, &.{}, .{
        .dflash_enabled = true,
        .dflash = &dm,
        .dflash_policy = true,
    });
    defer gen.deinit(allocator);
    const pol = &gen.dflash_pol.?;
    pol.pin_rows = arm.pin_rows;
    pol.drafting.owed = if (arm.copy) 0 else 3;
    if (arm.copy) gen.next_token_id = COPY_PATTERN[9];
    const plain_rounds: usize = if (arm.copy) 0 else 3;
    if (arm.trust) for (0..10) |b| for (0..40) |_| pol.calib.observe((@as(f32, @floatFromInt(b)) + 0.5) / 10.0, true);

    var got = std.ArrayList(u32).empty;
    defer got.deinit(allocator);
    var round: usize = 0;
    while (true) : (round += 1) {
        const plain_before = gen.dflash_plain;
        const attempts_before = gen.dflash_attempted;
        const last = gen.max_tokens -| gen.completion_tokens == 1;
        const res = (try gen.nextDflash(allocator)) orelse break;
        defer allocator.free(res.tokens);
        try got.appendSlice(allocator, res.tokens);
        if (round < plain_rounds) {
            try testing.expectEqual(@as(usize, 1), res.tokens.len);
            try testing.expectEqual(plain_before + 1, gen.dflash_plain);
            try testing.expectEqual(attempts_before, gen.dflash_attempted);
        }
        // A plain stretch rides the serial pipeline: one lookahead token is in the trunk's cache, unemitted.
        const lookahead: usize = if (gen.has_pending_logits or gen.has_pending_token) 1 else 0;
        try testing.expectEqual(prompt.len + gen.generated_ids.items.len - gen.unforwarded_tail + lookahead, gen.ctx.cache.step);
        // The assistant context reaches the trunk's cursor, appended or still parked.
        try testing.expectEqual(gen.ctx.cache.step, gen.dflashContextLen());
        // A drafted round appends whatever the plain ones left parked.
        if (gen.dflash_attempted != attempts_before) try testing.expectEqual(gen.ctx.cache.step, gen.dflash_ctx.?.absLen());
        _ = last;
    }
    try testing.expect(gen.dflash_plain >= plain_rounds);
    if (arm.copy) try testing.expect(gen.dflash_copies > 0);
    try testing.expect(!gen.spec_disabled_runtime);
    try testing.expectEqual(@as(usize, want), got.items.len);
    if (!arm.sampled) for (reference, got.items) |a, b| try testing.expectEqual(a, b);
    // The scheduler settles the context to the committed prefix, whatever lookahead the run ended on.
    const committed = prompt.len + gen.generated_ids.items.len - gen.unforwarded_tail;
    try gen.dflashSettle(allocator, committed);
    try testing.expectEqual(committed, gen.dflash_ctx.?.absLen());
    return gen.dflash_stats;
}

test "dflash: the round policy runs plain rounds between drafted ones, the stream stays serial's" {
    // A plain round is the verify forward at one row (no assistant forward): it must advance the trunk
    // and the assistant context exactly like a drafted round, so DFlash can resume after any stretch of
    // them; a draft the policy verifies only in part must not shift the accept, rollback or q indexing.
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_trunk = std.testing.tmpDir(.{});
    defer tmp_trunk.cleanup();
    var trunk_buf: [512]u8 = undefined;
    const trunk_path = trunk_buf[0..try tmp_trunk.dir.realPath(io, &trunk_buf)];
    try dflash_mod.TinyFix.writeTrunk(io, tmp_trunk.dir, trunk_path, s);
    var config = try model_mod.parseConfig(io, allocator, trunk_path);
    defer config.deinit(allocator);
    var tok_dummy: Tokenizer = undefined;
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
    var serial: [16]u32 = undefined;
    {
        var xfm = try plugin_driver.load(io, allocator, &config, trunk_path, mlx.gpuStream());
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, 16, SamplingParams{ .temperature = 0.0 }, &.{}, .{ .skip_lazy_preforward = true });
        defer gen.deinit(allocator);
        for (&serial) |*t| t.* = (try gen.next(allocator)).?;
    }

    // The v1 block drafter verifies its whole block.
    const v1 = try policyArm(.{ .v2 = false }, &serial);
    try testing.expect(v1.rows[4] > 0);
    // The DFlash2 selector: two rows of a three-draft block, then the whole block once it is trusted.
    const two = try policyArm(.{ .v2 = true, .pin_rows = 2 }, &serial);
    try testing.expect(two.rows[2] > 0 and two.rows[3] == 0 and two.rows[4] == 0);
    const trusted = try policyArm(.{ .v2 = true, .trust = true }, &serial);
    try testing.expect(trusted.rows[4] > 0);
    // A copied continuation (no assistant forward) is verified like any draft.
    _ = try policyArm(.{ .v2 = false, .copy = true }, &serial);
    _ = try policyArm(.{ .v2 = false, .copy = true, .sampled = true }, &serial);
    // Sampled requests: the Leviathan accept reads q at the draft's own stride, whatever was kept.
    _ = try policyArm(.{ .v2 = true, .pin_rows = 2, .sampled = true }, &serial);
    _ = try policyArm(.{ .v2 = true, .trust = true, .sampled = true }, &serial);
}
