//! MTP draft chain state: history stash, pre-drafts, prompt lookup, lazy pre-draft, padded head history.
const depth_bounds = @import("../mtp_depth_bounds.zig");
const HeadPlace = @import("../../../../plugins/qwen4_exp/mtp.zig").HeadPlace;
const std = @import("std");
const env = @import("../../../../env.zig");
const mlx = @import("../../../../mlx/ffi.zig");
const transformer_mod = @import("../../transformer.zig");
const tokenizer_mod = @import("../../../../sdk/tokenizer/tokenizer.zig");
const model_mod = @import("../../../../sdk/config/model.zig");
const log = @import("../../../../sdk/util/log.zig");
const mtp_lookup = @import("../mtp_lookup.zig");
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");
const mtp_acceptance = @import("../../../../sdk/spec/mtp_acceptance.zig");
const group_planner = @import("../mtp_group_planner.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const ForwardCtx = transformer_mod.ForwardCtx;
const KVCache = transformer_mod.KVCache;
const testing = std.testing;
const CommittedArgmax = @import("../verify.zig").CommittedArgmax;
const Generator = @import("../../generate/generator.zig").Generator;
const MAX_ROUND_DRAFTS = @import("../mtp_head_ref.zig").MAX_ROUND_DRAFTS;
const MtpGraphFn = @import("../tree.zig").MtpGraphFn;
const MtpRoundPlan = @import("ev_plan.zig").MtpRoundPlan;
const MtpRoundState = @import("round.zig").MtpRoundState;
const SamplingParams = @import("../../sampling/params.zig").SamplingParams;
const deinit = @import("../../generate/generator.zig").Generator.deinit;
const isEosId = @import("../../../../sdk/request/stops.zig").isEosId;
const mtpBatchedExactGraph = @import("batched_graph.zig").mtpBatchedExactGraph;
const mtpChainDispatch = @import("chain_build.zig").mtpChainDispatch;
const mtpPredraftEnabled = @import("trace.zig").mtpPredraftEnabled;
const sampleTokenLazy = @import("../../sampling/lazy.zig").sampleTokenLazy;


/// Qwen native-MTP speculative round. Structure mirrors `nextDrafter`
/// (same verify invariant: cache.step = prompt_len + emitted, t1 NOT in
/// cache on entry, verify input `[t1, draft[0..m-1]]`, bonus from row m,
/// partial-accept snapshot/restore + re-forward) with one addition: the
/// MTP head's committed-history KV cache. Draft steps append m temporary
/// entries built from MTP-PREDICTED hiddens; after the verify decision we
/// restore the round-boundary snapshot and re-append the committed pairs
/// from TRUE trunk hiddens, so the history never accumulates drift.
pub const MtpHistStash = struct {
    /// `[n]` int32 committed token ids: `[t1, drafts[0..accepted]]`.
    ids: mlx.mlx_array,
    /// `ids` on the host.
    host_ids: [MAX_ROUND_DRAFTS + 1]u32 = undefined,
    /// `[1, n, H]` trunk hiddens paired 1:1 with `ids` (a lazy concat of
    /// last_hidden + a verify-capture slice — the handle pins the ~90 KB
    /// parent until consumed, deliberately NOT a deep copy).
    hidden: mlx.mlx_array,
    n: usize,
    /// Head-cache position of ids[0]'s entry (the producing round's
    /// mtp_off0). The consume-time truncate drops the producing round's
    /// stale draft tail past it.
    off0: usize,
    /// The whole verify row `[t1, drafts[0..m]]` and its `[1, 1+m, H]` hiddens, for a
    /// padded append (`mtpPaddedHistory`); `ids`/`hidden` are its first `n` rows.
    pad: ?struct { ids: mlx.mlx_array, hidden: mlx.mlx_array, rows: usize } = null,

    pub fn deinit(self: *MtpHistStash) void {
        _ = mlx.mlx_array_free(self.ids);
        _ = mlx.mlx_array_free(self.hidden);
        if (self.pad) |p| {
            _ = mlx.mlx_array_free(p.ids);
            _ = mlx.mlx_array_free(p.hidden);
        }
    }
};

/// Round origin of the MTP head cache: with a pending stash the cache
/// still holds the PREVIOUS round's draft tail (its step is stale), so
/// the committed length is the stash origin plus the entries the stash
/// itself will append; without one, the cache is fully committed.
pub fn mtpRoundOff0(stash: ?MtpHistStash, cache_step: usize) usize {
    if (stash) |st| return st.off0 + st.n;
    return cache_step;
}

/// Committed-history length actually IN the head cache at rest — what a
/// prefix-cache commit may snapshot. Everything past it is speculative:
/// a pending stash's entries are NOT in the cache yet (the cache past
/// `stash.off0` is the producing round's stale draft tail), and a built
/// cross-round pre-draft has already appended NEXT-round draft entries
/// past ITS `off0`. The min over both boundaries is always safe.
pub fn mtpCommittedLen(cache_step: usize, pre_draft_off0: ?usize, stash_off0: ?usize) usize {
    var committed = cache_step;
    if (pre_draft_off0) |o| committed = @min(committed, o);
    if (stash_off0) |o| committed = @min(committed, o);
    return committed;
}

/// `mtpCommittedLen` over this Generator's live state; 0 when MTP is
/// not active.
pub fn mtpCommittedHistoryLen(self: *const Generator) usize {
    if (self.mtp_cache == null) return 0;
    return mtpCommittedLen(
        self.mtp_cache.?.step(),
        if (self.mtp_pre_draft) |*pd| pd.off0 else null,
        if (self.mtp_hist_stash) |*st| st.off0 else null,
    );
}

pub fn mtpMropeContext(self: *const Generator) ?mtp_mod.MropeContext {
    const positions = self.ctx.mrope_pos orelse return null;
    return .{
        .pos = positions,
        .total = self.ctx.mrope_total,
        .delta = self.ctx.mrope_delta,
        .base = self.mtp_position_base,
    };
}

/// The serial sampler's draw for every verify row: row i is the token at
/// generated index G + 1 + i (t1 sits at G), keyed by that index.
pub fn verifySerialSamples(self: *const Generator, logits: mlx.mlx_array, rows: usize) !CommittedArgmax {
    return self.verifySerialSamplesAt(logits, rows, null);
}

/// `verifySerialSamples` for a tree's rows: row i is the token at generated
/// index G + 1 + depth[i].
pub fn verifySerialSamplesAt(self: *const Generator, logits: mlx.mlx_array, rows: usize, depth: ?[]const i32) !CommittedArgmax {
    const s = self.xfm.s;
    const vocab = mlx.getShape(logits)[2];
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    const base = self.generated_ids.items.len + 1;
    for (0..rows) |i| {
        const ri: c_int = @intCast(i);
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, logits, &[_]c_int{ 0, ri, 0 }, 3, &[_]c_int{ 1, ri + 1, vocab }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var sp = self.sampling;
        sp.draw = base + (if (depth) |d| @as(usize, @intCast(d[i])) else i);
        const tok = sampleTokenLazy(row, sp, s);
        defer _ = mlx.mlx_array_free(tok);
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, tok, &[_]c_int{1}, 1, s));
        _ = mlx.mlx_vector_array_append_value(vec, flat);
    }
    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, cat, &[_]c_int{ 1, @intCast(rows) }, 2, s));
    return .{ .arr = out };
}

/// Serial-accept rounds of a sampled request draft coupled to the serial draws.
pub fn mtpCoupledDrafts(self: *const Generator) bool {
    return self.mtp_serial_accept and self.sampling.keyed and self.sampling.temperature > 0.01 and self.sampling.seed != null;
}

/// Allocate one request-local categorical draw index while the lazy MTP
/// graph is built. The same counter covers draft and correction samples,
/// including cross-round pre-drafts, so no two keyed draws reuse a key.
/// An argmax draft draws nothing.
pub fn mtpSamplingDraw(self: *Generator, params: SamplingParams) SamplingParams {
    if (params.temperature <= 0.01) return params;
    var draw = params;
    draw.draw = self.sampling.draw;
    self.sampling.draw +%= 1;
    return draw;
}

/// One round's lazily-built MTP draft chain — the Phase 0/1 state that
/// cross-round pre-drafting (`mtpMaybePreDraft`) moves into the PREVIOUS
/// round's tail. Owns every handle it holds; `deinit` frees whatever was
/// built so far, so it is safe on partial builds and on a pre-draft left
/// unconsumed at request end.
pub const MtpPreDraft = struct {
    plan: MtpRoundPlan,
    /// Head-cache position of this round's first draft entry.
    off0: usize,
    t1: u32,
    /// `[1]` int32 t1 — the verify input needs it again in Phase 3.
    t1_arr: mlx.mlx_array,
    /// Host draft ids, filled at the Phase 4b sync. len = plan.m_hi.
    drafts: []u32,
    /// Lazy `[1]` int32 draft ids. len = plan.m_hi; [0..n_drafted) valid.
    draft_arrs: []mlx.mlx_array,
    n_drafted: u32,
    /// Generated index the first draft proposes (a coupled draft's key).
    pos0: usize = 0,
    head_input_rows: u32 = 1,
    /// Chunk-A log-confidence graphs (two-chunk plans only). len m_lo.
    conf_arrs: ?[]mlx.mlx_array,
    n_conf: u32,
    /// Proposal densities q, present when depth 0 samples; an argmax
    /// depth's row is its one-hot. len m_hi; [0..n_qp) valid.
    q_probs: ?[]mlx.mlx_array,
    n_qp: u32,
    /// Chain hidden after the last built step (owned) — a chunk-B
    /// extension resumes from it; freed once the chain is complete.
    h_chain: ?mlx.mlx_array,
    /// Tokens actually drafted this round; starts at m_lo, grows to
    /// m_hi iff the confidence gate clears at the chunk boundary.
    m: u32,
    /// Drafts are a context continuation (`mtpLookupChain`): nothing to
    /// build, and no MTP statistics are fed.
    lookup: bool = false,
    /// Set when step 0 consumed a padded history append: every draft step sits past its dead rows.
    place: ?HeadPlace = null,

    pub fn deinit(self: *MtpPreDraft, allocator: std.mem.Allocator) void {
        _ = mlx.mlx_array_free(self.t1_arr);
        if (self.place) |p| _ = mlx.mlx_array_free(p.live_end);
        for (self.draft_arrs[0..self.n_drafted]) |arr| _ = mlx.mlx_array_free(arr);
        allocator.free(self.draft_arrs);
        if (self.conf_arrs) |slots| {
            for (slots[0..self.n_conf]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        }
        if (self.q_probs) |slots| {
            for (slots[0..self.n_qp]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        }
        if (self.h_chain) |h| _ = mlx.mlx_array_free(h);
        allocator.free(self.drafts);
    }
};

/// Allocate an empty draft chain for `plan` (nothing built yet).
pub fn mtpChainInit(self: *Generator, allocator: std.mem.Allocator, plan: MtpRoundPlan, t1: u32) !MtpPreDraft {
    const consider_ext = plan.m_hi > plan.m_lo;
    const sharp_drafts = self.mtpDraftSampling(0).temperature > 0.01;
    const drafts = try allocator.alloc(u32, plan.m_hi);
    errdefer allocator.free(drafts);
    const draft_arrs = try allocator.alloc(mlx.mlx_array, plan.m_hi);
    errdefer allocator.free(draft_arrs);
    const conf_arrs: ?[]mlx.mlx_array = if (consider_ext) try allocator.alloc(mlx.mlx_array, plan.m_lo) else null;
    errdefer if (conf_arrs) |slots| allocator.free(slots);
    const q_probs: ?[]mlx.mlx_array = if (sharp_drafts) try allocator.alloc(mlx.mlx_array, plan.m_hi) else null;
    const t1_i32: i32 = @intCast(t1);
    const t1_shape = [_]c_int{1};
    return .{
        .plan = plan,
        .off0 = mtpRoundOff0(self.mtp_hist_stash, self.mtp_cache.?.step()),
        .head_input_rows = @intCast(1 + if (self.mtp_hist_stash) |stash| stash.n else @as(usize, 0)),
        .t1 = t1,
        .t1_arr = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 1, .int32),
        .pos0 = self.generated_ids.items.len + 1,
        .drafts = drafts,
        .draft_arrs = draft_arrs,
        .n_drafted = 0,
        .conf_arrs = conf_arrs,
        .n_conf = 0,
        .q_probs = q_probs,
        .n_qp = 0,
        .h_chain = null,
        .m = plan.m_lo,
    };
}

/// A lookup draft is a point mass copied from the context, so it takes exact acceptance
/// (keep it with probability p) whatever the MTP mode: a lossy rule that accepts any
/// plausible token outright makes copying the context certain, and the output loops.
pub fn acceptGraphFor(self: *const Generator, lookup: bool) MtpGraphFn {
    return if (lookup) mtpBatchedExactGraph else self.mtp_accept_graph;
}

pub fn acceptPrefixFor(self: *const Generator, lookup: bool) mtp_acceptance.PrefixFn {
    return if (lookup) mtp_acceptance.exactPrefix else self.mtp_accept_prefix;
}

/// Prompt lookup may stand in for the MTP chain (the test seam's false keeps the chain).
pub var mtp_lookup_override: ?bool = null;
pub fn mtpLookupEnabled() bool {
    return mtp_lookup_override orelse true;
}

/// End a round that stays out of the cost table. The regime clock runs from
/// the last observed round, so an unobserved round must end the interval too.
pub fn mtpRoundUntimed(self: *Generator) void {
    if (self.mtp_regime_clock) |*c| c.reset();
    self.mtp_serial_clock = null;
    self.mtp_serial_warm = 0;
}

/// Price a lookup round into the table's lookup row: the wall since the previous round
/// ended, the quantity the width grid stores. No sample before the regime clock runs.
pub fn mtpLookupObserve(self: *Generator, drafts: u32, tokens: u32) void {
    const c = if (self.mtp_regime_clock) |*c| c else return;
    const ms = @as(f32, @floatFromInt(c.read())) / @as(f32, std.time.ns_per_ms);
    _ = self.xfm.round_cost.observeLookup(drafts, self.mtpKvLen(), ms, @floatFromInt(tokens), self.spec_cost_solo);
}

/// What this machine measured for an MTP round at `width` and for each lookup size.
pub fn mtpLookupCosts(self: *const Generator, width: u32) mtp_lookup.Costs {
    const t = &self.xfm.round_cost;
    const kv = self.mtpKvLen();
    var costs = mtp_lookup.Costs{ .mtp_ms = if (t.bucketToRead(kv)) |b| t.roundMs(width, b) else null };
    const bucket = t.bucketOf(kv);
    for (&costs.lookup_ms, 0..) |*ms, k| ms.* = t.lookupMs(@intCast(k), bucket);
    return costs;
}

/// Count a round's accepted drafts toward its kind and that kind's EMA.
pub fn mtpRoundAcceptObserve(self: *Generator, lookup: bool, drafted: u32, accepted: u32) void {
    if (lookup) {
        self.mtp_lookup_streak = accepted == drafted;
        self.mtp_lookup_accepted += accepted;
        self.mtp_lookup_ema = mtp_lookup.emaStep(self.mtp_lookup_ema, accepted);
    } else {
        self.mtp_accepted_tokens += accepted;
        self.mtp_round_ema = mtp_lookup.emaStep(self.mtp_round_ema, accepted);
        self.mtp_lookup_ema = mtp_lookup.driftStep(self.mtp_lookup_ema);
    }
}

/// The committed-token index, synced to `generated_ids`.
pub fn mtpLookupIndex(self: *Generator, allocator: std.mem.Allocator) !*mtp_lookup.Index {
    if (self.mtp_lookup_idx == null) {
        var idx = mtp_lookup.Index.init(allocator);
        errdefer idx.deinit();
        try idx.extend(self.prompt_ids_owned);
        self.mtp_lookup_idx = idx;
        self.mtp_lookup_gen = 0;
    }
    const gen = self.generated_ids.items;
    try self.mtp_lookup_idx.?.extend(gen[self.mtp_lookup_gen..]);
    self.mtp_lookup_gen = gen.len;
    return &self.mtp_lookup_idx.?;
}

/// A chain of the context's continuation after `t1`, or null when the gate
/// prefers the MTP head. Single-stream rounds only.
/// How many context-continuation drafts the lookup gate takes after `t1`; 0 = the MTP head.
pub fn mtpLookupDrafts(self: *Generator, allocator: std.mem.Allocator, plan: MtpRoundPlan, t1: u32) !u32 {
    if (!mtpLookupEnabled() or self.mtp_batch_head) return 0;
    if (group_planner.enabled() and self.mtp_planner_owned) return 0;
    const idx = try self.mtpLookupIndex(allocator);
    const remaining: u32 = @intCast(self.max_tokens -| self.completion_tokens -| 1);
    const lookup_cap = depth_bounds.lookupCap(mtp_lookup.MAX_DRAFT_STRONG, depth_bounds.active);
    const got = idx.match(t1, lookup_cap);
    return mtp_lookup.gate(got, remaining, self.mtp_lookup_ema, self.mtp_round_ema, plan.m_lo, self.mtp_lookup_streak, self.mtpLookupCosts(plan.m_lo));
}

pub fn mtpLookupChain(self: *Generator, allocator: std.mem.Allocator, plan: MtpRoundPlan, t1: u32) !?MtpPreDraft {
    const k = try self.mtpLookupDrafts(allocator, plan, t1);
    const lookup_cap = depth_bounds.lookupCap(mtp_lookup.MAX_DRAFT_STRONG, depth_bounds.active);
    if (k == 0 or k < depth_bounds.floorFor(depth_bounds.active, lookup_cap)) return null;
    const got = (try self.mtpLookupIndex(allocator)).match(t1, lookup_cap);
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[mtp] prompt-lookup drafts engaged: k={d} suffix={d}\n", .{ k, got.?.suffix });
    }
    // No MTP chain consumes the stash this round; left pending, it would grow
    // across lookup rounds into one oversized head forward.
    try self.mtpApplyStash();
    const drafts = try allocator.alloc(u32, k);
    errdefer allocator.free(drafts);
    @memcpy(drafts, got.?.draft[0..k]);
    const draft_arrs = try allocator.alloc(mlx.mlx_array, k);
    errdefer allocator.free(draft_arrs);
    const one = [_]c_int{1};
    for (drafts, draft_arrs) |d, *arr| {
        const v: i32 = @intCast(d);
        arr.* = mlx.mlx_array_new_data(&v, &one, 1, .int32);
    }
    const t1_i32: i32 = @intCast(t1);
    var lplan = plan;
    lplan.m_lo = k;
    lplan.m_hi = k;
    lplan.width_trial = false;
    return .{
        .plan = lplan,
        .off0 = mtpRoundOff0(self.mtp_hist_stash, self.mtp_cache.?.step()),
        .t1 = t1,
        .t1_arr = mlx.mlx_array_new_data(&t1_i32, &one, 1, .int32),
        .drafts = drafts,
        .draft_arrs = draft_arrs,
        .n_drafted = k,
        .conf_arrs = null,
        .n_conf = 0,
        .q_probs = null,
        .n_qp = 0,
        .h_chain = null,
        .m = k,
        .lookup = true,
    };
}

/// Round N's accept decision as lazy arrays: `a` (0-d int32) drafts accepted, the next
/// `t1` (`[1]` int32) and the next `h_prev` (`[1,1,H]`), all before any host read.
pub const MtpLazyVerdict = struct {
    a: mlx.mlx_array,
    t1: mlx.mlx_array,
    h_prev: mlx.mlx_array,

    pub fn deinit(self: *MtpLazyVerdict) void {
        _ = mlx.mlx_array_free(self.a);
        _ = mlx.mlx_array_free(self.t1);
        _ = mlx.mlx_array_free(self.h_prev);
    }
};

/// `am` `[1, 1+m]` (the verify argmax), `drafts` m `[1]` ids, `verify_hidden_all` `[1, >=1+m, H]`.
pub fn mtpLazyVerdict(s: mlx.mlx_stream, am: mlx.mlx_array, drafts: []const mlx.mlx_array, verify_hidden_all: mlx.mlx_array) !MtpLazyVerdict {
    const m: c_int = @intCast(drafts.len);
    var am_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am_flat);
    try mlx.check(mlx.mlx_reshape(&am_flat, am, &[_]c_int{-1}, 1, s));
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_astype(&ids, am_flat, .int32, s));
    // a = index of the first mismatch, with a mismatch appended at m: argmax takes the first max.
    var a_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_u);
    {
        var head = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(head);
        try mlx.check(mlx.mlx_slice(&head, ids, &[_]c_int{0}, 1, &[_]c_int{m}, 1, &[_]c_int{1}, 1, s));
        var d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d);
        try concatIds(&d, drafts, s);
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_equal(&eq, head, d, s));
        var miss = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(miss);
        try mlx.check(mlx.mlx_logical_not(&miss, eq, s));
        const tail = mlx.mlx_array_new_data(&[_]bool{true}, &[_]c_int{1}, 1, .bool_);
        defer _ = mlx.mlx_array_free(tail);
        var miss_ext = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(miss_ext);
        const parts = [_]mlx.mlx_array{ miss, tail };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&miss_ext, vec, 0, s));
        var miss_i = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(miss_i);
        try mlx.check(mlx.mlx_astype(&miss_i, miss_ext, .int32, s));
        try mlx.check(mlx.mlx_argmax_axis(&a_u, miss_i, 0, false, s));
    }
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_astype(&a, a_u, .int32, s));
    var a1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a1);
    try mlx.check(mlx.mlx_reshape(&a1, a, &[_]c_int{1}, 1, s));
    var t1 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(t1);
    try mlx.check(mlx.mlx_take(&t1, ids, a1, s));
    var h_prev = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&h_prev, verify_hidden_all, a1, 1, s));
    return .{ .a = a, .t1 = t1, .h_prev = h_prev };
}

/// `[n]` int32 ids from n one-element lazy ids of any rank and integer dtype.
/// Draft producers disagree on rank ([1] from the head, [1,1] from a sampler).
pub fn concatIds(out: *mlx.mlx_array, ids: []const mlx.mlx_array, s: mlx.mlx_stream) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (ids) |id| {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, id, &[_]c_int{1}, 1, s));
        try mlx.check(mlx.mlx_vector_array_append_value(vec, flat));
    }
    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
    try mlx.check(mlx.mlx_astype(out, cat, .int32, s));
}

/// What decides whether round N+1's chain is built from round N's lazy results.
pub const LazyGate = struct {
    enabled: bool,
    greedy: bool,
    solo: bool,
    padded: bool,
    lookup_round: bool,
    planner_owned: bool,
    spec_live: bool,
    /// A full accept still leaves budget for another round. Without one the chain
    /// would be dispatched and thrown away, and a head failure would fail a finished request.
    successor: bool,
};

pub fn mtpLazyPredraftAllowedFor(g: LazyGate) bool {
    if (!g.enabled or !g.greedy or !g.solo or !g.padded or !g.spec_live or !g.successor) return false;
    return !g.lookup_round and !g.planner_owned;
}

/// The test seam's false keeps the pre-draft at the round's tail, after the host read.
pub var mtp_lazy_predraft_override: ?bool = null;
pub fn mtpLazyPredraftEnabled() bool {
    if (!mtpPredraftEnabled()) return false;
    return mtp_lazy_predraft_override orelse true;
}

/// The gate for building round N+1's chain from round N's lazy results, read before any host read.
pub fn mtpLazyGate(self: *const Generator, st: *const MtpRoundState, greedy: bool, grouped: bool) LazyGate {
    return .{
        .enabled = mtpLazyPredraftEnabled(),
        .greedy = greedy,
        .solo = !grouped and !self.mtp_batch_head and self.mtp_group_cap == 0 and st.verify_len == 1 + st.chain.m,
        .padded = self.mtpPaddedHistory(),
        .lookup_round = st.chain.lookup,
        .planner_owned = self.mtp_planner_owned,
        .spec_live = !self.spec_disabled_runtime and self.mtp_serial_left == 0 and !self.mtp_planner_pending,
        .successor = self.max_tokens -| self.completion_tokens > 1 + st.chain.m,
    };
}

/// Round N+1's chunk-A chain from round N's lazy verdict, built and dispatched before the
/// host reads round N. Null when this round's plan ends speculation. `chain` is round N's.
/// The plan is drawn before round N updates acceptance, so an adaptive depth change lands
/// one round later than on the tail path. Drawing it again after the read would advance the
/// planner's probes and trials twice per round, so the lag is kept.
pub fn mtpLazyPreDraft(self: *Generator, allocator: std.mem.Allocator, chain: *const MtpPreDraft, am: mlx.mlx_array, verify_hidden_all: mlx.mlx_array) !?MtpPreDraft {
    const s = self.xfm.s;
    const m = chain.m;
    const plan = self.mtpRoundPlan();
    if (self.spec_disabled_runtime or self.mtp_serial_left > 0) return null;
    var v = try mtpLazyVerdict(s, am, chain.draft_arrs[0..m], verify_hidden_all);
    defer v.deinit();
    var upcoming = try self.mtpChainInit(allocator, plan, 0);
    errdefer upcoming.deinit(allocator);
    try mlx.check(mlx.mlx_array_set(&upcoming.t1_arr, v.t1));
    upcoming.h_chain = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&upcoming.h_chain.?, v.h_prev));

    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    {
        const parts = try allocator.alloc(mlx.mlx_array, 1 + m);
        defer allocator.free(parts);
        parts[0] = chain.t1_arr;
        @memcpy(parts[1..], chain.draft_arrs[0..m]);
        try concatIds(&ids, parts, s);
    }
    var hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden);
    try self.mtpPaddedHidden(&hidden, m, verify_hidden_all);
    const one = mlx.mlx_array_new_int(@intCast(chain.off0 + 1));
    defer _ = mlx.mlx_array_free(one);
    var live_end = mlx.mlx_array_new();
    mlx.check(mlx.mlx_add(&live_end, one, v.a, s)) catch |e| {
        _ = mlx.mlx_array_free(live_end);
        return e;
    };
    // Anything this appends past round N's origin is dropped if the round fails.
    errdefer self.mtp_cache.?.truncate(chain.off0, s) catch {};
    try self.mtpPaddedAppend(&upcoming, chain.off0, ids, hidden, 1 + m, live_end);
    try self.mtpChainBuild(&upcoming, 0, plan.m_lo);
    try mtpChainDispatch(&upcoming, 0, plan.m_lo);
    self.mtp_lazy_built += 1;
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[mtp] lazy predraft engaged\n", .{});
    }
    return upcoming;
}

/// Whether the host read confirms a lazy chain for the next round.
pub fn mtpLazyKeeps(self: *const Generator, exact: bool) bool {
    if (!exact or self.done or self.completion_tokens >= self.max_tokens) return false;
    if (isEosId(self.next_token_id, self.eos_token_ids)) return false;
    return !self.spec_disabled_runtime and self.mtp_serial_left == 0 and !self.mtp_planner_pending;
}

/// The round's tail: settle a lazy chain against the host read, or pre-draft as before.
/// A discarded chain leaves exactly the tail pre-draft's inputs: the head back at the
/// stash origin, the stash pending. `exact`: the host kept the raw greedy accept count.
/// `planned`: the lazy build already took this round's plan, so no second plan is drawn.
pub fn mtpPreDraftResolve(self: *Generator, allocator: std.mem.Allocator, lazy: *?MtpPreDraft, exact: bool, planned: bool) !void {
    if (lazy.* == null and planned) return;
    const plan = (lazy.* orelse return self.mtpMaybePreDraft(allocator)).plan;
    const keeps = self.mtpLazyKeeps(exact);
    // Before taking the chain: on error the round's errdefer still owns and rolls it back.
    const lookup = keeps and try self.mtpLookupDrafts(allocator, plan, self.next_token_id) > 0;
    const discard = !keeps or lookup or mtp_lazy_force_discard;
    var chain = lazy.*.?;
    lazy.* = null;
    const st = &self.mtp_hist_stash.?;
    if (discard) {
        chain.deinit(allocator);
        try self.mtp_cache.?.truncate(st.off0, self.xfm.s);
        if (lookup) self.mtp_pre_draft = try self.mtpLookupChain(allocator, plan, self.next_token_id);
        return;
    }
    chain.t1 = self.next_token_id;
    chain.off0 = st.off0 + st.n;
    st.deinit();
    self.mtp_hist_stash = null;
    self.mtp_pre_draft = chain;
    self.mtp_lazy_kept += 1;
}
/// Test seam: discard every lazy chain, as a lookup pick does.
pub var mtp_lazy_force_discard: bool = false;
/// Test seam: fail the round right after its lazy chain is dispatched.
pub var mtp_lazy_fail_after_dispatch: bool = false;

/// The test seam's false keeps the merged `1 + accepted` history step.
pub var mtp_padded_head_override: ?bool = null;
pub fn mtpPaddedHeadEnabled() bool {
    return mtp_padded_head_override orelse true;
}

/// Does this request's head append the whole verify row as history and draft past its
/// dead rows (`HeadPlace`)? The qwen4 head, solo, text-only.
pub fn mtpPaddedHistory(self: *const Generator) bool {
    if (!mtpPaddedHeadEnabled() or self.mtp_batch_head) return false;
    if (self.mtp.? != .qwen4) return false;
    return self.mtpMropeContext() == null;
}

/// Consume the stash as a padded `1+m`-row history append and place `chain` past it.
pub fn mtpPaddedHistoryAppend(self: *Generator, chain: *MtpPreDraft) !void {
    var st = self.mtp_hist_stash.?;
    self.mtp_hist_stash = null;
    defer st.deinit();
    const pad = st.pad.?;
    try self.mtpPaddedAppend(chain, st.off0, pad.ids, pad.hidden, pad.rows, mlx.mlx_array_new_int(@intCast(st.off0 + st.n)));
}

/// Truncate the head to `off0`, append `rows` history rows there, and place `chain` past
/// them with `live_end`, which this takes ownership of.
pub fn mtpPaddedAppend(self: *Generator, chain: *MtpPreDraft, off0: usize, ids: mlx.mlx_array, hidden: mlx.mlx_array, rows: usize, live_end: mlx.mlx_array) !void {
    errdefer _ = mlx.mlx_array_free(live_end);
    const mc = &self.mtp_cache.?;
    try mc.truncate(off0, self.xfm.s);
    const out = try self.mtp.?.forward(self.xfm, mc, ids, hidden, @intCast(off0), .none, null, null);
    if (out.logits.ctx != null) _ = mlx.mlx_array_free(out.logits);
    if (out.hidden_next.ctx != null) _ = mlx.mlx_array_free(out.hidden_next);
    if (out.rerank_x.ctx != null) _ = mlx.mlx_array_free(out.rerank_x);
    chain.place = .{ .live_end = live_end, .dead_end = @intCast(off0 + rows) };
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[mtp] padded head history engaged\n", .{});
    }
}

// ── tests ──


test "lazy predraft: the lazy verdict matches the host greedy verdict at zero, partial and full accepts" {
    // Bar: a, t1 and h_prev read back equal to mtpGreedyVerdict and the verify hidden at row a.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const drafts_host = [_]i32{ 5, 6, 7 };
    // Production draft ids come in mixed ranks ([1] from the head, [1,1] from a sampler).
    var drafts: [3]mlx.mlx_array = undefined;
    for (&drafts, drafts_host, 0..) |*d, v, i| {
        d.* = if (i == 1) mlx.mlx_array_new_data(&v, &[_]c_int{ 1, 1 }, 2, .int32) else mlx.mlx_array_new_data(&v, &[_]c_int{1}, 1, .int32);
    }
    defer for (drafts) |d| {
        _ = mlx.mlx_array_free(d);
    };
    var vh_host: [4 * 2]f32 = undefined;
    for (&vh_host, 0..) |*v, i| v.* = @floatFromInt(10 + i);
    const vh = mlx.mlx_array_new_data(&vh_host, &[_]c_int{ 1, 4, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(vh);
    const cases = [_][4]u32{ .{ 9, 6, 7, 1 }, .{ 5, 6, 8, 4 }, .{ 5, 6, 7, 11 } };
    for (cases) |am_host| {
        const am = mlx.mlx_array_new_data(&am_host, &[_]c_int{ 1, 4 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(am);
        var am_i32: [4]i32 = undefined;
        for (&am_i32, am_host) |*d, v| d.* = @intCast(v);
        const want = Generator.mtpGreedyVerdict(&am_i32, &[_]u32{ 5, 6, 7 }, 0, 1 << 30);
        var got = try Generator.mtpLazyVerdict(s, am, &drafts, vh);
        defer got.deinit();
        var a: i32 = -1;
        try mlx.check(mlx.mlx_array_eval(got.a));
        try mlx.check(mlx.mlx_array_item_int32(&a, got.a));
        try testing.expectEqual(@as(i32, @intCast(want.accepted)), a);
        try mlx.check(mlx.mlx_array_eval(got.t1));
        try testing.expectEqualSlices(c_int, &[_]c_int{1}, mlx.getShape(got.t1));
        try testing.expectEqual(@as(i32, @intCast(want.next_pending)), mlx.mlx_array_data_int32(got.t1).?[0]);
        var h = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h);
        try mlx.check(mlx.mlx_contiguous(&h, got.h_prev, false, s));
        try mlx.check(mlx.mlx_array_eval(h));
        try testing.expectEqualSlices(c_int, &[_]c_int{ 1, 1, 2 }, mlx.getShape(h));
        const row: usize = want.accepted;
        try testing.expectEqualSlices(f32, vh_host[row * 2 .. row * 2 + 2], mlx.mlx_array_data_float32(h).?[0..2]);
    }
}

test "lazy predraft: only solo greedy padded rounds with speculation live build a lazy chain" {
    // Bar: every open gate allows it; closing any single gate (sampled, group, lookup round, ...) forbids it.
    const open: Generator.LazyGate = .{ .enabled = true, .greedy = true, .solo = true, .padded = true, .lookup_round = false, .planner_owned = false, .spec_live = true, .successor = true };
    try testing.expect(Generator.mtpLazyPredraftAllowedFor(open));
    inline for (.{ "enabled", "greedy", "solo", "padded", "spec_live", "successor" }) |field| {
        var g = open;
        @field(g, field) = false;
        try testing.expect(!Generator.mtpLazyPredraftAllowedFor(g));
    }
    inline for (.{ "lookup_round", "planner_owned" }) |field| {
        var g = open;
        @field(g, field) = true;
        try testing.expect(!Generator.mtpLazyPredraftAllowedFor(g));
    }
}

pub const MtpChainTestSlot = struct {
    cache: KVCache,
    off: usize = 0,
    entries: []transformer_mod.SSMCacheEntry,
    gen: ?Generator = null,

    pub fn init(a: std.mem.Allocator, io: std.Io, xfm: *Transformer, tok: *Tokenizer, prompt: []const u32, depth: u32) !*@This() {
        const slot = try a.create(@This());
        errdefer a.destroy(slot);
        var cache = try KVCache.init(a, xfm.config.num_hidden_layers);
        errdefer cache.deinit();
        const entries = try a.alloc(transformer_mod.SSMCacheEntry, xfm.config.num_hidden_layers);
        slot.* = .{ .cache = cache, .entries = entries };
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        errdefer {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                transformer_mod.ssmFreeQsaState(e);
            }
            a.free(entries);
        }
        const ctx: ForwardCtx = .{ .cache = &slot.cache, .moe_seq_offset = &slot.off, .ssm_entries = entries, .capture_hidden = null, .vision_embeddings = null };
        slot.gen = try Generator.initWithOptions(io, a, xfm, tok, prompt, 256, .{ .temperature = 0 }, &.{}, .{
            .ctx = ctx,
            .mtp = .{ .qwen4 = xfm.qwen4Trunk().? },
            .mtp_enabled = true,
            .model_has_mtp = true,
            .mtp_depth = depth,
            .skip_lazy_preforward = true,
        });
        return slot;
    }

    pub fn deinit(slot: *@This(), a: std.mem.Allocator) void {
        if (slot.gen) |*g| g.deinit(a);
        for (slot.entries) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            transformer_mod.ssmFreeQsaState(e);
        }
        a.free(slot.entries);
        slot.cache.deinit();
        a.destroy(slot);
    }

    pub fn begin(g: *Generator, a: std.mem.Allocator, depth: u32, confidence: bool) !Generator.MtpRoundOpen {
        g.mtp_depth_current = depth;
        g.spec_cost_solo = false;
        g.mtp_batch_head = true;
        defer g.mtp_batch_head = false;
        const begun = try g.mtpRoundBegin(a);
        var open = switch (begun) {
            .open => |o| o,
            .verify => |v| {
                var st = v;
                st.deinit(a);
                return error.ExpectedOpenRound;
            },
            .done => |r| {
                if (r) |result| a.free(result.tokens);
                return error.MtpRoundDeclined;
            },
        };
        errdefer open.chain.deinit(a);
        try testing.expectEqual(depth, open.chain.m);
        if (confidence) {
            if (open.chain.conf_arrs != null) std.debug.print("[mtp fixture] existing confidence plan={any}\n", .{open.chain.plan});
            try testing.expect(open.chain.conf_arrs == null);
            open.chain.drafts = try a.realloc(open.chain.drafts, depth + 1);
            open.chain.draft_arrs = try a.realloc(open.chain.draft_arrs, depth + 1);
            open.chain.conf_arrs = try a.alloc(mlx.mlx_array, depth);
            open.chain.plan.m_hi = depth + 1;
            open.chain.plan.tau_ln = 1;
        }
        return open;
    }

    pub fn equal(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream, name: []const u8) !void {
        if (a.ctx == null or b.ctx == null) {
            try testing.expect(a.ctx == null and b.ctx == null);
            return;
        }
        var same = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(same);
        try mlx.check(mlx.mlx_array_equal(&same, a, b, false, s));
        try mlx.check(mlx.mlx_array_eval(same));
        var yes = false;
        try mlx.check(mlx.mlx_array_item_bool(&yes, same));
        if (!yes) std.debug.print("[mtp round mismatch] {s}: {any} vs {any}\n", .{ name, mlx.getShape(a), mlx.getShape(b) });
        try testing.expect(yes);
    }
};

/// A loaded qwen4_exp pack and a byte tokenizer for generator-level lazy pre-draft runs.
const LazyRig = struct {
    config: model_mod.ModelConfig,
    xfm: Transformer,
    tok: Tokenizer,

    fn load(a: std.mem.Allocator, io: std.Io) !*LazyRig {
        const model_dir = env.get(.QWEN4_TEST_MODEL) orelse return error.SkipZigTest;
        if (mlx.noGpuBackend()) return error.SkipZigTest;
        const rig = try a.create(LazyRig);
        errdefer a.destroy(rig);
        rig.config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
        rig.xfm = try @import("../../plugin/driver.zig").load(io, a, &rig.config, std.mem.span(model_dir), mlx.gpuStream());
        rig.tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
        return rig;
    }

    fn deinit(rig: *LazyRig, a: std.mem.Allocator) void {
        rig.tok.deinit();
        rig.xfm.deinit();
        rig.config.deinit(a);
        a.destroy(rig);
    }

    /// A prompt that repeats itself, so prompt lookup has a continuation to offer.
    fn prompt(buf: []u32) []u32 {
        for (buf, 0..) |*id, i| id.* = @intCast(1 + ((i % 24) * 7919) % 40000);
        return buf;
    }

    const Run = struct { ids: []u32, built: u32, kept: u32, lookup_rounds: u64, completion: u32, residue: bool };

    /// Greedy decode to `max_tokens` with the lazy pre-draft on or off.
    fn run(rig: *LazyRig, a: std.mem.Allocator, io: std.Io, lazy: bool, max_tokens: u32) !Run {
        mtp_lazy_predraft_override = lazy;
        defer mtp_lazy_predraft_override = null;
        var buf: [192]u32 = undefined;
        const slot = try MtpChainTestSlot.init(a, io, &rig.xfm, &rig.tok, prompt(&buf), 3);
        defer slot.deinit(a);
        const g = &slot.gen.?;
        g.max_tokens = max_tokens;
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(a);
        while (!g.done and g.completion_tokens < max_tokens) {
            const r = (try g.nextMtp(a)) orelse break;
            defer a.free(r.tokens);
            try ids.appendSlice(a, r.tokens);
        }
        // Whether a pre-draft outlives the last round (the eager tail builds one at max_tokens).
        const residue = g.mtp_pre_draft != null;
        return .{ .ids = try ids.toOwnedSlice(a), .built = g.mtp_lazy_built, .kept = g.mtp_lazy_kept, .lookup_rounds = g.mtp_lookup_rounds, .completion = g.completion_tokens, .residue = residue };
    }
};

test "lazy predraft: greedy tokens equal the eager padded path, lookup off and on, to and short of max_tokens (QWEN4_TEST_MODEL)" {
    // Bar: byte-identical ids; the lazy arm built and kept chains; the final rounds leave what the eager path leaves.
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const rig = try LazyRig.load(a, io);
    defer rig.deinit(a);
    const forced = depth_bounds.active;
    defer depth_bounds.active = forced;
    depth_bounds.active = .{ .min = 3, .max = 3 };
    defer mtp_lookup_override = null;
    for ([_]bool{ false, true }) |lookup| {
        mtp_lookup_override = lookup;
        for ([_]u32{ 120, 37 }) |max_tokens| {
            const eager = try rig.run(a, io, false, max_tokens);
            defer a.free(eager.ids);
            const lazy = try rig.run(a, io, true, max_tokens);
            defer a.free(lazy.ids);
            std.debug.print("[lazy predraft] lookup={} max={d} tokens={d}/{d} built={d} kept={d} lookup_rounds={d}/{d}\n", .{ lookup, max_tokens, eager.ids.len, lazy.ids.len, lazy.built, lazy.kept, eager.lookup_rounds, lazy.lookup_rounds });
            try testing.expectEqualSlices(u32, eager.ids, lazy.ids);
            try testing.expectEqual(@as(u32, 0), eager.built);
            try testing.expect(lazy.kept > 0);
            try testing.expectEqual(max_tokens, lazy.completion);
            try testing.expectEqual(eager.residue, lazy.residue);
            if (lookup) try testing.expect(lazy.lookup_rounds > 0);
        }
    }
    // A lookup pick discards the lazy chain; force that on every round so the discard path
    // (head back at the stash origin, stash pending) runs whatever lookup decides.
    mtp_lookup_override = false;
    mtp_lazy_force_discard = true;
    defer mtp_lazy_force_discard = false;
    const eager = try rig.run(a, io, false, 120);
    defer a.free(eager.ids);
    const discarded = try rig.run(a, io, true, 120);
    defer a.free(discarded.ids);
    std.debug.print("[lazy predraft] forced discard tokens={d}/{d} built={d} kept={d}\n", .{ eager.ids.len, discarded.ids.len, discarded.built, discarded.kept });
    try testing.expectEqualSlices(u32, eager.ids, discarded.ids);
    try testing.expect(discarded.built > 0);
    try testing.expectEqual(@as(u32, 0), discarded.kept);
}

test "lazy predraft: a failure after the lazy dispatch leaves the head committed-only (QWEN4_TEST_MODEL)" {
    // Bar: the failing round returns the injected error and the head ends at the committed end, no chain, no stash.
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const rig = try LazyRig.load(a, io);
    defer rig.deinit(a);
    const forced = depth_bounds.active;
    defer depth_bounds.active = forced;
    depth_bounds.active = .{ .min = 3, .max = 3 };
    mtp_lazy_predraft_override = true;
    defer mtp_lazy_predraft_override = null;
    var buf: [192]u32 = undefined;
    const slot = try MtpChainTestSlot.init(a, io, &rig.xfm, &rig.tok, LazyRig.prompt(&buf), 3);
    defer slot.deinit(a);
    const g = &slot.gen.?;
    for (0..3) |_| {
        const r = (try g.nextMtp(a)) orelse return error.MissingMtpRound;
        a.free(r.tokens);
    }
    try testing.expect(g.mtp_lazy_kept > 0);
    const committed = g.mtpCommittedHistoryLen();
    mtp_lazy_fail_after_dispatch = true;
    defer mtp_lazy_fail_after_dispatch = false;
    try testing.expectError(error.MtpLazyInjected, g.nextMtp(a));
    try testing.expect(g.mtp_pre_draft == null and g.mtp_hist_stash == null);
    try testing.expectEqual(committed, g.mtp_cache.?.step());
    try testing.expectEqual(committed, g.mtpCommittedHistoryLen());
}
