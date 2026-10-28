//! Prefill interleaving: decode share, boundary admission, SSD write-through, chunk width.

const std = @import("std");
const err_name = @import("../../sdk/util/err_name.zig");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const transformer_mod = @import("transformer.zig");
const generate_mod = @import("generate.zig");
const model_mod = @import("../../sdk/config/model.zig");
const prefix_cache_mod = @import("cache/prefix_cache.zig");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const testing = std.testing;

const slotExclusiveDecode = @import("batching/verdict.zig").slotExclusiveDecode;
const runDecodeTick = @import("decode.zig").runDecodeTick;
const Scheduler = @import("scheduler.zig").Scheduler;
const adaptiveChunkWidthFor = @import("scheduler/params.zig").adaptiveChunkWidthFor;
const Slot = @import("slot.zig").Slot;
const mtpCrowdThresholdFor = @import("spec/mtp_groups.zig").mtpCrowdThresholdFor;
const SpecTickMode = @import("spec/wiring.zig").SpecTickMode;
const slotSpecMode = @import("spec/wiring.zig").slotSpecMode;
const prefillAdmitted = @import("step.zig").prefillAdmitted;
const publishLiveKvResidency = @import("step.zig").publishLiveKvResidency;
const scheduler_mod = @import("scheduler.zig");

/// Allocate the slot's KVCache state (already done in Slot.init), construct
/// the per-slot Generator via `Generator.initWithOptions(.{ .ctx = slot.ctx,
/// .skip_lazy_preforward = true_for_regular, ... })`, and store it on the
/// slot. After return, the slot is ready for decode ticks.
pub fn prefillDecodeShare() f32 {
    return scheduler_mod.prefill_decode_share;
}

pub const InterleaveCtx = struct {
    sch: *Scheduler,
    /// The slot being prefilled: not in `decoding` yet, its KV still counts.
    slot: *Slot,
    decode_ns: u64 = 0,
    ticks: u32 = 0,
    /// Wall clock since the previous boundary's ticks ended: the chunk just forwarded.
    chunk_sw: io_util.Stopwatch,
};

/// Ticks a speculating slot must see company before it gives up speculation for the batch.
const COMPANY_YIELD_TICKS: u8 = 2;

pub fn companyStreak(streak: u8, has_company: bool) u8 {
    return if (has_company) streak +| 1 else 0;
}

/// Spec modes that give way to the batched group once a slot has company: DSpark, DFlash on a
/// sparse target (its wide verify reads every expert the block routes to), PLD on CUDA. A dense
/// DFlash slot yields only to a crowd (`crowd` others): below it the yielded slot ticks plain
/// beside the MTP verify group instead of inside it, which costs everyone a forward.
pub fn yieldsToCompany(mode: SpecTickMode, cuda: bool, sparse_target: bool, others: usize, crowd: usize) bool {
    return mode == .dspark or (mode == .dflash and (sparse_target or others >= crowd)) or (mode == .pld and cuda);
}

/// The first request of a burst was admitted alone, so it armed speculation and ticks
/// serial beside the group; once company is steady it yields.
pub fn specYieldTick(active: []const *Slot) void {
    for (active) |s| {
        const gen = if (s.legacy_gen) |*g| g else continue;
        if (gen.spec_disabled_runtime) continue;
        const mode = slotSpecMode(s, gen);
        var others: usize = 0;
        for (active) |o| {
            if (o != s and o.model == s.model) others += 1;
        }
        if (!yieldsToCompany(mode, mlx.cudaAvailable(), s.model.config.?.isMoe(), others, mtpCrowdThresholdFor(s))) {
            s.company_ticks = 0;
            continue;
        }
        s.company_ticks = companyStreak(s.company_ticks, others > 0);
        if (s.company_ticks >= COMPANY_YIELD_TICKS) gen.yieldSpecToCompany(@tagName(mode));
    }
}

/// Prefill width while other streams decode: a chunk boundary is their only yield point.
const COMPANY_PREFILL_CHUNK: u32 = 2048;

pub fn companyPrefillChunk(chunk: u32, decoding: usize) u32 {
    if (decoding == 0) return chunk;
    return if (chunk == 0) COMPANY_PREFILL_CHUNK else @min(chunk, COMPANY_PREFILL_CHUNK);
}

/// Largest accepted decode share: at 0.9 the decoders own nine times each chunk's time.
pub const PREFILL_DECODE_SHARE_MAX: f32 = 0.9;
/// Prefill width while a decode share is live and someone decodes.
pub const DECODE_SHARE_PREFILL_CHUNK: u32 = 1024;

/// `--prefill-decode-share` / MLX_SERVE_PREFILL_DECODE_SHARE text to a share in [0, 0.9].
/// Above 0.9 clamps; anything that is not a number >= 0 is an error.
pub fn parseDecodeShare(text: []const u8) error{InvalidDecodeShare}!f32 {
    const v = std.fmt.parseFloat(f32, text) catch return error.InvalidDecodeShare;
    if (std.math.isNan(v) or v < 0) return error.InvalidDecodeShare;
    return @min(v, PREFILL_DECODE_SHARE_MAX);
}

/// The decode share a server runs with when nothing names one: the boundary admission and the
/// smallest-tail-first batch only engage under a share, and without it a quick question waited
/// 90-120 s behind a 90k prefill (26.10.3 validation). `--prefill-decode-share 0` restores the old schedule.
pub const DEFAULT_DECODE_SHARE: f32 = 0.25;

pub fn resolveDecodeShare(flag: ?[]const u8) error{InvalidDecodeShare}!f32 {
    return parseDecodeShare(flag orelse return DEFAULT_DECODE_SHARE);
}

/// An echo prefill ranks every row against the whole vocab: the top-k sort peaks at five
/// [chunk, vocab] 4-byte arrays (1.25 GiB at 256 rows of Gemma's 262k vocab), beside the
/// chunk's own bf16 logits (128 MiB).
const PROMPT_LOGPROBS_CHUNK: u32 = 256;

pub fn promptLogprobsWidthCap(echo: bool, cap: u32) u32 {
    if (!echo) return cap;
    return if (cap == 0) PROMPT_LOGPROBS_CHUNK else @min(cap, PROMPT_LOGPROBS_CHUNK);
}

/// The width a prefill started beside live decoders may run at, 0 = no cap. Decided when
/// the prefill starts, after admission, so admission bills the uncapped width. A decoder
/// that finishes mid-prefill lifts the cap only where the adaptive width hook runs.
pub fn decodeShareAdmissionCap(decoding: usize, share: f32) u32 {
    if (decoding == 0 or share <= 0) return 0;
    return DECODE_SHARE_PREFILL_CHUNK;
}

/// Decode wall time owed after a prefill chunk of `chunk_ns`: chunk * S / (1 - S).
pub fn decodeShareBudgetNs(chunk_ns: u64, share: f32) u64 {
    if (share <= 0) return 0;
    const s: f64 = @min(share, PREFILL_DECODE_SHARE_MAX);
    return @intFromFloat(@as(f64, @floatFromInt(chunk_ns)) * s / (1 - s));
}

/// Is another decode tick owed at this boundary? Never fewer than `interleaveTicksFor`.
pub fn interleaveTickOwed(share: f32, chunk_ns: u64, first_ns: u64, ticks: u32, decode_ns: u64) bool {
    if (first_ns == 0) return false;
    if (ticks < interleaveTicksFor(chunk_ns, first_ns)) return true;
    return decode_ns < decodeShareBudgetNs(chunk_ns, share);
}

/// The adaptive width hook's pick, capped while a decode share is live and someone decodes.
pub fn decodeShareWidthCap(width: u32, decoding: usize, share: f32) u32 {
    const cap = decodeShareAdmissionCap(decoding, share);
    return if (cap == 0) width else @min(width, cap);
}

/// The largest uncached tail a request may have to be prefilled between another prefill's chunks:
/// one decode-share chunk, so the nested prefill costs the outer about one chunk of time.
pub const BOUNDARY_MAX_TAIL: usize = DECODE_SHARE_PREFILL_CHUNK;
/// Requests admitted at one chunk boundary, across its admission scan and the gaps between its owed
/// decode ticks. The wall-time cap in `BoundaryBudget` is the real bound.
pub const BOUNDARY_MAX_COUNT: u32 = 4;

/// What the boundary gate reads about a pending request.
pub const BoundaryCand = struct {
    exclusive: bool,
    has_media: bool,
    prompt_logprobs: bool,
    generator_path: bool,
    /// Module-side per-request state (`Caps.shared_readonly_state`) that a declined restore would reset under the outer.
    module_state: bool,
    /// Uncached tail, from `peekMatchedTokens`.
    tail: usize,
};

pub const BoundaryCtx = struct {
    share: f32,
    /// A prefill started at a boundary may tick decoders at its own boundaries but never admits.
    nested: bool,
    outer_exclusive: bool,
    decoding: usize,
    max_concurrent: usize,
    max_tail: usize,
};

/// May this pending request be prefilled at the outer prefill's chunk boundary?
pub fn boundaryEligible(c: BoundaryCand, x: BoundaryCtx) bool {
    if (x.share <= 0 or x.nested or x.outer_exclusive) return false;
    if (c.exclusive or c.has_media or c.prompt_logprobs or !c.generator_path or c.module_state) return false;
    if (x.decoding >= x.max_concurrent) return false;
    return c.tail <= x.max_tail;
}

/// What one boundary may spend on nested prefills, so the outer always advances.
pub const BoundaryBudget = struct {
    max_count: u32,
    ns_cap: u64,
    count: u32 = 0,
    spent_ns: u64 = 0,

    pub fn mayStart(self: BoundaryBudget) bool {
        return self.count < self.max_count and self.spent_ns < self.ns_cap;
    }

    pub fn charge(self: *BoundaryBudget, ns: u64) void {
        self.count += 1;
        self.spent_ns +|= ns;
    }
};

/// Stable ascending order of `tails` written to `order` as indices: smallest uncached tail first.
pub fn orderByTail(tails: []const usize, order: []usize) void {
    for (order[0..tails.len], 0..) |*o, i| o.* = i;
    const by = struct {
        fn lessThan(t: []const usize, a: usize, b: usize) bool {
            return t[a] < t[b];
        }
    };
    std.sort.insertion(usize, order[0..tails.len], tails, by.lessThan);
}

/// Undo a hot-cache restore the slot has not started on: hand the checkout back and empty the
/// slot's cache, so the request can wait in `pending` and restore again when its turn comes.
pub fn resetRestoredSlot(slot: *Slot) void {
    if (slot.model.prefix_cache) |hc| hc.releaseCheckout(@intFromPtr(slot), "tail over the boundary limit");
    emptySlotCache(slot);
    slot.restored_entry = 0;
}

/// Back to a cold slot: no restored KV, SSM state or module-side qwen4 state.
pub fn emptySlotCache(slot: *Slot) void {
    if (slot.ssm_entries) |ents| prefix_cache_mod.HotPrefixCache.resetSsmEntries(ents);
    if (slot.model.transformer) |xf| {
        slot.cache.truncate(0, xf.s) catch {};
        if (xf.qwen4Trunk()) |t| {
            t.resetQsaPooledRope();
            t.qwen4MtpResetOwned(slot.enable_mtp);
        }
    }
    slot.moe_seq_offset = 0;
    slot.cached_tokens = 0;
}

/// The first pending request that passes the boundary gate, removed from `pending` with its
/// `in_pass` held. `outer` is the prefill whose chunk boundary this is.
fn takeBoundaryCandidate(sch: *Scheduler, outer: *Slot) ?*Slot {
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    var decoding: usize = 0;
    for (sch.decoding.items) |s| {
        if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
        decoding += 1;
    }
    const x: BoundaryCtx = .{
        .share = prefillDecodeShare(),
        .nested = sch.boundary_depth > 0,
        .outer_exclusive = slotExclusiveDecode(outer),
        .decoding = decoding,
        .max_concurrent = sch.max_concurrent,
        .max_tail = BOUNDARY_MAX_TAIL,
    };
    const n = @min(sch.pending.items.len, 32);
    for (sch.pending.items[0..n], 0..) |s, i| {
        if (s.cancelled.load(.acquire)) continue;
        const c: BoundaryCand = .{
            .exclusive = slotExclusiveDecode(s),
            .has_media = s.vision_embeddings != null or s.media.len > 0,
            .prompt_logprobs = s.prompt_logprobs,
            .generator_path = s.model.transformer != null,
            .module_state = if (s.model.transformer) |xf| xf.config.caps().shared_readonly_state else false,
            .tail = slotUncachedTail(s),
        };
        if (!boundaryEligible(c, x)) continue;
        _ = s.in_pass.fetchAdd(1, .acq_rel);
        _ = sch.pending.orderedRemove(i);
        sch.boundary_slot = s;
        return s;
    }
    return null;
}

/// At a chunk boundary of `ic.slot`'s prefill: admit and prefill small pending requests, so their
/// first token does not wait for the whole prefill. The decode ticks that follow include them.
/// `budget` is the boundary's, shared by its admission scan and the gaps between its owed ticks.
/// Returns the wall time spent.
fn admitSmallAtBoundary(ic: *InterleaveCtx, share: f32, budget: *BoundaryBudget) u64 {
    const sch = ic.sch;
    if (share <= 0 or sch.boundary_depth > 0) return 0;
    var total = io_util.Stopwatch.init(sch.io);
    while (budget.mayStart()) {
        const slot = takeBoundaryCandidate(sch, ic.slot) orelse break;
        var sw = io_util.Stopwatch.init(sch.io);
        slot.boundary_max_tail = BOUNDARY_MAX_TAIL;
        sch.boundary_depth += 1;
        sch.boundary_outer = ic.slot;
        const saved_tokens = sch.inflight_prefill_tokens.load(.monotonic);
        const saved_expected = sch.inflight_prefill_expected.load(.monotonic);
        const outcome = prefillAdmitted(sch, slot);
        // runPrefill zeroes the progress gauges on exit; the outer prefill's go back.
        sch.inflight_prefill_tokens.store(saved_tokens, .monotonic);
        sch.inflight_prefill_expected.store(saved_expected, .monotonic);
        sch.boundary_outer = null;
        sch.boundary_depth -= 1;
        slot.boundary_max_tail = null;
        const wait = outcome == .held or outcome == .tail_too_large;
        // Back to the head of `pending` before `in_pass` drops, so `complete` never sees a slot that
        // is in no list.
        sch.queue_mu.lockUncancelable(sch.io);
        sch.boundary_slot = null;
        if (wait) sch.pending.insert(sch.allocator, 0, slot) catch slot.markError("OutOfMemory");
        sch.queue_mu.unlock(sch.io);
        _ = slot.in_pass.fetchSub(1, .acq_rel);
        if (wait) break;
        if (outcome == .decoding) log.info("[boundary] admitted a {d}-token request beside a {d}-token prefill\n", .{ slot.prompt_tokens, ic.slot.full_prompt.len });
        budget.charge(sw.read());
    }
    return total.read();
}

/// The tokens `slot` would still have to prefill, from a read-only peek at the hot cache. A slot
/// that skips the cache, or carries media (media rows are never matched by token), counts whole.
pub fn slotUncachedTail(slot: *Slot) usize {
    const total = slot.full_prompt.len;
    if (slot.skip_prefix_cache or slot.media.len > 0) return total;
    const hc = if (slot.model.prefix_cache) |p| p else return total;
    return total - @min(total, hc.peekMatchedTokens(slot.full_prompt, slot.has_tools, slot.cache.config, slot.ssm_entries != null));
}

pub const OwedTicks = struct { ticks: u32, spent_ns: u64 };

/// The ticks after a boundary's first one: run until none is owed or the decoders run out
/// (`tick` returns 0).
pub fn runOwedDecodeTicks(share: f32, chunk_ns: u64, first_ns: u64, ctx: *anyopaque, tick: *const fn (*anyopaque) u64) OwedTicks {
    var r: OwedTicks = .{ .ticks = 1, .spent_ns = first_ns };
    while (interleaveTickOwed(share, chunk_ns, first_ns, r.ticks, r.spent_ns)) {
        const ns = tick(ctx);
        if (ns == 0) break;
        r.ticks += 1;
        r.spent_ns +|= ns;
    }
    return r;
}

const INTERLEAVE_MAX_TICKS: u32 = 8;

/// Decode ticks owed at a chunk boundary: enough to keep the decoding streams at a quarter
/// of wall time (chunk / 3), at least one, capped so a slow chunk cannot stall its own TTFT.
pub fn interleaveTicksFor(chunk_ns: u64, tick_ns: u64) u32 {
    if (tick_ns == 0) return 1;
    return @intCast(std.math.clamp(chunk_ns / 3 / tick_ns, 1, INTERLEAVE_MAX_TICKS));
}

/// SSD-first write-through: persist each completed prefill chunk as the prefill produces it,
/// so a cancelled or killed prefill leaves a restorable chunk-aligned prefix. Armed only with
/// a live background writer, so the inference thread never pays the file write.
pub fn prefillWriteThroughCb(opaque_ctx: *anyopaque, abs_kv_pos: usize, cps: []const transformer_mod.SSMCheckpoint) void {
    const wc: *WriteThroughCtx = @ptrCast(@alignCast(opaque_ctx));
    const slot = wc.slot;
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |p| p else return;
    if (!hc.ssd_first) return;
    const d = if (hc.disk) |*dd| dd else return;
    if (d.module_context != null) return;
    if (d.writer == null) return;
    if (abs_kv_pos == 0 or abs_kv_pos > slot.full_prompt.len) return;
    const s = if (slot.model.transformer) |x| x.s else return;
    wc.chunks += 1;
    // RAM-backed mode banks one chunk as crash salvage. SSD-only mode must keep pace with
    // prefill or the first completed turn can leave most of its prefix unavailable after restart.
    _ = d.appendCommitBounded(
        slot.cache.entries,
        abs_kv_pos,
        slot.cache.config,
        slot.full_prompt[0..abs_kv_pos],
        slot.has_tools,
        // The items whose rows this chunk boundary covers; their keys travel with the rows.
        prefix_cache_mod.spansBelow(slot.media, abs_kv_pos),
        if (cps.len > 0) cps else null,
        s,
        if (hc.ram_enabled) WRITE_THROUGH_FLUSH_BOUND_BYTES else std.math.maxInt(u64),
    ) catch |err| {
        log.warn("  [disk-cache] prefill write-through failed: {s}\n", .{@errorName(err)});
    };
}

/// The write-through hook's per-call flush bound: one byte, so the chunk loop stops after the first chunk it writes.
pub const WRITE_THROUGH_FLUSH_BOUND_BYTES: u64 = 1;

pub const WriteThroughCtx = struct {
    slot: *Slot,
    chunks: u32 = 0,
};

/// The write-through only pays when this turn's prefill produces at least one whole disk
/// chunk: below that it persisted a warm turn's whole restored prefix inside TTFT
/// (+183/+369/+737 ms at 16k/32k/64k) for a prefix the end-of-request commit persists anyway.
fn writeThroughSpanReached(new_span: usize, chunk_tokens: u32) bool {
    return new_span >= chunk_tokens;
}

var write_through_span_declined_logged = std.atomic.Value(bool).init(false);
var write_through_off_logged = std.atomic.Value(bool).init(false);
var write_through_env_cached: ?bool = null;

/// `MLX_SERVE_SSD_WRITE_THROUGH=0` takes the write-through out of the prefill loop; the
/// end-of-request commit then persists the whole turn. A real two-arm tradeoff (crash-safe
/// prefix vs TTFT) and the only way to A/B its cost.
pub fn writeThroughEnabledFromEnv(raw: ?[]const u8) bool {
    const v = raw orelse return true;
    return !std.mem.eql(u8, v, "0");
}

/// Read on the inference thread only.
fn writeThroughEnabled() bool {
    if (write_through_env_cached) |v| return v;
    const v = blk: {
        const raw = env.get(.MLX_SERVE_SSD_WRITE_THROUGH) orelse break :blk writeThroughEnabledFromEnv(null);
        break :blk writeThroughEnabledFromEnv(std.mem.sliceTo(raw, 0));
    };
    write_through_env_cached = v;
    return v;
}

/// The one gate for the write-through: SSD-first + a live writer + a disk tier + a new span
/// worth at least one chunk.
pub fn writeThroughArmed(slot: *Slot, new_span: usize) bool {
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |p| p else return false;
    if (hc.disk) |*d| if (d.module_context != null) return false;
    if (!writeThroughEnabled()) {
        if (hc.ssd_first and !write_through_off_logged.swap(true, .monotonic)) {
            log.info("  [disk-cache] prefill write-through disabled by MLX_SERVE_SSD_WRITE_THROUGH=0 — the end-of-request commit persists every turn\n", .{});
        }
        return false;
    }
    if (!hc.ssd_first) return false;
    const d = if (hc.disk) |*dd| dd else return false;
    const writer_up = d.writer != null;
    if (!writer_up) return false;
    if (!writeThroughSpanReached(new_span, d.chunk_tokens)) {
        if (!write_through_span_declined_logged.swap(true, .monotonic)) {
            log.info("  [disk-cache] prefill write-through declined: {d} new tokens is under one chunk ({d}) — the end-of-request commit persists this turn\n", .{ new_span, d.chunk_tokens });
        }
        return false;
    }
    return true;
}

/// Context for `Generator.InitOptions.chunk_width_hook`. `cfg` is optional (embedded engines).
pub const ChunkWidthCtx = struct {
    sch: *Scheduler,
    cfg: ?*const model_mod.ModelConfig,
    kv_bits: u64,
};

/// All resident writers and unaccepted module files share the prefill's memory ceiling.
fn chunkWidthStagedBytes(wc: *ChunkWidthCtx) u64 {
    return @import("cache/budget.zig").stagedHostBytes(wc.sch);
}

pub fn chunkWidenConfirmCb(opaque_ctx: *anyopaque, pos: usize, want: u32) bool {
    const wc: *ChunkWidthCtx = @ptrCast(@alignCast(opaque_ctx));
    const cfg = wc.cfg orelse return false;
    const ok = scheduler_mod.prefill_chunk_widen_ok orelse return false;
    return ok(cfg, wc.kv_bits, pos, want, chunkWidthStagedBytes(wc));
}

pub fn chunkWidthCb(
    opaque_ctx: *anyopaque,
    pos: usize,
    cur: u32,
    cap: u32,
    st: *generate_mod.AdaptiveWidthState,
) u32 {
    const wc: *ChunkWidthCtx = @ptrCast(@alignCast(opaque_ctx));
    const cfg = wc.cfg orelse return cur;
    const pick = scheduler_mod.prefill_chunk_adapt orelse return cur;
    const next = pick(cfg, wc.kv_bits, pos, cur, cap, st, chunkWidthStagedBytes(wc));
    // Only a live adaptive width may move; elsewhere the admitted width stands.
    if (!adaptiveChunkWidthFor(cfg)) return next;
    return decodeShareWidthCap(next, scheduler_mod.liveDecodingCount(wc.sch), prefillDecodeShare());
}

/// What the owed-tick loop runs between boundaries: admit what arrived, then tick. Returns the
/// admission time plus the tick's (0 only when nothing ran, which ends the loop).
const OwedTickCtx = struct { ic: *InterleaveCtx, share: f32, budget: *BoundaryBudget };

fn admitThenTick(opaque_ctx: *anyopaque) u64 {
    const t: *OwedTickCtx = @ptrCast(@alignCast(opaque_ctx));
    const nested = admitSmallAtBoundary(t.ic, t.share, t.budget);
    return interleaveDecodeTick(t.ic.sch) +| nested;
}

pub fn interleaveDecodeTickCb(opaque_ctx: *anyopaque) void {
    const ic: *InterleaveCtx = @ptrCast(@alignCast(opaque_ctx));
    if (ic.ticks == 0) {
        log.debug("[interleave] engaged: decode ticks between prefill chunks\n", .{});
    }
    // Decode ticks keep the CUDA graphs a prefill dropped.
    const graphs_were = mlx.setCudaGraphs(true);
    defer _ = mlx.setCudaGraphs(graphs_were);
    const chunk_ns = ic.chunk_sw.read();
    const share = prefillDecodeShare();
    // Small pending requests first, so the ticks below include them; their time is hosted time. A
    // request that arrives during the owed ticks is admitted between them, so no newcomer waits
    // longer than the chunk in flight.
    var budget: BoundaryBudget = .{ .max_count = BOUNDARY_MAX_COUNT, .ns_cap = decodeShareBudgetNs(chunk_ns, share) };
    const nested_ns = admitSmallAtBoundary(ic, share, &budget);
    const first_ns = interleaveDecodeTick(ic.sch);
    var owed: OwedTickCtx = .{ .ic = ic, .share = share, .budget = &budget };
    const r = runOwedDecodeTicks(share, chunk_ns +| nested_ns, first_ns, &owed, admitThenTick);
    ic.ticks += r.ticks;
    ic.decode_ns +|= r.spent_ns +| nested_ns;
    const module_bytes = @import("cache/budget.zig").moduleStateBytes(ic.sch);
    {
        ic.sch.queue_mu.lockUncancelable(ic.sch.io);
        defer ic.sch.queue_mu.unlock(ic.sch.io);
        publishLiveKvResidency(ic.sch, ic.slot, module_bytes);
    }
    ic.chunk_sw.reset();
}

/// The interval since these slots' previous tick held other work (a prefill chunk, a media
/// step); the serial cell and the round-cost table must not fold it as a token's wall time.
/// Drop it; the next tick seeds afresh.
pub fn invalidateDecodeClocks(slots: []const *Slot) void {
    for (slots) |s| {
        if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
        if (s.legacy_gen) |*g| {
            g.invalidateSerialClock();
            g.invalidateRoundClock();
        }
    }
}

/// One decode tick for the streams currently decoding, run from INSIDE a
/// prefill (between chunks, and between the slots of one admitted batch).
/// Returns the tick's wall-clock ns (0 when no stream is active). The
/// prefilling slot is not in `decoding` yet, so the tick only advances OTHER
/// requests' Generators — same-thread MLX, no reentrancy into this prefill.
pub fn interleaveDecodeTick(sch: *Scheduler) u64 {
    var buf: [32]*Slot = undefined;
    var n: usize = 0;
    sch.queue_mu.lockUncancelable(sch.io);
    for (sch.decoding.items) |s| {
        if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
        if (n >= buf.len) break;
        buf[n] = s;
        _ = s.in_pass.fetchAdd(1, .acq_rel);
        n += 1;
    }
    sch.queue_mu.unlock(sch.io);
    defer for (buf[0..n]) |s| {
        _ = s.in_pass.fetchSub(1, .acq_rel);
    };
    if (n == 0) return 0;
    invalidateDecodeClocks(buf[0..n]);
    var sw = io_util.Stopwatch.init(sch.io);
    runDecodeTick(sch, buf[0..n]) catch |err| {
        log.err("[interleave] decode tick failed: {s}\n", .{if (err == error.EngineFailed) err_name.last() else @errorName(err)});
        for (buf[0..n]) |s| s.markError(@errorName(err));
    };
    const tick_ns = sw.read();
    for (buf[0..n]) |s| s.decode_ns +|= tick_ns;
    return tick_ns;
}

// ── tests ──

test "module disk tiers never arm or execute KV prefill write-through" {
    const disk = @import("cache/ssd/disk_tier.zig");
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(testing.io, &buf)];
    const Check = struct {
        fn header(_: ?*anyopaque, _: []const u8, _: *const disk.module_record.Envelope) bool {
            return true;
        }
    };
    var hc = prefix_cache_mod.HotPrefixCache.init(testing.allocator, 4);
    defer hc.deinit();
    hc.disk = try disk.DiskTier.initWithModule(testing.allocator, testing.io, base, "module", 0, 512, .{ .identity = "fixture", .validate = &Check.header });
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_first = true;
    var entry: @import("../../host/registry/registry.zig").LoadedModel = undefined;
    entry.prefix_cache = &hc;
    var slot: Slot = undefined;
    slot.model = &entry;
    var prompt: [512]u32 = @splat(3);
    slot.full_prompt = &prompt;
    var ctx: WriteThroughCtx = .{ .slot = &slot };
    try testing.expect(!writeThroughArmed(&slot, 512));
    prefillWriteThroughCb(&ctx, 512, &.{});
    try testing.expectEqual(@as(u32, 0), ctx.chunks);
    try testing.expectEqual(@as(usize, 0), hc.disk.?.entries.items.len);
    try testing.expectEqual(@as(u64, 0), hc.disk.?.bytes_persisted);
}

test "boundaryEligible: a small plain request is admitted only under a share, never nested, never beside an exclusive model" {
    const base_cand: BoundaryCand = .{ .exclusive = false, .has_media = false, .prompt_logprobs = false, .generator_path = true, .module_state = false, .tail = 31 };
    const base_ctx: BoundaryCtx = .{ .share = 0.5, .nested = false, .outer_exclusive = false, .decoding = 2, .max_concurrent = 8, .max_tail = BOUNDARY_MAX_TAIL };
    try testing.expect(boundaryEligible(base_cand, base_ctx));

    var x = base_ctx;
    x.share = 0;
    try testing.expect(!boundaryEligible(base_cand, x));
    x = base_ctx;
    x.nested = true;
    try testing.expect(!boundaryEligible(base_cand, x));
    x = base_ctx;
    x.outer_exclusive = true;
    try testing.expect(!boundaryEligible(base_cand, x));
    x = base_ctx;
    x.decoding = x.max_concurrent;
    try testing.expect(!boundaryEligible(base_cand, x));

    var c = base_cand;
    c.exclusive = true;
    try testing.expect(!boundaryEligible(c, base_ctx));
    c = base_cand;
    c.has_media = true;
    try testing.expect(!boundaryEligible(c, base_ctx));
    c = base_cand;
    c.prompt_logprobs = true;
    try testing.expect(!boundaryEligible(c, base_ctx));
    c = base_cand;
    c.generator_path = false;
    try testing.expect(!boundaryEligible(c, base_ctx));
    c = base_cand;
    c.module_state = true;
    try testing.expect(!boundaryEligible(c, base_ctx));

    // The limit is inclusive.
    c = base_cand;
    c.tail = base_ctx.max_tail;
    try testing.expect(boundaryEligible(c, base_ctx));
    c.tail = base_ctx.max_tail + 1;
    try testing.expect(!boundaryEligible(c, base_ctx));
}

test "BoundaryBudget: stops at the count or the time cap, whichever comes first" {
    var b: BoundaryBudget = .{ .max_count = 2, .ns_cap = 1000 };
    try testing.expect(b.mayStart());
    b.charge(10);
    try testing.expect(b.mayStart());
    b.charge(10);
    try testing.expect(!b.mayStart());

    var t: BoundaryBudget = .{ .max_count = 5, .ns_cap = 100 };
    t.charge(100);
    try testing.expect(!t.mayStart());

    var s: BoundaryBudget = .{ .max_count = 5, .ns_cap = std.math.maxInt(u64), .spent_ns = std.math.maxInt(u64) - 1 };
    s.charge(10);
    try testing.expectEqual(@as(u64, std.math.maxInt(u64)), s.spent_ns);
}

test "orderByTail: ascending by uncached tail, stable for equal tails" {
    const tails = [_]usize{ 500, 31, 31, 2000 };
    var order: [4]usize = undefined;
    orderByTail(&tails, &order);
    try testing.expectEqualSlices(usize, &.{ 1, 2, 0, 3 }, &order);

    var one: [1]usize = undefined;
    orderByTail(&[_]usize{7}, &one);
    try testing.expectEqualSlices(usize, &.{0}, &one);
    orderByTail(&[_]usize{}, order[0..0]);
}

test "MLX_SERVE_SSD_WRITE_THROUGH=0 takes mechanism 3 out of the prefill, and only that" {
    try testing.expect(writeThroughEnabledFromEnv(null));
    try testing.expect(writeThroughEnabledFromEnv(""));
    try testing.expect(writeThroughEnabledFromEnv("1"));
    try testing.expect(writeThroughEnabledFromEnv("true"));
    try testing.expect(!writeThroughEnabledFromEnv("0"));
}

test "writeThroughSpanReached: a sub-chunk warm turn never persists inside the prefill" {
    const chunk: u32 = 1024;
    // Warm turn: a restored 32k prefix plus a 31-token instruction tail.
    try testing.expect(!writeThroughSpanReached(31, chunk));
    try testing.expect(!writeThroughSpanReached(0, chunk));
    try testing.expect(!writeThroughSpanReached(1023, chunk));
    try testing.expect(writeThroughSpanReached(1024, chunk));
    try testing.expect(writeThroughSpanReached(4096, chunk));
    try testing.expect(writeThroughSpanReached(65_665, chunk));
    try testing.expect(writeThroughSpanReached(4096, 4096));
    try testing.expect(!writeThroughSpanReached(4095, 4096));
}

test "companyStreak: only consecutive ticks with company count" {
    try testing.expectEqual(@as(u8, 2), companyStreak(companyStreak(0, true), true));
    try testing.expectEqual(@as(u8, 0), companyStreak(1, false));
}

test "yieldsToCompany: DSpark and sparse-target DFlash everywhere, PLD only on CUDA" {
    try testing.expect(yieldsToCompany(.dspark, false, false, 1, 5));
    try testing.expect(yieldsToCompany(.dspark, true, true, 1, 5));
    try testing.expect(yieldsToCompany(.dflash, false, true, 1, 5));
    try testing.expect(yieldsToCompany(.dflash, true, true, 1, 5));
    try testing.expect(yieldsToCompany(.pld, true, false, 1, 5));
    try testing.expect(!yieldsToCompany(.pld, false, false, 1, 5));
    for ([_]SpecTickMode{ .mtp, .drafter, .regular }) |m| try testing.expect(!yieldsToCompany(m, true, true, 9, 5));
}

test "yieldsToCompany: a dense DFlash slot keeps speculating until the others are a crowd" {
    try testing.expect(!yieldsToCompany(.dflash, false, false, 1, 5));
    try testing.expect(!yieldsToCompany(.dflash, true, false, 4, 5));
    try testing.expect(yieldsToCompany(.dflash, false, false, 5, 5));
    try testing.expect(yieldsToCompany(.dflash, false, false, 7, 5));
}

test "companyPrefillChunk: a prefill narrows only while someone decodes" {
    try testing.expectEqual(@as(u32, 8192), companyPrefillChunk(8192, 0));
    try testing.expectEqual(@as(u32, 2048), companyPrefillChunk(8192, 1));
    try testing.expectEqual(@as(u32, 1024), companyPrefillChunk(1024, 3));
    try testing.expectEqual(@as(u32, 2048), companyPrefillChunk(0, 1));
}

test "promptLogprobsWidthCap: an echo prefill narrows to its cap, 0 (uncapped) included" {
    try testing.expectEqual(@as(u32, 0), promptLogprobsWidthCap(false, 0));
    try testing.expectEqual(@as(u32, 1024), promptLogprobsWidthCap(false, 1024));
    try testing.expectEqual(PROMPT_LOGPROBS_CHUNK, promptLogprobsWidthCap(true, 0));
    try testing.expectEqual(PROMPT_LOGPROBS_CHUNK, promptLogprobsWidthCap(true, 1024));
    try testing.expectEqual(@as(u32, 128), promptLogprobsWidthCap(true, 128));
}

test "decode share: a live share caps the width only while someone decodes" {
    const w = DECODE_SHARE_PREFILL_CHUNK;
    try testing.expectEqual(@as(u32, 0), decodeShareAdmissionCap(0, 0.5));
    try testing.expectEqual(@as(u32, 0), decodeShareAdmissionCap(2, 0));
    try testing.expectEqual(w, decodeShareAdmissionCap(1, 0.5));
    try testing.expectEqual(@as(u32, 8192), decodeShareWidthCap(8192, 1, 0));
    try testing.expectEqual(@as(u32, 8192), decodeShareWidthCap(8192, 0, 0.5));
    try testing.expectEqual(w, decodeShareWidthCap(8192, 1, 0.5));
    try testing.expectEqual(@as(u32, 512), decodeShareWidthCap(512, 1, 0.5));
}

test "decode share: parse clamps above 0.9 and rejects anything that is not a share" {
    try testing.expectEqual(@as(f32, 0), try parseDecodeShare("0"));
    try testing.expectEqual(@as(f32, 0.5), try parseDecodeShare("0.5"));
    try testing.expectEqual(PREFILL_DECODE_SHARE_MAX, try parseDecodeShare("0.95"));
    try testing.expectEqual(PREFILL_DECODE_SHARE_MAX, try parseDecodeShare("inf"));
    for ([_][]const u8{ "", "abc", "nan", "-0.3", "0.5x" }) |bad| {
        try testing.expectError(error.InvalidDecodeShare, parseDecodeShare(bad));
    }
    // Unset is the default; a bad flag is an error, not 0.
    try testing.expectEqual(@as(f32, 0.3), try resolveDecodeShare("0.3"));
    try testing.expectEqual(DEFAULT_DECODE_SHARE, try resolveDecodeShare(null));
    try testing.expectEqual(@as(f32, 0), try resolveDecodeShare("0"));
    try testing.expectError(error.InvalidDecodeShare, resolveDecodeShare("abc"));
}

test "decode share: the budget is chunk * S / (1 - S), zero when off" {
    const ms = std.time.ns_per_ms;
    try testing.expectEqual(@as(u64, 0), decodeShareBudgetNs(400 * ms, 0));
    try testing.expectApproxEqAbs(@as(f64, 400 * ms), @as(f64, @floatFromInt(decodeShareBudgetNs(400 * ms, 0.5))), 1e3);
    try testing.expectApproxEqAbs(@as(f64, 3600 * ms), @as(f64, @floatFromInt(decodeShareBudgetNs(400 * ms, 0.9))), 1e4);
    try testing.expectApproxEqAbs(@as(f64, 400 * ms * 3 / 7), @as(f64, @floatFromInt(decodeShareBudgetNs(400 * ms, 0.3))), 1e4);
}

const FakeDecodeTicks = struct {
    left: u32,
    ns: u64,
    calls: u32 = 0,
    fn tick(ctx: *anyopaque) u64 {
        const f: *FakeDecodeTicks = @ptrCast(@alignCast(ctx));
        f.calls += 1;
        if (f.left == 0) return 0;
        f.left -= 1;
        return f.ns;
    }
};

test "decode share: a boundary ticks to the budget, keeps today's count at S=0, stops when decoders run out" {
    const ms = std.time.ns_per_ms;
    // S=0.5 on a 400 ms chunk with 8 ms ticks: 50 ticks, 400 ms of decode.
    var full = FakeDecodeTicks{ .left = 1000, .ns = 8 * ms };
    const r = runOwedDecodeTicks(0.5, 400 * ms, 8 * ms, &full, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 50), r.ticks);
    try testing.expectEqual(@as(u64, 400 * ms), r.spent_ns);
    // S=0 is exactly today's schedule, whatever the chunk cost.
    for ([_]u64{ 100, 300, 2300, 8000 }) |c| {
        var legacy = FakeDecodeTicks{ .left = 1000, .ns = 8 * ms };
        const l = runOwedDecodeTicks(0, c * ms, 8 * ms, &legacy, FakeDecodeTicks.tick);
        try testing.expectEqual(interleaveTicksFor(c * ms, 8 * ms), l.ticks);
    }
    // The decoders finish after 3 more ticks: the 4th call returns 0 and the loop stops.
    var dry = FakeDecodeTicks{ .left = 3, .ns = 8 * ms };
    const d = runOwedDecodeTicks(0.5, 400 * ms, 8 * ms, &dry, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 4), d.ticks);
    try testing.expectEqual(@as(u32, 4), dry.calls);
    try testing.expectEqual(@as(u64, 32 * ms), d.spent_ns);
    // Nobody decoding at the boundary: the first tick measured 0, no more are tried.
    var none = FakeDecodeTicks{ .left = 1000, .ns = 8 * ms };
    const n = runOwedDecodeTicks(0.5, 2300 * ms, 0, &none, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 1), n.ticks);
    try testing.expectEqual(@as(u32, 0), none.calls);
}

test "interleaveTicksFor: decode keeps a quarter of wall time across a slow chunk, capped" {
    const ms = std.time.ns_per_ms;
    try testing.expectEqual(@as(u32, 8), interleaveTicksFor(8000 * ms, 80 * ms));
    try testing.expectEqual(@as(u32, 2), interleaveTicksFor(300 * ms, 50 * ms));
    try testing.expectEqual(@as(u32, 1), interleaveTicksFor(100 * ms, 80 * ms));
    try testing.expectEqual(@as(u32, 1), interleaveTicksFor(8000 * ms, 0));
}
