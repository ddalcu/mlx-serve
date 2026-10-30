//! Batched-decode verdicts: which slots share a forward, padding waste, exclusivity, admission fit.

const std = @import("std");
const transformer_mod = @import("../transformer.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const model_registry_mod = @import("../../../host/registry/registry.zig");
const log = @import("../../../sdk/util/log.zig");
const KVCache = transformer_mod.KVCache;
const testing = std.testing;

const Scheduler = @import("../scheduler.zig").Scheduler;
const SIBLING_MARGIN_BYTES = @import("../scheduler/params.zig").SIBLING_MARGIN_BYTES;
const Slot = @import("../slot.zig").Slot;

/// Slots one batched forward can carry; the tail decodes serial this tick.
pub const MAX_BATCH_GROUP = @import("../../../host/gpu_thread/scheduler.zig").MAX_BATCH_GROUP;

pub const BatchVerdict = @import("../../../host/gpu_thread/scheduler.zig").BatchVerdict;

/// Does the loaded model's config batch at all? The arch half of `batchVerdict`,
/// shared with `/props`, `/v1/models` and the serve-mode startup line.
pub fn configBatchesDecode(cfg: *const model_mod.ModelConfig) bool {
    return modelBatchable(cfg) or cfg.supportsBatchedGdnDecode();
}

/// One line per slot the first time it decodes serial beside live company;
/// the counter moves every tick so the rate is visible under `--metrics`.
pub fn noteSerial(sch: *Scheduler, slot: *Slot, why: BatchVerdict) void {
    if (sch.metrics) |m| m.decode_serial_total[@backingInt(why)].inc();
    if (slot.serial_reason_logged) return;
    slot.serial_reason_logged = true;
    log.info("[batched] slot serial: {s} (model={s})\n", .{ @tagName(why), slot.model.id });
}

/// Batched decode pads every slot's KV to the group's LONGEST (`padAndStackBatchedKV`),
/// so the tensor it builds is `N x kv_max`, not `sum(kv_len)`. A group mixing one
/// 100k-token stream with three 1k ones therefore materializes ~100x the bytes the
/// short slots need — on a qwen3_5 trunk (hd 256, `--ctx-size` up to 262144) that is
/// ~410 MB per full-attention layer, ~6.5 GB across the 16 of them, and NOTHING bills
/// it: it is a per-tick transient, invisible to `prefillTransientReserve` and to the
/// load-time gate. An uncatchable Metal OOM is the failure mode.
///
/// So the group is capped by PADDING WASTE, which is the quantity that actually hurts,
/// rather than by a length ratio: keep the largest prefix of the ascending-sorted
/// lengths whose padded tensor stays within `MAX_PAD_WASTE` of its useful bytes. The
/// slots that fall out are the LONGEST ones — the ones dominating kv_max — and they
/// decode serially this tick, so every slot still advances.
///
/// Returns how many of `kv_lens_asc` may batch together (0 or 1 = nobody batches).
/// The lengths are the arch's true attention KV length (`batchKvLenOf`), pre-tick counts,
/// so this is a heuristic bar on the padding the forward will build, not an accounting identity.
/// The padded tensor may be at most this multiple of the bytes the group
/// actually needs. It must stay BELOW 2.0 or a two-slot group can never be
/// vetoed: one 1-token slot beside one 200k slot pads to exactly 2x, which is
/// the worst case for N=2 and the pathological pair this cap exists for.
pub const MAX_PAD_WASTE: f64 = 1.5;

pub fn batchedKvKeepCount(kv_lens_asc: []const u32) usize {
    if (kv_lens_asc.len < 2) return 0;
    var k: usize = kv_lens_asc.len;
    while (k >= 2) : (k -= 1) {
        var sum: u64 = 0;
        for (kv_lens_asc[0..k]) |l| sum += l;
        if (sum == 0) return k; // nothing prefilled yet: no padding to waste
        const padded: f64 = @floatFromInt(@as(u64, k) * kv_lens_asc[k - 1]);
        if (padded <= MAX_PAD_WASTE * @as(f64, @floatFromInt(sum))) return k;
    }
    return 0;
}

/// A group whose longest billed slot is at most this long pads a tensor that is small in absolute
/// terms, and splitting it sends a slot through a whole serial forward instead. Under the qwen4
/// gather arm every long slot bills at the indexer budget (2052 rows), so a short sub-agent beside
/// long streams reads as a 2x pad and was split off.
pub const PAD_FREE_KV: u32 = 4096;

/// The floor above only engages once some slot in the group holds at least this much context:
/// below it the serial forward it saves is cheap and keeping the group whole does not pay.
pub const PAD_FREE_MIN_CTX: u32 = 131072;

/// Does the small-pad floor keep this group whole? `billed_max` is the longest billed length,
/// `ctx_max` the longest true context.
pub fn padFreeFloorApplies(billed_max: u32, ctx_max: u32) bool {
    return ctx_max >= PAD_FREE_MIN_CTX and billed_max <= PAD_FREE_KV;
}

/// `batchedKvKeepCount` for a group whose forward may attend per slot: past the per-slot
/// floor nothing is padded, so every slot batches. `ctx_max` is the group's longest true context.
pub fn groupKeepCount(kv_lens_asc: []const u32, per_slot_capable: bool, ctx_max: u32) usize {
    if (kv_lens_asc.len >= 2) {
        const billed_max = kv_lens_asc[kv_lens_asc.len - 1];
        if (per_slot_capable and billed_max >= transformer_mod.Transformer.BATCHED_PER_SLOT_ATTN_MIN_KV) return kv_lens_asc.len;
        if (padFreeFloorApplies(billed_max, ctx_max)) return kv_lens_asc.len;
    }
    return batchedKvKeepCount(kv_lens_asc);
}

/// The longest true context (not the billed length) among the group's caches.
pub fn groupCtxMax(caches: []const *const KVCache) u32 {
    var m: usize = 0;
    for (caches) |c| m = @max(m, c.kvLenForBatching());
    return @intCast(@min(m, std.math.maxInt(u32)));
}

/// Does the batched forward pad every slot's KV to the group's longest? GLM-5-Next attends per
/// slot at any length, so it has no padding to cap.
pub fn groupPadsKv(slot: *const Slot) bool {
    const cfg = slot.model.config orelse return true;
    return cfg.caps().batch_pads_kv;
}

/// The per-slot attention arm serves every batched trunk but qwen4's QSA reads.
pub fn groupAttendsPerSlot(slot: *const Slot) bool {
    const cfg = slot.model.config orelse return true;
    return !cfg.longCtxGated();
}

/// The padding waste the whole group would pay; reported by the cap's log.
pub fn batchedPadWaste(kv_lens_asc: []const u32) f64 {
    var sum: u64 = 0;
    for (kv_lens_asc) |l| sum += l;
    if (sum == 0 or kv_lens_asc.len == 0) return 1.0;
    const padded: f64 = @floatFromInt(@as(u64, kv_lens_asc.len) * kv_lens_asc[kv_lens_asc.len - 1]);
    return padded / @as(f64, @floatFromInt(sum));
}

/// One source for the length the batched group is sorted and capped by. `cache.step`
/// advances only on global layer 0, so on a linear-layer-0 trunk (GDN, gated-conv, Mamba2,
/// KDA) it is 0 forever and the pad-waste cap never fired. `KVCache.kvLenForBatching` reads
/// the first attention layer's own offset there.
pub fn batchKvLenOf(cache: *const KVCache, cfg: ?*const model_mod.ModelConfig) u32 {
    return batchKvLenOfWith(cache, cfg, 1);
}

pub fn batchKvLenOfWith(cache: *const KVCache, cfg: ?*const model_mod.ModelConfig, seq_len: c_int) u32 {
    const raw: u32 = @intCast(cache.kvLenForBatching());
    const c = cfg orelse return raw;
    if (!c.longCtxGated()) return raw;
    const gather_on = transformer_mod.qsaBatchedGatherOn(seq_len);
    const min_kv: u32 = @intCast(transformer_mod.qsaBatchedGatherFloor(seq_len, cache.config.scheme == .affine));
    return c.batchedEffectiveKvLen(raw, gather_on, min_kv);
}

pub fn fillGroupPadWasteKvLens(
    caches: []const *const KVCache,
    cfg: ?*const model_mod.ModelConfig,
    seq_len: c_int,
    out: []u32,
) void {
    for (caches, 0..) |c, i| out[i] = batchKvLenOfWith(c, cfg, seq_len);
}

/// Pure-config predicate: is this model's architecture compatible with the
/// batched-decode kernel? Used by `Scheduler.batchable` after slot-level
/// flags are checked. MoE / hybrid / encoder have shape mismatches with
/// `forwardBatchedDecode` and fall through to per-slot dispatch.
pub fn modelBatchable(cfg: *const model_mod.ModelConfig) bool {
    if (cfg.plugin) |b| return b.parsed.caps.batches_decode and b.parsed.cache.ssm_state.entries == 0;
    if (cfg.has_hybrid_layers) return false;
    if (cfg.full_attention_interval > 0) return false;
    if (cfg.is_encoder_only) return false;
    if (cfg.isMoe()) return false;
    return true;
}

/// A model whose per-request decode state is MODULE-OWNED (one per model,
/// not per slot) — at most ONE in-flight request may touch it. dsv4 today:
/// `Dsv4Model.dec_state` is rebuilt at cache.step==0 and advanced by every
/// decode tick, so a second interleaved slot deinit+rebuilds the active
/// request's state and both then append tokens to the ONE state (live
/// an app chat leaked into a pi stream, every word doubled).
/// Serial-tick interleave stays safe for per-slot-state archs (laguna/hy3)
/// — ask the transformer which archs own their decode state, NEVER key on
/// !modelBatchable, and never name ONE arch here: when a second module-owned
/// arch arrived, this function's hardcoded `dsv4` check silently never
/// reached the gate for it.
fn modelExclusiveDecode(model: *const model_registry_mod.LoadedModel) bool {
    const t = model.transformer orelse return false;
    if (t.config.plugin) |p| if (p.parsed.caps.slot_state) return false;
    return t.ownsModuleDecodeState();
}

/// Pure core of `slotExclusiveDecode`. `model_owns_state` (dsv4) wins outright and is never
/// released; the head clause is qwen4_exp's per-model head, released for good once the
/// adaptive switch moved the slot to serial (`Generator.mtpModuleHeadReleased`).
pub fn headExclusiveFor(
    model_owns_state: bool,
    head_module_owned: bool,
    slot_enable_mtp: bool,
    head_released: bool,
) bool {
    if (model_owns_state) return true;
    if (!head_module_owned or !slot_enable_mtp) return false;
    return !head_released;
}

/// Per-SLOT exclusivity: the model's own bit, OR a slot that will drive a
/// module-owned MTP head (qwen4: `Qwen4Mtp.cache` is one per model). Plain
/// slots on the same model keep interleaving/batching beside it; two MTP
/// slots serialize, until one of them releases the head.
pub fn slotExclusiveDecode(slot: *const Slot) bool {
    const head = slot.model.mtp;
    return headExclusiveFor(
        modelExclusiveDecode(slot.model),
        if (head) |h| h.moduleOwned() else false,
        slot.enable_mtp,
        if (slot.legacy_gen) |*g| g.mtpModuleHeadReleased() else false,
    );
}

/// A pending prefill that does not fit beside live requests waits for them to finish. Alone it
/// proceeds: the connection thread already admitted it, and nobody would ever free memory for it.
pub fn holdsForMemory(fits: bool, live_company: bool) bool {
    return !fits and live_company;
}

/// Does a prefill needing `need` fit `avail` beside siblings' unallocated promises, a running
/// media job's reserve, and the sibling margin? Metal OOM is uncatchable, so a media step's
/// transients are billed before chat may take them.
pub fn admissionFits(need: u64, avail: u64, promised: u64, gen_reserve: u64, live: usize) bool {
    const margin: u64 = if (live > 0) SIBLING_MARGIN_BYTES else 0;
    return need +| promised +| gen_reserve +| margin <= avail;
}

pub fn liveDecodingCount(sch: *Scheduler) usize {
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    var n: usize = 0;
    for (sch.decoding.items) |s| {
        if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
        n += 1;
    }
    return n;
}

// ── tests ──

test "a prefill that does not fit waits only while another request is live" {
    try testing.expect(holdsForMemory(false, true));
    try testing.expect(!holdsForMemory(false, false)); // alone: never waits, or it waits forever
    try testing.expect(!holdsForMemory(true, true));
}

test "a running media job's reserve is billed: a prefill that fits live memory alone holds" {
    const gb: u64 = 1 << 30;
    try testing.expect(admissionFits(10 * gb, 20 * gb, 0, 0, 0));
    try testing.expect(!admissionFits(10 * gb, 20 * gb, 0, 12 * gb, 0));
    try testing.expect(holdsForMemory(admissionFits(10 * gb, 20 * gb, 0, 12 * gb, 0), 12 * gb > 0));
    try testing.expect(!admissionFits(10 * gb, 20 * gb, 4 * gb, 7 * gb, 0)); // promises and reserve add
    try testing.expect(admissionFits(17 * gb, 20 * gb, 0, 0, 1));
    try testing.expect(!admissionFits(19 * gb, 20 * gb, 0, 0, 1)); // a sibling keeps its margin
}

test "modelBatchable rejects MoE / hybrid / encoder / sliding-window" {
    {
        var cfg = std.mem.zeroes(model_mod.ModelConfig);
        cfg.has_hybrid_layers = true;
        try testing.expect(!modelBatchable(&cfg));
    }
    {
        var cfg = std.mem.zeroes(model_mod.ModelConfig);
        cfg.full_attention_interval = 6;
        try testing.expect(!modelBatchable(&cfg));
    }
    {
        var cfg = std.mem.zeroes(model_mod.ModelConfig);
        cfg.is_encoder_only = true;
        try testing.expect(!modelBatchable(&cfg));
    }
    {
        // MoE: isMoe() returns true when num_experts > 0.
        var cfg = std.mem.zeroes(model_mod.ModelConfig);
        cfg.num_experts = 8;
        try testing.expect(!modelBatchable(&cfg));
    }
}

test "modelBatchable: a PARSED deepseek_v4 config can never route to batched decode" {
    // dsv4 is serial-only (module-owned per-request state); its exclusion
    // from `forwardBatchedDecode` rides isMoe(), so the parse arm must never
    // regress to leaving num_experts unset. Parse a minimal real-shaped
    // config rather than hand-building the struct.
    const json =
        \\{"model_type":"deepseek_v4","hidden_size":64,"num_hidden_layers":4,
        \\ "num_attention_heads":4,"num_key_value_heads":1,"head_dim":96,
        \\ "qk_rope_head_dim":32,"q_lora_rank":32,"o_lora_rank":16,"o_groups":2,
        \\ "sliding_window":8,"compress_ratios":[0,4,16,4],
        \\ "compress_rope_theta":160000.0,"rope_theta":10000.0,
        \\ "rope_scaling":{"factor":16,"original_max_position_embeddings":64,
        \\  "beta_fast":32,"beta_slow":1,"type":"yarn"},
        \\ "index_n_heads":2,"index_head_dim":32,"index_topk":4,
        \\ "n_routed_experts":256,"num_experts_per_tok":6,"num_hash_layers":1,
        \\ "n_shared_experts":1,"moe_intermediate_size":32,
        \\ "routed_scaling_factor":1.5,"swiglu_limit":10.0,"norm_topk_prob":true,
        \\ "scoring_func":"sqrtsoftplus","topk_method":"noaux_tc","hc_mult":4,
        \\ "hc_sinkhorn_iters":20,"hc_eps":1e-6,"rms_norm_eps":1e-6,
        \\ "vocab_size":64,"max_position_embeddings":4096}
    ;
    const cfg = try model_mod.parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("deepseek_v4", cfg.model_type);
    try testing.expect(cfg.isMoe());
    try testing.expect(!modelBatchable(&cfg));
}

test "batchedKvKeepCount: padding waste caps the group, and the long slots are the ones dropped" {
    // Even lengths: no padding waste, everybody batches.
    try testing.expectEqual(@as(usize, 4), batchedKvKeepCount(&[_]u32{ 1000, 1000, 1000, 1000 }));
    try testing.expectEqual(@as(usize, 4), batchedKvKeepCount(&[_]u32{ 900, 1000, 1100, 1200 }));

    // The case this exists for: three short streams and one long one. Batching
    // all four pads to 4 x 100000 = 400k against 103k useful (3.9x); dropping
    // the long one leaves 3 x 1000 vs 3000 (1.0x).
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 1000, 1000, 1000, 100_000 }));

    // Two long ones: the pair still batches together, since 2 x 100000 against
    // 200000 useful wastes nothing — the veto is about the padding, not length.
    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&[_]u32{ 100_000, 100_000 }));
    // One short slot among two long ones still batches: 3 x 100000 padded
    // against 200010 useful is 1.5x, inside the bar. The veto is about the
    // WASTE the padding creates, not about any slot being an outlier.
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 10, 100_000, 100_000 }));

    // One slot never "batches", and neither does an empty group.
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&[_]u32{1000}));
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&[_]u32{}));

    // Nothing prefilled yet: no padding to waste, so nothing is vetoed (the
    // ready gate, not this one, is what keeps unprefilled slots out).
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 0, 0, 0 }));

    // A pathological pair degrades to no batch at all rather than making the
    // 1-token slot build a 200k-wide tensor nobody billed. This is why the bar
    // has to sit below 2.0 — at 2.0 a pair is unvetoable by construction.
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&[_]u32{ 1, 200_000 }));
}

test "groupKeepCount: a group that attends per slot pads nothing, so the cap is the stacked arm's" {
    const skew = [_]u32{ 1000, 1000, 1000, 100_000 };
    try testing.expectEqual(@as(usize, 3), groupKeepCount(&skew, false, 0));
    try testing.expectEqual(@as(usize, 4), groupKeepCount(&skew, true, 0));
    // Below the per-slot floor the stacked arm runs and its cap holds.
    try testing.expectEqual(@as(usize, 0), groupKeepCount(&[_]u32{ 10, 900 }, true, 0));
}

test "groupKeepCount: the small-pad floor keeps a group whole only at long context" {
    // A long stream billed at the gather arm's 2052-row cap beside two short sub-agents: 3x pad.
    const billed = [_]u32{ 10, 10, 2052 };
    try testing.expectEqual(@as(usize, 2), groupKeepCount(&billed, false, PAD_FREE_MIN_CTX - 1));
    try testing.expectEqual(@as(usize, 3), groupKeepCount(&billed, false, PAD_FREE_MIN_CTX));
    // Past the floor's billed bound the cap holds at any context.
    try testing.expectEqual(@as(usize, 0), groupKeepCount(&[_]u32{ 1, PAD_FREE_KV + 1 }, false, 300_000));
    try testing.expectEqual(@as(usize, 2), groupKeepCount(&[_]u32{ 1, PAD_FREE_KV }, false, 300_000));
}

test "the pad-waste cap reads the arch's TRUE attention KV length, not cache.step" {
    // `cache.step` is 0 forever on a linear-layer-0 trunk, so the cap never fired there.
    const lens = [_]usize{ 1_000, 1_000, 60_000 };
    const first_attn = [_]u32{ 3, 2, 7 }; // GDN / gated-conv / KDA spacings
    var caches: [3]KVCache = undefined;
    var built: usize = 0;
    defer for (caches[0..built]) |*c| c.deinit();
    for (lens, first_attn, 0..) |len, fa, i| {
        caches[i] = try KVCache.init(testing.allocator, 32);
        built += 1;
        caches[i].entries[fa].initialized = true;
        caches[i].entries[fa].offset = len;
        try testing.expectEqual(@as(usize, 0), caches[i].step); // the trap
    }

    var q4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .legacy = .{ .caps = .{ .long_ctx = true } } };
    var kv_lens: [3]u32 = undefined;
    for (caches[0..], 0..) |*c, i| kv_lens[i] = batchKvLenOf(c, &q4);
    try testing.expectEqualSlices(u32, &[_]u32{ 1_000, 1_000, 60_000 }, &kv_lens);

    // Every linear-layer-0 trunk reads the same length: four equal slots must all keep batching.
    for ([_][]const u8{ "qwen3_5", "qwen3_5_moe", "qwen3_next", "lfm2", "nemotron_h", "bailing_hybrid" }) |mt| {
        var cfg = model_mod.ModelConfig{ .model_type = mt };
        for (caches[0..], lens) |*c, len| try testing.expectEqual(@as(u32, @intCast(len)), batchKvLenOf(c, &cfg));
    }
    for (caches[0..], lens) |*c, len| try testing.expectEqual(@as(u32, @intCast(len)), batchKvLenOf(c, null));

    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&kv_lens));
    try testing.expect(batchedPadWaste(&kv_lens) > MAX_PAD_WASTE);

    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 0, 0, 0 }));
    try testing.expectEqual(@as(f64, 1.0), batchedPadWaste(&[_]u32{ 0, 0, 0 }));
}

test "an attention-first trunk's batching lengths are unchanged by the fix" {
    // Dense / attention-first archs are byte-identical across this change.
    const lens = [_]usize{ 1_000, 1_000, 1_000, 100_000 };
    var llama = model_mod.ModelConfig{ .model_type = "llama" };
    var caches: [4]KVCache = undefined;
    var built: usize = 0;
    defer for (caches[0..built]) |*c| c.deinit();
    var kv_lens: [4]u32 = undefined;
    for (lens, 0..) |len, i| {
        caches[i] = try KVCache.init(testing.allocator, 8);
        built += 1;
        caches[i].step = len;
        caches[i].entries[0].initialized = true;
        caches[i].entries[0].offset = len;
        kv_lens[i] = batchKvLenOf(&caches[i], &llama);
    }
    try testing.expectEqualSlices(u32, &[_]u32{ 1_000, 1_000, 1_000, 100_000 }, &kv_lens);
    var q4b = model_mod.ModelConfig{ .model_type = "qwen4_exp", .legacy = .{ .caps = .{ .long_ctx = true } } };
    for (caches[0..], 0..) |*c, i| try testing.expectEqual(kv_lens[i], batchKvLenOf(c, &q4b));
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&kv_lens));
    try testing.expectEqual(
        batchedKvKeepCount(&[_]u32{ 1_000, 1_000, 1_000, 100_000 }),
        batchedKvKeepCount(&kv_lens),
    );
}

test "batchKvLenOf bills qwen4 selected length when the gather arm is on" {
    const prev = transformer_mod.qsa_batched_gather_override;
    defer transformer_mod.qsa_batched_gather_override = prev;
    transformer_mod.qsa_batched_gather_override = true;
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .legacy = .{ .caps = .{ .long_ctx = true } },
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    var cache = try KVCache.init(testing.allocator, 32);
    defer cache.deinit();
    cache.entries[3].initialized = true;
    cache.entries[3].offset = 162_000;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOf(&cache, &q4));
    transformer_mod.qsa_batched_gather_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOf(&cache, &q4));
    var llama = model_mod.ModelConfig{ .model_type = "llama" };
    cache.step = 162_000;
    cache.entries[0].initialized = true;
    cache.entries[0].offset = 162_000;
    transformer_mod.qsa_batched_gather_override = true;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOf(&cache, &llama));
}

test "batchKvLenOf bills raw when any gather switch is off or the slot is vision" {
    const prev_b = transformer_mod.qsa_batched_gather_override;
    const prev_g = transformer_mod.qsa_gather_override;
    const prev_d = transformer_mod.qsa_decode_gather_override;
    const prev_v = @import("../../../plugins/qwen4_exp/qsa/verify_gather.zig").qsa_verify_gather_override;
    defer {
        transformer_mod.qsa_batched_gather_override = prev_b;
        transformer_mod.qsa_gather_override = prev_g;
        transformer_mod.qsa_decode_gather_override = prev_d;
        @import("../../../plugins/qwen4_exp/qsa/verify_gather.zig").qsa_verify_gather_override = prev_v;
    }
    transformer_mod.qsa_batched_gather_override = true;
    transformer_mod.qsa_gather_override = true;
    transformer_mod.qsa_decode_gather_override = true;
    @import("../../../plugins/qwen4_exp/qsa/verify_gather.zig").qsa_verify_gather_override = true;
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .legacy = .{ .caps = .{ .long_ctx = true } },
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    var cache = try KVCache.init(testing.allocator, 32);
    defer cache.deinit();
    cache.entries[3].initialized = true;
    cache.entries[3].offset = 162_000;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 1));
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 4));
    transformer_mod.qsa_batched_gather_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOfWith(&cache, &q4, 1));
    transformer_mod.qsa_batched_gather_override = true;
    transformer_mod.qsa_gather_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOfWith(&cache, &q4, 1));
    transformer_mod.qsa_gather_override = true;
    transformer_mod.qsa_decode_gather_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOfWith(&cache, &q4, 1));
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 4));
    transformer_mod.qsa_decode_gather_override = true;
    @import("../../../plugins/qwen4_exp/qsa/verify_gather.zig").qsa_verify_gather_override = false;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 1));
    // The fused verify kernel still serves S=4 on blocks; with it off too, the mask bills raw kv.
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 4));
    const prev_k = transformer_mod.qsa_attn_kernel_override;
    defer transformer_mod.qsa_attn_kernel_override = prev_k;
    transformer_mod.qsa_attn_kernel_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOfWith(&cache, &q4, 4));
}

test "a 300k slot beside a 1k slot is billed sparse, so the pair groups (M-RoPE slots no longer bill raw)" {
    const prev_b = transformer_mod.qsa_batched_gather_override;
    const prev_g = transformer_mod.qsa_gather_override;
    const prev_d = transformer_mod.qsa_decode_gather_override;
    defer {
        transformer_mod.qsa_batched_gather_override = prev_b;
        transformer_mod.qsa_gather_override = prev_g;
        transformer_mod.qsa_decode_gather_override = prev_d;
    }
    transformer_mod.qsa_batched_gather_override = true;
    transformer_mod.qsa_gather_override = true;
    transformer_mod.qsa_decode_gather_override = true;
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .legacy = .{ .caps = .{ .long_ctx = true } },
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    var caches: [2]KVCache = undefined;
    var built: usize = 0;
    defer for (caches[0..built]) |*c| c.deinit();
    caches[0] = try KVCache.init(testing.allocator, 32);
    built = 1;
    caches[0].entries[3].initialized = true;
    caches[0].entries[3].offset = 300_000;
    caches[1] = try KVCache.init(testing.allocator, 32);
    built = 2;
    caches[1].entries[3].initialized = true;
    caches[1].entries[3].offset = 1_000;
    const ptrs = [_]*const KVCache{ &caches[0], &caches[1] };
    var billed: [2]u32 = undefined;
    fillGroupPadWasteKvLens(&ptrs, &q4, 1, &billed);
    try testing.expectEqual(@as(u32, 2052), billed[0]);
    try testing.expectEqual(@as(u32, 1_000), billed[1]);
    var billed_asc = billed;
    std.mem.sort(u32, &billed_asc, {}, std.sort.asc(u32));
    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&billed_asc));
}

test "S>=2 pad-waste floor: none under the fused verify kernel, else max of gather and verify mins" {
    const prev_b = transformer_mod.qsa_batched_gather_override;
    const prev_g = transformer_mod.qsa_gather_override;
    const prev_v = @import("../../../plugins/qwen4_exp/qsa/verify_gather.zig").qsa_verify_gather_override;
    const prev_gm = transformer_mod.qsa_gather_min_kv_override;
    const prev_vm = @import("../../../plugins/qwen4_exp/qsa/verify_gather.zig").qsa_verify_gather_min_kv_override;
    defer {
        transformer_mod.qsa_batched_gather_override = prev_b;
        transformer_mod.qsa_gather_override = prev_g;
        @import("../../../plugins/qwen4_exp/qsa/verify_gather.zig").qsa_verify_gather_override = prev_v;
        transformer_mod.qsa_gather_min_kv_override = prev_gm;
        @import("../../../plugins/qwen4_exp/qsa/verify_gather.zig").qsa_verify_gather_min_kv_override = prev_vm;
    }
    transformer_mod.qsa_batched_gather_override = true;
    transformer_mod.qsa_gather_override = true;
    @import("../../../plugins/qwen4_exp/qsa/verify_gather.zig").qsa_verify_gather_override = true;
    transformer_mod.qsa_gather_min_kv_override = 20_000;
    @import("../../../plugins/qwen4_exp/qsa/verify_gather.zig").qsa_verify_gather_min_kv_override = 16_384;
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .legacy = .{ .caps = .{ .long_ctx = true } },
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    var cache = try KVCache.init(testing.allocator, 32);
    defer cache.deinit();
    cache.entries[3].initialized = true;
    // The fused verify kernel reads O(budget) keys per row at any kv: no floor at S=4.
    const prev_k = transformer_mod.qsa_attn_kernel_override;
    defer transformer_mod.qsa_attn_kernel_override = prev_k;
    transformer_mod.qsa_attn_kernel_override = null;
    cache.entries[3].offset = 18_000;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 4));
    // Its kill switch restores the union gather's floor: max of the gather and verify mins.
    transformer_mod.qsa_attn_kernel_override = false;
    try testing.expectEqual(@as(u32, 18_000), batchKvLenOfWith(&cache, &q4, 4));
    cache.entries[3].offset = 162_000;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 4));
}

test "batchedEffectiveKvLen: qwen4 bills selected length, other archs keep raw kv" {
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .legacy = .{ .caps = .{ .long_ctx = true } },
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    const min_kv: u32 = 8192;
    try testing.expectEqual(@as(u32, 2052), q4.batchedEffectiveKvLen(162_000, true, min_kv));
    try testing.expectEqual(@as(u32, 2052), q4.batchedEffectiveKvLen(64_000, true, min_kv));
    try testing.expectEqual(@as(u32, 8000), q4.batchedEffectiveKvLen(8000, true, min_kv));
    try testing.expectEqual(@as(u32, 162_000), q4.batchedEffectiveKvLen(162_000, false, min_kv));
    var llama = model_mod.ModelConfig{ .model_type = "llama" };
    try testing.expectEqual(@as(u32, 162_000), llama.batchedEffectiveKvLen(162_000, true, min_kv));
    var q35 = model_mod.ModelConfig{ .model_type = "qwen3_5" };
    try testing.expectEqual(@as(u32, 162_000), q35.batchedEffectiveKvLen(162_000, true, min_kv));

    const billed = [_]u32{
        q4.batchedEffectiveKvLen(16_384, true, min_kv),
        q4.batchedEffectiveKvLen(162_000, true, min_kv),
    };
    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&billed));
    const raw_pair = [_]u32{ 16_384, 162_000 };
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&raw_pair));
    const other = [_]u32{ 1_000, 162_000 };
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&other));
}

test "modelBatchable permits pure-attention" {
    // Defaults are all zero / null → vanilla pure-attention path.
    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    try testing.expect(modelBatchable(&cfg));
}

test "a GDN trunk is batchable AND is not clamped by the server's concurrency gate" {
    // The two sites that decide "does this model batch?" must agree. They
    // disagreed once: the scheduler batched qwen3_5 while server.zig still
    // clamped --max-concurrent to 1 for anything with
    // full_attention_interval > 0, so asking for concurrency turned the
    // batched path OFF. Both now read ModelConfig.supportsBatchedGdnDecode.
    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    cfg.model_type = "qwen3_5";
    cfg.full_attention_interval = 4;
    try testing.expect(cfg.supportsBatchedGdnDecode());

    // The pure-config gate rejects it (it IS a hybrid), which is exactly why
    // the GDN predicate has to be consulted beside it.
    try testing.expect(!modelBatchable(&cfg));

    // The server's startup line, /props and /v1/models read this one predicate.
    try testing.expect(configBatchesDecode(&cfg));
}

test "supportsBatchedGdnDecode refuses every arch the batched GDN path does not model" {
    // A new arch on the shared moe forward must default to SERIAL, not ride
    // a kernel that never modelled its state.
    {
        var moe = std.mem.zeroes(model_mod.ModelConfig);
        moe.model_type = "hy_v3";
        moe.full_attention_interval = 4;
        moe.num_experts = 128;
        moe.num_experts_per_tok = 8;
        try testing.expect(!moe.supportsBatchedGdnDecode());
    }
    {
        var lfm2 = std.mem.zeroes(model_mod.ModelConfig);
        lfm2.model_type = "lfm2";
        lfm2.has_hybrid_layers = true;
        try testing.expect(!lfm2.supportsBatchedGdnDecode());
    }
    {
        var kda = std.mem.zeroes(model_mod.ModelConfig);
        kda.model_type = "bailing_hybrid";
        kda.full_attention_interval = 4;
        kda.kda_vector_gate = true;
        try testing.expect(!kda.supportsBatchedGdnDecode());
    }
    {
        // Pure attention: not a GDN trunk at all, rides the standard kernel.
        var dense = std.mem.zeroes(model_mod.ModelConfig);
        dense.model_type = "qwen3";
        try testing.expect(!dense.supportsBatchedGdnDecode());
    }
    // Served models (qwen4_exp, GLM-5-Next, Inkling) answer from their plugin's caps.
}

test "S21: a released module head drops the slot's exclusivity, and the MODEL bit is untouched" {
    try testing.expect(headExclusiveFor(false, true, true, false));
    try testing.expect(!headExclusiveFor(false, true, true, true));
    try testing.expect(!headExclusiveFor(false, true, false, false));
    try testing.expect(!headExclusiveFor(false, true, false, true));
    try testing.expect(!headExclusiveFor(false, false, true, false));
    try testing.expect(!headExclusiveFor(false, false, true, true));
    try testing.expect(headExclusiveFor(true, false, false, true));
    try testing.expect(headExclusiveFor(true, true, true, true));
}
