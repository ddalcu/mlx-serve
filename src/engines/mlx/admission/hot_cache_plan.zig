//! prefix-cache and SSM checkpoint settings; hot-cache plan, SSD-first budgets, prefixCacheMemForLoad

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const log = @import("../../../sdk/util/log.zig");
const prefix_cache_mod = @import("../cache/prefix_cache.zig");
const scheduler_mod = @import("../scheduler.zig");
const testing = std.testing;
const http_server = @import("../../../host/http/server.zig");

const manualContext = @import("../../../host/http/server.zig").manualContext;
const autoContextFrom = @import("context.zig").autoContextFrom;
const getEffectiveContextLength = @import("context.zig").getEffectiveContextLength;
const staticGpuMemoryCeiling = @import("../../../host/registry/memory_ceiling.zig").staticGpuMemoryCeiling;
const currentGpuMemoryCeiling = @import("../../../host/registry/memory_ceiling.zig").currentGpuMemoryCeiling;
const safeContextForBudget = @import("context.zig").safeContextForBudget;
const defaultKvBits = @import("bills.zig").defaultKvBits;
const prefillTransientReserve = @import("bills.zig").prefillTransientReserve;
const PREFILL_CHUNK_LADDER = @import("prefill_chunk.zig").PREFILL_CHUNK_LADDER;
const explicitPrefillChunk = @import("prefill_chunk.zig").explicitPrefillChunk;
const billedPrefillChunk = @import("prefill_chunk.zig").billedPrefillChunk;
const sizerCtxKvBytes = @import("prefill_chunk.zig").sizerCtxKvBytes;
const pinPrefillChunk = @import("prefill_chunk.zig").pinPrefillChunk;
const longCtxTestConfig = @import("bills_test.zig").longCtxTestConfig;
const qwen4DeployedTestConfig = @import("bills_test.zig").qwen4DeployedTestConfig;
const ctxSizingCacheReserve = @import("context.zig").ctxSizingCacheReserve;
const slotFixedKvBytes = @import("bills.zig").slotFixedKvBytes;
const kvBytesPerTokenAtBits = @import("bills.zig").kvBytesPerTokenAtBits;
const statePerTokenBilled = @import("bills.zig").statePerTokenBilled;
const retainedSsmCheckpointBytes = @import("bills.zig").retainedSsmCheckpointBytes;
const sessionBytesPerToken = @import("bills.zig").sessionBytesPerToken;
const perRequestPrefillChunkEnabled = @import("prefill_chunk.zig").perRequestPrefillChunkEnabled;

const PREFIX_CACHE_MEM_DEFAULT = http_server.PREFIX_CACHE_MEM_DEFAULT;

/// The hot-cache ask when nobody named one: one session at the working context, never under
/// 2 GB. Below one session the cache keeps only a prefix of the longest conversations, the
/// ones whose reuse saves the most prefill. Any other ask (an operator's, an embedder's) stands.
/// Bytes one cached session at `ctx_tokens` holds: its KV and state, plus the SSM checkpoints
/// a cold prefill of that length retains, which the commit path bills to the entry.
pub fn oneSessionEntryBytes(config: *const model_mod.ModelConfig, kv_bits: u64, ctx_tokens: u64, chunk: u64) u64 {
    return sessionBytesPerToken(config, kv_bits) *| ctx_tokens +| slotFixedKvBytes(config, kv_bits, chunk) +|
        retainedSsmCheckpointBytes(config, ctx_tokens, 0, chunk);
}

pub fn defaultPrefixCacheAsk(requested: u64, explicit: bool, session_kv: u64) u64 {
    if (explicit or requested != PREFIX_CACHE_MEM_DEFAULT) return requested;
    return @max(requested, session_kv);
}

/// What the hot cache was actually given for the loaded model, after `clampedPrefixCacheMem`.
/// Every post-load reserve reads it through `resolvedPrefixCacheMem()`. Atomic: written on the
/// inference thread, read from connection threads. `maxInt(u64)` is the unresolved sentinel,
/// not 0 (`--prefix-cache-mem 0` under SSD-first publishes a real zero).
const HOT_CACHE_MEM_UNRESOLVED: u64 = std.math.maxInt(u64);

var hot_cache_mem_resolved = std.atomic.Value(u64).init(HOT_CACHE_MEM_UNRESOLVED);

/// The hot-cache byte budget every post-load reserve must bill: the clamp's answer once loaded, the raw ask before.
pub fn resolvedPrefixCacheMem() u64 {
    if (http_server.launch().prefix_cache_capacity == 0 or !http_server.launch().prefix_cache_ram_enabled) return 0;
    const v = hot_cache_mem_resolved.load(.monotonic);
    return if (v != HOT_CACHE_MEM_UNRESOLVED) v else http_server.launch().prefix_cache_mem_bytes;
}

/// Publish the clamp's answer (inference thread, end of a load); `clearResolvedPrefixCacheMem`
/// resets it on unload so the next model does not inherit this one's budget.
pub fn publishResolvedPrefixCacheMem(v: u64) void {
    hot_cache_mem_resolved.store(v, .monotonic);
}

pub fn clearResolvedPrefixCacheMem() void {
    hot_cache_mem_resolved.store(HOT_CACHE_MEM_UNRESOLVED, .monotonic);
}

/// SSM checkpoints feed reusable prefix tiers; zero capacity disables all reuse.
pub fn effectiveSsmCheckpointStride(stride: u32, cache_capacity: u32, ram_enabled: bool, disk_bytes: u64) u32 {
    if (cache_capacity == 0 or (!ram_enabled and disk_bytes == 0)) return 0;
    return stride;
}

/// PURE: clamp the hot prefix cache's byte budget to what the loaded weights
/// leave under the GPU ceiling after the serving context's KV and the prefill
/// transient reserve. A 40 GB `--prefix-cache-mem` beside a ~70 GB pack was
/// never validated against this headroom: the cache filled toward its cap and
/// a 143k prefill died in an uncatchable Metal OOM (Flash-Next on
/// 128 GB). `requested == 0` (byte cap disabled) is bounded too — an uncapped
/// cache beside a large model is exactly that crash. Never returns 0:
/// `initWithMem` reads 0 as "no byte cap", the opposite of no headroom.
pub fn clampedPrefixCacheMem(
    requested: u64,
    gpu_ceiling: u64,
    active_weights: u64,
    ctx_kv_bytes: u64,
    transient_reserve: u64,
) u64 {
    const headroom = @max(gpu_ceiling -| (active_weights +| ctx_kv_bytes +| transient_reserve), 1);
    if (requested == 0) return headroom;
    return @min(requested, headroom);
}

/// What the hot cache gets and what it was charged for getting it.
pub const HotCachePlan = struct {
    /// The width the sizer pinned.
    chunk: u32,
    /// The width the clamp reserved for; not `chunk` on an arch that re-bills per request (`clampReserveWidth`).
    reserve_chunk: u32,
    reserve: u64,
    ctx_kv: u64,
    /// The clamped byte budget handed to `initWithMem`.
    budget: u64,
};

/// The whole chain in the one order that is monotone in the ask: chunk first
/// (ask-independent), then the cache takes the residual.
/// The prefill width the load-time clamp reserves for: the ladder floor on an arch that
/// re-bills its width per request (the load-time reserve is a promise to the first request
/// only). Billing the sizer's rung made the clamp non-monotone in the ceiling (a bigger
/// ceiling bought a wider rung and a smaller cache: 3873 vs 1076 MB). An explicit
/// `--prefill-chunk` is billed as-is; ungated archs keep the sizer's rung.
pub fn clampReserveWidth(config: *const model_mod.ModelConfig, pinned_width: u32) u32 {
    if (explicitPrefillChunk() > 0) return pinned_width;
    if (!perRequestPrefillChunkEnabled(config)) return pinned_width;
    return PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
}

pub fn planHotCache(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    ceiling: u64,
    active_weights: u64,
    ctx_tokens: u64,
    sizer_ctx_kv: u64,
    requested: u64,
    chunk_override: u32,
) HotCachePlan {
    const chunk = billedPrefillChunk(config, kv_bits, ceiling, active_weights, sizer_ctx_kv, requested, chunk_override);
    const reserve_chunk = clampReserveWidth(config, chunk);
    const reserve = prefillTransientReserve(config, kv_bits, reserve_chunk);
    const ctx_kv: u64 = sessionBytesPerToken(config, kv_bits) *| ctx_tokens +| slotFixedKvBytes(config, kv_bits, chunk);
    return .{
        .chunk = chunk,
        .reserve_chunk = reserve_chunk,
        .reserve = reserve,
        .ctx_kv = ctx_kv,
        .budget = clampedPrefixCacheMem(requested, ceiling, active_weights, ctx_kv, reserve),
    };
}

/// SSD-first budget semantics. The RAM-first formula treats the hot cache as what is left
/// after the live session's reserve, which double-counts (the resident entry's arrays are
/// shared with the live KV). Here the budget floors at one entry at the working context and
/// `--prefix-cache-mem` is the RAM allowance for idle entries on top (0 = none idle).
pub fn ssdFirstPrefixCacheMem(
    idle_requested: u64,
    gpu_ceiling: u64,
    active_weights: u64,
    ctx_kv_bytes: u64,
    transient_reserve: u64,
) u64 {
    const headroom = gpu_ceiling -| (active_weights +| ctx_kv_bytes +| transient_reserve);
    const idle = @min(idle_requested, headroom);
    // Never 0: `initWithMem` reads 0 as "no byte cap".
    return @max(ctx_kv_bytes +| idle, 1);
}

/// The cache reserve the context sizer bills: at most half the room beside the weights and the
/// prefill, so a small Mac's context is not starved for a cache the load-time clamp shrinks anyway.
pub fn ctxCacheShare(cache_reserve: u64, ceiling: u64, active_mem: u64, transient_reserve: u64) u64 {
    return @min(cache_reserve, (ceiling -| (active_mem +| transient_reserve)) / 2);
}

/// The context a load-time bill must use, before `pinAutoContext` has run: on an auto boot
/// `getEffectiveContextLength` re-derives through `computeMemoryContext`, which reserves the
/// very ask being resolved and answered a 1024-token placeholder. `cache_reserve` is
/// `CTX_SIZING_CACHE_RESERVE` on both arms so the billed session is the advertised one; it
/// must never be the ask or anything derived from it. Runs `autoContextFrom`, the same helper
/// `autoContextFor` runs.
pub fn resolvedContextForLoad(
    explicit_ctx: u32,
    pinned_ctx: u32,
    ceiling: u64,
    active_mem: u64,
    cache_reserve: u64,
    transient_reserve: u64,
    per_tok: u64,
    ctx_cap: u32,
) u32 {
    if (explicit_ctx > 0) return explicit_ctx;
    if (pinned_ctx > 0) return pinned_ctx;
    return autoContextFrom(safeContextForBudget(
        ceiling,
        active_mem,
        ctxCacheShare(cache_reserve, ceiling, active_mem, transient_reserve) +| transient_reserve,
        per_tok,
        0,
    ), ctx_cap);
}

/// The SSD-first call site. Takes the static ceiling (the budget is a property of the machine),
/// the same reserve constant the RAM arm and `computeMemoryContext` pass, and the boot's
/// configured KV width (billing at bf16 on a `--kv-quant 8` boot was the 22,464-vs-13,824 MB defect).
pub fn ssdFirstSessionTokensNow(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u32) u32 {
    return resolvedContextForLoad(
        manualContext(config),
        config.pinned_context,
        ceiling,
        active_mem,
        ctxSizingCacheReserve(config),
        prefillTransientReserve(config, kv_bits, chunk),
        sessionBytesPerToken(config, kv_bits),
        config.contextCap(),
    );
}

/// The bytes the SSD-first budget floors at: one session at the working context, at the width the cache stores.
pub fn ssdFirstSessionKvBytes(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u32) u64 {
    return sessionBytesPerToken(config, kv_bits) *|
        ssdFirstSessionTokensNow(config, kv_bits, ceiling, active_mem, chunk) +| slotFixedKvBytes(config, kv_bits, chunk);
}

/// The SSD-first arm of `prefixCacheMemForLoad`, gate and log included. Null when the arch
/// gate is off; the caller falls through to `clampedPrefixCacheMem`.
fn ssdFirstBudgetForLoad(
    config: *model_mod.ModelConfig,
    requested: u64,
    ceiling: u64,
    active_mem: usize,
    ctx_kv: u64,
    transient_reserve: u64,
    idle_out: *u64,
    quiet: bool,
) ?u64 {
    // The predicate, shared with the spill site: without a disk tier the mode's floor would be RAM the server cannot use.
    if (http_server.launch().prefix_cache_capacity == 0 or !prefix_cache_mod.ssdFirstActive(config, http_server.launch().prefix_cache_disk_bytes > 0, http_server.launch().prefix_cache_ram_enabled)) return null;
    const budget = ssdFirstPrefixCacheMem(
        requested,
        ceiling,
        active_mem,
        ctx_kv,
        transient_reserve,
    );
    // Publish the cache's cost BEYOND the live session, not the whole budget: `budget` includes
    // `ctx_kv`, and `computeMemoryContext` subtracts the published figure, so publishing the
    // whole would bill the session twice and halve the auto-context.
    publishResolvedPrefixCacheMem(budget -| ctx_kv);
    // The idle allowance, handed to the cache for `spillIdleEntries` to enforce.
    idle_out.* = budget -| ctx_kv;
    // On this arch `--prefix-cache-mem` is the idle allowance, not the whole cache; say so once.
    if (quiet) return budget;
    log.info("[hot-cache] SSD-first budget {d} MB = one session at the working context ({d} MB) + {d} MB idle (--prefix-cache-mem {d} MB = the IDLE allowance on this arch; 0 = no idle entries)\n", .{
        budget >> 20,
        ctx_kv >> 20,
        (budget -| ctx_kv) >> 20,
        requested >> 20,
    });
    return budget;
}

/// Impure wrapper for the model-load site (`Scheduler.doLoadOnInferenceThread`,
/// reached through the LoadParams/LoadRequest resolver pointer — the scheduler
/// deliberately has no server.zig import): the weights are resident there, so
/// `mlx_get_active_memory` is honest. Logs one line when the clamp bites, and publishes the
/// answer so the auto-context sizer and the ANE gate stop reserving a budget the cache never got.
/// The hot-cache reserve the context is sized against at load time on the RAM-first arm: a
/// constant (the server's own `--prefix-cache-mem` default), never the ask. Reading the ask
/// let a 60GB ask collapse the advertised context to 870 tokens; context is the primary
/// claimant and the cache the residual.
pub const CTX_SIZING_CACHE_RESERVE: u64 = 2 * 1024 * 1024 * 1024;

/// The previous context-sizing cache reserve: the raw `--prefix-cache-mem` ask. Kept for
/// ungated archs so their advertised `context_length` does not move.
/// The ungated arm's ask. A sliding-ring entry cannot be trimmed below its ring (`ringFloor`) and a
/// hybrid entry restores only at a checkpoint, so a budget under one session keeps a stale prefix of
/// the longest conversations: those ask for one session at the working context.
fn ungatedHotCacheAsk(config: *const model_mod.ModelConfig, requested: u64, explicit: bool, kv_bits: u64, ctx_tokens: u64, chunk: u64) u64 {
    if (!config.slidingRing() and !config.hasSsmLayers()) return requested;
    return defaultPrefixCacheAsk(requested, explicit, oneSessionEntryBytes(config, kv_bits, ctx_tokens, chunk));
}

pub fn legacyPrefixCacheAsk() u64 {
    if (http_server.launch().prefix_cache_capacity == 0) return 0;
    return http_server.launch().prefix_cache_mem_bytes; // legacy_ask_read
}

/// The context the RAM-first clamp bills, through the one shared pure resolver.
fn ramFirstContextForLoad(config: *const model_mod.ModelConfig, kv_bits: u64, active_mem: u64, chunk: u32) u32 {
    return resolvedContextForLoad(
        manualContext(config),
        config.pinned_context,
        currentGpuMemoryCeiling(config, active_mem),
        active_mem,
        ctxSizingCacheReserve(config),
        prefillTransientReserve(config, kv_bits, chunk),
        sessionBytesPerToken(config, kv_bits),
        config.contextCap(),
    );
}

/// `revise` marks a post-load re-clamp (#364): the cache's own resident bytes come off the
/// machine read so it never bills itself, and the budget lines stay quiet.
pub fn prefixCacheMemForLoad(config: *model_mod.ModelConfig, requested: u64, revise: scheduler_mod.BudgetRevise, idle_out: *u64) u64 {
    // RAM-first default: no idle allowance. Written first so every early return carries it.
    idle_out.* = 0;
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    active_mem -|= @as(usize, @intCast(@min(revise.exclude_bytes, std.math.maxInt(usize))));
    const kv_bits: u64 = defaultKvBits(config);
    // Arch gate: the budget resolver changed three inputs at once (static ceiling, the
    // load-time context resolver, the floor-width reserve), all measured on qwen4_exp alone.
    // Ungated is the previous `prefixCacheMemForLoad`.
    if (!config.longCtxGated()) {
        const chunk: u64 = pinPrefillChunk(config);
        // `statePerTokenBilled` is 0 off qwen4_exp.
        const ctx_kv: u64 = (kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(config)) *|
            getEffectiveContextLength(config) +| slotFixedKvBytes(config, kv_bits, chunk);
        const ask = ungatedHotCacheAsk(config, requested, http_server.launch().prefix_cache_mem_explicit, kv_bits, getEffectiveContextLength(config), chunk);
        const clamped = clampedPrefixCacheMem(
            ask,
            currentGpuMemoryCeiling(config, active_mem),
            active_mem,
            ctx_kv,
            prefillTransientReserve(config, kv_bits, chunk),
        );
        // Publishing the resolved budget is kept on every arch: the ANE gate used to reserve the raw ask.
        publishResolvedPrefixCacheMem(clamped);
        if (revise.quiet) return clamped;
        if (ask > 0 and clamped < ask) {
            log.info("[hot-cache] budget clamped {d} -> {d} MB (weights + ctx KV + prefill reserve vs GPU ceiling)\n", .{ ask >> 20, clamped >> 20 });
        } else if (ask != requested) {
            log.info("[hot-cache] budget {d} MB (one session at the working context: a sliding-ring or hybrid entry restores whole or at a checkpoint)\n", .{clamped >> 20});
        } else if (requested == 0) {
            log.info("[hot-cache] budget capped at {d} MB (no --prefix-cache-mem; weights + ctx KV + prefill reserve vs GPU ceiling)\n", .{clamped >> 20});
        }
        return clamped;
    }
    // SSD-first takes the budget before the RAM-first plan: its resident entry IS the live KV.
    const ssd_chunk: u64 = pinPrefillChunk(config);
    // Not `getEffectiveContextLength`: on an auto boot it is still unpinned here.
    const ssd_ctx_kv: u64 = ssdFirstSessionKvBytes(config, kv_bits, staticGpuMemoryCeiling(), active_mem, @intCast(ssd_chunk));
    // Static ceiling on both arms (the budget is a property of the machine), and the clamp
    // reserve at the same width `planHotCache` reserves for: the ladder floor on an arch that
    // re-bills per request. The session is billed at the pinned width, the budget at the floor.
    const ssd_clamp_reserve: u64 = prefillTransientReserve(config, kv_bits, clampReserveWidth(config, @intCast(ssd_chunk)));
    if (ssdFirstBudgetForLoad(config, requested, staticGpuMemoryCeiling(), active_mem, ssd_ctx_kv, ssd_clamp_reserve, idle_out, revise.quiet)) |b| return b;
    // Pin first, then hand the pinned width in as the override.
    const pinned: u32 = pinPrefillChunk(config);
    // Not `getEffectiveContextLength`: still a placeholder on an auto boot.
    const ctx_tokens = ramFirstContextForLoad(config, kv_bits, active_mem, pinned);
    const ask = defaultPrefixCacheAsk(requested, http_server.launch().prefix_cache_mem_explicit, oneSessionEntryBytes(config, kv_bits, ctx_tokens, pinned));
    // Static ceiling, not the live one: the budget must be reproducible boot to boot.
    const plan = planHotCache(
        config,
        kv_bits,
        staticGpuMemoryCeiling(),
        active_mem,
        ctx_tokens,
        sizerCtxKvBytes(config, kv_bits),
        ask,
        pinned,
    );
    publishResolvedPrefixCacheMem(plan.budget);
    if (revise.quiet) return plan.budget;
    // Say so when the box is under external pressure right now, but do not shrink the budget for it.
    const live_ceiling: u64 = currentGpuMemoryCeiling(config, active_mem);
    if (live_ceiling < staticGpuMemoryCeiling()) {
        const free_gb = @as(f64, @floatFromInt(live_ceiling -| active_mem)) / (1024.0 * 1024.0 * 1024.0);
        log.info("[hot-cache] budget {d} MB (static ceiling); free at load {d:.1} GB — live admission will evict as needed\n", .{ plan.budget >> 20, free_gb });
    }
    if (ask != requested) {
        log.info("[hot-cache] budget {d} MB = one session at the working context (no --prefix-cache-mem)\n", .{ask >> 20});
    }
    if (ask > 0 and plan.budget < ask) {
        log.info("[hot-cache] budget clamped {d} -> {d} MB (chunk {d}, reserve at width {d} = {d} MB, ctx KV {d} MB)\n", .{ ask >> 20, plan.budget >> 20, plan.chunk, plan.reserve_chunk, plan.reserve >> 20, plan.ctx_kv >> 20 });
    } else if (requested == 0) {
        log.info("[hot-cache] budget capped at {d} MB (no --prefix-cache-mem; chunk {d}, reserve at width {d} = {d} MB, ctx KV {d} MB)\n", .{ plan.budget >> 20, plan.chunk, plan.reserve_chunk, plan.reserve >> 20, plan.ctx_kv >> 20 });
    }
    return plan.budget;
}

// ── tests ──

test "effectiveSsmCheckpointStride: zero capacity disables both tiers, RAM-off preserves disk" {
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(256, 0, true, 4 << 30));
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(256, 0, false, 4 << 30));
    try std.testing.expectEqual(@as(u32, 256), effectiveSsmCheckpointStride(256, 32, true, 0));
    try std.testing.expectEqual(@as(u32, 256), effectiveSsmCheckpointStride(256, 32, false, 4 << 30));
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(256, 32, false, 0));
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(0, 32, true, 0));
}

test "clampedPrefixCacheMem: the budget never exceeds what the weights leave under the ceiling" {
    const t = std.testing;
    const GB: u64 = 1 << 30;
    // The crash's numbers: 96 GB ceiling, ~70 GB weights, ~6.4 GB of 262k-ctx
    // KV, ~4 GB prefill reserve, --prefix-cache-mem 40 GB → ~15.6 GB budget.
    const crash = clampedPrefixCacheMem(40 * GB, 96 * GB, 70 * GB, 6 * GB + (400 << 20), 4 * GB);
    try t.expect(crash >= 10 * GB and crash < 16 * GB);
    // Small model with plenty of headroom: the request is untouched.
    try t.expectEqual(8 * GB, clampedPrefixCacheMem(8 * GB, 96 * GB, 20 * GB, 2 * GB, 2 * GB));
    // requested == 0 (byte cap disabled today): headroom still bounds it.
    try t.expectEqual(96 * GB - 80 * GB, clampedPrefixCacheMem(0, 96 * GB, 70 * GB, 6 * GB, 4 * GB));
    // No headroom at all: never 0 — initWithMem reads 0 as "no byte cap".
    try t.expectEqual(@as(u64, 1), clampedPrefixCacheMem(40 * GB, 64 * GB, 70 * GB, 6 * GB, 4 * GB));
}

test "prefix cache default capacity covers interleaved agent flows" {
    // Claude Code-style clients interleave several conversation roots (main
    // thread, subagents, title generation). With capacity 1, every
    // interleaved request evicted the long system-prompt prefix and forced a
    // full re-prefill per turn. The byte budget (prefix_cache_mem_bytes)
    // still bounds memory.
    try testing.expect(http_server.launch().prefix_cache_capacity >= 4);
    try testing.expect(resolvedPrefixCacheMem() > 0);
}

test "ungatedHotCacheAsk: a sliding-ring or hybrid arch asks for one whole session, every other arch keeps its ask" {
    const t = std.testing;
    const mimo: model_mod.ModelConfig = .{ .model_type = "mimo_v2", .num_hidden_layers = 4, .head_dim = 192, .v_head_dim = 128, .num_key_value_heads = 8, .num_global_key_value_heads = 4, .sliding_window = 128, .sliding_window_pattern = 4 };
    const session = oneSessionEntryBytes(&mimo, 16, 1_000_000, 8192);
    try t.expect(session > PREFIX_CACHE_MEM_DEFAULT);
    try t.expectEqual(session, ungatedHotCacheAsk(&mimo, PREFIX_CACHE_MEM_DEFAULT, false, 16, 1_000_000, 8192));
    // An operator's number stands.
    try t.expectEqual(@as(u64, 1 << 30), ungatedHotCacheAsk(&mimo, 1 << 30, true, 16, 100_000, 8192));
    const llama: model_mod.ModelConfig = .{ .model_type = "llama" };
    try t.expectEqual(PREFIX_CACHE_MEM_DEFAULT, ungatedHotCacheAsk(&llama, PREFIX_CACHE_MEM_DEFAULT, false, 16, 100_000, 8192));
    // A hybrid entry restores only at a checkpoint, so a budget under one session keeps a stale prefix.
    const qwen: model_mod.ModelConfig = .{ .model_type = "qwen3_5", .num_hidden_layers = 64, .head_dim = 256, .num_attention_heads = 24, .num_key_value_heads = 4, .full_attention_interval = 4 };
    const qwen_session = oneSessionEntryBytes(&qwen, 16, 262_144, 8192);
    try t.expect(qwen_session > PREFIX_CACHE_MEM_DEFAULT);
    try t.expectEqual(qwen_session, ungatedHotCacheAsk(&qwen, PREFIX_CACHE_MEM_DEFAULT, false, 16, 262_144, 8192));
}

test "ssdFirstPrefixCacheMem: the active session's KV is billed ONCE and is the floor" {
    const t = std.testing;
    const GB: u64 = 1 << 30;
    // The 1M qwen4 case: 107.5 GB ceiling, ~70 GB weights, ~24 GB of 1M-ctx KV, ~3 GB reserve.
    const ceiling: u64 = 107 * GB;
    const weights: u64 = 70 * GB;
    const ctx_kv: u64 = 24 * GB;
    const transient: u64 = 3 * GB;

    // RAM-first clamps the cache to the leftover: less than half an entry.
    const ram_first = clampedPrefixCacheMem(0, ceiling, weights, ctx_kv, transient);
    try t.expect(ram_first < ctx_kv);

    // SSD-first floors at one entry and hands `--prefix-cache-mem` to idle entries on top.
    try t.expectEqual(ctx_kv, ssdFirstPrefixCacheMem(0, ceiling, weights, ctx_kv, transient));
    try t.expectEqual(ctx_kv + 4 * GB, ssdFirstPrefixCacheMem(4 * GB, ceiling, weights, ctx_kv, transient));
    const huge = ssdFirstPrefixCacheMem(500 * GB, ceiling, weights, ctx_kv, transient);
    try t.expectEqual(ctx_kv + (ceiling - weights - ctx_kv - transient), huge);
    try t.expectEqual(ctx_kv, ssdFirstPrefixCacheMem(8 * GB, 80 * GB, weights, ctx_kv, transient));
    try t.expect(ssdFirstPrefixCacheMem(0, 0, 0, 0, 0) >= 1);
}

test "resolvedContextForLoad: an auto boot bills the session it will serve, not the placeholder" {
    // On an auto boot the effective-context accessor answered a 1024-token placeholder during the load.
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    // This box: ~28,909 MiB after the weights, 20,736 B/token at 1M/kv8, ~3 GiB transient.
    const per_tok: u64 = 20_736;
    const active: u64 = 70_000 * MiB;
    const ceiling: u64 = active + 28_909 * MiB;
    const transient: u64 = 3 * 1024 * MiB;
    const cap: u32 = 1_048_576;

    // Auto boot, nothing pinned: the session is what the machine can serve. `cache_reserve = 0`
    // is a test isolation.
    const auto = resolvedContextForLoad(0, 0, ceiling, active, 0, transient, per_tok, cap);
    try t.expect(auto > 100_000);
    try t.expect(auto <= cap);

    // The placeholder bills ~26 MB of session and hands the idle allowance almost the whole machine.
    const placeholder_kv: u64 = per_tok * 1024;
    const real_kv: u64 = per_tok * auto;
    try t.expect(placeholder_kv < 30 * MiB);
    try t.expect(real_kv > 100 * placeholder_kv);
    const ask: u64 = 24 * 1024 * MiB;
    const bogus = ssdFirstPrefixCacheMem(ask, ceiling, active, placeholder_kv, transient);
    const fixed = ssdFirstPrefixCacheMem(ask, ceiling, active, real_kv, transient);
    try t.expect(bogus -| placeholder_kv == ask);
    // Under the real session bill the same ask is clamped to what is left after the weights, one
    // session and the transient (the helper's own contract, not a hand-picked byte count).
    const idle_headroom: u64 = ceiling - (active + real_kv + transient);
    try t.expectEqual(real_kv + idle_headroom, fixed);
    try t.expect(fixed -| real_kv < ask);
    try t.expect(fixed -| real_kv < (bogus -| placeholder_kv) / 2);

    // An explicit --ctx-size wins outright, and a pinned context is used as-is.
    try t.expectEqual(@as(u32, 262_144), resolvedContextForLoad(262_144, 0, ceiling, active, 0, transient, per_tok, cap));
    try t.expectEqual(@as(u32, 131_072), resolvedContextForLoad(0, 131_072, ceiling, active, 0, transient, per_tok, cap));
    try t.expectEqual(cap, resolvedContextForLoad(0, 0, active + 900_000 * MiB, active, 0, transient, per_tok, cap));
}

test "the published hot-cache budget is retired when the cache is dropped" {
    // The budget is per model, the global is process-wide.
    const t = std.testing;
    publishResolvedPrefixCacheMem(4096);
    try t.expectEqual(@as(u64, 4096), resolvedPrefixCacheMem());
    clearResolvedPrefixCacheMem();
    // Back to the ask, the honest pre-load answer.
    try t.expectEqual(http_server.launch().prefix_cache_mem_bytes, resolvedPrefixCacheMem());

    // A resolved zero is a real budget (`--prefix-cache-mem 0` under SSD-first); the sentinel is maxInt.
    publishResolvedPrefixCacheMem(0);
    try t.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
    clearResolvedPrefixCacheMem();
    try t.expectEqual(http_server.launch().prefix_cache_mem_bytes, resolvedPrefixCacheMem());
}

test "resolvedPrefixCacheMem: RAM-off bills zero despite a configured disk tier" {
    const capacity = http_server.launch().prefix_cache_capacity;
    const ram = http_server.launch().prefix_cache_ram_enabled;
    defer http_server.launch().prefix_cache_capacity = capacity;
    defer http_server.launch().prefix_cache_ram_enabled = ram;
    http_server.launch().prefix_cache_capacity = 8;
    http_server.launch().prefix_cache_ram_enabled = false;
    try testing.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
    http_server.launch().prefix_cache_ram_enabled = true;
    http_server.launch().prefix_cache_capacity = 0;
    try testing.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
}

test "defaultPrefixCacheAsk: an unnamed budget holds one session at the working context, never under 2 GB" {
    const t = std.testing;
    const gb: u64 = 1024 * 1024 * 1024;
    const session: u64 = 262144 * 24576; // 262k-token qwen4_exp session, ~6 GB
    try t.expectEqual(session, defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, session));
    // A short working context keeps the old floor.
    try t.expectEqual(PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, gb));
    // An operator's number stands, even one equal to the default.
    try t.expectEqual(PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, true, session));
    try t.expectEqual(32 * gb, defaultPrefixCacheAsk(32 * gb, true, session));
    try t.expectEqual(@as(u64, 0), defaultPrefixCacheAsk(0, true, session));
    // An embedder's own ask (ios_lib passes 256 MB) is not the default and stands.
    try t.expectEqual(256 * 1024 * 1024, defaultPrefixCacheAsk(256 * 1024 * 1024, false, session));
}

test "defaultPrefixCacheAsk: the machine's headroom still caps the defaulted ask" {
    const t = std.testing;
    const gb: u64 = 1024 * 1024 * 1024;
    const session: u64 = 262144 * 24576;
    const ask = defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, session);
    // 16 GB ceiling, 11 GB weights, 1 GB live context, 1.5 GB prefill reserve: 2.5 GB left.
    try t.expectEqual(gb * 5 / 2, clampedPrefixCacheMem(ask, 16 * gb, 11 * gb, gb, gb * 3 / 2));
    // A big machine gets the whole session.
    try t.expectEqual(session, clampedPrefixCacheMem(ask, 200 * gb, 80 * gb, gb, 2 * gb));
}

test "oneSessionEntryBytes: a cached session is billed with its SSM checkpoints" {
    const t = std.testing;
    var cfg = qwen4DeployedTestConfig();
    const ctx: u64 = 262_144;
    const chunk: u64 = 8192;
    const kv_only = sessionBytesPerToken(&cfg, 16) *| ctx +| cfg.qsaRingBytes();
    const entry = oneSessionEntryBytes(&cfg, 16, ctx, chunk);
    try t.expect(cfg.ssmCheckpointBytes() > 0);
    try t.expectEqual(kv_only + retainedSsmCheckpointBytes(&cfg, ctx, 0, chunk), entry);
    // The defaulted ask covers the whole entry, so the commit path never trims it.
    try t.expect(defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, entry) >= entry);
}

test "disabled prefix cache: sizing releases the cache reserve on every arch" {
    const saved_capacity = http_server.launch().prefix_cache_capacity;
    const saved_ask = http_server.launch().prefix_cache_mem_bytes;
    defer http_server.launch().prefix_cache_capacity = saved_capacity;
    defer http_server.launch().prefix_cache_mem_bytes = saved_ask;

    var gated = longCtxTestConfig();
    var other = longCtxTestConfig();
    other.model_type = "qwen3_5_moe";
    other.legacy.caps.long_ctx = std.mem.eql(u8, other.model_type, "qwen4_exp");
    const configs = [_]*model_mod.ModelConfig{ &gated, &other };
    for ([_]u64{ 0, 2 << 30, 60 << 30 }) |ask| {
        http_server.launch().prefix_cache_mem_bytes = ask;
        http_server.launch().prefix_cache_capacity = 0;
        for (configs) |cfg| {
            try testing.expectEqual(@as(u64, 0), ctxSizingCacheReserve(cfg));
        }
        try testing.expectEqual(@as(u64, 0), legacyPrefixCacheAsk());
        try testing.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
    }
    http_server.launch().prefix_cache_capacity = 32;
    http_server.launch().prefix_cache_mem_bytes = 10 << 30;
    try testing.expectEqual(CTX_SIZING_CACHE_RESERVE, ctxSizingCacheReserve(&gated));
    try testing.expectEqual(http_server.launch().prefix_cache_mem_bytes, ctxSizingCacheReserve(&other));
    try testing.expectEqual(http_server.launch().prefix_cache_mem_bytes, legacyPrefixCacheAsk());
}
