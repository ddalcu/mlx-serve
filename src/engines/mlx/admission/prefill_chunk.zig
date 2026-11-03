//! chunk ladder, prefillChunkCap, resolve/pin chunk; widthForRung + chunk tests; prefillNeededAtChunk, per-request
//! and adaptive chunk

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const sparse_score = @import("../../../sdk/blocks/attention/sparse_block/score.zig");
const generate_mod = @import("../generate.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const dsv4_mod = @import("../../../plugins/deepseek_v4/deepseek_v4.zig");
const log = @import("../../../sdk/util/log.zig");
const prefix_cache_mod = @import("../cache/prefix_cache.zig");
const scheduler_mod = @import("../scheduler.zig");
const http_server = @import("../../../host/http/server.zig");
const memory_ceiling = @import("../../../host/registry/memory_ceiling.zig");

const manualContext = @import("../../../host/http/server.zig").manualContext;
const resolvedPrefixCacheMem = @import("hot_cache_plan.zig").resolvedPrefixCacheMem;
const clearResolvedPrefixCacheMem = @import("hot_cache_plan.zig").clearResolvedPrefixCacheMem;
const autoContextFrom = @import("context.zig").autoContextFrom;
const getEffectiveContextLength = @import("context.zig").getEffectiveContextLength;
const physicalMemoryCeiling = @import("../../../host/registry/memory_ceiling.zig").physicalMemoryCeiling;
const currentGpuMemoryCeiling = @import("../../../host/registry/memory_ceiling.zig").currentGpuMemoryCeiling;
const safeContextForBudget = @import("context.zig").safeContextForBudget;
const defaultKvBits = @import("bills.zig").defaultKvBits;
const prefillTransientReserve = @import("bills.zig").prefillTransientReserve;
const prefillTransientReserveAtKv = @import("bills.zig").prefillTransientReserveAtKv;
const qsaMaskBytes = @import("../../../plugins/qwen4_exp/bill.zig").qsaMaskBytes;
const clampReserveWidth = @import("hot_cache_plan.zig").clampReserveWidth;
const planHotCache = @import("hot_cache_plan.zig").planHotCache;
const ssdFirstPrefixCacheMem = @import("hot_cache_plan.zig").ssdFirstPrefixCacheMem;
const resolvedContextForLoad = @import("hot_cache_plan.zig").resolvedContextForLoad;
const ssdFirstSessionTokensNow = @import("hot_cache_plan.zig").ssdFirstSessionTokensNow;
const ssdFirstSessionKvBytes = @import("hot_cache_plan.zig").ssdFirstSessionKvBytes;
const CTX_SIZING_CACHE_RESERVE = @import("hot_cache_plan.zig").CTX_SIZING_CACHE_RESERVE;
const legacyPrefixCacheAsk = @import("hot_cache_plan.zig").legacyPrefixCacheAsk;
const prefixCacheMemForLoad = @import("hot_cache_plan.zig").prefixCacheMemForLoad;
const longCtxTestConfig = @import("bills_test.zig").longCtxTestConfig;
const qsaScoreFusedOffGuard = @import("bills_test.zig").qsaScoreFusedOffGuard;
const qwen4RequestTestConfig = @import("bills_test.zig").qwen4RequestTestConfig;
const qwen4DeployedTestConfig = @import("bills_test.zig").qwen4DeployedTestConfig;
const slotFixedKvBytes = @import("bills.zig").slotFixedKvBytes;
const kvBytesPerTokenAtBits = @import("bills.zig").kvBytesPerTokenAtBits;
const WarmPrefix = @import("bills.zig").WarmPrefix;
const prefillMemoryNeeded = @import("bills.zig").prefillMemoryNeeded;
const dsv4PrefillMemoryNeeded = @import("../../../plugins/deepseek_v4/bill.zig").dsv4PrefillMemoryNeeded;
const dsv41_mod = @import("../../../plugins/deepseek_v41/deepseek_v41.zig");
const PREFILL_RUNTIME_FLOOR_BYTES = @import("bills.zig").PREFILL_RUNTIME_FLOOR_BYTES;

/// DeepSeek-V4.1's prompt pass (`deepseek_v41.prefillBytes`) over the engine's config.
pub fn dsv41PrefillMemoryNeeded(config: *const model_mod.ModelConfig, seq: u64, chunk: u64) u64 {
    return dsv41_mod.prefillBytes(config, config.kvBytesPerToken(), seq, chunk, PREFILL_RUNTIME_FLOOR_BYTES);
}
const prefillStreamBytesPerToken = @import("bills.zig").prefillStreamBytesPerToken;
const prefillDequantWeightBytes = @import("bills.zig").prefillDequantWeightBytes;
const prefillFfnWidth = @import("bills.zig").prefillFfnWidth;
const statePerTokenBilled = @import("bills.zig").statePerTokenBilled;
const reservedCacheTokens = @import("bills.zig").reservedCacheTokens;
const sessionBytesPerToken = @import("bills.zig").sessionBytesPerToken;
const glm5PrefillMemoryNeeded = @import("glm5_bill.zig").glm5PrefillMemoryNeeded;
const prefillRequestTerms = @import("bills.zig").prefillRequestTerms;
const AdmissionVerdict = @import("admit.zig").AdmissionVerdict;
const admissionVerdict = @import("admit.zig").admissionVerdict;
const admissionLogLevel = @import("admit.zig").admissionLogLevel;
const qwen4ExpOomConfig = @import("bills_test.zig").qwen4ExpOomConfig;
const qwen4ExpLive364kConfig = @import("bills_test.zig").qwen4ExpLive364kConfig;

/// Widths `resolvePrefillChunk` will step down through. Descending, floored at
/// `generate.PREFILL_CHUNK_FLOOR` — below that the score-budget path refuses to
/// go either, and a 256-token forward stops amortizing the per-chunk sweeps.
/// The top rung is `WIDE_PREFILL_CHUNK`, pinned only by an arch with no prefill score tensor.
pub const PREFILL_CHUNK_LADDER = [_]u32{ @intCast(generate_mod.WIDE_PREFILL_CHUNK), 8192, 4096, 2048, 1024, 512 };

/// How much of the post-weights serving budget a ONE-OFF prefill transient may
/// claim before the chunk steps down. A quarter: past that the machine is
/// trading the whole session's context for one forward's speed, which is the
/// trade that reported a 1024-token context on a 16 GB Mac.
pub const PREFILL_RESERVE_BUDGET_SHARE: u64 = 4;

/// The hot prefix cache's smallest useful budget: a prefill chunk that would price the cache
/// under this steps down instead. Stops `requested == 0` from resolving to the 1-byte floor.
pub const HOT_CACHE_FLOOR_BYTES: u64 = 1024 * 1024 * 1024;

/// PURE: the widest prefill chunk THIS machine can afford for THIS model.
///
/// The chunk multiplies the biggest term in the memory bill, and until now
/// nothing tied it to the machine: `--prefill-chunk` defaults to 8192, the app
/// never passes the flag, so a 16 GB Mac reserved the same 5-7 GB MLP envelope
/// a 128 GB one does. Measured on a 16 GB profile (Mistral-7B-4bit, weights
/// 3.80 GB against an 11.9 GB Metal ceiling): the reserve at chunk 8192 alone
/// exceeded what was left after the weights, so the sizer reported 1024 tokens
/// and the admission guard refused a 10k-token prompt for "needing" 9.2 GB —
/// a prompt whose measured peak is 2.39 GB. Stepping to chunk 512 costs prefill
/// throughput and buys 22k tokens of context back.
///
/// Never raises anything: `effectivePrefillChunk` takes the MIN of this and the
/// launch chunk, and an explicit `--prefill-chunk` outranks it entirely.
///
/// It does not read the hot-cache ask: the clamp downstream subtracts this chunk's reserve
/// from the cache's headroom, so feeding the ask in here was non-monotone (a bigger
/// `--prefix-cache-mem` stepped the chunk down and handed the cache MORE bytes). The chunk is
/// sized against the serving budget alone; the cache is the residual claimant. `ctx_kv_bytes`
/// is the KV a pinned context has already spoken for (0 under auto-context).
/// The byte cap one prefill forward's transient may claim, and the one place the sizer's two
/// changes (dropping the ask, adding the ctx bar) are gated by arch; every other arch keeps
/// the previous arithmetic.
pub fn prefillChunkCap(
    config: *const model_mod.ModelConfig,
    ceiling: u64,
    active_mem: u64,
    ctx_kv_bytes: u64,
    hot_cache_ask: u64,
) u64 {
    if (!config.longCtxGated()) {
        // Ungated: the ask is spoken for, and a quarter of what is left is the whole cap.
        const spoken_for: u64 = active_mem +| hot_cache_ask;
        return (ceiling -| spoken_for) / PREFILL_RESERVE_BUDGET_SHARE;
    }
    const serving_budget: u64 = ceiling -| active_mem;
    // Two ask-independent bars: a share of the serving budget, and whatever the pinned context leaves less a cache worth having.
    const share_bar: u64 = serving_budget / PREFILL_RESERVE_BUDGET_SHARE;
    // The ctx bar can only ever narrow; `--ctx-size` is a cap, not an allocation, so it can pin
    // chunk 512 for the life of a process. The narrowing is logged.
    const ctx_bar: u64 = (serving_budget -| ctx_kv_bytes) -| HOT_CACHE_FLOOR_BYTES;
    return @min(share_bar, ctx_bar);
}

pub fn resolvePrefillChunk(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    ceiling: u64,
    active_mem: u64,
    ctx_kv_bytes: u64,
    hot_cache_ask: u64,
) u32 {
    const cap: u64 = prefillChunkCap(config, ceiling, active_mem, ctx_kv_bytes, hot_cache_ask);
    for (PREFILL_CHUNK_LADDER) |chunk| {
        if (chunk == generate_mod.WIDE_PREFILL_CHUNK and config.prefillScoreHeadDim() != 0) continue;
        if (config.prefillChunkMax() != 0 and chunk > config.prefillChunkMax()) continue;
        if (prefillTransientReserve(config, kv_bits, chunk) <= cap) return chunk;
    }
    // Nothing fits the share — the model barely fits at all. Take the narrowest
    // rung: it is the smallest bill this box can be asked for, and returning 0
    // would read as "not pinned" and hand the forward the launch width.
    return PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
}

/// The width `--prefill-chunk` asked for, or 0. The bill must let it outrank the pin as the
/// forward does: `--prefill-chunk 4096` used to run against a 512-token reserve.
pub fn explicitPrefillChunk() u32 {
    if (!generate_mod.prefillCli().explicit) return 0;
    const raw: usize = @min(generate_mod.prefillCli().override, @as(usize, std.math.maxInt(u32)));
    return @intCast(raw);
}

/// The chunk this model is billed at, and therefore runs at. `chunk_override` (0 = none)
/// mirrors `generate.effectivePrefillChunk`'s precedence.
pub fn billedPrefillChunk(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    ceiling: u64,
    active_mem: u64,
    ctx_kv_bytes: u64,
    hot_cache_ask: u64,
    chunk_override: u32,
) u32 {
    if (chunk_override > 0) return chunk_override;
    return @min(resolvePrefillChunk(config, kv_bits, ceiling, active_mem, ctx_kv_bytes, hot_cache_ask), autoPrefillChunkCap(mlx.cudaAvailable()));
}

/// On CUDA a chunk past 2048 rows measured no faster (Llama 3.2 3B, RTX 5060 Ti: 4400 vs 4396
/// tok/s) while its prefill peak grew ~2.2 MB a row against 14k keys, which the reserve does not
/// bill; long prompts then ran the card out of memory. An explicit `--prefill-chunk` still wins.
pub fn autoPrefillChunkCap(cuda: bool) u32 {
    return if (cuda) 2048 else std.math.maxInt(u32);
}

/// The context KV the chunk sizer must leave standing: the pinned context's bill under an
/// explicit `--ctx-size`, 0 while the context is auto (the auto sizer adapts to the rung).
pub fn sizerCtxKvBytes(config: *const model_mod.ModelConfig, kv_bits: u64) u64 {
    if (manualContext(config) == 0) return 0;
    // Runs inside `pinPrefillChunk`: before the pin, bill the widest rung.
    const chunk: u64 = if (config.pinned_prefill_chunk > 0) config.pinned_prefill_chunk else generate_mod.WIDE_PREFILL_CHUNK;
    return sessionBytesPerToken(config, kv_bits) *| manualContext(config) +| slotFixedKvBytes(config, kv_bits, chunk);
}

/// Freeze this model's prefill chunk at load, from live memory. Idempotent.
/// Must run BEFORE the auto-context sizer: the sizer bills this chunk's
/// transient reserve, and `checkAttentionMemory` and
/// `generate.effectivePrefillChunk` then read the same frozen value, so the
/// bill and the forward cannot drift.
pub fn pinPrefillChunk(config: *model_mod.ModelConfig) u32 {
    if (config.pinned_prefill_chunk == 0) {
        var active_mem: usize = 0;
        _ = mlx.mlx_get_active_memory(&active_mem);
        const kv_bits: u64 = defaultKvBits(config);
        // Which ask: the gated arch reads the resolved budget; every other arch reads the raw
        // ask as before (the accessor would pin a second model against the first's budget).
        const hot_cache_ask = if (config.longCtxGated()) resolvedPrefixCacheMem() else legacyPrefixCacheAsk();
        // The override's precedence is the second gated input: ungated, `--prefill-chunk`
        // still outranks the pin only at forward time, as before (a known inconsistency).
        const pin_override: u32 = if (config.longCtxGated()) explicitPrefillChunk() else 0;
        config.pinned_prefill_chunk = billedPrefillChunk(
            config,
            kv_bits,
            currentGpuMemoryCeiling(config, active_mem),
            active_mem,
            sizerCtxKvBytes(config, kv_bits),
            hot_cache_ask,
            pin_override,
        );
        // Say when the ctx bar is the binding one.
        if (config.longCtxGated() and explicitPrefillChunk() == 0) {
            const share_only = resolvePrefillChunk(config, kv_bits, currentGpuMemoryCeiling(config, active_mem), active_mem, 0, hot_cache_ask);
            if (share_only > config.pinned_prefill_chunk) {
                log.info("[prefill] chunk {d} (ctx bar: the --ctx-size KV bill; share-only would allow {d})\n", .{ config.pinned_prefill_chunk, share_only });
            }
        }
        // Say it once per model, wherever the model was pinned from (startup
        // primary or on-demand load) — a narrowed prefill otherwise reads as an
        // unexplained slowdown. mlx-stream takes GLM-5.3's whole prompt in one
        // forward and chunks it itself.
        if (config.pinned_prefill_chunk < generate_mod.prefillCli().override and
            !generate_mod.prefillCli().explicit and !config.glm_stream)
        {
            log.info("Prefill chunk: {d} tokens (memory-sized down from {d}; --prefill-chunk overrides)\n", .{ config.pinned_prefill_chunk, generate_mod.prefillCli().override });
        }
    }
    return config.pinned_prefill_chunk;
}

/// The width a ladder rung actually forwards at for this prompt.
fn widthForRung(cfg: *const model_mod.ModelConfig, seq: u64, rung: u32) u32 {
    return @intCast(rungWidth(cfg, seq, rung, cfg.longCtxGated()));
}

/// Everything the admission guard bills for one request at one prefill width, so the
/// per-request chunk chooser prices a rung with the same function that admits it. `chunk` is
/// the width the forward will run, already through `generate.effectivePrefillChunk`.
pub fn prefillNeededAtChunk(
    config: *const model_mod.ModelConfig,
    seq: u64,
    max_tokens: u32,
    kv_bits: u64,
    chunk: u64,
    warm: WarmPrefix,
) u64 {
    // deepseek_v4 gets its own estimator: it sub-chunks prefill internally and its state is module-owned f32.
    const is_dsv4: bool = std.mem.eql(u8, config.model_type, "deepseek_v4") and config.dsv4_n_compress_ratios > 0;
    const heads: u64 = config.num_attention_heads;
    const layers: u64 = config.num_hidden_layers;
    const kv_heads: u64 = config.num_key_value_heads;
    const hdim: u64 = config.head_dim;
    const hidden: u64 = config.hidden_size;
    const ffn: u64 = prefillFfnWidth(config);
    if (is_dsv4) return dsv4PrefillMemoryNeeded(seq, layers, kv_heads * hdim, hidden, ffn, dsv4_mod.prefillSub(), config.prefillAttnKeys(seq));
    if (config.mlxStream()) return 0; // billed at construction (`mlx_stream.loadBytes`)
    // A streamed bank's slot rows are filled at the load beside the prompt working set it reserves (DeepSeek-V4.1's
    // `bill.streamBudget`); a longer prompt is the plugin's refusal by name.
    if (config.plugin) |b| if (b.parsed.caps.streamed_weights) return 0;
    if (config.isDsv41()) return dsv41PrefillMemoryNeeded(config, seq, chunk);
    const drafter: u64 = (seq +| @min(@as(u64, max_tokens), transformer_mod.KVCache.RESERVE_GEN_HEADROOM)) *| config.drafter_ctx_bytes_per_token;
    if (config.isGlm5()) return glm5PrefillMemoryNeeded(config, seq, max_tokens, kv_bits, chunk, warm) +| drafter;
    return prefillMemoryNeeded(seq, heads, kv_heads, config.kvBytesPerToken(), hdim, config.prefillScoreHeadDim(), hidden, ffn, kv_bits, chunk, config.prefillAttnKeys(seq), prefillStreamBytesPerToken(config), prefillDequantWeightBytes(config), prefillRequestTerms(config, seq, max_tokens, kv_bits, chunk, warm)) +
        qsaMaskBytes(config, @min(chunk, @max(seq, 1)), seq) +
        drafter;
}

/// Headroom a rung must clear beyond its own bill when the long-context gate is what widened
/// it: past that width the machine cap frozen at load no longer bounds the forward, and
/// `prefillMemoryNeeded` is documented to UNDER-bill quantized prefill by about this much.
/// Narrower rungs keep the plain bill, so no other admission decision moves.
pub const PREFILL_WIDE_RUNG_MARGIN_PCT: u64 = 22;

/// Test hook for `perRequestPrefillChunkEnabled`'s kill switch.
pub var per_request_chunk_override: ?bool = null;

/// The arch capability; the override is a test seam.
pub fn perRequestPrefillChunkEnabled(config: *const model_mod.ModelConfig) bool {
    if (!config.perRequestPrefillChunk()) return false;
    return per_request_chunk_override orelse true;
}

/// The widest ladder rung this request can afford, priced by the same estimator that admits
/// it against post-eviction memory. Candidates are priced at the width the forward will run
/// (`effectivePrefillChunk` caps by arch), and the width is returned, not the rung. An explicit
/// `--prefill-chunk` wins outright; nothing fitting means the ladder floor.
pub fn chooseRequestPrefillChunk(
    config: *const model_mod.ModelConfig,
    seq: u64,
    max_tokens: u32,
    kv_bits: u64,
    available: u64,
    load_time_pin: u32,
    chunk_override: u32,
    warm: WarmPrefix,
) u32 {
    // The arch gate first: an ungated arch's answer is the load-time pin, as before.
    if (!perRequestPrefillChunkEnabled(config)) return load_time_pin;
    if (chunk_override > 0) return chunk_override;
    for (PREFILL_CHUNK_LADDER) |rung| {
        const width: u64 = rungWidth(config, seq, rung, config.longCtxGated());
        const bill = prefillNeededAtChunk(config, seq, max_tokens, kv_bits, width, warm);
        const needed = if (width > rungWidth(config, seq, rung, false))
            bill *| (100 + PREFILL_WIDE_RUNG_MARGIN_PCT) / 100
        else
            bill;
        if (needed <= available) return @intCast(width);
    }
    return @intCast(rungWidth(config, seq, PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1], config.longCtxGated()));
}

/// The width one ladder rung forwards at. `long_ctx_gated = false` asks what the rung would be
/// without the long-context gate, i.e. the width the load-time machine cap still bounds.
fn rungWidth(config: *const model_mod.ModelConfig, seq: u64, rung: u32, long_ctx_gated: bool) u64 {
    // `effectivePrefillChunk` can return an explicit `--prefill-chunk` verbatim; clamp before narrowing.
    return @min(
        @as(u64, generate_mod.effectivePrefillChunk(
            config.prefillScoreHeadDim(),
            config.num_attention_heads,
            @intCast(seq),
            config.has_sliding_window,
            config.isMoe(),
            long_ctx_gated,
            rung,
        )),
        @as(u64, std.math.maxInt(u32)),
    );
}

/// Impure wrapper the scheduler reaches through `prefill_request_chunk`, mirroring
/// `prefillFitsNow`'s inputs. Reads live memory, so it must run after the eviction pass.
pub fn requestPrefillChunkNow(
    config: *const model_mod.ModelConfig,
    prompt_len: usize,
    max_tokens: u32,
    kv_cfg: transformer_mod.KVQuantConfig,
    warm_matched: u64,
    warm_capacity: u64,
    warm_will_donate: bool,
    enable_mtp: bool,
) u32 {
    const pin: u32 = config.pinned_prefill_chunk;
    const explicit: u32 = explicitPrefillChunk();
    if (explicit > 0) return explicit;
    if (!perRequestPrefillChunkEnabled(config)) return pin;

    const seq: u64 = @intCast(prompt_len);
    const kv_bits: u64 = if (kv_cfg.scheme == .off) 16 else kv_cfg.bits;
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    const available: u64 = currentGpuMemoryCeiling(config, active_mem) -| active_mem;
    const warm = WarmPrefix{ .matched_tokens = warm_matched, .capacity_tokens = warm_capacity, .will_donate = warm_will_donate, .mtp_on = enable_mtp };
    const chosen = chooseRequestPrefillChunk(config, seq, max_tokens, kv_bits, available, pin, 0, warm);
    if (chosen != pin) {
        const width: u64 = chosen;
        const terms = prefillRequestTerms(config, seq, max_tokens, kv_bits, width, warm);
        const kv_bytes: u64 = seq *| kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| terms.reserved_kv_bytes;
        log.debug("[prefill] chunk {d} for this request (reserve {d} MB beside KV {d} MB)\n", .{ width, prefillTransientReserve(config, kv_bits, width) >> 20, kv_bytes >> 20 });
    }
    return chosen;
}

/// Test hook for `adaptivePrefillChunkEnabled`'s kill switch.
pub var adaptive_chunk_override: ?bool = null;

/// The per-chunk adaptive width: subordinate to the per-request gate, plus "no operator pinned a width".
pub fn adaptivePrefillChunkEnabled(config: *const model_mod.ModelConfig) bool {
    // Arch first: the cheapest and most selective, and this runs once per chunk boundary.
    if (!config.perRequestPrefillChunk()) return false;
    if (explicitPrefillChunk() > 0) return false;
    if (!perRequestPrefillChunkEnabled(config)) return false;
    return adaptive_chunk_override orelse true;
}

/// Consecutive supporting probes a widening needs: the probe is taken after `mlx_clear_cache`
/// and never contains a chunk's own peak, so the second reading pays for that blind spot.
pub const PREFILL_WIDEN_PROBES: u8 = 2;

/// What one chunk of `width` costs beyond what is already resident: its transient envelope
/// plus the per-token state the chunk is about to write. The growth term is billed even under
/// the reservation (short prompts do grow per chunk).
pub fn prefillChunkCost(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    width: u64,
    kv_len: u64,
) u64 {
    const per_tok: u64 = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +|
        statePerTokenBilled(config);
    return prefillTransientReserveAtKv(config, kv_bits, width, kv_len) +| (width *| per_tok);
}

/// The width the next chunk runs at. Asymmetric because a Metal abort is uncatchable: HOLD
/// while `headroom >= cost(current)` (the estimator carries its own 5/4); STEP DOWN at once,
/// by as many rungs as it takes, floored at `PREFILL_CHUNK_FLOOR`; WIDEN one rung only at
/// 1.25x its cost, after `PREFILL_WIDEN_PROBES` supporting probes, up to `cap`, never after a
/// step-down. `cap` is the widest width this arch forwards for this prompt, not a raw rung.
pub fn adaptivePrefillWidth(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    kv_len: u64,
    headroom: u64,
    current: u32,
    cap: u32,
    st: *generate_mod.AdaptiveWidthState,
) u32 {
    if (cap == 0 or current == 0) return current;
    const floor: u32 = @min(cap, @as(u32, @intCast(generate_mod.PREFILL_CHUNK_FLOOR)));

    if (prefillChunkCost(config, kv_bits, current, kv_len) > headroom) {
        st.supporting = 0;
        st.ratcheted = true;
        var best: u32 = floor;
        for (PREFILL_CHUNK_LADDER) |rung| {
            const w: u32 = @min(rung, cap);
            if (w >= current) continue;
            if (prefillChunkCost(config, kv_bits, w, kv_len) <= headroom) {
                best = w;
                break;
            }
        }
        return @min(best, current);
    }

    var next: u32 = 0;
    for (PREFILL_CHUNK_LADDER) |rung| {
        const w: u32 = @min(rung, cap);
        if (w > current and (next == 0 or w < next)) next = w;
    }
    if (next == 0 or st.ratcheted) {
        st.supporting = 0;
        return current;
    }
    const asked: u64 = prefillChunkCost(config, kv_bits, next, kv_len) *| 5 / 4;
    if (asked > headroom) {
        st.supporting = 0;
        return current;
    }
    st.supporting +|= 1;
    if (st.supporting < PREFILL_WIDEN_PROBES) return current;
    st.supporting = 0;
    return next;
}

/// The live headroom one prefill chunk has to fit in: the GPU ceiling less what MLX's
/// allocator holds, read once per chunk boundary after its `mlx_clear_cache`. The same
/// expression `prefillAdmissionBill` compares against.
pub fn prefillHeadroomNow(config: *const model_mod.ModelConfig, staged_host: u64) u64 {
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    // The SSD write-through stages chunk blobs as host bytes behind a ~1 GiB permit that
    // `mlx_get_active_memory` does not count; on unified memory they are the same pool. Read at
    // the hook site (inference thread) and passed in.
    return prefillHeadroomFrom(currentGpuMemoryCeiling(config, active_mem), active_mem, staged_host);
}

/// Pure half of `prefillHeadroomNow`.
pub fn prefillHeadroomFrom(ceiling: u64, active: u64, staged_host: u64) u64 {
    return ceiling -| active -| staged_host;
}

/// Does a widen the pre-tick probe chose still fit after the interleave tick allocated for
/// co-tenant slots? Same 1.25 bar, same estimator, same live KV.
pub fn adaptivePrefillWidenStillFits(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    pos: usize,
    want: u32,
    staged_host: u64,
) bool {
    if (!adaptivePrefillChunkEnabled(config)) return false;
    const headroom = prefillHeadroomNow(config, staged_host);
    const asked: u64 = prefillChunkCost(config, kv_bits, want, pos) *| 5 / 4;
    if (asked <= headroom) return true;
    log.info("[prefill] widen to {d} withdrawn at pos {d} (headroom {d} MB after the decode tick, cost {d} MB)\n", .{
        want,
        pos,
        headroom >> 20,
        asked >> 20,
    });
    return false;
}

/// Impure wrapper the prefill loop reaches through `generate.prefill_chunk_adapt`. Logs one
/// line per change, never per boundary (the bench harness greps it).
pub fn adaptivePrefillWidthNow(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    pos: usize,
    current: u32,
    cap: u32,
    st: *generate_mod.AdaptiveWidthState,
    staged_host: u64,
) u32 {
    if (!adaptivePrefillChunkEnabled(config)) return current;
    const headroom = prefillHeadroomNow(config, staged_host);
    // `pos` is the live KV this chunk will attend to (see `prefillTransientReserveAtKv`).
    const next = adaptivePrefillWidth(config, kv_bits, pos, headroom, current, cap, st);
    if (next != current) {
        // Quote the number it compared: `prefillChunkCost` at this KV.
        log.info("[prefill] width {d} -> {d} at pos {d} (headroom {d} MB, cost {d} MB)\n", .{
            current,
            next,
            pos,
            headroom >> 20,
            prefillChunkCost(config, kv_bits, next, pos) >> 20,
        });
    }
    return next;
}

// ── tests ──

test "the auto prefill chunk stops at 2048 rows on CUDA only" {
    try std.testing.expectEqual(@as(u32, 2048), autoPrefillChunkCap(true));
    try std.testing.expectEqual(@as(u32, std.math.maxInt(u32)), autoPrefillChunkCap(false));
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 28;
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 8;
    cfg.head_dim = 128;
    cfg.hidden_size = 3072;
    cfg.intermediate_size = 8192;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    const roomy: u64 = 1 << 40;
    const chunk = billedPrefillChunk(&cfg, 16, roomy, 2 << 30, 0, 0, 0);
    try std.testing.expect(chunk <= autoPrefillChunkCap(mlx.cudaAvailable()));
    try std.testing.expectEqual(@as(u32, 4096), billedPrefillChunk(&cfg, 16, roomy, 2 << 30, 0, 0, 4096));
}

test "the load-time budget is reproducible: free RAM at load does not move it" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Two boots 11 minutes apart resolved the same 10 GB ask to 1076 and 9757 MB off the live ceiling.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const GB: u64 = 1000 * 1000 * 1000;
    const weights: u64 = 70 * GB;
    const ctx_tokens: u64 = 1_048_576; // the deployed --ctx-size
    const static_ceiling: u64 = 120_000 * 1024 * 1024; // the wired limit, both boots
    // `requested == 0` is the arm where the clamp alone decides: the budget IS the headroom.
    const uncapped: u64 = 0;

    const busy = physicalMemoryCeiling(static_ceiling, weights, 108_700 * 1000 * 1000 -| weights);
    const quiet = physicalMemoryCeiling(static_ceiling, weights, 116_600 * 1000 * 1000 -| weights);
    try t.expect(busy < quiet);

    const on_busy = planHotCache(&cfg, kv_bits, busy, weights, ctx_tokens, 0, uncapped, 0);
    const on_quiet = planHotCache(&cfg, kv_bits, quiet, weights, ctx_tokens, 0, uncapped, 0);
    try t.expect(on_busy.budget != on_quiet.budget);
    try t.expect(on_quiet.budget > on_busy.budget);

    // Billed against the static term, both boots agree.
    const a = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, uncapped, 0);
    const b = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, uncapped, 0);
    try t.expectEqual(a.budget, b.budget);
    try t.expect(a.budget >= on_quiet.budget);

    const capped_a = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, 24 * GB, 0);
    const capped_b = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, 24 * GB, 0);
    try t.expectEqual(capped_a.budget, capped_b.budget);
    try t.expect(capped_a.budget <= a.budget);

    try t.expectEqual(static_ceiling, physicalMemoryCeiling(static_ceiling, weights, 400 * GB));
}

test "an auto boot sizes the SAME context whatever the cache ask (live check #6)" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `--prefix-cache-mem 60GB` collapsed the advertised context to 870 tokens (ctx KV billed at 26 MB).
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB; // backed out of the live clamp line
    const live_ceiling: u64 = 109_395 * MiB;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);
    const transient: u64 = prefillTransientReserve(&cfg, kv_bits, 4096);
    const cap: u32 = cfg.contextCap();

    // The sizing reserve is a constant, so the answer cannot move with the ask.
    const ctx_default = resolvedContextForLoad(0, 0, live_ceiling, active, CTX_SIZING_CACHE_RESERVE, transient, per_tok, cap);
    const ctx_big_ask = resolvedContextForLoad(0, 0, live_ceiling, active, CTX_SIZING_CACHE_RESERVE, transient, per_tok, cap);
    try t.expectEqual(ctx_default, ctx_big_ask);
    try t.expect(ctx_default > 900_000); // a real context, not the floor

    // The defect's spellings (the granted budget, the raw ask) take at most half the room, never all of it.
    const vs_granted = resolvedContextForLoad(0, 0, live_ceiling, active, 48_673 * MiB, transient, per_tok, cap);
    try t.expect(vs_granted >= ctx_default / 3);
    const vs_raw_ask = resolvedContextForLoad(0, 0, live_ceiling, active, 60 * 1024 * MiB, transient, per_tok, cap);
    try t.expect(vs_raw_ask >= ctx_default / 3);
}

test "the clamp bills the context that will be SERVED, not the placeholder" {
    // With a placeholder context the clamp hands back nearly the whole ask.
    const t = std.testing;
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    var cfg = qwen4RequestTestConfig();
    cfg.mtp_override = false; // the trunk's bill; the head term has its own test
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const static_ceiling: u64 = 120_000 * MiB;
    const ask: u64 = 60 * 1024 * MiB;

    const placeholder = planHotCache(&cfg, kv_bits, static_ceiling, active, 1024, 0, ask, 4096);
    const real = planHotCache(&cfg, kv_bits, static_ceiling, active, 1_048_576, 0, ask, 4096);

    try t.expect(placeholder.budget > 45_000 * MiB); // reproduces the live 48,673
    // 13,056 KV + 5,376 state (one history copy + the f32 score bank).
    try t.expectEqual(@as(u64, 1_048_576) * (13_056 + 2_304) + cfg.qsaRingBytes(), real.ctx_kv);

    // Both budgets are their headroom, so the whole difference is the session each billed.
    try t.expectEqual(placeholder.budget - real.budget, real.ctx_kv - placeholder.ctx_kv);
    try t.expect(real.budget < placeholder.budget);
    try t.expect(placeholder.ctx_kv < 30 * MiB);
    try t.expect(real.ctx_kv > 15_000 * MiB);
    try t.expect(real.budget > 0);
}

test "the SSD-first budget bills the FLOOR reserve, at the deployed pack's live numbers" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `ssdFirstBudgetForLoad` short-circuits above `planHotCache`, so the floor-width reserve never ran on qwen4_exp.
    const t = std.testing;
    var cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const ceiling: u64 = 109_395 * MiB;

    const orig_ctx = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = orig_ctx;
    http_server.serverState().server_config.max_context_size = 0;
    cfg.pinned_context = 0;
    per_request_chunk_override = true;
    defer per_request_chunk_override = null;
    const orig_explicit = generate_mod.prefillCli().explicit;
    defer generate_mod.prefillCli().explicit = orig_explicit;
    generate_mod.prefillCli().explicit = false;

    // The pure chunk resolver: `pinPrefillChunk` reads live memory.
    const chunk: u32 = @intCast(generate_mod.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        0,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
    try t.expect(perRequestPrefillChunkEnabled(&cfg));
    const floor_w = clampReserveWidth(&cfg, chunk);
    try t.expectEqual(PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1], floor_w);
    try t.expect(floor_w < chunk);

    const reserve_pinned = prefillTransientReserve(&cfg, kv_bits, chunk);
    const reserve_floor = prefillTransientReserve(&cfg, kv_bits, floor_w);
    try t.expect(reserve_floor < reserve_pinned);

    const ctx_kv: u64 = (kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg)) *|
        ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk);
    const ask: u64 = 10 * 1024 * MiB;

    // The idle allowance is the residual after the reserve, so the whole difference lands there.
    const now = ssdFirstPrefixCacheMem(ask, ceiling, active, ctx_kv, reserve_floor);
    const before = ssdFirstPrefixCacheMem(ask, ceiling, active, ctx_kv, reserve_pinned);
    try t.expect(now >= before);

    // The RAM arm reserves the same floor for the same pack.
    const plan = planHotCache(&cfg, kv_bits, ceiling, active, 1_048_576, sizerCtxKvBytes(&cfg, kv_bits), ask, chunk);
    try t.expectEqual(floor_w, plan.reserve_chunk);
    try t.expectEqual(reserve_floor, plan.reserve);

    // The session the floor holds is the one the server advertises.
    const advertised = autoContextFrom(
        safeContextForBudget(ceiling, active, CTX_SIZING_CACHE_RESERVE +| reserve_pinned, kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg), 0),
        cfg.contextCap(),
    );
    try t.expectEqual(advertised, ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk));
}

test "SSD-first is gated on a DISK TIER: with --prefix-cache-disk off, qwen4_exp takes the RAM arm" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `ssdFirstEnabled()` never checked that a tier exists, and `--prefix-cache-disk` is off by default.
    const t = std.testing;
    var cfg = qwen4RequestTestConfig();

    const orig_disk = http_server.launch().prefix_cache_disk_bytes;
    defer http_server.launch().prefix_cache_disk_bytes = orig_disk;
    // `prefixCacheMemForLoad` publishes into a process global; put it back or a later test reads it.
    defer clearResolvedPrefixCacheMem();
    const orig_over = prefix_cache_mod.ssd_first_override;
    defer prefix_cache_mod.ssd_first_override = orig_over;
    prefix_cache_mod.ssd_first_override = true; // the env switch is not what is on trial
    // The budget is a property of the machine; the test's machine is a 128 GB one, not the runner.
    const orig_ceiling = memory_ceiling.overrides().static_ceiling;
    defer memory_ceiling.overrides().static_ceiling = orig_ceiling;
    memory_ceiling.overrides().static_ceiling = 109_395 * (1 << 20);

    try t.expect(!prefix_cache_mod.ssdFirstActive(&cfg, false, true));
    try t.expect(prefix_cache_mod.ssdFirstActive(&cfg, true, true));
    var dense = model_mod.ModelConfig{};
    dense.model_type = "qwen3";
    dense.legacy.caps.long_ctx = std.mem.eql(u8, dense.model_type, "qwen4_exp");
    try t.expect(!prefix_cache_mod.ssdFirstActive(&dense, true, true));
    try t.expect(prefix_cache_mod.ssdFirstActive(&dense, true, false));
    prefix_cache_mod.ssd_first_override = false;
    try t.expect(!prefix_cache_mod.ssdFirstActive(&cfg, true, true));
    prefix_cache_mod.ssd_first_override = true;

    // The SSD arm floors at one session's KV, the RAM arm is a residual bounded by the ask.
    const orig_ctx = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = orig_ctx;
    http_server.serverState().server_config.max_context_size = 1_048_576; // explicit: no auto-sizing in the way
    cfg.pinned_context = 0;
    const ask: u64 = 4 * 1024 * 1024 * 1024;

    http_server.launch().prefix_cache_disk_bytes = 0;
    var idle_off: u64 = 12345;
    const ram_arm = prefixCacheMemForLoad(&cfg, ask, .{}, &idle_off);
    try t.expectEqual(@as(u64, 0), idle_off);
    try t.expect(ram_arm <= ask);

    http_server.launch().prefix_cache_disk_bytes = 64 * 1024 * 1024 * 1024;
    var idle_on: u64 = 0;
    const ssd_arm = prefixCacheMemForLoad(&cfg, ask, .{}, &idle_on);
    try t.expect(ssd_arm > ask);
    try t.expect(ssd_arm > ram_arm);
    try t.expect(idle_on > 0 and idle_on <= ask);
    try t.expect(ssd_arm -| idle_on > ask); // the one-session floor
}

test "an auto boot advertises the session the SSD-first budget floor was billed for" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The load-time bill and the pinned context must be one session: the SSD arm billed with
    // `cache_reserve = 0` while `pinAutoContext` sized against `CTX_SIZING_CACHE_RESERVE`, so
    // the floor held RAM for a larger session than the one advertised.
    const t = std.testing;
    var cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const ceiling: u64 = 109_395 * MiB;

    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;
    http_server.serverState().server_config.max_context_size = 0;
    cfg.pinned_context = 0;

    const chunk: u32 = @intCast(generate_mod.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        0,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);
    const transient: u64 = prefillTransientReserve(&cfg, kv_bits, chunk);

    const advertised = autoContextFrom(
        safeContextForBudget(ceiling, active, CTX_SIZING_CACHE_RESERVE +| transient, per_tok, 0),
        cfg.contextCap(),
    );
    const billed = ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk);
    try t.expectEqual(advertised, billed);

    // Shown on a tighter box where the checkpoint cap does not bind.
    const tight: u64 = active + 24_000 * MiB;
    const billed_t = ssdFirstSessionTokensNow(&cfg, kv_bits, tight, active, chunk);
    const old_billed = resolvedContextForLoad(0, 0, tight, active, 0, transient, per_tok, cfg.contextCap());
    try t.expect(old_billed > billed_t);
    try t.expect(billed > 500_000);
    try t.expect(billed_t > 300_000);

    const idle: u64 = 4 * 1024 * MiB;
    const floor_now = ssdFirstPrefixCacheMem(idle, tight, active, per_tok *| @as(u64, billed_t), transient);
    const floor_old = ssdFirstPrefixCacheMem(idle, tight, active, per_tok *| @as(u64, old_billed), transient);
    try t.expect(floor_old > floor_now);
    try t.expectEqual(per_tok *| @as(u64, old_billed - billed_t), floor_old - floor_now);
}

test "the load-time session bill is billed at the boot's --kv-quant, not bf16" {
    // Live: the SSD-first budget billed the session at 22,464 MB with `--kv-quant 8` and with
    // `off` alike (dense 29,952 B/tok): `defaultKvBits` asked `global_scheduler`, which `serve`
    // assigns only after `Scheduler.init` performs the load.
    const t = std.testing;
    var cfg = qwen4RequestTestConfig();
    cfg.mtp_override = false; // the trunk's bill; the head term has its own test
    const MiB: u64 = 1 << 20;
    transformer_mod.qsa_history_share_override = true;
    defer transformer_mod.qsa_history_share_override = null;
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    try t.expectEqual(@as(u64, 2_304), statePerTokenBilled(&cfg));
    try t.expectEqual(@as(u64, 13_056), kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8));
    try t.expectEqual(@as(u64, 24_576), kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 16));

    // An explicit `--ctx-size`, so the two numbers differ in the width and nothing else.
    const saved_ctx = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = saved_ctx;
    http_server.serverState().server_config.max_context_size = 786_432;
    const ceiling: u64 = 109_395 * MiB;
    const active: u64 = 69_827 * MiB;
    const chunk: u32 = cfg.pinned_prefill_chunk;

    const saved_kv = http_server.launch().configured_kv_quant;
    defer http_server.launch().configured_kv_quant = saved_kv;

    // `--kv-quant 8`, no scheduler yet: the state the load runs in.
    http_server.launch().configured_kv_quant = transformer_mod.KVQuantConfig.affine(8);
    try t.expectEqual(@as(u64, 8), defaultKvBits(&cfg));
    try t.expectEqual(
        (13_056 + 2_304) * 786_432 + cfg.qsaRingBytes(),
        ssdFirstSessionKvBytes(&cfg, defaultKvBits(&cfg), ceiling, active, chunk),
    );

    http_server.launch().configured_kv_quant = transformer_mod.KVQuantConfig.dense;
    try t.expectEqual(@as(u64, 16), defaultKvBits(&cfg));
    try t.expectEqual(
        (24_576 + 2_304) * 786_432 + cfg.qsaRingBytes(),
        ssdFirstSessionKvBytes(&cfg, defaultKvBits(&cfg), ceiling, active, chunk),
    );
    http_server.launch().configured_kv_quant = null;
    try t.expectEqual(@as(u64, 16), defaultKvBits(&cfg));

    const idle: u64 = 10 * 1024 * MiB;
    const transient: u64 = prefillTransientReserve(&cfg, 8, chunk);
    const at8 = ssdFirstPrefixCacheMem(idle, ceiling, active, 13_824 * MiB, transient);
    const at16 = ssdFirstPrefixCacheMem(idle, ceiling, active, 22_464 * MiB, transient);
    try t.expectEqual(@as(u64, 8_640 * MiB), at16 -| at8);
}

test "an explicit --ctx-size boot never consults the session reserve" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `--ctx-size` wins in the resolver's first branch; asserted as invariance in the reserve.
    const t = std.testing;
    var cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);

    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;

    http_server.serverState().server_config.max_context_size = 262_144;
    cfg.pinned_context = 0;
    for ([_]u64{ 0, 40_000 * MiB, 109_395 * MiB, 900_000 * MiB }) |ceiling| {
        for ([_]u64{ 0, 69_827 * MiB }) |active| {
            for ([_]u32{ 512, 1024, 4096, 8192 }) |chunk| {
                try t.expectEqual(@as(u32, 262_144), ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk));
            }
        }
    }
    for ([_]u64{ 0, CTX_SIZING_CACHE_RESERVE, 99_000 * MiB }) |reserve| {
        try t.expectEqual(
            @as(u32, 262_144),
            resolvedContextForLoad(262_144, 0, 109_395 * MiB, 69_827 * MiB, reserve, 3 * 1024 * MiB, per_tok, cfg.contextCap()),
        );
    }

    http_server.serverState().server_config.max_context_size = 0;
    cfg.pinned_context = 131_072;
    try t.expectEqual(@as(u32, 131_072), ssdFirstSessionTokensNow(&cfg, kv_bits, 109_395 * MiB, 69_827 * MiB, 4096));
}

test "an explicit --ctx-size keeps the load-time context byte-identical" {
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    const per_tok: u64 = 20_736;
    try t.expectEqual(@as(u32, 1_048_576), resolvedContextForLoad(1_048_576, 0, 0, 0, 99_000 * MiB, 99_000 * MiB, per_tok, 262_144));
    try t.expectEqual(@as(u32, 262_144), resolvedContextForLoad(0, 262_144, 0, 0, 99_000 * MiB, 0, per_tok, 1_048_576));
}

test "clampReserveWidth: the load-time reserve is a promise to the FIRST request" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The ceiling inversion: same ask, `iogpu.wired_limit_mb 120000` got a 3.6x smaller cache
    // because the wider rung's reserve grew faster than the headroom. Billing the ladder floor
    // makes the budget monotone in the ceiling.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    const weights: u64 = 70 * GiB;
    const ctx_tokens: u64 = 1_048_576;
    const floor_rung = PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];

    const low = planHotCache(&cfg, kv_bits, weights + 28_909 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    const high = planHotCache(&cfg, kv_bits, weights + 43_000 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    try t.expectEqual(floor_rung, low.reserve_chunk);
    try t.expectEqual(floor_rung, high.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, floor_rung), low.reserve);
    // A bigger ceiling never buys a smaller cache.
    try t.expect(high.budget >= low.budget);
    try t.expect(low.chunk >= low.reserve_chunk);

    var prev: u64 = 0;
    var mb: u64 = 24_000;
    while (mb <= 60_000) : (mb += 1_000) {
        const p = planHotCache(&cfg, kv_bits, weights + mb * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
        try t.expect(p.budget >= prev);
        prev = p.budget;
    }

    // An explicit --prefill-chunk is billed as-is.
    generate_mod.prefillCli().override = 4096;
    generate_mod.prefillCli().explicit = true;
    const pinned = planHotCache(&cfg, kv_bits, weights + 43_000 * MiB, weights, ctx_tokens, 0, 10 * GiB, explicitPrefillChunk());
    generate_mod.prefillCli().override = 8192;
    generate_mod.prefillCli().explicit = false;
    try t.expectEqual(@as(u32, 4096), pinned.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, 4096), pinned.reserve);

    // Gate off: the clamp reserves at the sizer's rung.
    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    const off = planHotCache(&cfg, kv_bits, weights + 28_909 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    try t.expectEqual(off.chunk, off.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, off.chunk), off.reserve);
    per_request_chunk_override = null;

    var other = qwen4RequestTestConfig();
    other.model_type = "qwen3_5";
    other.legacy.caps.long_ctx = std.mem.eql(u8, other.model_type, "qwen4_exp");
    const neutral = planHotCache(&other, kv_bits, weights + 28_909 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    try t.expectEqual(neutral.chunk, neutral.reserve_chunk);
}

test "the per-token widths the request chooser is built on" {
    // Pin the two constants every number in the story is derived from.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    try t.expectEqual(@as(u32, 12), cfg.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 13_056), kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8));
    transformer_mod.qsa_history_share_override = true;
    defer transformer_mod.qsa_history_share_override = null;
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    try t.expectEqual(@as(u64, 768), cfg.qsaHistoryBytesPerToken());
    try t.expectEqual(@as(u64, 1_536), cfg.qsaScoreBankBytesPerToken());
    try t.expectEqual(@as(u64, 2_304), statePerTokenBilled(&cfg));
    try t.expectEqual(@as(u64, 15_360 * (1 << 20)), (13_056 + 2_304) * @as(u64, 1_048_576));
}

// Bar: the ring is billed once per slot, independent of prompt length.
test "a 512k QSA session drops the raw-key overbill and bills the ring once" {
    const t = std.testing;
    transformer_mod.qsa_history_share_override = true;
    defer transformer_mod.qsa_history_share_override = null;
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    const cfg = qwen4ExpOomConfig();
    const seq: u64 = 512 * 1024;
    const short = prefillRequestTerms(&cfg, 1024, 2048, 8, 4096, .{});
    const long = prefillRequestTerms(&cfg, seq, 2048, 8, 4096, .{});
    try t.expectEqual(cfg.qsaRingBytes(), short.slot_fixed_kv_bytes);
    try t.expectEqual(cfg.qsaRingBytes(), long.slot_fixed_kv_bytes);
    const reserved = @max(reservedCacheTokens(seq, 2048, 4096, getEffectiveContextLength(&cfg)), seq);
    const new_needed = prefillNeededAtChunk(&cfg, seq, 2048, 8, 4096, .{});
    const old_needed = new_needed +| (reserved * 3_072 * 5 / 4);
    const ceiling: u64 = 26_000 * 1024 * 1024;
    try t.expect(old_needed > ceiling);
    try t.expect(new_needed <= ceiling);
}

test "chooseRequestPrefillChunk: the machine's chunk cap bounds the per-request width" {
    // Flash Next on CUDA chose 8192 past the 2048 cap and MLX's expert gather failed at that width.
    const t = std.testing;
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    const saved = generate_mod.prefillCli().machine_cap;
    defer generate_mod.prefillCli().machine_cap = saved;
    generate_mod.prefillCli().machine_cap = autoPrefillChunkCap(true);
    const cfg = qwen4RequestTestConfig();
    const available: u64 = 28_909 << 20;
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, 10_406, 2048, 8, available, cfg.pinned_prefill_chunk, 0, .{}));
    try t.expectEqual(@as(u32, 8192), chooseRequestPrefillChunk(&cfg, 10_406, 2048, 8, available, cfg.pinned_prefill_chunk, 8192, .{}));
}

test "chooseRequestPrefillChunk: an ordinary prompt buys the wide chunk a 1M session cannot" {
    // `--ctx-size 1048576` with no chunk or cache flags must still prefill ordinary prompts at
    // the wide chunk; load-time sizing reserved for the whole session (rung 1024 for the boot).
    const t = std.testing;
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const available: u64 = 28_909 * MiB;
    const pin = cfg.pinned_prefill_chunk;

    try t.expectEqual(@as(u32, 8192), chooseRequestPrefillChunk(&cfg, 4096, 2048, kv_bits, available, pin, 0, .{}));
    try t.expectEqual(@as(u32, 8192), chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, available, pin, 0, .{}));
    try t.expectEqual(@as(u32, 8192), chooseRequestPrefillChunk(&cfg, 384_000, 2048, kv_bits, available, pin, 0, .{}));
    try t.expect(4096 > pin);

    // The full 1M context fits at 4096 on this box under the one-copy history bill.
    const full = chooseRequestPrefillChunk(&cfg, 1_048_576, 2048, kv_bits, available, pin, 0, .{});
    try t.expectEqual(@as(u32, 4096), full);
    try t.expect(prefillNeededAtChunk(&cfg, 1_048_576, 2048, kv_bits, 2048, .{}) <= available);
    try t.expect(prefillNeededAtChunk(&cfg, 1_048_576, 2048, kv_bits, 4096, .{}) <= available);

    // A shorter prompt can afford at least as wide a forward.
    var prev: u32 = 0;
    var i: usize = 0;
    const seqs = [_]u64{ 1_048_576, 384_000, 300_000, 60_000, 4096 };
    while (i < seqs.len) : (i += 1) {
        const w = chooseRequestPrefillChunk(&cfg, seqs[i], 2048, kv_bits, available, pin, 0, .{});
        try t.expect(w >= prev);
        prev = w;
    }
}

test "chooseRequestPrefillChunk: WIDEST that fits, at the boundary" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // "Widest that fits": priced at the exact bill of a width that width comes out; one byte
    // under, a strictly narrower one (except at the floor). Walks the config's own ladder.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 300_000;
    const pin = cfg.pinned_prefill_chunk;
    const floor_width: u32 = widthForRung(&cfg, seq, PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1]);

    var prev_width: u32 = 0;
    var distinct: usize = 0;
    for (PREFILL_CHUNK_LADDER) |rung| {
        const width = widthForRung(&cfg, seq, rung);
        if (width == prev_width) continue; // a rung this arch narrows onto a width already walked
        prev_width = width;
        distinct += 1;
        const bill = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, width, .{});
        // A rung the long-context gate widened is taken only past its margin.
        const needed = if (width > rungWidth(&cfg, seq, rung, false))
            bill * (100 + PREFILL_WIDE_RUNG_MARGIN_PCT) / 100
        else
            bill;
        try t.expectEqual(width, chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, needed, pin, 0, .{}));
        const under = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, needed - 1, pin, 0, .{});
        if (width == floor_width) {
            try t.expectEqual(floor_width, under);
        } else {
            try t.expect(under < width);
        }
    }
    try t.expect(distinct >= 4);

    // Nothing fits: the ladder floor, never 0 and never the load-time pin.
    try t.expectEqual(floor_width, chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 0, pin, 0, .{}));
}

test "chooseRequestPrefillChunk: the rung the gate widened clears its bill by the estimator's margin" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 300_000;
    const pin = cfg.pinned_prefill_chunk;

    const wide_bill = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 8192, .{});
    const narrow_bill = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 4096, .{});
    try t.expect(wide_bill > narrow_bill);
    // 8192 is a width only the gate offers: ungated, this rung forwards at 4096.
    try t.expectEqual(@as(u32, 8192), widthForRung(&cfg, seq, 8192));
    try t.expectEqual(@as(u64, 4096), rungWidth(&cfg, seq, 8192, false));

    const with_margin = wide_bill * (100 + PREFILL_WIDE_RUNG_MARGIN_PCT) / 100;
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, wide_bill, pin, 0, .{}));
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, with_margin - 1, pin, 0, .{}));
    try t.expectEqual(@as(u32, 8192), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, with_margin, pin, 0, .{}));
    // The rungs the gate did not widen are still taken at their plain bill.
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, narrow_bill, pin, 0, .{}));
}

test "prefillMemoryNeeded: the wider chunk bills every chunk-scaled term at the wider width" {
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 300_000;
    const ffn: u64 = prefillFfnWidth(&cfg);
    const stream: u64 = prefillStreamBytesPerToken(&cfg);
    const dq: u64 = prefillDequantWeightBytes(&cfg);
    const keys: u64 = cfg.prefillAttnKeys(seq);

    const narrow = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, ffn, kv_bits, 4096, keys, stream, dq, .{});
    const wide = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, ffn, kv_bits, 8192, keys, stream, dq, .{});

    const mlp_step: u64 = 8 * 4096 * @max(@as(u64, cfg.hidden_size), ffn) * 2;
    const envelope_step: u64 = @max(3 * mlp_step, mlp_step + 4096 * stream);
    try t.expect(wide >= narrow + envelope_step * 5 / 4);
    try t.expect(qsaMaskBytes(&cfg, 8192, seq) > qsaMaskBytes(&cfg, 4096, seq));
}

test "prefillNeededAtChunk: the deployed pack's 8192 step is at least the measured peak step" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = qwen4DeployedTestConfig();
    const kv_bits: u64 = 8;

    for ([_]u64{ 300_000, 326_351 }) |seq| {
        for ([_]bool{ false, true }) |mtp_on| {
            const narrow = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 4096, .{ .mtp_on = mtp_on });
            const wide = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 8192, .{ .mtp_on = mtp_on });
            // The widest active+cache step the 4096 -> 8192 trade measured on this pack was
            // +3.3 GB; the bill for the step must not sit under what the box pays for it.
            try t.expect(wide - narrow >= 3_300_000_000);
        }
    }
}

test "chooseRequestPrefillChunk: the explicit flag and the gate both outrank it" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const roomy: u64 = 200 * (@as(u64, 1) << 30);
    const pin = cfg.pinned_prefill_chunk;

    // An explicit `--prefill-chunk` wins outright and is billed as-is.
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, roomy, pin, 2048, .{}));
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, 0, pin, 2048, .{}));

    // Kill switch: the load-time pin for every request.
    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    try t.expectEqual(pin, chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, roomy, pin, 0, .{}));
    per_request_chunk_override = null;

    // Another arch keeps the load-time pin exactly.
    var other = qwen4RequestTestConfig();
    other.model_type = "qwen3_5";
    other.legacy.caps.long_ctx = std.mem.eql(u8, other.model_type, "qwen4_exp");
    try t.expect(!other.perRequestPrefillChunk());
    try t.expectEqual(pin, chooseRequestPrefillChunk(&other, 300_000, 2048, kv_bits, roomy, pin, 0, .{}));
    try t.expect(cfg.perRequestPrefillChunk());
}

test "admission tries the ladder DOWN before refusing: a prompt that fits only at 512" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The probe used to price the load-time pin while the scheduler chose a per-request width,
    // so a prompt that fit only at the floor was refused for a forward that never ran.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 384_000;
    const pin = cfg.pinned_prefill_chunk;
    const floor_width = widthForRung(&cfg, seq, PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1]);
    const pin_width = widthForRung(&cfg, seq, pin);

    try t.expect(pin_width > floor_width);
    const at_floor = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, floor_width, .{});
    const at_pin = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, pin_width, .{});
    try t.expect(at_floor < at_pin);

    // Exactly enough for the floor width and not a byte more.
    const available = at_floor;

    try t.expectEqual(floor_width, chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, available, pin, 0, .{}));
    try t.expect(prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, floor_width, .{}) <= available);

    // Gate off: the load-time pin does not fit.
    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    try t.expectEqual(pin, chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, available, pin, 0, .{}));
    try t.expect(prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, pin_width, .{}) > available);
    per_request_chunk_override = null;

    // An explicit `--prefill-chunk` keeps the refusal: the operator chose the width.
    generate_mod.prefillCli().override = 4096;
    generate_mod.prefillCli().explicit = true;
    defer {
        generate_mod.prefillCli().override = 8192;
        generate_mod.prefillCli().explicit = false;
    }
    const forced = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, available, pin, explicitPrefillChunk(), .{});
    try t.expectEqual(@as(u32, 4096), forced);
    try t.expect(prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, forced, .{}) > available);
}

test "the post-eviction re-ask never exceeds what live memory affords, and never runs on an arch that has no per-request width" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The composition the scheduler runs: the chooser is asked before and after the eviction
    // pass, and the second reading (live memory) runs.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 383_000;
    const pin: u32 = 512;
    const floor_width = widthForRung(&cfg, seq, PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1]);

    // Pre-eviction: nothing on the ladder fits, the chooser hands back the floor.
    const admitted = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 0, pin, 0, .{});
    try t.expectEqual(floor_width, admitted);

    // Post-eviction, with the cache's bytes returned: a wider rung fits.
    const roomy: u64 = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, widthForRung(&cfg, seq, 4096), .{});
    const reasked = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, roomy, pin, 0, .{});
    try t.expect(reasked > admitted);
    try t.expectEqual(reasked, scheduler_mod.postEvictionPrefillChunk(admitted, reasked).width);

    try t.expectEqual(admitted, scheduler_mod.postEvictionPrefillChunk(0, admitted).width);

    // The reverse reading (memory moved between the asks) runs at the floor.
    try t.expectEqual(admitted, scheduler_mod.postEvictionPrefillChunk(reasked, admitted).width);

    // A non-qwen4 arch is untouched end to end.
    var other = qwen4RequestTestConfig();
    other.model_type = "qwen3_5";
    other.legacy.caps.long_ctx = std.mem.eql(u8, other.model_type, "qwen4_exp");
    const other_pre = chooseRequestPrefillChunk(&other, seq, 2048, kv_bits, 0, pin, 0, .{});
    const other_post = chooseRequestPrefillChunk(&other, seq, 2048, kv_bits, roomy, pin, 0, .{});
    try t.expectEqual(pin, other_pre);
    try t.expectEqual(pin, other_post);
    try t.expectEqual(pin, scheduler_mod.postEvictionPrefillChunk(other_pre, other_post).width);

    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    const off_pre = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 0, pin, 0, .{});
    const off_post = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, roomy, pin, 0, .{});
    try t.expectEqual(pin, scheduler_mod.postEvictionPrefillChunk(off_pre, off_post).width);
}

test "admissionLogLevel: free on the first request after a load and at the edge, quiet in the middle" {
    const t = std.testing;
    const GiB: u64 = 1 << 30;

    // A fresh load is always free.
    try t.expectEqual(log.Level.info, admissionLogLevel(1 * GiB, 100 * GiB, true));

    // Inside 10% of what is available.
    try t.expectEqual(log.Level.info, admissionLogLevel(95 * GiB, 100 * GiB, false));
    try t.expectEqual(log.Level.info, admissionLogLevel(120 * GiB, 100 * GiB, false)); // already over
    // The boundary itself: exactly 0.9 is quiet.
    try t.expectEqual(log.Level.debug, admissionLogLevel(90 * GiB, 100 * GiB, false));
    try t.expectEqual(log.Level.info, admissionLogLevel(90 * GiB + 1, 100 * GiB, false));

    try t.expectEqual(log.Level.debug, admissionLogLevel(1 * GiB, 100 * GiB, false));

    try t.expectEqual(log.Level.info, admissionLogLevel(1 * GiB, 0, false));

    try t.expectEqual(log.Level.info, admissionLogLevel(std.math.maxInt(u64), 1, false));
    try t.expectEqual(log.Level.debug, admissionLogLevel(1, std.math.maxInt(u64), false));
}

test "admissionVerdict names the same three arms the guard acts on" {
    const t = std.testing;
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(.{ .needed = 10, .available = 20 }));
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(.{ .needed = 20, .available = 20 }));
    // Fits once the cache gives back what it can prove: `reclaimable`, never `evictable`.
    try t.expectEqual(AdmissionVerdict.evict, admissionVerdict(.{ .needed = 30, .available = 20, .reclaimable = 10 }));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(.{ .needed = 30, .available = 20, .evictable = 1000, .reclaimable = 0 }));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(.{ .needed = 100, .available = 20, .reclaimable = 10 }));

    try t.expectEqualStrings("admit", AdmissionVerdict.admit.name());
    try t.expectEqualStrings("evict", AdmissionVerdict.evict.name());
    try t.expectEqualStrings("refuse", AdmissionVerdict.refuse.name());
}

test "resolvePrefillChunk: a squeezed box steps the chunk down; a roomy one keeps it" {
    // The reserve at the widest chunk must never claim more than a quarter of
    // what is left after the weights. Numbers are the measured 16 GB profile: Metal's working
    // set ~11.9 GiB, Mistral-7B-4bit 3.80 GiB resident. The hot-cache budget is no longer an
    // input; the former `2 * GiB` argument is the pinned-context KV bar and is 0 under auto-context.
    const t = std.testing;
    const GiB: u64 = 1024 * 1024 * 1024;
    var mistral = model_mod.ModelConfig{};
    mistral.num_hidden_layers = 32;
    mistral.num_attention_heads = 32;
    mistral.num_key_value_heads = 8;
    mistral.head_dim = 128;
    mistral.hidden_size = 4096;
    mistral.intermediate_size = 14336;
    mistral.intermediate_size_declared = true;
    mistral.quant_bits = 4;
    mistral.max_position_embeddings = 32768;

    const ceiling_16: u64 = 11_918_000_000;
    const weights: u64 = 4_080_000_000;
    const narrowed = resolvePrefillChunk(&mistral, 16, ceiling_16, weights, 0, 0);
    try t.expect(narrowed < 8192);
    try t.expect(narrowed >= 512);
    // It is the WIDEST rung that fits, not simply the floor.
    try t.expect(prefillTransientReserve(&mistral, 16, narrowed) <=
        (ceiling_16 - weights) / 4);
    for (PREFILL_CHUNK_LADDER) |c| {
        if (c <= narrowed) break;
        try t.expect(prefillTransientReserve(&mistral, 16, c) > (ceiling_16 - weights) / 4);
    }

    // Same model, a machine with room: nothing narrows.
    try t.expectEqual(@as(u32, 8192), resolvePrefillChunk(&mistral, 16, 95 * GiB, weights, 0, 0));

    // A model that barely fits gets the narrowest rung rather than a wide one
    // it cannot pay for — and never 0, which would read as "not pinned".
    try t.expectEqual(@as(u32, 512), resolvePrefillChunk(&mistral, 16, weights + 64 * 1024 * 1024, weights, 0, 0));

    // Every rung is a real ladder entry, descending, floored at generate's own
    // minimum — a chunk generate would refuse is a chunk the bill cannot model.
    var prev: u32 = std.math.maxInt(u32);
    for (PREFILL_CHUNK_LADDER) |c| {
        try t.expect(c < prev);
        try t.expect(c >= generate_mod.PREFILL_CHUNK_FLOOR);
        prev = c;
    }
}

test "a loaded DFlash drafter's context K/V is billed per prompt row" {
    var cfg = qwen4ExpOomConfig();
    const base = prefillNeededAtChunk(&cfg, 32768, 1024, 16, 2048, .{});
    cfg.drafter_ctx_bytes_per_token = 20480;
    try std.testing.expectEqual(base + (32768 + 1024) * 20480, prefillNeededAtChunk(&cfg, 32768, 1024, 16, 2048, .{}));
}

/// The widest width the adaptive chooser may take for a prompt on this config.
fn adaptCapFor(cfg: *const model_mod.ModelConfig, seq: u64) u32 {
    return widthForRung(cfg, seq, PREFILL_CHUNK_LADDER[0]);
}

test "resolvePrefillChunk: only an arch with no prefill score tensor may pin the wide rung" {
    const mimo: model_mod.ModelConfig = .{ .model_type = "mimo_v2", .head_dim = 192, .num_attention_heads = 64, .num_key_value_heads = 8, .hidden_size = 4096, .intermediate_size = 16384, .num_hidden_layers = 48 };
    const dense: model_mod.ModelConfig = .{ .model_type = "llama", .head_dim = 128, .num_attention_heads = 32, .num_key_value_heads = 8, .hidden_size = 4096, .intermediate_size = 14336, .num_hidden_layers = 32 };
    const huge: u64 = 1 << 50;
    try std.testing.expectEqual(@as(u32, generate_mod.WIDE_PREFILL_CHUNK), resolvePrefillChunk(&mimo, 16, huge, 0, 0, 0));
    try std.testing.expectEqual(@as(u32, 8192), resolvePrefillChunk(&dense, 16, huge, 0, 0, 0));
}

test "adaptivePrefillWidth: a step-down is immediate, by as many rungs as it takes, and never 0" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The down direction does not wait for a trend or a margin.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 1_048_576;
    const kv: u64 = seq;
    const cap = adaptCapFor(&cfg, seq);

    {
        var st: generate_mod.AdaptiveWidthState = .{};
        const h = prefillChunkCost(&cfg, kv_bits, 2048, kv);
        try t.expect(h < prefillChunkCost(&cfg, kv_bits, cap, kv));
        try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, h, cap, cap, &st));
        try t.expect(st.ratcheted);
    }
    // Several rungs in one call.
    {
        var st: generate_mod.AdaptiveWidthState = .{};
        const h = prefillChunkCost(&cfg, kv_bits, 512, kv);
        try t.expectEqual(@as(u32, 512), adaptivePrefillWidth(&cfg, kv_bits, kv, h, cap, cap, &st));
    }
    // No headroom at all: the ladder floor, never 0 (the loop reads 0 as "no change").
    {
        var st: generate_mod.AdaptiveWidthState = .{};
        const got = adaptivePrefillWidth(&cfg, kv_bits, kv, 0, cap, cap, &st);
        try t.expectEqual(@as(u32, @intCast(generate_mod.PREFILL_CHUNK_FLOOR)), got);
        try t.expect(got != 0);
    }
    {
        var st: generate_mod.AdaptiveWidthState = .{};
        const floor: u32 = @intCast(generate_mod.PREFILL_CHUNK_FLOOR);
        try t.expectEqual(floor, adaptivePrefillWidth(&cfg, kv_bits, kv, 0, floor, cap, &st));
    }
}

test "adaptivePrefillWidth: HOLD is margin 1.0 — the admitted width is not second-guessed" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `prefillMemoryNeeded` already carries its own 5/4; a second 1.25 on the hold stepped the
    // 768k prompt down from the width admission admitted.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const kv: u64 = 786_432;
    const cap = adaptCapFor(&cfg, 786_432);
    const cost_2k = prefillChunkCost(&cfg, kv_bits, 2048, kv);

    var st: generate_mod.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, cost_2k, 2048, cap, &st));
    try t.expect(!st.ratcheted);
    try t.expect(cost_2k * 5 / 4 > cost_2k);
    var st2: generate_mod.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, cost_2k * 5 / 4 - 1, 2048, cap, &st2));
    var st3: generate_mod.AdaptiveWidthState = .{};
    try t.expect(adaptivePrefillWidth(&cfg, kv_bits, kv, cost_2k - 1, 2048, cap, &st3) < 2048);
}

test "adaptivePrefillWidth: a widen costs 1.25x AND two consecutive supporting probes" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Holding is free, growing is a bet: two consecutive probes.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const kv: u64 = 524_288;
    const cap = adaptCapFor(&cfg, 524_288);
    // Starts at half the cap so one widen step lands exactly on the cap.
    const from: u32 = cap / 2;
    const cost_up = prefillChunkCost(&cfg, kv_bits, cap, kv);

    var st: generate_mod.AdaptiveWidthState = .{};
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st));
    try t.expectEqual(@as(u8, 1), st.supporting);
    try t.expectEqual(cap, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st));
    try t.expectEqual(@as(u8, 0), st.supporting); // and the count restarts at the new width

    // Room for the wider chunk's bill but not its margin: never.
    var st2: generate_mod.AdaptiveWidthState = .{};
    const between = cost_up * 5 / 4 - 1;
    try t.expect(between >= cost_up);
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, between, from, cap, &st2));
    }

    // A single unsupporting probe resets the run.
    var st3: generate_mod.AdaptiveWidthState = .{};
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st3));
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, between, from, cap, &st3));
    try t.expectEqual(@as(u8, 0), st3.supporting);
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st3));
}

test "the tail-merge gate reads the ARCH, not the installed hook (serve installs it for everyone)" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `serve` installs `prefill_chunk_adapt` unconditionally, so `chunk_width_hook != null` is
    // true on every arch and reading it as the gate put the scaled bound back everywhere. This
    // builds the slot state the way `serve` does and asserts the two answers differ.
    const t = std.testing;
    scheduler_mod.prefill_chunk_adapt = &adaptivePrefillWidthNow;
    defer scheduler_mod.prefill_chunk_adapt = null;
    scheduler_mod.prefill_chunk_widen_ok = &adaptivePrefillWidenStillFits;
    defer scheduler_mod.prefill_chunk_widen_ok = null;
    scheduler_mod.prefill_chunk_adaptive_enabled = &adaptivePrefillChunkEnabled;
    defer scheduler_mod.prefill_chunk_adaptive_enabled = null;

    const hook_installed = scheduler_mod.prefill_chunk_adapt != null;
    try t.expect(hook_installed);

    var other = qwen4RequestTestConfig();
    other.model_type = "qwen3_5";
    other.legacy.caps.long_ctx = std.mem.eql(u8, other.model_type, "qwen4_exp");
    const adaptive = scheduler_mod.adaptiveChunkWidthFor(&other);
    try t.expect(!adaptive);
    try t.expect(hook_installed != adaptive);

    // A qwen3_5 slot keeps the flat bound, hook and all.
    try t.expectEqual(generate_mod.TAIL_MERGE_MAX, generate_mod.tailMergeMaxFor(512, adaptive));
    try t.expectEqual(@as(usize, 812), generate_mod.nextChunkEnd(0, 812, 512, false, 0, 0, adaptive, false));
    try t.expectEqual(@as(usize, 1), generate_mod.prefillChunkCount(812, 512, false, 0, 0, adaptive, false));

    const cfg = qwen4RequestTestConfig();
    per_request_chunk_override = true;
    defer per_request_chunk_override = null;
    adaptive_chunk_override = true;
    defer adaptive_chunk_override = null;
    const unpinned = explicitPrefillChunk() == 0;
    try t.expectEqual(unpinned, scheduler_mod.adaptiveChunkWidthFor(&cfg));
    if (unpinned) {
        try t.expectEqual(@as(usize, 64), generate_mod.tailMergeMaxFor(512, scheduler_mod.adaptiveChunkWidthFor(&cfg)));
        try t.expectEqual(@as(usize, 512), generate_mod.nextChunkEnd(0, 812, 512, false, 0, 0, scheduler_mod.adaptiveChunkWidthFor(&cfg), false));
    }
    try t.expect(!scheduler_mod.adaptiveChunkWidthFor(null));
    scheduler_mod.prefill_chunk_adaptive_enabled = null;
    try t.expect(!scheduler_mod.adaptiveChunkWidthFor(&cfg));
    try t.expect(scheduler_mod.prefill_chunk_adapt != null); // ...with the hook still installed
}

test "the tail-merge bound scales ONLY where the per-chunk adaptive width is live" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Every arch but an adaptive-width qwen4_exp keeps the flat `TAIL_MERGE_MAX` at every chunk width.
    const t = std.testing;

    var other = qwen4RequestTestConfig();
    other.model_type = "qwen3_5";
    other.legacy.caps.long_ctx = std.mem.eql(u8, other.model_type, "qwen4_exp");
    try t.expect(!other.perRequestPrefillChunk());
    try t.expect(!adaptivePrefillChunkEnabled(&other));
    const off = adaptivePrefillChunkEnabled(&other);
    try t.expectEqual(generate_mod.TAIL_MERGE_MAX, generate_mod.tailMergeMaxFor(512, off));
    try t.expectEqual(generate_mod.TAIL_MERGE_MAX, generate_mod.tailMergeMaxFor(2048, off));
    try t.expectEqual(@as(usize, 812), generate_mod.nextChunkEnd(0, 812, 512, false, 0, 0, off, false));

    const cfg = qwen4RequestTestConfig();
    try t.expect(cfg.perRequestPrefillChunk());
    per_request_chunk_override = true;
    defer per_request_chunk_override = null;
    adaptive_chunk_override = true;
    defer adaptive_chunk_override = null;
    const unpinned = explicitPrefillChunk() == 0;
    try t.expectEqual(unpinned, adaptivePrefillChunkEnabled(&cfg));
    try t.expectEqual(@as(usize, 64), generate_mod.tailMergeMaxFor(512, true));
    try t.expectEqual(@as(usize, 512), generate_mod.nextChunkEnd(0, 812, 512, false, 0, 0, true, false));
    // An operator pin turns the scaling back off.
    adaptive_chunk_override = false;
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
    try t.expectEqual(generate_mod.TAIL_MERGE_MAX, generate_mod.tailMergeMaxFor(512, adaptivePrefillChunkEnabled(&cfg)));
}

test "adaptivePrefillWidth: the ratchet is one-way, and the cap is a ceiling" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const kv: u64 = 1_048_576;
    const cap = adaptCapFor(&cfg, 1_048_576);

    // Step down once, then hand it the whole machine: it stays (a re-widen would be a metronome).
    var st: generate_mod.AdaptiveWidthState = .{};
    const narrowed = adaptivePrefillWidth(&cfg, kv_bits, kv, prefillChunkCost(&cfg, kv_bits, 1024, kv), cap, cap, &st);
    try t.expect(narrowed < cap);
    try t.expect(st.ratcheted);
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(narrowed, adaptivePrefillWidth(&cfg, kv_bits, kv, std.math.maxInt(u64) / 2, narrowed, cap, &st));
    }

    // Nothing above the arch cap, ever.
    var st2: generate_mod.AdaptiveWidthState = .{};
    i = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(cap, adaptivePrefillWidth(&cfg, kv_bits, kv, std.math.maxInt(u64) / 2, cap, cap, &st2));
    }
    // A cap of 0 is "do not touch the width".
    var st3: generate_mod.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, 0, 2048, 0, &st3));
}

test "adaptivePrefillWidth: a 1M prompt walks DOWN as the box fills, and a small one never moves" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Sweep: the width is the widest rung that fits, only narrows within one prefill, and is always forwardable.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const kv: u64 = 1_048_576;
    const cap = adaptCapFor(&cfg, 1_048_576);

    var st: generate_mod.AdaptiveWidthState = .{};
    var width: u32 = cap;
    var seen_narrower: usize = 0;
    var headroom: u64 = 50 * 1024 * MiB; // the wired-limit box, quiet
    var step: usize = 0;
    while (step < 64) : (step += 1) {
        const next = adaptivePrefillWidth(&cfg, kv_bits, kv, headroom, width, cap, &st);
        try t.expect(next <= width); // never widens after the first step-down
        try t.expect(next >= @as(u32, @intCast(generate_mod.PREFILL_CHUNK_FLOOR)));
        try t.expect(next <= cap);
        try t.expect(prefillChunkCost(&cfg, kv_bits, next, kv) <= headroom or
            next == @as(u32, @intCast(generate_mod.PREFILL_CHUNK_FLOOR)));
        if (next < width) seen_narrower += 1;
        width = next;
        headroom -|= 1024 * MiB; // a co-tenant, or the accreting state
    }
    try t.expect(seen_narrower >= 2);
    try t.expectEqual(@as(u32, @intCast(generate_mod.PREFILL_CHUNK_FLOOR)), width);

    // A 32k prompt on the same box: the width never moves.
    var st2: generate_mod.AdaptiveWidthState = .{};
    const cap32 = adaptCapFor(&cfg, 32_768);
    var w32: u32 = cap32;
    step = 0;
    while (step < 16) : (step += 1) {
        w32 = adaptivePrefillWidth(&cfg, kv_bits, 32_768, 50 * 1024 * MiB, w32, cap32, &st2);
        try t.expectEqual(cap32, w32);
    }
    try t.expect(!st2.ratcheted);
}

test "prefillHeadroomNow subtracts the SSD writer's staged HOST bytes" {
    // Staged host bytes come out of the same unified pool the widen bar reads.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    try t.expectEqual(90 * GiB, prefillHeadroomFrom(120 * GiB, 30 * GiB, 0));
    try t.expectEqual(89 * GiB, prefillHeadroomFrom(120 * GiB, 30 * GiB, GiB));
    // Saturating: a permit larger than the headroom refuses every widen.
    try t.expectEqual(@as(u64, 0), prefillHeadroomFrom(120 * GiB, 119 * GiB, 8 * GiB));
}

test "adaptivePrefillWidth: the widen prices the QSA sheet at the LIVE KV, not at one chunk" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `prefillTransientReserve` bills the QSA sheet at `kv = chunk`; past the indexer budget the
    // sheet costs its whole budget, so a widen decided on that number is unguarded.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const cap = adaptCapFor(&cfg, 786_432);
    const short_kv: u64 = 4096;
    const long_kv: u64 = 400_000;

    // The bill is strictly larger at the live KV.
    const at_short = prefillChunkCost(&cfg, kv_bits, cap, short_kv);
    const at_long = prefillChunkCost(&cfg, kv_bits, cap, long_kv);
    try t.expect(at_long > at_short);
    // `prefillTransientReserve` is exactly the `kv = chunk` case.
    try t.expectEqual(
        prefillTransientReserve(&cfg, kv_bits, cap),
        prefillTransientReserveAtKv(&cfg, kv_bits, cap, cap),
    );

    // Headroom that clears the short bill's 1.25 bar but not the long one: widen at 4k, refuse at 400k.
    const headroom: u64 = at_short * 5 / 4 + 1;
    try t.expect(headroom < at_long * 5 / 4);

    // Starts at half the cap so one widen step lands exactly on the cap.
    const from: u32 = cap / 2;
    var st_short: generate_mod.AdaptiveWidthState = .{};
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, short_kv, headroom, from, cap, &st_short));
    try t.expectEqual(cap, adaptivePrefillWidth(&cfg, kv_bits, short_kv, headroom, from, cap, &st_short));

    var st_long: generate_mod.AdaptiveWidthState = .{};
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, long_kv, headroom, from, cap, &st_long));
    }
    try t.expectEqual(@as(u8, 0), st_long.supporting);

    // The hold keeps its margin-1.0 semantics at the live KV.
    const hold = prefillChunkCost(&cfg, kv_bits, 2048, long_kv);
    var st_hold: generate_mod.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, long_kv, hold, 2048, cap, &st_hold));
    var st_down: generate_mod.AdaptiveWidthState = .{};
    try t.expect(adaptivePrefillWidth(&cfg, kv_bits, long_kv, hold - 1, 2048, cap, &st_down) < 2048);
}

test "adaptivePrefillChunkEnabled: the arch, both kill switches, and any pinned width" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    var other = qwen4RequestTestConfig();
    other.model_type = "qwen3_5_moe";
    other.legacy.caps.long_ctx = std.mem.eql(u8, other.model_type, "qwen4_exp");

    try t.expect(adaptivePrefillChunkEnabled(&cfg));
    // Another arch never adapts; one switch disables both.
    try t.expect(!adaptivePrefillChunkEnabled(&other));

    per_request_chunk_override = false;
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
    per_request_chunk_override = null;
    try t.expect(adaptivePrefillChunkEnabled(&cfg));

    adaptive_chunk_override = false;
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
    adaptive_chunk_override = null;

    // An operator who pinned a width pinned every forward.
    generate_mod.prefillCli().override = 4096;
    generate_mod.prefillCli().explicit = true;
    defer {
        generate_mod.prefillCli().override = 8192;
        generate_mod.prefillCli().explicit = false;
    }
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
}

test "a WARM turn is not billed for the prefix it is about to SHARE (786,707 @ 19,032 MB)" {
    const t = std.testing;
    // Live: a 786,707-token warm turn restored its own 786,676-token entry and was then refused
    // by name; the restore hands over the entry's buffers without a second copy, so those
    // ~14.5 GB were already inside `mlx_get_active_memory` and `needed` billed them again.
    const cfg = qwen4ExpOomConfig();
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    const seq: u64 = 786_707;
    const chunk: u64 = 512; // the width the ladder came down to that evening
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 200;
    const mb: u64 = 1024 * 1024;
    const available: u64 = 19_032 * mb;

    // 13,056 B of 8-bit KV and 5,376 B of state per token (one history copy + the score bank).
    // The credit prices KV only: the history is privatised on restore.
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    try t.expectEqual(@as(u64, 13_056), kv_per_tok);
    try t.expectEqual(@as(u64, 768), cfg.qsaHistoryBytesPerToken());
    try t.expectEqual(@as(u64, 2_304), statePerTokenBilled(&cfg));

    // Cold: the refusal, reproduced.
    const cold = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{});
    try t.expect(cold / mb > 18_000 and cold / mb < 23_000);
    try t.expect(cold > available);

    // Warm: the entry's buffers hold the capacity the previous turn reserved, which still covers
    // this turn's reservation, so nothing grows.
    const matched: u64 = 786_676;
    const capacity: u64 = 787_456;
    const reserved = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    try t.expectEqual(@as(u64, 787_419), reserved);
    try t.expect(reserved <= capacity);

    // `will_donate`: this restore checked out its entry, so the buffers are the slot's own.
    const warm_prefix = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = capacity, .will_donate = true };
    try t.expectEqual(matched, warm_prefix.creditedRows(reserved));

    // The credit is the handed-over rows at this arch's KV width and nothing else, inside the 5/4.
    const credit = matched * kv_per_tok;
    try t.expectEqual(@as(u64, 786_676 * 13_056), credit);
    try t.expect(credit < matched * (kv_per_tok + cfg.qsaHistoryBytesPerToken()));
    const warm = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, warm_prefix);
    // Isolated against a shared restore of the same span: `checkpoint_bytes` follows `matched` too.
    const shared_same_span = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = capacity });
    try t.expect(shared_same_span < cold);
    try t.expectEqual(shared_same_span - credit * 5 / 4, warm);

    // The verdict flips: admitted outright, no eviction pass. Under-crediting is the safe direction.
    try t.expect(warm < available);
    try t.expect(warm / mb > 6_500 and warm / mb < 8_500);

    // A cold prompt on the same box keeps the old refusal.
    try t.expect(prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{}) > available);
    try t.expect(prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{ .matched_tokens = 0, .capacity_tokens = capacity, .will_donate = true }) > available);
}

test "prefillChunkCap: the chunk sizer's two long-context changes are qwen4_exp-only" {
    // Both sizer changes (dropping the ask, the ctx bar) moved the prefill width on every arch;
    // the ungated arithmetic is the previous `resolvePrefillChunk`.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const ceiling: u64 = 120 * GiB;
    const weights: u64 = 69 * GiB;
    const ask: u64 = 10 * GiB;
    const ctx_kv: u64 = 20 * GiB;

    // Every other arch: the ask is spoken for, the context is not, no HOT_CACHE_FLOOR.
    for ([_][]const u8{ "qwen3_5", "qwen3_5_moe", "lfm2", "nemotron_h", "bailing_hybrid", "llama", "mistral" }) |mt| {
        var cfg = longCtxTestConfig();
        cfg.model_type = mt;
        cfg.legacy.caps.long_ctx = std.mem.eql(u8, cfg.model_type, "qwen4_exp");
        const baseline: u64 = (ceiling - (weights + ask)) / PREFILL_RESERVE_BUDGET_SHARE;
        try t.expectEqual(baseline, prefillChunkCap(&cfg, ceiling, weights, ctx_kv, ask));
        try t.expectEqual(baseline, prefillChunkCap(&cfg, ceiling, weights, 48 * GiB, ask));
        try t.expect(prefillChunkCap(&cfg, ceiling, weights, ctx_kv, 0) > baseline);
    }

    // qwen4_exp: the ask is ignored, and the ctx bar binds.
    var q4 = longCtxTestConfig();
    q4.model_type = "qwen4_exp";
    q4.legacy.caps.long_ctx = true;
    const serving: u64 = ceiling - weights;
    const share_bar: u64 = serving / PREFILL_RESERVE_BUDGET_SHARE;
    const want: u64 = @min(share_bar, (serving - ctx_kv) -| HOT_CACHE_FLOOR_BYTES);
    try t.expectEqual(want, prefillChunkCap(&q4, ceiling, weights, ctx_kv, ask));
    try t.expectEqual(
        prefillChunkCap(&q4, ceiling, weights, ctx_kv, 0),
        prefillChunkCap(&q4, ceiling, weights, ctx_kv, 24 * GiB),
    );
    try t.expect(prefillChunkCap(&q4, ceiling, weights, 48 * GiB, ask) < share_bar);
}

test "the live 364k session's admission numbers reproduce, to the megabyte" {
    // The anchor: fed the deployed config and the incident's prompt lengths, the estimator prints
    // the incident's own numbers (Turn A admitted at 16,233 MB; Turn B refused at 13,664 MB).
    const t = std.testing;
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    const cfg = qwen4ExpLive364kConfig();
    const mb: u64 = 1024 * 1024;
    const kv_bits: u64 = 8;
    // Turn A, width 2048, prompt 364,509, max_gen 421,923.
    try t.expectEqual(
        @as(u64, 12_189),
        prefillNeededAtChunk(&cfg, 364_509, 421_923, kv_bits, 2048, .{}) / mb,
    );
    try t.expectEqual(
        @as(u64, 10_774),
        prefillNeededAtChunk(&cfg, 368_208, 418_224, kv_bits, 512, .{}) / mb,
    );
}
