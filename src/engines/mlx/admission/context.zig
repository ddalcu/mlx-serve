//! auto context, getEffectiveContextLength; contextForBudget, computeMaxSafeContext;
//! computeMemoryContext, ctxSizingCacheReserve

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const generate_mod = @import("../generate.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const mlx_stream = @import("../modules/stream.zig");
const metrics = @import("../../../host/status.zig");
const system_vars = @import("../../../sdk/util/system.zig");
const testing = std.testing;
const http_server = @import("../../../host/http/server.zig");

const manualContext = @import("../../../host/http/server.zig").manualContext;
const physicalMemoryCeiling = @import("../../../host/registry/memory_ceiling.zig").physicalMemoryCeiling;
const currentGpuMemoryCeiling = @import("../../../host/registry/memory_ceiling.zig").currentGpuMemoryCeiling;
const defaultKvBits = @import("bills.zig").defaultKvBits;
const prefillTransientReserve = @import("bills.zig").prefillTransientReserve;
const pinPrefillChunk = @import("prefill_chunk.zig").pinPrefillChunk;
const ctxCacheShare = @import("hot_cache_plan.zig").ctxCacheShare;
const resolvedContextForLoad = @import("hot_cache_plan.zig").resolvedContextForLoad;
const CTX_SIZING_CACHE_RESERVE = @import("hot_cache_plan.zig").CTX_SIZING_CACHE_RESERVE;
const legacyPrefixCacheAsk = @import("hot_cache_plan.zig").legacyPrefixCacheAsk;
const longCtxTestConfig = @import("bills_test.zig").longCtxTestConfig;
const qsaScoreFusedOffGuard = @import("bills_test.zig").qsaScoreFusedOffGuard;
const qwen4RequestTestConfig = @import("bills_test.zig").qwen4RequestTestConfig;
const kvBytesPerTokenAtBits = @import("bills.zig").kvBytesPerTokenAtBits;
const statePerTokenBilled = @import("bills.zig").statePerTokenBilled;
const sessionBytesPerToken = @import("bills.zig").sessionBytesPerToken;
const clampMaxTokens = @import("../../../host/api/sampling.zig").clampMaxTokens;

/// Percentage of the memory-derived ceiling we actually admit as context when
/// `--ctx-size` is absent.
///
/// `computeMaxSafeContext` returns the largest context that FITS right now.
/// Serving at exactly that leaves nothing for the prefix cache to grow into, a
/// second model to load beside this one, or another app to take RAM — and the
/// Metal OOM it would eventually hit is uncatchable (see the auto-context
/// gotcha). Reserve headroom instead.
pub const auto_ctx_safety_pct: u32 = 85;

/// PURE: apply the safety margin and round down to a 1024 boundary so the
/// number reads sanely in logs and client configs. Never returns 0.
///
/// Takes the MEMORY-derived ceiling only. The model's own `max_position_embeddings`
/// is applied afterwards, un-margined: when the checkpoint's max is the binding
/// constraint there is nothing to reserve memory headroom against, and shaving
/// 15% off a 131,072-token model that comfortably fits in RAM just throws
/// context away.
pub fn safeAutoContext(raw: u32) u32 {
    const scaled: u64 = (@as(u64, raw) * auto_ctx_safety_pct) / 100;
    const rounded: u64 = (scaled / 1024) * 1024;
    if (rounded == 0) return @intCast(@max(scaled, 1));
    return @intCast(rounded);
}

/// This model's auto-context: memory ceiling with headroom, then capped by what
/// the checkpoint actually supports. `contextCap` is the rope-derived window:
/// `max_position_embeddings`, or `original_max_position_embeddings × factor` for
/// a YaRN-scaled checkpoint — the same derivation vLLM applies to
/// `max_model_len`, because a position past the scaled window aliases back
/// inside the ramp rather than reading as a longer distance.
/// The context the server advertises, from the largest context memory alone allows (85%
/// margin on the memory number, checkpoint cap afterwards). Shared by the load-time session
/// bill and the boot-time sizer so the two cannot spell the relation two ways.
pub fn autoContextFrom(memory_ctx: u32, ctx_cap: u32) u32 {
    const with_headroom = safeAutoContext(memory_ctx);
    return if (ctx_cap > 0) @min(with_headroom, ctx_cap) else with_headroom;
}

fn autoContextFor(config: *const model_mod.ModelConfig) u32 {
    return autoContextFrom(computeMemoryContext(config), config.contextCap());
}

/// Freeze this model's auto-context at load time. Idempotent; a no-op (and
/// never a write) when `--ctx-size` pinned it explicitly. Returns the value
/// `getEffectiveContextLength` will report from now on.
///
/// Called once per model, at startup for the `--model` primary and right after
/// an on-demand load. The advertised context must not drift: agent CLIs read it
/// once (pi/opencode never re-read `/v1/models`) and budget their own
/// `max_tokens` against it for the rest of the session.
pub fn pinAutoContext(config: *model_mod.ModelConfig) u32 {
    // Order is load-bearing: the sizer bills THIS chunk's transient reserve and
    // the admission guard bills the same frozen value, so the width has to be
    // resolved before any context is computed from it. Pinned even under an
    // explicit `--ctx-size`, because the guard reads it on every request.
    _ = pinPrefillChunk(config);
    if (manualContext(config) > 0) return manualContext(config);
    if (config.pinned_context == 0) {
        config.pinned_context = autoContextFor(config);
    }
    return config.pinned_context;
}

/// The startup line for a context `pinAutoContext` just froze: where the number came from.
pub fn logPinnedContext(config: *const model_mod.ModelConfig, pinned: u32) void {
    if (config.glm_stream) {
        log.info("Context size: {d} tokens (mlx-stream: the prompts this load billed and the generation past them)\n", .{mlx_stream.contextLength(config)});
    } else if (manualContext(config) > 0) {
        log.info("Context size: {d} tokens (manual)\n", .{manualContext(config)});
    } else if (config.mlxStream()) {
        log.info("Context size: {d} tokens (the streaming pack's standard request; --ctx-size widens it)\n", .{getEffectiveContextLength(config)});
    } else {
        const memory_ctx = computeMemoryContext(config);
        const memory_allows = safeAutoContext(memory_ctx);
        const rope_ctx = config.contextCap();
        if (rope_ctx > 0 and pinned >= rope_ctx) {
            // The checkpoint's own maximum binds; memory had room to spare.
            log.info("Context size: {d} tokens (auto: the model's maximum; memory would allow {d}) [pinned]\n", .{ pinned, memory_allows });
        } else {
            log.info("Context size: {d} tokens (auto: {d}% of the {d}-token memory ceiling, reserving headroom) [pinned]\n", .{ pinned, auto_ctx_safety_pct, memory_ctx });
        }
    }
}

pub fn getEffectiveContextLength(config: *const model_mod.ModelConfig) u32 {
    if (config.mlxStream()) return mlx_stream.contextLength(config);
    if (manualContext(config) > 0) return manualContext(config);
    if (config.pinned_context > 0) return config.pinned_context;
    // Not pinned yet (a discovery stub that was never loaded): compute from
    // current GPU memory rather than a fixed 16K cap.
    return autoContextFor(config);
}

/// PURE budget math (no MLX/Metal calls — unit-testable): the largest context
/// length whose KV cache + per-token working set fits under the GPU
/// working-set ceiling, after subtracting what's already resident
/// (`active_mem` — model weights + any resident hot-cache KV) AND the hot
/// prefix cache's FULL byte budget (`hot_cache_reserve`).
///
/// Reserving the whole cache budget — not just its current residency — budgets
/// for STEADY STATE. Over an agentic session the hot cache fills to its cap, so
/// an auto-context computed against an empty cache (24k tokens on a 16 GB Mac,
/// live qwen3_5_moe) later collides with the filled 2 GB cache plus a
/// large cold MoE prefill and the command buffer OOMs. Subtracting it up front
/// keeps the reported context stable across the session and within the ceiling
/// once the cache is full. The 0.64 factor (two ×4/5 margins) absorbs the
/// prefill compute spike that lands on top of `active_mem`.
/// PURE: the raw largest context the budget fits — no 1024 floor, no
/// `max_pos` clamp. Kept apart from `safeContextForBudget` because
/// `resolvePrefillChunk` compares rungs against each other and the floor makes
/// a rung that fits NOTHING look identical to one that fits 1024 tokens.
/// The 0.64 factor is the same two x4/5 margins documented below.
fn contextForBudget(working_set_limit: u64, active_mem: u64, hot_cache_reserve: u64, per_tok: u64) u64 {
    if (per_tok == 0) return 0;
    const spoken_for: u64 = active_mem +| hot_cache_reserve;
    const usable: u64 = if (working_set_limit > spoken_for) working_set_limit - spoken_for else 0;
    return ((usable * 4 / 5) * 4 / 5) / per_tok;
}

pub fn safeContextForBudget(
    working_set_limit: u64,
    active_mem: u64,
    hot_cache_reserve: u64,
    per_tok: u64,
    max_pos: u32,
) u32 {
    if (per_tok == 0) return 1024;
    const max_seq: u64 = contextForBudget(working_set_limit, active_mem, hot_cache_reserve, per_tok);
    if (max_seq < 1024) return 1024;

    var result: u32 = if (max_seq > std.math.maxInt(u32)) std.math.maxInt(u32) else @intCast(max_seq);
    // Cap at model's max position embeddings.
    if (max_pos > 0) result = @min(result, max_pos);
    return result;
}

/// Compute the maximum safe context length based on the GPU working-set
/// ceiling, the model's per-token footprint, and the hot prefix cache budget.
/// Linear in seq (per_tok × seq ≤ budget) — no seq² term, MLX's fused SDPA
/// tiles over seq and never materializes [heads, seq, seq]. See
/// `safeContextForBudget` for the steady-state reservation rationale.
/// `computeMemoryContext` capped by the checkpoint's `max_position_embeddings`.
/// This is the raw "largest context that fits" figure — reported in logs and
/// `/props` for diagnostics. The context we actually SERVE is `autoContextFor`,
/// which reserves headroom below it.
pub fn computeMaxSafeContext(config: *const model_mod.ModelConfig) u32 {
    const memory_ctx = computeMemoryContext(config);
    const max_pos = config.contextCap();
    return if (max_pos > 0) @min(memory_ctx, max_pos) else memory_ctx;
}

/// The largest context this model's per-token footprint fits into RAM right now, ignoring the checkpoint's own maximum.
pub fn computeMemoryContext(config: *const model_mod.ModelConfig) u32 {
    const heads: u64 = config.num_attention_heads;
    if (heads == 0) return 16384;

    //   KV cache: the arch's own caching-layer count and K/V widths, billed at the active kv-quant width.
    const kv_bits: u64 = defaultKvBits(config);
    const per_tok: u64 = sessionBytesPerToken(config, kv_bits);

    // `total_ctx = 0` asks for the unshrunk cap: the widest forward any prompt can run.
    const chunk: u64 = @intCast(generate_mod.effectivePrefillChunk(
        config.prefillScoreHeadDim(),
        config.num_attention_heads,
        0,
        config.has_sliding_window,
        config.isMoe(),
        config.longCtxGated(),
        config.pinned_prefill_chunk,
    ));

    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);

    // Real reachable ceiling, so auto-context shrinks under external memory pressure (#64).
    const ceiling = currentGpuMemoryCeiling(config, active_mem);
    const transient = prefillTransientReserve(config, kv_bits, chunk);
    return safeContextForBudget(
        ceiling,
        active_mem,
        // The hot prefix cache fills to this cap over a session, and a prefill has to land on
        // top of whatever context we report. `CTX_SIZING_CACHE_RESERVE`, not the resolved
        // budget and not the ask: the ask starved the context, the resolved budget was a
        // one-step loop. Gated: off the gate this is the raw ask, so `/v1/models` is unchanged.
        ctxCacheShare(ctxSizingCacheReserve(config), ceiling, active_mem, transient) +| transient,
        per_tok,
        // 0 = do not clamp to the checkpoint's max here; the caller applies it after the margin.
        0,
    );
}

/// The cache reserve the context sizer bills. Gated: the ask-independent constant. Ungated:
/// the raw `--prefix-cache-mem`. Both load-time wrappers must pass the same value.
pub fn ctxSizingCacheReserve(config: *const model_mod.ModelConfig) u64 {
    if (http_server.launch().prefix_cache_capacity == 0) return 0;
    return if (config.longCtxGated()) CTX_SIZING_CACHE_RESERVE else legacyPrefixCacheAsk();
}

// ── tests ──

test "safeAutoContext leaves headroom below the memory ceiling and rounds to 1024" {
    // The raw ceiling is the largest context that FITS. Running at exactly that
    // leaves nothing for the prefix cache to grow into, a second model, or
    // another app — so admit only `auto_ctx_safety_pct` of it.
    try testing.expectEqual(@as(u32, 79872), safeAutoContext(94729)); // 85% = 80519 -> 1024-floor
    try testing.expectEqual(@as(u32, 27648), safeAutoContext(32768)); // 85% = 27852 -> 1024-floor
    // Never rounds down to zero on a tiny ceiling.
    try testing.expect(safeAutoContext(1000) > 0);
    try testing.expect(safeAutoContext(1) > 0);
    // Monotonic, and always strictly inside the ceiling for real-sized contexts.
    var prev: u32 = 0;
    for ([_]u32{ 4096, 8192, 16384, 32768, 65536, 94729, 262144 }) |raw| {
        const got = safeAutoContext(raw);
        try testing.expect(got <= raw);
        try testing.expect(got >= prev);
        prev = got;
    }
}

test "autoContextFor: the safety margin applies to MEMORY, never to the model's own max" {
    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;
    http_server.serverState().server_config.max_context_size = 0;
    // The OS reserve alone eats a 7 GB CI runner's free RAM; it is not what is on trial.
    const orig_reserve = system_vars.os_reserve_override;
    defer system_vars.os_reserve_override = orig_reserve;
    system_vars.os_reserve_override = 0;

    // A tiny model whose `max_position_embeddings` is far below anything memory
    // could constrain: it must get its FULL declared context, un-margined.
    // (Regression: applying 85% AFTER the model-max clamp shaved 15% off a
    // 131,072-token checkpoint that fits in RAM with room to spare.)
    var small = model_mod.ModelConfig{};
    small.max_position_embeddings = 4096;
    small.num_attention_heads = 8;
    small.num_hidden_layers = 4;
    small.num_key_value_heads = 2;
    small.head_dim = 64;
    small.hidden_size = 512;
    small.intermediate_size = 1024;
    try testing.expectEqual(@as(u32, 4096), autoContextFor(&small));

    // No declared max: fall back to the margined memory ceiling.
    var unbounded = small;
    unbounded.max_position_embeddings = 0;
    const got = autoContextFor(&unbounded);
    try testing.expect(got > 0);
    try testing.expectEqual(safeAutoContext(computeMemoryContext(&unbounded)), got);
}

test "getEffectiveContextLength returns the PINNED value, not a fresh memory reading" {
    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;
    http_server.serverState().server_config.max_context_size = 0; // auto

    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;

    // Once pinned, the advertised context must NOT move — clients budget
    // against it, and a fresh memory reading drifts with system load.
    config.pinned_context = 12345;
    try testing.expectEqual(@as(u32, 12345), getEffectiveContextLength(&config));
    try testing.expectEqual(@as(u32, 12345), getEffectiveContextLength(&config));

    // ...but an explicit --ctx-size still wins.
    http_server.serverState().server_config.max_context_size = 4096;
    try testing.expectEqual(@as(u32, 4096), getEffectiveContextLength(&config));
}

test "pinAutoContext is idempotent and a no-op under --ctx-size" {
    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;

    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;

    http_server.serverState().server_config.max_context_size = 0;
    const first = pinAutoContext(&config);
    try testing.expect(first > 0);
    try testing.expectEqual(first, config.pinned_context);
    // Second call must not re-read memory and shift the value.
    try testing.expectEqual(first, pinAutoContext(&config));

    // Explicit --ctx-size: never overwrite, always report the override.
    var manual = model_mod.ModelConfig{};
    manual.max_position_embeddings = 131072;
    http_server.serverState().server_config.max_context_size = 8192;
    try testing.expectEqual(@as(u32, 8192), pinAutoContext(&manual));
    try testing.expectEqual(@as(u32, 0), manual.pinned_context);
}

test "a GLM-5.3 request of 16,387 prompt tokens asking max_tokens 131,072 keeps all of it: the arch's maxOutput, not the plugin's 8,192" {
    if (comptime !mlx_stream.built) return error.SkipZigTest;
    // The pack's parse records its arch's maxOutput (GLM-5.3's default, 131,072) beside its billed context.
    mlx_stream.noteMaxOutput("/packs/glm-5.3-clamp", 131072);
    var config: model_mod.ModelConfig = .{ .model_type = "glm_moe_dsa", .pack_dir = "/packs/glm-5.3-clamp", .glm_stream = true, .ctx_override = 16448 };
    try testing.expectEqual(@as(u32, 16448 + 131072), getEffectiveContextLength(&config));
    try testing.expectEqual(@as(u32, 131072), clampMaxTokens(131072, 16387, getEffectiveContextLength(&config)));
}

test "getEffectiveContextLength uses ctx_size override" {
    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;

    http_server.serverState().server_config.max_context_size = 4096;
    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 32768;
    try testing.expectEqual(@as(u32, 4096), getEffectiveContextLength(&config));
}

test "getEffectiveContextLength computes safe default from GPU memory" {
    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;

    http_server.serverState().server_config.max_context_size = 0;
    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;
    // Should compute a value based on GPU memory, not hardcoded 16K
    const computed = getEffectiveContextLength(&config);
    try testing.expect(computed > 0);
    try testing.expect(computed <= config.max_position_embeddings);

    // Explicit --ctx-size overrides the computed default
    http_server.serverState().server_config.max_context_size = 32768;
    try testing.expectEqual(@as(u32, 32768), getEffectiveContextLength(&config));
}

test "safeContextForBudget reserves the hot-cache budget (2026-06-19 OOM regression)" {
    const GB: u64 = 1 << 30;
    // qwen3_5_moe footprint that OOM'd a 16 GB Mac: 32 layers, 4 kv heads, head_dim 256.
    const kv_per_tok: u64 = 32 * 2 * 4 * 256 * 2; // 131072
    const work_per_tok: u64 = 8 * 9216 * 2; //        147456
    const per_tok: u64 = kv_per_tok + work_per_tok;

    // Mid-session: ~6 GB resident (model + a partly-filled hot cache), 2 GB cache cap.
    const ws: u64 = 12 * GB; // ~Metal recommended working set on a 16 GB Mac
    const active: u64 = 6 * GB;
    const cache: u64 = 2 * GB;

    const with_reserve = safeContextForBudget(ws, active, cache, per_tok, 0);
    const without_reserve = safeContextForBudget(ws, active, 0, per_tok, 0);

    // The whole point of the fix: subtracting the hot-cache budget shrinks the
    // reported context so it still fits once the cache fills (it didn't before
    // — auto-ctx was computed against an empty cache and later collided with it).
    try testing.expect(with_reserve < without_reserve);
    // Still usable, never floored to nothing.
    try testing.expect(with_reserve > 1024);
}

test "auto-context tightens under external memory pressure vs the pre-fix static ceiling (#64)" {
    const GB: u64 = 1 << 30;
    // qwen3_5_moe 35B-A3B footprint (per-token KV + working envelope).
    const per_tok: u64 = 328 * 1024;
    const static_max: u64 = 115 * GB;
    const active: u64 = 26 * GB;
    const mlx_cache: u64 = 4 * GB;
    const hot_cache_reserve: u64 = 2 * GB;
    const max_pos: u32 = 262144;

    // Pre-fix behaviour: budget straight against the static device max, blind to
    // free RAM — the exact code path that OOM'd #64.
    const prefix_ctx = safeContextForBudget(static_max, active, hot_cache_reserve, per_tok, max_pos);

    // Post-fix, idle (~84 GB free): ceiling ≈ static max ⇒ context essentially unchanged.
    const idle_ceiling = physicalMemoryCeiling(static_max, active + mlx_cache, 84 * GB);
    const idle_ctx = safeContextForBudget(idle_ceiling, active, hot_cache_reserve, per_tok, max_pos);
    try testing.expect(idle_ctx >= prefix_ctx * 9 / 10); // within 10% of the old generous value

    // Post-fix, docker holds ~45 GB (~35 GB free): ceiling collapses, so the safe
    // context is far smaller than the pre-fix static-max computation would allow.
    const busy_ceiling = physicalMemoryCeiling(static_max, active + mlx_cache, 35 * GB);
    const busy_ctx = safeContextForBudget(busy_ceiling, active, hot_cache_reserve, per_tok, max_pos);
    try testing.expect(busy_ctx < prefix_ctx / 2);
    try testing.expect(busy_ctx > 1024); // still usable, never floored to nothing
}

test "safeContextForBudget floors when memory is exhausted and caps at max_pos" {
    const GB: u64 = 1 << 30;
    const per_tok: u64 = 278528;
    // working set already below what's resident → no room → minimum floor.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(4 * GB, 5 * GB, 0, per_tok, 0));
    // active + reserve exceeds the ceiling → floor.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(8 * GB, 6 * GB, 4 * GB, per_tok, 0));
    // Plenty of memory, but capped at the model's max position embeddings.
    try testing.expectEqual(@as(u32, 4096), safeContextForBudget(256 * GB, 0, 0, per_tok, 4096));
    // Degenerate per_tok never divides by zero.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(8 * GB, 0, 0, 0, 0));
    // A budget that fits a few tokens gets the same floor as one that fits none.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(6 * GB + 100 * per_tok, 6 * GB, 0, per_tok, 0));
}

test "context sizer: the prompt-cache reserve never takes more than half the room beside the weights" {
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    const per_tok: u64 = 28 * 1024;
    const transient: u64 = 300 * MiB;
    const reserve: u64 = 2048 * MiB;

    // 16 GB Mac, 8 GB pack: 1.8 GB beside the weights. A full 2 GB reserve left the context 870 tokens.
    const small = resolvedContextForLoad(0, 0, 10_200 * MiB, 8_100 * MiB, reserve, transient, per_tok, 262_144);
    try t.expect(small >= 16_384);

    // 128 GB Mac: the room dwarfs the reserve, which is billed in full as before.
    const big = resolvedContextForLoad(0, 0, 100_000 * MiB, 60_000 * MiB, reserve, transient, per_tok, 0);
    try t.expectEqual(autoContextFrom(safeContextForBudget(100_000 * MiB, 60_000 * MiB, reserve +| transient, per_tok, 0), 0), big);
}

test "the advertised context does not move with the cache ask" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Sizing the context against the ask collapsed it (60GB ask -> 870 tokens); sizing against the
    // resolved budget was a one-step loop. A constant reserve closes both.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const live_ceiling: u64 = 109_395 * MiB;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);
    const transient: u64 = prefillTransientReserve(&cfg, kv_bits, 4096);

    // `max_pos` is 0: the checkpoint cap is applied by the caller after the margin.
    const memory_ctx = safeContextForBudget(live_ceiling, active, CTX_SIZING_CACHE_RESERVE +| transient, per_tok, 0);
    try t.expect(memory_ctx > 900_000);

    // The two defect spellings.
    const vs_ask = safeContextForBudget(live_ceiling, active, 60 * 1024 * MiB +| transient, per_tok, 0);
    const vs_resolved = safeContextForBudget(live_ceiling, active, 48_673 * MiB +| transient, per_tok, 0);
    try t.expect(vs_ask <= 1024);
    try t.expect(vs_resolved <= 1024);

    // The pair agrees at the advertised number: both sides go through `autoContextFrom`.
    const advertised = autoContextFrom(memory_ctx, cfg.contextCap());
    const clamp_ctx = resolvedContextForLoad(0, 0, live_ceiling, active, CTX_SIZING_CACHE_RESERVE, transient, per_tok, cfg.contextCap());
    try t.expectEqual(advertised, clamp_ctx);
}

test "ctxSizingCacheReserve: the advertised context is unchanged on every other arch" {
    // The constant changes the advertised `context_length` at every ask except the 2 GiB default,
    // and an agent CLI budgets against that number for a whole session.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const saved = http_server.launch().prefix_cache_mem_bytes;
    defer http_server.launch().prefix_cache_mem_bytes = saved;

    var q4 = longCtxTestConfig(); // declares qwen4_exp
    var other = longCtxTestConfig();
    other.model_type = "qwen3_5_moe";
    other.legacy.caps.long_ctx = std.mem.eql(u8, other.model_type, "qwen4_exp");

    for ([_]u64{ 0, 2 * GiB, 10 * GiB, 60 * GiB }) |ask| {
        http_server.launch().prefix_cache_mem_bytes = ask;
        try t.expectEqual(CTX_SIZING_CACHE_RESERVE, ctxSizingCacheReserve(&q4));
        try t.expectEqual(ask, ctxSizingCacheReserve(&other));
    }

    // At the default ask the two agree: a default boot is unchanged on every arch.
    http_server.launch().prefix_cache_mem_bytes = CTX_SIZING_CACHE_RESERVE;
    try t.expectEqual(ctxSizingCacheReserve(&q4), ctxSizingCacheReserve(&other));

    // The reserve really moves the advertised context on the ungated arch.
    const kv_bits: u64 = 8;
    const per_tok = kvBytesPerTokenAtBits(other.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&other);
    const ceiling: u64 = 120 * GiB;
    const active: u64 = 69 * GiB;
    const transient = prefillTransientReserve(&other, kv_bits, 4096);
    const small = safeContextForBudget(ceiling, active, 2 * GiB +| transient, per_tok, 0);
    const big = safeContextForBudget(ceiling, active, 40 * GiB +| transient, per_tok, 0);
    try t.expect(big < small);
}
