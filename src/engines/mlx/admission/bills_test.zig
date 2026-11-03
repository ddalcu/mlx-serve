//! test configs (qwen4, glm5) + bill tests

const std = @import("std");
const transformer_mod = @import("../transformer.zig");
const sparse_score = @import("../../../sdk/blocks/attention/sparse_block/score.zig");
const generate_mod = @import("../generate.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const prefix_cache_mod = @import("../cache/prefix_cache.zig");
const scheduler_mod = @import("../scheduler.zig");
const ane_mod = @import("../../../sdk/ane/offload.zig");
const http_server = @import("../../../host/http/server.zig");
const memory_ceiling = @import("../../../host/registry/memory_ceiling.zig");

const resolvedPrefixCacheMem = @import("hot_cache_plan.zig").resolvedPrefixCacheMem;
const omittedMaxTokensDefault = @import("../../../host/api/sampling.zig").omittedMaxTokensDefault;
const getEffectiveContextLength = @import("context.zig").getEffectiveContextLength;
const safeContextForBudget = @import("context.zig").safeContextForBudget;
const defaultKvBits = @import("bills.zig").defaultKvBits;
const prefillTransientReserve = @import("bills.zig").prefillTransientReserve;
const prefillTransientReserveAtKv = @import("bills.zig").prefillTransientReserveAtKv;
const qsaMaskBytes = @import("../../../plugins/qwen4_exp/bill.zig").qsaMaskBytes;
const aneGateHeadroom = @import("bills.zig").aneGateHeadroom;
const PREFILL_CHUNK_LADDER = @import("prefill_chunk.zig").PREFILL_CHUNK_LADDER;
const PREFILL_RESERVE_BUDGET_SHARE = @import("prefill_chunk.zig").PREFILL_RESERVE_BUDGET_SHARE;
const HOT_CACHE_FLOOR_BYTES = @import("prefill_chunk.zig").HOT_CACHE_FLOOR_BYTES;
const resolvePrefillChunk = @import("prefill_chunk.zig").resolvePrefillChunk;
const explicitPrefillChunk = @import("prefill_chunk.zig").explicitPrefillChunk;
const billedPrefillChunk = @import("prefill_chunk.zig").billedPrefillChunk;
const clampedPrefixCacheMem = @import("hot_cache_plan.zig").clampedPrefixCacheMem;
const planHotCache = @import("hot_cache_plan.zig").planHotCache;
const slotFixedKvBytes = @import("bills.zig").slotFixedKvBytes;
const kvBytesPerTokenAtBits = @import("bills.zig").kvBytesPerTokenAtBits;
const PREFILL_RUNTIME_FLOOR_BYTES = @import("bills.zig").PREFILL_RUNTIME_FLOOR_BYTES;
const WarmPrefix = @import("bills.zig").WarmPrefix;
const attnLayersPerEvalWindow = @import("bills.zig").attnLayersPerEvalWindow;
const PrefillRequestTerms = @import("bills.zig").PrefillRequestTerms;
const prefillMemoryNeeded = @import("bills.zig").prefillMemoryNeeded;
const prefillStreamBytesPerToken = @import("bills.zig").prefillStreamBytesPerToken;
const prefillStreamBytesPerTokenOn = @import("bills.zig").prefillStreamBytesPerTokenOn;
const prefillDequantWeightBytes = @import("bills.zig").prefillDequantWeightBytes;
const prefillFfnWidth = @import("bills.zig").prefillFfnWidth;
const statePerTokenBilled = @import("bills.zig").statePerTokenBilled;
const reservedCacheTokens = @import("bills.zig").reservedCacheTokens;
const retainedSsmCheckpointBytes = @import("bills.zig").retainedSsmCheckpointBytes;
const mtpHeadKvBytesPerToken = @import("bills.zig").mtpHeadKvBytesPerToken;
const mtpHeadBilled = @import("bills.zig").mtpHeadBilled;
const prefillRequestTerms = @import("bills.zig").prefillRequestTerms;
const prefillNeededAtChunk = @import("prefill_chunk.zig").prefillNeededAtChunk;
const prefillAdmissionBill = @import("admit.zig").prefillAdmissionBill;
const clampMaxTokens = @import("../../../host/api/sampling.zig").clampMaxTokens;
const attn256 = @import("../../../sdk/blocks/attention/attn256.zig");
const prefill_dq_gemm = @import("../../../sdk/blocks/linear/prefill_dq_gemm.zig");

/// A stand-in for the deployed long-context pack: qwen3_5-shaped GDN+MoE trunk with a QSA
/// indexer. Declares `qwen4_exp`: every long-context mechanism is gated on it.
pub fn longCtxTestConfig() model_mod.ModelConfig {
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "qwen4_exp"; // arch-scan-exempt: test fixture, not a gate
    cfg.legacy.caps.long_ctx = true;
    cfg.num_hidden_layers = 40;
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 256;
    cfg.hidden_size = 2560;
    cfg.intermediate_size = 9216;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    cfg.max_position_embeddings = 1048576;
    cfg.indexer_budget = 2048;
    cfg.indexer_n_heads = 4;
    cfg.indexer_head_dim = 128;
    cfg.indexer_compress_ratio = 4;
    return cfg;
}

/// The deployed long-context shape at the live per-token widths: 12 caching layers, hd 256,
/// 2 KV heads (13,056 B/tok at 8-bit); a 128-wide QSA indexer at ratio 4.
pub fn qsaScoreFusedOffGuard() struct {
    pub fn deinit(_: @This()) void {
        sparse_score.qsa_score_fused_override = null;
    }
} {
    sparse_score.qsa_score_fused_override = false;
    return .{};
}

pub fn qwen4RequestTestConfig() model_mod.ModelConfig {
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "qwen4_exp"; // arch-scan-exempt: test fixture, not a gate
    cfg.legacy.caps.long_ctx = true;
    cfg.num_hidden_layers = 48;
    cfg.full_attention_interval = 4;
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 256;
    // No sliding-band layers: the field defaults to true, which routes a different chunk policy.
    cfg.has_sliding_window = false;
    cfg.hidden_size = 2560;
    cfg.moe_intermediate_size = 1024;
    cfg.num_experts_per_tok = 8;
    cfg.num_experts = 512;
    cfg.quant_bits = 4;
    cfg.max_position_embeddings = 1048576;
    cfg.indexer_budget = 2048;
    cfg.indexer_n_heads = 4;
    cfg.indexer_head_dim = 128;
    cfg.indexer_compress_ratio = 4;
    cfg.pinned_prefill_chunk = 1024; // what load-time sizing gives at 1M ctx
    return cfg;
}

/// The geometry the served pack actually ships (`Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`):
/// a narrower expert with a shared one beside it, and the GatedDeltaNet head counts
/// `qwen4RequestTestConfig` leaves at zero. Both chunk-scaled terms of the bill read them, so
/// the fixture's step is not the deployed one.
pub fn qwen4DeployedTestConfig() model_mod.ModelConfig {
    var cfg = qwen4RequestTestConfig();
    cfg.moe_intermediate_size = 640;
    cfg.num_experts_per_tok = 10;
    cfg.shared_expert_intermediate_size = 640;
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = 48;
    cfg.linear_key_head_dim = 128;
    cfg.linear_value_head_dim = 128;
    cfg.linear_conv_kernel_dim = 4;
    cfg.quant_group_size = 64;
    cfg.quant_mode = .affine;
    return cfg;
}

/// GLM-5.3-Flash pack geometry after `model.zig` parses its nested text config.
pub fn glm5TestConfig() model_mod.ModelConfig {
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "glm5_next"; // arch-scan-exempt: test fixture, not a gate
    cfg.num_hidden_layers = 45;
    cfg.full_attention_interval = 4; // 11 DSA at idx%4==3, 34 KDA
    cfg.num_attention_heads = 64;
    cfg.num_key_value_heads = 64;
    cfg.head_dim = 256;
    cfg.hidden_size = 4096;
    cfg.dsa_kv_lora_rank = 512;
    cfg.dsa_head_dim = 256;
    cfg.dsa_index_topk = 2048;
    cfg.dsa_index_kpool = 4;
    cfg.dsa_index_head_dim = 128;
    cfg.dsa_index_heads = 32;
    cfg.linear_num_key_heads = 64;
    cfg.linear_key_head_dim = 128;
    cfg.linear_num_value_heads = 64;
    cfg.linear_value_head_dim = 128;
    cfg.linear_conv_kernel_dim = 4;
    cfg.num_experts = 288;
    cfg.num_experts_per_tok = 8;
    cfg.moe_intermediate_size = 2048;
    cfg.shared_expert_intermediate_size = 2048;
    cfg.intermediate_size = 12288;
    cfg.intermediate_size_declared = true;
    cfg.vocab_size = 154880;
    cfg.ctx_override = 147456;
    return cfg;
}

// ── tests ──

test "planHotCache: the budget is MONOTONE in the ask" {
    // The defect: a bigger ask stepped the chunk down, shrank the reserve and bought MORE cache
    // (10GB -> 3873 MB, 24GB -> 5703 MB, same box).
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const cfg = longCtxTestConfig();
    const kv_bits: u64 = 8;
    const ctx_tokens: u64 = 262144;
    const weights: u64 = 69 * GiB;
    const ceiling: u64 = weights + 28_909 * (1 << 20);

    var prev: u64 = 0;
    var ask: u64 = GiB;
    while (ask <= 64 * GiB) : (ask += GiB) {
        const plan = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, 0, ask, 0);
        try t.expect(plan.budget >= prev);
        try t.expect(plan.budget <= @max((ceiling -| weights) -| plan.ctx_kv -| plan.reserve, 1));
        prev = plan.budget;
    }

    // The two live asks now agree: the sizer no longer sees the ask.
    const at10 = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, 0, 10 * GiB, 0);
    const at24 = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, 0, 24 * GiB, 0);
    try t.expect(at24.budget >= at10.budget);
    try t.expectEqual(at10.chunk, at24.chunk);
}

test "planHotCache: an out-of-the-box boot gets the whole remaining headroom, never the 1-byte floor" {
    // One boot with no flags: `requested == 0` used to hand the sizer a zero reserve, so it took
    // a wide rung and the clamp returned its 1-byte floor.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const cfg = longCtxTestConfig();
    const kv_bits: u64 = 8;
    const weights: u64 = 69 * GiB;
    const ceiling: u64 = weights + 28_909 * (1 << 20);
    const ctx_tokens: u64 = 262144;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) + statePerTokenBilled(&cfg);
    const sizer_ctx_kv: u64 = per_tok * ctx_tokens + cfg.qsaRingBytes();

    const boot = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, sizer_ctx_kv, 0, 0);
    // Everything left over: the cache is the residual claimant.
    try t.expectEqual(@max((ceiling -| weights) -| boot.ctx_kv -| boot.reserve, 1), boot.budget);
    // A cache worth having, or the ladder bottomed out and the box is too small.
    try t.expect(boot.budget >= HOT_CACHE_FLOOR_BYTES or
        boot.chunk == PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1]);

    // The rung is the widest that clears both ask-independent bars.
    const serving: u64 = ceiling - weights;
    const cap: u64 = @min(serving / PREFILL_RESERVE_BUDGET_SHARE, (serving -| sizer_ctx_kv) -| HOT_CACHE_FLOOR_BYTES);
    for (PREFILL_CHUNK_LADDER) |c| {
        if (c <= boot.chunk) break;
        try t.expect(prefillTransientReserve(&cfg, kv_bits, c) > cap);
    }

    // An explicit ask can only take a slice of the same headroom.
    const huge = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, sizer_ctx_kv, 1024 * GiB, 0);
    try t.expectEqual(boot.budget, huge.budget);
    try t.expect(planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, sizer_ctx_kv, 4 * GiB, 0).budget <= boot.budget);
}

test "the clamp's live inversion, as arithmetic" {
    // The reported boot reduced to `clampedPrefixCacheMem`: ceiling - weights = 28,909 MiB, ctx KV 20,736 MiB.
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    const serving: u64 = 28_909 * MiB;
    const ctx_kv: u64 = 20_736 * MiB;
    const reserve_1024: u64 = 4_300 * MiB;
    const reserve_512: u64 = 2_470 * MiB;

    const ask10 = clampedPrefixCacheMem(10 * GiB, serving, 0, ctx_kv, reserve_1024);
    const ask24 = clampedPrefixCacheMem(24 * GiB, serving, 0, ctx_kv, reserve_512);
    try t.expectEqual(3_873 * MiB, ask10);
    try t.expectEqual(5_703 * MiB, ask24);
    try t.expect(ask10 < ask24); // the inversion, in the numbers that shipped

    // One ask-independent reserve: the clamp is `@min(ask, headroom)` over a headroom no ask can move.
    var prev: u64 = 0;
    var ask: u64 = GiB;
    while (ask <= 64 * GiB) : (ask += GiB) {
        const got = clampedPrefixCacheMem(ask, serving, 0, ctx_kv, reserve_1024);
        try t.expect(got >= prev);
        try t.expect(got <= @max((serving -| ctx_kv) -| reserve_1024, 1));
        prev = got;
    }
    try t.expectEqual(3_873 * MiB, clampedPrefixCacheMem(0, serving, 0, ctx_kv, reserve_1024));
}

test "an explicit --prefill-chunk is the chunk that gets BILLED" {
    // `--prefill-chunk 4096` used to run a 4096-token forward against a 512-token reserve.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    var cfg = longCtxTestConfig();
    cfg.head_dim = 128; // fused dim: boundedPrefillChunk returns its base untouched
    const kv_bits: u64 = 8;
    const weights: u64 = 69 * GiB;
    // Squeezed so the machine sizer would step below the flag.
    const ceiling: u64 = weights + 6 * GiB;

    generate_mod.prefillCli().override = 4096;
    generate_mod.prefillCli().explicit = true;
    defer {
        generate_mod.prefillCli().override = 8192;
        generate_mod.prefillCli().explicit = false;
    }
    try t.expectEqual(@as(u32, 4096), explicitPrefillChunk());
    const sized = resolvePrefillChunk(&cfg, kv_bits, ceiling, weights, 0, 0);
    try t.expect(sized < 4096); // the machine would have narrowed it

    const billed = billedPrefillChunk(&cfg, kv_bits, ceiling, weights, 0, 0, explicitPrefillChunk());
    try t.expectEqual(@as(u32, 4096), billed);

    // The other reader agrees.
    try t.expectEqual(@as(usize, 4096), generate_mod.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        40_000,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        billed,
    ));

    const plan = planHotCache(&cfg, kv_bits, ceiling, weights, 8192, 0, 2 * GiB, billed);
    try t.expectEqual(@as(u32, 4096), plan.chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, 4096), plan.reserve);
    try t.expect(plan.reserve > prefillTransientReserve(&cfg, kv_bits, sized));
}

test "qwen4RequestTestConfig does not leak qsa_score_fused_override" {
    const saved = sparse_score.qsa_score_fused_override;
    defer sparse_score.qsa_score_fused_override = saved;
    sparse_score.qsa_score_fused_override = null;
    _ = qwen4RequestTestConfig();
    try std.testing.expectEqual(@as(?bool, null), sparse_score.qsa_score_fused_override);
}

test "a warm append is billed the rows it ALLOCATES, not the rows it already holds" {
    // Bar: a checked-out restore credits the rows it holds even when the append outgrows the
    // buffer; the grow bills the new capacity plus one eval window of old buffers.
    const t = std.testing;
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    const mb: u64 = 1024 * 1024;
    const cfg = qwen4ExpLive364kConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 448_604;
    const matched: u64 = 448_531;
    const chunk: u64 = 512;
    const max_tokens: u32 = @intCast(getEffectiveContextLength(&cfg) - seq);
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg)), seq);

    // Pessimal reading of the entry: capacity exactly the rows it matched, so the append DOES
    // grow every attention layer.
    const grows = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = matched, .will_donate = true };
    const g = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, grows);
    try t.expectEqual(matched *| kv_per_tok, g.shared_resident_bytes);
    try t.expectEqual((reserved - matched) *| kv_per_tok, g.reserved_kv_bytes +| (seq - matched) *| kv_per_tok);
    // ...plus one eval window of old buffers: qwen4_exp caches 12 of 48 layers, one per window.
    try t.expectEqual(@as(u32, 1), attnLayersPerEvalWindow(&cfg, transformer_mod.Transformer.MOE_EVAL_EVERY_N_LAYERS));
    try t.expectEqual(@as(u64, 465), g.grow_coexist_bytes / mb);
    // The bill at the narrowest and a wider width.
    try t.expectEqual(@as(u64, 4_989), prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, grows) / mb);
    try t.expectEqual(@as(u64, 6_480), prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, 2048, grows) / mb);

    // A buffer the previous turn sized above this prompt grows nothing, so no window at all.
    const fits = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = 450_048, .will_donate = true };
    const f = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, fits);
    try t.expectEqual(@as(u64, 0), f.grow_coexist_bytes);
    try t.expectEqual(@as(u64, 4_407), prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, fits) / mb);

    // Byte-identical arms: a SHARED restore is copied whole by the first append, so it credits
    // nothing and bills no window either; a COLD prompt has nothing to credit.
    const shared = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = matched, .will_donate = false };
    const sh = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, shared);
    try t.expectEqual(@as(u64, 0), sh.shared_resident_bytes);
    try t.expectEqual(@as(u64, 0), sh.grow_coexist_bytes);
    const cold = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{});
    try t.expectEqual(@as(u64, 0), cold.shared_resident_bytes);
    try t.expectEqual(@as(u64, 0), cold.grow_coexist_bytes);
}

test "kvBytesPerTokenAtBits: the KV bill follows the CONFIGURED quant width" {
    const t = std.testing;
    // qwen3_5 27B dense: 16 caching layers x 4 kv heads x 256 head_dim x (K+V) x 2 B.
    const dense: u64 = 16 * 4 * 256 * 2 * 2; // 65536

    // Dense/off bills fp16 verbatim — no change for anyone not using --kv-quant.
    try t.expectEqual(dense, kvBytesPerTokenAtBits(dense, 16));
    try t.expectEqual(dense, kvBytesPerTokenAtBits(dense, 32));

    // Quantized widths carry the group scale/bias overhead (+0.5 bit/elem), the
    // SAME expression prefillMemoryNeeded bills — the sizer and the admission
    // guard must not disagree about what one token costs.
    try t.expectEqual(dense * 9 / 32, kvBytesPerTokenAtBits(dense, 4)); // 18432
    try t.expectEqual(dense * 17 / 32, kvBytesPerTokenAtBits(dense, 8));
    // 4-bit is what makes the 16 GB tier reachable: under a third of dense.
    try t.expect(kvBytesPerTokenAtBits(dense, 4) * 3 < dense);
}

test "auto-context on a 16 GB profile: activations are a chunk reserve, not a per-token multiplier" {
    const t = std.testing;
    // Qwen3.8-27B on a 16 GB Mac: ~10.6 GB reachable Metal working set, an
    // 8.6 GB iQ-MLX pack resident, --kv-quant 4, and the small prefill chunk
    // the machine can actually afford.
    const ceiling: u64 = 10_600 * 1000 * 1000;
    const weights: u64 = 8_600 * 1000 * 1000;
    const dense_kv: u64 = 16 * 4 * 256 * 2 * 2;
    const per_tok = kvBytesPerTokenAtBits(dense_kv, 4);

    var cfg = model_mod.ModelConfig{ .model_type = "qwen3_5" };
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 4;
    cfg.head_dim = 256;
    cfg.hidden_size = 5120;
    cfg.intermediate_size = 17408;
    cfg.intermediate_size_declared = true;
    // The real checkpoint's hybrid geometry: 64 layers, every 4th full
    // attention, 48 GatedDeltaNet layers whose chunk-wide q/k/v streams are
    // two thirds of this model's measured prefill transient.
    cfg.num_hidden_layers = 64;
    cfg.full_attention_interval = 4;
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = 48;

    const reserve = prefillTransientReserve(&cfg, 4, 512);
    const got = safeContextForBudget(ceiling, weights, reserve, per_tok, 262144);

    // The pre-fix formula billed 8 x max(hidden, ffn) x 2 = 272 KB PER TOKEN on
    // top of a 64 KB/token dense KV — 344 KB/token, which caps this machine at
    // under 4k tokens no matter what the weights cost.
    const old_per_tok: u64 = dense_kv + 8 * 17408 * 2;
    const old = safeContextForBudget(ceiling, weights, 0, old_per_tok, 262144);
    try t.expect(old < 4096);

    // Corrected, the same machine holds a real working context. The band is
    // what the RESERVE is worth, and the reserve is measured: this checkpoint
    // peaks 1.06 GB above steady state at chunk 512 (M4 Max), so
    // a sizer that reported the ~51k tokens the first cut of this fix did was
    // spending memory the prefill needs. Halve the chunk and it comes back.
    try t.expect(got > 15_000);
    try t.expect(got < 40_000);

    // The reserve is a CONSTANT of the chunk, so halving the chunk buys context
    // back rather than being invisible.
    const narrow = safeContextForBudget(ceiling, weights, prefillTransientReserve(&cfg, 4, 256), per_tok, 262144);
    try t.expect(narrow > got);
}

test "aneGateHeadroom: reserves a usable context and scales with the chunk" {
    const t = std.testing;
    var cfg = model_mod.ModelConfig{ .model_type = "qwen3_5" };
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 4;
    cfg.head_dim = 256;
    cfg.hidden_size = 5120;
    cfg.intermediate_size = 17408;
    cfg.intermediate_size_declared = true;
    cfg.num_hidden_layers = 64;
    cfg.full_attention_interval = 4;
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = 48;
    cfg.max_position_embeddings = 262144;

    const wide = aneGateHeadroom(&cfg, 8192);
    const narrow = aneGateHeadroom(&cfg, 1024);
    const gib = 1024 * 1024 * 1024;

    // Dominated by the chunk envelope, which is the whole point: the flat
    // 12 GB it replaces refused a measured +38% prefill at chunk 1024 while
    // under-reserving at chunk 8192.
    try t.expect(wide > 8 * gib);
    try t.expect(narrow < wide * 3 / 4);

    // The KV reserve is exactly MIN_CONTEXT_TOKENS worth — the guarantee that
    // an admitted offload leaves a usable context behind.
    try t.expect(narrow == ane_mod.GATE_BASELINE_BYTES +
        kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), defaultKvBits(&cfg)) * ane_mod.MIN_CONTEXT_TOKENS +
        resolvedPrefixCacheMem() +
        prefillTransientReserve(&cfg, defaultKvBits(&cfg), 1024));

    // A model that cannot reach the reserve context only reserves its own max.
    cfg.max_position_embeddings = 8192;
    try t.expect(aneGateHeadroom(&cfg, 1024) < narrow);
}

test "auto-context never sizes past the ceiling as weights grow" {
    const t = std.testing;
    const ceiling: u64 = 10_600 * 1000 * 1000;
    const dense_kv: u64 = 16 * 4 * 256 * 2 * 2;
    const per_tok = kvBytesPerTokenAtBits(dense_kv, 4);
    var cfg = model_mod.ModelConfig{};
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 4;
    cfg.head_dim = 256;
    cfg.hidden_size = 5120;
    cfg.intermediate_size = 17408;
    cfg.intermediate_size_declared = true;
    const reserve = prefillTransientReserve(&cfg, 4, 512);

    // Under-billing here ends in an uncatchable Metal OOM, not a 400: at every
    // weight size the reported context plus what is already spoken for has to
    // stay inside the ceiling, and the reported figure must fall as the pack
    // grows rather than staying pinned to the checkpoint's max.
    var prev: u32 = std.math.maxInt(u32);
    var weights: u64 = 2_000 * 1000 * 1000;
    while (weights < ceiling) : (weights += 1_000 * 1000 * 1000) {
        const ctx = safeContextForBudget(ceiling, weights, reserve, per_tok, 262144);
        try t.expect(weights + reserve + @as(u64, ctx) * per_tok <= ceiling or ctx == 1024);
        try t.expect(ctx <= prev);
        prev = ctx;
    }
    // Squeezed to nothing it floors rather than reporting a context that cannot exist.
    try t.expectEqual(@as(u32, 1024), safeContextForBudget(ceiling, ceiling, reserve, per_tok, 262144));
}

test "slotFixedKvBytes: a MiMo slot bills its sliding rings once, at keep + one chunk of rows" {
    const t = std.testing;
    // Layer 3 global (4 KV heads), layers 0-2 sliding (8 KV heads).
    const mimo: model_mod.ModelConfig = .{ .model_type = "mimo_v2", .num_hidden_layers = 4, .head_dim = 192, .v_head_dim = 128, .num_key_value_heads = 8, .num_global_key_value_heads = 4, .sliding_window = 128, .sliding_window_pattern = 4 };
    const rows: u64 = 128 + model_mod.ModelConfig.SLIDING_RING_SLACK + 8192;
    try t.expectEqual(3 * 8 * (192 + 128) * 2 * rows, slotFixedKvBytes(&mimo, 16, 8192));
    try t.expectEqual(kvBytesPerTokenAtBits(3 * 8 * (192 + 128) * 2, 8) * rows, slotFixedKvBytes(&mimo, 8, 8192));
    // The admission guard bills them on an ungated arch too.
    try t.expectEqual(slotFixedKvBytes(&mimo, 16, 8192), prefillRequestTerms(&mimo, 50_000, 1024, 16, 8192, .{}).slot_fixed_kv_bytes);
    // Every other arch keeps its old fixed term.
    const llama: model_mod.ModelConfig = .{ .model_type = "llama" };
    try t.expectEqual(llama.qsaRingBytes(), slotFixedKvBytes(&llama, 16, 8192));
}

test "kvBytesPerToken bills only the CACHING layers, at the arch's own K and V widths" {
    const t = std.testing;
    // Uniform arch: every layer caches, K and V are both head_dim wide.
    var dense = model_mod.ModelConfig{ .model_type = "qwen3" };
    dense.num_hidden_layers = 24;
    dense.num_key_value_heads = 8;
    dense.head_dim = 128;
    try t.expectEqual(@as(u32, 24), dense.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 24 * 8 * 256 * 2), dense.kvBytesPerToken());

    // bailing_hybrid (Ling 3.0): 6 of 24 layers are attention, and MLA stores
    // a 192-wide key against a 128-wide value. Billing it as a uniform arch
    // charged 196608 B/token against a real 61440 — a 3.2x over-bill that
    // pinned auto-context to under a third of what fits.
    var ling = model_mod.ModelConfig{ .model_type = "bailing_hybrid" };
    ling.num_hidden_layers = 24;
    ling.num_attention_heads = 16;
    ling.num_key_value_heads = 16;
    ling.head_dim = 128;
    ling.full_attention_interval = 4;
    ling.mla_kv_lora_rank = 512;
    ling.mla_qk_nope_head_dim = 128;
    ling.mla_qk_rope_head_dim = 64;
    ling.mla_v_head_dim = 128;
    try t.expectEqual(@as(u32, 6), ling.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 6 * 16 * (192 + 128) * 2), ling.kvBytesPerToken());
    try t.expect(ling.kvBytesPerToken() * 3 < 24 * 2 * 16 * 128 * 2);

    // MLA caches per ATTENTION head — it decompresses the latent to all of
    // them, so there is no grouping to bill against. Ling 3.0 ships 16/16, so
    // only an asymmetric config can tell the two spellings apart, and reading
    // `num_key_value_heads` here UNDER-bills (2 of 16 heads = 8x light) —
    // which surfaces as an uncatchable Metal OOM, not a refusal.
    var mla_grouped = ling;
    mla_grouped.num_key_value_heads = 2;
    try t.expectEqual(@as(u64, 6 * 16 * (192 + 128) * 2), mla_grouped.kvBytesPerToken());
    // ...while a NON-MLA arch still bills its KV heads, grouping and all.
    var gqa = dense;
    gqa.num_attention_heads = 32;
    gqa.num_key_value_heads = 8;
    try t.expectEqual(@as(u64, 24 * 8 * 256 * 2), gqa.kvBytesPerToken());

    // A hybrid WITHOUT MLA still drops its linear layers from the bill.
    var gdn = model_mod.ModelConfig{ .model_type = "qwen3_5_moe" };
    gdn.num_hidden_layers = 40;
    gdn.num_key_value_heads = 2;
    gdn.head_dim = 256;
    gdn.full_attention_interval = 4;
    try t.expectEqual(@as(u32, 10), gdn.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 10 * 2 * 512 * 2), gdn.kvBytesPerToken());
}

test "prefillMemoryNeeded: a prompt shorter than the chunk bills the prompt-width forward, not the chunk cap" {
    // hy_v3 with 111 GB of weights resident (~4 GB headroom),
    // a 31-TOKEN prompt was rejected "needs ~6246MB" — the MLP envelope was
    // billed at the hd-128 8192 chunk cap while the actual prefill runs ONE
    // 31-token forward (~25 MB). A forward is never wider than the prompt:
    // the envelope must use min(chunk, seq). Guard-tracks-reality rule.
    const small = prefillMemoryNeeded(31, 64, 8, 327680, 128, 128, 4096, 13312, 8, 31, 31, 0, 0, .{});
    const capped = prefillMemoryNeeded(31, 64, 8, 327680, 128, 128, 4096, 13312, 8, 8192, 31, 0, 0, .{});
    try std.testing.expectEqual(small, capped);
    // Sanity: the clamped estimate for the live 31-token case sits far below
    // the ~4 GB that was actually available. It is not zero — every prefill
    // pays PREFILL_RUNTIME_FLOOR_BYTES — but a tiny prompt pays little else,
    // and the dequant-weight term never fires this far under the route's
    // minimum forward width.
    try std.testing.expect(capped < 1024 * 1024 * 1024);
}

test "prefillMemoryNeeded: quantized KV is billed at its real width, not fp16" {
    const t = std.testing;
    attn256.fused256_override = false;
    defer attn256.fused256_override = null;
    // 26B-A4B shape at 100K ctx, chunk 512 (what the bounded chunk picks there).
    // fp16: kv 30x2x100000x8x256x2 = 24.576 GB; scores 16x512x100000x2 = 1.6384 GB;
    //       mlp 3x8x512x2816x2 = 69.2 MB; x1.25 margin.
    try t.expectEqual(
        @as(u64, 32_854_507_520 + 671_088_640),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{}),
    );
    // 4-bit: kv billed at (2*4+1)/16 bytes/elem (payload + group scale/bias)
    // = 6.912 GB, plus the 0.8192 GB per-layer dense dequant transient.
    try t.expectEqual(
        @as(u64, 11_798_507_520 + 671_088_640),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 4, 512, 100_000, 0, 0, .{}),
    );
    // Direction: quantized admission must be under half the fp16 bill.
    try t.expect(prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 4, 512, 100_000, 0, 0, .{}) * 2 <
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{}));
}

test "prefillMemoryNeeded: working set is chunk-bounded — the 255K MoE prompt is admittable" {
    const t = std.testing;
    attn256.fused256_override = false;
    defer attn256.fused256_override = null;
    // The PR-#69 failure case: Qwen3.6-35B-A3B shape (whose checkpoint OMITS
    // intermediate_size, so the struct default 15360 leaks into ffn) at 262144
    // tokens with 4-bit KV. kv 6.04 GB + scores 4.29 GB + dequant 0.54 GB +
    // mlp 0.377 GB, x1.25 = ~13.1 GiB — admittable on a big Mac.
    const needed = prefillMemoryNeeded(262_144, 16, 2, 81920, 256, 256, 2048, 15360, 4, 512, 262_144, 0, 0, .{});
    try t.expectEqual(@as(u64, 14_061_404_160 + 671_088_640), needed);
    try t.expect(needed < 16 << 30);
    // The retired seq-scaled envelope billed 8 x seq x ffn x 2 working bytes
    // (~64 GB) on top of fp16 KV (~21.5 GB) -> ~107 GB and a spurious 400.
    const old_estimate: u64 = (40 * 2 * 262_144 * 2 * 256 * 2 + 8 * 262_144 * 15360 * 2) * 5 / 4;
    try t.expect(needed < old_estimate / 7);
}

test "prefillMemoryNeeded: unfused head dims bill the materialized score scratch" {
    const t = std.testing;
    attn256.fused256_override = false;
    defer attn256.fused256_override = null;
    // Same shape, head_dim 128 vs 256 (kv scaled by hdim too, so compare the
    // full bills computed by hand): hd=128 fp16 -> kv 12.288 GB + 0 scores +
    // mlp 69.2 MB, x1.25.
    try t.expectEqual(
        @as(u64, 15_446_507_520 + 671_088_640),
        prefillMemoryNeeded(100_000, 16, 8, 122880, 128, 128, 2816, 2112, 16, 512, 100_000, 0, 0, .{}),
    );
    // hd=256 adds the [heads, chunk, seq] score tensor (the guard must see
    // what the composed SDPA path actually allocates). Decomposition: the
    // hd-128 bill + the KV doubling from 128->256 (12.288 GB x1.25) + the
    // score tensor (16x512x100000x2 = 1.6384 GB x1.25).
    const with_scores = prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{});
    try t.expectEqual(@as(u64, 15_446_507_520 + 671_088_640 + 15_360_000_000 + 2_048_000_000), with_scores);
}

test "prefillMemoryNeeded: the SCORE width decides the score term, not the stored width" {
    // bailing_hybrid: MLA contracts scores over qk_head_dim 192 while storing
    // values at v_head_dim 128 and declaring head_dim 128. mlx has a fused
    // vector kernel for that pair at DECODE and falls back to the composed
    // path at prefill widths, so the [heads, chunk, seq] score tensor is real —
    // but `prefillHeadDimFused(128)` is true, so passing the stored width for
    // both billed it at ZERO. Under-billing ends in an uncatchable Metal OOM
    // instead of a 400, which is the whole reason this guard exists.
    const t = std.testing;
    const kv_per_tok: u64 = 6 * 16 * (192 + 128) * 2; // 6 caching layers of 24
    const stored: u64 = 128;
    const scored: u64 = 192;
    const honest = prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, stored, scored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{});
    const blind = prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, stored, stored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{});
    // The difference is exactly the score tensor: heads x chunk x seq x 2, x1.25.
    const score_bytes: u64 = 16 * 4096 * 32_768 * 2;
    try t.expectEqual(honest - blind, score_bytes * 5 / 4);
    try t.expect(honest > 2 * blind);
    // The two widths are independent: at fp16 (no dequant transient) the
    // stored width touches nothing the score term reads.
    try t.expectEqual(
        honest,
        prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, scored, scored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{}),
    );
}

test "prefillMemoryNeeded: fused hd-256 kernel drops the score bill, keeps KV + dequant" {
    const t = std.testing;
    // Default (msv_attn_p256 active): the composed score scratch never exists,
    // so hd 256 bills exactly the hd-256 KV + mlp — the unfused bill minus the
    // 1.6384 GB x1.25 score term from the test above.
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    try t.expectEqual(
        @as(u64, 32_854_507_520 + 671_088_640 - 2_048_000_000),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{}),
    );
    // The quantized-KV dequant transient is a denseView property, NOT a score
    // property — it must survive the fused kernel (fires on every kv-quant
    // request regardless of head_dim).
    try t.expectEqual(
        @as(u64, 11_798_507_520 + 671_088_640 - 2_048_000_000),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 4, 512, 100_000, 0, 0, .{}),
    );
}

test "resolvePrefillChunk: the sizer and the guard bill the chunk that was pinned" {
    // The whole point of pinning: `prefillTransientReserve` (sizer) and
    // `prefillMemoryNeeded` (guard) must move together with the resolved width.
    // A narrower pin has to produce a SMALLER bill for the same prompt, or the
    // step-down bought nothing; and generate must resolve to the same width or
    // the forward runs wider than the bill and the Metal OOM is uncatchable.
    const t = std.testing;
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 32;
    cfg.num_attention_heads = 32;
    cfg.num_key_value_heads = 8;
    cfg.head_dim = 128;
    cfg.hidden_size = 4096;
    cfg.intermediate_size = 14336;
    cfg.intermediate_size_declared = true;

    const wide = prefillTransientReserve(&cfg, 16, 8192);
    const narrow = prefillTransientReserve(&cfg, 16, 512);
    try t.expect(narrow * 2 < wide);

    cfg.pinned_prefill_chunk = 512;
    try t.expectEqual(@as(usize, 512), generate_mod.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        40_000,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
    // Unpinned keeps the launch width — a model the sizer never ran on must
    // not silently get a narrower forward than the guard bills.
    const unpinned_prefill_chunk: usize = 0;
    try t.expectEqual(@as(usize, 8192), generate_mod.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        40_000,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        unpinned_prefill_chunk,
    ));
    // An explicit --prefill-chunk outranks the pin in BOTH directions.
    generate_mod.prefillCli().override = 2048;
    generate_mod.prefillCli().explicit = true;
    defer {
        generate_mod.prefillCli().override = 8192;
        generate_mod.prefillCli().explicit = false;
    }
    try t.expectEqual(@as(usize, 2048), generate_mod.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        40_000,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
}

test "prefillMemoryNeeded: every prefill peak the box measured, re-derived for the conv_state compaction" {
    // Peak GPU bytes above steady state for ONE request on a clean boot
    // (`--prefix-cache-entries 0 --no-mtp --no-drafter --no-pld`, /props
    // peak_bytes minus active_bytes, M4 Max) — byte-identical
    // across repeat boots, so these are exact figures, not samples. An
    // estimate BELOW a peak we have actually seen is the case that ends in an
    // uncatchable Metal OOM instead of a 400, so every row must be covered.
    // Pre-fix this function billed 5 of these 8 shapes short, worst 0.58x.
    // The qwen3.5 rows are the measured peaks minus the exact conv_state pin the compaction removed ((linear_layers - 5) x chunk x qkv x 2); the packs are not on the box to re-measure.
    const t = std.testing;
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;

    // qwen3_5 27B: hidden 5120, ffn 17408, 64 layers / 16 caching, hd 256,
    // 48 GatedDeltaNet layers x (2x16x128 + 48x128) elems x 2 B of stream.
    const q27_stream: u64 = 5 * (2 * 16 * 128 + 48 * 128) * 2;
    const q27_dq: u64 = 3 * 17408 * 5120 * 2;
    try t.expect(prefillMemoryNeeded(9320, 24, 4, 65536, 256, 256, 5120, 17408, 16, 2048, 9320, q27_stream, q27_dq, .{}) >= 2_174_449_280);
    try t.expect(prefillMemoryNeeded(9270, 24, 4, 65536, 256, 256, 5120, 17408, 16, 8192, 9270, q27_stream, q27_dq, .{}) >= 3_623_797_120);
    // qwen3_5 4B: hidden 2560, ffn 9216, 32 layers / 8 caching, 24 GDN layers.
    const q4_stream: u64 = 5 * (2 * 16 * 128 + 32 * 128) * 2;
    const q4_dq: u64 = 3 * 9216 * 2560 * 2;
    try t.expect(prefillMemoryNeeded(9340, 16, 4, 32768, 256, 256, 2560, 9216, 16, 1024, 9340, q4_stream, q4_dq, .{}) >= 1_329_232_896);
    try t.expect(prefillMemoryNeeded(9270, 16, 4, 32768, 256, 256, 2560, 9216, 16, 8192, 9270, q4_stream, q4_dq, .{}) >= 2_559_863_168);
    // gemma4 26B-A4B MoE: hidden 2816, top_k 8 x moe 704, 30 sliding layers.
    const g4_stream: u64 = 4 * 8 * 2 * (2816 + 704) * 2;
    const g4_dq: u64 = 3 * 5632 * 2816 * 2;
    try t.expect(prefillMemoryNeeded(8204, 16, 8, 245760, 256, 256, 2816, 5632, 16, 4096, 8204, g4_stream, g4_dq, .{}) >= 3_960_000_000);
    // lfm2 2.6B conv hybrid at the narrowest chunk measured: no stream term,
    // and the forward is under the dequant route's minimum width.
    try t.expect(prefillMemoryNeeded(10089, 32, 8, 16384, 64, 64, 2048, 10752, 16, 256, 10089, 0, 3 * 10752 * 2048 * 2, .{}) >= 782_000_000);
    try t.expect(prefillMemoryNeeded(10206, 32, 8, 16384, 64, 64, 2048, 10752, 16, 8192, 10206, 0, 3 * 10752 * 2048 * 2, .{}) >= 2_527_000_000);
    // muse_glimmer 30B dense: hidden 6656, ffn 19968, 52 layers, hd 128.
    try t.expect(prefillMemoryNeeded(9827, 32, 2, 53248, 128, 128, 6656, 19968, 16, 2048, 9827, 0, 3 * 19968 * 6656 * 2, .{}) >= 2_178_000_000);
}

test "prefillMemoryNeeded: the new terms fire only where the measurement put them" {
    const t = std.testing;
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    const q27 = .{ .seq = 9320, .h = 24, .kvh = 4, .kvpt = 65536, .hd = 256, .hidden = 5120, .ffn = 17408 };
    const stream: u64 = 48 * (2 * 16 * 128 + 48 * 128) * 2;
    const dq: u64 = 3 * 17408 * 5120 * 2;

    // The stream term is per CHUNK-TOKEN: it scales with the forward width and
    // not with the prompt, which is what the whole chunk-bounded model says.
    const at2k = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, stream, 0, .{});
    const at8k = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 8192, q27.seq, stream, 0, .{});
    const flat2k = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, 0, 0, .{});
    const mlp2k: u64 = 8 * 2048 * q27.ffn * 2;
    const floor: u64 = 512 * 1024 * 1024;
    // The stream arm wins on this arch — that IS the fix — so the bill is one
    // MLP envelope plus the streams, not the three-envelope floor.
    try t.expectEqual(@as(u64, (q27.seq * q27.kvpt + mlp2k + 2048 * stream + floor) * 5 / 4), at2k);
    try t.expectEqual(@as(u64, (q27.seq * q27.kvpt + 3 * mlp2k + floor) * 5 / 4), flat2k);
    try t.expect(at2k > flat2k);
    try t.expect(at8k > at2k);

    // A longer prompt at the same chunk adds KV only — the stream term does
    // not move, or it would be the per-token multiplier this class is about.
    const long = prefillMemoryNeeded(4 * q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, 4 * q27.seq, stream, 0, .{});
    try t.expectEqual(long - at2k, 3 * q27.seq * q27.kvpt * 5 / 4);

    // The dequant-weight term is chunk-INDEPENDENT (it is weights) but only
    // bills where the route can fire: at or above PREFILL_DQ_GEMM_MIN_M.
    const dq_wide = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, 0, dq, .{});
    try t.expectEqual(dq_wide - flat2k, dq * 5 / 4);
    const narrow_with = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 1024, q27.seq, 0, dq, .{});
    const narrow_without = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 1024, q27.seq, 0, 0, .{});
    try t.expectEqual(narrow_with, narrow_without);
    // A dense checkpoint passes 0 and is billed nothing for it (measured: a
    // bf16 lfm2 sits within 0.19 GB of the 8-bit one, which IS this term).
    try t.expectEqual(flat2k, prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, 0, 0, .{}));

    // An arch with neither keeps the 3-envelope bill it had, plus the floor —
    // no plain-attention arch's admission tightens by more than the floor.
    const dense_only = prefillMemoryNeeded(9827, 32, 2, 53248, 128, 128, 6656, 19968, 16, 2048, 9827, 0, 0, .{});
    const pre_fix: u64 = (9827 * 53248 + 3 * 8 * 2048 * 19968 * 2) * 5 / 4;
    try t.expectEqual(dense_only - pre_fix, 512 * 1024 * 1024 * 5 / 4);
}

test "prefill stream bill: GLM-5 on NAX bills no top_k replica or linear stream" {
    const t = std.testing;
    var glm = model_mod.ModelConfig{ .model_type = "glm5_next" };
    glm.num_hidden_layers = 45;
    glm.full_attention_interval = 4;
    glm.linear_num_key_heads = 64;
    glm.linear_num_value_heads = 64;
    glm.hidden_size = 4096;
    glm.num_experts = 288;
    glm.num_experts_per_tok = 8;
    glm.moe_intermediate_size = 2048;
    // Measured 380 KB per chunk-token on M5 (NAX), under the 3-envelope floor.
    try t.expectEqual(@as(u64, 0), prefillStreamBytesPerTokenOn(&glm, true));
    try t.expect(prefillStreamBytesPerTokenOn(&glm, false) > 0);
}

test "prefillStreamBytesPerToken: keyed on the arch's own geometry, zero for plain attention" {
    const t = std.testing;
    // qwen3_5 27B: 64 layers, every 4th full — 48 GatedDeltaNet layers, each
    // holding a chunk-wide q/k/v stream (2 x 16 x 128 key + 48 x 128 value).
    var q27 = model_mod.ModelConfig{ .model_type = "qwen3_5" };
    q27.num_hidden_layers = 64;
    q27.full_attention_interval = 4;
    q27.linear_num_key_heads = 16;
    q27.linear_num_value_heads = 48;
    try t.expectEqual(@as(u64, 5 * (2 * 16 * 128 + 48 * 128) * 2), prefillStreamBytesPerToken(&q27));

    var q4 = model_mod.ModelConfig{ .model_type = "qwen3_5" };
    q4.num_hidden_layers = 32;
    q4.full_attention_interval = 4;
    q4.linear_num_key_heads = 16;
    q4.linear_num_value_heads = 32;
    try t.expectEqual(@as(u64, 5 * (2 * 16 * 128 + 32 * 128) * 2), prefillStreamBytesPerToken(&q4));

    // gemma4 26B-A4B: no linear layers, but a MoE prefill gathers a top_k-fold
    // copy of the hidden stream beside the expert rows, for the 4 layers the
    // MoE eval cadence lets coexist.
    var g4 = model_mod.ModelConfig{ .model_type = "gemma4" };
    g4.num_hidden_layers = 30;
    g4.hidden_size = 2816;
    g4.num_experts = 128;
    g4.num_experts_per_tok = 8;
    g4.moe_intermediate_size = 704;
    try t.expectEqual(@as(u64, 4 * 8 * 2 * (2816 + 704) * 2), prefillStreamBytesPerToken(&g4));

    // A plain attention arch (muse, llama, mistral) declares neither and is
    // billed exactly what it was billed before this term existed.
    var dense = model_mod.ModelConfig{ .model_type = "muse_glimmer" };
    dense.num_hidden_layers = 52;
    dense.hidden_size = 6656;
    try t.expectEqual(@as(u64, 0), prefillStreamBytesPerToken(&dense));

    // lfm2's conv layers are NOT linear-attention layers: it declares no
    // linear head geometry, and its measured slope is one MLP envelope — the
    // term must not invent a stream for it.
    var lfm2 = model_mod.ModelConfig{ .model_type = "lfm2" };
    lfm2.num_hidden_layers = 30;
    lfm2.full_attention_interval = 4;
    lfm2.hidden_size = 2048;
    try t.expectEqual(@as(u64, 0), prefillStreamBytesPerToken(&lfm2));

    // nemotron_h MoE (3.5 Lightning 30B-A3B): a hybrid whose MoE blocks bill
    // the MoE stream term, and whose FFN width counts the shared expert.
    var nemo = model_mod.ModelConfig{ .model_type = "nemotron_h" };
    nemo.has_hybrid_layers = true;
    nemo.num_hidden_layers = 52;
    nemo.hidden_size = 2688;
    nemo.num_experts = 128;
    nemo.num_experts_per_tok = 6;
    nemo.moe_intermediate_size = 1856;
    nemo.shared_expert_intermediate_size = 3712;
    try t.expectEqual(@as(u64, 4 * 6 * 2 * (2688 + 1856) * 2), prefillStreamBytesPerToken(&nemo));
    try t.expectEqual(@as(u64, 1856 * 6 + 3712), prefillFfnWidth(&nemo));
}

test "prefillDequantWeightBytes: affine-quantized weights only, and it reads the route's kill switch" {
    const t = std.testing;
    var cfg = model_mod.ModelConfig{ .model_type = "muse_glimmer" };
    cfg.hidden_size = 6656;
    cfg.intermediate_size = 19968;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    cfg.quant_mode = .affine;
    try t.expectEqual(@as(u64, 3 * 19968 * 6656 * 2), prefillDequantWeightBytes(&cfg));

    // Dense bf16: the route never runs, and the measurement agrees — a bf16
    // checkpoint's prefill peak sits below the quantized one by this term.
    var dense = cfg;
    dense.quant_bits = 0;
    try t.expectEqual(@as(u64, 0), prefillDequantWeightBytes(&dense));

    // Non-affine (nvfp4/mxfp8) stays in mlx_quantized_matmul — no dequantized
    // copy exists to bill (measured: nvfp4 lands with the dq-off arm).
    var nvfp4 = cfg;
    nvfp4.quant_mode = .nvfp4;
    try t.expectEqual(@as(u64, 0), prefillDequantWeightBytes(&nvfp4));

    // The estimator must not bill an allocation the kill switch removed.
    prefill_dq_gemm.prefill_dq_gemm_override = false;
    defer prefill_dq_gemm.prefill_dq_gemm_override = null;
    try t.expectEqual(@as(u64, 0), prefillDequantWeightBytes(&cfg));
}

test "prefillMemoryNeeded: a sparse-attention arch bills its KEY BOUND, not the whole prompt" {
    const t = std.testing;
    // DeepSeek-V4-Flash shape: 64 heads, ONE latent kv head,
    // 43 layers, head_dim 512 — outside every fused kernel, so the score term
    // fires. But DSV4 attends to sliding_window(128) + index_topk(512) + sink
    // keys per query, NEVER the whole prompt (deepseek_v4.zig: `tk = wk +
    // n_sel`). Billing a dense 5806-wide key axis put a 5806-token prompt at
    // ~10.3 GB and 400-rejected it on a 1M-context model with 8.6 GB free.
    // The live request verbatim: 5806 tokens at kv-quant 8, chunk 5632, with
    // 8629 MB free. Pre-fix it billed 10277 MB — a dense 5806-wide key axis
    // PLUS the 15360 intermediate_size struct default — and 400'd.
    const available: u64 = 8629 * 1024 * 1024;
    const before = prefillMemoryNeeded(5806, 64, 1, 88064, 512, 512, 4096, 15360, 8, 5632, 5806, 0, 0, .{});
    try t.expectEqual(@as(u64, 10277 + 640), before / (1024 * 1024));
    try t.expect(before > available);
    // Fixed: 641 keys (window 128 + top-512 + sink) and the width the config
    // actually states (top_k 6 x moe_intermediate 2048 = 12288, what
    // prefillFfnWidth resolves for this checkpoint) — 4848 MB, admitted.
    const after = prefillMemoryNeeded(5806, 64, 1, 88064, 512, 512, 4096, 12288, 8, 5632, 641, 0, 0, .{});
    try t.expectEqual(@as(u64, 4848 + 640), after / (1024 * 1024));
    try t.expect(after < available);
    // The score term is the ONLY one the key bound may touch: same call with a
    // fused head_dim has no score scratch, so the bound changes nothing.
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    try t.expectEqual(
        prefillMemoryNeeded(5806, 64, 1, 44032, 256, 256, 4096, 2048, 8, 5632, 5806, 0, 0, .{}),
        prefillMemoryNeeded(5806, 64, 1, 44032, 256, 256, 4096, 2048, 8, 5632, 641, 0, 0, .{}),
    );
}

test "resolvePrefillChunk: the 16 GB tier gets a working context instead of the 1024 floor" {
    // The reported bug, end to end. Mistral-7B-4bit on a 16 GB Mac: ~11.9 GiB
    // Metal working set, 3.80 GiB of weights (measured), the hot prefix cache
    // at its 2 GB default. At the launch chunk the transient reserve alone is
    // larger than what is left, so the sizer floored the advertised context at
    // 1024 and the guard 400'd a 10,348-token prompt for "needing" 9.18 GiB —
    // a prompt whose peak measures 2.391 GiB (M4 Max, one request per boot).
    const t = std.testing;
    const GiB: u64 = 1024 * 1024 * 1024;
    const ceiling: u64 = 11_918_000_000;
    const weights: u64 = 4_080_000_000;
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 32;
    cfg.num_attention_heads = 32;
    cfg.num_key_value_heads = 8;
    cfg.head_dim = 128;
    cfg.hidden_size = 4096;
    cfg.intermediate_size = 14336;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    cfg.max_position_embeddings = 32768;
    const per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 16);

    const launch = safeContextForBudget(ceiling, weights, 2 * GiB +| prefillTransientReserve(&cfg, 16, 8192), per_tok, 32768);
    try t.expectEqual(@as(u32, 1024), launch); // the floor: nothing fits

    // The hot-cache budget is no longer an input; auto-context means the pinned-context bar is 0.
    const chunk = resolvePrefillChunk(&cfg, 16, ceiling, weights, 0, 0);
    const sized = safeContextForBudget(ceiling, weights, 2 * GiB +| prefillTransientReserve(&cfg, 16, chunk), per_tok, 32768);
    try t.expect(sized > 12_000);

    // And the guard now admits a real agent prompt at that context, billed
    // against the same chunk. 8.10 GiB is what a 16 GB Mac has free with only
    // the weights resident.
    const needed = prefillMemoryNeeded(
        10_348,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.kvBytesPerToken(),
        cfg.head_dim,
        cfg.prefillScoreHeadDim(),
        cfg.hidden_size,
        prefillFfnWidth(&cfg),
        16,
        chunk,
        cfg.prefillAttnKeys(10_348),
        prefillStreamBytesPerToken(&cfg),
        prefillDequantWeightBytes(&cfg),
        .{},
    );
    try t.expect(needed < ceiling - weights);
    // Still comfortably above the measured 2.391 GiB peak — narrowing the chunk
    // must not turn a conservative bill into an optimistic one.
    try t.expect(needed > 2_391_000_000);
}

test "prefillFfnWidth: a declared MoE width beats the struct default, which is only a fallback" {
    const t = std.testing;
    // DSV4's config ships NO intermediate_size at all, so the struct default
    // (15360) was billed as though declared. The width the config DOES state is
    // top_k(6) x moe_intermediate(2048) — a chunk gathers that many expert rows
    // per token, so billing the bare 2048 would UNDER-bill 3x, and under-
    // billing ends in an uncatchable Metal OOM rather than a 400.
    var moe = model_mod.ModelConfig{};
    moe.intermediate_size = 15360; // struct default, never seen in the JSON
    moe.intermediate_size_declared = false;
    moe.moe_intermediate_size = 2048;
    moe.num_experts_per_tok = 6;
    moe.shared_expert_intermediate_size = 0;
    try t.expectEqual(@as(u64, 12288), prefillFfnWidth(&moe));

    // The shared expert runs on every token too, so it adds rather than maxes.
    var shared = moe;
    shared.shared_expert_intermediate_size = 1024;
    try t.expectEqual(@as(u64, 13312), prefillFfnWidth(&shared));

    // A dense checkpoint is unaffected: no MoE width, declared ffn wins.
    var dense = model_mod.ModelConfig{};
    dense.intermediate_size = 13312;
    dense.intermediate_size_declared = true;
    dense.moe_intermediate_size = 0;
    dense.num_experts_per_tok = 0;
    try t.expectEqual(@as(u64, 13312), prefillFfnWidth(&dense));

    // A checkpoint that DECLARES a dense ffn keeps the conservative @max — a
    // hybrid with dense layers really does run the wider one.
    var declared = moe;
    declared.intermediate_size_declared = true;
    try t.expectEqual(@as(u64, 15360), prefillFfnWidth(&declared));

    // Declares nothing: the struct default is all we have, so it still wins.
    var bare = model_mod.ModelConfig{};
    bare.intermediate_size = 15360;
    bare.intermediate_size_declared = false;
    bare.moe_intermediate_size = 0;
    bare.shared_expert_intermediate_size = 0;
    try t.expectEqual(@as(u64, 15360), prefillFfnWidth(&bare));
}

/// The qwen4_exp shape of the live #353 report: Qwen3.8-Flash-Next mixed-4-8bit, `--ctx-size 1048576 --kv-quant 8`.
pub fn qwen4ExpOomConfig() model_mod.ModelConfig {
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "qwen4_exp"; // arch-scan-exempt: test fixture, not a gate
    cfg.legacy.caps.long_ctx = true;
    cfg.num_hidden_layers = 48;
    cfg.full_attention_interval = 4; // 12 caching layers, 36 GatedDeltaNet
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 256;
    cfg.hidden_size = 2560;
    cfg.moe_intermediate_size = 640;
    cfg.num_experts = 512;
    cfg.num_experts_per_tok = 10;
    cfg.shared_expert_intermediate_size = 640;
    cfg.linear_num_key_heads = 16;
    cfg.linear_key_head_dim = 128;
    cfg.linear_num_value_heads = 48;
    cfg.linear_value_head_dim = 128;
    cfg.linear_conv_kernel_dim = 4;
    cfg.indexer_budget = 512;
    cfg.indexer_head_dim = 128;
    cfg.indexer_compress_ratio = 4;
    // Pinned: with nothing pinned `getEffectiveContextLength` reads live GPU memory and every
    // expectation below depends on the box.
    cfg.pinned_context = 1_048_576;
    return cfg;
}

test "qwen4ExpOomConfig does not leak qsa_score_fused_override" {
    const saved = sparse_score.qsa_score_fused_override;
    defer sparse_score.qsa_score_fused_override = saved;
    sparse_score.qsa_score_fused_override = null;
    _ = qwen4ExpOomConfig();
    try std.testing.expectEqual(@as(?bool, null), sparse_score.qsa_score_fused_override);
}

test "the 458k prefill's two unbilled terms are billed: retained checkpoints and reserved capacity" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    // A 458,832-token cold prefill was admitted against 26.6 GB of headroom and died mid-prefill.
    const cfg = qwen4ExpOomConfig();
    const seq: u64 = 458_832;
    const chunk: u64 = 4096;

    // 1. SSM checkpoints: 36 GDN layers x (state + conv window), up to --ssm-checkpoint-max.
    const per_cp = cfg.ssmCheckpointBytes();
    // 36 x (1,572,864 + 61,440) B = 58.83 MB.
    try t.expect(per_cp > 58_000_000 and per_cp < 60_000_000);
    const saved_cp_max = http_server.launch().ssm_checkpoint_max;
    http_server.launch().ssm_checkpoint_max = 32;
    defer http_server.launch().ssm_checkpoint_max = saved_cp_max;
    const held = retainedSsmCheckpointBytes(&cfg, seq, 0, chunk);
    try t.expectEqual(@as(u64, http_server.launch().ssm_checkpoint_max) * per_cp, held);
    try t.expect(held > 1_800_000_000);
    // It saturates: 393k holds the same 32 as 458k.
    try t.expectEqual(held, retainedSsmCheckpointBytes(&cfg, 393_216, 0, chunk));
    // An arch with no linear layers is billed nothing.
    var dense = model_mod.ModelConfig{};
    dense.num_hidden_layers = 32;
    dense.num_attention_heads = 32;
    try t.expectEqual(@as(u64, 0), dense.ssmCheckpointBytes());
    try t.expectEqual(@as(u64, 0), retainedSsmCheckpointBytes(&dense, seq, 0, chunk));

    // 2. Reserved capacity: prompt + max_tokens + one chunk; the guard bills the headroom beyond
    // the prompt's own rows, not a second copy of the cache.
    const terms = prefillRequestTerms(&cfg, seq, 2048, 8, chunk, .{});
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8);
    try t.expectEqual((2048 + chunk) * kv_per_tok, terms.reserved_kv_bytes);
    try t.expect(terms.reserved_kv_bytes * 4 < seq * kv_per_tok); // headroom, not a copy
    // The indexer history is billed at one copy plus the score bank (`statePerTokenBilled`).
    try t.expectEqual(cfg.qsaHistoryBytesPerToken() + cfg.qsaScoreBankBytesPerToken(), statePerTokenBilled(&cfg));
    try t.expectEqual((seq + 2048 + chunk) * statePerTokenBilled(&cfg), terms.state_bytes);
    try t.expectEqual(held, terms.checkpoint_bytes);

    // 3. The bill moved by more than 3 GB, toward a 503 instead of a corpse.
    const with_terms = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, prefillFfnWidth(&cfg), 8, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), terms);
    const without = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, prefillFfnWidth(&cfg), 8, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), .{});
    try t.expect(with_terms > without + 3_000_000_000);
    // The sentinel runs through the same path: an omitted max_tokens must not move the bill past the context.
    const sentinel: u64 = omittedMaxTokensDefault(@intCast(@min(getEffectiveContextLength(&cfg), 1_048_576)));
    const with_sentinel = prefillRequestTerms(&cfg, seq, sentinel, 8, chunk, .{});
    const with_explicit = prefillRequestTerms(&cfg, seq, 2048, 8, chunk, .{});
    try t.expect(with_sentinel.reserved_kv_bytes < 8 * 1024 * 1024 * 1024);
    try t.expect(with_sentinel.state_bytes < 4 * with_explicit.state_bytes);
    try t.expectEqual(with_explicit.checkpoint_bytes, with_sentinel.checkpoint_bytes);

    // A short prompt is untouched.
    const short = prefillRequestTerms(&cfg, 4096, 2048, 8, chunk, .{});
    try t.expectEqual(@as(u64, 0), short.reserved_kv_bytes);
    try t.expectEqual(4096 * statePerTokenBilled(&cfg), short.state_bytes);
}

test "the reservation is bounded by the CONTEXT: an omitted max_tokens cannot bill 26 TB" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    // An omitted `max_tokens` is maxInt(u32)/4; the memory guard used to run before
    // `clampMaxTokens`, so the reservation billed ~26 TB of KV and refused every long prompt.
    const KVCache = transformer_mod.KVCache;
    const ctx: u64 = 262_144;
    const seq: u64 = 50_000;
    const sentinel: u64 = omittedMaxTokensDefault(@intCast(ctx));
    try t.expectEqual(@as(u64, std.math.maxInt(u32) / 4), sentinel);

    // (1) The cache clamps: the reservation cannot exceed the context plus a chunk.
    const reserved = KVCache.reservedTokens(seq, sentinel, 4096, ctx);
    // Bounded twice: by the generation headroom first, by the context second.
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + 4096, reserved);
    try t.expect(reserved < ctx);
    // (2) The four surfaces clamp first and hand the guard the clamped value.
    const clamped: u64 = clampMaxTokens(@intCast(sentinel), @intCast(seq), @intCast(ctx));
    try t.expectEqual(ctx - seq, clamped);
    try t.expectEqual(reserved, KVCache.reservedTokens(seq, clamped, 4096, ctx));

    try t.expectEqual(seq + 2048 + 4096, KVCache.reservedTokens(seq, 2048, 4096, ctx));
    // A caller that names no context is not context-clamped, but the headroom bound still applies.
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + 4096, KVCache.reservedTokens(seq, sentinel, 4096, 0));
    try t.expectEqual(ctx + 4096, KVCache.reservedTokens(ctx, sentinel, 4096, ctx));

    const cfg = qwen4ExpOomConfig();
    const terms = prefillRequestTerms(&cfg, seq, clamped, 8, 4096, .{});
    try t.expect(terms.reserved_kv_bytes < 4 * 1024 * 1024 * 1024);
}

/// The generic admission bill for `qwen4ExpOomConfig`, minus the live-memory read a unit test cannot make.
fn oomBillFor(cfg: *const model_mod.ModelConfig, seq: u64, chunk: u64, terms: PrefillRequestTerms) u64 {
    return prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, prefillFfnWidth(cfg), 8, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(cfg), prefillDequantWeightBytes(cfg), terms) +
        qsaMaskBytes(cfg, @min(chunk, @max(seq, 1)), seq);
}

test "an omitted max_tokens reserves a generation HEADROOM, not the rest of the context" {
    const t = std.testing;
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    // A 383,067-token prompt with no `max_tokens` was admitted by eviction and then refused on
    // the inference thread: `clampMaxTokens` made the reservation cover the whole context
    // (~13 GB for a generation that would never run). The reservation exists to stop buffer
    // coexistence during prefill, not to pre-buy a generation.
    const KVCache = transformer_mod.KVCache;
    const MB: u64 = 1024 * 1024;
    const cfg = qwen4ExpOomConfig();
    const ctx: u64 = getEffectiveContextLength(&cfg);
    try t.expectEqual(@as(u64, 1_048_576), ctx);
    const seq: u64 = 383_067;
    const chunk: u64 = 4096;
    const clamped: u64 = clampMaxTokens(@intCast(omittedMaxTokensDefault(@intCast(ctx))), @intCast(seq), @intCast(ctx));
    try t.expectEqual(ctx - seq, clamped); // 665,509 — the live number

    // The reservation is bounded by the headroom, not by the context.
    const reserved = KVCache.reservedTokens(seq, clamped, chunk, ctx);
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + chunk, reserved);
    try t.expect(reserved < seq + ctx / 8);
    try t.expectEqual(seq + 512 + chunk, KVCache.reservedTokens(seq, 512, chunk, ctx));
    try t.expectEqual(ctx + chunk, KVCache.reservedTokens(ctx - 16, clamped, chunk, ctx));

    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8);
    const terms = prefillRequestTerms(&cfg, seq, clamped, 8, chunk, .{});
    try t.expectEqual((KVCache.RESERVE_GEN_HEADROOM + chunk) * kv_per_tok, terms.reserved_kv_bytes);
    try t.expectEqual(reserved * statePerTokenBilled(&cfg), terms.state_bytes);

    // ...and the whole bill drops by more than 10 GB, from above the live headroom to under it.
    const ctx_bounded: PrefillRequestTerms = .{
        .reserved_kv_bytes = (ctx + chunk - seq) * kv_per_tok,
        .state_bytes = (ctx + chunk) * statePerTokenBilled(&cfg),
        .checkpoint_bytes = terms.checkpoint_bytes,
    };
    const bill = oomBillFor(&cfg, seq, chunk, terms);
    const old_bill = oomBillFor(&cfg, seq, chunk, ctx_bounded);
    try t.expect(old_bill > bill + 10 * 1024 * MB);
    try t.expect(old_bill > 28_000 * MB);
    try t.expect(bill < 30_000 * MB); // under the live "~37130MB available"
}

test "the QSA history is billed at the copies a slot HOLDS: one with the commit handoff, two with a lever off, plus the score bank" {
    const t = std.testing;
    const cfg = qwen4ExpOomConfig();
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    const one = cfg.qsaHistoryBytesPerToken();
    const bank = cfg.qsaScoreBankBytesPerToken();
    try t.expect(one > 0);
    try t.expect(bank > 0);
    defer transformer_mod.qsa_history_share_override = null;
    transformer_mod.qsa_history_share_override = true;
    try t.expectEqual(one + bank, statePerTokenBilled(&cfg));
    transformer_mod.qsa_history_share_override = false;
    try t.expectEqual(one * 2 + bank, statePerTokenBilled(&cfg));
    transformer_mod.qsa_history_share_override = true;

    const seq: u64 = 458_832;
    const terms = prefillRequestTerms(&cfg, seq, 2048, 8, 4096, .{});
    try t.expectEqual((seq + 2048 + 4096) * (one + bank), terms.state_bytes);

    var dense = model_mod.ModelConfig{};
    dense.num_hidden_layers = 32;
    dense.num_attention_heads = 32;
    try t.expectEqual(@as(u64, 0), statePerTokenBilled(&dense));
}

test "a chain extension that OUTGROWS the resident entry is credited the rows it holds" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    // Past the restored capacity the cache grows per layer: the resident rows stay credited and
    // the grow bills the new capacity beyond the old plus one eval window of old buffers.
    const cfg = qwen4ExpOomConfig();
    const seq: u64 = 1_047_556;
    const chunk: u64 = 512;
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 200;
    const capacity: u64 = 787_456; // what the 786k turn reserved

    const reserved = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    try t.expectEqual(@as(u64, 1_048_268), reserved);
    const extend = WarmPrefix{ .matched_tokens = 786_707, .capacity_tokens = capacity, .will_donate = true };
    try t.expect(reserved > capacity);
    try t.expect(extend.grows(seq));
    try t.expectEqual(@as(u64, 786_707), extend.creditedRows(reserved));
    const per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const ext_terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, extend);
    const window = attnLayersPerEvalWindow(&cfg, transformer_mod.Transformer.MOE_EVAL_EVERY_N_LAYERS);
    try t.expectEqual(capacity * per_tok / cfg.attnCacheLayerCount() * window, ext_terms.grow_coexist_bytes);
    try t.expectEqual(
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{}) -
            (786_707 * per_tok - ext_terms.grow_coexist_bytes) * 5 / 4,
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, extend),
    );

    // A buffer that already covers the reservation credits the full shared prefix: the gate is capacity, not prompt length.
    const roomy = WarmPrefix{ .matched_tokens = 786_707, .capacity_tokens = 1_048_576, .will_donate = true };
    try t.expectEqual(@as(u64, 786_707), roomy.creditedRows(reserved));
    const credit = 786_707 * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    try t.expectEqual(
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{}) - credit * 5 / 4,
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, roomy),
    );
}

/// The live 768k scenario's checkout inputs, as `prefix_cache.checkoutEligible` takes them.
const CheckoutCase = struct {
    ssd_first: bool = true,
    move_enabled: bool = true,
    pending_disk: bool = false,
    entry_tokens: usize = 786_695,
    shared: usize = 786_695,
    prompt_len: usize = 786_707,
    has_slot: bool = true,

    /// The WarmPrefix the scheduler builds, with `will_donate` from restore-move's own predicate.
    fn warm(self: CheckoutCase, matched: u64, capacity: u64) WarmPrefix {
        return .{
            .matched_tokens = matched,
            .capacity_tokens = capacity,
            .will_donate = prefix_cache_mod.HotPrefixCache.checkoutEligible(
                self.ssd_first,
                self.move_enabled,
                self.pending_disk,
                self.entry_tokens,
                self.shared,
                self.prompt_len,
                self.has_slot,
            ),
        };
    }
};

test "the warm KV credit fires ONLY where the restore checked its entry out" {
    const t = std.testing;
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    // `KVCache.restore` binds by refcount and mlx donates only at `use_count() == 1`, so on a
    // shared restore the first `writeAtOffset` copies the whole prefix: the credited bytes are
    // allocated. Only the checkout makes the credit true; `matched > 0` alone is not enough.
    const cfg = qwen4ExpOomConfig();
    const seq: u64 = 786_707;
    const chunk: u64 = 512;
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 200;
    const mb: u64 = 1024 * 1024;
    const matched: u64 = 786_676;
    const capacity: u64 = 787_456;
    const reserved = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    const cold = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{});

    // (1) qwen4_exp, SSD-first, full-prefix hit, switch on: the credit is real.
    const donating = (CheckoutCase{}).warm(matched, capacity);
    try t.expect(donating.will_donate);
    try t.expectEqual(matched, donating.creditedRows(reserved));
    const warm = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, donating);
    // Baseline = a shared restore of the same span, so the two credits are not folded into one comparison.
    const shared_same_span = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = capacity });
    try t.expectEqual(shared_same_span - matched * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) * 5 / 4, warm);
    try t.expect(warm / mb > 6_500 and warm / mb < 8_500);

    // Every arm below is the same restore minus one conjunct of the checkout, and every one shares.
    const shared_arms = [_]WarmPrefix{
        // (2) a partial hit: the entry describes a prefix this prompt left.
        (CheckoutCase{ .shared = 700_000 }).warm(matched, capacity),
        // (3) a non-qwen4 arch: a checkout can never be taken there.
        (CheckoutCase{ .ssd_first = false }).warm(matched, capacity),
        // (4) restore by move off: the refcount share.
        (CheckoutCase{ .move_enabled = false }).warm(matched, capacity),
        // (5) a pending disk record shares the same buffers.
        (CheckoutCase{ .pending_disk = true }).warm(matched, capacity),
        // (6) no slot to hand the buffers to (the connection thread's shape).
        (CheckoutCase{ .has_slot = false }).warm(matched, capacity),
    };
    for (shared_arms, 0..) |arm, i| {
        errdefer std.debug.print("shared arm {d} credited\n", .{i});
        try t.expect(!arm.will_donate);
        try t.expectEqual(@as(u64, 0), arm.creditedRows(reserved));
        try t.expectEqual(shared_same_span, prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, arm));
    }
    // ...and `cold` is still the number a prompt with NO resident prefix pays,
    // which is what the `.{}` arm above is for.
    try t.expect(cold > shared_same_span);

    // Capacity is a CEILING on the credit, not a switch: a buffer holding 700,000 rows donates
    // them and the request is billed the growth on top.
    const outgrown = (CheckoutCase{}).warm(matched, 700_000);
    try t.expect(outgrown.will_donate);
    try t.expectEqual(@as(u64, 700_000), outgrown.creditedRows(reserved));
    try t.expect(outgrown.grows(seq));
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const og_terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, outgrown);
    // 31 tokens wide, BELOW `prefillEvalCadenceApplies`: one eval for the whole layer loop, so
    // every old buffer is alive at once and the coexistence term cancels the credit exactly.
    try t.expect(seq - matched < 32);
    try t.expectEqual(700_000 * kv_per_tok, og_terms.grow_coexist_bytes);
    try t.expectEqual(shared_same_span, prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, outgrown));
    // An append wide enough to take the cadence pays ONE eval window instead, and that is the
    // difference between the old bill and the new one.
    const wide = WarmPrefix{ .matched_tokens = 700_000, .capacity_tokens = 700_000, .will_donate = true };
    const wide_terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, wide);
    const window = attnLayersPerEvalWindow(&cfg, transformer_mod.Transformer.MOE_EVAL_EVERY_N_LAYERS);
    try t.expect(window < cfg.attnCacheLayerCount());
    try t.expectEqual(700_000 * kv_per_tok / cfg.attnCacheLayerCount() * window, wide_terms.grow_coexist_bytes);
    try t.expect(prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, wide) < shared_same_span);
}

test "prefillRequestTerms: the reservation terms are qwen4_exp-only, a donated warm is credited everywhere" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The allocator side of the reservation is gated (`generate.reservedPrefillTokens`); the
    // guard side was not, so off qwen4_exp the bill charged headroom never reserved.
    const t = std.testing;
    const seq: u64 = 200_000; // well past RESERVE_MIN_TOKENS
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const kv_bits: u64 = 8;

    // A 27B-class GDN pack: `ssmCheckpointBytes()` is non-zero here.
    var q35 = qwen4ExpOomConfig();
    q35.model_type = "qwen3_5_moe";
    q35.legacy.caps.long_ctx = std.mem.eql(u8, q35.model_type, "qwen4_exp");
    try t.expect(q35.ssmCheckpointBytes() > 0);
    try t.expect(retainedSsmCheckpointBytes(&q35, seq, 0, chunk) > 0);

    // A donating warm is credited its resident rows on every arch (a checked-out entry is the slot's own
    // buffer); a shared one is not, and then the ungated bill is the previous expression plus the
    // slot's fixed KV.
    const warm = WarmPrefix{ .matched_tokens = 100_000, .capacity_tokens = 400_000, .will_donate = true };
    const shared = WarmPrefix{ .matched_tokens = 100_000, .capacity_tokens = 400_000 };
    for ([_][]const u8{ "qwen3_5", "qwen3_5_moe", "qwen3_next", "bailing_hybrid", "lfm2", "nemotron_h", "llama", "mistral" }) |mt| {
        var cfg = qwen4ExpOomConfig();
        cfg.model_type = mt;
        cfg.legacy.caps.long_ctx = std.mem.eql(u8, cfg.model_type, "qwen4_exp");
        const terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, warm);
        try t.expectEqual(@as(u64, 0), terms.reserved_kv_bytes);
        try t.expectEqual(@as(u64, 0), terms.checkpoint_bytes);
        try t.expectEqual(@as(u64, 0), terms.state_bytes);
        try t.expectEqual(@as(u64, 0), terms.grow_coexist_bytes);
        const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
        try t.expectEqual(100_000 * kv_per_tok, terms.shared_resident_bytes);
        const sh = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, shared);
        try t.expectEqual(@as(u64, 0), sh.shared_resident_bytes);
        try t.expectEqual(
            prefillMemoryNeeded(seq, 24, 2, cfg.kvBytesPerToken(), 256, 256, 2560, 9216, kv_bits, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), .{ .slot_fixed_kv_bytes = slotFixedKvBytes(&cfg, kv_bits, chunk) }),
            prefillMemoryNeeded(seq, 24, 2, cfg.kvBytesPerToken(), 256, 256, 2560, 9216, kv_bits, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), sh),
        );
    }

    const q4 = qwen4ExpOomConfig();
    const gated = prefillRequestTerms(&q4, seq, max_tokens, kv_bits, chunk, warm);
    try t.expect(gated.reserved_kv_bytes > 0);
    try t.expect(gated.checkpoint_bytes > 0);
    try t.expect(gated.state_bytes > 0);
    try t.expect(gated.shared_resident_bytes > 0);
}

test "GLM ledger: transient bills latent and both indexer views, not expanded heads" {
    const t = std.testing;
    var glm = glm5TestConfig();
    const seq: u64 = 131072;
    const dense = prefillTransientReserveAtKv(&glm, 16, 512, seq);
    const q8 = prefillTransientReserveAtKv(&glm, 8, 512, seq);
    try t.expectEqual(@as(u64, 11 * 1536), glm.kvBytesPerToken());
    try t.expectEqual(@as(u64, 8976), kvBytesPerTokenAtBits(glm.kvBytesPerToken(), 8));
    try t.expectEqual(seq * (1536 + 512) * 5 / 4, q8 - dense);
}

test "GLM ledger: non-GLM transient widths are unchanged" {
    const t = std.testing;
    var llama = model_mod.ModelConfig{ .model_type = "llama" };
    llama.num_key_value_heads = 8;
    llama.head_dim = 128;
    const dense = prefillTransientReserveAtKv(&llama, 16, 512, 131072);
    const q8 = prefillTransientReserveAtKv(&llama, 8, 512, 131072);
    try t.expectEqual(@as(u64, 131072 * 2 * 8 * 128 * 2 * 5 / 4), q8 - dense);
}

test "GLM ledger: unrelated ungated request terms are unchanged" {
    const t = std.testing;
    const seq: u64 = 100_000;
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const kv_bits: u64 = 8;
    for ([_][]const u8{ "qwen3_5", "lfm2", "nemotron_h", "llama" }) |mt| {
        var cfg = qwen4ExpOomConfig();
        cfg.model_type = mt;
        cfg.legacy.caps.long_ctx = std.mem.eql(u8, cfg.model_type, "qwen4_exp");
        const terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{});
        try t.expectEqual(@as(u64, 0), terms.checkpoint_bytes);
        try t.expectEqual(@as(u64, 0), terms.reserved_kv_bytes);
        try t.expectEqual(@as(u64, 0), terms.state_bytes);
        try t.expectEqual(@as(u64, 0), terms.shared_resident_bytes);
        try t.expectEqual(@as(u64, 0), terms.grow_coexist_bytes);
    }
}

test "prefillRequestTerms: qwen4 MTP head KV is billed when MTP is on and zero when off" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const saved_ov = transformer_mod.Transformer.mtp_head_kv_quant_override;
    defer transformer_mod.Transformer.mtp_head_kv_quant_override = saved_ov;
    transformer_mod.Transformer.mtp_head_kv_quant_override = false;
    const saved_boot = http_server.launch().configured_kv_quant;
    defer http_server.launch().configured_kv_quant = saved_boot;
    http_server.launch().configured_kv_quant = transformer_mod.KVQuantConfig.affine(8);
    var cfg = qwen4ExpOomConfig();
    cfg.mtp_override = false;
    try t.expect(!mtpHeadBilled(&cfg));
    const seq: u64 = 200_000;
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const kv_bits: u64 = 8;
    try t.expectEqual(@as(u64, 2048), mtpHeadKvBytesPerToken(&cfg));
    transformer_mod.Transformer.mtp_head_kv_quant_override = true;
    try t.expectEqual(@as(u64, 1088), mtpHeadKvBytesPerToken(&cfg));
    transformer_mod.Transformer.mtp_head_kv_quant_override = false;
    const off = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{});
    try t.expectEqual(@as(u64, 0), off.mtp_head_kv_bytes);
    const on = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{ .mtp_on = true });
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg)), seq);
    try t.expectEqual(reserved * 2048, on.mtp_head_kv_bytes);
    const n: u64 = cfg.attnCacheLayerCount();
    try t.expectEqual(cfg.qsaRingBytes() + cfg.qsaRingBytes() / n, on.slot_fixed_kv_bytes);
    try t.expectEqual(reserved * (statePerTokenBilled(&cfg) + statePerTokenBilled(&cfg) / n), on.state_bytes);
    try t.expectEqual(cfg.qsaRingBytes(), off.slot_fixed_kv_bytes);
    try t.expectEqual(reserved * statePerTokenBilled(&cfg), off.state_bytes);
    var other = cfg;
    other.model_type = "qwen3_5_moe";
    other.legacy.caps.long_ctx = std.mem.eql(u8, other.model_type, "qwen4_exp");
    const other_on = prefillRequestTerms(&other, seq, max_tokens, kv_bits, chunk, .{ .mtp_on = true });
    try t.expectEqual(@as(u64, 0), other_on.mtp_head_kv_bytes);
}

test "prefillRequestTerms: MTP head KV bills the boot scheme, not a request kv_quant override" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const saved_ov = transformer_mod.Transformer.mtp_head_kv_quant_override;
    defer transformer_mod.Transformer.mtp_head_kv_quant_override = saved_ov;
    transformer_mod.Transformer.mtp_head_kv_quant_override = true;
    const saved_boot = http_server.launch().configured_kv_quant;
    defer http_server.launch().configured_kv_quant = saved_boot;
    http_server.launch().configured_kv_quant = transformer_mod.KVQuantConfig.affine(8);
    var cfg = qwen4ExpOomConfig();
    cfg.mtp_override = null;
    const seq: u64 = 200_000;
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const request_bits: u64 = 4;
    const n: u64 = cfg.attnCacheLayerCount();
    const request_head = kvBytesPerTokenAtBits(cfg.kvBytesPerToken() / n, request_bits);
    try t.expectEqual(@as(u64, 576), request_head);
    try t.expectEqual(@as(u64, 1088), mtpHeadKvBytesPerToken(&cfg));
    const on = prefillRequestTerms(&cfg, seq, max_tokens, request_bits, chunk, .{ .mtp_on = true });
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg)), seq);
    try t.expectEqual(reserved * 1088, on.mtp_head_kv_bytes);
}

test "prefillAdmissionBill: every arch carries the evict-to-admit credits (#492)" {
    // A qwen3_5 100k prompt was refused with GBs of evictable hot cache resident.
    const t = std.testing;
    const MB: u64 = 1024 * 1024;
    var sch: scheduler_mod.Scheduler = undefined;
    sch.resident_hot_cache_bytes = .init(9000 * MB);
    sch.reclaimable_hot_cache_bytes = .init(8000 * MB);
    http_server.launch().global_scheduler = &sch;
    defer http_server.launch().global_scheduler = null;
    for ([_][]const u8{ "qwen3_5", "qwen3_5_moe", "llama", "gemma4", "qwen4_exp" }) |mt| {
        var cfg = qwen4ExpOomConfig();
        cfg.model_type = mt;
        cfg.legacy.caps.long_ctx = std.mem.eql(u8, cfg.model_type, "qwen4_exp");
        const bill = prefillAdmissionBill(&cfg, 100_000, 2048, null, null, .{});
        try t.expectEqual(9000 * MB, bill.evictable);
        try t.expectEqual(8000 * MB, bill.reclaimable);
    }
}

/// The live 364k agent session: `qwen4ExpOomConfig` plus the deployed pack's indexer budget 2048,
/// four indexer heads, affine-4 weights and `--ctx-size 786432`; reproduces the incident's bills to the MB.
pub fn qwen4ExpLive364kConfig() model_mod.ModelConfig {
    var cfg = qwen4ExpOomConfig();
    cfg.pinned_context = 786_432;
    cfg.indexer_budget = 2048;
    cfg.indexer_n_heads = 4;
    cfg.quant_bits = 4;
    cfg.quant_mode = .affine;
    cfg.max_position_embeddings = 1_048_576;
    return cfg;
}

test "a 448k warm append the free-RAM term refused fits under an explicitly raised wired limit" {
    const t = std.testing;
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    const mb: u64 = 1024 * 1024;
    const total_ram: u64 = 131_072 * mb;
    const working_set: u64 = 120_000 * mb;
    const footprint: u64 = 87_722 * mb;
    const saved = memory_ceiling.overrides().wired_limit_mb;
    defer memory_ceiling.overrides().wired_limit_mb = saved;
    memory_ceiling.overrides().wired_limit_mb = 120_000;
    var cfg = qwen4ExpLive364kConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 448_604;
    const max_tokens: u32 = @intCast(getEffectiveContextLength(&cfg) - seq);
    const shared = WarmPrefix{ .matched_tokens = 448_531, .capacity_tokens = 448_531, .will_donate = false };
    const needed = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, 512, shared);
    try t.expectEqual(@as(u64, 11_388), needed / mb);
    try t.expect(needed > memory_ceiling.gpuCeilingWithWiredFloor(working_set, footprint, 10_000 * mb, 0) -| footprint);
    const lifted = memory_ceiling.gpuCeilingWithWiredFloor(working_set, footprint, 12_934 * mb, memory_ceiling.wiredCeilingFloorForRam(&cfg, total_ram));
    try t.expectEqual(@as(u64, 24_086), (lifted -| footprint) / mb);
    try t.expect(needed <= lifted -| footprint);
}

test "a WARM append bills the checkpoints its prefill CAPTURES, not the prompt's" {
    // Turn B: prompt 368,208, restored 364,478 by move, refused by 290 MB at width 512. 1,795 MB
    // of the bill was 32 SSM checkpoints; a warm prefill forwards a 3,730-token tail and captures
    // at most one, and the entry's own checkpoints are already resident and merged at commit.
    const t = std.testing;
    sparse_score.qsa_score_fused_override = false;
    defer sparse_score.qsa_score_fused_override = null;
    const cfg = qwen4ExpLive364kConfig();
    const mb: u64 = 1024 * 1024;
    const per_cp = cfg.ssmCheckpointBytes();
    const seq: u64 = 368_208;
    const matched: u64 = 364_478;
    const chunk: u64 = 512;
    // The stride follows the uncapped base chunk (`prefill_chunk_override`), not the adaptive width.
    const saved_override = generate_mod.prefillCli().override;
    defer generate_mod.prefillCli().override = saved_override;
    generate_mod.prefillCli().override = 8192;
    const stride: u64 = 8192;
    try t.expectEqual(stride, generate_mod.effectiveSsmCheckpointStride(http_server.launch().ssm_checkpoint_stride, 8192));

    // Cold: `seq / stride` saturates at --ssm-checkpoint-max as before.
    try t.expectEqual(@as(u64, http_server.launch().ssm_checkpoint_max) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 0, chunk));
    // Warm: a 3,730-token append at stride 8192 crosses no boundary, so the end snapshot is the whole of it.
    try t.expectEqual(@as(u64, 1) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, matched, chunk));
    // A chain extension long enough to fill the ring still bills the cap: the
    // credit is proportional to the SPAN, not a blanket warm discount.
    try t.expectEqual(@as(u64, http_server.launch().ssm_checkpoint_max) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 60_000, chunk));
    // A span shorter than the backoff window snapshots nothing.
    try t.expectEqual(@as(u64, 0), retainedSsmCheckpointBytes(&cfg, seq, seq - 4, chunk));

    // The bill term by term, from the same inputs `prefillMemoryNeeded` reads.
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 418_224;
    // The restored buffer is smaller than the reservation, so it grows and the rows it holds
    // are still credited. This block keeps the original turn-B decomposition by pricing the
    // SHARED restore of the same span, whose credit is zero for the other reason.
    const turn_b_donating = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = 365_568, .will_donate = true };
    const turn_b = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = 365_568, .will_donate = false };
    const reserve_rows = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    try t.expectEqual(seq + transformer_mod.KVCache.RESERVE_GEN_HEADROOM + chunk, reserve_rows);
    try t.expect(reserve_rows > turn_b.capacity_tokens);
    try t.expectEqual(@as(u64, 0), turn_b.creditedRows(reserve_rows));
    try t.expectEqual(@as(u64, 364_478), turn_b_donating.creditedRows(reserve_rows));

    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, turn_b);
    // On a GDN MoE the envelope is the stream arm, not 3 x MLP.
    const mlp: u64 = 8 * chunk * @max(cfg.hidden_size, prefillFfnWidth(&cfg)) * 2;
    const envelope: u64 = @max(3 * mlp, mlp + chunk * prefillStreamBytesPerToken(&cfg));
    try t.expect(envelope > 3 * mlp);
    const gross =
        seq * kv_per_tok +
        terms.reserved_kv_bytes +
        terms.state_bytes +
        terms.checkpoint_bytes +
        terms.slot_fixed_kv_bytes +
        2 * seq * cfg.num_key_value_heads * cfg.head_dim * 2 +
        envelope +
        PREFILL_RUNTIME_FLOOR_BYTES;
    // No score scratch (hd 256 is fused) and no dequant-GEMM weights (fwd 512
    // is below PREFILL_DQ_GEMM_MIN_M) at this width.
    try t.expect(transformer_mod.prefillHeadDimFused(@intCast(cfg.prefillScoreHeadDim())));
    try t.expect(chunk < transformer_mod.PREFILL_DQ_GEMM_MIN_M);
    const needed = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, turn_b);
    try t.expectEqual(gross * 5 / 4 + qsaMaskBytes(&cfg, chunk, seq), needed);

    // AND THE VERDICT MOVES. 10,774 MB refused against 10,000 MB available;
    // the honest checkpoint term brings it to 9,722 MB, so the turn is served.
    const available: u64 = 10_000 * mb;
    try t.expect(needed < available);
    try t.expectEqual(@as(u64, 9_722), needed / mb);
    const before = gross - terms.checkpoint_bytes + @as(u64, http_server.launch().ssm_checkpoint_max) * per_cp;
    try t.expectEqual(@as(u64, 10_774), (before * 5 / 4 + qsaMaskBytes(&cfg, chunk, seq)) / mb);
    try t.expect((before * 5 / 4 + qsaMaskBytes(&cfg, chunk, seq)) > available);
}
