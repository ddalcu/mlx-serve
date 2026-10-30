//! defaultKvBits, prefillTransientReserve; aneGateHeadroom; slotFixedKvBytes .. prefillMemoryNeeded; prefill streams,
//! state, MTP head, session bytes; prefillRequestTerms

const std = @import("std");
const transformer_mod = @import("../transformer.zig");
const generate_mod = @import("../generate.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const ane_mod = @import("../../../sdk/ane/offload.zig");
const http_server = @import("../../../host/http/server.zig");

const configuredKvQuantFor = @import("../../../host/http/server.zig").configuredKvQuantFor;
const resolvedPrefixCacheMem = @import("hot_cache_plan.zig").resolvedPrefixCacheMem;
const getEffectiveContextLength = @import("context.zig").getEffectiveContextLength;
const qsaMaskBytes = @import("../../../plugins/qwen4_exp/bill.zig").qsaMaskBytes;

/// The process-wide KV width a request gets when it names none — the same
/// expression `checkAttentionMemory` falls back to, so the sizer and the
/// admission guard read one answer.
///
/// Through `configuredKvQuant`, so a load-time bill gets the boot's `--kv-quant`, not the dense fallback.
pub fn defaultKvBits(config: *const model_mod.ModelConfig) u64 {
    const cfg: transformer_mod.KVQuantConfig = configuredKvQuantFor(config);
    return if (cfg.scheme == .off) 16 else cfg.bits;
}

/// The prefill working set that is NOT the KV cache: score scratch, the
/// quantized-KV dequant, and the QKV/MLP transient envelope — all bounded by
/// the prefill CHUNK, at the widest chunk any prompt on this model can run.
///
/// It is a one-off reserve, not a per-token cost. `computeMemoryContext` used
/// to bill the MLP envelope (`8 × max(hidden, ffn) × 2`) against every token of
/// CONTEXT: 272 KB/token beside a 64 KB/token KV cache on a qwen3_5 27B, i.e.
/// 81% of the budget spent on a transient that never scales with the context
/// length, which caps a 16 GB Mac under 4k tokens whatever the weights cost.
/// Measured on the shipped 4-bit 27B: the peak above steady state
/// is flat in prompt length and tracks the chunk — 3.34 GB at chunk 2048 for
/// prompts from 3k to 51k tokens.
///
/// Same estimator as the admission guard with the KV term zeroed, because the
/// KV cache is exactly what the sizer is solving for.
pub fn prefillTransientReserve(config: *const model_mod.ModelConfig, kv_bits: u64, chunk: u64) u64 {
    return prefillTransientReserveAtKv(config, kv_bits, chunk, chunk);
}

/// `prefillTransientReserve` with the KV length made explicit. The load-time sizer asks at
/// `kv = chunk`; a per-chunk decision has a KV, and the attention score sheet and the QSA
/// sheet are linear in it (~0.5 GB under-billed at the default sheet, ~9 GB at 4096 MB).
pub fn prefillTransientReserveAtKv(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    chunk: u64,
    kv_len: u64,
) u64 {
    const seq: u64 = @max(kv_len, chunk);
    // The arch that bills its own prompt pass is priced the same way here as at admission (`prefillNeededAtChunk`).
    if (config.isDsv41()) return @import("prefill_chunk.zig").dsv41PrefillMemoryNeeded(config, seq, chunk);
    return prefillMemoryNeeded(
        seq,
        config.num_attention_heads,
        config.num_key_value_heads,
        0,
        config.head_dim,
        config.prefillScoreHeadDim(),
        config.hidden_size,
        prefillFfnWidth(config),
        kv_bits,
        chunk,
        config.prefillAttnKeys(seq),
        prefillStreamBytesPerToken(config),
        prefillDequantWeightBytes(config),
        .{ .slot_fixed_kv_bytes = slotFixedKvBytes(config, kv_bits, chunk), .dq_min_rows = transformer_mod.prefillDqGemmMinRows(config.quant_bits), .dequant_width_per_token = if (config.isGlm5()) glm5DequantWidth(config) else null },
    ) + qsaMaskBytes(config, @min(chunk, @max(seq, 1)), seq);
}

/// The ANE admission gate's headroom for THIS model at THIS chunk: the KV
/// cache for a context worth serving, the hot prefix cache, the prefill
/// chunk's transient envelope, and the non-model baseline. Every term is one
/// the auto-context sizer also reserves — the gate must not admit an offload
/// into memory the sizer has already spoken for.
///
/// Both model terms already exist as estimators — this is the "a gate that
/// runs BEFORE the estimator that knows better is the estimator" rule applied
/// to `ane.gateAllows`, which carried a flat 12 GB instead. That constant was
/// calibrated for the 27B at chunk 8192 and was wrong in BOTH directions once
/// `resolvePrefillChunk` sized the chunk down: measured on that geometry, the
/// envelope is 12.66 GB at chunk 8192 but 2.13 GB at chunk 1024.
///
/// The KV term is a RESERVE, not a prediction: see `ane.MIN_CONTEXT_TOKENS`.
pub fn aneGateHeadroom(config: *const model_mod.ModelConfig, chunk: u32) u64 {
    const kv_bits: u64 = defaultKvBits(config);
    const ctx: u64 = if (config.max_position_embeddings > 0)
        @min(ane_mod.MIN_CONTEXT_TOKENS, config.max_position_embeddings)
    else
        ane_mod.MIN_CONTEXT_TOKENS;
    return ane_mod.GATE_BASELINE_BYTES +|
        (kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) *| ctx) +|
        resolvedPrefixCacheMem() +|
        prefillTransientReserve(config, kv_bits, chunk);
}

/// Pure memory model behind checkAttentionMemory. All quantities in bytes,
/// 25% safety margin included. `chunk` must be the SAME prefill chunk
/// generate.zig will pick (generate_mod.effectivePrefillChunk) so the
/// admission guard and the real prefill cannot drift. Terms:
///   - kv: persistent for the request; billed at the ACTIVE kv-quant width
///     (+0.5 bit/elem group scale/bias overhead when quantized), not a
///     hardcoded fp16.
///   - scores: the composed-SDPA scratch [heads, chunk, seq] — materialized
///     only for head_dims no fused kernel covers (transformer.
///     prefillHeadDimFused: <= 128 via MLX, 256 via msv_attn_p256 — unfused
///     only for an exotic dim).
///     Bounded to ~one layer by the adaptive eval cadence
///     (transformer.prefillEvalCadence).
///   - dequant: dense-fp16 rebuild of the FULL quantized cache each layer
///     (KVCache.denseView under --kv-quant).
///   - mlp: QKV/MLP transient envelope; ~3 layers coexist between eval
///     points, and it is CHUNK-bounded, never seq-scaled — the old
///     8×seq×ffn envelope over-billed a 255K prompt by ~60 GB and rejected
///     requests that fit comfortably.
///
/// `attn_keys` is how many keys one query actually reads (config.
/// prefillAttnKeys): `seq` for dense causal attention, a much smaller bound for
/// a sparse arch. It scales the SCORE term only — KV/dequant are storage, and
/// a sparse reader still stores every latent it might later attend to.
/// `kv_per_tok` is the whole-model DENSE (f16) KV bytes for one token —
/// `ModelConfig.kvBytesPerToken()`. It replaced a `layers` parameter because
/// neither the layer count nor the per-head width is uniform once an arch
/// interleaves linear-attention layers or stores keys wider than values.
///
/// `hdim` and `score_hdim` are two DIFFERENT widths and only coincide on a
/// symmetric arch. `hdim` is the STORED width the quantized-KV dequant
/// transient is read at; `score_hdim` is the width the score is contracted at
/// (`ModelConfig.prefillScoreHeadDim`), which is what decides whether a fused
/// kernel exists — MLA scores over 192 while storing values at 128, and billing
/// the score term against `head_dim` 128 declared it fused and billed ZERO for
/// a composed path that really does materialize [heads, chunk, seq].
/// KV bytes for ONE token at the width the cache actually STORES. `dense` is
/// `ModelConfig.kvBytesPerToken()` — the arch's own caching-layer count and its
/// own K and V widths, at fp16. A quantized cache stores `kv_bits` per element
/// plus the group scale/bias pair, which is the `+1` in `(2·bits + 1)/32`
/// (~0.5 bit/elem at group 64 with bf16 params).
///
/// Both the auto-context sizer and the prefill admission guard bill through
/// here. The sizer used to bill fp16 unconditionally, so a `--kv-quant 4`
/// server reported — and served — under a third of the context it can hold.
/// An expert-bank pack's slot rows per routed layer when the model sets none (mlx-stream's fill): the rows the GPU
/// ceiling holds once the residents, the stream's transient rows, the KV of the model's context (else the standard
/// 16K prompt and 1K generation), one prompt chunk's transients and the wired margin are billed; every expert at most,
/// `num_experts_per_tok` at least.
pub fn expertBankFillRows(config: *const model_mod.ModelConfig) u32 {
    const dir = config.pack_dir orelse return config.num_experts;
    const streamed = @import("../../../sdk/weights/streamed.zig").Streamed;
    const io = std.Io.Threaded.global_single_threaded.io();
    const a = std.heap.page_allocator;
    const one = streamed.bankBytes(a, io, dir, 1) catch return config.num_experts;
    const two = streamed.bankBytes(a, io, dir, 2) catch return config.num_experts;
    const per_row = two - one;
    const kv_bits = defaultKvBits(config);
    const manual: u64 = http_server.manualContext(config);
    const ctx: u64 = if (manual > 0) manual else 16_384 + 1_024;
    const chunk: u64 = generate_mod.WIDE_PREFILL_CHUNK;
    const reserve = @import("../runtime/preflight.zig").modelDiskBytes(io, dir) +| (one - per_row) +|
        sessionBytesPerToken(config, kv_bits) *| ctx +| slotFixedKvBytes(config, kv_bits, chunk) +|
        prefillTransientReserve(config, kv_bits, chunk) +| http_server.serverState().wired_limit_margin_bytes;
    const rows = (http_server.staticGpuMemoryCeiling() -| reserve) / @max(per_row, 1);
    return @intCast(std.math.clamp(rows, @max(config.num_experts_per_tok, 1), config.num_experts));
}

/// KV every live slot holds whatever its length: the QSA key ring and the sliding rings
/// (`ModelConfig.slidingRing`), whose buffers hold keep + one prefill chunk of rows.
pub fn slotFixedKvBytes(config: *const model_mod.ModelConfig, kv_bits: u64, chunk: u64) u64 {
    const ring_rows: u64 = @as(u64, config.slidingKeepRows()) +| chunk;
    return config.qsaRingBytes() +| kvBytesPerTokenAtBits(config.slidingRowBytes(), kv_bits) *| ring_rows;
}

const sdk_bill = @import("../../../sdk/bill/prefill.zig");
pub const kvBytesPerTokenAtBits = sdk_bill.kvBytesPerTokenAtBits;
pub const PREFILL_RUNTIME_FLOOR_BYTES = sdk_bill.PREFILL_RUNTIME_FLOOR_BYTES;
pub const PrefillRequestTerms = sdk_bill.PrefillRequestTerms;
pub const prefillMemoryNeeded = sdk_bill.prefillMemoryNeeded;


/// How many MoE layers' gather-sort expansions coexist between eval points.
/// Mirrors `Transformer.MOE_EVAL_EVERY_N_LAYERS` — the MoE prefill loop evals
/// every 4th layer, so 4 layers' worth is what the peak holds.
const MOE_PREFILL_COEXIST: u64 = transformer_mod.Transformer.PREFILL_EVAL_CADENCE_DEFAULT;

/// What a warm turn already holds. A RAM hot-cache restore hands the slot the entry's KV
/// buffers without allocating another copy, so `available` is already net of them and
/// billing them again refused a 786k warm turn whose own entry was resident. KV only: the
/// QSA history is privatised on restore. Credited only while the prefix fits the destination
/// capacity (a grow allocates a fresh buffer beside the protected entry).
pub const WarmPrefix = struct {
    /// Prompt rows the hot cache restored. 0 = cold.
    matched_tokens: u64 = 0,
    /// Tokens of capacity the restored buffers hold (`KVCache.residentCapacityTokens`). The
    /// connection thread passes 0 and defers the warm decision to the inference thread.
    capacity_tokens: u64 = 0,
    /// Did the restore check out its entry (`prefix_cache.checkoutEligible`)? A shared restore
    /// is copied whole by the first append, so crediting it is an under-bill.
    will_donate: bool = false,
    mtp_on: bool = false,

    /// Rows of KV this request will not allocate: the rows the restored buffer already holds.
    /// Capacity is a ceiling on the credit, never a switch: a growing chain outruns its capacity
    /// on nearly every turn. The grow's own cost is `reserved - credited` plus
    /// `PrefillRequestTerms.grow_coexist_bytes`. `will_donate` gates the whole credit.
    pub fn creditedRows(self: WarmPrefix, reserved: u64) u64 {
        if (!self.will_donate) return 0;
        if (self.matched_tokens == 0) return 0;
        return @min(@min(self.matched_tokens, self.capacity_tokens), reserved);
    }

    /// Does this append grow the buffers at all? The per-layer grow fires on `offset + new_len
    /// > capacity`; a reservation above the prompt allocates nothing by itself.
    pub fn grows(self: WarmPrefix, seq: u64) bool {
        if (!self.will_donate or self.matched_tokens == 0) return false;
        return seq > self.capacity_tokens;
    }
};

/// PURE: the most KV-caching layers that fall inside one window of `window` consecutive layers,
/// i.e. how many old buffers can be alive at once at that eval cadence.
pub fn attnLayersPerEvalWindow(config: *const model_mod.ModelConfig, window: u32) u32 {
    const n = config.num_hidden_layers;
    if (window == 0 or n == 0) return 0;
    if (window >= n) return config.attnCacheLayerCount();
    var best: u32 = 0;
    var start: u32 = 0;
    while (start + window <= n) : (start += 1) {
        var count: u32 = 0;
        var i: u32 = start;
        while (i < start + window) : (i += 1) {
            if (!config.isLinearLayer(i)) count += 1;
        }
        if (count > best) best = count;
    }
    return @min(best, config.attnCacheLayerCount());
}

/// PURE: bytes of OLD KV buffer alive beside the new ones while a warm append grows the cache.
/// `growQuantBuf` drops our reference to the old buffer at once, but the graph node holds it
/// until it EVALUATES, so the coexistence window is the prefill loop's eval cadence rather than
/// the size of the cache. A forward narrower than `prefillEvalCadenceApplies` runs one eval for
/// the whole layer loop and therefore does pay the whole old cache.
fn growCoexistBytes(config: *const model_mod.ModelConfig, warm: WarmPrefix, seq: u64, kv_per_tok: u64) u64 {
    if (!warm.grows(seq)) return 0;
    const attn = config.attnCacheLayerCount();
    if (attn == 0) return 0;
    const span: u64 = seq -| warm.matched_tokens;
    const window: u32 = if (transformer_mod.Transformer.prefillEvalCadenceApplies(@intCast(@min(span, 1 << 20))))
        attnLayersPerEvalWindow(config, transformer_mod.Transformer.MOE_EVAL_EVERY_N_LAYERS)
    else
        attn;
    if (window == 0) return 0;
    // The old buffer's own size, spread over the layers that cache.
    return warm.capacity_tokens *| kv_per_tok / attn *| window;
}


/// `glmDsaWith` reads the old indexer tail before append, then the new latent + indexer.
/// The unused old latent is lazy and never materialized. Both indexer views may coexist.
pub fn glm5DequantWidth(config: *const model_mod.ModelConfig) u64 {
    return (@as(u64, config.dsa_kv_lora_rank) + 4 * @as(u64, config.dsa_index_head_dim)) * 2;
}


/// Per-TOKEN bytes of prefill working set that live OUTSIDE the MLP envelope,
/// because the arch runs streams the envelope does not model. MEASURED as the
/// slope of peak-above-steady-state against the prefill chunk (M4 Max,
/// one request per boot, byte-identical across repeats):
///
/// | checkpoint | measured B/chunk-token | one MLP envelope | this term |
/// |---|---|---|---|
/// | muse_glimmer 30B (dense) | 190,300 | 319,488 | 0 |
/// | lfm2 2.6B (conv hybrid) | 171,150 | 172,032 | 0 |
/// | qwen3_5 4B (GatedDeltaNet) | 513,500 | 147,456 | 393,216 |
/// | qwen3_5 27B (GatedDeltaNet) | 1,263,700 | 278,528 | 983,040 |
/// | gemma4 26B-A4B (MoE) | 486,800 | 90,112 | 450,560 |
///
/// Two families need it. A linear-attention hybrid holds one chunk-wide q/k/v
/// stream per LINEAR layer — all of them, not the ~3 the eval cadence bounds
/// (48 x 10240 elems on the 27B, 24 x 8192 on the 4B, both landing within 7%
/// of the measured excess over the envelope). A MoE prefill sorts and gathers,
/// which replicates the hidden stream `top_k` times per layer alongside the
/// expert rows, for the 4 layers the MoE eval cadence lets coexist.
///
/// The linear term bills only `cadence + 1` layers: since the conv_state tail is
/// an owned copy evaluated at each cadence point, older layers' chunk inputs are
/// released inside the loop (the old all-layers figure WAS that pin).
///
/// Attention-only archs return 0 and keep exactly the bill they had.
pub fn prefillStreamBytesPerToken(config: *const model_mod.ModelConfig) u64 {
    return prefillStreamBytesPerTokenOn(config, transformer_mod.verifyQmmNaxAvailable());
}

pub fn prefillStreamBytesPerTokenOn(config: *const model_mod.ModelConfig, nax: bool) u64 {
    // GLM-5 on NAX: the sorted expert gather reads tokens through its row map (no top_k
    // replica) and the per-core KDA recurrence holds no per-layer q/k/v stream.
    if (nax and config.isGlm5()) return 0;
    var per_tok: u64 = 0;
    const linear_layers: u64 = @as(u64, config.num_hidden_layers) -| config.attnCacheLayerCount();
    if (config.linear_num_value_heads > 0 and linear_layers > 0) {
        const qkv: u64 = 2 * @as(u64, config.linear_num_key_heads) * config.linear_key_head_dim +
            @as(u64, config.linear_num_value_heads) * config.linear_value_head_dim;
        const cadence_window: u64 = @as(u64, transformer_mod.Transformer.PREFILL_EVAL_CADENCE_DEFAULT) + 1;
        per_tok += @min(linear_layers, cadence_window) * qkv * 2;
    }
    if (config.isMoe()) {
        const top_k: u64 = @max(@as(u64, config.num_experts_per_tok), 1);
        // Puzzle experts contract in the latent width. The router and the
        // shared expert stay on hidden_size and are not this replica.
        const expert_in: u64 = if (config.moe_latent_size > 0) config.moe_latent_size else config.hidden_size;
        per_tok += MOE_PREFILL_COEXIST * top_k * 2 *
            (expert_in + config.moe_intermediate_size) * 2;
    }
    return per_tok;
}

/// The dequantized-weight working set `transformer.prefillDqGemm` materializes
/// (a bf16 copy of the weight plus its transpose, ~3 alive across the layer
/// loop). Chunk-INDEPENDENT — it is weights, not activations — and billed only
/// where the route can fire: affine-quantized weights, and (inside
/// `prefillMemoryNeeded`) forwards at least `PREFILL_DQ_GEMM_MIN_M` wide.
/// Reads the same kill switch the route does, so an A/B with the route off
/// does not get billed for it.
pub fn prefillDequantWeightBytes(config: *const model_mod.ModelConfig) u64 {
    if (config.quant_bits == 0 or config.quant_mode != .affine) return 0;
    if (!transformer_mod.prefillDqGemmEnabled()) return 0;
    return 3 * prefillFfnWidth(config) * @as(u64, config.hidden_size) * 2;
}

pub fn prefillFfnWidth(config: *const model_mod.ModelConfig) u64 {
    const per_tok: u64 = @max(@as(u64, config.num_experts_per_tok), 1);
    const moe_w: u64 = @as(u64, config.moe_intermediate_size) * per_tok +
        config.shared_expert_intermediate_size;
    const dense_w: u64 = if (config.intermediate_size_declared) config.intermediate_size else 0;
    const w = @max(dense_w, moe_w);
    return if (w > 0) w else config.intermediate_size;
}

/// Estimate peak GPU memory for prefill and reject if it would exceed the Metal
/// working-set ceiling. Exceeding it throws an uncatchable C++ exception on a
/// Metal completion-handler thread and kills the process, so PREVENTION is the
/// only lever — this guard must track what the prefill actually allocates
/// (prefillMemoryNeeded above; chunk choice from generate.effectivePrefillChunk).
/// `kv_override` is the per-request `kv_quant` body field where the surface
/// parses one (chat/messages/responses); null falls back to the process default.
/// `lm` is the resolved model: an embedded-engine model skips this guard
/// entirely (see `mlxMemoryGuardApplies`) — this is the single chokepoint, so
/// every current and future call site is covered without per-site gating.
/// Per-token bytes of the arch's out-of-cache request state as billed: the QSA indexer history
/// at the copies a live slot holds (one; two without history sharing), plus the f32 block-score bank the two-copy slack used to
/// hide. One helper because the auto-context sizer and the admission guard must agree.
pub fn statePerTokenBilled(config: *const model_mod.ModelConfig) u64 {
    const one = config.qsaHistoryBytesPerToken();
    if (one == 0) return 0;
    const copies: u64 = if (transformer_mod.qsaHistoryShareEnabled()) 1 else 2;
    return one * copies + config.qsaScoreBankBytesPerToken();
}

/// Tokens of cache capacity a request reserves at admission (prompt + generation + one chunk
/// of slack); `KVCache.reserve_tokens` allocates it on the first grow so the prefill never
/// runs a second grow. The guard and the allocator are the same number by construction.
pub fn reservedCacheTokens(seq: u64, max_tokens: u64, chunk: u64, ctx: u64) u64 {
    return transformer_mod.KVCache.reservedTokens(seq, max_tokens, chunk, ctx);
}

/// Bytes of SSM checkpoints a prefill is still holding at its last chunk (~1.9 GB on a 36-layer
/// GDN trunk, billed at zero before #353). Mirrors the prefill loop's stride. `matched` is the
/// restored prefix: a warm prefill captures only over `seq - matched` (the entry's own
/// checkpoints are already resident and merged, never re-allocated).
pub fn retainedSsmCheckpointBytes(config: *const model_mod.ModelConfig, seq: u64, matched: u64, chunk: u64) u64 {
    const per_cp = config.ssmCheckpointBytes();
    if (per_cp == 0 or http_server.launch().ssm_checkpoint_stride == 0) return 0;
    const stride: u64 = generate_mod.effectiveSsmCheckpointStride(
        http_server.launch().ssm_checkpoint_stride,
        @max(@as(usize, @intCast(chunk)), generate_mod.prefillCli().override),
    );
    if (stride == 0) return 0;
    // Stride-aligned capture positions in `(matched, seq]` plus the always-on end-of-prompt
    // snapshot, the only capture a warm append shorter than one stride makes.
    const span: u64 = seq -| @min(matched, seq);
    var n: u64 = seq / stride -| @min(matched, seq) / stride;
    if (span > generate_mod.SSM_SNAPSHOT_BACKOFF) n += 1;
    if (http_server.launch().ssm_checkpoint_max > 0) n = @min(n, http_server.launch().ssm_checkpoint_max);
    return n * per_cp;
}

/// Per-token bytes of the qwen4 MTP head's own KV at the scheme it was LOADED with,
/// billed once per request when MTP is on. It was resident and unbilled before.
pub fn mtpHeadKvBytesPerToken(config: *const model_mod.ModelConfig) u64 {
    if (!config.longCtxGated()) return 0;
    const n = config.attnCacheLayerCount();
    if (n == 0) return 0;
    const head_cfg = transformer_mod.Transformer.qwen4MtpHeadKvConfig(configuredKvQuantFor(config));
    const bits: u64 = if (head_cfg.scheme == .off) 16 else head_cfg.bits;
    return kvBytesPerTokenAtBits(config.kvBytesPerToken() / n, bits);
}

fn mtpHeadQsaRingBytes(config: *const model_mod.ModelConfig) u64 {
    if (!config.longCtxGated()) return 0;
    const n = config.attnCacheLayerCount();
    if (n == 0) return 0;
    return config.qsaRingBytes() / n;
}

fn mtpHeadStateBytesPerToken(config: *const model_mod.ModelConfig) u64 {
    if (!config.longCtxGated()) return 0;
    const n = config.attnCacheLayerCount();
    if (n == 0) return 0;
    return statePerTokenBilled(config) / n;
}

/// A loaded head drafts by default, so its KV is billed unless the model opts out.
pub fn mtpHeadBilled(config: *const model_mod.ModelConfig) bool {
    return config.mtp_override != false;
}

pub fn sessionBytesPerToken(config: *const model_mod.ModelConfig, kv_bits: u64) u64 {
    const head: u64 = if (mtpHeadBilled(config)) mtpHeadKvBytesPerToken(config) +| mtpHeadStateBytesPerToken(config) else 0;
    return kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(config) +| head;
}

/// The per-request terms of the admission bill, in one place.
pub fn prefillRequestTerms(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u64, kv_bits: u64, chunk: u64, warm: WarmPrefix) PrefillRequestTerms {
    // Arch gate for the reservation and state terms (measured on qwen4_exp alone; the
    // reservation's allocator twin is gated too, so an ungated guard billed memory never reserved).
    const dq_min_rows: u64 = transformer_mod.prefillDqGemmMinRows(config.quant_bits);
    if (!config.longCtxGated()) {
        // A checked-out entry is the slot's own buffer on every arch; both terms are zero for a share.
        const kv_per_tok = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits);
        return .{
            .shared_resident_bytes = warm.creditedRows(seq) *| kv_per_tok,
            .grow_coexist_bytes = growCoexistBytes(config, warm, seq, kv_per_tok),
            .slot_fixed_kv_bytes = slotFixedKvBytes(config, kv_bits, chunk),
            .dq_min_rows = dq_min_rows,
        };
    }
    // `reservedTokens` returns 0 below its length threshold; floor the reserved length at `seq`.
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(config)), seq);
    // Only the headroom is new here: the prompt's own rows are already billed.
    const kv_per_tok = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits);
    const mtp_on = warm.mtp_on;
    const head_per_tok: u64 = if (mtp_on) mtpHeadKvBytesPerToken(config) else 0;
    const head_qsa_ring: u64 = if (mtp_on) mtpHeadQsaRingBytes(config) else 0;
    const head_state_per_tok: u64 = if (mtp_on) mtpHeadStateBytesPerToken(config) else 0;
    // Rows the restore hands over without allocating another copy. KV only: the history is
    // privatised on restore (`seedCapBuf`), so crediting it cancelled the one-copy bill outright.
    const credited = warm.creditedRows(reserved);
    return .{
        .reserved_kv_bytes = (reserved -| seq) * kv_per_tok,
        .state_bytes = reserved * (statePerTokenBilled(config) +| head_state_per_tok),
        // The warm span, not the prompt; read regardless of `will_donate` (a shared restore skips the same rows).
        .checkpoint_bytes = retainedSsmCheckpointBytes(config, seq, warm.matched_tokens, chunk),
        .shared_resident_bytes = credited *| kv_per_tok,
        .grow_coexist_bytes = growCoexistBytes(config, warm, seq, kv_per_tok),
        .slot_fixed_kv_bytes = slotFixedKvBytes(config, kv_bits, chunk) +| head_qsa_ring,
        .mtp_head_kv_bytes = reserved *| head_per_tok,
        .dq_min_rows = dq_min_rows,
    };
}
