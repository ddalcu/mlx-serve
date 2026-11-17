//! The prefill working-set bill both an engine and a plugin price from plain numbers (`prefillMemoryNeeded`).

const prefillHeadDimFused = @import("../blocks/attention/attn256.zig").prefillHeadDimFused;
const PREFILL_DQ_GEMM_MIN_M = @import("../blocks/linear/prefill_dq_gemm.zig").PREFILL_DQ_GEMM_MIN_M;

pub fn kvBytesPerTokenAtBits(dense: u64, kv_bits: u64) u64 {
    if (kv_bits >= 16) return dense;
    return dense * (2 * kv_bits + 1) / 32;
}

/// The chunk-independent floor every prefill pays: MLX runtime scratch, the
/// KV cache's proportional capacity growth (old + new buffer coexist across a
/// grow), and the graph live-set no other term models. MEASURED as the
/// intercept of peak-above-steady-state against the chunk, on five checkpoints
/// spanning 2.6B to 30B (M4 Max): 0.39 GB (qwen3_5 27B), 0.67
/// (qwen3_5 4B), 0.81 (lfm2 2.6B), ~0.1 (gemma4 26B-A4B), 1.27 (muse 30B). It
/// does NOT scale with the weights — a dense bf16 checkpoint pays it too — so
/// it is billed as what it measures as: a constant.
pub const PREFILL_RUNTIME_FLOOR_BYTES: u64 = 512 * 1024 * 1024;

/// Per-request bytes that scale with the prompt rather than the chunk.
pub const PrefillRequestTerms = struct {
    /// Dense operands of one caching layer; null retains the generic GQA width.
    dequant_width_per_token: ?u64 = null,
    /// Cache capacity reserved at admission beyond `seq * kv_per_tok`: the generation headroom
    /// plus a chunk of slack, so no buffer grows during the prefill (#353).
    reserved_kv_bytes: u64 = 0,
    /// Per-request state the arch keeps outside the KV cache (the qwen4_exp QSA indexer history), at its reserved size.
    state_bytes: u64 = 0,
    /// SSM checkpoints the prefill is still holding when it reaches its last chunk.
    checkpoint_bytes: u64 = 0,
    /// Bytes of the KV terms above already resident in the buffer the restore handed this slot
    /// (`WarmPrefix`). KV only; subtracted once, in `prefillMemoryNeeded`.
    shared_resident_bytes: u64 = 0,
    /// Old KV buffers that coexist with the new ones while a warm append GROWS the cache.
    /// Zero when nothing grows and when the restore was shared (the whole copy is billed
    /// uncredited instead).
    grow_coexist_bytes: u64 = 0,
    /// KV the slot holds whatever its length (`slotFixedKvBytes`).
    slot_fixed_kv_bytes: u64 = 0,
    mtp_head_kv_bytes: u64 = 0,
    /// Forward width from which the dequant+GEMM route fires (`prefillDqGemmMinRows`).
    dq_min_rows: u64 = PREFILL_DQ_GEMM_MIN_M,
};

pub fn prefillMemoryNeeded(seq: u64, heads: u64, kv_heads: u64, kv_per_tok: u64, hdim: u64, score_hdim: u64, hidden: u64, ffn: u64, kv_bits: u64, chunk: u64, attn_keys: u64, stream_per_tok: u64, dequant_weights: u64, req: PrefillRequestTerms) u64 {
    // A forward is never wider than the prompt: a prompt shorter than the
    // chunk runs ONE seq-wide forward, so the per-chunk transient envelope
    // scales with min(chunk, seq) — billing the raw chunk cap rejected a
    // 31-token prompt for "needing" 6.2 GB on a memory-squeezed hy_v3 host
    // No-op for prompts >= one chunk.
    const fwd: u64 = @min(chunk, @max(seq, 1));
    const kv_bytes: u64 = seq * kvBytesPerTokenAtBits(kv_per_tok, kv_bits);
    const scores: u64 = if (!prefillHeadDimFused(@intCast(score_hdim))) heads * fwd * @min(attn_keys, seq) * 2 else 0;
    const dequant: u64 = if (kv_bits < 16) (if (req.dequant_width_per_token) |width| seq * width else 2 * seq * kv_heads * hdim * 2) else 0;
    const mlp: u64 = 8 * fwd * @max(hidden, ffn) * 2;
    // The MLP envelope alone is not the whole per-token working set: an arch
    // with its own prefill streams (linear-attention layers, MoE gather-sort
    // expansion — `prefillStreamBytesPerToken`) adds to it. Floored at the
    // historical 3-envelope bill so no arch's admission ever LOOSENS: that
    // figure is measured 1.7x-5x conservative on plain attention archs (muse,
    // lfm2) and would be a 33% UNDER-bill on a GatedDeltaNet hybrid.
    const envelope: u64 = @max(3 * mlp, mlp + fwd * stream_per_tok);
    // The dequant+GEMM prefill route (transformer.prefillDqGemm) materializes
    // a bf16 copy of the weight it is about to multiply plus its transpose,
    // and only at forwards wide enough to take that route. Measured
    // against the stock route: +0.51 GB on the qwen3_5 27B at chunk
    // 2048, +0.17-0.21 on the lfm2 2.6B, ~0 at chunk 8192 where the envelope
    // already dominates.
    const dq_weights: u64 = if (fwd >= req.dq_min_rows) dequant_weights else 0;
    const gross: u64 = kv_bytes + req.reserved_kv_bytes + req.state_bytes + req.checkpoint_bytes +
        req.grow_coexist_bytes + req.slot_fixed_kv_bytes + req.mtp_head_kv_bytes + scores + dequant + envelope + dq_weights + PREFILL_RUNTIME_FLOOR_BYTES;
    // The one place a warm turn's resident rows leave the bill, inside the 5/4.
    return (gross -| req.shared_resident_bytes) * 5 / 4;
}
