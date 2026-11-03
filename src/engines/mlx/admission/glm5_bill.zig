//! GLM-5.3-Flash's prefill bill (`glm5PrefillMemoryNeeded`): prompt KV, the f32 KDA checkpoints and live state, one
//! DSA layer's scratch and indexer pools, decode growth on the KV reserve's headroom. The engine's admission calls it
//! until `Bill.prefill_bytes` carries max_tokens and the warm prefix.

const std = @import("std");
const cache_mod = @import("../kv/cache.zig");
const prefill_dq_gemm = @import("../../../sdk/blocks/linear/prefill_dq_gemm.zig");
const verify_qmm_nax = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig");
const generate_mod = @import("../generate.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const http_server = @import("../../../host/http/server.zig");

const getEffectiveContextLength = @import("context.zig").getEffectiveContextLength;
const glm5TestConfig = @import("bills_test.zig").glm5TestConfig;
const slotFixedKvBytes = @import("bills.zig").slotFixedKvBytes;
const kvBytesPerTokenAtBits = @import("bills.zig").kvBytesPerTokenAtBits;
const PREFILL_RUNTIME_FLOOR_BYTES = @import("bills.zig").PREFILL_RUNTIME_FLOOR_BYTES;
const WarmPrefix = @import("bills.zig").WarmPrefix;
const glm5DequantWidth = @import("bills.zig").glm5DequantWidth;
const prefillStreamBytesPerToken = @import("bills.zig").prefillStreamBytesPerToken;
const prefillDequantWeightBytes = @import("bills.zig").prefillDequantWeightBytes;
const prefillFfnWidth = @import("bills.zig").prefillFfnWidth;
const retainedSsmCheckpointBytes = @import("bills.zig").retainedSsmCheckpointBytes;
const prefillNeededAtChunk = @import("prefill_chunk.zig").prefillNeededAtChunk;

/// Bytes per (query, key pool) GLM's DSA indexer holds while it selects (`selectRows`):
/// f32 scores, their negation and the argpartition result.
const GLM5_INDEXER_BYTES_PER_POOL: u64 = 12;

/// GLM-5's own bill. Prefill evals every 4 layers and each window holds one DSA layer, so the
/// prefill scratch is one layer's; decode grows the cache after the prefill is gone, so the
/// two are a max, not a sum.
pub fn glm5PrefillMemoryNeeded(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u64, kv_bits: u64, chunk: u64, warm: WarmPrefix) u64 {
    const ctx: u64 = getEffectiveContextLength(config);
    // Generation growth is billed to the KV reserve's headroom, as the drafter term is: an omitted
    // max_tokens is the rest of the window (1M here), and pricing that answer's KV refused a
    // 431-token prompt at 43 GB beside 36 GB free.
    const output = @min(max_tokens, cache_mod.KVCache.RESERVE_GEN_HEADROOM, if (ctx > 0) ctx -| seq else max_tokens);
    const end = seq +| output;
    const kv_tok = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits);
    const view_tok: u64 = if (kv_bits < 16) glm5DequantWidth(config) else 0;
    const fwd = @min(chunk, @max(seq -| warm.matched_tokens, 1));
    const nax = verify_qmm_nax.verifyQmmNaxAvailable();

    // Prompt KV, the f32 KDA checkpoints, the live state with two lag snapshots and a capture.
    const resident = seq *| kv_tok +| retainedSsmCheckpointBytes(config, seq, warm.matched_tokens, chunk) +|
        4 *| config.ssmCheckpointBytes() +| slotFixedKvBytes(config, kv_bits, chunk);

    const mlp: u64 = 8 * fwd * @max(@as(u64, config.hidden_size), prefillFfnWidth(config)) * 2;
    const envelope = @max(3 * mlp, mlp + fwd * prefillStreamBytesPerToken(config));
    const dq_weights = if (fwd >= prefill_dq_gemm.prefillDqGemmMinRows(config.quant_bits)) prefillDequantWeightBytes(config) else 0;
    // Generated tokens decode one row at a time and never build the score sheet: prompt pools only.
    const pools = seq / @max(@as(u64, config.dsa_index_kpool), 1);
    const indexed = nax and config.dsa_index_head_dim == 128 and config.hadamard_block == 0;
    // Off NAX the composed scores add three [S, H, P] f32 sheets (`maskedScores`).
    const indexer = fwd *| pools *| (GLM5_INDEXER_BYTES_PER_POOL + if (indexed) @as(u64, 0) else 12 * @as(u64, config.dsa_index_heads));
    // The fallback gathers latent rows per query; sparse NAX attention never allocates them.
    const sparse = nax and config.dsa_kv_lora_rank == 512 and config.num_attention_heads % 32 == 0;
    const selected = @min(seq, @as(u64, config.dsa_index_topk) +| config.dsa_index_kpool -| 1);
    const gather = if (sparse) 0 else fwd *| selected *| @as(u64, config.dsa_kv_lora_rank) *| 2;
    // One DSA layer's dense view and the old buffer its grow copies, both freed at the window's eval.
    const layer_kv = seq *| kv_tok / @max(@as(u64, config.attnCacheLayerCount()), 1);
    const prefill_scratch = envelope +| dq_weights +| indexer +| gather +| seq *| view_tok +| layer_kv;

    // A decode graph spans every layer: at the last grow all old buffers sit beside the new rows.
    const decode_growth = output *| kv_tok +| end *| kv_tok +| end *| view_tok;

    const gross = resident +| @max(prefill_scratch, decode_growth) +| PREFILL_RUNTIME_FLOOR_BYTES;
    return (gross -| warm.creditedRows(seq) *| kv_tok) * 5 / 4;
}

// ── tests ──

/// The checkpoint globals a GLM bill reads, pinned for a test and put back by `restore`.
const GlmBillGlobals = struct {
    stride: u32,
    max: u32,
    chunk: usize,

    fn pin() GlmBillGlobals {
        const saved: GlmBillGlobals = .{ .stride = http_server.launch().ssm_checkpoint_stride, .max = http_server.launch().ssm_checkpoint_max, .chunk = generate_mod.prefillCli().override };
        http_server.launch().ssm_checkpoint_stride = 256;
        http_server.launch().ssm_checkpoint_max = 16;
        generate_mod.prefillCli().override = 0;
        return saved;
    }

    fn restore(self: GlmBillGlobals) void {
        http_server.launch().ssm_checkpoint_stride = self.stride;
        http_server.launch().ssm_checkpoint_max = self.max;
        generate_mod.prefillCli().override = self.chunk;
    }
};

test "GLM bill: f32 checkpoints and the live state are billed" {
    const t = std.testing;
    const saved = GlmBillGlobals.pin();
    defer saved.restore();
    var glm = glm5TestConfig();
    // 34 KDA layers x (f32 state 64x128x128 + conv 3 x (2x64x128 + 64x128) x 2).
    try t.expectEqual(@as(u64, 147_619_840), glm.ssmCheckpointBytes());
    const with = glm5PrefillMemoryNeeded(&glm, 100_000, 64, 8, 4096, .{});
    http_server.launch().ssm_checkpoint_stride = 0;
    const without = glm5PrefillMemoryNeeded(&glm, 100_000, 64, 8, 4096, .{});
    // Effective stride max(256, 4096): 100000/4096 + 1 snapshot, capped at 16.
    try t.expectEqual(16 * glm.ssmCheckpointBytes() * 5 / 4, with - without);
    // A one-token prompt still holds the live state, its two lag snapshots and a capture.
    try t.expect(glm5PrefillMemoryNeeded(&glm, 1, 1, 8, 512, .{}) >= 4 * glm.ssmCheckpointBytes() * 5 / 4);
}

test "GLM bill: decode growth is billed to the generation headroom, never the auto budget's whole window" {
    const t = std.testing;
    const saved = GlmBillGlobals.pin();
    defer saved.restore();
    var glm = glm5TestConfig();
    glm.ctx_override = 1 << 20;
    const headroom = cache_mod.KVCache.RESERVE_GEN_HEADROOM;
    const kv_tok = glm.kvBytesPerToken();
    // A client that sets no max_tokens gets the rest of the window: that answer is billed like one of
    // `headroom` tokens, so a short prompt's bill does not depend on the window it was pegged to.
    const auto = glm5PrefillMemoryNeeded(&glm, 431, (1 << 20) - 431, 16, 4096, .{});
    try t.expectEqual(glm5PrefillMemoryNeeded(&glm, 431, headroom, 16, 4096, .{}), auto);
    try t.expectEqual(auto, glm5PrefillMemoryNeeded(&glm, 431, 4_000_000, 16, 4096, .{}));
    // Below the headroom the growth term still prices each output token's row and its old row,
    // and it stays a max against the prefill scratch, never a sum.
    const short = glm5PrefillMemoryNeeded(&glm, 431, 64, 16, 4096, .{});
    try t.expect(auto >= short);
    try t.expect(auto - short <= headroom * 2 * kv_tok * 5 / 4);
}

test "GLM bill: a prompt token costs its KV, one layer's grow copy and its indexer pools" {
    const t = std.testing;
    const saved = GlmBillGlobals.pin();
    defer saved.restore();
    var glm = glm5TestConfig();
    const kv_tok = glm.kvBytesPerToken();
    const per_pool: u64 = GLM5_INDEXER_BYTES_PER_POOL + if (verify_qmm_nax.verifyQmmNaxAvailable()) @as(u64, 0) else 12 * 32;
    const a = glm5PrefillMemoryNeeded(&glm, 100_100, 64, 16, 4096, .{});
    const b = glm5PrefillMemoryNeeded(&glm, 144_100, 64, 16, 4096, .{});
    // 44,000 more prompt tokens: their KV, a 1/11 share for the one DSA layer that grows, 11,000 pools.
    try t.expectEqual((44_000 * kv_tok + 4_000 * kv_tok + 4096 * 11_000 * per_pool) * 5 / 4, b - a);
}

test "GLM bill: the Q8 dense view is the latent + indexer width" {
    const t = std.testing;
    const saved = GlmBillGlobals.pin();
    defer saved.restore();
    var glm = glm5TestConfig();
    const seq: u64 = 110_000;
    const dense = glm5PrefillMemoryNeeded(&glm, seq, 64, 16, 4096, .{});
    const q8 = glm5PrefillMemoryNeeded(&glm, seq, 64, 8, 4096, .{});
    // Q8 shrinks the KV and the one-layer grow copy (12/11 of the row), and adds a 2048 B/token view.
    try t.expectEqual((seq * (16896 - 8976) * 12 / 11 - seq * 2048) * 5 / 4, dense - q8);
}

test "GLM bill: warm credit requires donation and never erases checkpoints" {
    const t = std.testing;
    const saved = GlmBillGlobals.pin();
    defer saved.restore();
    var glm = glm5TestConfig();
    const warm: WarmPrefix = .{ .matched_tokens = 65000, .capacity_tokens = 65536, .will_donate = true };
    var shared = warm;
    shared.will_donate = false;
    const warm_bill = prefillNeededAtChunk(&glm, 66000, 12288, 8, 512, warm);
    const shared_bill = prefillNeededAtChunk(&glm, 66000, 12288, 8, 512, shared);
    try t.expectEqual(@as(u64, 65000 * 8976 * 5 / 4), shared_bill - warm_bill);
    http_server.launch().ssm_checkpoint_stride = 0;
    try t.expect(prefillNeededAtChunk(&glm, 66000, 12288, 8, 512, warm) < warm_bill);
}

test "GLM bill: covers the measured peaks without doubling them" {
    const t = std.testing;
    if (!verify_qmm_nax.verifyQmmNaxAvailable()) return error.SkipZigTest;
    const saved = GlmBillGlobals.pin();
    defer saved.restore();
    var glm = glm5TestConfig();
    glm.ctx_override = 262144;
    glm.quant_bits = 4;
    // Measured cold peaks over active memory; `launch` is the run's --prefill-chunk (8192 = the default).
    const Point = struct { seq: u64, kv_bits: u64, chunk: u64, launch: usize, peak: u64 };
    for ([_]Point{
        .{ .seq = 66_015, .kv_bits = 16, .chunk = 4096, .launch = 8192, .peak = 6_430_000_000 },
        .{ .seq = 129_756, .kv_bits = 16, .chunk = 4096, .launch = 8192, .peak = 9_910_000_000 },
        .{ .seq = 199_017, .kv_bits = 16, .chunk = 4096, .launch = 8192, .peak = 12_640_000_000 },
        .{ .seq = 66_015, .kv_bits = 8, .chunk = 4096, .launch = 8192, .peak = 5_930_000_000 },
        .{ .seq = 129_758, .kv_bits = 8, .chunk = 4096, .launch = 8192, .peak = 9_010_000_000 },
        .{ .seq = 199_016, .kv_bits = 8, .chunk = 4096, .launch = 8192, .peak = 11_300_000_000 },
        .{ .seq = 199_026, .kv_bits = 16, .chunk = 8192, .launch = 8192, .peak = 18_300_000_000 },
        .{ .seq = 66_015, .kv_bits = 8, .chunk = 512, .launch = 512, .peak = 4_400_000_000 },
        .{ .seq = 124_681, .kv_bits = 8, .chunk = 512, .launch = 512, .peak = 5_170_000_000 },
    }) |p| {
        generate_mod.prefillCli().override = p.launch;
        const bill = prefillNeededAtChunk(&glm, p.seq, 64, p.kv_bits, p.chunk, .{});
        try t.expect(bill >= p.peak);
        // Long prompts are where a refusal costs: there the bill stays within 1.6x.
        try t.expect(bill <= p.peak * @as(u64, if (p.seq >= 120_000) 160 else 200) / 100);
    }
}

test "GLM bill: the 96 GB Q8 prompts stay under their admitted ceilings" {
    const t = std.testing;
    if (!verify_qmm_nax.verifyQmmNaxAvailable()) return error.SkipZigTest;
    const saved = GlmBillGlobals.pin();
    defer saved.restore();
    var glm = glm5TestConfig();
    // MiB ceilings at Q8, chunk 512 that a 96 GB Mac admitted: a fit there stays a fit.
    const Case = struct { seq: u64, max_tokens: u32, ceiling_mb: u64 };
    for ([_]Case{
        .{ .seq = 65_548, .max_tokens = 64, .ceiling_mb = 6473 },
        .{ .seq = 65_548, .max_tokens = 8192, .ceiling_mb = 6682 },
        .{ .seq = 65_548, .max_tokens = 32768, .ceiling_mb = 7313 },
        .{ .seq = 98_288, .max_tokens = 64, .ceiling_mb = 7314 },
        .{ .seq = 98_288, .max_tokens = 8192, .ceiling_mb = 7520 },
        .{ .seq = 98_288, .max_tokens = 32768, .ceiling_mb = 8151 },
        .{ .seq = 125_443, .max_tokens = 64, .ceiling_mb = 8011 },
        .{ .seq = 125_443, .max_tokens = 8192, .ceiling_mb = 8220 },
        .{ .seq = 125_443, .max_tokens = 32768, .ceiling_mb = 8572 },
    }) |c| {
        try t.expect(prefillNeededAtChunk(&glm, c.seq, c.max_tokens, 8, 512, .{}) >> 20 <= c.ceiling_mb);
    }
}
