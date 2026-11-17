//! GLM-5.3-Flash's prefill bill (`glm5PrefillMemoryNeeded`): prompt KV, the f32 KDA checkpoints and live state, one
//! DSA layer's scratch and indexer pools, decode growth on the KV reserve's headroom. The engine's admission calls it
//! until `Bill.prefill_bytes` carries max_tokens and the warm prefix.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
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
const mtpHeadBilled = @import("bills.zig").mtpHeadBilled;
const BillTerms = @import("../plugin/config.zig").Binding.BillTerms;

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

/// The host's hot prefix cache at its default cap: it keeps a request's decode-end checkpoint beside the decode, and
/// neither bill names it otherwise. A streamed bank's hot-cache budget never passes it (`streamedHotCacheCapped`).
pub const GLM5_HOT_CACHE_BYTES: u64 = 2048 << 20;

/// Whether the engine bills a streamed model's decode phase apart (`LoadCtx.stream_room_decode`): its bill's
/// `decode_room`. Unset (or a Bill that predates it) is the name rule: a built-in GLM on a streamed plugin that prices
/// no prompt of its own, and Ling (bailing_hybrid), which runs GLM's trunk though its plugin prices its prompt.
pub fn decodeBilled(config: *const model_mod.ModelConfig) bool {
    const terms = streamedTerms(config) orelse return false;
    return decodeRoomOf(config, terms);
}

/// Whether a decode-billed request reserves its whole KV at its prompt (`reservedPrefillTokens`) and its decode bill
/// prices that KV once: its bill's `reserves_request`.
pub fn reservesRequest(config: *const model_mod.ModelConfig) bool {
    const terms = streamedTerms(config) orelse return false;
    if (!decodeRoomOf(config, terms)) return false;
    return if (terms.flags) |f| f.reserves_request else false;
}

/// A streamed plugin's bill terms; null for a model that does not stream.
fn streamedTerms(config: *const model_mod.ModelConfig) ?BillTerms {
    const b = config.plugin orelse return null;
    if (!b.parsed.caps.streamed_weights) return null;
    return b.billTerms();
}

fn decodeRoomOf(config: *const model_mod.ModelConfig, terms: BillTerms) bool {
    const room: abi.DecodeRoom = if (terms.flags) |f| f.decode_room else .by_name;
    return switch (room) {
        .billed => true,
        else => std.mem.eql(u8, config.model_type, "bailing_hybrid") or (config.isGlm5() and !terms.prices_prompt),
    };
}

/// Whether a streamed plugin's hot cache is held to `GLM5_HOT_CACHE_BYTES` (`ungatedHotCacheAsk`): admission bills a
/// streamed pack at 0, so nothing evicts beside its next request. A plugin that commits through its module
/// (`CacheHooks.aux`, DeepSeek-V4.1) sizes its own bank and makes its cache's room before it appends
/// (`makeModuleRoom`), and keeps its own accounting.
pub fn streamedHotCacheCapped(config: *const model_mod.ModelConfig) bool {
    if (decodeBilled(config)) return true;
    const b = config.plugin orelse return false;
    if (!b.parsed.caps.streamed_weights) return false;
    const hooks = b.table.cache;
    return hooks.size < @sizeOf(abi.CacheHooks) or hooks.aux == null;
}

/// GLM-5's decode-phase bill at the same cell (`LoadCtx.stream_room_decode`): the prompt's KV and checkpoints, a
/// generation's growth and the runtime floor at the prompt bill's 5/4, plus the hot prefix cache. No prefill scratch:
/// the phase change frees it before decode's rows grow.
pub fn glm5DecodeMemoryNeeded(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u64, kv_bits: u64, chunk: u64) u64 {
    if (reservesRequest(config)) return reservedDecodeMemoryNeeded(config, seq, max_tokens, kv_bits, chunk);
    const ctx: u64 = getEffectiveContextLength(config);
    const output = @min(max_tokens, cache_mod.KVCache.RESERVE_GEN_HEADROOM, if (ctx > 0) ctx -| seq else max_tokens);
    const end = seq +| output;
    const kv_tok = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits);
    const view_tok: u64 = if (kv_bits < 16) glm5DequantWidth(config) else 0;
    const resident = seq *| kv_tok +| retainedSsmCheckpointBytes(config, seq, 0, chunk) +|
        4 *| config.ssmCheckpointBytes() +| slotFixedKvBytes(config, kv_bits, chunk);
    const decode_growth = output *| kv_tok +| end *| kv_tok +| end *| view_tok;
    return (resident +| decode_growth +| PREFILL_RUNTIME_FLOOR_BYTES) * 5 / 4 +| GLM5_HOT_CACHE_BYTES;
}

/// The decode bill of a request that reserved its whole KV at its prompt (`reservedPrefillTokens`): that KV once, at the
/// reserved rows, with the MTP head's rows beside it. No decode grow, so no old buffer sits beside the new. The hot
/// cache holds its resident entry and, while it commits, that entry's copy: two terms.
fn reservedDecodeMemoryNeeded(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u64, kv_bits: u64, chunk: u64) u64 {
    const rows = generate_mod.reservedPrefillTokens(config, seq, max_tokens, chunk);
    const kv_tok = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits);
    const view_tok: u64 = if (kv_bits < 16) glm5DequantWidth(config) else 0;
    const fixed = retainedSsmCheckpointBytes(config, seq, 0, chunk) +| 4 *| config.ssmCheckpointBytes() +|
        slotFixedKvBytes(config, kv_bits, chunk);
    const kv = rows *| (kv_tok +| view_tok +| mtpHeadRowBytes(config));
    return (fixed +| kv +| PREFILL_RUNTIME_FLOOR_BYTES) * 5 / 4 +| 2 *| GLM5_HOT_CACHE_BYTES;
}

/// Bytes a token takes in the MTP head's own cache, one dense bf16 trunk layer's row (GLM-5's mtp.0 a DSA layer, Ling's
/// NextN layer an MLA one); 0 without a head the bill counts.
fn mtpHeadRowBytes(config: *const model_mod.ModelConfig) u64 {
    const b = config.plugin orelse return 0;
    if (b.parsed.spec.mtp_layout == .none or !mtpHeadBilled(config)) return 0;
    const n = config.attnCacheLayerCount();
    return if (n == 0) 0 else config.kvBytesPerToken() / n;
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

test "GLM bill: the decode phase's bill holds no prefill scratch, so it sits below the prompt bill at the stream room's cell" {
    const saved = GlmBillGlobals.pin();
    defer saved.restore();
    const glm = glm5TestConfig();
    const seq: u64 = 16_384 + 1_024;
    const prompt = glm5PrefillMemoryNeeded(&glm, seq, 1, 16, generate_mod.WIDE_PREFILL_CHUNK, .{});
    const decode = glm5DecodeMemoryNeeded(&glm, seq, 1, 16, generate_mod.WIDE_PREFILL_CHUNK);
    try std.testing.expect(decode > GLM5_HOT_CACHE_BYTES);
    try std.testing.expect(decode < prompt + GLM5_HOT_CACHE_BYTES);
    // Its terms: the prompt's resident set and a generation's growth at 5/4, then the hot cache.
    const kv_tok = kvBytesPerTokenAtBits(glm.kvBytesPerToken(), 16);
    const resident = seq * kv_tok + retainedSsmCheckpointBytes(&glm, seq, 0, generate_mod.WIDE_PREFILL_CHUNK) + 4 * glm.ssmCheckpointBytes() + slotFixedKvBytes(&glm, 16, generate_mod.WIDE_PREFILL_CHUNK);
    const growth = kv_tok + (seq + 1) * kv_tok;
    try std.testing.expectEqual((resident + growth + PREFILL_RUNTIME_FLOOR_BYTES) * 5 / 4 + GLM5_HOT_CACHE_BYTES, decode);
}

/// A streamed plugin binding whose bill sets `flags` (decode room and reservation), with an MTP head; `priced` also
/// prices its prompt.
fn reserveBinding(table: *abi.ModelTable, comptime flags: abi.BillFlags, comptime priced: bool) @import("../plugin/config.zig").Binding {
    const Probe = struct {
        fn prefill(_: *const abi.Parsed, _: u64, _: u32, _: u8, _: u64) callconv(.c) u64 {
            return 3 << 30;
        }
        fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
            out.* = .{ .prefill_bytes = if (priced) &prefill else null, .flags = flags };
        }
    };
    table.bill = &Probe.bill;
    return .{ .table = table, .parsed = .{ .caps = .{ .streamed_weights = true }, .spec = .{ .mtp_layout = .in_checkpoint } } };
}

// Bar: a request that reserves its KV once grows its decode bill by the KV once a token (at 5/4) and the MTP head's
// rows, from a 19,404-token request to each limit, and bills a reply past 8,192 whole.
test "GLM bill: a reserving Ling's decode bills its reserved KV once, beside its MTP head's dense MLA rows" {
    const t = std.testing;
    const saved = GlmBillGlobals.pin();
    defer saved.restore();
    http_server.launch().ssm_checkpoint_stride = 0;
    var table: abi.ModelTable = undefined;
    var binding = reserveBinding(&table, .{ .decode_room = .billed, .reserves_request = true }, true);
    var cfg = try model_mod.parseConfigFromJson(t.allocator, @embedFile("../../../fixtures/model-configs/bailing_hybrid-ling-3.1-flash-exl3-2.60bpw.json"));
    const own = cfg.plugin;
    defer {
        cfg.plugin = own;
        cfg.deinit(t.allocator);
    }
    cfg.plugin = &binding;
    const wide = generate_mod.WIDE_PREFILL_CHUNK;
    const kv = cfg.kvBytesPerToken();
    // The head's one MLA layer, dense at every head: 64 x (192 + 128) bf16.
    const head: u64 = 40_960;
    const from: u64 = 19_404;
    for ([_]u64{ 49_975, 83_694, 150_999 }) |limit| {
        cfg.ctx_override = @intCast(limit);
        const grow = glm5DecodeMemoryNeeded(&cfg, limit - 1, 1, 16, wide) - glm5DecodeMemoryNeeded(&cfg, from, 1, 16, wide);
        try t.expectEqual((limit - 1 - from) * (kv + head) * 5 / 4, grow);
    }
    cfg.ctx_override = 49_975;
    try t.expectEqual(11_197 * (kv + head) * 5 / 4, glm5DecodeMemoryNeeded(&cfg, 33_591, 11_198, 16, wide) - glm5DecodeMemoryNeeded(&cfg, 33_591, 1, 16, wide));
}

// Bar: the head's rows are its own layer kind's: GLM-5's mtp.0 is a DSA layer (its latent row and the indexer's key
// and gate), another model's one attention layer at its KV heads.
test "GLM bill: a reserving model's MTP head rows are its own layer kind's" {
    const t = std.testing;
    const saved = GlmBillGlobals.pin();
    defer saved.restore();
    http_server.launch().ssm_checkpoint_stride = 0;
    const wide = generate_mod.WIDE_PREFILL_CHUNK;
    var table: abi.ModelTable = undefined;
    var binding = reserveBinding(&table, .{ .decode_room = .billed, .reserves_request = true }, false);
    var glm = glm5TestConfig();
    glm.plugin = &binding;
    glm.ctx_override = 83_694;
    const glm_head: u64 = (512 + 2 * 128) * 2;
    try t.expectEqual((83_693 - 19_404) * (glm.kvBytesPerToken() + glm_head) * 5 / 4, glm5DecodeMemoryNeeded(&glm, 83_693, 1, 16, wide) - glm5DecodeMemoryNeeded(&glm, 19_404, 1, 16, wide));
    var toy: model_mod.ModelConfig = .{ .model_type = "toy_hybrid", .num_hidden_layers = 4, .num_attention_heads = 8, .num_key_value_heads = 2, .head_dim = 64, .hidden_size = 512, .max_position_embeddings = 131_072 };
    toy.plugin = &binding;
    toy.ctx_override = 83_694;
    const toy_head: u64 = 2 * 2 * 64 * 2;
    try t.expectEqual((83_693 - 19_404) * (toy.kvBytesPerToken() + toy_head) * 5 / 4, glm5DecodeMemoryNeeded(&toy, 83_693, 1, 16, wide) - glm5DecodeMemoryNeeded(&toy, 19_404, 1, 16, wide));
    // Without a head, the KV alone.
    binding.parsed.spec.mtp_layout = .none;
    try t.expectEqual((83_693 - 19_404) * toy.kvBytesPerToken() * 5 / 4, glm5DecodeMemoryNeeded(&toy, 83_693, 1, 16, wide) - glm5DecodeMemoryNeeded(&toy, 19_404, 1, 16, wide));
}

// Bar: the head's row is one trunk layer's KV row, to the byte: GLM-5's DSA row, Ling's MLA row at every head, a GQA row.
test "GLM bill: an MTP head's row is one trunk layer's KV row on GLM-5, Ling and a GQA model" {
    const t = std.testing;
    var table: abi.ModelTable = undefined;
    var binding = reserveBinding(&table, .{ .decode_room = .billed, .reserves_request = true }, false);
    var glm = glm5TestConfig();
    glm.plugin = &binding;
    try t.expectEqual(@as(u64, (512 + 2 * 128) * 2), mtpHeadRowBytes(&glm));
    try t.expectEqual(glm.kvBytesPerToken() / glm.attnCacheLayerCount(), mtpHeadRowBytes(&glm));
    var ling = try model_mod.parseConfigFromJson(t.allocator, @embedFile("../../../fixtures/model-configs/bailing_hybrid-ling-3.1-flash-exl3-2.60bpw.json"));
    const own = ling.plugin;
    defer {
        ling.plugin = own;
        ling.deinit(t.allocator);
    }
    ling.plugin = &binding;
    try t.expectEqual(@as(u64, 40_960), mtpHeadRowBytes(&ling));
    try t.expectEqual(ling.kvBytesPerToken() / ling.attnCacheLayerCount(), mtpHeadRowBytes(&ling));
    var toy: model_mod.ModelConfig = .{ .model_type = "toy_hybrid", .num_hidden_layers = 4, .num_attention_heads = 8, .num_key_value_heads = 2, .head_dim = 64, .hidden_size = 512, .max_position_embeddings = 131_072 };
    toy.plugin = &binding;
    try t.expectEqual(@as(u64, 2 * 2 * 64 * 2), mtpHeadRowBytes(&toy));
    // No head the bill counts: none.
    binding.parsed.spec.mtp_layout = .none;
    try t.expectEqual(@as(u64, 0), mtpHeadRowBytes(&ling));
}

// Bar: a reserving request holds the hot cache's resident entry and, while it commits, that entry's copy: two hot-cache
// terms beside the reserved KV, the head's rows, the checkpoints, the slot's fixed rows and the runtime floor.
test "GLM bill: a reserving request's decode bill holds two hot-cache terms beside its KV once" {
    const t = std.testing;
    const saved = GlmBillGlobals.pin();
    defer saved.restore();
    http_server.launch().ssm_checkpoint_stride = 0;
    const wide = generate_mod.WIDE_PREFILL_CHUNK;
    var table: abi.ModelTable = undefined;
    var binding = reserveBinding(&table, .{ .decode_room = .billed, .reserves_request = true }, false);
    var glm = glm5TestConfig();
    glm.plugin = &binding;
    glm.ctx_override = 32_768;
    const rows: u64 = 18_380 + 1_024 + @import("../generate/init.zig").verify_rows_max;
    const fixed = 4 * glm.ssmCheckpointBytes() + slotFixedKvBytes(&glm, 16, wide);
    const kv = rows * (glm.kvBytesPerToken() + (512 + 2 * 128) * 2);
    try t.expectEqual((fixed + kv + PREFILL_RUNTIME_FLOOR_BYTES) * 5 / 4 + 2 * GLM5_HOT_CACHE_BYTES, glm5DecodeMemoryNeeded(&glm, 18_380, 1_024, 16, wide));
}

// Bar: a decode-billed model that does not set reserves_request keeps the grow term: a grow's old buffers beside the
// new rows; so does one that sets it without its own decode room.
test "GLM bill: without reserves_request, or without a decode room, the decode bill keeps its grow term" {
    const t = std.testing;
    const saved = GlmBillGlobals.pin();
    defer saved.restore();
    http_server.launch().ssm_checkpoint_stride = 0;
    const wide = generate_mod.WIDE_PREFILL_CHUNK;
    var table: abi.ModelTable = undefined;
    var binding = reserveBinding(&table, .{ .decode_room = .billed }, false);
    var glm = glm5TestConfig();
    glm.plugin = &binding;
    glm.ctx_override = 83_694;
    const kv = glm.kvBytesPerToken();
    try t.expectEqual((83_693 - 19_404) * 2 * kv * 5 / 4, glm5DecodeMemoryNeeded(&glm, 83_693, 1, 16, wide) - glm5DecodeMemoryNeeded(&glm, 19_404, 1, 16, wide));
    // The flag on a GLM that prices its own prompt: the name rule gives it no decode room.
    var no_room = reserveBinding(&table, .{ .reserves_request = true }, true);
    glm.plugin = &no_room;
    try t.expectEqual((83_693 - 19_404) * 2 * kv * 5 / 4, glm5DecodeMemoryNeeded(&glm, 83_693, 1, 16, wide) - glm5DecodeMemoryNeeded(&glm, 19_404, 1, 16, wide));
}

// Bar: a load reads its plugin's bill once, however many admission helpers ask.
test "GLM bill: a bound plugin's bill is read once for its decode room, reservation and hot-cache cap" {
    const t = std.testing;
    const Count = struct {
        var calls: u32 = 0;
        fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
            calls += 1;
            out.* = .{ .flags = .{ .decode_room = .billed, .reserves_request = true } };
        }
    };
    var table: abi.ModelTable = undefined;
    table.bill = &Count.bill;
    var binding: @import("../plugin/config.zig").Binding = .{ .table = &table, .parsed = .{ .caps = .{ .streamed_weights = true } } };
    binding.readBillTerms();
    var ling: model_mod.ModelConfig = .{ .model_type = "bailing_hybrid", .plugin = &binding };
    try t.expect(decodeBilled(&ling) and reservesRequest(&ling) and streamedHotCacheCapped(&ling));
    try t.expectEqual(@as(u32, 1), Count.calls);
}
