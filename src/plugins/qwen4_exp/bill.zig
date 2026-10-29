//! qwen4_exp's memory terms: the QSA indexer's per-slot ring, per-token history and score bank, the prefill's QSA
//! transient, the batched tick's effective KV and the SSM checkpoint, each the engine's own arithmetic.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const gather_mod = @import("qsa/gather.zig");
const score_mod = @import("../../sdk/blocks/attention/sparse_block/score.zig");
const attn256 = @import("../../sdk/blocks/attention/attn256.zig");
const model_mod = @import("../../sdk/config/model.zig");
const qconfig = @import("config.zig");
const toModelConfig = @import("model_config.zig").toModelConfig;

const bills = @import("../../engines/mlx/admission/bills.zig");
const prefillTransientReserve = bills.prefillTransientReserve;
const resolvePrefillChunk = @import("../../engines/mlx/admission/prefill_chunk.zig").resolvePrefillChunk;
const verify_qmm_nax = @import("../../sdk/blocks/linear/verify_qmm_nax.zig");

/// Bytes per (query, key) the QSA prefill holds for ONE live layer past the
/// indexer budget: the `[S, kv]` bool mask, its additive copy for the sdpa
/// arm, and the f32 indexer scores over kv/4 blocks (4 bytes/block = 1/key).
const QSA_MASK_BYTES_PER_KEY: u64 = 4;

/// qwen4_exp sparse attention materializes a `[S, kv]` mask per layer that
/// no envelope term models: 4096 x 25k keys = 410 MB, and a tight box met it
/// as a Metal OOM at 25k+. Zero for every arch without an indexer. The sizer
/// bills it at kv = chunk (no prompt yet); the admission guard at the real
/// prompt length.
pub fn qsaMaskBytes(config: *const model_mod.ModelConfig, fwd: u64, kv: u64) u64 {
    if (config.indexer_budget == 0) return 0;
    // Prefill widths gather by block index (row-chunked score sheet, no
    // [S, kv] mask); decode/verify widths still build the dense mask.
    if (fwd >= attn256.FUSED256_MIN_Q_LEN and gather_mod.qsaGatherEnabled())
        return score_mod.qsaPrefillTransientBytes(config.indexer_n_heads, fwd, kv, config.indexer_compress_ratio, config.indexer_head_dim);
    return QSA_MASK_BYTES_PER_KEY * fwd * kv * 5 / 4;
}

/// The engine's config of a parsed pack (no folder: nothing allocated).
fn engineConfig(p: *const abi.Parsed) model_mod.ModelConfig {
    return toModelConfig(std.heap.c_allocator, qconfig.of(p), null) catch unreachable;
}

pub fn bill(p: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    const cfg = engineConfig(p);
    out.* = .{
        .fixed_slot_bytes = cfg.qsaRingBytes(),
        .state_per_token = bills.statePerTokenBilled(&cfg),
        .ssm_checkpoint = cfg.ssmCheckpointBytes(),
        .prefill_bytes = &prefillBytes,
        .batched_kv_len = &batchedKvLen,
    };
}

/// The QSA transient a prefill chunk adds beside the dense terms, at the real prompt length.
fn prefillBytes(p: *const abi.Parsed, seq: u64, chunk: u32, _: u8, _: u64) callconv(.c) u64 {
    const cfg = engineConfig(p);
    return qsaMaskBytes(&cfg, @min(chunk, @max(seq, 1)), seq);
}

fn batchedKvLen(p: *const abi.Parsed, kv: u32, gather_on: bool, min: u32) callconv(.c) u32 {
    const cfg = engineConfig(p);
    return cfg.batchedEffectiveKvLen(kv, gather_on, min);
}

// ── tests ──

test "qwen4_exp bill: every term is the engine's arithmetic on the shipped pack" {
    const t = std.testing;
    const fixture = @import("../../sdk/testing/fixture.zig");
    const json = @embedFile("../../fixtures/model-configs/qwen4_exp-iq-mlx-4.7bpw.json");
    var want = try model_mod.parseConfigFromJson(t.allocator, json);
    defer want.deinit(t.allocator);
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try t.expect(qconfig.parse(&dir, .of(json), .{}, null, true, &p).isOk());
    defer qconfig.freeParsed(&p);
    var b: abi.Bill = .{};
    bill(&p, &.{ .context = 65536 }, &b);
    try t.expectEqual(want.qsaRingBytes(), b.fixed_slot_bytes);
    try t.expect(b.fixed_slot_bytes > 0);
    try t.expectEqual(bills.statePerTokenBilled(&want), b.state_per_token);
    try t.expectEqual(want.ssmCheckpointBytes(), b.ssm_checkpoint);
    try t.expectEqual(qsaMaskBytes(&want, 4096, 100_000), b.prefill_bytes.?(&p, 100_000, 4096, 0, 0));
    try t.expectEqual(want.batchedEffectiveKvLen(50_000, true, 4096), b.batched_kv_len.?(&p, 50_000, true, 4096));
    try t.expectEqual(@as(u32, 2048 + 4), b.batched_kv_len.?(&p, 50_000, true, 4096));
}

test "qsaMaskBytes: a qwen4_exp twin bills the QSA mask and steps a rung the qwen3_5 twin keeps" {
    const transformer_mod = @import("../../engines/mlx/transformer.zig");
    const t = std.testing;
    var twin = model_mod.ModelConfig{};
    twin.num_hidden_layers = 40;
    twin.num_attention_heads = 24;
    twin.num_key_value_heads = 2;
    twin.head_dim = 256;
    twin.hidden_size = 2560;
    twin.intermediate_size = 9216;
    twin.intermediate_size_declared = true;
    twin.quant_bits = 4;
    twin.max_position_embeddings = 262144;
    var q4 = twin;
    q4.indexer_budget = 2048;
    q4.indexer_n_heads = 4;
    q4.indexer_head_dim = 128;
    q4.indexer_compress_ratio = 4;
    try t.expectEqual(@as(u64, 0), qsaMaskBytes(&twin, 4096, 25000));
    // Decode/verify widths build the dense mask; prefill widths gather by
    // block and bill the bounded score sheet instead of 4 B x rows x keys.
    try t.expectEqual(@as(u64, 4 * 8 * 25000 * 5 / 4), qsaMaskBytes(&q4, 8, 25000));
    transformer_mod.qsa_gather_override = true;
    defer transformer_mod.qsa_gather_override = null;
    const saved_fused = score_mod.qsa_score_fused_override;
    const saved_nax = verify_qmm_nax.vqmm_nax_probe_override;
    defer {
        score_mod.qsa_score_fused_override = saved_fused;
        verify_qmm_nax.vqmm_nax_probe_override = saved_nax;
    }
    verify_qmm_nax.vqmm_nax_probe_override = false;
    score_mod.qsa_score_fused_override = true;
    const gathered_composed = qsaMaskBytes(&q4, 4096, 250_000);
    try t.expect(gathered_composed > 0);
    try t.expect(gathered_composed < 4 * 4096 * 250_000);
    try t.expectEqual(score_mod.qsaPrefillTransientBytes(4, 4096, 250_000, 4, 128), gathered_composed);
    verify_qmm_nax.vqmm_nax_probe_override = null;
    score_mod.qsa_score_fused_override = true;
    const gathered_fused = qsaMaskBytes(&q4, 4096, 250_000);
    try t.expectEqual(score_mod.qsaPrefillTransientBytes(4, 4096, 250_000, 4, 128), gathered_fused);
    if (score_mod.qsaScoreFusedActive()) {
        try t.expect(gathered_fused < gathered_composed);
    }
    score_mod.qsa_score_fused_override = false;
    verify_qmm_nax.vqmm_nax_probe_override = false;
    const gathered = qsaMaskBytes(&q4, 4096, 250_000);
    try t.expectEqual(gathered_composed, gathered);
    const twin_bill = prefillTransientReserve(&twin, 16, 8192);
    const q4_bill = prefillTransientReserve(&q4, 16, 8192);
    try t.expectEqual(twin_bill + qsaMaskBytes(&q4, 8192, 8192) + q4.qsaRingBytes() * 5 / 4, q4_bill);
    // A ceiling whose quarter-share sits between the two bills at 8192.
    const weights: u64 = 70_000_000_000;
    const ceiling = weights + (twin_bill + q4_bill) / 2 * 4;
    try t.expectEqual(@as(u32, 8192), resolvePrefillChunk(&twin, 16, ceiling, weights, 0, 0));
    try t.expectEqual(@as(u32, 4096), resolvePrefillChunk(&q4, 16, ceiling, weights, 0, 0));
}
