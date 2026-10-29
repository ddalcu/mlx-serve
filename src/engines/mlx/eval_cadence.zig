//! engines/mlx/eval_cadence.
//! Owns EVAL_EVERY_N_LAYERS, PREFILL_EVAL_CADENCE_DEFAULT, MOE_EVAL_EVERY_N_LAYERS, RECURRENCE_EVAL_INTERVAL, PREFILL_EVAL_TRANSIENT_BUDGET, PREFILL_EVAL_MIN_SEQ, and 8 more.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const ModelConfig = @import("../../sdk/config/model.zig").ModelConfig;
const checkpoint = @import("ssm/checkpoint.zig");
const gather_qmv = @import("../../sdk/blocks/moe/gather_qmv.zig");
const transformer = @import("transformer.zig");
const prefillHeadDimFused = @import("../../sdk/blocks/attention/attn256.zig").prefillHeadDimFused;
const testing = @import("kv/cache.zig").testing;
const SSMCacheEntry = @import("ssm/entry.zig").SSMCacheEntry;
const Transformer = @import("transformer.zig").Transformer;
const attn256_vars = @import("../../sdk/blocks/attention/attn256.zig");
const run = @import("../../sdk/bases/run.zig");

// ── Forward dispatch ──

pub const EVAL_EVERY_N_LAYERS = run.EVAL_EVERY_N_LAYERS;
pub const PREFILL_EVAL_CADENCE_DEFAULT = run.PREFILL_EVAL_CADENCE_DEFAULT;
pub const MOE_EVAL_EVERY_N_LAYERS: u32 = PREFILL_EVAL_CADENCE_DEFAULT;
pub const RECURRENCE_EVAL_INTERVAL = @import("../../sdk/bases/run.zig").RECURRENCE_EVAL_INTERVAL;

pub const PREFILL_EVAL_TRANSIENT_BUDGET = run.PREFILL_EVAL_TRANSIENT_BUDGET;

pub const PREFILL_EVAL_MIN_SEQ = @import("../../sdk/bases/run.zig").PREFILL_EVAL_MIN_SEQ;

/// Pure gate: does the mid-loop eval cadence apply to a forward of this
/// width? (True = real prefill chunk; false = decode/spec-verify shape.)
pub fn prefillEvalCadenceApplies(seq_len: c_int) bool {
    return seq_len >= PREFILL_EVAL_MIN_SEQ;
}

/// The cadence eval must NAME every conv_state beside `h`: the owned tail copy is
/// not in the residual's graph, so `mlx_array_eval(h)` alone would leave it lazy
/// and still holding its chunk parent. Re-appending an evaluated array is free.
pub fn evalCadencePoint(h: mlx.mlx_array, ssm_entries: ?[]SSMCacheEntry) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, h);
    var extra: u64 = 0;
    if (ssm_entries) |entries| {
        for (entries) |*e| {
            if (e.initialized and e.conv_state.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, e.conv_state);
                extra += 1;
            }
        }
    }
    if (extra == 0) {
        try mlx.check(mlx.mlx_array_eval(h));
        return;
    }
    checkpoint.cadence_conv_eval_count += extra;
    try mlx.check(mlx.mlx_eval(vec));
}

/// Decode-side intra-step async-eval ladder.
///
/// A decode step builds ~40 layers of graph on the CPU before anything is
/// evaluated, so the GPU sits idle through the build and the CPU sits idle
/// through the execution. `mlx_async_eval` at a layer boundary hands the
/// prefix of the graph to the GPU WITHOUT blocking, so the two overlap
/// inside one token. It adds no op and changes no value — the same graph is
/// evaluated in the same order, just started earlier — so output is
/// bit-identical with the ladder on or off.
///
/// This is distinct from the BLOCKING prefill cadence above, which exists
/// to bound transient memory on wide forwards; the ladder is additive and
/// never replaces it.
///
/// DEFAULT OFF — measured, not assumed. The mlxfast-challenge tree gets
/// +9.7% from exactly this (their `DARKBLOOM_DECODE_ASYNC_STAGE`), but
/// their worker cannot pipeline BETWEEN steps: its protocol is one token
/// per request, so the ladder is the only overlap it can buy. mlx-serve
/// already overlaps at the step boundary (`generate.zig`'s build-next →
/// async_eval → resolve-pending), so the overlap is already collected and
/// the ladder only adds submissions. Swept on Laguna XS 2.1 NVFP4 / M4 Max,
/// 512-token prompt / 128 decode steps, medians of 2:
///   off 39.88 | stride 8 40.22 | 4 40.37 | 2 41.08 | 1 41.49 ms/token
/// Monotonically worse as the stride tightens, so it ships off (`Transformer.decode_async_ladder` 0) except on
/// qwen4_exp batched decode (`QWEN4_DECODE_ASYNC_LADDER_DEFAULT`). Only decode-width forwards ladder: a spec-verify
/// or prefill forward already has enough GPU work per layer to keep the device busy.

// A PREFILL-side async-eval ladder (mlxfast-challenge prefillLadder:
// asyncEval every 3rd-4th layer at seq >= 512) was ported and measured
// (qwen3.8-27B-4bit, 43.7k-token prompt, M4 Max): off 227.9
// vs ladder 225.9 tok/s median — null-to-negative, same reason the
// decode ladder loses here: chunked prefill + the eval cadence already
// collect the overlap. Removed rather than shipped as a dead knob.

/// Fire the ladder at this layer boundary, if it is one. `h` is the live
/// residual; handing it to `mlx_async_eval` starts every op it depends on.
pub fn ladderStep(self: *Transformer, h: mlx.mlx_array, layer_idx: usize, seq_len: c_int) void {
    ladderStepMulti(&.{h}, self.decode_async_ladder, layer_idx, seq_len);
}

pub fn ladderStepMulti(arrs: []const mlx.mlx_array, stride: u32, layer_idx: usize, seq_len: c_int) void {
    if (seq_len != 1) return;
    if (stride == 0) return;
    if ((layer_idx + 1) % stride != 0) return;
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    // Best-effort: a failed async eval is a missed overlap, never a wrong
    // answer — the value is computed by the step's own terminal eval.
    _ = mlx.mlx_async_eval(vec);
    if (!gather_qmv.decode_ladder_engaged) {
        gather_qmv.decode_ladder_engaged = true;
        log.info("[decode] async-eval ladder engaged: stride={d}\n", .{stride});
    }
}

pub const prefillEvalCadence = run.prefillEvalCadence;

pub fn prefillEvalCadenceFor(default_cadence: u32, cfg: *const ModelConfig, chunk_len: u64, total_kv: u64, kv_dequant: bool) u32 {
    return prefillEvalCadence(default_cadence, cfg.prefillScoreHeadDim(), cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads, chunk_len, total_kv, kv_dequant);
}

// ── tests ──

test "prefillEvalCadence: small transients keep the coarse cadence" {
    const t = std.testing;
    attn256_vars.fused256_override = false;
    defer attn256_vars.fused256_override = null;
    // 26B geometry, short prompt: 16 heads x 2048 chunk x 2048 kv x 2B = 128 MB
    // of unfused scores — well inside the budget, keep today's cadence.
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 256, 16, 8, 2048, 2048, false));
    // The LIVE-measured no-regression point: a 5,140-token agent prompt
    // (845 MB scores) must keep cadence 4 — flipping it cost a measured 4.5%
    // prefill for zero memory benefit (peak 18.1 GB both ways).
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 256, 16, 8, 5140, 5140, false));
    // Fused head dim + dense KV: no transient at all, keep even the 48 default.
    try t.expectEqual(@as(u32, 48), Transformer.prefillEvalCadence(48, 128, 128, 8, 8, 8192, 500_000, false));
}

test "prefillEvalCadenceFor: MiMo bills no score tensor, so a long chunk keeps the coarse cadence" {
    const t = std.testing;
    attn256_vars.fused256_override = false;
    defer attn256_vars.fused256_override = null;
    const mimo: ModelConfig = .{ .model_type = "mimo_v2", .head_dim = 192, .num_attention_heads = 64, .num_key_value_heads = 8 };
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadenceFor(4, &mimo, 8192, 32768, false));
    // An unfused wide head still drops to per-layer evals.
    const wide: ModelConfig = .{ .model_type = "qwen3_5", .head_dim = 256, .num_attention_heads = 64, .num_key_value_heads = 8 };
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadenceFor(4, &wide, 8192, 32768, false));
}

test "prefillEvalCadence: fused hd-256 kernel drops the score term" {
    const t = std.testing;
    // With msv_attn_p256 active (the default) the score transient never
    // exists — hd 256 keeps the coarse cadence at any context, exactly like
    // hd 128. The dequant term must still fire under --kv-quant.
    attn256_vars.fused256_override = true;
    defer attn256_vars.fused256_override = null;
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 256, 16, 8, 8192, 102_448, false));
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(4, 256, 256, 16, 8, 8192, 400_000, true));
}

test "prefillEvalCadence: big unfused score tensor forces eval-per-layer" {
    const t = std.testing;
    attn256_vars.fused256_override = false;
    defer attn256_vars.fused256_override = null;
    // 16 heads x 8192 chunk x 16384 kv x 2B = 4 GiB > 2 GiB budget: bounds a
    // ~13 GB windowed transient to ~4 GB. (8192 chunk x 8192 kv sits exactly
    // ON the 2 GiB budget and deliberately keeps the coarse cadence.)
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(4, 256, 256, 16, 8, 8192, 16384, false));
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 256, 16, 8, 8192, 8192, false));
    // The capped 1024 chunk at 102K ctx still materializes 3.4 GB — the chunk
    // cap (generate.zig) does NOT restore the coarse cadence at long context.
    // (Live: this cadence measured peak 27.0 GB and +14% prefill vs baseline.)
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(4, 256, 256, 16, 8, 1024, 102_448, false));
}

test "prefillEvalCadenceApplies: spec-verify-width forwards skip the cadence entirely" {
    const t = std.testing;
    // Spec-decode verify forwards (PLD/drafter/MTP: seq 2..9) are
    // decode-shaped — KB-scale transients — but ride the prefill layer loop
    // because they're multi-token. The mid-loop eval() cadence only costs
    // them synchronous pipeline drains (measured: cadence 4 on the 64-layer
    // qwen3.6-27B = 16 drains per MTP round, ~13 ms of a 48 ms depth-1
    // round, superlinear in draft depth). seq-1 decode never ran cadence
    // evals, so exempting verify widths bounds nothing new.
    try t.expect(!Transformer.prefillEvalCadenceApplies(2));
    try t.expect(!Transformer.prefillEvalCadenceApplies(9));
    try t.expect(!Transformer.prefillEvalCadenceApplies(31));
    // Real prefill chunks keep the cadence (and the budget-driven flip).
    try t.expect(Transformer.prefillEvalCadenceApplies(32));
    try t.expect(Transformer.prefillEvalCadenceApplies(512));
    try t.expect(Transformer.prefillEvalCadenceApplies(8192));
}

test "prefillEvalCadence: quantized-KV dequant forces eval-per-layer even at fused head dims" {
    const t = std.testing;
    // hd 128 is fused (zero scores) but denseView rebuilds the FULL cache per
    // layer under --kv-quant: 2 x 600K x 8 x 128 x 2B = 2.46 GB > 2 GiB.
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(48, 128, 128, 8, 8, 8192, 600_000, true));
    // Same shape with dense fp16 KV: nothing to dequantize, keep coarse.
    try t.expectEqual(@as(u32, 48), Transformer.prefillEvalCadence(48, 128, 128, 8, 8, 8192, 600_000, false));
}
