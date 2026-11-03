//! `attn256.zig`'s tests, out of its import graph: a plugin unit that reaches `attn256.zig` never compiles them.

const attn256 = @import("attn256.zig");
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const sdpa = @import("sdpa.zig");
const FUSED256_DEFAULT_DISPATCH_BUDGET = attn256.FUSED256_DEFAULT_DISPATCH_BUDGET;
const attn256MaxDiff = attn256.attn256MaxDiff;
const attn256RandBf16 = attn256.attn256RandBf16;
const attn256RandBf16Scaled = attn256.attn256RandBf16Scaled;
const attn256Reference = attn256.attn256Reference;
const attn256SliceLastDim = attn256.attn256SliceLastDim;
const fused256KvChunkLen = attn256.fused256KvChunkLen;
const fusedSdpa256Masked = attn256.fusedSdpa256Masked;
const fusedSdpa256Prefill = attn256.fusedSdpa256Prefill;
const fusedSinkAttnPrefill = attn256.fusedSinkAttnPrefill;
const attn256QsaMask = @import("../../../plugins/qwen4_exp/qsa/gather.zig").attn256QsaMask;
const qsaMaskArm = @import("sparse_block/mask.zig").qsaMaskArm;

// ── fused hd-256 prefill attention parity ──

test "fusedSdpa256Prefill: causal parity vs composed SDPA (GQA, ragged shapes, chunk offset)" {
    const s = mlx.gpuStream();
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    // The causal arm yields to the stock fused kernel on NAX silicon, so on an
    // M5-class Mac this parity check would DECLINE instead of running — the
    // kernel under test would never be exercised. Pin the arm, not the host.
    sdpa.nax_sdpa_override = false;
    defer sdpa.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x256256);
    const rnd = prng.random();

    // qL=70 (partial 32-tile), kL=193 (partial 16-block), offset 123 (chunked
    // prefill bottom-right alignment), Hq=6/Hk=2 (gqa 3).
    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0; // 1/sqrt(256)

    const fused = (try fusedSdpa256Prefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);

    // Not byte-parity: the composed path rounds softmax probs to bf16 before
    // the AV matmul, the fused kernel keeps float32 — bf16-rounding-scale
    // differences only.
    // Measured 0.00049 (one bf16 ULP at 0.5 magnitude) on this seed.
    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.005);
}

test "fusedSdpa256Masked: QSA bool-mask parity vs composed 'array' SDPA (GQA, ragged tiles, kv chunking)" {
    const s = mlx.gpuStream();
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    attn256.qsa_fused_override = true;
    defer attn256.qsa_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x45a45a);
    const rnd = prng.random();

    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const mask = try attn256QsaMask(rnd, 70, 193);
    defer _ = mlx.mlx_array_free(mask);
    const scale: f32 = 1.0 / 16.0;

    const fused = (try fusedSdpa256Masked(s, q, k, v, scale, mask)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    try std.testing.expect(try attn256MaxDiff(fused, ref, s) < 0.005);

    // Wrong mask shape declines; a shape-valid mask under the kill switch too.
    const bad_shape = [_]c_int{ 1, 1, 70, 192 };
    const bad = try attn256RandBf16(rnd, &bad_shape, s);
    defer _ = mlx.mlx_array_free(bad);
    try std.testing.expect((try fusedSdpa256Masked(s, q, k, v, scale, bad)) == null);
    attn256.qsa_fused_override = false;
    try std.testing.expect((try fusedSdpa256Masked(s, q, k, v, scale, mask)) == null);
}

test "fusedSdpa256Masked: a last-axis slice with an unaligned base declines and the fallback reads exactly" {
    // `[3,259)` has a 6-byte base: the row stride (264) is divisible by 8 so the
    // existing stride checks pass, but `msv_attn_p256`'s `uint4` K/V staging
    // reads wrong values. The gate must decline it and the caller's fallback
    // (`qsaMaskArm`, the same arm production takes) must serve the shape.
    // `[0,256)` is the aligned twin: it keeps the fused kernel.
    const s = mlx.gpuStream();
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    attn256.qsa_fused_override = true;
    defer attn256.qsa_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x5E7_5E7);
    const rnd = prng.random();

    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k_pad = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 193, 264 }, s);
    defer _ = mlx.mlx_array_free(k_pad);
    const v_pad = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 193, 264 }, s);
    defer _ = mlx.mlx_array_free(v_pad);
    try mlx.check(mlx.mlx_array_eval(k_pad));
    try mlx.check(mlx.mlx_array_eval(v_pad));
    const mask = try attn256QsaMask(rnd, 70, 193);
    defer _ = mlx.mlx_array_free(mask);
    const scale: f32 = 1.0 / 16.0;

    const k = try attn256SliceLastDim(k_pad, 3, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256SliceLastDim(v_pad, 3, s);
    defer _ = mlx.mlx_array_free(v);
    try std.testing.expect((try fusedSdpa256Masked(s, q, k, v, scale, mask)) == null);
    const arm = try qsaMaskArm(s, q, k, v, scale, mask);
    defer _ = mlx.mlx_array_free(arm);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    try std.testing.expect(try attn256MaxDiff(arm, ref, s) < 0.005);

    // An aligned twin keeps the fast path.
    const k_al = try attn256SliceLastDim(k_pad, 0, s);
    defer _ = mlx.mlx_array_free(k_al);
    const v_al = try attn256SliceLastDim(v_pad, 0, s);
    defer _ = mlx.mlx_array_free(v_al);
    const fast = (try fusedSdpa256Masked(s, q, k_al, v_al, scale, mask)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fast);
    const ref_al = try attn256Reference(q, k_al, v_al, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref_al);
    try std.testing.expect(try attn256MaxDiff(fast, ref_al, s) < 0.005);
}

test "fusedSdpa256Prefill: sliding-band parity vs composed 'array' mask (Gemma local layers)" {
    const s = mlx.gpuStream();
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0xBA9D);
    const rnd = prng.random();

    const qL: c_int = 70;
    const kL: c_int = 193;
    const sw: c_int = 40;
    const q_shape = [_]c_int{ 1, 4, qL, 256 };
    const kv_shape = [_]c_int{ 1, 4, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0;

    // Reference mask: exactly createSlidingWindowMask semantics — masked when
    // col > row_abs (causal) or row_abs - col >= sw, row_abs = kL - qL + r.
    const mask_n: usize = @intCast(qL * kL);
    const mask_data = try std.testing.allocator.alloc(f32, mask_n);
    defer std.testing.allocator.free(mask_data);
    const off: c_int = kL - qL;
    var r: c_int = 0;
    while (r < qL) : (r += 1) {
        var c: c_int = 0;
        while (c < kL) : (c += 1) {
            const row_abs = off + r;
            const masked = (c > row_abs) or (row_abs - c >= sw);
            mask_data[@intCast(r * kL + c)] = if (masked) -std.math.inf(f32) else 0.0;
        }
    }
    const mask_shape = [_]c_int{ 1, 1, qL, kL };
    const mask_f32 = mlx.mlx_array_new_data(mask_data.ptr, &mask_shape, 4, .float32);
    defer _ = mlx.mlx_array_free(mask_f32);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));

    const fused = (try fusedSdpa256Prefill(s, q, k, v, scale, sw)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);

    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.02);
}

test "fusedSinkAttnPrefill: q/k 192, v 128 with sinks matches MLX sdpa (causal and band, with a cache prefix)" {
    // MiMo's attention: K 192 / V 128, GQA 8, a learned sink per head. The
    // reference is MLX's composed sdpa with the same sinks and an explicit mask.
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5191);
    const rnd = prng.random();
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    const Hq: c_int = 16;
    const Hk: c_int = 2;
    const qL: c_int = 96;
    for ([_]c_int{ 96, 200 }) |kL| for ([_]c_int{ 0, 40 }) |sw| {
        const q = try attn256RandBf16(rnd, &.{ 1, Hq, qL, 192 }, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try attn256RandBf16(rnd, &.{ 1, Hk, kL, 192 }, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try attn256RandBf16(rnd, &.{ 1, Hk, kL, 128 }, s);
        defer _ = mlx.mlx_array_free(v);
        const sinks = try attn256RandBf16Scaled(rnd, &.{Hq}, 4.0, s);
        defer _ = mlx.mlx_array_free(sinks);
        const scale: f32 = 1.0 / @sqrt(192.0);

        const mask_data = try std.testing.allocator.alloc(f32, @intCast(qL * kL));
        defer std.testing.allocator.free(mask_data);
        for (0..@intCast(qL)) |r| for (0..@intCast(kL)) |c| {
            const row_abs: c_int = kL - qL + @as(c_int, @intCast(r));
            const col: c_int = @intCast(c);
            const masked = col > row_abs or (sw > 0 and row_abs - col >= sw);
            mask_data[r * @as(usize, @intCast(kL)) + c] = if (masked) -std.math.inf(f32) else 0.0;
        };
        const mask_f32 = mlx.mlx_array_new_data(mask_data.ptr, &[_]c_int{ 1, 1, qL, kL }, 4, .float32);
        defer _ = mlx.mlx_array_free(mask_f32);
        var mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask);
        try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));

        const fused = (try fusedSinkAttnPrefill(s, q, k, v, scale, sw, sinks)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(fused);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, k, v, scale, "array", mask, sinks, false, s));
        const max_diff = try attn256MaxDiff(fused, ref, s);
        std.testing.expect(max_diff < 0.02) catch |e| {
            std.debug.print("\n[sink-attn] kL={d} sw={d} max diff {d:.4}\n", .{ kL, sw, max_diff });
            return e;
        };
    };
}

test "fusedSdpa256Prefill: declines cleanly outside its envelope" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xDEC1);
    const rnd = prng.random();

    // Wrong head_dim -> null.
    const q128_shape = [_]c_int{ 1, 4, 64, 128 };
    const q128 = try attn256RandBf16(rnd, &q128_shape, s);
    defer _ = mlx.mlx_array_free(q128);
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    try std.testing.expect((try fusedSdpa256Prefill(s, q128, q128, q128, 1.0, 0)) == null);

    // Decode/verify shapes (q_len < 16) -> null (sdpa_vector owns hd 256
    // there; stealing MTP verify forwards measured decode 48 -> 18 tok/s;
    // the 16 floor matches oMLX's _MIN_ROUTE_Q_LEN — with causal fused
    // default-on, a depth-8 MTP verify is q_len 9 and must NOT route here).
    const q1_shape = [_]c_int{ 1, 4, 1, 256 };
    const q8_shape = [_]c_int{ 1, 4, 8, 256 };
    const q15_shape = [_]c_int{ 1, 4, 15, 256 };
    const kv_shape = [_]c_int{ 1, 4, 64, 256 };
    const q1 = try attn256RandBf16(rnd, &q1_shape, s);
    defer _ = mlx.mlx_array_free(q1);
    const q8 = try attn256RandBf16(rnd, &q8_shape, s);
    defer _ = mlx.mlx_array_free(q8);
    const q15 = try attn256RandBf16(rnd, &q15_shape, s);
    defer _ = mlx.mlx_array_free(q15);
    const k1 = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k1);
    try std.testing.expect((try fusedSdpa256Prefill(s, q1, k1, k1, 1.0, 0)) == null);
    try std.testing.expect((try fusedSdpa256Prefill(s, q8, k1, k1, 1.0, 0)) == null);
    try std.testing.expect((try fusedSdpa256Prefill(s, q15, k1, k1, 1.0, 0)) == null);

    // Kill switch -> null even for a conforming call.
    attn256.fused256_override = false;
    const q_shape = [_]c_int{ 1, 4, 64, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    try std.testing.expect((try fusedSdpa256Prefill(s, q, q, q, 1.0, 0)) == null);
}

test "fusedSdpa256Prefill: both arms FUSED off NAX, causal yields to stock ON NAX" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x4A7E);
    const rnd = prng.random();

    // No master-switch override: the causal arm's historical net-loss (every
    // pre-budget ratio-gated variant lost same-boot on the 27B) was the IOGPU
    // preemption class — with the kv-chunk dispatch budget it wins live
    // ( same-session A/B: +2.9%/+2.3%/+4.6% at 8K/16K/32K on the
    // 27B), so causal is default-on now.
    std.debug.assert(attn256.fused256_override == null);
    const q_shape = [_]c_int{ 1, 6, 64, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const kv_shape = [_]c_int{ 1, 2, 64, 256 };
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);

    // Off NAX both arms engage — the budgeted-dispatch flip this test was
    // written for. The preference is PINNED rather than read off the host:
    // `naxSdpaPreferred()` defaults to whether the machine has NAX, so leaving
    // it to the silicon makes this assertion true on an M4 and false on an M5.
    sdpa.nax_sdpa_override = false;
    defer sdpa.nax_sdpa_override = null;
    {
        const causal = try fusedSdpa256Prefill(s, q, k, k, 1.0, 0);
        try std.testing.expect(causal != null);
        if (causal) |f| _ = mlx.mlx_array_free(f);
    }

    // On NAX the causal arm yields to the stock fused sdpa (mlx's own hd-256
    // kernel) while the BAND arm stays ours — mlx's NAX kernel takes no band.
    // Nothing pinned that asymmetry before; a decline is invisible from the
    // outside, since the composed fallback is output-equivalent.
    sdpa.nax_sdpa_override = true;
    {
        const causal = try fusedSdpa256Prefill(s, q, k, k, 1.0, 0);
        try std.testing.expect(causal == null);
    }

    const banded = try fusedSdpa256Prefill(s, q, k, k, 1.0, 40);
    try std.testing.expect(banded != null);
    if (banded) |f| _ = mlx.mlx_array_free(f);
}

test "fused256KvChunkLen: BK alignment, one-block floor, kl cap, budget-off" {
    const t = std.testing;
    // budget <= 0: single dispatch covering the full key axis.
    try t.expectEqual(@as(c_int, 193), fused256KvChunkLen(1, 6, 70, 193, 0));
    try t.expectEqual(@as(c_int, 65536), fused256KvChunkLen(1, 24, 2048, 65536, -1));
    // 27B live shape (24 heads, 2048-chunk q) at the default budget:
    // 250M / (24*2048) = 5086 keys -> floored to the BK(32) multiple 5056.
    try t.expectEqual(@as(c_int, 5056), fused256KvChunkLen(1, 24, 2048, 65536, FUSED256_DEFAULT_DISPATCH_BUDGET));
    // Never below one BK block even at absurdly small budgets…
    try t.expectEqual(@as(c_int, 32), fused256KvChunkLen(1, 24, 2048, 65536, 1));
    // …and never above the actual key length.
    try t.expectEqual(@as(c_int, 193), fused256KvChunkLen(1, 6, 70, 193, 1 << 40));
}

test "fusedSdpa256Prefill: budgeted kv chunking engages and is exact vs single dispatch" {
    const s = mlx.gpuStream();
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    // Causal arm: declines on NAX silicon unless the preference is pinned.
    sdpa.nax_sdpa_override = false;
    defer sdpa.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0xC4A2);
    const rnd = prng.random();

    // Hq=6/Hk=2, qL=70 (ragged q tile), kL=193 (ragged kv block). Per-key
    // work = 6*70 = 420; budget 40000 -> chunk 64 -> dispatches over
    // [0,64) [64,128) [128,192) [192,193) — 4, incl. a 1-key ragged tail.
    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0;

    attn256.fused256_budget_override = 0;
    const single = (try fusedSdpa256Prefill(s, q, k, v, scale, 0)) orelse {
        attn256.fused256_budget_override = null;
        return error.FusedDeclined;
    };
    defer _ = mlx.mlx_array_free(single);
    try std.testing.expectEqual(@as(u32, 1), attn256.fused256_last_dispatch_count);

    attn256.fused256_budget_override = 40_000;
    defer attn256.fused256_budget_override = null;
    const chunked = (try fusedSdpa256Prefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(chunked);
    // Engagement counted, not inferred: the budget must actually split.
    try std.testing.expectEqual(@as(u32, 4), attn256.fused256_last_dispatch_count);

    // The carry rides fp32 buffers — the exact register precision — so the
    // chunked result is BIT-IDENTICAL to the single dispatch, not just close.
    const max_diff = try attn256MaxDiff(single, chunked, s);
    try std.testing.expect(max_diff == 0.0);

    // And still correct in absolute terms vs the composed reference.
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);
    const ref_diff = try attn256MaxDiff(chunked, ref, s);
    try std.testing.expect(ref_diff < 0.005);
}

test "fusedSdpa256Prefill: band arm never chunks (single dispatch under a tiny budget)" {
    const s = mlx.gpuStream();
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    attn256.fused256_budget_override = 1; // would force max chunking on the causal arm
    defer attn256.fused256_budget_override = null;
    var prng = std.Random.DefaultPrng.init(0xBA9D2);
    const rnd = prng.random();

    const qL: c_int = 70;
    const kL: c_int = 193;
    const sw: c_int = 40;
    const q_shape = [_]c_int{ 1, 4, qL, 256 };
    const kv_shape = [_]c_int{ 1, 4, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);

    const fused = (try fusedSdpa256Prefill(s, q, k, v, 1.0, sw)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    // The band arm's block skip already bounds per-dispatch work; chunking it
    // would only add carry traffic. It must stay a single dispatch.
    try std.testing.expectEqual(@as(u32, 1), attn256.fused256_last_dispatch_count);

    // Unchanged band semantics (same reference as the band parity test).
    const mask_n: usize = @intCast(qL * kL);
    const mask_data = try std.testing.allocator.alloc(f32, mask_n);
    defer std.testing.allocator.free(mask_data);
    const off: c_int = kL - qL;
    var r: c_int = 0;
    while (r < qL) : (r += 1) {
        var c: c_int = 0;
        while (c < kL) : (c += 1) {
            const row_abs = off + r;
            const masked = (c > row_abs) or (row_abs - c >= sw);
            mask_data[@intCast(r * kL + c)] = if (masked) -std.math.inf(f32) else 0.0;
        }
    }
    const mask_shape = [_]c_int{ 1, 1, qL, kL };
    const mask_f32 = mlx.mlx_array_new_data(mask_data.ptr, &mask_shape, 4, .float32);
    defer _ = mlx.mlx_array_free(mask_f32);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));
    const ref = try attn256Reference(q, k, v, 1.0, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.02);
}

test "fusedSdpa256Prefill: causal parity at Qwen 24q/4kv geometry (gqa 6, ragged 64-row tile)" {
    const s = mlx.gpuStream();
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    // Causal arm: declines on NAX silicon unless the preference is pinned.
    sdpa.nax_sdpa_override = false;
    defer sdpa.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x27B27B);
    const rnd = prng.random();

    // The production Qwen3.6-27B full-attention geometry: 24 q heads over 4
    // kv heads (gqa 6), hd 256. qL=97 exercises a partial 64-row q tile,
    // kL=250 a partial kv block; offset 153 = chunked-prefill alignment.
    const q_shape = [_]c_int{ 1, 24, 97, 256 };
    const kv_shape = [_]c_int{ 1, 4, 250, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0;

    const fused = (try fusedSdpa256Prefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);

    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.005);
}
