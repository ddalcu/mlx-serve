//! engines/mlx/kv/cache: the KV cache lives in the SDK (`sdk/kv/cache.zig`); its tests stay here.

const std = @import("std");
const sdk_cache = @import("../../../sdk/kv/cache.zig");
const mlx = @import("../../../mlx/ffi.zig");
const kv_quant = @import("../cache/kv_quant.zig");
const KVQuantConfig = @import("../cache/kv_quant.zig").KVQuantConfig;
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const qsaArraysAllEqual = @import("../../../sdk/testing/parity.zig").qsaArraysAllEqual;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const bf16BitsEqual = @import("../../../sdk/testing/parity.zig").bf16BitsEqual;
const mlxDeviceUsable = @import("../../../sdk/testing/parity.zig").mlxDeviceUsable;
const testRandWeightBf16 = @import("../../../sdk/testing/parity.zig").testRandWeightBf16;
const SLIDING_TRIM_UNBOUNDED = @import("sliding.zig").SLIDING_TRIM_UNBOUNDED;
const slidingTailSpan = @import("sliding.zig").slidingTailSpan;
const slidingViewFor = @import("sliding.zig").slidingViewFor;
const attn256_vars = @import("../../../sdk/blocks/attention/attn256.zig");

pub const KVCacheEntry = sdk_cache.KVCacheEntry;
pub const newEmptyKVEntry = sdk_cache.newEmptyKVEntry;
pub const resetKVEntry = sdk_cache.resetKVEntry;
pub const freeKVEntry = sdk_cache.freeKVEntry;
pub const KVCache = sdk_cache.KVCache;
pub const KVCacheSnapshot = sdk_cache.KVCacheSnapshot;

// ── Tests ──

pub const testing = std.testing;

// ── tests ──

/// Helper: create a dummy K or V tensor of shape [1, 1, seq_len, 1].
fn testKV(seq_len: usize, s: mlx.mlx_stream) mlx.mlx_array {
    const sl: c_int = @intCast(seq_len);
    const shape = [_]c_int{ 1, 1, sl, 1 };
    var arr = mlx.mlx_array_new();
    _ = mlx.mlx_zeros(&arr, &shape, 4, .float32, s);
    return arr;
}

test "slidingTailSpan covers the whole block, not just the last query" {
    // Decode width: unchanged, exactly the window.
    try testing.expectEqual(@as(u32, 2048), slidingTailSpan(2048, 1, SLIDING_TRIM_UNBOUNDED));
    // A block of q queries needs window + q - 1 tail entries: the FIRST query
    // in the block reaches back furthest.
    try testing.expectEqual(@as(u32, 2052), slidingTailSpan(2048, 5, SLIDING_TRIM_UNBOUNDED));
    try testing.expectEqual(@as(u32, 2063), slidingTailSpan(2048, 16, SLIDING_TRIM_UNBOUNDED));
    // A PREFILL CHUNK is not special-cased any more: the old 32-wide cap kept
    // laguna and inkling — serial archs whose only multi-token forward IS a
    // chunk — from ever taking the trim.
    try testing.expectEqual(@as(u32, 2080), slidingTailSpan(2048, 33, SLIDING_TRIM_UNBOUNDED));
    try testing.expectEqual(@as(u32, 2559), slidingTailSpan(512, 2048, SLIDING_TRIM_UNBOUNDED));
    // Non-sliding layer and empty forward still decline.
    try testing.expectEqual(@as(u32, 0), slidingTailSpan(0, 5, SLIDING_TRIM_UNBOUNDED));
    try testing.expectEqual(@as(u32, 0), slidingTailSpan(2048, 0, SLIDING_TRIM_UNBOUNDED));
    // max_width 1 is how the policy pins a forward to decode-only trimming.
    try testing.expectEqual(@as(u32, 4), slidingTailSpan(4, 1, 1));
    try testing.expectEqual(@as(u32, 0), slidingTailSpan(4, 2, 1));
}

test "slidingViewFor declines the trim only where the fused kernel really bands" {
    var cfg = ModelConfig{};
    cfg.has_sliding_window = true;
    cfg.sliding_window = 512;

    // hd 256 with the fused kernel on. The kernel declines everything below
    // FUSED256_MIN_Q_LEN, so a spec-verify block is NOT band-masked in-kernel
    // and must take the trim — the predicate used to say `seq_len >= 2`, which
    // handed gemma nothing at exactly the widths spec decode runs at.
    cfg.head_dim = 256;
    attn256_vars.fused256_override = true;
    defer attn256_vars.fused256_override = null;
    {
        const decode = slidingViewFor(&cfg, 20000, 1);
        try testing.expect(!decode.band_in_kernel);
        try testing.expectEqual(@as(u32, 512), decode.span);
        try testing.expectEqual(@as(c_int, 512), decode.kv_len);

        const verify = slidingViewFor(&cfg, 20000, 8);
        try testing.expect(!verify.band_in_kernel);
        try testing.expectEqual(@as(u32, 519), verify.span);
        try testing.expectEqual(@as(c_int, 519), verify.kv_len);

        // At the kernel's floor and above, the band IS in-kernel: it derives
        // absolute positions from the K length, so the view stays WHOLE
        // (span 0) and any mask built for it is built at total_kv.
        const chunk = slidingViewFor(&cfg, 20000, FUSED256_MIN_Q_LEN);
        try testing.expect(chunk.band_in_kernel);
        try testing.expectEqual(@as(u32, 0), chunk.span);
        try testing.expectEqual(@as(c_int, 20000), chunk.kv_len);
    }

    // hd 128 (laguna, inkling): no fused kernel, so even a wide prefill chunk
    // trims — that is the whole win on a serial arch.
    cfg.head_dim = 128;
    {
        const chunk = slidingViewFor(&cfg, 32768, 2048);
        try testing.expect(!chunk.band_in_kernel);
        try testing.expectEqual(@as(u32, 2559), chunk.span);
        try testing.expectEqual(@as(c_int, 2559), chunk.kv_len);
    }

    // A cache shorter than the span is not trimmed, and the reported kv_len is
    // what the view will actually be — masks are built from it.
    {
        const short = slidingViewFor(&cfg, 300, 8);
        try testing.expectEqual(@as(c_int, 300), short.kv_len);
    }

    // Non-sliding config: no trim, kv_len is the whole cache.
    cfg.has_sliding_window = false;
    {
        const full = slidingViewFor(&cfg, 20000, 8);
        try testing.expectEqual(@as(u32, 0), full.span);
        try testing.expectEqual(@as(c_int, 20000), full.kv_len);
    }
}

test "KVCache trims the tail view for a BLOCK-wide update, not just decode" {
    // The spec-verify shape: a block of queries lands at the tail of a cache
    // already longer than the window. Before the block trim, `new_len > 1`
    // fell through with view_start 0 and the layer read the ENTIRE cache.
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // 10 prefill positions, untrimmed (a wide chunk keeps the full view).
    {
        const k = testKV(10, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(10, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        dv.deinit();
    }
    try testing.expectEqual(@as(usize, 10), cache.entries[0].offset);

    // Now a 3-wide verify block against a window of 4 → span 4 + 3 - 1 = 6.
    const span = slidingTailSpan(4, 3, SLIDING_TRIM_UNBOUNDED);
    try testing.expectEqual(@as(u32, 6), span);
    {
        const k = testKV(3, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(3, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, span);
        defer dv.deinit();
        const view_shape = mlx.getShape(dv.k);
        try testing.expectEqual(@as(c_int, 6), view_shape[2]);
        const v_shape = mlx.getShape(dv.v);
        try testing.expectEqual(@as(c_int, 6), v_shape[2]);
    }
    // Absolute bookkeeping is untouched by the view trim.
    try testing.expectEqual(@as(usize, 13), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 13), cache.step);
}

fn testKVWide(seq_len: usize, head_dim: c_int, s: mlx.mlx_stream) mlx.mlx_array {
    const sl: c_int = @intCast(seq_len);
    const shape = [_]c_int{ 1, 2, sl, head_dim };
    var arr = mlx.mlx_array_new();
    _ = mlx.mlx_zeros(&arr, &shape, 4, .float32, s);
    return arr;
}

test "KVCache resident bytes are the buffers it owns, never a restored donor's" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testing.expectEqual(@as(u64, 0), cache.residentBytes());
    const k = testKVWide(3, 64, s);
    defer _ = mlx.mlx_array_free(k);
    var dv = try cache.update(0, k, k, s, 0);
    dv.deinit();
    const e = cache.entries[0];
    const want = 2 * @as(u64, mlx.mlx_array_size(e.keys)) * @as(u64, mlx.mlx_array_itemsize(e.keys));
    try testing.expect(want > 0);
    try testing.expectEqual(want, cache.residentBytes());
    cache.entries[0].shared_view = true;
    try testing.expectEqual(@as(u64, 0), cache.residentBytes());
    cache.entries[0].shared_view = false;
}

test "KVCache carries a V head dim narrower than K (MLA 192/128)" {
    // MLA scores over nope+rope (192) but stores values at v_head_dim (128).
    // The dense path used to derive BOTH buffers from the K shape, so the V
    // slice_update wrote a 128-wide chunk into a 192-wide window and the cache
    // silently mis-shaped — grow, write, view and truncate each have to read
    // their OWN buffer's last dim. MLX's SDPA has a 192/128 vector kernel, so
    // this geometry is served, not padded.
    //
    // A cache entry's geometry is fixed by its FIRST write, so this needs its
    // own cache — the 192/128 chunk cannot land in a buffer another test's
    // narrow prefill already shaped.
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // Prefill 3 tokens, then two decode steps that grow both buffers.
    {
        const k = testKVWide(3, 192, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKVWide(3, 128, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        defer dv.deinit();
        try testing.expectEqual(@as(c_int, 192), mlx.getShape(dv.k)[3]);
        try testing.expectEqual(@as(c_int, 128), mlx.getShape(dv.v)[3]);
        try testing.expectEqual(@as(c_int, 3), mlx.getShape(dv.v)[2]);
    }
    for (0..2) |_| {
        const k = testKVWide(1, 192, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKVWide(1, 128, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        defer dv.deinit();
        try testing.expectEqual(@as(c_int, 192), mlx.getShape(dv.k)[3]);
        try testing.expectEqual(@as(c_int, 128), mlx.getShape(dv.v)[3]);
    }
    try testing.expectEqual(@as(usize, 5), cache.entries[0].offset);
    // The views must be evaluable: a mis-shaped slice_update throws here.
    try mlx.check(mlx.mlx_array_eval(cache.entries[0].key_view));
    try mlx.check(mlx.mlx_array_eval(cache.entries[0].value_view));

    // Truncation rebuilds both views from their own buffers.
    try cache.truncate(2, s);
    try testing.expectEqual(@as(c_int, 2), mlx.getShape(cache.entries[0].value_view)[2]);
    try testing.expectEqual(@as(c_int, 128), mlx.getShape(cache.entries[0].value_view)[3]);
    try testing.expectEqual(@as(c_int, 192), mlx.getShape(cache.entries[0].key_view)[3]);
    try mlx.check(mlx.mlx_array_eval(cache.entries[0].value_view));
}

test "KVCache affine quant carries an asymmetric K/V too (--kv-quant on MLA)" {
    // The QUANTIZED write path solved every buffer's packed and scale widths
    // from K's head dim, so at K 192 / V 128 the value buffer was 24 u32 wide
    // while the incoming chunk was 16 — an mlx slice_update mismatch inside a
    // path we cannot catch. Nothing gated the SCHEME (the "MLA never opts in"
    // note covers the fused READ kernels only), so `--kv-quant 8` plus this
    // arch was a request-killing shape error. Each of the six buffers is sized
    // from its own operand; identical on every symmetric arch.
    const s = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();

    const mk = struct {
        fn f(str: mlx.mlx_stream, len: c_int, d: c_int) !mlx.mlx_array {
            const shape = [_]c_int{ 1, 2, len, d };
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&a, &shape, 4, .bfloat16, str));
            return a;
        }
    }.f;

    // Prefill 4, then two decode steps (the second crosses a growth event).
    {
        const k = try mk(s, 4, 192);
        defer _ = mlx.mlx_array_free(k);
        const v = try mk(s, 4, 128);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        defer dv.deinit();
        try testing.expect(dv.has_quant_triple);
        // 8-bit affine, group 64: K packs 192/4 = 48 u32 with 3 groups, V
        // 128/4 = 32 u32 with 2 — the point of the fix.
        try testing.expectEqual(@as(c_int, 48), mlx.getShape(dv.k_triple_q)[3]);
        try testing.expectEqual(@as(c_int, 32), mlx.getShape(dv.v_triple_q)[3]);
        try testing.expectEqual(@as(c_int, 3), mlx.getShape(dv.k_triple_scales)[3]);
        try testing.expectEqual(@as(c_int, 2), mlx.getShape(dv.v_triple_scales)[3]);
        // The dequantized views must be evaluable at their own widths, and the
        // values must survive the round trip (ones in, ones out).
        try mlx.check(mlx.mlx_array_eval(dv.v));
        try testing.expectEqual(@as(c_int, 192), mlx.getShape(dv.k)[3]);
        try testing.expectEqual(@as(c_int, 128), mlx.getShape(dv.v)[3]);
        var vf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vf);
        try mlx.check(mlx.mlx_astype(&vf, dv.v, .float32, s));
        try mlx.check(mlx.mlx_array_eval(vf));
        const vd = mlx.mlx_array_data_float32(vf) orelse return error.InvalidDtype;
        try testing.expect(@abs(vd[0] - 1.0) < 1e-2);
        try testing.expect(@abs(vd[127] - 1.0) < 1e-2);
    }
    for (0..2) |_| {
        const k = try mk(s, 1, 192);
        defer _ = mlx.mlx_array_free(k);
        const v = try mk(s, 1, 128);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        defer dv.deinit();
        try mlx.check(mlx.mlx_array_eval(dv.k));
        try mlx.check(mlx.mlx_array_eval(dv.v));
    }
    try testing.expectEqual(@as(usize, 6), cache.entries[0].offset);

    // Truncation slices the value scale/bias views against THEIR width, not the
    // key scales' (2 groups vs 3).
    try cache.truncate(3, s);
    try testing.expectEqual(@as(c_int, 2), mlx.getShape(cache.entries[0].value_scales_view)[3]);
    try testing.expectEqual(@as(c_int, 3), mlx.getShape(cache.entries[0].key_scales_view)[3]);
    var restored = try cache.denseView(0, s);
    defer restored.deinit();
    try mlx.check(mlx.mlx_array_eval(restored.v));
    try testing.expectEqual(@as(c_int, 3), mlx.getShape(restored.v)[2]);
    try testing.expectEqual(@as(c_int, 128), mlx.getShape(restored.v)[3]);
}

test "a REFUSED KVCache re-init leaves the live cache intact, not freed" {
    // Every re-init site once read `cache.deinit(); cache = try init(...)`, so
    // a failed init left a FREED cache installed on the Transformer and its
    // owner's deinit freed every handle a second time — EXC_BAD_ACCESS in
    // `mlx_array_free`, several frames from the refusal that caused it. The
    // refusal here is the allocator's (the entries alloc of the second init);
    // under the testing allocator the pre-fix order trips the same double
    // free on this test's own `defer`.
    const s = mlx.gpuStream();
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 1 });
    var cache = try KVCache.initWithConfig(failing.allocator(), 2, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();

    // Real handles in the cache, so a double free has something to land on.
    {
        const k = testKVWide(3, 192, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKVWide(3, 128, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        dv.deinit();
    }
    try testing.expectEqual(@as(usize, 3), cache.entries[0].offset);

    try testing.expectError(error.OutOfMemory, cache.reinit(2, kv_quant.KVQuantConfig.affine(4)));
    failing.fail_index = std.math.maxInt(usize);

    // Untouched: same scheme, same contents, same entry count...
    try testing.expectEqual(kv_quant.Scheme.affine, cache.config.scheme);
    try testing.expectEqual(@as(u8, 8), cache.config.bits);
    try testing.expectEqual(@as(usize, 2), cache.entries.len);
    try testing.expectEqual(@as(usize, 3), cache.entries[0].offset);
    // ...and still SERVING, which a freed-but-readable cache would not be.
    {
        const k = testKVWide(1, 192, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKVWide(1, 128, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        defer dv.deinit();
        try mlx.check(mlx.mlx_array_eval(dv.k));
        try mlx.check(mlx.mlx_array_eval(dv.v));
    }
    try testing.expectEqual(@as(usize, 4), cache.entries[0].offset);

    // A re-init that SUCCEEDS still swaps — the fix must not turn the helper
    // into a no-op: new scheme, fresh entries, offsets back to zero.
    try cache.reinit(2, kv_quant.KVQuantConfig.affine(4));
    try testing.expectEqual(@as(u8, 4), cache.config.bits);
    try testing.expectEqual(@as(usize, 0), cache.entries[0].offset);
    try testing.expect(!cache.entries[0].initialized);
    try testing.expectEqual(@as(usize, 0), cache.step);
}

test "KVCache sliding window views return last max_seq entries" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // Simulate 3 prefill tokens (max_seq=4 sliding window)
    {
        const k = testKV(3, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(3, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 4);
        dv.deinit();
    }
    // After prefill: offset=3, step=3
    try testing.expectEqual(@as(usize, 3), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 3), cache.step);

    // Decode token 4 — still within window
    {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 4);
        dv.deinit();
    }
    try testing.expectEqual(@as(usize, 4), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 4), cache.step);

    // Decode token 5 — exceeds window, but buffer grows (no trimming)
    // Views return last 4 entries only
    {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 4);
        defer dv.deinit();
        // View should be 4 entries (max_seq), not 5
        const view_shape = mlx.getShape(dv.k);
        try testing.expectEqual(@as(c_int, 4), view_shape[2]);
    }
    // Buffer has 5 entries, but view shows 4. step=5 (absolute).
    try testing.expectEqual(@as(usize, 5), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 5), cache.step);

    // Decode tokens 6,7,8 — step keeps incrementing, views stay at max_seq
    for (0..3) |_| {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 4);
        defer dv.deinit();
        const view_shape = mlx.getShape(dv.k);
        try testing.expectEqual(@as(c_int, 4), view_shape[2]);
    }
    try testing.expectEqual(@as(usize, 8), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 8), cache.step);
}

test "KVCache updateAffine returns quant triples so fused attention can engage" {
    // Engagement-bug guard (docs/kv-quant-perf.md Phase 0/1): the decode
    // forward calls `cache.update(...)`, NOT `denseView`, so the triples must
    // ride the update path's DenseKVView too — otherwise `--kv-attn-mode
    // fused` silently never engages (output-identical to dense mode, which is
    // why an equivalence test can't see it).
    const s = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();

    const shape = [_]c_int{ 1, 2, 4, 64 };
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    try mlx.check(mlx.mlx_ones(&k, &shape, 4, .bfloat16, s));
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_ones(&v, &shape, 4, .bfloat16, s));

    var dv = try cache.update(0, k, v, s, 0);
    defer dv.deinit();
    try testing.expect(dv.has_quant_triple);
    try testing.expect(dv.k_triple_q.ctx != null);
    try testing.expect(dv.v_triple_q.ctx != null);
    try testing.expectEqual(@as(u8, 8), dv.bits);
    try testing.expectEqual(@as(u32, 64), dv.group_size);
}

test "KVCache affine quant hands attention the dtype it was fed" {
    const s = mlx.gpuStream();
    for ([_]mlx.mlx_dtype{ .float16, .bfloat16 }) |dt| {
        var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(8));
        defer cache.deinit();
        const shape = [_]c_int{ 1, 2, 4, 64 };
        var k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k);
        try mlx.check(mlx.mlx_ones(&k, &shape, 4, dt, s));
        var dv = try cache.update(0, k, k, s, 0);
        defer dv.deinit();
        try testing.expectEqual(dt, mlx.mlx_array_dtype(dv.k));
        try testing.expectEqual(dt, mlx.mlx_array_dtype(dv.v));
        var dv2 = try cache.denseView(0, s);
        defer dv2.deinit();
        try testing.expectEqual(dt, mlx.mlx_array_dtype(dv2.k));
    }
}

test "KVCache step resets on truncate" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // Add 5 tokens
    {
        const k = testKV(5, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(5, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }
    try testing.expectEqual(@as(usize, 5), cache.step);

    // Truncate to 3 (simulating KV cache reuse)
    try cache.truncate(3, s);
    try testing.expectEqual(@as(usize, 3), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 3), cache.step);
}

test "KVCache step without trimming matches offset" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // Add tokens without max_seq (no trimming)
    for (0..10) |_| {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }
    // Without trimming, step and offset should be equal
    try testing.expectEqual(@as(usize, 10), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 10), cache.step);
}

test "kvLenForBatching: the batched cap reads the ATTENTION layers, never layer 0's step" {
    // `KVCache.step` advances on global layer 0 only, so on a linear-layer-0 trunk it reads 0
    // forever; the probe is "is entry 0 initialized?", not an arch list.
    const families = [_]struct { layers: u32, first_attn: u32, len: usize }{
        .{ .layers = 48, .first_attn = 3, .len = 60_000 }, // qwen3_5 / qwen4_exp GDN
        .{ .layers = 16, .first_attn = 2, .len = 4_096 }, // lfm2 gated-conv
        .{ .layers = 52, .first_attn = 4, .len = 1_000 }, // nemotron_h mamba2
        .{ .layers = 32, .first_attn = 7, .len = 131_072 }, // bailing_hybrid KDA
    };
    for (families) |f| {
        var cache = try KVCache.init(testing.allocator, f.layers);
        defer cache.deinit();
        var li = f.first_attn;
        while (li < f.layers) : (li += 4) {
            cache.entries[li].initialized = true;
            cache.entries[li].offset = f.len;
        }
        try testing.expectEqual(@as(usize, 0), cache.step); // the trap
        try testing.expectEqual(f.len, cache.kvLenForBatching());
    }

    // Attention-first trunk: `step` stays the answer.
    var attn = try KVCache.init(testing.allocator, 4);
    defer attn.deinit();
    attn.step = 4096;
    attn.entries[0].initialized = true;
    attn.entries[0].offset = 4096;
    try testing.expectEqual(@as(usize, 4096), attn.kvLenForBatching());

    var cold = try KVCache.init(testing.allocator, 4);
    defer cold.deinit();
    try testing.expectEqual(@as(usize, 0), cold.kvLenForBatching());
}

test "KVCache step with multi-layer only increments once per update" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();

    // Update all 3 layers with 2 tokens each
    for (0..3) |layer| {
        const k = testKV(2, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(2, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(@intCast(layer), k, v, s, 0);
    }
    // step should be 2 (one sequence worth), not 6 (3 layers × 2)
    try testing.expectEqual(@as(usize, 2), cache.step);
}

// ── Cache snapshot/restore (for spec-decode rollback) ──────────────────────

test "KVCache snapshot/restore round-trip preserves entries and step" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();

    // Build state: 4 tokens across 2 layers.
    for (0..2) |layer| {
        const k = testKV(4, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(4, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(@intCast(layer), k, v, s, 0);
    }
    try testing.expectEqual(@as(usize, 4), cache.step);
    try testing.expectEqual(@as(usize, 4), cache.entries[0].offset);

    // Snapshot at this point.
    var snap = try cache.snapshot();
    defer snap.deinit();

    // Mutate cache: add 2 more tokens to layer 0 only (to verify per-layer
    // offset is captured, not just global step).
    {
        const k = testKV(2, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(2, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }
    try testing.expectEqual(@as(usize, 6), cache.step);
    try testing.expectEqual(@as(usize, 6), cache.entries[0].offset);

    // Restore — step and per-layer offsets revert to snapshot.
    try cache.restore(&snap);
    try testing.expectEqual(@as(usize, 4), cache.step);
    try testing.expectEqual(@as(usize, 4), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 4), cache.entries[1].offset);
}

test "KVCache snapshot then more updates does not corrupt the snapshot" {
    // Critical invariant for spec-decode rollback: if we snapshot, then verify, then
    // (rejected) restore, snapshot must NOT have been mutated by the intervening
    // updates. The buffer is shared via refcount but must not be aliased through
    // the cache entry pointer.
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    {
        const k = testKV(2, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(2, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }

    var snap = try cache.snapshot();
    defer snap.deinit();
    try testing.expectEqual(@as(usize, 2), snap.entries[0].offset);

    // Run several updates to grow the cache buffer (forces buffer reallocation
    // inside update(), which would invalidate a naive snapshot).
    for (0..6) |_| {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }
    try testing.expectEqual(@as(usize, 8), cache.entries[0].offset);

    // Snapshot still reports its captured state.
    try testing.expectEqual(@as(usize, 2), snap.entries[0].offset);
    try testing.expectEqual(@as(usize, 2), snap.step);

    try cache.restore(&snap);
    try testing.expectEqual(@as(usize, 2), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 2), cache.step);
}

test "KVCache: an append after a short restore sizes its copy from the prefix, not the donor (#492)" {
    // A 4k "hi" session restored from an 80k agent session's entry kept an 80k-capacity copy,
    // and its commit evicted the agent session to fit the budget.
    const s = mlx.gpuStream();
    const mk = struct {
        fn f(str: mlx.mlx_stream, len: c_int) !mlx.mlx_array {
            const shape = [_]c_int{ 1, 2, len, 64 };
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&a, &shape, 4, .bfloat16, str));
            return a;
        }
    }.f;
    for ([_]kv_quant.KVQuantConfig{ kv_quant.KVQuantConfig.dense, kv_quant.KVQuantConfig.affine(8) }) |cfg| {
        var donor = try KVCache.initWithConfig(testing.allocator, 1, cfg);
        defer donor.deinit();
        {
            const k = try mk(s, 8192);
            defer _ = mlx.mlx_array_free(k);
            _ = try donor.update(0, k, k, s, 0);
        }
        var snap = try donor.snapshot();
        defer snap.deinit();

        var cache = try KVCache.initWithConfig(testing.allocator, 1, cfg);
        defer cache.deinit();
        try cache.restore(&snap);
        try cache.truncate(100, s);
        const k = try mk(s, 1);
        defer _ = mlx.mlx_array_free(k);
        _ = try cache.update(0, k, k, s, 0);

        try testing.expectEqual(@as(usize, 101), cache.entries[0].offset);
        try testing.expect(KVCache.bufferCapacity(cache.entries[0].keys) <= 512);
        try testing.expect(KVCache.bufferCapacity(snap.entries[0].keys) >= 8192);
    }
}

test "KVCache.compactRows: a tree's accepted path lands as serial appends would, dense and quantized" {
    const s = mlx.gpuStream();
    const base: usize = 6;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 42));
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_random_normal(&xf, &[_]c_int{ 1, 2, base + 8, 64 }, 4, .float32, 0.0, 1.0, key, s));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));
    const rows = [_]u32{ 2, 5 };
    for ([_]kv_quant.KVQuantConfig{ kv_quant.KVQuantConfig.dense, kv_quant.KVQuantConfig.affine(8) }) |cfg| {
        var tree = try KVCache.initWithConfig(testing.allocator, 1, cfg);
        defer tree.deinit();
        _ = try tree.update(0, x, x, s, 0);
        try tree.compactRows(base, &rows, s);
        try tree.truncate(base + 1 + rows.len, s);

        var serial = try KVCache.initWithConfig(testing.allocator, 1, cfg);
        defer serial.deinit();
        const idx_host = [_]i32{ 0, 1, 2, 3, 4, 5, 6, @intCast(base + rows[0]), @intCast(base + rows[1]) };
        const idx = mlx.mlx_array_new_data(&idx_host, &[_]c_int{idx_host.len}, 1, .int32);
        defer _ = mlx.mlx_array_free(idx);
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_take_axis(&picked, x, idx, 2, s));
        _ = try serial.update(0, picked, picked, s, 0);

        var a = try tree.denseView(0, s);
        defer a.deinit();
        var b = try serial.denseView(0, s);
        defer b.deinit();
        try testing.expectEqualSlices(c_int, mlx.getShape(b.k), mlx.getShape(a.k));
        for ([_][2]mlx.mlx_array{ .{ a.k, b.k }, .{ a.v, b.v } }) |pair| {
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_array_equal(&eq, pair[0], pair[1], false, s));
            var same = false;
            try mlx.check(mlx.mlx_array_item_bool(&same, eq));
            try testing.expect(same);
        }
    }
}

test "KVCache snapshot/restore in a tight loop does not leak" {
    // testing.allocator is a TrackingAllocator — any unfreed allocation here
    // surfaces as a test failure at the leak-detection step.
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();

    {
        const k = testKV(3, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(3, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
        const k2 = testKV(3, s);
        defer _ = mlx.mlx_array_free(k2);
        const v2 = testKV(3, s);
        defer _ = mlx.mlx_array_free(v2);
        _ = try cache.update(1, k2, v2, s, 0);
    }

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        var snap = try cache.snapshot();
        defer snap.deinit();
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
        try cache.restore(&snap);
    }
}

const kv_append_kv_h: c_int = 2;
const kv_append_hd: c_int = 256;

fn kvAppendPacked(n: c_int, seq: c_int, seed: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const nn: usize = @intCast(n * kv_append_kv_h * seq * kv_append_hd);
    const buf = try testing.allocator.alloc(f32, nn);
    defer testing.allocator.free(buf);
    var st: u32 = seed;
    for (buf) |*x| {
        st = st *% 1664525 +% 1013904223;
        x.* = @as(f32, @floatFromInt(st >> 16)) / 65536.0 - 0.5;
    }
    const shape = [_]c_int{ n, kv_append_kv_h, seq, kv_append_hd };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f32_arr, .bfloat16, s));
    return out;
}

fn kvAppendPrefill(cache: *KVCache, tokens: usize, seed: u32, s: mlx.mlx_stream) !void {
    const k = try kvAppendPacked(1, @intCast(tokens), seed, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try kvAppendPacked(1, @intCast(tokens), seed ^ 0x9E3779B9, s);
    defer _ = mlx.mlx_array_free(v);
    var dv = try cache.update(0, k, v, s, 0);
    dv.deinit();
}

fn kvAppendOperandsEqual(a: *const KVCacheEntry, b: *const KVCacheEntry, s: mlx.mlx_stream) !void {
    try testing.expectEqual(a.offset, b.offset);
    try testing.expect(try qsaArraysAllEqual(a.keys, b.keys, s));
    try testing.expect(try qsaArraysAllEqual(a.values, b.values, s));
    try testing.expect(try qsaArraysAllEqual(a.keys_scales, b.keys_scales, s));
    try testing.expect(try qsaArraysAllEqual(a.keys_biases, b.keys_biases, s));
    try testing.expect(try qsaArraysAllEqual(a.values_scales, b.values_scales, s));
    try testing.expect(try qsaArraysAllEqual(a.values_biases, b.values_biases, s));
}

fn kvAppendDummyAttn(cache: *KVCache, s: mlx.mlx_stream) !mlx.mlx_array {
    var dv = try cache.denseView(0, s);
    defer dv.deinit();
    const ksh = mlx.getShape(dv.k);
    const q_shape = [_]c_int{ ksh[0], ksh[1], 1, ksh[3] };
    var q = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_ones(&q, &q_shape, 4, .bfloat16, s));
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kv_append_hd)));
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &out,
        q,
        dv.k,
        dv.v,
        scale,
        "",
        none,
        .{ .ctx = null },
        false,
        s,
    ));
    _ = mlx.mlx_array_free(q);
    return out;
}

test "batched KV append: affine-8 N=2/4/8 matches N solo updates at production geometry" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = testing.allocator;
    const s = mlx.gpuStream();
    const cfg = kv_quant.KVQuantConfig.affine(8);
    const ns = [_]usize{ 2, 4, 8 };
    const lens = [_]usize{ 2, 5, 3, 8, 4, 7, 6, 9 };
    for (ns) |n| {
        const solo = try alloc.alloc(KVCache, n);
        const batched = try alloc.alloc(KVCache, n);
        defer {
            for (solo) |*c| c.deinit();
            alloc.free(solo);
            for (batched) |*c| c.deinit();
            alloc.free(batched);
        }
        const solo_ptrs = try alloc.alloc(*KVCache, n);
        defer alloc.free(solo_ptrs);
        const batched_ptrs = try alloc.alloc(*KVCache, n);
        defer alloc.free(batched_ptrs);
        for (0..n) |i| {
            solo[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            batched[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            try kvAppendPrefill(&solo[i], lens[i], @intCast(0xA000 + i), s);
            try kvAppendPrefill(&batched[i], lens[i], @intCast(0xA000 + i), s);
            solo_ptrs[i] = &solo[i];
            batched_ptrs[i] = &batched[i];
        }
        const stacked_k = try kvAppendPacked(@intCast(n), 1, 0xC0FFE, s);
        defer _ = mlx.mlx_array_free(stacked_k);
        const stacked_v = try kvAppendPacked(@intCast(n), 1, 0x0D15EA, s);
        defer _ = mlx.mlx_array_free(stacked_v);
        try KVCache.appendFromStacked(stacked_k, stacked_v, solo_ptrs, 0, s, 0, false);
        try KVCache.appendFromStacked(stacked_k, stacked_v, batched_ptrs, 0, s, 0, true);
        for (0..n) |i| {
            try kvAppendOperandsEqual(&solo[i].entries[0], &batched[i].entries[0], s);
            const attn_s = try kvAppendDummyAttn(&solo[i], s);
            defer _ = mlx.mlx_array_free(attn_s);
            const attn_b = try kvAppendDummyAttn(&batched[i], s);
            defer _ = mlx.mlx_array_free(attn_b);
            try testing.expect(try qsaArraysAllEqual(attn_s, attn_b, s));
        }
    }
}

test "batched KV append: two restored siblings append and truncate independently; the snapshot stays byte-identical" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = testing.allocator;
    const s = mlx.gpuStream();
    const cfg = kv_quant.KVQuantConfig.affine(8);
    var src = try KVCache.initWithConfig(alloc, 1, cfg);
    defer src.deinit();
    try kvAppendPrefill(&src, 6, 0x51A0, s);
    var snap = try src.snapshot();
    defer snap.deinit();
    const snap_k = try materializedOwnedCopy(s, snap.entries[0].keys);
    defer _ = mlx.mlx_array_free(snap_k);
    const snap_v = try materializedOwnedCopy(s, snap.entries[0].values);
    defer _ = mlx.mlx_array_free(snap_v);
    const snap_ks = try materializedOwnedCopy(s, snap.entries[0].keys_scales);
    defer _ = mlx.mlx_array_free(snap_ks);
    const snap_kb = try materializedOwnedCopy(s, snap.entries[0].keys_biases);
    defer _ = mlx.mlx_array_free(snap_kb);
    const snap_vs = try materializedOwnedCopy(s, snap.entries[0].values_scales);
    defer _ = mlx.mlx_array_free(snap_vs);
    const snap_vb = try materializedOwnedCopy(s, snap.entries[0].values_biases);
    defer _ = mlx.mlx_array_free(snap_vb);
    try mlx.check(mlx.mlx_array_eval(snap_k));
    var a = try KVCache.initWithConfig(alloc, 1, cfg);
    defer a.deinit();
    var b = try KVCache.initWithConfig(alloc, 1, cfg);
    defer b.deinit();
    try a.restore(&snap);
    try b.restore(&snap);
    const stacked_k = try kvAppendPacked(2, 1, 0xABBA, s);
    defer _ = mlx.mlx_array_free(stacked_k);
    const stacked_v = try kvAppendPacked(2, 1, 0xBEEF, s);
    defer _ = mlx.mlx_array_free(stacked_v);
    var sibs = [_]*KVCache{ &a, &b };
    try KVCache.appendBatched(stacked_k, stacked_v, &sibs, 0, s, 0, cfg);
    try testing.expectEqual(@as(usize, 7), a.entries[0].offset);
    try testing.expectEqual(@as(usize, 7), b.entries[0].offset);
    try a.truncate(6, s);
    try testing.expectEqual(@as(usize, 6), a.entries[0].offset);
    try testing.expectEqual(@as(usize, 7), b.entries[0].offset);
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].keys, snap_k, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].values, snap_v, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].keys_scales, snap_ks, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].keys_biases, snap_kb, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].values_scales, snap_vs, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].values_biases, snap_vb, s));
    const k1 = try KVCache.sliceStackedRow(s, stacked_k, 0);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try KVCache.sliceStackedRow(s, stacked_v, 0);
    defer _ = mlx.mlx_array_free(v1);
    var dv = try a.update(0, k1, v1, s, 0);
    dv.deinit();
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].keys, snap_k, s));
}

test "batched KV append: per-layer op count is quantize-once plus per-slot writes, not N x (2 slices + update)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = testing.allocator;
    const s = mlx.gpuStream();
    const cfg = kv_quant.KVQuantConfig.affine(8);
    const ns = [_]usize{ 2, 4, 8 };
    const lens = [_]usize{ 2, 5, 3, 8, 4, 7, 6, 9 };
    var solo_at: [3]u64 = undefined;
    var bat_at: [3]u64 = undefined;
    for (ns, 0..) |n, ni| {
        const solo = try alloc.alloc(KVCache, n);
        const batched = try alloc.alloc(KVCache, n);
        defer {
            for (solo) |*c| c.deinit();
            alloc.free(solo);
            for (batched) |*c| c.deinit();
            alloc.free(batched);
        }
        const solo_ptrs = try alloc.alloc(*KVCache, n);
        defer alloc.free(solo_ptrs);
        const batched_ptrs = try alloc.alloc(*KVCache, n);
        defer alloc.free(batched_ptrs);
        for (0..n) |i| {
            solo[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            batched[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            try kvAppendPrefill(&solo[i], lens[i], @intCast(0xB000 + i), s);
            try kvAppendPrefill(&batched[i], lens[i], @intCast(0xB000 + i), s);
            solo_ptrs[i] = &solo[i];
            batched_ptrs[i] = &batched[i];
        }
        const stacked_k = try kvAppendPacked(@intCast(n), 1, 0x1111, s);
        defer _ = mlx.mlx_array_free(stacked_k);
        const stacked_v = try kvAppendPacked(@intCast(n), 1, 0x2222, s);
        defer _ = mlx.mlx_array_free(stacked_v);
        const c_solo = mlx.op_count.load(.monotonic);
        try KVCache.appendFromStacked(stacked_k, stacked_v, solo_ptrs, 0, s, 0, false);
        solo_at[ni] = mlx.op_count.load(.monotonic) - c_solo;
        const c_bat = mlx.op_count.load(.monotonic);
        try KVCache.appendFromStacked(stacked_k, stacked_v, batched_ptrs, 0, s, 0, true);
        bat_at[ni] = mlx.op_count.load(.monotonic) - c_bat;
        std.debug.print("batched KV append op-count N={d}: solo {d} batched {d}\n", .{ n, solo_at[ni], bat_at[ni] });
        try testing.expectEqual(24 * @as(u64, @intCast(n)), solo_at[ni]);
        try testing.expectEqual(8 + 18 * @as(u64, @intCast(n)), bat_at[ni]);
    }
    try testing.expect((bat_at[2] - bat_at[0]) < (solo_at[2] - solo_at[0]));
}

test "batched KV append: dense and affine-4 stay on the per-slot path" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const alloc = testing.allocator;
    const s = mlx.gpuStream();
    const schemes = [_]kv_quant.KVQuantConfig{
        kv_quant.KVQuantConfig.dense,
        kv_quant.KVQuantConfig.affine(4),
    };
    for (schemes) |cfg| {
        var solo: [2]KVCache = .{ try KVCache.initWithConfig(alloc, 1, cfg), try KVCache.initWithConfig(alloc, 1, cfg) };
        var batched: [2]KVCache = .{ try KVCache.initWithConfig(alloc, 1, cfg), try KVCache.initWithConfig(alloc, 1, cfg) };
        defer for (&solo) |*c| c.deinit();
        defer for (&batched) |*c| c.deinit();
        try kvAppendPrefill(&solo[0], 3, 0x10, s);
        try kvAppendPrefill(&solo[1], 5, 0x11, s);
        try kvAppendPrefill(&batched[0], 3, 0x10, s);
        try kvAppendPrefill(&batched[1], 5, 0x11, s);
        var solo_ptrs = [_]*KVCache{ &solo[0], &solo[1] };
        var batched_ptrs = [_]*KVCache{ &batched[0], &batched[1] };
        const stacked_k = try kvAppendPacked(2, 1, 0x30, s);
        defer _ = mlx.mlx_array_free(stacked_k);
        const stacked_v = try kvAppendPacked(2, 1, 0x31, s);
        defer _ = mlx.mlx_array_free(stacked_v);
        const c0 = mlx.op_count.load(.monotonic);
        try KVCache.appendFromStacked(stacked_k, stacked_v, &solo_ptrs, 0, s, 0, false);
        const solo_ops = mlx.op_count.load(.monotonic) - c0;
        const c1 = mlx.op_count.load(.monotonic);
        try KVCache.appendFromStacked(stacked_k, stacked_v, &batched_ptrs, 0, s, 0, true);
        const batched_ops = mlx.op_count.load(.monotonic) - c1;
        try testing.expectEqual(solo_ops, batched_ops);
        for (0..2) |i| {
            try testing.expectEqual(solo[i].entries[0].offset, batched[i].entries[0].offset);
            try testing.expect(try qsaArraysAllEqual(solo[i].entries[0].keys, batched[i].entries[0].keys, s));
            try testing.expect(try qsaArraysAllEqual(solo[i].entries[0].values, batched[i].entries[0].values, s));
            if (cfg.scheme == .affine) {
                try kvAppendOperandsEqual(&solo[i].entries[0], &batched[i].entries[0], s);
            }
        }
    }
}

// ── KV cache growth policy (issue #110) ──────────────────────────────────────

test "KVCache growth strands bounded bytes" {
    // Walk a decode from empty to 100_000 tokens one token at a time and count
    // what the grow-and-copy policy leaves behind. Every growth event allocates
    // a fresh capacity-sized buffer, copies the old contents in, and frees the
    // old one — and MLX only reuses a cached buffer when
    // `cached_size < requested + 2*page_size` (`buffer_cache.h`), so under the
    // old fixed +256 policy the superseded buffer is always 512 KB-plus past a
    // 32 KB tolerance and can never be reused for the NEXT KV buffer either.
    // On the reporter's 64-KiB/token model that is a whole KV cache orphaned
    // per 256 generated tokens.
    const walk = struct {
        const Result = struct { events: usize, superseded: usize, final: usize };
        fn run() Result {
            var cap: usize = 0;
            var events: usize = 0;
            var superseded: usize = 0;
            var n: usize = 1;
            while (n <= 100_000) : (n += 1) {
                const next = KVCache.nextCapacity(cap, n);
                if (next != cap) {
                    events += 1;
                    superseded += cap;
                    cap = next;
                }
            }
            return .{ .events = events, .superseded = superseded, .final = cap };
        }
    };

    const prop = walk.run();
    try testing.expect(prop.final >= 100_000);
    try testing.expect(prop.events <= 40);
    try testing.expect(prop.superseded <= 8 * prop.final);
}

test "KV growth does not ratchet the MLX buffer pool" {
    // The reporter's mechanism (issue #110) at a size that runs in seconds and
    // needs no checkpoint: single-token `update` calls against a real KVCache,
    // sampling MLX's allocator pool. Nothing here calls `mlx_clear_cache` —
    // that is the point. What the pool holds is exactly what the growth policy
    // orphaned.
    //
    // 4 layers x 2 kv heads x 64 head_dim, bf16 = 2 KiB per token across all K
    // and V buffers. Under the old +256 policy the walk to 20_000 allocates
    // ~810_000 token-slots and frees all but the last ≈ 41 MB. Measured peak
    // pool: 1544 MB linear vs 274 MB proportional, so the bar sits between them with ~2x margin
    // either side rather than on top of the passing number.
    const s = mlx.gpuStream();
    const LAYERS: u32 = 4;
    const N: usize = 20_000;
    const shape = [_]c_int{ 1, 2, 1, 64 };

    var cache = try KVCache.init(testing.allocator, LAYERS);
    defer cache.deinit();

    _ = mlx.mlx_clear_cache();

    var peak_cache: usize = 0;
    var t: usize = 0;
    while (t < N) : (t += 1) {
        var layer: u32 = 0;
        while (layer < LAYERS) : (layer += 1) {
            var k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k);
            var v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v);
            try mlx.check(mlx.mlx_zeros(&k, &shape, 4, .bfloat16, s));
            try mlx.check(mlx.mlx_zeros(&v, &shape, 4, .bfloat16, s));
            var view = try cache.update(layer, k, v, s, 0);
            view.deinit();
        }
        if (t % 32 == 0) {
            cache.evalState();
            var c: usize = 0;
            _ = mlx.mlx_get_cache_memory(&c);
            if (c > peak_cache) peak_cache = c;
        }
    }
    cache.evalState();
    var final_cache: usize = 0;
    _ = mlx.mlx_get_cache_memory(&final_cache);
    if (final_cache > peak_cache) peak_cache = final_cache;

    if (peak_cache > 512 << 20) {
        std.debug.print(
            "[kv-pool] peak MLX buffer pool {d} MB after {d} single-token updates\n",
            .{ peak_cache >> 20, N },
        );
        return error.BufferPoolRatcheted;
    }
}

test "growQuantBuf else-arm: a faulted zeros leaves the old buffer owned (no double free)" {
    // A failed mlx_zeros leaves the caller's old buffer owned. The free below
    // is then legal; freeing it first would free it twice.
    if (!mlxDeviceUsable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const heads: c_int = 1;
    const last_dim: c_int = 4;
    const old_cap: c_int = 2;
    const new_cap: c_int = 8;
    const old_shape = [_]c_int{ B, heads, old_cap, last_dim };

    var warm = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&warm, &old_shape, 4, .float32, s));
    try mlx.check(mlx.mlx_array_eval(warm));
    const c0 = mlx.op_count.load(.monotonic);
    try KVCache.growQuantBuf(s, &warm, .{ .cap = new_cap, .drop = 0, .kept = 0 }, B, heads, last_dim, .float32);
    const n_ops = mlx.op_count.load(.monotonic) - c0;
    try mlx.check(mlx.mlx_array_eval(warm));
    try testing.expectEqual(@as(c_int, new_cap), mlx.getShape(warm)[2]);
    _ = mlx.mlx_array_free(warm);
    try testing.expect(n_ops >= 1);

    var fired: usize = 0;
    var k: u64 = 1;
    while (k <= n_ops) : (k += 1) {
        var buf = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&buf, &old_shape, 4, .float32, s));
        try mlx.check(mlx.mlx_array_eval(buf));
        mlx.fault.arm(k);
        const r = KVCache.growQuantBuf(s, &buf, .{ .cap = new_cap, .drop = 0, .kept = 0 }, B, heads, last_dim, .float32);
        const did = mlx.fault.didFire();
        mlx.fault.disarm();
        if (r) |_| {
            // Not reached for k in 1..=n_ops; free the replacement in case.
            _ = mlx.mlx_array_free(buf);
        } else |err| {
            try testing.expectEqual(error.MlxError, err);
            try testing.expect(did);
            fired += 1;
            // This free must be legal; it is the assertion.
            _ = mlx.mlx_array_free(buf);
        }
    }
    try testing.expectEqual(n_ops, fired);
}

test "KVCache dense update: a faulted op leaves no freed view handle in the entry" {
    // The bar: after any failing op, `deinit` frees each handle the entry holds exactly once.
    if (!mlxDeviceUsable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const first = testKV(4, s);
    defer _ = mlx.mlx_array_free(first);

    // 4 more tokens fit the first update's buffer; 300 more take the grow path.
    var ops = [_]u64{ 0, 0 };
    for ([_]usize{ 4, 300 }, &ops) |len, *n_ops| {
        const kv = testKV(len, s);
        defer _ = mlx.mlx_array_free(kv);
        {
            var cache = try KVCache.init(testing.allocator, 1);
            defer cache.deinit();
            var dv = try cache.update(0, first, first, s, 0);
            dv.deinit();
            const c0 = mlx.op_count.load(.monotonic);
            var dv2 = try cache.update(0, kv, kv, s, 0);
            dv2.deinit();
            n_ops.* = mlx.op_count.load(.monotonic) - c0;
        }

        var k: u64 = 1;
        while (k <= n_ops.*) : (k += 1) {
            var cache = try KVCache.init(testing.allocator, 1);
            defer cache.deinit();
            var dv = try cache.update(0, first, first, s, 0);
            dv.deinit();
            mlx.fault.arm(k);
            const r = cache.update(0, kv, kv, s, 0);
            const did = mlx.fault.didFire();
            mlx.fault.disarm();
            try testing.expectError(error.MlxError, r);
            try testing.expect(did);
            // The views are built last, K then V: a fault before a view's own build leaves it empty.
            if (k + 1 < n_ops.*) try testing.expect(cache.entries[0].key_view.ctx == null);
            if (k < n_ops.*) try testing.expect(cache.entries[0].value_view.ctx == null);
        }
    }
    try testing.expect(ops[0] >= 4 and ops[1] > ops[0]);
}

test "generation past the reserved headroom takes the +25% growth policy, never a mid-prefill grow" {
    const t = std.testing;
    // The reservation is bounded at `RESERVE_GEN_HEADROOM` tokens of generation; past it the
    // ordinary proportional policy grows between decode steps.
    const reserve_headroom = KVCache.RESERVE_GEN_HEADROOM;
    const chunk: usize = 4096;
    const seq: usize = 40_000;
    const reserved: usize = @intCast(KVCache.reservedTokens(seq, 1_000_000, chunk, 0));
    try t.expectEqual(seq + reserve_headroom + chunk, reserved);

    const s_gpu = mlx.gpuStream();
    var cache = try KVCache.init(t.allocator, 1);
    defer cache.deinit();
    cache.reserve(reserved);
    var written: usize = 0;
    var allocs: usize = 0;
    var last_cap: usize = 0;
    var cap_at_prefill_end: usize = 0;
    const total: usize = reserved + reserve_headroom;
    while (written < total) {
        const n: c_int = @intCast(@min(chunk, total - written));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const count: f64 = @floatFromInt(n * 8);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, count, 1.0, .float32, s_gpu));
        var k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k);
        const shape = [_]c_int{ 1, 1, n, 8 };
        try mlx.check(mlx.mlx_reshape(&k, flat, &shape, 4, s_gpu));
        var view = try cache.update(0, k, k, s_gpu, 0);
        view.deinit();
        const cap_now: usize = KVCache.bufferCapacity(cache.entries[0].keys);
        if (cap_now != last_cap) {
            allocs += 1;
            last_cap = cap_now;
        }
        written += @intCast(n);
        if (written <= seq) cap_at_prefill_end = last_cap;
    }
    // The prefill never grew.
    try t.expect(cap_at_prefill_end >= reserved);
    try t.expectEqual(total, cache.entries[0].offset);
    try t.expect(allocs >= 2);
    try t.expectEqual(KVCache.nextCapacity(cap_at_prefill_end, cap_at_prefill_end + 1), cache.nextCapacityReserved(cap_at_prefill_end, cap_at_prefill_end + 1));
    try t.expect(last_cap >= total);
}

test "a reserved KV cache grows ONCE: no old+new buffer coexists during a long prefill" {
    const t = std.testing;
    // Walking a 458,832-token prompt at chunk 4096 under the proportional policy is a ladder
    // of grows, each holding two copies inside the chunk's lazy graph.
    const seq: usize = 458_832;
    const chunk: usize = 4096;
    var cap: usize = 0;
    var pos: usize = 0;
    var grows: usize = 0;
    while (pos < seq) {
        const end = @min(pos + chunk, seq);
        if (end > cap) {
            cap = KVCache.nextCapacity(cap, end);
            grows += 1;
        }
        pos = end;
    }
    try t.expect(grows > 20);

    // Reserved: one allocation, driven through the real write path. ctx 0 = no context named.
    const reserved: usize = @intCast(KVCache.reservedTokens(seq, 2048, chunk, 0));
    try t.expectEqual(seq + 2048 + chunk, reserved);
    var rcache = try KVCache.init(t.allocator, 1);
    defer rcache.deinit();
    rcache.reserve(reserved);
    var rcap: usize = 0;
    var rpos: usize = 0;
    var rgrows: usize = 0;
    while (rpos < seq) {
        const end = @min(rpos + chunk, seq);
        if (end > rcap) {
            rcap = rcache.nextCapacityReserved(rcap, end);
            rgrows += 1;
        }
        rpos = end;
    }
    try t.expectEqual(@as(usize, 1), rgrows);
    try t.expect(rcap >= reserved);

    const s_gpu = mlx.gpuStream();
    const walk: usize = 33_000; // past RESERVE_MIN_TOKENS
    const step: usize = 4096;
    const live_reserve: usize = @intCast(KVCache.reservedTokens(walk, 512, step, 0));
    var live = try KVCache.init(t.allocator, 1);
    defer live.deinit();
    live.reserve(live_reserve);
    var written: usize = 0;
    var caps: usize = 0;
    var last_cap: usize = 0;
    while (written < walk) {
        const n: c_int = @intCast(@min(step, walk - written));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const count: f64 = @floatFromInt(n * 8);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, count, 1.0, .float32, s_gpu));
        var k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k);
        const shape = [_]c_int{ 1, 1, n, 8 };
        try mlx.check(mlx.mlx_reshape(&k, flat, &shape, 4, s_gpu));
        var view = try live.update(0, k, k, s_gpu, 0);
        view.deinit();
        const cap_now: usize = KVCache.bufferCapacity(live.entries[0].keys);
        if (cap_now != last_cap) {
            caps += 1;
            last_cap = cap_now;
        }
        written += @intCast(n);
    }
    try t.expectEqual(@as(usize, 1), caps);
    try t.expect(last_cap >= live_reserve);
    try t.expectEqual(walk, live.entries[0].offset);
    try t.expect(rcap >= seq + 2048);

    try t.expectEqual(@as(u64, 0), KVCache.reservedTokens(4096, 2048, chunk, 0));
    var small = try KVCache.init(t.allocator, 1);
    defer small.deinit();
    small.reserve(@intCast(KVCache.reservedTokens(4096, 2048, chunk, 0)));
    try t.expectEqual(
        KVCache.nextCapacity(1024, 2048),
        small.nextCapacityReserved(1024, 2048),
    );
}

/// Bit equality of two strided views (cache slices are not row-contiguous).
fn ringViewsEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !bool {
    const ca = try materializedOwnedCopy(s, a);
    defer _ = mlx.mlx_array_free(ca);
    const cb = try materializedOwnedCopy(s, b);
    defer _ = mlx.mlx_array_free(cb);
    return bf16BitsEqual(ca, cb);
}

test "KVCache restore: a restored entry exposes its live rows to a read before the first update" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x7E57);
    var src = try KVCache.init(testing.allocator, 1);
    defer src.deinit();
    const k = try testRandWeightBf16(prng.random(), &.{ 1, 1, 25, 8 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try testRandWeightBf16(prng.random(), &.{ 1, 1, 25, 4 }, s);
    defer _ = mlx.mlx_array_free(v);
    var first = try src.update(0, k, v, s, 0);
    first.deinit();
    var snap = try src.snapshot();
    defer snap.deinit();

    // The GLM DSA layer reads the cached rows BEFORE it appends (the previous indexer rows for pooling).
    var back = try KVCache.init(testing.allocator, 1);
    defer back.deinit();
    try back.restore(&snap);
    const view = try back.denseView(0, s);
    try testing.expect(view.k.ctx != null and view.v.ctx != null);
    try testing.expectEqual(@as(c_int, 25), mlx.getShape(view.k)[2]);
    try testing.expectEqual(@as(c_int, 25), mlx.getShape(view.v)[2]);
    try testing.expect(try ringViewsEqual(s, view.k, k));
}

test "KVCache sliding ring: a sliding layer keeps its newest rows and serves the full buffer's window" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x51D1);
    var full = try KVCache.init(testing.allocator, 1);
    defer full.deinit();
    var ring = try KVCache.init(testing.allocator, 1);
    defer ring.deinit();
    const window: u32 = 8;
    const keep: u32 = 24;
    // Prefill chunks, decode steps, a spec rollback (negative), more decode.
    const plan = [_]i32{ 300, 300, 300, 300, 300, 300, 1, 1, 1, 1, 1, 1, 1, 1, -3, 1, 1, 1, -2, 1 };
    for (plan) |n| {
        if (n < 0) {
            const len = full.entries[0].offset - @as(usize, @intCast(-n));
            try full.truncate(len, s);
            try ring.truncate(len, s);
            continue;
        }
        const k = try testRandWeightBf16(prng.random(), &.{ 1, 2, n, 6 }, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try testRandWeightBf16(prng.random(), &.{ 1, 2, n, 4 }, s);
        defer _ = mlx.mlx_array_free(v);
        const span = window + @as(u32, @intCast(n)) - 1;
        const vf = try full.update(0, k, v, s, span);
        const vr = try ring.updateSliding(0, k, v, s, span, window, keep);
        try testing.expect(try ringViewsEqual(s, vf.k, vr.k));
        try testing.expect(try ringViewsEqual(s, vf.v, vr.v));
        try testing.expectEqual(full.entries[0].offset, ring.entries[0].offset);
        try testing.expect(KVCache.bufferCapacity(ring.entries[0].keys) <= 512);
    }
    try testing.expect(KVCache.bufferCapacity(full.entries[0].keys) >= 1800);

    // A snapshot restores and continues bit-for-bit.
    var snap = try ring.snapshot();
    defer snap.deinit();
    var back = try KVCache.init(testing.allocator, 1);
    defer back.deinit();
    try back.restore(&snap);
    const k = try testRandWeightBf16(prng.random(), &.{ 1, 2, 1, 6 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try testRandWeightBf16(prng.random(), &.{ 1, 2, 1, 4 }, s);
    defer _ = mlx.mlx_array_free(v);
    const vf = try full.update(0, k, v, s, window);
    const vb = try back.updateSliding(0, k, v, s, window, window, keep);
    try testing.expect(try ringViewsEqual(s, vf.k, vb.k));
    try testing.expect(try ringViewsEqual(s, vf.v, vb.v));

    // A rollback past the rows the ring still holds is refused, not served.
    try testing.expectError(error.SlidingRowsDropped, ring.truncate(100, s));

    // A ring that has dropped nothing rewinds anywhere, inside its first window too.
    var young = try KVCache.init(testing.allocator, 1);
    defer young.deinit();
    var yv = try young.updateSliding(0, k, v, s, window, window, keep);
    yv.deinit();
    for (0..4) |_| {
        var y = try young.updateSliding(0, k, v, s, window, window, keep);
        y.deinit();
    }
    try young.truncate(3, s);
    try testing.expectEqual(@as(usize, 3), young.entries[0].offset);
    var ysnap = try young.snapshot();
    defer ysnap.deinit();
    var ytrim = try ysnap.trimmedCopy(2, s);
    ytrim.deinit();
}
