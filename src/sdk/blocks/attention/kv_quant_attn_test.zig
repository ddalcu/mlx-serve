//! `kv_quant_attn.zig`'s tests, out of its import graph: a plugin unit that reaches `kv_quant_attn.zig` never compiles them.

const kv_quant_attn = @import("kv_quant_attn.zig");
const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const testing = std.testing;
const verifyQmmTile = @import("../linear/verify_qmm_nax.zig").verifyQmmTile;
const DenseKVView = @import("kv_view.zig").DenseKVView;
const splitCausalSdpa = @import("sdpa.zig").splitCausalSdpa;
const KV_ATTN_FUSED_MIN_TK = kv_quant_attn.KV_ATTN_FUSED_MIN_TK;
const QKV_VERIFY_MAX_TQ = kv_quant_attn.QKV_VERIFY_MAX_TQ;
const kvAttnFusedEligible = kv_quant_attn.kvAttnFusedEligible;
const kvAttnVerifyEligible = kv_quant_attn.kvAttnVerifyEligible;
const qkvAttnDecodeKernel = kv_quant_attn.qkvAttnDecodeKernel;
const qkvAttnMppKernel = kv_quant_attn.qkvAttnMppKernel;
const qkvAttnVerifyKernel = kv_quant_attn.qkvAttnVerifyKernel;
const qkvMppWins = kv_quant_attn.qkvMppWins;
const qkvVerifyKernelEnabledFrom = kv_quant_attn.qkvVerifyKernelEnabledFrom;
const qkvVerifyRowsPay = kv_quant_attn.qkvVerifyRowsPay;
const kv_quant = @import("kv_quant.zig");
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;

/// Shared harness for the decode quantized-KV kernel parity tests: prefill
/// `t_prefill` tokens + one decode token through a real affine KVCache (so
/// the views carry the REAL strided layout, capacity > logical length), then
/// compare the kernel against dense SDPA over the SAME dequantized views.
/// No-worse-than-reference (quantization error folds into both paths); the
/// kernel is NOT expected to bit-match (f32 in-register dequant vs the
/// bf16-rounded dense views + different reduction order — qmv class).
fn qkvDecParityCase(h_q: c_int, h_kv: c_int, d: c_int, bits: u8, t_prefill: usize, masked: bool) !void {
    const s = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(bits));
    defer cache.deinit();

    // Smooth ramp values (quantization-friendly but non-trivial).
    const mkArr = struct {
        fn f(str: mlx.mlx_stream, hh: c_int, tt: usize, dd: c_int, seed: f32) !mlx.mlx_array {
            const total: usize = @intCast(@as(usize, @intCast(hh)) * tt * @as(usize, @intCast(dd)));
            const buf = try testing.allocator.alloc(f32, total);
            defer testing.allocator.free(buf);
            for (buf, 0..) |*x, i| {
                const fi: f32 = @floatFromInt(i);
                x.* = @sin(fi * 0.37 + seed) * 0.8;
            }
            const shape = [_]c_int{ 1, hh, @intCast(tt), dd };
            const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var bf = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, str));
            return bf;
        }
    }.f;

    // Prefill then one decode token — the grow path leaves capacity slack,
    // so the update's views are STRIDED along the head axis (the layout the
    // kernel must index correctly; a contiguous-only test would pass green
    // on broken stride math).
    {
        const k = try mkArr(s, h_kv, t_prefill, d, 0.1);
        defer _ = mlx.mlx_array_free(k);
        const v = try mkArr(s, h_kv, t_prefill, d, 1.7);
        defer _ = mlx.mlx_array_free(v);
        var dv0 = try cache.update(0, k, v, s, 0);
        dv0.deinit();
    }
    const k1 = try mkArr(s, h_kv, 1, d, 2.9);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try mkArr(s, h_kv, 1, d, 3.3);
    defer _ = mlx.mlx_array_free(v1);
    var dv = try cache.update(0, k1, v1, s, 0);
    defer dv.deinit();
    try testing.expect(dv.has_quant_triple);

    const t_k: c_int = @intCast(t_prefill + 1);
    const q = try mkArr(s, h_q, 1, d, 5.1);
    defer _ = mlx.mlx_array_free(q);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    // Optional additive mask row: block the first 2 kv positions.
    var mask = mlx.mlx_array{ .ctx = null };
    defer if (mask.ctx != null) {
        _ = mlx.mlx_array_free(mask);
    };
    if (masked) {
        const tk_us: usize = @intCast(t_k);
        const mbuf = try testing.allocator.alloc(f32, tk_us);
        defer testing.allocator.free(mbuf);
        for (mbuf, 0..) |*x, i| x.* = if (i < 2) -std.math.inf(f32) else 0.0;
        const mshape = [_]c_int{ 1, 1, 1, t_k };
        const mf32 = mlx.mlx_array_new_data(mbuf.ptr, &mshape, 4, .float32);
        defer _ = mlx.mlx_array_free(mf32);
        var mbf = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&mbf, mf32, .bfloat16, s));
        mask = mbf;
    }

    // Reference: dense SDPA over the view's own dequantized K/V.
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref,
        q,
        dv.k,
        dv.v,
        scale,
        if (masked) "array" else "",
        if (masked) mask else none,
        .{ .ctx = null },
        false,
        s,
    ));

    const cand = (try qkvAttnDecodeKernel(
        s,
        q,
        &dv,
        scale,
        if (masked) "array" else "",
        if (masked) mask else none,
    )) orelse return error.KernelDeclined; // engagement IS the assertion

    defer _ = mlx.mlx_array_free(cand);
    const cand_shape = mlx.getShape(cand);
    try testing.expectEqual(@as(c_int, 1), cand_shape[0]);
    try testing.expectEqual(h_q, cand_shape[1]);
    try testing.expectEqual(@as(c_int, 1), cand_shape[2]);
    try testing.expectEqual(d, cand_shape[3]);

    // Compare in f32.
    const readFlat = struct {
        fn f(str: mlx.mlx_stream, arr: mlx.mlx_array, allocator: std.mem.Allocator) ![]f32 {
            var f32v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32v);
            try mlx.check(mlx.mlx_astype(&f32v, arr, .float32, str));
            const n: c_int = @intCast(mlx.mlx_array_size(f32v));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const sh = [_]c_int{n};
            try mlx.check(mlx.mlx_reshape(&flat, f32v, &sh, 1, str));
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            _ = mlx.mlx_vector_array_append_value(ev, flat);
            try mlx.check(mlx.mlx_eval(ev));
            const ptr = mlx.mlx_array_data_float32(flat) orelse return error.NullData;
            const out = try allocator.alloc(f32, @intCast(n));
            @memcpy(out, ptr[0..@intCast(n)]);
            return out;
        }
    }.f;
    const ref_flat = try readFlat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readFlat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        // NaN-blind comparisons pass on an all-NaN candidate (NaN > x is
        // false) — the composed causal arm shipped NaN for months behind
        // exactly this hole. Finiteness is asserted BEFORE the diff.
        try testing.expect(!std.math.isNan(c));
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    if (max_err >= 0.05) {
        std.debug.print("qkvDecParityCase Hq={d} Hkv={d} D={d} bits={d} Tk={d} masked={}: max_err={d}\n", .{ h_q, h_kv, d, bits, t_k, masked, max_err });
    }
    try testing.expect(max_err < 0.05);
}

test "qkv decode kernel parity: GQA ratios x bits x head dims (strided views)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    // T_k = 37+1 — not a multiple of NSG(4), exercises uneven simdgroup trip
    // counts; the second update leaves capacity slack so views are strided.
    try qkvDecParityCase(8, 2, 64, 8, 37, false);
    try qkvDecParityCase(8, 2, 64, 4, 37, false);
    try qkvDecParityCase(12, 2, 128, 8, 37, false);
    try qkvDecParityCase(12, 2, 128, 4, 37, false);
    try qkvDecParityCase(4, 4, 128, 8, 21, false); // MHA (GQA=1)
    try qkvDecParityCase(6, 1, 256, 8, 21, false); // hd-256 (gemma geometry)
}

test "qkv decode kernel parity: additive mask row (sliding decode shape)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    try qkvDecParityCase(8, 2, 128, 8, 37, true);
    try qkvDecParityCase(8, 2, 128, 4, 37, true);
}

test "qkv decode kernel parity: multi-block split-KV merge (T_k > BLOCK)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    // g=4/DK=64 → BLOCK 512, so 700+1 rows exercises the two-block partial
    // merge (m/l/O carry through the mlx-op reduction); the masked case
    // additionally crosses a block boundary with -inf rows.
    try qkvDecParityCase(8, 2, 64, 8, 700, false);
    try qkvDecParityCase(8, 2, 128, 8, 700, true);
}

fn qkvVerParityCase(h_q: c_int, h_kv: c_int, d: c_int, bits: u8, t_prefill: usize, t_q: c_int) !void {
    return qkvVerParityCaseWith(h_q, h_kv, d, bits, t_prefill, t_q, false);
}

fn qkvVerParityCaseWith(h_q: c_int, h_kv: c_int, d: c_int, bits: u8, t_prefill: usize, t_q: c_int, mpp: bool) !void {
    // Parity validates the explicit QA arm, independent of the measured
    // per-machine default (G17 defaults this kernel off).
    const old_qkv_ver_override = kv_quant_attn.qkv_ver_override;
    kv_quant_attn.qkv_ver_override = true;
    defer kv_quant_attn.qkv_ver_override = old_qkv_ver_override;
    const s = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(bits));
    defer cache.deinit();
    const mkArr = struct {
        fn f(str: mlx.mlx_stream, hh: c_int, tt: c_int, dd: c_int, seed: f32) !mlx.mlx_array {
            const total: usize = @intCast(hh * tt * dd);
            const buf = try testing.allocator.alloc(f32, total);
            defer testing.allocator.free(buf);
            for (buf, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.37 + seed) * 0.7;
            const shape = [_]c_int{ 1, hh, tt, dd };
            const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var bf = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, str));
            return bf;
        }
    }.f;
    if (t_prefill > 0) {
        const k0 = try mkArr(s, h_kv, @intCast(t_prefill), d, 0.4);
        defer _ = mlx.mlx_array_free(k0);
        const v0 = try mkArr(s, h_kv, @intCast(t_prefill), d, 1.3);
        defer _ = mlx.mlx_array_free(v0);
        var dv0 = try cache.update(0, k0, v0, s, 0);
        dv0.deinit();
    }
    // The verify chunk: t_q new K/V rows appended (the spec contract — the
    // draft rows are in the cache when attention runs).
    const k1 = try mkArr(s, h_kv, t_q, d, 2.9);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try mkArr(s, h_kv, t_q, d, 3.3);
    defer _ = mlx.mlx_array_free(v1);
    var dv = try cache.update(0, k1, v1, s, 0);
    defer dv.deinit();
    try testing.expect(dv.has_quant_triple);

    const q = try mkArr(s, h_q, t_q, d, 5.1);
    defer _ = mlx.mlx_array_free(q);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    // Reference: dense SDPA over the view's own dequantized K/V, "causal"
    // (end-aligned) — the exact mode the dense path serves these widths.
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, dv.k, dv.v, scale, "causal", none, .{ .ctx = null }, false, s));

    const cand = (if (mpp) try qkvAttnMppKernel(s, q, &dv, scale, "causal") else try qkvAttnVerifyKernel(s, q, &dv, scale, "causal")) orelse return error.KernelDeclined;
    defer _ = mlx.mlx_array_free(cand);
    const cand_shape = mlx.getShape(cand);
    try testing.expectEqual(@as(c_int, 1), cand_shape[0]);
    try testing.expectEqual(h_q, cand_shape[1]);
    try testing.expectEqual(t_q, cand_shape[2]);
    try testing.expectEqual(d, cand_shape[3]);

    const readFlat = struct {
        fn f(str: mlx.mlx_stream, arr: mlx.mlx_array, allocator: std.mem.Allocator) ![]f32 {
            var f32v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32v);
            try mlx.check(mlx.mlx_astype(&f32v, arr, .float32, str));
            const n: c_int = @intCast(mlx.mlx_array_size(f32v));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const sh = [_]c_int{n};
            try mlx.check(mlx.mlx_reshape(&flat, f32v, &sh, 1, str));
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            _ = mlx.mlx_vector_array_append_value(ev, flat);
            try mlx.check(mlx.mlx_eval(ev));
            const ptr = mlx.mlx_array_data_float32(flat) orelse return error.NullData;
            const out = try allocator.alloc(f32, @intCast(n));
            @memcpy(out, ptr[0..@intCast(n)]);
            return out;
        }
    }.f;
    const ref_flat = try readFlat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readFlat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        // Finiteness BEFORE the diff (NaN > x is false — the composed causal
        // arm shipped NaN for months behind exactly this hole).
        try testing.expect(!std.math.isNan(c));
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    if (max_err >= 0.05) {
        std.debug.print("qkvVerParityCase Hq={d} Hkv={d} D={d} bits={d} pre={d} Tq={d}: max_err={d}\n", .{ h_q, h_kv, d, bits, t_prefill, t_q, max_err });
    }
    try testing.expect(max_err < 0.05);
}

test "qkv verify kernel parity: TQ 2..8 x bits x GQA (strided views, causal tail)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    // Mutation-checked: dropping the in-kernel causal tail (gpos < lim) or
    // mis-ordering the (head, tq) row layout fails the Tq>=2 cases — the
    // reference is end-aligned SDPA over the SAME dequantized view, so the
    // tail is load-bearing, not decorative. 37+Tq rows keeps views strided
    // (capacity slack from the two-update sequence).
    try qkvVerParityCase(8, 2, 64, 8, 37, 2);
    try qkvVerParityCase(8, 2, 64, 4, 37, 3);
    try qkvVerParityCase(12, 2, 128, 8, 37, 4);
    try qkvVerParityCase(12, 2, 128, 4, 37, 5);
    try qkvVerParityCase(4, 4, 128, 8, 21, 6); // MHA (GQA=1)
    try qkvVerParityCase(6, 1, 256, 8, 21, 8); // hd-256, deepest width
    try qkvVerParityCase(24, 4, 256, 8, 21, 4); // the qwen3_5 live geometry
}

test "qkv matmul2d kernel parity: t_q 1..8 x bits, partial last page, many splits, pad rows" {
    if (verifyQmmTile() == .off) return error.SkipZigTest; // no matmul2d tile (CI's paravirtual GPU)
    for ([_]u8{ 8, 4 }) |bits| {
        for ([_]c_int{ 1, 2, 4, 5, 8 }) |tq| {
            try qkvVerParityCaseWith(24, 4, 256, bits, 700, tq, true);
            try qkvVerParityCaseWith(16, 8, 256, bits, 3000, tq, true);
        }
    }
    try qkvVerParityCaseWith(24, 4, 256, 8, 0, 4, true); // cache holds only the verify rows
}

test "qkv matmul2d kernel declines rows past its 32 KiB threadgroup budget (GQA 8 x t_q 8)" {
    if (verifyQmmTile() == .off) return error.SkipZigTest;
    try testing.expectError(error.KernelDeclined, qkvVerParityCaseWith(16, 2, 256, 4, 3000, 8, true));
    try qkvVerParityCaseWith(16, 2, 256, 4, 3000, 7, true);
}

test "qkvMppWins: the matmul2d window needs rows or length" {
    try testing.expect(qkvMppWins(4, 2048));
    try testing.expect(!qkvMppWins(2, 4096));
    try testing.expect(qkvMppWins(2, 8192));
    try testing.expect(!qkvMppWins(1, 8192));
    try testing.expect(qkvMppWins(1, 16384));
}

test "qkv verify kernel serves by default only the row counts that stay in registers" {
    try testing.expect(qkvVerifyRowsPay(6, 2));
    try testing.expect(!qkvVerifyRowsPay(6, 3));
    try testing.expect(qkvVerifyRowsPay(2, 6));
    try testing.expect(!qkvVerifyRowsPay(2, 8));
}

test "qkv verify kernel is off on G17" {
    try testing.expect(qkvVerifyKernelEnabledFrom(false));
    try testing.expect(!qkvVerifyKernelEnabledFrom(true));
}

test "qkv verify kernel parity: multi-block split-KV merge (T_k > BLOCK)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    // GQA 4 x TQ 2 → BLOCK 256; 700+2 rows exercises the multi-block carry,
    // including a final block whose tail rows are causally -inf for tq 0.
    try qkvVerParityCase(8, 2, 64, 8, 700, 2);
    try qkvVerParityCase(8, 2, 128, 8, 700, 6);
}

test "qkv verify kernel declines non-verify widths and foreign masks" {
    // Exercise the kernel's own decline predicates even where the machine
    // adoption default declines the entire arm first.
    const old_qkv_ver_override = kv_quant_attn.qkv_ver_override;
    kv_quant_attn.qkv_ver_override = true;
    defer kv_quant_attn.qkv_ver_override = old_qkv_ver_override;
    const s = mlx.gpuStream();
    var view: DenseKVView = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .has_quant_triple = false };
    const shape = [_]c_int{ 1, 4, 2, 64 };
    var q2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q2);
    try mlx.check(mlx.mlx_ones(&q2, &shape, 4, .bfloat16, s));
    // no triple → decline regardless of width
    try testing.expectEqual(@as(?mlx.mlx_array, null), try qkvAttnVerifyKernel(s, q2, &view, 1.0, "causal"));
    // real triple, wrong widths / modes
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();
    const kshape = [_]c_int{ 1, 2, 40, 64 };
    var kv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv);
    try mlx.check(mlx.mlx_ones(&kv, &kshape, 4, .bfloat16, s));
    var dv = try cache.update(0, kv, kv, s, 0);
    defer dv.deinit();
    var q1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q1);
    const q1shape = [_]c_int{ 1, 4, 1, 64 };
    try mlx.check(mlx.mlx_ones(&q1, &q1shape, 4, .bfloat16, s));
    try testing.expectEqual(@as(?mlx.mlx_array, null), try qkvAttnVerifyKernel(s, q1, &dv, 1.0, "causal")); // T_q 1 belongs to the decode kernel
    var q9 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q9);
    const q9shape = [_]c_int{ 1, 4, QKV_VERIFY_MAX_TQ + 1, 64 };
    try mlx.check(mlx.mlx_ones(&q9, &q9shape, 4, .bfloat16, s));
    try testing.expectEqual(@as(?mlx.mlx_array, null), try qkvAttnVerifyKernel(s, q9, &dv, 1.0, "causal")); // past the width cap
    var q4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q4);
    const q4shape = [_]c_int{ 1, 4, 4, 64 };
    try mlx.check(mlx.mlx_ones(&q4, &q4shape, 4, .bfloat16, s));
    try testing.expectEqual(@as(?mlx.mlx_array, null), try qkvAttnVerifyKernel(s, q4, &dv, 1.0, "array")); // sliding masks read dense
    try testing.expectEqual(@as(?mlx.mlx_array, null), try qkvAttnVerifyKernel(s, q4, &dv, 1.0, "")); // unmasked multi-row is not this kernel's contract
}

fn composedStridedCase(h_q: c_int, h_kv: c_int, d: c_int, bits: u8, t_pre: usize, t_chunk: c_int, mode: [:0]const u8) !void {
    const s = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(bits));
    defer cache.deinit();
    const mk = struct {
        fn f(str: mlx.mlx_stream, hh: c_int, tt: c_int, dd: c_int, seed: f32) !mlx.mlx_array {
            const total: usize = @intCast(hh * tt * dd);
            const buf = try testing.allocator.alloc(f32, total);
            defer testing.allocator.free(buf);
            for (buf, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.31 + seed) * 0.7;
            const shape = [_]c_int{ 1, hh, tt, dd };
            const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var bf = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, str));
            return bf;
        }
    }.f;
    if (t_pre > 0) {
        const k = try mk(s, h_kv, @intCast(t_pre), d, 0.2);
        defer _ = mlx.mlx_array_free(k);
        const v = try mk(s, h_kv, @intCast(t_pre), d, 1.1);
        defer _ = mlx.mlx_array_free(v);
        var dv0 = try cache.update(0, k, v, s, 0);
        dv0.deinit();
    }
    const k1 = try mk(s, h_kv, t_chunk, d, 2.3);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try mk(s, h_kv, t_chunk, d, 3.7);
    defer _ = mlx.mlx_array_free(v1);
    var dv = try cache.update(0, k1, v1, s, 0);
    defer dv.deinit();
    try testing.expect(dv.has_quant_triple);

    const t_k: c_int = @as(c_int, @intCast(t_pre)) + t_chunk;
    const q = try mk(s, h_q, t_chunk, d, 5.5);
    defer _ = mlx.mlx_array_free(q);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    // Optional additive mask (sliding-window prefill shape).
    var mask = mlx.mlx_array{ .ctx = null };
    defer if (mask.ctx != null) {
        _ = mlx.mlx_array_free(mask);
    };
    if (std.mem.eql(u8, mode, "array")) {
        const rows: usize = @intCast(t_chunk);
        const cols: usize = @intCast(t_k);
        const mbuf = try testing.allocator.alloc(f32, rows * cols);
        defer testing.allocator.free(mbuf);
        const off: usize = @intCast(t_k - t_chunk);
        for (0..rows) |r| {
            for (0..cols) |c| {
                // causal + band(3): visible iff c <= off+r and (off+r)-c < 3
                const vis = c <= off + r and (off + r) - c < 3;
                mbuf[r * cols + c] = if (vis) 0.0 else -std.math.inf(f32);
            }
        }
        const mshape = [_]c_int{ 1, 1, t_chunk, t_k };
        const mf32 = mlx.mlx_array_new_data(mbuf.ptr, &mshape, 4, .float32);
        defer _ = mlx.mlx_array_free(mf32);
        var mbf = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&mbf, mf32, .bfloat16, s));
        mask = mbf;
    }
    const none = mlx.mlx_array{ .ctx = null };
    const mask_used = if (mask.ctx != null) mask else none;

    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, dv.k, dv.v, scale, mode, mask_used, .{ .ctx = null }, false, s));

    const cand = try kv_quant.quantAttention(q, dv.kTriple(), dv.vTriple(), dv.bits, dv.group_size, scale, mode, mask_used, s);
    defer _ = mlx.mlx_array_free(cand);

    const readFlat = struct {
        fn f(str: mlx.mlx_stream, arr: mlx.mlx_array, allocator: std.mem.Allocator) ![]f32 {
            var f32v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32v);
            try mlx.check(mlx.mlx_astype(&f32v, arr, .float32, str));
            const n: c_int = @intCast(mlx.mlx_array_size(f32v));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const sh = [_]c_int{n};
            try mlx.check(mlx.mlx_reshape(&flat, f32v, &sh, 1, str));
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            _ = mlx.mlx_vector_array_append_value(ev, flat);
            try mlx.check(mlx.mlx_eval(ev));
            const ptr = mlx.mlx_array_data_float32(flat) orelse return error.NullData;
            const out = try allocator.alloc(f32, @intCast(n));
            @memcpy(out, ptr[0..@intCast(n)]);
            return out;
        }
    }.f;
    const ref_flat = try readFlat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readFlat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        // NaN-blind comparisons pass on an all-NaN candidate (NaN > x is
        // false) — the composed causal arm shipped NaN for months behind
        // exactly this hole. Finiteness is asserted BEFORE the diff.
        try testing.expect(!std.math.isNan(c));
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    if (max_err >= 0.05) std.debug.print("composedStridedCase Hq={d} Hkv={d} D={d} bits={d} pre={d} chunk={d} mode={s}: max_err={d}\n", .{ h_q, h_kv, d, bits, t_pre, t_chunk, mode, max_err });
    try testing.expect(max_err < 0.05);
}

test "composed quantAttention is correct on STRIDED cache views" {
    // MLX CUDA has no quantized_matmul for these multi-row shapes; its error would stay latched for later tests.
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    // Class guard (splitPackedGateUp rule): slice-born views into
    // mlx_quantized_matmul must compute correctly — the cache's packed views
    // are non-contiguous whenever buffer capacity > logical length. Covers
    // multi-row widths, the gemma E4B live geometry (single 15-token chunk,
    // hd 256, 4-bit — the "<pad><pad>" shape), and the sliding "array" mask.
    // (Serving only routes t_q == 1 here , but the helper's
    // contract is width-agnostic and the strided-view class is the point.)
    try composedStridedCase(8, 2, 64, 4, 20, 4, "causal");
    try composedStridedCase(8, 2, 256, 4, 0, 15, "causal");
    try composedStridedCase(8, 2, 256, 4, 0, 15, "array");
    try composedStridedCase(8, 2, 256, 4, 20, 15, "array");
}

test "qkv decode kernel declines non-decode widths and missing triples" {
    const s = mlx.gpuStream();
    // T_q = 2 → decline (only decode width may reach the kernel).
    var view: DenseKVView = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .has_quant_triple = false };
    const shape = [_]c_int{ 1, 4, 2, 64 };
    var q2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q2);
    try mlx.check(mlx.mlx_ones(&q2, &shape, 4, .bfloat16, s));
    // No triple → decline regardless of shape.
    try testing.expectEqual(@as(?mlx.mlx_array, null), try qkvAttnDecodeKernel(s, q2, &view, 1.0, "", .{ .ctx = null }));
}

test "qkv packed-attention surface µbench (diag kvq-ubench)" {
    // Per (t_q, kv): (a) packed kernel, (b) dequant + dense sdpa, (c) bf16
    // sdpa with no dequant. Arms interleave in one process and the cell order
    // reverses per pass; BATCH launches share one eval so the sync barrier
    // does not price the cell.
    if (!env.on(.KVQ_UBENCH)) return;
    const old_ver = kv_quant_attn.qkv_ver_override;
    kv_quant_attn.qkv_ver_override = true;
    defer kv_quant_attn.qkv_ver_override = old_ver;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    const H_Q: c_int = 24;
    const H_KV: c_int = 4;
    const D: c_int = 256;
    const BITS: u8 = 8;
    const kvs = [_]usize{ 2048, 4096, 8192, 16384, 32768, 65536 };
    const tqs = [_]c_int{ 1, 2, 3, 4, 6, 8 };
    const PASSES = 4;
    const BATCH = 8;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    const none = mlx.mlx_array{ .ctx = null };

    const randBf16 = struct {
        fn f(str: mlx.mlx_stream, shape: []const c_int, sd: f32) !mlx.mlx_array {
            var tmp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tmp);
            try mlx.check(mlx.mlx_random_normal(&tmp, shape.ptr, shape.len, .float32, 0.0, sd, .{ .ctx = null }, str));
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, tmp, .bfloat16, str));
            return out;
        }
    }.f;

    std.debug.print("\n[kvq-surface] Hq={d} Hkv={d} D={d} bits={d}\n[kvq-surface] {s:>6} {s:>3} {s:>9} {s:>9} {s:>9} {s:>9}\n", .{ H_Q, H_KV, D, BITS, "kv", "tq", "packed", "dq+sdpa", "bf16", "mpp" });
    for (kvs) |kv| {
        var ns: [tqs.len][4]u64 = @splat(@splat(0));
        for (tqs, 0..) |tq, ti| {
            // Own cache per width: the verify contract has the tq rows appended.
            var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(BITS));
            defer cache.deinit();
            {
                const k = try randBf16(s, &[_]c_int{ 1, H_KV, @intCast(kv - @as(usize, @intCast(tq))), D }, 0.2);
                defer _ = mlx.mlx_array_free(k);
                var dv0 = try cache.update(0, k, k, s, 0);
                dv0.deinit();
            }
            const k1 = try randBf16(s, &[_]c_int{ 1, H_KV, tq, D }, 0.2);
            defer _ = mlx.mlx_array_free(k1);
            var dv = try cache.update(0, k1, k1, s, 0);
            defer dv.deinit();
            var k_off = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_off);
            try mlx.check(mlx.mlx_contiguous(&k_off, dv.k, false, s));
            const q = try randBf16(s, &[_]c_int{ 1, H_Q, tq, D }, 0.5);
            defer _ = mlx.mlx_array_free(q);
            {
                const ev = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(ev);
                _ = mlx.mlx_vector_array_append_value(ev, k_off);
                _ = mlx.mlx_vector_array_append_value(ev, q);
                try mlx.check(mlx.mlx_eval(ev));
            }
            const mode: [:0]const u8 = if (tq == 1) "" else "causal";

            for (0..PASSES + 1) |pass| { // pass 0 = warmup
                for (0..4) |a0| {
                    const arm = if (pass % 2 == 0) a0 else 3 - a0;
                    const ev = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(ev);
                    const mark = std.Io.Timestamp.now(io, .boot);
                    for (0..BATCH) |_| {
                        var out = mlx.mlx_array_new();
                        defer _ = mlx.mlx_array_free(out);
                        switch (arm) {
                            0 => out = (if (tq == 1)
                                try qkvAttnDecodeKernel(s, q, &dv, scale, "", none)
                            else
                                try qkvAttnVerifyKernel(s, q, &dv, scale, "causal")) orelse return error.KernelDeclined,
                            3 => out = (try qkvAttnMppKernel(s, q, &dv, scale, mode)) orelse return error.KernelDeclined,
                            else => {
                                var kd = k_off;
                                if (arm == 1) kd = try kv_quant.dequantizeAffine(s, dv.k_triple_q, dv.k_triple_scales, dv.k_triple_biases, dv.group_size, dv.bits);
                                defer if (arm == 1) {
                                    _ = mlx.mlx_array_free(kd);
                                };
                                var vd = k_off;
                                if (arm == 1) vd = try kv_quant.dequantizeAffine(s, dv.v_triple_q, dv.v_triple_scales, dv.v_triple_biases, dv.group_size, dv.bits);
                                defer if (arm == 1) {
                                    _ = mlx.mlx_array_free(vd);
                                };
                                if (tq > 1) {
                                    if (try splitCausalSdpa(s, q, kd, vd, scale)) |sp| out = sp;
                                }
                                if (out.ctx == null) try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, kd, vd, scale, mode.ptr, none, .{ .ctx = null }, false, s));
                            },
                        }
                        _ = mlx.mlx_vector_array_append_value(ev, out);
                    }
                    try mlx.check(mlx.mlx_eval(ev));
                    if (pass > 0) ns[ti][arm] += @intCast(mark.untilNow(io, .boot).nanoseconds);
                }
            }
            _ = mlx.mlx_clear_cache();
        }
        for (tqs, 0..) |tq, ti| {
            var ms: [4]f64 = undefined;
            for (&ms, 0..) |*m, a| m.* = @as(f64, @floatFromInt(ns[ti][a])) / @as(f64, PASSES * BATCH) / 1e6;
            std.debug.print("[kvq-surface] {d:>6} {d:>3} {d:>9.3} {d:>9.3} {d:>9.3} {d:>9.3}\n", .{ kv, tq, ms[0], ms[1], ms[2], ms[3] });
        }
    }
}

test "kvAttnFusedEligible: fused reads are decode-width (t_q == 1) only" {
    // This is the explicit verify-kernel eligibility arm. G17's production
    // default is dense, so force the QA seam before testing width/floor rules.
    const old_qkv_ver_override = kv_quant_attn.qkv_ver_override;
    kv_quant_attn.qkv_ver_override = true;
    defer kv_quant_attn.qkv_ver_override = old_qkv_ver_override;
    // The ONE gate every fused call site reads. Verify widths (T_q 2..32
    // under MTP/spec) used to be admitted and fell to the composed qmm
    // chain — a deliberate choice that predates deep-MTP defaults and was
    // never A/B'd at spec depth. Measured (M4 Max, 8k+ ctx,
    // deep MTP rounds): 28 tok/s decode vs 60 on the dense read — the
    // composed chain runs 2 packed matmuls x every attention layer per spec
    // round, at qmm's M 2..7 dead zone. Decode width keeps the packed-read
    // kernel win; every other width reads the dense view (the reference
    // semantics). Short-KV layers (sliding windows) stay dense too: the
    // kernel's split-KV merge overhead measured a net loss there.
    const s = mlx.gpuStream();
    const long_shape = [_]c_int{ 1, 2, KV_ATTN_FUSED_MIN_TK, 16 };
    var kq_long = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kq_long);
    try mlx.check(mlx.mlx_zeros(&kq_long, &long_shape, 4, .uint32, s));
    const short_shape = [_]c_int{ 1, 2, 512, 16 };
    var kq_short = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kq_short);
    try mlx.check(mlx.mlx_zeros(&kq_short, &short_shape, 4, .uint32, s));

    var view_with: DenseKVView = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .has_quant_triple = true, .k_triple_q = kq_long };
    var view_short: DenseKVView = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .has_quant_triple = true, .k_triple_q = kq_short };
    var view_without: DenseKVView = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .has_quant_triple = false };
    try testing.expect(kvAttnFusedEligible(&view_with, 1));
    try testing.expect(!kvAttnFusedEligible(&view_with, 2)); // verify width
    try testing.expect(!kvAttnFusedEligible(&view_with, 9));
    try testing.expect(!kvAttnFusedEligible(&view_with, 32));
    try testing.expect(!kvAttnFusedEligible(&view_with, 2048)); // prefill width
    try testing.expect(!kvAttnFusedEligible(&view_short, 1)); // below the kv floor
    try testing.expect(!kvAttnFusedEligible(&view_without, 1));
    // Phase 2: verify widths (2..QKV_VERIFY_MAX_TQ) have their OWN gate,
    // serving the TQ-wide packed kernel — never the composed chain.
    try testing.expect(!kvAttnVerifyEligible(&view_with, 1)); // decode belongs to the decode kernel
    try testing.expect(kvAttnVerifyEligible(&view_with, 2));
    try testing.expect(kvAttnVerifyEligible(&view_with, QKV_VERIFY_MAX_TQ));
    try testing.expect(!kvAttnVerifyEligible(&view_with, QKV_VERIFY_MAX_TQ + 1)); // prefill chunks stay dense
    try testing.expect(!kvAttnVerifyEligible(&view_short, 4)); // below the kv floor
    try testing.expect(!kvAttnVerifyEligible(&view_without, 4));
}
