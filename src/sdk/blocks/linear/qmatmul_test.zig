//! `qmatmul.zig`'s tests, out of its import graph: a plugin unit that reaches `qmatmul.zig` never compiles them.

const qmatmul = @import("qmatmul.zig");
const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const qmv2 = @import("../qmm/qmv2.zig");
const qmv_cuda = @import("../qmm/qmv_cuda.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const prefill_dq_gemm = @import("prefill_dq_gemm.zig");
const qmv_rows = @import("qmv_rows.zig");
const testing = std.testing;
const transposeBf16Weight = @import("../../../mlx/array.zig").transposeBf16Weight;
const evalToF32 = @import("../../testing/parity.zig").evalToF32;
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const bf16Trunc = @import("../gdn/kernels.zig").bf16Trunc;
const PREFILL_DQ_GEMM_MIN_M = @import("prefill_dq_gemm.zig").PREFILL_DQ_GEMM_MIN_M;
const verifyQmm = @import("verify_qmm.zig").verifyQmm;
const device_vars = @import("../../../mlx/device.zig");
const qmatmulBits = qmatmul.qmatmulBits;
const qmvBatchLimit = qmatmul.qmvBatchLimit;
const simdQmmDecode = qmatmul.simdQmmDecode;
const simdQmmDecodeMinRows = qmatmul.simdQmmDecodeMinRows;
const ModelConfig = @import("../../config/model.zig").ModelConfig;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const qsaArraysAllEqual = @import("../../testing/parity.zig").qsaArraysAllEqual;

test "qmvBatchLimit mirrors MLX's qmv/qmm split on single- and dual-die gen-17 GPUs" {
    try testing.expectEqual(@as(c_int, 25), qmvBatchLimit(false, 2560, 640));
    try testing.expectEqual(@as(c_int, 18), qmvBatchLimit(true, 2560, 640));
    try testing.expectEqual(@as(c_int, 33), qmvBatchLimit(false, 2048, 512));
    try testing.expectEqual(@as(c_int, 32), qmvBatchLimit(true, 2048, 512));
    try testing.expectEqual(@as(c_int, 13), qmvBatchLimit(false, 2560, 10240));
    try testing.expectEqual(@as(c_int, 12), qmvBatchLimit(true, 6144, 2560));
}

test "qmatmul: a ternary 2-bit pack routes through qmv2; without the flag it stays stock" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const n: c_int = 2048;
    const k: c_int = 512;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const codes = try std.testing.allocator.alloc(u32, @divExact(2048 * 512, 16));
    defer std.testing.allocator.free(codes);
    for (codes) |*wd| wd.* = rnd.int(u32);
    const scv = try std.testing.allocator.alloc(f32, @divExact(2048 * 512, 128));
    defer std.testing.allocator.free(scv);
    for (scv) |*e| e.* = 0.01 + 0.01 * rnd.float(f32);
    var xv: [2 * 512]f32 = undefined;
    for (&xv) |*e| e.* = rnd.floatNorm(f32);
    const w = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ n, @divExact(k, 16) }, 2, .uint32);
    defer _ = mlx.mlx_array_free(w);
    const sc32 = mlx.mlx_array_new_data(scv.ptr, &[_]c_int{ n, @divExact(k, 128) }, 2, .float32);
    defer _ = mlx.mlx_array_free(sc32);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_astype(&sc, sc32, .bfloat16, s));
    // bias = +scale: the ternary kernel never reads biases, so it disagrees with stock by construction.
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_copy(&bi, sc, s));
    const x32 = mlx.mlx_array_new_data(&xv, &[_]c_int{ 1, 2, k }, 3, .float32);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

    device_vars.device_gen_override = .{ .gen = 13, .phone = false };
    defer device_vars.device_gen_override = null;
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.rht = null;
    xfm.bits_cache = .{};
    xfm.config = std.mem.zeroes(ModelConfig);
    xfm.config.quant_mode = .affine;
    xfm.config.quant_bits = 2;
    xfm.config.quant_group_size = 128;

    const kernel = (try qmv2.qmm(x, w, sc, bi, 2, 128, true, false, s)).?;
    defer _ = mlx.mlx_array_free(kernel);
    const stock = try qmatmulBits(x, w, sc, bi, 2, 128, .affine, s);
    defer _ = mlx.mlx_array_free(stock);
    {
        // The two paths must be tellable apart, or the routing check below is vacuous.
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, kernel, stock, false, s));
        var same = true;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try std.testing.expect(!same);
    }
    for ([_]bool{ true, false }) |flag| {
        xfm.ternary_2bit = flag;
        const got = try xfm.qmatmul(x, w, sc, bi);
        defer _ = mlx.mlx_array_free(got);
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, got, if (flag) kernel else stock, false, s));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try std.testing.expect(same);
    }
}

test "qmatmulBits dispatches to plain matmul when scales has null ctx (Unsloth Dynamic bf16)" {
    const s = mlx.gpuStream();

    // x: [1, 1, in=4], w: [out=2, in=4] (PyTorch convention).
    // Expected: x @ w.T = [1, 1, 2]
    var x_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_flat);
    try mlx.check(mlx.mlx_arange(&x_flat, 0.0, 4.0, 1.0, .float32, s));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    {
        const sh = [_]c_int{ 1, 1, 4 };
        try mlx.check(mlx.mlx_reshape(&x, x_flat, &sh, 3, s));
    }

    var w_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_flat);
    try mlx.check(mlx.mlx_arange(&w_flat, 0.0, 8.0, 1.0, .float32, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    {
        const sh = [_]c_int{ 2, 4 };
        try mlx.check(mlx.mlx_reshape(&w, w_flat, &sh, 2, s));
    }

    // Pre-transpose like initMoeLayers does for null-scales weights: [out, in] → [in, out]
    const w_t = try transposeBf16Weight(w, s);
    defer _ = mlx.mlx_array_free(w_t);

    // qmatmulBits with null sc/bi must plain-matmul x @ w_t.
    const null_sc = mlx.mlx_array{ .ctx = null };
    const null_bi = mlx.mlx_array{ .ctx = null };
    const got = try qmatmulBits(x, w_t, null_sc, null_bi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(got);

    // Reduce to host floats for comparison.
    var got_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got_f32);
    try mlx.check(mlx.mlx_astype(&got_f32, got, .float32, s));
    var got_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got_flat);
    {
        const sh = [_]c_int{2};
        try mlx.check(mlx.mlx_reshape(&got_flat, got_f32, &sh, 1, s));
    }
    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, got_flat);
        try mlx.check(mlx.mlx_eval(ev));
    }
    const data = mlx.mlx_array_data_float32(got_flat) orelse return error.TestUnexpectedNullData;
    // [0,1,2,3] @ [[0,4],[1,5],[2,6],[3,7]] = [0+1*1+2*2+3*3, 0+1*5+2*6+3*7] = [14, 38]
    // (w transposed from [[0,1,2,3],[4,5,6,7]] gives w_t = [[0,4],[1,5],[2,6],[3,7]])
    try testing.expectApproxEqAbs(@as(f32, 14.0), data[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 38.0), data[1], 1e-3);
}

test "qmatmulBits keeps row-axis MTP kernels out of plain batched projections" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const saved_kernel = qmv_rows.msv_qmv_rows_kernel;
    qmv_rows.msv_qmv_rows_kernel = null;
    defer {
        if (qmv_rows.msv_qmv_rows_kernel) |kernel| _ = mlx.mlx_fast_metal_kernel_free(kernel);
        qmv_rows.msv_qmv_rows_kernel = saved_kernel;
    }
    var prng = std.Random.DefaultPrng.init(0xB47C4);
    const w_bf = try attn256RandBf16(prng.random(), &.{ 512, 2560 }, s);
    defer _ = mlx.mlx_array_free(w_bf);
    for ([_]u32{ 4, 8 }) |bits| {
        var qvec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(qvec);
        try mlx.check(mlx.mlx_quantize(&qvec, w_bf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, qvec, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, qvec, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, qvec, 2));
        for ([_]c_int{ 2, 4, 5, 9 }) |rows| {
            const x = try attn256RandBf16(prng.random(), &.{ rows, 2560 }, s);
            defer _ = mlx.mlx_array_free(x);
            const y = try qmatmulBits(x, w, sc, bi, bits, 64, .affine, s);
            defer _ = mlx.mlx_array_free(y);
            try mlx.check(mlx.mlx_array_eval(y));
            if (qmv_rows.msv_qmv_rows_kernel != null) std.debug.print("generic projection invoked MTP row-axis kernel: rows={d} bits={d}\n", .{ rows, bits });
            try testing.expect(qmv_rows.msv_qmv_rows_kernel == null);
            // The dispatch order of qmatmulBits: simd_qmm takes 8+ rows where no verify tile does.
            const expected = (try simdQmmDecode(x, w, sc, bi, bits, 64, s)) orelse (try verifyQmm(s, x, w, sc, bi, bits, 64)) orelse (try qmv_cuda.qmm(x, w, sc, bi, bits, 64, s)) orelse blk: {
                var stock = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_quantized_matmul(&stock, x, w, sc, bi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));
                break :blk stock;
            };
            defer _ = mlx.mlx_array_free(expected);
            try testing.expect(try qsaArraysAllEqual(y, expected, s));
        }
    }
}

fn expectDqRoute(mode: QuantMode, bits: u32, gs: u32) !void {
    const s = mlx.gpuStream();
    const al = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xD0DE);
    const rnd = prng.random();
    const K: c_int = 256;
    const N: c_int = 384;
    const M: c_int = @intCast(PREFILL_DQ_GEMM_MIN_M);

    // Random bf16 weight quantized in `mode` (affine q4 gs64 is the 27B trunk class).
    const wn: usize = @intCast(N * K);
    const wbuf = try al.alloc(f32, wn);
    for (wbuf) |*v| v.* = bf16Trunc(rnd.float(f32) - 0.5);
    const wshape = [_]c_int{ N, K };
    const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 2, .float32);
    al.free(wbuf);
    defer _ = mlx.mlx_array_free(w32);
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), .{}, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    if (mode == .affine) try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

    const xn: usize = @intCast(M * K);
    const xbuf = try al.alloc(f32, xn);
    for (xbuf) |*v| v.* = bf16Trunc(rnd.float(f32) - 0.5);
    const xshape = [_]c_int{ 1, M, K };
    const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
    al.free(xbuf);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

    // f32 ground truth: fp32 dequant + fp32 GEMM.
    var wdq32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wdq32);
    try mlx.check(mlx.mlx_dequantize(&wdq32, wq, wsc, wbi, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = .float32, .has_value = true }, s));
    var wdq32_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wdq32_t);
    try mlx.check(mlx.mlx_transpose(&wdq32_t, wdq32, s));
    var gt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gt);
    try mlx.check(mlx.mlx_matmul(&gt, x32, wdq32_t, s));
    try mlx.check(mlx.mlx_array_eval(gt));
    const gtd = mlx.mlx_array_data_float32(gt) orelse return error.InvalidDtype;
    const on: usize = @intCast(M * N);

    // Stock qmm (route forced off).
    prefill_dq_gemm.prefill_dq_gemm_override = false;
    const stock = try qmatmulBits(x, wq, wsc, wbi, bits, gs, mode, s);
    defer _ = mlx.mlx_array_free(stock);
    const stock_f = try evalToF32(al, stock, on, s);
    defer al.free(stock_f);

    // Routed (default on): engagement must be COUNTED.
    prefill_dq_gemm.prefill_dq_gemm_override = true;
    defer prefill_dq_gemm.prefill_dq_gemm_override = null;
    const before = prefill_dq_gemm.prefill_dq_gemm_engaged;
    const routed = try qmatmulBits(x, wq, wsc, wbi, bits, gs, mode, s);
    defer _ = mlx.mlx_array_free(routed);
    try testing.expectEqual(before + 1, prefill_dq_gemm.prefill_dq_gemm_engaged);
    const routed_f = try evalToF32(al, routed, on, s);
    defer al.free(routed_f);

    var stock_err: f32 = 0;
    var routed_err: f32 = 0;
    for (0..on) |i| {
        stock_err = @max(stock_err, @abs(stock_f[i] - gtd[i]));
        routed_err = @max(routed_err, @abs(routed_f[i] - gtd[i]));
    }
    // bf16-weight-rounding is the only extra error source — same magnitude
    // class as the stock kernel's own bf16 output rounding.
    try testing.expect(routed_err <= 1.5 * stock_err + 5e-3);

    // Decode/verify widths never route (M below the floor).
    var x8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x8);
    {
        var x8_32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x8_32);
        const s8 = [_]c_int{ 0, 0, 0 };
        const e8 = [_]c_int{ 1, 8, K };
        const st8 = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&x8_32, x32, &s8, 3, &e8, 3, &st8, 3, s));
        try mlx.check(mlx.mlx_astype(&x8, x8_32, .bfloat16, s));
    }
    const small_before = prefill_dq_gemm.prefill_dq_gemm_engaged;
    const small = try qmatmulBits(x8, wq, wsc, wbi, bits, gs, mode, s);
    defer _ = mlx.mlx_array_free(small);
    try mlx.check(mlx.mlx_array_eval(small));
    try testing.expectEqual(small_before, prefill_dq_gemm.prefill_dq_gemm_engaged);
}

test "qmatmulBits: prefill-width affine calls take the dequant+GEMM route (engaged + no worse than stock)" {
    try expectDqRoute(.affine, 4, 64);
}

test "qmatmulBits: prefill-width MXFP8 calls take the dequant+GEMM route (engaged + no worse than stock)" {
    try expectDqRoute(.mxfp8, 8, 32);
}

test "qmatmulBits nvfp4 matches dequantize+matmul reference" {
    // NVFP4 (issue #24): packed-u32 weight + fp8-e4m3 uint8 scales, NO biases,
    // group_size 16. qmatmulBits must route the bias-less weight to mode
    // "nvfp4" — the legacy null-bias heuristic would misroute it to "mxfp8"
    // (bits=8) and produce garbage.
    const s = mlx.gpuStream();

    const OUT = 8;
    const IN = 64; // multiple of the nvfp4 group size (16)
    var w_host: [OUT * IN]f32 = undefined;
    for (&w_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5))) * 0.07;
    var w_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_f32);
    {
        const sh = [_]c_int{ OUT, IN };
        w_f32 = mlx.mlx_array_new_data(&w_host, &sh, 2, .float32);
    }
    var w_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_bf16);
    try mlx.check(mlx.mlx_astype(&w_bf16, w_f32, .bfloat16, s));

    // Quantize with mode=nvfp4 → vector [q, scales] (no biases element).
    var qvec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_quantize(&qvec, w_bf16, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", .{}, s));
    var q_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_w);
    var q_sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_sc);
    try mlx.check(mlx.mlx_vector_array_get(&q_w, qvec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q_sc, qvec, 1));

    const null_bi = mlx.mlx_array{ .ctx = null };

    // Reference: dequantize(mode=nvfp4) → x @ wᵀ in plain matmul.
    var w_deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_deq);
    try mlx.check(mlx.mlx_dequantize(&w_deq, q_w, q_sc, null_bi, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }, s));
    var w_deq_host: [OUT * IN]f32 = undefined;
    try testReadF32(w_deq, &w_deq_host, s);

    var x_host: [IN]f32 = undefined;
    for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2))) * 0.1;
    var x_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_f32);
    {
        const sh = [_]c_int{ 1, IN };
        x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 2, .float32);
    }
    var x_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_bf);
    try mlx.check(mlx.mlx_astype(&x_bf, x_f32, .bfloat16, s));
    var x_bf_host: [IN]f32 = undefined;
    try testReadF32(x_bf, &x_bf_host, s);

    // Ground truth from the dequantized weights.
    var gt: [OUT]f32 = undefined;
    for (0..OUT) |o| {
        var acc: f32 = 0;
        for (0..IN) |in| acc += x_bf_host[in] * w_deq_host[o * IN + in];
        gt[o] = acc;
    }

    const got = try qmatmulBits(x_bf, q_w, q_sc, null_bi, 4, 16, .nvfp4, s);
    defer _ = mlx.mlx_array_free(got);
    var got_host: [OUT]f32 = undefined;
    try testReadF32(got, &got_host, s);

    for (0..OUT) |o| try testing.expectApproxEqAbs(gt[o], got_host[o], 5e-2);
}

test "spec-verify width sweep µbench (diag verify-width-ubench)" {
    // What does one spec-verify forward cost as a function of the block width?
    // MLX routes transposed qmm by `get_qmv_batch_limit(K, N)`: below it the
    // weight-reuse qmv kernel runs (cost ~flat in M, bandwidth-bound), at and
    // above it the 32x32-tiled qmm_splitk runs (compute-bound, and a width of
    // 16 wastes half the 32-row tile). The limit is a per-shape number, so the
    // cliff is what this prints — a whole-model per-round estimate at each M.
    if (env.get(.VERIFY_WIDTH_UBENCH) == null) return error.SkipZigTest;
    const io_util = @import("../../util/io_util.zig");
    const tio = testing.io;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xBEEF);
    const rnd = prng.random();
    // One eval per BATCH of independent launches, not per launch: a per-call
    // eval barrier costs ~0.14 ms of sync on this box, which is more than the
    // whole kv projection and would swamp the very ratios being measured. A
    // real forward pipelines its dispatches, so the bench must too. Each
    // repetition slices its own x rows so the launches stay distinct ops.
    const WARM = 2;
    const ITERS = 6;
    const BATCH = 16;

    // Muse-Glimmer-30B trunk: hidden 6656, inter 19968, 32/2 heads hd128,
    // 52 layers, vocab 202048. `per_layer` counts how many times this shape
    // runs in one layer so the sweep can price a whole forward.
    const shapes = [_]struct { name: []const u8, k: c_int, n: c_int, per_layer: f64, layers: f64 }{
        .{ .name = "q", .k = 6656, .n = 4096, .per_layer = 1, .layers = 52 },
        .{ .name = "kv", .k = 6656, .n = 256, .per_layer = 2, .layers = 52 },
        .{ .name = "o", .k = 4096, .n = 6656, .per_layer = 1, .layers = 52 },
        .{ .name = "gate/up", .k = 6656, .n = 19968, .per_layer = 2, .layers = 52 },
        .{ .name = "down", .k = 19968, .n = 6656, .per_layer = 1, .layers = 52 },
        .{ .name = "lm_head", .k = 6656, .n = 202048, .per_layer = 1, .layers = 1 },
    };
    const widths = [_]c_int{ 1, 2, 3, 4, 5, 6, 7, 8, 10, 16 };
    var totals: [widths.len]f64 = @splat(0);

    std.debug.print("\n[vw-ubench] {s:>8} {s:>3} {s:>10} {s:>9}\n", .{ "shape", "M", "ms", "vs_M1" });
    for (shapes) |sh| {
        const wb = try testRandWeightBf16(rnd, &[_]c_int{ sh.n, sh.k }, s);
        defer _ = mlx.mlx_array_free(wb);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        var wsc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wsc);
        var wbi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wbi);
        try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));
        for ([_]mlx.mlx_array{ wq, wsc, wbi }) |a| try mlx.check(mlx.mlx_array_eval(a));

        var base_ms: f64 = 0;
        for (widths, 0..) |m, wi| {
            const xn: usize = @as(usize, @intCast(m * sh.k)) * BATCH;
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ BATCH, m, sh.k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var xall = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xall);
            try mlx.check(mlx.mlx_astype(&xall, x32, .bfloat16, s));
            var xs: [BATCH]mlx.mlx_array = undefined;
            defer for (&xs) |*a| {
                _ = mlx.mlx_array_free(a.*);
            };
            for (&xs, 0..) |*a, r| {
                a.* = mlx.mlx_array_new();
                const start = [_]c_int{ @intCast(r), 0, 0 };
                const stop = [_]c_int{ @as(c_int, @intCast(r)) + 1, m, sh.k };
                const strides = [_]c_int{ 1, 1, 1 };
                try mlx.check(mlx.mlx_slice(a, xall, &start, 3, &stop, 3, &strides, 3, s));
                try mlx.check(mlx.mlx_array_eval(a.*));
            }

            var it: usize = 0;
            var sw = io_util.Stopwatch.init(tio);
            while (it < WARM + ITERS) : (it += 1) {
                if (it == WARM) sw.reset();
                var outs: [BATCH]mlx.mlx_array = undefined;
                for (&outs, &xs) |*o, xi| o.* = try qmatmulBits(xi, wq, wsc, wbi, 4, 64, .affine, s);
                const vec = mlx.mlx_vector_array_new_data(&outs, outs.len);
                try mlx.check(mlx.mlx_eval(vec));
                _ = mlx.mlx_vector_array_free(vec);
                for (outs) |o| _ = mlx.mlx_array_free(o);
            }
            const ms = @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS * BATCH) / 1e6;
            if (wi == 0) base_ms = ms;
            totals[wi] += ms * sh.per_layer * sh.layers;
            std.debug.print("[vw-ubench] {s:>8} {d:>3} {d:>10.4} {d:>8.2}x\n", .{ sh.name, m, ms, ms / base_ms });
        }
    }
    std.debug.print("\n[vw-ubench] whole-forward estimate (all linears, 52 layers + lm_head)\n", .{});
    std.debug.print("[vw-ubench] {s:>3} {s:>10} {s:>9} {s:>12}\n", .{ "M", "ms", "vs_M1", "ms/token" });
    for (widths, totals) |m, tot| {
        std.debug.print("[vw-ubench] {d:>3} {d:>10.2} {d:>8.2}x {d:>12.2}\n", .{ m, tot, tot / totals[0], tot / @as(f64, @floatFromInt(m)) });
    }
}

test "simd_qmm decode rows: from 8 where the verify lanes stop at 7, off beside a tile" {
    try testing.expectEqual(@as(c_int, 8), simdQmmDecodeMinRows(.off));
    try testing.expectEqual(@as(c_int, 0), simdQmmDecodeMinRows(.nax));
    try testing.expectEqual(@as(c_int, 0), simdQmmDecodeMinRows(.shader));
}
