//! sdk/blocks/linear/prefill_dq_gemm.
//! Owns PREFILL_DQ_GEMM_MIN_M, prefillDqGemmMinRows, prefillDqGemmRowFloor, prefillDqGemmRowFloorNow, prefill_dq_gemm_env_cached, prefillDqGemmEnabled, and 2 more.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const testing = std.testing;
const lastDim = @import("../../../mlx/array.zig").lastDim;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const dualDieArch = @import("verify_projection.zig").dualDieArch;
const verifyArch = @import("verify_projection.zig").verifyArch;

// ── Prefill-width dequant+GEMM qmm route ──
// Stock MLX qmm_t runs a 32x32x32 steel tile; at prefill widths (M >= 2048)
// on the qwen 27B shapes that underuses the compute units by ~10% (µbench,
// M4 Max: gate/up q4 M=2048 stock 28.05 ms vs dequant+bf16-GEMM 25.36 ms
// INCLUDING the per-call dequant; pre-dequantized floor 24.64 ms). oMLX
// closes the same gap by re-instantiating qmm_t at bigger tiles (bm 64-128,
// their qwen35_q4_mlp patch); we get within ~3% of that with zero custom
// kernel: dequantize the weight to a bf16 [N,K] transient and run MLX's
// steel GEMM. The transient repeats identical sizes per layer, so the MLX
// allocator cache recycles it; numerics differ from in-kernel dequant only
// by the bf16 rounding of w = s*q + b (pinned no-worse-than-stock by test).
// Decode (M=1) and spec-verify widths never route.
pub const PREFILL_DQ_GEMM_MIN_M: usize = 2048;

/// Rows from which `prefillDqGemm` takes over from stock qmm. 2-bit qmm loses
/// from 384 rows (M4 Max, 5120x17408: +5% at 384, +10% at 1024); other widths
/// keep the 2048 they were tuned on.
pub fn prefillDqGemmMinRows(bits: u32) usize {
    return if (bits == 2) 384 else PREFILL_DQ_GEMM_MIN_M;
}

/// Rows from which one [M,K] x [N,K]^T call takes the route. Pure over its inputs.
/// `prefillDqGemmMinRows` stays the smallest floor any shape gets, which is what the
/// admission bill needs.
pub fn prefillDqGemmRowFloor(bits: u32, ultra_nax: bool, n: usize, k: usize) usize {
    const base = prefillDqGemmMinRows(bits);
    // Off NAX stock qmm is the 32x32 steel tile the base floors were tuned against. On the Ultra's NAX
    // stock qmm is the fused kernel, which holds out to ~0.9 * min(N, K) rows once both N and K are
    // 4096 or more. Below that (Flash-Next's K 640, K 2560 and N 2560 shapes) and at K >= 3N, where
    // the dequantized GEMM runs MLX's split-K kernel, dequant+GEMM wins from the base floor.
    if (!ultra_nax or bits == 2 or @min(n, k) < 4096 or k >= 3 * n) return base;
    return @max(base, @min(n, k) * 9 / 10);
}

/// The shape-aware floor is measured on the M5 Ultra only. The M5 Max crosses the other way on
/// K 5120 (dequant+GEMM wins from 2048 rows; MLX picks different GEMM tiles per arch suffix), so
/// every other chip keeps the base floor.
fn prefillDqGemmRowFloorNow(bits: u32, n: usize, k: usize) usize {
    const ultra_nax = naxAvailable() and dualDieArch(verifyArch() orelse "");
    return prefillDqGemmRowFloor(bits, ultra_nax, n, k);
}

/// Off on CUDA: there it measured slower than stock qmm (3845 vs 4400 tok/s, Llama 3.2 3B, RTX 5060 Ti) and its
/// bf16 weight transients stacked ~5 GB into the prefill peak.
pub fn prefillDqGemmEnabled() bool {
    return prefill_dq_gemm_override orelse !mlx.cudaAvailable();
}

/// Test seam: engagement is counted, never inferred from output equality.
pub var prefill_dq_gemm_engaged: u64 = 0;

/// The dequant+GEMM prefill route (see the block comment above
/// prefillDqGemmEnabled). Returns null when the call must stay on stock qmm.
pub fn prefillDqGemm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!prefillDqGemmEnabled()) return null;
    switch (bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return null,
    }
    const last = lastDim(x) orelse return null;
    if (last == 0) return null;
    const rows = mlx.mlx_array_size(x) / @as(usize, @intCast(last));
    const n: usize = @intCast(mlx.getShape(w)[0]);
    if (rows < prefillDqGemmRowFloorNow(bits, n, @intCast(last))) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;

    var dq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq);
    try mlx.check(mlx.mlx_dequantize(&dq, w, sc, bi, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = xd, .has_value = true }, s));
    var dq_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq_t);
    try mlx.check(mlx.mlx_transpose(&dq_t, dq, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_matmul(&out, x, dq_t, s));
    prefill_dq_gemm_engaged += 1;
    return out;
}

/// Test seam: forces the route on/off without the environment.
pub var prefill_dq_gemm_override: ?bool = null;

// ── tests ──

const qmatmulBits = @import("qmatmul.zig").qmatmulBits;

test "prefillDqGemmRowFloor: on the Ultra the floor follows min(N, K) from 4096, the rest keep theirs" {
    // M5 Ultra crossovers, stock NAX qmm vs dequant+GEMM: K 640 and 2560 win from 2048
    // (Flash-Next, including the 2048-row company chunk), K 5120 from ~4608 (Qwen3.8-27B).
    try std.testing.expectEqual(PREFILL_DQ_GEMM_MIN_M, prefillDqGemmRowFloor(4, true, 10240, 2560));
    try std.testing.expectEqual(PREFILL_DQ_GEMM_MIN_M, prefillDqGemmRowFloor(4, true, 2560, 640));
    try std.testing.expectEqual(@as(usize, 4608), prefillDqGemmRowFloor(4, true, 17408, 5120));
    // At K 6144 the smaller dimension sets it. N 2560 (Flash-Next GDN out, attn o) keeps the base
    // floor: the Python sweep had stock qmm ahead there, but the server's forward reads 0.9% slower
    // with stock below 2304 rows (4+4 A/B at 2,247 tokens), so only N >= 4096 gets a raised floor.
    try std.testing.expectEqual(PREFILL_DQ_GEMM_MIN_M, prefillDqGemmRowFloor(4, true, 2560, 6144));
    try std.testing.expectEqual(@as(usize, 4608), prefillDqGemmRowFloor(4, true, 5120, 6144));
    // K >= 3N runs MLX's split-K GEMM, which wins from 2048 (27B down, 5120x17408).
    try std.testing.expectEqual(PREFILL_DQ_GEMM_MIN_M, prefillDqGemmRowFloor(4, true, 5120, 17408));
    // Off the Ultra and at 2 bits the M4 Max floors stand.
    try std.testing.expectEqual(PREFILL_DQ_GEMM_MIN_M, prefillDqGemmRowFloor(4, false, 17408, 5120));
    try std.testing.expectEqual(@as(usize, 384), prefillDqGemmRowFloor(2, true, 17408, 5120));
    // The admission bill's floor is never above any shape's.
    try std.testing.expect(prefillDqGemmMinRows(4) <= prefillDqGemmRowFloor(4, true, 10240, 2560));
}

test "prefillDqGemm defaults on where it was measured a win, off on CUDA" {
    const saved = prefill_dq_gemm_override;
    defer prefill_dq_gemm_override = saved;
    prefill_dq_gemm_override = null;
    try std.testing.expectEqual(!mlx.cudaAvailable(), prefillDqGemmEnabled());
}

test "prefillDqGemm: 2-bit weights take the dequant route from 384 rows, other widths from 2048" {
    // Measured on the M4 Max: 2-bit qmm loses to dequant+GEMM from 384 rows
    // (+5%, +10% at 1024); the 2048 floor stays for the widths it was tuned on.
    prefill_dq_gemm_override = true; // the row floors, whatever the platform default
    defer prefill_dq_gemm_override = null;
    const s = mlx.gpuStream();
    const K: c_int = 256;
    const N: c_int = 128;
    var wz = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wz);
    try mlx.check(mlx.mlx_zeros(&wz, &[_]c_int{ N, K }, 2, .float16, s));
    for ([_]u32{ 2, 4 }) |bits| {
        const gs: u32 = if (bits == 2) 128 else 64;
        var q = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(q);
        try mlx.check(mlx.mlx_quantize(&q, wz, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&wq, q, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, q, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, q, 2));
        for ([_]c_int{ 256, 512 }) |m| {
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_zeros(&x, &[_]c_int{ 1, m, K }, 3, .float16, s));
            const before = prefill_dq_gemm_engaged;
            const y = try prefillDqGemm(x, wq, sc, bi, bits, gs, .affine, s);
            defer if (y) |a| {
                _ = mlx.mlx_array_free(a);
            };
            try testing.expectEqual(bits == 2 and m == 512, y != null);
            try testing.expectEqual(prefill_dq_gemm_engaged, before + @intFromBool(y != null));
        }
    }
}

test "prefill qmm µbench: stock qmm vs dequant+GEMM at 27B prefill shapes (diag prefill-qmm-ubench)" {
    if (env.get(.PREFILL_QMM_UBENCH) == null) return error.SkipZigTest;
    const io_util = @import("../../util/io_util.zig");
    const tio = testing.io;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xFEED);
    const rnd = prng.random();
    const WARM = 3;
    const ITERS = 10;

    // The 27B oQ4e prefill shapes: MLP gate/up (q4 gs64), MLP down (q5
    // gs64), GDN qkvz (q4). oMLX's prefill patch replaces exactly this class
    // with retiled qmm_t (bm 64-128 vs stock 32) — measure whether
    // dequant+dense-GEMM (steel bf16 gemm at near-peak) beats stock qmm
    // per-call, and what the amortized (pre-dequantized) ceiling is.
    const shapes = [_]struct { name: []const u8, k: c_int, n: c_int, bits: c_int }{
        .{ .name = "gate/up-q4", .k = 5120, .n = 17408, .bits = 4 },
        .{ .name = "down-q5", .k = 17408, .n = 5120, .bits = 5 },
        .{ .name = "qkvz-q4", .k = 5120, .n = 16384, .bits = 4 },
    };
    std.debug.print("\n[pqmm-ubench] {s:>10} {s:>5} {s:>9} {s:>12} {s:>12}\n", .{ "shape", "M", "stock_ms", "dq+gemm_ms", "gemm_ms" });
    for (shapes) |sh| {
        const wb = try testRandWeightBf16(rnd, &[_]c_int{ sh.n, sh.k }, s);
        defer _ = mlx.mlx_array_free(wb);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(sh.bits), "affine", .{}, s));
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

        // Pre-dequantized weight for the amortized-GEMM ceiling.
        var wdq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq);
        try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(sh.bits), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = .bfloat16, .has_value = true }, s));
        var wdq_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq_t);
        try mlx.check(mlx.mlx_transpose(&wdq_t, wdq, s));
        try mlx.check(mlx.mlx_array_eval(wdq_t));

        for ([_]c_int{ 2048, 4096 }) |m| {
            const xn: usize = @intCast(m * sh.k);
            const xbuf = try allocator.alloc(f32, xn);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, sh.k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            allocator.free(xbuf);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
            try mlx.check(mlx.mlx_array_eval(x));

            var stock_ms: f64 = 0;
            {
                var it: usize = 0;
                var sw = io_util.Stopwatch.init(tio);
                while (it < WARM + ITERS) : (it += 1) {
                    if (it == WARM) sw.reset();
                    const r1 = try qmatmulBits(x, wq, wsc, wbi, @intCast(sh.bits), 64, .affine, s);
                    try mlx.check(mlx.mlx_array_eval(r1));
                    _ = mlx.mlx_array_free(r1);
                }
                stock_ms = @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS) / 1e6;
            }
            var dq_ms: f64 = 0;
            {
                var it: usize = 0;
                var sw = io_util.Stopwatch.init(tio);
                while (it < WARM + ITERS) : (it += 1) {
                    if (it == WARM) sw.reset();
                    var dq = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_dequantize(&dq, wq, wsc, wbi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(sh.bits), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = .bfloat16, .has_value = true }, s));
                    var dq_t = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_transpose(&dq_t, dq, s));
                    var r2 = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_matmul(&r2, x, dq_t, s));
                    try mlx.check(mlx.mlx_array_eval(r2));
                    _ = mlx.mlx_array_free(dq);
                    _ = mlx.mlx_array_free(dq_t);
                    _ = mlx.mlx_array_free(r2);
                }
                dq_ms = @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS) / 1e6;
            }
            var gemm_ms: f64 = 0;
            {
                var it: usize = 0;
                var sw = io_util.Stopwatch.init(tio);
                while (it < WARM + ITERS) : (it += 1) {
                    if (it == WARM) sw.reset();
                    var r3 = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_matmul(&r3, x, wdq_t, s));
                    try mlx.check(mlx.mlx_array_eval(r3));
                    _ = mlx.mlx_array_free(r3);
                }
                gemm_ms = @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS) / 1e6;
            }
            _ = mlx.mlx_clear_cache();
            std.debug.print("[pqmm-ubench] {s:>10} {d:>5} {d:>9.2} {d:>12.2} {d:>12.2}\n", .{ sh.name, m, stock_ms, dq_ms, gemm_ms });
        }
    }
}
