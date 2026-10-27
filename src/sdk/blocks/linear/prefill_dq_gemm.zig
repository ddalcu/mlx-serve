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
    const cd = prefillComputeDtype(xd, mode, bi.ctx != null);

    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    if (cd != xd) {
        try mlx.check(mlx.mlx_astype(&xc, x, cd, s));
        f16_prefill_engaged += 1;
    }
    var dq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq);
    try mlx.check(mlx.mlx_dequantize(&dq, w, sc, bi, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = cd, .has_value = true }, s));
    var dq_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq_t);
    try mlx.check(mlx.mlx_transpose(&dq_t, dq, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_matmul(&out, if (cd != xd) xc else x, dq_t, s));
    if (cd != xd) {
        var back = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(back);
        try mlx.check(mlx.mlx_astype(&back, out, xd, s));
        _ = mlx.mlx_array_free(out);
        out = back;
    }
    prefill_dq_gemm_engaged += 1;
    return out;
}

/// Test seam: forces the route on/off without the environment.
pub var prefill_dq_gemm_override: ?bool = null;

// ── FP16 prefill GEMMs for bf16 models (opt-in lever MLX_SERVE_F16_PREFILL) ──
// GPUs without fast bf16 math run the steel GEMM and gather_qmm faster in fp16. With the lever on,
// the routed prefill GEMMs (prefillDqGemm here, the sorted affine gather in moe/experts.zig) cast a
// bf16 activation and the dequantized weight / quant params to fp16, multiply, and cast the result
// back to bf16. Lossy (fp16 overflows past 65504), hence opt-in. Decode and verify widths never
// reach these routes.
var f16_prefill_env_cached: ?bool = null;

/// Test seam: forces the lever on/off without the environment.
pub var f16_prefill_override: ?bool = null;

pub fn f16PrefillEnabled() bool {
    return f16_prefill_override orelse env.onCached(&f16_prefill_env_cached, .MLX_SERVE_F16_PREFILL);
}

/// Test seam: engagement is counted, never inferred from output equality.
pub var f16_prefill_engaged: u64 = 0;

/// The dtype a routed prefill GEMM multiplies in for activation dtype `xd`: fp16 for a bf16
/// activation over an affine pack with biases while the lever is on, else `xd`.
pub fn prefillComputeDtype(xd: mlx.mlx_dtype, mode: QuantMode, has_biases: bool) mlx.mlx_dtype {
    if (xd != .bfloat16 or mode != .affine or !has_biases or !f16PrefillEnabled()) return xd;
    return .float16;
}

// ── tests ──

const qmatmulBits = @import("qmatmul.zig").qmatmulBits;
const Scope = @import("../../../mlx/scope.zig").Scope;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const maxAbsDiffF32 = @import("../../testing/parity.zig").maxAbsDiffF32;

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

pub const TruthErr = struct { rms: f64, max: f64 };

/// rms and max |y - truth| over an f32 reference (test helper for the fp16 lever arms).
pub fn testErrVsTruth(y: mlx.mlx_array, truth: mlx.mlx_array, s: mlx.mlx_stream) !TruthErr {
    const n = mlx.mlx_array_size(truth);
    if (mlx.mlx_array_size(y) != n) return error.ShapeMismatch;
    const a = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(a);
    const b = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(b);
    try testReadF32(y, a, s);
    try testReadF32(truth, b, s);
    var sq: f64 = 0;
    var max: f64 = 0;
    for (a, b) |got, want| {
        const d: f64 = @abs(@as(f64, got) - @as(f64, want));
        if (!std.math.isFinite(d)) return error.NonFiniteValue;
        sq += d * d;
        max = @max(max, d);
    }
    return .{ .rms = @sqrt(sq / @as(f64, @floatFromInt(n))), .max = max };
}

/// An affine 4-bit gs64 pack of a random bf16 weight `shape`: q, scales, biases, and the f32 dequant (the truth's weight).
pub fn testAffinePack(sc: *Scope, rnd: std.Random, shape: []const c_int) !struct { q: mlx.mlx_array, scales: mlx.mlx_array, biases: mlx.mlx_array, w32: mlx.mlx_array } {
    const s = sc.s;
    const wb = try sc.keep(try testRandWeightBf16(rnd, shape, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var q = mlx.mlx_array_new();
    var scales = mlx.mlx_array_new();
    var biases = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&q, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&scales, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&biases, triple, 2));
    _ = try sc.keep(q);
    _ = try sc.keep(scales);
    _ = try sc.keep(biases);
    var w32 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_dequantize(&w32, q, try sc.astype(scales, .float32), try sc.astype(biases, .float32), mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    return .{ .q = q, .scales = scales, .biases = biases, .w32 = try sc.keep(w32) };
}

test "prefillDqGemm: the fp16 lever on a bf16 activation is no worse than the bf16 arm against f32 truth" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const saved_route = prefill_dq_gemm_override;
    defer prefill_dq_gemm_override = saved_route;
    prefill_dq_gemm_override = true;
    const saved_lever = f16_prefill_override;
    defer f16_prefill_override = saved_lever;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const s = sc.s;
    var prng = std.Random.DefaultPrng.init(0xF16);
    // The base floor: N and K under 4096 keep it on every chip, so this width takes the route.
    const M: c_int = @intCast(PREFILL_DQ_GEMM_MIN_M);
    const N: c_int = 256;
    const K: c_int = 512;
    const pack = try testAffinePack(&sc, prng.random(), &.{ N, K });
    // Activations past the weights' range, as a residual stream carries: |x| up to 8.
    const xr = try sc.keep(try testRandWeightBf16(prng.random(), &.{ 1, M, K }, s));
    const x = try sc.astype(try sc.mulS(try sc.astype(xr, .float32), 16), .bfloat16);
    const truth = try sc.matmul(try sc.astype(x, .float32), try sc.transpose(pack.w32, &.{ 1, 0 }));

    var errs: [2]TruthErr = undefined;
    for ([_]bool{ false, true }, 0..) |lever, i| {
        f16_prefill_override = lever;
        const route_before = prefill_dq_gemm_engaged;
        const lever_before = f16_prefill_engaged;
        const y = (try prefillDqGemm(x, pack.q, pack.scales, pack.biases, 4, 64, .affine, s)) orelse return error.RouteDeclined;
        defer _ = mlx.mlx_array_free(y);
        try testing.expectEqual(route_before + 1, prefill_dq_gemm_engaged);
        try testing.expectEqual(lever_before + @intFromBool(lever), f16_prefill_engaged);
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y));
        try testing.expectEqualSlices(c_int, mlx.getShape(truth), mlx.getShape(y));
        errs[i] = try testErrVsTruth(y, truth, s);
    }
    // Both arms round the output to bf16; the fp16 arm must not add error beyond that.
    try testing.expect(errs[1].rms <= 1.05 * errs[0].rms);
    try testing.expect(errs[1].max <= 1.05 * errs[0].max);
}

test "prefillDqGemm: the fp16 lever leaves a non-affine pack and an fp16 activation alone" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const saved_route = prefill_dq_gemm_override;
    defer prefill_dq_gemm_override = saved_route;
    prefill_dq_gemm_override = true;
    const saved_lever = f16_prefill_override;
    defer f16_prefill_override = saved_lever;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const s = sc.s;
    var prng = std.Random.DefaultPrng.init(0xF17);
    const M: c_int = @intCast(PREFILL_DQ_GEMM_MIN_M);
    const N: c_int = 256;
    const K: c_int = 512;
    const wb = try sc.keep(try testRandWeightBf16(prng.random(), &.{ N, K }, s));
    var pair = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(pair);
    try mlx.check(mlx.mlx_quantize(&pair, wb, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", .{}, s));
    var wq = mlx.mlx_array_new();
    var wsc = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&wq, pair, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, pair, 1));
    _ = try sc.keep(wq);
    _ = try sc.keep(wsc);
    const pack = try testAffinePack(&sc, prng.random(), &.{ N, K });
    const xb = try sc.keep(try testRandWeightBf16(prng.random(), &.{ 1, M, K }, s));
    const xh = try sc.astype(xb, .float16);

    const cases = [_]struct { x: mlx.mlx_array, q: mlx.mlx_array, scales: mlx.mlx_array, biases: mlx.mlx_array, gs: u32, mode: QuantMode }{
        .{ .x = xb, .q = wq, .scales = wsc, .biases = .{ .ctx = null }, .gs = 16, .mode = .nvfp4 },
        .{ .x = xh, .q = pack.q, .scales = pack.scales, .biases = pack.biases, .gs = 64, .mode = .affine },
    };
    for (cases) |c| {
        var ys: [2]mlx.mlx_array = undefined;
        for ([_]bool{ false, true }, 0..) |lever, i| {
            f16_prefill_override = lever;
            const lever_before = f16_prefill_engaged;
            ys[i] = (try prefillDqGemm(c.x, c.q, c.scales, c.biases, 4, c.gs, c.mode, s)) orelse return error.RouteDeclined;
            _ = try sc.keep(ys[i]);
            try testing.expectEqual(lever_before, f16_prefill_engaged);
            try testing.expectEqual(mlx.mlx_array_dtype(c.x), mlx.mlx_array_dtype(ys[i]));
        }
        try testing.expectEqual(@as(f32, 0), try maxAbsDiffF32(ys[0], ys[1], s));
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
