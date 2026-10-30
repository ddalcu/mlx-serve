//! `verify_qmm.zig`'s tests, out of its import graph: a plugin unit that reaches `verify_qmm.zig` never compiles them.

const verify_qmm = @import("verify_qmm.zig");
const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const testing = std.testing;
const gpuArchitecture = @import("../../../mlx/device.zig").gpuArchitecture;
const macosProductVersion = @import("../../../mlx/device.zig").macosProductVersion;
const naxArchIsG17 = @import("../../../mlx/device.zig").naxArchIsG17;
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const naxMinMForBits = @import("verify_qmm_nax.zig").naxMinMForBits;
const naxPadRows = @import("verify_qmm_nax.zig").naxPadRows;
const naxSliceRows = @import("verify_qmm_nax.zig").naxSliceRows;
const verifyQmmNaxAvailable = @import("verify_qmm_nax.zig").verifyQmmNaxAvailable;
const verifyQmmNaxEnabled = @import("verify_qmm_nax.zig").verifyQmmNaxEnabled;
const verifyQmmNaxEnabledForMFrom = @import("verify_qmm_nax.zig").verifyQmmNaxEnabledForMFrom;
const verifyQmmTile = @import("verify_qmm_nax.zig").verifyQmmTile;
const verify_qmm_nax_vars = @import("verify_qmm_nax.zig");
const MIXED_PLAIN_MIN_N = verify_qmm.MIXED_PLAIN_MIN_N;
const VQMM_MAX_ACC = verify_qmm.VQMM_MAX_ACC;
const VqmmLane = verify_qmm.VqmmLane;
const VqmmScalar = verify_qmm.VqmmScalar;
const VqmmTile = verify_qmm.VqmmTile;
const mixedNaxShapeEnabled = verify_qmm.mixedNaxShapeEnabled;
const mixedPlainShapeEnabled = verify_qmm.mixedPlainShapeEnabled;
const runVerifyQmmMsg = verify_qmm.runVerifyQmmMsg;
const verifyQmm = verify_qmm.verifyQmm;
const vqmmLaneFor = verify_qmm.vqmmLaneFor;
const vqmmLaneForTile = verify_qmm.vqmmLaneForTile;
const vqmmScalarEvictIndex = verify_qmm.vqmmScalarEvictIndex;
const vqmmTileFrom = verify_qmm.vqmmTileFrom;
const vqmmTileRows = verify_qmm.vqmmTileRows;
const OQE_FULL_O_BITS = @import("../../../engines/mlx/spec/mtp_nax_profile.zig").OQE_FULL_O_BITS;
const OQE_FULL_QK_BITS = @import("../../../engines/mlx/spec/mtp_nax_profile.zig").OQE_FULL_QK_BITS;
const OQE_FULL_V_BITS = @import("../../../engines/mlx/spec/mtp_nax_profile.zig").OQE_FULL_V_BITS;
const OQE_LINEAR_AB_BITS = @import("../../../engines/mlx/spec/mtp_nax_profile.zig").OQE_LINEAR_AB_BITS;
const OQE_LINEAR_OUT_BITS = @import("../../../engines/mlx/spec/mtp_nax_profile.zig").OQE_LINEAR_OUT_BITS;
const OQE_LINEAR_Z_BITS = @import("../../../engines/mlx/spec/mtp_nax_profile.zig").OQE_LINEAR_Z_BITS;
const OQE_MLP_DOWN_BITS = @import("../../../engines/mlx/spec/mtp_nax_profile.zig").OQE_MLP_DOWN_BITS;
const oqeLayerBits = @import("../../../engines/mlx/spec/mtp_nax_profile.zig").oqeLayerBits;

test "vqmm scalar cache never evicts a scalar the caller is still holding" {
    // The live crash, as a pure decision: every verifyQmm-family call site
    // takes K then N back to back and holds BOTH handles in one `inputs_arr`.
    // The old policy evicted slot 0 unconditionally, so the N call freed the
    // array the K call had just been handed — a dangling entry that killed the
    // server inside mlx_vector_array_new_data / _free.
    var cache: [16]VqmmScalar = undefined;
    for (&cache, 0..) |*e, i| {
        e.* = .{ .v = @intCast(i), .arr = .{ .ctx = null }, .tick = @intCast(i + 1) };
    }

    // Call 1 (K): the cache is full, so something is evicted and K takes its
    // slot with the newest tick.
    const k_slot = vqmmScalarEvictIndex(&cache);
    try std.testing.expectEqual(@as(usize, 0), k_slot); // oldest tick == slot 0 here
    cache[k_slot] = .{ .v = 1000, .arr = .{ .ctx = null }, .tick = 17 };

    // Call 2 (N): the victim must NOT be the slot K just took — the caller is
    // still holding that handle. Under the old always-slot-0 policy this
    // returned 0 and freed K.
    const n_slot = vqmmScalarEvictIndex(&cache);
    try std.testing.expect(n_slot != k_slot);
    cache[n_slot] = .{ .v = 2000, .arr = .{ .ctx = null }, .tick = 18 };

    // And it generalizes: after any run of insertions, the two most recently
    // ticked entries are never the next victim.
    var tick: u64 = 19;
    var round: usize = 0;
    while (round < 40) : (round += 1) {
        const victim = vqmmScalarEvictIndex(&cache);
        var newest: u64 = 0;
        var second: u64 = 0;
        for (cache) |e| {
            if (e.tick > newest) {
                second = newest;
                newest = e.tick;
            } else if (e.tick > second) second = e.tick;
        }
        try std.testing.expect(cache[victim].tick != newest);
        try std.testing.expect(cache[victim].tick != second);
        cache[victim] = .{ .v = @intCast(3000 + round), .arr = .{ .ctx = null }, .tick = tick };
        tick += 1;
    }
}

/// Assert a verify-kernel output is NO LESS ACCURATE than the stock qmm it
/// replaces, both measured against `wdq_t` — the SAME 4-bit weights dequantized
/// to fp32 (so quantization error cancels and only the two kernels' own
/// arithmetic is under test).
///
/// This is the machine-independent invariant. A direct kernel-vs-stock bound is
/// NOT: both paths accumulate fp32 in a different order and round to bf16, so on
/// heavily-cancelling data each already sits ~0.03 from truth on its worst
/// element and they may legitimately differ from each other by more than that.
/// Tuning a pair-agreement threshold on one GPU produces a test that is green on
/// the dev machine and red on the CI runner while the kernel is bit-for-bit as
/// accurate (measured: kernel-vs-truth == stock-vs-truth to four decimals).
/// A genuine defect — a partial-sum race, a register spill, bad indexing —
/// pushes kernel-vs-truth far past stock-vs-truth and is still caught here.
/// Parity bars for a verify lane, stated so that no bar is fit per GPU.
///
/// A per-chip max-error slack was tried first (M1 Pro measured kern_max 0.0313
/// against stock 0.0039 while cosine stayed at 0.999998, so the bar was
/// widened to 0.04 there). That is the wrong shape of guard twice over:
///
///   - Widening the MAX blinds the test to what it exists to catch. Cosine is
///     a GLOBAL measure, so a partial-sum race, a register spill or a bad
///     index — one element far off, everything else intact — passes a widened
///     max with cosine unmoved.
///   - `stock_max` is not a stable reference. The same shapes measure
///     stock_max 0.02-0.035 on M4 Max, so the M1 Pro row was not a lane
///     landing 8x worse than normal, it was STOCK landing unusually well
///     there. Anything keyed to stock's worst element inherits that.
///
/// So neither bar references stock's max any more:
///
///   - `GROSS_CEILING` is an ABSOLUTE ceiling on the worst element. On the
///     clamped-relative metric below, a bf16 output's own rounding noise sits
///     at 0.02-0.035 on every machine measured; a corrupted accumulator lands
///     orders of magnitude past 0.5. This is the single-element bar.
///   - `RMS_FACTOR` bounds the whole error distribution against stock's,
///     which is scale-free: when stock is unusually accurate the ratio is
///     what stays meaningful. This is the systematic bar. (M1 Pro's cosines
///     imply a ratio near 1.4; M4 Max measures 1.00 across every lane.)
///   - The cosine bars are unchanged and catch a broken reference.
const VerifyQmmParity = struct {
    /// Absolute ceiling on the worst element's clamped relative error.
    const GROSS_CEILING: f64 = 0.5;
    /// How much of stock's RMS error the lane may carry.
    const RMS_FACTOR: f64 = 3.0;
    /// Cosine may not fall below stock's by more than this.
    const COS_SLACK: f64 = 1e-5;
    /// …and both must actually be tracking the truth.
    const COS_FLOOR: f64 = 0.999;
};

/// Why a verify-lane parity check failed, or null when it passed. Pure so the
/// decision itself is unit-testable without a GPU — each arm is a distinct
/// failure shape the bars are meant to separate.
const VerifyQmmParityFail = enum { gross_element, systematic, cosine, not_tracking };

fn verifyQmmParityVerdict(
    kern_max: f64,
    rms_stock: f64,
    rms_kern: f64,
    cos_stock: f64,
    cos_kern: f64,
) ?VerifyQmmParityFail {
    if (cos_kern < VerifyQmmParity.COS_FLOOR) return .not_tracking;
    if (cos_kern < cos_stock - VerifyQmmParity.COS_SLACK) return .cosine;
    if (kern_max > VerifyQmmParity.GROSS_CEILING) return .gross_element;
    // A zero stock RMS means the reference is exact on this data; the ratio
    // then demands the lane be exact too, which is the strictest reading and
    // the one we want — an exactly-representable case a lane cannot reproduce
    // is a defect, not reduction-order noise.
    if (rms_kern > VerifyQmmParity.RMS_FACTOR * rms_stock) return .systematic;
    return null;
}

fn expectVerifyQmmNoWorseThanStock(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    wq: mlx.mlx_array,
    wsc: mlx.mlx_array,
    wbi: mlx.mlx_array,
    bits: u32,
    gs: u32,
    wdq_t: mlx.mlx_array,
    got: mlx.mlx_array,
    label: []const u8,
) !void {
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_quantized_matmul(&ref, x, wq, wsc, wbi, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));

    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var truth = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(truth);
    try mlx.check(mlx.mlx_matmul(&truth, xf, wdq_t, s));

    var tf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tf);
    var rf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rf);
    var gf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gf);
    try mlx.check(mlx.mlx_astype(&tf, truth, .float32, s));
    try mlx.check(mlx.mlx_astype(&rf, ref, .float32, s));
    try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
    try mlx.check(mlx.mlx_array_eval(tf));
    try mlx.check(mlx.mlx_array_eval(rf));
    try mlx.check(mlx.mlx_array_eval(gf));
    const td = mlx.mlx_array_data_float32(tf).?;
    const rd = mlx.mlx_array_data_float32(rf).?;
    const gd = mlx.mlx_array_data_float32(gf).?;

    const gsh = mlx.getShape(got);
    const count: usize = @intCast(gsh[1] * gsh[2]);
    var stock_max: f64 = 0;
    var kern_max: f64 = 0;
    var dot_s: f64 = 0;
    var dot_k: f64 = 0;
    var nt: f64 = 0;
    var ns: f64 = 0;
    var nk: f64 = 0;
    // First pass: cosines against the fp32 truth, plus each side's worst
    // element. Neither max is a BAR any more (see VerifyQmmParity) — stock's
    // is reported alongside the kernel's so a failure line shows both.
    for (0..count) |i| {
        const t: f64 = td[i];
        const r: f64 = rd[i];
        const g: f64 = gd[i];
        // Clamped denominator: a near-zero output of a heavily-cancelling dot
        // product must not manufacture a huge ratio from a tiny absolute error.
        const denom = @max(1.0, @abs(t));
        stock_max = @max(stock_max, @abs(r - t) / denom);
        kern_max = @max(kern_max, @abs(g - t) / denom);
        dot_s += r * t;
        dot_k += g * t;
        nt += t * t;
        ns += r * r;
        nk += g * g;
    }
    var se_s: f64 = 0;
    var se_k: f64 = 0;
    for (0..count) |i| {
        const t: f64 = td[i];
        const denom = @max(1.0, @abs(t));
        const es = (@as(f64, rd[i]) - t) / denom;
        const ek = (@as(f64, gd[i]) - t) / denom;
        se_s += es * es;
        se_k += ek * ek;
    }
    const n_f: f64 = @floatFromInt(count);
    const rms_stock = @sqrt(se_s / n_f);
    const rms_kern = @sqrt(se_k / n_f);
    const cos_stock = dot_s / (@sqrt(ns) * @sqrt(nt));
    const cos_kern = dot_k / (@sqrt(nk) * @sqrt(nt));

    if (env.get(.VQMM_PARITY_DEBUG) != null) {
        std.debug.print(
            "[vqmm-parity] {s:>16}: n={d} max {d:.5}/{d:.5} rms {d:.6}/{d:.6} (ratio {d:.3}) cos {d:.7}/{d:.7}\n",
            .{ label, count, stock_max, kern_max, rms_stock, rms_kern, if (rms_stock > 0) rms_kern / rms_stock else 0, cos_stock, cos_kern },
        );
    }

    if (verifyQmmParityVerdict(kern_max, rms_stock, rms_kern, cos_stock, cos_kern)) |why| {
        std.debug.print(
            "verifyQmm parity FAIL [{s}] ({s}): n={d} kernel max={d:.4} rms={d:.6} cos={d:.6}; " ++
                "stock max={d:.4} rms={d:.6} cos={d:.6}; bars: max<={d:.2} rms<={d:.1}x cos>=stock-{d}\n",
            .{
                label,                      @tagName(why),
                count,                      kern_max,
                rms_kern,                   cos_kern,
                stock_max,                  rms_stock,
                cos_stock,                  VerifyQmmParity.GROSS_CEILING,
                VerifyQmmParity.RMS_FACTOR, VerifyQmmParity.COS_SLACK,
            },
        );
        return error.TestExpectedApproxEq;
    }
}

test "verifyQmmParityVerdict: reduction-order noise passes, a gross or systematic error does not" {
    // Reduction-order noise: a worse worst-element than stock, cosine intact.
    // This is the M1 Pro measurement (kern_max 0.0313 while stock landed an
    // unusually good 0.0039 at cos 0.999998) and it must PASS with no
    // per-chip bar — the max is no longer judged against stock's at all.
    try testing.expect(verifyQmmParityVerdict(0.0313, 0.0009, 0.0013, 0.999999, 0.999998) == null);
    // M4 Max: every shipped lane measures ratio 1.00.
    try testing.expect(verifyQmmParityVerdict(0.0308, 0.0012, 0.0012, 0.999998, 0.999998) == null);
    // One accumulator corrupted: a single element, rms and cosine still fine
    // — the exact shape a max-slack widened to 0.04 would have waved through.
    try testing.expectEqual(
        VerifyQmmParityFail.gross_element,
        verifyQmmParityVerdict(12.0, 0.0012, 0.0013, 0.999999, 0.999997).?,
    );
    // Systematically noisier without a single gross element: rms is the bar
    // that sees it, and it is scale-free so an accurate stock cannot hide it.
    try testing.expectEqual(
        VerifyQmmParityFail.systematic,
        verifyQmmParityVerdict(0.4, 0.0009, 0.0040, 0.999999, 0.999995).?,
    );
    try testing.expectEqual(
        VerifyQmmParityFail.cosine,
        verifyQmmParityVerdict(0.005, 0.0012, 0.0012, 0.999999, 0.99990).?,
    );
    try testing.expectEqual(
        VerifyQmmParityFail.not_tracking,
        verifyQmmParityVerdict(0.005, 0.0012, 0.0012, 0.5, 0.4).?,
    );
    // A lane BETTER than stock is never a failure.
    try testing.expect(verifyQmmParityVerdict(0.0001, 0.0012, 0.0002, 0.999998, 0.999999) == null);
}

test "verifyQmm: the plain-SIMD tiles match stock qmm at 5/6/8-bit affine too" {
    // The split-K and msg tiles were 4-bit specializations by construction, so
    // every non-4-bit pack fell through to stock on the whole M1-M4 line — the
    // Qwen3.8-27B 6-bit pack (95.0% top-1 against bf16 vs the 4-bit pack's
    // 83.6%) paid the full lane-off penalty on every MTP round. The byte-
    // addressed unpack lifted from the NAX m16 tile fixes that; this pins it.
    //
    // Same bar as the 4-bit test: no worse than stock against an fp32 dequant
    // reference, never kernel-vs-kernel. Parity is asserted independently of
    // the PROFITABILITY predicate — `vqmm_mixed_plain_test_force` opens the
    // gate so a shape the measurement later declines is still proven correct.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xB175);
    const rnd = prng.random();

    verify_qmm.vqmm_mixed_plain_test_force = true;
    defer verify_qmm.vqmm_mixed_plain_test_force = null;
    // The NAX probe must not steal these rows on an M5 box: this test is about
    // the plain-SIMD lanes, and M 2..7 is below the NAX takeover anyway.
    verify_qmm_nax_vars.vqmm_nax_probe_override = false;
    defer verify_qmm_nax_vars.vqmm_nax_probe_override = null;

    const cases = [_]struct { k: c_int, n: c_int, gs: u32, label: []const u8 }{
        .{ .k = 5120, .n = 17408, .gs = 64, .label = "mlp" }, // MLP gate/up, split-K
        .{ .k = 2048, .n = 151936, .gs = 64, .label = "lm_head" }, // huge N, msg tile
        // The unpack is independent of the group size (gs only indexes the
        // scale/bias pair), but the forced-on A/B arm adopts every gs, so the
        // other two are proven rather than argued.
        .{ .k = 2048, .n = 5120, .gs = 32, .label = "gs32" },
        .{ .k = 2048, .n = 5120, .gs = 128, .label = "gs128" },
    };
    for ([_]u32{ 5, 6, 8 }) |bits| {
        for (cases) |cs| {
            const wb = try testRandWeightBf16(rnd, &[_]c_int{ cs.n, cs.k }, s);
            defer _ = mlx.mlx_array_free(wb);

            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
            var wq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            var wsc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wsc);
            var wbi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wbi);
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
            try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

            var wdq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq);
            try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 }, s));
            var wdq_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq_t);
            const t_axes = [_]c_int{ 1, 0 };
            try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

            for ([_]c_int{ 2, 4, 7 }) |m| {
                const xn: usize = @intCast(m * cs.k);
                const xbuf = try allocator.alloc(f32, xn);
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
                const xshape = [_]c_int{ 1, m, cs.k };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

                const got_opt = try verifyQmm(s, x, wq, wsc, wbi, bits, cs.gs);
                try testing.expect(got_opt != null);
                const got = got_opt.?;
                defer _ = mlx.mlx_array_free(got);
                const gsh = mlx.getShape(got);
                try testing.expectEqual(m, gsh[1]);
                try testing.expectEqual(cs.n, gsh[2]);
                try expectVerifyQmmNoWorseThanStock(s, x, wq, wsc, wbi, bits, cs.gs, wdq_t, got, cs.label);
            }

            // With the gate closed the same shape falls through to stock — the
            // profitability predicate is what decides adoption, and a lane that
            // cannot be declined is a lane that cannot be A/B'd.
            {
                verify_qmm.vqmm_mixed_plain_test_force = false;
                defer verify_qmm.vqmm_mixed_plain_test_force = true;
                const xbuf = try allocator.alloc(f32, @intCast(4 * cs.k));
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = 0.1;
                const xshape = [_]c_int{ 1, 4, cs.k };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
                try testing.expectEqual(@as(?mlx.mlx_array, null), try verifyQmm(s, x, wq, wsc, wbi, bits, cs.gs));
            }
        }
    }
}

test "mixedPlainShapeEnabled: adoption is what was MEASURED, per width and per shape" {
    const t = testing;
    // 4-bit is the original specialization and is always in.
    try t.expect(mixedPlainShapeEnabled(4, 64, 1024));

    // 6-bit is the measured win (1.10-1.22x on the Qwen3.8-27B 6-bit pack).
    try t.expect(mixedPlainShapeEnabled(6, 64, MIXED_PLAIN_MIN_N));
    try t.expect(mixedPlainShapeEnabled(6, 64, 17408));
    // Narrow projections (GDN ba, K/V) are where the byte-addressed unpack
    // loses to stock — the same shape-dependence mixedNaxShapeEnabled encodes,
    // and forcing them on gave back ~10 of the ~20 points.
    try t.expect(!mixedPlainShapeEnabled(6, 64, MIXED_PLAIN_MIN_N - 1));
    try t.expect(!mixedPlainShapeEnabled(6, 64, 1024));
    // 5-bit and gs 128 measured on the Flash Next oQ4e pack (z, out, o: N 2560/6144).
    for ([_]u32{ 5, 6 }) |b| for ([_]u32{ 64, 128 }) |g| {
        try t.expect(mixedPlainShapeEnabled(b, g, 2560));
        try t.expect(mixedPlainShapeEnabled(b, g, 6144));
    };
    try t.expect(!mixedPlainShapeEnabled(6, 32, 17408));

    // 8-bit HAS a correct kernel arm and a parity test, and is still default
    // OFF: measured 1.01-1.04x, inside what this harness resolves. One env var
    // from on — an unmeasured default is the thing this predicate exists to prevent.
    try t.expect(!mixedPlainShapeEnabled(8, 64, 17408));
    // A width with no unpack arm is never adopted, whatever the shape.
    for ([_]u32{ 2, 3, 7 }) |b| try t.expect(!mixedPlainShapeEnabled(b, 64, 17408));
}

test "verifyQmm: split-K + msg + NAX verify-width kernels match stock qmm (4-bit affine)" {
    // The spec-verify fast path: stock MLX qmm is tuned for M=1 (qmv) and
    // large M (steel); the 2..8-row verify shapes fall in a dead zone.
    // Port of MTPLX's split-K verify kernel family (verify_kernels.py,
    // Apache-2.0) — parity vs mlx_quantized_matmul at real 27B shapes.
    //
    // THE INVARIANT IS "NO LESS ACCURATE THAN STOCK", NOT "AGREES WITH STOCK".
    // Both paths accumulate in fp32 in a DIFFERENT ORDER and then round the
    // result to bf16, so the two outputs legitimately disagree by a few ULPs —
    // and on this data (K=5120 dot products of U(-0.5,0.5) → heavy cancellation)
    // each path already sits up to ~0.036 from the true value on its worst
    // element, purely from that bf16 output rounding. A direct kernel-vs-stock
    // bound therefore pins AGREEMENT tighter than bf16 permits: the original
    // `max_rel <= 0.02` / `cos > 0.99999` thresholds were tuned on one GPU and
    // the CI runner's GPU (different rounding, same correctness) failed them at
    // max_rel 0.0977 / cos 0.999988 — a green-here/red-there test that says
    // nothing about correctness. Measured: kernel-vs-truth equals stock-vs-truth
    // to four decimals on every shape.
    //
    // So both paths are measured against an fp32 DEQUANT REFERENCE (the same
    // 4-bit weights, dequantized, matmul'd in fp32 — the ground truth both are
    // approximating) and the kernel must be no worse than the stock kernel it
    // replaces. That is machine-independent, and a real defect (a partial-sum
    // race, a register spill, bad indexing) blows kernel-vs-truth far past
    // stock-vs-truth while a rounding-order difference cannot.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rnd = prng.random();

    const cases = [_]struct { k: c_int, n: c_int, gs: u32 }{
        .{ .k = 5120, .n = 17408, .gs = 64 }, // MLP gate/up
        .{ .k = 5120, .n = 1024, .gs = 32 }, // small-N above the gate, MTP gs-32 class
        .{ .k = 2048, .n = 5120, .gs = 128 }, // gs-128 arm
    };
    for (cases) |cs| {
        // Random dense weight → 4-bit affine triple.
        const wb = try testRandWeightBf16(rnd, &[_]c_int{ cs.n, cs.k }, s);
        defer _ = mlx.mlx_array_free(wb);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        var wsc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wsc);
        var wbi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wbi);
        try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

        // fp32 ground truth operand: the same 4-bit weights, dequantized.
        var wdq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq);
        try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 }, s));
        var wdq_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq_t);
        const t_axes = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

        // Kernel rows: split-K M 2..7 on every machine; the NAX m16 tile rows
        // (M 8..16, the padding path at 8/9/12 and the exact-16 tile) run only
        // where the M5-class probe is live (todo-m5-nax.md §9.3) — the probe
        // self-gates, so plain-SIMD machines skip them.
        const simd_ms = [_]c_int{ 2, 3, 4, 5, 6, 7 };
        // The M4 shader tile adds the 24-row tile (M 17..24).
        const nax_ms = [_]c_int{ 8, 9, 12, 16, 17, 24 };
        const tile_ms: usize = switch (verifyQmmTile()) {
            .off => 0,
            .nax => 4,
            .shader => nax_ms.len,
        };
        const total_ms: usize = simd_ms.len + tile_ms;
        var mi: usize = 0;
        while (mi < total_ms) : (mi += 1) {
            const m = if (mi < simd_ms.len) simd_ms[mi] else nax_ms[mi - simd_ms.len];
            const label: []const u8 = if (mi < simd_ms.len) "split-K" else "matmul2d tile";
            const xn: usize = @intCast(m * cs.k);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, cs.k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

            // Kernel path — must ENGAGE for every listed M, at EVERY column
            // tile the test seam can force; null is the shipped per-M default.
            for ([_]?c_int{null} ++ [_]?c_int{ 2, 4, 8 }) |bn_force| {
                if (bn_force) |bn| {
                    // The NAX tile ignores bn; only the split-K rows sweep.
                    if (mi >= simd_ms.len) continue;
                    if (@mod(cs.n, bn) != 0 or bn * m > VQMM_MAX_ACC) continue;
                }
                verify_qmm.vqmm_bn_test_override = bn_force;
                defer verify_qmm.vqmm_bn_test_override = null;
                const got_opt = try verifyQmm(s, x, wq, wsc, wbi, 4, cs.gs);
                try testing.expect(got_opt != null);
                const got = got_opt.?;
                defer _ = mlx.mlx_array_free(got);
                const gsh = mlx.getShape(got);
                try testing.expectEqual(@as(c_int, 1), gsh[0]);
                try testing.expectEqual(m, gsh[1]);
                try testing.expectEqual(cs.n, gsh[2]);

                try expectVerifyQmmNoWorseThanStock(s, x, wq, wsc, wbi, 4, cs.gs, wdq_t, got, label);
            }
        }

        // Ineligible widths fall through to stock (null). The NAX probe is
        // FORCED false so this pins the plain-SIMD (M1-M4) dispatch on every
        // machine, an M5 running the suite included: M=1 stays qmv, M 8..16
        // must NOT reach the split-K family (the measured spill cliff past 7
        // live row vectors), and M=17 is past every lane.
        {
            verify_qmm_nax_vars.vqmm_nax_probe_override = false;
            defer verify_qmm_nax_vars.vqmm_nax_probe_override = null;
            for ([_]c_int{ 1, 8, 16, 17 }) |bad_m| {
                const xn: usize = @intCast(bad_m * cs.k);
                const xbuf = try allocator.alloc(f32, xn);
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = 0.1;
                const xshape = [_]c_int{ 1, bad_m, cs.k };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
                try testing.expectEqual(@as(?mlx.mlx_array, null), try verifyQmm(s, x, wq, wsc, wbi, 4, cs.gs));
            }
        }

        // M=1 stays stock under the REAL probe too (MTPLX's tile nominally
        // covers M 1..16, but M=1 belongs to stock qmv on every machine).
        {
            const xbuf = try allocator.alloc(f32, @intCast(cs.k));
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = 0.1;
            const xshape = [_]c_int{ 1, 1, cs.k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
            try testing.expectEqual(@as(?mlx.mlx_array, null), try verifyQmm(s, x, wq, wsc, wbi, 4, cs.gs));
        }
    }

    // Mixed-precision trunk projections: the M5 NAX lane adopts affine
    // 5/6/8-bit weights on the measured wide shape class. Keep this self-gated
    // like the NAX rows above so non-G17 CI never attempts to compile an
    // unavailable MPP kernel.
    // Forced open: parity at g128 is proven apart from whether the shape is adopted.
    if (verifyQmmNaxEnabled()) {
        verify_qmm.vqmm_mixed_nax_test_force = true;
        defer verify_qmm.vqmm_mixed_nax_test_force = null;
        const k: c_int = 512;
        const n: c_int = 5120;
        for ([_]u32{ 64, 128 }) |gs| for ([_]u32{ 5, 6, 8 }) |bits| {
            const wb = try testRandWeightBf16(rnd, &[_]c_int{ n, k }, s);
            defer _ = mlx.mlx_array_free(wb);

            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(
                &triple,
                wb,
                mlx.mlx_optional_int.some(@intCast(gs)),
                mlx.mlx_optional_int.some(@intCast(bits)),
                "affine",
                .{},
                s,
            ));
            var wq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            var wsc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wsc);
            var wbi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wbi);
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
            try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

            var wdq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq);
            try mlx.check(mlx.mlx_dequantize(
                &wdq,
                wq,
                wsc,
                wbi,
                mlx.mlx_optional_int.some(@intCast(gs)),
                mlx.mlx_optional_int.some(@intCast(bits)),
                "affine",
                .{ .ctx = null },
                mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 },
                s,
            ));
            var wdq_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq_t);
            const t_axes = [_]c_int{ 1, 0 };
            try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

            const m: c_int = 8;
            const xn: usize = @intCast(m * k);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

            const got_opt = try verifyQmm(s, x, wq, wsc, wbi, bits, gs);
            try testing.expect(got_opt != null);
            const got = got_opt.?;
            defer _ = mlx.mlx_array_free(got);
            try expectVerifyQmmNoWorseThanStock(
                s,
                x,
                wq,
                wsc,
                wbi,
                bits,
                gs,
                wdq_t,
                got,
                switch (bits) {
                    5 => "nax oQe q5",
                    6 => "nax oQe q6",
                    else => "nax q8 candidate",
                },
            );

            // Below the NAX takeover row, default adoption remains the measured
            // width/shape surface: this q5/q6 gs64 N=5120 projection uses the
            // plain-SIMD lane, while q8 still falls through to stock.
            const x6buf = try allocator.alloc(f32, 6 * @as(usize, @intCast(k)));
            defer allocator.free(x6buf);
            for (x6buf) |*v| v.* = 0.1;
            const x6shape = [_]c_int{ 1, 6, k };
            const x6f = mlx.mlx_array_new_data(x6buf.ptr, &x6shape, 3, .float32);
            defer _ = mlx.mlx_array_free(x6f);
            var x6 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x6);
            try mlx.check(mlx.mlx_astype(&x6, x6f, .bfloat16, s));
            const got6_opt = try verifyQmm(s, x6, wq, wsc, wbi, bits, gs);
            if (bits != 8) {
                try testing.expect(got6_opt != null);
                const got6 = got6_opt.?;
                defer _ = mlx.mlx_array_free(got6);
                try expectVerifyQmmNoWorseThanStock(s, x6, wq, wsc, wbi, bits, gs, wdq_t, got6, "plain-SIMD q5/q6");
            } else {
                try testing.expectEqual(@as(?mlx.mlx_array, null), got6_opt);
            }
        };
    }

    // ── msg wide-tile parity (the huge-N/lm_head arm), called directly with
    // a RAGGED N (1000 % 32 != 0) so the in-kernel n0 guard is exercised. ──
    {
        const mk: c_int = 2048;
        const mn: c_int = 1000;
        const wb = try testRandWeightBf16(rnd, &[_]c_int{ mn, mk }, s);
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

        var wdq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq);
        try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 }, s));
        var wdq_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq_t);
        const t_axes = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

        var m: c_int = 2;
        while (m <= 6) : (m += 1) {
            const xn: usize = @intCast(m * mk);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, mk };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

            const xsh = mlx.getShape(x);
            const got = (try runVerifyQmmMsg(s, x, wq, wsc, wbi, 4, 64, m, mk, mn, .bfloat16, xsh)).?;
            defer _ = mlx.mlx_array_free(got);

            try expectVerifyQmmNoWorseThanStock(s, x, wq, wsc, wbi, 4, 64, wdq_t, got, "msg ragged-N");
        }
    }
}

test "vqmmLaneFor: NAX dispatch table (M 8..16 route to the m16 tile only when the probe is live)" {
    // Pure lane selection — todo-m5-nax.md §7. Hermetic on every machine:
    // no kernels are built, only the routing decision is pinned.
    const nax_off = false;
    const nax_on = true;

    // Plain-SIMD machine (probe false): the table is byte-identical to the
    // pre-NAX dispatch — M 8..16 fall through to stock.
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(1, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(2, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(7, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.msg, vqmmLaneFor(4, 5120, 151936, nax_off, 8));
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(16, 5120, 17408, nax_off, 8));
    // Geometry floors shared by every lane.
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(4, 5120, 256, nax_off, 8)); // N < 512
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(4, 100, 17408, nax_off, 8)); // K % 64 != 0
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(4, 5120, 17409, nax_off, 8)); // N % 4 != 0

    // M5 (probe true), default takeover width 8: M 2..7 KEEP the plain-SIMD
    // lanes (MTPLX's own dispatcher keeps SIMD through M=6 with NAX lit —
    // 16-row padding waste makes SIMD competitive at small M).
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(7, 5120, 17408, nax_on, 8));
    try testing.expectEqual(VqmmLane.msg, vqmmLaneFor(7, 5120, 151936, nax_on, 8));
    // M 8..16 route to the NAX tile when its stricter geometry holds…
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(8, 5120, 17408, nax_on, 8));
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(16, 5120, 17408, nax_on, 8));
    // …including the lm_head class natively (N=151936 % 32 == 0 — no msg
    // variant needed past M=7).
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(9, 5120, 151936, nax_on, 8));
    // Tile-ineligible geometry at M 8..16 stays stock: K % 256, N % 32.
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5184, 17408, nax_on, 8)); // 5184 % 256 == 64
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 17412, nax_on, 8)); // 17412 % 32 == 4
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(17, 5120, 17408, nax_on, 8)); // past every lane
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(1, 5120, 17408, nax_on, 8)); // M=1 stays qmv
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 480, nax_on, 8)); // tiny-N floor holds for NAX too

    // A lowered takeover width (5): M 5..7 route to NAX, M 4 keeps split-K, and the tile geometry is still required —
    // a lowered width never bypasses it, it falls back to the SIMD lanes.
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(5, 5120, 17408, nax_on, 5));
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(7, 5120, 151936, nax_on, 5));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(4, 5120, 17408, nax_on, 5));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(5, 5184, 17408, nax_on, 5)); // K % 256 fails, % 64 holds
}

test "verifyQmm lane table: the M4 shader tile serves M 8..24 in 8-row steps" {
    try testing.expectEqual(VqmmTile.shader, vqmmTileFrom(false, "applegpu_g16s", "26.5"));
    try testing.expectEqual(VqmmTile.off, vqmmTileFrom(false, "applegpu_g16s", "26.1"));
    try testing.expectEqual(VqmmTile.off, vqmmTileFrom(false, "applegpu_g15d", "26.5")); // unmeasured
    try testing.expectEqual(VqmmTile.off, vqmmTileFrom(false, "applegpu_g16p", "26.5")); // phone: unmeasured
    try testing.expectEqual(VqmmTile.nax, vqmmTileFrom(true, "applegpu_g17s", "26.5"));

    try testing.expectEqual(VqmmLane.splitk, vqmmLaneForTile(7, 5120, 17408, .shader, 8));
    for ([_]c_int{ 8, 12, 16, 17, 24 }) |m| try testing.expectEqual(VqmmLane.nax, vqmmLaneForTile(m, 5120, 17408, .shader, 8));
    try testing.expectEqual(VqmmLane.none, vqmmLaneForTile(25, 5120, 17408, .shader, 8));
    try testing.expectEqual(VqmmLane.none, vqmmLaneForTile(12, 5184, 17408, .shader, 8)); // K % 256
    // The M5 tile keeps its 16-row ceiling.
    try testing.expectEqual(VqmmLane.none, vqmmLaneForTile(17, 5120, 17408, .nax, 8));

    try testing.expectEqual(@as(c_int, 8), vqmmTileRows(.shader, 8));
    try testing.expectEqual(@as(c_int, 16), vqmmTileRows(.shader, 9));
    try testing.expectEqual(@as(c_int, 24), vqmmTileRows(.shader, 17));
    try testing.expectEqual(@as(c_int, 16), vqmmTileRows(.nax, 8));
}

test "mixedNaxShapeEnabled keeps measured q5/q6/q8 wins and rejects narrow regressions" {
    for ([_]c_int{ 8, 12 }) |m| {
        try testing.expect(mixedNaxShapeEnabled(4, 32, 1024, m));
        try testing.expect(mixedNaxShapeEnabled(5, 64, 5120, m));
        try testing.expect(mixedNaxShapeEnabled(6, 64, 5120, m));
        try testing.expect(mixedNaxShapeEnabled(8, 64, 5120, m));
        try testing.expect(!mixedNaxShapeEnabled(5, 64, 1024, m));
        try testing.expect(!mixedNaxShapeEnabled(6, 64, 1024, m));
        try testing.expect(!mixedNaxShapeEnabled(8, 64, 1024, m));
        try testing.expect(!mixedNaxShapeEnabled(5, 32, 5120, m));
        try testing.expect(!mixedNaxShapeEnabled(8, 32, 5120, m));
        // 8-bit at N 2560 (our Flash Next pack's o/out projections) loses at M 7-8.
        try testing.expect(!mixedNaxShapeEnabled(8, 64, 2560, m));
        try testing.expect(!mixedNaxShapeEnabled(8, 128, 5120, m));
        try testing.expect(!mixedNaxShapeEnabled(5, 128, 2559, m));
    }
    // Flash Next oQ4e's 5/6-bit g128 and 2560-wide projections: a loss at 8 verify rows, a win at 12.
    for ([_][3]c_int{ .{ 6, 128, 5120 }, .{ 5, 128, 2560 }, .{ 5, 64, 2560 } }) |c| {
        try testing.expect(!mixedNaxShapeEnabled(@intCast(c[0]), @intCast(c[1]), c[2], 8));
        try testing.expect(mixedNaxShapeEnabled(@intCast(c[0]), @intCast(c[1]), c[2], 9));
    }
}

test "oQ4e layer-role fingerprint pins every mixed override" {
    try testing.expectEqual(@as(usize, 30), std.mem.count(u8, OQE_MLP_DOWN_BITS, "5"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, OQE_MLP_DOWN_BITS, "6"));
    try testing.expectEqual(@as(usize, 11), std.mem.count(u8, OQE_LINEAR_Z_BITS, "5"));
    try testing.expectEqual(@as(usize, 20), std.mem.count(u8, OQE_LINEAR_AB_BITS, "5"));
    try testing.expectEqual(@as(usize, 48), std.mem.count(u8, OQE_LINEAR_OUT_BITS, "5"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, OQE_FULL_QK_BITS, "5"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, OQE_FULL_V_BITS, "6"));
    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, OQE_FULL_O_BITS, "5"));
    try testing.expectEqual(@as(u32, 5), oqeLayerBits(OQE_MLP_DOWN_BITS, 0));
    try testing.expectEqual(@as(u32, 4), oqeLayerBits(OQE_MLP_DOWN_BITS, 63));
}

test "verifyQmmNaxEnabledForMFrom mirrors flags, min-M, and dispatch geometry" {
    const K: c_int = 5120;
    const N: c_int = 151936;

    try testing.expect(verifyQmmNaxEnabledForMFrom(8, K, N, true, true, 8));
    try testing.expect(verifyQmmNaxEnabledForMFrom(9, K, N, true, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, K, N, false, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, K, N, true, false, 8));

    // The takeover knob is part of controller readiness, not just kernel
    // dispatch: min-M 9 leaves M=8 on stock even though M=9 is NAX.
    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, K, N, true, true, 9));
    try testing.expect(verifyQmmNaxEnabledForMFrom(9, K, N, true, true, 9));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(9, K, N, true, true, 10));
    try testing.expect(verifyQmmNaxEnabledForMFrom(8, K, N, true, true, 5));

    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, 5184, N, true, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, K, 151940, true, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, K, 480, true, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(1, K, N, true, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(17, K, N, true, true, 8));
}

test "verifyQmmNaxAvailable: false on every non-G17 device (kernel is never built here)" {
    // The real-device leg of the probe: on this M4 (and every CI runner) the
    // architecture is not applegpu_g17, so the probe must read false and the
    // NAX kernel object can never be constructed. On a real M5 the arch IS
    // G17 and this test only checks the plumbing returned something.
    var buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&buf) orelse {
        // No Metal device info at all: the probe must be false.
        try testing.expect(!verifyQmmNaxAvailable());
        return;
    };
    try testing.expect(arch.len > 0);
    var vbuf: [64]u8 = undefined;
    const ver = macosProductVersion(&vbuf) orelse "<none>";
    std.debug.print("[nax-probe] arch={s} macos={s} available={}\n", .{ arch, ver, verifyQmmNaxAvailable() });
    if (!naxArchIsG17(arch)) {
        try testing.expect(!verifyQmmNaxAvailable());
    }
}

test "naxMinMForBits: q8 takes over at its measured M7, the rest at M8" {
    try testing.expectEqual(@as(c_int, 8), naxMinMForBits(4));
    try testing.expectEqual(@as(c_int, 8), naxMinMForBits(6));
    try testing.expectEqual(@as(c_int, 7), naxMinMForBits(8));
}

test "NAX host scaffolding: zero-pad to 16 rows + slice-back are exact (runs off-M5)" {
    // The half of runVerifyQmmNax that CAN execute without G17 hardware —
    // the exact production helpers (naxPadRows / naxSliceRows) wrapped
    // around STOCK qmm standing in for the NAX kernel. Pins the mlx_pad
    // axis/value plumbing (pad rows are EXACT zeros in the activation
    // dtype, real rows byte-preserved), the slice bounds, and that zero
    // activation rows produce exactly-zero output rows (so on the M5 the
    // padded tile positions can never bleed into the sliced result).
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA110C);
    const rnd = prng.random();
    const K: c_int = 512;
    const N: c_int = 640; // % 32 == 0, NAX-shaped
    const gs: u32 = 64;

    const wb = try testRandWeightBf16(rnd, &[_]c_int{ N, K }, s);
    defer _ = mlx.mlx_array_free(wb);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

    for ([_]c_int{ 9, 16 }) |m| {
        const xn: usize = @intCast(m * K);
        const xbuf = try allocator.alloc(f32, xn);
        defer allocator.free(xbuf);
        for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
        const xshape = [_]c_int{ 1, m, K };
        const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

        // Pad half: [1, m, K] → [16, K]; real rows byte-preserved, pad rows 0.
        const x16 = try naxPadRows(s, x, m, 16, K, .bfloat16);
        defer _ = mlx.mlx_array_free(x16);
        const psh = mlx.getShape(x16);
        try testing.expectEqual(@as(usize, 2), psh.len);
        try testing.expectEqual(@as(c_int, 16), psh[0]);
        try testing.expectEqual(K, psh[1]);
        {
            var xf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xf);
            var pf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pf);
            try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
            try mlx.check(mlx.mlx_astype(&pf, x16, .float32, s));
            try mlx.check(mlx.mlx_array_eval(xf));
            try mlx.check(mlx.mlx_array_eval(pf));
            const xd_ = mlx.mlx_array_data_float32(xf).?;
            const pd_ = mlx.mlx_array_data_float32(pf).?;
            for (0..xn) |i| try testing.expectEqual(xd_[i], pd_[i]);
            for (xn..@intCast(16 * K)) |i| try testing.expectEqual(@as(f32, 0.0), pd_[i]);
        }

        // Slice half around stock qmm: zero rows in → exactly-zero rows out,
        // and the slice returns the first m rows byte-identically.
        var y16 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y16);
        try mlx.check(mlx.mlx_quantized_matmul(&y16, x16, wq, wsc, wbi, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(4), "affine", s));
        const ym = try naxSliceRows(s, y16, m, 16, N);
        defer _ = mlx.mlx_array_free(ym);
        const ysh = mlx.getShape(ym);
        try testing.expectEqual(@as(usize, 2), ysh.len);
        try testing.expectEqual(m, ysh[0]);
        try testing.expectEqual(N, ysh[1]);
        {
            var ff = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ff);
            var mf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(mf);
            try mlx.check(mlx.mlx_astype(&ff, y16, .float32, s));
            try mlx.check(mlx.mlx_astype(&mf, ym, .float32, s));
            try mlx.check(mlx.mlx_array_eval(ff));
            try mlx.check(mlx.mlx_array_eval(mf));
            const fd = mlx.mlx_array_data_float32(ff).?;
            const md = mlx.mlx_array_data_float32(mf).?;
            const mn: usize = @intCast(m * N);
            for (0..mn) |i| try testing.expectEqual(fd[i], md[i]);
            for (mn..@intCast(16 * N)) |i| try testing.expectEqual(@as(f32, 0.0), @abs(fd[i]));
        }
    }
}

test "verifyQmm µbench: kernel vs stock per 27B shape (diag vqmm-ubench)" {
    if (env.get(.VQMM_UBENCH) == null) return error.SkipZigTest;
    const io_util = @import("../../util/io_util.zig");
    const tio = testing.io;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xBEEF);
    const rnd = prng.random();
    const WARM = 2;
    const ITERS = 8;
    // One eval per BATCH of independent launches, never per launch: an eval
    // barrier costs ~0.14 ms of sync on this box, which is more than the whole
    // `out` projection and made every small shape a sync measurement. A real
    // verify forward pipelines its dispatches, so the bench must too.
    const BATCH = 12;

    // Qwen3.8-27B 4-bit g64 inventory: hidden 5120, inter 17408, hd 256,
    // 24/4 heads with an output gate, 64 layers = 48 GDN + 16 full attention,
    // vocab 248320. `per_fwd` is how many times the shape runs in ONE forward,
    // so the totals below are a whole-model qmm estimate at each width.
    const shapes = [_]struct { name: []const u8, k: c_int, n: c_int, per_fwd: f64 }{
        .{ .name = "gdn_in", .k = 5120, .n = 16384, .per_fwd = 48 },
        .{ .name = "attn_q", .k = 5120, .n = 12288, .per_fwd = 16 },
        .{ .name = "attn_kv", .k = 5120, .n = 1024, .per_fwd = 32 },
        .{ .name = "gate/up", .k = 5120, .n = 34816, .per_fwd = 64 },
        .{ .name = "down", .k = 17408, .n = 5120, .per_fwd = 64 },
        .{ .name = "out", .k = 6144, .n = 5120, .per_fwd = 64 },
        .{ .name = "lm_head", .k = 5120, .n = 248320, .per_fwd = 1 },
    };
    const widths = [_]c_int{ 1, 4, 8, 12, 16, 24, 32 };
    var stock_tot: [widths.len]f64 = @splat(0);
    var kern_tot: [widths.len]f64 = @splat(0);
    var flops: [widths.len]f64 = @splat(0);

    std.debug.print("\n[vqmm-ubench] {s:>8} {s:>3} {s:>10} {s:>10} {s:>8} {s:>10}\n", .{ "shape", "M", "stock_ms", "kernel_ms", "ratio", "kern_TF/s" });
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

            var stock_ms: f64 = 0;
            {
                var it: usize = 0;
                var sw = io_util.Stopwatch.init(tio);
                while (it < WARM + ITERS) : (it += 1) {
                    if (it == WARM) sw.reset();
                    var outs: [BATCH]mlx.mlx_array = undefined;
                    for (&outs, &xs) |*o, xi| {
                        o.* = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_quantized_matmul(o, xi, wq, wsc, wbi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", s));
                    }
                    const vec = mlx.mlx_vector_array_new_data(&outs, outs.len);
                    try mlx.check(mlx.mlx_eval(vec));
                    _ = mlx.mlx_vector_array_free(vec);
                    for (outs) |o| _ = mlx.mlx_array_free(o);
                }
                stock_ms = @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS * BATCH) / 1e6;
            }
            var kern_ms: f64 = 0;
            var engaged = true;
            {
                var it: usize = 0;
                var sw = io_util.Stopwatch.init(tio);
                while (it < WARM + ITERS) : (it += 1) {
                    if (it == WARM) sw.reset();
                    var outs: [BATCH]mlx.mlx_array = undefined;
                    var got: usize = 0;
                    for (&outs, &xs) |*o, xi| {
                        o.* = (try verifyQmm(s, xi, wq, wsc, wbi, 4, 64)) orelse break;
                        got += 1;
                    }
                    if (got < BATCH) {
                        for (outs[0..got]) |o| _ = mlx.mlx_array_free(o);
                        engaged = false;
                        break;
                    }
                    const vec = mlx.mlx_vector_array_new_data(&outs, outs.len);
                    try mlx.check(mlx.mlx_eval(vec));
                    _ = mlx.mlx_vector_array_free(vec);
                    for (outs) |o| _ = mlx.mlx_array_free(o);
                }
                kern_ms = @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS * BATCH) / 1e6;
            }
            const shape_flops = 2.0 * @as(f64, @floatFromInt(sh.k)) * @as(f64, @floatFromInt(sh.n)) * @as(f64, @floatFromInt(m));
            stock_tot[wi] += stock_ms * sh.per_fwd;
            flops[wi] += shape_flops * sh.per_fwd;
            if (engaged) {
                kern_tot[wi] += kern_ms * sh.per_fwd;
                std.debug.print("[vqmm-ubench] {s:>8} {d:>3} {d:>10.3} {d:>10.3} {d:>8.2} {d:>10.2}\n", .{ sh.name, m, stock_ms, kern_ms, kern_ms / stock_ms, shape_flops / kern_ms / 1e9 });
            } else {
                kern_tot[wi] += stock_ms * sh.per_fwd;
                std.debug.print("[vqmm-ubench] {s:>8} {d:>3} {d:>10.3} {s:>10} {s:>8} {d:>10.2}\n", .{ sh.name, m, stock_ms, "fallback", "-", shape_flops / stock_ms / 1e9 });
            }
        }
    }
    std.debug.print("\n[vqmm-ubench] whole-forward qmm estimate (64 layers + lm_head), served lane\n", .{});
    std.debug.print("[vqmm-ubench] {s:>3} {s:>10} {s:>10} {s:>9} {s:>10} {s:>12}\n", .{ "M", "stock_ms", "kern_ms", "vs_M1", "TFLOP/s", "marg/row_ms" });
    for (widths, 0..) |m, wi| {
        const marg = if (m == 1) 0.0 else (kern_tot[wi] - kern_tot[0]) / @as(f64, @floatFromInt(m - 1));
        std.debug.print("[vqmm-ubench] {d:>3} {d:>10.2} {d:>10.2} {d:>8.2}x {d:>10.2} {d:>12.3}\n", .{
            m, stock_tot[wi], kern_tot[wi], kern_tot[wi] / kern_tot[0], flops[wi] / kern_tot[wi] / 1e9, marg,
        });
    }
}

test "verifyQmm Flash Next shapes µbench: stock vs plain vs NAX per width (diag vqmm-fn-ubench)" {
    if (env.get(.VQMM_FN_UBENCH) == null) return error.SkipZigTest;
    const io_util = @import("../../util/io_util.zig");
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xF1A5);
    const rnd = prng.random();
    const BATCH = 12;
    const ITERS = 40;
    verify_qmm.vqmm_mixed_plain_test_force = true;
    defer verify_qmm.vqmm_mixed_plain_test_force = null;
    verify_qmm.vqmm_mixed_nax_test_force = true;
    defer verify_qmm.vqmm_mixed_nax_test_force = null;
    // Per-forward counts: Jundot oQ4e (mixed) and our iQ pack (non-expert 8-bit g64).
    const shapes = [_]struct { name: []const u8, bits: u32, gs: u32, k: c_int, n: c_int, per: f64 }{
        .{ .name = "J gdn_z", .bits = 5, .gs = 128, .k = 2560, .n = 6144, .per = 25 },
        .{ .name = "J gdn_out", .bits = 5, .gs = 128, .k = 6144, .n = 2560, .per = 36 },
        .{ .name = "J attn_o5", .bits = 5, .gs = 64, .k = 6144, .n = 2560, .per = 2 },
        .{ .name = "J attn_o6", .bits = 6, .gs = 64, .k = 6144, .n = 2560, .per = 3 },
        .{ .name = "J sh_gu", .bits = 8, .gs = 128, .k = 2560, .n = 640, .per = 96 },
        .{ .name = "J sh_down", .bits = 8, .gs = 128, .k = 640, .n = 2560, .per = 48 },
        .{ .name = "O gdn_qkv", .bits = 8, .gs = 64, .k = 2560, .n = 10240, .per = 36 },
        .{ .name = "O gdn_z", .bits = 8, .gs = 64, .k = 2560, .n = 6144, .per = 36 },
        .{ .name = "O gdn_out", .bits = 8, .gs = 64, .k = 6144, .n = 2560, .per = 36 },
        .{ .name = "O attn_q", .bits = 8, .gs = 64, .k = 2560, .n = 12288, .per = 12 },
        .{ .name = "O attn_o", .bits = 8, .gs = 64, .k = 6144, .n = 2560, .per = 12 },
        .{ .name = "O sh_gu", .bits = 8, .gs = 64, .k = 2560, .n = 640, .per = 96 },
        .{ .name = "O sh_down", .bits = 8, .gs = 64, .k = 640, .n = 2560, .per = 48 },
    };
    const widths = [_]c_int{ 3, 5, 7, 8, 12 };
    std.debug.print("\n[vqmm-fn] {s:>10} {s:>5} {s:>3} {s:>9} {s:>9} {s:>6} {s:>9}\n", .{ "shape", "q", "M", "stock_us", "lane_us", "ratio", "saved/fwd" });
    for (shapes) |sh| {
        const wb = try testRandWeightBf16(rnd, &[_]c_int{ sh.n, sh.k }, s);
        defer _ = mlx.mlx_array_free(wb);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(sh.gs)), mlx.mlx_optional_int.some(@intCast(sh.bits)), "affine", .{}, s));
        var q: [3]mlx.mlx_array = undefined;
        for (&q, 0..) |*a, i| {
            a.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(a, triple, i));
            try mlx.check(mlx.mlx_array_eval(a.*));
        }
        defer for (q) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (widths) |m| {
            var xs: [BATCH]mlx.mlx_array = undefined;
            for (&xs) |*a| {
                a.* = try testRandWeightBf16(rnd, &[_]c_int{ 1, m, sh.k }, s);
                try mlx.check(mlx.mlx_array_eval(a.*));
            }
            defer for (xs) |a| {
                _ = mlx.mlx_array_free(a);
            };
            var ns: [2]u64 = .{ 0, 0 };
            var lane_ok = true;
            for (0..ITERS + 2) |it| for ([_]usize{ it % 2, 1 - it % 2 }) |arm| {
                var outs: [BATCH]mlx.mlx_array = undefined;
                var got: usize = 0;
                var sw = io_util.Stopwatch.init(testing.io);
                for (&outs, xs) |*o, x| {
                    if (arm == 0) {
                        o.* = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_quantized_matmul(o, x, q[0], q[1], q[2], true, mlx.mlx_optional_int.some(@intCast(sh.gs)), mlx.mlx_optional_int.some(@intCast(sh.bits)), "affine", s));
                    } else o.* = (try verifyQmm(s, x, q[0], q[1], q[2], sh.bits, sh.gs)) orelse break;
                    got += 1;
                }
                if (got == BATCH) {
                    const vec = mlx.mlx_vector_array_new_data(&outs, BATCH);
                    try mlx.check(mlx.mlx_eval(vec));
                    _ = mlx.mlx_vector_array_free(vec);
                } else lane_ok = false;
                if (it >= 2) ns[arm] += sw.read();
                for (outs[0..got]) |o| _ = mlx.mlx_array_free(o);
            };
            const per_call = 1e3 * @as(f64, ITERS * BATCH);
            const st = @as(f64, @floatFromInt(ns[0])) / per_call;
            const ln = @as(f64, @floatFromInt(ns[1])) / per_call;
            if (lane_ok) {
                std.debug.print("[vqmm-fn] {s:>10} {d}/{d:>3} {d:>3} {d:>9.1} {d:>9.1} {d:>6.2} {d:>9.1}\n", .{ sh.name, sh.bits, sh.gs, m, st, ln, ln / st, (st - ln) * sh.per });
            } else std.debug.print("[vqmm-fn] {s:>10} {d}/{d:>3} {d:>3} {d:>9.1} {s:>9}\n", .{ sh.name, sh.bits, sh.gs, m, st, "declined" });
        }
    }
}

test "verifyQmm mixed-width NAX µbench (diag vqmm-mixed-ubench)" {
    if (env.get(.VQMM_MIXED_UBENCH) == null) return error.SkipZigTest;
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    const io_util = @import("../../util/io_util.zig");
    const tio = testing.io;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x0C4E);
    const rnd = prng.random();
    const WARM = 10;
    const ITERS = 80;
    const M: c_int = 8;

    const shapes = [_]struct { name: []const u8, bits: u32, k: c_int, n: c_int }{
        .{ .name = "q4-gate", .bits = 4, .k = 5120, .n = 17408 },
        .{ .name = "q5-in_z", .bits = 5, .k = 5120, .n = 6144 },
        .{ .name = "q5-qproj", .bits = 5, .k = 5120, .n = 12288 },
        .{ .name = "q5-kproj", .bits = 5, .k = 5120, .n = 1024 },
        .{ .name = "q5-out", .bits = 5, .k = 6144, .n = 5120 },
        .{ .name = "q5-down", .bits = 5, .k = 17408, .n = 5120 },
        .{ .name = "q6-v", .bits = 6, .k = 5120, .n = 1024 },
        .{ .name = "q6-down", .bits = 6, .k = 17408, .n = 5120 },
    };
    std.debug.print("\n[vqmm-mixed] {s:>8} {s:>4} {s:>10} {s:>10} {s:>8}\n", .{
        "shape", "bits", "stock_ms", "nax_ms", "ratio",
    });
    for (shapes) |sh| {
        const wb = try testRandWeightBf16(rnd, &[_]c_int{ sh.n, sh.k }, s);
        defer _ = mlx.mlx_array_free(wb);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(
            &triple,
            wb,
            mlx.mlx_optional_int.some(64),
            mlx.mlx_optional_int.some(@intCast(sh.bits)),
            "affine",
            .{},
            s,
        ));
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

        const xn: usize = @intCast(M * sh.k);
        const xbuf = try allocator.alloc(f32, xn);
        defer allocator.free(xbuf);
        for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
        const xshape = [_]c_int{ 1, M, sh.k };
        const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(x));

        // Interleave the two paths (and reverse their order every iteration)
        // so GPU frequency/thermal drift cannot systematically favor the path
        // measured second.
        var it: usize = 0;
        while (it < WARM) : (it += 1) {
            var stock = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_quantized_matmul(
                &stock,
                x,
                wq,
                wsc,
                wbi,
                true,
                mlx.mlx_optional_int.some(64),
                mlx.mlx_optional_int.some(@intCast(sh.bits)),
                "affine",
                s,
            ));
            try mlx.check(mlx.mlx_array_eval(stock));
            _ = mlx.mlx_array_free(stock);
            const nax = (try verifyQmm(s, x, wq, wsc, wbi, sh.bits, 64)) orelse
                return error.TestUnexpectedResult;
            try mlx.check(mlx.mlx_array_eval(nax));
            _ = mlx.mlx_array_free(nax);
        }
        var stock_ns: u64 = 0;
        var nax_ns: u64 = 0;
        it = 0;
        while (it < ITERS) : (it += 1) {
            if (it % 2 == 0) {
                var sw = io_util.Stopwatch.init(tio);
                var stock = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_quantized_matmul(
                    &stock,
                    x,
                    wq,
                    wsc,
                    wbi,
                    true,
                    mlx.mlx_optional_int.some(64),
                    mlx.mlx_optional_int.some(@intCast(sh.bits)),
                    "affine",
                    s,
                ));
                try mlx.check(mlx.mlx_array_eval(stock));
                _ = mlx.mlx_array_free(stock);
                stock_ns += sw.read();
                sw.reset();
                const nax = (try verifyQmm(s, x, wq, wsc, wbi, sh.bits, 64)) orelse
                    return error.TestUnexpectedResult;
                try mlx.check(mlx.mlx_array_eval(nax));
                _ = mlx.mlx_array_free(nax);
                nax_ns += sw.read();
            } else {
                var sw = io_util.Stopwatch.init(tio);
                const nax = (try verifyQmm(s, x, wq, wsc, wbi, sh.bits, 64)) orelse
                    return error.TestUnexpectedResult;
                try mlx.check(mlx.mlx_array_eval(nax));
                _ = mlx.mlx_array_free(nax);
                nax_ns += sw.read();
                sw.reset();
                var stock = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_quantized_matmul(
                    &stock,
                    x,
                    wq,
                    wsc,
                    wbi,
                    true,
                    mlx.mlx_optional_int.some(64),
                    mlx.mlx_optional_int.some(@intCast(sh.bits)),
                    "affine",
                    s,
                ));
                try mlx.check(mlx.mlx_array_eval(stock));
                _ = mlx.mlx_array_free(stock);
                stock_ns += sw.read();
            }
        }
        const stock_ms = @as(f64, @floatFromInt(stock_ns)) / @as(f64, ITERS) / 1e6;
        const nax_ms = @as(f64, @floatFromInt(nax_ns)) / @as(f64, ITERS) / 1e6;
        std.debug.print("[vqmm-mixed] {s:>8} {d:>4} {d:>10.3} {d:>10.3} {d:>8.2}\n", .{
            sh.name, sh.bits, stock_ms, nax_ms, nax_ms / stock_ms,
        });
    }
}
