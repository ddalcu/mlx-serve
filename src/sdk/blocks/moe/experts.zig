//! sdk/blocks/moe/experts.
//! Owns gatherExpertMm.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const mlx_gguf = @import("../../quant/gguf/root.zig");
const gather_qmm_nax = @import("gather_qmm_nax.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const testing = std.testing;
const splitPackedGateUp = @import("../../../mlx/array.zig").splitPackedGateUp;
const transposeBf16Weight = @import("../../../mlx/array.zig").transposeBf16Weight;
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const verifyQmmNaxAvailable = @import("../linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;

/// Gathered matmul for MoE expert dispatch — handles both quantized and dense bf16.
/// Quantized (sc.ctx != null): mlx_gather_qmm with transpose_w=true, w stored [E,out,in].
/// Dense bf16 (sc.ctx == null): mlx_gather_mm; w was pre-transposed to [E,in,out] at load
/// (maybeTransposeForBf16 + generalized transposeBf16Weight), so x @ w is correct with no
/// transpose flag — mirrors mlx-lm's `gather_mm(x, weight.swapaxes(-1,-2))`.
pub fn gatherExpertMm(res: *mlx.mlx_array, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, lhs_idx: mlx.mlx_array, rhs_idx: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode, sorted: bool, s: mlx.mlx_stream) !void {
    if (mlx_gguf.kernels.infoOf(w, sc)) |info| {
        const y = try mlx_gguf.kernels.gatherLinear(info, x, w, lhs_idx, rhs_idx, s);
        _ = mlx.mlx_array_free(res.*);
        res.* = y;
        return;
    }
    if (sc.ctx == null) {
        // mlx 0.31.2's `mlx_gather_mm` returns WRONG results with sorted_indices=true
        // for the dense (non-quantized) path — the quantized `mlx_gather_qmm` honors
        // the flag correctly, but the dense kernel does not. The sorted-indices flag is
        // only a performance hint (rhs_indices ARE sorted here), so forcing false is
        // always numerically correct; it just forgoes the sorted-stream optimization.
        // Without this, dense bf16 MoE prefill (the sorted path, S>1) produced fluent
        // but semantically-wrong output (e.g. Qwen3.6-35B-A3B-bf16 calling clean prompts
        // "jumbled"). Verified byte-identical to mlx-lm once forced false. `sorted` is
        // still honored by the quantized branch below, where gather_qmm handles it
        // correctly.
        try mlx.check(mlx.mlx_gather_mm(res, x, w, lhs_idx, rhs_idx, false, s));
    } else {
        if (sorted and lhs_idx.ctx == null and (mode == .affine or mode.isFp4())) {
            if (try gather_qmm_nax.sortedGather(x, w, sc, bi, rhs_idx, bits, group_size, mode.isFp4(), verifyQmmNaxAvailable(), s)) |out| {
                _ = mlx.mlx_array_free(res.*);
                res.* = out;
                return;
            }
        }
        const prefill_cuda = @import("moe_prefill_cuda.zig");
        const cuda_fmt: ?prefill_cuda.Format = switch (mode) {
            .nvfp4 => .nvfp4,
            .mxfp8 => .mxfp8,
            else => null,
        };
        if (sorted and lhs_idx.ctx == null and bi.ctx == null) if (cuda_fmt) |fmt| {
            if (try prefill_cuda.sortedGather(s, x, w, sc, rhs_idx, fmt)) |out| {
                _ = mlx.mlx_array_free(res.*);
                res.* = out;
                return;
            }
        };
        try mlx.check(mlx.mlx_gather_qmm(res, x, w, sc, bi, lhs_idx, rhs_idx, true, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), sorted, s));
    }
}

// ── tests ──

test "gatherExpertMm: a sorted nvfp4 gather past 65,535 rows matches the dequantized experts" {
    // A Flash Next chunk of 8192 rows routes 81,920 rows; MLX's CUDA gather put one block per row on
    // grid z (at most 65,535) and the launch failed. Bar: bf16 rounding of an fp32 dequant reference.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var sc = @import("../../../mlx/scope.zig").Scope.init(testing.allocator, s);
    defer sc.deinit();
    var prng = std.Random.DefaultPrng.init(0x65536);
    const E: c_int = 4;
    const N: c_int = 128; // qmm_sm80's tile: narrower falls to MLX's naive gather, same limit
    const K: c_int = 64;
    const R: c_int = 65_600;
    const wb = try testRandWeightBf16(prng.random(), &[_]c_int{ E, N, K }, s);
    defer _ = mlx.mlx_array_free(wb);
    var pair = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(pair);
    try mlx.check(mlx.mlx_quantize(&pair, wb, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", .{}, s));
    var wq = mlx.mlx_array_new();
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    defer _ = mlx.mlx_array_free(wsc);
    try mlx.check(mlx.mlx_vector_array_get(&wq, pair, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, pair, 1));
    var deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deq);
    try mlx.check(mlx.mlx_dequantize(&deq, wq, wsc, .{ .ctx = null }, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));

    const xs = try testRandWeightBf16(prng.random(), &[_]c_int{ R, 1, K }, s);
    defer _ = mlx.mlx_array_free(xs);
    const ids = try testing.allocator.alloc(i32, @intCast(R));
    defer testing.allocator.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(i * @as(usize, @intCast(E)) / @as(usize, @intCast(R)));
    const rhs = try sc.astype(try sc.fromI32(ids, &.{R}), .uint32);

    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try gatherExpertMm(&out, xs, wq, wsc, .{ .ctx = null }, .{ .ctx = null }, rhs, 4, 16, .nvfp4, true, s);

    // Reference: each expert's contiguous rows against its fp32 weights.
    const x2 = try sc.astype(try sc.reshape(xs, &.{ R, K }), .float32);
    var parts: [4]mlx.mlx_array = undefined;
    for (0..4) |e| {
        const lo: c_int = @intCast(e * @as(usize, @intCast(R)) / 4);
        const hi: c_int = @intCast((e + 1) * @as(usize, @intCast(R)) / 4);
        const w_e = try sc.reshape(try sc.slice(deq, 0, @intCast(e), @intCast(e + 1)), &.{ N, K });
        parts[e] = try sc.matmul(try sc.slice(x2, 0, lo, hi), try sc.transpose(w_e, &.{ 1, 0 }));
    }
    const ref = try sc.concat(&parts, 0);
    const got = try sc.astype(try sc.reshape(out, &.{ R, N }), .float32);
    const d = try sc.sub(got, ref);
    const err = try sc.sumAxis(try sc.sumAxis(try sc.mul(d, d), 1), 0);
    const nrm = try sc.sumAxis(try sc.sumAxis(try sc.mul(ref, ref), 1), 0);
    const both = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ err, nrm }, 2);
    defer _ = mlx.mlx_vector_array_free(both);
    try mlx.check(mlx.mlx_eval(both));
    var e2: f32 = 0;
    var n2: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&e2, err));
    try mlx.check(mlx.mlx_array_item_float32(&n2, nrm));
    try testing.expect(n2 > 0 and @sqrt(e2 / n2) < 0.01);
}

test "splitPackedGateUp slices a packed experts.gate_up_proj into gate/up halves" {
    // A packed bank holds each expert's gate and up projections in one
    // tensor [E, 2*M, X] with gate rows first (HF chunks the gate_up output
    // in halves: gate = [..., :M], up = [..., M:]). The split must slice
    // axis 1 so the same code serves the packed quantized weight, its
    // scales, and its biases (all share the [E, rows, X] layout).
    const s = mlx.gpuStream();
    const E = 2;
    const M = 2; // moe_intermediate rows per half
    const X = 3;

    var host: [E * 2 * M * X]f32 = undefined;
    for (&host, 0..) |*v, i| v.* = @floatFromInt(i);
    const sh = [_]c_int{ E, 2 * M, X };
    const packed_arr = mlx.mlx_array_new_data(&host, &sh, 3, .float32);
    defer _ = mlx.mlx_array_free(packed_arr);

    const pair = try splitPackedGateUp(packed_arr, s);
    defer _ = mlx.mlx_array_free(pair.gate);
    defer _ = mlx.mlx_array_free(pair.up);

    const gshape = mlx.getShape(pair.gate);
    const ushape = mlx.getShape(pair.up);
    try testing.expectEqualSlices(c_int, &[_]c_int{ E, M, X }, gshape);
    try testing.expectEqualSlices(c_int, &[_]c_int{ E, M, X }, ushape);

    // Expert e's gate = rows [0, M) of its 2M block; up = rows [M, 2M).
    var gate_host: [E * M * X]f32 = undefined;
    var up_host: [E * M * X]f32 = undefined;
    try testReadF32(pair.gate, &gate_host, s);
    try testReadF32(pair.up, &up_host, s);
    for (0..E) |e| {
        for (0..M) |r| {
            for (0..X) |c| {
                const out_idx = (e * M + r) * X + c;
                const gate_src: f32 = @floatFromInt((e * 2 * M + r) * X + c);
                const up_src: f32 = @floatFromInt((e * 2 * M + M + r) * X + c);
                try testing.expectApproxEqAbs(gate_src, gate_host[out_idx], 0.001);
                try testing.expectApproxEqAbs(up_src, up_host[out_idx], 0.001);
            }
        }
    }
}

test "splitPackedGateUp halves feed gather_qmm correctly (lazy-slice zeros regression)" {
    // CLASS-BUG regression: mlx_slice produces a lazy VIEW with parent
    // strides; feeding such a view to mlx_gather_qmm silently produced
    // all-ZERO expert outputs (gather_qmm assumes row-contiguous weight
    // buffers). Reading the slice via data pointers materializes it
    // correctly, so a value-equality test on the split alone cannot catch
    // this — the assertion must go THROUGH gather_qmm. splitPackedGateUp
    // now materializes contiguous halves; this test pins that by comparing
    // gather_qmm on the split gate half against gather_qmm on an
    // independently-built contiguous copy of the same rows.
    const s = mlx.gpuStream();
    // Geometry matters: small/toy shapes take a strided-tolerant kernel path
    // and do NOT reproduce the zeros — keep dims near the real checkpoint's
    // 4-bit/gs64 layout (full scale is E=128, M=704, IN=2816).
    const E = 16;
    const M = 128; // rows per half
    const IN = 512;
    const gs = 64;
    const bits = 4;

    // Dense [E, 2M, IN] bf16 source, then quantize.
    const w_host = try std.testing.allocator.alloc(f32, E * 2 * M * IN);
    defer std.testing.allocator.free(w_host);
    for (w_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8))) * 0.07;
    const wsh = [_]c_int{ E, 2 * M, IN };
    const w_f32 = mlx.mlx_array_new_data(w_host.ptr, &wsh, 3, .float32);
    defer _ = mlx.mlx_array_free(w_f32);
    var w_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_bf16);
    try mlx.check(mlx.mlx_astype(&w_bf16, w_f32, .bfloat16, s));

    var qvec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_quantize(&qvec, w_bf16, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, s));
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&q, qvec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, qvec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, qvec, 2));

    const q_pair = try splitPackedGateUp(q, s);
    defer _ = mlx.mlx_array_free(q_pair.gate);
    defer _ = mlx.mlx_array_free(q_pair.up);
    const s_pair = try splitPackedGateUp(sc, s);
    defer _ = mlx.mlx_array_free(s_pair.gate);
    defer _ = mlx.mlx_array_free(s_pair.up);
    const b_pair = try splitPackedGateUp(bi, s);
    defer _ = mlx.mlx_array_free(b_pair.gate);
    defer _ = mlx.mlx_array_free(b_pair.up);

    // Independent contiguous ground truth: quantize the gate rows directly.
    const g_host = try std.testing.allocator.alloc(f32, E * M * IN);
    defer std.testing.allocator.free(g_host);
    for (0..E) |e| {
        for (0..M) |r| {
            for (0..IN) |c| {
                g_host[(e * M + r) * IN + c] = w_host[(e * 2 * M + r) * IN + c];
            }
        }
    }
    const gsh = [_]c_int{ E, M, IN };
    const g_f32 = mlx.mlx_array_new_data(g_host.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(g_f32);
    var g_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_bf16);
    try mlx.check(mlx.mlx_astype(&g_bf16, g_f32, .bfloat16, s));
    var gvec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(gvec);
    try mlx.check(mlx.mlx_quantize(&gvec, g_bf16, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, s));
    var gq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gq);
    var gsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gsc);
    var gbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gbi);
    try mlx.check(mlx.mlx_vector_array_get(&gq, gvec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&gsc, gvec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&gbi, gvec, 2));

    // gather_qmm through both, on BOTH dispatch shapes moeMLP2 uses:
    //   decode:  x [1,1,1,1,IN], inds [1,1,1], sorted=false
    //   prefill: x [N,1,IN], sorted rhs inds [N], sorted=true  ← the shape
    //            the live zeros bug fired on
    var x_host: [IN]f32 = undefined;
    for (&x_host, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.11 - 0.3;

    // decode shape
    {
        const xsh = [_]c_int{ 1, 1, 1, 1, IN };
        const x_f32 = mlx.mlx_array_new_data(&x_host, &xsh, 5, .float32);
        defer _ = mlx.mlx_array_free(x_f32);
        var x_bf16 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_bf16);
        try mlx.check(mlx.mlx_astype(&x_bf16, x_f32, .bfloat16, s));

        for (0..E) |e| {
            var inds_host = [_]u32{@intCast(e)};
            const ish = [_]c_int{ 1, 1, 1 };
            const inds = mlx.mlx_array_new_data(&inds_host, &ish, 3, .uint32);
            defer _ = mlx.mlx_array_free(inds);
            const no_idx = mlx.mlx_array{ .ctx = null };

            var out_split = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(out_split);
            try gatherExpertMm(&out_split, x_bf16, q_pair.gate, s_pair.gate, b_pair.gate, no_idx, inds, bits, gs, .affine, false, s);
            var out_ref = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(out_ref);
            try gatherExpertMm(&out_ref, x_bf16, gq, gsc, gbi, no_idx, inds, bits, gs, .affine, false, s);

            var split_host: [M]f32 = undefined;
            var ref_host: [M]f32 = undefined;
            try testReadF32(out_split, &split_host, s);
            try testReadF32(out_ref, &ref_host, s);
            var nonzero = false;
            for (split_host, ref_host) |a, b| {
                try testing.expectApproxEqAbs(b, a, 0.02);
                if (@abs(a) > 0.001) nonzero = true;
            }
            // The original bug produced exact zeros — make that explicit.
            try testing.expect(nonzero);
        }
    }

    // sorted prefill shape: N=E rows, one per expert, sorted inds [0..E)
    {
        const xr_host = try std.testing.allocator.alloc(f32, E * IN);
        defer std.testing.allocator.free(xr_host);
        for (xr_host, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.05 - 0.2;
        const xsh = [_]c_int{ E, 1, IN };
        const x_f32 = mlx.mlx_array_new_data(xr_host.ptr, &xsh, 3, .float32);
        defer _ = mlx.mlx_array_free(x_f32);
        var x_bf16 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_bf16);
        try mlx.check(mlx.mlx_astype(&x_bf16, x_f32, .bfloat16, s));

        var inds_host: [E]u32 = undefined;
        for (&inds_host, 0..) |*v, i| v.* = @intCast(i);
        const ish = [_]c_int{E};
        const inds = mlx.mlx_array_new_data(&inds_host, &ish, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds);
        const no_idx = mlx.mlx_array{ .ctx = null };

        var out_split = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out_split);
        try gatherExpertMm(&out_split, x_bf16, q_pair.gate, s_pair.gate, b_pair.gate, no_idx, inds, bits, gs, .affine, true, s);
        var out_ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out_ref);
        try gatherExpertMm(&out_ref, x_bf16, gq, gsc, gbi, no_idx, inds, bits, gs, .affine, true, s);

        const split_host = try std.testing.allocator.alloc(f32, E * M);
        defer std.testing.allocator.free(split_host);
        const ref_host = try std.testing.allocator.alloc(f32, E * M);
        defer std.testing.allocator.free(ref_host);
        try testReadF32(out_split, split_host, s);
        try testReadF32(out_ref, ref_host, s);
        var nonzero = false;
        for (split_host, ref_host) |a, b| {
            try testing.expectApproxEqAbs(b, a, 0.02);
            if (@abs(a) > 0.001) nonzero = true;
        }
        try testing.expect(nonzero);
    }
}

test "gatherExpertMm dense bf16 matches per-expert ground truth (decode + prefill shapes)" {
    // Centerpiece TDD gate for fully-dense bf16 MoE. Proves the dense gather_mm
    // path (gatherExpertMm with null scales + generalized transposeBf16Weight)
    // computes the same per-expert matmul as (a) an independent fp32 ground truth
    // and (b) the established quantized gather_qmm path — for BOTH the decode
    // (S=1, unsorted, 5D x) and prefill ([N,1,in], sorted) call shapes used by
    // moeMLP2. If the historical Qwen3.6-A3B-bf16 generation bug lived in the
    // expert gather, this test fails for the right reason.
    const s = mlx.gpuStream();
    const alloc = testing.allocator;

    const E = 4;
    const IN = 32; // must be a multiple of gs
    const OUT = 8;
    const gs = 32; // mlx_quantize supports group sizes 32, 64, 128
    const bits = 8; // near-lossless quant for a tight cross-check

    // Build w_orig [E, OUT, IN] bf16 from a deterministic small-valued buffer.
    var w_host: [E * OUT * IN]f32 = undefined;
    for (&w_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6))) * 0.05;
    var w_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_f32);
    {
        const sh = [_]c_int{ E, OUT, IN };
        w_f32 = mlx.mlx_array_new_data(&w_host, &sh, 3, .float32);
    }
    var w_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_bf16);
    try mlx.check(mlx.mlx_astype(&w_bf16, w_f32, .bfloat16, s));

    // Quantize → [q, sc, bi]; dequantize back so the dense path consumes the
    // exact numbers gather_qmm sees internally (cancels quant error in the
    // dense-vs-quant comparison).
    var qvec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_quantize(&qvec, w_bf16, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, s));
    var q_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_w);
    var q_sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_sc);
    var q_bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_bi);
    try mlx.check(mlx.mlx_vector_array_get(&q_w, qvec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q_sc, qvec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q_bi, qvec, 2));

    var w_deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_deq);
    try mlx.check(mlx.mlx_dequantize(&w_deq, q_w, q_sc, q_bi, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, .{ .value = .bfloat16, .has_value = true }, s));
    // w_deq_t [E, IN, OUT] — the dense weight layout the loader produces.
    const w_deq_t = try transposeBf16Weight(w_deq, s);
    defer _ = mlx.mlx_array_free(w_deq_t);

    // Read dequantized weights to host for ground truth: w_deq_host[e*OUT*IN + o*IN + k].
    var w_deq_host: [E * OUT * IN]f32 = undefined;
    try testReadF32(w_deq, &w_deq_host, s);

    const null_sc = mlx.mlx_array{ .ctx = null };
    const null_bi = mlx.mlx_array{ .ctx = null };
    const no_idx = mlx.mlx_array{ .ctx = null };

    // ── Decode shape: x_exp [1,1,1,1,IN], inds [1,1,K], sorted=false ──
    {
        const K = 2;
        var x_host: [IN]f32 = undefined;
        for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2))) * 0.1;
        var x_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_f32);
        {
            const sh = [_]c_int{IN};
            x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 1, .float32);
        }
        var x_bf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_bf);
        try mlx.check(mlx.mlx_astype(&x_bf, x_f32, .bfloat16, s));
        var x_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_exp);
        {
            const sh = [_]c_int{ 1, 1, 1, 1, IN };
            try mlx.check(mlx.mlx_reshape(&x_exp, x_bf, &sh, 5, s));
        }
        var x_bf_host: [IN]f32 = undefined;
        try testReadF32(x_bf, &x_bf_host, s);

        const inds_host = [_]u32{ 1, 3 };
        var inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inds);
        {
            const sh = [_]c_int{ 1, 1, K };
            inds = mlx.mlx_array_new_data(&inds_host, &sh, 3, .uint32);
        }

        // Ground truth: gt[k*OUT + o] = sum_in x[in] * w_deq[inds[k]][o][in].
        var gt: [K * OUT]f32 = undefined;
        for (0..K) |k| {
            const e = inds_host[k];
            for (0..OUT) |o| {
                var acc: f32 = 0;
                for (0..IN) |in| acc += x_bf_host[in] * w_deq_host[e * OUT * IN + o * IN + in];
                gt[k * OUT + o] = acc;
            }
        }

        // Dense path.
        var dense5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense5);
        try gatherExpertMm(&dense5, x_exp, w_deq_t, null_sc, null_bi, no_idx, inds, bits, gs, .affine, false, s);
        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_squeeze(&dense, dense5, s)); // [K, OUT]
        const dense_host = try alloc.alloc(f32, K * OUT);
        defer alloc.free(dense_host);
        try testReadF32(dense, dense_host, s);

        // Quantized path (cross-check).
        var quant5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant5);
        try gatherExpertMm(&quant5, x_exp, q_w, q_sc, q_bi, no_idx, inds, bits, gs, .affine, false, s);
        var quant = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant);
        try mlx.check(mlx.mlx_squeeze(&quant, quant5, s));
        const quant_host = try alloc.alloc(f32, K * OUT);
        defer alloc.free(quant_host);
        try testReadF32(quant, quant_host, s);

        for (0..K * OUT) |i| {
            try testing.expectApproxEqAbs(gt[i], dense_host[i], 2e-2); // dense == ground truth
            try testing.expectApproxEqAbs(gt[i], quant_host[i], 2e-2); // quant agrees too
        }
    }

    // ── Prefill/sorted shape: x_rep [N,1,IN], sorted_inds [N], sorted=true ──
    {
        const N = 5;
        var x_host: [N * IN]f32 = undefined;
        for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3))) * 0.07;
        var x_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_f32);
        {
            const sh = [_]c_int{ N, 1, IN };
            x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 3, .float32);
        }
        var x_rep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rep);
        try mlx.check(mlx.mlx_astype(&x_rep, x_f32, .bfloat16, s));
        var x_bf_host: [N * IN]f32 = undefined;
        try testReadF32(x_rep, &x_bf_host, s);

        const sorted_host = [_]u32{ 0, 0, 1, 2, 3 }; // sorted experts
        var sorted_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted_inds);
        {
            const sh = [_]c_int{N};
            sorted_inds = mlx.mlx_array_new_data(&sorted_host, &sh, 1, .uint32);
        }

        // Ground truth: gt[i*OUT + o] = sum_in x[i][in] * w_deq[sorted[i]][o][in].
        var gt: [N * OUT]f32 = undefined;
        for (0..N) |i| {
            const e = sorted_host[i];
            for (0..OUT) |o| {
                var acc: f32 = 0;
                for (0..IN) |in| acc += x_bf_host[i * IN + in] * w_deq_host[e * OUT * IN + o * IN + in];
                gt[i * OUT + o] = acc;
            }
        }

        var dense3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense3);
        try gatherExpertMm(&dense3, x_rep, w_deq_t, null_sc, null_bi, no_idx, sorted_inds, bits, gs, .affine, true, s);
        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_squeeze(&dense, dense3, s)); // [N, OUT]
        const dense_host = try alloc.alloc(f32, N * OUT);
        defer alloc.free(dense_host);
        try testReadF32(dense, dense_host, s);

        var quant3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant3);
        try gatherExpertMm(&quant3, x_rep, q_w, q_sc, q_bi, no_idx, sorted_inds, bits, gs, .affine, true, s);
        var quant = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant);
        try mlx.check(mlx.mlx_squeeze(&quant, quant3, s));
        const quant_host = try alloc.alloc(f32, N * OUT);
        defer alloc.free(quant_host);
        try testReadF32(quant, quant_host, s);

        for (0..N * OUT) |i| {
            try testing.expectApproxEqAbs(gt[i], dense_host[i], 2e-2);
            try testing.expectApproxEqAbs(gt[i], quant_host[i], 2e-2);
        }
    }
}

test "gatherExpertMm nvfp4 matches per-expert dequantized reference" {
    // MoE expert dispatch on an nvfp4 checkpoint: gather_qmm must receive
    // mode="nvfp4" + null biases (decode AND prefill shapes, same calling
    // convention as moeMLP2).
    const s = mlx.gpuStream();

    const E = 4;
    const IN = 32;
    const OUT = 8;

    var w_host: [E * OUT * IN]f32 = undefined;
    for (&w_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6))) * 0.05;
    var w_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_f32);
    {
        const sh = [_]c_int{ E, OUT, IN };
        w_f32 = mlx.mlx_array_new_data(&w_host, &sh, 3, .float32);
    }
    var w_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_bf16);
    try mlx.check(mlx.mlx_astype(&w_bf16, w_f32, .bfloat16, s));

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
    const no_idx = mlx.mlx_array{ .ctx = null };

    var w_deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_deq);
    try mlx.check(mlx.mlx_dequantize(&w_deq, q_w, q_sc, null_bi, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }, s));
    var w_deq_host: [E * OUT * IN]f32 = undefined;
    try testReadF32(w_deq, &w_deq_host, s);

    // ── Decode shape: x_exp [1,1,1,1,IN], inds [1,1,K], sorted=false ──
    {
        const K = 2;
        var x_host: [IN]f32 = undefined;
        for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2))) * 0.1;
        var x_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_f32);
        {
            const sh = [_]c_int{IN};
            x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 1, .float32);
        }
        var x_bf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_bf);
        try mlx.check(mlx.mlx_astype(&x_bf, x_f32, .bfloat16, s));
        var x_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_exp);
        {
            const sh = [_]c_int{ 1, 1, 1, 1, IN };
            try mlx.check(mlx.mlx_reshape(&x_exp, x_bf, &sh, 5, s));
        }
        var x_bf_host: [IN]f32 = undefined;
        try testReadF32(x_bf, &x_bf_host, s);

        const inds_host = [_]u32{ 1, 3 };
        var inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inds);
        {
            const sh = [_]c_int{ 1, 1, K };
            inds = mlx.mlx_array_new_data(&inds_host, &sh, 3, .uint32);
        }

        var gt: [K * OUT]f32 = undefined;
        for (0..K) |k| {
            const e = inds_host[k];
            for (0..OUT) |o| {
                var acc: f32 = 0;
                for (0..IN) |in| acc += x_bf_host[in] * w_deq_host[e * OUT * IN + o * IN + in];
                gt[k * OUT + o] = acc;
            }
        }

        var quant5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant5);
        try gatherExpertMm(&quant5, x_exp, q_w, q_sc, null_bi, no_idx, inds, 4, 16, .nvfp4, false, s);
        var quant = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant);
        try mlx.check(mlx.mlx_squeeze(&quant, quant5, s)); // [K, OUT]
        var quant_host: [K * OUT]f32 = undefined;
        try testReadF32(quant, &quant_host, s);

        for (0..K * OUT) |i| try testing.expectApproxEqAbs(gt[i], quant_host[i], 5e-2);
    }

    // ── Prefill/sorted shape: x_rep [N,1,IN], sorted_inds [N], sorted=true ──
    {
        const N = 5;
        var x_host: [N * IN]f32 = undefined;
        for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3))) * 0.07;
        var x_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_f32);
        {
            const sh = [_]c_int{ N, 1, IN };
            x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 3, .float32);
        }
        var x_rep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rep);
        try mlx.check(mlx.mlx_astype(&x_rep, x_f32, .bfloat16, s));
        var x_bf_host: [N * IN]f32 = undefined;
        try testReadF32(x_rep, &x_bf_host, s);

        const sorted_host = [_]u32{ 0, 0, 1, 2, 3 };
        var sorted_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted_inds);
        {
            const sh = [_]c_int{N};
            sorted_inds = mlx.mlx_array_new_data(&sorted_host, &sh, 1, .uint32);
        }

        var gt: [N * OUT]f32 = undefined;
        for (0..N) |i| {
            const e = sorted_host[i];
            for (0..OUT) |o| {
                var acc: f32 = 0;
                for (0..IN) |in| acc += x_bf_host[i * IN + in] * w_deq_host[e * OUT * IN + o * IN + in];
                gt[i * OUT + o] = acc;
            }
        }

        var quant3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant3);
        try gatherExpertMm(&quant3, x_rep, q_w, q_sc, null_bi, no_idx, sorted_inds, 4, 16, .nvfp4, true, s);
        var quant = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant);
        try mlx.check(mlx.mlx_squeeze(&quant, quant3, s));
        var quant_host: [N * OUT]f32 = undefined;
        try testReadF32(quant, &quant_host, s);

        for (0..N * OUT) |i| try testing.expectApproxEqAbs(gt[i], quant_host[i], 5e-2);
    }
}
