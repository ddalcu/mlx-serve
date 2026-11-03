//! `sdpa.zig`'s tests, out of its import graph: a plugin unit that reaches `sdpa.zig` never compiles them.

const sdpa = @import("sdpa.zig");
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const testing = std.testing;
const attn256MaxDiff = @import("attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("attn256.zig").attn256RandBf16;
const attn256Reference = @import("attn256.zig").attn256Reference;
const sdpaForceFusedFor = sdpa.sdpaForceFusedFor;
const splitCausalSdpa = sdpa.splitCausalSdpa;
const splitMaskedSdpa256 = sdpa.splitMaskedSdpa256;
const attn256QsaMask = @import("../../../plugins/qwen4_exp/qsa/gather.zig").attn256QsaMask;

test "splitMaskedSdpa256: verify-width array-mask rows split through the vector kernel match one dispatch" {
    const s = mlx.gpuStream();
    sdpa.sdpa_split_override = true;
    defer sdpa.sdpa_split_override = null;
    var prng = std.Random.DefaultPrng.init(0x5b5b);
    const rnd = prng.random();
    // gqa 12 like qwen4_exp (24/2): qL 4 is the unfused fallback in stock MLX.
    const q_shape = [_]c_int{ 1, 24, 4, 256 };
    const kv_shape = [_]c_int{ 1, 2, 300, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const mask = try attn256QsaMask(rnd, 4, 300);
    defer _ = mlx.mlx_array_free(mask);
    const scale: f32 = 1.0 / 16.0;
    const split = (try splitMaskedSdpa256(s, q, k, v, scale, mask)) orelse return error.SplitDeclined;
    defer _ = mlx.mlx_array_free(split);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    try std.testing.expect(try attn256MaxDiff(split, ref, s) < 0.005);
    // qL 2 already fits the vector kernel: declined.
    const q2_shape = [_]c_int{ 1, 24, 2, 256 };
    const q2 = try attn256RandBf16(rnd, &q2_shape, s);
    defer _ = mlx.mlx_array_free(q2);
    const m2 = try attn256QsaMask(rnd, 2, 300);
    defer _ = mlx.mlx_array_free(m2);
    try std.testing.expect((try splitMaskedSdpa256(s, q2, k, v, scale, m2)) == null);
}

/// Like attn256MaxDiff but errors on any non-finite element on either side —
/// NaN > max_err is false, so an all-NaN pair would otherwise diff to 0.
fn attn256FiniteMaxDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, b, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd = mlx.mlx_array_data_float32(b32) orelse return error.InvalidDtype;
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    var max_diff: f32 = 0;
    for (0..n) |i| {
        if (!std.math.isFinite(ad[i]) or !std.math.isFinite(bd[i])) return error.NonFinite;
        max_diff = @max(max_diff, @abs(ad[i] - bd[i]));
    }
    return max_diff;
}

const CausalSplitCase = struct { hq: c_int, hkv: c_int, ql: c_int, kv: c_int, hd: c_int };

fn causalSplitCaseArrays(rnd: std.Random, c: CausalSplitCase, s: mlx.mlx_stream) ![3]mlx.mlx_array {
    const kL = c.kv + c.ql;
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, c.hq, c.ql, c.hd }, s);
    errdefer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, c.hkv, kL, c.hd }, s);
    errdefer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, c.hkv, kL, c.hd }, s);
    return .{ q, k, v };
}

/// Row r of a bottom-right causal block sees keys 0 .. kL-qL+r. One qL=1
/// vector call per row over exactly that prefix: an offset reference that
/// does not go through MLX's causal fallback.
fn causalPerRowReference(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const strides = [_]c_int{ 1, 1, 1, 1 };
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    var parts = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(std.testing.allocator);
    }
    var r: c_int = 0;
    while (r < qs[2]) : (r += 1) {
        const kend = ks[2] - qs[2] + r + 1;
        var q_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_r);
        var k_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_r);
        var v_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_r);
        try mlx.check(mlx.mlx_slice(&q_r, q, &[_]c_int{ 0, 0, r, 0 }, 4, &[_]c_int{ 1, qs[1], r + 1, qs[3] }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&k_r, k, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, ks[1], kend, ks[3] }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&v_r, v, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, ks[1], kend, ks[3] }, 4, &strides, 4, s));
        var o = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q_r, k_r, v_r, scale, "", none, .{ .ctx = null }, false, s));
        try parts.append(std.testing.allocator, o);
    }
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    return out;
}

test "splitCausalSdpa: gqa-aware groups match one causal sdpa across widths, kv and head dims" {
    const s = mlx.gpuStream();
    sdpa.sdpa_split_override = true;
    defer sdpa.sdpa_split_override = null;
    sdpa.nax_sdpa_override = false;
    defer sdpa.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x5D9A);
    const rnd = prng.random();
    const cases = [_]CausalSplitCase{
        // Qwen3.8-Flash-Next: gqa 12 (24/2), 2 rows per group.
        .{ .hq = 24, .hkv = 2, .ql = 3, .kv = 1, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 1, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 100, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 6, .kv = 100, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 15, .kv = 100, .hd = 256 },
        // kL >= 1024: MLX's 2-pass vector kernel on Max/Ultra.
        .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 2051, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 15, .kv = 2051, .hd = 256 },
        // gqa 6 (27B geometry): 6..9 as before, 10..15 now split too.
        .{ .hq = 12, .hkv = 2, .ql = 6, .kv = 187, .hd = 256 },
        .{ .hq = 12, .hkv = 2, .ql = 9, .kv = 184, .hd = 256 },
        .{ .hq = 12, .hkv = 2, .ql = 10, .kv = 100, .hd = 256 },
        .{ .hq = 12, .hkv = 2, .ql = 15, .kv = 100, .hd = 256 },
        // gqa 32: 1-row groups. gqa 3 at qL 11: an 8-row group + 3.
        .{ .hq = 32, .hkv = 1, .ql = 3, .kv = 100, .hd = 256 },
        .{ .hq = 6, .hkv = 2, .ql = 11, .kv = 100, .hd = 256 },
        // gqa 8 (gemma-like hd 256): 4-row groups.
        .{ .hq = 16, .hkv = 2, .ql = 5, .kv = 1, .hd = 256 },
    };
    const scale: f32 = 1.0 / 16.0;
    for (cases) |c| {
        const arrs = try causalSplitCaseArrays(rnd, c, s);
        defer for (arrs) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const split = (try splitCausalSdpa(s, arrs[0], arrs[1], arrs[2], scale)) orelse {
            std.debug.print("split declined: {any}\n", .{c});
            return error.SplitDeclined;
        };
        defer _ = mlx.mlx_array_free(split);
        // Tolerance, not byte-identity: the single call is MLX's unfused
        // fallback, a different reduction order (the verify-lane rule).
        const ref = try attn256Reference(arrs[0], arrs[1], arrs[2], scale, "causal", .{ .ctx = null }, s);
        defer _ = mlx.mlx_array_free(ref);
        const max_diff = try attn256FiniteMaxDiff(split, ref, s);
        if (max_diff >= 0.005) std.debug.print("case {any}: max diff {d}\n", .{ c, max_diff });
        try std.testing.expect(max_diff < 0.005);
    }
}

test "splitCausalSdpa: every row sees exactly its causal window (bit-identical to per-row vector calls below kL 1024)" {
    const s = mlx.gpuStream();
    sdpa.sdpa_split_override = true;
    defer sdpa.sdpa_split_override = null;
    sdpa.nax_sdpa_override = false;
    defer sdpa.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x5DA1);
    const rnd = prng.random();
    // kL < 1024 keeps both sides on the 1-pass vector kernel, which skips
    // masked keys, so each row's math depends only on its key set.
    const cases = [_]CausalSplitCase{
        .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 100, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 15, .kv = 37, .hd = 256 },
        .{ .hq = 12, .hkv = 2, .ql = 9, .kv = 100, .hd = 256 },
        .{ .hq = 32, .hkv = 1, .ql = 3, .kv = 1, .hd = 256 },
        .{ .hq = 16, .hkv = 2, .ql = 7, .kv = 100, .hd = 256 },
    };
    const scale: f32 = 1.0 / 16.0;
    for (cases) |c| {
        const arrs = try causalSplitCaseArrays(rnd, c, s);
        defer for (arrs) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const split = (try splitCausalSdpa(s, arrs[0], arrs[1], arrs[2], scale)) orelse return error.SplitDeclined;
        defer _ = mlx.mlx_array_free(split);
        const ref = try causalPerRowReference(arrs[0], arrs[1], arrs[2], scale, s);
        defer _ = mlx.mlx_array_free(ref);
        const max_diff = try attn256FiniteMaxDiff(split, ref, s);
        if (max_diff != 0) std.debug.print("case {any}: max diff {d}\n", .{ c, max_diff });
        try std.testing.expect(max_diff == 0);
    }
}

test "splitCausalSdpa: declines outside its envelope" {
    const s = mlx.gpuStream();
    sdpa.sdpa_split_override = true;
    defer sdpa.sdpa_split_override = null;
    sdpa.nax_sdpa_override = false;
    defer sdpa.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x5DEC);
    const rnd = prng.random();
    const declines = [_]CausalSplitCase{
        // The vector kernel already serves qL*gqa <= 32.
        .{ .hq = 24, .hkv = 2, .ql = 2, .kv = 64, .hd = 256 },
        .{ .hq = 12, .hkv = 2, .ql = 5, .kv = 64, .hd = 256 },
        .{ .hq = 6, .hkv = 2, .ql = 7, .kv = 64, .hd = 128 },
        // qL > 8 below hd 256: MLX's fused full kernel serves it.
        .{ .hq = 6, .hkv = 2, .ql = 10, .kv = 64, .hd = 128 },
        // Below hd 256 the split is unmeasured, so it declines even past the
        // vector wall (Llama/Mistral hd 128 gqa 8, hd 64).
        .{ .hq = 32, .hkv = 4, .ql = 6, .kv = 64, .hd = 128 },
        .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 64, .hd = 128 },
        .{ .hq = 16, .hkv = 2, .ql = 5, .kv = 64, .hd = 64 },
        // qL >= 16: fusedSdpa256Prefill's range.
        .{ .hq = 24, .hkv = 2, .ql = 16, .kv = 64, .hd = 256 },
        // hd 192: MLX prefers its unfused path there.
        .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 64, .hd = 192 },
        // Decode width.
        .{ .hq = 24, .hkv = 2, .ql = 1, .kv = 64, .hd = 256 },
    };
    for (declines) |c| {
        const arrs = try causalSplitCaseArrays(rnd, c, s);
        defer for (arrs) |a| {
            _ = mlx.mlx_array_free(a);
        };
        if ((try splitCausalSdpa(s, arrs[0], arrs[1], arrs[2], 1.0)) != null) {
            std.debug.print("split engaged outside envelope: {any}\n", .{c});
            return error.SplitEngaged;
        }
    }

    // Batch 2 -> null (the exactness argument is B==1 only).
    const qb = try attn256RandBf16(rnd, &[_]c_int{ 2, 24, 4, 256 }, s);
    defer _ = mlx.mlx_array_free(qb);
    const kb = try attn256RandBf16(rnd, &[_]c_int{ 2, 2, 64, 256 }, s);
    defer _ = mlx.mlx_array_free(kb);
    try std.testing.expect((try splitCausalSdpa(s, qb, kb, kb, 1.0)) == null);

    // KV shorter than q -> null (no window to split).
    const q7 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, 7, 256 }, s);
    defer _ = mlx.mlx_array_free(q7);
    const kshort = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 5, 256 }, s);
    defer _ = mlx.mlx_array_free(kshort);
    try std.testing.expect((try splitCausalSdpa(s, q7, kshort, kshort, 1.0)) == null);

    // Kill switch -> null even for a conforming call.
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 64, 256 }, s);
    defer _ = mlx.mlx_array_free(k);
    sdpa.sdpa_split_override = false;
    try std.testing.expect((try splitCausalSdpa(s, q7, k, k, 1.0)) == null);
}

test "splitCausalSdpa: yields to the fused NAX sdpa at hd 256 past 8 rows, still splits below" {
    const s = mlx.gpuStream();
    sdpa.sdpa_split_override = true;
    defer sdpa.sdpa_split_override = null;
    sdpa.nax_sdpa_override = true;
    defer sdpa.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x5DAA);
    const rnd = prng.random();
    // qL > 8 at hd 256 with NAX preferred: the caller's single call runs
    // MLX's fused NAX kernel (sdpaForceFused), so the split must decline.
    for ([_]CausalSplitCase{
        .{ .hq = 12, .hkv = 2, .ql = 10, .kv = 100, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 15, .kv = 100, .hd = 256 },
    }) |c| {
        const arrs = try causalSplitCaseArrays(rnd, c, s);
        defer for (arrs) |a| {
            _ = mlx.mlx_array_free(a);
        };
        if ((try splitCausalSdpa(s, arrs[0], arrs[1], arrs[2], 1.0)) != null) {
            std.debug.print("split pre-empted NAX: {any}\n", .{c});
            return error.SplitEngaged;
        }
    }
    // qL <= 8 never force-fuses (the vector wall would throw): split engages.
    const arrs = try causalSplitCaseArrays(rnd, .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 100, .hd = 256 }, s);
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const split = (try splitCausalSdpa(s, arrs[0], arrs[1], arrs[2], 1.0)) orelse return error.SplitDeclined;
    _ = mlx.mlx_array_free(split);
}

test "sdpaForceFusedFor: only shapes MLX has a fused kernel for (hd 256, > 8 rows, q <= k)" {
    try testing.expect(sdpaForceFusedFor(true, 16, 4096, 256));
    try testing.expect(sdpaForceFusedFor(true, 9, 9, 256));
    // <= 8 rows is the vector kernel; its q*gqa <= 32 wall makes force_fused THROW.
    try testing.expect(!sdpaForceFusedFor(true, 8, 4096, 256));
    // hd <= 128 already fuses on its own; hd 192/exotic dims are not ours to force.
    try testing.expect(!sdpaForceFusedFor(true, 16, 4096, 128));
    try testing.expect(!sdpaForceFusedFor(true, 16, 4096, 192));
    try testing.expect(!sdpaForceFusedFor(true, 16, 8, 256));
    try testing.expect(!sdpaForceFusedFor(false, 16, 4096, 256));
}

test "sdpa force_fused: hd-256 causal blocks the gate admits match the unforced call (plumbing + shape gate)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest; // CUDA has no fused hd-256 kernel to force
    // The shapes sdpaForceFusedFor admits (hd 256, > 8 rows, q <= k): the
    // forced kernel (steel on M1-M4, the NAX variant on M5) must agree with
    // the unforced dispatch (unfused at hd 256 on non-NAX) and never throw.
    // <= 8 rows is NOT covered on purpose: that is the vector kernel, and
    // force_fused on a gqa-8 block there throws (the gate's whole point).
    const s = mlx.mlx_default_gpu_stream_new();
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    const cases = [_][3]c_int{ .{ 16, 4096, 8 }, .{ 9, 512, 8 }, .{ 64, 64, 2 } }; // q, k, gqa
    for (cases) |c| {
        const q_shape = [_]c_int{ 1, 2 * c[2], c[0], 256 };
        const k_shape = [_]c_int{ 1, 2, c[1], 256 };
        try std.testing.expect(sdpaForceFusedFor(true, c[0], c[1], 256));
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try attn256RandBf16(rnd, &k_shape, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try attn256RandBf16(rnd, &k_shape, s);
        defer _ = mlx.mlx_array_free(v);
        var plain = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(plain);
        var forced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(forced);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&plain, q, k, v, 0.0625, "causal", none, .{ .ctx = null }, false, s));
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&forced, q, k, v, 0.0625, "causal", none, .{ .ctx = null }, true, s));
        const diff = try attn256MaxDiff(plain, forced, s);
        try std.testing.expect(diff < 5e-3);
    }
}
