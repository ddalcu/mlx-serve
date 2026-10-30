//! `fused_activations.zig`'s tests, out of its import graph: a plugin unit that reaches `fused_activations.zig` never compiles them.

const fused_activations = @import("fused_activations.zig");
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const testing = std.testing;
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;
const gdnParityRand = @import("../gdn/decode_fused.zig").gdnParityRand;
const sliceQkvPart = @import("../linear/fused_rows.zig").sliceQkvPart;
const clampedSwiGLU = @import("activations.zig").clampedSwiGLU;
const gptOssSwiGLU = @import("activations.zig").gptOssSwiGLU;
const all16BitPatterns = fused_activations.all16BitPatterns;
const fusedClampedSwiGLU = fused_activations.fusedClampedSwiGLU;
const fusedGeluErfMul = fused_activations.fusedGeluErfMul;
const fusedSwiGLU = fused_activations.fusedSwiGLU;
const geluErfChain = fused_activations.geluErfChain;
const swigluJoined = fused_activations.swigluJoined;
const swigluSigTable = fused_activations.swigluSigTable;
const qmatmulBits = @import("../linear/qmatmul.zig").qmatmulBits;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;

test "fused SwiGLU is bit-identical to the sigmoid+multiply+multiply chain" {
    // The reason this kernel exists instead of extending mlx_compile to the
    // silu arm: the compiled fusion is NOT bit-identical (greedy diverged at
    // token ~55 on Laguna XS). Assert what the compiled path could not.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x51117001);
    const rnd = prng.random();
    fused_activations.swiglu_fused_override = true;
    defer fused_activations.swiglu_fused_override = null;

    // Sizes that are and are not multiples of the threadgroup, so the grid
    // round-up and the kernel's bound check are both exercised.
    for ([_]c_int{ 8, 512, 4096, 4097 }) |n| {
        for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
            const cnt: usize = @intCast(n);
            const gbuf = try allocator.alloc(f32, cnt);
            defer allocator.free(gbuf);
            const ubuf = try allocator.alloc(f32, cnt);
            defer allocator.free(ubuf);
            for (gbuf, 0..) |*v, i| v.* = switch (i % 5) {
                0 => -30.0,
                1 => 25.0,
                2 => 0.0,
                else => (rnd.float(f32) - 0.5) * 10.0,
            };
            for (ubuf) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;

            const sh = [_]c_int{n};
            const g32 = mlx.mlx_array_new_data(gbuf.ptr, &sh, 1, .float32);
            defer _ = mlx.mlx_array_free(g32);
            const u32a = mlx.mlx_array_new_data(ubuf.ptr, &sh, 1, .float32);
            defer _ = mlx.mlx_array_free(u32a);
            var gate = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate);
            try mlx.check(mlx.mlx_astype(&gate, g32, dt, s));
            var up = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up);
            try mlx.check(mlx.mlx_astype(&up, u32a, dt, s));

            // Reference: exactly the ops computeGeglu's fallback issues.
            var sig = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sig);
            try mlx.check(mlx.mlx_sigmoid(&sig, gate, s));
            var act = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(act);
            try mlx.check(mlx.mlx_multiply(&act, gate, sig, s));
            var ref = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref);
            try mlx.check(mlx.mlx_multiply(&ref, act, up, s));

            const got = (try fusedSwiGLU(s, gate, up)) orelse return error.FusedSwigluDeclined;
            defer _ = mlx.mlx_array_free(got);

            const ref_host = try allocator.alloc(f32, cnt);
            defer allocator.free(ref_host);
            const got_host = try allocator.alloc(f32, cnt);
            defer allocator.free(got_host);
            try testReadF32(ref, ref_host, s);
            try testReadF32(got, got_host, s);
            try testing.expectEqualSlices(f32, ref_host, got_host);
        }
    }
}

test "fused SwiGLU config cache keys on the SHAPE, not the element count (Laguna restore crash)" {
    // A 16-token hot-cache-restore suffix prefill on Laguna
    // XS killed the server with "[matmul] (1,16,512) vs (8192,2048)". A MoE
    // layer's SwiGLU at 16 tokens is (1,16,512) = 8192 elements — the SAME
    // count, rank and dtype as a dense mlp_only layer at decode (1,1,8192).
    // The config cache was keyed on {n, tg, ndim, dtype}, so the second call
    // reused a config whose baked OUTPUT SHAPE was the first caller's, and
    // the dense down-proj matmul died on the mislabeled activation. The key
    // must be the full shape; equal products are not equal shapes.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    fused_activations.swiglu_fused_override = true;
    defer fused_activations.swiglu_fused_override = null;

    const shapes = [2][3]c_int{ .{ 1, 16, 512 }, .{ 1, 1, 8192 } };
    for (shapes) |sh| {
        const cnt: usize = 8192;
        const buf = try allocator.alloc(f32, cnt);
        defer allocator.free(buf);
        for (buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) - 3.0;
        const a32 = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
        defer _ = mlx.mlx_array_free(a32);
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_astype(&gate, a32, .bfloat16, s));

        const got = (try fusedSwiGLU(s, gate, gate)) orelse return error.FusedSwigluDeclined;
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_array_eval(got));
        const osh = mlx.getShape(got);
        try testing.expectEqualSlices(c_int, &sh, osh);
    }
}

test "fused GELU-erf gate is bit-identical to the op chain over the whole 16-bit domain" {
    // Bar: every bf16/f16 gate pattern times a handful of up values, byte for
    // byte against geluErfChain + mlx_multiply.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    fused_activations.swiglu_fused_override = true;
    defer fused_activations.swiglu_fused_override = null;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
        const one = try scalarOf(1.0, dt, s);
        defer _ = mlx.mlx_array_free(one);
        const half = try scalarOf(0.5, dt, s);
        defer _ = mlx.mlx_array_free(half);
        const gate = try all16BitPatterns(s, dt, allocator);
        defer _ = mlx.mlx_array_free(gate);
        for ([_]f32{ 1.0, -0.375, 3.140625, 1e-3 }) |uv| {
            const uscalar = try scalarOf(uv, dt, s);
            defer _ = mlx.mlx_array_free(uscalar);
            var up = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up);
            try mlx.check(mlx.mlx_multiply(&up, uscalar, one, s));
            var up_full = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_full);
            try mlx.check(mlx.mlx_broadcast_to(&up_full, up, mlx.getShape(gate).ptr, 1, s));
            var up_c = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_c);
            try mlx.check(mlx.mlx_contiguous(&up_c, up_full, false, s));
            const act = try geluErfChain(gate, one, half, s);
            defer _ = mlx.mlx_array_free(act);
            var ref = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref);
            try mlx.check(mlx.mlx_multiply(&ref, act, up_c, s));
            const got = (try fusedGeluErfMul(s, gate, up_c, one, half)) orelse return error.FusedGeluDeclined;
            defer _ = mlx.mlx_array_free(got);
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_array_equal(&eq, ref, got, true, s));
            try mlx.check(mlx.mlx_array_eval(eq));
            var eq_v: bool = false;
            try mlx.check(mlx.mlx_array_item_bool(&eq_v, eq));
            try testing.expect(eq_v);
        }
    }
}

test "fused clamped swiglu is bit-identical to the clampedSwiGLU chain" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x61A5);
    const rnd = prng.random();
    // Spread past +-10 so both clamps bite.
    const gate = try gdnParityRand(rnd, &.{ 3, 7, 2048 }, 16.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(gate);
    const up = try gdnParityRand(rnd, &.{ 3, 7, 2048 }, 16.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(up);
    const want = try clampedSwiGLU(s, gate, up, 10.0);
    defer _ = mlx.mlx_array_free(want);
    const got = (try fusedClampedSwiGLU(s, gate, up, 10.0)) orelse return error.SkipZigTest;
    defer _ = mlx.mlx_array_free(got);
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, want, got, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var eq_v: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&eq_v, eq));
    try testing.expect(eq_v);
}

test "joined swiglu is bit-identical to fusedSwiGLU over the two halves" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5316);
    const rnd = prng.random();
    for ([_]c_int{ 1, 16, 33 }) |rows| {
        const both = try gdnParityRand(rnd, &.{ 1, rows, 2 * 1088 }, 6.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(both);
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        var up = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up);
        try sliceQkvPart(&gate, both, 0, 1088, s);
        try sliceQkvPart(&up, both, 1088, 1088, s);
        const ref = (try fusedSwiGLU(s, gate, up)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(ref);
        const got = (try swigluJoined(s, both)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, ref, s));
    }
}

test "fused GELU-tanh gate is bit-identical to the current GELU path times up" {
    // The table is built from `Transformer.gelu` itself, so the bar is the
    // multiply: table[x] * up over every bf16 x and a few up values.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    fused_activations.swiglu_fused_override = true;
    defer fused_activations.swiglu_fused_override = null;
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.compiled_gelu = null;
    xfm.three = bf16Scalar(3.0, s);
    defer _ = mlx.mlx_array_free(xfm.three.?);
    xfm.gelu_inner = bf16Scalar(0.044715, s);
    defer _ = mlx.mlx_array_free(xfm.gelu_inner.?);
    xfm.gelu_coeff = bf16Scalar(0.7978845608028654, s);
    defer _ = mlx.mlx_array_free(xfm.gelu_coeff.?);
    xfm.one = bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(xfm.one);
    xfm.half = bf16Scalar(0.5, s);
    defer _ = mlx.mlx_array_free(xfm.half);
    const gate = try all16BitPatterns(s, .bfloat16, allocator);
    defer _ = mlx.mlx_array_free(gate);
    for ([_]f32{ 1.0, -0.375, 3.140625 }) |uv| {
        const uscalar = try scalarOf(uv, .bfloat16, s);
        defer _ = mlx.mlx_array_free(uscalar);
        var up_full = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_full);
        try mlx.check(mlx.mlx_broadcast_to(&up_full, uscalar, mlx.getShape(gate).ptr, 1, s));
        var up = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up);
        try mlx.check(mlx.mlx_contiguous(&up, up_full, false, s));
        const act = try xfm.gelu(gate);
        defer _ = mlx.mlx_array_free(act);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_multiply(&ref, act, up, s));
        const got = (try xfm.fusedGeluTanhMul(gate, up)) orelse return error.FusedGeluDeclined;
        defer _ = mlx.mlx_array_free(got);
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, ref, got, true, s));
        try mlx.check(mlx.mlx_array_eval(eq));
        var eq_v: bool = false;
        try mlx.check(mlx.mlx_array_item_bool(&eq_v, eq));
        try testing.expect(eq_v);
    }
}

test "fused SwiGLU matches the chain on EVERY finite bf16 gate value" {
    // Random sampling said bit-identical while the live model diverged at
    // token ~55, so the sigmoid is swept exhaustively instead: all 65536 bf16
    // bit patterns, which is the whole domain of the activation.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    fused_activations.swiglu_fused_override = true;
    defer fused_activations.swiglu_fused_override = null;

    const N: usize = 1 << 16;
    const sh = [_]c_int{@intCast(N)};
    // An f32 whose low 16 mantissa bits are zero converts to bf16 EXACTLY, so
    // walking the top half of the f32 word walks every bf16 pattern there is —
    // subnormals, both zeros, infinities and NaNs included.
    const gbuf = try allocator.alloc(f32, N);
    defer allocator.free(gbuf);
    for (gbuf, 0..) |*v, i| v.* = @bitCast(@as(u32, @intCast(i)) << 16);
    const g32 = mlx.mlx_array_new_data(gbuf.ptr, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(g32);
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_astype(&gate, g32, .bfloat16, s));

    const ones = try allocator.alloc(f32, N);
    defer allocator.free(ones);
    for (ones, 0..) |*v, i| v.* = if (i % 3 == 0) 1.0 else if (i % 3 == 1) -0.375 else 2.5;
    const up32 = mlx.mlx_array_new_data(ones.ptr, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(up32);
    var up = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_astype(&up, up32, .bfloat16, s));

    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, gate, s));
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_multiply(&act, gate, sig, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_multiply(&ref, act, up, s));

    const got = (try fusedSwiGLU(s, gate, up)) orelse return error.FusedSwigluDeclined;
    defer _ = mlx.mlx_array_free(got);

    const ref_host = try allocator.alloc(f32, N);
    defer allocator.free(ref_host);
    const got_host = try allocator.alloc(f32, N);
    defer allocator.free(got_host);
    try testReadF32(ref, ref_host, s);
    try testReadF32(got, got_host, s);
    var bad: usize = 0;
    var first: usize = 0;
    for (ref_host, got_host, 0..) |r, g, i| {
        // NaN inputs produce NaN on both sides; compare them as bit patterns.
        const same = (std.math.isNan(r) and std.math.isNan(g)) or r == g;
        if (!same) {
            if (bad == 0) first = i;
            bad += 1;
        }
    }
    if (bad != 0) {
        std.debug.print("swiglu mismatch on {d}/{d} bf16 values; first at 0x{X:0>4}: ref={d} got={d}\n", .{ bad, N, first, ref_host[first], got_host[first] });
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "fused SwiGLU declines mismatched shapes rather than reading out of bounds" {
    const s = mlx.gpuStream();
    fused_activations.swiglu_fused_override = true;
    defer fused_activations.swiglu_fused_override = null;
    const g_sh = [_]c_int{ 4, 8 };
    const u_sh = [_]c_int{ 4, 4 };
    var gbuf: [32]f32 = @splat(0.5);
    var ubuf: [16]f32 = @splat(0.25);
    const g32 = mlx.mlx_array_new_data(&gbuf, &g_sh, 2, .float32);
    defer _ = mlx.mlx_array_free(g32);
    const u32a = mlx.mlx_array_new_data(&ubuf, &u_sh, 2, .float32);
    defer _ = mlx.mlx_array_free(u32a);
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_astype(&g, g32, .bfloat16, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_astype(&u, u32a, .bfloat16, s));
    try testing.expect((try fusedSwiGLU(s, g, u)) == null);
    // f32 is declined outright — see the dtype gate's comment.
    try testing.expect((try fusedSwiGLU(s, g32, g32)) == null);
}

test "gpt_oss clamped SwiGLU matches the reference formula (f64 oracle)" {
    // Reference (mlx-lm gpt_oss.swiglu):
    //   x_glu    = clip(gate, max=limit)          — ASYMMETRIC: no lower clip
    //   x_linear = clip(up,  -limit, +limit)
    //   out      = x_glu * sigmoid(alpha*x_glu) * (x_linear + 1)
    // The three easy things to get wrong are all covered by the sweep below:
    // clipping gate on BOTH sides, forgetting the `+ 1` on the linear branch,
    // and applying alpha to the product rather than inside the sigmoid.
    const s = mlx.gpuStream();
    const t = std.testing;
    const limit: f32 = 7.0;
    const alpha: f32 = 1.702;

    // Values that straddle both clip boundaries in both directions.
    const vals = [_]f32{ -20.0, -8.5, -7.0, -3.25, -0.5, 0.0, 0.5, 3.25, 7.0, 8.5, 20.0 };
    const n: c_int = @intCast(vals.len * vals.len);
    var gate_h = try t.allocator.alloc(f32, vals.len * vals.len);
    defer t.allocator.free(gate_h);
    var up_h = try t.allocator.alloc(f32, vals.len * vals.len);
    defer t.allocator.free(up_h);
    for (vals, 0..) |g, i| {
        for (vals, 0..) |u, j| {
            gate_h[i * vals.len + j] = g;
            up_h[i * vals.len + j] = u;
        }
    }
    const shape = [_]c_int{n};
    const gate = mlx.mlx_array_new_data(gate_h.ptr, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(gate);
    const up = mlx.mlx_array_new_data(up_h.ptr, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(up);

    const out = try gptOssSwiGLU(s, gate, up, limit, alpha);
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    const got: [*]const f32 = @ptrCast(@alignCast(mlx.mlx_array_data_float32(out)));

    for (gate_h, up_h, 0..) |g, u, i| {
        const gd: f64 = @min(@as(f64, g), @as(f64, limit));
        const ud: f64 = @max(@min(@as(f64, u), @as(f64, limit)), -@as(f64, limit));
        const sig: f64 = 1.0 / (1.0 + @exp(-@as(f64, alpha) * gd));
        const want: f64 = gd * sig * (ud + 1.0);
        try t.expectApproxEqAbs(want, @as(f64, got[i]), 1e-4);
    }
}

test "the SwiGLU sigmoid table is MLX's sigmoid on the device that asks for it" {
    // CPU and GPU sigmoids round differently; a table cached from one device served the other, so a model's
    // output on one stream depended on which stream had asked first.
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    for ([_]mlx.mlx_stream{ cpu, gpu, cpu }) |s| for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
        const vals = try all16BitPatterns(s, dt, testing.allocator);
        defer _ = mlx.mlx_array_free(vals);
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_sigmoid(&want, vals, s));
        try mlx.check(mlx.mlx_array_eval(want));
        // Compared as bits: the NaN patterns are part of the table.
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_view(&g, try swigluSigTable(s, dt, testing.allocator), .uint16, cpu));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        try mlx.check(mlx.mlx_view(&w, want, .uint16, cpu));
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, g, w, false, cpu));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try testing.expect(same);
    };
}
