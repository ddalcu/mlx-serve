//! The Laguna-XS decode µbench (diag `laguna-ubench`): the dense bf16 attention GEMVs at the real shapes.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const testing = std.testing;
const transposeBf16Weight = @import("../../engines/mlx/load/weights.zig").transposeBf16Weight;

test "Laguna-XS decode µbench: bf16 attention projections (diag laguna-ubench)" {
    // Laguna XS 2.1 NVFP4 keeps attention, lm_head and the layer-0 MLP in BF16
    // and quantizes only the expert projections, so ~78% of the per-token DRAM
    // budget is DENSE bf16 GEMV — not the MoE. This measures the achievable
    // rate for that dominant term at the real shapes.
    //
    // CRITICAL: every layer gets its OWN weight buffers. An earlier version of
    // this bench reused four buffers across all 40 layers and reported
    // "3.26 GB at 449 GB/s"; it was re-reading ~100 MB of resident data 25x and
    // measuring CACHE bandwidth, not DRAM. A decode step reads each of the 40
    // layers' weights exactly ONCE from DRAM, so the bench must too — otherwise
    // it manufactures a bandwidth floor far below the real one and makes the
    // live path look pathological by comparison.
    if (!env.on(.LAGUNA_UBENCH)) return error.SkipZigTest;
    const io_util = @import("../../sdk/util/io_util.zig");
    const tio = testing.io;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const HID: c_int = 2048;
    const SLIDING_LAYERS: usize = 30; // 64 Q heads
    const FULL_LAYERS: usize = 10; //    48 Q heads
    const VOCAB: c_int = 100352;

    // bf16 weight [out, in], pre-transposed to [in, out] exactly like
    // `maybeTransposeForBf16` does at load — the live path's layout.
    const makeW = struct {
        fn go(a: std.mem.Allocator, out: c_int, in: c_int, str: mlx.mlx_stream, rndp: *std.Random.DefaultPrng) !mlx.mlx_array {
            const cnt: usize = @intCast(out * in);
            const buf = try a.alloc(f32, cnt);
            defer a.free(buf);
            const rr = rndp.random();
            for (buf) |*v| v.* = rr.float(f32) - 0.5;
            const sh = [_]c_int{ out, in };
            const w32 = mlx.mlx_array_new_data(buf.ptr, &sh, 2, .float32);
            defer _ = mlx.mlx_array_free(w32);
            var wb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wb);
            try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, str));
            const t = try transposeBf16Weight(wb, str);
            // Materialize so the view's parent can be freed and the layer owns
            // one contiguous buffer, matching a loaded checkpoint's residency.
            try mlx.check(mlx.mlx_array_eval(t));
            return t;
        }
    }.go;

    var prng = std.Random.DefaultPrng.init(0xA5A5);
    const NL = SLIDING_LAYERS + FULL_LAYERS;
    const qs = try allocator.alloc(mlx.mlx_array, NL);
    defer allocator.free(qs);
    const os_ = try allocator.alloc(mlx.mlx_array, NL);
    defer allocator.free(os_);
    const ks = try allocator.alloc(mlx.mlx_array, NL);
    defer allocator.free(ks);
    const vs = try allocator.alloc(mlx.mlx_array, NL);
    defer allocator.free(vs);
    defer for (0..NL) |i| {
        _ = mlx.mlx_array_free(qs[i]);
        _ = mlx.mlx_array_free(os_[i]);
        _ = mlx.mlx_array_free(ks[i]);
        _ = mlx.mlx_array_free(vs[i]);
    };
    for (0..NL) |i| {
        const heads: c_int = if (i < SLIDING_LAYERS) 64 else 48;
        qs[i] = try makeW(allocator, heads * 128, HID, s, &prng);
        os_[i] = try makeW(allocator, HID, heads * 128, s, &prng);
        ks[i] = try makeW(allocator, 8 * 128, HID, s, &prng);
        vs[i] = try makeW(allocator, 8 * 128, HID, s, &prng);
    }
    const lm = try makeW(allocator, VOCAB, HID, s, &prng);
    defer _ = mlx.mlx_array_free(lm);

    const xsh = [_]c_int{ 1, 1, HID };
    const xbuf = try allocator.alloc(f32, @intCast(HID));
    defer allocator.free(xbuf);
    for (xbuf) |*v| v.* = prng.random().float(f32) - 0.5;
    const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 3, .float32);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

    var bytes_total: f64 = 0;
    for (0..NL) |i| {
        const heads: f64 = if (i < SLIDING_LAYERS) 64 else 48;
        bytes_total += heads * 128 * 2048 * 2 * 2; // q + o
        bytes_total += 8 * 128 * 2048 * 2 * 2; // k + v
    }
    bytes_total += @as(f64, VOCAB) * 2048 * 2;

    const runToken = struct {
        fn go(str: mlx.mlx_stream, x0: mlx.mlx_array, q: []mlx.mlx_array, o: []mlx.mlx_array, k: []mlx.mlx_array, v: []mlx.mlx_array, l: mlx.mlx_array) !void {
            var h = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h, x0, x0, str));
            defer _ = mlx.mlx_array_free(h);
            for (0..q.len) |i| {
                const qp = try mm(str, h, q[i]);
                defer _ = mlx.mlx_array_free(qp);
                const kp = try mm(str, h, k[i]);
                defer _ = mlx.mlx_array_free(kp);
                const vp = try mm(str, h, v[i]);
                defer _ = mlx.mlx_array_free(vp);
                const op = try mm(str, qp, o[i]);
                defer _ = mlx.mlx_array_free(op);
                // Fold k/v into the residual so MLX cannot drop them as dead
                // code, and rotate h so no layer is a common subexpression.
                var kv_sum = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(kv_sum);
                try mlx.check(mlx.mlx_add(&kv_sum, kp, vp, str));
                var kv_sc = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(kv_sc);
                try mlx.check(mlx.mlx_sum_axis(&kv_sc, kv_sum, 2, true, str));
                var t = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&t, h, op, str));
                var t2 = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&t2, t, kv_sc, str));
                _ = mlx.mlx_array_free(t);
                _ = mlx.mlx_array_free(h);
                h = t2;
            }
            const logits = try mm(str, h, l);
            defer _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(logits));
        }
        fn mm(str: mlx.mlx_stream, a: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_matmul(&r, a, w, str));
            return r;
        }
    }.go;

    for (0..3) |_| try runToken(s, x, qs, os_, ks, vs, lm);
    const ITER: usize = 20;
    var sw = io_util.Stopwatch.init(tio);
    for (0..ITER) |_| try runToken(s, x, qs, os_, ks, vs, lm);
    const ms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / @as(f64, @floatFromInt(ITER));
    const gbps = bytes_total / (ms / 1000.0) / 1.0e9;
    std.debug.print("\n[laguna-ubench] 40 layers q/k/v/o + lm_head, DISTINCT weights per layer: {d:.3} ms/token ({d:.2} GB, {d:.0} GB/s)\n", .{ ms, bytes_total / 1.0e9, gbps });

    // Per-dispatch latency: a chain of N TRIVIAL dependent ops on a 1-element
    // array. No bytes move, so whatever this costs is pure per-op overhead —
    // the currency a decode step actually spends, since at seq_len 1 almost
    // every op in the model is this small.
    {
        const one_sh = [_]c_int{1};
        const oneval = [_]f32{1.0};
        const seed = mlx.mlx_array_new_data(&oneval, &one_sh, 1, .float32);
        defer _ = mlx.mlx_array_free(seed);
        for ([_]usize{ 200, 1400 }) |chain| {
            const chainOnce = struct {
                fn go(str: mlx.mlx_stream, sd: mlx.mlx_array, n: usize) !void {
                    var a = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&a, sd, sd, str));
                    defer _ = mlx.mlx_array_free(a);
                    for (0..n) |_| {
                        var b = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_add(&b, a, sd, str));
                        _ = mlx.mlx_array_free(a);
                        a = b;
                    }
                    try mlx.check(mlx.mlx_array_eval(a));
                }
            }.go;
            for (0..2) |_| try chainOnce(s, seed, chain);
            var sw2 = io_util.Stopwatch.init(tio);
            const IT: usize = 10;
            for (0..IT) |_| try chainOnce(s, seed, chain);
            const tms = @as(f64, @floatFromInt(sw2.read())) / 1.0e6 / @as(f64, @floatFromInt(IT));
            std.debug.print("[laguna-ubench] {d} trivial DEPENDENT ops: {d:.3} ms => {d:.1} us/dispatch\n", .{ chain, tms, tms * 1000.0 / @as(f64, @floatFromInt(chain)) });
        }
    }
}
