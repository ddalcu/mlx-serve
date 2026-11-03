//! `qmv_rows.zig`'s tests, out of its import graph: a plugin unit that reaches `qmv_rows.zig` never compiles them.

const qmv_rows = @import("qmv_rows.zig");
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const Weights = @import("../../weights/resident.zig").Weights;
const testing = std.testing;
const firstRowExactMiss = @import("../../quant/ternary.zig").firstRowExactMiss;
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const moeRowsSliceRow = @import("../moe/gather_qmv_rows.zig").moeRowsSliceRow;
const msvQmvRows = qmv_rows.msvQmvRows;
const ModelConfig = @import("../../config/model.zig").ModelConfig;

test "firstRowExactMiss: 4, 6 and 8-bit trunk projections are inside the row kernels, 3-bit is not; vision and mtp never count" {
    const Q = struct {
        fn put(w: *Weights, base: []const u8, bits: c_int) !void {
            const st = mlx.gpuStream();
            var zeros = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(zeros);
            try mlx.check(mlx.mlx_zeros(&zeros, &[_]c_int{ 64, 128 }, 2, .bfloat16, st));
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, zeros, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(bits), "affine", .{}, st));
            for ([_][]const u8{ ".weight", ".scales", ".biases" }, 0..) |suf, i| {
                var a = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(&a, triple, i));
                try w.map.put(try std.fmt.allocPrint(std.testing.allocator, "{s}{s}", .{ base, suf }), a);
            }
        }
    };
    const cfg: @import("../../quant/params.zig").QuantParams = .{ .bits = 4, .group_size = 64, .mode = .affine };
    var w = Weights.init(std.testing.allocator);
    defer w.deinit();
    try Q.put(&w, "m.layers.0.q", 4);
    try Q.put(&w, "m.layers.1.q", 6);
    try Q.put(&w, "m.layers.2.q", 8);
    try Q.put(&w, "vision_tower.fc", 3);
    try Q.put(&w, "mtp.layers.0.fc", 3);
    try std.testing.expectEqual(@as(?[]const u8, null), firstRowExactMiss(cfg, &w, .{ .ctx = null }));
    try Q.put(&w, "m.layers.3.q", 3);
    try std.testing.expectEqualStrings("m.layers.3.q.scales", firstRowExactMiss(cfg, &w, .{ .ctx = null }).?);
}

test "msv_qmv_rows is mlx_equal per row to stock M=1 qmv at every head shape, width and group" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    std.debug.print("[msv_qmv_rows] hermetic start\n", .{});
    var prng = std.Random.DefaultPrng.init(0x51500A01);
    const rnd = prng.random();
    const shapes = [_][2]c_int{
        .{ 2560, 2560 },
        .{ 2560, 12288 },
        .{ 2560, 512 },
        .{ 12288, 2560 },
        .{ 2560, 640 },
        .{ 2560, 248320 },
    };
    const bitss = [_]u32{ 3, 4, 5, 6, 8 };
    const ns = [_]c_int{ 2, 4, 8, 16 };
    for ([_]u32{ 64, 128 }) |gs| for (bitss) |bits| {
        for (shapes) |kn| {
            const K = kn[0];
            const Nout = kn[1];
            const w_bf = try testRandWeightBf16(rnd, &[_]c_int{ Nout, K }, s);
            defer _ = mlx.mlx_array_free(w_bf);
            var qvec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(qvec);
            try mlx.check(mlx.mlx_quantize(&qvec, w_bf, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, s));
            var w = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w);
            var sc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sc);
            var bi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bi);
            try mlx.check(mlx.mlx_vector_array_get(&w, qvec, 0));
            try mlx.check(mlx.mlx_vector_array_get(&sc, qvec, 1));
            try mlx.check(mlx.mlx_vector_array_get(&bi, qvec, 2));
            for (ns) |nrows| {
                const x = try attn256RandBf16(rnd, &[_]c_int{ nrows, K }, s);
                defer _ = mlx.mlx_array_free(x);
                const got = (try msvQmvRows(s, x, w, sc, bi, bits, gs)) orelse {
                    std.debug.print("[msv_qmv_rows] declined bits={d} K={d} N={d} rows={d}\n", .{ bits, K, Nout, nrows });
                    return error.QmvRowsDeclined;
                };
                defer _ = mlx.mlx_array_free(got);
                var r: c_int = 0;
                while (r < nrows) : (r += 1) {
                    const xr = try moeRowsSliceRow(s, x, r);
                    defer _ = mlx.mlx_array_free(xr);
                    var xr2 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(xr2);
                    try mlx.check(mlx.mlx_reshape(&xr2, xr, &[_]c_int{ 1, K }, 2, s));
                    var want = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(want);
                    try mlx.check(mlx.mlx_quantized_matmul(
                        &want,
                        xr2,
                        w,
                        sc,
                        bi,
                        true,
                        mlx.mlx_optional_int.some(@intCast(gs)),
                        mlx.mlx_optional_int.some(@intCast(bits)),
                        "affine",
                        s,
                    ));
                    const gr = try moeRowsSliceRow(s, got, r);
                    defer _ = mlx.mlx_array_free(gr);
                    const diff = try attn256MaxDiff(gr, want, s);
                    if (diff != 0) {
                        std.debug.print("[msv_qmv_rows] NE bits={d} K={d} N={d} rows={d} row={d} maxdiff={d}\n", .{ bits, K, Nout, nrows, r, diff });
                    }
                    try testing.expectEqual(@as(f32, 0.0), diff);
                }
                std.debug.print("[msv_qmv_rows] eq bits={d} gs={d} K={d} N={d} rows={d}\n", .{ bits, gs, K, Nout, nrows });
            }
        }
    };
}

test "row kernel declines a CPU stream" {
    const s = mlx.gpuStream();
    var fx = try @import("../../../engines/mlx/spec/mtp_test.zig").RerankFixture.init(s, 512, 2560, 4, 64, 11);
    defer fx.deinit();
    var prng = std.Random.DefaultPrng.init(12);
    const x = try attn256RandBf16(prng.random(), &.{ 2, 2560 }, s);
    defer _ = mlx.mlx_array_free(x);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const got = try msvQmvRows(cpu, x, fx.xfm.lm_head_w, fx.xfm.lm_head_s, fx.xfm.lm_head_b, 4, 64);
    defer if (got) |y| {
        _ = mlx.mlx_array_free(y);
    };
    try testing.expect(got == null);
}

test "row kernel declines scales and biases with another dtype" {
    const s = mlx.gpuStream();
    var fx = try @import("../../../engines/mlx/spec/mtp_test.zig").RerankFixture.init(s, 512, 2560, 4, 64, 11);
    defer fx.deinit();
    var prng = std.Random.DefaultPrng.init(12);
    const x = try attn256RandBf16(prng.random(), &.{ 2, 2560 }, s);
    defer _ = mlx.mlx_array_free(x);
    var sc32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc32);
    var bi32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi32);
    try mlx.check(mlx.mlx_astype(&sc32, fx.xfm.lm_head_s, .float32, s));
    try mlx.check(mlx.mlx_astype(&bi32, fx.xfm.lm_head_b, .float32, s));
    for ([_][2]mlx.mlx_array{ .{ sc32, fx.xfm.lm_head_b }, .{ fx.xfm.lm_head_s, bi32 } }) |pair| {
        const got = try msvQmvRows(s, x, fx.xfm.lm_head_w, pair[0], pair[1], 4, 64);
        defer if (got) |y| {
            _ = mlx.mlx_array_free(y);
        };
        try testing.expect(got == null);
    }
}

test "row kernel cache reuses alternating full-shape configurations" {
    const s = mlx.gpuStream();
    var fx = try @import("../../../engines/mlx/spec/mtp_test.zig").RerankFixture.init(s, 512, 2560, 8, 64, 11);
    defer fx.deinit();
    var prng = std.Random.DefaultPrng.init(12);
    const before = qmv_rows.msv_qmv_rows_cfgs.builds;
    for ([_]c_int{ 3, 5, 3 }) |rows| {
        const x = try attn256RandBf16(prng.random(), &.{ rows, 2560 }, s);
        defer _ = mlx.mlx_array_free(x);
        const y = (try msvQmvRows(s, x, fx.xfm.lm_head_w, fx.xfm.lm_head_s, fx.xfm.lm_head_b, 8, 64)) orelse return error.HeadRowsDeclined;
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
    }
    try testing.expectEqual(@as(usize, 2), qmv_rows.msv_qmv_rows_cfgs.builds - before);
}
