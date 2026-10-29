//! `hc_cuda.zig`'s tests, out of its import graph: a plugin unit that reaches `hc_cuda.zig` never compiles them.

const hc_cuda = @import("hc_cuda.zig");
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const testing = std.testing;
const Bag = hc_cuda.Bag;
const chain = hc_cuda.chain;
const expectClose = hc_cuda.expectClose;
const randArr = hc_cuda.randArr;
const rmsDiff = hc_cuda.rmsDiff;

test "hc_cuda: the fused read is no worse than the composed chain against f32 truth (dense bf16 and 8-bit, inject, pending write)" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const xfm = @import("../../../engines/mlx/transformer.zig");
    const s = mlx.gpuStream();
    const hc: c_int = 4;
    const h: c_int = 2560;
    const r: c_int = 320;
    const k = hc * h;
    const eps: f32 = 1e-6;
    for ([_]u32{ 16, 8 }) |bits| {
        for ([_]bool{ false, true }) |pending| {
            var b: Bag = .{};
            defer b.free();
            var prng = std.Random.DefaultPrng.init(0xC0DA + bits + @intFromBool(pending));
            const rnd = prng.random();
            const x = b.keep(try randArr(rnd, &.{ 1, 1, k }, 4.0, 0.0, .bfloat16, s));
            const nw = b.keep(try randArr(rnd, &.{ hc, h }, 1.0, 1.0, .bfloat16, s));
            const iw = b.keep(try randArr(rnd, &.{ k, hc }, 0.02, 0.0, .bfloat16, s));
            const wo = b.keep(try randArr(rnd, &.{ 1, 1, h }, 2.0, 0.0, .bfloat16, s));
            const wi = b.keep(try randArr(rnd, &.{ 1, 1, hc, 1 }, 1.0, 1.0, .bfloat16, s));
            var down = b.keep(try randArr(rnd, &.{ r, k }, 0.04, 0.0, .bfloat16, s));
            var up = b.keep(try randArr(rnd, &.{ k, r }, 0.2, 0.0, .bfloat16, s));
            var dq: [3]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null } };
            var uq: [3]mlx.mlx_array = dq;
            if (bits == 16) {
                for ([_]*mlx.mlx_array{ &dq[0], &uq[0] }, [_]mlx.mlx_array{ down, up }) |dst, src| {
                    dst.* = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_view(dst, src, .uint32, s));
                    _ = b.keep(dst.*);
                }
            } else {
                // The chain and the truth read the dequantized weights; the fused read the packed ones.
                for ([_]*[3]mlx.mlx_array{ &dq, &uq }, [_]*mlx.mlx_array{ &down, &up }) |dst, src| {
                    var tri = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(tri);
                    try mlx.check(mlx.mlx_quantize(&tri, src.*, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
                    for (dst, 0..) |*a, i| {
                        a.* = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_vector_array_get(a, tri, i));
                        _ = b.keep(a.*);
                    }
                    var deq = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_dequantize(&deq, dst[0], dst[1], dst[2], mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }, s));
                    src.* = b.keep(deq);
                }
            }
            const pend: ?xfm.HcPending = if (pending) .{ .out = wo, .inj = wi } else null;
            const before = hc_cuda.served;
            const got = (try xfm.hcReadFused(s, x, 1, 1, nw, dq[0], dq[1], dq[2], uq[0], uq[1], uq[2], iw, eps, hc, h, bits, if (bits == 16) 16 else 64, pend)) orelse return error.HcFusedDeclined;
            _ = b.keep(got.mixed);
            _ = b.keep(got.inj);
            if (got.stream.ctx != null) _ = b.keep(got.stream);
            try testing.expectEqual(before + 1, hc_cuda.served);
            try testing.expect((got.stream.ctx != null) == pending);
            const truth = try chain(&b, s, .float32, x, nw, down, up, iw, if (pending) wo else null, if (pending) wi else null, hc, h, eps);
            const comp = try chain(&b, s, .bfloat16, x, nw, down, up, iw, if (pending) wo else null, if (pending) wi else null, hc, h, eps);
            const names = [_][]const u8{ "mixed", "inj", "stream" };
            const fused = [_]mlx.mlx_array{ got.mixed, got.inj, got.stream };
            for (0..3) |i| {
                if (fused[i].ctx == null) continue;
                try expectClose(&b, s, fused[i], comp[i]);
                if (mlx.mlx_array_size(fused[i]) < 1024) continue;
                const ef = try rmsDiff(&b, s, fused[i], truth[i]);
                const ec = try rmsDiff(&b, s, comp[i], truth[i]);
                std.debug.print("hc_cuda bits={d} pending={} {s}: fused rms err {e:.3} chain {e:.3}\n", .{ bits, pending, names[i], ef, ec });
                try testing.expect(ef <= ec * 1.15 + 1e-7);
            }
        }
    }
}

test "hc_cuda: a multi-row read matches each row read alone, bit for bit" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const xfm = @import("../../../engines/mlx/transformer.zig");
    const s = mlx.gpuStream();
    const hc: c_int = 4;
    const h: c_int = 2560;
    const r: c_int = 320;
    const k = hc * h;
    const rows: c_int = 3;
    var b: Bag = .{};
    defer b.free();
    var prng = std.Random.DefaultPrng.init(0xC0DB);
    const rnd = prng.random();
    const x = b.keep(try randArr(rnd, &.{ 1, rows, k }, 4.0, 0.0, .bfloat16, s));
    const nw = b.keep(try randArr(rnd, &.{ hc, h }, 1.0, 1.0, .bfloat16, s));
    const iw = b.keep(try randArr(rnd, &.{ k, hc }, 0.02, 0.0, .bfloat16, s));
    const wo = b.keep(try randArr(rnd, &.{ 1, rows, h }, 2.0, 0.0, .bfloat16, s));
    const wi = b.keep(try randArr(rnd, &.{ 1, rows, hc, 1 }, 1.0, 1.0, .bfloat16, s));
    var dw = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_view(&dw, b.keep(try randArr(rnd, &.{ r, k }, 0.04, 0.0, .bfloat16, s)), .uint32, s));
    var uw = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_view(&uw, b.keep(try randArr(rnd, &.{ k, r }, 0.2, 0.0, .bfloat16, s)), .uint32, s));
    _ = b.keep(dw);
    _ = b.keep(uw);
    const none: mlx.mlx_array = .{ .ctx = null };
    const all = (try xfm.hcReadFused(s, x, 1, rows, nw, dw, none, none, uw, none, none, iw, 1e-6, hc, h, 16, 16, .{ .out = wo, .inj = wi })) orelse return error.HcFusedDeclined;
    _ = b.keep(all.mixed);
    _ = b.keep(all.inj);
    _ = b.keep(all.stream);
    var row: c_int = 0;
    while (row < rows) : (row += 1) {
        const Slice = struct {
            fn at(bb: *Bag, st: mlx.mlx_stream, a: mlx.mlx_array, rw: c_int) !mlx.mlx_array {
                const sh = mlx.getShape(a);
                var start: [4]c_int = @splat(0);
                var stop: [4]c_int = @splat(1);
                @memcpy(stop[0..sh.len], sh);
                start[1] = rw;
                stop[1] = rw + 1;
                var o = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&o, a, &start, sh.len, &stop, sh.len, &[_]c_int{ 1, 1, 1, 1 }, sh.len, st));
                return bb.keep(o);
            }
        };
        const one = (try xfm.hcReadFused(s, try Slice.at(&b, s, x, row), 1, 1, nw, dw, none, none, uw, none, none, iw, 1e-6, hc, h, 16, 16, .{ .out = try Slice.at(&b, s, wo, row), .inj = try Slice.at(&b, s, wi, row) })) orelse return error.HcFusedDeclined;
        _ = b.keep(one.mixed);
        _ = b.keep(one.inj);
        _ = b.keep(one.stream);
        const pairs = [_][2]mlx.mlx_array{ .{ all.mixed, one.mixed }, .{ all.inj, one.inj }, .{ all.stream, one.stream } };
        for (pairs) |p| {
            var eq = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_equal(&eq, try Slice.at(&b, s, p[0], row), p[1], false, s));
            _ = b.keep(eq);
            var same = false;
            try mlx.check(mlx.mlx_array_eval(eq));
            try mlx.check(mlx.mlx_array_item_bool(&same, eq));
            try testing.expect(same);
        }
    }
}
