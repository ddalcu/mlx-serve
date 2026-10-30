//! `verify_projection.zig`'s tests, out of its import graph: a plugin unit that reaches `verify_projection.zig` never compiles them.

const verify_projection = @import("verify_projection.zig");
const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const verify_qmm = @import("verify_qmm.zig");
const testing = std.testing;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const qmvBatchLimit = @import("qmatmul.zig").qmvBatchLimit;
const msvQmvRows = @import("qmv_rows.zig").msvQmvRows;
const dualDieArch = verify_projection.dualDieArch;
const mtpCoarsePairs = verify_projection.mtpCoarsePairs;
const sharedGateRowsEligible = verify_projection.sharedGateRowsEligible;
const verifyArch = verify_projection.verifyArch;
const verifyAttnProjection = verify_projection.verifyAttnProjection;
const verifyGdnProjection = verify_projection.verifyGdnProjection;
const verifyLmHeadProjection = verify_projection.verifyLmHeadProjection;
const verifySharedFor = verify_projection.verifySharedFor;
const verifySharedHardware = verify_projection.verifySharedHardware;
const verifySharedProjection = verify_projection.verifySharedProjection;
const verifyWideProjection = verify_projection.verifyWideProjection;
const mtp_mod = @import("../../spec/mtp/mtp.zig");
const qmatmulBits = @import("qmatmul.zig").qmatmulBits;
const model_mod = @import("../../weights/load.zig");
const qsaArraysAllEqual = @import("../../testing/parity.zig").qsaArraysAllEqual;

test "shared expert gate tail admits the joined verifier envelope only" {
    try testing.expect(sharedGateRowsEligible(&.{ 2, 2, 2, 2 }));
    try testing.expect(sharedGateRowsEligible(&.{ 5, 5, 5, 5 }));
    try testing.expect(sharedGateRowsEligible(&.{ 2, 3 }));
    try testing.expect(!sharedGateRowsEligible(&.{ 7, 7, 7, 7 }));
    try testing.expect(!sharedGateRowsEligible(&.{1}));
}

test "dualDieArch: only the Ultra's suffix" {
    try std.testing.expect(dualDieArch("applegpu_g17d"));
    try std.testing.expect(!dualDieArch("applegpu_g17s"));
    try std.testing.expect(!dualDieArch("applegpu_g17g"));
    try std.testing.expect(!dualDieArch(""));
}

test "verifySharedFor: dual-die M5 takes the fused verify kernels only for a multi-stream group" {
    try std.testing.expect(verifySharedFor("applegpu_g17s", 0));
    try std.testing.expect(verifySharedFor("applegpu_g17s", 4));
    try std.testing.expect(!verifySharedFor("applegpu_g17d", 0));
    try std.testing.expect(!verifySharedFor("applegpu_g17d", 1));
    try std.testing.expect(verifySharedFor("applegpu_g17d", 2));
    try std.testing.expect(verifySharedFor("applegpu_g17d", 4));
    try std.testing.expect(!verifySharedFor("applegpu_g16d", 4));
    try std.testing.expect(!verifySharedFor("applegpu_g16s", 0));
}

test "verify shared projection preserves solo qmv values across request widths, at g64 and g128" {
    // A two-stream group: dual-die M5 takes the joined kernels only then.
    const outer_group_rows = verify_projection.verify_group_rows;
    verify_projection.verify_group_rows = 2;
    defer verify_projection.verify_group_rows = outer_group_rows;
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    for ([_]u32{ 64, 128 }) |gs| for ([_][2]c_int{ .{ 2560, 640 }, .{ 640, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            const arch = verifyArch().?;
            if (total > 24 or total >= qmvBatchLimit(dualDieArch(arch), k, n)) {
                try testing.expect((try verifySharedProjection(s, x, w, sc, bi, lengths[0..count])) == null);
                continue;
            }
            const grouped = (try verifySharedProjection(s, x, w, sc, bi, lengths[0..count])) orelse return error.SharedProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, gs, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifySharedProjection(s, x, w, sc, bi, widths)) == null);
        }
        try testing.expect((try verifySharedProjection(s, x, w, sc, .{}, &.{ 4, 4 })) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifySharedProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 })) == null);
        const old = verify_qmm.vqmm_mixed_plain_test_force;
        verify_qmm.vqmm_mixed_plain_test_force = true;
        defer verify_qmm.vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifySharedProjection(s, x, w, sc, bi, &.{ 4, 4 })) == null);
    };
}

test "verify GDN projections preserve solo qmv values across request widths" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    for ([_][2]c_int{ .{ 2560, 10240 }, .{ 2560, 6144 }, .{ 2560, 48 }, .{ 6144, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 12 or lengths[0] > 6) {
                try testing.expect((try verifyGdnProjection(s, x, w, sc, bi, lengths[0..count])) == null);
                continue;
            }
            const grouped = (try verifyGdnProjection(s, x, w, sc, bi, lengths[0..count])) orelse return error.GdnProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifyGdnProjection(s, x, w, sc, bi, widths)) == null);
        }
        try testing.expect((try verifyGdnProjection(s, x, w, sc, .{}, &.{ 4, 4 })) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifyGdnProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 })) == null);
        const old = verify_qmm.vqmm_mixed_plain_test_force;
        verify_qmm.vqmm_mixed_plain_test_force = true;
        defer verify_qmm.vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifyGdnProjection(s, x, w, sc, bi, &.{ 4, 4 })) == null);
    }
}

test "verify attention projections preserve solo qmv values across request widths" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    for ([_][2]c_int{ .{ 2560, 12288 }, .{ 2560, 512 }, .{ 6144, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 12 or lengths[0] > 6) {
                try testing.expect((try verifyAttnProjection(s, x, w, sc, bi, lengths[0..count])) == null);
                continue;
            }
            const grouped = (try verifyAttnProjection(s, x, w, sc, bi, lengths[0..count])) orelse return error.AttnProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifyAttnProjection(s, x, w, sc, bi, widths)) == null);
        }
        try testing.expect((try verifyAttnProjection(s, x, w, sc, .{}, &.{ 4, 4 })) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifyAttnProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 })) == null);
        const old = verify_qmm.vqmm_mixed_plain_test_force;
        verify_qmm.vqmm_mixed_plain_test_force = true;
        defer verify_qmm.vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifyAttnProjection(s, x, w, sc, bi, &.{ 4, 4 })) == null);
    }
}

test "verify vocabulary projections preserve solo logits across request widths (MTP_LMHEAD_TEST_FILE)" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    const path = env.get(.MTP_LMHEAD_TEST_FILE) orelse return error.SkipZigTest;
    var weights = try model_mod.loadWeightsSingleFile(testing.allocator, std.mem.span(path));
    defer weights.deinit();
    const w = weights.get("language_model.lm_head.weight") orelse return error.MissingLmHead;
    const sc = weights.get("language_model.lm_head.scales") orelse return error.MissingLmHead;
    const bi = weights.get("language_model.lm_head.biases") orelse return error.MissingLmHead;
    const k: c_int = 2560;
    const n: c_int = 248320;
    {
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 12 or lengths[0] > 6) {
                try testing.expect((try verifyLmHeadProjection(s, x, w, sc, bi, lengths[0..count])) == null);
                continue;
            }
            const grouped = (try verifyLmHeadProjection(s, x, w, sc, bi, lengths[0..count])) orelse return error.LmHeadProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifyLmHeadProjection(s, x, w, sc, bi, widths)) == null);
        }
        try testing.expect((try verifyLmHeadProjection(s, x, w, sc, .{}, &.{ 4, 4 })) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifyLmHeadProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 })) == null);
        const old = verify_qmm.vqmm_mixed_plain_test_force;
        verify_qmm.vqmm_mixed_plain_test_force = true;
        defer verify_qmm.vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifyLmHeadProjection(s, x, w, sc, bi, &.{ 4, 4 })) == null);
    }
}

test "the wide dense verify tile preserves the stock QMV reduction" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x719E);
    for ([_][2]c_int{ .{ 2560, 10240 }, .{ 2560, 6144 }, .{ 6144, 2560 }, .{ 640, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weights = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weights);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weights, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]c_int{ 6, 8, 12 }) |m| {
            const x = try attn256RandBf16(prng.random(), &.{ 1, m, k }, s);
            defer _ = mlx.mlx_array_free(x);
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try verifyWideProjection(cpu, x, w, sc, bi, 6)) == null);
            try testing.expect((try verifyWideProjection(s, .{}, w, sc, bi, 6)) == null);
            try testing.expect((try verifyWideProjection(s, x, w, sc, .{}, 6)) == null);
            try testing.expect((try verifyWideProjection(s, x, w, sc, bi, 7)) == null);
            var reference = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(reference);
            try mlx.check(mlx.mlx_quantized_matmul(&reference, x, w, sc, bi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", s));
            for ([_]c_int{ 4, 6, 8, 12 }) |nv| {
                const got = (try verifyWideProjection(s, x, w, sc, bi, nv)) orelse return error.WideTileDeclined;
                defer _ = mlx.mlx_array_free(got);
                try testing.expect(try qsaArraysAllEqual(reference, got, s));
            }
        }
    }
}

test "MTP coarse pairs preserve row-kernel logits (MTP_LMHEAD_TEST_FILE)" {
    const path = env.get(.MTP_LMHEAD_TEST_FILE) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var weights = try model_mod.loadWeightsSingleFile(testing.allocator, std.mem.span(path));
    defer weights.deinit();
    const w = weights.get("language_model.lm_head.weight") orelse return error.MissingLmHead;
    const sc = weights.get("language_model.lm_head.scales") orelse return error.MissingLmHead;
    const bi = weights.get("language_model.lm_head.biases") orelse return error.MissingLmHead;
    var coarse = try mtp_mod.requantizeRows(s, w, sc, bi, 64, 8, "affine", 64, 3, 32768);
    defer coarse.deinit();
    var prng = std.Random.DefaultPrng.init(0xC0A45E);
    for ([_]c_int{ 2, 3, 4, 8 }) |rows| {
        const x = try attn256RandBf16(prng.random(), &.{ rows, 1, 2560 }, s);
        defer _ = mlx.mlx_array_free(x);
        const reference = (try msvQmvRows(s, x, coarse.w, coarse.s, coarse.b, 3, 64)) orelse return error.HeadRowsDeclined;
        defer _ = mlx.mlx_array_free(reference);
        const actual = (try mtpCoarsePairs(s, x, coarse.w, coarse.s, coarse.b, 3, 64)) orelse return error.CoarsePairsDeclined;
        defer _ = mlx.mlx_array_free(actual);
        try testing.expect(try qsaArraysAllEqual(reference, actual, s));
        if (rows == 4) {
            try testing.expect((try mtpCoarsePairs(s, x, coarse.w, coarse.s, .{}, 3, 64)) == null);
            try testing.expect((try mtpCoarsePairs(s, x, coarse.w, coarse.s, coarse.b, 2, 64)) == null);
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try mtpCoarsePairs(cpu, x, coarse.w, coarse.s, coarse.b, 3, 64)) == null);
            var wide = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wide);
            try mlx.check(mlx.mlx_reshape(&wide, x, &.{ 2, 2, 2560 }, 3, s));
            try testing.expect((try mtpCoarsePairs(s, wide, coarse.w, coarse.s, coarse.b, 3, 64)) == null);
            var x32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x32);
            try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
            try testing.expect((try mtpCoarsePairs(s, x32, coarse.w, coarse.s, coarse.b, 3, 64)) == null);
        }
    }
}
