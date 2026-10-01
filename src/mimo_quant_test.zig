const std = @import("std");
const mlx = @import("mlx.zig");
const transformer = @import("transformer.zig");

const testing = std.testing;

const E2M1 = [_]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };

fn e2m1Value(code: u8) f32 {
    const magnitude = E2M1[code & 0x7];
    return if ((code & 0x8) != 0) -magnitude else magnitude;
}

fn e8m0Value(code: u8) f32 {
    if (code == 0) return 0.0;
    if (code == 0xff) return std.math.inf(f32);
    return @exp2(@as(f32, @floatFromInt(@as(i16, code) - 127)));
}

fn bf16Bits(value: f32) u16 {
    return @truncate(@as(u32, @bitCast(value)) >> 16);
}

fn makeArray(data: anytype, shape: []const c_int, dtype: mlx.mlx_dtype) mlx.mlx_array {
    return mlx.mlx_array_new_data(@ptrCast(data.ptr), shape.ptr, @intCast(shape.len), dtype);
}

fn readF32(arr: mlx.mlx_array, out: []f32, s: mlx.mlx_stream) !void {
    var cast = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cast);
    try mlx.check(mlx.mlx_astype(&cast, arr, .float32, s));

    const ev = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_vector_array_append_value(ev, cast));
    try mlx.check(mlx.mlx_eval(ev));
    const data = mlx.mlx_array_data_float32(cast) orelse return error.TestUnexpectedNullData;
    @memcpy(out, data[0..out.len]);
}

/// HF's nativebytes carry two E2M1 nibbles per byte. MLX receives the same
/// bytes as little-endian U32 words, eight values per output row.
fn packNativeBytes(native_bytes: []const u8, packed_words: []u32) void {
    std.debug.assert(native_bytes.len == packed_words.len * 4);
    for (packed_words, 0..) |*word, i| {
        var value: u32 = 0;
        for (0..4) |byte| {
            value |= @as(u32, native_bytes[i * 4 + byte]) << @intCast(byte * 8);
        }
        word.* = value;
    }
}

fn nativeCode(native_bytes: []const u8, row: usize, k: usize, bytes_per_row: usize) u8 {
    const byte = native_bytes[row * bytes_per_row + k / 2];
    return if ((k & 1) == 0) byte & 0xf else byte >> 4;
}

fn expectFiniteApprox(expected: f32, actual: f32) !void {
    try testing.expect(std.math.isFinite(expected));
    try testing.expect(std.math.isFinite(actual));
    try testing.expectApproxEqAbs(expected, actual, 0.02 + @abs(expected) * 0.02);
}

fn hostDot(
    input: []const f32,
    native_bytes: []const u8,
    input_row: usize,
    output_row: usize,
    k: usize,
    bytes_per_row: usize,
    scale_code: u8,
) f32 {
    const scale = e8m0Value(scale_code);
    var sum: f32 = 0;
    for (0..k) |column| {
        const code = nativeCode(native_bytes, output_row, column, bytes_per_row);
        sum += input[input_row * k + column] * scale * e2m1Value(code);
    }
    return sum;
}

fn runSingleScale(stream: mlx.mlx_stream, scale_code: u8, weight_code: u8) !f32 {
    const x_shape = [_]c_int{ 1, 32 };
    var x_bits: [32]u16 = undefined;
    for (&x_bits) |*v| v.* = bf16Bits(1.0);
    const x = makeArray(x_bits[0..], &x_shape, .bfloat16);
    defer _ = mlx.mlx_array_free(x);

    const w_shape = [_]c_int{ 1, 4 };
    var native_bytes: [4 * 4]u8 = undefined;
    for (&native_bytes) |*byte| byte.* = weight_code | (weight_code << 4);
    var weights: [4]u32 = undefined;
    packNativeBytes(native_bytes[0..], weights[0..]);
    const w = makeArray(weights[0..], &w_shape, .uint32);
    defer _ = mlx.mlx_array_free(w);

    const scale_shape = [_]c_int{ 1, 1 };
    const scales = [_]u8{scale_code};
    const scale = makeArray(scales[0..], &scale_shape, .uint8);
    defer _ = mlx.mlx_array_free(scale);

    var native = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(native);
    try mlx.check(mlx.mlx_quantized_matmul(
        &native,
        x,
        w,
        scale,
        .{ .ctx = null },
        true,
        mlx.mlx_optional_int.some(32),
        mlx.mlx_optional_int.some(4),
        "mxfp4",
        stream,
    ));

    var got: [1]f32 = undefined;
    try readF32(native, &got, stream);
    return got[0];
}

test "mimo MXFP4 host reference covers E2M1 and E8M0 special values" {
    const expected_e2m1 = [_]f32{
        0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    };
    for (expected_e2m1, 0..) |expected, code| {
        const actual = e2m1Value(@intCast(code));
        if (code == 8) {
            try testing.expect(std.math.isNegativeZero(actual));
        } else {
            try testing.expectEqual(expected, actual);
        }
    }

    try testing.expectEqual(@as(f32, 0.0), e8m0Value(0));
    try testing.expectEqual(@as(f32, 0.25), e8m0Value(125));
    try testing.expectEqual(@as(f32, 0.5), e8m0Value(126));
    try testing.expectEqual(@as(f32, 1.0), e8m0Value(127));
    try testing.expectEqual(@as(f32, 2.0), e8m0Value(128));
    try testing.expectEqual(@as(f32, 4.0), e8m0Value(129));
    try testing.expect(std.math.isFinite(e8m0Value(254)));
    try testing.expectEqual(@as(f32, 1.7014118346046923e38), e8m0Value(254));
    try testing.expect(std.math.isInf(e8m0Value(255)));
}

test "mimo MXFP4 native scale zero and NaN semantics" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;

    const stream = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(stream);

    try testing.expectEqual(@as(f32, 0.0), try runSingleScale(stream, 0, 0x2));
    const inf = try runSingleScale(stream, 0xff, 0x2);
    try testing.expect(std.math.isInf(inf) and inf > 0);
    const nan = try runSingleScale(stream, 0xff, 0x0);
    try testing.expect(std.math.isNan(nan));
}

test "mimo MXFP4 native quantized_matmul matches independent bf16 dequant" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;

    const M = 3;
    const N = 5;
    const K = 32;
    const packed_cols = K / 8;
    const bytes_per_row = K / 2;
    const stream = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(stream);

    const x_pattern = [_]f32{
        1.0, -1.0, 0.5,  -0.5,  2.0, -2.0, 0.25,  -0.25,
        1.5, -1.5, 0.75, -0.75, 3.0, -3.0, 0.125, -0.125,
    };
    var input: [M * K]f32 = undefined;
    var input_bits: [M * K]u16 = undefined;
    for (&input, 0..) |*value, i| {
        value.* = x_pattern[i % x_pattern.len];
        input_bits[i] = bf16Bits(value.*);
    }

    const x_shape = [_]c_int{ M, K };
    const x = makeArray(input_bits[0..], &x_shape, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(x));

    var native_bytes: [N * bytes_per_row]u8 = undefined;
    for (0..N) |row| {
        for (0..bytes_per_row) |byte| {
            const low: u8 = @intCast((row * 5 + byte * 2) & 0xf);
            const high: u8 = @intCast((row * 5 + byte * 2 + 1) & 0xf);
            native_bytes[row * bytes_per_row + byte] = low | (high << 4);
        }
    }
    var packed_words: [N * packed_cols]u32 = undefined;
    packNativeBytes(native_bytes[0..], packed_words[0..]);
    const w_shape = [_]c_int{ N, packed_cols };
    const w = makeArray(packed_words[0..], &w_shape, .uint32);
    defer _ = mlx.mlx_array_free(w);

    const scale_codes = [_]u8{ 125, 126, 127, 128, 129 };
    const scale_shape = [_]c_int{ N, 1 };
    const scales = makeArray(scale_codes[0..], &scale_shape, .uint8);
    defer _ = mlx.mlx_array_free(scales);

    var native = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(native);
    try mlx.check(mlx.mlx_quantized_matmul(
        &native,
        x,
        w,
        scales,
        .{ .ctx = null },
        true,
        mlx.mlx_optional_int.some(32),
        mlx.mlx_optional_int.some(4),
        "mxfp4",
        stream,
    ));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(native));
    try testing.expectEqualSlices(c_int, &[_]c_int{ M, N }, mlx.getShape(native));

    var got: [M * N]f32 = undefined;
    try readF32(native, got[0..], stream);
    for (0..M) |m| {
        for (0..N) |n| {
            const expected = hostDot(input[0..], native_bytes[0..], m, n, K, bytes_per_row, scale_codes[n]);
            try expectFiniteApprox(expected, got[m * N + n]);
        }
    }
}

test "mimo MXFP4 native gather_qmm matches expert selector permutations" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;

    const E = 3;
    const R = 5;
    const N = 8;
    const K = 32;
    const packed_cols = K / 8;
    const bytes_per_row = K / 2;
    const stream = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(stream);

    const x_pattern = [_]f32{ 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25 };
    var input: [R * K]f32 = undefined;
    var input_bits: [R * K]u16 = undefined;
    for (&input, 0..) |*value, i| {
        value.* = x_pattern[i % x_pattern.len];
        input_bits[i] = bf16Bits(value.*);
    }
    const x_shape = [_]c_int{ R, 1, K };
    const x = makeArray(input_bits[0..], &x_shape, .bfloat16);
    defer _ = mlx.mlx_array_free(x);

    var native_bytes: [E * N * bytes_per_row]u8 = undefined;
    for (0..E) |expert| {
        for (0..N) |row| {
            for (0..bytes_per_row) |byte| {
                const low: u8 = @intCast((expert * 7 + row * 3 + byte * 2) & 0xf);
                const high: u8 = @intCast((expert * 7 + row * 3 + byte * 2 + 1) & 0xf);
                native_bytes[(expert * N + row) * bytes_per_row + byte] = low | (high << 4);
            }
        }
    }
    var packed_words: [E * N * packed_cols]u32 = undefined;
    packNativeBytes(native_bytes[0..], packed_words[0..]);
    const w_shape = [_]c_int{ E, N, packed_cols };
    const w = makeArray(packed_words[0..], &w_shape, .uint32);
    defer _ = mlx.mlx_array_free(w);

    var scale_codes: [E * N]u8 = undefined;
    for (0..E) |expert| {
        for (0..N) |row| scale_codes[expert * N + row] = @intCast(125 + ((expert + row) % 5));
    }
    const scale_shape = [_]c_int{ E, N, 1 };
    const scales = makeArray(scale_codes[0..], &scale_shape, .uint8);
    defer _ = mlx.mlx_array_free(scales);

    const empty = mlx.mlx_array{ .ctx = null };
    const rhs_shape = [_]c_int{R};
    const unsorted_ids = [_]u32{ 2, 0, 1, 2, 1 };
    const unsorted = makeArray(unsorted_ids[0..], &rhs_shape, .uint32);
    defer _ = mlx.mlx_array_free(unsorted);
    const sorted_ids = [_]u32{ 0, 1, 1, 2, 2 };
    const sorted = makeArray(sorted_ids[0..], &rhs_shape, .uint32);
    defer _ = mlx.mlx_array_free(sorted);

    for ([_]struct { ids: mlx.mlx_array, values: []const u32, sorted: bool }{
        .{ .ids = unsorted, .values = unsorted_ids[0..], .sorted = false },
        .{ .ids = sorted, .values = sorted_ids[0..], .sorted = true },
    }) |case| {
        var native = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(native);
        try mlx.check(mlx.mlx_gather_qmm(
            &native,
            x,
            w,
            scales,
            empty,
            empty,
            case.ids,
            true,
            mlx.mlx_optional_int.some(32),
            mlx.mlx_optional_int.some(4),
            "mxfp4",
            case.sorted,
            stream,
        ));
        try testing.expectEqualSlices(c_int, &[_]c_int{ R, 1, N }, mlx.getShape(native));

        var got: [R * N]f32 = undefined;
        try readF32(native, got[0..], stream);
        for (0..R) |row| {
            const expert = case.values[row];
            for (0..N) |column| {
                const weight_row = expert * N + column;
                const expected = hostDot(input[0..], native_bytes[0..], row, weight_row, K, bytes_per_row, scale_codes[weight_row]);
                try expectFiniteApprox(expected, got[row * N + column]);
            }
        }
    }
}

test "mimo MXFP4 custom gatherQmv declines the native mode" {
    const stream = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(stream);

    const E = 2;
    const N = 8;
    const K = 32;
    const w_shape = [_]c_int{ E, N, K / 8 };
    var weights: [E * N * (K / 8)]u32 = @splat(0x22222222);
    const w = makeArray(weights[0..], &w_shape, .uint32);
    defer _ = mlx.mlx_array_free(w);
    const scale_shape = [_]c_int{ E, N, 1 };
    var scale_data: [E * N]u8 = @splat(127);
    const scales = makeArray(scale_data[0..], &scale_shape, .uint8);
    defer _ = mlx.mlx_array_free(scales);
    const x_shape = [_]c_int{ 1, K };
    var x_bits: [K]u16 = @splat(bf16Bits(1.0));
    const x = makeArray(x_bits[0..], &x_shape, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const index_shape = [_]c_int{1};
    const ids_data = [_]u32{0};
    const ids = makeArray(ids_data[0..], &index_shape, .uint32);
    defer _ = mlx.mlx_array_free(ids);

    const declined = try transformer.gatherQmvGateUp(
        stream,
        x,
        w,
        scales,
        .{ .ctx = null },
        w,
        scales,
        .{ .ctx = null },
        ids,
        4,
        32,
        .mxfp4,
    );
    try testing.expect(declined == null);
}
