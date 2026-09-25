//! Optional Qwen-Image-2.1 kernels; callers retain an unfused fallback.
const std = @import("std");
const mlx = @import("mlx.zig");
const A = mlx.mlx_array;
const S = mlx.mlx_stream;

const rope_source =
    \\#pragma clang fp contract(off)
    \\#pragma clang fp reassociate(off)
    \\uint pair = thread_position_in_grid.x;
    \\uint pair_count = uint(x_shape[1]) * uint(x_shape[2]) * uint(x_shape[3] / 2);
    \\if (pair >= pair_count) return;
    \\uint half_dim = x_shape[3] / 2;
    \\uint token = pair / (x_shape[2] * half_dim);
    \\uint table = token * half_dim + pair % half_dim;
    \\float re = float(x[2 * pair]);
    \\float im = float(x[2 * pair + 1]);
    \\float c = cs[table];
    \\float s = sn[table];
    \\float rc = re * c;
    \\float is = im * s;
    \\float rs = re * s;
    \\float ic = im * c;
    \\out[2 * pair] = T(rc - is);
    \\out[2 * pair + 1] = T(rs + ic);
;

/// x [1,L,heads,head_dim], FP32 tables [1,L,1,head_dim/2,1].
/// Returns a new array in output_dtype (normalization may promote x to FP32).
/// The local kernel handle is released;
/// MLX owns the lazy primitive and caches compiled Metal pipelines itself.
pub fn applyRope(x: A, cos: A, sin: A, output_dtype: mlx.mlx_dtype, s: S) !A {
    const shape = mlx.getShape(x);
    if (shape.len != 4 or shape[0] != 1 or shape[1] <= 0 or shape[2] <= 0 or shape[3] <= 0 or @mod(shape[3], 2) != 0)
        return error.InvalidRopeShape;
    const table_shape = [_]c_int{ 1, shape[1], 1, @divExact(shape[3], 2), 1 };
    if (!std.mem.eql(c_int, mlx.getShape(cos), &table_shape) or !std.mem.eql(c_int, mlx.getShape(sin), &table_shape))
        return error.InvalidRopeShape;
    const dtype = mlx.mlx_array_dtype(x);
    if ((dtype != .float32 and dtype != .float16 and dtype != .bfloat16) or mlx.mlx_array_dtype(cos) != .float32 or mlx.mlx_array_dtype(sin) != .float32)
        return error.InvalidRopeDtype;
    if (output_dtype != .float32 and output_dtype != .float16 and output_dtype != .bfloat16)
        return error.InvalidRopeDtype;
    const pairs = std.math.cast(c_int, mlx.mlx_array_size(x) / 2) orelse return error.InvalidRopeShape;
    const names = [_][*:0]const u8{ "x", "cs", "sn" };
    const output_names = [_][*:0]const u8{"out"};
    const ins_names = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(ins_names);
    const outs_names = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(outs_names);
    const kernel = mlx.mlx_fast_metal_kernel_new("qwen_image21_rope_separate_rounding", ins_names, outs_names, rope_source, "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    defer _ = mlx.mlx_fast_metal_kernel_free(kernel);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.ptr, shape.len, output_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", output_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, pairs, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const inputs = [_]A{ x, cos, sin };
    const ins = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kernel, ins, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}

fn cast(x: A, dtype: mlx.mlx_dtype, s: S) !A {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, x, dtype, s));
    return out;
}

test "QwenImage fused RoPE matches separate FP32 operations for all compute dtypes" {
    const s = mlx.mlx_default_gpu_stream_new();
    const shape = [_]c_int{ 1, 3, 2, 8 };
    const table_shape = [_]c_int{ 1, 3, 1, 4, 1 };
    var values: [48]f32 = undefined;
    var cos: [12]f32 = undefined;
    var sin: [12]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.37) * 4;
    for (&cos, &sin, 0..) |*c, *v, i| {
        const angle = @as(f32, @floatFromInt(i)) * 0.29;
        c.* = @cos(angle);
        v.* = @sin(angle);
    }
    // (1+2^-13)*(1-2^-13)-1 rounds to zero with separate
    // FP32 operations but not with FMA; catches accidental contraction.
    values[0] = 1.0001220703125;
    values[1] = 1;
    cos[0] = 0.9998779296875;
    sin[0] = 1;
    const raw = mlx.mlx_array_new_data(&values, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(raw);
    const c = mlx.mlx_array_new_data(&cos, &table_shape, table_shape.len, .float32);
    defer _ = mlx.mlx_array_free(c);
    const sn = mlx.mlx_array_new_data(&sin, &table_shape, table_shape.len, .float32);
    defer _ = mlx.mlx_array_free(sn);
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16, .float16 }) |dtype| {
        const x = try cast(raw, dtype, s);
        defer _ = mlx.mlx_array_free(x);
        const xf = try cast(x, .float32, s);
        defer _ = mlx.mlx_array_free(xf);
        try mlx.check(mlx.mlx_array_eval(xf));
        const xp = mlx.mlx_array_data_float32(xf).?;
        var expected: [48]f32 = undefined;
        for (0..24) |p| {
            @setFloatMode(.strict);
            const t = (p / 8) * 4 + p % 4;
            const rc: f32 = xp[2 * p] * cos[t];
            const is: f32 = xp[2 * p + 1] * sin[t];
            const rs: f32 = xp[2 * p] * sin[t];
            const ic: f32 = xp[2 * p + 1] * cos[t];
            expected[2 * p] = rc - is;
            expected[2 * p + 1] = rs + ic;
        }
        const ref_raw = mlx.mlx_array_new_data(&expected, &shape, shape.len, .float32);
        defer _ = mlx.mlx_array_free(ref_raw);
        // Cover promoted FP32 normalization inputs returning BF16/FP16,
        // as well as every same-dtype path and reverse conversion.
        for ([_]mlx.mlx_dtype{ .float32, .bfloat16, .float16 }) |output_dtype| {
            const ref = try cast(ref_raw, output_dtype, s);
            defer _ = mlx.mlx_array_free(ref);
            const ref32 = try cast(ref, .float32, s);
            defer _ = mlx.mlx_array_free(ref32);
            const got = try applyRope(x, c, sn, output_dtype, s);
            defer _ = mlx.mlx_array_free(got);
            const got32 = try cast(got, .float32, s);
            defer _ = mlx.mlx_array_free(got32);
            try mlx.check(mlx.mlx_array_eval(ref32));
            try mlx.check(mlx.mlx_array_eval(got32));
            try std.testing.expectEqual(output_dtype, mlx.mlx_array_dtype(got));
            try std.testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(ref32).?[0..48], mlx.mlx_array_data_float32(got32).?[0..48]);
        }
    }
}
