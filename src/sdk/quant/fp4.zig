//! The fp4 formats every fp4 kernel takes: MXFP4 (e8m0 scale per 32) and NVFP4 (e4m3 scale per 16, ModelOpt's
//! optional per-expert global scales beside it), and the one Metal header that decodes either's group scale
//! (`kernels/common/fp4_scale.metal`). A kernel takes the format as its group size (`GS`), never as a second kernel.
const kernelSource = @import("../../mlx/kernels.zig").source;
const QuantMode = @import("params.zig").QuantMode;

pub const Format = enum {
    nvfp4,
    mxfp4,

    /// The scale group the format fixes.
    pub fn group(f: Format) u32 {
        return switch (f) {
            .nvfp4 => 16,
            .mxfp4 => 32,
        };
    }

    pub fn mode(f: Format) QuantMode {
        return switch (f) {
            .nvfp4 => .nvfp4,
            .mxfp4 => .mxfp4,
        };
    }
};

/// The fp4 format of a bank at `mode` and `group_size`; null for any other pair (affine, mxfp8, an fp4 mode at another
/// group size).
pub fn of(mode: QuantMode, group_size: u32) ?Format {
    return switch (mode) {
        .nvfp4 => if (group_size == 16) .nvfp4 else null,
        .mxfp4 => if (group_size == 32) .mxfp4 else null,
        else => null,
    };
}

/// `mlxserve_fp4_scale_of<GS>`: an e8m0 or e4m3 group scale as MLX reads it (its fp8.h, copied).
pub const SCALE_HEADER = kernelSource("kernels/common/fp4_scale.metal");

const std = @import("std");

test "fp4 formats: the (mode, group) pairs and nothing else" {
    try std.testing.expectEqual(Format.nvfp4, of(.nvfp4, 16).?);
    try std.testing.expectEqual(Format.mxfp4, of(.mxfp4, 32).?);
    try std.testing.expect(of(.nvfp4, 32) == null);
    try std.testing.expect(of(.mxfp4, 16) == null);
    try std.testing.expect(of(.affine, 32) == null);
    try std.testing.expect(of(.mxfp8, 32) == null);
    for ([_]Format{ .nvfp4, .mxfp4 }) |f| try std.testing.expectEqual(f, of(f.mode(), f.group()).?);
}

const mlx = @import("../../mlx/ffi.zig");

/// The 256 codes through `mlxserve_fp4_scale_of<GS>` on the GPU, as f32: [0, 256) the read as stored, [256, 512) the read
/// times a 1.0 loaded at run time (a product, flushed where the GPU flushes one).
fn decodeAll(gs: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var codes: [256]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast(i);
    const in = mlx.mlx_array_new_data(&codes, &[_]c_int{256}, 1, .uint8);
    defer _ = mlx.mlx_array_free(in);
    const ones: [16]f32 = @splat(1.0);
    const one = mlx.mlx_array_new_data(&ones, &[_]c_int{16}, 1, .float32);
    defer _ = mlx.mlx_array_free(one);
    const ins = [_][*:0]const u8{ "codes", "one" };
    const outs = [_][*:0]const u8{"y"};
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, 1);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("mlxserve_fp4_scale_codes", iv, ov, "const uint i = thread_position_in_grid.x;\nconst float v = mlxserve_fp4_scale_of<GS>(uint(codes[i]));\ny[i] = v;\ny[256 + i] = v * one[i % 16];\n", SCALE_HEADER, true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{512}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "GS", gs));
    const arrs = [_]mlx.mlx_array{ in, one };
    const v = mlx.mlx_vector_array_new_data(&arrs, arrs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, k, v, c, s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    try mlx.check(mlx.mlx_array_eval(y));
    return y;
}

/// MLX's fp8.h `operator float` on the host: fp8_e4m3 by bit placement into a half (0x7f / 0xff NaN) times 256,
/// fp8_e8m0 as a float exponent (0 is 2^-127, 0xff NaN).
fn mlxRead(f: Format, code: u8) f32 {
    return switch (f) {
        .nvfp4 => blk: {
            const v: u16 = code & 0x7F;
            const u: u16 = (v << 7) | (((v + 1) >> 7) << 14) | (@as(u16, code & 0x80) << 8);
            break :blk @as(f32, @floatCast(@as(f16, @bitCast(u)) * 256.0));
        },
        .mxfp4 => @bitCast(if (code == 0) @as(u32, 0x400000) else if (code == 0xFF) @as(u32, 0x7FC00000) else @as(u32, code) << 23),
    };
}

/// MLX's own read of the 256 scale codes: `mlx_dequantize` of a bank whose every e2m1 code is 1.0 (code 2) and whose
/// group g has scale code g, as f32 (element g * GS).
fn mlxAll(f: Format, s: mlx.mlx_stream) !mlx.mlx_array {
    const gs: usize = f.group();
    var words: [256 * 32 / 8]u32 = @splat(0x22222222);
    var codes: [256]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast(i);
    const k: c_int = @intCast(256 * gs);
    const w = mlx.mlx_array_new_data(&words, &[_]c_int{ 1, @divExact(k, 8) }, 2, .uint32);
    defer _ = mlx.mlx_array_free(w);
    const sc = mlx.mlx_array_new_data(&codes, &[_]c_int{ 1, 256 }, 2, .uint8);
    defer _ = mlx.mlx_array_free(sc);
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_dequantize(&y, w, sc, .{ .ctx = null }, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(4), f.mode().cstr(), .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    try mlx.check(mlx.mlx_array_eval(y));
    return y;
}

test "fp4 scale decode: every e4m3 and e8m0 code is MLX's fp8.h read, and times a run-time 1.0 MLX's dequant bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    for ([_]Format{ .nvfp4, .mxfp4 }) |f| {
        const ours = try decodeAll(@intCast(f.group()), s);
        defer _ = mlx.mlx_array_free(ours);
        const theirs = try mlxAll(f, s);
        defer _ = mlx.mlx_array_free(theirs);
        const o = (mlx.mlx_array_data_float32(ours) orelse return error.NoData)[0..512];
        const t = mlx.mlx_array_data_float32(theirs) orelse return error.NoData;
        for (0..256) |code| {
            const read = mlxRead(f, @intCast(code));
            const dq = t[code * @as(usize, f.group())];
            if (!sameValue(o[code], read) or !sameValue(o[256 + code], dq))
                std.debug.print("{t} code 0x{x:0>2}: ours {e} / x1 {e}, MLX read {e} / dequant {e}\n", .{ f, code, o[code], o[256 + code], read, dq });
            try std.testing.expect(sameValue(o[code], read));
            try std.testing.expect(sameValue(o[256 + code], dq));
        }
        // The dequant product flushes e8m0 0x00 (2^-127, a subnormal) to 0 on the GPU, MLX's and ours alike.
        if (f == .mxfp4) try std.testing.expectEqual(@as(f32, 0), o[256]);
    }
}

/// Bit for bit, or both NaN.
fn sameValue(a: f32, b: f32) bool {
    if (std.math.isNan(b)) return std.math.isNan(a);
    return @as(u32, @bitCast(a)) == @as(u32, @bitCast(b));
}

test "fp4 e2m1 read: every code (mlxserve_fp4_e2m1_value) is MLX's fp4 dequant at scale 1.0, bit for bit, -0 kept" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    // One mxfp4 group of 32 under e8m0 code 127 (1.0): codes 0..15 twice, low nibble first.
    var words = [_]u32{ 0x76543210, 0xFEDCBA98, 0x76543210, 0xFEDCBA98 };
    var scale = [_]u8{127};
    const w = mlx.mlx_array_new_data(&words, &[_]c_int{ 1, 4 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(w);
    const sc = mlx.mlx_array_new_data(&scale, &[_]c_int{ 1, 1 }, 2, .uint8);
    defer _ = mlx.mlx_array_free(sc);
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_dequantize(&want, w, sc, .{ .ctx = null }, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "mxfp4", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    try mlx.check(mlx.mlx_array_eval(want));
    const ins = [_][*:0]const u8{"w"};
    const outs = [_][*:0]const u8{"y"};
    const iv = mlx.mlx_vector_string_new_data(&ins, 1);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, 1);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("mlxserve_e2m1_codes", iv, ov, "const uint i = thread_position_in_grid.x;\ny[i] = mlxserve_fp4_e2m1_value((w[i / 8] >> (4 * (i % 8))) & 0xFu);\n", SCALE_HEADER, true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{32}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
    const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{w}, 1);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, k, v, c, s));
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    try mlx.check(mlx.mlx_array_eval(y));
    const got = (mlx.mlx_array_data_float32(y) orelse return error.NoData)[0..32];
    const ref = (mlx.mlx_array_data_float32(want) orelse return error.NoData)[0..32];
    for (got, ref) |g, r| try std.testing.expectEqual(@as(u32, @bitCast(r)), @as(u32, @bitCast(g)));
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(got[8])));
}

test "fp4 scale decode: the e4m3 pair read (mlxserve_e4m3x2_h) is MLX's read x 2^-8 for every code in both lanes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    // Word w = code c at bits 0-7 and code 255 - c at bits 16-23; bytes 8-15 / 24-31 hold junk the read ignores.
    var words: [256]u32 = undefined;
    for (&words, 0..) |*w, c| w.* = @as(u32, @intCast(c)) | (0xA5 << 8) | (@as(u32, @intCast(255 - c)) << 16) | (0x5A << 24);
    const in = mlx.mlx_array_new_data(&words, &[_]c_int{256}, 1, .uint32);
    defer _ = mlx.mlx_array_free(in);
    const ins = [_][*:0]const u8{"words"};
    const outs = [_][*:0]const u8{"y"};
    const iv = mlx.mlx_vector_string_new_data(&ins, 1);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, 1);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("mlxserve_e4m3x2_codes", iv, ov, "const uint i = thread_position_in_grid.x;\nconst half2 h = mlxserve_e4m3x2_h(words[i]);\ny[2 * i] = float(h.x);\ny[2 * i + 1] = float(h.y);\n", SCALE_HEADER ++ SCALE_HEADER, true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{512}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
    const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{in}, 1);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, k, v, c, s));
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    try mlx.check(mlx.mlx_array_eval(y));
    const got = (mlx.mlx_array_data_float32(y) orelse return error.NoData)[0..512];
    // The header twice in one source compiles once (its guard): a kernel may get it by more than one route.
    for (0..256) |code| {
        try std.testing.expect(sameValue(got[2 * code], mlxRead(.nvfp4, @intCast(code)) / 256.0));
        try std.testing.expect(sameValue(got[2 * code + 1], mlxRead(.nvfp4, @intCast(255 - code)) / 256.0));
    }
}
