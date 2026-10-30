//! A kernel's first-use probe: deterministic bf16 operands, a planted error, and the comparisons that judge it.

const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");

pub fn qsaProbeLcgBf16(s: mlx.mlx_stream, shape: []const c_int, seed: u32) ?mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = std.heap.page_allocator.alloc(f32, n) catch return null;
    defer std.heap.page_allocator.free(buf);
    var state = seed;
    for (buf) |*x| {
        state = state *% 1664525 +% 1013904223;
        x.* = @as(f32, @floatFromInt(state >> 8)) * (1.0 / 16777216.0) - 0.5;
    }
    const host = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(host);
    var out = mlx.mlx_array_new();
    if (mlx.mlx_astype(&out, host, .bfloat16, s) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    if (mlx.mlx_array_eval(out) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    return out;
}

pub fn qsaProbeCorruptFirst(s: mlx.mlx_stream, arr: mlx.mlx_array) ?mlx.mlx_array {
    var contig = mlx.mlx_array_new();
    if (mlx.mlx_contiguous(&contig, arr, false, s) != 0) {
        _ = mlx.mlx_array_free(contig);
        return null;
    }
    defer _ = mlx.mlx_array_free(contig);
    var f32a = mlx.mlx_array_new();
    if (mlx.mlx_astype(&f32a, contig, .float32, s) != 0) {
        _ = mlx.mlx_array_free(f32a);
        return null;
    }
    defer _ = mlx.mlx_array_free(f32a);
    if (mlx.mlx_array_eval(f32a) != 0) return null;
    const n = mlx.mlx_array_size(f32a);
    if (n == 0) return null;
    const src = mlx.mlx_array_data_float32(f32a) orelse return null;
    const buf = std.heap.page_allocator.alloc(f32, n) catch return null;
    defer std.heap.page_allocator.free(buf);
    @memcpy(buf, src[0..n]);
    buf[0] += 1.0;
    const shape = mlx.getShape(arr);
    const host = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(host);
    var out = mlx.mlx_array_new();
    if (mlx.mlx_astype(&out, host, mlx.mlx_array_dtype(arr), s) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    return out;
}

pub fn qsaProbeBitEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    if (mlx.mlx_array_equal(&eq, a, b, false, s) != 0) return false;
    if (mlx.mlx_array_eval(eq) != 0) return false;
    var ok: bool = false;
    if (mlx.mlx_array_item_bool(&ok, eq) != 0) return false;
    return ok;
}

pub fn qsaProbeAllFinite(s: mlx.mlx_stream, a: mlx.mlx_array) bool {
    var fin = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fin);
    if (mlx.mlx_isfinite(&fin, a, s) != 0) return false;
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    if (mlx.mlx_all(&all, fin, false, s) != 0) return false;
    if (mlx.mlx_array_eval(all) != 0) return false;
    var ok: bool = false;
    if (mlx.mlx_array_item_bool(&ok, all) != 0) return false;
    return ok;
}

pub fn qsaProbeMaxAbsDiff(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) ?f32 {
    var sub = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sub);
    if (mlx.mlx_subtract(&sub, a, b, s) != 0) return null;
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    if (mlx.mlx_abs(&ab, sub, s) != 0) return null;
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    if (mlx.mlx_max(&mx, ab, false, s) != 0) return null;
    if (mlx.mlx_array_eval(mx) != 0) return null;
    var v: f32 = 0;
    if (mlx.mlx_array_item_float32(&v, mx) != 0) return null;
    return v;
}
