//! Test-only kit for the media block parity tests: random inputs on the CPU stream and the
//! raw mlx-c primitives the media backends define file-locally, so each oracle copy reads
//! verbatim. Nothing here is reached outside tests.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");

pub const S = mlx.mlx_stream;
pub const A = mlx.mlx_array;
pub const Graph = ops.Graph;

pub fn cpu() S {
    return mlx.mlx_default_cpu_stream_new();
}

/// Standard normal x `scale` in `dt`, evaluated.
pub fn randn(g: *Graph, sh: []const c_int, seed: u64, dt: mlx.mlx_dtype, scale: f32) !A {
    const key = try g.call(mlx.mlx_random_key, .{seed});
    const x = try g.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, mlx.mlx_dtype.float32, @as(f32, 0), scale, key });
    const y = try g.astype(x, dt);
    try ops.eval(y);
    return y;
}

/// Takes an oracle's owned result into `g` (freed with it).
pub fn own(g: *Graph, x: A) !A {
    return g.adopt(x);
}

// ── the backends' file-local primitives, verbatim ──

pub fn addA(a: A, b: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}
pub fn subA(a: A, b: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, a, b, s));
    return o;
}
pub fn mulA(a: A, b: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}
pub fn divA(a: A, b: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&o, a, b, s));
    return o;
}
pub fn reshape(x: A, shape: []const c_int, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
pub fn transpose(x: A, axes: []const c_int, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}
pub fn astype(x: A, dt: mlx.mlx_dtype, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
pub fn contig(x: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
pub fn rms(x: A, w: A, eps: f32, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}
pub fn sigmoidA(x: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&o, x, s));
    return o;
}
pub fn silu(x: A, s: S) !A {
    const sig = try sigmoidA(x, s);
    defer _ = mlx.mlx_array_free(sig);
    return mulA(x, sig, s);
}
pub fn scalarF(v: f32) A {
    return mlx.mlx_array_new_float(v);
}
/// mage_flow / qwen_image: `v` in `ref`'s dtype (a bare f32 when `ref` is f32).
pub fn scalarLike(v: f32, ref: A, s: S) !A {
    const sc = scalarF(v);
    if (mlx.mlx_array_dtype(ref) == .float32) return sc;
    defer _ = mlx.mlx_array_free(sc);
    return astype(sc, mlx.mlx_array_dtype(ref), s);
}
pub fn concat(arrs: []const A, axis: c_int, s: S) !A {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
/// x[..., start:stop, ...] on one axis.
pub fn sliceAxis(x: A, axis: usize, start: c_int, stop: c_int, s: S) !A {
    const sh = mlx.getShape(x);
    var lo: [8]c_int = @splat(0);
    var hi: [8]c_int = undefined;
    const st: [8]c_int = @splat(1);
    @memcpy(hi[0..sh.len], sh);
    lo[axis] = start;
    hi[axis] = stop;
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, sh.len, &hi, sh.len, &st, sh.len, s));
    return o;
}
pub fn meanAxes(x: A, axes: []const c_int, keepdims: bool, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_mean_axes(&o, x, axes.ptr, axes.len, keepdims, s));
    return o;
}
pub fn meanLast(x: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_mean_axis(&o, x, -1, true, s));
    return o;
}
/// A dense projection with the backends' `forward(x, s)` shape: x @ w^T (+ b).
pub const RawLin = struct {
    w: A,
    b: ?A = null,
    pub fn forward(self: *const RawLin, x: A, s: S) !A {
        const wt = try transpose(self.w, &.{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(wt);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&o, x, wt, s));
        const b = self.b orelse return o;
        defer _ = mlx.mlx_array_free(o);
        return addA(o, b, s);
    }
};
