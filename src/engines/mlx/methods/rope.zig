//! The legacy `Transformer`'s methods over `sdk/blocks/rope/rope.zig` (what reads the engine's model state).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const Transformer = @import("../transformer.zig").Transformer;
const MropeCosSin = @import("../../../sdk/blocks/rope/mrope.zig").MropeCosSin;

/// The `[half]` f32 inverse-frequency spectrum this trunk ropes with. Caller owns the result.
pub fn ropeInvFreq(self: *Transformer, rope_dims: c_int, theta: f32) !mlx.mlx_array {
    const half = @divExact(rope_dims, 2);
    // YaRN: the load-time host spectrum (f64 → the tensor dtype), the SAME
    // table attention and the M-RoPE tables rotate with. Deriving inv_freq
    // from `theta` in-graph below is the unscaled rope only.
    if (self.yarnActive()) {
        const yf = self.yarn_inv_freq.?;
        std.debug.assert(half == @as(c_int, @intCast(yf.len)));
        const f32s = try self.allocator.alloc(f32, yf.len);
        defer self.allocator.free(f32s);
        for (yf, 0..) |v, i| f32s[i] = @floatCast(v);
        const shape = [_]c_int{@intCast(yf.len)};
        return mlx.mlx_array_new_data(f32s.ptr, &shape, 1, .float32);
    }
    var j = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(j);
    try mlx.check(mlx.mlx_arange(&j, 0, @floatFromInt(half), 1.0, .float32, self.s));
    // inv_freq_j = theta^(-2j/rope_dims) = exp(-j * 2 ln(theta) / rope_dims)
    const neg = mlx.mlx_array_new_float(-2.0 * @log(theta) / @as(f32, @floatFromInt(rope_dims)));
    defer _ = mlx.mlx_array_free(neg);
    var jl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(jl);
    try mlx.check(mlx.mlx_multiply(&jl, j, neg, self.s));
    var inv_freq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_exp(&inv_freq, jl, self.s)); // [half]
    return inv_freq;
}

/// The build half of `ropeAtFreqs`: cos/sin `[n, rope_dims]` for positions `base + step*i`. `inv_freq` is taken.
pub fn ropeCosSinFromFreqs(self: *Transformer, rope_dims: c_int, inv_freq: mlx.mlx_array, base: f32, step: f32, n: c_int, dt: mlx.mlx_dtype) !MropeCosSin {
    defer _ = mlx.mlx_array_free(inv_freq);
    std.debug.assert(mlx.getShape(inv_freq)[0] == @divExact(rope_dims, 2));
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    try mlx.check(mlx.mlx_arange(&pos, base, base + step * @as(f32, @floatFromInt(n)) - step * 0.5, step, .float32, self.s)); // [n]
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos, &[_]c_int{ n, 1 }, 2, self.s));
    var ang = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ang);
    try mlx.check(mlx.mlx_multiply(&ang, pos2, inv_freq, self.s)); // [n, half]
    const parts = [_]mlx.mlx_array{ ang, ang };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var emb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(emb);
    try mlx.check(mlx.mlx_concatenate_axis(&emb, vec, 1, self.s)); // [n, rd]
    var cos_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cos_f);
    try mlx.check(mlx.mlx_cos(&cos_f, emb, self.s));
    var sin_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sin_f);
    try mlx.check(mlx.mlx_sin(&sin_f, emb, self.s));
    var cosv = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cosv);
    try mlx.check(mlx.mlx_astype(&cosv, cos_f, dt, self.s));
    var sinv = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sinv);
    try mlx.check(mlx.mlx_astype(&sinv, sin_f, dt, self.s));
    if (self.yarnActive()) {
        const ms = try scalarOf(self.config.yarn_attention_factor, dt, self.s);
        defer _ = mlx.mlx_array_free(ms);
        var cos_s = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&cos_s, cosv, ms, self.s));
        _ = mlx.mlx_array_free(cosv);
        cosv = cos_s;
        var sin_s = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&sin_s, sinv, ms, self.s));
        _ = mlx.mlx_array_free(sinv);
        sinv = sin_s;
    }
    return .{ .cos = cosv, .sin = sinv };
}

/// The apply half of `ropeAtFreqs`: rotate the first `rope_dims` dims with borrowed cos/sin.
pub fn ropeApplyCosSin(self: *Transformer, x: mlx.mlx_array, rope_dims: c_int, cosv: mlx.mlx_array, sinv: mlx.mlx_array) !mlx.mlx_array {
    const half = @divExact(rope_dims, 2);
    const sh = mlx.getShape(x);
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var xr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xr);
    try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], rope_dims }, 4, &strides, 4, self.s));
    var xp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xp);
    try mlx.check(mlx.mlx_slice(&xp, x, &[_]c_int{ 0, 0, 0, rope_dims }, 4, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, 4, &strides, 4, self.s));
    var x1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x1);
    try mlx.check(mlx.mlx_slice(&x1, xr, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], half }, 4, &strides, 4, self.s));
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_slice(&x2, xr, &[_]c_int{ 0, 0, 0, half }, 4, &[_]c_int{ sh[0], sh[1], sh[2], rope_dims }, 4, &strides, 4, self.s));
    var nx2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nx2);
    try mlx.check(mlx.mlx_negative(&nx2, x2, self.s));
    const rparts = [_]mlx.mlx_array{ nx2, x1 };
    const rvec = mlx.mlx_vector_array_new_data(&rparts, 2);
    defer _ = mlx.mlx_vector_array_free(rvec);
    var rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot);
    try mlx.check(mlx.mlx_concatenate_axis(&rot, rvec, -1, self.s));
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_multiply(&a, xr, cosv, self.s));
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_multiply(&b, rot, sinv, self.s));
    var roped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(roped);
    try mlx.check(mlx.mlx_add(&roped, a, b, self.s));
    const oparts = [_]mlx.mlx_array{ roped, xp };
    const ovec = mlx.mlx_vector_array_new_data(&oparts, 2);
    defer _ = mlx.mlx_vector_array_free(ovec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, ovec, -1, self.s));
    return out;
}

/// Partial neox RoPE of `x [B,1,N,hd]` at positions `base + step*i` with explicit cos/sin.
pub fn ropeAtPositions(self: *Transformer, x: mlx.mlx_array, rope_dims: c_int, theta: f32, base: f32, step: f32, n: c_int) !mlx.mlx_array {
    return self.ropeAtFreqs(x, rope_dims, try self.ropeInvFreq(rope_dims, theta), base, step, n); // takes ownership
}

/// The shared tail of `ropeAtPositions`: rotate `x` at positions
/// `base + step*i` using the given inv_freq array (`[half]`, f32).
pub fn ropeAtFreqs(self: *Transformer, x: mlx.mlx_array, rope_dims: c_int, inv_freq: mlx.mlx_array, base: f32, step: f32, n: c_int) !mlx.mlx_array {
    const cs = try self.ropeCosSinFromFreqs(rope_dims, inv_freq, base, step, n, mlx.mlx_array_dtype(x));
    defer _ = mlx.mlx_array_free(cs.cos);
    defer _ = mlx.mlx_array_free(cs.sin);
    return self.ropeApplyCosSin(x, rope_dims, cs.cos, cs.sin);
}
