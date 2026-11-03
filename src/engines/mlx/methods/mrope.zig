//! The legacy `Transformer`'s methods over `sdk/blocks/rope/mrope.zig` (what reads the engine's model state).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const mrope = @import("../../../sdk/vision/mrope.zig");
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Transformer = @import("../transformer.zig").Transformer;
const yarnMscale = @import("yarn.zig").yarnMscale;
const MropeCosSin = @import("../../../sdk/blocks/rope/mrope.zig").MropeCosSin;

/// Build per-token interleaved-M-RoPE cos/sin [1,1,seq_len,rope_dims] (bf16).
/// Positions inside the prompt come from the explicit 3-D table; positions
/// beyond it are generated text and collapse to scalar `absolute + delta`.
/// `positions.base` lets a suffix-only speculative KV cache map its relative
/// offsets back to the full prompt table.
pub fn buildMropeCosSin(self: *Transformer, positions: mrope.PositionContext, offset: usize, seq_len: usize) !MropeCosSin {
    return self.mropeCosSinAt(positions, offset, 1, seq_len, .bfloat16);
}

/// cos/sin [1,1,n,rope_dims] at the `n` absolute positions `start +
/// stride*i`, in `dtype`. `stride 1` = a prefill chunk; the qwen4 QSA
/// indexer ropes pooled block keys at block-START positions (`stride =
/// ratio`) — the reference reads `full_cos[group_starts]`, which is 3-D
/// inside an image and `abs + delta` past the prompt, exactly like the
/// per-token table.
pub fn mropeCosSinAt(self: *Transformer, positions: mrope.PositionContext, start: usize, stride: usize, n: usize, dtype: mlx.mlx_dtype) !MropeCosSin {
    const cfg = &self.config;
    const rope_dims: usize = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
    const half = rope_dims / 2;

    const inv_freq = try self.allocator.alloc(f64, half);
    defer self.allocator.free(inv_freq);
    // One spectrum for every rope consumer: the YaRN table built at load
    // (when the config scales) or the plain theta^(-2j/d) one.
    if (self.yarnActive()) {
        const yf = self.yarn_inv_freq.?;
        std.debug.assert(yf.len == half);
        @memcpy(inv_freq, yf);
    } else mrope.computeInvFreq(inv_freq, rope_dims, cfg.rope_theta);
    const sel = try self.allocator.alloc(u8, half);
    defer self.allocator.free(sel);
    mrope.interleavedSelector(sel, cfg.mrope_section);

    const cos_buf = try self.allocator.alloc(f32, n * rope_dims);
    defer self.allocator.free(cos_buf);
    const sin_buf = try self.allocator.alloc(f32, n * rope_dims);
    defer self.allocator.free(sin_buf);
    mrope.fillCosSin(cos_buf, sin_buf, positions, start, stride, n, inv_freq, sel, rope_dims, yarnMscale(cfg));
    const shape = [_]c_int{ 1, 1, @intCast(n), @intCast(rope_dims) };
    const cf = mlx.mlx_array_new_data(cos_buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(cf);
    const sf = mlx.mlx_array_new_data(sin_buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(sf);
    var cos = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&cos, cf, dtype, self.s));
    errdefer _ = mlx.mlx_array_free(cos);
    var sin = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sin);
    try mlx.check(mlx.mlx_astype(&sin, sf, dtype, self.s));
    return .{ .cos = cos, .sin = sin };
}

/// Per-prefill-chunk M-RoPE cos/sin on `ctx.mrope_cos/sin_cur`, built once
/// per forward (freed by `endMropeChunk`). Forwards past the explicit
/// position table (decode, multi-token spec verify) leave them null and
/// take the scalar `offset + delta` path.
pub fn beginMropeChunk(self: *Transformer, ctx: *ForwardCtx, offset: usize, seq_len: usize, dtype: mlx.mlx_dtype) !void {
    if (ctx.mrope_pos == null or seq_len <= 1 or offset + seq_len > ctx.mrope_total) return;
    const cs = try self.mropeCosSinAt(mropeContext(ctx), offset, 1, seq_len, dtype);
    ctx.mrope_cos_cur = cs.cos;
    ctx.mrope_sin_cur = cs.sin;
}

/// Apply NeoX RoPE to the first `rope_dims` dims of `arr` [B,H,S,hd] using
/// precomputed cos/sin [1,1,S,rope_dims] (broadcast over B,H); pass remaining
/// dims through. Equivalent to `mlx_fast_rope(traditional=false)` but with
/// per-token (M-RoPE) angles instead of a scalar offset.
pub fn applyMrope(self: *Transformer, arr: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, rope_dims: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(arr);
    const b = sh[0];
    const h = sh[1];
    const s = sh[2];
    const hd = sh[3];
    const half = @divExact(rope_dims, 2);
    const st = [_]c_int{ 1, 1, 1, 1 };

    var rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot);
    try mlx.check(mlx.mlx_slice(&rot, arr, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ b, h, s, rope_dims }, 4, &st, 4, self.s));

    // rotate_half(rot) = concat(-rot[..,half:], rot[..,:half])
    var r2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r2);
    try mlx.check(mlx.mlx_slice(&r2, rot, &[_]c_int{ 0, 0, 0, half }, 4, &[_]c_int{ b, h, s, rope_dims }, 4, &st, 4, self.s));
    var r1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r1);
    try mlx.check(mlx.mlx_slice(&r1, rot, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ b, h, s, half }, 4, &st, 4, self.s));
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, r2, self.s));
    const rh_arrs = [_]mlx.mlx_array{ neg, r1 };
    const rh_vec = mlx.mlx_vector_array_new_data(&rh_arrs, 2);
    defer _ = mlx.mlx_vector_array_free(rh_vec);
    var rh = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rh);
    try mlx.check(mlx.mlx_concatenate_axis(&rh, rh_vec, -1, self.s));

    var xcos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xcos);
    try mlx.check(mlx.mlx_multiply(&xcos, rot, cos, self.s));
    var rsin = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsin);
    try mlx.check(mlx.mlx_multiply(&rsin, rh, sin, self.s));
    var out_rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_rot);
    try mlx.check(mlx.mlx_add(&out_rot, xcos, rsin, self.s));

    if (hd == rope_dims) {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, out_rot, mlx.mlx_array_dtype(arr), self.s));
        return out;
    }
    var pass = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pass);
    try mlx.check(mlx.mlx_slice(&pass, arr, &[_]c_int{ 0, 0, 0, rope_dims }, 4, &[_]c_int{ b, h, s, hd }, 4, &st, 4, self.s));
    const cat_arrs = [_]mlx.mlx_array{ out_rot, pass };
    const cat_vec = mlx.mlx_vector_array_new_data(&cat_arrs, 2);
    defer _ = mlx.mlx_vector_array_free(cat_vec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, cat_vec, -1, self.s));
    return out;
}

pub fn mropeContext(ctx: *const ForwardCtx) mrope.PositionContext {
    return .{ .pos = ctx.mrope_pos.?, .total = ctx.mrope_total, .delta = ctx.mrope_delta };
}

pub fn endMropeChunk(ctx: *ForwardCtx) void {
    if (ctx.mrope_cos_cur) |c| _ = mlx.mlx_array_free(c);
    if (ctx.mrope_sin_cur) |sn| _ = mlx.mlx_array_free(sn);
    ctx.mrope_cos_cur = null;
    ctx.mrope_sin_cur = null;
}
