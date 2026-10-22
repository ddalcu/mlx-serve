//! Op scope over mlx-c: every intermediate a forward makes is kept in the
//! scope and freed together, so a layer reads as the math it computes.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");

const S = mlx.mlx_stream;
const A = mlx.mlx_array;
const Weights = model_mod.Weights;

/// A projection resolved once at load: affine-quantized (width solved from the packed
/// geometry, never a literal) or dense, with an optional bias. Handles are borrowed from `Weights`.
pub const Linear = struct {
    w: A,
    scales: ?A = null,
    biases: A = .{ .ctx = null },
    bias: ?A = null,
    bits: u32 = 0,
    group_size: u32 = 0,

    pub fn forward(self: *const Linear, sc: *Scope, x: A) !A {
        const y = if (self.scales) |sca|
            try sc.qmm(x, self.w, sca, self.biases, self.group_size, self.bits)
        else
            try sc.matmul(x, try sc.transpose(self.w, &.{ 1, 0 }));
        return if (self.bias) |b| sc.add(y, b) else y;
    }
};

pub fn getW(w: *const Weights, comptime fmt: []const u8, args: anytype) !A {
    var buf: [160]u8 = undefined;
    const key = try std.fmt.bufPrint(&buf, fmt, args);
    return w.get(key) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{key});
        return error.MissingWeight;
    };
}

pub fn linear(w: *const Weights, comptime fmt: []const u8, args: anytype, in_dim: u32) !Linear {
    var buf: [160]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&buf, fmt, args);
    var l = Linear{ .w = try getW(w, "{s}.weight", .{prefix}) };
    if (w.get(try std.fmt.bufPrint(buf[prefix.len..], "{s}.scales", .{prefix}))) |sca| {
        l.scales = sca;
        l.biases = try getW(w, "{s}.biases", .{prefix});
        const qp = transformer_mod.affineParamsFromGeometry(l.w, sca, in_dim) orelse {
            log.err("unsolvable quant geometry for {s}\n", .{prefix});
            return error.BadQuantGeometry;
        };
        l.bits = qp.bits;
        l.group_size = qp.group_size;
    }
    l.bias = w.get(try std.fmt.bufPrint(buf[prefix.len..], "{s}.bias", .{prefix}));
    return l;
}

/// Evaluate every array once so no weight stays a lazy file read.
pub fn evalAll(w: *const Weights) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var it = w.map.valueIterator();
    while (it.next()) |v| _ = mlx.mlx_vector_array_append_value(vec, v.*);
    try mlx.check(mlx.mlx_eval(vec));
}

pub const Scope = struct {
    a: std.mem.Allocator,
    s: S,
    items: std.ArrayList(A) = .empty,

    pub fn init(a: std.mem.Allocator, s: S) Scope {
        return .{ .a = a, .s = s };
    }
    pub fn deinit(self: *Scope) void {
        for (self.items.items) |x| _ = mlx.mlx_array_free(x);
        self.items.deinit(self.a);
    }
    pub fn keep(self: *Scope, x: A) !A {
        self.items.append(self.a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }
    /// An op's output handle `o`, after the op returned `rc`.
    pub fn res(self: *Scope, rc: c_int, o: *const A) !A {
        if (rc != 0) {
            _ = mlx.mlx_array_free(o.*);
            return error.MlxError;
        }
        return self.keep(o.*);
    }
    /// A new handle on `x` that outlives the scope (caller frees).
    pub fn out(_: *Scope, x: A) A {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        return o;
    }

    pub fn add(sc: *Scope, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_add(&o, x, y, sc.s), &o);
    }
    pub fn sub(sc: *Scope, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_subtract(&o, x, y, sc.s), &o);
    }
    pub fn mul(sc: *Scope, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_multiply(&o, x, y, sc.s), &o);
    }
    pub fn scalar(sc: *Scope, v: f32) !A {
        return sc.keep(mlx.mlx_array_new_float(v));
    }
    pub fn addS(sc: *Scope, x: A, v: f32) !A {
        return sc.add(x, try sc.scalar(v));
    }
    pub fn mulS(sc: *Scope, x: A, v: f32) !A {
        return sc.mul(x, try sc.scalar(v));
    }
    pub fn matmul(sc: *Scope, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_matmul(&o, x, y, sc.s), &o);
    }
    pub fn tanh(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_tanh(&o, x, sc.s), &o);
    }
    pub fn sigmoid(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_sigmoid(&o, x, sc.s), &o);
    }
    pub fn silu(sc: *Scope, x: A) !A {
        return sc.mul(x, try sc.sigmoid(x));
    }
    pub fn relu(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_maximum(&o, x, try sc.scalarLike(x, 0), sc.s), &o);
    }
    pub fn astype(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_astype(&o, x, dt, sc.s), &o);
    }
    pub fn reshape(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    pub fn transpose(sc: *Scope, x: A, axes: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, sc.s), &o);
    }
    pub fn broadcast(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_broadcast_to(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    /// x[..., start:stop, ...] on one axis.
    pub fn slice(sc: *Scope, x: A, axis: usize, start: c_int, stop: c_int) !A {
        const sh = mlx.getShape(x);
        var lo: [8]c_int = @splat(0);
        var hi: [8]c_int = undefined;
        const st: [8]c_int = @splat(1);
        @memcpy(hi[0..sh.len], sh);
        lo[axis] = start;
        hi[axis] = stop;
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_slice(&o, x, &lo, sh.len, &hi, sh.len, &st, sh.len, sc.s), &o);
    }
    pub fn concat(sc: *Scope, xs: []const A, axis: c_int) !A {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (xs) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_concatenate_axis(&o, vec, axis, sc.s), &o);
    }
    /// x @ w^T (+ b) — PyTorch Linear over the last axis.
    pub fn lin(sc: *Scope, x: A, w: A, b: ?A) !A {
        const y = try sc.matmul(x, try sc.transpose(w, &.{ 1, 0 }));
        return if (b) |bb| sc.add(y, bb) else y;
    }
    pub fn rms(sc: *Scope, x: A, w: A, eps: f32) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_fast_rms_norm(&o, x, w, eps, sc.s), &o);
    }
    pub fn rope(sc: *Scope, x: A, dims: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_fast_rope(&o, x, dims, false, mlx.mlx_optional_float.some(10000.0), 1.0, 0, .{ .ctx = null }, sc.s), &o);
    }
    pub fn sdpa(sc: *Scope, q: A, k: A, v: A, scale: f32) !A {
        return sc.sdpaMode(q, k, v, scale, "");
    }
    /// [B,L,H*D] → [B,H,L,D]
    pub fn heads(sc: *Scope, x: A, h: c_int, d: c_int) !A {
        const sh = mlx.getShape(x);
        return sc.transpose(try sc.reshape(x, &.{ sh[0], sh[1], h, d }), &.{ 0, 2, 1, 3 });
    }
    /// [B,H,L,D] → [B,L,H*D]
    pub fn merge(sc: *Scope, x: A) !A {
        const sh = mlx.getShape(x);
        return sc.reshape(try sc.transpose(x, &.{ 0, 2, 1, 3 }), &.{ sh[0], sh[2], sh[1] * sh[3] });
    }
    pub fn div(sc: *Scope, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_divide(&o, x, y, sc.s), &o);
    }
    pub fn sumAxis(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_sum_axis(&o, x, axis, true, sc.s), &o);
    }
    pub fn maxAxis(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_max_axis(&o, x, axis, true, sc.s), &o);
    }
    pub fn exp(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_exp(&o, x, sc.s), &o);
    }
    pub fn sin(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_sin(&o, x, sc.s), &o);
    }
    pub fn cos(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_cos(&o, x, sc.s), &o);
    }
    /// A scalar in `x`'s dtype: an f32 scalar would promote a bf16 operand.
    pub fn scalarLike(sc: *Scope, x: A, v: f32) !A {
        return sc.astype(try sc.scalar(v), mlx.mlx_array_dtype(x));
    }
    pub fn mulLike(sc: *Scope, x: A, v: f32) !A {
        return sc.mul(x, try sc.scalarLike(x, v));
    }
    pub fn zeros(sc: *Scope, shape: []const c_int, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_zeros(&o, shape.ptr, shape.len, dt, sc.s), &o);
    }
    /// Rows of `table` at `ids` (axis 0).
    pub fn take(sc: *Scope, table: A, ids: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_take_axis(&o, table, ids, 0, sc.s), &o);
    }
    pub fn takeAxis(sc: *Scope, table: A, ids: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_take_axis(&o, table, ids, axis, sc.s), &o);
    }
    pub fn neg(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_negative(&o, x, sc.s), &o);
    }
    pub fn softmax(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_softmax_axis(&o, x, axis, false, sc.s), &o);
    }
    pub fn argpartition(sc: *Scope, x: A, kth: c_int, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_argpartition_axis(&o, x, kth, axis, sc.s), &o);
    }
    pub fn less(sc: *Scope, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_less(&o, x, y, sc.s), &o);
    }
    pub fn arange(sc: *Scope, start: c_int, stop: c_int, step: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_arange(&o, @floatFromInt(start), @floatFromInt(stop), @floatFromInt(step), .int32, sc.s), &o);
    }
    /// `lo`/`hi` zero rows added on one axis.
    pub fn padAxis(sc: *Scope, x: A, axis: c_int, lo: c_int, hi: c_int) !A {
        var o = mlx.mlx_array_new();
        const ax = [_]c_int{axis};
        return sc.res(mlx.mlx_pad(&o, x, &ax, 1, &[_]c_int{lo}, 1, &[_]c_int{hi}, 1, try sc.scalar(0), "constant", sc.s), &o);
    }
    /// `dst` with `update` written over [start, stop).
    pub fn sliceUpdate(sc: *Scope, dst: A, update: A, start: []const c_int, stop: []const c_int) !A {
        const st: [8]c_int = @splat(1);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_slice_update(&o, dst, update, start.ptr, start.len, stop.ptr, stop.len, &st, start.len, sc.s), &o);
    }
    /// Channels-last 1-D convolution, weight [out, k, in].
    pub fn conv1d(sc: *Scope, x: A, w: A, stride: c_int, padding: c_int, dilation: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_conv1d(&o, x, w, stride, padding, dilation, 1, sc.s), &o);
    }
    pub fn convT1d(sc: *Scope, x: A, w: A, stride: c_int, padding: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_conv_transpose1d(&o, x, w, stride, padding, 1, 0, 1, sc.s), &o);
    }
    /// x @ dequant(w)^T for an affine-quantized weight.
    pub fn qmm(sc: *Scope, x: A, w: A, scales: A, biases: A, group_size: u32, bits: u32) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_quantized_matmul(&o, x, w, scales, biases, true, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", sc.s), &o);
    }
    /// Rotate-half RoPE over the last `dims` of [B,H,T,D] at positions offset..offset+T.
    pub fn ropeAt(sc: *Scope, x: A, dims: c_int, base: f32, offset: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_fast_rope(&o, x, dims, false, mlx.mlx_optional_float.some(base), 1.0, offset, .{ .ctx = null }, sc.s), &o);
    }
    /// `mode` is "" (none) or "causal".
    pub fn sdpaMode(sc: *Scope, q: A, k: A, v: A, scale: f32, mode: [*:0]const u8) !A {
        var o = mlx.mlx_array_new();
        const none = A{ .ctx = null };
        return sc.res(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode, none, none, false, sc.s), &o);
    }
    /// A boolean (true = attend) or additive `mask`; a null handle is no mask.
    pub fn sdpaMasked(sc: *Scope, q: A, k: A, v: A, scale: f32, mask: A) !A {
        var o = mlx.mlx_array_new();
        const none = A{ .ctx = null };
        const mode: [*:0]const u8 = if (mask.ctx != null) "array" else "";
        return sc.res(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode, mask, none, false, sc.s), &o);
    }
    /// Channels-last 2-D convolution, weight [out, kh, kw, in], square stride and padding.
    pub fn conv2d(sc: *Scope, x: A, w: A, stride: c_int, padding: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_conv2d(&o, x, w, stride, stride, padding, padding, 1, 1, 1, sc.s), &o);
    }
    pub fn layerNorm(sc: *Scope, x: A, w: A, b: A, eps: f32) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_fast_layer_norm(&o, x, w, b, eps, sc.s), &o);
    }
    /// Exact GELU: x * (1 + erf(x / sqrt 2)) / 2.
    pub fn geluErf(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        const e = try sc.res(mlx.mlx_erf(&o, try sc.mulLike(x, std.math.sqrt1_2), sc.s), &o);
        return sc.mul(x, try sc.mulLike(try sc.add(e, try sc.scalarLike(e, 1.0)), 0.5));
    }
    pub fn fromI32(sc: *Scope, data: []const i32, shape: []const c_int) !A {
        return sc.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .int32));
    }
    pub fn fromF32(sc: *Scope, data: []const f32, shape: []const c_int) !A {
        return sc.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32));
    }
};
