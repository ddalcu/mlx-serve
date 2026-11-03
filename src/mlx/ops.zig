//! MLX for Zig: a Graph owns every array it makes until it ends; `call` runs any mlx-c op.
//! A helper that returns one array opens its own `scope()` and hands the result up with `keep`.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("ffi.zig");

pub const Array = mlx.mlx_array;
pub const Dtype = mlx.mlx_dtype;
pub const Stream = mlx.mlx_stream;

/// Handles a Graph made and has not freed yet; counted in tests only (leak checks).
var live_for_test = std.atomic.Value(i64).init(0);

pub fn liveForTest() i64 {
    return live_for_test.load(.monotonic);
}

fn counted(x: Array) Array {
    if (builtin.is_test) _ = live_for_test.fetchAdd(1, .monotonic);
    return x;
}

fn release(x: Array) void {
    if (builtin.is_test) _ = live_for_test.fetchSub(1, .monotonic);
    _ = mlx.mlx_array_free(x);
}

/// The handles one Graph owns: an inline buffer first, a heap list past it.
const Owned = struct {
    const INLINE = 64;
    buf: [INLINE]Array = undefined,
    n: usize = 0,
    heap: std.ArrayList(Array) = .empty,

    fn add(o: *Owned, a: std.mem.Allocator, x: Array) !void {
        if (o.n < INLINE) {
            o.buf[o.n] = x;
            o.n += 1;
        } else try o.heap.append(a, x);
    }

    /// Takes `x` out of the list (newest first); false when this Graph does not own it.
    fn remove(o: *Owned, x: Array) bool {
        var i = o.heap.items.len;
        while (i > 0) {
            i -= 1;
            if (o.heap.items[i].ctx == x.ctx) {
                _ = o.heap.swapRemove(i);
                return true;
            }
        }
        i = o.n;
        while (i > 0) {
            i -= 1;
            if (o.buf[i].ctx != x.ctx) continue;
            o.n -= 1;
            o.buf[i] = o.buf[o.n];
            if (o.heap.pop()) |last| {
                o.buf[o.n] = last;
                o.n += 1;
            }
            return true;
        }
        return false;
    }

    fn deinit(o: *Owned, a: std.mem.Allocator) void {
        for (o.buf[0..o.n]) |x| release(x);
        for (o.heap.items) |x| release(x);
        o.heap.deinit(a);
        o.* = .{};
    }
};

pub const Mask = union(enum) { none, causal, array: Array };

pub const Graph = struct {
    a: std.mem.Allocator,
    s: Stream,
    owned: Owned = .{},

    /// Never copy a Graph once it owns arrays: both copies would free them.
    pub fn init(a: std.mem.Allocator, s: Stream) Graph {
        return .{ .a = a, .s = s };
    }

    /// A child on the same stream: a helper's own lifetime.
    pub fn scope(g: *const Graph) Graph {
        return .init(g.a, g.s);
    }

    /// Frees every array not kept elsewhere.
    pub fn deinit(g: *Graph) void {
        g.owned.deinit(g.a);
    }

    /// Any mlx-c op of the shape `f(*out, args..., [stream])`; the compiler checks `args`
    /// against mlx-c's signature, so an MLX bump that changes an op is a compile error.
    pub fn call(g: *Graph, comptime f: anytype, args: anytype) !Array {
        const info = @typeInfo(@TypeOf(f)).@"fn";
        comptime std.debug.assert(info.param_types[0].? == *Array and info.return_type.? == c_int);
        const takes_stream = comptime info.param_types[info.param_types.len - 1].? == Stream;
        var out = counted(mlx.mlx_array_new());
        errdefer release(out);
        try mlx.check(if (takes_stream) @call(.auto, f, .{&out} ++ args ++ .{g.s}) else @call(.auto, f, .{&out} ++ args));
        try g.owned.add(g.a, out);
        return out;
    }

    /// Takes ownership of a handle made outside the Graph (freed here on failure).
    pub fn adopt(g: *Graph, x: Array) !Array {
        return g.adoptCounted(counted(x));
    }

    fn adoptCounted(g: *Graph, x: Array) !Array {
        g.owned.add(g.a, x) catch |e| {
            release(x);
            return e;
        };
        return x;
    }

    /// A reference on `x` this Graph owns: how a child's result outlives the child.
    pub fn keep(g: *Graph, x: Array) !Array {
        var r = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&r, x);
        return g.adopt(r);
    }

    /// A reference on `x` that leaves the Graph: the caller frees it.
    pub fn escape(_: *const Graph, x: Array) Array {
        var r = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&r, x);
        return r;
    }

    /// Frees `x` now (before an eval that must not pin it); `x` must be this Graph's.
    pub fn drop(g: *Graph, x: Array) void {
        const found = g.owned.remove(x);
        std.debug.assert(found);
        if (found) release(x);
    }

    // ── one line per op ──
    // zig fmt: off
    pub fn add(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_add, .{ x, y }); }
    pub fn sub(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_subtract, .{ x, y }); }
    pub fn mul(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_multiply, .{ x, y }); }
    pub fn div(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_divide, .{ x, y }); }
    pub fn matmul(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_matmul, .{ x, y }); }
    pub fn maximum(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_maximum, .{ x, y }); }
    pub fn minimum(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_minimum, .{ x, y }); }
    pub fn less(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_less, .{ x, y }); }
    pub fn greater(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_greater, .{ x, y }); }
    pub fn equal(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_equal, .{ x, y }); }
    pub fn where(g: *Graph, c: Array, x: Array, y: Array) !Array { return g.call(mlx.mlx_where, .{ c, x, y }); }
    pub fn exp(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_exp, .{x}); }
    pub fn log(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_log, .{x}); }
    pub fn sin(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_sin, .{x}); }
    pub fn cos(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_cos, .{x}); }
    pub fn tanh(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_tanh, .{x}); }
    pub fn sigmoid(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_sigmoid, .{x}); }
    pub fn sqrt(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_sqrt, .{x}); }
    pub fn rsqrt(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_rsqrt, .{x}); }
    pub fn square(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_square, .{x}); }
    pub fn abs(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_abs, .{x}); }
    pub fn neg(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_negative, .{x}); }
    pub fn erf(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_erf, .{x}); }
    pub fn astype(g: *Graph, x: Array, dt: Dtype) !Array { return g.call(mlx.mlx_astype, .{ x, dt }); }
    pub fn contiguous(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_contiguous, .{ x, false }); }
    pub fn reshape(g: *Graph, x: Array, sh: []const c_int) !Array { return g.call(mlx.mlx_reshape, .{ x, sh.ptr, sh.len }); }
    pub fn transpose(g: *Graph, x: Array, axes: []const c_int) !Array { return g.call(mlx.mlx_transpose_axes, .{ x, axes.ptr, axes.len }); }
    pub fn broadcast(g: *Graph, x: Array, sh: []const c_int) !Array { return g.call(mlx.mlx_broadcast_to, .{ x, sh.ptr, sh.len }); }
    pub fn expandDims(g: *Graph, x: Array, axis: c_int) !Array { return g.call(mlx.mlx_expand_dims, .{ x, axis }); }
    pub fn take(g: *Graph, x: Array, ids: Array, axis: c_int) !Array { return g.call(mlx.mlx_take_axis, .{ x, ids, axis }); }
    pub fn takeAlong(g: *Graph, x: Array, ids: Array, axis: c_int) !Array { return g.call(mlx.mlx_take_along_axis, .{ x, ids, axis }); }
    pub fn repeat(g: *Graph, x: Array, n: c_int, axis: c_int) !Array { return g.call(mlx.mlx_repeat_axis, .{ x, n, axis }); }
    pub fn sum(g: *Graph, x: Array, axis: c_int, keepdims: bool) !Array { return g.call(mlx.mlx_sum_axis, .{ x, axis, keepdims }); }
    pub fn mean(g: *Graph, x: Array, axis: c_int, keepdims: bool) !Array { return g.call(mlx.mlx_mean_axis, .{ x, axis, keepdims }); }
    pub fn max(g: *Graph, x: Array, axis: c_int, keepdims: bool) !Array { return g.call(mlx.mlx_max_axis, .{ x, axis, keepdims }); }
    pub fn argmax(g: *Graph, x: Array, axis: c_int, keepdims: bool) !Array { return g.call(mlx.mlx_argmax_axis, .{ x, axis, keepdims }); }
    pub fn argsort(g: *Graph, x: Array, axis: c_int) !Array { return g.call(mlx.mlx_argsort_axis, .{ x, axis }); }
    pub fn argpartition(g: *Graph, x: Array, kth: c_int, axis: c_int) !Array { return g.call(mlx.mlx_argpartition_axis, .{ x, kth, axis }); }
    pub fn softmax(g: *Graph, x: Array, axis: c_int, precise: bool) !Array { return g.call(mlx.mlx_softmax_axis, .{ x, axis, precise }); }
    pub fn zeros(g: *Graph, sh: []const c_int, dt: Dtype) !Array { return g.call(mlx.mlx_zeros, .{ sh.ptr, sh.len, dt }); }
    pub fn ones(g: *Graph, sh: []const c_int, dt: Dtype) !Array { return g.call(mlx.mlx_ones, .{ sh.ptr, sh.len, dt }); }
    pub fn full(g: *Graph, sh: []const c_int, v: Array, dt: Dtype) !Array { return g.call(mlx.mlx_full, .{ sh.ptr, sh.len, v, dt }); }
    pub fn arange(g: *Graph, start: f64, stop: f64, step: f64, dt: Dtype) !Array { return g.call(mlx.mlx_arange, .{ start, stop, step, dt }); }
    pub fn rmsNorm(g: *Graph, x: Array, w: Array, eps: f32) !Array { return g.call(mlx.mlx_fast_rms_norm, .{ x, w, eps }); }
    pub fn layerNorm(g: *Graph, x: Array, w: Array, b: Array, eps: f32) !Array { return g.call(mlx.mlx_fast_layer_norm, .{ x, w, b, eps }); }
    /// Rotate-half RoPE over the last `dims` at positions offset.. (scale 1, no freqs table).
    pub fn rope(g: *Graph, x: Array, dims: c_int, base: f32, offset: c_int) !Array { return g.call(mlx.mlx_fast_rope, .{ x, dims, false, mlx.mlx_optional_float.some(base), 1.0, offset, Array{} }); }
    pub fn sdpa(g: *Graph, q: Array, k: Array, v: Array, scale: f32, mask: Mask) !Array {
        const mode: [*:0]const u8 = switch (mask) { .none => "", .causal => "causal", .array => "array" };
        return g.call(mlx.mlx_fast_scaled_dot_product_attention, .{ q, k, v, scale, mode, if (mask == .array) mask.array else Array{}, Array{}, false });
    }
    /// x @ dequant(w)^T for an affine-quantized `w`.
    pub fn qmm(g: *Graph, x: Array, w: Array, scales: Array, biases: Array, group_size: c_int, bits: c_int) !Array { return g.call(mlx.mlx_quantized_matmul, .{ x, w, scales, biases, true, mlx.mlx_optional_int.some(group_size), mlx.mlx_optional_int.some(bits), "affine" }); }
    /// Channels-last; weight [out, k, in].
    pub fn conv1d(g: *Graph, x: Array, w: Array, stride: c_int, padding: c_int, dilation: c_int, groups: c_int) !Array { return g.call(mlx.mlx_conv1d, .{ x, w, stride, padding, dilation, groups }); }
    pub fn convTranspose1d(g: *Graph, x: Array, w: Array, stride: c_int, padding: c_int, dilation: c_int, output_padding: c_int, groups: c_int) !Array { return g.call(mlx.mlx_conv_transpose1d, .{ x, w, stride, padding, dilation, output_padding, groups }); }
    /// Channels-last; weight [out, kh, kw, in]; square stride, padding and dilation.
    pub fn conv2d(g: *Graph, x: Array, w: Array, stride: c_int, padding: c_int, dilation: c_int, groups: c_int) !Array { return g.call(mlx.mlx_conv2d, .{ x, w, stride, stride, padding, padding, dilation, dilation, groups }); }

    // zig fmt: on

    /// x[start:stop] per axis, unit strides.
    pub fn slice(g: *Graph, x: Array, start: []const c_int, stop: []const c_int) !Array {
        const ones8: [8]c_int = @splat(1);
        return g.call(mlx.mlx_slice, .{ x, start.ptr, start.len, stop.ptr, stop.len, &ones8, start.len });
    }

    /// x[..., start:stop, ...] on one axis.
    pub fn sliceAxis(g: *Graph, x: Array, axis: usize, start: c_int, stop: c_int) !Array {
        const sh = shape(x);
        var lo: [8]c_int = @splat(0);
        var hi: [8]c_int = undefined;
        @memcpy(hi[0..sh.len], sh);
        lo[axis] = start;
        hi[axis] = stop;
        return g.slice(x, lo[0..sh.len], hi[0..sh.len]);
    }

    /// `dst` with `update` written over [start, stop), unit strides.
    pub fn sliceUpdate(g: *Graph, dst: Array, update: Array, start: []const c_int, stop: []const c_int) !Array {
        const ones8: [8]c_int = @splat(1);
        return g.call(mlx.mlx_slice_update, .{ dst, update, start.ptr, start.len, stop.ptr, stop.len, &ones8, start.len });
    }

    pub fn concat(g: *Graph, xs: []const Array, axis: c_int) !Array {
        const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(v);
        return g.call(mlx.mlx_concatenate_axis, .{ v, axis });
    }

    /// `lo`/`hi` zero rows added on one axis (the pad value is cast to x's dtype by MLX).
    pub fn padAxis(g: *Graph, x: Array, axis: c_int, lo: c_int, hi: c_int) !Array {
        const zero = mlx.mlx_array_new_float(0);
        defer _ = mlx.mlx_array_free(zero);
        return g.call(mlx.mlx_pad, .{ x, &[_]c_int{axis}, 1, &[_]c_int{lo}, 1, &[_]c_int{hi}, 1, zero, "constant" });
    }

    /// Two keys from one.
    pub fn splitKey(g: *Graph, key: Array) ![2]Array {
        var k1 = counted(mlx.mlx_array_new());
        var k2 = counted(mlx.mlx_array_new());
        const rc = mlx.mlx_random_split(&k1, &k2, key, g.s);
        g.owned.add(g.a, k1) catch |e| {
            release(k1);
            release(k2);
            return e;
        };
        _ = try g.adoptCounted(k2);
        try mlx.check(rc);
        return .{ k1, k2 };
    }

    // ── scalars: in the operand's dtype, never a bare f32 that promotes bf16 ──

    /// A 0-d `v` in `dt` (cast from f32 only when `dt` is not f32).
    pub fn scalar(g: *Graph, v: f32, dt: Dtype) !Array {
        const raw = mlx.mlx_array_new_float(v);
        if (dt == .float32) return g.adopt(raw);
        defer _ = mlx.mlx_array_free(raw);
        return g.astype(raw, dt);
    }
    pub fn scalarLike(g: *Graph, x: Array, v: f32) !Array {
        return g.scalar(v, dtype(x));
    }
    pub fn addScalar(g: *Graph, x: Array, v: f32) !Array {
        return g.add(x, try g.scalarLike(x, v));
    }
    pub fn mulScalar(g: *Graph, x: Array, v: f32) !Array {
        return g.mul(x, try g.scalarLike(x, v));
    }

    /// A host slice copied into an array of the matching dtype.
    pub fn fromSlice(g: *Graph, comptime T: type, data: []const T, sh: []const c_int) !Array {
        const dt: Dtype = switch (T) {
            f32 => .float32,
            i32 => .int32,
            u32 => .uint32,
            bool => .bool_,
            else => @compileError("fromSlice: no dtype for " ++ @typeName(T)),
        };
        return g.adopt(mlx.mlx_array_new_data(data.ptr, sh.ptr, @intCast(sh.len), dt));
    }

    // ── composites ──

    pub fn silu(g: *Graph, x: Array) !Array {
        var c = g.scope();
        defer c.deinit();
        return g.keep(try c.mul(x, try c.sigmoid(x)));
    }

    /// x @ w^T (+ b): PyTorch Linear over the last axis.
    pub fn linear(g: *Graph, x: Array, w: Array, b: ?Array) !Array {
        var c = g.scope();
        defer c.deinit();
        const y = try c.matmul(x, try c.transpose(w, &.{ 1, 0 }));
        return g.keep(if (b) |bb| try c.add(y, bb) else y);
    }

    /// [B,L,H*D] → [B,H,L,D]
    pub fn splitHeads(g: *Graph, x: Array, h: c_int, d: c_int) !Array {
        var c = g.scope();
        defer c.deinit();
        const sh = shape(x);
        return g.keep(try c.transpose(try c.reshape(x, &.{ sh[0], sh[1], h, d }), &.{ 0, 2, 1, 3 }));
    }

    /// [B,H,L,D] → [B,L,H*D]
    pub fn mergeHeads(g: *Graph, x: Array) !Array {
        var c = g.scope();
        defer c.deinit();
        const sh = shape(x);
        return g.keep(try c.reshape(try c.transpose(x, &.{ 0, 2, 1, 3 }), &.{ sh[0], sh[2], sh[1] * sh[3] }));
    }
};

pub fn shape(x: Array) []const c_int {
    return mlx.getShape(x);
}

pub fn dtype(x: Array) Dtype {
    return mlx.mlx_array_dtype(x);
}

pub fn eval(x: Array) !void {
    try mlx.check(mlx.mlx_array_eval(x));
}

/// One eval over several outputs (they share the work of their common inputs).
pub fn evalMany(xs: []const Array) !void {
    const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
}

// ── tests ──

const testing = std.testing;

fn cpu() Stream {
    return mlx.mlx_default_cpu_stream_new();
}

/// The bytes of `x`, row-major (evaluates it).
fn bytesOf(a: std.mem.Allocator, x: Array) ![]u8 {
    var g = Graph.init(a, cpu());
    defer g.deinit();
    const c = try g.contiguous(x);
    try eval(c);
    const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
    const p = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    return a.dupe(u8, p[0..n]);
}

/// `got` and `want` agree in dtype, shape and every bit, and a float `want` is finite.
pub fn expectBitEqual(got: Array, want: Array) !void {
    try testing.expectEqual(dtype(want), dtype(got));
    try testing.expectEqualSlices(c_int, shape(want), shape(got));
    const a = try bytesOf(testing.allocator, got);
    defer testing.allocator.free(a);
    const b = try bytesOf(testing.allocator, want);
    defer testing.allocator.free(b);
    try testing.expectEqualSlices(u8, b, a);
    switch (dtype(want)) {
        .float16, .bfloat16, .float32 => {
            var g = Graph.init(testing.allocator, cpu());
            defer g.deinit();
            const f = try g.contiguous(try g.astype(want, .float32));
            try eval(f);
            const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
            for (d[0..mlx.mlx_array_size(f)]) |v| try testing.expect(std.math.isFinite(v));
        },
        else => {},
    }
}

/// The wrapper's output equals the raw mlx-c call `f(&out, args..., s)`.
fn expectSameAs(got: Array, comptime f: anytype, args: anytype, s: Stream) !void {
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(@call(.auto, f, .{&want} ++ args ++ .{s}));
    try expectBitEqual(got, want);
}

fn randn(g: *Graph, sh: []const c_int, seed: u64) !Array {
    const key = try g.call(mlx.mlx_random_key, .{seed});
    return g.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, Dtype.float32, @as(f32, 0), @as(f32, 1), key });
}

test "ops: every one-line wrapper is bit-equal to its raw mlx-c call" {
    const s = cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const x = try randn(&g, &.{ 2, 3, 4 }, 1);
    const y = try randn(&g, &.{ 2, 3, 4 }, 2);
    const p = try g.abs(x);
    const m = try randn(&g, &.{ 2, 4, 3 }, 3);
    const cond = try g.less(x, y);
    const ids = try g.fromSlice(i32, &.{ 2, 0, 1 }, &.{3});
    const none = Array{};

    try expectSameAs(try g.add(x, y), mlx.mlx_add, .{ x, y }, s);
    try expectSameAs(try g.sub(x, y), mlx.mlx_subtract, .{ x, y }, s);
    try expectSameAs(try g.mul(x, y), mlx.mlx_multiply, .{ x, y }, s);
    try expectSameAs(try g.div(x, y), mlx.mlx_divide, .{ x, y }, s);
    try expectSameAs(try g.matmul(x, m), mlx.mlx_matmul, .{ x, m }, s);
    try expectSameAs(try g.maximum(x, y), mlx.mlx_maximum, .{ x, y }, s);
    try expectSameAs(try g.minimum(x, y), mlx.mlx_minimum, .{ x, y }, s);
    try expectSameAs(cond, mlx.mlx_less, .{ x, y }, s);
    try expectSameAs(try g.greater(x, y), mlx.mlx_greater, .{ x, y }, s);
    try expectSameAs(try g.equal(x, x), mlx.mlx_equal, .{ x, x }, s);
    try expectSameAs(try g.where(cond, x, y), mlx.mlx_where, .{ cond, x, y }, s);
    try expectSameAs(try g.exp(x), mlx.mlx_exp, .{x}, s);
    try expectSameAs(try g.log(p), mlx.mlx_log, .{p}, s);
    try expectSameAs(try g.sin(x), mlx.mlx_sin, .{x}, s);
    try expectSameAs(try g.cos(x), mlx.mlx_cos, .{x}, s);
    try expectSameAs(try g.tanh(x), mlx.mlx_tanh, .{x}, s);
    try expectSameAs(try g.sigmoid(x), mlx.mlx_sigmoid, .{x}, s);
    try expectSameAs(try g.sqrt(p), mlx.mlx_sqrt, .{p}, s);
    try expectSameAs(try g.rsqrt(p), mlx.mlx_rsqrt, .{p}, s);
    try expectSameAs(try g.square(x), mlx.mlx_square, .{x}, s);
    try expectSameAs(p, mlx.mlx_abs, .{x}, s);
    try expectSameAs(try g.neg(x), mlx.mlx_negative, .{x}, s);
    try expectSameAs(try g.erf(x), mlx.mlx_erf, .{x}, s);
    try expectSameAs(try g.astype(x, .bfloat16), mlx.mlx_astype, .{ x, Dtype.bfloat16 }, s);
    try expectSameAs(try g.contiguous(x), mlx.mlx_contiguous, .{ x, false }, s);
    try expectSameAs(try g.reshape(x, &.{ 6, 4 }), mlx.mlx_reshape, .{ x, &[_]c_int{ 6, 4 }, @as(usize, 2) }, s);
    try expectSameAs(try g.transpose(x, &.{ 2, 0, 1 }), mlx.mlx_transpose_axes, .{ x, &[_]c_int{ 2, 0, 1 }, @as(usize, 3) }, s);
    try expectSameAs(try g.broadcast(p, &.{ 5, 2, 3, 4 }), mlx.mlx_broadcast_to, .{ p, &[_]c_int{ 5, 2, 3, 4 }, @as(usize, 4) }, s);
    try expectSameAs(try g.expandDims(x, 1), mlx.mlx_expand_dims, .{ x, @as(c_int, 1) }, s);
    try expectSameAs(try g.take(x, ids, 1), mlx.mlx_take_axis, .{ x, ids, @as(c_int, 1) }, s);
    const along = try g.argsort(x, -1);
    try expectSameAs(along, mlx.mlx_argsort_axis, .{ x, @as(c_int, -1) }, s);
    try expectSameAs(try g.takeAlong(x, along, -1), mlx.mlx_take_along_axis, .{ x, along, @as(c_int, -1) }, s);
    try expectSameAs(try g.repeat(x, 2, 1), mlx.mlx_repeat_axis, .{ x, @as(c_int, 2), @as(c_int, 1) }, s);
    try expectSameAs(try g.sum(x, 1, true), mlx.mlx_sum_axis, .{ x, @as(c_int, 1), true }, s);
    try expectSameAs(try g.mean(x, -1, false), mlx.mlx_mean_axis, .{ x, @as(c_int, -1), false }, s);
    try expectSameAs(try g.max(x, 2, true), mlx.mlx_max_axis, .{ x, @as(c_int, 2), true }, s);
    try expectSameAs(try g.argmax(x, 2, false), mlx.mlx_argmax_axis, .{ x, @as(c_int, 2), false }, s);
    try expectSameAs(try g.argpartition(x, 1, -1), mlx.mlx_argpartition_axis, .{ x, @as(c_int, 1), @as(c_int, -1) }, s);
    try expectSameAs(try g.softmax(x, -1, true), mlx.mlx_softmax_axis, .{ x, @as(c_int, -1), true }, s);
    try expectSameAs(try g.zeros(&.{ 2, 3 }, .bfloat16), mlx.mlx_zeros, .{ &[_]c_int{ 2, 3 }, @as(usize, 2), Dtype.bfloat16 }, s);
    try expectSameAs(try g.ones(&.{ 3, 2 }, .float32), mlx.mlx_ones, .{ &[_]c_int{ 3, 2 }, @as(usize, 2), Dtype.float32 }, s);
    const seven = try g.scalar(7, .float32);
    try expectSameAs(try g.full(&.{ 2, 2 }, seven, .int32), mlx.mlx_full, .{ &[_]c_int{ 2, 2 }, @as(usize, 2), seven, Dtype.int32 }, s);
    try expectSameAs(try g.arange(1, 9, 2, .int32), mlx.mlx_arange, .{ @as(f64, 1), @as(f64, 9), @as(f64, 2), Dtype.int32 }, s);

    const w4 = try randn(&g, &.{4}, 4);
    const b4 = try randn(&g, &.{4}, 5);
    try expectSameAs(try g.rmsNorm(x, w4, 1e-5), mlx.mlx_fast_rms_norm, .{ x, w4, @as(f32, 1e-5) }, s);
    try expectSameAs(try g.layerNorm(x, w4, b4, 1e-5), mlx.mlx_fast_layer_norm, .{ x, w4, b4, @as(f32, 1e-5) }, s);
    const q = try randn(&g, &.{ 1, 2, 5, 8 }, 6);
    const k = try randn(&g, &.{ 1, 2, 5, 8 }, 7);
    const v = try randn(&g, &.{ 1, 2, 5, 8 }, 8);
    try expectSameAs(try g.rope(q, 8, 10000, 3), mlx.mlx_fast_rope, .{ q, @as(c_int, 8), false, mlx.mlx_optional_float.some(10000), @as(f32, 1), @as(c_int, 3), none }, s);
    const sdpa = mlx.mlx_fast_scaled_dot_product_attention;
    try expectSameAs(try g.sdpa(q, k, v, 0.3, .none), sdpa, .{ q, k, v, @as(f32, 0.3), "", none, none, false }, s);
    try expectSameAs(try g.sdpa(q, k, v, 0.3, .causal), sdpa, .{ q, k, v, @as(f32, 0.3), "causal", none, none, false }, s);
    const bias = try randn(&g, &.{ 5, 5 }, 9);
    try expectSameAs(try g.sdpa(q, k, v, 0.3, .{ .array = bias }), sdpa, .{ q, k, v, @as(f32, 0.3), "array", bias, none, false }, s);

    const wq = try randn(&g, &.{ 8, 64 }, 10);
    const xq = try randn(&g, &.{ 3, 64 }, 11);
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, wq, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "affine", none, s));
    var qw = mlx.mlx_array_new();
    var qs = mlx.mlx_array_new();
    var qb = mlx.mlx_array_new();
    _ = mlx.mlx_vector_array_get(&qw, qv, 0);
    _ = mlx.mlx_vector_array_get(&qs, qv, 1);
    _ = mlx.mlx_vector_array_get(&qb, qv, 2);
    _ = try g.adopt(qw);
    _ = try g.adopt(qs);
    _ = try g.adopt(qb);
    try expectSameAs(try g.qmm(xq, qw, qs, qb, 32, 4), mlx.mlx_quantized_matmul, .{ xq, qw, qs, qb, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "affine" }, s);

    const sig = try randn(&g, &.{ 1, 10, 4 }, 12);
    const cw = try randn(&g, &.{ 6, 3, 4 }, 13);
    try expectSameAs(try g.conv1d(sig, cw, 1, 1, 1, 1), mlx.mlx_conv1d, .{ sig, cw, @as(c_int, 1), @as(c_int, 1), @as(c_int, 1), @as(c_int, 1) }, s);
    const tw = try randn(&g, &.{ 6, 4, 4 }, 14);
    try expectSameAs(try g.convTranspose1d(sig, tw, 2, 1, 1, 0, 1), mlx.mlx_conv_transpose1d, .{ sig, tw, @as(c_int, 2), @as(c_int, 1), @as(c_int, 1), @as(c_int, 0), @as(c_int, 1) }, s);
    const img = try randn(&g, &.{ 1, 6, 6, 4 }, 15);
    const kw = try randn(&g, &.{ 5, 3, 3, 4 }, 16);
    try expectSameAs(try g.conv2d(img, kw, 2, 1, 1, 1), mlx.mlx_conv2d, .{ img, kw, @as(c_int, 2), @as(c_int, 2), @as(c_int, 1), @as(c_int, 1), @as(c_int, 1), @as(c_int, 1), @as(c_int, 1) }, s);

    const one3 = [_]c_int{ 1, 1, 1 };
    try expectSameAs(try g.slice(x, &.{ 0, 1, 1 }, &.{ 2, 3, 3 }), mlx.mlx_slice, .{ x, &[_]c_int{ 0, 1, 1 }, @as(usize, 3), &[_]c_int{ 2, 3, 3 }, @as(usize, 3), &one3, @as(usize, 3) }, s);
    try expectSameAs(try g.sliceAxis(x, 2, 1, 3), mlx.mlx_slice, .{ x, &[_]c_int{ 0, 0, 1 }, @as(usize, 3), &[_]c_int{ 2, 3, 3 }, @as(usize, 3), &one3, @as(usize, 3) }, s);
    const upd = try randn(&g, &.{ 2, 1, 4 }, 17);
    try expectSameAs(try g.sliceUpdate(x, upd, &.{ 0, 1, 0 }, &.{ 2, 2, 4 }), mlx.mlx_slice_update, .{ x, upd, &[_]c_int{ 0, 1, 0 }, @as(usize, 3), &[_]c_int{ 2, 2, 4 }, @as(usize, 3), &one3, @as(usize, 3) }, s);
    const pair = mlx.mlx_vector_array_new_data(&[_]Array{ x, y }, 2);
    defer _ = mlx.mlx_vector_array_free(pair);
    try expectSameAs(try g.concat(&.{ x, y }, 1), mlx.mlx_concatenate_axis, .{ pair, @as(c_int, 1) }, s);
    const zero = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(zero);
    try expectSameAs(try g.padAxis(x, 1, 2, 1), mlx.mlx_pad, .{ x, &[_]c_int{1}, @as(usize, 1), &[_]c_int{2}, @as(usize, 1), &[_]c_int{1}, @as(usize, 1), zero, "constant" }, s);
}

test "ops: composites equal the op chains they name" {
    const s = cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const x = try randn(&g, &.{ 1, 3, 8 }, 20);
    const w = try randn(&g, &.{ 6, 8 }, 21);
    const b = try randn(&g, &.{6}, 22);
    try expectBitEqual(try g.silu(x), try g.mul(x, try g.sigmoid(x)));
    try expectBitEqual(try g.linear(x, w, b), try g.add(try g.matmul(x, try g.transpose(w, &.{ 1, 0 })), b));
    try expectBitEqual(try g.linear(x, w, null), try g.matmul(x, try g.transpose(w, &.{ 1, 0 })));
    const h = try g.splitHeads(x, 2, 4);
    try testing.expectEqualSlices(c_int, &.{ 1, 2, 3, 4 }, shape(h));
    try expectBitEqual(h, try g.transpose(try g.reshape(x, &.{ 1, 3, 2, 4 }), &.{ 0, 2, 1, 3 }));
    try expectBitEqual(try g.mergeHeads(h), x);
    const keys = try g.splitKey(try g.call(mlx.mlx_random_key, .{@as(u64, 5)}));
    var k1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k1);
    var k2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k2);
    var k0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k0);
    try mlx.check(mlx.mlx_random_key(&k0, 5));
    try mlx.check(mlx.mlx_random_split(&k1, &k2, k0, s));
    try expectBitEqual(keys[0], k1);
    try expectBitEqual(keys[1], k2);
}

test "ops: scalars take the operand's dtype; an f32 scalar is a bare f32 array" {
    const s = cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const xb = try g.astype(try randn(&g, &.{4}, 30), .bfloat16);
    try testing.expectEqual(Dtype.bfloat16, dtype(try g.scalar(0.5, .bfloat16)));
    try testing.expectEqual(Dtype.bfloat16, dtype(try g.addScalar(xb, 1e-3)));
    try testing.expectEqual(Dtype.bfloat16, dtype(try g.mulScalar(xb, 3)));
    const raw = mlx.mlx_array_new_float(0.25);
    defer _ = mlx.mlx_array_free(raw);
    try expectBitEqual(try g.scalar(0.25, .float32), raw);
    const xf = try randn(&g, &.{4}, 31);
    try expectSameAs(try g.mulScalar(xf, 0.25), mlx.mlx_multiply, .{ xf, raw }, s);
    const v = try g.fromSlice(f32, &.{ 1, -2, 3.5 }, &.{3});
    try testing.expectEqual(Dtype.float32, dtype(v));
    const u = try g.fromSlice(u32, &.{ 7, 8 }, &.{ 1, 2 });
    try testing.expectEqual(Dtype.uint32, dtype(u));
}

test "ops: call checks arity and appends the stream only where mlx-c takes one" {
    var g = Graph.init(testing.allocator, cpu());
    defer g.deinit();
    const key = try g.call(mlx.mlx_random_key, .{@as(u64, 3)}); // no stream parameter
    const n = try g.call(mlx.mlx_random_normal, .{ &[_]c_int{2}, @as(usize, 1), Dtype.float32, @as(f32, 0), @as(f32, 1), key });
    try testing.expectEqualSlices(c_int, &.{2}, shape(n));
}

test "ops: deinit frees every array; keep survives its child; drop frees now" {
    const before = liveForTest();
    {
        var g = Graph.init(testing.allocator, cpu());
        defer g.deinit();
        for (0..200) |i| _ = try g.scalar(@floatFromInt(i), .float32); // past the inline buffer
        try testing.expectEqual(before + 200, liveForTest());

        const kept = blk: {
            var c = g.scope();
            defer c.deinit();
            const t = try c.addScalar(try c.scalar(2, .float32), 1);
            break :blk try g.keep(t);
        };
        try testing.expectEqual(before + 201, liveForTest());
        try eval(kept);
        var got: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&got, kept));
        try testing.expectEqual(@as(f32, 3), got);

        const tmp = try g.scalar(9, .float32);
        g.drop(tmp);
        g.drop(kept);
        try testing.expectEqual(before + 200, liveForTest());
        // Dropping inline entries while the heap holds more keeps every survivor reachable.
        var first = g.owned.buf[0];
        g.drop(first);
        first = g.owned.buf[5];
        g.drop(first);
        try testing.expectEqual(before + 198, liveForTest());
    }
    try testing.expectEqual(before, liveForTest());
}

test "ops: an error mid-chain leaks nothing" {
    const before = liveForTest();
    var k: u64 = 1;
    while (true) : (k += 1) {
        var g = Graph.init(testing.allocator, cpu());
        mlx.fault.arm(k);
        const r = chain(&g);
        const fired = mlx.fault.didFire();
        mlx.fault.disarm();
        g.deinit();
        try testing.expectEqual(before, liveForTest());
        if (!fired) {
            _ = try r;
            break;
        }
        try testing.expectError(error.MlxError, r);
    }
    try testing.expect(k > 5);
}

fn chain(g: *Graph) !Array {
    const x = try randn(g, &.{ 2, 8 }, 40);
    const w = try randn(g, &.{ 4, 8 }, 41);
    const h = try g.splitHeads(try g.reshape(try g.linear(x, w, null), &.{ 1, 2, 4 }), 2, 2);
    return g.silu(try g.mergeHeads(h));
}

test "ops: the allocator failing on spill frees the op's handle" {
    var fa = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    const before = liveForTest();
    {
        var g = Graph.init(fa.allocator(), cpu());
        defer g.deinit();
        const key = try g.call(mlx.mlx_random_key, .{@as(u64, 1)});
        for (1..Owned.INLINE) |_| _ = try g.scalar(1, .float32);
        try testing.expectError(error.OutOfMemory, g.sigmoid(key));
        try testing.expectError(error.OutOfMemory, g.scalar(2, .float32));
        try testing.expectError(error.OutOfMemory, g.splitKey(key));
    }
    try testing.expectEqual(before, liveForTest());
}
