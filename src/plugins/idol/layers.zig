//! Idol's shared pieces (speridlabs/idol, Apache-2.0): the reference's mixed precision (Linear in the block dtype,
//! norms, embeddings and RoPE angles in f32), the per-head RMS norm, the position embeddings and the attention
//! blocks the three transformers (denoiser, octree decoder, Gaussian decoder) compose.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const scope_mod = @import("../../mlx/scope.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const graph_call = @import("../../sdk/media/graph_call.zig").call;
const gelu_blk = @import("../../sdk/blocks/mlp/gelu.zig");
const glue = @import("glue.zig");

pub const A = mlx.mlx_array;
pub const Scope = scope_mod.Scope;
pub const Linear = scope_mod.Linear;
pub const Weights = model_mod.Weights;
pub const NONE = A{ .ctx = null };

pub const LN_EPS: f32 = 1e-6;
pub const HEAD_NORM_EPS: f32 = 1e-12;
pub const FINAL_LN_EPS: f32 = 1e-5;
pub const HEAD_DIM: usize = 64;

/// `x` cast to `like`'s dtype.
pub fn asLike(sc: *Scope, x: A, like: A) !A {
    const dt = mlx.mlx_array_dtype(like);
    return if (mlx.mlx_array_dtype(x) == dt) x else sc.astype(x, dt);
}

pub fn asDtype(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
    return if (mlx.mlx_array_dtype(x) == dt) x else sc.astype(x, dt);
}

/// The reference's gelu_approx, its constants in x's dtype.
pub fn geluTanh(sc: *Scope, x: A) !A {
    return sc.keep(try graph_call(sc.s, gelu_blk.geluTanh, .{ x, mlx.mlx_array_dtype(x), gelu_blk.SQRT_2_PI_LOW }));
}

/// `LayerNorm32`: statistics in f32 whatever x's dtype, no affine unless `w`/`b` are given, out in x's dtype.
pub fn layerNorm32(sc: *Scope, x: A, w: A, b: A, eps: f32) !A {
    return sc.layerNorm(x, w, b, eps);
}

/// `x * (1 + scale) + shift` with `scale` and `shift` `[1, C]`, broadcast over tokens; the 1 in x's dtype.
pub fn modulate(sc: *Scope, x: A, shift: A, scale: A) !A {
    const one = try sc.scalarLike(x, 1);
    const s = try sc.add(try asLike(sc, scale, x), one);
    return sc.add(try sc.mul(x, s), try asLike(sc, shift, x));
}

/// `modulation` `[1, 6C]` split into its six `[1, 1, C]` chunks (shift/scale/gate for attention then MLP).
pub fn chunks6(sc: *Scope, m: A, c: usize) ![6]A {
    var out: [6]A = undefined;
    for (&out, 0..) |*p, i| p.* = try sc.reshape(try sc.slice(m, 1, @intCast(i * c), @intCast((i + 1) * c)), &.{ 1, 1, @intCast(c) });
    return out;
}

/// `MultiHeadRMSNorm`: `x` `[1, L, H, D]` normalized per head in f32 and scaled by `gamma [H, D]`, back in x's dtype.
pub fn headNorm(sc: *Scope, x: A, gamma: A, ones: A) !A {
    const dt = mlx.mlx_array_dtype(x);
    const n = try sc.rms(try asDtype(sc, x, .float32), ones, HEAD_NORM_EPS);
    return asDtype(sc, try sc.mul(n, try asDtype(sc, gamma, .float32)), dt);
}

/// Attention over heads-major `[1, H, L, D]` q and `[1, H, Lk, D]` k, v, `mask` an additive key bias or a null
/// handle; returns `[1, L, H*D]`.
pub fn attendHeads(sc: *Scope, q: A, k: A, v: A, mask: A) !A {
    const sh = mlx.getShape(q);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(sh[3])));
    const o = try sc.sdpaMasked(q, k, v, scale, mask);
    return sc.reshape(try sc.transpose(o, &.{ 0, 2, 1, 3 }), &.{ sh[0], sh[2], sh[1] * sh[3] });
}

/// `attendHeads` over `[1, L, H, D]` q and `[1, Lk, H, D]` k, v.
pub fn attend(sc: *Scope, q: A, k: A, v: A, mask: A) !A {
    const perm = [_]c_int{ 0, 2, 1, 3 };
    return attendHeads(sc, try sc.transpose(q, &perm), try sc.transpose(k, &perm), try sc.transpose(v, &perm), mask);
}

/// One column group `[1, L, n, D]` of a fused `[1, L, k*n*D]` projection, `i` of `k`.
pub fn part(sc: *Scope, y: A, k: c_int, i: c_int, heads: c_int, d: c_int) !A {
    const l = mlx.getShape(y)[1];
    const r = try sc.reshape(y, &.{ 1, l, k, heads, d });
    return sc.reshape(try sc.slice(r, 2, i, i + 1), &.{ 1, l, heads, d });
}

/// The decoders' `MultiHeadAttention`: RMS-normalized q and k, self (`to_qkv`) or cross (`to_q` + `to_kv`).
pub const Mha = struct {
    qkv: ?Linear,
    q: ?Linear,
    kv: ?Linear,
    out: Linear,
    q_gamma: A,
    k_gamma: A,

    pub fn load(w: *const Weights, comptime fmt: []const u8, args: anytype, channels: u32, ctx_channels: u32, cross: bool) !Mha {
        var buf: [96]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, fmt, args);
        return .{
            .qkv = if (cross) null else try scope_mod.linear(w, "{s}.to_qkv", .{p}, channels),
            .q = if (cross) try scope_mod.linear(w, "{s}.to_q", .{p}, channels) else null,
            .kv = if (cross) try scope_mod.linear(w, "{s}.to_kv", .{p}, ctx_channels) else null,
            .out = try scope_mod.linear(w, "{s}.to_out", .{p}, channels),
            .q_gamma = try scope_mod.getW(w, "{s}.q_rms_norm.gamma", .{p}),
            .k_gamma = try scope_mod.getW(w, "{s}.k_rms_norm.gamma", .{p}),
        };
    }

    pub fn forward(self: *const Mha, sc: *Scope, x: A, context: ?A, ones: A) !A {
        const heads = mlx.getShape(self.q_gamma)[0];
        const d = mlx.getShape(self.q_gamma)[1];
        if (self.qkv) |qkv| {
            const y = try qkv.forward(sc, x);
            const q = try headNorm(sc, try part(sc, y, 3, 0, heads, d), self.q_gamma, ones);
            const k = try headNorm(sc, try part(sc, y, 3, 1, heads, d), self.k_gamma, ones);
            return self.out.forward(sc, try attend(sc, q, k, try part(sc, y, 3, 2, heads, d), NONE));
        }
        const ctx = context orelse return error.MissingContext;
        const l = mlx.getShape(x)[1];
        const q = try headNorm(sc, try sc.reshape(try self.q.?.forward(sc, x), &.{ 1, l, heads, d }), self.q_gamma, ones);
        const kv = try self.kv.?.forward(sc, ctx);
        const k = try headNorm(sc, try part(sc, kv, 2, 0, heads, d), self.k_gamma, ones);
        return self.out.forward(sc, try attend(sc, q, k, try part(sc, kv, 2, 1, heads, d), NONE));
    }
};

/// `FeedForwardNet` / `MLP`: Linear, tanh GELU, Linear.
pub const Mlp = struct {
    fc1: Linear,
    fc2: Linear,

    pub fn load(w: *const Weights, comptime fmt: []const u8, args: anytype, channels: u32, hidden: u32) !Mlp {
        var buf: [96]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, fmt, args);
        return .{
            .fc1 = try scope_mod.linear(w, "{s}.mlp.0", .{p}, channels),
            .fc2 = try scope_mod.linear(w, "{s}.mlp.2", .{p}, hidden),
        };
    }

    /// The activations fc2 reads.
    pub fn activate(self: *const Mlp, sc: *Scope, x: A) !A {
        const y = try self.fc1.project(sc, x);
        return if (self.fc1.bias) |b|
            if (try glue.biasGelu(y, b, sc.s)) |g| try sc.keep(g) else try geluTanh(sc, try sc.add(y, b))
        else
            try geluTanh(sc, y);
    }

    pub fn forward(self: *const Mlp, sc: *Scope, x: A) !A {
        return self.fc2.forward(sc, try self.activate(sc, x));
    }
};

/// `PcdAbsolutePositionEmbedder` (the denoiser's token positions, in f32): sin and cos of `2 pi p 2^f` for every
/// coordinate, 2^0..2^15 then fractional octaves, zero-padded to `channels`. `pos` is `[n, 3]`.
pub fn absolutePosition(gpa: std.mem.Allocator, pos: []const f32, channels: usize) ![]f32 {
    const n = pos.len / 3;
    const freq_dim = channels / 3 / 2;
    const max_res: usize = 16;
    const freqs = try gpa.alloc(f32, freq_dim);
    defer gpa.free(freqs);
    const res_dim = if (freq_dim > max_res) freq_dim - max_res else 0;
    for (freqs, 0..) |*f, i| {
        const e: f32 = if (i < max_res) @floatFromInt(i) else @as(f32, @floatFromInt(i - max_res)) / @as(f32, @floatFromInt(res_dim)) * @as(f32, max_res);
        f.* = std.math.pow(f32, 2, e);
    }
    const out = try gpa.alloc(f32, n * channels);
    @memset(out, 0);
    for (0..n) |t| for (0..3) |c| for (freqs, 0..) |f, i| {
        const ang: f32 = pos[t * 3 + c] * f * 2 * std.math.pi;
        const base = t * channels + c * 2 * freq_dim;
        out[base + i] = @floatCast(@sin(@as(f64, ang)));
        out[base + freq_dim + i] = @floatCast(@cos(@as(f64, ang)));
    };
    return out;
}

/// `PcdAbsolutePositionEmbedderv2` (the decoders' query points): `2^linspace(0, 10)` times pi, sin and cos, f32.
pub fn pointPosition(gpa: std.mem.Allocator, pos: []const f32, channels: usize) ![]f32 {
    const n = pos.len / 3;
    const freq_dim = channels / 3 / 2;
    const out = try gpa.alloc(f32, n * channels);
    @memset(out, 0);
    const freqs = try gpa.alloc(f32, freq_dim);
    defer gpa.free(freqs);
    for (freqs, 0..) |*f, i| {
        const lg: f32 = 10.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(freq_dim - 1));
        f.* = std.math.pow(f32, 2, lg);
    }
    for (0..n) |t| for (0..3) |c| for (freqs, 0..) |f, i| {
        const ang: f32 = pos[t * 3 + c] * f * std.math.pi;
        const base = t * channels + c * 2 * freq_dim;
        out[base + i] = @floatCast(@sin(@as(f64, ang)));
        out[base + freq_dim + i] = @floatCast(@cos(@as(f64, ang)));
    };
    return out;
}

/// `SinusoidalEmbedder`'s input: cos then sin of `t * exp(-ln(max_period) i / 128)` (times 2 pi when `two_pi`).
pub fn sinusoid(out: *[256]f32, t: f32, max_period: f32, two_pi: bool) void {
    for (0..128) |i| {
        const f: f32 = @exp(-@log(max_period) * @as(f32, @floatFromInt(i)) / 128.0);
        var a: f32 = t * f;
        if (two_pi) a *= 2 * std.math.pi;
        out[i] = @cos(a);
        out[128 + i] = @sin(a);
    }
}

/// Casts one stored f32 tensor to `dt` in place (a Linear's weight or bias), one at a time to hold the peak down.
pub fn castTensor(w: *Weights, key: []const u8, dt: mlx.mlx_dtype) !void {
    const e = w.map.getPtr(key) orelse return;
    if (mlx.mlx_array_dtype(e.*) != .float32) return;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_astype(&o, e.*, dt, s));
    try mlx.check(mlx.mlx_array_eval(o));
    _ = mlx.mlx_array_free(e.*);
    e.* = o;
}

/// Every dense float Linear weight under `prefixes` stored transposed and contiguous as `<name>.weight_t` (`[in, out]`,
/// what MLX's GEMM reads fastest) in place of `<name>.weight`.
pub fn transposeLinears(gpa: std.mem.Allocator, w: *Weights, prefixes: []const []const u8) !void {
    var keys: std.ArrayList([]const u8) = .empty;
    defer keys.deinit(gpa);
    var it = w.map.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        var in_prefix = false;
        for (prefixes) |p| in_prefix = in_prefix or std.mem.startsWith(u8, key, p);
        const dt = mlx.mlx_array_dtype(e.value_ptr.*);
        if (!in_prefix or !std.mem.endsWith(u8, key, ".weight") or mlx.mlx_array_ndim(e.value_ptr.*) != 2) continue;
        if (dt != .float16 and dt != .bfloat16 and dt != .float32) continue;
        try keys.append(gpa, key);
    }
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    for (keys.items) |key| {
        var buf: [160]u8 = undefined;
        const base = key[0 .. key.len - ".weight".len];
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, w.get(key).?, &[_]c_int{ 1, 0 }, 2, s));
        var c = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, t, false, s));
        try mlx.check(mlx.mlx_array_eval(c));
        try w.put(try std.fmt.bufPrint(&buf, "{s}.weight_t", .{base}), c);
        w.remove(key);
    }
}

/// Casts every f32 Linear (`*.weight` matrix and its `.bias`) under one of `prefixes` to `dt`, as the reference's
/// `convert_module_to_f16` does; norm weights (1-D), gammas and shift tables stay f32.
pub fn castLinears(gpa: std.mem.Allocator, w: *Weights, prefixes: []const []const u8, dt: mlx.mlx_dtype) !void {
    var keys: std.ArrayList([]const u8) = .empty;
    defer keys.deinit(gpa);
    var it = w.map.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        var in_prefix = false;
        for (prefixes) |p| in_prefix = in_prefix or std.mem.startsWith(u8, key, p);
        if (!in_prefix) continue;
        if (std.mem.endsWith(u8, key, ".weight")) {
            if (mlx.mlx_array_ndim(e.value_ptr.*) == 2) try keys.append(gpa, key);
        } else if (std.mem.endsWith(u8, key, ".bias")) {
            var buf: [160]u8 = undefined;
            const partner = std.fmt.bufPrint(&buf, "{s}.weight", .{key[0 .. key.len - 5]}) catch continue;
            const pw = w.get(partner) orelse continue;
            if (mlx.mlx_array_ndim(pw) == 2) try keys.append(gpa, key);
        }
    }
    for (keys.items) |k| try castTensor(w, k, dt);
}

// ── tests ──

const testing = std.testing;

test "idol: a block Linear stored transposed projects as the original" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var w = Weights.init(a);
    defer w.deinit();
    var sc = Scope.init(a, s);
    defer sc.deinit();
    var vals: [6 * 8 + 6 + 3 * 8]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 7 % 13)) - 6)) / 8;
    try w.put("blocks.0.fc.weight", sc.out(try sc.astype(try sc.fromF32(vals[0..48], &.{ 6, 8 }), .float16)));
    try w.put("blocks.0.fc.bias", sc.out(try sc.astype(try sc.fromF32(vals[48..54], &.{6}), .float16)));
    const x = try sc.astype(try sc.fromF32(vals[54..], &.{ 3, 8 }), .float16);
    const want = try (try scope_mod.linear(&w, "blocks.0.fc", .{}, 8)).forward(&sc, x);
    try mlx.check(mlx.mlx_array_eval(want));
    try transposeLinears(a, &w, &.{"blocks."});
    try testing.expect(w.get("blocks.0.fc.weight") == null);
    try testing.expectEqualSlices(c_int, &.{ 8, 6 }, mlx.getShape(w.get("blocks.0.fc.weight_t").?));
    const got = try (try scope_mod.linear(&w, "blocks.0.fc", .{}, 8)).forward(&sc, x);
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, got, want, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try testing.expect(same);
}
