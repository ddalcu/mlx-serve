//! Shared low-level building blocks for the native FLUX image backends
//! (`flux1.zig`, `t5.zig`, and future flux modules). These mirror the
//! file-private helpers in `flux.zig` (FLUX.2 klein); they are lifted here so
//! the FLUX.1 pipeline can reuse them without duplicating or perturbing the
//! working FLUX.2 code. See docs/reference.md (FLUX.1 section).
//!
//! Quantized weights are affine (u32 packed weight + bf16 scales + bf16
//! biases, group_size inferred per weight); dequant-free via
//! `mlx_quantized_matmul`.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const lora_mod = @import("lora.zig");

const Weights = model_mod.Weights;
pub const S = mlx.mlx_stream;

// ── Low-level mlx wrappers ──

pub inline fn matmul(x: mlx.mlx_array, w_t: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, x, w_t, s));
    return o;
}
pub inline fn addA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}
pub inline fn mulA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}
pub inline fn subA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, a, b, s));
    return o;
}
pub inline fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
pub inline fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}
pub inline fn rms(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}
pub inline fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
pub inline fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    return mulA(x, sig, s);
}
/// Exact GELU: 0.5·x·(1+erf(x/√2)) — `nn.gelu`.
pub fn geluErf(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const inv_sqrt2 = mlx.mlx_array_new_float(0.7071067811865476);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    const scaled = try mulA(x, inv_sqrt2, s);
    defer _ = mlx.mlx_array_free(scaled);
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_erf(&e, scaled, s));
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const onep = try addA(e, one, s);
    defer _ = mlx.mlx_array_free(onep);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, onep, s);
}
/// Tanh-approx GELU (`gelu_new`): 0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³))).
/// This is the activation for T5 v1.1 (google/t5-v1_1-xxl) gated FF.
pub fn geluTanh(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c0 = mlx.mlx_array_new_float(0.7978845608028654); // √(2/π)
    defer _ = mlx.mlx_array_free(c0);
    const c1 = mlx.mlx_array_new_float(0.044715);
    defer _ = mlx.mlx_array_free(c1);
    // x³
    const x2 = try mulA(x, x, s);
    defer _ = mlx.mlx_array_free(x2);
    const x3 = try mulA(x2, x, s);
    defer _ = mlx.mlx_array_free(x3);
    const cx3 = try mulA(x3, c1, s);
    defer _ = mlx.mlx_array_free(cx3);
    const inner = try addA(x, cx3, s);
    defer _ = mlx.mlx_array_free(inner);
    const scaled = try mulA(inner, c0, s);
    defer _ = mlx.mlx_array_free(scaled);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_tanh(&t, scaled, s));
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const onep = try addA(t, one, s);
    defer _ = mlx.mlx_array_free(onep);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, onep, s);
}
pub fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
/// Slice along `axis` of a 3-D [d0,d1,d2] array: [start,stop) on that axis.
pub fn slice3(x: mlx.mlx_array, axis: usize, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var lo = [_]c_int{ 0, 0, 0 };
    var hi = [_]c_int{ sh[0], sh[1], sh[2] };
    const st = [_]c_int{ 1, 1, 1 };
    lo[axis] = start;
    hi[axis] = stop;
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, 3, &hi, 3, &st, 3, s));
    return o;
}

// ── Weight fetch / shape helpers ──

pub fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[flux] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingFluxWeight;
    };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, a));
    return o;
}
pub fn ownOpt(w: *const Weights, key: []const u8) ?mlx.mlx_array {
    const a = w.get(key) orelse return null;
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_array_set(&o, a)) catch return null;
    return o;
}
pub fn fmtKey(a: std.mem.Allocator, comptime f: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, f, args);
}
/// Rows (axis-0 extent) of a weight, 0 if absent.
pub fn rowsOf(w: *const Weights, key: []const u8) u32 {
    const a = w.get(key) orelse return 0;
    const sh = mlx.getShape(a);
    if (sh.len == 0 or sh[0] < 0) return 0;
    return @intCast(sh[0]);
}
/// Count consecutive indexed weights `fmt.{0..}` until the first gap.
pub fn countIndexed(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8) u32 {
    const max_blocks = 256;
    var n: u32 = 0;
    while (n < max_blocks) : (n += 1) {
        const key = fmtKey(a, fmt, .{n}) catch return n;
        defer a.free(key);
        if (w.get(key) == null) return n;
    }
    return n;
}

// ── Quantized Linear (affine; bits/group inferred per weight) ──

pub const QuantGeometry = struct { bits: u32, group_size: u32 };

fn inferQuantGeometry(w_cols: usize, s_cols: usize) QuantGeometry {
    const fallback: QuantGeometry = .{ .bits = 4, .group_size = 64 };
    if (s_cols == 0 or (w_cols * 32) % s_cols != 0) return fallback;
    const product = w_cols * 32 / s_cols; // bits · group_size
    const valid_bits = [_]u32{ 2, 3, 4, 5, 6, 8 };
    for ([_]u32{ 64, 32, 128 }) |gs| {
        if (product % gs != 0) continue;
        const bits: u32 = @intCast(product / gs);
        for (valid_bits) |vb| {
            if (bits == vb) return .{ .bits = bits, .group_size = gs };
        }
    }
    return fallback;
}
pub fn inferQuantGeometryOf(w: mlx.mlx_array, scales: mlx.mlx_array) QuantGeometry {
    const wsh = mlx.getShape(w);
    const ssh = mlx.getShape(scales);
    if (wsh.len == 0 or ssh.len == 0) return .{ .bits = 4, .group_size = 64 };
    const w_cols: usize = @intCast(wsh[wsh.len - 1]);
    const s_cols: usize = @intCast(ssh[ssh.len - 1]);
    return inferQuantGeometry(w_cols, s_cols);
}

pub const QLinear = struct {
    w: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array, // quant zero-points (NOT additive bias)
    add_bias: ?mlx.mlx_array = null, // optional additive bias
    bits: u32 = 4,
    group_size: u32 = 64,
    lora_refs: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined,
    lora_count: u8 = 0,

    pub fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8) !QLinear {
        const wk = try fmtKey(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const sk = try fmtKey(a, "{s}.scales", .{prefix});
        defer a.free(sk);
        const bk = try fmtKey(a, "{s}.biases", .{prefix});
        defer a.free(bk);
        const ak = try fmtKey(a, "{s}.bias", .{prefix});
        defer a.free(ak);
        var ql: QLinear = .{
            .w = try ownWeight(w, wk),
            .scales = try ownWeight(w, sk),
            .biases = try ownWeight(w, bk),
            .add_bias = ownOpt(w, ak),
        };
        const geo = inferQuantGeometryOf(ql.w, ql.scales);
        ql.bits = geo.bits;
        ql.group_size = geo.group_size;
        return ql;
    }
    pub fn deinit(self: *QLinear) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.scales);
        _ = mlx.mlx_array_free(self.biases);
        if (self.add_bias) |b| _ = mlx.mlx_array_free(b);
    }
    pub fn forward(self: *const QLinear, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(&o, x, self.w, self.scales, self.biases, true, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", s));
        if (self.add_bias) |b| {
            const r = try addA(o, b, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        if (self.lora_count > 0) {
            const d = try lora_mod.deltaSum(x, self.lora_refs[0..self.lora_count], s);
            defer _ = mlx.mlx_array_free(d);
            const r = try addA(o, d, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        return o;
    }
    pub fn setLoraRefs(self: *QLinear, refs: []const lora_mod.Ref) void {
        self.lora_count = @intCast(refs.len);
        @memcpy(self.lora_refs[0..refs.len], refs);
    }
    pub fn clearLoraRefs(self: *QLinear) void {
        self.lora_count = 0;
    }
};

/// Dequantize a quantized embedding/weight table → bf16 [rows, cols].
pub fn dequantTable(w_q: mlx.mlx_array, scales: mlx.mlx_array, biases: mlx.mlx_array, bits: u32, gs: u32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    const null_gs = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_dequantize(&o, w_q, scales, biases, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", null_gs, .{ .value = .bfloat16, .has_value = true }, s));
    return o;
}
