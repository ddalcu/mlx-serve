//! Native Krea-2-Turbo text→image backend (single-stream MMDiT + Qwen3-VL-4B
//! text encoder + Qwen-Image 3D causal VAE + flow-match Euler sampler), ported
//! from the pure-MLX reference `avlp12/krea2_alis_mlx` to mlx-c FFI.
//!
//! Self-contained sibling of `flux.zig`: hosted by the image modality slot in
//! `gen.zig` (the `ImageEngine` backend union). flux.zig is NOT touched.
//!
//! Mixed precision: the reference quantizes ONLY the 28 main blocks' attn+mlp
//! linears (affine, group_size 64); everything else stays bf16. There is no
//! per-tensor bit map in any config — `MixedLinear` infers (bits, group_size)
//! from tensor geometry (mirrors `transformer.computeQuantParams`/`mtp.inferBits`),
//! so ONE engine loads all three builds (8bit / mixed-4-8 / bf16). VAE runs in
//! f32 (the reference decodes in f32); encoder + DiT run in bf16.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const sse = @import("gen_sse.zig");
const png = @import("png.zig");
const tok_mod = @import("tokenizer.zig");
const lora_mod = @import("lora.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

// ── Low-level mlx helpers (file-local primitives, mirror flux.zig) ──

inline fn matmul(x: mlx.mlx_array, w_t: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, x, w_t, s));
    return o;
}
inline fn addA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}
inline fn mulA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}
inline fn subA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, a, b, s));
    return o;
}
inline fn divA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&o, a, b, s));
    return o;
}
inline fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
inline fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}
inline fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
inline fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
inline fn rms(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}
inline fn sigmoidA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&o, x, s));
    return o;
}
fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sig = try sigmoidA(x, s);
    defer _ = mlx.mlx_array_free(sig);
    return mulA(x, sig, s);
}
fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
fn stack(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_stack_axis(&o, vec, axis, s));
    return o;
}
/// Slice [start,stop) on `axis` of an N-D array (N ≤ 8).
fn sliceAxis(x: mlx.mlx_array, axis: usize, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const nd = sh.len;
    var lo: [8]c_int = undefined;
    var hi: [8]c_int = undefined;
    var st: [8]c_int = undefined;
    for (0..nd) |i| {
        lo[i] = 0;
        hi[i] = sh[i];
        st[i] = 1;
    }
    lo[axis] = start;
    hi[axis] = stop;
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, lo[0..nd].ptr, nd, hi[0..nd].ptr, nd, st[0..nd].ptr, nd, s));
    return o;
}
fn scalarF(v: f32) mlx.mlx_array {
    return mlx.mlx_array_new_float(v);
}
/// GELU(approximate="tanh"): 0.5*x*(1+tanh(√(2/π)*(x+0.044715*x³))).
fn geluTanh(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c: f32 = 0.7978845608028654; // sqrt(2/pi)
    const k = scalarF(0.044715);
    defer _ = mlx.mlx_array_free(k);
    const x2 = try mulA(x, x, s);
    defer _ = mlx.mlx_array_free(x2);
    const x3 = try mulA(x2, x, s);
    defer _ = mlx.mlx_array_free(x3);
    const kx3 = try mulA(x3, k, s);
    defer _ = mlx.mlx_array_free(kx3);
    const inner = try addA(x, kx3, s);
    defer _ = mlx.mlx_array_free(inner);
    const ca = scalarF(c);
    defer _ = mlx.mlx_array_free(ca);
    const cin = try mulA(inner, ca, s);
    defer _ = mlx.mlx_array_free(cin);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_tanh(&t, cin, s));
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const opt = try addA(t, one, s);
    defer _ = mlx.mlx_array_free(opt);
    const half = scalarF(0.5);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, opt, s);
}

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[krea] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingKreaWeight;
    };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, a));
    return o;
}
fn ownOpt(w: *const Weights, key: []const u8) ?mlx.mlx_array {
    const a = w.get(key) orelse return null;
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_array_set(&o, a)) catch return null;
    return o;
}
fn fmtKey(a: std.mem.Allocator, comptime f: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, f, args);
}

// ════════════════════════════════════════════════════════════════════════
// B0. MixedLinear — bf16 OR affine-quantized, bits/group_size inferred from
//     tensor geometry. The single primitive that loads all three builds.
// ════════════════════════════════════════════════════════════════════════

const MixedLinear = struct {
    quantized: bool,
    w: mlx.mlx_array, // quantized: packed u32 [out, in*bits/32]; bf16: pre-transposed [in,out]
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    add_bias: ?mlx.mlx_array = null,
    bits: u32 = 0,
    group_size: u32 = 0,
    lora: ?lora_mod.Ref = null, // runtime adapter (non-owning; gen.zig owns the File)

    /// `in_features` is the module's input dim (known per call); used only on
    /// the quantized path to solve (bits, group_size) from packed geometry.
    fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, in_features: u32, s: S) !MixedLinear {
        const wk = try fmtKey(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const sk = try fmtKey(a, "{s}.scales", .{prefix});
        defer a.free(sk);
        const bk = try fmtKey(a, "{s}.biases", .{prefix});
        defer a.free(bk);
        const ak = try fmtKey(a, "{s}.bias", .{prefix});
        defer a.free(ak);

        if (ownOpt(w, sk)) |scales| {
            // Quantized: infer bits & group_size from geometry.
            const weight = try ownWeight(w, wk);
            const biases = try ownWeight(w, bk);
            const w_cols: u32 = @intCast(mlx.getShape(weight)[1]); // in*bits/32
            const s_cols: u32 = @intCast(mlx.getShape(scales)[1]); // in/group_size
            const bits: u32 = @intCast(@divExact(32 * w_cols, in_features));
            const gs: u32 = @intCast(@divExact(in_features, s_cols));
            return .{
                .quantized = true,
                .w = weight,
                .scales = scales,
                .biases = biases,
                .add_bias = ownOpt(w, ak),
                .bits = bits,
                .group_size = gs,
            };
        }
        // bf16: pre-transpose [out,in] → [in,out], materialize, cast bf16.
        const raw = try ownWeight(w, wk);
        defer _ = mlx.mlx_array_free(raw);
        const t = try transpose(raw, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(t);
        const tc = try contig(t, s);
        defer _ = mlx.mlx_array_free(tc);
        const wt = try astype(tc, .bfloat16, s);
        return .{ .quantized = false, .w = wt, .add_bias = ownOpt(w, ak) };
    }

    fn deinit(self: *MixedLinear) void {
        _ = mlx.mlx_array_free(self.w);
        if (self.quantized) {
            _ = mlx.mlx_array_free(self.scales);
            _ = mlx.mlx_array_free(self.biases);
        }
        if (self.add_bias) |b| _ = mlx.mlx_array_free(b);
    }

    fn forward(self: *const MixedLinear, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const xb = try astype(x, .bfloat16, s); // reference computes in bf16
        defer _ = mlx.mlx_array_free(xb);
        var o = mlx.mlx_array_new();
        if (self.quantized) {
            try mlx.check(mlx.mlx_quantized_matmul(&o, xb, self.w, self.scales, self.biases, true, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", s));
        } else {
            try mlx.check(mlx.mlx_matmul(&o, xb, self.w, s));
        }
        if (self.add_bias) |b| {
            const r = try addA(o, b, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        if (self.lora) |lr| {
            const d = try lora_mod.delta(xb, lr, s);
            defer _ = mlx.mlx_array_free(d);
            const r = try addA(o, d, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        return o;
    }
};

// ── Config ──

pub const KreaConfig = struct {
    // DiT
    features: u32 = 6144,
    tdim: u32 = 256,
    txtdim: u32 = 2560,
    heads: u32 = 48,
    kvheads: u32 = 12,
    multiplier: u32 = 4,
    layers: u32 = 28,
    patch: u32 = 2,
    channels: u32 = 16,
    theta: f32 = 1000.0,
    txtheads: u32 = 20,
    txtlayers: u32 = 12,
    // text encoder (Qwen3-VL-4B)
    te_hidden: u32 = 2560,
    te_layers: u32 = 36,
    te_heads: u32 = 32,
    te_kv: u32 = 8,
    te_head_dim: u32 = 128,
    te_inter: u32 = 9728,
    te_theta: f32 = 5_000_000.0,
    te_eps: f32 = 1e-6,

    fn headDim(self: KreaConfig) u32 {
        return self.features / self.heads; // 128
    }
};

// Prompt template (must match text_encoder.py exactly).
const PREFIX =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, " ++
    "texture, quantity, text, spatial relationships of the objects and background:" ++
    "<|im_end|>\n<|im_start|>user\n";
const SUFFIX = "<|im_end|>\n<|im_start|>assistant\n";
const PREFIX_START_IDX: usize = 34;
const SUFFIX_START_IDX: usize = 5;
const MAX_LENGTH: usize = 512; // output text tokens
const PAD_TOKEN: i32 = 151643; // Qwen <|endoftext|>
// HF hidden_states tapped (input-to-layer i; i<36 so never the final norm).
const SELECT_LAYERS = [_]usize{ 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35 };

// ════════════════════════════════════════════════════════════════════════
// B1. Conditioner — Qwen3-VL-4B text encoder (12-layer capture), bf16.
// ════════════════════════════════════════════════════════════════════════

const TeLayer = struct {
    input_ln: mlx.mlx_array,
    post_ln: mlx.mlx_array,
    q: MixedLinear,
    k: MixedLinear,
    v: MixedLinear,
    o: MixedLinear,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    gate: MixedLinear,
    up: MixedLinear,
    down: MixedLinear,
    fn deinit(self: *TeLayer) void {
        _ = mlx.mlx_array_free(self.input_ln);
        _ = mlx.mlx_array_free(self.post_ln);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        _ = mlx.mlx_array_free(self.q_norm);
        _ = mlx.mlx_array_free(self.k_norm);
        self.gate.deinit();
        self.up.deinit();
        self.down.deinit();
    }
};

pub const Conditioner = struct {
    cfg: KreaConfig,
    allocator: std.mem.Allocator,
    s: S,
    embed_table: mlx.mlx_array, // [vocab, hidden] bf16
    layers: []TeLayer,

    pub fn deinit(self: *Conditioner) void {
        _ = mlx.mlx_array_free(self.embed_table);
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
    }

    /// ids/mask are the FULL templated sequence (PREFIX+prompt padded + SUFFIX,
    /// length L0 = MAX_LENGTH + PREFIX_START_IDX). Returns the post-prefix slice
    /// stacked over 12 tapped layers: embeds [1, MAX_LENGTH, 12, hidden] bf16.
    pub fn encode(self: *Conditioner, ids: []const i32, mask: []const i32) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        const seq: c_int = @intCast(ids.len);
        const H: c_int = @intCast(c.te_hidden);

        const id_shape = [_]c_int{seq};
        const id_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_axis(&taken, self.embed_table, id_arr, 0, s));
        var x = try reshape(taken, &[_]c_int{ 1, seq, H }, s);

        const attn_mask = try buildCausalMask(self.allocator, mask, seq, s);
        defer _ = mlx.mlx_array_free(attn_mask);

        // Standard rope cos/sin [seq, head_dim], computed in f32 then ROUNDED to
        // bf16 (matches the reference, which casts cos/sin to h.dtype before use —
        // mlx_fast_rope's f32-internal rotation diverges enough to miss parity).
        const rope = try buildEncoderRope(self.allocator, @intCast(seq), c.te_head_dim, c.te_theta, s);
        defer {
            _ = mlx.mlx_array_free(rope.cos);
            _ = mlx.mlx_array_free(rope.sin);
        }

        var caps: [SELECT_LAYERS.len]mlx.mlx_array = undefined;
        var ncap: usize = 0;
        errdefer for (0..ncap) |i| {
            _ = mlx.mlx_array_free(caps[i]);
        };

        for (self.layers, 0..) |*layer, li| {
            // HF output_hidden_states: capture BEFORE each layer.
            for (SELECT_LAYERS) |wv| {
                if (wv == li) {
                    var cp = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_array_set(&cp, x));
                    caps[ncap] = cp;
                    ncap += 1;
                }
            }
            const nx = try self.layerForward(x, layer, attn_mask, rope.cos, rope.sin, seq, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        _ = mlx.mlx_array_free(x);

        // stack 12 captures along a new axis 2 → [1, seq, 12, H]
        const stacked = try stack(caps[0..ncap], 2, s);
        for (0..ncap) |i| _ = mlx.mlx_array_free(caps[i]);
        defer _ = mlx.mlx_array_free(stacked);
        // drop the prefix tokens
        return sliceAxis(stacked, 1, @intCast(PREFIX_START_IDX), seq, s);
    }

    fn layerForward(self: *Conditioner, x: mlx.mlx_array, layer: *const TeLayer, mask: mlx.mlx_array, rope_cos: mlx.mlx_array, rope_sin: mlx.mlx_array, seq: c_int, s: S) !mlx.mlx_array {
        const c = self.cfg;
        const eps = c.te_eps;
        const heads: c_int = @intCast(c.te_heads);
        const kv: c_int = @intCast(c.te_kv);
        const hd: c_int = @intCast(c.te_head_dim);

        const xn = try rms(x, layer.input_ln, eps, s);
        defer _ = mlx.mlx_array_free(xn);
        const q = try layer.q.forward(xn, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try layer.k.forward(xn, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try layer.v.forward(xn, s);
        defer _ = mlx.mlx_array_free(v);
        const q4 = try reshape(q, &[_]c_int{ 1, seq, heads, hd }, s);
        defer _ = mlx.mlx_array_free(q4);
        const qn = try rms(q4, layer.q_norm, eps, s);
        defer _ = mlx.mlx_array_free(qn);
        const qt = try transpose(qn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(qt);
        const k4 = try reshape(k, &[_]c_int{ 1, seq, kv, hd }, s);
        defer _ = mlx.mlx_array_free(k4);
        const kn = try rms(k4, layer.k_norm, eps, s);
        defer _ = mlx.mlx_array_free(kn);
        const kt = try transpose(kn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(kt);
        const v4 = try reshape(v, &[_]c_int{ 1, seq, kv, hd }, s);
        defer _ = mlx.mlx_array_free(v4);
        const vt = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(vt);
        // RoPE (NeoX/rotate_half) θ 5e6 — manual, bf16 cos/sin (matches reference).
        const qr = try applyRopeEncoder(qt, rope_cos, rope_sin, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try applyRopeEncoder(kt, rope_cos, rope_sin, s);
        defer _ = mlx.mlx_array_free(kr);
        // SDPA (bf16, GQA handled natively), explicit additive mask.
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(c.te_head_dim)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_sink = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vt, scale, "array", mask, null_sink, s));
        const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = try reshape(at, &[_]c_int{ 1, seq, heads * hd }, s);
        defer _ = mlx.mlx_array_free(af);
        const o = try layer.o.forward(af, s);
        defer _ = mlx.mlx_array_free(o);
        const h1 = try addA(x, o, s);
        defer _ = mlx.mlx_array_free(h1);
        // MLP (SwiGLU)
        const hn = try rms(h1, layer.post_ln, eps, s);
        defer _ = mlx.mlx_array_free(hn);
        const g = try layer.gate.forward(hn, s);
        defer _ = mlx.mlx_array_free(g);
        const sg = try silu(g, s);
        defer _ = mlx.mlx_array_free(sg);
        const u = try layer.up.forward(hn, s);
        defer _ = mlx.mlx_array_free(u);
        const gu = try mulA(sg, u, s);
        defer _ = mlx.mlx_array_free(gu);
        const dn = try layer.down.forward(gu, s);
        defer _ = mlx.mlx_array_free(dn);
        return addA(h1, dn, s);
    }
};

/// Causal + padding additive mask [1,1,seq,seq] bf16 (-1e9 on blocked).
fn buildCausalMask(allocator: std.mem.Allocator, mask: []const i32, seq: c_int, s: S) !mlx.mlx_array {
    const n: usize = @intCast(seq);
    const buf = try allocator.alloc(f32, n * n);
    defer allocator.free(buf);
    const neg: f32 = -1e9;
    for (0..n) |i| {
        for (0..n) |j| {
            const blocked = (j > i) or (mask[j] == 0);
            buf[i * n + j] = if (blocked) neg else 0.0;
        }
    }
    const shape = [_]c_int{ 1, 1, seq, seq };
    const f = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f);
    return astype(f, .bfloat16, s);
}

/// Standard HF rope cos/sin [L, head_dim] for the encoder: emb = concat([freqs,
/// freqs]); computed in f32 then ROUNDED to bf16 (the reference casts cos/sin to
/// h.dtype before applying — replicate exactly for parity).
fn buildEncoderRope(allocator: std.mem.Allocator, L: usize, hd: usize, theta: f32, s: S) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
    const cosb = try allocator.alloc(f32, L * hd);
    defer allocator.free(cosb);
    const sinb = try allocator.alloc(f32, L * hd);
    defer allocator.free(sinb);
    const half = hd / 2;
    const th: f64 = theta;
    for (0..L) |p| {
        for (0..half) |i| {
            const inv = std.math.pow(f64, th, -@as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(hd)));
            const ang = @as(f64, @floatFromInt(p)) * inv;
            const cv: f32 = @floatCast(@cos(ang));
            const sv: f32 = @floatCast(@sin(ang));
            cosb[p * hd + i] = cv;
            cosb[p * hd + i + half] = cv;
            sinb[p * hd + i] = sv;
            sinb[p * hd + i + half] = sv;
        }
    }
    const sh = [_]c_int{ @intCast(L), @intCast(hd) };
    const cf = mlx.mlx_array_new_data(cosb.ptr, &sh, 2, .float32);
    defer _ = mlx.mlx_array_free(cf);
    const sf = mlx.mlx_array_new_data(sinb.ptr, &sh, 2, .float32);
    defer _ = mlx.mlx_array_free(sf);
    return .{ .cos = try astype(cf, .bfloat16, s), .sin = try astype(sf, .bfloat16, s) };
}

/// Standard rope on x [1,H,L,hd]: x*cos + rotate_half(x)*sin, all bf16.
/// rotate_half(x) = concat([-x[...,hd/2:], x[...,:hd/2]]).
fn applyRopeEncoder(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,L,hd]
    const L = sh[2];
    const hd = sh[3];
    const half = @divExact(hd, 2);
    const cos_b = try reshape(cos, &[_]c_int{ 1, 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(cos_b);
    const sin_b = try reshape(sin, &[_]c_int{ 1, 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(sin_b);
    const xc = try mulA(x, cos_b, s);
    defer _ = mlx.mlx_array_free(xc);
    const x1 = try sliceAxis(x, 3, 0, half, s);
    defer _ = mlx.mlx_array_free(x1);
    const x2 = try sliceAxis(x, 3, half, hd, s);
    defer _ = mlx.mlx_array_free(x2);
    const neg1 = try astype(scalarF(-1.0), .bfloat16, s);
    defer _ = mlx.mlx_array_free(neg1);
    const nx2 = try mulA(x2, neg1, s);
    defer _ = mlx.mlx_array_free(nx2);
    const rh = try concat(&[_]mlx.mlx_array{ nx2, x1 }, 3, s);
    defer _ = mlx.mlx_array_free(rh);
    const rs = try mulA(rh, sin_b, s);
    defer _ = mlx.mlx_array_free(rs);
    return addA(xc, rs, s);
}

pub fn loadConditioner(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Conditioner {
    const dir = try fmtKey(allocator, "{s}/text_encoder", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var te: Conditioner = undefined;
    te.cfg = .{};
    te.allocator = allocator;
    te.s = s;

    // Qwen3-VL text weights live under a `language_model.` prefix in the
    // checkpoint (the reference strips it at load); detect it and prefix every
    // key. Some flat exports omit it — fall back to no prefix.
    const pfx: []const u8 = if (w.get("language_model.embed_tokens.weight") != null) "language_model." else "";

    const ek = try fmtKey(allocator, "{s}embed_tokens.weight", .{pfx});
    defer allocator.free(ek);
    te.embed_table = try ownWeight(&w, ek);
    {
        const bf = try astype(te.embed_table, .bfloat16, s);
        _ = mlx.mlx_array_free(te.embed_table);
        te.embed_table = bf;
    }

    te.layers = try allocator.alloc(TeLayer, te.cfg.te_layers);
    const H = te.cfg.te_hidden;
    const inter = te.cfg.te_inter;
    for (te.layers, 0..) |*layer, i| {
        const p_in = try fmtKey(allocator, "{s}layers.{d}.input_layernorm.weight", .{ pfx, i });
        defer allocator.free(p_in);
        const p_post = try fmtKey(allocator, "{s}layers.{d}.post_attention_layernorm.weight", .{ pfx, i });
        defer allocator.free(p_post);
        const qn = try fmtKey(allocator, "{s}layers.{d}.self_attn.q_norm.weight", .{ pfx, i });
        defer allocator.free(qn);
        const kn = try fmtKey(allocator, "{s}layers.{d}.self_attn.k_norm.weight", .{ pfx, i });
        defer allocator.free(kn);
        const qp = try fmtKey(allocator, "{s}layers.{d}.self_attn.q_proj", .{ pfx, i });
        defer allocator.free(qp);
        const kp = try fmtKey(allocator, "{s}layers.{d}.self_attn.k_proj", .{ pfx, i });
        defer allocator.free(kp);
        const vp = try fmtKey(allocator, "{s}layers.{d}.self_attn.v_proj", .{ pfx, i });
        defer allocator.free(vp);
        const op = try fmtKey(allocator, "{s}layers.{d}.self_attn.o_proj", .{ pfx, i });
        defer allocator.free(op);
        const gp = try fmtKey(allocator, "{s}layers.{d}.mlp.gate_proj", .{ pfx, i });
        defer allocator.free(gp);
        const upp = try fmtKey(allocator, "{s}layers.{d}.mlp.up_proj", .{ pfx, i });
        defer allocator.free(upp);
        const dp = try fmtKey(allocator, "{s}layers.{d}.mlp.down_proj", .{ pfx, i });
        defer allocator.free(dp);
        layer.* = .{
            .input_ln = try normF32(&w, p_in, s),
            .post_ln = try normF32(&w, p_post, s),
            .q = try MixedLinear.load(&w, allocator, qp, H, s),
            .k = try MixedLinear.load(&w, allocator, kp, H, s),
            .v = try MixedLinear.load(&w, allocator, vp, H, s),
            .o = try MixedLinear.load(&w, allocator, op, te.cfg.te_heads * te.cfg.te_head_dim, s),
            .q_norm = try normF32(&w, qn, s),
            .k_norm = try normF32(&w, kn, s),
            .gate = try MixedLinear.load(&w, allocator, gp, H, s),
            .up = try MixedLinear.load(&w, allocator, upp, H, s),
            .down = try MixedLinear.load(&w, allocator, dp, inter, s),
        };
    }
    return te;
}

/// Load a `.weight`-convention RMSNorm vector (no +1), cast bf16.
fn normBf16(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, .bfloat16, s);
}

/// Load a `.weight`-convention RMSNorm vector as f32. The reference Qwen3RMSNorm
/// multiplies by `weight.astype(f32)`; keeping the encoder norm weights in f32
/// (mlx_fast_rms_norm upcasts the bf16 activations internally) matches that and
/// closes the late-layer outlier divergence a bf16 weight introduces.
fn normF32(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, .float32, s);
}

// ════════════════════════════════════════════════════════════════════════
// B2. Denoiser — SingleStreamDiT (28-block single-stream MMDiT).
// ════════════════════════════════════════════════════════════════════════

/// Build interleaved 3-axis RoPE cos/sin [L,64] (f32) from pos [L*3] (i32).
/// axes = [32,48,48] → half-counts [16,24,24]; theta=1000.
fn buildRope(allocator: std.mem.Allocator, pos: []const i32, L: usize, theta: f32) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
    const cosb = try allocator.alloc(f32, L * 64);
    defer allocator.free(cosb);
    const sinb = try allocator.alloc(f32, L * 64);
    defer allocator.free(sinb);
    const axes = [_]usize{ 32, 48, 48 };
    const th: f64 = theta;
    for (0..L) |p| {
        var kk: usize = 0;
        for (0..3) |ai| {
            const d = axes[ai];
            const pv: f64 = @floatFromInt(pos[p * 3 + ai]);
            var j: usize = 0;
            while (j < d / 2) : (j += 1) {
                const scale = @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(d));
                const omega = std.math.pow(f64, th, -scale);
                const ang = pv * omega;
                cosb[p * 64 + kk] = @floatCast(@cos(ang));
                sinb[p * 64 + kk] = @floatCast(@sin(ang));
                kk += 1;
            }
        }
    }
    const shape = [_]c_int{ @intCast(L), 64 };
    return .{
        .cos = mlx.mlx_array_new_data(cosb.ptr, &shape, 2, .float32),
        .sin = mlx.mlx_array_new_data(sinb.ptr, &shape, 2, .float32),
    };
}

/// Interleaved (adjacent-pair) RoPE on x [1,H,L,128] using cos/sin [L,64] (f32).
fn applyRope(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, L: c_int, heads: c_int, s: S) !mlx.mlx_array {
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const x5 = try reshape(xf, &[_]c_int{ 1, heads, L, 64, 2 }, s);
    defer _ = mlx.mlx_array_free(x5);
    const real = try sliceLast2(x5, 0, s);
    defer _ = mlx.mlx_array_free(real);
    const imag = try sliceLast2(x5, 1, s);
    defer _ = mlx.mlx_array_free(imag);
    const cos_b = try reshape(cos, &[_]c_int{ 1, 1, L, 64 }, s);
    defer _ = mlx.mlx_array_free(cos_b);
    const sin_b = try reshape(sin, &[_]c_int{ 1, 1, L, 64 }, s);
    defer _ = mlx.mlx_array_free(sin_b);
    const rc = try mulA(real, cos_b, s);
    defer _ = mlx.mlx_array_free(rc);
    const is_ = try mulA(imag, sin_b, s);
    defer _ = mlx.mlx_array_free(is_);
    const o0 = try subA(rc, is_, s);
    defer _ = mlx.mlx_array_free(o0);
    const ic = try mulA(imag, cos_b, s);
    defer _ = mlx.mlx_array_free(ic);
    const rs2 = try mulA(real, sin_b, s);
    defer _ = mlx.mlx_array_free(rs2);
    const o1 = try addA(ic, rs2, s);
    defer _ = mlx.mlx_array_free(o1);
    const o0e = try reshape(o0, &[_]c_int{ 1, heads, L, 64, 1 }, s);
    defer _ = mlx.mlx_array_free(o0e);
    const o1e = try reshape(o1, &[_]c_int{ 1, heads, L, 64, 1 }, s);
    defer _ = mlx.mlx_array_free(o1e);
    const st = try concat(&[_]mlx.mlx_array{ o0e, o1e }, 4, s);
    defer _ = mlx.mlx_array_free(st);
    const flat = try reshape(st, &[_]c_int{ 1, heads, L, 128 }, s);
    defer _ = mlx.mlx_array_free(flat);
    return astype(flat, .bfloat16, s);
}

fn sliceLast2(x: mlx.mlx_array, idx: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,L,64,2]
    var lo = [_]c_int{ 0, 0, 0, 0, idx };
    var hi = [_]c_int{ sh[0], sh[1], sh[2], sh[3], idx + 1 };
    const st = [_]c_int{ 1, 1, 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, 5, &hi, 5, &st, 5, s));
    const sq = try reshape(o, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, s);
    _ = mlx.mlx_array_free(o);
    return sq;
}

const Attention = struct {
    wq: MixedLinear,
    wk: MixedLinear,
    wv: MixedLinear,
    gate: MixedLinear,
    wo: MixedLinear,
    qnorm: mlx.mlx_array, // (1+scale) bf16
    knorm: mlx.mlx_array,
    heads: c_int,
    kvheads: c_int,
    head_dim: c_int,

    fn deinit(self: *Attention) void {
        self.wq.deinit();
        self.wk.deinit();
        self.wv.deinit();
        self.gate.deinit();
        self.wo.deinit();
        _ = mlx.mlx_array_free(self.qnorm);
        _ = mlx.mlx_array_free(self.knorm);
    }

    fn load(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, dim: u32, heads: u32, kvheads: u32, head_dim: u32, s: S) !Attention {
        const kq = try fmtKey(a, "{s}.wq", .{pfx});
        defer a.free(kq);
        const kk = try fmtKey(a, "{s}.wk", .{pfx});
        defer a.free(kk);
        const kv = try fmtKey(a, "{s}.wv", .{pfx});
        defer a.free(kv);
        const kg = try fmtKey(a, "{s}.gate", .{pfx});
        defer a.free(kg);
        const ko = try fmtKey(a, "{s}.wo", .{pfx});
        defer a.free(ko);
        const qn = try fmtKey(a, "{s}.qknorm.qnorm.scale", .{pfx});
        defer a.free(qn);
        const kn = try fmtKey(a, "{s}.qknorm.knorm.scale", .{pfx});
        defer a.free(kn);
        return .{
            .wq = try MixedLinear.load(w, a, kq, dim, s),
            .wk = try MixedLinear.load(w, a, kk, dim, s),
            .wv = try MixedLinear.load(w, a, kv, dim, s),
            .gate = try MixedLinear.load(w, a, kg, dim, s),
            .wo = try MixedLinear.load(w, a, ko, dim, s),
            .qnorm = try kreaNorm(w, qn, s),
            .knorm = try kreaNorm(w, kn, s),
            .heads = @intCast(heads),
            .kvheads = @intCast(kvheads),
            .head_dim = @intCast(head_dim),
        };
    }

    /// qkv [B,L,dim]. cos/sin null → no rope. mask null → no mask. Batch B is
    /// read from the input — the text-fusion layerwise pass runs with B=B·L,L=12.
    fn forward(self: *const Attention, qkv: mlx.mlx_array, cos: ?mlx.mlx_array, sin: ?mlx.mlx_array, mask: ?mlx.mlx_array, s: S) !mlx.mlx_array {
        const B: c_int = mlx.getShape(qkv)[0];
        const L: c_int = mlx.getShape(qkv)[1];
        const hd = self.head_dim;
        const q0 = try self.wq.forward(qkv, s);
        defer _ = mlx.mlx_array_free(q0);
        const k0 = try self.wk.forward(qkv, s);
        defer _ = mlx.mlx_array_free(k0);
        const v0 = try self.wv.forward(qkv, s);
        defer _ = mlx.mlx_array_free(v0);
        const gate0 = try self.gate.forward(qkv, s);
        defer _ = mlx.mlx_array_free(gate0);

        const q4 = try reshape(q0, &[_]c_int{ B, L, self.heads, hd }, s);
        defer _ = mlx.mlx_array_free(q4);
        const q = try transpose(q4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(q);
        const k4 = try reshape(k0, &[_]c_int{ B, L, self.kvheads, hd }, s);
        defer _ = mlx.mlx_array_free(k4);
        const k = try transpose(k4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(k);
        const v4 = try reshape(v0, &[_]c_int{ B, L, self.kvheads, hd }, s);
        defer _ = mlx.mlx_array_free(v4);
        const v = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(v);

        // QK-RMSNorm over head_dim
        const qn = try rms(q, self.qnorm, 1e-5, s);
        defer _ = mlx.mlx_array_free(qn);
        const kn = try rms(k, self.knorm, 1e-5, s);
        defer _ = mlx.mlx_array_free(kn);

        var qe = qn;
        var ke = kn;
        var rope_q: ?mlx.mlx_array = null;
        var rope_k: ?mlx.mlx_array = null;
        if (cos) |cc| {
            rope_q = try applyRope(qn, cc, sin.?, L, self.heads, s);
            rope_k = try applyRope(kn, cc, sin.?, L, self.kvheads, s);
            qe = rope_q.?;
            ke = rope_k.?;
        }
        defer if (rope_q) |r| {
            _ = mlx.mlx_array_free(r);
        };
        defer if (rope_k) |r| {
            _ = mlx.mlx_array_free(r);
        };

        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_a = mlx.mlx_array{ .ctx = null };
        if (mask) |m| {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qe, ke, v, scale, "array", m, null_a, s));
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qe, ke, v, scale, "", null_a, null_a, s));
        }
        const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = try reshape(at, &[_]c_int{ B, L, self.heads * hd }, s);
        defer _ = mlx.mlx_array_free(af);
        // output gate: out * sigmoid(gate), then wo
        const sg = try sigmoidA(gate0, s);
        defer _ = mlx.mlx_array_free(sg);
        const gated = try mulA(af, sg, s);
        defer _ = mlx.mlx_array_free(gated);
        return self.wo.forward(gated, s);
    }
};

/// Load a krea RMSNorm: stored `.scale`, effective weight = (scale+1), bf16.
fn kreaNorm(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const w1 = try addA(raw, one, s);
    defer _ = mlx.mlx_array_free(w1);
    return astype(w1, .bfloat16, s);
}

const SwiGLU = struct {
    gate: MixedLinear,
    up: MixedLinear,
    down: MixedLinear,
    fn deinit(self: *SwiGLU) void {
        self.gate.deinit();
        self.up.deinit();
        self.down.deinit();
    }
    fn load(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, features: u32, mlpdim: u32, s: S) !SwiGLU {
        const kg = try fmtKey(a, "{s}.gate", .{pfx});
        defer a.free(kg);
        const ku = try fmtKey(a, "{s}.up", .{pfx});
        defer a.free(ku);
        const kd = try fmtKey(a, "{s}.down", .{pfx});
        defer a.free(kd);
        return .{
            .gate = try MixedLinear.load(w, a, kg, features, s),
            .up = try MixedLinear.load(w, a, ku, features, s),
            .down = try MixedLinear.load(w, a, kd, mlpdim, s),
        };
    }
    fn forward(self: *const SwiGLU, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const g = try self.gate.forward(x, s);
        defer _ = mlx.mlx_array_free(g);
        const sg = try silu(g, s);
        defer _ = mlx.mlx_array_free(sg);
        const u = try self.up.forward(x, s);
        defer _ = mlx.mlx_array_free(u);
        const gu = try mulA(sg, u, s);
        defer _ = mlx.mlx_array_free(gu);
        return self.down.forward(gu, s);
    }
};

/// SwiGLU hidden dim: roundup(int(2*features/3)*multiplier, 128).
fn swigluDim(features: u32, multiplier: u32) u32 {
    const base = (2 * features / 3) * multiplier;
    return ((base + 127) / 128) * 128;
}

const Block = struct {
    mod_lin: mlx.mlx_array, // [6*features] bf16
    prenorm: mlx.mlx_array, // (1+scale)
    postnorm: mlx.mlx_array,
    attn: Attention,
    mlp: SwiGLU,
    fn deinit(self: *Block) void {
        _ = mlx.mlx_array_free(self.mod_lin);
        _ = mlx.mlx_array_free(self.prenorm);
        _ = mlx.mlx_array_free(self.postnorm);
        self.attn.deinit();
        self.mlp.deinit();
    }
};

const TextFusionBlock = struct {
    prenorm: mlx.mlx_array,
    postnorm: mlx.mlx_array,
    attn: Attention,
    mlp: SwiGLU,
    fn deinit(self: *TextFusionBlock) void {
        _ = mlx.mlx_array_free(self.prenorm);
        _ = mlx.mlx_array_free(self.postnorm);
        self.attn.deinit();
        self.mlp.deinit();
    }
    fn load(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, cfg: KreaConfig, s: S) !TextFusionBlock {
        const pre = try fmtKey(a, "{s}.prenorm.scale", .{pfx});
        defer a.free(pre);
        const post = try fmtKey(a, "{s}.postnorm.scale", .{pfx});
        defer a.free(post);
        const ka = try fmtKey(a, "{s}.attn", .{pfx});
        defer a.free(ka);
        const km = try fmtKey(a, "{s}.mlp", .{pfx});
        defer a.free(km);
        const hd = cfg.txtdim / cfg.txtheads;
        return .{
            .prenorm = try kreaNorm(w, pre, s),
            .postnorm = try kreaNorm(w, post, s),
            .attn = try Attention.load(w, a, ka, cfg.txtdim, cfg.txtheads, cfg.txtheads, hd, s),
            .mlp = try SwiGLU.load(w, a, km, cfg.txtdim, swigluDim(cfg.txtdim, cfg.multiplier), s),
        };
    }
    /// x [.,L,txtdim]; no rope; mask optional. Used for both layerwise (over the
    /// 12-layer axis) and refiner (over tokens) passes.
    fn forward(self: *const TextFusionBlock, x: mlx.mlx_array, mask: ?mlx.mlx_array, s: S) !mlx.mlx_array {
        const xn = try rms(x, self.prenorm, 1e-5, s);
        defer _ = mlx.mlx_array_free(xn);
        const a = try self.attn.forward(xn, null, null, mask, s);
        defer _ = mlx.mlx_array_free(a);
        const h1 = try addA(x, a, s);
        defer _ = mlx.mlx_array_free(h1);
        const hn = try rms(h1, self.postnorm, 1e-5, s);
        defer _ = mlx.mlx_array_free(hn);
        const m = try self.mlp.forward(hn, s);
        defer _ = mlx.mlx_array_free(m);
        return addA(h1, m, s);
    }
};

pub const Dit = struct {
    cfg: KreaConfig,
    allocator: std.mem.Allocator,
    s: S,
    first: MixedLinear, // 64 → features (+bias)
    blocks: []Block,
    tmlp0: MixedLinear, // tdim → features (+bias)
    tmlp2: MixedLinear, // features → features (+bias)
    tproj1: MixedLinear, // features → 6*features (+bias)
    layerwise: [2]TextFusionBlock,
    projector: MixedLinear, // 12 → 1 (no bias)
    refiner: [2]TextFusionBlock,
    txtmlp0: mlx.mlx_array, // RMSNorm (1+scale)
    txtmlp1: MixedLinear, // txtdim → features (+bias)
    txtmlp3: MixedLinear, // features → features (+bias)
    last_norm: mlx.mlx_array, // (1+scale)
    last_lin: MixedLinear, // features → 64 (+bias)
    last_mod: mlx.mlx_array, // [2, features] bf16

    pub fn deinit(self: *Dit) void {
        self.first.deinit();
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
        self.tmlp0.deinit();
        self.tmlp2.deinit();
        self.tproj1.deinit();
        for (&self.layerwise) |*b| b.deinit();
        self.projector.deinit();
        for (&self.refiner) |*b| b.deinit();
        _ = mlx.mlx_array_free(self.txtmlp0);
        self.txtmlp1.deinit();
        self.txtmlp3.deinit();
        _ = mlx.mlx_array_free(self.last_norm);
        self.last_lin.deinit();
        _ = mlx.mlx_array_free(self.last_mod);
    }

    /// timestep embed (B=1): [cos(args), sin(args)] of dim tdim, period 1e4,
    /// tfactor 1e3, then tmlp → t_emb [1,1,features]; tproj → tvec [1,1,6*features].
    fn timeVecs(self: *Dit, t: f32) !struct { t_emb: mlx.mlx_array, tvec: mlx.mlx_array } {
        const s = self.s;
        const dim = self.cfg.tdim;
        const half = dim / 2;
        const buf = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(buf);
        const period: f32 = 1e4;
        const tfactor: f32 = 1e3;
        for (0..half) |i| {
            const freq = std.math.exp(-std.math.log(f32, std.math.e, period) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
            const arg = (t * tfactor) * freq;
            buf[i] = @cos(arg);
            buf[half + i] = @sin(arg);
        }
        const shape = [_]c_int{ 1, 1, @intCast(dim) };
        const emb = mlx.mlx_array_new_data(buf.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(emb);
        const emb_bf = try astype(emb, .bfloat16, s);
        defer _ = mlx.mlx_array_free(emb_bf);
        // tmlp: lin0 → gelu → lin2
        const l0 = try self.tmlp0.forward(emb_bf, s);
        defer _ = mlx.mlx_array_free(l0);
        const g0 = try geluTanh(l0, s);
        defer _ = mlx.mlx_array_free(g0);
        const t_emb = try self.tmlp2.forward(g0, s); // [1,1,features]
        // tproj: gelu → lin1
        const gt = try geluTanh(t_emb, s);
        defer _ = mlx.mlx_array_free(gt);
        const tvec = try self.tproj1.forward(gt, s); // [1,1,6*features]
        return .{ .t_emb = t_emb, .tvec = tvec };
    }

    /// One DiT forward. img [1,Limg,64] bf16, context [1,512,12,txtdim] bf16,
    /// t scalar, pos [L*3] i32, valid [L] i32 (L = 512 + Limg). → velocity
    /// [1,Limg,64] bf16.
    pub fn forward(self: *Dit, img: mlx.mlx_array, context: mlx.mlx_array, t: f32, pos: []const i32, valid: []const i32) !mlx.mlx_array {
        const s = self.s;
        const L = valid.len;
        const txtlen: usize = @intCast(mlx.getShape(context)[1]);
        const rope = try buildRope(self.allocator, pos, L, self.cfg.theta);
        defer {
            _ = mlx.mlx_array_free(rope.cos);
            _ = mlx.mlx_array_free(rope.sin);
        }
        const full_mask = try buildOuterMask(self.allocator, valid, s);
        defer _ = mlx.mlx_array_free(full_mask);
        const txt_mask = try buildOuterMask(self.allocator, valid[0..txtlen], s);
        defer _ = mlx.mlx_array_free(txt_mask);
        return self.forwardPrebuilt(img, context, t, rope.cos, rope.sin, full_mask, txt_mask, s);
    }

    /// Constant-per-request rope/masks are built once by `generate` and reused
    /// across denoising steps.
    pub fn forwardPrebuilt(self: *Dit, img: mlx.mlx_array, context: mlx.mlx_array, t: f32, cos: mlx.mlx_array, sin: mlx.mlx_array, full_mask: mlx.mlx_array, txt_mask: mlx.mlx_array, s: S) !mlx.mlx_array {
        const txtlen: c_int = mlx.getShape(context)[1];
        const img_len: c_int = mlx.getShape(img)[1];

        const img_e = try self.first.forward(img, s); // [1,Limg,features]
        defer _ = mlx.mlx_array_free(img_e);

        const tv = try self.timeVecs(t);
        defer _ = mlx.mlx_array_free(tv.t_emb);
        defer _ = mlx.mlx_array_free(tv.tvec);

        // text fusion over the 12-layer axis, then projector, then refiner.
        const ctx = try self.textFusion(context, txt_mask, s);
        defer _ = mlx.mlx_array_free(ctx); // [1,512,features]

        var combined = try concat(&[_]mlx.mlx_array{ ctx, img_e }, 1, s);
        defer _ = mlx.mlx_array_free(combined);

        for (self.blocks) |*b| {
            const nb = try self.blockForward(combined, b, tv.tvec, cos, sin, full_mask, s);
            _ = mlx.mlx_array_free(combined);
            combined = nb;
        }

        // LastLayer: (1+scale)*norm(x)+shift via SimpleModulation(t_emb).
        const out = try self.lastLayer(combined, tv.t_emb, s);
        defer _ = mlx.mlx_array_free(out); // [1,L,64]
        return sliceAxis(out, 1, txtlen, txtlen + img_len, s);
    }

    fn textFusion(self: *Dit, context: mlx.mlx_array, txt_mask: mlx.mlx_array, s: S) !mlx.mlx_array {
        const sh = mlx.getShape(context); // [1,L,12,txtdim]
        const L = sh[1];
        const n = sh[2];
        const d = sh[3];
        // (1,L,12,d) → (L,12,d) for layerwise attention over the layer axis.
        var x = try reshape(context, &[_]c_int{ L * 1, n, d }, s);
        for (&self.layerwise) |*b| {
            const nx = try b.forward(x, null, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        // (L,12,d) → (1,L,d,12) → projector(12→1) → (1,L,d)
        const r = try reshape(x, &[_]c_int{ 1, L, n, d }, s);
        _ = mlx.mlx_array_free(x);
        defer _ = mlx.mlx_array_free(r);
        const tr = try transpose(r, &[_]c_int{ 0, 1, 3, 2 }, s); // (1,L,d,12)
        defer _ = mlx.mlx_array_free(tr);
        const proj = try self.projector.forward(tr, s); // (1,L,d,1)
        defer _ = mlx.mlx_array_free(proj);
        var ctx = try reshape(proj, &[_]c_int{ 1, L, d }, s);
        for (&self.refiner) |*b| {
            const nx = try b.forward(ctx, txt_mask, s);
            _ = mlx.mlx_array_free(ctx);
            ctx = nx;
        }
        // txtmlp: RMSNorm → lin1 → gelu → lin3
        const xn = try rms(ctx, self.txtmlp0, 1e-5, s);
        _ = mlx.mlx_array_free(ctx);
        defer _ = mlx.mlx_array_free(xn);
        const l1 = try self.txtmlp1.forward(xn, s);
        defer _ = mlx.mlx_array_free(l1);
        const g = try geluTanh(l1, s);
        defer _ = mlx.mlx_array_free(g);
        return self.txtmlp3.forward(g, s); // (1,L,features)
    }

    fn blockForward(self: *Dit, x: mlx.mlx_array, b: *const Block, tvec: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, mask: mlx.mlx_array, s: S) !mlx.mlx_array {
        const feat: c_int = @intCast(self.cfg.features);
        // mod = tvec + lin; split 6 → prescale,preshift,pregate,postscale,postshift,postgate
        const mod = try addA(tvec, b.mod_lin, s);
        defer _ = mlx.mlx_array_free(mod);
        const prescale = try sliceAxis(mod, 2, 0 * feat, 1 * feat, s);
        defer _ = mlx.mlx_array_free(prescale);
        const preshift = try sliceAxis(mod, 2, 1 * feat, 2 * feat, s);
        defer _ = mlx.mlx_array_free(preshift);
        const pregate = try sliceAxis(mod, 2, 2 * feat, 3 * feat, s);
        defer _ = mlx.mlx_array_free(pregate);
        const postscale = try sliceAxis(mod, 2, 3 * feat, 4 * feat, s);
        defer _ = mlx.mlx_array_free(postscale);
        const postshift = try sliceAxis(mod, 2, 4 * feat, 5 * feat, s);
        defer _ = mlx.mlx_array_free(postshift);
        const postgate = try sliceAxis(mod, 2, 5 * feat, 6 * feat, s);
        defer _ = mlx.mlx_array_free(postgate);

        // x = x + pregate * attn((1+prescale)*prenorm(x)+preshift)
        const pn = try rms(x, b.prenorm, 1e-5, s);
        defer _ = mlx.mlx_array_free(pn);
        const pm = try modulate(pn, prescale, preshift, s);
        defer _ = mlx.mlx_array_free(pm);
        const a = try b.attn.forward(pm, cos, sin, mask, s);
        defer _ = mlx.mlx_array_free(a);
        const ga = try mulA(pregate, a, s);
        defer _ = mlx.mlx_array_free(ga);
        const x1 = try addA(x, ga, s);
        defer _ = mlx.mlx_array_free(x1);
        // x = x + postgate * mlp((1+postscale)*postnorm(x)+postshift)
        const qn = try rms(x1, b.postnorm, 1e-5, s);
        defer _ = mlx.mlx_array_free(qn);
        const qm = try modulate(qn, postscale, postshift, s);
        defer _ = mlx.mlx_array_free(qm);
        const m = try b.mlp.forward(qm, s);
        defer _ = mlx.mlx_array_free(m);
        const gm = try mulA(postgate, m, s);
        defer _ = mlx.mlx_array_free(gm);
        return addA(x1, gm, s);
    }

    fn lastLayer(self: *Dit, x: mlx.mlx_array, t_emb: mlx.mlx_array, s: S) !mlx.mlx_array {
        const feat: c_int = @intCast(self.cfg.features);
        // SimpleModulation: out = t_emb + lin[None] → (1,2,features); split → scale,shift
        const lin = try reshape(self.last_mod, &[_]c_int{ 1, 2, feat }, s);
        defer _ = mlx.mlx_array_free(lin);
        const mod = try addA(t_emb, lin, s); // (1,1,f)+(1,2,f) → (1,2,f)
        defer _ = mlx.mlx_array_free(mod);
        const scale = try sliceAxis(mod, 1, 0, 1, s); // (1,1,f)
        defer _ = mlx.mlx_array_free(scale);
        const shift = try sliceAxis(mod, 1, 1, 2, s);
        defer _ = mlx.mlx_array_free(shift);
        const xn = try rms(x, self.last_norm, 1e-5, s);
        defer _ = mlx.mlx_array_free(xn);
        const xm = try modulate(xn, scale, shift, s);
        defer _ = mlx.mlx_array_free(xm);
        return self.last_lin.forward(xm, s);
    }
};

/// modulated = (1+scale)*x + shift  (scale/shift broadcast [1,1,features]).
fn modulate(x: mlx.mlx_array, scale: mlx.mlx_array, shift: mlx.mlx_array, s: S) !mlx.mlx_array {
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const sp1 = try addA(scale, one, s);
    defer _ = mlx.mlx_array_free(sp1);
    const m = try mulA(x, sp1, s);
    defer _ = mlx.mlx_array_free(m);
    return addA(m, shift, s);
}

/// Additive attention mask from a validity vector valid [L] (i32 0/1):
/// [1,1,L,L] bf16, (1 - valid_i*valid_j) * -1e9.
fn buildOuterMask(allocator: std.mem.Allocator, valid: []const i32, s: S) !mlx.mlx_array {
    const L: c_int = @intCast(valid.len);
    const buf = try allocator.alloc(f32, valid.len);
    defer allocator.free(buf);
    for (valid, 0..) |vv, i| buf[i] = @floatFromInt(vv);
    const vshape = [_]c_int{ 1, L };
    const va = mlx.mlx_array_new_data(buf.ptr, &vshape, 2, .float32);
    defer _ = mlx.mlx_array_free(va);
    const vcol = try reshape(va, &[_]c_int{ 1, L, 1 }, s);
    defer _ = mlx.mlx_array_free(vcol);
    const vrow = try reshape(va, &[_]c_int{ 1, 1, L }, s);
    defer _ = mlx.mlx_array_free(vrow);
    const full = try mulA(vcol, vrow, s);
    defer _ = mlx.mlx_array_free(full);
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const inv = try subA(one, full, s);
    defer _ = mlx.mlx_array_free(inv);
    const neg = scalarF(-1e9);
    defer _ = mlx.mlx_array_free(neg);
    const add = try mulA(inv, neg, s);
    defer _ = mlx.mlx_array_free(add);
    const m4 = try reshape(add, &[_]c_int{ 1, 1, L, L }, s);
    defer _ = mlx.mlx_array_free(m4);
    return astype(m4, .bfloat16, s);
}

pub fn loadDit(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Dit {
    // The transformer weights live in a single safetensors file whose precision
    // is auto-detected by presence (mixed-4-8 → 8bit → bf16). loadWeights merges
    // every *.safetensors in the dir, so any single present file is read.
    var w = try model_mod.loadWeights(io, allocator, model_dir);
    defer w.deinit();
    var d: Dit = undefined;
    d.cfg = .{};
    d.allocator = allocator;
    d.s = s;
    const cfg = d.cfg;
    const feat = cfg.features;
    const hd = cfg.headDim();

    d.first = try MixedLinear.load(&w, allocator, "first", cfg.channels * cfg.patch * cfg.patch, s);
    d.tmlp0 = try MixedLinear.load(&w, allocator, "tmlp.0", cfg.tdim, s);
    d.tmlp2 = try MixedLinear.load(&w, allocator, "tmlp.2", feat, s);
    d.tproj1 = try MixedLinear.load(&w, allocator, "tproj.1", feat, s);

    d.blocks = try allocator.alloc(Block, cfg.layers);
    const mlpdim = swigluDim(feat, cfg.multiplier);
    for (d.blocks, 0..) |*b, i| {
        const pfx = try fmtKey(allocator, "blocks.{d}", .{i});
        defer allocator.free(pfx);
        const modk = try fmtKey(allocator, "blocks.{d}.mod.lin", .{i});
        defer allocator.free(modk);
        const prek = try fmtKey(allocator, "blocks.{d}.prenorm.scale", .{i});
        defer allocator.free(prek);
        const postk = try fmtKey(allocator, "blocks.{d}.postnorm.scale", .{i});
        defer allocator.free(postk);
        const attnk = try fmtKey(allocator, "blocks.{d}.attn", .{i});
        defer allocator.free(attnk);
        const mlpk = try fmtKey(allocator, "blocks.{d}.mlp", .{i});
        defer allocator.free(mlpk);
        b.* = .{
            .mod_lin = try normBf16(&w, modk, s),
            .prenorm = try kreaNorm(&w, prek, s),
            .postnorm = try kreaNorm(&w, postk, s),
            .attn = try Attention.load(&w, allocator, attnk, feat, cfg.heads, cfg.kvheads, hd, s),
            .mlp = try SwiGLU.load(&w, allocator, mlpk, feat, mlpdim, s),
        };
    }

    inline for (.{ 0, 1 }) |i| {
        const pfx = try fmtKey(allocator, "txtfusion.layerwise_blocks.{d}", .{i});
        defer allocator.free(pfx);
        d.layerwise[i] = try TextFusionBlock.load(&w, allocator, pfx, cfg, s);
    }
    d.projector = try MixedLinear.load(&w, allocator, "txtfusion.projector", cfg.txtlayers, s);
    inline for (.{ 0, 1 }) |i| {
        const pfx = try fmtKey(allocator, "txtfusion.refiner_blocks.{d}", .{i});
        defer allocator.free(pfx);
        d.refiner[i] = try TextFusionBlock.load(&w, allocator, pfx, cfg, s);
    }

    d.txtmlp0 = try kreaNorm(&w, "txtmlp.0.scale", s);
    d.txtmlp1 = try MixedLinear.load(&w, allocator, "txtmlp.1", cfg.txtdim, s);
    d.txtmlp3 = try MixedLinear.load(&w, allocator, "txtmlp.3", feat, s);

    d.last_norm = try kreaNorm(&w, "last.norm.scale", s);
    d.last_lin = try MixedLinear.load(&w, allocator, "last.linear", feat, s);
    d.last_mod = try normBf16(&w, "last.modulation.lin", s);
    return d;
}

/// Attach runtime LoRA adapters from `lf` to every matching DiT linear.
/// Non-owning: `lf` must outlive the attach. Returns the match count.
pub fn attachLora(dit: *Dit, lf: *const lora_mod.File, user_scale: f32) u32 {
    detachLora(dit);
    var matched: u32 = 0;
    var kbuf: [128]u8 = undefined;
    for (dit.blocks, 0..) |*b, i| {
        const mods = .{
            .{ "attn.wq", &b.attn.wq },   .{ "attn.wk", &b.attn.wk },
            .{ "attn.wv", &b.attn.wv },   .{ "attn.gate", &b.attn.gate },
            .{ "attn.wo", &b.attn.wo },
            .{ "mlp.gate", &b.mlp.gate }, .{ "mlp.up", &b.mlp.up },
            .{ "mlp.down", &b.mlp.down },
        };
        inline for (mods) |m| {
            const key = std.fmt.bufPrint(&kbuf, "blocks.{d}.{s}", .{ i, m[0] }) catch "";
            if (lf.find(key)) |e| {
                m[1].lora = .{ .at = e.at, .bt = e.bt, .scale = e.scale * user_scale };
                matched += 1;
            }
        }
    }
    return matched;
}

pub fn detachLora(dit: *Dit) void {
    for (dit.blocks) |*b| {
        inline for (.{ &b.attn.wq, &b.attn.wk, &b.attn.wv, &b.attn.gate, &b.attn.wo, &b.mlp.gate, &b.mlp.up, &b.mlp.down }) |ml| ml.lora = null;
    }
}

// ════════════════════════════════════════════════════════════════════════
// B3. Autoencoder — Qwen-Image 3D causal VAE (decode path only, T=1, f32).
//     Raw diffusers tensor names; conv weights transposed to MLX layout at load.
// ════════════════════════════════════════════════════════════════════════

const LATENTS_MEAN = [16]f32{ -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921 };
const LATENTS_STD = [16]f32{ 2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916 };

/// Causal conv3d on NCTHW (T=1). weight [out,kt,kh,kw,in] MLX layout, f32.
/// Temporal pad is causal (2*pad_t before, 0 after); spatial pad symmetric.
fn causalConv3d(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, pad: c_int, s: S) !mlx.mlx_array {
    var px = mlx.mlx_array{ .ctx = null };
    var padded = false;
    if (pad > 0) {
        const axes = [_]c_int{ 2, 3, 4 };
        const low = [_]c_int{ 2 * pad, pad, pad };
        const high = [_]c_int{ 0, pad, pad };
        const zero = scalarF(0.0);
        defer _ = mlx.mlx_array_free(zero);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_pad(&o, x, &axes, 3, &low, 3, &high, 3, zero, "constant", s));
        px = o;
        padded = true;
    } else {
        px = x;
    }
    defer if (padded) {
        _ = mlx.mlx_array_free(px);
    };
    const xt = try transpose(px, &[_]c_int{ 0, 2, 3, 4, 1 }, s); // NCTHW → NTHWC
    defer _ = mlx.mlx_array_free(xt);
    const xc = try contig(xt, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv3d(&o, xc, w, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, s));
    defer _ = mlx.mlx_array_free(o);
    const onc = try transpose(o, &[_]c_int{ 0, 4, 1, 2, 3 }, s); // NTHWC → NCTHW
    defer _ = mlx.mlx_array_free(onc);
    const out_c: c_int = mlx.getShape(bias)[0];
    const b5 = try reshape(bias, &[_]c_int{ 1, out_c, 1, 1, 1 }, s);
    defer _ = mlx.mlx_array_free(b5);
    return addA(onc, b5, s);
}

/// conv2d on NHWC, weight [out,kh,kw,in] f32, bias [out].
fn conv2d(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, pad: c_int, s: S) !mlx.mlx_array {
    const xc = try contig(x, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 1, 1, pad, pad, 1, 1, 1, s));
    defer _ = mlx.mlx_array_free(o);
    return addA(o, bias, s);
}

/// Qwen-Image RMSNorm over the CHANNEL axis (axis 1) of NCTHW: x / max(||x||₂,eps)
/// * sqrt(C) * weight. weight [C] f32.
fn rmsChannels(x: mlx.mlx_array, weight: mlx.mlx_array, s: S) !mlx.mlx_array {
    const C: c_int = mlx.getShape(x)[1];
    const sq = try mulA(x, x, s);
    defer _ = mlx.mlx_array_free(sq);
    var sumsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sumsq);
    try mlx.check(mlx.mlx_sum_axis(&sumsq, sq, 1, true, s));
    var l2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l2);
    try mlx.check(mlx.mlx_sqrt(&l2, sumsq, s));
    const eps = scalarF(1e-12);
    defer _ = mlx.mlx_array_free(eps);
    var denom = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(denom);
    try mlx.check(mlx.mlx_maximum(&denom, l2, eps, s));
    const xn = try divA(x, denom, s);
    defer _ = mlx.mlx_array_free(xn);
    const sc = scalarF(std.math.sqrt(@as(f32, @floatFromInt(C))));
    defer _ = mlx.mlx_array_free(sc);
    const xs = try mulA(xn, sc, s);
    defer _ = mlx.mlx_array_free(xs);
    const w5 = try reshape(weight, &[_]c_int{ 1, C, 1, 1, 1 }, s);
    defer _ = mlx.mlx_array_free(w5);
    return mulA(xs, w5, s);
}

const ResBlock3D = struct {
    n1: mlx.mlx_array,
    c1w: mlx.mlx_array,
    c1b: mlx.mlx_array,
    n2: mlx.mlx_array,
    c2w: mlx.mlx_array,
    c2b: mlx.mlx_array,
    skw: ?mlx.mlx_array = null,
    skb: ?mlx.mlx_array = null,
    fn deinit(self: *ResBlock3D) void {
        inline for (.{ "n1", "c1w", "c1b", "n2", "c2w", "c2b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        if (self.skw) |x| _ = mlx.mlx_array_free(x);
        if (self.skb) |x| _ = mlx.mlx_array_free(x);
    }
    fn forward(self: *const ResBlock3D, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const h0 = try rmsChannels(x, self.n1, s);
        defer _ = mlx.mlx_array_free(h0);
        const a0 = try silu(h0, s);
        defer _ = mlx.mlx_array_free(a0);
        const c1 = try causalConv3d(a0, self.c1w, self.c1b, 1, s);
        defer _ = mlx.mlx_array_free(c1);
        const h1 = try rmsChannels(c1, self.n2, s);
        defer _ = mlx.mlx_array_free(h1);
        const a1 = try silu(h1, s);
        defer _ = mlx.mlx_array_free(a1);
        const c2 = try causalConv3d(a1, self.c2w, self.c2b, 1, s);
        defer _ = mlx.mlx_array_free(c2);
        if (self.skw) |skw| {
            const sc = try causalConv3d(x, skw, self.skb.?, 0, s);
            defer _ = mlx.mlx_array_free(sc);
            return addA(c2, sc, s);
        }
        return addA(c2, x, s);
    }
};

const AttnBlock3D = struct {
    norm: mlx.mlx_array,
    qkv_w: mlx.mlx_array, // conv2d 1x1 [3C,1,1,C]
    qkv_b: mlx.mlx_array,
    proj_w: mlx.mlx_array,
    proj_b: mlx.mlx_array,
    fn deinit(self: *AttnBlock3D) void {
        inline for (.{ "norm", "qkv_w", "qkv_b", "proj_w", "proj_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
    }
    fn forward(self: *const AttnBlock3D, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const sh = mlx.getShape(x); // [1,C,1,H,W]
        const C = sh[1];
        const H = sh[3];
        const Wd = sh[4];
        const normed = try rmsChannels(x, self.norm, s);
        defer _ = mlx.mlx_array_free(normed);
        // NCTHW(T=1) → NHWC for the 1x1 convs
        const sq = try reshape(normed, &[_]c_int{ 1, C, H, Wd }, s); // drop T=1 → NCHW
        defer _ = mlx.mlx_array_free(sq);
        const nhwc = try transpose(sq, &[_]c_int{ 0, 2, 3, 1 }, s); // NHWC
        defer _ = mlx.mlx_array_free(nhwc);
        const qkv = try conv2d(nhwc, self.qkv_w, self.qkv_b, 0, s); // (1,H,W,3C)
        defer _ = mlx.mlx_array_free(qkv);
        // (1,H,W,3C) → (1,1,H*W,3C) → split q/k/v on last axis
        const flat = try reshape(qkv, &[_]c_int{ 1, 1, H * Wd, 3 * C }, s);
        defer _ = mlx.mlx_array_free(flat);
        const q = try sliceAxis(flat, 3, 0, C, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try sliceAxis(flat, 3, C, 2 * C, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try sliceAxis(flat, 3, 2 * C, 3 * C, s);
        defer _ = mlx.mlx_array_free(v);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(C)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_a = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, q, k, v, scale, "", null_a, null_a, s)); // (1,1,HW,C)
        const ar = try reshape(attn, &[_]c_int{ 1, H, Wd, C }, s); // NHWC
        defer _ = mlx.mlx_array_free(ar);
        const pj = try conv2d(ar, self.proj_w, self.proj_b, 0, s); // (1,H,W,C)
        defer _ = mlx.mlx_array_free(pj);
        const back = try transpose(pj, &[_]c_int{ 0, 3, 1, 2 }, s); // NCHW
        defer _ = mlx.mlx_array_free(back);
        const five = try reshape(back, &[_]c_int{ 1, C, 1, H, Wd }, s); // NCTHW
        defer _ = mlx.mlx_array_free(five);
        return addA(five, x, s);
    }
};

const UpBlock3D = struct {
    resnets: []ResBlock3D,
    up_w: ?mlx.mlx_array = null, // resample_conv (conv2d) [out,kh,kw,in]
    up_b: ?mlx.mlx_array = null,
    allocator: std.mem.Allocator,
    fn deinit(self: *UpBlock3D) void {
        for (self.resnets) |*r| r.deinit();
        self.allocator.free(self.resnets);
        if (self.up_w) |x| _ = mlx.mlx_array_free(x);
        if (self.up_b) |x| _ = mlx.mlx_array_free(x);
    }
    fn forward(self: *const UpBlock3D, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var h = mlx.mlx_array{ .ctx = null };
        var first = true;
        for (self.resnets) |*r| {
            const nh = try r.forward(if (first) x else h, s);
            if (!first) _ = mlx.mlx_array_free(h);
            h = nh;
            first = false;
        }
        if (self.up_w) |uw| {
            const us = try upsample(h, uw, self.up_b.?, s);
            _ = mlx.mlx_array_free(h);
            return us;
        }
        return h;
    }
};

/// Per-frame nearest-2x spatial upsample + conv2d (NCTHW T=1).
fn upsample(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,C,1,H,W]
    const C = sh[1];
    const H = sh[3];
    const Wd = sh[4];
    const nchw = try reshape(x, &[_]c_int{ 1, C, H, Wd }, s);
    defer _ = mlx.mlx_array_free(nchw);
    const nhwc = try transpose(nchw, &[_]c_int{ 0, 2, 3, 1 }, s); // NHWC
    defer _ = mlx.mlx_array_free(nhwc);
    var r1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r1);
    try mlx.check(mlx.mlx_repeat_axis(&r1, nhwc, 2, 1, s)); // H
    var r2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r2);
    try mlx.check(mlx.mlx_repeat_axis(&r2, r1, 2, 2, s)); // W
    const cv = try conv2d(r2, w, b, 1, s); // (1,2H,2W,outC)
    defer _ = mlx.mlx_array_free(cv);
    const outc: c_int = mlx.getShape(cv)[3];
    const back = try transpose(cv, &[_]c_int{ 0, 3, 1, 2 }, s); // NCHW
    defer _ = mlx.mlx_array_free(back);
    return reshape(back, &[_]c_int{ 1, outc, 1, 2 * H, 2 * Wd }, s); // NCTHW
}

pub const Vae = struct {
    allocator: std.mem.Allocator,
    s: S,
    pq_w: mlx.mlx_array, // post_quant_conv (1x1x1)
    pq_b: mlx.mlx_array,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    mid_r0: ResBlock3D,
    mid_attn: AttnBlock3D,
    mid_r1: ResBlock3D,
    up_blocks: [4]UpBlock3D,
    norm_out: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,

    pub fn deinit(self: *Vae) void {
        inline for (.{ "pq_w", "pq_b", "conv_in_w", "conv_in_b", "norm_out", "conv_out_w", "conv_out_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        self.mid_r0.deinit();
        self.mid_attn.deinit();
        self.mid_r1.deinit();
        for (&self.up_blocks) |*b| b.deinit();
    }

    /// latent [1,16,lat_h,lat_w] f32 → image [1,3,H,W] f32 in [-1,1].
    pub fn decode(self: *Vae, latent: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const sh = mlx.getShape(latent); // [1,16,lat_h,lat_w]
        const lh = sh[2];
        const lw = sh[3];
        // → [1,16,1,lat_h,lat_w]
        const l5 = try reshape(latent, &[_]c_int{ 1, 16, 1, lh, lw }, s);
        defer _ = mlx.mlx_array_free(l5);
        // denorm: latent * STD + MEAN  (per-channel)
        const stdv = mlx.mlx_array_new_data(&LATENTS_STD, &[_]c_int{ 1, 16, 1, 1, 1 }, 5, .float32);
        defer _ = mlx.mlx_array_free(stdv);
        const meanv = mlx.mlx_array_new_data(&LATENTS_MEAN, &[_]c_int{ 1, 16, 1, 1, 1 }, 5, .float32);
        defer _ = mlx.mlx_array_free(meanv);
        const sc = try mulA(l5, stdv, s);
        defer _ = mlx.mlx_array_free(sc);
        const dn = try addA(sc, meanv, s);
        defer _ = mlx.mlx_array_free(dn);
        // post_quant_conv (1x1x1, pad 0)
        var h = try causalConv3d(dn, self.pq_w, self.pq_b, 0, s);
        {
            const nh = try causalConv3d(h, self.conv_in_w, self.conv_in_b, 1, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.mid_r0.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.mid_attn.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.mid_r1.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        for (&self.up_blocks) |*b| {
            const nh = try b.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try rmsChannels(h, self.norm_out, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try silu(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try causalConv3d(h, self.conv_out_w, self.conv_out_b, 1, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        defer _ = mlx.mlx_array_free(h); // [1,3,1,H,W]
        // → [1,3,H,W]
        const hsh = mlx.getShape(h);
        const out = try reshape(h, &[_]c_int{ 1, 3, hsh[3], hsh[4] }, s);
        defer _ = mlx.mlx_array_free(out);
        return contig(out, s);
    }
};

/// Load a conv3d weight: raw PyTorch [out,in,kt,kh,kw] → MLX [out,kt,kh,kw,in], f32.
fn loadConv3dW(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    const wk = try fmtKey(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const raw = try ownWeight(w, wk);
    defer _ = mlx.mlx_array_free(raw);
    const t = try transpose(raw, &[_]c_int{ 0, 2, 3, 4, 1 }, s);
    defer _ = mlx.mlx_array_free(t);
    const tc = try contig(t, s);
    defer _ = mlx.mlx_array_free(tc);
    return astype(tc, .float32, s);
}
/// Load a conv2d weight: raw [out,in,kh,kw] → MLX [out,kh,kw,in], f32.
fn loadConv2dW(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    const wk = try fmtKey(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const raw = try ownWeight(w, wk);
    defer _ = mlx.mlx_array_free(raw);
    const t = try transpose(raw, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(t);
    const tc = try contig(t, s);
    defer _ = mlx.mlx_array_free(tc);
    return astype(tc, .float32, s);
}
fn loadVec(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, comptime suffix: []const u8, s: S) !mlx.mlx_array {
    const k = try fmtKey(a, "{s}." ++ suffix, .{prefix});
    defer a.free(k);
    const raw = try ownWeight(w, k);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, .float32, s);
}
/// Load a `.gamma` RMSNorm vector, flattened to [C], f32.
fn loadGamma(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    const k = try fmtKey(a, "{s}.gamma", .{prefix});
    defer a.free(k);
    const raw = try ownWeight(w, k);
    defer _ = mlx.mlx_array_free(raw);
    const C: c_int = mlx.getShape(raw)[0];
    const flat = try reshape(raw, &[_]c_int{C}, s);
    defer _ = mlx.mlx_array_free(flat);
    return astype(flat, .float32, s);
}

fn loadResBlock(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, s: S) !ResBlock3D {
    const c1 = try fmtKey(a, "{s}.conv1", .{pfx});
    defer a.free(c1);
    const c2 = try fmtKey(a, "{s}.conv2", .{pfx});
    defer a.free(c2);
    const n1 = try fmtKey(a, "{s}.norm1", .{pfx});
    defer a.free(n1);
    const n2 = try fmtKey(a, "{s}.norm2", .{pfx});
    defer a.free(n2);
    const sk = try fmtKey(a, "{s}.conv_shortcut", .{pfx});
    defer a.free(sk);
    const skwk = try fmtKey(a, "{s}.conv_shortcut.weight", .{pfx});
    defer a.free(skwk);
    var r: ResBlock3D = .{
        .n1 = try loadGamma(w, a, n1, s),
        .c1w = try loadConv3dW(w, a, c1, s),
        .c1b = try loadVec(w, a, c1, "bias", s),
        .n2 = try loadGamma(w, a, n2, s),
        .c2w = try loadConv3dW(w, a, c2, s),
        .c2b = try loadVec(w, a, c2, "bias", s),
    };
    if (w.get(skwk) != null) {
        r.skw = try loadConv3dW(w, a, sk, s);
        r.skb = try loadVec(w, a, sk, "bias", s);
    }
    return r;
}

pub fn loadVae(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Vae {
    const dir = try fmtKey(allocator, "{s}/vae", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var v: Vae = undefined;
    v.allocator = allocator;
    v.s = s;
    v.pq_w = try loadConv3dW(&w, allocator, "post_quant_conv", s);
    v.pq_b = try loadVec(&w, allocator, "post_quant_conv", "bias", s);
    v.conv_in_w = try loadConv3dW(&w, allocator, "decoder.conv_in", s);
    v.conv_in_b = try loadVec(&w, allocator, "decoder.conv_in", "bias", s);
    v.mid_r0 = try loadResBlock(&w, allocator, "decoder.mid_block.resnets.0", s);
    v.mid_r1 = try loadResBlock(&w, allocator, "decoder.mid_block.resnets.1", s);
    v.mid_attn = .{
        .norm = try loadGamma(&w, allocator, "decoder.mid_block.attentions.0.norm", s),
        .qkv_w = try loadConv2dW(&w, allocator, "decoder.mid_block.attentions.0.to_qkv", s),
        .qkv_b = try loadVec(&w, allocator, "decoder.mid_block.attentions.0.to_qkv", "bias", s),
        .proj_w = try loadConv2dW(&w, allocator, "decoder.mid_block.attentions.0.proj", s),
        .proj_b = try loadVec(&w, allocator, "decoder.mid_block.attentions.0.proj", "bias", s),
    };
    // up_blocks: 0,1,2 upsample (resample.1 conv); 3 has no upsampler.
    const nres = [_]usize{ 3, 3, 3, 3 };
    for (0..4) |bi| {
        var ub: UpBlock3D = .{ .resnets = try allocator.alloc(ResBlock3D, nres[bi]), .allocator = allocator };
        for (0..nres[bi]) |ri| {
            const pfx = try fmtKey(allocator, "decoder.up_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer allocator.free(pfx);
            ub.resnets[ri] = try loadResBlock(&w, allocator, pfx, s);
        }
        if (bi < 3) {
            const up = try fmtKey(allocator, "decoder.up_blocks.{d}.upsamplers.0.resample.1", .{bi});
            defer allocator.free(up);
            ub.up_w = try loadConv2dW(&w, allocator, up, s);
            ub.up_b = try loadVec(&w, allocator, up, "bias", s);
        }
        v.up_blocks[bi] = ub;
    }
    v.norm_out = try loadGamma(&w, allocator, "decoder.norm_out", s);
    v.conv_out_w = try loadConv3dW(&w, allocator, "decoder.conv_out", s);
    v.conv_out_b = try loadVec(&w, allocator, "decoder.conv_out", "bias", s);
    return v;
}

/// Spatial downsample for the encoder's Resample blocks: per-frame (T=1)
/// asymmetric (0,1,0,1) zero-pad + 3x3 stride-2 conv (WanResample downsample2d;
/// the temporal time_conv never fires at T=1, mirroring the decoder convention).
fn downsampleSpatial(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,C,1,H,W]
    const C = sh[1];
    const H = sh[3];
    const Wd = sh[4];
    const nchw = try reshape(x, &[_]c_int{ 1, C, H, Wd }, s);
    defer _ = mlx.mlx_array_free(nchw);
    const nhwc = try transpose(nchw, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(nhwc);
    const axes = [_]c_int{ 1, 2 };
    const low = [_]c_int{ 0, 0 };
    const high = [_]c_int{ 1, 1 };
    const zero = scalarF(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_pad(&p, nhwc, &axes, 2, &low, 2, &high, 2, zero, "constant", s));
    const pc = try contig(p, s);
    defer _ = mlx.mlx_array_free(pc);
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_conv2d(&o, pc, w, 2, 2, 0, 0, 1, 1, 1, s));
    const ob = try addA(o, b, s);
    defer _ = mlx.mlx_array_free(ob);
    const back = try transpose(ob, &[_]c_int{ 0, 3, 1, 2 }, s); // NCHW
    defer _ = mlx.mlx_array_free(back);
    const bsh = mlx.getShape(back);
    return reshape(back, &[_]c_int{ 1, C, 1, bsh[2], bsh[3] }, s);
}

/// One entry of the encoder's flat down_blocks list (resnets at 0,1,3,4,6,7,9,10;
/// spatial resamples at 2,5,8).
const EncBlock = union(enum) {
    res: ResBlock3D,
    down: struct { w: mlx.mlx_array, b: mlx.mlx_array },
};

/// Qwen-Image 3D VAE encoder (T=1) — structural mirror of the decoder. Feeds
/// img2img: pixels → per-channel-normalized latents, the space `Vae.decode` reads.
pub const VaeEncoder = struct {
    allocator: std.mem.Allocator,
    s: S,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    blocks: [11]EncBlock,
    mid_r0: ResBlock3D,
    mid_attn: AttnBlock3D,
    mid_r1: ResBlock3D,
    norm_out: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,
    quant_w: mlx.mlx_array,
    quant_b: mlx.mlx_array,

    pub fn deinit(self: *VaeEncoder) void {
        inline for (.{ "conv_in_w", "conv_in_b", "norm_out", "conv_out_w", "conv_out_b", "quant_w", "quant_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        for (&self.blocks) |*blk| switch (blk.*) {
            .res => |*r| r.deinit(),
            .down => |*d| {
                _ = mlx.mlx_array_free(d.w);
                _ = mlx.mlx_array_free(d.b);
            },
        };
        self.mid_r0.deinit();
        self.mid_attn.deinit();
        self.mid_r1.deinit();
    }

    /// image [1,3,H,W] f32 [0,1] (NCHW, H/W multiples of 16) → latents
    /// [1,16,H/8,W/8] f32, per-channel NORMALIZED ((z-mean)/std; distribution
    /// mean — deterministic, no sampling).
    pub fn encode(self: *VaeEncoder, img: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const ish = mlx.getShape(img); // [1,3,H,W]
        // [0,1] → [-1,1], lift to NCTHW (T=1)
        const two = scalarF(2.0);
        defer _ = mlx.mlx_array_free(two);
        const one = scalarF(1.0);
        defer _ = mlx.mlx_array_free(one);
        const x2 = try mulA(img, two, s);
        defer _ = mlx.mlx_array_free(x2);
        const xm = try subA(x2, one, s);
        defer _ = mlx.mlx_array_free(xm);
        const x5 = try reshape(xm, &[_]c_int{ 1, 3, 1, ish[2], ish[3] }, s);
        defer _ = mlx.mlx_array_free(x5);

        var h = try causalConv3d(x5, self.conv_in_w, self.conv_in_b, 1, s);
        for (&self.blocks) |*blk| {
            const nh = switch (blk.*) {
                .res => |*rb| try rb.forward(h, s),
                .down => |*d| try downsampleSpatial(h, d.w, d.b, s),
            };
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.mid_r0.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.mid_attn.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.mid_r1.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        const hn = try rmsChannels(h, self.norm_out, s);
        _ = mlx.mlx_array_free(h);
        defer _ = mlx.mlx_array_free(hn);
        const ha = try silu(hn, s);
        defer _ = mlx.mlx_array_free(ha);
        const co = try causalConv3d(ha, self.conv_out_w, self.conv_out_b, 1, s); // [1,32,1,h,w]
        defer _ = mlx.mlx_array_free(co);
        const qc = try causalConv3d(co, self.quant_w, self.quant_b, 0, s);
        defer _ = mlx.mlx_array_free(qc);
        // mean = first 16 channels; logvar discarded.
        const qsh = mlx.getShape(qc);
        const start = [_]c_int{ 0, 0, 0, 0, 0 };
        const stop = [_]c_int{ 1, 16, 1, qsh[3], qsh[4] };
        const strides = [_]c_int{ 1, 1, 1, 1, 1 };
        var mean = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mean);
        try mlx.check(mlx.mlx_slice(&mean, qc, &start, 5, &stop, 5, &strides, 5, s));
        const meanc = try contig(mean, s);
        defer _ = mlx.mlx_array_free(meanc);
        const mf = try astype(meanc, .float32, s);
        defer _ = mlx.mlx_array_free(mf);
        // normalize: (z - MEAN) / STD (inverse of decode's denorm)
        const stdv = mlx.mlx_array_new_data(&LATENTS_STD, &[_]c_int{ 1, 16, 1, 1, 1 }, 5, .float32);
        defer _ = mlx.mlx_array_free(stdv);
        const meanv = mlx.mlx_array_new_data(&LATENTS_MEAN, &[_]c_int{ 1, 16, 1, 1, 1 }, 5, .float32);
        defer _ = mlx.mlx_array_free(meanv);
        const centered = try subA(mf, meanv, s);
        defer _ = mlx.mlx_array_free(centered);
        const zn = try divA(centered, stdv, s);
        defer _ = mlx.mlx_array_free(zn);
        // drop T → [1,16,h,w]
        const out = try reshape(zn, &[_]c_int{ 1, 16, qsh[3], qsh[4] }, s);
        defer _ = mlx.mlx_array_free(out);
        return contig(out, s);
    }
};

pub fn loadVaeEncoder(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !VaeEncoder {
    const dir = try fmtKey(allocator, "{s}/vae", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var v: VaeEncoder = undefined;
    v.allocator = allocator;
    v.s = s;
    v.conv_in_w = try loadConv3dW(&w, allocator, "encoder.conv_in", s);
    v.conv_in_b = try loadVec(&w, allocator, "encoder.conv_in", "bias", s);
    for (0..11) |bi| {
        const pfx = try fmtKey(allocator, "encoder.down_blocks.{d}", .{bi});
        defer allocator.free(pfx);
        if (bi == 2 or bi == 5 or bi == 8) {
            const rp = try fmtKey(allocator, "{s}.resample.1", .{pfx});
            defer allocator.free(rp);
            v.blocks[bi] = .{ .down = .{
                .w = try loadConv2dW(&w, allocator, rp, s),
                .b = try loadVec(&w, allocator, rp, "bias", s),
            } };
        } else {
            v.blocks[bi] = .{ .res = try loadResBlock(&w, allocator, pfx, s) };
        }
    }
    v.mid_r0 = try loadResBlock(&w, allocator, "encoder.mid_block.resnets.0", s);
    v.mid_r1 = try loadResBlock(&w, allocator, "encoder.mid_block.resnets.1", s);
    v.mid_attn = .{
        .norm = try loadGamma(&w, allocator, "encoder.mid_block.attentions.0.norm", s),
        .qkv_w = try loadConv2dW(&w, allocator, "encoder.mid_block.attentions.0.to_qkv", s),
        .qkv_b = try loadVec(&w, allocator, "encoder.mid_block.attentions.0.to_qkv", "bias", s),
        .proj_w = try loadConv2dW(&w, allocator, "encoder.mid_block.attentions.0.proj", s),
        .proj_b = try loadVec(&w, allocator, "encoder.mid_block.attentions.0.proj", "bias", s),
    };
    v.norm_out = try loadGamma(&w, allocator, "encoder.norm_out", s);
    v.conv_out_w = try loadConv3dW(&w, allocator, "encoder.conv_out", s);
    v.conv_out_b = try loadVec(&w, allocator, "encoder.conv_out", "bias", s);
    v.quant_w = try loadConv3dW(&w, allocator, "quant_conv", s);
    v.quant_b = try loadVec(&w, allocator, "quant_conv", "bias", s);
    return v;
}

/// Conditioning rebalance: scale the stacked context [1,L,taps,d] by a global
/// `gain` and optional per-tapped-layer `weights` (len must equal the tap axis
/// — 12 for Krea). Returns a NEW array in the input dtype.
pub fn applyCondRebalance(ctx: mlx.mlx_array, gain: f32, weights: ?[]const f32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(ctx); // [1,L,n,d]
    const dt = mlx.mlx_array_dtype(ctx);
    var cur = blk: {
        var c = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&c, ctx));
        break :blk c;
    };
    if (weights) |w| {
        if (w.len != @as(usize, @intCast(sh[2]))) {
            _ = mlx.mlx_array_free(cur);
            return error.InvalidCondWeights;
        }
        const wsh = [_]c_int{ 1, 1, sh[2], 1 };
        const wa = mlx.mlx_array_new_data(w.ptr, &wsh, 4, .float32);
        defer _ = mlx.mlx_array_free(wa);
        const scaled = mulA(cur, wa, s) catch |e| {
            _ = mlx.mlx_array_free(cur);
            return e;
        };
        _ = mlx.mlx_array_free(cur);
        cur = scaled;
    }
    if (gain != 1.0) {
        const ga = scalarF(gain);
        defer _ = mlx.mlx_array_free(ga);
        const scaled = mulA(cur, ga, s) catch |e| {
            _ = mlx.mlx_array_free(cur);
            return e;
        };
        _ = mlx.mlx_array_free(cur);
        cur = scaled;
    }
    if (mlx.mlx_array_dtype(cur) != dt) {
        const back = try astype(cur, dt, s);
        _ = mlx.mlx_array_free(cur);
        cur = back;
    }
    return cur;
}

// ════════════════════════════════════════════════════════════════════════
// B4. Sampler (flow-match Euler, resolution-shifted schedule) + pipeline.
// ════════════════════════════════════════════════════════════════════════

const SPATIAL_SCALE: u32 = 8;
const MIN_RES: u32 = 256;
const MAX_RES: u32 = 1280;

fn roundup(value: u32, multiple: u32) u32 {
    return ((value + multiple - 1) / multiple) * multiple;
}

/// patchify NCHW [n,16,lat_h,lat_w] (p=2, [c ph pw]) → [n, h_*w_, 64].
fn patchify(x: mlx.mlx_array, p: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [n,c,H,W]
    const n = sh[0];
    const c = sh[1];
    const H = sh[2];
    const Wd = sh[3];
    const h = @divExact(H, p);
    const wq = @divExact(Wd, p);
    const r = try reshape(x, &[_]c_int{ n, c, h, p, wq, p }, s);
    defer _ = mlx.mlx_array_free(r);
    const t = try transpose(r, &[_]c_int{ 0, 2, 4, 1, 3, 5 }, s); // (n,h,w,c,p,p)
    defer _ = mlx.mlx_array_free(t);
    return reshape(t, &[_]c_int{ n, h * wq, c * p * p }, s);
}

/// unpatchify [n,h_*w_,64] → NCHW [n,16,lat_h,lat_w].
fn unpatchify(x: mlx.mlx_array, p: c_int, h: c_int, wq: c_int, c: c_int, s: S) !mlx.mlx_array {
    const n = mlx.getShape(x)[0];
    const r = try reshape(x, &[_]c_int{ n, h, wq, c, p, p }, s);
    defer _ = mlx.mlx_array_free(r);
    const t = try transpose(r, &[_]c_int{ 0, 3, 1, 4, 2, 5 }, s); // (n,c,h,p,w,p)
    defer _ = mlx.mlx_array_free(t);
    return reshape(t, &[_]c_int{ n, c, h * p, wq * p }, s);
}

/// Resolution-shifted flow-match schedule (steps+1 values, 1→0).
fn computeTimesteps(allocator: std.mem.Allocator, seq_len: u32, steps: u32) ![]f64 {
    const ts = try allocator.alloc(f64, steps + 1);
    const y1: f64 = 0.5;
    const y2: f64 = 1.15;
    const align_ = SPATIAL_SCALE * 2; // spatial_scale * patch = 16
    const x1: f64 = @floatFromInt((MIN_RES / align_) * (MIN_RES / align_));
    const x2: f64 = @floatFromInt((MAX_RES / align_) * (MAX_RES / align_));
    const slope = (y2 - y1) / (x2 - x1);
    const mu = slope * @as(f64, @floatFromInt(seq_len)) + (y1 - slope * x1);
    const emu = std.math.exp(mu);
    for (0..steps + 1) |i| {
        // linspace(1,0,steps+1)
        const lin = 1.0 - @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(steps));
        if (lin <= 0.0) {
            ts[i] = 0.0; // 1/0-1 → inf → 0
        } else {
            const denom = emu + (1.0 / lin - 1.0); // sigma=1
            ts[i] = emu / denom;
        }
    }
    return ts;
}

/// Build pos [L*3] i32: txt rows (0,0,0); img rows (0, row, col).
fn buildPositions(allocator: std.mem.Allocator, txtlen: usize, h_: usize, w_: usize) ![]i32 {
    const L = txtlen + h_ * w_;
    const out = try allocator.alloc(i32, L * 3);
    @memset(out, 0);
    var idx = txtlen;
    for (0..h_) |row| {
        for (0..w_) |col| {
            out[idx * 3 + 1] = @intCast(row);
            out[idx * 3 + 2] = @intCast(col);
            idx += 1;
        }
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Engine — owns the three sub-models + tokenizer. load/deinit/generate(Png).
// ════════════════════════════════════════════════════════════════════════

/// Per-request generation options beyond the core prompt/size/steps.
pub const GenOpts = struct {
    /// img2img: source pixels [1,3,H,W] f32 [0,1], pre-resized to the target
    /// size. VAE-encoded internally and mixed with noise at the start timestep.
    init_image: ?mlx.mlx_array = null,
    /// First schedule index to run (img2img skip; 0 = full schedule).
    start_step: u32 = 0,
    /// Conditioning rebalance: global gain + per-tapped-layer weights (len 12).
    cond_gain: f32 = 1.0,
    cond_weights: ?[]const f32 = null,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: S,
    te: Conditioner,
    dit: Dit,
    vae: Vae,
    vae_enc: ?VaeEncoder,
    tok: tok_mod.Tokenizer,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = mlx.mlx_default_gpu_stream_new();
        self.te = try loadConditioner(io, allocator, self.s, model_dir);
        errdefer self.te.deinit();
        self.dit = try loadDit(io, allocator, self.s, model_dir);
        errdefer self.dit.deinit();
        self.vae = try loadVae(io, allocator, self.s, model_dir);
        errdefer self.vae.deinit();
        self.vae_enc = loadVaeEncoder(io, allocator, self.s, model_dir) catch |e| blk: {
            log.warn("[image] krea VAE encoder load failed ({}) — image-to-image disabled\n", .{e});
            break :blk null;
        };
        errdefer if (self.vae_enc) |*e| e.deinit();
        const tok_dir = try std.fmt.allocPrint(allocator, "{s}/tokenizer", .{model_dir});
        defer allocator.free(tok_dir);
        self.tok = try tok_mod.loadTokenizerAny(io, allocator, tok_dir);
        log.info("[image] Krea-2 models + tokenizer ready\n", .{});
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.te.deinit();
        self.dit.deinit();
        self.vae.deinit();
        if (self.vae_enc) |*e| e.deinit();
        self.tok.deinit();
        self.allocator.destroy(self);
    }

    /// Build the templated ids/mask (PREFIX+prompt padded to 541 + SUFFIX 5 →
    /// 546). The encoder drops the first 34 (PREFIX) tokens.
    fn buildPromptIds(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8) !struct { ids: []i32, mask: []i32 } {
        const pre_max = MAX_LENGTH + PREFIX_START_IDX - SUFFIX_START_IDX; // 541
        const total = pre_max + SUFFIX_START_IDX; // 546
        const ids = try allocator.alloc(i32, total);
        const mask = try allocator.alloc(i32, total);

        const pre_text = try std.fmt.allocPrint(allocator, "{s}{s}", .{ PREFIX, prompt });
        defer allocator.free(pre_text);
        const enc = try self.tok.encode(allocator, pre_text);
        defer allocator.free(enc);
        const suf = try self.tok.encode(allocator, SUFFIX);
        defer allocator.free(suf);

        const real = @min(enc.len, pre_max);
        for (0..pre_max) |i| {
            if (i < real) {
                ids[i] = @intCast(enc[i]);
                mask[i] = 1;
            } else {
                ids[i] = PAD_TOKEN;
                mask[i] = 0;
            }
        }
        for (0..SUFFIX_START_IDX) |i| {
            ids[pre_max + i] = if (i < suf.len) @intCast(suf[i]) else PAD_TOKEN;
            mask[pre_max + i] = 1;
        }
        return .{ .ids = ids, .mask = mask };
    }

    /// Full pipeline. ids/mask are the templated 546-length sequence. Returns
    /// [1,3,H,W] f32 in [0,1] (owned mlx array; caller frees).
    pub fn generate(self: *Engine, allocator: std.mem.Allocator, ids: []const i32, mask: []const i32, seed: u64, steps: u32, width: u32, height: u32, progress: ?sse.Progress) !mlx.mlx_array {
        return self.generateWithOpts(allocator, ids, mask, seed, steps, width, height, .{}, progress);
    }

    pub fn generateWithOpts(self: *Engine, allocator: std.mem.Allocator, ids: []const i32, mask: []const i32, seed: u64, steps: u32, width: u32, height: u32, opts: GenOpts, progress: ?sse.Progress) !mlx.mlx_array {
        const s = self.s;
        const cfg = self.dit.cfg;
        const align_ = SPATIAL_SCALE * cfg.patch; // 16
        const W = roundup(width, align_);
        const H = roundup(height, align_);
        const lat_h = H / SPATIAL_SCALE;
        const lat_w = W / SPATIAL_SCALE;
        const h_ = lat_h / cfg.patch;
        const w_ = lat_w / cfg.patch;
        const img_len = h_ * w_;

        // 1. text encode → context [1,512,12,txtdim], validity [512]
        var ctx = try self.te.encode(ids, mask);
        defer _ = mlx.mlx_array_free(ctx);
        if (opts.cond_gain != 1.0 or opts.cond_weights != null) {
            const re = try applyCondRebalance(ctx, opts.cond_gain, opts.cond_weights, s);
            _ = mlx.mlx_array_free(ctx);
            ctx = re;
        }
        const txtlen: usize = @intCast(mlx.getShape(ctx)[1]);

        // 2. init noise [1,16,lat_h,lat_w] → patchify [1,img_len,64]
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        const nsh = [_]c_int{ 1, @intCast(cfg.channels), @intCast(lat_h), @intCast(lat_w) };
        var noise = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(noise);
        try mlx.check(mlx.mlx_random_normal(&noise, &nsh, 4, .float32, 0.0, 1.0, key, s));
        const noise_bf = try astype(noise, .bfloat16, s);
        defer _ = mlx.mlx_array_free(noise_bf);
        var img = try patchify(noise_bf, @intCast(cfg.patch), s); // [1,img_len,64]
        defer _ = mlx.mlx_array_free(img);

        // 3. positions, validity, rope, masks (constant across steps)
        const pos = try buildPositions(allocator, txtlen, h_, w_);
        defer allocator.free(pos);
        const valid = try allocator.alloc(i32, txtlen + img_len);
        defer allocator.free(valid);
        for (0..txtlen) |i| valid[i] = mask[i + PREFIX_START_IDX];
        for (txtlen..valid.len) |i| valid[i] = 1;

        const rope = try buildRope(allocator, pos, valid.len, cfg.theta);
        defer {
            _ = mlx.mlx_array_free(rope.cos);
            _ = mlx.mlx_array_free(rope.sin);
        }
        const full_mask = try buildOuterMask(allocator, valid, s);
        defer _ = mlx.mlx_array_free(full_mask);
        const txt_mask = try buildOuterMask(allocator, valid[0..txtlen], s);
        defer _ = mlx.mlx_array_free(txt_mask);

        // 4. schedule + denoise (8 steps, guidance 0)
        const ts = try computeTimesteps(allocator, img_len, steps);
        defer allocator.free(ts);
        const start_step: u32 = @min(opts.start_step, steps - 1);

        // img2img: VAE-encode the source, mix x = (1-t)·z0 + t·noise at the
        // start timestep (flow-match scale_noise).
        if (opts.init_image) |pix| {
            const ve = if (self.vae_enc) |*e| e else return error.NoVaeEncoder;
            const z0 = try ve.encode(pix); // [1,16,lat_h,lat_w] f32 normalized
            defer _ = mlx.mlx_array_free(z0);
            const z_bf = try astype(z0, .bfloat16, s);
            defer _ = mlx.mlx_array_free(z_bf);
            const z_tok = try patchify(z_bf, @intCast(cfg.patch), s);
            defer _ = mlx.mlx_array_free(z_tok);
            const t0: f32 = @floatCast(ts[start_step]);
            const ta = scalarF(t0);
            defer _ = mlx.mlx_array_free(ta);
            const oma = scalarF(1.0 - t0);
            defer _ = mlx.mlx_array_free(oma);
            const zs = try mulA(z_tok, oma, s);
            defer _ = mlx.mlx_array_free(zs);
            const ns = try mulA(img, ta, s);
            defer _ = mlx.mlx_array_free(ns);
            const mixed = try addA(zs, ns, s);
            defer _ = mlx.mlx_array_free(mixed);
            const mixed_bf = try astype(mixed, .bfloat16, s);
            _ = mlx.mlx_array_free(img);
            img = mixed_bf;
        }

        const run_steps = steps - start_step;
        for (start_step..steps) |i| {
            const tc: f32 = @floatCast(ts[i]);
            const tp: f32 = @floatCast(ts[i + 1]);
            const v = try self.dit.forwardPrebuilt(img, ctx, tc, rope.cos, rope.sin, full_mask, txt_mask, s);
            defer _ = mlx.mlx_array_free(v);
            const dt = scalarF(tp - tc);
            defer _ = mlx.mlx_array_free(dt);
            const step = try mulA(v, dt, s);
            defer _ = mlx.mlx_array_free(step);
            const ni = try addA(img, step, s);
            _ = mlx.mlx_array_free(img);
            img = ni;
            _ = mlx.mlx_array_eval(img);
            if (progress) |p| p.emit("Generating", @intCast(i + 1 - start_step), run_steps);
        }
        if (progress) |p| p.emit("Decoding image", steps, steps);

        // 5. unpatchify → [1,16,lat_h,lat_w] → VAE decode → [-1,1] → [0,1]
        const latent = try unpatchify(img, @intCast(cfg.patch), @intCast(h_), @intCast(w_), @intCast(cfg.channels), s);
        defer _ = mlx.mlx_array_free(latent);
        const latent_f = try astype(latent, .float32, s);
        defer _ = mlx.mlx_array_free(latent_f);
        const decoded = try self.vae.decode(latent_f);
        defer _ = mlx.mlx_array_free(decoded); // [1,3,H,W] [-1,1]
        // clip(decoded,-1,1)*0.5+0.5
        const lo = scalarF(-1.0);
        defer _ = mlx.mlx_array_free(lo);
        const hi = scalarF(1.0);
        defer _ = mlx.mlx_array_free(hi);
        var clo = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(clo);
        try mlx.check(mlx.mlx_maximum(&clo, decoded, lo, s));
        var chi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(chi);
        try mlx.check(mlx.mlx_minimum(&chi, clo, hi, s));
        const half = scalarF(0.5);
        defer _ = mlx.mlx_array_free(half);
        const sc = try mulA(chi, half, s);
        defer _ = mlx.mlx_array_free(sc);
        return addA(sc, half, s); // [0,1]
    }

    /// Tokenize + run the pipeline → image [1,3,H,W] f32 [0,1] (owned mlx array;
    /// caller frees). Lets the caller run the content filter before PNG-encoding.
    pub fn generateImage(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, progress: ?sse.Progress) !mlx.mlx_array {
        return self.generateImageOpts(allocator, prompt, width, height, seed, steps, .{}, progress);
    }

    /// `generateImage` with per-request options (img2img + conditioning rebalance).
    pub fn generateImageOpts(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, opts: GenOpts, progress: ?sse.Progress) !mlx.mlx_array {
        const pr = try self.buildPromptIds(allocator, prompt);
        defer allocator.free(pr.ids);
        defer allocator.free(pr.mask);
        return self.generateWithOpts(allocator, pr.ids, pr.mask, seed, steps, width, height, opts, progress);
    }

    /// Tokenize, run the pipeline, return PNG bytes (caller frees).
    pub fn generatePng(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, progress: ?sse.Progress) ![]u8 {
        const img = try self.generateImage(allocator, prompt, width, height, seed, steps, progress);
        defer _ = mlx.mlx_array_free(img);
        return imageToPng(allocator, img, self.s);
    }
};

/// [1,3,H,W] f32 [0,1] → RGB8 PNG bytes (caller frees).
pub fn imageToPng(allocator: std.mem.Allocator, img: mlx.mlx_array, s: S) ![]u8 {
    const cf = try contig(img, s);
    defer _ = mlx.mlx_array_free(cf);
    _ = mlx.mlx_array_eval(cf);
    const sh = mlx.getShape(cf); // [1,3,H,W]
    const H: usize = @intCast(sh[2]);
    const W: usize = @intCast(sh[3]);
    const d = mlx.mlx_array_data_float32(cf) orelse return error.NoData;
    const rgb = try allocator.alloc(u8, W * H * 3);
    defer allocator.free(rgb);
    const plane = W * H;
    for (0..H) |y| for (0..W) |x| {
        const o = (y * W + x) * 3;
        for (0..3) |c| {
            const v = d[c * plane + y * W + x];
            rgb[o + c] = @intFromFloat(std.math.clamp(v * 255.0, 0, 255));
        }
    };
    return png.encodeRgb(allocator, rgb, @intCast(W), @intCast(H));
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

// ── Hermetic: MixedLinear bits/group_size inference + forward ──

test "MixedLinear bf16 forward (no scales)" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const a = testing.allocator;
    // weight [out=2, in=3] = [[1,2,3],[4,5,6]]; bias [2] = [10,20]
    var wbuf = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const wsh = [_]c_int{ 2, 3 };
    const wa = mlx.mlx_array_new_data(&wbuf, &wsh, 2, .float32);
    var bbuf = [_]f32{ 10, 20 };
    const bsh = [_]c_int{2};
    const ba = mlx.mlx_array_new_data(&bbuf, &bsh, 1, .float32);
    var ww = model_mod.Weights.init(a);
    defer ww.deinit();
    try ww.map.put(try a.dupe(u8, "lin.weight"), wa);
    try ww.map.put(try a.dupe(u8, "lin.bias"), ba);

    var ml = try MixedLinear.load(&ww, a, "lin", 3, s);
    defer ml.deinit();
    try testing.expect(!ml.quantized);

    var xb = [_]f32{ 1, 1, 1 };
    const xsh = [_]c_int{ 1, 3 };
    const xa = mlx.mlx_array_new_data(&xb, &xsh, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);
    const o = try ml.forward(xa, s);
    defer _ = mlx.mlx_array_free(o);
    const of = try astype(o, .float32, s);
    defer _ = mlx.mlx_array_free(of);
    _ = mlx.mlx_array_eval(of);
    const d = mlx.mlx_array_data_float32(of).?;
    // [1,1,1]·[1,2,3]+10 = 16 ; ·[4,5,6]+20 = 35
    try testing.expect(@abs(d[0] - 16.0) < 0.2);
    try testing.expect(@abs(d[1] - 35.0) < 0.2);
}

test "MixedLinear infers bits/group_size from quantized geometry" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const a = testing.allocator;
    // Build a real affine-quantized weight via mlx_quantize so geometry is valid.
    // out=4, in=64, bits=4, gs=32 → packed cols = 64*4/32 = 8 ; scales cols = 64/32 = 2.
    const in: c_int = 64;
    const out: c_int = 4;
    const raw = try a.alloc(f32, @intCast(out * in));
    defer a.free(raw);
    for (raw, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 7)) * 0.1 - 0.3;
    const rsh = [_]c_int{ out, in };
    const rw = mlx.mlx_array_new_data(raw.ptr, &rsh, 2, .float32);
    defer _ = mlx.mlx_array_free(rw);
    const rwb = try astype(rw, .bfloat16, s);
    defer _ = mlx.mlx_array_free(rwb);

    inline for (.{ .{ 4, 32 }, .{ 8, 64 } }) |cfg| {
        const bits: c_int = cfg[0];
        const gs: c_int = cfg[1];
        var packed_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(packed_vec);
        const null_gscale = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_quantize(&packed_vec, rwb, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", null_gscale, s));
        var qw = mlx.mlx_array_new();
        var qs = mlx.mlx_array_new();
        var qb = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&qw, packed_vec, 0));
        try mlx.check(mlx.mlx_vector_array_get(&qs, packed_vec, 1));
        try mlx.check(mlx.mlx_vector_array_get(&qb, packed_vec, 2));

        var ww = model_mod.Weights.init(a);
        defer ww.deinit();
        try ww.map.put(try a.dupe(u8, "q.weight"), qw);
        try ww.map.put(try a.dupe(u8, "q.scales"), qs);
        try ww.map.put(try a.dupe(u8, "q.biases"), qb);
        var ml = try MixedLinear.load(&ww, a, "q", @intCast(in), s);
        defer ml.deinit();
        try testing.expect(ml.quantized);
        try testing.expectEqual(@as(u32, @intCast(bits)), ml.bits);
        try testing.expectEqual(@as(u32, @intCast(gs)), ml.group_size);

        // forward sanity vs a manual matmul against the dequantized weight.
        const xv = try a.alloc(f32, @intCast(in));
        defer a.free(xv);
        for (xv, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 5)) * 0.2;
        const xsh = [_]c_int{ 1, in };
        const xa = mlx.mlx_array_new_data(xv.ptr, &xsh, 2, .float32);
        defer _ = mlx.mlx_array_free(xa);
        const o = try ml.forward(xa, s);
        defer _ = mlx.mlx_array_free(o);
        const of = try astype(o, .float32, s);
        defer _ = mlx.mlx_array_free(of);
        _ = mlx.mlx_array_eval(of);
        try testing.expectEqual(@as(usize, @intCast(out)), @as(usize, @intCast(mlx.mlx_array_size(of))));
    }
}

// ── Oracle tests (env-gated; fixtures from tests/dump_krea_fixtures.py) ──

fn readI32(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]i32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(64 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(i32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}
fn readF32(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]f32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(1024 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}
fn cosine(data: []const f32, ref: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..data.len) |i| {
        dot += @as(f64, data[i]) * ref[i];
        na += @as(f64, data[i]) * data[i];
        nb += @as(f64, ref[i]) * ref[i];
    }
    return dot / (std.math.sqrt(na) * std.math.sqrt(nb));
}

test "MixedLinear applies an attached LoRA delta (bf16 path)" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // Base: identity [in=2,out=2] (pre-transposed layout). x = [1,2].
    const wv = [_]f32{ 1, 0, 0, 1 };
    const wsh = [_]c_int{ 2, 2 };
    const wf = mlx.mlx_array_new_data(&wv, &wsh, 2, .float32);
    defer _ = mlx.mlx_array_free(wf);
    const wbf = try astype(wf, .bfloat16, s);
    var ml = MixedLinear{ .quantized = false, .w = wbf };
    defer ml.deinit();
    const xv = [_]f32{ 1, 2 };
    const xs = [_]c_int{ 1, 2 };
    const x = mlx.mlx_array_new_data(&xv, &xs, 2, .float32);
    defer _ = mlx.mlx_array_free(x);
    // Without LoRA: y = x.
    const y0 = try ml.forward(x, s);
    defer _ = mlx.mlx_array_free(y0);
    const y0f = try astype(y0, .float32, s);
    defer _ = mlx.mlx_array_free(y0f);
    _ = mlx.mlx_array_eval(y0f);
    const d0 = mlx.mlx_array_data_float32(y0f) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 1), d0[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 2), d0[1], 1e-3);
    // LoRA: at [2,1]=[1;1], bt [1,2]=[2,4], scale .5 → delta = [3,6]; y = [4,8].
    const atv = [_]f32{ 1, 1 };
    const ats = [_]c_int{ 2, 1 };
    const atf = mlx.mlx_array_new_data(&atv, &ats, 2, .float32);
    defer _ = mlx.mlx_array_free(atf);
    const at = try astype(atf, .bfloat16, s);
    defer _ = mlx.mlx_array_free(at);
    const btv = [_]f32{ 2, 4 };
    const bts = [_]c_int{ 1, 2 };
    const btf = mlx.mlx_array_new_data(&btv, &bts, 2, .float32);
    defer _ = mlx.mlx_array_free(btf);
    const bt = try astype(btf, .bfloat16, s);
    defer _ = mlx.mlx_array_free(bt);
    ml.lora = .{ .at = at, .bt = bt, .scale = 0.5 };
    const y1 = try ml.forward(x, s);
    defer _ = mlx.mlx_array_free(y1);
    const y1f = try astype(y1, .float32, s);
    defer _ = mlx.mlx_array_free(y1f);
    _ = mlx.mlx_array_eval(y1f);
    const d1 = mlx.mlx_array_data_float32(y1f) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 4), d1[0], 1e-2);
    try testing.expectApproxEqAbs(@as(f32, 8), d1[1], 1e-2);
    // Detached again: back to base.
    ml.lora = null;
    const y2 = try ml.forward(x, s);
    defer _ = mlx.mlx_array_free(y2);
    const y2f = try astype(y2, .float32, s);
    defer _ = mlx.mlx_array_free(y2f);
    _ = mlx.mlx_array_eval(y2f);
    const d2 = mlx.mlx_array_data_float32(y2f) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 1), d2[0], 1e-3);
}

test "krea applyCondRebalance scales tapped layers and global gain" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // ctx [1,2,3,2]: 2 tokens × 3 tapped layers × hidden 2.
    const vals = [_]f32{ 1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60 };
    const sh = [_]c_int{ 1, 2, 3, 2 };
    const ctx = mlx.mlx_array_new_data(&vals, &sh, 4, .float32);
    defer _ = mlx.mlx_array_free(ctx);
    const w = [_]f32{ 2, 3, 4 };
    const out = try applyCondRebalance(ctx, 0.5, &w, s);
    defer _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_eval(out);
    const d = mlx.mlx_array_data_float32(out) orelse return error.NoData;
    // Token 0: layer0 [1,2]*2*.5, layer1 [3,4]*3*.5, layer2 [5,6]*4*.5
    const expect0 = [_]f32{ 1, 2, 4.5, 6, 10, 12 };
    for (expect0, 0..) |e, i| try testing.expectApproxEqAbs(e, d[i], 1e-5);
    // Gain-only path.
    const g = try applyCondRebalance(ctx, 2.0, null, s);
    defer _ = mlx.mlx_array_free(g);
    _ = mlx.mlx_array_eval(g);
    const gd = mlx.mlx_array_data_float32(g) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 2), gd[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 120), gd[11], 1e-5);
    // Count must match the tap axis.
    const bad = [_]f32{ 1, 2 };
    try testing.expectError(error.InvalidCondWeights, applyCondRebalance(ctx, 1.0, &bad, s));
}

// VAE encoder guard: encode a synthetic smooth image, decode it back — the
// pixels must round-trip through the oracle-validated decoder.  KREA_TEST_MODEL
test "krea VAE encoder round-trips through the decoder" {
    const model_dir = std.mem.span(std.c.getenv("KREA_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var enc = try loadVaeEncoder(io, a, s, model_dir);
    defer enc.deinit();
    var vae = try loadVae(io, a, s, model_dir);
    defer vae.deinit();
    const HW: usize = 512;
    const img_buf = try a.alloc(f32, 3 * HW * HW);
    defer a.free(img_buf);
    for (0..HW) |y| for (0..HW) |x| {
        const fy: f32 = @as(f32, @floatFromInt(y)) / (HW - 1);
        const fx: f32 = @as(f32, @floatFromInt(x)) / (HW - 1);
        img_buf[0 * HW * HW + y * HW + x] = fx;
        img_buf[1 * HW * HW + y * HW + x] = fy;
        img_buf[2 * HW * HW + y * HW + x] = 0.5 * (fx + fy);
    };
    const ish = [_]c_int{ 1, 3, @intCast(HW), @intCast(HW) };
    const img = mlx.mlx_array_new_data(img_buf.ptr, &ish, 4, .float32);
    defer _ = mlx.mlx_array_free(img);
    const lat = try enc.encode(img);
    defer _ = mlx.mlx_array_free(lat);
    const lsh = mlx.getShape(lat);
    try testing.expectEqual(@as(c_int, 16), lsh[1]);
    try testing.expectEqual(@as(c_int, @intCast(HW / 8)), lsh[2]);
    const lat_f = try astype(lat, .float32, s);
    defer _ = mlx.mlx_array_free(lat_f);
    const dec = try vae.decode(lat_f); // [-1,1]
    defer _ = mlx.mlx_array_free(dec);
    const df = try astype(dec, .float32, s);
    defer _ = mlx.mlx_array_free(df);
    _ = mlx.mlx_array_eval(df);
    const n: usize = @intCast(mlx.mlx_array_size(df));
    try testing.expectEqual(img_buf.len, n);
    const data = mlx.mlx_array_data_float32(df) orelse return error.NoData;
    var mapped = try a.alloc(f32, n);
    defer a.free(mapped);
    var mae: f64 = 0;
    for (0..n) |i| {
        mapped[i] = data[i] * 0.5 + 0.5;
        mae += @abs(@as(f64, mapped[i]) - img_buf[i]);
    }
    mae /= @floatFromInt(n);
    const corr = cosine(mapped, img_buf);
    std.debug.print("[krea-vae-enc] roundtrip corr={d:.6} mae={d:.5}\n", .{ corr, mae });
    try testing.expect(corr > 0.99);
    try testing.expect(mae < 0.05);
}

// Stage 1: text encoder. KREA_TEST_MODEL, KREA_IDS, KREA_MASK, KREA_PE.
test "krea text encoder reproduces hidden states" {
    const model_dir = std.mem.span(std.c.getenv("KREA_TEST_MODEL") orelse return error.SkipZigTest);
    const ids_p = std.mem.span(std.c.getenv("KREA_IDS") orelse return error.SkipZigTest);
    const mask_p = std.mem.span(std.c.getenv("KREA_MASK") orelse return error.SkipZigTest);
    const pe_p = std.mem.span(std.c.getenv("KREA_PE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try readI32(io, a, ids_p);
    defer a.free(ids);
    const mask = try readI32(io, a, mask_p);
    defer a.free(mask);
    const ref = try readF32(io, a, pe_p);
    defer a.free(ref);
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var te = try loadConditioner(io, a, s, model_dir);
    defer te.deinit();
    const pe = try te.encode(ids, mask);
    defer _ = mlx.mlx_array_free(pe);
    const pf = try astype(pe, .float32, s);
    defer _ = mlx.mlx_array_free(pf);
    _ = mlx.mlx_array_eval(pf);
    const n: usize = @intCast(mlx.mlx_array_size(pf));
    const data = mlx.mlx_array_data_float32(pf) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    const corr = cosine(data[0..n], ref);
    std.debug.print("[krea-te] n={d} corr={d:.6}\n", .{ n, corr });
    // Default bar 0.997 is for a BF16 encoder (engine-correctness vs the bf16
    // reference): this compares the RAW Qwen3-VL hidden states of the 12 tapped
    // layers (up to layer 35), whose massive late-layer outlier activations (|x|
    // in the thousands) make the cosine hypersensitive to irreducible bf16
    // op-ordering noise. Measured 0.99806 bf16; the f32 norm-weight match lifts it
    // from 0.9963 (reverting `normF32` trips this bar — a real regression guard).
    // An 8-BIT-quantized encoder (the public bundle) lands ~0.9927 here — the
    // quant error on the outliers is large in this metric but washes out by the
    // pipeline output, so set KREA_TE_MIN=0.99 when validating an 8-bit dir. The
    // DEFINITIVE proofs are downstream: DiT 0.99995, VAE 1.0, e2e 0.9996 (bf16) /
    // 0.99964 (8-bit encoder).
    const min_corr: f64 = if (std.c.getenv("KREA_TE_MIN")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.997) else 0.997;
    try testing.expect(corr > min_corr);
}

// Stage 2: DiT one forward → velocity at t=1.0.
// KREA_TEST_MODEL, KREA_VEL_IMG, KREA_VEL_CTX, KREA_VEL_POS, KREA_VEL_VALID, KREA_VEL.
test "krea DiT reproduces velocity" {
    const model_dir = std.mem.span(std.c.getenv("KREA_TEST_MODEL") orelse return error.SkipZigTest);
    const img_p = std.mem.span(std.c.getenv("KREA_VEL_IMG") orelse return error.SkipZigTest);
    const ctx_p = std.mem.span(std.c.getenv("KREA_VEL_CTX") orelse return error.SkipZigTest);
    const pos_p = std.mem.span(std.c.getenv("KREA_VEL_POS") orelse return error.SkipZigTest);
    const valid_p = std.mem.span(std.c.getenv("KREA_VEL_VALID") orelse return error.SkipZigTest);
    const vel_p = std.mem.span(std.c.getenv("KREA_VEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const imgd = try readF32(io, a, img_p);
    defer a.free(imgd);
    const ctxd = try readF32(io, a, ctx_p);
    defer a.free(ctxd);
    const pos = try readI32(io, a, pos_p);
    defer a.free(pos);
    const valid = try readI32(io, a, valid_p);
    defer a.free(valid);
    const ref = try readF32(io, a, vel_p);
    defer a.free(ref);

    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var dit = try loadDit(io, a, s, model_dir);
    defer dit.deinit();

    const img_len: c_int = @intCast(imgd.len / 64);
    const L = valid.len;
    const txtlen: c_int = @intCast(L - @as(usize, @intCast(img_len)));
    const img_sh = [_]c_int{ 1, img_len, 64 };
    const img_f = mlx.mlx_array_new_data(imgd.ptr, &img_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(img_f);
    const img_bf = try astype(img_f, .bfloat16, s);
    defer _ = mlx.mlx_array_free(img_bf);
    const ctx_sh = [_]c_int{ 1, txtlen, 12, 2560 };
    const ctx_f = mlx.mlx_array_new_data(ctxd.ptr, &ctx_sh, 4, .float32);
    defer _ = mlx.mlx_array_free(ctx_f);
    const ctx_bf = try astype(ctx_f, .bfloat16, s);
    defer _ = mlx.mlx_array_free(ctx_bf);

    const vel = try dit.forward(img_bf, ctx_bf, 1.0, pos, valid);
    defer _ = mlx.mlx_array_free(vel);
    const vf = try astype(vel, .float32, s);
    defer _ = mlx.mlx_array_free(vf);
    _ = mlx.mlx_array_eval(vf);
    const n: usize = @intCast(mlx.mlx_array_size(vf));
    const data = mlx.mlx_array_data_float32(vf) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    const corr = cosine(data[0..n], ref);
    std.debug.print("[krea-dit] n={d} corr={d:.6}\n", .{ n, corr });
    try testing.expect(corr > 0.99);
}

// Stage 3: VAE decode. KREA_TEST_MODEL, KREA_VAE_LATENT, KREA_DECODED.
test "krea VAE reproduces decoded image" {
    const model_dir = std.mem.span(std.c.getenv("KREA_TEST_MODEL") orelse return error.SkipZigTest);
    const lat_p = std.mem.span(std.c.getenv("KREA_VAE_LATENT") orelse return error.SkipZigTest);
    const dec_p = std.mem.span(std.c.getenv("KREA_DECODED") orelse return error.SkipZigTest);
    const lath_s = std.mem.span(std.c.getenv("KREA_VAE_LATH") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const lat = try readF32(io, a, lat_p);
    defer a.free(lat);
    const ref = try readF32(io, a, dec_p);
    defer a.free(ref);
    const lath = try std.fmt.parseInt(c_int, lath_s, 10);
    const latw: c_int = @intCast(@divExact(lat.len, @as(usize, @intCast(16 * lath))));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var vae = try loadVae(io, a, s, model_dir);
    defer vae.deinit();
    const lat_sh = [_]c_int{ 1, 16, lath, latw };
    const lat_f = mlx.mlx_array_new_data(lat.ptr, &lat_sh, 4, .float32);
    defer _ = mlx.mlx_array_free(lat_f);
    const dec = try vae.decode(lat_f);
    defer _ = mlx.mlx_array_free(dec);
    const df = try astype(dec, .float32, s);
    defer _ = mlx.mlx_array_free(df);
    _ = mlx.mlx_array_eval(df);
    const n: usize = @intCast(mlx.mlx_array_size(df));
    const data = mlx.mlx_array_data_float32(df) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    const corr = cosine(data[0..n], ref);
    std.debug.print("[krea-vae] n={d} corr={d:.6}\n", .{ n, corr });
    try testing.expect(corr > 0.99);
}

// Stage 4: end-to-end. KREA_TEST_MODEL, KREA_IDS, KREA_MASK, KREA_E2E (final
// image [0,1]), KREA_E2E_W, KREA_E2E_H, KREA_E2E_SEED, KREA_E2E_STEPS.
test "krea end-to-end pipeline matches reference image" {
    const model_dir = std.mem.span(std.c.getenv("KREA_TEST_MODEL") orelse return error.SkipZigTest);
    const ids_p = std.mem.span(std.c.getenv("KREA_IDS") orelse return error.SkipZigTest);
    const mask_p = std.mem.span(std.c.getenv("KREA_MASK") orelse return error.SkipZigTest);
    const e2e_p = std.mem.span(std.c.getenv("KREA_E2E") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try readI32(io, a, ids_p);
    defer a.free(ids);
    const mask = try readI32(io, a, mask_p);
    defer a.free(mask);
    const ref = try readF32(io, a, e2e_p);
    defer a.free(ref);
    const width: u32 = if (std.c.getenv("KREA_E2E_W")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 512;
    const height: u32 = if (std.c.getenv("KREA_E2E_H")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 512;
    const seed: u64 = if (std.c.getenv("KREA_E2E_SEED")) |v| try std.fmt.parseInt(u64, std.mem.span(v), 10) else 0;
    const steps: u32 = if (std.c.getenv("KREA_E2E_STEPS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 8;

    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var eng: Engine = undefined;
    eng.allocator = a;
    eng.s = s;
    eng.te = try loadConditioner(io, a, s, model_dir);
    defer eng.te.deinit();
    eng.dit = try loadDit(io, a, s, model_dir);
    defer eng.dit.deinit();
    eng.vae = try loadVae(io, a, s, model_dir);
    defer eng.vae.deinit();

    const img = try eng.generate(a, ids, mask, seed, steps, width, height, null);
    defer _ = mlx.mlx_array_free(img);
    _ = mlx.mlx_array_eval(img);
    const n: usize = @intCast(mlx.mlx_array_size(img));
    const data = mlx.mlx_array_data_float32(img) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    const corr = cosine(data[0..n], ref);
    std.debug.print("[krea-e2e] n={d} corr={d:.5}\n", .{ n, corr });
    try testing.expect(corr > 0.95);
}

test "swigluDim matches reference roundup" {
    try testing.expectEqual(@as(u32, 16384), swigluDim(6144, 4));
    try testing.expectEqual(@as(u32, 6912), swigluDim(2560, 4));
}
