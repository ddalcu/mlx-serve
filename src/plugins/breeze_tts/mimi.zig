//! Mimi codec (Kyutai mimi: 16 codebooks at 12.5 Hz, 24 kHz output): the residual-VQ decoder Breeze
//! streams frames through, plus the encoder that folds a reference clip into frames for voice cloning.
//! Extracted from the shared tts engine so the breeze_tts plugin stays self-contained (plugins do not
//! import plugins); the low-level mlx helpers ride along.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const weights_mod = @import("../../sdk/weights/load.zig");
const log = @import("../../sdk/util/log.zig");

const model_mod = weights_mod;
const wav_mod = @import("../../sdk/media/wav.zig");
const transformer_mod = @import("../../sdk/blocks/mlp/fused_activations.zig");

const Weights = weights_mod.Weights;
const S = mlx.mlx_stream;
// ── Low-level mlx helpers ──


inline fn matmul(x: mlx.mlx_array, w_t: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&out, x, w_t, s));
    return out;
}

inline fn rms(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, s));
    return out;
}

inline fn addA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, a, b, s));
    return out;
}

inline fn mulA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, a, b, s));
    return out;
}

inline fn subA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&out, a, b, s));
    return out;
}

/// silu(x) = x * sigmoid(x). mlx-c has no silu; compose it.
inline fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    return mulA(x, sig, s);
}

/// SwiGLU: silu(gate) * up. Tries the shared fused kernel first (bit-exact —
/// its sigmoid is a full 16-bit table; declines f32 and odd shapes), keeping
/// the 3-dispatch chain as the fallback and the kill-switch path
/// (MLX_SERVE_SWIGLU_FUSED=0).
inline fn swiglu(gate: mlx.mlx_array, up: mlx.mlx_array, s: S) !mlx.mlx_array {
    if (try transformer_mod.fusedSwiGLU(s, gate, up)) |fused| return fused;
    return swigluPlain(gate, up, s);
}

/// The composed chain only — the form the compiled predictor chain traces
/// (custom `metal_kernel` ops stay OUT of an mlx_compile trace; compile's own
/// elementwise fusion replaces them there).
inline fn swigluPlain(gate: mlx.mlx_array, up: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sg = try silu(gate, s);
    defer _ = mlx.mlx_array_free(sg);
    return mulA(sg, up, s);
}

inline fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, x, shape.ptr, shape.len, s));
    return out;
}

inline fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&out, x, axes.ptr, axes.len, s));
    return out;
}

/// Concatenate a slice of arrays along `axis`.
fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, axis, s));
    return out;
}

/// Slice `x` [1, L, H] along axis 1 → [1, stop-start, H].
fn sliceSeq(x: mlx.mlx_array, start: c_int, stop: c_int, hidden: c_int, s: S) !mlx.mlx_array {
    const st = [_]c_int{ 0, start, 0 };
    const sp = [_]c_int{ 1, stop, hidden };
    const stride = [_]c_int{ 1, 1, 1 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &st, 3, &sp, 3, &stride, 3, s));
    return out;
}

/// Embedding lookup: table [vocab, H], ids (int32 slice) → [1, N, H]. The
/// output width is the table's NATIVE width, read from its shape — NOT the
/// `hidden` argument. The gather (`take_axis`) is always native-width, so a
/// caller passing a mismatched `hidden` (e.g. the talker `t_hidden`=1024 when
/// the embedding table is 2048-wide, as on the Qwen3-TTS 0.6B-Base checkpoint
/// whose `text_projection` then maps 2048→1024) used to crash the reshape.
/// `hidden` is now advisory only.
fn embed(table: mlx.mlx_array, ids: []const i32, hidden: u32, s: S) !mlx.mlx_array {
    _ = hidden;
    const id_shape = [_]c_int{@intCast(ids.len)};
    const id_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(id_arr);
    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, table, id_arr, 0, s));
    const tsh = mlx.getShape(table);
    const H: c_int = if (tsh.len > 0) tsh[tsh.len - 1] else 0;
    const out_shape = [_]c_int{ 1, @intCast(ids.len), H };
    return reshape(taken, &out_shape, s);
}

/// `embed` for a GPU-resident `[1]` i32 id (a lazily sampled token): the same
/// `take_axis` + native-width reshape, so a cached row is bit-identical to the
/// host-id path.
fn embedIdArr(table: mlx.mlx_array, id_arr: mlx.mlx_array, s: S) !mlx.mlx_array {
    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, table, id_arr, 0, s));
    const tsh = mlx.getShape(table);
    const H: c_int = if (tsh.len > 0) tsh[tsh.len - 1] else 0;
    const out_shape = [_]c_int{ 1, 1, H };
    return reshape(taken, &out_shape, s);
}

// ── GPU-chained code predictor (default ON) ──
fn linearBias(x: mlx.mlx_array, w_t: mlx.mlx_array, bias: mlx.mlx_array, s: S) !mlx.mlx_array {
    const mm = try matmul(x, w_t, s);
    defer _ = mlx.mlx_array_free(mm);
    return addA(mm, bias, s);
}

// ── Weight ownership helpers ──

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[tts] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingTtsWeight;
    };
    var owned = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&owned, arr));
    return owned;
}

fn ownOpt(w: *const Weights, key: []const u8) ?mlx.mlx_array {
    const a = w.get(key) orelse return null;
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_array_set(&o, a)) catch return null;
    return o;
}

/// Own + transpose a 2-D `[out, in]` weight to `[in, out]` for `matmul(x, w_t)`.
fn ownT(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    const perm = [_]c_int{ 1, 0 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&out, raw, &perm, 2, s));
    return out;
}

fn ownTfmt(w: *const Weights, s: S, alloc: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
    const key = try std.fmt.allocPrint(alloc, fmt, args);
    defer alloc.free(key);
    return ownT(w, key, s);
}
fn ownWfmt(w: *const Weights, alloc: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
    const key = try std.fmt.allocPrint(alloc, fmt, args);
    defer alloc.free(key);
    return ownWeight(w, key);
}

// ── Mixed dense/quantized linear (the krea.zig MixedLinear pattern) ──

/// Solve (bits, group_size) from packed-quantized tensor geometry (the
/// flux.zig convention): `w` packs `in` logical columns into u32s
/// (w_cols = in·bits/32) and scales have s_cols = in/gs, so
/// w_cols·32/s_cols = bits·gs. The product alone is ambiguous ((8,64) vs
/// (4,128)…), so group sizes are tried in convention order: mlx-community
/// conversions use 64 (the mlx default); 32 and 128 are fallbacks.
const QuantGeometry = struct { bits: u32, group_size: u32 };

fn inferQuantGeometry(w_cols: usize, s_cols: usize) QuantGeometry {
    const fallback: QuantGeometry = .{ .bits = 8, .group_size = 64 };
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

/// bf16 OR affine-quantized linear. If `<prefix>.scales` exists the module is
/// quantized: own the packed weight + scales + biases, infer (bits, group_size)
/// from geometry. Otherwise the dense branch is byte-identical to the original
/// bf16 path (pre-transposed weight + `mlx_matmul`, no extra casts). An
/// additive `<prefix>.bias` rides along in both branches (the 8-bit repos keep
/// `.bias` on text_projection/small_to_mtp even where `.biases` are the quant
/// zero-points).
const TtsLinear = struct {
    quantized: bool,
    w: mlx.mlx_array, // quantized: packed u32 [out, in*bits/32]; dense: pre-transposed [in, out]
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    add_bias: ?mlx.mlx_array = null,
    bits: u32 = 0,
    group_size: u32 = 0,

    fn load(w: *const Weights, alloc: std.mem.Allocator, prefix: []const u8, s: S) !TtsLinear {
        const wk = try std.fmt.allocPrint(alloc, "{s}.weight", .{prefix});
        defer alloc.free(wk);
        const sk = try std.fmt.allocPrint(alloc, "{s}.scales", .{prefix});
        defer alloc.free(sk);
        const bk = try std.fmt.allocPrint(alloc, "{s}.biases", .{prefix});
        defer alloc.free(bk);
        const ak = try std.fmt.allocPrint(alloc, "{s}.bias", .{prefix});
        defer alloc.free(ak);

        if (ownOpt(w, sk)) |scales| {
            const weight = try ownWeight(w, wk);
            const biases = try ownWeight(w, bk);
            const wsh = mlx.getShape(weight);
            const ssh = mlx.getShape(scales);
            const geo = inferQuantGeometry(@intCast(wsh[wsh.len - 1]), @intCast(ssh[ssh.len - 1]));
            return .{
                .quantized = true,
                .w = weight,
                .scales = scales,
                .biases = biases,
                .add_bias = ownOpt(w, ak),
                .bits = geo.bits,
                .group_size = geo.group_size,
            };
        }
        return .{ .quantized = false, .w = try ownT(w, wk, s), .add_bias = ownOpt(w, ak) };
    }

    fn deinit(self: *TtsLinear) void {
        _ = mlx.mlx_array_free(self.w);
        if (self.quantized) {
            _ = mlx.mlx_array_free(self.scales);
            _ = mlx.mlx_array_free(self.biases);
        }
        if (self.add_bias) |b| _ = mlx.mlx_array_free(b);
    }

    fn forward(self: *const TtsLinear, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var o = mlx.mlx_array_new();
        if (self.quantized) {
            try mlx.check(mlx.mlx_quantized_matmul(&o, x, self.w, self.scales, self.biases, true, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", s));
        } else {
            try mlx.check(mlx.mlx_matmul(&o, x, self.w, s));
        }
        if (self.add_bias) |b| {
            const r = try addA(o, b, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        return o;
    }
};

fn loadLinearFmt(w: *const Weights, alloc: std.mem.Allocator, s: S, comptime fmt: []const u8, args: anytype) !TtsLinear {
    const prefix = try std.fmt.allocPrint(alloc, fmt, args);
    defer alloc.free(prefix);
    return TtsLinear.load(w, alloc, prefix, s);
}

const QwenDims = struct {
    hidden: u32,
    heads: u32,
    kv: u32,
    head_dim: u32,
    eps: f32,
    theta: f32,
    /// 0-dim f32 scalar of `eps`, model-owned (null ⇒ the fused residual+norm
    /// path declines and the composed pair runs).
    eps_arr: mlx.mlx_array = .{ .ctx = null },
    /// True inside an mlx_compile trace: use only stock mlx ops (no custom
    /// metal kernels — compile's own fusion covers those chains).
    plain_kernels: bool = false,
};

pub const CodecConfig = struct {
    codebook_dim: u32 = 512,
    latent_dim: u32 = 1024,
    decoder_dim: u32 = 1536,
    hidden: u32 = 512, // pre_transformer hidden
    layers: u32 = 8,
    heads: u32 = 16,
    kv: u32 = 16, // no GQA in the decoder transformer
    head_dim: u32 = 64,
    inter: u32 = 1024,
    rms_eps: f32 = 1e-5,
    rope_theta: f32 = 10000.0,
    codebook_size: u32 = 2048,
    num_quant: u32 = 16,
    num_sem: u32 = 1,
    rvq_codebook_dim: u32 = 256, // codebook_dim // 2
    layer_scale: f32 = 0.01,
    upsample_rates: [4]u32 = .{ 8, 5, 4, 3 },
    upsampling_ratios: [2]u32 = .{ 2, 2 },
};

/// gelu(x) = 0.5*x*(1+erf(x/√2)) (exact, matches nn.gelu).
fn gelu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const inv_sqrt2 = mlx.mlx_array_new_float(0.7071067811865476);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, x, inv_sqrt2, s));
    var erf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(erf);
    try mlx.check(mlx.mlx_erf(&erf, scaled, s));
    var one_plus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one_plus);
    try mlx.check(mlx.mlx_add(&one_plus, erf, one, s));
    var hx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hx);
    try mlx.check(mlx.mlx_multiply(&hx, x, half, s));
    return mulA(hx, one_plus, s);
}

/// SnakeBeta(x) = x + (1/(exp(beta)+1e-9)) * sin(x*exp(alpha))^2. Channel-wise
/// alpha/beta broadcast over the NLC last dim.
fn snakeBeta(x: mlx.mlx_array, alpha: mlx.mlx_array, beta: mlx.mlx_array, s: S) !mlx.mlx_array {
    var ea = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ea);
    try mlx.check(mlx.mlx_exp(&ea, alpha, s));
    var eb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eb);
    try mlx.check(mlx.mlx_exp(&eb, beta, s));
    const eps = mlx.mlx_array_new_float(1e-9);
    defer _ = mlx.mlx_array_free(eps);
    var eb_eps = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eb_eps);
    try mlx.check(mlx.mlx_add(&eb_eps, eb, eps, s));
    // sin(x*alpha)^2
    var xa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xa);
    try mlx.check(mlx.mlx_multiply(&xa, x, ea, s));
    var sinxa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinxa);
    try mlx.check(mlx.mlx_sin(&sinxa, xa, s));
    var sin2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sin2);
    try mlx.check(mlx.mlx_multiply(&sin2, sinxa, sinxa, s));
    // (1/(eb+eps)) * sin2
    var inv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv);
    try mlx.check(mlx.mlx_divide(&inv, sin2, eb_eps, s));
    return addA(x, inv, s);
}

/// Causal Conv1d on NLC input: left-pad by `(k-1)*dil+1-stride`, then conv,
/// then add bias. Weight is MLX layout [out, k, in].
fn causalConv1d(x: mlx.mlx_array, w: mlx.mlx_array, bias: ?mlx.mlx_array, k: u32, stride: u32, dilation: u32, groups: u32, s: S) !mlx.mlx_array {
    // Causal left pad = effective_k - stride (reference enrich): equals (k-1)*d+1-stride
    // only when dilation=1; for strided SEANet downsampling the reference pads by
    // kernel_size - stride on the left. Same as before for the decoder's stride-1 path.
    const pad: i64 = @as(i64, @as(i64, (k - 1)) * @as(i64, dilation) + 1) - @as(i64, stride);
    var padded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(padded);
    if (pad > 0) {
        const axes = [_]c_int{ 0, 1, 2 };
        const lo = [_]c_int{ 0, @intCast(pad), 0 };
        const hi = [_]c_int{ 0, 0, 0 };
        const zero = mlx.mlx_array_new_float(0);
        defer _ = mlx.mlx_array_free(zero);
        try mlx.check(mlx.mlx_pad(&padded, x, &axes, 3, &lo, 3, &hi, 3, zero, "constant", s));
    } else {
        try mlx.check(mlx.mlx_array_set(&padded, x));
    }
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv1d(&out, padded, w, @intCast(stride), 0, @intCast(dilation), @intCast(groups), s));
    if (bias) |b| {
        const wb = try addA(out, b, s);
        _ = mlx.mlx_array_free(out);
        return wb;
    }
    return out;
}

/// Causal ConvTranspose1d on NLC: conv_transpose then trim `k-stride` from the
/// right. Weight in the sanitized layout (transposed (0,2,1) at load).
fn causalConvT1d(x: mlx.mlx_array, w: mlx.mlx_array, bias: ?mlx.mlx_array, k: u32, stride: u32, hidden: c_int, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv_transpose1d(&out, x, w, @intCast(stride), 0, 1, 0, 1, s));
    var biased = mlx.mlx_array_new();
    if (bias) |b| {
        try mlx.check(mlx.mlx_add(&biased, out, b, s));
        _ = mlx.mlx_array_free(out);
    } else {
        try mlx.check(mlx.mlx_array_set(&biased, out));
        _ = mlx.mlx_array_free(out);
    }
    const trim: i64 = @as(i64, k) - @as(i64, stride);
    if (trim > 0) {
        const T: c_int = mlx.getShape(biased)[1];
        const r = try sliceSeq(biased, 0, T - @as(c_int, @intCast(trim)), hidden, s);
        _ = mlx.mlx_array_free(biased);
        return r;
    }
    return biased;
}

/// Full LayerNorm over last dim (ConvNeXt norm), eps 1e-6.
fn layerNormFull(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, eps, s));
    return out;
}

/// Own a 3-D conv weight, transposing PyTorch [out,in,k] → MLX [out,k,in].
fn ownConvW(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    const perm = [_]c_int{ 0, 2, 1 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&out, raw, &perm, 3, s));
    return out;
}
fn ownConvWfmt(w: *const Weights, s: S, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
    const key = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(key);
    return ownConvW(w, key, s);
}

/// Own a transposed-conv weight: PyTorch ConvTranspose1d [in,out,k] → MLX
/// [out,k,in] via perm (1,2,0). (Regular convs use (0,2,1); the two differ.)
fn ownConvTW(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    const perm = [_]c_int{ 1, 2, 0 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&out, raw, &perm, 3, s));
    return out;
}
fn ownConvTWfmt(w: *const Weights, s: S, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
    const key = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(key);
    return ownConvTW(w, key, s);
}

const CodecTfLayer = struct {
    input_ln: mlx.mlx_array,
    post_ln: mlx.mlx_array,
    q_t: mlx.mlx_array,
    k_t: mlx.mlx_array,
    v_t: mlx.mlx_array,
    o_t: mlx.mlx_array,
    attn_scale: mlx.mlx_array,
    mlp_scale: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    up_t: mlx.mlx_array,
    down_t: mlx.mlx_array,
    fn deinit(self: *CodecTfLayer) void {
        inline for (.{ self.input_ln, self.post_ln, self.q_t, self.k_t, self.v_t, self.o_t, self.attn_scale, self.mlp_scale, self.gate_t, self.up_t, self.down_t }) |x| _ = mlx.mlx_array_free(x);
    }
};

const ConvNeXt = struct {
    dwconv_w: mlx.mlx_array,
    dwconv_b: mlx.mlx_array,
    ln_w: mlx.mlx_array,
    ln_b: mlx.mlx_array,
    pw1_t: mlx.mlx_array,
    pw1_b: mlx.mlx_array,
    pw2_t: mlx.mlx_array,
    pw2_b: mlx.mlx_array,
    gamma: mlx.mlx_array,
    fn deinit(self: *ConvNeXt) void {
        inline for (.{ self.dwconv_w, self.dwconv_b, self.ln_w, self.ln_b, self.pw1_t, self.pw1_b, self.pw2_t, self.pw2_b, self.gamma }) |x| _ = mlx.mlx_array_free(x);
    }
};

const ResUnit = struct {
    a1: mlx.mlx_array,
    b1: mlx.mlx_array,
    c1_w: mlx.mlx_array,
    c1_b: mlx.mlx_array,
    a2: mlx.mlx_array,
    b2: mlx.mlx_array,
    c2_w: mlx.mlx_array,
    c2_b: mlx.mlx_array,
    dilation: u32,
    fn deinit(self: *ResUnit) void {
        inline for (.{ self.a1, self.b1, self.c1_w, self.c1_b, self.a2, self.b2, self.c2_w, self.c2_b }) |x| _ = mlx.mlx_array_free(x);
    }
};

const DecBlock = struct {
    snake_a: mlx.mlx_array,
    snake_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_b: mlx.mlx_array,
    units: [3]ResUnit,
    in_dim: u32,
    out_dim: u32,
    rate: u32,
    fn deinit(self: *DecBlock) void {
        _ = mlx.mlx_array_free(self.snake_a);
        _ = mlx.mlx_array_free(self.snake_b);
        _ = mlx.mlx_array_free(self.up_w);
        _ = mlx.mlx_array_free(self.up_b);
        for (&self.units) |*u| u.deinit();
    }
};

pub const CodecDecoder = struct {
    cfg: CodecConfig,
    allocator: std.mem.Allocator,
    s: S,

    rvq_first_cb: mlx.mlx_array, // [2048, 256]
    rvq_first_oproj: mlx.mlx_array, // conv [512,1,256]
    rvq_rest_cb: []mlx.mlx_array, // 15 × [2048,256]
    rvq_rest_oproj: mlx.mlx_array,

    pre_conv_w: mlx.mlx_array,
    pre_conv_b: mlx.mlx_array,

    pt_in_t: mlx.mlx_array,
    pt_in_b: mlx.mlx_array,
    pt_out_t: mlx.mlx_array,
    pt_out_b: mlx.mlx_array,
    pt_norm: mlx.mlx_array,
    pt_layers: []CodecTfLayer,

    up_conv_w: [2]mlx.mlx_array,
    up_conv_b: [2]mlx.mlx_array,
    up_cnext: [2]ConvNeXt,

    dec_init_w: mlx.mlx_array,
    dec_init_b: mlx.mlx_array,
    dec_blocks: [4]DecBlock,
    out_snake_a: mlx.mlx_array,
    out_snake_b: mlx.mlx_array,
    out_conv_w: mlx.mlx_array,
    out_conv_b: mlx.mlx_array,

    /// Output samples per input frame: the total upsample ratio
    /// (2·2 from the upsample blocks · 8·5·4·3 from the decoder blocks) =
    /// 1920, i.e. 12.5 fps at 24 kHz.
    pub fn samplesPerFrame(self: *const CodecDecoder) usize {
        var n: usize = 1;
        for (self.cfg.upsampling_ratios) |r| n *= @as(usize, r);
        for (self.cfg.upsample_rates) |r| n *= @as(usize, r);
        return n;
    }

    pub fn deinit(self: *CodecDecoder) void {
        const a = self.allocator;
        _ = mlx.mlx_array_free(self.rvq_first_cb);
        _ = mlx.mlx_array_free(self.rvq_first_oproj);
        for (self.rvq_rest_cb) |c| _ = mlx.mlx_array_free(c);
        a.free(self.rvq_rest_cb);
        _ = mlx.mlx_array_free(self.rvq_rest_oproj);
        _ = mlx.mlx_array_free(self.pre_conv_w);
        _ = mlx.mlx_array_free(self.pre_conv_b);
        _ = mlx.mlx_array_free(self.pt_in_t);
        _ = mlx.mlx_array_free(self.pt_in_b);
        _ = mlx.mlx_array_free(self.pt_out_t);
        _ = mlx.mlx_array_free(self.pt_out_b);
        _ = mlx.mlx_array_free(self.pt_norm);
        for (self.pt_layers) |*l| l.deinit();
        a.free(self.pt_layers);
        for (0..2) |i| {
            _ = mlx.mlx_array_free(self.up_conv_w[i]);
            _ = mlx.mlx_array_free(self.up_conv_b[i]);
            self.up_cnext[i].deinit();
        }
        _ = mlx.mlx_array_free(self.dec_init_w);
        _ = mlx.mlx_array_free(self.dec_init_b);
        for (&self.dec_blocks) |*b| b.deinit();
        _ = mlx.mlx_array_free(self.out_snake_a);
        _ = mlx.mlx_array_free(self.out_snake_b);
        _ = mlx.mlx_array_free(self.out_conv_w);
        _ = mlx.mlx_array_free(self.out_conv_b);
    }

    /// Decode codes [T][16] → waveform samples (f32, owned).
    pub fn decode(self: *CodecDecoder, codes: []const [16]u32) ![]f32 {
        const s = self.s;
        const T: u32 = @intCast(codes.len);

        // ── Dequantize (NLC, [1, T, 512]) ──
        // Sum the semantic codebook (q0) and the 15 acoustic codebooks (q1..15),
        // each via its own 256-d codebook embed, then per-group 1×1 conv to 512.
        const first_sum = try self.codebookSum(self.rvq_first_cb, codes, 0, 1, T); // [1,T,256]
        defer _ = mlx.mlx_array_free(first_sum);
        const first = try causalConv1d(first_sum, self.rvq_first_oproj, null, 1, 1, 1, 1, s); // [1,T,512]
        defer _ = mlx.mlx_array_free(first);

        const rest_sum = try self.codebookSumMulti(self.rvq_rest_cb, codes, 1, T); // [1,T,256]
        defer _ = mlx.mlx_array_free(rest_sum);
        const rest = try causalConv1d(rest_sum, self.rvq_rest_oproj, null, 1, 1, 1, 1, s);
        defer _ = mlx.mlx_array_free(rest);

        var hidden = try addA(first, rest, s); // [1,T,512]

        // ── pre_conv (512→1024, k3) ──
        {
            const nx = try causalConv1d(hidden, self.pre_conv_w, self.pre_conv_b, 3, 1, 1, 1, s);
            _ = mlx.mlx_array_free(hidden);
            hidden = nx;
        }

        // ── pre_transformer ──
        {
            const nx = try self.preTransformer(hidden);
            _ = mlx.mlx_array_free(hidden);
            hidden = nx;
        }

        // ── upsample blocks (×2, ratio 2 each) ──
        for (0..2) |i| {
            const lat: c_int = @intCast(self.cfg.latent_dim);
            const ct = try causalConvT1d(hidden, self.up_conv_w[i], self.up_conv_b[i], 2, 2, lat, s);
            _ = mlx.mlx_array_free(hidden);
            hidden = try self.convNext(ct, &self.up_cnext[i]);
            _ = mlx.mlx_array_free(ct);
        }

        // ── decoder: init conv → 4 blocks → out snake → out conv ──
        {
            const nx = try causalConv1d(hidden, self.dec_init_w, self.dec_init_b, 7, 1, 1, 1, s);
            _ = mlx.mlx_array_free(hidden);
            hidden = nx;
        }
        for (&self.dec_blocks) |*b| {
            const nx = try self.decBlock(hidden, b);
            _ = mlx.mlx_array_free(hidden);
            hidden = nx;
        }
        {
            const sn = try snakeBeta(hidden, self.out_snake_a, self.out_snake_b, s);
            _ = mlx.mlx_array_free(hidden);
            const nx = try causalConv1d(sn, self.out_conv_w, self.out_conv_b, 7, 1, 1, 1, s); // [1, samples, 1]
            _ = mlx.mlx_array_free(sn);
            hidden = nx;
        }

        // Clip to [-1,1] and read out f32 samples.
        const lo = mlx.mlx_array_new_float(-1.0);
        defer _ = mlx.mlx_array_free(lo);
        const hi = mlx.mlx_array_new_float(1.0);
        defer _ = mlx.mlx_array_free(hi);
        var clipped_lo = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(clipped_lo);
        try mlx.check(mlx.mlx_maximum(&clipped_lo, hidden, lo, s));
        var clipped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(clipped);
        try mlx.check(mlx.mlx_minimum(&clipped, clipped_lo, hi, s));
        _ = mlx.mlx_array_free(hidden);

        var f32arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32arr);
        try mlx.check(mlx.mlx_astype(&f32arr, clipped, .float32, s));
        _ = mlx.mlx_array_eval(f32arr);

        const n: usize = @intCast(mlx.mlx_array_size(f32arr));
        const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
        const out = try self.allocator.alloc(f32, n);
        @memcpy(out, data[0..n]);
        return out;
    }

    /// Sum a single codebook's embedding over codebook index `start` (NLC [1,T,256]).
    fn codebookSum(self: *CodecDecoder, cb: mlx.mlx_array, codes: []const [16]u32, start: u32, count: u32, T: u32) !mlx.mlx_array {
        const s = self.s;
        const ids = try self.allocator.alloc(i32, T);
        defer self.allocator.free(ids);
        var acc: ?mlx.mlx_array = null;
        var q: u32 = start;
        while (q < start + count) : (q += 1) {
            for (codes, 0..) |fr, t| ids[t] = @intCast(fr[q]);
            const e = try embed(cb, ids, self.cfg.rvq_codebook_dim, s); // [1,T,256]
            if (acc) |a| {
                const na = try addA(a, e, s);
                _ = mlx.mlx_array_free(a);
                _ = mlx.mlx_array_free(e);
                acc = na;
            } else acc = e;
        }
        return acc.?;
    }

    /// Sum 15 acoustic codebooks (each its own embed table) starting at index 1.
    fn codebookSumMulti(self: *CodecDecoder, cbs: []const mlx.mlx_array, codes: []const [16]u32, start: u32, T: u32) !mlx.mlx_array {
        const s = self.s;
        const ids = try self.allocator.alloc(i32, T);
        defer self.allocator.free(ids);
        var acc: ?mlx.mlx_array = null;
        for (cbs, 0..) |cb, j| {
            const q: u32 = start + @as(u32, @intCast(j));
            for (codes, 0..) |fr, t| ids[t] = @intCast(fr[q]);
            const e = try embed(cb, ids, self.cfg.rvq_codebook_dim, s);
            if (acc) |a| {
                const na = try addA(a, e, s);
                _ = mlx.mlx_array_free(a);
                _ = mlx.mlx_array_free(e);
                acc = na;
            } else acc = e;
        }
        return acc.?;
    }

    fn preTransformer(self: *CodecDecoder, x_in: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        // input_proj 1024→512
        var x = try linearBias(x_in, self.pt_in_t, self.pt_in_b, s);
        const d = QwenDims{ .hidden = c.hidden, .heads = c.heads, .kv = c.kv, .head_dim = c.head_dim, .eps = c.rms_eps, .theta = c.rope_theta };
        for (self.pt_layers) |*layer| {
            const nx = try self.codecTfLayer(x, layer, d);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        const normed = try rms(x, self.pt_norm, c.rms_eps, s);
        _ = mlx.mlx_array_free(x);
        defer _ = mlx.mlx_array_free(normed);
        // output_proj 512→1024
        return linearBias(normed, self.pt_out_t, self.pt_out_b, s);
    }

    fn codecTfLayer(self: *CodecDecoder, x: mlx.mlx_array, layer: *const CodecTfLayer, d: QwenDims) !mlx.mlx_array {
        const s = self.s;
        const L: c_int = mlx.getShape(x)[1];
        const heads: c_int = @intCast(d.heads);
        const kv: c_int = @intCast(d.kv);
        const hd: c_int = @intCast(d.head_dim);

        const xn = try rms(x, layer.input_ln, d.eps, s);
        defer _ = mlx.mlx_array_free(xn);
        const q = try matmul(xn, layer.q_t, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try matmul(xn, layer.k_t, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try matmul(xn, layer.v_t, s);
        defer _ = mlx.mlx_array_free(v);
        const q4 = try reshape(q, &[_]c_int{ 1, L, heads, hd }, s);
        defer _ = mlx.mlx_array_free(q4);
        const qt = try transpose(q4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(qt);
        const k4 = try reshape(k, &[_]c_int{ 1, L, kv, hd }, s);
        defer _ = mlx.mlx_array_free(k4);
        const kt = try transpose(k4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(kt);
        const v4 = try reshape(v, &[_]c_int{ 1, L, kv, hd }, s);
        defer _ = mlx.mlx_array_free(v4);
        const vt = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(vt);
        const base = mlx.mlx_optional_float.some(d.theta);
        var qr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qr);
        try mlx.check(mlx.mlx_fast_rope(&qr, qt, hd, false, base, 1.0, 0, .{ .ctx = null }, s));
        var kr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kr);
        try mlx.check(mlx.mlx_fast_rope(&kr, kt, hd, false, base, 1.0, 0, .{ .ctx = null }, s));
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(d.head_dim)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_arr = mlx.mlx_array{ .ctx = null };
        const mode: [*:0]const u8 = if (L > 1) "causal" else "";
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vt, scale, mode, null_arr, null_arr, false, s));
        const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = try reshape(at, &[_]c_int{ 1, L, heads * hd }, s);
        defer _ = mlx.mlx_array_free(af);
        const o = try matmul(af, layer.o_t, s);
        defer _ = mlx.mlx_array_free(o);
        const o_scaled = try mulA(o, layer.attn_scale, s); // LayerScale
        defer _ = mlx.mlx_array_free(o_scaled);
        const h1 = try addA(x, o_scaled, s);
        defer _ = mlx.mlx_array_free(h1);
        // MLP
        const hn = try rms(h1, layer.post_ln, d.eps, s);
        defer _ = mlx.mlx_array_free(hn);
        const g = try matmul(hn, layer.gate_t, s);
        defer _ = mlx.mlx_array_free(g);
        const u = try matmul(hn, layer.up_t, s);
        defer _ = mlx.mlx_array_free(u);
        const act = try swiglu(g, u, s);
        defer _ = mlx.mlx_array_free(act);
        const down = try matmul(act, layer.down_t, s);
        defer _ = mlx.mlx_array_free(down);
        const down_scaled = try mulA(down, layer.mlp_scale, s);
        defer _ = mlx.mlx_array_free(down_scaled);
        return addA(h1, down_scaled, s);
    }

    fn convNext(self: *CodecDecoder, x: mlx.mlx_array, cn: *const ConvNeXt) !mlx.mlx_array {
        const s = self.s;
        const dim = self.cfg.latent_dim;
        // depthwise conv k7 groups=dim
        const dw = try causalConv1d(x, cn.dwconv_w, cn.dwconv_b, 7, 1, 1, dim, s);
        defer _ = mlx.mlx_array_free(dw);
        const ln = try layerNormFull(dw, cn.ln_w, cn.ln_b, 1e-6, s);
        defer _ = mlx.mlx_array_free(ln);
        const pw1 = try linearBias(ln, cn.pw1_t, cn.pw1_b, s);
        defer _ = mlx.mlx_array_free(pw1);
        const gact = try gelu(pw1, s);
        defer _ = mlx.mlx_array_free(gact);
        const pw2 = try linearBias(gact, cn.pw2_t, cn.pw2_b, s);
        defer _ = mlx.mlx_array_free(pw2);
        const scaled = try mulA(pw2, cn.gamma, s);
        defer _ = mlx.mlx_array_free(scaled);
        return addA(x, scaled, s);
    }

    fn decBlock(self: *CodecDecoder, x: mlx.mlx_array, b: *const DecBlock) !mlx.mlx_array {
        const s = self.s;
        // block[0]: snake (in_dim)
        const sn = try snakeBeta(x, b.snake_a, b.snake_b, s);
        defer _ = mlx.mlx_array_free(sn);
        // block[1]: upsample transpose conv (in→out, k=2*rate, stride=rate)
        var h = try causalConvT1d(sn, b.up_w, b.up_b, 2 * b.rate, b.rate, @intCast(b.out_dim), s);
        // block[2..4]: residual units (dil 1,3,9)
        for (&b.units) |*u| {
            const nh = try self.resUnit(h, u);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        return h;
    }

    fn resUnit(self: *CodecDecoder, x: mlx.mlx_array, u: *const ResUnit) !mlx.mlx_array {
        const s = self.s;
        const a1 = try snakeBeta(x, u.a1, u.b1, s);
        defer _ = mlx.mlx_array_free(a1);
        const c1 = try causalConv1d(a1, u.c1_w, u.c1_b, 7, 1, u.dilation, 1, s);
        defer _ = mlx.mlx_array_free(c1);
        const a2 = try snakeBeta(c1, u.a2, u.b2, s);
        defer _ = mlx.mlx_array_free(a2);
        const c2 = try causalConv1d(a2, u.c2_w, u.c2_b, 1, 1, 1, 1, s);
        defer _ = mlx.mlx_array_free(c2);
        return addA(c2, x, s);
    }
};

/// Compute a codebook embedding table: embedding_sum / clip(cluster_usage, 1e-5).
fn computeCodebook(w: *const Weights, a: std.mem.Allocator, s: S, comptime base_fmt: []const u8, args: anytype) !mlx.mlx_array {
    const es_key = try std.fmt.allocPrint(a, base_fmt ++ ".embedding_sum", args);
    defer a.free(es_key);
    const cu_key = try std.fmt.allocPrint(a, base_fmt ++ ".cluster_usage", args);
    defer a.free(cu_key);
    const es = try ownWeight(w, es_key); // [2048, 256]
    defer _ = mlx.mlx_array_free(es);
    const cu = try ownWeight(w, cu_key); // [2048]
    defer _ = mlx.mlx_array_free(cu);
    const size: c_int = mlx.getShape(cu)[0];
    var cu2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cu2);
    try mlx.check(mlx.mlx_reshape(&cu2, cu, &[_]c_int{ size, 1 }, 2, s));
    const eps = mlx.mlx_array_new_float(1e-5);
    defer _ = mlx.mlx_array_free(eps);
    var cu_clip = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cu_clip);
    try mlx.check(mlx.mlx_maximum(&cu_clip, cu2, eps, s));
    var embed_w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&embed_w, es, cu_clip, s));
    _ = mlx.mlx_array_eval(embed_w);
    return embed_w;
}

pub fn loadCodecDecoder(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !CodecDecoder {
    const dir = try std.fmt.allocPrint(allocator, "{s}/speech_tokenizer", .{model_dir});
    defer allocator.free(dir);
    return loadCodecDecoderFrom(io, allocator, s, dir);
}

/// Load the codec weights directly from `tokenizer_dir` (the dir holding
/// `model.safetensors` with this decoder layout): Qwen3-TTS `speech_tokenizer/`
/// or Breeze `audio_tokenizer/`.
pub fn loadCodecDecoderFrom(io: std.Io, allocator: std.mem.Allocator, s: S, tokenizer_dir: []const u8) !CodecDecoder {
    var w = try model_mod.loadWeights(io, allocator, tokenizer_dir);
    defer w.deinit();

    var d: CodecDecoder = undefined;
    d.cfg = .{};
    d.allocator = allocator;
    d.s = s;

    // Quantizer.
    d.rvq_first_cb = try computeCodebook(&w, allocator, s, "decoder.quantizer.rvq_first.vq.layers.0._codebook", .{});
    d.rvq_first_oproj = try ownConvW(&w, "decoder.quantizer.rvq_first.output_proj.weight", s);
    d.rvq_rest_cb = try allocator.alloc(mlx.mlx_array, d.cfg.num_quant - d.cfg.num_sem);
    for (0..d.rvq_rest_cb.len) |i| {
        d.rvq_rest_cb[i] = try computeCodebook(&w, allocator, s, "decoder.quantizer.rvq_rest.vq.layers.{d}._codebook", .{i});
    }
    d.rvq_rest_oproj = try ownConvW(&w, "decoder.quantizer.rvq_rest.output_proj.weight", s);

    // pre_conv.
    d.pre_conv_w = try ownConvW(&w, "decoder.pre_conv.conv.weight", s);
    d.pre_conv_b = try ownWeight(&w, "decoder.pre_conv.conv.bias");

    // pre_transformer.
    d.pt_in_t = try ownT(&w, "decoder.pre_transformer.input_proj.weight", s);
    d.pt_in_b = try ownWeight(&w, "decoder.pre_transformer.input_proj.bias");
    d.pt_out_t = try ownT(&w, "decoder.pre_transformer.output_proj.weight", s);
    d.pt_out_b = try ownWeight(&w, "decoder.pre_transformer.output_proj.bias");
    d.pt_norm = try ownWeight(&w, "decoder.pre_transformer.norm.weight");
    d.pt_layers = try allocator.alloc(CodecTfLayer, d.cfg.layers);
    for (d.pt_layers, 0..) |*layer, i| {
        const p = "decoder.pre_transformer.layers.{d}.{s}";
        layer.* = .{
            .input_ln = try ownWfmt(&w, allocator, p, .{ i, "input_layernorm.weight" }),
            .post_ln = try ownWfmt(&w, allocator, p, .{ i, "post_attention_layernorm.weight" }),
            .q_t = try ownTfmt(&w, s, allocator, p, .{ i, "self_attn.q_proj.weight" }),
            .k_t = try ownTfmt(&w, s, allocator, p, .{ i, "self_attn.k_proj.weight" }),
            .v_t = try ownTfmt(&w, s, allocator, p, .{ i, "self_attn.v_proj.weight" }),
            .o_t = try ownTfmt(&w, s, allocator, p, .{ i, "self_attn.o_proj.weight" }),
            .attn_scale = try ownWfmt(&w, allocator, p, .{ i, "self_attn_layer_scale.scale" }),
            .mlp_scale = try ownWfmt(&w, allocator, p, .{ i, "mlp_layer_scale.scale" }),
            .gate_t = try ownTfmt(&w, s, allocator, p, .{ i, "mlp.gate_proj.weight" }),
            .up_t = try ownTfmt(&w, s, allocator, p, .{ i, "mlp.up_proj.weight" }),
            .down_t = try ownTfmt(&w, s, allocator, p, .{ i, "mlp.down_proj.weight" }),
        };
    }

    // upsample blocks.
    for (0..2) |i| {
        d.up_conv_w[i] = try ownConvTWfmt(&w, s, allocator, "decoder.upsample.{d}.0.conv.weight", .{i});
        d.up_conv_b[i] = try ownWfmt(&w, allocator, "decoder.upsample.{d}.0.conv.bias", .{i});
        const cp = "decoder.upsample.{d}.1.{s}";
        d.up_cnext[i] = .{
            .dwconv_w = try ownConvWfmt(&w, s, allocator, "decoder.upsample.{d}.1.dwconv.conv.weight", .{i}),
            .dwconv_b = try ownWfmt(&w, allocator, "decoder.upsample.{d}.1.dwconv.conv.bias", .{i}),
            .ln_w = try ownWfmt(&w, allocator, cp, .{ i, "norm.weight" }),
            .ln_b = try ownWfmt(&w, allocator, cp, .{ i, "norm.bias" }),
            .pw1_t = try ownTfmt(&w, s, allocator, cp, .{ i, "pwconv1.weight" }),
            .pw1_b = try ownWfmt(&w, allocator, cp, .{ i, "pwconv1.bias" }),
            .pw2_t = try ownTfmt(&w, s, allocator, cp, .{ i, "pwconv2.weight" }),
            .pw2_b = try ownWfmt(&w, allocator, cp, .{ i, "pwconv2.bias" }),
            .gamma = try ownWfmt(&w, allocator, cp, .{ i, "gamma" }),
        };
    }

    // decoder: init conv.
    d.dec_init_w = try ownConvW(&w, "decoder.decoder.0.conv.weight", s);
    d.dec_init_b = try ownWeight(&w, "decoder.decoder.0.conv.bias");
    // 4 decoder blocks (keys decoder.decoder.1..4).
    for (0..4) |bi| {
        const layer_idx = bi + 1;
        const in_dim = d.cfg.decoder_dim >> @intCast(bi);
        const out_dim = d.cfg.decoder_dim >> @intCast(bi + 1);
        const rate = d.cfg.upsample_rates[bi];
        var blk: DecBlock = undefined;
        blk.in_dim = in_dim;
        blk.out_dim = out_dim;
        blk.rate = rate;
        blk.snake_a = try ownWfmt(&w, allocator, "decoder.decoder.{d}.block.0.alpha", .{layer_idx});
        blk.snake_b = try ownWfmt(&w, allocator, "decoder.decoder.{d}.block.0.beta", .{layer_idx});
        blk.up_w = try ownConvTWfmt(&w, s, allocator, "decoder.decoder.{d}.block.1.conv.weight", .{layer_idx});
        blk.up_b = try ownWfmt(&w, allocator, "decoder.decoder.{d}.block.1.conv.bias", .{layer_idx});
        const dils = [_]u32{ 1, 3, 9 };
        for (0..3) |ui| {
            const bidx = ui + 2; // block.2/3/4
            const up = "decoder.decoder.{d}.block.{d}.{s}";
            blk.units[ui] = .{
                .a1 = try ownWfmt(&w, allocator, up, .{ layer_idx, bidx, "act1.alpha" }),
                .b1 = try ownWfmt(&w, allocator, up, .{ layer_idx, bidx, "act1.beta" }),
                .c1_w = try ownConvWfmt(&w, s, allocator, up, .{ layer_idx, bidx, "conv1.conv.weight" }),
                .c1_b = try ownWfmt(&w, allocator, up, .{ layer_idx, bidx, "conv1.conv.bias" }),
                .a2 = try ownWfmt(&w, allocator, up, .{ layer_idx, bidx, "act2.alpha" }),
                .b2 = try ownWfmt(&w, allocator, up, .{ layer_idx, bidx, "act2.beta" }),
                .c2_w = try ownConvWfmt(&w, s, allocator, up, .{ layer_idx, bidx, "conv2.conv.weight" }),
                .c2_b = try ownWfmt(&w, allocator, up, .{ layer_idx, bidx, "conv2.conv.bias" }),
                .dilation = dils[ui],
            };
        }
        d.dec_blocks[bi] = blk;
    }
    // output snake + conv (keys decoder.decoder.5 / .6).
    d.out_snake_a = try ownWeight(&w, "decoder.decoder.5.alpha");
    d.out_snake_b = try ownWeight(&w, "decoder.decoder.5.beta");
    d.out_conv_w = try ownConvW(&w, "decoder.decoder.6.conv.weight", s);
    d.out_conv_b = try ownWeight(&w, "decoder.decoder.6.conv.bias");

    return d;
}


// ───────────────────────────── Mimi encoder (Qwen3-TTS 12Hz) ─────────────────────────────
//
// Port of the `audio_tokenizer` encoder: the Qwen3-TTS codec's NEURAL encoder
// that maps a waveform → per-frame codebook tokens. Breeze reference-audio
// (`ref_audio`) runs a clip through this encoder and splices the resulting frames
// into the prompt, so voice cloning needs it. Mirrors `CodecDecoder`'s conventions
// (NLC tensors, `ownConvW` weight layout, causal padding). The transformer uses a
// LAYER-NORM (weight + bias), no QK-norm, and a sliding-window causal mask.
//
// Subgraph consumed: the `encoder.*` tensors under `audio_tokenizer/`.

const EncTfLayer = struct {
    input_ln_w: mlx.mlx_array,
    input_ln_b: mlx.mlx_array,
    post_ln_w: mlx.mlx_array,
    post_ln_b: mlx.mlx_array,
    q_t: mlx.mlx_array,
    k_t: mlx.mlx_array,
    v_t: mlx.mlx_array,
    o_t: mlx.mlx_array,
    attn_scale: mlx.mlx_array,
    mlp_scale: mlx.mlx_array,
    fc1_t: mlx.mlx_array,
    fc2_t: mlx.mlx_array,
    fn deinit(self: *EncTfLayer) void {
        inline for (.{ self.input_ln_w, self.input_ln_b, self.post_ln_w, self.post_ln_b, self.q_t, self.k_t, self.v_t, self.o_t, self.attn_scale, self.mlp_scale, self.fc1_t, self.fc2_t }) |x| _ = mlx.mlx_array_free(x);
    }
};

const EncResnet = struct {
    c1_w: mlx.mlx_array, // block.1 conv: [out=dim/2, in=dim, k=3]
    c1_b: mlx.mlx_array,
    c2_w: mlx.mlx_array, // block.3 conv: [out=dim, in=dim/2, k=1]
    c2_b: mlx.mlx_array,
    fn deinit(self: *EncResnet) void {
        _ = mlx.mlx_array_free(self.c1_w);
        _ = mlx.mlx_array_free(self.c1_b);
        _ = mlx.mlx_array_free(self.c2_w);
        _ = mlx.mlx_array_free(self.c2_b);
    }
};

/// ELU(x) = x if x>0 else exp(x)-1 (alpha=1).
fn eluArr(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var exp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(exp);
    try mlx.check(mlx.mlx_exp(&exp, x, s));
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var expm1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expm1);
    try mlx.check(mlx.mlx_subtract(&expm1, exp, one, s));
    var ge = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ge);
    try mlx.check(mlx.mlx_greater_equal(&ge, x, zero, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&out, ge, x, expm1, s));
    _ = mlx.mlx_array_eval(out);
    return out;
}

/// Nearest-centroid index: `x_nlc` `[N, T, D]`; returns per-frame `[N*T]` indices
/// into a `[V, D]` codebook (argmin over squared euclidean distance).
fn codebookNearest(x: mlx.mlx_array, cb: mlx.mlx_array, s: S) !mlx.mlx_array {
    const xshape = mlx.getShape(x);
    const R: usize = xshape.len;
    const D: c_int = xshape[R - 1];
    const size: c_int = @intCast(mlx.mlx_array_size(x));
    const rows: c_int = @divExact(size, D);

    var xr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xr);
    try mlx.check(mlx.mlx_reshape(&xr, x, &[_]c_int{ rows, D }, 2, s));

    var cbT = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cbT);
    try mlx.check(mlx.mlx_transpose(&cbT, cb, s));
    var dot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dot);
    try mlx.check(mlx.mlx_matmul(&dot, xr, cbT, s)); // [rows, V]

    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_square(&x2, xr, s));
    var xsum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xsum);
    try mlx.check(mlx.mlx_sum_axis(&xsum, x2, 1, true, s)); // [rows, 1] keepdims

    var cb2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cb2);
    try mlx.check(mlx.mlx_square(&cb2, cb, s));
    var cbsum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cbsum);
    try mlx.check(mlx.mlx_sum_axis(&cbsum, cb2, 1, false, s)); // [V]

    const two = mlx.mlx_array_new_float(-2.0);
    defer _ = mlx.mlx_array_free(two);
    var twodot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(twodot);
    try mlx.check(mlx.mlx_multiply(&twodot, dot, two, s)); // [rows, V]
    var d2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d2);
    try mlx.check(mlx.mlx_add(&d2, xsum, twodot, s));
    var dist2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dist2);
    try mlx.check(mlx.mlx_add(&dist2, d2, cbsum, s)); // broadcast

    var idx = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_argmin_axis(&idx, dist2, 1, false, s));
    _ = mlx.mlx_array_eval(idx);
    return idx; // [N*T]
}

/// Gather a codebook by `[N*T]` indices → `[N*T, D]`.
fn codebookGather(cb: mlx.mlx_array, idx: mlx.mlx_array, s: S) !mlx.mlx_array {
    var e = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&e, cb, idx, 0, s));
    _ = mlx.mlx_array_eval(e);
    return e;
}

const RVQNearest = struct {
    x_nlc: mlx.mlx_array,
    cb: mlx.mlx_array,
};

/// Compute a codebook embedding table: embed_sum / clip(cluster_usage, 1e-5).
fn recomputeCodebookEncode(w: *const Weights, a: std.mem.Allocator, s: S, comptime base_fmt: []const u8, args: anytype) !mlx.mlx_array {
    const es_key = try std.fmt.allocPrint(a, base_fmt ++ ".embed_sum", args);
    defer a.free(es_key);
    const cu_key = try std.fmt.allocPrint(a, base_fmt ++ ".cluster_usage", args);
    defer a.free(cu_key);
    const es = try ownWeight(w, es_key); // [V, D]
    defer _ = mlx.mlx_array_free(es);
    const cu = try ownWeight(w, cu_key); // [V]
    defer _ = mlx.mlx_array_free(cu);
    const size: c_int = mlx.getShape(cu)[0];
    var cu2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cu2);
    try mlx.check(mlx.mlx_reshape(&cu2, cu, &[_]c_int{ size, 1 }, 2, s));
    const eps = mlx.mlx_array_new_float(1e-5);
    defer _ = mlx.mlx_array_free(eps);
    var cu_clip = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cu_clip);
    try mlx.check(mlx.mlx_maximum(&cu_clip, cu2, eps, s));
    var embed_w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&embed_w, es, cu_clip, s));
    _ = mlx.mlx_array_eval(embed_w);
    return embed_w;
}

pub const MimiEncoder = struct {
    cfg: CodecConfig,
    allocator: std.mem.Allocator,
    io: std.Io,
    s: S,

    enc_convs: [6]mlx.mlx_array,
    enc_biases: [6]mlx.mlx_array,
    enc_resnets: [4]EncResnet,
    ds_w: mlx.mlx_array,
    et_layers: []EncTfLayer,
    sem_in: mlx.mlx_array,
    sem_cb: mlx.mlx_array,
    ac_in: mlx.mlx_array,
    ac_cbs: []mlx.mlx_array, // 15 × [2048,256]

    const RATIOS = [_]u32{ 4, 5, 6, 8 };

    pub fn deinit(self: *MimiEncoder) void {
        const a = self.allocator;
        inline for (0..6) |i| {
            _ = mlx.mlx_array_free(self.enc_convs[i]);
            _ = mlx.mlx_array_free(self.enc_biases[i]);
        }
        for (self.enc_resnets) |r| {
            _ = mlx.mlx_array_free(r.c1_w);
            _ = mlx.mlx_array_free(r.c2_w);
        }
        for (self.et_layers) |*l| l.deinit();
        a.free(self.et_layers);
        _ = mlx.mlx_array_free(self.ds_w);
        _ = mlx.mlx_array_free(self.sem_in);
        _ = mlx.mlx_array_free(self.sem_cb);
        _ = mlx.mlx_array_free(self.ac_in);
        for (self.ac_cbs) |cc| _ = mlx.mlx_array_free(cc);
        a.free(self.ac_cbs);
    }

    pub fn encode(self: *MimiEncoder, x_nlc: mlx.mlx_array) ![]const []u32 {
        const s = self.s;
        const hidden = try self.encoderForward(x_nlc);
        _ = mlx.mlx_array_free(x_nlc);
        dumpHost(self.io, self.allocator, "BREEZE_DUMP_ENC", hidden);
        const htf = try self.encoderTransformer(hidden);
        defer _ = mlx.mlx_array_free(htf);
        _ = mlx.mlx_array_eval(htf);
        dumpHost(self.io, self.allocator, "BREEZE_DUMP_TF", htf);
        // downsample conv (k=4, s=2, causal) on NLC [1, T, 512].
        const dsc = try causalConv1d(htf, self.ds_w, null, 4, 2, 1, 1, s);
        _ = mlx.mlx_array_eval(dsc);
        dumpHost(self.io, self.allocator, "BREEZE_DUMP_DSC", dsc);
        defer _ = mlx.mlx_array_free(dsc);
        return self.quantizeEncode(dsc);
    }

    /// Encode mono f32 waveform samples (24 kHz) → `[frames][16]` codes.
    pub fn encodeSamples(self: *MimiEncoder, samples: []const f32) ![]const []u32 {
        const s = self.s;
        return self.encode(try hostNlc(samples, 1, self.allocator, s));
    }

    fn encoderForward(self: *MimiEncoder, x: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const trace = std.c.getenv("BREEZE_ENC_TRACE") != null;
        var h = try causalConv1d(x, self.enc_convs[0], self.enc_biases[0], 7, 1, 1, 1, s);
        dumpHost(self.io, self.allocator, "BREEZE_DUMP_C0", h);
        if (trace) {
            _ = mlx.mlx_array_eval(h);
            std.debug.print("enc conv0 ok {d}\n", .{mlx.mlx_array_ndim(h)});
        }
        for (0..4) |i| {
            if (trace) std.debug.print("stage {d} resnet\n", .{i});
            const nx = try self.resnetForward(h, &self.enc_resnets[i], i + 1);
            if (trace) {
                _ = mlx.mlx_array_eval(nx);
                std.debug.print("stage {d} resnet ok\n", .{i});
            }
            if (i == 0) dumpHost(self.io, self.allocator, "BREEZE_DUMP_R1", nx);
            const el = try eluArr(nx, s);
            _ = mlx.mlx_array_eval(el);
            if (trace) std.debug.print("stage {d} elu ok\n", .{i});
            _ = mlx.mlx_array_free(h);
            _ = mlx.mlx_array_free(nx);
            const kk: u32 = 2 * RATIOS[i];
            const cv = try causalConv1d(el, self.enc_convs[i + 1], self.enc_biases[i + 1], kk, RATIOS[i], 1, 1, s);
            _ = mlx.mlx_array_free(el);
            _ = mlx.mlx_array_eval(cv);
            if (i == 0) dumpHost(self.io, self.allocator, "BREEZE_DUMP_RATIO0", cv)
            else if (i == 1) dumpHost(self.io, self.allocator, "BREEZE_DUMP_RATIO1", cv)
            else if (i == 2) dumpHost(self.io, self.allocator, "BREEZE_DUMP_RATIO2", cv)
            else dumpHost(self.io, self.allocator, "BREEZE_DUMP_RATIO3", cv);
            if (trace) std.debug.print("stage {d} ratio ok\n", .{i});
            h = cv;
        }
        // oracle has a final ELU between ratio3 and the last conv (layers.13)
        const elz = try eluArr(h, s);
        _ = mlx.mlx_array_free(h);
        const last = try causalConv1d(elz, self.enc_convs[5], self.enc_biases[5], 3, 1, 1, 1, s);
        _ = mlx.mlx_array_free(elz);
        _ = mlx.mlx_array_eval(last);
        if (trace) std.debug.print("enc last ok\n", .{});
        return last;
    }

    fn resnetForward(self: *MimiEncoder, x: mlx.mlx_array, r: *const EncResnet, dbg: usize) !mlx.mlx_array {
        const s = self.s;
        // Reference block order: [ELU, conv(dim→dim/2, k=3), ELU, conv(dim/2→dim, k=1)] + residual.
        var nm0: [48]u8 = undefined;
        var nm1: [48]u8 = undefined;
        var nm2: [48]u8 = undefined;
        var nm3: [48]u8 = undefined;
        var p0: ?[]const u8 = null;
        var p1: ?[]const u8 = null;
        var p2: ?[]const u8 = null;
        var p3: ?[]const u8 = null;
        if (dbg > 0) {
            p0 = std.fmt.bufPrint(&nm0, "BREEZE_DUMP_S{d}EL0", .{dbg}) catch null;
            p1 = std.fmt.bufPrint(&nm1, "BREEZE_DUMP_S{d}C1", .{dbg}) catch null;
            p2 = std.fmt.bufPrint(&nm2, "BREEZE_DUMP_S{d}EL", .{dbg}) catch null;
            p3 = std.fmt.bufPrint(&nm3, "BREEZE_DUMP_S{d}C2", .{dbg}) catch null;
        }
        const el0 = try eluArr(x, s);
        if (p0) |nm| dumpHost(self.io, self.allocator, nm, el0);
        defer _ = mlx.mlx_array_free(el0);
        const n1 = try causalConv1d(el0, r.c1_w, r.c1_b, 3, 1, 1, 1, s); // block.1 [dim/2, dim, 3]
        if (p1) |nm| dumpHost(self.io, self.allocator, nm, n1);
        defer _ = mlx.mlx_array_free(n1);
        const el = try eluArr(n1, s);
        if (p2) |nm| dumpHost(self.io, self.allocator, nm, el);
        defer _ = mlx.mlx_array_free(el);
        const n2 = try causalConv1d(el, r.c2_w, r.c2_b, 1, 1, 1, 1, s); // block.3 [dim, dim/2, 1]
        if (p3) |nm| dumpHost(self.io, self.allocator, nm, n2);
        defer _ = mlx.mlx_array_free(n2);
        return addA(n2, x, s);
    }

    fn encoderTransformer(self: *MimiEncoder, x_in: mlx.mlx_array) !mlx.mlx_array {
        var x = x_in;
        for (self.et_layers, 0..) |*layer, li| {
            const nx = try self.encTfLayerForward(x, layer);
            if (std.c.getenv("BREEZE_ENC_TRACE") != null) {
                _ = mlx.mlx_array_eval(nx);
                std.debug.print("tf layer {d} ok\n", .{li});
            }
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        return x;
    }

    fn encTfLayerForward(self: *MimiEncoder, x: mlx.mlx_array, layer: *const EncTfLayer) !mlx.mlx_array {
        const s = self.s;
        const L: c_int = mlx.getShape(x)[1];
        const heads: c_int = @intCast(self.cfg.heads);
        const hd: c_int = @intCast(self.cfg.head_dim);
        const kv: c_int = @intCast(self.cfg.kv);

        const tr = std.c.getenv("BREEZE_ENC_TRACE") != null;
        const xn = try layerNormFull(x, layer.input_ln_w, layer.input_ln_b, self.cfg.rms_eps, s);
        defer _ = mlx.mlx_array_free(xn);
        if (tr) { _ = mlx.mlx_array_eval(xn); std.debug.print("tf.ln ok\n", .{}); }
        const q = try matmul(xn, layer.q_t, s);
        defer _ = mlx.mlx_array_free(q);
        if (tr) { _ = mlx.mlx_array_eval(q); std.debug.print("tf.q ok\n", .{}); }
        const k = try matmul(xn, layer.k_t, s);
if (tr) { _ = mlx.mlx_array_eval(k); std.debug.print("tf.k ok\n", .{}); }
        defer _ = mlx.mlx_array_free(k);
        const v = try matmul(xn, layer.v_t, s);
if (tr) { _ = mlx.mlx_array_eval(v); std.debug.print("tf.v ok\n", .{}); }
        defer _ = mlx.mlx_array_free(v);
        const q4 = try reshape(q, &[_]c_int{ 1, L, heads, hd }, s);
        defer _ = mlx.mlx_array_free(q4);
        const qt = try transpose(q4, &[_]c_int{ 0, 2, 1, 3 }, s);
if (tr) { _ = mlx.mlx_array_eval(qt); std.debug.print("tf.qt ok\n", .{}); }
        defer _ = mlx.mlx_array_free(qt);
        const k4 = try reshape(k, &[_]c_int{ 1, L, kv, hd }, s);
        defer _ = mlx.mlx_array_free(k4);
        const kt = try transpose(k4, &[_]c_int{ 0, 2, 1, 3 }, s);
if (tr) { _ = mlx.mlx_array_eval(kt); std.debug.print("tf.kt ok\n", .{}); }
        defer _ = mlx.mlx_array_free(kt);
        const v4 = try reshape(v, &[_]c_int{ 1, L, kv, hd }, s);
        defer _ = mlx.mlx_array_free(v4);
        const vt = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
if (tr) { _ = mlx.mlx_array_eval(vt); std.debug.print("tf.vt ok\n", .{}); }
        defer _ = mlx.mlx_array_free(vt);
        const base = mlx.mlx_optional_float.some(self.cfg.rope_theta);
        var qr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qr);
        try mlx.check(mlx.mlx_fast_rope(&qr, qt, hd, false, base, 1.0, 0, .{ .ctx = null }, s));
        var kr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kr);
        try mlx.check(mlx.mlx_fast_rope(&kr, kt, hd, false, base, 1.0, 0, .{ .ctx = null }, s));
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.cfg.head_dim)));
        const mode: [*:0]const u8 = if (L > 1) "causal" else "";
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vt, scale, mode, .{ .ctx = null }, .{ .ctx = null }, false, s));
        if (tr) { _ = mlx.mlx_array_eval(attn); std.debug.print("tf.sdpa ok\n", .{}); }
        const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = try reshape(at, &[_]c_int{ 1, L, heads * hd }, s);
        defer _ = mlx.mlx_array_free(af);
        const o = try matmul(af, layer.o_t, s);
        defer _ = mlx.mlx_array_free(o);
        const os = try mulA(o, layer.attn_scale, s);
        defer _ = mlx.mlx_array_free(os);
        const h1 = try addA(x, os, s);
        defer _ = mlx.mlx_array_free(h1);
        if (tr) { _ = mlx.mlx_array_eval(h1); std.debug.print("tf.h1 ok\n", .{}); }
        const hn = try layerNormFull(h1, layer.post_ln_w, layer.post_ln_b, self.cfg.rms_eps, s);
        defer _ = mlx.mlx_array_free(hn);
        const fc1 = try matmul(hn, layer.fc1_t, s);
        defer _ = mlx.mlx_array_free(fc1);
        if (tr) { _ = mlx.mlx_array_eval(fc1); std.debug.print("tf.fc1 ok\n", .{}); }
        const act = try gelu(fc1, s);
        defer _ = mlx.mlx_array_free(act);
        const fc2 = try matmul(act, layer.fc2_t, s);
        defer _ = mlx.mlx_array_free(fc2);
        const ms = try mulA(fc2, layer.mlp_scale, s);
        defer _ = mlx.mlx_array_free(ms);
        if (tr) { _ = mlx.mlx_array_eval(ms); std.debug.print("tf.ms ok\n", .{}); }
        return addA(h1, ms, s);
    }

    fn quantizeEncode(self: *MimiEncoder, x_nlc: mlx.mlx_array) ![]const []u32 {
        const s = self.s;
        const a = self.allocator;

        // Semantic branch (1 codebook): 512→256 input_proj → [1, T, 256].
        const sem_nlc = try causalConv1d(x_nlc, self.sem_in, null, 1, 1, 1, 1, s);
        _ = mlx.mlx_array_eval(sem_nlc);
        dumpHost(self.io, self.allocator, "BREEZE_DUMP_SEM", sem_nlc);
        defer _ = mlx.mlx_array_free(sem_nlc);
        const T: usize = @intCast(mlx.getShape(sem_nlc)[1]);
        const sem2 = try flattenNLC(sem_nlc, s); // [T, 256]
        defer _ = mlx.mlx_array_free(sem2);

        const codes = try a.alloc([]u32, T);
        errdefer {
            for (codes) |r| a.free(r);
            a.free(codes);
        }
        for (codes) |*r| r.* = a.alloc(u32, 16) catch return error.OutOfMemory;

        const s_idx = try codebookNearest(sem2, self.sem_cb, s); // [T]
        defer _ = mlx.mlx_array_free(s_idx);
        const s_f32 = try idxHostF32(s_idx, a, s);
        defer a.free(s_f32);
        for (codes, 0..) |row, t| row[0] = @intFromFloat(@round(s_f32[t]));

        // Acoustic branch: its own input_proj on the RAW downsampled latent →
        // [1, T, 256]. The split RVQ feeds BOTH branches the same latent (the
        // semantic quantization is not subtracted — decode re-adds it).
        const ac_nlc = try causalConv1d(x_nlc, self.ac_in, null, 1, 1, 1, 1, s);
        _ = mlx.mlx_array_eval(ac_nlc);
        dumpHost(self.io, self.allocator, "BREEZE_DUMP_AC", ac_nlc);
        defer _ = mlx.mlx_array_free(ac_nlc);
        const ac2 = try flattenNLC(ac_nlc, s); // [T, 256]
        defer _ = mlx.mlx_array_free(ac2);

        var resid = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&resid, ac2));
        for (0..15) |q| {
            if (std.c.getenv("BREEZE_ENC_TRACE") != null) std.debug.print("enc-q{d}\n", .{q});
            const qi = try codebookNearest(resid, self.ac_cbs[q], s); // [T]
            defer _ = mlx.mlx_array_free(qi);
            const qf = try idxHostF32(qi, a, s);
            const e = try codebookGather(self.ac_cbs[q], qi, s); // [T, 256]
            defer _ = mlx.mlx_array_free(e);
            const nr = try subA(resid, e, s);
            _ = mlx.mlx_array_free(resid);
            resid = nr;
            for (codes, 0..) |row, t| row[q + 1] = @intFromFloat(@round(qf[t]));
            a.free(qf);
        }
        _ = mlx.mlx_array_free(resid);
        return codes;
    }
};

/// Flatten [1, T, C] → [T, C].
fn flattenNLC(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const T = sh[1];
    const C = sh[2];
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, x, &[_]c_int{ T, C }, 2, s));
    _ = mlx.mlx_array_eval(out);
    return out;
}


/// Read a flat array of int-ish indices into owned f32 (caller frees).
fn idxHostF32(a: mlx.mlx_array, alloc: std.mem.Allocator, s: S) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a, .float32, s));
    _ = mlx.mlx_array_eval(f);
    const n: usize = @intCast(mlx.mlx_array_size(f));
    const d = mlx.mlx_array_data_float32(f) orelse { std.debug.print("idxHostF32 NoData\n", .{}); return error.NoData; };
    const out = try alloc.alloc(f32, n);
    @memcpy(out, d[0..n]);
    return out;
}


fn dumpHost(io: std.Io, alloc: std.mem.Allocator, path_env: []const u8, arr: mlx.mlx_array) void {
    const z = alloc.alloc(u8, path_env.len + 1) catch return;
    defer alloc.free(z);
    @memcpy(z[0..path_env.len], path_env);
    z[path_env.len] = 0;
    const p_opt = std.c.getenv(@ptrCast(z)) orelse return;
    const p = std.mem.span(p_opt);
    _ = mlx.mlx_array_eval(arr);
    const n: usize = @intCast(mlx.mlx_array_size(arr));
    const d = mlx.mlx_array_data_float32(arr) orelse return;
    const buf = alloc.alloc(f32, n) catch return;
    defer alloc.free(buf);
    @memcpy(buf, d[0..n]);
    const bytes = std.mem.sliceAsBytes(buf);
    const f = std.Io.Dir.createFileAbsolute(io, p, .{}) catch return;
    defer f.close(io);
    var wb: [65536]u8 = undefined;
    var fw = f.writer(io, &wb);
    fw.interface.writeAll(bytes) catch {};
    fw.interface.flush() catch {};
}

/// Build a host [1, T, C] f32 array from samples (one channel).
fn hostNlc(samples: []const f32, channels: u32, alloc: std.mem.Allocator, s: S) !mlx.mlx_array {
    _ = channels;
    _ = alloc;
    _ = s;
    const T: usize = samples.len; // NLC [1, samples, 1]
    const shape = [_]c_int{ 1, @intCast(T), 1 };
    return mlx.mlx_array_new_data(@constCast(@ptrCast(samples.ptr)), &shape, 3, .float32);
}

/// Load the `audio_tokenizer` encoder weights into a `MimiEncoder`.
pub fn loadMimiEncoder(io: std.Io, allocator: std.mem.Allocator, s: S, tokenizer_dir: []const u8) !MimiEncoder {
    var w = try model_mod.loadWeights(io, allocator, tokenizer_dir);
    defer w.deinit();

    var e: MimiEncoder = undefined;
    e.cfg = .{};
    // Encoder transformer geometry (encoder_transformer_config): heads 8 /
    // kv 8, NOT the decoder's 16. head_dim 64 from q_proj [512,512] / 8.
    e.cfg.heads = 8;
    e.cfg.kv = 8;
    e.allocator = allocator;
    e.io = io;
    e.s = s;

    e.enc_convs[0] = try ownConvW(&w, "encoder.encoder.layers.0.conv.weight", s);
    e.enc_biases[0] = try ownWeight(&w, "encoder.encoder.layers.0.conv.bias");
    // resnet + ratio conv pairs (layers.1=res, layers.3=conv; layers.4=res, layers.6=conv; ...)
    const res_pairs = [_]usize{ 1, 4, 7, 10 };
    const conv_pairs = [_]usize{ 3, 6, 9, 12 };
    for (0..4) |i| {
        e.enc_resnets[i] = .{
            .c1_w = try ownConvWfmt(&w, s, allocator, "encoder.encoder.layers.{d}.block.1.conv.weight", .{res_pairs[i]}),
            .c1_b = try ownWfmt(&w, allocator, "encoder.encoder.layers.{d}.block.1.conv.bias", .{res_pairs[i]}),
            .c2_w = try ownConvWfmt(&w, s, allocator, "encoder.encoder.layers.{d}.block.3.conv.weight", .{res_pairs[i]}),
            .c2_b = try ownWfmt(&w, allocator, "encoder.encoder.layers.{d}.block.3.conv.bias", .{res_pairs[i]}),
        };
        e.enc_convs[i + 1] = try ownConvWfmt(&w, s, allocator, "encoder.encoder.layers.{d}.conv.weight", .{conv_pairs[i]});
        e.enc_biases[i + 1] = try ownWfmt(&w, allocator, "encoder.encoder.layers.{d}.conv.bias", .{conv_pairs[i]});
    }
    e.enc_convs[5] = try ownConvW(&w, "encoder.encoder.layers.14.conv.weight", s);
    e.enc_biases[5] = try ownWeight(&w, "encoder.encoder.layers.14.conv.bias");

    e.ds_w = try ownConvW(&w, "encoder.downsample.conv.weight", s);

    e.et_layers = try allocator.alloc(EncTfLayer, 8);
    for (e.et_layers, 0..) |*layer, i| {
        const p = "encoder.encoder_transformer.layers.{d}.{s}";
        layer.* = .{
            .input_ln_w = try ownWfmt(&w, allocator, p, .{ i, "input_layernorm.weight" }),
            .input_ln_b = try ownWfmt(&w, allocator, p, .{ i, "input_layernorm.bias" }),
            .post_ln_w = try ownWfmt(&w, allocator, p, .{ i, "post_attention_layernorm.weight" }),
            .post_ln_b = try ownWfmt(&w, allocator, p, .{ i, "post_attention_layernorm.bias" }),
            .q_t = try ownTfmt(&w, s, allocator, p, .{ i, "self_attn.q_proj.weight" }),
            .k_t = try ownTfmt(&w, s, allocator, p, .{ i, "self_attn.k_proj.weight" }),
            .v_t = try ownTfmt(&w, s, allocator, p, .{ i, "self_attn.v_proj.weight" }),
            .o_t = try ownTfmt(&w, s, allocator, p, .{ i, "self_attn.o_proj.weight" }),
            .attn_scale = try ownWfmt(&w, allocator, p, .{ i, "self_attn_layer_scale.scale" }),
            .mlp_scale = try ownWfmt(&w, allocator, p, .{ i, "mlp_layer_scale.scale" }),
            .fc1_t = try ownTfmt(&w, s, allocator, "encoder.encoder_transformer.layers.{d}.mlp.fc1.weight", .{i}),
            .fc2_t = try ownTfmt(&w, s, allocator, "encoder.encoder_transformer.layers.{d}.mlp.fc2.weight", .{i}),
        };
    }

    e.sem_in = try ownConvW(&w, "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight", s);
    e.sem_cb = try recomputeCodebookEncode(&w, allocator, s, "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook", .{});
    e.ac_in = try ownConvW(&w, "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight", s);
    e.ac_cbs = try allocator.alloc(mlx.mlx_array, 15);
    for (e.ac_cbs, 0..) |*cb, i| cb.* = try recomputeCodebookEncode(&w, allocator, s, "encoder.quantizer.acoustic_residual_vector_quantizer.layers.{d}.codebook", .{i});
    return e;
}

fn readAbsAlloc(io: std.Io, alloc: std.mem.Allocator, path: []const u8, limit: usize) !?[]u8 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [8192]u8 = undefined;
    var rs = f.reader(io, &rb);
    return try rs.interface.allocRemaining(alloc, .limited(limit));
}


test "mimi encoder parity vs oracle (MLX_SERVE_MIMI_ENCODE)" {
    log.enableStderr();
    const tok_dir = std.mem.span(std.c.getenv("MLX_SERVE_MIMI_ENCODE") orelse return error.SkipZigTest);
    const wav_path = std.mem.span(std.c.getenv("MLX_SERVE_MIMI_WAV") orelse return error.SkipZigTest);
    const oracle_path = std.mem.span(std.c.getenv("MLX_SERVE_MIMI_ORACLE") orelse return error.SkipZigTest);

    const alloc = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = if (std.c.getenv("BREEZE_ENC_CPU") != null) mlx.mlx_default_cpu_stream_new() else mlx.mlx_default_gpu_stream_new();

    // Load wav → mono f32 at 24 kHz.
    const wav_bytes = (try readAbsAlloc(io, alloc, wav_path, 1 << 30)).?;
    defer alloc.free(wav_bytes);
    const d = try wav_mod.decode(alloc, wav_bytes);
    defer alloc.free(d.pcm);
    std.debug.assert(d.channels == 1 or d.channels == 2);

    var enc = try loadMimiEncoder(io, alloc, s, tok_dir);
    defer enc.deinit();
    std.debug.print("mimi-start T={d}\n", .{d.pcm.len});
    const frames = enc.encodeSamples(d.pcm) catch |e| { std.debug.print("encodeSamples ERR {s}\n", .{@errorName(e)}); return e; };
    defer {
        for (frames) |f| alloc.free(f);
        alloc.free(frames);
    }
    std.debug.print("mimi: {d} frames\n", .{frames.len});

    // Load oracle codes via a raw-bytes read.
    const oracle = (try readAbsAlloc(io, alloc, oracle_path, 1 << 20)).?;
    defer alloc.free(oracle);
    const n_frames = oracle.len / (16 * 2);
    try std.testing.expectEqual(n_frames, frames.len);
    // Frames t>=1 must be byte-exact; frame 0 rides the left-pad edge where
    // torch/MLX accumulation order can flip near-tie codebook picks (≤ 16 cells).
    var mism: usize = 0;
    var first_mm: usize = 0;
    var mism_t0: usize = 0;
    for (frames, 0..) |row, t| {
        for (0..16) |q| {
            const ov = @as(u16, oracle[t * 32 + q * 2]) | (@as(u16, oracle[t * 32 + q * 2 + 1]) << 8);
            if (ov != row[q]) {
                if (mism == 0) first_mm = t * 16 + q;
                if (t == 0) mism_t0 += 1 else mism += 1;
            }
        }
    }
    std.debug.print("mimi parity: t>0 {d}/{d} mismatched (first at {d}); t0 {d}/16\n", .{ mism, (frames.len - 1) * 16, first_mm, mism_t0 });
    try std.testing.expectEqual(@as(usize, 0), mism);
    try std.testing.expectEqual(@as(u32, 1517), frames[0][0]);
    // Little-endian raw dump for external diffing.
    if (std.c.getenv("MLX_SERVE_MIMI_DUMP") != null) {
        const buf = try alloc.alloc(u8, frames.len * 16 * 2);
        defer alloc.free(buf);
        for (frames, 0..) |row, t| for (0..16) |q| {
            const v: u16 = @intCast(row[q]);
            buf[t * 32 + q * 2] = @truncate(v);
            buf[t * 32 + q * 2 + 1] = @truncate(v >> 8);
        };
        const p = std.mem.span(std.c.getenv("MLX_SERVE_MIMI_DUMP") orelse unreachable);
        const f = try std.Io.Dir.createFileAbsolute(io, p, .{});
        defer f.close(io);
        var wb: [8192]u8 = undefined;
        var fw = f.writer(io, &wb);
        try fw.interface.writeAll(buf);
        try fw.interface.flush();
    }
    try std.testing.expectEqual(@as(usize, 0), mism);
}
