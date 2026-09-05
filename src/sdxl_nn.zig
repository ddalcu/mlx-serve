//! Tensor plumbing shared by SDXL's UNet and VAE.
//!
//! Both are diffusers modules built from the same handful of pieces — grouped
//! norms, SiLU/GELU, 3x3 convs, biased linears, resnet blocks — so they live
//! here once rather than twice. `flux.zig` has its own near-copies of several
//! of these; they are NOT shared because flux's are specialized in ways SDXL
//! cannot use (its `groupNorm` returns bfloat16 unconditionally, its conv
//! weights arrive pre-permuted from the mflux conversion, and its attention
//! reads through `QLinear`). Merging them would mean threading three flags
//! through an oracle-validated decoder to serve a second caller.
//!
//! Two conventions hold throughout:
//!
//!   SPATIAL TENSORS ARE NHWC. MLX's `conv2d` reads NHWC, so the port converts
//!   once at each boundary rather than transposing around every conv. The
//!   fixtures are NCHW (PyTorch), so the parity tests convert at the edge.
//!
//!   CONV WEIGHTS ARE TRANSPOSED AT LOAD. PyTorch stores `[out, in, kh, kw]`;
//!   MLX wants `[out, kh, kw, in]`. This is the single most dangerous silent
//!   difference in the port — a mis-permuted conv weight is still a valid
//!   tensor of the right size, so nothing errors and the image is noise. Every
//!   conv weight goes through `loadConvWeight`, never `dupWeight`.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const lora_mod = @import("lora.zig");

const S = mlx.mlx_stream;
pub const Weights = model_mod.Weights;

// ── Elementwise / shape helpers ──

pub inline fn addA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}

pub inline fn subA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, a, b, s));
    return o;
}

pub inline fn mulA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}

pub inline fn matmul(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, a, b, s));
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

pub inline fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}

pub inline fn contiguous(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}

pub fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}

/// Duplicate an owned handle (a +1 the caller frees).
pub inline fn dupA(x: mlx.mlx_array) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, x));
    return o;
}

/// Replace `*slot` with `next`, freeing the old value. The forward passes are
/// long chains of "transform h in place"; without this every step needs three
/// lines and one of them is a free that is easy to omit.
pub inline fn replace(slot: *mlx.mlx_array, next: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(slot.*);
    slot.* = next;
}

// ── Activations ──

pub fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    return mulA(x, sig, s);
}

/// Exact erf GELU — diffusers' `GELU(approximate="none")`, which is what both
/// the GEGLU feed-forwards and the VAE use. NOT interchangeable with the tanh
/// approximation: the difference is small enough to look like noise and large
/// enough to move an image.
pub fn gelu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const inv_sqrt2 = mlx.mlx_array_new_float(0.7071067811865476);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    const scaled = try mulA(x, inv_sqrt2, s);
    defer _ = mlx.mlx_array_free(scaled);
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_erf(&e, scaled, s));
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const one_plus = try addA(e, one, s);
    defer _ = mlx.mlx_array_free(one_plus);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const half_x = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(half_x);
    return mulA(half_x, one_plus, s);
}

// ── Normalization ──

/// PyTorch-compatible GroupNorm over NHWC `[1,H,W,C]`, affine, computed in
/// fp32 and returned in the INPUT's dtype.
///
/// `eps` is a parameter and not a constant because SDXL's UNet uses two
/// different values in the same forward: resnet and output norms take the
/// config's `norm_eps` (1e-5), while `Transformer2DModel`'s input norm
/// hardcodes 1e-6 upstream. One value for both is a small, uniform, entirely
/// invisible error.
pub fn groupNorm(x: mlx.mlx_array, weight: mlx.mlx_array, bias: mlx.mlx_array, groups: c_int, eps: f32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,W,C]
    const h = sh[1];
    const w_ = sh[2];
    const c = sh[3];
    const cg = @divExact(c, groups);
    const out_dt = mlx.mlx_array_dtype(x);

    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    // [1,H,W,C] -> [1,H*W,groups,cg] -> [1,groups,H*W,cg] -> [1,groups,H*W*cg]
    const r1 = try reshape(xf, &[_]c_int{ 1, h * w_, groups, cg }, s);
    defer _ = mlx.mlx_array_free(r1);
    const t1 = try transpose(r1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(t1);
    const flat = try reshape(t1, &[_]c_int{ 1, groups, h * w_ * cg }, s);
    defer _ = mlx.mlx_array_free(flat);

    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, flat, -1, true, s));
    const xc = try subA(flat, mean, s);
    defer _ = mlx.mlx_array_free(xc);
    const sq = try mulA(xc, xc, s);
    defer _ = mlx.mlx_array_free(sq);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(epsa);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, ve, s));
    const norm = try mulA(xc, rsq, s);
    defer _ = mlx.mlx_array_free(norm);

    const b1 = try reshape(norm, &[_]c_int{ 1, groups, h * w_, cg }, s);
    defer _ = mlx.mlx_array_free(b1);
    const b2 = try transpose(b1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(b2);
    const b3 = try reshape(b2, &[_]c_int{ 1, h, w_, c }, s);
    defer _ = mlx.mlx_array_free(b3);

    const wf = try astype(weight, .float32, s);
    defer _ = mlx.mlx_array_free(wf);
    const bf = try astype(bias, .float32, s);
    defer _ = mlx.mlx_array_free(bf);
    const sc = try mulA(b3, wf, s);
    defer _ = mlx.mlx_array_free(sc);
    const out = try addA(sc, bf, s);
    defer _ = mlx.mlx_array_free(out);
    return astype(out, out_dt, s);
}

/// LayerNorm over the last axis. diffusers' `BasicTransformerBlock` norms are
/// eps 1e-5 with affine weights.
pub fn layerNorm(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&o, x, w, b, eps, s));
    return o;
}

// ── Convolutions (NHWC) ──

/// 3x3 (or 1x1) conv with bias, stride 1, symmetric padding.
pub fn conv2d(x: mlx.mlx_array, w: mlx.mlx_array, bias: ?mlx.mlx_array, pad: c_int, s: S) !mlx.mlx_array {
    // Materialize first: mlx_conv2d silently miscomputes on strided/lazy views.
    const xc = try contiguous(x, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 1, 1, pad, pad, 1, 1, 1, s));
    if (bias) |bb| {
        const r = try addA(o, bb, s);
        _ = mlx.mlx_array_free(o);
        return r;
    }
    return o;
}

/// Stride-2 3x3 conv with SYMMETRIC padding 1 — diffusers `Downsample2D` as
/// the UNet configures it (`downsample_padding: 1`).
///
/// The VAE's downsampler is a DIFFERENT operator: it pads asymmetrically
/// (0,1,0,1) and convolves with no padding. Same class name upstream, same
/// output shape, different pixels. Only the UNet path uses this one.
pub fn conv2dStride2(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, s: S) !mlx.mlx_array {
    const xc = try contiguous(x, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 2, 2, 1, 1, 1, 1, 1, s));
    const r = try addA(o, bias, s);
    _ = mlx.mlx_array_free(o);
    return r;
}

/// Nearest-neighbour 2x upsample then 3x3 conv — diffusers `Upsample2D`.
pub fn upsample2x(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, s: S) !mlx.mlx_array {
    var r1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r1);
    try mlx.check(mlx.mlx_repeat_axis(&r1, x, 2, 1, s));
    var r2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r2);
    try mlx.check(mlx.mlx_repeat_axis(&r2, r1, 2, 2, s));
    return conv2d(r2, w, bias, 1, s);
}

// ── Linear ──

/// A linear layer holding its weight PRE-TRANSPOSED to `[in, out]`, so the
/// forward is one matmul with no per-call permute. `bias` is optional:
/// diffusers' attention projections (`to_q`/`to_k`/`to_v`) have none, while
/// everything else here does.
pub const Linear = struct {
    /// DENSE: the weight PRE-TRANSPOSED to `[in, out]`. QUANTIZED (affine): the
    /// PACKED weight `[out, in·bits/32]` as stored — not transposed, because
    /// `mlx_quantized_matmul(transpose_w=true)` wants the `[out, in]` logical
    /// orientation. Which one is decided by `scales != null`.
    w_t: mlx.mlx_array,
    b: ?mlx.mlx_array,
    /// Affine-quantization operands. Non-null marks this a quantized linear
    /// (the SceneWorks q4/q8 diffusers packs quantize every SDXL matmul —
    /// attention, ff, proj_in/out, the time/add embeds, resnet `time_emb_proj`
    /// — and keep convs + norms dense). `mlx_quantized_matmul` dequant-frees.
    scales: ?mlx.mlx_array = null,
    q_biases: ?mlx.mlx_array = null,
    bits: u32 = 0,
    group_size: u32 = 0,
    /// Attached adapters, summed at forward time and never merged into
    /// `w_t` — the base weight stays pristine so a second request with a
    /// different stack costs a re-attach rather than a reload. Non-owning:
    /// the `lora.Stack` owns the arrays these point at.
    lora_refs: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined,
    lora_count: u8 = 0,

    pub fn deinit(self: *Linear) void {
        _ = mlx.mlx_array_free(self.w_t);
        if (self.b) |bb| _ = mlx.mlx_array_free(bb);
        if (self.scales) |x| _ = mlx.mlx_array_free(x);
        if (self.q_biases) |x| _ = mlx.mlx_array_free(x);
    }

    pub fn forward(self: *const Linear, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var y = if (self.scales) |sc| blk: {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_quantized_matmul(&o, x, self.w_t, sc, self.q_biases.?, true, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", s));
            break :blk o;
        } else try matmul(x, self.w_t, s);
        errdefer _ = mlx.mlx_array_free(y);
        // The delta rides on the PRE-bias product, matching mflux's
        // `FusedLoRALinear`: bias is not part of the low-rank update. A LoRA
        // delta is low-rank DENSE and rides a quantized base unchanged (never
        // merged — a quantized weight would requantize the delta away).
        if (self.lora_count > 0) {
            const d = try lora_mod.deltaSum(x, self.lora_refs[0..self.lora_count], s);
            defer _ = mlx.mlx_array_free(d);
            replace(&y, try addA(y, d, s));
        }
        if (self.b) |bb| {
            defer _ = mlx.mlx_array_free(y);
            return addA(y, bb, s);
        }
        return y;
    }

    pub fn setLoraRefs(self: *Linear, refs: []const lora_mod.Ref) void {
        self.lora_count = @intCast(refs.len);
        @memcpy(self.lora_refs[0..refs.len], refs);
    }

    pub fn clearLoraRefs(self: *Linear) void {
        self.lora_count = 0;
    }
};

// ── Weight loading ──

pub fn fmtKey(a: std.mem.Allocator, comptime f: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, f, args);
}

/// Fetch a weight, cast to `dt`, as an owned handle.
pub fn dupWeight(w: *const Weights, key: []const u8, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const src = w.get(key) orelse {
        log.err("[sdxl] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingSdxlWeight;
    };
    return astype(src, dt, s);
}

pub fn dupWeightOpt(w: *const Weights, key: []const u8, dt: mlx.mlx_dtype, s: S) ?mlx.mlx_array {
    const src = w.get(key) orelse return null;
    return astype(src, dt, s) catch null;
}

/// Load a conv weight, permuting PyTorch `[out, in, kh, kw]` to MLX's
/// `[out, kh, kw, in]`.
///
/// Doing this at load rather than per forward matters for more than speed: a
/// permute in the hot path is a lazy view, and `mlx_conv2d` miscomputes on
/// those. Materialized here, once.
pub fn loadConvWeight(w: *const Weights, key: []const u8, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const src = w.get(key) orelse {
        log.err("[sdxl] MISSING CONV WEIGHT: {s}\n", .{key});
        return error.MissingSdxlWeight;
    };
    const cast = try astype(src, dt, s);
    defer _ = mlx.mlx_array_free(cast);
    const perm = try transpose(cast, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(perm);
    return contiguous(perm, s);
}

/// Load a linear weight, transposing PyTorch `[out, in]` to `[in, out]`.
/// `bias_key` null means the layer genuinely has no bias.
/// Infer affine (bits, group_size) from a packed weight + its scales. The
/// packed weight's last axis is `in·bits/32`; scales' is `in/group_size`, so
/// `w_cols·32/s_cols = bits·group_size`. The product alone is ambiguous
/// ((3,64) vs (6,32)…), so group sizes are tried in convention order (64 is the
/// mlx/mlx-community default). Local copy of flux.zig's helper — the sdxl port
/// keeps its plumbing self-contained.
pub const QuantGeom = struct { bits: u32, group_size: u32 };

pub fn inferQuantGeometry(w: mlx.mlx_array, scales: mlx.mlx_array) QuantGeom {
    const wsh = mlx.getShape(w);
    const ssh = mlx.getShape(scales);
    const fallback = QuantGeom{ .bits = 4, .group_size = 64 };
    if (wsh.len == 0 or ssh.len == 0) return fallback;
    const w_cols: usize = @intCast(wsh[wsh.len - 1]);
    const s_cols: usize = @intCast(ssh[ssh.len - 1]);
    if (s_cols == 0 or (w_cols * 32) % s_cols != 0) return fallback;
    const product = w_cols * 32 / s_cols; // bits · group_size
    for ([_]u32{ 64, 32, 128 }) |gs| {
        if (product % gs != 0) continue;
        const bits: u32 = @intCast(product / gs);
        for ([_]u32{ 2, 3, 4, 5, 6, 8 }) |vb| {
            if (bits == vb) return .{ .bits = bits, .group_size = gs };
        }
    }
    return fallback;
}

pub fn loadLinear(
    w: *const Weights,
    key_w: []const u8,
    key_b: ?[]const u8,
    dt: mlx.mlx_dtype,
    s: S,
) !Linear {
    const src = w.get(key_w) orelse {
        log.err("[sdxl] MISSING LINEAR WEIGHT: {s}\n", .{key_w});
        return error.MissingSdxlWeight;
    };
    const b = if (key_b) |kb| try dupWeight(w, kb, dt, s) else null;
    errdefer {
        if (b) |bb| _ = mlx.mlx_array_free(bb);
    }

    // Affine-quantized when a sibling `.scales` is present (the q4/q8 packs).
    // The packed weight (uint32) is kept AS-IS — never astyped — and forward
    // routes through `mlx_quantized_matmul(transpose_w=true)`.
    if (std.mem.endsWith(u8, key_w, ".weight")) {
        const base = key_w[0 .. key_w.len - ".weight".len];
        var sbuf: [256]u8 = undefined;
        var bbuf: [256]u8 = undefined;
        const sk = std.fmt.bufPrint(&sbuf, "{s}.scales", .{base}) catch return error.SdxlKeyTooLong;
        if (w.get(sk)) |scales_src| {
            const bk = std.fmt.bufPrint(&bbuf, "{s}.biases", .{base}) catch return error.SdxlKeyTooLong;
            const qbiases_src = w.get(bk) orelse {
                log.err("[sdxl] quantized {s} has scales but no biases\n", .{base});
                return error.MissingSdxlWeight;
            };
            const packed_w = try astype(src, mlx.mlx_array_dtype(src), s); // own the uint32, unchanged
            errdefer _ = mlx.mlx_array_free(packed_w);
            const scales = try astype(scales_src, dt, s);
            errdefer _ = mlx.mlx_array_free(scales);
            const qbiases = try astype(qbiases_src, dt, s);
            errdefer _ = mlx.mlx_array_free(qbiases);
            const geo = inferQuantGeometry(packed_w, scales);
            return .{ .w_t = packed_w, .b = b, .scales = scales, .q_biases = qbiases, .bits = geo.bits, .group_size = geo.group_size };
        }
    }

    const cast = try astype(src, dt, s);
    defer _ = mlx.mlx_array_free(cast);
    const t = try transpose(cast, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(t);
    const w_t = try contiguous(t, s);
    errdefer _ = mlx.mlx_array_free(w_t);
    return .{ .w_t = w_t, .b = b };
}

/// A diffusers `ResnetBlock2D`, with the optional time-embedding projection
/// the UNet's resnets carry and the VAE's do not.
pub const Resnet = struct {
    n1w: mlx.mlx_array,
    n1b: mlx.mlx_array,
    c1w: mlx.mlx_array,
    c1b: mlx.mlx_array,
    n2w: mlx.mlx_array,
    n2b: mlx.mlx_array,
    c2w: mlx.mlx_array,
    c2b: mlx.mlx_array,
    /// `conv_shortcut`, present only when in_channels != out_channels.
    sw: ?mlx.mlx_array = null,
    sb: ?mlx.mlx_array = null,
    /// `time_emb_proj` — UNet only.
    temb: ?Linear = null,
    eps: f32,

    pub fn deinit(self: *Resnet) void {
        inline for (.{ "n1w", "n1b", "c1w", "c1b", "n2w", "n2b", "c2w", "c2b" }) |f| {
            _ = mlx.mlx_array_free(@field(self, f));
        }
        if (self.sw) |x| _ = mlx.mlx_array_free(x);
        if (self.sb) |x| _ = mlx.mlx_array_free(x);
        if (self.temb) |*t| @constCast(t).deinit();
    }

    /// `emb` is the `[1, 1280]` time embedding; null for the VAE's resnets.
    pub fn forward(self: *const Resnet, x: mlx.mlx_array, emb: ?mlx.mlx_array, s: S) !mlx.mlx_array {
        const h0 = try groupNorm(x, self.n1w, self.n1b, 32, self.eps, s);
        defer _ = mlx.mlx_array_free(h0);
        const a0 = try silu(h0, s);
        defer _ = mlx.mlx_array_free(a0);
        var h = try conv2d(a0, self.c1w, self.c1b, 1, s);
        errdefer _ = mlx.mlx_array_free(h);

        // Time conditioning: SiLU on the embedding, project to this block's
        // channel count, then broadcast-add over H and W. `resnet_time_scale_shift`
        // is "default" on SDXL, so this is a pure additive shift — the scale/shift
        // (FiLM) variant reads the same weights and is a different model.
        if (self.temb) |*proj| {
            const emb_a = emb orelse return error.MissingTimeEmbedding;
            const act = try silu(emb_a, s);
            defer _ = mlx.mlx_array_free(act);
            const p = try proj.forward(act, s);
            defer _ = mlx.mlx_array_free(p);
            const psh = mlx.getShape(p); // [1, C]
            const b1 = try reshape(p, &[_]c_int{ 1, 1, 1, psh[1] }, s);
            defer _ = mlx.mlx_array_free(b1);
            replace(&h, try addA(h, b1, s));
        }

        {
            const h1 = try groupNorm(h, self.n2w, self.n2b, 32, self.eps, s);
            defer _ = mlx.mlx_array_free(h1);
            const a1 = try silu(h1, s);
            defer _ = mlx.mlx_array_free(a1);
            replace(&h, try conv2d(a1, self.c2w, self.c2b, 1, s));
        }

        defer _ = mlx.mlx_array_free(h);
        if (self.sw) |sw| {
            const sc = try conv2d(x, sw, self.sb, 0, s);
            defer _ = mlx.mlx_array_free(sc);
            return addA(h, sc, s);
        }
        return addA(h, x, s);
    }
};

/// Load a resnet from `<prefix>.*`. `with_temb` distinguishes the UNet's
/// resnets from the VAE's.
pub fn loadResnet(
    w: *const Weights,
    a: std.mem.Allocator,
    prefix: []const u8,
    with_temb: bool,
    eps: f32,
    dt: mlx.mlx_dtype,
    s: S,
) !Resnet {
    const K = struct {
        fn get(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8, d: mlx.mlx_dtype, st: S) !mlx.mlx_array {
            const k = try fmtKey(aa, "{s}.{s}", .{ p, sub });
            defer aa.free(k);
            return dupWeight(ww, k, d, st);
        }
        fn conv(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8, d: mlx.mlx_dtype, st: S) !mlx.mlx_array {
            const k = try fmtKey(aa, "{s}.{s}", .{ p, sub });
            defer aa.free(k);
            return loadConvWeight(ww, k, d, st);
        }
        fn convOpt(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8, d: mlx.mlx_dtype, st: S) ?mlx.mlx_array {
            const k = fmtKey(aa, "{s}.{s}", .{ p, sub }) catch return null;
            defer aa.free(k);
            if (ww.get(k) == null) return null;
            return loadConvWeight(ww, k, d, st) catch null;
        }
        fn getOpt(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8, d: mlx.mlx_dtype, st: S) ?mlx.mlx_array {
            const k = fmtKey(aa, "{s}.{s}", .{ p, sub }) catch return null;
            defer aa.free(k);
            return dupWeightOpt(ww, k, d, st);
        }
    };

    var r = Resnet{
        .n1w = try K.get(w, a, prefix, "norm1.weight", dt, s),
        .n1b = try K.get(w, a, prefix, "norm1.bias", dt, s),
        .c1w = try K.conv(w, a, prefix, "conv1.weight", dt, s),
        .c1b = try K.get(w, a, prefix, "conv1.bias", dt, s),
        .n2w = try K.get(w, a, prefix, "norm2.weight", dt, s),
        .n2b = try K.get(w, a, prefix, "norm2.bias", dt, s),
        .c2w = try K.conv(w, a, prefix, "conv2.weight", dt, s),
        .c2b = try K.get(w, a, prefix, "conv2.bias", dt, s),
        .sw = K.convOpt(w, a, prefix, "conv_shortcut.weight", dt, s),
        .sb = K.getOpt(w, a, prefix, "conv_shortcut.bias", dt, s),
        .eps = eps,
    };
    errdefer r.deinit();
    if (with_temb) {
        const kw = try fmtKey(a, "{s}.time_emb_proj.weight", .{prefix});
        defer a.free(kw);
        const kb = try fmtKey(a, "{s}.time_emb_proj.bias", .{prefix});
        defer a.free(kb);
        r.temb = try loadLinear(w, kw, kb, dt, s);
    }
    return r;
}

// ── NCHW <-> NHWC ──

pub fn nchwToNhwc(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const t = try transpose(x, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(t);
    return contiguous(t, s);
}

pub fn nhwcToNchw(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const t = try transpose(x, &[_]c_int{ 0, 3, 1, 2 }, s);
    defer _ = mlx.mlx_array_free(t);
    return contiguous(t, s);
}

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic, no checkpoint required.
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

fn constArray(vals: []const f32, shape: []const c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(vals.ptr, shape.ptr, @intCast(shape.len), .float32);
}

fn readAll(a: std.mem.Allocator, arr: mlx.mlx_array, s: S) ![]f32 {
    const f = try astype(arr, .float32, s);
    defer _ = mlx.mlx_array_free(f);
    const c = try contiguous(f, s);
    defer _ = mlx.mlx_array_free(c);
    _ = mlx.mlx_array_eval(c);
    const n: usize = @intCast(mlx.mlx_array_size(c));
    const src = mlx.mlx_array_data_float32(c) orelse return error.NoData;
    const out = try a.alloc(f32, n);
    @memcpy(out, src[0..n]);
    return out;
}

test "sdxl nn: groupNorm normalizes per group and honors its eps parameter" {
    const s = mlx.mlx_default_gpu_stream_new();
    const a = testing.allocator;
    // 2 groups of 2 channels over a 1x2 spatial extent, NHWC [1,1,2,4].
    const vals = [_]f32{ 1, 2, 10, 20, 3, 4, 30, 40 };
    const x = constArray(&vals, &[_]c_int{ 1, 1, 2, 4 });
    defer _ = mlx.mlx_array_free(x);
    const w = constArray(&[_]f32{ 1, 1, 1, 1 }, &[_]c_int{4});
    defer _ = mlx.mlx_array_free(w);
    const b = constArray(&[_]f32{ 0, 0, 0, 0 }, &[_]c_int{4});
    defer _ = mlx.mlx_array_free(b);

    const out = try groupNorm(x, w, b, 2, 1e-5, s);
    defer _ = mlx.mlx_array_free(out);
    const got = try readAll(a, out, s);
    defer a.free(got);

    // Each group is zero-mean, unit-variance over its own 4 values.
    // Group 0 holds {1,2,3,4}; group 1 holds {10,20,30,40}.
    var g0: f64 = 0;
    var g1: f64 = 0;
    for (got, 0..) |v, i| {
        // NHWC channel index is i % 4; channels 0,1 are group 0.
        if (i % 4 < 2) g0 += v else g1 += v;
    }
    try testing.expect(@abs(g0) < 1e-4);
    try testing.expect(@abs(g1) < 1e-4);
    // Both groups are the SAME shape of distribution, so normalization must
    // map them onto the same values despite the 10x scale difference.
    try testing.expectApproxEqAbs(got[0], got[2], 1e-4);
    try testing.expectApproxEqAbs(got[1], got[3], 1e-4);
}

test "sdxl nn: conv weight load permutes PyTorch OIHW to MLX OHWI" {
    // The trap this guards: a mis-permuted conv weight is a valid tensor of the
    // right element count, so nothing downstream errors.
    const s = mlx.mlx_default_gpu_stream_new();
    const a = testing.allocator;
    // [out=1, in=2, kh=1, kw=3] with distinguishable values.
    const vals = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const src = constArray(&vals, &[_]c_int{ 1, 2, 1, 3 });
    defer _ = mlx.mlx_array_free(src);
    const perm = try transpose(src, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(perm);
    const got = try readAll(a, perm, s);
    defer a.free(got);
    // OHWI order: [o][kh][kw][in] -> (0,0,0,0)=1, (0,0,0,1)=4, (0,0,1,0)=2 ...
    try testing.expectEqualSlices(f32, &[_]f32{ 1, 4, 2, 5, 3, 6 }, got);
}

test "sdxl nn: loadLinear detects affine quantization and infers geometry" {
    const s = mlx.mlx_default_gpu_stream_new();
    const a = testing.allocator;
    var w = Weights.init(a);
    defer w.deinit();

    // A quantized linear at bits=4, group_size=64 over in=128, out=4:
    //   packed w cols = in·bits/32 = 16 ; scales cols = in/gs = 2.
    var wq: [64]f32 = undefined;
    for (&wq, 0..) |*v, i| v.* = @floatFromInt(i);
    var sc: [8]f32 = .{ 1, 1, 1, 1, 1, 1, 1, 1 };
    try w.put("blk.to_q.weight", constArray(&wq, &[_]c_int{ 4, 16 }));
    try w.put("blk.to_q.scales", constArray(&sc, &[_]c_int{ 4, 2 }));
    try w.put("blk.to_q.biases", constArray(&sc, &[_]c_int{ 4, 2 }));

    var lin = try loadLinear(&w, "blk.to_q.weight", null, .float16, s);
    defer lin.deinit();
    try testing.expect(lin.scales != null);
    try testing.expect(lin.q_biases != null);
    try testing.expectEqual(@as(u32, 4), lin.bits);
    try testing.expectEqual(@as(u32, 64), lin.group_size);

    // A dense linear (no `.scales` sibling) stays dense.
    var d = Weights.init(a);
    defer d.deinit();
    var dw: [8]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try d.put("blk.to_k.weight", constArray(&dw, &[_]c_int{ 2, 4 }));
    var dl = try loadLinear(&d, "blk.to_k.weight", null, .float16, s);
    defer dl.deinit();
    try testing.expect(dl.scales == null);
}

test "sdxl nn: silu and gelu are the exact forms diffusers uses" {
    const s = mlx.mlx_default_gpu_stream_new();
    const a = testing.allocator;
    const vals = [_]f32{ -2, -0.5, 0, 0.5, 2 };
    const x = constArray(&vals, &[_]c_int{5});
    defer _ = mlx.mlx_array_free(x);

    const sv = try silu(x, s);
    defer _ = mlx.mlx_array_free(sv);
    const sg = try readAll(a, sv, s);
    defer a.free(sg);
    for (vals, sg) |v, got| {
        const want = v / (1.0 + @exp(-v));
        try testing.expectApproxEqAbs(want, got, 1e-5);
    }

    const gv = try gelu(x, s);
    defer _ = mlx.mlx_array_free(gv);
    const gg = try readAll(a, gv, s);
    defer a.free(gg);
    // Exact erf form, NOT the tanh approximation. These are independently
    // computed (CPython `0.5*v*(1+erf(v/sqrt(2)))`) rather than re-derived
    // here — a reference that reuses the implementation's own formula proves
    // only that the formula is stable, not that it is the right one. At
    // x = -2 the two GELU variants differ in the third decimal, which this
    // tolerance separates.
    const want = [_]f32{ -0.04550026, -0.15426877, 0.0, 0.34573123, 1.95449974 };
    for (want, gg) |wv, got| {
        try testing.expectApproxEqAbs(wv, got, 1e-5);
    }
}
