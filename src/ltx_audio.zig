//! Native LTX-2.3 AUDIO decode path — Zig + mlx-c port.
//!
//! The LTX "AudioVideo" DiT jointly denoises a video latent AND an audio latent;
//! `ltx_video.zig` decodes only the video and frees the audio latent. This file
//! turns that audio latent `[1, Na, 128]` into a stereo PCM waveform so the
//! generated video can carry sound.
//!
//! Pipeline (mirrors Lightricks/LTX-2 `packages/ltx-core` audio branch):
//!   DiT audio latent [1, Na, 128]
//!     → denormalize (per_channel_statistics) + unpatchify → VAE latent [1, 8, 16, T]
//!     → audio_vae.decoder (2D conv VAE, freq-upsample ×4) → mel [1, 2, 64, T]
//!     → vocoder (BigVGAN: conv_pre, 6× ConvTranspose1d, 18 anti-aliased
//!       snakebeta AMPBlocks, conv_post) → 16 kHz stereo waveform
//!     → (optional) bwe_generator → 48 kHz.
//!
//! Weights live in ONE file `…_audio_vae.safetensors` (loaded via
//! `ltx_video.loadComponent`): `audio_vae.*` (102 tensors, bf16) + `vocoder.*`
//! (1227 tensors, bf16). Weight-norm is PRE-FOLDED into `.weight` (no
//! `.weight_g/.weight_v`). Conv weights are PyTorch-layout and transposed to MLX
//! layout at use (see `conv1d`/`conv2d`/`convTranspose1d`).

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const ltx = @import("ltx_video.zig");

const S = mlx.mlx_stream;

// ════════════════════════════════════════════════════════════════════════
// Conv primitives. The checkpoint stores PyTorch weight layouts; MLX wants
// channels-last input and weight `[C_out, ...spatial, C_in]`. Each helper
// transposes the weight from the stored PyTorch layout to MLX layout.
// ════════════════════════════════════════════════════════════════════════

fn transposeTo(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&out, x, axes.ptr, axes.len, s));
    return out;
}

fn contiguous(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&out, x, false, s));
    return out;
}

fn addBiasLast(out_in: mlx.mlx_array, bias: ?mlx.mlx_array, ndim: usize, s: S) !mlx.mlx_array {
    const b = bias orelse return out_in;
    defer _ = mlx.mlx_array_free(out_in);
    var shape = [_]c_int{ 1, 1, 1, 1 };
    shape[ndim - 1] = mlx.getShape(b)[0];
    const br = try reshape(b, shape[0..ndim], s);
    defer _ = mlx.mlx_array_free(br);
    var ob = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&ob, out_in, br, s));
    return ob;
}

/// 1D conv with an MLX-layout weight `[C_out, k, C_in/groups]` (no transpose).
pub fn conv1dMlx(input: mlx.mlx_array, weight_mlx: mlx.mlx_array, bias: ?mlx.mlx_array, stride: c_int, padding: c_int, dilation: c_int, groups: c_int, s: S) !mlx.mlx_array {
    const w = try contiguous(weight_mlx, s);
    defer _ = mlx.mlx_array_free(w);
    const xc = try contiguous(input, s);
    defer _ = mlx.mlx_array_free(xc);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv1d(&out, xc, w, stride, padding, dilation, groups, s));
    return addBiasLast(out, bias, 3, s);
}

/// 1D transposed conv with an MLX-layout weight `[C_out, k, C_in/groups]`.
pub fn convTranspose1dMlx(input: mlx.mlx_array, weight_mlx: mlx.mlx_array, bias: ?mlx.mlx_array, stride: c_int, padding: c_int, out_padding: c_int, groups: c_int, s: S) !mlx.mlx_array {
    const w = try contiguous(weight_mlx, s);
    defer _ = mlx.mlx_array_free(w);
    const xc = try contiguous(input, s);
    defer _ = mlx.mlx_array_free(xc);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv_transpose1d(&out, xc, w, stride, padding, 1, out_padding, groups, s));
    return addBiasLast(out, bias, 3, s);
}

/// 1D conv. `input` is NLC `[B, L, C_in]`; `weight` is PyTorch Conv1d layout
/// `[C_out, C_in/groups, k]`; `bias` optional `[C_out]`. Returns NLC `[B, L', C_out]`.
pub fn conv1d(input: mlx.mlx_array, weight_pt: mlx.mlx_array, bias: ?mlx.mlx_array, stride: c_int, padding: c_int, dilation: c_int, groups: c_int, s: S) !mlx.mlx_array {
    // PyTorch [C_out, C_in/g, k] → MLX [C_out, k, C_in/g].
    const w = try transposeTo(weight_pt, &[_]c_int{ 0, 2, 1 }, s);
    defer _ = mlx.mlx_array_free(w);
    return conv1dMlx(input, w, bias, stride, padding, dilation, groups, s);
}

/// 1D transposed conv. `input` NLC `[B, L, C_in]`; `weight` PyTorch
/// ConvTranspose1d layout `[C_in, C_out/groups, k]`; `bias` optional `[C_out]`.
pub fn convTranspose1d(input: mlx.mlx_array, weight_pt: mlx.mlx_array, bias: ?mlx.mlx_array, stride: c_int, padding: c_int, out_padding: c_int, groups: c_int, s: S) !mlx.mlx_array {
    // PyTorch ConvTranspose1d weight [C_in, C_out/g, k] → MLX [C_out/g... ] : for
    // groups=1 this is [C_out, k, C_in]. axes {1,2,0}.
    const w = try transposeTo(weight_pt, &[_]c_int{ 1, 2, 0 }, s);
    defer _ = mlx.mlx_array_free(w);
    return convTranspose1dMlx(input, w, bias, stride, padding, out_padding, groups, s);
}

/// 2D conv with an MLX-layout weight `[C_out, kh, kw, C_in]` (no transpose).
/// `input` NHWC `[B, H, W, C_in]`; `bias` optional `[C_out]`.
pub fn conv2dMlx(input: mlx.mlx_array, weight_mlx: mlx.mlx_array, bias: ?mlx.mlx_array, stride: [2]c_int, padding: [2]c_int, s: S) !mlx.mlx_array {
    const w = try contiguous(weight_mlx, s);
    defer _ = mlx.mlx_array_free(w);
    const xc = try contiguous(input, s);
    defer _ = mlx.mlx_array_free(xc);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&out, xc, w, stride[0], stride[1], padding[0], padding[1], 1, 1, 1, s));
    return addBiasLast(out, bias, 4, s);
}

/// 2D conv. `input` NHWC `[B, H, W, C_in]`; `weight` PyTorch Conv2d layout
/// `[C_out, C_in, kh, kw]`; `bias` optional `[C_out]`.
pub fn conv2d(input: mlx.mlx_array, weight_pt: mlx.mlx_array, bias: ?mlx.mlx_array, stride: [2]c_int, padding: [2]c_int, s: S) !mlx.mlx_array {
    // [C_out, C_in, kh, kw] → MLX [C_out, kh, kw, C_in].
    const wt = try transposeTo(weight_pt, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(wt);
    const w = try contiguous(wt, s);
    defer _ = mlx.mlx_array_free(w);
    const xc = try contiguous(input, s);
    defer _ = mlx.mlx_array_free(xc);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&out, xc, w, stride[0], stride[1], padding[0], padding[1], 1, 1, 1, s));
    if (bias) |b| {
        defer _ = mlx.mlx_array_free(out);
        const br = try reshape(b, &[_]c_int{ 1, 1, 1, mlx.getShape(b)[0] }, s);
        defer _ = mlx.mlx_array_free(br);
        var ob = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&ob, out, br, s));
        return ob;
    }
    return out;
}

fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, x, shape.ptr, shape.len, s));
    return out;
}

fn sliceAxis(x: mlx.mlx_array, axis: usize, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var st = [_]c_int{ 0, 0, 0, 0, 0 };
    var sp = [_]c_int{ 0, 0, 0, 0, 0 };
    var str = [_]c_int{ 1, 1, 1, 1, 1 };
    for (0..sh.len) |i| sp[i] = sh[i];
    st[axis] = start;
    sp[axis] = stop;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, st[0..sh.len].ptr, sh.len, sp[0..sh.len].ptr, sh.len, str[0..sh.len].ptr, sh.len, s));
    return out;
}

/// Replicate (edge) padding of `lo`/`hi` elements along `axis` — mirrors
/// PyTorch `F.pad(mode="replicate")` which `mlx_pad` doesn't offer directly.
fn replicatePad(x: mlx.mlx_array, axis: usize, lo: u32, hi: u32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const n = sh[axis];
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var lp: ?mlx.mlx_array = null;
    var hp: ?mlx.mlx_array = null;
    defer if (lp) |a| {
        _ = mlx.mlx_array_free(a);
    };
    defer if (hp) |a| {
        _ = mlx.mlx_array_free(a);
    };
    if (lo > 0) {
        const first = try sliceAxis(x, axis, 0, 1, s);
        defer _ = mlx.mlx_array_free(first);
        var p = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&p, first, @intCast(lo), @intCast(axis), s));
        lp = p;
        _ = mlx.mlx_vector_array_append_value(vec, p);
    }
    _ = mlx.mlx_vector_array_append_value(vec, x);
    if (hi > 0) {
        const last = try sliceAxis(x, axis, n - 1, n, s);
        defer _ = mlx.mlx_array_free(last);
        var p = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&p, last, @intCast(hi), @intCast(axis), s));
        hp = p;
        _ = mlx.mlx_vector_array_append_value(vec, p);
    }
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, @intCast(axis), s));
    return out;
}

fn expFn(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_exp(&out, x, s));
    return out;
}

/// SnakeBeta activation (BigVGAN v2). `x` is NLC `[B, L, C]`; `alpha`/`beta`
/// are raw log-space params `[C]`. Returns `x + (1/(exp(beta)+eps)) * sin(x*exp(alpha))^2`.
pub fn snakeBeta(x: mlx.mlx_array, alpha: mlx.mlx_array, beta: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c = mlx.getShape(alpha)[0];
    const ar = try reshape(alpha, &[_]c_int{ 1, 1, c }, s);
    defer _ = mlx.mlx_array_free(ar);
    const brr = try reshape(beta, &[_]c_int{ 1, 1, c }, s);
    defer _ = mlx.mlx_array_free(brr);
    const ea = try expFn(ar, s);
    defer _ = mlx.mlx_array_free(ea);
    const eb = try expFn(brr, s);
    defer _ = mlx.mlx_array_free(eb);
    // sin(x*ea)^2
    var xa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xa);
    try mlx.check(mlx.mlx_multiply(&xa, x, ea, s));
    var sn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sn);
    try mlx.check(mlx.mlx_sin(&sn, xa, s));
    var sn2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sn2);
    try mlx.check(mlx.mlx_square(&sn2, sn, s));
    // 1/(eb+eps)
    const eps = mlx.mlx_array_new_float(1e-9);
    defer _ = mlx.mlx_array_free(eps);
    var ebe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ebe);
    try mlx.check(mlx.mlx_add(&ebe, eb, eps, s));
    var inv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_divide(&inv, one, ebe, s));
    var term = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(term);
    try mlx.check(mlx.mlx_multiply(&term, sn2, inv, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, x, term, s));
    return out;
}

/// Build a depthwise MLX-layout filter `[C, k, 1]` from a stored `[1, 1, k]`
/// anti-alias kernel (same filter broadcast across all C channels).
fn depthwiseFilter(stored: mlx.mlx_array, c: c_int, s: S) !mlx.mlx_array {
    // Accept either the MLX `[1, k, 1]` (q4 repo) or PyTorch `[1, 1, k]` layout —
    // one of the two non-leading dims is 1, so their product is the kernel size.
    const sh = mlx.getShape(stored);
    const k = sh[1] * sh[2];
    const flat = try reshape(stored, &[_]c_int{ 1, k, 1 }, s);
    defer _ = mlx.mlx_array_free(flat);
    var b = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_broadcast_to(&b, flat, &[_]c_int{ c, k, 1 }, 3, s));
    const out = try contiguous(b, s);
    _ = mlx.mlx_array_free(b);
    return out;
}

/// Anti-aliased activation (BigVGAN `Activation1d`, ratio 2, k=12): upsample ×2
/// (replicate-pad 5, depthwise transposed conv, ×2 scale, crop 15/15) → snakeBeta
/// → downsample ÷2 (replicate-pad 5/6, depthwise strided conv). Length-preserving.
/// `up_filt`/`down_filt` are the stored `[1,1,12]` kernels.
pub fn antiAliasSnakeBeta(x: mlx.mlx_array, alpha: mlx.mlx_array, beta: mlx.mlx_array, up_filt: mlx.mlx_array, down_filt: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c = mlx.getShape(x)[2];
    // ── upsample ×2 ──
    const up_w = try depthwiseFilter(up_filt, c, s);
    defer _ = mlx.mlx_array_free(up_w);
    const padded = try replicatePad(x, 1, 5, 5, s);
    defer _ = mlx.mlx_array_free(padded);
    var ct = try convTranspose1dMlx(padded, up_w, null, 2, 0, 0, c, s);
    // ×ratio(2)
    {
        const two = mlx.mlx_array_new_float(2.0);
        defer _ = mlx.mlx_array_free(two);
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, ct, two, s));
        _ = mlx.mlx_array_free(ct);
        ct = scaled;
    }
    // crop [15:-15] on L
    const lup = mlx.getShape(ct)[1];
    const cropped = try sliceAxis(ct, 1, 15, lup - 15, s);
    _ = mlx.mlx_array_free(ct);
    defer _ = mlx.mlx_array_free(cropped);
    const cc = try contiguous(cropped, s);
    defer _ = mlx.mlx_array_free(cc);
    // ── activation ──
    const act = try snakeBeta(cc, alpha, beta, s);
    defer _ = mlx.mlx_array_free(act);
    // ── downsample ÷2 ──
    const down_w = try depthwiseFilter(down_filt, c, s);
    defer _ = mlx.mlx_array_free(down_w);
    const dpad = try replicatePad(act, 1, 5, 6, s);
    defer _ = mlx.mlx_array_free(dpad);
    return conv1dMlx(dpad, down_w, null, 2, 0, 1, c, s);
}

fn siluA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, x, sig, s));
    return out;
}

fn clamp(x: mlx.mlx_array, lo: f32, hi: f32, s: S) !mlx.mlx_array {
    const loa = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(loa);
    const hia = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hia);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_maximum(&t, x, loa, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_minimum(&out, t, hia, s));
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Audio VAE decoder (taming/LDM-style 2D conv decoder over mel spectrograms).
// Works in NHWC `[B, H=time, W=freq, C]`. causality_axis = HEIGHT (time):
// CausalConv2d pads time (2,0) and freq (1,1) for k=3, then conv pad=0.
// Norm = PixelNorm over channels (last axis), eps 1e-6.
// ════════════════════════════════════════════════════════════════════════

fn freeReplace(x: *mlx.mlx_array, new: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(x.*);
    x.* = new;
}

fn dbg(tag: []const u8, x: mlx.mlx_array) void {
    if (std.c.getenv("LTX_AUDIO_TRACE") == null) return;
    const sh = mlx.getShape(x);
    std.debug.print("[ltx-audio-trace] {s}: ndim={d} shape=", .{ tag, sh.len });
    for (sh) |d| std.debug.print("{d},", .{d});
    std.debug.print("\n", .{});
}

/// PixelNorm over channels (last axis), fp32-accurate via fused rms_norm with a
/// ones weight (mlx_fast_rms_norm crashes on a null weight).
fn pixelNorm(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const c = sh[sh.len - 1];
    const one_val = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one_val);
    var ones_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones_w);
    try mlx.check(mlx.mlx_full(&ones_w, &[_]c_int{c}, 1, one_val, .bfloat16, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, ones_w, 1e-6, s));
    return out;
}

fn key(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

/// Causal 2D conv (HEIGHT/time axis) by weight base key. `x` NHWC; weight is the
/// q4 MLX layout `[O, kh, kw, I]`. For k=1 (nin_shortcut) no padding is applied.
fn causalConv2d(comp: *const ltx.Component, base: []const u8, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var wbuf: [256]u8 = undefined;
    var bbuf: [256]u8 = undefined;
    const w = comp.get(key(&wbuf, "{s}.weight", .{base})) orelse return error.MissingAudioWeight;
    const b = comp.get(key(&bbuf, "{s}.bias", .{base}));
    const kh = mlx.getShape(w)[1]; // MLX [O, kh, kw, I]
    if (kh == 1) return conv2dMlx(x, w, b, .{ 1, 1 }, .{ 0, 0 }, s);
    // HEIGHT-causal pad: time(axis1) lo=kh-1 hi=0; freq(axis2) lo=kw/2 hi=kw/2.
    const kw = mlx.getShape(w)[2];
    const axes = [_]c_int{ 1, 2 };
    const lo = [_]c_int{ kh - 1, @divFloor(kw, 2) };
    const hi = [_]c_int{ 0, @divFloor(kw, 2) };
    const zero = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(zero);
    var padded = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_pad(&padded, x, &axes, 2, &lo, 2, &hi, 2, zero, "constant", s));
    defer _ = mlx.mlx_array_free(padded);
    return conv2dMlx(padded, w, b, .{ 1, 1 }, .{ 0, 0 }, s);
}

/// Pre-activation ResnetBlock: h = conv2(silu(pn(conv1(silu(pn(x)))))); shortcut
/// = nin_shortcut(x) if present; return shortcut + h.
fn resnetBlock2d(comp: *const ltx.Component, base: []const u8, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var b1: [256]u8 = undefined;
    const pn1 = try pixelNorm(x, s);
    defer _ = mlx.mlx_array_free(pn1);
    const a1 = try siluA(pn1, s);
    defer _ = mlx.mlx_array_free(a1);
    const c1 = try causalConv2d(comp, key(&b1, "{s}.conv1.conv", .{base}), a1, s);
    defer _ = mlx.mlx_array_free(c1);
    const pn2 = try pixelNorm(c1, s);
    defer _ = mlx.mlx_array_free(pn2);
    const a2 = try siluA(pn2, s);
    defer _ = mlx.mlx_array_free(a2);
    var b2: [256]u8 = undefined;
    const c2 = try causalConv2d(comp, key(&b2, "{s}.conv2.conv", .{base}), a2, s);
    defer _ = mlx.mlx_array_free(c2);
    // shortcut
    var sk: [256]u8 = undefined;
    var skbuf: [256]u8 = undefined;
    const shortcut: mlx.mlx_array = if (comp.get(key(&skbuf, "{s}.nin_shortcut.conv.weight", .{base})) != null)
        try causalConv2d(comp, key(&sk, "{s}.nin_shortcut.conv", .{base}), x, s)
    else blk: {
        var cp = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&cp, x));
        break :blk cp;
    };
    defer _ = mlx.mlx_array_free(shortcut);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, shortcut, c2, s));
    return out;
}

/// Nearest ×2 upsample on time(axis1) AND freq(axis2), then causal 3×3 conv, then
/// drop the first time frame (causal). Matches `Upsample` (HEIGHT axis).
fn vaeUpsample(comp: *const ltx.Component, base: []const u8, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var up = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_repeat_axis(&up, x, 2, 1, s));
    var up2 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_repeat_axis(&up2, up, 2, 2, s));
    _ = mlx.mlx_array_free(up);
    const upc = try contiguous(up2, s);
    _ = mlx.mlx_array_free(up2);
    defer _ = mlx.mlx_array_free(upc);
    var cb: [256]u8 = undefined;
    const conv = try causalConv2d(comp, key(&cb, "{s}.conv.conv", .{base}), upc, s);
    defer _ = mlx.mlx_array_free(conv);
    // drop first time frame
    const ht = mlx.getShape(conv)[1];
    const cropped = try sliceAxis(conv, 1, 1, ht, s);
    defer _ = mlx.mlx_array_free(cropped);
    return contiguous(cropped, s);
}

/// Decode a DiT audio latent `[1, Na, 128]` → stereo mel `[1, 2, T, 64]` (NCHW).
pub fn audioVaeDecode(comp: *const ltx.Component, latent: mlx.mlx_array, s: S) !mlx.mlx_array {
    const na = mlx.getShape(latent)[1];
    // ── denormalize on the [1,Na,128] patchified latent ──
    // Key spelling differs by source: q4 repo uses `_mean_of_means`/`_std_of_means`;
    // the distilled checkpoint uses the hyphenated `mean-of-means`/`std-of-means`.
    const mean = comp.get("audio_vae.per_channel_statistics._mean_of_means") orelse
        comp.get("audio_vae.per_channel_statistics.mean-of-means") orelse return error.MissingAudioWeight;
    const std_ = comp.get("audio_vae.per_channel_statistics._std_of_means") orelse
        comp.get("audio_vae.per_channel_statistics.std-of-means") orelse return error.MissingAudioWeight;
    const mr = try reshape(mean, &[_]c_int{ 1, 1, 128 }, s);
    defer _ = mlx.mlx_array_free(mr);
    const sr = try reshape(std_, &[_]c_int{ 1, 1, 128 }, s);
    defer _ = mlx.mlx_array_free(sr);
    var x = mlx.mlx_array_new();
    {
        var xs = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&xs, latent, sr, s));
        try mlx.check(mlx.mlx_add(&x, xs, mr, s));
        _ = mlx.mlx_array_free(xs);
    }
    _ = mlx.mlx_array_eval(x);
    dbg("after denorm", x);
    // ── unpatchify [1,Na,128] → NHWC [1, Na(time), 16(freq), 8(ch)] ──
    // 128 = c(8) outer × f(16) inner ("b t (c f) -> b c t f"); NHWC wants C last.
    {
        const r = try reshape(x, &[_]c_int{ 1, na, 8, 16 }, s); // b t c f
        freeReplace(&x, r);
        const t = try transposeTo(x, &[_]c_int{ 0, 1, 3, 2 }, s); // b t f c
        freeReplace(&x, t);
        const cc = try contiguous(x, s);
        freeReplace(&x, cc);
    }
    _ = mlx.mlx_array_eval(x);
    dbg("after unpatchify", x);
    // ── conv_in (8→512) ──
    {
        const nx = try causalConv2d(comp, "audio_vae.decoder.conv_in.conv", x, s);
        freeReplace(&x, nx);
    }
    _ = mlx.mlx_array_eval(x);
    dbg("after conv_in", x);
    // ── mid: block_1, block_2 (no attn; attn_resolutions empty) ──
    inline for (.{ "block_1", "block_2" }) |blk| {
        const nx = try resnetBlock2d(comp, "audio_vae.decoder.mid." ++ blk, x, s);
        freeReplace(&x, nx);
        _ = mlx.mlx_array_eval(x);
        dbg("after mid." ++ blk, x);
    }
    // ── up path: level 2 → 1 → 0; each 3 res blocks; upsample if level != 0 ──
    var level: i32 = 2;
    while (level >= 0) : (level -= 1) {
        var blk: u32 = 0;
        while (blk < 3) : (blk += 1) {
            var bb: [256]u8 = undefined;
            const nx = try resnetBlock2d(comp, key(&bb, "audio_vae.decoder.up.{d}.block.{d}", .{ level, blk }), x, s);
            freeReplace(&x, nx);
        }
        _ = mlx.mlx_array_eval(x);
        dbg("after up blocks (level)", x);
        if (level != 0) {
            var ub: [256]u8 = undefined;
            const nx = try vaeUpsample(comp, key(&ub, "audio_vae.decoder.up.{d}.upsample", .{level}), x, s);
            freeReplace(&x, nx);
        }
        _ = mlx.mlx_array_eval(x);
        dbg("after up level done", x);
    }
    // ── norm_out + silu + conv_out (→2) ──
    {
        const pn = try pixelNorm(x, s);
        freeReplace(&x, pn);
        const a = try siluA(x, s);
        freeReplace(&x, a);
        const co = try causalConv2d(comp, "audio_vae.decoder.conv_out.conv", x, s);
        freeReplace(&x, co);
    }
    // NHWC [1, T, 64, 2] → NCHW [1, 2, T, 64].
    const t = try transposeTo(x, &[_]c_int{ 0, 3, 1, 2 }, s);
    freeReplace(&x, t);
    const out = try contiguous(x, s);
    freeReplace(&x, out);
    _ = mlx.mlx_array_eval(x);
    return x;
}

// ════════════════════════════════════════════════════════════════════════
// Vocoder (BigVGAN v2). mel `[1, 2, T, 64]` (NCHW) → stereo waveform. Works in
// NLC `[B, L, C]`. conv_pre 128→1536; 6 ConvTranspose1d upsamples
// (rates 5,2,2,2,2,2 = ×160); 18 AMPBlocks (6 stages × kernels 3,7,11) averaged
// per stage; anti-aliased snakeBeta act_post; conv_post→2; clamp(-1,1).
// ════════════════════════════════════════════════════════════════════════

const VOC = "vocoder";
const up_rates = [_]c_int{ 5, 2, 2, 2, 2, 2 };
const res_kernels = [_]c_int{ 3, 7, 11 };
const res_dils = [_]c_int{ 1, 3, 5 };

/// Anti-aliased snakeBeta keyed by the act base (`…act`, `…upsample`, `…downsample.lowpass`).
fn actByKey(comp: *const ltx.Component, base: []const u8, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var ab: [256]u8 = undefined;
    var bb: [256]u8 = undefined;
    var ub: [256]u8 = undefined;
    var db: [256]u8 = undefined;
    // A wrong-keyed but loadable checkpoint must surface as an error the
    // caller downgrades to "video stays silent" — never a `.?` panic that
    // kills the whole unified server (handleVideo catches errors, not panics).
    const alpha = comp.get(key(&ab, "{s}.act.alpha", .{base})) orelse return error.MissingAudioWeight;
    const beta = comp.get(key(&bb, "{s}.act.beta", .{base})) orelse return error.MissingAudioWeight;
    const uf = comp.get(key(&ub, "{s}.upsample.filter", .{base})) orelse return error.MissingAudioWeight;
    const df = comp.get(key(&db, "{s}.downsample.lowpass.filter", .{base})) orelse return error.MissingAudioWeight;
    return antiAliasSnakeBeta(x, alpha, beta, uf, df, s);
}

/// 1D conv by weight base key (`…weight`/`…bias`), q4 MLX layout `[O, k, I]`.
fn conv1dByKey(comp: *const ltx.Component, base: []const u8, x: mlx.mlx_array, stride: c_int, padding: c_int, dilation: c_int, s: S) !mlx.mlx_array {
    var wb: [256]u8 = undefined;
    var bb: [256]u8 = undefined;
    const w = comp.get(key(&wb, "{s}.weight", .{base})) orelse return error.MissingAudioWeight;
    const b = comp.get(key(&bb, "{s}.bias", .{base}));
    return conv1dMlx(x, w, b, stride, padding, dilation, 1, s);
}

/// One AMPBlock1: 3 sublayers (dil 1,3,5), each act→conv1(dil)→act→conv2(dil1)→residual.
fn ampBlock(comp: *const ltx.Component, idx: u32, kernel: c_int, x_in: mlx.mlx_array, s: S) !mlx.mlx_array {
    var x = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&x, x_in));
    var j: u32 = 0;
    while (j < 3) : (j += 1) {
        var ab: [256]u8 = undefined;
        var cb: [256]u8 = undefined;
        const a1 = try actByKey(comp, key(&ab, "{s}.resblocks.{d}.acts1.{d}", .{ VOC, idx, j }), x, s);
        defer _ = mlx.mlx_array_free(a1);
        const dil = res_dils[j];
        const pad1 = @divFloor(dil * (kernel - 1), 2);
        const c1 = try conv1dByKey(comp, key(&cb, "{s}.resblocks.{d}.convs1.{d}", .{ VOC, idx, j }), a1, 1, pad1, dil, s);
        defer _ = mlx.mlx_array_free(c1);
        var ab2: [256]u8 = undefined;
        var cb2: [256]u8 = undefined;
        const a2 = try actByKey(comp, key(&ab2, "{s}.resblocks.{d}.acts2.{d}", .{ VOC, idx, j }), c1, s);
        defer _ = mlx.mlx_array_free(a2);
        const pad2 = @divFloor(kernel - 1, 2);
        const c2 = try conv1dByKey(comp, key(&cb2, "{s}.resblocks.{d}.convs2.{d}", .{ VOC, idx, j }), a2, 1, pad2, 1, s);
        defer _ = mlx.mlx_array_free(c2);
        var nx = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&nx, x, c2, s));
        freeReplace(&x, nx);
    }
    return x;
}

/// Vocode stereo mel `[1, 2, T, 64]` (NCHW) → waveform NLC `[1, L, 2]`.
pub fn vocode(comp: *const ltx.Component, mel_nchw: mlx.mlx_array, s: S) !mlx.mlx_array {
    const T = mlx.getShape(mel_nchw)[2];
    // NCHW [1,2,T,64] → NLC [1, T, 128] with channel = stereo*64 + freq.
    var x = mlx.mlx_array_new();
    {
        const p = try transposeTo(mel_nchw, &[_]c_int{ 0, 2, 1, 3 }, s); // b t s f
        defer _ = mlx.mlx_array_free(p);
        const pc = try contiguous(p, s);
        defer _ = mlx.mlx_array_free(pc);
        x = try reshape(pc, &[_]c_int{ 1, T, 128 }, s);
    }
    // conv_pre (128→1536, k7, pad3)
    {
        const nx = try conv1dByKey(comp, VOC ++ ".conv_pre", x, 1, 3, 1, s);
        freeReplace(&x, nx);
    }
    // upsample stages
    var i: u32 = 0;
    while (i < up_rates.len) : (i += 1) {
        var ub: [256]u8 = undefined;
        const wkey = key(&ub, "{s}.ups.{d}", .{ VOC, i });
        var wb: [256]u8 = undefined;
        var bb: [256]u8 = undefined;
        const w = comp.get(key(&wb, "{s}.weight", .{wkey})) orelse return error.MissingAudioWeight;
        const b = comp.get(key(&bb, "{s}.bias", .{wkey}));
        const kc = mlx.getShape(w)[1]; // q4 MLX ConvTranspose [Cout, k, Cin]
        const stride = up_rates[i];
        const pad = @divFloor(kc - stride, 2);
        {
            const nx = try convTranspose1dMlx(x, w, b, stride, pad, 0, 1, s);
            freeReplace(&x, nx);
        }
        // 3 resblocks (kernels 3,7,11) averaged
        var acc: ?mlx.mlx_array = null;
        var kidx: u32 = 0;
        while (kidx < 3) : (kidx += 1) {
            const rb_idx = i * 3 + kidx;
            const rb = try ampBlock(comp, rb_idx, res_kernels[kidx], x, s);
            if (acc) |a| {
                var sum = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&sum, a, rb, s));
                _ = mlx.mlx_array_free(a);
                _ = mlx.mlx_array_free(rb);
                acc = sum;
            } else acc = rb;
        }
        // mean = acc / 3
        const three = mlx.mlx_array_new_float(3.0);
        defer _ = mlx.mlx_array_free(three);
        var mean = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_divide(&mean, acc.?, three, s));
        _ = mlx.mlx_array_free(acc.?);
        freeReplace(&x, mean);
        _ = mlx.mlx_array_eval(x);
    }
    // act_post + conv_post + clamp
    {
        const ap = try actByKey(comp, VOC ++ ".act_post", x, s);
        freeReplace(&x, ap);
        const cp = try conv1dByKey(comp, VOC ++ ".conv_post", x, 1, 3, 1, s);
        freeReplace(&x, cp);
        const cl = try clamp(x, -1.0, 1.0, s);
        freeReplace(&x, cl);
    }
    _ = mlx.mlx_array_eval(x);
    return x; // [1, L, 2]
}

/// Full audio decode: DiT latent `[1, Na, 128]` → interleaved stereo PCM f32 in
/// `[-1, 1]`, length `L*2` (L frames, 2 channels), at 16 kHz. Caller owns the slice.
pub const Waveform = struct {
    pcm: []f32, // interleaved L*R
    frames: u32,
    channels: u32 = 2,
    sample_rate: u32 = 16000,
    pub fn deinit(self: *Waveform, alloc: std.mem.Allocator) void {
        alloc.free(self.pcm);
    }
};

pub fn decodeAudio(alloc: std.mem.Allocator, comp: *const ltx.Component, latent: mlx.mlx_array, s: S) !Waveform {
    const mel = try audioVaeDecode(comp, latent, s);
    defer _ = mlx.mlx_array_free(mel);
    const wav = try vocode(comp, mel, s); // [1, L, 2]
    defer _ = mlx.mlx_array_free(wav);
    var f32a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32a);
    try mlx.check(mlx.mlx_astype(&f32a, wav, .float32, s));
    _ = mlx.mlx_array_eval(f32a);
    const sh = mlx.getShape(wav); // [1, L, 2]
    const frames: usize = @intCast(sh[1]);
    const n = frames * 2;
    const ptr = mlx.mlx_array_data_float32(f32a) orelse return error.AudioDecodeFailed;
    const pcm = try alloc.alloc(f32, n);
    @memcpy(pcm, ptr[0..n]);
    return .{ .pcm = pcm, .frames = @intCast(frames) };
}

/// Load + merge the q4 audio model: `audio_vae.safetensors` (`audio_vae.*`) +
/// `vocoder.safetensors` (`vocoder.*`) into ONE Component, then materialize all
/// weights (the conv graph needs real, not lazy-load, arrays). Both files are the
/// MLX-layout q4 format from `dgrauet/ltx-2.3-mlx-q4`.
pub fn loadAudioComponents(allocator: std.mem.Allocator, audio_vae_path: [:0]const u8, vocoder_path: [:0]const u8, s: S) !ltx.Component {
    var comp = try ltx.loadComponent(allocator, audio_vae_path, s);
    errdefer comp.deinit();
    var voc = try ltx.loadComponent(allocator, vocoder_path, s);
    // Transfer voc's entries into comp (ownership of keys + arrays moves), then
    // free voc's hashmap storage only — NOT the keys/arrays (now owned by comp).
    var it = voc.map.iterator();
    while (it.next()) |e| try comp.map.put(e.key_ptr.*, e.value_ptr.*);
    voc.map.deinit();
    var it2 = comp.map.iterator();
    while (it2.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*);
    return comp;
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

fn arange(shape: []const c_int, s: S) !mlx.mlx_array {
    var n: c_int = 1;
    for (shape) |d| n *= d;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_arange(&flat, 0.0, @floatFromInt(n), 1.0, .float32, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, flat, shape.ptr, shape.len, s));
    return out;
}

test "conv1d output length matches stride/pad/kernel" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // input [B=1, L=10, C_in=3]; weight PyTorch [C_out=4, C_in=3, k=3]
    const x = try arange(&[_]c_int{ 1, 10, 3 }, s);
    defer _ = mlx.mlx_array_free(x);
    const w = try arange(&[_]c_int{ 4, 3, 3 }, s);
    defer _ = mlx.mlx_array_free(w);
    const y = try conv1d(x, w, null, 1, 1, 1, 1, s); // pad=1, k=3, stride=1 → L unchanged
    defer _ = mlx.mlx_array_free(y);
    _ = mlx.mlx_array_eval(y);
    const sh = mlx.getShape(y);
    try testing.expectEqual(@as(c_int, 1), sh[0]);
    try testing.expectEqual(@as(c_int, 10), sh[1]); // (10 + 2*1 - 3)/1 + 1 = 10
    try testing.expectEqual(@as(c_int, 4), sh[2]);
}

test "convTranspose1d upsamples length by stride" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // input [1, L=8, C_in=6]; weight PyTorch ConvTranspose [C_in=6, C_out=4, k=4], stride 2
    const x = try arange(&[_]c_int{ 1, 8, 6 }, s);
    defer _ = mlx.mlx_array_free(x);
    const w = try arange(&[_]c_int{ 6, 4, 4 }, s);
    defer _ = mlx.mlx_array_free(w);
    const y = try convTranspose1d(x, w, null, 2, 0, 0, 1, s);
    defer _ = mlx.mlx_array_free(y);
    _ = mlx.mlx_array_eval(y);
    const sh = mlx.getShape(y);
    // L_out = (L-1)*stride - 2*pad + k + out_pad = 7*2 + 4 = 18
    try testing.expectEqual(@as(c_int, 1), sh[0]);
    try testing.expectEqual(@as(c_int, 18), sh[1]);
    try testing.expectEqual(@as(c_int, 4), sh[2]);
}

fn evalRead(arr: mlx.mlx_array, s: S) []const f32 {
    _ = mlx.mlx_array_eval(arr);
    var f = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&f, arr, .float32, s);
    _ = mlx.mlx_array_eval(f);
    const n: usize = @intCast(blk: {
        var p: c_int = 1;
        for (mlx.getShape(arr)) |d| p *= d;
        break :blk p;
    });
    const ptr = mlx.mlx_array_data_float32(f).?;
    return ptr[0..n];
}

fn lit(vals: []const f32, shape: []const c_int, s: S) !mlx.mlx_array {
    _ = s;
    return mlx.mlx_array_new_data(vals.ptr, shape.ptr, @intCast(shape.len), .float32);
}

test "snakeBeta matches reference formula" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // x=0.5, alpha=beta=0 → exp=1 → 0.5 + sin(0.5)^2
    const x = try lit(&[_]f32{0.5}, &[_]c_int{ 1, 1, 1 }, s);
    defer _ = mlx.mlx_array_free(x);
    const a = try lit(&[_]f32{0.0}, &[_]c_int{1}, s);
    defer _ = mlx.mlx_array_free(a);
    const b = try lit(&[_]f32{0.0}, &[_]c_int{1}, s);
    defer _ = mlx.mlx_array_free(b);
    const y = try snakeBeta(x, a, b, s);
    defer _ = mlx.mlx_array_free(y);
    const got = evalRead(y, s)[0];
    const want: f32 = 0.5 + std.math.pow(f32, @sin(0.5), 2.0);
    try testing.expectApproxEqAbs(want, got, 1e-5);
}

test "conv1dMlx depthwise keeps channels independent" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // input [1, L=4, C=2]: ch0=1s, ch1=10s (NLC interleaved per position)
    const x = try lit(&[_]f32{ 1, 10, 1, 10, 1, 10, 1, 10 }, &[_]c_int{ 1, 4, 2 }, s);
    defer _ = mlx.mlx_array_free(x);
    // MLX weight [C_out=2, k=1, C_in/groups=1]: ch0×2, ch1×3
    const w = try lit(&[_]f32{ 2, 3 }, &[_]c_int{ 2, 1, 1 }, s);
    defer _ = mlx.mlx_array_free(w);
    const y = try conv1dMlx(x, w, null, 1, 0, 1, 2, s);
    defer _ = mlx.mlx_array_free(y);
    const d = evalRead(y, s);
    // shape [1,4,2]; ch0 should be 2, ch1 should be 30 at every position
    try testing.expectApproxEqAbs(@as(f32, 2.0), d[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 30.0), d[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 2.0), d[6], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 30.0), d[7], 1e-4);
}

test "antiAliasSnakeBeta is length-preserving" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const x = try arange(&[_]c_int{ 1, 20, 4 }, s);
    defer _ = mlx.mlx_array_free(x);
    const a = try lit(&[_]f32{ 0, 0, 0, 0 }, &[_]c_int{4}, s);
    defer _ = mlx.mlx_array_free(a);
    const b = try lit(&[_]f32{ 0, 0, 0, 0 }, &[_]c_int{4}, s);
    defer _ = mlx.mlx_array_free(b);
    // dummy normalized-ish filters [1,1,12]
    var fv: [12]f32 = undefined;
    for (&fv) |*v| v.* = 1.0 / 12.0;
    const uf = try lit(&fv, &[_]c_int{ 1, 1, 12 }, s);
    defer _ = mlx.mlx_array_free(uf);
    const df = try lit(&fv, &[_]c_int{ 1, 1, 12 }, s);
    defer _ = mlx.mlx_array_free(df);
    const y = try antiAliasSnakeBeta(x, a, b, uf, df, s);
    defer _ = mlx.mlx_array_free(y);
    _ = mlx.mlx_array_eval(y);
    const sh = mlx.getShape(y);
    try testing.expectEqual(@as(c_int, 1), sh[0]);
    try testing.expectEqual(@as(c_int, 20), sh[1]);
    try testing.expectEqual(@as(c_int, 4), sh[2]);
}

// Weights-gated end-to-end structural test (no Python oracle needed). Point
// LTX_AUDIO_TEST_MODEL at the q4 model dir (with audio_vae.safetensors +
// vocoder.safetensors).
fn loadAudioComp(alloc: std.mem.Allocator) !?ltx.Component {
    const raw = std.c.getenv("LTX_AUDIO_TEST_MODEL") orelse return null;
    const dir = std.mem.span(raw);
    var ap: [1024]u8 = undefined;
    var vp: [1024]u8 = undefined;
    const audio_path = try std.fmt.bufPrintZ(&ap, "{s}/audio_vae.safetensors", .{dir});
    const voc_path = try std.fmt.bufPrintZ(&vp, "{s}/vocoder.safetensors", .{dir});
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    return try loadAudioComponents(alloc, audio_path, voc_path, cpu_s);
}

test "audio VAE + vocoder end-to-end: shape, finiteness, bounded, exact length" {
    const alloc = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    var comp = (try loadAudioComp(alloc)) orelse return error.SkipZigTest;
    defer comp.deinit();

    // Synthetic latent [1, Na, 128] (bf16) — structured (not constant) so output varies.
    const na: c_int = 8;
    var buf: [8 * 128]f32 = undefined;
    for (0..@intCast(na)) |t| {
        for (0..128) |c| {
            buf[t * 128 + c] = 0.3 * @sin(@as(f32, @floatFromInt(t * 7 + c)) * 0.05);
        }
    }
    const lf = mlx.mlx_array_new_data(&buf, &[_]c_int{ 1, na, 128 }, 3, .float32);
    defer _ = mlx.mlx_array_free(lf);
    var latent = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(latent);
    try mlx.check(mlx.mlx_astype(&latent, lf, .bfloat16, s));

    // mel intermediate shape: [1, 2, 4*Na-3, 64]
    const mel = try audioVaeDecode(&comp, latent, s);
    defer _ = mlx.mlx_array_free(mel);
    const msh = mlx.getShape(mel);
    try testing.expectEqual(@as(c_int, 1), msh[0]);
    try testing.expectEqual(@as(c_int, 2), msh[1]);
    try testing.expectEqual(@as(c_int, 4 * na - 3), msh[2]);
    try testing.expectEqual(@as(c_int, 64), msh[3]);

    var wav = try decodeAudio(alloc, &comp, latent, s);
    defer wav.deinit(alloc);
    // exact length: mel_T * 160 (hop), interleaved ×2 channels
    const expect_frames: u32 = @intCast((4 * na - 3) * 160);
    try testing.expectEqual(expect_frames, wav.frames);
    try testing.expectEqual(@as(usize, expect_frames * 2), wav.pcm.len);
    try testing.expectEqual(@as(u32, 16000), wav.sample_rate);

    var max_abs: f32 = 0;
    var sumsq: f64 = 0;
    for (wav.pcm) |v| {
        try testing.expect(std.math.isFinite(v));
        try testing.expect(v >= -1.0 and v <= 1.0);
        max_abs = @max(max_abs, @abs(v));
        sumsq += @as(f64, v) * @as(f64, v);
    }
    // not pure silence (clamp keeps [-1,1]; structured latent must produce signal)
    try testing.expect(max_abs > 1e-4);
    const rms = @sqrt(sumsq / @as(f64, @floatFromInt(wav.pcm.len)));
    std.debug.print("[ltx-audio] frames={d} rms={d:.5} max={d:.4}\n", .{ wav.frames, rms, max_abs });
}

// ── Reference parity (cos) oracles — USER-RUN, gated. Fixtures come from
// tests/dump_ltx_audio_fixtures.py (needs torch + the LTX-2 reference). Each is
// a flat little-endian f32 `.raw`. Skips cleanly when env vars are unset.
fn readRawF32(io: std.Io, alloc: std.mem.Allocator, path: []const u8) ![]f32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(alloc, .limited(1024 * 1024 * 1024));
    defer alloc.free(bytes);
    const n = bytes.len / 4;
    const out = try alloc.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}

fn cosine(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    const n = @min(a.len, b.len);
    for (0..n) |i| {
        dot += @as(f64, a[i]) * b[i];
        na += @as(f64, a[i]) * a[i];
        nb += @as(f64, b[i]) * b[i];
    }
    return dot / (std.math.sqrt(na) * std.math.sqrt(nb));
}

fn bf16From(data: []const f32, shape: []const c_int, s: S) !mlx.mlx_array {
    const f = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

test "audio VAE decode matches the reference mel (cos)" {
    const alloc = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    const lat_p = std.mem.span(std.c.getenv("LTX_AUDIO_LATENT") orelse return error.SkipZigTest);
    const mel_p = std.mem.span(std.c.getenv("LTX_AUDIO_MEL") orelse return error.SkipZigTest);
    const io = std.Io.Threaded.global_single_threaded.io();
    var comp = (try loadAudioComp(alloc)) orelse return error.SkipZigTest;
    defer comp.deinit();

    const lat = try readRawF32(io, alloc, lat_p);
    defer alloc.free(lat);
    const ref_mel = try readRawF32(io, alloc, mel_p);
    defer alloc.free(ref_mel);
    const na: c_int = @intCast(lat.len / 128);
    const latent = try bf16From(lat, &[_]c_int{ 1, na, 128 }, s);
    defer _ = mlx.mlx_array_free(latent);

    const mel = try audioVaeDecode(&comp, latent, s);
    defer _ = mlx.mlx_array_free(mel);
    const got = evalRead(mel, s);
    const cos = cosine(got, ref_mel);
    const min: f64 = if (std.c.getenv("LTX_AUDIO_MEL_MIN")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.99) else 0.99;
    std.debug.print("[ltx-audio] VAE mel cos={d:.5} (min {d:.3}), n={d} vs {d}\n", .{ cos, min, got.len, ref_mel.len });
    try testing.expect(cos > min);
}

test "vocoder matches the reference waveform (cos)" {
    const alloc = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    const mel_p = std.mem.span(std.c.getenv("LTX_AUDIO_MEL") orelse return error.SkipZigTest);
    const wave_p = std.mem.span(std.c.getenv("LTX_AUDIO_WAVE") orelse return error.SkipZigTest);
    const io = std.Io.Threaded.global_single_threaded.io();
    var comp = (try loadAudioComp(alloc)) orelse return error.SkipZigTest;
    defer comp.deinit();

    const mel = try readRawF32(io, alloc, mel_p);
    defer alloc.free(mel);
    const ref_wave = try readRawF32(io, alloc, wave_p);
    defer alloc.free(ref_wave);
    const T: c_int = @intCast(mel.len / (2 * 64)); // [1,2,T,64]
    const mel_a = try bf16From(mel, &[_]c_int{ 1, 2, T, 64 }, s);
    defer _ = mlx.mlx_array_free(mel_a);

    const wav = try vocode(&comp, mel_a, s); // [1, L, 2] interleaved L,R,L,R…
    defer _ = mlx.mlx_array_free(wav);
    const got = evalRead(wav, s);
    // The dump script saves the reference waveform in the SAME interleaved [L,2]
    // layout (see tests/dump_ltx_audio_fixtures.py), so cosine is order-aligned.
    const cos = cosine(got, ref_wave);
    const min: f64 = if (std.c.getenv("LTX_AUDIO_WAVE_MIN")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.9) else 0.9;
    std.debug.print("[ltx-audio] vocoder wave cos={d:.5} (min {d:.3}), n={d} vs {d}\n", .{ cos, min, got.len, ref_wave.len });
    try testing.expect(cos > min);
}

test "conv2d valid shape with PyTorch-layout weight" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // input NHWC [1, H=16, W=20, C_in=8]; weight PyTorch [C_out=512, C_in=8, kh=3, kw=3]
    const x = try arange(&[_]c_int{ 1, 16, 20, 8 }, s);
    defer _ = mlx.mlx_array_free(x);
    const w = try arange(&[_]c_int{ 512, 8, 3, 3 }, s);
    defer _ = mlx.mlx_array_free(w);
    const y = try conv2d(x, w, null, .{ 1, 1 }, .{ 1, 1 }, s);
    defer _ = mlx.mlx_array_free(y);
    _ = mlx.mlx_array_eval(y);
    const sh = mlx.getShape(y);
    try testing.expectEqual(@as(c_int, 1), sh[0]);
    try testing.expectEqual(@as(c_int, 16), sh[1]);
    try testing.expectEqual(@as(c_int, 20), sh[2]);
    try testing.expectEqual(@as(c_int, 512), sh[3]);
}

test "audioVaeDecode surfaces MissingAudioWeight instead of panicking on a wrong-keyed checkpoint" {
    // A loadable-but-foreign safetensors passes gen.zig's load gate; request-time
    // lookups must ERROR (handleVideo downgrades to a silent video), never `.?`.
    const allocator = std.testing.allocator;
    var comp = ltx.Component{ .map = std.StringHashMap(mlx.mlx_array).init(allocator), .allocator = allocator };
    defer comp.deinit();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const vals = [_]f32{0.0} ** 128;
    const latent = mlx.mlx_array_new_data(&vals, &[_]c_int{ 1, 1, 128 }, 3, .float32);
    defer _ = mlx.mlx_array_free(latent);
    try std.testing.expectError(error.MissingAudioWeight, audioVaeDecode(&comp, latent, s));

    // vocode's first conv lookup takes the same path.
    const mel = mlx.mlx_array_new_data(&vals, &[_]c_int{ 1, 2, 1, 64 }, 4, .float32);
    defer _ = mlx.mlx_array_free(mel);
    try std.testing.expectError(error.MissingAudioWeight, vocode(&comp, mel, s));
}
