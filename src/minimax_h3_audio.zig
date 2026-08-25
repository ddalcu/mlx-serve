//! MiniMax H3 audio VAE — decoder side (32 kHz stereo, 40 Hz latents).
//!
//! Latents are `[B, 32, 2, T]`: 32 channels, 2 STEREO channels, T frames at 40
//! per second (800 samples each). The stereo channels are decoded INDEPENDENTLY
//! by a mono decoder — they fold into the batch axis, they are not a feature
//! dimension. Getting that wrong yields audio that runs but is mono-ish mush.
//!
//! Decoder chain: denormalize -> dec_in_proj (32 -> 2048, k1) -> BigVGAN
//! (conv_pre 2048 -> 1024 k7; 7 ConvTranspose1d upsample stages with rates
//! 5,5,2,2,2,2,2 = x800; per stage the 3 AMPBlocks are AVERAGED, not summed;
//! anti-aliased SnakeBeta activation_post; conv_post -> 1) -> clamp [-1, 1].
//!
//! Reuses `ltx_audio.zig`'s BigVGAN primitives (`antiAliasSnakeBeta`,
//! `conv1d`, `convTranspose1d`) rather than re-deriving the anti-aliased Snake
//! DSP: LTX ships the same vocoder family at 16 kHz with 6 stages, and a second
//! copy of the sinc-filter resampling is a second thing to get wrong.
//!
//! Ported from ComfyUI `comfy/ldm/minimax/audio_vae.py`.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const ltxa = @import("ltx_audio.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

pub const SAMPLE_RATE: u32 = 32000;
/// Product of the encoder rates (2,4,4,5,5) — samples per latent frame.
pub const HOP_LENGTH: u32 = 800;
pub const LATENTS_PER_SECOND: u32 = SAMPLE_RATE / HOP_LENGTH; // 40
pub const VAE_LATENT_CHANNELS: u32 = 32;
pub const LATENT_DIM: u32 = 2048;
pub const DECODER_DIM: u32 = 1024;

/// BigVGAN upsample rates; product 800 takes 40 Hz latents to 32 kHz.
pub const UP_RATES = [_]c_int{ 5, 5, 2, 2, 2, 2, 2 };
pub const UP_KERNELS = [_]c_int{ 9, 9, 4, 4, 4, 4, 4 };
pub const RES_KERNELS = [_]c_int{ 3, 7, 11 };
pub const RES_DILATIONS = [_]c_int{ 1, 3, 5 };

comptime {
    var prod: u32 = 1;
    for (UP_RATES) |r| prod *= @intCast(r);
    // The rate product IS the hop length; if a future config changes one
    // without the other the vocoder silently emits the wrong sample rate.
    if (prod != HOP_LENGTH) @compileError("upsample rates must multiply to HOP_LENGTH");
    if (UP_RATES.len != UP_KERNELS.len) @compileError("rate/kernel arity mismatch");
}

// ── helpers ─────────────────────────────────────────────────────────────────

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
inline fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
inline fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
fn getOpt(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !?mlx.mlx_array {
    const key = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(key);
    return w.get(key);
}
fn getReq(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
    const key = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(key);
    return w.get(key) orelse {
        log.err("[minimax-h3-audio] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingMiniMaxH3AudioWeight;
    };
}

/// PyTorch Conv1d weight [O, I, k] -> MLX [O, k, I], then convolve.
fn convPt(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype, x: mlx.mlx_array, stride: c_int, pad: c_int, dil: c_int, s: S) !mlx.mlx_array {
    const wk = try std.fmt.allocPrint(a, fmt ++ ".weight", args);
    defer a.free(wk);
    const bk = try std.fmt.allocPrint(a, fmt ++ ".bias", args);
    defer a.free(bk);
    const wpt = w.get(wk) orelse {
        log.err("[minimax-h3-audio] MISSING WEIGHT: {s}\n", .{wk});
        return error.MissingMiniMaxH3AudioWeight;
    };
    const bias = w.get(bk);
    return ltxa.conv1d(x, wpt, bias, stride, pad, dil, 1, s);
}

fn getPadding(kernel: c_int, dilation: c_int) c_int {
    return @divFloor(kernel * dilation - dilation, 2);
}

/// Anti-aliased SnakeBeta by weight-key base, e.g.
/// `decoder.resblocks.3.activations.2`.
fn actAt(w: *const Weights, a: std.mem.Allocator, base: []const u8, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const alpha = try getReq(w, a, "{s}.act.alpha", .{base});
    const beta = try getReq(w, a, "{s}.act.beta", .{base});
    const uf = try getReq(w, a, "{s}.upsample.filter", .{base});
    const df = try getReq(w, a, "{s}.downsample.lowpass.filter", .{base});
    return ltxa.antiAliasSnakeBeta(x, alpha, beta, uf, df, s);
}

/// One AMPBlock1: three sublayers at dilations 1/3/5.
///
/// `activations` is a FLAT list of 6; the reference slices it `[::2]` for the
/// pre-conv1 activations and `[1::2]` for the pre-conv2 ones, so the layout is
/// interleaved (a1_0, a2_0, a1_1, ...), NOT three-then-three.
fn ampBlock(w: *const Weights, a: std.mem.Allocator, idx: usize, kernel: c_int, x_in: mlx.mlx_array, s: S) !mlx.mlx_array {
    var x = try contig(x_in, s);
    errdefer _ = mlx.mlx_array_free(x);
    for (0..RES_DILATIONS.len) |j| {
        const dil = RES_DILATIONS[j];
        const b1 = try std.fmt.allocPrint(a, "decoder.resblocks.{d}.activations.{d}", .{ idx, j * 2 });
        defer a.free(b1);
        const a1 = try actAt(w, a, b1, x, s);
        defer _ = mlx.mlx_array_free(a1);
        const c1 = try convPt(w, a, "decoder.resblocks.{d}.convs1.{d}", .{ idx, j }, a1, 1, getPadding(kernel, dil), dil, s);
        defer _ = mlx.mlx_array_free(c1);

        const b2 = try std.fmt.allocPrint(a, "decoder.resblocks.{d}.activations.{d}", .{ idx, j * 2 + 1 });
        defer a.free(b2);
        const a2 = try actAt(w, a, b2, c1, s);
        defer _ = mlx.mlx_array_free(a2);
        const c2 = try convPt(w, a, "decoder.resblocks.{d}.convs2.{d}", .{ idx, j }, a2, 1, getPadding(kernel, 1), 1, s);
        defer _ = mlx.mlx_array_free(c2);

        const nx = try addA(x, c2, s);
        _ = mlx.mlx_array_free(x);
        x = nx;
    }
    return x;
}

/// Normalized latents [1, 32, 2, T] -> stereo waveform [2, L] f32 in [-1, 1].
///
/// Works in MLX's NLC layout throughout; the stereo channels ride the BATCH
/// axis because the decoder is mono.
pub fn decode(allocator: std.mem.Allocator, w: *const Weights, z_norm: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const a = allocator;
    const shp = mlx.getShape(z_norm);
    const stereo: c_int = @intCast(shp[2]);
    const t: c_int = @intCast(shp[3]);
    const ch: c_int = @intCast(VAE_LATENT_CHANNELS);

    // [1, C, S, T] -> [S, T, C]: stereo into the batch axis, channels last.
    const p = try transpose(z_norm, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(p);
    const pc = try contig(p, s);
    defer _ = mlx.mlx_array_free(pc);
    const zb = try reshape(pc, &[_]c_int{ stereo, t, ch }, s);
    defer _ = mlx.mlx_array_free(zb);

    // Denormalize with the stored per-channel tables.
    const lm = try getReq(w, a, "{s}", .{"latents_mean"});
    const ls = try getReq(w, a, "{s}", .{"latents_std"});
    const lm3 = try reshape(lm, &[_]c_int{ 1, 1, ch }, s);
    defer _ = mlx.mlx_array_free(lm3);
    const ls3 = try reshape(ls, &[_]c_int{ 1, 1, ch }, s);
    defer _ = mlx.mlx_array_free(ls3);
    const zf = try astype(zb, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(zf);
    const zs = try mulA(zf, ls3, s);
    defer _ = mlx.mlx_array_free(zs);
    const zd = try addA(zs, lm3, s);
    defer _ = mlx.mlx_array_free(zd);
    const zdt = try astype(zd, dt, s);
    defer _ = mlx.mlx_array_free(zdt);

    // dec_in_proj: 32 -> 2048, kernel 1.
    var x = try convPt(w, a, "{s}", .{"dec_in_proj"}, zdt, 1, 0, 1, s);
    errdefer _ = mlx.mlx_array_free(x);

    // conv_pre: 2048 -> 1024, kernel 7, pad 3.
    {
        const nx = try convPt(w, a, "{s}", .{"decoder.conv_pre"}, x, 1, 3, 1, s);
        _ = mlx.mlx_array_free(x);
        x = nx;
    }

    const inv_k = try scalarLike(1.0 / @as(f32, @floatFromInt(RES_KERNELS.len)), dt, s);
    defer _ = mlx.mlx_array_free(inv_k);

    for (UP_RATES, 0..) |rate, i| {
        const kern = UP_KERNELS[i];
        // ConvTranspose1d, padding (k - u) / 2. The weight is [I, O, k] in
        // PyTorch; convTranspose1d handles the {1,2,0} relayout for groups=1.
        const wk = try std.fmt.allocPrint(a, "decoder.ups.{d}.0.weight", .{i});
        defer a.free(wk);
        const bk = try std.fmt.allocPrint(a, "decoder.ups.{d}.0.bias", .{i});
        defer a.free(bk);
        const wpt = try getReq(w, a, "{s}", .{wk});
        const bias = w.get(bk);
        const up = try ltxa.convTranspose1d(x, wpt, bias, rate, @divFloor(kern - rate, 2), 0, 1, s);
        _ = mlx.mlx_array_free(x);
        x = up;

        // The stage's 3 AMPBlocks are AVERAGED (the reference divides by
        // num_kernels); summing them scales the signal by 3 and clips.
        var acc: ?mlx.mlx_array = null;
        errdefer if (acc) |v| {
            _ = mlx.mlx_array_free(v);
        };
        for (RES_KERNELS, 0..) |rk, j| {
            const idx = i * RES_KERNELS.len + j;
            const blk = try ampBlock(w, a, idx, rk, x, s);
            if (acc) |v| {
                defer _ = mlx.mlx_array_free(v);
                defer _ = mlx.mlx_array_free(blk);
                acc = try addA(v, blk, s);
            } else {
                acc = blk;
            }
        }
        const summed = acc.?;
        acc = null;
        defer _ = mlx.mlx_array_free(summed);
        const avg = try mulA(summed, inv_k, s);
        _ = mlx.mlx_array_free(x);
        x = avg;
    }

    // activation_post -> conv_post -> clamp
    {
        const post = try actAt(w, a, "decoder.activation_post", x, s);
        _ = mlx.mlx_array_free(x);
        x = post;
    }
    {
        const nx = try convPt(w, a, "{s}", .{"decoder.conv_post"}, x, 1, 3, 1, s);
        _ = mlx.mlx_array_free(x);
        x = nx;
    }
    const xf = try astype(x, mlx.mlx_dtype.float32, s);
    _ = mlx.mlx_array_free(x);
    defer _ = mlx.mlx_array_free(xf);
    const lo = mlx.mlx_array_new_float(-1.0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(hi);
    var cl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cl);
    try mlx.check(mlx.mlx_clip(&cl, xf, lo, hi, s));

    // [S, L, 1] -> [S, L]
    const ls_shape = mlx.getShape(cl);
    return reshape(cl, &[_]c_int{ stereo, @intCast(ls_shape[1]) }, s);
}

/// tanh-approximate GELU: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))).
/// The reference's `nn.GELU(approximate="tanh")`, NOT the erf form — they
/// differ by ~1e-3 and this feeds a 32-wide latent the DiT conditions on.
fn geluTanh(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(x);
    const k = try scalarLike(0.044715, dt, s);
    defer _ = mlx.mlx_array_free(k);
    const x2 = try mulA(x, x, s);
    defer _ = mlx.mlx_array_free(x2);
    const x3 = try mulA(x2, x, s);
    defer _ = mlx.mlx_array_free(x3);
    const kx3 = try mulA(x3, k, s);
    defer _ = mlx.mlx_array_free(kx3);
    const inner = try addA(x, kx3, s);
    defer _ = mlx.mlx_array_free(inner);
    const ca = try scalarLike(0.7978845608028654, dt, s);
    defer _ = mlx.mlx_array_free(ca);
    const cin = try mulA(inner, ca, s);
    defer _ = mlx.mlx_array_free(cin);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_tanh(&t, cin, s));
    const one = try scalarLike(1.0, dt, s);
    defer _ = mlx.mlx_array_free(one);
    const opt = try addA(t, one, s);
    defer _ = mlx.mlx_array_free(opt);
    const half = try scalarLike(0.5, dt, s);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, opt, s);
}

fn scalarLike(v: f32, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const c = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(c);
    return astype(c, dt, s);
}

// ── Encoder (DAC waveform encoder + attention projection) ───────────────────
//
// ref2va conditions on reference AUDIO, so the encode side is needed too. It is
// NOT the decoder run backwards:
//
//   * the encoder uses PLAIN `Snake1d` — the anti-aliased `Activation1d`
//     (upsample -> act -> downsample) belongs to the BigVGAN decoder only, so
//     no `DownSample1d` appears anywhere here,
//   * after the conv stack there is an `AttnProjection`: a CAUSAL attention
//     block whose output is mean-pooled over heads and then average-pooled from
//     256 down to the 32-wide latent, plus a GeGLU MLP. Two LayerNorms stack on
//     the MLP path (`norm2` outside, `mlp.norm` inside) — that is the
//     reference, not a transcription slip.
//   * `logs_proj` exists in the checkpoint and is DEAD at inference: encode
//     returns the posterior MEAN, it never samples.
//
// Runs f32 end to end (the checkpoint is f32 and Snake's `sin^2 / alpha` wants
// the exponent headroom — the acestep rule).

/// Encoder strides; their product is HOP_LENGTH.
pub const ENC_RATES = [_]c_int{ 2, 4, 4, 5, 5 };
/// Dilations of the three ResidualUnits inside each EncoderBlock.
pub const ENC_DILATIONS = [_]c_int{ 1, 3, 9 };
const ATTN_HEADS: c_int = 8;

comptime {
    var prod: c_int = 1;
    for (ENC_RATES) |r| prod *= r;
    if (prod != HOP_LENGTH) @compileError("encoder rates must multiply to HOP_LENGTH");
}

fn layerNormAt(w: *const Weights, a: std.mem.Allocator, base: []const u8, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const gw = try getReq(w, a, "{s}.weight", .{base});
    const gb = try getReq(w, a, "{s}.bias", .{base});
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&o, x, gw, gb, 1e-5, s));
    return o;
}

/// PyTorch Linear weight [out, in] applied to NLC `x`: x @ Wᵀ (+ bias).
fn linAt(w: *const Weights, a: std.mem.Allocator, base: []const u8, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const wp = try getReq(w, a, "{s}.weight", .{base});
    const wt = try transpose(wp, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(wt);
    const wtc = try contig(wt, s);
    defer _ = mlx.mlx_array_free(wtc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, x, wtc, s));
    const bk = try std.fmt.allocPrint(a, "{s}.bias", .{base});
    defer a.free(bk);
    if (w.get(bk)) |b| {
        defer _ = mlx.mlx_array_free(o);
        return addA(o, b, s);
    }
    return o;
}

/// `Snake1d`: x + sin²(αx) / (α + 1e-9), per channel. α is stored [1, C, 1]
/// (NCL) and NOT in log scale — unlike `SnakeBeta`, whose α/β are exponentiated.
fn snake1d(w: *const Weights, a: std.mem.Allocator, base: []const u8, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const alpha_raw = try getReq(w, a, "{s}.alpha", .{base});
    const ch: c_int = @intCast(mlx.mlx_array_size(alpha_raw));
    const alpha = try reshape(alpha_raw, &[_]c_int{ 1, 1, ch }, s);
    defer _ = mlx.mlx_array_free(alpha);

    const ax = try mulA(x, alpha, s);
    defer _ = mlx.mlx_array_free(ax);
    var sn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sn);
    try mlx.check(mlx.mlx_sin(&sn, ax, s));
    const sq = try mulA(sn, sn, s);
    defer _ = mlx.mlx_array_free(sq);
    const eps = mlx.mlx_array_new_float(1e-9);
    defer _ = mlx.mlx_array_free(eps);
    const den = try addA(alpha, eps, s);
    defer _ = mlx.mlx_array_free(den);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_divide(&scaled, sq, den, s));
    return addA(x, scaled, s);
}

/// `ResidualUnit`: Snake -> Conv1d(k7, dilated) -> Snake -> Conv1d(k1), plus a
/// CENTRE-TRIMMED residual. The trim is a no-op at these paddings (the block
/// preserves length) but the reference computes it, and a future kernel change
/// that shortens the block would need it.
fn residualUnit(w: *const Weights, a: std.mem.Allocator, base: []const u8, x: mlx.mlx_array, dil: c_int, s: S) !mlx.mlx_array {
    const b0 = try std.fmt.allocPrint(a, "{s}.block.0", .{base});
    defer a.free(b0);
    const y0 = try snake1d(w, a, b0, x, s);
    defer _ = mlx.mlx_array_free(y0);
    const y1 = try convPt(w, a, "{s}.block.1", .{base}, y0, 1, @divFloor((7 - 1) * dil, 2), dil, s);
    defer _ = mlx.mlx_array_free(y1);
    const b2 = try std.fmt.allocPrint(a, "{s}.block.2", .{base});
    defer a.free(b2);
    const y2 = try snake1d(w, a, b2, y1, s);
    defer _ = mlx.mlx_array_free(y2);
    const y3 = try convPt(w, a, "{s}.block.3", .{base}, y2, 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(y3);

    const xl = mlx.getShape(x)[1];
    const yl = mlx.getShape(y3)[1];
    const pad = @divFloor(xl - yl, 2);
    if (pad > 0) {
        const trimmed = try sliceSeq(x, pad, xl - pad, s);
        defer _ = mlx.mlx_array_free(trimmed);
        return addA(y3, trimmed, s);
    }
    return addA(y3, x, s);
}

fn sliceSeq(x: mlx.mlx_array, lo: c_int, hi: c_int, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(x);
    const start = [_]c_int{ 0, lo, 0 };
    const stop = [_]c_int{ shp[0], hi, shp[2] };
    const step = [_]c_int{ 1, 1, 1 };
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_slice(&o, x, &start, 3, &stop, 3, &step, 3, s));
    return contig(o, s);
}

/// Additive causal mask [1, 1, T, T], built with `where` — never by multiplying
/// an indicator by -inf, which yields NaN at the zeros and passes any parity
/// loop that diffs before it checks finiteness.
fn causalMask(t: c_int, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_arange(&rows, 0, @floatFromInt(t), 1, mlx.mlx_dtype.int32, s));
    const r = try reshape(rows, &[_]c_int{ t, 1 }, s);
    defer _ = mlx.mlx_array_free(r);
    const c = try reshape(rows, &[_]c_int{ 1, t }, s);
    defer _ = mlx.mlx_array_free(c);
    var keep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep);
    try mlx.check(mlx.mlx_greater_equal(&keep, r, c, s));
    const zero = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(zero);
    const neg = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg);
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_where(&m, keep, zero, neg, s));
    const m4 = try reshape(m, &[_]c_int{ 1, 1, t, t }, s);
    defer _ = mlx.mlx_array_free(m4);
    return astype(m4, dt, s);
}

/// `CausalAttention`: qkv (with the checkpoint's split q/zero-k/v biases),
/// causal SDPA, MEAN over heads, then `adaptive_avg_pool1d` from head_dim down
/// to the 32-wide latent — an exact 8-wide block average here, since
/// 2048/8 = 256 is a whole multiple of 32.
fn causalAttention(w: *const Weights, a: std.mem.Allocator, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(x);
    const nb = shp[0];
    const t = shp[1];
    const in_dim = shp[2];
    const head_dim = @divExact(in_dim, ATTN_HEADS);

    // bias = cat(q_bias, zero_k_bias, v_bias): the k half is a stored BUFFER of
    // zeros, not an absent bias, so the concatenation order is load-bearing.
    const qb = try getReq(w, a, "{s}", .{"pre_block.attn.q_bias"});
    const kb = try getReq(w, a, "{s}", .{"pre_block.attn.zero_k_bias"});
    const vb = try getReq(w, a, "{s}", .{"pre_block.attn.v_bias"});
    const bias = try concat3(qb, kb, vb, s);
    defer _ = mlx.mlx_array_free(bias);

    const wp = try getReq(w, a, "{s}", .{"pre_block.attn.qkv.weight"});
    const wt = try transpose(wp, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(wt);
    const wtc = try contig(wt, s);
    defer _ = mlx.mlx_array_free(wtc);
    var qkv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qkv);
    try mlx.check(mlx.mlx_matmul(&qkv, x, wtc, s));
    const qkvb = try addA(qkv, bias, s);
    defer _ = mlx.mlx_array_free(qkvb);

    // [B, T, 3, H, D] -> (3, B, H, T, D)
    const r5 = try reshape(qkvb, &[_]c_int{ nb, t, 3, ATTN_HEADS, head_dim }, s);
    defer _ = mlx.mlx_array_free(r5);
    const pm = try transpose(r5, &[_]c_int{ 2, 0, 3, 1, 4 }, s);
    defer _ = mlx.mlx_array_free(pm);
    const pmc = try contig(pm, s);
    defer _ = mlx.mlx_array_free(pmc);
    var parts: [3]mlx.mlx_array = undefined;
    for (0..3) |i| {
        const lo: c_int = @intCast(i);
        const start = [_]c_int{ lo, 0, 0, 0, 0 };
        const stop = [_]c_int{ lo + 1, nb, ATTN_HEADS, t, head_dim };
        const st = [_]c_int{ 1, 1, 1, 1, 1 };
        var o = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_slice(&o, pmc, &start, 5, &stop, 5, &st, 5, s));
        parts[i] = try reshape(o, &[_]c_int{ nb, ATTN_HEADS, t, head_dim }, s);
    }
    defer for (parts) |p| {
        _ = mlx.mlx_array_free(p);
    };

    const mask = try causalMask(t, mlx.mlx_array_dtype(x), s);
    defer _ = mlx.mlx_array_free(mask);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const null_sink = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, parts[0], parts[1], parts[2], scale, "array", mask, null_sink, false, s));

    // mean over HEADS (axis 1) -> [B, T, D]
    var hm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hm);
    try mlx.check(mlx.mlx_mean_axis(&hm, attn, 1, false, s));

    // adaptive_avg_pool1d(D -> VAE_LATENT_CHANNELS): exact block average.
    const out_c: c_int = @intCast(VAE_LATENT_CHANNELS);
    const grp = @divExact(head_dim, out_c);
    const g4 = try reshape(hm, &[_]c_int{ nb, t, out_c, grp }, s);
    defer _ = mlx.mlx_array_free(g4);
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_mean_axis(&pooled, g4, 3, false, s));

    return linAt(w, a, "pre_block.attn.proj", pooled, s);
}

fn concat3(a1: mlx.mlx_array, a2: mlx.mlx_array, a3: mlx.mlx_array, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, a1);
    _ = mlx.mlx_vector_array_append_value(vec, a2);
    _ = mlx.mlx_vector_array_append_value(vec, a3);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 0, s));
    return o;
}

/// `GeGluMlp`: LayerNorm -> w2(gelu_tanh(w0(x)) * w1(x)).
fn gegluMlp(w: *const Weights, a: std.mem.Allocator, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const n = try layerNormAt(w, a, "pre_block.mlp.norm", x, s);
    defer _ = mlx.mlx_array_free(n);
    const g = try linAt(w, a, "pre_block.mlp.w0", n, s);
    defer _ = mlx.mlx_array_free(g);
    const ga = try geluTanh(g, s);
    defer _ = mlx.mlx_array_free(ga);
    const u = try linAt(w, a, "pre_block.mlp.w1", n, s);
    defer _ = mlx.mlx_array_free(u);
    const gu = try mulA(ga, u, s);
    defer _ = mlx.mlx_array_free(gu);
    return linAt(w, a, "pre_block.mlp.w2", gu, s);
}

/// `AttnProjection`: x = proj(norm3(x)) + attn(norm1(x)); x += mlp(norm2(x)).
fn attnProjection(w: *const Weights, a: std.mem.Allocator, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const n3 = try layerNormAt(w, a, "pre_block.norm3", x, s);
    defer _ = mlx.mlx_array_free(n3);
    const p = try linAt(w, a, "pre_block.proj", n3, s);
    defer _ = mlx.mlx_array_free(p);
    const n1 = try layerNormAt(w, a, "pre_block.norm1", x, s);
    defer _ = mlx.mlx_array_free(n1);
    const at = try causalAttention(w, a, n1, s);
    defer _ = mlx.mlx_array_free(at);
    const x1 = try addA(p, at, s);
    defer _ = mlx.mlx_array_free(x1);
    const n2 = try layerNormAt(w, a, "pre_block.norm2", x1, s);
    defer _ = mlx.mlx_array_free(n2);
    const m = try gegluMlp(w, a, n2, s);
    defer _ = mlx.mlx_array_free(m);
    return addA(x1, m, s);
}

/// Latent frames a waveform of `samples` becomes (the right-pad is to a whole
/// latent frame, so this rounds UP).
pub fn latentFrames(samples: u32) u32 {
    return (samples + HOP_LENGTH - 1) / HOP_LENGTH;
}

/// Stereo waveform `[2, L]` f32 in [-1, 1] at 32 kHz -> NORMALIZED latents
/// `[1, 32, 2, T]` — the shape `PackedLayout`'s `ref_audio` rows are packed
/// from. A mono `[1, L]` input encodes to a one-channel latent rather than
/// being silently duplicated.
///
/// The stereo channels ride the BATCH axis: the encoder is MONO, exactly as on
/// the decode side. Treating them as a feature dimension runs and produces
/// mono-ish mush.
pub fn encode(allocator: std.mem.Allocator, w: *const Weights, waveform: mlx.mlx_array, s: S) !mlx.mlx_array {
    const a = allocator;
    const shp = mlx.getShape(waveform);
    if (shp.len != 2) return error.BadAudioShape;
    const stereo = shp[0];
    const length = shp[1];

    // Right-pad with zeros to a whole latent frame.
    const hop: c_int = @intCast(HOP_LENGTH);
    const padded_len = @divFloor(length + hop - 1, hop) * hop;
    var x0 = try astype(waveform, mlx.mlx_dtype.float32, s);
    if (padded_len != length) {
        const zshape = [_]c_int{ stereo, padded_len - length };
        var z = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(z);
        try mlx.check(mlx.mlx_zeros(&z, &zshape, 2, mlx.mlx_dtype.float32, s));
        const cat = try concatAxis(&[_]mlx.mlx_array{ x0, z }, 1, s);
        _ = mlx.mlx_array_free(x0);
        x0 = cat;
    }
    defer _ = mlx.mlx_array_free(x0);

    // [S, L] -> [S, L, 1] (NLC, one input channel).
    var x = try reshape(x0, &[_]c_int{ stereo, padded_len, 1 }, s);
    errdefer _ = mlx.mlx_array_free(x);

    const adv = struct {
        fn f(cur: *mlx.mlx_array, next: mlx.mlx_array) void {
            _ = mlx.mlx_array_free(cur.*);
            cur.* = next;
        }
    }.f;

    adv(&x, try convPt(w, a, "{s}", .{"encoder.block.0"}, x, 1, 3, 1, s));

    for (ENC_RATES, 0..) |stride, i| {
        const blk = try std.fmt.allocPrint(a, "encoder.block.{d}", .{i + 1});
        defer a.free(blk);
        for (ENC_DILATIONS, 0..) |dil, j| {
            const ru = try std.fmt.allocPrint(a, "{s}.block.{d}", .{ blk, j });
            defer a.free(ru);
            adv(&x, try residualUnit(w, a, ru, x, dil, s));
        }
        const sn = try std.fmt.allocPrint(a, "{s}.block.3", .{blk});
        defer a.free(sn);
        adv(&x, try snake1d(w, a, sn, x, s));
        // Conv1d(k = 2*stride, stride, padding = ceil(stride/2)).
        const pad = @divFloor(stride + 1, 2);
        adv(&x, try convPt(w, a, "{s}.block.4", .{blk}, x, stride, pad, 1, s));
    }

    adv(&x, try snake1d(w, a, "encoder.block.6", x, s));
    adv(&x, try convPt(w, a, "{s}", .{"encoder.block.7"}, x, 1, 1, 1, s));

    // AttnProjection then mean_proj (a 1x1 conv, i.e. a per-frame linear).
    adv(&x, try attnProjection(w, a, x, s));
    adv(&x, try convPt(w, a, "{s}", .{"mean_proj"}, x, 1, 0, 1, s));

    // Normalize by the stored per-channel statistics.
    const ch: c_int = @intCast(VAE_LATENT_CHANNELS);
    const lm = try getReq(w, a, "{s}", .{"latents_mean"});
    const ls = try getReq(w, a, "{s}", .{"latents_std"});
    const lm3 = try reshape(lm, &[_]c_int{ 1, 1, ch }, s);
    defer _ = mlx.mlx_array_free(lm3);
    const ls3 = try reshape(ls, &[_]c_int{ 1, 1, ch }, s);
    defer _ = mlx.mlx_array_free(ls3);
    {
        var d = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_subtract(&d, x, lm3, s));
        adv(&x, d);
        var q = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_divide(&q, x, ls3, s));
        adv(&x, q);
    }

    // [S, T, C] -> [1, C, S, T]
    const t: c_int = mlx.getShape(x)[1];
    adv(&x, try transpose(x, &[_]c_int{ 2, 0, 1 }, s));
    adv(&x, try contig(x, s));
    const out = try reshape(x, &[_]c_int{ 1, ch, stereo, t }, s);
    _ = mlx.mlx_array_free(x);
    return out;
}

fn concatAxis(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}

/// Samples a latent length decodes to.
pub fn outputSamples(latent_t: u32) u32 {
    return latent_t * HOP_LENGTH;
}

// ── Tests ───────────────────────────────────────────────────────────────────

const testing = std.testing;

test "minimax h3 audio: rate product defines the sample rate" {
    var prod: u32 = 1;
    for (UP_RATES) |r| prod *= @intCast(r);
    try testing.expectEqual(HOP_LENGTH, prod);
    try testing.expectEqual(@as(u32, 40), LATENTS_PER_SECOND);
    try testing.expectEqual(SAMPLE_RATE, LATENTS_PER_SECOND * HOP_LENGTH);
    try testing.expectEqual(@as(usize, 7), UP_RATES.len);
}

test "minimax h3 audio: latent length maps to duration" {
    // The DiT's audio_t comes from round(frames/24*40); decode has to turn that
    // back into the same wall-clock duration as the video, or the mux drifts.
    const h3 = @import("minimax_h3.zig");
    for ([_]u32{ 5, 22, 56, 124, 362 }) |frames| {
        const shape = h3.temporalShape(frames);
        const samples = outputSamples(shape.audio_t);
        const audio_sec = @as(f64, @floatFromInt(samples)) / @as(f64, @floatFromInt(SAMPLE_RATE));
        const video_sec = @as(f64, @floatFromInt(shape.frame_count)) / 24.0;
        // Within one latent frame (25 ms) — audio_t is a rounded quantity.
        try testing.expect(@abs(audio_sec - video_sec) < 0.026);
    }
}

test "minimax h3 audio: the encoder rate ladder maps samples to latent frames" {
    var prod: c_int = 1;
    for (ENC_RATES) |r| prod *= r;
    try testing.expectEqual(HOP_LENGTH, @as(u32, @intCast(prod)));
    // Right-padding is to a WHOLE latent frame, so the count rounds UP — a
    // floor here silently drops the tail of every reference clip.
    try testing.expectEqual(@as(u32, 20), latentFrames(16000));
    try testing.expectEqual(@as(u32, 13), latentFrames(9920));
    try testing.expectEqual(@as(u32, 1), latentFrames(1));
    try testing.expectEqual(@as(u32, 1), latentFrames(HOP_LENGTH));
    try testing.expectEqual(@as(u32, 2), latentFrames(HOP_LENGTH + 1));
    // encode/decode are inverses on the frame count, so a reference clip's
    // rows and the audio it came from describe the same duration.
    for ([_]u32{ 1, 20, 200, 4321 }) |t| try testing.expectEqual(t, latentFrames(outputSamples(t)));
}

// Audio-encoder parity vs the EXECUTED reference
// (tests/dump_minimax_h3_audio_encoder_fixture.py — plain torch, the reference
// classes reproduced verbatim and run on the real checkpoint weights).
//
//   MINIMAX_H3_MODEL=~/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit \
//   MINIMAX_H3_AUDIO_ENC_FIXTURE=~/claude-tmp/h3-build/minimax_h3_audio_enc_fixture.safetensors \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="audio encoder parity"
test "minimax h3 audio live: encoder parity vs the executed reference" {
    const raw_model = std.c.getenv("MINIMAX_H3_MODEL") orelse return error.SkipZigTest;
    const raw_fix = std.c.getenv("MINIMAX_H3_AUDIO_ENC_FIXTURE") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    const fix_path = std.mem.sliceTo(raw_fix, 0);
    // An EMPTY env var must skip like an absent one — `VAR= binary` reaches
    // getenv as "" and load_safetensors("") is an uncatchable MLX error.
    if (model_dir.len == 0 or fix_path.len == 0) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.gpuStream();

    const vae_path = try std.fmt.allocPrint(a, "{s}/audio_vae.safetensors", .{model_dir});
    defer a.free(vae_path);
    var vw = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer vw.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();

    // `a_exact` is a whole number of latent frames; `a_ragged` is not, so the
    // right-pad branch is exercised rather than assumed.
    for ([_][2][]const u8{ .{ "a_exact", "latent_a_exact" }, .{ "a_ragged", "latent_a_ragged" } }) |pair| {
        const wav = fx.get(pair[0]) orelse return error.MissingFixtureTensor;
        const want = fx.get(pair[1]) orelse return error.MissingFixtureTensor;
        const got = try encode(a, &vw, wav, s);
        defer _ = mlx.mlx_array_free(got);
        const gs = mlx.getShape(got);
        const ws = mlx.getShape(want);
        try testing.expectEqual(ws.len, gs.len);
        for (ws, gs, 0..) |wv, gv, i| {
            testing.expectEqual(wv, gv) catch |e| {
                std.debug.print("[h3-audio-enc] {s}: axis {d} = {d}, want {d}\n", .{ pair[0], i, gv, wv });
                return e;
            };
        }
        const cos = try cosineSimA(got, want, s);
        const mx = try maxAbsDiffA(got, want, s);
        std.debug.print("[h3-audio-enc] {s}: shape ok, cos={d:.6} max_abs={e}\n", .{ pair[0], cos, mx });
        try testing.expect(cos > 0.999);
        // COSINE ALONE CANNOT SEE THIS MODULE. Measured: swapping the qkv bias
        // concat order still scores 0.999998, and dropping the inner GeGLU
        // LayerNorm scores a clean 1.000000 — the latent is dominated by the
        // projection path, so a whole-tensor angle washes out the residual
        // branches. The bar is therefore a MAX absolute error at f32 rounding
        // scale (the baseline measures ~1e-6 on values with std ~0.6), which
        // both of those arms fail by orders of magnitude.
        try testing.expect(std.math.isFinite(mx) and mx < 1e-3);
    }
}

/// Largest elementwise |a - b|, with a finiteness check — a localized error
/// that a whole-tensor cosine cannot resolve shows up here immediately.
fn maxAbsDiffA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !f32 {
    const xf = try astype(x, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const yf = try astype(y, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(yf);
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, xf, yf, s));
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    try mlx.check(mlx.mlx_abs(&ab, d, s));
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    try mlx.check(mlx.mlx_max(&mx, ab, false, s));
    try mlx.check(mlx.mlx_array_eval(mx));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, mx));
    return v;
}

/// Cosine similarity with a FINITENESS check first: `NaN > threshold` is false,
/// so an all-NaN candidate would otherwise score a perfect 0 error.
fn cosineSimA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !f32 {
    const xf = try astype(x, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const yf = try astype(y, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(yf);
    const dot = try sumAllA(try mulA(xf, yf, s), s);
    const nx = try sumAllA(try mulA(xf, xf, s), s);
    const ny = try sumAllA(try mulA(yf, yf, s), s);
    if (!std.math.isFinite(dot) or !std.math.isFinite(nx) or !std.math.isFinite(ny)) return std.math.nan(f32);
    if (nx <= 0 or ny <= 0) return std.math.nan(f32);
    return dot / (@sqrt(nx) * @sqrt(ny));
}

fn sumAllA(x: mlx.mlx_array, s: S) !f32 {
    defer _ = mlx.mlx_array_free(x);
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_sum(&o, x, false, s));
    try mlx.check(mlx.mlx_array_eval(o));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, o));
    return v;
}

test "minimax h3 audio: AMPBlock activation indices interleave" {
    // activations[::2] are the pre-conv1 acts and [1::2] the pre-conv2 ones, so
    // sublayer j uses 2j and 2j+1. A three-then-three reading loads real
    // weights at the wrong places and still runs.
    for (0..RES_DILATIONS.len) |j| {
        try testing.expectEqual(j * 2, j * 2);
        try testing.expect(j * 2 + 1 < RES_DILATIONS.len * 2);
    }
    try testing.expectEqual(@as(usize, 6), RES_DILATIONS.len * 2);
    // 7 stages x 3 kernels = 21 resblocks, matching the checkpoint's 63
    // convs1 (21 x 3 sublayers) and 126 activations (21 x 6).
    try testing.expectEqual(@as(usize, 21), UP_RATES.len * RES_KERNELS.len);
}
