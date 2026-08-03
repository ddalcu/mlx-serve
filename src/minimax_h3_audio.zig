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

fn scalarLike(v: f32, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const c = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(c);
    return astype(c, dt, s);
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
