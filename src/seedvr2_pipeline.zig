//! SeedVR2 sampling + conditioning — the thin layer that turns the DiT into a
//! restorer. Pure math, no MLX, so the sampler can be pinned without weights.
//!
//! Transcribed from ByteDance-Seed/SeedVR `common/diffusion/*` and
//! `projects/video_diffusion_sr/infer.py` (Apache-2.0). Spec: `docs/seedvr2-arch.md` §4.
//!
//! THE ONE-STEP PATH IS MUCH SIMPLER THAN THE MACHINERY SUGGESTS. With
//! `steps=1`, `arange(1.0, 0.0, -1.0)` yields a SINGLE timestep at `T`, so
//! Euler's `zip(timesteps[:-1], timesteps[1:])` loop body never runs and the
//! `return_endpoint` branch does all the work. That branch is
//! `convert_from_pred(..., v_lerp)`, and on the lerp schedule `A(t) + B(t) == 1`
//! identically, so it collapses to
//!
//!     x_0 = x_t - (t/T) * pred
//!
//! and at `t == T` to `x_0 = noise - pred`. One forward, one subtraction.

const std = @import("std");

/// `schedule.T` — the lerp schedule's time range. Timesteps are on 0..1000,
/// NOT 0..1; the sinusoidal embedding is fed the unnormalised value.
pub const T: f32 = 1000.0;

/// `A(t) = 1 - t/T` — the x_0 coefficient.
pub fn coeffA(t: f32) f32 {
    return 1.0 - t / T;
}

/// `B(t) = t/T` — the x_T coefficient.
pub fn coeffB(t: f32) f32 {
    return t / T;
}

/// `x_t = A(t)*x_0 + B(t)*x_T`.
pub fn forwardMix(x0: f32, xt_end: f32, t: f32) f32 {
    return coeffA(t) * x0 + coeffB(t) * xt_end;
}

/// Recover `x_0` from a `v_lerp` prediction:
/// `pred_x_0 = (x_t - B(t)*pred) / (A(t) + B(t))`.
///
/// The denominator is identically 1 on the lerp schedule — it is written out
/// because the reference writes it out, and because a schedule where it is NOT
/// 1 would silently change this formula.
pub fn predToX0(x_t: f32, pred: f32, t: f32) f32 {
    return (x_t - coeffB(t) * pred) / (coeffA(t) + coeffB(t));
}

/// Recover `x_T` from a `v_lerp` prediction.
pub fn predToXT(x_t: f32, pred: f32, t: f32) f32 {
    return (x_t + coeffA(t) * pred) / (coeffA(t) + coeffB(t));
}

/// Uniform-trailing sampling timesteps, `steps` of them, before the
/// resolution-dependent transform. Caller owns the slice.
///
/// `torch.arange(1.0, 0.0, -1.0/steps)` — note this yields exactly `steps`
/// values STARTING at 1.0, so the first (and for `steps == 1`, only) entry is
/// `T`. There is no trailing zero: Euler's pairwise loop is therefore empty at
/// one step and the endpoint branch is what runs.
pub fn trailingTimesteps(a: std.mem.Allocator, steps: u32) ![]f32 {
    std.debug.assert(steps > 0);
    const out = try a.alloc(f32, steps);
    const inc = 1.0 / @as(f32, @floatFromInt(steps));
    for (0..steps) |i| {
        out[i] = (1.0 - inc * @as(f32, @floatFromInt(i))) * T;
    }
    return out;
}

/// VAE geometry the shift function measures resolution in.
pub const TEMPORAL_DOWNSAMPLE: u32 = 4;
pub const SPATIAL_DOWNSAMPLE: u32 = 8;

fn linThrough(x1: f64, y1: f64, x2: f64, y2: f64, x: f64) f64 {
    const m = (y2 - y1) / (x2 - x1);
    const b = y1 - m * x1;
    return m * x + b;
}

/// The resolution-dependent shift factor.
///
/// Measured in PIXELS, recovered from the latent grid: `frames = (t-1)*4 + 1`,
/// `height = h*8`, `width = w*8`. Images and videos use DIFFERENT lines, chosen
/// by `frames > 1`:
///   image: through (256*256, 1.0) and (1024*1024, 3.2)
///   video: through (256*256*37, 1.0) and (1280*720*145, 5.0)
///
/// Both lines are extrapolated, not clamped — a small input legitimately yields
/// a shift BELOW 1 (a 2x64x48 latent gives 0.956). Clamping it to 1 would be a
/// silent behaviour change at low resolution.
pub fn shiftFactor(latent_t: u32, latent_h: u32, latent_w: u32) f64 {
    const frames: f64 = @floatFromInt((latent_t - 1) * TEMPORAL_DOWNSAMPLE + 1);
    const height: f64 = @floatFromInt(latent_h * SPATIAL_DOWNSAMPLE);
    const width: f64 = @floatFromInt(latent_w * SPATIAL_DOWNSAMPLE);
    if (frames > 1.0) {
        return linThrough(256.0 * 256.0 * 37.0, 1.0, 1280.0 * 720.0 * 145.0, 5.0, height * width * frames);
    }
    return linThrough(256.0 * 256.0, 1.0, 1024.0 * 1024.0, 3.2, height * width);
}

/// Apply the resolution shift to a timestep.
///
/// `t' = T * (shift * τ) / (1 + (shift-1) * τ)` with `τ = t/T`.
///
/// **This is the identity at `t == T`** for every shift — `s*1/(1+s-1) = 1` —
/// which is why the one-step path needs no resolution handling at all. It
/// matters only for multi-step sampling.
pub fn transformTimestep(t: f32, latent_t: u32, latent_h: u32, latent_w: u32) f32 {
    const shift = shiftFactor(latent_t, latent_h, latent_w);
    const tau: f64 = @as(f64, t) / @as(f64, T);
    const shifted = shift * tau / (1.0 + (shift - 1.0) * tau);
    return @floatCast(shifted * @as(f64, T));
}

/// Channel layout of the DiT's video input.
///
/// `vid_in_channels = 33` = 16 noisy latent + 16 conditioning latent + 1 mask.
/// The reference builds `cond = zeros[t,h,w,17]`, writes the LOW-RES encoded
/// latent into `[..., :16]` and 1.0 into `[..., 16]` for task "sr" (every
/// position is conditioned; the mask channel exists for the i2v/v2v tasks that
/// condition only the first frames), then concatenates `[x_t, cond]` on the
/// channel axis.
pub const LATENT_CHANNELS: u32 = 16;
pub const COND_CHANNELS: u32 = LATENT_CHANNELS + 1;
pub const VID_IN_CHANNELS: u32 = LATENT_CHANNELS + COND_CHANNELS;

/// `cond_noise_scale` is 0.0 in the SeedVR2 entry point: the conditioning
/// latent is CLEAN. A port that noises it (following the SeedVR1 recipe, where
/// this is 0.25) degrades the restoration for no visible error.
pub const COND_NOISE_SCALE: f32 = 0.0;

/// The one-step SeedVR2 recipe.
pub const OneStep = struct {
    /// `cfg_scale = 1.0` in `inference_seedvr2_3b.py`, so the unconditional
    /// forward never runs and `neg_emb.pt` is never read. Exposed so a future
    /// multi-step/CFG mode can set it rather than rediscovering it.
    pub const cfg_scale: f32 = 1.0;
    pub const steps: u32 = 1;

    /// The single timestep: `T`, untouched by the resolution transform.
    pub fn timestep() f32 {
        return T;
    }

    /// `x_0 = noise - pred` at `t == T`.
    pub fn resolve(noise: f32, pred: f32) f32 {
        return predToX0(noise, pred, T);
    }
};

/// NOT IMPLEMENTED, deliberately. `projects/video_diffusion_sr/color_fix.py`
/// (`wavelet_reconstruction`) is referenced by the inference script but is NOT
/// in the upstream repository — the script guards on its absence and sets
/// `use_colorfix = False`, and ComfyUI's implementations do not have it either.
/// There is no reference to port and no fixture to pin, so writing one would be
/// inventing a post-process and calling it SeedVR2's.
pub const color_fix_available = false;

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "lerp coefficients sum to one at every timestep" {
    // The `A(t) + B(t)` denominator in convert_from_pred is identically 1 here.
    // If a future schedule breaks that, predToX0 is silently wrong.
    var t: f32 = 0;
    while (t <= T) : (t += 37.0) {
        try testing.expectApproxEqAbs(@as(f32, 1.0), coeffA(t) + coeffB(t), 1e-6);
    }
}

test "v_lerp conversion inverts the forward mix" {
    // Round trip: build x_t from a known (x_0, x_T), predict v = x_T - x_0, and
    // recover both endpoints.
    const x0: f32 = 0.37;
    const xt_end: f32 = -1.24;
    const v = xt_end - x0;
    var t: f32 = 0;
    while (t <= T) : (t += 125.0) {
        const x_t = forwardMix(x0, xt_end, t);
        try testing.expectApproxEqAbs(x0, predToX0(x_t, v, t), 1e-5);
        try testing.expectApproxEqAbs(xt_end, predToXT(x_t, v, t), 1e-5);
    }
}

test "one step at T collapses to noise minus prediction" {
    // THE whole sampler for SeedVR2. If this ever stops being a plain
    // subtraction, the single-step assumption has been broken somewhere.
    try testing.expectEqual(@as(f32, 1000.0), OneStep.timestep());
    const noise: f32 = 0.8;
    const pred: f32 = 0.3;
    try testing.expectApproxEqAbs(noise - pred, OneStep.resolve(noise, pred), 1e-6);
}

test "trailing timesteps start at T and have no trailing zero" {
    const a = testing.allocator;
    const one = try trailingTimesteps(a, 1);
    defer a.free(one);
    try testing.expectEqual(@as(usize, 1), one.len);
    try testing.expectEqual(T, one[0]);

    // `arange(1.0, 0.0, -1/steps)` yields exactly `steps` values from 1.0 down,
    // exclusive of 0. A port that appends a final 0 turns the one-step case
    // into a pairwise Euler step and changes the result.
    const four = try trailingTimesteps(a, 4);
    defer a.free(four);
    try testing.expectEqual(@as(usize, 4), four.len);
    try testing.expectApproxEqAbs(@as(f32, 1000.0), four[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 750.0), four[1], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 500.0), four[2], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 250.0), four[3], 1e-3);
}

test "oracle: shift factors from the reference's own lines" {
    // ORACLE, computed by running the reference's get_lin_function pair.
    const cases = [_]struct { t: u32, h: u32, w: u32, shift: f64 }{
        .{ .t = 1, .h = 90, .w = 160, .shift = 2.9158333333 }, // 720p still
        .{ .t = 1, .h = 32, .w = 32, .shift = 1.0000000000 }, // 256x256 still, the anchor
        .{ .t = 5, .h = 90, .w = 160, .shift = 1.4037086754 }, // 17-frame 720p
        .{ .t = 1, .h = 135, .w = 240, .shift = 5.4939583333 }, // 1080p still
        .{ .t = 2, .h = 64, .w = 48, .shift = 0.9560453283 }, // extrapolates BELOW 1
    };
    for (cases) |c| {
        const got = shiftFactor(c.t, c.h, c.w);
        testing.expectApproxEqAbs(c.shift, got, 1e-9) catch |err| {
            std.debug.print("latent {d}x{d}x{d}: want shift {d:.10} got {d:.10}\n", .{ c.t, c.h, c.w, c.shift, got });
            return err;
        };
    }
}

test "images and videos use different shift lines" {
    // `frames > 1` is the discriminator, and `frames = (t-1)*4 + 1`, so a
    // single LATENT frame is an image and two is a 5-frame video. Applying the
    // image line to a video (or vice versa) is a large, silent change.
    const img = shiftFactor(1, 90, 160);
    const vid = shiftFactor(2, 90, 160);
    try testing.expect(img != vid);
    // The image line at exactly 256x256 pixels is its anchor point, 1.0.
    try testing.expectApproxEqAbs(@as(f64, 1.0), shiftFactor(1, 32, 32), 1e-9);
    // The video line's anchor is 256x256x37 pixels: 37 frames = latent t 10.
    try testing.expectApproxEqAbs(@as(f64, 1.0), shiftFactor(10, 32, 32), 1e-9);
}

test "oracle: the timestep transform is the identity at T for every resolution" {
    // This is why the one-step path needs no resolution handling. Verified
    // against the reference at t=1000 for five geometries.
    const cases = [_][3]u32{ .{ 1, 90, 160 }, .{ 1, 32, 32 }, .{ 5, 90, 160 }, .{ 1, 135, 240 }, .{ 2, 64, 48 } };
    for (cases) |c| {
        const got = transformTimestep(T, c[0], c[1], c[2]);
        testing.expectApproxEqAbs(T, got, 1e-2) catch |err| {
            std.debug.print("latent {any}: t=1000 transformed to {d}\n", .{ c, got });
            return err;
        };
    }
}

test "oracle: the timestep transform at t=500 matches the reference" {
    // ORACLE. The transform only bites away from T, so a mid-schedule value is
    // the only place its correctness is observable.
    const cases = [_]struct { t: u32, h: u32, w: u32, out: f32 }{
        .{ .t = 1, .h = 90, .w = 160, .out = 744.6265162801 },
        .{ .t = 1, .h = 32, .w = 32, .out = 500.0 },
        .{ .t = 5, .h = 90, .w = 160, .out = 583.9762071742 },
        .{ .t = 1, .h = 135, .w = 240, .out = 846.0107150877 },
        .{ .t = 2, .h = 64, .w = 48, .out = 488.7644035877 },
    };
    for (cases) |c| {
        const got = transformTimestep(500.0, c.t, c.h, c.w);
        testing.expectApproxEqAbs(c.out, got, 1e-3) catch |err| {
            std.debug.print("latent {d}x{d}x{d} t=500: want {d:.6} got {d:.6}\n", .{ c.t, c.h, c.w, c.out, got });
            return err;
        };
    }
}

test "channel budget adds up to the DiT's declared input width" {
    // 16 noisy + 16 condition + 1 mask = 33. This is the number vid_in.proj was
    // built for (its weight is [dim, 33*2*2] = [dim, 132]).
    try testing.expectEqual(@as(u32, 33), VID_IN_CHANNELS);
    try testing.expectEqual(@as(u32, 17), COND_CHANNELS);
    try testing.expectEqual(@as(u32, 132), VID_IN_CHANNELS * 2 * 2);
}

test "the conditioning latent is clean in the SeedVR2 recipe" {
    // SeedVR1's config carries noise_scale 0.25; the SeedVR2 entry point sets
    // cond_noise_scale = 0.0. Inheriting the v1 value degrades restoration with
    // no visible error anywhere.
    try testing.expectEqual(@as(f32, 0.0), COND_NOISE_SCALE);
    try testing.expectEqual(@as(f32, 1.0), OneStep.cfg_scale);
    try testing.expectEqual(@as(u32, 1), OneStep.steps);
}
