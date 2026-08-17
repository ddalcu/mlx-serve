//! Stable Diffusion XL — scheduler + micro-conditioning math.
//!
//! FIRST SLICE of the SDXL port, deliberately weight-free and MLX-free: every
//! function here is pure arithmetic over f32 slices, so it is testable with no
//! checkpoint, no GPU and no oracle. The tensor work (two CLIP text encoders, the
//! UNet, the VAE) lands separately; nothing in this file is wired into
//! `gen.ImageBackend` yet, and discovery is deliberately NOT taught about SDXL
//! repos until an engine arm exists to load one — a model the server advertises
//! and then cannot load is worse than one it ignores.
//!
//! SDXL differs from every image backend already served here in three ways that
//! shape this file:
//!
//!   1. It is EPSILON-prediction on a DISCRETE beta schedule, not flow-matching.
//!      krea/flux/mage_flow all integrate a flow field over t in [0,1]; SDXL
//!      integrates sigmas derived from `alphas_cumprod` over 1000 train steps.
//!   2. It needs real CFG (two forwards per step, or one batch-2 forward). The
//!      distilled backends here run guidance-free at 4-8 steps.
//!   3. Its conditioning carries a MICRO-CONDITIONING vector (`add_time_ids`)
//!      describing the training crop, alongside the pooled text embedding.
//!      Getting it wrong does not error — it shifts composition and framing,
//!      which reads as a bad checkpoint.
//!
//! ORACLE STATUS: the constants and formulas are transcribed from diffusers'
//! `EulerDiscreteScheduler` + `StableDiffusionXLPipeline`. They are NOT yet
//! pinned against an executed reference — the tests below assert invariants and
//! hand-computable values only. Before this ships, it needs a fixture dumped
//! from diffusers (the `tests/dump_*_fixtures.py` pattern), because a scheduler
//! that is subtly wrong produces plausible images, not obvious failures.

const std = @import("std");

// ── Training schedule constants (SDXL base + refiner share these) ──

pub const NUM_TRAIN_TIMESTEPS: usize = 1000;
pub const BETA_START: f64 = 0.00085;
pub const BETA_END: f64 = 0.012;

/// How inference timesteps are spread over the 1000 training steps. SDXL's
/// config declares "leading"; the others exist because a checkpoint's own
/// `scheduler_config.json` may say otherwise and silently picking ours would be
/// the `LtxVersion` class — a config field deciding numerics.
pub const TimestepSpacing = enum { leading, trailing, linspace };

/// `beta_schedule: "scaled_linear"` — betas are the SQUARE of a linear ramp
/// between the square roots of the endpoints, not a linear ramp between the
/// endpoints. This is the single easiest constant to transcribe wrongly, and
/// doing so yields a schedule that still denoises, just to the wrong picture.
pub fn scaledLinearBetas(out: []f64) void {
    const n = out.len;
    if (n == 0) return;
    const lo = @sqrt(BETA_START);
    const hi = @sqrt(BETA_END);
    if (n == 1) {
        out[0] = lo * lo;
        return;
    }
    const step = (hi - lo) / @as(f64, @floatFromInt(n - 1));
    for (out, 0..) |*b, i| {
        const v = lo + step * @as(f64, @floatFromInt(i));
        b.* = v * v;
    }
}

/// `alphas_cumprod[i] = prod(1 - beta[0..=i])`.
pub fn alphasCumprod(betas: []const f64, out: []f64) void {
    std.debug.assert(betas.len == out.len);
    var running: f64 = 1.0;
    for (betas, 0..) |b, i| {
        running *= (1.0 - b);
        out[i] = running;
    }
}

/// The full training sigma ladder: `sqrt((1 - acp) / acp)`, ASCENDING in i
/// (sigma grows as alphas_cumprod decays).
pub fn trainSigmas(acp: []const f64, out: []f64) void {
    std.debug.assert(acp.len == out.len);
    for (acp, 0..) |a, i| out[i] = @sqrt((1.0 - a) / a);
}

/// The training timestep indices for `steps` inference steps.
///
/// "leading" walks 0, ratio, 2*ratio, … with `ratio = num_train // steps`, then
/// REVERSES so sampling runs high-noise → low-noise. diffusers adds
/// `steps_offset` (1 for SDXL) to each; that offset is the caller's, kept out of
/// here so the spacing rule stays one idea.
pub fn timestepIndices(spacing: TimestepSpacing, steps: usize, out: []usize) void {
    std.debug.assert(out.len == steps);
    if (steps == 0) return;
    const n = NUM_TRAIN_TIMESTEPS;
    switch (spacing) {
        .leading => {
            const ratio = n / steps;
            for (out, 0..) |*t, i| t.* = (steps - 1 - i) * ratio;
        },
        .trailing => {
            const ratio = n / steps;
            // n - 1, n - 1 - ratio, … (already descending)
            for (out, 0..) |*t, i| t.* = (n - 1) -| (i * ratio);
        },
        .linspace => {
            if (steps == 1) {
                out[0] = n - 1;
                return;
            }
            const span = @as(f64, @floatFromInt(n - 1));
            const denom = @as(f64, @floatFromInt(steps - 1));
            for (out, 0..) |*t, i| {
                const asc = span * @as(f64, @floatFromInt(steps - 1 - i)) / denom;
                t.* = @intFromFloat(@round(asc));
            }
        },
    }
}

/// The inference sigma ladder: the training sigmas sampled at `indices`, with a
/// terminal 0.0 appended. `out.len` must be `indices.len + 1` — that trailing
/// zero is what makes the last Euler step land on a clean latent, and omitting
/// it leaves the final image one step short of denoised.
pub fn inferenceSigmas(train: []const f64, indices: []const usize, out: []f64) void {
    std.debug.assert(out.len == indices.len + 1);
    for (indices, 0..) |t, i| out[i] = train[t];
    out[indices.len] = 0.0;
}

/// What a fresh latent is scaled by before the first step:
/// `sqrt(max_sigma^2 + 1)`. NOT `max_sigma` — the Euler formulation keeps the
/// latent in a `sqrt(sigma^2 + 1)`-normalised space.
pub fn initNoiseSigma(sigmas: []const f64) f64 {
    var m: f64 = 0.0;
    for (sigmas) |s| m = @max(m, s);
    return @sqrt(m * m + 1.0);
}

/// `scale_model_input`: what the UNet is actually handed at this sigma.
pub fn scaleModelInput(sigma: f64) f64 {
    return 1.0 / @sqrt(sigma * sigma + 1.0);
}

/// One Euler step for an EPSILON-prediction model, as a pair of scalar
/// coefficients applied per-element: `next = a*sample + b*eps`.
///
/// Derivation (diffusers `EulerDiscreteScheduler.step`):
///   pred_x0    = sample - sigma * eps
///   derivative = (sample - pred_x0) / sigma  =  eps
///   next       = sample + derivative * (sigma_next - sigma)
/// so the sample coefficient is 1 and the eps coefficient is `sigma_next - sigma`.
/// Returned as a struct anyway: an ancestral or v-prediction variant changes
/// BOTH, and a caller written against a bare scalar would silently keep the 1.
pub const EulerStep = struct { sample: f64, eps: f64 };

pub fn eulerStep(sigma: f64, sigma_next: f64) EulerStep {
    return .{ .sample = 1.0, .eps = sigma_next - sigma };
}

/// Classifier-free guidance: `uncond + scale * (cond - uncond)`, returned as
/// coefficients so the caller can apply them to whole tensors.
/// scale <= 1 collapses to the conditional branch alone.
pub const CfgMix = struct { uncond: f64, cond: f64 };

pub fn cfgMix(scale: f64) CfgMix {
    return .{ .uncond = 1.0 - scale, .cond = scale };
}

// ── Micro-conditioning ──

/// SDXL's `add_time_ids`: SIX values in this exact order —
/// `original_size(h, w) ++ crops_coords_top_left(top, left) ++ target_size(h, w)`.
///
/// Height precedes width in every pair, which is the opposite of the `WxH`
/// spelling the HTTP surface uses, and nothing downstream can detect the swap:
/// the UNet consumes them as a sinusoidal embedding, so a transposed pair
/// produces a coherent image framed for the wrong aspect. `original_size`
/// declares the resolution the image is meant to look like it was TRAINED at
/// (upstream default: the target size), and `crops_coords_top_left` of (0,0)
/// means "uncropped", which is what makes subjects centred rather than cut off.
pub const TimeIds = [6]f32;

pub fn addTimeIds(
    original_h: u32,
    original_w: u32,
    crop_top: u32,
    crop_left: u32,
    target_h: u32,
    target_w: u32,
) TimeIds {
    return .{
        @floatFromInt(original_h), @floatFromInt(original_w),
        @floatFromInt(crop_top),   @floatFromInt(crop_left),
        @floatFromInt(target_h),   @floatFromInt(target_w),
    };
}

/// The defaults the pipeline uses when a request says nothing: the source is
/// declared to be the size being generated, uncropped.
pub fn defaultTimeIds(height: u32, width: u32) TimeIds {
    return addTimeIds(height, width, 0, 0, height, width);
}

// ════════════════════════════════════════════════════════════════════════
// Tests — invariants and hand-computable values only. See ORACLE STATUS above:
// none of this is yet pinned against an executed diffusers reference.
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "scaled_linear betas square a ramp between the SQRT endpoints" {
    var betas: [NUM_TRAIN_TIMESTEPS]f64 = undefined;
    scaledLinearBetas(&betas);
    // Endpoints are exact by construction.
    try testing.expectApproxEqAbs(BETA_START, betas[0], 1e-12);
    try testing.expectApproxEqAbs(BETA_END, betas[betas.len - 1], 1e-12);
    // The midpoint is the giveaway: a LINEAR schedule would put it at
    // (start+end)/2 = 0.006425. scaled_linear puts it well below that.
    const mid = betas[betas.len / 2];
    const linear_mid = (BETA_START + BETA_END) / 2.0;
    try testing.expect(mid < linear_mid);
    // Monotonic increasing throughout.
    for (1..betas.len) |i| try testing.expect(betas[i] > betas[i - 1]);
}

test "alphas_cumprod decays monotonically and stays in (0, 1)" {
    var betas: [NUM_TRAIN_TIMESTEPS]f64 = undefined;
    var acp: [NUM_TRAIN_TIMESTEPS]f64 = undefined;
    scaledLinearBetas(&betas);
    alphasCumprod(&betas, &acp);
    try testing.expect(acp[0] < 1.0 and acp[0] > 0.999);
    for (1..acp.len) |i| try testing.expect(acp[i] < acp[i - 1]);
    try testing.expect(acp[acp.len - 1] > 0.0);
}

test "train sigmas ascend with the timestep index" {
    var betas: [NUM_TRAIN_TIMESTEPS]f64 = undefined;
    var acp: [NUM_TRAIN_TIMESTEPS]f64 = undefined;
    var sig: [NUM_TRAIN_TIMESTEPS]f64 = undefined;
    scaledLinearBetas(&betas);
    alphasCumprod(&betas, &acp);
    trainSigmas(&acp, &sig);
    for (1..sig.len) |i| try testing.expect(sig[i] > sig[i - 1]);
    // sigma_0 is small (barely any noise) and sigma_max is large.
    try testing.expect(sig[0] < 0.05);
    try testing.expect(sig[sig.len - 1] > 10.0);
}

test "leading spacing walks the training steps and runs high noise first" {
    var idx: [50]usize = undefined;
    timestepIndices(.leading, 50, &idx);
    // ratio = 1000/50 = 20, reversed → 980, 960, …, 20, 0.
    try testing.expectEqual(@as(usize, 980), idx[0]);
    try testing.expectEqual(@as(usize, 960), idx[1]);
    try testing.expectEqual(@as(usize, 0), idx[49]);
    for (1..idx.len) |i| try testing.expect(idx[i] < idx[i - 1]);
}

test "every spacing is strictly descending and in range" {
    for ([_]TimestepSpacing{ .leading, .trailing, .linspace }) |sp| {
        for ([_]usize{ 1, 4, 25, 30, 50 }) |steps| {
            var buf: [50]usize = undefined;
            const idx = buf[0..steps];
            timestepIndices(sp, steps, idx);
            for (idx) |t| try testing.expect(t < NUM_TRAIN_TIMESTEPS);
            for (1..idx.len) |i| {
                try testing.expect(idx[i] < idx[i - 1]);
            }
        }
    }
}

test "the inference ladder ends at exactly zero" {
    var betas: [NUM_TRAIN_TIMESTEPS]f64 = undefined;
    var acp: [NUM_TRAIN_TIMESTEPS]f64 = undefined;
    var train: [NUM_TRAIN_TIMESTEPS]f64 = undefined;
    scaledLinearBetas(&betas);
    alphasCumprod(&betas, &acp);
    trainSigmas(&acp, &train);

    var idx: [30]usize = undefined;
    timestepIndices(.leading, 30, &idx);
    var sigmas: [31]f64 = undefined;
    inferenceSigmas(&train, &idx, &sigmas);

    try testing.expectEqual(@as(f64, 0.0), sigmas[30]);
    // Descending, because the indices descend and train sigmas ascend.
    for (1..sigmas.len) |i| try testing.expect(sigmas[i] < sigmas[i - 1]);
}

test "init noise sigma is sqrt(max^2 + 1), not max" {
    const sigmas = [_]f64{ 14.6146, 5.0, 1.0, 0.0 };
    const got = initNoiseSigma(&sigmas);
    try testing.expectApproxEqRel(@sqrt(14.6146 * 14.6146 + 1.0), got, 1e-12);
    // The distinction is small but real, and always upward.
    try testing.expect(got > 14.6146);
}

test "scale_model_input is 1 at sigma 0 and shrinks as sigma grows" {
    try testing.expectApproxEqAbs(@as(f64, 1.0), scaleModelInput(0.0), 1e-12);
    try testing.expect(scaleModelInput(14.6) < scaleModelInput(1.0));
    try testing.expect(scaleModelInput(1.0) < 1.0);
}

test "an euler step moves the sample by eps times the sigma delta" {
    const s = eulerStep(10.0, 6.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), s.sample, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -4.0), s.eps, 1e-12);

    // The final step (sigma_next == 0) must land exactly on pred_x0:
    // next = sample - sigma*eps, which IS the x0 prediction.
    const last = eulerStep(3.0, 0.0);
    try testing.expectApproxEqAbs(@as(f64, -3.0), last.eps, 1e-12);
}

test "cfg at scale 1 is the conditional branch alone" {
    const one = cfgMix(1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), one.uncond, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), one.cond, 1e-12);
    // The coefficients always sum to 1 — it is an extrapolation along the
    // uncond→cond line, so a scale of 7.5 overshoots the conditional.
    const g = cfgMix(7.5);
    try testing.expectApproxEqAbs(@as(f64, 1.0), g.uncond + g.cond, 1e-12);
    try testing.expect(g.uncond < 0.0);
}

test "add_time_ids is height-first in every pair" {
    // A landscape target: 1344 wide by 768 tall. Height leads.
    const ids = defaultTimeIds(768, 1344);
    try testing.expectEqual(@as(f32, 768), ids[0]); // original h
    try testing.expectEqual(@as(f32, 1344), ids[1]); // original w
    try testing.expectEqual(@as(f32, 0), ids[2]); // crop top
    try testing.expectEqual(@as(f32, 0), ids[3]); // crop left
    try testing.expectEqual(@as(f32, 768), ids[4]); // target h
    try testing.expectEqual(@as(f32, 1344), ids[5]); // target w
    // Guard against the transpose that produces a coherent, wrongly-framed
    // image: on a non-square size the pair must not be equal.
    try testing.expect(ids[0] != ids[1]);
}

test "explicit crop coords survive into the vector" {
    const ids = addTimeIds(1024, 1024, 64, 32, 768, 512);
    try testing.expectEqual(@as(f32, 64), ids[2]);
    try testing.expectEqual(@as(f32, 32), ids[3]);
    try testing.expectEqual(@as(f32, 768), ids[4]);
    try testing.expectEqual(@as(f32, 512), ids[5]);
}
