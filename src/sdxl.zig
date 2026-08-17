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

// ── Geometry ──
//
// Fixed by the architecture, not read from a checkpoint: every SDXL build
// shares them, and a repo that disagrees is not SDXL. They live here so the
// encoder/UNet/VAE files can assert against ONE copy.

/// The VAE's spatial downsample. A 1024x1024 image is a 128x128 latent.
pub const VAE_SCALE_FACTOR: u32 = 8;

/// Latent channels the UNet works in.
pub const LATENT_CHANNELS: u32 = 4;

/// SDXL's VAE scaling factor. **0.13025**, NOT SD 1.5's 0.18215 — the two are
/// the same field in the same place in `vae/config.json` and differ by 40%, so
/// pasting the familiar number produces images that decode with visibly wrong
/// contrast rather than failing. Read from the checkpoint when present; this is
/// the fallback for a pack that ships no vae config.
pub const VAE_SCALING_FACTOR: f32 = 0.13025;

/// The two text encoders. SDXL concatenates their PENULTIMATE hidden states
/// along the feature axis (768 + 1280 = 2048, the UNet's cross-attention dim)
/// and takes the POOLED output from the bigG tower ALONE for the micro-
/// conditioning embedding. Using the last hidden state instead of the
/// penultimate is a silent quality regression, not an error.
pub const CLIP_L_HIDDEN: u32 = 768;
pub const CLIP_BIG_G_HIDDEN: u32 = 1280;
pub const CROSS_ATTENTION_DIM: u32 = CLIP_L_HIDDEN + CLIP_BIG_G_HIDDEN;

/// The pooled projection fed to the micro-conditioning embedder — bigG's
/// projection dim, which happens to equal its hidden size.
pub const POOLED_PROJECTION_DIM: u32 = CLIP_BIG_G_HIDDEN;

/// Both towers are trained at 77 tokens; longer prompts are truncated by the
/// pipeline (weighted-embedding tricks are a downstream concern).
pub const MAX_PROMPT_TOKENS: u32 = 77;

/// Latent grid for an image of this size. Returns null when the size is not a
/// clean multiple of the VAE scale — the caller decides whether that is a
/// refusal or a snap, exactly as `clampFluxDim` does for FLUX.
pub fn latentDims(width: u32, height: u32) ?struct { w: u32, h: u32 } {
    if (width == 0 or height == 0) return null;
    if (width % VAE_SCALE_FACTOR != 0 or height % VAE_SCALE_FACTOR != 0) return null;
    return .{ .w = width / VAE_SCALE_FACTOR, .h = height / VAE_SCALE_FACTOR };
}

// ── Repo fingerprint ──
//
// DELIBERATELY NOT WIRED into `model_discovery` or `gen.peekModelType` yet.
// A model the server discovers and then cannot load is the incomplete-media-pack
// class: discovery registers it, the loader falls through to something that was
// never meant to read it, and the failure surfaces as a crash rather than a
// named refusal. This predicate lands in discovery in the same change as the
// `ImageBackend` arm that can serve it, never before.

/// diffusers `_class_name` values that describe a checkpoint our SDXL engine
/// would load. The plain and img2img/inpaint pipelines share one UNet, one VAE
/// and the same pair of text encoders — they differ only in how the initial
/// latent is prepared, which is a request-shape question, not a checkpoint one.
pub fn isSdxlPipelineClass(class_name: []const u8) bool {
    const known = [_][]const u8{
        "StableDiffusionXLPipeline",
        "StableDiffusionXLImg2ImgPipeline",
        "StableDiffusionXLInpaintPipeline",
    };
    for (known) |k| if (std.mem.eql(u8, class_name, k)) return true;
    return false;
}

/// True when `model_index.json` bytes describe an SDXL pipeline.
///
/// Keyed on the DECLARED class, never on directory shape: `unet/` + `vae/` +
/// `text_encoder/` describes most of diffusers, and SD 1.5 has all three with
/// only ONE text encoder. The `text_encoder_2` entry is what separates XL from
/// its predecessor, so it is required too — a repo declaring the XL class
/// without it cannot be loaded by an XL engine.
pub fn indexDeclaresSdxl(allocator: std.mem.Allocator, index_json: []const u8) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, index_json, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const obj = parsed.value.object;
    const cn = obj.get("_class_name") orelse return false;
    if (cn != .string or !isSdxlPipelineClass(cn.string)) return false;
    return obj.get("text_encoder_2") != null;
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

test "the two encoder widths sum to the UNet's cross-attention dim" {
    try testing.expectEqual(@as(u32, 2048), CROSS_ATTENTION_DIM);
    try testing.expectEqual(CLIP_L_HIDDEN + CLIP_BIG_G_HIDDEN, CROSS_ATTENTION_DIM);
    // The pooled vector comes from bigG ALONE, so it is 1280 and not 2048 —
    // wiring the concatenated width here is the mistake this pins.
    try testing.expectEqual(@as(u32, 1280), POOLED_PROJECTION_DIM);
}

test "the VAE scaling factor is SDXL's, not SD 1.5's" {
    // Same field, same place in vae/config.json, 40% apart. Pasting the
    // familiar 0.18215 decodes with visibly wrong contrast rather than failing.
    try testing.expectApproxEqAbs(@as(f32, 0.13025), VAE_SCALING_FACTOR, 1e-9);
    try testing.expect(VAE_SCALING_FACTOR != 0.18215);
}

test "latent dims divide by 8 and refuse a size that does not" {
    const a = latentDims(1024, 1024).?;
    try testing.expectEqual(@as(u32, 128), a.w);
    try testing.expectEqual(@as(u32, 128), a.h);
    const b = latentDims(1344, 768).?;
    try testing.expectEqual(@as(u32, 168), b.w);
    try testing.expectEqual(@as(u32, 96), b.h);
    try testing.expect(latentDims(1023, 1024) == null);
    try testing.expect(latentDims(0, 512) == null);
}

test "the sdxl fingerprint keys on the declared class plus the second encoder" {
    const a = testing.allocator;
    const xl =
        \\{"_class_name":"StableDiffusionXLPipeline","unet":["diffusers","UNet2DConditionModel"],
        \\ "text_encoder":["transformers","CLIPTextModel"],
        \\ "text_encoder_2":["transformers","CLIPTextModelWithProjection"]}
    ;
    try testing.expect(indexDeclaresSdxl(a, xl));

    // SD 1.5 has unet + vae + text_encoder and is NOT XL — the directory shape
    // alone cannot tell them apart, which is why the class is the key.
    const sd15 =
        \\{"_class_name":"StableDiffusionPipeline","unet":["diffusers","UNet2DConditionModel"],
        \\ "text_encoder":["transformers","CLIPTextModel"]}
    ;
    try testing.expect(!indexDeclaresSdxl(a, sd15));

    // Declaring the XL class without the second tower is not loadable by an XL
    // engine, so it is not a match.
    const half =
        \\{"_class_name":"StableDiffusionXLPipeline","unet":["diffusers","UNet2DConditionModel"]}
    ;
    try testing.expect(!indexDeclaresSdxl(a, half));

    try testing.expect(!indexDeclaresSdxl(a, "not json"));
    try testing.expect(!indexDeclaresSdxl(a, "[]"));
    try testing.expect(!indexDeclaresSdxl(a, "{}"));
}

test "img2img and inpaint share the checkpoint, so they share the fingerprint" {
    try testing.expect(isSdxlPipelineClass("StableDiffusionXLPipeline"));
    try testing.expect(isSdxlPipelineClass("StableDiffusionXLImg2ImgPipeline"));
    try testing.expect(isSdxlPipelineClass("StableDiffusionXLInpaintPipeline"));
    try testing.expect(!isSdxlPipelineClass("StableDiffusionPipeline"));
    try testing.expect(!isSdxlPipelineClass("FluxPipeline"));
    try testing.expect(!isSdxlPipelineClass(""));
}

test "SDXL is not yet reachable from discovery or the gen dispatch" {
    // Class guard, not a feature test: this file is deliberately unwired until
    // an ImageBackend arm exists. If someone adds the model_type without the
    // engine, discovery registers a model whose load falls through to a reader
    // that was never meant to see it — the incomplete-media-pack class.
    const gen = @import("gen.zig");
    try testing.expect(gen.modalityFromType("sdxl") == null);
    try testing.expect(gen.modalityFromType("stable_diffusion_xl") == null);
}
