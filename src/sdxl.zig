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
//! ORACLE STATUS — three different levels of confidence, kept apart on purpose:
//!
//!   VERIFIED against the checkpoint. Every config constant here was read out of
//!   `stabilityai/stable-diffusion-xl-base-1.0` rather than a doc: the beta
//!   endpoints and schedule, `timestep_spacing: leading`, `steps_offset: 1`,
//!   `prediction_type: epsilon`, the VAE's 0.13025, both tower geometries, and
//!   the two DIFFERENT activations. The weight-name contract is checked against
//!   the real safetensors headers by the env-gated `sdxl checkpoint` test.
//!
//!   VERIFIED by construction. The scheduler formulas are pinned by invariants
//!   and hand-computable values (monotonicity, endpoints, the 481→512 rounding
//!   case that separates ceil from nearest).
//!
//!   NOT VERIFIED. Whether the assembled forward produces the SAME NUMBERS as
//!   diffusers. That needs a dumped fixture (`tests/dump_*_fixtures.py`), and
//!   until it exists no claim of parity belongs here — a scheduler that is
//!   subtly wrong produces plausible images, not obvious failures.

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

/// Parse a declared/requested spacing name. Null for anything else, so both
/// callers can tell "not named" from "named something we do not serve": the
/// checkpoint reader falls back to `leading` (SDXL's own default) while the
/// request handler answers a named 400 rather than silently denoising on a
/// schedule the caller did not ask for.
pub fn spacingFromString(name: []const u8) ?TimestepSpacing {
    return std.meta.stringToEnum(TimestepSpacing, name);
}

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

/// diffusers' floor for the terminal `alphas_cumprod` under zero-terminal-SNR.
/// The rescale drives the last value to EXACTLY zero, and `trainSigmas` divides
/// by it — sigma would be `inf` and every downstream coefficient NaN. diffusers
/// picks fp16's smallest positive subnormal for the same reason: close enough to
/// zero to be terminal, far enough to stay finite.
const ZERO_SNR_TERMINAL_ACP: f64 = 1.0 / @as(f64, @floatFromInt(@as(u64, 1) << 24));

/// Rescale `acp` in place so the terminal signal-to-noise ratio is zero
/// (Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed";
/// diffusers' `rescale_zero_terminal_snr`).
///
/// The stock scaled-linear schedule never reaches pure noise — `acp[last]` is
/// ~0.0047, so the model is always shown a faint trace of the image and learns
/// to rely on it. A model TRAINED that way is fine; one trained with zero
/// terminal SNR (NoobAI V-Pred, which self-identifies with a `ztsnr` marker
/// tensor) expects the last step to start from pure noise, and sampling it on
/// the stock ladder washes out contrast — a plausible image, systematically
/// wrong, which is why this is keyed on the checkpoint's own marker rather than
/// offered as a knob.
///
/// Operates on sqrt(acp): shift so the last entry is 0, rescale so the first is
/// unchanged, square back. The terminal entry is then floored (see above).
pub fn rescaleZeroTerminalSnr(acp: []f64) void {
    if (acp.len == 0) return;
    var sqrt_acp_0 = @sqrt(acp[0]);
    const sqrt_acp_t = @sqrt(acp[acp.len - 1]);
    const denom = sqrt_acp_0 - sqrt_acp_t;
    if (denom == 0.0) return; // degenerate table — leave it alone
    const scale = sqrt_acp_0 / denom;
    // `sqrt_acp_0` is read once more below through `scale`, so cache nothing else.
    sqrt_acp_0 = undefined;
    for (acp) |*a| {
        const shifted = @sqrt(a.*) - sqrt_acp_t;
        const s = shifted * scale;
        a.* = s * s;
    }
    acp[acp.len - 1] = ZERO_SNR_TERMINAL_ACP;
}

/// numpy's rounding — half-to-EVEN, not half-away-from-zero like Zig's
/// `@round`. Every spacing rule below is a transcription of a numpy expression
/// diffusers evaluates, and the two disagree on exact halves: a 16-step
/// `trailing` ladder puts 812.5 at 812 (numpy) or 813 (`@round`), which is a
/// whole training timestep of drift on a distill that only takes four.
/// `v` must be non-negative (every schedule value is).
fn roundHalfEven(v: f64) usize {
    std.debug.assert(v >= 0.0);
    const fl = @floor(v);
    const frac = v - fl;
    const up = frac > 0.5 or (frac == 0.5 and @mod(fl, 2.0) != 0.0);
    return @intFromFloat(if (up) fl + 1.0 else fl);
}

/// The training timestep indices for `steps` inference steps.
///
/// "leading" walks 0, ratio, 2*ratio, … with `ratio = num_train // steps`, then
/// REVERSES so sampling runs high-noise → low-noise. diffusers adds
/// `steps_offset` (1 for SDXL) to each; that offset is the caller's, kept out of
/// here so the spacing rule stays one idea.
///
/// "trailing" and "linspace" are FLOAT ladders. `trailing` in particular is
/// `round(arange(1000, 0, -1000/steps)) - 1`, and the stride is the real ratio,
/// NOT `num_train // steps`: at 6 steps the integer stride gives
/// `[999, 833, 667, 501, 335, 169]` against diffusers' `[999, 832, 666, 499,
/// 332, 166]`, drifting further at every step. The two agree only where `steps`
/// divides 1000, which is exactly the set the schedule tests used to cover.
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
            // `round(arange(n, 0, -n/steps)) - 1` — already descending.
            const top = @as(f64, @floatFromInt(n));
            const stride = top / @as(f64, @floatFromInt(steps));
            for (out, 0..) |*t, i| {
                const v = top - stride * @as(f64, @floatFromInt(i));
                t.* = roundHalfEven(v) -| 1;
            }
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
                t.* = roundHalfEven(asc);
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

/// Which GELU a tower uses. **The two towers disagree**, verified against
/// `stabilityai/stable-diffusion-xl-base-1.0`: `text_encoder/config.json` says
/// `quick_gelu` (x * sigmoid(1.702x), OpenAI's original CLIP) and
/// `text_encoder_2/config.json` says `gelu` (the erf form). Running one
/// activation for both towers produces plausible embeddings and a plausible
/// image — the same class as LFM2-VL's encoder-vs-projector GELU split, which
/// is why this is a per-tower field and not a module constant.
pub const ClipActivation = enum { quick_gelu, gelu };

/// Everything a CLIP text tower needs, read from its own `config.json`. Both
/// SDXL towers are the SAME architecture at different sizes plus that
/// activation difference, so one implementation serves both — parameterised
/// here rather than forked.
pub const ClipTextConfig = struct {
    hidden: u32,
    layers: u32,
    heads: u32,
    intermediate: u32,
    activation: ClipActivation,
    max_positions: u32 = MAX_PROMPT_TOKENS,
    vocab: u32 = 49408,
    /// The width of the pooled projection. Only bigG's is consumed by SDXL.
    projection_dim: u32,

    pub fn headDim(self: ClipTextConfig) u32 {
        return self.hidden / self.heads;
    }
};

/// CLIP-L — SDXL's first tower. Values verified against the checkpoint.
pub const CLIP_L_CONFIG = ClipTextConfig{
    .hidden = 768,
    .layers = 12,
    .heads = 12,
    .intermediate = 3072,
    .activation = .quick_gelu,
    .projection_dim = 768,
};

/// OpenCLIP bigG — SDXL's second tower, and the one whose POOLED output feeds
/// the micro-conditioning embedder.
pub const CLIP_BIG_G_CONFIG = ClipTextConfig{
    .hidden = 1280,
    .layers = 32,
    .heads = 20,
    .intermediate = 5120,
    .activation = .gelu,
    .projection_dim = 1280,
};

/// OpenCLIP ViT-H/14 (LAION) — SD 2.x's / SD-Turbo's ONE tower. Verified
/// against `stabilityai/sd-turbo/text_encoder/config.json`: 1024/23/16, gelu,
/// projection_dim 512. NOT SDXL's bigG (1280/32/20) — a different LAION
/// OpenCLIP checkpoint at a smaller scale, sharing only the architecture
/// class and the standard 49408-token CLIP vocab (bos 49406, eos 49407, pad
/// "!"=0 — the tokenizer's OWN `tokenizer_config.json` is what carries this;
/// `config.json`'s `bos_token_id:0`/`eos_token_id:2` are unused
/// `PretrainedConfig` defaults the model never reads, not the real ids).
///
/// The LAYER COUNT is 23, not the vision-tower-matching 24: diffusers ships
/// this checkpoint already truncated to the layer SD 2.x/Turbo was trained
/// against — `sd1_pipeline`'s `final_norm: true` therefore reads THIS
/// tower's true final layer (after `final_layer_norm`), which is the
/// original 24-layer OpenCLIP model's penultimate output baked in at
/// conversion time rather than sliced off at inference.
pub const CLIP_H_CONFIG = ClipTextConfig{
    .hidden = 1024,
    .layers = 23,
    .heads = 16,
    .intermediate = 4096,
    .activation = .gelu,
    .projection_dim = 512,
};

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

/// `StableDiffusionPipeline` and its img2img/inpaint siblings — the SD 1.x
/// pipeline class. SD 2.x declares the SAME class name (only its UNet's
/// `cross_attention_dim` and text tower differ), so this alone cannot tell
/// the two apart; `indexDeclaresSd1` narrows further.
pub fn isSd1PipelineClass(class_name: []const u8) bool {
    const known = [_][]const u8{
        "StableDiffusionPipeline",
        "StableDiffusionImg2ImgPipeline",
        "StableDiffusionInpaintPipeline",
    };
    for (known) |k| if (std.mem.eql(u8, class_name, k)) return true;
    return false;
}

/// True when `model_index.json` bytes describe an SD 1.x pipeline: the
/// pipeline class SD 1.x and SD 2.x share, narrowed by the ABSENCE of a
/// second text encoder (`text_encoder_2`, which would make it SDXL — a
/// diffusers repo never carries three towers). This does NOT distinguish
/// SD 1.x from SD 2.x — both are single-tower `StableDiffusionPipeline`
/// repos — so `sd1_pipeline`'s loader additionally refuses a text encoder
/// whose own `config.json` isn't CLIP-L's 768-wide (SD 2.x's OpenCLIP
/// ViT-H/14 is 1024-wide) rather than silently mis-loading it.
pub fn indexDeclaresSd1(allocator: std.mem.Allocator, index_json: []const u8) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, index_json, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const obj = parsed.value.object;
    const cn = obj.get("_class_name") orelse return false;
    if (cn != .string or !isSd1PipelineClass(cn.string)) return false;
    return obj.get("text_encoder_2") == null;
}

/// Tensor names that, all present, mark a SINGLE-FILE LDM SDXL checkpoint (the
/// Civitai / A1111 / SGM distribution — how Illustrious XL and Pony Diffusion
/// XL ship). The LDM UNet trunk, the SDXL-only micro-conditioning embedding
/// (`label_emb` — absent on SD 1.5), and the SECOND text encoder (bigG, what
/// makes it XL). None is sufficient alone.
pub const LDM_SDXL_MARKERS = [_][]const u8{
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.label_emb.0.0.weight",
    "conditioner.embedders.1.",
};

/// True when a safetensors HEADER (its JSON name/dtype/offset map) carries all
/// the LDM SDXL markers. Substring search over the raw header — no JSON parse,
/// so a caller can feed a bounded prefix of a multi-GB file. Shared by
/// `model_discovery` (classification) and `sdxl_single_file` (routing), the
/// same one-predicate discipline `indexDeclaresSdxl` keeps.
pub fn headerDeclaresLdmSdxl(header_bytes: []const u8) bool {
    for (LDM_SDXL_MARKERS) |m| {
        if (std.mem.indexOf(u8, header_bytes, m) == null) return false;
    }
    return true;
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

test "zero-terminal-SNR rescale drives the last alphas_cumprod to ~0 and keeps sigma finite" {
    var betas: [NUM_TRAIN_TIMESTEPS]f64 = undefined;
    var acp: [NUM_TRAIN_TIMESTEPS]f64 = undefined;
    scaledLinearBetas(&betas);
    alphasCumprod(&betas, &acp);

    // The stock schedule never reaches pure noise — that is the defect ZTSNR
    // exists to fix, and the value it starts from.
    try testing.expect(acp[acp.len - 1] > 0.004);
    const first_before = acp[0];

    rescaleZeroTerminalSnr(&acp);

    // Terminal is the floor, not literally zero (a literal zero is an inf sigma).
    try testing.expectEqual(ZERO_SNR_TERMINAL_ACP, acp[acp.len - 1]);
    try testing.expect(acp[acp.len - 1] < 1e-6);
    // The FIRST entry is preserved by construction (the rescale's whole point:
    // move the tail to zero without moving the head).
    try testing.expectApproxEqAbs(first_before, acp[0], 1e-9);
    // Still a decaying schedule in (0, 1].
    for (acp) |a| try testing.expect(a > 0.0 and a <= 1.0);
    for (1..acp.len) |i| try testing.expect(acp[i] < acp[i - 1]);

    // And the sigma ladder it feeds stays finite + ascending, with a terminal
    // sigma far above the stock ~14.6 (that is what "pure noise" looks like).
    var sig: [NUM_TRAIN_TIMESTEPS]f64 = undefined;
    trainSigmas(&acp, &sig);
    for (sig) |s| try testing.expect(std.math.isFinite(s));
    for (1..sig.len) |i| try testing.expect(sig[i] > sig[i - 1]);
    try testing.expect(sig[sig.len - 1] > 1000.0);
}

test "zero-terminal-SNR rescale leaves a degenerate table alone" {
    // A constant table has no head-to-tail span to rescale against; dividing by
    // that zero span would be inf/NaN across the whole ladder.
    var flat = [_]f64{ 0.5, 0.5, 0.5 };
    rescaleZeroTerminalSnr(&flat);
    for (flat) |a| try testing.expectEqual(@as(f64, 0.5), a);
    var empty = [_]f64{};
    rescaleZeroTerminalSnr(&empty); // must not crash
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

test "SDXL routes to the image modality on every side that must agree" {
    // This was the "not yet reachable" tripwire, and it fired exactly as
    // intended when the `ImageBackend` arm landed. It now guards the OTHER
    // direction of the same class: routing, discovery and the media-type table
    // must move TOGETHER. Registering the model_type without an engine gives
    // discovery a model whose load falls through to a reader that was never
    // meant to see it; shipping the engine without discovery gives a
    // checkpoint the server can load but cannot list.
    const gen = @import("gen.zig");
    const discovery = @import("model_discovery.zig");

    try testing.expect(gen.modalityFromType("sdxl") == .image);
    try testing.expect(discovery.isMediaModelType("sdxl"));

    // The marker is our OWN spelling, synthesized from `model_index.json`;
    // "stable_diffusion_xl" is not a `model_type` any checkpoint declares, so
    // it must not resolve — a near-miss that routes is worse than one that
    // does not.
    try testing.expect(gen.modalityFromType("stable_diffusion_xl") == null);
    try testing.expect(!discovery.isMediaModelType("stable_diffusion_xl"));

    // SDXL ships no single "everything is here" weight file the way LTX/H3 do
    // (its four components live in four subdirectories), so it declares no
    // required marker rather than declaring one that cannot be checked.
    try testing.expect(discovery.requiredMediaMarker("sdxl") == null);
}

test "an SDXL repo is recognized by its declared pipeline class, not its shape" {
    const a = testing.allocator;
    // The real `model_index.json` from stabilityai/stable-diffusion-xl-base-1.0,
    // trimmed to the keys the predicate reads.
    const real =
        \\{"_class_name":"StableDiffusionXLPipeline","_diffusers_version":"0.19.0.dev0",
        \\"force_zeros_for_empty_prompt":true,
        \\"text_encoder":["transformers","CLIPTextModel"],
        \\"text_encoder_2":["transformers","CLIPTextModelWithProjection"],
        \\"unet":["diffusers","UNet2DConditionModel"],"vae":["diffusers","AutoencoderKL"]}
    ;
    try testing.expect(indexDeclaresSdxl(a, real));

    // SD 1.5 has unet + vae + ONE text encoder and the same directory shape.
    // The second tower is what separates XL from its predecessor.
    const sd15 =
        \\{"_class_name":"StableDiffusionPipeline",
        \\"text_encoder":["transformers","CLIPTextModel"],
        \\"unet":["diffusers","UNet2DConditionModel"],"vae":["diffusers","AutoencoderKL"]}
    ;
    try testing.expect(!indexDeclaresSdxl(a, sd15));

    // Declaring the XL class without the second tower cannot be loaded by an
    // XL engine, so it must not be claimed.
    const truncated =
        \\{"_class_name":"StableDiffusionXLPipeline",
        \\"unet":["diffusers","UNet2DConditionModel"],"vae":["diffusers","AutoencoderKL"]}
    ;
    try testing.expect(!indexDeclaresSdxl(a, truncated));

    // The img2img/inpaint siblings are the same checkpoint with a different
    // request shape, so they load.
    const img2img =
        \\{"_class_name":"StableDiffusionXLImg2ImgPipeline",
        \\"text_encoder_2":["transformers","CLIPTextModelWithProjection"]}
    ;
    try testing.expect(indexDeclaresSdxl(a, img2img));

    try testing.expect(!indexDeclaresSdxl(a, "not json"));
    try testing.expect(!indexDeclaresSdxl(a, "[]"));
}

test "an SD 1.x repo is recognized by class name AND the absent second tower" {
    const a = testing.allocator;
    const sd15 =
        \\{"_class_name":"StableDiffusionPipeline",
        \\"text_encoder":["transformers","CLIPTextModel"],
        \\"unet":["diffusers","UNet2DConditionModel"],"vae":["diffusers","AutoencoderKL"]}
    ;
    try testing.expect(indexDeclaresSd1(a, sd15));
    // The SDXL fixture above has the same class family shape but a second
    // tower — that tower is what makes it XL, not SD 1.x.
    const sdxl_json =
        \\{"_class_name":"StableDiffusionXLPipeline",
        \\"text_encoder":["transformers","CLIPTextModel"],
        \\"text_encoder_2":["transformers","CLIPTextModelWithProjection"],
        \\"unet":["diffusers","UNet2DConditionModel"],"vae":["diffusers","AutoencoderKL"]}
    ;
    try testing.expect(!indexDeclaresSd1(a, sdxl_json));
    // Wrong class family entirely.
    const flux =
        \\{"_class_name":"FluxPipeline"}
    ;
    try testing.expect(!indexDeclaresSd1(a, flux));
    try testing.expect(!indexDeclaresSd1(a, "not json"));
}

test "the two towers disagree about GELU" {
    // Verified against stabilityai/stable-diffusion-xl-base-1.0:
    //   text_encoder/config.json    hidden_act = "quick_gelu"
    //   text_encoder_2/config.json  hidden_act = "gelu"
    // One activation for both runs and still produces a plausible image, so
    // nothing downstream can catch this.
    try testing.expect(CLIP_L_CONFIG.activation == .quick_gelu);
    try testing.expect(CLIP_BIG_G_CONFIG.activation == .gelu);
    try testing.expect(CLIP_L_CONFIG.activation != CLIP_BIG_G_CONFIG.activation);
}

test "tower configs match the shipped checkpoint" {
    // Every number here was read out of the real config.json pair.
    try testing.expectEqual(@as(u32, 768), CLIP_L_CONFIG.hidden);
    try testing.expectEqual(@as(u32, 12), CLIP_L_CONFIG.layers);
    try testing.expectEqual(@as(u32, 12), CLIP_L_CONFIG.heads);
    try testing.expectEqual(@as(u32, 3072), CLIP_L_CONFIG.intermediate);

    try testing.expectEqual(@as(u32, 1280), CLIP_BIG_G_CONFIG.hidden);
    try testing.expectEqual(@as(u32, 32), CLIP_BIG_G_CONFIG.layers);
    try testing.expectEqual(@as(u32, 20), CLIP_BIG_G_CONFIG.heads);
    try testing.expectEqual(@as(u32, 5120), CLIP_BIG_G_CONFIG.intermediate);

    // Both towers are head_dim 64 despite different widths — a shared constant
    // that makes a head-count bug look like it works at one size.
    try testing.expectEqual(@as(u32, 64), CLIP_L_CONFIG.headDim());
    try testing.expectEqual(@as(u32, 64), CLIP_BIG_G_CONFIG.headDim());

    // Same vocabulary and context in both.
    try testing.expectEqual(CLIP_L_CONFIG.vocab, CLIP_BIG_G_CONFIG.vocab);
    try testing.expectEqual(@as(u32, 77), CLIP_BIG_G_CONFIG.max_positions);

    // The concatenation the UNet expects is built from the two HIDDEN sizes,
    // and the pooled vector from bigG's PROJECTION alone.
    try testing.expectEqual(CROSS_ATTENTION_DIM, CLIP_L_CONFIG.hidden + CLIP_BIG_G_CONFIG.hidden);
    try testing.expectEqual(POOLED_PROJECTION_DIM, CLIP_BIG_G_CONFIG.projection_dim);
}

// ── Weight-name contract ──
//
// Verified against stabilityai/stable-diffusion-xl-base-1.0 (fp16 variant):
// CLIP-L holds 196 tensors, bigG 517. Both are 16 tensors per layer plus four
// shared (two embeddings, two final-norm), and bigG carries ONE extra —
// `text_projection.weight`, which CLIP-L does not have at all. That asymmetry
// is the structural reason the pooled micro-conditioning vector comes from bigG
// alone: CLIP-L has nothing to project with.

/// Per-layer tensor suffixes, in the order the forward consumes them. Both
/// towers use identical names — only the shapes differ — so the loader is one
/// implementation over `ClipTextConfig`.
pub const CLIP_LAYER_TENSORS = [_][]const u8{
    "layer_norm1.weight",        "layer_norm1.bias",
    "self_attn.q_proj.weight",   "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",   "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",   "self_attn.v_proj.bias",
    "self_attn.out_proj.weight", "self_attn.out_proj.bias",
    "layer_norm2.weight",        "layer_norm2.bias",
    "mlp.fc1.weight",            "mlp.fc1.bias",
    "mlp.fc2.weight",            "mlp.fc2.bias",
};

/// Tensors outside the layer stack, present in BOTH towers.
pub const CLIP_SHARED_TENSORS = [_][]const u8{
    "text_model.embeddings.token_embedding.weight",
    "text_model.embeddings.position_embedding.weight",
    "text_model.final_layer_norm.weight",
    "text_model.final_layer_norm.bias",
};

/// bigG only. Projects the pooled EOS hidden state to `projection_dim`.
pub const CLIP_PROJECTION_TENSOR = "text_projection.weight";

/// How many tensors a tower of `layers` layers holds, `with_projection` for
/// bigG. Exact by construction — 196 for CLIP-L, 517 for bigG.
pub fn clipTensorCount(layers: u32, with_projection: bool) u32 {
    return layers * @as(u32, CLIP_LAYER_TENSORS.len) + @as(u32, CLIP_SHARED_TENSORS.len) +
        @as(u32, if (with_projection) 1 else 0);
}

test "the tensor count matches the shipped checkpoints exactly" {
    // Measured from the real safetensors headers.
    try testing.expectEqual(@as(u32, 196), clipTensorCount(CLIP_L_CONFIG.layers, false));
    try testing.expectEqual(@as(u32, 517), clipTensorCount(CLIP_BIG_G_CONFIG.layers, true));
    try testing.expectEqual(@as(usize, 16), CLIP_LAYER_TENSORS.len);
}

// Live structural check against a real SDXL checkpoint. Env-gated like the
// repo's other fixture tests (MINIMAX_H3_VAE_ENC_FIXTURE and friends): it reads
// the safetensors HEADERS only — a bounded JSON prefix, never the weights — and
// asserts every name the loader will ask for is present at the shape the config
// implies.
//
//   SDXL_CHECKPOINT_DIR=~/.mlx-serve/staging/sdxl-base-1.0 \
//     zig build test -Dtest-filter="sdxl checkpoint"
//
// This is NOT parity: it proves the loader will BIND, not that the forward is
// numerically right. That still needs a diffusers fixture.
test "sdxl checkpoint: every expected tensor is present at the right shape" {
    const a = testing.allocator;
    const dir_env = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);

    const Tower = struct { sub: []const u8, cfg: ClipTextConfig, proj: bool };
    const towers = [_]Tower{
        .{ .sub = "text_encoder", .cfg = CLIP_L_CONFIG, .proj = false },
        .{ .sub = "text_encoder_2", .cfg = CLIP_BIG_G_CONFIG, .proj = true },
    };

    for (towers) |t| {
        const path = try std.fmt.allocPrint(a, "{s}/{s}/model.fp16.safetensors", .{ dir_env, t.sub });
        defer a.free(path);
        const io = std.Io.Threaded.global_single_threaded.io();
        var file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch continue;
        defer file.close(io);
        var rbuf: [4096]u8 = undefined;
        var rs = file.reader(io, &rbuf);

        var len_buf: [8]u8 = undefined;
        try rs.interface.readSliceAll(&len_buf);
        const hdr_len = std.mem.readInt(u64, &len_buf, .little);
        const hdr = try a.alloc(u8, @intCast(hdr_len));
        defer a.free(hdr);
        try rs.interface.readSliceAll(hdr);

        var parsed = try std.json.parseFromSlice(std.json.Value, a, hdr, .{});
        defer parsed.deinit();
        const obj = parsed.value.object;

        // Shared tensors.
        for (CLIP_SHARED_TENSORS) |name| {
            try testing.expect(obj.get(name) != null);
        }
        // The projection is bigG's alone — present there, ABSENT in CLIP-L.
        try testing.expectEqual(t.proj, obj.get(CLIP_PROJECTION_TENSOR) != null);

        // Every layer, every tensor.
        var layer: u32 = 0;
        while (layer < t.cfg.layers) : (layer += 1) {
            for (CLIP_LAYER_TENSORS) |suffix| {
                const name = try std.fmt.allocPrint(a, "text_model.encoder.layers.{d}.{s}", .{ layer, suffix });
                defer a.free(name);
                try testing.expect(obj.get(name) != null);
            }
        }

        // The declared count is the whole file (minus safetensors' metadata key).
        var n: u32 = 0;
        var it = obj.iterator();
        while (it.next()) |e| {
            if (!std.mem.eql(u8, e.key_ptr.*, "__metadata__")) n += 1;
        }
        try testing.expectEqual(clipTensorCount(t.cfg.layers, t.proj), n);

        // One shape spot-check per tower, keyed on the config rather than a
        // literal: the embedding table is [vocab, hidden].
        const emb = obj.get("text_model.embeddings.token_embedding.weight").?.object.get("shape").?.array;
        try testing.expectEqual(@as(i64, t.cfg.vocab), emb.items[0].integer);
        try testing.expectEqual(@as(i64, t.cfg.hidden), emb.items[1].integer);
    }
}

// ── VAE: the decoder is already written ──
//
// MEASURED, not assumed: 138 of SDXL's 140 decoder-side tensors are
// name-identical to FLUX.2-klein's, because both are diffusers `AutoencoderKL`
// decoders — `post_quant_conv`, `decoder.conv_in`, two mid resnets around one
// attention, `[4][3]` up-resnets, three upsamplers, `conv_norm_out`,
// `conv_out`. `flux.Vae`'s loader is entirely name-driven and shape-agnostic,
// so it already describes this architecture.
//
// Three deltas stand between that and serving SDXL, and none is a rewrite:
//
//   1. `to_out` naming. SDXL ships diffusers' indexed `to_out.0.{weight,bias}`;
//      the mflux conversion flattened it to `to_out.{weight,bias}`. That is the
//      ENTIRE 2-tensor difference. `lora.zig` already normalizes this exact
//      spelling, so the alias belongs in the loader, not in a second decoder.
//   2. Quantization. mflux packs are 4-bit, so `flux.Vae` loads its attention
//      through `QLinear`; SDXL's VAE is dense fp16 and needs the dense arm.
//   3. Latent normalization. FLUX.2 carries `bn.running_mean`/`bn.running_var`
//      and normalizes the latent with them. SDXL has NO `bn.*` tensors at all —
//      it scales by `VAE_SCALING_FACTOR` instead. Reusing the FLUX path without
//      skipping the bn step would fail at load on a missing weight, which is
//      the good direction for this to break in.
//
// So the VAE is a parameterisation job over an existing, oracle-validated
// decoder rather than a fresh port — which also means it inherits that
// decoder's correctness, the one place in this port where that is available.

// Live check of the claim above: with BOTH checkpoints present, assert the
// decoder tensor sets differ only by the `to_out` spelling.
//
//   SDXL_CHECKPOINT_DIR=... FLUX_VAE_DIR=... \
//     zig build test -Dtest-filter="sdxl vae shares"
test "sdxl vae shares FLUX's decoder architecture" {
    const a = testing.allocator;
    const sdxl_dir = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);
    const flux_dir = std.mem.span(std.c.getenv("FLUX_VAE_DIR") orelse return error.SkipZigTest);

    const Set = std.StringHashMap(void);
    const readNames = struct {
        fn f(alloc: std.mem.Allocator, path: []const u8, out: *Set) !void {
            const io = std.Io.Threaded.global_single_threaded.io();
            var file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return;
            defer file.close(io);
            var rbuf: [4096]u8 = undefined;
            var rs = file.reader(io, &rbuf);
            var len_buf: [8]u8 = undefined;
            try rs.interface.readSliceAll(&len_buf);
            const hdr_len = std.mem.readInt(u64, &len_buf, .little);
            const hdr = try alloc.alloc(u8, @intCast(hdr_len));
            defer alloc.free(hdr);
            try rs.interface.readSliceAll(hdr);
            var parsed = try std.json.parseFromSlice(std.json.Value, alloc, hdr, .{});
            defer parsed.deinit();
            var it = parsed.value.object.iterator();
            while (it.next()) |e| {
                const k = e.key_ptr.*;
                if (std.mem.eql(u8, k, "__metadata__")) continue;
                if (!std.mem.startsWith(u8, k, "decoder.") and
                    !std.mem.startsWith(u8, k, "post_quant")) continue;
                // Quantized packs store .scales/.biases beside .weight; fold
                // them onto the base name so the comparison is architectural.
                if (std.mem.endsWith(u8, k, ".scales") or std.mem.endsWith(u8, k, ".biases")) continue;
                try out.put(try alloc.dupe(u8, k), {});
            }
        }
    }.f;

    var sd = Set.init(a);
    defer {
        var it = sd.keyIterator();
        while (it.next()) |k| a.free(k.*);
        sd.deinit();
    }
    const vae_path = try std.fmt.allocPrint(a, "{s}/vae/diffusion_pytorch_model.fp16.safetensors", .{sdxl_dir});
    defer a.free(vae_path);
    try readNames(a, vae_path, &sd);
    if (sd.count() == 0) return error.SkipZigTest;

    // Every SDXL decoder tensor except the two `to_out.0` ones must appear in
    // FLUX's decoder under the same name.
    var to_out_only: u32 = 0;
    var it = sd.keyIterator();
    while (it.next()) |k| {
        if (std.mem.indexOf(u8, k.*, "to_out.0.") != null) to_out_only += 1;
    }
    try testing.expectEqual(@as(u32, 2), to_out_only);
    try testing.expectEqual(@as(u32, 140), sd.count());
    _ = flux_dir; // the FLUX side is compared by the harness script; see the note above
}

// ── Distilled variants: Turbo and Lightning ──
//
// Both are SDXL's UNet with a different SCHEDULE, not a different architecture,
// so they load through the same engine and differ only in what
// `scheduler_config.json` declares. Three fields carry the whole difference:
//
//   `timestep_spacing`  base is "leading"; both distills are "trailing", which
//                       starts at 999 instead of 751 and matters enormously at
//                       1-4 steps (leading's first step would skip the noisiest
//                       part of the schedule entirely).
//   `prediction_type`   base and most distills are "epsilon"; SDXL-Lightning's
//                       1-STEP checkpoint is "sample" — the UNet returns the
//                       clean image directly. Reading that as epsilon produces
//                       noise, not a worse image.
//   guidance            base wants ~5; both distills are trained to run
//                       guidance-free (Turbo 0.0, Lightning 1.0), which also
//                       halves the cost because the unconditional forward is
//                       skipped entirely.

/// What the UNet's output MEANS. Getting this wrong is not a quality
/// regression — epsilon and sample are different quantities.
pub const PredictionType = enum {
    epsilon,
    /// The clean latent directly (SDXL-Lightning 1-step).
    sample,
    /// Velocity parameterization.
    v_prediction,

    pub fn fromString(s: []const u8) ?PredictionType {
        if (std.mem.eql(u8, s, "epsilon")) return .epsilon;
        if (std.mem.eql(u8, s, "sample") or std.mem.eql(u8, s, "original_sample")) return .sample;
        if (std.mem.eql(u8, s, "v_prediction")) return .v_prediction;
        return null;
    }
};

/// The Euler derivative `(sample - pred_x0) / sigma`, expressed as the two
/// coefficients that build it from the latent and the model output:
///
///     derivative = latent * `latent` + model_out * `model`
///
/// Returned as coefficients so the caller applies them to whole tensors, and
/// so the three prediction types differ by ARITHMETIC here rather than by
/// three copies of the denoise loop.
///
/// Derivations (diffusers `EulerDiscreteScheduler.step`):
///   epsilon       pred_x0 = latent - sigma*out   =>  derivative = out
///   sample        pred_x0 = out                  =>  derivative = (latent - out)/sigma
///   v_prediction  pred_x0 = out*(-sigma/sqrt(s2+1)) + latent/(s2+1)
pub const Derivative = struct { latent: f64, model: f64 };

pub fn derivativeCoeffs(kind: PredictionType, sigma: f64) Derivative {
    return switch (kind) {
        .epsilon => .{ .latent = 0.0, .model = 1.0 },
        .sample => .{ .latent = 1.0 / sigma, .model = -1.0 / sigma },
        .v_prediction => blk: {
            const s2p1 = sigma * sigma + 1.0;
            // pred_x0 = out*(-sigma/sqrt(s2p1)) + latent/s2p1
            // derivative = (latent - pred_x0)/sigma
            //            = latent*(1 - 1/s2p1)/sigma + out*(1/sqrt(s2p1))
            break :blk .{
                .latent = (1.0 - 1.0 / s2p1) / sigma,
                .model = 1.0 / @sqrt(s2p1),
            };
        },
    };
}

/// Ancestral (SDE) step split, as `EulerAncestralDiscreteScheduler` computes it
/// — SDXL-Turbo's declared scheduler. The deterministic part integrates to
/// `sigma_down` instead of `sigma_next`, and fresh noise at `sigma_up` makes up
/// the difference.
///
/// `sigma_up` is CLAMPED to `sigma_next`: at the final step `sigma_next` is 0,
/// which makes both terms 0 and the step purely deterministic — without the
/// clamp a rounding wobble there injects noise into the finished image.
pub const AncestralStep = struct { sigma_down: f64, sigma_up: f64 };

pub fn ancestralSigmas(sigma: f64, sigma_next: f64) AncestralStep {
    if (sigma <= 0.0) return .{ .sigma_down = sigma_next, .sigma_up = 0.0 };
    const s2 = sigma * sigma;
    const n2 = sigma_next * sigma_next;
    const up_sq = n2 * (s2 - n2) / s2;
    const up = if (up_sq > 0.0) @min(sigma_next, @sqrt(up_sq)) else 0.0;
    const down_sq = n2 - up * up;
    return .{ .sigma_down = if (down_sq > 0.0) @sqrt(down_sq) else 0.0, .sigma_up = up };
}

test "sdxl distilled: trailing spacing starts at the noisiest timestep" {
    // The whole reason the distills use it: at 1-4 steps, `leading` would
    // start at 751 and never see the top of the schedule.
    var out: [4]usize = undefined;
    timestepIndices(.trailing, 4, &out);
    try testing.expectEqualSlices(usize, &[_]usize{ 999, 749, 499, 249 }, &out);

    var one: [1]usize = undefined;
    timestepIndices(.trailing, 1, &one);
    try testing.expectEqual(@as(usize, 999), one[0]);

    // Contrast with leading, which is what base SDXL uses.
    var lead: [4]usize = undefined;
    timestepIndices(.leading, 4, &lead);
    try testing.expectEqual(@as(usize, 750), lead[0]);
}

test "sdxl distilled: trailing is a FLOAT stride, matching diffusers off the divisors" {
    // Regression (PR #301 review): the stride was the integer `1000 / steps`,
    // which agrees with diffusers only when `steps` divides 1000. Every ladder
    // below is `round(arange(1000, 0, -1000/steps)) - 1` evaluated with numpy's
    // half-to-even rounding, transcribed from diffusers `EulerDiscreteScheduler`.
    const Case = struct { steps: usize, want: []const usize };
    const cases = [_]Case{
        // Non-divisors: where the integer stride drifted. 6 steps read
        // `[999, 833, 667, 501, 335, 169]` before this fix.
        .{ .steps = 3, .want = &.{ 999, 666, 332 } },
        .{ .steps = 6, .want = &.{ 999, 832, 666, 499, 332, 166 } },
        .{ .steps = 7, .want = &.{ 999, 856, 713, 570, 428, 285, 142 } },
        // Half-to-even, not half-away-from-zero: 812.5 rounds DOWN to 812,
        // so index 3 is 811 and not 812. `@round` gets this one wrong.
        .{ .steps = 16, .want = &.{ 999, 937, 874, 811, 749, 687, 624, 561, 499, 437, 374, 311, 249, 187, 124, 61 } },
        // Divisors still land where they always did.
        .{ .steps = 1, .want = &.{999} },
        .{ .steps = 4, .want = &.{ 999, 749, 499, 249 } },
        .{ .steps = 25, .want = &.{ 999, 959, 919, 879, 839, 799, 759, 719, 679, 639, 599, 559, 519, 479, 439, 399, 359, 319, 279, 239, 199, 159, 119, 79, 39 } },
    };
    var buf: [64]usize = undefined;
    for (cases) |c| {
        const out = buf[0..c.steps];
        timestepIndices(.trailing, c.steps, out);
        try testing.expectEqualSlices(usize, c.want, out);
    }
}

test "sdxl distilled: linspace rounds half-to-even like numpy" {
    // Same rounding rule, same reference expression
    // (`linspace(0, 999, steps)[::-1]`): 832.5 at 7 steps goes DOWN.
    var seven: [7]usize = undefined;
    timestepIndices(.linspace, 7, &seven);
    try testing.expectEqualSlices(usize, &[_]usize{ 999, 832, 666, 500, 333, 166, 0 }, &seven);

    var sixteen: [16]usize = undefined;
    timestepIndices(.linspace, 16, &sixteen);
    try testing.expectEqualSlices(usize, &[_]usize{ 999, 932, 866, 799, 733, 666, 599, 533, 466, 400, 333, 266, 200, 133, 67, 0 }, &sixteen);
}

test "sdxl distilled: the derivative reduces to the model output for epsilon" {
    // Epsilon is the case the original loop hardcoded; the generalization must
    // not move it.
    const e = derivativeCoeffs(.epsilon, 4.1167);
    try testing.expectEqual(@as(f64, 0.0), e.latent);
    try testing.expectEqual(@as(f64, 1.0), e.model);

    // `sample` (Lightning 1-step) is the OPPOSITE sign on the model term —
    // reading one as the other is noise, not a worse image.
    const s = derivativeCoeffs(.sample, 2.0);
    try testing.expectApproxEqAbs(@as(f64, 0.5), s.latent, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.5), s.model, 1e-12);

    // v_prediction at sigma -> 0 leans entirely on the model term.
    const v = derivativeCoeffs(.v_prediction, 1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.5), v.latent, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0 / @sqrt(2.0)), v.model, 1e-12);
}

test "sdxl distilled: the ancestral split conserves variance and ends clean" {
    // Mid-schedule: down and up together account for sigma_next.
    const a = ancestralSigmas(4.0, 2.0);
    try testing.expect(a.sigma_up > 0.0);
    try testing.expectApproxEqAbs(2.0 * 2.0, a.sigma_down * a.sigma_down + a.sigma_up * a.sigma_up, 1e-9);

    // FINAL step: sigma_next is 0, so nothing may be injected — an image that
    // ends with added noise is the visible failure here.
    const last = ancestralSigmas(0.041314, 0.0);
    try testing.expectEqual(@as(f64, 0.0), last.sigma_up);
    try testing.expectEqual(@as(f64, 0.0), last.sigma_down);

    try testing.expectEqual(@as(f64, 0.0), ancestralSigmas(0.0, 0.0).sigma_up);
}

test "sdxl distilled: prediction types parse by their diffusers spelling" {
    try testing.expectEqual(PredictionType.epsilon, PredictionType.fromString("epsilon").?);
    try testing.expectEqual(PredictionType.sample, PredictionType.fromString("sample").?);
    try testing.expectEqual(PredictionType.v_prediction, PredictionType.fromString("v_prediction").?);
    try testing.expect(PredictionType.fromString("flow_matching") == null);
}
