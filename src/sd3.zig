//! Stable Diffusion 3.5 — flow-matching schedule + MMDiT geometry.
//!
//! FIRST SLICE of the SD 3.5 port, deliberately weight-free and MLX-free: every
//! function here is pure arithmetic over f64 slices, so it is testable with no
//! checkpoint, no GPU and no oracle. The tensor work (the MMDiT itself, the T5
//! encoder, the 16-channel VAE) lands in `sd3_mmdit.zig`, `t5_encoder.zig` and
//! `sd3_pipeline.zig`.
//!
//! SD 3.5 shares almost nothing numerically with the SDXL family this repo
//! already serves, and the three differences are the whole file:
//!
//!   1. FLOW MATCHING, not epsilon on a discrete beta schedule. There is no
//!      `alphas_cumprod`, no `scale_model_input`, and no terminal-SNR question:
//!      sigma IS the time coordinate, running 1 -> 0, and the model predicts the
//!      velocity `(noise - sample)`. One Euler step is
//!      `x <- x + (sigma_next - sigma) * v`, and that is the entire integrator.
//!   2. The ladder is RESOLUTION-INDEPENDENT but SHIFTED. `shift: 3.0` bends the
//!      uniform sigma ladder toward high noise, which is where a flow model does
//!      its composition. Reading it as 1.0 does not error — it produces images
//!      that are structurally mushy and colour-flat, the same failure class as a
//!      v-prediction checkpoint sampled as epsilon.
//!   3. THREE text encoders, two of them CLIP. The CLIP pair is geometrically
//!      identical to SDXL's (768/12/12 quick_gelu and 1280/32/20 gelu), which is
//!      why `sdxl_clip.zig` binds them unchanged; the third is a T5-XXL encoder
//!      at width 4096, which nothing in this repo had.
//!
//! ORACLE STATUS — kept at the three levels `sdxl.zig` established:
//!
//!   VERIFIED against the checkpoint. Every constant here was read out of
//!   `stabilityai/stable-diffusion-3.5-large`'s own configs rather than a doc:
//!   `transformer/config.json` (38 layers, 38 heads x 64, patch 2, in/out 16,
//!   `pos_embed_max_size` 192, `joint_attention_dim` 4096,
//!   `caption_projection_dim` 2432, `pooled_projection_dim` 2048, `qk_norm`
//!   "rms_norm"), `vae/config.json` (16 latent channels, scaling 1.5305, shift
//!   0.0609, and NO quant/post-quant conv, which is where it parts company with
//!   SDXL's VAE), and `scheduler/scheduler_config.json` (shift 3.0, 1000 train
//!   steps). Large and Large-Turbo declare byte-identical configs — the Turbo
//!   difference is weights plus its recommended 4 steps at guidance 1.
//!
//!   VERIFIED by construction. The schedule formulas below are pinned by
//!   endpoints, monotonicity and hand-computable values.
//!
//!   NOT VERIFIED here. Whether the assembled forward produces diffusers'
//!   NUMBERS — that is `tests/dump_sd3_fixtures.py`'s job, and the claim belongs
//!   in the file that runs against the fixture, not in this one.

const std = @import("std");
const testing = std.testing;

// ── Schedule (FlowMatchEulerDiscreteScheduler) ──────────────────────────

/// `num_train_timesteps`. Only ever used to SCALE sigmas into the timestep
/// values the transformer is handed: a flow model's "timestep" is
/// `sigma * 1000`, not an index into a 1000-entry table the way SDXL's is.
pub const NUM_TRAIN_TIMESTEPS: f64 = 1000.0;

/// `shift` from `scheduler/scheduler_config.json`. 3.0 on both SD 3.5 Large and
/// Large-Turbo. SD 3 Medium shipped 3.0 as well; the field is read rather than
/// assumed because it is exactly the `LtxVersion` class of trap — a config
/// field that silently decides numerics.
pub const DEFAULT_SHIFT: f64 = 3.0;

/// What `scheduler/scheduler_config.json` declares.
///
/// Deliberately NOT merged with `sdxl_pipeline.SchedulerConfig`: that struct's
/// every field (`timestep_spacing`, `steps_offset`, `prediction_type`,
/// `rescale_betas_zero_snr`) is meaningless under flow matching, and a shared
/// struct where each half ignores the other half's fields is how a default
/// leaks across families.
pub const SchedulerConfig = struct {
    shift: f64 = DEFAULT_SHIFT,
    /// Guidance the checkpoint expects when the request says nothing. Large
    /// wants ~4.5 (stability's own card and diffusers' example); Turbo is
    /// trained guidance-free at 1.0, where the pipeline skips the unconditional
    /// forward entirely — half the work per step on top of 4 steps instead of
    /// 28. Not a config field: nothing in the repo declares it, so it is keyed
    /// on the checkpoint (see `sd3_pipeline`).
    default_guidance: f32 = 4.5,
    /// Steps the checkpoint is meant for when the request says nothing.
    default_steps: u32 = 28,
};

/// The Turbo distill's defaults. Same transformer, same VAE, same towers, same
/// declared scheduler — a different number of steps and no guidance.
pub const TURBO_CONFIG = SchedulerConfig{ .shift = DEFAULT_SHIFT, .default_guidance = 1.0, .default_steps = 4 };

/// The inference sigma ladder for `steps` steps at `shift`.
///
/// diffusers `FlowMatchEulerDiscreteScheduler.set_timesteps` with no dynamic
/// shifting: sigmas start as `linspace(1, 1/steps, steps)` — note the LOW end is
/// `1/steps` and not 0 — then each is bent by
///
///     sigma <- shift * sigma / (1 + (shift - 1) * sigma)
///
/// and a terminal 0.0 is appended. `out.len` must be `steps + 1`; that trailing
/// zero is what makes the final Euler step land on a clean latent, and dropping
/// it leaves the image one step short of denoised (the same trap as
/// `sdxl.inferenceSigmas`).
///
/// The shift is a MOBIUS map on [0,1] fixing both endpoints, so the ladder stays
/// in [0,1] and stays strictly decreasing for any `shift > 0` — which is what
/// makes the invariants below total rather than a spot check.
pub fn inferenceSigmas(steps: usize, shift: f64, out: []f64) void {
    std.debug.assert(out.len == steps + 1);
    if (steps == 0) return;
    const lo = 1.0 / @as(f64, @floatFromInt(steps));
    for (0..steps) |i| {
        const t = if (steps == 1) 0.0 else @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(steps - 1));
        const uniform = 1.0 + (lo - 1.0) * t;
        out[i] = shiftSigma(uniform, shift);
    }
    out[steps] = 0.0;
}

/// The shift map itself. `shift == 1` is the identity, which is why a checkpoint
/// that declares nothing still samples correctly on the unshifted ladder.
pub fn shiftSigma(sigma: f64, shift: f64) f64 {
    return shift * sigma / (1.0 + (shift - 1.0) * sigma);
}

/// The timestep value the transformer is handed for a given sigma. A flow
/// model's time input is the sigma itself, scaled to the training range — NOT an
/// index. Handing it the index would put the conditioning off by a factor of
/// 1000 and produce noise.
pub fn timestepForSigma(sigma: f64) f32 {
    return @floatCast(sigma * NUM_TRAIN_TIMESTEPS);
}

/// One Euler step: `x_next = x + (sigma_next - sigma) * v`.
///
/// Returned as a coefficient rather than applied, so the caller does the tensor
/// work and this stays testable without MLX. The model's output IS the velocity
/// — there is no epsilon/v-prediction/sample branch here, because flow matching
/// has exactly one parameterisation. `derivativeCoeffs` in `sdxl.zig` exists
/// precisely because the discrete-beta family does not.
pub fn eulerStepCoeff(sigma: f64, sigma_next: f64) f64 {
    return sigma_next - sigma;
}

// ── MMDiT geometry (transformer/config.json) ────────────────────────────

/// SD 3.5 Large's `transformer/config.json`, verbatim — the DEFAULTS only.
/// Every field is PARSED off the checkpoint (`parseMmditConfig`), because
/// Medium is a genuinely different transformer under the same `_class_name`:
/// 24 layers at inner 1536, `pos_embed_max_size` 384, and a
/// `dual_attention_layers` list that turns the first 13 blocks into MMDiT-**X**.
/// Branching on a model NAME or a parameter count instead would be the exact
/// trap this repo names repeatedly — a config field silently deciding numerics.
///
/// `attention_head_dim` here is a real head DIM (64), unlike SDXL's UNet config
/// where diffusers' identically-named field is the head COUNT — a trap this repo
/// has already been bitten by, so the two are never read by a shared helper.
pub const MmditConfig = struct {
    num_layers: u32 = 38,
    num_attention_heads: u32 = 38,
    attention_head_dim: u32 = 64,
    in_channels: u32 = 16,
    out_channels: u32 = 16,
    patch_size: u32 = 2,
    /// The learned positional embedding is stored for a `pos_embed_max_size`
    /// square of PATCHES (192 -> up to 3072px) and CROPPED CENTRALLY to the
    /// request's patch grid. Not interpolated: the crop is what makes a
    /// non-1024 canvas keep the same local scale the model trained at.
    pos_embed_max_size: u32 = 192,
    /// T5's width — the sequence the caption stream arrives at, before
    /// `context_embedder` projects it to `innerDim`.
    joint_attention_dim: u32 = 4096,
    /// CLIP-L's 768 + CLIP-G's 1280, the pooled vector added to the timestep
    /// embedding.
    pooled_projection_dim: u32 = 2048,
    /// `sample_size` * `patch_size` = the pixel canvas the model trained on
    /// (128 * 8 latent stride = 1024px).
    sample_size: u32 = 128,
    /// `qk_norm: "rms_norm"` — both streams' queries and keys are RMS-normed
    /// per head before attention. Absent on SD 3 Medium; present on every 3.5.
    qk_norm: bool = true,
    /// `dual_attention_layers`: the block indices carrying a SECOND, image-only
    /// self-attention (MMDiT-X). ABSENT on Large/Large-Turbo, `[0..12]` on
    /// 3.5 Medium. Owned by the config struct, freed by `freeMmditConfig`.
    ///
    /// Empty and absent are the same thing here — unlike `negative_prompt` on
    /// the SDXL surface, there is no "declared empty" meaning to preserve.
    dual_attention_layers: []const u32 = &.{},

    /// `num_attention_heads * attention_head_dim`. Also `caption_projection_dim`
    /// and the `pooled_projection_dim` MLP's output width: everything inside the
    /// joint blocks runs at this one width.
    pub fn innerDim(self: MmditConfig) u32 {
        return self.num_attention_heads * self.attention_head_dim;
    }

    pub fn isDualAttention(self: MmditConfig, layer: u32) bool {
        for (self.dual_attention_layers) |l| {
            if (l == layer) return true;
        }
        return false;
    }
};

/// SD 3.5 Medium's own numbers, for tests and for the residency estimator. The
/// LOADER never reads this — it parses the checkpoint. It exists so a reader can
/// see what the second shape is without fetching a gated repo.
pub const MEDIUM_CONFIG = MmditConfig{
    .num_layers = 24,
    .num_attention_heads = 24,
    .pos_embed_max_size = 384,
    .dual_attention_layers = &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
};

/// Parse `transformer/config.json`.
///
/// Every field is optional against the Large defaults: a checkpoint that omits
/// one is declaring Large's value, which is how `_class_name`-identical repos
/// stay one code path. `dual_attention_layers` is allocated and must be released
/// with `freeMmditConfig`.
pub fn parseMmditConfig(allocator: std.mem.Allocator, json_bytes: []const u8) !MmditConfig {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadTransformerConfig;
    const root = parsed.value.object;

    var cfg = MmditConfig{};
    const u32Field = struct {
        fn read(obj: std.json.ObjectMap, name: []const u8, dst: *u32) void {
            // A JSON `null` is how diffusers spells "not set" for an optional
            // int, and reading `.integer` off one panics on an unchecked
            // `std.json.Value` — the same null-guard rule `sd1_pipeline` needed.
            const v = obj.get(name) orelse return;
            if (v != .integer) return;
            dst.* = @intCast(v.integer);
        }
    };
    u32Field.read(root, "num_layers", &cfg.num_layers);
    u32Field.read(root, "num_attention_heads", &cfg.num_attention_heads);
    u32Field.read(root, "attention_head_dim", &cfg.attention_head_dim);
    u32Field.read(root, "in_channels", &cfg.in_channels);
    u32Field.read(root, "out_channels", &cfg.out_channels);
    u32Field.read(root, "patch_size", &cfg.patch_size);
    u32Field.read(root, "pos_embed_max_size", &cfg.pos_embed_max_size);
    u32Field.read(root, "joint_attention_dim", &cfg.joint_attention_dim);
    u32Field.read(root, "pooled_projection_dim", &cfg.pooled_projection_dim);
    u32Field.read(root, "sample_size", &cfg.sample_size);

    // `qk_norm` is a STRING naming the norm, not a bool: "rms_norm" on every
    // 3.5, absent on SD 3 Medium. Anything else is a norm we do not implement,
    // and quietly running unnormed would be a plausible-image failure.
    if (root.get("qk_norm")) |v| {
        cfg.qk_norm = switch (v) {
            .string => |name| std.mem.eql(u8, name, "rms_norm"),
            .null => false,
            else => return error.UnsupportedQkNorm,
        };
        if (v == .string and !cfg.qk_norm) return error.UnsupportedQkNorm;
    } else {
        cfg.qk_norm = false;
    }

    if (root.get("dual_attention_layers")) |v| {
        if (v == .array) {
            const out = try allocator.alloc(u32, v.array.items.len);
            errdefer allocator.free(out);
            for (v.array.items, out) |item, *o| {
                if (item != .integer) return error.BadDualAttentionLayers;
                o.* = @intCast(item.integer);
            }
            cfg.dual_attention_layers = out;
        }
    }
    return cfg;
}

pub fn freeMmditConfig(allocator: std.mem.Allocator, cfg: MmditConfig) void {
    if (cfg.dual_attention_layers.len > 0) allocator.free(cfg.dual_attention_layers);
}

/// The number of PATCH tokens a latent of `h` x `w` becomes.
pub fn patchGrid(cfg: MmditConfig, latent_h: u32, latent_w: u32) struct { h: u32, w: u32 } {
    return .{ .h = latent_h / cfg.patch_size, .w = latent_w / cfg.patch_size };
}

/// Where the central crop of the stored `pos_embed_max_size` grid starts, for a
/// patch grid of `n` on that axis.
///
/// diffusers: `top = (max - h) // 2`. Off-centre by one and every generation is
/// subtly mis-framed with nothing to error on — the same silent class as
/// SDXL's `add_time_ids`.
pub fn posEmbedCropStart(cfg: MmditConfig, n: u32) u32 {
    if (n >= cfg.pos_embed_max_size) return 0;
    return (cfg.pos_embed_max_size - n) / 2;
}

// ── VAE (vae/config.json) ───────────────────────────────────────────────

/// SD 3.5's VAE is an `AutoencoderKL` of the same SHAPE as SDXL's
/// (`block_out_channels` [128, 256, 512, 512], `layers_per_block` 2,
/// `norm_num_groups` 32, silu, mid-block attention on), with three differences
/// that all matter:
///
///   * 16 latent channels, not 4.
///   * `use_quant_conv` and `use_post_quant_conv` are BOTH false. SDXL's VAE
///     runs a 1x1 conv either side of the latent; SD 3.5's does not, and binding
///     one that is not in the checkpoint is a missing-weight error, while
///     SKIPPING one that is there silently changes the latent basis.
///   * The latent is SHIFTED as well as scaled: `(z - shift) * scaling` on
///     encode and `z / scaling + shift` on decode. SDXL's shift is implicitly 0,
///     so a scale-only decode of an SD 3.5 latent is a uniform colour cast.
pub const VaeConfig = struct {
    latent_channels: u32 = 16,
    scaling_factor: f64 = 1.5305,
    shift_factor: f64 = 0.0609,
    /// `force_upcast: true`, exactly as on SDXL: this VAE overflows fp16 on real
    /// latents, so it decodes at f32 whatever the DiT runs at.
    force_upcast: bool = true,
};

/// Latent -> VAE input. The inverse of `encodeScale`.
pub fn decodeScale(cfg: VaeConfig, z: f64) f64 {
    return z / cfg.scaling_factor + cfg.shift_factor;
}

/// VAE output -> latent.
pub fn encodeScale(cfg: VaeConfig, z: f64) f64 {
    return (z - cfg.shift_factor) * cfg.scaling_factor;
}

/// The VAE downsamples by 8 in each axis, so a pixel canvas must be a multiple
/// of `patch_size * 8` = 16 for the patch grid to be exact. Rounds DOWN to keep
/// a requested size from silently growing past a memory budget the caller sized.
pub const PIXELS_PER_LATENT: u32 = 8;

pub fn latentSideFor(pixels: u32) u32 {
    return pixels / PIXELS_PER_LATENT;
}

// ── Discovery (model_index.json) ────────────────────────────────────────

/// diffusers `_class_name` values describing a checkpoint this engine loads.
/// The plain and img2img pipelines share one transformer, one VAE and the same
/// three text encoders — they differ only in how the initial latent is prepared,
/// which is a request-shape question, not a checkpoint one.
pub fn isSd3PipelineClass(class_name: []const u8) bool {
    const known = [_][]const u8{
        "StableDiffusion3Pipeline",
        "StableDiffusion3Img2ImgPipeline",
        "StableDiffusion3InpaintPipeline",
    };
    for (known) |k| if (std.mem.eql(u8, class_name, k)) return true;
    return false;
}

/// True when `model_index.json` bytes describe an SD 3.x pipeline.
///
/// Keyed on the DECLARED class plus the THIRD tower, mirroring how
/// `sdxl.indexDeclaresSdxl` keys on the second. `text_encoder_3` is what makes
/// this repo shape unmistakable: SD 3.5 is the only diffusers family here that
/// carries three text encoders, and the T5 tower is exactly the component an
/// SDXL or SD 1.x engine would have no idea what to do with.
///
/// ONE predicate, called by BOTH `gen.peekModelType` and
/// `model_discovery.peekSd3Index` — the documented-duplication rule the SDXL
/// pair already follows. A checkpoint discovery can see but gen cannot classify
/// is a model the server advertises and then cannot load.
pub fn indexDeclaresSd3(allocator: std.mem.Allocator, index_json: []const u8) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, index_json, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const obj = parsed.value.object;
    const cn = obj.get("_class_name") orelse return false;
    if (cn != .string or !isSd3PipelineClass(cn.string)) return false;
    return obj.get("text_encoder_3") != null and obj.get("transformer") != null;
}

// ── Tests ───────────────────────────────────────────────────────────────

test "sd3 schedule: the shifted ladder runs 1 -> 0, strictly down, with the terminal zero" {
    // The Mobius shift fixes both endpoints, so these hold for any shift: the
    // first sigma is the shifted 1.0 (= 1.0), the last SIGMA is the shifted
    // 1/steps, and out[steps] is the appended 0.0 that ends the integration.
    for ([_]f64{ 1.0, 3.0, 6.0 }) |shift| {
        for ([_]usize{ 1, 4, 28, 50 }) |steps| {
            var buf: [64]f64 = undefined;
            const out = buf[0 .. steps + 1];
            inferenceSigmas(steps, shift, out);
            try testing.expectApproxEqAbs(@as(f64, 1.0), out[0], 1e-12);
            try testing.expectEqual(@as(f64, 0.0), out[steps]);
            for (0..steps) |i| {
                try testing.expect(out[i] > out[i + 1]);
                try testing.expect(out[i] > 0.0 and out[i] <= 1.0);
            }
        }
    }
}

test "sd3 schedule: shift 3 bends the ladder toward high noise, and shift 1 is the identity" {
    // The whole point of `shift`. At 28 steps the midpoint sigma is ~0.5
    // unshifted and ~0.75 at shift 3 — the model spends more of its budget where
    // a flow model does its composition. Reading the field as 1.0 does not
    // error; it produces structurally mushy images.
    var plain: [29]f64 = undefined;
    var shifted: [29]f64 = undefined;
    inferenceSigmas(28, 1.0, &plain);
    inferenceSigmas(28, 3.0, &shifted);
    for (0..28) |i| {
        // Identity at the fixed endpoints, strictly higher everywhere between.
        if (i == 0) {
            try testing.expectApproxEqAbs(plain[i], shifted[i], 1e-12);
        } else {
            try testing.expect(shifted[i] > plain[i]);
        }
    }
    // Hand-computable: uniform 0.5 -> 3*0.5 / (1 + 2*0.5) = 0.75.
    try testing.expectApproxEqAbs(@as(f64, 0.75), shiftSigma(0.5, 3.0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.5), shiftSigma(0.5, 1.0), 1e-12);
}

test "sd3 schedule: the low end is 1/steps, not 0" {
    // diffusers' `linspace(1, 1/steps, steps)`. Ending the ladder AT zero would
    // make the last Euler step a no-op and waste a forward; ending it at 1/steps
    // and appending the zero is what spends every step.
    var four: [5]f64 = undefined;
    inferenceSigmas(4, 1.0, &four);
    try testing.expectApproxEqAbs(@as(f64, 0.25), four[3], 1e-12);
    try testing.expectEqual(@as(f64, 0.0), four[4]);

    // Turbo's own ladder, shifted: 1/4 = 0.25 -> 3*.25/(1+2*.25) = 0.5.
    var turbo: [5]f64 = undefined;
    inferenceSigmas(4, TURBO_CONFIG.shift, &turbo);
    try testing.expectApproxEqAbs(@as(f64, 0.5), turbo[3], 1e-12);
}

test "sd3 schedule: the timestep handed to the DiT is sigma*1000, and the Euler step is a plain difference" {
    // A flow model's time input is the sigma scaled to the training range. Off
    // by the 1000 and the conditioning is meaningless.
    try testing.expectApproxEqAbs(@as(f32, 1000.0), timestepForSigma(1.0), 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 750.0), timestepForSigma(0.75), 1e-3);
    // `x + (sigma_next - sigma) * v`: the coefficient is NEGATIVE going down the
    // ladder, which is what makes the velocity subtract noise.
    try testing.expect(eulerStepCoeff(1.0, 0.75) < 0.0);
    try testing.expectApproxEqAbs(@as(f64, -0.25), eulerStepCoeff(1.0, 0.75), 1e-12);
}

test "sd3 geometry: inner width, patch grid and the CENTRAL pos-embed crop" {
    const cfg = MmditConfig{};
    try testing.expectEqual(@as(u32, 2432), cfg.innerDim());
    // A 1024px canvas: 128 latent, 64 patches a side.
    const g = patchGrid(cfg, latentSideFor(1024), latentSideFor(1024));
    try testing.expectEqual(@as(u32, 64), g.h);
    try testing.expectEqual(@as(u32, 64), g.w);
    // `(192 - 64) / 2` — centred, not top-left. Off by one here mis-frames every
    // generation with nothing to error on.
    try testing.expectEqual(@as(u32, 64), posEmbedCropStart(cfg, 64));
    // Odd remainders FLOOR, matching python's `//`: (192-61)/2 = 65, not 66.
    try testing.expectEqual(@as(u32, 65), posEmbedCropStart(cfg, 61));
    // A grid at or past the stored size takes the whole thing.
    try testing.expectEqual(@as(u32, 0), posEmbedCropStart(cfg, 192));
    try testing.expectEqual(@as(u32, 0), posEmbedCropStart(cfg, 256));
}

test "sd3 mmdit config: Large and Medium are ONE code path, told apart by the config" {
    const a = testing.allocator;
    // Verbatim `transformer/config.json` from
    // `stabilityai/stable-diffusion-3.5-large` (via its ungated mirror).
    const large_json =
        \\{"_class_name":"SD3Transformer2DModel","attention_head_dim":64,
        \\"caption_projection_dim":2432,"in_channels":16,"joint_attention_dim":4096,
        \\"num_attention_heads":38,"num_layers":38,"out_channels":16,"patch_size":2,
        \\"pooled_projection_dim":2048,"pos_embed_max_size":192,"qk_norm":"rms_norm",
        \\"sample_size":128}
    ;
    const large = try parseMmditConfig(a, large_json);
    defer freeMmditConfig(a, large);
    try testing.expectEqual(@as(u32, 38), large.num_layers);
    try testing.expectEqual(@as(u32, 2432), large.innerDim());
    try testing.expectEqual(@as(u32, 192), large.pos_embed_max_size);
    try testing.expect(large.qk_norm);
    // The discriminator: Large has NO dual-attention layers.
    try testing.expectEqual(@as(usize, 0), large.dual_attention_layers.len);
    try testing.expect(!large.isDualAttention(0));

    // Verbatim from `stable-diffusion-3.5-medium`. Same `_class_name`, a
    // genuinely different transformer — which is why nothing may branch on a
    // model NAME.
    const medium_json =
        \\{"_class_name":"SD3Transformer2DModel","dual_attention_layers":[0,1,2,3,4,5,6,7,8,9,10,11,12],
        \\"attention_head_dim":64,"caption_projection_dim":1536,"in_channels":16,
        \\"joint_attention_dim":4096,"num_attention_heads":24,"num_layers":24,
        \\"out_channels":16,"patch_size":2,"pooled_projection_dim":2048,
        \\"pos_embed_max_size":384,"qk_norm":"rms_norm","sample_size":128}
    ;
    const medium = try parseMmditConfig(a, medium_json);
    defer freeMmditConfig(a, medium);
    try testing.expectEqual(@as(u32, 24), medium.num_layers);
    try testing.expectEqual(@as(u32, 1536), medium.innerDim());
    try testing.expectEqual(@as(u32, 384), medium.pos_embed_max_size);
    try testing.expectEqual(@as(usize, 13), medium.dual_attention_layers.len);
    try testing.expect(medium.isDualAttention(12));
    try testing.expect(!medium.isDualAttention(13));
    // Same everywhere else — the shared fields are what make one code path work.
    try testing.expectEqual(large.patch_size, medium.patch_size);
    try testing.expectEqual(large.in_channels, medium.in_channels);
    try testing.expectEqual(large.joint_attention_dim, medium.joint_attention_dim);
    try testing.expectEqual(large.pooled_projection_dim, medium.pooled_projection_dim);
    try testing.expectEqual(MEDIUM_CONFIG.innerDim(), medium.innerDim());
    try testing.expectEqual(MEDIUM_CONFIG.pos_embed_max_size, medium.pos_embed_max_size);

    // Medium's bigger stored grid is what lets it frame a larger canvas without
    // interpolating: 384 patches = 6144px of latent stride.
    try testing.expectEqual(@as(u32, 160), posEmbedCropStart(medium, 64));
}

test "sd3 mmdit config: an omitted field means Large's value, and a norm we do not serve is REFUSED" {
    const a = testing.allocator;
    // A checkpoint that omits a field is declaring the default, which is what
    // keeps `_class_name`-identical repos on one code path.
    const sparse = try parseMmditConfig(a, "{\"num_layers\":12}");
    defer freeMmditConfig(a, sparse);
    try testing.expectEqual(@as(u32, 12), sparse.num_layers);
    try testing.expectEqual(@as(u32, 38), sparse.num_attention_heads);
    try testing.expectEqual(@as(u32, 2), sparse.patch_size);

    // `qk_norm` is a STRING naming a norm. An unknown one must not quietly run
    // unnormed — that is a plausible-image failure with nothing to error on.
    try testing.expectError(error.UnsupportedQkNorm, parseMmditConfig(a, "{\"qk_norm\":\"layer_norm\"}"));
    // Absent or an explicit null is SD 3 (non-3.5), which genuinely has no
    // qk-norm — that is a real shape, not an error.
    const none = try parseMmditConfig(a, "{\"qk_norm\":null}");
    defer freeMmditConfig(a, none);
    try testing.expect(!none.qk_norm);
    const missing = try parseMmditConfig(a, "{}");
    defer freeMmditConfig(a, missing);
    try testing.expect(!missing.qk_norm);
}

test "sd3 vae: the latent is SHIFTED as well as scaled, and the pair round-trips" {
    const cfg = VaeConfig{};
    try testing.expectEqual(@as(u32, 16), cfg.latent_channels);
    // SDXL's VAE has no shift, so a scale-only decode of an SD 3.5 latent is a
    // uniform colour cast — plausible output, systematically wrong.
    try testing.expect(cfg.shift_factor != 0.0);
    for ([_]f64{ -3.0, 0.0, 0.25, 7.5 }) |v| {
        try testing.expectApproxEqAbs(v, decodeScale(cfg, encodeScale(cfg, v)), 1e-12);
    }
    // Pinned against the checkpoint's own numbers rather than the round trip
    // alone, which any (scale, shift) pair satisfies.
    try testing.expectApproxEqAbs(@as(f64, 1.5305), cfg.scaling_factor, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0609), cfg.shift_factor, 1e-12);
}

test "sd3 turbo differs from Large in STEPS and GUIDANCE, never in the schedule" {
    // Same transformer, same VAE, same towers, byte-identical scheduler config.
    // A future reader looking for the Turbo arm should find it here and nowhere
    // else — the pipeline must not grow a second schedule for it.
    const base = SchedulerConfig{};
    try testing.expectEqual(base.shift, TURBO_CONFIG.shift);
    try testing.expect(TURBO_CONFIG.default_guidance <= 1.0);
    try testing.expect(TURBO_CONFIG.default_steps < base.default_steps);
}

test "sd3 discovery: the THIRD tower is what makes the repo shape unmistakable" {
    const a = testing.allocator;
    // Verbatim `model_index.json` from `stabilityai/stable-diffusion-3.5-large`.
    const sd3_index =
        \\{"_class_name":"StableDiffusion3Pipeline","_diffusers_version":"0.30.3.dev0",
        \\"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"],
        \\"text_encoder":["transformers","CLIPTextModelWithProjection"],
        \\"text_encoder_2":["transformers","CLIPTextModelWithProjection"],
        \\"text_encoder_3":["transformers","T5EncoderModel"],
        \\"tokenizer":["transformers","CLIPTokenizer"],
        \\"tokenizer_2":["transformers","CLIPTokenizer"],
        \\"tokenizer_3":["transformers","T5TokenizerFast"],
        \\"transformer":["diffusers","SD3Transformer2DModel"],
        \\"vae":["diffusers","AutoencoderKL"]}
    ;
    try testing.expect(indexDeclaresSd3(a, sd3_index));
    // Medium and Turbo declare the same class and the same component set.
    try testing.expect(isSd3PipelineClass("StableDiffusion3Img2ImgPipeline"));

    // The families already served must NOT answer to this, and this must not
    // answer to them — a repo classified twice is the double-residency class.
    const sdxl_index =
        \\{"_class_name":"StableDiffusionXLPipeline","unet":["diffusers","UNet2DConditionModel"],
        \\"text_encoder":["transformers","CLIPTextModel"],
        \\"text_encoder_2":["transformers","CLIPTextModelWithProjection"],
        \\"vae":["diffusers","AutoencoderKL"]}
    ;
    const sd1_index =
        \\{"_class_name":"StableDiffusionPipeline","unet":["diffusers","UNet2DConditionModel"],
        \\"text_encoder":["transformers","CLIPTextModel"],"vae":["diffusers","AutoencoderKL"]}
    ;
    try testing.expect(!indexDeclaresSd3(a, sdxl_index));
    try testing.expect(!indexDeclaresSd3(a, sd1_index));

    // A declared SD3 class WITHOUT the T5 tower is a repo this engine cannot
    // load, so it must not be claimed — the `indexDeclaresSdxl` rule about a
    // missing `text_encoder_2`, one tower along.
    const no_t5 =
        \\{"_class_name":"StableDiffusion3Pipeline","transformer":["diffusers","SD3Transformer2DModel"],
        \\"text_encoder":["transformers","CLIPTextModelWithProjection"]}
    ;
    try testing.expect(!indexDeclaresSd3(a, no_t5));

    // Garbage in never panics — discovery walks arbitrary user dirs.
    try testing.expect(!indexDeclaresSd3(a, "not json"));
    try testing.expect(!indexDeclaresSd3(a, "[]"));
    try testing.expect(!indexDeclaresSd3(a, "{}"));
    try testing.expect(!indexDeclaresSd3(a, "{\"_class_name\":42}"));
}
