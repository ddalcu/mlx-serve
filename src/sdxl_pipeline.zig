//! The SDXL txt2img pipeline — diffusers' `StableDiffusionXLPipeline`.
//!
//! Composes the four pieces that land separately (two CLIP towers, the UNet,
//! the VAE decoder) into the loop that actually makes an image, and owns the
//! three decisions none of them can make alone:
//!
//!   THE `steps_offset` COMPOSITION. `sdxl.timestepIndices` deliberately leaves
//!   the offset out so the spacing rule stays one idea. SDXL's scheduler config
//!   sets `steps_offset: 1`, and the offset applies to BOTH the timestep handed
//!   to the UNet and the index the sigma is read at — apply it to one and not
//!   the other and the model is denoising at a sigma it was not told about.
//!   `buildSchedule` is the one place they are derived together, and it is
//!   pinned against diffusers' own `EulerDiscreteScheduler` below.
//!
//!   `force_zeros_for_empty_prompt`. With an empty negative prompt, SDXL's
//!   unconditional branch is ZEROS — not the encoding of the empty string.
//!   Those are different tensors: the empty string still carries BOS, EOS and
//!   75 pads through both towers and comes out non-zero. The checkpoint
//!   declares which behaviour it wants, and this reads that flag.
//!
//!   TWO FORWARDS PER STEP. Guidance is evaluated as two separate batch-1 UNet
//!   forwards rather than one batch-2 forward. A batch-2 pass would be faster,
//!   but every reshape in `sdxl_unet.zig` is written against batch 1; widening
//!   it is a real change with its own parity risk, so it is deliberately left
//!   as the obvious next optimization rather than smuggled in here.
//!
//! ORACLE STATUS: the schedule is pinned numerically against diffusers. The
//! components are each pinned against diffusers separately. The COMPOSITION —
//! that these particular pieces in this order produce the same image as
//! `StableDiffusionXLPipeline` — is covered by `tests/test_sdxl_gen.sh`, which
//! checks structure and sanity rather than pixels.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const sdxl = @import("sdxl.zig");
const clip = @import("sdxl_clip.zig");
const clip_tok = @import("sdxl_tokenizer.zig");
const unet_mod = @import("sdxl_unet.zig");
const vae_mod = @import("sdxl_vae.zig");
const nn = @import("sdxl_nn.zig");
const krea = @import("krea.zig");
const sse = @import("gen_sse.zig");
const model_mod = @import("model.zig");
const single_file = @import("sdxl_single_file.zig");

const S = mlx.mlx_stream;

/// SDXL base's scheduler config: `timestep_spacing: leading`, `steps_offset: 1`.
/// The distilled variants override both — see `SchedulerConfig`.
pub const STEPS_OFFSET: usize = 1;

/// What `scheduler/scheduler_config.json` declares. This is the ENTIRE
/// difference between base SDXL, SDXL-Turbo and SDXL-Lightning: same UNet,
/// same VAE, same text towers, different schedule. Reading it rather than
/// hardcoding base's values is what lets one engine serve all three.
pub const SchedulerConfig = struct {
    spacing: sdxl.TimestepSpacing = .leading,
    /// Applied ONLY on `leading` — diffusers adds it in that branch alone, and
    /// applying it to `trailing` would shift a distill's carefully placed
    /// 1-4 timesteps off the values it was trained on.
    steps_offset: usize = STEPS_OFFSET,
    prediction: sdxl.PredictionType = .epsilon,
    /// `EulerAncestralDiscreteScheduler` (SDXL-Turbo's declared sampler) adds
    /// fresh noise each step; plain Euler does not.
    ancestral: bool = false,
    /// Zero-terminal-SNR training (diffusers `rescale_betas_zero_snr`). A
    /// v-prediction anime finetune ships it; the stock schedule washes it out.
    zero_snr: bool = false,
    /// Guidance the checkpoint expects when the request says nothing. Base
    /// wants ~5; the distills are trained guidance-free, and at <= 1 the
    /// pipeline skips the unconditional forward entirely — half the work.
    default_guidance: f32 = 5.0,

    /// Only `leading` carries the offset.
    pub fn effectiveOffset(self: SchedulerConfig) usize {
        return if (self.spacing == .leading) self.steps_offset else 0;
    }
};

pub const GenOpts = struct {
    width: u32 = 1024,
    height: u32 = 1024,
    steps: u32 = 30,
    /// `guidance_scale`. NULL means "whatever this checkpoint was distilled
    /// for" (`SchedulerConfig.default_guidance`): ~5 on base, 1.0 on Turbo and
    /// Lightning. At <= 1 the unconditional forward is skipped entirely, which
    /// is half the work — so a distill is twice as fast per step as well as
    /// needing an order of magnitude fewer steps.
    guidance: ?f32 = null,
    seed: u64 = 0,
    /// ABSENT (null) and EMPTY ("") are different requests, and diffusers
    /// draws the line in the same place: `zero_out_negative_prompt` is
    /// `negative_prompt is None and force_zeros_for_empty_prompt`. So a
    /// request that says nothing gets the zeroed unconditional branch, while
    /// one that explicitly sends an empty string gets the empty string
    /// ENCODED — which is a different tensor, since "" still carries BOS, EOS
    /// and 75 pads through both towers. Collapsing the two to `""` silently
    /// picks one behaviour for both.
    negative_prompt: ?[]const u8 = null,
    /// Override the checkpoint's declared timestep spacing.
    ///
    /// Needed because SDXL-Lightning ships as a LoRA over BASE SDXL, and the
    /// base pack's `scheduler_config.json` says `leading` — the adapter
    /// changes the weights, not the config beside them. Lightning is trained
    /// for `trailing`, and at 4 steps the difference is not subtle: leading
    /// starts at t=751 and never sees the top of the schedule. diffusers users
    /// do the same thing by hand (`EulerDiscreteScheduler.from_config(...,
    /// timestep_spacing="trailing")`).
    spacing: ?sdxl.TimestepSpacing = null,
    /// Micro-conditioning overrides; null means `sdxl.defaultTimeIds`.
    original_size: ?[2]u32 = null,
    crop_top_left: ?[2]u32 = null,
    target_size: ?[2]u32 = null,
    /// An explicit UNIT-variance starting latent, `[1, 4, h/8, w/8]`, used in
    /// place of the seeded draw. It is scaled by `init_noise_sigma` here, the
    /// same as a drawn one — diffusers does that scaling inside the pipeline
    /// too, so a fixture stores the unscaled tensor and both sides apply it.
    ///
    /// This exists for the end-to-end parity test: the random draw is the only
    /// legitimate difference between our pipeline and diffusers', so removing
    /// it is what makes the two comparable. Borrowed, never freed here.
    init_latent: ?mlx.mlx_array = null,
    /// img2img: source pixels `[1,3,H,W]` f32 `[0,1]`, pre-resized to the
    /// target size. VAE-encoded internally and mixed with fresh noise at
    /// `sigmas[start_step]` — diffusers' `EulerDiscreteScheduler.add_noise`
    /// convention (`sample + noise * sigma`), not flow-match's linear blend.
    init_image: ?mlx.mlx_array = null,
    /// First schedule index to run (img2img skip; 0 = full schedule, i.e. the
    /// same fresh-noise start as no `init_image` at all).
    start_step: u32 = 0,
    /// Conditioning injected in place of running our own text encoders — the
    /// `[1,77,2048]` stream and the `[1,1280]` pooled vector. PARITY
    /// ATTRIBUTION ONLY: with these set, an end-to-end comparison against
    /// diffusers isolates the denoise loop from the text towers, which is the
    /// only way to tell "the loop is wrong" from "the loop faithfully
    /// amplifies a small encoder difference". Borrowed, never freed here.
    cond_override: ?struct { ctx: mlx.mlx_array, pooled: mlx.mlx_array } = null,
};

/// One resolved denoising schedule.
pub const Schedule = struct {
    /// Timesteps handed to the UNet, descending. Length `steps`.
    timesteps: []f32,
    /// Sigmas, length `steps + 1`; the last is 0.
    sigmas: []f64,
    /// What a fresh latent is multiplied by before the first step.
    init_noise_sigma: f64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Schedule) void {
        self.allocator.free(self.timesteps);
        self.allocator.free(self.sigmas);
    }
};

/// Derive the timesteps and sigmas together, applying `steps_offset` to both.
///
/// diffusers computes `timesteps = arange(steps) * ratio` reversed, `+= offset`,
/// then reads the sigma ladder AT THOSE TIMESTEPS. The offset is therefore not
/// cosmetic: it shifts which sigma each step runs at.
pub fn buildSchedule(
    allocator: std.mem.Allocator,
    steps: usize,
    spacing: sdxl.TimestepSpacing,
    offset: usize,
    /// Rescale the schedule so the terminal SNR is zero. A property of how the
    /// checkpoint was TRAINED (NoobAI V-Pred ships a `ztsnr` marker tensor), not
    /// a quality knob — see `sdxl.rescaleZeroTerminalSnr`.
    zero_snr: bool,
) !Schedule {
    if (steps == 0) return error.ZeroSteps;

    const betas = try allocator.alloc(f64, sdxl.NUM_TRAIN_TIMESTEPS);
    defer allocator.free(betas);
    sdxl.scaledLinearBetas(betas);
    const acp = try allocator.alloc(f64, sdxl.NUM_TRAIN_TIMESTEPS);
    defer allocator.free(acp);
    sdxl.alphasCumprod(betas, acp);
    if (zero_snr) sdxl.rescaleZeroTerminalSnr(acp);
    const train = try allocator.alloc(f64, sdxl.NUM_TRAIN_TIMESTEPS);
    defer allocator.free(train);
    sdxl.trainSigmas(acp, train);

    const idx = try allocator.alloc(usize, steps);
    defer allocator.free(idx);
    sdxl.timestepIndices(spacing, steps, idx);

    // The offset applies to BOTH the reported timestep and the sigma lookup.
    // Clamped so an offset past the table's end cannot index out of it.
    const shifted = try allocator.alloc(usize, steps);
    defer allocator.free(shifted);
    for (idx, shifted) |v, *o| o.* = @min(v + offset, sdxl.NUM_TRAIN_TIMESTEPS - 1);

    const sigmas = try allocator.alloc(f64, steps + 1);
    errdefer allocator.free(sigmas);
    sdxl.inferenceSigmas(train, shifted, sigmas);

    const timesteps = try allocator.alloc(f32, steps);
    errdefer allocator.free(timesteps);
    for (shifted, timesteps) |v, *o| o.* = @floatFromInt(v);

    return .{
        .timesteps = timesteps,
        .sigmas = sigmas,
        .init_noise_sigma = sdxl.initNoiseSigma(sigmas),
        .allocator = allocator,
    };
}

/// A prompt encoded through both towers.
const Conditioning = struct {
    /// `[1, 77, 2048]` — CLIP-L's penultimate concatenated with bigG's.
    ctx: mlx.mlx_array,
    /// `[1, 1280]` — bigG's pooled projection.
    pooled: mlx.mlx_array,

    fn deinit(self: *Conditioning) void {
        _ = mlx.mlx_array_free(self.ctx);
        _ = mlx.mlx_array_free(self.pooled);
    }
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: S,
    tok_l: clip_tok.ClipTokenizer,
    tok_g: clip_tok.ClipTokenizer,
    tower_l: clip.TextTower,
    tower_g: clip.TextTower,
    unet: unet_mod.Unet,
    vae: vae_mod.Decoder,
    /// Null when the pack's `vae/` has no `encoder.*` tensors — a single-file
    /// checkpoint (`loadSingleFile`; `sdxl_single_file` deliberately converts
    /// decode-only) or a folder repo whose encoder failed to load. img2img is
    /// unavailable either way (`ImageEngine.supportsImg2Img`).
    vae_enc: ?vae_mod.Encoder,
    /// From `model_index.json`; see the file header.
    force_zeros_for_empty_prompt: bool,
    /// From `scheduler/scheduler_config.json` — what makes this base, Turbo
    /// or Lightning.
    sched_cfg: SchedulerConfig,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const s = mlx.mlx_default_gpu_stream_new();
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = s;
        self.force_zeros_for_empty_prompt = readForceZeros(io, allocator, model_dir);
        self.sched_cfg = readSchedulerConfig(io, allocator, model_dir);
        log.info("[sdxl] schedule: spacing={s} offset={d} prediction={s} ancestral={} guidance={d:.1}\n", .{
            @tagName(self.sched_cfg.spacing),    self.sched_cfg.effectiveOffset(),
            @tagName(self.sched_cfg.prediction), self.sched_cfg.ancestral,
            self.sched_cfg.default_guidance,
        });

        self.tok_l = try clip_tok.load(io, allocator, model_dir, "tokenizer");
        errdefer self.tok_l.deinit();
        self.tok_g = try clip_tok.load(io, allocator, model_dir, "tokenizer_2");
        errdefer self.tok_g.deinit();

        self.tower_l = try clip.loadTower(io, allocator, s, model_dir, "text_encoder", sdxl.CLIP_L_CONFIG, clip.towerDtype());
        errdefer self.tower_l.deinit();
        self.tower_g = try clip.loadTower(io, allocator, s, model_dir, "text_encoder_2", sdxl.CLIP_BIG_G_CONFIG, clip.towerDtype());
        errdefer self.tower_g.deinit();

        // The UNet serves at fp16; the VAE must not (`force_upcast`).
        // `SDXL_UNET_F32=1` widens the UNet, which is how a parity gap gets
        // split into "wrong math" and "accumulated precision" — a logic error
        // does not improve when the dtype widens.
        const unet_dtype: mlx.mlx_dtype = if (std.c.getenv("SDXL_UNET_F32") != null) .float32 else .float16;
        self.unet = try unet_mod.load(io, allocator, s, model_dir, unet_dtype);
        errdefer self.unet.deinit();
        self.vae = try vae_mod.load(io, allocator, s, model_dir, vae_mod.DEFAULT_DTYPE);
        errdefer self.vae.deinit();
        self.vae_enc = vae_mod.loadEncoder(io, allocator, s, model_dir, vae_mod.DEFAULT_DTYPE) catch |e| blk: {
            log.warn("[sdxl] VAE encoder load failed ({}) — image-to-image disabled\n", .{e});
            break :blk null;
        };
        return self;
    }

    /// Load a SINGLE-FILE SDXL checkpoint (the Civitai / A1111 distribution:
    /// one `.safetensors` in LDM key naming, no configs, no tokenizer files —
    /// how Illustrious XL and Pony Diffusion XL are shipped).
    ///
    /// The blob is loaded once, converted to the diffusers layout the folder
    /// path already binds (`sdxl_single_file`), and fed through the same
    /// `*FromWeights` binders. Configs are SDXL-standard constants: the geometry
    /// is `unet_mod.BASE_CONFIG`, the schedule the base default (leading /
    /// epsilon / guidance 5), the tokenizer the embedded CLIP BPE. A checkpoint
    /// with a non-standard schedule (a Turbo/Lightning single-file) still runs
    /// but at base defaults — the single-file format carries nothing that could
    /// say otherwise.
    pub fn loadSingleFile(allocator: std.mem.Allocator, abs_path: []const u8) !*Engine {
        const s = mlx.mlx_default_gpu_stream_new();
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = s;
        self.force_zeros_for_empty_prompt = true; // SDXL base default
        self.sched_cfg = .{};

        var ldm = try model_mod.loadWeightsSingleFile(allocator, abs_path);
        defer ldm.deinit();
        if (!single_file.isLdmSdxl(&ldm)) return error.NotAnSdxlCheckpoint;

        // The checkpoint's own marker tensors outrank every default here: a
        // v-prediction model sampled as epsilon produces a plausible, silently
        // wrong image (see `single_file.TrainingMarkers`).
        const markers = single_file.markersOf(&ldm);
        if (markers.v_prediction) self.sched_cfg.prediction = .v_prediction;
        if (markers.zero_snr) self.sched_cfg.zero_snr = true;
        if (markers.any()) {
            log.info("[sdxl] checkpoint markers: v_pred={} ztsnr={}\n", .{ markers.v_prediction, markers.zero_snr });
        }

        var w = try single_file.convert(allocator, &ldm);
        defer w.deinit();

        self.tok_l = try clip_tok.initFromBytes(allocator, single_file.CLIP_VOCAB_JSON, single_file.CLIP_MERGES_TXT, single_file.CLIP_L_PAD_ID);
        errdefer self.tok_l.deinit();
        self.tok_g = try clip_tok.initFromBytes(allocator, single_file.CLIP_VOCAB_JSON, single_file.CLIP_MERGES_TXT, single_file.CLIP_BIGG_PAD_ID);
        errdefer self.tok_g.deinit();

        // The two towers read SEPARATE maps — identical `text_model.*` key names.
        self.tower_l = try clip.loadTowerFromWeights(allocator, s, &w.clip_l, "text_encoder", sdxl.CLIP_L_CONFIG, clip.towerDtype());
        errdefer self.tower_l.deinit();
        self.tower_g = try clip.loadTowerFromWeights(allocator, s, &w.clip_g, "text_encoder_2", sdxl.CLIP_BIG_G_CONFIG, clip.towerDtype());
        errdefer self.tower_g.deinit();

        const unet_dtype: mlx.mlx_dtype = if (std.c.getenv("SDXL_UNET_F32") != null) .float32 else .float16;
        // BASE_CONFIG's slices are comptime literals — never freed, so owns_cfg=false.
        self.unet = try unet_mod.loadFromWeights(allocator, s, &w.main, unet_mod.BASE_CONFIG, unet_dtype, false);
        errdefer self.unet.deinit();
        self.vae = try vae_mod.loadFromWeights(allocator, s, &w.main, sdxl.VAE_SCALING_FACTOR, vae_mod.DEFAULT_DTYPE);
        // `sdxl_single_file.convertVae` deliberately converts decode-only — the
        // LDM checkpoint's encoder tensors are never carried into `w.main`, so
        // there is nothing here to bind. img2img is unavailable on a
        // single-file checkpoint until that converter grows an encoder arm.
        self.vae_enc = null;

        log.info("[sdxl] loaded single-file checkpoint: {s}\n", .{abs_path});
        return self;
    }

    /// Load from a resolved model directory, choosing the layout: a diffusers
    /// repo (has `model_index.json`) goes through `load`; a directory holding a
    /// single LDM `.safetensors` and no configs goes through `loadSingleFile`.
    /// This is the entry point the media registry calls.
    pub fn loadAuto(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        // A diffusers repo is canonical when its index is present.
        const idx = try std.fmt.allocPrint(allocator, "{s}/model_index.json", .{model_dir});
        defer allocator.free(idx);
        const has_index = if (std.Io.Dir.openFileAbsolute(io, idx, .{})) |f| blk: {
            f.close(io);
            break :blk true;
        } else |_| false;
        if (!has_index) {
            if (try single_file.findLdmSdxlFile(io, allocator, model_dir)) |path| {
                defer allocator.free(path);
                return loadSingleFile(allocator, path);
            }
        }
        return load(io, allocator, model_dir);
    }

    pub fn deinit(self: *Engine) void {
        self.tok_l.deinit();
        self.tok_g.deinit();
        self.tower_l.deinit();
        self.tower_g.deinit();
        self.unet.deinit();
        self.vae.deinit();
        if (self.vae_enc) |*e| e.deinit();
        self.allocator.destroy(self);
    }

    /// Run both towers over one prompt and assemble the conditioning.
    ///
    /// The two penultimate states are concatenated on the FEATURE axis in the
    /// order (CLIP-L, bigG) — 768 then 1280, making 2048. Swapping them keeps
    /// the shape and destroys the cross-attention.
    fn encodePrompt(self: *Engine, text: []const u8) !Conditioning {
        const a = self.allocator;
        const s = self.s;

        var ids_l = try self.tok_l.encode(a, text);
        defer ids_l.deinit();
        var ids_g = try self.tok_g.encode(a, text);
        defer ids_g.deinit();

        var enc_l = try self.tower_l.encode(ids_l.ids, ids_l.eos_index, false);
        defer enc_l.deinit();
        var enc_g = try self.tower_g.encode(ids_g.ids, ids_g.eos_index, false);
        defer enc_g.deinit();

        const pooled = enc_g.pooled orelse return error.MissingPooledEmbedding;
        const ctx = try nn.concat(&[_]mlx.mlx_array{ enc_l.penultimate, enc_g.penultimate }, 2, s);
        errdefer _ = mlx.mlx_array_free(ctx);
        const pooled_owned = try nn.dupA(pooled);
        return .{ .ctx = ctx, .pooled = pooled_owned };
    }

    /// The unconditional branch. An ABSENT negative prompt on a checkpoint
    /// that declares `force_zeros_for_empty_prompt` is ZEROS, not the encoding
    /// of "" — see `GenOpts.negative_prompt` for why those differ.
    fn encodeNegative(self: *Engine, text: ?[]const u8, like: *const Conditioning) !Conditioning {
        if (text == null and self.force_zeros_for_empty_prompt) {
            const s = self.s;
            // Multiply by zero rather than allocate: this inherits the shape
            // AND the dtype of the conditional branch, which is what the
            // guidance mix has to line up against.
            const zero = mlx.mlx_array_new_float(0.0);
            defer _ = mlx.mlx_array_free(zero);
            const ctx = try nn.mulA(like.ctx, zero, s);
            errdefer _ = mlx.mlx_array_free(ctx);
            const pooled = try nn.mulA(like.pooled, zero, s);
            return .{ .ctx = ctx, .pooled = pooled };
        }
        return self.encodePrompt(text orelse "");
    }

    /// Generate one image. Returns NCHW `[1, 3, H, W]` float32 in [0, 1].
    pub fn generate(
        self: *Engine,
        prompt: []const u8,
        opts: GenOpts,
        progress: ?sse.Progress,
    ) !mlx.mlx_array {
        const a = self.allocator;
        const s = self.s;

        const dims = sdxl.latentDims(opts.width, opts.height) orelse return error.UnsupportedResolution;
        if (opts.steps == 0) return error.ZeroSteps;

        var scfg = self.sched_cfg;
        if (opts.spacing) |sp| {
            scfg.spacing = sp;
            // A caller asking for trailing is asking for a distilled schedule,
            // which is guidance-free unless they also named a guidance.
            if (sp == .trailing) scfg.default_guidance = 1.0;
        }
        var sched = try buildSchedule(a, opts.steps, scfg.spacing, scfg.effectiveOffset(), scfg.zero_snr);
        defer sched.deinit();

        var cond = if (opts.cond_override) |ov| Conditioning{
            .ctx = try nn.dupA(ov.ctx),
            .pooled = try nn.dupA(ov.pooled),
        } else try self.encodePrompt(prompt);
        defer cond.deinit();
        var uncond = try self.encodeNegative(opts.negative_prompt, &cond);
        defer uncond.deinit();

        const os = opts.original_size orelse [2]u32{ opts.height, opts.width };
        const ct = opts.crop_top_left orelse [2]u32{ 0, 0 };
        const ts = opts.target_size orelse [2]u32{ opts.height, opts.width };
        const time_ids = sdxl.addTimeIds(os[0], os[1], ct[0], ct[1], ts[0], ts[1]);

        const start_step: u32 = @min(opts.start_step, @as(u32, @intCast(opts.steps)) - 1);

        // ── Starting latent. img2img: VAE-encode the source and mix with
        // fresh noise at `sigmas[start_step]` (`sample + noise * sigma` —
        // diffusers' `EulerDiscreteScheduler.add_noise`). Otherwise a fresh
        // draw scaled into the Euler formulation's space, which is the same
        // formula with an all-zero source at `sigmas[0]`.
        var latent = if (opts.init_image) |pix| blk: {
            const ve = if (self.vae_enc) |*e| e else return error.NoVaeEncoder;
            const z0 = try ve.encode(pix);
            defer _ = mlx.mlx_array_free(z0);
            var key = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(key);
            try mlx.check(mlx.mlx_random_key(&key, opts.seed));
            const shape = [_]c_int{ 1, @intCast(sdxl.LATENT_CHANNELS), @intCast(dims.h), @intCast(dims.w) };
            var noise = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&noise, &shape, 4, .float32, 0.0, 1.0, key, s));
            defer _ = mlx.mlx_array_free(noise);
            const sigma0 = mlx.mlx_array_new_float(@floatCast(sched.sigmas[start_step]));
            defer _ = mlx.mlx_array_free(sigma0);
            const noise_scaled = try nn.mulA(noise, sigma0, s);
            defer _ = mlx.mlx_array_free(noise_scaled);
            break :blk try nn.addA(z0, noise_scaled, s);
        } else blk: {
            const noise = if (opts.init_latent) |given| given else nz: {
                var key = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(key);
                try mlx.check(mlx.mlx_random_key(&key, opts.seed));
                const shape = [_]c_int{ 1, @intCast(sdxl.LATENT_CHANNELS), @intCast(dims.h), @intCast(dims.w) };
                var drawn = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_random_normal(&drawn, &shape, 4, .float32, 0.0, 1.0, key, s));
                break :nz drawn;
            };
            // An injected latent is borrowed; a drawn one is ours to release.
            defer if (opts.init_latent == null) {
                _ = mlx.mlx_array_free(noise);
            };
            const scale = mlx.mlx_array_new_float(@floatCast(sched.init_noise_sigma));
            defer _ = mlx.mlx_array_free(scale);
            break :blk try nn.mulA(noise, scale, s);
        };
        defer _ = mlx.mlx_array_free(latent);

        // A request that says nothing takes the CHECKPOINT's own default, so a
        // Turbo pack runs guidance-free without the caller having to know.
        const guidance = opts.guidance orelse scfg.default_guidance;
        const do_cfg = guidance > 1.0;
        const mix = sdxl.cfgMix(guidance);

        if (std.c.getenv("MLX_SERVE_SDXL_TRACE") != null) {
            std.debug.print("[sdxl-trace] init sigma={d:.6} latent_std={d:.6} ctx_std={d:.6} pooled_std={d:.6}\n", .{
                sched.init_noise_sigma, try stdOf(latent, s), try stdOf(cond.ctx, s), try stdOf(cond.pooled, s),
            });
        }

        const run_steps = opts.steps - start_step;
        for (start_step..opts.steps) |i| {
            // Poll for a departed client BEFORE spending a step on it. A
            // guided SDXL step is two full UNet forwards, and without this a
            // cancelled request burns the GPU to completion with every other
            // request queued behind it.
            if (progress) |p| if (p.cancelled()) return error.Cancelled;

            const sigma = sched.sigmas[i];
            const sigma_next = sched.sigmas[i + 1];

            // `scale_model_input` — the UNet sees the latent divided by
            // sqrt(sigma^2+1), never the raw one.
            const scaled = blk: {
                const c = mlx.mlx_array_new_float(@floatCast(sdxl.scaleModelInput(sigma)));
                defer _ = mlx.mlx_array_free(c);
                break :blk try nn.mulA(latent, c, s);
            };
            defer _ = mlx.mlx_array_free(scaled);

            const eps_cond = try self.unet.forward(scaled, sched.timesteps[i], cond.ctx, cond.pooled, &time_ids);
            defer _ = mlx.mlx_array_free(eps_cond);

            const eps = if (do_cfg) blk: {
                const eps_un = try self.unet.forward(scaled, sched.timesteps[i], uncond.ctx, uncond.pooled, &time_ids);
                defer _ = mlx.mlx_array_free(eps_un);
                const cu = mlx.mlx_array_new_float(@floatCast(mix.uncond));
                defer _ = mlx.mlx_array_free(cu);
                const cc = mlx.mlx_array_new_float(@floatCast(mix.cond));
                defer _ = mlx.mlx_array_free(cc);
                const a1 = try nn.mulA(eps_un, cu, s);
                defer _ = mlx.mlx_array_free(a1);
                const a2 = try nn.mulA(eps_cond, cc, s);
                defer _ = mlx.mlx_array_free(a2);
                break :blk try nn.addA(a1, a2, s);
            } else try nn.dupA(eps_cond);
            defer _ = mlx.mlx_array_free(eps);

            // Euler on the RAW latent, not the scaled one handed to the UNet.
            //
            //   derivative = latent*c.latent + model_out*c.model
            //   next       = latent + derivative * (sigma_target - sigma)
            //
            // The coefficients are what the prediction type changes; for
            // epsilon they are (0, 1) and this reduces to the original
            // `latent + eps*dt`.
            const c = sdxl.derivativeCoeffs(scfg.prediction, sigma);
            const anc = sdxl.ancestralSigmas(sigma, sigma_next);
            const sigma_target = if (scfg.ancestral) anc.sigma_down else sigma_next;

            const eps_f32 = try nn.astype(eps, .float32, s);
            defer _ = mlx.mlx_array_free(eps_f32);
            const deriv = blk: {
                const cm = mlx.mlx_array_new_float(@floatCast(c.model));
                defer _ = mlx.mlx_array_free(cm);
                const from_model = try nn.mulA(eps_f32, cm, s);
                if (c.latent == 0.0) break :blk from_model;
                defer _ = mlx.mlx_array_free(from_model);
                const cl = mlx.mlx_array_new_float(@floatCast(c.latent));
                defer _ = mlx.mlx_array_free(cl);
                const from_latent = try nn.mulA(latent, cl, s);
                defer _ = mlx.mlx_array_free(from_latent);
                break :blk try nn.addA(from_model, from_latent, s);
            };
            defer _ = mlx.mlx_array_free(deriv);

            const dc = mlx.mlx_array_new_float(@floatCast(sigma_target - sigma));
            defer _ = mlx.mlx_array_free(dc);
            const delta = try nn.mulA(deriv, dc, s);
            defer _ = mlx.mlx_array_free(delta);
            nn.replace(&latent, try nn.addA(latent, delta, s));

            // Ancestral (Turbo): the deterministic part only reached
            // `sigma_down`; fresh noise at `sigma_up` makes up the rest. Zero
            // on the final step, so a finished image is never re-noised.
            if (scfg.ancestral and anc.sigma_up > 0.0) {
                var key = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(key);
                try mlx.check(mlx.mlx_random_key(&key, opts.seed +% i +% 1));
                const shape = mlx.getShape(latent);
                var noise = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(noise);
                try mlx.check(mlx.mlx_random_normal(&noise, shape.ptr, shape.len, .float32, 0.0, 1.0, key, s));
                const su = mlx.mlx_array_new_float(@floatCast(anc.sigma_up));
                defer _ = mlx.mlx_array_free(su);
                const scaled_noise = try nn.mulA(noise, su, s);
                defer _ = mlx.mlx_array_free(scaled_noise);
                nn.replace(&latent, try nn.addA(latent, scaled_noise, s));
            }
            _ = mlx.mlx_array_eval(latent);

            // `MLX_SERVE_SDXL_TRACE=1` prints the per-step latent statistics.
            // A denoise that is subtly wrong still produces an image, so the
            // per-step std against the reference is how the divergence gets
            // located to a step rather than guessed at.
            if (std.c.getenv("MLX_SERVE_SDXL_TRACE") != null) {
                // stderr, not `log.info`: this has to be readable from inside
                // `zig build test`, which swallows stdout.
                std.debug.print("[sdxl-trace] step={d} t={d:.1} sigma={d:.6} next={d:.6} std={d:.6}\n", .{
                    i, sched.timesteps[i], sigma, sigma_next, try stdOf(latent, s),
                });
            }

            // Free the per-step allocator pool; a diffusion loop allocates the
            // same shapes every step, but the pool still grows without this.
            _ = mlx.mlx_clear_cache();

            if (progress) |p| p.emit("Generating", @intCast(i + 1 - start_step), run_steps);
        }
        if (progress) |p| p.emit("Decoding image", opts.steps, opts.steps);

        // ── Decode. [-1,1] -> [0,1] here, so callers (PNG, tests) all see one
        // convention.
        const decoded = try self.vae.decodeLatent(latent);
        defer _ = mlx.mlx_array_free(decoded);
        const half = mlx.mlx_array_new_float(0.5);
        defer _ = mlx.mlx_array_free(half);
        const scaled_img = try nn.mulA(decoded, half, s);
        defer _ = mlx.mlx_array_free(scaled_img);
        const shifted = try nn.addA(scaled_img, half, s);
        defer _ = mlx.mlx_array_free(shifted);
        const out = try nn.astype(shifted, .float32, s);
        _ = mlx.mlx_array_eval(out);
        return out;
    }

    /// Generate and encode as PNG bytes.
    pub fn generatePng(
        self: *Engine,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        opts: GenOpts,
        progress: ?sse.Progress,
    ) ![]u8 {
        const img = try self.generate(prompt, opts, progress);
        defer _ = mlx.mlx_array_free(img);
        return krea.imageToPng(allocator, img, self.s);
    }
};

/// Population standard deviation of an array, for the trace.
fn stdOf(x: mlx.mlx_array, s: S) !f64 {
    const f = try nn.astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(f);
    const c = try nn.contiguous(f, s);
    defer _ = mlx.mlx_array_free(c);
    _ = mlx.mlx_array_eval(c);
    const n: usize = @intCast(mlx.mlx_array_size(c));
    if (n == 0) return 0;
    const d = mlx.mlx_array_data_float32(c) orelse return 0;
    var sum: f64 = 0;
    for (0..n) |i| sum += d[i];
    const mean = sum / @as(f64, @floatFromInt(n));
    var acc: f64 = 0;
    for (0..n) |i| {
        const dv = @as(f64, d[i]) - mean;
        acc += dv * dv;
    }
    return @sqrt(acc / @as(f64, @floatFromInt(n)));
}

/// Read `scheduler/scheduler_config.json`.
///
/// Absent or unreadable falls back to base SDXL's values, which is the right
/// default: a pack missing the file is far more likely to be a base mirror
/// than a distill, and a distill run on base's schedule looks obviously wrong
/// (soft, washed) rather than subtly so.
fn readSchedulerConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) SchedulerConfig {
    const cfg = SchedulerConfig{};
    const path = std.fmt.allocPrint(allocator, "{s}/scheduler/scheduler_config.json", .{model_dir}) catch return cfg;
    defer allocator.free(path);
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return cfg;
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return cfg;
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = rs.interface.allocRemaining(allocator, .limited(1 << 20)) catch return cfg;
    defer allocator.free(content);
    return parseSchedulerConfig(allocator, content);
}

/// The pure half of `readSchedulerConfig`, split out so the three variants'
/// real config bytes can be pinned without a checkpoint on disk.
pub fn parseSchedulerConfig(allocator: std.mem.Allocator, content: []const u8) SchedulerConfig {
    var cfg = SchedulerConfig{};
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return cfg;
    defer parsed.deinit();
    if (parsed.value != .object) return cfg;
    const o = parsed.value.object;

    if (o.get("timestep_spacing")) |v| {
        if (v == .string) {
            // An unrecognised name falls back to SDXL's own declared default
            // rather than refusing: this is the CHECKPOINT talking, and a pack
            // that names a spacing we do not serve should still generate.
            cfg.spacing = sdxl.spacingFromString(v.string) orelse .leading;
        }
    }
    if (o.get("steps_offset")) |v| {
        if (v == .integer and v.integer >= 0) cfg.steps_offset = @intCast(v.integer);
    }
    if (o.get("prediction_type")) |v| {
        if (v == .string) {
            // An UNKNOWN prediction type keeps epsilon rather than guessing:
            // the three are different quantities, and a wrong one is noise.
            if (sdxl.PredictionType.fromString(v.string)) |pt| cfg.prediction = pt;
        }
    }
    // The declared sampler class decides whether noise is injected each step.
    if (o.get("_class_name")) |v| {
        if (v == .string) cfg.ancestral = std.mem.indexOf(u8, v.string, "Ancestral") != null;
    }
    if (o.get("rescale_betas_zero_snr")) |v| {
        if (v == .bool) cfg.zero_snr = v.bool;
    }

    // Guidance is NOT in this file — it is a property of how the checkpoint was
    // distilled, so it is inferred from the schedule. `trailing` at these step
    // counts is the distills' signature, and both are trained guidance-free.
    if (cfg.spacing == .trailing) cfg.default_guidance = 1.0;

    return cfg;
}

/// Read `force_zeros_for_empty_prompt` from `model_index.json`. Absent means
/// true — that is diffusers' own default for this pipeline class.
fn readForceZeros(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    const path = std.fmt.allocPrint(allocator, "{s}/model_index.json", .{model_dir}) catch return true;
    defer allocator.free(path);
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return true;
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return true;
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = rs.interface.allocRemaining(allocator, .limited(1 << 20)) catch return true;
    defer allocator.free(content);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return true;
    defer parsed.deinit();
    if (parsed.value != .object) return true;
    const v = parsed.value.object.get("force_zeros_for_empty_prompt") orelse return true;
    return if (v == .bool) v.bool else true;
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

// The sampler is the component that is wrong SILENTLY — a mis-derived schedule
// denoises to a plausible image of the wrong thing. These are diffusers'
// numbers for this checkpoint's own scheduler config, read out of
// `EulerDiscreteScheduler.set_timesteps(8)`.
//
// TOLERANCE, and why it is not tighter: diffusers builds its betas as a
// float32 tensor and takes a 1000-step `cumprod` in float32, so its sigmas
// carry accumulated rounding. We accumulate in f64. The gap is MEASURED, not
// assumed — at index 876, torch in float32 gives 7.371846675 and the same
// torch expression in float64 gives 7.371844096, which is our value to every
// digit. So ours is the more accurate of the two and the ~3e-6 disagreement is
// the reference's drift, not ours. It is ~4e-7 relative, far below anything
// that reaches a pixel. Tightening this test would mean reproducing float32
// accumulation error on purpose.
test "sdxl schedule: timesteps and sigmas match diffusers EulerDiscreteScheduler" {
    const a = testing.allocator;
    var sched = try buildSchedule(a, 8, .leading, STEPS_OFFSET, false);
    defer sched.deinit();

    // The timesteps are integers and must match EXACTLY — there is no
    // precision argument available for these.
    const want_t = [_]f32{ 876, 751, 626, 501, 376, 251, 126, 1 };
    try testing.expectEqualSlices(f32, &want_t, sched.timesteps);

    const want_sigma = [_]f64{ 7.371847, 4.116698, 2.510922, 1.623693, 1.076032, 0.698399, 0.402218, 0.041314, 0.0 };
    try testing.expectEqual(want_sigma.len, sched.sigmas.len);
    for (want_sigma, sched.sigmas) |w, got| {
        try testing.expectApproxEqAbs(w, got, 1e-5);
    }

    // init_noise_sigma is sqrt(max_sigma^2 + 1), NOT max_sigma — the Euler
    // formulation keeps the latent in a sqrt(sigma^2+1)-normalised space.
    try testing.expectApproxEqAbs(@as(f64, 7.4393630027771), sched.init_noise_sigma, 1e-5);
    // Pin the relationship itself, which no precision argument touches.
    try testing.expectApproxEqAbs(
        @sqrt(sched.sigmas[0] * sched.sigmas[0] + 1.0),
        sched.init_noise_sigma,
        1e-12,
    );

    // The terminal zero is what lands the last step on a clean latent.
    try testing.expectEqual(@as(f64, 0.0), sched.sigmas[sched.sigmas.len - 1]);
}

test "sdxl schedule: steps_offset moves the sigma, not just the label" {
    const a = testing.allocator;
    var with = try buildSchedule(a, 8, .leading, 1, false);
    defer with.deinit();
    var without = try buildSchedule(a, 8, .leading, 0, false);
    defer without.deinit();

    // The offset shifts the reported timestep by exactly 1 ...
    for (with.timesteps, without.timesteps) |w, o| {
        try testing.expectApproxEqAbs(w - o, 1.0, 1e-6);
    }
    // ... and it also moves the sigma each step runs at. This is the half that
    // is easy to drop: applying the offset only to the label leaves the model
    // denoising at a sigma it was not told about.
    var any_differs = false;
    for (with.sigmas, without.sigmas) |w, o| {
        if (@abs(w - o) > 1e-9) any_differs = true;
    }
    try testing.expect(any_differs);
}

test "sdxl schedule: sigmas descend monotonically to zero at every step count" {
    const a = testing.allocator;
    for ([_]usize{ 1, 2, 4, 8, 20, 30, 50 }) |steps| {
        var sched = try buildSchedule(a, steps, .leading, STEPS_OFFSET, false);
        defer sched.deinit();
        try testing.expectEqual(steps, sched.timesteps.len);
        try testing.expectEqual(steps + 1, sched.sigmas.len);
        for (1..sched.sigmas.len) |i| {
            // Strictly decreasing: a flat pair is a wasted step, an increasing
            // pair runs the Euler update backwards.
            try testing.expect(sched.sigmas[i] < sched.sigmas[i - 1]);
        }
        try testing.expectEqual(@as(f64, 0.0), sched.sigmas[steps]);
        try testing.expect(sched.init_noise_sigma > 1.0);
    }
    try testing.expectError(error.ZeroSteps, buildSchedule(a, 0, .leading, STEPS_OFFSET, false));
}

// END-TO-END parity. The component fixtures pin each piece in isolation and
// cannot see a COMPOSITION error — swapped tower concat order, a dropped
// `scale_model_input`, guidance on the wrong branch, the Euler step taken on
// the scaled latent instead of the raw one. Each of those builds a plausible
// image out of correct parts.
//
// Injecting diffusers' own starting latent removes the only thing that
// legitimately differs between the two runs, so the final images must agree.
//
//   SDXL_CHECKPOINT_DIR=~/.mlx-serve/staging/sdxl-base-1.0 \
//   SDXL_PIPELINE_FIXTURE=~/.mlx-serve/staging/sdxl_pipeline_fixture.safetensors \
//     zig build test -Dtest-filter="sdxl pipeline parity"
test "sdxl pipeline parity: a full denoise matches diffusers" {
    const dir = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("SDXL_PIPELINE_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();

    const latents = fx.get("in.latents") orelse return error.MissingFixtureLatents;
    const ref = fx.get("out.image") orelse return error.MissingFixtureImage;
    const cfg = fx.get("cfg") orelse return error.MissingFixtureCfg;
    _ = mlx.mlx_array_eval(cfg);
    const cfgp = mlx.mlx_array_data_float32(cfg).?;

    // The generation parameters come FROM the fixture, so the two sides cannot
    // describe different runs.
    var opts = GenOpts{
        .width = @intFromFloat(cfgp[0]),
        .height = @intFromFloat(cfgp[1]),
        .steps = @intFromFloat(cfgp[2]),
        .guidance = cfgp[3],
        .init_latent = latents,
    };

    // `SDXL_COND_FIXTURE` additionally injects diffusers' own conditioning,
    // which isolates the DENOISE LOOP from the text towers. Run both ways: the
    // difference between the two numbers is exactly what the encoders
    // contribute, and neither number alone can tell those apart.
    var cond_fx: ?model_mod.Weights = null;
    defer if (cond_fx) |*c| c.deinit();
    if (std.c.getenv("SDXL_COND_FIXTURE")) |p| {
        cond_fx = try model_mod.loadWeightsSingleFile(a, std.mem.span(p));
        opts.cond_override = .{
            .ctx = cond_fx.?.get("ctx") orelse return error.MissingFixtureCtx,
            .pooled = cond_fx.?.get("pooled") orelse return error.MissingFixturePooled,
        };
    }

    var eng = try Engine.load(io, a, dir);
    defer eng.deinit();

    const img = try eng.generate("a photo of a cat", opts, null);
    defer _ = mlx.mlx_array_free(img);
    _ = mlx.mlx_array_eval(img);

    try testing.expectEqualSlices(c_int, mlx.getShape(ref), mlx.getShape(img));

    const g_arr = try flatF32Pipe(img, s);
    defer _ = mlx.mlx_array_free(g_arr);
    const r_arr = try flatF32Pipe(ref, s);
    defer _ = mlx.mlx_array_free(r_arr);
    const n = mlx.mlx_array_size(g_arr);
    const g = mlx.mlx_array_data_float32(g_arr).?;
    const r = mlx.mlx_array_data_float32(r_arr).?;

    var dot: f64 = 0;
    var ng: f64 = 0;
    var nr: f64 = 0;
    var abs_err: f64 = 0;
    for (0..n) |i| {
        const gv: f64 = g[i];
        const rv: f64 = r[i];
        try testing.expect(std.math.isFinite(gv));
        dot += gv * rv;
        ng += gv * gv;
        nr += rv * rv;
        abs_err += @abs(gv - rv);
    }
    const cos = dot / (@sqrt(ng) * @sqrt(nr));
    const rms_ratio = @sqrt(ng) / @sqrt(nr);
    const mae = abs_err / @as(f64, @floatFromInt(n));
    std.debug.print("[sdxl-parity] pipeline image: cos={d:.6} rms_ratio={d:.6} mae={d:.5}\n", .{ cos, rms_ratio, mae });

    // MEASURED on this checkpoint, 4 guided steps at 512x512:
    //
    //   fp16 towers + fp16 UNet            cos 0.9971  mae 0.027   (the default)
    //   float32 both (…_F32=1)             cos 0.9982  mae 0.020
    //   diffusers' own conditioning        cos 0.9972  mae 0.026
    //
    // The third row is what makes the first trustworthy: injecting diffusers'
    // conditioning (`SDXL_COND_FIXTURE`) barely moves the number, so the text
    // towers are NOT the limiting term and the denoise loop is faithful — with
    // reference conditioning the per-step latent stds track the reference to
    // ~0.1% at every step. Widening to float32 is what moves it, which is what
    // "the residual is precision" looks like from the outside.
    //
    // The bar is set for the fp16 default and deliberately not tightened to
    // the fp32 number, but it is far tighter than any COMPOSITION error
    // survives: a swapped tower concat, a dropped `scale_model_input`, the
    // Euler step taken on the scaled latent, or guidance on the wrong branch
    // all land near cos 0, not 0.99. The negative-prompt branch is the one
    // that does NOT — encoding "" instead of zeroing scores cos 0.975 here,
    // which is exactly why `GenOpts.negative_prompt` distinguishes absent from
    // empty and why this test leaves it at its default.
    try testing.expect(cos > 0.995);
    try testing.expect(rms_ratio > 0.99 and rms_ratio < 1.01);
    try testing.expect(mae < 0.04);
}

// The conditioning path — our tokenizer into both towers into one 2048-wide
// stream — is not covered by the component fixtures: the CLIP parity test
// drives FIXED ids, so it cannot see a tokenizer disagreement, and nothing
// pins the concat ORDER (768 then 1280) or which position the pooled vector is
// read from. All three produce a running pipeline.
//
//   SDXL_CHECKPOINT_DIR=~/.mlx-serve/staging/sdxl-base-1.0 \
//   SDXL_COND_FIXTURE=~/.mlx-serve/staging/sdxl_cond_fixture.safetensors \
//     zig build test -Dtest-filter="sdxl conditioning parity"
test "sdxl conditioning parity: both towers assemble as diffusers assembles them" {
    const dir = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("SDXL_COND_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const ref_ctx = fx.get("ctx") orelse return error.MissingFixtureCtx;
    const ref_pooled = fx.get("pooled") orelse return error.MissingFixturePooled;

    var eng = try Engine.load(io, a, dir);
    defer eng.deinit();

    var cond = try eng.encodePrompt("a photo of a cat");
    defer cond.deinit();
    _ = mlx.mlx_array_eval(cond.ctx);
    _ = mlx.mlx_array_eval(cond.pooled);

    try testing.expectEqualSlices(c_int, mlx.getShape(ref_ctx), mlx.getShape(cond.ctx));
    try testing.expectEqualSlices(c_int, mlx.getShape(ref_pooled), mlx.getShape(cond.pooled));
    try expectCloseCond("cond", "ctx", cond.ctx, ref_ctx, s);
    try expectCloseCond("cond", "pooled", cond.pooled, ref_pooled, s);
}

fn expectCloseCond(who: []const u8, what: []const u8, got: mlx.mlx_array, ref: mlx.mlx_array, s: S) !void {
    const g_arr = try flatF32Pipe(got, s);
    defer _ = mlx.mlx_array_free(g_arr);
    const r_arr = try flatF32Pipe(ref, s);
    defer _ = mlx.mlx_array_free(r_arr);
    const n = mlx.mlx_array_size(g_arr);
    const g = mlx.mlx_array_data_float32(g_arr).?;
    const r = mlx.mlx_array_data_float32(r_arr).?;
    var dot: f64 = 0;
    var ng: f64 = 0;
    var nr: f64 = 0;
    for (0..n) |i| {
        try testing.expect(std.math.isFinite(@as(f64, g[i])));
        dot += @as(f64, g[i]) * @as(f64, r[i]);
        ng += @as(f64, g[i]) * @as(f64, g[i]);
        nr += @as(f64, r[i]) * @as(f64, r[i]);
    }
    const cos = dot / (@sqrt(ng) * @sqrt(nr));
    const rms_ratio = @sqrt(ng) / @sqrt(nr);
    std.debug.print("[sdxl-parity] {s} {s}: cos={d:.6} rms_ratio={d:.6}\n", .{ who, what, cos, rms_ratio });
    try testing.expect(cos > 0.999);
    try testing.expect(rms_ratio > 0.99 and rms_ratio < 1.01);
}

// Regression guard for the bug the end-to-end parity actually caught: an
// ABSENT negative prompt zeroes the unconditional branch, an EMPTY one encodes
// the empty string. Both run, both produce an image, and the difference is
// worth cos 0.975 vs 0.997 against diffusers — invisible to every component
// test, because no component is wrong.
//
//   SDXL_CHECKPOINT_DIR=~/.mlx-serve/staging/sdxl-base-1.0 \
//     zig build test -Dtest-filter="sdxl negative prompt"
test "sdxl negative prompt: absent zeroes the branch, empty encodes it" {
    const dir = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var eng = try Engine.load(io, a, dir);
    defer eng.deinit();
    try testing.expect(eng.force_zeros_for_empty_prompt);

    var cond = try eng.encodePrompt("a photo of a cat");
    defer cond.deinit();

    // Absent -> exactly zero.
    var absent = try eng.encodeNegative(null, &cond);
    defer absent.deinit();
    try testing.expectApproxEqAbs(@as(f64, 0.0), try stdOf(absent.ctx, s), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try stdOf(absent.pooled, s), 1e-12);

    // Explicitly empty -> the encoding of "", which is NOT zero: the empty
    // string still carries BOS, EOS and 75 pads through both towers.
    var empty = try eng.encodeNegative("", &cond);
    defer empty.deinit();
    try testing.expect(try stdOf(empty.ctx, s) > 0.01);
    try testing.expect(try stdOf(empty.pooled, s) > 0.01);
}

fn flatF32Pipe(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var f = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, x, .float32, s));
    const cnt: c_int = @intCast(mlx.mlx_array_size(x));
    const shape = [_]c_int{cnt};
    var flat = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&flat, f, &shape, 1, s));
    _ = mlx.mlx_array_free(f);
    _ = mlx.mlx_array_eval(flat);
    return flat;
}

test "sdxl pipeline: cfg mix is a partition of unity" {
    // uncond*(1-w) + cond*w — the coefficients must sum to 1 or the epsilon
    // changes magnitude with the guidance scale, which reads as a contrast bug.
    for ([_]f64{ 1.0, 5.0, 7.5, 12.0 }) |w| {
        const m = sdxl.cfgMix(w);
        try testing.expectApproxEqAbs(@as(f64, 1.0), m.uncond + m.cond, 1e-12);
    }
    // At scale 1 the conditional branch stands alone, which is why the
    // pipeline skips the second forward there.
    const one = sdxl.cfgMix(1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), one.uncond, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), one.cond, 1e-12);
}

test "sdxl turbo: the schedule comes from the checkpoint, not from base's defaults" {
    const a = testing.allocator;

    // stabilityai/sdxl-turbo's actual scheduler_config.json.
    const turbo =
        \\{"_class_name":"EulerAncestralDiscreteScheduler","_diffusers_version":"0.24.0.dev0",
        \\"beta_end":0.012,"beta_schedule":"scaled_linear","beta_start":0.00085,
        \\"interpolation_type":"linear","num_train_timesteps":1000,"prediction_type":"epsilon",
        \\"steps_offset":1,"timestep_spacing":"trailing"}
    ;
    const t = parseSchedulerConfig(a, turbo);
    try testing.expectEqual(sdxl.TimestepSpacing.trailing, t.spacing);
    // Ancestral, from the DECLARED sampler class — Turbo re-noises each step.
    try testing.expect(t.ancestral);
    try testing.expectEqual(sdxl.PredictionType.epsilon, t.prediction);
    // `steps_offset: 1` is present and must be IGNORED: diffusers applies it
    // only on `leading`, and shifting Turbo's 1-4 timesteps off 999/749/... is
    // exactly the kind of change that still renders.
    try testing.expectEqual(@as(usize, 0), t.effectiveOffset());
    // Distilled => guidance-free, which also skips the unconditional forward.
    try testing.expectApproxEqAbs(@as(f32, 1.0), t.default_guidance, 1e-6);

    // Base SDXL, for contrast: leading, offset APPLIED, guided, not ancestral.
    const base =
        \\{"_class_name":"EulerDiscreteScheduler","prediction_type":"epsilon",
        \\"steps_offset":1,"timestep_spacing":"leading"}
    ;
    const b = parseSchedulerConfig(a, base);
    try testing.expectEqual(sdxl.TimestepSpacing.leading, b.spacing);
    try testing.expectEqual(@as(usize, 1), b.effectiveOffset());
    try testing.expect(!b.ancestral);
    try testing.expect(b.default_guidance > 1.0);

    // A missing/garbage file falls back to BASE's values — a pack without the
    // file is far more likely to be a base mirror, and a distill run on base's
    // schedule looks obviously wrong rather than subtly so.
    const fallback = parseSchedulerConfig(a, "not json");
    try testing.expectEqual(sdxl.TimestepSpacing.leading, fallback.spacing);
    try testing.expectEqual(sdxl.PredictionType.epsilon, fallback.prediction);

    // An unknown prediction type keeps epsilon rather than guessing: the three
    // are different quantities and a wrong one is noise, not a worse image.
    const weird = parseSchedulerConfig(a, "{\"prediction_type\":\"flow_matching\"}");
    try testing.expectEqual(sdxl.PredictionType.epsilon, weird.prediction);
}

test "sdxl turbo: a trailing schedule needs no offset and starts at the top" {
    const a = testing.allocator;
    // Turbo's real shape: 4 steps, trailing, no offset.
    var sched = try buildSchedule(a, 4, .trailing, 0, false);
    defer sched.deinit();
    try testing.expectEqualSlices(f32, &[_]f32{ 999, 749, 499, 249 }, sched.timesteps);
    // 1 step is the headline case and must still reach the noisiest timestep.
    var one = try buildSchedule(a, 1, .trailing, 0, false);
    defer one.deinit();
    try testing.expectEqualSlices(f32, &[_]f32{999}, one.timesteps);
    try testing.expectEqual(@as(f64, 0.0), one.sigmas[1]);
}
