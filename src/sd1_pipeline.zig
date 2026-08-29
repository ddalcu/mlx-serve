//! Stable Diffusion 1.x AND 2.x txt2img/img2img pipeline — diffusers'
//! `StableDiffusionPipeline` (the same pipeline class both share; SD-Turbo is
//! an SD 2.1 distill, not an SD 1.5 one, and lands here too).
//!
//! Reuses SDXL's building blocks wholesale rather than forking them, because
//! SD 1.x/2.x IS the same `UNet2DConditionModel` + `AutoencoderKL` + CLIP
//! family at a different config, not a different architecture:
//!
//!   - `sdxl_unet.Unet`, generalized with `UnetConfig.has_micro_conditioning`
//!     (see that file's header) — this family's config declares no
//!     `addition_embed_type` (older SD 1.x configs omit the key; SD-Turbo's
//!     newer diffusers version writes it out as a literal JSON `null` —
//!     `parseConfig` treats both as "no micro-conditioning", NOT just the
//!     omitted-key shape), so the same forward runs without the pooled-
//!     text/crop-ids augmentation SDXL's `add_embedding` adds.
//!     `cross_attention_dim` (768 on SD 1.x, 1024 on SD 2.x/Turbo, 2048 on
//!     SDXL) is read from the checkpoint, never hardcoded, so the SAME
//!     attention code binds any of the three widths.
//!   - `sdxl_vae.Decoder`/`Encoder` — architecturally identical; only
//!     `scaling_factor` differs (0.18215 here vs SDXL's 0.13025), and that is
//!     already a config-read parameter on both sides, not a constant baked
//!     into the forward.
//!   - `sdxl_clip.TextTower`, loaded ONCE — `sdxl.CLIP_L_CONFIG` (768-wide,
//!     quick_gelu) for SD 1.x, `sdxl.CLIP_H_CONFIG` (1024-wide, gelu, 23
//!     layers) for SD 2.x/Turbo, chosen from the checkpoint's OWN
//!     `text_encoder/config.json` `hidden_size` — never assumed from the
//!     `StableDiffusionPipeline` class name, which both share.
//!     `encode(..., final_norm: true)`: this family reads
//!     `CLIPTextModel(ids)[0]` with no `clip_skip`, i.e. the LAST hidden
//!     state AFTER `final_layer_norm` — the opposite of SDXL's
//!     penultimate-pre-norm convention, which is exactly what `final_norm`
//!     was added to select.
//!   - `sdxl.zig`'s schedule math (`scaledLinearBetas`/`alphasCumprod`/
//!     `trainSigmas`/`derivativeCoeffs`/`scaleModelInput`/`cfgMix`/
//!     `ancestralSigmas`) and `sdxl_pipeline.buildSchedule` /
//!     `parseSchedulerConfig` — this family trains on the SAME
//!     `beta_start=0.00085, beta_end=0.012, scaled_linear` schedule SDXL
//!     does, and the scheduler config is read the same way SDXL's three
//!     variants (base/Turbo/Lightning) are told apart: `scheduler/
//!     scheduler_config.json`'s `timestep_spacing`/`_class_name`/
//!     `prediction_type` decide spacing/ancestral-renoise/prediction, never
//!     hardcoded per checkpoint. Only the BASE guidance default (7.5, not
//!     SDXL's 5.0) is this family's own constant — a `trailing`-spacing
//!     distill (SD-Turbo) still overrides it to 1.0, same as SDXL's Turbo.
//!
//! What's NOT reused, deliberately out of scope for this port: single-file
//! (Civitai/A1111 LDM) loading — `sdxl_single_file.zig` converts decode-only
//! and has no SD 1.x/2.x key map yet, so only diffusers-folder repos load
//! here — and LoRA attachment, even though `sdxl_unet.attachLora` would bind
//! this family's UNet for free (same module tree); `lora.Arch` needs a
//! `.sd1` entry before that path is safe to expose.
//!
//! ORACLE STATUS: a structural port over already-pinned pieces
//! (`sdxl_unet`/`sdxl_vae`/`sdxl_clip` each carry their own parity tests),
//! but the COMPOSITION here — that these pieces in this order reproduce
//! `StableDiffusionPipeline` — has NOT been checked against a live diffusers
//! run in this change (no local SD 1.5/2.1 checkpoint or Python environment
//! was available; the SD-Turbo configs above WERE fetched and verified
//! against the real `stabilityai/sd-turbo` repo, so the CONFIG shapes are
//! real, not guessed — the untested part is the numerical forward). Treat it
//! the way `sdxl_pipeline.zig`'s own header treats its composition claim,
//! minus the pin: run the `tests/dump_sdxl_*_fixture*.py` shape of script
//! against a real checkpoint before trusting output quality, and pin a
//! `sd1 pipeline parity`-style test here once that exists.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const sdxl = @import("sdxl.zig");
const clip = @import("sdxl_clip.zig");
const clip_tok = @import("sdxl_tokenizer.zig");
const unet_mod = @import("sdxl_unet.zig");
const vae_mod = @import("sdxl_vae.zig");
const nn = @import("sdxl_nn.zig");
const sdxl_pipeline = @import("sdxl_pipeline.zig");
const sse = @import("gen_sse.zig");
const model_mod = @import("model.zig");

const S = mlx.mlx_stream;

/// SD 1.x's VAE scaling factor — NOT SDXL's 0.13025 (see `sdxl.zig`'s own
/// warning on that constant). Only a fallback: real checkpoints declare
/// `scaling_factor` in `vae/config.json`, which `vae_mod.load` already reads
/// ahead of any default.
pub const VAE_SCALING_FACTOR: f32 = 0.18215;

/// SD 1.x/2.x's own base guidance default — 7.5, NOT SDXL's 5.0. Only used
/// when the scheduler config doesn't override it (a `trailing`-spacing
/// distill like SD-Turbo overrides to 1.0, mirroring SDXL Turbo).
pub const BASE_GUIDANCE_DEFAULT: f32 = 7.5;

pub const GenOpts = struct {
    width: u32 = 512,
    height: u32 = 512,
    steps: u32 = 30,
    /// `guidance_scale`. NULL means "whatever this checkpoint was trained/
    /// distilled for" (`Engine.sched_cfg.default_guidance`) — SD-Turbo wants
    /// ~1 (guidance-free), SD 1.5 base wants ~7.5. Mirrors
    /// `sdxl_pipeline.GenOpts.guidance` exactly.
    guidance: ?f32 = null,
    seed: u64 = 0,
    negative_prompt: ?[]const u8 = null,
    /// img2img: source pixels `[1,3,H,W]` f32 `[0,1]`. See `sdxl_pipeline`'s
    /// identical field for the mixing convention (`sigmas[start_step]`).
    init_image: ?mlx.mlx_array = null,
    start_step: u32 = 0,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: S,
    tok: clip_tok.ClipTokenizer,
    tower: clip.TextTower,
    unet: unet_mod.Unet,
    vae: vae_mod.Decoder,
    /// Null when the pack's `vae/` has no `encoder.*` tensors, or failed to
    /// load — img2img unavailable either way.
    vae_enc: ?vae_mod.Encoder,
    /// From `scheduler/scheduler_config.json` — what makes this SD 1.5 base
    /// vs SD-Turbo (same shape SDXL base/Turbo/Lightning are told apart by).
    sched_cfg: sdxl_pipeline.SchedulerConfig,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const s = mlx.mlx_default_gpu_stream_new();
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = s;

        self.sched_cfg = readSchedulerConfig(io, allocator, model_dir);
        log.info("[sd1] schedule: spacing={s} prediction={s} ancestral={} guidance={d:.1}\n", .{
            @tagName(self.sched_cfg.spacing), @tagName(self.sched_cfg.prediction),
            self.sched_cfg.ancestral,         self.sched_cfg.default_guidance,
        });

        self.tok = try clip_tok.load(io, allocator, model_dir, "tokenizer");
        errdefer self.tok.deinit();

        // Choose the tower config from the checkpoint's OWN declared width —
        // SD 1.x (768, CLIP-L) and SD 2.x/Turbo (1024, OpenCLIP-H) share the
        // same `StableDiffusionPipeline` class, so the class alone cannot
        // tell them apart. Any other width is an architecture this port has
        // not built and is refused by name rather than guessed at.
        const hidden = readTextEncoderHidden(io, allocator, model_dir);
        const tower_cfg = if (hidden) |h| blk: {
            if (h == sdxl.CLIP_L_CONFIG.hidden) break :blk sdxl.CLIP_L_CONFIG;
            if (h == sdxl.CLIP_H_CONFIG.hidden) break :blk sdxl.CLIP_H_CONFIG;
            log.err("[sd1] {s}/text_encoder is {d}-wide — neither CLIP-L's {d} (SD 1.x) nor OpenCLIP-H's {d} (SD 2.x/Turbo)\n", .{ model_dir, h, sdxl.CLIP_L_CONFIG.hidden, sdxl.CLIP_H_CONFIG.hidden });
            return error.UnsupportedTextEncoderWidth;
        } else sdxl.CLIP_L_CONFIG; // no config.json read: trust the weights, same as before this check existed

        self.tower = try clip.loadTower(io, allocator, s, model_dir, "text_encoder", tower_cfg, clip.towerDtype());
        errdefer self.tower.deinit();

        const unet_dtype: mlx.mlx_dtype = if (std.c.getenv("SDXL_UNET_F32") != null) .float32 else .float16;
        self.unet = try unet_mod.load(io, allocator, s, model_dir, unet_dtype);
        errdefer self.unet.deinit();
        if (self.unet.cfg.has_micro_conditioning) {
            log.err("[sd1] {s}/unet declares SDXL-style micro-conditioning (addition_embed_type) — not an SD 1.x/2.x checkpoint\n", .{model_dir});
            return error.NotAnSd1Checkpoint;
        }
        if (self.unet.cfg.cross_attention_dim != tower_cfg.hidden) {
            log.err("[sd1] {s}/unet cross_attention_dim={d} does not match the loaded tower's {d}-wide output\n", .{ model_dir, self.unet.cfg.cross_attention_dim, tower_cfg.hidden });
            return error.MismatchedCrossAttentionDim;
        }

        self.vae = try vae_mod.load(io, allocator, s, model_dir, vae_mod.DEFAULT_DTYPE);
        errdefer self.vae.deinit();
        self.vae_enc = vae_mod.loadEncoder(io, allocator, s, model_dir, vae_mod.DEFAULT_DTYPE) catch |e| blk: {
            log.warn("[sd1] VAE encoder load failed ({}) — image-to-image disabled\n", .{e});
            break :blk null;
        };

        log.info("[sd1] loaded checkpoint: {s} (tower={d}-wide)\n", .{ model_dir, tower_cfg.hidden });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.tok.deinit();
        self.tower.deinit();
        self.unet.deinit();
        self.vae.deinit();
        if (self.vae_enc) |*e| e.deinit();
        self.allocator.destroy(self);
    }

    /// Encode one prompt through the single CLIP-L tower at SD 1.x's own
    /// convention (`final_norm: true` — see the file header).
    fn encodePrompt(self: *Engine, text: []const u8) !mlx.mlx_array {
        const a = self.allocator;
        var ids = try self.tok.encode(a, text);
        defer ids.deinit();
        var enc = try self.tower.encode(ids.ids, ids.eos_index, true);
        defer enc.deinit();
        return nn.dupA(enc.penultimate);
    }

    /// Generate one image. Returns NCHW `[1, 3, H, W]` float32 in `[0, 1]`.
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

        const scfg = self.sched_cfg;
        var sched = try sdxl_pipeline.buildSchedule(a, opts.steps, scfg.spacing, scfg.effectiveOffset(), scfg.zero_snr);
        defer sched.deinit();

        const ctx = try self.encodePrompt(prompt);
        defer _ = mlx.mlx_array_free(ctx);
        const uncond = try self.encodePrompt(opts.negative_prompt orelse "");
        defer _ = mlx.mlx_array_free(uncond);

        const start_step: u32 = @min(opts.start_step, @as(u32, @intCast(opts.steps)) - 1);

        // ── Starting latent. img2img: VAE-encode the source and mix with
        // fresh noise at `sigmas[start_step]`, same convention as
        // `sdxl_pipeline.zig`'s identical block.
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
            var key = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(key);
            try mlx.check(mlx.mlx_random_key(&key, opts.seed));
            const shape = [_]c_int{ 1, @intCast(sdxl.LATENT_CHANNELS), @intCast(dims.h), @intCast(dims.w) };
            var drawn = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&drawn, &shape, 4, .float32, 0.0, 1.0, key, s));
            defer _ = mlx.mlx_array_free(drawn);
            const scale = mlx.mlx_array_new_float(@floatCast(sched.init_noise_sigma));
            defer _ = mlx.mlx_array_free(scale);
            break :blk try nn.mulA(drawn, scale, s);
        };
        defer _ = mlx.mlx_array_free(latent);

        // A request that says nothing takes the CHECKPOINT's own default, so
        // a Turbo pack runs guidance-free without the caller having to know.
        const guidance = opts.guidance orelse scfg.default_guidance;
        const do_cfg = guidance > 1.0;
        const mix = sdxl.cfgMix(guidance);

        const run_steps = opts.steps - start_step;
        for (start_step..opts.steps) |i| {
            if (progress) |p| if (p.cancelled()) return error.Cancelled;

            const sigma = sched.sigmas[i];
            const sigma_next = sched.sigmas[i + 1];

            const scaled = blk: {
                const c = mlx.mlx_array_new_float(@floatCast(sdxl.scaleModelInput(sigma)));
                defer _ = mlx.mlx_array_free(c);
                break :blk try nn.mulA(latent, c, s);
            };
            defer _ = mlx.mlx_array_free(scaled);

            // No text_embeds/time_ids — this UNet's `has_micro_conditioning`
            // is false, so `forward` takes the plain time-embedding path.
            const eps_cond = try self.unet.forward(scaled, sched.timesteps[i], ctx, null, null);
            defer _ = mlx.mlx_array_free(eps_cond);

            const eps = if (do_cfg) blk: {
                const eps_un = try self.unet.forward(scaled, sched.timesteps[i], uncond, null, null);
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

            // Euler on the raw latent, not the scaled one handed to the UNet.
            // `derivativeCoeffs` reads the scheduler's OWN prediction type —
            // epsilon on every checkpoint verified so far, but read rather
            // than hardcoded the way SDXL's v-prediction finetunes need it.
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

            // Ancestral (Turbo-style): the deterministic part only reached
            // `sigma_down`; fresh noise at `sigma_up` makes up the rest. Zero
            // on the final step, so a finished image is never re-noised.
            if (scfg.ancestral and anc.sigma_up > 0.0) {
                var noise_key = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(noise_key);
                try mlx.check(mlx.mlx_random_key(&noise_key, opts.seed +% i +% 1));
                const shape = mlx.getShape(latent);
                var re_noise = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(re_noise);
                try mlx.check(mlx.mlx_random_normal(&re_noise, shape.ptr, shape.len, .float32, 0.0, 1.0, noise_key, s));
                const su = mlx.mlx_array_new_float(@floatCast(anc.sigma_up));
                defer _ = mlx.mlx_array_free(su);
                const scaled_noise = try nn.mulA(re_noise, su, s);
                defer _ = mlx.mlx_array_free(scaled_noise);
                nn.replace(&latent, try nn.addA(latent, scaled_noise, s));
            }
            _ = mlx.mlx_array_eval(latent);

            _ = mlx.mlx_clear_cache();
            if (progress) |p| p.emit("Generating", @intCast(i + 1 - start_step), run_steps);
        }
        if (progress) |p| p.emit("Decoding image", opts.steps, opts.steps);

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
};

/// Read `<model_dir>/scheduler/scheduler_config.json` via
/// `sdxl_pipeline.parseSchedulerConfig` (spacing/prediction/ancestral/
/// zero_snr — the same shape that tells SDXL base/Turbo/Lightning apart),
/// then override its 5.0-default with THIS family's own 7.5 base default —
/// `parseSchedulerConfig` only ever overrides `default_guidance` DOWN (to
/// 1.0, for a `trailing`-spacing distill), so a non-distilled config still
/// carries SDXL's constant unless corrected here.
fn readSchedulerConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) sdxl_pipeline.SchedulerConfig {
    var cfg = sdxl_pipeline.SchedulerConfig{};
    const path = std.fmt.allocPrint(allocator, "{s}/scheduler/scheduler_config.json", .{model_dir}) catch return withBaseGuidance(cfg);
    defer allocator.free(path);
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return withBaseGuidance(cfg);
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return withBaseGuidance(cfg);
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = rs.interface.allocRemaining(allocator, .limited(1 << 20)) catch return withBaseGuidance(cfg);
    defer allocator.free(content);
    cfg = sdxl_pipeline.parseSchedulerConfig(allocator, content);
    return withBaseGuidance(cfg);
}

fn withBaseGuidance(cfg: sdxl_pipeline.SchedulerConfig) sdxl_pipeline.SchedulerConfig {
    var c = cfg;
    if (c.spacing != .trailing) c.default_guidance = BASE_GUIDANCE_DEFAULT;
    return c;
}

/// Read `<model_dir>/text_encoder/config.json`'s `hidden_size`, or null on
/// any read/parse error (a checkpoint that omits it is trusted rather than
/// refused — the field is a SD-2.x-detection heuristic, not a requirement).
fn readTextEncoderHidden(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?u32 {
    const path = std.fmt.allocPrint(allocator, "{s}/text_encoder/config.json", .{model_dir}) catch return null;
    defer allocator.free(path);
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return null;
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = rs.interface.allocRemaining(allocator, .limited(1 << 20)) catch return null;
    defer allocator.free(content);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const v = parsed.value.object.get("hidden_size") orelse return null;
    return switch (v) {
        .integer => @intCast(v.integer),
        else => null,
    };
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "sd1: VAE_SCALING_FACTOR is SD 1.x's, not SDXL's" {
    try testing.expectApproxEqAbs(@as(f32, 0.18215), VAE_SCALING_FACTOR, 1e-9);
    try testing.expect(VAE_SCALING_FACTOR != sdxl.VAE_SCALING_FACTOR);
}

test "sd1: withBaseGuidance corrects the default for a non-distilled config" {
    // `parseSchedulerConfig`'s own zero-value struct carries SDXL's 5.0 —
    // this family's base checkpoints want 7.5, and only a `trailing`-spacing
    // distill should keep the lower value `parseSchedulerConfig` derives.
    const base = withBaseGuidance(.{ .spacing = .leading });
    try testing.expectApproxEqAbs(@as(f32, 7.5), base.default_guidance, 1e-6);
    const distilled = withBaseGuidance(.{ .spacing = .trailing, .default_guidance = 1.0 });
    try testing.expectApproxEqAbs(@as(f32, 1.0), distilled.default_guidance, 1e-6);
}

test "sd1: SD-Turbo's real scheduler_config.json parses trailing + guidance-free" {
    // stabilityai/sd-turbo/scheduler/scheduler_config.json, real bytes
    // (fetched and verified against the live repo).
    const a = testing.allocator;
    const json =
        \\{"_class_name":"EulerDiscreteScheduler","_diffusers_version":"0.24.0.dev0",
        \\"beta_end":0.012,"beta_schedule":"scaled_linear","beta_start":0.00085,
        \\"clip_sample":false,"interpolation_type":"linear","num_train_timesteps":1000,
        \\"prediction_type":"epsilon","sample_max_value":1.0,"set_alpha_to_one":false,
        \\"sigma_max":null,"sigma_min":null,"skip_prk_steps":true,"steps_offset":1,
        \\"timestep_spacing":"trailing","timestep_type":"discrete","trained_betas":null,
        \\"use_karras_sigmas":false}
    ;
    var cfg = sdxl_pipeline.parseSchedulerConfig(a, json);
    cfg = withBaseGuidance(cfg);
    try testing.expectEqual(sdxl.TimestepSpacing.trailing, cfg.spacing);
    try testing.expectEqual(sdxl.PredictionType.epsilon, cfg.prediction);
    try testing.expect(!cfg.ancestral); // plain EulerDiscreteScheduler, not Ancestral
    try testing.expect(!cfg.zero_snr);
    try testing.expectApproxEqAbs(@as(f32, 1.0), cfg.default_guidance, 1e-6);
    // trailing never carries the offset (SDXL's own `effectiveOffset` rule).
    try testing.expectEqual(@as(usize, 0), cfg.effectiveOffset());
}

test "sd1: SD-Turbo's real text_encoder/config.json selects CLIP_H_CONFIG" {
    // stabilityai/sd-turbo/text_encoder/config.json's hidden_size (1024,
    // real bytes fetched and verified against the live repo) must resolve to
    // CLIP_H_CONFIG, never CLIP_L_CONFIG (768) or a refusal.
    try testing.expectEqual(@as(u32, 1024), sdxl.CLIP_H_CONFIG.hidden);
    try testing.expectEqual(@as(u32, 23), sdxl.CLIP_H_CONFIG.layers);
    try testing.expectEqual(@as(u32, 16), sdxl.CLIP_H_CONFIG.heads);
    try testing.expect(sdxl.CLIP_H_CONFIG.activation == .gelu);
    try testing.expect(sdxl.CLIP_H_CONFIG.hidden != sdxl.CLIP_L_CONFIG.hidden);
}

test "sd1: readTextEncoderHidden reads hidden_size, tolerates absence" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    // A directory that doesn't exist: null, not an error.
    try testing.expectEqual(@as(?u32, null), readTextEncoderHidden(io, a, "/nonexistent-sd1-test-dir"));
}
