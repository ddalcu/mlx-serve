//! Stable Diffusion 3.5 — the pipeline: three text encoders, an MMDiT, a
//! 16-channel VAE, and a flow-matching Euler loop.
//!
//! One engine serves all three checkpoints. **Large**, **Large-Turbo** and
//! **Medium** declare the same `_class_name`, the same VAE, the same towers and
//! the same scheduler; what differs is the transformer's own config (Medium is
//! MMDiT-X at 24 layers) and, for Turbo, how many steps at what guidance. Both
//! differences are READ — `sd3.parseMmditConfig` for the first, the checkpoint's
//! identity for the second — never branched on a name.
//!
//! THE THINGS THIS FILE OWNS THAT ARE EASY TO GET WRONG:
//!
//!   * **The prompt is THREE encoders stitched into one sequence.** CLIP-L's and
//!     CLIP-G's penultimate hidden states concatenate on the CHANNEL axis to
//!     2048, are zero-padded out to T5's 4096, and are then concatenated with
//!     T5's 256 tokens on the SEQUENCE axis — CLIP FIRST. The pooled vector is a
//!     separate object: CLIP-L's and CLIP-G's PROJECTED pooled outputs
//!     concatenated to 2048, which conditions the timestep embedding rather than
//!     entering attention. Two streams, two rules; swapping either axis produces
//!     a right-shaped tensor and a wrong image.
//!
//!   * **BOTH CLIP towers are `CLIPTextModelWithProjection` here.** On SDXL only
//!     bigG carries a `text_projection`, and `sdxl_clip` already treats it as
//!     optional — but on SD 3.5 a missing projection on EITHER tower is a
//!     checkpoint we cannot pool, so it is refused at load rather than silently
//!     pooling half the vector.
//!
//!   * **There is no `force_zeros_for_empty_prompt` here.** SDXL distinguishes
//!     an ABSENT negative prompt (a zeroed unconditional branch) from an EMPTY
//!     one (the empty string, encoded), and that distinction is worth cos 0.997
//!     vs 0.975 against diffusers. `StableDiffusion3Pipeline` has no such flag:
//!     it encodes `""` for an absent negative prompt. Carrying SDXL's rule
//!     across would zero a branch diffusers encodes.
//!
//!   * **CFG is ONE batch-2 forward, not two.** The MMDiT contract takes B of 1
//!     or 2 and the uncond row goes FIRST, matching diffusers'
//!     `cat([negative, positive])` — so the split after the forward has to read
//!     row 0 as unconditional. Reversed, guidance runs backwards and the image
//!     is an anti-prompt.
//!
//!   * **The latent is shifted as well as scaled** (`sd3.decodeScale`). SDXL's
//!     shift is implicitly zero, so a scale-only decode is a uniform colour cast
//!     — plausible output, systematically wrong.
//!
//! ORACLE STATUS. The pieces are pinned individually: `sd3.zig`'s schedule and
//! geometry by construction, and the MMDiT, T5 and VAE each against a dumped
//! diffusers/transformers fixture in their own files. What is NOT pinned here is
//! the COMPOSITION — that needs `tests/dump_sd3_pipeline_fixture.py` run against
//! a real checkpoint, and until that has been run no claim of end-to-end parity
//! belongs in this header. The same status `sd1_pipeline.zig` carries.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const sd3 = @import("sd3.zig");
const sdxl = @import("sdxl.zig");
const clip = @import("sdxl_clip.zig");
const clip_tok = @import("sdxl_tokenizer.zig");
const nn = @import("sdxl_nn.zig");
const t5_mod = @import("t5_encoder.zig");
const t5_tok_mod = @import("t5_tokenizer.zig");
const mmdit_mod = @import("sd3_mmdit.zig");
const vae_mod = @import("sd3_vae.zig");
const sse = @import("gen_sse.zig");

const S = mlx.mlx_stream;

/// T5 sequence length. diffusers' `max_sequence_length` default for SD 3.5;
/// the checkpoint accepts up to 512, but every extra token is a row through a
/// 38-layer joint attention on BOTH streams, so the default is what we serve.
pub const T5_MAX_TOKENS: usize = 256;

/// CLIP contributes exactly 77 tokens per tower — the towers' own
/// `max_position_embeddings`.
pub const CLIP_TOKENS: usize = clip_tok.MAX_TOKENS;

/// The joint sequence the MMDiT sees: CLIP's 77 then T5's 256.
pub const JOINT_TOKENS: usize = CLIP_TOKENS + T5_MAX_TOKENS;

pub const GenOpts = struct {
    width: u32 = 1024,
    height: u32 = 1024,
    /// NULL means "whatever this checkpoint wants" — 28 on Large, 4 on Turbo.
    steps: ?u32 = null,
    /// NULL means the checkpoint's own default (4.5 on Large, 1.0 on Turbo).
    /// At <= 1 the unconditional row is dropped and the forward runs at batch 1,
    /// which is half the work per step on top of Turbo's seven-times-fewer steps.
    guidance: ?f32 = null,
    seed: u64 = 0,
    /// Unlike SDXL, absent and empty are the SAME request here — see the header.
    negative_prompt: ?[]const u8 = null,
    /// img2img: NCHW `[1, 3, H, W]` float32 in [-1, 1]. Borrowed.
    init_image: ?mlx.mlx_array = null,
    /// Where on the ladder an img2img run starts. 0 is a full generation.
    start_step: u32 = 0,
};

/// The two conditioning objects, which travel together and are shaped
/// differently on purpose.
pub const Conditioning = struct {
    /// `[1, JOINT_TOKENS, 4096]` — what enters joint attention.
    ctx: mlx.mlx_array,
    /// `[1, 2048]` — what conditions the timestep embedding.
    pooled: mlx.mlx_array,

    pub fn deinit(self: *Conditioning) void {
        _ = mlx.mlx_array_free(self.ctx);
        _ = mlx.mlx_array_free(self.pooled);
    }
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: S,
    mmdit_cfg: sd3.MmditConfig,
    sched_cfg: sd3.SchedulerConfig,
    vae_cfg: sd3.VaeConfig,

    tok_l: clip_tok.ClipTokenizer,
    tok_g: clip_tok.ClipTokenizer,
    tok_t5: t5_tok_mod.T5Tokenizer,
    tower_l: clip.TextTower,
    tower_g: clip.TextTower,
    t5: t5_mod.T5Encoder,
    mmdit: *mmdit_mod.Mmdit,
    vae: vae_mod.Vae,
    vae_enc: ?vae_mod.Encoder,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const s = mlx.mlx_default_gpu_stream_new();
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = s;
        self.vae_cfg = .{};

        // The transformer's own config is the ONLY thing that tells Large from
        // Medium — same `_class_name`, same everything else.
        const cfg_path = try std.fmt.allocPrint(allocator, "{s}/transformer/config.json", .{model_dir});
        defer allocator.free(cfg_path);
        const cfg_bytes = readFileAlloc(io, allocator, cfg_path) catch |e| {
            log.err("[sd3] transformer/config.json unreadable ({}) — not an SD 3.5 repo?\n", .{e});
            return error.MissingTransformerConfig;
        };
        defer allocator.free(cfg_bytes);
        self.mmdit_cfg = try sd3.parseMmditConfig(allocator, cfg_bytes);
        errdefer sd3.freeMmditConfig(allocator, self.mmdit_cfg);

        self.sched_cfg = readSchedulerConfig(io, allocator, model_dir, self.mmdit_cfg);
        log.info("[sd3] transformer: layers={d} heads={d}x{d} inner={d} pos_embed_max={d} dual_attn={d} qk_norm={}\n", .{
            self.mmdit_cfg.num_layers,          self.mmdit_cfg.num_attention_heads,
            self.mmdit_cfg.attention_head_dim,  self.mmdit_cfg.innerDim(),
            self.mmdit_cfg.pos_embed_max_size,  self.mmdit_cfg.dual_attention_layers.len,
            self.mmdit_cfg.qk_norm,
        });
        log.info("[sd3] schedule: shift={d:.2} default_steps={d} default_guidance={d:.1}\n", .{
            self.sched_cfg.shift, self.sched_cfg.default_steps, self.sched_cfg.default_guidance,
        });

        self.tok_l = try clip_tok.load(io, allocator, model_dir, "tokenizer");
        errdefer self.tok_l.deinit();
        self.tok_g = try clip_tok.load(io, allocator, model_dir, "tokenizer_2");
        errdefer self.tok_g.deinit();
        self.tok_t5 = try t5_tok_mod.load(io, allocator, model_dir, "tokenizer_3");
        errdefer self.tok_t5.deinit();

        self.tower_l = try clip.loadTower(io, allocator, s, model_dir, "text_encoder", sdxl.CLIP_L_CONFIG, clip.towerDtype());
        errdefer self.tower_l.deinit();
        self.tower_g = try clip.loadTower(io, allocator, s, model_dir, "text_encoder_2", sdxl.CLIP_BIG_G_CONFIG, clip.towerDtype());
        errdefer self.tower_g.deinit();
        // BOTH towers are `CLIPTextModelWithProjection` here. `sdxl_clip` treats
        // `text_projection` as optional because SDXL's CLIP-L genuinely has
        // none; on SD 3.5 a missing one means half the pooled vector would be
        // silently absent, so it is a load error rather than a quiet default.
        if (self.tower_l.text_projection == null or self.tower_g.text_projection == null) {
            log.err("[sd3] a text encoder is missing `text_projection` — SD 3.5 pools from BOTH towers\n", .{});
            return error.MissingTextProjection;
        }

        self.t5 = try t5_mod.load(io, allocator, s, model_dir, "text_encoder_3", clip.towerDtype());
        errdefer self.t5.deinit();

        self.mmdit = try mmdit_mod.load(io, allocator, s, model_dir, ditDtype());
        errdefer self.mmdit.deinit();

        // `force_upcast: true`, exactly as on SDXL — this VAE overflows fp16 on
        // real latents, which is a BLACK IMAGE, not a rounding error.
        self.vae = try vae_mod.Vae.load(io, allocator, s, model_dir, vae_mod.DEFAULT_DTYPE);
        errdefer self.vae.deinit();
        self.vae_enc = vae_mod.Encoder.load(io, allocator, s, model_dir, vae_mod.DEFAULT_DTYPE) catch |e| blk: {
            log.warn("[sd3] VAE encoder load failed ({}) — image-to-image disabled\n", .{e});
            break :blk null;
        };
        return self;
    }

    pub fn deinit(self: *Engine) void {
        if (self.vae_enc) |*e| e.deinit();
        self.vae.deinit();
        self.mmdit.deinit();
        self.t5.deinit();
        self.tower_g.deinit();
        self.tower_l.deinit();
        self.tok_t5.deinit();
        self.tok_g.deinit();
        self.tok_l.deinit();
        sd3.freeMmditConfig(self.allocator, self.mmdit_cfg);
        self.allocator.destroy(self);
    }

    /// Encode one prompt into the two conditioning objects.
    ///
    /// The CLIP half is exactly SDXL's — penultimate hidden states, pooled from
    /// the EOS row through `text_projection`. What is new is the third tower and
    /// the two different concatenation axes: channels for the CLIP pair and for
    /// the pooled pair, SEQUENCE for CLIP-against-T5. Both concats put CLIP
    /// first, matching diffusers.
    pub fn encodePrompt(self: *Engine, text: []const u8) !Conditioning {
        const a = self.allocator;
        const s = self.s;

        var enc_l_ids = try self.tok_l.encode(a, text);
        defer enc_l_ids.deinit();
        var enc_g_ids = try self.tok_g.encode(a, text);
        defer enc_g_ids.deinit();

        var enc_l = try self.tower_l.encode(enc_l_ids.ids, enc_l_ids.eos_index, false);
        defer enc_l.deinit();
        var enc_g = try self.tower_g.encode(enc_g_ids.ids, enc_g_ids.eos_index, false);
        defer enc_g.deinit();

        // Pooled: [1, 768] ++ [1, 1280] = [1, 2048], the width the MMDiT's
        // `pooled_projection` MLP expects.
        const pooled = try nn.concat(&[_]mlx.mlx_array{ enc_l.pooled.?, enc_g.pooled.? }, 1, s);
        errdefer _ = mlx.mlx_array_free(pooled);

        // CLIP context: [1, 77, 768] ++ [1, 77, 1280] = [1, 77, 2048], then
        // zero-padded on the CHANNEL axis out to T5's 4096. The pad is what lets
        // one sequence carry two different widths; it is not a reshape.
        const clip_ctx = try nn.concat(&[_]mlx.mlx_array{ enc_l.penultimate, enc_g.penultimate }, 2, s);
        defer _ = mlx.mlx_array_free(clip_ctx);
        const padded = try padChannels(clip_ctx, @intCast(self.mmdit_cfg.joint_attention_dim), s);
        defer _ = mlx.mlx_array_free(padded);

        // T5: a FIXED-length id buffer padded with pad id 0. SD 3 does not pass
        // an attention mask, so the pad rows are real inputs the model was
        // trained to see — truncating the buffer instead would change the
        // sequence length the joint attention runs at.
        const t5_ids = try self.tok_t5.encodePadded(a, text, T5_MAX_TOKENS);
        defer a.free(t5_ids);
        const t5_ctx = try self.t5.forward(t5_ids, s);
        defer _ = mlx.mlx_array_free(t5_ctx);

        // SEQUENCE axis, CLIP first.
        const ctx = try nn.concat(&[_]mlx.mlx_array{ padded, t5_ctx }, 1, s);
        return .{ .ctx = ctx, .pooled = pooled };
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

        // The VAE downsamples by 8 and the MMDiT patches by 2, so a canvas must
        // be a multiple of 16 for the patch grid to be exact. Refused rather
        // than rounded: a silently resized canvas is a request the caller did
        // not make.
        const stride = sd3.PIXELS_PER_LATENT * self.mmdit_cfg.patch_size;
        if (opts.width == 0 or opts.height == 0) return error.UnsupportedResolution;
        if (opts.width % stride != 0 or opts.height % stride != 0) return error.UnsupportedResolution;
        const lh = sd3.latentSideFor(opts.height);
        const lw = sd3.latentSideFor(opts.width);

        const steps = opts.steps orelse self.sched_cfg.default_steps;
        if (steps == 0) return error.ZeroSteps;

        const sigmas = try a.alloc(f64, steps + 1);
        defer a.free(sigmas);
        sd3.inferenceSigmas(steps, self.sched_cfg.shift, sigmas);

        var cond = try self.encodePrompt(prompt);
        defer cond.deinit();
        // Absent and empty are the SAME request here — SD 3 has no
        // `force_zeros_for_empty_prompt`. See the header.
        var uncond = try self.encodePrompt(opts.negative_prompt orelse "");
        defer uncond.deinit();

        const guidance = opts.guidance orelse self.sched_cfg.default_guidance;
        const do_cfg = guidance > 1.0;

        const start_step: u32 = @min(opts.start_step, steps - 1);
        var latent = try self.initialLatent(opts, lh, lw, sigmas[start_step]);
        defer _ = mlx.mlx_array_free(latent);

        // The two conditioning rows, stacked ONCE rather than per step:
        // uncond FIRST, matching diffusers' `cat([negative, positive])`.
        var batch_ctx: ?mlx.mlx_array = null;
        var batch_pooled: ?mlx.mlx_array = null;
        defer if (batch_ctx) |x| {
            _ = mlx.mlx_array_free(x);
        };
        defer if (batch_pooled) |x| {
            _ = mlx.mlx_array_free(x);
        };
        if (do_cfg) {
            batch_ctx = try nn.concat(&[_]mlx.mlx_array{ uncond.ctx, cond.ctx }, 0, s);
            batch_pooled = try nn.concat(&[_]mlx.mlx_array{ uncond.pooled, cond.pooled }, 0, s);
        }

        if (std.c.getenv("MLX_SERVE_SD3_TRACE") != null) {
            std.debug.print("[sd3-trace] shift={d:.3} sigma0={d:.6} steps={d} cfg={} joint_tokens={d}\n", .{
                self.sched_cfg.shift, sigmas[start_step], steps, do_cfg, JOINT_TOKENS,
            });
        }

        for (start_step..steps) |i| {
            // Poll for a departed client BEFORE spending a step on it: a guided
            // SD 3.5 Large step is a batch-2 forward through 38 joint blocks,
            // and without this a cancelled request burns the GPU to completion
            // with every other request queued behind it.
            if (progress) |p| if (p.cancelled()) return error.Cancelled;

            const sigma = sigmas[i];
            const timestep = sd3.timestepForSigma(sigma);

            // Flow matching hands the model the RAW latent — there is no
            // `scale_model_input` here, and dividing by sqrt(sigma^2+1) the way
            // the SDXL loop does would be a silent, plausible corruption.
            const v = if (do_cfg) blk: {
                const stacked = try nn.concat(&[_]mlx.mlx_array{ latent, latent }, 0, s);
                defer _ = mlx.mlx_array_free(stacked);
                const both = try self.mmdit.forward(stacked, batch_ctx.?, batch_pooled.?, timestep, s);
                defer _ = mlx.mlx_array_free(both);
                break :blk try mixGuidance(both, guidance, s);
            } else try self.mmdit.forward(latent, cond.ctx, cond.pooled, timestep, s);
            defer _ = mlx.mlx_array_free(v);

            // `x += (sigma_next - sigma) * v`. The whole integrator.
            const c = mlx.mlx_array_new_float(@floatCast(sd3.eulerStepCoeff(sigma, sigmas[i + 1])));
            defer _ = mlx.mlx_array_free(c);
            const dx = try nn.mulA(v, c, s);
            defer _ = mlx.mlx_array_free(dx);
            const next = try nn.addA(latent, dx, s);
            _ = mlx.mlx_array_free(latent);
            latent = next;

            if (progress) |p| p.emit("denoise", @intCast(i - start_step + 1), @intCast(steps - start_step));
        }

        return self.decodeLatent(latent);
    }

    /// Latent -> pixels in [0, 1]. The unscale is `z / scaling + shift`; SDXL's
    /// shift is implicitly zero, so a scale-only decode here is a uniform colour
    /// cast — plausible output, systematically wrong.
    fn decodeLatent(self: *Engine, latent: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const inv = mlx.mlx_array_new_float(@floatCast(1.0 / self.vae_cfg.scaling_factor));
        defer _ = mlx.mlx_array_free(inv);
        const scaled = try nn.mulA(latent, inv, s);
        defer _ = mlx.mlx_array_free(scaled);
        const shift = mlx.mlx_array_new_float(@floatCast(self.vae_cfg.shift_factor));
        defer _ = mlx.mlx_array_free(shift);
        const z = try nn.addA(scaled, shift, s);
        defer _ = mlx.mlx_array_free(z);

        const pix = try self.vae.decode(z, s);
        defer _ = mlx.mlx_array_free(pix);
        // The VAE emits [-1, 1]; the caller wants [0, 1].
        const half = mlx.mlx_array_new_float(0.5);
        defer _ = mlx.mlx_array_free(half);
        const halved = try nn.mulA(pix, half, s);
        defer _ = mlx.mlx_array_free(halved);
        return try nn.addA(halved, half, s);
    }

    /// The starting latent. img2img VAE-encodes the source and mixes fresh noise
    /// at `sigma` (flow matching's `(1 - sigma) * x0 + sigma * noise`, which is
    /// NOT the discrete family's `x0 + sigma * noise`); a plain generation is
    /// pure noise, which is that same formula at sigma 1.
    fn initialLatent(self: *Engine, opts: GenOpts, lh: u32, lw: u32, sigma: f64) !mlx.mlx_array {
        const s = self.s;
        const shape = [_]c_int{ 1, @intCast(self.mmdit_cfg.in_channels), @intCast(lh), @intCast(lw) };
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, opts.seed));
        var noise = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&noise, &shape, 4, .float32, 0.0, 1.0, key, s));
        defer _ = mlx.mlx_array_free(noise);

        const src = opts.init_image orelse return try nn.dupA(noise);
        const ve = if (self.vae_enc) |*e| e else return error.NoVaeEncoder;
        const mean = try ve.encodeMean(src, s);
        defer _ = mlx.mlx_array_free(mean);
        // Into the model's latent space before it is mixed with noise — the
        // scale/shift is the pipeline's, not the VAE's.
        const z0 = blk: {
            const sh = mlx.mlx_array_new_float(@floatCast(-self.vae_cfg.shift_factor));
            defer _ = mlx.mlx_array_free(sh);
            const shifted = try nn.addA(mean, sh, s);
            defer _ = mlx.mlx_array_free(shifted);
            const sc = mlx.mlx_array_new_float(@floatCast(self.vae_cfg.scaling_factor));
            defer _ = mlx.mlx_array_free(sc);
            break :blk try nn.mulA(shifted, sc, s);
        };
        defer _ = mlx.mlx_array_free(z0);

        const cs = mlx.mlx_array_new_float(@floatCast(sigma));
        defer _ = mlx.mlx_array_free(cs);
        const cz = mlx.mlx_array_new_float(@floatCast(1.0 - sigma));
        defer _ = mlx.mlx_array_free(cz);
        const a1 = try nn.mulA(noise, cs, s);
        defer _ = mlx.mlx_array_free(a1);
        const a2 = try nn.mulA(z0, cz, s);
        defer _ = mlx.mlx_array_free(a2);
        return try nn.addA(a1, a2, s);
    }
};

/// The DiT's serving dtype. bf16 rather than fp16: SD 3.5 was released in bf16
/// and stability's own card samples it there, and unlike the VAE the DiT has no
/// `force_upcast` to fall back on. `SD3_DIT_F32=1` widens it, which is how a
/// parity gap gets split into "wrong math" and "accumulated precision" — a logic
/// error does not improve when the dtype widens.
pub fn ditDtype() mlx.mlx_dtype {
    return if (std.c.getenv("SD3_DIT_F32") != null) .float32 else .bfloat16;
}

/// `[2, ...]` model output -> the guided `[1, ...]`.
///
/// `uncond + guidance * (cond - uncond)`, with row 0 unconditional. Reversed,
/// guidance runs backwards and the result is an anti-prompt: a confident,
/// well-formed image of the wrong thing.
fn mixGuidance(both: mlx.mlx_array, guidance: f32, s: S) !mlx.mlx_array {
    const un = try sliceBatch(both, 0, s);
    defer _ = mlx.mlx_array_free(un);
    const co = try sliceBatch(both, 1, s);
    defer _ = mlx.mlx_array_free(co);
    const diff = try nn.subA(co, un, s);
    defer _ = mlx.mlx_array_free(diff);
    const g = mlx.mlx_array_new_float(guidance);
    defer _ = mlx.mlx_array_free(g);
    const scaled = try nn.mulA(diff, g, s);
    defer _ = mlx.mlx_array_free(scaled);
    return try nn.addA(un, scaled, s);
}

/// Row `b` of a batched NCHW array, keeping the leading axis at length 1 so the
/// result is shaped like a batch-1 forward's output rather than needing a
/// reshape at every use.
fn sliceBatch(x: mlx.mlx_array, b: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var lo = [_]c_int{ 0, 0, 0, 0 };
    var hi = [_]c_int{ sh[0], sh[1], sh[2], sh[3] };
    const st = [_]c_int{ 1, 1, 1, 1 };
    lo[0] = b;
    hi[0] = b + 1;
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, 4, &hi, 4, &st, 4, s));
    return o;
}

/// Zero-pad `x`'s LAST axis out to `width`. Returns a fresh array even when no
/// padding is needed, so the caller's ownership rule is uniform.
fn padChannels(x: mlx.mlx_array, width: c_int, s: S) !mlx.mlx_array {
    const shape = mlx.getShape(x);
    const have = shape[shape.len - 1];
    if (have == width) return try nn.dupA(x);
    if (have > width) return error.ContextWiderThanJointDim;
    var zeros_shape: [3]c_int = .{ shape[0], shape[1], width - have };
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, &zeros_shape, 3, mlx.mlx_array_dtype(x), s));
    return try nn.concat(&[_]mlx.mlx_array{ x, zeros }, 2, s);
}

fn readFileAlloc(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    return try rs.interface.allocRemaining(allocator, .limited(1 << 20));
}

/// `scheduler/scheduler_config.json` plus the step/guidance defaults the
/// checkpoint does NOT declare.
///
/// Turbo is the reason this is not just a JSON read: `stable-diffusion-3.5-large`
/// and `-large-turbo` ship BYTE-IDENTICAL scheduler configs, so nothing in the
/// scheduler dir can tell them apart. What can is the transformer — no, it
/// cannot either; the two declare identical transformer configs as well. So the
/// distill is detected the only way it is visible on disk: the `_name_or_path`
/// or the model dir's own name. That is a NAME check, which this file otherwise
/// refuses to do, and it is confined to the one thing genuinely not encoded in
/// any config — with base's defaults as the safe fallback, since running Turbo
/// at 28 steps and guidance 4.5 is slow and over-saturated but not broken, while
/// running Large at 4 steps and guidance 1 is noise.
fn readSchedulerConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, mmdit_cfg: sd3.MmditConfig) sd3.SchedulerConfig {
    _ = mmdit_cfg;
    var cfg = sd3.SchedulerConfig{};
    const path = std.fmt.allocPrint(allocator, "{s}/scheduler/scheduler_config.json", .{model_dir}) catch return cfg;
    defer allocator.free(path);
    if (readFileAlloc(io, allocator, path)) |bytes| {
        defer allocator.free(bytes);
        if (std.json.parseFromSlice(std.json.Value, allocator, bytes, .{})) |parsed| {
            defer parsed.deinit();
            if (parsed.value == .object) {
                if (parsed.value.object.get("shift")) |v| switch (v) {
                    .float => |f| cfg.shift = f,
                    .integer => |n| cfg.shift = @floatFromInt(n),
                    else => {},
                };
            }
        } else |_| {}
    } else |_| {}
    if (dirNamesTurbo(model_dir)) {
        cfg.default_guidance = sd3.TURBO_CONFIG.default_guidance;
        cfg.default_steps = sd3.TURBO_CONFIG.default_steps;
        log.info("[sd3] turbo distill detected from the model dir — {d} steps, guidance {d:.1}\n", .{ cfg.default_steps, cfg.default_guidance });
    }
    return cfg;
}

/// The single name check this file allows, and only for the step/guidance
/// defaults a request can override anyway. See `readSchedulerConfig`.
pub fn dirNamesTurbo(model_dir: []const u8) bool {
    var i: usize = 0;
    while (i + 5 <= model_dir.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(model_dir[i .. i + 5], "turbo")) return true;
    }
    return false;
}

// ── Tests ───────────────────────────────────────────────────────────────

test "sd3 pipeline: the joint sequence is CLIP's 77 then T5's 256" {
    // The MMDiT sees one sequence carrying two different encoders. Getting the
    // length wrong is not a shape error — CLIP's rows would simply sit where T5's
    // belong, which the model reads as a different prompt.
    try testing.expectEqual(@as(usize, 77), CLIP_TOKENS);
    try testing.expectEqual(@as(usize, 256), T5_MAX_TOKENS);
    try testing.expectEqual(@as(usize, 333), JOINT_TOKENS);
}

test "sd3 pipeline: turbo defaults come from the dir name, and base is the safe fallback" {
    // `stable-diffusion-3.5-large` and `-large-turbo` ship byte-identical
    // scheduler AND transformer configs, so nothing on disk except the name
    // separates them. Base's defaults are the fallback on purpose: Turbo at 28
    // steps is slow and over-saturated, Large at 4 steps and guidance 1 is noise.
    try testing.expect(dirNamesTurbo("/models/stabilityai/stable-diffusion-3.5-large-turbo"));
    try testing.expect(dirNamesTurbo("/models/SD3.5-Large-TURBO"));
    try testing.expect(!dirNamesTurbo("/models/stabilityai/stable-diffusion-3.5-large"));
    try testing.expect(!dirNamesTurbo("/models/stabilityai/stable-diffusion-3.5-medium"));
    // Not a substring of a longer word we would be wrong about, but also not
    // anchored — a user's own dir name is not a contract.
    try testing.expect(!dirNamesTurbo("/models/turb"));

    const base = sd3.SchedulerConfig{};
    try testing.expectEqual(@as(u32, 28), base.default_steps);
    try testing.expectApproxEqAbs(@as(f32, 4.5), base.default_guidance, 1e-6);
}

test "sd3 pipeline: a canvas must be a multiple of patch*8, and it is REFUSED not rounded" {
    // The VAE downsamples by 8, the MMDiT patches by 2. A silently resized
    // canvas is a request the caller did not make — and on this family it would
    // also change the pos-embed crop, so the framing moves too.
    const cfg = sd3.MmditConfig{};
    const stride = sd3.PIXELS_PER_LATENT * cfg.patch_size;
    try testing.expectEqual(@as(u32, 16), stride);
    for ([_]u32{ 512, 1024, 1536, 768 }) |ok| try testing.expectEqual(@as(u32, 0), ok % stride);
    for ([_]u32{ 1000, 1023, 500 }) |bad| try testing.expect(bad % stride != 0);
}
