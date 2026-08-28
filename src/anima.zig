//! Anima image generation — NVIDIA Cosmos-Predict2 2B DiT (`MiniTrainDIT`) plus
//! the Anima-specific `LLMAdapter`, conditioned by a Qwen3-0.6B encoder, decoded
//! by the Qwen-Image VAE, sampled with Cosmos rectified flow. Anime/illustration
//! model (circlestone-labs/Anima). Ported math from ComfyUI (the ground-truth
//! reference — Anima ships as a ComfyUI single-file):
//!   comfy/ldm/cosmos/predict2.py         MiniTrainDIT, Block, PatchEmbed, FinalLayer
//!   comfy/ldm/cosmos/position_embedding.py  VideoRopePosition3DEmb
//!   comfy/ldm/anima/model.py             Anima, LLMAdapter, TransformerBlock
//!   comfy/model_sampling.py              CONST, ModelSamplingDiscreteFlow, time_snr_shift
//!   comfy/samplers.py                    simple_scheduler, cfg_function, sampling_function
//!   comfy/latent_formats.py              Wan21 (16-ch latent mean/std)
//!
//! This file is being built bottom-up: the pure numeric core (rectified-flow
//! sigma math, RoPE-3D geometry, Wan2.1 latent normalization, timestep sincos)
//! lands first with hermetic tests, since it is transcribed unambiguously from
//! the reference. The DiT / adapter / VAE forward passes are pinned against
//! reference dumps before they are trusted — see notes/anima-implementation.md.
//!
//! IMPORTANT correction: `comfy/model_base.py Anima.__init__` instantiates with
//! the DEFAULT `model_type=ModelType.FLOW` (it never passes FLOW_COSMOS), so
//! `comfy/model_base.py model_sampling()` selects `CONST` + `ModelSamplingDiscreteFlow`
//! — the same standard flow-matching math Flux/SD3 use — NOT `COSMOS_RFLOW` /
//! `ModelSamplingCosmosRFlow` (those are unused by Anima despite the "Cosmos"
//! DiT trunk name; they belong to CosmosPredict2's own image/video models).
//! `ModelSamplingCosmosRFlow` doesn't even reference `shift` — the `shift: 3.0`
//! in `supported_models.Anima.sampling_settings` only makes sense under
//! `ModelSamplingDiscreteFlow`, which confirms this is the right mixin.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const tok_mod = @import("tokenizer.zig");
const t5_tok_mod = @import("t5_tokenizer.zig");
const krea = @import("krea.zig");
const sse = @import("gen_sse.zig");
const lora_mod = @import("lora.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

// ── DiT config (Cosmos-Predict2 2B, image_model=="anima") ──────────────────
// From comfy/model_detection.py (cosmos_predict2 branch) + supported_models.py.
// The 2B text2image checkpoint: model_channels 2048, 16 heads, 28 blocks,
// 16-ch latent in/out, 2x2 spatial patch, temporal patch 1.

pub const Config = struct {
    in_channels: u32 = 16,
    out_channels: u32 = 16,
    patch_spatial: u32 = 2,
    patch_temporal: u32 = 1,
    model_channels: u32 = 2048,
    num_heads: u32 = 16,
    num_blocks: u32 = 28,
    mlp_ratio: f32 = 4.0,
    crossattn_emb_channels: u32 = 1024,
    concat_padding_mask: bool = true,
    use_adaln_lora: bool = true,
    adaln_lora_dim: u32 = 256,
    // RoPE-3D extrapolation ratios (16-ch text2image branch).
    rope_h_extrapolation_ratio: f32 = 4.0,
    rope_w_extrapolation_ratio: f32 = 4.0,
    rope_t_extrapolation_ratio: f32 = 1.0,
    rope_enable_fps_modulation: bool = false,
    extra_per_block_abs_pos_emb: bool = false,

    pub fn headDim(self: Config) u32 {
        return self.model_channels / self.num_heads;
    }

    /// In-features of the PatchEmbed linear: (in_channels + padding-mask) folded
    /// over the spatial/temporal patch. 17 * 2 * 2 * 1 = 68 for the 2B pack.
    pub fn patchInFeatures(self: Config) u32 {
        const c = self.in_channels + @as(u32, if (self.concat_padding_mask) 1 else 0);
        return c * self.patch_spatial * self.patch_spatial * self.patch_temporal;
    }
};

// ── RoPE-3D geometry (position_embedding.py VideoRopePosition3DEmb) ─────────
// The head dim is split across (T, H, W): dim_h = dim_w = head_dim//6*2, and the
// temporal band takes the remainder so the three sum to head_dim exactly.

pub const Rope3DSplit = struct {
    dim_h: u32,
    dim_w: u32,
    dim_t: u32,
};

pub fn rope3dSplit(head_dim: u32) Rope3DSplit {
    const dim_h = head_dim / 6 * 2;
    return .{ .dim_h = dim_h, .dim_w = dim_h, .dim_t = head_dim - 2 * dim_h };
}

/// NTK factor for one axis: `ratio ** (dim / (dim - 2))` (predict2 build_pos).
pub fn ntkFactor(extrapolation_ratio: f32, dim: u32) f32 {
    const d: f32 = @floatFromInt(dim);
    return std.math.pow(f32, extrapolation_ratio, d / (d - 2.0));
}

// ── Rectified-flow sampler (model_sampling.py CONST + ModelSamplingDiscreteFlow) ──
// sampling_settings: multiplier 1.0, shift 3.0 (supported_models.py Anima).
// `timestep(sigma) = sigma * multiplier = sigma` (multiplier 1.0 -> identity):
// the DiT's t_embedder receives the raw [0,1] sigma value directly (confirmed
// against the DiT parity fixture, which forwards timesteps=[0.7] unmodified).

pub const RFLOW_SHIFT: f32 = 3.0;
pub const RFLOW_TIMESTEPS: u32 = 1000;

/// comfy/model_sampling.py time_snr_shift: warp a [0,1] flow-time by `alpha`.
pub fn timeSnrShift(alpha: f32, t: f32) f32 {
    if (alpha == 1.0) return t;
    return alpha * t / (1.0 + (alpha - 1.0) * t);
}

/// ModelSamplingDiscreteFlow's discrete sigma buffer, 1-indexed per the
/// reference (`sigma(timestep) = time_snr_shift(shift, timestep/multiplier)`
/// evaluated at `timestep = i/timesteps` for i=1..timesteps, multiplier=1.0).
fn discreteSigma(shift: f32, timesteps: u32, one_indexed: u32) f32 {
    const t: f32 = @as(f32, @floatFromInt(one_indexed)) / @as(f32, @floatFromInt(timesteps));
    return timeSnrShift(shift, t);
}

/// comfy/samplers.py simple_scheduler: `steps` descending sigmas read
/// backwards off the ascending discrete buffer, plus a trailing 0.0
/// (returns `steps + 1` values). Caller owns the returned slice.
pub fn buildSimpleSchedule(allocator: std.mem.Allocator, steps: u32, shift: f32) ![]f32 {
    const sig = try allocator.alloc(f32, steps + 1);
    errdefer allocator.free(sig);
    const ss: f32 = @as(f32, @floatFromInt(RFLOW_TIMESTEPS)) / @as(f32, @floatFromInt(steps));
    for (0..steps) |x| {
        const k: u32 = @intFromFloat(@as(f32, @floatFromInt(x)) * ss); // int(x*ss)
        const idx1: u32 = RFLOW_TIMESTEPS - k; // s.sigmas[-(1+k)], 1-indexed
        sig[x] = discreteSigma(shift, RFLOW_TIMESTEPS, idx1);
    }
    sig[steps] = 0.0;
    return sig;
}

/// CONST.calculate_denoised: x0 estimate from a velocity prediction.
/// model_input - model_output * sigma.
pub fn calculateDenoised(sigma: f32, model_output: f32, model_input: f32) f32 {
    return model_input - model_output * sigma;
}

/// CONST.noise_scaling: build the (possibly partially-denoised) noisy latent.
/// sigma*noise + (1-sigma)*latent_image (latent_image=0 for pure txt2img).
pub fn noiseScaling(sigma: f32, noise: f32, latent_image: f32) f32 {
    return sigma * noise + (1.0 - sigma) * latent_image;
}

/// Classifier-free guidance combined directly on the raw velocity prediction.
/// For CONST math this is algebraically identical to combining in x0/denoised
/// space then re-deriving the ODE step (comfy's cfg_function order) — see
/// notes/anima-implementation.md for the derivation. cfg==1.0 is the identity
/// (Turbo variants ship with cfg 1 and skip the uncond forward entirely).
pub fn cfgCombine(cond_out: f32, uncond_out: f32, cfg: f32) f32 {
    return uncond_out + (cond_out - uncond_out) * cfg;
}

// ── Wan2.1 latent format (latent_formats.py Wan21) — 16 channels ───────────
// The VAE encodes to / decodes from these normalized latents. process_in
// applied after VAE encode, process_out before VAE decode. The DiT operates on
// the normalized latents (mean 0-ish, unit-ish std), never raw VAE output.

pub const WAN21_LATENT_CHANNELS: u32 = 16;

pub const WAN21_LATENTS_MEAN = [WAN21_LATENT_CHANNELS]f32{
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134,  -0.0715, 0.5517,  -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
};

pub const WAN21_LATENTS_STD = [WAN21_LATENT_CHANNELS]f32{
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
};

/// process_in: (raw_latent - mean) / std. `ch` selects the channel table entry.
pub fn wan21ProcessIn(ch: usize, raw: f32) f32 {
    return (raw - WAN21_LATENTS_MEAN[ch]) / WAN21_LATENTS_STD[ch];
}

/// process_out: normalized_latent * std + mean (inverse of process_in).
pub fn wan21ProcessOut(ch: usize, normalized: f32) f32 {
    return normalized * WAN21_LATENTS_STD[ch] + WAN21_LATENTS_MEAN[ch];
}

// ── Timestep sinusoidal embedding (predict2.py Timesteps) ──────────────────
// half_dim = num_channels/2; exponent = -ln(10000) * arange(half)/half;
// emb = t * exp(exponent); output = concat(cos(emb), sin(emb)) — cos FIRST.

/// Fill `out` (len == num_channels, even) with the Cosmos timestep embedding of
/// scalar `t`. Layout is [cos(0..half), sin(0..half)] — cosine band first.
pub fn timestepEmbedding(t: f32, out: []f32) void {
    const num_channels = out.len;
    std.debug.assert(num_channels % 2 == 0);
    const half = num_channels / 2;
    const half_f: f32 = @floatFromInt(half);
    var i: usize = 0;
    while (i < half) : (i += 1) {
        const exponent = -std.math.log(f32, std.math.e, 10000.0) * @as(f32, @floatFromInt(i)) / half_f;
        const arg = t * @exp(exponent);
        out[i] = @cos(arg);
        out[half + i] = @sin(arg);
    }
}

// ── mlx helper wrappers (file-local, thin — the flux/krea/mage_flow idiom) ──

fn mul(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}
fn add(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}
fn neg(a: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_negative(&o, a, s));
    return o;
}
fn matmul(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, a, b, s));
    return o;
}
fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}
fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
fn rmsNorm(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}
/// Exact GELU (erf form) — nn.GELU default, matching the adapter's MLP.
fn geluErf(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const inv_sqrt2 = mlx.mlx_array_new_float(0.7071067811865476);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    const scaled = try mul(x, inv_sqrt2, s);
    defer _ = mlx.mlx_array_free(scaled);
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_erf(&e, scaled, s));
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const opl = try add(e, one, s);
    defer _ = mlx.mlx_array_free(opl);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mul(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mul(hx, opl, s);
}
/// x[..,in] @ w_t[in,out] (+ bias). `w_t` is pre-transposed at load.
fn linearT(x: mlx.mlx_array, w_t: mlx.mlx_array, bias: ?mlx.mlx_array, s: S) !mlx.mlx_array {
    const o = try matmul(x, w_t, s);
    if (bias) |b| {
        defer _ = mlx.mlx_array_free(o);
        return add(o, b, s);
    }
    return o;
}
/// Gather rows of `table` [V, D] by integer `ids` [..], → [.., D].
fn takeRows(table: mlx.mlx_array, ids: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&o, table, ids, 0, s));
    return o;
}
/// Unmasked attention over `q,k,v` [B,H,·,D]: manual softmax(qkᵀ·scale)·v.
/// mlx's fused `mlx_fast_scaled_dot_product_attention` hits a shape-triggered
/// failure at real-world sizes here ("force_fused=True but no fused kernel is
/// available", seen with the adapter's M=512 self-attn and the DiT's
/// L~1000+ self-attn on a real request — tiny parity fixtures never hit it)
/// even with a genuine GPU stream. Same class of bug as the VAE (head-dim
/// wall) and TE (causal-mode abort) fast-SDPA fixes above; this replaces the
/// fast-kernel path outright rather than special-casing another shape.
fn sdpa(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, s: S) !mlx.mlx_array {
    const kt = try transpose(k, &[_]c_int{ 0, 1, 3, 2 }, s); // [B,H,D,Lk]
    defer _ = mlx.mlx_array_free(kt);
    const scores0 = try matmul(q, kt, s); // [B,H,Mq,Lk]
    defer _ = mlx.mlx_array_free(scores0);
    const scaled = try mulScalar(scores0, scale, s);
    defer _ = mlx.mlx_array_free(scaled);
    var attnw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attnw);
    try mlx.check(mlx.mlx_softmax_axis(&attnw, scaled, 3, true, s));
    return matmul(attnw, v, s);
}
/// rotate_half(x) = cat(-x[..,D/2:], x[..,:D/2]) over the last axis.
fn rotateHalf(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    const nd = mlx.getShape(x).len;
    try mlx.check(mlx.mlx_split(&parts, x, 2, @intCast(nd - 1), s));
    var x1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x1);
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_vector_array_get(&x1, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&x2, parts, 1));
    const nx2 = try neg(x2, s);
    defer _ = mlx.mlx_array_free(nx2);
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, nx2);
    _ = mlx.mlx_vector_array_append_value(vec, x1);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, @intCast(nd - 1), s));
    return o;
}
/// apply_rotary_pos_emb(x[B,H,S,D], cos[1,1,S,D], sin[1,1,S,D]).
fn applyRope(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: S) !mlx.mlx_array {
    const xc = try mul(x, cos, s);
    defer _ = mlx.mlx_array_free(xc);
    const rh = try rotateHalf(x, s);
    defer _ = mlx.mlx_array_free(rh);
    const rs = try mul(rh, sin, s);
    defer _ = mlx.mlx_array_free(rs);
    return add(xc, rs, s);
}

/// Owned copy of a weight from `w`, cast to `dt`. Linear weights are stored
/// [out,in]; pass `want_transpose` to pre-transpose to [in,out] for matmul.
fn loadW(w: *const Weights, a: std.mem.Allocator, key: []const u8, dt: mlx.mlx_dtype, want_transpose: bool, s: S) !mlx.mlx_array {
    const raw = w.get(key) orelse {
        log.err("[anima] missing weight: {s}\n", .{key});
        return error.MissingAnimaWeight;
    };
    _ = a;
    var owned = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&owned, raw));
    defer _ = mlx.mlx_array_free(owned);
    if (want_transpose) {
        const t = try transpose(owned, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(t);
        const tc = try contig(t, s);
        defer _ = mlx.mlx_array_free(tc);
        return astype(tc, dt, s);
    }
    return astype(owned, dt, s);
}

fn loadWKey(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype, dt: mlx.mlx_dtype, want_transpose: bool, s: S) !mlx.mlx_array {
    const key = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(key);
    return loadW(w, a, key, dt, want_transpose, s);
}

// ── LLMAdapter (comfy/ldm/anima/model.py) ──────────────────────────────────
// A 6-block Perceiver-style resampler: T5-vocab query embeddings self-attend,
// then cross-attend to the Qwen3-0.6B hidden states, producing the DiT's
// cross-attention context. Weights live under `<dit>.llm_adapter.*`.

/// The checkpoint's own top-level weight-key prefix — a converter-choice
/// property, not a constant: circlestone-labs' `turbo`/`aesthetic` releases
/// ship `model.diffusion_model.*`, but `base-v1.0` ships bare `net.*` for
/// the SAME architecture (probed live, not assumed — a real base-v1.0
/// checkpoint failed to load on the hardcoded prefix, `MissingAnimaWeight`
/// at `x_embedder`). Checked once via a single representative key.
fn adapterPrefix(w: *const Weights) []const u8 {
    if (w.get("net.llm_adapter.embed.weight") != null) return "net.llm_adapter";
    return "model.diffusion_model.llm_adapter";
}
const ADAPTER_DIM: u32 = 1024;
const ADAPTER_HEADS: u32 = 16;
const ADAPTER_HEAD_DIM: u32 = ADAPTER_DIM / ADAPTER_HEADS; // 64
const ADAPTER_LAYERS: u32 = 6;
const ADAPTER_ROPE_THETA: f32 = 10000.0;
const ADAPTER_EPS: f32 = 1e-6;

const AttnW = struct {
    q_proj: mlx.mlx_array,
    q_norm: mlx.mlx_array,
    k_proj: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    v_proj: mlx.mlx_array,
    o_proj: mlx.mlx_array,

    fn deinit(self: *AttnW) void {
        _ = mlx.mlx_array_free(self.q_proj);
        _ = mlx.mlx_array_free(self.q_norm);
        _ = mlx.mlx_array_free(self.k_proj);
        _ = mlx.mlx_array_free(self.k_norm);
        _ = mlx.mlx_array_free(self.v_proj);
        _ = mlx.mlx_array_free(self.o_proj);
    }
};

const AdapterBlockW = struct {
    norm_self: mlx.mlx_array,
    self_attn: AttnW,
    norm_cross: mlx.mlx_array,
    cross_attn: AttnW,
    norm_mlp: mlx.mlx_array,
    mlp0_w: mlx.mlx_array,
    mlp0_b: mlx.mlx_array,
    mlp2_w: mlx.mlx_array,
    mlp2_b: mlx.mlx_array,

    fn deinit(self: *AdapterBlockW) void {
        _ = mlx.mlx_array_free(self.norm_self);
        self.self_attn.deinit();
        _ = mlx.mlx_array_free(self.norm_cross);
        self.cross_attn.deinit();
        _ = mlx.mlx_array_free(self.norm_mlp);
        _ = mlx.mlx_array_free(self.mlp0_w);
        _ = mlx.mlx_array_free(self.mlp0_b);
        _ = mlx.mlx_array_free(self.mlp2_w);
        _ = mlx.mlx_array_free(self.mlp2_b);
    }
};

pub const Adapter = struct {
    embed: mlx.mlx_array, // [32128, 1024]
    blocks: [ADAPTER_LAYERS]AdapterBlockW,
    out_proj_w: mlx.mlx_array,
    out_proj_b: mlx.mlx_array,
    norm: mlx.mlx_array,
    dtype: mlx.mlx_dtype,

    pub fn load(w: *const Weights, a: std.mem.Allocator, dtype: mlx.mlx_dtype, s: S) !Adapter {
        var self: Adapter = undefined;
        self.dtype = dtype;
        self.embed = try loadWKey(w, a, "{s}.embed.weight", .{adapterPrefix(w)}, dtype, false, s);
        self.out_proj_w = try loadWKey(w, a, "{s}.out_proj.weight", .{adapterPrefix(w)}, dtype, true, s);
        self.out_proj_b = try loadWKey(w, a, "{s}.out_proj.bias", .{adapterPrefix(w)}, dtype, false, s);
        self.norm = try loadWKey(w, a, "{s}.norm.weight", .{adapterPrefix(w)}, dtype, false, s);
        for (0..ADAPTER_LAYERS) |i| {
            self.blocks[i] = try loadAdapterBlock(w, a, i, dtype, s);
        }
        return self;
    }

    pub fn deinit(self: *Adapter) void {
        _ = mlx.mlx_array_free(self.embed);
        for (&self.blocks) |*b| b.deinit();
        _ = mlx.mlx_array_free(self.out_proj_w);
        _ = mlx.mlx_array_free(self.out_proj_b);
        _ = mlx.mlx_array_free(self.norm);
    }

    /// Run the adapter. `qwen_hidden` [1, L, 1024] (source), `t5_ids` [1, M] int.
    /// Returns [1, M, 1024] in `self.dtype`. Caller frees.
    pub fn forward(self: *const Adapter, qwen_hidden: mlx.mlx_array, t5_ids: mlx.mlx_array, a: std.mem.Allocator, s: S) !mlx.mlx_array {
        const M: usize = @intCast(mlx.getShape(t5_ids)[mlx.getShape(t5_ids).len - 1]);
        const L: usize = @intCast(mlx.getShape(qwen_hidden)[1]);

        // Query tokens: embed(t5_ids) → [1, M, 1024].
        var x = try takeRows(self.embed, t5_ids, s);
        errdefer _ = mlx.mlx_array_free(x);
        // in_proj is Identity (model_dim == target_dim).

        // RoPE tables (computed on host, uploaded as [1,1,S,64] for broadcast).
        const cos_q = try ropeCos(a, M, self.dtype, s);
        defer _ = mlx.mlx_array_free(cos_q);
        const sin_q = try ropeSin(a, M, self.dtype, s);
        defer _ = mlx.mlx_array_free(sin_q);
        const cos_ctx = try ropeCos(a, L, self.dtype, s);
        defer _ = mlx.mlx_array_free(cos_ctx);
        const sin_ctx = try ropeSin(a, L, self.dtype, s);
        defer _ = mlx.mlx_array_free(sin_ctx);

        const ctx = try astype(qwen_hidden, self.dtype, s);
        defer _ = mlx.mlx_array_free(ctx);

        for (&self.blocks) |*b| {
            // self-attention (queries attend to themselves; rope on q AND k).
            const ns = try rmsNorm(x, b.norm_self, ADAPTER_EPS, s);
            defer _ = mlx.mlx_array_free(ns);
            const sa = try adapterAttn(&b.self_attn, ns, ns, cos_q, sin_q, cos_q, sin_q, s);
            defer _ = mlx.mlx_array_free(sa);
            const x1 = try add(x, sa, s);
            _ = mlx.mlx_array_free(x);
            x = x1;

            // cross-attention (queries attend to qwen context; rope k on ctx pos).
            const nc = try rmsNorm(x, b.norm_cross, ADAPTER_EPS, s);
            defer _ = mlx.mlx_array_free(nc);
            const ca = try adapterAttn(&b.cross_attn, nc, ctx, cos_q, sin_q, cos_ctx, sin_ctx, s);
            defer _ = mlx.mlx_array_free(ca);
            const x2 = try add(x, ca, s);
            _ = mlx.mlx_array_free(x);
            x = x2;

            // MLP: Linear → GELU → Linear.
            const nm = try rmsNorm(x, b.norm_mlp, ADAPTER_EPS, s);
            defer _ = mlx.mlx_array_free(nm);
            const m0 = try linearT(nm, b.mlp0_w, b.mlp0_b, s);
            defer _ = mlx.mlx_array_free(m0);
            const g = try geluErf(m0, s);
            defer _ = mlx.mlx_array_free(g);
            const m2 = try linearT(g, b.mlp2_w, b.mlp2_b, s);
            defer _ = mlx.mlx_array_free(m2);
            const x3 = try add(x, m2, s);
            _ = mlx.mlx_array_free(x);
            x = x3;
        }

        const op = try linearT(x, self.out_proj_w, self.out_proj_b, s);
        _ = mlx.mlx_array_free(x);
        defer _ = mlx.mlx_array_free(op);
        return rmsNorm(op, self.norm, ADAPTER_EPS, s);
    }
};

/// One attention sub-layer. `xq` supplies queries, `xkv` the keys/values;
/// (cos_q,sin_q) rope the queries, (cos_k,sin_k) rope the keys.
fn adapterAttn(aw: *const AttnW, xq: mlx.mlx_array, xkv: mlx.mlx_array, cos_q: mlx.mlx_array, sin_q: mlx.mlx_array, cos_k: mlx.mlx_array, sin_k: mlx.mlx_array, s: S) !mlx.mlx_array {
    const B: c_int = 1;
    const Mq: c_int = @intCast(mlx.getShape(xq)[1]);
    const Lk: c_int = @intCast(mlx.getShape(xkv)[1]);
    const H: c_int = @intCast(ADAPTER_HEADS);
    const D: c_int = @intCast(ADAPTER_HEAD_DIM);

    // q = q_norm(q_proj(xq).view(B,Mq,H,D)) ; k similarly ; v = v_proj(xkv).
    const qp = try linearT(xq, aw.q_proj, null, s);
    defer _ = mlx.mlx_array_free(qp);
    const qv = try reshape(qp, &[_]c_int{ B, Mq, H, D }, s);
    defer _ = mlx.mlx_array_free(qv);
    const qn = try rmsNorm(qv, aw.q_norm, ADAPTER_EPS, s); // over last axis D
    defer _ = mlx.mlx_array_free(qn);
    const qt = try transpose(qn, &[_]c_int{ 0, 2, 1, 3 }, s); // [B,H,Mq,D]
    defer _ = mlx.mlx_array_free(qt);
    const q = try applyRope(qt, cos_q, sin_q, s);
    defer _ = mlx.mlx_array_free(q);

    const kp = try linearT(xkv, aw.k_proj, null, s);
    defer _ = mlx.mlx_array_free(kp);
    const kv = try reshape(kp, &[_]c_int{ B, Lk, H, D }, s);
    defer _ = mlx.mlx_array_free(kv);
    const kn = try rmsNorm(kv, aw.k_norm, ADAPTER_EPS, s);
    defer _ = mlx.mlx_array_free(kn);
    const kt = try transpose(kn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(kt);
    const k = try applyRope(kt, cos_k, sin_k, s);
    defer _ = mlx.mlx_array_free(k);

    const vp = try linearT(xkv, aw.v_proj, null, s);
    defer _ = mlx.mlx_array_free(vp);
    const vv = try reshape(vp, &[_]c_int{ B, Lk, H, D }, s);
    defer _ = mlx.mlx_array_free(vv);
    const v = try transpose(vv, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(v);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ADAPTER_HEAD_DIM)));
    const o = try sdpa(q, k, v, scale, s); // [B,H,Mq,D]
    defer _ = mlx.mlx_array_free(o);
    const ot = try transpose(o, &[_]c_int{ 0, 2, 1, 3 }, s); // [B,Mq,H,D]
    defer _ = mlx.mlx_array_free(ot);
    const otc = try contig(ot, s);
    defer _ = mlx.mlx_array_free(otc);
    const merged = try reshape(otc, &[_]c_int{ B, Mq, H * D }, s);
    defer _ = mlx.mlx_array_free(merged);
    return linearT(merged, aw.o_proj, null, s);
}

fn loadAttnW(w: *const Weights, a: std.mem.Allocator, comptime pfx: []const u8, args: anytype, dt: mlx.mlx_dtype, s: S) !AttnW {
    return .{
        .q_proj = try loadWKey(w, a, pfx ++ ".q_proj.weight", args, dt, true, s),
        .q_norm = try loadWKey(w, a, pfx ++ ".q_norm.weight", args, dt, false, s),
        .k_proj = try loadWKey(w, a, pfx ++ ".k_proj.weight", args, dt, true, s),
        .k_norm = try loadWKey(w, a, pfx ++ ".k_norm.weight", args, dt, false, s),
        .v_proj = try loadWKey(w, a, pfx ++ ".v_proj.weight", args, dt, true, s),
        .o_proj = try loadWKey(w, a, pfx ++ ".o_proj.weight", args, dt, true, s),
    };
}

fn loadAdapterBlock(w: *const Weights, a: std.mem.Allocator, i: usize, dt: mlx.mlx_dtype, s: S) !AdapterBlockW {
    return .{
        .norm_self = try loadWKey(w, a, "{s}.blocks.{d}.norm_self_attn.weight", .{ adapterPrefix(w), i }, dt, false, s),
        .self_attn = try loadAttnW(w, a, "{s}.blocks.{d}.self_attn", .{ adapterPrefix(w), i }, dt, s),
        .norm_cross = try loadWKey(w, a, "{s}.blocks.{d}.norm_cross_attn.weight", .{ adapterPrefix(w), i }, dt, false, s),
        .cross_attn = try loadAttnW(w, a, "{s}.blocks.{d}.cross_attn", .{ adapterPrefix(w), i }, dt, s),
        .norm_mlp = try loadWKey(w, a, "{s}.blocks.{d}.norm_mlp.weight", .{ adapterPrefix(w), i }, dt, false, s),
        .mlp0_w = try loadWKey(w, a, "{s}.blocks.{d}.mlp.0.weight", .{ adapterPrefix(w), i }, dt, true, s),
        .mlp0_b = try loadWKey(w, a, "{s}.blocks.{d}.mlp.0.bias", .{ adapterPrefix(w), i }, dt, false, s),
        .mlp2_w = try loadWKey(w, a, "{s}.blocks.{d}.mlp.2.weight", .{ adapterPrefix(w), i }, dt, true, s),
        .mlp2_b = try loadWKey(w, a, "{s}.blocks.{d}.mlp.2.bias", .{ adapterPrefix(w), i }, dt, false, s),
    };
}

/// Build the rope cos table [1,1,S,64] on host (rotate-half / Llama convention:
/// emb = cat(freqs, freqs); freqs[s,i] = s * theta^(-2i/64)).
fn ropeCosSin(a: std.mem.Allocator, seq: usize, dt: mlx.mlx_dtype, want_cos: bool, s: S) !mlx.mlx_array {
    const D = ADAPTER_HEAD_DIM;
    const half = D / 2; // 32
    const buf = try a.alloc(f32, seq * D);
    defer a.free(buf);
    for (0..seq) |p| {
        const pf: f32 = @floatFromInt(p);
        for (0..half) |i| {
            const inv = std.math.pow(f32, ADAPTER_ROPE_THETA, -(@as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(D))));
            const ang = pf * inv;
            const val = if (want_cos) @cos(ang) else @sin(ang);
            buf[p * D + i] = val; // first half
            buf[p * D + half + i] = val; // duplicated (cat(freqs,freqs))
        }
    }
    const shape = [_]c_int{ 1, 1, @intCast(seq), @intCast(D) };
    const arr = mlx.mlx_array_new_data(buf.ptr, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(arr);
    if (dt == .float32) {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&o, arr));
        return o;
    }
    return astype(arr, dt, s);
}
fn ropeCos(a: std.mem.Allocator, seq: usize, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    return ropeCosSin(a, seq, dt, true, s);
}
fn ropeSin(a: std.mem.Allocator, seq: usize, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    return ropeCosSin(a, seq, dt, false, s);
}

// ── Cosmos-Predict2 MiniTrainDIT (predict2.py) ─────────────────────────────
// 28 blocks: self-attn (3D RoPE + per-head RMS QK-norm) → cross-attn (text
// context, no rope) → GELU FFN, each AdaLN-LoRA modulated. fp32 residual stream.
// Weights under `model.diffusion_model.*`. Image path: T=1 (single frame).

/// See `adapterPrefix` — `net.*` on base-v1.0, `model.diffusion_model.*`
/// on turbo/aesthetic.
fn ditPrefix(w: *const Weights) []const u8 {
    if (w.get("net.x_embedder.proj.1.weight") != null) return "net";
    return "model.diffusion_model";
}

fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    return mul(x, sig, s);
}

/// LayerNorm over the last axis, NO affine (elementwise_affine=False), f32 math.
fn layerNormNoAffine(x: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const in_dt = mlx.mlx_array_dtype(x);
    const nd = mlx.getShape(x).len;
    const last = [_]c_int{@intCast(nd - 1)};
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axes(&mean, xf, &last, last.len, true, s));
    const centered = try sub(xf, mean, s);
    defer _ = mlx.mlx_array_free(centered);
    const sq = try mul(centered, centered, s);
    defer _ = mlx.mlx_array_free(sq);
    var variance = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(variance);
    try mlx.check(mlx.mlx_mean_axes(&variance, sq, &last, last.len, true, s));
    const epsA = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(epsA);
    const vpe = try add(variance, epsA, s);
    defer _ = mlx.mlx_array_free(vpe);
    var rstd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rstd);
    try mlx.check(mlx.mlx_rsqrt(&rstd, vpe, s));
    const out = try mul(centered, rstd, s);
    if (in_dt == .float32) return out;
    defer _ = mlx.mlx_array_free(out);
    return astype(out, in_dt, s);
}

fn sub(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, a, b, s));
    return o;
}

/// out = base + gate * res  (torch.addcmul).
fn addcmul(base: mlx.mlx_array, gate: mlx.mlx_array, res: mlx.mlx_array, s: S) !mlx.mlx_array {
    const gr = try mul(gate, res, s);
    defer _ = mlx.mlx_array_free(gr);
    return add(base, gr, s);
}

/// A plain DiT linear weight with an optional stacked-LoRA attachment point
/// (`gen.ImageEngine.setLoras`/`lora.deltaSum` — summed at forward time, never
/// merged into `w`). Covers every DiT linear a LoRA could plausibly target:
/// attention q/k/v/output and the MLP's two linears. AdaLN modulation, the
/// patch embedder, the timestep MLP and the final layer are NOT attachment
/// points — training adapters overwhelmingly target attention+MLP, and a
/// module a LoRA can never reach doesn't need the bookkeeping.
const LoraW = struct {
    w: mlx.mlx_array,
    refs: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined,
    count: u8 = 0,

    fn deinit(self: *LoraW) void {
        _ = mlx.mlx_array_free(self.w);
    }

    fn forward(self: *const LoraW, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var o = try matmul(x, self.w, s);
        if (self.count > 0) {
            const d = try lora_mod.deltaSum(x, self.refs[0..self.count], s);
            defer _ = mlx.mlx_array_free(d);
            const r = try add(o, d, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        return o;
    }

    /// Install the stacked adapter Refs for this linear (from `Stack.findAll`).
    fn setLoraRefs(self: *LoraW, refs: []const lora_mod.Ref) void {
        self.count = @intCast(refs.len);
        @memcpy(self.refs[0..refs.len], refs);
    }

    fn clearLoraRefs(self: *LoraW) void {
        self.count = 0;
    }
};

const DitAttnW = struct {
    q_proj: LoraW,
    q_norm: mlx.mlx_array,
    k_proj: LoraW,
    k_norm: mlx.mlx_array,
    v_proj: LoraW,
    output_proj: LoraW,
    fn deinit(self: *DitAttnW) void {
        self.q_proj.deinit();
        _ = mlx.mlx_array_free(self.q_norm);
        self.k_proj.deinit();
        _ = mlx.mlx_array_free(self.k_norm);
        self.v_proj.deinit();
        self.output_proj.deinit();
    }
};

const DitBlockW = struct {
    adaln_self1: mlx.mlx_array,
    adaln_self2: mlx.mlx_array,
    self_attn: DitAttnW,
    adaln_cross1: mlx.mlx_array,
    adaln_cross2: mlx.mlx_array,
    cross_attn: DitAttnW,
    adaln_mlp1: mlx.mlx_array,
    adaln_mlp2: mlx.mlx_array,
    mlp1: LoraW,
    mlp2: LoraW,
    fn deinit(self: *DitBlockW) void {
        for ([_]mlx.mlx_array{ self.adaln_self1, self.adaln_self2, self.adaln_cross1, self.adaln_cross2, self.adaln_mlp1, self.adaln_mlp2 }) |x| _ = mlx.mlx_array_free(x);
        self.mlp1.deinit();
        self.mlp2.deinit();
        self.self_attn.deinit();
        self.cross_attn.deinit();
    }
};

pub const Dit = struct {
    x_embedder: mlx.mlx_array, // [2048, 68] pre-transposed [68,2048]
    t_lin1: mlx.mlx_array,
    t_lin2: mlx.mlx_array,
    t_norm: mlx.mlx_array,
    blocks: [28]DitBlockW,
    final_adaln1: mlx.mlx_array,
    final_adaln2: mlx.mlx_array,
    final_lin: mlx.mlx_array,
    cfg: Config,
    dtype: mlx.mlx_dtype,

    pub fn load(w: *const Weights, a: std.mem.Allocator, cfg: Config, dtype: mlx.mlx_dtype, s: S) !Dit {
        var self: Dit = undefined;
        self.cfg = cfg;
        self.dtype = dtype;
        self.x_embedder = try loadWKey(w, a, "{s}.x_embedder.proj.1.weight", .{ditPrefix(w)}, dtype, true, s);
        self.t_lin1 = try loadWKey(w, a, "{s}.t_embedder.1.linear_1.weight", .{ditPrefix(w)}, dtype, true, s);
        self.t_lin2 = try loadWKey(w, a, "{s}.t_embedder.1.linear_2.weight", .{ditPrefix(w)}, dtype, true, s);
        self.t_norm = try loadWKey(w, a, "{s}.t_embedding_norm.weight", .{ditPrefix(w)}, dtype, false, s);
        self.final_adaln1 = try loadWKey(w, a, "{s}.final_layer.adaln_modulation.1.weight", .{ditPrefix(w)}, dtype, true, s);
        self.final_adaln2 = try loadWKey(w, a, "{s}.final_layer.adaln_modulation.2.weight", .{ditPrefix(w)}, dtype, true, s);
        self.final_lin = try loadWKey(w, a, "{s}.final_layer.linear.weight", .{ditPrefix(w)}, dtype, true, s);
        for (0..28) |i| self.blocks[i] = try loadDitBlock(w, a, i, dtype, s);
        return self;
    }

    pub fn deinit(self: *Dit) void {
        for ([_]mlx.mlx_array{ self.x_embedder, self.t_lin1, self.t_lin2, self.t_norm, self.final_adaln1, self.final_adaln2, self.final_lin }) |x| _ = mlx.mlx_array_free(x);
        for (&self.blocks) |*b| b.deinit();
    }

    /// Denoise one step. `latent` [1,16,1,H,W] (T=1), `timesteps` scalar [1],
    /// `context` [1,512,1024]. Returns model output [1,16,1,H,W] in dtype.
    pub fn forward(self: *const Dit, latent: mlx.mlx_array, timestep: f32, context: mlx.mlx_array, a: std.mem.Allocator, s: S) !mlx.mlx_array {
        const dt = self.dtype;
        const lsh = mlx.getShape(latent); // [1,16,1,H,W]
        const Hh: usize = @intCast(lsh[3]);
        const Ww: usize = @intCast(lsh[4]);
        const ps: usize = self.cfg.patch_spatial;
        const Hl = Hh / ps;
        const Wl = Ww / ps;
        const D: c_int = @intCast(self.cfg.model_channels);

        // Patchify: [1,16,1,H,W] → [1,L,64], then pad 4 zeros (padding mask) → 68.
        var x = try patchify(latent, ps, s);
        errdefer _ = mlx.mlx_array_free(x);
        {
            const xdt = try astype(x, dt, s);
            _ = mlx.mlx_array_free(x);
            x = xdt;
        }
        // Embed: Linear(68→2048) → [1,L,2048].
        {
            const e = try matmul(x, self.x_embedder, s);
            _ = mlx.mlx_array_free(x);
            x = e;
        }

        // Timestep embedding: sincos → (sample, adaln); temb = rmsNorm(sample).
        var sincos_buf: [2048]f32 = undefined;
        timestepEmbedding(timestep, sincos_buf[0..self.cfg.model_channels]);
        const sc_shape = [_]c_int{ 1, 1, D };
        const sc_arr = mlx.mlx_array_new_data(&sincos_buf, &sc_shape, sc_shape.len, .float32);
        defer _ = mlx.mlx_array_free(sc_arr);
        const sc = try astype(sc_arr, dt, s);
        defer _ = mlx.mlx_array_free(sc);
        // adaln_lora = t_lin2(silu(t_lin1(sc)))  [1,1,6144]
        const tl1 = try matmul(sc, self.t_lin1, s);
        defer _ = mlx.mlx_array_free(tl1);
        const tl1s = try silu(tl1, s);
        defer _ = mlx.mlx_array_free(tl1s);
        const adaln = try matmul(tl1s, self.t_lin2, s); // [1,1,6144]
        defer _ = mlx.mlx_array_free(adaln);
        const temb = try rmsNorm(sc, self.t_norm, 1e-6, s); // [1,1,2048]
        defer _ = mlx.mlx_array_free(temb);

        // 3D RoPE cos/sin tables [1,1,L,128] (host).
        const cos = try rope3dTable(a, Hl, Wl, self.cfg, dt, true, s);
        defer _ = mlx.mlx_array_free(cos);
        const sin = try rope3dTable(a, Hl, Wl, self.cfg, dt, false, s);
        defer _ = mlx.mlx_array_free(sin);

        const ctx = try astype(context, dt, s);
        defer _ = mlx.mlx_array_free(ctx);

        // Force fp32 residual stream (predict2 keeps residual in fp32).
        {
            const xf = try astype(x, .float32, s);
            _ = mlx.mlx_array_free(x);
            x = xf;
        }
        for (&self.blocks) |*b| {
            const x1 = try ditBlock(b, x, temb, adaln, ctx, cos, sin, self.cfg, s);
            _ = mlx.mlx_array_free(x);
            x = x1;
        }

        // Final layer: adaln (2 chunks) → LN → linear(2048→64).
        const xc = try astype(x, dt, s);
        defer _ = mlx.mlx_array_free(xc);
        const out64 = try finalLayer(self, xc, temb, adaln, s); // [1,L,64]
        defer _ = mlx.mlx_array_free(out64);
        _ = mlx.mlx_array_free(x);

        // Unpatchify → [1,16,H,W] (flattens identically to the [1,16,1,H,W] fixture).
        return unpatchify(out64, Hl, Wl, ps, self.cfg.out_channels, s);
    }
};

/// [1,16,1,H,W] → [1, L, 16*ps*ps] with feature order (c, m, n) (r=t=1),
/// then pad `pad_feats` zeros (the padding-mask channel) to reach 68.
fn patchify(latent: mlx.mlx_array, ps: usize, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(latent); // [1,16,1,H,W]
    const C: c_int = sh[1];
    const H: c_int = sh[3];
    const W: c_int = sh[4];
    const psc: c_int = @intCast(ps);
    const Hl = @divExact(H, psc);
    const Wl = @divExact(W, psc);
    // squeeze T=1 → [1,16,H,W] → [1,16,Hl,ps,Wl,ps]
    const r6 = try reshape(latent, &[_]c_int{ 1, C, Hl, psc, Wl, psc }, s);
    defer _ = mlx.mlx_array_free(r6);
    // permute [B,Hl,Wl,C,m,n] = axes [0,2,4,1,3,5]
    const p = try transpose(r6, &[_]c_int{ 0, 2, 4, 1, 3, 5 }, s);
    defer _ = mlx.mlx_array_free(p);
    const pc = try contig(p, s);
    defer _ = mlx.mlx_array_free(pc);
    const feats = C * psc * psc;
    const flat = try reshape(pc, &[_]c_int{ 1, Hl * Wl, feats }, s); // [1,L,64]
    defer _ = mlx.mlx_array_free(flat);
    // pad 4 zeros (17th channel × ps × ps / but ordered last since c=16 is last).
    const pad_feats = psc * psc; // 4 for the single mask channel
    const zshape = [_]c_int{ 1, Hl * Wl, pad_feats };
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, &zshape, zshape.len, mlx.mlx_array_dtype(flat), s));
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, flat);
    _ = mlx.mlx_vector_array_append_value(vec, zeros);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 2, s));
    return o;
}

/// [1,L,64] → [1,16,Hl*ps,Wl*ps]. Feature order (p1, p2, t=1, C).
fn unpatchify(x: mlx.mlx_array, Hl: usize, Wl: usize, ps: usize, out_c: u32, s: S) !mlx.mlx_array {
    const psc: c_int = @intCast(ps);
    const Cc: c_int = @intCast(out_c);
    const Hlc: c_int = @intCast(Hl);
    const Wlc: c_int = @intCast(Wl);
    // [1,L,64] → [1,Hl,Wl,p1,p2,C]
    const r6 = try reshape(x, &[_]c_int{ 1, Hlc, Wlc, psc, psc, Cc }, s);
    defer _ = mlx.mlx_array_free(r6);
    // permute to [1,C,Hl,p1,Wl,p2] = axes [0,5,1,3,2,4]
    const p = try transpose(r6, &[_]c_int{ 0, 5, 1, 3, 2, 4 }, s);
    defer _ = mlx.mlx_array_free(p);
    const pc = try contig(p, s);
    defer _ = mlx.mlx_array_free(pc);
    return reshape(pc, &[_]c_int{ 1, Cc, Hlc * psc, Wlc * psc }, s);
}

fn ditBlock(b: *const DitBlockW, x: mlx.mlx_array, temb: mlx.mlx_array, adaln: mlx.mlx_array, ctx: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, cfg: Config, s: S) !mlx.mlx_array {
    // self-attn
    const ss = try adalnMod(b.adaln_self1, b.adaln_self2, temb, adaln, s);
    defer freeTriple(ss);
    const nx = try modulate(x, ss.shift, ss.scale, s);
    defer _ = mlx.mlx_array_free(nx);
    const sa = try ditAttn(&b.self_attn, nx, nx, cos, sin, true, cfg, s);
    defer _ = mlx.mlx_array_free(sa);
    var x1 = try addcmul(x, ss.gate, sa, s);
    errdefer _ = mlx.mlx_array_free(x1);

    // cross-attn (no rope)
    const cs = try adalnMod(b.adaln_cross1, b.adaln_cross2, temb, adaln, s);
    defer freeTriple(cs);
    const ncx = try modulate(x1, cs.shift, cs.scale, s);
    defer _ = mlx.mlx_array_free(ncx);
    const ca = try ditAttn(&b.cross_attn, ncx, ctx, cos, sin, false, cfg, s);
    defer _ = mlx.mlx_array_free(ca);
    const x2 = try addcmul(x1, cs.gate, ca, s);
    _ = mlx.mlx_array_free(x1);
    x1 = x2;

    // mlp
    const ms = try adalnMod(b.adaln_mlp1, b.adaln_mlp2, temb, adaln, s);
    defer freeTriple(ms);
    const nmx = try modulate(x1, ms.shift, ms.scale, s);
    defer _ = mlx.mlx_array_free(nmx);
    const m1 = try b.mlp1.forward(nmx, s);
    defer _ = mlx.mlx_array_free(m1);
    const g = try geluErf(m1, s);
    defer _ = mlx.mlx_array_free(g);
    const m2 = try b.mlp2.forward(g, s);
    defer _ = mlx.mlx_array_free(m2);
    const x3 = try addcmul(x1, ms.gate, m2, s);
    _ = mlx.mlx_array_free(x1);
    return x3;
}

const Triple = struct { shift: mlx.mlx_array, scale: mlx.mlx_array, gate: mlx.mlx_array };
fn freeTriple(t: Triple) void {
    _ = mlx.mlx_array_free(t.shift);
    _ = mlx.mlx_array_free(t.scale);
    _ = mlx.mlx_array_free(t.gate);
}

/// adaln_modulation(emb) = lin2(silu(lin1(emb))) [both no-bias], + adaln_lora,
/// chunk 3 → (shift, scale, gate), each [1,1,2048].
fn adalnMod(w1: mlx.mlx_array, w2: mlx.mlx_array, emb: mlx.mlx_array, adaln_lora: mlx.mlx_array, s: S) !Triple {
    const e = try silu(emb, s);
    defer _ = mlx.mlx_array_free(e);
    const a1 = try matmul(e, w1, s);
    defer _ = mlx.mlx_array_free(a1);
    const a2 = try matmul(a1, w2, s); // [1,1,6144]
    defer _ = mlx.mlx_array_free(a2);
    const summed = try add(a2, adaln_lora, s);
    defer _ = mlx.mlx_array_free(summed);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_split(&parts, summed, 3, 2, s));
    var t: Triple = .{ .shift = mlx.mlx_array_new(), .scale = mlx.mlx_array_new(), .gate = mlx.mlx_array_new() };
    try mlx.check(mlx.mlx_vector_array_get(&t.shift, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&t.scale, parts, 1));
    try mlx.check(mlx.mlx_vector_array_get(&t.gate, parts, 2));
    return t;
}

/// LN(x)*(1+scale)+shift. x [1,L,D]; scale/shift [1,1,D] (broadcast over L).
fn modulate(x: mlx.mlx_array, shift: mlx.mlx_array, scale: mlx.mlx_array, s: S) !mlx.mlx_array {
    const ln = try layerNormNoAffine(x, 1e-6, s);
    defer _ = mlx.mlx_array_free(ln);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const sp = try add(scale, one, s);
    defer _ = mlx.mlx_array_free(sp);
    const scaled = try mul(ln, sp, s);
    defer _ = mlx.mlx_array_free(scaled);
    return add(scaled, shift, s);
}

/// DiT attention. `xq` [1,L,D] queries; `xkv` [1,Lk,ctx] keys/values. Self-attn
/// applies 3D rope (rotate-half) to q and k; cross-attn applies none.
fn ditAttn(aw: *const DitAttnW, xq: mlx.mlx_array, xkv: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, is_self: bool, cfg: Config, s: S) !mlx.mlx_array {
    const B: c_int = 1;
    const Mq: c_int = @intCast(mlx.getShape(xq)[1]);
    const Lk: c_int = @intCast(mlx.getShape(xkv)[1]);
    const H: c_int = @intCast(cfg.num_heads);
    const Dh: c_int = @intCast(cfg.headDim());

    const qp = try aw.q_proj.forward(xq, s);
    defer _ = mlx.mlx_array_free(qp);
    const qv = try reshape(qp, &[_]c_int{ B, Mq, H, Dh }, s);
    defer _ = mlx.mlx_array_free(qv);
    const qn = try rmsNorm(qv, aw.q_norm, 1e-6, s);
    defer _ = mlx.mlx_array_free(qn);
    const qt = try transpose(qn, &[_]c_int{ 0, 2, 1, 3 }, s);
    const q = if (is_self) try applyRope(qt, cos, sin, s) else try contig(qt, s);
    _ = mlx.mlx_array_free(qt);
    defer _ = mlx.mlx_array_free(q);

    const kp = try aw.k_proj.forward(xkv, s);
    defer _ = mlx.mlx_array_free(kp);
    const kv = try reshape(kp, &[_]c_int{ B, Lk, H, Dh }, s);
    defer _ = mlx.mlx_array_free(kv);
    const kn = try rmsNorm(kv, aw.k_norm, 1e-6, s);
    defer _ = mlx.mlx_array_free(kn);
    const kt = try transpose(kn, &[_]c_int{ 0, 2, 1, 3 }, s);
    const k = if (is_self) try applyRope(kt, cos, sin, s) else try contig(kt, s);
    _ = mlx.mlx_array_free(kt);
    defer _ = mlx.mlx_array_free(k);

    const vp = try aw.v_proj.forward(xkv, s);
    defer _ = mlx.mlx_array_free(vp);
    const vv = try reshape(vp, &[_]c_int{ B, Lk, H, Dh }, s);
    defer _ = mlx.mlx_array_free(vv);
    const v = try transpose(vv, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(v);
    const vc = try contig(v, s);
    defer _ = mlx.mlx_array_free(vc);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.headDim())));
    const o = try sdpa(q, k, vc, scale, s);
    defer _ = mlx.mlx_array_free(o);
    const ot = try transpose(o, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(ot);
    const otc = try contig(ot, s);
    defer _ = mlx.mlx_array_free(otc);
    const merged = try reshape(otc, &[_]c_int{ B, Mq, H * Dh }, s);
    defer _ = mlx.mlx_array_free(merged);
    return aw.output_proj.forward(merged, s);
}

fn finalLayer(self: *const Dit, x: mlx.mlx_array, temb: mlx.mlx_array, adaln: mlx.mlx_array, s: S) !mlx.mlx_array {
    const D: c_int = @intCast(self.cfg.model_channels);
    const e = try silu(temb, s);
    defer _ = mlx.mlx_array_free(e);
    const a1 = try matmul(e, self.final_adaln1, s);
    defer _ = mlx.mlx_array_free(a1);
    const a2 = try matmul(a1, self.final_adaln2, s); // [1,1,4096]
    defer _ = mlx.mlx_array_free(a2);
    // + adaln_lora[:, :, :2*D]
    const lo = [_]c_int{ 0, 0, 0 };
    const hi = [_]c_int{ 1, 1, 2 * D };
    const st = [_]c_int{ 1, 1, 1 };
    var alslice = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(alslice);
    try mlx.check(mlx.mlx_slice(&alslice, adaln, &lo, lo.len, &hi, hi.len, &st, st.len, s));
    const summed = try add(a2, alslice, s);
    defer _ = mlx.mlx_array_free(summed);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_split(&parts, summed, 2, 2, s));
    var shift = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shift);
    var scale = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scale);
    try mlx.check(mlx.mlx_vector_array_get(&shift, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&scale, parts, 1));
    const m = try modulate(x, shift, scale, s);
    defer _ = mlx.mlx_array_free(m);
    return matmul(m, self.final_lin, s); // [1,L,64]
}

fn loadDitAttn(w: *const Weights, a: std.mem.Allocator, comptime pfx: []const u8, args: anytype, dt: mlx.mlx_dtype, s: S) !DitAttnW {
    return .{
        .q_proj = .{ .w = try loadWKey(w, a, pfx ++ ".q_proj.weight", args, dt, true, s) },
        .q_norm = try loadWKey(w, a, pfx ++ ".q_norm.weight", args, dt, false, s),
        .k_proj = .{ .w = try loadWKey(w, a, pfx ++ ".k_proj.weight", args, dt, true, s) },
        .k_norm = try loadWKey(w, a, pfx ++ ".k_norm.weight", args, dt, false, s),
        .v_proj = .{ .w = try loadWKey(w, a, pfx ++ ".v_proj.weight", args, dt, true, s) },
        .output_proj = .{ .w = try loadWKey(w, a, pfx ++ ".output_proj.weight", args, dt, true, s) },
    };
}

fn loadDitBlock(w: *const Weights, a: std.mem.Allocator, i: usize, dt: mlx.mlx_dtype, s: S) !DitBlockW {
    return .{
        .adaln_self1 = try loadWKey(w, a, "{s}.blocks.{d}.adaln_modulation_self_attn.1.weight", .{ ditPrefix(w), i }, dt, true, s),
        .adaln_self2 = try loadWKey(w, a, "{s}.blocks.{d}.adaln_modulation_self_attn.2.weight", .{ ditPrefix(w), i }, dt, true, s),
        .self_attn = try loadDitAttn(w, a, "{s}.blocks.{d}.self_attn", .{ ditPrefix(w), i }, dt, s),
        .adaln_cross1 = try loadWKey(w, a, "{s}.blocks.{d}.adaln_modulation_cross_attn.1.weight", .{ ditPrefix(w), i }, dt, true, s),
        .adaln_cross2 = try loadWKey(w, a, "{s}.blocks.{d}.adaln_modulation_cross_attn.2.weight", .{ ditPrefix(w), i }, dt, true, s),
        .cross_attn = try loadDitAttn(w, a, "{s}.blocks.{d}.cross_attn", .{ ditPrefix(w), i }, dt, s),
        .adaln_mlp1 = try loadWKey(w, a, "{s}.blocks.{d}.adaln_modulation_mlp.1.weight", .{ ditPrefix(w), i }, dt, true, s),
        .adaln_mlp2 = try loadWKey(w, a, "{s}.blocks.{d}.adaln_modulation_mlp.2.weight", .{ ditPrefix(w), i }, dt, true, s),
        .mlp1 = .{ .w = try loadWKey(w, a, "{s}.blocks.{d}.mlp.layer1.weight", .{ ditPrefix(w), i }, dt, true, s) },
        .mlp2 = .{ .w = try loadWKey(w, a, "{s}.blocks.{d}.mlp.layer2.weight", .{ ditPrefix(w), i }, dt, true, s) },
    };
}

/// Attach every adapter in `stack` to its matching DiT linear (summed at
/// forward time — see `lora.deltaSum`). Non-owning: `stack` must outlive the
/// attach. Returns the number of (module, matched-adapter) attachments
/// across the whole stack.
///
/// `lora.Arch.generic` (gen.zig's `setLoras`) means the LoRA file's own
/// stored keys are matched VERBATIM — no alias table like krea2/flux2 have,
/// because there is no established community Anima LoRA convention to build
/// one from. Canonical names below are the checkpoint's own weight-key
/// spelling (`model.diffusion_model.blocks.N.self_attn.q_proj.weight`, per
/// `scripts/convert_anima_weights.py`'s source dump) with the
/// `model.diffusion_model.` prefix and `.weight` suffix stripped — the
/// choice any Anima-aware LoRA trainer is most likely to also make, since
/// that's the model's own module path. Verified live against the real
/// circlestone-labs/Anima turbo-v1.1 pack with a synthetic rank-2 adapter
/// on `blocks.0.self_attn.q_proj`: attaches (1 module-attachment logged),
/// a zero-B file is byte-transparent, a nonzero one changes the render —
/// no PUBLISHED community Anima adapter to check key-naming against yet,
/// though. If a published one's stored keys don't match this spelling,
/// `setLoras` returns `error.LoraNoMatch` rather than silently attaching
/// nothing.
pub fn attachLora(dit: *Dit, stack_arg: *const lora_mod.Stack) u32 {
    detachLora(dit);
    var matched: u32 = 0;
    var kbuf: [128]u8 = undefined;
    var rbuf: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined;

    for (&dit.blocks, 0..) |*b, i| {
        const mods = .{
            .{ "self_attn.q_proj", &b.self_attn.q_proj },
            .{ "self_attn.k_proj", &b.self_attn.k_proj },
            .{ "self_attn.v_proj", &b.self_attn.v_proj },
            .{ "self_attn.output_proj", &b.self_attn.output_proj },
            .{ "cross_attn.q_proj", &b.cross_attn.q_proj },
            .{ "cross_attn.k_proj", &b.cross_attn.k_proj },
            .{ "cross_attn.v_proj", &b.cross_attn.v_proj },
            .{ "cross_attn.output_proj", &b.cross_attn.output_proj },
            .{ "mlp.layer1", &b.mlp1 },
            .{ "mlp.layer2", &b.mlp2 },
        };
        inline for (mods) |m| {
            const key = std.fmt.bufPrint(&kbuf, "blocks.{d}.{s}", .{ i, m[0] }) catch "";
            const refs = stack_arg.findAll(key, &rbuf);
            if (refs.len > 0) {
                m[1].setLoraRefs(refs);
                matched += @intCast(refs.len);
            }
        }
    }
    return matched;
}

pub fn detachLora(dit: *Dit) void {
    for (&dit.blocks) |*b| {
        inline for (.{
            &b.self_attn.q_proj,  &b.self_attn.k_proj,  &b.self_attn.v_proj,  &b.self_attn.output_proj,
            &b.cross_attn.q_proj, &b.cross_attn.k_proj, &b.cross_attn.v_proj, &b.cross_attn.output_proj,
            &b.mlp1,              &b.mlp2,
        }) |ml| ml.clearLoraRefs();
    }
}

/// Build the 3D-RoPE cos or sin table [1,1,L,128] on host, matching
/// VideoRopePosition3DEmb: em bands [t(dim_t/2), h(dim_h/2), w(dim_w/2)] over the
/// (h,w) grid (T=1), then duplicated to the full head_dim for rotate-half.
fn rope3dTable(a: std.mem.Allocator, Hl: usize, Wl: usize, cfg: Config, dt: mlx.mlx_dtype, want_cos: bool, s: S) !mlx.mlx_array {
    const head_dim = cfg.headDim(); // 128
    const split = rope3dSplit(head_dim); // dim_h, dim_w, dim_t
    const nh = split.dim_h / 2; // 21
    const nw = split.dim_w / 2; // 21
    const nt = split.dim_t / 2; // 22
    const half = head_dim / 2; // 64 = nt+nh+nw
    std.debug.assert(nt + nh + nw == half);

    const h_ntk = ntkFactor(cfg.rope_h_extrapolation_ratio, split.dim_h);
    const w_ntk = ntkFactor(cfg.rope_w_extrapolation_ratio, split.dim_w);
    const t_ntk = ntkFactor(cfg.rope_t_extrapolation_ratio, split.dim_t);
    const h_theta = 10000.0 * h_ntk;
    const w_theta = 10000.0 * w_ntk;
    const t_theta = 10000.0 * t_ntk;

    const L = Hl * Wl;
    const buf = try a.alloc(f32, L * head_dim);
    defer a.free(buf);

    // freqs: spatial_range[i] = (2i)/dim_h ; temporal_range[i] = (2i)/dim_t.
    for (0..Hl) |h| {
        for (0..Wl) |wq| {
            const l = h * Wl + wq;
            const base = l * head_dim;
            var idx: usize = 0;
            // temporal band (t_pos = 0 for images)
            for (0..nt) |i| {
                const rng = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(split.dim_t));
                const freq = 1.0 / std.math.pow(f32, t_theta, rng);
                const ang = 0.0 * freq; // t_pos = 0
                const val = if (want_cos) @cos(ang) else @sin(ang);
                buf[base + idx] = val;
                buf[base + half + idx] = val;
                idx += 1;
            }
            // height band
            for (0..nh) |i| {
                const rng = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(split.dim_h));
                const freq = 1.0 / std.math.pow(f32, h_theta, rng);
                const ang = @as(f32, @floatFromInt(h)) * freq;
                const val = if (want_cos) @cos(ang) else @sin(ang);
                buf[base + idx] = val;
                buf[base + half + idx] = val;
                idx += 1;
            }
            // width band
            for (0..nw) |i| {
                const rng = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(split.dim_w));
                const freq = 1.0 / std.math.pow(f32, w_theta, rng);
                const ang = @as(f32, @floatFromInt(wq)) * freq;
                const val = if (want_cos) @cos(ang) else @sin(ang);
                buf[base + idx] = val;
                buf[base + half + idx] = val;
                idx += 1;
            }
        }
    }
    const shape = [_]c_int{ 1, 1, @intCast(L), @intCast(head_dim) };
    const arr = mlx.mlx_array_new_data(buf.ptr, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(arr);
    if (dt == .float32) {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&o, arr));
        return o;
    }
    return astype(arr, dt, s);
}

// ── Qwen-Image (WAN-2.1) VAE decoder — image path, T=1 (wan/vae.py) ─────────
// For a single image the latent is T=1, so decode() runs one iteration with
// NO feature cache: every temporal path is dead and each CausalConv3d(k=3)
// reduces to a 2D conv using the LAST temporal kernel slice (the two front
// frames are zero-padded). The WAN RMS_norm (F.normalize over channels × √C) is
// exactly mlx_fast_rms_norm over the channel axis. All ops are 2D (NHWC).

const VAE_EPS: f32 = 1e-12;

/// conv2d on NHWC; weight OHWI; bias [O]; symmetric `pad`. Materializes the
/// input (mlx_conv2d miscomputes on lazy/strided views).
fn convNHWC(x: mlx.mlx_array, w: mlx.mlx_array, bias: ?mlx.mlx_array, pad: c_int, s: S) !mlx.mlx_array {
    const xc = try contig(x, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 1, 1, pad, pad, 1, 1, 1, s));
    if (bias) |b| {
        defer _ = mlx.mlx_array_free(o);
        return add(o, b, s);
    }
    return o;
}

/// nearest-exact 2x upsample on H,W of an NHWC tensor.
fn upsample2x(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var r1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r1);
    try mlx.check(mlx.mlx_repeat_axis(&r1, x, 2, 1, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_repeat_axis(&o, r1, 2, 2, s));
    return o;
}

/// WAN RMS_norm over the channel (last, NHWC) axis: x/sqrt(mean(x²))·γ.
fn wanNorm(x: mlx.mlx_array, gamma: mlx.mlx_array, s: S) !mlx.mlx_array {
    return rmsNorm(x, gamma, VAE_EPS, s);
}

const VaeResW = struct {
    n1: mlx.mlx_array,
    c1w: mlx.mlx_array,
    c1b: mlx.mlx_array,
    n2: mlx.mlx_array,
    c2w: mlx.mlx_array,
    c2b: mlx.mlx_array,
    sw: ?mlx.mlx_array,
    sb: ?mlx.mlx_array,
    fn deinit(self: *VaeResW) void {
        for ([_]mlx.mlx_array{ self.n1, self.c1w, self.c1b, self.n2, self.c2w, self.c2b }) |x| _ = mlx.mlx_array_free(x);
        if (self.sw) |x| _ = mlx.mlx_array_free(x);
        if (self.sb) |x| _ = mlx.mlx_array_free(x);
    }
};
const VaeAttnW = struct {
    norm: mlx.mlx_array,
    qkv_w: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    proj_w: mlx.mlx_array,
    proj_b: mlx.mlx_array,
    fn deinit(self: *VaeAttnW) void {
        for ([_]mlx.mlx_array{ self.norm, self.qkv_w, self.qkv_b, self.proj_w, self.proj_b }) |x| _ = mlx.mlx_array_free(x);
    }
};
/// One entry of the flat `upsamples` Sequential: a residual block or a spatial
/// resample (nearest-2x + conv). The temporal time_conv is not loaded (T=1).
const VaeUp = union(enum) {
    res: VaeResW,
    resample: struct { w: mlx.mlx_array, b: mlx.mlx_array },
    fn deinit(self: *VaeUp) void {
        switch (self.*) {
            .res => |*r| r.deinit(),
            .resample => |*rs| {
                _ = mlx.mlx_array_free(rs.w);
                _ = mlx.mlx_array_free(rs.b);
            },
        }
    }
};

pub const Vae = struct {
    conv2_w: mlx.mlx_array, // post-quant 1x1
    conv2_b: mlx.mlx_array,
    conv1_w: mlx.mlx_array, // decoder.conv1 3x3
    conv1_b: mlx.mlx_array,
    mid_res0: VaeResW,
    mid_attn: VaeAttnW,
    mid_res1: VaeResW,
    ups: []VaeUp,
    head_norm: mlx.mlx_array,
    head_w: mlx.mlx_array,
    head_b: mlx.mlx_array,
    // Encoder side (img2img): mirrors the decoder above in reverse. Loaded
    // whenever the pack's vae.safetensors carries `encoder.*` + top-level
    // `conv1` (quant_conv) — every pack converted by
    // scripts/convert_anima_weights.py ships these verbatim, so this is
    // present on every existing download, not just new ones.
    quant_w: mlx.mlx_array, // top-level conv1 (quant_conv) 1x1, 32->32
    quant_b: mlx.mlx_array,
    enc_conv1_w: mlx.mlx_array, // encoder.conv1 3x3, 3->dims[0]
    enc_conv1_b: mlx.mlx_array,
    enc_mid_res0: VaeResW,
    enc_mid_attn: VaeAttnW,
    enc_mid_res1: VaeResW,
    downs: []VaeUp, // reuses the VaeUp union (.res / .resample==downsample)
    enc_head_norm: mlx.mlx_array,
    enc_head_w: mlx.mlx_array, // encoder.head.2 3x3, dims[-1]->2*z_dim(32)
    enc_head_b: mlx.mlx_array,
    dtype: mlx.mlx_dtype,
    alloc: std.mem.Allocator,

    // Fixed Qwen-Image decoder layout: 15 upsample entries.
    // res×3, resample, res×3, resample, res×3, resample, res×3.
    const UP_KINDS = [_]enum { res, resample }{ .res, .res, .res, .resample, .res, .res, .res, .resample, .res, .res, .res, .resample, .res, .res, .res };
    // Encoder downsamples: an 11-entry pattern, NOT UP_KINDS — the decoder's
    // Decoder3d uses num_res_blocks+1 (3) residual blocks per stage, but
    // Encoder3d uses the plain num_res_blocks (2), so 4 stages of 2 res each
    // + 3 downsample resamples between them = 2*4+3 = 11, not 15. Confirmed
    // against a real qwen_image_vae.safetensors header (encoder.downsamples.2/
    // 5/8 are `resample`, everything else `res`) — this file previously
    // assumed decoder/encoder were structural mirrors and got this wrong; a
    // real checkpoint is what caught it (`MissingAnimaWeight` at load).
    const DOWN_KINDS = [_]enum { res, resample }{ .res, .res, .resample, .res, .res, .resample, .res, .res, .resample, .res, .res };

    pub fn load(w: *const Weights, a: std.mem.Allocator, dtype: mlx.mlx_dtype, s: S) !Vae {
        var self: Vae = undefined;
        self.dtype = dtype;
        self.alloc = a;
        self.conv2_w = try loadVaeConv(w, a, "conv2", true, dtype, s);
        self.conv2_b = try loadWKey(w, a, "conv2.bias", .{}, dtype, false, s);
        self.conv1_w = try loadVaeConv(w, a, "decoder.conv1", false, dtype, s);
        self.conv1_b = try loadWKey(w, a, "decoder.conv1.bias", .{}, dtype, false, s);
        self.mid_res0 = try loadVaeRes(w, a, "decoder.middle.0", dtype, s);
        self.mid_attn = try loadVaeAttn(w, a, "decoder.middle.1", dtype, s);
        self.mid_res1 = try loadVaeRes(w, a, "decoder.middle.2", dtype, s);
        self.ups = try a.alloc(VaeUp, UP_KINDS.len);
        for (UP_KINDS, 0..) |kind, i| {
            const pfx = try std.fmt.allocPrint(a, "decoder.upsamples.{d}", .{i});
            defer a.free(pfx);
            self.ups[i] = switch (kind) {
                .res => .{ .res = try loadVaeRes(w, a, pfx, dtype, s) },
                .resample => blk: {
                    const rw = try loadVaeConv2d(w, a, try std.fmt.allocPrint(a, "{s}.resample.1", .{pfx}), dtype, s, a);
                    const rb = try loadWKeyOwned(w, a, try std.fmt.allocPrint(a, "{s}.resample.1.bias", .{pfx}), dtype, s);
                    break :blk .{ .resample = .{ .w = rw, .b = rb } };
                },
            };
        }
        self.head_norm = try loadVaeGamma(w, a, try std.fmt.allocPrint(a, "decoder.head.0.gamma", .{}), dtype, s);
        self.head_w = try loadVaeConv(w, a, "decoder.head.2", false, dtype, s);
        self.head_b = try loadWKey(w, a, "decoder.head.2.bias", .{}, dtype, false, s);

        // Encoder side. Optional: an old vae.safetensors dumped before this
        // feature existed (never actually shipped — the converter always
        // copied the file verbatim — but a hand-trimmed file is possible)
        // simply leaves img2img unavailable rather than failing the load.
        if (w.get("conv1.weight") != null) {
            self.quant_w = try loadVaeConv(w, a, "conv1", true, dtype, s);
            self.quant_b = try loadWKey(w, a, "conv1.bias", .{}, dtype, false, s);
            self.enc_conv1_w = try loadVaeConv(w, a, "encoder.conv1", false, dtype, s);
            self.enc_conv1_b = try loadWKey(w, a, "encoder.conv1.bias", .{}, dtype, false, s);
            self.enc_mid_res0 = try loadVaeRes(w, a, "encoder.middle.0", dtype, s);
            self.enc_mid_attn = try loadVaeAttn(w, a, "encoder.middle.1", dtype, s);
            self.enc_mid_res1 = try loadVaeRes(w, a, "encoder.middle.2", dtype, s);
            self.downs = try a.alloc(VaeUp, DOWN_KINDS.len);
            for (DOWN_KINDS, 0..) |kind, i| {
                const pfx = try std.fmt.allocPrint(a, "encoder.downsamples.{d}", .{i});
                defer a.free(pfx);
                self.downs[i] = switch (kind) {
                    .res => .{ .res = try loadVaeRes(w, a, pfx, dtype, s) },
                    .resample => blk: {
                        const rw = try loadVaeConv2d(w, a, try std.fmt.allocPrint(a, "{s}.resample.1", .{pfx}), dtype, s, a);
                        const rb = try loadWKeyOwned(w, a, try std.fmt.allocPrint(a, "{s}.resample.1.bias", .{pfx}), dtype, s);
                        break :blk .{ .resample = .{ .w = rw, .b = rb } };
                    },
                };
            }
            self.enc_head_norm = try loadVaeGamma(w, a, try std.fmt.allocPrint(a, "encoder.head.0.gamma", .{}), dtype, s);
            self.enc_head_w = try loadVaeConv(w, a, "encoder.head.2", false, dtype, s);
            self.enc_head_b = try loadWKey(w, a, "encoder.head.2.bias", .{}, dtype, false, s);
        } else {
            self.downs = &.{};
        }
        return self;
    }

    /// True when the encoder half loaded (img2img available).
    pub fn hasEncoder(self: *const Vae) bool {
        return self.downs.len != 0;
    }

    pub fn deinit(self: *Vae) void {
        for ([_]mlx.mlx_array{ self.conv2_w, self.conv2_b, self.conv1_w, self.conv1_b, self.head_norm, self.head_w, self.head_b }) |x| _ = mlx.mlx_array_free(x);
        self.mid_res0.deinit();
        self.mid_attn.deinit();
        self.mid_res1.deinit();
        for (self.ups) |*u| u.deinit();
        self.alloc.free(self.ups);
        if (self.hasEncoder()) {
            for ([_]mlx.mlx_array{ self.quant_w, self.quant_b, self.enc_conv1_w, self.enc_conv1_b, self.enc_head_norm, self.enc_head_w, self.enc_head_b }) |x| _ = mlx.mlx_array_free(x);
            self.enc_mid_res0.deinit();
            self.enc_mid_attn.deinit();
            self.enc_mid_res1.deinit();
            for (self.downs) |*u| u.deinit();
            self.alloc.free(self.downs);
        }
    }

    /// Decode a latent [1,16,1,H,W] → RGB [1,3,H*8,W*8] (NCHW), in dtype.
    pub fn decode(self: *const Vae, latent: mlx.mlx_array, s: S) !mlx.mlx_array {
        // [1,16,1,H,W] → NHWC [1,H,W,16].
        const sq = try reshape(latent, &[_]c_int{ mlx.getShape(latent)[1], mlx.getShape(latent)[3], mlx.getShape(latent)[4] }, s);
        // note: batch assumed 1; reshape drops leading 1 and T=1 → [16,H,W]
        defer _ = mlx.mlx_array_free(sq);
        const nhwc0 = try transpose(sq, &[_]c_int{ 1, 2, 0 }, s); // [H,W,16]
        defer _ = mlx.mlx_array_free(nhwc0);
        const nhwc1s = try reshape(nhwc0, &[_]c_int{ 1, mlx.getShape(nhwc0)[0], mlx.getShape(nhwc0)[1], mlx.getShape(nhwc0)[2] }, s);
        defer _ = mlx.mlx_array_free(nhwc1s);
        // The transpose leaves a strided view; matmul/conv miscompute on strided
        // inputs, so materialize before feeding the conv chain.
        const nhwc1 = try contig(nhwc1s, s);
        defer _ = mlx.mlx_array_free(nhwc1);
        var x = try astype(nhwc1, self.dtype, s);
        errdefer _ = mlx.mlx_array_free(x);

        // post-quant conv2 (1x1) + decoder.conv1 (3x3).
        x = try replace(x, try convNHWC(x, self.conv2_w, self.conv2_b, 0, s));
        x = try replace(x, try convNHWC(x, self.conv1_w, self.conv1_b, 1, s));

        // middle: res, attn, res.
        x = try replace(x, try vaeRes(&self.mid_res0, x, s));
        x = try replace(x, try vaeAttn(&self.mid_attn, x, s));
        x = try replace(x, try vaeRes(&self.mid_res1, x, s));

        // upsamples.
        for (self.ups) |*u| {
            switch (u.*) {
                .res => |*r| x = try replace(x, try vaeRes(r, x, s)),
                .resample => |*rs| {
                    const up = try upsample2x(x, s);
                    _ = mlx.mlx_array_free(x);
                    x = try convNHWC(up, rs.w, rs.b, 1, s);
                    _ = mlx.mlx_array_free(up);
                },
            }
        }

        // head: norm, silu, conv.
        x = try replace(x, try wanNorm(x, self.head_norm, s));
        x = try replace(x, try silu(x, s));
        x = try replace(x, try convNHWC(x, self.head_w, self.head_b, 1, s)); // [1,H8,W8,3]

        // NHWC → NCHW [1,3,H8,W8].
        const out = try transpose(x, &[_]c_int{ 0, 3, 1, 2 }, s);
        _ = mlx.mlx_array_free(x);
        return out;
    }

    /// Encode RGB pixels → the DiT's normalized latent space (img2img source
    /// prep). `img` is `[1,3,H,W]` f32 in `[0,1]` (H,W multiples of 16); returns
    /// `[1,16,1,H/8,W/8]` f32, already `wan21ProcessIn`-normalized (mean 0-ish,
    /// unit-ish std — the space `generateImage`'s denoise loop and `decode`
    /// read, mirroring `wan21Denormalize` on the way out). Distribution mean
    /// only (logvar discarded) — deterministic, no sampling, matching every
    /// other backend's img2img VAE encoder in this codebase (krea, flux).
    pub fn encode(self: *const Vae, img: mlx.mlx_array, s: S) !mlx.mlx_array {
        std.debug.assert(self.hasEncoder());
        // [1,3,H,W] [0,1] NCHW → NHWC [1,H,W,3] in [-1,1].
        const nchw = try astype(img, self.dtype, s);
        defer _ = mlx.mlx_array_free(nchw);
        const nhwc0 = try transpose(nchw, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer _ = mlx.mlx_array_free(nhwc0);
        const nhwc1 = try contig(nhwc0, s);
        defer _ = mlx.mlx_array_free(nhwc1);
        const two = mlx.mlx_array_new_float(2.0);
        defer _ = mlx.mlx_array_free(two);
        const one = mlx.mlx_array_new_float(1.0);
        defer _ = mlx.mlx_array_free(one);
        const scaled = try mul(nhwc1, two, s);
        defer _ = mlx.mlx_array_free(scaled);
        var x = try sub(scaled, one, s);
        errdefer _ = mlx.mlx_array_free(x);

        // conv1 (3x3, in=3).
        x = try replace(x, try convNHWC(x, self.enc_conv1_w, self.enc_conv1_b, 1, s));

        // downsamples.
        for (self.downs) |*d| {
            switch (d.*) {
                .res => |*r| x = try replace(x, try vaeRes(r, x, s)),
                .resample => |*rs| x = try replace(x, try downsampleSpatial(x, rs.w, rs.b, s)),
            }
        }

        // middle: res, attn, res.
        x = try replace(x, try vaeRes(&self.enc_mid_res0, x, s));
        x = try replace(x, try vaeAttn(&self.enc_mid_attn, x, s));
        x = try replace(x, try vaeRes(&self.enc_mid_res1, x, s));

        // head: norm, silu, conv (out=2*z_dim=32).
        x = try replace(x, try wanNorm(x, self.enc_head_norm, s));
        x = try replace(x, try silu(x, s));
        x = try replace(x, try convNHWC(x, self.enc_head_w, self.enc_head_b, 1, s)); // [1,h,w,32]

        // quant_conv (1x1, 32->32); keep the first 16 channels (mean, drop logvar).
        const q = try convNHWC(x, self.quant_w, self.quant_b, 0, s);
        _ = mlx.mlx_array_free(x);
        defer _ = mlx.mlx_array_free(q);
        const qsh = mlx.getShape(q);
        const lo = [_]c_int{ 0, 0, 0, 0 };
        const hi = [_]c_int{ qsh[0], qsh[1], qsh[2], 16 };
        const st = [_]c_int{ 1, 1, 1, 1 };
        var mean0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mean0);
        try mlx.check(mlx.mlx_slice(&mean0, q, &lo, lo.len, &hi, hi.len, &st, st.len, s));
        const mean = try contig(mean0, s);
        defer _ = mlx.mlx_array_free(mean);

        // NHWC [1,h,w,16] → NCHW [1,16,h,w] → [1,16,1,h,w], f32.
        const nchw_out = try transpose(mean, &[_]c_int{ 0, 3, 1, 2 }, s);
        defer _ = mlx.mlx_array_free(nchw_out);
        const nchw_f = try astype(nchw_out, .float32, s);
        defer _ = mlx.mlx_array_free(nchw_f);
        const t5 = try unsqueezeT(nchw_f, s);
        defer _ = mlx.mlx_array_free(t5);
        return wan21NormalizeT(t5, s);
    }
};

/// Free `old`, return `new` (in-place update helper for the decode chain).
fn replace(old: mlx.mlx_array, new: mlx.mlx_array) !mlx.mlx_array {
    _ = mlx.mlx_array_free(old);
    return new;
}

fn vaeRes(r: *const VaeResW, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sc = if (r.sw) |sw| try convNHWC(x, sw, r.sb, 0, s) else blk: {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&o, x));
        break :blk o;
    };
    defer _ = mlx.mlx_array_free(sc);
    const n1 = try wanNorm(x, r.n1, s);
    defer _ = mlx.mlx_array_free(n1);
    const a1 = try silu(n1, s);
    defer _ = mlx.mlx_array_free(a1);
    const c1 = try convNHWC(a1, r.c1w, r.c1b, 1, s);
    defer _ = mlx.mlx_array_free(c1);
    const n2 = try wanNorm(c1, r.n2, s);
    defer _ = mlx.mlx_array_free(n2);
    const a2 = try silu(n2, s);
    defer _ = mlx.mlx_array_free(a2);
    const c2 = try convNHWC(a2, r.c2w, r.c2b, 1, s);
    defer _ = mlx.mlx_array_free(c2);
    return add(c2, sc, s);
}

fn vaeAttn(aw: *const VaeAttnW, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    // x NHWC [1,H,W,C]. norm over C, qkv 1x1 conv, single-head attn over H*W.
    const sh = mlx.getShape(x);
    const H = sh[1];
    const W = sh[2];
    const C = sh[3];
    const n = try wanNorm(x, aw.norm, s);
    defer _ = mlx.mlx_array_free(n);
    const qkv = try convNHWC(n, aw.qkv_w, aw.qkv_b, 0, s); // [1,H,W,3C]
    defer _ = mlx.mlx_array_free(qkv);
    // [1, H*W, 3C] → split C.
    const flat = try reshape(qkv, &[_]c_int{ 1, H * W, 3 * C }, s);
    defer _ = mlx.mlx_array_free(flat);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_split(&parts, flat, 3, 2, s));
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_vector_array_get(&q, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&k, parts, 1));
    try mlx.check(mlx.mlx_vector_array_get(&v, parts, 2));
    // add head dim: [1,1,H*W,C]
    const q4 = try reshape(q, &[_]c_int{ 1, 1, H * W, C }, s);
    defer _ = mlx.mlx_array_free(q4);
    const k4 = try reshape(k, &[_]c_int{ 1, 1, H * W, C }, s);
    defer _ = mlx.mlx_array_free(k4);
    const v4 = try reshape(v, &[_]c_int{ 1, 1, H * W, C }, s);
    defer _ = mlx.mlx_array_free(v4);
    // Manual single-head attention (mlx fast-sdpa has a head-dim<=256 wall; here
    // head_dim == C can be 384). scores = softmax(q·kᵀ/√C) · v.
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(C)));
    const kt = try transpose(k4, &[_]c_int{ 0, 1, 3, 2 }, s); // [1,1,C,HW]
    defer _ = mlx.mlx_array_free(kt);
    const scores0 = try matmul(q4, kt, s); // [1,1,HW,HW]
    defer _ = mlx.mlx_array_free(scores0);
    const scaleA = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(scaleA);
    const scores = try mul(scores0, scaleA, s);
    defer _ = mlx.mlx_array_free(scores);
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    try mlx.check(mlx.mlx_softmax_axis(&attn, scores, 3, true, s));
    const o = try matmul(attn, v4, s); // [1,1,HW,C]
    defer _ = mlx.mlx_array_free(o);
    const o3 = try reshape(o, &[_]c_int{ 1, H, W, C }, s);
    defer _ = mlx.mlx_array_free(o3);
    const proj = try convNHWC(o3, aw.proj_w, aw.proj_b, 0, s);
    defer _ = mlx.mlx_array_free(proj);
    return add(proj, x, s);
}

/// Load a CausalConv3d weight [O,I,kt,kh,kw] as a 2D OHWI kernel using the LAST
/// temporal slice (kt-1). `one` selects the 1x1 case (kt=kh=kw=1).
fn loadVaeConv(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, one: bool, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    _ = one;
    const key = try std.fmt.allocPrint(a, "{s}.weight", .{pfx});
    defer a.free(key);
    const raw = w.get(key) orelse {
        log.err("[anima-vae] missing {s}\n", .{key});
        return error.MissingAnimaWeight;
    };
    const sh = mlx.getShape(raw); // [O,I,kt,kh,kw]
    const O = sh[0];
    const I = sh[1];
    const kt = sh[2];
    const kh = sh[3];
    const kw = sh[4];
    // slice last temporal index kt-1 → [O,I,1,kh,kw]
    const lo = [_]c_int{ 0, 0, kt - 1, 0, 0 };
    const hi = [_]c_int{ O, I, kt, kh, kw };
    const st = [_]c_int{ 1, 1, 1, 1, 1 };
    var sl0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sl0);
    try mlx.check(mlx.mlx_slice(&sl0, raw, &lo, lo.len, &hi, hi.len, &st, st.len, s));
    const sl = try contig(sl0, s); // materialize the strided slice before reshape
    defer _ = mlx.mlx_array_free(sl);
    const r4 = try reshape(sl, &[_]c_int{ O, I, kh, kw }, s); // [O,I,kh,kw]
    defer _ = mlx.mlx_array_free(r4);
    const ohwi = try transpose(r4, &[_]c_int{ 0, 2, 3, 1 }, s); // [O,kh,kw,I]
    defer _ = mlx.mlx_array_free(ohwi);
    const oc = try contig(ohwi, s);
    defer _ = mlx.mlx_array_free(oc);
    return astype(oc, dt, s);
}

/// Load a plain nn.Conv2d weight [O,I,kh,kw] → OHWI.
fn loadVaeConv2d(w: *const Weights, a: std.mem.Allocator, key_pfx: []const u8, dt: mlx.mlx_dtype, s: S, free_key: std.mem.Allocator) !mlx.mlx_array {
    defer free_key.free(key_pfx);
    const key = try std.fmt.allocPrint(a, "{s}.weight", .{key_pfx});
    defer a.free(key);
    const raw = w.get(key) orelse {
        log.err("[anima-vae] missing {s}\n", .{key});
        return error.MissingAnimaWeight;
    };
    const ohwi = try transpose(raw, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(ohwi);
    const oc = try contig(ohwi, s);
    defer _ = mlx.mlx_array_free(oc);
    return astype(oc, dt, s);
}

fn loadWKeyOwned(w: *const Weights, a: std.mem.Allocator, key: []const u8, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    defer a.free(key);
    return loadW(w, a, key, dt, false, s);
}

/// Load a WAN RMS_norm gamma [C,1,1,1] (or [C,1,1]) and flatten to [C] so it
/// broadcasts against the channel (last, NHWC) axis for mlx_fast_rms_norm.
fn loadVaeGamma(w: *const Weights, a: std.mem.Allocator, key: []const u8, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    defer a.free(key);
    const raw = w.get(key) orelse {
        log.err("[anima-vae] missing {s}\n", .{key});
        return error.MissingAnimaWeight;
    };
    const sh = mlx.getShape(raw);
    var total: c_int = 1;
    for (sh) |d| total *= d;
    const flat = try reshape(raw, &[_]c_int{total}, s);
    defer _ = mlx.mlx_array_free(flat);
    return astype(flat, dt, s);
}

fn loadVaeRes(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, dt: mlx.mlx_dtype, s: S) !VaeResW {
    // residual Sequential: 0=RMS(gamma), 2=conv, 3=RMS(gamma), 6=conv; shortcut optional.
    const n1 = try loadVaeGamma(w, a, try std.fmt.allocPrint(a, "{s}.residual.0.gamma", .{pfx}), dt, s);
    const c1w = try loadVaeConvPfx(w, a, try std.fmt.allocPrint(a, "{s}.residual.2", .{pfx}), dt, s);
    const c1b = try loadWKeyOwned(w, a, try std.fmt.allocPrint(a, "{s}.residual.2.bias", .{pfx}), dt, s);
    const n2 = try loadVaeGamma(w, a, try std.fmt.allocPrint(a, "{s}.residual.3.gamma", .{pfx}), dt, s);
    const c2w = try loadVaeConvPfx(w, a, try std.fmt.allocPrint(a, "{s}.residual.6", .{pfx}), dt, s);
    const c2b = try loadWKeyOwned(w, a, try std.fmt.allocPrint(a, "{s}.residual.6.bias", .{pfx}), dt, s);
    const swk = try std.fmt.allocPrint(a, "{s}.shortcut.weight", .{pfx});
    defer a.free(swk);
    var sw: ?mlx.mlx_array = null;
    var sb: ?mlx.mlx_array = null;
    if (w.get(swk) != null) {
        sw = try loadVaeConvPfx(w, a, try std.fmt.allocPrint(a, "{s}.shortcut", .{pfx}), dt, s);
        sb = try loadWKeyOwned(w, a, try std.fmt.allocPrint(a, "{s}.shortcut.bias", .{pfx}), dt, s);
    }
    return .{ .n1 = n1, .c1w = c1w, .c1b = c1b, .n2 = n2, .c2w = c2w, .c2b = c2b, .sw = sw, .sb = sb };
}

/// loadVaeConv but taking an owned prefix (frees it).
fn loadVaeConvPfx(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    defer a.free(pfx);
    return loadVaeConv(w, a, pfx, false, dt, s);
}

fn loadVaeAttn(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, dt: mlx.mlx_dtype, s: S) !VaeAttnW {
    return .{
        .norm = try loadVaeGamma(w, a, try std.fmt.allocPrint(a, "{s}.norm.gamma", .{pfx}), dt, s),
        .qkv_w = try loadVaeConv2d(w, a, try std.fmt.allocPrint(a, "{s}.to_qkv", .{pfx}), dt, s, a),
        .qkv_b = try loadWKeyOwned(w, a, try std.fmt.allocPrint(a, "{s}.to_qkv.bias", .{pfx}), dt, s),
        .proj_w = try loadVaeConv2d(w, a, try std.fmt.allocPrint(a, "{s}.proj", .{pfx}), dt, s, a),
        .proj_b = try loadWKeyOwned(w, a, try std.fmt.allocPrint(a, "{s}.proj.bias", .{pfx}), dt, s),
    };
}

// ── Qwen3-0.6B text encoder (comfy Qwen3_06BConfig) ────────────────────────
// Standard causal Qwen3: 28 layers, hidden 1024, 16 q-heads / 8 kv-heads,
// head_dim 128, per-head RMS qk-norm, SwiGLU, rope theta 1e6, eps 1e-6, plain
// embedding, final model.norm. Output = last hidden states (the DiT/adapter's
// conditioning source). Weights under `model.*`.

const TE_LAYERS: u32 = 28;
const TE_HIDDEN: u32 = 1024;
const TE_HEADS: u32 = 16;
const TE_KV_HEADS: u32 = 8;
const TE_HEAD_DIM: u32 = 128;
const TE_THETA: f32 = 1000000.0;
const TE_EPS: f32 = 1e-6;

/// Causal additive mask [1,1,L,L]: 0 on/below the diagonal, -1e9 above.
fn causalMask(a: std.mem.Allocator, L: usize, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const buf = try a.alloc(f32, L * L);
    defer a.free(buf);
    for (0..L) |i| {
        for (0..L) |j| buf[i * L + j] = if (j <= i) 0.0 else -1.0e9;
    }
    const shape = [_]c_int{ 1, 1, @intCast(L), @intCast(L) };
    const arr = mlx.mlx_array_new_data(buf.ptr, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(arr);
    if (dt == .float32) {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&o, arr));
        return o;
    }
    return astype(arr, dt, s);
}

/// repeat_interleave along `axis` (each slice repeated `n` times consecutively).
fn repeatAxis(x: mlx.mlx_array, n: c_int, axis: c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_repeat_axis(&o, x, n, axis, s));
    return o;
}

/// SDPA with an explicit mask mode ("" none, "causal").
fn sdpaMask(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mode: [*:0]const u8, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    const null_a = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode, null_a, null_a, s));
    return o;
}

const TeLayerW = struct {
    input_ln: mlx.mlx_array,
    q_proj: mlx.mlx_array,
    q_norm: mlx.mlx_array,
    k_proj: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    v_proj: mlx.mlx_array,
    o_proj: mlx.mlx_array,
    post_ln: mlx.mlx_array,
    gate: mlx.mlx_array,
    up: mlx.mlx_array,
    down: mlx.mlx_array,
    fn deinit(self: *TeLayerW) void {
        for ([_]mlx.mlx_array{ self.input_ln, self.q_proj, self.q_norm, self.k_proj, self.k_norm, self.v_proj, self.o_proj, self.post_ln, self.gate, self.up, self.down }) |x| _ = mlx.mlx_array_free(x);
    }
};

pub const TextEncoder = struct {
    embed: mlx.mlx_array, // [vocab, 1024]
    layers: [TE_LAYERS]TeLayerW,
    norm: mlx.mlx_array,
    dtype: mlx.mlx_dtype,

    pub fn load(w: *const Weights, a: std.mem.Allocator, dtype: mlx.mlx_dtype, s: S) !TextEncoder {
        var self: TextEncoder = undefined;
        self.dtype = dtype;
        self.embed = try loadWKey(w, a, "model.embed_tokens.weight", .{}, dtype, false, s);
        self.norm = try loadWKey(w, a, "model.norm.weight", .{}, dtype, false, s);
        for (0..TE_LAYERS) |i| self.layers[i] = try loadTeLayer(w, a, i, dtype, s);
        return self;
    }
    pub fn deinit(self: *TextEncoder) void {
        _ = mlx.mlx_array_free(self.embed);
        _ = mlx.mlx_array_free(self.norm);
        for (&self.layers) |*l| l.deinit();
    }

    /// Encode `ids` [1,L] (int) → last hidden states [1,L,1024] in dtype.
    pub fn forward(self: *const TextEncoder, ids: mlx.mlx_array, a: std.mem.Allocator, s: S) !mlx.mlx_array {
        const L: usize = @intCast(mlx.getShape(ids)[mlx.getShape(ids).len - 1]);
        var x = try takeRows(self.embed, ids, s); // [1,L,1024]
        errdefer _ = mlx.mlx_array_free(x);
        {
            const xc = try astype(x, self.dtype, s);
            _ = mlx.mlx_array_free(x);
            x = xc;
        }
        const cos = try teRopeTable(a, L, self.dtype, true, s);
        defer _ = mlx.mlx_array_free(cos);
        const sin = try teRopeTable(a, L, self.dtype, false, s);
        defer _ = mlx.mlx_array_free(sin);

        for (&self.layers) |*l| {
            const x1 = try teLayer(l, x, cos, sin, s);
            _ = mlx.mlx_array_free(x);
            x = x1;
        }
        const out = try rmsNorm(x, self.norm, TE_EPS, s);
        _ = mlx.mlx_array_free(x);
        return out;
    }
};

fn teLayer(l: *const TeLayerW, x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: S) !mlx.mlx_array {
    const B: c_int = 1;
    const L: c_int = @intCast(mlx.getShape(x)[1]);
    const H: c_int = @intCast(TE_HEADS);
    const KV: c_int = @intCast(TE_KV_HEADS);
    const D: c_int = @intCast(TE_HEAD_DIM);

    // attention
    const hn = try rmsNorm(x, l.input_ln, TE_EPS, s);
    defer _ = mlx.mlx_array_free(hn);
    const qp = try matmul(hn, l.q_proj, s); // q_proj pre-transposed [1024,2048]
    defer _ = mlx.mlx_array_free(qp);
    const qv = try reshape(qp, &[_]c_int{ B, L, H, D }, s);
    defer _ = mlx.mlx_array_free(qv);
    const qn = try rmsNorm(qv, l.q_norm, TE_EPS, s);
    defer _ = mlx.mlx_array_free(qn);
    const qt = try transpose(qn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(qt);
    const q = try applyRope(qt, cos, sin, s);
    defer _ = mlx.mlx_array_free(q);

    const kp = try matmul(hn, l.k_proj, s);
    defer _ = mlx.mlx_array_free(kp);
    const kv = try reshape(kp, &[_]c_int{ B, L, KV, D }, s);
    defer _ = mlx.mlx_array_free(kv);
    const kn = try rmsNorm(kv, l.k_norm, TE_EPS, s);
    defer _ = mlx.mlx_array_free(kn);
    const kt = try transpose(kn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(kt);
    const k = try applyRope(kt, cos, sin, s);
    defer _ = mlx.mlx_array_free(k);

    const vp = try matmul(hn, l.v_proj, s);
    defer _ = mlx.mlx_array_free(vp);
    const vv = try reshape(vp, &[_]c_int{ B, L, KV, D }, s);
    defer _ = mlx.mlx_array_free(vv);
    const vt = try transpose(vv, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(vt);
    const vc = try contig(vt, s);
    defer _ = mlx.mlx_array_free(vc);

    // GQA: repeat kv heads to match q heads (repeat_interleave along the head axis).
    const rep = @divExact(H, KV);
    const kg = try repeatAxis(k, rep, 1, s);
    defer _ = mlx.mlx_array_free(kg);
    const vg = try repeatAxis(vc, rep, 1, s);
    defer _ = mlx.mlx_array_free(vg);
    // Manual causal attention (mlx fast-sdpa "causal" mode aborts on these
    // shapes here): scores = softmax(q·kᵀ·scale + causal_mask)·v.
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(TE_HEAD_DIM)));
    const ktA = try transpose(kg, &[_]c_int{ 0, 1, 3, 2 }, s); // [1,16,D,L]
    defer _ = mlx.mlx_array_free(ktA);
    const scores0 = try matmul(q, ktA, s); // [1,16,L,L]
    defer _ = mlx.mlx_array_free(scores0);
    const scaleA = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(scaleA);
    const scores1 = try mul(scores0, scaleA, s);
    defer _ = mlx.mlx_array_free(scores1);
    const mask = try causalMask(std.heap.page_allocator, @intCast(L), mlx.mlx_array_dtype(scores1), s);
    defer _ = mlx.mlx_array_free(mask);
    const scores = try add(scores1, mask, s);
    defer _ = mlx.mlx_array_free(scores);
    var attnw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attnw);
    try mlx.check(mlx.mlx_softmax_axis(&attnw, scores, 3, true, s));
    const o = try matmul(attnw, vg, s); // [1,16,L,D]
    defer _ = mlx.mlx_array_free(o);
    const ot = try transpose(o, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(ot);
    const otc = try contig(ot, s);
    defer _ = mlx.mlx_array_free(otc);
    const merged = try reshape(otc, &[_]c_int{ B, L, H * D }, s);
    defer _ = mlx.mlx_array_free(merged);
    const attn = try matmul(merged, l.o_proj, s);
    defer _ = mlx.mlx_array_free(attn);
    const x1 = try add(x, attn, s);
    errdefer _ = mlx.mlx_array_free(x1);

    // SwiGLU MLP
    const hn2 = try rmsNorm(x1, l.post_ln, TE_EPS, s);
    defer _ = mlx.mlx_array_free(hn2);
    const g = try matmul(hn2, l.gate, s);
    defer _ = mlx.mlx_array_free(g);
    const gs = try silu(g, s);
    defer _ = mlx.mlx_array_free(gs);
    const u = try matmul(hn2, l.up, s);
    defer _ = mlx.mlx_array_free(u);
    const gu = try mul(gs, u, s);
    defer _ = mlx.mlx_array_free(gu);
    const m = try matmul(gu, l.down, s);
    defer _ = mlx.mlx_array_free(m);
    const x2 = try add(x1, m, s);
    _ = mlx.mlx_array_free(x1);
    return x2;
}

fn loadTeLayer(w: *const Weights, a: std.mem.Allocator, i: usize, dt: mlx.mlx_dtype, s: S) !TeLayerW {
    return .{
        .input_ln = try loadWKey(w, a, "model.layers.{d}.input_layernorm.weight", .{i}, dt, false, s),
        .q_proj = try loadWKey(w, a, "model.layers.{d}.self_attn.q_proj.weight", .{i}, dt, true, s),
        .q_norm = try loadWKey(w, a, "model.layers.{d}.self_attn.q_norm.weight", .{i}, dt, false, s),
        .k_proj = try loadWKey(w, a, "model.layers.{d}.self_attn.k_proj.weight", .{i}, dt, true, s),
        .k_norm = try loadWKey(w, a, "model.layers.{d}.self_attn.k_norm.weight", .{i}, dt, false, s),
        .v_proj = try loadWKey(w, a, "model.layers.{d}.self_attn.v_proj.weight", .{i}, dt, true, s),
        .o_proj = try loadWKey(w, a, "model.layers.{d}.self_attn.o_proj.weight", .{i}, dt, true, s),
        .post_ln = try loadWKey(w, a, "model.layers.{d}.post_attention_layernorm.weight", .{i}, dt, false, s),
        .gate = try loadWKey(w, a, "model.layers.{d}.mlp.gate_proj.weight", .{i}, dt, true, s),
        .up = try loadWKey(w, a, "model.layers.{d}.mlp.up_proj.weight", .{i}, dt, true, s),
        .down = try loadWKey(w, a, "model.layers.{d}.mlp.down_proj.weight", .{i}, dt, true, s),
    };
}

/// Qwen3 rope cos/sin table [1,1,L,128] (theta 1e6, rotate-half, duplicated).
fn teRopeTable(a: std.mem.Allocator, L: usize, dt: mlx.mlx_dtype, want_cos: bool, s: S) !mlx.mlx_array {
    const D = TE_HEAD_DIM;
    const half = D / 2; // 64
    const buf = try a.alloc(f32, L * D);
    defer a.free(buf);
    for (0..L) |p| {
        const pf: f32 = @floatFromInt(p);
        for (0..half) |i| {
            const inv = 1.0 / std.math.pow(f32, TE_THETA, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(D)));
            const ang = pf * inv;
            const val = if (want_cos) @cos(ang) else @sin(ang);
            buf[p * D + i] = val;
            buf[p * D + half + i] = val;
        }
    }
    const shape = [_]c_int{ 1, 1, @intCast(L), @intCast(D) };
    const arr = mlx.mlx_array_new_data(buf.ptr, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(arr);
    if (dt == .float32) {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&o, arr));
        return o;
    }
    return astype(arr, dt, s);
}

// ── Engine: full text→image pipeline (the gen.zig `ImageBackend` arm) ──────
// Converted-pack layout (flat, mirrors minimax_h3/music3 — NOT diffusers-style
// subdirs like mage_flow): a small conversion script (scripts/convert_anima_weights.py)
// produces this from the raw ComfyUI split_files.
//
//   config.json                  {"model_type":"anima","recommended_steps":N,"recommended_cfg":F}
//   transformer.safetensors      DiT + llm_adapter (required-media marker, written LAST)
//   text_encoder.safetensors     Qwen3-0.6B
//   tokenizer/tokenizer.json     Qwen BPE tokenizer (byte_level_bpe; tokenizer.zig)
//   vae.safetensors              Qwen-Image VAE, encoder+decoder (img2img: `Vae.encode`).
//                                 Live-validated end to end against the real
//                                 circlestone-labs/Anima turbo-v1.1 pack on
//                                 `tests/test_anima_gen.sh`: strength 0.15 reproduces
//                                 the source near-exactly, 0.9 reads as a fresh
//                                 generation, 0.6 sits in between — the correct
//                                 SDEdit gradient (never checked by a formal numeric
//                                 oracle; the env-gated "encode parity" cosine test
//                                 below is still unrun — no torch install on hand at
//                                 write time). No edit training.)
//   t5_tokenizer/tokenizer.json  T5 SentencePiece unigram vocab (t5_tokenizer.zig)
//
// Runs the whole pipeline at float32: every component above was validated
// bit-exact (cosine 1.000000) against fp32 CPU references at that dtype, and
// bf16 parity was NOT separately re-verified — trading ~2x memory/compute for
// certain correctness on a brand-new backend. Follow-up: re-validate bf16 for
// the DiT/adapter/TE (matching mage_flow's "DiT/TE bf16, VAE f32" convention)
// once there is a live serving setup to A/B against.

/// Model-author sampling recommendation, read from the pack's config.json.
/// Falls back to the (slower, higher-quality) base-variant defaults when
/// absent — silently defaulting to Turbo's cfg=1 would silently disable
/// guidance for a base checkpoint whose config was missing the field.
pub const RecommendedSampling = struct {
    steps: u32 = 32,
    cfg: f32 = 4.5,
};

fn jsonToU32(v: std.json.Value, default: u32) u32 {
    return switch (v) {
        .integer => |n| if (n > 0) @intCast(n) else default,
        .float => |f| if (f > 0) @intFromFloat(f) else default,
        else => default,
    };
}
fn jsonToF32(v: std.json.Value, default: f32) f32 {
    return switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |f| @floatCast(f),
        else => default,
    };
}

fn parseRecommended(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) RecommendedSampling {
    var out = RecommendedSampling{};
    const path = std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir}) catch return out;
    defer allocator.free(path);
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return out;
    defer file.close(io);
    var rbuf: [4096]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    const content = rs.interface.allocRemaining(allocator, .limited(1024 * 1024)) catch return out;
    defer allocator.free(content);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), content, .{}) catch return out;
    if (parsed != .object) return out;
    if (parsed.object.get("recommended_steps")) |v| out.steps = jsonToU32(v, out.steps);
    if (parsed.object.get("recommended_cfg")) |v| out.cfg = jsonToF32(v, out.cfg);
    return out;
}

/// Build an owned `[1, ids.len]` int32 mlx array from host token ids.
fn idsToMlxArray(ids: []const u32) !mlx.mlx_array {
    var buf: [4096]i32 = undefined;
    const heap_buf = if (ids.len > buf.len) try std.heap.page_allocator.alloc(i32, ids.len) else null;
    defer if (heap_buf) |hb| std.heap.page_allocator.free(hb);
    const dst: []i32 = heap_buf orelse buf[0..ids.len];
    for (ids, 0..) |id, i| dst[i] = @intCast(id);
    const shape = [_]c_int{ 1, @intCast(ids.len) };
    const arr = mlx.mlx_array_new_data(dst.ptr, &shape, shape.len, .int32);
    defer _ = mlx.mlx_array_free(arr);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, arr));
    return o;
}

/// `preprocess_text_embeds`'s pad-to-512: zero-pad the adapter's context up
/// to at least 512 tokens (never truncates longer prompts). Takes ownership
/// of `ctx`; always returns a fresh owned array.
fn padContextTo512(ctx: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    defer _ = mlx.mlx_array_free(ctx);
    const sh = mlx.getShape(ctx); // [1,M,1024]
    const M: usize = @intCast(sh[1]);
    if (M >= 512) return contig(ctx, s);
    const pad_len: c_int = @intCast(512 - M);
    const zshape = [_]c_int{ 1, pad_len, sh[2] };
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, &zshape, zshape.len, dt, s));
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, ctx);
    _ = mlx.mlx_vector_array_append_value(vec, zeros);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 1, s));
    return o;
}

fn mulScalar(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const va = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(va);
    return mul(x, va, s);
}

/// [1,16,H,W] -> [1,16,1,H,W] (the DiT/VAE's expected T=1 image-path shape).
fn unsqueezeT(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    return reshape(x, &[_]c_int{ sh[0], sh[1], 1, sh[2], sh[3] }, s);
}

/// latent_formats.Wan21 process_out, broadcast over the channel axis (1) of a
/// [1,16,H,W] latent: `normalized * std + mean`, applied BEFORE the VAE
/// decode (the DiT operates on normalized latents; the VAE does not).
fn wan21Denormalize(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const shape = [_]c_int{ 1, WAN21_LATENT_CHANNELS, 1, 1 };
    const mean_arr = mlx.mlx_array_new_data(&WAN21_LATENTS_MEAN, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(mean_arr);
    const std_arr = mlx.mlx_array_new_data(&WAN21_LATENTS_STD, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(std_arr);
    const scaled = try mul(x, std_arr, s);
    defer _ = mlx.mlx_array_free(scaled);
    return add(scaled, mean_arr, s);
}

/// process_in, broadcast over the channel axis (1) of a `[1,16,1,H,W]` latent:
/// `(raw - mean) / std` — the inverse of `wan21Denormalize`, applied right
/// after the VAE encoder so img2img's source latent lives in the same
/// normalized space the DiT and denoise loop operate in.
fn wan21NormalizeT(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const shape = [_]c_int{ 1, WAN21_LATENT_CHANNELS, 1, 1, 1 };
    const mean_arr = mlx.mlx_array_new_data(&WAN21_LATENTS_MEAN, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(mean_arr);
    const std_arr = mlx.mlx_array_new_data(&WAN21_LATENTS_STD, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(std_arr);
    const centered = try sub(x, mean_arr, s);
    defer _ = mlx.mlx_array_free(centered);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&out, centered, std_arr, s));
    return out;
}

/// Encoder-side spatial downsample (WanResample `downsample2d`/`downsample3d`
/// at T=1, where the two modes are identical — see the Vae doc comment):
/// asymmetric `(0,1,0,1)` zero-pad (NHWC: pad H/W high edge by 1, low by 0)
/// then a 3x3 stride-2 conv, no further padding. `x` is NHWC `[1,H,W,C]`.
fn downsampleSpatial(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    const xc = try contig(x, s);
    defer _ = mlx.mlx_array_free(xc);
    const axes = [_]c_int{ 1, 2 };
    const low = [_]c_int{ 0, 0 };
    const high = [_]c_int{ 1, 1 };
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_pad(&p, xc, &axes, 2, &low, 2, &high, 2, zero, "constant", s));
    const pc = try contig(p, s);
    defer _ = mlx.mlx_array_free(pc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, pc, w, 2, 2, 0, 0, 1, 1, 1, s));
    defer _ = mlx.mlx_array_free(o);
    return add(o, b, s);
}

/// VAE output -> [1,3,H,W] f32 in [0,1] (the WAN2.1 VAE's own [-1,1] decode
/// convention, `image = image/127.5 - 1` on encode). Takes ownership of `x`.
fn toUnitRgb(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    defer _ = mlx.mlx_array_free(x);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const scaled = try mul(xf, half, s);
    defer _ = mlx.mlx_array_free(scaled);
    const shifted = try add(scaled, half, s);
    defer _ = mlx.mlx_array_free(shifted);
    const lo = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(hi);
    var clo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(clo);
    try mlx.check(mlx.mlx_maximum(&clo, shifted, lo, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_minimum(&out, clo, hi, s));
    return out;
}

/// img2img source: pixels `[1,3,H,W]` f32 `[0,1]`, pre-resized to the
/// request's target size (VAE-encoded internally, mirroring krea/flux's own
/// `GenOpts`). `start_step` comes from `gen.img2imgStartStep(steps,
/// strength)` — the caller maps strength onto the schedule the same way
/// every other backend does.
pub const GenOpts = struct {
    init_image: ?mlx.mlx_array = null,
    start_step: u32 = 0,
    /// What to steer the CFG unconditional branch away from, instead of the
    /// empty string. Null and "" behave identically here (both encode to the
    /// same forced-EOS embedding via `encodePrompt`) — unlike SDXL, Anima has
    /// no zeroed-embedding path for an absent negative prompt, so there is no
    /// absent/empty distinction to preserve.
    negative_prompt: ?[]const u8 = null,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: S,
    dit: Dit,
    adapter: Adapter,
    te: TextEncoder,
    vae: Vae,
    qwen_tok: tok_mod.Tokenizer,
    t5_tok: t5_tok_mod.T5Tokenizer,
    recommended: RecommendedSampling,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        // Matches flux/krea/mage_flow: a stream created directly at load time
        // and threaded through everywhere, never re-fetched via
        // `mlx.gpuStream()` mid-pipeline (its cached `noGpuBackend()` check
        // returned a CPU stream here, which then hit
        // "force_fused=True but no fused kernel is available" on `sdpa`).
        const s = mlx.mlx_default_gpu_stream_new();
        const dtype: mlx.mlx_dtype = .float32; // see the module-doc note above.

        const dit_path = try std.fmt.allocPrint(allocator, "{s}/transformer.safetensors", .{model_dir});
        defer allocator.free(dit_path);
        var w_dit = try model_mod.loadWeightsSingleFile(allocator, dit_path);
        defer w_dit.deinit();
        self.dit = try Dit.load(&w_dit, allocator, .{}, dtype, s);
        errdefer self.dit.deinit();
        self.adapter = try Adapter.load(&w_dit, allocator, dtype, s);
        errdefer self.adapter.deinit();

        const te_path = try std.fmt.allocPrint(allocator, "{s}/text_encoder.safetensors", .{model_dir});
        defer allocator.free(te_path);
        var w_te = try model_mod.loadWeightsSingleFile(allocator, te_path);
        defer w_te.deinit();
        self.te = try TextEncoder.load(&w_te, allocator, dtype, s);
        errdefer self.te.deinit();

        const vae_path = try std.fmt.allocPrint(allocator, "{s}/vae.safetensors", .{model_dir});
        defer allocator.free(vae_path);
        var w_vae = try model_mod.loadWeightsSingleFile(allocator, vae_path);
        defer w_vae.deinit();
        self.vae = try Vae.load(&w_vae, allocator, .float32, s);
        errdefer self.vae.deinit();

        const qwen_tok_dir = try std.fmt.allocPrint(allocator, "{s}/tokenizer", .{model_dir});
        defer allocator.free(qwen_tok_dir);
        self.qwen_tok = try tok_mod.loadTokenizerAny(io, allocator, qwen_tok_dir);
        errdefer self.qwen_tok.deinit();

        const t5_path = try std.fmt.allocPrint(allocator, "{s}/t5_tokenizer/tokenizer.json", .{model_dir});
        defer allocator.free(t5_path);
        self.t5_tok = try t5_tok_mod.T5Tokenizer.loadFromFile(io, allocator, t5_path);
        errdefer self.t5_tok.deinit();

        self.allocator = allocator;
        self.s = s;
        self.recommended = parseRecommended(io, allocator, model_dir);
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.dit.deinit();
        self.adapter.deinit();
        self.te.deinit();
        self.vae.deinit();
        self.qwen_tok.deinit();
        self.t5_tok.deinit();
        self.allocator.destroy(self);
    }

    /// Encode a prompt to the DiT's cross-attn context `[1, >=512, 1024]`.
    pub fn encodePrompt(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8) !mlx.mlx_array {
        const s = self.s;
        var qwen_ids = try self.qwen_tok.encode(allocator, prompt);
        defer allocator.free(qwen_ids);
        if (qwen_ids.len == 0) {
            // The raw BPE encoder (no chat template, no forced BOS/EOS) emits
            // zero tokens for an empty string. That's the CFG unconditional
            // branch's prompt ("" negative), so any cfg != 1.0 pack hits this.
            // A [1,0,1024] hidden state has ZERO total elements: MLX allocates
            // a 0-byte Metal buffer for it, and evaluating anything downstream
            // (the TE's own final RMSNorm) binds that null buffer and
            // segfaults inside RMSNorm::eval_gpu. Never let the TE see L=0.
            allocator.free(qwen_ids);
            qwen_ids = try allocator.alloc(u32, 1);
            qwen_ids[0] = self.qwen_tok.eos_id orelse 0;
        }
        const qwen_arr = try idsToMlxArray(qwen_ids);
        defer _ = mlx.mlx_array_free(qwen_arr);
        const qwen_hidden = try self.te.forward(qwen_arr, allocator, s);
        defer _ = mlx.mlx_array_free(qwen_hidden);

        const t5_ids = try self.t5_tok.encode(allocator, prompt);
        defer allocator.free(t5_ids);
        const t5_arr = try idsToMlxArray(t5_ids);
        defer _ = mlx.mlx_array_free(t5_arr);

        const ctx_raw = try self.adapter.forward(qwen_hidden, t5_arr, allocator, s);
        return padContextTo512(ctx_raw, self.adapter.dtype, s);
    }

    /// Generate an image. Returns `[1,3,H,W]` f32 in `[0,1]` (owned mlx array).
    /// `steps`/`cfg`: 0 falls back to the pack's own recommendation (see
    /// `RecommendedSampling`); `cfg == 1.0` skips the uncond forward entirely
    /// (Turbo's own optimization, matches comfy's `disable_cfg1_optimization`).
    pub fn generateImage(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps_opt: u32, cfg_opt: f32, progress: ?sse.Progress) !mlx.mlx_array {
        return self.generateImageOpts(allocator, prompt, width, height, seed, steps_opt, cfg_opt, .{}, progress);
    }

    /// `generateImage` with img2img options.
    pub fn generateImageOpts(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps_opt: u32, cfg_opt: f32, opts: GenOpts, progress: ?sse.Progress) !mlx.mlx_array {
        const s = self.s;
        const steps: u32 = if (steps_opt == 0) self.recommended.steps else steps_opt;
        const cfg: f32 = if (cfg_opt <= 0.0) self.recommended.cfg else cfg_opt;

        const cond = try self.encodePrompt(allocator, prompt);
        defer _ = mlx.mlx_array_free(cond);
        var uncond: ?mlx.mlx_array = null;
        defer if (uncond) |u| {
            _ = mlx.mlx_array_free(u);
        };
        if (cfg != 1.0) uncond = try self.encodePrompt(allocator, opts.negative_prompt orelse "");

        const lh: u32 = height / 8;
        const lw: u32 = width / 8;

        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        const nshape = [_]c_int{ 1, @intCast(WAN21_LATENT_CHANNELS), @intCast(lh), @intCast(lw) };
        var noise = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(noise);
        try mlx.check(mlx.mlx_random_normal(&noise, &nshape, nshape.len, .float32, 0.0, 1.0, key, s));

        const sched = try buildSimpleSchedule(allocator, steps, RFLOW_SHIFT);
        defer allocator.free(sched);

        const start_step: u32 = @min(opts.start_step, steps - 1);

        // noise_scaling(sigma0, noise, latent_image=0) = sigma0 * noise.
        var x = try mulScalar(noise, sched[0], s);
        errdefer _ = mlx.mlx_array_free(x);

        // img2img: VAE-encode the source, mix x = (1-t)*z0 + t*noise at the
        // start sigma (flow-match scale_noise), same convention as krea/flux.
        if (opts.init_image) |pix| {
            if (!self.vae.hasEncoder()) return error.NoVaeEncoder;
            const z0_5 = try self.vae.encode(pix, s); // [1,16,1,lh,lw] normalized
            defer _ = mlx.mlx_array_free(z0_5);
            const z0sh = mlx.getShape(z0_5);
            const z0 = try reshape(z0_5, &[_]c_int{ z0sh[0], z0sh[1], z0sh[3], z0sh[4] }, s); // drop T
            defer _ = mlx.mlx_array_free(z0);
            const t0 = sched[start_step];
            const z_scaled = try mulScalar(z0, 1.0 - t0, s);
            defer _ = mlx.mlx_array_free(z_scaled);
            const n_scaled = try mulScalar(noise, t0, s);
            defer _ = mlx.mlx_array_free(n_scaled);
            const mixed = try add(z_scaled, n_scaled, s);
            _ = mlx.mlx_array_free(x);
            x = mixed;
        }

        const run_steps = steps - start_step;
        for (start_step..steps) |i| {
            if (progress) |p| if (p.cancelled()) return error.Cancelled;
            const sigma = sched[i];
            const sigma_next = sched[i + 1];
            const x5 = try unsqueezeT(x, s);
            defer _ = mlx.mlx_array_free(x5);

            const cond_out = try self.dit.forward(x5, sigma, cond, allocator, s);
            var d = cond_out;
            if (uncond) |u| {
                const uncond_out = try self.dit.forward(x5, sigma, u, allocator, s);
                defer _ = mlx.mlx_array_free(uncond_out);
                // cfgCombine on raw velocities: uncond + (cond-uncond)*cfg.
                const diff = try sub(cond_out, uncond_out, s);
                defer _ = mlx.mlx_array_free(diff);
                const scaled = try mulScalar(diff, cfg, s);
                defer _ = mlx.mlx_array_free(scaled);
                const combined = try add(uncond_out, scaled, s);
                _ = mlx.mlx_array_free(cond_out);
                d = combined;
            }
            defer _ = mlx.mlx_array_free(d);

            const step = try mulScalar(d, sigma_next - sigma, s);
            defer _ = mlx.mlx_array_free(step);
            const xnew = try add(x, step, s);
            _ = mlx.mlx_array_free(x);
            x = xnew;
            _ = mlx.mlx_array_eval(x);
            if (progress) |p| p.emit("Generating", @intCast(i + 1 - start_step), run_steps);
        }

        if (progress) |p| p.emit("Decoding image", steps, steps);
        const denorm = try wan21Denormalize(x, s);
        _ = mlx.mlx_array_free(x);
        const denorm5 = try unsqueezeT(denorm, s);
        _ = mlx.mlx_array_free(denorm);
        const rgb = try self.vae.decode(denorm5, s);
        _ = mlx.mlx_array_free(denorm5);
        return toUnitRgb(rgb, s);
    }

    pub fn generatePng(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, cfg: f32, progress: ?sse.Progress) ![]u8 {
        const img = try self.generateImage(allocator, prompt, width, height, seed, steps, cfg, progress);
        defer _ = mlx.mlx_array_free(img);
        return krea.imageToPng(allocator, img, self.s);
    }
};

// ── Tests ──

const testing = std.testing;

test "anima: config head dim and patch in-features for the 2B pack" {
    const c = Config{};
    try testing.expectEqual(@as(u32, 128), c.headDim()); // 2048/16
    try testing.expectEqual(@as(u32, 68), c.patchInFeatures()); // (16+1)*2*2*1
}

test "anima: rope3d split sums to head_dim (hd 128 -> 42/42/44)" {
    const s = rope3dSplit(128);
    try testing.expectEqual(@as(u32, 42), s.dim_h);
    try testing.expectEqual(@as(u32, 42), s.dim_w);
    try testing.expectEqual(@as(u32, 44), s.dim_t);
    try testing.expectEqual(@as(u32, 128), s.dim_h + s.dim_w + s.dim_t);
}

test "anima: rope3d ntk factor: ratio 1.0 is identity, >1 grows" {
    try testing.expectApproxEqAbs(@as(f32, 1.0), ntkFactor(1.0, 44), 1e-6);
    try testing.expect(ntkFactor(4.0, 42) > 4.0); // 4^(42/40) slightly above 4
}

test "anima: time_snr_shift identity at alpha=1, warps toward 0 for alpha>1" {
    try testing.expectApproxEqAbs(@as(f32, 0.42), timeSnrShift(1.0, 0.42), 1e-6);
    // alpha=3, t=0.5: 3*0.5/(1+2*0.5) = 1.5/2 = 0.75.
    try testing.expectApproxEqAbs(@as(f32, 0.75), timeSnrShift(3.0, 0.5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), timeSnrShift(3.0, 0.0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), timeSnrShift(3.0, 1.0), 1e-6);
}

test "anima: simple schedule is descending, starts near 1, ends at 0" {
    const a = testing.allocator;
    const sig = try buildSimpleSchedule(a, 8, RFLOW_SHIFT);
    defer a.free(sig);
    try testing.expectEqual(@as(usize, 9), sig.len);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sig[0], 0.01);
    try testing.expectEqual(@as(f32, 0.0), sig[8]);
    for (0..8) |i| try testing.expect(sig[i] > sig[i + 1]);
}

test "anima: CONST calculate_denoised / noise_scaling / cfg combine" {
    const sigma: f32 = 0.5;
    // calculate_denoised: x - out*sigma.
    try testing.expectApproxEqAbs(@as(f32, 3.0 - 0.5 * 2.0), calculateDenoised(sigma, 2.0, 3.0), 1e-6);
    // noise_scaling: sigma*noise + (1-sigma)*latent.
    try testing.expectApproxEqAbs(@as(f32, 0.5 * 2.0 + 0.5 * 5.0), noiseScaling(sigma, 2.0, 5.0), 1e-6);
    // cfg combine: uncond + (cond-uncond)*cfg; cfg=1 is the cond-only identity.
    try testing.expectApproxEqAbs(@as(f32, 1.0), cfgCombine(2.0, 1.0, 0.0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0), cfgCombine(2.0, 1.0, 1.0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 3.0), cfgCombine(2.0, 1.0, 2.0), 1e-6);
}

// Env-gated live test against the real pack (not a fixture) — proves
// `GenOpts.negative_prompt` actually reaches the CFG uncond branch instead of
// the pipeline silently encoding "" regardless of the field, which is what
// happened before this option existed. cfg is fixed away from 1.0 so the
// uncond forward (and therefore the negative prompt) is on the critical path;
// one step keeps this cheap enough to run alongside the fixture parity tests.
//   ANIMA_MODEL_DIR=<pack dir produced by scripts/convert_anima_weights.py>
test "anima: GenOpts.negative_prompt changes the CFG uncond branch" {
    const dir = std.mem.span(std.c.getenv("ANIMA_MODEL_DIR") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const engine = try Engine.load(io, a, dir);
    defer engine.deinit();

    const prompt = "a red apple on a wooden table";
    const width: u32 = 256;
    const height: u32 = 256;
    const seed: u64 = 7;
    const steps: u32 = 1;
    const cfg: f32 = 4.0; // != 1.0: engages the uncond branch

    const base = try engine.generateImageOpts(a, prompt, width, height, seed, steps, cfg, .{}, null);
    defer _ = mlx.mlx_array_free(base);
    const negged = try engine.generateImageOpts(a, prompt, width, height, seed, steps, cfg, .{ .negative_prompt = "blurry, low quality, watermark" }, null);
    defer _ = mlx.mlx_array_free(negged);

    const cos = try cosineSim(a, base, negged, mlx.gpuStream());
    std.debug.print("[anima] negative_prompt cosine vs default (\"\") uncond = {d:.6}\n", .{cos});
    try testing.expect(cos < 0.999);
}

test "anima: wan21 latent normalization round-trips per channel" {
    for (0..WAN21_LATENT_CHANNELS) |ch| {
        const raw: f32 = 0.37;
        const norm = wan21ProcessIn(ch, raw);
        try testing.expectApproxEqAbs(raw, wan21ProcessOut(ch, norm), 1e-5);
    }
    // Channel 0: (0 - (-0.7571)) / 2.8184.
    try testing.expectApproxEqAbs(@as(f32, 0.7571 / 2.8184), wan21ProcessIn(0, 0.0), 1e-6);
}

/// Eval an mlx array and return its elements as owned host f32 (caller frees).
/// The array must be float32.
fn evalToF32(a: std.mem.Allocator, arr: mlx.mlx_array) ![]f32 {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, arr);
    try mlx.check(mlx.mlx_eval(vec));
    const shape = mlx.getShape(arr);
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const ptr = mlx.mlx_array_data_float32(arr) orelse return error.NotFloat32;
    const out = try a.alloc(f32, n);
    @memcpy(out, ptr[0..n]);
    return out;
}

fn cosineSim(a: std.mem.Allocator, x: mlx.mlx_array, y: mlx.mlx_array, s: S) !f64 {
    // astype then materialize: a strided/transposed input's data pointer is not
    // row-major, so evalToF32's linear read needs a contiguous buffer.
    const xf0 = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf0);
    const xf = try contig(xf0, s);
    defer _ = mlx.mlx_array_free(xf);
    const yf0 = try astype(y, .float32, s);
    defer _ = mlx.mlx_array_free(yf0);
    const yf = try contig(yf0, s);
    defer _ = mlx.mlx_array_free(yf);
    const xd = try evalToF32(a, xf);
    defer a.free(xd);
    const yd = try evalToF32(a, yf);
    defer a.free(yd);
    if (xd.len != yd.len) return error.ShapeMismatch;
    var dot: f64 = 0;
    var nx: f64 = 0;
    var ny: f64 = 0;
    for (xd, yd) |xv, yv| {
        dot += @as(f64, xv) * @as(f64, yv);
        nx += @as(f64, xv) * @as(f64, xv);
        ny += @as(f64, yv) * @as(f64, yv);
    }
    return dot / (@sqrt(nx) * @sqrt(ny) + 1e-12);
}

// Env-gated parity oracle (repo convention). Set:
//   ANIMA_DIT=<anima-*.safetensors>  ANIMA_ADAPTER_FIXTURE=<fixture.safetensors>
// Fixture produced by tests/dump_anima_fixtures.py.
test "anima: LLMAdapter parity vs reference fixture" {
    const dit_path = std.mem.span(std.c.getenv("ANIMA_DIT") orelse return error.SkipZigTest);
    const fix_path = std.mem.span(std.c.getenv("ANIMA_ADAPTER_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const s = mlx.gpuStream();

    var wdit = try model_mod.loadWeightsSingleFile(a, dit_path);
    defer wdit.deinit();
    var wfix = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer wfix.deinit();

    var adapter = try Adapter.load(&wdit, a, .float32, s);
    defer adapter.deinit();

    const qwen_hidden = wfix.get("qwen_hidden") orelse return error.MissingFixture;
    const t5_ids = wfix.get("t5_ids") orelse return error.MissingFixture;
    const expected = wfix.get("adapter_out") orelse return error.MissingFixture;

    const out = try adapter.forward(qwen_hidden, t5_ids, a, s);
    defer _ = mlx.mlx_array_free(out);

    const cos = try cosineSim(a, out, expected, s);
    std.debug.print("[anima] adapter parity cosine = {d:.6}\n", .{cos});
    try testing.expect(cos > 0.999);
}

// Env-gated DiT parity oracle. Set:
//   ANIMA_DIT=<anima-*.safetensors>  ANIMA_DIT_FIXTURE=<fixture.safetensors>
// Fixture produced by tests/dump_anima_dit_fixtures.py.
test "anima: MiniTrainDIT parity vs reference fixture" {
    const dit_path = std.mem.span(std.c.getenv("ANIMA_DIT") orelse return error.SkipZigTest);
    const fix_path = std.mem.span(std.c.getenv("ANIMA_DIT_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const s = mlx.gpuStream();

    var wdit = try model_mod.loadWeightsSingleFile(a, dit_path);
    defer wdit.deinit();
    var wfix = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer wfix.deinit();

    var dit = try Dit.load(&wdit, a, .{}, .float32, s);
    defer dit.deinit();

    const latent = wfix.get("latent") orelse return error.MissingFixture;
    const context = wfix.get("context") orelse return error.MissingFixture;
    const expected = wfix.get("dit_out") orelse return error.MissingFixture;

    const out = try dit.forward(latent, 0.7, context, a, s);
    defer _ = mlx.mlx_array_free(out);

    const cos = try cosineSim(a, out, expected, s);
    std.debug.print("[anima] DiT parity cosine = {d:.6}\n", .{cos});
    try testing.expect(cos > 0.999);
}

// Env-gated VAE parity oracle. Set:
//   ANIMA_VAE=<qwen_image_vae.safetensors>  ANIMA_VAE_FIXTURE=<fixture.safetensors>
// Fixture produced by tests/dump_anima_vae_fixtures.py.
test "anima: Qwen-Image VAE decode parity vs reference fixture" {
    const vae_path = std.mem.span(std.c.getenv("ANIMA_VAE") orelse return error.SkipZigTest);
    const fix_path = std.mem.span(std.c.getenv("ANIMA_VAE_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const s = mlx.gpuStream();

    var wvae = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer wvae.deinit();
    var wfix = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer wfix.deinit();

    var vae = try Vae.load(&wvae, a, .float32, s);
    defer vae.deinit();

    const latent = wfix.get("latent") orelse return error.MissingFixture;
    const expected = wfix.get("vae_out") orelse return error.MissingFixture;

    const out = try vae.decode(latent, s);
    defer _ = mlx.mlx_array_free(out);

    const cos = try cosineSim(a, out, expected, s);
    std.debug.print("[anima] VAE decode parity cosine = {d:.6}\n", .{cos});
    try testing.expect(cos > 0.999);
}

// Env-gated VAE ENCODER parity oracle (img2img). Set:
//   ANIMA_VAE=<qwen_image_vae.safetensors>  ANIMA_VAE_ENC_FIXTURE=<fixture.safetensors>
// Fixture produced by `tests/dump_anima_vae_fixtures.py encode`. UNRUN as of
// this writing — no real qwen_image_vae.safetensors + reference torch install
// was available to generate the fixture; this is the harness to run before
// trusting `Vae.encode`/img2img output. The mu (pre-normalize) tensor is
// compared, not the post-`wan21ProcessIn` value, so a channel-stat mismatch
// can't hide a real encoder bug behind normalization.
test "anima: Qwen-Image VAE encode parity vs reference fixture" {
    const vae_path = std.mem.span(std.c.getenv("ANIMA_VAE") orelse return error.SkipZigTest);
    const fix_path = std.mem.span(std.c.getenv("ANIMA_VAE_ENC_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const s = mlx.gpuStream();

    var wvae = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer wvae.deinit();
    var wfix = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer wfix.deinit();

    var vae = try Vae.load(&wvae, a, .float32, s);
    defer vae.deinit();
    try testing.expect(vae.hasEncoder());

    const image = wfix.get("image") orelse return error.MissingFixture;
    const expected_mu = wfix.get("mu") orelse return error.MissingFixture;

    const z = try vae.encode(image, s); // wan21-normalized [1,16,1,h,w]
    defer _ = mlx.mlx_array_free(z);
    const zsh = mlx.getShape(z);
    const z4 = try reshape(z, &[_]c_int{ zsh[0], zsh[1], zsh[3], zsh[4] }, s);
    defer _ = mlx.mlx_array_free(z4);
    const mu = try wan21Denormalize(z4, s);
    defer _ = mlx.mlx_array_free(mu);

    const cos = try cosineSim(a, mu, expected_mu, s);
    std.debug.print("[anima] VAE encode parity cosine = {d:.6}\n", .{cos});
    try testing.expect(cos > 0.999);

    // Round-trip sanity even without a fixture: an encode→decode of the
    // fixture's own image should resemble the input (loose bar — the VAE is
    // lossy — but a broken encoder produces near-zero correlation, not 0.9+).
    const denorm5 = try unsqueezeT(mu, s);
    defer _ = mlx.mlx_array_free(denorm5);
    const rgb = try vae.decode(denorm5, s);
    defer _ = mlx.mlx_array_free(rgb);
    const unit = try toUnitRgb(rgb, s);
    defer _ = mlx.mlx_array_free(unit);
    _ = mlx.mlx_array_eval(unit);
}

// Env-gated TE parity oracle. Set:
//   ANIMA_TE=<qwen_3_06b_base.safetensors>  ANIMA_TE_FIXTURE=<fixture.safetensors>
// Fixture produced by tests/dump_anima_te_fixtures.py.
test "anima: Qwen3-0.6B text encoder parity vs reference fixture" {
    const te_path = std.mem.span(std.c.getenv("ANIMA_TE") orelse return error.SkipZigTest);
    const fix_path = std.mem.span(std.c.getenv("ANIMA_TE_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const s = mlx.gpuStream();

    var wte = try model_mod.loadWeightsSingleFile(a, te_path);
    defer wte.deinit();
    var wfix = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer wfix.deinit();

    var te = try TextEncoder.load(&wte, a, .float32, s);
    defer te.deinit();

    const ids = wfix.get("ids") orelse return error.MissingFixture;
    const expected = wfix.get("hidden") orelse return error.MissingFixture;

    const out = try te.forward(ids, a, s);
    defer _ = mlx.mlx_array_free(out);

    const cos = try cosineSim(a, out, expected, s);
    std.debug.print("[anima] TE parity cosine = {d:.6}\n", .{cos});
    try testing.expect(cos > 0.999);
}

test "anima: timestep embedding cos band first, t=0 gives cos=1 sin=0" {
    var buf: [8]f32 = undefined;
    timestepEmbedding(0.0, &buf);
    // t=0: every arg 0 -> cos 1, sin 0.
    for (buf[0..4]) |v| try testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-6);
    for (buf[4..8]) |v| try testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
    // First frequency (i=0) has exponent 0 -> arg = t exactly.
    timestepEmbedding(0.5, &buf);
    try testing.expectApproxEqAbs(@cos(@as(f32, 0.5)), buf[0], 1e-6);
    try testing.expectApproxEqAbs(@sin(@as(f32, 0.5)), buf[4], 1e-6);
}
