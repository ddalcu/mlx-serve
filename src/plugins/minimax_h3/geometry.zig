//! MiniMax-H3's geometry and the request arithmetic its bills share with the forward, MLX-free: the frame ladder,
//! the DiT / text-encoder / video-VAE geometry, the speed recipe.

const std = @import("std");
const env = @import("../../env.zig");

pub const FPS: u32 = 24;
pub const AUDIO_LATENT_FPS: u32 = 40;
/// Spatial compression: VAE 16x, then the DiT's 2x2 patch → 32x effective.
pub const VAE_SPATIAL: u32 = 16;
pub const PATCH_H: u32 = 2;
pub const PATCH_W: u32 = 2;
/// The video VAE's ratios (16x spatial, 4x temporal).
pub const VAE_RATIO: u32 = 16;
pub const VAE_RATIO_T: u32 = 4;

/// Snap UP to the model's 17k+5 grid. 40 -> 56, 200 -> 209, 363 -> 379.
pub fn alignFrameCount(n: u32) u32 {
    // Saturates near the u32 limit, where no ladder rung exists; the transport cap refuses it.
    return n +| (22 - n % 17) % 17;
}

/// Latent frames for an aligned source-frame count.
pub fn videoLatentT(frame_count: u32) u32 {
    if (frame_count <= 5) return 2;
    return ((frame_count - 5) / 17) * 5 + 2;
}

pub const TemporalShape = struct { frame_count: u32, latent_t: u32, audio_t: u32 };

/// Aligned frame count + its latent-frame and audio-latent counts.
pub fn temporalShape(length: u32) TemporalShape {
    // 2^24 frames is past what the transport cap admits at any canvas, and keeps audio_t in range.
    const frame_count = alignFrameCount(std.math.clamp(length, 5, 1 << 24));
    const fc: f64 = @floatFromInt(frame_count);
    // audio_t = round(frame_count / 24 * 40) = round(frame_count * 5 / 3). The
    // reference uses Python's banker's rounding, but frame_count*5/3 can only
    // have fraction 0, 1/3 or 2/3 — never exactly 1/2 — so the two agree and
    // plain rounding is safe here. adaptCanvas is NOT so lucky; see below.
    const audio_t = @round(fc / @as(f64, @floatFromInt(FPS)) * @as(f64, @floatFromInt(AUDIO_LATENT_FPS)));
    return .{
        .frame_count = frame_count,
        .latent_t = videoLatentT(frame_count),
        .audio_t = @intFromFloat(audio_t),
    };
}

pub const Config = struct {
    hidden_size: u32 = 5376,
    num_layers: u32 = 50,
    token_refiner_num_layers: u32 = 2,
    num_attention_heads: u32 = 56,
    attention_head_dim: u32 = 128,
    ffn_hidden_size: u32 = 14336,
    latents_dim: u32 = 24,
    audio_latents_dim: u32 = 32,
    text_dim: u32 = 5120,
    timestep_input_dim: u32 = 256,
    time_embed_hidden_size: u32 = 5376,
    time_embed_dim: u32 = 2688,
    rope_inv_freq_len: u32 = 16,
    norm_eps: f32 = 1e-5,
    qk_norm_eps: f32 = 1e-5,
    final_norm_eps: f32 = 1e-5,

    pub fn innerDim(self: Config) u32 {
        return self.num_attention_heads * self.attention_head_dim;
    }
    /// Rotary width: 3 axes x inv_freq_len, doubled by the split-half pairing.
    /// 96 of head_dim 128 — the remaining 32 dims per head pass through.
    pub fn rotDim(self: Config) u32 {
        return self.rope_inv_freq_len * 3 * 2;
    }
    pub fn videoPatchDim(self: Config) u32 {
        return self.latents_dim * PATCH_H * PATCH_W;
    }
};

pub const TeConfig = struct {
    hidden: c_int = 5120,
    heads: c_int = 64,
    kv_heads: c_int = 8,
    head_dim: c_int = 128,
    intermediate: c_int = 25600,
    layers: u32 = 50,
    theta: f64 = 5_000_000.0,
    eps: f32 = 1e-6,
};

/// The pack's geometry (`config.json`'s `transformer`, `text_encoder` and `video_vae` sections; the published values
/// where a key is absent).
pub const Geometry = struct {
    dit: Config = .{},
    te: TeConfig = .{},
    vae: DecCfg = .{},
};

pub const SpeedConfig = struct { step_cache: f64, bcast_k: u32 };

/// The eyeball verdict on the same-seed capstone pair made the
/// fast recipe (velocity cache 0.05 + attention broadcast k=2 — 2.83x at the
/// trained config) DEFAULT-ON, with `"fast": false` on the request as the
/// off switch for final renders ("non fast just a smidge better").
///
/// Turbo and the fast recipe are ALTERNATIVES, not a stack: at 4-8 steps the
/// velocity cache's slow-drift premise is false (every step is a giant leap)
/// and the broadcast gates cover most of the schedule anyway, so `turbo`
/// forces both off regardless of `fast`.
/// A PDD student is `distilled` the same way.
pub fn resolveSpeed(fast: ?bool, distilled: bool) SpeedConfig {
    const want_fast = !distilled and (fast orelse true);
    return .{
        .step_cache = if (want_fast) 0.05 else 0,
        .bcast_k = if (want_fast) 2 else 0,
    };
}

/// A registered variable's value; unset and empty both read as null.
pub fn envStr(comptime name: []const u8) ?[]const u8 {
    const raw = env.named(name) orelse return null;
    const slice = std.mem.sliceTo(raw, 0);
    return if (slice.len == 0) null else slice;
}

/// Decoder geometry (ViT3DDecoder defaults, confirmed against the checkpoint).
pub const DecCfg = struct {
    layers: u32 = 36,
    heads: u32 = 32,
    head_dim: u32 = 64,
    in_channels: u32 = 24,
    out_channels: u32 = 3,
    patch_size: u32 = VAE_RATIO,
    patch_size_t: u32 = VAE_RATIO_T,
    rope_theta: f64 = 100.0,
    rope_dim_ratio: f64 = 0.75,
    num_register_tokens: u32 = 4,
    eps: f32 = 1e-5,

    pub fn dim(self: DecCfg) u32 {
        return self.heads * self.head_dim;
    }
    /// RotaryEmbeddingND(dim_head * rope_dim_ratio) with n_dim=3: the inv_freq
    /// arange step is 2*n_dim/dim, so the count is ceil(dim / (2*n_dim)).
    pub fn ropeFreqs(self: DecCfg) u32 {
        const rd: u32 = @intFromFloat(@as(f64, @floatFromInt(self.head_dim)) * self.rope_dim_ratio);
        return (rd + 5) / 6;
    }
    /// Rotated width per head: 3 axes x freqs x 2 (split-half pairing).
    pub fn rotDim(self: DecCfg) u32 {
        return self.ropeFreqs() * 3 * 2;
    }
    pub fn outPatchDim(self: DecCfg) u32 {
        return self.out_channels * self.patch_size_t * self.patch_size * self.patch_size;
    }
};
