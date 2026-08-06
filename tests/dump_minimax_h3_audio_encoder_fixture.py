#!/usr/bin/env python3
"""Dump MiniMax H3 audio-VAE ENCODER parity fixtures — the reference EXECUTES.

ref2va conditions on reference AUDIO, so the encode side of the audio VAE is
needed. It is not the decoder run backwards, and two things about it are easy
to assume wrongly:

  * the encoder uses PLAIN `Snake1d`. The anti-aliased `Activation1d`
    (upsample -> act -> downsample, with its kaiser-sinc `DownSample1d`) is a
    BigVGAN *decoder* component and appears nowhere on this path.
  * after the conv stack there is an `AttnProjection`: causal attention whose
    output is mean-pooled over heads and then average-pooled from head_dim
    down to the 32-wide latent, plus a GeGLU MLP with TWO stacked LayerNorms
    (`norm2` outside, `mlp.norm` inside).

The reference classes are reproduced here VERBATIM from
`comfy/ldm/minimax/audio_vae.py` (only `comfy.ops.*` swapped for `nn.*`, which
`disable_weight_init` is), and run against the real checkpoint weights — so a
green Zig test means agreement with the reference's own output.

Usage:
    uv run --with torch --with numpy --with safetensors \
        tests/dump_minimax_h3_audio_encoder_fixture.py \
        --vae ~/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit/audio_vae.safetensors \
        --out ~/claude-tmp/h3-build/minimax_h3_audio_enc_fixture.safetensors
"""

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

SAMPLE_RATE = 32000
HOP_LENGTH = 800
VAE_LATENT_CHANNELS = 32
LATENT_DIM = 2048


def snake(x, alpha, beta):
    t = torch.sin(alpha * x)
    return t.mul(t).mul((beta + 1e-9).reciprocal()).add(x)


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha, self.alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return y + x


class EncoderBlock(nn.Module):
    def __init__(self, dim=16, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride,
                      padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model=64, strides=(2, 4, 4, 5, 5), d_latent=2048):
        super().__init__()
        block = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block += [EncoderBlock(d_model, stride=stride)]
        block += [Snake1d(d_model), nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class GeGluMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)).mul(self.w1(x)))


class CausalAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.head_dim = in_dim // num_heads
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.empty(in_dim))
        self.v_bias = nn.Parameter(torch.empty(in_dim))
        self.register_buffer("zero_k_bias", torch.empty(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = F.linear(x, weight=self.qkv.weight,
                       bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = F.adaptive_avg_pool1d(torch.mean(x, dim=1), self.out_dim)
        return self.proj(x)


class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(in_features=out_dim, hidden_features=int(out_dim * mlp_ratio))

    def forward(self, x):
        x = self.proj(self.norm3(x)).add(self.attn(self.norm1(x)))
        return x.add(self.mlp(self.norm2(x)))


def synth_stereo(seconds, seed):
    """A deterministic stereo signal whose two CHANNELS DIFFER.

    Identical channels would let a port that collapses stereo into the feature
    axis — the exact bug the decode side documents — pass unnoticed, the same
    way a static video hides a temporal error.
    """
    n = int(seconds * SAMPLE_RATE)
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
    left = (torch.sin(2 * math.pi * (220 + 40 * seed) * t)
            * (0.5 + 0.5 * torch.sin(2 * math.pi * 1.5 * t)))
    right = (torch.sin(2 * math.pi * (330 + 40 * seed) * t + 0.7)
             * (0.5 + 0.5 * torch.cos(2 * math.pi * 0.9 * t)))
    return torch.stack([left, right]).mul(0.8).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    enc = Encoder(64, (2, 4, 4, 5, 5), LATENT_DIM)
    pre_block = AttnProjection(LATENT_DIM, VAE_LATENT_CHANNELS, num_heads=8)
    mean_proj = nn.Conv1d(VAE_LATENT_CHANNELS, VAE_LATENT_CHANNELS, 1)

    tensors = {}
    with safe_open(args.vae, framework="pt") as f:
        for k in f.keys():
            if (k.startswith("encoder.") or k.startswith("pre_block.")
                    or k.startswith("mean_proj.") or k in ("latents_mean", "latents_std")):
                tensors[k] = f.get_tensor(k).float()

    enc.load_state_dict({k[len("encoder."):]: v for k, v in tensors.items()
                         if k.startswith("encoder.")})
    pre_block.load_state_dict({k[len("pre_block."):]: v for k, v in tensors.items()
                               if k.startswith("pre_block.")})
    mean_proj.load_state_dict({k[len("mean_proj."):]: v for k, v in tensors.items()
                               if k.startswith("mean_proj.")})
    for m in (enc, pre_block, mean_proj):
        m.eval()

    lm = tensors["latents_mean"].view(1, -1, 1)
    ls = tensors["latents_std"].view(1, -1, 1)

    def encode(waveform):
        """The reference's `MiniMaxH3AudioVAE.encode`, for a [2, L] input."""
        with torch.no_grad():
            w = waveform.unsqueeze(0)                    # [1, 2, L]
            b, s, length = w.shape
            right_pad = math.ceil(length / HOP_LENGTH) * HOP_LENGTH - length
            w = F.pad(w, (0, right_pad))
            x = w.reshape(b * s, 1, -1)
            x = enc(x)
            x = pre_block(x.transpose(1, 2)).transpose(1, 2)
            z = mean_proj(x)
            z = (z - lm) / ls
            return z.reshape(b, s, z.shape[1], z.shape[2]).permute(0, 2, 1, 3)

    out = {}
    # 0.5 s is a whole number of latent frames (20); 0.31 s is NOT, so the
    # right-pad-to-a-whole-frame branch is exercised rather than assumed.
    for name, secs, seed in (("a_exact", 0.5, 1), ("a_ragged", 0.31, 2)):
        wav = synth_stereo(secs, seed)
        lat = encode(wav)
        out[name] = wav
        out["latent_" + name] = lat.float().contiguous()
        print(f"{name}: {tuple(wav.shape)} -> {tuple(lat.shape)} "
              f"mean {lat.mean().item():+.4f} std {lat.std().item():.4f}")

    save_file(out, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
