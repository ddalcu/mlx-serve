#!/usr/bin/env python3
"""Dump MiniMax H3 video-VAE ENCODER parity fixtures — the reference EXECUTES.

Unlike the DiT (whose reference block calls comfy_kitchen CUDA kernels and can
only be transcribed), the conv encoder is plain torch: CausalConv3d, per-frame
GroupNorm, reflect padding, strided downsamples. This script reproduces
`comfy/ldm/minimax/vae.py`'s encoder VERBATIM for the single-frame (T==1) path
and runs it against the real checkpoint weights, so a green Zig test proves
agreement with the reference's own output, not with a transcription.

The T==1 causal semantics: the front temporal pads are all zeros, so a full
zero-pad + conv3d is bit-equivalent to the reference's `autopad="causal_zero"`
tap truncation (zero frames contribute nothing; the bias is applied once
either way). We implement the zero-pad form because it needs no comfy.ops.

Two cases are dumped:
  x        [1,3,1,128,128]  — single tile (128 <= 256)
  x_tiled  [1,3,1,384,384]  — exercises the reference's own tiled_encode
                              (split [0,128] overlap 128, latent blend + trim)

Inputs are in [-1, 1] like the server's request path; the dump applies the
same pixel normalization `encode()` does.

Usage:
    uv run --with torch --with numpy --with safetensors \
        tests/dump_minimax_h3_vae_encoder_fixture.py \
        --vae ~/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit/video_vae.safetensors \
        --out ~/claude-tmp/h3-build/minimax_h3_vae_enc_fixture.safetensors
"""

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN_KEY = "latents_mean"
LATENTS_STD_KEY = "latents_std"

TILE_SIZE = 256
TILE_OVERLAP_MIN = 64
VAE_RATIO = 16


class CausalConv3d(nn.Conv3d):
    # Reflect spatial padding, causal (zeros, front-only) temporal padding.
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

    def forward(self, x):
        if sum(self.causal_padding) == 0:
            return super().forward(x)
        x = F.pad(x, (self.causal_padding[2], self.causal_padding[2],
                      self.causal_padding[1], self.causal_padding[1], 0, 0), mode="reflect")
        # T==1 path: full zero front-pad == the reference's tap truncation.
        x = F.pad(x, (0, 0, 0, 0, self.causal_padding[0] * 2, 0), mode="constant")
        return super().forward(x)


class TemporalIsolatedGroupNorm(nn.GroupNorm):
    def forward(self, x):
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
            x = super().forward(x)
            return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return super().forward(x)


def group_norm_3d(num_channels):
    return TemporalIsolatedGroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3,
                                 padding=(1, 0, 0), stride=(time_stride, space_stride, space_stride))

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = group_norm_3d(in_channels)
        self.norm2 = group_norm_3d(out_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return h + x


class EncoderFCN3D(nn.Module):
    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels, double_z=True):
        super().__init__()
        self.num_levels = len(ch_mult)
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * self.num_levels
        self.num_res_blocks = num_res_blocks
        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        block_out = block_mid
        self.conv_in = CausalConv3d(in_channels, block_in[0], kernel_size=3, padding=1)
        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            down = nn.Module()
            down.block = nn.ModuleList()
            for i in range(self.num_res_blocks[i_level]):
                down.block.append(ResnetBlock3D(
                    in_channels=block_in[i_level] if i == 0 else block_mid[i_level],
                    out_channels=block_mid[i_level]))
            if space_down[i_level] * time_down[i_level] > 1:
                down.downsample = Downsample3D(block_mid[i_level], block_out[i_level],
                                               time_stride=time_down[i_level], space_stride=space_down[i_level])
            self.down.append(down)
        self.norm_out = group_norm_3d(block_out[-1])
        self.conv_out = CausalConv3d(block_out[-1], 2 * z_channels if double_z else z_channels,
                                     kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


def split_tiles(input_len):
    if TILE_SIZE >= input_len:
        return [0], [input_len], []
    N = math.ceil(input_len / TILE_SIZE)
    while True:
        overlaps = [TILE_OVERLAP_MIN] * (N - 1)
        remaining = TILE_SIZE * N - sum(overlaps) - input_len
        if remaining < 0:
            N += 1
        else:
            break
    for i in range(remaining // VAE_RATIO):
        overlaps[i % (N - 1)] += VAE_RATIO
    starts = [0]
    for i in range(N - 1):
        starts.append(starts[-1] + TILE_SIZE - overlaps[i])
    return starts, [TILE_SIZE] * N, overlaps


def blend(a, b, blend_extent, dim):
    blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
    positions = torch.arange(blend_extent, dtype=b.dtype)
    weight_a = 1 - positions / blend_extent
    weight_b = positions / blend_extent
    shape = [1] * a.ndim
    shape[dim] = blend_extent
    weight_a = weight_a.view(shape)
    weight_b = weight_b.view(shape)
    slice_a = [slice(None)] * a.ndim
    slice_a[dim] = slice(-blend_extent, None)
    slice_b = [slice(None)] * b.ndim
    slice_b[dim] = slice(0, blend_extent)
    blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
    if blend_extent < b.shape[dim]:
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)
    return blended


def tiled_encode(encode_moments, x):
    height, width = x.shape[-2], x.shape[-1]
    y_idx, y_len, y_overlap = split_tiles(height)
    x_idx, x_len, x_overlap = split_tiles(width)
    rows = []
    for i_pos, i_len in zip(y_idx, y_len):
        row = []
        for j_pos, j_len in zip(x_idx, x_len):
            row.append(encode_moments(x[..., i_pos:i_pos + i_len, j_pos:j_pos + j_len]))
        rows.append(row)
    ly = [o // VAE_RATIO for o in y_overlap]
    lx = [o // VAE_RATIO for o in x_overlap]
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend(rows[i - 1][j], tile, ly[i - 1], dim=-2)
            if j > 0:
                tile = blend(row[j - 1], tile, lx[j - 1], dim=-1)
            if i < len(rows) - 1:
                tile = tile[..., :-ly[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, :-lx[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def synth_image(size, seed):
    """Deterministic structured test image in [-1,1] — smooth + edges so the
    convs see real gradients, reproducible without torch RNG portability."""
    y = torch.linspace(0, 1, size).view(1, 1, 1, size, 1)
    x = torch.linspace(0, 1, size).view(1, 1, 1, 1, size)
    c0 = torch.sin(2 * math.pi * (x * 3 + seed * 0.1)) * torch.cos(2 * math.pi * y * 2)
    c1 = (x > 0.5).float() * 2 - 1
    c2 = torch.sin(2 * math.pi * (x + y) * 5)
    img = torch.cat([c0, c1.expand_as(c0), c2], dim=1) * 0.8
    return img.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    enc = EncoderFCN3D(ch=128, ch_mult=[1, 2, 2, 4, 4, 8],
                       space_down=[2, 2, 2, 2, 1, 1], time_down=[1, 2, 2, 1, 1, 1],
                       num_res_blocks=2, in_channels=3, z_channels=24, double_z=True)
    quant = nn.Conv3d(48, 48, 1)

    tensors = {}
    with safe_open(args.vae, framework="pt") as f:
        for k in f.keys():
            if k.startswith("encoder.") or k.startswith("quant_conv.") or k in (LATENTS_MEAN_KEY, LATENTS_STD_KEY):
                tensors[k] = f.get_tensor(k).float()
    enc.load_state_dict({k[len("encoder."):]: v for k, v in tensors.items() if k.startswith("encoder.")})
    quant.load_state_dict({k[len("quant_conv."):]: v for k, v in tensors.items() if k.startswith("quant_conv.")})
    enc.eval()
    quant.eval()
    lm = tensors[LATENTS_MEAN_KEY].view(1, -1, 1, 1, 1)
    ls = tensors[LATENTS_STD_KEY].view(1, -1, 1, 1, 1)

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)

    def encode_moments(x):
        with torch.no_grad():
            return quant(enc(x))

    def encode(x_pm1):
        x = (x_pm1 + 1.0) * 0.5
        x = (x - mean) / std
        moments = tiled_encode(encode_moments, x)
        m = torch.chunk(moments, 2, dim=1)[0]
        return (m - lm) / ls

    out = {}
    for name, size in (("x", 128), ("x_tiled", 384)):
        img = synth_image(size, seed=3 if size == 128 else 7)
        lat = encode(img)
        out[name] = img.contiguous()
        out["latent" + ("_tiled" if size == 384 else "")] = lat.float().contiguous()
        print(f"{name}: {tuple(img.shape)} -> {tuple(lat.shape)} "
              f"mean {lat.mean().item():+.4f} std {lat.std().item():.4f}")

    save_file(out, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
