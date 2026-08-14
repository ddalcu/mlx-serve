#!/usr/bin/env python3
"""Dump LTX-2.5 DiffVAE-decoder parity fixtures (.raw) for the Zig oracle in src/ltx_diffvae_forward.zig.

The reference decoder is `Lightricks/LTX-2`
`ltx_core/model/video_vae/diffusion_video_decoder.py` + its `transformer/`
package. Its production neighborhood-attention backend is NATTEN, which is
CUDA-only — but the same package vendors a pure-torch NA
(`transformer/fallback_na/eager.py`, Apache-2.0, from comfy-kitchen) with
identical `na3d` semantics, so the oracle runs on CPU with nothing but torch.
The modules below are a faithful TRANSCRIPTION of those files, not an import:
`ltx_core` pulls in the whole pipeline (devices, tiling, Disposable, …) for four
small nn.Modules.

Dumps fp32 on CPU. MPS fp16 quietly decorrelates a stack this deep, and every
comparison here concatenates context with x, so a cosine alone cannot see a
scale error — the Zig side asserts `rms_ratio` beside every cosine.

USER-RUN, not CI: it needs torch and the 0.83 GB decoder checkpoint.

Usage:
    python3 tests/dump_ltx_diffvae_fixtures.py <pack_dir> [OUT_DIR]

`pack_dir` holds `vae_diffusion_decoder.safetensors` (the 8-bit LTX-2.5 pack
ships it; the 4-bit pack does not). Writes flat little-endian f32:

    latent.raw    [1,128,T,H,W]        the input latent (BCFHW)
    x_t.raw       [1,F,H4,W4,48]       the noise the sampler starts from
    stage0..3.raw [1,T,H,W,C]          each det stage's OUTPUT (post-upsample)
    context.raw   [1,F5,H5,W5,256]     the context volume the diff blocks read
    vpred.raw     [1,F,H4,W4,48]       one forward_diff_step at the shipped timestep
    pixels.raw    [1,3,F,Hpx,Wpx]      the finished decode (BCFHW)

then prints the `export LTX_DIFFVAE_*` block to paste before:
    zig build test -Doptimize=ReleaseFast -Dtest-filter="diffvae parity"
"""

import math
import os
import sys

import torch
from torch import nn
from torch.nn import functional as F

PREFIX = "vae_diffusion_decoder."

# The checkpoint ships no VAE config, so these are the reference's production
# `L` layout — the same numbers `ltx_diffvae.production` pins, and every one of
# them that a weight SHAPE can confirm does confirm.
STAGE_CHANNELS = (2048, 1024, 512, 512, 256)
STAGE_DEPTHS = (4, 6, 4, 2, 8)
STAGE_KERNELS = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5), (3, 3, 3))
UPSAMPLES = (((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2))
STAGE5_KERNEL = (3, 7, 7)
HEAD_DIM = 64
PATCH_SIZE = 4
T_EMB_DIM = 384
# What the diffusion stage predicts, in how many steps, and the timestep scale.
# The reference reads all three from a `vae` config that no pack ships, so its
# class defaults ("v", 2, x1) are a guess — and running the weights that way
# decodes to static. Measured: x0, one step, timesteps x1000.
NUM_STEPS = 1
TIMESTEP_SCALE = 1000.0

LATENT_SHAPE = (1, 128, 3, 8, 8)
SEED = 7


# ── vendored NA (fallback_na/eager.py), trimmed to the CPU path ───────────


def _window_bounds(length: int, kernel: int) -> tuple[list[int], list[int]]:
    """NATTEN SHIFTS the window inward at a boundary; it does not clamp-and-mask."""
    kernel = min(kernel, length)
    lo = length - kernel
    half = kernel // 2
    starts = [min(max(i - half, 0), lo) for i in range(length)]
    return starts, [s + kernel for s in starts]


def na3d(q, k, v, kernel_size):
    """`(B,T,H,W,NH,HD)` neighborhood attention; Q is already scaled (scale=1).

    Same semantics as the vendored eager backend, gathered rather than masked:
    one python step per (frame, H-slab), everything else vectorized. The mask
    formulation in `eager.py` materializes an `[Nq, Nk]` block per tile group,
    which is fine on a GPU and not on this.
    """
    batch, t, h, w, nh, hd = q.shape
    dims = (t, h, w)
    kernels = [min(kk, d) for kk, d in zip(kernel_size, dims)]
    kt, kh, kw = kernels
    starts = [torch.tensor(_window_bounds(d, kk)[0]) for d, kk in zip(dims, kernels)]
    hidx = starts[1][:, None] + torch.arange(kh)   # H, kh
    widx = starts[2][:, None] + torch.arange(kw)   # W, kw
    # Bound one gather to ~64 MB of f32.
    per_row = kt * kh * w * kw * nh * hd
    h_step = max(1, min(h, (1 << 24) // max(1, per_row)))
    out = torch.empty_like(v)
    for ti in range(t):
        t0 = int(starts[0][ti])
        kt_slice_k = k[:, t0 : t0 + kt]
        kt_slice_v = v[:, t0 : t0 + kt]
        for h0 in range(0, h, h_step):
            h1 = min(h0 + h_step, h)
            rows = hidx[h0:h1]                                   # hs, kh
            ks = kt_slice_k[:, :, rows][:, :, :, :, widx]        # B,kt,hs,kh,W,kw,NH,HD
            vs = kt_slice_v[:, :, rows][:, :, :, :, widx]
            ks = ks.permute(0, 2, 4, 1, 3, 5, 6, 7).reshape(batch, h1 - h0, w, kt * kh * kw, nh, hd)
            vs = vs.permute(0, 2, 4, 1, 3, 5, 6, 7).reshape(batch, h1 - h0, w, kt * kh * kw, nh, hd)
            qs = q[:, ti, h0:h1]                                 # B,hs,W,NH,HD
            scores = torch.einsum("bhwnd,bhwknd->bhwkn", qs, ks)
            probs = scores.softmax(dim=3)
            out[:, ti, h0:h1] = torch.einsum("bhwkn,bhwknd->bhwnd", probs, vs)
    return out


# ── transcribed modules ──────────────────────────────────────────────────


def rope_inv_freqs(dim: int, base: float = 10000.0) -> torch.Tensor:
    import numpy as np

    exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
    return torch.from_numpy(1.0 / np.power(float(base), exponents)).to(torch.float32)


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return (d_t, d_hw, d_hw)


def rot_abs_axis(xc: torch.Tensor, pos: torch.Tensor, inv: torch.Tensor, axis: int) -> torch.Tensor:
    pairs = xc.reshape(*xc.shape[:-1], xc.shape[-1] // 2, 2)
    xe = pairs[..., 0].to(torch.float32)
    xo = pairs[..., 1].to(torch.float32)
    shape = [1, 1, 1, 1, 1, inv.shape[0]]
    shape[axis] = pos.shape[0]
    ang = (pos[:, None] * inv[None, :]).reshape(shape)
    c, s = ang.cos(), ang.sin()
    return torch.stack([xe * c - xo * s, xe * s + xo * c], dim=-1).reshape(xc.shape).to(xc.dtype)


class NeighborhoodAttention3D(nn.Module):
    def __init__(self, dim: int, kernel_size, head_dim: int = HEAD_DIM):
        super().__init__()
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5
        self.rope_dim_split = default_rope_dim_split(head_dim)
        for name, d in zip(("t", "h", "w"), self.rope_dim_split):
            self.register_buffer(f"rope_inv_{name}", rope_inv_freqs(d), persistent=False)
        # The checkpoint ships to_q/to_k/to_v split already (the reference nests
        # them under `attn.qkv` and splits a fused `qkv.weight` at load).
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        d_t, d_h, _ = self.rope_dim_split
        t, h, w = x.shape[1], x.shape[2], x.shape[3]
        ar = lambda n: torch.arange(n, dtype=torch.float32, device=x.device)  # noqa: E731
        xt = rot_abs_axis(x[..., :d_t], ar(t), self.rope_inv_t, axis=1)
        xh = rot_abs_axis(x[..., d_t : d_t + d_h], ar(h), self.rope_inv_h, axis=2)
        xw = rot_abs_axis(x[..., d_t + d_h :], ar(w), self.rope_inv_w, axis=3)
        return torch.cat([xt, xh, xw], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, t, h, w, _ = x.shape
        shape = (batch, t, h, w, self.num_heads, self.head_dim)
        q = self.to_q(x).view(shape)
        k = self.to_k(x).view(shape)
        v = self.to_v(x).view(shape)
        q = self.q_norm(q) * self.scale
        k = self.k_norm(k)
        out = na3d(self._rope(q), self._rope(k), v, self.kernel_size)
        return self.proj(out.reshape(batch, t, h, w, self.dim))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def _mlp_hidden(dim: int, ratio: float = 4.0) -> int:
    return (int(dim * ratio) + 15) // 16 * 16


class NABlock(nn.Module):
    def __init__(self, dim: int, kernel_size, head_dim: int = HEAD_DIM):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, _mlp_hidden(dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class LinearPixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels: int, stride, reduction: int = 1):
        super().__init__()
        self.stride = tuple(stride)
        out = math.prod(self.stride) * in_channels // reduction
        self.proj = nn.Linear(in_channels, out, bias=True)

    def forward(self, x, drop_leading_frame: bool = True):
        b, t, h, w, _ = x.shape
        p1, p2, p3 = self.stride
        x = self.proj(x)
        c = x.shape[-1] // (p1 * p2 * p3)
        # "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c" — channel MINOR.
        x = x.reshape(b, t, h, w, c, p1, p2, p3)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, t * p1, h * p2, w * p3, c)
        if p1 == 2 and drop_leading_frame:
            x = x[:, 1:]
        return x.contiguous()


class AdaLNZero(nn.Module):
    NUM_CHUNKS = 7

    def __init__(self, dim: int, t_emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)

    def forward(self, t_emb):
        chunks = self.proj(F.silu(t_emb)).chunk(self.NUM_CHUNKS, dim=-1)
        return tuple(c[:, None, None, None, :] for c in chunks)


class TimestepEmbedder(nn.Module):
    """PixArtAlphaCombinedTimestepSizeEmbeddings with size_emb_dim=0."""

    def __init__(self, dim: int, sin_dim: int = 256):
        super().__init__()
        self.sin_dim = sin_dim
        self.linear1 = nn.Linear(sin_dim, dim, bias=True)
        self.linear2 = nn.Linear(dim, dim, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.sin_dim // 2
        exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        emb = t[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([emb.cos(), emb.sin()], dim=-1)  # flip_sin_to_cos=True
        return self.linear2(F.silu(self.linear1(emb)))


class CombinedDiffusionNABlock(nn.Module):
    def __init__(self, dim: int, kernel_size, context_channels: int, head_dim: int = HEAD_DIM):
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(AdaLNZero.NUM_CHUNKS, dim))
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, _mlp_hidden(dim))

    def forward(self, x, context, modulation):
        mods = [modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)]
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = mods
        x = x + self.context_proj(context)
        x = x + self.attn(self.norm1(x) * (1.0 + scale_msa) + shift_msa)
        return x + self.mlp(self.norm2(x) * (1.0 + scale_mlp) + shift_mlp)


class DiffusionVideoDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("std_of_means", torch.ones(128))
        self.register_buffer("mean_of_means", torch.zeros(128))
        self.conv_in = nn.Linear(128, STAGE_CHANNELS[0], bias=True)
        self.det_stages = nn.ModuleList(
            nn.ModuleList(NABlock(STAGE_CHANNELS[i], STAGE_KERNELS[i]) for _ in range(STAGE_DEPTHS[i]))
            for i in range(4)
        )
        self.upsamples = nn.ModuleList(
            LinearPixelShuffleUpsample(STAGE_CHANNELS[i], UPSAMPLES[i][0], UPSAMPLES[i][1]) for i in range(4)
        )
        c5 = STAGE_CHANNELS[4]
        self.t_embedder = TimestepEmbedder(T_EMB_DIM)
        self.conv_in_x_t = nn.Linear(3 * PATCH_SIZE**2, c5, bias=True)
        self.shared_adaln = AdaLNZero(c5, T_EMB_DIM)
        self.diff_blocks = nn.ModuleList(
            CombinedDiffusionNABlock(c5, STAGE5_KERNEL, c5) for _ in range(STAGE_DEPTHS[4])
        )
        self.norm_out = nn.RMSNorm(c5, eps=1e-6)
        self.conv_out = nn.Linear(c5, 3 * PATCH_SIZE**2, bias=True)

    def stages(self, latent_bcfhw, capture: list):
        x = latent_bcfhw * self.std_of_means.view(1, -1, 1, 1, 1) + self.mean_of_means.view(1, -1, 1, 1, 1)
        x = self.conv_in(x.permute(0, 2, 3, 4, 1))
        for i in range(4):
            for blk in self.det_stages[i]:
                x = blk(x)
            x = self.upsamples[i](x, drop_leading_frame=True)
            capture.append(x)
        return x

    def diff_step(self, context, x_t, t):
        modulation = self.shared_adaln(self.t_embedder(t))
        x = self.conv_in_x_t(x_t)
        for blk in self.diff_blocks:
            x = blk(x, context, modulation)
        return self.conv_out(self.norm_out(x))


# ── weight remap ─────────────────────────────────────────────────────────


def load_weights(model: nn.Module, path: str) -> None:
    from safetensors.torch import load_file

    raw = load_file(path)
    remapped = {}
    for key, value in raw.items():
        if not key.startswith(PREFIX):
            continue
        k = key[len(PREFIX) :]
        if k == "type_emb":
            # Shipped but never read by the reference decoder's forward.
            continue
        k = k.replace("per_channel_statistics.mean", "mean_of_means")
        k = k.replace("per_channel_statistics.std", "std_of_means")
        k = k.replace("t_embedder.timestep_embedder.", "t_embedder.")
        remapped[k] = value.to(torch.float32)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    real_missing = [m for m in missing if not m.endswith(("rope_inv_t", "rope_inv_h", "rope_inv_w"))]
    if real_missing or unexpected:
        raise SystemExit(f"weight remap drifted:\n  missing={real_missing}\n  unexpected={unexpected}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pack = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./ltx_diffvae_fixtures"
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(pack, "vae_diffusion_decoder.safetensors")
    if not os.path.exists(ckpt):
        raise SystemExit(f"{ckpt} not found — the 8-bit LTX-2.5 pack ships it, the 4-bit pack does not")

    torch.manual_seed(SEED)
    model = DiffusionVideoDecoder().eval()
    load_weights(model, ckpt)

    latent = torch.randn(LATENT_SHAPE)
    with torch.no_grad():
        capture: list[torch.Tensor] = []
        context = model.stages(latent, capture)
        f5, h5, w5 = context.shape[1:4]
        x_t = torch.randn(1, f5, h5, w5, 3 * PATCH_SIZE**2)
        t = torch.tensor([TIMESTEP_SCALE])
        vpred = model.diff_step(context, x_t, t)

        # The sampler: linspace(1, 1/n, n) scaled by TIMESTEP_SCALE; every step
        # but the last is a reverse Euler update, and the last prediction IS x0.
        ts = torch.linspace(1.0, 1.0 / NUM_STEPS, NUM_STEPS)
        x = x_t
        for i, t_now in enumerate(ts):
            pred = model.diff_step(context, x, (t_now * TIMESTEP_SCALE).reshape(1))
            if i + 1 == len(ts):
                x = pred
            else:
                x = x - (t_now - ts[i + 1]) * pred
        # unpatchify "b (c p r q) f h w -> b c (f p) (h q) (w r)" with p=1,
        # applied channels-last then permuted to BCFHW.
        b, f, h, w, _ = x.shape
        px = x.reshape(b, f, h, w, 3, PATCH_SIZE, PATCH_SIZE)
        px = px.permute(0, 1, 2, 6, 3, 5, 4).reshape(b, f, h * PATCH_SIZE, w * PATCH_SIZE, 3)
        px = px.permute(0, 4, 1, 2, 3).contiguous()

    def dump(name: str, arr: torch.Tensor) -> str:
        path = os.path.join(out_dir, name)
        with open(path, "wb") as fh:
            fh.write(arr.detach().to(torch.float32).contiguous().numpy().tobytes())
        print(f"  {name:14s} {tuple(arr.shape)}")
        return os.path.abspath(path)

    print("fixtures:")
    paths = {
        "LATENT": dump("latent.raw", latent),
        "STAGE0": dump("stage0.raw", capture[0]),
        "STAGE1": dump("stage1.raw", capture[1]),
        "STAGE2": dump("stage2.raw", capture[2]),
        "CONTEXT": dump("context.raw", capture[3]),
        "XT": dump("x_t.raw", x_t),
        "VPRED": dump("vpred.raw", vpred),
        "PIXELS": dump("pixels.raw", px),
    }
    print()
    for k, v in paths.items():
        print(f"export LTX_DIFFVAE_{k}={v}")


if __name__ == "__main__":
    main()
