#!/usr/bin/env python3
"""Dump MiniMax H3 DiT block parity fixtures.

WHAT THIS PROVES, AND WHAT IT DOES NOT. The reference block calls into
comfy_kitchen CUDA kernels (fused RMSNorm+rope, fused SwiGLU, its attention
entry point) which cannot run on this machine, so the math below is a
TRANSCRIPTION of `comfy/ldm/minimax/model.py` rather than the reference
executing. A green test therefore proves the Zig port agrees with an
independently written implementation of the same spec -- it catches the
MLX-side slips this port is actually prone to (wrong axis, wrong split order,
rope applied to the whole head) but it cannot catch a misreading shared by both
implementations. The layout fixtures in `dump_minimax_h3_layout.py` DO execute
the reference; this one cannot, and says so.

Everything the fixture exercises is a place where a port can be silently wrong
and still produce a running model:
  * the AdaLN reshape/chunk order (modality stride, expand order)
  * qkv split, per-head RMSNorm BEFORE rope, partial split-half rope with the
    top 32 of 128 dims left unrotated
  * the SwiGLU gate/up half order in the fused fc1
  * cos-before-sin in the timestep embedding

Computed in FLOAT32 on CPU so the fixture isolates STRUCTURE from precision;
the Zig parity test runs its side in f32 too. bf16 end-to-end behaviour is a
separate question answered by the live run, not by this file.

Usage:
    uv run --with torch --with numpy --with safetensors \
        tests/dump_minimax_h3_fixtures.py \
        --dit ~/claude-tmp/h3-build/src/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
        --out src/fixtures/minimax_h3_dit.safetensors
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

# Geometry is parameterized so the committed fixture can use a SMALL config.
# The traps this pins (reshape/chunk order, rope pass-through, gate/up order)
# are structural and config-independent, so a toy config exercises them at a
# size that fits in the repo. `--real` runs the shipped 5376/56/128 geometry
# against the actual checkpoint instead.
class Geo:
    def __init__(self, hidden, heads, head_dim, ffn, time_embed_dim, inv_freq_len, timestep_dim):
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.ffn = ffn
        self.time_embed_dim = time_embed_dim
        self.inv_freq_len = inv_freq_len
        self.timestep_dim = timestep_dim
        # 3 axes x inv_freq_len, doubled by the split-half pairing.
        self.rot = inv_freq_len * 3 * 2
        assert self.rot < head_dim, "rope must be PARTIAL or the pass-through tail is untested"

SHIPPED = Geo(5376, 56, 128, 14336, 2688, 16, 256)
TOY = Geo(256, 4, 32, 128, 32, 4, 32)


def rms_norm(x, weight, eps):
    v = x.float()
    out = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out


def apply_rope_split_half(x, cos, sin, geo):
    """Partial split-half rope on [..., head_dim].

    Dims [0, rot/2) pair with [rot/2, rot); [rot, head_dim) pass through. The
    pass-through is the part a port silently gets wrong.
    """
    half = geo.rot // 2
    x1, x2, tail = x[..., :half], x[..., half:geo.rot], x[..., geo.rot:]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.cat([o1, o2, tail], dim=-1)


def build_rope(positions, inv_freq):
    """[S,3] positions + [16] inv_freq -> cos/sin [1, S, 1, 48]."""
    per_axis = positions.float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
    ang = per_axis.reshape(positions.shape[0], -1)  # concat(t, h, w) = 48
    return torch.cos(ang)[None, :, None, :], torch.sin(ang)[None, :, None, :]


def attention(x, w, cos, sin, geo):
    s = x.shape[0]
    qkv = F.linear(x, w["attn.qkv_proj.weight"])
    q, k, v = qkv.split(geo.heads * geo.head_dim, dim=-1)
    q = q.view(1, s, geo.heads, geo.head_dim)
    k = k.view(1, s, geo.heads, geo.head_dim)
    v = v.view(1, s, geo.heads, geo.head_dim)
    # Per-head RMSNorm THEN rope -- the reference's fused kernel order.
    q = rms_norm(q, w["attn.q_norm.weight"], 1e-5)
    k = rms_norm(k, w["attn.k_norm.weight"], 1e-5)
    q = apply_rope_split_half(q, cos, sin, geo)
    k = apply_rope_split_half(k, cos, sin, geo)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)  # full bidirectional, no mask
    out = out.transpose(1, 2).reshape(s, geo.heads * geo.head_dim)
    return F.linear(out, w["attn.out_proj.weight"])


def mlp(x, w, geo):
    # fc1 emits gate and up FUSED; gate is the FIRST half.
    y = F.linear(x, w["mlp.fc1.weight"])
    gate, up = y.chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, w["mlp.fc2.weight"])


def time_embed(t_vals, w, geo):
    half = geo.timestep_dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t_vals.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # COS FIRST
    h = F.linear(emb, w["time_embedder.proj_in.weight"], w["time_embedder.proj_in.bias"])
    return F.linear(F.silu(h), w["time_embedder.proj_out.weight"], w["time_embedder.proj_out.bias"])


def adaln(t_emb, w, expand, modalities, geo):
    y = F.linear(F.silu(t_emb), w["adaln_proj.linear.weight"], w["adaln_proj.linear.bias"])
    # [M, expand*hidden*modalities] -> [M*modalities, expand*hidden] -> chunk.
    # Row order is (timestep, modality); a transposed view here silently swaps
    # which stream each modulation lands on.
    y = y.view(y.shape[0] * modalities, expand * geo.hidden)
    return y.chunk(expand, dim=-1)


def block_forward(h, t_emb, w, runs, cos, sin, geo):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln(t_emb, w, 6, 3, geo)

    def mod_scale_shift(v, shift, scale):
        out = v.clone()
        for a, b, row in runs:
            out[a:b] = out[a:b] * (1.0 + scale[row]) + shift[row]
        return out

    def mod_gate(x, gate, other):
        out = x.clone()
        for a, b, row in runs:
            out[a:b] = out[a:b] + other[a:b] * gate[row]
        return out

    n1 = rms_norm(h, w["norm1.weight"], 1e-5)
    m1 = mod_scale_shift(n1, shift_msa, scale_msa)
    h = mod_gate(h, gate_msa, attention(m1, w, cos, sin, geo))
    n2 = rms_norm(h, w["norm2.weight"], 1e-5)
    m2 = mod_scale_shift(n2, shift_mlp, scale_mlp)
    return mod_gate(h, gate_mlp, mlp(m2, w, geo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", default=None,
                    help="checkpoint to read real block-0 weights from (--real only)")
    ap.add_argument("--real", action="store_true",
                    help="use the shipped geometry + real weights instead of the toy fixture")
    ap.add_argument("--out", default="src/fixtures/minimax_h3_dit.safetensors")
    ap.add_argument("--seq", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    geo = SHIPPED if args.real else TOY
    seq = args.seq

    block_keys = ["norm1.weight", "norm2.weight", "attn.qkv_proj.weight", "attn.q_norm.weight",
                  "attn.k_norm.weight", "attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight",
                  "adaln_proj.linear.weight", "adaln_proj.linear.bias"]
    top_keys = ["rope.inv_freq", "time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
                "time_embedder.proj_out.weight", "time_embedder.proj_out.bias"]

    w = {}
    if args.real:
        if not args.dit:
            raise SystemExit("--real needs --dit")
        # Lazy per-tensor reads: the 62 GB checkpoint never lands in RAM.
        with safe_open(args.dit, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for k in block_keys:
                full = f"blocks.0.{k}"
                if full not in keys:
                    raise SystemExit(f"missing {full}")
                w[k] = f.get_tensor(full).float()
            for k in top_keys:
                if k not in keys:
                    raise SystemExit(f"missing {k}")
                w[k] = f.get_tensor(k).float()
    else:
        inner = geo.heads * geo.head_dim
        # Small random weights: the traps being pinned are structural, so the
        # VALUES are irrelevant as long as they are asymmetric enough that a
        # transposed or mis-split read cannot coincidentally agree.
        def rnd(*shape):
            return torch.randn(*shape) * 0.05
        w["norm1.weight"] = rnd(geo.hidden) + 1.0
        w["norm2.weight"] = rnd(geo.hidden) + 1.0
        w["attn.qkv_proj.weight"] = rnd(inner * 3, geo.hidden)
        w["attn.q_norm.weight"] = rnd(geo.head_dim) + 1.0
        w["attn.k_norm.weight"] = rnd(geo.head_dim) + 1.0
        w["attn.out_proj.weight"] = rnd(geo.hidden, inner)
        w["mlp.fc1.weight"] = rnd(geo.ffn * 2, geo.hidden)
        w["mlp.fc2.weight"] = rnd(geo.hidden, geo.ffn)
        w["adaln_proj.linear.weight"] = rnd(6 * geo.hidden * 3, geo.time_embed_dim)
        w["adaln_proj.linear.bias"] = rnd(6 * geo.hidden * 3)
        w["rope.inv_freq"] = torch.tensor(
            [1.0 / (100.0 ** (i / geo.inv_freq_len)) for i in range(geo.inv_freq_len)])
        w["time_embedder.proj_in.weight"] = rnd(geo.hidden, geo.timestep_dim)
        w["time_embedder.proj_in.bias"] = rnd(geo.hidden)
        w["time_embedder.proj_out.weight"] = rnd(geo.time_embed_dim, geo.hidden)
        w["time_embedder.proj_out.bias"] = rnd(geo.time_embed_dim)

    # A packed sequence shaped like the real thing: text, then audio, then
    # video, each on its own modality tag, across two distinct timesteps. Three
    # runs with DIFFERENT mod rows is the minimum that can catch a modulation
    # landing on the wrong stream.
    text_len = seq // 4
    audio_len = seq // 4
    runs = [(0, text_len, 0 * 3 + 1),
            (text_len, text_len + audio_len, 1 * 3 + 2),
            (text_len + audio_len, seq, 0 * 3 + 0)]

    positions = torch.zeros(seq, 3)
    positions[:, 0] = torch.arange(seq, dtype=torch.float32) * 1.6666666666666667
    positions[:, 1] = torch.linspace(-4.0, 4.0, seq)
    positions[:, 2] = torch.linspace(-6.0, 6.0, seq)
    cos, sin = build_rope(positions, w["rope.inv_freq"])

    t_vals = torch.tensor([0.5, 0.8], dtype=torch.float32)
    t_emb = time_embed(t_vals, w, geo)

    h_in = torch.randn(seq, geo.hidden) * 0.5
    h_out = block_forward(h_in, t_emb, w, runs, cos, sin, geo)

    # Individual stages too, so a failure localizes instead of just saying
    # "the block is wrong".
    attn_in = rms_norm(h_in, w["norm1.weight"], 1e-5)
    attn_out = attention(attn_in, w, cos, sin, geo)
    mlp_out = mlp(attn_in, w, geo)

    out = dict(w)
    out.update({
        "x.positions": positions.contiguous(),
        "x.rope_cos": cos.squeeze(0).squeeze(1).contiguous(),
        "x.rope_sin": sin.squeeze(0).squeeze(1).contiguous(),
        "x.t_vals": t_vals.contiguous(),
        "x.t_emb": t_emb.contiguous(),
        "x.h_in": h_in.contiguous(),
        "x.h_out": h_out.contiguous(),
        "x.attn_in": attn_in.contiguous(),
        "x.attn_out": attn_out.contiguous(),
        "x.mlp_out": mlp_out.contiguous(),
        "x.runs": torch.tensor(runs, dtype=torch.int32).contiguous(),
    })
    out = {k: v.contiguous().float() if v.dtype.is_floating_point else v.contiguous()
           for k, v in out.items()}

    # A fixture holding NaN would "pass" any comparison that checks closeness
    # before it checks finiteness.
    for k, v in out.items():
        if v.dtype.is_floating_point and not torch.isfinite(v).all():
            raise SystemExit(f"non-finite values in {k} -- refusing to write a fixture that cannot fail")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file(out, args.out, metadata={
        "seq": str(seq), "hidden": str(geo.hidden), "heads": str(geo.heads),
        "head_dim": str(geo.head_dim), "ffn": str(geo.ffn),
        "time_embed_dim": str(geo.time_embed_dim), "inv_freq_len": str(geo.inv_freq_len),
        "timestep_dim": str(geo.timestep_dim), "rot": str(geo.rot),
        "real": "1" if args.real else "0",
    })
    total = sum(v.numel() for v in out.values())
    print(f"wrote {args.out}  ({total * 4 / 1e6:.2f} MB f32, {len(out)} tensors)")
    print(f"  geometry hidden={geo.hidden} heads={geo.heads} head_dim={geo.head_dim} "
          f"rot={geo.rot} (tail {geo.head_dim - geo.rot} unrotated) seq={seq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
