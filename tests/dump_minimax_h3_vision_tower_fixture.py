#!/usr/bin/env python3
"""Dump Qwen3-VL VISION TOWER parity fixtures for MiniMax H3 — reference EXECUTES.

H3's conditioning tower is 529 tensors inside `text_encoder.safetensors`, and
its output is spliced straight into the LM sequence. A wrong patch order, pos-
embed order or rotary layout there produces conditioning that is silently WRONG
rather than absent — the model runs and follows something that is not the image.

The reference classes (`Qwen35VisionModel` + Qwen3-VL's DeepStack merger) are
reproduced VERBATIM from ComfyUI's `comfy/text_encoders/qwen35.py` and
`qwen3vl.py`, with `comfy.ops.disable_weight_init` swapped for plain `nn.*`
(which is what it is) and `optimized_attention` for `F.scaled_dot_product_attention`.

Our pack stores the tower QUANTIZED (affine 8-bit, group 64), so the weights are
dequantized through mlx first: the oracle then proves our tower MATH agrees with
the reference's on the SAME weights, which is the question. Comparing against
the upstream bf16 file would fold quantization error into every diff and could
not tell a layout bug from a rounding one.

Usage:
    uv run --with torch --with numpy --with mlx --with safetensors \
        tests/dump_minimax_h3_vision_tower_fixture.py \
        --model ~/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit \
        --out ~/claude-tmp/h3-build/minimax_h3_vit_fixture.safetensors
"""

import argparse
import math

import mlx.core as mx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

HIDDEN = 1152
HEADS = 16
HEAD_DIM = HIDDEN // HEADS  # 72
INTER = 4304
DEPTH = 27
OUT_HIDDEN = 5120
MERGE = 2
GRID_SIDE = 48
PATCH = 16
TEMPORAL = 2
DEEPSTACK_IDX = [8, 16, 24]


# ── reference (verbatim) ────────────────────────────────────────────────────

def apply_rope(xq, xk, freqs_cis):
    cos, sin, nsin = freqs_cis
    q = xq * cos
    h = q.shape[-1] // 2
    q = torch.cat([q[..., :h] + xq[..., h:] * nsin, q[..., h:] + xq[..., :h] * sin], dim=-1)
    k = xk * cos
    k = torch.cat([k[..., :h] + xk[..., h:] * nsin, k[..., h:] + xk[..., :h] * sin], dim=-1)
    return q, k


class VisionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = nn.Linear(HIDDEN, INTER)
        self.linear_fc2 = nn.Linear(INTER, HIDDEN)

    def forward(self, x):
        # NOTE approximate="tanh" here, and NOT in the mergers below.
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class VisionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(HIDDEN, HIDDEN * 3)
        self.proj = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x, cu_seqlens, pos_emb):
        n = x.shape[0]
        q, k, v = self.qkv(x).reshape(n, 3, HEADS, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rope(q, k, pos_emb)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outs = []
        for qs, ks, vs in zip(torch.split(q, lengths, 0), torch.split(k, lengths, 0),
                              torch.split(v, lengths, 0)):
            o = F.scaled_dot_product_attention(qs.transpose(0, 1).unsqueeze(0),
                                               ks.transpose(0, 1).unsqueeze(0),
                                               vs.transpose(0, 1).unsqueeze(0))
            outs.append(o.squeeze(0).transpose(0, 1))
        return self.proj(torch.cat(outs, 0).reshape(n, -1))


class VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(HIDDEN, eps=1e-6)
        self.norm2 = nn.LayerNorm(HIDDEN, eps=1e-6)
        self.attn = VisionAttention()
        self.mlp = VisionMLP()

    def forward(self, x, cu_seqlens, pos_emb):
        x = x + self.attn(self.norm1(x), cu_seqlens, pos_emb)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    """Main merger: LayerNorm BEFORE the 2x2 shuffle, DEFAULT (erf) gelu."""

    def __init__(self):
        super().__init__()
        self.merge_dim = HIDDEN * MERGE * MERGE
        self.norm = nn.LayerNorm(HIDDEN, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merge_dim, self.merge_dim)
        self.linear_fc2 = nn.Linear(self.merge_dim, OUT_HIDDEN)

    def forward(self, x):
        x = self.norm(x).view(-1, self.merge_dim)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class DeepstackMerger(nn.Module):
    """DeepStack merger: POSTshuffle LayerNorm, DEFAULT (erf) gelu."""

    def __init__(self):
        super().__init__()
        self.merge_dim = HIDDEN * MERGE * MERGE
        self.norm = nn.LayerNorm(self.merge_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merge_dim, self.merge_dim)
        self.linear_fc2 = nn.Linear(self.merge_dim, OUT_HIDDEN)

    def forward(self, x):
        x = self.norm(x.view(-1, self.merge_dim))
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class VisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed_proj = nn.Conv3d(3, HIDDEN, kernel_size=[TEMPORAL, PATCH, PATCH],
                                          stride=[TEMPORAL, PATCH, PATCH], bias=True)
        self.pos_embed = nn.Embedding(GRID_SIDE * GRID_SIDE, HIDDEN)
        self.blocks = nn.ModuleList([VisionBlock() for _ in range(DEPTH)])
        self.merger = PatchMerger()
        self.deepstack_merger_list = nn.ModuleList([DeepstackMerger() for _ in DEEPSTACK_IDX])

    def rot_pos_emb(self, grid_thw):
        dim = HEAD_DIM // 2
        inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        max_hw = max(max(h, w) for _, h, w in grid_thw)
        table = torch.outer(torch.arange(max_hw, dtype=torch.float), inv)
        ids = []
        for t, h, w in grid_thw:
            mh, mw = h // MERGE, w // MERGE
            rows = torch.arange(mh)[:, None, None, None] * MERGE + torch.arange(MERGE)[None, None, :, None]
            cols = torch.arange(mw)[None, :, None, None] * MERGE + torch.arange(MERGE)[None, None, None, :]
            rows = rows.expand(mh, mw, MERGE, MERGE).reshape(-1)
            cols = cols.expand(mh, mw, MERGE, MERGE).reshape(-1)
            c = torch.stack((rows, cols), -1)
            if t > 1:
                c = c.repeat(t, 1)
            ids.append(c)
        return table[torch.cat(ids)].flatten(1)

    def pos_embed_interpolate(self, grid_thw):
        idx_list = [[] for _ in range(4)]
        wt_list = [[] for _ in range(4)]
        for t, h, w in grid_thw:
            hi = torch.linspace(0, GRID_SIDE - 1, h)
            wi = torch.linspace(0, GRID_SIDE - 1, w)
            hf, wf = hi.int(), wi.int()
            hc = (hi.int() + 1).clip(max=GRID_SIDE - 1)
            wc = (wi.int() + 1).clip(max=GRID_SIDE - 1)
            dh, dw = hi - hf, wi - wf
            bh, bhc = hf * GRID_SIDE, hc * GRID_SIDE
            inds = [(bh[None].T + wf[None]).flatten(), (bh[None].T + wc[None]).flatten(),
                    (bhc[None].T + wf[None]).flatten(), (bhc[None].T + wc[None]).flatten()]
            wts = [((1 - dh)[None].T * (1 - dw)[None]).flatten(), ((1 - dh)[None].T * dw[None]).flatten(),
                   (dh[None].T * (1 - dw)[None]).flatten(), (dh[None].T * dw[None]).flatten()]
            for j in range(4):
                idx_list[j].extend(inds[j].tolist())
                wt_list[j].extend(wts[j].tolist())
        it = torch.tensor(idx_list, dtype=torch.long)
        wt = torch.tensor(wt_list, dtype=torch.float)
        pe = self.pos_embed(it) * wt[:, :, None]
        pe = pe[0] + pe[1] + pe[2] + pe[3]
        pe = pe.split([h * w for _, h, w in grid_thw])
        out = []
        for p, (t, h, w) in zip(pe, grid_thw):
            p = p.repeat(t, 1)
            out.append(p.view(t, h // MERGE, MERGE, w // MERGE, MERGE, -1)
                        .permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
        return torch.cat(out)

    def forward(self, pixel_values, grid_thw):
        x = pixel_values.view(-1, 3, TEMPORAL, PATCH, PATCH)
        x = self.patch_embed_proj(x).view(-1, HIDDEN)
        x = x + self.pos_embed_interpolate(grid_thw)
        rpe = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rpe, rpe), dim=-1)
        cos, sin = emb.cos().unsqueeze(-2), emb.sin().unsqueeze(-2)
        half = sin.shape[-1] // 2
        pos_emb = (cos, sin[..., :half], -sin[..., half:])
        lens = []
        for t, h, w in grid_thw:
            lens += [h * w] * t
        cu = F.pad(torch.tensor(lens, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
        ds = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, cu, pos_emb)
            if i in DEEPSTACK_IDX:
                ds.append(self.deepstack_merger_list[DEEPSTACK_IDX.index(i)](x))
        return self.merger(x), ds


# ── weight loading (dequantize our affine pack) ─────────────────────────────

def load_visual(model_dir):
    path = f"{model_dir}/text_encoder.safetensors"
    raw = mx.load(path)
    out = {}
    keys = [k for k in raw if k.startswith("visual.")]
    bases = set()
    for k in keys:
        if k.endswith(".scales") or k.endswith(".biases"):
            bases.add(k.rsplit(".", 1)[0])
    for k in keys:
        if k.endswith(".scales") or k.endswith(".biases"):
            continue
        base = k.rsplit(".", 1)[0]
        leaf = k.rsplit(".", 1)[1]
        if leaf == "weight" and base in bases:
            w = mx.dequantize(raw[k], raw[base + ".scales"], raw[base + ".biases"],
                              group_size=64, bits=8)
        else:
            w = raw[k]
        out[k] = torch.from_numpy(np.array(w.astype(mx.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    tw = load_visual(args.model)
    tower = VisionTower()
    sd = {}
    sd["patch_embed_proj.weight"] = tw["visual.patch_embed.proj.weight"]
    sd["patch_embed_proj.bias"] = tw["visual.patch_embed.proj.bias"]
    sd["pos_embed.weight"] = tw["visual.pos_embed.weight"]
    for i in range(DEPTH):
        for a, b in (("norm1", "norm1"), ("norm2", "norm2")):
            sd[f"blocks.{i}.{a}.weight"] = tw[f"visual.blocks.{i}.{b}.weight"]
            sd[f"blocks.{i}.{a}.bias"] = tw[f"visual.blocks.{i}.{b}.bias"]
        for a, b in (("attn.qkv", "attn.qkv"), ("attn.proj", "attn.proj"),
                     ("mlp.linear_fc1", "mlp.linear_fc1"), ("mlp.linear_fc2", "mlp.linear_fc2")):
            sd[f"blocks.{i}.{a}.weight"] = tw[f"visual.blocks.{i}.{b}.weight"]
            sd[f"blocks.{i}.{a}.bias"] = tw[f"visual.blocks.{i}.{b}.bias"]
    for pre, src in [("merger", "visual.merger")] + \
                    [(f"deepstack_merger_list.{j}", f"visual.deepstack_merger_list.{j}") for j in range(3)]:
        for leaf in ("norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias",
                     "linear_fc2.weight", "linear_fc2.bias"):
            sd[f"{pre}.{leaf}"] = tw[f"{src}.{leaf}"]
    tower.load_state_dict(sd)
    tower.eval()

    # A high-contrast LEFT/RIGHT split, the same structure the live fl2va test
    # uses — an asymmetric image is what makes a mirrored patch order visible.
    S = args.size
    img = torch.zeros(1, S, S, 3)
    img[:, :, : S // 2, :] = 20 / 255.0
    img[:, :, S // 2:, :] = 235 / 255.0
    # normalize to [-1,1] like process_qwen2vl_images with mean/std 0.5
    chw = img.permute(0, 3, 1, 2)[0]
    chw = (chw - 0.5) / 0.5
    gh, gw = S // PATCH, S // PATCH
    pv = chw.unsqueeze(0).repeat(TEMPORAL, 1, 1, 1)
    patches = pv.reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH).contiguous()

    with torch.no_grad():
        merged, ds = tower(flat, [(1, gh, gw)])

    out = {
        "pixel_values": flat,
        "frames": pv.contiguous(),          # [2,3,H,W], what our engine patchifies
        "merged": merged.contiguous(),
        "deepstack_0": ds[0].contiguous(),
        "deepstack_1": ds[1].contiguous(),
        "deepstack_2": ds[2].contiguous(),
        "grid": torch.tensor([1, gh, gw], dtype=torch.int32),
    }
    save_file(out, args.out)
    print(f"grid 1x{gh}x{gw}  patches {flat.shape}  merged {tuple(merged.shape)} "
          f"mean {merged.mean().item():+.4f} std {merged.std().item():.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
