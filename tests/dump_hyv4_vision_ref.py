#!/usr/bin/env python3
"""Reference dump for Hy4-flash's vision tower (user-run; needs torch + torchvision).

Transcribes `HYV4VLVisionModel` from transformers main (`hy_v4_vl/modeling_hy_v4_vl.py`) over the
TensorFold pack's `vit.*` tensors and the pack's own processor (`HYV4VLImageProcessor`: smart
resize factor patch*merge, bicubic, rescale 1/255, mean/std 0.5, merge-major patchify). Writes
`vision_fixture.json` (input pixels, grid, expected image features) for the Zig tower's env-gated
parity test. CPU, f32, ~1 GB of weights.

Usage: ~/.venv-vllm-metal/bin/python tests/dump_hyv4_vision_ref.py \
         --pack ~/.mlx-serve/models/TensorFold/Hy4-flash-MLX-4bit-MTP --out ~/claude-tmp/hy4-vision
"""

import argparse
import json
import math
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def smart_resize(height, width, factor, min_pixels, max_pixels):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio too large")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def load_vision_weights(pack):
    idx = json.load(open(os.path.join(pack, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for name, shard in idx.items():
        if not name.startswith("vit."):
            continue
        # Load shard once per file.
        key = name[len("vit."):]
        f = load_file(os.path.join(pack, shard), device="cpu")
        out[key] = f[name].float()
    return out


class PatchEmbed(torch.nn.Module):
    def __init__(self, cfg, w):
        super().__init__()
        self.patch_size = cfg["patch_size"]
        self.position_edge = cfg["max_image_size"] // self.patch_size
        self.patch_embedding = torch.nn.Conv2d(3, cfg["hidden_size"], self.patch_size, self.patch_size)
        self.patch_embedding.weight.data.copy_(w["embeddings.patch_embedding.weight"])
        self.patch_embedding.bias.data.copy_(w["embeddings.patch_embedding.bias"])
        self.position_embedding = torch.nn.Embedding(self.position_edge**2 + 1, cfg["hidden_size"])
        self.position_embedding.weight.data.copy_(w["embeddings.position_embedding.weight"])

    def interpolate(self, height, width):
        table = self.position_embedding.weight[1:, :].reshape(1, self.position_edge, self.position_edge, -1)
        table = table.permute(0, 3, 1, 2).float()
        scale_h = (int(height) + 0.1) / self.position_edge
        scale_w = (int(width) + 0.1) / self.position_edge
        out = F.interpolate(table, scale_factor=(scale_h, scale_w), mode="bilinear", align_corners=False)
        return out.permute(0, 2, 3, 1).reshape(1, -1, table.shape[1])

    def forward(self, pixel_values, grid_thw):
        n = pixel_values.shape[0]
        x = pixel_values.reshape(n, 3, self.patch_size, self.patch_size)
        emb = self.patch_embedding(x).flatten(-2).squeeze(-1).reshape(n, -1)
        pos = []
        for t, h, w in grid_thw:
            pos.append(self.interpolate(h, w).squeeze(0).repeat(t, 1))
        return (emb + torch.concat(pos, 0)).unsqueeze(0)


class Attention(torch.nn.Module):
    def __init__(self, cfg, w, i):
        super().__init__()
        self.heads = cfg["num_attention_heads"]
        self.head_dim = cfg["hidden_size"] // self.heads
        self.scale = self.head_dim**-0.5
        p = f"layers.{i}.self_attn."
        self.q = torch.nn.Linear(cfg["hidden_size"], self.heads * self.head_dim)
        self.k = torch.nn.Linear(cfg["hidden_size"], self.heads * self.head_dim)
        self.v = torch.nn.Linear(cfg["hidden_size"], self.heads * self.head_dim)
        self.o = torch.nn.Linear(self.heads * self.head_dim, cfg["hidden_size"])
        for lin, leaf in ((self.q, "q_proj"), (self.k, "k_proj"), (self.v, "v_proj"), (self.o, "o_proj")):
            lin.weight.data.copy_(w[p + leaf + ".weight"])
            lin.bias.data.copy_(w[p + leaf + ".bias"])

    def forward(self, x):
        q = self.q(x).reshape(1, x.shape[1], self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(1, x.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(1, x.shape[1], self.heads, self.head_dim).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=False)
        att = att.transpose(1, 2).reshape(1, x.shape[1], -1)
        return self.o(att)


class Block(torch.nn.Module):
    def __init__(self, cfg, w, i):
        super().__init__()
        eps = cfg["rms_norm_eps"]
        p = f"layers.{i}."
        self.n1 = torch.nn.LayerNorm(cfg["hidden_size"], eps=eps)
        self.n2 = torch.nn.LayerNorm(cfg["hidden_size"], eps=eps)
        self.n1.weight.data.copy_(w[p + "input_layernorm.weight"])
        self.n1.bias.data.copy_(w[p + "input_layernorm.bias"])
        self.n2.weight.data.copy_(w[p + "post_attention_layernorm.weight"])
        self.n2.bias.data.copy_(w[p + "post_attention_layernorm.bias"])
        self.attn = Attention(cfg, w, i)
        self.fc1 = torch.nn.Linear(cfg["hidden_size"], cfg["intermediate_size"])
        self.fc2 = torch.nn.Linear(cfg["intermediate_size"], cfg["hidden_size"])
        self.fc1.weight.data.copy_(w[p + "mlp.dense_h_to_4h.weight"])
        self.fc1.bias.data.copy_(w[p + "mlp.dense_h_to_4h.bias"])
        self.fc2.weight.data.copy_(w[p + "mlp.dense_4h_to_h.weight"])
        self.fc2.bias.data.copy_(w[p + "mlp.dense_4h_to_h.bias"])

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.n2(x))))
        return x


class Merger(torch.nn.Module):
    def __init__(self, cfg, w, merge=2):
        super().__init__()
        self.merge = merge
        eps = cfg["rms_norm_eps"]
        self.norm = torch.nn.RMSNorm(cfg["hidden_size"], eps=eps)
        self.norm.weight.data.copy_(w["perceive.norm.weight"])
        self.conv0 = torch.nn.Conv2d(cfg["hidden_size"], cfg["hidden_size"] * merge * merge, merge, merge)
        self.conv1 = torch.nn.Conv2d(cfg["hidden_size"] * merge * merge, cfg["out_hidden_size"], 1, 1)
        self.conv0.weight.data.copy_(w["perceive.proj.0.weight"])
        self.conv0.bias.data.copy_(w["perceive.proj.0.bias"])
        self.conv1.weight.data.copy_(w["perceive.proj.2.weight"])
        self.conv1.bias.data.copy_(w["perceive.proj.2.bias"])
        self.newline = torch.nn.Parameter(w["perceive.image_newline"].clone())

    def forward(self, x, gh, gw):
        x = self.norm(x)
        x = x.reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
        x = self.conv0(x)
        x = F.gelu(x)
        x = self.conv1(x)
        nl = self.newline.reshape(1, -1, 1, 1).expand(x.shape[0], -1, x.shape[2], 1)
        x = torch.cat([x, nl], dim=-1)
        x = x.reshape(1, x.shape[1], -1).permute(0, 2, 1)
        return x.reshape(1, -1, x.shape[2])


def patchify(img, patch=16, merge=2):
    # img: [1, 3, H, W] normalized
    _, c, h, w = img.shape
    gh, gw = h // patch, w // patch
    p = img.reshape(1, c, gh // merge, merge, patch, gw // merge, merge, patch)
    p = p.permute(0, 2, 3, 5, 6, 1, 4, 7)
    return p.unsqueeze(6).reshape(1, gh * gw, c * 1 * patch * patch).squeeze(0), gh, gw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=os.path.expanduser("~/.mlx-serve/models/TensorFold/Hy4-flash-MLX-4bit-MTP"))
    ap.add_argument("--out", default=os.path.expanduser("~/claude-tmp/hy4-vision"))
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.pack, "config.json")))
    proc = json.load(open(os.path.join(a.pack, "processor_config.json")))["image_processor"]
    vc = cfg["vision_config"]
    w = load_vision_weights(a.pack)
    torch.manual_seed(0)

    # A deterministic pattern (RGB, [0,255]) with the processor's bounds.
    yy, xx = np.meshgrid(np.arange(a.height), np.arange(a.width), indexing="ij")
    img = np.zeros((a.height, a.width, 3), dtype=np.uint8)
    img[..., 0] = (xx * 255 // max(a.width - 1, 1)).astype(np.uint8)
    img[..., 1] = (yy * 255 // max(a.height - 1, 1)).astype(np.uint8)
    img[..., 2] = ((xx + yy) * 255 // max(a.width + a.height - 2, 1)).astype(np.uint8)
    pix = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    factor = proc["patch_size"] * proc["merge_size"]
    h, wd = smart_resize(a.height, a.width, factor, proc["size"]["shortest_edge"], proc["size"]["longest_edge"])
    pix = torch.nn.functional.interpolate(pix, size=(h, wd), mode="bicubic", antialias=True)
    pix = (pix - 0.5) / 0.5
    patches, gh, gw = patchify(pix, proc["patch_size"], proc["merge_size"])

    embed = PatchEmbed(vc, w)
    blocks = [Block(vc, w, i) for i in range(vc["num_hidden_layers"])]
    merger = Merger(vc, w, proc["merge_size"])
    x = embed(patches, [[1, gh, gw]])
    for b in blocks:
        x = b(x)
    feat = merger(x.squeeze(0), gh, gw)  # [1, rows, 3072]

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "image.npy"), img)
    patches.numpy().astype("<f4").tofile(os.path.join(a.out, "patches.bin"))
    feat.squeeze(0).detach().numpy().astype("<f4").tofile(os.path.join(a.out, "features.bin"))
    json.dump({"hidden": int(patches.shape[1]), "out_hidden": int(feat.shape[2]),
               "gh": gh, "gw": gw, "rows": int(feat.shape[1]), "patches": int(patches.numel())},
              open(os.path.join(a.out, "meta.json"), "w"))
    print("wrote", a.out, "rows", feat.shape[1], "grid", gh, gw, "resized", h, wd)


if __name__ == "__main__":
    main()
