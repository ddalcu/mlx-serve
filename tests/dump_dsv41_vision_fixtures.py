#!/usr/bin/env python3
"""Dump DeepSeek-V4.1-Flash VISION parity fixtures — the reference EXECUTES.

Runs the release's own `inference/image_processor.py` and `inference/vision.py` (ViT + Aligner,
fp32, CPU) on OUR pack's dense bf16 tower weights and saves every stage `src/plugins/deepseek_v41/vision.zig`
has to reproduce, plus the span the trunk reads (delimiter embeddings in place).

Preprocessing cases start from RGB bytes decoded here (our server decodes with stb, the
reference with libjpeg; the decoders are not under test):
  house  the repo's house.jpeg: fits the budget, so it is aligned up and letterboxed
  noise  a random 61x97 image: below min_pixels, so it is scaled up first
  wide   a random 1000x7000 image: past the 1024-token budget

Usage:
    venv/bin/python -I tests/dump_dsv41_vision_fixtures.py \
        --release "/Volumes/G Drive SSD/model-src/DeepSeek-V4.1-Flash-1017" \
        --model /Volumes/4TB/scratch/DeepSeek-V4.1-Flash-1017-MLX-mxfp4-mtp \
        --image tests/fixtures/house.jpeg \
        --out ~/claude-tmp/dsv41-vision/fixture.safetensors
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

TOWER_CASES = ("noise", "house")
VISION_KEYS = ("vision.", "aligner.", "image_start", "image_end", "image_newline")


def vision_args(cfg):
    vc = cfg["vision_config"]
    return SimpleNamespace(
        dim=cfg["text_config"]["hidden_size"] if "text_config" in cfg else cfg["hidden_size"],
        vision_n_layers=vc["num_hidden_layers"],
        vision_dim=vc["hidden_size"],
        vision_n_heads=vc["num_attention_heads"],
        vision_inter_dim=vc["intermediate_size"],
        vision_patch_size=vc["patch_size"],
        vision_rope_theta=float(vc.get("rope_theta", 10000.0)),
        vision_downsample_ratio=vc["downsample_ratio"],
        vision_max_n_token=vc["max_image_tokens"],
        vision_min_pixels=vc["min_pixels"],
        vision_max_wh_ratio=vc.get("max_wh_ratio"),
    )


def load_tower(model_dir, args, vision_mod):
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({f for k, f in index.items() if k.startswith(VISION_KEYS)})
    state = {}
    for shard in shards:
        f = safe_open(os.path.join(model_dir, shard), framework="pt")
        for k in f.keys():
            if not k.startswith(VISION_KEYS):
                continue
            if k.endswith((".scales", ".biases")):
                raise SystemExit(f"{k} is quantized — this dumper expects a dense tower")
            state[k] = f.get_tensor(k).float()
    vit = vision_mod.ViT(args).float().eval()
    # convert.py renamed the blocks' `mlp` to `ffn`; vision.py keeps `mlp`.
    vit_state = {k[len("vision."):].replace(".ffn.", ".mlp."): v for k, v in state.items() if k.startswith("vision.")}
    missing, unexpected = vit.load_state_dict(vit_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"vit missing {missing[:5]} unexpected {unexpected[:5]}")
    aligner = vision_mod.Aligner(args).float().eval()
    missing, unexpected = aligner.load_state_dict({k[len("aligner."):]: v for k, v in state.items() if k.startswith("aligner.")}, strict=False)
    if missing or unexpected:
        raise SystemExit(f"aligner missing {missing[:5]} unexpected {unexpected[:5]}")
    return vit, aligner, state["image_start"], state["image_end"], state["image_newline"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, help="the DeepSeek release dir (its inference/ is imported)")
    ap.add_argument("--model", required=True, help="an MLX pack with the dense tower")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.release, "inference"))
    import image_processor as ip  # noqa: E402
    import vision as vision_mod  # noqa: E402

    cfg = json.load(open(os.path.join(a.model, "config.json")))
    args = vision_args(cfg)
    rng = np.random.default_rng(0x61A5)
    sources = {
        "house": np.asarray(Image.open(a.image).convert("RGB")),
        "noise": rng.integers(0, 256, size=(61, 97, 3), dtype=np.uint8),
        "wide": rng.integers(0, 256, size=(1000, 7000, 3), dtype=np.uint8),
    }
    out = {}
    pix = {}
    for name, rgb in sources.items():
        import io
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")  # lossless: the record's bytes decode to `rgb`
        patches, vit_h, vit_w, llm_h, llm_w = ip.load_image({"data": buf.getvalue()}, args)
        out[f"{name}_rgb"] = torch.from_numpy(np.ascontiguousarray(rgb))
        out[f"{name}_patches"] = patches.float().reshape(patches.shape[0], -1).contiguous()
        out[f"{name}_grid"] = torch.tensor([vit_h, vit_w, llm_h, llm_w], dtype=torch.int32)
        pix[name] = (patches.float(), vit_h, vit_w, llm_h, llm_w)
        print(f"[{name}] {rgb.shape[1]}x{rgb.shape[0]} -> vit {vit_h}x{vit_w}, llm {llm_h}x{llm_w} ({ip.num_image_tokens(llm_h, llm_w)} tokens)")

    vit, aligner, start, end, newline = load_tower(a.model, args, vision_mod)
    for name in TOWER_CASES:
        patches, vit_h, vit_w, llm_h, llm_w = pix[name]
        with torch.no_grad():
            tower = vit(patches, vit_h, vit_w)
            feats = aligner(tower, vit_h, vit_w)
        types = ip.image_token_types(llm_h, llm_w)
        span = torch.empty(types.numel(), feats.shape[-1])
        span[types == ip.IMAGE_START] = start
        span[types == ip.IMAGE_END] = end
        span[types == ip.IMAGE_NEW_LINE] = newline
        span[types == ip.IMAGE] = feats
        out[f"{name}_tower"] = tower.contiguous()
        out[f"{name}_features"] = feats.contiguous()
        out[f"{name}_span"] = span.unsqueeze(0).contiguous()
        print(f"[{name}] tower {tuple(tower.shape)} features {tuple(feats.shape)} span {tuple(span.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(out, a.out)
    print(f"wrote {a.out} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
