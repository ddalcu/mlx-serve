#!/usr/bin/env python3
"""Dump GLM-5.3-Flash (glm5_next) VISION parity fixtures — the reference EXECUTES.

Runs transformers' own `Glm5NextImageProcessorPil` and `Glm5NextVisionModel`
(fp32, CPU) on OUR pack's tower weights and saves every stage
`src/glm_vision.zig` has to reproduce. TensorFold's MLX packs ship the tower
dense bf16, so both sides read the same numbers and a diff is a layout/math bug.

Preprocessing cases start from RGB bytes decoded here (our server decodes with
stb, the reference with libjpeg; the decoders are not under test):
  house  the repo's house.jpeg: already budget-sized, so it is PADDED, never resized
  noise  a random 61x97 image: below the 16-token floor, so it is UPSCALED then padded
  wide   a random 7000x1000 image: past the 8000-token cap at a skewed aspect

Usage:
    venv/bin/python -I tests/dump_glm5_next_vision_fixtures.py \
        --model ~/.mlx-serve/models/TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP \
        --image tests/fixtures/house.jpeg \
        --out ~/claude-tmp/glm-vision/glm_vision_fixture.safetensors
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
from transformers.models.glm5_next.image_processing_pil_glm5_next import Glm5NextImageProcessorPil
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

TOWER_CASES = ("noise", "house")


def load_tower(model_dir, vcfg):
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({f for k, f in index.items() if k.startswith("vision_tower.")})
    state = {}
    for shard in shards:
        f = safe_open(os.path.join(model_dir, shard), framework="pt")
        for k in f.keys():
            if not k.startswith("vision_tower."):
                continue
            v = f.get_tensor(k)
            if k.endswith((".scales", ".biases")):
                raise SystemExit(f"{k} is quantized — this dumper expects a dense tower")
            k = k[len("vision_tower."):]
            # MLX conv layouts are channels-last; torch wants channels-first.
            if k == "patch_embed.proj.weight":
                v = v.permute(0, 4, 1, 2, 3)
            elif k == "downsample.weight":
                v = v.permute(0, 3, 1, 2)
            state[k] = v.float()
    cfg = Glm5NextVisionConfig(**{k: v for k, v in vcfg.items() if k != "model_type"})
    cfg._attn_implementation = "eager"
    model = Glm5NextVisionModel(cfg).float().eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"missing {missing[:5]} unexpected {unexpected[:5]}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_json = json.load(open(os.path.join(args.model, "config.json")))
    proc_cfg = json.load(open(os.path.join(args.model, "processor_config.json")))["image_processor"]
    proc = Glm5NextImageProcessorPil(**{k: v for k, v in proc_cfg.items() if k != "image_processor_type"})

    rng = np.random.default_rng(0x61A5)
    sources = {
        "house": np.asarray(Image.open(args.image).convert("RGB")),
        "noise": rng.integers(0, 256, size=(61, 97, 3), dtype=np.uint8),
        "wide": rng.integers(0, 256, size=(1000, 7000, 3), dtype=np.uint8),
    }

    out = {}
    pixels = {}
    for name, rgb in sources.items():
        feat = proc(images=Image.fromarray(rgb), return_tensors="pt")
        pv, grid = feat["pixel_values"], feat["image_grid_thw"][0]
        out[f"{name}_rgb"] = torch.from_numpy(np.ascontiguousarray(rgb))
        out[f"{name}_pixel_values"] = pv.float().contiguous()
        out[f"{name}_grid"] = grid[1:].to(torch.int32)
        pixels[name] = (pv.float(), grid)
        print(f"[{name}] {rgb.shape[0]}x{rgb.shape[1]} -> grid {int(grid[1])}x{int(grid[2])} ({pv.shape[0] // 4} tokens)")

    model = load_tower(args.model, cfg_json["vision_config"])
    post_ln = {}
    model.post_layernorm.register_forward_hook(lambda _m, _i, o: post_ln.__setitem__("x", o))
    for name in TOWER_CASES:
        pv, grid = pixels[name]
        with torch.no_grad():
            res = model(pv, grid_thw=grid[None])
        out[f"{name}_post_ln"] = post_ln["x"].contiguous()
        out[f"{name}_downsample"] = res.last_hidden_state.contiguous()
        out[f"{name}_features"] = res.pooler_output.contiguous()
        print(f"[{name}] features {tuple(res.pooler_output.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out, args.out)
    print(f"wrote {args.out} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
