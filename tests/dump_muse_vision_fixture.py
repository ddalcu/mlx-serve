#!/usr/bin/env python3
"""Dump Muse-Glimmer VISION TOWER parity fixtures — the reference EXECUTES.

Runs `transformers.models.muse_glimmer`'s own vision half (tower + adapter +
projection + the weight-less perception norm) on OUR checkpoint's weights and
saves every stage src/muse_vision.zig has to reproduce. The tower's output is
spliced straight into the LM sequence, so a wrong window order, rotary axis or
pixel-shuffle interleave produces conditioning that is silently WRONG rather
than absent.

Our packs store the tower QUANTIZED (affine, group 64). The weights are
dequantized through mlx first so a diff is a layout/math bug and never
quantization error — at 8 bits `dequantize(pack)` is exact, so both sides see
the same numbers.

Usage:
    uv run --with torch --with mlx --with numpy --with safetensors \
        --with 'transformers>=5.15' \
        tests/dump_muse_vision_fixture.py \
        --model ~/.mlx-serve/models/ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit \
        --out ~/claude-tmp/muse-vision/muse_vision_fixture.safetensors
"""

import argparse
import json
import os

import mlx.core as mx
import numpy as np
import torch
from safetensors.torch import save_file

from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerVisionConfig
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerRMSNorm,
    MuseGlimmerVisionAdapter,
    MuseGlimmerVisionModel,
)

# Both dims must EXCEED the 32-patch window so the permutation is exercised:
# this grid produces four windows of 1024 / 128 / 64 / 8 patches. A grid that
# fits one window makes window and full attention identical and proves nothing.
GRID_H, GRID_W = 34, 36


def load_vision_weights(model_dir):
    """Every `*vision*` tensor, dequantized, as torch fp32 under bare keys."""
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    shards = {f for k, f in index.items() if "vision" in k}
    raw = {}
    for shard in shards:
        for k, v in mx.load(os.path.join(model_dir, shard)).items():
            if "vision" in k:
                raw[k] = v

    quant = json.load(open(os.path.join(model_dir, "config.json")))["quantization"]
    out = {}
    for k, v in raw.items():
        if k.endswith((".scales", ".biases")):
            continue
        stem = k[: -len(".weight")] if k.endswith(".weight") else k
        if stem + ".scales" in raw:
            v = mx.dequantize(v, raw[stem + ".scales"], raw[stem + ".biases"],
                              group_size=quant["group_size"], bits=quant["bits"])
        bare = k[len("model."):] if k.startswith("model.") else k
        out[bare] = torch.from_numpy(np.array(v.astype(mx.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model_dir = os.path.expanduser(args.model)

    cfg_json = json.load(open(os.path.join(model_dir, "config.json")))
    vcfg = MuseGlimmerVisionConfig(**cfg_json["vision_config"])
    w = load_vision_weights(model_dir)

    torch.manual_seed(0)
    tower = MuseGlimmerVisionModel(vcfg).float().eval()
    tower.load_state_dict({k[len("vision_tower."):]: v for k, v in w.items()
                           if k.startswith("vision_tower.")})

    class Outer:
        out_hidden_size = cfg_json["out_hidden_size"]
        projector_hidden_size = cfg_json["projector_hidden_size"]
        projector_hidden_act = cfg_json["projector_hidden_act"]

    adapter = MuseGlimmerVisionAdapter(Outer()).float().eval()
    adapter.load_state_dict({k[len("vision_adapter."):]: v for k, v in w.items()
                             if k.startswith("vision_adapter.")})
    projection = torch.nn.Linear(Outer.projector_hidden_size,
                                 cfg_json["text_config"]["hidden_size"], bias=False).float().eval()
    projection.load_state_dict({"weight": w["vision_projection.weight"]})
    perception_norm = MuseGlimmerRMSNorm(eps=cfg_json["text_config"]["rms_norm_eps"], with_scale=False)

    patch_dim = vcfg.patch_temporal * 3 * vcfg.patch_size**2
    g = torch.Generator().manual_seed(7)
    pixel_values = torch.rand(GRID_H * GRID_W, patch_dim, generator=g) * 2 - 1
    grid_thw = torch.tensor([[1, GRID_H, GRID_W]])

    with torch.no_grad():
        pos_embeds = tower.patch_embedder(pixel_values, grid_thw) - pixel_values @ tower.patch_embedder.patch_embedding.weight.T
        merged = tower(pixel_values=pixel_values, grid_thw=grid_thw).last_hidden_state
        features = perception_norm(projection(adapter(merged)))

    save_file({
        "pixel_values": pixel_values.contiguous(),
        "grid_thw": grid_thw.to(torch.int32).contiguous(),
        "pos_embeds": pos_embeds.contiguous(),
        "merged": merged.contiguous(),
        "features": features.contiguous(),
    }, os.path.expanduser(args.out))
    print(f"grid {GRID_H}x{GRID_W} -> merged {tuple(merged.shape)} features {tuple(features.shape)}")
    print(f"features rms {features.pow(2).mean().sqrt():.4f}")


if __name__ == "__main__":
    main()
