#!/usr/bin/env python3
"""Dump MiniCPM-V 4.6 VISION TOWER parity fixtures — the reference EXECUTES.

Runs mlx-vlm's own `minicpmv4_6` modules (VisionModel + VitMerger + Merger) on
OUR pack's weights and saves every stage `src/minicpm_vision.zig` has to
reproduce. mlx-vlm 0.5.0+ is the implementation the mlx-community conversion
was made WITH, so this is the conversion's own math — a diff is a layout or
ordering bug in the port, never quantization noise (the tower ships dense bf16
in this pack; only the language model is quantized).

The delicate stage is the vit_merger: a 2x2 window attention + fold applied at
tower layer `insert_layer_id`, whose windows run [py, px] within each 2x2
block — a transposed window order or a swapped fold residual changes the
features while keeping every SHAPE identical, so the zig side asserts VALUES.

Inputs are generated as [n, 588] patch features (row-major patches, per-patch
[py, px, c] channel-innermost) — the exact layout `lfm2_vision.buildPixelValues`
emits and the packed path in mlx-vlm's `get_vision_embedding` consumes — so
both sides read bit-identical inputs without an image decoder.

Usage:
    tests/dump_minicpm_vision_fixtures.py \
        --model ~/models/mlx-community/MiniCPM-V-4.6-4bit \
        --out ~/claude-tmp/minicpm-vision/minicpm_vision_fixture.safetensors
"""

import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(__file__))
# Vendored copy of the reference modules (see the file header of this dir in
# the env): config + vision + the mergers, imported WITHOUT the package's
# model __init__ so the 0.5.0 release's missing qwen3_5 helpers never load.
import importlib

def _load_reference_modules():
    import mlx_vlm.models.minicpmv4_6.config as config
    import mlx_vlm.models.minicpmv4_6.vision as vision
    import mlx_vlm.models.minicpmv4_6.minicpmv4_6 as model
    return config, vision, model

config_mod, vision_mod, model_mod = _load_reference_modules()

# (grid_h, grid_w): the 448px source view, one slice cell, and a non-square
# source — every grid a real request produces must be even on both axes.
GRIDS = {"a": (32, 32), "b": (40, 28), "c": (28, 32)}

def load_vision_tensors(model_dir):
    """The tower + mergers' tensors under their own namespaces, as mx arrays."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        index = json.load(open(index_path))["weight_map"]
        shards = sorted({
            f for k, f in index.items()
            if k.startswith(("vision_tower.", "vit_merger.", "merger."))
        })
    else:
        shards = ["model.safetensors"]
    # mx.load speaks safetensors natively and keeps bf16 (numpy can't).
    weights = {}
    for shard in shards:
        for k, v in mx.load(os.path.join(model_dir, shard)).items():
            weights[k] = v
    return weights

def build_reference(config_path):
    cfg = json.load(open(config_path))
    vc = cfg["vision_config"]
    vision_config = config_mod.VisionConfig(
        model_type=vc.get("model_type", "minicpmv4_6_vision"),
        hidden_size=vc["hidden_size"],
        intermediate_size=vc["intermediate_size"],
        num_hidden_layers=vc["num_hidden_layers"],
        num_attention_heads=vc["num_attention_heads"],
        num_channels=vc.get("num_channels", 3),
        image_size=vc.get("image_size", 448),
        patch_size=vc["patch_size"],
        hidden_act=vc.get("hidden_act", "gelu_pytorch_tanh"),
        layer_norm_eps=vc.get("layer_norm_eps", 1e-6),
    )
    tower = vision_mod.VisionModel(vision_config)
    insert_layer_id = int(cfg.get("insert_layer_id", 6))
    use_vit_merger = str(cfg.get("downsample_mode", "16x")) != "4x"
    vit = model_mod.VitMerger(
        vision_hidden_size=vision_config.hidden_size,
        merged_hidden_size=vision_config.window_intermediate_size,
        num_heads=vision_config.num_attention_heads,
        merge_group_size=vision_config.window_kernel_size,
    )
    merger = model_mod.Merger(
        hidden_size=vision_config.hidden_size,
        out_size=cfg["text_config"]["hidden_size"],
        merger_times=int(cfg.get("merger_times", 1) or 1),
        merge_kernel_size=tuple(cfg.get("merge_kernel_size", (2, 2))),
    )
    return tower, vit, merger, insert_layer_id, use_vit_merger

def run_case(name, gh, gw, tower, vit, merger, insert_layer_id, use_vit_merger, out):
    rng = np.random.default_rng(hash(name) % (2**32))
    n = gh * gw
    patches = rng.standard_normal((n, 3 * 14 * 14)).astype(np.float32) * 0.5
    out[f"{name}_patches"] = patches

    # [n, 588] -> packed (C, patch, n*patch): patch p row-major, per-patch
    # [py, px, c]. mlx-vlm's packed path reshapes (14, n*14, 3) back into
    # per-patch features, so this is the bit-identical input.
    packed = patches.reshape(n, 14, 14, 3).transpose(1, 0, 2, 3).reshape(14, n * 14, 3).transpose(2, 0, 1)
    cur = mx.array(packed)[None]  # [1, C, patch, n*patch]

    # mlx-vlm's get_vision_embedding: (3, 14, n*14) -> (14, n*14, 3) HWC,
    # batched -> the packed-embedding path keys on height == patch.
    cur_pixels = mx.expand_dims(cur[0].transpose(1, 2, 0), 0)
    cur_tgt = mx.array([[gh, gw]], dtype=mx.int32)

    hidden = tower.embeddings(cur_pixels, tgt_sizes=cur_tgt)
    hidden = hidden.astype(tower.embeddings.patch_embedding.weight.dtype)
    grid_h, grid_w = gh, gw
    for layer_index, layer in enumerate(tower.encoder.layers):
        hidden = layer(hidden, attention_mask=None)
        if use_vit_merger and layer_index == insert_layer_id:
            merged, grid_h, grid_w = vit(hidden[0], grid_h, grid_w)
            hidden = mx.expand_dims(merged, 0)
    hidden = tower.post_layernorm(hidden)
    feats, final_h, final_w = merger(hidden[0], grid_h, grid_w)
    mx.eval(hidden, feats)

    # bf16 has no numpy buffer protocol — widen on the mlx side first.
    out[f"{name}_hidden"] = np.array(hidden[0].astype(mx.float32), dtype=np.float32)
    out[f"{name}_grid"] = np.array([grid_h, grid_w], dtype=np.int32)
    out[f"{name}_features"] = np.array(feats.astype(mx.float32), dtype=np.float32)
    print(f"[{name}] grid {gh}x{gw} -> merged {grid_h}x{grid_w}, {feats.shape[0]} tokens")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tower, vit, merger, insert_layer_id, use_vit_merger = build_reference(
        os.path.join(args.model, "config.json"))
    weights = load_vision_tensors(args.model)
    tower.load_weights(
        [(k.removeprefix("vision_tower."), v) for k, v in weights.items()
         if k.startswith("vision_tower.")], strict=True)
    vit.load_weights(
        [(k.removeprefix("vit_merger."), v) for k, v in weights.items()
         if k.startswith("vit_merger.")], strict=True)
    merger.load_weights(
        [(k.removeprefix("merger."), v) for k, v in weights.items()
         if k.startswith("merger.")], strict=True)
    mx.eval(tower.parameters(), vit.parameters(), merger.parameters())

    out = {}
    for name, (gh, gw) in GRIDS.items():
        run_case(name, gh, gw, tower, vit, merger, insert_layer_id, use_vit_merger, out)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out, args.out)
    print(f"wrote {args.out} ({len(out)} tensors)")

if __name__ == "__main__":
    main()
