#!/usr/bin/env python3
"""Dump LFM2-VL VISION TOWER parity fixtures — the reference EXECUTES.

Runs transformers' own `Siglip2VisionModel` plus LFM2-VL's projector on OUR
pack's weights and saves every stage `src/lfm2_vision.zig` has to reproduce.
The tower's output is spliced straight into the LM sequence, so a wrong patch
order, position resample or unshuffle interleave produces conditioning that is
silently WRONG rather than absent.

LiquidAI's MLX packs quantize only the language model — `vision_tower.*` and
`multi_modal_projector.*` ship dense bf16 in every width — so both sides read
the same numbers and a diff is a layout/math bug, never quantization error.

The position resample is the delicate stage: transformers uses
`F.interpolate(mode="bilinear", antialias=True)`, which for a grid axis SHORTER
than 16 is a real downscale. (mlx-vlm 0.6.3 uses bicubic there instead —
cos 0.99 and rms 1.13x against this, so it is not a usable oracle.)

Usage:
    uv run --with torch --with torchvision --with numpy --with safetensors \
        --with 'transformers>=5.15' \
        tests/dump_lfm2_vision_fixtures.py \
        --model "/Volumes/G Drive SSD/models-dl/LiquidAI/LFM2.5-VL-3B-MLX-4bit" \
        --out ~/claude-tmp/lfm2-vision/lfm2_vision_fixture.safetensors
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel

# Source image sizes chosen for what the PATCH GRID does to the position table.
# 33x47 lands on a 14x20 grid: the height axis DOWNSCALES from the stored 16
# (where antialias is load-bearing) while the width axis upscales, so one
# forward covers both directions. 512x512 is the 32x32 token ceiling.
CASES = {"a": (33, 47), "b": (512, 512)}

# Extra position-resample-only grids: the widest aspect the token budget allows
# in each direction, where an antialias-free filter diverges most.
POS_GRIDS = [(14, 20), (32, 32), (32, 8), (8, 32), (26, 36), (16, 64)]


def load_tensors(model_dir, want):
    """Every tensor whose key contains `want`, as torch fp32 under bare keys."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        index = json.load(open(index_path))["weight_map"]
        shards = sorted({f for k, f in index.items() if want in k})
    else:
        shards = ["model.safetensors"]
    out = {}
    for shard in shards:
        for k, v in load_file(os.path.join(model_dir, shard)).items():
            if want in k:
                if k.endswith((".scales", ".biases")):
                    raise SystemExit(f"{k} is quantized — this dumper expects a dense {want}")
                out[k] = v.float()
    if not out:
        raise SystemExit(f"no {want} tensors in {model_dir}")
    return out


def build_tower(model_dir, vcfg_json):
    cfg = Siglip2VisionConfig(
        hidden_size=vcfg_json["hidden_size"],
        intermediate_size=vcfg_json["intermediate_size"],
        num_hidden_layers=vcfg_json["num_hidden_layers"],
        num_attention_heads=vcfg_json["num_attention_heads"],
        num_channels=vcfg_json.get("num_channels", 3),
        patch_size=vcfg_json["patch_size"],
        num_patches=vcfg_json.get("num_patches", 256),
        layer_norm_eps=vcfg_json.get("layer_norm_eps", 1e-6),
        hidden_act=vcfg_json.get("hidden_act", "gelu_pytorch_tanh"),
        vision_use_head=vcfg_json.get("vision_use_head", False),
        attention_dropout=0.0,
    )
    model = Siglip2VisionModel(cfg).eval()
    raw = load_tensors(model_dir, "vision_tower")
    state = {k[len("vision_tower.") :]: v for k, v in raw.items()}
    # transformers 5.x: Siglip2VisionModel IS the tower (no `.vision_model`
    # wrapper), and it still builds an attention-pool `head` that this
    # checkpoint does not ship (`vision_use_head: false`) and never runs.
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("head.")]
    if unexpected:
        raise SystemExit(f"unexpected vision keys: {unexpected[:5]}")
    if missing:
        raise SystemExit(f"missing vision keys: {missing[:5]}")
    return model, cfg


def project(feature_hw_c, proj, factor):
    """LFM2-VL's Lfm2VlMultiModalProjector on a [1, h, w, C] feature map."""
    b, w, h, c = feature_hw_c.shape  # reference's own (mis)naming, ops verbatim
    x = feature_hw_c.reshape(b, w, h // factor, c * factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(b, h // factor, w // factor, c * factor**2)
    x = x.permute(0, 2, 1, 3)
    x = F.linear(x, proj["multi_modal_projector.linear_1.weight"], proj["multi_modal_projector.linear_1.bias"])
    x = F.gelu(x)
    return F.linear(x, proj["multi_modal_projector.linear_2.weight"], proj["multi_modal_projector.linear_2.bias"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_json = json.load(open(os.path.join(args.model, "config.json")))
    vcfg = cfg_json["vision_config"]
    patch = vcfg["patch_size"]
    factor = cfg_json["downsample_factor"]
    model, cfg = build_tower(args.model, vcfg)
    proj = load_tensors(args.model, "multi_modal_projector")

    # The processor is only used for the RESIZE decision; the patchify + the
    # normalization are reproduced here so the dumped pixel_values are the
    # exact bytes our decoder must build (and so a torchvision resample
    # difference can be attributed rather than silently absorbed).
    from transformers.models.lfm2_vl.image_processing_lfm2_vl import Lfm2VlImageProcessor

    ip = Lfm2VlImageProcessor(**json.load(open(os.path.join(args.model, "processor_config.json")))["image_processor"])

    out = {}
    for name, (src_h, src_w) in CASES.items():
        rng = np.random.default_rng(0x1F2B if name == "a" else 0x5EED)
        img_u8 = rng.integers(0, 256, size=(3, src_h, src_w), dtype=np.uint8)
        img = torch.from_numpy(img_u8).float()[None]

        tw, th = ip.smart_resize(src_h, src_w, factor, ip.min_image_tokens, ip.max_image_tokens, patch)
        if ip._is_image_too_large(src_h, src_w, ip.max_image_tokens, patch, factor, ip.max_pixels_tolerance):
            raise SystemExit(f"case {name} ({src_h}x{src_w}) tiles — pick a single-tile size")

        from torchvision.transforms.v2 import functional as tvF

        resized = tvF.resize(img, [th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
        resized = resized.clamp(0, 255) / 255.0
        mean = torch.tensor(ip.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(ip.image_std).view(1, 3, 1, 1)
        resized = (resized - mean) / std

        gh, gw = th // patch, tw // patch
        # (B, C, gh, p, gw, p) -> (B, gh, gw, p, p, C): channel is the INNERMOST
        # feature axis, unlike every other tower we serve.
        patches = resized.reshape(1, 3, gh, patch, gw, patch).permute(0, 2, 4, 3, 5, 1).reshape(1, gh * gw, -1)

        spatial = torch.tensor([[gh, gw]], dtype=torch.long)
        mask = torch.ones((1, gh * gw), dtype=torch.long)
        with torch.no_grad():
            hidden = model(pixel_values=patches, spatial_shapes=spatial, pixel_attention_mask=mask).last_hidden_state
            projected = project(hidden.reshape(1, gh, gw, -1), proj, factor)
            feats = projected.reshape(-1, projected.shape[-1])

        out[f"{name}_pixel_values"] = patches[0].contiguous()
        out[f"{name}_grid"] = torch.tensor([gh, gw], dtype=torch.int32)
        out[f"{name}_src"] = torch.tensor([src_h, src_w], dtype=torch.int32)
        out[f"{name}_image_u8"] = torch.from_numpy(img_u8).contiguous()
        out[f"{name}_hidden"] = hidden[0].contiguous()
        out[f"{name}_features"] = feats.contiguous()
        print(f"[{name}] {src_h}x{src_w} -> {th}x{tw} grid {gh}x{gw} "
              f"({gh * gw} patches, {(gh // factor) * (gw // factor)} tokens) features {tuple(feats.shape)}")

    pe = model.embeddings.position_embedding.weight
    side = int(cfg.num_patches**0.5)
    pe = pe.reshape(side, side, -1).permute(2, 0, 1)[None]
    for gh, gw in POS_GRIDS:
        with torch.no_grad():
            r = F.interpolate(pe, size=(gh, gw), mode="bilinear", align_corners=False, antialias=True)
        out[f"pos_{gh}x{gw}"] = r.reshape(pe.shape[1], gh * gw).transpose(0, 1).contiguous()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out, args.out)
    print(f"wrote {args.out} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
