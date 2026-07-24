#!/usr/bin/env python3
"""Dump a MageFlow Qwen3-VL vision-tower parity fixture for src/mage_flow.zig (E7.2).

Runs the pure-MLX REFERENCE (ivanfioravanti/mflux@mage-flow-mlx) vision model in
FP32 over the reference PROCESSOR's pixel_values for one synthetic image, and
writes one safetensors the Zig oracle loads (env MAGEFLOW_VIT_FIXTURE). Feeding
the reference pixel_values decouples the tower from preprocessing (E7.3). Tensors:
    pixel_values [Npatch, 1536]  f32 — the ViT input (verbatim to Zig)
    grid_thw     [1, 3]  int32       — (t, gh, gw)
    merged       [Ntok, 2560]  f32   — visual(...) merged features
    deepstack0/1/2 [Ntok, 2560] f32  — the 3 DeepStack merger outputs

USER-RUN (accepts the MIT Turbo license, has the checkpoint downloaded):
    <mflux>/.venv/bin/python tests/dump_mageflow_vit_fixture.py \
        ~/.mlx-serve/models/microsoft/Mage-Flow-Turbo <mflux_repo_root> [OUT]

Then:
    MAGEFLOW_VIT_FIXTURE=<OUT>/mageflow_vit.safetensors \
    zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow ViT"
"""

import glob
import os
import sys

import mlx.core as mx
import numpy as np
from PIL import Image


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    model_dir = os.path.abspath(sys.argv[1])
    ref_root = os.path.abspath(sys.argv[2])
    out_dir = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.abspath("mageflow_fixtures")
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, os.path.join(ref_root, "src"))

    from mlx.utils import tree_unflatten

    from mflux.models.mage_flow.model.mage_flow_text_encoder.processor import (
        MageFlowQwen3VLImageProcessor,
    )
    from mflux.models.mage_flow.model.mage_flow_text_encoder.vision_model import (
        MageFlowQwen3VLVisionModel,
    )
    from mflux.models.mage_flow.weights.mage_flow_weight_mapping import MageFlowWeightMapping

    te_dir = os.path.join(model_dir, "text_encoder")

    # ── Build the vision tower and load only the visual.* weights (fp32) ──
    visual = MageFlowQwen3VLVisionModel(
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=1024,
        num_heads=16,
        intermediate_size=4096,
        depth=24,
        spatial_merge_size=2,
        num_position_embeddings=2304,
        out_hidden_size=2560,
        deepstack_visual_indexes=(5, 11, 17),
    )
    flat = {}
    for shard in sorted(glob.glob(os.path.join(te_dir, "*.safetensors"))):
        for k, v in mx.load(shard).items():
            if not k.startswith("model.visual."):
                continue
            mapped = MageFlowWeightMapping.transform_text_encoder_key(k)  # -> visual.*
            if mapped is None:
                continue
            v = MageFlowWeightMapping.transform_text_encoder_weight(mapped, v)
            flat[mapped[len("visual.") :]] = v.astype(mx.float32)
    visual.update(tree_unflatten(list(flat.items())))
    mx.eval(visual.parameters())
    print(f"[dump] loaded {len(flat)} visual tensors (fp32)")

    # ── Reference processor → pixel_values + grid_thw for one synthetic image ──
    rng = np.random.default_rng(11)
    ys = np.linspace(0, 255, 224, dtype=np.float32)[:, None, None]
    xs = np.linspace(0, 255, 224, dtype=np.float32)[None, :, None]
    img = (0.5 * ys + 0.5 * xs + 20.0 * rng.standard_normal((224, 224, 3))).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(np.broadcast_to(img, (224, 224, 3)).copy())
    proc = MageFlowQwen3VLImageProcessor(max_long_edge=384)
    pixel_values, grid_thw = proc.preprocess([pil])
    pixel_values = mx.array(pixel_values).astype(mx.float32)
    grid_thw = mx.array(grid_thw)
    print(f"[dump] pixel_values={pixel_values.shape} grid_thw={np.asarray(grid_thw).tolist()}")

    merged, deepstack = visual(pixel_values, grid_thw, return_deepstack=True)
    mx.eval(merged, *deepstack)
    print(f"[dump] merged={merged.shape}; deepstack={[d.shape for d in deepstack]}")

    fixture = {
        "pixel_values": pixel_values.astype(mx.float32),
        "grid_thw": grid_thw.astype(mx.int32),
        "merged": merged.astype(mx.float32),
        "deepstack0": deepstack[0].astype(mx.float32),
        "deepstack1": deepstack[1].astype(mx.float32),
        "deepstack2": deepstack[2].astype(mx.float32),
    }
    out_path = os.path.join(out_dir, "mageflow_vit.safetensors")
    mx.save_safetensors(out_path, fixture)
    print(f"[dump] → {out_path}")
    print("\nRun the Zig oracle:")
    print(f"  MAGEFLOW_VIT_FIXTURE={out_path} \\")
    print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow ViT"')


if __name__ == "__main__":
    main()
