#!/usr/bin/env python3
"""Build a Qwen3-VL vision parity fixture for mlx-serve's QwenVision encoder.

Runs the mlx-vlm reference vision tower on a fixture image and dumps:
  - pixel_values.bin   float32 [N, C*tps*ps*ps]   (the processor's patches)
  - ref_embeds.bin     float32 [N_merged, out_hidden]  (post-merger embeddings)
  - manifest.json      shapes + grid_thw (full patch grid [t, gh, gw])

The Zig side (tests/test_qwen_vision_parity.sh → a QWEN_VISION_FIXTURE-gated test)
loads the same checkpoint, runs QwenVision.forward(pixel_values, gh, gw), and
asserts max-abs-diff against ref_embeds. Feeding the reference pixel_values
straight in isolates the ViT math from preprocessing — the same "reference
intermediate → our impl" pattern used for the Gemma 4 vision and MTP parity tests.

Usage:
  python3 tests/build_qwen_vision_fixture.py \
      --model ~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit \
      --image tests/fixtures/house.jpeg \
      --out   /tmp/qwen_vision_fixture
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default="tests/fixtures/house.jpeg")
    ap.add_argument("--out", default="/tmp/qwen_vision_fixture")
    args = ap.parse_args()

    model_path = os.path.expanduser(args.model)
    os.makedirs(args.out, exist_ok=True)

    import mlx.core as mx
    from mlx_vlm.utils import load
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor
    from PIL import Image

    # 1. Preprocess with the LOCAL processor whose patchify layout QwenVision mirrors.
    proc = Qwen3VLImageProcessor()
    img = Image.open(args.image).convert("RGB")
    out = proc([img])  # pass PIL; the processor's _to_numpy_image does HWC→CHW
    pixel_values = np.asarray(out["pixel_values"], dtype=np.float32)  # [N, 1536]
    grid_thw = np.asarray(out["image_grid_thw"], dtype=np.int64)      # [1, 3] = [t, gh, gw]
    t, gh, gw = (int(x) for x in grid_thw[0])
    print(f"pixel_values={pixel_values.shape} grid_thw={grid_thw.tolist()} (t={t} gh={gh} gw={gw})")

    # 2. Reference vision tower forward.
    model, _ = load(model_path)
    vt = getattr(model, "vision_tower", None) or getattr(model.model, "vision_tower")
    embeds, _deepstack = vt(mx.array(pixel_values), mx.array(grid_thw))
    embeds = np.asarray(embeds.astype(mx.float32))  # [N_merged, out_hidden]
    print(f"ref_embeds={embeds.shape}")

    pixel_values.tofile(os.path.join(args.out, "pixel_values.bin"))
    embeds.astype(np.float32).tofile(os.path.join(args.out, "ref_embeds.bin"))
    manifest = {
        "model": model_path,
        "image": args.image,
        "grid_thw": grid_thw[0].tolist(),  # [t, gh, gw] full patch grid
        "pixel_values_shape": list(pixel_values.shape),
        "ref_embeds_shape": list(embeds.shape),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote fixture to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
