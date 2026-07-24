#!/usr/bin/env python3
"""Build a CPU-only Qwen image-preprocessing parity fixture.

The fixture starts from decoded RGB bytes and captures the reference
processor's resized/normalized/patchified pixel_values. It deliberately does
not load a model or run the vision tower.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="tests/fixtures/house.jpeg")
    parser.add_argument("--out", default="/tmp/qwen_preprocess_fixture")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--temporal-patch-size", type=int, default=2)
    parser.add_argument("--merge-size", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=56 * 56)
    parser.add_argument("--max-pixels", type=int, default=14 * 14 * 4 * 1280)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    source_rgb = np.asarray(image, dtype=np.uint8)
    source_h, source_w, channels = source_rgb.shape
    if channels != 3:
        raise ValueError(f"expected RGB input, got shape {source_rgb.shape}")

    processor = Qwen3VLImageProcessor(
        patch_size=args.patch_size,
        temporal_patch_size=args.temporal_patch_size,
        merge_size=args.merge_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    output = processor([image])
    pixel_values = np.asarray(output["pixel_values"], dtype=np.float32)
    grid_thw = np.asarray(output["image_grid_thw"], dtype=np.int64)
    _, grid_h, grid_w = (int(value) for value in grid_thw[0])
    resized_h = grid_h * args.patch_size
    resized_w = grid_w * args.patch_size

    os.makedirs(args.out, exist_ok=True)
    source_rgb.tofile(os.path.join(args.out, "source_rgb.bin"))
    pixel_values.tofile(os.path.join(args.out, "pixel_values.bin"))
    manifest = {
        "source_height": source_h,
        "source_width": source_w,
        "resized_height": resized_h,
        "resized_width": resized_w,
        "patch_size": args.patch_size,
        "temporal_patch_size": args.temporal_patch_size,
        "merge_size": args.merge_size,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "pixel_values_length": int(pixel_values.size),
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(
        f"source={source_w}x{source_h} resized={resized_w}x{resized_h} "
        f"grid={grid_w}x{grid_h} pixel_values={pixel_values.shape}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
