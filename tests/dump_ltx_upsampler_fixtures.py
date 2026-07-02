#!/usr/bin/env python3
"""Dump LTX-2 LATENT-UPSAMPLER parity fixtures (.raw) for the Zig oracle test in src/ltx_video.zig.

Runs the ltx_core_mlx REFERENCE ``LatentUpsampler`` (spatial x2) on a fixed
random latent and writes byte-exact golden tensors that `zig build test`
compares the Zig ``upsampleLatentX2`` port against (cosine parity > 0.999).

USER-RUN: install the reference package (`ltx_core_mlx`, MLX) from a
ltx-2-mlx checkout. NOT run in CI; the hermetic groupNorm/pixelShuffle unit
tests in src/ltx_video.zig cover the port's math without it.

Usage:
    python3 tests/dump_ltx_upsampler_fixtures.py <spatial_upscaler_x2_v1_1.safetensors> [OUT_DIR]

Writes (flat little-endian f32):
    latent.raw    [1, 128, 3, 4, 5]   the un-normalized input latent (BCFHW)
    upscaled.raw  [1, 128, 3, 8, 10]  reference upsampler(latent)

Then it prints the `export LTX_UP_*` block to paste before:
    LTX_TEST_MODEL=<model_dir> LTX_UP_LATENT=…/latent.raw LTX_UP_OUT=…/upscaled.raw \
      zig build test -Doptimize=ReleaseFast -Dtest-filter="upsampler"
"""

import json
import os
import sys

import mlx.core as mx

SEED = 0
SHAPE = (1, 128, 3, 4, 5)  # B, C, F, H, W


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ckpt = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./ltx_upsampler_fixtures"
    os.makedirs(out_dir, exist_ok=True)

    from ltx_core_mlx.model.upsampler import LatentUpsampler
    from ltx_core_mlx.utils.weights import load_split_safetensors

    cfg_path = os.path.join(os.path.dirname(ckpt), "spatial_upscaler_x2_v1_1_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f).get("config", {})
        up = LatentUpsampler.from_config(cfg)
    else:
        up = LatentUpsampler(mid_channels=1024)
    weights = load_split_safetensors(ckpt, prefix="spatial_upscaler_x2_v1_1.")
    up.load_weights(list(weights.items()))

    mx.random.seed(SEED)
    latent = mx.random.normal(SHAPE).astype(mx.bfloat16)
    out = up(latent)
    mx.eval(out)

    def dump(name: str, arr: mx.array) -> str:
        path = os.path.join(out_dir, name)
        arr.astype(mx.float32).flatten().astype(mx.float32)
        with open(path, "wb") as f:
            f.write(bytes(memoryview(mx.contiguous(arr.astype(mx.float32)))))
        return path

    lp = dump("latent.raw", latent)
    op = dump("upscaled.raw", out)
    print(f"input  {latent.shape} -> output {out.shape}")
    print(f"export LTX_UP_LATENT={os.path.abspath(lp)}")
    print(f"export LTX_UP_OUT={os.path.abspath(op)}")


if __name__ == "__main__":
    main()
