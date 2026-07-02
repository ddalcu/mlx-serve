#!/usr/bin/env python3
"""Dump LTX-2 VAE ENCODER parity fixtures (.raw) for the Zig oracle test in src/ltx_video.zig.

Runs the ltx_core_mlx REFERENCE ``VideoEncoder.encode`` on a fixed single-image
(F=1) pixel tensor and writes byte-exact golden tensors that `zig build test`
compares the Zig ``vaeEncode`` port against (cosine parity > 0.998).

The encoder is the HARD DEPENDENCY for native I2V (first-frame conditioning):
the q4 bundle ships only ``vae_decoder.safetensors``; ``vae_encoder.safetensors``
(~0.6 GB, listed in the repo's split_model.json) must be downloaded separately:

    hf download dgrauet/ltx-2.3-mlx-q4 vae_encoder.safetensors \
        --local-dir ~/.mlx-serve/models/dgrauet/ltx-2.3-mlx-q4

USER-RUN: you accept LTX-2's license, download the model, and install the
reference package (`ltx_core_mlx`, MLX). NOT run in CI; the weights-gated
structural test in src/ltx_video.zig covers the port without it.

Usage:
    python3 tests/dump_ltx_vae_encoder_fixtures.py <VAE_ENCODER.safetensors> [OUT_DIR] [H] [W]

Writes (flat little-endian f32):
    pixels.raw   [1, 3, 1, H, W]   the input image tensor in [-1, 1] (BCFHW), the
                                   EXACT input src/ltx_video.zig vaeEncode expects
    latent.raw   [1, 128, 1, H/32, W/32]  reference VideoEncoder.encode(pixels)

Then it prints the `export LTX_ENC_*` block to paste before:
    LTX_TEST_MODEL=<model_dir> LTX_ENC_PIXELS=…/pixels.raw LTX_ENC_LATENT=…/latent.raw \
      zig build test -Doptimize=ReleaseFast -Dtest-filter="VAE encoder"
"""

import os
import sys

import mlx.core as mx


# Fixed, reproducible fixture parameters. A single image (F=1) — the only shape
# the I2V conditioning path ever encodes. H/W must be multiples of 32.
SEED = 0
DEFAULT_H = 256
DEFAULT_W = 256


def _load_encoder(ckpt: str):
    from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder

    enc = VideoEncoder()
    raw = mx.load(ckpt)
    # Strip the "vae_encoder." prefix and remap the underscore-prefixed stats keys
    # (mirror ltx_pipelines_mlx.utils.blocks.ImageConditioner.load).
    weights = {}
    for k, v in raw.items():
        nk = k
        if nk.startswith("vae_encoder."):
            nk = nk[len("vae_encoder.") :]
        nk = nk.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means")
        weights[nk] = v
    enc.load_weights(list(weights.items()))
    return enc


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ckpt = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./ltx_enc_fixtures"
    H = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_H
    W = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_W
    assert H % 32 == 0 and W % 32 == 0, "H and W must be multiples of 32"
    os.makedirs(out_dir, exist_ok=True)

    enc = _load_encoder(ckpt)

    # A fixed, deterministic image in [-1, 1]. A smooth spatial gradient (not pure
    # noise) so the encoder's local 3x3 convs see real structure.
    mx.random.seed(SEED)
    yy = mx.arange(H).reshape(1, 1, 1, H, 1) / H
    xx = mx.arange(W).reshape(1, 1, 1, 1, W) / W
    base = (yy + xx) / 2.0  # [1,1,1,H,W] in [0,1]
    base = mx.broadcast_to(base, (1, 3, 1, H, W))
    jitter = mx.random.normal((1, 3, 1, H, W)) * 0.05
    pixels = mx.clip(base + jitter, 0.0, 1.0) * 2.0 - 1.0  # → [-1, 1]
    pixels = pixels.astype(mx.bfloat16)

    latent = enc.encode(pixels)  # (1, 128, 1, H/32, W/32)
    mx.eval(latent)

    def save(name, t):
        arr = mx.array(t).astype(mx.float32)
        mx.eval(arr)
        import numpy as np

        np.array(arr, dtype="<f4").ravel().tofile(os.path.join(out_dir, name))
        return os.path.join(out_dir, name), t.shape

    px_p, px_s = save("pixels.raw", pixels)
    lat_p, lat_s = save("latent.raw", latent)

    print(f"pixels {tuple(px_s)} -> {px_p}")
    print(f"latent {tuple(lat_s)} -> {lat_p}")
    print()
    print("Paste this to run the Zig parity oracle:")
    model_dir = os.path.dirname(os.path.abspath(ckpt))
    print(
        f'  LTX_TEST_MODEL="{model_dir}" \\\n'
        f'  LTX_ENC_PIXELS="{px_p}" LTX_ENC_LATENT="{lat_p}" LTX_ENC_H={H} LTX_ENC_W={W} \\\n'
        f'  zig build test -Doptimize=ReleaseFast -Dtest-filter="VAE encoder"'
    )


if __name__ == "__main__":
    main()
