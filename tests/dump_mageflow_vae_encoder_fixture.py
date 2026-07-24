#!/usr/bin/env python3
"""Dump a MageFlow VAE-encoder parity fixture for src/mage_flow.zig (E7.1).

Runs the pure-MLX REFERENCE (ivanfioravanti/mflux@mage-flow-mlx) MageVAE encoder
in FP32 over a DETERMINISTIC synthetic pixel tensor (no image file — the plan
decouples resize/preprocess from the encoder), and writes one safetensors the
Zig oracle loads (env MAGEFLOW_VAE_ENC_FIXTURE). Tensors:
    pixels   [1, 3, H, W]  f32  — NCHW input in [-1, 1] (fed verbatim to Zig)
    mean     [1, 128, H/16, W/16] f32 — deterministic posterior mean (the port test)
    logvar   [1, 128, H/16, W/16] f32 — clipped log-variance
    latent   [1, 128, H/16, W/16] f32 — mean + exp(0.5*logvar)*N(0,1;seed)
    packed   [1, H/16*W/16, 128]  f32 — pack_latents(latent) (edit ref-latent layout)

The encoder network is fully exercised by `mean` (no RNG), so the Zig test
asserts cosine>0.999 on `mean`; `latent`/`packed` document the sampling layout.

USER-RUN (accepts the MIT Turbo license, has the checkpoint downloaded):
    <mflux>/.venv/bin/python tests/dump_mageflow_vae_encoder_fixture.py \
        ~/.mlx-serve/models/microsoft/Mage-Flow-Turbo <mflux_repo_root> [OUT]

Then:
    MAGEFLOW_VAE_ENC_FIXTURE=<OUT>/mageflow_vae_enc.safetensors \
    zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow VAE encode"
"""

import glob
import os
import sys

import mlx.core as mx
import numpy as np

SEED = 7
H, W = 256, 256  # /16 grid → 16x16 latent


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

    from mflux.models.mage_flow.model.mage_flow_vae.vae import MageVAE
    from mflux.models.mage_flow.weights.mage_flow_weight_mapping import MageFlowWeightMapping

    vae_dir = os.path.join(model_dir, "vae")

    # ── Build MageVAE and load the mapped, conv-transposed VAE weights (fp32) ──
    vae = MageVAE(sample_posterior=True)
    flat = {}
    for shard in sorted(glob.glob(os.path.join(vae_dir, "*.safetensors"))):
        for k, v in mx.load(shard).items():
            mapped = MageFlowWeightMapping.transform_vae_key(k)
            if mapped is None:
                continue
            v = MageFlowWeightMapping.transform_vae_weight(mapped, v)
            flat[mapped] = v.astype(mx.float32)
    vae.update(tree_unflatten(list(flat.items())))
    mx.eval(vae.parameters())
    print(f"[dump] loaded {len(flat)} VAE tensors (fp32)")

    # ── Deterministic pixel tensor in [-1, 1], NCHW ──
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal((1, 3, H, W)).astype(np.float32)
    # smooth it a touch so it resembles image statistics, then clamp to [-1,1]
    ys = np.linspace(-1, 1, H, dtype=np.float32)[None, None, :, None]
    xs = np.linspace(-1, 1, W, dtype=np.float32)[None, None, None, :]
    grad = (ys + xs) * 0.5
    pix = np.clip(0.7 * grad + 0.3 * base, -1.0, 1.0).astype(np.float32)
    pixels = mx.array(pix)

    mean_nchw, logvar_nchw = vae.encode_moments(pixels)
    key = mx.random.key(SEED)
    latent = vae.encode(pixels, key=key)

    # pack_latents: NCHW → transpose(0,2,3,1) → reshape [B, H'W', 128]
    from mflux.models.mage_flow.latent_creator import MageFlowLatentCreator

    packed = MageFlowLatentCreator.pack_latents(latent)
    packed = packed.reshape(1, packed.shape[0] * packed.shape[1], packed.shape[2])
    mx.eval(mean_nchw, logvar_nchw, latent, packed)

    fixture = {
        "pixels": pixels.astype(mx.float32),
        "mean": mean_nchw.astype(mx.float32),
        "logvar": logvar_nchw.astype(mx.float32),
        "latent": latent.astype(mx.float32),
        "packed": packed.astype(mx.float32),
    }
    out_path = os.path.join(out_dir, "mageflow_vae_enc.safetensors")
    mx.save_safetensors(out_path, fixture)
    print(f"[dump] pixels={pixels.shape} mean={mean_nchw.shape} latent={latent.shape} → {out_path}")
    print("\nRun the Zig oracle:")
    print(f"  MAGEFLOW_VAE_ENC_FIXTURE={out_path} \\")
    print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow VAE encode"')


if __name__ == "__main__":
    main()
