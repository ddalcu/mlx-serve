#!/usr/bin/env python3
"""Dump the MageFlow VAE-DECODE parity fixture for src/mage_flow.zig.

Runs the pure-MLX REFERENCE (ivanfioravanti/mflux@mage-flow-mlx) `MageVAE.decode`
in FP32 over a deterministic latent and writes one safetensors the Zig oracle
loads (env MAGEFLOW_VAE_FIXTURE). Tensors:
    z        [1, 128, lh, lw] f32 — the input latent, NCHW (fed verbatim to Zig)
    cond     [1, lh, lw, 384] f32 — the `_Decoder` (y_embedder) output, the
             BISECTION tap: it separates "the decoder is wrong" from "the
             denoiser is wrong" instead of leaving one end-to-end number
    decoded  [1, 3, lh*16, lw*16] f32 — the full decode

This was the one Phase-1 fixture whose dump script never got persisted (it lived
in an ephemeral scratchpad), so `MageFlow VAE decode parity` could not be re-run
by anyone — an oracle nobody can reproduce is a claim, not a test.

USER-RUN (accepts the MIT Turbo license, has the checkpoint downloaded):
    <mflux>/.venv/bin/python tests/dump_mageflow_vae_fixture.py \
        ~/.mlx-serve/models/microsoft/Mage-Flow-Turbo <mflux_repo_root> [OUT]

Then:
    MAGEFLOW_TEST_MODEL=~/.mlx-serve/models/microsoft/Mage-Flow-Turbo \
    MAGEFLOW_VAE_FIXTURE=<OUT>/mageflow_vae.safetensors \
    zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow VAE decode"
"""

import glob
import os
import sys

import mlx.core as mx
import numpy as np

SEED = 3
LH, LW = 4, 4  # → a 64x64 decode; big enough to exercise every block, fast to dump


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
    vae = MageVAE(sample_posterior=True)
    flat = {}
    for shard in sorted(glob.glob(os.path.join(vae_dir, "*.safetensors"))):
        for k, v in mx.load(shard).items():
            mapped = MageFlowWeightMapping.transform_vae_key(k)
            if mapped is None:
                continue
            flat[mapped] = MageFlowWeightMapping.transform_vae_weight(mapped, v).astype(mx.float32)
    vae.update(tree_unflatten(list(flat.items())))
    mx.eval(vae.parameters())
    print(f"[dump] loaded {len(flat)} VAE tensors (fp32)")

    rng = np.random.default_rng(SEED)
    z = mx.array(rng.standard_normal((1, 128, LH, LW)).astype(np.float32))

    # Mirror `MageVAE.decode`'s internals so the bisection tap is the SAME
    # tensor the Zig side calls `decoderCond`.
    latent = mx.transpose(z, (0, 2, 3, 1))
    cond = vae.decoder_model.y_embedder.decoder(latent)
    decoded = vae.decode(z)
    mx.eval(cond, decoded)

    fixture = {
        "z": z.astype(mx.float32),
        "cond": cond.astype(mx.float32),
        "decoded": decoded.astype(mx.float32),
    }
    out_path = os.path.join(out_dir, "mageflow_vae.safetensors")
    mx.save_safetensors(out_path, fixture)
    print(f"[dump] z={z.shape} cond={cond.shape} decoded={decoded.shape} → {out_path}")
    print("\nRun the Zig oracle:")
    print(f"  MAGEFLOW_TEST_MODEL={model_dir} MAGEFLOW_VAE_FIXTURE={out_path} \\")
    print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow VAE decode"')


if __name__ == "__main__":
    main()
