#!/usr/bin/env python3
"""Dump the two MageFlow DiT forward-parity fixtures for src/mage_flow.zig.

Runs the pure-MLX REFERENCE (ivanfioravanti/mflux@mage-flow-mlx) transformer in
FP32 over deterministic inputs and writes BOTH single-forward fixtures in one
pass (they share all the setup — loading the 8 GB transformer twice to dump two
tensors would be the only difference):

  mageflow_dit.safetensors         (env MAGEFLOW_DIT_FIXTURE)
      img [1, 64, 128]   txt [1, 16, 2560]   out [1, 64, 128]   — lh=lw=8, t=0.7, no mask
  mageflow_dit_masked.safetensors  (env MAGEFLOW_DIT_MASKED_FIXTURE)
      img [1, 256, 128]  txt [1, 24, 2560]  mask [1, 24]  out [1, 256, 128]
      — lh=lw=16, t=0.3, the last 6 text positions PADDED

The masked case is the one that matters: it pins the additive attention mask and
a larger image grid, and a mask bug is invisible in the unmasked fixture. Both
dump scripts for these lived in an ephemeral scratchpad during Phases 1-3 and
were never persisted, so neither oracle could be re-run by anyone.

USER-RUN (accepts the MIT Turbo license, has the checkpoint downloaded):
    <mflux>/.venv/bin/python tests/dump_mageflow_dit_fixture.py \
        ~/.mlx-serve/models/microsoft/Mage-Flow-Turbo <mflux_repo_root> [OUT]

Then:
    MAGEFLOW_TEST_MODEL=~/.mlx-serve/models/microsoft/Mage-Flow-Turbo \
    MAGEFLOW_DIT_FIXTURE=<OUT>/mageflow_dit.safetensors \
    MAGEFLOW_DIT_MASKED_FIXTURE=<OUT>/mageflow_dit_masked.safetensors \
    zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow DiT"
"""

import os
import sys

import mlx.core as mx
import numpy as np

SEED = 5


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    model_dir = os.path.abspath(sys.argv[1])
    ref_root = os.path.abspath(sys.argv[2])
    out_dir = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.abspath("mageflow_fixtures")
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, os.path.join(ref_root, "src"))

    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.mage_flow.variants.txt2img.mage_flow import MageFlow

    m = MageFlow(model_config=ModelConfig.mage_flow_turbo(), model_path=model_dir)
    dit = m.transformer
    # The component oracles compare in fp32; the checkpoint loads bf16 native.
    from mlx.utils import tree_map

    dit.update(tree_map(lambda p: p.astype(mx.float32), dit.parameters()))
    mx.eval(dit.parameters())
    print("[dump] transformer loaded and cast to fp32")

    rng = np.random.default_rng(SEED)

    def forward(lh, lw, ltxt, t, pad):
        img = mx.array(rng.standard_normal((1, lh * lw, 128)).astype(np.float32))
        txt = mx.array((0.05 * rng.standard_normal((1, ltxt, 2560))).astype(np.float32))
        mask = None
        if pad:
            keep = np.ones((1, ltxt), dtype=np.int32)
            keep[0, ltxt - pad :] = 0
            mask = mx.array(keep)
        out = dit(img, txt, t, (1, lh, lw), text_attention_mask=mask)
        mx.eval(img, txt, out)
        return img, txt, mask, out

    img, txt, _, out = forward(8, 8, 16, 0.7, 0)
    p = os.path.join(out_dir, "mageflow_dit.safetensors")
    mx.save_safetensors(p, {"img": img, "txt": txt, "out": out.astype(mx.float32)})
    print(f"[dump] unmasked img={img.shape} txt={txt.shape} out={out.shape} → {p}")

    img, txt, mask, out = forward(16, 16, 24, 0.3, 6)
    pm = os.path.join(out_dir, "mageflow_dit_masked.safetensors")
    mx.save_safetensors(pm, {"img": img, "txt": txt, "mask": mask.astype(mx.int32), "out": out.astype(mx.float32)})
    print(f"[dump] masked   img={img.shape} txt={txt.shape} mask={mask.shape} out={out.shape} → {pm}")

    print("\nRun the Zig oracles:")
    print(f"  MAGEFLOW_TEST_MODEL={model_dir} \\")
    print(f"  MAGEFLOW_DIT_FIXTURE={p} MAGEFLOW_DIT_MASKED_FIXTURE={pm} \\")
    print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow DiT"')


if __name__ == "__main__":
    main()
