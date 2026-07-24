#!/usr/bin/env python3
"""Dump a MageFlow EDIT-loop parity fixture for src/mage_flow.zig (E7.5).

Runs the pure-MLX REFERENCE (ivanfioravanti/mflux@mage-flow-mlx) edit denoise
loop — concat([target, refs]) → transformer with multi-image RoPE → slice to the
target tokens → Euler — over the LOCAL txt2img transformer at a tiny resolution.
This validates the edit-loop ASSEMBLY (multi-image RoPE, concat, target-slice,
Euler) independent of the Edit checkpoint's weights. Env MAGEFLOW_EDIT_E2E_FIXTURE.
Tensors (native bf16, stored f32):
    noise   [1, HW, 128]        packed target noise
    refs    [1, nrefs*HW, 128]  constant reference latents
    txt     [1, Ltxt, 2560]     conditioning
    sigmas  [N+1]               static-shift FlowMatchEuler sigmas
    lat1..N [1, HW, 128]        per-step target latents
    final   [1, HW, 128]        final target latents

USER-RUN (accepts the MIT Turbo license, has the Turbo checkpoint downloaded):
    <mflux>/.venv/bin/python tests/dump_mageflow_edit_e2e_fixture.py \
        ~/.mlx-serve/models/microsoft/Mage-Flow-Turbo <mflux_repo_root> [OUT]

Then:
    MAGEFLOW_TEST_MODEL=~/.mlx-serve/models/microsoft/Mage-Flow-Turbo \
    MAGEFLOW_EDIT_E2E_FIXTURE=<OUT>/mageflow_edit_e2e.safetensors \
    zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow edit loop"
"""

import os
import sys

import mlx.core as mx

LH = LW = 8          # HW = 64 latent tokens
NREFS = 1
LTXT = 16
N = 4
SEED = 42


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    model_dir = os.path.abspath(sys.argv[1])
    ref_root = os.path.abspath(sys.argv[2])
    out_dir = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.abspath("mageflow_fixtures")
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, os.path.join(ref_root, "src"))

    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.mage_flow.variants.pipeline_helpers import make_velocity_predictor
    from mflux.models.mage_flow.variants.txt2img.mage_flow import MageFlow

    m = MageFlow(model_config=ModelConfig.mage_flow_turbo(), model_path=model_dir)  # bf16 native

    HW = LH * LW
    key = mx.random.key(SEED)
    k1, k2, k3 = mx.random.split(key, 3)
    noise = mx.random.normal((1, HW, 128), key=k1).astype(mx.bfloat16)
    refs = mx.random.normal((1, NREFS * HW, 128), key=k2).astype(mx.bfloat16)
    txt = mx.random.normal((1, LTXT, 2560), key=k3).astype(mx.bfloat16)
    mask = mx.ones((1, LTXT), dtype=mx.int32)

    cfg = Config(model_config=m.model_config, num_inference_steps=N, height=LH * 16, width=LW * 16, guidance=1.0, scheduler="mage_flow")
    sig = cfg.scheduler.sigmas
    shapes = [(1, LH, LW)] * (1 + NREFS)
    predict = make_velocity_predictor(
        transformer=m.transformer, text_embeddings=txt, text_attention_mask=mask,
        image_shapes=shapes, guidance=1.0, target_length=HW, compile_model=False,
    )

    fixture = {"noise": noise, "refs": refs, "txt": txt, "sigmas": sig.astype(mx.float32)}
    target = noise
    for step in range(N):
        model_input = mx.concatenate([target, refs], axis=1)
        velocity = predict(model_input, sig[step])
        target = cfg.scheduler.step(noise=velocity, timestep=step, latents=target, sigmas=sig)
        fixture["lat%d" % (step + 1)] = target
        mx.eval(target)
    fixture["final"] = target

    out_path = os.path.join(out_dir, "mageflow_edit_e2e.safetensors")
    mx.save_safetensors(out_path, {k: v.astype(mx.float32) for k, v in fixture.items()})
    print(f"[dump] noise={noise.shape} refs={refs.shape} final={target.shape} → {out_path}")
    print("\nRun the Zig oracle:")
    print(f"  MAGEFLOW_EDIT_E2E_FIXTURE={out_path} \\")
    print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow edit loop"')


if __name__ == "__main__":
    main()
