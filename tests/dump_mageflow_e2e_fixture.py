#!/usr/bin/env python3
"""Dump a MageFlow end-to-end DiT-loop parity fixture for src/mage_flow.zig.

Runs the pure-MLX REFERENCE (ivanfioravanti/mflux@mage-flow-mlx) transformer in
its native bf16 over the reference scheduler + Euler loop at a small resolution,
and writes a safetensors the Zig oracle replays step-by-step (env
MAGEFLOW_E2E_BF16_FIXTURE). This pins the DiT loop glue (scheduler sigmas, Euler,
pack order, bf16 timestep rounding) against the reference. Tensors:
    noise   [1,256,128]  packed initial latents (plain N(0,1), seed 42)
    txt     [1,seq,2560] conditioning (reference bf16 encode, stored f32)
    mask    [1,seq]      text attention mask
    sigmas  [5]          static-shift FlowMatchEuler sigmas
    v0      [1,256,128]  step-0 velocity
    lat1..4 [1,256,128]  per-step latents
    final   [1,256,128]  final latents

USER-RUN (accepts the MIT Turbo license, has the checkpoint downloaded):
    <mflux>/.venv/bin/python tests/dump_mageflow_e2e_fixture.py \
        ~/.mlx-serve/models/microsoft/Mage-Flow-Turbo <mflux_repo_root> [OUT]

Then:
    MAGEFLOW_TEST_MODEL=~/.mlx-serve/models/microsoft/Mage-Flow-Turbo \
    MAGEFLOW_E2E_BF16_FIXTURE=<OUT>/mageflow_e2e_bf16.safetensors \
    zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow end-to-end"
"""

import os
import sys

import mlx.core as mx

PROMPT = "a red fox sitting in the snow, photorealistic, golden hour lighting"
H = W = 256
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
    from mflux.models.mage_flow.latent_creator import MageFlowLatentCreator
    from mflux.models.mage_flow.variants.pipeline_helpers import make_velocity_predictor
    from mflux.models.mage_flow.variants.txt2img.mage_flow import MageFlow

    m = MageFlow(model_config=ModelConfig.mage_flow_turbo(), model_path=model_dir)  # bf16 native
    txt, mask = m._encode_prompt_pair(prompt=PROMPT, negative_prompt=None, guidance=1.0)
    noise = MageFlowLatentCreator.create_noise(
        seed=SEED, height=H, width=W, gaussian_shading=False, dtype=mx.bfloat16
    )
    cfg = Config(model_config=m.model_config, num_inference_steps=N, height=H, width=W, guidance=1.0, scheduler="mage_flow")
    sig = cfg.scheduler.sigmas
    shapes = [(1, H // 16, W // 16)]
    predict = make_velocity_predictor(
        transformer=m.transformer, text_embeddings=txt, text_attention_mask=mask,
        image_shapes=shapes, guidance=1.0, compile_model=False,
    )

    fixture = {"noise": noise, "txt": txt, "mask": mask.astype(mx.int32), "sigmas": sig.astype(mx.float32)}
    lat = noise
    for step in range(N):
        velocity = predict(lat, sig[step])
        if step == 0:
            fixture["v0"] = velocity
        lat = cfg.scheduler.step(noise=velocity, timestep=step, latents=lat, sigmas=sig)
        fixture["lat%d" % (step + 1)] = lat
        mx.eval(lat)
    fixture["final"] = lat

    out_path = os.path.join(out_dir, "mageflow_e2e_bf16.safetensors")
    mx.save_safetensors(out_path, {k: v.astype(mx.float32) for k, v in fixture.items()})
    print(f"[dump] noise={noise.shape} txt={txt.shape} final={lat.shape} → {out_path}")
    print("\nRun the Zig oracle:")
    print(f"  MAGEFLOW_TEST_MODEL={model_dir} \\")
    print(f"  MAGEFLOW_E2E_BF16_FIXTURE={out_path} \\")
    print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow end-to-end"')


if __name__ == "__main__":
    main()
