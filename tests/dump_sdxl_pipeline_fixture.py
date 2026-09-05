#!/usr/bin/env python3
"""Dump a full SDXL pipeline run as an end-to-end oracle.

The component fixtures pin each piece against diffusers separately. They cannot
see a COMPOSITION error — a swapped tower concat order, a dropped
`scale_model_input`, guidance applied to the wrong branch, the Euler step taken
on the scaled latent instead of the raw one. Every one of those produces a
plausible image from correct parts.

This closes that gap by removing the only thing that legitimately differs
between the two implementations: the random latent. It is generated here, saved,
and INJECTED into the Zig pipeline, so both sides denoise the same starting
point and the final images must agree.

    python3 tests/dump_sdxl_pipeline_fixture.py \
        --model ~/.mlx-serve/staging/sdxl-base-1.0 \
        --out   ~/.mlx-serve/staging/sdxl_pipeline_fixture.safetensors

Small and short on purpose (512x512, 4 steps): this is a numerical agreement
check, not an image-quality one, and a CPU fp32 SDXL run is not cheap.
"""
import argparse
import os
import sys

import torch
from safetensors.torch import save_file

PROMPT = "a photo of a cat"
# None, not "": diffusers zeroes the unconditional branch only when the
# negative prompt is ABSENT (`negative_prompt is None and
# force_zeros_for_empty_prompt`). Passing "" instead encodes the empty string,
# which is a different tensor and a different branch from the one our default
# `GenOpts.negative_prompt = null` takes.
NEGATIVE = None
WIDTH = 512
HEIGHT = 512
STEPS = 4
GUIDANCE = 5.0
SEED = 20260817


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_dir = os.path.expanduser(args.model)
    out = os.path.expanduser(args.out)

    try:
        from diffusers import StableDiffusionXLPipeline
    except ImportError:
        sys.exit("diffusers is required to generate this fixture")

    print(f"loading pipeline from {model_dir} ...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_dir, torch_dtype=torch.float32, variant="fp16", add_watermarker=False
    )
    pipe.set_progress_bar_config(disable=True)

    # The shared starting point. Unit noise: the pipeline scales it by
    # `init_noise_sigma` itself, and so does ours, so the fixture stores the
    # UNSCALED tensor to keep that step on the implementation's side of the line.
    g = torch.Generator().manual_seed(SEED)
    latents = torch.randn(
        1, 4, HEIGHT // 8, WIDTH // 8, generator=g, dtype=torch.float32
    )

    print(f"denoising {STEPS} steps at {WIDTH}x{HEIGHT} ...", flush=True)
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        latents=latents,
        output_type="pt",
    ).images

    # `output_type="pt"` is [0,1] NCHW, which is the convention our
    # `Engine.generate` returns.
    tensors = {
        "in.latents": latents.contiguous(),
        "out.image": image.detach().float().contiguous(),
        "cfg": torch.tensor(
            [float(WIDTH), float(HEIGHT), float(STEPS), float(GUIDANCE)]
        ),
    }
    save_file(tensors, out)
    print(f"wrote {out}")
    for k, v in tensors.items():
        print(f"  {k:16s} {tuple(v.shape)}")
    print(f"  prompt: {PROMPT!r}  negative: {NEGATIVE!r}")
    print(f"  image range [{float(image.min()):.4f}, {float(image.max()):.4f}]")


if __name__ == "__main__":
    main()
