#!/usr/bin/env python3
"""Dump reference outputs for SDXL's UNet and VAE decoder.

The oracle for `src/sdxl_unet.zig` and `src/sdxl_vae.zig`. Runs diffusers' own
`UNet2DConditionModel` / `AutoencoderKL` over the checkpoint's real weights and
saves both the INPUTS and the outputs, so the Zig side runs the same forward on
the same numbers rather than a forward that merely looks similar.

    python3 tests/dump_sdxl_unet_fixtures.py \
        --model ~/.mlx-serve/staging/sdxl-base-1.0 \
        --out   ~/.mlx-serve/staging/sdxl_unet_fixture.safetensors

Runs on CPU in float32 DELIBERATELY, for the reason recorded in the repo's
gotchas: MPS fp16 quietly decorrelates deep towers, and a reference computed
less precisely than the thing it checks is not a reference. The checkpoint's
fp16 weights are upcast on load (`torch_dtype=torch.float32`).

The latent is 16x16 (a 128px image). That is small enough that a CPU fp32
forward is quick, and still exercises every structural path: all three down
blocks including the attention-free first, both downsamplers, the mid block's
10 transformer layers, all three up blocks with their skip concatenations, and
both upsamplers.

INTERMEDIATE captures are dumped alongside the final output on purpose. A UNet
that disagrees only at the end is a needle in 1680 tensors; with the time
embedding, the post-conv_in activation, every down-block skip, and the mid
output pinned, a mismatch names its own block.
"""
import argparse
import os
import sys

import torch
from safetensors.torch import save_file

# Latent geometry. Shared with the Zig test — a fixture generated at a
# different size than the implementation under test is not an oracle.
LATENT_H = 16
LATENT_W = 16
SEQ = 77
SEED = 20260817

# A mid-schedule timestep. Deliberately NOT 0 or 999: the sinusoidal embedding
# is degenerate at the endpoints, so an off-by-one in flip_sin_to_cos or the
# freq_shift would pass at a boundary and fail everywhere else.
TIMESTEP = 501


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="SDXL checkpoint dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_dir = os.path.expanduser(args.model)
    out = os.path.expanduser(args.out)

    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel
    except ImportError:
        sys.exit("diffusers is required to generate this fixture")

    g = torch.Generator().manual_seed(SEED)
    sample = torch.randn(1, 4, LATENT_H, LATENT_W, generator=g, dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, SEQ, 2048, generator=g, dtype=torch.float32)
    text_embeds = torch.randn(1, 1280, generator=g, dtype=torch.float32)
    # The real micro-conditioning vector for a 1024x1024 render with no crop.
    time_ids = torch.tensor([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]])
    timestep = torch.tensor(TIMESTEP, dtype=torch.float32)

    tensors = {
        "in.sample": sample,
        "in.encoder_hidden_states": encoder_hidden_states,
        "in.text_embeds": text_embeds,
        "in.time_ids": time_ids,
        "in.timestep": timestep.reshape(1),
    }

    print(f"loading UNet from {model_dir}/unet ...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        model_dir, subfolder="unet", torch_dtype=torch.float32, variant="fp16"
    )
    unet.eval()

    # ── Intermediate captures, so a mismatch names its own block. Hooks read
    # module OUTPUTS; the down-block hooks capture the skip tuple each block
    # hands the up path, which is where a resnet/attention ordering bug shows.
    caps = {}

    def cap(name):
        def hook(_m, _i, o):
            # `.clone()` is load-bearing: a down block's LAST res sample is the
            # same tensor object as its hidden output, and safetensors refuses
            # to save aliased storage.
            if isinstance(o, tuple):
                # CrossAttnDownBlock2D returns (hidden, res_samples tuple)
                caps[f"mid.{name}.hidden"] = o[0].detach().float().clone()
                for j, r in enumerate(o[1]):
                    caps[f"mid.{name}.res{j}"] = r.detach().float().clone()
            else:
                caps[f"mid.{name}"] = o.detach().float().clone()
        return hook

    unet.time_embedding.register_forward_hook(cap("time_emb"))
    unet.add_embedding.register_forward_hook(cap("add_emb"))
    unet.conv_in.register_forward_hook(cap("conv_in"))
    for i, blk in enumerate(unet.down_blocks):
        blk.register_forward_hook(cap(f"down{i}"))
    unet.mid_block.register_forward_hook(cap("mid_block"))
    for i, blk in enumerate(unet.up_blocks):
        blk.register_forward_hook(cap(f"up{i}"))

    added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
    with torch.no_grad():
        noise_pred = unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

    tensors["out.noise_pred"] = noise_pred.detach().float()
    tensors.update(caps)
    print(f"  unet out {tuple(noise_pred.shape)}", flush=True)
    del unet

    # ── VAE decoder. Its input is a SCALED latent: the pipeline divides by
    # scaling_factor before decoding, so the fixture stores the post-division
    # tensor the Zig decoder is handed, not the raw latent.
    print(f"loading VAE from {model_dir}/vae ...", flush=True)
    vae = AutoencoderKL.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=torch.float32, variant="fp16"
    )
    vae.eval()
    scaling = vae.config.scaling_factor
    print(f"  scaling_factor {scaling}", flush=True)

    g2 = torch.Generator().manual_seed(SEED + 1)
    raw_latent = torch.randn(1, 4, LATENT_H, LATENT_W, generator=g2, dtype=torch.float32)
    vae_in = raw_latent / scaling
    with torch.no_grad():
        decoded = vae.decode(vae_in).sample

    tensors["in.vae_latent"] = vae_in
    tensors["out.vae_decoded"] = decoded.detach().float()
    print(f"  vae out {tuple(decoded.shape)}", flush=True)

    tensors = {k: v.contiguous() for k, v in tensors.items()}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_file(tensors, out)
    print(f"wrote {out} ({len(tensors)} tensors)")
    for k in sorted(tensors):
        print(f"  {k:36s} {tuple(tensors[k].shape)}")


if __name__ == "__main__":
    main()
