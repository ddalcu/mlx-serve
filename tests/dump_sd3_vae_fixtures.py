#!/usr/bin/env python3
"""Dump reference inputs/outputs for SD 3.5's 16-channel VAE.

The oracle for `src/sd3_vae.zig`. Runs diffusers' own `AutoencoderKL` — the
real class, not a transcription — and saves the INPUTS beside the outputs, so
the Zig side runs the same forward on the same numbers.

Two modes:

    # BUILD: a tiny random-weight AutoencoderKL at SD 3.5's config SHAPE.
    # Writes the model AND the fixture, so the Zig test binds exactly the
    # weights the reference ran. ~4 MB, no 20 GB download.
    python3 tests/dump_sd3_vae_fixtures.py build --out ~/.mlx-serve/staging/sd3-vae-tiny

    # MODEL: the same fixture against a real checkpoint's `vae/`.
    python3 tests/dump_sd3_vae_fixtures.py model \
        --model ~/.mlx-serve/staging/sd3.5-large \
        --out   ~/.mlx-serve/staging/sd3-vae-real

Both modes write `<out>/fixture.safetensors`; `build` also writes `<out>/vae/`
so the directory is a complete, self-describing checkpoint. The Zig test reads
`SD3_VAE_FIXTURE_DIR` (fixture + model) and optionally `SD3_VAE_MODEL_DIR` (a
real checkpoint whose `vae/` holds the weights instead).

Everything runs on CPU in float32 DELIBERATELY: a reference computed less
precisely than the thing it checks is not a reference, and this repo has
already been bitten by MPS fp16 decorrelating deep towers.

WHY A TINY MODEL PINS THE ARCHITECTURE. The Zig side reads its stage and
resnet counts out of the weight names, so a 4-stage / 2-resnet tiny model
exercises every structural path the real one does: all four down and up
stages, all three downsamplers and upsamplers, both mid-block resnets and the
mid-block attention. What it deliberately does NOT pin is width — the real
`block_out_channels` is [128, 256, 512, 512] and this is [32, 64, 64, 64] —
so `model` mode exists for the real numbers.

INTERMEDIATES are dumped alongside the final outputs on purpose. A decoder
that disagrees only at the end is a needle in 244 tensors; with conv_in, the
mid block and every up block pinned, a mismatch names its own block.
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

# SD 3.5's VAE, read out of `vae/config.json` on
# `adamo1139/stable-diffusion-3.5-large-ungated` (an exact ungated mirror of
# the gated stability repo) rather than from documentation. The three fields
# that part company with SDXL's VAE are `latent_channels`, the two
# `use_*_quant_conv` flags, and `shift_factor`.
LATENT_CHANNELS = 16
SCALING_FACTOR = 1.5305
SHIFT_FACTOR = 0.0609

# The tiny model's widths. Four stages and two resnets per stage — the real
# topology at a width a CPU fp32 forward finishes instantly at. 32 is the
# floor: `norm_num_groups` is 32 and a group norm needs at least one channel
# per group.
TINY_BLOCK_OUT = (32, 64, 64, 64)
LAYERS_PER_BLOCK = 2
NORM_NUM_GROUPS = 32

# A 4x4 latent is a 32px image at the family's 8x stride. Small, and still
# leaves the mid block a 16-token attention rather than a degenerate 1-token
# one.
LATENT_H = 4
LATENT_W = 4
SEED = 20260904


def build_tiny(out_dir):
    """A random-weight `AutoencoderKL` at SD 3.5's config shape, saved to disk."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * len(TINY_BLOCK_OUT),
        up_block_types=("UpDecoderBlock2D",) * len(TINY_BLOCK_OUT),
        block_out_channels=TINY_BLOCK_OUT,
        layers_per_block=LAYERS_PER_BLOCK,
        act_fn="silu",
        latent_channels=LATENT_CHANNELS,
        norm_num_groups=NORM_NUM_GROUPS,
        sample_size=LATENT_H * 8,
        scaling_factor=SCALING_FACTOR,
        shift_factor=SHIFT_FACTOR,
        force_upcast=True,
        mid_block_add_attention=True,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )

    # Deterministic, non-degenerate init. Default init leaves every norm weight
    # at exactly 1.0 and every bias at 0.0 — a fixture that cannot tell a bound
    # norm weight from an unbound one. Perturbing them is what makes the
    # binding of all 244 tensors observable in the output.
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if p.dim() == 1:
                noise = torch.randn(p.shape, generator=g, dtype=torch.float32)
                p.copy_(1.0 + 0.1 * noise if name.endswith("weight") else 0.1 * noise)
            else:
                fan_in = p[0].numel()
                noise = torch.randn(p.shape, generator=g, dtype=torch.float32)
                p.copy_(noise / (fan_in ** 0.5))

    vae = vae.to(torch.float32).eval()
    vae.save_pretrained(os.path.join(out_dir, "vae"))
    cfg_path = os.path.join(out_dir, "vae", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    # The presence of the quant convs is a CONFIG decision on both sides. If
    # diffusers ever stops writing these fields the Zig loader falls back to
    # diffusers' own default (True) and fails LOUDLY on the missing tensor,
    # which is the failure we want — but the fixture must not be the thing
    # that silently drops them.
    for k in ("use_quant_conv", "use_post_quant_conv"):
        if cfg.get(k) is not False:
            sys.exit(f"tiny config did not record {k}: false — got {cfg.get(k)!r}")
    return vae


def load_real(model_dir):
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=torch.float32
    ).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "model"))
    ap.add_argument("--model", help="checkpoint dir (mode=model); reads <dir>/vae")
    ap.add_argument("--out", required=True, help="fixture directory to write")
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    try:
        import diffusers  # noqa: F401
    except ImportError:
        sys.exit("diffusers is required to generate this fixture")

    if args.mode == "build":
        vae = build_tiny(out_dir)
        latent_h, latent_w = LATENT_H, LATENT_W
    else:
        if not args.model:
            sys.exit("mode=model needs --model")
        vae = load_real(os.path.expanduser(args.model))
        # 16x16 latent = a 128px image: big enough to be a real forward on the
        # real widths, small enough for a CPU fp32 pass.
        latent_h, latent_w = 16, 16

    cfg = vae.config
    print(
        f"vae: latent_channels={cfg.latent_channels} "
        f"blocks={tuple(cfg.block_out_channels)} "
        f"quant_conv={getattr(cfg, 'use_quant_conv', True)} "
        f"post_quant_conv={getattr(cfg, 'use_post_quant_conv', True)} "
        f"scaling={cfg.scaling_factor} shift={getattr(cfg, 'shift_factor', None)}",
        flush=True,
    )
    if cfg.latent_channels != LATENT_CHANNELS:
        sys.exit(f"expected {LATENT_CHANNELS} latent channels, got {cfg.latent_channels}")

    g = torch.Generator().manual_seed(SEED + 1)
    tensors = {}

    # ── Decode. `vae.decode` is handed an ALREADY-unscaled latent: SD 3.5's
    # `z / scaling + shift` is the PIPELINE's arithmetic (`sd3.decodeScale`),
    # not the VAE's. Both tensors are stored so the pipeline lane can pin its
    # own half against the same numbers.
    raw = torch.randn(1, LATENT_CHANNELS, latent_h, latent_w, generator=g, dtype=torch.float32)
    scaling = float(cfg.scaling_factor)
    shift = float(getattr(cfg, "shift_factor", 0.0) or 0.0)
    vae_in = raw / scaling + shift

    caps = {}

    def cap(name):
        def hook(_m, _i, o):
            caps[name] = o.detach().float().clone()
        return hook

    handles = [
        vae.decoder.conv_in.register_forward_hook(cap("mid.dec.conv_in")),
        vae.decoder.mid_block.register_forward_hook(cap("mid.dec.mid_block")),
        vae.encoder.conv_in.register_forward_hook(cap("mid.enc.conv_in")),
        vae.encoder.mid_block.register_forward_hook(cap("mid.enc.mid_block")),
    ]
    for i, blk in enumerate(vae.decoder.up_blocks):
        handles.append(blk.register_forward_hook(cap(f"mid.dec.up{i}")))
    for i, blk in enumerate(vae.encoder.down_blocks):
        handles.append(blk.register_forward_hook(cap(f"mid.enc.down{i}")))

    with torch.no_grad():
        decoded = vae.decode(vae_in).sample

    tensors["in.latent_raw"] = raw
    tensors["in.vae_in"] = vae_in
    tensors["out.decoded"] = decoded.detach().float()
    print(f"  decode {tuple(vae_in.shape)} -> {tuple(decoded.shape)}", flush=True)

    # ── Encode. The pipeline hands the encoder pixels already in [-1, 1] (the
    # model's own range) and takes the distribution MEAN, never a sample — an
    # img2img source must be reproducible. The `(z - shift) * scaling` that
    # turns this into a latent is again the pipeline's.
    image = torch.rand(1, 3, latent_h * 8, latent_w * 8, generator=g, dtype=torch.float32) * 2.0 - 1.0
    with torch.no_grad():
        posterior = vae.encode(image).latent_dist
    mean = posterior.mean.detach().float()

    tensors["in.image"] = image
    tensors["out.encoded_mean"] = mean
    tensors["out.encoded_latent"] = (mean - shift) * scaling
    print(f"  encode {tuple(image.shape)} -> {tuple(mean.shape)}", flush=True)

    for h in handles:
        h.remove()
    tensors.update(caps)

    # The scale/shift pair the CALLER applies, carried so a fixture is
    # self-describing rather than depending on a constant typed twice.
    tensors["cfg.scaling_factor"] = torch.tensor([scaling], dtype=torch.float32)
    tensors["cfg.shift_factor"] = torch.tensor([shift], dtype=torch.float32)

    out = os.path.join(out_dir, "fixture.safetensors")
    save_file({k: v.contiguous() for k, v in tensors.items()}, out)
    print(f"wrote {out} ({len(tensors)} tensors)")
    for k in sorted(tensors):
        print(f"  {k:28s} {tuple(tensors[k].shape)}")


if __name__ == "__main__":
    main()
