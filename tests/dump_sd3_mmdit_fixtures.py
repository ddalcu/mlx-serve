#!/usr/bin/env python3
"""Dump reference inputs, INTERMEDIATES and outputs for SD 3.5's MMDiT.

The oracle for `src/sd3_mmdit.zig`. Runs diffusers' own `SD3Transformer2DModel`
— the reference class, not a transcription of it — and saves everything the Zig
side needs to run the SAME forward over the SAME numbers.

Two modes, and the first is the one CI can afford:

    # TINY: a random-weight model of the real class, both shapes.
    python3 tests/dump_sd3_mmdit_fixtures.py build \\
        --out ~/.mlx-serve/staging/sd3_mmdit_tiny_large
    python3 tests/dump_sd3_mmdit_fixtures.py build --dual \\
        --out ~/.mlx-serve/staging/sd3_mmdit_tiny_medium

    # REAL: parity against a downloaded checkpoint (the user runs this).
    python3 tests/dump_sd3_mmdit_fixtures.py real \\
        --model ~/.mlx-serve/staging/stable-diffusion-3.5-medium \\
        --out   ~/.mlx-serve/staging/sd3_mmdit_real

`build` writes a COMPLETE little checkpoint —
`<out>/transformer/{config.json,diffusion_pytorch_model.safetensors}` — plus
`<out>/fixture.safetensors`. The Zig test then drives its real `Mmdit.load`
against a real diffusers-shaped directory, so the config reader and the weight
binder are under test too, not just the arithmetic. `real` writes only the
fixture; the checkpoint is already on disk.

Why a tiny random model is a real oracle here: every structural decision in the
MMDiT (the CENTRE crop of the positional embedding, the concat ORDER of the two
streams, which of the nine modulation chunks gates what, the scale-before-shift
chunking of `AdaLayerNormContinuous`, the truncated last block) is a property of
the ARCHITECTURE and is exercised at any width. This is the
`dump_qwen4_exp_fixtures.py build` precedent, and it pins the shape without a
20 GB download.

TWO SHAPES, because SD 3.5 ships two and they are not the same network:

  * Large / Large-Turbo — 38 layers, no `dual_attention_layers`. Every block's
    image modulation is 6 chunks.
  * Medium — 24 layers with `dual_attention_layers: [0..12]`, i.e. MMDiT-X:
    those blocks carry a SECOND self-attention on the image stream, their
    `norm1` is `SD35AdaLayerNormZeroX` producing NINE chunks, and `attn2` has
    its own qk-norm and no `to_add_out`.

The last block of BOTH is truncated (`context_pre_only`): its `norm1_context` is
an `AdaLayerNormContinuous` (2 chunks, `scale` FIRST) rather than the 6-chunk
`AdaLayerNormZero`, it has no `attn.to_add_out` and no `ff_context`, because the
text stream's output is thrown away. Binding it like the others is a
missing-weight error; emulating it wrong silently changes the last block.

INTERMEDIATES are dumped generously and deliberately. A 38-layer tower that
disagrees only at the end is a needle in a haystack; with the patch-embedded
tokens, the conditioning vector, the context projection, EVERY block's two
outputs and the pre-unpatchify tensor pinned, a mismatch names its own block.

Runs on CPU in float32. A reference computed less precisely than the thing it
checks is not a reference.
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

# Geometry shared with the Zig test. A fixture generated at a different size
# than the implementation under test is not an oracle.
LATENT_H = 8
LATENT_W = 8
TEXT_SEQ = 7
SEED = 20260904

# A mid-schedule timestep. Flow matching hands the transformer `sigma * 1000`,
# so this is sigma 0.6285 — deliberately neither endpoint: the sinusoidal
# embedding is degenerate at 0, where a flip_sin_to_cos or freq-shift error
# passes and then fails everywhere else.
TIMESTEP = 628.5

# The tiny shape. `num_layers` 3 is the minimum that has a first block, a
# middle block and a TRUNCATED last block all at once; with --dual, layers 0
# and 1 are MMDiT-X and layer 2 is the truncated one, so a single fixture
# covers dual, plain and truncated in one tower.
TINY = dict(
    sample_size=8,
    patch_size=2,
    in_channels=16,
    out_channels=16,
    num_layers=3,
    num_attention_heads=2,
    attention_head_dim=8,
    joint_attention_dim=32,
    caption_projection_dim=16,  # == num_attention_heads * attention_head_dim
    pooled_projection_dim=24,
    pos_embed_max_size=8,  # 8 > the 4x4 patch grid, so the CENTRE crop is live
    qk_norm="rms_norm",
)


def randomize(model, seed):
    """Give every parameter a distinguishable value.

    `nn.Linear`'s default init is already non-degenerate, but diffusers'
    `RMSNorm` initialises its weight to ONES — and a qk-norm weight of all ones
    makes "multiply by the weight" and "skip the weight" the same forward. A
    real checkpoint's are trained and are not ones (verified: SD 3.5's
    `attn.norm_q.weight` is a trained bf16 vector), so they are randomised here
    or the fixture cannot see that bug at all.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm_q.weight") or name.endswith("norm_k.weight"):
                # Around 1, the way a trained scale sits, but not AT 1.
                p.copy_(1.0 + 0.25 * torch.randn(p.shape, generator=g))
    return model


def capture(model, tensors):
    """Hook every structural boundary and record its output."""
    handles = []

    def save(name):
        def hook(_mod, _inp, out):
            if isinstance(out, tuple):
                # JointTransformerBlock returns (encoder_hidden_states, hidden_states),
                # and the text half is None on the truncated last block.
                enc, hid = out
                if enc is not None:
                    tensors[f"{name}.txt"] = enc.detach().float().contiguous()
                tensors[f"{name}.img"] = hid.detach().float().contiguous()
            else:
                tensors[name] = out.detach().float().contiguous()

        return hook

    handles.append(model.pos_embed.register_forward_hook(save("cap.pos_embed")))
    handles.append(model.time_text_embed.register_forward_hook(save("cap.temb")))
    handles.append(model.context_embedder.register_forward_hook(save("cap.context")))
    for i, blk in enumerate(model.transformer_blocks):
        handles.append(blk.register_forward_hook(save(f"cap.block{i}")))
    handles.append(model.norm_out.register_forward_hook(save("cap.norm_out")))
    handles.append(model.proj_out.register_forward_hook(save("cap.proj_out")))
    return handles


def run(model, cfg, out_dir, batch):
    """Forward the model over seeded inputs and write `<out_dir>/fixture.safetensors`."""
    g = torch.Generator().manual_seed(SEED)
    in_ch = cfg["in_channels"]
    hidden = torch.randn(batch, in_ch, LATENT_H, LATENT_W, generator=g, dtype=torch.float32)
    enc = torch.randn(batch, TEXT_SEQ, cfg["joint_attention_dim"], generator=g, dtype=torch.float32)
    pooled = torch.randn(batch, cfg["pooled_projection_dim"], generator=g, dtype=torch.float32)
    # diffusers takes a per-batch-element timestep; the pipeline hands the same
    # value to both CFG halves, which is what our `forward(timestep: f32)` does.
    timestep = torch.full((batch,), TIMESTEP, dtype=torch.float32)

    tensors = {
        "in.hidden_states": hidden,
        "in.encoder_hidden_states": enc,
        "in.pooled_projections": pooled,
        "in.timestep": timestep,
    }
    handles = capture(model, tensors)
    with torch.no_grad():
        out = model(
            hidden_states=hidden,
            encoder_hidden_states=enc,
            pooled_projections=pooled,
            timestep=timestep,
            return_dict=True,
        ).sample
    for h in handles:
        h.remove()
    tensors["out.sample"] = out.detach().float().contiguous()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fixture.safetensors")
    save_file(tensors, path)
    print(f"wrote {path}")
    for k, v in tensors.items():
        print(f"  {k:24s} {tuple(v.shape)}  rms={v.float().pow(2).mean().sqrt():.5f}")


def cmd_build(args):
    from diffusers import SD3Transformer2DModel

    cfg = dict(TINY)
    if args.dual:
        # Layers 0 and 1 are MMDiT-X; layer 2 is the truncated last block. The
        # real Medium is [0..12] of 24 — same rule, fewer layers.
        cfg["dual_attention_layers"] = (0, 1)

    torch.manual_seed(SEED)
    model = SD3Transformer2DModel(**cfg).to(torch.float32).eval()
    randomize(model, SEED + 1)

    tdir = os.path.join(args.out, "transformer")
    os.makedirs(tdir, exist_ok=True)
    model.save_pretrained(tdir, safe_serialization=True)
    # `save_pretrained` writes the config with the class's OWN defaults filled
    # in, which is exactly the file the Zig loader must survive.
    with open(os.path.join(tdir, "config.json")) as f:
        print(json.dumps(json.load(f), indent=2))
    run(model, cfg, args.out, args.batch)


def cmd_real(args):
    from diffusers import SD3Transformer2DModel

    model_dir = os.path.expanduser(args.model)
    print(f"loading transformer from {model_dir}/transformer ...", flush=True)
    model = SD3Transformer2DModel.from_pretrained(
        model_dir, subfolder="transformer", torch_dtype=torch.float32
    ).eval()
    run(model, dict(model.config), os.path.expanduser(args.out), args.batch)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="tiny random-weight model of the real class")
    b.add_argument("--out", required=True)
    b.add_argument("--dual", action="store_true", help="MMDiT-X (the Medium shape)")
    b.add_argument("--batch", type=int, default=2, help="1 or 2 (the CFG batch)")

    r = sub.add_parser("real", help="parity against a downloaded checkpoint")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--batch", type=int, default=2)

    args = ap.parse_args()
    try:
        import diffusers  # noqa: F401
    except ImportError:
        sys.exit("diffusers is required to generate this fixture")
    {"build": cmd_build, "real": cmd_real}[args.cmd](args)


if __name__ == "__main__":
    main()
