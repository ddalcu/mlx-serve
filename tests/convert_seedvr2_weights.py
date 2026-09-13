#!/usr/bin/env python3
"""
Convert a SeedVR2 release into the layout mlx-serve's restore engine loads.

    uv run --with torch --with safetensors --with numpy \\
        tests/convert_seedvr2_weights.py \\
        --dit  ~/.mlx-serve/downloads/seedvr2-3b/seedvr2_ema_3b_fp16.safetensors \\
        --vae  ~/.mlx-serve/downloads/seedvr2-3b/ema_vae_fp16.safetensors \\
        --pos-emb ~/.mlx-serve/downloads/seedvr2-3b/pos_emb.pt \\
        --out  ~/.mlx-serve/models/ByteDance-Seed/SeedVR2-3B-MLX-Serve

Produces:

    config.json            {"model_type": "seedvr2", ...}
    vae.safetensors        encoder.* + decoder.*      (~0.5 GB fp16)
    dit.safetensors        the 635 NaDiT tensors      (~6.8 GB fp16)
    pos_emb.safetensors    {"pos_emb": [L, 5120]}     (~0.6 MB)

WHY A CONVERTER AT ALL, when two of the three inputs are already safetensors:

  * `pos_emb.pt` is a PICKLE. It is the only text conditioning SeedVR2 has —
    there is no text encoder at runtime — and it cannot be loaded without torch.
    Baking it into the pack is what makes the pack self-contained.
  * The engine looks up fixed filenames. Shipping `dit.safetensors` rather than
    `seedvr2_ema_3b_fp16.safetensors` means the loader does not have to guess
    among the six differently-named mirrors of the same weights.
  * `dit.safetensors` doubles as the pack's COMPLETENESS MARKER
    (`model_discovery.requiredMediaMarker`). It is the largest file and lands
    last, so a half-finished download is invisible to discovery rather than
    registering as a model and dying in the text loader.

Nothing is requantized here. The bytes are copied through in whatever dtype the
source carries; a quantized mirror is a separate step.
"""

import argparse
import json
import os
import shutil
import sys

# The manifest the Zig side expects, mirrored from src/seedvr2_manifest.zig.
EXPECTED_DIT_TENSORS = 635
SPLIT_SHARED_BOUNDARY = 10
NUM_LAYERS = 32


def load_any(path):
    """Accept safetensors or a torch pickle; return a plain dict of tensors."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "module", "ema", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            return sd[key]
    return sd


def check_dit(sd):
    """Fail LOUDLY on a checkpoint whose layout the engine cannot load.

    The split/shared boundary is the one thing that is invisible downstream:
    every tensor shape matches under either naming, so a checkpoint with the
    wrong boundary loads 22 layers of nothing and still emits an image.
    """
    problems = []
    if len(sd) != EXPECTED_DIT_TENSORS:
        problems.append(f"expected {EXPECTED_DIT_TENSORS} tensors, found {len(sd)}")

    split, shared = set(), set()
    for k in sd:
        parts = k.split(".")
        if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit():
            i = int(parts[1])
            if ".vid." in k or ".txt." in k:
                split.add(i)
            if ".all." in k:
                shared.add(i)
    want_split = set(range(SPLIT_SHARED_BOUNDARY))
    want_shared = set(range(SPLIT_SHARED_BOUNDARY, NUM_LAYERS))
    if split != want_split:
        problems.append(f"split-weight layers are {sorted(split)}, expected 0..{SPLIT_SHARED_BOUNDARY - 1}")
    if shared != want_shared:
        problems.append(f"shared-weight layers are {sorted(shared)}, expected {SPLIT_SHARED_BOUNDARY}..{NUM_LAYERS - 1}")
    if split & shared:
        problems.append(f"layers with BOTH namings: {sorted(split & shared)}")

    # The rope table is a stored per-layer buffer whose length encodes the whole
    # mmrope3d derivation: rope_dim/3 = 42, arange(0,42,2)[:21] -> 21.
    ropes = [k for k in sd if k.endswith("attn.rope.rope.freqs")]
    if len(ropes) != NUM_LAYERS:
        problems.append(f"{len(ropes)} rope freq buffers, expected {NUM_LAYERS}")
    elif tuple(sd[ropes[0]].shape) != (21,):
        problems.append(f"rope freqs are {tuple(sd[ropes[0]].shape)}, expected (21,)")

    for required in ("vid_in.proj.weight", "txt_in.weight", "emb_in.proj_out.weight",
                     "vid_out_norm.weight", "vid_out_ada.out_shift",
                     "vid_out_ada.out_scale", "vid_out.proj.weight"):
        if required not in sd:
            problems.append(f"missing {required}")
    return problems


def check_vae(sd):
    problems = []
    enc = [k for k in sd if k.startswith("encoder.")]
    dec = [k for k in sd if k.startswith("decoder.")]
    if not enc:
        problems.append("no encoder.* tensors")
    if not dec:
        problems.append("no decoder.* tensors")
    # time_receptive_field is "full" for this checkpoint: resnet convs are
    # (3,3,3). A (1,3,3) here is the "half" variant and will not load.
    k = "encoder.down_blocks.0.resnets.0.conv1.weight"
    if k in sd and tuple(sd[k].shape)[2:] != (3, 3, 3):
        problems.append(f"{k} is {tuple(sd[k].shape)}; expected a (3,3,3) kernel "
                        "(time_receptive_field='full')")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--pos-emb", required=True, help="pos_emb.pt from the ByteDance repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="overwrite an existing pack")
    args = ap.parse_args()

    from safetensors.torch import save_file

    out = os.path.expanduser(args.out)
    marker = os.path.join(out, "dit.safetensors")
    if os.path.exists(marker) and not args.force:
        print(f"refusing to overwrite {marker} (pass --force)")
        return 1
    os.makedirs(out, exist_ok=True)

    print(f"reading DiT  {args.dit}")
    dit = load_any(os.path.expanduser(args.dit))
    problems = check_dit(dit)
    if problems:
        print("\n  DiT checkpoint does not match the expected layout:")
        for p in problems:
            print(f"    - {p}")
        print("\n  Refusing to write a pack the engine cannot load.")
        return 1
    print(f"  ok: {len(dit)} tensors, split 0..9 / shared 10..31")

    print(f"reading VAE  {args.vae}")
    vae = load_any(os.path.expanduser(args.vae))
    problems = check_vae(vae)
    if problems:
        print("\n  VAE checkpoint does not match the expected layout:")
        for p in problems:
            print(f"    - {p}")
        return 1
    print(f"  ok: {len(vae)} tensors")

    print(f"reading text embedding {args.pos_emb}")
    import torch
    pos = torch.load(os.path.expanduser(args.pos_emb), map_location="cpu", weights_only=True)
    if isinstance(pos, (list, tuple)):
        pos = pos[0]
    if isinstance(pos, dict):
        pos = next(iter(pos.values()))
    pos = pos.detach().float().contiguous()
    if pos.ndim == 3 and pos.shape[0] == 1:
        pos = pos[0]
    if pos.ndim != 2 or pos.shape[1] != 5120:
        print(f"  pos_emb has shape {tuple(pos.shape)}; expected [L, 5120] "
              "(txt_in_dim from the 3B config)")
        return 1
    print(f"  ok: {tuple(pos.shape)}")

    # VAE and text embedding first, DiT LAST — the DiT is the completeness
    # marker, so writing it first would make a crashed conversion look complete.
    print(f"\nwriting {out}")
    save_file(vae, os.path.join(out, "vae.safetensors"))
    print("  vae.safetensors")
    save_file({"pos_emb": pos}, os.path.join(out, "pos_emb.safetensors"))
    print("  pos_emb.safetensors")

    config = {
        "model_type": "seedvr2",
        "architectures": ["NaDiT"],
        "vid_dim": 2560,
        "num_layers": 32,
        "mm_layers": 10,
        "heads": 20,
        "head_dim": 128,
        "vid_in_channels": 33,
        "vid_out_channels": 16,
        "patch_size": [1, 2, 2],
        "rope_dim": 128,
        "window": [4, 3, 3],
        "vae": {
            "latent_channels": 16,
            "spatial_downsample_factor": 8,
            "temporal_downsample_factor": 4,
            "scaling_factor": 0.9152,
        },
        "_converted_by": "tests/convert_seedvr2_weights.py",
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    print("  config.json")

    tmp = marker + ".partial"
    save_file(dit, tmp)
    shutil.move(tmp, marker)
    print("  dit.safetensors  (completeness marker — written last, atomically)")
    print(f"\ndone. Point the server at it:\n  mlx-serve --model-dir {os.path.dirname(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
