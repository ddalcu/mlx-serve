#!/usr/bin/env python3
"""Dump reference outputs for SDXL's two CLIP text towers.

The oracle for `src/sdxl_clip.zig`. Runs transformers' own CLIPTextModel /
CLIPTextModelWithProjection over the checkpoint's real weights and saves what
SDXL actually consumes:

    penultimate  hidden_states[-2]        [1, seq, hidden]   (pre-final-norm)
    pooled       text_embeds (bigG only)  [1, projection_dim]

Both are saved per tower. The token ids are fixed and shared with the Zig test
so the two sides describe the same forward — a fixture generated from different
inputs than the implementation under test is not an oracle.

    python3 tests/dump_sdxl_clip_fixtures.py \
        --model ~/.mlx-serve/staging/sdxl-base-1.0 \
        --out   ~/.mlx-serve/staging/sdxl_clip_fixture.safetensors

Runs on CPU in float32 DELIBERATELY: MPS fp16 quietly decorrelates deep towers
(the repo has that scar from the ViT/DiT fixtures), and a reference computed
less precisely than the thing it is checking is not a reference.
"""
import argparse
import os
import sys

import torch
from safetensors.torch import save_file

# Shared with the Zig test. A short padded window whose EOS is deliberately NOT
# at the last position, so pooling-at-the-end shows up as a mismatch instead of
# passing by luck.
IDS = [49406, 320, 1125, 539, 49407, 0, 0, 0]
EOS_INDEX = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="SDXL checkpoint dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_dir = os.path.expanduser(args.model)
    out = os.path.expanduser(args.out)

    try:
        from transformers import CLIPTextModel, CLIPTextModelWithProjection
    except ImportError:
        sys.exit("transformers is required to generate this fixture")

    ids = torch.tensor([IDS], dtype=torch.long)
    tensors = {}

    # ── CLIP-L: penultimate only. It has no text_projection, so SDXL takes no
    # pooled vector from it.
    # The checkpoint ships the fp16 VARIANT filenames (model.fp16.safetensors),
    # so the variant has to be named or transformers looks for model.safetensors
    # and reports the directory as empty.
    te1 = CLIPTextModel.from_pretrained(
        os.path.join(model_dir, "text_encoder"), torch_dtype=torch.float32, variant="fp16"
    ).eval()
    with torch.no_grad():
        o1 = te1(ids, output_hidden_states=True)
    tensors["clip_l.penultimate"] = o1.hidden_states[-2].contiguous()
    print(f"clip_l.penultimate  {tuple(tensors['clip_l.penultimate'].shape)}")

    # ── bigG: penultimate AND the pooled projection SDXL feeds to the
    # micro-conditioning embedder.
    te2 = CLIPTextModelWithProjection.from_pretrained(
        os.path.join(model_dir, "text_encoder_2"), torch_dtype=torch.float32, variant="fp16"
    ).eval()
    with torch.no_grad():
        o2 = te2(ids, output_hidden_states=True)
    tensors["big_g.penultimate"] = o2.hidden_states[-2].contiguous()
    tensors["big_g.pooled"] = o2.text_embeds.contiguous()
    print(f"big_g.penultimate   {tuple(tensors['big_g.penultimate'].shape)}")
    print(f"big_g.pooled        {tuple(tensors['big_g.pooled'].shape)}")

    # The ids ride along so the Zig side cannot silently drift onto a different
    # prompt than the fixture describes.
    tensors["ids"] = ids.to(torch.int32)
    tensors["eos_index"] = torch.tensor([EOS_INDEX], dtype=torch.int32)

    save_file({k: v.contiguous() for k, v in tensors.items()}, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
