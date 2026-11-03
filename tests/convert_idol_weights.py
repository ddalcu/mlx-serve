#!/usr/bin/env python3
"""Repackage `speridlabs/idol` (Apache-2.0) as an mlx-serve pack: the release's folder plus the two image encoders it
reads, laid out the way the loader wants them (needs mlx).

  hf download speridlabs/idol --local-dir <idol>
  hf download facebook/dinov3-vith16plus-pretrain-lvd1689m --local-dir <dinov3>        (gated: accept the terms)
  hf download black-forest-labs/FLUX.2-klein-4B --include 'vae/*' --local-dir <flux2>  (the FLUX.2 VAE, ungated copy)

  python3 tests/convert_idol_weights.py --idol <idol> --dinov3 <dinov3> --flux2 <flux2> --license <LICENSE> --out <pack> [--bits 8]
  python3 tests/convert_idol_weights.py --self-test        # hermetic

What changes against the release:
  - the three transformers' Linear weights and biases (the blocks, as the reference converts them) are float16;
    `--bits 8` stores each Linear whose input is a multiple of 64 as affine 8-bit, group 64. Embeddings, norms, shift
    tables and the RoPE phases stay float32.
  - DINOv3 rides as float16 (its 2-D weights), with its licence.
  - the FLUX.2 VAE keeps only what Idol reads (encoder, quant_conv, batch-norm statistics): convs channels-last
    (OHWI), `to_out.0` renamed `to_out`, the middle attention's four Linears affine 8-bit group 64 (the loader reads
    only quantized ones there). config.json gains `model_type: idol`.
  - BiRefNet's tensors are kept as released.
Install to ~/.mlx-serve/models/local/<name>/ or any folder `--model-dir` scans.
"""

import argparse
import json
import os
import shutil
import sys

import mlx.core as mx

GROUP = 64
BLOCK_PREFIXES = ("blocks.", "noise_refiner.", "context_refiner.")


def is_block_linear(name, shape):
    return name.endswith(".weight") and len(shape) == 2 and name.startswith(BLOCK_PREFIXES)


def convert_transformer(tensors, bits):
    """Linear (2-D `.weight` and its `.bias`) of the blocks to f16, or affine-quantized at `bits` < 16."""
    out = {}
    linear_bases = {k[: -len(".weight")] for k, v in tensors.items() if is_block_linear(k, v.shape)}
    for name, t in tensors.items():
        base = name.rsplit(".", 1)[0]
        if name.endswith(".weight") and base in linear_bases:
            w16 = t.astype(mx.float16)
            if bits < 16 and t.shape[1] % GROUP == 0:
                q, s, b = mx.quantize(t.astype(mx.float32), group_size=GROUP, bits=bits)
                out[name], out[base + ".scales"], out[base + ".biases"] = q, s.astype(mx.float16), b.astype(mx.float16)
            else:
                out[name] = w16
        elif name.endswith(".bias") and base in linear_bases:
            out[name] = t.astype(mx.float16)
        else:
            out[name] = t
    return out


def convert_dinov3(tensors):
    return {k: (v.astype(mx.float16) if v.ndim >= 2 and v.dtype == mx.float32 else v) for k, v in tensors.items()}


def convert_vae(tensors):
    """Encoder-only, mflux's layout."""
    out = {}
    for name, t in tensors.items():
        if name.startswith(("decoder.", "post_quant_conv.")) or name.endswith("num_batches_tracked"):
            continue
        name = name.replace(".to_out.0.", ".to_out.")
        if name.startswith("bn."):
            out[name] = t.astype(mx.float32)
        elif name.endswith(".weight") and t.ndim == 4:
            out[name] = mx.transpose(t, (0, 2, 3, 1))
        elif ".mid_block.attentions." in name and name.endswith(".weight") and t.ndim == 2:
            base = name[: -len(".weight")]
            q, s, b = mx.quantize(t.astype(mx.float32), group_size=GROUP, bits=8)
            out[name], out[base + ".scales"], out[base + ".biases"] = q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)
        else:
            out[name] = t
    return out


def load(path):
    return mx.load(path)


def save(path, tensors):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mx.save_safetensors(path, tensors, metadata={"format": "mlx"})


README = """\
---
base_model: speridlabs/idol
base_model_relation: quantized
library_name: mlx-serve
license: apache-2.0
pipeline_tag: image-to-3d
tags:
- mlx
- mlx-serve
- image-to-3d
- gaussian-splatting
- glb
---

# Idol for mlx-serve ({title})

[speridlabs/idol](https://huggingface.co/speridlabs/idol) repackaged for [mlx-serve](https://mlxserve.com/) on Apple
Silicon: one picture of an object becomes a 3D Gaussian splat, background removal included, no Python. `POST
/v1/3d/generations` answers a GLB: a mesh of textured discs any viewer draws solid (Preview, Quick Look), or with
`"geometry": "splat"` the `KHR_gaussian_splatting` points. About 40 s for 50 steps and 262,144 Gaussians on an M-series
Ultra, 3.6 GB resident.

- The three transformers' Linear layers are {linear}; norms, embeddings and the RoPE phases are as released.
- Includes everything Idol reads, so nothing else downloads: BiRefNet (background removal, as released), the DINOv3
  ViT-H+/16 encoder (float16; its licence is in `dinov3/LICENSE.md`) and the FLUX.2 VAE encoder (`vae/`, from
  black-forest-labs/FLUX.2-klein-4B, Apache-2.0; the middle attention is 8-bit).

## Licences

The Idol weights are Apache-2.0 (`LICENSE`, `NOTICE`). The bundled DINOv3 encoder is Meta's, under the DINOv3 License
(`dinov3/LICENSE.md`), which travels with any redistribution of it.

## Use

`mlx-serve --model-dir <folder holding this repo>`, or pick Idol in the app's 3D pane.

## About mlx-serve

A native Zig server for Apple Silicon. No Python, no cloud, no Electron. One binary: chat, images, video, music, speech
and 3D on MLX, behind OpenAI- and Anthropic-compatible APIs, with a signed menu-bar app.

[mlxserve.com](https://mlxserve.com/) · [GitHub](https://github.com/ddalcu/mlx-serve)

Converted by `tests/convert_idol_weights.py` in the mlx-serve repository.
"""

NOTICE = """\
Idol
Copyright (c) speridlabs. Licensed under the Apache License, Version 2.0 (see LICENSE).

This pack redistributes the Idol weights (https://huggingface.co/speridlabs/idol), converted to float16 / 8-bit and
laid out for mlx-serve, together with:

- DINOv3 ViT-H+/16 (facebook/dinov3-vith16plus-pretrain-lvd1689m), Copyright (c) Meta Platforms, Inc., under the DINOv3
  License (dinov3/LICENSE.md), which travels with any redistribution of it.
- The FLUX.2 VAE (black-forest-labs/FLUX.2-klein-4B, vae/), Apache License 2.0, converted to channels-last.
- BiRefNet (ZhengPeng7/BiRefNet), MIT License, as packaged in the Idol release (birefnet.safetensors).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idol")
    ap.add_argument("--dinov3")
    ap.add_argument("--flux2")
    ap.add_argument("--out")
    ap.add_argument("--license", help="Idol's LICENSE (Apache-2.0, in the idol-sdk repository)")
    ap.add_argument("--bits", type=int, choices=[8, 16], default=16)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.idol and a.dinov3 and a.flux2 and a.out and a.license):
        ap.error("--idol, --dinov3, --flux2, --license and --out are required")
    for name in ("denoiser", "decoder", "decoder_gs"):
        save(os.path.join(a.out, name + ".safetensors"), convert_transformer(load(os.path.join(a.idol, name + ".safetensors")), a.bits))
        print("wrote", name)
    shutil.copy(os.path.join(a.idol, "birefnet.safetensors"), os.path.join(a.out, "birefnet.safetensors"))
    cfg = json.load(open(os.path.join(a.idol, "config.json")))
    cfg["model_type"] = "idol"
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    save(os.path.join(a.out, "dinov3", "model.safetensors"), convert_dinov3(load(os.path.join(a.dinov3, "model.safetensors"))))
    for f in ("config.json", "LICENSE.md"):
        shutil.copy(os.path.join(a.dinov3, f), os.path.join(a.out, "dinov3", f))
    vae = os.path.join(a.flux2, "vae")
    save(os.path.join(a.out, "vae", "model.safetensors"), convert_vae(load(os.path.join(vae, "diffusion_pytorch_model.safetensors"))))
    shutil.copy(os.path.join(vae, "config.json"), os.path.join(a.out, "vae", "config.json"))
    shutil.copy(a.license, os.path.join(a.out, "LICENSE"))
    open(os.path.join(a.out, "NOTICE"), "w").write(NOTICE)
    title, linear = ("8-bit", "affine 8-bit, group 64, where the width allows (the rest float16)") if a.bits == 8 else ("float16", "float16")
    open(os.path.join(a.out, "README.md"), "w").write(README.format(title=title, linear=linear))
    print("pack:", a.out)


def self_test():
    w = mx.random.normal((128, 64))
    t = {
        "blocks.0.attn.qkv.weight": w, "blocks.0.attn.qkv.bias": mx.zeros((128,)),
        "blocks.0.attn.q_norm.gamma": mx.ones((2, 64)), "blocks.0.shift_table": mx.zeros((1, 8)),
        "input_layer.weight": w, "context_refiner.0.norm1.weight": mx.ones((64,)),
        "blocks.0.cross_attn.to_kv.weight": mx.random.normal((32, 16)),
    }
    f = convert_transformer(t, 16)
    assert f["blocks.0.attn.qkv.weight"].dtype == mx.float16 and f["blocks.0.attn.qkv.bias"].dtype == mx.float16
    assert f["input_layer.weight"].dtype == mx.float32 and f["context_refiner.0.norm1.weight"].dtype == mx.float32
    assert f["blocks.0.attn.q_norm.gamma"].dtype == mx.float32 and f["blocks.0.shift_table"].dtype == mx.float32
    q = convert_transformer(t, 8)
    assert q["blocks.0.attn.qkv.weight"].dtype == mx.uint32 and "blocks.0.attn.qkv.scales" in q
    assert q["blocks.0.cross_attn.to_kv.weight"].dtype == mx.float16, "a width that is not a multiple of 64 stays dense"
    back = mx.dequantize(q["blocks.0.attn.qkv.weight"], q["blocks.0.attn.qkv.scales"], q["blocks.0.attn.qkv.biases"], group_size=64, bits=8)
    assert float(mx.abs(back - w).max()) < 0.05
    vae = convert_vae({
        "encoder.conv_in.weight": mx.zeros((8, 3, 3, 3)), "decoder.conv_in.weight": mx.zeros((8, 3, 3, 3)),
        "encoder.mid_block.attentions.0.to_out.0.weight": mx.random.normal((64, 64)),
        "encoder.mid_block.attentions.0.to_out.0.bias": mx.zeros((64,)),
        "bn.running_mean": mx.zeros((4,), dtype=mx.bfloat16), "bn.num_batches_tracked": mx.zeros((), dtype=mx.int64),
    })
    assert vae["encoder.conv_in.weight"].shape == (8, 3, 3, 3) and not any(k.startswith("decoder.") for k in vae)
    assert "encoder.mid_block.attentions.0.to_out.scales" in vae and vae["bn.running_mean"].dtype == mx.float32
    print("self-test ok")


if __name__ == "__main__":
    sys.exit(main())
