#!/usr/bin/env python3
"""Repackage `Qwen/Qwen-Image-2.1` as a quantized mlx-serve pack.

USER-RUN (needs mlx + the ~33 GB bf16 repo). Streams one tensor at a time, so it
runs on a 32 GB Mac. The pack keeps the checkpoint's own layout and key names
(what src/qwen_image.zig loads) with three changes:

  (a) the DiT block linears and the text-encoder layer linears are
      affine-quantized (group 64). `MfLinear` decides dense-vs-quantized PER
      TENSOR from a `.scales` sibling, so nothing else records the choice;
  (b) what the engine never reads is dropped: the Qwen3-VL vision tower and
      `lm_head` (text-to-image only), and the VAE's per-frame `time_conv`s;
  (c) a root `config.json` carries `model_type: qwen_image21`, which is how
      discovery classifies the pack.

    --preset 32gb   DiT 8-bit + text encoder 8-bit   ~17 GB
    --preset 16gb   DiT 4-bit + text encoder 4-bit   ~10 GB (engine stages the TE)

NOT quantized: the VAE (f32, load-bearing), `embed_tokens` (read with
`mlx_take_axis`, never a matmul), norms, and the DiT's small or shared linears
(img_in, txt_in, timestep embedder, the ONE modulation every block reads,
norm_out, proj_out: ~0.3 GB together).

    python3 tests/convert_qwen_image21_weights.py --src <repo> --out <dir> --preset 32gb
    python3 tests/convert_qwen_image21_weights.py --self-test

Apache-2.0 upstream.
"""

import argparse
import glob
import json
import os
import shutil
import sys

GROUP_SIZE = 64
PRESETS = {"32gb": (8, 8), "16gb": (4, 4)}

# component -> (quantize prefixes a 2D .weight may carry, dropped key fragments)
# the tower prefixes name its matmul linears only: pos_embed is a gather-read table
# (embed_tokens precedent) and patch_embed a conv, so both stay dense
COMPONENTS = {
    "transformer": (("transformer_blocks.",), ()),
    "text_encoder": (
        ("model.language_model.layers.", "model.visual.blocks.", "model.visual.merger.", "model.visual.deepstack_merger_list."),
        ("lm_head.",),
    ),
    "vae": (None, (".time_conv.",)),
}
COPY = {
    "": ["model_index.json", "LICENSE"],
    "scheduler": ["scheduler_config.json"],
    "transformer": ["config.json"],
    "vae": ["config.json"],
    "text_encoder": ["config.json"],
    "processor": ["tokenizer.json", "tokenizer_config.json"],
}


README = """---
license: apache-2.0
base_model: Qwen/Qwen-Image-2.1
base_model_relation: quantized
library_name: mlx-serve
tags:
  - mlx
  - mlx-serve
  - quantized
  - text-to-image
pipeline_tag: text-to-image
---

# Qwen-Image-2.1 MLX-Serve {bits}-bit

{bits}-bit pack of [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) for
[mlx-serve](https://github.com/ddalcu/mlx-serve): {size_gb:.1f} GB, for {target} Macs.

> **Not released yet.** These packs load only on the mlx-serve branch
> [`feat/qwen-image-2.1`](https://github.com/ddalcu/mlx-serve/pull/477). No released
> mlx-serve or MLX Core build can run them. This notice goes away when the PR ships.

![sample]({sample})

## What is in it

The checkpoint's own diffusers layout and key names, with the DiT block linears and the
text-encoder layer linears affine-quantized to {bits}-bit (group 64), plus the Qwen3-VL
vision tower kept with its 2D linears quantized the same way (patch embed and position
table stay dense), so the pack also carries instruction-edit capability. Kept dense: the VAE (f32), `embed_tokens`, norms, and the DiT's small or
shared linears. Dropped: `lm_head` (text-to-image only) and the VAE's per-frame
`time_conv`s. Built by `tests/convert_qwen_image21_weights.py --preset {preset}`.

## Measured (M1 Pro, 32 GB)

| Pack | Size | Steps | Wall clock incl. load | Peak memory |
|---|---|---|---|---|
| 8-bit | 1024x1024 | 40 | 985 s (~23 s/step) | 12.95 GB |
| 4-bit | 1024x1024 | 3 | 87 s | 9.55 GB |
| 4-bit | 512x512 | 20 | 118 s | - |

On a Mac the full set would crowd, mlx-serve loads the text encoder per request and frees
it before the denoise, so the resident set is the DiT and VAE.

## Run it (from the branch)

```sh
git clone -b feat/qwen-image-2.1 https://github.com/ddalcu/mlx-serve && cd mlx-serve
./scripts/fetch-zig.sh && ./scripts/build-mlx.sh && .zig-toolchain/zig build -Doptimize=ReleaseFast
./zig-out/bin/mlx-serve pull {repo}
./zig-out/bin/mlx-serve serve
curl localhost:11234/v1/images/generations -H 'Content-Type: application/json' \\
  -d '{{"model":"{repo}","prompt":"a red fox in fresh snow","size":"1024x1024"}}'
```

40 steps when `steps` is omitted. `guidance_scale` above 1 with a `negative_prompt` runs
real CFG (two forwards per step). `image` + `strength` does image-to-image.

Apache-2.0, same as the base model.
"""

CARDS = {
    "32gb": dict(bits=8, target="32 GB", repo="ddalcu/Qwen-Image-2.1-MLX-Serve-8bit",
                 sample="https://raw.githubusercontent.com/ddalcu/mlx-serve/feat/qwen-image-2.1/website/screenshots/qwen-image-2.1-8bit-1024.jpg"),
    "16gb": dict(bits=4, target="16 GB", repo="ddalcu/Qwen-Image-2.1-MLX-Serve-4bit",
                 sample="https://raw.githubusercontent.com/ddalcu/mlx-serve/feat/qwen-image-2.1/website/screenshots/qwen-image-2.1-4bit-512.jpg"),
}


def write_card(out, preset, size_gb):
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(README.format(preset=preset, size_gb=size_gb, **CARDS[preset]))


def should_drop(component, name):
    return any(frag in name for frag in COMPONENTS[component][1])


def should_quantize(component, name, shape, bits):
    prefixes = COMPONENTS[component][0]
    if prefixes is None or bits >= 16 or not any(name.startswith(p) for p in prefixes):
        return False
    return name.endswith(".weight") and len(shape) == 2 and shape[1] % GROUP_SIZE == 0


def convert_component(src, out, component, bits):
    import mlx.core as mx

    os.makedirs(os.path.join(out, component), exist_ok=True)
    shards = sorted(glob.glob(os.path.join(src, component, "*.safetensors")))
    if not shards:
        sys.exit(f"no safetensors under {src}/{component}")
    total = 0
    for shard in shards:
        result = {}
        for name, tensor in mx.load(shard).items():
            if should_drop(component, name):
                continue
            if should_quantize(component, name, tensor.shape, bits):
                base = name[: -len(".weight")]
                new = dict(zip((name, base + ".scales", base + ".biases"), mx.quantize(tensor, group_size=GROUP_SIZE, bits=bits)))
            else:
                new = {name: tensor}
            mx.eval(*new.values())
            result.update(new)
        dst = os.path.join(out, component, os.path.basename(shard))
        mx.save_safetensors(dst, result)
        total += os.path.getsize(dst)
        print(f"  {component}/{os.path.basename(shard)}: {len(result)} tensors, {os.path.getsize(dst) / 1e9:.2f} GB")
        del result
        mx.clear_cache()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="32gb")
    ap.add_argument("--dit-bits", type=int)
    ap.add_argument("--te-bits", type=int)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--card-only", type=float, metavar="GB", help="rewrite <out>/README.md for an existing pack of this size")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.card_only:
        return write_card(args.out, args.preset, args.card_only)
    if not args.src or not args.out:
        ap.error("--src and --out are required")

    dit_bits, te_bits = PRESETS[args.preset]
    dit_bits, te_bits = args.dit_bits or dit_bits, args.te_bits or te_bits
    total = 0
    for component, bits in (("transformer", dit_bits), ("text_encoder", te_bits), ("vae", 16)):
        total += convert_component(args.src, args.out, component, bits)
    for sub, names in COPY.items():
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
        for name in names:
            path = os.path.join(args.src, sub, name)
            if os.path.exists(path):
                shutil.copy(path, os.path.join(args.out, sub, name))
    root = {"model_type": "qwen_image21", "quantization": {"group_size": GROUP_SIZE, "dit_bits": dit_bits, "te_bits": te_bits}}
    json.dump(root, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    write_card(args.out, args.preset, total / 1e9)
    print(f"done: {total / 1e9:.2f} GB of weights -> {args.out}")


def self_test():
    q = should_quantize
    assert q("transformer", "transformer_blocks.0.attn.to_q.weight", (4096, 4096), 8)
    assert q("transformer", "transformer_blocks.3.img_mlp.out.weight", (4096, 12288), 4)
    assert not q("transformer", "transformer_blocks.0.attn.norm_q.weight", (128,), 8)
    assert not q("transformer", "modulation.1.weight", (16384, 4096), 8)
    assert not q("transformer", "img_in.weight", (4096, 64), 8)
    assert q("text_encoder", "model.language_model.layers.0.mlp.down_proj.weight", (4096, 12288), 8)
    assert not q("text_encoder", "model.language_model.embed_tokens.weight", (151936, 4096), 8)
    assert not q("text_encoder", "model.language_model.layers.0.mlp.down_proj.weight", (4096, 12288), 16)
    assert not q("vae", "decoder.conv_in.weight", (1152, 64, 3, 3), 8)
    assert should_drop("text_encoder", "lm_head.weight")
    # tower kept for the edit path: 2D linears quantize, pos_embed/patch-embed/norms stay dense
    assert not should_drop("text_encoder", "model.visual.blocks.0.attn.qkv.weight")
    assert q("text_encoder", "model.visual.blocks.0.attn.qkv.weight", (3456, 1152), 4)
    assert q("text_encoder", "model.visual.blocks.0.mlp.linear_fc1.weight", (4304, 1152), 8)
    assert q("text_encoder", "model.visual.merger.linear_fc1.weight", (2048, 4608), 4)
    assert q("text_encoder", "model.visual.deepstack_merger_list.0.linear_fc1.weight", (2048, 4608), 4)
    assert not q("text_encoder", "model.visual.pos_embed.weight", (2304, 1152), 4)
    assert not q("text_encoder", "model.visual.patch_embed.proj.weight", (1152, 3, 2, 16, 16), 4)
    assert not q("text_encoder", "model.visual.blocks.0.norm1.weight", (1152,), 4)
    assert should_drop("vae", "decoder.up_blocks.0.upsampler.time_conv.weight")
    assert not should_drop("vae", "decoder.up_blocks.0.upsampler.resample.1.weight")
    for preset in CARDS:
        assert "Not released yet" in README.format(preset=preset, size_gb=1.0, **CARDS[preset])
    print("ok")


if __name__ == "__main__":
    main()
