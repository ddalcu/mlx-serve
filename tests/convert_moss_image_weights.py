#!/usr/bin/env python3
"""Repackage `OpenMOSS-Team/MOSS-Image-SFT` (or its mlx-community bf16 mirror) as a
quantized mlx-serve pack.

USER-RUN (needs mlx + the ~27 GB bf16 repo). Streams one shard at a time. The pack
keeps the release's diffusers layout and key names (what src/moss_image.zig loads)
with three changes:

  (a) the DiT block linears (`layers.N.proj.*`) and every text-encoder linear,
      `embed_tokens` included (mlx-lm's Qwen3.5 set: our qwen3_5 forward gathers
      quantized rows), are affine-quantized (group 64). The loaders decide
      dense-vs-quantized PER TENSOR from a `.scales` sibling;
  (b) the text encoder's vision tower is dropped (text-to-image only), and with it
      the shard indexes: loaders glob the shards;
  (c) text_encoder/config.json gains mlx-lm's `quantization` block.

NOT quantized: the VAE (f32 at decode), norms, and the DiT's small or shared
linears (text refiner blocks, the shared modulation, time and patch embedders:
~0.7 GB together).

    python3 tests/convert_moss_image_weights.py --src <repo> --out <dir> [--bits 8]
    python3 tests/convert_moss_image_weights.py --self-test
"""

import argparse
import glob
import json
import os
import shutil
import sys

GROUP_SIZE = 64

# component -> (quantize prefixes a 2D .weight may carry, dropped prefixes)
COMPONENTS = {
    "transformer": (("layers.",), ()),
    "text_encoder": (("language_model.",), ("visual.",)),
    "vae": ((), ()),
}
COPY = {
    "": ["model_index.json", "LICENSE"],
    "scheduler": ["scheduler_config.json"],
    "transformer": ["config.json"],
    "vae": ["config.json"],
    "tokenizer": ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
}


def should_drop(component, name):
    return name.startswith(COMPONENTS[component][1]) if COMPONENTS[component][1] else False


def should_quantize(component, name, shape):
    prefixes = COMPONENTS[component][0]
    return bool(prefixes) and name.startswith(prefixes) and name.endswith(".weight") and len(shape) == 2 and shape[1] % GROUP_SIZE == 0


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
            if should_quantize(component, name, tensor.shape):
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
    ap.add_argument("--bits", type=int, choices=(4, 8), default=8)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.src or not args.out:
        ap.error("--src and --out are required")

    total = sum(convert_component(args.src, args.out, c, args.bits) for c in COMPONENTS)
    for sub, names in COPY.items():
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
        for name in names:
            path = os.path.join(args.src, sub, name)
            if os.path.exists(path):
                shutil.copy(path, os.path.join(args.out, sub, name))
    with open(os.path.join(args.src, "text_encoder", "config.json")) as f:
        te = json.load(f)
    te["quantization"] = te["quantization_config"] = {"group_size": GROUP_SIZE, "bits": args.bits, "mode": "affine"}
    with open(os.path.join(args.out, "text_encoder", "config.json"), "w") as f:
        json.dump(te, f, indent=2)
    print(f"done: {total / 1e9:.2f} GB of weights -> {args.out}")


def self_test():
    q = should_quantize
    assert q("transformer", "layers.0.proj.ffn_up_gate.weight", (16384, 3072))
    assert q("transformer", "layers.41.proj.ffn_down.weight", (3072, 8192))
    assert not q("transformer", "layers.0.norm1.weight", (3072,))
    assert not q("transformer", "shared_block_modulation.weight", (18432, 3072))
    assert not q("transformer", "text_embedder.blocks.0.qkv.weight", (9216, 3072))
    assert not q("transformer", "video_embedder.weight", (3072, 128))
    assert q("text_encoder", "language_model.layers.0.mlp.down_proj.weight", (4096, 12288))
    assert q("text_encoder", "language_model.layers.0.linear_attn.in_proj_a.weight", (32, 4096))
    assert q("text_encoder", "language_model.embed_tokens.weight", (248320, 4096))
    assert not q("text_encoder", "language_model.layers.0.linear_attn.conv1d.weight", (8192, 1, 4))
    assert not q("text_encoder", "language_model.norm.weight", (4096,))
    assert should_drop("text_encoder", "visual.blocks.0.attn.qkv.weight")
    assert not should_drop("text_encoder", "language_model.layers.0.mlp.up_proj.weight")
    assert not q("vae", "decoder.conv_in.weight", (384, 32, 3, 3))
    print("ok")


if __name__ == "__main__":
    main()
