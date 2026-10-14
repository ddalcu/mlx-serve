#!/usr/bin/env python3
"""openbmb/MiniCPM-V-4.7-35B-A3B (model_type minicpmv4_7) bf16 -> an MLX pack in the
layout mlx-serve loads: mlx-lm's qwen3_5_moe naming for the trunk, mlx-vlm's for the tower.

    model.language_model.*            -> language_model.model.*
    lm_head.weight                    -> language_model.lm_head.weight
    mlp.experts.gate_up_proj [E,2I,H] -> mlp.switch_mlp.{gate,up}_proj.weight [E,I,H]
    mlp.experts.down_proj    [E,H,I]  -> mlp.switch_mlp.down_proj.weight
    linear_attn.conv1d.weight [C,1,K] -> [C,K,1]
    model.vision_tower.vit_merger.*   -> vit_merger.*      (dense, mlx-vlm's minicpmv4_6 naming)
    model.vision_tower.*              -> vision_tower.*   (dense; patch_embedding as OHWI)
    model.merger.*                    -> merger.*         (dense)

The trunk's RMSNorms compute `x * (1 + w)`, so the `+1` is folded into the stored weight
(`linear_attn.norm` is the gated norm with a plain weight and is left alone). Every 2-D trunk
weight whose row is a multiple of the group size is affine-quantized at `--bits` like mlx-lm
convert does, except the router and `shared_expert_gate`, which keep 8 bits. `--bits 16`
writes the bf16 pack.

    python3 tests/convert_minicpmv4_7.py --src <hf dir> --dst <pack dir> [--bits 4] [--group-size 64]
"""

import argparse
import json
import os
import shutil
import sys

import mlx.core as mx

SHARD_BYTES = 4 * 1024 ** 3
COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
              "preprocessor_config.json", "processor_config.json", "LICENSE")
NORM_FOLD_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight", "language_model.model.norm.weight",
                      ".self_attn.q_norm.weight", ".self_attn.k_norm.weight")
EIGHT_BIT_SUFFIXES = (".mlp.gate.weight", ".mlp.shared_expert_gate.weight")


def rename(key):
    if key.startswith("model.language_model."):
        return "language_model.model." + key[len("model.language_model."):]
    if key == "lm_head.weight":
        return "language_model.lm_head.weight"
    if key.startswith("model.vision_tower.vit_merger."):
        return key[len("model.vision_tower."):]
    if key.startswith("model.vision_tower.") or key.startswith("model.merger."):
        return key[len("model."):]
    raise SystemExit(f"unexpected tensor {key}")


def convert_tensor(key, arr):
    """Yield (name, array) for one source tensor, before quantization."""
    name = rename(key)
    if name.endswith(".mlp.experts.gate_up_proj"):
        base = name[: -len("experts.gate_up_proj")]
        mid = arr.shape[-2] // 2
        yield base + "switch_mlp.gate_proj.weight", arr[..., :mid, :]
        yield base + "switch_mlp.up_proj.weight", arr[..., mid:, :]
    elif name.endswith(".mlp.experts.down_proj"):
        yield name[: -len("experts.down_proj")] + "switch_mlp.down_proj.weight", arr
    elif name.endswith("linear_attn.conv1d.weight"):
        yield name, arr.transpose(0, 2, 1)
    elif name.endswith(NORM_FOLD_SUFFIXES) and arr.ndim == 1:
        yield name, (arr.astype(mx.float32) + 1.0).astype(arr.dtype)
    elif name == "vision_tower.embeddings.patch_embedding.weight":
        yield name, arr.transpose(0, 2, 3, 1)
    else:
        yield name, arr


def bits_for(name, arr, args):
    """Quantization bits for a tensor, or None to keep it as stored."""
    if args.bits >= 16 or name.startswith(("vision_tower.", "vit_merger.", "merger.")):
        return None
    if arr.ndim < 2 or not name.endswith(".weight") or arr.shape[-1] % args.group_size != 0:
        return None
    if name.endswith(EIGHT_BIT_SUFFIXES):
        return 8
    return args.bits


class ShardWriter:
    def __init__(self, dst):
        self.dst, self.pending, self.pending_bytes, self.files, self.weight_map = dst, {}, 0, [], {}

    def add(self, name, arr):
        self.pending[name] = arr
        self.pending_bytes += arr.nbytes
        if self.pending_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        tmp = f"model-part-{len(self.files):05d}.safetensors"
        mx.eval(list(self.pending.values()))
        mx.save_safetensors(os.path.join(self.dst, tmp), self.pending, metadata={"format": "mlx"})
        for name in self.pending:
            self.weight_map[name] = tmp
        self.files.append(tmp)
        self.pending, self.pending_bytes = {}, 0

    def finish(self):
        self.flush()
        total = len(self.files)
        names = {tmp: f"model-{i + 1:05d}-of-{total:05d}.safetensors" for i, tmp in enumerate(self.files)}
        for tmp, final in names.items():
            os.rename(os.path.join(self.dst, tmp), os.path.join(self.dst, final))
        size = sum(os.path.getsize(os.path.join(self.dst, f)) for f in names.values())
        index = {"metadata": {"total_size": size}, "weight_map": {k: names[v] for k, v in sorted(self.weight_map.items())}}
        with open(os.path.join(self.dst, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 5, 6, 8, 16))
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    if os.path.exists(args.dst) and os.listdir(args.dst):
        raise SystemExit(f"{args.dst} is not empty")
    os.makedirs(args.dst, exist_ok=True)
    index = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))["weight_map"]
    shards = sorted(set(index.values()))
    for s in shards:
        if not os.path.exists(os.path.join(args.src, s)):
            raise SystemExit(f"missing shard {s}: the download is incomplete")

    writer = ShardWriter(args.dst)
    quantized = {}
    for n, shard in enumerate(shards):
        print(f"[{n + 1}/{len(shards)}] {shard}", flush=True)
        for key, arr in mx.load(os.path.join(args.src, shard)).items():
            for name, out in convert_tensor(key, arr):
                bits = bits_for(name, out, args)
                if bits is None:
                    writer.add(name, out)
                    continue
                wq, scales, biases = mx.quantize(out, group_size=args.group_size, bits=bits)
                stem = name[: -len(".weight")]
                writer.add(name, wq)
                writer.add(stem + ".scales", scales)
                writer.add(stem + ".biases", biases)
                if bits != args.bits:
                    quantized[stem] = {"group_size": args.group_size, "bits": bits}
    writer.finish()

    cfg = json.load(open(os.path.join(args.src, "config.json")))
    if args.bits < 16:
        q = {"group_size": args.group_size, "bits": args.bits, "mode": "affine", **quantized}
        cfg["quantization"] = cfg["quantization_config"] = q
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for name in COPY_FILES:
        if os.path.exists(os.path.join(args.src, name)):
            shutil.copy(os.path.join(args.src, name), args.dst)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    sys.exit(main())
