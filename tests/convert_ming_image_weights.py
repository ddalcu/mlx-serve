#!/usr/bin/env python3
"""Quantize `inclusionAI/Ming-Image-0.2-Design` (or -0.1-Design) into mlx-community's Ming-Image layout.

USER-RUN (needs mlx + the ~53 GB release). Streams one shard at a time. The pack is the release's component tree
with MLX affine `.scales` / `.biases` beside each quantized weight (group 64) and a `quantization` block in each
quantized component's config.json: what mlx-community/Ming-Image-0.1-Design-{4bit,8bit} publish, so the packs load
in mlx-serve (src/plugins/ming_image/) and in the ming-image-swift port alike.

Quantized (the mlx-community tiers):
  mllm/        attention, the dense and shared-expert MLPs, every routed expert   --mllm-bits (4 or 8)
  connector/   every decoder-layer linear                                          8
  transformer/ the blocks' attention and feed-forward linears                      8 (a DiT under 8 bits is damage)
Kept as released: the MoE routers, embeddings, LM head, norms, the vision tower, mlp/, the VAE and the DiT's
conditioning layers (adaLN, embedders, the final layer).

    python3 tests/convert_ming_image_weights.py --src <release dir> --out <dir> --mllm-bits 4
    python3 tests/convert_ming_image_weights.py --self-test
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

GROUP_SIZE = 64

QUANTIZED = {
    "mllm": re.compile(r"^model\.model\.layers\.\d+\.(attention\.(query_key_value|dense)|mlp\.((experts\.\d+|shared_experts)\.)?(gate|up|down)_proj)\.weight$"),
    "connector": re.compile(r"^model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$"),
    "transformer": re.compile(r"^(layers|noise_refiner|context_refiner)\.\d+\.(attention\.(to_[qkv]|to_out\.0)|feed_forward\.w[123])\.weight$"),
}

# Measured on an M5 Ultra (mlx-serve, 12 steps, guidance 1); quality = PSNR of the same seed against the bf16
# release over six photographic prompts (1024² to 2560x1440).
TIERS = {
    4: {"size": "21 GB", "resident": "17.6", "peak1k": "20.4", "peak2k": "28.9", "psnr": "18.8-28.3 (median 22.4)",
        "macs": "32 GB for 1024², 48 GB for 2048²"},
    8: {"size": "29 GB", "resident": "25.4", "peak1k": "28.2", "peak2k": "36.7", "psnr": "24.8-33.2 (median 29.9)",
        "macs": "48 GB for 1024², 64 GB for 2048²"},
}

README = """---
license: mit
license_link: https://huggingface.co/inclusionAI/Ming-Image-0.2-Design/blob/main/LICENSE
base_model: inclusionAI/Ming-Image-0.2-Design
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-to-image
tags:
- mlx-serve
- mlx
- apple-silicon
- text-to-image
- image-generation
- rgba
- graphic-design
- text-rendering
- {bits}-bit
---

# Ming-Image-0.2-Design — MLX-Serve {bits}-bit

The {bits}-bit MLX conversion of [inclusionAI/Ming-Image-0.2-Design](https://huggingface.co/inclusionAI/Ming-Image-0.2-Design)
for [mlx-serve](https://github.com/ddalcu/mlx-serve), running natively on Apple Silicon. Text to image with native
alpha: every image comes back as an RGBA PNG, from photographs and product shots to posters, UI drafts and text-heavy
designs.

| | |
|---|---|
| Download | {size} (the bf16 release is 53 GB) |
| Resident after load | {resident} GB |
| Peak while generating | {peak1k} GB at 1024², {peak2k} GB at 2048² |
| Runs on | a Mac with {macs} (macOS gives the GPU about 3/4 of RAM) |

## Speed

M5 Ultra, 12 steps (the release's default), one image, wall clock through the HTTP API:

| Canvas | This pack | bf16 release |
|---|---|---|
| 1024x1024 | 7 s | 7 s |
| 2560x1440 / 1440x2560 | 33 s | 34 s |
| 2048x2048 | 40 s | 41 s |

Quantization buys memory here, not speed: the DiT is compute-bound at these canvases.

## Quality

Same prompt and seed as the bf16 release, PSNR of the render against the bf16 render: {psnr} dB over six
photographic prompts. The {bits}-bit pack keeps the composition, lighting and fine texture; what drifts first is
generated lettering on signs.

## What is quantized

Affine, group size 64, in [mlx-community's Ming-Image layout](https://huggingface.co/mlx-community/Ming-Image-0.1-Design-8bit):
the release's component tree with MLX `.scales` / `.biases` beside each quantized weight and a `quantization` block in
each quantized component's `config.json`.

| Component | Here |
|---|---|
| `mllm/` 16B-A1.4B Bailing-MoE MLLM: attention, dense, shared and the 256 routed experts | {bits}-bit |
| `connector/` Qwen2-1.5B connector | 8-bit |
| `transformer/` 6B DiT attention and feed-forward | 8-bit |
| routers, embeddings, LM head, norms, the vision tower, `mlp/`, the VAE, the DiT's conditioning layers | as released |

Written by `tests/convert_ming_image_weights.py` in the mlx-serve repository; the port's parity against the release's
own code is `tests/dump_ming_image_fixtures.py`.

## Use

```bash
mlx-serve --serve --model-dir ~/.mlx-serve/models
curl http://127.0.0.1:11234/v1/images/generations -H 'Content-Type: application/json' -d '{{
  "model": "Ming-Image-0.2-Design-MLX-Serve-{bits}bit",
  "prompt": "Studio product photograph of a mechanical wristwatch with a deep blue sunburst dial on black slate",
  "size": "1024x1024"
}}'
```

Or pick it in the MLX Serve app's image tab. A size snaps to the release's nearest aspect bucket (512, 1024 and 2048
tables, 1:4 to 4:1). `steps` defaults to 12; `guidance_scale` above 1 runs the release's guidance (two forwards per
step); the release has no negative prompt.

Edits: send one picture with `"mode": "edit"` and an instruction (`"image"` as base64, e.g. "Change the yellow
raincoat to red" or "Remove the background"). The result keeps the picture's aspect at about 1 MP, alpha included;
an edit takes about 19 s at 1024² on an M5 Ultra (the reference doubles the DiT's sequence).

Prompting: describe the picture as a photographer or designer would. Brand and magazine names in a prompt get typeset
into the image (it is a design model), so leave them out unless you want the lettering.

## License

MIT, as the release ([LICENSE](LICENSE), copied unchanged). Model by inclusionAI.
"""


def card(bits):
    return README.format(bits=bits, **TIERS[bits])


def should_quantize(component, name, shape):
    rx = QUANTIZED.get(component)
    return rx is not None and bool(rx.match(name)) and len(shape) == 2 and shape[1] % GROUP_SIZE == 0


def convert_component(src, out, component, bits):
    import mlx.core as mx

    os.makedirs(os.path.join(out, component), exist_ok=True)
    shards = sorted(glob.glob(os.path.join(src, component, "*.safetensors")))
    if not shards:
        sys.exit(f"no safetensors under {src}/{component}")
    weight_map, total = {}, 0
    for shard in shards:
        result = {}
        for name, tensor in mx.load(shard).items():
            if should_quantize(component, name, tensor.shape):
                # The release computes in bf16 (its connector ships f32): quantize what it multiplies, so the
                # scales and biases are bf16 too.
                tensor = tensor.astype(mx.bfloat16)
                base = name[: -len(".weight")]
                new = dict(zip((name, base + ".scales", base + ".biases"), mx.quantize(tensor, group_size=GROUP_SIZE, bits=bits)))
            else:
                new = {name: tensor}
            mx.eval(*new.values())
            result.update(new)
        dst = os.path.join(out, component, os.path.basename(shard))
        mx.save_safetensors(dst, result, metadata={"format": "mlx"})
        weight_map.update({k: os.path.basename(shard) for k in result})
        total += os.path.getsize(dst)
        print(f"  {component}/{os.path.basename(shard)}: {len(result)} tensors, {os.path.getsize(dst) / 1e9:.2f} GB")
        del result
        mx.clear_cache()
    if len(shards) > 1 or glob.glob(os.path.join(src, component, "*.index.json")):
        index = glob.glob(os.path.join(src, component, "*.index.json"))
        name = os.path.basename(index[0]) if index else "model.safetensors.index.json"
        with open(os.path.join(out, component, name), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=2)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--mllm-bits", type=int, choices=(4, 8), default=4)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.src or not args.out:
        ap.error("--src and --out are required")
    if os.path.abspath(args.src) == os.path.abspath(args.out):
        ap.error("--out must differ from --src")

    bits = {"mllm": args.mllm_bits, "connector": 8, "transformer": 8}
    total = 0
    for component in ("mllm", "connector", "transformer", "mlp", "vae"):
        total += convert_component(args.src, args.out, component, bits.get(component, 0))
    # Every non-weight file as released (configs, tokenizer, scheduler, license), then our card.
    for root, _, files in os.walk(args.src):
        rel = os.path.relpath(root, args.src)
        if rel.startswith(".cache") or rel.startswith("assets"):
            continue
        for f in files:
            if f.endswith(".safetensors") or f.endswith(".safetensors.index.json"):
                continue
            os.makedirs(os.path.join(args.out, rel), exist_ok=True)
            shutil.copy(os.path.join(root, f), os.path.join(args.out, rel, f))
    for component, b in bits.items():
        path = os.path.join(args.out, component, "config.json")
        with open(path) as f:
            cfg = json.load(f)
        cfg["quantization"] = {"group_size": GROUP_SIZE, "bits": b}
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(card(args.mllm_bits))
    print(f"done: {total / 1e9:.2f} GB of weights -> {args.out}")


def self_test():
    q = should_quantize
    assert q("mllm", "model.model.layers.0.attention.query_key_value.weight", (3072, 2048))
    assert q("mllm", "model.model.layers.3.attention.dense.weight", (2048, 2048))
    assert q("mllm", "model.model.layers.0.mlp.down_proj.weight", (2048, 5120))
    assert q("mllm", "model.model.layers.7.mlp.experts.255.gate_proj.weight", (512, 2048))
    assert q("mllm", "model.model.layers.7.mlp.shared_experts.up_proj.weight", (512, 2048))
    assert not q("mllm", "model.model.layers.7.mlp.gate.weight", (256, 2048))
    assert not q("mllm", "model.model.layers.7.mlp.image_gate.weight", (256, 2048))
    assert not q("mllm", "model.model.word_embeddings.weight", (157184, 2048))
    assert not q("mllm", "model.lm_head.weight", (157184, 2048))
    assert not q("mllm", "vision.blocks.0.attn.qkv.weight", (3840, 1280))
    assert not q("mllm", "model.model.layers.0.attention.q_norm.weight", (128,))
    assert q("connector", "model.layers.27.self_attn.o_proj.weight", (1536, 1536))
    assert q("connector", "model.layers.0.mlp.down_proj.weight", (1536, 8960))
    assert not q("connector", "model.embed_tokens.weight", (151936, 1536))
    assert q("transformer", "layers.29.attention.to_out.0.weight", (3840, 3840))
    assert q("transformer", "noise_refiner.1.feed_forward.w2.weight", (3840, 10240))
    assert q("transformer", "context_refiner.0.attention.to_q.weight", (3840, 3840))
    assert not q("transformer", "layers.0.adaLN_modulation.0.weight", (15360, 256))
    assert not q("transformer", "cap_embedder.1.weight", (3840, 2560))
    assert not q("transformer", "all_x_embedder.2-1.weight", (3840, 64))
    assert not q("transformer", "all_final_layer.2-1.linear.weight", (64, 3840))
    assert not q("vae", "decoder.conv_in.weight", (384, 16, 3, 3, 3))
    assert not q("mlp", "proj_in.weight", (1536, 2048))
    for b in (4, 8):
        assert "base_model_relation: quantized" in card(b) and f"{b}-bit" in card(b)
    print("ok")


if __name__ == "__main__":
    main()
