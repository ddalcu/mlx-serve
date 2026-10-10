#!/usr/bin/env python3
"""m-a-p/YuE2-3B + m-a-p/YuE2-Vae -> the YuE2 pack mlx-serve loads, in the layout of
ahmadw/YuE2-3B-MLX (whose 8bit/ and bf16/ folders this reproduces tensor for tensor).

  model.safetensors  every Linear under model.layers plus lm_head affine-quantized: the AR
                     path at --bits, the nar_* path at --nar-bits, one --group-size; the
                     embedding, norms and time/latent projections stay bf16, and
                     time_embedder.mlp.{0,2} becomes time_embedder.fc{1,2}. --bits 16 = bf16.
  vae.safetensors    the decoder alone, weight norm folded, MLX conv layout [out, k, in], f32
  config.json (+ quantization), vae_config.json, qwen.tiktoken, yue2_generation_config.json,
  README.md

  hf download m-a-p/YuE2-3B --local-dir src/model; hf download m-a-p/YuE2-Vae --local-dir src/vae
  uv run --with mlx tests/convert_yue2.py --model src/model --vae src/vae --bits 4 \\
      --dst ~/.mlx-serve/models/ddalcu/YuE2-3B-MLX-Serve-4bit
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx

README = """\
---
license: cc-by-nc-4.0
base_model: m-a-p/YuE2-3B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-to-audio
tags:
- mlx
- music-generation
- yue2
language:
- en
- zh
---

# YuE2-3B — MLX Serve ({title})

[m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) with the [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae)
decoder folded in, {weights}, in the layout of [ahmadw/YuE2-3B-MLX](https://huggingface.co/ahmadw/YuE2-3B-MLX).
Converted from the m-a-p release by mlx-serve's `tests/convert_yue2.py`.
Lyrics and a style prompt in, a 48 kHz stereo song out, planned from an editable ABC score.

Runs in [mlx-serve](https://github.com/ddalcu/mlx-serve) (`POST /v1/audio/music-generations` and the app's Music tab):

    mlx-serve pull ddalcu/YuE2-3B-MLX-Serve-{tag}

Files: `model.safetensors` (backbone), `vae.safetensors` (decoder), `config.json`, `vae_config.json`,
`qwen.tiktoken`, `yue2_generation_config.json`.

## License

Weights are **CC BY-NC 4.0 (non-commercial)**, inherited from YuE2-3B and YuE2-Vae. Credit YuE2 (m-a-p) when you
publish output. The SnakeBeta and Oobleck decoder code are MIT (NVIDIA, Stability AI); see the upstream
`THIRD_PARTY_NOTICES.md`.
"""

TIME_EMBEDDER = {"time_embedder.mlp.0.": "time_embedder.fc1.", "time_embedder.mlp.2.": "time_embedder.fc2."}
CONV_TRANSPOSE = re.compile(r"^layers\.\d+\.layers\.1\.weight_v$")


def rename(key):
    for old, new in TIME_EMBEDDER.items():
        if key.startswith(old):
            return new + key[len(old):]
    return key


def quantizable(key, w):
    return key.endswith(".weight") and w.ndim == 2 and (key.startswith("model.layers.") or key == "lm_head.weight")


def backbone(src, bits, nar_bits, group_size):
    out = {}
    for key, w in mx.load(str(src)).items():
        key = rename(key)
        if bits == 16 or not quantizable(key, w):
            out[key] = w
            continue
        wq, scales, biases = mx.quantize(w, group_size, nar_bits if ".nar_" in key else bits)
        base = key[: -len("weight")]
        out[key], out[base + "scales"], out[base + "biases"] = wq, scales, biases
    return out


def decoder(src):
    """decoder.* (g, v) pairs -> one MLX-layout conv weight each; the encoder is dropped."""
    weights = mx.load(str(src))
    out = {}
    for key, value in weights.items():
        if not key.startswith("decoder.") or key.endswith("weight_g"):
            continue
        key = key[len("decoder."):]
        if key.endswith("weight_v"):
            g = weights["decoder." + key[:-1] + "g"]
            w = g * value / mx.sqrt(mx.sum(value.astype(mx.float32) ** 2, axis=(1, 2), keepdims=True))
            # torch Conv1d [out, in, k] -> [out, k, in]; ConvTranspose1d [in, out, k] -> [out, k, in]
            out[key[:-2]] = w.transpose(1, 2, 0) if CONV_TRANSPOSE.match(key) else w.transpose(0, 2, 1)
        else:
            out[key] = value
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="m-a-p/YuE2-3B snapshot dir")
    ap.add_argument("--vae", type=Path, required=True, help="m-a-p/YuE2-Vae snapshot dir")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--bits", type=int, default=8, choices=[4, 5, 6, 8, 16], help="AR path; 16 = bf16, nothing quantized")
    ap.add_argument("--nar-bits", type=int, default=8, choices=[4, 8], help="nar_* path (flow matching)")
    ap.add_argument("--group-size", type=int, default=64, choices=[32, 64, 128])
    args = ap.parse_args()
    if args.dst.exists():
        raise SystemExit(f"{args.dst} exists")
    args.dst.mkdir(parents=True)

    cfg = json.loads((args.model / "config.json").read_text())
    weights = backbone(args.model / "model.safetensors", args.bits, args.nar_bits, args.group_size)
    if args.bits != 16:
        cfg["quantization"] = {"group_size": args.group_size, "bits": args.bits, "nar_bits": args.nar_bits, "mode": "affine"}
    mx.save_safetensors(str(args.dst / "model.safetensors"), weights, metadata={"format": "mlx"})
    (args.dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    mx.save_safetensors(str(args.dst / "vae.safetensors"), decoder(args.vae / "model.safetensors"), metadata={"format": "mlx"})
    shutil.copy2(args.vae / "config.json", args.dst / "vae_config.json")
    for name in ("qwen.tiktoken", "yue2_generation_config.json"):
        shutil.copy2(args.model / name, args.dst / name)

    if args.bits == 16:
        title, tag, desc = "bf16", "bf16", "the backbone in bf16 as released"
    else:
        mix = "" if args.nar_bits == args.bits else f", {args.nar_bits}-bit on the flow-matching (NAR) path"
        title, tag = f"{args.bits}-bit", f"{args.bits}bit"
        desc = f"{args.bits}-bit affine (group size {args.group_size}) on the backbone{mix}"
    (args.dst / "README.md").write_text(README.format(title=title, tag=tag, weights=desc))
    print(f"[done] {args.dst} ({len(weights)} backbone tensors)")


if __name__ == "__main__":
    main()
