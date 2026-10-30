#!/usr/bin/env python3
"""Convert nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B to the MLX layout mlx-audio's `nemotron_transcribe` loads.

  python -I scripts/convert_nemotron_transcribe.py --source <download> --out <new dir> [--bits 8 --group-size 128]

Keys, transposes and the quantization scope follow mlx-audio (Blaizzy/mlx-audio#1087, `Model.sanitize`): only `llm.*`
linears, the embedding and the head are quantized; the perception encoders stay bf16, Mamba dynamics, the router and its
correction bias stay f32. The routed experts' `fc2` reads 1856 columns, which no 128-wide group divides, so it takes
`--fc2-group-size` (a per-path entry in the config's `quantization`), the MTP block's included. The MTP head
(`llm.mtp.*`) quantizes like the trunk; the CTC timestamp head (`ctc_timestamp.*`) keeps NVIDIA's names, layouts and f32,
with its SentencePiece blob (the transformer's `ffn.net.*` renamed as the encoders'); `alignment_tokenizer/` rides along. Needs mlx. An existing output dir is refused.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

SIDE_DIRS = ["alignment_tokenizer"]
SIDE_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json", "generation_config.json"]
KEEP_F32 = (".A_log", ".D", ".dt_bias", ".e_score_correction_bias", ".gate.weight")
KEEP_RAW = "ctc_timestamp."  # f32 weights, the SentencePiece blob and the integer scalars stay as NVIDIA wrote them
SHARD_BYTES = 4 << 30


def sanitize(key, value):
    """mlx-audio's key map; None drops the tensor."""
    if key.endswith("._extra_state"):
        return None
    for old, new in (
        ("llm.model.embed_tokens.", "llm.backbone.embeddings."),
        ("llm.model.norm.", "llm.backbone.norm_f."),
        ("llm.model.layers.", "llm.backbone.layers."),
        ("._fp32_params.", "."),
        (".ffn.net.0.", ".ffn.linear1."),
        (".ffn.net.3.", ".ffn.linear2."),
        (".preprocessor.featurizer.", ".preprocessor."),
    ):
        key = key.replace(old, new)
    if key.startswith(KEEP_RAW):
        return key, value
    if key.endswith(".experts.gate_and_up_projs"):
        key, value = key.replace(".experts.gate_and_up_projs", ".switch_mlp.fc1.weight"), value.transpose(0, 2, 1)
    elif key.endswith(".experts.down_projs"):
        key, value = key.replace(".experts.down_projs", ".switch_mlp.fc2.weight"), value.transpose(0, 2, 1)
    elif key.endswith(".conv1d.weight") and value.shape[1] == 1:
        value = value.transpose(0, 2, 1)
    elif key.endswith(".subpixel_upsample.weight") and value.shape[-1] == 3:
        value = value.transpose(0, 2, 1)
    return key, value


def quantizable(key, value):
    """The modules `nn.quantize` reaches in `llm.*`: linears, the embedding table, the head, the expert banks."""
    return (
        key.startswith("llm.")
        and key.endswith(".weight")
        and value.ndim >= 2
        and not key.endswith((".conv1d.weight", ".gate.weight"))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--fc2-group-size", type=int, default=64)
    ap.add_argument("--bf16", action="store_true", help="write no quantization (the reference pack)")
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists")
    args.out.mkdir(parents=True)

    config = json.loads((args.source / "config.json").read_text())
    shards, shard, size, index, overrides = [], {}, 0, {}, {}

    def flush():
        nonlocal shard, size
        if not shard:
            return
        name = f"model-{len(shards) + 1:05d}.safetensors"
        mx.save_safetensors(str(args.out / name), shard, metadata={"format": "mlx"})
        index.update({k: name for k in shard})
        shards.append(name)
        shard, size = {}, 0

    def put(key, value):
        nonlocal size
        mx.eval(value)
        shard[key] = value
        size += value.nbytes
        if size >= SHARD_BYTES:
            flush()

    source = mx.load(str(args.source / "model.safetensors"))
    for src in list(source):
        got = sanitize(src, source.pop(src))
        if got is None:
            continue
        key, value = got
        if key.startswith(KEEP_RAW):
            put(key, value)
            continue
        if key.startswith("llm.") and key.endswith(".gate.weight"):
            value = value.astype(mx.float32)
        elif value.dtype == mx.float32 and not key.endswith(KEEP_F32):
            value = value.astype(mx.bfloat16)
        if args.bf16 or not quantizable(key, value):
            put(key, value)
            continue
        gs = args.fc2_group_size if key.endswith(".switch_mlp.fc2.weight") else args.group_size
        w, scales, biases = mx.quantize(value, group_size=gs, bits=args.bits)
        base = key[: -len(".weight")]
        put(key, w)
        put(base + ".scales", scales)
        put(base + ".biases", biases)
        if gs != args.group_size:
            overrides[base] = {"group_size": gs, "bits": args.bits}
    flush()

    (args.out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}, indent=1))
    if not args.bf16:
        config["quantization"] = {"group_size": args.group_size, "bits": args.bits, "mode": "affine", **overrides}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    for name in SIDE_DIRS:
        if (args.source / name).is_dir():
            shutil.copytree(args.source / name, args.out / name)
    for name in SIDE_FILES:
        if (args.source / name).exists():
            shutil.copy2(args.source / name, args.out / name)
    # The inline template is the older one: it lacks the audio parts `chat_template.jinja` renders, and a file beside
    # the config loses to an inline entry.
    if (args.out / "chat_template.jinja").exists():
        tok = json.loads((args.out / "tokenizer_config.json").read_text())
        tok.pop("chat_template", None)
        (args.out / "tokenizer_config.json").write_text(json.dumps(tok, indent=2, ensure_ascii=False))
    print(f"wrote {len(index)} tensors in {len(shards)} shards to {args.out}")


if __name__ == "__main__":
    main()
