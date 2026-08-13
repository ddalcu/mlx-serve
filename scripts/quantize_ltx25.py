#!/usr/bin/env python3
"""Quantize the bf16 `mlx-community/ltx-2.5-mlx` pack into the q4 layout
mlx-serve's LTX engine consumes.

The 2.5 release ships bf16 only (38 GB per DiT variant, 23.8 GB text encoder).
Our loader classifies a tensor as quantized iff a sibling `<name>.scales`
exists, and `dQLin`/`gQLin` read affine group-64 weights — the same shape
`dgrauet/ltx-2.3-mlx-q4` has. So this reproduces that pack's recipe on 2.5:

  quantization: {bits: 4, group_size: 64, only_transformer_blocks: true}

Per DiT variant that is exactly 34 linears x 48 blocks = 1632 weights
(to_q/to_k/to_v/to_out/to_gate_logits on six attention modules, plus
proj_in/proj_out on the two feed-forwards). Everything else — adaLN tables,
patchify/proj_out, the connector, both VAEs, the vocoder and both upscalers —
stays at its shipped dtype, byte-for-byte.

The text encoder is a normal Gemma-4-unified checkpoint (`model.language_model`
prefix, which `model.resolveWeightPrefix` already accepts), so it is quantized
the way any of our 4-bit Gemma packs is: every 2-D projection whose contraction
dim divides the group size, plus the embedding table.

CPU-only by construction (`mx.set_default_device(mx.cpu)`) — this runs beside
whatever is using the GPU.
"""

import argparse
import json
import os
import shutil
import struct
import sys
import time

import mlx.core as mx

mx.set_default_device(mx.cpu)

# The 34 per-block linear module names. Anything else under transformer_blocks
# (scale_shift tables, q_norm/k_norm) keeps its shipped dtype.
DIT_LINEAR_LEAVES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "to_gate_logits",
    "proj_in",
    "proj_out",
)

# Components 2.5 re-ships byte-identical to 2.3 (the converter verified this),
# plus the two configs the engine reads. Copied, never re-encoded.
PASSTHROUGH = (
    "connector.safetensors",
    "vae_decoder.safetensors",
    "vae_encoder.safetensors",
    "audio_vae.safetensors",
    "vocoder.safetensors",
    "spatial_upscaler_x2_v1_1.safetensors",
    "temporal_upscaler_x2_v1_0.safetensors",
    "config.json",
    "embedded_config.json",
    "spatial_upscaler_x2_v1_1_config.json",
    "temporal_upscaler_x2_v1_0_config.json",
    # The LTX-2.x Community License and the Acceptable Use Policy it
    # incorporates BY REFERENCE both travel with a derivative — shipping the
    # license without the AUP leaves a dangling reference. The README is
    # deliberately NOT copied: upstream's describes upstream's repo.
    "LICENSE.md",
    "ltx-acceptable-use-policy-snapshot-2026-08-12.pdf",
)

TE_SUBDIR = "gemma4-12b-ltx-v1"
TE_PASSTHROUGH = (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "processor_config.json",
    "chat_template.jinja",
)


def header_metadata(path):
    """The `__metadata__` block of a safetensors file (upstream ships the LTX
    license text in there; the conversion notes say it is preserved)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__")


def is_dit_target(key):
    if ".transformer_blocks." not in key or not key.endswith(".weight"):
        return False
    return key[: -len(".weight")].rsplit(".", 1)[-1] in DIT_LINEAR_LEAVES


def is_te_target(key, arr, group_size):
    if not key.endswith(".weight") or arr.ndim != 2:
        return False
    if arr.shape[-1] % group_size:
        return False
    leaf = key[: -len(".weight")].rsplit(".", 1)[-1]
    if leaf.endswith("_norm") or leaf.endswith("layernorm"):
        return False
    return True


def quantize_file(src, dst, select, bits, group_size, label):
    if os.path.exists(dst):
        print(f"  {label}: exists, skipping", flush=True)
        return
    t0 = time.time()
    weights = mx.load(src)
    meta = header_metadata(src)
    out, n_q = {}, 0
    for key in sorted(weights):
        w = weights[key]
        if select(key, w):
            wq, sc, bi = mx.quantize(w, group_size=group_size, bits=bits)
            mx.eval(wq, sc, bi)
            out[key] = wq
            out[key[: -len(".weight")] + ".scales"] = sc
            out[key[: -len(".weight")] + ".biases"] = bi
            n_q += 1
            if n_q % 128 == 0:
                mx.clear_cache()
                print(f"    {label}: {n_q} quantized", flush=True)
        else:
            out[key] = w
    del weights
    # mlx_save_safetensors silently APPENDS ".safetensors" to a path that
    # lacks it, so a plain ".partial" suffix lands beside the file it meant
    # to be and the rename then fails on a name nothing wrote.
    tmp = dst + ".partial.safetensors"
    mx.save_safetensors(tmp, out, metadata=meta or {})
    os.replace(tmp, dst)
    del out
    mx.clear_cache()
    gb = os.path.getsize(dst) / 1e9
    print(
        f"  {label}: {n_q} weights quantized -> {gb:.2f} GB ({time.time()-t0:.0f}s)",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    src, dst = args.src, args.dst
    os.makedirs(dst, exist_ok=True)

    for name in PASSTHROUGH:
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if not os.path.exists(s):
            print(f"  skip missing {name}", flush=True)
            continue
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
            continue
        print(f"  copy {name} ({os.path.getsize(s)/1e9:.2f} GB)", flush=True)
        shutil.copyfile(s, d)

    for variant in ("distilled", "dev"):
        name = f"transformer-{variant}.safetensors"
        s = os.path.join(src, name)
        if not os.path.exists(s):
            print(f"  skip missing {name}", flush=True)
            continue
        quantize_file(
            s,
            os.path.join(dst, name),
            lambda k, w: is_dit_target(k),
            args.bits,
            args.group_size,
            name,
        )

    # ── Text encoder ──
    te_src = os.path.join(src, TE_SUBDIR)
    te_dst = os.path.join(dst, TE_SUBDIR)
    os.makedirs(te_dst, exist_ok=True)
    for name in TE_PASSTHROUGH:
        s = os.path.join(te_src, name)
        if os.path.exists(s) and not os.path.exists(os.path.join(te_dst, name)):
            shutil.copyfile(s, os.path.join(te_dst, name))
    quantize_file(
        os.path.join(te_src, "model.safetensors"),
        os.path.join(te_dst, "model.safetensors"),
        lambda k, w: is_te_target(k, w, args.group_size),
        args.bits,
        args.group_size,
        "text encoder",
    )
    te_cfg = json.load(open(os.path.join(te_src, "config.json")))
    te_cfg["quantization"] = {"group_size": args.group_size, "bits": args.bits}
    json.dump(te_cfg, open(os.path.join(te_dst, "config.json"), "w"), indent=2)

    # ── Pack manifests (same shape as the 2.3 pack's) ──
    json.dump(
        {
            "quantization": {
                "bits": args.bits,
                "group_size": args.group_size,
                "only_transformer_blocks": True,
            }
        },
        open(os.path.join(dst, "quantize_config.json"), "w"),
        indent=2,
    )
    json.dump(
        {
            "format": "split",
            "model_version": "2.5.0",
            "components": [
                "connector",
                "vae_decoder",
                "vae_encoder",
                "audio_vae",
                "vocoder",
                "spatial_upscaler_x2_v1_1",
                "temporal_upscaler_x2_v1_0",
            ],
            "transformer_variants": ["distilled", "dev"],
            "text_encoder": TE_SUBDIR,
            "source": "mlx-community/ltx-2.5-mlx",
            "quantized": True,
            "quantization_bits": args.bits,
        },
        open(os.path.join(dst, "split_model.json"), "w"),
        indent=2,
    )
    print("done:", dst, flush=True)


if __name__ == "__main__":
    sys.exit(main())
