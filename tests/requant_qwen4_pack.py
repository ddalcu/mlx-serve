#!/usr/bin/env python3
"""Re-pack an mlx-serve qwen4_exp pack to a different non-expert width without
re-downloading: every affine tensor whose geometry solves to `--from-bits` is
dequantized and re-quantized at `--to-bits` (group 64). Routed experts
(`.switch_mlp.`), the n-gram table and everything bf16 are copied through
(the table is hard-linked, it is 32 GB).

  python3 tests/requant_qwen4_pack.py --src <pack> --dst <pack-4bit> [--to-bits 4]
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dsv4_weights import write_safetensors_raw  # noqa: E402
from convert_qwen38_flash_next import read_header, read_raw  # noqa: E402

GS = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--from-bits", type=int, default=8)
    ap.add_argument("--to-bits", type=int, default=4)
    ap.add_argument("--keep", default="lm_head,embed_tokens",
                    help="comma list of name substrings left at the source width")
    a = ap.parse_args()
    src, dst = Path(os.path.expanduser(a.src)), Path(os.path.expanduser(a.dst))
    dst.mkdir(parents=True, exist_ok=True)
    keep = [k for k in a.keep.split(",") if k]
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    files = sorted(set(idx["weight_map"].values()))
    out_map, total, n_req, saved = {}, 0, 0, 0
    for f in files:
        hdr, off = read_header(src / f)
        out = {}
        names = set(hdr)
        for name, meta in hdr.items():
            if name.endswith(".scales") or name.endswith(".biases"):
                continue
            arr = read_raw(src / f, off, meta)
            base = name[:-len(".weight")] if name.endswith(".weight") else name
            is_q = meta["dtype"] == "U32" and base + ".scales" in names
            requant = False
            if is_q and ".switch_mlp." not in base and not any(k in base for k in keep):
                sc = read_raw(src / f, off, hdr[base + ".scales"])
                in_dim = sc.shape[-1] * GS
                bits = arr.shape[-1] * 32 // in_dim
                requant = bits == a.from_bits
            if requant:
                sc = read_raw(src / f, off, hdr[base + ".scales"])
                bi = read_raw(src / f, off, hdr[base + ".biases"])
                w = mx.dequantize(mx.array(arr), mx.array(sc).view(mx.bfloat16), mx.array(bi).view(mx.bfloat16),
                                  group_size=GS, bits=a.from_bits)
                wq, s2, b2 = mx.quantize(w, group_size=GS, bits=a.to_bits)
                mx.eval(wq, s2, b2)
                out[base + ".weight"] = ("U32", wq.shape, np.array(wq, copy=False).tobytes())
                out[base + ".scales"] = ("BF16", s2.shape, np.array(s2.view(mx.uint16), copy=False).tobytes())
                out[base + ".biases"] = ("BF16", b2.shape, np.array(b2.view(mx.uint16), copy=False).tobytes())
                saved += arr.nbytes - out[base + ".weight"][2].__len__()
                n_req += 1
            else:
                out[name] = (meta["dtype"], arr.shape, np.ascontiguousarray(arr).tobytes())
                if is_q:
                    for sfx in (".scales", ".biases"):
                        m2 = hdr[base + sfx]
                        out[base + sfx] = (m2["dtype"], m2["shape"], np.ascontiguousarray(read_raw(src / f, off, m2)).tobytes())
        write_safetensors_raw(str(dst / f), out)
        for k in out:
            out_map[k] = f
        total += sum(len(v[2]) for v in out.values())
        print(f"  {f}: {len(out)} tensors", flush=True)
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": out_map}, indent=2))
    for f in src.iterdir():
        if f.name.endswith(".safetensors") or f.name.startswith(".") or f.name == "model.safetensors.index.json":
            continue
        if f.name == "ngram_table.bin":
            try:
                os.link(f, dst / f.name)
            except OSError:
                shutil.copy2(f, dst / f.name)
        elif f.is_file():
            shutil.copy2(f, dst / f.name)
    cfg = json.loads((dst / "config.json").read_text())
    cfg["quantization"] = {"group_size": GS, "bits": a.to_bits, "mode": "affine"}
    cfg["quantization_config"] = cfg["quantization"]
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"done: {n_req} tensors {a.from_bits}->{a.to_bits}-bit, saved {saved/1e9:.1f} GB, trunk {total/1e9:.1f} GB")


if __name__ == "__main__":
    sys.exit(main())
