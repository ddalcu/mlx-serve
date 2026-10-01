#!/usr/bin/env python3
"""Make a community oMLX "oQ" Qwen3.8-Flash-Next pack (e.g. mrmurphydotdev/Swift1.5-Qwen3.8-Flash-Next-oQ4e-mtp)
loadable by mlx-serve.

  python3 tests/convert_oq_flash_next.py --src <oq pack dir> --dst <new pack dir>

What the oQ layout does differently, and what this does about it:
  - `mtp.*` and `vision_tower.*` become `language_model.mtp.*` and `model.visual.*`.
  - The 128 `...ngram_embedding.shards.K.{weight,scales,biases}` tensors in the trunk become one
    `ngram_table.bin` plus the `ngram_table` config block (without it the load fails with MissingNgramTable).
  - Norm weights are stored zero-centered (applied as 1 + w); mlx-serve's pack has the +1 folded in
    (NORM_FOLD_SUFFIXES in convert_qwen38_flash_next.py), so the same norms get +1 here.
  - `text_config.rope_parameters.type` becomes `rope_type`.
Per-tensor bits and group sizes (the config's per-path `quantization` entries) need no conversion: the
loader solves them from tensor shapes.
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dsv4_weights import bf16_to_f32, f32_to_bf16_u16, write_safetensors_raw  # noqa: E402
from convert_qwen38_flash_next import (NgramTable, VISION_PREFIX, needs_norm_fold,  # noqa: E402
                                       read_header, read_raw, rename)

NGRAM_SHARD = re.compile(r"\.ngram_embedding\.shards\.(\d+)\.(weight|scales|biases)$")
NGRAM_WEIGHT_SCALE = re.compile(r"\.ngram_embedding\.weight_scale$")
PARTS = ("weight", "scales", "biases")
FOLDED_Q_NORM_MEAN = 0.9


def oq_name(k):
    if k.startswith("vision_tower."):
        return VISION_PREFIX + k[len("vision_tower."):]
    return rename(k)


def tensor(src, index, name):
    path = str(src / index[name])
    header, data_off = read_header(path)
    return header[name], read_raw(path, data_off, header[name])


def check_pack(src, index):
    q_norm = next((k for k in sorted(index) if k.endswith("self_attn.q_norm.weight")), None)
    if q_norm is not None:
        meta, arr = tensor(src, index, q_norm)
        if float(bf16_to_f32(arr).mean()) > FOLDED_Q_NORM_MEAN:
            raise SystemExit(f"{q_norm} already has the +1 folded in; a second fold would break the norms")
    for k in sorted(index):
        if NGRAM_WEIGHT_SCALE.search(k):
            meta, arr = tensor(src, index, k)
            if meta["dtype"] != "BF16" or not np.all(bf16_to_f32(arr) == 1.0):
                raise SystemExit(f"{k} is not 1.0; mlx-serve never reads an n-gram weight_scale")


def write_ngram_table(src, dst, index, cfg):
    shards = {p: {} for p in PARTS}
    for k in index:
        m = NGRAM_SHARD.search(k)
        if m:
            shards[m.group(2)][int(m.group(1))] = k
    n = len(shards["weight"])
    if n == 0 or any(sorted(shards[p]) != list(range(n)) for p in PARTS):
        raise SystemExit("the n-gram shards are missing or not a contiguous 0..N-1 run for every part")
    first = shards["weight"][0]
    entry = cfg["quantization"][first[:first.rindex(".weight")]]
    bits, gs = entry["bits"], entry["group_size"]
    triples = [[(m["dtype"], m["shape"], a.tobytes()) for m, a in (tensor(src, index, shards[p][i]) for p in PARTS)]
               for i in range(n)]
    dim = triples[0][1][1][1] * gs
    table_path = dst / "ngram_table.bin"
    if table_path.exists():
        table_path.unlink()
    table = NgramTable(str(table_path), sum(t[0][1][0] for t in triples), bits, gs, dim)
    row0 = 0
    for t in triples:
        row0 += table.write(row0, t)
    return {"file": "ngram_table.bin", "bits": bits, "group_size": gs}


def convert(src, dst):
    src, dst = Path(src), Path(dst)
    index_doc = json.loads((src / "model.safetensors.index.json").read_text())
    index = index_doc["weight_map"]
    check_pack(src, index)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    ngram_block = write_ngram_table(src, dst, index, cfg)
    weight_map, total = {}, 0
    for fn in sorted(set(index.values())):
        header, data_off = read_header(str(src / fn))
        out = {}
        for name, meta in header.items():
            if NGRAM_SHARD.search(name) or NGRAM_WEIGHT_SCALE.search(name):
                continue
            arr = read_raw(str(src / fn), data_off, meta)
            if needs_norm_fold(name):
                arr = f32_to_bf16_u16(bf16_to_f32(arr) + 1.0)
            out[oq_name(name)] = (meta["dtype"], meta["shape"], arr.tobytes())
        if not out:
            continue
        write_safetensors_raw(str(dst / fn), out)
        weight_map.update({k: fn for k in out})
        total += sum(len(t[2]) for t in out.values())
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    cfg["ngram_table"] = ngram_block
    rope = cfg.get("text_config", {}).get("rope_parameters")
    if rope is not None:
        rope["rope_type"] = rope.pop("type", rope.get("rope_type", "default"))
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for p in src.iterdir():
        if p.is_file() and p.suffix != ".safetensors" and p.name not in ("model.safetensors.index.json", "config.json"):
            shutil.copy2(p, dst / p.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the oQ pack directory")
    ap.add_argument("--dst", required=True, help="output directory for the mlx-serve pack")
    args = ap.parse_args()
    convert(os.path.expanduser(args.src), os.path.expanduser(args.dst))
    print("done:", args.dst)


if __name__ == "__main__":
    main()
