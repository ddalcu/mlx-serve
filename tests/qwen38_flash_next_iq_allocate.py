#!/usr/bin/env python3
"""Per-layer expert width allocation for the Qwen3.8-Flash-Next iQ pack.

  measure   For every (layer, gate_up|down) group: quantize a random sample of
            experts at each candidate (bits, group_size) with the imatrix-weighted
            search (dsv4_imatrix.weighted_affine_quant) and record the weighted
            RELATIVE reconstruction error (error energy / signal energy, both
            imatrix-weighted). Experts are sampled because the metric is a mean
            over slabs; the converter applies the chosen width to the whole bank.
  allocate  Spend an expert byte budget greedily: every group starts at the floor,
            the upgrade with the best error-reduction-per-byte is bought until the
            budget runs out. The last `--tail-layers` layers are pinned at 4x64
            (a low-bit tail is the turn-level agent-loop trap). Seconds per run.

Output of `allocate` is the converter's `--alloc` file:
  {"layers.N.gate_up": {"bits": b, "group_size": g}, "layers.N.down": {...}, ...}

  venv/bin/python tests/qwen38_flash_next_iq_allocate.py measure --src <hf bf16> \
      --imatrix im.safetensors --out errors.json
  python3 tests/qwen38_flash_next_iq_allocate.py allocate --errors errors.json \
      --budget-gb 44 --out alloc.json
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dsv4_weights import bf16_to_f32  # noqa: E402
from dsv4_imatrix import weighted_affine_quant  # noqa: E402

CANDIDATES = ((2, 64), (2, 128), (3, 64), (3, 128), (4, 64))
PREFIX = "model.language_model.layers."


def ckey(bits, gs):
    return f"{bits}x{gs}"


def bytes_per_param(bits, gs):
    return (bits * 8 + 256 // gs) / 64.0   # packed bits + bf16 scale/bias per group, in bytes


def bank_memmap(src, index, name):
    path = Path(src) / index["weight_map"][name]
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    meta = hdr[name]
    assert meta["dtype"] == "BF16"
    b, _ = meta["data_offsets"]
    return np.memmap(path, dtype=np.uint16, mode="r", offset=8 + hlen + b, shape=tuple(meta["shape"]))


_G = {}


def _init(src, index, imatrix_path):
    from safetensors.numpy import load_file
    _G["src"], _G["index"] = src, index
    _G["im"] = load_file(imatrix_path)


def _measure(task):
    layer, role, experts = task
    name = f"{PREFIX}{layer}.mlp.experts.{'gate_up_proj' if role == 'gate_up' else 'down_proj'}"
    bank = bank_memmap(_G["src"], _G["index"], name)
    E, out_dim, in_dim = bank.shape
    ch_all = _G["im"][name]
    assert ch_all.shape == (E * in_dim,)
    cands = [c for c in CANDIDATES if in_dim % c[1] == 0]
    err = {ckey(*c): 0.0 for c in cands}
    for e in experts:
        w = bf16_to_f32(np.ascontiguousarray(bank[e]))
        ch = ch_all[e * in_dim:(e + 1) * in_dim]
        for bits, gs in cands:
            _, st = weighted_affine_quant(w, bits, gs, ch, return_stats=True)
            err[ckey(bits, gs)] += st["weighted_rel_err"] / len(experts)
    return f"layers.{layer}.{role}", {"err": err, "params": int(E * out_dim * in_dim)}


def cmd_measure(args):
    index = json.loads((Path(args.src) / "model.safetensors.index.json").read_text())
    cfg = json.loads((Path(args.src) / "config.json").read_text())
    n_layers = cfg.get("text_config", cfg)["num_hidden_layers"]
    n_experts = cfg.get("text_config", cfg)["num_experts"]
    rng = random.Random(args.seed)
    tasks = [(l, role, sorted(rng.sample(range(n_experts), args.experts)))
             for l in range(n_layers) for role in ("gate_up", "down")]
    out = {}
    with mp.get_context("fork").Pool(args.jobs, initializer=_init, initargs=(args.src, index, args.imatrix)) as pool:
        for i, (k, rec) in enumerate(pool.imap_unordered(_measure, tasks)):
            out[k] = rec
            print(f"[{i+1}/{len(tasks)}] {k} " + " ".join(f"{c}={v:.4f}" for c, v in rec["err"].items()), flush=True)
    Path(args.out).write_text(json.dumps({"experts_sampled": args.experts, "seed": args.seed, "groups": out}, indent=1))


def cmd_allocate(args):
    groups = json.loads(Path(args.errors).read_text())["groups"]
    n_layers = 1 + max(int(k.split(".")[1]) for k in groups)
    pinned = {k for k in groups if int(k.split(".")[1]) >= n_layers - args.tail_layers}
    def floor_for(k):
        if k.endswith(".down") and args.down_floor:
            b, g = args.down_floor.split("x")
            return int(b), int(g)
        return (2, 128) if ckey(2, 128) in groups[k]["err"] else (2, 64)

    state = {k: ((4, 64) if k in pinned else floor_for(k)) for k in groups}

    def cost(k, c):
        return groups[k]["params"] * bytes_per_param(*c)

    def err(k, c):
        return groups[k]["err"][ckey(*c)] * groups[k]["params"]   # error weighted by size

    spent = sum(cost(k, c) for k, c in state.items())
    budget = args.budget_gb * 1e9
    while True:
        best = None
        for k in groups:
            if k in pinned:
                continue
            cur = state[k]
            for c in CANDIDATES:
                if ckey(*c) not in groups[k]["err"]:
                    continue
                dc = cost(k, c) - cost(k, cur)
                if dc <= 0 or spent + dc > budget:
                    continue
                gain = (err(k, cur) - err(k, c)) / dc
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, k, c)
        if best is None:
            break
        _, k, c = best
        spent += cost(k, c) - cost(k, state[k])
        state[k] = c
    total_err = sum(err(k, c) for k, c in state.items()) / sum(g["params"] for g in groups.values())
    from collections import Counter
    print(f"spent {spent/1e9:.2f} GB of {args.budget_gb} GB, size-weighted rel err {total_err:.5f}")
    print("widths:", dict(Counter(ckey(*c) for c in state.values())))
    for l in range(n_layers):
        print(f"  layer {l:2d}: gate_up {ckey(*state[f'layers.{l}.gate_up'])}  down {ckey(*state[f'layers.{l}.down'])}")
    Path(args.out).write_text(json.dumps(
        {k: {"bits": c[0], "group_size": c[1]} for k, c in sorted(state.items())}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--src", required=True)
    m.add_argument("--imatrix", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--experts", type=int, default=24)
    m.add_argument("--seed", type=int, default=20260912)
    m.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    m.set_defaults(fn=cmd_measure)
    a = sub.add_parser("allocate")
    a.add_argument("--errors", required=True)
    a.add_argument("--budget-gb", type=float, required=True, help="expert bytes incl. the pinned tail")
    a.add_argument("--tail-layers", type=int, default=2)
    a.add_argument("--down-floor", default=None, help="e.g. 3x128: never below this on the down projections")
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_allocate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
