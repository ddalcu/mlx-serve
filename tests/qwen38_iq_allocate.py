#!/usr/bin/env python3
"""Per-weight error measurement + greedy bit allocation for the Qwen3.8-27B iQ-MLX pack.

Two subcommands, deliberately split so the expensive half runs once:

  measure   For every quantizable weight, quantize a random sample of its output
            rows at each candidate (bits, group_size) with `weighted_affine_quant`
            (tests/dsv4_imatrix.py — the measured, mx.quantize-byte-compatible
            implementation) and record the imatrix-weighted reconstruction error,
            scaled back to the full tensor. Rows are sampled because the metric is
            a mean over output rows and concentrates hard; the CHOSEN setting is
            applied to the full tensor by the converter, never by this file.

  allocate  Spend a byte budget greedily: start every free weight at the floor and
            repeatedly buy the upgrade with the best error-reduction-per-byte until
            the budget runs out. Re-runnable at a new budget/floor in seconds — the
            phase-4 loop is "battery fails -> raise the floor -> allocate again",
            and it must not mean re-measuring.

The objective summed across weights is ABSOLUTE weighted error, sum_j om_j (dW_j)^2
over the tensor, which is proportional to the expected output-error energy that
weight contributes per token. It is not corrected for depth: a perturbation early in
the residual stream and one late in it are treated alike (the standard
llama.cpp/AWQ approximation).

  python3 tests/qwen38_iq_allocate.py measure  --imatrix im.safetensors --out err.json
  python3 tests/qwen38_iq_allocate.py allocate --errors err.json --budget-gb 9.0 --out alloc.json
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv4_imatrix import weighted_affine_quant  # noqa: E402
from convert_dsv4_weights import bf16_to_f32  # noqa: E402

DEFAULT_SRC = "/Volumes/G Drive SSD/models-src/Qwen3.8-27B"
GROUP_SIZES = (64, 128)
BIT_WIDTHS = (2, 3, 4)
# Ordered cheapest-first; the greedy walk only ever steps forward through this.
CANDIDATES = tuple(sorted(
    ((b, g) for b in BIT_WIDTHS for g in GROUP_SIZES),
    key=lambda bg: bg[0] + 32.0 / bg[1]))


def cand_key(bits, gs):
    return f"{bits}x{gs}"


def bytes_for(params, bits, gs):
    """Exactly what the converter writes: packed weights + bf16 scales + bf16 biases."""
    return params * (bits * 8 + 256 // gs) // 64


# ============================================================
# Weight inventory
# ============================================================

def classify(name):
    """(class, quantizable, pinned) for a SOURCE weight name.

    Pinned = never below 4-bit/gs-64: attention projections and the GDN a/b gates
    are 1.2 GB of cheap insurance, and the MTP head is the 3.3x decode lever whose
    fast paths (verifyQmmLane, the NAX m16 profile) are 4-bit/gs-64 specialisations.
    The vision tower is dropped from this build entirely."""
    if name.startswith("model.visual."):
        return "vision", False, False
    if not name.endswith(".weight"):
        return "dense", False, False
    if name.startswith("mtp."):
        if name == "mtp.fc.weight":
            return "mtp_fc_dense", False, False    # dense in every shipping qwen MTP build
        return "mtp", True, True
    if "embed_tokens" in name:
        return "embed", True, False
    if name == "lm_head.weight":
        return "lm_head", True, False
    if ".self_attn." in name:
        if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")):
            return "attn", True, True
        return "dense", False, False               # q_norm / k_norm
    if ".linear_attn." in name:
        if name.endswith(("in_proj_a.weight", "in_proj_b.weight")):
            return "gdn_ab", True, True
        if name.endswith("in_proj_qkv.weight"):
            return "gdn_qkv", True, False
        if name.endswith("in_proj_z.weight"):
            return "gdn_z", True, False
        if name.endswith("out_proj.weight"):
            return "gdn_out", True, False
        return "dense", False, False               # conv1d / norm / A_log / dt_bias
    if name.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
        return "mlp_gate_up", True, False
    if name.endswith("mlp.down_proj.weight"):
        return "mlp_down", True, False
    return "dense", False, False


def read_headers(src):
    """{name: (shard, dtype, shape, byte_offset_in_data, nbytes)} over every shard."""
    src = Path(src)
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for shard in sorted(set(weight_map.values())):
        path = src / shard
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        data_off = 8 + hlen
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            out[name] = (str(path), meta["dtype"], tuple(meta["shape"]),
                         data_off + begin, end - begin)
    return out


def read_rows(entry, row_idx):
    """Read only the sampled OUTPUT rows of a 2-D bf16 tensor, as f32 [len(row_idx), in]."""
    path, dtype, shape, off, _ = entry
    assert dtype == "BF16" and len(shape) == 2, f"{path}: unexpected {dtype} {shape}"
    in_dim = shape[1]
    row_bytes = in_dim * 2
    buf = np.empty((len(row_idx), in_dim), dtype=np.uint16)
    with open(path, "rb") as f:
        for i, r in enumerate(sorted(row_idx)):
            f.seek(off + r * row_bytes)
            raw = f.read(row_bytes)
            assert len(raw) == row_bytes, f"{path}: short row read at {r}"
            buf[i] = np.frombuffer(raw, dtype=np.uint16)
    return bf16_to_f32(buf)


# ============================================================
# measure
# ============================================================

_JOB = {}


def _init_worker(headers, imatrix_path, rows, seed):
    _JOB["headers"] = headers
    _JOB["rows"] = rows
    _JOB["seed"] = seed
    if imatrix_path:
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        _JOB["im"] = {k: np.array(v, copy=True) for k, v in mx.load(imatrix_path).items()}
    else:
        _JOB["im"] = {}


def _measure_one(name):
    entry = _JOB["headers"][name]
    _, _, shape, _, _ = entry
    out_dim, in_dim = shape
    n_rows = min(_JOB["rows"], out_dim)
    rng = random.Random(hash((_JOB["seed"], name)) & 0xFFFFFFFF)
    row_idx = rng.sample(range(out_dim), n_rows)
    w = read_rows(entry, row_idx)

    im = _JOB["im"].get(name)
    if im is None:
        # No calibration entry (the embedding table is gather-read, so no
        # activation flows through it as a matmul input). Uniform weights make
        # the search MLX's own minmax plus a least-squares refit — still a fair
        # error measurement, just an uncalibrated one.
        ch = np.ones(in_dim, dtype=np.float32)
        calibrated = False
    else:
        assert im.shape == (in_dim,), f"{name}: imatrix {im.shape} != ({in_dim},)"
        ch = im.astype(np.float32)
        calibrated = True

    errs = {}
    for bits, gs in CANDIDATES:
        if in_dim % gs != 0:
            continue
        _, stats = weighted_affine_quant(w, bits, gs, ch, return_stats=True)
        errs[cand_key(bits, gs)] = stats["weighted_err"] * (out_dim / n_rows)
    return name, {"shape": [out_dim, in_dim], "params": out_dim * in_dim,
                  "calibrated": calibrated, "err": errs}


def cmd_measure(args):
    headers = read_headers(args.src)
    todo, dense_bytes, skipped = [], 0, {}
    for name, (_, dtype, shape, _, nbytes) in sorted(headers.items()):
        cls, quantizable, _ = classify(name)
        if cls == "vision":
            continue                                  # dropped from this build
        if quantizable and len(shape) == 2 and shape[1] % 64 == 0:
            todo.append(name)
        else:
            dense_bytes += nbytes
            skipped[cls] = skipped.get(cls, 0) + nbytes

    print(f"{len(todo)} quantizable weights, dense residue {dense_bytes/1e9:.3f} GB "
          f"({ {k: round(v/1e9, 3) for k, v in skipped.items()} })", flush=True)

    t0 = time.time()
    results = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(args.jobs, initializer=_init_worker,
                  initargs=(headers, args.imatrix, args.rows, args.seed)) as pool:
        for i, (name, rec) in enumerate(pool.imap_unordered(_measure_one, todo, chunksize=1), 1):
            results[name] = rec
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)

    uncal = [n for n, r in results.items() if not r["calibrated"]]
    out = {
        "src": args.src,
        "imatrix": args.imatrix,
        "sample_rows": args.rows,
        "dense_bytes": dense_bytes,
        "uncalibrated": uncal,
        "weights": results,
    }
    Path(os.path.expanduser(args.out)).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out} — {len(results)} weights, {len(uncal)} uncalibrated, "
          f"{time.time()-t0:.0f}s", flush=True)


# ============================================================
# allocate
# ============================================================

def cmd_allocate(args):
    data = json.loads(Path(os.path.expanduser(args.errors)).read_text())
    weights = data["weights"]
    budget = int(args.budget_gb * 1e9)
    floor = (args.floor_bits, args.floor_group)
    pin = (4, 64)

    state, order = {}, {}
    spent = data["dense_bytes"]
    for name, rec in weights.items():
        cls, _, pinned = classify(name)
        avail = [c for c in CANDIDATES if cand_key(*c) in rec["err"]]
        assert avail, f"{name}: no candidate fits its geometry"
        if pinned:
            start = pin if pin in avail else avail[-1]
            steps = []
        else:
            start = floor if floor in avail else avail[0]
            # Only forward steps that actually reduce error are purchasable: a
            # wider weight that measures WORSE is not an upgrade at any price.
            steps = [c for c in avail
                     if bytes_for(rec["params"], *c) > bytes_for(rec["params"], *start)
                     and rec["err"][cand_key(*c)] < rec["err"][cand_key(*start)]]
            steps.sort(key=lambda c: bytes_for(rec["params"], *c))
        state[name] = start
        order[name] = steps
        spent += bytes_for(rec["params"], *start)

    print(f"floor {floor[0]}b/gs{floor[1]}: {spent/1e9:.3f} GB, budget {budget/1e9:.3f} GB",
          flush=True)
    if spent > budget:
        raise SystemExit(f"floor alone exceeds the budget by {(spent-budget)/1e9:.3f} GB")

    def best_step():
        best = None
        for name, steps in order.items():
            rec = weights[name]
            cur = state[name]
            cur_b = bytes_for(rec["params"], *cur)
            cur_e = rec["err"][cand_key(*cur)]
            for c in steps:
                nb = bytes_for(rec["params"], *c)
                if nb <= cur_b:
                    continue
                de = cur_e - rec["err"][cand_key(*c)]
                if de <= 0:
                    continue
                if nb - cur_b > budget - spent:
                    continue
                gain = de / (nb - cur_b)
                if best is None or gain > best[0]:
                    best = (gain, name, c, nb - cur_b)
        return best

    bought = 0
    while True:
        step = best_step()
        if step is None:
            break
        _, name, cand, cost = step
        state[name] = cand
        spent += cost
        bought += 1
        if bought % 50 == 0:
            print(f"  {bought} upgrades, {spent/1e9:.3f} GB", flush=True)

    per_class, total_err = {}, 0.0
    for name, cand in state.items():
        rec = weights[name]
        cls, _, _ = classify(name)
        d = per_class.setdefault(cls, {"params": 0, "bytes": 0, "widths": {}})
        d["params"] += rec["params"]
        d["bytes"] += bytes_for(rec["params"], *cand)
        d["widths"][cand_key(*cand)] = d["widths"].get(cand_key(*cand), 0) + 1
        total_err += rec["err"][cand_key(*cand)]

    alloc = {
        "errors_from": args.errors,
        "budget_gb": args.budget_gb,
        "floor": list(floor),
        "predicted_bytes": spent,
        "dense_bytes": data["dense_bytes"],
        "upgrades": bought,
        "objective": total_err,
        "per_class": per_class,
        "allocation": {n: {"bits": c[0], "group_size": c[1]} for n, c in sorted(state.items())},
    }
    Path(os.path.expanduser(args.out)).write_text(json.dumps(alloc, indent=1))

    print(f"\npredicted pack {spent/1e9:.3f} GB after {bought} upgrades "
          f"(objective {total_err:.4g})")
    print(f"{'class':14s} {'params':>10s} {'GB':>7s}  widths")
    for cls in sorted(per_class, key=lambda c: -per_class[c]["bytes"]):
        d = per_class[cls]
        print(f"{cls:14s} {d['params']/1e9:9.3f}B {d['bytes']/1e9:7.3f}  "
              + " ".join(f"{k}:{v}" for k, v in sorted(d["widths"].items())))
    print(f"{'dense':14s} {'':10s} {data['dense_bytes']/1e9:7.3f}")
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure")
    m.add_argument("--src", default=DEFAULT_SRC)
    m.add_argument("--imatrix", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--rows", type=int, default=512)
    m.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    m.add_argument("--seed", type=int, default=20260814)
    m.set_defaults(fn=cmd_measure)

    a = sub.add_parser("allocate")
    a.add_argument("--errors", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--budget-gb", type=float, default=9.0)
    a.add_argument("--floor-bits", type=int, default=2)
    a.add_argument("--floor-group", type=int, default=128)
    a.set_defaults(fn=cmd_allocate)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
