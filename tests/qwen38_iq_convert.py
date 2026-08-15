#!/usr/bin/env python3
"""Qwen/Qwen3.8-27B bf16 -> imatrix-calibrated mixed-width MLX affine, in the
layout mlx-serve loads. Text-only: the vision tower is dropped.

Per-weight (bits, group_size) come from the allocation table
(tests/qwen38_iq_allocate.py), not from a single `bits` argument. A weight with
an imatrix entry is quantized by `weighted_affine_quant`; anything the corpus
never exercised as a matmul input (the embedding table, the MTP head) falls back
to plain `mx.quantize`, and both are logged per weight.

Renames to the mlx-community qwen3_5 nesting, which is what mlx-serve's
`resolveWeightPrefix` (NESTED_PREFIX) actually looks up:
    model.language_model.*  -> language_model.model.*
    lm_head.*               -> language_model.lm_head.*
    mtp.*                   -> language_model.mtp.*
    model.visual.*          -> DROPPED (text-only build)

Load-bearing transforms carried over from the 4-bit/8-bit converter:
  - the delta-encoded norms get their `+1` folded in (decoder norms, final norm,
    q/k_norm — NOT linear_attn.norm, and NOT the mtp.* head, whose loader folds
    it itself; double-shifting either breaks the model)
  - depthwise conv1d ships HF's [C, 1, K] and MLX conv1d reads [C, K, 1]

The single `quantization` block declares 4-bit/gs-64 — the pack's fast-path
tier, and what `mtpNaxProfileEnabledForTrunk` gates on — with a per-tensor
override map beside it in the house style. mlx-serve itself never reads either
for an affine weight: `computeQuantParams` solves (bits, group_size) from the
packed geometry at every call site that has an input-dim hint, and `--verify`
below re-solves exactly that way and asserts it matches the allocation.

  python3 tests/qwen38_iq_convert.py --src <bf16> --dst <out> \
      --alloc alloc.json --imatrix imatrix.safetensors --verify
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dsv4_weights import (bf16_to_f32, f32_to_bf16_u16,  # noqa: E402
                                  mlx_affine_quant, write_safetensors_raw)
from dsv4_imatrix import weighted_affine_quant  # noqa: E402
from qwen38_iq_allocate import read_headers  # noqa: E402

SHARD_BYTES = 5 * 1024**3
COPY_FILES = ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "chat_template.jinja", "LICENSE")

# Qwen3.8 ships ZERO-CENTERED (delta-encoded) RMSNorm weights on the decoder
# layers, the final norm and q/k_norm: the reference layer computes `1 + w`.
# mlx-serve's runtime is `rmsnorm(x) * w`, so the +1 is baked in here. Same list
# mlx-lm shifts, and the same exclusions: linear_attn.norm and the vision tower
# are not delta-encoded. `mtp.*` is left raw on purpose — mlx-serve's MTP loader
# detects and folds it itself, and double-shifting breaks drafting.
NORM_SHIFT_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)


def rename(k):
    if k.startswith("model.language_model."):
        return "language_model.model." + k[len("model.language_model."):]
    if k.startswith("mtp."):
        return "language_model.mtp." + k[len("mtp."):]
    if k == "lm_head.weight":
        return "language_model.lm_head.weight"
    return k


def needs_norm_shift(name, ndim):
    if ndim != 1 or ".mtp." in name:
        return False
    if name == "language_model.model.norm.weight":
        return True
    return name.startswith("language_model.model.layers.") and name.endswith(NORM_SHIFT_SUFFIXES)


def read_tensor_f32(entry):
    """Whole 2-D bf16 tensor as f32 [out, in]."""
    path, dtype, shape, off, nbytes = entry
    assert dtype == "BF16", f"{path}: unexpected dtype {dtype}"
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(nbytes)
    assert len(raw) == nbytes, f"{path}: short read"
    return bf16_to_f32(np.frombuffer(raw, dtype=np.uint16).reshape(shape))


def read_tensor_raw(entry):
    """Whole tensor in its raw container dtype (bf16 stays uint16)."""
    path, dtype, shape, off, nbytes = entry
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(nbytes)
    assert len(raw) == nbytes, f"{path}: short read"
    np_dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
             "I32": np.int32, "I64": np.int64, "U8": np.uint8}[dtype]
    return np.frombuffer(raw, dtype=np_dt).reshape(shape), dtype


# ============================================================
# Workers
# ============================================================

_JOB = {}


def _init_worker(headers, alloc, imatrix_path):
    _JOB["headers"] = headers
    _JOB["alloc"] = alloc
    if imatrix_path:
        from convert_dsv4_weights import ShardReader
        r = ShardReader(imatrix_path)
        _JOB["im"] = {n: r.read(n)[0] for n in r.names()}
    else:
        _JOB["im"] = {}


def _quantize_one(name):
    entry = _JOB["headers"][name]
    spec = _JOB["alloc"][name]
    bits, gs = spec["bits"], spec["group_size"]
    w = read_tensor_f32(entry)
    ch = _JOB["im"].get(name)
    if ch is None:
        triples = mlx_affine_quant(w, bits, group_size=gs)
        calibrated = False
    else:
        assert ch.shape == (w.shape[1],), f"{name}: imatrix {ch.shape} vs in {w.shape[1]}"
        triples = weighted_affine_quant(w, bits, gs, ch.astype(np.float32))
        calibrated = True
    return name, bits, gs, calibrated, triples


# ============================================================
# Verification
# ============================================================

def solve_geometry(w_cols, s_cols, in_dim):
    """`transformer.affineParamsFromGeometry`, verbatim: bits and group_size come
    from the PACKED shape, which is the only thing mlx-serve reads for an affine
    weight. Returns None where the engine would return null."""
    if w_cols == 0 or s_cols == 0 or in_dim == 0:
        return None
    if (w_cols * 32) % in_dim != 0 or in_dim % s_cols != 0:
        return None
    bits = (w_cols * 32) // in_dim
    gs = in_dim // s_cols
    if bits not in (2, 3, 4, 5, 6, 8) or gs not in (32, 64, 128):
        return None
    return bits, gs


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--alloc", required=True)
    ap.add_argument("--imatrix", default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(os.path.expanduser(args.dst))
    dst.mkdir(parents=True, exist_ok=True)
    alloc = json.loads(Path(os.path.expanduser(args.alloc)).read_text())["allocation"]
    headers = read_headers(src)

    order = []
    for name in sorted(headers):
        if name.startswith("model.visual."):
            continue
        order.append((name, "q" if name in alloc else "d"))
    qnames = [n for n, k in order if k == "q"]
    print(f"{len(order)} tensors ({len(qnames)} quantized, "
          f"{len(order)-len(qnames)} dense), vision tower dropped", flush=True)

    out, out_bytes, out_idx, out_map, total = {}, 0, 0, {}, 0
    stats = {"calibrated": 0, "plain": 0, "shift": 0, "conv": 0, "widths": {}}
    verify_fail = []
    t0 = time.time()

    def flush():
        nonlocal out, out_bytes, out_idx, total
        if not out:
            return
        out_idx += 1
        fname = f"model-{out_idx:05d}.safetensors"
        write_safetensors_raw(str(dst / fname), out)
        for k in out:
            out_map[k] = fname
        total += out_bytes
        print(f"  wrote {fname}  {out_bytes/1e9:.2f} GB  ({len(out)} tensors, "
              f"{time.time()-t0:.0f}s)", flush=True)
        out, out_bytes = {}, 0

    def emit(nk, triple):
        nonlocal out_bytes
        out[nk] = triple
        out_bytes += len(triple[2])

    ctx = mp.get_context("fork")
    with ctx.Pool(args.jobs, initializer=_init_worker,
                  initargs=(headers, alloc, args.imatrix)) as pool:
        qiter = pool.imap(_quantize_one, qnames, chunksize=1)
        for name, kind in order:
            nk = rename(name)
            if kind == "q":
                got, bits, gs, calibrated, triples = next(qiter)
                assert got == name, f"pool order drifted: {got} != {name}"
                base = nk[:-len(".weight")]
                emit(base + ".weight", triples[0])
                emit(base + ".scales", triples[1])
                emit(base + ".biases", triples[2])
                stats["calibrated" if calibrated else "plain"] += 1
                key = f"{bits}x{gs}"
                stats["widths"][key] = stats["widths"].get(key, 0) + 1
                if args.verify:
                    in_dim = headers[name][2][1]
                    solved = solve_geometry(triples[0][1][-1], triples[1][1][-1], in_dim)
                    if solved != (bits, gs):
                        verify_fail.append((nk, solved, (bits, gs)))
            else:
                arr, dt = read_tensor_raw(headers[name])
                if nk.endswith("conv1d.weight") and arr.ndim == 3:
                    # HF ships depthwise conv as [C, 1, K]; MLX conv1d reads
                    # [C_out, K, C_in/groups] and mlx-serve consumes it as-is
                    # (it transposes contracted 2-D weights, never a conv).
                    arr = np.ascontiguousarray(np.swapaxes(arr, 1, 2))
                    stats["conv"] += 1
                if needs_norm_shift(nk, arr.ndim):
                    assert dt == "BF16", f"{nk}: norm shift on {dt}"
                    arr = f32_to_bf16_u16(bf16_to_f32(arr) + 1.0)
                    stats["shift"] += 1
                emit(nk, (dt, arr.shape, np.ascontiguousarray(arr).tobytes()))
            if out_bytes >= SHARD_BYTES:
                flush()
    flush()

    if verify_fail:
        for nk, solved, want in verify_fail[:10]:
            print(f"  VERIFY FAIL {nk}: geometry solves to {solved}, allocated {want}")
        raise SystemExit(f"{len(verify_fail)} weights would not resolve to their allocated width")

    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": out_map}, indent=2))

    cfg = json.loads((src / "config.json").read_text())
    # Text-only: `has_vision` is set from `vision_config` presence, so dropping
    # the block is what makes the pack honestly declare itself.
    cfg.pop("vision_config", None)
    qb = {"group_size": 64, "bits": 4, "mode": "affine"}
    for name, spec in sorted(alloc.items()):
        qb[rename(name)[:-len(".weight")]] = {
            "group_size": spec["group_size"], "bits": spec["bits"], "mode": "affine"}
    cfg["quantization"] = qb
    cfg["quantization_config"] = qb
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    for f in COPY_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)

    print(f"\ndone: {total/1e9:.2f} GB in {out_idx} shards, {time.time()-t0:.0f}s")
    print(f"  calibrated {stats['calibrated']}  plain mx.quantize {stats['plain']}  "
          f"norm+1 {stats['shift']}  conv1d transposed {stats['conv']}")
    print("  widths: " + " ".join(f"{k}:{v}" for k, v in sorted(stats["widths"].items())))
    if args.verify:
        print(f"  verify: all {len(qnames)} quantized weights re-solve to their "
              f"allocated (bits, group_size) from packed geometry alone")


if __name__ == "__main__":
    sys.exit(main())
