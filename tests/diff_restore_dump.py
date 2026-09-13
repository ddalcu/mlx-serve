#!/usr/bin/env python3
"""Diff two restore-dump safetensors files.

Usage:
  tests/diff_restore_dump.py a.safetensors b.safetensors
  tests/diff_restore_dump.py a.safetensors b.safetensors --only layers.0.qsa
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path


_DTYPE = {
    "BOOL": ("?", 1),
    "U8": ("B", 1),
    "U16": ("H", 2),
    "U32": ("I", 4),
    "U64": ("Q", 8),
    "I8": ("b", 1),
    "I16": ("h", 2),
    "I32": ("i", 4),
    "I64": ("q", 8),
    "F16": ("e", 2),
    "F32": ("f", 4),
    "F64": ("d", 8),
    "BF16": None,
}


def _load_st(path: Path) -> dict[str, object]:
    try:
        import mlx.core as mx  # type: ignore

        loaded = mx.load(str(path))
        out = {}
        for k, v in loaded.items():
            out[k] = v.astype(mx.float32).reshape(-1).tolist() if hasattr(v, "astype") else v
        return out
    except Exception:
        pass
    try:
        from safetensors.numpy import load_file  # type: ignore
        import numpy as np  # type: ignore

        loaded = load_file(str(path))
        return {k: np.asarray(v, dtype=np.float32).reshape(-1) for k, v in loaded.items()}
    except Exception:
        pass
    return _load_st_raw(path)


def _load_st_raw(path: Path) -> dict[str, list[float]]:
    data = path.read_bytes()
    n = struct.unpack_from("<Q", data, 0)[0]
    header = json.loads(data[8 : 8 + n].decode("utf-8"))
    body = data[8 + n :]
    out: dict[str, list[float]] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = spec["dtype"]
        shape = spec["shape"]
        start, stop = spec["data_offsets"]
        blob = body[start:stop]
        if dtype == "BF16":
            import array

            u16 = array.array("H")
            u16.frombytes(blob)
            out[name] = [_bf16_to_f32(x) for x in u16]
            continue
        fmt = _DTYPE.get(dtype)
        if fmt is None:
            out[name] = list(blob)
            continue
        code, width = fmt
        count = (stop - start) // width
        vals = struct.unpack_from("<" + code * count, blob, 0)
        out[name] = [float(x) for x in vals]
        _ = shape
    return out


def _bf16_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u << 16))[0]


def _as_list(v: object) -> list[float]:
    if hasattr(v, "reshape"):
        try:
            return [float(x) for x in v.reshape(-1).tolist()]
        except Exception:
            pass
    if isinstance(v, list):
        return [float(x) for x in v]
    return [float(v)]


def _first_diff(a: list[float], b: list[float]) -> tuple[int, float, float] | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, a[i], b[i]
    if len(a) != len(b):
        return n, float(len(a)), float(len(b))
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Diff two restore-dump safetensors files.")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--only", default="", help="only compare tensor names with this prefix")
    args = p.parse_args()
    pa, pb = Path(args.a), Path(args.b)
    ta, tb = _load_st(pa), _load_st(pb)
    names = sorted(set(ta) | set(tb))
    if args.only:
        names = [n for n in names if n.startswith(args.only)]
    diffs = 0
    for name in names:
        if name not in ta:
            print(f"{name}  missing in {pa.name}")
            diffs += 1
            continue
        if name not in tb:
            print(f"{name}  missing in {pb.name}")
            diffs += 1
            continue
        la, lb = _as_list(ta[name]), _as_list(tb[name])
        fd = _first_diff(la, lb)
        if fd is None:
            print(f"{name}  identical n={len(la)}")
            continue
        i, va, vb = fd
        max_abs = max(abs(x - y) for x, y in zip(la, lb)) if la and lb else float("inf")
        print(f"{name}  max_abs={max_abs:.6g} first_diff={i} a={va} b={vb} na={len(la)} nb={len(lb)}")
        diffs += 1
    print(f"{diffs} differing tensors / {len(names)} compared")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
