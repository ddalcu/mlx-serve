"""Build an MTP sidecar (mtp/weights.safetensors) from the ORIGINAL
Qwen/Qwen3.6-27B bf16 weights — no third-party intermediate.

Range-downloads only the mtp.* tensors (~840 MB of the 54 GB repo for the
dense 27B head) via HTTP Range requests against the safetensors shards,
quantizes the attention/MLP linears to INT4 group-32 with mlx (standard affine
layout, same geometry the published sidecars use), and keeps fc + norms bf16.
A MoE head (one `mlp.experts.N.*` triple per expert, e.g. Qwen3.5-122B-A10B)
is written as stacked `switch_mlp` banks at INT4 group-64, the layout the
trunk's own experts use.

Usage (any python with mlx + numpy + requests/urllib):
  python3 tests/build_mtp_sidecar.py [REPO] [OUT_DIR]
Defaults: Qwen/Qwen3.6-27B -> /tmp/mtp-sidecar-qwen-original/
"""

import json
import re
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.6-27B"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/mtp-sidecar-qwen-original")
BASE = f"https://huggingface.co/{REPO}/resolve/main"
QUANT_SUFFIXES = ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                  "o_proj.weight", "gate_proj.weight", "up_proj.weight",
                  "down_proj.weight")


def fetch(url: str, start: int | None = None, end: int | None = None, retries: int = 4) -> bytes:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            if start is not None:
                req.add_header("Range", f"bytes={start}-{end}")
            with urllib.request.urlopen(req) as r:
                data = r.read()
            if start is not None and len(data) != end - start + 1:
                raise OSError(f"short read: {len(data)} of {end - start + 1}")
            return data
        except Exception as err:  # noqa: BLE001 — retry CDN hiccups
            last_err = err
            print(f"    retry {attempt + 1} after: {err}", flush=True)
    raise last_err


def shard_header(shard: str):
    url = f"{BASE}/{shard}"
    n = struct.unpack("<Q", fetch(url, 0, 7))[0]
    hdr = json.loads(fetch(url, 8, 8 + n - 1))
    return url, 8 + n, hdr


def main() -> None:
    index = json.loads(fetch(f"{BASE}/model.safetensors.index.json").decode())
    weight_map = index["weight_map"]
    mtp_keys = sorted(k for k in weight_map if k.startswith("mtp."))
    assert mtp_keys, "no mtp.* tensors in this repo"
    headers = {shard: shard_header(shard) for shard in sorted({weight_map[k] for k in mtp_keys})}

    def get(key: str):
        url, data_base, hdr = headers[weight_map[key]]
        meta = hdr[key]
        assert meta["dtype"] == "BF16", (key, meta["dtype"])
        begin, end = meta["data_offsets"]
        raw = fetch(url, data_base + begin, data_base + end - 1)
        return key, np.frombuffer(raw, dtype=np.uint16).reshape(meta["shape"])

    out: dict[str, mx.array] = {}
    experts: dict[str, dict[int, np.ndarray]] = {}
    expert_key = re.compile(r"^(mtp\.layers\.\d+\.mlp)\.experts\.(\d+)\.(\w+_proj)\.weight$")
    with ThreadPoolExecutor(8) as pool:
        for key, u16 in pool.map(get, mtp_keys):
            m = expert_key.match(key)
            if m:
                experts.setdefault(f"{m[1]}.switch_mlp.{m[3]}", {})[int(m[2])] = u16
                continue
            print(f"  {key}  {list(u16.shape)}", flush=True)
            w = mx.array(u16).view(mx.bfloat16)
            if key.endswith(QUANT_SUFFIXES):
                wq, scales, biases = mx.quantize(w, group_size=32, bits=4)
                out[key] = wq
                out[key.replace(".weight", ".scales")] = scales.astype(mx.bfloat16)
                out[key.replace(".weight", ".biases")] = biases.astype(mx.bfloat16)
            elif len(u16.shape) == 1:
                # Qwen stores RMS-norm weights delta-encoded: the layer
                # computes (1 + w). MLX-converted checkpoints (and our
                # runtime) expect the +1 baked in — verified by diffing the
                # published sidecar against the raw repo tensors (every norm
                # off by exactly 1.0, fc byte-identical).
                out[key] = (w.astype(mx.float32) + 1.0).astype(mx.bfloat16)
            else:
                out[key] = w
            mx.eval(*out.values())

    for base, banks in sorted(experts.items()):
        stacked = mx.array(np.stack([banks[e] for e in range(len(banks))])).view(mx.bfloat16)
        wq, scales, biases = mx.quantize(stacked, group_size=64, bits=4)
        out[base + ".weight"] = wq
        out[base + ".scales"] = scales.astype(mx.bfloat16)
        out[base + ".biases"] = biases.astype(mx.bfloat16)
        mx.eval(wq, scales, biases)
        print(f"  {base}  {len(banks)} experts -> {list(wq.shape)}", flush=True)

    (OUT / "mtp").mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(OUT / "mtp" / "weights.safetensors"), out)
    total = sum(v.nbytes for v in out.values())
    print(f"wrote {len(out)} tensors ({total / 1e6:.0f} MB) to {OUT}/mtp/weights.safetensors")


if __name__ == "__main__":
    main()
