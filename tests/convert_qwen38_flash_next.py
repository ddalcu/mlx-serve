#!/usr/bin/env python3
"""Qwen/Qwen3.8-Flash-Next (model_type qwen4_exp) bf16 -> MLX affine pack in
the layout mlx-serve loads. Streams shard by shard from the Hub: download one,
quantize it, delete it. Peak scratch = a few HF shards, never the 360 GB.

Layout / renames:
    model.language_model.*          -> language_model.model.*
    lm_head.*                       -> language_model.lm_head.*
    mtp.*                           -> language_model.mtp.*
    mlp.experts.gate_up_proj [E,2I,H] -> mlp.switch_mlp.{gate,up}_proj [E,I,H]
    mlp.experts.down_proj    [E,H,I]  -> mlp.switch_mlp.down_proj
    model.visual.*                  -> skipped by the stream; `--add-vision`
        appends them bf16 pass-through as `model-vision.safetensors` (one HF
        shard holds the whole tower) and restores `vision_config`
    ple.ple_embedding.ngram_embedding.shard_N -> ONE merged table in
        `ngram_table.bin` (safetensors-format, NOT a *.safetensors name so the
        directory loader never mlx-loads it; the engine gathers rows from the
        mmap on the host).

Widths: routed experts `--bits` (default 4, gs 64); n-gram table
`--ngram-bits` (default 4, gs 32 because the row width is 160; 3/5/6 need
mlx-serve >= 26.9.1, older readers unpack them as noise, #305); every other
2-D projection `--nonexpert-bits` (default 8, 4 = the -all pack) gs 64; embed_tokens 4-bit gs 64; 2-D weights with fewer
than 32 rows (router `mlp.gate`, `shared_expert_gate`, `block_inject_weight`,
GDN `in_proj_a/b`) and every 1-D tensor stay bf16.

Every `Qwen4ExpTextRMSNorm` in this arch computes `x * (1 + w)`; the `+1` is
folded here (hc_norm, q/k_norm, indexer q/k_layernorm, ple norm_*, mtp
pre_fc_norm_*). `linear_attn.norm` is the gated norm with a plain weight and
is left alone. Depthwise conv1d weights ship HF's [C, 1, K] and MLX conv1d
reads [C, K, 1].

  python3 tests/convert_qwen38_flash_next.py --dst ~/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit \
      --stage ~/claude-tmp/qwen38-flash-next/stage
  python3 tests/convert_qwen38_flash_next.py --add-vision --dst <pack> --stage <dir>   # or --src <hf dir>
"""

import argparse
import json
import os
import shutil
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dsv4_weights import (bf16_to_f32, f32_to_bf16_u16,  # noqa: E402
                                  mlx_affine_quant, write_safetensors_raw)

REPO = "Qwen/Qwen3.8-Flash-Next"
README = """\
---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx-serve
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- qwen4_exp
- moe
- sparse-attention
- ngram-embedding
---

# Qwen3.8-Flash-Next for mlx-serve (4-bit experts, 8-bit rest)

mlx-serve pack of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
the Qwen4 preview architecture (`model_type: qwen4_exp`). Runs on a 128 GB Mac
with about 75 GB resident. Includes the MTP head and the vision tower (image
and video input).

```bash
mlx-serve --model ddalcu/{repo_name} --serve
```

Measured on an M4 Max 128 GB (mlx-serve, first port, no tuning yet): ~67-69 GB
resident, decode 29-34 tok/s serial, prefill ~400 tok/s on a 25k-token prompt,
a needle at 24.8k tokens recovered with sparse attention engaged. The MTP head
loads and drafts (~1 accepted token per round) but its round cost is not yet
competitive with serial decode, so leave it off for now.

## What is different about this model

This is not a Qwen3.5-style pack. Three things around the usual GDN + MoE trunk:

- **Gated residual streams.** The residual is 4 streams wide (4 x 2560). Every
  block reads a sigmoid-mixed average of the normalized streams and writes back
  through per-stream scalar gates. The final mixer replaces the usual final norm.
- **N-gram embedding (51B parameters).** A second embedding table indexed by
  hashed bigrams and trigrams of the token ids: 16 heads, each a prime-sized
  bucket space of ~20M rows, 160 dims per row, injected once before layer 1.
  It is a lookup, no compute, which is why Qwen quotes the model as 125B: the
  full checkpoint is 125B trunk + 51B n-gram + 4B MTP = 180B (360 GB bf16).
- **Qwen Sparse Attention.** Past 2048 tokens each attention layer only reads
  the 512 most relevant 4-token blocks per query (picked by a small indexer),
  plus the query's own partial block. Attention cost stays flat with context.
  Native 262k context.

## How this pack stores the n-gram table

The 51B table is NOT in the safetensors shards. It is one merged 4-bit table
in `ngram_table.bin` ({ngram_gb:.1f} GB, safetensors format, `.bin` so nothing
mlx-loads it). mlx-serve mmaps the file and, per token, dequantizes the 16 rows
it needs on the CPU (16 x 80 bytes) and uploads only the resulting 2560-vector.
The table never becomes resident: its cost is page cache, which the OS evicts
as needed. That is the difference between this pack and mlx-lm style packs
that ship the table as 128 quantized tensors and load it onto the GPU
(+32 GB resident, ~107 GB total for a 4-bit pack).

Expected effect: decode speed unchanged (16 tiny reads against a ~20 ms step),
cold-cache prefill of very long prompts may pay up to ~1 s per 8k tokens of
random reads on the SSD, warm cache is free. No user-space cache is needed,
the page cache already is an LRU over exactly this access pattern.

## Widths

| tensors | width |
|---|---|
| routed experts (512 x 48 layers, the 121B) | 4-bit, group 64 |
| attention, GDN, hyper-connections, indexer, shared experts | {nonexpert_bits}-bit, group 64 |
| lm_head | 8-bit, group 64 |
| embed_tokens | 4-bit, group 64 |
| n-gram table | 4-bit, group 32 (row width 160) |
| routers, inject gates, norms, convs, SSM state | bf16 |
| MTP head | same policy as the trunk |

Every `(1 + w)` RMSNorm has the `+1` folded into the stored weight; depthwise
convs are transposed to MLX's `[C, K, 1]`; `experts.gate_up_proj` is split into
`switch_mlp.gate_proj` / `up_proj`. The vision tower ships dense bf16 in
`model-vision.safetensors` (~0.9 GB).

## Serving notes

- **Memory.** ~75 GB resident plus KV cache. mlx-serve sizes the context to
  what fits; `--kv-quant 8` halves the cache.
- **MTP.** The checkpoint's own 1-layer speculative head is loaded from the
  pack and works (`--mtp` or per-request `"enable_mtp": true`), but as of this
  build it decodes slower than serial. Default-off; a later mlx-serve release
  will flip it once the round cost is fixed.
- **v1 limits in mlx-serve.** One request at a time (no batched decode), no
  prefix-cache reuse between turns yet, PLD/DFlash speculation off (MTP is the
  speculative path). Very long prompts (past ~64k) want a smaller
  `--prefill-chunk` because the sparse-attention selection is built per chunk.
- **Thinking** is on by default (`"enable_thinking": false` turns it off).
  Tools use Qwen3.8's XML call format; mlx-serve parses and schema-coerces it.
- **Images and video** go through the Qwen3-VL-style tower (`model.visual.*`,
  dense bf16). MTP is declined on image turns (serial decode).

## Conversion

`tests/convert_qwen38_flash_next.py` in the mlx-serve repo. It streams the
360 GB bf16 checkpoint shard by shard from the Hub (download, quantize, delete),
so it converts on a machine with ~150 GB free. The engine was validated against
HF transformers (trunk) and the vLLM/SGLang MTP math on a tiny random model
before the full conversion.
"""

SHARD_BYTES = 2 * 1024 ** 3
VISION_FILE = "model-vision.safetensors"
VISION_PREFIX = "model.visual."
COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json", "vocab.json", "merges.txt", "LICENSE")
NORM_FOLD_SUFFIXES = (
    "hc_norm.weight", "q_norm.weight", "k_norm.weight",
    "q_layernorm.weight", "k_layernorm.weight",
    "ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight",
    "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
)
NGRAM_MARK = ".ple.ple_embedding.ngram_embedding.shard_"


def rename(k):
    if k.startswith("model.language_model."):
        k = "language_model.model." + k[len("model.language_model."):]
    elif k.startswith("mtp."):
        k = "language_model.mtp." + k[len("mtp."):]
    elif k == "lm_head.weight":
        k = "language_model.lm_head.weight"
    return k


def needs_norm_fold(name):
    return name.endswith(NORM_FOLD_SUFFIXES)


def width_for(name, shape, nonexpert_bits=8):
    """(bits, group_size) or None for bf16 pass-through."""
    if len(shape) != 2 or shape[0] < 32 or shape[1] % 64 != 0:
        return None
    if name.endswith("embed_tokens.weight"):
        return 4, 64
    return nonexpert_bits, 64


def read_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    return header, 8 + hlen


def read_raw(path, data_off, meta):
    b, e = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_off + b)
        raw = f.read(e - b)
    assert len(raw) == e - b, f"{path}: short read"
    np_dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U32": np.uint32,
             "I32": np.int32, "I64": np.int64, "U8": np.uint8}[meta["dtype"]]
    return np.frombuffer(raw, dtype=np_dt).reshape(meta["shape"])


def quant(arr_u16, bits, gs):
    """bf16 [.., out, in] -> triples, quantizing 3-D banks as one 2-D matrix."""
    shape = arr_u16.shape
    w = bf16_to_f32(arr_u16.reshape(-1, shape[-1]))
    wq, sc, bi = mlx_affine_quant(w, bits, group_size=gs)
    lead = tuple(shape[:-1])
    return ((wq[0], lead + (wq[1][-1],), wq[2]),
            (sc[0], lead + (sc[1][-1],), sc[2]),
            (bi[0], lead + (bi[1][-1],), bi[2]))


class NgramTable:
    """Merged quantized n-gram table, written row-block by row-block."""

    def __init__(self, path, rows, bits, gs, dim):
        self.path, self.rows, self.bits, self.gs, self.dim = path, rows, bits, gs, dim
        self.wcols = dim * bits // 32
        self.scols = dim // gs
        self.w_bytes = rows * self.wcols * 4
        self.s_bytes = rows * self.scols * 2
        header = {
            "__metadata__": {"format": "mlx-serve-ngram", "bits": str(bits), "group_size": str(gs)},
            "weight": {"dtype": "U32", "shape": [rows, self.wcols], "data_offsets": [0, self.w_bytes]},
            "scales": {"dtype": "BF16", "shape": [rows, self.scols],
                       "data_offsets": [self.w_bytes, self.w_bytes + self.s_bytes]},
            "biases": {"dtype": "BF16", "shape": [rows, self.scols],
                       "data_offsets": [self.w_bytes + self.s_bytes, self.w_bytes + 2 * self.s_bytes]},
        }
        hjson = json.dumps(header).encode()
        hjson += b" " * ((8 - len(hjson) % 8) % 8)
        self.data_off = 8 + len(hjson)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", len(hjson)))
                f.write(hjson)
                f.truncate(self.data_off + self.w_bytes + 2 * self.s_bytes)

    def write(self, row0, triples):
        (wdt, wshape, wraw), (_, sshape, sraw), (_, _, braw) = triples
        n = wshape[0]
        assert wshape[1] == self.wcols and sshape[1] == self.scols
        with open(self.path, "r+b") as f:
            f.seek(self.data_off + row0 * self.wcols * 4)
            f.write(wraw)
            f.seek(self.data_off + self.w_bytes + row0 * self.scols * 2)
            f.write(sraw)
            f.seek(self.data_off + self.w_bytes + self.s_bytes + row0 * self.scols * 2)
            f.write(braw)
        return n


class Prefetcher:
    def __init__(self, files, stage, ahead):
        from huggingface_hub import hf_hub_download
        self.dl = hf_hub_download
        self.files, self.stage, self.ahead = files, stage, ahead
        self.done = {}
        self.lock = threading.Condition()
        self.next_idx = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _fetch(self, fname):
        for attempt in range(8):
            try:
                return self.dl(REPO, fname, local_dir=self.stage)
            except Exception as e:  # noqa: BLE001
                print(f"  download retry {attempt} {fname}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"download failed: {fname}")

    def _run(self):
        for i, fname in enumerate(self.files):
            with self.lock:
                while i - self.next_idx >= self.ahead:
                    self.lock.wait()
            path = self._fetch(fname)
            with self.lock:
                self.done[i] = path
                self.lock.notify_all()

    def get(self, i):
        with self.lock:
            while i not in self.done:
                self.lock.wait()
            self.next_idx = i + 1
            self.lock.notify_all()
            return self.done.pop(i)


def add_vision(args):
    """Append the bf16 vision tower to an existing pack: only the HF shard(s)
    holding `model.visual.*` are fetched, every tensor passes through
    unquantized into VISION_FILE, the index and config.json are merged.
    Idempotent."""
    dst = Path(os.path.expanduser(args.dst))
    idx_path = dst / "model.safetensors.index.json"
    cfg_path = dst / "config.json"
    index = json.loads(idx_path.read_text())
    cfg = json.loads(cfg_path.read_text())
    if (dst / VISION_FILE).exists() and VISION_FILE in index["weight_map"].values() and "vision_config" in cfg:
        print(f"{dst.name}: vision tower already present")
        return 0
    local = args.src is not None
    if local:
        stage = Path(os.path.expanduser(args.src))
    else:
        from huggingface_hub import hf_hub_download
        stage = Path(os.path.expanduser(args.stage))
        stage.mkdir(parents=True, exist_ok=True)
        for f in ("config.json", "model.safetensors.index.json"):
            hf_hub_download(REPO, f, local_dir=stage)
    hf_cfg = json.loads((stage / "config.json").read_text())
    if (dst / VISION_FILE).exists():
        # A sibling pack's shard hard-linked in (same bytes): merge only.
        header, _ = read_header(str(dst / VISION_FILE))
        out = {k: (m["dtype"], m["shape"], b"\0" * (m["data_offsets"][1] - m["data_offsets"][0])) for k, m in header.items()}
    else:
        if (stage / "model.safetensors.index.json").exists():
            wm = json.loads((stage / "model.safetensors.index.json").read_text())["weight_map"]
            files = sorted({v for k, v in wm.items() if k.startswith(VISION_PREFIX)})
        else:
            files = ["model.safetensors"]
        out = {}
        for fname in files:
            path = str(stage / fname) if local else hf_hub_download(REPO, fname, local_dir=str(stage))
            header, data_off = read_header(path)
            for name in sorted(header):
                if not name.startswith(VISION_PREFIX):
                    continue
                meta = header[name]
                out[name] = (meta["dtype"], meta["shape"], read_raw(path, data_off, meta).tobytes())
            if not local:
                os.remove(path)
        if not out:
            raise SystemExit(f"no {VISION_PREFIX}* tensors in {files}")
        write_safetensors_raw(str(dst / VISION_FILE), out)
    added = 0
    for k, t in out.items():
        if index["weight_map"].get(k) != VISION_FILE:
            added += len(t[2])
        index["weight_map"][k] = VISION_FILE
    index["metadata"]["total_size"] = index["metadata"].get("total_size", 0) + added
    idx_path.write_text(json.dumps(index, indent=2))
    cfg["vision_config"] = hf_cfg["vision_config"]
    cfg["language_model_only"] = False
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {VISION_FILE}: {len(out)} tensors, {os.path.getsize(dst / VISION_FILE)/1e9:.2f} GB")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True)
    ap.add_argument("--stage", default=None, help="download staging dir (Hub mode)")
    ap.add_argument("--src", default=None, help="local HF checkpoint dir instead of the Hub")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--ngram-bits", type=int, default=4)
    ap.add_argument("--nonexpert-bits", type=int, default=8, help="width for every non-expert 2-D projection (4 = the -all pack)")
    ap.add_argument("--ahead", type=int, default=2)
    ap.add_argument("--add-vision", action="store_true", help="append the bf16 vision tower to the pack at --dst (no re-stream)")
    args = ap.parse_args()
    if args.add_vision:
        return add_vision(args)

    dst = Path(os.path.expanduser(args.dst))
    dst.mkdir(parents=True, exist_ok=True)
    local = args.src is not None
    if local:
        stage = Path(os.path.expanduser(args.src))
        for f in COPY_FILES:
            if (stage / f).exists():
                shutil.copy2(stage / f, dst / f)
    else:
        from huggingface_hub import hf_hub_download
        stage = Path(os.path.expanduser(args.stage))
        stage.mkdir(parents=True, exist_ok=True)
        for f in COPY_FILES + ("config.json", "model.safetensors.index.json"):
            try:
                p = hf_hub_download(REPO, f, local_dir=stage)
            except Exception:  # noqa: BLE001
                continue
            if f not in ("config.json", "model.safetensors.index.json"):
                shutil.copy2(p, dst / f)
    if (stage / "model.safetensors.index.json").exists():
        index = json.loads((stage / "model.safetensors.index.json").read_text())
        files = sorted(set(index["weight_map"].values()))
        names = list(index["weight_map"])
    else:
        files = ["model.safetensors"]
        names = list(read_header(stage / "model.safetensors")[0])
    ngram_names = sorted((k for k in names if NGRAM_MARK in k),
                         key=lambda k: int(k.rsplit("_", 1)[1].split(".")[0]))
    n_ngram_shards = len(ngram_names)

    state_path = dst / ".convert_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else \
        {"done": [], "out_idx": 0, "out_map": {}, "total": 0, "ngram_rows": None}
    todo = [f for f in files if f not in state["done"]]
    print(f"{len(files)} HF shards, {len(todo)} to do, {n_ngram_shards} n-gram shards", flush=True)
    pf = None if local else Prefetcher(todo, str(stage), args.ahead)
    ngram = None
    if state["ngram_rows"]:
        ngram = NgramTable(str(dst / "ngram_table.bin"), state["ngram_rows"], args.ngram_bits, 32, 160)

    out, out_bytes = {}, 0
    t0 = time.time()

    def flush():
        nonlocal out, out_bytes
        if not out:
            return
        state["out_idx"] += 1
        fname = f"model-{state['out_idx']:05d}.safetensors"
        write_safetensors_raw(str(dst / fname), out)
        for k in out:
            state["out_map"][k] = fname
        state["total"] += out_bytes
        print(f"  wrote {fname} {out_bytes/1e9:.2f} GB ({len(out)} tensors)", flush=True)
        out, out_bytes = {}, 0

    def emit(nk, triple):
        nonlocal out_bytes
        out[nk] = triple
        out_bytes += len(triple[2])

    def emit_q(nk, arr, bits, gs):
        wq, sc, bi = quant(arr, bits, gs)
        base = nk[:-len(".weight")] if nk.endswith(".weight") else nk
        emit(base + ".weight", wq)
        emit(base + ".scales", sc)
        emit(base + ".biases", bi)

    for i, fname in enumerate(todo):
        path = str(stage / fname) if local else pf.get(i)
        header, data_off = read_header(path)
        ts = time.time()
        for name in sorted(header):
            meta = header[name]
            if name.startswith("model.visual."):
                continue
            if NGRAM_MARK in name:
                shard_idx = int(name.rsplit("_", 1)[1].split(".")[0])
                arr = read_raw(path, data_off, meta)
                if ngram is None:
                    rows = arr.shape[0] * n_ngram_shards
                    state["ngram_rows"] = rows
                    ngram = NgramTable(str(dst / "ngram_table.bin"), rows, args.ngram_bits, 32, arr.shape[1])
                ngram.write(shard_idx * arr.shape[0], quant(arr, args.ngram_bits, 32))
                continue
            nk = rename(name)
            arr = read_raw(path, data_off, meta)
            if nk.endswith(".mlp.experts.gate_up_proj"):
                half = arr.shape[1] // 2
                base = nk[:-len("experts.gate_up_proj")] + "switch_mlp."
                emit_q(base + "gate_proj.weight", np.ascontiguousarray(arr[:, :half]), args.bits, 64)
                emit_q(base + "up_proj.weight", np.ascontiguousarray(arr[:, half:]), args.bits, 64)
                continue
            if nk.endswith(".mlp.experts.down_proj"):
                emit_q(nk[:-len("experts.down_proj")] + "switch_mlp.down_proj.weight", arr, args.bits, 64)
                continue
            w = width_for(nk, arr.shape, args.nonexpert_bits) if meta["dtype"] == "BF16" else None
            if w:
                emit_q(nk, arr, *w)
                continue
            if nk.endswith("conv1d.weight") and arr.ndim == 3:
                arr = np.ascontiguousarray(np.swapaxes(arr, 1, 2))
            if needs_norm_fold(nk):
                assert meta["dtype"] == "BF16", nk
                arr = f32_to_bf16_u16(bf16_to_f32(arr) + 1.0)
            emit(nk, (meta["dtype"], arr.shape, np.ascontiguousarray(arr).tobytes()))
            if out_bytes >= SHARD_BYTES:
                flush()
        flush()
        if not local:
            os.remove(path)
        state["done"].append(fname)
        state_path.write_text(json.dumps(state))
        print(f"[{i+1}/{len(todo)}] {fname} done in {time.time()-ts:.0f}s, "
              f"elapsed {(time.time()-t0)/60:.1f} min, out {state['total']/1e9:.1f} GB", flush=True)

    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": state["total"]}, "weight_map": state["out_map"]}, indent=2))
    cfg = json.loads((stage / "config.json").read_text())
    cfg.pop("vision_config", None)
    cfg["language_model_only"] = True
    cfg["quantization"] = {"group_size": 64, "bits": args.bits, "mode": "affine"}
    cfg["quantization_config"] = cfg["quantization"]
    cfg["ngram_table"] = {"file": "ngram_table.bin", "bits": args.ngram_bits, "group_size": 32}
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    (dst / "README.md").write_text(README.format(
        repo_name=dst.name, ngram_gb=os.path.getsize(dst / "ngram_table.bin") / 1e9, nonexpert_bits=args.nonexpert_bits))
    state_path.unlink()
    print(f"done: {state['total']/1e9:.1f} GB trunk + ngram_table.bin "
          f"{os.path.getsize(dst / 'ngram_table.bin')/1e9:.1f} GB in {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    sys.exit(main())
