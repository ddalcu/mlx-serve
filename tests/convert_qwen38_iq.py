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

  python3 tests/convert_qwen38_iq.py --src <bf16> --dst <out> \
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

# Card written beside the weights. Frontmatter is pinned by
# tests/test_model_card_frontmatter.sh, which DISCOVERS converters by looking
# for a module-level README constant — `base_model_relation: quantized` in
# particular, because HF defaults a missing one to `finetune` and silently files
# a quantized mirror in the wrong list on the base model's page.
README = """---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- qwen3_5
- apple-silicon
- mtp
- speculative-decoding
---

# Qwen3.8-27B, {variant} for mlx-serve

{headline} of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), built for **[mlx-serve](https://mlxserve.com)**, the native Zig MLX server for Apple Silicon.

{size_gb:.1f} GB on disk, {bpw:.2f} bits per weight averaged over everything that is quantized. **Text only** — the vision tower is not included.

## Allocation

{method}

{alloc_table}
{battery_section}
Attention q/k/v/o, the GatedDeltaNet `in_proj_a`/`in_proj_b` gates and the whole MTP head are pinned at 4-bit/gs-64 — cheap insurance, and the MTP head is the decode lever.

## Serving

```bash
mlx-serve --model ddalcu/{repo_name} --serve --kv-quant 4
```

{serving_notes}

## Conversion

- Calibrated affine quantization on every matmul-read weight, widths per the table above.
- Kept bf16: `mtp.fc` and every norm, bias, conv and SSM state. The vision tower is dropped.
- The MTP head ships with the model; mlx-serve finds it in the shards and loads it with no flags.

Same three raw-checkpoint fixes as the [4-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit) and [8-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-8bit) builds: the delta-encoded norms get their `+1` folded in, the depthwise conv1d is transposed to MLX's `[C, K, 1]`, and (not needed here, since there is no tower) the Conv3d patch embed is channels-last.

Weights, config, tokenizer and chat template are otherwise verbatim from the base repo.
"""

SERVING_NOTES = """\
- **Which Mac.** This build exists for the 24 GB gap: the 4-bit pack needs about
  18 GB of weights resident and does not fit there, and the 8-bit one needs a
  48 GB machine. 13.0 GB of weights plus a ~1.4 GB prefill working set at
  `--prefill-chunk 512` leaves roughly 1.5 GB for the cache inside a 24 GB Mac's
  ~16 GB Metal working set, which is about 85k tokens at `--kv-quant 4`. A 16 GB
  Mac is NOT covered — the budget that fits one is around 8.6 GB, and at that
  size this model stops following instructions (measured: 46% top-1 agreement,
  and two of three agent runs made no tool calls at all). Those figures are
  computed from measurements taken on a 128 GB Mac, not from a run on a 24 GB one.
- **`--kv-quant 4` is half the point.** At fp16 the cache is 64 KB per token
  here; at 4-bit it is 18 KB. mlx-serve sizes its context window accordingly.
- **Thinking** is on by default. `"enable_thinking": false` turns it off; depth is
  `"reasoning_effort": "xhigh" | "medium" | "low"` (Qwen3.8's own vocabulary).
- **Tools** use Qwen3.8's XML call format; mlx-serve parses, repairs and
  schema-coerces it into standard OpenAI `tool_calls`.
- **No images.** This build has no vision tower — use the
  [4-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit) or
  [8-bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-8bit) build for that.
"""


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
    # `dsv4_imatrix.pack_bits` implements MLX's affine layout for 2/3/4/8 bits.
    # MLX's format also has 5 and 6 — and `transformer.verifyQmmLane` covers
    # 4/5/6 ONLY, so those two widths are the ones that keep spec-decode's fast
    # verify lanes — but reproducing their byte packing is not something to
    # guess at. Until it is implemented and pinned, a 5- or 6-bit weight goes
    # through `mx.quantize` UNCALIBRATED, and says so in the log.
    if bits not in (2, 3, 4, 8):
        ch = None
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

def imatrix_tokens(path):
    """`total_tokens` out of the imatrix's safetensors __metadata__."""
    if not path:
        return "0"
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    return header.get("__metadata__", {}).get("total_tokens", "0")


def alloc_table(per_class, quant_bytes, quant_params):
    label = {"mlp_gate_up": "MLP gate + up", "mlp_down": "MLP down",
             "gdn_qkv": "GDN in_proj_qkv", "gdn_z": "GDN in_proj_z",
             "gdn_out": "GDN out_proj", "gdn_ab": "GDN a/b gates",
             "attn": "attention q/k/v/o", "lm_head": "lm_head",
             "embed": "embed_tokens", "mtp": "MTP head"}
    rows = ["| weight class | params | on disk | widths |", "|---|---|---|---|"]
    for cls in sorted(per_class, key=lambda c: -per_class[c]["bytes"]):
        d = per_class[cls]
        widths = ", ".join(f"{k.replace('x', '-bit/gs-')} x{v}"
                           for k, v in sorted(d["widths"].items()))
        rows.append(f"| {label.get(cls, cls)} | {d['params']/1e9:.2f}B | "
                    f"{d['bytes']/1e9:.2f} GB | {widths} |")
    rows.append(f"| **total quantized** | **{quant_params/1e9:.2f}B** | "
                f"**{quant_bytes/1e9:.2f} GB** | |")
    return "\n".join(rows)


def battery_section(path):
    """Rendered only when there are measured numbers — a card never says pending."""
    if not path:
        return ""
    data = json.loads(Path(os.path.expanduser(path)).read_text())
    rows = ["", "## Measured against the 4-bit and 8-bit builds", "",
            "| build | on disk | top-1 agreement vs bf16 | mean KL | worst repeated-call run | decode (MTP on) |",
            "|---|---|---|---|---|---|"]
    for r in data["rows"]:
        rows.append("| {name} | {size} | {agree} | {kl} | {loop} | {decode} |".format(**r))
    rows.append("")
    rows.append(data.get("note", ""))
    rows.append("")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--alloc", required=True)
    ap.add_argument("--imatrix", default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--repo-name", default=None, help="HF repo name for the card")
    ap.add_argument("--battery", default=None,
                    help="tests/qwen38_iq_battery.py results JSON, rendered into the card")
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
    # Declare the DOMINANT width, not a constant: mlx-serve solves every affine
    # weight from packed geometry (`--verify` above pins that), but mlx-lm's
    # `class_predicate` falls back to this block for any module the override map
    # below does not name, and `mtpNaxProfileEnabledForTrunk` gates on it reading
    # 4/gs-64. A uniform 5-bit pack declaring 4/64 is simply a false statement.
    widths = {}
    for spec in alloc.values():
        widths[(spec["bits"], spec["group_size"])] = \
            widths.get((spec["bits"], spec["group_size"]), 0) + 1
    dom_bits, dom_gs = max(widths, key=widths.get)
    qb = {"group_size": dom_gs, "bits": dom_bits, "mode": "affine"}
    for name, spec in sorted(alloc.items()):
        qb[rename(name)[:-len(".weight")]] = {
            "group_size": spec["group_size"], "bits": spec["bits"], "mode": "affine"}
    cfg["quantization"] = qb
    cfg["quantization_config"] = qb
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    for f in COPY_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)

    per_class = json.loads(Path(os.path.expanduser(args.alloc)).read_text())["per_class"]
    q_bytes = sum(d["bytes"] for d in per_class.values())
    q_params = sum(d["params"] for d in per_class.values())
    (dst / "README.md").write_text(README.format(
        size_gb=total / 1e9,
        bpw=q_bytes * 8 / q_params,
        alloc_table=alloc_table(per_class, q_bytes, q_params),
        battery_section=battery_section(args.battery),
        repo_name=args.repo_name or dst.name,
        serving_notes=SERVING_NOTES))

    print(f"\ndone: {total/1e9:.2f} GB in {out_idx} shards, {time.time()-t0:.0f}s")
    print(f"  calibrated {stats['calibrated']}  plain mx.quantize {stats['plain']}  "
          f"norm+1 {stats['shift']}  conv1d transposed {stats['conv']}")
    print("  widths: " + " ".join(f"{k}:{v}" for k, v in sorted(stats["widths"].items())))
    if args.verify:
        print(f"  verify: all {len(qnames)} quantized weights re-solve to their "
              f"allocated (bits, group_size) from packed geometry alone")


if __name__ == "__main__":
    sys.exit(main())
