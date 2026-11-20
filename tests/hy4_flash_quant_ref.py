#!/usr/bin/env python3
"""Quantized + external-Engram variant of the tiny hy_v4_flash fixture.

The fp32 fixture in hy4_flash_ref.py exercises neither affine weights nor the
external `engram_table.bin` a real pack ships, so its 3e-5 logit match proves
nothing about real-B. This script builds the same tiny model, but writes:

  * every >=32-row 2-D linear and every expert bank as MLX affine triples
    (`weight` U32 + BF16 `scales`/`biases`), 8-bit gs64 for ordinary linears,
    4-bit gs64 for embed/lm_head, 3-bit gate/up and 4-bit down for experts
    (4-bit on the tail layers);
  * all `multi_head_embedding.embedding.weight` tables as ONE 4-bit gs32
    safetensors-format `engram_table.bin`, named by config key `engram_table`
    with per-layer row offsets;
  * `fixture.safetensors` logits computed by the SAME NumPy oracle, but with
    each weight replaced by its dequantized value, so a correct engine still
    matches at bf16-rounding tolerance instead of fp32-exact equality.

Requires a Python with `mlx` (for the affine quantizer), e.g.
  /Users/harikrishnan/.cache/uv/archive-v0/-PS9oDhHRL-0bsxZ/bin/python \
      tests/hy4_flash_quant_ref.py --out /tmp/hy4quant
"""

import argparse
import json
import os
import re
import sys

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hy4_flash_ref as R  # noqa: E402
import convert_dsv4_weights as C  # noqa: E402

F = np.float32


def dtype_name(a):
    return {
        np.dtype(np.float32): "F32",
        np.dtype(np.int32): "I32",
        np.dtype(np.int64): "I64",
        np.dtype(np.uint8): "U8",
        np.dtype(np.uint16): "U16",
    }[a.dtype]


def dequant(triple, bits, gs):
    wt, sc, bi = triple
    q = np.frombuffer(wt[2], np.uint32).reshape(wt[1])
    s = np.frombuffer(sc[2], np.uint16).reshape(sc[1])
    b = np.frombuffer(bi[2], np.uint16).reshape(bi[1])
    return C.mlx_affine_dequant_f32(q, s, b, bits, gs)


def build(cfg, w, out, only="all"):
    quant_engram = only in ("all", "engram")
    quant_experts = only in ("all", "experts")
    quant_other = only in ("all", "linears")
    pack = {}
    oracle = dict(w)
    engram = []  # (weight bytes, scales bytes, biases bytes, rows, layer)
    for name in sorted(w):
        a = w[name]
        is_table = "multi_head_embedding.embedding.weight" in name
        layer = None
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            layer = int(m.group(1))
        if is_table and not quant_engram:
            pack[name] = ("F32", a.shape, np.ascontiguousarray(a).tobytes())
            continue
        if is_table:
            triple = C.mlx_affine_quant(a, 4, 32)
            oracle[name] = dequant(triple, 4, 32)
            engram.append((triple, a.shape[0], layer))
            continue
        if len(a.shape) >= 2 and (name.endswith(".weight") or name.endswith("_proj")):
            use = True
            if name.startswith("model.hc_head.") or ".hc_" in name or "intra_group_ape" in name:
                use = False
            if a.shape[0] < 32:
                use = False
            if use:
                is_expert = "experts.gate_up_proj" in name or "experts.down_proj" in name
                if not quant_other or (is_expert and not quant_experts):
                    pack[name] = (dtype_name(a), a.shape, np.ascontiguousarray(a).tobytes())
                    continue
                if is_expert:
                    bits = 4 if layer is None or layer >= cfg["layers"] - 2 else (3 if "gate_up" in name else 4)
                    gs = 64
                elif name in ("model.embed_tokens.weight", "lm_head.weight"):
                    bits, gs = 4, 64
                else:
                    bits, gs = 8, 64
                triple = C.mlx_affine_quant(a, bits, gs)
                oracle[name] = dequant(triple, bits, gs)
                base = name[:-len(".weight")] if name.endswith(".weight") else name
                pack[base + ".weight"] = triple[0]
                pack[base + ".scales"] = triple[1]
                pack[base + ".biases"] = triple[2]
                continue
        pack[name] = (dtype_name(a), a.shape, np.ascontiguousarray(a).tobytes())

    # One shared external table, appended by table order (sorted by layer).
    engram.sort(key=lambda x: x[2] if x[2] is not None else 0)
    if not engram:
        C.write_safetensors_raw(os.path.join(out, "model.safetensors"), pack)
        return oracle, None
    total_rows = sum(t[1] for t in engram)
    head_dim = cfg["head_dim"]
    wcols = head_dim * 4 // 32
    scols = head_dim // 32
    w = b"".join(t[0][0][2] for t in engram)
    s = b"".join(t[0][1][2] for t in engram)
    b = b"".join(t[0][2][2] for t in engram)
    C.write_safetensors_raw(os.path.join(out, "engram_table.bin"), {
        "weight": ("U32", (total_rows, wcols), w),
        "scales": ("BF16", (total_rows, scols), s),
        "biases": ("BF16", (total_rows, scols), b),
    }, {"format": "mlx-serve-engram", "bits": "4", "group_size": "32", "head_dim": str(head_dim)})
    spec = {
        "file": "engram_table.bin", "bits": 4, "group_size": 32, "head_dim": head_dim,
        "rows": total_rows, "tables": len(engram),
        "layers": [t[2] for t in engram],
        "table_rows": [t[1] for t in engram],
    }

    C.write_safetensors_raw(os.path.join(out, "model.safetensors"), pack)
    return oracle, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", choices=("all", "engram", "experts", "linears"), default="all")
    ap.add_argument("--indexer", action="store_true", help="keep the tiny top-k indexer (unstable at qmm rounding)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Every linear's input dim is a multiple of gs 64; the oracle's fp32 tiny
    # config (hidden 32, q_lora 12, ...) cannot be affine-quantized at gs 64.
    cfg = dict(
        hidden=64, vocab=64, layers=4, eps=1e-4, hc_mult=2, hc_magnitude=2.0,
        indexed_heads=2, swa_heads=4, qk_nope=8, rope_dim=8, v_dim=32,
        swa_qk_nope=8, swa_v_dim=16, kv_lora=64, q_lora=64,
        indexed_theta=10000.0, swa_theta=1000.0, mrope=[2, 1, 1],
        layer_types=["indexed_attention", "sliding_attention", "indexed_attention", "sliding_attention"],
        mlp_types=["dense", "sparse", "sparse", "sparse"],
        experts=32, top_k=2, moe_inter=64, latent=64, inter=64, route_scale=1.5, swiglu_limit=1.0,
        sliding=4, ratio=2, idx_heads=2, idx_head_dim=64, group_topk=3, candidate_topk=8,
        engram_layers=[1, 2], max_ngram=3, head_dim=32, lookup_size=80, pad=2,
        engram_sizes=[[11, 13], [17, 19]], hash_mult=[3, 5, 7],
    )
    if not args.indexer:
        cfg["candidate_topk"] = 100000
    rng = np.random.default_rng(args.seed)
    cfg["ids"] = rng.integers(3, cfg["vocab"] - 1, size=args.seq).tolist()
    w = R.make_weights(cfg, rng)
    # A random router puts two experts within rounding distance; one flipped
    # top-2 choice swamps the logits and hides whether affine math is right.
    # Pin a decisive margin so the comparison measures quantization, not ties.
    for li in range(cfg["layers"]):
        k = f"model.layers.{li}.mlp.gate.e_score_correction_bias"
        if k in w:
            w[k] = np.arange(cfg["experts"], 0, -1, dtype=F)

    oracle_w, spec = build(cfg, w, args.out, args.only)
    logits, layers = R.Ref(cfg, oracle_w).forward(cfg["ids"])

    fx = {"ids": np.array(cfg["ids"], np.int32), "logits": logits.astype(F)}
    for i, h in enumerate(layers):
        fx[f"layer.{i}"] = h.astype(F)
    save_file(fx, os.path.join(args.out, "fixture.safetensors"))

    cfg["quantization"] = {"group_size": 64, "bits": 3, "mode": "affine",
                           "nonexpert_bits": 8, "embed_bits": 4, "tail_layers": 2}
    if spec is not None:
        cfg["engram_table"] = spec
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {args.out}: {len(w)} tensors -> {len(oracle_w)} oracle tensors, "
          f"engram {spec['rows']} rows across {spec['tables']} layers")


if __name__ == "__main__":
    main()
