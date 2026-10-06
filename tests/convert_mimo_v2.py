#!/usr/bin/env python3
"""XiaomiMiMo/MiMo-V2.6-Flash (MOPD or RL release, model_type mimo_v2) -> an MLX
pack in mlx-lm's layout (ml-explore/mlx-lm#1219, the mlx-community mxfp4-q8
packs), so it loads in mlx-serve and mlx-lm alike. The source is FP8 (e4m3,
f32 128x128 block scales) for the attention QKV and the dense layer-0 MLP,
MXFP4 per expert for the routed banks, bf16 for the rest.

Layout / renames:
    mlp.experts.N.{gate,up,down}_proj.{weight,weight_scale} (U8, per expert)
        -> mlp.switch_mlp.{gate,up,down}_proj.{weight,scales}: the same bytes
           restacked [E, out, in/8] u32 + [E, out, in/32] u8 E8M0 scales, i.e.
           MLX `mxfp4` (lossless; a byte is two E2M1 codes, low nibble first)
    self_attn.qkv_proj (FP8) -> self_attn.{q,k,v}_proj. The source stores
        TENSOR-PARALLEL slabs: `tp` rank-local [q_r | k_r | v_r] row groups,
        each tiled by its own 128-row scale blocks (a rank's last tile can be
        partial). Dequantized per rank, then regrouped into global q, k, v.
    attention_value_scale, sinks, router and norms stay as released (the
        engine scales V at runtime, as mlx-lm does).
    model.mtp.layers.N.* -> same names, split + quantized like the trunk
        (`model-mtp.safetensors`); visual.* bf16 pass-through
        (`model-vision.safetensors`). The audio encoder is dropped.

Widths: every 2-D projection outside the experts (q/k/v/o, dense MLP,
embed_tokens, lm_head, MTP eh_proj) at `--bits` (8, gs 64, affine), or bf16
with `--bits 16` (the quality reference), each named in the config's
`quantization` block as mlx-lm expects.

  venv/bin/python tests/convert_mimo_v2.py --src ~/.mlx-serve/models/ddalcu/MiMo-V2.6-Flash-MOPD \
      --dst ~/.mlx-serve/models/ddalcu/MiMo-V2.6-Flash-MLX-Serve-MXFP4-Q8
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

BLOCK = 128
COPY_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "chat_template.jinja", "generation_config.json", "preprocessor_config.json", "LICENSE"]
README = """\
---
base_model: XiaomiMiMo/MiMo-V2.6-Flash-MOPD
base_model_relation: quantized
library_name: mlx-serve
license: mit
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- mimo_v2
- moe
- mxfp4
---

# MiMo-V2.6-Flash for mlx-serve ({width})

[XiaomiMiMo/MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)
converted for [mlx-serve](https://github.com/ddalcu/mlx-serve) on Apple Silicon.

- Routed experts: the release's own MXFP4 bytes, restacked into MLX `mxfp4` banks (lossless).
- Attention, the dense layer-0 MLP, embeddings and lm_head: {trunk}.
- Text and the three native MTP heads; the MiMo-ViT tower ships bf16. Audio is not included.

Converted by `tests/convert_mimo_v2.py` in the mlx-serve repository.
"""


class Source:
    """Tensor reader over the release's index: every shard opened once."""

    def __init__(self, root: Path):
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.where = index["weight_map"]
        self.handles = {}

    def handle(self, name: str):
        fname = self.where[name]
        if fname not in self.handles:
            self.handles[fname] = safe_open(str(self.root / fname), framework="pt")
        return self.handles[fname]

    def get(self, name: str) -> torch.Tensor:
        return self.handle(name).get_tensor(name)

    def shape(self, name: str) -> list:
        return self.handle(name).get_slice(name).get_shape()

    def has(self, name: str) -> bool:
        return name in self.where


def tp_fits(parts, widths, scale_rows: int) -> set:
    """Rank counts that shard whole heads of q, k, v (`widths` rows each) and
    tile each rank-local [q | k | v] slab into exactly `scale_rows` 128-row
    blocks. A slab that is a whole number of blocks fits every divisor, so only
    a tensor with a partial tile pins the count."""
    n = sum(parts)
    return {tp for tp in (1, 2, 4, 8)
            if all(p % (tp * w) == 0 for p, w in zip(parts, widths)) and scale_rows == tp * -(-(n // tp) // BLOCK)}


def solve_qkv_tp(src: "Source", cfg: dict) -> int:
    """The one rank count consistent with every fused QKV in the checkpoint."""
    fits = {1, 2, 4, 8}
    names = [n for n in src.where if n.endswith("self_attn.qkv_proj.weight_scale_inv")]
    for name in names:
        layer = re.search(r"layers\.(\d+)\.", name)
        sliding = "mtp" in name or cfg["hybrid_layer_pattern"][int(layer.group(1))] == 1
        fits &= tp_fits(attn_geometry(cfg, sliding), head_widths(cfg, sliding), src.shape(name)[0])
    assert len(fits) == 1, f"tensor-parallel layout ambiguous or unknown: {sorted(fits)}"
    return fits.pop()


def fp8_dequant(w: torch.Tensor, scale_inv: torch.Tensor, parts=None, tp: int = 1) -> torch.Tensor:
    """f32 [N, K] = code * the tile's scale. `parts` (q, k, v rows) marks a
    rank-slabbed fused QKV (`tp` rank-local slabs, each tiled on its own); the
    result is regrouped into global q, k, v."""
    n, k = w.shape
    sr, sc = scale_inv.shape
    assert sc == -(-k // BLOCK), f"scale cols {sc} vs K {k}"
    codes = w.to(torch.float32)
    col_scale = scale_inv.repeat_interleave(BLOCK, dim=1)[:, :k]
    if parts is None:
        assert sr == -(-n // BLOCK), f"scale rows {sr} vs N {n}"
        return codes * col_scale.repeat_interleave(BLOCK, dim=0)[:n]
    assert sr == tp * -(-(n // tp) // BLOCK), f"tp {tp} does not tile rows {n} into {sr} scale rows"
    rows, bpr = n // tp, -(-(n // tp) // BLOCK)
    out = {p: [] for p in range(3)}
    for r in range(tp):
        slab = codes[r * rows:(r + 1) * rows] * col_scale[r * bpr:(r + 1) * bpr].repeat_interleave(BLOCK, dim=0)[:rows]
        o = 0
        for p, size in enumerate(parts):
            out[p].append(slab[o:o + size // tp])
            o += size // tp
    return [torch.cat(out[p]) for p in range(3)]


def to_mx(t: torch.Tensor, dtype=mx.bfloat16) -> mx.array:
    if t.dtype == torch.uint8:
        return mx.array(t.numpy())
    return mx.array(t.to(torch.float32).numpy()).astype(dtype)


# Per-module quantization overrides for the config (mlx-lm's class_predicate).
OVERRIDES = {}


def put_linear(out: dict, name: str, w: mx.array, bits: int):
    """`name.weight` (+ `.scales`/`.biases` when quantized); w is [out, in]."""
    w = w.astype(mx.bfloat16)
    if bits == 16:
        out[name + ".weight"] = w
        OVERRIDES[name] = False
        return
    OVERRIDES[name] = {"group_size": 64, "bits": bits, "mode": "affine"}
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    out[name + ".weight"], out[name + ".scales"], out[name + ".biases"] = q, s, b


def attn_geometry(cfg: dict, sliding: bool):
    heads = cfg["swa_num_attention_heads"] if sliding else cfg["num_attention_heads"]
    kv = cfg["swa_num_key_value_heads"] if sliding else cfg["num_key_value_heads"]
    hd = cfg["swa_head_dim"] if sliding else cfg["head_dim"]
    vd = cfg["swa_v_head_dim"] if sliding else cfg["v_head_dim"]
    return heads * hd, kv * hd, kv * vd


def head_widths(cfg: dict, sliding: bool):
    hd = cfg["swa_head_dim"] if sliding else cfg["head_dim"]
    return hd, hd, cfg["swa_v_head_dim"] if sliding else cfg["v_head_dim"]


def convert_attention(src: Source, cfg: dict, pre: str, sliding: bool, tp: int, bits: int, out: dict):
    parts = attn_geometry(cfg, sliding)
    w = src.get(pre + "self_attn.qkv_proj.weight")
    if w.dtype == torch.float8_e4m3fn:
        q, k, v = fp8_dequant(w, src.get(pre + "self_attn.qkv_proj.weight_scale_inv"), parts, tp)
    else:
        q, k, v = torch.split(w.to(torch.float32), list(parts))
    for name, t in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
        put_linear(out, pre + "self_attn." + name, to_mx(t), bits)
    put_linear(out, pre + "self_attn.o_proj", to_mx(src.get(pre + "self_attn.o_proj.weight")), bits)
    if src.has(pre + "self_attn.attention_sink_bias"):
        copy_plain(src, [pre + "self_attn.attention_sink_bias"], out)


def dense_weight(src: Source, name: str) -> torch.Tensor:
    w = src.get(name)
    if w.dtype == torch.float8_e4m3fn:
        return fp8_dequant(w, src.get(name + "_scale_inv"))
    return w


def convert_dense_mlp(src: Source, pre: str, bits: int, out: dict):
    for p in ("gate_proj", "up_proj", "down_proj"):
        put_linear(out, f"{pre}mlp.{p}", to_mx(dense_weight(src, f"{pre}mlp.{p}.weight")), bits)


def convert_experts(src: Source, cfg: dict, pre: str, out: dict):
    n = cfg["n_routed_experts"]
    for p in ("gate_proj", "up_proj", "down_proj"):
        ws, ss = [], []
        for e in range(n):
            base = f"{pre}mlp.experts.{e}.{p}"
            w, s = src.get(base + ".weight"), src.get(base + ".weight_scale")
            assert w.dtype == torch.uint8 and s.dtype == torch.uint8, f"{base}: expected MXFP4 U8 bytes"
            ws.append(w.numpy())
            ss.append(s.numpy())
        out[f"{pre}mlp.switch_mlp.{p}.weight"] = mx.array(np.stack(ws)).view(mx.uint32)
        out[f"{pre}mlp.switch_mlp.{p}.scales"] = mx.array(np.stack(ss))
    copy_plain(src, [pre + "mlp.gate.weight"], out)
    out[pre + "mlp.gate.e_score_correction_bias"] = to_mx(src.get(pre + "mlp.gate.e_score_correction_bias"), mx.float32)


def copy_plain(src: Source, names, out: dict):
    for n in names:
        out[n] = to_mx(src.get(n))


class Writer:
    def __init__(self, dst: Path):
        self.dst, self.weight_map, self.total = dst, {}, 0

    def save(self, fname: str, tensors: dict):
        mx.eval(tensors)
        mx.save_safetensors(str(self.dst / fname), tensors, metadata={"format": "mlx"})
        for k, v in tensors.items():
            self.weight_map[k] = fname
            self.total += v.nbytes

    def finish(self):
        index = {"metadata": {"total_size": self.total}, "weight_map": dict(sorted(self.weight_map.items()))}
        (self.dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def pack_config(cfg: dict) -> dict:
    out = {k: v for k, v in cfg.items() if k not in ("quantization_config", "auto_map")}
    out["quantization"] = {"group_size": 32, "bits": 4, "mode": "mxfp4", **OVERRIDES}
    out["quantization_config"] = out["quantization"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--bits", type=int, default=8, choices=(4, 6, 8, 16))
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--no-mtp", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((args.src / "config.json").read_text())
    assert cfg["model_type"] == "mimo_v2", cfg["model_type"]
    src = Source(args.src)
    args.dst.mkdir(parents=True, exist_ok=True)
    writer = Writer(args.dst)
    pattern = cfg["hybrid_layer_pattern"]
    moe = cfg["moe_layer_freq"]
    tp = solve_qkv_tp(src, cfg)
    print(f"fused QKV: {tp} tensor-parallel slabs", flush=True)
    t0 = time.time()

    head = {}
    put_linear(head, "model.embed_tokens", to_mx(src.get("model.embed_tokens.weight")), args.bits)
    put_linear(head, "lm_head", to_mx(src.get("lm_head.weight")), args.bits)
    copy_plain(src, ["model.norm.weight"], head)
    writer.save("model-head.safetensors", head)

    for li in range(cfg["num_hidden_layers"]):
        pre = f"model.layers.{li}."
        out = {}
        copy_plain(src, [pre + "input_layernorm.weight", pre + "post_attention_layernorm.weight"], out)
        convert_attention(src, cfg, pre, pattern[li] == 1, tp, args.bits, out)
        if moe[li]:
            convert_experts(src, cfg, pre, out)
        else:
            convert_dense_mlp(src, pre, args.bits, out)
        writer.save(f"model-layer{li:02d}.safetensors", out)
        print(f"[{time.time() - t0:7.1f}s] layer {li} ({len(out)} tensors)", flush=True)

    if not args.no_mtp:
        out = {}
        for mi in range(cfg.get("num_nextn_predict_layers", 0)):
            pre = f"model.mtp.layers.{mi}."
            copy_plain(src, [pre + n for n in ("enorm.weight", "hnorm.weight", "input_layernorm.weight",
                                                "pre_mlp_layernorm.weight", "final_layernorm.weight")], out)
            put_linear(out, pre + "eh_proj", to_mx(src.get(pre + "eh_proj.weight")), args.bits)
            convert_attention(src, cfg, pre, True, tp, args.bits, out)
            convert_dense_mlp(src, pre, args.bits, out)
        if out:
            writer.save("model-mtp.safetensors", out)

    if not args.no_vision:
        names = [n for n in src.where if n.startswith("visual.")]
        if names:
            out = {}
            copy_plain(src, names, out)
            writer.save("model-vision.safetensors", out)

    writer.finish()
    (args.dst / "config.json").write_text(json.dumps(pack_config(cfg), indent=2) + "\n")
    for f in COPY_FILES:
        if (args.src / f).exists():
            shutil.copy2(args.src / f, args.dst / f)
    trunk = "bf16" if args.bits == 16 else f"{args.bits}-bit affine, group 64"
    width = "MXFP4 experts, bf16 trunk" if args.bits == 16 else f"MXFP4 experts, {args.bits}-bit trunk"
    (args.dst / "README.md").write_text(README.format(width=width, trunk=trunk))
    print(f"done in {time.time() - t0:.0f}s: {writer.total / 2**30:.2f} GiB -> {args.dst}")


if __name__ == "__main__":
    sys.exit(main())
