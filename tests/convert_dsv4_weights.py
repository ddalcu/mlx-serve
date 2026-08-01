#!/usr/bin/env python3
"""Convert deepseek-ai/DeepSeek-V4-Flash (fp8 spine + fp4 experts) into our
mixed-quant MLX mirror.

Mix (v1, sized to beat antirez's 80.8 GB IQ2XXS GGUF on every tensor class
while fitting a 128 GB Mac):
  - routed expert w1/w3 (gate/up): affine 2-bit gs64  (2.5 bpw vs IQ2_XXS 2.06)
  - routed expert w2 (down):       affine 3-bit gs64  (3.5 bpw vs Q2_K 2.56)
  - attention / shared experts / indexer wq_b: affine 8-bit gs64 (~Q8_0)
  - embed + head: affine 8-bit gs64
  - compressor wkv/wgate, indexer weights_proj, gate.weight: bf16 (fp32-sensitive
    compression path; router)
  - norms / hc_* / ape / attn_sink / gate.bias / tid2eid: verbatim
Experts are stacked into [256, out, in] banks (gather_qmm layout):
  layers.N.ffn.experts.{w1,w2,w3}.{weight,scales,biases}

Requantization dequantizes the source EXACTLY (e4m3/e8m0/e2m1 are all exact in
bf16: <=8 mantissa bits, shared exponent range) and re-packs with mx.quantize,
so the mirror is engine-native affine with bf16 scales+biases.

Usage:
  python3 tests/convert_dsv4_weights.py --self-test
  python3 tests/convert_dsv4_weights.py --src ~/.mlx-serve/staging/DeepSeek-V4-Flash-0731 \
      --out ~/.mlx-serve/models/ddalcu/DeepSeek-V4-Flash-0731-MLX-Serve-mixed-2-3-8bit [--dry-run] [--groups N-M]
"""

import argparse
import gc
import json
import os
import re
import struct
import sys
import tempfile

import numpy as np

# mlx imported lazily so --self-test's pure-numpy parts run anywhere.
_mx = None


def mx():
    global _mx
    if _mx is None:
        import mlx.core

        _mx = mlx.core
    return _mx


# ============================================================
# Source formats: e4m3 / e8m0 / e2m1 decode (numpy LUTs)
# ============================================================

def build_e4m3_lut():
    """256-entry f32 LUT for float8_e4m3fn (bias 7, no inf, NaN at S.1111.111)."""
    lut = np.empty(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exp = (code >> 3) & 0x0F
        man = code & 0x07
        if exp == 0x0F and man == 0x07:
            lut[code] = np.nan
        elif exp == 0:
            lut[code] = sign * (man / 8.0) * 2.0 ** (-6)
        else:
            lut[code] = sign * (1.0 + man / 8.0) * 2.0 ** (exp - 7)
    return lut


def build_e8m0_lut():
    """256-entry f32 LUT for float8_e8m0fnu: 2^(code-127), NaN at 255."""
    codes = np.arange(256, dtype=np.float32)
    lut = np.exp2(codes - 127.0).astype(np.float32)
    lut[255] = np.nan
    return lut


# e2m1 nibble table, verbatim from the reference convert.py (FP4_TABLE).
E2M1_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

E4M3_LUT = build_e4m3_lut()
E8M0_LUT = build_e8m0_lut()


def dequant_fp8_block(w_u8, scale_u8, block=128):
    """Dequantize e4m3 weight [out, in] with e8m0 scales per [block, block]."""
    w = E4M3_LUT[w_u8]
    s = E8M0_LUT[scale_u8]
    assert not np.isnan(s).any(), "e8m0 scale contains NaN code 255"
    out_dim, in_dim = w.shape
    s_full = np.repeat(np.repeat(s, block, axis=0), block, axis=1)
    return w * s_full[:out_dim, :in_dim]


def dequant_fp4(w_i8, scale_u8, group=32):
    """Dequantize packed e2m1 weight [out, in//2] (LOW nibble first) with e8m0
    scales [out, in//group] to f32 [out, in]."""
    b = w_i8.view(np.uint8)
    lo = E2M1_TABLE[b & 0x0F]
    hi = E2M1_TABLE[(b >> 4) & 0x0F]
    w = np.stack([lo, hi], axis=-1).reshape(b.shape[0], b.shape[1] * 2)
    s = E8M0_LUT[scale_u8]
    assert not np.isnan(s).any(), "e8m0 scale contains NaN code 255"
    return w * np.repeat(s, group, axis=1)


# ============================================================
# Raw safetensors I/O (numpy has no fp8 dtypes; read bytes ourselves)
# ============================================================

DTYPE_BYTES = {
    "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1,
    "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "U32": 4,
}


class ShardReader:
    """Random-access tensor reader over one safetensors file."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(hlen))
        self.data_off = 8 + hlen
        self.header.pop("__metadata__", None)

    def names(self):
        return list(self.header.keys())

    def read(self, name):
        """Returns (numpy array in a raw container dtype, safetensors dtype str)."""
        meta = self.header[name]
        dt, shape = meta["dtype"], meta["shape"]
        begin, end = meta["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.data_off + begin)
            buf = f.read(end - begin)
        nbytes = int(np.prod(shape)) * DTYPE_BYTES[dt] if shape else DTYPE_BYTES[dt]
        assert len(buf) == nbytes, f"{name}: expected {nbytes} bytes, got {len(buf)}"
        np_dt = {
            "F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "U8": np.uint8,
            "I8": np.int8, "BF16": np.uint16, "F16": np.float16,
            "F32": np.float32, "I32": np.int32, "I64": np.int64, "U32": np.uint32,
        }[dt]
        arr = np.frombuffer(buf, dtype=np_dt).reshape(shape)
        return arr, dt


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_u16(f32):
    """Round-to-nearest-even f32 -> bf16 bit pattern (matches hardware/mlx)."""
    u = f32.astype(np.float32).view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def write_safetensors_raw(path, tensors, metadata=None):
    """Write tensors given as (dtype_str, shape, raw_bytes) triples."""
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for name, (dt, shape, raw) in tensors.items():
        header[name] = {"dtype": dt, "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    hjson = json.dumps(header).encode()
    pad = (8 - (len(hjson) % 8)) % 8
    hjson += b" " * pad
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for _, (_, _, raw) in tensors.items():
            f.write(raw)
    os.replace(tmp, path)


# ============================================================
# Requantization to MLX affine
# ============================================================

def mlx_affine_quant(w_f32, bits, group_size=64):
    """f32 -> (packed u32, bf16 scales, bf16 biases) via mx.quantize on bf16 input.

    Returns raw-bytes triples ready for write_safetensors_raw."""
    m = mx()
    wb = m.array(f32_to_bf16_u16(np.ascontiguousarray(w_f32)).view(np.uint16)).view(m.bfloat16)
    wb = wb.reshape(w_f32.shape)
    wq, scales, biases = m.quantize(wb, group_size=group_size, bits=bits)
    m.eval(wq, scales, biases)
    out = (
        ("U32", wq.shape, np.array(wq, copy=False).tobytes()),
        ("BF16", scales.shape, np.array(scales.view(m.uint16), copy=False).tobytes()),
        ("BF16", biases.shape, np.array(biases.view(m.uint16), copy=False).tobytes()),
    )
    del wq, scales, biases, wb
    return out


def mlx_affine_dequant_f32(wq_u32, scales_u16, biases_u16, bits, group_size=64):
    """Inverse of mlx_affine_quant, for verification."""
    m = mx()
    wq = m.array(wq_u32)
    scales = m.array(scales_u16).view(m.bfloat16)
    biases = m.array(biases_u16).view(m.bfloat16)
    w = m.dequantize(wq, scales, biases, group_size=group_size, bits=bits)
    m.eval(w)
    return np.array(w.astype(m.float32), copy=False)


# ============================================================
# Tensor classification -> conversion action
# ============================================================

EXPERT_RE = re.compile(r"^(?P<pfx>(?:layers|mtp)\.\d+\.ffn)\.experts\.(?P<eid>\d+)\.(?P<proj>w[123])\.(?P<kind>weight|scale)$")

# fp8 spine linears -> affine 8-bit. `main_proj` is DSpark's
# [dim*len(target_layers), dim] entry projection; the superseded preview's
# e_proj/h_proj pair is deliberately NOT listed, so a preview checkpoint fails
# loud on `unclassified tensor` instead of converting into a mirror no
# supported engine path can drive.
SPINE_8BIT_RE = re.compile(
    r"\.(wq_a|wq_b|wkv|wo_a|wo_b|shared_experts\.w[123]|main_proj)\.(weight|scale)$"
)
# bf16 linears quantized to 8-bit
BF16_8BIT_RE = re.compile(r"^(embed|head)\.weight$")
# bf16/f32 tensors kept verbatim (compressor path is fp32-sensitive; router raw)
KEEP_RE = re.compile(
    r"(\.compressor\.(ape|norm\.weight|wgate\.weight|wkv\.weight)$"
    r"|\.indexer\.weights_proj\.weight$"
    r"|\.gate\.(weight|bias|tid2eid)$"
    r"|(^|\.)(attn_norm|ffn_norm|q_norm|kv_norm|main_norm|norm)\.weight$"
    r"|\.attn_sink$"
    # DSpark heads (0731): markov_w1 is an EMBEDDING (gathered, never a matmul
    # operand — quantizing it would return packed garbage, the converter's
    # lookup-table rule) and markov_w2 / confidence proj are small enough that
    # dense costs ~130 MB total. All three feed draft ACCEPTANCE, so they stay
    # verbatim.
    r"|\.markov_head\.markov_w[12]\.weight$"
    r"|\.confidence_head\.proj\.weight$"
    r"|hc_(attn|ffn|head)_(fn|base|scale)$)"
)

SOURCE_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"

README = """---
license: mit
base_model: {base}
base_model_relation: quantized
library_name: mlx-serve
tags:
  - mlx
  - mlx-serve
  - deepseek
  - quantized
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731 — mixed 2/3/8-bit for mlx-serve

A {gb} GB mixed-precision MLX conversion of
[{base}](https://huggingface.co/{base}), built to run the full 284B-A13B model
on a single Apple Silicon Mac with **128 GB or more** of unified memory.

Runs on [**mlx-serve**](https://github.com/ddalcu/mlx-serve) — a native Zig
inference server for MLX models on Apple Silicon, with no Python in the serving
path. It speaks the OpenAI, Anthropic and Ollama HTTP APIs, so existing clients
(Claude Code, pi, opencode, Open WebUI, …) point at it unchanged.

```bash
mlx-serve --model <this-repo-dir> --serve --port 11434
```

DeepSeek-V4-Flash's architecture is implemented natively in mlx-serve: MQA over
a single 512-dim latent, window-128 raw attention plus gated-pooling compressed
history with a top-512 indexer, per-head attention sinks, Sinkhorn hyper-
connections, hash-routed early MoE layers, and the DSML tool-call format. No
llama.cpp, no GGUF conversion, no Python runtime.

## How it was quantized

Every tensor class is sized by what it costs and how much it matters, rather
than one global bit width:

| Tensors | Precision |
|---|---|
| Routed expert gate/up (`w1`/`w3`) | affine **2-bit**, group size 64 |
| Routed expert down (`w2`) | affine **3-bit**, group size 64 |
| Attention, shared experts, indexer, `main_proj` | affine **8-bit**, group size 64 |
| Embedding + LM head | affine **8-bit**, group size 64 |
| DSpark draft stages (`mtp.*`) — experts | affine **4-bit**, group size 64 |
| Compressor `wkv`/`wgate`, indexer `weights_proj`, router `gate.weight` | bf16 |
| Norms, hyper-connection params, `ape`, attention sinks, router bias, hash table | verbatim |

Two choices worth explaining. The down-projection keeps **3-bit** while gate/up
drop to 2-bit: it is the most quantization-sensitive of the three, and decode on
this model is latency-bound rather than bandwidth-bound, so shrinking it further
buys disk rather than speed. The DSpark draft stages keep **4-bit** even though
the trunk goes lower: they are a rounding error on disk, and a draft the trunk
rejects costs a full verify forward, so their quality multiplies throughput.

The compressor path is fp32-sensitive by design and the router is read raw, so
neither is quantized. Lookup tables (embeddings, the token→expert hash, DSpark's
Markov table) are never packed — they are gathered, not multiplied.

Conversion is exact where it can be: the source's fp8 (e4m3 + e8m0 block scales)
and fp4 (e2m1 + e8m0 group scales) formats all fit losslessly in bf16, so the
weights are decoded exactly and re-packed with MLX's own affine quantizer. The
mirror is engine-native — no dequantize-on-load step at runtime.

## What is included

Weights, tokenizer, and a chat template transcribed from the release's own
`encoding/encoding_dsv4.py` and verified **byte-exact** against it across chat
and thinking modes, tool definitions, DSML tool-call history, multi-turn
drop-thinking, and all three reasoning-effort levels. `generation_config.json`
carries the reference's own default sampling (temperature 0.6), not the wild
1.0/1.0 signature the source ships.

DSpark speculative-decoding weights (3 draft stages) are included and converted;
mlx-serve does not drive them yet.

## Requirements

- Apple Silicon Mac, **128 GB+** unified memory (~110 GB resident)
- macOS 26.2 or newer
- [mlx-serve](https://github.com/ddalcu/mlx-serve)

Built with `tests/convert_dsv4_weights.py` from the mlx-serve repo.
"""

EXPERT_BITS = {"w1": 2, "w2": 3, "w3": 2}
# DSpark draft stages (`mtp.*`) carry their own expert banks and are TINY next
# to the trunk (~7 GB of source against 159), while their output quality is
# multiplicative on decode throughput: a draft the trunk rejects costs a whole
# verify forward. So they get uniform 4-bit rather than the trunk's 2/3-bit
# mix — ~1.5 GB more on disk to protect the acceptance rate.
MTP_EXPERT_BITS = {"w1": 4, "w2": 4, "w3": 4}
GROUP_SIZE = 64


def expert_bits(pfx, proj):
    """Expert bit width by MODULE: draft stages are not the trunk."""
    return (MTP_EXPERT_BITS if pfx.startswith("mtp.") else EXPERT_BITS)[proj]


def classify(name):
    """Returns one of: ('expert', pfx, eid, proj, kind) | ('spine8', base) |
    ('bf16_8', base) | ('keep',) — base = tensor name without .weight/.scale."""
    m = EXPERT_RE.match(name)
    if m:
        return ("expert", m.group("pfx"), int(m.group("eid")), m.group("proj"), m.group("kind"))
    if KEEP_RE.search(name):
        return ("keep",)
    m = SPINE_8BIT_RE.search(name)
    if m:
        return ("spine8", name.rsplit(".", 1)[0])
    if BF16_8BIT_RE.match(name):
        return ("bf16_8", name.rsplit(".", 1)[0])
    # indexer.wq_b is fp8 (matches spine list? no: indexer.wq_b -> ".wq_b." matches SPINE_8BIT_RE)
    raise ValueError(f"unclassified tensor: {name}")


def group_of(name):
    """Output shard group: 'layer.N' | 'mtp' | 'top'."""
    m = re.match(r"^layers\.(\d+)\.", name)
    if m:
        return f"layer.{m.group(1)}"
    if name.startswith("mtp."):
        return "mtp"
    return "top"


# ============================================================
# Conversion driver
# ============================================================

def expected_groups(n_layers):
    return [f"layer.{i}" for i in range(n_layers)] + ["mtp", "top"]


def plan_groups(weight_map):
    groups = {}
    for name in weight_map:
        groups.setdefault(group_of(name), []).append(name)
    return groups


def bits_for(name):
    cls = classify(name)
    if cls[0] == "expert":
        return expert_bits(cls[1], cls[3])
    if cls[0] in ("spine8", "bf16_8"):
        return 8
    return None


def dry_run_size(src):
    """Exact output byte count from the index + shard headers."""
    wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    readers = {}
    total = 0
    by_class = {}
    for name, shard in sorted(wm.items()):
        if shard not in readers:
            readers[shard] = ShardReader(os.path.join(src, shard))
        meta = readers[shard].header[name]
        dt, shape = meta["dtype"], meta["shape"]
        cls = classify(name)
        if cls[0] == "keep":
            n = int(np.prod(shape)) * DTYPE_BYTES[dt]
            key = "keep"
        elif cls[0] == "expert":
            if cls[4] == "scale":
                continue
            out_d, in_d = shape[0], shape[1] * 2  # packed fp4
            bits = expert_bits(cls[1], cls[3])
            n = out_d * (in_d * bits // 8 + in_d // GROUP_SIZE * 4)
            key = f"expert.{cls[3]}({bits}b)"
        else:
            if name.endswith(".scale"):
                continue
            out_d, in_d = shape
            n = out_d * (in_d + in_d // GROUP_SIZE * 4)
            key = "8bit"
        total += n
        by_class[key] = by_class.get(key, 0) + n
    return total, by_class


def convert_tensor(name, arr, dt, out_tensors, verify_stats=None):
    """Convert a single non-expert tensor into out_tensors (raw triples)."""
    cls = classify(name)
    if cls[0] == "keep":
        out_tensors[name] = (dt, arr.shape, arr.tobytes())
        return
    if name.endswith(".scale"):
        return  # consumed alongside its .weight
    raise AssertionError(f"convert_tensor called on {name} without pairing")


def convert(src, out, only_groups=None, verify=False):
    os.makedirs(out, exist_ok=True)
    wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(src, "config.json")))
    n_layers = cfg["num_hidden_layers"]
    groups = plan_groups(wm)
    order = [g for g in expected_groups(n_layers) if g in groups]
    assert set(order) == set(groups), f"unexpected groups: {set(groups) - set(order)}"
    if only_groups is not None:
        order = [g for g in order if g in only_groups]

    readers = {}

    def read(name):
        shard = wm[name]
        if shard not in readers:
            readers[shard] = ShardReader(os.path.join(src, shard))
        return readers[shard].read(name)

    manifest_path = os.path.join(out, ".convert-manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    quant_cfg = {"group_size": GROUP_SIZE, "bits": 8, "mode": "affine"}
    index_map = {}
    total_size = 0

    for gi, gname in enumerate(order):
        shard_file = f"model-{gname.replace('.', '-')}.safetensors"
        shard_path = os.path.join(out, shard_file)
        names = sorted(groups[gname])
        if manifest.get(gname) and os.path.exists(shard_path) and \
                os.path.getsize(shard_path) == manifest[gname]["size"]:
            print(f"[{gi+1}/{len(order)}] {gname}: already converted, skipping")
            for tn in manifest[gname]["tensors"]:
                index_map[tn] = shard_file
            total_size += manifest[gname]["size"]
            continue

        out_tensors = {}
        # 1) stacked expert banks
        expert_members = [n for n in names if EXPERT_RE.match(n)]
        # A group can hold MORE THAN ONE expert prefix: every trunk group is a
        # single layer, but the `mtp` group carries all of DSpark's stages
        # (mtp.0/1/2.ffn). Taking the prefix from members[0] silently dropped
        # stages 1 and 2 — the conversion succeeded, the index looked sane,
        # and only a per-stage tensor count showed the missing banks.
        n_exp = cfg["n_routed_experts"]
        for pfx in sorted({EXPERT_RE.match(n).group("pfx") for n in expert_members}):
            for proj in ("w1", "w2", "w3"):
                    bits = expert_bits(pfx, proj)
                    wq_l, sc_l, bi_l = [], [], []
                    for eid in range(n_exp):
                        w_i8, _ = read(f"{pfx}.experts.{eid}.{proj}.weight")
                        s_u8, _ = read(f"{pfx}.experts.{eid}.{proj}.scale")
                        f32 = dequant_fp4(w_i8, s_u8)
                        (wq, sc, bi) = mlx_affine_quant(f32, bits, GROUP_SIZE)
                        wq_l.append(wq); sc_l.append(sc); bi_l.append(bi)
                        del f32, w_i8, s_u8
                    for kind, parts in (("weight", wq_l), ("scales", sc_l), ("biases", bi_l)):
                        dtype = parts[0][0]
                        shape = [n_exp] + list(parts[0][1])
                        raw = b"".join(p[2] for p in parts)
                        out_tensors[f"{pfx}.experts.{proj}.{kind}"] = (dtype, shape, raw)
                    quant_cfg[f"{pfx}.experts.{proj}"] = {"group_size": GROUP_SIZE, "bits": bits, "mode": "affine"}
                    del wq_l, sc_l, bi_l
                    gc.collect()

        # 2) everything else
        for name in names:
            if EXPERT_RE.match(name):
                continue
            cls = classify(name)
            if name.endswith(".scale"):
                continue
            arr, dt = read(name)
            if cls[0] == "keep":
                out_tensors[name] = (dt, arr.shape, arr.tobytes())
                continue
            base = cls[1]
            if cls[0] == "spine8":
                s_u8, sdt = read(base + ".scale")
                assert dt == "F8_E4M3" and sdt == "F8_E8M0", f"{name}: {dt}/{sdt}"
                f32 = dequant_fp8_block(arr, s_u8)
            else:  # bf16_8
                assert dt == "BF16", f"{name}: {dt}"
                f32 = bf16_to_f32(arr)
            wq, sc, bi = mlx_affine_quant(f32, 8, GROUP_SIZE)
            out_tensors[base + ".weight"] = wq
            out_tensors[base + ".scales"] = sc
            out_tensors[base + ".biases"] = bi
            quant_cfg[base] = {"group_size": GROUP_SIZE, "bits": 8, "mode": "affine"}
            if verify:
                back = mlx_affine_dequant_f32(
                    np.frombuffer(wq[2], dtype=np.uint32).reshape(wq[1]),
                    np.frombuffer(sc[2], dtype=np.uint16).reshape(sc[1]),
                    np.frombuffer(bi[2], dtype=np.uint16).reshape(bi[1]), 8, GROUP_SIZE)
                cos = float((f32 * back).sum() / (np.linalg.norm(f32) * np.linalg.norm(back) + 1e-30))
                assert cos > 0.999, f"{name}: 8-bit requant cosine {cos}"
            del f32, arr
        write_safetensors_raw(shard_path, out_tensors, metadata={"format": "mlx"})
        size = os.path.getsize(shard_path)
        manifest[gname] = {"size": size, "tensors": sorted(out_tensors.keys())}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        for tn in out_tensors:
            index_map[tn] = shard_file
        total_size += size
        print(f"[{gi+1}/{len(order)}] {gname}: {len(out_tensors)} tensors, {size/1e9:.2f} GB (total {total_size/1e9:.1f} GB)")
        out_tensors.clear()
        readers.clear()
        gc.collect()
        try:
            mx().clear_cache()
        except AttributeError:
            pass

    if only_groups is None:
        # index + config + tokenizer files
        with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": index_map}, f)
        # Rebuild the per-path quantization dict from the FULL index — the
        # per-run accumulator misses groups converted by earlier --groups
        # invocations (they skip via manifest and never re-classify).
        stacked_re = re.compile(r"\.experts\.(w[123])\.weight$")
        quant_cfg = {"group_size": GROUP_SIZE, "bits": 8, "mode": "affine"}
        for tn in index_map:
            if tn.endswith(".weight") and tn[:-7] + ".scales" in index_map:
                m = stacked_re.search(tn)
                # `tn` is the FULL stacked path, so its own `mtp.` prefix is
                # what picks the draft-stage widths — the engine dequants from
                # this dict, so a trunk-vs-draft mixup here is silent garbage.
                bits = expert_bits(tn, m.group(1)) if m else 8
                quant_cfg[tn[:-7]] = {"group_size": GROUP_SIZE, "bits": bits, "mode": "affine"}
        out_cfg = dict(cfg)
        out_cfg.pop("quantization_config", None)  # fp8 source config must not leak
        out_cfg["quantization"] = quant_cfg
        out_cfg["mlx_serve_converter"] = {
            "source": "deepseek-ai/DeepSeek-V4-Flash",
            "mix": "experts w1/w3 2-bit gs64, w2 3-bit gs64; spine/embed/head 8-bit gs64; compressor/router bf16",
        }
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(out_cfg, f, indent=2)
        import shutil
        if os.path.exists(os.path.join(src, "tokenizer.json")):
            shutil.copyfile(os.path.join(src, "tokenizer.json"), os.path.join(out, "tokenizer.json"))
        # tokenizer_config gets our chat template injected (validated against
        # the release's own encoding_dsv4.py golden pairs — see
        # src/fixtures/dsv4_chat_template.jinja; the mlx-community template
        # double-emits </think> on assistant history turns, ours does not).
        if os.path.exists(os.path.join(src, "tokenizer_config.json")):
            tc = json.load(open(os.path.join(src, "tokenizer_config.json")))
            tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "src", "fixtures", "dsv4_chat_template.jinja")
            if os.path.exists(tpl_path):
                tc["chat_template"] = open(tpl_path).read()
            with open(os.path.join(out, "tokenizer_config.json"), "w") as f:
                json.dump(tc, f, indent=2, ensure_ascii=False)
        # The source generation_config is the wild-sampling signature (1.0/1.0)
        # while the reference generate.py defaults to temperature 0.6 — ship
        # the reference's default so omitted request fields sample sanely.
        gen = {"bos_token_id": cfg.get("bos_token_id", 0), "eos_token_id": cfg.get("eos_token_id", 1),
               "do_sample": True, "temperature": 0.6, "top_p": 1.0}
        with open(os.path.join(out, "generation_config.json"), "w") as f:
            json.dump(gen, f, indent=2)
        with open(os.path.join(out, "README.md"), "w") as f:
            f.write(README.format(base=SOURCE_REPO, gb=f"{total_size/1e9:.1f}"))
        print(f"DONE: {total_size/1e9:.1f} GB -> {out}")


# ============================================================
# Self-tests
# ============================================================

def self_test():
    failures = []

    def check(cond, label):
        print(("PASS " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    # e4m3 LUT spot values
    check(E4M3_LUT[0x00] == 0.0, "e4m3 0x00 == 0")
    check(E4M3_LUT[0x38] == 1.0, "e4m3 0x38 == 1.0")
    check(E4M3_LUT[0x7E] == 448.0, "e4m3 0x7E == 448 (max)")
    check(np.isnan(E4M3_LUT[0x7F]) and np.isnan(E4M3_LUT[0xFF]), "e4m3 NaN codes")
    check(E4M3_LUT[0x01] == 2.0 ** -9, "e4m3 smallest subnormal 2^-9")
    check(np.all(E4M3_LUT[0x80:0xFF] == -E4M3_LUT[0x00:0x7F]), "e4m3 sign symmetry")
    pos = E4M3_LUT[1:0x7F]
    check(np.all(np.diff(pos) > 0), "e4m3 positive monotonic")

    # e8m0 LUT
    check(E8M0_LUT[127] == 1.0 and E8M0_LUT[128] == 2.0 and E8M0_LUT[126] == 0.5, "e8m0 spot values")
    check(np.isnan(E8M0_LUT[255]), "e8m0 NaN at 255")

    # e2m1 table matches reference FP4_TABLE ordering
    check(E2M1_TABLE[7] == 6.0 and E2M1_TABLE[15] == -6.0 and E2M1_TABLE[1] == 0.5,
          "e2m1 table spot values")

    # fp8 block dequant: 256x256 with distinct block scales
    w = np.full((256, 256), 0x38, dtype=np.uint8)  # all 1.0
    s = np.array([[127, 128], [126, 129]], dtype=np.uint8)  # 1,2 / 0.5,4
    d = dequant_fp8_block(w, s, block=128)
    check(d[0, 0] == 1.0 and d[0, 255] == 2.0 and d[255, 0] == 0.5 and d[255, 255] == 4.0,
          "fp8 block dequant broadcast")

    # fp4 dequant: nibble order LOW first
    # byte 0x21 -> low=1 (0.5), high=2 (1.0)
    w4 = np.array([[0x21] * 16], dtype=np.int8)  # in_dim 32 = one group
    s4 = np.array([[128]], dtype=np.uint8)  # scale 2.0
    d4 = dequant_fp4(w4, s4, group=32)
    check(d4.shape == (1, 32) and d4[0, 0] == 1.0 and d4[0, 1] == 2.0,
          "fp4 dequant low-nibble-first + scale")

    # bf16 round-trip
    x = np.array([1.0, -2.5, 3.14159, 65504.0, 1e-8], dtype=np.float32)
    rt = bf16_to_f32(f32_to_bf16_u16(x))
    check(np.allclose(rt, x, rtol=1e-2), "bf16 round trip")
    check(bf16_to_f32(f32_to_bf16_u16(np.array([1.0], dtype=np.float32)))[0] == 1.0,
          "bf16 exact 1.0")

    # affine quant round-trip via mlx
    rng = np.random.default_rng(7)
    w = rng.standard_normal((64, 128)).astype(np.float32)
    for bits, min_cos in ((8, 0.9999), (3, 0.98), (2, 0.92)):
        wq, sc, bi = mlx_affine_quant(w, bits, 64)
        back = mlx_affine_dequant_f32(
            np.frombuffer(wq[2], dtype=np.uint32).reshape(wq[1]),
            np.frombuffer(sc[2], dtype=np.uint16).reshape(sc[1]),
            np.frombuffer(bi[2], dtype=np.uint16).reshape(bi[1]), bits, 64)
        cos = float((w * back).sum() / (np.linalg.norm(w) * np.linalg.norm(back) + 1e-30))
        check(cos > min_cos, f"affine {bits}-bit round trip cosine {cos:.4f} > {min_cos}")

    # safetensors writer -> reader round trip incl. fp8 dtype
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.safetensors")
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        b = np.arange(8, dtype=np.uint8).reshape(2, 4)
        write_safetensors_raw(p, {
            "a": ("F32", a.shape, a.tobytes()),
            "b": ("F8_E4M3", b.shape, b.tobytes()),
        }, metadata={"format": "mlx"})
        r = ShardReader(p)
        ra, _ = r.read("a")
        rb, dtb = r.read("b")
        check(np.array_equal(ra, a) and np.array_equal(rb.view(np.uint8), b) and dtb == "F8_E4M3",
              "raw safetensors writer/reader round trip")
        # mlx can read our written file too
        try:
            loaded = mx().load(p)
            check(np.array_equal(np.array(loaded["a"]), a), "mlx loads our safetensors")
        except Exception as e:  # fp8 tensor may be rejected by mlx load
            check(False, f"mlx load: {e}")

    # end-to-end mini conversion with a fabricated 1-layer checkpoint
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src"); os.makedirs(src)
        n_exp, d_in, d_moe = 4, 128, 64
        rng = np.random.default_rng(3)
        tensors = {}
        # experts fp4: random codes, scale 1.0. TWO DSpark stages share the
        # single `mtp` output group — the stacking loop took its prefix from
        # members[0] and silently dropped every other stage (live 2026-07-31:
        # mtp.1/mtp.2 lost all 256 expert banks each, and the conversion,
        # the index and the byte total all looked fine).
        for pfx in ("layers.0.ffn", "mtp.0.ffn", "mtp.1.ffn"):
            for e in range(n_exp):
                for proj, (o, i) in (("w1", (d_moe, d_in)), ("w3", (d_moe, d_in)), ("w2", (d_in, d_moe))):
                    codes = rng.integers(0, 256, size=(o, i // 2), dtype=np.uint8).astype(np.int8)
                    tensors[f"{pfx}.experts.{e}.{proj}.weight"] = ("I8", (o, i // 2), codes.tobytes())
                    s = np.full((o, i // 32), 127, dtype=np.uint8)
                    tensors[f"{pfx}.experts.{e}.{proj}.scale"] = ("F8_E8M0", s.shape, s.tobytes())
        # one fp8 spine linear (128x128 -> single scale block)
        wq_codes = rng.integers(0, 0x7F, size=(d_in, d_in), dtype=np.uint8)
        tensors["layers.0.attn.wq_a.weight"] = ("F8_E4M3", (d_in, d_in), wq_codes.tobytes())
        tensors["layers.0.attn.wq_a.scale"] = ("F8_E8M0", (1, 1), np.array([[127]], dtype=np.uint8).tobytes())
        # keep-class tensors
        nw = rng.standard_normal(d_in).astype(np.float32)
        tensors["layers.0.attn_norm.weight"] = ("BF16", (d_in,), f32_to_bf16_u16(nw).tobytes())
        gw = rng.standard_normal((n_exp, d_in)).astype(np.float32)
        tensors["layers.0.ffn.gate.weight"] = ("BF16", (n_exp, d_in), f32_to_bf16_u16(gw).tobytes())
        # embed/head
        ew = rng.standard_normal((256, d_in)).astype(np.float32)
        tensors["embed.weight"] = ("BF16", (256, d_in), f32_to_bf16_u16(ew).tobytes())
        tensors["head.weight"] = ("BF16", (256, d_in), f32_to_bf16_u16(ew).tobytes())
        write_safetensors_raw(os.path.join(src, "model-00001.safetensors"), tensors)
        json.dump({"weight_map": {k: "model-00001.safetensors" for k in tensors}},
                  open(os.path.join(src, "model.safetensors.index.json"), "w"))
        json.dump({"num_hidden_layers": 1, "n_routed_experts": n_exp},
                  open(os.path.join(src, "config.json"), "w"))

        outd = os.path.join(td, "out")
        convert(src, outd, verify=True)
        idx = json.load(open(os.path.join(outd, "model.safetensors.index.json")))
        ocfg = json.load(open(os.path.join(outd, "config.json")))
        check("layers.0.ffn.experts.w1.weight" in idx["weight_map"], "e2e stacked bank in index")
        check(ocfg["quantization"]["layers.0.ffn.experts.w1"]["bits"] == 2
              and ocfg["quantization"]["layers.0.ffn.experts.w2"]["bits"] == 3,
              "e2e per-path quant config bits")
        check("quantization_config" not in ocfg, "e2e fp8 quantization_config dropped")
        # EVERY expert prefix in a group must be stacked, not just the first.
        for st in ("mtp.0", "mtp.1"):
            check(all(f"{st}.ffn.experts.{p}.weight" in idx["weight_map"] for p in ("w1", "w2", "w3")),
                  f"e2e {st} expert banks stacked (multi-prefix group)")
        # Draft stages carry their own (higher) widths, keyed off the path.
        check(ocfg["quantization"]["mtp.1.ffn.experts.w2"]["bits"] == MTP_EXPERT_BITS["w2"]
              and ocfg["quantization"]["layers.0.ffn.experts.w2"]["bits"] == EXPERT_BITS["w2"],
              "e2e draft stages quantized apart from the trunk")
        # load bank via mlx, dequant expert 1 w1, compare vs direct source dequant
        shard = os.path.join(outd, "model-layer-0.safetensors")
        loaded = mx().load(shard)
        m = mx()
        bank_back = m.dequantize(loaded["layers.0.ffn.experts.w1.weight"][1],
                                 loaded["layers.0.ffn.experts.w1.scales"][1],
                                 loaded["layers.0.ffn.experts.w1.biases"][1],
                                 group_size=GROUP_SIZE, bits=2)
        src_codes = np.frombuffer(tensors["layers.0.ffn.experts.1.w1.weight"][2],
                                  dtype=np.int8).reshape(d_moe, d_in // 2)
        src_scale = np.frombuffer(tensors["layers.0.ffn.experts.1.w1.scale"][2],
                                  dtype=np.uint8).reshape(d_moe, d_in // 32)
        ref = dequant_fp4(src_codes, src_scale)
        got = np.array(bank_back.astype(m.float32))
        cos = float((ref * got).sum() / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-30))
        check(cos > 0.90, f"e2e expert bank 2-bit vs source cosine {cos:.4f}")
        # keep-class survived verbatim
        kept = np.array(loaded["layers.0.attn_norm.weight"].astype(m.float32))
        check(np.allclose(kept, bf16_to_f32(f32_to_bf16_u16(nw))), "e2e keep-class verbatim")

    print(f"\n{len(failures)} failures" if failures else "\nALL SELF-TESTS PASS")
    return 1 if failures else 0


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--src", type=str)
    ap.add_argument("--out", type=str)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--groups", type=str, help="e.g. 'layer.0,layer.1' or '0-5'")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    assert args.src and args.out, "--src and --out required"
    src = os.path.expanduser(args.src)
    out = os.path.expanduser(args.out)

    if args.dry_run:
        total, by_class = dry_run_size(src)
        for k, v in sorted(by_class.items()):
            print(f"  {k:20s} {v/1e9:8.2f} GB")
        print(f"  {'TOTAL':20s} {total/1e9:8.2f} GB")
        sys.exit(0)

    only = None
    if args.groups:
        only = set()
        for part in args.groups.split(","):
            if "-" in part and not part.startswith("layer") and not part.startswith("mtp"):
                a, b = part.split("-")
                only.update(f"layer.{i}" for i in range(int(a), int(b) + 1))
            else:
                only.add(part)
    convert(src, out, only_groups=only, verify=args.verify)
