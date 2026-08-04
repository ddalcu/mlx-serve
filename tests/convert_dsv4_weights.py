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
        # The converter runs beside the live server: every residual mx call
        # (spine 8-bit requant, verification dequants) stays off the GPU.
        _mx.set_default_device(_mx.cpu)
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

# DeepSeek-V4-Flash-0731 — iQ-MLX {bpw} bpw

An **iQ-MLX** conversion of [{base}](https://huggingface.co/{base}) at
**{bpw} bits per weight** ({gb} GB), built to run the full 284B-A13B model on
a single Apple Silicon Mac with **128 GB or more** of unified memory.

## What is iQ-MLX

iQ-MLX brings llama.cpp's imatrix school of quantization to native MLX affine
layouts: per-input-channel importance is measured on real calibration traffic
(mean-squared activation over millions of tokens, per expert), every
quantization group's scale/bias is chosen by an importance-weighted
least-squares search instead of min/max, and the per-layer/per-projection bit
widths themselves are allocated by a greedy error-per-byte knapsack over the
measured error surface — the byte budget goes exactly where the calibration
says it buys the most quality. The packs stay byte-compatible with stock MLX
affine, so the result loads like any MLX model: no custom codec, no
dequantize-on-load, no runtime cost.

How it differs from what you may know:

- **vs plain MLX 4bit / mixed_2_6**: those are uncalibrated min/max with
  static recipes; iQ-MLX measures which channels and layers matter and fits
  scales to that.
- **vs DWQ**: DWQ distills against the fp teacher on a GPU (hours of training);
  iQ-MLX is a CPU-only search you can run on the same box that serves the
  model, at comparable calibration quality for MoE experts.
- **vs AWQ**: activation-aware scale folding is a min/max compensation. We
  measured it against the weighted search on this model: redundant on gate/up
  (under 1 percent) and harmful on down projections — so iQ-MLX deliberately
  does not fold.
- **vs GGUF IQ/UD**: same philosophy (imatrix calibration, dynamic per-layer
  bits), but emitting native MLX affine tensors instead of GGUF — no llama.cpp
  in the serving path.

The bpw figure in the name is the honest size metric: stored bits (weights +
scales + biases) divided by parameters, comparable across iQ-MLX builds and
against GGUF tiers (IQ2_XXS is 2.06, Q4_K_M is about 4.8). Higher bpw = less
quantized.

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
| Routed expert gate/up (`w1`/`w3`), early layers | affine **2-bit**, group size 128, **imatrix-calibrated** |
| Routed expert gate/up (`w1`/`w3`), layers 7-8 and 16-38 | affine **3-bit**, group size 128, **imatrix-calibrated** |
| Routed expert down (`w2`) | affine **3-bit**, group size 128, **imatrix-calibrated** |
| Routed experts, tail layers 39-42 (all projections) | affine **4-bit**, group size 64, **imatrix-calibrated** |
| Attention, shared experts, indexer, `main_proj` | affine **8-bit**, group size 64 |
| Embedding + LM head | affine **8-bit**, group size 64 |
| DSpark draft stages (`mtp.*`) — experts | affine **4-bit**, group size 64 |
| Compressor `wkv`/`wgate`, indexer `weights_proj`, router `gate.weight` | bf16 |
| Norms, hyper-connection params, `ape`, attention sinks, router bias, hash table | verbatim |

The layer/projection split is not hand-picked: the whole error surface (every
layer x projection at every candidate width, sampled per expert against the
imatrix objective) was measured, and the byte budget allocated greedily by
error reduction per byte. The data put nearly all of it into gate/up on the
later two thirds of the stack — consistent with what agent-level testing
showed about late layers and decision quality. The 4-bit tail (layers 39-42)
is kept from the previous build: it is what eliminated turn-level agent
repetition loops in A/B testing.

The routed experts — 277B of the 284B — are quantized with the iQ-MLX
activation-calibrated search rather than plain min/max: per-input-channel
importance comes from an importance matrix collected over **2.9M tokens** of
chat-formatted traffic through the 0731 weights themselves (per-expert channel
granularity), and each quantization group's scale/bias pair is chosen by a
weighted multi-start search with alternating least-squares refinement (the
llama.cpp `make_qkx2_quants` pattern). Channels that actually fire reconstruct
better; at 2-3 bits this is worth more than finer group granularity, which is
why the experts use group size 128 and spend the saved bytes on extra bits
where the error surface says they matter.

One more choice worth explaining: the DSpark draft stages keep **4-bit**,
uncalibrated (the imatrix does not cover them). They are a rounding error on
disk, and a draft the trunk rejects costs a full verify forward, so their
quality multiplies throughput.

The compressor path is fp32-sensitive by design and the router is read raw, so
neither is quantized. Lookup tables (embeddings, the token→expert hash, DSpark's
Markov table) are never packed — they are gathered, not multiplied.

Conversion is exact where it can be: the source's fp8 (e4m3 + e8m0 block scales)
and fp4 (e2m1 + e8m0 group scales) formats all fit losslessly in bf16, so the
weights are decoded exactly before requantization. The calibrated expert packs
are byte-compatible with MLX's affine layout, so the mirror is engine-native —
no dequantize-on-load step at runtime.

## What is included

Weights, tokenizer, and a chat template transcribed from the release's own
`encoding/encoding_dsv4.py` and verified **byte-exact** against it across chat
and thinking modes, tool definitions, DSML tool-call history, multi-turn
drop-thinking, and all three reasoning-effort levels. `generation_config.json`
carries the reference's own default sampling (temperature 0.6), not the wild
1.0/1.0 signature the source ships.

DSpark speculative-decoding weights (3 draft stages) are included and
converted; mlx-serve drives them with `--dspark` (block-parallel speculative
decode, greedy and sampled).

## Requirements

- Apple Silicon Mac, **128 GB+** unified memory (~118 GB resident; raise the
  GPU limit with `sudo sysctl iogpu.wired_limit_mb=124000`). On 128 GB
  machines mlx-serve auto-disables DSpark when the stages don't fit and
  serves serial — `--dspark` engages on larger machines (192 GB+).
- macOS 26.2 or newer
- [mlx-serve](https://github.com/ddalcu/mlx-serve)

Built with `tests/convert_dsv4_weights.py` from the mlx-serve repo (the
iQ-MLX pipeline: imatrix parser, weighted affine search, and the greedy
bit-allocation sweep).
"""

# The release's own headline parameter count (284B-A13B), EXCLUDING the DSpark
# draft stages — the bpw denominator must match what the trunk bytes cover.
N_PARAMS_MAIN = 284_000_000_000

EXPERT_BITS = {"w1": 2, "w2": 3, "w3": 2}
# DSpark draft stages (`mtp.*`) carry their own expert banks and are TINY next
# to the trunk (~7 GB of source against 159), while their output quality is
# multiplicative on decode throughput: a draft the trunk rejects costs a whole
# verify forward. So they get uniform 4-bit rather than the trunk's 2/3-bit
# mix — ~1.5 GB more on disk to protect the acceptance rate.
MTP_EXPERT_BITS = {"w1": 4, "w2": 4, "w3": 4}
# Spine/embed/head and the DSpark draft stages keep gs 64 (the draft shard
# must stay byte-identical across imatrix rebuilds). Trunk routed experts
# moved to gs 128 in the imatrix round: at 2-3 bits, bit width beats group
# granularity on this model's own weight distributions (3b/g128 = 0.193
# rel-err at 3.25 bpw vs 2b/g32 = 0.364 at 3.00), so g64's extra scales were
# ~8 GiB spent on the weaker lever.
SPINE_GROUP = 64
EXPERT_GROUP = 128


# Per-layer (bits, gs) override for TRUNK routed experts, set from
# --expert-override "A-B=BITS:GS". Draft stages (mtp.*) never consult it —
# their shard must stay byte-identical across rebuilds.
EXPERT_OVERRIDES = {}


def parse_expert_override(spec):
    """"37-42=4:64" (layer-wide) | "5=w2@4:128" (one projection) ->
    {layer: {proj_or_'*': (bits, gs)}}. Comma-joined parts MERGE, so a mixed
    plan ("0=w1@3:128,0=w2@4:128,39-42=4:64") lands per layer per projection."""
    out = {}
    for part in spec.split(","):
        rng_s, _, bg = part.partition("=")
        proj = "*"
        if "@" in bg:
            proj, _, bg = bg.partition("@")
            assert proj in ("w1", "w2", "w3"), f"bad projection {proj!r} in {part!r}"
        bits_s, _, gs_s = bg.partition(":")
        lo, _, hi = rng_s.partition("-")
        for li in range(int(lo), int(hi or lo) + 1):
            out.setdefault(li, {})[proj] = (int(bits_s), int(gs_s))
    return out


def set_expert_overrides(ov):
    """Accepts the parsed per-projection form; bare (bits, gs) values are
    normalized to layer-wide entries for older call sites."""
    global EXPERT_OVERRIDES
    EXPERT_OVERRIDES = {li: (v if isinstance(v, dict) else {"*": tuple(v)})
                        for li, v in ov.items()}


def _trunk_override(pfx, proj):
    if pfx.startswith("mtp."):
        return None
    m = re.match(r"^layers\.(\d+)\.", pfx)
    if not m:
        return None
    ov = EXPERT_OVERRIDES.get(int(m.group(1)))
    if ov is None:
        return None
    return ov.get(proj, ov.get("*"))


def expert_bits(pfx, proj):
    """Expert bit width by MODULE: draft stages are not the trunk."""
    ov = _trunk_override(pfx, proj)
    if ov is not None:
        return ov[0]
    return (MTP_EXPERT_BITS if pfx.startswith("mtp.") else EXPERT_BITS)[proj]


def expert_group(pfx, proj):
    """Expert group size by MODULE and projection: draft stages keep the
    spine's gs 64; a per-projection override carries its own gs."""
    ov = _trunk_override(pfx, proj)
    if ov is not None:
        return ov[1]
    return SPINE_GROUP if pfx.startswith("mtp.") else EXPERT_GROUP


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
            gs = expert_group(cls[1], cls[3])
            n = out_d * (in_d * bits // 8 + in_d // gs * 4)
            key = f"expert.{cls[3]}({bits}b/g{gs})"
        else:
            if name.endswith(".scale"):
                continue
            out_d, in_d = shape
            n = out_d * (in_d + in_d // SPINE_GROUP * 4)
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


def _trunk_expert_task(args):
    """Worker: read one trunk expert's fp4 pair, dequant, weighted-quantize.
    Pure numpy (never imports mlx), so a process pool is safe beside the live
    server. Opens its own ShardReader — no shared fds."""
    src, w_shard, s_shard, w_name, s_name, bits, gs, cw = args
    w_i8, _ = ShardReader(os.path.join(src, w_shard)).read(w_name)
    s_u8, _ = ShardReader(os.path.join(src, s_shard)).read(s_name)
    f32 = dequant_fp4(w_i8, s_u8)
    import dsv4_imatrix as imx
    return imx.weighted_affine_quant(f32, bits, gs, cw)


def convert(src, out, only_groups=None, verify=False, imatrix=None, imatrix_name=None):
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

    import dsv4_imatrix as imx
    quant_cfg = {"group_size": SPINE_GROUP, "bits": 8, "mode": "affine"}
    index_map = {}
    total_size = 0
    pool = None
    workers = int(os.environ.get("MLX_SERVE_CONVERT_WORKERS", "6"))

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
            is_trunk = not pfx.startswith("mtp.")
            layer = int(re.match(r"^layers\.(\d+)\.", pfx).group(1)) if is_trunk else None
            for proj in ("w1", "w2", "w3"):
                    bits = expert_bits(pfx, proj)
                    gs = expert_group(pfx, proj)
                    wq_l, sc_l, bi_l = [], [], []
                    if is_trunk:
                        # Weighted (imatrix-calibrated) path; uniform weights
                        # when no imatrix is in play. A provided imatrix with
                        # a missing trunk entry is a HARD error inside
                        # expert_channel_weights — never a silent minmax.
                        w0 = f"{pfx}.experts.0.{proj}.weight"
                        meta0 = ShardReader(os.path.join(src, wm[w0])).header[w0]
                        in_dim = meta0["shape"][1] * 2  # packed fp4
                        tasks = []
                        for eid in range(n_exp):
                            wn = f"{pfx}.experts.{eid}.{proj}.weight"
                            sn = f"{pfx}.experts.{eid}.{proj}.scale"
                            cw = (imx.expert_channel_weights(imatrix, layer, proj, eid, in_dim, n_exp)
                                  if imatrix is not None
                                  else np.ones(in_dim, dtype=np.float32))
                            tasks.append((src, wm[wn], wm[sn], wn, sn, bits, gs, cw))
                        if workers > 1 and n_exp >= 32:
                            if pool is None:
                                from concurrent.futures import ProcessPoolExecutor
                                pool = ProcessPoolExecutor(max_workers=workers)
                            results = pool.map(_trunk_expert_task, tasks)
                        else:
                            results = map(_trunk_expert_task, tasks)
                        for wq, sc, bi in results:
                            wq_l.append(wq); sc_l.append(sc); bi_l.append(bi)
                    else:
                        for eid in range(n_exp):
                            w_i8, _ = read(f"{pfx}.experts.{eid}.{proj}.weight")
                            s_u8, _ = read(f"{pfx}.experts.{eid}.{proj}.scale")
                            f32 = dequant_fp4(w_i8, s_u8)
                            (wq, sc, bi) = mlx_affine_quant(f32, bits, gs)
                            wq_l.append(wq); sc_l.append(sc); bi_l.append(bi)
                            del f32, w_i8, s_u8
                    for kind, parts in (("weight", wq_l), ("scales", sc_l), ("biases", bi_l)):
                        dtype = parts[0][0]
                        shape = [n_exp] + list(parts[0][1])
                        raw = b"".join(p[2] for p in parts)
                        out_tensors[f"{pfx}.experts.{proj}.{kind}"] = (dtype, shape, raw)
                    quant_cfg[f"{pfx}.experts.{proj}"] = {"group_size": gs, "bits": bits, "mode": "affine"}
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
            wq, sc, bi = mlx_affine_quant(f32, 8, SPINE_GROUP)
            out_tensors[base + ".weight"] = wq
            out_tensors[base + ".scales"] = sc
            out_tensors[base + ".biases"] = bi
            quant_cfg[base] = {"group_size": SPINE_GROUP, "bits": 8, "mode": "affine"}
            if verify:
                back = mlx_affine_dequant_f32(
                    np.frombuffer(wq[2], dtype=np.uint32).reshape(wq[1]),
                    np.frombuffer(sc[2], dtype=np.uint16).reshape(sc[1]),
                    np.frombuffer(bi[2], dtype=np.uint16).reshape(bi[1]), 8, SPINE_GROUP)
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

    if pool is not None:
        pool.shutdown()

    if only_groups is None:
        # index + config + tokenizer files
        with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": index_map}, f)
        # Rebuild the per-path quantization dict from the FULL index — the
        # per-run accumulator misses groups converted by earlier --groups
        # invocations (they skip via manifest and never re-classify).
        stacked_re = re.compile(r"\.experts\.(w[123])\.weight$")
        quant_cfg = {"group_size": SPINE_GROUP, "bits": 8, "mode": "affine"}
        for tn in index_map:
            if tn.endswith(".weight") and tn[:-7] + ".scales" in index_map:
                m = stacked_re.search(tn)
                # `tn` is the FULL stacked path, so its own `mtp.` prefix is
                # what picks the draft-stage widths — the engine dequants from
                # this dict, so a trunk-vs-draft mixup here is silent garbage.
                bits = expert_bits(tn, m.group(1)) if m else 8
                gs = expert_group(tn, m.group(1)) if m else SPINE_GROUP
                quant_cfg[tn[:-7]] = {"group_size": gs, "bits": bits, "mode": "affine"}
        out_cfg = dict(cfg)
        out_cfg.pop("quantization_config", None)  # fp8 source config must not leak
        out_cfg["quantization"] = quant_cfg
        out_cfg["mlx_serve_converter"] = {
            "source": "deepseek-ai/DeepSeek-V4-Flash",
            "method": "iQ-MLX (imatrix-weighted affine search + greedy per-layer bit allocation)",
            "mix": "experts base w1/w3 2-bit + w2 3-bit gs128, per-layer plan in `quantization`; "
                   "spine/embed/head 8-bit gs64; DSpark stages 4-bit gs64; compressor/router bf16",
            "calibration": imatrix_name if imatrix is not None else "uniform (no imatrix)",
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
        # bpw headline: trunk bits over trunk params — the DSpark stages are
        # extra parameters on top of the 284B main model, so both sides of the
        # division exclude them. Computed from actual emitted bytes so the
        # card can never drift from the files.
        mtp_bytes = manifest.get("mtp", {}).get("size", 0)
        bpw = (total_size - mtp_bytes) * 8 / N_PARAMS_MAIN
        with open(os.path.join(out, "README.md"), "w") as f:
            f.write(README.format(base=SOURCE_REPO, gb=f"{total_size/1e9:.1f}",
                                  bpw=f"{bpw:.2f}"))
        print(f"DONE: {total_size/1e9:.1f} GB ({bpw:.2f} bpw) -> {out}")


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
        n_exp, d_in, d_moe = 4, 256, 128
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
        # one fp8 spine linear (256x256 -> 2x2 scale blocks at block 128)
        wq_codes = rng.integers(0, 0x7F, size=(d_in, d_in), dtype=np.uint8)
        tensors["layers.0.attn.wq_a.weight"] = ("F8_E4M3", (d_in, d_in), wq_codes.tobytes())
        tensors["layers.0.attn.wq_a.scale"] = ("F8_E8M0", (2, 2), np.full((2, 2), 127, dtype=np.uint8).tobytes())
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
        # Trunk experts move to gs 128 (imatrix round); draft stages and the
        # spine keep gs 64 — the DSpark shard must stay byte-identical.
        check(ocfg["quantization"]["layers.0.ffn.experts.w1"]["group_size"] == 128
              and ocfg["quantization"]["layers.0.ffn.experts.w2"]["group_size"] == 128
              and ocfg["quantization"]["mtp.1.ffn.experts.w2"]["group_size"] == 64
              and ocfg["quantization"]["layers.0.attn.wq_a"]["group_size"] == 64
              and ocfg["quantization"]["group_size"] == 64,
              "e2e trunk experts gs128, draft stages + spine gs64")
        # The model card carries the iQ-MLX brand and a computed bpw headline
        # (from actual emitted bytes — the card can never drift from the files).
        card = open(os.path.join(outd, "README.md")).read()
        check("iQ-MLX" in card and "What is iQ-MLX" in card, "e2e card carries the iQ-MLX section")
        check(" bpw" in card and "{bpw}" not in card, "e2e card bpw placeholder filled")
        # load bank via mlx, dequant expert 1 w1, compare vs direct source dequant
        shard = os.path.join(outd, "model-layer-0.safetensors")
        loaded = mx().load(shard)
        m = mx()
        bank_back = m.dequantize(loaded["layers.0.ffn.experts.w1.weight"][1],
                                 loaded["layers.0.ffn.experts.w1.scales"][1],
                                 loaded["layers.0.ffn.experts.w1.biases"][1],
                                 group_size=EXPERT_GROUP, bits=2)
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

        # imatrix-calibrated arm: fabricated per-expert channel weights for
        # the trunk (draft stages have no entries — that must NOT error), and
        # a missing TRUNK entry must be a hard error, never a silent minmax.
        im = {}
        rng_im = np.random.default_rng(9)
        for proj, ind in (("w1", d_in), ("w3", d_in), ("w2", d_moe)):
            stem = {"w1": "ffn_gate_exps", "w2": "ffn_down_exps", "w3": "ffn_up_exps"}[proj]
            im[f"blk.0.{stem}.weight"] = (1, np.abs(rng_im.standard_normal(ind * n_exp)).astype(np.float32))
        outd2 = os.path.join(td, "out-imx")
        try:
            convert(src, outd2, verify=True, imatrix=im)
            idx2 = json.load(open(os.path.join(outd2, "model.safetensors.index.json")))
            check("layers.0.ffn.experts.w1.weight" in idx2["weight_map"]
                  and "mtp.1.ffn.experts.w1.weight" in idx2["weight_map"],
                  "e2e imatrix conversion covers trunk + entry-less draft stages")
            sizes_match = all(
                json.load(open(os.path.join(outd, ".convert-manifest.json")))[g]["tensors"]
                == json.load(open(os.path.join(outd2, ".convert-manifest.json")))[g]["tensors"]
                for g in ("layer.0", "mtp"))
            check(sizes_match, "e2e imatrix arm emits the same tensor set")
            # The mtp group never consults the imatrix -> byte-identical shards.
            mtp_a = open(os.path.join(outd, "model-mtp.safetensors"), "rb").read()
            mtp_b = open(os.path.join(outd2, "model-mtp.safetensors"), "rb").read()
            check(mtp_a == mtp_b, "e2e draft-stage shard byte-identical with/without imatrix")
        except TypeError as e:
            check(False, f"e2e imatrix conversion ({e})")
        broken = {k: v for k, v in im.items() if "gate" not in k}
        try:
            convert(src, os.path.join(td, "out-broken"), imatrix=broken)
            check(False, "e2e missing trunk imatrix entry is a hard error")
        except (AssertionError, TypeError) as e:
            check(isinstance(e, AssertionError) and "missing imatrix entry" in str(e),
                  f"e2e missing trunk imatrix entry is a hard error ({e})")

        # Per-layer expert override (--expert-override): trunk layers named in
        # the spec re-quantize at the given bits/gs; everything else — other
        # trunk layers, draft stages, spine — is untouched.
        try:
            ov = parse_expert_override("37-42=4:64")
            check(ov == {li: {"*": (4, 64)} for li in range(37, 43)}, "override: range parse")
            ov2 = parse_expert_override("5=3:32")
            check(ov2 == {5: {"*": (3, 32)}}, "override: single-layer parse")
            set_expert_overrides({37: (4, 64)})
            check(expert_bits("layers.37.ffn", "w1") == 4
                  and expert_bits("layers.37.ffn", "w2") == 4, "override: expert_bits hit")
            check(expert_group("layers.37.ffn", "w1") == 64, "override: expert_group hit")
            check(expert_bits("layers.36.ffn", "w1") == EXPERT_BITS["w1"]
                  and expert_group("layers.36.ffn", "w1") == EXPERT_GROUP, "override: miss untouched")
            check(expert_bits("mtp.0.ffn", "w1") == MTP_EXPERT_BITS["w1"]
                  and expert_group("mtp.0.ffn", "w1") == SPINE_GROUP, "override: mtp untouched")
            # full stacked path (quant-config rebuild spelling) also resolves
            check(expert_bits("layers.37.ffn.experts.w1.weight", "w1") == 4,
                  "override: stacked-path spelling hit")
        except NameError as e:
            check(False, f"override: helpers missing ({e})")
        finally:
            try:
                set_expert_overrides({})
            except NameError:
                pass
        # Per-PROJECTION override grammar "LAYER=PROJ@BITS:GS" (the greedy
        # allocator emits mixed plans like w2-only upgrades): a named
        # projection resolves to its own (bits, gs), the layer's OTHER
        # projections stay on the defaults, and the layer-wide form still
        # applies to all three. Specs merge across comma-joined parts.
        try:
            ov3 = parse_expert_override("5=w2@4:128")
            check(ov3 == {5: {"w2": (4, 128)}}, "override: per-proj parse")
            ov4 = parse_expert_override("0-1=w1@3:128,0=w2@4:128,39-40=4:64")
            check(ov4 == {0: {"w1": (3, 128), "w2": (4, 128)},
                          1: {"w1": (3, 128)},
                          39: {"*": (4, 64)}, 40: {"*": (4, 64)}},
                  "override: mixed per-proj + layer-wide merge")
            set_expert_overrides(parse_expert_override("37=w2@4:128"))
            check(expert_bits("layers.37.ffn", "w2") == 4
                  and expert_bits("layers.37.ffn", "w1") == EXPERT_BITS["w1"]
                  and expert_bits("layers.37.ffn", "w3") == EXPERT_BITS["w3"],
                  "override: per-proj bits hit only the named projection")
            check(expert_group("layers.37.ffn", "w2") == 128
                  and expert_group("layers.37.ffn", "w1") == EXPERT_GROUP,
                  "override: per-proj gs hit only the named projection")
            check(expert_bits("mtp.0.ffn", "w2") == MTP_EXPERT_BITS["w2"],
                  "override: per-proj mtp untouched")
        except Exception as e:
            check(False, f"override: per-proj grammar missing ({e})")
        finally:
            try:
                set_expert_overrides({})
            except NameError:
                pass
        # e2e: overriding the fixture's one trunk layer flips its banks to
        # 4-bit gs64 in the emitted quant config; draft stages byte-identical.
        try:
            set_expert_overrides(parse_expert_override("0=4:64"))
            outd3 = os.path.join(td, "out-ov")
            convert(src, outd3, verify=True)
            ocfg3 = json.load(open(os.path.join(outd3, "config.json")))
            check(all(ocfg3["quantization"][f"layers.0.ffn.experts.{p}"] ==
                      {"group_size": 64, "bits": 4, "mode": "affine"} for p in ("w1", "w2", "w3")),
                  "override e2e: trunk banks 4-bit gs64 in quant config")
            check(ocfg3["quantization"]["mtp.1.ffn.experts.w2"]["bits"] == MTP_EXPERT_BITS["w2"],
                  "override e2e: draft stages untouched")
            mtp_c = open(os.path.join(outd3, "model-mtp.safetensors"), "rb").read()
            check(mtp_c == mtp_a, "override e2e: draft-stage shard byte-identical")
            w4 = json.load(open(os.path.join(outd3, ".convert-manifest.json")))["layer.0"]
            check(w4["size"] > json.load(open(os.path.join(outd, ".convert-manifest.json")))["layer.0"]["size"],
                  "override e2e: 4-bit layer shard larger than 2/3-bit")
        except NameError as e:
            check(False, f"override e2e: helpers missing ({e})")
        finally:
            try:
                set_expert_overrides({})
            except NameError:
                pass

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
    ap.add_argument("--expert-override", type=str, default=None,
                    help="per-layer trunk expert requant, e.g. '37-42=4:64'")
    ap.add_argument("--imatrix", type=str, default=None,
                    help="imatrix .dat for trunk-expert calibration (default: the "
                         "staged antirez .dat; pass 'none' for an uncalibrated build)")
    args = ap.parse_args()

    if args.expert_override:
        set_expert_overrides(parse_expert_override(args.expert_override))
    if args.self_test:
        sys.exit(self_test())

    assert args.src and args.out, "--src and --out required"
    src = os.path.expanduser(args.src)
    out = os.path.expanduser(args.out)

    import dsv4_imatrix as imx
    imatrix = None
    imatrix_name = None
    if args.imatrix != "none" and not args.dry_run:
        im_path = os.path.expanduser(args.imatrix) if args.imatrix else imx.IMATRIX_DEFAULT
        # A missing imatrix must never fall back to a silent minmax build:
        # demand the file or an explicit 'none'.
        assert os.path.exists(im_path), \
            f"imatrix not found at {im_path} — pass --imatrix none for an uncalibrated build"
        imatrix = imx.load_imatrix(im_path)
        imatrix_name = os.path.basename(im_path)
        print(f"[imatrix] {imatrix_name}: {len(imatrix)} entries")

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
    convert(src, out, only_groups=only, verify=args.verify,
            imatrix=imatrix, imatrix_name=imatrix_name)
