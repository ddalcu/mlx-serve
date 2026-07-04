#!/usr/bin/env python3
"""Convert the UniRig stage-1 SKELETON checkpoint into mlx-serve's layout.

USER-RUN (needs torch + mlx + the 1.44 GB Lightning ckpt). Not run in CI. Produces
the single-file model dir the native Zig skeleton engine (`src/unirig_skeleton.zig`,
Phase 3) loads:

    <out>/config.json         {"model_type":"unirig_skeleton", ...}
    <out>/skeleton.safetensors  ar.* (OPT-350m decoder) + enc.* (michelangelo
                                perceiver) + output_proj.* (512->1024 bridge)
    <out>/LICENSE             UniRig repo MIT license (weights are MIT)
    <out>/NOTICE              records the GPLv3-code / MIT-weights split

The BINDING converted-name contract is `tests/unirig_weights_contract.md`; this
script implements it exactly. It mirrors the SHAPE converter
`tests/convert_hunyuan3d_weights.py` — read that first — reusing its
`deinterleave_qkv` per-head un-interleave, its `Source` strict pop/leftover
accounting, its `should_quantize` predicate, and its `mx.save_safetensors` +
`--self-test` + `--bits {8,16}` structure.

Two structural facts (contract §2):
  - Lightning `model.` prefix on every tensor; strip it. NO optimizer/EMA state.
  - The michelangelo `c_qkv`/`c_kv` carry the SAME per-head interleave as the
    HY3D ShapeVAE michelangelo attention (identical `MultiheadAttention` classes),
    so they are de-interleaved (heads 8, head_dim 64) at convert time; the Zig
    side does a STANDARD head reshape. The OPT `*_proj` are plain HF — NOT touched.

Usage:
    python3 tests/convert_unirig_weights.py --src <ckpt file OR dir> [--out DIR] [--bits {8,16}]
    python3 tests/convert_unirig_weights.py --self-test   # synthetic unit tests, no ckpt/torch/mlx needed

--src is the `model.ckpt` file, or a dir containing
`skeleton/articulation-xl_quantization_256/model.ckpt` (or any `*.ckpt` under it).
If --src is omitted the download hint is printed.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

# ── config facts (verified 2026-07-04 against the ckpt inventory + reference) ──
# OPT-350m decoder (unirig_ar_350m_1024_81920_float32.yaml + facebook/opt-350m base)
AR_LAYERS = 24
AR_HIDDEN = 1024
AR_HEADS = 16
AR_HEAD_DIM = AR_HIDDEN // AR_HEADS            # 64
AR_FFN = 4096
AR_MAX_POS = 3076
AR_POS_OFFSET = 2
AR_POS_ROWS = AR_MAX_POS + AR_POS_OFFSET       # 3078
VOCAB_SIZE = 267

# michelangelo perceiver encoder (mesh_encoder block of the AR yaml)
ENC_WIDTH = 512
ENC_HEADS = 8
ENC_HEAD_DIM = ENC_WIDTH // ENC_HEADS          # 64
ENC_LAYERS = 16
NUM_FREQS = 8
FOURIER_OUT = 3 * (NUM_FREQS * 2 + 1)          # 51
POINT_FEATS = 3
INPUT_PROJ_IN = FOURIER_OUT + POINT_FEATS      # 54
TOKEN_NUM = 1024
NUM_LATENTS = 512
OUTPUT_PROJ_IN = ENC_WIDTH                      # 512
OUTPUT_PROJ_OUT = AR_HIDDEN                     # 1024

# tokenizer (tokenizer_parts_articulationxl_256.yaml + tokenizer_part.py:22-45)
NUM_DISCRETE = 256
CONTINUOUS_RANGE = [-1.0, 1.0]
TOK = {
    "branch": 256, "bos": 257, "eos": 258, "pad": 259, "spring": 260,
    "body": 261, "hand": 262, "cls_none": 263,
    "vroid": 264, "mixamo": 265, "articulationxl": 266,
}

# sampling (ar_inference_articulationxl.yaml generate_kwargs)
SAMPLING = {
    "max_new_tokens": 2048, "num_beams": 15, "do_sample": True,
    "top_k": 5, "top_p": 0.95, "repetition_penalty": 3.0, "temperature": 1.5,
    "assign_cls": "articulationxl", "seed": 12345,
}

# sampler / normalization (inference_ar_transform.yaml)
NUM_SAMPLES = 65536
VERTEX_SAMPLES = 8192

GROUP_SIZE = 64
MIN_QUANT_DIM = 512

EXPECT_SOURCE_TENSORS = 585
EXPECT_OUTPUT_TENSORS = 618            # fp16 logical count (after the de-interleave split)


# ── (T2) attention per-head de-interleave (mirrored from convert_hunyuan3d_weights.py) ──
def deinterleave_qkv(w, heads, head_dim, n_members):
    """Undo the reference `cat.view(heads, M*head_dim).split(head_dim)` interleave.

    `w` is the fused weight of shape [n_members*heads*head_dim, in]. Returns a list
    of `n_members` arrays, each [heads*head_dim, in], such that a STANDARD per-head
    reshape of `x @ Wm.T` reproduces reference member m. Row map:

        Wm[h*head_dim + j] = w[h*(n_members*head_dim) + m*head_dim + j]
    """
    out_rows = n_members * heads * head_dim
    assert w.shape[0] == out_rows, (
        f"deinterleave_qkv: got {w.shape[0]} rows, expected {out_rows} "
        f"(heads={heads} head_dim={head_dim} n_members={n_members})"
    )
    members = []
    for m in range(n_members):
        idx = np.empty(heads * head_dim, dtype=np.int64)
        for h in range(heads):
            base = h * (n_members * head_dim) + m * head_dim
            idx[h * head_dim:(h + 1) * head_dim] = np.arange(base, base + head_dim)
        members.append(w[idx])
    return members


def _reference_view_split(x, w, heads, head_dim, n_members):
    """The reference forward, in numpy: out = x@w.T, view(heads, M*hd), split(hd)."""
    out = x @ w.T
    lead = out.shape[:-1]
    out = out.reshape(*lead, heads, n_members * head_dim)
    return [out[..., m * head_dim:(m + 1) * head_dim] for m in range(n_members)]


# ── quantization predicate (identical to the SHAPE converter) ─────────────────
def should_quantize(name, shape, bits):
    """A linear .weight is quantized iff bits==8, ndim == 2, last dim % GROUP_SIZE
    == 0, and min of the two dims >= 512. Everything else (norms/biases/gathered
    embedding tables/tiny projections) stays fp16."""
    if bits != 8:
        return False
    if not name.endswith(".weight"):
        return False                               # biases, embed_tokens/embed_positions (T4)
    if len(shape) != 2:
        return False                               # 1-D norms
    out_f, in_f = shape[-2], shape[-1]
    if in_f % GROUP_SIZE != 0:
        return False
    return min(out_f, in_f) >= MIN_QUANT_DIM


# ── ckpt loading + strict accounting ──────────────────────────────────────────
def _load_torch_ckpt(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as err:                       # noqa: BLE001
        print(f"[warn] mmap weights_only load failed ({err}); retrying non-mmap", flush=True)
        return torch.load(path, map_location="cpu", weights_only=True)


def _state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt


def t2np(x):
    """Torch tensor -> contiguous numpy float16 (fp32/bf16 sources cast down)."""
    import torch
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)
    return np.ascontiguousarray(x.detach().cpu().to(torch.float16).numpy())


class Source:
    """A flat name->numpy(fp16) dict with strict pop/leftover accounting."""

    def __init__(self, flat_np, label):
        self.d = flat_np
        self.label = label
        self.used = set()

    def pop(self, key):
        if key not in self.d:
            near = [k for k in self.d if k.rsplit(".", 1)[0] == key.rsplit(".", 1)[0]]
            if not near:
                pref = key.split(".")[0]
                near = [k for k in self.d if k.startswith(pref)][:12]
            raise SystemExit(
                f"[FATAL] {self.label}: MISSING required source key '{key}'.\n"
                f"        Nearby keys: {near[:12]}"
            )
        self.used.add(key)
        return self.d[key]

    def leftover(self):
        return sorted(k for k in self.d if k not in self.used)


# ── mapping builders ──────────────────────────────────────────────────────────
def build_ar(src):
    """model.transformer.* -> ar.* (contract §4). Plain HF OPT layout; no de-interleave."""
    out = {}
    dec = "model.decoder"
    out["ar.embed_tokens"] = src.pop(f"{dec}.embed_tokens.weight")          # T4: drop .weight
    out["ar.embed_positions"] = src.pop(f"{dec}.embed_positions.weight")    # T4: drop .weight
    out["ar.final_norm.weight"] = src.pop(f"{dec}.final_layer_norm.weight")
    out["ar.final_norm.bias"] = src.pop(f"{dec}.final_layer_norm.bias")
    out["ar.lm_head.weight"] = src.pop("lm_head.weight")

    for i in range(AR_LAYERS):
        s = f"{dec}.layers.{i}"
        b = f"ar.layers.{i}"
        for src_p, dst_p in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("out_proj", "out")):
            out[f"{b}.attn.{dst_p}.weight"] = src.pop(f"{s}.self_attn.{src_p}.weight")
            out[f"{b}.attn.{dst_p}.bias"] = src.pop(f"{s}.self_attn.{src_p}.bias")
        out[f"{b}.attn_norm.weight"] = src.pop(f"{s}.self_attn_layer_norm.weight")
        out[f"{b}.attn_norm.bias"] = src.pop(f"{s}.self_attn_layer_norm.bias")
        out[f"{b}.mlp.fc1.weight"] = src.pop(f"{s}.fc1.weight")
        out[f"{b}.mlp.fc1.bias"] = src.pop(f"{s}.fc1.bias")
        out[f"{b}.mlp.fc2.weight"] = src.pop(f"{s}.fc2.weight")
        out[f"{b}.mlp.fc2.bias"] = src.pop(f"{s}.fc2.bias")
        out[f"{b}.mlp_norm.weight"] = src.pop(f"{s}.final_layer_norm.weight")
        out[f"{b}.mlp_norm.bias"] = src.pop(f"{s}.final_layer_norm.bias")
    return out


def build_encoder(src):
    """model.mesh_encoder.encoder.* -> enc.* (contract §5). c_qkv/c_kv de-interleaved."""
    out = {}
    out["enc.input_proj.weight"] = src.pop("input_proj.weight")
    out["enc.input_proj.bias"] = src.pop("input_proj.bias")

    # --- cross-attention block (ResidualCrossAttentionBlock, pre-LN) ---
    ca = "cross_attn"
    out["enc.cross_attn.ln1.weight"] = src.pop(f"{ca}.ln_1.weight")
    out["enc.cross_attn.ln1.bias"] = src.pop(f"{ca}.ln_1.bias")
    out["enc.cross_attn.ln2.weight"] = src.pop(f"{ca}.ln_2.weight")
    out["enc.cross_attn.ln2.bias"] = src.pop(f"{ca}.ln_2.bias")
    out["enc.cross_attn.attn.q.weight"] = src.pop(f"{ca}.attn.c_q.weight")   # standard
    ck, cv = deinterleave_qkv(src.pop(f"{ca}.attn.c_kv.weight"), ENC_HEADS, ENC_HEAD_DIM, 2)
    out["enc.cross_attn.attn.k.weight"], out["enc.cross_attn.attn.v.weight"] = ck, cv
    out["enc.cross_attn.attn.out.weight"] = src.pop(f"{ca}.attn.c_proj.weight")
    out["enc.cross_attn.attn.out.bias"] = src.pop(f"{ca}.attn.c_proj.bias")
    out["enc.cross_attn.ln3.weight"] = src.pop(f"{ca}.ln_3.weight")
    out["enc.cross_attn.ln3.bias"] = src.pop(f"{ca}.ln_3.bias")
    out["enc.cross_attn.mlp.fc1.weight"] = src.pop(f"{ca}.mlp.c_fc.weight")
    out["enc.cross_attn.mlp.fc1.bias"] = src.pop(f"{ca}.mlp.c_fc.bias")
    out["enc.cross_attn.mlp.fc2.weight"] = src.pop(f"{ca}.mlp.c_proj.weight")
    out["enc.cross_attn.mlp.fc2.bias"] = src.pop(f"{ca}.mlp.c_proj.bias")

    # --- self-attention blocks (ResidualAttentionBlock, pre-LN) ---
    for i in range(ENC_LAYERS):
        s = f"self_attn.resblocks.{i}"
        b = f"enc.blocks.{i}"
        out[f"{b}.ln1.weight"] = src.pop(f"{s}.ln_1.weight")
        out[f"{b}.ln1.bias"] = src.pop(f"{s}.ln_1.bias")
        q, k, v = deinterleave_qkv(src.pop(f"{s}.attn.c_qkv.weight"), ENC_HEADS, ENC_HEAD_DIM, 3)
        out[f"{b}.attn.q.weight"], out[f"{b}.attn.k.weight"], out[f"{b}.attn.v.weight"] = q, k, v
        out[f"{b}.attn.out.weight"] = src.pop(f"{s}.attn.c_proj.weight")
        out[f"{b}.attn.out.bias"] = src.pop(f"{s}.attn.c_proj.bias")
        out[f"{b}.ln2.weight"] = src.pop(f"{s}.ln_2.weight")
        out[f"{b}.ln2.bias"] = src.pop(f"{s}.ln_2.bias")
        out[f"{b}.mlp.fc1.weight"] = src.pop(f"{s}.mlp.c_fc.weight")
        out[f"{b}.mlp.fc1.bias"] = src.pop(f"{s}.mlp.c_fc.bias")
        out[f"{b}.mlp.fc2.weight"] = src.pop(f"{s}.mlp.c_proj.weight")
        out[f"{b}.mlp.fc2.bias"] = src.pop(f"{s}.mlp.c_proj.bias")

    out["enc.ln_post.weight"] = src.pop("ln_post.weight")
    out["enc.ln_post.bias"] = src.pop("ln_post.bias")
    return out


def build_output_proj(src):
    """model.output_proj.* -> output_proj.* (contract §6)."""
    return {
        "output_proj.weight": src.pop("output_proj.weight"),
        "output_proj.bias": src.pop("output_proj.bias"),
    }


# leftover keys legitimately dropped (not an error), per namespace. The skeleton
# ckpt carries none of these today; they document what a future ckpt might add.
DROP_RULES = {
    "ar": [],
    "encoder": [
        ("fourier_embedder.frequencies", "non-persistent Fourier buffer (recomputed; absent in this ckpt)"),
    ],
    "output_proj": [],
}


def enforce_leftovers(src, ns):
    dropped, fatal = [], []
    for k in src.leftover():
        reason = None
        for pat, why in DROP_RULES[ns]:
            if (k == pat) or (pat.endswith(".") and k.startswith(pat)):
                reason = why
                break
        (dropped if reason else fatal).append((k, reason))
    if dropped:
        print(f"[{ns}] dropped {len(dropped)} source key(s):")
        for k, why in dropped:
            print(f"    {k}  ({why})")
    if fatal:
        raise SystemExit(
            f"[FATAL] {ns}: {len(fatal)} UNMAPPED source key(s) with no drop rule "
            f"(fix the mapping or add a DROP_RULES entry):\n" +
            "\n".join(f"    {k}" for k, _ in fatal[:40])
        )


# ── save (quantize + write one file) ──────────────────────────────────────────
def save_safetensors(out_np, path, bits):
    import mlx.core as mx
    packed = {}
    n_quant = 0
    for name, arr in out_np.items():
        arr = np.ascontiguousarray(arr, dtype=np.float16)
        if should_quantize(name, arr.shape, bits):
            wq, scales, biases = mx.quantize(mx.array(arr), group_size=GROUP_SIZE, bits=bits)
            base = name[: -len(".weight")]
            packed[f"{base}.weight"] = wq
            packed[f"{base}.scales"] = scales.astype(mx.float16)
            packed[f"{base}.biases"] = biases.astype(mx.float16)
            n_quant += 1
        else:
            packed[name] = mx.array(arr)
    mx.eval(*packed.values())
    mx.save_safetensors(path, packed)
    nbytes = sum(v.nbytes for v in packed.values())
    return len(packed), n_quant, nbytes


# ── config.json ───────────────────────────────────────────────────────────────
def write_config(out, bits):
    cfg = {
        "model_type": "unirig_skeleton",
        "quant": "8bit" if bits == 8 else "fp16",
        "tokenizer": {
            "num_discrete": NUM_DISCRETE,
            "continuous_range": CONTINUOUS_RANGE,
            "vocab_size": VOCAB_SIZE,
            "token_branch": TOK["branch"], "token_bos": TOK["bos"],
            "token_eos": TOK["eos"], "token_pad": TOK["pad"], "token_spring": TOK["spring"],
            "parts": {"body": TOK["body"], "hand": TOK["hand"]},
            "token_cls_none": TOK["cls_none"],
            "cls": {"vroid": TOK["vroid"], "mixamo": TOK["mixamo"], "articulationxl": TOK["articulationxl"]},
            "default_cls": "articulationxl",
            "skeleton_name_templates": {
                "vroid": "configs/skeleton/vroid.yaml",
                "mixamo": "configs/skeleton/mixamo.yaml",
            },
        },
        "ar": {
            "arch": "opt",
            "num_hidden_layers": AR_LAYERS, "hidden_size": AR_HIDDEN,
            "num_attention_heads": AR_HEADS, "head_dim": AR_HEAD_DIM, "ffn_dim": AR_FFN,
            "activation_function": "relu", "word_embed_proj_dim": AR_HIDDEN,
            "do_layer_norm_before": True,
            "max_position_embeddings": AR_MAX_POS, "n_positions": AR_MAX_POS,
            "position_offset": AR_POS_OFFSET, "pos_embed_rows": AR_POS_ROWS,
            "layer_norm_eps": 1e-05, "vocab_size": VOCAB_SIZE, "tie_word_embeddings": True,
        },
        "encoder": {
            "arch": "michelangelo_perceiver",
            "width": ENC_WIDTH, "heads": ENC_HEADS, "head_dim": ENC_HEAD_DIM,
            "num_encoder_layers": ENC_LAYERS, "num_freqs": NUM_FREQS,
            "include_pi": False, "include_input": True, "fourier_out_dim": FOURIER_OUT,
            "point_feats": POINT_FEATS, "input_proj_in": INPUT_PROJ_IN,
            "num_latents": NUM_LATENTS, "token_num": TOKEN_NUM, "presample": TOKEN_NUM * 4,
            "presample_seed": 0, "fps_ratio": 0.25, "fps_random_start": False,
            "use_full_input": True, "use_ln_post": True, "qkv_bias": False,
            "mlp_ratio": 4, "mlp_act": "gelu",
        },
        "cond": {
            "num_mesh_tokens": TOKEN_NUM, "output_proj_in": OUTPUT_PROJ_IN,
            "output_proj_out": OUTPUT_PROJ_OUT, "start_tokens": ["bos", "cls"],
            "logits_slice_from": "post_cond",
        },
        "sampling": SAMPLING,
        "sampler": {
            "num_samples": NUM_SAMPLES, "vertex_samples": VERTEX_SAMPLES, "method": "mix",
            "normalize_into": CONTINUOUS_RANGE, "normals_transformed": False,
        },
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] wrote config.json (quant={cfg['quant']})")


def copy_license(src_dir, out):
    import shutil
    found = []
    roots = [src_dir, os.path.dirname(src_dir.rstrip("/"))]
    for name in ("LICENSE", "NOTICE"):
        for r in roots:
            if not r:
                continue
            cands = [os.path.join(r, name)] + glob.glob(os.path.join(r, "**", name), recursive=True)
            hit = next((c for c in cands if os.path.isfile(c)), None)
            if hit:
                shutil.copyfile(hit, os.path.join(out, name))
                found.append(name)
                break
    if "LICENSE" not in found:
        print("[license] no UniRig LICENSE found near --src — copy the UniRig MIT license manually")
    else:
        print(f"[license] copied {', '.join(found)}")
    # Always (re)write NOTICE documenting the GPLv3-code / MIT-weights split.
    with open(os.path.join(out, "NOTICE"), "w") as f:
        f.write(
            "UniRig stage-1 skeleton weights converted for mlx-serve.\n\n"
            "Weights: MIT (VAST-AI/UniRig, HF model card `license: mit`; repo LICENSE is MIT).\n"
            "The michelangelo point-cloud encoder SOURCE CODE is GPLv3 (derived from\n"
            "NeuralCarver/Michelangelo). Only the numeric weights (MIT) are converted here;\n"
            "the mlx-serve Zig engine re-derives the perceiver clean-room and does NOT copy\n"
            "the GPLv3 source. See tests/unirig_weights_contract.md sec 9.\n"
        )


DOWNLOAD_HINT = """\
--src not given. Download the stage-1 skeleton ckpt to an EXTERNAL-disk scratch dir
(1.44 GB; do NOT pull it onto the internal disk), then re-run with --src:

  hf download VAST-AI/UniRig \\
      --include "skeleton/articulation-xl_quantization_256/*" \\
      --local-dir /Volumes/Sandisk_1TB/hy3d-scratch/unirig-ckpt

  python3 tests/convert_unirig_weights.py --src /Volumes/Sandisk_1TB/hy3d-scratch/unirig-ckpt
"""


def _resolve_ckpt(src):
    if os.path.isfile(src) and src.endswith(".ckpt"):
        return src
    for g in ("skeleton/articulation-xl_quantization_256/model.ckpt",
              "**/articulation-xl_quantization_256/model.ckpt", "**/model.ckpt", "*.ckpt"):
        hits = sorted(glob.glob(os.path.join(src, g), recursive=True))
        if hits:
            return hits[0]
    return None


def convert(src, out, bits):
    ckpt = _resolve_ckpt(src)
    if ckpt is None:
        raise SystemExit(f"[FATAL] could not find a .ckpt under {src}")
    print(f"[load] skeleton ckpt: {ckpt}")
    sd = _state_dict(_load_torch_ckpt(ckpt))
    if len(sd) != EXPECT_SOURCE_TENSORS:
        print(f"[warn] source tensor-count drift: {len(sd)} (exp {EXPECT_SOURCE_TENSORS})")

    # strip the Lightning `model.` prefix (T1); FATAL if any key lacks it.
    bad = [k for k in sd if not k.startswith("model.")]
    if bad:
        raise SystemExit(f"[FATAL] {len(bad)} key(s) not under 'model.' prefix (e.g. {bad[:5]})")

    def sub(prefix):
        return {k[len(prefix):]: t2np(v) for k, v in sd.items() if k.startswith(prefix)}

    ar_np = sub("model.transformer.")                 # -> model.decoder.* + lm_head.weight
    enc_np = sub("model.mesh_encoder.encoder.")       # -> input_proj/cross_attn/self_attn/ln_post
    op_np = {k[len("model."):]: t2np(v) for k, v in sd.items() if k.startswith("model.output_proj.")}
    for label, d in (("ar", ar_np), ("encoder", enc_np), ("output_proj", op_np)):
        if not d:
            raise SystemExit(f"[FATAL] no source keys for the '{label}' namespace — unexpected layout")

    os.makedirs(out, exist_ok=True)

    print("[map] ar (OPT decoder) ...")
    ar_src = Source(ar_np, "ar")
    ar_out = build_ar(ar_src)
    enforce_leftovers(ar_src, "ar")

    print("[map] encoder (michelangelo perceiver) ...")
    enc_src = Source(enc_np, "encoder")
    enc_out = build_encoder(enc_src)
    enforce_leftovers(enc_src, "encoder")

    print("[map] output_proj ...")
    op_src = Source(op_np, "output_proj")
    op_out = build_output_proj(op_src)
    enforce_leftovers(op_src, "output_proj")

    merged = {**ar_out, **enc_out, **op_out}
    n_logical = len(merged)
    if n_logical != EXPECT_OUTPUT_TENSORS:
        print(f"[warn] output logical-tensor drift: {n_logical} (exp {EXPECT_OUTPUT_TENSORS})")
    print(f"[map] ar={len(ar_out)} enc={len(enc_out)} output_proj={len(op_out)} "
          f"=> {n_logical} logical tensors")

    n, nq, nbytes = save_safetensors(merged, os.path.join(out, "skeleton.safetensors"), bits)
    print(f"[save] skeleton.safetensors: {n} physical tensors ({nq} quantized @ {bits}-bit), "
          f"{nbytes / 1e6:.1f} MB")

    write_config(out, bits)
    copy_license(os.path.dirname(ckpt), out)

    print("\n[done] wrote model dir:", out)
    print(f"[done] {nbytes / 1e9:.2f} GB, {n_logical} logical tensors "
          f"({nq} quantized @ {bits}-bit)")


# ── self-test (no ckpt / torch / mlx needed) ──────────────────────────────────
def self_test():
    rng = np.random.default_rng(0)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print("[self-test] deinterleave_qkv reproduces the reference view+split")
    # the two michelangelo shapes the encoder uses, plus a hand-checkable tiny case
    cases = [
        ("enc self  c_qkv", ENC_HEADS, ENC_HEAD_DIM, 3, ENC_WIDTH),   # [1536,512]
        ("enc cross c_kv ", ENC_HEADS, ENC_HEAD_DIM, 2, ENC_WIDTH),   # [1024,512]
        ("tiny          ", 2, 2, 3, 4),
    ]
    for label, heads, hd, m, in_f in cases:
        out_rows = m * heads * hd
        w = rng.standard_normal((out_rows, in_f)).astype(np.float32)
        x = rng.standard_normal((5, in_f)).astype(np.float32)
        ref = _reference_view_split(x, w, heads, hd, m)
        members = deinterleave_qkv(w, heads, hd, m)
        good = all(np.allclose((x @ members[mi].T).reshape(5, heads, hd), ref[mi], atol=1e-4)
                   for mi in range(m))
        check(good, f"{label} heads={heads} hd={hd} M={m} in={in_f}")

    print("[self-test] deinterleave_qkv row map matches the hand-derived tiny example")
    w = np.arange(3 * 2 * 2).reshape(12, 1).astype(np.float32)
    q, k, v = deinterleave_qkv(w, heads=2, head_dim=2, n_members=3)
    check(q[:, 0].tolist() == [0, 1, 6, 7], "q rows = [0,1,6,7]")
    check(k[:, 0].tolist() == [2, 3, 8, 9], "k rows = [2,3,8,9]")
    check(v[:, 0].tolist() == [4, 5, 10, 11], "v rows = [4,5,10,11]")

    print("[self-test] build_ar / build_encoder / build_output_proj map every source key (1:1)")
    # synthesize a full source state_dict at the true shapes, run the builders, and
    # assert no leftover + expected output counts.
    # ar source (post model.transformer. strip)
    ar_np = {}
    dec = "model.decoder"
    ar_np[f"{dec}.embed_tokens.weight"] = np.zeros((VOCAB_SIZE, AR_HIDDEN), np.float16)
    ar_np[f"{dec}.embed_positions.weight"] = np.zeros((AR_POS_ROWS, AR_HIDDEN), np.float16)
    ar_np[f"{dec}.final_layer_norm.weight"] = np.zeros((AR_HIDDEN,), np.float16)
    ar_np[f"{dec}.final_layer_norm.bias"] = np.zeros((AR_HIDDEN,), np.float16)
    ar_np["lm_head.weight"] = np.zeros((VOCAB_SIZE, AR_HIDDEN), np.float16)
    for i in range(AR_LAYERS):
        s = f"{dec}.layers.{i}"
        for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
            ar_np[f"{s}.self_attn.{p}.weight"] = np.zeros((AR_HIDDEN, AR_HIDDEN), np.float16)
            ar_np[f"{s}.self_attn.{p}.bias"] = np.zeros((AR_HIDDEN,), np.float16)
        ar_np[f"{s}.self_attn_layer_norm.weight"] = np.zeros((AR_HIDDEN,), np.float16)
        ar_np[f"{s}.self_attn_layer_norm.bias"] = np.zeros((AR_HIDDEN,), np.float16)
        ar_np[f"{s}.fc1.weight"] = np.zeros((AR_FFN, AR_HIDDEN), np.float16)
        ar_np[f"{s}.fc1.bias"] = np.zeros((AR_FFN,), np.float16)
        ar_np[f"{s}.fc2.weight"] = np.zeros((AR_HIDDEN, AR_FFN), np.float16)
        ar_np[f"{s}.fc2.bias"] = np.zeros((AR_HIDDEN,), np.float16)
        ar_np[f"{s}.final_layer_norm.weight"] = np.zeros((AR_HIDDEN,), np.float16)
        ar_np[f"{s}.final_layer_norm.bias"] = np.zeros((AR_HIDDEN,), np.float16)
    ar_src = Source(ar_np, "ar")
    ar_out = build_ar(ar_src)
    check(ar_src.leftover() == [], f"ar: 0 leftover (got {len(ar_src.leftover())})")
    check(len(ar_out) == 389, f"ar: 389 output tensors (got {len(ar_out)})")

    enc_np = {}
    enc_np["input_proj.weight"] = np.zeros((ENC_WIDTH, INPUT_PROJ_IN), np.float16)
    enc_np["input_proj.bias"] = np.zeros((ENC_WIDTH,), np.float16)
    for ln in ("ln_1", "ln_2", "ln_3"):
        enc_np[f"cross_attn.{ln}.weight"] = np.zeros((ENC_WIDTH,), np.float16)
        enc_np[f"cross_attn.{ln}.bias"] = np.zeros((ENC_WIDTH,), np.float16)
    enc_np["cross_attn.attn.c_q.weight"] = np.zeros((ENC_WIDTH, ENC_WIDTH), np.float16)
    enc_np["cross_attn.attn.c_kv.weight"] = np.zeros((ENC_WIDTH * 2, ENC_WIDTH), np.float16)
    enc_np["cross_attn.attn.c_proj.weight"] = np.zeros((ENC_WIDTH, ENC_WIDTH), np.float16)
    enc_np["cross_attn.attn.c_proj.bias"] = np.zeros((ENC_WIDTH,), np.float16)
    enc_np["cross_attn.mlp.c_fc.weight"] = np.zeros((ENC_WIDTH * 4, ENC_WIDTH), np.float16)
    enc_np["cross_attn.mlp.c_fc.bias"] = np.zeros((ENC_WIDTH * 4,), np.float16)
    enc_np["cross_attn.mlp.c_proj.weight"] = np.zeros((ENC_WIDTH, ENC_WIDTH * 4), np.float16)
    enc_np["cross_attn.mlp.c_proj.bias"] = np.zeros((ENC_WIDTH,), np.float16)
    for i in range(ENC_LAYERS):
        s = f"self_attn.resblocks.{i}"
        enc_np[f"{s}.ln_1.weight"] = np.zeros((ENC_WIDTH,), np.float16)
        enc_np[f"{s}.ln_1.bias"] = np.zeros((ENC_WIDTH,), np.float16)
        enc_np[f"{s}.attn.c_qkv.weight"] = np.zeros((ENC_WIDTH * 3, ENC_WIDTH), np.float16)
        enc_np[f"{s}.attn.c_proj.weight"] = np.zeros((ENC_WIDTH, ENC_WIDTH), np.float16)
        enc_np[f"{s}.attn.c_proj.bias"] = np.zeros((ENC_WIDTH,), np.float16)
        enc_np[f"{s}.ln_2.weight"] = np.zeros((ENC_WIDTH,), np.float16)
        enc_np[f"{s}.ln_2.bias"] = np.zeros((ENC_WIDTH,), np.float16)
        enc_np[f"{s}.mlp.c_fc.weight"] = np.zeros((ENC_WIDTH * 4, ENC_WIDTH), np.float16)
        enc_np[f"{s}.mlp.c_fc.bias"] = np.zeros((ENC_WIDTH * 4,), np.float16)
        enc_np[f"{s}.mlp.c_proj.weight"] = np.zeros((ENC_WIDTH, ENC_WIDTH * 4), np.float16)
        enc_np[f"{s}.mlp.c_proj.bias"] = np.zeros((ENC_WIDTH,), np.float16)
    enc_np["ln_post.weight"] = np.zeros((ENC_WIDTH,), np.float16)
    enc_np["ln_post.bias"] = np.zeros((ENC_WIDTH,), np.float16)
    enc_src = Source(enc_np, "encoder")
    enc_out = build_encoder(enc_src)
    check(enc_src.leftover() == [], f"encoder: 0 leftover (got {len(enc_src.leftover())})")
    check(len(enc_out) == 227, f"encoder: 227 output tensors (got {len(enc_out)})")

    op_np = {"output_proj.weight": np.zeros((OUTPUT_PROJ_OUT, OUTPUT_PROJ_IN), np.float16),
             "output_proj.bias": np.zeros((OUTPUT_PROJ_OUT,), np.float16)}
    op_src = Source(op_np, "output_proj")
    op_out = build_output_proj(op_src)
    check(op_src.leftover() == [] and len(op_out) == 2, "output_proj: 0 leftover, 2 tensors")

    total = len(ar_out) + len(enc_out) + len(op_out)
    check(total == EXPECT_OUTPUT_TENSORS, f"total {total} == {EXPECT_OUTPUT_TENSORS} logical tensors")
    # de-interleave split adds exactly 33 over the 585 source (see contract §1)
    check(total == 585 + 33, "output = 585 source + 33 de-interleave split tensors")

    print("[self-test] should_quantize predicate")
    table = [
        ("ar.layers.0.attn.q.weight", (1024, 1024), 8, True),
        ("ar.layers.0.mlp.fc1.weight", (4096, 1024), 8, True),
        ("ar.layers.0.mlp.fc2.weight", (1024, 4096), 8, True),
        ("enc.cross_attn.attn.q.weight", (512, 512), 8, True),
        ("enc.blocks.0.attn.k.weight", (512, 512), 8, True),        # post de-interleave
        ("enc.blocks.0.mlp.fc1.weight", (2048, 512), 8, True),
        ("enc.blocks.0.mlp.fc2.weight", (512, 2048), 8, True),
        ("output_proj.weight", (1024, 512), 8, True),
        # left fp16
        ("ar.lm_head.weight", (267, 1024), 8, False),               # min 267 < 512
        ("ar.embed_tokens", (267, 1024), 8, False),                 # not .weight (T4)
        ("ar.embed_positions", (3078, 1024), 8, False),             # not .weight (T4)
        ("enc.input_proj.weight", (512, 54), 8, False),             # in 54 < 512 and 54%64!=0
        ("ar.layers.0.attn_norm.weight", (1024,), 8, False),        # 1-D norm
        ("ar.layers.0.attn.q.bias", (1024,), 8, False),             # bias
        ("output_proj.bias", (1024,), 8, False),
        ("ar.layers.0.attn.q.weight", (1024, 1024), 16, False),     # bits=16 -> never
    ]
    for name, shape, bits, want in table:
        check(should_quantize(name, shape, bits) == want,
              f"{name} {tuple(shape)} bits={bits} -> {should_quantize(name, shape, bits)}")

    print("[self-test] tokenizer vocab offsets match tokenizer_part.py")
    exp = {"branch": 256, "bos": 257, "eos": 258, "pad": 259, "spring": 260,
           "body": 261, "hand": 262, "cls_none": 263,
           "vroid": 264, "mixamo": 265, "articulationxl": 266}
    check(TOK == exp and VOCAB_SIZE == 267, "offsets 256..266, vocab 267")

    print("\n[self-test] " + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Convert the UniRig stage-1 skeleton ckpt for mlx-serve.")
    ap.add_argument("--src", default=None,
                    help="model.ckpt file, or a dir containing skeleton/.../model.ckpt")
    ap.add_argument("--out", default=None, help="output model dir (default depends on --bits)")
    ap.add_argument("--bits", type=int, default=8, choices=(8, 16),
                    help="8 = mlx affine 8-bit for eligible linears (default); 16 = fp16 everywhere")
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic unit tests (de-interleave / mapping / quantize rule) and exit")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())
    if args.src is None:
        print(DOWNLOAD_HINT)
        sys.exit(1)

    out = args.out or os.path.expanduser(
        "~/.mlx-serve/models/local/unirig-skeleton-8bit" if args.bits == 8
        else "~/.mlx-serve/models/local/unirig-skeleton-fp16"
    )
    convert(os.path.abspath(args.src), os.path.abspath(out), args.bits)


if __name__ == "__main__":
    main()
