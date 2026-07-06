#!/usr/bin/env python3
"""Convert ACE-Step v1.5 XL Turbo checkpoints into mlx-serve's layout.

USER-RUN (needs torch + mlx + safetensors + the HF checkpoints). Not run in CI.
Produces the model dir the native Zig engine (`src/acestep.zig`) loads:

    <out>/config.json          {"model_type":"acestep", ...}
    <out>/model.safetensors    DiT decoder + condition encoder + silence_latent
    <out>/vae.safetensors      AutoencoderOobleck (weight-norm FUSED, MLX conv layouts)
    <out>/text_encoder/        Qwen3-Embedding-0.6B copied verbatim (bf16, standard qwen3)
    <out>/LICENSE              copied if present

The tensor-name contract is BINDING between this script and the Zig loader; it is
documented in music-dossier.md §3. Verified against the reference sources
(modeling_acestep_v15_xl_turbo.py + the pipeline's own MLX reference
acestep/models/mlx/{dit_convert,vae_convert}.py). Structural transforms:

  (a) Conv1d weights           PT [out, in, K]  -> MLX [out, K, in]
      ConvTranspose1d weights  PT [in, out, K]  -> MLX [out, K, in]
      (decoder.proj_in.1 / decoder.proj_out.1, renamed to drop the Sequential index;
       every VAE conv / conv_t1)
  (b) VAE weight_norm fusion   w = g * v / ||v||  (norm over per-out-channel flatten
      for Conv1d, per-IN-channel for ConvTranspose1d), then (a).
  (c) Snake alpha/beta         PT [1, C, 1] -> [C], kept FLOAT32 (exp() headroom).
  (d) silence_latent.pt        [1, 64, 15000] -> transpose -> `silence_latent` [1, 15000, 64].
  (e) FSQ tokenizer/detokenizer/attention-pooler DROPPED (cover-mode only, not text2music).
  (f) fp32 source -> bf16 dense; --bits 8 quantizes eligible 2-D linears with mlx
      affine group_size 64 (packed uint32 .weight + bf16 .scales/.biases).

Usage:
    python3 tests/convert_acestep_weights.py --src-xl <acestep-v15-xl-turbo dir> \
        --src-main <Ace-Step1.5 dir> [--out DIR] [--bits {8,16}]
    python3 tests/convert_acestep_weights.py --self-test   # no ckpt/torch/mlx needed
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

# ── config facts (config.json of ACE-Step/acestep-v15-xl-turbo, verified) ─────
HIDDEN = 2560
NUM_LAYERS = 32
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 9728
ENC_HIDDEN = 2048
ENC_INTERMEDIATE = 6144
ENC_HEADS = 16
ENC_KV_HEADS = 8
NUM_LYRIC_LAYERS = 8
NUM_TIMBRE_LAYERS = 4
TEXT_HIDDEN = 1024
ACOUSTIC_DIM = 64
IN_CHANNELS = 192
PATCH_SIZE = 2
SLIDING_WINDOW = 128
ROPE_THETA = 1000000.0
RMS_EPS = 1e-6
TIMBRE_FIX_FRAME = 750
SILENCE_FRAMES = 15000
SAMPLE_RATE = 48000
VAE_HOP = 1920  # 2*4*4*6*10 -> latents at exactly 25 Hz
VAE_DOWNSAMPLING = [2, 4, 4, 6, 10]
VAE_CHANNEL_MULTIPLES = [1, 2, 4, 8, 16]
VAE_ENCODER_HIDDEN = 128
VAE_DECODER_CHANNELS = 128

GROUP_SIZE = 64


# ── (b) weight_norm fusion ────────────────────────────────────────────────────
def fuse_weight_norm(weight_g, weight_v, eps=1e-9):
    """w = g * v / ||v||, norm over the flatten of all dims but 0 (torch dim=0).

    For Conv1d       g is [out, 1, 1], v is [out, in, K]  -> per-out-channel norm.
    For ConvTranspose1d g is [in, 1, 1], v is [in, out, K] -> per-IN-channel norm.
    Returns the fused weight in the ORIGINAL torch shape (before axis swap).
    """
    v_flat = weight_v.reshape(weight_v.shape[0], -1)
    norm = np.linalg.norm(v_flat.astype(np.float64), axis=1).astype(np.float32)
    norm = norm.reshape(weight_g.shape)
    return (weight_g * weight_v / (norm + eps)).astype(np.float32)


# ── (a) conv layout swaps ─────────────────────────────────────────────────────
def conv1d_to_mlx(w):
    """PT Conv1d [out, in, K] -> MLX [out, K, in]."""
    assert w.ndim == 3, f"conv1d weight must be 3-D, got {w.shape}"
    return np.ascontiguousarray(w.transpose(0, 2, 1))


def convtranspose1d_to_mlx(w):
    """PT ConvTranspose1d [in, out, K] -> MLX [out, K, in]."""
    assert w.ndim == 3, f"conv_t weight must be 3-D, got {w.shape}"
    return np.ascontiguousarray(w.transpose(1, 2, 0))


# ── (f) quantization rule (pure predicate) ────────────────────────────────────
def should_quantize(name, shape, bits):
    """A linear .weight is quantized iff 2-D, in-features % GROUP_SIZE == 0, and
    min(out, in) >= 512. Norms, tables (3-D), convs (3-D), biases, silence latent,
    and small projections (time_embed.linear_1 has in=256) stay bf16."""
    if bits != 8:
        return False
    if not name.endswith(".weight"):
        return False
    if len(shape) != 2:
        return False
    out_f, in_f = shape
    if in_f % GROUP_SIZE != 0:
        return False
    return min(out_f, in_f) >= 512


# ── source accounting (pop/leftover discipline) ───────────────────────────────
class Source:
    def __init__(self, flat_np, label):
        self.d = flat_np
        self.label = label
        self.used = set()

    def pop(self, key):
        if key not in self.d:
            near = [k for k in self.d if k.rsplit(".", 1)[0] == key.rsplit(".", 1)[0]]
            if not near:
                pref = ".".join(key.split(".")[:2])
                near = [k for k in self.d if k.startswith(pref)][:12]
            raise SystemExit(
                f"[FATAL] {self.label}: MISSING required source key '{key}'.\n"
                f"        Nearby keys: {near[:12]}"
            )
        self.used.add(key)
        return self.d[key]

    def leftover(self):
        return sorted(k for k in self.d if k not in self.used)


# ── DiT + condition encoder mapping ──────────────────────────────────────────
def _map_attention(out, src, dst_prefix, src_prefix, has_bias=False):
    """q/k/v/o projections + q/k head norms — names pass through unchanged."""
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        out[f"{dst_prefix}.{p}.weight"] = src.pop(f"{src_prefix}.{p}.weight")
        if has_bias:
            out[f"{dst_prefix}.{p}.bias"] = src.pop(f"{src_prefix}.{p}.bias")
    out[f"{dst_prefix}.q_norm.weight"] = src.pop(f"{src_prefix}.q_norm.weight")
    out[f"{dst_prefix}.k_norm.weight"] = src.pop(f"{src_prefix}.k_norm.weight")


def _map_mlp(out, src, dst_prefix, src_prefix):
    for p in ("gate_proj", "up_proj", "down_proj"):
        out[f"{dst_prefix}.mlp.{p}.weight"] = src.pop(f"{src_prefix}.mlp.{p}.weight")


def _map_timestep_embed(out, src, name):
    for p in ("linear_1", "linear_2", "time_proj"):
        out[f"decoder.{name}.{p}.weight"] = src.pop(f"decoder.{name}.{p}.weight")
        out[f"decoder.{name}.{p}.bias"] = src.pop(f"decoder.{name}.{p}.bias")


def _map_encoder_layer(out, src, dst_prefix, src_prefix):
    """AceStepEncoderLayer: pre-norm self-attn + pre-norm SwiGLU MLP."""
    _map_attention(out, src, f"{dst_prefix}.self_attn", f"{src_prefix}.self_attn")
    out[f"{dst_prefix}.input_layernorm.weight"] = src.pop(f"{src_prefix}.input_layernorm.weight")
    out[f"{dst_prefix}.post_attention_layernorm.weight"] = src.pop(
        f"{src_prefix}.post_attention_layernorm.weight")
    _map_mlp(out, src, dst_prefix, src_prefix)


def build_model(src, silence_np):
    """XL-turbo checkpoint (decoder.* + encoder.* + null_condition_emb) -> converted names.

    Names pass through unchanged EXCEPT decoder.proj_in.1/proj_out.1 (Sequential
    index dropped + conv layout swapped). FSQ tokenizer/detokenizer/pooler dropped.
    """
    out = {}

    # -- DiT decoder --------------------------------------------------------
    w = src.pop("decoder.proj_in.1.weight")
    assert list(w.shape) == [HIDDEN, IN_CHANNELS, PATCH_SIZE], f"proj_in shape {w.shape}"
    out["decoder.proj_in.weight"] = conv1d_to_mlx(w)
    out["decoder.proj_in.bias"] = src.pop("decoder.proj_in.1.bias")

    w = src.pop("decoder.proj_out.1.weight")
    assert list(w.shape) == [HIDDEN, ACOUSTIC_DIM, PATCH_SIZE], f"proj_out shape {w.shape}"
    out["decoder.proj_out.weight"] = convtranspose1d_to_mlx(w)
    out["decoder.proj_out.bias"] = src.pop("decoder.proj_out.1.bias")

    _map_timestep_embed(out, src, "time_embed")
    _map_timestep_embed(out, src, "time_embed_r")

    out["decoder.condition_embedder.weight"] = src.pop("decoder.condition_embedder.weight")
    out["decoder.condition_embedder.bias"] = src.pop("decoder.condition_embedder.bias")
    out["decoder.norm_out.weight"] = src.pop("decoder.norm_out.weight")
    out["decoder.scale_shift_table"] = src.pop("decoder.scale_shift_table")

    for i in range(NUM_LAYERS):
        b = f"decoder.layers.{i}"
        _map_attention(out, src, f"{b}.self_attn", f"{b}.self_attn")
        _map_attention(out, src, f"{b}.cross_attn", f"{b}.cross_attn")
        for norm in ("self_attn_norm", "cross_attn_norm", "mlp_norm"):
            out[f"{b}.{norm}.weight"] = src.pop(f"{b}.{norm}.weight")
        _map_mlp(out, src, b, b)
        tbl = src.pop(f"{b}.scale_shift_table")
        assert list(tbl.shape) == [1, 6, HIDDEN], f"{b}.scale_shift_table shape {tbl.shape}"
        out[f"{b}.scale_shift_table"] = tbl

    # -- condition encoder ----------------------------------------------------
    out["encoder.text_projector.weight"] = src.pop("encoder.text_projector.weight")

    le = "encoder.lyric_encoder"
    out[f"{le}.embed_tokens.weight"] = src.pop(f"{le}.embed_tokens.weight")
    out[f"{le}.embed_tokens.bias"] = src.pop(f"{le}.embed_tokens.bias")
    out[f"{le}.norm.weight"] = src.pop(f"{le}.norm.weight")
    for i in range(NUM_LYRIC_LAYERS):
        _map_encoder_layer(out, src, f"{le}.layers.{i}", f"{le}.layers.{i}")

    te = "encoder.timbre_encoder"
    out[f"{te}.embed_tokens.weight"] = src.pop(f"{te}.embed_tokens.weight")
    out[f"{te}.embed_tokens.bias"] = src.pop(f"{te}.embed_tokens.bias")
    out[f"{te}.norm.weight"] = src.pop(f"{te}.norm.weight")
    st = src.pop(f"{te}.special_token")
    assert list(st.shape) == [1, 1, ENC_HIDDEN], f"timbre special_token shape {st.shape}"
    out[f"{te}.special_token"] = st
    for i in range(NUM_TIMBRE_LAYERS):
        _map_encoder_layer(out, src, f"{te}.layers.{i}", f"{te}.layers.{i}")

    out["null_condition_emb"] = src.pop("null_condition_emb")

    # -- silence latent ([1,64,15000] on disk -> [1,15000,64]) -----------------
    assert list(silence_np.shape) == [1, ACOUSTIC_DIM, SILENCE_FRAMES], (
        f"silence_latent shape {silence_np.shape}, expected [1, 64, 15000]")
    out["silence_latent"] = np.ascontiguousarray(silence_np.transpose(0, 2, 1))

    return out


# leftover keys that are legitimately dropped, per namespace.
MODEL_DROP_RULES = [
    ("tokenizer.", "FSQ audio tokenizer — cover-mode only, not used for text2music"),
    ("detokenizer.", "FSQ audio detokenizer — cover-mode only, not used for text2music"),
]


def enforce_leftovers(src, drop_rules):
    dropped, fatal = [], []
    for k in src.leftover():
        reason = None
        for pat, why in drop_rules:
            if (k == pat) or (pat.endswith(".") and k.startswith(pat)):
                reason = why
                break
        (dropped if reason else fatal).append((k, reason))
    if dropped:
        shown = {}
        for k, why in dropped:
            shown[why] = shown.get(why, 0) + 1
        print(f"[{src.label}] dropped {len(dropped)} source key(s):")
        for why, n in shown.items():
            print(f"    {n:5d}  {why}")
    if fatal:
        raise SystemExit(
            f"[FATAL] {src.label}: {len(fatal)} UNMAPPED source key(s) with no drop rule:\n"
            + "\n".join(f"    {k}" for k, _ in fatal[:40]))


# ── VAE mapping ───────────────────────────────────────────────────────────────
def build_vae(src):
    """AutoencoderOobleck state dict -> fused, MLX-layout names (names otherwise
    unchanged). Both encoder and decoder are converted (encoder feeds M3's
    reference-audio timbre + cover paths; it's only 169M params total)."""
    out = {}
    all_keys = sorted(src.d.keys())
    for key in all_keys:
        if key in src.used:
            continue
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            g = src.pop(key)
            v = src.pop(base + ".weight_v")
            w = fuse_weight_norm(g, v)
            if ".conv_t1" in base:
                w = convtranspose1d_to_mlx(w)
            else:
                w = conv1d_to_mlx(w)
            out[base + ".weight"] = w
        elif key.endswith(".weight_v"):
            continue  # handled with its weight_g
        elif key.endswith(".alpha") or key.endswith(".beta"):
            val = src.pop(key)
            assert val.ndim == 3 and val.shape[0] == 1 and val.shape[2] == 1, (
                f"snake param {key} shape {val.shape}")
            out[key] = np.ascontiguousarray(val.reshape(-1))  # stays fp32
        elif key.endswith(".bias"):
            out[key] = src.pop(key)
        else:
            raise SystemExit(f"[FATAL] vae: unexpected source key '{key}'")
    return out


# ── save (quantize + write) ───────────────────────────────────────────────────
def save_model_safetensors(out_np, path, bits):
    import mlx.core as mx
    packed = {}
    n_quant = 0
    for name, arr in out_np.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if should_quantize(name, arr.shape, bits):
            wq, scales, biases = mx.quantize(mx.array(arr), group_size=GROUP_SIZE, bits=bits)
            base = name[: -len(".weight")]
            packed[f"{base}.weight"] = wq
            packed[f"{base}.scales"] = scales.astype(mx.bfloat16)
            packed[f"{base}.biases"] = biases.astype(mx.bfloat16)
            n_quant += 1
        else:
            packed[name] = mx.array(arr).astype(mx.bfloat16)
    mx.eval(*packed.values())
    mx.save_safetensors(path, packed)
    nbytes = sum(v.nbytes for v in packed.values())
    return len(packed), n_quant, nbytes


def save_vae_safetensors(out_np, path):
    """VAE stays dense bf16 EXCEPT Snake alpha/beta which stay fp32 (exp headroom)."""
    import mlx.core as mx
    packed = {}
    for name, arr in out_np.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if name.endswith(".alpha") or name.endswith(".beta"):
            packed[name] = mx.array(arr)  # fp32
        else:
            packed[name] = mx.array(arr).astype(mx.bfloat16)
    mx.eval(*packed.values())
    mx.save_safetensors(path, packed)
    return len(packed), sum(v.nbytes for v in packed.values())


# ── loading helpers ───────────────────────────────────────────────────────────
def load_sharded_safetensors_fp32(model_dir):
    """Load model-0000N-of-00004.safetensors shards -> flat {name: np.float32}."""
    from safetensors import safe_open
    shards = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
    if not shards:
        single = os.path.join(model_dir, "model.safetensors")
        if os.path.isfile(single):
            shards = [single]
    if not shards:
        raise SystemExit(f"[FATAL] no model*.safetensors under {model_dir}")
    flat = {}
    for s in shards:
        print(f"[load] {os.path.basename(s)}")
        with safe_open(s, framework="np") as f:
            for k in f.keys():
                flat[k] = np.asarray(f.get_tensor(k), dtype=np.float32)
    return flat


def load_silence_latent(model_dir):
    import torch
    p = os.path.join(model_dir, "silence_latent.pt")
    if not os.path.isfile(p):
        raise SystemExit(f"[FATAL] silence_latent.pt not found under {model_dir}")
    t = torch.load(p, map_location="cpu", weights_only=True)
    return np.ascontiguousarray(t.detach().to(torch.float32).numpy())


def load_vae_bf16(vae_dir):
    import torch
    from safetensors.torch import load_file
    p = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
    if not os.path.isfile(p):
        raise SystemExit(f"[FATAL] VAE safetensors not found at {p}")
    sd = load_file(p)
    return {k: np.ascontiguousarray(v.detach().to(torch.float32).numpy()) for k, v in sd.items()}


TEXT_ENCODER_FILES = [
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
]


def copy_text_encoder(src_main, out):
    te_src = os.path.join(src_main, "Qwen3-Embedding-0.6B")
    if not os.path.isdir(te_src):
        raise SystemExit(f"[FATAL] Qwen3-Embedding-0.6B not found under {src_main}")
    te_out = os.path.join(out, "text_encoder")
    os.makedirs(te_out, exist_ok=True)
    copied = 0
    for name in TEXT_ENCODER_FILES:
        p = os.path.join(te_src, name)
        if os.path.isfile(p):
            shutil.copyfile(p, os.path.join(te_out, name))
            copied += 1
        elif name in ("config.json", "model.safetensors", "tokenizer.json"):
            raise SystemExit(f"[FATAL] text encoder missing required file {name}")
    print(f"[text_encoder] copied {copied} files verbatim (bf16 qwen3, ~1.2 GB)")


def write_config(out, bits):
    cfg = {
        "model_type": "acestep",
        "model_version": "turbo",
        "quant": "8bit" if bits == 8 else "bf16",
        "hidden_size": HIDDEN, "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS, "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM, "intermediate_size": INTERMEDIATE,
        "encoder_hidden_size": ENC_HIDDEN, "encoder_intermediate_size": ENC_INTERMEDIATE,
        "encoder_num_attention_heads": ENC_HEADS, "encoder_num_key_value_heads": ENC_KV_HEADS,
        "num_lyric_encoder_hidden_layers": NUM_LYRIC_LAYERS,
        "num_timbre_encoder_hidden_layers": NUM_TIMBRE_LAYERS,
        "text_hidden_dim": TEXT_HIDDEN, "audio_acoustic_hidden_dim": ACOUSTIC_DIM,
        "in_channels": IN_CHANNELS, "patch_size": PATCH_SIZE,
        "sliding_window": SLIDING_WINDOW, "rope_theta": ROPE_THETA,
        "rms_norm_eps": RMS_EPS, "timbre_fix_frame": TIMBRE_FIX_FRAME,
        "sample_rate": SAMPLE_RATE, "vae_hop": VAE_HOP,
        "vae_downsampling_ratios": VAE_DOWNSAMPLING,
        "vae_channel_multiples": VAE_CHANNEL_MULTIPLES,
        "vae_encoder_hidden_size": VAE_ENCODER_HIDDEN,
        "vae_decoder_channels": VAE_DECODER_CHANNELS,
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] wrote config.json (quant={cfg['quant']})")


def copy_license(src_xl, out):
    for name in ("LICENSE", "LICENSE.md"):
        p = os.path.join(src_xl, name)
        if os.path.isfile(p):
            shutil.copyfile(p, os.path.join(out, "LICENSE"))
            print("[license] copied LICENSE")
            return
    print("[license] no LICENSE under --src-xl (MIT per README; add manually before publishing)")


def convert(src_xl, src_main, out, bits):
    os.makedirs(out, exist_ok=True)

    print("[load] XL-turbo shards (fp32, ~20 GB — mmap'd per shard) ...")
    flat = load_sharded_safetensors_fp32(src_xl)
    silence = load_silence_latent(src_xl)

    print("[map] DiT + condition encoder ...")
    msrc = Source(flat, "model")
    model_out = build_model(msrc, silence)
    enforce_leftovers(msrc, MODEL_DROP_RULES)
    del flat

    n, nq, nbytes = save_model_safetensors(model_out, os.path.join(out, "model.safetensors"), bits)
    print(f"[save] model.safetensors: {n} tensors ({nq} quantized), {nbytes/1e9:.2f} GB")
    del model_out

    print("[load] VAE ...")
    vae_flat = load_vae_bf16(os.path.join(src_main, "vae"))
    vsrc = Source(vae_flat, "vae")
    vae_out = build_vae(vsrc)
    enforce_leftovers(vsrc, [])
    n, nbytes = save_vae_safetensors(vae_out, os.path.join(out, "vae.safetensors"))
    print(f"[save] vae.safetensors: {n} tensors, {nbytes/1e6:.1f} MB")

    copy_text_encoder(src_main, out)
    write_config(out, bits)
    copy_license(src_xl, out)
    print(f"\n[done] converted model at: {out}")


# ── self-test (no torch/mlx/ckpt) ─────────────────────────────────────────────
def self_test():
    rng = np.random.default_rng(0)

    # weight_norm fusion reproduces torch semantics: w = g * v / ||v||_dim0-flatten
    for shape, gshape in [((8, 4, 7), (8, 1, 1)), ((6, 12, 20), (6, 1, 1))]:
        v = rng.standard_normal(shape).astype(np.float32)
        g = rng.standard_normal(gshape).astype(np.float32)
        w = fuse_weight_norm(g, v)
        expect = np.empty_like(v)
        for o in range(shape[0]):
            expect[o] = g[o, 0, 0] * v[o] / np.linalg.norm(v[o].reshape(-1))
        assert np.allclose(w, expect, atol=1e-5), "weight_norm fusion mismatch"
    print("[self-test] weight_norm fusion OK")

    # conv layout swaps: value-level checks, not just shapes
    w = rng.standard_normal((5, 3, 2)).astype(np.float32)  # [out, in, K]
    m = conv1d_to_mlx(w)
    assert m.shape == (5, 2, 3)
    assert m[4, 1, 2] == w[4, 2, 1]
    wt = rng.standard_normal((3, 5, 2)).astype(np.float32)  # [in, out, K]
    mt = convtranspose1d_to_mlx(wt)
    assert mt.shape == (5, 2, 3)
    assert mt[4, 1, 2] == wt[2, 4, 1]
    print("[self-test] conv layout swaps OK")

    # quantization predicate
    assert should_quantize("decoder.layers.0.self_attn.q_proj.weight", (4096, 2560), 8)
    assert should_quantize("decoder.time_embed.time_proj.weight", (15360, 2560), 8)
    assert not should_quantize("decoder.time_embed.linear_1.weight", (2560, 256), 8), \
        "time linear_1 (in=256) must stay dense"
    assert not should_quantize("decoder.layers.0.scale_shift_table", (1, 6, 2560), 8)
    assert not should_quantize("decoder.norm_out.weight", (2560,), 8)
    assert not should_quantize("silence_latent", (1, 15000, 64), 8)
    assert not should_quantize("decoder.proj_in.weight", (2560, 2, 192), 8), \
        "convs are 3-D and must stay dense"
    assert not should_quantize("decoder.layers.0.self_attn.q_proj.weight", (4096, 2560), 16)
    print("[self-test] quantization predicate OK")

    # build_model mapping on a synthetic mini checkpoint: verify pop/rename/transform
    mini = {}
    mini["decoder.proj_in.1.weight"] = rng.standard_normal((HIDDEN, IN_CHANNELS, PATCH_SIZE)).astype(np.float32)
    mini["decoder.proj_in.1.bias"] = rng.standard_normal((HIDDEN,)).astype(np.float32)
    mini["decoder.proj_out.1.weight"] = rng.standard_normal((HIDDEN, ACOUSTIC_DIM, PATCH_SIZE)).astype(np.float32)
    mini["decoder.proj_out.1.bias"] = rng.standard_normal((ACOUSTIC_DIM,)).astype(np.float32)
    for name in ("time_embed", "time_embed_r"):
        mini[f"decoder.{name}.linear_1.weight"] = rng.standard_normal((HIDDEN, 256)).astype(np.float32)
        mini[f"decoder.{name}.linear_1.bias"] = rng.standard_normal((HIDDEN,)).astype(np.float32)
        mini[f"decoder.{name}.linear_2.weight"] = rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float32)
        mini[f"decoder.{name}.linear_2.bias"] = rng.standard_normal((HIDDEN,)).astype(np.float32)
        mini[f"decoder.{name}.time_proj.weight"] = rng.standard_normal((HIDDEN * 6, HIDDEN)).astype(np.float32)
        mini[f"decoder.{name}.time_proj.bias"] = rng.standard_normal((HIDDEN * 6,)).astype(np.float32)
    mini["decoder.condition_embedder.weight"] = rng.standard_normal((HIDDEN, ENC_HIDDEN)).astype(np.float32)
    mini["decoder.condition_embedder.bias"] = rng.standard_normal((HIDDEN,)).astype(np.float32)
    mini["decoder.norm_out.weight"] = rng.standard_normal((HIDDEN,)).astype(np.float32)
    mini["decoder.scale_shift_table"] = rng.standard_normal((1, 2, HIDDEN)).astype(np.float32)

    def add_attn(prefix, q_out, kv_out, hidden):
        mini[f"{prefix}.q_proj.weight"] = rng.standard_normal((q_out, hidden)).astype(np.float32)
        mini[f"{prefix}.k_proj.weight"] = rng.standard_normal((kv_out, hidden)).astype(np.float32)
        mini[f"{prefix}.v_proj.weight"] = rng.standard_normal((kv_out, hidden)).astype(np.float32)
        mini[f"{prefix}.o_proj.weight"] = rng.standard_normal((hidden, q_out)).astype(np.float32)
        mini[f"{prefix}.q_norm.weight"] = rng.standard_normal((HEAD_DIM,)).astype(np.float32)
        mini[f"{prefix}.k_norm.weight"] = rng.standard_normal((HEAD_DIM,)).astype(np.float32)

    def add_mlp(prefix, hidden, inter):
        mini[f"{prefix}.mlp.gate_proj.weight"] = rng.standard_normal((inter, hidden)).astype(np.float32)
        mini[f"{prefix}.mlp.up_proj.weight"] = rng.standard_normal((inter, hidden)).astype(np.float32)
        mini[f"{prefix}.mlp.down_proj.weight"] = rng.standard_normal((hidden, inter)).astype(np.float32)

    for i in range(NUM_LAYERS):
        b = f"decoder.layers.{i}"
        add_attn(f"{b}.self_attn", NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM, HIDDEN)
        add_attn(f"{b}.cross_attn", NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM, HIDDEN)
        for norm in ("self_attn_norm", "cross_attn_norm", "mlp_norm"):
            mini[f"{b}.{norm}.weight"] = rng.standard_normal((HIDDEN,)).astype(np.float32)
        add_mlp(b, HIDDEN, INTERMEDIATE)
        mini[f"{b}.scale_shift_table"] = rng.standard_normal((1, 6, HIDDEN)).astype(np.float32)

    mini["encoder.text_projector.weight"] = rng.standard_normal((ENC_HIDDEN, TEXT_HIDDEN)).astype(np.float32)
    for enc, nl, in_dim in (("lyric_encoder", NUM_LYRIC_LAYERS, TEXT_HIDDEN),
                            ("timbre_encoder", NUM_TIMBRE_LAYERS, ACOUSTIC_DIM)):
        e = f"encoder.{enc}"
        mini[f"{e}.embed_tokens.weight"] = rng.standard_normal((ENC_HIDDEN, in_dim)).astype(np.float32)
        mini[f"{e}.embed_tokens.bias"] = rng.standard_normal((ENC_HIDDEN,)).astype(np.float32)
        mini[f"{e}.norm.weight"] = rng.standard_normal((ENC_HIDDEN,)).astype(np.float32)
        for i in range(nl):
            b = f"{e}.layers.{i}"
            add_attn(f"{b}.self_attn", ENC_HEADS * HEAD_DIM, ENC_KV_HEADS * HEAD_DIM, ENC_HIDDEN)
            mini[f"{b}.input_layernorm.weight"] = rng.standard_normal((ENC_HIDDEN,)).astype(np.float32)
            mini[f"{b}.post_attention_layernorm.weight"] = rng.standard_normal((ENC_HIDDEN,)).astype(np.float32)
            add_mlp(b, ENC_HIDDEN, ENC_INTERMEDIATE)
    mini["encoder.timbre_encoder.special_token"] = rng.standard_normal((1, 1, ENC_HIDDEN)).astype(np.float32)
    mini["null_condition_emb"] = rng.standard_normal((1, 1, ENC_HIDDEN)).astype(np.float32)
    # FSQ leftovers that must be DROPPED, not fatal
    mini["tokenizer.quantizer.project_in.weight"] = rng.standard_normal((6, ENC_HIDDEN)).astype(np.float32)
    mini["detokenizer.special_tokens"] = rng.standard_normal((1, 5, ENC_HIDDEN)).astype(np.float32)

    silence = rng.standard_normal((1, ACOUSTIC_DIM, SILENCE_FRAMES)).astype(np.float32)
    msrc = Source(dict(mini), "model")
    out = build_model(msrc, silence)
    enforce_leftovers(msrc, MODEL_DROP_RULES)

    # 829 source tensors in the real ckpt; here: same count minus the FSQ family
    # sanity: converted names present + transforms applied
    assert out["decoder.proj_in.weight"].shape == (HIDDEN, PATCH_SIZE, IN_CHANNELS)
    assert out["decoder.proj_in.weight"][7, 1, 3] == mini["decoder.proj_in.1.weight"][7, 3, 1]
    assert out["decoder.proj_out.weight"].shape == (ACOUSTIC_DIM, PATCH_SIZE, HIDDEN)
    assert out["decoder.proj_out.weight"][5, 1, 9] == mini["decoder.proj_out.1.weight"][9, 5, 1]
    assert out["silence_latent"].shape == (1, SILENCE_FRAMES, ACOUSTIC_DIM)
    assert out["silence_latent"][0, 123, 7] == silence[0, 7, 123]
    assert "tokenizer.quantizer.project_in.weight" not in out
    # decoder non-layer (proj_in 2 + proj_out 2 + time embeds 12 + cond_embedder 2
    # + norm_out 1 + table 1 = 20) + 32×19 layers + text_projector + lyric (3 + 8×11)
    # + timbre (4 + 4×11) + null + silence = 770 (= 829 source − 60 FSQ + 1 silence)
    expected = 20 + NUM_LAYERS * 19 + 1 + (3 + NUM_LYRIC_LAYERS * 11) + (4 + NUM_TIMBRE_LAYERS * 11) + 1 + 1
    assert len(out) == expected, f"converted tensor count {len(out)} != {expected}"
    print(f"[self-test] build_model mapping OK ({len(out)} tensors)")

    # build_vae on a synthetic weight-norm'd mini VAE
    vmini = {
        "decoder.conv1.weight_g": rng.standard_normal((2048, 1, 1)).astype(np.float32),
        "decoder.conv1.weight_v": rng.standard_normal((2048, 64, 7)).astype(np.float32),
        "decoder.conv1.bias": rng.standard_normal((2048,)).astype(np.float32),
        "decoder.block.0.conv_t1.weight_g": rng.standard_normal((2048, 1, 1)).astype(np.float32),
        "decoder.block.0.conv_t1.weight_v": rng.standard_normal((2048, 1024, 20)).astype(np.float32),
        "decoder.block.0.conv_t1.bias": rng.standard_normal((1024,)).astype(np.float32),
        "decoder.snake1.alpha": rng.standard_normal((1, 128, 1)).astype(np.float32),
        "decoder.snake1.beta": rng.standard_normal((1, 128, 1)).astype(np.float32),
    }
    vsrc = Source(dict(vmini), "vae")
    vout = build_vae(vsrc)
    enforce_leftovers(vsrc, [])
    assert vout["decoder.conv1.weight"].shape == (2048, 7, 64)
    assert vout["decoder.block.0.conv_t1.weight"].shape == (1024, 20, 2048)
    assert vout["decoder.snake1.alpha"].shape == (128,)
    assert "decoder.conv1.weight_g" not in vout and "decoder.conv1.weight_v" not in vout
    print("[self-test] build_vae mapping OK")

    print("\nALL SELF-TESTS PASSED")


DOWNLOAD_HINT = """\
--src-xl / --src-main not given. Download to an EXTERNAL-disk scratch dir (the XL
checkpoint is ~20 GB fp32), then re-run:

  huggingface-cli download ACE-Step/acestep-v15-xl-turbo \\
      --local-dir /Volumes/Sandisk_1TB/acestep/acestep-v15-xl-turbo

  huggingface-cli download ACE-Step/Ace-Step1.5 \\
      --include "vae/*" "Qwen3-Embedding-0.6B/*" \\
      --local-dir /Volumes/Sandisk_1TB/acestep/Ace-Step1.5

  python3 tests/convert_acestep_weights.py \\
      --src-xl /Volumes/Sandisk_1TB/acestep/acestep-v15-xl-turbo \\
      --src-main /Volumes/Sandisk_1TB/acestep/Ace-Step1.5
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-xl", help="dir of ACE-Step/acestep-v15-xl-turbo snapshot")
    ap.add_argument("--src-main", help="dir of ACE-Step/Ace-Step1.5 snapshot (vae/ + Qwen3-Embedding-0.6B/)")
    ap.add_argument("--out", default=None,
                    help="output dir (default ~/.mlx-serve/models/local/acestep-v15-xl-turbo-{8bit,bf16})")
    ap.add_argument("--bits", type=int, default=8, choices=(8, 16),
                    help="8 = quantize big linears (default), 16 = dense bf16 parity build")
    ap.add_argument("--self-test", action="store_true", help="run synthetic unit tests and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.src_xl or not args.src_main:
        print(DOWNLOAD_HINT)
        sys.exit(1)

    out = args.out
    if out is None:
        suffix = "8bit" if args.bits == 8 else "bf16"
        out = os.path.expanduser(f"~/.mlx-serve/models/local/acestep-v15-xl-turbo-{suffix}")
    convert(args.src_xl, args.src_main, out, args.bits)


if __name__ == "__main__":
    main()
