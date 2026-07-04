#!/usr/bin/env python3
"""Convert Tencent Hunyuan3D-2.1 PAINT (texture) checkpoints into mlx-serve's layout.

USER-RUN (needs torch + mlx + safetensors + the ~9 GB checkpoints). Not run in CI.
Sibling of the SHAPE converter `tests/convert_hunyuan3d_weights.py` — read that first;
this reuses its `should_quantize` predicate, `mx.save_safetensors` path, `--self-test`
and `--bits {8,16}` structure. The BINDING converted-name contract is
`tests/hy3d_paint_weights_contract.md`; this script implements it exactly.

Produces the four-weight paint model dir the native Zig engine loads:

    <out>/config.json          {"model_type":"hunyuan3d_2_1_paint", ...}
    <out>/unet.safetensors      main 2.5D UNet (SD2.1 + MDA/RA/MA/DINO + learned tokens + ImageProj)
    <out>/unet_dual.safetensors reference-stream UNet (plain SD2.1 copy, 4-ch conv_in)
    <out>/vae.safetensors       standard SD2.x AutoencoderKL (encoder + decoder, fp16, NEVER quant)
    <out>/dino.safetensors      DINOv2-giant image encoder (mask_token dropped)
    <out>/LICENSE, <out>/NOTICE copied from --src / reference tree if present

Unlike the SHAPE converter there is NO per-head QKV de-interleave, NO MoE expert stacking,
and NO tensor fusion/splitting: every attention here (base SD, MDA, RA, MA, DINO x-attn, VAE
mid-block) stores to_q/to_k/to_v (query/key/value) as SEPARATE Linears, so the mapping is
strictly 1:1. Only NAMES, CONV LAYOUT (NCHW->NHWC/OHWI), and DTYPE change (+ optional quant).

Transforms (see contract §2):
  T1  conv .weight NCHW->OHWI  transpose(0,2,3,1)                        (all ndim-4 weights)
  T2  1x1 convs stay 4-D OHWI [O,1,1,I]  (NOT flattened to Linear — matches flux/krea VAE)
  T3  drop the ".transformer." wrapper segment (Basic2p5DTransformerBlock)
  T4  flatten ".processor." (MDA/RA extra material projections)
  T5  ".to_out.0"->".to_out", ".to_out_mr.0"->".to_out_mr" (drop the ModuleList index)
  T6  fp32/bf16 -> fp16
  T7  GeGLU/SwiGLU kept WHOLE; split is a forward-time chunk (documented + self-tested)

Usage:
    python3 tests/convert_hunyuan3d_paint_weights.py --src <paint snapshot dir> [--dino DIR] [--out DIR] [--bits {8,16}]
    python3 tests/convert_hunyuan3d_paint_weights.py --self-test   # synthetic unit tests, no ckpt/torch/mlx needed

--src is the dir containing `unet/diffusion_pytorch_model.bin`, `vae/diffusion_pytorch_model.bin`,
`scheduler/scheduler_config.json` (e.g. .../hunyuan3d-paintpbr-v2-1). --dino defaults to a sibling
`dinov2-giant/` next to --src. If --src is omitted the download hint is printed.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# ── config facts (verified 2026-07-04 against ckpts + reference sources) ───────
PBR_SETTINGS = ["albedo", "mr"]
NUM_VIEWS = 6                 # demo max_selected_view_num (6..9 supported)
VIEW_RESOLUTION = 512        # multiview diffusion H=W (demo default; 512 or 768)
GUIDANCE_SCALE = 3.0
NUM_INFERENCE_STEPS = 30

# main UNet (diffusers UNet2DConditionModel, SD2.1 shape)
UNET_IN_CHANNELS = 12        # 4 latent + 4 normal-embeds + 4 position-embeds (config.json LIES with 4)
UNET_OUT_CHANNELS = 4
CROSS_ATTENTION_DIM = 1024
BLOCK_OUT_CHANNELS = [320, 640, 1280, 1280]
LAYERS_PER_BLOCK = 2
ATTENTION_HEAD_DIM = [5, 10, 20, 20]   # diffusers "heads per block"; head_dim = 64 everywhere
UNET_SAMPLE_SIZE = 64
NORM_NUM_GROUPS = 32
UNET_DUAL_IN_CHANNELS = 4

# VAE (diffusers AutoencoderKL, SD2.x)
VAE_BLOCK_OUT_CHANNELS = [128, 256, 512, 512]
VAE_LATENT_CHANNELS = 4
VAE_SAMPLE_SIZE = 768
VAE_SCALING_FACTOR = 0.18215   # NOT in shipped vae/config.json; diffusers SD default

# DINOv2-giant
DINO_HIDDEN = 1536
DINO_LAYERS = 40
DINO_HEADS = 24
DINO_HEAD_DIM = DINO_HIDDEN // DINO_HEADS      # 64
DINO_INTERMEDIATE = 4096
DINO_PATCH = 14
DINO_IMAGE_SIZE = 518
DINO_TOKENS = (DINO_IMAGE_SIZE // DINO_PATCH) ** 2 + 1   # 1370
DINO_LAYERSCALE_INIT = 1.0

# ImageProjModel (lives in unet.safetensors)
IMAGEPROJ_CLIP_DIM = 1536
IMAGEPROJ_NUM_TOKENS = 4

GROUP_SIZE = 64
MIN_QUANT_DIM = 512

# expected source key counts (verified) — sanity-checked at convert time
EXPECT = {"unet": 1061, "unet_dual": 686, "vae": 248, "dino": 727}


# ── (T1/T2) conv layout ───────────────────────────────────────────────────────
def conv_to_ohwi(w):
    """PyTorch conv weight [O,I,kH,kW] -> MLX OHWI [O,kH,kW,I], contiguous.

    Applies to EVERY ndim-4 weight (3x3 and 1x1 alike; 1x1 stays 4-D [O,1,1,I],
    NOT flattened to Linear — matches the flux/krea VAE conv2d convention).
    """
    assert w.ndim == 4, f"conv_to_ohwi expects ndim 4, got {w.shape}"
    return np.ascontiguousarray(np.transpose(w, (0, 2, 3, 1)))


# ── (T3/T4/T5) UNet / unet_dual canonical name ────────────────────────────────
def canon_unet_name(name):
    """Rename a prefix-stripped unet/unet_dual key to its canonical form.

    Injective. Removes the Basic2p5DTransformerBlock ".transformer." wrapper (T3),
    flattens ".processor." (T4), and drops the to_out ModuleList index (T5). All
    other keys (conv_in, resnets, time_embedding, learned_text_clip_*,
    image_proj_model_dino, *norm*, attn_multiview/refview/dino base projections)
    pass through unchanged.
    """
    name = name.replace(".transformer.", ".")     # T3
    name = name.replace(".processor.", ".")        # T4
    name = name.replace(".to_out.0.", ".to_out.")  # T5  (base + attn_multiview/refview/dino)
    name = name.replace(".to_out_mr.0.", ".to_out_mr.")  # T5  (MDA / RA per-material)
    return name


# ── DINOv2-giant canonical name (HF Dinov2Model -> short scheme) ──────────────
_DINO_TOP = {
    "embeddings.cls_token": "cls_token",
    "embeddings.position_embeddings": "pos_embed",
    "embeddings.patch_embeddings.projection.weight": "patch_embed.weight",
    "embeddings.patch_embeddings.projection.bias": "patch_embed.bias",
    "layernorm.weight": "norm.weight",
    "layernorm.bias": "norm.bias",
}
_DINO_LAYER_SUB = {
    "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
    "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
    "attention.attention.query.weight": "attn.q.weight",
    "attention.attention.query.bias": "attn.q.bias",
    "attention.attention.key.weight": "attn.k.weight",
    "attention.attention.key.bias": "attn.k.bias",
    "attention.attention.value.weight": "attn.v.weight",
    "attention.attention.value.bias": "attn.v.bias",
    "attention.output.dense.weight": "attn.out.weight",
    "attention.output.dense.bias": "attn.out.bias",
    "layer_scale1.lambda1": "ls1",
    "layer_scale2.lambda1": "ls2",
    "mlp.weights_in.weight": "mlp.w_in.weight",   # SwiGLU (T7): x1|x2 = chunk(2); out = silu(x1)*x2
    "mlp.weights_in.bias": "mlp.w_in.bias",
    "mlp.weights_out.weight": "mlp.w_out.weight",
    "mlp.weights_out.bias": "mlp.w_out.bias",
}
_DINO_DROP = {"embeddings.mask_token"}   # masked-pretraining token, unused at inference


def dino_canon_name(name):
    """HF Dinov2Model key -> canonical short name. Raises on unknown (strict)."""
    if name in _DINO_TOP:
        return _DINO_TOP[name]
    m = re.match(r"encoder\.layer\.(\d+)\.(.+)$", name)
    if m:
        idx, rest = m.group(1), m.group(2)
        if rest not in _DINO_LAYER_SUB:
            raise KeyError(f"unmapped DINO layer subkey: {rest} (in {name})")
        return f"layers.{idx}.{_DINO_LAYER_SUB[rest]}"
    raise KeyError(f"unmapped DINO key: {name}")


# ── (§8) quantization predicate — identical to the SHAPE converter ────────────
def should_quantize(name, shape, bits):
    """A linear .weight is quantized iff bits==8, ndim in {2,3}, last dim % 64 == 0,
    and min of the last two dims >= 512. Everything else (norms/embeddings/
    layerscales/conv OHWI/1x1/learned tokens/tiny projections) stays fp16.

    ndim 4 (OHWI convs, incl. 1x1 [O,1,1,I]) is excluded automatically."""
    if bits != 8:
        return False
    if not name.endswith(".weight"):
        return False
    if len(shape) not in (2, 3):
        return False
    out_f, in_f = shape[-2], shape[-1]
    if in_f % GROUP_SIZE != 0:
        return False
    return min(out_f, in_f) >= MIN_QUANT_DIM


# ── (T7) GeGLU / SwiGLU split semantics (documentation, kept whole) ───────────
def geglu_split(proj_out):
    """diffusers GEGLU: value, gate = chunk(2, -1); out = value * gelu(gate).
    value = FIRST half, gate = SECOND half."""
    inner = proj_out.shape[-1] // 2
    return proj_out[..., :inner], proj_out[..., inner:]


def swiglu_split(w_in_out):
    """HF Dinov2SwiGLUFFN: x1, x2 = chunk(2, -1); out = silu(x1) * x2.
    x1 = FIRST half (gets SiLU) — the OPPOSITE half from GeGLU's activation."""
    inner = w_in_out.shape[-1] // 2
    return w_in_out[..., :inner], w_in_out[..., inner:]


# ── ckpt / safetensors loading ────────────────────────────────────────────────
def _load_torch_ckpt(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as err:                          # noqa: BLE001
        print(f"[warn] mmap weights_only load failed ({err}); retrying non-mmap", flush=True)
        return torch.load(path, map_location="cpu", weights_only=True)


def _state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt


def t2np(x):
    """Torch tensor -> contiguous numpy float16 (fp32/bf16 cast down)."""
    import torch
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)
    return np.ascontiguousarray(x.detach().cpu().to(torch.float16).numpy())


def subdict_np(sd, prefix):
    """{k[len(prefix):] : fp16 numpy} for every key under `prefix`."""
    return {k[len(prefix):]: t2np(v) for k, v in sd.items() if k.startswith(prefix)}


def load_dino_np(dino_dir):
    """Load facebook/dinov2-giant into {name: fp16 numpy}, per-tensor (low peak)."""
    st = os.path.join(dino_dir, "model.safetensors")
    if os.path.isfile(st):
        from safetensors import safe_open
        out = {}
        with safe_open(st, framework="np") as f:
            for k in f.keys():
                out[k] = np.ascontiguousarray(f.get_tensor(k).astype(np.float16))
        return out
    # fall back to the .bin pickle
    bin_path = os.path.join(dino_dir, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        sd = _state_dict(_load_torch_ckpt(bin_path))
        return {k: t2np(v) for k, v in sd.items()}
    raise SystemExit(f"[FATAL] no model.safetensors / pytorch_model.bin under {dino_dir}")


# ── generic rule-based component conversion ───────────────────────────────────
def convert_component(flat_np, canon_fn, drop_keys, label):
    """Apply canon_fn + conv transpose to every key; drop `drop_keys`; assert 1:1
    (no name collision) and that every non-dropped key is consumed."""
    out = {}
    dropped = []
    src_of = {}
    for k in sorted(flat_np):
        if k in drop_keys:
            dropped.append(k)
            continue
        try:
            name = canon_fn(k)
        except KeyError as err:
            raise SystemExit(f"[FATAL] {label}: {err}")
        arr = flat_np[k]
        if arr.ndim == 4:                     # T1/T2: every ndim-4 weight is a conv
            arr = conv_to_ohwi(arr)
        if name in out:
            raise SystemExit(
                f"[FATAL] {label}: canonical-name COLLISION '{name}' "
                f"from both '{src_of[name]}' and '{k}' (mapping is not 1:1)"
            )
        out[name] = arr
        src_of[name] = k
    if dropped:
        print(f"[{label}] dropped {len(dropped)} key(s): {dropped}")
    print(f"[{label}] mapped {len(out)} tensors (1:1)")
    return out


# ── save (quantize + write), reused from the SHAPE converter ──────────────────
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
def read_scheduler_config(src):
    p = os.path.join(src, "scheduler", "scheduler_config.json")
    if not os.path.isfile(p):
        print(f"[warn] no scheduler_config.json under {src}; using the documented DDIM defaults")
        return {
            "_class_name": "DDIMScheduler", "num_train_timesteps": 1000,
            "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear",
            "prediction_type": "v_prediction", "set_alpha_to_one": True, "steps_offset": 1,
            "timestep_spacing": "trailing", "rescale_betas_zero_snr": True,
            "clip_sample": False, "trained_betas": None,
        }
    with open(p) as f:
        cfg = json.load(f)
    cfg.pop("_diffusers_version", None)
    return cfg


def write_config(out, src, bits):
    cfg = {
        "model_type": "hunyuan3d_2_1_paint",
        "quant": "8bit" if bits == 8 else "fp16",
        "pbr_settings": PBR_SETTINGS,
        "num_views": NUM_VIEWS,
        "view_resolution": VIEW_RESOLUTION,
        "guidance_scale": GUIDANCE_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "primary_camera_azims": [0, 90, 180, 270, 0, 180],
        "primary_camera_elevs": [0, 0, 0, 0, 90, -90],
        "primary_view_weights": [1, 0.1, 0.5, 0.1, 0.05, 0.05],
        "unet": {
            "in_channels": UNET_IN_CHANNELS, "out_channels": UNET_OUT_CHANNELS,
            "cross_attention_dim": CROSS_ATTENTION_DIM, "block_out_channels": BLOCK_OUT_CHANNELS,
            "layers_per_block": LAYERS_PER_BLOCK, "attention_head_dim": ATTENTION_HEAD_DIM,
            "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                                 "CrossAttnDownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D",
                               "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
            "sample_size": UNET_SAMPLE_SIZE, "norm_num_groups": NORM_NUM_GROUPS,
            "norm_eps": 1e-05, "act_fn": "silu", "use_linear_projection": True,
            "transformer_layers_per_block": 1,
        },
        "unet_dual": {"in_channels": UNET_DUAL_IN_CHANNELS},
        "vae": {
            "in_channels": 3, "out_channels": 3, "latent_channels": VAE_LATENT_CHANNELS,
            "block_out_channels": VAE_BLOCK_OUT_CHANNELS, "layers_per_block": 2,
            "norm_num_groups": NORM_NUM_GROUPS, "sample_size": VAE_SAMPLE_SIZE,
            "scaling_factor": VAE_SCALING_FACTOR,
        },
        "dino": {
            "hidden_size": DINO_HIDDEN, "num_layers": DINO_LAYERS, "num_heads": DINO_HEADS,
            "head_dim": DINO_HEAD_DIM, "patch_size": DINO_PATCH, "image_size": DINO_IMAGE_SIZE,
            "num_tokens": DINO_TOKENS, "mlp": "swiglu", "intermediate_size": DINO_INTERMEDIATE,
            "layerscale_init": DINO_LAYERSCALE_INIT, "layer_norm_eps": 1e-06, "qkv_bias": True,
        },
        "image_proj": {
            "clip_embeddings_dim": IMAGEPROJ_CLIP_DIM, "cross_attention_dim": CROSS_ATTENTION_DIM,
            "num_context_tokens": IMAGEPROJ_NUM_TOKENS,
        },
        "scheduler": read_scheduler_config(src),
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] wrote config.json (quant={cfg['quant']})")


def copy_license(src, dino, out):
    import shutil
    # search --src, its parent, --dino, and a sibling reference/hy3dpaint tree
    roots = [src, os.path.dirname(src.rstrip("/")), dino]
    found = []
    for name in ("LICENSE", "NOTICE"):
        cands = []
        for r in roots:
            if not r:
                continue
            cands.append(os.path.join(r, name))
            cands += glob.glob(os.path.join(r, "**", name), recursive=True)
        for c in cands:
            if os.path.isfile(c):
                shutil.copyfile(c, os.path.join(out, name))
                found.append(name)
                break
    if found:
        print(f"[license] copied {', '.join(found)}")
    else:
        print("[license] no LICENSE/NOTICE found — copy the Tencent Hunyuan community license manually")


# ── main conversion ───────────────────────────────────────────────────────────
def convert(src, dino_dir, out, bits):
    unet_bin = os.path.join(src, "unet", "diffusion_pytorch_model.bin")
    vae_bin = os.path.join(src, "vae", "diffusion_pytorch_model.bin")
    for p in (unet_bin, vae_bin):
        if not os.path.isfile(p):
            raise SystemExit(f"[FATAL] missing checkpoint: {p}")
    if not os.path.isdir(dino_dir):
        raise SystemExit(f"[FATAL] missing DINOv2-giant dir: {dino_dir}")

    os.makedirs(out, exist_ok=True)
    summary = []

    # --- UNet .bin (holds unet.* + unet_dual.*) ---
    print(f"[load] UNet ckpt: {unet_bin}")
    unet_sd = _state_dict(_load_torch_ckpt(unet_bin))
    n_unet = sum(1 for k in unet_sd if k.startswith("unet."))
    n_dual = sum(1 for k in unet_sd if k.startswith("unet_dual."))
    if n_unet != EXPECT["unet"] or n_dual != EXPECT["unet_dual"]:
        print(f"[warn] key-count drift: unet.* {n_unet} (exp {EXPECT['unet']}), "
              f"unet_dual.* {n_dual} (exp {EXPECT['unet_dual']}) — verify the checkpoint")

    print("[map] unet ...")
    unet_np = subdict_np(unet_sd, "unet.")
    unet_out = convert_component(unet_np, canon_unet_name, drop_keys=set(), label="unet")
    n, nq, nb = save_safetensors(unet_out, os.path.join(out, "unet.safetensors"), bits)
    summary.append(("unet", n, nq, nb)); del unet_np, unet_out
    print(f"[save] unet.safetensors: {n} tensors ({nq} quantized), {nb / 1e6:.1f} MB")

    print("[map] unet_dual ...")
    dual_np = subdict_np(unet_sd, "unet_dual.")
    dual_out = convert_component(dual_np, canon_unet_name, drop_keys=set(), label="unet_dual")
    n, nq, nb = save_safetensors(dual_out, os.path.join(out, "unet_dual.safetensors"), bits)
    summary.append(("unet_dual", n, nq, nb)); del dual_np, dual_out, unet_sd
    print(f"[save] unet_dual.safetensors: {n} tensors ({nq} quantized), {nb / 1e6:.1f} MB")

    # --- VAE .bin (fp32, never quantized) ---
    print(f"[load] VAE ckpt: {vae_bin}")
    vae_sd = _state_dict(_load_torch_ckpt(vae_bin))
    if len(vae_sd) != EXPECT["vae"]:
        print(f"[warn] VAE key-count drift: {len(vae_sd)} (exp {EXPECT['vae']})")
    vae_np = {k: t2np(v) for k, v in vae_sd.items()}
    vae_out = convert_component(vae_np, lambda k: k, drop_keys=set(), label="vae")
    n, nq, nb = save_safetensors(vae_out, os.path.join(out, "vae.safetensors"), bits=16)  # never quant
    summary.append(("vae", n, nq, nb)); del vae_np, vae_out, vae_sd
    print(f"[save] vae.safetensors: {n} tensors ({nq} quantized), {nb / 1e6:.1f} MB")

    # --- DINOv2-giant ---
    print(f"[load] DINOv2-giant: {dino_dir}")
    dino_np = load_dino_np(dino_dir)
    if len(dino_np) != EXPECT["dino"]:
        print(f"[warn] DINO key-count drift: {len(dino_np)} (exp {EXPECT['dino']})")
    dino_out = convert_component(dino_np, dino_canon_name, drop_keys=_DINO_DROP, label="dino")
    if dino_out.get("pos_embed") is not None and dino_out["pos_embed"].shape[1] != DINO_TOKENS:
        raise SystemExit(f"[FATAL] dino pos_embed token count {dino_out['pos_embed'].shape[1]} "
                         f"!= {DINO_TOKENS}; wrong DINO variant?")
    n, nq, nb = save_safetensors(dino_out, os.path.join(out, "dino.safetensors"), bits)
    summary.append(("dino", n, nq, nb)); del dino_np, dino_out
    print(f"[save] dino.safetensors: {n} tensors ({nq} quantized), {nb / 1e6:.1f} MB")

    write_config(out, src, bits)
    copy_license(src, dino_dir, out)

    total = sum(s[3] for s in summary)
    print("\n[done] wrote paint model dir:", out)
    print(f"[done] total {total / 1e9:.2f} GB across {sum(s[1] for s in summary)} tensors "
          f"({sum(s[2] for s in summary)} quantized @ {bits}-bit)")


DOWNLOAD_HINT = """\
--src not given. Point --src at the paint snapshot dir and --dino at DINOv2-giant, both on an
EXTERNAL disk (they total ~9 GB; do NOT pull onto the internal disk):

  huggingface-cli download tencent/Hunyuan3D-2.1 \\
      hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin \\
      hunyuan3d-paintpbr-v2-1/vae/diffusion_pytorch_model.bin \\
      hunyuan3d-paintpbr-v2-1/scheduler/scheduler_config.json \\
      LICENSE NOTICE \\
      --local-dir /Volumes/Sandisk_1TB/hy3d-ckpts

  huggingface-cli download facebook/dinov2-giant \\
      --local-dir /Volumes/Sandisk_1TB/hy3d-ckpts/dinov2-giant

  python3 tests/convert_hunyuan3d_paint_weights.py \\
      --src /Volumes/Sandisk_1TB/hy3d-ckpts/hunyuan3d-paintpbr-v2-1 \\
      --dino /Volumes/Sandisk_1TB/hy3d-ckpts/dinov2-giant
"""


# ── self-test (no ckpt / torch / mlx needed) ──────────────────────────────────
def self_test():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    rng = np.random.default_rng(0)

    print("[self-test] conv_to_ohwi transposes [O,I,kH,kW] -> [O,kH,kW,I]")
    w = rng.standard_normal((5, 3, 2, 4)).astype(np.float32)
    o = conv_to_ohwi(w)
    check(o.shape == (5, 2, 4, 3), f"shape {o.shape}")
    check(bool(np.all(o[:, 1, 3, 2] == w[:, 2, 1, 3])), "element [.,1,3,2] == source [.,2,1,3]")
    check(o.flags["C_CONTIGUOUS"], "output is contiguous")
    w11 = rng.standard_normal((8, 8, 1, 1)).astype(np.float32)   # 1x1 stays 4-D
    check(conv_to_ohwi(w11).shape == (8, 1, 1, 8), "1x1 conv -> [O,1,1,I] (still 4-D, not linear)")

    print("[self-test] canon_unet_name (T3/T4/T5)")
    umap = [
        ("down_blocks.0.attentions.0.transformer_blocks.0.transformer.attn1.to_q.weight",
         "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"),
        ("down_blocks.0.attentions.0.transformer_blocks.0.transformer.attn1.to_out.0.weight",
         "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.weight"),
        ("down_blocks.0.attentions.0.transformer_blocks.0.transformer.attn1.to_out.0.bias",
         "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.bias"),
        ("mid_block.attentions.0.transformer_blocks.0.transformer.attn1.processor.to_q_mr.weight",
         "mid_block.attentions.0.transformer_blocks.0.attn1.to_q_mr.weight"),
        ("mid_block.attentions.0.transformer_blocks.0.transformer.attn1.processor.to_out_mr.0.weight",
         "mid_block.attentions.0.transformer_blocks.0.attn1.to_out_mr.weight"),
        ("up_blocks.1.attentions.0.transformer_blocks.0.attn_refview.processor.to_v_mr.weight",
         "up_blocks.1.attentions.0.transformer_blocks.0.attn_refview.to_v_mr.weight"),
        ("up_blocks.1.attentions.0.transformer_blocks.0.attn_refview.processor.to_out_mr.0.bias",
         "up_blocks.1.attentions.0.transformer_blocks.0.attn_refview.to_out_mr.bias"),
        ("up_blocks.1.attentions.0.transformer_blocks.0.attn_multiview.to_out.0.weight",
         "up_blocks.1.attentions.0.transformer_blocks.0.attn_multiview.to_out.weight"),
        ("up_blocks.1.attentions.0.transformer_blocks.0.attn_dino.to_k.weight",
         "up_blocks.1.attentions.0.transformer_blocks.0.attn_dino.to_k.weight"),
        ("down_blocks.0.attentions.0.transformer_blocks.0.transformer.ff.net.0.proj.weight",
         "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight"),
        ("down_blocks.0.attentions.0.transformer_blocks.0.transformer.ff.net.2.weight",
         "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.weight"),
        # untouched keys pass through
        ("conv_in.weight", "conv_in.weight"),
        ("learned_text_clip_albedo", "learned_text_clip_albedo"),
        ("image_proj_model_dino.proj.weight", "image_proj_model_dino.proj.weight"),
        ("down_blocks.1.resnets.0.conv_shortcut.weight", "down_blocks.1.resnets.0.conv_shortcut.weight"),
        ("time_embedding.linear_2.weight", "time_embedding.linear_2.weight"),
    ]
    for src_k, want in umap:
        got = canon_unet_name(src_k)
        check(got == want, f"{src_k[-60:]} -> {got}")
    # injectivity on the representative set
    names = [canon_unet_name(k) for k, _ in umap]
    check(len(names) == len(set(names)), "canon_unet_name is injective on the sample set")

    print("[self-test] dino_canon_name")
    dmap = [
        ("embeddings.cls_token", "cls_token"),
        ("embeddings.position_embeddings", "pos_embed"),
        ("embeddings.patch_embeddings.projection.weight", "patch_embed.weight"),
        ("embeddings.patch_embeddings.projection.bias", "patch_embed.bias"),
        ("layernorm.weight", "norm.weight"),
        ("encoder.layer.0.norm1.weight", "layers.0.norm1.weight"),
        ("encoder.layer.7.attention.attention.query.weight", "layers.7.attn.q.weight"),
        ("encoder.layer.7.attention.attention.key.bias", "layers.7.attn.k.bias"),
        ("encoder.layer.39.attention.output.dense.weight", "layers.39.attn.out.weight"),
        ("encoder.layer.3.layer_scale1.lambda1", "layers.3.ls1"),
        ("encoder.layer.3.layer_scale2.lambda1", "layers.3.ls2"),
        ("encoder.layer.12.mlp.weights_in.weight", "layers.12.mlp.w_in.weight"),
        ("encoder.layer.12.mlp.weights_out.bias", "layers.12.mlp.w_out.bias"),
    ]
    for src_k, want in dmap:
        got = dino_canon_name(src_k)
        check(got == want, f"{src_k} -> {got}")
    try:
        dino_canon_name("encoder.layer.0.some.unknown.key")
        check(False, "unknown DINO key should raise")
    except KeyError:
        check(True, "unknown DINO key raises KeyError")

    print("[self-test] GeGLU (value=first, gate=second) vs SwiGLU (x1=first silu, x2=second)")
    out = rng.standard_normal((4, 2560)).astype(np.float32)     # proj output, inner=1280
    v, g = geglu_split(out)
    a, b = np.split(out, 2, axis=-1)                            # diffusers chunk(2)
    check(v.shape == (4, 1280) and np.allclose(v, a) and np.allclose(g, b),
          "GeGLU value==first half, gate==second half")
    win = rng.standard_normal((4, 8192)).astype(np.float32)     # SwiGLU w_in output, inner=4096
    x1, x2 = swiglu_split(win)
    c, d = np.split(win, 2, axis=-1)
    check(x1.shape == (4, 4096) and np.allclose(x1, c) and np.allclose(x2, d),
          "SwiGLU x1==first half (SiLU'd), x2==second half")

    print("[self-test] should_quantize predicate")
    table = [
        # (name, shape, bits, want)
        ("up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight", (1280, 1280), 8, True),
        ("up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight", (1280, 1024), 8, True),
        ("up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight", (10240, 1280), 8, True),
        ("up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.weight", (1280, 5120), 8, True),
        ("time_embedding.linear_2.weight", (1280, 1280), 8, True),
        ("image_proj_model_dino.proj.weight", (4096, 1536), 8, True),
        ("layers.0.attn.q.weight", (1536, 1536), 8, True),                 # DINO attn
        ("layers.0.mlp.w_in.weight", (8192, 1536), 8, True),               # DINO SwiGLU in
        ("layers.0.mlp.w_out.weight", (1536, 4096), 8, True),              # DINO SwiGLU out
        # left fp16
        ("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight", (320, 320), 8, False),
        ("down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight", (2560, 320), 8, False),
        ("down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight", (320, 1024), 8, False),
        ("time_embedding.linear_1.weight", (1280, 320), 8, False),         # min 320 < 512
        ("image_proj_model_dino.norm.weight", (1024,), 8, False),          # 1-D
        ("learned_text_clip_albedo", (77, 1024), 8, False),                # not .weight
        ("cls_token", (1, 1, 1536), 8, False),                             # not .weight
        ("layers.0.ls1", (1536,), 8, False),                               # not .weight
        ("conv_in.weight", (320, 3, 3, 12), 8, False),                     # OHWI conv, ndim 4
        ("decoder.up_blocks.0.resnets.0.conv_shortcut.weight", (256, 1, 1, 512), 8, False),  # 1x1 OHWI
        ("post_quant_conv.weight", (4, 1, 1, 4), 8, False),                # 1x1 OHWI
        ("layers.0.attn.q.bias", (1536,), 8, False),                       # bias
        ("up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight", (1280, 1280), 16, False),  # bits!=8
    ]
    for name, shape, bits, want in table:
        got = should_quantize(name, shape, bits)
        check(got == want, f"{name[-52:]} {tuple(shape)} bits={bits} -> {got}")

    print("\n[self-test] " + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Convert Hunyuan3D-2.1 PAINT checkpoints for mlx-serve.")
    ap.add_argument("--src", default=None,
                    help="paint snapshot dir (contains unet/, vae/, scheduler/)")
    ap.add_argument("--dino", default=None,
                    help="DINOv2-giant dir (default: sibling 'dinov2-giant' next to --src)")
    ap.add_argument("--out", default=None, help="output model dir (default depends on --bits)")
    ap.add_argument("--bits", type=int, default=8, choices=(8, 16),
                    help="8 = mlx affine 8-bit for eligible UNet+DINO linears (default); 16 = fp16 everywhere")
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic unit tests (transforms / renames / quantize rule) and exit")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if args.src is None:
        print(DOWNLOAD_HINT)
        sys.exit(1)

    src = os.path.abspath(args.src)
    dino = os.path.abspath(args.dino) if args.dino else os.path.join(os.path.dirname(src.rstrip("/")), "dinov2-giant")
    out = args.out or os.path.expanduser(
        "~/.mlx-serve/models/local/hunyuan3d-2-1-paint-8bit" if args.bits == 8
        else "~/.mlx-serve/models/local/hunyuan3d-2-1-paint-fp16"
    )
    convert(src, dino, os.path.abspath(out), args.bits)


if __name__ == "__main__":
    main()
