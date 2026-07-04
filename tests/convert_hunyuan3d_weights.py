#!/usr/bin/env python3
"""Convert Tencent Hunyuan3D-2.1 SHAPE checkpoints into mlx-serve's layout.

USER-RUN (needs torch + mlx + the ~8 GB HF checkpoints). Not run in CI. Produces
the three-file model dir the native Zig engine (`src/hunyuan3d.zig`) loads:

    <out>/config.json            {"model_type":"hunyuan3d_2_1", ...}
    <out>/dit.safetensors        the HunYuanDiTPlain denoiser
    <out>/conditioner.safetensors  DINOv2-Large image encoder
    <out>/vae.safetensors        ShapeVAE DECODER + geo_decoder (encoder dropped)
    <out>/LICENSE, <out>/NOTICE  copied from --src if present

The mapping from the reference module attribute names to the BINDING canonical
names in `hy3d_weights_contract.md` was verified against the reference sources
(hy3dshape/models/denoisers/hunyuandit.py, moe_layers.py, conditioner.py,
autoencoders/{model,attention_blocks}.py). Three structural transforms:

  (a) ATTENTION per-head de-interleave. BOTH the VAE and the DiT attentions fuse
      q/k/v (or k/v) then do `cat.view(heads, M*head_dim).split(head_dim)` in the
      forward — a per-head interleave, NOT a plain [q|k|v] block concat. So a plain
      row split silently produces garbage. We bake the permutation at convert time
      (`deinterleave_qkv`) so the Zig side does a STANDARD head reshape. This covers:
        - VAE self-attn   c_qkv[3072,1024]                    -> q,k,v [1024,1024]  (heads 16, hd 64, M 3)
        - VAE geo x-attn  c_kv[2048,1024]                     -> k,v   [1024,1024]  (heads 16, hd 64, M 2); c_q standard
        - DiT self-attn   cat(to_q,to_k,to_v)[6144,2048]      -> q,k,v [2048,2048]  (heads 16, hd 128, M 3)
        - DiT cross-attn  cat(to_k,to_v)[4096,1024]           -> k,v   [2048,1024]  (heads 16, hd 128, M 2); to_q standard
      Per-head order is preserved so the [head_dim] q/k norm weights are unchanged.
  (b) MoE experts stacked into [num_experts, ...] tensors in expert order (gather_qmm layout).
  (c) fp16 dense everywhere (source is fp16); --bits 8 quantizes eligible linears with
      mlx affine group_size 64 (packed uint32 .weight + fp16 .scales/.biases).

Usage:
    python3 tests/convert_hunyuan3d_weights.py --src <HF snapshot or ckpt dir> [--out DIR] [--bits {8,16}]
    python3 tests/convert_hunyuan3d_weights.py --self-test    # synthetic unit tests, no ckpt/torch/mlx needed

If --src is omitted it prints the huggingface-cli download command (targeting an
external-disk scratch dir) and exits.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

# ── config facts (from hunyuan3d-dit-v2-1/config.yaml, verified) ──────────────
HIDDEN = 2048
DEPTH = 21
HEADS = 16
HEAD_DIM = HIDDEN // HEADS          # 128
CONTEXT_DIM = 1024
NUM_LATENTS = 4096
EMBED_DIM = 64
VAE_WIDTH = 1024
VAE_HEADS = 16
VAE_HEAD_DIM = VAE_WIDTH // VAE_HEADS  # 64
VAE_DECODER_LAYERS = 16
NUM_FREQS = 8
SCALE_FACTOR = 1.0039506158752403
NUM_MOE_LAYERS = 6
NUM_EXPERTS = 8
MOE_TOP_K = 2
DINO_HIDDEN = 1024
DINO_LAYERS = 24
DINO_HEADS = 16
DINO_PATCH = 14
DINO_IMAGE_SIZE = 518
DINO_TOKENS = (DINO_IMAGE_SIZE // DINO_PATCH) ** 2 + 1  # 1370

# use_moe when depth - layer <= num_moe_layers  ->  layers 15..20
FIRST_MOE_LAYER = DEPTH - NUM_MOE_LAYERS            # 15
# skip_connection when layer > depth // 2  ->  layers 11..20
FIRST_SKIP_LAYER = DEPTH // 2 + 1                  # 11

GROUP_SIZE = 64


# ── (a) attention per-head de-interleave ─────────────────────────────────────
def deinterleave_qkv(w, heads, head_dim, n_members):
    """Undo the reference `cat.view(heads, M*head_dim).split(head_dim)` interleave.

    `w` is the fused weight of shape [n_members*heads*head_dim, in] (the vertical
    stack of the fused members' output rows). Returns a list of `n_members` arrays,
    each [heads*head_dim, in], such that a STANDARD per-head reshape of `x @ Wm.T`
    reproduces reference member m. Row map:

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
    out = x @ w.T                                   # [.., M*heads*head_dim]
    lead = out.shape[:-1]
    out = out.reshape(*lead, heads, n_members * head_dim)
    return [out[..., m * head_dim:(m + 1) * head_dim] for m in range(n_members)]


# ── (c) quantization rule (pure predicate) ───────────────────────────────────
def should_quantize(name, shape, bits):
    """A linear .weight is quantized iff ndim in {2,3} (3 = stacked experts),
    last dim % GROUP_SIZE == 0, and min of the last two dims >= 512. Everything
    else (norms/embeddings/layerscales/conv/gate/tiny-projections) stays fp16."""
    if bits != 8:
        return False
    if not name.endswith(".weight"):
        return False                                # biases, embeddings (cls_token/pos_embed), layerscales
    if len(shape) not in (2, 3):
        return False                                # 1-D norms, 4-D conv patch-embed
    out_f, in_f = shape[-2], shape[-1]
    if in_f % GROUP_SIZE != 0:
        return False
    return min(out_f, in_f) >= 512


# ── ckpt loading helpers ─────────────────────────────────────────────────────
def _load_torch_ckpt(path):
    import torch
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as err:                        # noqa: BLE001
        print(f"[warn] weights_only=True failed ({err}); retrying weights_only=False", flush=True)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def _flatten_ckpt(ckpt):
    """Return a flat {str: tensor} dict.

    Handles the nested torch form ({'model':sd, 'vae':sd, 'conditioner':sd}) by
    prefixing each namespace, {'state_dict': sd} wrappers, and already-flat dicts.
    """
    import torch

    def is_tensor(v):
        return isinstance(v, torch.Tensor)

    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]

    ns_keys = [k for k in ("model", "vae", "conditioner") if isinstance(ckpt.get(k), dict)]
    if ns_keys:
        flat = {}
        for ns in ns_keys:
            for k, v in ckpt[ns].items():
                flat[f"{ns}.{k}"] = v
        # tolerate stray top-level tensors alongside the namespaces
        for k, v in ckpt.items():
            if k not in ns_keys and is_tensor(v):
                flat[k] = v
        return flat
    return dict(ckpt)


def t2np(x):
    """Torch tensor -> contiguous numpy float16 (fp32/bf16 sources are cast down)."""
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


# ── mapping builders ─────────────────────────────────────────────────────────
def build_dit(src):
    """model.* -> canonical DiT names."""
    out = {}
    out["x_embedder.weight"] = src.pop("x_embedder.weight")
    out["x_embedder.bias"] = src.pop("x_embedder.bias")
    # timestep MLP is nn.Sequential(Linear, GELU, Linear): .0 and .2
    out["t_embedder.mlp1.weight"] = src.pop("t_embedder.mlp.0.weight")
    out["t_embedder.mlp1.bias"] = src.pop("t_embedder.mlp.0.bias")
    out["t_embedder.mlp2.weight"] = src.pop("t_embedder.mlp.2.weight")
    out["t_embedder.mlp2.bias"] = src.pop("t_embedder.mlp.2.bias")

    for i in range(DEPTH):
        b = f"blocks.{i}"
        s = f"blocks.{i}"
        # --- self-attention (attn1): de-interleave to_q/to_k/to_v ---
        out[f"{b}.norm1.weight"] = src.pop(f"{s}.norm1.weight")
        out[f"{b}.norm1.bias"] = src.pop(f"{s}.norm1.bias")
        fused = np.concatenate([
            src.pop(f"{s}.attn1.to_q.weight"),
            src.pop(f"{s}.attn1.to_k.weight"),
            src.pop(f"{s}.attn1.to_v.weight"),
        ], axis=0)
        q, k, v = deinterleave_qkv(fused, HEADS, HEAD_DIM, 3)
        out[f"{b}.attn1.q.weight"], out[f"{b}.attn1.k.weight"], out[f"{b}.attn1.v.weight"] = q, k, v
        out[f"{b}.attn1.q_norm.weight"] = src.pop(f"{s}.attn1.q_norm.weight")
        out[f"{b}.attn1.k_norm.weight"] = src.pop(f"{s}.attn1.k_norm.weight")
        out[f"{b}.attn1.out.weight"] = src.pop(f"{s}.attn1.out_proj.weight")
        out[f"{b}.attn1.out.bias"] = src.pop(f"{s}.attn1.out_proj.bias")
        # --- cross-attention (attn2): q standard, k/v de-interleaved ---
        out[f"{b}.norm2.weight"] = src.pop(f"{s}.norm2.weight")
        out[f"{b}.norm2.bias"] = src.pop(f"{s}.norm2.bias")
        out[f"{b}.attn2.q.weight"] = src.pop(f"{s}.attn2.to_q.weight")
        fused_kv = np.concatenate([
            src.pop(f"{s}.attn2.to_k.weight"),
            src.pop(f"{s}.attn2.to_v.weight"),
        ], axis=0)
        k2, v2 = deinterleave_qkv(fused_kv, HEADS, HEAD_DIM, 2)
        out[f"{b}.attn2.k.weight"], out[f"{b}.attn2.v.weight"] = k2, v2
        out[f"{b}.attn2.q_norm.weight"] = src.pop(f"{s}.attn2.q_norm.weight")
        out[f"{b}.attn2.k_norm.weight"] = src.pop(f"{s}.attn2.k_norm.weight")
        out[f"{b}.attn2.out.weight"] = src.pop(f"{s}.attn2.out_proj.weight")
        out[f"{b}.attn2.out.bias"] = src.pop(f"{s}.attn2.out_proj.bias")
        # --- FFN ---
        out[f"{b}.norm3.weight"] = src.pop(f"{s}.norm3.weight")
        out[f"{b}.norm3.bias"] = src.pop(f"{s}.norm3.bias")
        if i >= FIRST_MOE_LAYER:
            out[f"{b}.moe.gate.weight"] = src.pop(f"{s}.moe.gate.weight")
            fc1_w, fc1_b, fc2_w, fc2_b = [], [], [], []
            for e in range(NUM_EXPERTS):
                ep = f"{s}.moe.experts.{e}"
                fc1_w.append(src.pop(f"{ep}.net.0.proj.weight"))
                fc1_b.append(src.pop(f"{ep}.net.0.proj.bias"))
                fc2_w.append(src.pop(f"{ep}.net.2.weight"))
                fc2_b.append(src.pop(f"{ep}.net.2.bias"))
            out[f"{b}.moe.experts.fc1.weight"] = np.stack(fc1_w, axis=0)
            out[f"{b}.moe.experts.fc1.bias"] = np.stack(fc1_b, axis=0)
            out[f"{b}.moe.experts.fc2.weight"] = np.stack(fc2_w, axis=0)
            out[f"{b}.moe.experts.fc2.bias"] = np.stack(fc2_b, axis=0)
            out[f"{b}.moe.shared.fc1.weight"] = src.pop(f"{s}.moe.shared_experts.net.0.proj.weight")
            out[f"{b}.moe.shared.fc1.bias"] = src.pop(f"{s}.moe.shared_experts.net.0.proj.bias")
            out[f"{b}.moe.shared.fc2.weight"] = src.pop(f"{s}.moe.shared_experts.net.2.weight")
            out[f"{b}.moe.shared.fc2.bias"] = src.pop(f"{s}.moe.shared_experts.net.2.bias")
        else:
            out[f"{b}.mlp.fc1.weight"] = src.pop(f"{s}.mlp.fc1.weight")
            out[f"{b}.mlp.fc1.bias"] = src.pop(f"{s}.mlp.fc1.bias")
            out[f"{b}.mlp.fc2.weight"] = src.pop(f"{s}.mlp.fc2.weight")
            out[f"{b}.mlp.fc2.bias"] = src.pop(f"{s}.mlp.fc2.bias")
        # --- U-ViT skip ---
        if i >= FIRST_SKIP_LAYER:
            out[f"{b}.skip.linear.weight"] = src.pop(f"{s}.skip_linear.weight")
            out[f"{b}.skip.linear.bias"] = src.pop(f"{s}.skip_linear.bias")
            out[f"{b}.skip.norm.weight"] = src.pop(f"{s}.skip_norm.weight")
            out[f"{b}.skip.norm.bias"] = src.pop(f"{s}.skip_norm.bias")

    out["final.norm.weight"] = src.pop("final_layer.norm_final.weight")
    out["final.norm.bias"] = src.pop("final_layer.norm_final.bias")
    out["final.linear.weight"] = src.pop("final_layer.linear.weight")
    out["final.linear.bias"] = src.pop("final_layer.linear.bias")
    return out


def build_conditioner(src):
    """conditioner.main_image_encoder.model.* (HF Dinov2Model) -> canonical names."""
    out = {}
    out["cls_token"] = src.pop("embeddings.cls_token")
    pos = src.pop("embeddings.position_embeddings")
    if pos.shape[1] != DINO_TOKENS:
        raise SystemExit(
            f"[FATAL] conditioner: pos_embed token count {pos.shape[1]} != {DINO_TOKENS} "
            f"(image_size {DINO_IMAGE_SIZE} / patch {DINO_PATCH}); wrong DINO variant?"
        )
    out["pos_embed"] = pos
    out["patch_embed.weight"] = src.pop("embeddings.patch_embeddings.projection.weight")
    out["patch_embed.bias"] = src.pop("embeddings.patch_embeddings.projection.bias")

    for i in range(DINO_LAYERS):
        b = f"layers.{i}"
        s = f"encoder.layer.{i}"
        out[f"{b}.norm1.weight"] = src.pop(f"{s}.norm1.weight")
        out[f"{b}.norm1.bias"] = src.pop(f"{s}.norm1.bias")
        # HF Dinov2SelfAttention is STANDARD (no fused interleave) -> plain rename
        out[f"{b}.attn.q.weight"] = src.pop(f"{s}.attention.attention.query.weight")
        out[f"{b}.attn.q.bias"] = src.pop(f"{s}.attention.attention.query.bias")
        out[f"{b}.attn.k.weight"] = src.pop(f"{s}.attention.attention.key.weight")
        out[f"{b}.attn.k.bias"] = src.pop(f"{s}.attention.attention.key.bias")
        out[f"{b}.attn.v.weight"] = src.pop(f"{s}.attention.attention.value.weight")
        out[f"{b}.attn.v.bias"] = src.pop(f"{s}.attention.attention.value.bias")
        out[f"{b}.attn.out.weight"] = src.pop(f"{s}.attention.output.dense.weight")
        out[f"{b}.attn.out.bias"] = src.pop(f"{s}.attention.output.dense.bias")
        out[f"{b}.ls1"] = src.pop(f"{s}.layer_scale1.lambda1")
        out[f"{b}.norm2.weight"] = src.pop(f"{s}.norm2.weight")
        out[f"{b}.norm2.bias"] = src.pop(f"{s}.norm2.bias")
        out[f"{b}.mlp.fc1.weight"] = src.pop(f"{s}.mlp.fc1.weight")
        out[f"{b}.mlp.fc1.bias"] = src.pop(f"{s}.mlp.fc1.bias")
        out[f"{b}.mlp.fc2.weight"] = src.pop(f"{s}.mlp.fc2.weight")
        out[f"{b}.mlp.fc2.bias"] = src.pop(f"{s}.mlp.fc2.bias")
        out[f"{b}.ls2"] = src.pop(f"{s}.layer_scale2.lambda1")

    out["norm.weight"] = src.pop("layernorm.weight")
    out["norm.bias"] = src.pop("layernorm.bias")
    return out


def build_vae(src):
    """vae.* -> canonical DECODER + geo names (encoder dropped by leftover policy)."""
    out = {}
    out["post_kl.weight"] = src.pop("post_kl.weight")
    out["post_kl.bias"] = src.pop("post_kl.bias")

    for i in range(VAE_DECODER_LAYERS):
        b = f"blocks.{i}"
        s = f"transformer.resblocks.{i}"
        out[f"{b}.ln1.weight"] = src.pop(f"{s}.ln_1.weight")
        out[f"{b}.ln1.bias"] = src.pop(f"{s}.ln_1.bias")
        # VAE self-attn: fused c_qkv -> de-interleave q/k/v (heads 16, head_dim 64)
        q, k, v = deinterleave_qkv(src.pop(f"{s}.attn.c_qkv.weight"), VAE_HEADS, VAE_HEAD_DIM, 3)
        out[f"{b}.attn.q.weight"], out[f"{b}.attn.k.weight"], out[f"{b}.attn.v.weight"] = q, k, v
        out[f"{b}.attn.q_norm.weight"] = src.pop(f"{s}.attn.attention.q_norm.weight")
        out[f"{b}.attn.q_norm.bias"] = src.pop(f"{s}.attn.attention.q_norm.bias")
        out[f"{b}.attn.k_norm.weight"] = src.pop(f"{s}.attn.attention.k_norm.weight")
        out[f"{b}.attn.k_norm.bias"] = src.pop(f"{s}.attn.attention.k_norm.bias")
        out[f"{b}.attn.out.weight"] = src.pop(f"{s}.attn.c_proj.weight")
        out[f"{b}.attn.out.bias"] = src.pop(f"{s}.attn.c_proj.bias")
        out[f"{b}.ln2.weight"] = src.pop(f"{s}.ln_2.weight")
        out[f"{b}.ln2.bias"] = src.pop(f"{s}.ln_2.bias")
        out[f"{b}.mlp.fc1.weight"] = src.pop(f"{s}.mlp.c_fc.weight")
        out[f"{b}.mlp.fc1.bias"] = src.pop(f"{s}.mlp.c_fc.bias")
        out[f"{b}.mlp.fc2.weight"] = src.pop(f"{s}.mlp.c_proj.weight")
        out[f"{b}.mlp.fc2.bias"] = src.pop(f"{s}.mlp.c_proj.bias")

    g = "geo_decoder"
    ca = f"{g}.cross_attn_decoder"
    out["geo.query_proj.weight"] = src.pop(f"{g}.query_proj.weight")
    out["geo.query_proj.bias"] = src.pop(f"{g}.query_proj.bias")
    out["geo.ln1.weight"] = src.pop(f"{ca}.ln_1.weight")
    out["geo.ln1.bias"] = src.pop(f"{ca}.ln_1.bias")
    out["geo.ln2.weight"] = src.pop(f"{ca}.ln_2.weight")
    out["geo.ln2.bias"] = src.pop(f"{ca}.ln_2.bias")
    out["geo.ln3.weight"] = src.pop(f"{ca}.ln_3.weight")
    out["geo.ln3.bias"] = src.pop(f"{ca}.ln_3.bias")
    # geo cross-attn: c_q standard, c_kv de-interleaved (heads 16, head_dim 64)
    out["geo.attn.q.weight"] = src.pop(f"{ca}.attn.c_q.weight")
    gk, gv = deinterleave_qkv(src.pop(f"{ca}.attn.c_kv.weight"), VAE_HEADS, VAE_HEAD_DIM, 2)
    out["geo.attn.k.weight"], out["geo.attn.v.weight"] = gk, gv
    out["geo.attn.q_norm.weight"] = src.pop(f"{ca}.attn.attention.q_norm.weight")
    out["geo.attn.q_norm.bias"] = src.pop(f"{ca}.attn.attention.q_norm.bias")
    out["geo.attn.k_norm.weight"] = src.pop(f"{ca}.attn.attention.k_norm.weight")
    out["geo.attn.k_norm.bias"] = src.pop(f"{ca}.attn.attention.k_norm.bias")
    out["geo.attn.out.weight"] = src.pop(f"{ca}.attn.c_proj.weight")
    out["geo.attn.out.bias"] = src.pop(f"{ca}.attn.c_proj.bias")
    out["geo.mlp.fc1.weight"] = src.pop(f"{ca}.mlp.c_fc.weight")
    out["geo.mlp.fc1.bias"] = src.pop(f"{ca}.mlp.c_fc.bias")
    out["geo.mlp.fc2.weight"] = src.pop(f"{ca}.mlp.c_proj.weight")
    out["geo.mlp.fc2.bias"] = src.pop(f"{ca}.mlp.c_proj.bias")
    out["geo.ln_post.weight"] = src.pop(f"{g}.ln_post.weight")
    out["geo.ln_post.bias"] = src.pop(f"{g}.ln_post.bias")
    out["geo.out_proj.weight"] = src.pop(f"{g}.output_proj.weight")
    out["geo.out_proj.bias"] = src.pop(f"{g}.output_proj.bias")
    return out


# leftover keys that are legitimately dropped (not an error), per namespace.
DROP_RULES = {
    "dit": [],
    "conditioner": [
        ("embeddings.mask_token", "masked-pretraining token, unused at inference"),
    ],
    "vae": [
        ("pre_kl.", "VAE encoder head (posterior mean/logvar), not used for decode"),
        ("encoder.", "VAE point-cloud encoder, not used for decode"),
        ("fourier_embedder.", "non-persistent buffer (should be absent)"),
    ],
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
        shown = {}
        for k, why in dropped:
            shown.setdefault(why, 0)
            shown[why] += 1
        for why, n in shown.items():
            print(f"    {n:5d}  {why}")
    if fatal:
        raise SystemExit(
            f"[FATAL] {ns}: {len(fatal)} UNMAPPED source key(s) with no drop rule "
            f"(fix the mapping or add a DROP_RULES entry):\n" +
            "\n".join(f"    {k}" for k, _ in fatal[:40])
        )


# ── save (quantize + write) ──────────────────────────────────────────────────
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


# ── src resolution ───────────────────────────────────────────────────────────
def _find_ckpt(src, kind_globs):
    for g in kind_globs:
        hits = sorted(glob.glob(os.path.join(src, g)))
        if hits:
            return hits[0]
    return None


def resolve_ckpts(src):
    dit = _find_ckpt(src, [
        "hunyuan3d-dit-v2-1/model.fp16.ckpt", "hunyuan3d-dit-v2-1/model*.ckpt",
        "*dit*/model.fp16.ckpt", "*dit*/model*.ckpt",
    ])
    vae = _find_ckpt(src, [
        "hunyuan3d-vae-v2-1/model.fp16.ckpt", "hunyuan3d-vae-v2-1/model*.ckpt",
        "*vae*/model.fp16.ckpt", "*vae*/model*.ckpt",
    ])
    if dit is None and os.path.isfile(src) and src.endswith(".ckpt"):
        dit = src
    return dit, vae


DOWNLOAD_HINT = """\
--src not given. Download the two SHAPE checkpoints to an EXTERNAL-disk scratch dir
(they are ~8 GB; do NOT pull them onto the internal disk), then re-run with --src:

  huggingface-cli download tencent/Hunyuan3D-2.1 \\
      hunyuan3d-dit-v2-1/model.fp16.ckpt \\
      hunyuan3d-vae-v2-1/model.fp16.ckpt \\
      LICENSE NOTICE \\
      --local-dir /Volumes/Sandisk_1TB/hy3d-ckpts

  python3 tests/convert_hunyuan3d_weights.py --src /Volumes/Sandisk_1TB/hy3d-ckpts
"""


def copy_license(src, out):
    import shutil
    found = []
    for name in ("LICENSE", "NOTICE"):
        for cand in [os.path.join(src, name)] + glob.glob(os.path.join(src, "**", name), recursive=True):
            if os.path.isfile(cand):
                shutil.copyfile(cand, os.path.join(out, name))
                found.append(name)
                break
    if found:
        print(f"[license] copied {', '.join(found)}")
    else:
        print("[license] no LICENSE/NOTICE found under --src (copy the Tencent Hunyuan community license manually)")


def write_config(out, bits):
    cfg = {
        "model_type": "hunyuan3d_2_1",
        "quant": "8bit" if bits == 8 else "fp16",
        "hidden_size": HIDDEN, "depth": DEPTH, "num_heads": HEADS,
        "context_dim": CONTEXT_DIM, "num_latents": NUM_LATENTS, "embed_dim": EMBED_DIM,
        "vae_width": VAE_WIDTH, "vae_heads": VAE_HEADS, "vae_decoder_layers": VAE_DECODER_LAYERS,
        "num_freqs": NUM_FREQS, "scale_factor": SCALE_FACTOR,
        "num_moe_layers": NUM_MOE_LAYERS, "num_experts": NUM_EXPERTS, "moe_top_k": MOE_TOP_K,
        "dino_hidden": DINO_HIDDEN, "dino_layers": DINO_LAYERS, "dino_heads": DINO_HEADS,
        "dino_patch": DINO_PATCH, "dino_image_size": DINO_IMAGE_SIZE,
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] wrote config.json (quant={cfg['quant']})")


def convert(src, out, bits):
    dit_ckpt, vae_ckpt = resolve_ckpts(src)
    if dit_ckpt is None:
        raise SystemExit(f"[FATAL] could not find the DiT ckpt (model.fp16.ckpt) under {src}")
    print(f"[load] DiT ckpt: {dit_ckpt}")
    dit_flat_t = _flatten_ckpt(_load_torch_ckpt(dit_ckpt))

    # split namespaces of the DiT ckpt
    model_np = {k[len("model."):]: t2np(v) for k, v in dit_flat_t.items() if k.startswith("model.")}
    CP = "conditioner.main_image_encoder.model."
    cond_np = {k[len(CP):]: t2np(v) for k, v in dit_flat_t.items() if k.startswith(CP)}
    other_cond = [k for k in dit_flat_t if k.startswith("conditioner.") and not k.startswith(CP)]
    if other_cond:
        print(f"[conditioner] dropping {len(other_cond)} non-'{CP}*' conditioner key(s) "
              f"(e.g. {other_cond[:3]})")
    vae_np = {k[len("vae."):]: t2np(v) for k, v in dit_flat_t.items() if k.startswith("vae.")}

    if not model_np:
        raise SystemExit("[FATAL] DiT ckpt has no 'model.*' keys — unexpected layout")
    if not cond_np:
        raise SystemExit(f"[FATAL] DiT ckpt has no '{CP}*' conditioner keys — unexpected layout")

    if vae_np:
        print(f"[vae] using VAE bundled in the DiT ckpt ({len(vae_np)} tensors; matches runtime pipeline)")
    else:
        if vae_ckpt is None:
            raise SystemExit("[FATAL] DiT ckpt has no 'vae.*' keys and no standalone VAE ckpt found")
        print(f"[load] standalone VAE ckpt: {vae_ckpt}")
        vflat = _flatten_ckpt(_load_torch_ckpt(vae_ckpt))
        # standalone may be flat (post_kl.* ...) or nested under 'vae.'
        if any(k.startswith("vae.") for k in vflat):
            vae_np = {k[len("vae."):]: t2np(v) for k, v in vflat.items() if k.startswith("vae.")}
        else:
            vae_np = {k: t2np(v) for k, v in vflat.items()}

    os.makedirs(out, exist_ok=True)

    print("[map] DiT ...")
    dit_src = Source(model_np, "dit")
    dit_out = build_dit(dit_src)
    enforce_leftovers(dit_src, "dit")

    print("[map] conditioner ...")
    cond_src = Source(cond_np, "conditioner")
    cond_out = build_conditioner(cond_src)
    enforce_leftovers(cond_src, "conditioner")

    print("[map] VAE ...")
    vae_src = Source(vae_np, "vae")
    vae_out = build_vae(vae_src)
    enforce_leftovers(vae_src, "vae")

    summary = []
    for name, table in [("dit", dit_out), ("conditioner", cond_out), ("vae", vae_out)]:
        n, nq, nbytes = save_safetensors(table, os.path.join(out, f"{name}.safetensors"), bits)
        summary.append((name, n, nq, nbytes))
        print(f"[save] {name}.safetensors: {n} tensors ({nq} quantized), {nbytes / 1e6:.1f} MB")

    write_config(out, bits)
    copy_license(src, out)

    total = sum(s[3] for s in summary)
    print("\n[done] wrote model dir:", out)
    print(f"[done] total {total / 1e9:.2f} GB across "
          f"{sum(s[1] for s in summary)} tensors "
          f"({sum(s[2] for s in summary)} quantized @ {bits}-bit)")


# ── self-test ────────────────────────────────────────────────────────────────
def self_test():
    rng = np.random.default_rng(0)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print("[self-test] deinterleave_qkv reproduces the reference view+split")
    # cover all four attention shapes used by the engine
    cases = [
        ("VAE self  ", 16, 64, 3, 1024),
        ("VAE geo kv", 16, 64, 2, 1024),
        ("DiT self  ", 16, 128, 3, 2048),
        ("DiT cross ", 16, 128, 2, 1024),
        ("tiny      ", 2, 2, 3, 4),  # hand-checkable
    ]
    for label, heads, hd, m, in_f in cases:
        out_rows = m * heads * hd
        w = rng.standard_normal((out_rows, in_f)).astype(np.float32)
        x = rng.standard_normal((5, in_f)).astype(np.float32)
        ref = _reference_view_split(x, w, heads, hd, m)
        members = deinterleave_qkv(w, heads, hd, m)
        good = True
        for mi in range(m):
            mine = (x @ members[mi].T).reshape(5, heads, hd)
            good = good and np.allclose(mine, ref[mi], atol=1e-4)
        check(good, f"{label} heads={heads} hd={hd} M={m} in={in_f}")

    # explicit hand-derived tiny case: the doc example
    print("[self-test] deinterleave_qkv row map matches the hand-derived tiny example")
    w = np.arange(3 * 2 * 2).reshape(12, 1).astype(np.float32)  # rows 0..11 as ids
    q, k, v = deinterleave_qkv(w, heads=2, head_dim=2, n_members=3)
    # head0 block = rows 0..5 -> q[0:2]=0,1 k[0:2]=2,3 v[0:2]=4,5; head1 = rows 6..11
    check(q[:, 0].tolist() == [0, 1, 6, 7], "q rows = [0,1,6,7]")
    check(k[:, 0].tolist() == [2, 3, 8, 9], "k rows = [2,3,8,9]")
    check(v[:, 0].tolist() == [4, 5, 10, 11], "v rows = [4,5,10,11]")

    print("[self-test] expert stacking preserves expert order")
    experts = [np.full((3, 4), e, dtype=np.float16) for e in range(NUM_EXPERTS)]
    stacked = np.stack(experts, axis=0)
    check(stacked.shape == (NUM_EXPERTS, 3, 4), f"shape {stacked.shape}")
    check(all(int(stacked[e, 0, 0]) == e for e in range(NUM_EXPERTS)), "order 0..7 preserved")

    print("[self-test] should_quantize predicate")
    table = [
        ("blocks.0.attn1.q.weight", (2048, 2048), 8, True),
        ("blocks.0.attn2.k.weight", (2048, 1024), 8, True),
        ("blocks.15.moe.experts.fc1.weight", (8, 8192, 2048), 8, True),
        ("blocks.15.moe.experts.fc2.weight", (8, 2048, 8192), 8, True),
        ("blocks.15.moe.gate.weight", (8, 2048), 8, False),      # min(8,2048)<512
        ("x_embedder.weight", (2048, 64), 8, False),             # min<512
        ("final.linear.weight", (64, 2048), 8, False),           # min<512
        ("geo.query_proj.weight", (1024, 51), 8, False),         # 51%64!=0
        ("geo.out_proj.weight", (1, 1024), 8, False),            # min<512
        ("post_kl.weight", (1024, 64), 8, False),                # min<512
        ("patch_embed.weight", (1024, 3, 14, 14), 8, False),     # ndim 4 (conv)
        ("pos_embed", (1, 1370, 1024), 8, False),                # not .weight
        ("cls_token", (1, 1, 1024), 8, False),                   # not .weight
        ("layers.0.ls1", (1024,), 8, False),                     # layerscale, not .weight
        ("blocks.0.norm1.weight", (2048,), 8, False),            # 1-D norm
        ("blocks.0.attn1.out.bias", (2048,), 8, False),          # bias
        ("blocks.15.moe.experts.fc1.bias", (8, 8192), 8, False), # stacked bias, not .weight
        ("layers.0.mlp.fc1.weight", (4096, 1024), 8, True),      # DINO mlp
        ("blocks.0.attn1.q.weight", (2048, 2048), 16, False),    # bits=16 -> never
    ]
    for name, shape, bits, want in table:
        got = should_quantize(name, shape, bits)
        check(got == want, f"{name} {tuple(shape)} bits={bits} -> {got}")

    print("\n[self-test] " + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Convert Hunyuan3D-2.1 shape checkpoints for mlx-serve.")
    ap.add_argument("--src", default=None,
                    help="HF snapshot dir (contains hunyuan3d-dit-v2-1/ and hunyuan3d-vae-v2-1/) or a ckpt dir")
    ap.add_argument("--out", default=None, help="output model dir (default depends on --bits)")
    ap.add_argument("--bits", type=int, default=8, choices=(8, 16),
                    help="8 = mlx affine 8-bit for eligible linears (default); 16 = fp16 everywhere")
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic unit tests (deinterleave / expert stacking / quantize rule) and exit")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if args.src is None:
        print(DOWNLOAD_HINT)
        sys.exit(1)

    out = args.out or os.path.expanduser(
        "~/.mlx-serve/models/local/hunyuan3d-2-1-8bit" if args.bits == 8
        else "~/.mlx-serve/models/local/hunyuan3d-2-1-fp16"
    )
    convert(os.path.abspath(args.src), os.path.abspath(out), args.bits)


if __name__ == "__main__":
    main()
