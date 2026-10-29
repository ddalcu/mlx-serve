#!/usr/bin/env python3
"""Oracle for src/plugins/trellis2/trellis2.zig: Microsoft's TRELLIS.2-4B reference modules run in
fp32 on the CPU, on the real weights.

The reference needs CUDA for flash-attn, FlexGEMM's sparse conv, CuMesh and
o_voxel. Their shims below are pure torch and follow FlexGEMM's own torch
reference (`EXPLICIT_GEMM`: im2col over the neighbour map, no kernel flip,
kernel axes k0,k1,k2 over coords[:,1..3] row-major), so the model code under
test is Microsoft's, unchanged.

USER-RUN. Needs torch + safetensors + numpy and two checkouts:

  git clone https://github.com/microsoft/TRELLIS.2        <ref>
  hf download microsoft/TRELLIS.2-4B --include 'ckpts/*'  --local-dir <weights>
  hf download microsoft/TRELLIS-image-large \\
      --include 'ckpts/ss_dec_conv3d_16l8_fp16.*' --local-dir <ss-dec>

  python -I tests/dump_trellis2_fixtures.py --ref <ref> --weights <weights> \\
      --ss-dec <ss-dec> --out <fixtures dir>
  TRELLIS2_FIXTURES=<out> TRELLIS2_WEIGHTS=<weights> TRELLIS2_SS_DEC=<ss-dec>/ckpts \\
      zig build test -Doptimize=ReleaseFast -Dtest-filter="trellis2"

Writes one safetensors per stage under --out (all f32 / i32, seed 0):
  ss_dec, ss_flow, slat_shape512, slat_tex512, shape_dec, tex_dec
and, from transformers' own DINOv3ViTModel on a TINY random model (the real checkpoint is gated):
  dinov3.safetensors (input + features) and dinov3_tiny/{config.json,model.safetensors} (the weights in
  the published `layer.N.*` layout)
"""

import argparse
import importlib
import importlib.machinery
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

SEED = 0
# One timestep as the sampler passes it (1000 * t).
T_MODEL = 437.5
# DINOv3 ViT-L/16 at 512 px: 1 CLS + 4 registers + 32x32 patches.
COND_TOKENS = 1 + 4 + 32 * 32
COND_DIM = 1024


# ── shims for the CUDA-only imports ───────────────────────────────────────────


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)  # find_spec() callers (transformers) need one
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _varlen_sdpa(q, k, v, cu_q, cu_k):
    outs = []
    for i in range(len(cu_q) - 1):
        qi = q[cu_q[i]:cu_q[i + 1]].permute(1, 0, 2)
        ki = k[cu_k[i]:cu_k[i + 1]].permute(1, 0, 2)
        vi = v[cu_k[i]:cu_k[i + 1]].permute(1, 0, 2)
        outs.append(F.scaled_dot_product_attention(qi, ki, vi).permute(1, 0, 2))
    return torch.cat(outs, 0)


def neighbor_map(coords, spatial, kernel, dilation):
    """[N, V] long, -1 where no active voxel sits at that offset."""
    n = coords.shape[0]
    w, h, d = spatial
    c = coords.long()
    key = ((c[:, 0] * w + c[:, 1]) * h + c[:, 2]) * d + c[:, 3]
    sorted_key, order = torch.sort(key)
    offs = torch.stack(torch.meshgrid(*[
        torch.arange(-(k // 2) * dl, k // 2 * dl + 1, dl) for k, dl in zip(kernel, dilation)
    ], indexing="ij"), dim=-1).reshape(-1, 3)
    nb = c[:, None, :].repeat(1, offs.shape[0], 1)
    nb[:, :, 1:] += offs[None]
    nb = nb.reshape(-1, 4)
    ok = (nb[:, 1] >= 0) & (nb[:, 1] < w) & (nb[:, 2] >= 0) & (nb[:, 2] < h) & (nb[:, 3] >= 0) & (nb[:, 3] < d)
    nkey = ((nb[:, 0] * w + nb[:, 1]) * h + nb[:, 2]) * d + nb[:, 3]
    pos = torch.clamp(torch.searchsorted(sorted_key, nkey), 0, n - 1)
    ok &= sorted_key[pos] == nkey
    out = torch.full((nb.shape[0],), -1, dtype=torch.long)
    out[ok] = order[pos[ok]]
    return out.reshape(n, -1)


def sparse_submanifold_conv3d(feats, coords, shape, weight, bias, cache, dilation):
    _, _, w, h, d = shape
    co, k0, k1, k2, ci = weight.shape
    if cache is None:
        cache = neighbor_map(coords, (w, h, d), (k0, k1, k2), dilation)
    n, v = cache.shape
    col = torch.zeros(n * v, ci, dtype=feats.dtype)
    m = cache.reshape(-1) >= 0
    col[m] = feats[cache.reshape(-1)[m]]
    out = col.reshape(n, v * ci) @ weight.reshape(co, v * ci).T
    if bias is not None:
        out = out + bias
    return out, cache


def install_shims():
    fa = _module(
        "flash_attn",
        flash_attn_varlen_qkvpacked_func=lambda qkv, cu, mx: _varlen_sdpa(*qkv.unbind(1), cu, cu),
        flash_attn_varlen_kvpacked_func=lambda q, kv, cq, ck, mq, mk: _varlen_sdpa(q, *kv.unbind(1), cq, ck),
        flash_attn_varlen_func=lambda q, k, v, cq, ck, mq, mk: _varlen_sdpa(q, k, v, cq, ck),
    )
    spconv = _module("flex_gemm.ops.spconv", set_algorithm=lambda a: None, set_hashmap_ratio=lambda r: None,
                     sparse_submanifold_conv3d=sparse_submanifold_conv3d)
    grid = _module("flex_gemm.ops.grid_sample", grid_sample_3d=None)
    ops = _module("flex_gemm.ops", spconv=spconv, grid_sample=grid)
    _module("flex_gemm", ops=ops)
    _module("cumesh")
    ov = _module("o_voxel")
    ov.convert = _module("o_voxel.convert", flexible_dual_grid_to_mesh=flexible_dual_grid_to_mesh)


def flexible_dual_grid_to_mesh(coords, dual_vertices, intersected_flag, split_weight, aabb, voxel_size=None,
                               grid_size=None, train=False):
    """o_voxel's `flexible_dual_grid_to_mesh` (inference), with a dict for the CUDA hashmap."""
    assert not train
    aabb = torch.tensor(aabb, dtype=torch.float32)
    if isinstance(grid_size, int):
        grid_size = [grid_size] * 3
    grid_size = torch.tensor(grid_size, dtype=torch.int32)
    voxel_size = (aabb[1] - aabb[0]) / grid_size
    n = dual_vertices.shape[0]
    offs = torch.tensor([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ], dtype=torch.int).unsqueeze(0)
    table = {tuple(c): i for i, c in enumerate(coords.tolist())}
    nb = coords.reshape(n, 1, 1, 3) + offs
    conn = nb[intersected_flag]
    m = conn.shape[0]
    idx = torch.tensor([table.get(tuple(c), -1) for c in conn.reshape(-1, 3).tolist()], dtype=torch.long).reshape(m, 4)
    quads = idx[(idx >= 0).all(dim=1)]
    verts = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)
    s1 = torch.tensor([0, 1, 2, 0, 2, 3])
    s2 = torch.tensor([0, 1, 3, 3, 1, 2])
    sw = split_weight[quads]
    w02 = sw[:, 0] * sw[:, 2]
    w13 = sw[:, 1] * sw[:, 3]
    tris = torch.where(w02 > w13, quads[:, s1], quads[:, s2]).reshape(-1, 3)
    return verts, tris


# ── model construction ───────────────────────────────────────────────────────


def build(ref, ckpt_stem, dtype=torch.float32):
    """The reference module for `<ckpt_stem>.json/.safetensors`, fp32 on the CPU."""
    cfg = json.load(open(ckpt_stem + ".json"))
    args = dict(cfg["args"])
    if "dtype" in args:
        args["dtype"] = "float32"
    if "use_fp16" in args:
        args["use_fp16"] = False
    models = importlib.import_module("trellis2.models")
    m = getattr(models, cfg["name"])(**args)
    sd = load_file(ckpt_stem + ".safetensors")
    missing, unexpected = m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    # RoPE phase tables are rebuilt at construction; anything else missing is a bug.
    missing = [k for k in missing if "rope_phases" not in k and "pos_emb" not in k]
    assert not missing, f"missing weights: {missing[:5]}"
    assert not unexpected, f"unexpected weights: {unexpected[:5]}"
    return m.float().eval()


def f32(x):
    return x.detach().float().contiguous()


def tap(store, name):
    def hook(_m, _i, out):
        store[name] = f32(out.feats if hasattr(out, "feats") else out)
    return hook


def blob_coords(rng, grid, count):
    """`count` distinct coords of a `grid`^3 volume inside a ball at its centre."""
    c = (grid - 1) / 2
    pts = np.array([(x, y, z) for x in range(grid) for y in range(grid) for z in range(grid)])
    d = np.linalg.norm(pts - c, axis=1) + rng.uniform(0, 0.5, len(pts))
    keep = pts[np.argsort(d)[:count]]
    keep = keep[np.lexsort((keep[:, 2], keep[:, 1], keep[:, 0]))]
    return torch.from_numpy(np.concatenate([np.zeros((count, 1), dtype=np.int64), keep], axis=1)).int()


def dump_dinov3(out_dir, save):
    """TRELLIS's `DinoV3FeatureExtractor.extract_features` on a tiny random DINOv3: every layer, the
    final non-affine LayerNorm, all tokens (CLS + registers + patches). A non-square input pins the
    patch-coordinate order."""
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    cfg = DINOv3ViTConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                          patch_size=16, image_size=64, num_register_tokens=4, rope_theta=100.0)
    model = DINOv3ViTModel(cfg).eval()
    g = torch.Generator().manual_seed(SEED)
    for p in model.parameters():
        p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    image = torch.randn(1, 3, 64, 96, generator=g)
    hs = model.embeddings(image, bool_masked_pos=None)
    pe = model.rope_embeddings(image)
    for layer in model.model.layer:
        hs = layer(hs, position_embeddings=pe)
    out = F.layer_norm(hs, hs.shape[-1:])
    # The published checkpoint keeps the encoder layers at the top level (`layer.N.*`).
    sd = {k.replace("model.layer.", "layer.", 1): v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    d = os.path.join(out_dir, "dinov3_tiny")
    os.makedirs(d, exist_ok=True)
    save_file(sd, os.path.join(d, "model.safetensors"))
    json.dump({k: getattr(cfg, k) for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                                            "patch_size", "num_register_tokens", "rope_theta", "use_gated_mlp",
                                            "layer_norm_eps")}, open(os.path.join(d, "config.json"), "w"))
    save("dinov3", {"image": image, "out": out})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="clone of microsoft/TRELLIS.2")
    ap.add_argument("--weights", required=True, help="TRELLIS.2-4B snapshot (has ckpts/)")
    ap.add_argument("--ss-dec", required=True, help="TRELLIS-image-large snapshot (has ckpts/ss_dec_*)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list of stages")
    args = ap.parse_args()

    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
    os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"
    install_shims()
    # The package __init__ pulls in pipelines and renderers; the models need none of it.
    pkg = types.ModuleType("trellis2")
    pkg.__path__ = [os.path.join(args.ref, "trellis2")]
    sys.modules["trellis2"] = pkg
    sp = importlib.import_module("trellis2.modules.sparse")
    unet = importlib.import_module("trellis2.models.sc_vaes.sparse_unet_vae")

    os.makedirs(args.out, exist_ok=True)
    only = set(filter(None, args.only.split(",")))
    ck = os.path.join(args.weights, "ckpts")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.set_grad_enabled(False)

    def want(name):
        return not only or name in only

    def save(name, tensors):
        save_file({k: (v.int().clone() if v.dtype in (torch.int32, torch.int64) else f32(v).clone())
                   for k, v in tensors.items()}, os.path.join(args.out, name + ".safetensors"))
        print(f"wrote {name}: " + ", ".join(f"{k}{list(v.shape)}" for k, v in tensors.items()), flush=True)

    cond = torch.randn(1, COND_TOKENS, COND_DIM)
    t = torch.tensor([T_MODEL])

    if want("dinov3"):
        dump_dinov3(args.out, save)

    if want("ss_dec"):
        m = build(args.ref, os.path.join(args.ss_dec, "ckpts", "ss_dec_conv3d_16l8_fp16"))
        z = torch.randn(1, 8, 16, 16, 16)
        save("ss_dec", {"z": z, "logits": m(z)})

    if want("ss_flow"):
        m = build(args.ref, os.path.join(ck, "ss_flow_img_dit_1_3B_64_bf16"))
        taps = {}
        m.input_layer.register_forward_hook(tap(taps, "h_in"))
        m.blocks[0].register_forward_hook(tap(taps, "blk0"))
        m.blocks[-1].register_forward_hook(tap(taps, "blk_last"))
        x = torch.randn(1, 8, 16, 16, 16)
        out = m(x, t, cond)
        save("ss_flow", {"x": x, "t": t, "cond": cond, "out": out, **taps})

    def slat_inputs(grid, count, channels):
        coords = blob_coords(rng, grid, count)
        feats = torch.randn(count, channels)
        return coords, feats

    if want("slat_shape512"):
        m = build(args.ref, os.path.join(ck, "slat_flow_img2shape_dit_1_3B_512_bf16"))
        coords, feats = slat_inputs(32, 700, 32)
        x = sp.SparseTensor(feats=feats, coords=coords)
        taps = {}
        m.input_layer.register_forward_hook(tap(taps, "h_in"))
        m.blocks[0].register_forward_hook(tap(taps, "blk0"))
        m.blocks[-1].register_forward_hook(tap(taps, "blk_last"))
        out = m(x, t, cond)
        save("slat_shape512", {"coords": coords, "x": feats, "t": t, "cond": cond, "out": out.feats, **taps})

    if want("slat_tex512"):
        m = build(args.ref, os.path.join(ck, "slat_flow_imgshape2tex_dit_1_3B_512_bf16"))
        coords, feats = slat_inputs(32, 700, 32)
        concat = torch.randn(700, 32)
        x = sp.SparseTensor(feats=feats, coords=coords)
        cc = sp.SparseTensor(feats=concat, coords=coords)
        out = m(x, t, cond, concat_cond=cc)
        save("slat_tex512", {"coords": coords, "x": feats, "concat": concat, "t": t, "cond": cond, "out": out.feats})

    if want("shape_dec") or want("tex_dec"):
        pj = json.load(open(os.path.join(args.weights, "pipeline.json")))["args"]
        def norm(key):
            n = pj[key]
            return torch.tensor(n["mean"]), torch.tensor(n["std"])
        smean, sstd = norm("shape_slat_normalization")
        # A latent the size of a real 512 asset's core: ~120 tokens on the /16 grid.
        coords, z = slat_inputs(32, 120, 32)
        slat_feats = z * sstd + smean
        slat = sp.SparseTensor(feats=slat_feats, coords=coords)

        dec = build(args.ref, os.path.join(ck, "shape_dec_next_dc_f16c32_fp16"))
        dec.set_resolution(512)
        h, subs = unet.SparseUnetVaeDecoder.forward(dec, slat, return_subs=True)
        raw = h.feats
        margin = dec.voxel_margin
        vertices = (1 + 2 * margin) * torch.sigmoid(raw[:, 0:3]) - margin
        intersected = raw[:, 3:6] > 0
        quad_lerp = F.softplus(raw[:, 6:7])
        mv, mt = flexible_dual_grid_to_mesh(h.coords[:, 1:], vertices, intersected, quad_lerp, [[-.5] * 3, [.5] * 3], grid_size=512)
        sub_tensors = {f"sub{i}": s.feats.float() for i, s in enumerate(subs)}
        sub_coords = {f"sub{i}_coords": s.coords for i, s in enumerate(subs)}
        print(f"shape_dec: {coords.shape[0]} tokens -> {raw.shape[0]} voxels, {mt.shape[0]} triangles", flush=True)
        if want("shape_dec"):
            save("shape_dec", {"coords": coords, "slat": slat_feats, "out_coords": h.coords, "raw": raw,
                               "verts": mv, "tris": mt.int(), **sub_tensors, **sub_coords})

        if want("tex_dec"):
            tmean, tstd = norm("tex_slat_normalization")
            tex_z = torch.randn(coords.shape[0], 32)
            tex = sp.SparseTensor(feats=tex_z * tstd + tmean, coords=coords)
            tdec = build(args.ref, os.path.join(ck, "tex_dec_next_dc_f16c32_fp16"))
            tout = tdec(tex, guide_subs=subs)
            save("tex_dec", {"coords": coords, "slat": tex.feats, "out_coords": tout.coords, "raw": tout.feats,
                             "final": tout.feats * 0.5 + 0.5})


if __name__ == "__main__":
    main()
