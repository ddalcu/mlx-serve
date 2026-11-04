#!/usr/bin/env python3
"""Oracle for src/plugins/pixal3d/pixal3d.zig and src/plugins/pixal3d/naf.zig: TencentARC's Pixal3D reference modules (its TRELLIS.2
backbone with pixel-aligned projection conditioning) and valeoai's NAF upsampler, run in fp32 on the CPU.

The reference needs CUDA for natten (NAF's neighbourhood attention), FlexGEMM, CuMesh and o_voxel; none
of the modules dumped here touch the last three. natten is replaced by `na2d` below: a gather over the
window NATTEN's own flex mask defines (centre clamped into the dilation group), checked here against
natten's pure-python flex backend on small shapes before anything is dumped. Everything else is the
reference code, unchanged.

USER-RUN. Needs torch + torchvision + transformers + safetensors + numpy + pillow + einops and:

  git clone https://github.com/TencentARC/Pixal3D                        <ref>
  git clone https://github.com/valeoai/NAF                               <naf>
  pip download natten==0.21.0 --no-deps (unpack; only its python flex backend is imported)  <natten-src>
  hf download TencentARC/Pixal3D --local-dir <weights>
  hf download camenduru/dinov3-vitl16-pretrain-lvd1689m --local-dir <dinov3>
  curl -L https://github.com/valeoai/NAF/releases/download/model/naf_release.pth -o <naf.pth>

  python -I tests/dump_pixal3d_fixtures.py --ref <ref> --naf <naf> --natten <natten-src>/src \\
      --weights <weights> --dinov3 <dinov3> --naf-weights <naf.pth> --image <rgba cutout png> \\
      --out <fixtures dir>
  PIXAL3D_FIXTURES=<out> PIXAL3D_WEIGHTS=<weights> PIXAL3D_DINOV3=<dinov3> PIXAL3D_NAF=<out>/naf \\
      PIXAL3D_IMAGE=<rgba cutout png> zig build test -Doptimize=ReleaseFast -Dtest-filter="pixal3d"

Writes under --out (f32 / i32, seed 0): naf/model.safetensors (the NAF state dict as safetensors, the
pack's layout), preprocess, naf_a, naf_b, proj_ss, proj_shape512, proj_shape1024, proj_tex1024, proj_mv,
ss_flow, slat_shape512, slat_tex1024.
"""

import argparse
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file

SEED = 0
T_MODEL = 437.5
FOV = 0.2  # the reference demo's manual FOV default (radians)
# The four stage configs of inference.py's IMAGE_COND_CONFIGS.
STAGES = {
    "ss": dict(image_size=512, grid_resolution=16),
    "shape512": dict(image_size=512, grid_resolution=32, use_naf_upsample=True, naf_target_size=512),
    "shape1024": dict(image_size=1024, grid_resolution=64, use_naf_upsample=True, naf_target_size=512),
    "tex1024": dict(image_size=1024, grid_resolution=64, use_naf_upsample=True, naf_target_size=1024),
}


def _module(name, path=None, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    if path is not None:
        m.__path__ = [path]
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── natten: the dilated neighbourhood window, as natten's flex mask defines it ────────────────────


def _window(length, dilation, kernel):
    """[kernel, length] key positions per query position along one axis."""
    idx = torch.arange(length)
    group = idx % dilation
    local = idx // dilation
    pad = 1 - ((group + (dilation - (length % dilation))) // dilation)
    group_len = length // dilation + pad
    left = kernel // 2
    right = kernel // 2 + (kernel % 2 - 1)
    centre = torch.minimum(torch.maximum(local, torch.full_like(local, left)), group_len - 1 - right)
    return torch.stack([(centre - left + a) * dilation + group for a in range(kernel)])


def na2d(q, k, v, kernel_size, dilation=1, stride=1, scale=None, backend=None, **_):
    """[B, X, Y, H, D] q/k and [B, X, Y, H, Dv] v; row chunks bound the gathers."""
    assert stride in (1, (1, 1))
    kx, ky = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    dx, dy = (dilation, dilation) if isinstance(dilation, int) else dilation
    b, nx, ny, h, d = q.shape
    scale = d ** -0.5 if scale is None else scale
    wx, wy = _window(nx, dx, kx), _window(ny, dy, ky)
    out = torch.empty(b, nx, ny, h, v.shape[-1], dtype=q.dtype)
    rows = 32
    for r0 in range(0, nx, rows):
        r1 = min(nx, r0 + rows)
        qc = q[:, r0:r1]
        scores = []
        for a in range(kx):
            krow = k[:, wx[a, r0:r1]]
            for c in range(ky):
                scores.append((qc * krow[:, :, wy[c]]).sum(-1) * scale)
        p = torch.softmax(torch.stack(scores, -1), -1)
        acc = torch.zeros(b, r1 - r0, ny, h, v.shape[-1], dtype=q.dtype)
        for a in range(kx):
            vrow = v[:, wx[a, r0:r1]]
            for c in range(ky):
                acc += p[..., a * ky + c, None] * vrow[:, :, wy[c]]
        out[:, r0:r1] = acc
    return out


def check_na2d_against_natten(natten_src):
    """The shim against natten's own flex backend, value dims split to the q/k width it requires."""
    sys.path.insert(0, natten_src)
    real = importlib.import_module("natten.functional")
    sys.path.remove(natten_src)
    for name in [m for m in sys.modules if m == "natten" or m.startswith("natten.")]:
        del sys.modules[name]
    g = torch.Generator().manual_seed(SEED)
    for (n, dil) in [((24, 24), 2), ((36, 18), (4, 2)), ((16, 16), 1)]:
        q = torch.randn(1, *n, 2, 8, generator=g)
        k = torch.randn(1, *n, 2, 8, generator=g)
        v = torch.randn(1, *n, 2, 16, generator=g)
        want = torch.cat([real.na2d(q, k, v[..., i:i + 8], kernel_size=9, dilation=dil, backend="flex-fna") for i in (0, 8)], -1)
        got = na2d(q, k, v, kernel_size=9, dilation=dil)
        err = (got - want).abs().max().item()
        assert err < 1e-5, f"na2d shim differs from natten at {n} dilation {dil}: {err}"
    print("na2d shim == natten flex backend", flush=True)


def install_shims(ref, naf_repo):
    _module("natten", na2d=na2d)
    # NAF imports `src.layers`; `src.model`'s __init__ pulls in every other upsampler, so only naf.py loads.
    _module("src", os.path.join(naf_repo, "src"))
    importlib.import_module("src.layers")
    _module("src.model", os.path.join(naf_repo, "src", "model"))
    naf_mod = _load("src.model.naf", os.path.join(naf_repo, "src", "model", "naf.py"))
    # Pixal3D's package __init__ pulls in pipelines and renderers; its models and the extractor need none of it.
    _module("pixal3d", os.path.join(ref, "pixal3d"))
    for sub in ("trainers", "trainers/flow_matching", "trainers/flow_matching/mixins"):
        _module("pixal3d." + sub.replace("/", "."), os.path.join(ref, "pixal3d", sub))
    # The extractor reads `model.layer` (transformers 4); transformers >= 5 nests the encoder under `model.`.
    from transformers import DINOv3ViTModel
    if not hasattr(DINOv3ViTModel, "layer"):
        DINOv3ViTModel.layer = property(lambda self: self.model.layer)
    proj = importlib.import_module("pixal3d.trainers.flow_matching.mixins.image_conditioned_proj")
    return naf_mod, proj


def f32(x):
    return x.detach().float().contiguous()


def tap(store, name):
    def hook(_m, _i, out):
        store[name] = f32(out.feats if hasattr(out, "feats") else out)
    return hook


def build(ckpt_stem):
    """The reference module for `<ckpt_stem>.json/.safetensors`, fp32 on the CPU."""
    cfg = json.load(open(ckpt_stem + ".json"))
    args = dict(cfg["args"])
    if "dtype" in args:
        args["dtype"] = "float32"
    if "use_fp16" in args:
        args["use_fp16"] = False
    models = importlib.import_module("pixal3d.models")
    m = getattr(models, cfg["name"])(**args)
    sd = load_file(ckpt_stem + ".safetensors")
    # The SS flow ships its RoPE table (complex64): `.float()` would drop the sine half.
    missing, unexpected = m.load_state_dict({k: v if v.is_complex() else v.float() for k, v in sd.items()}, strict=False)
    missing = [k for k in missing if "rope_phases" not in k and "pos_emb" not in k]
    unexpected = [k for k in unexpected if "rope_phases" not in k]
    assert not missing, f"missing weights: {missing[:5]}"
    assert not unexpected, f"unexpected weights: {unexpected[:5]}"
    return m.float().eval()


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    """inference.py's own, verbatim (that file imports o_voxel and the whole pipeline at the top)."""
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = float((focal_length * image_resolution / 32.0).item())
    x_ndc = xt - image_resolution / 2.0
    distance_x = f_pixels * xw / x_ndc - yw
    return float(distance_x)


def preprocess_image(input):
    """Pixal3DImageTo3DPipeline.preprocess_image, the alpha branch (the cutout carries its alpha)."""
    alpha = np.array(input)[:, :, 3]
    assert not np.all(alpha == 255)
    max_size = max(input.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
    output = input
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    rgb = output[:, :, :3]
    a = output[:, :, 3:4]
    bg = np.array((0, 0, 0), dtype=np.float32) / 255.0
    output = rgb * a + bg * (1.0 - a)
    return Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8))


def to_cond_tensor(image, image_size):
    """inference_mv.py's, verbatim."""
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
    rgb = torch.tensor(np.array(image.convert('RGB'))).permute(2, 0, 1).float() / 255.0
    return rgb * alpha.unsqueeze(0)


def blob_coords(rng, grid, count):
    """`count` distinct coords of a `grid`^3 volume inside a ball at its centre, x-major sorted."""
    c = (grid - 1) / 2
    pts = np.stack(np.meshgrid(*[np.arange(grid)] * 3, indexing="ij"), -1).reshape(-1, 3)
    d = np.linalg.norm(pts - c, axis=1) + rng.uniform(0, 0.5, len(pts))
    keep = pts[np.argsort(d)[:count]]
    keep = keep[np.lexsort((keep[:, 2], keep[:, 1], keep[:, 0]))]
    return torch.from_numpy(np.concatenate([np.zeros((count, 1), dtype=np.int64), keep], axis=1)).int()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="clone of TencentARC/Pixal3D")
    ap.add_argument("--naf", required=True, help="clone of valeoai/NAF")
    ap.add_argument("--natten", required=True, help="natten 0.21.0 sdist src/ (pure-python import only)")
    ap.add_argument("--weights", required=True, help="TencentARC/Pixal3D snapshot (has ckpts/)")
    ap.add_argument("--dinov3", required=True, help="DINOv3 ViT-L/16 (config.json + model.safetensors)")
    ap.add_argument("--naf-weights", required=True, help="naf_release.pth")
    ap.add_argument("--image", required=True, help="RGBA cutout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list of stages")
    args = ap.parse_args()

    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_CONV_BACKEND"] = "none"
    check_na2d_against_natten(args.natten)
    naf_mod, proj = install_shims(args.ref, args.naf)
    sp = importlib.import_module("pixal3d.modules.sparse")

    os.makedirs(args.out, exist_ok=True)
    only = set(filter(None, args.only.split(",")))
    ck = os.path.join(args.weights, "ckpts")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.set_grad_enabled(False)

    def want(name):
        return not only or name in only

    def save(name, tensors):
        save_file({k: (v.int().clone() if v.dtype in (torch.int32, torch.int64, torch.uint8) else f32(v).clone())
                   for k, v in tensors.items()}, os.path.join(args.out, name + ".safetensors"))
        print(f"wrote {name}: " + ", ".join(f"{k}{list(v.shape)}" for k, v in tensors.items()), flush=True)

    naf_sd = torch.load(args.naf_weights, map_location="cpu", weights_only=True)
    os.makedirs(os.path.join(args.out, "naf"), exist_ok=True)
    save_file({k: v.contiguous() for k, v in naf_sd.items()}, os.path.join(args.out, "naf", "model.safetensors"))

    def make_naf():
        m = naf_mod.NAF()
        m.load_state_dict(naf_sd)
        return m.eval()

    naf_model = make_naf()

    def patched_load_naf(self):
        if self.naf_model is None:
            self.naf_model = naf_model

    proj.DinoV3ProjFeatureExtractor._load_naf = patched_load_naf
    # The extractor moves PIL inputs to CUDA; the CPU run keeps them where they are.
    torch.Tensor.cuda = lambda self, *a, **k: self

    cutout = Image.open(args.image).convert("RGBA")
    pre = preprocess_image(cutout)

    if want("preprocess"):
        save("preprocess", {"rgb": torch.from_numpy(np.array(pre)).int(),
                            "mv512": to_cond_tensor(cutout, 512)})

    if want("naf_a") or want("naf_b"):
        # NAF alone at the two image/output ratios Pixal3D uses, on random low-res features; the hr map at a
        # spread of pixels (the corners and edges included: they pin the window clamp).
        for name, img_size, lr, out_size in (("naf_a", 512, 32, 512), ("naf_b", 1024, 64, 512)):
            if not want(name):
                continue
            img = torch.from_numpy(np.array(pre.resize((img_size, img_size), Image.Resampling.LANCZOS))).permute(2, 0, 1)[None].float() / 255
            feats = torch.randn(1, 1024, lr, lr)
            hr = naf_model(img, feats, (out_size, out_size))
            pix = rng.integers(0, out_size, size=(1500, 2))
            edge = np.array([[0, 0], [0, out_size - 1], [out_size - 1, 0], [out_size - 1, out_size - 1], [3, 200], [out_size - 2, 77]])
            pix = np.concatenate([edge, pix])
            sel = hr[0, :, pix[:, 0], pix[:, 1]].T
            save(name, {"image": img, "feats": feats, "pix": torch.from_numpy(pix).int(), "out": sel})

    distance = distance_from_fov(FOV, torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0, 511]), 1.0, 512)
    print(f"camera: fov {FOV} distance {distance:.6f}", flush=True)
    for stage, cfg in STAGES.items():
        name = "proj_" + stage
        if not want(name):
            continue
        ext = proj.DinoV3ProjFeatureExtractor(model_name=args.dinov3, **cfg).eval()
        g, z = ext([pre], camera_angle_x=torch.tensor([FOV]), distance=torch.tensor([distance]), mesh_scale=torch.tensor([1.0]))
        r = cfg["grid_resolution"]
        if stage == "ss":
            save(name, {"global": g, "proj": z[0], "distance": torch.tensor([distance])})
        else:
            coords = blob_coords(rng, r, 3000)
            grid = z.reshape(1, r, r, r, -1)
            sel = grid[0, coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()]
            save(name, {"global": g, "coords": coords, "proj": sel, "distance": torch.tensor([distance])})
        del ext

    if want("proj_mv"):
        # Two views at the shipped rig's first two poses: the cutout and its mirror image.
        rig = json.load(open(os.path.join(args.ref, "assets", "mv_images", "example", "transforms.json")))
        frames = rig["frames"][:2]
        views = [cutout, cutout.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]
        tm = torch.tensor([fr["transform_matrix"] for fr in frames], dtype=torch.float32)[None]
        fov = torch.full((1, 2), float(rig["camera_angle_x"]))
        dist = torch.norm(tm[:, :, :3, 3], dim=-1)
        out = {"transform_matrix": tm[0], "fov": fov[0], "mv1024": to_cond_tensor(views[1], 1024)}
        for stage in ("ss", "shape512", "shape1024"):
            cfg = STAGES[stage]
            ext = proj.DinoV3ProjMultiViewFeatureExtractor(model_name=args.dinov3, multiview_fusion="average", **cfg).eval()
            imgs = torch.stack([to_cond_tensor(v, cfg["image_size"]) for v in views])[None]
            g, z = ext(imgs, camera_angle_x=fov, distance=dist, mesh_scale=torch.tensor([1.0]), transform_matrix=tm)
            r = cfg["grid_resolution"]
            out[stage + "_global"] = g
            if stage == "ss":
                out[stage + "_proj"] = z[0]
            else:
                coords = blob_coords(rng, r, 2000)
                out[stage + "_coords"] = coords
                out[stage + "_proj"] = z.reshape(1, r, r, r, -1)[0, coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()]
            del ext
        save("proj_mv", out)

    t = torch.tensor([T_MODEL])
    glob = torch.randn(1, 5, 1024)

    if want("ss_flow"):
        m = build(os.path.join(ck, "ss_flow_img_dit_1_3B_64_bf16"))
        taps = {}
        m.input_layer.register_forward_hook(tap(taps, "h_in"))
        m.blocks[0].register_forward_hook(tap(taps, "blk0"))
        m.blocks[-1].register_forward_hook(tap(taps, "blk_last"))
        x = torch.randn(1, 8, 16, 16, 16)
        pr = torch.randn(1, 16 ** 3, 1024)
        out = m(x, t, {"global": glob, "proj": pr})
        save("ss_flow", {"x": x, "t": t, "global": glob, "proj": pr[0], "out": out, **taps})
        del m

    def slat(name, stem, concat):
        m = build(os.path.join(ck, stem))
        coords = blob_coords(rng, 32, 700)
        feats = torch.randn(700, 32)
        pr = torch.randn(700, 2048)
        x = sp.SparseTensor(feats=feats, coords=coords)
        cond = {"global": glob, "proj": sp.SparseTensor(feats=pr, coords=coords)}
        taps = {}
        m.input_layer.register_forward_hook(tap(taps, "h_in"))
        m.blocks[0].register_forward_hook(tap(taps, "blk0"))
        m.blocks[-1].register_forward_hook(tap(taps, "blk_last"))
        extra = {}
        if concat:
            cc = torch.randn(700, 32)
            out = m(x, t, cond, concat_cond=sp.SparseTensor(feats=cc, coords=coords))
            extra["concat"] = cc
        else:
            out = m(x, t, cond)
        save(name, {"coords": coords, "x": feats, "t": t, "global": glob, "proj": pr, "out": out.feats, **extra, **taps})

    if want("slat_shape512"):
        slat("slat_shape512", "slat_flow_img2shape_dit_1_3B_512_bf16", False)
    if want("slat_tex1024"):
        slat("slat_tex1024", "slat_flow_imgshape2tex_dit_1_3B_1024_bf16", True)


if __name__ == "__main__":
    main()
