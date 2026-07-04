#!/usr/bin/env python3
"""Dump Hunyuan3D-2.1 SHAPE parity fixtures (.raw f32) for the Zig oracle tests
in `src/hunyuan3d.zig`.

USER-RUN (needs torch + the ~8 GB HF checkpoints + a clone of the reference repo).
Runs the REFERENCE PyTorch pipeline and writes byte-exact golden tensors that
`zig build test` compares the Zig port against (cosine parity), mirroring
`tests/dump_krea_fixtures.py`. Five oracle taps:

  1 DINO features   : post-preprocess input [1,3,518,518] + last_hidden_state [1,1370,1024]
  2 DiT velocity    : one-step v at sigma 0.5, CFG batch [2,4096,64] + context [2,1370,1024]
  3 latent -> SDF   : final latents [1,4096,64] (pre-scale-factor) + 4096 fixed points -> logits [1,4096,1]
  4 full denoise    : 10-step denoise from an INJECTED noise -> final latents [1,4096,64]
  5 e2e SDF grid    : dense grid at octree_resolution 128 (129^3) + mesh vertex count + bbox

Usage:
    python3 tests/dump_hunyuan3d_fixtures.py --repo <Hunyuan3D-2.1/hy3dshape> --model <HF snapshot dir> \
        [--out DIR] [--image IMG] [--device mps|cpu] [--e2e-steps N] [--test-model CONVERTED_DIR]

  --repo        clone dir that contains the `hy3dshape` package (repo/hy3dshape or repo root)
  --model       HF snapshot with hunyuan3d-dit-v2-1/ and hunyuan3d-vae-v2-1/ (tencent/Hunyuan3D-2.1)
  --out         fixture output dir (default tests/fixtures/hy3d)
  --test-model  the CONVERTED mlx-serve model dir the Zig test loads (echoed as HY3D_TEST_MODEL)

Everything runs fp16 on the reference device (mps if available), dumped as f32
little-endian. Fixed seed 0 everywhere.

It prints the exact `export HY3D_*` block to paste before running the Zig oracles:
    HY3D_TEST_MODEL=<converted dir> HY3D_DINO_IN=... [all vars] \
        zig build test -Doptimize=ReleaseFast -Dtest-filter="hunyuan3d"
"""

import argparse
import os
import sys

import numpy as np

SEED = 0
VEL_SIGMA = 0.5
DENOISE_STEPS = 10
GUIDANCE_SCALE = 5.0
BOX_V = 1.01
NUM_LATENTS = 4096
EMBED_DIM = 64
CONTEXT_DIM = 1024
DINO_TOKENS = 1370


def pick_device(requested):
    import torch
    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synth_image():
    """Deterministic RGBA test image: a filled disk with a color gradient on a
    transparent background (so `recenter`'s alpha bbox is well-defined)."""
    from PIL import Image
    n = 256
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    cx = cy = (n - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = r < (n * 0.36)
    rgba = np.zeros((n, n, 4), dtype=np.uint8)
    rgba[..., 0] = (xx / n * 255).astype(np.uint8)
    rgba[..., 1] = (yy / n * 255).astype(np.uint8)
    rgba[..., 2] = ((1.0 - r / r.max()) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(inside, 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def main():
    ap = argparse.ArgumentParser(description="Dump Hunyuan3D-2.1 shape parity fixtures.")
    ap.add_argument("--repo", required=True, help="clone dir containing the hy3dshape package")
    ap.add_argument("--model", required=True, help="HF snapshot dir (tencent/Hunyuan3D-2.1)")
    ap.add_argument("--out", default=os.path.join("tests", "fixtures", "hy3d"))
    ap.add_argument("--image", default=None, help="optional source image (else a deterministic synthetic one)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--e2e-res", type=int, default=128, help="octree_resolution for the e2e grid oracle (#5)")
    # 8000 = reference default. Bigger chunks explode on MPS: the geo cross-attn
    # materializes a [16, chunk, 4096] f32 score matrix (50k chunk → a single
    # 13.1 GB Metal buffer → hard assertion).
    ap.add_argument("--num-chunks", type=int, default=8000, help="volume-decode query chunk size")
    ap.add_argument("--test-model",
                    default=os.path.expanduser("~/.mlx-serve/models/local/hunyuan3d-2-1-8bit"),
                    help="converted mlx-serve model dir (echoed as HY3D_TEST_MODEL)")
    args = ap.parse_args()

    # make the reference package importable (accept repo root or the hy3dshape
    # parent). The clone nests hy3dshape/hy3dshape/, so require the package's
    # __init__.py — a bare dir match picks the outer namespace dir and
    # `hy3dshape.pipelines` fails to import.
    for cand in (args.repo, os.path.join(args.repo, "hy3dshape")):
        if os.path.isfile(os.path.join(cand, "hy3dshape", "__init__.py")):
            sys.path.insert(0, cand)
            break
    else:
        sys.path.insert(0, args.repo)

    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    # ── memory-lean load (16 GB Macs) ─────────────────────────────────────
    # The reference loader torch.loads the full 7.4 GB ckpt AND instantiates
    # fp32 modules before casting — a ~22 GB transient that swap-thrashes a
    # 16 GB machine. Two patches keep the peak under ~8 GB: mmap the ckpt
    # (file-backed tensors, near-zero resident) and build modules fp16 so
    # load_state_dict never materializes an fp32 copy.
    _orig_torch_load = torch.load

    def _mmap_load(*a, **k):
        k.setdefault("map_location", "cpu")
        try:
            return _orig_torch_load(*a, **{**k, "mmap": True})
        except (RuntimeError, TypeError):  # legacy (non-zip) ckpt format
            return _orig_torch_load(*a, **k)

    torch.load = _mmap_load
    torch.set_default_dtype(torch.float16)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device(args.device)
    dtype = torch.float16
    print(f"[dump] repo={args.repo}\n[dump] model={args.model}\n[dump] out={out_dir}\n[dump] device={device}")

    def save_f32(name, arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().float().cpu().numpy()
        np.asarray(arr, dtype="<f4").ravel().tofile(os.path.join(out_dir, name))
        return np.asarray(arr).shape

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model, device=device, dtype=dtype, use_safetensors=False)
    torch.set_default_dtype(torch.float32)  # only the module BUILD is fp16
    torch.load = _orig_torch_load
    model, vae, conditioner, scheduler = pipe.model, pipe.vae, pipe.conditioner, pipe.scheduler
    model.eval()
    vae.eval()
    conditioner.eval()
    num_train = scheduler.config.num_train_timesteps

    # ── image -> pipeline conditioner input [1,3,512,512] in [-1,1] ──────────
    from PIL import Image
    img = Image.open(args.image).convert("RGBA") if args.image else synth_image()
    cond_inputs = pipe.prepare_image(img, None)
    cond_image = cond_inputs.pop("image").to(device=device, dtype=dtype)

    # ── oracle 1: DINO. Hook the Dinov2Model to capture post-preprocess input + output ──
    dino_model = conditioner.main_image_encoder.model
    cap = {}

    def pre_hook(_m, inp):
        cap["in"] = inp[0].detach()

    def post_hook(_m, _inp, out):
        cap["out"] = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]).detach()

    h1 = dino_model.register_forward_pre_hook(pre_hook)
    h2 = dino_model.register_forward_hook(post_hook)
    with torch.inference_mode():
        cond = pipe.encode_cond(
            image=cond_image,
            additional_cond_inputs={k: v for k, v in cond_inputs.items()},
            do_classifier_free_guidance=True,
            dual_guidance=False,
        )
    h1.remove()
    h2.remove()
    ctx2 = cond["main"].to(device=device, dtype=dtype)  # [2,1370,1024] = cat([real, zeros])
    assert cap["in"].shape[-2:] == (518, 518), f"unexpected DINO input {tuple(cap['in'].shape)}"
    assert tuple(cap["out"].shape) == (1, DINO_TOKENS, CONTEXT_DIM), tuple(cap["out"].shape)
    assert tuple(ctx2.shape) == (2, DINO_TOKENS, CONTEXT_DIM), tuple(ctx2.shape)
    print("[o1] DINO in", save_f32("hy3d_dino_in.raw", cap["in"]),
          "out", save_f32("hy3d_dino_out.raw", cap["out"]))
    # cond is computed once — retire the DINO encoder from the GPU (16 GB Macs).
    conditioner.to("cpu")
    if device == "mps":
        torch.mps.empty_cache()

    # ── oracle 2: DiT one-step velocity at sigma 0.5 (CFG batch of 2) ────────
    torch.manual_seed(SEED + 1)
    vel_lat = torch.randn(2, NUM_LATENTS, EMBED_DIM, device=device, dtype=dtype)
    vel_sigma = torch.full((2,), VEL_SIGMA, device=device, dtype=dtype)
    with torch.inference_mode():
        vel = model(vel_lat, vel_sigma, cond)  # forward(x, t, contexts) -> [2,4096,64]
    assert tuple(vel.shape) == (2, NUM_LATENTS, EMBED_DIM), tuple(vel.shape)
    save_f32("hy3d_vel_lat.raw", vel_lat)
    save_f32("hy3d_vel_ctx.raw", ctx2)
    save_f32("hy3d_vel.raw", vel)
    print(f"[o2] DiT velocity [2,4096,64] at sigma={VEL_SIGMA}")

    # ── oracle 4: 10-step denoise from an INJECTED noise (drive the scheduler) ──
    torch.manual_seed(SEED + 2)
    init_noise = torch.randn(1, NUM_LATENTS, EMBED_DIM, device=device, dtype=dtype)
    sigmas = np.linspace(0, 1, DENOISE_STEPS)
    scheduler.set_timesteps(sigmas=sigmas, device=device)  # mirrors retrieve_timesteps(sigmas=...)
    latents = init_noise * getattr(scheduler, "init_noise_sigma", 1.0)
    with torch.inference_mode():
        for t in scheduler.timesteps:
            model_in = torch.cat([latents] * 2)
            ts = t.expand(model_in.shape[0]).to(latents.dtype) / num_train
            noise_pred = model(model_in, ts, cond)
            nc, nu = noise_pred.chunk(2)
            noise_pred = nu + GUIDANCE_SCALE * (nc - nu)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
    denoise_out = latents  # [1,4096,64] pre-scale-factor
    assert tuple(denoise_out.shape) == (1, NUM_LATENTS, EMBED_DIM), tuple(denoise_out.shape)
    save_f32("hy3d_denoise_init.raw", init_noise)
    # cond half ONLY ([1,1370,1024]) — the Zig denoise builds its own zeros
    # uncond; ctx2's second (zeros) half would silently shift the layout.
    save_f32("hy3d_denoise_ctx.raw", ctx2[:1])
    save_f32("hy3d_denoise_out.raw", denoise_out)
    print(f"[o4] denoise {DENOISE_STEPS} steps -> [1,4096,64]")
    # The DiT (6.6 GB) is done — oracles 3 + 5 only need the VAE; retire it
    # before the chunked geo-decoder storm (16 GB Macs).
    model.to("cpu")
    if device == "mps":
        torch.mps.empty_cache()

    # ── oracle 3: latent -> SDF on 4096 fixed points (reuse the denoised latent) ──
    sdf_lat = denoise_out  # pre-scale-factor
    torch.manual_seed(SEED + 3)
    sdf_pts = (torch.rand(1, NUM_LATENTS, 3, device=device, dtype=dtype) * 2.0 - 1.0)
    with torch.inference_mode():
        lat_set = vae(sdf_lat / vae.scale_factor)                 # post_kl + transformer -> [1,4096,1024]
        sdf_logits = vae.geo_decoder(queries=sdf_pts, latents=lat_set)   # [1,4096,1]
    assert tuple(sdf_logits.shape) == (1, NUM_LATENTS, 1), tuple(sdf_logits.shape)
    save_f32("hy3d_sdf_lat.raw", sdf_lat)
    save_f32("hy3d_sdf_pts.raw", sdf_pts)
    save_f32("hy3d_sdf.raw", sdf_logits)
    print("[o3] latent->SDF [1,4096,1] on 4096 fixed points")

    # ── oracle 5: e2e SDF grid at octree_resolution 128 (129^3) + mesh sanity ──
    # Reuses oracle 4's trajectory verbatim: the Zig e2e oracle reads
    # HY3D_DENOISE_INIT/CTX/STEPS and re-denoises itself, so the grid here must
    # come from the SAME injected noise / context / step count — a fresh
    # trajectory would fail the oracle spuriously.
    with torch.inference_mode():
        e2e_set = vae(denoise_out / vae.scale_factor)
        # mirror ShapeVAE.latents2mesh: volume_decoder -> surface_extractor
        grid = vae.volume_decoder(
            e2e_set, vae.geo_decoder,
            bounds=BOX_V, num_chunks=args.num_chunks,
            octree_resolution=args.e2e_res, enable_pbar=True,
        )  # [1, R+1, R+1, R+1] float, x-major (ij meshgrid)
    grid_np = grid[0].detach().float().cpu().numpy()
    res1 = args.e2e_res + 1
    assert grid_np.shape == (res1, res1, res1), grid_np.shape
    save_f32("hy3d_e2e_grid.raw", grid_np)  # x-major: idx=(ix*res1+iy)*res1+iz

    meshes = vae.surface_extractor(
        grid, mc_level=0.0, bounds=BOX_V, octree_resolution=args.e2e_res)
    mesh = meshes[0]
    if mesh is None:
        print("[o5][WARN] marching_cubes produced no surface (grid may have no zero crossing)")
        nvert = 0
        bbox = np.zeros(6, dtype=np.float32)
    else:
        verts = np.asarray(mesh.mesh_v, dtype=np.float32)
        nvert = int(verts.shape[0])
        bbox = np.concatenate([verts.min(0), verts.max(0)]).astype(np.float32)
    save_f32("hy3d_e2e_bbox.raw", bbox)  # [xmin,ymin,zmin,xmax,ymax,zmax]
    print(f"[o5] e2e grid {res1}^3, nvert={nvert}, bbox={bbox.tolist()}")

    # ── print the env block to run the Zig oracles ──
    o = out_dir
    env = {
        "HY3D_TEST_MODEL": os.path.abspath(os.path.expanduser(args.test_model)),
        "HY3D_DINO_IN": f"{o}/hy3d_dino_in.raw",
        "HY3D_DINO_OUT": f"{o}/hy3d_dino_out.raw",
        "HY3D_VEL_LAT": f"{o}/hy3d_vel_lat.raw",
        "HY3D_VEL_CTX": f"{o}/hy3d_vel_ctx.raw",
        "HY3D_VEL_SIGMA": str(VEL_SIGMA),
        "HY3D_VEL": f"{o}/hy3d_vel.raw",
        "HY3D_SDF_LAT": f"{o}/hy3d_sdf_lat.raw",
        "HY3D_SDF_PTS": f"{o}/hy3d_sdf_pts.raw",
        "HY3D_SDF": f"{o}/hy3d_sdf.raw",
        "HY3D_DENOISE_INIT": f"{o}/hy3d_denoise_init.raw",
        "HY3D_DENOISE_CTX": f"{o}/hy3d_denoise_ctx.raw",
        "HY3D_DENOISE_STEPS": str(DENOISE_STEPS),
        "HY3D_DENOISE_OUT": f"{o}/hy3d_denoise_out.raw",
        "HY3D_E2E_GRID": f"{o}/hy3d_e2e_grid.raw",
        "HY3D_E2E_RES": str(args.e2e_res),
        "HY3D_E2E_NVERT": str(nvert),
        "HY3D_E2E_BBOX": f"{o}/hy3d_e2e_bbox.raw",
    }
    print("\n# ── paste this to run the Zig oracle tests ──")
    print(" \\\n".join(f"{k}={v}" for k, v in env.items()) + " \\")
    print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="hunyuan3d"')


if __name__ == "__main__":
    main()
