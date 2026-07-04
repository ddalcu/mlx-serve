#!/usr/bin/env python3
"""Dump Hunyuan3D-2.1 PAINT parity fixtures (.raw f32) for the Zig oracle tests
in `src/hunyuan3d_paint.zig`.

USER-RUN (needs torch + diffusers + transformers + the paint checkpoints).
Runs the REFERENCE PyTorch components on deterministic inputs and writes
byte-exact golden tensors that `zig build test` compares against at cosine
thresholds, mirroring `tests/dump_hunyuan3d_fixtures.py`. Oracle taps:

  1 VAE encode  : post-preprocess image [1,3,512,512] -> posterior MEAN*0.18215 [1,4,64,64]
  2 VAE decode  : scaled latents [1,4,64,64] -> image [1,3,512,512]
  3 DINO        : post-preprocess [1,3,224,224] -> last_hidden_state [1,257,1536]
                  + ImageProjModel output [1,1028,1024] (proj/norm weights read
                  from the unet ckpt, where they physically live)
  4 REF hiddens : dual-UNet (reference stream) at timestep 0 over fixed
                  ref_latents [1,4,64,64] -> post-norm1 hidden cache for 3
                  canary blocks (first down / mid / last up)      [--with-unet]
  5 UNET step   : ONE full 2.5D UNet forward on fixed latents [12,4,64,64]
                  (2 materials x 6 views) + all conditioning -> v [12,4,64,64]
                                                                 [--with-unet]

Memory discipline (16 GB Mac, house rule): torch.load mmap=True, fp16 default
dtype around module construction, retire each sub-model to CPU + empty_cache
the moment its fixtures are dumped. Deterministic: seed 0, no .sample() calls
(posterior MEAN), injected noise from a seeded generator.

Usage:
    source /Volumes/Sandisk_1TB/hy3d-scratch/venv/bin/activate
    python3 tests/dump_hunyuan3d_paint_fixtures.py \
        --snapshot /Volumes/Sandisk_1TB/hy3d-scratch/snapshot \
        --repo /Volumes/Sandisk_1TB/hy3d-scratch/reference \
        [--out /Volumes/Sandisk_1TB/hy3d-scratch/fixtures-paint] \
        [--test-model /Volumes/Sandisk_1TB/hy3d-scratch/hunyuan3d-2-1-paint-fp16] \
        [--with-unet] [--device mps|cpu]

Prints the `export HY3DP_*` env block for the Zig oracles.
"""

import argparse
import gc
import os
import sys

import numpy as np

SEED = 0
SCALING = 0.18215


def pick_device(requested):
    import torch
    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dump(path, tensor):
    arr = tensor.detach().to("cpu", dtype=None).float().numpy().astype("<f4")
    arr.tofile(path)
    print(f"[dump] {os.path.basename(path)}  shape={tuple(tensor.shape)}  {arr.nbytes/1e6:.1f} MB")


def synth_image_chw(size, channels=3):
    """Deterministic [-1,1] image: radial disk + per-channel gradients (no RNG)."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    disk = (r < 0.33).astype(np.float32)
    ch0 = disk * (0.2 + 0.8 * x) + (1 - disk) * 1.0
    ch1 = disk * (0.2 + 0.8 * y) + (1 - disk) * 1.0
    ch2 = disk * (1.0 - 0.6 * r) + (1 - disk) * 1.0
    img01 = np.stack([ch0, ch1, ch2])[None]  # [1,3,S,S] in [0,1]
    return (img01 - 0.5) * 2.0


def synth_dino_input():
    """Deterministic ImageNet-normalized [1,3,224,224] input (post-preprocess)."""
    img01 = (synth_image_chw(224) + 1.0) / 2.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    return (img01 - mean) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True,
                    help="dir containing hunyuan3d-paintpbr-v2-1/ and dinov2-giant/")
    ap.add_argument("--repo", default=None,
                    help="Hunyuan3D-2.1 reference clone (needed for --with-unet)")
    ap.add_argument("--out", default="tests/fixtures/hy3d_paint")
    ap.add_argument("--test-model", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--with-unet", action="store_true",
                    help="also dump oracle 4 (dual-UNet ref hiddens) + 5 (full 2.5D step)")
    args = ap.parse_args()

    import torch
    torch.manual_seed(SEED)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    paint_dir = os.path.join(args.snapshot, "hunyuan3d-paintpbr-v2-1")
    env = {}

    # ── 1+2: VAE (AutoencoderKL, fp32 ckpt -> run fp16 like the pipeline) ──
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(os.path.join(paint_dir, "vae"), torch_dtype=torch.float16)
    vae = vae.to(device).eval()

    enc_in = torch.from_numpy(synth_image_chw(512).astype(np.float32))
    with torch.no_grad():
        posterior = vae.encode(enc_in.to(device, torch.float16)).latent_dist
        enc_out = posterior.mean.float() * SCALING  # deterministic MEAN, not .sample()
    dump(os.path.join(args.out, "hy3dp_vae_enc_in.raw"), enc_in)
    dump(os.path.join(args.out, "hy3dp_vae_enc_out.raw"), enc_out)
    env["HY3DP_VAE_ENC_IN"] = "hy3dp_vae_enc_in.raw"
    env["HY3DP_VAE_ENC_OUT"] = "hy3dp_vae_enc_out.raw"

    # Decode a DIFFERENT deterministic latent (not the encode output — keeps the
    # two oracles independent).
    g = torch.Generator(device="cpu").manual_seed(SEED)
    dec_in = torch.randn(1, 4, 64, 64, generator=g, dtype=torch.float32) * SCALING
    with torch.no_grad():
        dec_out = vae.decode((dec_in / SCALING).to(device, torch.float16)).sample.float()
    dump(os.path.join(args.out, "hy3dp_vae_dec_in.raw"), dec_in)
    dump(os.path.join(args.out, "hy3dp_vae_dec_out.raw"), dec_out)
    env["HY3DP_VAE_DEC_IN"] = "hy3dp_vae_dec_in.raw"
    env["HY3DP_VAE_DEC_OUT"] = "hy3dp_vae_dec_out.raw"

    vae = vae.to("cpu")
    del vae, posterior
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # ── 3: DINOv2-giant + ImageProjModel ──
    # FP32 on CPU deliberately: dinov2-giant hidden magnitudes reach ~400 and
    # torch-MPS fp16 inference of it collapses to corr 0.30 vs fp32 (measured
    # 2026-07-04) — an MPS numerics artifact, not reference behavior. The f32
    # CPU run is the deterministic ground truth the (f32-accumulating) MLX
    # engine is compared against.
    from transformers import Dinov2Model
    dino = Dinov2Model.from_pretrained(os.path.join(args.snapshot, "dinov2-giant"),
                                       torch_dtype=torch.float32).eval()
    dino_in = torch.from_numpy(synth_dino_input().astype(np.float32))
    with torch.no_grad():
        dino_out = dino(dino_in).last_hidden_state.float().cpu()
    dump(os.path.join(args.out, "hy3dp_dino_in.raw"), dino_in)
    dump(os.path.join(args.out, "hy3dp_dino_out.raw"), dino_out)
    env["HY3DP_DINO_IN"] = "hy3dp_dino_in.raw"
    env["HY3DP_DINO_OUT"] = "hy3dp_dino_out.raw"
    dino = dino.to("cpu")
    del dino
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # ImageProjModel weights live in the unet ckpt; apply them manually
    # (Linear 1536->4096 -> reshape [*,4,1024] -> LayerNorm(1024)).
    unet_bin = os.path.join(paint_dir, "unet", "diffusion_pytorch_model.bin")
    sd = torch.load(unet_bin, map_location="cpu", weights_only=True, mmap=True)
    pw = sd["unet.image_proj_model_dino.proj.weight"].float()
    pb = sd["unet.image_proj_model_dino.proj.bias"].float()
    nw = sd["unet.image_proj_model_dino.norm.weight"].float()
    nb = sd["unet.image_proj_model_dino.norm.bias"].float()
    with torch.no_grad():
        t = dino_out @ pw.T + pb                       # [1,257,4096]
        t = t.reshape(1, 257 * 4, 1024)                # [1,1028,1024]
        proj = torch.nn.functional.layer_norm(t, (1024,), nw, nb)
    dump(os.path.join(args.out, "hy3dp_dino_proj.raw"), proj)
    env["HY3DP_DINO_PROJ"] = "hy3dp_dino_proj.raw"
    del sd, pw, pb, nw, nb
    gc.collect()

    # ── 4+5: dual-UNet ref hiddens + one full 2.5D step (heavy; opt-in) ──
    if args.with_unet:
        if not args.repo:
            sys.exit("--with-unet needs --repo (reference clone)")
        dump_unet_oracles(args, env, device)

    # ── env block ──
    print("\n# paste before running the Zig oracles:")
    parts = []
    if args.test_model:
        parts.append(f"HY3DP_TEST_MODEL={args.test_model}")
    for k, v in env.items():
        parts.append(f"{k}={os.path.abspath(os.path.join(args.out, v))}")
    print(" \\\n".join(parts) + " \\")
    print('zig build test -Doptimize=ReleaseFast -Dtest-filter="hy3d-paint oracle"')


def synth_position_maps():
    """Deterministic position maps [1,6,3,512,512], values [0,1], bg EXACTLY 1
    (the voxel-index code detects background via position != 1)."""
    maps = []
    for v in range(6):
        y, x = np.mgrid[0:512, 0:512].astype(np.float32) / 511.0
        r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        disk = r < 0.3
        px = np.where(disk, 0.2 + 0.1 * v + 0.4 * x, 1.0)
        py = np.where(disk, 0.3 + 0.05 * v + 0.4 * y, 1.0)
        pz = np.where(disk, 0.5 + 0.3 * r, 1.0)
        maps.append(np.stack([px, py, pz]))
    return np.stack(maps)[None].astype(np.float32)


def dump_unet_oracles(args, env, device):
    """Oracle 4 (dual-UNet cached ref hiddens at t=0) + 5 (one full 2.5D step).

    Input contract (mirrored by the Zig oracle tests — see
    tests/hy3d_paint_unet_dossier.md "Oracle 4/5 input contract"):
      ref_latents [1,1,4,64,64] = randn(seed 1)*0.18215; sample [1,2,6,4,64,64]
      = randn(seed 2); embeds_normal/position [1,6,4,64,64] = randn(seed 3/4)
      *0.18215; position_maps = synth_position_maps(); dino = oracle-3 raw
      features; t=999; single forward, NO CFG (ref_scale/mva_scale 1.0).
    Canary taps (oracle 4): condition_embed_dict[down_0_0_0 / mid_0_0 / up_3_2_0].
    Output (oracle 5): v prediction [12,4,64,64].
    """
    import torch
    sys.path.insert(0, os.path.join(args.repo, "hy3dpaint"))
    paint_dir = os.path.join(args.snapshot, "hunyuan3d-paintpbr-v2-1")

    from hunyuanpaintpbr.unet.modules import UNet2p5DConditionModel
    from hunyuanpaintpbr.unet import attn_processor as ap_mod
    from einops import rearrange as _rr

    # DEVICE-ONLY patch: the reference SelfAttnProcessor2_0.__call__ hardcodes
    # `.to("cuda:0")` (attn_processor.py:750), which asserts on MPS/CPU. Same
    # math, device-preserving. No other change.
    def _mda_call(self, attn, hidden_states, encoder_hidden_states=None,
                  attention_mask=None, temb=None, *args, **kwargs):
        B = hidden_states.size(0)
        results = []
        for token, pbr_hs in zip(self.pbr_setting, torch.split(hidden_states, 1, dim=1)):
            processed = _rr(pbr_hs, "b n_pbrs n l c -> (b n_pbrs n) l c")
            results.append(self.process_single(attn, processed, None, attention_mask, temb, token, False))
        outs = [_rr(r, "(b n_pbrs n) l c -> b n_pbrs n l c", b=B, n_pbrs=1) for r in results]
        return torch.cat(outs, dim=1)

    ap_mod.SelfAttnProcessor2_0.__call__ = _mda_call

    # MEMORY-ONLY patch: torch-MPS SDPA materializes the score matrix for the
    # 24,576-token multiview attention (11.25 GiB > the Metal buffer cap).
    # Chunking the QUERY dim is mathematically exact (each softmax row is
    # complete); 2048-query chunks keep every buffer < 2.5 GB.
    _sdpa = torch.nn.functional.scaled_dot_product_attention

    def _chunked_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        L = q.shape[-2]
        if attn_mask is not None or is_causal or L <= 8192:
            return _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)
        outs = []
        for s0 in range(0, L, 2048):
            outs.append(_sdpa(q[..., s0:s0 + 2048, :], k, v, dropout_p=dropout_p, **kw))
        return torch.cat(outs, dim=-2)

    torch.nn.functional.scaled_dot_product_attention = _chunked_sdpa

    # FP32 on CPU (the "fp16-fragile giants" gotcha): the 24,576-token MA
    # softmax at fp16-MPS costs ~1.3 cos points of pure fixture noise (measured:
    # full-step cos 0.9865 vs an engine whose base-UNet blocks hit 0.99999 on
    # the same run). CPU fp32 is the deterministic ground truth; the chunked
    # SDPA patch above also bounds CPU RAM. Slow (~minutes/forward) but run once.
    device = "cpu"
    unet = UNet2p5DConditionModel.from_pretrained(
        os.path.join(paint_dir, "unet"), torch_dtype=torch.float32)
    unet = unet.to(device).eval()

    g = lambda seed: torch.Generator(device="cpu").manual_seed(seed)
    ref_latents = (torch.randn(1, 1, 4, 64, 64, generator=g(1)) * SCALING).to(torch.float16).to(device, torch.float32)
    sample = torch.randn(1, 2, 6, 4, 64, 64, generator=g(2)).to(torch.float16).to(device, torch.float32)
    embeds_normal = (torch.randn(1, 6, 4, 64, 64, generator=g(3)) * SCALING).to(torch.float16).to(device, torch.float32)
    embeds_position = (torch.randn(1, 6, 4, 64, 64, generator=g(4)) * SCALING).to(torch.float16).to(device, torch.float32)
    position_maps = torch.from_numpy(synth_position_maps()).to(torch.float16).to(device, torch.float32)
    dino_path = os.path.join(args.out, "hy3dp_dino_out.raw")
    dino_np = np.fromfile(dino_path, dtype="<f4").reshape(1, 257, 1536)
    dino_hidden = torch.from_numpy(dino_np).to(torch.float16).to(device, torch.float32)

    prompt_embeds = torch.stack(
        [unet.unet.learned_text_clip_albedo, unet.unet.learned_text_clip_mr], dim=0
    ).unsqueeze(0).to(torch.float16).to(device, torch.float32)  # [1,2,77,1024]

    # `cache` must be OUR dict object: forward mutates the kwargs-dict it
    # receives, which is a fresh dict unless we pass one in explicitly.
    shared_cache = {}
    cached_condition = dict(
        ref_latents=ref_latents,
        embeds_normal=embeds_normal,
        embeds_position=embeds_position,
        position_maps=position_maps,
        dino_hidden_states=dino_hidden,
        mva_scale=1.0,
        ref_scale=1.0,
        cache=shared_cache,
    )
    with torch.no_grad():
        out = unet(sample, torch.tensor(999, device=device), prompt_embeds, **cached_condition)
    v = out[0] if isinstance(out, (tuple, list)) else out

    cache = shared_cache["condition_embed_dict"]
    for name, key in (("down", "down_0_0_0"), ("mid", "mid_0_0"), ("up", "up_3_2_0")):
        dump(os.path.join(args.out, f"hy3dp_ref_{name}.raw"), cache[key].float())
        env[f"HY3DP_REF_{name.upper()}"] = f"hy3dp_ref_{name}.raw"

    dump(os.path.join(args.out, "hy3dp_ref_in.raw"), ref_latents.float())
    dump(os.path.join(args.out, "hy3dp_unet_latents.raw"), sample.float())
    dump(os.path.join(args.out, "hy3dp_unet_normal.raw"), embeds_normal.float())
    dump(os.path.join(args.out, "hy3dp_unet_position.raw"), embeds_position.float())
    dump(os.path.join(args.out, "hy3dp_unet_posmaps.raw"), position_maps.float())
    dump(os.path.join(args.out, "hy3dp_unet_v.raw"), v.float())
    env["HY3DP_REF_IN"] = "hy3dp_ref_in.raw"
    env["HY3DP_UNET_LATENTS"] = "hy3dp_unet_latents.raw"
    env["HY3DP_UNET_NORMAL"] = "hy3dp_unet_normal.raw"
    env["HY3DP_UNET_POSITION"] = "hy3dp_unet_position.raw"
    env["HY3DP_UNET_POSMAPS"] = "hy3dp_unet_posmaps.raw"
    env["HY3DP_UNET_V"] = "hy3dp_unet_v.raw"


if __name__ == "__main__":
    main()
