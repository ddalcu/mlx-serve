#!/usr/bin/env python3
"""Oracle for src/plugins/idol: speridlabs' Idol reference (idol-sdk, Apache-2.0) run on the CPU/MPS with a local copy
of both gated image encoders, dumping what the Zig port is checked against.

USER-RUN. Needs torch + torchvision + transformers>=4.56 + diffusers>=0.36 + tensordict + safetensors + pillow + numpy:

  git clone https://github.com/speridlabs/idol-sdk                        <sdk>
  hf download speridlabs/idol --local-dir <weights>
  hf download facebook/dinov3-vith16plus-pretrain-lvd1689m --local-dir <dinov3>     (gated: accept the terms)
  hf download black-forest-labs/FLUX.2-klein-4B --include 'vae/*' --local-dir <flux2>   (the FLUX.2 VAE, ungated copy)

  python3 tests/dump_idol_fixtures.py --sdk <sdk> --weights <weights> --dinov3 <dinov3> --flux2 <flux2> \\
      --image <any image> --out <fixtures dir> [--device mps]
  IDOL_FIXTURES=<out> IDOL_WEIGHTS=<weights> IDOL_DINOV3=<dinov3> IDOL_FLUX2=<flux2> \\
      zig build test -Doptimize=ReleaseFast -Dtest-filter="idol"

Writes <out>/fixtures.safetensors (f32 / i32, seed 0) and <out>/prepared.png:
  rgb_in, birefnet_alpha, prepared, feature1, feature2, feature2_mean, noise_latent, noise_camera, t, v_latent, v_camera,
  v_latent_uncond, oct_coords, oct_level, oct_logits, gs_points, gs_raw, gs_xyz, gs_color, gs_scaling,
  gs_rotation, gs_opacity, sample_latent, sample_camera, sample_steps
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

SEED = 0
T_MODEL = 700.0
SAMPLE_STEPS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdk", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dinov3", required=True)
    ap.add_argument("--flux2", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", default="synthetic")
    args = ap.parse_args()
    sys.path.insert(0, args.sdk)
    os.makedirs(args.out, exist_ok=True)

    import json
    from safetensors.torch import load_file
    from tensordict import TensorDict

    from idol.conditioning import ImageEncoder
    from idol.models import BiRefNet, GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder
    from idol.preprocess import CROP_MARGIN, preprocess_image
    from idol.sampler import sample_flow

    from idol.models.layers import RePo3DRotaryEmbedding

    dev = torch.device(args.device)
    cfg = json.load(open(os.path.join(args.weights, "config.json")))

    def load(model, name):
        model.load_state_dict(load_file(os.path.join(args.weights, f"{name}.safetensors")), strict=True)
        return model.eval().requires_grad_(False).to(dev)

    denoiser = load(LatentSeqMMFlowModel(**cfg["denoiser"]["args"]), "denoiser")
    decoder = load(OctreeProbabilityFixedlenDecoder(**cfg["decoder"]["args"]), "decoder")
    decoder_gs = load(GaussianFixedlenDecoder(**cfg["decoder_gs"]["args"]), "decoder_gs")
    for m in denoiser.modules():
        if isinstance(m, RePo3DRotaryEmbedding):
            m.real_valued = True  # complex64 has no MPS kernel; the real (cos, sin) form is the same rotation
    # BiRefNet runs fp32 on the CPU (torchvision's deform_conv2d has no MPS kernel).
    birefnet = BiRefNet()
    birefnet.load_state_dict(load_file(os.path.join(args.weights, "birefnet.safetensors")), strict=True)
    birefnet = birefnet.eval().requires_grad_(False)

    out = {}
    image = Image.open(args.image)
    rgb = image.convert("RGB")
    out["rgb_in"] = torch.from_numpy(np.array(rgb)).float()
    # The alpha BiRefNet predicts for the image resized as preprocess_image does (short side 1024).
    w, h = rgb.size
    scale = cfg["image_size"] / min(w, h)
    resized = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    cut = birefnet.remove_background(resized)
    out["birefnet_alpha"] = torch.from_numpy(np.array(cut.getchannel(3))).float()
    prepared = preprocess_image(image, background_remover=birefnet, image_size=cfg["image_size"],
                                crop_margin=CROP_MARGIN[args.mode])
    prepared.save(os.path.join(args.out, "prepared.png"))
    out["prepared"] = torch.from_numpy(np.array(prepared)).float()

    enc = ImageEncoder(local(args.dinov3), local(args.flux2), dinov3_dtype="fp32", device=dev)
    pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(dev)
    cond = enc(pixels)
    out["feature1"] = cond["feature1"].cpu()
    out["feature2"] = cond["feature2"].cpu()  # the reference conditions on a SAMPLE of the VAE posterior
    # The same tokens from the posterior's mean: the deterministic part the port is held to exactly.
    vae = enc.flux2_vae
    posterior = vae.encode(pixels.to(torch.bfloat16) * 2 - 1).latent_dist
    lat = posterior.mean
    b, c, h, w = lat.shape
    lat = lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    mu = vae.bn.running_mean.view(1, -1, 1, 1).to(lat.dtype)
    sd = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(lat.dtype)
    tokens = ((lat - mu) / sd).float().flatten(2).transpose(1, 2)
    out["feature2_mean"] = torch.cat([torch.zeros(1, 5, tokens.shape[2], device=tokens.device), tokens], dim=1).cpu()
    del enc

    g = torch.Generator().manual_seed(SEED)
    noise_latent = torch.randn(1, denoiser.q_token_length, denoiser.in_channels, generator=g)
    noise_camera = torch.randn(1, 1, denoiser.cam_channels, generator=g)
    out["noise_latent"], out["noise_camera"] = noise_latent, noise_camera
    x = TensorDict({"latent": noise_latent.to(dev), "camera": noise_camera.to(dev)}, batch_size=1, device=dev)
    t = torch.tensor([T_MODEL], device=dev, dtype=torch.float32)
    out["t"] = t.cpu()
    with torch.no_grad():
        v = denoiser(x, t, cond)
        out["v_latent"], out["v_camera"] = v["latent"].float().cpu(), v["camera"].float().cpu()
        vu = denoiser(x, t, cond.apply(torch.zeros_like))
        out["v_latent_uncond"] = vu["latent"].float().cpu()

        # Octree decoder: logits for fixed cell centres at one level.
        level = 3
        res = 1 << (level - 1)
        gi = torch.Generator().manual_seed(SEED + 1)
        coords_int = torch.randint(0, res, (1, 96, 3), generator=gi)
        centres = ((coords_int.float() + 0.5) / res).to(dev)
        lv = torch.full((1,), 1 << level, dtype=torch.long, device=dev)
        lat = v["latent"].float()  # any [1, 8192, 16] latent
        out["oct_coords"] = centres.cpu()
        out["oct_level"] = lv.cpu().float()
        out["oct_logits"] = decoder(centres, lv, lat).float().cpu()
        out["oct_latent"] = lat.cpu()

        # Gaussian decoder on fixed anchors.
        pts = torch.rand(1, 256, 3, generator=torch.Generator().manual_seed(SEED + 2)).to(dev)
        raw = decoder_gs(pts, lat)
        gau = decoder_gs.to_representation(pts, raw)[0]
        out["gs_points"] = pts.cpu()
        out["gs_raw"] = raw.float().cpu()
        out["gs_xyz"] = gau.xyz.float().cpu()
        out["gs_color"] = gau.features_dc.float().cpu().flatten(1)
        out["gs_scaling"] = gau.scaling.float().cpu()
        out["gs_rotation"] = gau.rotation.float().cpu()
        out["gs_opacity"] = gau.opacity.float().cpu()

        # A short guided run from the same noise: the whole sampler.
        sample = sample_flow(denoiser, x, cond=cond, neg_cond=cond.apply(torch.zeros_like), steps=SAMPLE_STEPS,
                             rescale_t=3.0, cfg_strength=3.0, verbose=False)
        out["sample_latent"] = sample["latent"].float().cpu()
        out["sample_camera"] = sample["camera"].float().cpu()
        out["sample_steps"] = torch.tensor([float(SAMPLE_STEPS)])

    save_file({k: v.contiguous().float() for k, v in out.items()}, os.path.join(args.out, "fixtures.safetensors"))
    print("wrote", os.path.join(args.out, "fixtures.safetensors"), {k: tuple(v.shape) for k, v in out.items()})


def local(path):
    """ImageEncoder takes hub ids; FLUX.2's loader wants (repo, subfolder='vae'), so give it the folder."""
    return path


if __name__ == "__main__":
    main()
