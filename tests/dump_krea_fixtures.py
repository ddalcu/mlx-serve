#!/usr/bin/env python3
"""Dump Krea-2-Turbo parity fixtures (.raw) for the Zig oracle tests in src/krea.zig.

This runs the pure-MLX REFERENCE (avlp12/krea2_alis_mlx) and writes byte-exact
golden tensors that `zig build test` compares the Zig port against (cosine
parity). It is USER-RUN: you accept Krea's gated license, download the model, and
`pip install mflux transformers huggingface_hub` per the reference's
requirements.txt.

Usage:
    python3 tests/dump_krea_fixtures.py <ASSEMBLED_KREA_DIR> [OUT_DIR] [REF_DIR]

  ASSEMBLED_KREA_DIR  a directory containing:
      transformer_8bit.safetensors  (or transformer_mixed_4_8.safetensors / turbo.safetensors)
      text_encoder/*.safetensors    (Qwen3-VL-4B, language_model.* keys)
      vae/*.safetensors             (Qwen-Image VAE)
      tokenizer/*                   (HF Qwen tokenizer)
      config.json                   ({"model_type":"krea2_turbo"})  — needed by the server, not here
  OUT_DIR             where to write the .raw fixtures (default: ./krea_fixtures)
  REF_DIR             the krea2_alis_mlx reference checkout (must contain krea2/)
                      (default: the path baked in below; override if you moved it)

It prints the exact `export KREA_*` env block to paste before running:
    KREA_TEST_MODEL=<ASSEMBLED_KREA_DIR> KREA_IDS=... [all vars] zig build test \
        -Doptimize=ReleaseFast -Dtest-filter="krea"

The four oracles validated: text encoder (cos>0.9995), DiT velocity (cos>0.99),
VAE decode (cos>0.99), end-to-end pixels (cos>0.95).
"""

import os
import sys

import numpy as np

# Fixed, reproducible fixture parameters. Keep in sync with the env defaults in
# the krea.zig e2e oracle (W/H 512, seed 0, steps 8).
PROMPT = "a red fox in the snow, photorealistic"
W, H, STEPS, SEED = 512, 512, 8, 0

DEFAULT_REF = os.environ.get("KREA2_REF", os.path.expanduser("~/krea2_ref"))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    model_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.abspath("krea_fixtures")
    ref_dir = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_REF
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, ref_dir)

    import mlx.core as mx
    from krea2 import sampling
    from krea2.pipeline import Krea2Pipeline, resolve_weights
    from krea2.text_encoder import PREFIX, PREFIX_START_IDX, SELECT_LAYERS, SUFFIX, SUFFIX_START_IDX

    def save_f32(name, arr):
        np.asarray(arr, dtype=np.float32).tofile(os.path.join(out_dir, name))

    def save_i32(name, arr):
        np.asarray(arr, dtype=np.int32).reshape(-1).tofile(os.path.join(out_dir, name))

    print(f"[dump] model_dir={model_dir}\n[dump] out_dir={out_dir}\n[dump] ref_dir={ref_dir}")
    precision, tpath = resolve_weights(model_dir, precision=None, download=False)
    if tpath is None:
        sys.exit(f"No transformer_*.safetensors found in {model_dir}")
    print(f"[dump] precision={precision} transformer={tpath}")
    pipe = Krea2Pipeline(transformer_path=tpath, precision=precision, base_dir=model_dir)
    T, vae, enc = pipe.transformer, pipe.vae, pipe.encoder

    # ── encoder: replicate Qwen3VLConditioner.__call__ to capture ids/mask ──
    tok = enc.tokenizer
    text = [PREFIX + PROMPT]
    suf = tok(text=[SUFFIX], return_tensors="np")
    inp = tok(
        text, truncation=True, padding="max_length",
        max_length=enc.max_length + PREFIX_START_IDX - SUFFIX_START_IDX, return_tensors="np",
    )
    input_ids = np.concatenate([inp["input_ids"], suf["input_ids"]], axis=1)  # (1, 546)
    attn = np.concatenate([inp["attention_mask"], suf["attention_mask"]], axis=1)
    save_i32("krea_ids.raw", input_ids[0])
    save_i32("krea_mask.raw", attn[0])

    all_hs = enc.model(mx.array(input_ids.astype(np.int32)), mx.array(attn.astype(np.float32)))
    stacked = mx.stack([all_hs[i] for i in SELECT_LAYERS], axis=2)[:, PREFIX_START_IDX:]  # (1,512,12,2560)
    mx.eval(stacked)
    save_f32("krea_pe.raw", np.array(stacked.astype(mx.float32)))
    ctx = stacked.astype(mx.bfloat16)
    out_mask = mx.array(attn.astype(np.float32))[:, PREFIX_START_IDX:]  # (1,512) validity
    print(f"[dump] encoder ctx shape={tuple(stacked.shape)}")

    # ── shared geometry + seed-exact noise (explicit key, matching the Zig path) ──
    patch = T.cfg.patch
    comp = vae.spatial_scale  # 8
    align = comp * patch  # 16
    Wd, Hd = sampling.roundup(W, align), sampling.roundup(H, align)
    lat_h, lat_w = Hd // comp, Wd // comp
    h_, w_ = lat_h // patch, lat_w // patch
    key = mx.random.key(SEED)
    noise = mx.random.normal((1, vae.latent_channels, lat_h, lat_w), key=key).astype(mx.bfloat16)
    img0 = sampling.patchify(noise, patch)  # (1, h_*w_, 64)
    txtlen = ctx.shape[1]
    pos = sampling.build_positions(1, txtlen, h_, w_)  # (L,3)
    valid = mx.concatenate([out_mask, mx.ones((1, h_ * w_))], axis=1)  # (1,L) validity

    # ── DiT: step-0 velocity at t=1.0 ──
    t = mx.full((1,), 1.0, dtype=mx.bfloat16)
    v = T(img0, ctx, t, pos, valid)
    mx.eval(v)
    save_f32("krea_vel_img.raw", np.array(img0.astype(mx.float32)))
    save_f32("krea_vel_ctx.raw", np.array(ctx.astype(mx.float32)))
    save_i32("krea_vel_pos.raw", np.array(pos).astype(np.int32))
    save_i32("krea_vel_valid.raw", np.array(valid[0]).astype(np.int32))
    save_f32("krea_vel.raw", np.array(v.astype(mx.float32)))
    print(f"[dump] DiT velocity shape={tuple(v.shape)} (img_len={h_ * w_}, L={txtlen + h_ * w_})")

    # ── full denoise → realistic latent → VAE decode (oracle + e2e) ──
    x1 = (256 // align) ** 2
    x2 = (1280 // align) ** 2
    ts = sampling.timesteps(img0.shape[1], STEPS, x1, x2)
    cur = img0
    for tc, tp in zip(ts[:-1], ts[1:]):
        tt = mx.full((1,), tc, dtype=mx.bfloat16)
        vv = T(cur, ctx, tt, pos, valid)
        cur = cur + (tp - tc) * vv
        mx.eval(cur)
    latent = sampling.unpatchify(cur, patch, h_, w_, vae.latent_channels)  # (1,16,lat_h,lat_w)
    save_f32("krea_vae_latent.raw", np.array(latent.astype(mx.float32)))

    raw_dec = vae.decode(latent.astype(mx.float32))  # (1,3,1,H,W) in ~[-1,1]
    raw_dec0 = raw_dec[:, :, 0]  # (1,3,H,W) — matches Zig vae.decode output
    mx.eval(raw_dec0)
    save_f32("krea_decoded.raw", np.array(raw_dec0.astype(mx.float32)))

    e2e = (mx.clip(raw_dec, -1, 1) * 0.5 + 0.5)[:, :, 0]  # (1,3,H,W) in [0,1]
    mx.eval(e2e)
    save_f32("krea_e2e.raw", np.array(e2e.astype(mx.float32)))
    print(f"[dump] decoded shape={tuple(raw_dec0.shape)} lat_h={lat_h} lat_w={lat_w}")

    # ── print the env block to run the oracles ──
    o = out_dir
    env = {
        "KREA_TEST_MODEL": model_dir,
        "KREA_IDS": f"{o}/krea_ids.raw",
        "KREA_MASK": f"{o}/krea_mask.raw",
        "KREA_PE": f"{o}/krea_pe.raw",
        "KREA_VEL_IMG": f"{o}/krea_vel_img.raw",
        "KREA_VEL_CTX": f"{o}/krea_vel_ctx.raw",
        "KREA_VEL_POS": f"{o}/krea_vel_pos.raw",
        "KREA_VEL_VALID": f"{o}/krea_vel_valid.raw",
        "KREA_VEL": f"{o}/krea_vel.raw",
        "KREA_VAE_LATENT": f"{o}/krea_vae_latent.raw",
        "KREA_VAE_LATH": str(lat_h),
        "KREA_DECODED": f"{o}/krea_decoded.raw",
        "KREA_E2E": f"{o}/krea_e2e.raw",
        "KREA_E2E_W": str(W),
        "KREA_E2E_H": str(H),
        "KREA_E2E_SEED": str(SEED),
        "KREA_E2E_STEPS": str(STEPS),
    }
    print("\n# ── paste this to run the Zig oracle tests ──")
    print(" \\\n".join(f"{k}={v}" for k, v in env.items()) + " \\")
    print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="krea"')


if __name__ == "__main__":
    main()
