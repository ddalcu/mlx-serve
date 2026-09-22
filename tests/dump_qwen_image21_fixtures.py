#!/usr/bin/env python3
"""Dump the Qwen-Image-2.1 parity fixtures for src/qwen_image.zig.

Builds a TINY random-weight DiT + VAE with the pure-MLX reference classes
(mflux PR #736, ivanfioravanti/mflux@qwen-image-2.1), writes them as a pack in
the checkpoint's OWN layout and key names (what the engine loads), and dumps
one forward of each in FP32. No real checkpoint needed.

    <venv>/bin/python tests/dump_qwen_image21_fixtures.py <mflux_repo_root> <OUT>

Then:
    QWEN_IMAGE_TEST_MODEL=<OUT>/pack \
    QWEN_IMAGE_FIXTURE=<OUT>/qwen_image21_fixture.safetensors \
    zig build test -Doptimize=ReleaseFast -Dtest-filter="QwenImage"
"""

import json
import os
import sys

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

SEED = 7
DIT = dict(in_channels=8, out_channels=8, num_layers=2, attention_head_dim=16, num_attention_heads=2,
           context_in_dim=24, mlp_ratio=3, axes_dims_rope=(4, 6, 6), eps=1e-6)
VAE = dict(base_dim=8, decoder_base_dim=12, z_dim=8, dim_mult=(1, 2, 4, 8, 8), num_res_blocks=2,
           temperal_downsample=(False, True, True, True), in_channels=4, out_channels=4)


def randomize(module, rng):
    """Every parameter ~N(0, 0.3) except norm scales, kept near their identity."""
    params = []
    for name, p in tree_flatten(module.parameters()):
        if name.endswith("freqs") or "tables" in name:
            continue
        v = rng.normal(0.0, 0.3, p.shape).astype(np.float32)
        if p.ndim == 1 and ("norm" in name and name.endswith("weight")):
            v = (1.0 + 0.1 * v) if "text_norm" not in name else 0.1 * v
        params.append((name, mx.array(v)))
    module.load_weights(params, strict=False)
    return dict(params)


def dit_to_hf(params):
    return {k.replace("modulation.layers.1.", "modulation.1."): v for k, v in params.items()}


def vae_to_hf(params):
    out = {}
    for k, v in params.items():
        k = k.replace(".downsampler.conv.", ".downsampler.resample.1.").replace(".upsampler.conv.", ".upsampler.resample.1.")
        k = k.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias")
        if v.ndim == 4:  # mlx OHWI -> torch OIHW
            v = mx.transpose(v, (0, 3, 1, 2))
        if "norm" in k and k.endswith(".weight"):
            k = k[: -len("weight")] + "gamma"
            v = v.reshape(-1, 1, 1) if "attentions" in k else v.reshape(-1, 1, 1, 1)
        out[k] = v
    return out


def main():
    mflux_root, out_dir = sys.argv[1], sys.argv[2]
    sys.path.insert(0, os.path.join(mflux_root, "src"))
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
    from mflux.models.qwen21.model.qwen21_vae.qwen21_causal_conv import Qwen21CausalConv
    from mflux.models.qwen21.model.qwen21_vae.qwen21_decoder import Qwen21Decoder
    from mflux.models.qwen21.model.qwen21_vae.qwen21_encoder import Qwen21Encoder

    ModelConfig.precision = mx.float32
    rng = np.random.default_rng(SEED)
    pack = os.path.join(out_dir, "pack")
    for sub in ("transformer", "vae"):
        os.makedirs(os.path.join(pack, sub), exist_ok=True)
    fx = {}

    # ── DiT: one padding-free forward (the segmented block-causal path) ──
    dit = Qwen21Transformer(**DIT)
    dit_params = randomize(dit, rng)
    text_len, lat_h, lat_w, t = 5, 4, 6, 0.7
    img = mx.array(rng.normal(size=(1, lat_h * lat_w, DIT["in_channels"])).astype(np.float32))
    txt = mx.array(rng.normal(size=(1, text_len, DIT["context_in_dim"])).astype(np.float32))
    cos, sin = dit.pos_embed(text_len, lat_h, lat_w)
    rows = mx.array(np.array([t, 0.0], dtype=np.float32))
    fx["dit_img"], fx["dit_txt"] = img, txt
    fx["dit_t"] = mx.array(np.array([t], dtype=np.float32))
    fx["dit_lat_hw"] = mx.array(np.array([lat_h, lat_w], dtype=np.int32))
    fx["dit_rope_cos"], fx["dit_rope_sin"] = cos, sin
    fx["dit_out"] = dit._forward(img, txt, rows, cos, sin, None)
    mx.save_safetensors(os.path.join(pack, "transformer", "diffusion_pytorch_model.safetensors"), dit_to_hf(dit_params))
    cfg = dict(DIT, axes_dims_rope=list(DIT["axes_dims_rope"]), patch_size=1, causal_condition=True,
               _class_name="QwenImage21Transformer2DModel")
    json.dump(cfg, open(os.path.join(pack, "transformer", "config.json"), "w"), indent=2)

    # ── VAE: decode + encode, mean/std applied exactly as Qwen21VAE does ──
    z = VAE["z_dim"]
    enc = Qwen21Encoder(in_channels=4, dim=VAE["base_dim"], z_dim=2 * z, dim_mult=VAE["dim_mult"],
                        num_res_blocks=2, temperal_downsample=VAE["temperal_downsample"])
    dec = Qwen21Decoder(dim=VAE["decoder_base_dim"], z_dim=z, dim_mult=VAE["dim_mult"], num_res_blocks=2,
                        temperal_upsample=VAE["temperal_downsample"][::-1], out_channels=4)
    quant, post = Qwen21CausalConv(2 * z, 2 * z, 1, 0), Qwen21CausalConv(z, z, 1, 0)
    vae_params = {}
    for prefix, mod in (("encoder", enc), ("decoder", dec), ("quant_conv", quant), ("post_quant_conv", post)):
        vae_params.update({f"{prefix}.{k}": v for k, v in randomize(mod, rng).items()})
    mean = rng.normal(size=(z,)).astype(np.float32)
    std = (3.0 + rng.random(size=(z,))).astype(np.float32)
    m4, s4 = mx.array(mean).reshape(1, z, 1, 1), mx.array(std).reshape(1, z, 1, 1)

    lat = mx.array(rng.normal(size=(1, z, 2, 3)).astype(np.float32))
    fx["vae_latent"] = lat
    fx["vae_decoded"] = dec(post(lat * s4 + m4))[:, :3]
    image = mx.array(rng.uniform(-1, 1, size=(1, 3, 32, 48)).astype(np.float32))
    fx["vae_image"] = image
    rgba = mx.concatenate([image, mx.ones_like(image[:, :1])], axis=1)
    fx["vae_encoded"] = (quant(enc(rgba))[:, :z] - m4) / s4
    mx.save_safetensors(os.path.join(pack, "vae", "diffusion_pytorch_model.safetensors"), vae_to_hf(vae_params))
    vcfg = dict(VAE, dim_mult=list(VAE["dim_mult"]), temperal_downsample=list(VAE["temperal_downsample"]),
                latents_mean=mean.tolist(), latents_std=std.tolist(), _class_name="AutoencoderKLQwenImage21")
    json.dump(vcfg, open(os.path.join(pack, "vae", "config.json"), "w"), indent=2)
    json.dump({"_class_name": "QwenImage21Pipeline"}, open(os.path.join(pack, "model_index.json"), "w"))

    mx.eval(list(fx.values()))
    mx.save_safetensors(os.path.join(out_dir, "qwen_image21_fixture.safetensors"), fx)
    for k, v in fx.items():
        print(k, v.shape, v.dtype)


if __name__ == "__main__":
    main()
