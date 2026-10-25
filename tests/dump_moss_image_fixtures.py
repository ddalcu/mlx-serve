#!/usr/bin/env python3
"""Dump the MOSS-Image parity fixtures for src/moss_image.zig.

Builds a TINY random-weight pack with the reference classes themselves: the
upstream `pipeline_moss_image.py` (MOSSImageTransformer + MOSSImageScheduler),
transformers' Qwen3_5Model as the text encoder and diffusers'
AutoencoderKLFlux2, each saved with `save_pretrained` so the pack has the
release's own layout and key names. The real tokenizer is copied in. Every
oracle below is one call into those classes, in FP32 on CPU.

    PYTHONPATH=<diffusers>=0.38> <venv>/bin/python tests/dump_moss_image_fixtures.py \
        <pipeline_moss_image.py> <real tokenizer dir> <OUT>

Then:
    MOSS_IMAGE_TEST_MODEL=<OUT>/pack \
    MOSS_IMAGE_FIXTURE=<OUT>/moss_image_fixture.safetensors \
    zig build test -Doptimize=ReleaseFast -Dtest-filter="MOSS-Image"
"""

import importlib.util
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

SEED = 11
PROMPT = 'A poster titled "MOSS" with a cat.'
NEGATIVE = "worst quality"
HEIGHT, WIDTH = 64, 96  # latent 4 x 6: non-square catches a transposed grid
TIMESTEP = 700.0
SKIP = [1]
E2E_STEPS, E2E_CFG = 4, 3.0

TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "attention_bias": False,
    "attn_output_gate": True,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 128,
    "intermediate_size": 256,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 1,
    "linear_num_value_heads": 2,
    "linear_value_head_dim": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 4096,
    "num_attention_heads": 2,
    "num_hidden_layers": 4,
    "num_key_value_heads": 1,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
    "tie_word_embeddings": False,
    "vocab_size": 248320,
}
VISION_CONFIG = {"model_type": "qwen3_5_vision", "depth": 1, "hidden_size": 32, "intermediate_size": 64,
                 "num_heads": 2, "out_hidden_size": 128, "patch_size": 16, "spatial_merge_size": 2,
                 "temporal_patch_size": 2, "num_position_embeddings": 64, "in_channels": 3,
                 "deepstack_visual_indexes": []}
DIT = dict(dim=64, ffn_dim=96, num_heads=2, num_layers=3, text_dim=128, freq_dim=256,
           latent_channels=128, rope_base=10000.0)
VAE = dict(in_channels=3, out_channels=3, latent_channels=32, block_out_channels=(32, 32, 32, 32),
           decoder_block_out_channels=(32, 32, 64, 64), layers_per_block=2, norm_num_groups=32,
           patch_size=(2, 2), use_quant_conv=True, use_post_quant_conv=True, mid_block_add_attention=True,
           batch_norm_eps=1e-4)


def load_reference(path):
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def randomize(module, gen):
    # Release norms are trained away from their init: zero-init (HF's 1+w) or
    # unit weights would hide a missing offset, so every tensor gets noise.
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 1 and name.endswith("weight"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.1 + (0.0 if "language_model" in name and "linear_attn.norm" not in name else 1.0))
            elif "A_log" in name or "dt_bias" in name:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * (0.5 / max(1, p.shape[-1]) ** 0.5))


def main():
    ref_path, tok_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    ref = load_reference(ref_path)
    from diffusers import AutoencoderKLFlux2
    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5Model

    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    pack = os.path.join(out, "pack")
    shutil.rmtree(pack, ignore_errors=True)
    os.makedirs(pack)

    tok = AutoTokenizer.from_pretrained(tok_dir)
    shutil.copytree(tok_dir, os.path.join(pack, "tokenizer"))

    cfg = Qwen3_5Config(text_config=TEXT_CONFIG, vision_config=VISION_CONFIG)
    te = Qwen3_5Model(cfg).float().eval()
    randomize(te, gen)
    te.save_pretrained(os.path.join(pack, "text_encoder"))

    dit = ref.MOSSImageTransformer(**DIT).float().eval()
    randomize(dit, gen)
    dit.save_pretrained(os.path.join(pack, "transformer"))

    vae = AutoencoderKLFlux2(**VAE).float().eval()
    randomize(vae, gen)
    with torch.no_grad():
        vae.bn.running_mean.copy_(torch.randn(vae.bn.running_mean.shape, generator=gen) * 0.2)
        vae.bn.running_var.copy_(torch.rand(vae.bn.running_var.shape, generator=gen) + 0.5)
    vae.save_pretrained(os.path.join(pack, "vae"))

    sched = ref.MOSSImageScheduler()
    sched.save_pretrained(os.path.join(pack, "scheduler"))
    with open(os.path.join(pack, "model_index.json"), "w") as f:
        json.dump({"_class_name": "MOSSImagePipeline", "_diffusers_version": "0.38.0",
                   "scheduler": ["pipeline_moss_image", "MOSSImageScheduler"],
                   "text_encoder": ["transformers", "Qwen3_5Model"],
                   "tokenizer": ["transformers", "Qwen2Tokenizer"],
                   "transformer": ["pipeline_moss_image", "MOSSImageTransformer"],
                   "vae": ["diffusers", "AutoencoderKLFlux2"]}, f, indent=2)

    pipe = ref.MOSSImagePipeline(dit, sched, te, tok, vae)
    fx = {}
    with torch.no_grad():
        formatted = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True)
        fx["te_ids"] = tok(formatted, return_tensors="pt").input_ids[0].to(torch.int32)
        pos, pos_mask = pipe.encode_prompt(PROMPT)
        neg, neg_mask = pipe.encode_prompt(NEGATIVE)
        fx["te_hidden"] = pos
        fx["te_hidden_neg"] = neg

        h, w = HEIGHT // 16, WIDTH // 16
        lat = torch.randn((1, 128, h, w), generator=gen)
        fx["dit_latents"] = lat
        fx["dit_out"] = dit(lat, TIMESTEP, pos, pos_mask).sample
        fx["dit_out_skip"] = dit(lat, TIMESTEP, neg, neg_mask, skip_layers=SKIP).sample

        for n in (50, 8, 4):
            sched.set_timesteps(n)
            fx[f"sigmas_{n}"] = sched.sigmas.float().clone()

        z = torch.randn((1, 128, h, w), generator=gen)
        fx["vae_in"] = z
        bn, eps = vae.bn, vae.config.batch_norm_eps
        zz = z * (bn.running_var + eps).sqrt().view(1, -1, 1, 1) + bn.running_mean.view(1, -1, 1, 1)
        zz = zz.reshape(1, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, 2 * h, 2 * w)
        fx["vae_out"] = vae.decode(zz, return_dict=False)[0]

        init = torch.randn((1, 128, h, w), generator=gen)
        fx["e2e_init"] = init
        image = pipe(PROMPT, negative_prompt=NEGATIVE, height=HEIGHT, width=WIDTH,
                     num_inference_steps=E2E_STEPS, guidance_scale=E2E_CFG, uncond_skip_layers=SKIP,
                     latents=init.clone(), output_type="pt").images
        fx["e2e_image"] = image

    save_file({k: v.contiguous() for k, v in fx.items()}, os.path.join(out, "moss_image_fixture.safetensors"))
    print("wrote", out)


if __name__ == "__main__":
    main()
