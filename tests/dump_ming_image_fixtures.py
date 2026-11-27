#!/usr/bin/env python3
"""Dump the Ming-Image parity fixtures for src/plugins/ming_image/.

Builds a TINY random-weight pack with the release's own classes (inclusionAI/Ming-Image: BailingMM2Native over
BailingMoeV2, the Qwen2 connector, DiffusionTransformer, AutoencoderKLQwenImage), each saved with
`save_pretrained` into the release's component tree, the real MLLM tokenizer copied in. The configs are the
release's with the sizes shrunk, so every switch the release reads keeps its value. Every oracle is one call into
the release's code, in FP32 on CPU:
  - the prompt ids (the processor's chat template, the fast tokenizer);
  - `get_condition_embeds_for_image_gen`: the learnable stream (connector output) and the direct stream;
  - one DiT forward at an odd canvas (both caption and image need the pad-to-32);
  - one VAE decode;
  - a 3-step pipeline from fixed noise, with and without CFG.

    <venv>/bin/python tests/dump_ming_image_fixtures.py <Ming-Image repo> <release pack dir (configs + mllm tokenizer)> <OUT>

Then:
    MING_IMAGE_TEST_MODEL=<OUT>/pack MING_IMAGE_FIXTURE=<OUT>/ming_image_fixture.safetensors \
    zig build test -Dtest-filter="ming_image oracle"
"""

import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

SEED = 7
PROMPT = 'A minimal poster that says "Ming" over a red sun.'
HEIGHT, WIDTH = 80, 112  # latent 10 x 14 -> 5 x 7 = 35 image tokens: pads to 64
STEPS = 3
CFG = 2.5
T_MODEL = 0.37  # the DiT's t in [0, 1] (the pipeline passes (1000 - t) / 1000)


def shrink(cfg, **kw):
    out = dict(cfg)
    out.update(kw)
    return out


def randomize(module, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 1 and ("norm" in name or "layernorm" in name or name.endswith(".gamma")):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                fan = p.shape[-1] if p.ndim > 1 else 1
                p.copy_(torch.randn(p.shape, generator=g) * (0.8 / fan**0.5 if p.ndim > 1 else 0.05))
        for name, b in module.named_buffers():
            if name.endswith("expert_bias"):
                b.copy_(0.05 * torch.randn(b.shape, generator=g))


def main():
    repo, release, out = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, repo)
    from configuration_bailingmm2 import BailingMM2Config
    from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from diffusion.transformer import DiffusionTransformer
    from diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

    torch.manual_seed(SEED)
    pack = os.path.join(out, "pack")
    shutil.rmtree(pack, ignore_errors=True)
    os.makedirs(pack)

    # ── mllm/ ──
    rel = json.load(open(os.path.join(release, "mllm", "config.json")))
    llm = shrink(rel["llm_config"], hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=2,
                 num_key_value_heads=1, head_dim=128, num_experts=8, num_experts_per_tok=2, n_group=2, topk_group=1,
                 moe_intermediate_size=64, _attn_implementation="eager")
    vis = shrink(rel["vision_config"], depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=32,
                 fullatt_block_indexes=[0], _attn_implementation="eager")
    cfg = BailingMM2Config(mlp_depth=rel.get("mlp_depth", 2), llm_config=llm, vision_config=vis)
    cfg.architectures = rel["architectures"]
    model = BailingMM2NativeForConditionalGeneration(cfg)
    randomize(model, 1)
    model.save_pretrained(os.path.join(pack, "mllm"), safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(release, "mllm", f), os.path.join(pack, "mllm", f))
    shutil.copy(os.path.join(release, "config.json"), os.path.join(pack, "config.json"))

    # ── connector/ ──
    crel = json.load(open(os.path.join(release, "connector", "config.json")))
    ccfg = Qwen2Config(**shrink({k: v for k, v in crel.items() if k not in ("architectures", "layer_types", "dtype")},
                                hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
                                num_key_value_heads=1, vocab_size=64, max_window_layers=2))
    conn = Qwen2ForCausalLM(ccfg)
    randomize(conn, 2)
    conn.save_pretrained(os.path.join(pack, "connector"), safe_serialization=True)

    # ── mlp/ ──
    h, ch, cap, inner = 64, 32, 24, 64
    mlp_cfg = json.load(open(os.path.join(release, "mlp", "config.json")))
    mlp_cfg.update(img_gen_scales=[2], selected_hidden_states_layers=[1, 2, 3], diffusion_c_input_dim=cap,
                   diffusion_inner_dim=inner)
    os.makedirs(os.path.join(pack, "mlp"))
    json.dump(mlp_cfg, open(os.path.join(pack, "mlp", "config.json"), "w"))
    g = torch.Generator().manual_seed(3)
    rn = lambda *s: torch.randn(*s, generator=g)
    save_file({
        "query_tokens_dict.2x2": torch.nn.functional.normalize(rn(4, h), dim=-1),
        "proj_in.weight": rn(ch, h) * 0.1, "proj_in.bias": rn(ch) * 0.05,
        "proj_out.weight": rn(cap, ch) * 0.15, "proj_out.bias": rn(cap) * 0.05,
        "proj_directvlm.0.weight": 1.0 + 0.1 * rn(3 * h),
        "proj_directvlm.1.weight": rn(inner, 3 * h) * 0.07, "proj_directvlm.1.bias": rn(inner) * 0.05,
    }, os.path.join(pack, "mlp", "model.safetensors"))

    # ── transformer/, vae/, scheduler/ ──
    dit = DiffusionTransformer(in_channels=16, dim=64, n_layers=2, n_refiner_layers=1, n_heads=2, n_kv_heads=2,
                               cap_feat_dim=cap, axes_dims=[8, 12, 12], axes_lens=[1024, 64, 64],
                               alignment_padding_mode="zero_masked", multi_frame_output=False)
    randomize(dit, 4)
    dit.save_pretrained(os.path.join(pack, "transformer"), safe_serialization=True)
    vae = AutoencoderKLQwenImage(base_dim=8, z_dim=16, input_channels=4, latents_mean=None, latents_std=None)
    randomize(vae, 5)
    vae.save_pretrained(os.path.join(pack, "vae"), safe_serialization=True)
    vcfg = json.load(open(os.path.join(pack, "vae", "config.json")))
    vcfg.update(scaling_factor=8.0064, shift_factor=0.0)
    json.dump(vcfg, open(os.path.join(pack, "vae", "config.json"), "w"))
    shutil.copytree(os.path.join(release, "scheduler"), os.path.join(pack, "scheduler"))

    # ── the oracles, through the release's loader ──
    ref = BailingMM2NativeForConditionalGeneration.from_pretrained(pack, load_image_gen=True, torch_dtype=torch.float32,
                                                                    image_gen_device="cpu", attn_implementation="eager")
    ref.eval()
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(pack, "mllm", "tokenizer.json"))
    text = ("<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|>"
            f"<role>HUMAN</role>{PROMPT}<|role_end|><role>ASSISTANT</role>")
    ids = torch.tensor([tok.encode(text, add_special_tokens=False)])
    fx = {"ids": ids[0].to(torch.int32)}
    with torch.no_grad():
        learn, direct = ref.get_condition_embeds_for_image_gen(ids, torch.ones_like(ids), None, None, False, None, None)
        fx["learn"], fx["direct"] = learn[0].float().contiguous(), direct[0].float().contiguous()

        gen = ref.diffusion_loss
        lat = torch.randn(1, 16, HEIGHT // 8, WIDTH // 8, generator=torch.Generator().manual_seed(SEED))
        dit_out = gen.train_model(list(lat.unsqueeze(2).unbind(0)), torch.tensor([T_MODEL]), [learn[0]], return_dict=False,
                              encoder_hidden_states_2=[direct[0]])[0]
        fx["dit_latent"], fx["dit_out"] = lat.contiguous(), dit_out[0][:, 0].float().contiguous()

        z = torch.randn(1, 16, 1, HEIGHT // 8, WIDTH // 8, generator=torch.Generator().manual_seed(SEED + 1))
        fx["vae_in"] = z[:, :, 0].contiguous()
        fx["vae_out"] = gen.vae.decode(z, return_dict=False)[0][:, :, 0].clamp(-1, 1).float().contiguous()

        noise = torch.randn(1, 16, HEIGHT // 8, WIDTH // 8, generator=torch.Generator().manual_seed(SEED + 2))
        fx["e2e_noise"] = noise.contiguous()
        for name, cfg_scale in (("e2e", 1.0), ("e2e_cfg", CFG)):
            img = gen.pipelines(prompt_embeds=[learn[0]], negative_prompt_embeds=[learn[0] * 0],
                                prompt_embeds_2=[direct[0]], negative_prompt_embeds_2=[direct[0] * 0],
                                guidance_scale=cfg_scale, num_inference_steps=STEPS, height=HEIGHT, width=WIDTH,
                                latents=noise.clone(), output_type="pt", device="cpu", sample_mode="argmax").images
            fx[name] = img.float().contiguous()

    meta = {"prompt": PROMPT, "steps": str(STEPS), "cfg": str(CFG), "t": str(T_MODEL)}
    save_file(fx, os.path.join(out, "ming_image_fixture.safetensors"), metadata=meta)
    for k, v in fx.items():
        print(f"{k:12s} {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
