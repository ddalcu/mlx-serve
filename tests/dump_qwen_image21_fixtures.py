#!/usr/bin/env python3
"""Dump the Qwen-Image-2.1 parity fixtures for src/qwen_image.zig.

Builds a TINY random-weight DiT + VAE with the pure-MLX reference classes
(mflux PR #736, ivanfioravanti/mflux@qwen-image-2.1), writes them as a pack in
the checkpoint's OWN layout and key names (what the engine loads), and dumps
one forward of each in FP32. No real checkpoint needed.

The EDIT half adds instruction-edit references produced by the actual pinned
classes (diffusers@8b3c707 + transformers Qwen3VL, never hand-rolled math):
the pack's text_encoder/ becomes a full tiny Qwen3VLForConditionalGeneration
(LM + Qwen3-VL tower, seeded random fp32), the real processor/ is copied in,
and the diffusers QwenImage21Pipeline drives the reference preprocessing,
prompt encoding (pre-final-RMSNorm hook), joint construction, and ONE edit DiT
forward. <OUT>/pack-notower is the same pack minus the model.visual.* tensors
(edit capability off).

    <venv>/bin/python tests/dump_qwen_image21_fixtures.py <mflux_repo_root> <OUT>
    # run env: PYTHONPATH=<diffusers/src>:<mflux/src>; real processor files at
    # ~/claude-tmp/qwen21-src/processor (copied in, never re-downloaded).

Then:
    QWEN_IMAGE_TEST_MODEL=<OUT>/pack \
    QWEN_IMAGE_FIXTURE=<OUT>/qwen_image21_fixture.safetensors \
    zig build test -Doptimize=ReleaseFast -Dtest-filter="QwenImage"

The edit DiT oracle runs the PACK's transformer weights at the pack's own
config — including its axes_dims_rope (4,6,6), because diffusers requires
sum(axes_dims_rope) == attention_head_dim and the pack's head dim is 16 (the
real checkpoint's (16,56,56) pairs with head dim 128). The real (16,56,56)
layout is still pinned, as a pure QwenImage21Rope table, in
edit_rope_cos_real/edit_rope_sin_real.
"""

import glob
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

# ── edit-half constants ──
# The Zig TextEncoder hardcodes the LM geometry; the tower is sized by the
# pack's text_encoder/config.json vision_config. LM hidden == the DiT's
# context_in_dim (24) so edit_hidden feeds txt_in unmodified.
EDIT_SEED_TE = 21        # tiny Qwen3VL random init
EDIT_SEED_SRC = 22       # reference RGBA sources
EDIT_SEED_LAT = 23       # packed edit latents
EDIT_SEED_VAE = 24       # real-alpha VAE-encode oracle input
EDIT_PROMPT = "make the fox wear a tiny hat"
EDIT_NEG_PROMPT = "blurry"
EDIT_TIMESTEP = 700.0    # scheduler-style t; the DiT sees t/1000
EDIT_OUTPUT_RESOLUTION = 1024
EDIT_SRC_SHAPES = ((384, 512), (256, 256))  # (h, w) RGBA sources, primary first
# The real Qwen-Image-2.1 processor/tokenizer files, copied verbatim into the packs.
EDIT_PROCESSOR_SRC = os.path.expanduser("~/claude-tmp/qwen21-src/processor")
EDIT_SCHEDULER_SRC = os.path.expanduser("~/claude-tmp/qwen21-src/scheduler/scheduler_config.json")
# Mirrors the real text_encoder/config.json structurally with tiny numbers.
EDIT_TE_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "dtype": "float32",
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 24,
        "initializer_range": 0.02,
        "intermediate_size": 64,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": False,
    "transformers_version": "5.17.0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [1, 2, 3],
        "depth": 4,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 128,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 256,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 24,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}


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


def f32(t):
    return mx.array(t.detach().cpu().numpy().astype(np.float32))


def i32(t):
    return mx.array(t.detach().cpu().numpy().astype(np.int32))


class SpyProcessor:
    """Records every processor call _get_qwen_prompt_embeds makes (ids, pixel_values, grids)."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def __call__(self, **kwargs):
        out = self.inner(**kwargs)
        self.calls.append((kwargs, out))
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)


def dump_edit_fixtures(pack, vae):
    """Drive the pinned diffusers/transformers reference classes on the tiny pack.

    vae = the t2i half's {enc, quant, m4, s4, z} — the mflux VAE oracle with a
    real-alpha RGBA input reuses the SAME pack weights.
    """
    import shutil

    import torch
    from PIL import Image as PILImage
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    # ── tiny Qwen3VL text encoder: seeded random weights in the real layout ──
    te_dir = os.path.join(pack, "text_encoder")
    os.makedirs(te_dir, exist_ok=True)
    te_cfg_path = os.path.join(te_dir, "config.json")
    json.dump(EDIT_TE_CONFIG, open(te_cfg_path, "w"), indent=2)
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    cfg = Qwen3VLConfig.from_pretrained(te_dir)
    torch.manual_seed(EDIT_SEED_TE)
    model = Qwen3VLForConditionalGeneration(cfg).float()
    sd = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(sd, os.path.join(te_dir, "model.safetensors"))
    del model

    # ── real processor + scheduler, the pack's model_index/root config ──
    shutil.copytree(EDIT_PROCESSOR_SRC, os.path.join(pack, "processor"), dirs_exist_ok=True)
    os.makedirs(os.path.join(pack, "scheduler"), exist_ok=True)
    shutil.copy(EDIT_SCHEDULER_SRC, os.path.join(pack, "scheduler", "scheduler_config.json"))
    json.dump({"_class_name": "QwenImage21Pipeline", "_diffusers_version": "0.37.0.dev0",
               "processor": ["transformers", "Qwen3VLProcessor"],
               "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
               "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
               "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
               "vae": ["diffusers", "AutoencoderKLQwenImage21"]},
              open(os.path.join(pack, "model_index.json"), "w"), indent=2)
    json.dump({"model_type": "qwen_image21"}, open(os.path.join(pack, "config.json"), "w"), indent=2)

    # ── the reference pipeline on the tiny pack ──
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

    pipe = QwenImage21Pipeline.from_pretrained(pack, torch_dtype=torch.float32)
    fx = {}
    drop_idx = pipe._drop_idx
    system_prefix = f"<|im_start|>system\n{pipe.sys_prompt}<|im_end|>\n"
    prefix_ids = pipe.processor.tokenizer(system_prefix, add_special_tokens=False)["input_ids"]
    assert drop_idx == len(prefix_ids), (drop_idx, len(prefix_ids))
    fx["edit_drop_idx"] = i32(torch.tensor([drop_idx]))

    # ── reference sources + the pipeline's one-resize preprocessing ──
    rng = np.random.default_rng(EDIT_SEED_SRC)
    sources = []
    for h, w in EDIT_SRC_SHAPES:
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        alpha = rng.integers(0, 256, (h, w), dtype=np.uint8)
        sources.append(PILImage.fromarray(np.dstack([rgb, alpha]), "RGBA"))
    fx["edit_source_rgba"] = mx.array(np.asarray(sources[0], dtype=np.float32))
    fx["edit_source_rgba_2"] = mx.array(np.asarray(sources[1], dtype=np.float32))

    input_images, vae_images, input_image_sizes = [], [], []
    res2 = EDIT_OUTPUT_RESOLUTION * EDIT_OUTPUT_RESOLUTION
    for img in sources:
        input_width, input_height, _ = calculate_dimensions(res2, img.size[0] / img.size[1])
        input_image_sizes.append((input_width, input_height))
        input_images.append(pipe.image_processor.resize(img, width=input_width, height=input_height))
        vae_images.append(pipe.image_processor.preprocess(img, width=input_width, height=input_height).unsqueeze(2))

    # The VLM copy, composited over white (pipeline_qwenimage21.py verbatim).
    composited = []
    for img in input_images:
        white = PILImage.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=img.getchannel("A"))
        composited.append(white)
    fx["edit_resized_rgb"] = mx.array(np.asarray(composited[0], dtype=np.float32))
    fx["edit_vae_input"] = f32(vae_images[0][:, :, 0])

    # ── prompt embeds: the pipeline's own path, processor spied ──
    pipe.processor = SpyProcessor(pipe.processor)
    emb, emb_mask, pad = pipe._get_qwen_prompt_embeds([EDIT_PROMPT], image=input_images)
    mi = pipe.processor.calls[-1][1]
    fx["edit_input_ids"] = i32(mi.input_ids)
    fx["edit_attention_mask"] = i32(mi.attention_mask)
    fx["edit_grids"] = i32(mi.image_grid_thw)
    fx["edit_pixel_values"] = f32(mi.pixel_values)
    for seen, mine in zip(pipe.processor.calls[0][0]["images"], composited):
        assert np.array_equal(np.asarray(seen), np.asarray(mine)), "composite drifted from the pipeline's"

    neg_emb, _, _ = pipe._get_qwen_prompt_embeds([EDIT_NEG_PROMPT], image=input_images)
    mi_neg = pipe.processor.calls[-1][1]
    fx["edit_neg_input_ids"] = i32(mi_neg.input_ids)
    fx["edit_neg_attention_mask"] = i32(mi_neg.attention_mask)
    fx["edit_hidden"] = f32(emb)
    fx["edit_neg_hidden"] = f32(neg_emb)
    fx["edit_pad_mask"] = i32(pad)

    # ── tower outputs via the model's own get_image_features ──
    feats = pipe.text_encoder.model.get_image_features(pixel_values=mi.pixel_values, image_grid_thw=mi.image_grid_thw)
    fx["edit_vit_merged"] = f32(torch.cat(feats.pooler_output, dim=0))
    for i, deep in enumerate(feats.deepstack_features):
        fx[f"edit_vit_deepstack_{i}"] = f32(deep)

    # ── joint construction exactly as the transformer forward does ──
    target_width, target_height, _ = calculate_dimensions(res2, sources[-1].size[0] / sources[-1].size[1])
    multiple_of = pipe.vae_scale_factor * 2
    height = target_height // multiple_of * multiple_of
    width = target_width // multiple_of * multiple_of
    img_shapes = [(1, vae_h // pipe.vae_scale_factor, vae_w // pipe.vae_scale_factor)
                  for vae_w, vae_h in input_image_sizes]
    img_shapes.append((1, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor))
    target_tokens = img_shapes[-1][1] * img_shapes[-1][2]
    img_mask = torch.cat([pad[0], torch.ones(target_tokens // 4, dtype=torch.bool)])[None]
    repeats = torch.where(img_mask, 4, 1)[0]
    expanded = torch.repeat_interleave(img_mask[0], repeats)
    fx["edit_img_shapes"] = i32(torch.tensor(img_shapes))
    image_ids, target_token_mask = pipe.transformer.build_token_metadata(expanded, list(img_shapes))
    fx["edit_image_ids"] = i32(image_ids)
    fx["edit_target_token_mask"] = i32(target_token_mask)
    rotary = pipe.transformer.pos_embed(list(img_shapes), expanded, device=torch.device("cpu"))
    fx["edit_rope_cos"] = f32(rotary.real)
    fx["edit_rope_sin"] = f32(rotary.imag)

    # The real checkpoint's rope layout as a pure QwenImage21Rope table (the
    # pack DiT cannot run it: sum(axes) must equal its head dim 16).
    rope_real = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])(list(img_shapes), expanded,
                                                                    device=torch.device("cpu"))
    fx["edit_rope_cos_real"] = f32(rope_real.real)
    fx["edit_rope_sin_real"] = f32(rope_real.imag)

    # ── ONE edit DiT forward on the pack's weights and config ──
    # encode_prompt None-s an all-valid mask before the transformer sees it.
    if emb_mask is not None and emb_mask.all():
        emb_mask = None
    gen = torch.Generator().manual_seed(EDIT_SEED_LAT)
    ref_tokens = sum(s[1] * s[2] for s in img_shapes[:-1])
    latents = torch.randn(1, ref_tokens + target_tokens, pipe.transformer.config.in_channels, generator=gen)
    fx["edit_latents"] = f32(latents)
    timestep = torch.tensor([EDIT_TIMESTEP]) / 1000
    out = pipe.transformer(hidden_states=latents, encoder_hidden_states=emb, timestep=timestep,
                           img_shapes=[list(img_shapes)], img_mask=img_mask,
                           encoder_hidden_states_mask=emb_mask, attention_kwargs={},
                           kv_cache=None, kv_cache_mode=None, return_dict=False)[0]
    fx["edit_dit_t"] = mx.array(np.array([EDIT_TIMESTEP], dtype=np.float32))
    fx["edit_dit_out"] = f32(out[:, -target_tokens:])

    # ── mflux VAE encode with a real (non-constant) alpha channel ──
    vr = np.random.default_rng(EDIT_SEED_VAE)
    rgb = vr.uniform(-1, 1, (1, 3, 32, 48)).astype(np.float32)
    alpha = vr.uniform(0.05, 1.0, (1, 1, 32, 48)).astype(np.float32)
    rgba = mx.concatenate([mx.array(rgb), mx.array(alpha)], axis=1)
    fx["edit_vae_image"] = rgba
    fx["edit_vae_encoded"] = (vae["quant"](vae["enc"](rgba))[:, :vae["z"]] - vae["m4"]) / vae["s4"]

    # ── pack-notower: same pack minus model.visual.* + vision_config ──
    notower = pack + "-notower"
    shutil.copytree(pack, notower, dirs_exist_ok=True)
    lm_only = {k: v for k, v in load_file(os.path.join(te_dir, "model.safetensors")).items()
               if not k.startswith("model.visual.")}
    save_file(lm_only, os.path.join(notower, "text_encoder", "model.safetensors"))
    cfg_no_vis = json.load(open(te_cfg_path))
    del cfg_no_vis["vision_config"]
    json.dump(cfg_no_vis, open(os.path.join(notower, "text_encoder", "config.json"), "w"), indent=2)
    with safe_open(os.path.join(te_dir, "model.safetensors"), framework="pt") as f:
        assert any(k.startswith("model.visual.") for k in f.keys()), "pack lost the tower"
    with safe_open(os.path.join(notower, "text_encoder", "model.safetensors"), framework="pt") as f:
        assert not any(k.startswith("model.visual.") for k in f.keys()), "pack-notower kept the tower"

    # ── pack-flatdit: same pack, DiT keys spelled the mlx-community way
    # (time embedder flattened, modulation.0) — the loader's probe bar. ──
    flat = pack + "-flatdit"
    shutil.copytree(pack, flat, dirs_exist_ok=True)
    dit_src = os.path.join(pack, "transformer")
    flat_dit = {}
    for shard in sorted(glob.glob(os.path.join(dit_src, "*.safetensors"))):
        for k, v in load_file(shard).items():
            nk = (k.replace("time_text_embed.timestep_embedder.", "time_text_embed.")
                   .replace("modulation.1", "modulation.0"))
            flat_dit[nk] = v
    save_file(flat_dit, os.path.join(flat, "transformer", "model.safetensors"))
    for shard in sorted(glob.glob(os.path.join(dit_src, "*.safetensors"))):
        os.remove(os.path.join(flat, "transformer", os.path.basename(shard)))
    with safe_open(os.path.join(flat, "transformer", "model.safetensors"), framework="pt") as f:
        assert "time_text_embed.linear_1.weight" in f.keys(), "flat DiT rename missing t1"
        assert "modulation.0.weight" in f.keys(), "flat DiT rename missing modulation"

    # ── pack-mcspell: the mlx-community spelling — `vision_tower.*` tower keys,
    # `language_model.model.*` LM keys, embed + pos_embed tables 4-bit gs64
    # (the loader probes the spelling and dequantizes the gather tables) ──
    mc = pack + "-mcspell"
    shutil.copytree(pack, mc, dirs_exist_ok=True)
    renamed = {}
    for k, v in mx.load(os.path.join(te_dir, "model.safetensors")).items():
        if k.startswith("model.visual."):
            renamed["vision_tower." + k[len("model.visual."):]] = v
        elif k.startswith("model.language_model."):
            renamed["language_model.model." + k[len("model.language_model."):]] = v
        else:
            renamed[k] = v
    # gs64 packing needs a /64 hidden; the tiny fixture's LM hidden is 24, so
    # the embed table pads to the next multiple of 64 (loader fixture — the
    # real packs' 3584 qualifies natively; LM layer widths never enter the load).
    emb = renamed["language_model.model.embed_tokens.weight"]
    pad = (-emb.shape[1]) % 64
    padded = emb if pad == 0 else mx.concatenate(
        [emb, mx.zeros((emb.shape[0], pad), dtype=emb.dtype)], axis=1)
    qw, qs, qb = mx.quantize(padded, group_size=64, bits=4)
    renamed["language_model.model.embed_tokens.weight"] = qw
    renamed["language_model.model.embed_tokens.scales"] = qs
    renamed["language_model.model.embed_tokens.biases"] = qb
    pos = renamed["vision_tower.pos_embed.weight"]
    pw, ps, pb = mx.quantize(pos, group_size=64, bits=4)
    renamed["vision_tower.pos_embed.weight"] = pw
    renamed["vision_tower.pos_embed.scales"] = ps
    renamed["vision_tower.pos_embed.biases"] = pb
    # mx.eval BEFORE save: quantize graphs evaluated lazily at save time
    # wrote triples that do not round-trip.
    mx.eval(qw, qs, qb, pw, ps, pb)
    mx.save_safetensors(os.path.join(mc, "text_encoder", "model.safetensors"), renamed)
    check = mx.load(os.path.join(mc, "text_encoder", "model.safetensors"))
    assert "vision_tower.patch_embed.proj.weight" in check, "pack-mcspell lost the tower"
    assert "language_model.model.embed_tokens.scales" in check, "pack-mcspell embed left dense"
    assert "vision_tower.pos_embed.scales" in check, "pack-mcspell pos_embed left dense"
    return fx


def check_tensors(fx):
    for k, v in fx.items():
        a = np.asarray(v)
        if v.dtype == mx.float32:
            assert np.isfinite(a).all(), f"{k} has non-finite values"
        assert (a != 0).any(), f"{k} is all zero"


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

    # ── edit half: tiny Qwen3VL pack + reference-pipeline fixtures (rewrites
    # the model_index above with the full component map the pipeline loads) ──
    fx.update(dump_edit_fixtures(pack, dict(enc=enc, quant=quant, m4=m4, s4=s4, z=z)))

    mx.eval(list(fx.values()))
    check_tensors(fx)
    mx.save_safetensors(os.path.join(out_dir, "qwen_image21_fixture.safetensors"), fx)
    for k, v in fx.items():
        print(k, v.shape, v.dtype)


if __name__ == "__main__":
    main()
