#!/usr/bin/env python3
"""Dump a MageFlow edit text-encoder parity fixture for src/mage_flow.zig (E7.4).

Runs the pure-MLX REFERENCE (ivanfioravanti/mflux@mage-flow-mlx) full Qwen3-VL
text encoder edit path (vision-embed replace + DeepStack scatter-add at LM layers
0/1/2, sequential position_ids, drop-64) in FP32 over a fixed prompt + one
reference image, and writes one safetensors the Zig oracle loads
(env MAGEFLOW_TE_EDIT_FIXTURE). Feeding the reference input_ids + pixel_values
decouples the LM path from tokenization/preprocessing (E7.3). Tensors:
    input_ids      [1, L]  int32
    attention_mask [1, L]  int32
    pixel_values   [Npatch, 1536] f32
    image_grid_thw [1, 3]  int32
    embeddings     [1, L-64, 2560] f32 — the DiT conditioning
    out_mask       [1, L-64] int32
    source_rgb     [H, W, 3] f32 (0..255) — the processor's INPUT image, so the
                   Zig oracle can re-derive pixel_values itself (E7.3 parity)
    input_ids_2img [1, L2] int32 — the same prompt templated for TWO reference
                   images, pinning the multi-reference placeholder layout

USER-RUN (accepts the MIT Turbo license, has the checkpoint downloaded):
    <mflux>/.venv/bin/python tests/dump_mageflow_te_edit_fixture.py \
        ~/.mlx-serve/models/microsoft/Mage-Flow-Turbo <mflux_repo_root> [OUT]

Then:
    MAGEFLOW_TE_EDIT_FIXTURE=<OUT>/mageflow_te_edit.safetensors \
    zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow TE edit"
"""

import glob
import os
import sys

import mlx.core as mx
import numpy as np
from PIL import Image

PROMPT = "make it a snowy winter scene at golden hour"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    model_dir = os.path.abspath(sys.argv[1])
    ref_root = os.path.abspath(sys.argv[2])
    out_dir = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.abspath("mageflow_fixtures")
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, os.path.join(ref_root, "src"))

    from mlx.utils import tree_unflatten
    from transformers import AutoTokenizer

    from mflux.models.mage_flow.model.mage_flow_text_encoder.processor import MageFlowQwen3VLProcessor
    from mflux.models.mage_flow.model.mage_flow_text_encoder.prompt_processor import MageFlowPromptProcessor
    from mflux.models.mage_flow.model.mage_flow_text_encoder.text_encoder import MageFlowTextEncoder
    from mflux.models.mage_flow.weights.mage_flow_weight_mapping import MageFlowWeightMapping

    te_dir = os.path.join(model_dir, "text_encoder")

    vision_config = dict(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        intermediate_size=4096,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=2304,
        deepstack_visual_indexes=(5, 11, 17),
    )
    te = MageFlowTextEncoder(
        vocab_size=151936,
        hidden_size=2560,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=9728,
        rope_theta=5_000_000.0,
        rms_norm_eps=1e-6,
        head_dim=128,
        mrope_section=(24, 20, 20),
        image_token_id=151655,
        vision_start_token_id=151652,
        vision_config=vision_config,
    )

    flat = {}
    for shard in sorted(glob.glob(os.path.join(te_dir, "*.safetensors"))):
        for k, v in mx.load(shard).items():
            mapped = MageFlowWeightMapping.transform_text_encoder_key(k)
            if mapped is None:
                continue
            v = MageFlowWeightMapping.transform_text_encoder_weight(mapped, v)
            flat[mapped] = v.astype(mx.float32)
    te.update(tree_unflatten(list(flat.items())))
    mx.eval(te.parameters())
    print(f"[dump] loaded {len(flat)} text-encoder tensors (fp32)")

    # ── Reference processor: build the templated edit inputs for one image ──
    proc = MageFlowQwen3VLProcessor(tokenizer=AutoTokenizer.from_pretrained(te_dir))
    rng = np.random.default_rng(19)
    img = Image.fromarray(rng.integers(0, 255, (256, 256, 3)).astype(np.uint8))
    formatted = MageFlowPromptProcessor.format_edit(PROMPT, num_images=1)
    inputs = proc(text=[formatted], images=[img], padding=True, truncation=True, max_length=2048 + 64)
    L = int(inputs["input_ids"].shape[1])
    position_ids = mx.broadcast_to(mx.arange(L, dtype=mx.int32)[None, :], (1, L))
    print(f"[dump] L={L}; image tokens={int((np.asarray(inputs['input_ids'])==151655).sum())}")

    # Multi-reference templating: the SAME processor path with two images, ids
    # only. Composition ("put the cat from image 1 on the sofa from image 2") is
    # half the edit feature, and its per-image header/placeholder layout is pure
    # extrapolation from the 1-image case unless something pins it.
    formatted2 = MageFlowPromptProcessor.format_edit(PROMPT, num_images=2)
    inputs2 = proc(text=[formatted2], images=[img, img], padding=True, truncation=True, max_length=2048 + 64)
    print(f"[dump] 2-image L={int(inputs2['input_ids'].shape[1])}")

    hidden = te(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        position_ids=position_ids,
    )
    embeddings, out_mask = MageFlowPromptProcessor.process_edit_hidden_states(hidden, inputs["attention_mask"])
    mx.eval(hidden, embeddings, out_mask)

    fixture = {
        "input_ids": inputs["input_ids"].astype(mx.int32),
        "attention_mask": inputs["attention_mask"].astype(mx.int32),
        "pixel_values": inputs["pixel_values"].astype(mx.float32),
        "image_grid_thw": inputs["image_grid_thw"].astype(mx.int32),
        "embeddings": embeddings.astype(mx.float32),
        "out_mask": out_mask.astype(mx.int32),
        # The processor's INPUT pixels, so the Zig side can re-derive
        # pixel_values from raw RGB and prove its own preprocessing (E7.3)
        # matches — the one link the other fixtures deliberately decouple.
        "source_rgb": mx.array(np.asarray(img, dtype=np.float32)),
        "input_ids_2img": inputs2["input_ids"].astype(mx.int32),
    }
    out_path = os.path.join(out_dir, "mageflow_te_edit.safetensors")
    mx.save_safetensors(out_path, fixture)
    print(f"[dump] embeddings={embeddings.shape} → {out_path}")
    print("\nRun the Zig oracle:")
    print(f"  MAGEFLOW_TE_EDIT_FIXTURE={out_path} \\")
    print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow TE edit"')


if __name__ == "__main__":
    main()
