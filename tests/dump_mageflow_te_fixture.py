#!/usr/bin/env python3
"""Dump a MageFlow Qwen3-VL text-encoder parity fixture for src/mage_flow.zig.

Runs the pure-MLX REFERENCE (ivanfioravanti/mflux@mage-flow-mlx) text encoder in
FP32 over a fixed prompt and writes a single safetensors the Zig oracle test
loads (env MAGEFLOW_TE_FIXTURE). Tensors:
    input_ids       [1, L]  int32  — the templated + tokenized prompt
    attention_mask  [1, L]  int32  — all ones for a single non-padded prompt
    hidden_full     [1, L, 2560]   — final-norm hidden states (pre-trim)
    embeddings      [1, L-34, 2560]— after drop-34 + right-pad (batch max)
    out_mask        [1, L-34] int32 — the trimmed attention mask

USER-RUN (accepts the MIT Turbo license, has the checkpoint downloaded):
    <mflux>/.venv/bin/python tests/dump_mageflow_te_fixture.py \
        ~/.mlx-serve/models/microsoft/Mage-Flow-Turbo <mflux_repo_root> [OUT]

Then:
    MAGEFLOW_TEST_MODEL=~/.mlx-serve/models/microsoft/Mage-Flow-Turbo \
    MAGEFLOW_TE_FIXTURE=<OUT>/mageflow_te.safetensors \
    zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow TE"
"""

import glob
import os
import sys

import mlx.core as mx
import numpy as np

PROMPT = "a red fox sitting in the snow, photorealistic, golden hour lighting"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    model_dir = os.path.abspath(sys.argv[1])
    ref_root = os.path.abspath(sys.argv[2])
    out_dir = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.abspath("mageflow_fixtures")
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, os.path.join(ref_root, "src"))

    from transformers import AutoTokenizer

    from mflux.models.mage_flow.model.mage_flow_text_encoder.prompt_processor import MageFlowPromptProcessor
    from mflux.models.mage_flow.model.mage_flow_text_encoder.text_encoder import (
        MageFlowQwen3VLLanguageModel,
        build_mrope_position_ids,
    )

    te_dir = os.path.join(model_dir, "text_encoder")

    # ── Build the LM backbone and load only the language_model.* weights (fp32) ──
    lm = MageFlowQwen3VLLanguageModel(
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
    )

    flat = {}
    for shard in sorted(glob.glob(os.path.join(te_dir, "*.safetensors"))):
        for k, v in mx.load(shard).items():
            if k in ("lm_head.weight", "model.visual.rotary_pos_emb.inv_freq"):
                continue
            if not k.startswith("model.language_model."):
                continue  # skip visual.* for the text-only fixture
            flat[k[len("model.language_model.") :]] = v.astype(mx.float32)

    from mlx.utils import tree_unflatten

    lm.update(tree_unflatten(list(flat.items())))
    mx.eval(lm.parameters())
    print(f"[dump] loaded {len(flat)} language_model tensors (fp32)")

    # ── Tokenize the templated prompt exactly like MageFlowConditioning ──
    tok = AutoTokenizer.from_pretrained(te_dir)
    formatted = MageFlowPromptProcessor.format_text_to_image(PROMPT)
    enc = tok([formatted], padding=True, truncation=True, max_length=2048 + 34, return_tensors="np")
    input_ids = mx.array(np.asarray(enc["input_ids"]))
    attention_mask = mx.array(np.asarray(enc["attention_mask"]))
    L = int(input_ids.shape[1])
    print(f"[dump] prompt tokenized to L={L}; first ids={np.asarray(input_ids)[0, :6].tolist()}")

    # ── Text-only path: mrope positions collapse to sequential on all 3 axes ──
    position_ids, _ = build_mrope_position_ids(input_ids, None, attention_mask)
    hidden_full = lm(
        inputs_embeds=lm.embed_tokens(input_ids),
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    embeddings, out_mask = MageFlowPromptProcessor.process_text_to_image_hidden_states(
        hidden_full, attention_mask
    )
    mx.eval(hidden_full, embeddings, out_mask)

    fixture = {
        "input_ids": input_ids.astype(mx.int32),
        "attention_mask": attention_mask.astype(mx.int32),
        "hidden_full": hidden_full.astype(mx.float32),
        "embeddings": embeddings.astype(mx.float32),
        "out_mask": out_mask.astype(mx.int32),
    }
    out_path = os.path.join(out_dir, "mageflow_te.safetensors")
    mx.save_safetensors(out_path, fixture)
    print(f"[dump] hidden_full={hidden_full.shape} embeddings={embeddings.shape} → {out_path}")
    print("\nRun the Zig oracle:")
    print(f"  MAGEFLOW_TEST_MODEL={model_dir} \\")
    print(f"  MAGEFLOW_TE_FIXTURE={out_path} \\")
    print('  zig build test -Doptimize=ReleaseFast -Dtest-filter="MageFlow TE"')


if __name__ == "__main__":
    main()
