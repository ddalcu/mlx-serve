#!/usr/bin/env python3
"""Dump DFlash2 reference oracles from the REAL incoai/Qwen3.8-27B-DFlash2.

Reference = z-lab/dflash `dflash/model_mlx.py` (the released MLX
implementation with full DFlash2 support: grouped dynamic causal convs +
candidate selector), driven directly on its own classes — bf16, the dtype the
checkpoint ships and the engine serves.

The trunk is never loaded: the block forward is driven through the layer
stack with a random noise-embeds block (the same shape rawEmbedding hands
the engine), and the selector is driven with SPARSE synthetic logits (base
-10.0 + 64 distinct-valued ids per position) plus the reference's own block
hidden — both sides materialize the identical [m, V] array, so the traced
path ids must match EXACTLY while the hidden parity stays cos + rms_ratio.

Run:
    uv run --with mlx --with mlx-lm --with numpy \
        python3 tests/dump_dflash2_fixtures.py \
        [--assistant-dir ~/.mlx-serve/models/incoai/Qwen3.8-27B-DFlash2] \
        [--dflash-repo ~/claude-tmp/dflash2/dflash] \
        [--out ~/claude-tmp/dflash2_fixtures.json]

Then:
    DFLASH2_FIXTURES=~/claude-tmp/dflash2_fixtures.json \
    DFLASH2_ASSISTANT_DIR=~/.mlx-serve/models/incoai/Qwen3.8-27B-DFlash2 \
        zig build test -Doptimize=ReleaseFast -Dtest-filter="dflash2 fixture"
"""

import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assistant-dir", default=os.path.expanduser("~/.mlx-serve/models/incoai/Qwen3.8-27B-DFlash2"))
    ap.add_argument("--dflash-repo", default=os.path.expanduser("~/claude-tmp/dflash2/dflash"))
    ap.add_argument("--out", default=os.path.expanduser("~/claude-tmp/dflash2_fixtures.json"))
    ap.add_argument("--n-ctx", type=int, default=12)
    args = ap.parse_args()

    sys.path.insert(0, args.dflash_repo)
    import mlx.core as mx
    from dflash.model_mlx import DFlash2DraftModel, DFlashConfig

    with open(os.path.join(args.assistant_dir, "config.json")) as f:
        cfg = json.load(f)
    dc = cfg["dflash_config"]
    rope = cfg.get("rope_parameters") or cfg.get("rope_scaling")
    config = DFlashConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=(rope or {}).get("rope_theta", 10000.0),
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=dc["block_size"],
        target_layer_ids=tuple(dc["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=dc["mask_token_id"],
        rope_scaling=rope,
        layer_types=tuple(cfg["layer_types"]),
        sliding_window=cfg.get("sliding_window"),
        conv_kernel_size=dc["conv_kernel_size"],
        conv_group_size=dc["conv_group_size"],
        selector_rank=dc["selector_rank"],
        selector_top_k=dc["selector_top_k"],
        is_causal=cfg.get("is_causal"),
    )
    weights = dict(mx.load(os.path.join(args.assistant_dir, "model.safetensors")))
    for name in ("predecessor_codebook", "successor_codebook"):
        key = f"candidate_selector.{name}"
        weights[f"{key}.weight"] = weights.pop(key)
    draft = DFlash2DraftModel(config)
    draft.eval()
    draft.load_weights(list(weights.items()))
    mx.eval(draft.parameters())

    h = config.hidden_size
    nt = len(config.target_layer_ids)
    bs = config.block_size
    n_ctx = args.n_ctx
    rng = np.random.default_rng(20260818)

    # Inputs travel IN the fixture (bf16-rounded so both sides read the same
    # values after the astype).
    ctx_stream = rng.standard_normal((1, n_ctx, nt * h)).astype(np.float32) * 0.5
    noise1 = rng.standard_normal((1, bs, h)).astype(np.float32) * 0.5
    ctx_bf = mx.array(ctx_stream).astype(mx.bfloat16)
    noise_bf = mx.array(noise1).astype(mx.bfloat16)

    # ── Block forward through the reference layer stack (no embed table) ──
    cache = draft.make_cache()
    h_ctx = draft.hidden_norm(draft.fc(ctx_bf))
    x = noise_bf
    for layer, c in zip(draft.layers, cache):
        x = layer(x, h_ctx, draft.rope, c)
    hidden = draft.norm(x)  # [1, bs, h]
    mx.eval(hidden)

    # ── Selector on sparse synthetic logits + the reference's OWN hidden ──
    m = bs - 1
    v = config.vocab_size
    n_active = 64
    active_ids = np.stack([
        rng.choice(v, size=n_active, replace=False) for _ in range(m)
    ]).astype(np.int64)
    # Distinct values well above the -10 floor → the top-16 set is exact.
    active_vals = (rng.permuted(np.tile(np.arange(n_active, dtype=np.float32), (m, 1)), axis=1) * 0.25
                   + rng.standard_normal((m, n_active)).astype(np.float32) * 1e-3)
    logits_np = np.full((1, m, v), -10.0, dtype=np.float32)
    for t in range(m):
        logits_np[0, t, active_ids[t]] = active_vals[t]
    logits = mx.array(logits_np)
    anchor_id = 4242
    # Hidden stays bf16 (its serving dtype) — the Zig side rebuilds it bf16
    # from this fixture, so both selector inputs are bit-identical.
    path, cands, _q = draft.candidate_selector.select(
        hidden[:, 1:], logits, mx.array([anchor_id]), temperature=0.0
    )
    mx.eval(path, cands)

    out = {
        "hidden_size": h,
        "n_targets": nt,
        "block_size": bs,
        "n_ctx": n_ctx,
        "anchor_id": anchor_id,
        "top_k": config.selector_top_k,
        "ctx_stream": ctx_stream.reshape(-1).tolist(),
        "noise1": noise1.reshape(-1).tolist(),
        "round1_hidden": np.array(hidden.astype(mx.float32)).reshape(-1).tolist(),
        "active_ids": active_ids.reshape(-1).tolist(),
        "active_vals": active_vals.reshape(-1).tolist(),
        "n_active": n_active,
        "vocab_size": v,
        "path_ids": np.array(path).reshape(-1).tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}: block hidden [1,{bs},{h}] + greedy path {out['path_ids']}")


if __name__ == "__main__":
    main()
