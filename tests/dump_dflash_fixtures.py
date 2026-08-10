#!/usr/bin/env python3
"""Dump DFlash reference oracles from the REAL Muse-Glimmer-30B assistant.

Reference = transformers' `MuseGlimmerAssistantModel` (released in 5.15.0)
plus `DFlashCache` — the executable upstream implementation, run directly on
CPU fp32 (MPS fp16 quietly decorrelates; house rule). No stubs needed: the
model class ships in the released package.

Dumps, all fp32 with seeded random inputs carried IN the fixture file:
  * encoder    — context projection (`encoder.fc` → RMS norm) over a random
                 [1, n_ctx, n_targets*hidden] context stream.
  * round1     — one full draft-block forward: context K/V populated from the
                 stream above, noise block [1, block_size, hidden] at
                 positions [n_ctx, n_ctx + block_size).
  * round2     — a SECOND round after `crop(-block_size)` + a k-token context
                 delta, exactly the reference candidate-generator protocol:
                 pins absolute-position RoPE and the never-cache-block-K/V
                 contract across rounds.

Run (the assistant is ~5.11 GB; loading fp32 needs ~11 GB RAM):
    uv run --with torch --with transformers --with numpy --with safetensors \
        python3 tests/dump_dflash_fixtures.py \
        [--assistant-dir <dir with config.json + model.safetensors>] \
        [--out ~/claude-tmp/dflash_fixtures.json]

Then:
    DFLASH_FIXTURES=~/claude-tmp/dflash_fixtures.json \
    DFLASH_ASSISTANT_DIR=<same dir> \
        zig build test -Doptimize=ReleaseFast -Dtest-filter="dflash fixture"
"""

import argparse
import glob
import json
import os
import sys


DEFAULT_DIR_GLOBS = [
    "~/.mlx-serve/models/meta-models/Muse-Glimmer-30B-assistant",
    "~/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-assistant/snapshots/*",
]


def find_assistant_dir(arg):
    if arg:
        return os.path.expanduser(arg)
    for pattern in DEFAULT_DIR_GLOBS:
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        for h in reversed(hits):
            if os.path.exists(os.path.join(h, "config.json")):
                return h
    sys.exit("assistant checkpoint not found; pass --assistant-dir")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assistant-dir")
    ap.add_argument("--out", default=os.path.expanduser("~/claude-tmp/dflash_fixtures.json"))
    ap.add_argument("--n-ctx", type=int, default=12)
    ap.add_argument("--n-delta", type=int, default=5)
    args = ap.parse_args()

    import torch

    from transformers import AutoConfig
    from transformers.cache_utils import DFlashCache
    from transformers.models.muse_glimmer_assistant import MuseGlimmerAssistantModel

    torch.manual_seed(7)
    adir = find_assistant_dir(args.assistant_dir)
    print(f"assistant: {adir}")

    config = AutoConfig.from_pretrained(adir)
    model = MuseGlimmerAssistantModel.from_pretrained(adir, dtype=torch.float32)
    model.eval()

    H = config.hidden_size
    NT = len(config.target_layer_ids)
    BS = config.block_size
    n_ctx, n_delta = args.n_ctx, args.n_delta

    # Residual-stream-scale random inputs, dumped alongside the outputs so
    # the Zig side replays the exact bytes (no cross-language RNG contract).
    ctx_stream = torch.randn(1, n_ctx, NT * H)
    ctx_delta = torch.randn(1, n_delta, NT * H)
    noise1 = torch.randn(1, BS, H)
    noise2 = torch.randn(1, BS, H)

    out = {
        "hidden_size": H,
        "n_targets": NT,
        "block_size": BS,
        "n_ctx": n_ctx,
        "n_delta": n_delta,
        "ctx_stream": ctx_stream.flatten().tolist(),
        "ctx_delta": ctx_delta.flatten().tolist(),
        "noise1": noise1.flatten().tolist(),
        "noise2": noise2.flatten().tolist(),
    }

    with torch.no_grad():
        # ── encoder projection oracle ──
        enc = model.encoder(ctx_stream)
        out["encoder_out"] = enc.flatten().tolist()

        # ── round 1: ctx [0, n_ctx), block at [n_ctx, n_ctx+BS) ──
        cache = DFlashCache(config=config)
        cache.set_previous_accepted_tokens(n_ctx)
        pos1 = torch.arange(n_ctx + BS).unsqueeze(0)
        mask1 = torch.ones(1, n_ctx + BS, dtype=torch.long)
        r1 = model(
            noise_embeds=noise1,
            context_hidden_states=ctx_stream,
            position_ids=pos1,
            attention_mask=mask1,
            past_key_values=cache,
            use_cache=True,
        )
        out["round1_hidden"] = r1.last_hidden_state.flatten().tolist()

        # ── round 2: evict the block, append an n_delta context, new anchor ──
        cache.crop(-BS)
        cache.set_previous_accepted_tokens(n_delta)
        pos2 = torch.arange(n_ctx, n_ctx + n_delta + BS).unsqueeze(0)
        mask2 = torch.ones(1, n_ctx + n_delta + BS, dtype=torch.long)
        r2 = model(
            noise_embeds=noise2,
            context_hidden_states=ctx_delta,
            position_ids=pos2,
            attention_mask=mask2,
            past_key_values=cache,
            use_cache=True,
        )
        out["round2_hidden"] = r2.last_hidden_state.flatten().tolist()

    os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    with open(os.path.expanduser(args.out), "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
