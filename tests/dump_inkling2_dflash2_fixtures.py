#!/usr/bin/env python3
"""Dump DFlash2 reference oracles for Inkling 2 Mini's draft (`incoai/Inkling-2-Mini-DFlash2`).

Reference = z-lab/dflash `dflash/model_mlx.py`, driven on its own classes on the REAL draft weights, bf16. The trunk
is never loaded: the block forward takes a random block of embedding rows and random tap rows, and the selector is
driven with sparse synthetic logits. The selector applies the softcap AFTER the correction
(`selector_softcap_mode: after_correction`, as the sglang adapter shipped beside the weights does): z-lab's `select`
predates the field, so it is repeated here.

Run (DRAFT = the draft's folder, REPO = a clone of https://github.com/z-lab/dflash):
    uv run --with mlx --with mlx-lm --with numpy --with huggingface_hub \
        python3 tests/dump_inkling2_dflash2_fixtures.py --draft-dir DRAFT --dflash-repo REPO --out FILE
Then:
    INKLING2_DFLASH2_DIR=DRAFT INKLING2_DFLASH2_FIXTURES=FILE zig build test -Dtest-filter="inkling2 dflash2"
"""

import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-dir", required=True, help="the draft's folder (config.json, model.safetensors)")
    ap.add_argument("--dflash-repo", required=True, help="a clone of z-lab/dflash")
    ap.add_argument("--out", required=True, help="the fixture file to write (.safetensors)")
    args = ap.parse_args()

    sys.path.insert(0, args.dflash_repo)
    import mlx.core as mx
    import dflash.model_mlx as M

    M.snapshot_download = lambda path, **kw: path  # the draft is a local directory
    model = M.load_draft(args.draft_dir)
    cfg = model.config
    hidden, block, top_k = cfg.hidden_size, cfg.block_size, cfg.selector_top_k
    cap = float(cfg.final_logit_softcapping)
    valid_vocab = 200058
    rng = np.random.default_rng(20261124)
    out, meta = {}, {"block": block, "top_k": top_k, "softcap": cap, "hidden": hidden, "valid_vocab": valid_vocab, "cases": []}

    def bf16(x):
        return mx.array(x.astype(np.float32)).astype(mx.bfloat16)

    # The block forward, as `DFlashDraftModel.hidden_states` after the embedding: a context of S rows whose first is at
    # absolute position `first`, so the anchor sits at `first + S` (rows past the window are dropped by the attention).
    for name, ctx_rows, first in (("short", 40, 100), ("long", 2100, 5000)):
        noise = bf16(rng.normal(0, 1.0, (1, block, hidden)))
        taps = bf16(rng.normal(0, 1.0, (1, ctx_rows, len(cfg.target_layer_ids) * hidden)))
        cache = model.make_cache()
        for c in cache:
            c.offset = first
        h = noise
        h_ctx = model.hidden_norm(model.fc(taps))
        for layer, c in zip(model.layers, cache):
            h = layer(h, h_ctx, model.rope, c)
        h = model.norm(h)
        mx.eval(h)
        out[f"{name}.noise"], out[f"{name}.taps"], out[f"{name}.hidden"] = noise, taps, h
        meta["cases"].append({"name": name, "ctx_rows": ctx_rows, "first": first})

    # The selector on the short case's block rows (rows 1..) and sparse raw logits: 24 distinct spikes a position.
    rows = out["short.hidden"][:, 1:]
    n = rows.shape[1]
    logits = np.full((1, n, valid_vocab), -10.0, np.float32)
    for pos in range(n):
        ids = rng.choice(valid_vocab, 24, replace=False)
        logits[0, pos, ids] = rng.permutation(24).astype(np.float32) * 0.4 + rng.normal(0, 0.01, 24) + 2.0
    logits = mx.array(logits)
    anchor = 4242
    sel = model.candidate_selector
    candidates = mx.argpartition(logits, -top_k, axis=-1)[..., -top_k:]
    unary = mx.take_along_axis(logits, candidates, axis=-1)
    proj = sel.hidden_projection(rows)
    predecessor = mx.array([anchor])
    path, all_scores = [], []
    for pos in range(n):
        edges = mx.sum(sel.predecessor_codebook(predecessor)[:, None] * proj[:, pos, None] * sel.successor_codebook(candidates[:, pos]), axis=-1)
        scores = cap * mx.tanh((unary[:, pos] + edges) / cap)
        all_scores.append(scores)
        predecessor = mx.take_along_axis(candidates[:, pos], mx.argmax(scores, axis=-1)[:, None], axis=-1)[:, 0]
        path.append(predecessor)
    scores = mx.stack(all_scores, axis=1)
    mx.eval(scores, candidates)
    out["sel.logits_ids"] = mx.argpartition(logits, -24, axis=-1)[..., -24:].astype(mx.int32)
    out["sel.logits_vals"] = mx.take_along_axis(logits, out["sel.logits_ids"], axis=-1)
    out["sel.rows"] = rows
    out["sel.candidates"] = mx.sort(candidates, axis=-1).astype(mx.int32)
    out["sel.scores_sorted"] = mx.take_along_axis(scores, mx.argsort(candidates, axis=-1), axis=-1)
    out["sel.path"] = mx.stack(path, axis=1).astype(mx.int32)
    out["sel.q1"] = mx.softmax(out["sel.scores_sorted"], axis=-1)
    meta["anchor"] = anchor
    mx.eval(*out.values())
    mx.save_safetensors(args.out, out, metadata={"meta": json.dumps(meta)})
    print("wrote", args.out, {k: tuple(v.shape) for k, v in out.items()})
    print("path", out["sel.path"].tolist())


if __name__ == "__main__":
    main()
