#!/usr/bin/env python3
"""Dump reference oracles for the Inkling Small (`model_type: inkling_mm_model`) port.

Reference = the checkpoint's own bundled `inkling_mlx/` package (MLX-native,
Apache-2.0, numerically validated against transformers PR #47347) — no torch
needed. Two modes:

  components (default, no checkpoint load, seconds):
      Tiny random-seeded geometry through the reference modules, fp32:
        * sconv     — ShortConvolution full-sequence AND incremental-cache
                      outputs (they must agree; Zig is checked against both).
        * rel_bias  — RelativeLogits bias for a prefill (start 0) and a decode
                      (Lq=1, start>0) case, global (extent) and sliding.
        * router    — Router top-k indices/weights + shared gammas (fp32
                      sigmoid → +bias select → logsigmoid-softmax sink chain).
        * attn      — one full Attention block I/O (global + sliding layer).
        * layer     — one sparse DecoderLayer I/O (attention + MoE + 4 sconvs).
      → JSON consumed by env-gated Zig tests (INKLING_FIXTURES).

  --greedy (loads the real ~112 GB checkpoint, minutes; NEVER run while
      mlx-serve is up — double-load OOM):
      Greedy continuations for fixed prompts → ground truth for the live
      engine equivalence gate.

Run:
    python3 tests/dump_inkling_fixtures.py \
        [--repo-dir ~/.cache/huggingface/hub/models--pipenetwork--Inkling-Small-MLX-REAP25-4bit/snapshots/<hash>] \
        [--out /tmp/inkling_fixtures.json]
    python3 tests/dump_inkling_fixtures.py --greedy [--max-new-tokens 32] \
        [--out /tmp/inkling_greedy.json]

Then:
    INKLING_FIXTURES=/tmp/inkling_fixtures.json \
        zig build test -Dtest-filter="inkling"
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

DEFAULT_REPO_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--pipenetwork--Inkling-Small-MLX-REAP25-4bit/snapshots/*"
)


def find_repo_dir(arg: str | None) -> str:
    if arg:
        return os.path.expanduser(arg)
    hits = sorted(glob.glob(DEFAULT_REPO_GLOB))
    if not hits:
        sys.exit(f"no snapshot under {DEFAULT_REPO_GLOB}; pass --repo-dir")
    return hits[-1]


def arr(x) -> list:
    import mlx.core as mx

    return np.array(x.astype(mx.float32)).tolist()


def dump_components(repo_dir: str, out_path: str) -> None:
    import mlx.core as mx

    sys.path.insert(0, repo_dir)
    from inkling_mlx.attention import Attention, RelativeLogits
    from inkling_mlx.cache import LayerCache
    from inkling_mlx.common import ShortConvolution
    from inkling_mlx.config import TextConfig
    from inkling_mlx.layers import DecoderLayer
    from inkling_mlx.moe import Router

    mx.random.seed(7)
    out: dict = {}

    # Tiny but structurally faithful: 2 layers (0 = global full-attn + sparse
    # MoE... layer types below), heads 4 / kv 2 / head_dim 16 (scale 1/16),
    # d_rel 4, extent 8 > sliding_window 4 so the two bias regimes differ.
    cfg = TextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        vocab_size=64,
        unpadded_vocab_size=60,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=16,
        sliding_window_size=4,
        d_rel=4,
        rel_extent=8,
        log_scaling_n_floor=None,  # tiny ctx: log-scaling is a no-op anyway
        rms_norm_eps=1e-6,
        sconv_kernel_size=4,
        dense_mlp_idx=0,  # both layers sparse
        dense_intermediate_size=96,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=3,
        n_shared_experts=2,
        route_scale=8.0,
        logits_mup_width_multiplier=16.0,
        local_layer_ids=[1],  # layer 0 global, layer 1 sliding
    )
    out["config"] = {
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "d_rel": cfg.d_rel,
        "rel_extent": cfg.rel_extent,
        "sliding_window_size": cfg.sliding_window_size,
        "sconv_kernel_size": cfg.sconv_kernel_size,
        "n_routed_experts": cfg.n_routed_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "n_shared_experts": cfg.n_shared_experts,
        "route_scale": cfg.route_scale,
    }

    def randn(*shape, scale=0.5):
        return (mx.random.normal(shape) * scale).astype(mx.float32)

    # ---- sconv: full-sequence vs incremental must agree; dump both ----
    C, K, S = 6, 4, 5
    sc = ShortConvolution(C, K)
    sc.weight = randn(C, K, 1)
    x = randn(1, S, C)
    full = sc(x)

    class _CacheSlot:  # ConvCache without the import
        state = None

    inc_cache = _CacheSlot()
    inc_steps = []
    for t in range(S):
        inc_steps.append(sc(x[:, t : t + 1, :], cache=inc_cache))
    inc = mx.concatenate(inc_steps, axis=1)
    out["sconv"] = {
        "weight": arr(sc.weight),
        "x": arr(x),
        "y_full": arr(full),
        "y_incremental": arr(inc),
        "final_cache_state": arr(inc_cache.state),
    }

    # ---- rel_bias: prefill (start 0) + decode (Lq=1 at start>extent) ----
    H, Lq, d_rel, extent = 4, 5, 4, 8
    rl = RelativeLogits(d_rel, extent)
    rl.proj = randn(d_rel, extent)
    rel_states = randn(1, Lq, H, d_rel)
    q_pos = mx.arange(Lq)
    kv_pos = mx.arange(Lq)
    bias_prefill = rl(rel_states, q_pos, kv_pos)
    kv_len_dec = 12  # > extent so the zero-beyond-extent region is exercised
    rel_states_dec = randn(1, 1, H, d_rel)
    bias_decode = rl(rel_states_dec, mx.arange(1) + (kv_len_dec - 1), mx.arange(kv_len_dec))
    out["rel_bias"] = {
        "proj": arr(rl.proj),
        "prefill": {"rel_states": arr(rel_states), "bias": arr(bias_prefill)},
        "decode": {
            "rel_states": arr(rel_states_dec),
            "kv_len": kv_len_dec,
            "q_pos": kv_len_dec - 1,
            "bias": arr(bias_decode),
        },
    }

    # ---- router: fp32 sigmoid+bias select, logsigmoid-softmax sink ----
    rt = Router(cfg)
    rt.weight = randn(cfg.n_routed_experts + cfg.n_shared_experts, cfg.hidden_size)
    rt.bias = randn(cfg.n_routed_experts, scale=0.05)
    rt.global_scale = mx.array([1.25], dtype=mx.float32)
    xr = randn(1, 3, cfg.hidden_size)
    tw, ti, sg = rt(xr)
    out["router"] = {
        "weight": arr(rt.weight),
        "bias": arr(rt.bias),
        "global_scale": 1.25,
        "x": arr(xr),
        "topk_weights": arr(tw),
        "topk_idx": np.array(ti).tolist(),
        "shared_gammas": arr(sg),
    }

    # ---- attention: one global-layer + one sliding-layer block, prefill ----
    def dump_attn(layer_idx: int, key: str) -> None:
        at = Attention(cfg, layer_idx)
        for name in ("wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud"):
            lin = getattr(at, name)
            lin.weight = randn(*lin.weight.shape)
        at.q_norm.weight = mx.ones(cfg.head_dim) + randn(cfg.head_dim, scale=0.1)
        at.k_norm.weight = mx.ones(cfg.head_dim) + randn(cfg.head_dim, scale=0.1)
        at.k_sconv.weight = randn(*at.k_sconv.weight.shape)
        at.v_sconv.weight = randn(*at.v_sconv.weight.shape)
        at.rel_logits_proj.proj = randn(at.rel_logits_proj.d_rel if hasattr(at.rel_logits_proj, "d_rel") else cfg.d_rel, at.rel_logits_proj.rel_extent)
        xa = randn(1, 6, cfg.hidden_size)
        ya = at(xa)
        out[key] = {
            "layer_idx": layer_idx,
            "weights": {
                "wq_du": arr(at.wq_du.weight),
                "wk_dv": arr(at.wk_dv.weight),
                "wv_dv": arr(at.wv_dv.weight),
                "wr_du": arr(at.wr_du.weight),
                "wo_ud": arr(at.wo_ud.weight),
                "q_norm": arr(at.q_norm.weight),
                "k_norm": arr(at.k_norm.weight),
                "k_sconv": arr(at.k_sconv.weight),
                "v_sconv": arr(at.v_sconv.weight),
                "rel_proj": arr(at.rel_logits_proj.proj),
            },
            "x": arr(xa),
            "y": arr(ya),
        }

    dump_attn(0, "attn_global")
    dump_attn(1, "attn_sliding")

    # ---- one sparse DecoderLayer, prefill + one cached decode step ----
    mx.random.seed(11)
    dl = DecoderLayer(cfg, 0)
    params = dl.parameters()

    def randomize(tree):
        if isinstance(tree, dict):
            return {k: randomize(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [randomize(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.5).astype(mx.float32)

    dl.update(randomize(params))
    xl = randn(1, 6, cfg.hidden_size)
    cache = LayerCache()
    yl = dl(xl, start_pos=0, cache=cache)
    x_step = randn(1, 1, cfg.hidden_size)
    y_step = dl(x_step, start_pos=6, cache=cache)
    flat_params = {}

    def flatten(tree, prefix=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                flatten(v, f"{prefix}{k}." if not prefix.endswith(".") or prefix == "" else f"{prefix}{k}.")
        elif isinstance(tree, list):
            for i, v in enumerate(tree):
                flatten(v, f"{prefix}{i}.")
        else:
            flat_params[prefix[:-1]] = arr(tree)

    flatten(dl.parameters())
    out["layer_sparse"] = {
        "params": flat_params,
        "x": arr(xl),
        "y": arr(yl),
        "x_step": arr(x_step),
        "y_step": arr(y_step),
    }

    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"wrote component fixtures to {out_path} ({os.path.getsize(out_path)} bytes)")


PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
    "<|message_user|><|content_text|>What is 2+2? Answer with just the number.<|end_message|><|message_model|>",
]


def dump_greedy(repo_dir: str, out_path: str, max_new_tokens: int) -> None:
    import mlx.core as mx

    sys.path.insert(0, repo_dir)
    from inkling_mlx.generate import greedy_generate, load_tokenizer
    from inkling_mlx.load import load

    try:
        mx.set_wired_limit(int(120e9))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] set_wired_limit: {e}")

    print(f"[load] {repo_dir} (eager)")
    model, config = load(repo_dir)
    tok = load_tokenizer(repo_dir)

    results = []
    for p in PROMPTS:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        out_ids = greedy_generate(model, config, ids, max_new_tokens=max_new_tokens)
        new_ids = out_ids[len(ids) :]
        results.append({
            "prompt": p,
            "prompt_ids": ids,
            "generated_ids": new_ids,
            "generated_text": tok.decode(new_ids),
        })
        print(f"[greedy] {p!r} -> {tok.decode(new_ids)!r}")

    with open(out_path, "w") as f:
        json.dump({"repo": repo_dir, "max_new_tokens": max_new_tokens, "results": results}, f, indent=1)
    print(f"wrote greedy ground truth to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    repo_dir = find_repo_dir(args.repo_dir)
    if args.greedy:
        dump_greedy(repo_dir, args.out or "/tmp/inkling_greedy.json", args.max_new_tokens)
    else:
        dump_components(repo_dir, args.out or "/tmp/inkling_fixtures.json")


if __name__ == "__main__":
    main()
