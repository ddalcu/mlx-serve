#!/usr/bin/env python3
"""Dump reference oracles for the poolside Laguna S 2.1 (`model_type: laguna`) port.

The 117.6B/8.5B-active checkpoint can't run locally, so — per the fixture rule
(dump deep nets on CPU in fp32; fp16-on-MPS silently decorrelates) — we
instantiate the self-contained reference `modeling_laguna.py` with a TINY random
config and dump per-module oracles the Zig engine is checked against:

  * yarn.inv_freq / yarn.freqs  — the full-attention YaRN rotary frequencies
    (mlx_fast_rope consumes `freqs`; angle = pos/freqs). Pinned in Zig by
    `computeYarnFreqs` under LAGUNA_FIXTURES (src/engines/mlx/transformer.zig).
  * attn_full / attn_sliding    — one LagunaAttention block I/O (softplus gate,
    per-layer heads, YaRN vs default rope, sliding window).
  * router                      — sigmoid + e_score_correction_bias top-k select
    (unbiased gathered weights, L1 renorm, ×routed_scaling_factor).
  * moe                         — full LagunaSparseMoeBlock I/O (routed + ungated
    shared expert, route scale).
  * logits                      — end-to-end LagunaForCausalLM logits for a fixed
    input id sequence (the whole forward: per-layer heads, gate, YaRN, sliding
    mask, sigmoid MoE, dense layer 0).

Run (downloads torch + transformers on first use; the reference .py files come
from the base repo, which ships modeling_laguna.py + configuration_laguna.py):

    uv run --with torch --with transformers --with numpy \
        python3 tests/dump_laguna_fixtures.py \
        [--ref-dir <dir with modeling_laguna.py>] \
        [--repo poolside/Laguna-S-2.1-NVFP4] \
        [--out-json /tmp/laguna_fixtures.json] [--out-npz /tmp/laguna_modules.npz]

Then the Zig yarn oracle:
    LAGUNA_FIXTURES=/tmp/laguna_fixtures.json \
        zig build test -Dtest-filter="laguna yarn parity"

The module/logits oracles in the .npz are for a Python-side or manual forward
comparison once the engine runs the tiny model on GPU (build the tiny MLX model
with --build-tiny-mlx to load it into mlx-serve).
"""

import argparse
import json
import os
import sys
import importlib.util


# Tiny but STRUCTURALLY faithful config: one full/sliding group so per-layer
# heads (full=4, sliding=6) and both rope regimes are exercised; a handful of
# experts; layer 0 dense (mlp_only_layers). head_dim 128 kept (partial 0.5 →
# rotary_dim 64, the real geometry the YaRN freqs are calibrated for).
def tiny_config(LagunaConfig):
    return LagunaConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        gating="per-head",
        hidden_act="silu",
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        sliding_window=8,
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
        num_attention_heads_per_layer=[4, 6, 6, 6],
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        norm_topk_prob=True,
        mlp_only_layers=[0],
        moe_routed_scaling_factor=2.5,
        moe_router_logit_softcapping=0.0,
        rope_parameters={
            "full_attention": {
                "rope_theta": 500000.0,
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 8192,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "attention_factor": 1.3465735902799727,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
    )


def load_reference(ref_dir, repo):
    """Import configuration_laguna + modeling_laguna from a local dir or HF."""
    if ref_dir is None:
        from huggingface_hub import snapshot_download
        ref_dir = snapshot_download(repo, allow_patterns=["*.py"])
    sys.path.insert(0, ref_dir)

    def _load(name, fname):
        spec = importlib.util.spec_from_file_location(name, os.path.join(ref_dir, fname))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg_mod = _load("configuration_laguna", "configuration_laguna.py")
    model_mod = _load("modeling_laguna", "modeling_laguna.py")
    return cfg_mod.LagunaConfig, model_mod.LagunaForCausalLM


def dump(args):
    import torch
    import numpy as np

    torch.manual_seed(0)
    LagunaConfig, LagunaForCausalLM = load_reference(args.ref_dir, args.repo)
    config = tiny_config(LagunaConfig)

    # CPU fp32 — deep-net fixture rule (fp16/MPS decorrelates silently).
    model = LagunaForCausalLM(config).to(torch.float32).eval()

    json_out = {
        "config": {
            "hidden_size": config.hidden_size,
            "head_dim": config.head_dim,
            "num_attention_heads_per_layer": list(config.num_attention_heads_per_layer),
            "num_key_value_heads": config.num_key_value_heads,
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_routed_scaling_factor": float(config.moe_routed_scaling_factor),
            "sliding_window": config.sliding_window,
        },
    }
    npz = {}

    # ── YaRN rotary frequencies (full-attention layers) ──
    # rotary_emb is built from the full_attention sub-config (flattened by the
    # reference). inv_freq is the buffer; freqs = 1/inv_freq is what
    # mlx_fast_rope consumes (angle = pos/freqs). attention_scaling is the mscale.
    rotary = model.model.rotary_emb
    inv_freq = rotary.inv_freq.detach().float().cpu().numpy()
    assert np.all(inv_freq != 0), "rotary inv_freq is all-zero — transformers>=5 buffer-materialization bug"
    json_out["yarn"] = {
        "inv_freq": inv_freq.tolist(),
        "freqs": (1.0 / inv_freq).tolist(),
        "attention_scaling": float(rotary.attention_scaling),
        "rope_theta": 500000.0,
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 8192,
        "partial_rotary_factor": 0.5,
    }

    # ── Fixed input + full forward (E2E logits oracle) ──
    input_ids = torch.arange(1, 13, dtype=torch.long).unsqueeze(0)  # [1, 12]
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    npz["input_ids"] = input_ids.cpu().numpy()
    npz["logits"] = out.logits.detach().float().cpu().numpy()

    # ── Per-module oracles: capture I/O of layer 1 (full? no — layer 0 is dense,
    # layer 1 is the first sliding MoE layer; layer 0 full-attn). Hook the first
    # full-attn block (layer 0) and the first sliding block (layer 1). ──
    hidden = model.model.embed_tokens(input_ids)
    npz["block_input_hidden"] = hidden.detach().float().cpu().numpy()

    # Full-attention block (layer 0): input_layernorm → self_attn.
    l0 = model.model.layers[0]
    normed0 = l0.input_layernorm(hidden)
    pe_full = model.model.rotary_emb(hidden, torch.arange(hidden.shape[1]).unsqueeze(0))
    with torch.no_grad():
        attn0, _ = l0.self_attn(hidden_states=normed0, position_embeddings=pe_full, attention_mask=None)
    npz["attn_full_out"] = attn0.detach().float().cpu().numpy()

    # Sliding block (layer 1): default rope.
    l1 = model.model.layers[1]
    normed1 = l1.input_layernorm(hidden)
    swa_rotary = model.model.swa_rotary_emb if model.model.swa_rotary_emb is not None else model.model.rotary_emb
    pe_swa = swa_rotary(hidden, torch.arange(hidden.shape[1]).unsqueeze(0))
    with torch.no_grad():
        attn1, _ = l1.self_attn(hidden_states=normed1, position_embeddings=pe_swa, attention_mask=None)
    npz["attn_sliding_out"] = attn1.detach().float().cpu().numpy()

    # Router + MoE block (layer 1 is a LagunaSparseMoeBlock).
    moe_block = l1.mlp
    moe_in = l1.post_attention_layernorm(hidden + attn1)
    with torch.no_grad():
        _, routing_weights, selected = moe_block.gate(moe_in.reshape(-1, config.hidden_size))
        moe_out = moe_block(moe_in)
    npz["router_weights"] = routing_weights.detach().float().cpu().numpy()
    npz["router_selected"] = selected.detach().cpu().numpy()
    npz["moe_out"] = moe_out.detach().float().cpu().numpy()

    with open(args.out_json, "w") as f:
        json.dump(json_out, f, indent=2)
    np.savez(args.out_npz, **npz)
    print(f"wrote yarn/config oracle → {args.out_json}")
    print(f"wrote module + logits oracles → {args.out_npz}")
    print(f"  yarn: {len(inv_freq)} freqs, attention_scaling={rotary.attention_scaling:.6f}")
    print(f"  logits shape {npz['logits'].shape}, attn_full {npz['attn_full_out'].shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default=None, help="dir containing modeling_laguna.py (else download --repo)")
    ap.add_argument("--repo", default="poolside/Laguna-S-2.1-NVFP4", help="HF repo shipping the reference .py")
    ap.add_argument("--out-json", default="/tmp/laguna_fixtures.json")
    ap.add_argument("--out-npz", default="/tmp/laguna_modules.npz")
    args = ap.parse_args()
    dump(args)


if __name__ == "__main__":
    main()
