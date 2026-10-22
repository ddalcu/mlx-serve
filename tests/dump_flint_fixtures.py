#!/usr/bin/env python3
"""Engine oracle for Recursive Flint (`model_type: flint`) on a TINY random model.

Builds a random Flint with the release's structure (dense leading layers, sliding
and NoPE full-attention layers, learned sinks, sigmoid top-k MoE with an ungated
shared expert), runs Recursive's own MLX model (`rec_mlx/model.py` from
github.com/recursive-oss/rec) in f32 on the CPU, and writes two packs plus their
logits:

  pack-bf16/  dense bf16 weights;          fixture-bf16.safetensors
  pack-q4/    the release's storage: 4-bit g64 affine projections and experts,
              8-bit embeddings, bf16 router, packed weights stored as uint8;
              fixture-q4.safetensors (reference run on the dequantized weights)

Every expert runs (top-k = all), so a near-tie in expert selection cannot fail
the comparison (bf16 flips sigmoid picks a few rows in 40); selection itself is
exercised by the live model.

  ~/claude-tmp/flint-venv/bin/python tests/dump_flint_fixtures.py \\
      --rec <rec checkout>/engine/src --out ~/claude-tmp/flint-tiny
  FLINT_MODEL=~/claude-tmp/flint-tiny/pack-q4 FLINT_FIXTURE=~/claude-tmp/flint-tiny/fixture-q4.safetensors \\
      zig build test -Dtest-filter="flint fixture"
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

LAYER_TYPES = ["full_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention"]
CONFIG = {
    "architectures": ["FlintForCausalLM"],
    "model_type": "flint",
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 128,
    "num_hidden_layers": len(LAYER_TYPES),
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "num_experts": 8,
    "num_experts_per_tok": 8,
    "norm_topk_prob": True,
    "vocab_size": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "sliding_window": 9,
    "use_sliding_window": True,
    "layer_types": LAYER_TYPES,
    "no_rope_layer_indices": [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"],
    "mlp_only_layers": [0, 1],
    "router_score_function": "sigmoid",
    "softmax_type": "learnable",
    "tie_word_embeddings": False,
    "max_position_embeddings": 262144,
    "bos_token_id": 0,
    "eos_token_id": [1],
    "torch_dtype": "bfloat16",
}
T = 40


def random_weights(rng):
    c = CONFIG
    d, hd, h, kv = c["hidden_size"], c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
    w = {}

    def lin(name, out, inp, scale=None):
        w[name] = rng.standard_normal((out, inp), dtype=np.float32) * (scale or inp**-0.5)

    w["model.embed_tokens.weight"] = rng.standard_normal((c["vocab_size"], d), dtype=np.float32)
    lin("lm_head.weight", c["vocab_size"], d)
    w["model.norm.weight"] = 1.0 + 0.1 * rng.standard_normal(d, dtype=np.float32)
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}"
        w[f"{p}.input_layernorm.weight"] = 1.0 + 0.1 * rng.standard_normal(d, dtype=np.float32)
        w[f"{p}.post_attention_layernorm.weight"] = 1.0 + 0.1 * rng.standard_normal(d, dtype=np.float32)
        lin(f"{p}.self_attn.q_proj.weight", h * hd, d)
        lin(f"{p}.self_attn.k_proj.weight", kv * hd, d)
        lin(f"{p}.self_attn.v_proj.weight", kv * hd, d)
        lin(f"{p}.self_attn.o_proj.weight", d, h * hd)
        w[f"{p}.self_attn.q_norm.weight"] = 1.0 + 0.1 * rng.standard_normal(hd, dtype=np.float32)
        w[f"{p}.self_attn.k_norm.weight"] = 1.0 + 0.1 * rng.standard_normal(hd, dtype=np.float32)
        # Sinks large enough to move the softmax: a zero sink would not test the term.
        w[f"{p}.self_attn.softmax_offset"] = 2.0 * rng.standard_normal(h, dtype=np.float32)
        if i in c["mlp_only_layers"]:
            lin(f"{p}.mlp.gate_proj.weight", c["intermediate_size"], d)
            lin(f"{p}.mlp.up_proj.weight", c["intermediate_size"], d)
            lin(f"{p}.mlp.down_proj.weight", d, c["intermediate_size"])
            continue
        e, m, sh = c["num_experts"], c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
        lin(f"{p}.mlp.gate.weight", e, d, 2.0 * d**-0.5)
        for n, (out, inp) in {"gate_proj": (m, d), "up_proj": (m, d), "down_proj": (d, m)}.items():
            w[f"{p}.mlp.switch_mlp.{n}.weight"] = rng.standard_normal((e, out, inp), dtype=np.float32) * inp**-0.5
        lin(f"{p}.mlp.shared_expert.gate_proj.weight", sh, d)
        lin(f"{p}.mlp.shared_expert.up_proj.weight", sh, d)
        lin(f"{p}.mlp.shared_expert.down_proj.weight", d, sh)
    return {k: mx.array(v) for k, v in w.items()}


def bf16_round(w):
    return {k: v.astype(mx.bfloat16).astype(mx.float32) for k, v in w.items()}


def quantize_release(w):
    """The release's storage, and the f32 values it decodes to."""
    stored, decoded = {}, {}
    for k, v in w.items():
        bits = 8 if k == "model.embed_tokens.weight" else 4
        quantized = k.endswith(".weight") and v.ndim >= 2 and not k.endswith(".mlp.gate.weight")
        if not quantized:
            stored[k] = v.astype(mx.bfloat16)
            decoded[k] = stored[k].astype(mx.float32)
            continue
        q, s, b = mx.quantize(v.astype(mx.bfloat16), group_size=64, bits=bits)
        base = k[: -len(".weight")]
        stored[k] = q.view(mx.uint8)
        stored[f"{base}.scales"] = s
        stored[f"{base}.biases"] = b
        decoded[k] = mx.dequantize(q, s, b, group_size=64, bits=bits).astype(mx.float32)
    return stored, decoded


def reference_logits(rec_model, weights, ids):
    model = rec_model.Model(rec_model.ModelArgs.from_dict(CONFIG))
    model.load_weights(list(model.sanitize(dict(weights)).items()), strict=True)
    logits = model(mx.array(ids)[None])
    mx.eval(logits)
    return logits[0].astype(mx.float32)


def write_pack(path, stored, quantization):
    path.mkdir(parents=True, exist_ok=True)
    cfg = dict(CONFIG)
    if quantization:
        cfg["quantization"] = quantization
    (path / "config.json").write_text(json.dumps(cfg, indent=2))
    mx.save_safetensors(str(path / "model.safetensors"), stored, metadata={"format": "mlx"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True, help="rec checkout's engine/src (holds rec_mlx/)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.rec)
    from rec_mlx import model as rec_model

    mx.set_default_device(mx.cpu)
    out = Path(args.out).expanduser()
    rng = np.random.default_rng(20261021)
    w = random_weights(rng)
    ids = rng.integers(2, CONFIG["vocab_size"], size=T, dtype=np.int32)

    dense = bf16_round(w)
    write_pack(out / "pack-bf16", {k: v.astype(mx.bfloat16) for k, v in dense.items()}, None)
    logits = reference_logits(rec_model, dense, ids)
    mx.save_safetensors(str(out / "fixture-bf16.safetensors"), {"input_ids": mx.array(ids), "logits_full": logits})

    stored, decoded = quantize_release(w)
    write_pack(out / "pack-q4", stored, {"group_size": 64, "bits": 4})
    logits = reference_logits(rec_model, decoded, ids)
    mx.save_safetensors(str(out / "fixture-q4.safetensors"), {"input_ids": mx.array(ids), "logits_full": logits})
    print(f"wrote {out}: {T} tokens, logits {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
