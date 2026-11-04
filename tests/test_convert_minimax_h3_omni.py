#!/usr/bin/env python3
"""Hermetic check of the FastH3 Omni DiT mapping in convert_minimax_h3_weights.py
on a tiny synthetic diffusers checkpoint: names land on the engine's, Q/K/V fuse
in that order, SwiGLU halves become (gate, value), the VSA gate and the PDD head
bank pass through whole and RoPE inverse frequencies are generated.

    uv run --with mlx python tests/test_convert_minimax_h3_omni.py
"""
import json
import os
import sys
import tempfile

import mlx.core as mx

mx.set_default_device(mx.cpu)

sys.path.insert(0, os.path.dirname(__file__))
from convert_minimax_h3_weights import fastvideo_dit  # noqa: E402

H, FFN, GRID = 8, 6, 4


def tiny_checkpoint(d):
    cfg = {"num_layers": 1, "num_refiner_layers": 1, "num_attention_heads": 2,
           "rope_freq_dim": 4, "rope_theta": 10000.0}
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    t = {}
    for stem in ("transformer_blocks.0.", "token_refiner.refiner_blocks.0."):
        for i, p in enumerate("qkv"):
            t[stem + f"attn.to_{p}.weight"] = mx.full((H, H), float(i + 1))
        t[stem + "ff.net.0.proj.weight"] = mx.concatenate([mx.full((FFN, H), 7.0), mx.full((FFN, H), 9.0)])
        t[stem + "ff.net.2.weight"] = mx.zeros((H, FFN))
    t["transformer_blocks.0.attn.to_gate_compress.weight"] = mx.zeros((H, H))
    t["proj_out.weight"] = mx.arange(GRID * 3 * H, dtype=mx.float32).reshape(GRID * 3, H)
    t["norm_out.linear.weight"] = mx.zeros((2 * H, H))
    t["context_embedder.weight"] = mx.zeros((H, H))
    mx.save_safetensors(os.path.join(d, "diffusion_pytorch_model-00001-of-00001.safetensors"), t)


def main():
    with tempfile.TemporaryDirectory() as d:
        tiny_checkpoint(d)
        out = fastvideo_dit(d)
        assert set(out) == {
            "blocks.0.attn.qkv_proj.weight", "blocks.0.mlp.fc1.weight", "blocks.0.mlp.fc2.weight",
            "token_refiner.blocks.0.attn.qkv_proj.weight", "token_refiner.blocks.0.mlp.fc1.weight",
            "token_refiner.blocks.0.mlp.fc2.weight", "final_layer.video_out.weight",
            "final_layer.adaln_proj.linear.weight", "condition_proj.weight", "rope.inv_freq",
            "blocks.0.attn.to_gate_compress.weight",
        }, sorted(out)
        qkv = out["blocks.0.attn.qkv_proj.weight"]
        assert [float(qkv[i * H, 0]) for i in range(3)] == [1.0, 2.0, 3.0]
        fc1 = out["blocks.0.mlp.fc1.weight"]
        assert (float(fc1[0, 0]), float(fc1[FFN, 0])) == (9.0, 7.0), "gate half must come first"
        assert out["final_layer.video_out.weight"].shape == (GRID * 3, H)
        want = [1.0, 0.1, 0.01, 0.001]
        assert len(out["rope.inv_freq"]) == 4 and all(abs(float(a) - b) < 1e-6 for a, b in zip(out["rope.inv_freq"], want))
    print("ok")


if __name__ == "__main__":
    main()
