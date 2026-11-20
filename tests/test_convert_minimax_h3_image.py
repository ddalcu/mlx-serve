#!/usr/bin/env python3
"""Hermetic check of the MiniMax-H3-Image DiT in convert_minimax_h3_weights.py on a tiny synthetic snapshot: the
release's names pass through, its per-head interleaved Q/K/V become the [q | k | v] blocks the engine splits, its
single output head gives way to the PDD heads stacked step-major, and a release that carries an audio stream or heads
of another grid is refused.

    uv run --with mlx python tests/test_convert_minimax_h3_image.py
"""
import json
import os
import sys
import tempfile

import mlx.core as mx

mx.set_default_device(mx.cpu)

sys.path.insert(0, os.path.dirname(__file__))
from convert_minimax_h3_weights import image_dit  # noqa: E402

H, OUT, STEPS = 8, 4, 3
HEADS, HD = 2, 4


def tiny_snapshot(d, heads=STEPS, audio=False):
    os.makedirs(os.path.join(d, "transformer"), exist_ok=True)
    # The release interleaves per head: rows [h0 q | h0 k | h0 v | h1 q | ...], HD rows each.
    qkv = mx.arange(3 * H * H, dtype=mx.float32).reshape(3 * H, H)
    t = {"blocks.0.attn.qkv_proj.weight": qkv, "token_refiner.blocks.0.attn.qkv_proj.weight": qkv,
         "final_layer.video_out.weight": mx.zeros((OUT, H)),
         "final_layer.video_out.bias": mx.zeros((OUT,))}
    if audio:
        t["audio_patch_proj.weight"] = mx.zeros((H, 2))
    mx.save_safetensors(os.path.join(d, "transformer", "model-00001-of-00002.safetensors"), t)
    mx.save_safetensors(os.path.join(d, "transformer", "model-00002-of-00002.safetensors"), {"rope.inv_freq": mx.ones((4,))})
    json.dump({"num_attention_heads": HEADS, "attention_head_dim": HD}, open(os.path.join(d, "transformer", "config.json"), "w"))
    w = mx.arange(heads * OUT * H, dtype=mx.float32).reshape(heads, OUT, H)
    b = mx.arange(heads * OUT, dtype=mx.float32).reshape(heads, OUT)
    mx.save_safetensors(os.path.join(d, "pdd_fused_heads.safetensors"), {"video_out.weight": w, "video_out.bias": b})
    sigmas = [1.0 - i / STEPS for i in range(STEPS)] + [0.0]
    json.dump({"num_inference_steps": STEPS, "sigmas": sigmas}, open(os.path.join(d, "pdd_config.json"), "w"))


def refused(d):
    try:
        image_dit(d)
    except SystemExit:
        return True
    return False


def main():
    with tempfile.TemporaryDirectory() as d:
        tiny_snapshot(d)
        out = image_dit(d)
        assert set(out) == {"blocks.0.attn.qkv_proj.weight", "token_refiner.blocks.0.attn.qkv_proj.weight",
                            "rope.inv_freq", "final_layer.video_out.weight", "final_layer.video_out.bias"}, sorted(out)
        # [q | k | v]: the engine's row r of part p, head h is the release's row (h * 3 + p) * HD + r.
        for name in ("blocks.0.attn.qkv_proj.weight", "token_refiner.blocks.0.attn.qkv_proj.weight"):
            got = out[name]
            for p in range(3):
                for h in range(HEADS):
                    for r in range(HD):
                        src_row = (h * 3 + p) * HD + r
                        assert float(got[p * H + h * HD + r, 0]) == src_row * H, (name, p, h, r)
        w, b = out["final_layer.video_out.weight"], out["final_layer.video_out.bias"]
        assert w.shape == (STEPS * OUT, H) and b.shape == (STEPS * OUT,)
        # Step k's head is rows [k*OUT, (k+1)*OUT): the bank the engine slices per step.
        assert float(w[OUT, 0]) == OUT * H and float(b[2 * OUT]) == 2 * OUT
    with tempfile.TemporaryDirectory() as d:
        tiny_snapshot(d, heads=STEPS + 1)
        assert refused(d), "heads of another grid must be refused"
    with tempfile.TemporaryDirectory() as d:
        tiny_snapshot(d, audio=True)
        assert refused(d), "an audio stream is not the image release"
    print("ok")


if __name__ == "__main__":
    main()
