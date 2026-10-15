#!/usr/bin/env python3
"""`convert_minicpmv4_7.py` on a tiny synthetic HF checkpoint: names, the fused-expert split,
the conv1d transpose, the folded norms and the quantization widths.
Run: venv/bin/python tests/test_convert_minicpmv4_7.py"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
H, I, E = 64, 64, 2


def rnd(*shape):
    return mx.random.normal(shape).astype(mx.bfloat16)


def hf_tensors():
    lm = "model.language_model."
    layer = lm + "layers.0."
    return {
        lm + "embed_tokens.weight": rnd(128, H),
        lm + "norm.weight": rnd(H),
        "lm_head.weight": rnd(128, H),
        layer + "input_layernorm.weight": rnd(H),
        layer + "post_attention_layernorm.weight": rnd(H),
        layer + "linear_attn.conv1d.weight": rnd(6, 1, 4),
        layer + "linear_attn.norm.weight": rnd(16),
        layer + "linear_attn.in_proj_qkv.weight": rnd(128, H),
        layer + "self_attn.q_norm.weight": rnd(16),
        layer + "mlp.experts.gate_up_proj": rnd(E, 2 * I, H),
        layer + "mlp.experts.down_proj": rnd(E, H, I),
        layer + "mlp.gate.weight": rnd(E, H),
        layer + "mlp.shared_expert_gate.weight": rnd(1, H),
        "model.vision_tower.embeddings.patch_embedding.weight": rnd(8, 3, 14, 14),
        "model.vision_tower.encoder.layers.0.mlp.fc1.weight": rnd(16, 64),
        "model.vision_tower.vit_merger.linear_1.weight": rnd(64, 64),
        "model.merger.mlp.0.linear_1.weight": rnd(64, 64),
    }


def convert(tmp, bits):
    src, dst = Path(tmp, "src"), Path(tmp, f"dst{bits}")
    src.mkdir(exist_ok=True)
    tensors = hf_tensors()
    mx.save_safetensors(str(src / "model-00001-of-00001.safetensors"), tensors)
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}))
    (src / "config.json").write_text(json.dumps({"model_type": "minicpmv4_7"}))
    subprocess.run([sys.executable, str(HERE / "convert_minicpmv4_7.py"), "--src", str(src), "--dst", str(dst),
                    "--bits", str(bits)], check=True, capture_output=True)
    return tensors, mx.load(str(next(dst.glob("model-*.safetensors")))), json.loads((dst / "config.json").read_text())


class ConvertMiniCpmV47(unittest.TestCase):
    def test_bf16_pack_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out, cfg = convert(tmp, 16)
            lm = "language_model.model.layers.0."
            self.assertNotIn("quantization", cfg)
            self.assertTrue(mx.array_equal(out["language_model.lm_head.weight"], src["lm_head.weight"]))
            gu = src["model.language_model.layers.0.mlp.experts.gate_up_proj"]
            self.assertTrue(mx.array_equal(out[lm + "mlp.switch_mlp.gate_proj.weight"], gu[:, :I]))
            self.assertTrue(mx.array_equal(out[lm + "mlp.switch_mlp.up_proj.weight"], gu[:, I:]))
            self.assertEqual(out[lm + "linear_attn.conv1d.weight"].shape, (6, 4, 1))
            # Folded norms gain 1; the gated linear-attention norm and the head do not.
            for name, key in ((lm + "input_layernorm.weight", "model.language_model.layers.0.input_layernorm.weight"),
                              ("language_model.model.norm.weight", "model.language_model.norm.weight"),
                              (lm + "self_attn.q_norm.weight", "model.language_model.layers.0.self_attn.q_norm.weight")):
                want = (src[key].astype(mx.float32) + 1.0).astype(mx.bfloat16)
                self.assertTrue(mx.array_equal(out[name], want), name)
            self.assertTrue(mx.array_equal(out[lm + "linear_attn.norm.weight"],
                                           src["model.language_model.layers.0.linear_attn.norm.weight"]))
            self.assertEqual(out["vision_tower.embeddings.patch_embedding.weight"].shape, (8, 14, 14, 3))
            self.assertIn("merger.mlp.0.linear_1.weight", out)
            self.assertIn("vit_merger.linear_1.weight", out)

    def test_quantized_pack_widths(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out, cfg = convert(tmp, 4)
            lm = "language_model.model.layers.0."
            self.assertEqual((cfg["quantization"]["bits"], cfg["quantization"]["group_size"]), (4, 64))
            self.assertEqual(cfg["quantization"][lm + "mlp.gate"]["bits"], 8)
            self.assertEqual(cfg["quantization"][lm + "mlp.shared_expert_gate"]["bits"], 8)
            up = out[lm + "mlp.switch_mlp.up_proj.weight"]
            self.assertEqual(up.shape, (E, I, H * 4 // 32))
            deq = mx.dequantize(up, out[lm + "mlp.switch_mlp.up_proj.scales"], out[lm + "mlp.switch_mlp.up_proj.biases"],
                                group_size=64, bits=4)
            want = src["model.language_model.layers.0.mlp.experts.gate_up_proj"][:, I:].astype(mx.float32)
            self.assertLess(float(mx.abs(deq.astype(mx.float32) - want).max()), 0.6)
            self.assertEqual(out[lm + "mlp.gate.weight"].shape, (E, H * 8 // 32))
            # The tower stays dense.
            self.assertEqual(out["vision_tower.encoder.layers.0.mlp.fc1.weight"].dtype, mx.bfloat16)
            self.assertNotIn("vision_tower.encoder.layers.0.mlp.fc1.scales", out)
            self.assertNotIn("vit_merger.linear_1.scales", out)


if __name__ == "__main__":
    unittest.main()
