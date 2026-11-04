#!/usr/bin/env python3
"""Hermetic numerical and packet guards for the NVIDIA LiLiCorr converter."""

import importlib.util
from pathlib import Path
import json
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SPEC = importlib.util.spec_from_file_location("convert_lilicorr", Path(__file__).with_name("convert_nemotron_lilicorr.py"))
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class Nvfp4Tests(unittest.TestCase):
    def test_packed_order_fp8_scales_and_global_scale(self):
        # Known E2M1 values, including both signs and non-power-of-two magnitudes.
        packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8)
        scales = torch.tensor([[1.5, 0.125]], dtype=torch.float8_e4m3fn)
        global_scale = torch.tensor(0.25)
        codes = [0., .5, 1., 1.5, 2., 3., 4., 6., 0., -.5, -1., -1.5, -2., -3., -4., -6.]
        expected = torch.tensor([[x * .375 for x in codes] + [x * .03125 for x in codes]])
        actual = converter.dequantize_nvfp4(packed, scales, global_scale)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_malformed_packets(self):
        packed = torch.zeros((2, 16), dtype=torch.uint8)
        scales = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
        cases = [
            (packed.to(torch.int8), scales, torch.tensor(1.)),
            (packed, scales.float(), torch.tensor(1.)),
            (packed, scales[:1], torch.tensor(1.)),
            (packed, scales, torch.ones(2)),
            (packed, scales, torch.tensor(float("nan"))),
            (packed, scales, torch.tensor(0.)),
            (packed, scales, torch.tensor(-1.)),
            (packed, torch.full((2, 2), float("nan"), dtype=torch.float8_e4m3fn), torch.tensor(1.)),
            (packed, (-torch.ones((2, 2))).to(torch.float8_e4m3fn), torch.tensor(1.)),
        ]
        for args in cases:
            with self.subTest(dtypes=[x.dtype for x in args], shapes=[tuple(x.shape) for x in args]):
                with self.assertRaises(ValueError):
                    converter.dequantize_nvfp4(*args)

    def test_attention_bias_uses_query_minus_key_and_same_slot(self):
        relative = torch.tensor([[0., 1., 2., 3., 4., 5., 6.], [0., -1., -2., -3., -4., -5., -6.]], dtype=torch.bfloat16)
        same = torch.tensor([10., 20.], dtype=torch.bfloat16)
        bias = converter.attention_bias(relative, same, block_size=4, top_k=2)
        expected = torch.tensor([
            [13., 13., 2., 2., 1., 1.],
            [13., 13., 2., 2., 1., 1.],
            [4., 4., 13., 13., 2., 2.],
            [4., 4., 13., 13., 2., 2.],
            [5., 5., 4., 4., 13., 13.],
            [5., 5., 4., 4., 13., 13.],
        ], dtype=torch.bfloat16)
        torch.testing.assert_close(bias[0], expected, rtol=0, atol=0)
        self.assertEqual(bias.shape, (2, 6, 6))
        self.assertEqual(float(bias[1, 0, 0]), 17.)
        self.assertEqual(float(bias[1, 2, 0]), -4.)


def scalar_reference(packed, scales, global_scale):
    # Scalar E2M1 decode from exponent/mantissa fields, independent of the LUT.
    rows = []
    for row, scale_row in zip(packed.tolist(), scales.float().tolist()):
        values = []
        for byte in row:
            for code in (byte & 15, byte >> 4):
                exponent, mantissa = (code >> 1) & 3, code & 1
                magnitude = mantissa * .5 if exponent == 0 else (1 + mantissa * .5) * 2 ** (exponent - 1)
                values.append((-magnitude if code & 8 and magnitude else magnitude) * scale_row[len(values) // 16] * float(global_scale))
        rows.append(values)
    return torch.tensor(rows)


def tiny_checkpoint(root):
    config = dict(architectures=["LiLiCorrDraftModel"], hidden_size=32, head_dim=8,
                  num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=1,
                  intermediate_size=64, block_size=4, vocab_size=32,
                  has_own_lm_head=True, draft_vocab_size=32,
                  dflash_config=dict(target_layer_ids=[0, 1], projector_type="lilicorr",
                      lilicorr_hidden_size=16, lilicorr_num_layers=1, lilicorr_num_heads=2,
                      lilicorr_mlp_ratio=2., lilicorr_factor_dim=16, lilicorr_candidate_topk=3,
                      conv_kernel_size=2, conv_group_size=16),
                  quantization_config=dict(quant_algo="W4A16_NVFP4", group_size=16))
    tensors = {name: torch.full(shape, .25, dtype=torch.bfloat16) for name, shape in converter.required_shapes(config).items()}
    for name in ("fc.weight", "lm_head.weight", "lilicorr.context_proj.weight"):
        rows, cols = tensors[name].shape
        tensors[name] = torch.arange(rows * cols // 2).to(torch.uint8).reshape(rows, cols // 2)
        base = name.removesuffix(".weight")
        tensors[base + ".weight_scale"] = torch.ones((rows, cols // 16), dtype=torch.float8_e4m3fn)
        tensors[base + ".weight_scale_2"] = torch.tensor(.125)
    (root / "config.json").write_text(json.dumps(config))
    save_file(tensors, str(root / "model.safetensors"))
    return config, tensors


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_preserves_own_head_and_explicit_lilicorr_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            config, tensors = tiny_checkpoint(source)
            manifest = converter.convert_checkpoint(source, root / "output", scalar_reference, shard_bytes=4096)
            self.assertEqual(manifest["quantized_matrices"], 3)
            self.assertEqual(manifest["source_tensor_count"], len(tensors))
            self.assertEqual(manifest["output_tensor_count"], len(tensors) - 7)
            output = root / "output"
            result_config = json.loads((output / "config.json").read_text())
            self.assertTrue(result_config["has_own_lm_head"])
            self.assertEqual(result_config["draft_vocab_size"], 32)
            self.assertNotIn("quantization_config", result_config)
            self.assertEqual(json.loads((source / "config.json").read_text()), config)
            index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
            self.assertNotIn("lilicorr.layers.0.attn.in_proj_weight", index)
            self.assertIn("lilicorr.layers.0.attn.in_proj.weight", index)
            self.assertNotIn("lilicorr.relative_slot_bias", index)
            self.assertNotIn("lilicorr.same_slot_bias", index)
            self.assertIn("lilicorr.attention_bias", index)
            with safe_open(str(output / index["lm_head.weight"]), framework="pt") as weights:
                expected = scalar_reference(tensors["lm_head.weight"], tensors["lm_head.weight_scale"], tensors["lm_head.weight_scale_2"]).to(torch.bfloat16)
                torch.testing.assert_close(weights.get_tensor("lm_head.weight"), expected, rtol=0, atol=0)
            with safe_open(str(output / index["lilicorr.slot_embedding"]), framework="pt") as weights:
                self.assertEqual(weights.get_tensor("lilicorr.slot_embedding").shape, (3, 16))
            with self.assertRaises(ValueError):
                converter.convert_checkpoint(source, output, scalar_reference)

    def test_incomplete_and_orphan_packets_leave_no_partial_output(self):
        for mutation in ("missing_scale", "orphan_scale", "missing_head"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                source.mkdir()
                _, tensors = tiny_checkpoint(source)
                if mutation == "missing_scale":
                    del tensors["lm_head.weight_scale_2"]
                elif mutation == "orphan_scale":
                    tensors["orphan.weight_scale"] = torch.ones((1, 1), dtype=torch.float8_e4m3fn)
                else:
                    del tensors["lm_head.weight"]
                save_file(tensors, str(source / "model.safetensors"))
                with self.assertRaises(ValueError):
                    converter.convert_checkpoint(source, root / "output", scalar_reference, shard_bytes=4096)
                self.assertFalse((root / "output").exists())
                self.assertEqual(sorted(x.name for x in root.iterdir()), ["source"])


if __name__ == "__main__":
    unittest.main()
