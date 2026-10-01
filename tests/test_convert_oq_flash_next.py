#!/usr/bin/env python3
"""Hermetic check of tests/convert_oq_flash_next.py on a tiny synthetic oQ pack:  python3 tests/test_convert_oq_flash_next.py"""
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dsv4_weights import bf16_to_f32, f32_to_bf16_u16, write_safetensors_raw  # noqa: E402
import convert_oq_flash_next as oq  # noqa: E402

NG = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
ROWS, WCOLS, SCOLS = 2, 4, 1  # dim 32 at 4 bits, group size 32


def bf16(vals):
    a = f32_to_bf16_u16(np.array(vals, dtype=np.float32))
    return ("BF16", list(a.shape), a.tobytes())


def u32(seed, shape):
    a = (np.arange(int(np.prod(shape)), dtype=np.uint32) + seed).reshape(shape)
    return ("U32", list(shape), a.tobytes())


def make_pack(root, q_norm=(0.25, -0.5, 0.0, 1.5), weight_scale=1.0):
    shard1 = {
        "mtp.fc.weight": bf16([[1.0, 2.0], [3.0, 4.0]]),
        "vision_tower.blocks.0.norm1.weight": bf16([0.5, 0.5]),
        "language_model.model.layers.0.self_attn.q_norm.weight": bf16(q_norm),
        "language_model.model.layers.0.linear_attn.norm.weight": bf16([0.9, 0.9, 0.9, 0.9]),
        f"{NG}.weight_scale": bf16([weight_scale]),
    }
    shard2 = {"language_model.model.embed_tokens.weight": u32(7, (2, 2))}
    for k in range(3):
        sh = shard1 if k < 2 else shard2
        sh[f"{NG}.shards.{k}.weight"] = u32(100 * k, (ROWS, WCOLS))
        sh[f"{NG}.shards.{k}.scales"] = bf16([[float(k + 1)]] * ROWS)
        sh[f"{NG}.shards.{k}.biases"] = bf16([[-float(k + 1)]] * ROWS)
    weight_map = {}
    for fn, sh in (("model-00001-of-00002.safetensors", shard1), ("model-00002-of-00002.safetensors", shard2)):
        write_safetensors_raw(str(root / fn), sh)
        weight_map.update({k: fn for k in sh})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}))
    cfg = {
        "model_type": "qwen4_exp",
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine",
                         f"{NG}.shards.0": {"bits": 4, "group_size": 32, "mode": "affine"}},
        "text_config": {"rope_parameters": {"type": "default", "rope_theta": 10000000}},
    }
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "tokenizer.json").write_text("{}")


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, m in header.items():
            if k == "__metadata__":
                continue
            f.seek(base + m["data_offsets"][0])
            out[k] = (m["dtype"], m["shape"], f.read(m["data_offsets"][1] - m["data_offsets"][0]))
    return header.get("__metadata__"), out


class ConvertOqFlashNext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src, self.dst = Path(self.tmp.name) / "src", Path(self.tmp.name) / "dst"
        self.src.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def tensors(self):
        out = {}
        for p in sorted(self.dst.glob("model-*.safetensors")):
            out.update(read_st(p)[1])
        return out

    def test_prefixes_are_renamed_and_ngram_shards_leave_the_trunk(self):
        make_pack(self.src)
        oq.convert(self.src, self.dst)
        names = set(self.tensors())
        self.assertIn("language_model.mtp.fc.weight", names)
        self.assertIn("model.visual.blocks.0.norm1.weight", names)
        self.assertFalse([n for n in names if n.startswith(("mtp.", "vision_tower.")) or "ngram_embedding" in n])
        index = json.loads((self.dst / "model.safetensors.index.json").read_text())["weight_map"]
        self.assertEqual(set(index), names)

    def test_norms_gain_one_and_the_gated_norm_does_not(self):
        make_pack(self.src)
        oq.convert(self.src, self.dst)
        t = self.tensors()
        q = bf16_to_f32(np.frombuffer(t["language_model.model.layers.0.self_attn.q_norm.weight"][2], dtype=np.uint16))
        np.testing.assert_array_equal(q, [1.25, 0.5, 1.0, 2.5])
        g = bf16_to_f32(np.frombuffer(t["language_model.model.layers.0.linear_attn.norm.weight"][2], dtype=np.uint16))
        np.testing.assert_allclose(g, bf16_to_f32(f32_to_bf16_u16(np.full(4, 0.9, dtype=np.float32))))

    def test_ngram_shards_concatenate_into_the_table_in_shard_order(self):
        make_pack(self.src)
        oq.convert(self.src, self.dst)
        meta, table = read_st(self.dst / "ngram_table.bin")
        self.assertEqual(meta, {"format": "mlx-serve-ngram", "bits": "4", "group_size": "32"})
        self.assertEqual(table["weight"][1], [3 * ROWS, WCOLS])
        self.assertEqual(table["weight"][2], b"".join(u32(100 * k, (ROWS, WCOLS))[2] for k in range(3)))
        self.assertEqual(table["scales"][2], b"".join(bf16([[float(k + 1)]] * ROWS)[2] for k in range(3)))
        self.assertEqual(table["biases"][2], b"".join(bf16([[-float(k + 1)]] * ROWS)[2] for k in range(3)))

    def test_config_gains_the_table_block_and_rope_type(self):
        make_pack(self.src)
        oq.convert(self.src, self.dst)
        cfg = json.loads((self.dst / "config.json").read_text())
        self.assertEqual(cfg["ngram_table"], {"file": "ngram_table.bin", "bits": 4, "group_size": 32})
        self.assertEqual(cfg["text_config"]["rope_parameters"], {"rope_type": "default", "rope_theta": 10000000})
        self.assertTrue((self.dst / "tokenizer.json").exists())

    def test_a_pack_whose_norms_are_already_folded_is_refused(self):
        make_pack(self.src, q_norm=(1.25, 0.5, 1.0, 2.5))
        with self.assertRaises(SystemExit) as e:
            oq.convert(self.src, self.dst)
        self.assertIn("already", str(e.exception))

    def test_a_non_unit_ngram_weight_scale_is_refused(self):
        make_pack(self.src, weight_scale=2.0)
        with self.assertRaises(SystemExit) as e:
            oq.convert(self.src, self.dst)
        self.assertIn("weight_scale", str(e.exception))


if __name__ == "__main__":
    unittest.main()
