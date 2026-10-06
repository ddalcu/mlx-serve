#!/usr/bin/env python3
"""`convert_qwen38_flash_next.py --bf16` on a tiny synthetic HF checkpoint: the pack keeps every
tensor bf16 (renamed, experts split, conv1d transposed, norms folded) and the n-gram table is a
raw bf16 `ngram_table.bin` (`"bits":"16"`). Run: venv/bin/python tests/test_convert_qwen38_flash_next_bf16.py"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from convert_dsv4_weights import bf16_to_f32, f32_to_bf16_u16, write_safetensors_raw  # noqa: E402

NG = ".ple.ple_embedding.ngram_embedding.shard_"
E, I, H, ROWS, DIM = 2, 4, 8, 3, 8


def bf16(a):
    return np.ascontiguousarray(f32_to_bf16_u16(np.asarray(a, np.float32)))


def triple(a):
    return ("BF16", a.shape, a.tobytes())


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    meta = header.pop("__metadata__", {})
    return meta, header, data


def tensor(header, data, name):
    m = header[name]
    b, e = m["data_offsets"]
    return np.frombuffer(data[b:e], np.uint16).reshape(m["shape"])


def convert(tmp, tensors, *flags):
    src, dst = Path(tmp, "src"), Path(tmp, "dst")
    src.mkdir()
    write_safetensors_raw(str(src / "model-1.safetensors"), tensors)
    (src / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-1.safetensors" for k in tensors}}))
    (src / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "num_hidden_layers": 1}))
    run = subprocess.run([sys.executable, str(HERE / "convert_qwen38_flash_next.py"),
                          "--src", str(src), "--dst", str(dst), *flags], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    index = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    header, data = {}, {}
    for f in sorted(set(index.values())):
        _, header[f], data[f] = read_st(dst / f)
    return dst, index, header, data


class Bf16Pack(unittest.TestCase):
    def get(self, index, header, data, name, dtype="BF16"):
        f = index[name]
        self.assertEqual(header[f][name]["dtype"], dtype, name)
        return tensor(header[f], data[f], name) if dtype == "BF16" else None

    def test_pack_stays_bf16(self):
        rng = np.random.default_rng(0)
        gate_up = bf16(rng.standard_normal((E, 2 * I, H)))
        down = bf16(rng.standard_normal((E, H, I)))
        conv = bf16(rng.standard_normal((6, 1, 4)))
        qnorm = bf16(rng.standard_normal(4) * 0.1)
        proj = bf16(rng.standard_normal((40, 64)))  # big enough that the default path would quantize it
        shards = [bf16(rng.standard_normal((ROWS, DIM))) for _ in range(2)]
        pfx = "model.language_model.layers.0."
        tensors = {
            "lm_head.weight": triple(proj),
            pfx + "mlp.experts.gate_up_proj": triple(gate_up),
            pfx + "mlp.experts.down_proj": triple(down),
            pfx + "linear_attn.conv1d.weight": triple(conv),
            pfx + "self_attn.q_norm.weight": triple(qnorm),
            pfx + "ple" + NG[len(".ple"):] + "0.weight": triple(shards[0]),
            pfx + "ple" + NG[len(".ple"):] + "1.weight": triple(shards[1]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            dst, index, header, data = convert(tmp, tensors, "--bf16")
            self.assertFalse([k for k in index if k.endswith((".scales", ".biases"))], "quantized tensors in a bf16 pack")
            get = lambda n: self.get(index, header, data, n)
            np.testing.assert_array_equal(get("language_model.lm_head.weight"), proj)
            base = "language_model.model.layers.0.mlp.switch_mlp."
            np.testing.assert_array_equal(get(base + "gate_proj.weight"), gate_up[:, :I])
            np.testing.assert_array_equal(get(base + "up_proj.weight"), gate_up[:, I:])
            np.testing.assert_array_equal(get(base + "down_proj.weight"), down)
            np.testing.assert_array_equal(get("language_model.model.layers.0.linear_attn.conv1d.weight"),
                                          np.swapaxes(conv, 1, 2))
            np.testing.assert_array_equal(get("language_model.model.layers.0.self_attn.q_norm.weight"),
                                          bf16(bf16_to_f32(qnorm) + 1.0))

            meta, h, d = read_st(dst / "ngram_table.bin")
            self.assertEqual((meta["bits"], meta["format"]), ("16", "mlx-serve-ngram"))
            self.assertEqual(set(h), {"weight"})
            self.assertEqual((h["weight"]["dtype"], h["weight"]["shape"]), ("BF16", [2 * ROWS, DIM]))
            np.testing.assert_array_equal(tensor(h, d, "weight"), np.concatenate(shards))

            cfg = json.loads((dst / "config.json").read_text())
            self.assertNotIn("quantization", cfg)
            self.assertNotIn("quantization_config", cfg)
            self.assertEqual(cfg["ngram_table"]["bits"], 16)

    def test_bf16_spine_with_quantized_experts_and_raw_table(self):
        """Experts 8-bit g32; non-experts, embeddings, lm_head and the n-gram table stay bf16."""
        rng = np.random.default_rng(1)
        i, h = 32, 64
        gate_up = bf16(rng.standard_normal((E, 2 * i, h)))
        down = bf16(rng.standard_normal((E, h, i)))
        spine = {"lm_head.weight": bf16(rng.standard_normal((40, 64))),
                 "model.language_model.embed_tokens.weight": bf16(rng.standard_normal((40, 64))),
                 "model.language_model.layers.0.self_attn.q_proj.weight": bf16(rng.standard_normal((64, 64)))}
        shards = [bf16(rng.standard_normal((ROWS, DIM))) for _ in range(2)]
        pfx = "model.language_model.layers.0."
        tensors = {k: triple(v) for k, v in spine.items()}
        tensors.update({pfx + "mlp.experts.gate_up_proj": triple(gate_up), pfx + "mlp.experts.down_proj": triple(down),
                        pfx + "ple" + NG[len(".ple"):] + "0.weight": triple(shards[0]),
                        pfx + "ple" + NG[len(".ple"):] + "1.weight": triple(shards[1])})
        with tempfile.TemporaryDirectory() as tmp:
            dst, index, header, data = convert(tmp, tensors, "--bits", "8", "--expert-gs", "32", "--nonexpert-bits", "16",
                                               "--embed-bits", "16", "--ngram-bits", "16")
            for name, src in {"language_model.lm_head.weight": spine["lm_head.weight"],
                              "language_model.model.embed_tokens.weight": spine["model.language_model.embed_tokens.weight"],
                              "language_model.model.layers.0.self_attn.q_proj.weight": spine["model.language_model.layers.0.self_attn.q_proj.weight"]}.items():
                np.testing.assert_array_equal(self.get(index, header, data, name), src)
                self.assertNotIn(name[:-len(".weight")] + ".scales", index, name)
            base = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
            f = index[base + ".weight"]
            self.assertEqual(header[f][base + ".weight"]["dtype"], "U32")
            self.assertEqual(header[f][base + ".weight"]["shape"], [E, i, h * 8 // 32])
            self.assertEqual(header[index[base + ".scales"]][base + ".scales"]["shape"], [E, i, h // 32])
            import mlx.core as mx
            mx.set_default_device(mx.cpu)
            w, sc, bi = (mx.array(tensor_u32(header[index[base + k]], data[index[base + k]], base + k)) if k == ".weight"
                         else mx.array(self.get(index, header, data, base + k)).view(mx.bfloat16) for k in (".weight", ".scales", ".biases"))
            deq = mx.dequantize(w, sc, bi, group_size=32, bits=8)
            src = mx.array(np.ascontiguousarray(gate_up[:, :i].reshape(-1, h))).view(mx.bfloat16)
            want = mx.dequantize(*mx.quantize(src, group_size=32, bits=8), group_size=32, bits=8).reshape(E, i, h)
            self.assertTrue(bool(mx.array_equal(deq, want)))

            meta, hh, d = read_st(dst / "ngram_table.bin")
            self.assertEqual((meta["bits"], set(hh)), ("16", {"weight"}))
            np.testing.assert_array_equal(tensor(hh, d, "weight"), np.concatenate(shards))
            cfg = json.loads((dst / "config.json").read_text())
            self.assertEqual(cfg["quantization"]["bits"], 8)
            self.assertEqual(cfg["ngram_table"]["bits"], 16)


def tensor_u32(header, data, name):
    m = header[name]
    b, e = m["data_offsets"]
    return np.frombuffer(data[b:e], np.uint32).reshape(m["shape"])


if __name__ == "__main__":
    unittest.main()
