#!/usr/bin/env python3
"""Contract tests for tests/dump_mimo_v2_fixtures.py (the mimo_v2 oracle).

Hermetic half (stdlib only): the fixture SCHEMA the Zig parity test will be
written against, and the torch-missing SKIP path.  E2E half (torch +
transformers + numpy + safetensors): dump a tiny random MiMoV2 twice (second
run --offline over the cached reference), pin byte-determinism of the fixture,
and prove the self-verify accepts its own output and REJECTS a corrupted one.

A passing run prints NOTHING on either stream (repo rule); failure detail goes
to stderr, and MIMO_V2_FIXTURE_DEBUG=1 streams diagnostics (incl. skips).

    python3 tests/test_dump_mimo_v2_fixtures.py
    uv run --with torch --with "transformers==5.3.0" --with numpy --with safetensors \
        tests/test_dump_mimo_v2_fixtures.py
"""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "dump_mimo_v2_fixtures.py"


def load_script():
    spec = importlib.util.spec_from_file_location("dump_mimo_v2_fixtures", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_header(path):
    """safetensors header: (tensor-info dict, metadata dict)."""
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    meta = {k: v for k, v in hdr.get("__metadata__", {}).items()}
    tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}
    return tensors, meta


# The fixture schema IS the contract with the Zig parity test.  Shapes come
# from TINY: T=166 tokens (160 prefill crossing the 128 window + 6 cached
# decode steps), H=384, V=128, 16 experts top-4, rope_dim 64, GA layers carry
# 4 heads and SWA layers 6.
T, H, V, E, K, TD = 166, 384, 128, 16, 4, 6
HEADS = {0: 4, 1: 6, 2: 6, 3: 4}
MOE_LAYERS = (1, 2, 3)

EXPECTED_FIXTURE = {
    "input_ids": ("I32", (T,)),
    "embed_out": ("F32", (T, H)),
    "logits_full": ("F32", (T, V)),
    "logit_margin": ("F32", (T,)),
    "final_norm": ("F32", (T, H)),
    "moe_route_gap": ("F32", (T,)),
    "rope_cos_ga": ("F32", (T, 64)),
    "rope_sin_ga": ("F32", (T, 64)),
    "rope_cos_swa": ("F32", (T, 64)),
    "rope_sin_swa": ("F32", (T, 64)),
    "cache_logits": ("F32", (T, V)),
    "cache_final_norm": ("F32", (T, H)),
}
for i in range(4):
    EXPECTED_FIXTURE[f"stream_{i}"] = ("F32", (T, H))
    EXPECTED_FIXTURE[f"l{i}_attn_out"] = ("F32", (T, H))
    EXPECTED_FIXTURE[f"l{i}_mlp_out"] = ("F32", (T, H))
    EXPECTED_FIXTURE[f"l{i}_attn_probs"] = ("F32", (HEADS[i], T, T))
    EXPECTED_FIXTURE[f"l{i}_attn_sink"] = ("F32", (HEADS[i], T))
    EXPECTED_FIXTURE[f"l{i}_vis_from"] = ("I32", (T,))
    EXPECTED_FIXTURE[f"cache_stream_{i}"] = ("F32", (T, H))
    EXPECTED_FIXTURE[f"cache_l{i}_attn_out"] = ("F32", (T, H))
    EXPECTED_FIXTURE[f"cache_l{i}_mlp_out"] = ("F32", (T, H))
    EXPECTED_FIXTURE[f"cache_l{i}_attn_probs_dec"] = ("F32", (HEADS[i], TD, T))
    EXPECTED_FIXTURE[f"cache_l{i}_attn_sink_dec"] = ("F32", (HEADS[i], TD))
    EXPECTED_FIXTURE[f"cache_l{i}_vis_from_dec"] = ("I32", (TD,))
    if i in MOE_LAYERS:
        EXPECTED_FIXTURE[f"l{i}_moe_scores"] = ("F32", (T, E))
        EXPECTED_FIXTURE[f"l{i}_moe_topk_idx"] = ("I32", (T, K))
        EXPECTED_FIXTURE[f"l{i}_moe_topk_w_pre"] = ("F32", (T, K))
        EXPECTED_FIXTURE[f"l{i}_moe_topk_w_post"] = ("F32", (T, K))
        EXPECTED_FIXTURE[f"l{i}_moe_rank_gaps"] = ("F32", (T, E - 1))
        EXPECTED_FIXTURE[f"l{i}_moe_route_gap"] = ("F32", (T,))
        EXPECTED_FIXTURE[f"cache_l{i}_moe_scores_dec"] = ("F32", (TD, E))
        EXPECTED_FIXTURE[f"cache_l{i}_moe_topk_idx_dec"] = ("I32", (TD, K))
        EXPECTED_FIXTURE[f"cache_l{i}_moe_topk_w_pre_dec"] = ("F32", (TD, K))
        EXPECTED_FIXTURE[f"cache_l{i}_moe_topk_w_post_dec"] = ("F32", (TD, K))
EXPECTED_FIXTURE["stream_4"] = ("F32", (T, H))
EXPECTED_FIXTURE["cache_stream_4"] = ("F32", (T, H))

EXPECTED_WEIGHTS = {
    "model.embed_tokens.weight": (V, H),
    "model.norm.weight": (H,),
    "lm_head.weight": (V, H),
}
for i in range(4):
    ga = i in (0, 3)          # hybrid_layer_pattern: 0 = GA/global (sink off)
    heads, kv = (4, 2) if ga else (6, 2)
    EXPECTED_WEIGHTS[f"model.layers.{i}.input_layernorm.weight"] = (H,)
    EXPECTED_WEIGHTS[f"model.layers.{i}.post_attention_layernorm.weight"] = (H,)
    # fused_qkv layout: one projection carrying q (head_dim 192) + k (192) + v (128)
    EXPECTED_WEIGHTS[f"model.layers.{i}.self_attn.qkv_proj.weight"] = (heads * 192 + kv * 192 + kv * 128, H)
    EXPECTED_WEIGHTS[f"model.layers.{i}.self_attn.o_proj.weight"] = (H, heads * 128)
    if not ga:                # add_swa_attention_sink_bias: per-head sink, SWA only
        EXPECTED_WEIGHTS[f"model.layers.{i}.self_attn.attention_sink_bias"] = (heads,)
    if i == 0:                # moe_layer_freq[0] = 0: the dense SwiGLU
        EXPECTED_WEIGHTS["model.layers.0.mlp.gate_proj.weight"] = (1536, H)
        EXPECTED_WEIGHTS["model.layers.0.mlp.up_proj.weight"] = (1536, H)
        EXPECTED_WEIGHTS["model.layers.0.mlp.down_proj.weight"] = (H, 1536)
    else:
        EXPECTED_WEIGHTS[f"model.layers.{i}.mlp.gate.weight"] = (E, H)
        EXPECTED_WEIGHTS[f"model.layers.{i}.mlp.gate.e_score_correction_bias"] = (E,)
        for e in range(E):
            EXPECTED_WEIGHTS[f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"] = (192, H)
            EXPECTED_WEIGHTS[f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"] = (192, H)
            EXPECTED_WEIGHTS[f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"] = (H, 192)


class SchemaContractTests(unittest.TestCase):
    """No torch: the schema helper and the rope-dim math (pure stdlib)."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_script()

    def test_module_imports_without_torch(self):
        self.assertNotIn("torch", self.mod.__dict__,
                         "torch must be imported lazily, not at module import")

    def test_rope_dim_math(self):
        # int(192 * 0.334) = 64: the partial-rotary width the Zig arm must use.
        self.assertEqual(self.mod.rope_dim_for(192, 0.334), 64)
        self.assertEqual(self.mod.rope_dim_for(192, 1.0), 192)
        with self.assertRaises(self.mod.FixtureError):
            self.mod.rope_dim_for(191, 0.5)

    def test_fixture_schema_matches_the_zig_contract(self):
        schema = self.mod.fixture_schema(self.mod.tiny_dims())
        self.assertEqual(set(schema), set(EXPECTED_FIXTURE))
        for key, (dtype, shape) in EXPECTED_FIXTURE.items():
            self.assertEqual(schema[key], (dtype, tuple(shape)), key)

    def test_weight_schema_matches_the_checkpoint_naming(self):
        schema = self.mod.weight_schema(self.mod.tiny_dims())
        self.assertEqual(set(schema), set(EXPECTED_WEIGHTS))
        for key, shape in EXPECTED_WEIGHTS.items():
            self.assertEqual(tuple(schema[key]), tuple(shape), key)


class SkipWithoutTorchTests(unittest.TestCase):
    """Missing torch: a loud SKIP-style refusal, never faked fixtures."""

    def test_missing_torch_skips_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            blocker = Path(td) / "blocker"
            blocker.mkdir()
            (blocker / "torch.py").write_text("raise ImportError('blocked for test')\n")
            out = Path(td) / "out"
            env = dict(os.environ, PYTHONPATH=str(blocker))
            r = subprocess.run([sys.executable, str(SCRIPT), str(out)],
                               capture_output=True, env=env, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertTrue(r.stderr.startswith("SKIP"), r.stderr)
            self.assertFalse(out.joinpath("fixture.safetensors").exists())


class DumpEndToEndTests(unittest.TestCase):
    """Needs torch + transformers; SKIPs as a class without them."""

    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
            raise unittest.SkipTest("torch/transformers unavailable")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.cache = cls.root / "ref-cache"
        cls.out1 = cls.root / "out1"
        cls.out2 = cls.root / "out2"
        r = subprocess.run([sys.executable, str(SCRIPT), str(cls.out1),
                            "--seed", "1234", "--cache-dir", str(cls.cache)],
                           capture_output=True, text=True)
        cls.dump1 = r
        if r.returncode != 0:
            raise AssertionError(f"dump failed rc={r.returncode}\n{r.stdout}\n{r.stderr}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def test_outputs_exist_with_exact_schema(self):
        for name in ("config.json", "model.safetensors", "model.safetensors.index.json", "fixture.safetensors"):
            self.assertTrue((self.out1 / name).is_file(), name)
        tensors, meta = read_header(self.out1 / "fixture.safetensors")
        self.assertEqual({k: (v["dtype"], tuple(v["shape"])) for k, v in tensors.items()},
                         {k: (d, tuple(s)) for k, (d, s) in EXPECTED_FIXTURE.items()})
        self.assertIn("mimo_v2_fixture", meta)
        info = json.loads(meta["mimo_v2_fixture"])
        self.assertEqual(info["schema_version"], 1)
        margins = info["margins"]
        for key in ("logits_max_abs", "streams_max_abs", "bound", "logit_margin_min", "moe_route_gap_min"):
            self.assertIn(key, margins)
        self.assertLessEqual(margins["logits_max_abs"], margins["bound"])
        self.assertLessEqual(margins["streams_max_abs"], margins["bound"])
        wtensors, _ = read_header(self.out1 / "model.safetensors")
        self.assertEqual({k: tuple(v["shape"]) for k, v in wtensors.items()},
                         {k: tuple(s) for k, s in EXPECTED_WEIGHTS.items()})
        index = json.loads((self.out1 / "model.safetensors.index.json").read_text())
        self.assertEqual(set(index["weight_map"]), set(EXPECTED_WEIGHTS))
        cfg = json.loads((self.out1 / "config.json").read_text())
        self.assertEqual(cfg["model_type"], "mimo_v2")
        self.assertEqual(cfg["head_dim"], 192)
        self.assertEqual(cfg["v_head_dim"], 128)
        self.assertEqual(cfg["partial_rotary_factor"], 0.334)

    def test_self_verify_accepts_its_own_output(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--verify", str(self.out1)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_self_verify_rejects_a_corrupted_fixture(self):
        from safetensors.numpy import load_file, save_file
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad"
            shutil.copytree(self.out1, bad)
            arrs = load_file(str(bad / "fixture.safetensors"))
            arrs["l1_attn_sink"] = arrs["l1_attn_sink"] + 0.25  # breaks sum(keys)+sink == 1
            _, meta = read_header(bad / "fixture.safetensors")
            save_file(arrs, str(bad / "fixture.safetensors"), metadata=meta)
            r = subprocess.run([sys.executable, str(SCRIPT), "--verify", str(bad)],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("attn", r.stderr)

    def test_self_verify_rejects_nonfinite_stream_and_attention(self):
        import numpy as np
        from safetensors.numpy import load_file, save_file
        cases = (("fixture.safetensors", "stream_1", (0, 0), np.nan),
                 ("fixture.safetensors", "l1_attn_probs", (0, 0, 0), np.inf),
                 ("model.safetensors", "model.embed_tokens.weight",
                  (0, 0), np.nan))
        for filename, tensor, index, bad_value in cases:
            with self.subTest(tensor=tensor):
                with tempfile.TemporaryDirectory() as td:
                    bad = Path(td) / "bad"
                    shutil.copytree(self.out1, bad)
                    arrs = load_file(str(bad / filename))
                    corrupted = np.array(arrs[tensor], copy=True)
                    corrupted[index] = bad_value
                    arrs[tensor] = corrupted
                    _, meta = read_header(bad / filename)
                    save_file(arrs, str(bad / filename),
                              metadata=meta)
                    r = subprocess.run(
                        [sys.executable, str(SCRIPT), "--verify", str(bad)],
                        capture_output=True, text=True)
                    self.assertNotEqual(r.returncode, 0)
                    self.assertIn("non-finite", r.stderr)
                    self.assertIn(tensor, r.stderr)

    def test_self_verify_rejects_nonfinite_margin_metadata(self):
        from safetensors.numpy import load_file, save_file
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad"
            shutil.copytree(self.out1, bad)
            arrs = load_file(str(bad / "fixture.safetensors"))
            _, meta = read_header(bad / "fixture.safetensors")
            info = json.loads(meta["mimo_v2_fixture"])
            info["margins"]["bound"] = float("nan")
            meta["mimo_v2_fixture"] = json.dumps(info)
            save_file(arrs, str(bad / "fixture.safetensors"), metadata=meta)
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "--verify", str(bad)],
                capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("non-finite", r.stderr)
            self.assertIn("margins.bound", r.stderr)

    def test_offline_cache_hit_is_byte_identical(self):
        # Second run over the cached reference, hermetic (--offline): the
        # fixture must be byte-identical (seeded weights + seeded prompt).
        r = subprocess.run([sys.executable, str(SCRIPT), str(self.out2),
                            "--seed", "1234", "--offline", "--cache-dir", str(self.cache)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        for name in ("fixture.safetensors", "model.safetensors"):
            a = hashlib.sha256((self.out1 / name).read_bytes()).hexdigest()
            b = hashlib.sha256((self.out2 / name).read_bytes()).hexdigest()
            self.assertEqual(a, b, name)

    def test_offline_without_reference_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            r = subprocess.run([sys.executable, str(SCRIPT), str(out),
                                "--offline", "--cache-dir", str(Path(td) / "empty-cache")],
                           capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse(out.joinpath("fixture.safetensors").exists())

    def test_ref_path_copies_the_reference_into_a_fresh_cache(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            fresh = Path(td) / "fresh-cache"
            r = subprocess.run([sys.executable, str(SCRIPT), str(out),
                                "--seed", "1234", "--offline", "--ref", str(self.cache / "mimo_v2_ref"),
                                "--cache-dir", str(fresh)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            for name in ("modeling_mimo_v2.py", "configuration_mimo_v2.py"):
                self.assertTrue((fresh / "mimo_v2_ref" / name).is_file(), name)
            self.assertTrue(out.joinpath("fixture.safetensors").is_file())


if __name__ == "__main__":
    debug = os.environ.get("MIMO_V2_FIXTURE_DEBUG")
    buf = io.StringIO()
    stream = sys.stderr if debug else buf
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(stream=stream, verbosity=2 if debug else 1).run(suite)
    if not result.wasSuccessful() and stream is buf:
        sys.stderr.write(buf.getvalue())
    if not result.wasSuccessful():
        sys.exit(1)
    if result.skipped:
        details = "; ".join(f"{test}: {reason}"
                            for test, reason in result.skipped)
        sys.stderr.write(f"SKIP: {len(result.skipped)} test(s) skipped")
        if details:
            sys.stderr.write(f": {details}")
        sys.stderr.write("\n")
        sys.exit(2)
    sys.exit(0 if result.wasSuccessful() else 1)