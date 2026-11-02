"""Hermetic tests for EXL3 pack-fixture safety in the disk-cache probe."""

import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest


PROBE_PATH = Path(__file__).with_name("module_prefix_cache_disk_probe.py")
SPEC = importlib.util.spec_from_file_location("module_prefix_cache_disk_probe", PROBE_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class Exl3FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
        (self.source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"trunk.weight": "model.safetensors"}}))
        (self.source / "model.safetensors").write_bytes(b"selected trunk weights")
        (self.source / "experts.bin").write_bytes(b"disposable bank input" * 4096)
        (self.source / "expert-manifest.json").write_text("{}\n")
        (self.source / "expert-manifest-v2.json").write_text("{\"version\":2}\n")

    def tearDown(self):
        self.temp.cleanup()

    def make_fixture(self, name, identity_file):
        dest = self.root / name
        args = type("Args", (), {"model": str(self.source), "dest": str(dest),
                                  "identity_file": identity_file})()
        PROBE.fixture(args)
        return dest

    def test_exl3_fixture_keeps_bank_and_manifests_linked_by_default(self):
        dest = self.make_fixture("linked", None)
        self.assertTrue((dest / "experts.bin").is_symlink())
        self.assertTrue((dest / "expert-manifest.json").is_symlink())
        self.assertFalse((dest / "expert-manifest-v2.json").is_symlink())
        stamp = json.loads((dest / "module-test-fixture.json").read_text())
        self.assertEqual(stamp["name"], "expert-manifest-v2.json")
        self.assertFalse((dest / stamp["name"]).is_symlink())

    def test_manifest_stat_mutation_changes_only_the_wrapper_copy(self):
        source = self.source / "expert-manifest-v2.json"
        source_stat = source.stat().st_mtime_ns
        dest = self.make_fixture("manifest", "expert-manifest-v2.json")
        copy = dest / "expert-manifest-v2.json"
        self.assertFalse(copy.is_symlink())
        args = type("Args", (), {"pack": str(dest)})()
        PROBE.touch_pack(args)
        self.assertNotEqual(copy.stat().st_mtime_ns, source_stat)
        self.assertEqual(source.stat().st_mtime_ns, source_stat)
        PROBE.restore_stat(args)
        self.assertEqual(copy.stat().st_mtime_ns, source_stat)
        self.assertEqual(source.stat().st_mtime_ns, source_stat)

    def test_bank_stat_mutation_uses_only_a_disposable_clone(self):
        source = self.source / "experts.bin"
        source_stat = source.stat().st_mtime_ns
        dest = self.make_fixture("bank", "experts.bin")
        copy = dest / "experts.bin"
        if copy.is_symlink():
            self.skipTest("filesystem does not support copy-on-write clones")
        self.assertEqual(copy.read_bytes(), source.read_bytes())
        args = type("Args", (), {"pack": str(dest)})()
        PROBE.touch_pack(args)
        self.assertNotEqual(copy.stat().st_mtime_ns, source_stat)
        self.assertEqual(source.stat().st_mtime_ns, source_stat)

    def test_large_streamed_bank_does_not_become_a_ram_threshold(self):
        shards = sum(path.stat().st_size for path in self.source.glob("*.safetensors"))
        bank_size = 256 << 30
        original_stat = Path.stat
        bank_stat_calls = []

        def stat_with_large_bank(path, *args, **kwargs):
            value = original_stat(path, *args, **kwargs)
            if path.name == "experts.bin":
                bank_stat_calls.append(path)
                return SimpleNamespace(st_mode=value.st_mode, st_size=bank_size)
            return value

        with patch.object(Path, "stat", stat_with_large_bank):
            need = PROBE.required_model_bytes(self.source)
        self.assertEqual(need, shards + (8 << 30))
        self.assertLess(need, bank_size)
        self.assertEqual(bank_stat_calls, [])

    def test_success_receipt_retains_case_summary_and_is_private(self):
        log = self.root / "driver.log"
        log.write_text("PASS: ram A/B/A\nSKIP: optional old reader\n")
        binary = self.root / "mlx-serve"
        binary.write_bytes(b"synthetic executable")
        output = self.root / "receipts" / "run.json"
        args = type("Args", (), {
            "log": str(log), "output": str(output), "exit_code": 0,
            "started_utc": "2026-10-29T00:00:00Z", "model": str(self.source),
            "binary": str(binary), "ram_budget": "8GB", "disk_budget": "100GB",
        })()
        PROBE.receipt(args)
        row = json.loads(output.read_text())
        self.assertEqual(row["status"], "passed")
        self.assertEqual(row["passed_cases"], ["PASS: ram A/B/A"])
        self.assertEqual(row["skipped_cases"], ["SKIP: optional old reader"])
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        self.assertEqual(output.parent.stat().st_mode & 0o777, 0o700)

    def test_request_forwards_bearer_and_generic_headers_without_printing_values(self):
        output = self.root / "request.json"
        captured = {}

        def fake_chat(base, messages, **kwargs):
            request, body = PROBE.build_chat_request(base, messages, **kwargs)
            captured["request"] = request
            return {"usage": {"prompt_tokens": 20,
                              "prompt_tokens_details": {"cached_tokens": 3}},
                    "ttft_s": 0.125}

        args = SimpleNamespace(base="http://127.0.0.1:11234",
                               request_url="http://127.0.0.1:8090",
                               api_key_env="PROBE_TEST_SECRET", seed=None,
                               conversation_id=None, max_tokens=8,
                               header=["X-LLM-Backend: flash-next"], output=str(output))
        stdout = io.StringIO()
        with patch.dict(os.environ, {"PROBE_TEST_SECRET": "private-test-token"}), \
             patch.object(PROBE, "chat", side_effect=fake_chat), \
             contextlib.redirect_stdout(stdout):
            PROBE.request(args)

        request = captured["request"]
        self.assertEqual(request.full_url, "http://127.0.0.1:8090/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer private-test-token")
        self.assertEqual(request.get_header("X-llm-backend"), "flash-next")
        self.assertNotIn("private-test-token", stdout.getvalue())
        self.assertNotIn("flash-next", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
