#!/usr/bin/env python3
"""Kernel checkout bytes must match their manifest even with core.autocrlf enabled."""

import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest


class KernelCheckoutTest(unittest.TestCase):
    def test_windows_checkout_preserves_kernel_bytes(self):
        root = Path(__file__).resolve().parents[1]
        manifest = Path("src/kernels/digests.txt")
        entries = [line.split("  ", 1) for line in (root / manifest).read_text().splitlines()]

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            checkout = Path(directory) / "checkout"
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            files = [Path(".gitattributes"), manifest, *(Path("src") / name for _, name in entries)]
            for relative in files:
                target = repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((root / relative).read_bytes())
            subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=false", "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "-c", "core.autocrlf=true", "checkout-index", "--all", f"--prefix={checkout.as_posix()}/"],
                check=True,
            )

            self.assertEqual((root / manifest).read_bytes(), (checkout / manifest).read_bytes())
            for expected, name in entries:
                with self.subTest(kernel=name):
                    self.assertEqual(expected, hashlib.sha256((checkout / "src" / name).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
