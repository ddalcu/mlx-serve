#!/usr/bin/env python3
"""Exercise the tiny MiMo reference model through HTTP, including MTP rollback.

Run after dump_mimo_v2_fixtures.py and dump_mimo_v2_mtp_fixtures.py.
The temporary tokenizer exists only for this synthetic lifecycle fixture.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture", type=Path)
    ap.add_argument("--bin", type=Path, default=Path("zig-out/bin/mlx-serve"))
    ap.add_argument("--kv-quant", type=int, choices=(0, 8), default=0)
    args = ap.parse_args()
    config = json.loads((args.fixture / "config.json").read_text())
    if config.get("vocab_size") != 128 or config.get("hidden_size") != 384 or config.get("num_hidden_layers") != 4:
        ap.error("fixture must be the tiny reference model (128 vocabulary, 384 hidden, 4 layers)")
    if not (args.fixture / "model_mtp.safetensors").is_file():
        ap.error("run dump_mimo_v2_mtp_fixtures.py first")
    with tempfile.TemporaryDirectory(prefix="mlx-mimo-http-") as td:
        root = Path(td)
        model = root / "tiny-mimo"
        model.mkdir()
        for name in ("config.json", "model.safetensors.index.json", "model.safetensors", "model_mtp.safetensors"):
            shutil.copyfile(args.fixture / name, model / name)
        vocab = {chr(i): i for i in range(2, 128)} | {"<pad>": 0, "<eos>": 1}
        special = [dict(id=i, content=s, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
                   for i, s in enumerate(("<pad>", "<eos>"))]
        (model / "tokenizer.json").write_text(json.dumps(dict(
            version="1.0", added_tokens=special, normalizer=None, pre_tokenizer=None,
            post_processor=None, decoder=None,
            model=dict(type="BPE", unk_token="<pad>", vocab=vocab, merges=[]))))
        (model / "tokenizer_config.json").write_text(json.dumps(dict(
            eos_token="<eos>", pad_token="<pad>", add_bos_token=False,
            chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}")))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        env = os.environ | {"MLX_ENABLE_TF32": "0", "MLX_SERVE_MTP_FORCE_DEPTH": "3"}
        log_path = root / "server.log"
        with log_path.open("w") as log:
            proc = subprocess.Popen([str(args.bin.resolve()), "--model", str(model), "--serve",
                                     "--host", "127.0.0.1", "--port", str(port), "--ctx-size", "1024",
                                     "--no-vision", "--no-drafter", "--log-level", "debug"] +
                                    (["--kv-quant", "8"] if args.kv_quant else ["--kv-quant", "off"]),
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("server exited during load")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1):
                        break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("server startup timed out")
                    time.sleep(0.1)
            outputs = []
            for index, enabled in enumerate((False, True, True, True)):
                if index == 3:
                    unload = urllib.request.Request(base + "/v1/unload-model", data=json.dumps({"model": "tiny-mimo"}).encode(),
                                                    headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(unload, timeout=30) as resp:
                        assert resp.status == 200
                body = dict(model="tiny-mimo", prompt="The quick brown fox jumps over the lazy dog. " * 4,
                            max_tokens=12, temperature=0, enable_mtp=enabled, enable_pld=False,
                            logit_bias={"1": -100}, stream=False)
                req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    out = json.load(resp)
                assert out["usage"]["completion_tokens"] == 12, out
                outputs.append(out["choices"][0]["text"])
            assert all(output == outputs[0] for output in outputs), outputs
            log_text = log_path.read_text()
            assert "[mimo-mtp] 3 heads loaded" in log_text, log_text[-4000:]
            assert "[spec-stats] mode=mtp" in log_text, log_text[-4000:]
            assert "cached /" in log_text, log_text[-4000:]
            print("PASS: tiny MiMo HTTP generation, forced three-head MTP equivalence, warm-cache reuse, and unload/reload")
        except BaseException:
            print(log_path.read_text()[-12000:])
            raise
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
