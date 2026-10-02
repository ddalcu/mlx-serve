#!/usr/bin/env python3
"""Linux llama KV regression; set LLAMA_GGUF_MODEL to a local instruct GGUF."""
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def request(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def run(binary, model, mode, headless, logs, flags=None):
    label = "headless-alias" if flags else ("headless-8" if headless else (mode or "default"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log_path = logs / f"{label}.log"
    with tempfile.TemporaryDirectory(prefix="llama-kv-empty-") as empty, log_path.open("w") as log:
        if headless:
            fixture = Path(empty) / "fixture"
            fixture.mkdir()
            (fixture / "model.gguf").symlink_to(model)
        cmd = [str(binary), "--serve", "--host", "127.0.0.1", "--port", str(port),
               "--ctx-size", "4096", "--log-file", "off", "--no-prevent-sleep"]
        cmd += ["--model-dir", empty] if headless else ["--model", str(model)]
        if flags:
            cmd += flags
        elif mode is not None:
            cmd += ["--kv-quant", mode]
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                assert proc.poll() is None, f"{label}: server exited; see {log_path}"
                try:
                    request(base, "/health")
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.2)
            else:
                raise AssertionError(f"{label}: startup timed out")
            model_id = request(base, "/v1/models")["data"][0]["id"]
            if headless:
                request(base, "/v1/load-model", {"model": model_id})
            response = request(base, "/v1/chat/completions", {
                "model": model_id, "messages": [{"role": "user", "content": "Reply with exactly the single word hello."}],
                "temperature": 0, "max_tokens": 24,
                # llama contexts keep their load-time type despite a body override.
                "kv_quant": 4 if mode == "8" else 8,
            })
            assert response["choices"][0]["message"]["content"].strip(), response
            models = request(base, "/v1/models")["data"]
            loaded = [m for m in models if m.get("loaded")]
            assert len(loaded) == 1 and loaded[0]["meta"]["kv_quant"] == (mode or "off"), loaded
            (logs / f"{label}.json").write_text(json.dumps(response, indent=2) + "\n")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    text = log_path.read_text()
    expected = {None: "f16", "off": "f16", "8": "q8_0", "4": "q4_0"}[mode]
    caches = re.findall(r"llama_kv_cache[^\n]*size =\s*([\d.]+) MiB[^\n]*K \((\w+)\):[^\n]*V \((\w+)\):", text)
    assert caches, f"{label}: no KV allocation report; see {log_path}"
    assert all(k == v == expected for _, k, v in caches), f"{label}: expected {expected}, got {caches}"
    if mode in ("4", "8"):
        assert re.search(r"flash_attn\s*=\s*enabled", text), f"{label}: flash attention not enabled"
    size = sum(float(size) for size, _, _ in caches)
    print(f"PASS {label}: K/V={expected}, KV={size:.2f} MiB, content={response['choices'][0]['message']['content']!r}", flush=True)
    return size


def main():
    model = os.environ.get("LLAMA_GGUF_MODEL")
    if sys.platform != "linux" or not model:
        print("SKIP: Linux + LLAMA_GGUF_MODEL required")
        return
    # Keep a .gguf symlink's spelling: engine routing depends on the extension.
    model = Path(os.path.abspath(model))
    assert model.is_file(), model
    binary = Path(os.environ.get("MLX_SERVE_BINARY", "zig-out/bin/mlx-serve")).resolve()
    with tempfile.TemporaryDirectory(prefix="llama-kv-test-") as tmp:
        logs = Path(os.environ.get("KV_QUANT_LOG_DIR", tmp))
        logs.mkdir(parents=True, exist_ok=True)
        q8 = run(binary, model, "8", False, logs)
        f16 = run(binary, model, "off", False, logs)
        q4 = run(binary, model, "4", False, logs)
        assert 0 < q4 < q8 < f16, (q4, q8, f16)
        assert run(binary, model, "8", True, logs) == q8
        assert run(binary, model, "8", True, logs, ["--kv-quant", "4", "--llama-kv-quant", "8"]) == q8
        assert run(binary, model, None, False, logs) == f16
        print(f"PASS KV reduction: Q8 {100 * (1 - q8 / f16):.1f}%, Q4 {100 * (1 - q4 / f16):.1f}%")


if __name__ == "__main__":
    main()
