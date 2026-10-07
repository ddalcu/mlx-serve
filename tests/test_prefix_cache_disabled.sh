#!/usr/bin/env bash
# Usage: test_prefix_cache_disabled.sh MODEL [PORT] [extra server flags...]
set -euo pipefail
cd "$(dirname "$0")/.."
source tests/_lib_models.sh
model="${1:-$(find_model mlx-community/Qwen3.5-0.8B-MLX-4bit)}"
[ -n "$model" ] || { echo "SKIP test_prefix_cache_disabled: no model (pass <model-dir>)"; exit 0; }
port="${2:-19286}"
shift "$(( $# >= 2 ? 2 : $# ))"
python3 - "${BINARY:-./zig-out/bin/mlx-serve}" "$model" "$port" "$@" <<'PY'
import json
import subprocess
import sys
import tempfile
import time
import urllib.request

binary, model, port, *extra = sys.argv[1:]
base = f"http://127.0.0.1:{port}"
with tempfile.TemporaryFile() as log:
    proc = subprocess.Popen([
        binary, "--model", model, "--serve", "--host", "127.0.0.1",
        "--port", port, "--no-vision", "--no-mtp",
        "--prefix-cache-entries", "0", "--prefix-cache-mem", "60GB",
        *extra,
    ], stdout=log, stderr=log)
    try:
        for _ in range(240):
            if proc.poll() is not None:
                raise RuntimeError("server exited before becoming ready")
            try:
                with urllib.request.urlopen(base + "/props", timeout=2) as r:
                    props = json.load(r)
                if "settings" in props:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError("server did not become ready")
        for _ in range(3):
            with urllib.request.urlopen(base + "/props", timeout=5) as r:
                props = json.load(r)
            assert props["settings"]["prefix_cache"]["mem_bytes"] == 0, props
            request = urllib.request.Request(
                base + "/v1/completions",
                data=json.dumps({"model": "default", "prompt": "Hello", "max_tokens": 2}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as r:
                response = json.load(r)
            assert response.get("choices"), response
        print("PASS: disabled prefix cache reports zero RAM and serves three requests")
    except BaseException:
        log.seek(0)
        print(log.read().decode(errors="replace"), file=sys.stderr)
        raise
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
PY
