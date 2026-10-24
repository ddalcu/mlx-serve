#!/usr/bin/env python3
"""Issue #951: unloaded stream packs expose only backend-supported modalities.

Run after a ReleaseFast build: python3 tests/test_stream_pack_capabilities.py
Uses synthetic metadata, no weights, an isolated HOME and its own loopback server.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    binary = Path(os.environ.get("MLX_SERVE_BINARY", "./zig-out/bin/mlx-serve")).resolve()
    # Keep temporary fixtures under the caller's scratch directory when provided.
    with tempfile.TemporaryDirectory(
        prefix="stream-pack-capabilities-", dir=os.environ.get("JCODE_SCRATCH_DIR")
    ) as temporary:
        root = Path(temporary)
        models = root / "models"
        home = root / "home"
        models.mkdir()
        home.mkdir()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        # Loopback requests must not inherit an operator's HTTP proxy.
        client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def request(route, method="GET"):
            req = urllib.request.Request(base + route, method=method)
            with client.open(req, timeout=3) as response:
                return json.load(response)

        log_path = root / "server.log"
        with log_path.open("w") as log:
            process = subprocess.Popen([
                str(binary), "--serve", "--host", "127.0.0.1", "--port", str(port),
                "--model-dir", str(models), "--no-prevent-sleep", "--log-file", "off",
            ], env={**os.environ, "HOME": str(home)}, stdout=log, stderr=log)
            try:
                for _ in range(120):
                    if process.poll() is not None:
                        raise RuntimeError("fixture server exited before readiness")
                    try:
                        request("/health")  # urlopen rejects loading/503 responses.
                        break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(0.25)
                else:
                    raise RuntimeError("fixture server did not become ready")
                assert request("/v1/models")["data"] == [], "HOME/model root not isolated"

                expected = {}
                for model_type in (
                    "deepseek_v41", "glm_moe_dsa", "mimo_v2", "mimo_v2_flash",
                    "glm5_next", "glm5_next_text", "qwen3_5",
                ):
                    for bank, marker in ((False, None), (True, None), (False, {}), (True, {})):
                        name = f"{model_type}-bank{int(bank)}-marker{int(marker is not None)}"
                        directory = models / "fixture" / name
                        directory.mkdir(parents=True)
                        config = {"model_type": model_type, "vision_config": {}, "video_token_id": 42}
                        if marker is not None:
                            config["expert_streaming"] = marker
                        (directory / "config.json").write_text(json.dumps(config))
                        (directory / "chat_template.jinja").write_text("{{ messages }}")
                        (directory / "model.safetensors").write_bytes(b"stub")
                        if bank:
                            (directory / "experts.bin").write_bytes(b"stub")
                        stream = (
                            model_type in ("deepseek_v41", "glm_moe_dsa") and bank
                        ) or (
                            model_type in ("mimo_v2", "mimo_v2_flash", "glm5_next", "glm5_next_text")
                            and marker is not None
                        )
                        expected[f"fixture/{name}"] = not stream and not model_type.endswith("_text")
                assert request("/v1/models/rescan", "POST")["added"] == len(expected)
                rows = {row["id"]: row for row in request("/v1/models")["data"]}
                assert rows.keys() == expected.keys(), "discovery omitted or added fixture rows"
                for model_id, vision in expected.items():
                    row = rows[model_id]
                    assert row["loaded"] is False, model_id
                    assert row["bytes_resident"] == 0, model_id
                    assert ("vision" in row["capabilities"]) == vision, (model_id, row)
                    assert "chat" in row["capabilities"], (model_id, row)
                    modalities = ["text", "image", "video"] if vision else ["text"]
                    assert row["input_modalities"] == modalities, (model_id, row)
                assert request("/v1/models/rescan", "POST")["added"] == 0
                print(f"PASS: {len(expected)} unloaded stream/resident capability cases via HTTP")
            except Exception:
                log.flush()
                print(log_path.read_text()[-6000:])
                raise
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


if __name__ == "__main__":
    main()
