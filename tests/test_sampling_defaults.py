#!/usr/bin/env python3
"""min_p reaches the sampler from the request body and from generation_config.json.

Bar: the model dir is mirrored with generation_config `min_p: 1` (keep only the
argmax). At temperature 2 with the same seed, an omitted min_p draws the greedy
token, an explicit `min_p: 1` draws it too, and an explicit `min_p: 0` overrides
the default and draws the same non-greedy token as before.

  tests/test_sampling_defaults.py [model_dir] [port]

model_dir also reads SAMPLING_MODEL; defaults to Qwen3.5-0.8B-MLX-4bit.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path.home() / ".mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"
MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SAMPLING_MODEL", DEFAULT_MODEL))
PORT = int(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("SAMPLING_PORT", "11384"))
BINARY = Path(os.environ.get("BINARY", ROOT / "zig-out/bin/mlx-serve"))


def post(path, body, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("sampling test exceeded 240 seconds")
    request = Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=min(45, remaining)) as response:
        return json.load(response)


def completion(prompt, seed, min_p, temperature, deadline):
    body = {
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": temperature,
        "top_p": 1,
        "top_k": 2,
        "seed": seed,
        "logprobs": 2,
    }
    if min_p is not None:
        body["min_p"] = min_p
    return post("/v1/completions", body, deadline)["choices"][0]


def unique_max(choice):
    entries = choice["logprobs"]["top_logprobs"][0]
    scores = sorted((v for v in entries.values() if isinstance(v, (int, float))), reverse=True)
    return len(scores) >= 2 and scores[0] - scores[1] > 1e-4


def run(deadline):
    case = None
    prompts = ("The capital of France is", "Once upon a time,", "The next number after 7 is", "I think that")
    for prompt in prompts:
        for seed in range(4):
            greedy = completion(prompt, seed, 0, 0, deadline)
            sampled = completion(prompt, seed, 0, 2, deadline)
            a, b = greedy["text"], sampled["text"]
            if a.strip() and b.strip() and a != b and unique_max(greedy):
                case = prompt, seed, a, b
                break
        if case:
            break
    assert case is not None, "model produced no discriminating one-token sample"

    prompt, seed, greedy, sampled = case
    omitted = completion(prompt, seed, None, 2, deadline)["text"]
    explicit = completion(prompt, seed, 1, 2, deadline)["text"]
    neutral = completion(prompt, seed, 0, 2, deadline)["text"]
    assert omitted == greedy, ("generation_config min_p default not applied", greedy, omitted)
    assert explicit == greedy, ("request min_p 1 not applied", greedy, explicit)
    assert neutral == sampled, ("request min_p 0 did not override the default", sampled, neutral)
    print(f"PASS: min_p default + explicit + neutral override (prompt={prompt!r} seed={seed})")


def main():
    if not MODEL.is_dir():
        sys.exit(f"FAIL: model dir not found: {MODEL}")
    if not BINARY.is_file():
        sys.exit(f"FAIL: {BINARY} missing; build mlx-serve first")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", PORT))
    with tempfile.TemporaryDirectory(prefix="mlx-serve-sampling-") as temp:
        scratch = Path(temp) / "model"
        scratch.mkdir()
        for item in MODEL.resolve().iterdir():
            if item.name != "generation_config.json":
                (scratch / item.name).symlink_to(item)
        config_path = MODEL / "generation_config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        config["min_p"] = 1
        (scratch / "generation_config.json").write_text(json.dumps(config) + "\n")
        home = Path(temp) / "home"
        home.mkdir()
        log_path = Path(temp) / "server.log"
        deadline = time.monotonic() + 240
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [str(BINARY), "--model", str(scratch), "--serve", "--host", "127.0.0.1", "--port", str(PORT),
                 "--no-pld", "--no-mtp", "--no-drafter", "--prefix-cache-entries", "0"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "HOME": str(home)},
            )
            try:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"server exited {process.returncode}")
                    try:
                        with urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as health:
                            if health.status == 200:
                                break
                    except OSError:
                        time.sleep(0.5)
                else:
                    raise TimeoutError("server did not become healthy")
                run(deadline)
            except Exception:
                print(log_path.read_text(errors="replace")[-4000:], file=sys.stderr)
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
