#!/usr/bin/env python3
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
MODEL = os.environ.get("SAMPLING_MODEL")
PORT = int(os.environ.get("SAMPLING_PORT", "11384"))
BINARY = Path(os.environ.get("MLX_SERVE_BIN", ROOT / "zig-out/bin/mlx-serve"))


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
    scores = sorted((value for value in entries.values() if isinstance(value, (int, float))), reverse=True)
    return len(scores) >= 2 and scores[0] - scores[1] > 1e-4


def penalized_completion(**overrides):
    body = {
        "messages": [{"role": "user", "content": "Write the word apple 40 times separated by spaces. Here is the start: " + "apple " * 8}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 48,
        "temperature": 0,
        "min_p": 0,
        **overrides,
    }
    return body


def run(deadline):
    case = None
    prompts = ("The capital of France is", "Once upon a time,", "The next number after 7 is", "I think that")
    for prompt in prompts:
        for seed in range(4):
            greedy = completion(prompt, seed, 0, 0, deadline)
            neutral = completion(prompt, seed, 0, 2, deadline)
            a, b = greedy["text"], neutral["text"]
            if a.strip() and b.strip() and a != b and unique_max(greedy):
                case = prompt, seed, a, b
                break
        if case:
            break
    if case is not None:
        prompt, seed, greedy, neutral = case
        omitted = completion(prompt, seed, None, 2, deadline)["text"]
        explicit = completion(prompt, seed, 1, 2, deadline)["text"]
        repeated_neutral = completion(prompt, seed, 0, 2, deadline)["text"]
        assert (omitted, explicit, repeated_neutral) == (greedy, greedy, neutral), (
            "min_p default or explicit neutral override failed",
            (greedy, neutral, omitted, explicit, repeated_neutral),
        )

    response = {"input": "Hi", "max_output_tokens": 1, "temperature": 0}
    inherited = post("/v1/responses", response, deadline)
    neutral_response = post("/v1/responses", {**response, "presence_penalty": 0}, deadline)
    assert inherited["presence_penalty"] == 2, inherited["presence_penalty"]
    assert neutral_response["presence_penalty"] == 0, neutral_response["presence_penalty"]

    neutral = post("/v1/chat/completions", penalized_completion(repeat_penalty=1, presence_penalty=0), deadline)["choices"][0]["message"]["content"]
    repeat_default = post("/v1/chat/completions", penalized_completion(presence_penalty=0), deadline)["choices"][0]["message"]["content"]
    repeat_explicit = post("/v1/chat/completions", penalized_completion(repeat_penalty=4, presence_penalty=0), deadline)["choices"][0]["message"]["content"]
    assert repeat_default == repeat_explicit and repeat_default != neutral, (
        "model repetition default did not shape multi-token output",
        (neutral, repeat_default, repeat_explicit),
    )
    presence_default = post("/v1/chat/completions", penalized_completion(repeat_penalty=1), deadline)["choices"][0]["message"]["content"]
    presence_explicit = post("/v1/chat/completions", penalized_completion(repeat_penalty=1, presence_penalty=2), deadline)["choices"][0]["message"]["content"]
    assert presence_default == presence_explicit and presence_default != neutral, (
        "model presence default did not shape multi-token output",
        (neutral, presence_default, presence_explicit),
    )
    assert case is not None, "model produced no discriminating one-token min_p sample"
    print("PASS: min_p, presence and repetition defaults; explicit and neutral overrides")


def main():
    if not MODEL:
        print("SKIP: set SAMPLING_MODEL to a small MLX model directory")
        return
    source = Path(MODEL).resolve()
    if not source.is_dir() or not BINARY.is_file():
        raise ValueError("SAMPLING_MODEL directory and MLX_SERVE_BIN must exist")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", PORT))
    with tempfile.TemporaryDirectory(prefix="mlx-serve-sampling-") as temp:
        scratch = Path(temp) / "model"
        scratch.mkdir()
        for item in source.iterdir():
            if item.name != "generation_config.json":
                (scratch / item.name).symlink_to(item)
        config_path = source / "generation_config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        config.update(min_p=1, presence_penalty=2, repetition_penalty=4)
        (scratch / "generation_config.json").write_text(json.dumps(config) + "\n")
        log_path = Path(temp) / "server.log"
        deadline = time.monotonic() + 240
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [str(BINARY), "--model", str(scratch), "--serve", "--host", "127.0.0.1", "--port", str(PORT),
                 "--no-pld", "--no-mtp", "--no-drafter", "--prefix-cache-entries", "0"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
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
