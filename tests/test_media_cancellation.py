"""Live cancellation/concurrency smoke checks for the local media branch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
import urllib.request

from test_media_history import ask, image, media_turn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "system", "content": "Inspect images honestly."},
                media_turn(image((255, 0, 0)), "Describe the image, then list 500 numbered observations.")]
    body = {"model": "mlx-serve", "messages": messages, "max_tokens": 4096,
            "temperature": 0, "enable_thinking": False, "stream": True}
    request = urllib.request.Request(args.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    chunks = 0
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            value = json.loads(line[6:])
            if any(c.get("delta", {}).get("content") for c in value.get("choices", [])):
                chunks += 1
                if chunks == 10:
                    break
    assert chunks == 10, "Generation ended before the cancellation point"
    deadline = time.monotonic() + 30
    while True:
        with urllib.request.urlopen(args.url + "/metrics", timeout=5) as response:
            metrics = dict(line.split() for line in response.read().decode().splitlines() if line and not line.startswith("#"))
        if float(metrics["vllm:num_requests_running"]) == 0:
            break
        assert time.monotonic() < deadline, "Cancelled slot did not retire"
        time.sleep(0.2)
    messages.append({"role": "user", "content": "Stop enumerating. What color is the image? One word."})
    after = ask(args.url, "mlx-serve", messages)
    assert re.search(r"\bred\b", after["answer"], re.I), after
    assert after["cached"] > 0, after
    def concurrent(rgb):
        return ask(args.url, "mlx-serve", [media_turn(image(rgb), "What color is the image? One word.")])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(concurrent, [(0, 0, 255), (255, 0, 0)]))
    for result, color in zip(results, ["blue", "red"]):
        assert re.search(r"\b" + color + r"\b", result["answer"], re.I), result
    evidence = {"cancelled_after_chunks": chunks, "followup": after, "concurrent": results}
    (args.output / "cancellation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
