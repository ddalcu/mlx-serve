#!/usr/bin/env python3
"""Fixed task set for the DSV4 native-mirror vs ds4-GGUF quality comparison.

Runs against an ALREADY-RUNNING server (the two engines must never be loaded
concurrently — double-load OOM); the orchestration boots them sequentially.

Usage: python3 tests/dsv4_task_compare.py --port 18811 --label native --out /tmp/dsv4_native.json
"""
import argparse
import json
import time
import urllib.request

TASKS = [
    ("math", "What is 17 * 23? Answer with just the number.", ["391"]),
    ("fact", "What is the capital of Australia? Answer with just the city name.", ["canberra"]),
    ("chem", "What is the chemical symbol for gold? Answer with just the symbol.", ["au"]),
    ("code", "Write a Python expression that computes the sum of squares of 1 to 10. Reply with only the expression.", ["sum("]),
    ("logic", "Is 97 a prime number? Answer with just yes or no.", ["yes"]),
]


def run(port, label, out_path, max_tokens=40):
    results = []
    for name, prompt, needles in TASKS:
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.loads(r.read())
        dt = time.time() - t0
        text = resp["choices"][0]["message"]["content"]
        ok = any(n.lower() in text.lower() for n in needles)
        results.append({"task": name, "ok": ok, "seconds": round(dt, 1),
                        "text": text, "usage": resp.get("usage"),
                        "timings": resp.get("timings")})
        print(f"[{label}] {name}: {'PASS' if ok else 'FAIL'} ({dt:.0f}s) {text[:80]!r}", flush=True)
    score = sum(r["ok"] for r in results)
    summary = {"label": label, "score": f"{score}/{len(TASKS)}", "results": results}
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{label}] SCORE {score}/{len(TASKS)} -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()
    run(args.port, args.label, args.out, args.max_tokens)
