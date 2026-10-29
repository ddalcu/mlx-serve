#!/usr/bin/env python3
"""Greedy output parity and speed of one GGUF across engines.

Engines: `mlx` (this repo, mlx-serve --mlx-gguf), `llama` (the libllama
mlx-serve embeds, --engine llama), `llamacpp` (a llama-server binary, when
--llama-server is given). Every engine serves the same file on the same
/v1/chat/completions requests at temperature 0; the texts must match and the
timings come from each server's own `timings` object.

    scripts/gguf/parity.py /path/model.gguf
    scripts/gguf/parity.py /path/model.gguf --tokens 128 --llama-server ~/llama.cpp/build/bin/llama-server
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

PROMPTS = [
    "Rename the variable `count` to `total` in this function and return only the code:\n\n"
    "def summarize(items):\n    count = 0\n    for it in items:\n        count += it.value\n    return count / len(items)\n",
    "In three sentences, why does the sky look blue?",
    "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation.",
    "List five prime numbers greater than 100 and explain how you checked one of them.",
]
# A long prompt for prefill timing: the same paragraph repeated.
LONG = ("The Metal shading language runs one simdgroup of 32 threads in lockstep, so a loop costs every thread "
        "the longest trip count of any of them, and a kernel that stripes rows unevenly wastes lanes. ") * 40


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_health(port, proc, timeout):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not come up")


RAW = False  # --raw: /v1/completions on the bare prompt, no chat template (template-sensitive models)


def chat(port, prompt, tokens, thinking, logprobs=False):
    body = {"model": "m", "max_tokens": tokens, "temperature": 0, "seed": 1}
    if RAW:
        body["prompt"] = prompt
        if logprobs:
            body["logprobs"] = 2
    else:
        body["messages"] = [{"role": "user", "content": prompt}]
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 2
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/{'completions' if RAW else 'chat/completions'}",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    wall = time.time() - t0
    choice = out["choices"][0]
    msg = choice.get("message") or {"content": choice.get("text")}
    # A max_tokens cut inside a multi-byte character leaves a U+FFFD on some servers and nothing on others.
    text = (msg.get("content") or "").rstrip("\ufffd")
    if msg.get("reasoning_content"):
        text = "<think>" + msg["reasoning_content"] + "</think>" + text
    usage = out.get("usage", {})
    timings = out.get("timings") or usage.get("timings") or {}
    n = usage.get("completion_tokens", 0)
    # [(token text, margin between the chosen token and the runner-up, nats)]
    steps = []
    lp = choice.get("logprobs") or {}
    if RAW:
        for tok, top in zip(lp.get("tokens") or [], lp.get("top_logprobs") or []):
            alts = sorted(top.values(), reverse=True)
            steps.append((tok, alts[0] - alts[1] if len(alts) > 1 else float("inf")))
    for st in (lp.get("content") or []):
        alts = sorted((a["logprob"] for a in st.get("top_logprobs", [])), reverse=True)
        steps.append((st["token"], alts[0] - alts[1] if len(alts) > 1 else float("inf")))
    return {
        "text": text,
        "steps": steps,
        "n": n,
        "wall": wall,
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_n": timings.get("prompt_n", usage.get("prompt_tokens")),
        "tps": timings.get("predicted_per_second") or (n / wall if wall else 0),
    }


def run_engine(name, cmd, log_dir, model, tokens, thinking, load_timeout, forked=None):
    """`forked(i, text)`: whether prompt i needs a second, logprobs pass (the
    timed pass never carries logprobs, they cost a sync per token)."""
    port = free_port()
    log = open(os.path.join(log_dir, f"{name}.log"), "w")
    proc = subprocess.Popen([a.replace("{port}", str(port)) for a in cmd], stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_health(port, proc, load_timeout)
        chat(port, "warm up", 8, thinking)  # kernels + caches
        results = [chat(port, p, tokens, thinking) for p in PROMPTS]
        long = chat(port, LONG + "\n\nSummarize the paragraph above in one sentence.", 16, thinking)
        for i, r in enumerate(results):
            if forked and forked(i, r["text"]):
                r["steps"] = chat(port, PROMPTS[i], tokens, thinking, logprobs=True)["steps"]
        return results, long
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(__file__), "..", "..", "zig-out", "bin", "mlx-serve"))
    ap.add_argument("--llama-server", help="llama.cpp's llama-server binary, adds the `llamacpp` engine")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--thinking", action="store_true", help="leave the model's thinking on (off by default, keeps the outputs short)")
    ap.add_argument("--load-timeout", type=int, default=600)
    ap.add_argument("--margin", type=float, default=1.0, help="a fork counts as numerics when the mlx top-2 margin there is under this many nats")
    ap.add_argument("--raw", action="store_true", help="bare /v1/completions prompts, no chat template")
    ap.add_argument("--log-dir", default=os.path.expanduser("~/claude-tmp/gguf-prod/parity"))
    ap.add_argument("--engines", default="mlx,llama", help="comma separated subset of mlx,llama,llamacpp")
    args = ap.parse_args()
    global RAW
    RAW = args.raw
    os.makedirs(args.log_dir, exist_ok=True)
    tag = os.path.basename(args.model).replace(".gguf", "")
    log_dir = os.path.join(args.log_dir, tag)
    os.makedirs(log_dir, exist_ok=True)

    engines = {
        "mlx": [args.bin, "--model", args.model, "--serve", "--port", "{port}", "--mlx-gguf", "--no-pld", "--no-drafter", "--no-mtp"],
        "llama": [args.bin, "--model", args.model, "--serve", "--port", "{port}", "--engine", "llama"],
    }
    if args.llama_server:
        engines["llamacpp"] = [args.llama_server, "-m", args.model, "--port", "{port}", "-ngl", "99", "--jinja", "-c", "8192"]
    wanted = [e for e in args.engines.split(",") if e in engines]

    runs = {}
    # mlx last: it can then see the other engines' texts and fetch logprobs only where they fork.
    for name in sorted(wanted, key=lambda n: n == "mlx"):
        print(f"[{tag}] {name} ...", flush=True)
        ref = "llamacpp" if "llamacpp" in runs else next(iter(runs), None)
        forked = (lambda i, text: ref is not None and runs[ref][0][i]["text"] != text) if name == "mlx" else None
        runs[name] = run_engine(name, engines[name], log_dir, args.model, args.tokens, args.thinking, args.load_timeout, forked)

    ok = True
    print(f"\n{tag}: {args.tokens} tokens greedy, {'raw prompts' if RAW else 'thinking ' + ('on' if args.thinking else 'off')}")
    print(f"{'prompt':<8}" + "".join(f"{n + ' tok/s':>16}" for n in wanted) + "   match")
    for i in range(len(PROMPTS)):
        texts = [runs[n][0][i]["text"] for n in wanted]
        match = runs["mlx"][0][i]["text"] == runs["llamacpp" if "llamacpp" in runs else wanted[0]][0][i]["text"] if "mlx" in runs else all(t == texts[0] for t in texts)
        verdict = "yes"
        if not match and "mlx" in runs:
            # Greedy outputs fork where two candidates are nearly tied and the
            # engines round differently; that is numerics, not a bug. Find the
            # mlx token at the fork and its margin over the runner-up.
            # A fresh llama.cpp is the reference when present (the libllama mlx-serve embeds is older).
            ref = "llamacpp" if "llamacpp" in runs else next(n for n in wanted if n != "mlx")
            other = runs[ref][0][i]["text"]
            acc = ""
            margin = None
            for tok, m in runs["mlx"][0][i]["steps"]:
                if not other.startswith(acc + tok):
                    margin = m
                    break
                acc += tok
            near_tie = margin is not None and margin < args.margin
            verdict = f"fork, margin {margin:.2f} nats ({'near tie' if near_tie else 'REAL'})" if margin is not None else "NO"
            match = near_tie
        ok &= match
        print(f"{i:<8}" + "".join(f"{runs[n][0][i]['tps']:>16.1f}" for n in wanted) + f"   {verdict}")
        if not match:
            for n, t in zip(wanted, texts):
                print(f"    {n:<9} {t[:160]!r}")
    print(f"{'prefill':<8}" + "".join(
        f"{(runs[n][1]['prompt_ms'] or 0):>13.0f} ms" for n in wanted) + f"   ({runs['mlx'][1]['prompt_n'] if 'mlx' in runs else '?'} prompt tokens)")
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump({n: {"prompts": r[0], "long": r[1]} for n, r in runs.items()}, f, indent=1)
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
