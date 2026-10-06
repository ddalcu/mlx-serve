#!/usr/bin/env python3
"""Kolibri-1 (kolibri1) live end-to-end on an MLX pack.

Boots the pack, then checks: the model is advertised, a short greedy answer,
thinking-off really answers (no reasoning text), a tool call, a prompt past the
513-key sliding window (a dense fallback would still answer, so the fused router
engagement is read from the server's own log line). SKIPs without the pack.

  KOLIBRI1_MODEL=<pack dir> python3 tests/test_kolibri1.py [port]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL = os.environ.get("KOLIBRI1_MODEL", "")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 11412
BIN = os.environ.get("MLX_SERVE_BIN", "./zig-out/bin/mlx-serve")
LOG = os.path.expanduser(f"~/claude-tmp/kolibri1-live/server-{PORT}.log")
URL = f"http://127.0.0.1:{PORT}"

if not os.path.isfile(os.path.join(MODEL, "config.json")):
    print(f"SKIP: no pack at {MODEL!r} (set KOLIBRI1_MODEL)")
    sys.exit(0)


def call(path, body=None, timeout=600):
    req = urllib.request.Request(
        URL + path, json.dumps(body).encode() if body else None, {"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(content, **extra):
    body = {"model": "k", "messages": [{"role": "user", "content": content}], "temperature": 0,
            "max_tokens": 64, "reasoning_effort": "none", **extra}
    return call("/v1/chat/completions", body)["choices"][0]


os.makedirs(os.path.dirname(LOG), exist_ok=True)
server = subprocess.Popen(
    [BIN, "--model", MODEL, "--serve", "--host", "127.0.0.1", "--port", str(PORT),
     "--ctx-size", "8192", "--prefill-chunk", "1024"],
    stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
)
passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    passed, failed = passed + bool(ok), failed + (not ok)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


try:
    for _ in range(240):
        try:
            if call("/health", timeout=2).get("status") == "ok":
                break
        except Exception:
            time.sleep(1)
    else:
        sys.exit(f"server did not come up, see {LOG}")

    ids = [m["id"] for m in call("/v1/models")["data"]]
    check("[1] model advertised", len(ids) == 1, str(ids))

    c = chat("What is 17 + 25? Answer with just the number.")
    check("[2] greedy answer", c["message"]["content"].strip() == "42", repr(c["message"]["content"]))
    check("[3] thinking-off carries no reasoning text", not c["message"].get("reasoning_content"))

    items = " ".join(f"Item {i}: the value is {i * 7 % 13}." for i in range(160))  # ~1.9k tokens, past the window
    c = chat(items + "\n\nWhat is the value of Item 3? Answer with just the number.")
    check("[4] retrieval past the sliding window", c["message"]["content"].strip() == "8", repr(c["message"]["content"]))

    tools = [{"type": "function", "function": {"name": "get_weather", "description": "weather for a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    c = chat("What is the weather in Berlin? Use the tool.", tools=tools, tool_choice="auto", max_tokens=128)
    calls = c["message"].get("tool_calls") or []
    check("[5] tool call", c["finish_reason"] == "tool_calls" and calls and calls[0]["function"]["name"] == "get_weather")
finally:
    server.terminate()
    server.wait(timeout=60)

log = open(LOG).read()
check("[6] fused router engaged", "fused router kernel engaged: mode=logit_bias" in log)
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
