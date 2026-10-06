#!/bin/bash
# test_repeat_penalty.sh — a penalty shapes the reply on every path that serves it.
#
# Bar: at temp 0, repeat/frequency/presence penalties change a looping reply
# without logprobs, the same bytes as with them, streamed or not, and under a
# json_schema; frequency_penalty x is repeat_penalty 1+x on chat and /v1/completions; a
# penalized request never runs a PLD round (verify reads raw logits).
#
#   PENALTY_TEST_MODEL=<dir> ./tests/test_repeat_penalty.sh [port]
#
# Any chat model works; defaults to Qwen3.5-0.8B-MLX-4bit. SKIPs without one.
set -uo pipefail

MODEL="${PENALTY_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}"
PORT="${1:-11294}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
BASE="http://127.0.0.1:$PORT"

[ -d "$MODEL" ] || { echo "SKIP: no model at $MODEL"; exit 0; }
[ -x "$BIN" ]   || { echo "fail: build mlx-serve first"; exit 1; }

LOG="$(mktemp)"
pkill -f "mlx-serve --serve.*port $PORT" 2>/dev/null
for _ in $(seq 1 30); do lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
"$BIN" --serve --model "$MODEL" --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -f "$LOG"' EXIT

for _ in $(seq 1 180); do curl -sf -m 2 "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf -m 2 "$BASE/health" >/dev/null 2>&1 || { echo "fail: server never came up"; tail -20 "$LOG"; exit 1; }

python3 - "$BASE" "$LOG" <<'PY'
import json, sys, urllib.request
BASE, LOG = sys.argv[1], sys.argv[2]
fails = []

def ck(name, cond, detail=""):
    print(("  \033[32mPASS\033[0m  " if cond else "  \033[31mFAIL\033[0m  ") + name + ("  " + detail if not cond else ""))
    if not cond: fails.append(name)

def pld_rounds():
    return open(LOG, errors="replace").read().count("[spec-stats] mode=pld")

# Seeded with the repetition so PLD's prompt gate arms on the plain request.
PROMPT = "Write the word apple 40 times separated by spaces. Here is the start: " + "apple " * 8
SCHEMA = {"type": "json_schema", "json_schema": {"name": "w", "strict": True, "schema": {
    "type": "object", "properties": {"words": {"type": "array", "items": {"type": "string"}}},
    "required": ["words"], "additionalProperties": False}}}

def chat(prompt=PROMPT, stream=False, **extra):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 120,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": stream, **extra}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        if not stream:
            return json.load(r)["choices"][0]["message"]["content"] or ""
        out = []
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            for c in json.loads(line[6:]).get("choices", []):
                out.append((c.get("delta") or {}).get("content") or "")
        return "".join(out)

before = pld_rounds()
plain = chat(repeat_penalty=1.0)
plain_pld = pld_rounds() - before
before = pld_rounds()
pen = chat(repeat_penalty=2.0)
pen_pld = pld_rounds() - before

ck("repeat_penalty changes the reply without logprobs", pen != plain, f"{pen.count('apple')} apples")
ck("same bytes as with logprobs", pen == chat(repeat_penalty=2.0, logprobs=True, top_logprobs=1))
ck("same bytes streamed", pen == chat(repeat_penalty=2.0, stream=True))
ck("frequency_penalty changes the reply", chat(frequency_penalty=1.5) != plain)
ck("presence_penalty changes the reply", chat(presence_penalty=2.0) != plain)
if plain_pld:
    ck("a penalized request runs no PLD round", pen_pld == 0, f"{pen_pld} rounds")
else:
    print("  SKIP  PLD did not engage on the plain request; the spec gate is unexercised")

import threading
duo = {}
# The companion's prompt keeps PLD's gate shut, so both slots are batch-eligible.
ts = [threading.Thread(target=lambda: duo.__setitem__("plain", chat("Tell a long story about a fox.", max_tokens=240))),
      threading.Thread(target=lambda: duo.__setitem__("pen", chat(repeat_penalty=2.0, max_tokens=240)))]
for t in ts: t.start()
for t in ts: t.join()
ck("a penalized request beside a plain one keeps its solo bytes", duo.get("pen") == chat(repeat_penalty=2.0, max_tokens=240))

def completion(**extra):
    body = {"prompt": "apple apple apple apple apple apple", "temperature": 0, "max_tokens": 40, **extra}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["text"]

# frequency_penalty x is read as repeat_penalty 1 + x on every surface.
for name, call in (("chat", chat), ("completions", completion)):
    freq, rep = call(frequency_penalty=1.5), call(repeat_penalty=2.5)
    ck(f"{name}: frequency_penalty 1.5 == repeat_penalty 2.5, and penalized", freq == rep and rep != call(),
       f"freq={freq[:60]!r} rep={rep[:60]!r}")

pj = "Write the word apple 40 times. Return JSON with a words array."
j1 = chat(pj, response_format=SCHEMA, repeat_penalty=1.0)
j4 = chat(pj, response_format=SCHEMA, repeat_penalty=4.0)
ck("repeat_penalty changes a json_schema reply", j1 != j4)

print("FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
PY
