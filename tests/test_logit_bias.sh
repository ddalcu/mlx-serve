#!/bin/bash
# test_logit_bias.sh — OpenAI `logit_bias` reaches an engine that serves it and is refused by name where it is not.
# On a GGUF (llama.cpp) model, at temp 0: -100 on the plain reply's tokens changes the reply, +100 on one token makes
# the reply that token repeated, on chat (stream and not) and /v1/completions. On an MLX model a non-empty
# `logit_bias` is a 400 naming it, and an empty one is served.
#
#   LLAMA_GGUF_MODEL=/path/to/model.gguf [LOGIT_BIAS_MLX_MODEL=<dir>] ./tests/test_logit_bias.sh [port]
#   # GGUF unset = SKIP; the MLX half defaults to Qwen3.5-0.8B-MLX-4bit and SKIPs without it

source "$(dirname "$0")/_lib_models.sh"

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"

GGUF="${LLAMA_GGUF_MODEL:-}"
MLX="${LOGIT_BIAS_MLX_MODEL:-$(find_model mlx-community/Qwen3.5-0.8B-MLX-4bit || true)}"
PORT="${1:-11296}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"

[ -n "$GGUF" ] || { echo "SKIP: set LLAMA_GGUF_MODEL=/path/to/model.gguf"; exit 0; }
[ -x "$BIN" ]  || { echo "fail: build mlx-serve first"; exit 1; }

LOG="$(mktemp)"; SRV=""
trap 'kill $SRV 2>/dev/null; rm -f "$LOG"' EXIT
FAILS=0

serve() {
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    "$BIN" --serve --model "$1" --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
    SRV=$!
    srv_wait_health "http://127.0.0.1:$PORT" 180 1 || { echo "fail: server never came up on $1"; tail -20 "$LOG"; exit 1; }
}

check() {
    python3 - "http://127.0.0.1:$PORT" "$1" <<'PY'
import json, sys, urllib.error, urllib.request
BASE, MODE = sys.argv[1], sys.argv[2]
fails = []

def ck(name, cond, detail=""):
    print(("  \033[32mPASS\033[0m  " if cond else "  \033[31mFAIL\033[0m  ") + name + ("  " + detail if not cond else ""))
    if not cond: fails.append(name)

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

PROMPT = "What is the capital of France? Answer with one word."

def chat(bias=None, stream=False):
    body = {"messages": [{"role": "user", "content": PROMPT}], "temperature": 0, "max_tokens": 4, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}
    if bias is not None: body["logit_bias"] = bias
    code, text = post("/v1/chat/completions", body)
    if code != 200: return code, text
    if not stream: return code, json.loads(text)["choices"][0]["message"]["content"] or ""
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            for c in json.loads(line[6:]).get("choices", []):
                out.append((c.get("delta") or {}).get("content") or "")
    return code, "".join(out)

def complete(bias=None):
    body = {"prompt": "The capital of France is", "temperature": 0, "max_tokens": 4}
    if bias is not None: body["logit_bias"] = bias
    code, text = post("/v1/completions", body)
    return code, (json.loads(text)["choices"][0]["text"] if code == 200 else text)

def ids(text):
    return json.loads(post("/tokenize", {"content": text})[1])["tokens"]

def piece(t):
    return json.loads(post("/detokenize", {"tokens": [t]})[1])["content"]

if MODE == "gguf":
    force = ids(" zebra")[-1]
    want = piece(force) * 2
    for name, fn in (("chat", lambda b=None: chat(b)), ("chat stream", lambda b=None: chat(b, stream=True)),
                     ("completions", complete)):
        code, plain = fn()
        ck(f"{name}: plain reply", code == 200 and plain.strip() != "", repr(plain))
        _, banned = fn({str(t): -100 for t in ids(plain.strip())})
        ck(f"{name}: -100 on the reply's tokens changes it", banned.strip() != plain.strip(), f"{banned!r} vs {plain!r}")
        _, forced = fn({str(force): 100})
        ck(f"{name}: +100 on one token repeats it", forced.startswith(want), f"{forced!r}, want {want!r}...")
else:
    code, text = chat({"15": -100})
    ck("mlx: a logit_bias is a 400", code == 400, f"{code} {text[:200]}")
    ck("mlx: the 400 names logit_bias", "logit_bias" in text, text[:200])
    code, text = complete({"15": -100})
    ck("mlx: /v1/completions refuses it too", code == 400 and "logit_bias" in text, f"{code} {text[:200]}")
    code, text = chat({})
    ck("mlx: an empty logit_bias is served", code == 200 and text.strip() != "", f"{code} {text[:200]}")

sys.exit(1 if fails else 0)
PY
}

echo "→ GGUF: $GGUF"
serve "$GGUF"
check gguf || FAILS=$((FAILS + 1))

if [ -n "$MLX" ] && [ -d "$MLX" ]; then
    echo "→ MLX: $MLX"
    serve "$MLX"
    check mlx || FAILS=$((FAILS + 1))
else
    echo "SKIP: MLX half (no Qwen3.5-0.8B-MLX-4bit; set LOGIT_BIAS_MLX_MODEL)"
fi

[ "$FAILS" -eq 0 ] && echo "PASS: test_logit_bias" || { echo "FAIL: test_logit_bias"; exit 1; }
