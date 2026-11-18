#!/usr/bin/env bash
# The embedded llama.cpp engine decodes a Q2_0 GGUF: Ternary-Bonsai-2-27B answers three short questions correctly
# (an older pinned llama.cpp loaded the file and returned mixed-script garbage at normal speed).
#
#   Q2_0_GGUF=/path/to/Ternary-Bonsai-2-27B-Q2_0.gguf ./tests/test_llama_q2_0.sh [port]   # unset = SKIP
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${MLX_SERVE_BIN:-$ROOT/zig-out/bin/mlx-serve}"
PORT="${1:-11433}"
MODEL="${Q2_0_GGUF:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "SKIP test_llama_q2_0: set Q2_0_GGUF=/path/to/Ternary-Bonsai-2-27B-Q2_0.gguf to run"
    exit 0
fi
[ -x "$BIN" ] || { echo "FAIL: build mlx-serve first"; exit 1; }
WORK="$(mktemp -d)"
SERVER_PID=""
cleanup() {
    srv_stop "$SERVER_PID"
    rm -rf "$WORK"
}
trap cleanup EXIT

mkdir -p "$WORK/home"
HOME="$WORK/home" "$BIN" --serve --model "$MODEL" --host 127.0.0.1 --port "$PORT" --log-file off >"$WORK/server.log" 2>&1 &
SERVER_PID=$!
srv_wait_health "http://127.0.0.1:$PORT" 300 1 "$SERVER_PID" || { tail -20 "$WORK/server.log" >&2; exit 1; }
grep -q "llama.cpp engine" "$WORK/server.log" || { echo "FAIL: the file did not route to the llama.cpp engine"; exit 1; }

python3 -I - "$PORT" <<'PY'
import json, sys, urllib.request
cases = [
    ("What is 17 x 23? Answer with the number only.", "391"),
    ("What is the capital of Australia? Answer with one word.", "canberra"),
    ("Reverse the letters of the word 'drafter'. Answer with the reversed word only.", "retfard"),
]
fails = 0
for prompt, want in cases:
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1024}
    req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.load(r)["choices"][0]["message"]["content"] or ""
    ok = want in text.lower()
    fails += 0 if ok else 1
    print(("PASS" if ok else "FAIL") + f" {want!r} in the answer to {prompt!r}: {text[:120]!r}")
sys.exit(1 if fails else 0)
PY
