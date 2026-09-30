#!/bin/bash
# The prefill's sampled token streams as its OWN first delta, and the stream
# still carries exactly the non-stream bytes (the early token is not sent
# twice), on the drafted arm, the serial arm and at max_tokens 1.
#
#   ./tests/test_first_token_early.sh [port]
#   FIRST_TOKEN_TEST_MODEL=<dir with a thinking chat model>   (default: the Qwen3.8-27B pack)
set -u
PORT=${1:-8093}
BASE="http://127.0.0.1:$PORT"
MODEL="${FIRST_TOKEN_TEST_MODEL:-$HOME/.mlx-serve/models/ddalcu/Qwen3.8-27B-MLX-Serve-4bit}"
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -d "$MODEL" ]; then echo "SKIP test_first_token_early: $MODEL not found"; exit 0; fi

LOG=$(mktemp)
"$BINARY" --model "$MODEL" --serve --port "$PORT" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null; rm -f "$LOG"' EXIT
for _ in $(seq 1 600); do curl -sf "$BASE/health" >/dev/null && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
curl -sf "$BASE/health" >/dev/null || { echo "FAIL server did not start"; tail -20 "$LOG"; exit 1; }

python3 - "$BASE" <<'EOF'
import json, sys, urllib.request
base = sys.argv[1]
def post(path, body):
    return urllib.request.urlopen(urllib.request.Request(base + path, json.dumps(body).encode(), {"content-type": "application/json"}))
def ntokens(text):
    return len(json.load(post("/tokenize", {"content": text}))["tokens"])
fails = 0
for label, extra, max_tokens in [("drafted", {}, 48), ("serial", {"enable_drafter": False, "enable_mtp": False}, 48), ("max_tokens 1", {}, 1)]:
    body = {"model": "x", "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": "medium",
            "messages": [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]}
    body.update(extra)
    ns = json.load(post("/v1/chat/completions", body))
    msg = ns["choices"][0]["message"]
    want = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    deltas = []
    for line in post("/v1/chat/completions", dict(body, stream=True)):
        if not line.startswith(b"data: {"): continue
        for ch in json.loads(line[6:]).get("choices") or []:
            d = ch.get("delta") or {}
            t = (d.get("reasoning_content") or "") + (d.get("content") or "")
            if t: deltas.append(t)
    got = "".join(deltas)
    ok = got == want and deltas and ntokens(deltas[0]) == 1
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {label}: first delta {deltas[0] if deltas else None!r}, stream == non-stream: {got == want}")
    if got != want: print(f"  stream:     {got!r}\n  non-stream: {want!r}")
sys.exit(1 if fails else 0)
EOF
