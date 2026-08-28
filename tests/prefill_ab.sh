#!/usr/bin/env bash
# Prefill-time A/B for a kill switch. Boots once per arm and reads the SERVER's
# own `timings.prompt_ms`, never client-side timing.
#
#   tests/prefill_ab.sh <model-dir> <ENV_VAR> <on-value> <off-value> [reps]
#
# Every request carries a unique nonce so the prefix cache cannot serve it, and
# the script FLAGS a reading whose cached token count exceeds the chat-template
# header (~40) — a cached prefill is not a prefill (the trap in docs/gotchas:
# vary the leading text and assert the cache did not serve the body).
set -uo pipefail

MODEL="${1:?usage: prefill_ab.sh <model-dir> <ENV_VAR> <on-value> <off-value> [reps]}"
VAR="${2:?}"; ONV="${3:?}"; OFFV="${4:?}"; REPS="${5:-3}"
PORT="${PORT:-8093}"
BIN="${BIN:-./zig-out/bin/mlx-serve}"

PROMPT_BODY=$(python3 -c "
para='A B-tree is a self-balancing search tree whose nodes hold many keys and many children, which keeps its height small and suits block-oriented storage well. '
print(para*int('${PREFILL_AB_PARAS:-260}'))")

arm() {
  local val="$1" label="$2"
  pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 2
  env "$VAR=$val" "$BIN" --model "$MODEL" --serve --port "$PORT" >"$HOME/claude-tmp/prefill-ab/$PORT-${label// /}.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 900); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  for r in $(seq 1 "$REPS"); do
    python3 - "$PORT" "$r" "$label" <<PY
import json, sys, urllib.request, uuid
port, rep, label = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({
  "model": "m",
  "messages": [{"role": "user", "content": "nonce-%s-%s. %s\nSummarise in one word." % (uuid.uuid4(), rep, """$PROMPT_BODY""")}],
  "max_tokens": 4, "temperature": 0,
}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port,
                             data=body, headers={"content-type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=600))
t = d.get("timings") or {}
u = d.get("usage") or {}
cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
# The chat-template header is always cached; only a real prefix hit invalidates.
status = "OK" if cached <= 64 else "CACHED-INVALID"
print("%s rep%s prompt_ms=%s prompt_tokens=%s cached=%s %s"
      % (label, rep, t.get("prompt_ms"), u.get("prompt_tokens"), cached, status))
PY
  done
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}

arm "$ONV"  "on "
arm "$OFFV" "off"
arm "$ONV"  "on "
arm "$OFFV" "off"
