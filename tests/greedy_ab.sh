#!/usr/bin/env bash
# greedy_ab.sh — the acceptance bar for any change that claims to preserve output: boots the same model twice, once per
# arm of a kill switch, and diffs the temp-0 chat continuation.
#
#   tests/greedy_ab.sh <model-dir> <ENV_VAR> <off-value> [max_tokens] [prompt]
#
# Reports completion_tokens and finish_reason so an early stop cannot pass as agreement. Use >= 200 tokens: INT4
# near-tie and 1-in-65536 rounding classes surface between tokens 50 and 100.

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"

MODEL="${1:?usage: greedy_ab.sh <model-dir> <ENV_VAR> <off-value> [max_tokens] [prompt]}"
VAR="${2:?}"
OFFVAL="${3:?}"
MAXTOK="${4:-200}"
PROMPT="${5:-Explain how a B-tree index speeds up a database range scan, step by step.}"
PORT="${PORT:-8098}"
BIN="${BIN:-./zig-out/bin/mlx-serve}"

run_arm() {
  local val="$1" out="$2"
  pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
  sleep 2
  if [ -n "$val" ]; then
    env "$VAR=$val" "$BIN" --model "$MODEL" --serve --port "$PORT" >/tmp/greedy-ab-$PORT.log 2>&1 &
  else
    "$BIN" --model "$MODEL" --serve --port "$PORT" >/tmp/greedy-ab-$PORT.log 2>&1 &
  fi
  local pid=$!
  srv_wait_health "http://127.0.0.1:$PORT" 600 1 "$pid" || true
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "$(printf '{"model":"m","messages":[{"role":"user","content":%s}],"max_tokens":%d,"temperature":0}' \
          "$(printf '%s' "$PROMPT" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" "$MAXTOK")" \
    | python3 -c '
import json,sys
d=json.load(sys.stdin)
c=d["choices"][0]["message"]
sys.stderr.write("  completion_tokens=%s finish=%s\n"%(d.get("usage",{}).get("completion_tokens"),d["choices"][0].get("finish_reason")))
print((c.get("reasoning_content") or "")+"\n----\n"+(c.get("content") or ""))' >"$out"
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}

run_arm "$OFFVAL" /tmp/greedy-off.txt
run_arm ""        /tmp/greedy-on.txt

if diff -q /tmp/greedy-off.txt /tmp/greedy-on.txt >/dev/null; then
  echo "IDENTICAL ($(wc -c </tmp/greedy-on.txt) bytes)"
else
  echo "DIVERGED"
  diff /tmp/greedy-off.txt /tmp/greedy-on.txt | head -20
  echo "--- first differing byte:"
  cmp /tmp/greedy-off.txt /tmp/greedy-on.txt
fi
