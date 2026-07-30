#!/usr/bin/env bash
# Greedy on/off comparison for a decode-path change. Boots the same model twice,
# once per arm of a kill switch, and diffs the temp-0 continuation.
#
#   tests/greedy_ab.sh <model-dir> <ENV_VAR> <off-value> [max_tokens] [prompt]
#
# The switch's OFF arm is the control; a byte-identical diff is the acceptance
# bar for any change that claims to preserve output.
set -uo pipefail

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
  for _ in $(seq 1 600); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
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
