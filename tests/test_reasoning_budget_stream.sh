#!/bin/bash
# A reasoning BUDGET must SHORTEN the thought, not hide it.
#
# With `tools` present the streaming chat path used to gate incremental
# reasoning on `reasoning_budget < 0`: a capped request showed NOTHING for the
# whole generation and then received one truncated dump at the end, so a capped
# agent session looked frozen (live 2026-08-14, pi on Qwen3.8-27B). Capping and
# streaming are not in conflict — you never exceed a cap you stop emitting at.
#
# Checks, tools + an explicit tiny `reasoning_budget_tokens`:
#   1. reasoning arrives in MORE THAN ONE delta (incremental, not one dump)
#   2. the first reasoning delta lands well before the stream ends
#   3. total streamed reasoning stays inside the budget
#
# Usage: ./tests/test_reasoning_budget_stream.sh [model_dir] [port]
set -u

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-4B-MLX-4bit}"
PORT="${2:-11267}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
BUDGET=24
PASS=0
FAIL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

check() {
    if [ "$2" = "1" ]; then PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $1";
    else FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC} $1"; fi
}

if [ ! -d "$MODEL" ]; then echo "skip: model not found ($MODEL)"; exit 0; fi

LOG=$(mktemp); OUT=$(mktemp)
"$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -f "$LOG" "$OUT"' EXIT
for _ in $(seq 1 120); do sleep 2; curl -sf "$BASE/health" >/dev/null 2>&1 && break; done
if ! curl -sf "$BASE/health" >/dev/null 2>&1; then echo "server failed to start"; tail -5 "$LOG"; exit 1; fi

echo "[budget-stream] === $(basename "$MODEL") ==="
curl -sN -m 300 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d "{
  \"model\":\"x\",
  \"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Think about which unit to use first.\"}],
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",
    \"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}],
  \"max_tokens\":600, \"temperature\":0, \"stream\":true, \"enable_thinking\":true,
  \"reasoning_budget_tokens\":$BUDGET}" > "$OUT"

python3 - "$OUT" "$BUDGET" <<'PY'
import json,sys
lines=[l[6:] for l in open(sys.argv[1]).read().splitlines() if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
n_reasoning=0; first_at=None; total=""
for i,l in enumerate(lines):
    try: d=json.loads(l)
    except Exception: continue
    for c in d.get("choices",[]):
        r=(c.get("delta") or {}).get("reasoning_content")
        if r:
            n_reasoning+=1; total+=r
            if first_at is None: first_at=i
print(f"REASONING_DELTAS={n_reasoning}")
print(f"FIRST_AT={first_at if first_at is not None else -1}")
print(f"TOTAL_EVENTS={len(lines)}")
print(f"REASONING_CHARS={len(total)}")
PY
eval "$(python3 - "$OUT" <<'PY'
import json,sys
lines=[l[6:] for l in open(sys.argv[1]).read().splitlines() if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
n=0; first=None; total=""
for i,l in enumerate(lines):
    try: d=json.loads(l)
    except Exception: continue
    for c in d.get("choices",[]):
        r=(c.get("delta") or {}).get("reasoning_content")
        if r:
            n+=1; total+=r
            if first is None: first=i
print(f"N={n}; FIRST={first if first is not None else -1}; EVENTS={len(lines)}; CHARS={len(total)}")
PY
)"

check "reasoning arrives in more than one delta (not one end-of-stream dump)" "$([ "$N" -gt 1 ] && echo 1 || echo 0)"
check "first reasoning delta lands in the first half of the stream" "$([ "$FIRST" -ge 0 ] && [ "$FIRST" -lt $((EVENTS / 2 + 1)) ] && echo 1 || echo 0)"
# ~4 chars/token is generous; the cap is enforced in tokens.
check "streamed reasoning stays inside the budget" "$([ "$CHARS" -le $((BUDGET * 8)) ] && echo 1 || echo 0)"

echo "[budget-stream] $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
