#!/bin/bash
# A reasoning BUDGET must SHORTEN the thought, not hide it — and END it: at the
# budget the server commits the early-stop line and the closer through the
# model, so the answer (or tool call) still arrives inside max_tokens.
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
#   3. total streamed reasoning stays inside the budget plus the forced close
#   4. the thought is CLOSED by the server (early-stop line) and the turn
#      finishes with content or a tool call, never `length` (stream + non-stream)
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
fin=""; has_answer=0
for l in lines:
    try: d=json.loads(l)
    except Exception: continue
    for c in d.get("choices",[]):
        if c.get("finish_reason"): fin=c["finish_reason"]
        dl=c.get("delta") or {}
        if dl.get("content") or dl.get("tool_calls"): has_answer=1
closed=1 if "Considering the limited time" in total else 0
print(f"N={n}; FIRST={first if first is not None else -1}; EVENTS={len(lines)}; CHARS={len(total)}; FIN={fin}; ANSWER={has_answer}; CLOSED={closed}")
PY
)"

check "reasoning arrives in more than one delta (not one end-of-stream dump)" "$([ "$N" -gt 1 ] && echo 1 || echo 0)"
check "first reasoning delta lands in the first half of the stream" "$([ "$FIRST" -ge 0 ] && [ "$FIRST" -lt $((EVENTS / 2 + 1)) ] && echo 1 || echo 0)"
# ~4 chars/token is generous; the cap is enforced in tokens, plus the ~30-token forced close.
check "streamed reasoning stays inside the budget plus the forced close" "$([ "$CHARS" -le $(((BUDGET + 40) * 8)) ] && echo 1 || echo 0)"
check "the server closed the thought (early-stop line streamed as reasoning)" "$CLOSED"
check "the turn ends with an answer or tool call, not length (got '$FIN')" "$([ "$ANSWER" = 1 ] && [ "$FIN" != "length" ] && echo 1 || echo 0)"

# Non-stream: same bound, whole body.
curl -s -m 300 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d "{
  \"model\":\"x\",
  \"messages\":[{\"role\":\"user\",\"content\":\"Explain in two sentences why the sky is blue. Think it through carefully first.\"}],
  \"max_tokens\":600, \"temperature\":0, \"enable_thinking\":true,
  \"reasoning_budget_tokens\":$BUDGET}" > "$OUT"
eval "$(python3 - "$OUT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); c=d["choices"][0]; m=c["message"]
r=m.get("reasoning_content") or ""; content=m.get("content") or ""
print(f"NS_FIN={c.get('finish_reason')}; NS_CLOSED={1 if 'Considering the limited time' in r else 0}; NS_CONTENT={1 if content.strip() else 0}; NS_RCHARS={len(r)}")
PY
)"
check "non-stream: thought closed by the server" "$NS_CLOSED"
check "non-stream: content present and finish_reason stop (got '$NS_FIN')" "$([ "$NS_CONTENT" = 1 ] && [ "$NS_FIN" = "stop" ] && echo 1 || echo 0)"
check "non-stream: reasoning inside the budget plus the forced close" "$([ "$NS_RCHARS" -le $(((BUDGET + 40) * 8)) ] && echo 1 || echo 0)"

# /v1/responses: the same budget, enforced the same way (it used to be parsed and dropped).
curl -s -m 300 "$BASE/v1/responses" -H 'content-type: application/json' -d "{
  \"model\":\"x\", \"input\":\"Explain in two sentences why the sky is blue. Think it through carefully first.\",
  \"max_output_tokens\":600, \"temperature\":0, \"reasoning\":{\"effort\":\"high\"},
  \"reasoning_budget_tokens\":$BUDGET}" > "$OUT"
eval "$(python3 - "$OUT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); r=""; text=""
for it in d.get("output",[]):
    if it.get("type")=="reasoning":
        for part in (it.get("summary") or [])+(it.get("content") or []): r+=part.get("text","")
    if it.get("type")=="message":
        for part in it.get("content") or []: text+=part.get("text","")
print(f"RS_STATUS={d.get('status')}; RS_CLOSED={1 if 'Considering the limited time' in r else 0}; RS_TEXT={1 if text.strip() else 0}; RS_RCHARS={len(r)}")
PY
)"
check "responses: thought closed by the server" "$RS_CLOSED"
check "responses: message text present, status completed (got '$RS_STATUS')" "$([ "$RS_TEXT" = 1 ] && [ "$RS_STATUS" = "completed" ] && echo 1 || echo 0)"
check "responses: reasoning inside the budget plus the forced close" "$([ "$RS_RCHARS" -le $(((BUDGET + 40) * 8)) ] && echo 1 || echo 0)"

curl -sN -m 300 "$BASE/v1/responses" -H 'content-type: application/json' -d "{
  \"model\":\"x\", \"input\":\"Explain in two sentences why the sky is blue. Think it through carefully first.\",
  \"max_output_tokens\":600, \"temperature\":0, \"stream\":true, \"reasoning\":{\"effort\":\"high\"},
  \"reasoning_budget_tokens\":$BUDGET}" > "$OUT"
eval "$(python3 - "$OUT" <<'PY'
import json,sys
r=""; done=""
for l in open(sys.argv[1]).read().splitlines():
    if not l.startswith("data: "): continue
    try: d=json.loads(l[6:])
    except Exception: continue
    if d.get("type","").startswith("response.reasoning") and d.get("type","").endswith(".delta"): r+=d.get("delta","")
    if d.get("type")=="response.completed": done=d["response"].get("status","")
print(f"RSS_CLOSED={1 if 'Considering the limited time' in r else 0}; RSS_STATUS={done}")
PY
)"
check "responses stream: thought closed by the server" "$RSS_CLOSED"
check "responses stream: completes (got '$RSS_STATUS')" "$([ "$RSS_STATUS" = "completed" ] && echo 1 || echo 0)"

echo "[budget-stream] $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
