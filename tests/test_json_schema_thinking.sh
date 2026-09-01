#!/bin/bash
# Issue #331: schema JSON must reach the final content channel when thinking is
# requested. The assertions are server invariants, not a grade of model prose.
# Usage: ./tests/test_json_schema_thinking.sh [model_dir] [port]

set -u

MODEL=${1:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}
PORT=${2:-8137}
BASE="http://127.0.0.1:$PORT"
LOG=/tmp/mlx-serve-json-schema-thinking.log
BIN=${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}
PASS=0
FAIL=0
TOTAL=0

if [ ! -d "$MODEL" ]; then echo "SKIP: model not found at $MODEL"; exit 0; fi
if [ ! -x "$BIN" ]; then echo "FAIL: mlx-serve not built"; exit 1; fi
command -v jq >/dev/null 2>&1 || { echo "FAIL: jq is required"; exit 1; }
if curl -sf "$BASE/health" >/dev/null 2>&1; then echo "FAIL: port $PORT is in use"; exit 1; fi

"$BIN" serve --port "$PORT" --host 127.0.0.1 --log-level info --model "$MODEL" >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

for i in $(seq 1 180); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -80 "$LOG"; exit 1; fi
    if [ "$i" -eq 180 ]; then echo "FAIL: server did not start"; exit 1; fi
    sleep 1
done

record() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

valid_schema_json() { jq -e 'type == "object" and (.answer | type == "string")' >/dev/null 2>&1; }
valid_json_object() { jq -e 'type == "object"' >/dev/null 2>&1; }

SCHEMA='{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}'
PROMPT='Consider the request carefully, then return a short answer in the required field.'

chat_body() {
    local stream=$1 max_tokens=$2 format=$3 effort=${4:-medium}
    curl -s -m 600 -N "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
      \"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
      \"temperature\":0,\"max_tokens\":$max_tokens,\"stream\":$stream,\"reasoning_effort\":\"$effort\",
      \"response_format\":$format}"
}

check_chat_nonstream() {
    local label=$1 body=$2 mode=$3
    local content
    content=$(echo "$body" | jq -r '.choices[0].message.content // ""' 2>/dev/null)
    if [ -z "$content" ]; then record "$label" FAIL "empty content"; return; fi
    if [ "$mode" = schema ]; then
        echo "$content" | valid_schema_json && record "$label" PASS "" || record "$label" FAIL "invalid schema JSON: ${content:0:120}"
    else
        echo "$content" | valid_json_object && record "$label" PASS "" || record "$label" FAIL "invalid JSON object: ${content:0:120}"
    fi
}

check_chat_stream() {
    local label=$1 events=$2
    local content
    content=$(echo "$events" | sed -n 's/^data: //p' | jq -rj 'select(type=="object") | .choices[0].delta.content // empty' 2>/dev/null)
    if [ -n "$content" ] && echo "$content" | valid_schema_json; then record "$label" PASS ""
    else record "$label" FAIL "invalid streamed content: ${content:0:120}"; fi
}

FORMAT_SCHEMA="{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"answer\",\"strict\":true,\"schema\":$SCHEMA}}"
FORMAT_OBJECT='{"type":"json_object"}'

echo "=== Chat Completions ==="
check_chat_nonstream "schema, thinking off constrains from token zero" "$(chat_body false 256 "$FORMAT_SCHEMA" none)" schema
check_chat_nonstream "schema, thinking on reaches content" "$(chat_body false 2048 "$FORMAT_SCHEMA")" schema
check_chat_stream "stream reconstructs schema JSON from content deltas" "$(chat_body true 2048 "$FORMAT_SCHEMA")"
check_chat_nonstream "json_object reaches content" "$(chat_body false 2048 "$FORMAT_OBJECT")" object

echo "=== Reserved final-answer tail and repeated small-model case ==="
for run in 1 2 3; do
    check_chat_nonstream "short-limit run $run returns schema JSON" "$(chat_body false 65 "$FORMAT_SCHEMA")" schema
done

echo "=== Responses API ==="
RESP=$(curl -s -m 600 "$BASE/v1/responses" -H 'Content-Type: application/json' -d "{
  \"model\":\"mlx-serve\",\"input\":\"$PROMPT\",\"temperature\":0,\"max_output_tokens\":2048,
  \"reasoning\":{\"effort\":\"medium\"},\"text\":{\"format\":{\"type\":\"json_schema\",\"name\":\"answer\",\"strict\":true,\"schema\":$SCHEMA}}}"
)
RTEXT=$(echo "$RESP" | jq -r '[.output[]? | select(.type=="message") | .content[]? | select(.type=="output_text") | .text] | join("")' 2>/dev/null)
if [ -n "$RTEXT" ] && echo "$RTEXT" | valid_schema_json; then record "Responses output_text contains schema JSON" PASS ""
else record "Responses output_text contains schema JSON" FAIL "${RTEXT:0:120}"; fi

RSTREAM=$(curl -s -m 600 -N "$BASE/v1/responses" -H 'Content-Type: application/json' -d "{
  \"model\":\"mlx-serve\",\"input\":\"$PROMPT\",\"temperature\":0,\"max_output_tokens\":2048,\"stream\":true,
  \"reasoning\":{\"effort\":\"medium\"},\"text\":{\"format\":{\"type\":\"json_schema\",\"name\":\"answer\",\"strict\":true,\"schema\":$SCHEMA}}}"
)
RTEXT=$(echo "$RSTREAM" | sed -n 's/^data: //p' | jq -rj 'select(.type=="response.output_text.delta") | .delta' 2>/dev/null)
if [ -n "$RTEXT" ] && echo "$RTEXT" | valid_schema_json; then record "Responses stream reconstructs schema JSON" PASS ""
else record "Responses stream reconstructs schema JSON" FAIL "${RTEXT:0:120}"; fi

echo "=== Anthropic Messages ==="
ANTH=$(curl -s -m 600 "$BASE/v1/messages" -H 'Content-Type: application/json' -d "{
  \"model\":\"mlx-serve\",\"max_tokens\":2048,\"temperature\":0,
  \"thinking\":{\"type\":\"enabled\",\"budget_tokens\":256},
  \"output_config\":{\"effort\":\"medium\",\"format\":{\"type\":\"json_schema\",\"schema\":$SCHEMA}},
  \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}"
)
ATEXT=$(echo "$ANTH" | jq -r '[.content[]? | select(.type=="text") | .text] | join("")' 2>/dev/null)
if [ -n "$ATEXT" ] && echo "$ATEXT" | valid_schema_json; then record "Anthropic text contains schema JSON" PASS ""
else record "Anthropic text contains schema JSON" FAIL "${ATEXT:0:120}"; fi

echo "=== Server invariants ==="
grep -q '\[grammar\] deferring JSON schema' "$LOG" \
    && record "supported prompt defers grammar" PASS "" \
    || record "supported prompt defers grammar" FAIL "missing deferral log"
grep -Eq '\[grammar\] reasoning boundary (reached|forced)' "$LOG" \
    && record "deferred grammar activates naturally or by recovery" PASS "" \
    || record "deferred grammar activates naturally or by recovery" FAIL "missing activation log"
if grep -q '\[spec-stats\]' "$LOG"; then record "constrained requests never speculate" FAIL "speculative stats found"
else record "constrained requests never speculate" PASS ""; fi

echo
echo "=== $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
