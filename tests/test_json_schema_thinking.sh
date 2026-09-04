#!/bin/bash
# Guard (issue #331): schema JSON must reach CONTENT on every mask-building
# surface. Supported unlimited Qwen reasoning may defer the grammar; requests
# with a finite response-side reasoning budget retain the thinking-off fallback.
#
# Greedy throughout. Usage: ./tests/test_json_schema_thinking.sh [model_dir] [port]

set -u

MODEL=${1:-~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}
PORT=${2:-8137}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

MODEL=$(eval echo "$MODEL")
if [ ! -d "$MODEL" ]; then echo "SKIP: model not found at $MODEL"; exit 0; fi
if [ ! -x "./zig-out/bin/mlx-serve" ]; then
    echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi
command -v jq >/dev/null 2>&1 || { echo "FAIL: jq is required"; exit 1; }

LOG=/tmp/mlx-serve-json-schema-thinking.log
./zig-out/bin/mlx-serve serve --port $PORT --host 127.0.0.1 --log-level info \
    --model "$MODEL" >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 120); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    if [ "$i" -eq 120 ]; then echo "FAIL: server did not start within 120s"; exit 1; fi
    sleep 1
done

run_test() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

SCHEMA='{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}'
PROMPT="A basket starts with 17 apples. Six are removed, then three groups of two are added. Give the final count."

echo "=== /v1/chat/completions: reasoning_effort + json_schema, non-stream ==="
BODY=$(curl -s -m 300 "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"mlx-serve\", \"max_tokens\": 512, \"temperature\": 0, \"stream\": false,
    \"reasoning_effort\": \"medium\",
    \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"answer\", \"strict\": true, \"schema\": $SCHEMA}},
    \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}]
}")
CONTENT=$(echo "$BODY" | jq -r '.choices[0].message.content // ""')
if echo "$CONTENT" | jq -e 'has("answer")' >/dev/null 2>&1; then
    run_test "schema JSON lands in content (chat, non-stream)" PASS ""
else
    RC=$(echo "$BODY" | jq -r '.choices[0].message.reasoning_content // ""' | head -c 120)
    run_test "schema JSON lands in content (chat, non-stream)" FAIL "content: '$(echo "$CONTENT" | head -c 120)' reasoning: '$RC'"
fi

echo "=== /v1/chat/completions: same request, stream ==="
STREAM=$(curl -s -m 300 -N "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"mlx-serve\", \"max_tokens\": 512, \"temperature\": 0, \"stream\": true,
    \"reasoning_effort\": \"medium\",
    \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"answer\", \"strict\": true, \"schema\": $SCHEMA}},
    \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}]
}")
SCONTENT=$(echo "$STREAM" | grep '^data: {' | sed 's/^data: //' | jq -rj '.choices[0].delta.content // empty' 2>/dev/null)
if echo "$SCONTENT" | jq -e 'has("answer")' >/dev/null 2>&1; then
    run_test "schema JSON lands in delta.content (chat, stream)" PASS ""
else
    run_test "schema JSON lands in delta.content (chat, stream)" FAIL "streamed content: '$(echo "$SCONTENT" | head -c 120)'"
fi

echo "=== /v1/responses: reasoning.effort + text.format json_schema ==="
BODY=$(curl -s -m 300 "$BASE/v1/responses" -H "Content-Type: application/json" -d "{
    \"model\": \"mlx-serve\", \"max_output_tokens\": 2048, \"temperature\": 0, \"stream\": false,
    \"reasoning\": {\"effort\": \"medium\"},
    \"text\": {\"format\": {\"type\": \"json_schema\", \"name\": \"answer\", \"strict\": true, \"schema\": $SCHEMA}},
    \"input\": \"$PROMPT\"
}")
RTEXT=$(echo "$BODY" | jq -r '[.output[]? | select(.type=="message") | .content[]? | select(.type=="output_text") | .text] | join("")')
if echo "$RTEXT" | jq -e 'has("answer")' >/dev/null 2>&1; then
    run_test "schema JSON lands in output_text (responses)" PASS ""
else
    run_test "schema JSON lands in output_text (responses)" FAIL "output_text: '$(echo "$RTEXT" | head -c 120)'"
fi

echo "=== /v1/messages: finite thinking budget + json_schema, stream ==="
STREAM=$(curl -s -m 300 -N "$BASE/v1/messages" -H "Content-Type: application/json" -d "{
    \"model\": \"mlx-serve\", \"max_tokens\": 512, \"temperature\": 0, \"stream\": true,
    \"thinking\": {\"type\": \"enabled\", \"budget_tokens\": 256},
    \"output_config\": {\"format\": {\"type\": \"json_schema\", \"schema\": $SCHEMA}},
    \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}]
}")
MTEXT=$(echo "$STREAM" | sed -n 's/^data: //p' | jq -rj 'select(.type=="content_block_delta" and .delta.type=="text_delta") | .delta.text' 2>/dev/null)
if echo "$MTEXT" | jq -e 'has("answer")' >/dev/null 2>&1; then
    run_test "schema JSON lands in text delta (messages, finite budget)" PASS ""
else
    run_test "schema JSON lands in text delta (messages, finite budget)" FAIL "streamed text: '$(echo "$MTEXT" | head -c 120)'"
fi

grep -q '\[grammar\] finite reasoning budget; rerendered with thinking off' "$LOG" \
    && run_test "finite budgets retain thinking-off fallback" PASS "" \
    || run_test "finite budgets retain thinking-off fallback" FAIL "missing fallback log"
grep -q '\[grammar\] deferring JSON schema' "$LOG" \
    && run_test "supported unlimited request defers grammar" PASS "" \
    || run_test "supported unlimited request defers grammar" FAIL "missing deferral log"
grep -Eq '\[grammar\] reasoning boundary (reached|forced)' "$LOG" \
    && run_test "deferred grammar activates at a boundary" PASS "" \
    || run_test "deferred grammar activates at a boundary" FAIL "missing activation log"

echo
echo "=== $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ] || exit 1
