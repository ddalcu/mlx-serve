#!/bin/bash
# Guard: Anthropic `output_config` (the 2026 spelling Claude Code sends) is
# honored on /v1/messages.
#
# Ignored, it served every Claude Code request at the arch default with an
# UNLIMITED thinking budget (live 2026-08-16: 8-minute retries at 16k thinking
# tokens each) and answered `format: json_schema` requests with markdown prose
# the client rejects — which is what fed the retry loop.
#
# The observable contract, which is what this pins:
#   - `effort` is an explicit thinking signal: "none" turns thinking OFF even
#     with tools present (where this arch would default on), any other word
#     turns it ON
#   - `format: {type: "json_schema", schema: ...}` is enforced by the grammar
#     mask: the reply's `text` content is valid JSON carrying the schema's
#     required keys — and it lands in CONTENT, not inside a thinking block
#     (a token-0 mask cannot express "think first, then JSON", so a schema
#     request forces thinking off)
#
# Greedy throughout. Usage: ./tests/test_output_config.sh [model_dir] [port]

set -u

MODEL=${1:-~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}
PORT=${2:-8135}
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

./zig-out/bin/mlx-serve serve --port $PORT --host 127.0.0.1 --log-level info \
    --model "$MODEL" >/tmp/mlx-serve-output-config.log 2>&1 &
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

msg() {
    curl -s -m 300 "$BASE/v1/messages" -H "Content-Type: application/json" \
        -H "anthropic-version: 2023-06-01" -d "$1"
}

TOOLS='[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]'

echo "=== effort \"none\" turns thinking OFF (tools present, where the arch defaults on) ==="
TYPES=$(msg "{
    \"model\": \"mlx-serve\", \"max_tokens\": 512, \"temperature\": 0, \"stream\": false,
    \"output_config\": {\"effort\": \"none\"},
    \"tools\": $TOOLS,
    \"messages\": [{\"role\": \"user\", \"content\": \"Why is the sky blue? One sentence, no tools needed.\"}]
}" | jq -r '[.content[].type] | join(",")')
if echo "$TYPES" | grep -q "thinking"; then
    run_test "effort none suppresses the thinking block" FAIL "got blocks: $TYPES"
else
    run_test "effort none suppresses the thinking block" PASS ""
fi

echo "=== effort \"high\" turns thinking ON ==="
TYPES=$(msg '{
    "model": "mlx-serve", "max_tokens": 1024, "temperature": 0, "stream": false,
    "output_config": {"effort": "high"},
    "messages": [{"role": "user", "content": "Why is the sky blue? One sentence."}]
}' | jq -r '[.content[].type] | join(",")')
if echo "$TYPES" | grep -q "thinking"; then
    run_test "effort high produces a thinking block" PASS ""
else
    run_test "effort high produces a thinking block" FAIL "got blocks: $TYPES"
fi

echo "=== format json_schema is ENFORCED, and lands in content ==="
BODY=$(msg '{
    "model": "mlx-serve", "max_tokens": 512, "temperature": 0, "stream": false,
    "output_config": {"effort": "high", "format": {"type": "json_schema", "schema": {
        "type": "object",
        "properties": {"title": {"type": "string"}, "summary": {"type": "string"}},
        "required": ["title", "summary"], "additionalProperties": false}}},
    "messages": [{"role": "user", "content": "Summarize: The quick brown fox jumps over the lazy dog."}]
}')
TEXT=$(echo "$BODY" | jq -r '[.content[]? | select(.type=="text") | .text] | join("")')
# Both halves in one shot: the text parses as JSON AND carries the required
# keys. Markdown prose (the pre-fix answer) fails the parse; JSON that leaked
# into a thinking block fails because `text` is empty.
if echo "$TEXT" | jq -e 'has("title") and has("summary")' >/dev/null 2>&1; then
    run_test "schema-conforming JSON in content" PASS ""
else
    run_test "schema-conforming JSON in content" FAIL "text: $(echo "$TEXT" | head -c 160)"
fi
if echo "$BODY" | jq -r '[.content[].type] | join(",")' | grep -q "thinking"; then
    run_test "schema request forces thinking off (mask is content-only)" FAIL \
        "thinking block present beside a masked generation"
else
    run_test "schema request forces thinking off (mask is content-only)" PASS ""
fi

echo
echo "=== $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ] || exit 1
