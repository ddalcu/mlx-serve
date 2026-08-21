#!/bin/bash
# Guard: continuing a partial assistant reply extends it — it does not restart.
#
# A conversation ending in an assistant message used to render that message as
# HISTORY and then let add_generation_prompt open a SECOND assistant turn, so
# the model answered the doubled turn by starting over. Continuation ends the
# prompt mid-turn instead, on the partial text.
#
# The observable contract, which is what this pins:
#   - the reply CONTINUES rather than repeating the prefix back
#   - the two surfaces ask for it their own documented way: an explicit
#     `continue_final_message` on /v1/chat/completions, an implicit trailing
#     assistant message on /v1/messages (Anthropic's own behaviour)
#   - without the flag, /v1/chat/completions is unchanged — a trailing
#     assistant message is history and a fresh turn is answered
#
# Greedy (temperature 0) throughout: the assertions are about WHERE the prompt
# ends, and a sampled reply makes a "did it restart?" check a coin flip.
#
# Usage: ./tests/test_continuation.sh [model_dir] [port]

set -u

MODEL=${1:-~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}
PORT=${2:-8133}
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
    --model "$MODEL" >/tmp/mlx-serve-continuation.log 2>&1 &
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

PARTIAL="The three primary colors are red, blue, and"

echo "=== /v1/chat/completions: continue_final_message ==="
BODY=$(curl -s -m 300 "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"mlx-serve\",
    \"messages\": [
        {\"role\": \"user\", \"content\": \"Name the three primary colors in one sentence.\"},
        {\"role\": \"assistant\", \"content\": \"$PARTIAL\"}
    ],
    \"continue_final_message\": true,
    \"temperature\": 0, \"max_tokens\": 40, \"stream\": false
}")
CONT=$(echo "$BODY" | jq -r '.choices[0].message.content // ""')

if [ -n "$CONT" ]; then
    run_test "continuation returns content" PASS ""
else
    run_test "continuation returns content" FAIL "empty: $(echo "$BODY" | head -c 200)"
fi

# The whole point: the model is handed its own unfinished sentence, so what
# comes back is the REST of it — never the prefix again. A restart is the
# doubled-turn bug, and it looks exactly like the prefix being repeated.
if echo "$CONT" | grep -qi "three primary colors are"; then
    run_test "continuation does not restate the prefix" FAIL "restarted: $(echo "$CONT" | head -c 120)"
else
    run_test "continuation does not restate the prefix" PASS ""
fi

echo "=== /v1/chat/completions: WITHOUT the flag, the prompt is a full turn ==="
# Measured on the PROMPT, not on the reply. The first cut of this compared the
# two replies and demanded they differ — but "yellow." is the natural next word
# whether the model is finishing the sentence or answering the question afresh,
# so a working feature failed the test. What the flag changes is where the
# prompt ENDS: a continuation stops mid-turn, an ordinary turn closes the
# assistant message and opens a new one, which costs the close tag plus a turn
# header. Structural, and true of every checkpoint.
TOKENS_CONT=$(echo "$BODY" | jq -r '.usage.prompt_tokens // 0')
TOKENS_PLAIN=$(curl -s -m 300 "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"mlx-serve\",
    \"messages\": [
        {\"role\": \"user\", \"content\": \"Name the three primary colors in one sentence.\"},
        {\"role\": \"assistant\", \"content\": \"$PARTIAL\"}
    ],
    \"temperature\": 0, \"max_tokens\": 40, \"stream\": false
}" | jq -r '.usage.prompt_tokens // 0')

if [ "$TOKENS_CONT" -gt 0 ] && [ "$TOKENS_PLAIN" -gt "$TOKENS_CONT" ]; then
    run_test "the flag shortens the prompt (no close tag, no second turn header)" PASS \
        "" # measured 27 vs 32 on llama-3.2
else
    run_test "the flag shortens the prompt (no close tag, no second turn header)" FAIL \
        "continuation=$TOKENS_CONT plain=$TOKENS_PLAIN — the flag is not reaching the renderer"
fi

echo "=== /v1/messages: a trailing assistant message IS the request ==="
ABODY=$(curl -s -m 300 "$BASE/v1/messages" -H "Content-Type: application/json" \
    -H "anthropic-version: 2023-06-01" -d "{
    \"model\": \"mlx-serve\",
    \"messages\": [
        {\"role\": \"user\", \"content\": \"Name the three primary colors in one sentence.\"},
        {\"role\": \"assistant\", \"content\": \"$PARTIAL\"}
    ],
    \"temperature\": 0, \"max_tokens\": 40, \"stream\": false
}")
ACONT=$(echo "$ABODY" | jq -r '[.content[]? | select(.type=="text") | .text] | join("")')

if [ -n "$ACONT" ]; then
    run_test "/v1/messages continues with no flag" PASS ""
else
    run_test "/v1/messages continues with no flag" FAIL "empty: $(echo "$ABODY" | head -c 200)"
fi
if echo "$ACONT" | grep -qi "three primary colors are"; then
    run_test "/v1/messages does not restate the prefix" FAIL "restarted: $(echo "$ACONT" | head -c 120)"
else
    run_test "/v1/messages does not restate the prefix" PASS ""
fi

echo
echo "=== $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ] || exit 1
