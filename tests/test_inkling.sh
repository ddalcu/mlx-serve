#!/bin/bash
# Inkling Small (inkling_mm_model) integration test — env-gated on a local
# checkpoint (the ~112 GB REAP25 build; loading takes minutes and wants the
# machine otherwise idle):
#
#   INKLING_TEST_MODEL=~/.cache/huggingface/hub/models--pipenetwork--Inkling-Small-MLX-REAP25-4bit/snapshots/<hash> \
#       ./tests/test_inkling.sh
#
# Pins the live chat surface end-to-end: greedy raw-completion equivalence
# (prefix vs the reference ground truth), channel-marker stripping on both
# thinking arms, streaming delta cleanliness, and a full tool round
# (call → result → answer). Complements the hermetic layers: the fixture
# parity units (INKLING_FIXTURES), the template render test, and the format
# corpus entries.

set -euo pipefail

MODEL="${INKLING_TEST_MODEL:-}"
if [ -z "$MODEL" ]; then
    echo "SKIP: INKLING_TEST_MODEL not set"
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo "FAIL: $MODEL/config.json not found"
    exit 1
fi

PORT="${INKLING_TEST_PORT:-11341}"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
# Trailing X's: macOS mktemp does not substitute a mid-name XXXXXX.
LOG=$(mktemp /tmp/inkling_test_serve.XXXXXX)

"$BIN" --model "$MODEL" --serve --port "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting for server (load takes minutes at 112 GB)..."
for _ in $(seq 1 200); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: server died during load"; tail -20 "$LOG"; exit 1
    fi
    sleep 3
done
curl -s -m 3 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' || { echo "FAIL: no health"; exit 1; }

pass=0; fail=0
check() { # name, got, expected-substring
    if echo "$2" | grep -qF "$3"; then
        echo "PASS: $1"; pass=$((pass+1))
    else
        echo "FAIL: $1"; echo "  got:      $2"; echo "  expected: $3"; fail=$((fail+1))
    fi
}
refuse() { # name, got, forbidden-substring
    if echo "$2" | grep -qF "$3"; then
        echo "FAIL: $1 (leaked '$3')"; echo "  got: $2"; fail=$((fail+1))
    else
        echo "PASS: $1"; pass=$((pass+1))
    fi
}

# [1] Raw completion greedy PREFIX. Only the prefix is a contract: INT4
#     kernel-order divergence past the first tokens is the sanctioned class, and
#     this checkpoint is 2-bit REAP-pruned — it answers " Paris." correctly and
#     then repeats the sentence instead of moving on to Germany. That is the
#     checkpoint's continuation behaviour, not a serving defect: identical with
#     MLX_SERVE_SLIDING_BLOCK_TRIM=0 (and the trim never engages at this prompt
#     length), and all 12 chat/tool/thinking checks below pass. Asserting the
#     whole sentence made this a capability expectation, so it asserted more
#     than the comment claimed.
RAW=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","prompt":"The capital of France is","max_tokens":16,"temperature":0}' \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])")
check "raw greedy prefix" "$RAW" " Paris."
refuse "raw completion leaks no Inkling markers" "$RAW" "<|"
echo "  raw continuation (informational): $RAW"

# [2] Chat, thinking OFF: clean content, no channel markers.
OFF=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":32,"temperature":0}')
check "thinking-off content" "$OFF" '"content":"4"'
refuse "thinking-off no marker leak" "$OFF" '<|content_text|>'

# [3] Chat, thinking ON: reasoning split out, content clean.
ON=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 3*7? Answer with just the number."}],"max_tokens":200,"temperature":0,"reasoning_effort":"low"}')
check "thinking-on content" "$ON" '"content":"21"'
check "thinking-on reasoning present" "$ON" '"reasoning_content"'
refuse "thinking-on no marker leak" "$ON" '<|end_message|>'

# [4] Streaming with thinking: no marker tokens in any delta.
STREAM=$(curl -s -m 300 -N "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 5+5? Answer with just the number."}],"max_tokens":200,"temperature":0,"reasoning_effort":"low","stream":true}')
check "stream content delta" "$STREAM" '"content":"10"'
refuse "stream no marker leak" "$STREAM" '<|content_'

# [5] Tool call: name + valid-JSON args, finish_reason tool_calls.
TOOLS='[{"type":"function","function":{"name":"get_time","description":"Get the current time in a timezone","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]'
TC=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"What time is it in Tokyo right now? Use the tool.\"}],\"tools\":$TOOLS,\"max_tokens\":300,\"temperature\":0}")
check "tool call name" "$TC" '"name":"get_time"'
check "tool call args" "$TC" 'Asia/Tokyo'
check "tool finish reason" "$TC" '"finish_reason":"tool_calls"'

# [6] Tool round-trip: result consumed into a clean final answer.
RT=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"What time is it in Tokyo right now? Use the tool.\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"arguments\":\"{\\\"timezone\\\":\\\"Asia/Tokyo\\\"}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"2026-07-31 09:14 JST\"}],\"tools\":$TOOLS,\"max_tokens\":200,\"temperature\":0}")
check "tool round-trip answer" "$RT" '09:14'
refuse "tool round-trip no marker leak" "$RT" '<|message_'

echo
echo "inkling: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
