#!/bin/bash
# DeepSeek-V4-Flash NATIVE engine integration test — env-gated on the local
# mixed-quant mirror (~118 GB; loading takes minutes and wants the machine
# otherwise idle):
#
#   DSV4_TEST_MODEL=~/.mlx-serve/models/ddalcu/DeepSeek-V4-Flash-0731-MLX-Serve-mixed-2-3-8bit \
#       ./tests/test_dsv4.sh
#
# Pins the live serving surface end-to-end: greedy raw-completion prefix (the
# python-oracle ground truth), chat on both thinking arms, DSML tool calling
# (call → result → answer), and streaming delta cleanliness. Booted WITH
# --pld --mtp to prove the scheduler's dsv4 spec hard-off guards hold (the
# app always passes both flags). Complements the hermetic layers: the
# DSV4_MINI fixture/decode-equivalence gates, the template render test, and
# the format-corpus "dsv4-dsml" family.

set -euo pipefail

MODEL="${DSV4_TEST_MODEL:-}"
if [ -z "$MODEL" ]; then
    echo "SKIP: DSV4_TEST_MODEL not set"
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo "FAIL: $MODEL/config.json not found"
    exit 1
fi

PORT="${DSV4_TEST_PORT:-11351}"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
# X's must be TRAILING: macOS mktemp does not substitute a mid-name
# XXXXXX, so `…XXXXXX.log` created a file with that literal name once and
# then collided on every later run ("mkstemp failed: File exists").
LOG=$(mktemp /tmp/dsv4_test_serve.XXXXXX)

# Template byte-pin, when the release's own encoder is on disk. Every prompt
# this script then sends is only meaningful if our transcription still agrees
# with `encoding_dsv4.py` byte for byte — an encoder change (0731 moved
# `reasoning_effort` to low|high|max) would otherwise show up as a vague
# quality regression rather than a diff.
ENCODING_DIR="${DSV4_ENCODING_DIR:-$HOME/.mlx-serve/staging/DeepSeek-V4-Flash-0731/encoding}"
if [ -f "$ENCODING_DIR/encoding_dsv4.py" ]; then
    echo "[0/6] template A/B vs the release encoder"
    python3 "$(dirname "$0")/dsv4_template_ab.py" --encoding "$ENCODING_DIR" | tail -1
else
    echo "[0/6] SKIP template A/B (no encoding_dsv4.py at $ENCODING_DIR)"
fi

"$BIN" --model "$MODEL" --serve --port "$PORT" --pld --mtp > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting for server (load takes minutes at ~118 GB)..."
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

# [1] Raw completion greedy prefix (python-oracle ground truth: " Paris.").
RAW=$(curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])")
check "raw greedy prefix" "$RAW" " Paris."

# [2] Chat, thinking OFF (chat mode): correct arithmetic, clean content.
OFF=$(curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 17 * 23? Answer with just the number."}],"max_tokens":32,"temperature":0}')
check "thinking-off content" "$OFF" '391'
refuse "thinking-off no think leak" "$OFF" '</think>'

# [3] Chat, thinking ON: reasoning split out, content clean.
ON=$(curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 3*7? Answer with just the number."}],"max_tokens":512,"temperature":0,"reasoning_effort":"high"}')
check "thinking-on content" "$ON" '21'
check "thinking-on reasoning present" "$ON" '"reasoning_content"'
refuse "thinking-on no think leak" "$ON" '<think>'

# [4] Streaming: no DSML/think markers in any delta.
STREAM=$(curl -s -m 600 -N "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 5+5? Answer with just the number."}],"max_tokens":64,"temperature":0,"stream":true}')
check "stream content delta" "$STREAM" '10'
refuse "stream no DSML leak" "$STREAM" 'DSML'

# [5] DSML tool call: name + valid-JSON args, finish_reason tool_calls.
TOOLS='[{"type":"function","function":{"name":"get_time","description":"Get the current time in a timezone","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]'
TC=$(curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"What time is it in Tokyo right now? Use the tool.\"}],\"tools\":$TOOLS,\"max_tokens\":400,\"temperature\":0}")
check "tool call name" "$TC" '"name":"get_time"'
check "tool call args" "$TC" 'Tokyo'
check "tool finish reason" "$TC" '"finish_reason":"tool_calls"'
refuse "tool call no DSML leak in content" "$TC" 'DSML'

# [6] Tool round-trip: result consumed into a clean final answer.
RT=$(curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"What time is it in Tokyo right now? Use the tool.\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"arguments\":\"{\\\"timezone\\\":\\\"Asia/Tokyo\\\"}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"2026-07-31 09:14 JST\"}],\"tools\":$TOOLS,\"max_tokens\":300,\"temperature\":0}")
# 0731 phrases the time as "9:14 AM (JST)" (leading zero dropped) — assert
# the semantic content (the tool result's time reached the answer), not the
# checkpoint's phrasing. '9:14' matches both spellings.
check "tool round-trip answer" "$RT" '9:14'
refuse "tool round-trip no DSML leak" "$RT" 'DSML｜tool_calls>'

echo
echo "dsv4: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
