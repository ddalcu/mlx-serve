#!/bin/bash
# Muse-Glimmer-30B (muse_glimmer) integration test — env-gated on a local
# checkpoint (the 33 GB 8-bit mirror):
#
#   MUSE_TEST_MODEL=~/.mlx-serve/models/ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit \
#       ./tests/test_muse_glimmer.sh
#
# Pins the live surface end-to-end: tokenizer digit grouping (the "8 4" echo
# class), channel split (to=self reasoning / to=user content), the committed
# to=user channel on silent no-tools requests (no reasoning pass at all), an
# ATEM tool round (call → typed args → result → answer), and streaming delta
# cleanliness (no channel markers or ` to=` header fragments in content).
# Complements the hermetic layers: the byte-pinned render test, the muse
# format-corpus entries, and the channel/ATEM/pretokenizer unit tests.

set -euo pipefail

MODEL="${MUSE_TEST_MODEL:-}"
if [ -z "$MODEL" ]; then
    echo "SKIP: MUSE_TEST_MODEL not set"
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo "FAIL: $MODEL/config.json not found"
    exit 1
fi

PORT="${MUSE_TEST_PORT:-11351}"
BASE="http://127.0.0.1:$PORT"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
LOG=$(mktemp /tmp/muse_test_serve.XXXXXX)

# Optional: MUSE_TEST_DRAFTER points at the DFlash assistant — the whole
# suite must stay green with speculative decoding engaged.
DRAFTER_ARGS=()
if [ -n "${MUSE_TEST_DRAFTER:-}" ]; then
    DRAFTER_ARGS=(--drafter "$MUSE_TEST_DRAFTER")
fi

# macOS ships bash 3.2, where "${arr[@]}" on an EMPTY array under `set -u` is
# an unbound-variable error — so the no-drafter path died before the server
# even booted. The +expansion form yields nothing when unset instead.
"$BIN" --model "$MODEL" ${DRAFTER_ARGS[@]+"${DRAFTER_ARGS[@]}"} --serve --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting for server..."
for _ in $(seq 1 120); do
    curl -s -m 2 "$BASE/health" > /dev/null 2>&1 && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: server died during load"; tail -20 "$LOG"; exit 1
    fi
    sleep 3
done

pass=0; fail=0
check() { # name, got, expected-substring
    if echo "$2" | grep -qF "$3"; then
        echo "PASS $1"; pass=$((pass+1))
    else
        echo "FAIL $1"; echo "  wanted substring: $3"; echo "  got: $(echo "$2" | head -c 400)"; fail=$((fail+1))
    fi
}
check_absent() { # name, got, forbidden-substring
    if echo "$2" | grep -qF "$3"; then
        echo "FAIL $1 (forbidden '$3' present)"; echo "  got: $(echo "$2" | head -c 400)"; fail=$((fail+1))
    else
        echo "PASS $1"; pass=$((pass+1))
    fi
}

# [1] Tokenizer digit grouping: "84" must be ONE token (the per-digit split
# echoed "8 4" live 2026-08-10). 84 = id 6601 in this vocab.
TOK=$(curl -s "$BASE/tokenize" -H 'Content-Type: application/json' -d '{"content": "What is 84?"}')
check "[1] digit grouping (84 = one token)" "$TOK" "6601"

# [2] Non-streaming chat with thinking ON (explicit opt-in — the default for a
# no-tools request is now the committed to=user channel): reasoning and content
# split; the INVARIANT is the split, the model's exact words are a checkpoint
# expectation.
R=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "mlx-serve", "stream": false, "max_tokens": 250, "enable_thinking": true,
  "messages": [{"role": "user", "content": "What is 84 * 3 / 2? Answer briefly."}]}')
check "[2] answer content carries 126" "$R" "126"
check "[2] reasoning_content present" "$R" "reasoning_content"
check_absent "[2] no channel markers in body" "$R" "<|message|>"
check_absent "[2] no header text in body" "$R" "to=user"

# [2b] SILENT no-tools request: the prompt commits ` to=user<|message|>`, so no
# reasoning pass runs at all — content only, no reasoning_content, and the
# first visible token arrives without a hidden thinking pass in front of it.
# An explicit enable_thinking:false lands on the same committed-channel path.
RB=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "mlx-serve", "stream": false, "max_tokens": 250,
  "messages": [{"role": "user", "content": "What is 84 * 3 / 2? Answer briefly."}]}')
check "[2b] answer content carries 126" "$RB" "126"
check_absent "[2b] no reasoning_content (channel committed in prompt)" "$RB" "reasoning_content"
check_absent "[2b] no channel markers in body" "$RB" "<|message|>"
check_absent "[2b] no header text in body" "$RB" "to=user"

# [3] ATEM tool call: name + typed args.
T=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "mlx-serve", "stream": false, "max_tokens": 400,
  "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}')
check "[3] tool call emitted" "$T" '"tool_calls"'
check "[3] tool name" "$T" '"get_weather"'
check "[3] city arg" "$T" 'Paris'
check_absent "[3] no ATEM markup leaked" "$T" "<atem:"

# [4] Tool round-trip: result flows into the answer.
RT=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "mlx-serve", "stream": false, "max_tokens": 250,
  "messages": [
    {"role": "user", "content": "What is the weather in Paris? Use the tool."},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 21, \"conditions\": \"partly cloudy\"}"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}')
check "[4] round-trip answer uses the result" "$RT" "21"

# [5] Streaming: content deltas clean of markers and header fragments.
S=$(curl -s -N -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "mlx-serve", "stream": true, "max_tokens": 250,
  "messages": [{"role": "user", "content": "What is 15% of 80? Brief."}]}' \
  | grep '^data: {' | python3 -c '
import json, sys
content = []
for line in sys.stdin:
    d = json.loads(line[6:])
    for ch in d.get("choices", []):
        c = ch.get("delta", {}).get("content")
        if c: content.append(c)
print("CONTENT:" + "".join(content))')
check "[5] streamed content arrived" "$S" "CONTENT:"
check "[5] streamed answer carries 12" "$S" "12"
check_absent "[5] no markers in streamed content" "$S" "<|"
check_absent "[5] no header text in streamed content" "$S" "to=user"

echo
echo "muse_glimmer integration: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
