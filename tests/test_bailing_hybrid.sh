#!/bin/bash
# Ling 3.0 (bailing_hybrid) integration test — env-gated on a local checkpoint
# (the 4.2 GB 4-bit mirror):
#
#   LING_TEST_MODEL=~/.mlx-serve/models/rapid-mlx/Ling-3.0-tiny-MLX-4bit \
#       ./tests/test_bailing_hybrid.sh
#
# The arch is a KDA + MLA hybrid: three Kimi-Delta-Attention (linear,
# fixed-size recurrent state) layers for every Multi-head-Latent-Attention
# layer. Both mixers are new here, so the checks are chosen to fail loudly if
# either is wrong rather than to pin the model's prose:
#
#   [1] short answer            — the forward composes at all
#   [2] thinking split          — the template's `thinking_option` default is
#                                 'on', so a request that names no preference
#                                 must come back with reasoning_content
#   [3] GLM-style tool call     — <tool_call>name<arg_key>…<arg_value> grammar
#   [4] tool round-trip         — the result reaches the answer
#   [5] needle at long range    — MLA's rope/cache and the KDA recurrence both
#                                 have to be right thousands of tokens back;
#                                 this is the check a broken gate index, a
#                                 mis-strided per-channel decay or a wrong
#                                 rope mode fails while [1]-[4] still pass
#   [6] prefix reuse            — the same prompt twice reports cached tokens
#                                 AND returns the same answer (the hybrid
#                                 SSM-checkpoint path)
#   [7] streaming cleanliness   — no think tags or tool markup in the deltas
#
# Complements the hermetic layers: the config-parse test in model.zig, the
# vectorized-gate kernel equivalence and kdaGateChain tests in transformer.zig,
# and the asymmetric-KV-cache test.

set -euo pipefail

MODEL="${LING_TEST_MODEL:-}"
if [ -z "$MODEL" ]; then
    echo "SKIP: LING_TEST_MODEL not set"
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo "FAIL: $MODEL/config.json not found"
    exit 1
fi

PORT="${LING_TEST_PORT:-11357}"
BASE="http://127.0.0.1:$PORT"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
LOG=$(mktemp /tmp/ling_test_serve.XXXXXX)

"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
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
# /health answers as soon as the socket binds — the model is still loading
# behind it. Wait for the load's own ready line, timeout scaled to size.
MODEL_MB=$(du -sm "$MODEL" 2>/dev/null | awk '{print $1}')
READY_SECS=$(( 300 + ${MODEL_MB:-0} / 100 ))
for _ in $(seq 1 $((READY_SECS / 3)) ); do
    grep -q "Model ready (loaded on inference thread)" "$LOG" && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: server died during load"; tail -20 "$LOG"; exit 1
    fi
    sleep 3
done
if ! grep -q "Model ready (loaded on inference thread)" "$LOG"; then
    echo "FAIL: model did not finish loading in ${READY_SECS}s"; tail -20 "$LOG"; exit 1
fi

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

# [0] The model is discovered under its own architecture.
M=$(curl -s -m 30 "$BASE/v1/models")
check "[0] advertised as bailing_hybrid" "$M" '"architecture":"bailing_hybrid"'
check "[0] chat capability" "$M" '"chat"'

# [1] Short deterministic answer — thinking off so the answer is the content.
R1=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "mlx-serve", "stream": false, "max_tokens": 60, "temperature": 0,
  "enable_thinking": false,
  "messages": [{"role": "user", "content": "What is the capital of France? Answer with one word."}]}')
check "[1] answers Paris" "$R1" "Paris"
check_absent "[1] no think tags leaked" "$R1" "<think>"
check_absent "[1] no role markers leaked" "$R1" "<role>"

# [2] A request that names NO thinking preference gets the template's own
# default (on) — the mode is otherwise unreachable through our API.
R2=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "mlx-serve", "stream": false, "max_tokens": 500, "temperature": 0.6,
  "messages": [{"role": "user", "content": "A farmer has 17 sheep. All but 9 run away. How many are left?"}]}')
check "[2] reasoning_content present by default" "$R2" "reasoning_content"
check "[2] answer content carries 9" "$R2" "9"
check_absent "[2] no think tags in the body" "$R2" "</think>"

# [3] Tool call (GLM-style arg_key/arg_value grammar).
T=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "mlx-serve", "stream": false, "max_tokens": 500, "temperature": 0.3,
  "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}')
check "[3] tool call emitted" "$T" '"tool_calls"'
check "[3] tool name" "$T" '"get_weather"'
check "[3] city arg" "$T" 'Paris'
check_absent "[3] no tool markup leaked" "$T" "<tool_call>"
check_absent "[3] no arg markup leaked" "$T" "<arg_key>"

# [4] Tool round-trip: the result reaches the answer.
RT=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "mlx-serve", "stream": false, "max_tokens": 400, "temperature": 0.3,
  "enable_thinking": false,
  "messages": [
    {"role": "user", "content": "What is the weather in Paris? Use the tool."},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 21, \"conditions\": \"partly cloudy\"}"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}')
check "[4] round-trip answer uses the result" "$RT" "21"

# [5] Needle at long range. The filler is long enough to cross several prefill
# chunks and grow the MLA cache repeatedly; a wrong per-channel gate stride in
# the KDA kernel or a wrong rope mode in MLA loses the fact while every short
# prompt above still reads fine.
NEEDLE=$(python3 - <<'PY'
import json
filler = ('The archives of the northern library are kept at a steady twelve degrees, '
          'and the shelves are ordered by the year of acquisition rather than by subject. ')
n = 90
body = filler * n + 'IMPORTANT FACT: the vault combination for the Aldergate safe is 74-19-52. ' + filler * n
msg = body + '\n\nQuestion: what is the vault combination for the Aldergate safe? Answer with just the digits.'
print(json.dumps({"model": "mlx-serve", "stream": False, "max_tokens": 200,
                  "temperature": 0, "enable_thinking": False,
                  "messages": [{"role": "user", "content": msg}]}))
PY
)
R5=$(curl -s -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "$NEEDLE")
check "[5] needle recovered at long range" "$R5" "74-19-52"

# [6] Prefix reuse: the second identical request reports cached tokens and
# must produce the SAME answer — a hybrid restores recurrent state, not just
# a KV slice, so a broken restore shows up as a changed answer here.
REQ6='{"model": "mlx-serve", "stream": false, "max_tokens": 60, "temperature": 0,
  "enable_thinking": false,
  "messages": [{"role": "user", "content": "My dog is called Biscuit and I like teal. What is my dog called?"}]}'
A=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ6")
B=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ6")
check "[6] first answer" "$A" "Biscuit"
check "[6] second answer after reuse" "$B" "Biscuit"
CACHED=$(echo "$B" | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["prompt_tokens_details"]["cached_tokens"])')
if [ "$CACHED" -gt 0 ]; then
    echo "PASS [6] prefix cache engaged ($CACHED tokens)"; pass=$((pass+1))
else
    echo "FAIL [6] prefix cache reported 0 cached tokens"; fail=$((fail+1))
fi

# [7] Streaming: deltas carry no think tags or tool markup.
S=$(curl -s -N -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "mlx-serve", "stream": true, "max_tokens": 250, "temperature": 0,
  "enable_thinking": false,
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
check "[7] streamed content arrived" "$S" "CONTENT:"
check "[7] streamed answer carries 12" "$S" "12"
check_absent "[7] no think tags in streamed content" "$S" "<think"
check_absent "[7] no tool markup in streamed content" "$S" "<tool_call"
check_absent "[7] no role markers in streamed content" "$S" "<role>"

echo
echo "bailing_hybrid integration: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
