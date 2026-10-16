#!/bin/bash
# GLM-5.3 (glm_moe_dsa) live end-to-end on an mlx-stream pack (the trunk beside an experts.bin bank), env-gated
# (the load takes minutes and wants the machine otherwise idle; GLM53_TEST_FLAGS adds server flags):
#
#   GLM53_TEST_MODEL=<pack> GLM53_TEST_FLAGS="--wired-margin-gib 2" ./tests/test_glm53.sh
#
#   [0] advertised as glm_moe_dsa         [3] one tool call, no tool markup
#   [1] short answer, thinking off        [4] tool round-trip uses the result
#   [2] thinking on by default            [5] mlx-stream serves the pack, no MLX error
#   [2b] low effort answers, no think tags
#
# Hermetic counterparts: the glm_moe_dsa parse and routing tests in model.zig, the GLM-5.3 context test in
# arch/mlx_stream.zig, and the GLM effort and tool-history render tests in chat.zig.

set -euo pipefail

MODEL="${GLM53_TEST_MODEL:-}"
if [ -z "$MODEL" ]; then echo "SKIP: GLM53_TEST_MODEL not set"; exit 0; fi
if [ ! -f "$MODEL/config.json" ]; then echo "FAIL: $MODEL/config.json not found"; exit 1; fi
if [ ! -f "$MODEL/experts.bin" ]; then echo "FAIL: $MODEL/experts.bin not found (not an mlx-stream pack)"; exit 1; fi

PORT="${GLM53_TEST_PORT:-11371}"
BASE="http://127.0.0.1:$PORT"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
LOG=$(mktemp /tmp/glm53_test_serve.XXXXXX)
SCRATCH_HOME=$(mktemp -d /tmp/glm53_test_home.XXXXXX)
EXTRA_FLAGS=()
[ -n "${GLM53_TEST_FLAGS:-}" ] && read -ra EXTRA_FLAGS <<< "$GLM53_TEST_FLAGS"

HOME="$SCRATCH_HOME" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; rm -rf "$SCRATCH_HOME"; }
trap cleanup EXIT

echo "waiting for server (load takes minutes)..."
for _ in $(seq 1 600); do
    curl -s -m 2 "$BASE/health" > /dev/null 2>&1 && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "FAIL: server died during load"; tail -20 "$LOG"; exit 1; fi
    sleep 3
done
curl -s -m 3 "$BASE/health" | grep -q '"ok"' || { echo "FAIL: no health"; tail -20 "$LOG"; exit 1; }

pass=0; fail=0
check() { if grep -qF "$3" <<< "$2"; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1"; echo "  wanted: $3"; echo "  got: $(echo "$2" | head -c 400)"; fail=$((fail+1)); fi; }
check_absent() { if grep -qF "$3" <<< "$2"; then echo "FAIL $1 ('$3' present)"; echo "  got: $(echo "$2" | head -c 400)"; fail=$((fail+1)); else echo "PASS $1"; pass=$((pass+1)); fi; }
chat() { curl -s -m 1800 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "$1"; }
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'

M=$(curl -s -m 30 "$BASE/v1/models")
check "[0] advertised as glm_moe_dsa" "$M" '"architecture":"glm_moe_dsa"'

R1=$(chat '{"max_tokens":60,"temperature":0,"enable_thinking":false,"messages":[{"role":"user","content":"What is the capital of France? Answer with one word."}]}')
check "[1] answers Paris" "$R1" "Paris"
check_absent "[1] no reasoning" "$R1" '"reasoning_content"'

# No reasoning_effort: the template's default effort (max) reasons before it answers.
R2=$(chat '{"max_tokens":2000,"temperature":0,"messages":[{"role":"user","content":"What is 3*7? Answer with just the number."}]}')
check "[2] reasoning_content by default" "$R2" '"reasoning_content"'
check "[2] answer 21" "$R2" "21"
check_absent "[2] no think tags" "$R2" "</think>"

# Low effort: the template still opens <think>, and the model may close it at once, with no reasoning.
R2B=$(chat '{"max_tokens":2000,"temperature":0,"reasoning_effort":"low","messages":[{"role":"user","content":"What is 3*7? Answer with just the number."}]}')
check "[2b] low effort answer 21" "$R2B" "21"
check_absent "[2b] no think open tag" "$R2B" "<think>"
check_absent "[2b] no think close tag" "$R2B" "</think>"

T=$(chat '{"max_tokens":2000,"temperature":0,"reasoning_effort":"low","messages":[{"role":"user","content":"What is the weather in Paris right now? Use the tool."}],"tools":'"$TOOLS"'}')
check "[3] tool call name" "$T" '"name":"get_weather"'
check "[3] tool call args" "$T" 'Paris'
check "[3] tool finish reason" "$T" '"finish_reason":"tool_calls"'
check_absent "[3] no tool markup" "$T" "<tool_call>"
check_absent "[3] no arg markup" "$T" "<arg_key>"

RT=$(chat '{"max_tokens":2000,"temperature":0,"reasoning_effort":"low","messages":[
  {"role":"user","content":"What is the weather in Paris? Use the tool."},
  {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}]},
  {"role":"tool","tool_call_id":"c1","content":"{\"temp_c\": 21, \"conditions\": \"partly cloudy\"}"}],"tools":'"$TOOLS"'}')
check "[4] round-trip answer uses the result" "$RT" "21"
check_absent "[4] template rendered" "$(cat "$LOG")" "jinja render failed"

check "[5] mlx-stream loads the pack" "$(grep '\[mlx-stream\] loaded' "$LOG" || true)" 'tensors from'
check "[5] the glm_moe_dsa arch serves it" "$(grep 'glm_moe_dsa: admission' "$LOG" || true)" 'prompt /'
check_absent "[5] no MLX error" "$(cat "$LOG")" '[mlx]'

echo
echo "glm53: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
