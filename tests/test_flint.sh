#!/bin/bash
# Recursive Flint (flint) live end-to-end on an MLX pack (RecursiveSI/flint-int4):
#
#   FLINT_PACK=/path/to/RecursiveSI/flint-int4 ./tests/test_flint.sh
#
#   [0] advertised as flint                [4] tool round-trip renders the template, not the fallback
#   [1] short answer, no think tags        [5] needle ~6k tokens back (past the 513-token window: the NoPE
#   [2] the model's own <think> is split       full layers carry it)
#       into reasoning_content             [6] prefix reuse: cached tokens, same answer
#   [3] tool call, no markup in content    [7] streaming carries no think/tool markup
#                                          [8] /v1/messages answers
#                                          [9] two concurrent requests decode batched, both answer
#
# Flint decides on each turn whether to think (its template has no switch), so [1] and [2] assert
# the split, never that it thought. Hermetic counterparts: the config-parse and uint8-weight tests in
# engines/mlx/model.zig, the template render test in host/chat/chat.zig, and `flint fixture` in engines/mlx/transformer.zig.

set -euo pipefail

MODEL="${FLINT_PACK:-}"
if [ -z "$MODEL" ]; then echo "SKIP: FLINT_PACK not set"; exit 0; fi
if [ ! -f "$MODEL/config.json" ]; then echo "FAIL: $MODEL/config.json not found"; exit 1; fi

PORT="${FLINT_TEST_PORT:-11373}"
BASE="http://127.0.0.1:$PORT"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
LOG=$(mktemp "${TMPDIR:-/tmp}/flint_test_serve.XXXXXX")
SCRATCH_HOME=$(mktemp -d "${TMPDIR:-/tmp}/flint_test_home.XXXXXX")

HOME="$SCRATCH_HOME" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --ctx-size 32768 > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; rm -rf "$SCRATCH_HOME"; }
trap cleanup EXIT

for _ in $(seq 1 200); do
    grep -q "Model ready (loaded on inference thread)" "$LOG" && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "FAIL: server died during load"; tail -20 "$LOG"; exit 1; fi
    sleep 3
done
grep -q "Model ready (loaded on inference thread)" "$LOG" || { echo "FAIL: model did not load"; tail -20 "$LOG"; exit 1; }

pass=0; fail=0
check() { if grep -qF "$3" <<< "$2"; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1"; echo "  wanted: $3"; echo "  got: $(echo "$2" | head -c 400)"; fail=$((fail+1)); fi; }
check_absent() { if grep -qF "$3" <<< "$2"; then echo "FAIL $1 ('$3' present)"; echo "  got: $(echo "$2" | head -c 400)"; fail=$((fail+1)); else echo "PASS $1"; pass=$((pass+1)); fi; }
chat() { curl -s -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "$1"; }
content() { python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content") or "")'; }
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'

check "[0] advertised as flint" "$(curl -s -m 30 "$BASE/v1/models")" '"architecture":"flint"'

R1=$(chat '{"max_tokens":2000,"temperature":0,"messages":[{"role":"user","content":"What is the capital of France? Answer with one word."}]}')
check "[1] answers Paris" "$(echo "$R1" | content)" "Paris"
check_absent "[1] no think tags in content" "$(echo "$R1" | content)" "think>"

R2=$(chat '{"max_tokens":6000,"temperature":0,"messages":[{"role":"user","content":"A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost, in cents? Think it through."}]}')
check "[2] answer carries 5" "$(echo "$R2" | content)" "5"
check_absent "[2] no think tags in content" "$(echo "$R2" | content)" "think>"
if grep -qF '"reasoning_content"' <<< "$R2"; then echo "  (thought: reasoning_content present)"; fi

T=$(chat '{"max_tokens":4000,"temperature":0,"messages":[{"role":"user","content":"What is the weather in Paris right now? Use the tool."}],"tools":'"$TOOLS"'}')
check "[3] tool call emitted" "$T" '"tool_calls"'
check "[3] Paris argument" "$T" 'Paris'
check_absent "[3] no tool markup" "$T" "<tool_call>"
check_absent "[3] no arg markup" "$T" "<arg_key>"

RT=$(chat '{"max_tokens":4000,"temperature":0,"messages":[
  {"role":"user","content":"What is the weather in Paris? Use the tool."},
  {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}]},
  {"role":"tool","tool_call_id":"c1","content":"{\"temp_c\": 21, \"conditions\": \"partly cloudy\"}"}],"tools":'"$TOOLS"'}')
check "[4] round-trip answer uses the result" "$(echo "$RT" | content)" "21"
check_absent "[4] no fallback turn markup" "$RT" "<start_of_turn>"
check_absent "[4] template rendered" "$(cat "$LOG")" "jinja"

NEEDLE=$(python3 - <<'PY'
import json
filler = " ".join(f"Section {i}: the weather report says clouds drift over hills and rivers flow to the sea." for i in range(300))
msg = "The vault combination for the Aldergate safe is 74-19-52.\n\n" + filler + "\n\nWhat is the vault combination for the Aldergate safe? Answer with just the digits."
print(json.dumps({"max_tokens": 3000, "temperature": 0, "messages": [{"role": "user", "content": msg}]}))
PY
)
DIGITS=$(chat "$NEEDLE" | content | python3 -c 'import re,sys; print(re.sub(r"\D", "", sys.stdin.read()))')
check "[5] needle recovered ~6k tokens back" "$DIGITS" "741952"

REQ6='{"max_tokens":2000,"temperature":0,"messages":[{"role":"user","content":"My dog is called Biscuit and I like teal. What is my dog called? One word."}]}'
A=$(chat "$REQ6"); B=$(chat "$REQ6")
check "[6] first answer" "$(echo "$A" | content)" "Biscuit"
[ "$(echo "$A" | content)" = "$(echo "$B" | content)" ] && { echo "PASS [6] same answer after reuse"; pass=$((pass+1)); } || { echo "FAIL [6] answer changed after reuse"; fail=$((fail+1)); }
CACHED=$(echo "$B" | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["prompt_tokens_details"]["cached_tokens"])')
[ "$CACHED" -gt 0 ] && { echo "PASS [6] prefix cache engaged ($CACHED tokens)"; pass=$((pass+1)); } || { echo "FAIL [6] 0 cached tokens"; fail=$((fail+1)); }

S=$(curl -s -N -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{"stream":true,"max_tokens":3000,"temperature":0,"tools":'"$TOOLS"',
  "messages":[{"role":"user","content":"What is 15% of 80? Brief, no tools."}]}' | grep '^data: {' | python3 -c '
import json, sys
r, c = [], []
for line in sys.stdin:
    for ch in json.loads(line[6:]).get("choices", []):
        d = ch.get("delta", {})
        r.append(d.get("reasoning_content") or ""); c.append(d.get("content") or "")
print("CONTENT:" + "".join(c))')
check "[7] streamed answer carries 12" "$S" "12"
check_absent "[7] no think tags streamed" "$S" "think>"
check_absent "[7] no tool markup streamed" "$S" "<tool_call"

MSG=$(curl -s -m 300 "$BASE/v1/messages" -H 'Content-Type: application/json' -d '{"model":"flint","max_tokens":2000,"messages":[{"role":"user","content":"Name the largest planet in one word."}]}')
check "[8] /v1/messages answers Jupiter" "$MSG" "Jupiter"

P1=$(mktemp "${TMPDIR:-/tmp}/flint_p1.XXXXXX"); P2=$(mktemp "${TMPDIR:-/tmp}/flint_p2.XXXXXX")
chat '{"max_tokens":2000,"temperature":0,"messages":[{"role":"user","content":"What is the capital of Japan? One word."}]}' > "$P1" & J1=$!
chat '{"max_tokens":2000,"temperature":0,"messages":[{"role":"user","content":"Name the largest planet in one word."}]}' > "$P2" & J2=$!
wait "$J1" "$J2"
check "[9] concurrent answer 1" "$(content < "$P1")" "Tokyo"
check "[9] concurrent answer 2" "$(content < "$P2")" "Jupiter"
check "[9] batched decode engaged" "$(cat "$LOG")" "[batched] gdn batched decode engaged"
rm -f "$P1" "$P2"

check_absent "no MLX error" "$(cat "$LOG")" "[mlx]"
echo "---- $pass passed, $fail failed (log: $LOG)"
[ "$fail" -eq 0 ]
