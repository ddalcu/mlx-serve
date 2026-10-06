#!/bin/bash
# MiMo-V2.6-Flash (mimo_v2) live end-to-end on a converted pack (tests/convert_mimo_v2.py):
#
#   MIMO_MODEL=~/.mlx-serve/models/ddalcu/MiMo-V2.6-Flash-MLX-Serve-MXFP4-Q8 ./tests/test_mimo_v2.sh
#
#   [0] advertised as mimo_v2       [4] tool round-trip (native `tool` turn)
#   [1] short answer, thinking off  [5] needle past the 128-token window and a prefill chunk
#   [2] thinking on by default      [6] prefix reuse: cached tokens, same answer
#   [3] parallel tool calls         [7] streaming carries no think/tool markup
#   [8] the checkpoint's MTP heads draft (spec-stats mode=mtp)
#   [9] a whole-file edit: prompt-lookup drafts commit long runs beside the heads
#
# Hermetic counterparts: the config-parse tests in model.zig, the generic-role-header
# render test in chat.zig, the `mimo_v2 fixture` parity test in transformer.zig and the
# `mimo mtp heads` oracle in mimo_mtp.zig.

set -euo pipefail

MODEL="${MIMO_MODEL:-}"
if [ -z "$MODEL" ]; then echo "SKIP: MIMO_MODEL not set"; exit 0; fi
if [ ! -f "$MODEL/config.json" ]; then echo "FAIL: $MODEL/config.json not found"; exit 1; fi

PORT="${MIMO_TEST_PORT:-11359}"
BASE="http://127.0.0.1:$PORT"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
LOG=$(mktemp /tmp/mimo_test_serve.XXXXXX)
SCRATCH_HOME=$(mktemp -d /tmp/mimo_test_home.XXXXXX)

HOME="$SCRATCH_HOME" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --ctx-size 32768 > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; rm -rf "$SCRATCH_HOME"; }
trap cleanup EXIT

for _ in $(seq 1 300); do
    grep -q "Model ready (loaded on inference thread)" "$LOG" && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "FAIL: server died during load"; tail -20 "$LOG"; exit 1; fi
    sleep 3
done
grep -q "Model ready (loaded on inference thread)" "$LOG" || { echo "FAIL: model did not load"; tail -20 "$LOG"; exit 1; }

pass=0; fail=0
check() { if grep -qF "$3" <<< "$2"; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1"; echo "  wanted: $3"; echo "  got: $(echo "$2" | head -c 400)"; fail=$((fail+1)); fi; }
check_absent() { if grep -qF "$3" <<< "$2"; then echo "FAIL $1 ('$3' present)"; echo "  got: $(echo "$2" | head -c 400)"; fail=$((fail+1)); else echo "PASS $1"; pass=$((pass+1)); fi; }
chat() { curl -s -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "$1"; }
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'

M=$(curl -s -m 30 "$BASE/v1/models")
check "[0] advertised as mimo_v2" "$M" '"architecture":"mimo_v2"'

R1=$(chat '{"max_tokens":60,"temperature":0,"enable_thinking":false,"messages":[{"role":"user","content":"What is the capital of France? Answer with one word."}]}')
check "[1] answers Paris" "$R1" "Paris"
check_absent "[1] no think tags" "$R1" "<think>"

R2=$(chat '{"max_tokens":600,"temperature":0,"messages":[{"role":"user","content":"A farmer has 17 sheep. All but 9 run away. How many are left?"}]}')
check "[2] reasoning_content by default" "$R2" '"reasoning_content"'
check "[2] answer carries 9" "$R2" "9"
check_absent "[2] no think tags" "$R2" "</think>"

T=$(chat '{"max_tokens":600,"temperature":0,"messages":[{"role":"user","content":"What is the weather in Paris and in Tokyo right now? Use the tool."}],"tools":'"$TOOLS"'}')
check "[3] tool calls emitted" "$T" '"tool_calls"'
check "[3] Paris call" "$T" 'Paris'
check "[3] Tokyo call" "$T" 'Tokyo'
check_absent "[3] no tool markup" "$T" "<tool_call>"
check_absent "[3] no parameter markup" "$T" "<parameter="

RT=$(chat '{"max_tokens":400,"temperature":0,"enable_thinking":false,"messages":[
  {"role":"user","content":"What is the weather in Paris? Use the tool."},
  {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}]},
  {"role":"tool","tool_call_id":"c1","content":"{\"temp_c\": 21, \"conditions\": \"partly cloudy\"}"}],"tools":'"$TOOLS"'}')
check "[4] round-trip answer uses the result" "$RT" "21"

NEEDLE=$(python3 - <<'PY'
import json
filler = " ".join(f"Section {i}: the weather report says clouds drift over hills and rivers flow to the sea." for i in range(400))
msg = "The vault combination for the Aldergate safe is 74-19-52.\n\n" + filler + "\n\nWhat is the vault combination for the Aldergate safe? Answer with just the digits."
print(json.dumps({"max_tokens": 200, "temperature": 0, "enable_thinking": False, "messages": [{"role": "user", "content": msg}]}))
PY
)
DIGITS=$(chat "$NEEDLE" | python3 -c 'import json,re,sys; print(re.sub(r"\D", "", json.load(sys.stdin)["choices"][0]["message"]["content"]))')
check "[5] needle recovered at ~8k tokens" "$DIGITS" "741952"

REQ6='{"max_tokens":60,"temperature":0,"enable_thinking":false,"messages":[{"role":"user","content":"My dog is called Biscuit and I like teal. What is my dog called?"}]}'
A=$(chat "$REQ6"); B=$(chat "$REQ6")
check "[6] first answer" "$A" "Biscuit"
CA=$(echo "$A" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
CB=$(echo "$B" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
[ "$CA" = "$CB" ] && { echo "PASS [6] same answer after reuse"; pass=$((pass+1)); } || { echo "FAIL [6] answer changed after reuse"; fail=$((fail+1)); }
CACHED=$(echo "$B" | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["prompt_tokens_details"]["cached_tokens"])')
[ "$CACHED" -gt 0 ] && { echo "PASS [6] prefix cache engaged ($CACHED tokens)"; pass=$((pass+1)); } || { echo "FAIL [6] 0 cached tokens"; fail=$((fail+1)); }

S=$(curl -s -N -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{"stream":true,"max_tokens":400,"temperature":0,
  "messages":[{"role":"user","content":"What is 15% of 80? Brief."}]}' | grep '^data: {' | python3 -c '
import json, sys
r, c = [], []
for line in sys.stdin:
    for ch in json.loads(line[6:]).get("choices", []):
        d = ch.get("delta", {})
        r.append(d.get("reasoning_content") or ""); c.append(d.get("content") or "")
print("REASONING:" + "".join(r)); print("CONTENT:" + "".join(c))')
check "[7] streamed answer carries 12" "$S" "12"
check_absent "[7] no think tags streamed" "$S" "<think"
check_absent "[7] no tool markup streamed" "$S" "<tool_call"

EDIT=$(python3 - "$(dirname "$0")/convert_mimo_v2.py" <<'PY'
import json, sys
src = open(sys.argv[1]).read()[:6000]
msg = "Here is a file:\n```python\n" + src + "\n```\nRename the function `fp8_dequant` to `dequant_fp8` everywhere and return the complete updated file, nothing else."
print(json.dumps({"max_tokens": 1500, "temperature": 0, "enable_thinking": False, "messages": [{"role": "user", "content": msg}]}))
PY
)
E=$(chat "$EDIT" || true)
check "[9] edit answered" "$E" "dequant_fp8"
LOOKUP=$(grep -oE "lookup=[0-9]+" "$LOG" | tail -1 | cut -d= -f2)
[ "${LOOKUP:-0}" -gt 0 ] && { echo "PASS [9] lookup drafts engaged ($LOOKUP)"; pass=$((pass+1)); } || { echo "FAIL [9] no lookup rounds"; fail=$((fail+1)); }

check "[8] MTP heads loaded" "$(cat "$LOG")" "MiMo MTP heads ready (3 heads"
check "[8] MTP drafted" "$(cat "$LOG")" "[spec-stats] mode=mtp"
check_absent "[8] no draft past the last head" "$(grep '\[spec-stats\] mode=mtp' "$LOG")" "depth=6"

check_absent "[log] no MLX error" "$(cat "$LOG")" "[mlx]"
echo
echo "mimo_v2 integration: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
