#!/bin/bash
# Tool-call boundary safety test for spec-decode.
#
# Send a streaming chat completion with `tools` AND `enable_pld:true`.
# The auto-disable for tools should kick in (server log:
# `pld=disabled (tools present)` or equivalent), and the streamed tool-call
# arguments must be valid JSON when concatenated, with no spec-drafted token
# leakage (which would manifest as duplicate brace pairs, repeated keys, or
# spurious tokens).
#
# Same coverage for `--drafter` + tools + stream when a Gemma 4 target +
# drafter pair is available.
#
# Usage:
#   TOOL_STREAM_TEST_MODEL=/path/to/model ./tests/test_tool_call_streaming_spec.sh [port]

set -e

PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${TOOL_STREAM_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}"
DRAFTER="${TOOL_STREAM_TEST_DRAFTER:-$HOME/.mlx-serve/models/mlx-community/gemma-4-E4B-it-assistant-bf16}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_tool_call_streaming_spec: model directory not found ($MODEL)."
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing."
    exit 1
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first."
    exit 1
fi

TARGET_TYPE=$(python3 -c "import json; print(json.load(open('$MODEL/config.json')).get('model_type',''))" 2>/dev/null || echo "")
USE_DRAFTER=0
DRAFTER_ARGS=()
case "$TARGET_TYPE" in
    gemma4|gemma4_text)
        if [ -d "$DRAFTER" ] && [ -f "$DRAFTER/config.json" ]; then
            USE_DRAFTER=1
            DRAFTER_ARGS=(--drafter "$DRAFTER")
        fi
        ;;
esac

echo "== tool-call streaming + spec-decode boundary safety =="
echo "  model:   $MODEL  (model_type=$TARGET_TYPE)"
echo "  drafter: $([ "$USE_DRAFTER" = "1" ] && echo "$DRAFTER" || echo "(none)")"
echo

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

LOGFILE=$(mktemp)
"$BINARY" --model "$MODEL" --serve --port "$PORT" --pld "${DRAFTER_ARGS[@]}" --log-level info > "$LOGFILE" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    rm -f "$LOGFILE"
}
trap cleanup EXIT

UP=0
for i in $(seq 1 60); do
    if curl -s -f "$BASE/health" > /dev/null 2>&1; then UP=1; break; fi
    sleep 1
done
if [ "$UP" != "1" ]; then
    echo -e "${RED}FAIL${NC} server did not become healthy in 60s"
    tail -30 "$LOGFILE"
    exit 1
fi
echo "  server up (PID=$SERVER_PID)"
echo

PASS=0
FAIL=0

# Helper: run a streaming tool-call request, return the concatenated arguments
# string and whether the stream emitted a tool_call.
run_stream_toolcall() {
    local label="$1" extra_flags_json="$2"
    local payload
    payload=$(python3 -c "
import json
extra = json.loads('''$extra_flags_json''')
body = {
    'model': 'mlx-serve',
    'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant. Use tools to answer factual questions about system state.'},
        {'role': 'user', 'content': 'Run \"date\" using the shell tool.'}
    ],
    'tools': [{
        'type': 'function',
        'function': {
            'name': 'shell',
            'description': 'Run a shell command',
            'parameters': {
                'type': 'object',
                'properties': {'command': {'type': 'string'}},
                'required': ['command']
            }
        }
    }],
    'max_tokens': 128,
    'temperature': 0.0,
    'stream': True
}
body.update(extra)
print(json.dumps(body))
")
    echo "  ($label) sending streaming tool-call request..."
    local sse
    sse=$(echo "$payload" | curl -s -N -X POST -H "Content-Type: application/json" \
        -d @- "$BASE/v1/chat/completions" 2>&1)

    # Extract concatenated tool_call.function.arguments deltas + name + finish_reason.
    local report
    report=$(echo "$sse" | python3 -c "
import sys, json
args_buf = ''
name = ''
finish = ''
events = 0
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: '): continue
    payload = line[6:].strip()
    if payload == '[DONE]' or not payload: continue
    try:
        ev = json.loads(payload)
    except:
        continue
    events += 1
    for ch in ev.get('choices', []) or []:
        delta = ch.get('delta', {}) or {}
        for tc in delta.get('tool_calls', []) or []:
            fn = tc.get('function', {}) or {}
            if 'name' in fn and fn['name']:
                name = fn['name']
            if 'arguments' in fn and fn['arguments']:
                args_buf += fn['arguments']
        fr = ch.get('finish_reason')
        if fr:
            finish = fr
print('EVENTS=%d' % events)
print('NAME=%s' % name)
print('FINISH=%s' % finish)
print('ARGS=%s' % args_buf)
")
    local n_events name_v finish_v args_v
    n_events=$(echo "$report" | grep '^EVENTS=' | head -1 | cut -d= -f2)
    name_v=$(echo "$report" | grep '^NAME=' | head -1 | cut -d= -f2-)
    finish_v=$(echo "$report" | grep '^FINISH=' | head -1 | cut -d= -f2-)
    args_v=$(echo "$report" | grep '^ARGS=' | head -1 | cut -d= -f2-)

    echo "    events=$n_events name='$name_v' finish='$finish_v' args='$args_v'"

    # Validate.
    if [ "$n_events" -lt 1 ]; then
        echo -e "  ${RED}FAIL${NC} $label: zero SSE events received"
        FAIL=$((FAIL + 1))
        return 1
    fi
    if [ "$finish_v" != "tool_calls" ]; then
        echo -e "  ${RED}FAIL${NC} $label: finish_reason=$finish_v (expected tool_calls)"
        FAIL=$((FAIL + 1))
        return 1
    fi
    if [ "$name_v" != "shell" ]; then
        echo -e "  ${RED}FAIL${NC} $label: tool name='$name_v' (expected 'shell')"
        FAIL=$((FAIL + 1))
        return 1
    fi
    if [ -z "$args_v" ]; then
        echo -e "  ${RED}FAIL${NC} $label: empty tool_call arguments"
        FAIL=$((FAIL + 1))
        return 1
    fi
    # Arguments must be valid JSON, must contain 'command'.
    local valid
    valid=$(python3 -c "
import json, sys
try:
    obj = json.loads('''$args_v''')
    print('OK' if isinstance(obj, dict) and 'command' in obj else 'BAD_SHAPE')
except Exception as e:
    print('BAD_JSON:' + str(e))
")
    if [ "$valid" != "OK" ]; then
        echo -e "  ${RED}FAIL${NC} $label: tool_call arguments not valid JSON object with 'command' (got: $valid)"
        echo "    raw args: $args_v"
        FAIL=$((FAIL + 1))
        return 1
    fi
    # Spec-leak heuristic: count braces — must be 1 open + 1 close.
    local open_n close_n
    open_n=$(python3 -c "print('$args_v'.count('{'))")
    close_n=$(python3 -c "print('$args_v'.count('}'))")
    if [ "$open_n" != "1" ] || [ "$close_n" != "1" ]; then
        echo -e "  ${RED}FAIL${NC} $label: brace count anomaly (open=$open_n close=$close_n) — possible spec-decode leak into tool buffer"
        echo "    raw args: $args_v"
        FAIL=$((FAIL + 1))
        return 1
    fi
    echo -e "  ${GREEN}PASS${NC} $label: args is single-object valid JSON, no leak"
    PASS=$((PASS + 1))
    return 0
}

# ── Test A: streaming + tools + enable_pld:true ──
echo "--- Test A: stream + tools + enable_pld:true ---"
run_stream_toolcall "A pld+tools+stream" '{"enable_pld": true}' || true
echo

# ── Test B: streaming + tools + enable_drafter:true (only when drafter loaded) ──
if [ "$USE_DRAFTER" = "1" ]; then
    echo "--- Test B: stream + tools + enable_drafter:true ---"
    run_stream_toolcall "B drafter+tools+stream" '{"enable_drafter": true}' || true
    echo
fi

# Inspect log for the auto-disable lines we expect.
if grep -E "pld=disabled|drafter=disabled|tools present" "$LOGFILE" > /dev/null 2>&1; then
    echo "  log shows spec auto-disable line(s) (expected):"
    grep -E "pld=disabled|drafter=disabled|tools present" "$LOGFILE" | sed 's/^/    /'
    echo
fi

echo "== summary: $PASS pass / $FAIL fail =="
if [ "$FAIL" -gt 0 ]; then
    echo
    echo "Recent server log:"
    tail -40 "$LOGFILE" | sed 's/^/    /'
    exit 1
fi
echo -e "${GREEN}PASS${NC} tool-call streaming spec-decode boundary safety"
exit 0
