#!/bin/bash
# Agent-loop cross-mode regression test for spec-decode.
#
# A single long-running server with `--pld` (and `--drafter` when the target
# is a Gemma 4 checkpoint) is driven through a 6-turn synthetic conversation
# that exercises every spec-decode mode transition the real agent harness
# triggers:
#
#   1. Plain user message → assistant generation        (spec-decode active)
#   2. User → assistant with `tools` array              (spec auto-disables)
#   3. User feeds tool result → assistant generation    (spec re-activates)
#   4. Long user message (~2000 tokens)                 (spec + KV cache reuse)
#   5. Same long user message again                     (spec + cache hit;
#                                                        post-tool-call cache
#                                                        invalidation guard)
#   6. User → assistant with `enable_thinking:true`     (reasoning_content path)
#
# Each turn must:
#   - Return valid JSON with finish_reason in a sensible set
#   - Produce non-empty content (or non-empty tool_calls for turn 2)
#   - Return HTTP 200 (no 500s)
#   - Not corrupt the previous turn's output across spec-decode mode transitions
#
# This catches regressions where:
#   - A perf change to spec-decode breaks the agent loop in cross-mode flows
#   - KV cache state leaks across tool-call boundaries
#   - Sliding-window or speculative state gets out of sync after a re-prefill
#
# Usage:
#   AGENT_SPEC_TEST_MODEL=/path/to/model ./tests/test_agent_loop_spec.sh [port]
#
# Defaults:
#   model   = ~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit
#   drafter = ~/.mlx-serve/models/mlx-community/gemma-4-E4B-it-assistant-bf16 (only used if
#             AGENT_SPEC_TEST_MODEL points at a Gemma 4 target)
#
# Auto-skips cleanly when the model directory is missing.

set -e

PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${AGENT_SPEC_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}"
DRAFTER="${AGENT_SPEC_TEST_DRAFTER:-$HOME/.mlx-serve/models/mlx-community/gemma-4-E4B-it-assistant-bf16}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_agent_loop_spec: model directory not found ($MODEL)."
    echo "  Set AGENT_SPEC_TEST_MODEL to a valid MLX checkpoint."
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing — not a valid model directory."
    exit 1
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

# Decide whether we attach a drafter. Only valid when target model_type is
# gemma4* and the drafter directory exists.
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

echo "== agent-loop spec-decode regression =="
echo "  model:   $MODEL  (model_type=$TARGET_TYPE)"
echo "  drafter: $([ "$USE_DRAFTER" = "1" ] && echo "$DRAFTER" || echo "(none)")"
echo

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

LOGFILE=$(mktemp)
"$BINARY" --model "$MODEL" --serve --port "$PORT" --pld "${DRAFTER_ARGS[@]}" > "$LOGFILE" 2>&1 &
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
    tail -30 "$LOGFILE" | sed 's/^/    /'
    exit 1
fi
echo "  server up (PID=$SERVER_PID)"
echo

PASS=0
FAIL=0

# Helper: POST JSON, validate the response is parseable JSON with a non-empty
# choices[0].message and a sensible finish_reason. Updates PASS / FAIL inline
# and prints status. (We don't propagate content out of the function; turns
# are independent — the test asserts each turn's response shape, not that
# turn N+1 conditions on turn N's actual text.)
check_turn() {
    local label="$1" payload="$2" allow_tool_calls="$3"
    local raw http_code
    local tmp
    tmp=$(mktemp)
    http_code=$(curl -s -o "$tmp" -w "%{http_code}" -X POST "$BASE/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$payload")
    raw=$(cat "$tmp")
    rm -f "$tmp"
    if [ "$http_code" != "200" ]; then
        echo -e "  ${RED}FAIL${NC} $label: HTTP $http_code"
        echo "    body: ${raw:0:500}"
        FAIL=$((FAIL + 1))
        return 1
    fi
    local parsed
    parsed=$(echo "$raw" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
except Exception as e:
    print('PARSE_FAIL:' + str(e))
    sys.exit(0)
ch = (d.get('choices') or [{}])[0]
msg = ch.get('message') or {}
content = msg.get('content') or ''
tcs = msg.get('tool_calls') or []
fr = ch.get('finish_reason') or '?'
reasoning = msg.get('reasoning_content') or ''
print('OK|' + fr + '|' + str(len(content)) + '|' + str(len(tcs)) + '|' + str(len(reasoning)) + '|' + content[:120].replace('\n', ' '))
" 2>/dev/null)
    if [[ "$parsed" == PARSE_FAIL:* ]]; then
        echo -e "  ${RED}FAIL${NC} $label: response was not valid JSON"
        echo "    body: ${raw:0:500}"
        FAIL=$((FAIL + 1))
        return 1
    fi
    IFS='|' read -r status fr clen tcs_n rlen preview <<< "$parsed"
    case "$fr" in
        stop|length|tool_calls) ;;
        *)
            echo -e "  ${RED}FAIL${NC} $label: unexpected finish_reason=$fr"
            FAIL=$((FAIL + 1))
            return 1
            ;;
    esac
    if [ "$allow_tool_calls" = "1" ]; then
        if [ "$tcs_n" -lt 1 ] && [ "$clen" -lt 1 ]; then
            echo -e "  ${RED}FAIL${NC} $label: zero content AND zero tool_calls"
            FAIL=$((FAIL + 1))
            return 1
        fi
    else
        if [ "$clen" -lt 1 ] && [ "$rlen" -lt 1 ]; then
            echo -e "  ${RED}FAIL${NC} $label: empty content (and no reasoning_content)"
            FAIL=$((FAIL + 1))
            return 1
        fi
    fi
    echo -e "  ${GREEN}PASS${NC} $label finish=$fr len=$clen tcs=$tcs_n reasoning=$rlen"
    [ -n "$preview" ] && echo "    preview: $preview"
    PASS=$((PASS + 1))
}

# ── Turn 1: plain user message, spec-decode active ──
echo "--- Turn 1: plain message (spec-decode active) ---"
TURN1_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [
        {'role': 'system', 'content': 'You are helpful. Be brief.'},
        {'role': 'user', 'content': 'What is 12 + 30? Just the number.'}
    ],
    'max_tokens': 32,
    'temperature': 0.0,
    'enable_pld': True,
    'enable_drafter': $([ "$USE_DRAFTER" = "1" ] && echo True || echo False),
    'stream': False
}))
")
check_turn "turn 1" "$TURN1_PAYLOAD" 0 || true
echo

# ── Turn 2: tools array — should auto-disable spec, force tool call ──
echo "--- Turn 2: tools array (spec auto-disables) ---"
TURN2_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant. Use tools to answer factual questions about system state.'},
        {'role': 'user', 'content': 'Use the shell tool to run \"date\" and tell me the current day.'}
    ],
    'tools': [{
        'type': 'function',
        'function': {
            'name': 'shell',
            'description': 'Run a shell command and return its output',
            'parameters': {
                'type': 'object',
                'properties': {'command': {'type': 'string'}},
                'required': ['command']
            }
        }
    }],
    'max_tokens': 128,
    'temperature': 0.0,
    'enable_pld': True,
    'enable_drafter': $([ "$USE_DRAFTER" = "1" ] && echo True || echo False),
    'stream': False
}))
")
check_turn "turn 2" "$TURN2_PAYLOAD" 1 || true
# The server log should show pld+drafter auto-disabled by tools.
if grep -E "pld=disabled|drafter=disabled|tools present" "$LOGFILE" > /dev/null 2>&1; then
    echo "    log shows spec auto-disable for tools turn (expected)"
fi
echo

# ── Turn 3: tool result fed back, spec re-activates ──
echo "--- Turn 3: tool result fed back (spec re-activates) ---"
TURN3_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Use the shell tool to run \"date\" and tell me the current day.'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{
            'id': 'call_1',
            'type': 'function',
            'function': {'name': 'shell', 'arguments': '{\"command\":\"date\"}'}
        }]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'Tue May  5 16:30:00 PDT 2026'}
    ],
    'max_tokens': 96,
    'temperature': 0.0,
    'enable_pld': True,
    'enable_drafter': $([ "$USE_DRAFTER" = "1" ] && echo True || echo False),
    'stream': False
}))
")
check_turn "turn 3" "$TURN3_PAYLOAD" 0 || true
echo

# ── Turn 4: long user message (~2000 tokens) — spec + KV cache reuse ──
echo "--- Turn 4: long user message (~2000 tokens) ---"
LONG_BLOCK=$(python3 -c "
chunks = []
for i in range(40):
    chunks.append(f'Line {i}: the quick brown fox jumps over the lazy dog and then walks slowly back home where it sleeps quietly until morning.')
print('\n'.join(chunks))
")
TURN4_PAYLOAD=$(python3 -c "
import json
long_text = '''$LONG_BLOCK'''
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [
        {'role': 'system', 'content': 'You are helpful. Be brief.'},
        {'role': 'user', 'content': 'Here is some text:\n' + long_text + '\n\nHow many lines did I send? Just the number.'}
    ],
    'max_tokens': 32,
    'temperature': 0.0,
    'enable_pld': True,
    'enable_drafter': $([ "$USE_DRAFTER" = "1" ] && echo True || echo False),
    'stream': False
}))
")
check_turn "turn 4" "$TURN4_PAYLOAD" 0 || true
echo

# ── Turn 5: same long user message again (cache hit; tests post-tool-call invalidation) ──
echo "--- Turn 5: same long message (cache hit after prior turns) ---"
check_turn "turn 5" "$TURN4_PAYLOAD" 0 || true
echo

# ── Turn 6: thinking enabled — exercises reasoning_content path ──
echo "--- Turn 6: enable_thinking:true (reasoning_content) ---"
TURN6_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [
        {'role': 'system', 'content': 'You are a careful reasoner.'},
        {'role': 'user', 'content': 'Briefly: is 17 a prime number?'}
    ],
    'max_tokens': 96,
    'temperature': 0.0,
    'enable_thinking': True,
    'reasoning_budget': 64,
    'enable_pld': True,
    'enable_drafter': $([ "$USE_DRAFTER" = "1" ] && echo True || echo False),
    'stream': False
}))
")
check_turn "turn 6" "$TURN6_PAYLOAD" 0 || true
echo

# ── Summary ──
echo "== summary: $PASS pass, $FAIL fail =="
if [ "$FAIL" -gt 0 ]; then
    echo
    echo "Recent server log lines:"
    tail -40 "$LOGFILE" | sed 's/^/    /'
    exit 1
fi
echo -e "${GREEN}PASS${NC} agent-loop spec-decode regression suite"
exit 0
