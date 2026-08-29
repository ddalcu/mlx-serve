#!/bin/bash
# Long-conversation sliding-window regression for spec-decode.
#
# Build a single conversation that grows past the model's sliding window
# threshold (Gemma 4 E4B = 512 tokens). Send N user messages in sequence
# against ONE long-running server, each with spec-decode enabled. After each
# turn, assert the response decoded correctly:
#   - HTTP 200
#   - parseable JSON
#   - finish_reason in {stop, length}
#   - non-empty content
#   - no NaN tokens, no <pad>-only output
#
# The point of the test is to catch:
#   - KV cache view bugs during multi-token verify forwards at offsets past
#     the sliding window
#   - Drafter cross-attn into target's K/V at offsets > sliding_window
#   - SSM/conv state corruption across long-context decodes (hybrid models)
#
# Multi-turn growth is built by appending each assistant response back into
# the next request — the conversation gets monotonically longer as we go.
#
# Defaults:
#   model = ~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit
#   N = 12 turns of ~50-token unique content (more than enough to cross 512)
#
# Usage:
#   LONG_CONV_TEST_MODEL=/path/to/model ./tests/test_long_conversation_spec.sh [port]

set -e

PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
N_TURNS="${LONG_CONV_TURNS:-12}"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${LONG_CONV_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}"
DRAFTER="${LONG_CONV_TEST_DRAFTER:-$HOME/.mlx-serve/models/mlx-community/gemma-4-E4B-it-assistant-bf16}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_long_conversation_spec: model directory not found ($MODEL)."
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

echo "== long-conversation spec-decode regression =="
echo "  model:   $MODEL  (model_type=$TARGET_TYPE)"
echo "  drafter: $([ "$USE_DRAFTER" = "1" ] && echo "$DRAFTER" || echo "(none)")"
echo "  turns:   $N_TURNS"
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
    tail -30 "$LOGFILE"
    exit 1
fi
echo "  server up (PID=$SERVER_PID)"
echo

# Conversation history (JSON-encodable list). We build it incrementally in a
# Python child process for each turn so we can persist between turns via the
# transcript file.
TRANSCRIPT=$(mktemp)
echo '[]' > "$TRANSCRIPT"

PASS=0
FAIL=0

for turn in $(seq 1 "$N_TURNS"); do
    # Each user message is unique enough to defeat any echo-friendly path.
    USER_MSG="Turn $turn question: please rate (out of 10) the following descriptor: '$(python3 -c "
import random, string
random.seed($turn)
words = []
for _ in range(8):
    w = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4,9)))
    words.append(w)
print(' '.join(words))
")'. Reply with one short sentence."

    PAYLOAD=$(python3 -c "
import json, sys
hist = json.load(open('$TRANSCRIPT'))
hist.append({'role': 'user', 'content': '''$USER_MSG'''})
out = {
    'model': 'mlx-serve',
    'messages': hist,
    'max_tokens': 64,
    'temperature': 0.0,
    'enable_pld': True,
    'enable_drafter': $([ "$USE_DRAFTER" = "1" ] && echo True || echo False),
    'stream': False
}
print(json.dumps(out))
")

    tmp=$(mktemp)
    http_code=$(curl -s -o "$tmp" -w "%{http_code}" -X POST "$BASE/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$PAYLOAD")
    raw=$(cat "$tmp"); rm -f "$tmp"

    if [ "$http_code" != "200" ]; then
        echo -e "  ${RED}FAIL${NC} turn $turn: HTTP $http_code"
        echo "    body: ${raw:0:300}"
        FAIL=$((FAIL + 1))
        continue
    fi

    parsed=$(echo "$raw" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
except Exception as e:
    print('PARSE_FAIL'); sys.exit(0)
ch = (d.get('choices') or [{}])[0]
msg = ch.get('message') or {}
content = msg.get('content') or ''
fr = ch.get('finish_reason') or '?'
prompt_t = (d.get('usage') or {}).get('prompt_tokens', 0)
comp_t = (d.get('usage') or {}).get('completion_tokens', 0)
# Strip pad tokens for emptiness check
cleaned = content.replace('<pad>', '').strip()
print(f'OK|{fr}|{prompt_t}|{comp_t}|{len(content)}|{len(cleaned)}|{content[:80]}'.replace('\n',' '))
")
    if [ "$parsed" = "PARSE_FAIL" ]; then
        echo -e "  ${RED}FAIL${NC} turn $turn: invalid JSON"
        echo "    body: ${raw:0:300}"
        FAIL=$((FAIL + 1))
        continue
    fi
    IFS='|' read -r status fr prompt_t comp_t clen cleaned_len preview <<< "$parsed"
    case "$fr" in
        stop|length) ;;
        *)
            echo -e "  ${RED}FAIL${NC} turn $turn: bad finish_reason=$fr"
            FAIL=$((FAIL + 1))
            continue
            ;;
    esac
    if [ "$cleaned_len" -lt 1 ]; then
        echo -e "  ${RED}FAIL${NC} turn $turn: pad-only or empty content (raw len=$clen, cleaned len=$cleaned_len)"
        FAIL=$((FAIL + 1))
        continue
    fi
    # NaN-token symptom: long runs of identical char (often '!' or 0xFFFD)
    NAN_HIT=$(python3 -c "
import re
c = '''$preview'''
m = re.search(r'(.)\1{20,}', c)
print('1' if m else '0')
")
    if [ "$NAN_HIT" = "1" ]; then
        echo -e "  ${RED}FAIL${NC} turn $turn: 20+ char repeating run (likely NaN tokens): $preview"
        FAIL=$((FAIL + 1))
        continue
    fi
    echo -e "  ${GREEN}PASS${NC} turn $turn finish=$fr prompt_t=$prompt_t comp_t=$comp_t"
    PASS=$((PASS + 1))

    # Append the assistant message back into the transcript.
    ASST_CONTENT=$(echo "$raw" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
    python3 -c "
import json
hist = json.load(open('$TRANSCRIPT'))
hist.append({'role': 'user', 'content': '''$USER_MSG'''})
hist.append({'role': 'assistant', 'content': '''$ASST_CONTENT'''})
json.dump(hist, open('$TRANSCRIPT','w'))
"
done

rm -f "$TRANSCRIPT"

echo
echo "== summary: $PASS pass / $FAIL fail (out of $N_TURNS) =="
if [ "$FAIL" -gt 0 ]; then
    echo
    echo "Recent server log lines:"
    tail -40 "$LOGFILE" | sed 's/^/    /'
    exit 1
fi
echo -e "${GREEN}PASS${NC} long-conversation spec-decode regression"
exit 0
