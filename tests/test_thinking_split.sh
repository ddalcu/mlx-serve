#!/bin/bash
# Regression tests for server-side reasoning/content separation (TODO #13).
#
# Pins the Qwen 3.6 truncated-thinking leak: the chat template injects the
# `<think>\n` opener into the generation prompt, so when generation stops
# (length) before `</think>`, the model's output contains NO think tags at all.
# Pre-fix, the non-streaming path dumped that truncated reasoning into
# `choices[].message.content`. Post-fix it must land in `reasoning_content`
# (with content empty) on every surface, and usage must report
# `completion_tokens_details.reasoning_tokens` / `output_tokens_details`.
#
# Usage: ./tests/test_thinking_split.sh [model_dir] [port]
#   model_dir must be a model whose chat template injects a think opener when
#   thinking is on (Qwen 3.5/3.6). Default: Qwen3.6-35B-A3B-4bit (LM Studio dir).
#
# Starts its own server on [port].

set -u

MODEL="${1:-$HOME/.lmstudio/models/mlx-community/Qwen3.6-35B-A3B-4bit}"
PORT="${2:-11261}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}

if [ ! -d "$MODEL" ]; then
    echo "SKIP: model dir not found: $MODEL"
    exit 0
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 1
"$BINARY" --model "$MODEL" --serve --port "$PORT" --ctx-size 8192 --log-level info > /tmp/test_thinking_split.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for _ in $(seq 1 90); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    sleep 2
done
curl -sf "$BASE/health" >/dev/null 2>&1 || { echo "FAIL: server did not come up"; exit 1; }

PROMPT='{"role":"user","content":"What is 17*23? Be brief."}'

echo "1. chat/completions non-stream, truncated thinking (max_tokens=120)"
R=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":120,\"temperature\":0,\"enable_thinking\":true}")
CONTENT=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message'].get('content') or '')")
REASONING=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message'].get('reasoning_content') or '')")
RTOK=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin)['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0))")
check "truncated thinking is NOT in content" "$([ -z "$CONTENT" ] && echo 1 || echo 0)"
check "truncated thinking IS in reasoning_content" "$([ -n "$REASONING" ] && echo 1 || echo 0)"
check "usage reports reasoning_tokens > 0" "$([ "$RTOK" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"

echo "2. chat/completions non-stream, full thinking round (max_tokens=2000)"
R=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":2000,\"temperature\":0,\"enable_thinking\":true}")
CONTENT=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message'].get('content') or '')")
REASONING=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message'].get('reasoning_content') or '')")
FINISH=$(echo "$R" | python3 -c "import json,sys; c=json.load(sys.stdin)['choices'][0]; print((c.get('finish_details') or {}).get('type') or c['finish_reason'])")
# Greedy thinking can loop on some quants until the cap or the loop guard ends it: a
# reply that ended inside the thought has no answer to place, so only the split is checked.
if { [ "$FINISH" = "length" ] || [ "$FINISH" = "repetition_loop" ]; } && [ -z "$CONTENT" ]; then
    echo "  skip answer (391) is in content — ended inside the thought ($FINISH)"
else
    check "answer (391) is in content" "$(echo "$CONTENT" | grep -q 391 && echo 1 || echo 0)"
fi
check "content has no think tags" "$(echo "$CONTENT" | grep -q '</think>\|<think>' && echo 0 || echo 1)"
check "reasoning_content non-empty" "$([ -n "$REASONING" ] && echo 1 || echo 0)"

echo "3. chat/completions thinking OFF stays clean"
R=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":100,\"temperature\":0}")
CONTENT=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message'].get('content') or '')")
check "thinking-off content is the answer" "$(echo "$CONTENT" | grep -q 391 && echo 1 || echo 0)"
check "thinking-off content has no tags/reasoning" "$(echo "$CONTENT" | grep -qi 'think' && echo 0 || echo 1)"

echo "4. /v1/messages truncated thinking lands in thinking block"
R=$(curl -s "$BASE/v1/messages" -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"max_tokens\":120,\"messages\":[$PROMPT],\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":4000}}")
TEXTBLK=$(echo "$R" | python3 -c "
import json,sys; r=json.load(sys.stdin)
print(''.join(b.get('text','') for b in r.get('content',[]) if b['type']=='text'))")
THINKBLK=$(echo "$R" | python3 -c "
import json,sys; r=json.load(sys.stdin)
print(''.join(b.get('thinking','') for b in r.get('content',[]) if b['type']=='thinking'))")
check "anthropic: truncated thinking not in text block" "$([ -z "$TEXTBLK" ] && echo 1 || echo 0)"
check "anthropic: thinking block non-empty" "$([ -n "$THINKBLK" ] && echo 1 || echo 0)"

echo "5. /v1/responses truncated thinking → reasoning item + usage detail"
R=$(curl -s "$BASE/v1/responses" -H 'Content-Type: application/json' \
    -d '{"model":"x","input":"What is 17*23? Be brief.","max_output_tokens":120,"reasoning":{"effort":"medium"}}')
MSGTXT=$(echo "$R" | python3 -c "
import json,sys; r=json.load(sys.stdin)
out=''
for it in r.get('output',[]):
    if it.get('type')=='message':
        out+=''.join(c.get('text','') for c in it.get('content',[]))
print(out)")
RTOK=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin).get('usage',{}).get('output_tokens_details',{}).get('reasoning_tokens',0))")
check "responses: truncated thinking not in message text" "$([ -z "$MSGTXT" ] && echo 1 || echo 0)"
check "responses: usage reasoning_tokens > 0" "$([ "$RTOK" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"

echo "6. stream + thinking: deltas split, no leak"
R=$(curl -sN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":120,\"temperature\":0,\"enable_thinking\":true,\"stream\":true}" | python3 -c "
import json,sys
c=r=''
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('data:'): continue
    p=line[5:].strip()
    if p=='[DONE]': break
    try: o=json.loads(p)
    except: continue
    d=(o.get('choices') or [{}])[0].get('delta',{})
    c+=d.get('content') or ''
    r+=d.get('reasoning_content') or ''
print(f'{len(c)}|{len(r)}')")
CLEN="${R%%|*}"; RLEN="${R##*|}"
check "stream: truncated thinking went to reasoning deltas" "$([ "$RLEN" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
check "stream: no content deltas during truncated think" "$([ "$CLEN" = "0" ] && echo 1 || echo 0)"

echo "7. stream + thinking + TOOLS present: answer arrives as content (pi regression)"
# With tools, the stream takes the tool-buffer path. Pre-fix, after the think
# block closed mid-stream the handler kept treating buffered text as
# "incomplete thinking" and flushed the visible answer as reasoning_content at
# end-of-stream — agent clients (pi) showed an empty final answer.
R=$(curl -sN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":2000,\"temperature\":0,\"enable_thinking\":true,\"stream\":true,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"description\":\"Run a shell command\",\"parameters\":{\"type\":\"object\",\"properties\":{\"cmd\":{\"type\":\"string\"}},\"required\":[\"cmd\"]}}}]}" | python3 -c "
import json,sys
c=r=''
calls=False
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('data:'): continue
    p=line[5:].strip()
    if p=='[DONE]': break
    try: o=json.loads(p)
    except: continue
    d=(o.get('choices') or [{}])[0].get('delta',{})
    c+=d.get('content') or ''
    r+=d.get('reasoning_content') or ''
    calls=calls or bool(d.get('tool_calls'))
print(json.dumps({'content':c,'reasoning_len':len(r),'called':calls}))")
CONTENT=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin)['content'])")
RLEN=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin)['reasoning_len'])")
CALLED=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin)['called'])")
check "tools+thinking: reasoning still separated" "$([ "$RLEN" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
# The model may answer or reach for the tool (Qwen3.5-4B runs `echo $((17*23))`): only an answer must land in content.
if [ "$CALLED" = True ]; then
    echo "  skip tools+thinking: answer in content — the model called the tool instead"
else
    check "tools+thinking: answer (391) in content deltas" "$(echo "$CONTENT" | grep -q 391 && echo 1 || echo 0)"
fi

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
