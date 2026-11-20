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
source "$(dirname "$0")/lib/server.sh"

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

srv_preclean "$PORT"
sleep 1
"$BINARY" --model "$MODEL" --serve --port "$PORT" --ctx-size 8192 --log-level info > /tmp/test_thinking_split.log 2>&1 &
SERVER_PID=$!
trap 'srv_stop $SERVER_PID' EXIT

srv_wait_health "$BASE" 90 2 || { echo "FAIL: server did not come up"; exit 1; }

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

echo "8. usage.completion_tokens_details.reasoning_tokens: non-stream and stream (include_usage), plain and with tools"
# Reasoning in the reply means a count above 0 and within completion_tokens, none means no completion_tokens_details.
# The tools case runs the stream's tool-gated arm and, when the model calls, the tool-call reply; plain runs the think arm.
WEATHER_TOOL='[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
for CASE in "true|plain|What is 17+25? Answer with just the number." \
            "true|tools|What is the weather in Paris right now? Use the get_weather tool." \
            "false|plain|What is 17+25? Answer with just the number."; do
    IFS='|' read -r THINK ARM Q <<< "$CASE"
    EXTRA=""
    [ "$ARM" = tools ] && EXTRA=",\"tools\":$WEATHER_TOOL"
    for STREAM in false true; do
        [ "$STREAM" = true ] && SEXTRA=",\"stream\":true,\"stream_options\":{\"include_usage\":true}" || SEXTRA=""
        read -r RLEN CALLED RTOK CTOK <<< "$(curl -sN -m 180 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
            -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":800,\"temperature\":0,\"enable_thinking\":$THINK$EXTRA$SEXTRA}" | python3 -c "
import json,sys
raw=sys.stdin.read(); r=''; called=False; u={}
if raw.lstrip().startswith('data:'):
    for line in raw.splitlines():
        p=line[5:].strip() if line.startswith('data:') else ''
        if not p or p=='[DONE]': continue
        o=json.loads(p); u=o.get('usage') or u
        d=(o.get('choices') or [{}])[0].get('delta',{})
        r+=d.get('reasoning_content') or ''; called=called or bool(d.get('tool_calls'))
else:
    o=json.loads(raw); m=o['choices'][0]['message']; u=o['usage']
    r=m.get('reasoning_content') or ''; called=bool(m.get('tool_calls'))
print(len(r), called, (u.get('completion_tokens_details') or {}).get('reasoning_tokens','missing'), u.get('completion_tokens','missing'))")"
        DESC="stream=$STREAM $ARM thinking=$THINK called=$CALLED"
        echo "  $DESC: reasoning ${RLEN}B, reasoning_tokens=$RTOK completion_tokens=$CTOK"
        if [ "$RLEN" -gt 0 ] 2>/dev/null; then
            check "$DESC: reasoning_tokens above 0" "$([ "$RTOK" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
            check "$DESC: reasoning_tokens within completion_tokens" "$([ "$RTOK" -le "$CTOK" ] 2>/dev/null && echo 1 || echo 0)"
        else
            check "$DESC: no reasoning_tokens without reasoning" "$([ "$RTOK" = missing ] && echo 1 || echo 0)"
        fi
    done
done

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
