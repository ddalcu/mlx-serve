#!/bin/bash
# Stress test: KV cache reuse + sliding window boundary
set -uo pipefail
PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0

pass() { PASS=$((PASS+1)); echo "✅ PASS: $1"; }
fail() { FAIL=$((FAIL+1)); echo "❌ FAIL: $1 — $2"; }

echo "=== Stress Test: Sliding Window + KV Cache Reuse ==="
echo ""

# Test 1: Short request then large request (triggers KV cache reuse across sliding window boundary)
echo "--- Test 1: Short then Large (KV cache reuse + window overflow) ---"
# First: short request to seed KV cache
RESP=$(curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Hi"}],"max_tokens":16,"stream":true}' 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then
    echo "  Short request OK"
else
    fail "short_seed" "failed"
fi

# Second: large system prompt (pushes past sliding window with existing cache)
LARGE=$(python3 -c "
import json
sys = 'You are a helpful assistant. ' * 200
msgs = [{'role':'system','content':sys},{'role':'user','content':'What is your name?'}]
print(json.dumps({'model':'mlx-serve','messages':msgs,'max_tokens':64,'stream':True}))
")
RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$LARGE" 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then
    pass "short_then_large"
else
    echo "  Response: $(echo "$RESP" | head -3)"
    fail "short_then_large" "crashed or no DONE"
fi

# Test 2: Multiple large requests back-to-back (KV cache reuse each time)
echo "--- Test 2: Back-to-back large requests ---"
for i in 1 2 3; do
    BODY=$(python3 -c "
import json
sys = 'Context line $i. ' * 100
msgs = [{'role':'system','content':sys},{'role':'user','content':'Reply with just OK.'}]
print(json.dumps({'model':'mlx-serve','messages':msgs,'max_tokens':32,'stream':True}))
")
    RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "$BODY" 2>&1)
    if echo "$RESP" | grep -q "data: \[DONE\]"; then
        echo "  Large request $i OK"
    else
        fail "back_to_back_large_$i" "failed"
        break
    fi
done
pass "back_to_back_large"

# Test 3: Long multi-turn conversation (accumulating history past window)
echo "--- Test 3: Long multi-turn conversation ---"
MSGS='[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi! How can I help?"}]'
for i in $(seq 1 8); do
    MSGS=$(python3 -c "
import json
msgs = json.loads('$MSGS')
msgs.append({'role':'user','content':'Tell me about topic $i in detail please. Give me a long answer.'})
print(json.dumps(msgs))
")
    BODY=$(python3 -c "
import json
msgs = json.loads('''$MSGS''')
print(json.dumps({'model':'mlx-serve','messages':msgs,'max_tokens':64,'stream':True}))
")
    RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "$BODY" 2>&1)
    if echo "$RESP" | grep -q "data: \[DONE\]"; then
        # Extract assistant response and add to history
        ASSISTANT=$(echo "$RESP" | grep "^data: " | grep -v "\[DONE\]" | sed 's/^data: //' | python3 -c "
import sys, json
content = ''
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        c = d.get('choices',[{}])[0].get('delta',{}).get('content','')
        content += c
    except: pass
print(json.dumps(content[:100]))
" 2>/dev/null)
        MSGS=$(python3 -c "
import json
msgs = json.loads('''$MSGS''')
msgs.append({'role':'assistant','content':json.loads('$ASSISTANT')})
print(json.dumps(msgs))
" 2>/dev/null)
        echo "  Turn $i OK ($(echo "$MSGS" | wc -c | tr -d ' ') bytes history)"
    else
        fail "long_multiturn_$i" "failed at turn $i"
        break
    fi
done
pass "long_multiturn"

# Test 4: Agent-like scenario (system prompt + multi-turn + tool results)
echo "--- Test 4: Agent mode simulation ---"
AGENT=$(python3 -c "
import json
system = '''You are an AI agent with tools: shell, read_file, write_file, web_browse, applescript.
When asked, create a plan. Format: <plan>...</plan>
''' + 'Additional context. ' * 100
msgs = [
    {'role':'system','content':system},
    {'role':'user','content':'List files in current directory'},
    {'role':'assistant','content':'I will use the shell tool to list files.\n<plan>\n1. shell: ls -la\n</plan>'},
    {'role':'user','content':'Tool results: file1.txt file2.py file3.md'},
    {'role':'assistant','content':'The directory contains 3 files: file1.txt, file2.py, and file3.md.'},
    {'role':'user','content':'Now read file1.txt and summarize it'},
]
print(json.dumps({'model':'mlx-serve','messages':msgs,'max_tokens':128,'stream':True}))
")
RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$AGENT" 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then
    pass "agent_simulation"
else
    fail "agent_simulation" "failed"
fi

# Test 5: Thinking mode with large context
echo "--- Test 5: Thinking with large context ---"
THINK=$(python3 -c "
import json
sys = 'You are a math tutor. ' * 100
msgs = [{'role':'system','content':sys},{'role':'user','content':'What is the integral of x^2?'}]
print(json.dumps({'model':'mlx-serve','messages':msgs,'max_tokens':128,'stream':True,'enable_thinking':True}))
")
RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$THINK" 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then
    pass "thinking_large_context"
else
    fail "thinking_large_context" "failed"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
exit $FAIL
