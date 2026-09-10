#!/bin/bash
# Test: KV cache reuse with sliding window models.
# Verifies that the server reuses the KV cache prefix across requests
# instead of doing a full reset when prompts exceed the sliding window.
# Usage: ./tests/test_kv_cache_sliding_window.sh [port]

PORT=${1:-8080}
BASE="http://127.0.0.1:$PORT"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
PASS=0
FAIL=0
TOTAL=0

assert_gt() {
    TOTAL=$((TOTAL + 1))
    local desc="$1" val="$2" threshold="$3"
    if [ "$val" -gt "$threshold" ] 2>/dev/null; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $desc ($val > $threshold)"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $desc ($val <= $threshold)"
    fi
}

assert_contains() {
    TOTAL=$((TOTAL + 1))
    local desc="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -q "$needle"; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $desc"
        echo -e "    expected: $needle"
        echo -e "    in: ${haystack:0:200}"
    fi
}

assert_not_contains() {
    TOTAL=$((TOTAL + 1))
    local desc="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -q "$needle"; then
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $desc"
        echo -e "    should NOT contain: $needle"
        echo -e "    in: ${haystack:0:200}"
    else
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $desc"
    fi
}

echo "=== KV Cache Sliding Window Tests ==="
echo "Server: $BASE"

if ! curl -sf "$BASE/health" > /dev/null 2>&1; then
    echo "SKIP: Server not running on port $PORT"
    exit 0
fi

# Check if model has sliding window
PROPS=$(curl -sf "$BASE/props")
SW=$(echo "$PROPS" | python3 -c "
import sys,json
d = json.load(sys.stdin)
mi = d.get('model_info',{})
# Gemma 4 models have sliding window
model = d.get('default_generation_settings',{}).get('model','')
if 'gemma4' in model or 'gemma-4' in model:
    print('yes')
else:
    print('no')
" 2>/dev/null)

if [ "$SW" != "yes" ]; then
    echo "SKIP: Model does not use sliding window attention"
    exit 0
fi

echo ""

# Build a shared prefix (large enough to exceed the 512-token sliding window)
# We use a big system prompt that stays constant across requests
SYSTEM_PROMPT="You are a helpful assistant. Always answer in exactly one short sentence. Do not elaborate."
# Pad the system prompt to make it substantial
PADDING=$(python3 -c "print('Context: ' + 'word ' * 500)")
FULL_SYSTEM="${SYSTEM_PROMPT} ${PADDING}"

# --- Test 1: First request (cold cache) ---
echo "--- Test 1: First request (cold cache, establishes prefix) ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"test\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$FULL_SYSTEM\"},
      {\"role\": \"user\", \"content\": \"What is 2+2?\"}
    ],
    \"max_tokens\": 32,
    \"temperature\": 0.0,
    \"stream\": false
  }" 2>&1)
TOKENS1=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])" 2>/dev/null)
echo "  prompt_tokens: $TOKENS1"
assert_gt "prompt > 512 (exceeds sliding window)" "$TOKENS1" 512

# --- Test 2: Second request with same prefix (should reuse cache) ---
echo ""
echo "--- Test 2: Second request, same prefix (should reuse KV cache) ---"
RESULT2=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"test\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$FULL_SYSTEM\"},
      {\"role\": \"user\", \"content\": \"What is 3+3?\"}
    ],
    \"max_tokens\": 32,
    \"temperature\": 0.0,
    \"stream\": false
  }" 2>&1)
TOKENS2=$(echo "$RESULT2" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])" 2>/dev/null)
echo "  prompt_tokens: $TOKENS2"

# --- Test 3: Check server logs for cache behavior ---
# We capture the last few log lines to check if cache was reused vs reset
echo ""
echo "--- Test 3: Verify cache reuse in server behavior ---"

# Send a third request and measure timing — if cache is reused, prefill should be faster
# (fewer tokens to encode)
START=$(python3 -c "import time; print(int(time.time()*1000))")
RESULT3=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"test\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$FULL_SYSTEM\"},
      {\"role\": \"user\", \"content\": \"What is 4+4?\"}
    ],
    \"max_tokens\": 32,
    \"temperature\": 0.0,
    \"stream\": false
  }" 2>&1)
END=$(python3 -c "import time; print(int(time.time()*1000))")
TIME3=$((END - START))
echo "  request time: ${TIME3}ms"

# Now send a DIFFERENT prefix to force a cache miss, then resend the original
echo ""
echo "--- Test 4: Different prefix (cache miss) then original prefix again ---"
curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [
      {"role": "system", "content": "You are a pirate."},
      {"role": "user", "content": "Say arrr."}
    ],
    "max_tokens": 16,
    "temperature": 0.0,
    "stream": false
  }' > /dev/null 2>&1

# Now resend original — this is the key test: is it a full cache miss?
START=$(python3 -c "import time; print(int(time.time()*1000))")
RESULT4=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"test\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$FULL_SYSTEM\"},
      {\"role\": \"user\", \"content\": \"What is 5+5?\"}
    ],
    \"max_tokens\": 32,
    \"temperature\": 0.0,
    \"stream\": false
  }" 2>&1)
END=$(python3 -c "import time; print(int(time.time()*1000))")
TIME4=$((END - START))
echo "  cache-miss request time: ${TIME4}ms"
echo "  cache-hit request time (test 3): ${TIME3}ms"

# The claim is REUSE and the server reports it as `cached_tokens`; wall clock
# cannot see a ~500-token prefill difference under 32 tokens of decode.
CACHED3=$(echo "$RESULT3" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0))" 2>/dev/null)
CACHED4=$(echo "$RESULT4" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0))" 2>/dev/null)
echo "  cached_tokens: warm=$CACHED3, after an intervening prefix=$CACHED4"
assert_gt "the warm request reuses the system prefix" "${CACHED3:-0}" 400
assert_gt "the prefix survives an intervening conversation" "${CACHED4:-0}" 400

echo ""
echo "--- Test 5: Multi-turn with tools (simulates Claude Code agent loop) ---"
# This is the most important test: Claude Code sends the same big system prompt
# with tools on every agent loop iteration. Cache should reuse the prefix.

TOOLS='[{"type":"function","function":{"name":"shell","description":"Execute a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]'

# First request with tools
START=$(python3 -c "import time; print(int(time.time()*1000))")
R1=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"test\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$FULL_SYSTEM\"},
      {\"role\": \"user\", \"content\": \"What is today's date?\"}
    ],
    \"tools\": $TOOLS,
    \"max_tokens\": 64,
    \"temperature\": 0.0,
    \"stream\": false
  }" 2>&1)
END=$(python3 -c "import time; print(int(time.time()*1000))")
TOOL_TIME1=$((END - START))
TOOL_TOKENS1=$(echo "$R1" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])" 2>/dev/null)
echo "  first tool request: ${TOOL_TIME1}ms, ${TOOL_TOKENS1} prompt tokens"

# Second request with tools (same prefix — should reuse cache)
START=$(python3 -c "import time; print(int(time.time()*1000))")
R2=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"test\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$FULL_SYSTEM\"},
      {\"role\": \"user\", \"content\": \"What is today's date?\"},
      {\"role\": \"assistant\", \"content\": null, \"tool_calls\": [{\"id\": \"call_1\", \"type\": \"function\", \"function\": {\"name\": \"shell\", \"arguments\": \"{\\\"command\\\": \\\"date\\\"}\"}}]},
      {\"role\": \"tool\", \"tool_call_id\": \"call_1\", \"content\": \"Mon Apr 7 2026\"},
      {\"role\": \"user\", \"content\": \"Thanks, what year is it?\"}
    ],
    \"tools\": $TOOLS,
    \"max_tokens\": 64,
    \"temperature\": 0.0,
    \"stream\": false
  }" 2>&1)
END=$(python3 -c "import time; print(int(time.time()*1000))")
TOOL_TIME2=$((END - START))
TOOL_TOKENS2=$(echo "$R2" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])" 2>/dev/null)
echo "  second tool request: ${TOOL_TIME2}ms, ${TOOL_TOKENS2} prompt tokens"
assert_gt "second request has more prompt tokens (conversation grew)" "$TOOL_TOKENS2" "$TOOL_TOKENS1"

echo ""
echo "=== Summary ==="
echo -e "Total: $TOTAL  ${GREEN}Passed: $PASS${NC}  ${RED}Failed: $FAIL${NC}"
if [ $FAIL -gt 0 ]; then
    exit 1
else
    echo "All KV cache sliding window tests passed."
    exit 0
fi
