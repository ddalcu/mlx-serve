#!/bin/bash
# Integration tests for mlx-serve API endpoints.
# Usage: ./tests/integration_test.sh [model_dir] [port]
#
# Requires a model to be available. Defaults to qwen3-4b.
# Builds a debug binary, starts the server, runs tests, then kills it.

set -euo pipefail

MODEL_DIR="${1:-/Users/david/.cache/mlx-models/qwen3-4b}"
PORT="${2:-8095}"
BASE="http://localhost:$PORT"
BINARY="./zig-out/bin/mlx-serve"
PASS=0
FAIL=0
TOTAL=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

assert_eq() {
    TOTAL=$((TOTAL + 1))
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $desc"
        echo -e "    expected: $expected"
        echo -e "    actual:   $actual"
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
        echo -e "    expected to contain: $needle"
        echo -e "    actual: ${haystack:0:200}"
    fi
}

assert_not_contains() {
    TOTAL=$((TOTAL + 1))
    local desc="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -q "$needle"; then
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $desc"
        echo -e "    should NOT contain: $needle"
    else
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $desc"
    fi
}

assert_json_field() {
    TOTAL=$((TOTAL + 1))
    local desc="$1" field="$2" json="$3"
    local val
    val=$(echo "$json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(eval('d$field'))" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$val" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $desc (=$val)"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $desc"
        echo -e "    field $field not found in: ${json:0:200}"
    fi
}

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Build ──
echo -e "${YELLOW}Building...${NC}"
zig build 2>&1
echo ""

# ── Check model exists ──
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}Model not found: $MODEL_DIR${NC}"
    exit 1
fi

# ── Start server ──
echo -e "${YELLOW}Starting server on port $PORT...${NC}"
"$BINARY" --model "$MODEL_DIR" --serve --port "$PORT" --log-level warn --ctx-size 4096 &
SERVER_PID=$!

# Wait for health
for i in $(seq 1 30); do
    if curl -s "$BASE/health" 2>/dev/null | grep -q '"ok"'; then
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Server failed to start${NC}"
        exit 1
    fi
    sleep 1
done
echo -e "${GREEN}Server ready (PID=$SERVER_PID)${NC}"
echo ""

# ── Test: Health ──
echo -e "${YELLOW}Test: Health endpoint${NC}"
RESP=$(curl -s "$BASE/health")
assert_eq "GET /health returns ok" '{"status":"ok"}' "$RESP"
echo ""

# ── Test: Models ──
echo -e "${YELLOW}Test: Models endpoint${NC}"
RESP=$(curl -s "$BASE/v1/models")
assert_contains "GET /v1/models has data array" '"data"' "$RESP"
assert_contains "GET /v1/models has model id" '"id"' "$RESP"
assert_contains "GET /v1/models has context_length" '"context_length"' "$RESP"
echo ""

# ── Test: Props ──
echo -e "${YELLOW}Test: Props endpoint${NC}"
RESP=$(curl -s "$BASE/props")
assert_contains "GET /props has n_ctx" '"n_ctx"' "$RESP"
assert_contains "GET /props has total_slots" '"total_slots"' "$RESP"
echo ""

# ── Test: Chat completions (non-streaming) ──
echo -e "${YELLOW}Test: Chat completions (non-streaming)${NC}"
RESP=$(curl -s "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":10,"temperature":0}')
assert_contains "response has choices" '"choices"' "$RESP"
assert_contains "response has usage" '"usage"' "$RESP"
assert_contains "response has model" '"model"' "$RESP"
assert_contains "response has finish_reason" '"finish_reason"' "$RESP"
assert_json_field "has prompt_tokens" '["usage"]["prompt_tokens"]' "$RESP"
assert_json_field "has completion_tokens" '["usage"]["completion_tokens"]' "$RESP"
# Content should contain "4"
CONTENT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
assert_contains "answer contains 4" "4" "$CONTENT"
echo ""

# ── Test: Chat completions (streaming) ──
echo -e "${YELLOW}Test: Chat completions (streaming)${NC}"
RESP=$(curl -s "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":5,"stream":true,"temperature":0}')
assert_contains "streaming has data: prefix" "data: " "$RESP"
assert_contains "streaming has [DONE]" "[DONE]" "$RESP"
assert_contains "streaming has role delta" '"role":"assistant"' "$RESP"
echo ""

# ── Test: Chat with system message ──
echo -e "${YELLOW}Test: System message support${NC}"
RESP=$(curl -s "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"system","content":"You always respond in French"},{"role":"user","content":"Say hello"}],"max_tokens":20,"temperature":0}')
assert_contains "system message accepted" '"choices"' "$RESP"
echo ""

# ── Test: Temperature and sampling ──
echo -e "${YELLOW}Test: Sampling parameters${NC}"
RESP=$(curl -s "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Pick a number 1-10"}],"max_tokens":10,"temperature":0.8,"top_p":0.9,"top_k":40}')
assert_contains "sampling params accepted" '"choices"' "$RESP"
echo ""

# ── Test: Stop sequences ──
echo -e "${YELLOW}Test: Stop sequences${NC}"
RESP=$(curl -s "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Count from 1 to 20, one number per line"}],"max_tokens":200,"temperature":0,"stop":["5"]}')
FINISH=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['finish_reason'])" 2>/dev/null || echo "")
assert_eq "stop sequence triggers stop finish_reason" "stop" "$FINISH"
echo ""

# ── Test: Seed reproducibility ──
echo -e "${YELLOW}Test: Seed reproducibility${NC}"
RESP1=$(curl -s "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Pick a random word"}],"max_tokens":5,"temperature":0.5,"seed":42}')
RESP2=$(curl -s "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Pick a random word"}],"max_tokens":5,"temperature":0.5,"seed":42}')
C1=$(echo "$RESP1" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "a")
C2=$(echo "$RESP2" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "b")
assert_eq "same seed produces same output" "$C1" "$C2"
echo ""

# ── Test: Tool calling ──
echo -e "${YELLOW}Test: Tool calling${NC}"
RESP=$(curl -s "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "messages":[{"role":"user","content":"What is the weather in Tokyo?"}],
        "tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}],
        "max_tokens":200,"temperature":0
    }')
assert_contains "tool call has tool_calls" '"tool_calls"' "$RESP"
assert_contains "tool call has function name" '"get_weather"' "$RESP"
FINISH=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['finish_reason'])" 2>/dev/null || echo "")
assert_eq "tool call finish_reason is tool_calls" "tool_calls" "$FINISH"
echo ""

# ── Test: Text completions ──
echo -e "${YELLOW}Test: Text completions${NC}"
RESP=$(curl -s "$BASE/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"The capital of France is","max_tokens":10,"temperature":0}')
assert_contains "completions has choices" '"choices"' "$RESP"
assert_contains "completions has text field" '"text"' "$RESP"
echo ""

# ── Test: Tokenize / Detokenize ──
echo -e "${YELLOW}Test: Tokenize / Detokenize${NC}"
TOK_RESP=$(curl -s -X POST "$BASE/tokenize" \
    -H "Content-Type: application/json" \
    -d '{"content":"Hello, world!"}')
assert_contains "tokenize returns tokens" '"tokens"' "$TOK_RESP"

# Extract tokens and round-trip
TOKENS=$(echo "$TOK_RESP" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['tokens']))" 2>/dev/null || echo "[]")
DETOK_RESP=$(curl -s -X POST "$BASE/detokenize" \
    -H "Content-Type: application/json" \
    -d "{\"tokens\":$TOKENS}")
assert_contains "detokenize returns content" '"content"' "$DETOK_RESP"
DETOK_CONTENT=$(echo "$DETOK_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['content'])" 2>/dev/null || echo "")
assert_eq "round-trip tokenize/detokenize" "Hello, world!" "$DETOK_CONTENT"
echo ""

# ── Test: Error responses ──
echo -e "${YELLOW}Test: Error responses${NC}"
# Invalid JSON
RESP=$(curl -s "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d 'not json')
assert_contains "invalid JSON returns error" '"error"' "$RESP"

# Missing messages
RESP=$(curl -s "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"max_tokens":10}')
assert_contains "missing messages returns error" '"error"' "$RESP"

# 404
RESP=$(curl -s "$BASE/v1/nonexistent")
assert_contains "unknown endpoint returns 404 error" '"error"' "$RESP"
echo ""

# ── Test: stream_options include_usage ──
echo -e "${YELLOW}Test: Streaming with include_usage${NC}"
RESP=$(curl -s "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":3,"stream":true,"stream_options":{"include_usage":true},"temperature":0}')
assert_contains "streaming usage has prompt_tokens" '"prompt_tokens"' "$RESP"
assert_contains "streaming usage has completion_tokens" '"completion_tokens"' "$RESP"
echo ""

# ── Test: Logprobs ──
echo -e "${YELLOW}Test: Logprobs${NC}"
RESP=$(curl -s "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":3,"logprobs":true,"top_logprobs":3,"temperature":0}')
assert_contains "logprobs has content" '"logprobs"' "$RESP"
assert_contains "logprobs has top_logprobs" '"top_logprobs"' "$RESP"
echo ""

# ── Test: CORS ──
echo -e "${YELLOW}Test: CORS preflight${NC}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X OPTIONS "$BASE/v1/chat/completions")
assert_eq "OPTIONS returns 204" "204" "$HTTP_CODE"
echo ""

# ── Summary ──
echo ""
echo -e "═══════════════════════════════════"
if [ $FAIL -eq 0 ]; then
    echo -e "  ${GREEN}ALL $TOTAL TESTS PASSED${NC}"
else
    echo -e "  ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC} (out of $TOTAL)"
fi
echo -e "═══════════════════════════════════"

exit $FAIL
