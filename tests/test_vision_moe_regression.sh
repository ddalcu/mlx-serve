#!/bin/bash
# Regression tests for bugfix/vision_moe_gemma merge.
# Exercises: optional field init, BitsCache, vision pipeline, MoE routing.
# Self-contained: builds binary, starts server, runs tests, kills server.
#
# Usage: ./tests/test_vision_moe_regression.sh [model_dir] [port]
# Default model: ~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit
# Default port: 8097

set -euo pipefail

MODEL_DIR="${1:-${MLX_SERVE_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}}"
PORT="${2:-8097}"
BASE="http://127.0.0.1:$PORT"
BINARY="./zig-out/bin/mlx-serve"
PASS=0
FAIL=0
SKIP=0
TOTAL=0
LOG="/tmp/mlx-serve-regression-$$.log"
FIXTURES="$(dirname "$0")/fixtures"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
DIM='\033[2m'
NC='\033[0m'

# ── Helpers ──

run_test() {
    local name="$1" result="$2" detail="${3:-}"
    TOTAL=$((TOTAL + 1))
    if [ "$result" = "PASS" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $name"
    elif [ "$result" = "SKIP" ]; then
        SKIP=$((SKIP + 1))
        echo -e "  ${YELLOW}SKIP${NC} $name — $detail"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $name"
        [ -n "$detail" ] && echo -e "    ${DIM}$detail${NC}"
    fi
}

assert_not_empty() {
    local desc="$1" value="$2"
    run_test "$desc" "$([ -n "$value" ] && echo PASS || echo FAIL)" "got empty"
}

assert_contains() {
    local desc="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -q "$needle"; then
        run_test "$desc" "PASS"
    else
        run_test "$desc" "FAIL" "expected to contain '$needle', got: ${haystack:0:200}"
    fi
}

assert_server_alive() {
    local desc="$1"
    if curl -sf "$BASE/health" | grep -q ok; then
        run_test "Server alive after: $desc" "PASS"
    else
        run_test "Server alive after: $desc" "FAIL" "server crashed — last 10 lines of stderr:"
        tail -10 "$LOG" 2>/dev/null | while IFS= read -r line; do
            echo -e "    ${DIM}$line${NC}"
        done
    fi
}

check_no_crash() {
    local desc="$1"
    if grep -qiE 'segfault|SIGSEGV|panic|illegal|bus error|abort|unreachable' "$LOG" 2>/dev/null; then
        run_test "No crash signals: $desc" "FAIL" "crash pattern found in stderr"
        grep -iE 'segfault|SIGSEGV|panic|illegal|bus error|abort|unreachable' "$LOG" | head -5 | while IFS= read -r line; do
            echo -e "    ${DIM}$line${NC}"
        done
    else
        run_test "No crash signals: $desc" "PASS"
    fi
}

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Detect model capabilities from config.json ──

echo "=== Vision/MoE Regression Tests ==="
echo "Model: $MODEL_DIR"
echo "Port:  $PORT"
echo ""

if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${YELLOW}SKIP${NC}: no model at $MODEL_DIR (pass a dir as \$1 or set MLX_SERVE_TEST_MODEL)"
    exit 0
fi

CONFIG_FILE="$MODEL_DIR/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}config.json not found in model dir${NC}"
    exit 1
fi

MODEL_TYPE=$(python3 -c "import json; d=json.load(open('$CONFIG_FILE')); print(d.get('model_type','unknown'))" 2>/dev/null || echo "unknown")
NUM_EXPERTS=$(python3 -c "import json; d=json.load(open('$CONFIG_FILE')); print(d.get('num_local_experts', d.get('num_experts', 0)))" 2>/dev/null || echo "0")
HAS_VISION=$(python3 -c "import json; d=json.load(open('$CONFIG_FILE')); print('yes' if d.get('vision_config') or d.get('vision_tower') else 'no')" 2>/dev/null || echo "no")

echo "Model type:  $MODEL_TYPE"
echo "Experts:     $NUM_EXPERTS"
echo "Vision:      $HAS_VISION"
echo ""

# ── Build ──

echo -e "${YELLOW}Building...${NC}"
{ [ -x ./.zig-toolchain/zig ] && ZIG=./.zig-toolchain/zig || ZIG=zig; "$ZIG" build -Doptimize=ReleaseFast 2>&1; }
echo ""

# ── Start server ──

echo -e "${YELLOW}Starting server on port $PORT...${NC}"
"$BINARY" --model "$MODEL_DIR" --serve --port "$PORT" --log-level warn --ctx-size 4096 > "$LOG" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 30); do
    if curl -sf "$BASE/health" 2>/dev/null | grep -q ok; then
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Server failed to start within 60s${NC}"
        tail -20 "$LOG" 2>/dev/null
        exit 1
    fi
    sleep 2
done
echo -e "${GREEN}Server ready (PID=$SERVER_PID)${NC}"
echo ""

TOOLS_JSON='[{"type":"function","function":{"name":"get_weather","description":"Get current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"}},"required":["location"]}}}]'

# ═══════════════════════════════════════════════════════
# Section 1: Basic Forward Pass (ALL models)
# Exercises: optional field init, BitsCache, transformer forward path
# ═══════════════════════════════════════════════════════
echo -e "${YELLOW}Section 1: Basic Forward Pass${NC}"
echo ""

# 1a: Non-streaming chat completion
echo -e "${DIM}1a: Non-streaming chat completion${NC}"
RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":20,"temperature":0,"stream":false}')
CONTENT=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo "")
assert_not_empty "Non-streaming returns content" "$CONTENT"
assert_contains "Response has usage" '"usage"' "$RESP"
assert_server_alive "non-streaming chat"

# 1b: Streaming chat completion
echo ""
echo -e "${DIM}1b: Streaming chat completion${NC}"
STREAM=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20,"temperature":0,"stream":true}')
HAS_DATA=$(echo "$STREAM" | grep -c 'data:' || true)
HAS_DONE=$(echo "$STREAM" | grep -c '\[DONE\]' || true)
run_test "Streaming has data events" "$([ "$HAS_DATA" -gt 0 ] && echo PASS || echo FAIL)" "$HAS_DATA events"
run_test "Streaming has [DONE]" "$([ "$HAS_DONE" -gt 0 ] && echo PASS || echo FAIL)"
assert_server_alive "streaming chat"

# 1c: Chat with tools
echo ""
echo -e "${DIM}1c: Chat with tools${NC}"
RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Tokyo?\"}],\"tools\":$TOOLS_JSON,\"max_tokens\":200,\"temperature\":0,\"stream\":false}")
assert_contains "Tool response has choices" '"choices"' "$RESP"
# Model may or may not call the tool — either is fine for this regression test
assert_server_alive "chat with tools"

# 1d: 5 sequential requests (KV cache reuse, BitsCache stability)
echo ""
echo -e "${DIM}1d: Sequential requests (KV cache + BitsCache stability)${NC}"
for i in 1 2 3 4 5; do
    RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
      -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"What is $i + $i? Just the number.\"}],\"max_tokens\":10,\"temperature\":0,\"stream\":false}")
    CONTENT=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo "")
    if [ -z "$CONTENT" ]; then
        run_test "Sequential request $i returns content" "FAIL" "empty response"
        break
    fi
done
run_test "All 5 sequential requests returned content" "$([ -n "$CONTENT" ] && echo PASS || echo FAIL)"
assert_server_alive "5 sequential requests"

# 1e: Long prompt (~200 repeated phrases)
echo ""
echo -e "${DIM}1e: Long prompt (context handling)${NC}"
LONG_PROMPT=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 200 + 'Now say OK.')")
RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG_PROMPT\"}],\"max_tokens\":10,\"temperature\":0,\"stream\":false}" 2>/dev/null || echo '{"error":"timeout or crash"}')
assert_contains "Long prompt returns choices or error gracefully" '"choices"\|"error"' "$RESP"
assert_server_alive "long prompt"

# 1f: Multi-turn conversation (4 messages)
echo ""
echo -e "${DIM}1f: Multi-turn conversation${NC}"
RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{
    "model":"mlx-serve",
    "messages":[
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"My name is Alice."},
        {"role":"assistant","content":"Hello Alice!"},
        {"role":"user","content":"What is my name? One word."}
    ],
    "max_tokens":20,"temperature":0,"stream":false}')
CONTENT=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo "")
assert_not_empty "Multi-turn returns content" "$CONTENT"
assert_server_alive "multi-turn conversation"

# ═══════════════════════════════════════════════════════
# Section 2: Vision Pipeline (if model has vision)
# Exercises: embedding scaling, clipped linears, RMS norm ordering, patch padding/masking
# ═══════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}Section 2: Vision Pipeline${NC}"

if [ "$HAS_VISION" = "yes" ]; then
    if [ ! -f "$FIXTURES/house.jpeg" ]; then
        run_test "Vision fixture exists" "SKIP" "tests/fixtures/house.jpeg not found"
    else
        # 2a: JPEG image request
        echo ""
        echo -e "${DIM}2a: JPEG image request${NC}"
        VISION_RESP=$(python3 -c "
import base64, json, urllib.request
with open('$FIXTURES/house.jpeg', 'rb') as f: img = f.read()
b64 = base64.b64encode(img).decode()
msg = {'model':'mlx-serve','max_tokens':30,'temperature':0.0,'stream':False,
       'messages':[{'role':'user','content':[
           {'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{b64}'}},
           {'type':'text','text':'What is in this image? One sentence.'}]}]}
req = urllib.request.Request('$BASE/v1/chat/completions', json.dumps(msg).encode(), {'Content-Type':'application/json'})
resp = urllib.request.urlopen(req, timeout=180)
print(json.loads(resp.read())['choices'][0]['message']['content'].strip())
" 2>/dev/null || echo "")
        assert_not_empty "Vision JPEG returns content" "$VISION_RESP"
        assert_server_alive "vision JPEG"

        # 2b: Streaming + image
        echo ""
        echo -e "${DIM}2b: Streaming + image${NC}"
        VISION_STREAM=$(python3 -c "
import base64, json, urllib.request
with open('$FIXTURES/house.jpeg', 'rb') as f: img = f.read()
b64 = base64.b64encode(img).decode()
msg = {'model':'mlx-serve','max_tokens':30,'temperature':0.0,'stream':True,
       'messages':[{'role':'user','content':[
           {'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{b64}'}},
           {'type':'text','text':'Describe this briefly.'}]}]}
req = urllib.request.Request('$BASE/v1/chat/completions', json.dumps(msg).encode(), {'Content-Type':'application/json'})
resp = urllib.request.urlopen(req, timeout=180)
print(resp.read().decode())
" 2>/dev/null || echo "")
        HAS_STREAM_DATA=$(echo "$VISION_STREAM" | grep -c 'data:' || true)
        run_test "Vision streaming has data events" "$([ "$HAS_STREAM_DATA" -gt 0 ] && echo PASS || echo FAIL)" "$HAS_STREAM_DATA events"
        assert_server_alive "vision streaming"

        # 2c: Text-only after vision request (no stale embeddings)
        echo ""
        echo -e "${DIM}2c: Text-only after vision request${NC}"
        RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
          -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 3+3? Just the number."}],"max_tokens":10,"temperature":0,"stream":false}')
        CONTENT=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo "")
        assert_not_empty "Text-only after vision returns content" "$CONTENT"
        assert_server_alive "text after vision"
    fi
else
    run_test "Vision tests" "SKIP" "model has no vision config"
fi

# ═══════════════════════════════════════════════════════
# Section 3: --no-vision restart (if vision model)
# Exercises: vision config defaults harmless when encoder disabled
# ═══════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}Section 3: --no-vision restart${NC}"

if [ "$HAS_VISION" = "yes" ]; then
    echo -e "${DIM}Stopping server for --no-vision restart...${NC}"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true

    NO_VISION_LOG="/tmp/mlx-serve-regression-novision-$$.log"
    "$BINARY" --model "$MODEL_DIR" --serve --port "$PORT" --log-level warn --ctx-size 4096 --no-vision > "$NO_VISION_LOG" 2>&1 &
    SERVER_PID=$!

    STARTED=false
    for i in $(seq 1 30); do
        if curl -sf "$BASE/health" 2>/dev/null | grep -q ok; then
            STARTED=true
            break
        fi
        sleep 2
    done

    if [ "$STARTED" = "true" ]; then
        echo -e "${GREEN}Server restarted with --no-vision${NC}"

        RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
          -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 5+5? Just the number."}],"max_tokens":10,"temperature":0,"stream":false}')
        CONTENT=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo "")
        assert_not_empty "Text completion with --no-vision" "$CONTENT"
        assert_server_alive "--no-vision text completion"

        check_no_crash "--no-vision server"

        # Restore normal server for remaining tests
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true

        # Append no-vision log to main log
        cat "$NO_VISION_LOG" >> "$LOG" 2>/dev/null || true
        rm -f "$NO_VISION_LOG"

        "$BINARY" --model "$MODEL_DIR" --serve --port "$PORT" --log-level warn --ctx-size 4096 > "$LOG" 2>&1 &
        SERVER_PID=$!

        for i in $(seq 1 30); do
            if curl -sf "$BASE/health" 2>/dev/null | grep -q ok; then break; fi
            sleep 2
        done
        echo -e "${GREEN}Server restored for remaining tests${NC}"
    else
        run_test "--no-vision server start" "FAIL" "server did not start within 60s"
        tail -20 "$NO_VISION_LOG" 2>/dev/null
        rm -f "$NO_VISION_LOG"

        # Restore normal server
        "$BINARY" --model "$MODEL_DIR" --serve --port "$PORT" --log-level warn --ctx-size 4096 > "$LOG" 2>&1 &
        SERVER_PID=$!
        for i in $(seq 1 30); do
            if curl -sf "$BASE/health" 2>/dev/null | grep -q ok; then break; fi
            sleep 2
        done
    fi
else
    run_test "--no-vision tests" "SKIP" "model has no vision config"
fi

# ═══════════════════════════════════════════════════════
# Section 4: MoE-specific (if num_experts > 0)
# Exercises: MoE forward pass, expert routing, BitsCache with MoE
# ═══════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}Section 4: MoE-specific${NC}"

if [ "$NUM_EXPERTS" -gt 0 ] 2>/dev/null; then
    # 4a: Chat with thinking (exercises forwardMoe with reasoning path)
    echo ""
    echo -e "${DIM}4a: MoE with thinking enabled${NC}"
    RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
      -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 7 * 8? Think step by step."}],"max_tokens":300,"temperature":0,"stream":false,"enable_thinking":true}')
    CONTENT=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content",""))' 2>/dev/null || echo "")
    assert_not_empty "MoE thinking returns content" "$CONTENT"
    assert_server_alive "MoE thinking"

    # 4b: MoE with tools
    echo ""
    echo -e "${DIM}4b: MoE with tools${NC}"
    RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
      -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],\"tools\":$TOOLS_JSON,\"max_tokens\":200,\"temperature\":0,\"stream\":false}")
    assert_contains "MoE tool response has choices" '"choices"' "$RESP"
    assert_server_alive "MoE tools"

    # 4c: Sequential MoE requests (BitsCache with expert routing stability)
    echo ""
    echo -e "${DIM}4c: Sequential MoE requests${NC}"
    for i in 1 2 3; do
        RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
          -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to $i in French.\"}],\"max_tokens\":20,\"temperature\":0,\"stream\":false}")
        CONTENT=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo "")
        if [ -z "$CONTENT" ]; then
            run_test "MoE sequential request $i" "FAIL" "empty response"
            break
        fi
    done
    run_test "MoE sequential requests stable" "$([ -n "$CONTENT" ] && echo PASS || echo FAIL)"
    assert_server_alive "MoE sequential"
else
    run_test "MoE tests" "SKIP" "model has $NUM_EXPERTS experts (need >0)"
fi

# ═══════════════════════════════════════════════════════
# Section 5: Final Crash Scan
# ═══════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}Section 5: Final Crash Scan${NC}"
check_no_crash "full test run"

# ── Summary ──
echo ""
echo "======================================================="
echo -e "  ${GREEN}Passed: $PASS${NC}  ${RED}Failed: $FAIL${NC}  ${YELLOW}Skipped: $SKIP${NC}  Total: $TOTAL"
echo "======================================================="
echo "  Model: $MODEL_TYPE | Vision: $HAS_VISION | Experts: $NUM_EXPERTS"
echo "  Log: $LOG"
echo "======================================================="

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
