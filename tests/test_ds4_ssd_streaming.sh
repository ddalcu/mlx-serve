#!/bin/bash
# Integration test for ds4 SSD weight-streaming (issue #39).
#
# DeepSeek-V4-Flash is larger than RAM on many machines; without streaming the
# ds4 engine maps + warms the full model and OOMs ("metal failed to map model
# views; aborting startup"). `--ssd-streaming` makes ds4 skip full residency and
# stream expert weights from SSD with an in-RAM cache, so it loads + serves.
#
# This pins: with --ssd-streaming the engine reports streaming mode, loads, and
# generates coherent text over the OpenAI API.
#
# Usage: DS4_GGUF_MODEL=/path/to/DeepSeek-V4-Flash-*.gguf ./tests/test_ds4_ssd_streaming.sh [port]
#   Needs a DeepSeek-V4-Flash GGUF + a machine where the ds4 engine runs (Metal).

set -u

MODEL="${DS4_GGUF_MODEL:-$HOME/.mlx-serve/models/antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
PORT="${1:-11298}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
LOG=/tmp/test_ds4_ssd_streaming.log
PASS=0
FAIL=0

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"; fi
}

if [ ! -f "$MODEL" ]; then
    echo "SKIP: ds4 GGUF not found: $MODEL (set DS4_GGUF_MODEL)"
    exit 0
fi
if [ ! -x "$BINARY" ]; then
    echo "SKIP: binary not found: $BINARY (build: zig build -Doptimize=ReleaseFast)"
    exit 0
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

echo ""
echo "── ds4 --ssd-streaming: load + generate (issue #39) ──"
# Force the ds4 engine and enable SSD streaming.
"$BINARY" --model "$MODEL" --serve --port "$PORT" --engine ds4 --ssd-streaming \
    --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT

# ds4 streaming skips warmup so startup is fast, but allow generous time for the
# initial metal map on a big model.
ok=0
for _ in $(seq 1 240); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break  # server died
    sleep 1
done
check "server became healthy with --ssd-streaming" "$ok"

# ds4 prints "SSD streaming mode enabled" (residency + warmup skipped) when the
# flag reaches the engine. This is the proof the flag is wired through.
check "ds4 reports SSD streaming mode enabled" \
    "$(grep -qiE "SSD streaming.*enabled|streaming mode enabled|residency.*skipped" "$LOG" && echo 1 || echo 0)"
check "no OOM / metal-map abort in startup" \
    "$(grep -qiE "failed to map model views|OutOfMemory|Insufficient Memory|EngineOpenFailed" "$LOG" && echo 0 || echo 1)"

if [ "$ok" = "1" ]; then
    RESP=$(curl -s -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"mlx-serve","messages":[{"role":"user","content":"In one short sentence, what is the capital of France?"}],"max_tokens":40,"temperature":0}')
    check "chat completion returned choices" \
        "$(echo "$RESP" | grep -q '"choices"' && echo 1 || echo 0)"
    # Coherence canary: the answer should mention Paris.
    check "response is coherent (mentions Paris)" \
        "$(echo "$RESP" | grep -qi "paris" && echo 1 || echo 0)"
fi

kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true

echo ""
echo "── ds4 --ssd-streaming: headless on-demand load (the app's path) ──"
ROOTDIR=$(mktemp -d); PACK=$(basename "$(dirname "$MODEL")")
ln -s "$(dirname "$MODEL")" "$ROOTDIR/$PACK"
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true; rm -rf "$ROOTDIR"' EXIT
# The default resident cap (the Metal working set) applies: a bill of the whole file would refuse this load.
"$BINARY" --serve --port "$PORT" --model-dir "$ROOTDIR" --ssd-streaming --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
ok=0
for _ in $(seq 1 60); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && { ok=1; break; }
    sleep 1
done
check "headless server became healthy" "$ok"
if [ "$ok" = "1" ]; then
    RESP=$(curl -s -m 900 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$PACK\",\"messages\":[{\"role\":\"user\",\"content\":\"In one short sentence, what is the capital of France?\"}],\"max_tokens\":40,\"temperature\":0}")
    check "on-demand load reaches ds4 with SSD streaming" \
        "$(grep -qiE "SSD streaming mode enabled" "$LOG" && echo 1 || echo 0)"
    check "on-demand load returned choices" \
        "$(echo "$RESP" | grep -q '"choices"' && echo 1 || echo 0)"
    check "the streaming bill fits the resident cap" \
        "$(grep -q "Refusing to load" "$LOG" && echo 0 || echo 1)"
fi

echo ""
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}PASS${NC} $TOTAL/$TOTAL tests passed"
    exit 0
else
    echo -e "${RED}FAIL${NC} $FAIL/$TOTAL tests failed"
    echo "--- server log (last 25 lines) ---"; tail -25 "$LOG" 2>/dev/null || true
    exit 1
fi
