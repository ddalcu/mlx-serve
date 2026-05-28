#!/bin/bash
# test_tokenize_cache.sh — Iteration 2 of overnight perf push.
#
# Phase 4 #3 plan: pre-tokenize stable prompt content so warm-cache requests
# don't pay the render+tokenize bill twice. The 8-hour bench surfaced
# tokenize_ms = 236ms on a 1813-token Gemma 4 prompt — 7× the actual
# Metal prefill time on a KV-cache hit. A per-LoadedModel tokenize cache
# makes the second hit on an identical prompt drop tokenize_ms to ≤ 1ms.
#
# This is the regression test. Reverting the cache in src/chat.zig (or the
# handler plumbing in src/server.zig) MUST flip this test red.
#
# Env:
#   MODEL    Any MLX model. Default: Gemma 4 E4B 4-bit under Sandisk.
#   PORT     Default 19105.
#   BINARY   Default ./zig-out/bin/mlx-serve.

set -uo pipefail

PORT="${PORT:-19105}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
MODEL="${MODEL:-/Volumes/Sandisk_1TB/Models/mlx-community/gemma-4-e4b-it-4bit}"
BASE="http://127.0.0.1:$PORT"

[ -d "$MODEL" ] || { echo "SKIP: model dir missing: $MODEL"; exit 0; }
[ -x "$BIN" ]   || { echo "fail: build mlx-serve first ($BIN)"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }

pkill -9 -f "mlx-serve.*port $PORT" 2>/dev/null
sleep 1

LOG="$(mktemp)"
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    pkill -9 -f "mlx-serve.*port $PORT" 2>/dev/null
    rm -f "$LOG"
}
trap cleanup EXIT INT TERM

# Use --prefix-cache-entries 4 so the KV cache also engages and we can read
# cached_n to confirm both layers (tokenize cache + KV cache) are warm.
"$BIN" --model "$MODEL" --serve --port "$PORT" --ctx-size 8192 \
    --prefix-cache-entries 4 --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 180); do
    curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -20 "$LOG"; exit 1; }
    sleep 0.5
done

# A medium-length user prompt (~400 chars). Long enough that tokenize takes
# real time on cold; short enough to keep the test snappy.
PROMPT="Continue the story in exactly four sentences. The kingdom of Avalon was beset by trials. Each season brought new challenges to its people, but the king remained steadfast. The relentless winter finally broke, revealing a frozen wasteland where the people had barely survived. With spring arriving, the king ordered the village elders to gather supplies and plan the rebuilding efforts together."

BODY=$(jq -nc --arg p "$PROMPT" \
    '{messages:[{role:"user",content:$p}],max_tokens:1,temperature:0,stream:false}')

# Cold turn
COLD="$(curl -sf --max-time 90 -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$BODY")"
COLD_TOK=$(echo "$COLD" | jq -r '.timings.tokenize_ms')
COLD_PROMPT_MS=$(echo "$COLD" | jq -r '.timings.prompt_ms')

# Warm turn — identical body, so both tokenize cache + KV cache should hit.
WARM="$(curl -sf --max-time 90 -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$BODY")"
WARM_TOK=$(echo "$WARM" | jq -r '.timings.tokenize_ms')
WARM_PROMPT_MS=$(echo "$WARM" | jq -r '.timings.prompt_ms')
WARM_CACHED=$(echo "$WARM" | jq -r '.timings.cached_n')

echo "--- cold ---"
echo "  tokenize_ms=$COLD_TOK  prompt_ms=$COLD_PROMPT_MS"
echo "--- warm ---"
echo "  tokenize_ms=$WARM_TOK  prompt_ms=$WARM_PROMPT_MS  cached_n=$WARM_CACHED"

EC=0

# Assertion 1: warm tokenize_ms is well under cold tokenize_ms (cache HIT).
# Threshold: warm < 25% of cold (cold has actual tokenize work; warm should
# be near zero — even a dup of a 200-tok slice). Hardware-agnostic.
if ! python3 -c "import sys; cold=float(sys.argv[1]); warm=float(sys.argv[2]); assert cold > 0.5 and warm < cold * 0.25, f'warm {warm} not < 25% of cold {cold}'" \
    "$COLD_TOK" "$WARM_TOK"; then
    echo "❌ tokenize cache did not engage: warm $WARM_TOK ms vs cold $COLD_TOK ms"
    EC=1
else
    pct=$(python3 -c "print(f'{(float(\"$WARM_TOK\")/float(\"$COLD_TOK\")):.1%}')")
    echo "✅ warm tokenize_ms $WARM_TOK ms is $pct of cold $COLD_TOK ms"
fi

# Assertion 2: KV cache engaged too (orthogonal check; if this fails, the
# whole warm path is suspect, not just the tokenize cache).
if [ "$WARM_CACHED" = "0" ] || [ -z "$WARM_CACHED" ]; then
    echo "❌ KV cache did not engage on warm turn (cached_n=$WARM_CACHED)"
    EC=1
else
    echo "✅ KV cache engaged (cached_n=$WARM_CACHED)"
fi

cleanup

if [ "$EC" -ne 0 ]; then
    echo
    echo "FAIL: tokenize cache regression."
fi
exit "$EC"
